//! JIT Bridge - Coordinates compilation and tier-up decisions.
//!
//! The JitBridge is the central orchestrator for JIT compilation,
//! connecting the profiler, compiler, code cache, and executor.
//!
//! # Responsibilities
//!
//! - Monitor function hotness via profiler
//! - Trigger compilation when thresholds are crossed
//! - Manage synchronous and asynchronous compilation
//! - Coordinate code invalidation on deoptimization
//!
//! # Performance
//!
//! - Lock-free hot path for compiled code lookup
//! - Background compilation to avoid blocking interpreter
//! - Batched compilation requests to reduce overhead

use std::sync::Arc;

use prism_compiler::bytecode::CodeObject;
use prism_jit::ir::{Graph, GraphBuilder};
use prism_jit::ir::builder::translator::BytecodeTranslator;
use prism_jit::ir::operators::{ControlOp, Operator};
use prism_jit::opt::OptPipeline;
use prism_jit::regalloc::{AllocatorConfig, LinearScanAllocator, LivenessAnalysis};
use prism_jit::runtime::{CodeCache, CompiledEntry, ReturnAbi, RuntimeConfig};
use prism_jit::tier1::codegen::TemplateCompiler;
use prism_jit::tier2::{CodeEmitter, InstructionSelector};

use crate::compilation_queue::CompilationQueue;
use crate::jit_executor::{DeoptReason, ExecutionResult, JitExecutor};
use crate::profiler::{CodeId, Profiler, TierUpDecision};
use crate::tier1_lowering::lower_code_to_templates;

// =============================================================================
// Bridge Configuration
// =============================================================================

/// Configuration for the JIT bridge.
#[derive(Debug, Clone)]
pub struct BridgeConfig {
    /// Enable JIT compilation.
    pub enabled: bool,
    /// Tier 1 (template) compilation threshold.
    pub tier1_threshold: u64,
    /// Tier 2 (optimizing) compilation threshold.
    pub tier2_threshold: u64,
    /// Enable background compilation.
    pub background_compilation: bool,
    /// Enable OSR (on-stack replacement).
    pub enable_osr: bool,
    /// Maximum compilation queue size.
    pub max_queue_size: usize,
    /// Maximum compiled code cache size in bytes.
    pub max_code_size: usize,
}

impl Default for BridgeConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            tier1_threshold: 1_000,
            tier2_threshold: 10_000,
            background_compilation: true,
            enable_osr: true,
            max_queue_size: 64,
            max_code_size: 64 * 1024 * 1024,
        }
    }
}

impl BridgeConfig {
    /// Create configuration for testing.
    pub fn for_testing() -> Self {
        Self {
            enabled: true,
            tier1_threshold: 10,
            tier2_threshold: 100,
            background_compilation: false,
            enable_osr: false,
            max_queue_size: 4,
            max_code_size: 1024 * 1024,
        }
    }

    /// Create disabled configuration.
    pub fn disabled() -> Self {
        Self {
            enabled: false,
            ..Default::default()
        }
    }
}

// =============================================================================
// Compilation State
// =============================================================================

/// State of a function's compilation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompilationState {
    /// Not compiled, running in interpreter.
    Interpreted,
    /// Compilation queued but not started.
    Queued,
    /// Currently being compiled.
    Compiling,
    /// Compiled at Tier 1 (template).
    CompiledTier1,
    /// Compiled at Tier 2 (optimizing).
    CompiledTier2,
    /// Compilation failed, stay in interpreter.
    Failed,
    /// Invalidated, needs recompilation.
    Invalidated,
}

// =============================================================================
// JIT Bridge
// =============================================================================

/// The JIT bridge coordinating compilation and execution.
pub struct JitBridge {
    /// Configuration.
    config: BridgeConfig,
    /// JIT executor for running compiled code.
    executor: JitExecutor,
    /// Template compiler for Tier 1.
    tier1_compiler: TemplateCompiler,
    /// Code cache (shared with executor and compilation queue).
    code_cache: Arc<CodeCache>,
    /// Background compilation queue (lazy-initialized when needed).
    compilation_queue: Option<CompilationQueue>,
}

impl JitBridge {
    /// Create a new JIT bridge.
    pub fn new(config: BridgeConfig) -> Self {
        let runtime_config = RuntimeConfig {
            max_code_size: config.max_code_size,
            compiler_threads: if config.background_compilation { 1 } else { 0 },
            tier_up_threshold: config.tier1_threshold as u32,
            enable_osr: config.enable_osr,
            enable_speculation: true,
        };

        let code_cache = Arc::new(CodeCache::new(runtime_config.max_code_size));
        let executor = JitExecutor::new(Arc::clone(&code_cache));
        let tier1_compiler = TemplateCompiler::new_runtime();

        // Create compilation queue if background compilation is enabled
        let compilation_queue = if config.background_compilation {
            Some(CompilationQueue::new(
                Arc::clone(&code_cache),
                config.max_queue_size,
            ))
        } else {
            None
        };

        Self {
            config,
            executor,
            tier1_compiler,
            code_cache,
            compilation_queue,
        }
    }

    /// Create bridge with default configuration.
    pub fn with_defaults() -> Self {
        Self::new(BridgeConfig::default())
    }

    /// Check if JIT is enabled.
    #[inline]
    pub fn is_enabled(&self) -> bool {
        self.config.enabled
    }

    // =========================================================================
    // Compiled Code Lookup
    // =========================================================================

    /// Look up compiled code for a function.
    ///
    /// This is the hot path - must be as fast as possible.
    #[inline]
    pub fn lookup(&self, code_id: u64) -> Option<Arc<CompiledEntry>> {
        if !self.config.enabled {
            return None;
        }
        self.code_cache.lookup(code_id)
    }

    /// Check if a function has compiled code.
    #[inline]
    pub fn is_compiled(&self, code_id: u64) -> bool {
        self.lookup(code_id).is_some()
    }

    // =========================================================================
    // Tier-Up Checking
    // =========================================================================

    /// Check if a function should tier up based on profiler data.
    ///
    /// Returns the tier-up decision without triggering compilation.
    #[inline]
    pub fn check_tier_up(&self, profiler: &Profiler, code_id: CodeId) -> TierUpDecision {
        if !self.config.enabled {
            return TierUpDecision::None;
        }

        let count = profiler.call_count(code_id);

        if count >= self.config.tier2_threshold {
            // Check if already at Tier 2
            if let Some(entry) = self.code_cache.lookup(code_id.0 as u64) {
                if entry.tier() >= 2 {
                    return TierUpDecision::None;
                }
            }
            return TierUpDecision::Tier2;
        }

        if count >= self.config.tier1_threshold {
            // Check if already compiled
            if self.code_cache.lookup(code_id.0 as u64).is_some() {
                return TierUpDecision::None;
            }
            return TierUpDecision::Tier1;
        }

        TierUpDecision::None
    }

    /// Check if a loop should trigger OSR.
    #[inline]
    pub fn check_osr(&self, profiler: &Profiler, code_id: CodeId, loop_offset: u32) -> bool {
        if !self.config.enabled || !self.config.enable_osr {
            return false;
        }

        let count = profiler.loop_count(code_id, loop_offset);
        count >= self.config.tier1_threshold
    }

    // =========================================================================
    // Compilation
    // =========================================================================

    /// Compile a function synchronously.
    ///
    /// This blocks until compilation is complete. Use for testing
    /// or when background compilation is disabled.
    pub fn compile_sync(&mut self, code: &Arc<CodeObject>) -> Result<Arc<CompiledEntry>, String> {
        let code_id = code_id_from_arc(code);

        // Check if already compiled
        if let Some(entry) = self.code_cache.lookup(code_id) {
            return Ok(entry);
        }

        let entry = compile_tier1_entry(code, &mut self.tier1_compiler)?;

        // Insert into cache (this wraps in Arc internally)
        self.code_cache.insert(entry);

        // Lookup and return the Arc-wrapped entry
        self.code_cache
            .lookup(code_id)
            .ok_or_else(|| "Failed to lookup after insertion".to_string())
    }

    /// Compile a function with Tier 2 optimizations synchronously.
    ///
    /// This uses the full optimizing pipeline:
    /// 1. Bytecode → Sea-of-Nodes IR Graph
    /// 2. Optimization passes (DCE, GVN, LICM, Inlining)
    /// 3. Liveness analysis
    /// 4. Register allocation (Linear Scan)
    /// 5. Instruction selection
    /// 6. Code emission
    pub fn compile_tier2(&mut self, code: &Arc<CodeObject>) -> Result<Arc<CompiledEntry>, String> {
        let code_id = code_id_from_arc(code);

        // Check if already compiled at Tier 2
        if let Some(entry) = self.code_cache.lookup(code_id) {
            if entry.tier() >= 2 {
                return Ok(entry);
            }
        }

        let entry = compile_tier2_entry(code)?;

        // Insert into cache
        self.code_cache.insert(entry);

        // Lookup and return the Arc-wrapped entry
        self.code_cache
            .lookup(code_id)
            .ok_or_else(|| "Failed to lookup after Tier 2 insertion".to_string())
    }

    /// Request asynchronous compilation.
    ///
    /// Enqueues the code object for background compilation. Returns immediately.
    /// The compiled code will appear in the shared code cache once the worker
    /// thread finishes compilation.
    pub fn compile_async(&self, code: Arc<CodeObject>, tier: u8) {
        if let Some(ref queue) = self.compilation_queue {
            queue.enqueue(code, tier);
        }
        // If no queue (background_compilation=false), silently drop.
        // Caller should have checked config before calling this path.
    }

    // =========================================================================
    // Execution
    // =========================================================================

    /// Execute compiled code for a function.
    ///
    /// # Arguments
    ///
    /// * `entry` - The compiled code entry
    /// * `frame` - The interpreter frame
    ///
    /// # Returns
    ///
    /// Execution result (return value, deopt info, or exception).
    pub fn execute(
        &mut self,
        entry: &CompiledEntry,
        frame: &mut crate::frame::Frame,
    ) -> ExecutionResult {
        self.executor.execute(entry, frame)
    }

    /// Execute at an OSR entry point.
    pub fn execute_osr(
        &mut self,
        entry: &CompiledEntry,
        frame: &mut crate::frame::Frame,
        osr_bc_offset: u32,
    ) -> ExecutionResult {
        self.executor.execute_osr(entry, frame, osr_bc_offset)
    }

    // =========================================================================
    // Invalidation
    // =========================================================================

    /// Invalidate compiled code for a function.
    ///
    /// Called when assumptions made during compilation are violated
    /// (e.g., class shape changed, global redefined).
    pub fn invalidate(&self, code_id: u64) {
        self.code_cache.remove(code_id);
    }

    /// Invalidate all compiled code.
    pub fn invalidate_all(&self) {
        self.code_cache.clear();
    }

    /// Handle deoptimization - may invalidate code.
    pub fn handle_deopt(&self, code_id: u64, reason: DeoptReason) {
        // For certain deopt reasons, invalidate the code
        match reason {
            DeoptReason::TypeGuard | DeoptReason::CacheMiss => {
                // Might want to recompile with different assumptions
                // For now, just invalidate
                self.invalidate(code_id);
            }
            DeoptReason::UncommonTrap => {
                // Definitely invalidate - we hit a path we didn't expect
                self.invalidate(code_id);
            }
            _ => {
                // Other reasons don't require invalidation
            }
        }
    }

    // =========================================================================
    // Stats
    // =========================================================================

    /// Get the number of compiled functions.
    pub fn compiled_count(&self) -> usize {
        self.code_cache.len()
    }

    /// Get total compiled code size.
    pub fn compiled_size(&self) -> usize {
        self.code_cache.total_size()
    }

    /// Get executor reference.
    pub fn executor(&self) -> &JitExecutor {
        &self.executor
    }

    /// Get executor mutable reference.
    pub fn executor_mut(&mut self) -> &mut JitExecutor {
        &mut self.executor
    }
}

// =============================================================================
// Helper Functions
// =============================================================================

/// Get code ID from Arc<CodeObject>.
#[inline]
fn code_id_from_arc(code: &Arc<CodeObject>) -> u64 {
    Arc::as_ptr(code) as u64
}

/// Compile a code object into a Tier 1 cache entry.
pub(crate) fn compile_tier1_entry(
    code: &Arc<CodeObject>,
    compiler: &mut TemplateCompiler,
) -> Result<CompiledEntry, String> {
    let code_id = code_id_from_arc(code);
    let instructions = lower_code_to_templates(code)?;
    let compiled = compiler.compile(code.register_count, &instructions)?;
    Ok(CompiledEntry::from_executable_buffer(code_id, compiled.code).with_tier(1))
}

/// Compile a code object into a Tier 2 cache entry.
pub(crate) fn compile_tier2_entry(code: &Arc<CodeObject>) -> Result<CompiledEntry, String> {
    let code_id = code_id_from_arc(code);

    // Stage 1: Convert bytecode to Sea-of-Nodes IR
    let builder = GraphBuilder::new(code.register_count as usize, code.arg_count as usize);
    let translator = BytecodeTranslator::new(builder, code);
    let mut graph = translator
        .translate()
        .map_err(|e| format!("Tier 2 bytecode translation failed: {}", e))?;

    // Stage 2: Run optimization pipeline
    let mut pipeline = OptPipeline::new();
    let _stats = pipeline.run(&mut graph);

    // Stage 2.5: Reject graphs that Tier 2 lowering cannot currently handle.
    // This avoids silently generating incorrect machine code.
    validate_tier2_lowering_support(&graph)?;

    // Stage 3: Liveness analysis
    let liveness = LivenessAnalysis::analyze(&graph);
    let intervals = liveness.into_intervals();

    // Stage 4: Register allocation
    let allocator = LinearScanAllocator::new(AllocatorConfig::default());
    let (alloc_map, _alloc_stats) = allocator.allocate(intervals);

    // Stage 5: Instruction selection
    let mfunc = InstructionSelector::select(&graph, &alloc_map);

    // Stage 6: Code emission
    let compiled =
        CodeEmitter::emit(&mfunc).map_err(|e| format!("Tier 2 code emission failed: {}", e))?;

    Ok(
        CompiledEntry::from_executable_buffer(code_id, compiled.code)
            .with_tier(2)
            .with_return_abi(ReturnAbi::RawValueBits),
    )
}

fn validate_tier2_lowering_support(graph: &Graph) -> Result<(), String> {
    for (id, node) in graph.iter() {
        if node.is_dead() {
            continue;
        }

        let supported = match node.op {
            Operator::ConstInt(_)
            | Operator::ConstFloat(_)
            | Operator::ConstBool(_)
            | Operator::ConstNone
            | Operator::IntOp(_)
            | Operator::FloatOp(_)
            | Operator::IntCmp(_)
            | Operator::FloatCmp(_)
            | Operator::Bitwise(_)
            | Operator::Phi
            | Operator::LoopPhi => true,
            // Tier 2 currently does not materialize parameter nodes from JitFrameState.
            // Reject them explicitly to avoid generating undefined reads.
            Operator::Parameter(_) => false,
            Operator::Control(
                ControlOp::Start
                | ControlOp::End
                | ControlOp::Return
                | ControlOp::If
                | ControlOp::Region
                | ControlOp::Loop,
            ) => true,
            _ => false,
        };

        if !supported {
            return Err(format!(
                "Tier 2 lowering does not support operator {:?} at node {:?}",
                node.op, id
            ));
        }
    }

    Ok(())
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bridge_config_default() {
        let config = BridgeConfig::default();
        assert!(config.enabled);
        assert_eq!(config.tier1_threshold, 1_000);
        assert_eq!(config.tier2_threshold, 10_000);
        assert_eq!(config.max_code_size, 64 * 1024 * 1024);
    }

    #[test]
    fn test_bridge_config_testing() {
        let config = BridgeConfig::for_testing();
        assert!(config.enabled);
        assert_eq!(config.tier1_threshold, 10);
        assert!(!config.background_compilation);
        assert_eq!(config.max_code_size, 1024 * 1024);
    }

    #[test]
    fn test_bridge_creation() {
        let bridge = JitBridge::new(BridgeConfig::for_testing());
        assert!(bridge.is_enabled());
        assert_eq!(bridge.compiled_count(), 0);
    }

    #[test]
    fn test_bridge_disabled() {
        let bridge = JitBridge::new(BridgeConfig::disabled());
        assert!(!bridge.is_enabled());
        assert!(bridge.lookup(123).is_none());
    }

    #[test]
    fn test_compilation_state() {
        let state = CompilationState::Interpreted;
        assert_eq!(state, CompilationState::Interpreted);
    }

    #[test]
    fn test_compile_tier2_sets_tier_and_raw_value_abi() {
        use prism_compiler::bytecode::{Instruction, Opcode, Register};

        let mut bridge = JitBridge::new(BridgeConfig::for_testing());
        let mut code = CodeObject::new("tier2_test", "<test>");
        code.register_count = 1;
        code.instructions = vec![
            Instruction::op_d(Opcode::LoadNone, Register::new(0)),
            Instruction::op_d(Opcode::Return, Register::new(0)),
        ]
        .into_boxed_slice();
        let code = Arc::new(code);

        let entry = bridge.compile_tier2(&code).unwrap();
        assert_eq!(entry.tier(), 2);
        assert_eq!(entry.return_abi(), ReturnAbi::RawValueBits);
    }

    #[test]
    fn test_compile_tier2_rejects_unsupported_generic_arithmetic() {
        use prism_compiler::bytecode::{Instruction, Opcode, Register};
        use prism_core::Value;

        let mut bridge = JitBridge::new(BridgeConfig::for_testing());
        let mut code = CodeObject::new("tier2_unsupported", "<test>");
        code.register_count = 3;
        code.instructions = vec![
            Instruction::op_di(Opcode::LoadConst, Register::new(0), 0),
            Instruction::op_di(Opcode::LoadConst, Register::new(1), 1),
            Instruction::op_dss(Opcode::Add, Register::new(2), Register::new(0), Register::new(1)),
            Instruction::op_d(Opcode::Return, Register::new(2)),
        ]
        .into_boxed_slice();
        code.constants = vec![Value::int(10).unwrap(), Value::int(20).unwrap()].into_boxed_slice();
        let code = Arc::new(code);

        let err = bridge
            .compile_tier2(&code)
            .expect_err("unsupported generic arithmetic must not install tier2 code");
        assert!(err.contains("does not support operator"));
    }

    #[test]
    fn test_compile_tier2_rejects_parameter_nodes_until_materialized() {
        use prism_compiler::bytecode::{Instruction, Opcode, Register};

        let mut bridge = JitBridge::new(BridgeConfig::for_testing());
        let mut code = CodeObject::new("tier2_param", "<test>");
        code.register_count = 1;
        code.arg_count = 1;
        code.instructions = vec![Instruction::op_d(Opcode::Return, Register::new(0))].into_boxed_slice();
        let code = Arc::new(code);

        let err = bridge
            .compile_tier2(&code)
            .expect_err("parameterized code should fail until tier2 loads parameters from frame state");
        assert!(err.contains("Parameter"));
    }

    #[test]
    fn test_compile_tier2_rejects_uninitialized_register_reads() {
        use prism_compiler::bytecode::{Instruction, Opcode, Register};

        let mut bridge = JitBridge::new(BridgeConfig::for_testing());
        let mut code = CodeObject::new("tier2_uninit", "<test>");
        code.register_count = 3;
        code.instructions = vec![
            Instruction::op_dss(Opcode::Add, Register::new(2), Register::new(0), Register::new(1)),
            Instruction::op_d(Opcode::Return, Register::new(2)),
        ]
        .into_boxed_slice();
        let code = Arc::new(code);

        let err = bridge
            .compile_tier2(&code)
            .expect_err("uninitialized register reads should fail translation");
        assert!(err.contains("uninitialized register"));
    }
}
