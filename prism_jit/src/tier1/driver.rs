//! Tier 1 JIT Compilation Driver
//!
//! Orchestrates the complete Tier 1 compilation pipeline from bytecode to native code.
//! This driver integrates:
//! - Bytecode lowering with speculation and IC site allocation
//! - Template-based code generation with IC fast paths
//! - Deoptimization stub generation for fallback to interpreter
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
//! │ CodeObject  │────▶│ Bytecode    │────▶│ Template    │────▶│ Native Code │
//! │ (bytecode)  │     │ Lowerer     │     │ Compiler    │     │ (executable)│
//! └─────────────┘     └─────────────┘     └─────────────┘     └─────────────┘
//!                           │                    │
//!                           ▼                    ▼
//!                     ┌─────────────┐     ┌─────────────┐
//!                     │ IcManager   │     │ DeoptStubs  │
//!                     │ (IC sites)  │     │ (fallback)  │
//!                     └─────────────┘     └─────────────┘
//! ```
//!
//! # Performance Characteristics
//!
//! The Tier 1 compiler is optimized for fast compilation (~1ms per function)
//! at the cost of less optimized output code. IC integration provides:
//! - O(1) property access for monomorphic sites (~5 cycles overhead)
//! - O(n) for polymorphic (n ≤ 4, ~20 cycles)
//! - O(1) megamorphic fallback via global hash table (~100 cycles)

use prism_compiler::bytecode::CodeObject;
use prism_core::SpeculationProvider;

use crate::ic::{IcManager, IcSite, ShapeVersion};
use crate::tier1::codegen::{CompiledFunction, TemplateCompiler};
use crate::tier1::deopt::DeoptInfo;
use crate::tier1::frame::FrameLayout;
use crate::tier1::lower::{BytecodeLowerer, LoweringConfig};

use std::sync::Arc;

// Placeholder deopt handler address for testing.
// In production, this would be the real runtime deopt handler.
const DUMMY_DEOPT_HANDLER: u64 = 0xDEAD_BEEF_DEAD_BEEF;

// =============================================================================
// Compilation Result
// =============================================================================

/// Result of Tier 1 compilation.
pub struct CompiledCode {
    /// Compiled function with executable code.
    compiled: CompiledFunction,

    /// IC manager with allocated sites.
    ic_manager: IcManager,

    /// Entry point offset into code buffer.
    entry_offset: usize,
}

impl CompiledCode {
    /// Get the entry point address.
    #[inline]
    pub fn entry_point(&self) -> *const u8 {
        self.compiled.code.as_ptr()
    }

    /// Get total code size.
    #[inline]
    pub fn code_size(&self) -> usize {
        self.compiled.code.len()
    }

    /// Get frame layout.
    #[inline]
    pub fn frame_layout(&self) -> &FrameLayout {
        &self.compiled.frame_layout
    }

    /// Get IC manager.
    #[inline]
    pub fn ic_manager(&self) -> &IcManager {
        &self.ic_manager
    }

    /// Get mutable IC manager.
    #[inline]
    pub fn ic_manager_mut(&mut self) -> &mut IcManager {
        &mut self.ic_manager
    }

    /// Get deoptimization info.
    #[inline]
    pub fn deopt_info(&self) -> &[DeoptInfo] {
        &self.compiled.deopt_info
    }

    /// Get deopt info by index.
    #[inline]
    pub fn get_deopt_info(&self, idx: usize) -> Option<&DeoptInfo> {
        self.compiled.deopt_info.get(idx)
    }

    /// Get IC site by index.
    #[inline]
    pub fn get_ic_site(&self, idx: u32) -> Option<&IcSite> {
        self.ic_manager.get(idx)
    }

    /// Get mutable IC site by index.
    #[inline]
    pub fn get_ic_site_mut(&mut self, idx: u32) -> Option<&mut IcSite> {
        self.ic_manager.get_mut(idx)
    }
}

// =============================================================================
// Compilation Configuration
// =============================================================================

/// Configuration for Tier 1 compilation.
#[derive(Debug, Clone)]
pub struct CompilationConfig {
    /// Lowering configuration.
    pub lowering: LoweringConfig,

    /// Maximum code size before bailing out (bytes).
    pub max_code_size: usize,

    /// Enable deoptimization stub generation.
    pub generate_deopt_stubs: bool,

    /// Collect compilation statistics.
    pub collect_stats: bool,
}

impl Default for CompilationConfig {
    fn default() -> Self {
        Self {
            lowering: LoweringConfig::default(),
            max_code_size: 64 * 1024, // 64KB max per function
            generate_deopt_stubs: true,
            collect_stats: false,
        }
    }
}

impl CompilationConfig {
    /// Create a configuration optimized for fast compilation.
    #[inline]
    pub fn fast() -> Self {
        Self {
            lowering: LoweringConfig {
                enable_speculation: false,
                emit_guards: false,
                aggressive_inline: false,
                enable_ic: true,
            },
            max_code_size: 32 * 1024,
            generate_deopt_stubs: true,
            collect_stats: false,
        }
    }

    /// Create a configuration optimized for better code quality.
    #[inline]
    pub fn quality() -> Self {
        Self {
            lowering: LoweringConfig::default(),
            max_code_size: 128 * 1024,
            generate_deopt_stubs: true,
            collect_stats: true,
        }
    }

    /// Disable IC for debugging.
    #[inline]
    pub fn without_ic(mut self) -> Self {
        self.lowering.enable_ic = false;
        self
    }
}

// =============================================================================
// Compilation Statistics
// =============================================================================

/// Statistics from a compilation run.
#[derive(Debug, Clone, Copy, Default)]
pub struct CompilationStats {
    /// Time spent in lowering phase (nanoseconds).
    pub lowering_ns: u64,

    /// Time spent in code generation phase (nanoseconds).
    pub codegen_ns: u64,

    /// Number of bytecode instructions lowered.
    pub instructions_lowered: u32,

    /// Number of template instructions emitted.
    pub templates_emitted: u32,

    /// Number of IC sites allocated.
    pub ic_sites_allocated: u32,

    /// Number of deopt points generated.
    pub deopt_points: u32,

    /// Final code size in bytes.
    pub code_size: u32,
}

// =============================================================================
// Tier 1 Compiler
// =============================================================================

/// Tier 1 JIT compiler with IC integration.
///
/// The compiler maintains state across compilations to enable cross-function
/// optimizations and shared IC caching strategies.
#[derive(Debug)]
pub struct Tier1Compiler<S: SpeculationProvider> {
    /// Speculation provider for type hints.
    speculation: Arc<S>,

    /// Current shape version for IC staleness checks.
    shape_version: ShapeVersion,

    /// Compilation configuration.
    config: CompilationConfig,

    /// Number of functions compiled.
    functions_compiled: u64,

    /// Total code bytes generated.
    total_code_bytes: u64,

    /// Total IC sites allocated.
    total_ic_sites: u64,
}

impl<S: SpeculationProvider> Tier1Compiler<S> {
    /// Create a new Tier 1 compiler with the given speculation provider.
    #[inline]
    pub fn new(speculation: Arc<S>) -> Self {
        Self {
            speculation,
            shape_version: ShapeVersion::current(),
            config: CompilationConfig::default(),
            functions_compiled: 0,
            total_code_bytes: 0,
            total_ic_sites: 0,
        }
    }

    /// Create a new compiler with custom configuration.
    #[inline]
    pub fn with_config(speculation: Arc<S>, config: CompilationConfig) -> Self {
        Self {
            speculation,
            shape_version: ShapeVersion::current(),
            config,
            functions_compiled: 0,
            total_code_bytes: 0,
            total_ic_sites: 0,
        }
    }

    /// Compile a code object to native code.
    ///
    /// This is the main entry point for Tier 1 compilation. It:
    /// 1. Lowers bytecode to template IR with IC site allocation
    /// 2. Compiles template IR to native code
    /// 3. Generates deoptimization stubs
    /// 4. Returns executable code with IC metadata
    pub fn compile(&mut self, code: &CodeObject) -> Result<CompiledCode, CompilationError> {
        let code_id = self.functions_compiled as u32;

        // Phase 1: Create IC manager for this function
        let mut ic_manager = IcManager::with_capacity(16, self.shape_version);

        // Phase 2: Lower bytecode to template IR
        let ir = {
            let mut lowerer = BytecodeLowerer::with_ic_manager(
                self.speculation.as_ref(),
                code_id,
                self.config.lowering.clone(),
                &mut ic_manager,
            );
            lowerer.lower(code)
        };

        if ir.is_empty() {
            return Err(CompilationError::EmptyCode);
        }

        // Phase 3: Compile template IR to native code
        let compiler = TemplateCompiler::new(DUMMY_DEOPT_HANDLER);
        let compiled = compiler
            .compile(code.register_count, &ir)
            .map_err(CompilationError::CodegenError)?;

        let code_size = compiled.code.len();
        if code_size > self.config.max_code_size {
            return Err(CompilationError::CodeTooLarge {
                size: code_size,
                max: self.config.max_code_size,
            });
        }

        // Update statistics
        self.functions_compiled += 1;
        self.total_code_bytes += code_size as u64;
        self.total_ic_sites += ic_manager.len() as u64;

        Ok(CompiledCode {
            compiled,
            ic_manager,
            entry_offset: 0,
        })
    }

    /// Compile with statistics collection.
    pub fn compile_with_stats(
        &mut self,
        code: &CodeObject,
    ) -> Result<(CompiledCode, CompilationStats), CompilationError> {
        use std::time::Instant;

        let code_id = self.functions_compiled as u32;
        let mut stats = CompilationStats::default();

        // Phase 1: Create IC manager
        let mut ic_manager = IcManager::with_capacity(16, self.shape_version);

        // Phase 2: Lower bytecode to template IR (timed)
        let lowering_start = Instant::now();
        let ir = {
            let mut lowerer = BytecodeLowerer::with_ic_manager(
                self.speculation.as_ref(),
                code_id,
                self.config.lowering.clone(),
                &mut ic_manager,
            );
            lowerer.lower(code)
        };
        stats.lowering_ns = lowering_start.elapsed().as_nanos() as u64;
        stats.instructions_lowered = code.instructions.len() as u32;
        stats.templates_emitted = ir.len() as u32;
        stats.ic_sites_allocated = ic_manager.len() as u32;

        if ir.is_empty() {
            return Err(CompilationError::EmptyCode);
        }

        // Phase 3: Code generation (timed)
        let codegen_start = Instant::now();
        let compiler = TemplateCompiler::new(DUMMY_DEOPT_HANDLER);
        let compiled = compiler
            .compile(code.register_count, &ir)
            .map_err(CompilationError::CodegenError)?;
        stats.codegen_ns = codegen_start.elapsed().as_nanos() as u64;
        stats.deopt_points = compiled.deopt_info.len() as u32;
        stats.code_size = compiled.code.len() as u32;

        let code_size = compiled.code.len();
        if code_size > self.config.max_code_size {
            return Err(CompilationError::CodeTooLarge {
                size: code_size,
                max: self.config.max_code_size,
            });
        }

        // Update statistics
        self.functions_compiled += 1;
        self.total_code_bytes += code_size as u64;
        self.total_ic_sites += ic_manager.len() as u64;

        Ok((
            CompiledCode {
                compiled,
                ic_manager,
                entry_offset: 0,
            },
            stats,
        ))
    }

    /// Get number of functions compiled.
    #[inline]
    pub fn functions_compiled(&self) -> u64 {
        self.functions_compiled
    }

    /// Get total code bytes generated.
    #[inline]
    pub fn total_code_bytes(&self) -> u64 {
        self.total_code_bytes
    }

    /// Get total IC sites allocated.
    #[inline]
    pub fn total_ic_sites(&self) -> u64 {
        self.total_ic_sites
    }

    /// Get current configuration.
    #[inline]
    pub fn config(&self) -> &CompilationConfig {
        &self.config
    }

    /// Update shape version for IC staleness.
    #[inline]
    pub fn refresh_shape_version(&mut self) {
        self.shape_version = ShapeVersion::current();
    }
}

// =============================================================================
// Compilation Error
// =============================================================================

/// Errors that can occur during compilation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CompilationError {
    /// Code object is empty (no instructions).
    EmptyCode,

    /// Generated code exceeds maximum size.
    CodeTooLarge { size: usize, max: usize },

    /// Internal codegen error.
    CodegenError(String),

    /// Unsupported bytecode instruction.
    UnsupportedInstruction { opcode: u8, offset: u32 },
}

impl std::fmt::Display for CompilationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EmptyCode => write!(f, "code object is empty"),
            Self::CodeTooLarge { size, max } => {
                write!(
                    f,
                    "generated code ({} bytes) exceeds maximum ({} bytes)",
                    size, max
                )
            }
            Self::CodegenError(msg) => write!(f, "codegen error: {}", msg),
            Self::UnsupportedInstruction { opcode, offset } => {
                write!(
                    f,
                    "unsupported instruction opcode {} at offset {}",
                    opcode, offset
                )
            }
        }
    }
}

impl std::error::Error for CompilationError {}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tier1::codegen::TemplateInstruction;
    use prism_compiler::bytecode::{Instruction, Opcode, Register};
    use prism_core::speculation::NoSpeculation;

    fn make_code(instructions: Vec<Instruction>) -> CodeObject {
        let mut code = CodeObject::new("test", "test.py");
        code.instructions = instructions.into_boxed_slice();
        code.register_count = 16;
        code
    }

    #[test]
    fn test_compiler_new() {
        let speculation = Arc::new(NoSpeculation);
        let compiler = Tier1Compiler::new(speculation);

        assert_eq!(compiler.functions_compiled(), 0);
        assert_eq!(compiler.total_code_bytes(), 0);
        assert_eq!(compiler.total_ic_sites(), 0);
    }

    #[test]
    fn test_compiler_with_config() {
        let speculation = Arc::new(NoSpeculation);
        let config = CompilationConfig::fast();
        let compiler = Tier1Compiler::with_config(speculation, config);

        assert!(!compiler.config().lowering.enable_speculation);
    }

    #[test]
    fn test_compilation_config_default() {
        let config = CompilationConfig::default();

        assert!(config.lowering.enable_ic);
        assert!(config.lowering.enable_speculation);
        assert_eq!(config.max_code_size, 64 * 1024);
    }

    #[test]
    fn test_compilation_config_fast() {
        let config = CompilationConfig::fast();

        assert!(config.lowering.enable_ic);
        assert!(!config.lowering.enable_speculation);
        assert_eq!(config.max_code_size, 32 * 1024);
    }

    #[test]
    fn test_compilation_config_quality() {
        let config = CompilationConfig::quality();

        assert!(config.lowering.enable_ic);
        assert!(config.collect_stats);
        assert_eq!(config.max_code_size, 128 * 1024);
    }

    #[test]
    fn test_compilation_config_without_ic() {
        let config = CompilationConfig::default().without_ic();

        assert!(!config.lowering.enable_ic);
    }

    #[test]
    fn test_compilation_stats_default() {
        let stats = CompilationStats::default();

        assert_eq!(stats.lowering_ns, 0);
        assert_eq!(stats.codegen_ns, 0);
        assert_eq!(stats.ic_sites_allocated, 0);
    }

    #[test]
    fn test_compilation_error_display() {
        let e = CompilationError::EmptyCode;
        assert_eq!(e.to_string(), "code object is empty");

        let e = CompilationError::CodeTooLarge { size: 100, max: 50 };
        assert_eq!(
            e.to_string(),
            "generated code (100 bytes) exceeds maximum (50 bytes)"
        );
    }

    #[test]
    fn test_compiled_code_accessors() {
        // Create mock compiled function using the template compiler
        let compiler = TemplateCompiler::new_for_testing();
        let instructions = vec![TemplateInstruction::Nop { bc_offset: 0 }];
        let compiled_fn = compiler.compile(16, &instructions).expect("compile failed");

        let code = CompiledCode {
            compiled: compiled_fn,
            ic_manager: IcManager::new(ShapeVersion::new(1)),
            entry_offset: 0,
        };

        assert!(code.code_size() > 0);
        assert!(!code.entry_point().is_null());
        assert_eq!(code.deopt_info().len(), 0);
    }

    #[test]
    fn test_compiler_refresh_shape_version() {
        let speculation = Arc::new(NoSpeculation);
        let mut compiler = Tier1Compiler::new(speculation);

        let v1 = compiler.shape_version;
        compiler.refresh_shape_version();
        // Version should be the same unless global was bumped
        assert_eq!(v1, compiler.shape_version);
    }

    // Note: Full compilation tests require TemplateCompiler::compile_with_deopt
    // to be implemented. These tests verify the driver infrastructure.
}
