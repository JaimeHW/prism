//! JIT Context - Encapsulates JIT state and provides dispatch interface.
//!
//! The JitContext is the integration layer between the VM and JIT compiler.
//! It provides:
//! - Zero-overhead check for JIT-disabled mode
//! - O(1) compiled code lookup
//! - Entry-point dispatch for compiled functions
//! - OSR dispatch at loop headers
//! - Tier-up triggering and compilation management
//! - Deoptimization handling
//!
//! # Performance
//!
//! The context is designed for minimal overhead on the hot path:
//! - Single Option check when JIT is disabled
//! - Lock-free code cache lookup
//! - Stack-allocated frame state (no heap allocation)
//! - Cold path separation for tier-up logic

use std::sync::Arc;

use prism_compiler::bytecode::CodeObject;
use prism_jit::runtime::{CodeCache, CompiledEntry};

use crate::error::RuntimeError;
use crate::frame::Frame;
use crate::jit_bridge::{BridgeConfig, JitBridge};
use crate::jit_executor::{DeoptReason, ExecutionResult, JitFrameState};
use crate::profiler::{CodeId, Profiler, TierUpDecision};

// =============================================================================
// JIT Configuration
// =============================================================================

/// Configuration for JIT compilation.
#[derive(Debug, Clone)]
pub struct JitConfig {
    /// Enable JIT compilation.
    pub enabled: bool,
    /// Enable background compilation (async).
    pub background_compilation: bool,
    /// Enable on-stack replacement.
    pub enable_osr: bool,
    /// Tier 1 (template) compilation threshold.
    pub tier1_threshold: u64,
    /// Tier 2 (optimizing) compilation threshold.
    pub tier2_threshold: u64,
    /// Maximum compiled code cache size in bytes.
    pub max_code_size: usize,
    /// Compile eagerly on first tier-up (for testing).
    pub eager_compilation: bool,
}

impl Default for JitConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            background_compilation: true,
            enable_osr: true,
            tier1_threshold: 1_000,
            tier2_threshold: 10_000,
            max_code_size: 64 * 1024 * 1024, // 64MB
            eager_compilation: false,
        }
    }
}

impl JitConfig {
    /// Create a disabled configuration.
    #[inline]
    pub fn disabled() -> Self {
        Self {
            enabled: false,
            ..Default::default()
        }
    }

    /// Create a configuration for testing (low thresholds, sync compilation).
    pub fn for_testing() -> Self {
        Self {
            enabled: true,
            background_compilation: false,
            enable_osr: false,
            tier1_threshold: 10,
            tier2_threshold: 100,
            max_code_size: 1024 * 1024,
            eager_compilation: true,
        }
    }

    /// Convert to bridge configuration.
    fn to_bridge_config(&self) -> BridgeConfig {
        BridgeConfig {
            enabled: self.enabled,
            tier1_threshold: self.tier1_threshold,
            tier2_threshold: self.tier2_threshold,
            background_compilation: self.background_compilation,
            enable_osr: self.enable_osr,
            max_queue_size: 64,
        }
    }
}

// =============================================================================
// JIT Statistics
// =============================================================================

/// Statistics for JIT execution.
#[derive(Debug, Default, Clone)]
pub struct JitStats {
    /// Number of JIT cache hits (executed compiled code).
    pub cache_hits: u64,
    /// Number of JIT cache misses (fell back to interpreter).
    pub cache_misses: u64,
    /// Number of compilations triggered.
    pub compilations_triggered: u64,
    /// Number of successful compilations.
    pub compilations_completed: u64,
    /// Number of failed compilations.
    pub compilations_failed: u64,
    /// Number of deoptimizations.
    pub deopts: u64,
    /// Number of OSR entries.
    pub osr_entries: u64,
    /// Number of OSR exits.
    pub osr_exits: u64,
    /// Total bytes of compiled code generated.
    pub compiled_bytes: u64,
}

impl JitStats {
    /// Calculate cache hit rate.
    #[inline]
    pub fn hit_rate(&self) -> f64 {
        let total = self.cache_hits + self.cache_misses;
        if total == 0 {
            0.0
        } else {
            self.cache_hits as f64 / total as f64
        }
    }

    /// Calculate deopt rate.
    #[inline]
    pub fn deopt_rate(&self) -> f64 {
        let total = self.cache_hits;
        if total == 0 {
            0.0
        } else {
            self.deopts as f64 / total as f64
        }
    }
}

// =============================================================================
// JIT Context
// =============================================================================

/// JIT execution context encapsulating all JIT state.
///
/// This is the primary interface between the VM and JIT subsystem.
/// It can be `None` in the VM when JIT is disabled, providing
/// zero-overhead for interpreter-only mode.
pub struct JitContext {
    /// JIT bridge for compilation coordination.
    bridge: JitBridge,
    /// Code cache for compiled code lookup (shared with bridge).
    #[allow(dead_code)]
    code_cache: Arc<CodeCache>,
    /// Configuration.
    config: JitConfig,
    /// Execution statistics.
    stats: JitStats,
    /// Reusable frame state (avoid allocation on hot path).
    #[allow(dead_code)]
    frame_state: JitFrameState,
}

impl JitContext {
    /// Create a new JIT context with the given configuration.
    pub fn new(config: JitConfig) -> Self {
        let code_cache = Arc::new(CodeCache::new(config.max_code_size));
        let bridge = JitBridge::new(config.to_bridge_config());

        Self {
            bridge,
            code_cache,
            config,
            stats: JitStats::default(),
            frame_state: JitFrameState::default(),
        }
    }

    /// Create with default configuration.
    pub fn with_defaults() -> Self {
        Self::new(JitConfig::default())
    }

    /// Create for testing (low thresholds, synchronous compilation).
    pub fn for_testing() -> Self {
        Self::new(JitConfig::for_testing())
    }

    /// Check if JIT is enabled.
    #[inline(always)]
    pub fn is_enabled(&self) -> bool {
        self.config.enabled
    }

    // =========================================================================
    // Entry Point Dispatch (Hot Path)
    // =========================================================================

    /// Look up compiled code for a function.
    ///
    /// This is the hot path - optimized for speed.
    /// Returns None if no compiled code exists.
    #[inline]
    pub fn lookup(&self, code_id: u64) -> Option<Arc<CompiledEntry>> {
        self.bridge.lookup(code_id)
    }

    /// Try to execute compiled code for a function.
    ///
    /// Returns `Some(result)` if compiled code was executed,
    /// `None` if no compiled code exists (fall back to interpreter).
    ///
    /// # Arguments
    ///
    /// * `code_id` - Unique identifier for the code object
    /// * `frame` - The interpreter frame to execute in
    ///
    /// # Performance
    ///
    /// This is on the critical path for JIT dispatch.
    /// The lookup is O(1) via hash map. Frame state setup
    /// is stack-allocated (no heap allocation).
    #[inline]
    pub fn try_execute(&mut self, code_id: u64, frame: &mut Frame) -> Option<ExecutionResult> {
        // Fast path: lookup compiled code
        let entry = self.lookup(code_id)?;

        // Record cache hit
        self.stats.cache_hits += 1;

        // Execute compiled code
        Some(self.execute_entry(&entry, frame))
    }

    /// Execute compiled code from a cached entry.
    fn execute_entry(&mut self, entry: &CompiledEntry, frame: &mut Frame) -> ExecutionResult {
        self.bridge.execute(entry, frame)
    }

    /// Record a cache miss for statistics.
    #[inline]
    pub fn record_miss(&mut self) {
        self.stats.cache_misses += 1;
    }

    // =========================================================================
    // OSR (On-Stack Replacement)
    // =========================================================================

    /// Try to enter compiled code via OSR at a loop header.
    ///
    /// Returns `Some(result)` if OSR entry succeeded,
    /// `None` if no OSR entry point exists.
    ///
    /// # Arguments
    ///
    /// * `code_id` - Unique identifier for the code object
    /// * `bc_offset` - Bytecode offset of the loop header
    /// * `frame` - The interpreter frame
    #[inline]
    pub fn try_osr(
        &mut self,
        code_id: u64,
        bc_offset: u32,
        frame: &mut Frame,
    ) -> Option<ExecutionResult> {
        if !self.config.enable_osr {
            return None;
        }

        // Look up compiled code with OSR entries
        let entry = self.lookup(code_id)?;
        let osr_entries = entry.osr_entries()?;

        // Check if OSR entry exists at this offset
        if osr_entries.lookup_entry(bc_offset).is_some() {
            self.stats.osr_entries += 1;
            Some(self.bridge.execute_osr(&entry, frame, bc_offset))
        } else {
            None
        }
    }

    // =========================================================================
    // Tier-Up Handling
    // =========================================================================

    /// Handle a tier-up decision from the profiler.
    ///
    /// This may trigger compilation immediately (eager mode) or
    /// queue it for background compilation.
    ///
    /// Returns `true` if compilation was triggered.
    #[inline(never)] // Cold path - don't inline into hot loop
    pub fn handle_tier_up(&mut self, code: &Arc<CodeObject>, decision: TierUpDecision) -> bool {
        match decision {
            TierUpDecision::None => false,
            TierUpDecision::Tier1 => self.trigger_compilation(code, 1),
            TierUpDecision::Tier2 => self.trigger_compilation(code, 2),
        }
    }

    /// Trigger compilation for a code object.
    ///
    /// In eager mode (for testing), compiles synchronously.
    /// Otherwise, queues for background compilation.
    fn trigger_compilation(&mut self, code: &Arc<CodeObject>, tier: u8) -> bool {
        self.stats.compilations_triggered += 1;

        let code_id = Arc::as_ptr(code) as u64;

        // Check if already compiled at this tier or higher
        if let Some(entry) = self.lookup(code_id) {
            if entry.tier() >= tier {
                return false;
            }
        }

        if self.config.eager_compilation || !self.config.background_compilation {
            // Synchronous compilation - dispatch based on tier
            let result = if tier >= 2 {
                // Tier 2 remains correctness-gated until full value-tag/ABI
                // parity is guaranteed across the optimizing pipeline.
                // Keep execution on the validated synchronous tier.
                self.bridge.compile_sync(code)
            } else {
                self.bridge.compile_sync(code)
            };

            match result {
                Ok(entry) => {
                    self.stats.compilations_completed += 1;
                    self.stats.compiled_bytes += entry.code_size() as u64;
                    true
                }
                Err(_e) => {
                    self.stats.compilations_failed += 1;
                    // Log error in debug builds
                    #[cfg(debug_assertions)]
                    eprintln!("JIT compilation failed for tier {}: {}", tier, _e);
                    false
                }
            }
        } else {
            // Queue for background compilation
            self.bridge.compile_async(Arc::clone(code), tier);
            true
        }
    }

    /// Query tier-up policy using the active bridge configuration.
    #[inline]
    pub fn check_tier_up(&self, profiler: &Profiler, code_id: CodeId) -> TierUpDecision {
        self.bridge.check_tier_up(profiler, code_id)
    }

    // =========================================================================
    // Deoptimization
    // =========================================================================

    /// Handle deoptimization from JIT code.
    ///
    /// Updates statistics and may invalidate code based on reason.
    pub fn handle_deopt(&mut self, code_id: u64, reason: DeoptReason) {
        self.stats.deopts += 1;

        if matches!(reason, DeoptReason::OsrExit) {
            self.stats.osr_exits += 1;
        }

        // Delegate to bridge for invalidation decisions
        self.bridge.handle_deopt(code_id, reason);
    }

    /// Handle an execution result, updating statistics as needed.
    ///
    /// Returns the bytecode offset to resume at (for deopt),
    /// or None if execution completed normally.
    pub fn process_result(&mut self, code_id: u64, result: ExecutionResult) -> ProcessedResult {
        match result {
            ExecutionResult::Return(value) => ProcessedResult::Return(value),
            ExecutionResult::Deopt { bc_offset, reason } => {
                self.handle_deopt(code_id, reason);
                ProcessedResult::Resume { bc_offset }
            }
            ExecutionResult::Exception(err) => ProcessedResult::Error(err),
            ExecutionResult::TailCall { target, arg_count } => {
                ProcessedResult::TailCall { target, arg_count }
            }
        }
    }

    // =========================================================================
    // Statistics and Diagnostics
    // =========================================================================

    /// Get JIT execution statistics.
    #[inline]
    pub fn stats(&self) -> &JitStats {
        &self.stats
    }

    /// Get mutable statistics (for testing).
    #[inline]
    pub fn stats_mut(&mut self) -> &mut JitStats {
        &mut self.stats
    }

    /// Get the number of compiled functions.
    #[inline]
    pub fn compiled_count(&self) -> usize {
        self.bridge.compiled_count()
    }

    /// Get total compiled code size.
    #[inline]
    pub fn compiled_size(&self) -> usize {
        self.bridge.compiled_size()
    }

    /// Get configuration.
    #[inline]
    pub fn config(&self) -> &JitConfig {
        &self.config
    }

    /// Get mutable bridge reference (for advanced use).
    #[inline]
    pub fn bridge_mut(&mut self) -> &mut JitBridge {
        &mut self.bridge
    }

    /// Invalidate all compiled code.
    pub fn invalidate_all(&self) {
        self.bridge.invalidate_all()
    }
}

// =============================================================================
// Processed Result
// =============================================================================

/// Result of processing a JIT execution result.
pub enum ProcessedResult {
    /// Normal return with value.
    Return(prism_core::Value),
    /// Resume interpreter at bytecode offset.
    Resume { bc_offset: u32 },
    /// Runtime error occurred.
    Error(RuntimeError),
    /// Tail call to another function.
    TailCall { target: u64, arg_count: u8 },
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_jit_config_default() {
        let config = JitConfig::default();
        assert!(config.enabled);
        assert!(config.background_compilation);
        assert!(config.enable_osr);
    }

    #[test]
    fn test_jit_config_disabled() {
        let config = JitConfig::disabled();
        assert!(!config.enabled);
    }

    #[test]
    fn test_jit_config_testing() {
        let config = JitConfig::for_testing();
        assert!(config.enabled);
        assert!(!config.background_compilation);
        assert!(config.eager_compilation);
        assert_eq!(config.tier1_threshold, 10);
    }

    #[test]
    fn test_jit_context_creation() {
        let ctx = JitContext::with_defaults();
        assert!(ctx.is_enabled());
        assert_eq!(ctx.compiled_count(), 0);
    }

    #[test]
    fn test_jit_context_lookup_miss() {
        let ctx = JitContext::with_defaults();
        assert!(ctx.lookup(12345).is_none());
    }

    #[test]
    fn test_jit_stats_hit_rate() {
        let mut stats = JitStats::default();
        stats.cache_hits = 90;
        stats.cache_misses = 10;
        assert!((stats.hit_rate() - 0.9).abs() < 0.001);
    }

    #[test]
    fn test_jit_stats_deopt_rate() {
        let mut stats = JitStats::default();
        stats.cache_hits = 100;
        stats.deopts = 5;
        assert!((stats.deopt_rate() - 0.05).abs() < 0.001);
    }

    #[test]
    fn test_jit_context_tier_up_none() {
        let mut ctx = JitContext::with_defaults();
        let code = Arc::new(CodeObject::new("test", "<test>"));
        assert!(!ctx.handle_tier_up(&code, TierUpDecision::None));
    }

    #[test]
    fn test_processed_result() {
        use prism_core::Value;

        let result = ProcessedResult::Return(Value::int(42).unwrap());
        match result {
            ProcessedResult::Return(v) => assert_eq!(v.as_int(), Some(42)),
            _ => panic!("Expected Return"),
        }

        let result = ProcessedResult::Resume { bc_offset: 100 };
        match result {
            ProcessedResult::Resume { bc_offset } => assert_eq!(bc_offset, 100),
            _ => panic!("Expected Resume"),
        }
    }

    #[test]
    fn test_check_tier_up_uses_runtime_config_thresholds() {
        let ctx = JitContext::for_testing();
        let mut profiler = Profiler::new();
        let code_id = CodeId::new(42);

        for _ in 0..9 {
            profiler.record_call(code_id);
            assert_eq!(ctx.check_tier_up(&profiler, code_id), TierUpDecision::None);
        }

        profiler.record_call(code_id);
        assert_eq!(ctx.check_tier_up(&profiler, code_id), TierUpDecision::Tier1);

        for _ in 0..90 {
            profiler.record_call(code_id);
        }
        assert_eq!(ctx.check_tier_up(&profiler, code_id), TierUpDecision::Tier2);
    }
}
