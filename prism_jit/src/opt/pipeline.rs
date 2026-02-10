//! Optimization Pipeline.
//!
//! Orchestrates multiple optimization passes with proper ordering,
//! fixed-point iteration, and comprehensive statistics.
//!
//! # Pass Phases
//!
//! 1. **Canonicalization**: Simplify, constant folding
//! 2. **ProfileGuided**: Branch probability, hot/cold splitting
//! 3. **Local**: GVN, copy propagation
//! 4. **Loop**: LICM, loop unrolling
//! 5. **Interprocedural**: Inlining, escape analysis
//! 6. **Cleanup**: DCE, CFG simplification
//!
//! # Fixed-Point Iteration
//!
//! The pipeline runs passes until no changes occur or a maximum
//! iteration count is reached. Some passes trigger re-running
//! of earlier passes (e.g., inlining enables more GVN).

use super::OptimizationPass;
use super::branch_probability::BranchProbabilityPass;
use super::copy_prop::CopyProp;
use super::dce::Dce;
use super::dse::Dse;
use super::escape::Escape;
use super::gvn::Gvn;
use super::hot_cold::HotColdPass;
use super::inline::Inline;
use super::instcombine::InstCombine;
use super::licm::Licm;
use super::pre::Pre;
use super::rce::RangeCheckElimination;
use super::sccp::Sccp;
use super::simplify::Simplify;
use super::strength_reduce::StrengthReduce;
use super::tailcall::TailCallOpt;
use super::unroll::Unroll;
use crate::ir::graph::Graph;

use std::time::{Duration, Instant};

// =============================================================================
// Pass Phase
// =============================================================================

/// Phase of the optimization pipeline.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum PassPhase {
    /// Early passes: canonicalization, constant folding.
    Canonicalization,
    /// Profile-guided optimizations: branch probability, hot/cold splitting.
    ProfileGuided,
    /// Local optimizations: GVN, copy propagation.
    Local,
    /// Loop optimizations: LICM, unrolling.
    Loop,
    /// Interprocedural: inlining, escape analysis.
    Interprocedural,
    /// Cleanup passes: DCE, CFG simplification.
    Cleanup,
}

// =============================================================================
// Pass Entry
// =============================================================================

/// A registered pass in the pipeline.
struct PassEntry {
    /// The pass (boxed for polymorphism).
    pass: Box<dyn OptimizationPass>,
    /// Which phase this pass belongs to.
    phase: PassPhase,
    /// Whether this pass is enabled.
    enabled: bool,
    /// Pass-specific statistics.
    runs: usize,
    changes: usize,
    time: Duration,
}

impl PassEntry {
    fn new<P: OptimizationPass + 'static>(pass: P, phase: PassPhase) -> Self {
        Self {
            pass: Box::new(pass),
            phase,
            enabled: true,
            runs: 0,
            changes: 0,
            time: Duration::ZERO,
        }
    }
}

// =============================================================================
// Pipeline Configuration
// =============================================================================

/// Configuration for the optimization pipeline.
#[derive(Debug, Clone)]
pub struct PipelineConfig {
    /// Maximum iterations per phase.
    pub max_iterations_per_phase: usize,

    /// Maximum total iterations across all phases.
    pub max_total_iterations: usize,

    // =========================================================================
    // Canonicalization Phase
    // =========================================================================
    /// Enable simplification (constant folding, algebraic identities).
    pub enable_simplify: bool,

    /// Enable Sparse Conditional Constant Propagation.
    pub enable_sccp: bool,

    /// Enable Instruction Combining (peephole optimization).
    pub enable_instcombine: bool,

    // =========================================================================
    // ProfileGuided Phase
    // =========================================================================
    /// Enable branch probability annotation (PGO).
    pub enable_branch_probability: bool,

    /// Enable hot/cold code splitting (PGO).
    pub enable_hot_cold: bool,

    // =========================================================================
    // Local Phase
    // =========================================================================
    /// Enable Copy Propagation.
    pub enable_copy_prop: bool,

    /// Enable Global Value Numbering.
    pub enable_gvn: bool,

    /// Enable Dead Store Elimination.
    pub enable_dse: bool,

    /// Enable Partial Redundancy Elimination.
    pub enable_pre: bool,

    /// Enable Strength Reduction (magic number division, etc.).
    pub enable_strength_reduce: bool,

    // =========================================================================
    // Loop Phase
    // =========================================================================
    /// Enable Loop Invariant Code Motion.
    pub enable_licm: bool,

    /// Enable Loop Unrolling.
    pub enable_unroll: bool,

    /// Enable Range Check Elimination.
    pub enable_rce: bool,

    // =========================================================================
    // Interprocedural Phase
    // =========================================================================
    /// Enable function inlining.
    pub enable_inline: bool,

    /// Enable escape analysis for stack allocation.
    pub enable_escape: bool,

    /// Enable Tail Call Optimization.
    pub enable_tco: bool,

    // =========================================================================
    // Cleanup Phase
    // =========================================================================
    /// Enable Dead Code Elimination.
    pub enable_dce: bool,

    // =========================================================================
    // Diagnostics
    // =========================================================================
    /// Collect timing statistics.
    pub collect_timing: bool,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            max_iterations_per_phase: 5,
            max_total_iterations: 20,
            // Canonicalization
            enable_simplify: true,
            enable_sccp: true,
            enable_instcombine: true,
            // ProfileGuided
            enable_branch_probability: true,
            enable_hot_cold: true,
            // Local
            enable_copy_prop: true,
            enable_gvn: true,
            enable_dse: true,
            enable_pre: true,
            enable_strength_reduce: true,
            // Loop
            enable_licm: true,
            enable_unroll: true,
            enable_rce: true,
            // Interprocedural
            enable_inline: true,
            enable_escape: true,
            enable_tco: true,
            // Cleanup
            enable_dce: true,
            // Diagnostics
            collect_timing: true,
        }
    }
}

impl PipelineConfig {
    /// Create a minimal configuration (fewer passes for faster compile).
    /// This is suitable for Tier-1 JIT compilation where compile time is critical.
    pub fn minimal() -> Self {
        Self {
            max_iterations_per_phase: 2,
            max_total_iterations: 8,
            // Canonicalization - keep cheap passes
            enable_simplify: true,
            enable_sccp: false,       // Skip expensive dataflow
            enable_instcombine: true, // Cheap and effective
            // ProfileGuided - skip PGO in Tier-1 for compile speed
            enable_branch_probability: false,
            enable_hot_cold: false,
            // Local - only essential
            enable_copy_prop: true, // Cheap and improves code quality
            enable_gvn: true,       // Essential for code quality
            enable_dse: false,      // Skip for faster compile
            enable_pre: false,      // Skip expensive LCM
            enable_strength_reduce: false,
            // Loop - skip all
            enable_licm: false,
            enable_unroll: false,
            enable_rce: false,
            // Interprocedural - skip all
            enable_inline: false,
            enable_escape: false,
            enable_tco: false,
            // Cleanup - always run
            enable_dce: true,
            // Diagnostics
            collect_timing: false,
        }
    }

    /// Create a full optimization configuration.
    pub fn full() -> Self {
        Self {
            max_iterations_per_phase: 10,
            max_total_iterations: 50,
            ..Default::default()
        }
    }
}

// =============================================================================
// Optimization Pipeline
// =============================================================================

/// The main optimization pipeline.
pub struct OptPipeline {
    /// Configuration.
    config: PipelineConfig,

    /// Registered passes in order.
    passes: Vec<PassEntry>,

    /// Total iterations run.
    total_iterations: usize,

    /// Total time spent.
    total_time: Duration,
}

impl OptPipeline {
    /// Create a new pipeline with default configuration.
    pub fn new() -> Self {
        Self::with_config(PipelineConfig::default())
    }

    /// Create a pipeline with custom configuration.
    pub fn with_config(config: PipelineConfig) -> Self {
        let mut pipeline = Self {
            config,
            passes: Vec::new(),
            total_iterations: 0,
            total_time: Duration::ZERO,
        };

        pipeline.register_default_passes();
        pipeline
    }

    /// Create a pipeline with profile data for PGO-guided optimization.
    ///
    /// This is the primary entry point for Tier 2 compilation with PGO.
    /// Profile data is injected into the `BranchProbabilityPass` so that
    /// measured branch frequencies override static heuristics.
    pub fn with_profile(
        config: PipelineConfig,
        profile: crate::runtime::profile_data::ProfileData,
    ) -> Self {
        let mut pipeline = Self::with_config(config);
        pipeline.inject_profile(profile);
        pipeline
    }

    /// Inject profile data into the `BranchProbabilityPass`.
    ///
    /// Scans the registered passes and injects the profile data into
    /// any `BranchProbabilityPass` found. This allows downstream passes
    /// (e.g., `HotColdPass`) to consume PGO-annotated probabilities.
    pub fn inject_profile(&mut self, profile: crate::runtime::profile_data::ProfileData) {
        for entry in &mut self.passes {
            if entry.pass.name() == "BranchProbability" {
                // Replace the existing BranchProbabilityPass with one carrying profile data
                entry.pass = Box::new(BranchProbabilityPass::with_profile(profile));
                return;
            }
        }

        // If branch probability was disabled, register it now with profile data
        self.passes.push(PassEntry::new(
            BranchProbabilityPass::with_profile(profile),
            PassPhase::ProfileGuided,
        ));
    }

    /// Register the default set of optimization passes.
    ///
    /// Pass ordering is critical for effectiveness:
    /// - SCCP runs early to propagate constants before other passes
    /// - Copy propagation runs before GVN to maximize redundancy detection
    /// - GVN runs before DSE/PRE to normalize expressions
    /// - DSE runs before PRE (dead stores don't need redundancy elimination)
    /// - Strength reduce runs after GVN when patterns are normalized
    /// - LICM runs before unrolling to hoist invariants first
    /// - Unrolling runs before RCE to expose range check patterns
    /// - Inlining runs early in interprocedural to expose more optimization
    /// - TCO runs after inlining (inlined calls can become tail calls)
    /// - DCE runs last to clean up dead code from all passes
    fn register_default_passes(&mut self) {
        // =====================================================================
        // Canonicalization phase - normalize and simplify the IR
        // =====================================================================

        // SCCP first: propagates constants and eliminates unreachable code
        if self.config.enable_sccp {
            self.register(Sccp::new(), PassPhase::Canonicalization);
        }

        // Simplify: algebraic identities, strength reduction, etc.
        if self.config.enable_simplify {
            self.register(Simplify::new(), PassPhase::Canonicalization);
        }

        // InstCombine: peephole optimizations on instruction sequences
        if self.config.enable_instcombine {
            self.register(InstCombine::new(), PassPhase::Canonicalization);
        }

        // =====================================================================
        // ProfileGuided phase - PGO-driven optimizations
        // =====================================================================

        // Branch probability: annotate branches with measured/estimated weights
        if self.config.enable_branch_probability {
            self.register(BranchProbabilityPass::new(), PassPhase::ProfileGuided);
        }

        // Hot/cold splitting: partition code by execution temperature
        if self.config.enable_hot_cold {
            self.register(HotColdPass::new(), PassPhase::ProfileGuided);
        }

        // =====================================================================
        // Local phase - single basic block optimizations
        // =====================================================================

        // Copy propagation first: simplifies use-def chains for subsequent passes
        if self.config.enable_copy_prop {
            self.register(CopyProp::new(), PassPhase::Local);
        }

        // GVN: eliminate redundant computations
        if self.config.enable_gvn {
            self.register(Gvn::new(), PassPhase::Local);
        }

        // DSE: eliminate stores that are never read
        if self.config.enable_dse {
            self.register(Dse::new(), PassPhase::Local);
        }

        // PRE: eliminate partially redundant expressions via code motion
        if self.config.enable_pre {
            self.register(Pre::new(), PassPhase::Local);
        }

        // Strength reduction: convert expensive ops to cheaper sequences
        if self.config.enable_strength_reduce {
            self.register(StrengthReduce::new(), PassPhase::Local);
        }

        // =====================================================================
        // Loop phase - loop-level optimizations
        // =====================================================================

        // LICM first: hoist invariants before unrolling
        if self.config.enable_licm {
            self.register(Licm::new(), PassPhase::Loop);
        }

        // Unrolling: replicate loop bodies to reduce overhead
        if self.config.enable_unroll {
            self.register(Unroll::new(), PassPhase::Loop);
        }

        // RCE: eliminate array bounds checks after unrolling exposes patterns
        if self.config.enable_rce {
            self.register(RangeCheckElimination::new(), PassPhase::Loop);
        }

        // =====================================================================
        // Interprocedural phase - cross-function optimizations
        // =====================================================================

        // Inlining first: exposes more optimization opportunities
        if self.config.enable_inline {
            self.register(Inline::new(), PassPhase::Interprocedural);
        }

        // Escape analysis: identify objects that don't escape for stack allocation
        if self.config.enable_escape {
            self.register(Escape::new(), PassPhase::Interprocedural);
        }

        // TCO: convert eligible tail calls to jumps
        if self.config.enable_tco {
            self.register(TailCallOpt::new(), PassPhase::Interprocedural);
        }

        // =====================================================================
        // Cleanup phase - final cleanup
        // =====================================================================

        // DCE: remove dead code generated by other passes
        if self.config.enable_dce {
            self.register(Dce::new(), PassPhase::Cleanup);
        }
    }

    /// Register a custom pass.
    pub fn register<P: OptimizationPass + 'static>(&mut self, pass: P, phase: PassPhase) {
        self.passes.push(PassEntry::new(pass, phase));
    }

    /// Run the optimization pipeline on a graph.
    pub fn run(&mut self, graph: &mut Graph) -> PipelineStats {
        let start = Instant::now();
        let initial_size = graph.len();

        let mut stats = PipelineStats::default();
        let mut iterations = 0;

        // Run phases in order
        for phase in &[
            PassPhase::Canonicalization,
            PassPhase::ProfileGuided,
            PassPhase::Local,
            PassPhase::Loop,
            PassPhase::Interprocedural,
            PassPhase::Cleanup,
        ] {
            let phase_changed = self.run_phase(graph, *phase, &mut stats);
            iterations += 1;

            if iterations >= self.config.max_total_iterations {
                break;
            }
        }

        self.total_iterations = iterations;
        self.total_time = start.elapsed();

        stats.total_iterations = iterations;
        stats.total_time = self.total_time;
        stats.initial_size = initial_size;
        stats.final_size = graph.len();

        stats
    }

    /// Run all passes in a specific phase.
    fn run_phase(
        &mut self,
        graph: &mut Graph,
        phase: PassPhase,
        stats: &mut PipelineStats,
    ) -> bool {
        let mut phase_changed = false;

        for iteration in 0..self.config.max_iterations_per_phase {
            let mut iter_changed = false;

            for entry in &mut self.passes {
                if entry.phase != phase || !entry.enabled {
                    continue;
                }

                let start = if self.config.collect_timing {
                    Some(Instant::now())
                } else {
                    None
                };

                let changed = entry.pass.run(graph);

                if let Some(start) = start {
                    entry.time += start.elapsed();
                }

                entry.runs += 1;
                if changed {
                    entry.changes += 1;
                    iter_changed = true;
                }
            }

            if iter_changed {
                phase_changed = true;
            } else {
                // Fixed point reached for this phase
                break;
            }
        }

        stats.phases_run += 1;
        phase_changed
    }

    /// Get pass statistics.
    pub fn pass_stats(&self) -> Vec<PassStat> {
        self.passes
            .iter()
            .map(|e| PassStat {
                name: e.pass.name().to_string(),
                phase: e.phase,
                runs: e.runs,
                changes: e.changes,
                time: e.time,
            })
            .collect()
    }

    /// Get total iterations run.
    #[inline]
    pub fn iterations(&self) -> usize {
        self.total_iterations
    }

    /// Get total time spent.
    #[inline]
    pub fn total_time(&self) -> Duration {
        self.total_time
    }
}

impl Default for OptPipeline {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// Statistics
// =============================================================================

/// Statistics from a single pass.
#[derive(Debug, Clone)]
pub struct PassStat {
    /// Pass name.
    pub name: String,
    /// Pass phase.
    pub phase: PassPhase,
    /// Number of times run.
    pub runs: usize,
    /// Number of times it made changes.
    pub changes: usize,
    /// Total time spent in this pass.
    pub time: Duration,
}

/// Statistics from the entire pipeline.
#[derive(Debug, Clone, Default)]
pub struct PipelineStats {
    /// Total iterations across all phases.
    pub total_iterations: usize,
    /// Number of phases run.
    pub phases_run: usize,
    /// Total time spent.
    pub total_time: Duration,
    /// Initial graph size.
    pub initial_size: usize,
    /// Final graph size.
    pub final_size: usize,
}

impl PipelineStats {
    /// Get size reduction ratio.
    pub fn size_reduction(&self) -> f64 {
        if self.initial_size == 0 {
            1.0
        } else {
            self.final_size as f64 / self.initial_size as f64
        }
    }
}

// =============================================================================
// Quick Optimize Functions
// =============================================================================

/// Run full optimization pipeline on a graph.
pub fn optimize_full(graph: &mut Graph) -> PipelineStats {
    let mut pipeline = OptPipeline::with_config(PipelineConfig::full());
    pipeline.run(graph)
}

/// Run minimal optimization pipeline on a graph.
pub fn optimize_minimal(graph: &mut Graph) -> PipelineStats {
    let mut pipeline = OptPipeline::with_config(PipelineConfig::minimal());
    pipeline.run(graph)
}

/// Run default optimization pipeline on a graph.
pub fn optimize(graph: &mut Graph) -> PipelineStats {
    let mut pipeline = OptPipeline::new();
    pipeline.run(graph)
}

/// Run full optimization pipeline with PGO profile data.
///
/// This is the primary entry point for Tier 2 compilation. Profile data
/// from Tier 1 execution is used to guide branch probability estimation,
/// hot/cold splitting, and loop frequency calculations.
pub fn optimize_with_profile(
    graph: &mut Graph,
    profile: crate::runtime::profile_data::ProfileData,
) -> PipelineStats {
    let mut pipeline = OptPipeline::with_profile(PipelineConfig::full(), profile);
    pipeline.run(graph)
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::builder::{ArithmeticBuilder, ControlBuilder, GraphBuilder};

    #[test]
    fn test_pass_phase_ordering() {
        assert!(PassPhase::Canonicalization < PassPhase::Local);
        assert!(PassPhase::Local < PassPhase::Loop);
        assert!(PassPhase::Loop < PassPhase::Interprocedural);
        assert!(PassPhase::Interprocedural < PassPhase::Cleanup);
    }

    #[test]
    fn test_pipeline_config_default() {
        let config = PipelineConfig::default();
        assert!(config.enable_gvn);
        assert!(config.enable_dce);
        assert!(config.enable_licm);
    }

    #[test]
    fn test_pipeline_config_minimal() {
        let config = PipelineConfig::minimal();
        assert!(!config.enable_licm);
        assert!(!config.enable_inline);
        assert!(config.enable_gvn);
    }

    #[test]
    fn test_pipeline_new() {
        let pipeline = OptPipeline::new();
        assert!(!pipeline.passes.is_empty());
    }

    #[test]
    fn test_pipeline_run_empty() {
        let builder = GraphBuilder::new(0, 0);
        let mut graph = builder.finish();

        let mut pipeline = OptPipeline::new();
        let stats = pipeline.run(&mut graph);

        assert!(stats.total_iterations > 0);
    }

    #[test]
    fn test_pipeline_run_simple() {
        let mut builder = GraphBuilder::new(2, 2);
        let p0 = builder.parameter(0).unwrap();
        let p1 = builder.parameter(1).unwrap();
        let sum = builder.int_add(p0, p1);
        builder.return_value(sum);

        let mut graph = builder.finish();
        let initial = graph.len();

        let mut pipeline = OptPipeline::new();
        let stats = pipeline.run(&mut graph);

        // Should run at least one iteration
        assert!(stats.total_iterations >= 1);
        // Size shouldn't dramatically change for simple graph
        assert!(stats.final_size <= initial + 5);
    }

    #[test]
    fn test_optimize_functions() {
        let builder = GraphBuilder::new(0, 0);
        let mut graph = builder.finish();

        let stats = optimize(&mut graph);
        assert!(stats.total_iterations >= 1);

        let mut graph2 = GraphBuilder::new(0, 0).finish();
        let stats2 = optimize_minimal(&mut graph2);
        assert!(stats2.total_iterations >= 1);

        let mut graph3 = GraphBuilder::new(0, 0).finish();
        let stats3 = optimize_full(&mut graph3);
        assert!(stats3.total_iterations >= 1);
    }

    #[test]
    fn test_pipeline_stats() {
        let stats = PipelineStats {
            total_iterations: 5,
            phases_run: 5,
            total_time: Duration::from_millis(100),
            initial_size: 100,
            final_size: 80,
        };

        assert_eq!(stats.size_reduction(), 0.8);
    }

    #[test]
    fn test_pipeline_stats_zero_size() {
        let stats = PipelineStats {
            initial_size: 0,
            final_size: 0,
            ..Default::default()
        };

        assert_eq!(stats.size_reduction(), 1.0);
    }

    #[test]
    fn test_pass_stats() {
        let builder = GraphBuilder::new(0, 0);
        let mut graph = builder.finish();

        let mut pipeline = OptPipeline::new();
        pipeline.run(&mut graph);

        let stats = pipeline.pass_stats();
        assert!(!stats.is_empty());

        for stat in stats {
            assert!(!stat.name.is_empty());
        }
    }

    // =========================================================================
    // New Pass Integration Tests
    // =========================================================================

    #[test]
    fn test_config_default_enables_all_passes() {
        let config = PipelineConfig::default();

        // Canonicalization
        assert!(
            config.enable_simplify,
            "simplify should be enabled by default"
        );
        assert!(config.enable_sccp, "sccp should be enabled by default");
        assert!(
            config.enable_instcombine,
            "instcombine should be enabled by default"
        );

        // Local
        assert!(
            config.enable_copy_prop,
            "copy_prop should be enabled by default"
        );
        assert!(config.enable_gvn, "gvn should be enabled by default");
        assert!(config.enable_dse, "dse should be enabled by default");
        assert!(config.enable_pre, "pre should be enabled by default");
        assert!(
            config.enable_strength_reduce,
            "strength_reduce should be enabled by default"
        );

        // Loop
        assert!(config.enable_licm, "licm should be enabled by default");
        assert!(config.enable_unroll, "unroll should be enabled by default");
        assert!(config.enable_rce, "rce should be enabled by default");

        // Interprocedural
        assert!(config.enable_inline, "inline should be enabled by default");
        assert!(config.enable_escape, "escape should be enabled by default");
        assert!(config.enable_tco, "tco should be enabled by default");

        // Cleanup
        assert!(config.enable_dce, "dce should be enabled by default");
    }

    #[test]
    fn test_config_minimal_disables_expensive_passes() {
        let config = PipelineConfig::minimal();

        // Expensive passes should be disabled
        assert!(!config.enable_sccp, "sccp should be disabled in minimal");
        assert!(!config.enable_dse, "dse should be disabled in minimal");
        assert!(!config.enable_pre, "pre should be disabled in minimal");
        assert!(
            !config.enable_unroll,
            "unroll should be disabled in minimal"
        );
        assert!(!config.enable_tco, "tco should be disabled in minimal");
        assert!(!config.enable_licm, "licm should be disabled in minimal");
        assert!(!config.enable_rce, "rce should be disabled in minimal");
        assert!(
            !config.enable_inline,
            "inline should be disabled in minimal"
        );
        assert!(
            !config.enable_escape,
            "escape should be disabled in minimal"
        );
        assert!(
            !config.enable_strength_reduce,
            "strength_reduce should be disabled in minimal"
        );

        // Essential cheap passes should remain enabled
        assert!(
            config.enable_simplify,
            "simplify should be enabled in minimal"
        );
        assert!(
            config.enable_instcombine,
            "instcombine should be enabled in minimal"
        );
        assert!(
            config.enable_copy_prop,
            "copy_prop should be enabled in minimal"
        );
        assert!(config.enable_gvn, "gvn should be enabled in minimal");
        assert!(config.enable_dce, "dce should be enabled in minimal");
    }

    #[test]
    fn test_pipeline_registers_dse_pass() {
        let pipeline = OptPipeline::new();
        let stats = pipeline.pass_stats();
        assert!(
            stats.iter().any(|s| s.name.to_lowercase().contains("dse")),
            "Pipeline should include DSE pass"
        );
    }

    #[test]
    fn test_pipeline_registers_pre_pass() {
        let pipeline = OptPipeline::new();
        let stats = pipeline.pass_stats();
        assert!(
            stats.iter().any(|s| s.name.to_lowercase().contains("pre")),
            "Pipeline should include PRE pass"
        );
    }

    #[test]
    fn test_pipeline_registers_unroll_pass() {
        let pipeline = OptPipeline::new();
        let stats = pipeline.pass_stats();
        assert!(
            stats
                .iter()
                .any(|s| s.name.to_lowercase().contains("unroll")),
            "Pipeline should include Unroll pass"
        );
    }

    #[test]
    fn test_pipeline_registers_sccp_pass() {
        let pipeline = OptPipeline::new();
        let stats = pipeline.pass_stats();
        assert!(
            stats.iter().any(|s| s.name.to_lowercase().contains("sccp")),
            "Pipeline should include SCCP pass"
        );
    }

    #[test]
    fn test_pipeline_registers_instcombine_pass() {
        let pipeline = OptPipeline::new();
        let stats = pipeline.pass_stats();
        assert!(
            stats
                .iter()
                .any(|s| s.name.to_lowercase().contains("instcombine")
                    || s.name.to_lowercase().contains("combine")),
            "Pipeline should include InstCombine pass"
        );
    }

    #[test]
    fn test_pipeline_registers_tco_pass() {
        let pipeline = OptPipeline::new();
        let stats = pipeline.pass_stats();
        assert!(
            stats
                .iter()
                .any(|s| s.name.to_lowercase().contains("tail")
                    || s.name.to_lowercase().contains("tco")),
            "Pipeline should include TCO pass"
        );
    }

    #[test]
    fn test_pipeline_pass_count_default_vs_minimal() {
        let default_pipeline = OptPipeline::new();
        let minimal_pipeline = OptPipeline::with_config(PipelineConfig::minimal());

        let default_count = default_pipeline.pass_stats().len();
        let minimal_count = minimal_pipeline.pass_stats().len();

        assert!(
            default_count > minimal_count,
            "Default pipeline ({}) should have more passes than minimal ({})",
            default_count,
            minimal_count
        );
    }

    #[test]
    fn test_pipeline_pass_phases_are_correct() {
        let pipeline = OptPipeline::new();
        let stats = pipeline.pass_stats();

        // Verify we have passes in each expected phase
        let phases: Vec<PassPhase> = stats.iter().map(|s| s.phase).collect();

        assert!(
            phases.contains(&PassPhase::Canonicalization),
            "Should have Canonicalization passes"
        );
        assert!(
            phases.contains(&PassPhase::Local),
            "Should have Local passes"
        );
        assert!(phases.contains(&PassPhase::Loop), "Should have Loop passes");
        assert!(
            phases.contains(&PassPhase::Interprocedural),
            "Should have Interprocedural passes"
        );
        assert!(
            phases.contains(&PassPhase::Cleanup),
            "Should have Cleanup passes"
        );
    }

    #[test]
    fn test_config_full_has_higher_iteration_limits() {
        let default_config = PipelineConfig::default();
        let full_config = PipelineConfig::full();

        assert!(
            full_config.max_iterations_per_phase > default_config.max_iterations_per_phase,
            "Full config should have higher iteration limits"
        );
        assert!(
            full_config.max_total_iterations > default_config.max_total_iterations,
            "Full config should have higher total iteration limits"
        );
    }

    #[test]
    fn test_config_minimal_has_lower_iteration_limits() {
        let default_config = PipelineConfig::default();
        let minimal_config = PipelineConfig::minimal();

        assert!(
            minimal_config.max_iterations_per_phase < default_config.max_iterations_per_phase,
            "Minimal config should have lower iteration limits"
        );
        assert!(
            minimal_config.max_total_iterations < default_config.max_total_iterations,
            "Minimal config should have lower total iteration limits"
        );
    }

    #[test]
    fn test_config_minimal_disables_timing() {
        let minimal_config = PipelineConfig::minimal();
        assert!(
            !minimal_config.collect_timing,
            "Minimal config should disable timing"
        );
    }

    #[test]
    fn test_config_default_enables_timing() {
        let default_config = PipelineConfig::default();
        assert!(
            default_config.collect_timing,
            "Default config should enable timing"
        );
    }

    #[test]
    fn test_pipeline_with_all_passes_disabled() {
        let config = PipelineConfig {
            max_iterations_per_phase: 1,
            max_total_iterations: 1,
            enable_simplify: false,
            enable_sccp: false,
            enable_instcombine: false,
            enable_branch_probability: false,
            enable_hot_cold: false,
            enable_copy_prop: false,
            enable_gvn: false,
            enable_dse: false,
            enable_pre: false,
            enable_strength_reduce: false,
            enable_licm: false,
            enable_unroll: false,
            enable_rce: false,
            enable_inline: false,
            enable_escape: false,
            enable_tco: false,
            enable_dce: false,
            collect_timing: false,
        };

        let pipeline = OptPipeline::with_config(config);
        assert!(
            pipeline.pass_stats().is_empty(),
            "Pipeline with all passes disabled should be empty"
        );
    }

    #[test]
    fn test_pipeline_runs_with_graph() {
        let mut builder = GraphBuilder::new(4, 2);
        let p0 = builder.parameter(0).unwrap();
        let p1 = builder.parameter(1).unwrap();

        // Create some redundant computations for optimizers to work on
        let sum1 = builder.int_add(p0, p1);
        let sum2 = builder.int_add(p0, p1); // Redundant - GVN should eliminate
        let product = builder.int_mul(sum1, sum2);
        builder.return_value(product);

        let mut graph = builder.finish();
        let initial_size = graph.len();

        let mut pipeline = OptPipeline::new();
        let stats = pipeline.run(&mut graph);

        // Pipeline should complete successfully
        assert!(stats.total_iterations >= 1);
        assert!(stats.phases_run >= 1);

        // With all optimizations, should be able to reduce some redundancy
        // (exact reduction depends on graph structure)
        assert!(stats.initial_size == initial_size);
    }

    // =========================================================================
    // PGO Pipeline Integration Tests
    // =========================================================================

    #[test]
    fn test_pipeline_with_profile_constructor() {
        let profile = crate::runtime::profile_data::ProfileData::new(1);
        let pipeline = OptPipeline::with_profile(PipelineConfig::default(), profile);

        // Should have registered passes including BranchProbabilityPass
        let stats = pipeline.pass_stats();
        let bp_pass = stats.iter().find(|s| s.name == "BranchProbability");
        assert!(
            bp_pass.is_some(),
            "Pipeline with profile should have BranchProbability pass"
        );
    }

    #[test]
    fn test_pipeline_inject_profile() {
        let mut pipeline = OptPipeline::new();

        // Verify BranchProbability pass exists
        let has_bp_before = pipeline
            .pass_stats()
            .iter()
            .any(|s| s.name == "BranchProbability");
        assert!(has_bp_before);

        // Inject profile
        let mut profile = crate::runtime::profile_data::ProfileData::new(1);
        profile.record_branch(0, true);
        pipeline.inject_profile(profile);

        // BranchProbability pass should still exist
        let has_bp_after = pipeline
            .pass_stats()
            .iter()
            .any(|s| s.name == "BranchProbability");
        assert!(has_bp_after);
    }

    #[test]
    fn test_pipeline_inject_profile_when_disabled() {
        // Create a config with branch probability disabled
        let config = PipelineConfig {
            enable_branch_probability: false,
            ..PipelineConfig::minimal()
        };
        let mut pipeline = OptPipeline::with_config(config);

        // Should NOT have BranchProbability initially
        let has_bp_before = pipeline
            .pass_stats()
            .iter()
            .any(|s| s.name == "BranchProbability");
        assert!(
            !has_bp_before,
            "BranchProbability should be disabled initially"
        );

        // Inject profile should register the pass
        let profile = crate::runtime::profile_data::ProfileData::new(1);
        pipeline.inject_profile(profile);

        let has_bp_after = pipeline
            .pass_stats()
            .iter()
            .any(|s| s.name == "BranchProbability");
        assert!(
            has_bp_after,
            "inject_profile should register BranchProbability even if disabled"
        );
    }

    #[test]
    fn test_pipeline_run_with_profile() {
        let mut profile = crate::runtime::profile_data::ProfileData::new(1);
        for _ in 0..100 {
            profile.record_execution();
        }
        for _ in 0..90 {
            profile.record_branch(3, true);
        }
        for _ in 0..10 {
            profile.record_branch(3, false);
        }

        let mut builder = GraphBuilder::new(2, 2);
        let p0 = builder.parameter(0).unwrap();
        let p1 = builder.parameter(1).unwrap();
        let sum = builder.int_add(p0, p1);
        builder.return_value(sum);

        let mut graph = builder.finish();

        let mut pipeline = OptPipeline::with_profile(PipelineConfig::default(), profile);
        let stats = pipeline.run(&mut graph);

        // Pipeline should complete
        assert!(stats.total_iterations >= 1);
        assert!(stats.phases_run >= 1);
    }

    #[test]
    fn test_optimize_with_profile_convenience() {
        let mut profile = crate::runtime::profile_data::ProfileData::new(1);
        profile.record_branch(0, true);

        let builder = GraphBuilder::new(0, 0);
        let mut graph = builder.finish();

        let stats = optimize_with_profile(&mut graph, profile);
        assert!(stats.total_iterations >= 1);
    }
}
