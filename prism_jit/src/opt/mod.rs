//! Optimization passes for the Sea-of-Nodes IR.
//!
//! This module contains the optimization pipeline and individual optimization passes:
//!
//! - **GVN (Global Value Numbering)**: Eliminates redundant computations
//! - **DCE (Dead Code Elimination)**: Removes unreachable nodes
//! - **Simplify**: Algebraic simplifications and constant folding
//!
//! # Optimization Pipeline
//!
//! The optimization pipeline runs passes in a specific order:
//!
//! 1. **Simplify**: Constant folding and algebraic simplifications
//! 2. **GVN**: Eliminate redundant computations
//! 3. **DCE**: Remove dead nodes
//!
//! The pipeline runs until a fixed point is reached (no more changes).

pub mod dce;
pub mod escape;
pub mod gvn;
pub mod inline;
pub mod licm;
pub mod loop_analyzer;
pub mod pipeline;
pub mod simplify;

// Re-export key types
pub use escape::{Escape, EscapeAnalysis, EscapeState};
pub use inline::{Inline, InlineConfig};
pub use licm::Licm;
pub use loop_analyzer::LoopInvariantAnalysis;
pub use pipeline::{OptPipeline, PipelineConfig};

use crate::ir::graph::Graph;

// =============================================================================
// Optimization Pass Trait
// =============================================================================

/// A single optimization pass over the IR graph.
pub trait OptimizationPass {
    /// The name of this pass (for debugging/logging).
    fn name(&self) -> &'static str;

    /// Run the optimization pass on the graph.
    /// Returns true if the graph was modified.
    fn run(&mut self, graph: &mut Graph) -> bool;
}

// =============================================================================
// Optimization Pipeline
// =============================================================================

/// Configuration for the optimization pipeline.
#[derive(Debug, Clone)]
pub struct OptConfig {
    /// Maximum number of iterations before giving up.
    pub max_iterations: usize,
    /// Enable GVN.
    pub enable_gvn: bool,
    /// Enable DCE.
    pub enable_dce: bool,
    /// Enable algebraic simplifications.
    pub enable_simplify: bool,
}

impl Default for OptConfig {
    fn default() -> Self {
        OptConfig {
            max_iterations: 10,
            enable_gvn: true,
            enable_dce: true,
            enable_simplify: true,
        }
    }
}

/// Statistics from the optimization pipeline.
#[derive(Debug, Default, Clone)]
pub struct OptStats {
    /// Number of iterations run.
    pub iterations: usize,
    /// Number of nodes removed by DCE.
    pub nodes_removed: usize,
    /// Number of nodes deduplicated by GVN.
    pub nodes_deduplicated: usize,
    /// Number of simplifications applied.
    pub simplifications: usize,
}

/// Run the optimization pipeline on a graph.
pub fn optimize(graph: &mut Graph, config: &OptConfig) -> OptStats {
    let mut stats = OptStats::default();

    for iteration in 0..config.max_iterations {
        stats.iterations = iteration + 1;
        let mut changed = false;

        // Simplify first (enables more GVN/DCE opportunities)
        if config.enable_simplify {
            let mut pass = simplify::Simplify::new();
            if pass.run(graph) {
                changed = true;
                stats.simplifications += pass.simplifications();
            }
        }

        // GVN to eliminate redundant computations
        if config.enable_gvn {
            let mut pass = gvn::Gvn::new();
            if pass.run(graph) {
                changed = true;
                stats.nodes_deduplicated += pass.deduplicated();
            }
        }

        // DCE to remove dead nodes
        if config.enable_dce {
            let mut pass = dce::Dce::new();
            if pass.run(graph) {
                changed = true;
                stats.nodes_removed += pass.removed();
            }
        }

        if !changed {
            break;
        }
    }

    stats
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::builder::{
        ArithmeticBuilder, ContainerBuilder, ControlBuilder, GraphBuilder, ObjectBuilder,
    };

    #[test]
    fn test_default_config() {
        let config = OptConfig::default();
        assert!(config.enable_gvn);
        assert!(config.enable_dce);
        assert!(config.enable_simplify);
        assert_eq!(config.max_iterations, 10);
    }

    #[test]
    fn test_optimize_simple() {
        let mut builder = GraphBuilder::new(4, 2);

        // Create a simple computation: p0 + p1
        let p0 = builder.parameter(0).unwrap();
        let p1 = builder.parameter(1).unwrap();
        let sum = builder.int_add(p0, p1);
        let _ret = builder.return_value(sum);

        let mut graph = builder.finish();
        let config = OptConfig::default();

        let stats = optimize(&mut graph, &config);
        assert!(stats.iterations >= 1);
    }
}
