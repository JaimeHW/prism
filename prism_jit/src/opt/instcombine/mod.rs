//! Instruction Combining (InstCombine) optimization pass.
//!
//! Combines and simplifies sequences of instructions using pattern matching.
//!
//! # Algorithm Overview
//!
//! 1. Build worklist of all instructions
//! 2. Pop instruction from worklist
//! 3. Try all applicable patterns
//! 4. If transformed, add uses to worklist
//! 5. Repeat until worklist empty
//!
//! # Pattern Categories
//!
//! - Arithmetic: x + 0, x * 1, x - x, etc.
//! - Bitwise: x & 0, x | -1, x ^ x, etc.
//! - Comparison: x < x, x == x, etc.
//! - Memory: load after store, redundant stores
//! - Control: branch on constant, dead paths

mod patterns;
mod worklist;

pub use patterns::{Pattern, PatternMatch, PatternRegistry};
pub use worklist::Worklist;

use crate::ir::graph::Graph;
use crate::ir::node::NodeId;
use crate::opt::OptimizationPass;

// =============================================================================
// InstCombine Statistics
// =============================================================================

/// Statistics from instruction combining.
#[derive(Debug, Clone, Default)]
pub struct InstCombineStats {
    /// Patterns successfully applied.
    pub patterns_applied: usize,
    /// Instructions eliminated.
    pub instructions_eliminated: usize,
    /// Instructions simplified.
    pub instructions_simplified: usize,
    /// New instructions created.
    pub instructions_created: usize,
    /// Total instructions analyzed.
    pub instructions_analyzed: usize,
}

impl InstCombineStats {
    /// Net instruction reduction.
    pub fn net_reduction(&self) -> isize {
        self.instructions_eliminated as isize - self.instructions_created as isize
    }

    /// Merge statistics.
    pub fn merge(&mut self, other: &InstCombineStats) {
        self.patterns_applied += other.patterns_applied;
        self.instructions_eliminated += other.instructions_eliminated;
        self.instructions_simplified += other.instructions_simplified;
        self.instructions_created += other.instructions_created;
        self.instructions_analyzed += other.instructions_analyzed;
    }
}

// =============================================================================
// InstCombine Configuration
// =============================================================================

/// Configuration for instruction combining.
#[derive(Debug, Clone)]
pub struct InstCombineConfig {
    /// Maximum worklist iterations (prevent infinite loops).
    pub max_iterations: usize,
    /// Enable arithmetic patterns.
    pub enable_arithmetic: bool,
    /// Enable bitwise patterns.
    pub enable_bitwise: bool,
    /// Enable comparison patterns.
    pub enable_comparison: bool,
    /// Enable memory patterns.
    pub enable_memory: bool,
    /// Enable control flow patterns.
    pub enable_control: bool,
}

impl Default for InstCombineConfig {
    fn default() -> Self {
        Self {
            max_iterations: 10000,
            enable_arithmetic: true,
            enable_bitwise: true,
            enable_comparison: true,
            enable_memory: true,
            enable_control: true,
        }
    }
}

// =============================================================================
// InstCombine Pass
// =============================================================================

/// Instruction combining optimization pass.
#[derive(Debug)]
pub struct InstCombine {
    /// Configuration.
    config: InstCombineConfig,
    /// Statistics from last run.
    stats: InstCombineStats,
    /// Pattern registry.
    patterns: PatternRegistry,
}

impl InstCombine {
    /// Create a new instruction combine pass.
    pub fn new() -> Self {
        Self {
            config: InstCombineConfig::default(),
            stats: InstCombineStats::default(),
            patterns: PatternRegistry::new(),
        }
    }

    /// Create with custom configuration.
    pub fn with_config(config: InstCombineConfig) -> Self {
        Self {
            config,
            stats: InstCombineStats::default(),
            patterns: PatternRegistry::new(),
        }
    }

    /// Get statistics from last run.
    pub fn stats(&self) -> &InstCombineStats {
        &self.stats
    }

    /// Get the configuration.
    pub fn config(&self) -> &InstCombineConfig {
        &self.config
    }

    /// Run instruction combining.
    fn run_instcombine(&mut self, graph: &mut Graph) -> bool {
        self.stats = InstCombineStats::default();

        // Build initial worklist
        let mut worklist = Worklist::new();
        for i in 0..graph.len() {
            let id = NodeId::new(i as u32);
            if graph.get(id).is_some() {
                worklist.push(id);
            }
        }

        self.stats.instructions_analyzed = worklist.len();

        let mut changed = false;
        let mut iterations = 0;

        // Process worklist
        while let Some(node_id) = worklist.pop() {
            if iterations >= self.config.max_iterations {
                break;
            }
            iterations += 1;

            // Try to match patterns
            if let Some(matched) = self.try_patterns(graph, node_id) {
                // Apply the transformation
                if self.apply_pattern(graph, &matched, &mut worklist) {
                    changed = true;
                    self.stats.patterns_applied += 1;
                }
            }
        }

        changed
    }

    /// Try to match any pattern on a node.
    fn try_patterns(&self, graph: &Graph, node_id: NodeId) -> Option<PatternMatch> {
        let node = graph.get(node_id)?;

        // Try each enabled pattern category
        for pattern in self.patterns.iter() {
            // Check if pattern category is enabled
            if !self.is_pattern_enabled(pattern) {
                continue;
            }

            // Try to match
            if let Some(matched) = pattern.try_match(graph, node_id, &node.op) {
                return Some(matched);
            }
        }

        None
    }

    /// Check if a pattern category is enabled.
    fn is_pattern_enabled(&self, pattern: &Pattern) -> bool {
        match pattern.category() {
            PatternCategory::Arithmetic => self.config.enable_arithmetic,
            PatternCategory::Bitwise => self.config.enable_bitwise,
            PatternCategory::Comparison => self.config.enable_comparison,
            PatternCategory::Memory => self.config.enable_memory,
            PatternCategory::Control => self.config.enable_control,
        }
    }

    /// Apply a pattern match to the graph.
    fn apply_pattern(
        &mut self,
        graph: &mut Graph,
        matched: &PatternMatch,
        worklist: &mut Worklist,
    ) -> bool {
        // Apply the transformation
        matched.apply(graph);

        // Add affected nodes to worklist
        if let Some(replacement) = matched.replacement() {
            // Add uses of the replacement
            for &use_id in graph.uses(replacement) {
                worklist.push(use_id);
            }
            self.stats.instructions_simplified += 1;
        }

        if matched.eliminated() {
            self.stats.instructions_eliminated += 1;
        }

        if matched.created_new() {
            self.stats.instructions_created += 1;
        }

        true
    }
}

impl Default for InstCombine {
    fn default() -> Self {
        Self::new()
    }
}

impl OptimizationPass for InstCombine {
    fn name(&self) -> &'static str {
        "instcombine"
    }

    fn run(&mut self, graph: &mut Graph) -> bool {
        self.run_instcombine(graph)
    }
}

// =============================================================================
// Pattern Category
// =============================================================================

/// Categories of patterns.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PatternCategory {
    /// Arithmetic patterns (add, sub, mul, div).
    Arithmetic,
    /// Bitwise patterns (and, or, xor, shifts).
    Bitwise,
    /// Comparison patterns.
    Comparison,
    /// Memory patterns.
    Memory,
    /// Control flow patterns.
    Control,
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // InstCombineStats Tests
    // =========================================================================

    #[test]
    fn test_stats_default() {
        let stats = InstCombineStats::default();
        assert_eq!(stats.patterns_applied, 0);
        assert_eq!(stats.instructions_eliminated, 0);
    }

    #[test]
    fn test_stats_net_reduction_positive() {
        let stats = InstCombineStats {
            patterns_applied: 1,
            instructions_eliminated: 5,
            instructions_simplified: 2,
            instructions_created: 2,
            instructions_analyzed: 10,
        };
        assert_eq!(stats.net_reduction(), 3);
    }

    #[test]
    fn test_stats_net_reduction_negative() {
        let stats = InstCombineStats {
            patterns_applied: 1,
            instructions_eliminated: 1,
            instructions_simplified: 0,
            instructions_created: 3,
            instructions_analyzed: 10,
        };
        assert_eq!(stats.net_reduction(), -2);
    }

    #[test]
    fn test_stats_merge() {
        let mut stats1 = InstCombineStats {
            patterns_applied: 3,
            instructions_eliminated: 5,
            instructions_simplified: 2,
            instructions_created: 1,
            instructions_analyzed: 20,
        };
        let stats2 = InstCombineStats {
            patterns_applied: 2,
            instructions_eliminated: 3,
            instructions_simplified: 1,
            instructions_created: 1,
            instructions_analyzed: 10,
        };
        stats1.merge(&stats2);
        assert_eq!(stats1.patterns_applied, 5);
        assert_eq!(stats1.instructions_eliminated, 8);
        assert_eq!(stats1.instructions_analyzed, 30);
    }

    // =========================================================================
    // InstCombineConfig Tests
    // =========================================================================

    #[test]
    fn test_config_default() {
        let config = InstCombineConfig::default();
        assert_eq!(config.max_iterations, 10000);
        assert!(config.enable_arithmetic);
        assert!(config.enable_bitwise);
        assert!(config.enable_comparison);
        assert!(config.enable_memory);
        assert!(config.enable_control);
    }

    #[test]
    fn test_config_custom() {
        let config = InstCombineConfig {
            max_iterations: 100,
            enable_arithmetic: false,
            enable_bitwise: true,
            enable_comparison: true,
            enable_memory: false,
            enable_control: false,
        };
        assert_eq!(config.max_iterations, 100);
        assert!(!config.enable_arithmetic);
    }

    // =========================================================================
    // InstCombine Pass Tests
    // =========================================================================

    #[test]
    fn test_instcombine_new() {
        let ic = InstCombine::new();
        assert_eq!(ic.stats().patterns_applied, 0);
    }

    #[test]
    fn test_instcombine_with_config() {
        let config = InstCombineConfig {
            max_iterations: 500,
            ..Default::default()
        };
        let ic = InstCombine::with_config(config);
        assert_eq!(ic.config().max_iterations, 500);
    }

    #[test]
    fn test_instcombine_name() {
        let ic = InstCombine::new();
        assert_eq!(ic.name(), "instcombine");
    }

    #[test]
    fn test_instcombine_default() {
        let ic = InstCombine::default();
        assert!(ic.config().enable_arithmetic);
    }

    #[test]
    fn test_instcombine_empty_graph() {
        let mut graph = Graph::new();
        let mut ic = InstCombine::new();
        let changed = ic.run(&mut graph);
        assert!(!changed);
    }

    #[test]
    fn test_instcombine_simple_graph() {
        let mut graph = Graph::new();
        graph.const_int(42);

        let mut ic = InstCombine::new();
        ic.run(&mut graph);

        assert!(ic.stats().instructions_analyzed >= 1);
    }

    // =========================================================================
    // PatternCategory Tests
    // =========================================================================

    #[test]
    fn test_pattern_category_equality() {
        assert_eq!(PatternCategory::Arithmetic, PatternCategory::Arithmetic);
        assert_ne!(PatternCategory::Arithmetic, PatternCategory::Bitwise);
    }

    #[test]
    fn test_pattern_category_debug() {
        let cat = PatternCategory::Bitwise;
        assert!(format!("{:?}", cat).contains("Bitwise"));
    }

    #[test]
    fn test_is_pattern_enabled() {
        let mut config = InstCombineConfig::default();
        config.enable_arithmetic = false;

        let ic = InstCombine::with_config(config);

        // Create a dummy arithmetic pattern
        let pattern = Pattern::new(PatternCategory::Arithmetic);
        assert!(!ic.is_pattern_enabled(&pattern));

        let pattern = Pattern::new(PatternCategory::Bitwise);
        assert!(ic.is_pattern_enabled(&pattern));
    }
}
