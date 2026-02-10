//! Pattern matching for instruction combining.
//!
//! This module provides the pattern matching infrastructure:
//! - Pattern trait for custom patterns
//! - PatternMatch for representing matched patterns
//! - PatternRegistry for managing available patterns

mod arithmetic;
mod bitwise;
mod comparison;
mod control;
mod memory;

pub use arithmetic::ArithmeticPatterns;
pub use bitwise::BitwisePatterns;
pub use comparison::ComparisonPatterns;
pub use control::ControlPatterns;
pub use memory::MemoryPatterns;

use crate::ir::graph::Graph;
use crate::ir::node::NodeId;
use crate::ir::operators::Operator;

use super::PatternCategory;

// =============================================================================
// Pattern Match
// =============================================================================

/// A matched pattern transformation.
#[derive(Debug)]
pub struct PatternMatch {
    /// The node being transformed.
    target: NodeId,
    /// Replacement node (if any).
    replacement: Option<NodeId>,
    /// Pattern that matched.
    pattern_name: &'static str,
    /// Whether the original was eliminated.
    eliminated: bool,
    /// Whether a new node was created.
    created_new: bool,
}

impl PatternMatch {
    /// Create a new pattern match.
    pub fn new(target: NodeId, pattern_name: &'static str) -> Self {
        Self {
            target,
            replacement: None,
            pattern_name,
            eliminated: false,
            created_new: false,
        }
    }

    /// Create a replacement match.
    pub fn replace(target: NodeId, replacement: NodeId, pattern_name: &'static str) -> Self {
        Self {
            target,
            replacement: Some(replacement),
            pattern_name,
            eliminated: true,
            created_new: false,
        }
    }

    /// Get the target node.
    pub fn target(&self) -> NodeId {
        self.target
    }

    /// Get the replacement node.
    pub fn replacement(&self) -> Option<NodeId> {
        self.replacement
    }

    /// Get the pattern name.
    pub fn pattern_name(&self) -> &'static str {
        self.pattern_name
    }

    /// Check if original was eliminated.
    pub fn eliminated(&self) -> bool {
        self.eliminated
    }

    /// Check if a new node was created.
    pub fn created_new(&self) -> bool {
        self.created_new
    }

    /// Apply this pattern match to the graph.
    /// Returns the replacement node if this was a replacement operation.
    pub fn apply(&self, _graph: &mut Graph) -> Option<NodeId> {
        // Note: Actual replacement is done by the InstCombine pass
        // which has full access to graph mutation APIs.
        self.replacement
    }
}

// =============================================================================
// Pattern
// =============================================================================

/// A single pattern for instruction combining.
#[derive(Debug)]
pub struct Pattern {
    /// Category of this pattern.
    category: PatternCategory,
    /// Name of this pattern.
    name: &'static str,
}

impl Pattern {
    /// Create a new pattern.
    pub fn new(category: PatternCategory) -> Self {
        Self {
            category,
            name: "unknown",
        }
    }

    /// Create a named pattern.
    pub fn named(category: PatternCategory, name: &'static str) -> Self {
        Self { category, name }
    }

    /// Get the pattern category.
    pub fn category(&self) -> PatternCategory {
        self.category
    }

    /// Get the pattern name.
    #[allow(dead_code)]
    pub fn name(&self) -> &'static str {
        self.name
    }

    /// Try to match this pattern on a node.
    pub fn try_match(&self, graph: &Graph, node: NodeId, op: &Operator) -> Option<PatternMatch> {
        match self.category {
            PatternCategory::Arithmetic => ArithmeticPatterns::try_match(graph, node, op),
            PatternCategory::Bitwise => BitwisePatterns::try_match(graph, node, op),
            PatternCategory::Comparison => ComparisonPatterns::try_match(graph, node, op),
            PatternCategory::Memory => MemoryPatterns::try_match(graph, node, op),
            PatternCategory::Control => ControlPatterns::try_match(graph, node, op),
        }
    }
}

// =============================================================================
// Pattern Registry
// =============================================================================

/// Registry of available patterns.
#[derive(Debug)]
pub struct PatternRegistry {
    /// All registered patterns.
    patterns: Vec<Pattern>,
}

impl PatternRegistry {
    /// Create a new registry with default patterns.
    pub fn new() -> Self {
        let mut patterns = Vec::new();

        // Register all pattern categories
        patterns.push(Pattern::named(PatternCategory::Arithmetic, "arithmetic"));
        patterns.push(Pattern::named(PatternCategory::Bitwise, "bitwise"));
        patterns.push(Pattern::named(PatternCategory::Comparison, "comparison"));
        patterns.push(Pattern::named(PatternCategory::Memory, "memory"));
        patterns.push(Pattern::named(PatternCategory::Control, "control"));

        Self { patterns }
    }

    /// Iterate over patterns.
    pub fn iter(&self) -> impl Iterator<Item = &Pattern> {
        self.patterns.iter()
    }

    /// Get the number of patterns.
    #[allow(dead_code)]
    pub fn len(&self) -> usize {
        self.patterns.len()
    }

    /// Check if registry is empty.
    #[allow(dead_code)]
    pub fn is_empty(&self) -> bool {
        self.patterns.is_empty()
    }
}

impl Default for PatternRegistry {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // PatternMatch Tests
    // =========================================================================

    #[test]
    fn test_pattern_match_new() {
        let m = PatternMatch::new(NodeId::new(1), "test");
        assert_eq!(m.target(), NodeId::new(1));
        assert_eq!(m.replacement(), None);
        assert_eq!(m.pattern_name(), "test");
        assert!(!m.eliminated());
    }

    #[test]
    fn test_pattern_match_replace() {
        let m = PatternMatch::replace(NodeId::new(1), NodeId::new(2), "test");
        assert_eq!(m.target(), NodeId::new(1));
        assert_eq!(m.replacement(), Some(NodeId::new(2)));
        assert!(m.eliminated());
    }

    #[test]
    fn test_pattern_match_created_new() {
        let mut m = PatternMatch::new(NodeId::new(1), "test");
        assert!(!m.created_new());
        m.created_new = true;
        assert!(m.created_new());
    }

    // =========================================================================
    // Pattern Tests
    // =========================================================================

    #[test]
    fn test_pattern_new() {
        let p = Pattern::new(PatternCategory::Arithmetic);
        assert_eq!(p.category(), PatternCategory::Arithmetic);
        assert_eq!(p.name(), "unknown");
    }

    #[test]
    fn test_pattern_named() {
        let p = Pattern::named(PatternCategory::Bitwise, "and_zero");
        assert_eq!(p.category(), PatternCategory::Bitwise);
        assert_eq!(p.name(), "and_zero");
    }

    // =========================================================================
    // PatternRegistry Tests
    // =========================================================================

    #[test]
    fn test_registry_new() {
        let reg = PatternRegistry::new();
        assert!(!reg.is_empty());
        assert_eq!(reg.len(), 5);
    }

    #[test]
    fn test_registry_default() {
        let reg = PatternRegistry::default();
        assert!(!reg.is_empty());
    }

    #[test]
    fn test_registry_iter() {
        let reg = PatternRegistry::new();
        let count = reg.iter().count();
        assert_eq!(count, 5);
    }

    #[test]
    fn test_registry_categories() {
        let reg = PatternRegistry::new();
        let categories: Vec<_> = reg.iter().map(|p| p.category()).collect();

        assert!(categories.contains(&PatternCategory::Arithmetic));
        assert!(categories.contains(&PatternCategory::Bitwise));
        assert!(categories.contains(&PatternCategory::Comparison));
        assert!(categories.contains(&PatternCategory::Memory));
        assert!(categories.contains(&PatternCategory::Control));
    }
}
