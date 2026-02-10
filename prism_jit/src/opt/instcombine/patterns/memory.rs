//! Memory patterns for instruction combining.
//!
//! Patterns include:
//! - Load after store to same location -> use stored value
//! - Store after store to same location -> eliminate first store
//! - Load of constant -> fold to constant
//!
//! Note: These patterns require sophisticated alias analysis for full
//! implementation. This module provides the infrastructure for such
//! patterns but the actual matching is conservative.

use crate::ir::graph::Graph;
use crate::ir::node::NodeId;
use crate::ir::operators::{MemoryOp, Operator};

use super::PatternMatch;

// =============================================================================
// Memory Patterns
// =============================================================================

/// Memory pattern matcher.
pub struct MemoryPatterns;

impl MemoryPatterns {
    /// Try to match a memory pattern.
    pub fn try_match(graph: &Graph, node: NodeId, op: &Operator) -> Option<PatternMatch> {
        match op {
            Operator::Memory(mem_op) => Self::try_memory(graph, node, *mem_op),
            _ => None,
        }
    }

    /// Try to match memory operation patterns.
    fn try_memory(_graph: &Graph, _node: NodeId, _op: MemoryOp) -> Option<PatternMatch> {
        // Memory patterns require alias analysis which is done
        // separately in the DSE pass. InstCombine handles simpler
        // algebraic patterns; memory optimization is deferred to DSE.
        None
    }

    /// Check if a node is a fresh allocation.
    #[allow(dead_code)]
    fn is_fresh_alloc(graph: &Graph, node: NodeId) -> bool {
        if let Some(n) = graph.get(node) {
            matches!(
                n.op,
                Operator::Memory(MemoryOp::Alloc) | Operator::Memory(MemoryOp::AllocArray)
            )
        } else {
            false
        }
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::node::InputList;

    #[test]
    fn test_is_fresh_alloc() {
        let mut graph = Graph::new();
        let alloc = graph.add_node(Operator::Memory(MemoryOp::Alloc), InputList::Empty);

        assert!(MemoryPatterns::is_fresh_alloc(&graph, alloc));
    }

    #[test]
    fn test_is_fresh_alloc_array() {
        let mut graph = Graph::new();
        let alloc = graph.add_node(Operator::Memory(MemoryOp::AllocArray), InputList::Empty);

        assert!(MemoryPatterns::is_fresh_alloc(&graph, alloc));
    }

    #[test]
    fn test_is_not_fresh_alloc() {
        let mut graph = Graph::new();
        let c = graph.const_int(42);

        assert!(!MemoryPatterns::is_fresh_alloc(&graph, c));
    }

    #[test]
    fn test_memory_patterns_defer_to_dse() {
        // Memory patterns are handled by DSE, not InstCombine
        let mut graph = Graph::new();
        let alloc = graph.add_node(Operator::Memory(MemoryOp::Alloc), InputList::Empty);
        let val = graph.const_int(42);
        let store = graph.add_node(
            Operator::Memory(MemoryOp::StoreField),
            InputList::Pair(alloc, val),
        );

        // InstCombine defers memory patterns to DSE
        let m = MemoryPatterns::try_match(&graph, store, &Operator::Memory(MemoryOp::StoreField));
        assert!(m.is_none());
    }

    #[test]
    fn test_load_no_match() {
        let mut graph = Graph::new();
        let alloc = graph.add_node(Operator::Memory(MemoryOp::Alloc), InputList::Empty);
        let load = graph.add_node(
            Operator::Memory(MemoryOp::LoadField),
            InputList::Single(alloc),
        );

        let m = MemoryPatterns::try_match(&graph, load, &Operator::Memory(MemoryOp::LoadField));
        assert!(m.is_none());
    }
}
