//! Bitwise patterns for instruction combining.
//!
//! Patterns include:
//! - x & 0 -> 0
//! - x & -1 -> x
//! - x | 0 -> x
//! - x | -1 -> -1
//! - x ^ 0 -> x
//! - x & x -> x
//! - x | x -> x

use crate::ir::graph::Graph;
use crate::ir::node::NodeId;
use crate::ir::operators::{BitwiseOp, Operator};

use super::PatternMatch;

// =============================================================================
// Bitwise Patterns
// =============================================================================

/// Bitwise pattern matcher.
pub struct BitwisePatterns;

impl BitwisePatterns {
    /// Try to match a bitwise pattern.
    pub fn try_match(graph: &Graph, node: NodeId, op: &Operator) -> Option<PatternMatch> {
        match op {
            Operator::Bitwise(bw_op) => Self::try_bitwise(graph, node, *bw_op),
            _ => None,
        }
    }

    /// Try to match bitwise operation patterns.
    fn try_bitwise(graph: &Graph, node: NodeId, op: BitwiseOp) -> Option<PatternMatch> {
        let n = graph.get(node)?;

        // Use get() since as_slice() only works for Many variant
        let lhs = n.inputs.get(0)?;
        let rhs = n.inputs.get(1)?;

        match op {
            BitwiseOp::And => Self::try_and_patterns(graph, node, lhs, rhs),
            BitwiseOp::Or => Self::try_or_patterns(graph, node, lhs, rhs),
            BitwiseOp::Xor => Self::try_xor_patterns(graph, node, lhs, rhs),
            BitwiseOp::Shl | BitwiseOp::Shr => Self::try_shift_patterns(graph, node, lhs, rhs),
            _ => None,
        }
    }

    /// AND patterns: x & 0 -> 0, x & -1 -> x, x & x -> x
    fn try_and_patterns(
        graph: &Graph,
        node: NodeId,
        lhs: NodeId,
        rhs: NodeId,
    ) -> Option<PatternMatch> {
        if Self::is_zero(graph, rhs) {
            return Some(PatternMatch::replace(node, rhs, "and_zero_right"));
        }
        if Self::is_zero(graph, lhs) {
            return Some(PatternMatch::replace(node, lhs, "and_zero_left"));
        }
        if Self::is_all_ones(graph, rhs) {
            return Some(PatternMatch::replace(node, lhs, "and_all_ones_right"));
        }
        if Self::is_all_ones(graph, lhs) {
            return Some(PatternMatch::replace(node, rhs, "and_all_ones_left"));
        }
        if lhs == rhs {
            return Some(PatternMatch::replace(node, lhs, "and_self"));
        }
        None
    }

    /// OR patterns: x | 0 -> x, x | -1 -> -1, x | x -> x
    fn try_or_patterns(
        graph: &Graph,
        node: NodeId,
        lhs: NodeId,
        rhs: NodeId,
    ) -> Option<PatternMatch> {
        if Self::is_zero(graph, rhs) {
            return Some(PatternMatch::replace(node, lhs, "or_zero_right"));
        }
        if Self::is_zero(graph, lhs) {
            return Some(PatternMatch::replace(node, rhs, "or_zero_left"));
        }
        if Self::is_all_ones(graph, rhs) {
            return Some(PatternMatch::replace(node, rhs, "or_all_ones_right"));
        }
        if Self::is_all_ones(graph, lhs) {
            return Some(PatternMatch::replace(node, lhs, "or_all_ones_left"));
        }
        if lhs == rhs {
            return Some(PatternMatch::replace(node, lhs, "or_self"));
        }
        None
    }

    /// XOR patterns: x ^ 0 -> x
    fn try_xor_patterns(
        graph: &Graph,
        node: NodeId,
        lhs: NodeId,
        rhs: NodeId,
    ) -> Option<PatternMatch> {
        if Self::is_zero(graph, rhs) {
            return Some(PatternMatch::replace(node, lhs, "xor_zero_right"));
        }
        if Self::is_zero(graph, lhs) {
            return Some(PatternMatch::replace(node, rhs, "xor_zero_left"));
        }
        None
    }

    /// Shift patterns: x << 0 -> x, x >> 0 -> x
    fn try_shift_patterns(
        graph: &Graph,
        node: NodeId,
        lhs: NodeId,
        rhs: NodeId,
    ) -> Option<PatternMatch> {
        if Self::is_zero(graph, rhs) {
            return Some(PatternMatch::replace(node, lhs, "shift_zero"));
        }
        if Self::is_zero(graph, lhs) {
            return Some(PatternMatch::replace(node, lhs, "zero_shift"));
        }
        None
    }

    /// Check if a node is a zero constant.
    fn is_zero(graph: &Graph, node: NodeId) -> bool {
        if let Some(n) = graph.get(node) {
            matches!(n.op, Operator::ConstInt(0))
        } else {
            false
        }
    }

    /// Check if a node is all ones (-1 for signed).
    fn is_all_ones(graph: &Graph, node: NodeId) -> bool {
        if let Some(n) = graph.get(node) {
            matches!(n.op, Operator::ConstInt(-1))
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
    fn test_and_zero() {
        let mut graph = Graph::new();
        let x = graph.const_int(42);
        let zero = graph.const_int(0);
        let and = graph.add_node(Operator::Bitwise(BitwiseOp::And), InputList::Pair(x, zero));

        let m = BitwisePatterns::try_match(&graph, and, &Operator::Bitwise(BitwiseOp::And));
        assert!(m.is_some());
        assert_eq!(m.unwrap().replacement(), Some(zero));
    }

    #[test]
    fn test_and_all_ones() {
        let mut graph = Graph::new();
        let x = graph.const_int(42);
        let all_ones = graph.const_int(-1);
        let and = graph.add_node(
            Operator::Bitwise(BitwiseOp::And),
            InputList::Pair(x, all_ones),
        );

        let m = BitwisePatterns::try_match(&graph, and, &Operator::Bitwise(BitwiseOp::And));
        assert!(m.is_some());
        assert_eq!(m.unwrap().replacement(), Some(x));
    }

    #[test]
    fn test_and_self() {
        let mut graph = Graph::new();
        let x = graph.const_int(42);
        let and = graph.add_node(Operator::Bitwise(BitwiseOp::And), InputList::Pair(x, x));

        let m = BitwisePatterns::try_match(&graph, and, &Operator::Bitwise(BitwiseOp::And));
        assert!(m.is_some());
        assert_eq!(m.unwrap().pattern_name(), "and_self");
    }

    #[test]
    fn test_or_zero() {
        let mut graph = Graph::new();
        let x = graph.const_int(42);
        let zero = graph.const_int(0);
        let or = graph.add_node(Operator::Bitwise(BitwiseOp::Or), InputList::Pair(x, zero));

        let m = BitwisePatterns::try_match(&graph, or, &Operator::Bitwise(BitwiseOp::Or));
        assert!(m.is_some());
        assert_eq!(m.unwrap().replacement(), Some(x));
    }

    #[test]
    fn test_or_all_ones() {
        let mut graph = Graph::new();
        let x = graph.const_int(42);
        let all_ones = graph.const_int(-1);
        let or = graph.add_node(
            Operator::Bitwise(BitwiseOp::Or),
            InputList::Pair(x, all_ones),
        );

        let m = BitwisePatterns::try_match(&graph, or, &Operator::Bitwise(BitwiseOp::Or));
        assert!(m.is_some());
        assert_eq!(m.unwrap().replacement(), Some(all_ones));
    }

    #[test]
    fn test_or_self() {
        let mut graph = Graph::new();
        let x = graph.const_int(42);
        let or = graph.add_node(Operator::Bitwise(BitwiseOp::Or), InputList::Pair(x, x));

        let m = BitwisePatterns::try_match(&graph, or, &Operator::Bitwise(BitwiseOp::Or));
        assert!(m.is_some());
        assert_eq!(m.unwrap().pattern_name(), "or_self");
    }

    #[test]
    fn test_xor_zero() {
        let mut graph = Graph::new();
        let x = graph.const_int(42);
        let zero = graph.const_int(0);
        let xor = graph.add_node(Operator::Bitwise(BitwiseOp::Xor), InputList::Pair(x, zero));

        let m = BitwisePatterns::try_match(&graph, xor, &Operator::Bitwise(BitwiseOp::Xor));
        assert!(m.is_some());
        assert_eq!(m.unwrap().replacement(), Some(x));
    }

    #[test]
    fn test_shift_zero() {
        let mut graph = Graph::new();
        let x = graph.const_int(42);
        let zero = graph.const_int(0);
        let shl = graph.add_node(Operator::Bitwise(BitwiseOp::Shl), InputList::Pair(x, zero));

        let m = BitwisePatterns::try_match(&graph, shl, &Operator::Bitwise(BitwiseOp::Shl));
        assert!(m.is_some());
        assert_eq!(m.unwrap().replacement(), Some(x));
    }

    #[test]
    fn test_zero_shift() {
        let mut graph = Graph::new();
        let zero = graph.const_int(0);
        let x = graph.const_int(5);
        let shl = graph.add_node(Operator::Bitwise(BitwiseOp::Shl), InputList::Pair(zero, x));

        let m = BitwisePatterns::try_match(&graph, shl, &Operator::Bitwise(BitwiseOp::Shl));
        assert!(m.is_some());
        assert_eq!(m.unwrap().replacement(), Some(zero));
    }

    #[test]
    fn test_is_zero() {
        let mut graph = Graph::new();
        let zero = graph.const_int(0);
        let one = graph.const_int(1);

        assert!(BitwisePatterns::is_zero(&graph, zero));
        assert!(!BitwisePatterns::is_zero(&graph, one));
    }

    #[test]
    fn test_is_all_ones() {
        let mut graph = Graph::new();
        let all_ones = graph.const_int(-1);
        let zero = graph.const_int(0);

        assert!(BitwisePatterns::is_all_ones(&graph, all_ones));
        assert!(!BitwisePatterns::is_all_ones(&graph, zero));
    }
}
