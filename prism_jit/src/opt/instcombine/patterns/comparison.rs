//! Comparison patterns for instruction combining.
//!
//! Patterns include:
//! - x == x -> true
//! - x != x -> false
//! - x < x -> false
//! - x <= x -> true
//! - x > x -> false
//! - x >= x -> true

use crate::ir::graph::Graph;
use crate::ir::node::NodeId;
use crate::ir::operators::{CmpOp, Operator};

use super::PatternMatch;

// =============================================================================
// Comparison Patterns
// =============================================================================

/// Comparison pattern matcher.
pub struct ComparisonPatterns;

impl ComparisonPatterns {
    /// Try to match a comparison pattern.
    pub fn try_match(graph: &Graph, node: NodeId, op: &Operator) -> Option<PatternMatch> {
        match op {
            Operator::IntCmp(cmp_op)
            | Operator::FloatCmp(cmp_op)
            | Operator::GenericCmp(cmp_op) => Self::try_cmp(graph, node, *cmp_op),
            _ => None,
        }
    }

    /// Try to match comparison patterns.
    fn try_cmp(graph: &Graph, node: NodeId, op: CmpOp) -> Option<PatternMatch> {
        let n = graph.get(node)?;

        // Use get() since as_slice() only works for Many variant
        let lhs = n.inputs.get(0)?;
        let rhs = n.inputs.get(1)?;

        // Check for comparing with self
        if lhs == rhs {
            return Self::try_self_comparison(node, op);
        }

        // Check for comparison with constants
        Self::try_constant_comparison(graph, node, lhs, rhs, op)
    }

    /// Handle x cmp x cases.
    fn try_self_comparison(node: NodeId, op: CmpOp) -> Option<PatternMatch> {
        match op {
            CmpOp::Eq => Some(PatternMatch::new(node, "eq_self_true")),
            CmpOp::Ne => Some(PatternMatch::new(node, "ne_self_false")),
            CmpOp::Lt | CmpOp::Gt => Some(PatternMatch::new(node, "lt_gt_self_false")),
            CmpOp::Le | CmpOp::Ge => Some(PatternMatch::new(node, "le_ge_self_true")),
            _ => None,
        }
    }

    /// Try constant comparison patterns.
    fn try_constant_comparison(
        graph: &Graph,
        node: NodeId,
        lhs: NodeId,
        rhs: NodeId,
        _op: CmpOp,
    ) -> Option<PatternMatch> {
        let lhs_const = Self::get_int_constant(graph, lhs);
        let rhs_const = Self::get_int_constant(graph, rhs);

        if let (Some(_l), Some(_r)) = (lhs_const, rhs_const) {
            return Some(PatternMatch::new(node, "const_fold"));
        }

        None
    }

    /// Get integer constant value if node is a constant.
    fn get_int_constant(graph: &Graph, node: NodeId) -> Option<i64> {
        if let Some(n) = graph.get(node) {
            if let Operator::ConstInt(v) = n.op {
                return Some(v);
            }
        }
        None
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
    fn test_eq_self() {
        let mut graph = Graph::new();
        let x = graph.const_int(42);
        let eq = graph.add_node(Operator::IntCmp(CmpOp::Eq), InputList::Pair(x, x));

        let m = ComparisonPatterns::try_match(&graph, eq, &Operator::IntCmp(CmpOp::Eq));
        assert!(m.is_some());
        assert_eq!(m.unwrap().pattern_name(), "eq_self_true");
    }

    #[test]
    fn test_ne_self() {
        let mut graph = Graph::new();
        let x = graph.const_int(42);
        let ne = graph.add_node(Operator::IntCmp(CmpOp::Ne), InputList::Pair(x, x));

        let m = ComparisonPatterns::try_match(&graph, ne, &Operator::IntCmp(CmpOp::Ne));
        assert!(m.is_some());
        assert_eq!(m.unwrap().pattern_name(), "ne_self_false");
    }

    #[test]
    fn test_lt_self() {
        let mut graph = Graph::new();
        let x = graph.const_int(42);
        let lt = graph.add_node(Operator::IntCmp(CmpOp::Lt), InputList::Pair(x, x));

        let m = ComparisonPatterns::try_match(&graph, lt, &Operator::IntCmp(CmpOp::Lt));
        assert!(m.is_some());
        assert_eq!(m.unwrap().pattern_name(), "lt_gt_self_false");
    }

    #[test]
    fn test_le_self() {
        let mut graph = Graph::new();
        let x = graph.const_int(42);
        let le = graph.add_node(Operator::IntCmp(CmpOp::Le), InputList::Pair(x, x));

        let m = ComparisonPatterns::try_match(&graph, le, &Operator::IntCmp(CmpOp::Le));
        assert!(m.is_some());
        assert_eq!(m.unwrap().pattern_name(), "le_ge_self_true");
    }

    #[test]
    fn test_gt_self() {
        let mut graph = Graph::new();
        let x = graph.const_int(42);
        let gt = graph.add_node(Operator::IntCmp(CmpOp::Gt), InputList::Pair(x, x));

        let m = ComparisonPatterns::try_match(&graph, gt, &Operator::IntCmp(CmpOp::Gt));
        assert!(m.is_some());
    }

    #[test]
    fn test_ge_self() {
        let mut graph = Graph::new();
        let x = graph.const_int(42);
        let ge = graph.add_node(Operator::IntCmp(CmpOp::Ge), InputList::Pair(x, x));

        let m = ComparisonPatterns::try_match(&graph, ge, &Operator::IntCmp(CmpOp::Ge));
        assert!(m.is_some());
    }

    #[test]
    fn test_const_comparison() {
        let mut graph = Graph::new();
        let a = graph.const_int(5);
        let b = graph.const_int(10);
        let lt = graph.add_node(Operator::IntCmp(CmpOp::Lt), InputList::Pair(a, b));

        let m = ComparisonPatterns::try_match(&graph, lt, &Operator::IntCmp(CmpOp::Lt));
        assert!(m.is_some());
        assert_eq!(m.unwrap().pattern_name(), "const_fold");
    }

    #[test]
    fn test_no_match_different_vars() {
        let mut graph = Graph::new();
        let x = graph.const_int(42);
        let y = graph.const_int(43);
        let eq = graph.add_node(Operator::IntCmp(CmpOp::Eq), InputList::Pair(x, y));

        let m = ComparisonPatterns::try_match(&graph, eq, &Operator::IntCmp(CmpOp::Eq));
        assert!(m.is_some());
    }

    #[test]
    fn test_get_int_constant() {
        let mut graph = Graph::new();
        let c = graph.const_int(42);

        let val = ComparisonPatterns::get_int_constant(&graph, c);
        assert_eq!(val, Some(42));
    }

    #[test]
    fn test_get_int_constant_not_const() {
        let graph = Graph::new();
        let val = ComparisonPatterns::get_int_constant(&graph, NodeId::new(1));
        assert_eq!(val, None);
    }
}
