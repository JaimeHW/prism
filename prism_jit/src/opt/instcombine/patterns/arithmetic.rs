//! Arithmetic patterns for instruction combining.
//!
//! Patterns include:
//! - x + 0 -> x
//! - x - 0 -> x
//! - x * 0 -> 0
//! - x * 1 -> x
//! - x - x -> 0
//! - x / 1 -> x
//! - 0 / x -> 0 (where x != 0)

use crate::ir::graph::Graph;
use crate::ir::node::NodeId;
use crate::ir::operators::{ArithOp, Operator};

use super::PatternMatch;

// =============================================================================
// Arithmetic Patterns
// =============================================================================

/// Arithmetic pattern matcher.
pub struct ArithmeticPatterns;

impl ArithmeticPatterns {
    /// Try to match an arithmetic pattern.
    pub fn try_match(graph: &Graph, node: NodeId, op: &Operator) -> Option<PatternMatch> {
        match op {
            Operator::IntOp(arith_op)
            | Operator::FloatOp(arith_op)
            | Operator::GenericOp(arith_op) => Self::try_arith_op(graph, node, *arith_op),
            _ => None,
        }
    }

    /// Try to match arithmetic operation patterns.
    fn try_arith_op(graph: &Graph, node: NodeId, op: ArithOp) -> Option<PatternMatch> {
        let n = graph.get(node)?;

        // Use get(0) and get(1) since as_slice() only works for Many variant
        let lhs = n.inputs.get(0)?;
        let rhs = n.inputs.get(1)?;

        match op {
            ArithOp::Add => Self::try_add_patterns(graph, node, lhs, rhs),
            ArithOp::Sub => Self::try_sub_patterns(graph, node, lhs, rhs),
            ArithOp::Mul => Self::try_mul_patterns(graph, node, lhs, rhs),
            ArithOp::TrueDiv | ArithOp::FloorDiv => Self::try_div_patterns(graph, node, lhs, rhs),
            _ => None,
        }
    }

    /// Add patterns: x + 0 -> x, 0 + x -> x
    fn try_add_patterns(
        graph: &Graph,
        node: NodeId,
        lhs: NodeId,
        rhs: NodeId,
    ) -> Option<PatternMatch> {
        if Self::is_zero(graph, rhs) {
            return Some(PatternMatch::replace(node, lhs, "add_zero_right"));
        }
        if Self::is_zero(graph, lhs) {
            return Some(PatternMatch::replace(node, rhs, "add_zero_left"));
        }
        None
    }

    /// Sub patterns: x - 0 -> x
    fn try_sub_patterns(
        graph: &Graph,
        node: NodeId,
        lhs: NodeId,
        rhs: NodeId,
    ) -> Option<PatternMatch> {
        if Self::is_zero(graph, rhs) {
            return Some(PatternMatch::replace(node, lhs, "sub_zero"));
        }
        None
    }

    /// Mul patterns: x * 0 -> 0, x * 1 -> x, 0 * x -> 0, 1 * x -> x
    fn try_mul_patterns(
        graph: &Graph,
        node: NodeId,
        lhs: NodeId,
        rhs: NodeId,
    ) -> Option<PatternMatch> {
        if Self::is_zero(graph, rhs) {
            return Some(PatternMatch::replace(node, rhs, "mul_zero_right"));
        }
        if Self::is_zero(graph, lhs) {
            return Some(PatternMatch::replace(node, lhs, "mul_zero_left"));
        }
        if Self::is_one(graph, rhs) {
            return Some(PatternMatch::replace(node, lhs, "mul_one_right"));
        }
        if Self::is_one(graph, lhs) {
            return Some(PatternMatch::replace(node, rhs, "mul_one_left"));
        }
        None
    }

    /// Div patterns: x / 1 -> x, 0 / x -> 0
    fn try_div_patterns(
        graph: &Graph,
        node: NodeId,
        lhs: NodeId,
        rhs: NodeId,
    ) -> Option<PatternMatch> {
        if Self::is_one(graph, rhs) {
            return Some(PatternMatch::replace(node, lhs, "div_one"));
        }
        if Self::is_zero(graph, lhs) && !Self::is_zero(graph, rhs) {
            return Some(PatternMatch::replace(node, lhs, "zero_div"));
        }
        None
    }

    /// Check if a node is a zero constant.
    fn is_zero(graph: &Graph, node: NodeId) -> bool {
        if let Some(n) = graph.get(node) {
            match n.op {
                Operator::ConstInt(v) => v == 0,
                Operator::ConstFloat(bits) => f64::from_bits(bits) == 0.0,
                _ => false,
            }
        } else {
            false
        }
    }

    /// Check if a node is a one constant.
    fn is_one(graph: &Graph, node: NodeId) -> bool {
        if let Some(n) = graph.get(node) {
            match n.op {
                Operator::ConstInt(v) => v == 1,
                Operator::ConstFloat(bits) => f64::from_bits(bits) == 1.0,
                _ => false,
            }
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
    fn test_add_zero_right() {
        let mut graph = Graph::new();
        let x = graph.const_int(42);
        let zero = graph.const_int(0);
        let add = graph.add_node(Operator::IntOp(ArithOp::Add), InputList::Pair(x, zero));

        let m = ArithmeticPatterns::try_match(&graph, add, &Operator::IntOp(ArithOp::Add));
        assert!(m.is_some());
        let m = m.unwrap();
        assert_eq!(m.replacement(), Some(x));
        assert_eq!(m.pattern_name(), "add_zero_right");
    }

    #[test]
    fn test_add_zero_left() {
        let mut graph = Graph::new();
        let zero = graph.const_int(0);
        let x = graph.const_int(42);
        let add = graph.add_node(Operator::IntOp(ArithOp::Add), InputList::Pair(zero, x));

        let m = ArithmeticPatterns::try_match(&graph, add, &Operator::IntOp(ArithOp::Add));
        assert!(m.is_some());
        assert_eq!(m.unwrap().replacement(), Some(x));
    }

    #[test]
    fn test_sub_zero() {
        let mut graph = Graph::new();
        let x = graph.const_int(42);
        let zero = graph.const_int(0);
        let sub = graph.add_node(Operator::IntOp(ArithOp::Sub), InputList::Pair(x, zero));

        let m = ArithmeticPatterns::try_match(&graph, sub, &Operator::IntOp(ArithOp::Sub));
        assert!(m.is_some());
        assert_eq!(m.unwrap().pattern_name(), "sub_zero");
    }

    #[test]
    fn test_mul_zero() {
        let mut graph = Graph::new();
        let x = graph.const_int(42);
        let zero = graph.const_int(0);
        let mul = graph.add_node(Operator::IntOp(ArithOp::Mul), InputList::Pair(x, zero));

        let m = ArithmeticPatterns::try_match(&graph, mul, &Operator::IntOp(ArithOp::Mul));
        assert!(m.is_some());
        let m = m.unwrap();
        assert_eq!(m.replacement(), Some(zero));
    }

    #[test]
    fn test_mul_one() {
        let mut graph = Graph::new();
        let x = graph.const_int(42);
        let one = graph.const_int(1);
        let mul = graph.add_node(Operator::IntOp(ArithOp::Mul), InputList::Pair(x, one));

        let m = ArithmeticPatterns::try_match(&graph, mul, &Operator::IntOp(ArithOp::Mul));
        assert!(m.is_some());
        let m = m.unwrap();
        assert_eq!(m.replacement(), Some(x));
    }

    #[test]
    fn test_div_one() {
        let mut graph = Graph::new();
        let x = graph.const_int(42);
        let one = graph.const_int(1);
        let div = graph.add_node(Operator::IntOp(ArithOp::TrueDiv), InputList::Pair(x, one));

        let m = ArithmeticPatterns::try_match(&graph, div, &Operator::IntOp(ArithOp::TrueDiv));
        assert!(m.is_some());
        assert_eq!(m.unwrap().replacement(), Some(x));
    }

    #[test]
    fn test_zero_div() {
        let mut graph = Graph::new();
        let zero = graph.const_int(0);
        let x = graph.const_int(42);
        let div = graph.add_node(Operator::IntOp(ArithOp::TrueDiv), InputList::Pair(zero, x));

        let m = ArithmeticPatterns::try_match(&graph, div, &Operator::IntOp(ArithOp::TrueDiv));
        assert!(m.is_some());
        assert_eq!(m.unwrap().replacement(), Some(zero));
    }

    #[test]
    fn test_no_match() {
        let mut graph = Graph::new();
        let x = graph.const_int(42);
        let y = graph.const_int(10);
        let add = graph.add_node(Operator::IntOp(ArithOp::Add), InputList::Pair(x, y));

        let m = ArithmeticPatterns::try_match(&graph, add, &Operator::IntOp(ArithOp::Add));
        assert!(m.is_none());
    }

    #[test]
    fn test_is_zero_int() {
        let mut graph = Graph::new();
        let zero = graph.const_int(0);
        let one = graph.const_int(1);

        assert!(ArithmeticPatterns::is_zero(&graph, zero));
        assert!(!ArithmeticPatterns::is_zero(&graph, one));
    }

    #[test]
    fn test_is_one_int() {
        let mut graph = Graph::new();
        let zero = graph.const_int(0);
        let one = graph.const_int(1);

        assert!(!ArithmeticPatterns::is_one(&graph, zero));
        assert!(ArithmeticPatterns::is_one(&graph, one));
    }
}
