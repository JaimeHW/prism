//! Control flow patterns for instruction combining.
//!
//! Patterns include:
//! - Branch on constant condition -> unconditional

use crate::ir::graph::Graph;
use crate::ir::node::NodeId;
use crate::ir::operators::{ControlOp, Operator};

use super::PatternMatch;

// =============================================================================
// Control Patterns
// =============================================================================

/// Control flow pattern matcher.
pub struct ControlPatterns;

impl ControlPatterns {
    /// Try to match a control flow pattern.
    pub fn try_match(graph: &Graph, node: NodeId, op: &Operator) -> Option<PatternMatch> {
        match op {
            Operator::Control(ctrl_op) => Self::try_control(graph, node, *ctrl_op),
            _ => None,
        }
    }

    /// Try to match control operation patterns.
    fn try_control(graph: &Graph, node: NodeId, op: ControlOp) -> Option<PatternMatch> {
        match op {
            ControlOp::If => Self::try_if_patterns(graph, node),
            _ => None,
        }
    }

    /// Try if/branch patterns.
    fn try_if_patterns(graph: &Graph, node: NodeId) -> Option<PatternMatch> {
        let n = graph.get(node)?;

        // Use get() since as_slice() only works for Many variant
        let condition = n.inputs.get(0)?;

        // Branch on constant condition
        if let Some(cond_val) = Self::get_bool_constant(graph, condition) {
            if cond_val {
                return Some(PatternMatch::new(node, "branch_true"));
            } else {
                return Some(PatternMatch::new(node, "branch_false"));
            }
        }

        None
    }

    /// Get boolean constant value if node is a constant.
    fn get_bool_constant(graph: &Graph, node: NodeId) -> Option<bool> {
        if let Some(n) = graph.get(node) {
            match n.op {
                Operator::ConstBool(v) => Some(v),
                Operator::ConstInt(v) => Some(v != 0),
                _ => None,
            }
        } else {
            None
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
    fn test_branch_true() {
        let mut graph = Graph::new();
        let cond = graph.add_node(Operator::ConstBool(true), InputList::Empty);
        let if_node = graph.add_node(Operator::Control(ControlOp::If), InputList::Single(cond));

        let m = ControlPatterns::try_match(&graph, if_node, &Operator::Control(ControlOp::If));
        assert!(m.is_some());
        assert_eq!(m.unwrap().pattern_name(), "branch_true");
    }

    #[test]
    fn test_branch_false() {
        let mut graph = Graph::new();
        let cond = graph.add_node(Operator::ConstBool(false), InputList::Empty);
        let if_node = graph.add_node(Operator::Control(ControlOp::If), InputList::Single(cond));

        let m = ControlPatterns::try_match(&graph, if_node, &Operator::Control(ControlOp::If));
        assert!(m.is_some());
        assert_eq!(m.unwrap().pattern_name(), "branch_false");
    }

    #[test]
    fn test_branch_int_true() {
        let mut graph = Graph::new();
        let cond = graph.const_int(1);
        let if_node = graph.add_node(Operator::Control(ControlOp::If), InputList::Single(cond));

        let m = ControlPatterns::try_match(&graph, if_node, &Operator::Control(ControlOp::If));
        assert!(m.is_some());
        assert_eq!(m.unwrap().pattern_name(), "branch_true");
    }

    #[test]
    fn test_branch_int_false() {
        let mut graph = Graph::new();
        let cond = graph.const_int(0);
        let if_node = graph.add_node(Operator::Control(ControlOp::If), InputList::Single(cond));

        let m = ControlPatterns::try_match(&graph, if_node, &Operator::Control(ControlOp::If));
        assert!(m.is_some());
        assert_eq!(m.unwrap().pattern_name(), "branch_false");
    }

    #[test]
    fn test_no_match_variable_condition() {
        let mut graph = Graph::new();
        // Non-constant condition (an allocation)
        let alloc = graph.add_node(
            Operator::Memory(crate::ir::operators::MemoryOp::Alloc),
            InputList::Empty,
        );
        let if_node = graph.add_node(Operator::Control(ControlOp::If), InputList::Single(alloc));

        let m = ControlPatterns::try_match(&graph, if_node, &Operator::Control(ControlOp::If));
        assert!(m.is_none());
    }

    #[test]
    fn test_get_bool_constant_true() {
        let mut graph = Graph::new();
        let c = graph.add_node(Operator::ConstBool(true), InputList::Empty);

        assert_eq!(ControlPatterns::get_bool_constant(&graph, c), Some(true));
    }

    #[test]
    fn test_get_bool_constant_false() {
        let mut graph = Graph::new();
        let c = graph.add_node(Operator::ConstBool(false), InputList::Empty);

        assert_eq!(ControlPatterns::get_bool_constant(&graph, c), Some(false));
    }

    #[test]
    fn test_get_bool_constant_int() {
        let mut graph = Graph::new();
        let c = graph.const_int(5);

        assert_eq!(ControlPatterns::get_bool_constant(&graph, c), Some(true));
    }
}
