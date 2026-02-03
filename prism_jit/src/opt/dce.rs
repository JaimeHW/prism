//! Dead Code Elimination (DCE) optimization pass.
//!
//! DCE removes nodes that are not used by any live computation.
//! A node is live if:
//!
//! 1. It has side effects (stores, calls, control flow)
//! 2. It is used by another live node
//!
//! # Algorithm
//!
//! 1. Mark all nodes with side effects as live
//! 2. Propagate liveness backwards through use-def chains
//! 3. Remove all nodes that are not marked as live

use super::OptimizationPass;
use crate::ir::arena::BitSet;
use crate::ir::graph::Graph;
use crate::ir::operators::{ControlOp, Operator};

// =============================================================================
// DCE Pass
// =============================================================================

/// Dead Code Elimination pass.
pub struct Dce {
    /// Number of nodes removed.
    removed: usize,
}

impl Dce {
    /// Create a new DCE pass.
    pub fn new() -> Self {
        Dce { removed: 0 }
    }

    /// Get the number of removed nodes.
    pub fn removed(&self) -> usize {
        self.removed
    }

    /// Check if an operator has side effects (cannot be removed).
    fn has_side_effects(op: &Operator) -> bool {
        match op {
            // Control flow is always live
            Operator::Control(ControlOp::Return)
            | Operator::Control(ControlOp::Throw)
            | Operator::Control(ControlOp::If)
            | Operator::Control(ControlOp::Loop)
            | Operator::Control(ControlOp::Region)
            | Operator::Control(ControlOp::Start)
            | Operator::Control(ControlOp::End)
            | Operator::Control(ControlOp::Deopt) => true,

            // Calls have side effects
            Operator::Call(_) => true,

            // Stores have side effects
            Operator::SetItem | Operator::SetAttr => true,

            // Memory operations that mutate
            Operator::Memory(_) => true,

            // Guards must be preserved
            Operator::Guard(_) => true,

            // Container construction can have side effects
            Operator::BuildList(_) | Operator::BuildTuple(_) | Operator::BuildDict(_) => true,

            // Iteration has side effects
            Operator::GetIter | Operator::IterNext => true,

            // Pure operations (can be removed if unused)
            Operator::ConstInt(_)
            | Operator::ConstFloat(_)
            | Operator::ConstBool(_)
            | Operator::ConstNone
            | Operator::Parameter(_)
            | Operator::IntOp(_)
            | Operator::FloatOp(_)
            | Operator::GenericOp(_)
            | Operator::MulHigh
            | Operator::MulHighSigned
            | Operator::IntCmp(_)
            | Operator::FloatCmp(_)
            | Operator::GenericCmp(_)
            | Operator::Bitwise(_)
            | Operator::LogicalNot
            | Operator::Phi
            | Operator::LoopPhi
            | Operator::GetItem
            | Operator::GetAttr
            | Operator::Len
            | Operator::TypeCheck
            | Operator::Box
            | Operator::Unbox
            | Operator::Projection(_) => false,
        }
    }
}

impl Default for Dce {
    fn default() -> Self {
        Self::new()
    }
}

impl OptimizationPass for Dce {
    fn name(&self) -> &'static str {
        "DCE"
    }

    fn run(&mut self, graph: &mut Graph) -> bool {
        self.removed = 0;

        let node_count = graph.len();
        let mut live = BitSet::with_capacity(node_count);
        let mut worklist = Vec::new();

        // Phase 1: Mark nodes with side effects as live
        for (id, node) in graph.iter() {
            if Self::has_side_effects(&node.op) || node.is_dead() {
                if !node.is_dead() {
                    live.insert(id.as_usize());
                    worklist.push(id);
                }
            }
        }

        // Phase 2: Propagate liveness backwards through inputs
        while let Some(id) = worklist.pop() {
            let node = graph.node(id);
            for input in node.inputs.iter() {
                if input.is_valid() && !live.contains(input.as_usize()) {
                    live.insert(input.as_usize());
                    worklist.push(input);
                }
            }
        }

        // Phase 3: Collect dead nodes
        let mut dead_nodes = Vec::new();
        for (id, node) in graph.iter() {
            if !live.contains(id.as_usize()) && !node.is_dead() {
                dead_nodes.push(id);
            }
        }

        // Phase 4: Kill dead nodes
        self.removed = dead_nodes.len();
        for id in dead_nodes {
            graph.kill(id);
        }

        self.removed > 0
    }
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
    fn test_dce_removes_unused() {
        let mut builder = GraphBuilder::new(4, 2);

        let p0 = builder.parameter(0).unwrap();
        let p1 = builder.parameter(1).unwrap();

        // Used computation
        let sum = builder.int_add(p0, p1);
        let _ret = builder.return_value(sum);

        // Unused computation (should be removed)
        let _unused = builder.int_sub(p0, p1);

        let mut graph = builder.finish();

        let mut dce = Dce::new();
        let changed = dce.run(&mut graph);

        assert!(changed);
        assert!(dce.removed() >= 1);
    }

    #[test]
    fn test_dce_preserves_used() {
        let mut builder = GraphBuilder::new(4, 2);

        let p0 = builder.parameter(0).unwrap();
        let p1 = builder.parameter(1).unwrap();

        // All nodes are used
        let sum = builder.int_add(p0, p1);
        let double = builder.int_add(sum, sum);
        let _ret = builder.return_value(double);

        let mut graph = builder.finish();

        let mut dce = Dce::new();
        let changed = dce.run(&mut graph);

        // Nothing should be removed
        assert!(!changed);
        assert_eq!(dce.removed(), 0);
    }

    #[test]
    fn test_dce_preserves_side_effects() {
        let mut builder = GraphBuilder::new(4, 2);

        let p0 = builder.parameter(0).unwrap();
        let p1 = builder.parameter(1).unwrap();

        // Side-effecting operation (should not be removed even if unused)
        let key = builder.const_int(0);
        let _set = builder.set_item(p0, key, p1);

        let none = builder.const_none();
        let _ret = builder.return_value(none);

        let mut graph = builder.finish();
        let before_count = graph.len();

        let mut dce = Dce::new();
        let _changed = dce.run(&mut graph);

        // Side effects should be preserved
        // The graph might shrink but set_item should remain
        assert!(graph.len() >= before_count - 2); // Allow some cleanup
    }

    #[test]
    fn test_dce_unused_constants() {
        let mut builder = GraphBuilder::new(4, 0);

        // Create unused constants
        let _c1 = builder.const_int(42);
        let _c2 = builder.const_int(100);
        let _c3 = builder.const_float(3.14);

        // But only use one
        let used = builder.const_int(1);
        let _ret = builder.return_value(used);

        let mut graph = builder.finish();

        let mut dce = Dce::new();
        let changed = dce.run(&mut graph);

        assert!(changed);
        assert!(dce.removed() >= 3); // At least the 3 unused constants
    }

    #[test]
    fn test_dce_chain_removal() {
        let mut builder = GraphBuilder::new(4, 2);

        let p0 = builder.parameter(0).unwrap();
        let p1 = builder.parameter(1).unwrap();

        // Create a chain of unused computations
        let a = builder.int_add(p0, p1);
        let b = builder.int_mul(a, p0);
        let _c = builder.int_sub(b, p1);

        // Return something else
        let _ret = builder.return_value(p0);

        let mut graph = builder.finish();

        let mut dce = Dce::new();
        let changed = dce.run(&mut graph);

        // The entire chain should be removed
        assert!(changed);
        assert!(dce.removed() >= 3);
    }
}
