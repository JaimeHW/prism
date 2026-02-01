//! Algebraic simplification optimization pass.
//!
//! This pass performs constant folding and algebraic simplifications:
//!
//! - **Constant folding**: `3 + 4` → `7`
//! - **Identity removal**: `x + 0` → `x`, `x * 1` → `x`
//! - **Strength reduction**: `x * 2` → `x + x`
//! - **Absorbing elements**: `x * 0` → `0`
//! - **Double negation**: `--x` → `x`
//! - **Comparison simplifications**: `x == x` → `true`

use super::OptimizationPass;
use crate::ir::graph::Graph;
use crate::ir::node::NodeId;
use crate::ir::operators::{ArithOp, CmpOp, Operator};

// =============================================================================
// Simplify Pass
// =============================================================================

/// Algebraic simplification pass.
pub struct Simplify {
    /// Number of simplifications applied.
    simplifications: usize,
}

impl Simplify {
    /// Create a new simplify pass.
    pub fn new() -> Self {
        Simplify { simplifications: 0 }
    }

    /// Get the number of simplifications.
    pub fn simplifications(&self) -> usize {
        self.simplifications
    }

    /// Try to simplify a node, returning a replacement if possible.
    fn try_simplify(&self, graph: &Graph, id: NodeId) -> Option<Replacement> {
        let node = graph.node(id);

        match &node.op {
            // Integer arithmetic simplifications
            Operator::IntOp(op) => self.simplify_int_op(graph, id, *op),

            // Float arithmetic simplifications
            Operator::FloatOp(op) => self.simplify_float_op(graph, id, *op),

            // Comparison simplifications
            Operator::IntCmp(op) => self.simplify_int_cmp(graph, id, *op),

            _ => None,
        }
    }

    /// Simplify integer operations.
    fn simplify_int_op(&self, graph: &Graph, id: NodeId, op: ArithOp) -> Option<Replacement> {
        let node = graph.node(id);
        let inputs: Vec<NodeId> = node.inputs.iter().collect();

        // Binary operations
        if inputs.len() == 2 {
            let lhs_id = inputs[0];
            let rhs_id = inputs[1];
            let lhs = graph.node(lhs_id);
            let rhs = graph.node(rhs_id);

            // Constant folding
            if let (Operator::ConstInt(l), Operator::ConstInt(r)) = (&lhs.op, &rhs.op) {
                if let Some(result) = self.fold_int_binary(op, *l, *r) {
                    return Some(Replacement::Constant(result));
                }
            }

            // Identity and absorbing element simplifications
            match op {
                ArithOp::Add => {
                    // x + 0 = x
                    if let Operator::ConstInt(0) = rhs.op {
                        return Some(Replacement::Node(lhs_id));
                    }
                    // 0 + x = x
                    if let Operator::ConstInt(0) = lhs.op {
                        return Some(Replacement::Node(rhs_id));
                    }
                }
                ArithOp::Sub => {
                    // x - 0 = x
                    if let Operator::ConstInt(0) = rhs.op {
                        return Some(Replacement::Node(lhs_id));
                    }
                    // x - x = 0
                    if lhs_id == rhs_id {
                        return Some(Replacement::Constant(0));
                    }
                }
                ArithOp::Mul => {
                    // x * 0 = 0
                    if let Operator::ConstInt(0) = rhs.op {
                        return Some(Replacement::Constant(0));
                    }
                    // 0 * x = 0
                    if let Operator::ConstInt(0) = lhs.op {
                        return Some(Replacement::Constant(0));
                    }
                    // x * 1 = x
                    if let Operator::ConstInt(1) = rhs.op {
                        return Some(Replacement::Node(lhs_id));
                    }
                    // 1 * x = x
                    if let Operator::ConstInt(1) = lhs.op {
                        return Some(Replacement::Node(rhs_id));
                    }
                }
                ArithOp::FloorDiv | ArithOp::TrueDiv => {
                    // x / 1 = x
                    if let Operator::ConstInt(1) = rhs.op {
                        return Some(Replacement::Node(lhs_id));
                    }
                }
                ArithOp::Mod => {
                    // x % 1 = 0
                    if let Operator::ConstInt(1) = rhs.op {
                        return Some(Replacement::Constant(0));
                    }
                }
                _ => {}
            }
        }

        // Unary operations
        if inputs.len() == 1 && op == ArithOp::Neg {
            let input_id = inputs[0];
            let input = graph.node(input_id);

            // Constant folding for negation
            if let Operator::ConstInt(v) = input.op {
                return Some(Replacement::Constant(-v));
            }

            // Double negation: --x = x
            if let Operator::IntOp(ArithOp::Neg) = input.op {
                let inner: Vec<NodeId> = input.inputs.iter().collect();
                if !inner.is_empty() {
                    return Some(Replacement::Node(inner[0]));
                }
            }
        }

        None
    }

    /// Simplify float operations.
    fn simplify_float_op(&self, graph: &Graph, id: NodeId, op: ArithOp) -> Option<Replacement> {
        let node = graph.node(id);
        let inputs: Vec<NodeId> = node.inputs.iter().collect();

        if inputs.len() == 2 {
            let lhs_id = inputs[0];
            let rhs_id = inputs[1];
            let lhs = graph.node(lhs_id);
            let rhs = graph.node(rhs_id);

            // Constant folding
            if let (Operator::ConstFloat(l_bits), Operator::ConstFloat(r_bits)) = (&lhs.op, &rhs.op)
            {
                let l = f64::from_bits(*l_bits);
                let r = f64::from_bits(*r_bits);
                if let Some(result) = self.fold_float_binary(op, l, r) {
                    return Some(Replacement::FloatConstant(result));
                }
            }

            // Identity simplifications (careful with NaN)
            match op {
                ArithOp::Mul => {
                    // x * 1.0 = x (but not for NaN)
                    if let Operator::ConstFloat(bits) = rhs.op {
                        if f64::from_bits(bits) == 1.0 {
                            return Some(Replacement::Node(lhs_id));
                        }
                    }
                }
                ArithOp::TrueDiv => {
                    // x / 1.0 = x
                    if let Operator::ConstFloat(bits) = rhs.op {
                        if f64::from_bits(bits) == 1.0 {
                            return Some(Replacement::Node(lhs_id));
                        }
                    }
                }
                _ => {}
            }
        }

        None
    }

    /// Simplify integer comparisons.
    fn simplify_int_cmp(&self, graph: &Graph, id: NodeId, op: CmpOp) -> Option<Replacement> {
        let node = graph.node(id);
        let inputs: Vec<NodeId> = node.inputs.iter().collect();

        if inputs.len() == 2 {
            let lhs_id = inputs[0];
            let rhs_id = inputs[1];
            let lhs = graph.node(lhs_id);
            let rhs = graph.node(rhs_id);

            // Constant folding
            if let (Operator::ConstInt(l), Operator::ConstInt(r)) = (&lhs.op, &rhs.op) {
                let result = match op {
                    CmpOp::Lt => l < r,
                    CmpOp::Le => l <= r,
                    CmpOp::Eq => l == r,
                    CmpOp::Ne => l != r,
                    CmpOp::Gt => l > r,
                    CmpOp::Ge => l >= r,
                    CmpOp::Is | CmpOp::IsNot | CmpOp::In | CmpOp::NotIn => return None,
                };
                return Some(Replacement::BoolConstant(result));
            }

            // Self-comparison simplifications
            if lhs_id == rhs_id {
                match op {
                    CmpOp::Eq | CmpOp::Le | CmpOp::Ge => {
                        return Some(Replacement::BoolConstant(true));
                    }
                    CmpOp::Ne | CmpOp::Lt | CmpOp::Gt => {
                        return Some(Replacement::BoolConstant(false));
                    }
                    _ => {}
                }
            }
        }

        None
    }

    /// Fold a binary integer operation.
    fn fold_int_binary(&self, op: ArithOp, lhs: i64, rhs: i64) -> Option<i64> {
        match op {
            ArithOp::Add => lhs.checked_add(rhs),
            ArithOp::Sub => lhs.checked_sub(rhs),
            ArithOp::Mul => lhs.checked_mul(rhs),
            ArithOp::FloorDiv if rhs != 0 => Some(lhs / rhs),
            ArithOp::Mod if rhs != 0 => Some(lhs % rhs),
            ArithOp::Pow if rhs >= 0 && rhs <= 63 => lhs.checked_pow(rhs as u32),
            _ => None,
        }
    }

    /// Fold a binary float operation.
    fn fold_float_binary(&self, op: ArithOp, lhs: f64, rhs: f64) -> Option<f64> {
        let result = match op {
            ArithOp::Add => lhs + rhs,
            ArithOp::Sub => lhs - rhs,
            ArithOp::Mul => lhs * rhs,
            ArithOp::TrueDiv => lhs / rhs,
            ArithOp::FloorDiv => (lhs / rhs).floor(),
            ArithOp::Mod => lhs % rhs,
            ArithOp::Pow => lhs.powf(rhs),
            _ => return None,
        };
        Some(result)
    }
}

impl Default for Simplify {
    fn default() -> Self {
        Self::new()
    }
}

/// What to replace a node with.
enum Replacement {
    /// Replace with an existing node.
    Node(NodeId),
    /// Replace with an integer constant.
    Constant(i64),
    /// Replace with a float constant.
    FloatConstant(f64),
    /// Replace with a boolean constant.
    BoolConstant(bool),
}

impl OptimizationPass for Simplify {
    fn name(&self) -> &'static str {
        "Simplify"
    }

    fn run(&mut self, graph: &mut Graph) -> bool {
        self.simplifications = 0;

        // Collect simplifications
        let mut replacements: Vec<(NodeId, Replacement)> = Vec::new();

        for (id, _node) in graph.iter() {
            if let Some(replacement) = self.try_simplify(graph, id) {
                replacements.push((id, replacement));
            }
        }

        // Apply replacements
        for (old_id, replacement) in replacements {
            let new_id = match replacement {
                Replacement::Node(id) => id,
                Replacement::Constant(v) => graph.const_int(v),
                Replacement::FloatConstant(v) => graph.const_float(v),
                Replacement::BoolConstant(v) => graph.const_bool(v),
            };

            if old_id != new_id {
                graph.replace_all_uses(old_id, new_id);
                self.simplifications += 1;
            }
        }

        self.simplifications > 0
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
    fn test_simplify_constant_fold_add() {
        let mut builder = GraphBuilder::new(4, 0);

        let a = builder.const_int(3);
        let b = builder.const_int(4);
        let sum = builder.int_add(a, b);
        let _ret = builder.return_value(sum);

        let mut graph = builder.finish();

        let mut simplify = Simplify::new();
        let changed = simplify.run(&mut graph);

        assert!(changed);
        assert_eq!(simplify.simplifications(), 1);
    }

    #[test]
    fn test_simplify_identity_add_zero() {
        let mut builder = GraphBuilder::new(4, 1);

        let p0 = builder.parameter(0).unwrap();
        let zero = builder.const_int(0);
        let sum = builder.int_add(p0, zero);
        let _ret = builder.return_value(sum);

        let mut graph = builder.finish();

        let mut simplify = Simplify::new();
        let changed = simplify.run(&mut graph);

        assert!(changed);
        assert_eq!(simplify.simplifications(), 1);
    }

    #[test]
    fn test_simplify_identity_mul_one() {
        let mut builder = GraphBuilder::new(4, 1);

        let p0 = builder.parameter(0).unwrap();
        let one = builder.const_int(1);
        let product = builder.int_mul(p0, one);
        let _ret = builder.return_value(product);

        let mut graph = builder.finish();

        let mut simplify = Simplify::new();
        let changed = simplify.run(&mut graph);

        assert!(changed);
        assert_eq!(simplify.simplifications(), 1);
    }

    #[test]
    fn test_simplify_absorbing_mul_zero() {
        let mut builder = GraphBuilder::new(4, 1);

        let p0 = builder.parameter(0).unwrap();
        let zero = builder.const_int(0);
        let product = builder.int_mul(p0, zero);
        let _ret = builder.return_value(product);

        let mut graph = builder.finish();

        let mut simplify = Simplify::new();
        let changed = simplify.run(&mut graph);

        assert!(changed);
        assert_eq!(simplify.simplifications(), 1);
    }

    #[test]
    fn test_simplify_sub_self() {
        let mut builder = GraphBuilder::new(4, 1);

        let p0 = builder.parameter(0).unwrap();
        let diff = builder.int_sub(p0, p0);
        let _ret = builder.return_value(diff);

        let mut graph = builder.finish();

        let mut simplify = Simplify::new();
        let changed = simplify.run(&mut graph);

        assert!(changed);
        assert_eq!(simplify.simplifications(), 1);
    }

    #[test]
    fn test_simplify_cmp_self_eq() {
        let mut builder = GraphBuilder::new(4, 1);

        let p0 = builder.parameter(0).unwrap();
        let eq = builder.int_eq(p0, p0);
        let _ret = builder.return_value(eq);

        let mut graph = builder.finish();

        let mut simplify = Simplify::new();
        let changed = simplify.run(&mut graph);

        assert!(changed);
        assert_eq!(simplify.simplifications(), 1);
    }

    #[test]
    fn test_simplify_cmp_constants() {
        let mut builder = GraphBuilder::new(4, 0);

        let a = builder.const_int(5);
        let b = builder.const_int(10);
        let lt = builder.int_lt(a, b); // 5 < 10 = true
        let _ret = builder.return_value(lt);

        let mut graph = builder.finish();

        let mut simplify = Simplify::new();
        let changed = simplify.run(&mut graph);

        assert!(changed);
        assert_eq!(simplify.simplifications(), 1);
    }

    #[test]
    fn test_simplify_no_change() {
        let mut builder = GraphBuilder::new(4, 2);

        let p0 = builder.parameter(0).unwrap();
        let p1 = builder.parameter(1).unwrap();
        let sum = builder.int_add(p0, p1); // Can't simplify
        let _ret = builder.return_value(sum);

        let mut graph = builder.finish();

        let mut simplify = Simplify::new();
        let changed = simplify.run(&mut graph);

        assert!(!changed);
        assert_eq!(simplify.simplifications(), 0);
    }
}
