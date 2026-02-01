//! Global Value Numbering (GVN) optimization pass.
//!
//! GVN eliminates redundant computations by finding nodes that compute
//! the same value and replacing all uses with a single canonical computation.
//!
//! # Algorithm
//!
//! 1. For each node, compute a hash based on its operator and input IDs
//! 2. If a node with the same hash exists, compare structurally
//! 3. If structurally identical, replace all uses of the duplicate
//!
//! # Complexity
//!
//! O(n) time where n is the number of nodes, assuming good hash distribution.

use super::OptimizationPass;
use crate::ir::graph::Graph;
use crate::ir::node::NodeId;
use crate::ir::operators::Operator;

use std::collections::HashMap;

// =============================================================================
// Value Numbering Key
// =============================================================================

/// A key for value numbering that captures the essential properties of a node.
#[derive(Clone, PartialEq, Eq, Hash)]
struct GvnKey {
    /// The operator.
    op: Operator,
    /// Input node IDs (as raw u32 values for hashing).
    inputs: Vec<u32>,
}

impl GvnKey {
    /// Create a GVN key for a node.
    fn new(op: Operator, inputs: &[NodeId]) -> Self {
        GvnKey {
            op,
            inputs: inputs.iter().map(|id| id.index()).collect(),
        }
    }
}

// =============================================================================
// GVN Pass
// =============================================================================

/// Global Value Numbering pass.
pub struct Gvn {
    /// Number of nodes deduplicated.
    deduplicated: usize,
}

impl Gvn {
    /// Create a new GVN pass.
    pub fn new() -> Self {
        Gvn { deduplicated: 0 }
    }

    /// Get the number of deduplicated nodes.
    pub fn deduplicated(&self) -> usize {
        self.deduplicated
    }

    /// Check if an operator is eligible for GVN.
    fn is_gvn_eligible(op: &Operator) -> bool {
        match op {
            // Constants are always eligible (pure, no side effects)
            Operator::ConstInt(_)
            | Operator::ConstFloat(_)
            | Operator::ConstBool(_)
            | Operator::ConstNone => true,

            // Pure arithmetic operations
            Operator::IntOp(_) | Operator::FloatOp(_) | Operator::GenericOp(_) => true,

            // Pure comparisons
            Operator::IntCmp(_) | Operator::FloatCmp(_) | Operator::GenericCmp(_) => true,

            // Bitwise operations
            Operator::Bitwise(_) => true,

            // Logical not
            Operator::LogicalNot => true,

            // Box/unbox are pure
            Operator::Box | Operator::Unbox => true,

            // Projections are pure
            Operator::Projection(_) => true,

            // These have side effects or are context-dependent
            Operator::Parameter(_)
            | Operator::Phi
            | Operator::LoopPhi
            | Operator::Control(_)
            | Operator::Guard(_)
            | Operator::Call(_)
            | Operator::Memory(_)
            | Operator::GetItem
            | Operator::SetItem
            | Operator::GetAttr
            | Operator::SetAttr
            | Operator::GetIter
            | Operator::IterNext
            | Operator::Len
            | Operator::TypeCheck
            | Operator::BuildList(_)
            | Operator::BuildTuple(_)
            | Operator::BuildDict(_) => false,
        }
    }
}

impl Default for Gvn {
    fn default() -> Self {
        Self::new()
    }
}

impl OptimizationPass for Gvn {
    fn name(&self) -> &'static str {
        "GVN"
    }

    fn run(&mut self, graph: &mut Graph) -> bool {
        self.deduplicated = 0;

        // Map from GVN key to canonical node ID
        let mut value_table: HashMap<GvnKey, NodeId> = HashMap::new();

        // Map from original node to replacement
        let mut replacements: HashMap<NodeId, NodeId> = HashMap::new();

        // First pass: build value table and find duplicates
        for (id, node) in graph.iter() {
            // Skip nodes not eligible for GVN
            if !Self::is_gvn_eligible(&node.op) {
                continue;
            }

            // Create key from operator and inputs
            let inputs: Vec<NodeId> = node.inputs.iter().collect();
            let key = GvnKey::new(node.op, &inputs);

            // Check if we've seen this computation before
            if let Some(&canonical) = value_table.get(&key) {
                // Found a duplicate - record replacement
                if canonical != id {
                    replacements.insert(id, canonical);
                    self.deduplicated += 1;
                }
            } else {
                // New computation - add to table
                value_table.insert(key, id);
            }
        }

        // Second pass: apply replacements
        if !replacements.is_empty() {
            for (old, new) in &replacements {
                graph.replace_all_uses(*old, *new);
            }
        }

        self.deduplicated > 0
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
    fn test_gvn_duplicate_constants() {
        let mut builder = GraphBuilder::new(4, 0);

        // Create duplicate constants
        let a = builder.const_int(42);
        let b = builder.const_int(42);
        let sum = builder.int_add(a, b);
        let _ret = builder.return_value(sum);

        let mut graph = builder.finish();
        let initial_len = graph.len();

        let mut gvn = Gvn::new();
        let changed = gvn.run(&mut graph);

        assert!(changed);
        assert!(gvn.deduplicated() >= 1);
        // The duplicate constant should be deduplicated
        assert!(graph.len() <= initial_len);
    }

    #[test]
    fn test_gvn_duplicate_arithmetic() {
        let mut builder = GraphBuilder::new(4, 2);

        // p0 + p1
        let p0 = builder.parameter(0).unwrap();
        let p1 = builder.parameter(1).unwrap();
        let sum1 = builder.int_add(p0, p1);

        // p0 + p1 again (duplicate)
        let sum2 = builder.int_add(p0, p1);

        // Use both
        let result = builder.int_add(sum1, sum2);
        let _ret = builder.return_value(result);

        let mut graph = builder.finish();

        let mut gvn = Gvn::new();
        let changed = gvn.run(&mut graph);

        // sum2 should be replaced with sum1
        assert!(changed);
        assert_eq!(gvn.deduplicated(), 1);
    }

    #[test]
    fn test_gvn_different_ops_not_merged() {
        let mut builder = GraphBuilder::new(4, 2);

        let p0 = builder.parameter(0).unwrap();
        let p1 = builder.parameter(1).unwrap();

        // Different operations should not be merged
        let sum = builder.int_add(p0, p1);
        let sub = builder.int_sub(p0, p1);

        let result = builder.int_add(sum, sub);
        let _ret = builder.return_value(result);

        let mut graph = builder.finish();

        let mut gvn = Gvn::new();
        let changed = gvn.run(&mut graph);

        // Nothing to deduplicate
        assert!(!changed);
        assert_eq!(gvn.deduplicated(), 0);
    }

    #[test]
    fn test_gvn_comparison() {
        let mut builder = GraphBuilder::new(4, 2);

        let p0 = builder.parameter(0).unwrap();
        let p1 = builder.parameter(1).unwrap();

        // Same comparison twice
        let lt1 = builder.int_lt(p0, p1);
        let lt2 = builder.int_lt(p0, p1);

        let _ = builder.const_bool(true); // Use both somehow
        builder.set_register(0, lt1);
        builder.set_register(1, lt2);

        let mut graph = builder.finish();

        let mut gvn = Gvn::new();
        let changed = gvn.run(&mut graph);

        assert!(changed);
        assert_eq!(gvn.deduplicated(), 1);
    }

    #[test]
    fn test_gvn_preserves_side_effects() {
        let mut builder = GraphBuilder::new(4, 2);

        let p0 = builder.parameter(0).unwrap();
        let p1 = builder.parameter(1).unwrap();

        // GetItem has side effects, should not be merged
        let get1 = builder.get_item(p0, p1);
        let get2 = builder.get_item(p0, p1);

        let result = builder.int_add(get1, get2);
        let _ret = builder.return_value(result);

        let mut graph = builder.finish();

        let mut gvn = Gvn::new();
        let changed = gvn.run(&mut graph);

        // get_item should not be deduplicated (has side effects)
        assert!(!changed);
        assert_eq!(gvn.deduplicated(), 0);
    }
}
