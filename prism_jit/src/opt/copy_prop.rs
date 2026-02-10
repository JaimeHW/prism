//! Copy Propagation optimization pass.
//!
//! Copy Propagation replaces uses of a copied value with the original value,
//! eliminating redundant copies and enabling more optimization opportunities.
//!
//! # Algorithm
//!
//! 1. Identify all copy-like operations (Phi with single value input, identity operations)
//! 2. For each copy `x = y`, replace all uses of `x` with `y`
//! 3. Mark the copy as dead for subsequent DCE
//!
//! # Performance
//!
//! - O(n) where n = number of nodes
//! - Enables more GVN opportunities by exposing identical values
//! - Reduces register pressure by eliminating unnecessary moves

use super::OptimizationPass;
use crate::ir::graph::Graph;
use crate::ir::node::{InputList, NodeId};
use crate::ir::operators::Operator;

use rustc_hash::FxHashMap;

// =============================================================================
// Copy Propagation Pass
// =============================================================================

/// Copy Propagation statistics.
#[derive(Debug, Clone, Default)]
pub struct CopyPropStats {
    /// Number of copies identified.
    pub copies_found: usize,
    /// Number of uses rewritten.
    pub uses_rewritten: usize,
    /// Number of copies eliminated (became dead).
    pub copies_eliminated: usize,
    /// Number of phi nodes simplified.
    pub phis_simplified: usize,
}

/// Copy Propagation optimization pass.
///
/// This pass eliminates redundant copies by replacing uses of copied values
/// with the original value. It handles:
///
/// - Direct copies (move operations)
/// - Phi nodes with a single non-self value input
/// - Phi nodes where all value inputs are identical
#[derive(Debug)]
pub struct CopyProp {
    /// Statistics from the last run.
    stats: CopyPropStats,
    /// Map from copied node to original.
    copies: FxHashMap<NodeId, NodeId>,
    /// Aggressive mode for more thorough analysis.
    aggressive: bool,
}

impl CopyProp {
    /// Create a new copy propagation pass.
    pub fn new() -> Self {
        Self {
            stats: CopyPropStats::default(),
            copies: FxHashMap::default(),
            aggressive: false,
        }
    }

    /// Create an aggressive copy propagation pass.
    ///
    /// Aggressive mode performs:
    /// - Transitive copy chain resolution
    /// - Cross-block copy propagation
    pub fn aggressive() -> Self {
        Self {
            stats: CopyPropStats::default(),
            copies: FxHashMap::default(),
            aggressive: true,
        }
    }

    /// Get statistics from the last run.
    pub fn stats(&self) -> &CopyPropStats {
        &self.stats
    }

    /// Get number of copies found.
    #[inline]
    pub fn copies_found(&self) -> usize {
        self.stats.copies_found
    }

    /// Get number of uses rewritten.
    #[inline]
    pub fn uses_rewritten(&self) -> usize {
        self.stats.uses_rewritten
    }

    /// Run copy propagation on the graph.
    fn run_copy_prop(&mut self, graph: &mut Graph) -> bool {
        self.stats = CopyPropStats::default();
        self.copies.clear();

        // Phase 1: Identify all copies
        self.identify_copies(graph);

        if self.copies.is_empty() {
            return false;
        }

        // Phase 2: Resolve transitive copy chains
        if self.aggressive {
            self.resolve_copy_chains();
        }

        // Phase 3: Rewrite uses
        let changed = self.rewrite_uses(graph);

        changed
    }

    /// Identify all copy-like operations in the graph.
    ///
    /// For Phi nodes, the first input is typically the control region,
    /// so we only look at value inputs starting from index 1.
    fn identify_copies(&mut self, graph: &Graph) {
        for (node_id, node) in graph.iter() {
            let original = match &node.op {
                // Phi with single value input (after region) is a copy
                // Phi structure: inputs[0] = region, inputs[1..] = values
                Operator::Phi if node.inputs.len() == 2 => {
                    // Single value input after region
                    self.stats.phis_simplified += 1;
                    match node.inputs.get(1) {
                        Some(val) => val,
                        None => continue,
                    }
                }

                // Phi where all value inputs are the same (excluding region and self-references)
                Operator::Phi if node.inputs.len() > 2 => {
                    let first = match node.inputs.get(1) {
                        Some(val) => val,
                        None => continue,
                    };

                    // Check if all value inputs (from index 1) are the same
                    let mut all_same = true;
                    for i in 2..node.inputs.len() {
                        if let Some(input) = node.inputs.get(i) {
                            if input != first && input != node_id {
                                all_same = false;
                                break;
                            }
                        }
                    }

                    if all_same {
                        self.stats.phis_simplified += 1;
                        first
                    } else {
                        continue;
                    }
                }

                // LoopPhi has: inputs[0] = loop header, inputs[1] = initial, inputs[2] = back edge
                // If initial == back edge (excluding self-references), it's a copy
                Operator::LoopPhi if node.inputs.len() >= 2 => {
                    let initial = match node.inputs.get(1) {
                        Some(val) => val,
                        None => continue,
                    };

                    if node.inputs.len() == 2 {
                        // No back edge yet
                        self.stats.phis_simplified += 1;
                        initial
                    } else if node.inputs.len() == 3 {
                        let back = match node.inputs.get(2) {
                            Some(val) => val,
                            None => continue,
                        };
                        if initial == back || back == node_id {
                            self.stats.phis_simplified += 1;
                            initial
                        } else {
                            continue;
                        }
                    } else {
                        continue;
                    }
                }

                // Projection(0) with single input is identity
                Operator::Projection(0) if node.inputs.len() == 1 => match node.inputs.get(0) {
                    Some(val) => val,
                    None => continue,
                },

                _ => continue,
            };

            // Don't create self-loops
            if original != node_id {
                self.copies.insert(node_id, original);
                self.stats.copies_found += 1;
            }
        }
    }

    /// Resolve transitive copy chains.
    fn resolve_copy_chains(&mut self) {
        let keys: Vec<NodeId> = self.copies.keys().copied().collect();

        for key in keys {
            let mut current = key;
            let mut chain = Vec::new();

            // Walk the chain
            while let Some(&next) = self.copies.get(&current) {
                if chain.contains(&next) {
                    break;
                }
                chain.push(current);
                current = next;
            }

            // Update all entries in the chain to point to the root
            for node in chain {
                self.copies.insert(node, current);
            }
        }
    }

    /// Rewrite uses of copied values to use originals.
    fn rewrite_uses(&mut self, graph: &mut Graph) -> bool {
        let mut changed = false;
        let node_ids: Vec<NodeId> = graph.iter().map(|(id, _)| id).collect();

        for node_id in node_ids {
            if let Some(node) = graph.get(node_id) {
                let mut new_inputs = Vec::with_capacity(node.inputs.len());
                let mut any_changed = false;

                for input in node.inputs.iter() {
                    let resolved = self.resolve_copy(input);
                    if resolved != input {
                        self.stats.uses_rewritten += 1;
                        any_changed = true;
                    }
                    new_inputs.push(resolved);
                }

                if any_changed {
                    if let Some(node_mut) = graph.get_mut(node_id) {
                        node_mut.inputs = InputList::from_slice(&new_inputs);
                    }
                    changed = true;
                }
            }
        }

        // Count eliminated copies
        for &copy in self.copies.keys() {
            if graph.uses(copy).is_empty() {
                self.stats.copies_eliminated += 1;
            }
        }

        changed
    }

    /// Resolve a potentially copied value to its original.
    fn resolve_copy(&self, node: NodeId) -> NodeId {
        self.copies.get(&node).copied().unwrap_or(node)
    }
}

impl Default for CopyProp {
    fn default() -> Self {
        Self::new()
    }
}

impl OptimizationPass for CopyProp {
    fn name(&self) -> &'static str {
        "copy_prop"
    }

    fn run(&mut self, graph: &mut Graph) -> bool {
        self.run_copy_prop(graph)
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::builder::{ArithmeticBuilder, ControlBuilder, GraphBuilder};
    use crate::ir::types::ValueType;

    /// Helper to create a phi node directly in the graph
    fn make_phi(graph: &mut Graph, region: NodeId, values: &[NodeId]) -> NodeId {
        let mut inputs = InputList::default();
        inputs.push(region);
        for &v in values {
            inputs.push(v);
        }
        graph.add_node_with_type(Operator::Phi, inputs, ValueType::Top)
    }

    /// Helper to create a loop phi node
    fn make_loop_phi(
        graph: &mut Graph,
        loop_header: NodeId,
        initial: NodeId,
        back_edge: Option<NodeId>,
    ) -> NodeId {
        let mut inputs = InputList::default();
        inputs.push(loop_header);
        inputs.push(initial);
        if let Some(back) = back_edge {
            inputs.push(back);
        }
        graph.add_node_with_type(Operator::LoopPhi, inputs, ValueType::Int64)
    }

    // =========================================================================
    // CopyPropStats Tests
    // =========================================================================

    #[test]
    fn test_copy_prop_stats_default() {
        let stats = CopyPropStats::default();
        assert_eq!(stats.copies_found, 0);
        assert_eq!(stats.uses_rewritten, 0);
        assert_eq!(stats.copies_eliminated, 0);
        assert_eq!(stats.phis_simplified, 0);
    }

    #[test]
    fn test_copy_prop_stats_clone() {
        let stats = CopyPropStats {
            copies_found: 5,
            uses_rewritten: 10,
            copies_eliminated: 3,
            phis_simplified: 2,
        };
        let cloned = stats.clone();
        assert_eq!(cloned.copies_found, 5);
        assert_eq!(cloned.uses_rewritten, 10);
    }

    // =========================================================================
    // CopyProp Construction Tests
    // =========================================================================

    #[test]
    fn test_copy_prop_new() {
        let pass = CopyProp::new();
        assert_eq!(pass.copies_found(), 0);
        assert_eq!(pass.uses_rewritten(), 0);
        assert!(!pass.aggressive);
    }

    #[test]
    fn test_copy_prop_aggressive() {
        let pass = CopyProp::aggressive();
        assert!(pass.aggressive);
    }

    #[test]
    fn test_copy_prop_default() {
        let pass = CopyProp::default();
        assert!(!pass.aggressive);
    }

    #[test]
    fn test_copy_prop_name() {
        let pass = CopyProp::new();
        assert_eq!(pass.name(), "copy_prop");
    }

    // =========================================================================
    // Empty Graph Tests
    // =========================================================================

    #[test]
    fn test_copy_prop_empty_graph() {
        let builder = GraphBuilder::new(0, 0);
        let mut graph = builder.finish();

        let mut pass = CopyProp::new();
        let changed = pass.run(&mut graph);

        assert!(!changed);
        assert_eq!(pass.copies_found(), 0);
    }

    #[test]
    fn test_copy_prop_no_copies() {
        let mut builder = GraphBuilder::new(4, 2);
        let p0 = builder.parameter(0).unwrap();
        let p1 = builder.parameter(1).unwrap();
        let sum = builder.int_add(p0, p1);
        builder.return_value(sum);

        let mut graph = builder.finish();
        let initial_size = graph.len();

        let mut pass = CopyProp::new();
        let changed = pass.run(&mut graph);

        assert!(!changed);
        assert_eq!(graph.len(), initial_size);
    }

    // =========================================================================
    // Single Input Phi Tests
    // =========================================================================

    #[test]
    fn test_copy_prop_single_input_phi() {
        let mut builder = GraphBuilder::new(8, 2);
        let p0 = builder.parameter(0).unwrap();
        let region = builder.control();

        let mut graph = builder.finish();

        // Create a phi with single value input
        let phi = make_phi(&mut graph, region, &[p0]);

        // Create an add using the phi
        let add = graph.add_node_with_type(
            Operator::IntOp(crate::ir::operators::ArithOp::Add),
            InputList::Pair(phi, phi),
            ValueType::Int64,
        );

        let mut pass = CopyProp::new();
        let changed = pass.run(&mut graph);

        assert!(changed);
        assert_eq!(pass.stats().phis_simplified, 1);

        // Verify the add now uses p0 directly
        let add_node = graph.get(add).unwrap();
        assert!(add_node.inputs.iter().all(|i| i == p0));
    }

    // =========================================================================
    // Uniform Phi Tests
    // =========================================================================

    #[test]
    fn test_copy_prop_uniform_phi() {
        let mut builder = GraphBuilder::new(8, 2);
        let p0 = builder.parameter(0).unwrap();
        let region = builder.control();

        let mut graph = builder.finish();

        // Phi where all value inputs are the same
        let phi = make_phi(&mut graph, region, &[p0, p0, p0]);

        // Use it
        let add = graph.add_node_with_type(
            Operator::IntOp(crate::ir::operators::ArithOp::Add),
            InputList::Pair(phi, phi),
            ValueType::Int64,
        );

        let mut pass = CopyProp::new();
        let changed = pass.run(&mut graph);

        assert!(changed);
        assert_eq!(pass.stats().phis_simplified, 1);

        let add_node = graph.get(add).unwrap();
        assert!(add_node.inputs.iter().all(|i| i == p0));
    }

    // =========================================================================
    // Transitive Chain Tests
    // =========================================================================

    #[test]
    fn test_copy_prop_transitive_chain() {
        let mut builder = GraphBuilder::new(12, 2);
        let p0 = builder.parameter(0).unwrap();
        let region = builder.control();

        let mut graph = builder.finish();

        // Chain: phi1 = p0, phi2 = phi1, phi3 = phi2
        let phi1 = make_phi(&mut graph, region, &[p0]);
        let phi2 = make_phi(&mut graph, region, &[phi1]);
        let phi3 = make_phi(&mut graph, region, &[phi2]);

        // Use phi3
        let add = graph.add_node_with_type(
            Operator::IntOp(crate::ir::operators::ArithOp::Add),
            InputList::Pair(phi3, phi3),
            ValueType::Int64,
        );

        let mut pass = CopyProp::aggressive();
        let changed = pass.run(&mut graph);

        assert!(changed);
        assert_eq!(pass.stats().copies_found, 3);

        // In aggressive mode, add should use p0
        let add_node = graph.get(add).unwrap();
        assert!(add_node.inputs.iter().all(|i| i == p0));
    }

    // =========================================================================
    // LoopPhi Tests
    // =========================================================================

    #[test]
    fn test_copy_prop_trivial_loop_phi() {
        let mut builder = GraphBuilder::new(8, 2);
        let p0 = builder.parameter(0).unwrap();
        let entry = builder.control();
        let loop_head = builder.loop_header(entry);

        let mut graph = builder.finish();

        // Loop phi with only initial value
        let loop_phi = make_loop_phi(&mut graph, loop_head, p0, None);

        // Use it
        let add = graph.add_node_with_type(
            Operator::IntOp(crate::ir::operators::ArithOp::Add),
            InputList::Pair(loop_phi, loop_phi),
            ValueType::Int64,
        );

        let mut pass = CopyProp::new();
        let changed = pass.run(&mut graph);

        assert!(changed);
        assert_eq!(pass.stats().phis_simplified, 1);

        let add_node = graph.get(add).unwrap();
        assert!(add_node.inputs.iter().all(|i| i == p0));
    }

    #[test]
    fn test_copy_prop_loop_phi_same_back_edge() {
        let mut builder = GraphBuilder::new(8, 2);
        let p0 = builder.parameter(0).unwrap();
        let entry = builder.control();
        let loop_head = builder.loop_header(entry);

        let mut graph = builder.finish();

        // Loop phi where initial == back edge
        let loop_phi = make_loop_phi(&mut graph, loop_head, p0, Some(p0));

        let add = graph.add_node_with_type(
            Operator::IntOp(crate::ir::operators::ArithOp::Add),
            InputList::Pair(loop_phi, loop_phi),
            ValueType::Int64,
        );

        let mut pass = CopyProp::new();
        let changed = pass.run(&mut graph);

        assert!(changed);

        let add_node = graph.get(add).unwrap();
        assert!(add_node.inputs.iter().all(|i| i == p0));
    }

    // =========================================================================
    // Multiple Copies Tests
    // =========================================================================

    #[test]
    fn test_copy_prop_multiple_independent_copies() {
        let mut builder = GraphBuilder::new(12, 4);
        let p0 = builder.parameter(0).unwrap();
        let p1 = builder.parameter(1).unwrap();
        let region = builder.control();

        let mut graph = builder.finish();

        let phi_a = make_phi(&mut graph, region, &[p0]);
        let phi_b = make_phi(&mut graph, region, &[p1]);

        let sum = graph.add_node_with_type(
            Operator::IntOp(crate::ir::operators::ArithOp::Add),
            InputList::Pair(phi_a, phi_b),
            ValueType::Int64,
        );

        let mut pass = CopyProp::new();
        let changed = pass.run(&mut graph);

        assert!(changed);
        assert_eq!(pass.stats().copies_found, 2);

        let sum_node = graph.get(sum).unwrap();
        let inputs: Vec<_> = sum_node.inputs.iter().collect();
        assert!(inputs.contains(&p0));
        assert!(inputs.contains(&p1));
    }

    // =========================================================================
    // Preserves Non-Copies Tests
    // =========================================================================

    #[test]
    fn test_copy_prop_preserves_non_copies() {
        let mut builder = GraphBuilder::new(12, 4);
        let p0 = builder.parameter(0).unwrap();
        let p1 = builder.parameter(1).unwrap();
        let region = builder.control();

        let mut graph = builder.finish();

        // Real phi with different value inputs (not a copy)
        let real_phi = make_phi(&mut graph, region, &[p0, p1]);

        // Copy phi
        let copy_phi = make_phi(&mut graph, region, &[p0]);

        let sum = graph.add_node_with_type(
            Operator::IntOp(crate::ir::operators::ArithOp::Add),
            InputList::Pair(real_phi, copy_phi),
            ValueType::Int64,
        );

        let mut pass = CopyProp::new();
        let changed = pass.run(&mut graph);

        assert!(changed);
        assert_eq!(pass.stats().copies_found, 1);

        let sum_node = graph.get(sum).unwrap();
        let inputs: Vec<_> = sum_node.inputs.iter().collect();
        assert!(inputs.contains(&real_phi));
        assert!(inputs.contains(&p0));
    }

    // =========================================================================
    // Edge Case Tests
    // =========================================================================

    #[test]
    fn test_copy_prop_idempotent() {
        let mut builder = GraphBuilder::new(8, 2);
        let p0 = builder.parameter(0).unwrap();
        let region = builder.control();

        let mut graph = builder.finish();

        let phi = make_phi(&mut graph, region, &[p0]);
        graph.add_node_with_type(
            Operator::IntOp(crate::ir::operators::ArithOp::Add),
            InputList::Pair(phi, phi),
            ValueType::Int64,
        );

        // First run
        let mut pass = CopyProp::new();
        let changed1 = pass.run(&mut graph);
        assert!(changed1);

        // Second run should be no-op
        let mut pass2 = CopyProp::new();
        let changed2 = pass2.run(&mut graph);
        assert!(!changed2);
    }

    // =========================================================================
    // Projection Tests
    // =========================================================================

    #[test]
    fn test_copy_prop_projection_single_input() {
        let mut builder = GraphBuilder::new(8, 2);
        let p0 = builder.parameter(0).unwrap();

        let mut graph = builder.finish();

        // Projection(0) with single input
        let projected = graph.add_node_with_type(
            Operator::Projection(0),
            InputList::Single(p0),
            ValueType::Int64,
        );

        let result = graph.add_node_with_type(
            Operator::IntOp(crate::ir::operators::ArithOp::Add),
            InputList::Pair(projected, projected),
            ValueType::Int64,
        );

        let mut pass = CopyProp::new();
        let changed = pass.run(&mut graph);

        assert!(changed);
        assert_eq!(pass.stats().copies_found, 1);

        let result_node = graph.get(result).unwrap();
        assert!(result_node.inputs.iter().all(|i| i == p0));
    }
}
