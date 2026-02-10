//! Availability Analysis for PRE.
//!
//! An expression is **available** at a point if it has been computed
//! on every path from the program entry to that point.
//!
//! Uses forward data flow analysis.

use rustc_hash::FxHashSet;

use crate::ir::graph::Graph;
use crate::ir::node::NodeId;
use crate::ir::operators::{ControlOp, MemoryOp, Operator};

use super::{ExprId, ExpressionTable};

// =============================================================================
// Availability Analysis
// =============================================================================

/// Result of availability analysis.
#[derive(Debug)]
pub struct AvailabilityAnalysis {
    /// Expressions available at each node.
    avail_at: Vec<FxHashSet<ExprId>>,
    /// Number of nodes analyzed.
    #[allow(dead_code)]
    node_count: usize,
    /// Empty set for out-of-bounds access.
    empty: FxHashSet<ExprId>,
}

impl AvailabilityAnalysis {
    /// Compute availability for the graph.
    pub fn compute(graph: &Graph, expr_table: &ExpressionTable) -> Self {
        let node_count = graph.len();
        let mut analysis = Self {
            avail_at: vec![FxHashSet::default(); node_count],
            node_count,
            empty: FxHashSet::default(),
        };

        analysis.analyze(graph, expr_table);
        analysis
    }

    /// Run forward dataflow analysis.
    fn analyze(&mut self, graph: &Graph, expr_table: &ExpressionTable) {
        // Initialize: entry has nothing available
        // (nodes 0 and 1 are typically Start and End)

        // Forward propagation until fixpoint
        let mut changed = true;
        let mut iteration = 0;
        const MAX_ITERATIONS: usize = 100;

        while changed && iteration < MAX_ITERATIONS {
            changed = false;
            iteration += 1;

            // Process nodes in forward order
            for i in 0..graph.len() {
                let node_id = NodeId::new(i as u32);

                if let Some(node) = graph.get(node_id) {
                    let old_size = self.avail_at[i].len();

                    // Availability propagates forward from inputs
                    // Expression is available if available at ALL inputs (predecessors)
                    let mut first = true;
                    let mut intersection = FxHashSet::default();
                    let inputs: Vec<_> = node.inputs.iter().collect();

                    // Compute intersection of available sets from all inputs
                    for &input in &inputs {
                        let input_idx = input.index() as usize;
                        if input_idx < self.avail_at.len() {
                            if first {
                                intersection = self.avail_at[input_idx].clone();
                                first = false;
                            } else {
                                intersection.retain(|e| self.avail_at[input_idx].contains(e));
                            }
                        }
                    }

                    // Check for operations that kill availability
                    if Self::kills_availability(&node.op) {
                        // Side effects kill some availability
                        // For simplicity, clear all for memory ops
                        intersection.clear();
                    }

                    // Add expressions defined by each input (they're computed before us)
                    for &input in &inputs {
                        if let Some(input_expr) = expr_table.get_expr_id(input) {
                            intersection.insert(input_expr);
                        }
                    }

                    // Add expression defined at this node (becomes available after)
                    if let Some(expr_id) = expr_table.get_expr_id(node_id) {
                        intersection.insert(expr_id);
                    }

                    // Merge into current set
                    for expr_id in intersection {
                        if self.avail_at[i].insert(expr_id) {
                            changed = true;
                        }
                    }

                    if self.avail_at[i].len() != old_size {
                        changed = true;
                    }
                }
            }
        }
    }

    /// Check if an operator kills availability of expressions.
    fn kills_availability(op: &Operator) -> bool {
        matches!(
            op,
            Operator::Memory(MemoryOp::StoreField)
                | Operator::Memory(MemoryOp::StoreElement)
                | Operator::Control(ControlOp::Throw)
                | Operator::Control(ControlOp::Deopt)
        )
    }

    /// Check if an expression is available at a node.
    pub fn is_available(&self, node: NodeId, expr: ExprId) -> bool {
        let idx = node.index() as usize;
        if idx < self.avail_at.len() {
            self.avail_at[idx].contains(&expr)
        } else {
            false
        }
    }

    /// Get all available expressions at a node.
    pub fn available_at(&self, node: NodeId) -> &FxHashSet<ExprId> {
        let idx = node.index() as usize;
        if idx < self.avail_at.len() {
            &self.avail_at[idx]
        } else {
            &self.empty
        }
    }

    /// Get the number of nodes with available expressions.
    pub fn nodes_with_availability(&self) -> usize {
        self.avail_at.iter().filter(|s| !s.is_empty()).count()
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::node::InputList;
    use crate::ir::operators::ArithOp;

    #[test]
    fn test_availability_empty_graph() {
        let graph = Graph::new();
        let expr_table = ExpressionTable::build(&graph);
        let avail = AvailabilityAnalysis::compute(&graph, &expr_table);

        assert_eq!(avail.nodes_with_availability(), 0);
    }

    #[test]
    fn test_availability_single_constant() {
        let mut graph = Graph::new();
        let c = graph.const_int(42);

        let expr_table = ExpressionTable::build(&graph);
        let avail = AvailabilityAnalysis::compute(&graph, &expr_table);

        // The constant should be available where it's defined
        let expr_id = expr_table.get_expr_id(c).unwrap();
        assert!(avail.is_available(c, expr_id));
    }

    #[test]
    fn test_availability_arithmetic() {
        let mut graph = Graph::new();
        let a = graph.const_int(1);
        let b = graph.const_int(2);
        let sum = graph.add_node(Operator::IntOp(ArithOp::Add), InputList::Pair(a, b));

        let expr_table = ExpressionTable::build(&graph);
        let avail = AvailabilityAnalysis::compute(&graph, &expr_table);

        // Sum should be available at its definition
        let sum_expr = expr_table.get_expr_id(sum).unwrap();
        assert!(avail.is_available(sum, sum_expr));
    }

    #[test]
    fn test_availability_propagates_forward() {
        let mut graph = Graph::new();
        let a = graph.const_int(1);
        let b = graph.const_int(2);
        let sum = graph.add_node(Operator::IntOp(ArithOp::Add), InputList::Pair(a, b));

        let expr_table = ExpressionTable::build(&graph);
        let avail = AvailabilityAnalysis::compute(&graph, &expr_table);

        // Constants should be available at the sum node
        let a_expr = expr_table.get_expr_id(a).unwrap();
        let b_expr = expr_table.get_expr_id(b).unwrap();

        assert!(avail.is_available(sum, a_expr));
        assert!(avail.is_available(sum, b_expr));
    }

    #[test]
    fn test_availability_nodes_with_availability() {
        let mut graph = Graph::new();
        graph.const_int(1);
        graph.const_int(2);

        let expr_table = ExpressionTable::build(&graph);
        let avail = AvailabilityAnalysis::compute(&graph, &expr_table);

        assert!(avail.nodes_with_availability() >= 2);
    }

    #[test]
    fn test_availability_at_empty() {
        let graph = Graph::new();
        let expr_table = ExpressionTable::build(&graph);
        let avail = AvailabilityAnalysis::compute(&graph, &expr_table);

        let set = avail.available_at(NodeId::new(0));
        assert!(set.is_empty());
    }

    #[test]
    fn test_availability_out_of_bounds() {
        let graph = Graph::new();
        let expr_table = ExpressionTable::build(&graph);
        let avail = AvailabilityAnalysis::compute(&graph, &expr_table);

        assert!(!avail.is_available(NodeId::new(1000), ExprId::new(0)));
    }

    #[test]
    fn test_kills_availability_store() {
        assert!(AvailabilityAnalysis::kills_availability(&Operator::Memory(
            MemoryOp::StoreField
        )));
    }

    #[test]
    fn test_kills_availability_throw() {
        assert!(AvailabilityAnalysis::kills_availability(
            &Operator::Control(ControlOp::Throw)
        ));
    }

    #[test]
    fn test_kills_availability_arithmetic() {
        assert!(!AvailabilityAnalysis::kills_availability(&Operator::IntOp(
            ArithOp::Add
        )));
    }

    #[test]
    fn test_kills_availability_load() {
        // Loads don't kill availability
        assert!(!AvailabilityAnalysis::kills_availability(
            &Operator::Memory(MemoryOp::LoadField)
        ));
    }

    #[test]
    fn test_available_at_returns_set() {
        let mut graph = Graph::new();
        let c = graph.const_int(42);

        let expr_table = ExpressionTable::build(&graph);
        let avail = AvailabilityAnalysis::compute(&graph, &expr_table);

        let set = avail.available_at(c);
        assert!(!set.is_empty());
    }
}
