//! Anticipation Analysis for PRE.
//!
//! An expression is **anticipated** at a point if it will definitely
//! be computed on every path from that point to the program exit.
//!
//! Uses backward data flow analysis.

use rustc_hash::FxHashSet;

use crate::ir::graph::Graph;
use crate::ir::node::NodeId;
use crate::ir::operators::{ControlOp, Operator};

use super::{ExprId, ExpressionTable};

// =============================================================================
// Anticipation Analysis
// =============================================================================

/// Result of anticipation analysis.
#[derive(Debug)]
pub struct AnticipationAnalysis {
    /// Expressions anticipated at each node.
    antic_at: Vec<FxHashSet<ExprId>>,
    /// Number of nodes analyzed.
    #[allow(dead_code)]
    node_count: usize,
    /// Empty set for out-of-bounds access.
    empty: FxHashSet<ExprId>,
}

impl AnticipationAnalysis {
    /// Compute anticipation for the graph.
    pub fn compute(graph: &Graph, expr_table: &ExpressionTable) -> Self {
        let node_count = graph.len();
        let mut analysis = Self {
            antic_at: vec![FxHashSet::default(); node_count],
            node_count,
            empty: FxHashSet::default(),
        };

        analysis.analyze(graph, expr_table);
        analysis
    }

    /// Run backward dataflow analysis.
    fn analyze(&mut self, graph: &Graph, expr_table: &ExpressionTable) {
        // Initialize: expressions are anticipated where they're defined
        for i in 0..graph.len() {
            let node_id = NodeId::new(i as u32);
            if let Some(expr_id) = expr_table.get_expr_id(node_id) {
                self.antic_at[i].insert(expr_id);
            }
        }

        // Backward propagation until fixpoint
        let mut changed = true;
        let mut iteration = 0;
        const MAX_ITERATIONS: usize = 100;

        while changed && iteration < MAX_ITERATIONS {
            changed = false;
            iteration += 1;

            // Process nodes in reverse order (rough backward traversal)
            for i in (0..graph.len()).rev() {
                let node_id = NodeId::new(i as u32);

                // Get uses (successors in data flow sense)
                let uses = graph.uses(node_id);
                if uses.is_empty() {
                    continue;
                }

                // Anticipation propagates backward:
                // If an expression is anticipated at ALL successors, it's anticipated here
                let old_size = self.antic_at[i].len();

                // Get intersection of successor anticipation sets
                let mut first = true;
                let mut intersection = FxHashSet::default();

                for &succ_id in uses {
                    let succ_idx = succ_id.index() as usize;
                    if succ_idx < self.antic_at.len() {
                        if first {
                            intersection = self.antic_at[succ_idx].clone();
                            first = false;
                        } else {
                            intersection.retain(|e| self.antic_at[succ_idx].contains(e));
                        }
                    }
                }

                // Check for control flow that blocks anticipation
                if let Some(node) = graph.get(node_id) {
                    if Self::blocks_anticipation(&node.op) {
                        // Control flow nodes don't propagate anticipation
                        intersection.clear();
                    }
                }

                // Merge intersection into current set
                for expr_id in intersection {
                    if self.antic_at[i].insert(expr_id) {
                        changed = true;
                    }
                }

                if self.antic_at[i].len() != old_size {
                    changed = true;
                }
            }
        }
    }

    /// Check if an operator blocks anticipation propagation.
    fn blocks_anticipation(op: &Operator) -> bool {
        matches!(
            op,
            Operator::Control(ControlOp::If)
                | Operator::Control(ControlOp::Throw)
                | Operator::Control(ControlOp::Deopt)
        )
    }

    /// Check if an expression is anticipated at a node.
    pub fn is_anticipated(&self, node: NodeId, expr: ExprId) -> bool {
        let idx = node.index() as usize;
        if idx < self.antic_at.len() {
            self.antic_at[idx].contains(&expr)
        } else {
            false
        }
    }

    /// Get all anticipated expressions at a node.
    pub fn anticipated_at(&self, node: NodeId) -> &FxHashSet<ExprId> {
        let idx = node.index() as usize;
        if idx < self.antic_at.len() {
            &self.antic_at[idx]
        } else {
            &self.empty
        }
    }

    /// Get the number of nodes with anticipated expressions.
    pub fn nodes_with_anticipation(&self) -> usize {
        self.antic_at.iter().filter(|s| !s.is_empty()).count()
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
    fn test_anticipation_empty_graph() {
        let graph = Graph::new();
        let expr_table = ExpressionTable::build(&graph);
        let antic = AnticipationAnalysis::compute(&graph, &expr_table);

        assert_eq!(antic.nodes_with_anticipation(), 0);
    }

    #[test]
    fn test_anticipation_single_constant() {
        let mut graph = Graph::new();
        let c = graph.const_int(42);

        let expr_table = ExpressionTable::build(&graph);
        let antic = AnticipationAnalysis::compute(&graph, &expr_table);

        // The constant should be anticipated where it's defined
        let expr_id = expr_table.get_expr_id(c).unwrap();
        assert!(antic.is_anticipated(c, expr_id));
    }

    #[test]
    fn test_anticipation_arithmetic() {
        let mut graph = Graph::new();
        let a = graph.const_int(1);
        let b = graph.const_int(2);
        let sum = graph.add_node(Operator::IntOp(ArithOp::Add), InputList::Pair(a, b));

        let expr_table = ExpressionTable::build(&graph);
        let antic = AnticipationAnalysis::compute(&graph, &expr_table);

        // Sum should be anticipated at its definition
        let sum_expr = expr_table.get_expr_id(sum).unwrap();
        assert!(antic.is_anticipated(sum, sum_expr));
    }

    #[test]
    fn test_anticipation_not_anticipated_wrong_node() {
        let mut graph = Graph::new();
        let a = graph.const_int(1);
        let b = graph.const_int(2);

        let expr_table = ExpressionTable::build(&graph);
        let antic = AnticipationAnalysis::compute(&graph, &expr_table);

        // Expression for 'a' should not be anticipated at 'b'
        let a_expr = expr_table.get_expr_id(a).unwrap();
        assert!(!antic.is_anticipated(b, a_expr));
    }

    #[test]
    fn test_anticipation_nodes_with_anticipation() {
        let mut graph = Graph::new();
        graph.const_int(1);
        graph.const_int(2);

        let expr_table = ExpressionTable::build(&graph);
        let antic = AnticipationAnalysis::compute(&graph, &expr_table);

        assert!(antic.nodes_with_anticipation() >= 2);
    }

    #[test]
    fn test_anticipation_at_empty() {
        let graph = Graph::new();
        let expr_table = ExpressionTable::build(&graph);
        let antic = AnticipationAnalysis::compute(&graph, &expr_table);

        let set = antic.anticipated_at(NodeId::new(0));
        assert!(set.is_empty());
    }

    #[test]
    fn test_anticipation_out_of_bounds() {
        let graph = Graph::new();
        let expr_table = ExpressionTable::build(&graph);
        let antic = AnticipationAnalysis::compute(&graph, &expr_table);

        // Out of bounds node should return empty
        assert!(!antic.is_anticipated(NodeId::new(1000), ExprId::new(0)));
    }

    #[test]
    fn test_blocks_anticipation_if() {
        assert!(AnticipationAnalysis::blocks_anticipation(
            &Operator::Control(ControlOp::If)
        ));
    }

    #[test]
    fn test_blocks_anticipation_throw() {
        assert!(AnticipationAnalysis::blocks_anticipation(
            &Operator::Control(ControlOp::Throw)
        ));
    }

    #[test]
    fn test_blocks_anticipation_arithmetic() {
        assert!(!AnticipationAnalysis::blocks_anticipation(
            &Operator::IntOp(ArithOp::Add)
        ));
    }

    #[test]
    fn test_anticipated_at_returns_set() {
        let mut graph = Graph::new();
        let c = graph.const_int(42);

        let expr_table = ExpressionTable::build(&graph);
        let antic = AnticipationAnalysis::compute(&graph, &expr_table);

        let set = antic.anticipated_at(c);
        assert!(!set.is_empty());
    }
}
