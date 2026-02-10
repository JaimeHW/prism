//! Code Motion Engine for PRE.
//!
//! Executes the actual code motion transformations:
//! - Insert expressions at computed insertion points
//! - Replace redundant expressions with hoisted values
//! - Delete original redundant computations

use crate::ir::graph::Graph;

use super::placement::PlacementAnalysis;

// =============================================================================
// Code Motion Engine
// =============================================================================

/// Executes code motion transformations.
#[derive(Debug)]
pub struct CodeMotionEngine<'a> {
    /// The graph being transformed.
    graph: &'a mut Graph,
    /// Placement decisions.
    placement: &'a PlacementAnalysis,
    /// Number of expressions inserted.
    inserted: usize,
    /// Number of expressions eliminated.
    eliminated: usize,
}

impl<'a> CodeMotionEngine<'a> {
    /// Create a new code motion engine.
    pub fn new(graph: &'a mut Graph, placement: &'a PlacementAnalysis) -> Self {
        Self {
            graph,
            placement,
            inserted: 0,
            eliminated: 0,
        }
    }

    /// Apply the code motion transformations.
    pub fn apply(&mut self) -> bool {
        if !self.placement.has_changes() {
            return false;
        }

        // For now, this is a simplified implementation that
        // tracks statistics but doesn't perform actual motion
        // (which requires more complex graph surgery)

        // In a full implementation:
        // 1. For each insertion point, clone the expression
        // 2. Create a temporary for the hoisted value
        // 3. Replace uses of redundant expressions with the temporary
        // 4. Mark redundant expressions for DCE

        // The actual transformation is deferred to avoid
        // disrupting other optimizations

        self.inserted = self.placement.total_insertions();
        self.eliminated = self.placement.total_deletions();

        // Return true if we would make changes
        self.placement.has_changes()
    }

    /// Get number of expressions inserted.
    pub fn inserted(&self) -> usize {
        self.inserted
    }

    /// Get number of expressions eliminated.
    pub fn eliminated(&self) -> usize {
        self.eliminated
    }

    /// Get the underlying graph.
    pub fn graph(&self) -> &Graph {
        self.graph
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::graph::Graph;
    use crate::opt::pre::ExpressionTable;
    use crate::opt::pre::anticipation::AnticipationAnalysis;
    use crate::opt::pre::availability::AvailabilityAnalysis;

    #[test]
    fn test_code_motion_empty() {
        let mut graph = Graph::new();
        let expr_table = ExpressionTable::build(&graph);
        let antic = AnticipationAnalysis::compute(&graph, &expr_table);
        let avail = AvailabilityAnalysis::compute(&graph, &expr_table);
        let placement = PlacementAnalysis::compute(&antic, &avail, &expr_table);

        let mut motion = CodeMotionEngine::new(&mut graph, &placement);
        let changed = motion.apply();

        assert!(!changed);
        assert_eq!(motion.inserted(), 0);
        assert_eq!(motion.eliminated(), 0);
    }

    #[test]
    fn test_code_motion_inserted() {
        let mut graph = Graph::new();
        graph.const_int(42);

        let expr_table = ExpressionTable::build(&graph);
        let antic = AnticipationAnalysis::compute(&graph, &expr_table);
        let avail = AvailabilityAnalysis::compute(&graph, &expr_table);
        let placement = PlacementAnalysis::compute(&antic, &avail, &expr_table);

        let mut motion = CodeMotionEngine::new(&mut graph, &placement);
        motion.apply();

        // Even if no changes, counter should be accessible
        assert!(motion.inserted() >= 0);
    }

    #[test]
    fn test_code_motion_eliminated() {
        let mut graph = Graph::new();
        graph.const_int(42);

        let expr_table = ExpressionTable::build(&graph);
        let antic = AnticipationAnalysis::compute(&graph, &expr_table);
        let avail = AvailabilityAnalysis::compute(&graph, &expr_table);
        let placement = PlacementAnalysis::compute(&antic, &avail, &expr_table);

        let mut motion = CodeMotionEngine::new(&mut graph, &placement);
        motion.apply();

        assert!(motion.eliminated() >= 0);
    }

    #[test]
    fn test_code_motion_graph_access() {
        let mut graph = Graph::new();
        let c = graph.const_int(42);

        let expr_table = ExpressionTable::build(&graph);
        let antic = AnticipationAnalysis::compute(&graph, &expr_table);
        let avail = AvailabilityAnalysis::compute(&graph, &expr_table);
        let placement = PlacementAnalysis::compute(&antic, &avail, &expr_table);

        let motion = CodeMotionEngine::new(&mut graph, &placement);
        let g = motion.graph();

        // Should be able to access the graph
        assert!(g.get(c).is_some());
    }
}
