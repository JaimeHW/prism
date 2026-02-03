//! Loop-Invariant Code Motion (LICM).
//!
//! LICM moves loop-invariant computations out of loops to reduce
//! redundant work. This is one of the most effective loop optimizations.
//!
//! # Algorithm
//!
//! 1. Build loop analysis (headers, bodies, nesting)
//! 2. Compute loop-invariant nodes via fixed-point iteration
//! 3. Filter to hoistable nodes (pure, won't trap, dominates uses)
//! 4. Move hoistable nodes to loop preheader
//!
//! # Example
//!
//! Before LICM:
//! ```text
//! loop:
//!   t1 = a * b      // invariant: a,b don't change in loop
//!   t2 = t1 + i     // NOT invariant: depends on i
//!   i = i + 1
//!   if (i < n) goto loop
//! ```
//!
//! After LICM:
//! ```text
//! preheader:
//!   t1 = a * b      // hoisted!
//! loop:
//!   t2 = t1 + i
//!   i = i + 1
//!   if (i < n) goto loop
//! ```

use super::loop_analyzer::LoopInvariantAnalysis;
use super::OptimizationPass;
use crate::ir::cfg::{BlockId, Cfg, DominatorTree, LoopAnalysis};
use crate::ir::graph::Graph;
use crate::ir::node::{NodeFlags, NodeId};

// =============================================================================
// LICM Pass
// =============================================================================

/// Loop-Invariant Code Motion optimization pass.
#[derive(Debug)]
pub struct Licm {
    /// Number of nodes hoisted.
    hoisted: usize,
    /// Number of loops processed.
    loops_processed: usize,
    /// Enable aggressive hoisting (hoist even if might trap).
    aggressive: bool,
}

impl Licm {
    /// Create a new LICM pass.
    #[inline]
    pub fn new() -> Self {
        Self {
            hoisted: 0,
            loops_processed: 0,
            aggressive: false,
        }
    }

    /// Create an aggressive LICM pass.
    pub fn aggressive() -> Self {
        Self {
            hoisted: 0,
            loops_processed: 0,
            aggressive: true,
        }
    }

    /// Get number of nodes hoisted.
    #[inline]
    pub fn hoisted(&self) -> usize {
        self.hoisted
    }

    /// Get number of loops processed.
    #[inline]
    pub fn loops_processed(&self) -> usize {
        self.loops_processed
    }

    /// Run LICM on the graph.
    fn run_licm(&mut self, graph: &mut Graph) -> bool {
        // Build analyses
        let cfg = Cfg::build(graph);
        let dom = DominatorTree::build(&cfg);
        let loops = LoopAnalysis::compute(&cfg, &dom);

        if loops.loops.is_empty() {
            return false;
        }

        // Compute invariants
        let invariant_analysis = LoopInvariantAnalysis::compute(graph, &cfg, &dom, &loops);

        let mut changed = false;

        // Process loops from innermost to outermost
        // (inner loops first ensures their invariants are hoisted before
        // we process outer loops)
        let mut loop_order: Vec<usize> = (0..loops.loops.len()).collect();
        loop_order.sort_by_key(|&i| std::cmp::Reverse(loops.loops[i].depth));

        for loop_idx in loop_order {
            let loop_info = &loops.loops[loop_idx];

            // Collect hoistable nodes in order
            let mut to_hoist: Vec<NodeId> = invariant_analysis.hoistable_nodes(loop_idx).collect();

            if to_hoist.is_empty() {
                continue;
            }

            // Sort by dependency order (inputs before users)
            to_hoist = Self::topological_sort(graph, to_hoist);

            // Hoist each node
            for node_id in to_hoist {
                if self.hoist_node(graph, node_id, loop_info.header) {
                    self.hoisted += 1;
                    changed = true;
                }
            }

            self.loops_processed += 1;
        }

        changed
    }

    /// Sort nodes in topological order (dependencies first).
    fn topological_sort(graph: &Graph, nodes: Vec<NodeId>) -> Vec<NodeId> {
        use std::collections::HashSet;

        let node_set: HashSet<_> = nodes.iter().copied().collect();
        let mut sorted = Vec::with_capacity(nodes.len());
        let mut visited = HashSet::new();
        let mut temp_mark = HashSet::new();

        fn visit(
            graph: &Graph,
            node: NodeId,
            node_set: &HashSet<NodeId>,
            sorted: &mut Vec<NodeId>,
            visited: &mut HashSet<NodeId>,
            temp_mark: &mut HashSet<NodeId>,
        ) {
            if visited.contains(&node) {
                return;
            }
            if temp_mark.contains(&node) {
                // Cycle detected - shouldn't happen in DAG
                return;
            }

            temp_mark.insert(node);

            // Visit inputs that are also in our set
            let n = graph.node(node);
            for input in n.inputs.iter() {
                if node_set.contains(&input) {
                    visit(graph, input, node_set, sorted, visited, temp_mark);
                }
            }

            temp_mark.remove(&node);
            visited.insert(node);
            sorted.push(node);
        }

        for node in nodes {
            visit(
                graph,
                node,
                &node_set,
                &mut sorted,
                &mut visited,
                &mut temp_mark,
            );
        }

        sorted
    }

    /// Hoist a single node.
    ///
    /// In Sea-of-Nodes, "hoisting" means scheduling the node to execute
    /// before the loop. This is done by:
    ///
    /// 1. Marking the node with LOOP_INVARIANT and HOISTED flags
    /// 2. Removing the IN_LOOP flag if present
    /// 3. The scheduler uses these flags to place the node in the preheader
    ///
    /// # Arguments
    ///
    /// * `graph` - The IR graph to modify
    /// * `node_id` - The node to hoist
    /// * `_header` - The loop header block (for future control input updates)
    ///
    /// # Returns
    ///
    /// `true` if the node was successfully hoisted, `false` otherwise
    fn hoist_node(&mut self, graph: &mut Graph, node_id: NodeId, _header: BlockId) -> bool {
        // Verify node still exists and is valid
        let node = match graph.get_mut(node_id) {
            Some(n) => n,
            None => return false,
        };

        // Skip if already hoisted
        if node.flags.contains(NodeFlags::HOISTED) {
            return false;
        }

        // Skip if pinned (cannot be moved)
        if node.flags.contains(NodeFlags::PINNED) {
            return false;
        }

        // Skip if dead
        if node.flags.contains(NodeFlags::DEAD) {
            return false;
        }

        // Mark as loop invariant and hoisted
        // The scheduler will use these flags to place the node before the loop
        node.flags.insert(NodeFlags::LOOP_INVARIANT);
        node.flags.insert(NodeFlags::HOISTED);

        // Remove IN_LOOP flag since node is now conceptually outside the loop
        node.flags.remove(NodeFlags::IN_LOOP);

        true
    }
}

impl Default for Licm {
    fn default() -> Self {
        Self::new()
    }
}

impl OptimizationPass for Licm {
    fn name(&self) -> &'static str {
        "licm"
    }

    fn run(&mut self, graph: &mut Graph) -> bool {
        self.run_licm(graph)
    }
}

// =============================================================================
// LICM Statistics
// =============================================================================

/// Detailed statistics from LICM.
#[derive(Debug, Clone, Default)]
pub struct LicmStats {
    /// Number of nodes hoisted.
    pub nodes_hoisted: usize,
    /// Number of loops analyzed.
    pub loops_analyzed: usize,
    /// Number of loops with hoistable code.
    pub loops_with_hoists: usize,
    /// Number of invariant nodes found (but not all hoisted).
    pub invariant_nodes: usize,
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::builder::{ArithmeticBuilder, ControlBuilder, GraphBuilder};

    #[test]
    fn test_licm_new() {
        let licm = Licm::new();
        assert_eq!(licm.hoisted(), 0);
        assert_eq!(licm.loops_processed(), 0);
        assert!(!licm.aggressive);
    }

    #[test]
    fn test_licm_aggressive() {
        let licm = Licm::aggressive();
        assert!(licm.aggressive);
    }

    #[test]
    fn test_licm_name() {
        let licm = Licm::new();
        assert_eq!(licm.name(), "licm");
    }

    #[test]
    fn test_licm_no_loops() {
        let mut builder = GraphBuilder::new(2, 2);

        let p0 = builder.parameter(0).unwrap();
        let p1 = builder.parameter(1).unwrap();
        let sum = builder.int_add(p0, p1);
        builder.return_value(sum);

        let mut graph = builder.finish();
        let mut licm = Licm::new();

        let changed = licm.run(&mut graph);
        assert!(!changed);
        assert_eq!(licm.hoisted(), 0);
    }

    #[test]
    fn test_topological_sort_empty() {
        let builder = GraphBuilder::new(0, 0);
        let graph = builder.finish();

        let sorted = Licm::topological_sort(&graph, vec![]);
        assert!(sorted.is_empty());
    }

    #[test]
    fn test_topological_sort_single() {
        let mut builder = GraphBuilder::new(2, 2);
        let p0 = builder.parameter(0).unwrap();
        let graph = builder.finish();

        let sorted = Licm::topological_sort(&graph, vec![p0]);
        assert_eq!(sorted.len(), 1);
        assert_eq!(sorted[0], p0);
    }

    #[test]
    fn test_topological_sort_chain() {
        let mut builder = GraphBuilder::new(3, 2);
        let p0 = builder.parameter(0).unwrap();
        let p1 = builder.parameter(1).unwrap();
        let sum = builder.int_add(p0, p1);
        let graph = builder.finish();

        // Sort should put p0, p1 before sum
        let sorted = Licm::topological_sort(&graph, vec![sum, p0, p1]);
        assert_eq!(sorted.len(), 3);
        // sum should be last (depends on p0, p1)
        assert_eq!(sorted[2], sum);
    }

    #[test]
    fn test_licm_stats_default() {
        let stats = LicmStats::default();
        assert_eq!(stats.nodes_hoisted, 0);
        assert_eq!(stats.loops_analyzed, 0);
    }

    // -------------------------------------------------------------------------
    // hoist_node Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_hoist_node_sets_flags() {
        let mut builder = GraphBuilder::new(2, 2);
        let p0 = builder.parameter(0).unwrap();
        let p1 = builder.parameter(1).unwrap();
        let sum = builder.int_add(p0, p1);
        let mut graph = builder.finish();

        // Initially no flags
        assert!(!graph.node(sum).flags.contains(NodeFlags::LOOP_INVARIANT));
        assert!(!graph.node(sum).flags.contains(NodeFlags::HOISTED));

        let mut licm = Licm::new();
        let header = BlockId::new(0);
        let hoisted = licm.hoist_node(&mut graph, sum, header);

        assert!(hoisted);
        assert!(graph.node(sum).flags.contains(NodeFlags::LOOP_INVARIANT));
        assert!(graph.node(sum).flags.contains(NodeFlags::HOISTED));
    }

    #[test]
    fn test_hoist_node_removes_in_loop_flag() {
        let mut builder = GraphBuilder::new(2, 2);
        let p0 = builder.parameter(0).unwrap();
        let p1 = builder.parameter(1).unwrap();
        let sum = builder.int_add(p0, p1);
        let mut graph = builder.finish();

        // Simulate node being inside a loop
        graph.node_mut(sum).flags.insert(NodeFlags::IN_LOOP);
        assert!(graph.node(sum).flags.contains(NodeFlags::IN_LOOP));

        let mut licm = Licm::new();
        let header = BlockId::new(0);
        let hoisted = licm.hoist_node(&mut graph, sum, header);

        assert!(hoisted);
        assert!(!graph.node(sum).flags.contains(NodeFlags::IN_LOOP));
    }

    #[test]
    fn test_hoist_node_skips_already_hoisted() {
        let mut builder = GraphBuilder::new(2, 2);
        let p0 = builder.parameter(0).unwrap();
        let p1 = builder.parameter(1).unwrap();
        let sum = builder.int_add(p0, p1);
        let mut graph = builder.finish();

        // Mark as already hoisted
        graph.node_mut(sum).flags.insert(NodeFlags::HOISTED);

        let mut licm = Licm::new();
        let header = BlockId::new(0);
        let hoisted = licm.hoist_node(&mut graph, sum, header);

        // Should return false since already hoisted
        assert!(!hoisted);
    }

    #[test]
    fn test_hoist_node_skips_pinned() {
        let mut builder = GraphBuilder::new(2, 2);
        let p0 = builder.parameter(0).unwrap();
        let p1 = builder.parameter(1).unwrap();
        let sum = builder.int_add(p0, p1);
        let mut graph = builder.finish();

        // Mark as pinned (cannot be moved)
        graph.node_mut(sum).flags.insert(NodeFlags::PINNED);

        let mut licm = Licm::new();
        let header = BlockId::new(0);
        let hoisted = licm.hoist_node(&mut graph, sum, header);

        // Should return false since pinned
        assert!(!hoisted);
        assert!(!graph.node(sum).flags.contains(NodeFlags::HOISTED));
    }

    #[test]
    fn test_hoist_node_skips_dead() {
        let mut builder = GraphBuilder::new(2, 2);
        let p0 = builder.parameter(0).unwrap();
        let p1 = builder.parameter(1).unwrap();
        let sum = builder.int_add(p0, p1);
        let mut graph = builder.finish();

        // Mark as dead
        graph.node_mut(sum).flags.insert(NodeFlags::DEAD);

        let mut licm = Licm::new();
        let header = BlockId::new(0);
        let hoisted = licm.hoist_node(&mut graph, sum, header);

        // Should return false since dead
        assert!(!hoisted);
    }
}
