//! Interference Graph
//!
//! Builds and manages the interference graph for register allocation.
//! Two virtual registers interfere if their live ranges overlap.
//!
//! # Data Structures
//!
//! - Adjacency matrix (sparse) for O(1) interference queries
//! - Adjacency list for O(degree) neighbor enumeration
//!
//! # Usage
//!
//! ```ignore
//! let igraph = InterferenceGraph::build(&intervals, num_vregs);
//! if igraph.interferes(v1, v2) {
//!     // Cannot allocate to same register
//! }
//! ```

use super::interval::LiveInterval;
use super::{PReg, VReg};
use std::collections::{HashMap, HashSet};

// =============================================================================
// Interference Graph
// =============================================================================

/// The interference graph for register allocation.
///
/// Uses a compressed representation for memory efficiency:
/// - Adjacency sets for each node
/// - Move edges stored separately for coalescing
#[derive(Debug)]
pub struct InterferenceGraph {
    /// Adjacency sets for each vreg.
    adjacency: HashMap<VReg, HashSet<VReg>>,
    /// Move edges (source, dest) - candidate for coalescing.
    move_edges: Vec<(VReg, VReg)>,
    /// Pre-colored vregs (fixed to specific physical registers).
    precolored: HashMap<VReg, PReg>,
    /// Degree of each node.
    degrees: HashMap<VReg, u32>,
    /// Number of allocatable GPR registers.
    k_gpr: u32,
    /// Number of allocatable XMM registers (128-bit).
    k_xmm: u32,
    /// Number of allocatable YMM registers (256-bit).
    k_ymm: u32,
    /// Number of allocatable ZMM registers (512-bit).
    k_zmm: u32,
}

impl InterferenceGraph {
    /// Create a new empty interference graph.
    ///
    /// # Parameters
    /// - `k_gpr`: Number of allocatable GPR registers
    /// - `k_xmm`: Number of allocatable XMM registers (128-bit)
    /// - `k_ymm`: Number of allocatable YMM registers (256-bit)
    /// - `k_zmm`: Number of allocatable ZMM registers (512-bit)
    pub fn new(k_gpr: u32, k_xmm: u32, k_ymm: u32, k_zmm: u32) -> Self {
        InterferenceGraph {
            adjacency: HashMap::new(),
            move_edges: Vec::new(),
            precolored: HashMap::new(),
            degrees: HashMap::new(),
            k_gpr,
            k_xmm,
            k_ymm,
            k_zmm,
        }
    }

    /// Create a new interference graph with legacy 2-parameter API.
    ///
    /// Uses default values for YMM (16) and ZMM (32) counts.
    #[inline]
    pub fn new_legacy(k_gpr: u32, k_xmm: u32) -> Self {
        Self::new(k_gpr, k_xmm, 16, 32)
    }

    /// Build an interference graph from live intervals.
    ///
    /// # Parameters
    /// - `intervals`: Live intervals to build the graph from
    /// - `k_gpr`: Number of allocatable GPR registers
    /// - `k_xmm`: Number of allocatable XMM registers (128-bit)
    /// - `k_ymm`: Number of allocatable YMM registers (256-bit)
    /// - `k_zmm`: Number of allocatable ZMM registers (512-bit)
    pub fn build(
        intervals: &[LiveInterval],
        k_gpr: u32,
        k_xmm: u32,
        k_ymm: u32,
        k_zmm: u32,
    ) -> Self {
        let mut graph = InterferenceGraph::new(k_gpr, k_xmm, k_ymm, k_zmm);

        // Ensure all vregs have entries
        for interval in intervals {
            graph.adjacency.entry(interval.vreg).or_default();
            graph.degrees.entry(interval.vreg).or_insert(0);
        }

        // O(n²) pairwise comparison - can be optimized with interval tree
        // For now, use a sweep-line algorithm for better performance
        graph.build_with_sweep(intervals);

        graph
    }

    /// Build an interference graph using legacy 2-parameter API.
    ///
    /// Uses default values for YMM (16) and ZMM (32) counts.
    #[inline]
    pub fn build_legacy(intervals: &[LiveInterval], k_gpr: u32, k_xmm: u32) -> Self {
        Self::build(intervals, k_gpr, k_xmm, 16, 32)
    }

    /// Build using sweep-line algorithm for O(n log n) performance.
    fn build_with_sweep(&mut self, intervals: &[LiveInterval]) {
        if intervals.is_empty() {
            return;
        }

        // Create events for interval starts and ends
        #[derive(Debug, Clone, Copy, PartialEq, Eq)]
        enum EventKind {
            Start,
            End,
        }

        let mut events: Vec<(u32, EventKind, usize)> = Vec::new();

        for (idx, interval) in intervals.iter().enumerate() {
            if interval.is_empty() {
                continue;
            }

            if let (Some(first), Some(last)) = (interval.first_range(), interval.last_range()) {
                events.push((first.start.raw(), EventKind::Start, idx));
                events.push((last.end.raw(), EventKind::End, idx));
            }
        }

        // Sort: starts before ends at the same position
        events.sort_by(|a, b| {
            a.0.cmp(&b.0).then_with(|| match (&a.1, &b.1) {
                (EventKind::Start, EventKind::End) => std::cmp::Ordering::Less,
                (EventKind::End, EventKind::Start) => std::cmp::Ordering::Greater,
                _ => std::cmp::Ordering::Equal,
            })
        });

        // Sweep through events
        let mut active: Vec<usize> = Vec::new();

        for (_, kind, idx) in events {
            match kind {
                EventKind::Start => {
                    // Add interference edges with all currently active intervals
                    let vreg = intervals[idx].vreg;

                    for &other_idx in &active {
                        let other_vreg = intervals[other_idx].vreg;

                        // Must check actual overlap since we're using ranges
                        if intervals[idx].overlaps(&intervals[other_idx]) {
                            self.add_edge(vreg, other_vreg);
                        }
                    }

                    active.push(idx);
                }
                EventKind::End => {
                    // Remove from active set
                    if let Some(pos) = active.iter().position(|&x| x == idx) {
                        active.swap_remove(pos);
                    }
                }
            }
        }
    }

    /// Add an interference edge between two vregs.
    pub fn add_edge(&mut self, v1: VReg, v2: VReg) {
        if v1 == v2 {
            return;
        }

        // Add to adjacency lists
        let added1 = self.adjacency.entry(v1).or_default().insert(v2);
        let added2 = self.adjacency.entry(v2).or_default().insert(v1);

        // Update degrees if this is a new edge
        if added1 {
            *self.degrees.entry(v1).or_insert(0) += 1;
        }
        if added2 {
            *self.degrees.entry(v2).or_insert(0) += 1;
        }
    }

    /// Remove an interference edge.
    pub fn remove_edge(&mut self, v1: VReg, v2: VReg) {
        if let Some(adj) = self.adjacency.get_mut(&v1) {
            if adj.remove(&v2) {
                if let Some(d) = self.degrees.get_mut(&v1) {
                    *d = d.saturating_sub(1);
                }
            }
        }
        if let Some(adj) = self.adjacency.get_mut(&v2) {
            if adj.remove(&v1) {
                if let Some(d) = self.degrees.get_mut(&v2) {
                    *d = d.saturating_sub(1);
                }
            }
        }
    }

    /// Add a move edge (candidate for coalescing).
    pub fn add_move_edge(&mut self, from: VReg, to: VReg) {
        if from != to {
            self.move_edges.push((from, to));
        }
    }

    /// Mark a vreg as pre-colored.
    pub fn set_precolored(&mut self, vreg: VReg, preg: PReg) {
        self.precolored.insert(vreg, preg);
    }

    /// Check if two vregs interfere.
    #[inline]
    pub fn interferes(&self, v1: VReg, v2: VReg) -> bool {
        if v1 == v2 {
            return true; // A vreg interferes with itself
        }
        self.adjacency
            .get(&v1)
            .map(|adj| adj.contains(&v2))
            .unwrap_or(false)
    }

    /// Get the neighbors of a vreg.
    pub fn neighbors(&self, vreg: VReg) -> impl Iterator<Item = VReg> + '_ {
        self.adjacency
            .get(&vreg)
            .into_iter()
            .flat_map(|adj| adj.iter().copied())
    }

    /// Get the degree of a vreg.
    #[inline]
    pub fn degree(&self, vreg: VReg) -> u32 {
        self.degrees.get(&vreg).copied().unwrap_or(0)
    }

    /// Check if a vreg is pre-colored.
    pub fn is_precolored(&self, vreg: VReg) -> bool {
        self.precolored.contains_key(&vreg)
    }

    /// Get the pre-colored register for a vreg.
    pub fn get_precolor(&self, vreg: VReg) -> Option<PReg> {
        self.precolored.get(&vreg).copied()
    }

    /// Get the number of colors (registers) for a register class.
    ///
    /// Returns the number of allocatable registers based on register class:
    /// - `RegClass::Int` / `RegClass::Any`: GPR count
    /// - `RegClass::Float`: XMM count
    /// - `RegClass::Vec256`: YMM count
    /// - `RegClass::Vec512`: ZMM count
    pub fn k_for_class(&self, reg_class: super::RegClass) -> u32 {
        match reg_class {
            super::RegClass::Int | super::RegClass::Any => self.k_gpr,
            super::RegClass::Float => self.k_xmm,
            super::RegClass::Vec256 => self.k_ymm,
            super::RegClass::Vec512 => self.k_zmm,
        }
    }

    /// Get the number of colors (registers) for float vs int.
    ///
    /// Legacy API - prefer `k_for_class()` for full vector support.
    #[inline]
    pub fn k(&self, is_float: bool) -> u32 {
        if is_float { self.k_xmm } else { self.k_gpr }
    }

    /// Get the number of allocatable GPR registers.
    #[inline]
    pub fn k_gpr(&self) -> u32 {
        self.k_gpr
    }

    /// Get the number of allocatable XMM registers.
    #[inline]
    pub fn k_xmm(&self) -> u32 {
        self.k_xmm
    }

    /// Get the number of allocatable YMM registers.
    #[inline]
    pub fn k_ymm(&self) -> u32 {
        self.k_ymm
    }

    /// Get the number of allocatable ZMM registers.
    #[inline]
    pub fn k_zmm(&self) -> u32 {
        self.k_zmm
    }

    /// Get all move edges.
    pub fn move_edges(&self) -> &[(VReg, VReg)] {
        &self.move_edges
    }

    /// Get all vregs.
    pub fn vregs(&self) -> impl Iterator<Item = VReg> + '_ {
        self.adjacency.keys().copied()
    }

    /// Get the number of vregs.
    pub fn len(&self) -> usize {
        self.adjacency.len()
    }

    /// Check if the graph is empty.
    pub fn is_empty(&self) -> bool {
        self.adjacency.is_empty()
    }

    /// Merge two vregs (for coalescing).
    /// The second vreg is merged into the first.
    pub fn coalesce(&mut self, keep: VReg, remove: VReg) {
        // Get neighbors of the removed vreg
        let neighbors: Vec<VReg> = self.neighbors(remove).collect();

        // Add edges from keep to all neighbors of remove
        for neighbor in neighbors {
            if neighbor != keep {
                self.remove_edge(remove, neighbor);
                self.add_edge(keep, neighbor);
            }
        }

        // Remove the vreg
        self.adjacency.remove(&remove);
        self.degrees.remove(&remove);

        // Transfer precolor if remove was precolored
        if let Some(preg) = self.precolored.remove(&remove) {
            self.precolored.insert(keep, preg);
        }

        // Update move edges
        self.move_edges
            .retain(|&(from, to)| from != remove && to != remove);
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::super::RegClass;
    use super::super::interval::{LiveRange, ProgPoint};
    use super::*;

    fn make_interval(vreg: u32, start: u32, end: u32) -> LiveInterval {
        let mut interval = LiveInterval::new(VReg::new(vreg), RegClass::Int);
        interval.add_range(LiveRange::new(
            ProgPoint::before(start),
            ProgPoint::before(end),
        ));
        interval
    }

    #[test]
    fn test_no_interference() {
        // [0, 10) and [20, 30) don't overlap
        let intervals = vec![make_interval(0, 0, 10), make_interval(1, 20, 30)];

        let graph = InterferenceGraph::build_legacy(&intervals, 14, 15);

        assert!(!graph.interferes(VReg::new(0), VReg::new(1)));
    }

    #[test]
    fn test_interference() {
        // [0, 20) and [10, 30) overlap
        let intervals = vec![make_interval(0, 0, 20), make_interval(1, 10, 30)];

        let graph = InterferenceGraph::build_legacy(&intervals, 14, 15);

        assert!(graph.interferes(VReg::new(0), VReg::new(1)));
        assert!(graph.interferes(VReg::new(1), VReg::new(0)));
    }

    #[test]
    fn test_degree() {
        // v0: [0, 30), v1: [10, 20), v2: [25, 35)
        // v0 interferes with v1 and v2
        // v1 interferes with v0 only
        // v2 interferes with v0 only
        let intervals = vec![
            make_interval(0, 0, 30),
            make_interval(1, 10, 20),
            make_interval(2, 25, 35),
        ];

        let graph = InterferenceGraph::build_legacy(&intervals, 14, 15);

        assert_eq!(graph.degree(VReg::new(0)), 2);
        assert_eq!(graph.degree(VReg::new(1)), 1);
        assert_eq!(graph.degree(VReg::new(2)), 1);
    }

    #[test]
    fn test_coalesce() {
        let intervals = vec![
            make_interval(0, 0, 30),
            make_interval(1, 10, 20),
            make_interval(2, 25, 35),
        ];

        let mut graph = InterferenceGraph::build_legacy(&intervals, 14, 15);

        // Coalesce v1 into v0
        graph.coalesce(VReg::new(0), VReg::new(1));

        // v1 should be gone
        assert!(!graph.adjacency.contains_key(&VReg::new(1)));

        // v0 should still interfere with v2
        assert!(graph.interferes(VReg::new(0), VReg::new(2)));
    }

    // =========================================================================
    // Vector Register Class Tests
    // =========================================================================

    #[test]
    fn test_vector_register_counts_full_api() {
        // Test with explicit YMM and ZMM counts
        let graph = InterferenceGraph::new(14, 16, 15, 31);

        assert_eq!(graph.k_gpr(), 14);
        assert_eq!(graph.k_xmm(), 16);
        assert_eq!(graph.k_ymm(), 15);
        assert_eq!(graph.k_zmm(), 31);
    }

    #[test]
    fn test_vector_register_counts_legacy_api() {
        // Legacy API should provide default YMM=16, ZMM=32
        let graph = InterferenceGraph::new_legacy(14, 16);

        assert_eq!(graph.k_gpr(), 14);
        assert_eq!(graph.k_xmm(), 16);
        assert_eq!(graph.k_ymm(), 16); // Default
        assert_eq!(graph.k_zmm(), 32); // Default
    }

    #[test]
    fn test_k_for_class_int() {
        let graph = InterferenceGraph::new(14, 16, 15, 31);

        assert_eq!(graph.k_for_class(RegClass::Int), 14);
        assert_eq!(graph.k_for_class(RegClass::Any), 14);
    }

    #[test]
    fn test_k_for_class_float() {
        let graph = InterferenceGraph::new(14, 16, 15, 31);

        assert_eq!(graph.k_for_class(RegClass::Float), 16);
    }

    #[test]
    fn test_k_for_class_vec256() {
        let graph = InterferenceGraph::new(14, 16, 15, 31);

        assert_eq!(graph.k_for_class(RegClass::Vec256), 15);
    }

    #[test]
    fn test_k_for_class_vec512() {
        let graph = InterferenceGraph::new(14, 16, 15, 31);

        assert_eq!(graph.k_for_class(RegClass::Vec512), 31);
    }

    #[test]
    fn test_legacy_k_method_matches_k_for_class() {
        let graph = InterferenceGraph::new(14, 16, 15, 31);

        // Legacy k() method should match GPR/XMM
        assert_eq!(graph.k(false), graph.k_for_class(RegClass::Int));
        assert_eq!(graph.k(true), graph.k_for_class(RegClass::Float));
    }

    #[test]
    fn test_build_with_vector_counts() {
        let intervals = vec![make_interval(0, 0, 10)];

        let graph = InterferenceGraph::build(&intervals, 13, 15, 14, 30);

        assert_eq!(graph.k_gpr(), 13);
        assert_eq!(graph.k_xmm(), 15);
        assert_eq!(graph.k_ymm(), 14);
        assert_eq!(graph.k_zmm(), 30);
    }

    #[test]
    fn test_build_legacy_with_default_vector_counts() {
        let intervals = vec![make_interval(0, 0, 10)];

        let graph = InterferenceGraph::build_legacy(&intervals, 13, 15);

        assert_eq!(graph.k_gpr(), 13);
        assert_eq!(graph.k_xmm(), 15);
        assert_eq!(graph.k_ymm(), 16); // Default
        assert_eq!(graph.k_zmm(), 32); // Default
    }

    #[test]
    fn test_all_register_classes_accessible() {
        let graph = InterferenceGraph::new(14, 16, 15, 31);

        // Verify all register classes return correct values
        let test_cases = [
            (RegClass::Int, 14),
            (RegClass::Any, 14),
            (RegClass::Float, 16),
            (RegClass::Vec256, 15),
            (RegClass::Vec512, 31),
        ];

        for (class, expected) in test_cases {
            assert_eq!(
                graph.k_for_class(class),
                expected,
                "k_for_class({:?}) should return {}",
                class,
                expected
            );
        }
    }
}
