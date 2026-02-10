//! Worklist for efficient instruction combining.
//!
//! The worklist uses a VecDeque for FIFO ordering with deduplication
//! to avoid processing the same instruction multiple times.

use rustc_hash::FxHashSet;
use std::collections::VecDeque;

use crate::ir::node::NodeId;

// =============================================================================
// Worklist
// =============================================================================

/// Worklist for instruction combining.
#[derive(Debug)]
pub struct Worklist {
    /// The queue of nodes to process.
    queue: VecDeque<NodeId>,
    /// Set of nodes currently in the queue (for deduplication).
    in_queue: FxHashSet<NodeId>,
    /// Total nodes ever added.
    total_added: usize,
    /// Total nodes processed (popped).
    total_processed: usize,
}

impl Worklist {
    /// Create a new empty worklist.
    pub fn new() -> Self {
        Self {
            queue: VecDeque::new(),
            in_queue: FxHashSet::default(),
            total_added: 0,
            total_processed: 0,
        }
    }

    /// Create a worklist with initial capacity.
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            queue: VecDeque::with_capacity(capacity),
            in_queue: FxHashSet::default(),
            total_added: 0,
            total_processed: 0,
        }
    }

    /// Push a node onto the worklist.
    /// Returns true if the node was added (not already present).
    pub fn push(&mut self, node: NodeId) -> bool {
        if self.in_queue.insert(node) {
            self.queue.push_back(node);
            self.total_added += 1;
            true
        } else {
            false
        }
    }

    /// Push multiple nodes onto the worklist.
    pub fn push_all(&mut self, nodes: impl IntoIterator<Item = NodeId>) {
        for node in nodes {
            self.push(node);
        }
    }

    /// Pop the next node from the worklist.
    pub fn pop(&mut self) -> Option<NodeId> {
        while let Some(node) = self.queue.pop_front() {
            self.in_queue.remove(&node);
            self.total_processed += 1;
            return Some(node);
        }
        None
    }

    /// Check if the worklist is empty.
    pub fn is_empty(&self) -> bool {
        self.queue.is_empty()
    }

    /// Get the current size of the worklist.
    pub fn len(&self) -> usize {
        self.queue.len()
    }

    /// Check if a node is in the worklist.
    pub fn contains(&self, node: NodeId) -> bool {
        self.in_queue.contains(&node)
    }

    /// Get total nodes ever added.
    pub fn total_added(&self) -> usize {
        self.total_added
    }

    /// Get total nodes processed.
    pub fn total_processed(&self) -> usize {
        self.total_processed
    }

    /// Clear the worklist.
    pub fn clear(&mut self) {
        self.queue.clear();
        self.in_queue.clear();
    }
}

impl Default for Worklist {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_worklist_new() {
        let wl = Worklist::new();
        assert!(wl.is_empty());
        assert_eq!(wl.len(), 0);
    }

    #[test]
    fn test_worklist_with_capacity() {
        let wl = Worklist::with_capacity(100);
        assert!(wl.is_empty());
    }

    #[test]
    fn test_worklist_push() {
        let mut wl = Worklist::new();
        let added = wl.push(NodeId::new(1));
        assert!(added);
        assert_eq!(wl.len(), 1);
        assert!(!wl.is_empty());
    }

    #[test]
    fn test_worklist_push_dedup() {
        let mut wl = Worklist::new();
        wl.push(NodeId::new(1));
        let added = wl.push(NodeId::new(1));
        assert!(!added); // Duplicate
        assert_eq!(wl.len(), 1);
    }

    #[test]
    fn test_worklist_push_all() {
        let mut wl = Worklist::new();
        wl.push_all([NodeId::new(1), NodeId::new(2), NodeId::new(3)]);
        assert_eq!(wl.len(), 3);
    }

    #[test]
    fn test_worklist_push_all_dedup() {
        let mut wl = Worklist::new();
        wl.push_all([NodeId::new(1), NodeId::new(1), NodeId::new(2)]);
        assert_eq!(wl.len(), 2);
    }

    #[test]
    fn test_worklist_pop() {
        let mut wl = Worklist::new();
        wl.push(NodeId::new(1));
        wl.push(NodeId::new(2));

        let first = wl.pop();
        assert_eq!(first, Some(NodeId::new(1)));
        assert_eq!(wl.len(), 1);

        let second = wl.pop();
        assert_eq!(second, Some(NodeId::new(2)));
        assert!(wl.is_empty());
    }

    #[test]
    fn test_worklist_pop_empty() {
        let mut wl = Worklist::new();
        assert_eq!(wl.pop(), None);
    }

    #[test]
    fn test_worklist_fifo_order() {
        let mut wl = Worklist::new();
        wl.push(NodeId::new(1));
        wl.push(NodeId::new(2));
        wl.push(NodeId::new(3));

        assert_eq!(wl.pop(), Some(NodeId::new(1)));
        assert_eq!(wl.pop(), Some(NodeId::new(2)));
        assert_eq!(wl.pop(), Some(NodeId::new(3)));
    }

    #[test]
    fn test_worklist_contains() {
        let mut wl = Worklist::new();
        wl.push(NodeId::new(1));

        assert!(wl.contains(NodeId::new(1)));
        assert!(!wl.contains(NodeId::new(2)));
    }

    #[test]
    fn test_worklist_contains_after_pop() {
        let mut wl = Worklist::new();
        wl.push(NodeId::new(1));
        wl.pop();

        assert!(!wl.contains(NodeId::new(1)));
    }

    #[test]
    fn test_worklist_total_added() {
        let mut wl = Worklist::new();
        wl.push(NodeId::new(1));
        wl.push(NodeId::new(2));
        wl.push(NodeId::new(1)); // Dup - not added

        assert_eq!(wl.total_added(), 2);
    }

    #[test]
    fn test_worklist_total_processed() {
        let mut wl = Worklist::new();
        wl.push(NodeId::new(1));
        wl.push(NodeId::new(2));
        wl.pop();

        assert_eq!(wl.total_processed(), 1);
    }

    #[test]
    fn test_worklist_clear() {
        let mut wl = Worklist::new();
        wl.push(NodeId::new(1));
        wl.push(NodeId::new(2));
        wl.clear();

        assert!(wl.is_empty());
        assert!(!wl.contains(NodeId::new(1)));
    }

    #[test]
    fn test_worklist_default() {
        let wl = Worklist::default();
        assert!(wl.is_empty());
    }

    #[test]
    fn test_worklist_repush_after_pop() {
        let mut wl = Worklist::new();
        wl.push(NodeId::new(1));
        wl.pop();

        // Should be able to add again after pop
        let added = wl.push(NodeId::new(1));
        assert!(added);
        assert_eq!(wl.len(), 1);
    }

    #[test]
    fn test_worklist_stress() {
        let mut wl = Worklist::new();

        // Add many nodes
        for i in 0..1000 {
            wl.push(NodeId::new(i));
        }
        assert_eq!(wl.len(), 1000);

        // Process all
        while wl.pop().is_some() {}

        assert!(wl.is_empty());
        assert_eq!(wl.total_added(), 1000);
        assert_eq!(wl.total_processed(), 1000);
    }
}
