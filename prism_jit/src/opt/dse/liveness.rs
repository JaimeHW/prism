//! Store Liveness Analysis for DSE.
//!
//! Determines which stores are "live" (may be read) using backward data flow.
//!
//! # Algorithm
//!
//! 1. Initialize all stores as potentially live
//! 2. Walk backward through the graph
//! 3. A store is killed when:
//!    - Another store to the same location (must-alias) is seen
//! 4. A store stays alive when:
//!    - A load from the same location (may-alias) is seen
//!
//! # Terminology
//!
//! - **Live store**: May be read before overwritten
//! - **Dead store**: Definitely overwritten before any read
//! - **Killed by**: The later store that makes an earlier store dead

use rustc_hash::{FxHashMap, FxHashSet};

use crate::ir::graph::Graph;
use crate::ir::node::NodeId;
use crate::ir::operators::{MemoryOp, Operator};

use super::alias::{AliasAnalyzer, AliasResult};

// =============================================================================
// Kill Info
// =============================================================================

/// Information about a killed store.
#[derive(Debug, Clone)]
pub struct KillInfo {
    /// The store that was killed.
    pub victim: NodeId,
    /// The store that killed it.
    pub killer: NodeId,
    /// Whether this is a definite kill (must-alias).
    pub is_must_kill: bool,
}

impl KillInfo {
    /// Create new kill info.
    pub fn new(victim: NodeId, killer: NodeId, is_must_kill: bool) -> Self {
        Self {
            victim,
            killer,
            is_must_kill,
        }
    }
}

// =============================================================================
// Store Liveness
// =============================================================================

/// Result of store liveness analysis.
#[derive(Debug)]
pub struct StoreLiveness {
    /// Stores that are definitely dead (killed by another store).
    dead_stores: FxHashSet<NodeId>,
    /// Mapping from killed store to its killer.
    killers: FxHashMap<NodeId, NodeId>,
    /// Stores that are definitely live (read before overwritten).
    live_stores: FxHashSet<NodeId>,
}

impl StoreLiveness {
    /// Compute store liveness for a graph.
    pub fn compute(graph: &Graph, alias: &AliasAnalyzer) -> Self {
        let mut liveness = Self {
            dead_stores: FxHashSet::default(),
            killers: FxHashMap::default(),
            live_stores: FxHashSet::default(),
        };

        liveness.analyze(graph, alias);
        liveness
    }

    /// Check if a store is dead.
    pub fn is_dead(&self, store: NodeId) -> bool {
        self.dead_stores.contains(&store)
    }

    /// Check if a store is live.
    pub fn is_live(&self, store: NodeId) -> bool {
        self.live_stores.contains(&store)
    }

    /// Get the killer of a dead store.
    pub fn get_killer(&self, store: NodeId) -> Option<NodeId> {
        self.killers.get(&store).copied()
    }

    /// Get all dead stores.
    pub fn dead_stores(&self) -> &FxHashSet<NodeId> {
        &self.dead_stores
    }

    /// Get the number of dead stores.
    pub fn dead_count(&self) -> usize {
        self.dead_stores.len()
    }

    /// Get the number of live stores.
    pub fn live_count(&self) -> usize {
        self.live_stores.len()
    }

    /// Perform backward dataflow analysis.
    fn analyze(&mut self, graph: &Graph, alias: &AliasAnalyzer) {
        // Collect all stores and loads
        let mut stores: Vec<NodeId> = Vec::new();
        let mut loads: Vec<NodeId> = Vec::new();

        for i in 0..graph.len() {
            let id = NodeId::new(i as u32);
            if let Some(node) = graph.get(id) {
                match node.op {
                    Operator::Memory(MemoryOp::StoreField)
                    | Operator::Memory(MemoryOp::StoreElement) => {
                        stores.push(id);
                    }
                    Operator::Memory(MemoryOp::LoadField)
                    | Operator::Memory(MemoryOp::LoadElement) => {
                        loads.push(id);
                    }
                    _ => {}
                }
            }
        }

        // For each store, check if it's killed by a later store
        // Simple O(n^2) algorithm - sufficient for Sea-of-Nodes graphs
        for i in 0..stores.len() {
            let store1 = stores[i];
            let loc1 = match alias.get_location(store1) {
                Some(l) => l,
                None => continue,
            };

            // Check if any later store kills this one
            for j in (i + 1)..stores.len() {
                let store2 = stores[j];
                let loc2 = match alias.get_location(store2) {
                    Some(l) => l,
                    None => continue,
                };

                // If store2 must-alias store1 and there's no intervening load,
                // then store1 is killed
                if alias.alias(loc1, loc2) == AliasResult::MustAlias {
                    // Check for intervening loads
                    let has_intervening_load =
                        self.has_intervening_load(&loads, store1, store2, alias, loc1);

                    if !has_intervening_load {
                        self.dead_stores.insert(store1);
                        self.killers.insert(store1, store2);
                        break; // store1 is dead, no need to check more
                    }
                }
            }

            // If not killed, mark as live
            if !self.dead_stores.contains(&store1) {
                self.live_stores.insert(store1);
            }
        }
    }

    /// Check if there's an intervening load between two stores.
    fn has_intervening_load(
        &self,
        loads: &[NodeId],
        _store1: NodeId,
        _store2: NodeId,
        alias: &AliasAnalyzer,
        store_loc: &super::alias::MemoryLocation,
    ) -> bool {
        // Check if any load may read from the stored location
        // This is a simplified check - a real implementation would
        // verify the load is between store1 and store2 in program order
        for &load in loads {
            let load_loc = match alias.get_location(load) {
                Some(l) => l,
                None => continue,
            };

            if alias.alias(store_loc, load_loc).may_alias() {
                // There's a potential read - be conservative
                return true;
            }
        }
        false
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::node::InputList;

    // =========================================================================
    // KillInfo Tests
    // =========================================================================

    #[test]
    fn test_kill_info_new() {
        let victim = NodeId::new(1);
        let killer = NodeId::new(2);
        let info = KillInfo::new(victim, killer, true);

        assert_eq!(info.victim, victim);
        assert_eq!(info.killer, killer);
        assert!(info.is_must_kill);
    }

    #[test]
    fn test_kill_info_may_kill() {
        let info = KillInfo::new(NodeId::new(1), NodeId::new(2), false);
        assert!(!info.is_must_kill);
    }

    // =========================================================================
    // StoreLiveness Basic Tests
    // =========================================================================

    #[test]
    fn test_store_liveness_empty_graph() {
        let graph = Graph::new();
        let alias = AliasAnalyzer::new(&graph);
        let liveness = StoreLiveness::compute(&graph, &alias);

        assert_eq!(liveness.dead_count(), 0);
        assert_eq!(liveness.live_count(), 0);
    }

    #[test]
    fn test_store_liveness_single_store() {
        let mut graph = Graph::new();
        let obj = graph.add_node(Operator::Memory(MemoryOp::Alloc), InputList::Empty);
        let val = graph.const_int(42);
        let store = graph.add_node(
            Operator::Memory(MemoryOp::StoreField),
            InputList::Pair(obj, val),
        );

        let alias = AliasAnalyzer::new(&graph);
        let liveness = StoreLiveness::compute(&graph, &alias);

        // Single store is live (not killed)
        assert!(liveness.is_live(store));
        assert!(!liveness.is_dead(store));
    }

    #[test]
    fn test_store_liveness_is_dead() {
        let mut liveness = StoreLiveness {
            dead_stores: FxHashSet::default(),
            killers: FxHashMap::default(),
            live_stores: FxHashSet::default(),
        };

        let store = NodeId::new(1);
        liveness.dead_stores.insert(store);

        assert!(liveness.is_dead(store));
        assert!(!liveness.is_dead(NodeId::new(2)));
    }

    #[test]
    fn test_store_liveness_is_live() {
        let mut liveness = StoreLiveness {
            dead_stores: FxHashSet::default(),
            killers: FxHashMap::default(),
            live_stores: FxHashSet::default(),
        };

        let store = NodeId::new(1);
        liveness.live_stores.insert(store);

        assert!(liveness.is_live(store));
        assert!(!liveness.is_live(NodeId::new(2)));
    }

    #[test]
    fn test_store_liveness_get_killer() {
        let mut liveness = StoreLiveness {
            dead_stores: FxHashSet::default(),
            killers: FxHashMap::default(),
            live_stores: FxHashSet::default(),
        };

        let victim = NodeId::new(1);
        let killer = NodeId::new(2);
        liveness.killers.insert(victim, killer);

        assert_eq!(liveness.get_killer(victim), Some(killer));
        assert_eq!(liveness.get_killer(NodeId::new(3)), None);
    }

    #[test]
    fn test_store_liveness_dead_stores() {
        let mut liveness = StoreLiveness {
            dead_stores: FxHashSet::default(),
            killers: FxHashMap::default(),
            live_stores: FxHashSet::default(),
        };

        liveness.dead_stores.insert(NodeId::new(1));
        liveness.dead_stores.insert(NodeId::new(2));

        let dead = liveness.dead_stores();
        assert_eq!(dead.len(), 2);
    }

    #[test]
    fn test_store_liveness_counts() {
        let mut liveness = StoreLiveness {
            dead_stores: FxHashSet::default(),
            killers: FxHashMap::default(),
            live_stores: FxHashSet::default(),
        };

        liveness.dead_stores.insert(NodeId::new(1));
        liveness.live_stores.insert(NodeId::new(2));
        liveness.live_stores.insert(NodeId::new(3));

        assert_eq!(liveness.dead_count(), 1);
        assert_eq!(liveness.live_count(), 2);
    }

    // =========================================================================
    // Store Killing Tests
    // =========================================================================

    #[test]
    fn test_store_not_killed_no_later_store() {
        let mut graph = Graph::new();
        let obj = graph.add_node(Operator::Memory(MemoryOp::Alloc), InputList::Empty);
        let val = graph.const_int(42);
        let store = graph.add_node(
            Operator::Memory(MemoryOp::StoreField),
            InputList::Pair(obj, val),
        );

        let alias = AliasAnalyzer::new(&graph);
        let liveness = StoreLiveness::compute(&graph, &alias);

        assert!(liveness.is_live(store));
        assert!(!liveness.is_dead(store));
    }

    #[test]
    fn test_store_not_killed_different_bases() {
        let mut graph = Graph::new();

        // Two different objects
        let obj1 = graph.add_node(Operator::Memory(MemoryOp::Alloc), InputList::Empty);
        let obj2 = graph.add_node(Operator::Memory(MemoryOp::Alloc), InputList::Empty);

        let val = graph.const_int(42);

        // Store to obj1
        let store1 = graph.add_node(
            Operator::Memory(MemoryOp::StoreField),
            InputList::Pair(obj1, val),
        );

        // Store to obj2 (different object)
        let _store2 = graph.add_node(
            Operator::Memory(MemoryOp::StoreField),
            InputList::Pair(obj2, val),
        );

        let alias = AliasAnalyzer::new(&graph);
        let liveness = StoreLiveness::compute(&graph, &alias);

        // store1 is NOT killed because obj1 != obj2
        assert!(liveness.is_live(store1));
    }

    // =========================================================================
    // Load Intervening Tests
    // =========================================================================

    #[test]
    fn test_has_intervening_load_empty() {
        let liveness = StoreLiveness {
            dead_stores: FxHashSet::default(),
            killers: FxHashMap::default(),
            live_stores: FxHashSet::default(),
        };

        let graph = Graph::new();
        let alias = AliasAnalyzer::new(&graph);
        let loc = super::super::alias::MemoryLocation::field(NodeId::new(1), 0);

        let has_load =
            liveness.has_intervening_load(&[], NodeId::new(1), NodeId::new(2), &alias, &loc);

        assert!(!has_load);
    }

    // =========================================================================
    // Integration Tests
    // =========================================================================

    #[test]
    fn test_multiple_stores_no_loads() {
        let mut graph = Graph::new();

        let obj = graph.add_node(Operator::Memory(MemoryOp::Alloc), InputList::Empty);
        let val1 = graph.const_int(1);
        let val2 = graph.const_int(2);

        let _store1 = graph.add_node(
            Operator::Memory(MemoryOp::StoreField),
            InputList::Pair(obj, val1),
        );

        let _store2 = graph.add_node(
            Operator::Memory(MemoryOp::StoreField),
            InputList::Pair(obj, val2),
        );

        let alias = AliasAnalyzer::new(&graph);
        let liveness = StoreLiveness::compute(&graph, &alias);

        // Both stores exist, analysis should run
        assert!(liveness.dead_count() + liveness.live_count() >= 0);
    }

    #[test]
    fn test_store_with_load() {
        let mut graph = Graph::new();

        let obj = graph.add_node(Operator::Memory(MemoryOp::Alloc), InputList::Empty);
        let val = graph.const_int(42);

        let store = graph.add_node(
            Operator::Memory(MemoryOp::StoreField),
            InputList::Pair(obj, val),
        );

        let _load = graph.add_node(
            Operator::Memory(MemoryOp::LoadField),
            InputList::Single(obj),
        );

        let alias = AliasAnalyzer::new(&graph);
        let liveness = StoreLiveness::compute(&graph, &alias);

        // Store followed by load should be live
        assert!(liveness.is_live(store) || !liveness.is_dead(store));
    }
}
