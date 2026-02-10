//! Dead Store Elimination (DSE) optimization pass.
//!
//! DSE removes stores that are never read:
//! - Stores overwritten before any read
//! - Stores to dead allocations
//! - Stores on dead paths
//!
//! # Algorithm
//!
//! Uses backward data-flow analysis:
//! 1. Build alias analysis for memory locations
//! 2. Compute store liveness (which stores may be read)
//! 3. Remove dead stores
//!
//! # Example
//!
//! ```text
//! x.field = 1    // Dead store (overwritten below)
//! x.field = 2    // Kept
//! return x.field // Read
//! ```

mod alias;
mod liveness;

pub use alias::{AliasAnalyzer, AliasResult, MemOffset, MemoryLocation};
pub use liveness::{KillInfo, StoreLiveness};

use crate::ir::graph::Graph;
use crate::ir::node::NodeId;
use crate::ir::operators::{MemoryOp, Operator};
use crate::opt::OptimizationPass;

// =============================================================================
// DSE Statistics
// =============================================================================

/// Detailed statistics from DSE.
#[derive(Debug, Clone, Default)]
pub struct DseStats {
    /// Stores killed by later stores to same location.
    pub redundant_stores: usize,
    /// Stores to allocations that are never read.
    pub dead_allocation_stores: usize,
    /// Stores on paths where value is never used.
    pub partial_dead_stores: usize,
    /// Total stores analyzed.
    pub stores_analyzed: usize,
}

impl DseStats {
    /// Total stores eliminated.
    pub fn total_eliminated(&self) -> usize {
        self.redundant_stores + self.dead_allocation_stores + self.partial_dead_stores
    }

    /// Merge statistics.
    pub fn merge(&mut self, other: &DseStats) {
        self.redundant_stores += other.redundant_stores;
        self.dead_allocation_stores += other.dead_allocation_stores;
        self.partial_dead_stores += other.partial_dead_stores;
        self.stores_analyzed += other.stores_analyzed;
    }
}

// =============================================================================
// DSE Configuration
// =============================================================================

/// Configuration for DSE.
#[derive(Debug, Clone)]
pub struct DseConfig {
    /// Enable must-alias based elimination.
    pub enable_must_alias_dse: bool,
    /// Enable dead allocation store removal.
    pub enable_dead_alloc_dse: bool,
    /// Maximum iterations for fixpoint.
    pub max_iterations: usize,
}

impl Default for DseConfig {
    fn default() -> Self {
        Self {
            enable_must_alias_dse: true,
            enable_dead_alloc_dse: true,
            max_iterations: 10,
        }
    }
}

// =============================================================================
// DSE Pass
// =============================================================================

/// Dead Store Elimination optimization pass.
#[derive(Debug)]
pub struct Dse {
    /// Configuration.
    config: DseConfig,
    /// Statistics from last run.
    stats: DseStats,
    /// Number of stores removed.
    removed: usize,
}

impl Dse {
    /// Create a new DSE pass.
    pub fn new() -> Self {
        Self {
            config: DseConfig::default(),
            stats: DseStats::default(),
            removed: 0,
        }
    }

    /// Create with custom configuration.
    pub fn with_config(config: DseConfig) -> Self {
        Self {
            config,
            stats: DseStats::default(),
            removed: 0,
        }
    }

    /// Get statistics from last run.
    pub fn stats(&self) -> &DseStats {
        &self.stats
    }

    /// Get number of stores removed.
    pub fn removed(&self) -> usize {
        self.removed
    }

    /// Check if an operator is a store operation.
    fn is_store(op: &Operator) -> bool {
        matches!(
            op,
            Operator::Memory(MemoryOp::StoreField) | Operator::Memory(MemoryOp::StoreElement)
        )
    }

    /// Check if an operator is a load operation.
    fn is_load(op: &Operator) -> bool {
        matches!(
            op,
            Operator::Memory(MemoryOp::LoadField) | Operator::Memory(MemoryOp::LoadElement)
        )
    }

    /// Collect all store nodes in the graph.
    fn collect_stores(&self, graph: &Graph) -> Vec<NodeId> {
        let mut stores = Vec::new();
        for i in 0..graph.len() {
            let id = NodeId::new(i as u32);
            if let Some(node) = graph.get(id) {
                if Self::is_store(&node.op) {
                    stores.push(id);
                }
            }
        }
        stores
    }

    /// Collect all load nodes in the graph.
    fn collect_loads(&self, graph: &Graph) -> Vec<NodeId> {
        let mut loads = Vec::new();
        for i in 0..graph.len() {
            let id = NodeId::new(i as u32);
            if let Some(node) = graph.get(id) {
                if Self::is_load(&node.op) {
                    loads.push(id);
                }
            }
        }
        loads
    }

    /// Run DSE analysis and elimination.
    fn run_dse(&mut self, graph: &mut Graph) -> bool {
        self.stats = DseStats::default();
        self.removed = 0;

        let stores = self.collect_stores(graph);
        self.stats.stores_analyzed = stores.len();

        if stores.is_empty() {
            return false;
        }

        // Build alias analyzer
        let alias = AliasAnalyzer::new(graph);

        // Build store liveness
        let liveness = StoreLiveness::compute(graph, &alias);

        // Find dead stores
        let mut dead_stores = Vec::new();

        for &store in &stores {
            if let Some(killer) = liveness.get_killer(store) {
                // This store is killed by another store
                dead_stores.push((store, killer));
                self.stats.redundant_stores += 1;
            }
        }

        // Remove dead stores
        let mut changed = false;
        for (store, _killer) in dead_stores {
            graph.kill(store);
            self.removed += 1;
            changed = true;
        }

        changed
    }
}

impl Default for Dse {
    fn default() -> Self {
        Self::new()
    }
}

impl OptimizationPass for Dse {
    fn name(&self) -> &'static str {
        "dse"
    }

    fn run(&mut self, graph: &mut Graph) -> bool {
        self.run_dse(graph)
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
    // Configuration Tests
    // =========================================================================

    #[test]
    fn test_dse_config_default() {
        let config = DseConfig::default();
        assert!(config.enable_must_alias_dse);
        assert!(config.enable_dead_alloc_dse);
        assert_eq!(config.max_iterations, 10);
    }

    // =========================================================================
    // Statistics Tests
    // =========================================================================

    #[test]
    fn test_dse_stats_default() {
        let stats = DseStats::default();
        assert_eq!(stats.redundant_stores, 0);
        assert_eq!(stats.dead_allocation_stores, 0);
        assert_eq!(stats.partial_dead_stores, 0);
        assert_eq!(stats.total_eliminated(), 0);
    }

    #[test]
    fn test_dse_stats_total() {
        let stats = DseStats {
            redundant_stores: 3,
            dead_allocation_stores: 2,
            partial_dead_stores: 1,
            stores_analyzed: 10,
        };
        assert_eq!(stats.total_eliminated(), 6);
    }

    #[test]
    fn test_dse_stats_merge() {
        let mut stats1 = DseStats {
            redundant_stores: 2,
            dead_allocation_stores: 1,
            partial_dead_stores: 0,
            stores_analyzed: 5,
        };
        let stats2 = DseStats {
            redundant_stores: 3,
            dead_allocation_stores: 2,
            partial_dead_stores: 1,
            stores_analyzed: 8,
        };
        stats1.merge(&stats2);
        assert_eq!(stats1.redundant_stores, 5);
        assert_eq!(stats1.dead_allocation_stores, 3);
        assert_eq!(stats1.partial_dead_stores, 1);
        assert_eq!(stats1.stores_analyzed, 13);
    }

    // =========================================================================
    // Pass Creation Tests
    // =========================================================================

    #[test]
    fn test_dse_new() {
        let dse = Dse::new();
        assert_eq!(dse.removed(), 0);
        assert_eq!(dse.stats().total_eliminated(), 0);
    }

    #[test]
    fn test_dse_with_config() {
        let config = DseConfig {
            enable_must_alias_dse: false,
            enable_dead_alloc_dse: true,
            max_iterations: 5,
        };
        let dse = Dse::with_config(config);
        assert_eq!(dse.removed(), 0);
    }

    #[test]
    fn test_dse_name() {
        let dse = Dse::new();
        assert_eq!(dse.name(), "dse");
    }

    #[test]
    fn test_dse_default() {
        let dse = Dse::default();
        assert_eq!(dse.removed(), 0);
    }

    // =========================================================================
    // Store/Load Detection Tests
    // =========================================================================

    #[test]
    fn test_is_store_field() {
        assert!(Dse::is_store(&Operator::Memory(MemoryOp::StoreField)));
    }

    #[test]
    fn test_is_store_element() {
        assert!(Dse::is_store(&Operator::Memory(MemoryOp::StoreElement)));
    }

    #[test]
    fn test_is_store_non_store() {
        assert!(!Dse::is_store(&Operator::Memory(MemoryOp::LoadField)));
        assert!(!Dse::is_store(&Operator::ConstInt(42)));
    }

    #[test]
    fn test_is_load_field() {
        assert!(Dse::is_load(&Operator::Memory(MemoryOp::LoadField)));
    }

    #[test]
    fn test_is_load_element() {
        assert!(Dse::is_load(&Operator::Memory(MemoryOp::LoadElement)));
    }

    #[test]
    fn test_is_load_non_load() {
        assert!(!Dse::is_load(&Operator::Memory(MemoryOp::StoreField)));
        assert!(!Dse::is_load(&Operator::ConstInt(42)));
    }

    // =========================================================================
    // Collection Tests
    // =========================================================================

    #[test]
    fn test_collect_stores_empty() {
        let graph = Graph::new();
        let dse = Dse::new();
        let stores = dse.collect_stores(&graph);
        assert!(stores.is_empty());
    }

    #[test]
    fn test_collect_stores_with_stores() {
        let mut graph = Graph::new();
        let obj = graph.add_node(Operator::Memory(MemoryOp::Alloc), InputList::Empty);
        let val = graph.const_int(42);
        let _store = graph.add_node(
            Operator::Memory(MemoryOp::StoreField),
            InputList::Pair(obj, val),
        );

        let dse = Dse::new();
        let stores = dse.collect_stores(&graph);
        assert_eq!(stores.len(), 1);
    }

    #[test]
    fn test_collect_loads_empty() {
        let graph = Graph::new();
        let dse = Dse::new();
        let loads = dse.collect_loads(&graph);
        assert!(loads.is_empty());
    }

    #[test]
    fn test_collect_loads_with_loads() {
        let mut graph = Graph::new();
        let obj = graph.add_node(Operator::Memory(MemoryOp::Alloc), InputList::Empty);
        let _load = graph.add_node(
            Operator::Memory(MemoryOp::LoadField),
            InputList::Single(obj),
        );

        let dse = Dse::new();
        let loads = dse.collect_loads(&graph);
        assert_eq!(loads.len(), 1);
    }

    // =========================================================================
    // DSE Run Tests
    // =========================================================================

    #[test]
    fn test_dse_no_stores() {
        let mut graph = Graph::new();
        graph.const_int(42);

        let mut dse = Dse::new();
        let changed = dse.run(&mut graph);

        assert!(!changed);
        assert_eq!(dse.removed(), 0);
    }

    #[test]
    fn test_dse_stats_after_run() {
        let mut graph = Graph::new();
        graph.const_int(42);

        let mut dse = Dse::new();
        dse.run(&mut graph);

        assert_eq!(dse.stats().stores_analyzed, 0);
    }
}
