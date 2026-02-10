//! Scalar Replacement of Aggregates (SRA)
//!
//! Scalar replacement eliminates heap allocations by replacing object fields
//! with individual SSA values. This is one of the most powerful optimizations
//! for dynamic languages where temporary objects are common.
//!
//! # Requirements for Scalar Replacement
//!
//! An allocation can be scalar-replaced if:
//! 1. It does not escape (EscapeState::NoEscape)
//! 2. All field accesses have constant indices
//! 3. All uses are known loads/stores (no unknown pointer operations)
//! 4. No aliasing with other objects
//!
//! # Algorithm
//!
//! 1. Collect all field accesses using FieldTracker
//! 2. Create SSA variables for each field
//! 3. Replace StoreField with direct value assignment
//! 4. Replace LoadField with the current SSA value (with phi nodes at merges)
//! 5. Delete the allocation and all field access nodes
//!
//! # Example
//!
//! Before:
//! ```text
//! obj = Alloc(Point)
//! StoreField(obj, "x", 10)
//! StoreField(obj, "y", 20)
//! x = LoadField(obj, "x")
//! y = LoadField(obj, "y")
//! result = Add(x, y)
//! ```
//!
//! After:
//! ```text
//! field_x = 10
//! field_y = 20
//! result = Add(field_x, field_y)
//! ```
//!
//! The allocation and all field accesses are eliminated!

use super::field_tracking::{FieldAccessKind, FieldIndex, FieldTracker};
use crate::ir::graph::Graph;
use crate::ir::node::{InputList, NodeId};
use crate::ir::operators::Operator;
use crate::ir::types::ValueType;
use rustc_hash::FxHashMap;

// =============================================================================
// Scalar Replacement Result
// =============================================================================

/// Result of scalar replacement.
#[derive(Debug, Clone)]
pub struct ScalarReplacementResult {
    /// Whether the replacement was successful.
    pub success: bool,
    /// Number of field loads eliminated.
    pub loads_eliminated: usize,
    /// Number of field stores eliminated.
    pub stores_eliminated: usize,
    /// Number of phi nodes created.
    pub phis_created: usize,
    /// The eliminated allocation node.
    pub allocation: NodeId,
    /// All nodes killed.
    pub killed_nodes: Vec<NodeId>,
}

impl ScalarReplacementResult {
    /// Create a failure result.
    pub fn failure(allocation: NodeId) -> Self {
        Self {
            success: false,
            loads_eliminated: 0,
            stores_eliminated: 0,
            phis_created: 0,
            allocation,
            killed_nodes: Vec::new(),
        }
    }
}

// =============================================================================
// Scalar Replacement Configuration
// =============================================================================

/// Configuration for scalar replacement.
#[derive(Debug, Clone)]
pub struct ScalarReplacementConfig {
    /// Maximum number of fields to scalar replace.
    pub max_fields: usize,
    /// Maximum number of accesses to process.
    pub max_accesses: usize,
    /// Whether to create default values for uninitialized fields.
    pub create_defaults: bool,
    /// Allow partial replacement (some fields only).
    pub allow_partial: bool,
}

impl Default for ScalarReplacementConfig {
    fn default() -> Self {
        Self {
            max_fields: 64,
            max_accesses: 256,
            create_defaults: true,
            allow_partial: false,
        }
    }
}

// =============================================================================
// Field Value Tracker
// =============================================================================

/// Tracks the current SSA value for each field during transformation.
#[derive(Debug)]
struct FieldValueTracker {
    /// Current value for each field.
    values: FxHashMap<u32, NodeId>,
    /// Type for each field.
    types: FxHashMap<u32, ValueType>,
    /// Default value nodes created.
    defaults: FxHashMap<u32, NodeId>,
}

impl FieldValueTracker {
    /// Create a new tracker.
    fn new() -> Self {
        Self {
            values: FxHashMap::default(),
            types: FxHashMap::default(),
            defaults: FxHashMap::default(),
        }
    }

    /// Set the current value for a field.
    fn set(&mut self, field: u32, value: NodeId, ty: ValueType) {
        self.values.insert(field, value);
        self.types.insert(field, ty);
    }

    /// Get the current value for a field.
    fn get(&self, field: u32) -> Option<NodeId> {
        self.values.get(&field).copied()
    }

    /// Get or create a default value for a field.
    fn get_or_default(&mut self, graph: &mut Graph, field: u32, ty: ValueType) -> NodeId {
        if let Some(val) = self.values.get(&field) {
            return *val;
        }

        // Check if we already created a default
        if let Some(default) = self.defaults.get(&field) {
            return *default;
        }

        // Create default value based on type
        let default = Self::create_default_value(graph, ty);
        self.defaults.insert(field, default);
        self.values.insert(field, default);
        default
    }

    /// Create a default value for a type.
    fn create_default_value(graph: &mut Graph, ty: ValueType) -> NodeId {
        match ty {
            ValueType::Int64 => graph.const_int(0),
            ValueType::Float64 => graph.const_float(0.0),
            ValueType::Bool => graph.const_bool(false),
            ValueType::None => {
                graph.add_node_with_type(Operator::ConstNone, InputList::Empty, ValueType::None)
            }
            _ => {
                // For unknown types, use None as default
                graph.add_node_with_type(Operator::ConstNone, InputList::Empty, ValueType::None)
            }
        }
    }
}

// =============================================================================
// Scalar Replacer
// =============================================================================

/// Performs scalar replacement for a single allocation.
#[derive(Debug)]
pub struct ScalarReplacer {
    /// Configuration.
    config: ScalarReplacementConfig,
}

impl ScalarReplacer {
    /// Create a new scalar replacer with default config.
    pub fn new() -> Self {
        Self {
            config: ScalarReplacementConfig::default(),
        }
    }

    /// Create with custom configuration.
    pub fn with_config(config: ScalarReplacementConfig) -> Self {
        Self { config }
    }

    /// Attempt to scalar-replace an allocation.
    pub fn replace(&self, graph: &mut Graph, allocation: NodeId) -> ScalarReplacementResult {
        // Step 1: Track all field accesses
        let tracker = FieldTracker::analyze(graph, allocation);

        // Step 2: Check if scalar replacement is possible
        if !self.can_replace(&tracker) {
            return ScalarReplacementResult::failure(allocation);
        }

        // Step 3: Perform the replacement
        self.perform_replacement(graph, allocation, &tracker)
    }

    /// Check if scalar replacement is possible for this allocation.
    fn can_replace(&self, tracker: &FieldTracker) -> bool {
        // Must not have dynamic accesses
        if tracker.has_dynamic_access {
            return false;
        }

        // Must not have unknown uses
        if tracker.has_unknown_uses {
            return false;
        }

        // Must have some accesses
        if tracker.accesses.is_empty() {
            return false;
        }

        // Check limits
        if tracker.num_fields() > self.config.max_fields {
            return false;
        }

        if tracker.accesses.len() > self.config.max_accesses {
            return false;
        }

        true
    }

    /// Perform the actual replacement.
    fn perform_replacement(
        &self,
        graph: &mut Graph,
        allocation: NodeId,
        tracker: &FieldTracker,
    ) -> ScalarReplacementResult {
        let mut result = ScalarReplacementResult {
            success: true,
            loads_eliminated: 0,
            stores_eliminated: 0,
            phis_created: 0,
            allocation,
            killed_nodes: Vec::new(),
        };

        // Create value tracker for field -> SSA value mapping
        let mut value_tracker = FieldValueTracker::new();

        // Initialize field types from tracker
        for field in tracker.accessed_fields() {
            let ty = tracker.field_type(field);
            value_tracker.types.insert(field, ty);
        }

        // Process accesses in order
        // For a simple case without control flow, we process linearly
        // A full implementation would handle phi nodes at merge points

        for access in &tracker.accesses {
            match access.kind {
                FieldAccessKind::Store => {
                    if let FieldIndex::Constant(field) = access.field {
                        if let Some(value) = access.value {
                            // Record the stored value as the current field value
                            value_tracker.set(field, value, access.value_type);
                            result.stores_eliminated += 1;
                            result.killed_nodes.push(access.node);
                        }
                    }
                }
                FieldAccessKind::Load => {
                    if let FieldIndex::Constant(field) = access.field {
                        // Get the current value for this field
                        let current_value = if self.config.create_defaults {
                            value_tracker.get_or_default(graph, field, access.value_type)
                        } else {
                            match value_tracker.get(field) {
                                Some(v) => v,
                                None => {
                                    // No value available and defaults disabled
                                    result.success = false;
                                    return result;
                                }
                            }
                        };

                        // Replace all uses of the load with the SSA value
                        graph.replace_all_uses(access.node, current_value);
                        result.loads_eliminated += 1;
                        result.killed_nodes.push(access.node);
                    }
                }
            }
        }

        // Kill all the eliminated nodes
        for node in &result.killed_nodes {
            graph.kill(*node);
        }

        // Kill the allocation itself
        // First check if all uses are eliminated
        let remaining_uses: Vec<NodeId> = graph.uses(allocation).to_vec();
        if remaining_uses.is_empty() {
            graph.kill(allocation);
            result.killed_nodes.push(allocation);
        } else {
            // Some uses remain - cannot fully eliminate
            if !self.config.allow_partial {
                result.success = false;
            }
        }

        result
    }
}

impl Default for ScalarReplacer {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// Advanced Scalar Replacement (with control flow)
// =============================================================================

/// Handles scalar replacement with phi node insertion at control flow merges.
#[derive(Debug)]
pub struct AdvancedScalarReplacer {
    /// Base replacer.
    base: ScalarReplacer,
}

impl AdvancedScalarReplacer {
    /// Create a new advanced replacer.
    pub fn new() -> Self {
        Self {
            base: ScalarReplacer::new(),
        }
    }

    /// Attempt replacement with phi insertion.
    pub fn replace(&self, graph: &mut Graph, allocation: NodeId) -> ScalarReplacementResult {
        let tracker = FieldTracker::analyze(graph, allocation);

        if !self.can_replace(&tracker) {
            return ScalarReplacementResult::failure(allocation);
        }

        // Check if we need phi nodes (control flow merges)
        if self.needs_phi_insertion(&tracker, graph) {
            self.replace_with_phis(graph, allocation, &tracker)
        } else {
            // Simple case - use base replacer
            self.base.perform_replacement(graph, allocation, &tracker)
        }
    }

    /// Check if replacement is possible.
    fn can_replace(&self, tracker: &FieldTracker) -> bool {
        self.base.can_replace(tracker)
    }

    /// Check if phi nodes are needed.
    fn needs_phi_insertion(&self, tracker: &FieldTracker, graph: &Graph) -> bool {
        // Check if any load might have multiple reaching stores
        for field in tracker.accessed_fields() {
            let stores: Vec<_> = tracker.stores_for_field(field).collect();
            let loads: Vec<_> = tracker.loads_for_field(field).collect();

            // Multiple stores to same field with loads
            if stores.len() > 1 && !loads.is_empty() {
                // Check if stores are in different control regions
                let store_controls: Vec<_> = stores.iter().filter_map(|s| s.control).collect();

                if store_controls.len() > 1 {
                    // Different control regions - need phi
                    return true;
                }
            }
        }

        false
    }

    /// Replace with phi node insertion.
    fn replace_with_phis(
        &self,
        graph: &mut Graph,
        allocation: NodeId,
        tracker: &FieldTracker,
    ) -> ScalarReplacementResult {
        let mut result = ScalarReplacementResult {
            success: true,
            loads_eliminated: 0,
            stores_eliminated: 0,
            phis_created: 0,
            allocation,
            killed_nodes: Vec::new(),
        };

        // For each field, collect all stores and their control dependencies
        let mut field_stores: FxHashMap<u32, Vec<(NodeId, Option<NodeId>)>> = FxHashMap::default();
        let mut field_loads: FxHashMap<u32, Vec<NodeId>> = FxHashMap::default();

        for access in &tracker.accesses {
            if let FieldIndex::Constant(field) = access.field {
                match access.kind {
                    FieldAccessKind::Store => {
                        if let Some(value) = access.value {
                            field_stores
                                .entry(field)
                                .or_default()
                                .push((value, access.control));
                        }
                    }
                    FieldAccessKind::Load => {
                        field_loads.entry(field).or_default().push(access.node);
                    }
                }
            }
        }

        // Process each field
        for field in tracker.accessed_fields() {
            let stores = field_stores.get(&field).cloned().unwrap_or_default();
            let loads = field_loads.get(&field).cloned().unwrap_or_default();

            if loads.is_empty() {
                // No loads - just kill stores
                for access in tracker.stores_for_field(field) {
                    result.stores_eliminated += 1;
                    result.killed_nodes.push(access.node);
                }
                continue;
            }

            // Determine the value to use for loads
            let field_value = if stores.len() == 1 {
                // Single store - use its value directly
                stores[0].0
            } else if stores.len() > 1 {
                // Multiple stores - need to create phi
                let field_type = tracker.field_type(field);

                // Find a merge point (region node) that dominates all loads
                // For now, use a simplified approach: create phi at first load's region
                let phi_values: Vec<NodeId> = stores.iter().map(|(v, _)| *v).collect();

                // Find region for phi
                let region = self.find_merge_region(graph, &stores);

                if let Some(region) = region {
                    let phi = graph.phi(region, &phi_values, field_type);
                    result.phis_created += 1;
                    phi
                } else {
                    // Can't find merge point - use last store value
                    stores.last().map(|(v, _)| *v).unwrap_or_else(|| {
                        graph.const_int(0) // Fallback
                    })
                }
            } else {
                // No stores but has loads - use default
                let field_type = tracker.field_type(field);
                FieldValueTracker::create_default_value(graph, field_type)
            };

            // Replace all loads with the field value
            for load in loads {
                graph.replace_all_uses(load, field_value);
                result.loads_eliminated += 1;
                result.killed_nodes.push(load);
            }

            // Kill all stores
            for access in tracker.stores_for_field(field) {
                result.stores_eliminated += 1;
                if !result.killed_nodes.contains(&access.node) {
                    result.killed_nodes.push(access.node);
                }
            }
        }

        // Kill all eliminated nodes
        for node in &result.killed_nodes {
            graph.kill(*node);
        }

        // Try to kill allocation
        let remaining_uses: Vec<NodeId> = graph.uses(allocation).to_vec();
        if remaining_uses.is_empty() {
            graph.kill(allocation);
            result.killed_nodes.push(allocation);
        }

        result
    }

    /// Find a region node that can serve as a merge point.
    fn find_merge_region(
        &self,
        graph: &Graph,
        stores: &[(NodeId, Option<NodeId>)],
    ) -> Option<NodeId> {
        // Collect all control dependencies
        let controls: Vec<NodeId> = stores.iter().filter_map(|(_, ctrl)| *ctrl).collect();

        if controls.is_empty() {
            return None;
        }

        // Look for a region that all controls flow into
        // For now, just find any region in the graph as a simple heuristic
        for (id, node) in graph.iter() {
            if matches!(
                node.op,
                Operator::Control(crate::ir::operators::ControlOp::Region)
            ) {
                // Check if this region is reachable from all controls
                // Simplified: just return first region found
                return Some(id);
            }
        }

        None
    }
}

impl Default for AdvancedScalarReplacer {
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
    use crate::ir::graph::Graph;
    use smallvec::SmallVec;

    // -------------------------------------------------------------------------
    // ScalarReplacementResult Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_scalar_replacement_result_failure() {
        let result = ScalarReplacementResult::failure(NodeId::new(5));

        assert!(!result.success);
        assert_eq!(result.allocation, NodeId::new(5));
        assert_eq!(result.loads_eliminated, 0);
        assert_eq!(result.stores_eliminated, 0);
        assert!(result.killed_nodes.is_empty());
    }

    #[test]
    fn test_scalar_replacement_result_success() {
        let result = ScalarReplacementResult {
            success: true,
            loads_eliminated: 5,
            stores_eliminated: 3,
            phis_created: 1,
            allocation: NodeId::new(10),
            killed_nodes: vec![NodeId::new(1), NodeId::new(2)],
        };

        assert!(result.success);
        assert_eq!(result.loads_eliminated, 5);
        assert_eq!(result.stores_eliminated, 3);
        assert_eq!(result.phis_created, 1);
    }

    // -------------------------------------------------------------------------
    // ScalarReplacementConfig Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_config_default() {
        let config = ScalarReplacementConfig::default();

        assert_eq!(config.max_fields, 64);
        assert_eq!(config.max_accesses, 256);
        assert!(config.create_defaults);
        assert!(!config.allow_partial);
    }

    #[test]
    fn test_config_custom() {
        let config = ScalarReplacementConfig {
            max_fields: 32,
            max_accesses: 128,
            create_defaults: false,
            allow_partial: true,
        };

        assert_eq!(config.max_fields, 32);
        assert!(!config.create_defaults);
        assert!(config.allow_partial);
    }

    // -------------------------------------------------------------------------
    // FieldValueTracker Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_field_value_tracker_new() {
        let tracker = FieldValueTracker::new();
        assert!(tracker.values.is_empty());
        assert!(tracker.types.is_empty());
    }

    #[test]
    fn test_field_value_tracker_set_get() {
        let mut tracker = FieldValueTracker::new();

        tracker.set(0, NodeId::new(10), ValueType::Int64);
        tracker.set(1, NodeId::new(11), ValueType::Float64);

        assert_eq!(tracker.get(0), Some(NodeId::new(10)));
        assert_eq!(tracker.get(1), Some(NodeId::new(11)));
        assert_eq!(tracker.get(2), None);
    }

    #[test]
    fn test_field_value_tracker_get_or_default_existing() {
        let mut tracker = FieldValueTracker::new();
        let mut graph = Graph::new();

        tracker.set(0, NodeId::new(10), ValueType::Int64);

        let value = tracker.get_or_default(&mut graph, 0, ValueType::Int64);
        assert_eq!(value, NodeId::new(10));
    }

    #[test]
    fn test_field_value_tracker_get_or_default_int() {
        let mut tracker = FieldValueTracker::new();
        let mut graph = Graph::new();

        let value = tracker.get_or_default(&mut graph, 0, ValueType::Int64);

        // Should create a const int 0
        let node = graph.get(value).unwrap();
        assert!(matches!(node.op, Operator::ConstInt(0)));
    }

    #[test]
    fn test_field_value_tracker_get_or_default_float() {
        let mut tracker = FieldValueTracker::new();
        let mut graph = Graph::new();

        let value = tracker.get_or_default(&mut graph, 0, ValueType::Float64);

        // Should create a const float 0.0
        let node = graph.get(value).unwrap();
        assert!(matches!(node.op, Operator::ConstFloat(_)));
    }

    #[test]
    fn test_field_value_tracker_get_or_default_bool() {
        let mut tracker = FieldValueTracker::new();
        let mut graph = Graph::new();

        let value = tracker.get_or_default(&mut graph, 0, ValueType::Bool);

        let node = graph.get(value).unwrap();
        assert!(matches!(node.op, Operator::ConstBool(false)));
    }

    #[test]
    fn test_field_value_tracker_get_or_default_none() {
        let mut tracker = FieldValueTracker::new();
        let mut graph = Graph::new();

        let value = tracker.get_or_default(&mut graph, 0, ValueType::None);

        let node = graph.get(value).unwrap();
        assert!(matches!(node.op, Operator::ConstNone));
    }

    #[test]
    fn test_field_value_tracker_caches_defaults() {
        let mut tracker = FieldValueTracker::new();
        let mut graph = Graph::new();

        let value1 = tracker.get_or_default(&mut graph, 0, ValueType::Int64);
        let value2 = tracker.get_or_default(&mut graph, 0, ValueType::Int64);

        // Should return the same node
        assert_eq!(value1, value2);
    }

    // -------------------------------------------------------------------------
    // ScalarReplacer Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_scalar_replacer_new() {
        let replacer = ScalarReplacer::new();
        assert_eq!(replacer.config.max_fields, 64);
    }

    #[test]
    fn test_scalar_replacer_with_config() {
        let config = ScalarReplacementConfig {
            max_fields: 16,
            ..Default::default()
        };
        let replacer = ScalarReplacer::with_config(config);
        assert_eq!(replacer.config.max_fields, 16);
    }

    #[test]
    fn test_scalar_replacer_empty_graph() {
        let replacer = ScalarReplacer::new();
        let mut graph = Graph::new();

        // Try to replace a non-existent allocation
        let result = replacer.replace(&mut graph, NodeId::new(999));

        // Should fail (no accesses)
        assert!(!result.success);
    }

    #[test]
    fn test_scalar_replacer_default() {
        let replacer = ScalarReplacer::default();
        assert_eq!(replacer.config.max_fields, 64);
    }

    // -------------------------------------------------------------------------
    // AdvancedScalarReplacer Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_advanced_replacer_new() {
        let replacer = AdvancedScalarReplacer::new();
        assert_eq!(replacer.base.config.max_fields, 64);
    }

    #[test]
    fn test_advanced_replacer_default() {
        let replacer = AdvancedScalarReplacer::default();
        assert_eq!(replacer.base.config.max_fields, 64);
    }

    #[test]
    fn test_advanced_replacer_empty_graph() {
        let replacer = AdvancedScalarReplacer::new();
        let mut graph = Graph::new();

        let result = replacer.replace(&mut graph, NodeId::new(999));

        assert!(!result.success);
    }

    // -------------------------------------------------------------------------
    // Integration Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_create_default_value_int64() {
        let mut graph = Graph::new();
        let value = FieldValueTracker::create_default_value(&mut graph, ValueType::Int64);

        let node = graph.get(value).unwrap();
        assert!(matches!(node.op, Operator::ConstInt(0)));
        assert_eq!(node.ty, ValueType::Int64);
    }

    #[test]
    fn test_create_default_value_float64() {
        let mut graph = Graph::new();
        let value = FieldValueTracker::create_default_value(&mut graph, ValueType::Float64);

        let node = graph.get(value).unwrap();
        // Float 0.0 stored as bits
        assert!(matches!(node.op, Operator::ConstFloat(_)));
    }

    #[test]
    fn test_create_default_value_unknown_type() {
        let mut graph = Graph::new();
        let value = FieldValueTracker::create_default_value(&mut graph, ValueType::Object);

        let node = graph.get(value).unwrap();
        // Unknown types default to None
        assert!(matches!(node.op, Operator::ConstNone));
    }

    #[test]
    fn test_can_replace_dynamic_access() {
        let replacer = ScalarReplacer::new();

        let mut tracker = FieldTracker::new(NodeId::new(0));
        tracker.has_dynamic_access = true;

        assert!(!replacer.can_replace(&tracker));
    }

    #[test]
    fn test_can_replace_unknown_uses() {
        let replacer = ScalarReplacer::new();

        let mut tracker = FieldTracker::new(NodeId::new(0));
        tracker.has_unknown_uses = true;

        assert!(!replacer.can_replace(&tracker));
    }

    #[test]
    fn test_can_replace_no_accesses() {
        let replacer = ScalarReplacer::new();
        let tracker = FieldTracker::new(NodeId::new(0));

        assert!(!replacer.can_replace(&tracker));
    }

    #[test]
    fn test_can_replace_too_many_fields() {
        let config = ScalarReplacementConfig {
            max_fields: 2,
            ..Default::default()
        };
        let replacer = ScalarReplacer::with_config(config);

        let mut tracker = FieldTracker::new(NodeId::new(0));
        // Add accesses for 3 fields
        tracker.stores_by_field.insert(0, SmallVec::new());
        tracker.stores_by_field.insert(1, SmallVec::new());
        tracker.stores_by_field.insert(2, SmallVec::new());
        tracker
            .accesses
            .push(super::super::field_tracking::FieldAccess {
                node: NodeId::new(1),
                object: NodeId::new(0),
                field: FieldIndex::Constant(0),
                kind: FieldAccessKind::Store,
                value: Some(NodeId::new(2)),
                value_type: ValueType::Int64,
                control: None,
            });

        assert!(!replacer.can_replace(&tracker));
    }

    #[test]
    fn test_can_replace_too_many_accesses() {
        let config = ScalarReplacementConfig {
            max_accesses: 2,
            ..Default::default()
        };
        let replacer = ScalarReplacer::with_config(config);

        let mut tracker = FieldTracker::new(NodeId::new(0));
        tracker.stores_by_field.insert(0, SmallVec::new());

        // Add 3 accesses
        for i in 0..3 {
            tracker
                .accesses
                .push(super::super::field_tracking::FieldAccess {
                    node: NodeId::new(i + 1),
                    object: NodeId::new(0),
                    field: FieldIndex::Constant(0),
                    kind: FieldAccessKind::Store,
                    value: Some(NodeId::new(10)),
                    value_type: ValueType::Int64,
                    control: None,
                });
        }

        assert!(!replacer.can_replace(&tracker));
    }

    #[test]
    fn test_can_replace_valid() {
        let replacer = ScalarReplacer::new();

        let mut tracker = FieldTracker::new(NodeId::new(0));
        tracker.stores_by_field.insert(0, SmallVec::new());
        tracker
            .accesses
            .push(super::super::field_tracking::FieldAccess {
                node: NodeId::new(1),
                object: NodeId::new(0),
                field: FieldIndex::Constant(0),
                kind: FieldAccessKind::Store,
                value: Some(NodeId::new(2)),
                value_type: ValueType::Int64,
                control: None,
            });

        assert!(replacer.can_replace(&tracker));
    }
}
