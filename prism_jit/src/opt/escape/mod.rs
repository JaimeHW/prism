//! Escape Analysis and Object Optimization
//!
//! This module provides comprehensive escape analysis and related optimizations:
//!
//! - **Escape Analysis**: Determines which allocations escape their creating function
//! - **Scalar Replacement**: Eliminates allocations by replacing fields with SSA values
//! - **Stack Allocation**: Moves non-escaping allocations from heap to stack
//!
//! # Module Structure
//!
//! - `field_tracking`: Tracks field accesses for scalar replacement
//! - `scalar_replace`: Scalar replacement of aggregates (SRA)
//! - `stack_alloc`: Stack allocation transformation
//!
//! # Optimization Pipeline
//!
//! The escape analysis pass runs in three phases:
//!
//! 1. **Analysis Phase**: Compute escape states for all allocations
//! 2. **Scalar Replacement Phase**: Try to eliminate non-escaping allocations
//! 3. **Stack Allocation Phase**: Convert remaining allocations to stack
//!
//! # Example
//!
//! ```text
//! Before:
//!   point = Alloc(Point)      # Heap allocation
//!   StoreField(point, x, 10)
//!   StoreField(point, y, 20)
//!   sum = LoadField(point, x) + LoadField(point, y)
//!   return sum
//!
//! After Scalar Replacement:
//!   # No allocation!
//!   field_x = 10
//!   field_y = 20
//!   sum = field_x + field_y
//!   return sum
//! ```

pub mod field_tracking;
pub mod scalar_replace;
pub mod stack_alloc;

use crate::ir::graph::Graph;
use crate::ir::node::NodeId;
use crate::ir::operators::{ControlOp, MemoryOp, Operator};
use rustc_hash::FxHashMap;
use std::collections::{HashSet, VecDeque};

// Re-exports
pub use field_tracking::{
    FieldAccess, FieldAccessKind, FieldIndex, FieldMap, FieldState, FieldTracker,
};
pub use scalar_replace::{
    AdvancedScalarReplacer, ScalarReplacementConfig, ScalarReplacementResult, ScalarReplacer,
};
pub use stack_alloc::{
    BatchStackAllocator, ObjectSizeEstimator, StackAllocConfig, StackAllocFailure,
    StackAllocResult, StackAllocStats, StackAllocator, StackFrameLayout, StackSlot,
};

// =============================================================================
// Escape State
// =============================================================================

/// The escape state of an allocation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum EscapeState {
    /// Object does not escape the function.
    NoEscape,
    /// Object escapes via arguments to called methods.
    ArgEscape,
    /// Object escapes globally (stored to heap, returned, etc.).
    GlobalEscape,
}

impl EscapeState {
    /// Check if the object can be stack-allocated.
    #[inline]
    pub fn can_stack_allocate(self) -> bool {
        matches!(self, EscapeState::NoEscape | EscapeState::ArgEscape)
    }

    /// Check if the object can be scalar-replaced.
    #[inline]
    pub fn can_scalar_replace(self) -> bool {
        self == EscapeState::NoEscape
    }

    /// Merge two escape states (takes the more conservative one).
    #[inline]
    pub fn merge(self, other: EscapeState) -> EscapeState {
        std::cmp::max(self, other)
    }
}

impl Default for EscapeState {
    fn default() -> Self {
        EscapeState::NoEscape
    }
}

// =============================================================================
// Object Type
// =============================================================================

/// Type of allocated object.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ObjectType {
    /// Generic object.
    Object,
    /// Array/list.
    Array,
    /// Dictionary.
    Dict,
    /// Tuple.
    Tuple,
    /// Closure/function.
    Closure,
    /// Set.
    Set,
    /// Unknown type.
    Unknown,
}

// =============================================================================
// Allocation Info
// =============================================================================

/// Information about an allocation site.
#[derive(Debug, Clone)]
pub struct AllocationInfo {
    /// Node ID of the allocation.
    pub node: NodeId,
    /// Computed escape state.
    pub escape_state: EscapeState,
    /// Type of the allocated object.
    pub object_type: ObjectType,
    /// Whether all uses are known.
    pub all_uses_known: bool,
    /// Estimated size in bytes.
    pub estimated_size: Option<usize>,
    /// Number of field accesses.
    pub access_count: usize,
}

impl AllocationInfo {
    /// Check if this allocation can be optimized.
    pub fn can_optimize(&self) -> bool {
        self.escape_state != EscapeState::GlobalEscape && self.all_uses_known
    }
}

// =============================================================================
// Escape Analysis
// =============================================================================

/// Escape analysis result.
#[derive(Debug)]
pub struct EscapeAnalysis {
    /// Escape state per allocation node.
    states: FxHashMap<NodeId, EscapeState>,
    /// Detailed info per allocation.
    allocations: Vec<AllocationInfo>,
    /// Number of non-escaping allocations.
    non_escaping: usize,
    /// Number of arg-escaping allocations.
    arg_escaping: usize,
    /// Number of globally-escaping allocations.
    global_escaping: usize,
}

impl EscapeAnalysis {
    /// Compute escape analysis for a graph.
    pub fn compute(graph: &Graph) -> Self {
        let mut analysis = Self {
            states: FxHashMap::default(),
            allocations: Vec::new(),
            non_escaping: 0,
            arg_escaping: 0,
            global_escaping: 0,
        };

        // Find all allocation sites
        let alloc_sites: Vec<NodeId> = graph
            .iter()
            .filter_map(|(id, node)| {
                if Self::is_allocation(&node.op) {
                    Some(id)
                } else {
                    None
                }
            })
            .collect();

        // Analyze each allocation
        for alloc in alloc_sites {
            let info = Self::analyze_allocation(graph, alloc);

            match info.escape_state {
                EscapeState::NoEscape => analysis.non_escaping += 1,
                EscapeState::ArgEscape => analysis.arg_escaping += 1,
                EscapeState::GlobalEscape => analysis.global_escaping += 1,
            }

            analysis.states.insert(alloc, info.escape_state);
            analysis.allocations.push(info);
        }

        analysis
    }

    /// Check if an operator is an allocation.
    #[inline]
    fn is_allocation(op: &Operator) -> bool {
        matches!(
            op,
            Operator::Memory(MemoryOp::Alloc)
                | Operator::Memory(MemoryOp::AllocArray)
                | Operator::BuildList(_)
                | Operator::BuildTuple(_)
                | Operator::BuildDict(_)
        )
    }

    /// Analyze a single allocation site.
    fn analyze_allocation(graph: &Graph, alloc: NodeId) -> AllocationInfo {
        let mut state = EscapeState::NoEscape;
        let mut all_uses_known = true;
        let mut access_count = 0;

        // BFS through all uses
        let mut visited = HashSet::new();
        let mut worklist = VecDeque::new();

        worklist.push_back(alloc);
        visited.insert(alloc);

        while let Some(node_id) = worklist.pop_front() {
            for &user_id in graph.uses(node_id) {
                if visited.contains(&user_id) {
                    continue;
                }
                visited.insert(user_id);

                let user = graph.node(user_id);
                let use_escape = Self::classify_use(&user.op);

                state = state.merge(use_escape);

                // Count field accesses
                if matches!(
                    user.op,
                    Operator::Memory(MemoryOp::LoadField)
                        | Operator::Memory(MemoryOp::StoreField)
                        | Operator::Memory(MemoryOp::LoadElement)
                        | Operator::Memory(MemoryOp::StoreElement)
                ) {
                    access_count += 1;
                }

                // If used as input to phi, propagate to phi users
                if use_escape != EscapeState::GlobalEscape && Self::should_propagate(&user.op) {
                    worklist.push_back(user_id);
                }

                if !Self::is_known_use(&user.op) {
                    all_uses_known = false;
                }
            }
        }

        AllocationInfo {
            node: alloc,
            escape_state: state,
            object_type: Self::infer_object_type(graph, alloc),
            all_uses_known,
            estimated_size: Self::estimate_size(graph, alloc),
            access_count,
        }
    }

    /// Classify a use of an allocated object.
    fn classify_use(op: &Operator) -> EscapeState {
        match op {
            // Field operations don't cause escape
            Operator::Memory(MemoryOp::LoadField)
            | Operator::Memory(MemoryOp::LoadElement)
            | Operator::GetItem
            | Operator::GetAttr => EscapeState::NoEscape,

            // Store to another object - escapes
            Operator::Memory(MemoryOp::StoreField)
            | Operator::Memory(MemoryOp::StoreElement)
            | Operator::SetItem
            | Operator::SetAttr => {
                // If storing TO this object, it doesn't escape
                // If storing this object INTO another, it escapes
                // For now, be conservative
                EscapeState::GlobalEscape
            }

            // Return - escapes globally
            Operator::Control(ControlOp::Return) => EscapeState::GlobalEscape,

            // Phi - propagate (will analyze phi users)
            Operator::Phi | Operator::LoopPhi => EscapeState::NoEscape,

            // Call - arg escape
            Operator::Call(_) => EscapeState::ArgEscape,

            // Guards/type checks - don't cause escape
            Operator::Guard(_) | Operator::TypeCheck => EscapeState::NoEscape,

            // Comparison - doesn't cause escape
            Operator::IntCmp(_) | Operator::FloatCmp(_) | Operator::GenericCmp(_) => {
                EscapeState::NoEscape
            }

            // Container ops that read - no escape
            Operator::GetIter | Operator::IterNext => EscapeState::NoEscape,

            // Unknown - conservative
            _ => EscapeState::GlobalEscape,
        }
    }

    /// Check if we should propagate through an operation.
    fn should_propagate(op: &Operator) -> bool {
        matches!(op, Operator::Phi | Operator::LoopPhi)
    }

    /// Check if a use is known/understood.
    fn is_known_use(op: &Operator) -> bool {
        matches!(
            op,
            Operator::Memory(_)
                | Operator::Control(_)
                | Operator::IntCmp(_)
                | Operator::FloatCmp(_)
                | Operator::GenericCmp(_)
                | Operator::Guard(_)
                | Operator::TypeCheck
                | Operator::Call(_)
                | Operator::Phi
                | Operator::LoopPhi
                | Operator::GetItem
                | Operator::SetItem
                | Operator::GetAttr
                | Operator::SetAttr
                | Operator::GetIter
                | Operator::IterNext
        )
    }

    /// Infer object type from allocation.
    fn infer_object_type(graph: &Graph, alloc: NodeId) -> ObjectType {
        let node = graph.node(alloc);
        match node.op {
            Operator::Memory(MemoryOp::AllocArray) => ObjectType::Array,
            Operator::Memory(MemoryOp::Alloc) => ObjectType::Object,
            Operator::BuildList(_) => ObjectType::Array,
            Operator::BuildTuple(_) => ObjectType::Tuple,
            Operator::BuildDict(_) => ObjectType::Dict,
            _ => ObjectType::Unknown,
        }
    }

    /// Estimate size for an allocation.
    fn estimate_size(graph: &Graph, alloc: NodeId) -> Option<usize> {
        let node = graph.node(alloc);
        match node.op {
            Operator::BuildList(count) => Some(64 + count as usize * 8),
            Operator::BuildTuple(count) => Some(32 + count as usize * 8),
            Operator::BuildDict(count) => Some(72 + count as usize * 24),
            _ => None,
        }
    }

    // =========================================================================
    // Query API
    // =========================================================================

    /// Get escape state for an allocation.
    #[inline]
    pub fn escape_state(&self, node: NodeId) -> Option<EscapeState> {
        self.states.get(&node).copied()
    }

    /// Check if an allocation can be stack-allocated.
    #[inline]
    pub fn can_stack_allocate(&self, node: NodeId) -> bool {
        self.states
            .get(&node)
            .map(|s| s.can_stack_allocate())
            .unwrap_or(false)
    }

    /// Check if an allocation can be scalar-replaced.
    #[inline]
    pub fn can_scalar_replace(&self, node: NodeId) -> bool {
        self.states
            .get(&node)
            .map(|s| s.can_scalar_replace())
            .unwrap_or(false)
    }

    /// Get all allocations that can be stack-allocated.
    pub fn stack_allocatable(&self) -> impl Iterator<Item = NodeId> + '_ {
        self.states
            .iter()
            .filter(|(_, s)| s.can_stack_allocate())
            .map(|(n, _)| *n)
    }

    /// Get all allocations that can be scalar-replaced.
    pub fn scalar_replaceable(&self) -> impl Iterator<Item = NodeId> + '_ {
        self.states
            .iter()
            .filter(|(_, s)| s.can_scalar_replace())
            .map(|(n, _)| *n)
    }

    /// Get detailed allocation info.
    pub fn allocations(&self) -> &[AllocationInfo] {
        &self.allocations
    }

    /// Get allocation info by node.
    pub fn allocation_info(&self, node: NodeId) -> Option<&AllocationInfo> {
        self.allocations.iter().find(|a| a.node == node)
    }

    /// Get number of non-escaping allocations.
    #[inline]
    pub fn non_escaping_count(&self) -> usize {
        self.non_escaping
    }

    /// Get number of arg-escaping allocations.
    #[inline]
    pub fn arg_escaping_count(&self) -> usize {
        self.arg_escaping
    }

    /// Get number of globally-escaping allocations.
    #[inline]
    pub fn global_escaping_count(&self) -> usize {
        self.global_escaping
    }

    /// Get total number of allocations.
    #[inline]
    pub fn total_allocations(&self) -> usize {
        self.allocations.len()
    }
}

// =============================================================================
// Escape Analysis Pass
// =============================================================================

/// Escape analysis optimization pass statistics.
#[derive(Debug, Clone, Default)]
pub struct EscapeStats {
    /// Number of allocations analyzed.
    pub allocations_analyzed: usize,
    /// Number of non-escaping allocations.
    pub non_escaping: usize,
    /// Number of allocations scalar-replaced.
    pub scalar_replaced: usize,
    /// Number of allocations stack-allocated.
    pub stack_allocated: usize,
    /// Total nodes eliminated.
    pub nodes_eliminated: usize,
}

/// Escape analysis optimization pass.
#[derive(Debug)]
pub struct Escape {
    /// Configuration for scalar replacement.
    scalar_config: ScalarReplacementConfig,
    /// Configuration for stack allocation.
    stack_config: StackAllocConfig,
    /// Statistics.
    stats: EscapeStats,
    /// Enable aggressive optimization.
    aggressive: bool,
}

impl Escape {
    /// Create a new escape analysis pass.
    pub fn new() -> Self {
        Self {
            scalar_config: ScalarReplacementConfig::default(),
            stack_config: StackAllocConfig::default(),
            stats: EscapeStats::default(),
            aggressive: false,
        }
    }

    /// Create an aggressive escape analysis pass.
    pub fn aggressive() -> Self {
        Self {
            scalar_config: ScalarReplacementConfig {
                max_fields: 128,
                max_accesses: 512,
                ..Default::default()
            },
            stack_config: StackAllocConfig::aggressive(),
            stats: EscapeStats::default(),
            aggressive: true,
        }
    }

    /// Get statistics.
    pub fn stats(&self) -> &EscapeStats {
        &self.stats
    }

    /// Get number of stack allocations.
    #[inline]
    pub fn stack_allocated(&self) -> usize {
        self.stats.stack_allocated
    }

    /// Get number of scalar replacements.
    #[inline]
    pub fn scalar_replaced(&self) -> usize {
        self.stats.scalar_replaced
    }

    /// Run escape analysis pass.
    pub fn run(&mut self, graph: &mut Graph) -> bool {
        let analysis = EscapeAnalysis::compute(graph);
        self.stats.allocations_analyzed = analysis.total_allocations();
        self.stats.non_escaping = analysis.non_escaping_count();

        let mut changed = false;

        // Phase 1: Try scalar replacement for non-escaping allocations
        let replacer = if self.aggressive {
            AdvancedScalarReplacer::new()
        } else {
            AdvancedScalarReplacer::new()
        };

        let scalar_candidates: Vec<NodeId> = analysis.scalar_replaceable().collect();
        for alloc in scalar_candidates {
            let result = replacer.replace(graph, alloc);
            if result.success {
                self.stats.scalar_replaced += 1;
                self.stats.nodes_eliminated += result.killed_nodes.len();
                changed = true;
            }
        }

        // Phase 2: Try stack allocation for remaining non-escaping allocations
        let mut stack_allocator = StackAllocator::with_config(self.stack_config.clone());

        for alloc_info in analysis.allocations() {
            // Skip if already scalar-replaced or globally escapes
            if !alloc_info.can_optimize() {
                continue;
            }

            // Check if still exists in graph (might have been eliminated)
            if graph.get(alloc_info.node).is_none() {
                continue;
            }

            let escapes_to_args = alloc_info.escape_state == EscapeState::ArgEscape;
            let result =
                stack_allocator.try_stack_allocate(graph, alloc_info.node, escapes_to_args);

            if result.success {
                self.stats.stack_allocated += 1;
                changed = true;
            }
        }

        changed
    }
}

impl Default for Escape {
    fn default() -> Self {
        Self::new()
    }
}

// Implement OptimizationPass trait
impl super::OptimizationPass for Escape {
    fn name(&self) -> &'static str {
        "escape"
    }

    fn run(&mut self, graph: &mut Graph) -> bool {
        Escape::run(self, graph)
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::builder::{ControlBuilder, GraphBuilder};

    // -------------------------------------------------------------------------
    // EscapeState Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_escape_state_ordering() {
        assert!(EscapeState::NoEscape < EscapeState::ArgEscape);
        assert!(EscapeState::ArgEscape < EscapeState::GlobalEscape);
    }

    #[test]
    fn test_escape_state_merge() {
        assert_eq!(
            EscapeState::NoEscape.merge(EscapeState::NoEscape),
            EscapeState::NoEscape
        );
        assert_eq!(
            EscapeState::NoEscape.merge(EscapeState::ArgEscape),
            EscapeState::ArgEscape
        );
        assert_eq!(
            EscapeState::ArgEscape.merge(EscapeState::GlobalEscape),
            EscapeState::GlobalEscape
        );
        assert_eq!(
            EscapeState::GlobalEscape.merge(EscapeState::NoEscape),
            EscapeState::GlobalEscape
        );
    }

    #[test]
    fn test_escape_state_can_stack_allocate() {
        assert!(EscapeState::NoEscape.can_stack_allocate());
        assert!(EscapeState::ArgEscape.can_stack_allocate());
        assert!(!EscapeState::GlobalEscape.can_stack_allocate());
    }

    #[test]
    fn test_escape_state_can_scalar_replace() {
        assert!(EscapeState::NoEscape.can_scalar_replace());
        assert!(!EscapeState::ArgEscape.can_scalar_replace());
        assert!(!EscapeState::GlobalEscape.can_scalar_replace());
    }

    #[test]
    fn test_escape_state_default() {
        assert_eq!(EscapeState::default(), EscapeState::NoEscape);
    }

    // -------------------------------------------------------------------------
    // ObjectType Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_object_type_equality() {
        assert_eq!(ObjectType::Array, ObjectType::Array);
        assert_ne!(ObjectType::Array, ObjectType::Dict);
        assert_ne!(ObjectType::Object, ObjectType::Tuple);
    }

    #[test]
    fn test_object_type_hash() {
        use std::collections::HashSet;
        let mut set = HashSet::new();
        set.insert(ObjectType::Array);
        set.insert(ObjectType::Dict);
        set.insert(ObjectType::Array); // Duplicate
        assert_eq!(set.len(), 2);
    }

    // -------------------------------------------------------------------------
    // AllocationInfo Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_allocation_info_can_optimize() {
        let info = AllocationInfo {
            node: NodeId::new(0),
            escape_state: EscapeState::NoEscape,
            object_type: ObjectType::Object,
            all_uses_known: true,
            estimated_size: Some(64),
            access_count: 5,
        };
        assert!(info.can_optimize());

        let info2 = AllocationInfo {
            escape_state: EscapeState::GlobalEscape,
            ..info.clone()
        };
        assert!(!info2.can_optimize());

        let info3 = AllocationInfo {
            all_uses_known: false,
            ..info.clone()
        };
        assert!(!info3.can_optimize());
    }

    // -------------------------------------------------------------------------
    // EscapeAnalysis Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_escape_analysis_empty() {
        let builder = GraphBuilder::new(0, 0);
        let graph = builder.finish();

        let analysis = EscapeAnalysis::compute(&graph);
        assert_eq!(analysis.non_escaping_count(), 0);
        assert_eq!(analysis.arg_escaping_count(), 0);
        assert_eq!(analysis.global_escaping_count(), 0);
        assert_eq!(analysis.total_allocations(), 0);
    }

    #[test]
    fn test_escape_analysis_query() {
        let analysis = EscapeAnalysis {
            states: {
                let mut m = FxHashMap::default();
                m.insert(NodeId::new(0), EscapeState::NoEscape);
                m.insert(NodeId::new(1), EscapeState::ArgEscape);
                m.insert(NodeId::new(2), EscapeState::GlobalEscape);
                m
            },
            allocations: vec![],
            non_escaping: 1,
            arg_escaping: 1,
            global_escaping: 1,
        };

        assert_eq!(
            analysis.escape_state(NodeId::new(0)),
            Some(EscapeState::NoEscape)
        );
        assert!(analysis.can_scalar_replace(NodeId::new(0)));
        assert!(analysis.can_stack_allocate(NodeId::new(0)));

        assert!(analysis.can_stack_allocate(NodeId::new(1)));
        assert!(!analysis.can_scalar_replace(NodeId::new(1)));

        assert!(!analysis.can_stack_allocate(NodeId::new(2)));
        assert!(!analysis.can_scalar_replace(NodeId::new(2)));
    }

    #[test]
    fn test_escape_analysis_iterators() {
        let analysis = EscapeAnalysis {
            states: {
                let mut m = FxHashMap::default();
                m.insert(NodeId::new(0), EscapeState::NoEscape);
                m.insert(NodeId::new(1), EscapeState::NoEscape);
                m.insert(NodeId::new(2), EscapeState::ArgEscape);
                m.insert(NodeId::new(3), EscapeState::GlobalEscape);
                m
            },
            allocations: vec![],
            non_escaping: 2,
            arg_escaping: 1,
            global_escaping: 1,
        };

        let scalar: Vec<_> = analysis.scalar_replaceable().collect();
        assert_eq!(scalar.len(), 2);

        let stack: Vec<_> = analysis.stack_allocatable().collect();
        assert_eq!(stack.len(), 3);
    }

    // -------------------------------------------------------------------------
    // Escape Pass Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_escape_pass_new() {
        let escape = Escape::new();
        assert_eq!(escape.stack_allocated(), 0);
        assert_eq!(escape.scalar_replaced(), 0);
        assert!(!escape.aggressive);
    }

    #[test]
    fn test_escape_pass_aggressive() {
        let escape = Escape::aggressive();
        assert!(escape.aggressive);
    }

    #[test]
    fn test_escape_pass_no_allocs() {
        let mut builder = GraphBuilder::new(1, 1);
        let p0 = builder.parameter(0).unwrap();
        builder.return_value(p0);

        let mut graph = builder.finish();
        let mut escape = Escape::new();

        let changed = escape.run(&mut graph);
        assert!(!changed);
        assert_eq!(escape.stats().allocations_analyzed, 0);
    }

    #[test]
    fn test_escape_pass_default() {
        let escape = Escape::default();
        assert_eq!(escape.stats().scalar_replaced, 0);
    }

    #[test]
    fn test_escape_pass_name() {
        use super::super::OptimizationPass;
        let escape = Escape::new();
        assert_eq!(OptimizationPass::name(&escape), "escape");
    }

    // -------------------------------------------------------------------------
    // EscapeStats Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_escape_stats_default() {
        let stats = EscapeStats::default();
        assert_eq!(stats.allocations_analyzed, 0);
        assert_eq!(stats.non_escaping, 0);
        assert_eq!(stats.scalar_replaced, 0);
        assert_eq!(stats.stack_allocated, 0);
        assert_eq!(stats.nodes_eliminated, 0);
    }

    // -------------------------------------------------------------------------
    // Integration Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_full_analysis_pipeline() {
        // Create a simple graph
        let builder = GraphBuilder::new(4, 2);
        let graph = builder.finish();

        let analysis = EscapeAnalysis::compute(&graph);

        // Should have counted allocations correctly
        let total = analysis.non_escaping_count()
            + analysis.arg_escaping_count()
            + analysis.global_escaping_count();
        assert_eq!(total, analysis.total_allocations());
    }

    #[test]
    fn test_is_allocation() {
        assert!(EscapeAnalysis::is_allocation(&Operator::Memory(
            MemoryOp::Alloc
        )));
        assert!(EscapeAnalysis::is_allocation(&Operator::Memory(
            MemoryOp::AllocArray
        )));
        assert!(EscapeAnalysis::is_allocation(&Operator::BuildList(5)));
        assert!(EscapeAnalysis::is_allocation(&Operator::BuildTuple(3)));
        assert!(EscapeAnalysis::is_allocation(&Operator::BuildDict(2)));

        assert!(!EscapeAnalysis::is_allocation(&Operator::ConstInt(42)));
        assert!(!EscapeAnalysis::is_allocation(&Operator::Phi));
    }

    #[test]
    fn test_classify_use() {
        // Non-escaping uses
        assert_eq!(
            EscapeAnalysis::classify_use(&Operator::Memory(MemoryOp::LoadField)),
            EscapeState::NoEscape
        );
        assert_eq!(
            EscapeAnalysis::classify_use(&Operator::GetItem),
            EscapeState::NoEscape
        );
        assert_eq!(
            EscapeAnalysis::classify_use(&Operator::Guard(crate::ir::operators::GuardKind::Type)),
            EscapeState::NoEscape
        );

        // Arg escape
        assert_eq!(
            EscapeAnalysis::classify_use(&Operator::Call(crate::ir::operators::CallKind::Direct)),
            EscapeState::ArgEscape
        );

        // Global escape
        assert_eq!(
            EscapeAnalysis::classify_use(&Operator::Control(ControlOp::Return)),
            EscapeState::GlobalEscape
        );
    }

    #[test]
    fn test_infer_object_type() {
        use crate::ir::graph::Graph;
        use crate::ir::node::InputList;

        let mut graph = Graph::new();

        // Create different allocation types
        let list_alloc = graph.add_node_with_type(
            Operator::BuildList(5),
            InputList::Empty,
            crate::ir::types::ValueType::List,
        );
        let tuple_alloc = graph.add_node_with_type(
            Operator::BuildTuple(3),
            InputList::Empty,
            crate::ir::types::ValueType::Tuple,
        );
        let dict_alloc = graph.add_node_with_type(
            Operator::BuildDict(2),
            InputList::Empty,
            crate::ir::types::ValueType::Dict,
        );

        assert_eq!(
            EscapeAnalysis::infer_object_type(&graph, list_alloc),
            ObjectType::Array
        );
        assert_eq!(
            EscapeAnalysis::infer_object_type(&graph, tuple_alloc),
            ObjectType::Tuple
        );
        assert_eq!(
            EscapeAnalysis::infer_object_type(&graph, dict_alloc),
            ObjectType::Dict
        );
    }

    #[test]
    fn test_estimate_size() {
        use crate::ir::graph::Graph;
        use crate::ir::node::InputList;

        let mut graph = Graph::new();

        let list_alloc = graph.add_node_with_type(
            Operator::BuildList(10),
            InputList::Empty,
            crate::ir::types::ValueType::List,
        );
        let tuple_alloc = graph.add_node_with_type(
            Operator::BuildTuple(5),
            InputList::Empty,
            crate::ir::types::ValueType::Tuple,
        );

        // List: 64 + 10*8 = 144
        assert_eq!(EscapeAnalysis::estimate_size(&graph, list_alloc), Some(144));
        // Tuple: 32 + 5*8 = 72
        assert_eq!(EscapeAnalysis::estimate_size(&graph, tuple_alloc), Some(72));
    }
}
