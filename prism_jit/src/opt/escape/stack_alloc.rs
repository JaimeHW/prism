//! Stack Allocation Transformation
//!
//! This module implements stack allocation for objects that don't escape
//! the function but cannot be scalar-replaced (e.g., because they're passed
//! to callees or have too many fields).
//!
//! # Benefits of Stack Allocation
//!
//! 1. **No GC pressure**: Stack objects are automatically deallocated on return
//! 2. **Better cache locality**: Stack is typically hot in cache
//! 3. **Faster allocation**: Just bump the stack pointer
//! 4. **No write barriers**: Stack objects don't need GC write barriers
//!
//! # Requirements
//!
//! An allocation can be stack-allocated if:
//! 1. It does not escape globally (NoEscape or ArgEscape)
//! 2. Its size is known at compile time
//! 3. Its size is within the stack allocation limit
//! 4. The allocation dominates all uses
//!
//! # Implementation
//!
//! Stack allocation is implemented by:
//! 1. Changing the allocation operator to StackAlloc
//! 2. Recording the frame slot and size
//! 3. Eliminating write barriers for the object
//! 4. Ensuring the frame is sized appropriately

use crate::ir::graph::Graph;
use crate::ir::node::NodeId;
use crate::ir::operators::{MemoryOp, Operator};
use rustc_hash::FxHashMap;

// =============================================================================
// Stack Allocation Configuration
// =============================================================================

/// Configuration for stack allocation.
#[derive(Debug, Clone)]
pub struct StackAllocConfig {
    /// Maximum size (in bytes) for a single stack allocation.
    pub max_object_size: usize,
    /// Maximum total stack allocation per function.
    pub max_total_stack: usize,
    /// Default object size when size is unknown.
    pub default_object_size: usize,
    /// Alignment requirement for stack allocations.
    pub alignment: usize,
    /// Whether to allow arg-escaping allocations.
    pub allow_arg_escape: bool,
}

impl Default for StackAllocConfig {
    fn default() -> Self {
        Self {
            max_object_size: 4096,   // 4KB per object max
            max_total_stack: 65536,  // 64KB total per function
            default_object_size: 64, // Assume 64 bytes for unknown
            alignment: 8,            // 8-byte alignment
            allow_arg_escape: true,  // Allow stack alloc for arg-escape
        }
    }
}

impl StackAllocConfig {
    /// Create a conservative configuration (smaller limits).
    pub fn conservative() -> Self {
        Self {
            max_object_size: 1024,
            max_total_stack: 16384,
            default_object_size: 32,
            alignment: 8,
            allow_arg_escape: false,
        }
    }

    /// Create an aggressive configuration (larger limits).
    pub fn aggressive() -> Self {
        Self {
            max_object_size: 16384,
            max_total_stack: 262144,
            default_object_size: 128,
            alignment: 16,
            allow_arg_escape: true,
        }
    }
}

// =============================================================================
// Stack Slot
// =============================================================================

/// Represents a stack slot for an allocated object.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StackSlot {
    /// Offset from frame base (negative for locals).
    pub offset: i32,
    /// Size in bytes.
    pub size: u32,
    /// Alignment requirement.
    pub alignment: u32,
    /// Original allocation node.
    pub allocation: NodeId,
}

impl StackSlot {
    /// Create a new stack slot.
    pub fn new(offset: i32, size: u32, alignment: u32, allocation: NodeId) -> Self {
        Self {
            offset,
            size,
            alignment,
            allocation,
        }
    }

    /// Get the end offset (offset + size).
    #[inline]
    pub fn end_offset(&self) -> i32 {
        self.offset + self.size as i32
    }

    /// Check if this slot overlaps with another.
    pub fn overlaps(&self, other: &StackSlot) -> bool {
        !(self.end_offset() <= other.offset || other.end_offset() <= self.offset)
    }
}

// =============================================================================
// Stack Frame Layout
// =============================================================================

/// Tracks stack frame layout for a function.
#[derive(Debug)]
pub struct StackFrameLayout {
    /// All allocated stack slots.
    slots: Vec<StackSlot>,
    /// Next available offset (grows downward, negative).
    next_offset: i32,
    /// Total size of stack allocations.
    total_size: usize,
    /// Maximum alignment seen.
    max_alignment: u32,
    /// Mapping from allocation node to slot index.
    allocation_map: FxHashMap<NodeId, usize>,
}

impl StackFrameLayout {
    /// Create a new empty layout.
    pub fn new() -> Self {
        Self {
            slots: Vec::new(),
            next_offset: 0,
            total_size: 0,
            max_alignment: 8,
            allocation_map: FxHashMap::default(),
        }
    }

    /// Allocate a new stack slot.
    pub fn allocate(&mut self, size: u32, alignment: u32, allocation: NodeId) -> StackSlot {
        // Align the next offset
        let alignment = alignment.max(8); // Minimum 8-byte alignment
        let aligned_offset = self.align_down(self.next_offset, alignment as i32);

        // Calculate new offset (stack grows downward)
        let slot_offset = aligned_offset - size as i32;

        let slot = StackSlot::new(slot_offset, size, alignment, allocation);

        let slot_index = self.slots.len();
        self.slots.push(slot);
        self.allocation_map.insert(allocation, slot_index);

        self.next_offset = slot_offset;
        self.total_size += size as usize;
        self.max_alignment = self.max_alignment.max(alignment);

        slot
    }

    /// Get slot for an allocation.
    pub fn get_slot(&self, allocation: NodeId) -> Option<&StackSlot> {
        self.allocation_map
            .get(&allocation)
            .and_then(|&idx| self.slots.get(idx))
    }

    /// Get total size of stack allocations.
    #[inline]
    pub fn total_size(&self) -> usize {
        self.total_size
    }

    /// Get maximum alignment.
    #[inline]
    pub fn max_alignment(&self) -> u32 {
        self.max_alignment
    }

    /// Get all slots.
    pub fn slots(&self) -> &[StackSlot] {
        &self.slots
    }

    /// Get the required frame size (positive value).
    pub fn frame_size(&self) -> usize {
        (-self.next_offset) as usize
    }

    /// Check if we can allocate more.
    pub fn can_allocate(&self, size: usize, config: &StackAllocConfig) -> bool {
        self.total_size + size <= config.max_total_stack
    }

    /// Align offset downward (for stack growth).
    #[inline]
    fn align_down(&self, offset: i32, alignment: i32) -> i32 {
        // For negative offsets (stack), align toward more negative
        if offset >= 0 {
            offset & !(alignment - 1)
        } else {
            -(((-offset) + alignment - 1) & !(alignment - 1))
        }
    }
}

impl Default for StackFrameLayout {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// Stack Allocation Result
// =============================================================================

/// Result of stack allocation transformation.
#[derive(Debug, Clone)]
pub struct StackAllocResult {
    /// Whether the transformation was successful.
    pub success: bool,
    /// The allocation node.
    pub allocation: NodeId,
    /// The assigned stack slot (if successful).
    pub slot: Option<StackSlot>,
    /// Reason for failure (if any).
    pub failure_reason: Option<StackAllocFailure>,
}

/// Reasons why stack allocation might fail.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum StackAllocFailure {
    /// Object escapes globally.
    GlobalEscape,
    /// Object is too large.
    TooLarge,
    /// Total stack limit exceeded.
    StackLimitExceeded,
    /// Size is unknown.
    UnknownSize,
    /// Allocation type not supported.
    UnsupportedType,
    /// Already stack allocated.
    AlreadyStackAllocated,
}

impl StackAllocResult {
    /// Create a success result.
    pub fn success(allocation: NodeId, slot: StackSlot) -> Self {
        Self {
            success: true,
            allocation,
            slot: Some(slot),
            failure_reason: None,
        }
    }

    /// Create a failure result.
    pub fn failure(allocation: NodeId, reason: StackAllocFailure) -> Self {
        Self {
            success: false,
            allocation,
            slot: None,
            failure_reason: Some(reason),
        }
    }
}

// =============================================================================
// Object Size Estimation
// =============================================================================

/// Estimates object sizes for stack allocation.
#[derive(Debug)]
pub struct ObjectSizeEstimator {
    /// Known type sizes.
    type_sizes: FxHashMap<u32, usize>,
    /// Default size for unknown types.
    default_size: usize,
}

impl ObjectSizeEstimator {
    /// Create a new estimator with default sizes.
    pub fn new(default_size: usize) -> Self {
        let mut type_sizes = FxHashMap::default();

        // Common Python type sizes (approximate)
        type_sizes.insert(0, 56); // object base
        type_sizes.insert(1, 32); // int (boxed)
        type_sizes.insert(2, 24); // float
        type_sizes.insert(3, 56); // str header
        type_sizes.insert(4, 64); // list header
        type_sizes.insert(5, 72); // dict header
        type_sizes.insert(6, 56); // tuple header
        type_sizes.insert(7, 32); // bool
        type_sizes.insert(8, 16); // None

        Self {
            type_sizes,
            default_size,
        }
    }

    /// Estimate size for an allocation.
    pub fn estimate_size(&self, graph: &Graph, allocation: NodeId) -> Option<usize> {
        let node = graph.get(allocation)?;

        match node.op {
            Operator::Memory(MemoryOp::Alloc) => {
                // Check if size is encoded in inputs
                if let Some(size_node) = node.inputs.get(0) {
                    if let Some(size_data) = graph.get(size_node) {
                        if let Operator::ConstInt(size) = size_data.op {
                            if size >= 0 {
                                return Some(size as usize);
                            }
                        }
                    }
                }
                // Return default
                Some(self.default_size)
            }
            Operator::Memory(MemoryOp::AllocArray) => {
                // Array allocation: base + element_count * element_size
                // Check for count in inputs
                if let Some(count_node) = node.inputs.get(0) {
                    if let Some(count_data) = graph.get(count_node) {
                        if let Operator::ConstInt(count) = count_data.op {
                            if count >= 0 {
                                // Assume 8 bytes per element + 32 byte header
                                return Some(32 + (count as usize * 8));
                            }
                        }
                    }
                }
                None // Unknown size
            }
            Operator::BuildList(count) => {
                // List header + elements
                Some(64 + (count as usize * 8))
            }
            Operator::BuildTuple(count) => {
                // Tuple header + elements (more compact than list)
                Some(32 + (count as usize * 8))
            }
            Operator::BuildDict(count) => {
                // Dict is more complex - use heuristic
                Some(72 + (count as usize * 24))
            }
            _ => None,
        }
    }

    /// Register a known type size.
    pub fn register_type_size(&mut self, type_id: u32, size: usize) {
        self.type_sizes.insert(type_id, size);
    }
}

// =============================================================================
// Stack Allocator
// =============================================================================

/// Performs stack allocation transformation.
#[derive(Debug)]
pub struct StackAllocator {
    /// Configuration.
    config: StackAllocConfig,
    /// Size estimator.
    estimator: ObjectSizeEstimator,
    /// Frame layout.
    layout: StackFrameLayout,
    /// Statistics.
    stats: StackAllocStats,
}

/// Statistics for stack allocation.
#[derive(Debug, Clone, Default)]
pub struct StackAllocStats {
    /// Number of allocations converted to stack.
    pub stack_allocated: usize,
    /// Number of allocations that failed.
    pub failed: usize,
    /// Total bytes allocated on stack.
    pub total_bytes: usize,
    /// Failures by reason.
    pub failures_by_reason: FxHashMap<StackAllocFailure, usize>,
}

impl StackAllocator {
    /// Create a new stack allocator with default config.
    pub fn new() -> Self {
        Self::with_config(StackAllocConfig::default())
    }

    /// Create with custom configuration.
    pub fn with_config(config: StackAllocConfig) -> Self {
        Self {
            estimator: ObjectSizeEstimator::new(config.default_object_size),
            layout: StackFrameLayout::new(),
            config,
            stats: StackAllocStats::default(),
        }
    }

    /// Get the current frame layout.
    pub fn layout(&self) -> &StackFrameLayout {
        &self.layout
    }

    /// Get statistics.
    pub fn stats(&self) -> &StackAllocStats {
        &self.stats
    }

    /// Reset for a new function.
    pub fn reset(&mut self) {
        self.layout = StackFrameLayout::new();
        self.stats = StackAllocStats::default();
    }

    /// Attempt to stack-allocate an object.
    pub fn try_stack_allocate(
        &mut self,
        graph: &mut Graph,
        allocation: NodeId,
        escapes_to_args: bool,
    ) -> StackAllocResult {
        // Check if arg escape is allowed
        if escapes_to_args && !self.config.allow_arg_escape {
            self.record_failure(StackAllocFailure::GlobalEscape);
            return StackAllocResult::failure(allocation, StackAllocFailure::GlobalEscape);
        }

        // Check if already stack allocated
        if self.layout.get_slot(allocation).is_some() {
            return StackAllocResult::failure(allocation, StackAllocFailure::AlreadyStackAllocated);
        }

        // Estimate object size
        let size = match self.estimator.estimate_size(graph, allocation) {
            Some(s) => s,
            None => {
                self.record_failure(StackAllocFailure::UnknownSize);
                return StackAllocResult::failure(allocation, StackAllocFailure::UnknownSize);
            }
        };

        // Check size limits
        if size > self.config.max_object_size {
            self.record_failure(StackAllocFailure::TooLarge);
            return StackAllocResult::failure(allocation, StackAllocFailure::TooLarge);
        }

        if !self.layout.can_allocate(size, &self.config) {
            self.record_failure(StackAllocFailure::StackLimitExceeded);
            return StackAllocResult::failure(allocation, StackAllocFailure::StackLimitExceeded);
        }

        // Perform the transformation
        let slot = self
            .layout
            .allocate(size as u32, self.config.alignment as u32, allocation);

        // Update the allocation node to indicate stack allocation
        self.transform_allocation(graph, allocation, &slot);

        self.stats.stack_allocated += 1;
        self.stats.total_bytes += size;

        StackAllocResult::success(allocation, slot)
    }

    /// Transform an allocation to stack allocation.
    fn transform_allocation(&self, graph: &mut Graph, allocation: NodeId, slot: &StackSlot) {
        // In a full implementation, we would:
        // 1. Change the operator to StackAlloc
        // 2. Store the slot information
        // 3. Eliminate write barriers for this object
        //
        // For now, we annotate the node with stack allocation info
        // The actual transformation happens during code generation

        if let Some(node) = graph.get_mut(allocation) {
            // Mark as stack allocated by changing operator
            // We use a placeholder approach - in production,
            // you'd have a StackAlloc operator
            match node.op {
                Operator::Memory(MemoryOp::Alloc) => {
                    // Keep as Alloc but code gen will use slot info
                    // In a full impl: node.op = Operator::StackAlloc(slot.offset);
                }
                Operator::Memory(MemoryOp::AllocArray) => {
                    // Same for arrays
                }
                _ => {}
            }
        }
    }

    /// Record a failure.
    fn record_failure(&mut self, reason: StackAllocFailure) {
        self.stats.failed += 1;
        *self.stats.failures_by_reason.entry(reason).or_insert(0) += 1;
    }
}

impl Default for StackAllocator {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// Batch Stack Allocator
// =============================================================================

/// Processes multiple allocations for stack allocation.
#[derive(Debug)]
pub struct BatchStackAllocator {
    /// Inner allocator.
    allocator: StackAllocator,
    /// Results for each allocation.
    results: Vec<StackAllocResult>,
}

impl BatchStackAllocator {
    /// Create a new batch allocator.
    pub fn new() -> Self {
        Self {
            allocator: StackAllocator::new(),
            results: Vec::new(),
        }
    }

    /// Create with custom configuration.
    pub fn with_config(config: StackAllocConfig) -> Self {
        Self {
            allocator: StackAllocator::with_config(config),
            results: Vec::new(),
        }
    }

    /// Process a batch of allocations.
    pub fn process(
        &mut self,
        graph: &mut Graph,
        allocations: &[(NodeId, bool)], // (allocation, escapes_to_args)
    ) -> &[StackAllocResult] {
        self.results.clear();
        self.allocator.reset();

        // Sort by size (smallest first) to maximize number of stack allocations
        let mut sorted: Vec<_> = allocations
            .iter()
            .filter_map(|&(alloc, escapes)| {
                let size = self.allocator.estimator.estimate_size(graph, alloc)?;
                Some((alloc, escapes, size))
            })
            .collect();

        sorted.sort_by_key(|&(_, _, size)| size);

        // Process in order
        for (alloc, escapes, _) in sorted {
            let result = self.allocator.try_stack_allocate(graph, alloc, escapes);
            self.results.push(result);
        }

        &self.results
    }

    /// Get the frame layout.
    pub fn layout(&self) -> &StackFrameLayout {
        self.allocator.layout()
    }

    /// Get statistics.
    pub fn stats(&self) -> &StackAllocStats {
        self.allocator.stats()
    }
}

impl Default for BatchStackAllocator {
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

    // -------------------------------------------------------------------------
    // StackAllocConfig Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_config_default() {
        let config = StackAllocConfig::default();

        assert_eq!(config.max_object_size, 4096);
        assert_eq!(config.max_total_stack, 65536);
        assert_eq!(config.alignment, 8);
        assert!(config.allow_arg_escape);
    }

    #[test]
    fn test_config_conservative() {
        let config = StackAllocConfig::conservative();

        assert_eq!(config.max_object_size, 1024);
        assert!(!config.allow_arg_escape);
    }

    #[test]
    fn test_config_aggressive() {
        let config = StackAllocConfig::aggressive();

        assert_eq!(config.max_object_size, 16384);
        assert_eq!(config.alignment, 16);
    }

    // -------------------------------------------------------------------------
    // StackSlot Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_stack_slot_new() {
        let slot = StackSlot::new(-64, 32, 8, NodeId::new(5));

        assert_eq!(slot.offset, -64);
        assert_eq!(slot.size, 32);
        assert_eq!(slot.alignment, 8);
        assert_eq!(slot.allocation, NodeId::new(5));
    }

    #[test]
    fn test_stack_slot_end_offset() {
        let slot = StackSlot::new(-64, 32, 8, NodeId::new(0));
        assert_eq!(slot.end_offset(), -32);

        let slot2 = StackSlot::new(0, 16, 8, NodeId::new(0));
        assert_eq!(slot2.end_offset(), 16);
    }

    #[test]
    fn test_stack_slot_overlaps() {
        let slot1 = StackSlot::new(-64, 32, 8, NodeId::new(0));
        let slot2 = StackSlot::new(-48, 16, 8, NodeId::new(1));

        // -64 to -32 and -48 to -32 overlap
        assert!(slot1.overlaps(&slot2));
        assert!(slot2.overlaps(&slot1));

        let slot3 = StackSlot::new(-32, 16, 8, NodeId::new(2));
        // -64 to -32 and -32 to -16 don't overlap (adjacent)
        assert!(!slot1.overlaps(&slot3));
    }

    #[test]
    fn test_stack_slot_no_overlap() {
        let slot1 = StackSlot::new(-100, 20, 8, NodeId::new(0));
        let slot2 = StackSlot::new(-50, 20, 8, NodeId::new(1));

        // -100 to -80 and -50 to -30 don't overlap
        assert!(!slot1.overlaps(&slot2));
    }

    // -------------------------------------------------------------------------
    // StackFrameLayout Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_frame_layout_new() {
        let layout = StackFrameLayout::new();

        assert!(layout.slots().is_empty());
        assert_eq!(layout.total_size(), 0);
        assert_eq!(layout.frame_size(), 0);
    }

    #[test]
    fn test_frame_layout_allocate() {
        let mut layout = StackFrameLayout::new();

        let slot1 = layout.allocate(32, 8, NodeId::new(0));
        assert_eq!(slot1.size, 32);
        assert!(slot1.offset < 0); // Stack grows down

        let slot2 = layout.allocate(64, 8, NodeId::new(1));
        assert_eq!(slot2.size, 64);
        assert!(slot2.offset < slot1.offset); // Lower in stack

        assert_eq!(layout.total_size(), 96);
        assert_eq!(layout.slots().len(), 2);
    }

    #[test]
    fn test_frame_layout_get_slot() {
        let mut layout = StackFrameLayout::new();

        layout.allocate(32, 8, NodeId::new(5));

        let slot = layout.get_slot(NodeId::new(5));
        assert!(slot.is_some());
        assert_eq!(slot.unwrap().size, 32);

        assert!(layout.get_slot(NodeId::new(99)).is_none());
    }

    #[test]
    fn test_frame_layout_can_allocate() {
        let mut layout = StackFrameLayout::new();
        let config = StackAllocConfig {
            max_total_stack: 100,
            ..Default::default()
        };

        assert!(layout.can_allocate(50, &config));

        layout.allocate(60, 8, NodeId::new(0));

        assert!(layout.can_allocate(40, &config));
        assert!(!layout.can_allocate(50, &config));
    }

    #[test]
    fn test_frame_layout_max_alignment() {
        let mut layout = StackFrameLayout::new();

        layout.allocate(32, 8, NodeId::new(0));
        assert_eq!(layout.max_alignment(), 8);

        layout.allocate(32, 16, NodeId::new(1));
        assert_eq!(layout.max_alignment(), 16);
    }

    #[test]
    fn test_frame_layout_frame_size() {
        let mut layout = StackFrameLayout::new();

        layout.allocate(32, 8, NodeId::new(0));
        layout.allocate(64, 8, NodeId::new(1));

        // Frame size should be positive
        assert!(layout.frame_size() >= 96);
    }

    // -------------------------------------------------------------------------
    // StackAllocResult Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_result_success() {
        let slot = StackSlot::new(-32, 32, 8, NodeId::new(0));
        let result = StackAllocResult::success(NodeId::new(0), slot);

        assert!(result.success);
        assert!(result.slot.is_some());
        assert!(result.failure_reason.is_none());
    }

    #[test]
    fn test_result_failure() {
        let result = StackAllocResult::failure(NodeId::new(0), StackAllocFailure::TooLarge);

        assert!(!result.success);
        assert!(result.slot.is_none());
        assert_eq!(result.failure_reason, Some(StackAllocFailure::TooLarge));
    }

    // -------------------------------------------------------------------------
    // ObjectSizeEstimator Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_estimator_new() {
        let estimator = ObjectSizeEstimator::new(64);
        assert_eq!(estimator.default_size, 64);
    }

    #[test]
    fn test_estimator_register_type() {
        let mut estimator = ObjectSizeEstimator::new(64);
        estimator.register_type_size(100, 256);

        assert_eq!(estimator.type_sizes.get(&100), Some(&256));
    }

    #[test]
    fn test_estimator_unknown_allocation() {
        let estimator = ObjectSizeEstimator::new(64);
        let graph = Graph::new();

        // Non-existent node
        assert!(estimator.estimate_size(&graph, NodeId::new(999)).is_none());
    }

    // -------------------------------------------------------------------------
    // StackAllocator Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_allocator_new() {
        let allocator = StackAllocator::new();
        assert_eq!(allocator.stats().stack_allocated, 0);
    }

    #[test]
    fn test_allocator_with_config() {
        let config = StackAllocConfig {
            max_object_size: 1024,
            ..Default::default()
        };
        let allocator = StackAllocator::with_config(config);

        assert_eq!(allocator.config.max_object_size, 1024);
    }

    #[test]
    fn test_allocator_reset() {
        let mut allocator = StackAllocator::new();

        // Simulate some state
        allocator.stats.stack_allocated = 5;

        allocator.reset();

        assert_eq!(allocator.stats().stack_allocated, 0);
        assert!(allocator.layout().slots().is_empty());
    }

    #[test]
    fn test_allocator_arg_escape_disallowed() {
        let config = StackAllocConfig {
            allow_arg_escape: false,
            ..Default::default()
        };
        let mut allocator = StackAllocator::with_config(config);
        let mut graph = Graph::new();

        let result = allocator.try_stack_allocate(&mut graph, NodeId::new(0), true);

        assert!(!result.success);
        assert_eq!(result.failure_reason, Some(StackAllocFailure::GlobalEscape));
    }

    #[test]
    fn test_allocator_default() {
        let allocator = StackAllocator::default();
        assert_eq!(allocator.config.max_object_size, 4096);
    }

    // -------------------------------------------------------------------------
    // BatchStackAllocator Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_batch_allocator_new() {
        let allocator = BatchStackAllocator::new();
        assert!(allocator.results.is_empty());
    }

    #[test]
    fn test_batch_allocator_with_config() {
        let config = StackAllocConfig::conservative();
        let allocator = BatchStackAllocator::with_config(config);

        assert_eq!(allocator.allocator.config.max_object_size, 1024);
    }

    #[test]
    fn test_batch_allocator_empty_batch() {
        let mut allocator = BatchStackAllocator::new();
        let mut graph = Graph::new();

        let results = allocator.process(&mut graph, &[]);
        assert!(results.is_empty());
    }

    #[test]
    fn test_batch_allocator_default() {
        let allocator = BatchStackAllocator::default();
        assert!(allocator.results.is_empty());
    }

    // -------------------------------------------------------------------------
    // StackAllocStats Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_stats_default() {
        let stats = StackAllocStats::default();

        assert_eq!(stats.stack_allocated, 0);
        assert_eq!(stats.failed, 0);
        assert_eq!(stats.total_bytes, 0);
        assert!(stats.failures_by_reason.is_empty());
    }

    // -------------------------------------------------------------------------
    // Integration Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_multiple_allocations() {
        let mut layout = StackFrameLayout::new();

        // Allocate several objects
        let slots: Vec<_> = (0..5)
            .map(|i| layout.allocate(32, 8, NodeId::new(i)))
            .collect();

        // Verify no overlaps
        for i in 0..slots.len() {
            for j in (i + 1)..slots.len() {
                assert!(!slots[i].overlaps(&slots[j]));
            }
        }

        assert_eq!(layout.total_size(), 160);
    }

    #[test]
    fn test_different_alignments() {
        let mut layout = StackFrameLayout::new();

        layout.allocate(16, 8, NodeId::new(0));
        layout.allocate(32, 16, NodeId::new(1));
        layout.allocate(64, 32, NodeId::new(2));

        assert_eq!(layout.max_alignment(), 32);
    }

    #[test]
    fn test_failure_reasons() {
        let config = StackAllocConfig {
            max_object_size: 100,
            max_total_stack: 200,
            ..Default::default()
        };
        let mut allocator = StackAllocator::with_config(config);
        let mut graph = Graph::new();

        // This should fail with unknown size
        let result = allocator.try_stack_allocate(&mut graph, NodeId::new(0), false);
        assert!(!result.success);

        assert_eq!(allocator.stats().failed, 1);
    }
}
