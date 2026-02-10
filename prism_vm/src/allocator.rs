//! High-performance GC allocator for VM object allocation.
//!
//! This module provides a typed allocation API that bridges the gap between
//! the low-level `GcHeap` and the high-level VM operations. It ensures:
//!
//! - **Type Safety**: Allocations are properly typed and sized
//! - **Zero-Copy Initialization**: Objects are constructed directly in GC memory
//! - **Trace Enforcement**: Only `Trace`-implementing types can be allocated
//! - **Cache-Friendly Access**: Allocator is designed for hot-path inline expansion
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────┐
//! │                       VM Operations                              │
//! │  (containers.rs, calls.rs, objects.rs, subscript.rs)            │
//! └────────────────────────────┬────────────────────────────────────┘
//!                              │ alloc<T>() / alloc_value<T>()
//!                              ▼
//! ┌─────────────────────────────────────────────────────────────────┐
//! │                       GcAllocator                                │
//! │  - Typed allocation with automatic sizing                       │
//! │  - Direct Value construction                                    │
//! │  - OOM handling with AllocResult                                │
//! └────────────────────────────┬────────────────────────────────────┘
//!                              │ heap.alloc(size)
//!                              ▼
//! ┌─────────────────────────────────────────────────────────────────┐
//! │                       GcHeap                                     │
//! │  - Nursery (bump allocation)                                    │
//! │  - Old space (block allocation)                                 │
//! │  - Large object space                                           │
//! └─────────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Performance
//!
//! The allocator is designed for maximum performance:
//!
//! - **Inline Allocation**: All methods are `#[inline]` for hot-path optimization
//! - **No Branching**: Common path has minimal branches
//! - **Cache Locality**: Objects are allocated in contiguous nursery space
//! - **Zero Overhead**: Thin wrapper with no runtime checks beyond OOM

use prism_core::Value;
use prism_gc::GcHeap;
use prism_gc::trace::Trace;

use std::alloc::Layout;
use std::ptr::NonNull;

// =============================================================================
// GC Allocator
// =============================================================================

/// High-performance typed GC allocator.
///
/// Provides zero-cost typed allocation over the underlying `GcHeap`.
/// All allocations are properly sized and aligned for the target type.
///
/// # Usage
///
/// ```ignore
/// let allocator = GcAllocator::new(&heap);
///
/// // Allocate a list
/// let list_ptr = allocator.alloc(ListObject::new())?;
///
/// // Allocate and get Value directly
/// let list_val = allocator.alloc_value(ListObject::new())?;
/// ```
///
/// # Thread Safety
///
/// `GcAllocator` borrows the heap and is not `Send` or `Sync`.
/// Each thread should use its own allocator instance.
pub struct GcAllocator<'h> {
    /// Reference to the GC heap for allocation.
    heap: &'h GcHeap,
}

impl<'h> GcAllocator<'h> {
    /// Create a new allocator for the given heap.
    ///
    /// # Performance
    ///
    /// This is a zero-cost operation that just stores a reference.
    #[inline]
    pub const fn new(heap: &'h GcHeap) -> Self {
        Self { heap }
    }

    /// Allocate and initialize a GC-managed object.
    ///
    /// Returns a raw pointer to the allocated object, or `None` if
    /// the nursery is full and collection is needed.
    ///
    /// # Type Parameters
    ///
    /// - `T`: The type to allocate. Must implement `Trace` for GC integration.
    ///
    /// # Safety
    ///
    /// The returned pointer is valid until the next GC collection that
    /// determines the object is unreachable. Callers must ensure the
    /// object is rooted before the next GC safe point.
    ///
    /// # Performance
    ///
    /// - Fast path: Single bump-pointer increment + object initialization
    /// - Inlined for zero call overhead on hot paths
    #[inline]
    pub fn alloc<T: Trace>(&self, value: T) -> Option<*mut T> {
        let layout = Layout::new::<T>();
        let size = layout.size().max(8); // Minimum 8-byte allocation

        // Allocate raw memory from heap
        let ptr = self.heap.alloc(size)?;

        // Initialize the object in-place
        let typed_ptr = ptr.as_ptr() as *mut T;
        unsafe {
            std::ptr::write(typed_ptr, value);
        }

        Some(typed_ptr)
    }

    /// Allocate an object and return it as a Value.
    ///
    /// Convenience method that combines allocation with Value construction.
    /// This is the primary allocation method for VM operations.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let val = allocator.alloc_value(ListObject::from_slice(&[
    ///     Value::int(1).unwrap(),
    ///     Value::int(2).unwrap(),
    /// ]))?;
    /// vm.set_register(dst, val);
    /// ```
    ///
    /// # Performance
    ///
    /// Single inline call with no additional overhead over `alloc()`.
    #[inline]
    pub fn alloc_value<T: Trace>(&self, value: T) -> Option<Value> {
        let ptr = self.alloc(value)?;
        Some(Value::object_ptr(ptr as *const ()))
    }

    /// Try to allocate, with explicit OOM handling.
    ///
    /// Returns `AllocResult` for cases where the caller needs to
    /// distinguish between "needs collection" and "out of memory".
    #[inline]
    pub fn try_alloc<T: Trace>(&self, value: T) -> AllocResult<*mut T> {
        match self.alloc(value) {
            Some(ptr) => AllocResult::Ok(ptr),
            None => {
                // Check if this was nursery-full or true OOM
                if self.heap.should_minor_collect() {
                    AllocResult::NeedsCollection
                } else {
                    AllocResult::OutOfMemory
                }
            }
        }
    }

    /// Allocate with explicit size and alignment.
    ///
    /// Low-level allocation for variable-sized objects or custom layouts.
    /// The caller is responsible for initialization.
    ///
    /// # Safety
    ///
    /// - The returned memory is uninitialized
    /// - Must be initialized before next GC
    /// - Size must be accurate for the object
    #[inline]
    pub fn alloc_raw(&self, size: usize) -> Option<NonNull<u8>> {
        self.heap.alloc(size)
    }

    /// Allocate directly in the old generation.
    ///
    /// Use for long-lived objects that should skip the nursery.
    /// This avoids copying during minor collections.
    ///
    /// # Use Cases
    ///
    /// - Module-level objects
    /// - Cached/interned objects
    /// - Objects known to be long-lived
    #[inline]
    pub fn alloc_tenured<T: Trace>(&self, value: T) -> Option<*mut T> {
        let layout = Layout::new::<T>();
        let size = layout.size().max(8);

        let ptr = self.heap.alloc_tenured(size)?;

        let typed_ptr = ptr.as_ptr() as *mut T;
        unsafe {
            std::ptr::write(typed_ptr, value);
        }

        Some(typed_ptr)
    }

    /// Check if the allocator can satisfy an allocation of the given size.
    ///
    /// Non-allocating check for pre-flight verification.
    #[inline]
    pub fn can_alloc(&self, size: usize) -> bool {
        // Check if nursery has space
        !self.heap.should_minor_collect() || size >= self.heap.config().large_object_threshold
    }

    /// Get heap statistics for monitoring.
    #[inline]
    pub fn stats(&self) -> &prism_gc::GcStats {
        self.heap.stats()
    }
}

// =============================================================================
// Allocation Results
// =============================================================================

/// Result type for allocation operations with explicit error cases.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AllocResult<T> {
    /// Allocation succeeded.
    Ok(T),
    /// Nursery full - minor collection may free space.
    NeedsCollection,
    /// True out-of-memory condition.
    OutOfMemory,
}

impl<T> AllocResult<T> {
    /// Convert to Option, discarding error information.
    #[inline]
    pub fn ok(self) -> Option<T> {
        match self {
            AllocResult::Ok(val) => Some(val),
            _ => None,
        }
    }

    /// Check if allocation succeeded.
    #[inline]
    pub fn is_ok(&self) -> bool {
        matches!(self, AllocResult::Ok(_))
    }

    /// Check if collection is needed.
    #[inline]
    pub fn needs_collection(&self) -> bool {
        matches!(self, AllocResult::NeedsCollection)
    }

    /// Map the success value.
    #[inline]
    pub fn map<U, F: FnOnce(T) -> U>(self, f: F) -> AllocResult<U> {
        match self {
            AllocResult::Ok(val) => AllocResult::Ok(f(val)),
            AllocResult::NeedsCollection => AllocResult::NeedsCollection,
            AllocResult::OutOfMemory => AllocResult::OutOfMemory,
        }
    }

    /// Unwrap or panic with message.
    #[inline]
    pub fn expect(self, msg: &str) -> T {
        match self {
            AllocResult::Ok(val) => val,
            AllocResult::NeedsCollection => panic!("{}: needs collection", msg),
            AllocResult::OutOfMemory => panic!("{}: out of memory", msg),
        }
    }

    /// Unwrap or panic.
    #[inline]
    pub fn unwrap(self) -> T {
        self.expect("allocation failed")
    }
}

// =============================================================================
// Convenience Traits
// =============================================================================

/// Extension trait for allocating from any heap reference.
pub trait HeapAllocExt {
    /// Create an allocator for this heap.
    fn allocator(&self) -> GcAllocator<'_>;
}

impl HeapAllocExt for GcHeap {
    #[inline]
    fn allocator(&self) -> GcAllocator<'_> {
        GcAllocator::new(self)
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use prism_gc::config::GcConfig;
    use prism_gc::trace::Tracer;

    /// Simple traceable test object.
    #[derive(Debug, Clone, PartialEq)]
    struct TestObject {
        value: i64,
        data: [u8; 32],
    }

    impl TestObject {
        fn new(value: i64) -> Self {
            Self {
                value,
                data: [0; 32],
            }
        }
    }

    unsafe impl Trace for TestObject {
        fn trace(&self, _tracer: &mut dyn Tracer) {}
    }

    /// Object with nested references for trace testing.
    #[derive(Debug)]
    struct NestedObject {
        values: Vec<Value>,
    }

    unsafe impl Trace for NestedObject {
        fn trace(&self, tracer: &mut dyn Tracer) {
            for v in &self.values {
                tracer.trace_value(*v);
            }
        }
    }

    // -------------------------------------------------------------------------
    // GcAllocator Construction Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_allocator_new() {
        let heap = GcHeap::new(GcConfig::default());
        let _allocator = GcAllocator::new(&heap);
        // Just verify it compiles and doesn't panic
    }

    #[test]
    fn test_allocator_from_heap_ext() {
        let heap = GcHeap::new(GcConfig::default());
        let _allocator = heap.allocator();
    }

    // -------------------------------------------------------------------------
    // Basic Allocation Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_alloc_simple() {
        let heap = GcHeap::new(GcConfig::default());
        let allocator = GcAllocator::new(&heap);

        let ptr = allocator.alloc(TestObject::new(42));
        assert!(ptr.is_some());

        let obj = unsafe { &*ptr.unwrap() };
        assert_eq!(obj.value, 42);
    }

    #[test]
    fn test_alloc_multiple() {
        let heap = GcHeap::new(GcConfig::default());
        let allocator = GcAllocator::new(&heap);

        let ptrs: Vec<_> = (0..100)
            .map(|i| allocator.alloc(TestObject::new(i)))
            .collect();

        // All allocations should succeed
        assert!(ptrs.iter().all(|p| p.is_some()));

        // Verify each object has correct value
        for (i, ptr) in ptrs.into_iter().enumerate() {
            let obj = unsafe { &*ptr.unwrap() };
            assert_eq!(obj.value, i as i64);
        }
    }

    #[test]
    fn test_alloc_different_types() {
        let heap = GcHeap::new(GcConfig::default());
        let allocator = GcAllocator::new(&heap);

        // Allocate different sized objects
        let small = allocator.alloc(42i64);
        let medium = allocator.alloc(TestObject::new(100));
        let large = allocator.alloc(NestedObject { values: vec![] });

        assert!(small.is_some());
        assert!(medium.is_some());
        assert!(large.is_some());
    }

    // -------------------------------------------------------------------------
    // alloc_value Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_alloc_value() {
        let heap = GcHeap::new(GcConfig::default());
        let allocator = GcAllocator::new(&heap);

        let val = allocator.alloc_value(TestObject::new(42));
        assert!(val.is_some());

        let val = val.unwrap();
        assert!(val.as_object_ptr().is_some());
    }

    #[test]
    fn test_alloc_value_roundtrip() {
        let heap = GcHeap::new(GcConfig::default());
        let allocator = GcAllocator::new(&heap);

        let val = allocator.alloc_value(TestObject::new(999)).unwrap();
        let ptr = val.as_object_ptr().unwrap() as *const TestObject;
        let obj = unsafe { &*ptr };

        assert_eq!(obj.value, 999);
    }

    // -------------------------------------------------------------------------
    // try_alloc Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_try_alloc_success() {
        let heap = GcHeap::new(GcConfig::default());
        let allocator = GcAllocator::new(&heap);

        let result = allocator.try_alloc(TestObject::new(42));
        assert!(result.is_ok());
    }

    #[test]
    fn test_try_alloc_result_map() {
        let heap = GcHeap::new(GcConfig::default());
        let allocator = GcAllocator::new(&heap);

        let result = allocator
            .try_alloc(TestObject::new(42))
            .map(|ptr| unsafe { (*ptr).value });

        assert_eq!(result.ok(), Some(42));
    }

    // -------------------------------------------------------------------------
    // alloc_tenured Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_alloc_tenured() {
        let heap = GcHeap::new(GcConfig::default());
        let allocator = GcAllocator::new(&heap);

        let ptr = allocator.alloc_tenured(TestObject::new(42));
        assert!(ptr.is_some());

        let obj = unsafe { &*ptr.unwrap() };
        assert_eq!(obj.value, 42);
    }

    #[test]
    fn test_alloc_tenured_is_in_old_gen() {
        let heap = GcHeap::new(GcConfig::default());
        let allocator = GcAllocator::new(&heap);

        let ptr = allocator.alloc_tenured(TestObject::new(42)).unwrap();
        // Tenured allocations go to old space
        assert!(heap.is_old(ptr as *const ()));
    }

    // -------------------------------------------------------------------------
    // alloc_raw Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_alloc_raw() {
        let heap = GcHeap::new(GcConfig::default());
        let allocator = GcAllocator::new(&heap);

        let ptr = allocator.alloc_raw(64);
        assert!(ptr.is_some());
    }

    #[test]
    fn test_alloc_raw_various_sizes() {
        let heap = GcHeap::new(GcConfig::default());
        let allocator = GcAllocator::new(&heap);

        for size in [8, 16, 32, 64, 128, 256, 512, 1024] {
            let ptr = allocator.alloc_raw(size);
            assert!(ptr.is_some(), "Failed to allocate {} bytes", size);
        }
    }

    // -------------------------------------------------------------------------
    // can_alloc Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_can_alloc_initially_true() {
        let heap = GcHeap::new(GcConfig::default());
        let allocator = GcAllocator::new(&heap);

        assert!(allocator.can_alloc(64));
    }

    // -------------------------------------------------------------------------
    // AllocResult Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_alloc_result_ok() {
        let result: AllocResult<i32> = AllocResult::Ok(42);
        assert!(result.is_ok());
        assert!(!result.needs_collection());
        assert_eq!(result.ok(), Some(42));
    }

    #[test]
    fn test_alloc_result_needs_collection() {
        let result: AllocResult<i32> = AllocResult::NeedsCollection;
        assert!(!result.is_ok());
        assert!(result.needs_collection());
        assert_eq!(result.ok(), None);
    }

    #[test]
    fn test_alloc_result_oom() {
        let result: AllocResult<i32> = AllocResult::OutOfMemory;
        assert!(!result.is_ok());
        assert!(!result.needs_collection());
        assert_eq!(result.ok(), None);
    }

    #[test]
    fn test_alloc_result_map_ok() {
        let result: AllocResult<i32> = AllocResult::Ok(42);
        let mapped = result.map(|x| x * 2);
        assert_eq!(mapped.ok(), Some(84));
    }

    #[test]
    fn test_alloc_result_map_error() {
        let result: AllocResult<i32> = AllocResult::NeedsCollection;
        let mapped = result.map(|x| x * 2);
        assert!(mapped.needs_collection());
    }

    #[test]
    fn test_alloc_result_unwrap() {
        let result: AllocResult<i32> = AllocResult::Ok(42);
        assert_eq!(result.unwrap(), 42);
    }

    #[test]
    #[should_panic(expected = "needs collection")]
    fn test_alloc_result_unwrap_panic_collection() {
        let result: AllocResult<i32> = AllocResult::NeedsCollection;
        let _ = result.unwrap();
    }

    #[test]
    #[should_panic(expected = "out of memory")]
    fn test_alloc_result_unwrap_panic_oom() {
        let result: AllocResult<i32> = AllocResult::OutOfMemory;
        let _ = result.unwrap();
    }

    // -------------------------------------------------------------------------
    // Memory Layout Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_allocation_alignment() {
        let heap = GcHeap::new(GcConfig::default());
        let allocator = GcAllocator::new(&heap);

        // All allocations should be 8-byte aligned
        for _ in 0..100 {
            let ptr = allocator.alloc(TestObject::new(0)).unwrap();
            assert_eq!(ptr as usize % 8, 0, "Pointer not 8-byte aligned");
        }
    }

    #[test]
    fn test_allocations_non_overlapping() {
        let heap = GcHeap::new(GcConfig::default());
        let allocator = GcAllocator::new(&heap);

        let mut ptrs: Vec<*mut TestObject> = Vec::new();

        for i in 0..50 {
            let ptr = allocator.alloc(TestObject::new(i)).unwrap();
            ptrs.push(ptr);
        }

        // Check no two allocations overlap
        let size = std::mem::size_of::<TestObject>();
        for i in 0..ptrs.len() {
            for j in (i + 1)..ptrs.len() {
                let start_i = ptrs[i] as usize;
                let end_i = start_i + size;
                let start_j = ptrs[j] as usize;
                let end_j = start_j + size;

                assert!(
                    end_i <= start_j || end_j <= start_i,
                    "Allocations {} and {} overlap",
                    i,
                    j
                );
            }
        }
    }

    // -------------------------------------------------------------------------
    // Stats Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_allocator_stats() {
        let heap = GcHeap::new(GcConfig::default());
        let allocator = GcAllocator::new(&heap);

        let initial = allocator
            .stats()
            .objects_allocated
            .load(std::sync::atomic::Ordering::Relaxed);

        allocator.alloc(TestObject::new(0));

        let after = allocator
            .stats()
            .objects_allocated
            .load(std::sync::atomic::Ordering::Relaxed);
        assert!(after > initial, "Object allocation count should increase");
    }

    // -------------------------------------------------------------------------
    // Edge Case Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_alloc_zero_sized_type() {
        let heap = GcHeap::new(GcConfig::default());
        let allocator = GcAllocator::new(&heap);

        // Zero-sized type
        #[derive(Debug)]
        struct Empty;

        unsafe impl Trace for Empty {
            fn trace(&self, _tracer: &mut dyn Tracer) {}
        }

        // Should still allocate minimum size
        let ptr = allocator.alloc(Empty);
        assert!(ptr.is_some());
    }

    #[test]
    fn test_alloc_with_values() {
        let heap = GcHeap::new(GcConfig::default());
        let allocator = GcAllocator::new(&heap);

        // Allocate object with Value references
        let obj = NestedObject {
            values: vec![
                Value::int(1).unwrap(),
                Value::int(2).unwrap(),
                Value::int(3).unwrap(),
            ],
        };

        let ptr = allocator.alloc(obj).unwrap();
        let allocated = unsafe { &*ptr };

        assert_eq!(allocated.values.len(), 3);
    }
}
