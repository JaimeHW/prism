//! GC-Managed Arrays with Inline Data Storage
//!
//! GcArray provides contiguous, GC-managed storage for elements.
//! Unlike Vec<T> which allocates data on the system heap, GcArray
//! stores data inline with the object header, enabling:
//!
//! - Better cache locality (header + data are contiguous)
//! - GC compaction support (the GC can move the entire array)
//! - Proper tracing of contained references
//!
//! # Layout
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────┐
//! │  GcHeader  │  GcArray<T>  │  data[0]  data[1]  ...  data[N] │
//! │  (8 bytes) │   (header)   │        (inline elements)        │
//! └─────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Usage
//!
//! ```ignore
//! // Create via GC heap allocation
//! let array: GcRef<GcArray<Value>> = heap.alloc_array(capacity);
//!
//! // Access elements
//! array.push(value);
//! let v = array.get(0);
//! ```

use std::alloc::Layout;
use std::marker::PhantomData;
use std::ptr::NonNull;

// =============================================================================
// GcArray Header
// =============================================================================

/// A GC-managed array with inline element storage.
///
/// The elements are stored directly after the GcArrayHeader in memory,
/// providing contiguous allocation and excellent cache locality.
#[repr(C)]
pub struct GcArray<T> {
    /// Capacity of the array in elements.
    capacity: u32,
    /// Current length (number of used elements).
    len: u32,
    /// Marker for element type.
    _marker: PhantomData<T>,
    // Elements follow directly after this header in memory.
    // We use a flexible array member pattern.
}

impl<T> GcArray<T> {
    /// Calculate the layout for a GcArray with the given capacity.
    ///
    /// Returns (total_size, data_offset) where:
    /// - total_size: Total bytes needed for header + elements
    /// - data_offset: Offset from GcArray start to first element
    pub fn layout_for_capacity(capacity: usize) -> (Layout, usize) {
        let header_layout = Layout::new::<GcArray<T>>();
        let element_layout = Layout::new::<T>();

        // Calculate aligned offset for elements
        let data_offset = header_layout.size();
        let aligned_offset =
            (data_offset + element_layout.align() - 1) & !(element_layout.align() - 1);

        // Total size: header + padding + elements
        let elements_size = capacity.saturating_mul(element_layout.size());
        let total_size = aligned_offset.saturating_add(elements_size);

        // Create the combined layout with proper alignment
        let layout = Layout::from_size_align(
            total_size,
            header_layout.align().max(element_layout.align()),
        )
        .expect("Invalid GcArray layout");

        (layout, aligned_offset)
    }

    /// Get the total allocation size in bytes for a given capacity.
    pub fn allocation_size(capacity: usize) -> usize {
        Self::layout_for_capacity(capacity).0.size()
    }

    /// Initialize a GcArray header in the given memory location.
    ///
    /// # Safety
    ///
    /// - `ptr` must point to a valid memory region for a GcArray with the given capacity.
    /// - The memory must be properly aligned.
    /// - The caller must ensure the memory is zeroed or properly initialized.
    pub unsafe fn init_at(ptr: NonNull<Self>, capacity: usize) {
        let array = ptr.as_ptr();
        (*array).capacity = capacity.min(u32::MAX as usize) as u32;
        (*array).len = 0;
    }

    /// Get the capacity of this array.
    #[inline]
    pub fn capacity(&self) -> usize {
        self.capacity as usize
    }

    /// Get the current length (number of elements).
    #[inline]
    pub fn len(&self) -> usize {
        self.len as usize
    }

    /// Check if the array is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Check if the array is full.
    #[inline]
    pub fn is_full(&self) -> bool {
        self.len as usize >= self.capacity as usize
    }

    /// Get a pointer to the element data.
    ///
    /// # Safety
    ///
    /// The returned pointer is valid only as long as the GcArray is alive.
    #[inline]
    pub fn data_ptr(&self) -> *const T {
        let (_, data_offset) = Self::layout_for_capacity(0);
        unsafe { (self as *const Self as *const u8).add(data_offset) as *const T }
    }

    /// Get a mutable pointer to the element data.
    ///
    /// # Safety
    ///
    /// The returned pointer is valid only as long as the GcArray is alive.
    #[inline]
    pub fn data_ptr_mut(&mut self) -> *mut T {
        let (_, data_offset) = Self::layout_for_capacity(0);
        unsafe { (self as *mut Self as *mut u8).add(data_offset) as *mut T }
    }

    /// Get an element by index.
    ///
    /// Returns None if the index is out of bounds.
    #[inline]
    pub fn get(&self, index: usize) -> Option<&T> {
        if index < self.len as usize {
            unsafe { Some(&*self.data_ptr().add(index)) }
        } else {
            None
        }
    }

    /// Get a mutable reference to an element by index.
    ///
    /// Returns None if the index is out of bounds.
    #[inline]
    pub fn get_mut(&mut self, index: usize) -> Option<&mut T> {
        if index < self.len as usize {
            unsafe { Some(&mut *self.data_ptr_mut().add(index)) }
        } else {
            None
        }
    }

    /// Get an element by index without bounds checking.
    ///
    /// # Safety
    ///
    /// The caller must ensure that `index < self.len()`.
    #[inline]
    pub unsafe fn get_unchecked(&self, index: usize) -> &T {
        &*self.data_ptr().add(index)
    }

    /// Get a mutable reference to an element without bounds checking.
    ///
    /// # Safety
    ///
    /// The caller must ensure that `index < self.len()`.
    #[inline]
    pub unsafe fn get_unchecked_mut(&mut self, index: usize) -> &mut T {
        &mut *self.data_ptr_mut().add(index)
    }

    /// Set an element at the given index.
    ///
    /// # Panics
    ///
    /// Panics if index >= len.
    #[inline]
    pub fn set(&mut self, index: usize, value: T) {
        assert!(index < self.len as usize, "Index out of bounds");
        unsafe {
            std::ptr::write(self.data_ptr_mut().add(index), value);
        }
    }

    /// Push an element to the end of the array.
    ///
    /// Returns true if successful, false if the array is full.
    #[inline]
    pub fn push(&mut self, value: T) -> bool {
        let len = self.len as usize;
        if len >= self.capacity as usize {
            return false; // Full
        }

        unsafe {
            std::ptr::write(self.data_ptr_mut().add(len), value);
        }
        self.len += 1;
        true
    }

    /// Pop an element from the end of the array.
    ///
    /// Returns None if the array is empty.
    pub fn pop(&mut self) -> Option<T> {
        if self.len == 0 {
            return None;
        }

        self.len -= 1;
        unsafe { Some(std::ptr::read(self.data_ptr().add(self.len as usize))) }
    }

    /// Clear all elements from the array.
    pub fn clear(&mut self) {
        // Drop all elements
        for i in 0..self.len as usize {
            unsafe {
                std::ptr::drop_in_place(self.data_ptr_mut().add(i));
            }
        }
        self.len = 0;
    }

    /// Get an iterator over the elements.
    pub fn iter(&self) -> impl Iterator<Item = &T> {
        (0..self.len as usize).map(move |i| unsafe { &*self.data_ptr().add(i) })
    }

    /// Get a mutable iterator over the elements.
    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut T> {
        let len = self.len as usize;
        let data = self.data_ptr_mut();
        (0..len).map(move |i| unsafe { &mut *data.add(i) })
    }

    /// Get a slice over all elements.
    pub fn as_slice(&self) -> &[T] {
        unsafe { std::slice::from_raw_parts(self.data_ptr(), self.len as usize) }
    }

    /// Get a mutable slice over all elements.
    pub fn as_slice_mut(&mut self) -> &mut [T] {
        unsafe { std::slice::from_raw_parts_mut(self.data_ptr_mut(), self.len as usize) }
    }
}

impl<T: Clone> GcArray<T> {
    /// Fill the array with a cloned value up to capacity.
    pub fn fill(&mut self, value: T) {
        for i in self.len as usize..self.capacity as usize {
            unsafe {
                std::ptr::write(self.data_ptr_mut().add(i), value.clone());
            }
        }
        self.len = self.capacity;
    }

    /// Extend from a slice, returning number of elements copied.
    pub fn extend_from_slice(&mut self, slice: &[T]) -> usize {
        let available = self.capacity as usize - self.len as usize;
        let to_copy = slice.len().min(available);

        for (i, item) in slice.iter().take(to_copy).enumerate() {
            unsafe {
                std::ptr::write(self.data_ptr_mut().add(self.len as usize + i), item.clone());
            }
        }
        self.len += to_copy as u32;
        to_copy
    }
}

impl<T: Copy> GcArray<T> {
    /// Copy elements from another slice (faster for Copy types).
    pub fn copy_from_slice(&mut self, slice: &[T]) -> usize {
        let available = self.capacity as usize - self.len as usize;
        let to_copy = slice.len().min(available);

        if to_copy > 0 {
            unsafe {
                std::ptr::copy_nonoverlapping(
                    slice.as_ptr(),
                    self.data_ptr_mut().add(self.len as usize),
                    to_copy,
                );
            }
            self.len += to_copy as u32;
        }
        to_copy
    }
}

// =============================================================================
// Drop Implementation
// =============================================================================

impl<T> Drop for GcArray<T> {
    fn drop(&mut self) {
        // Drop all elements
        for i in 0..self.len as usize {
            unsafe {
                std::ptr::drop_in_place(self.data_ptr_mut().add(i));
            }
        }
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // Helper to create a GcArray for testing
    fn alloc_test_array<T>(capacity: usize) -> (*mut GcArray<T>, Layout) {
        let (layout, _) = GcArray::<T>::layout_for_capacity(capacity);
        let ptr = unsafe { std::alloc::alloc_zeroed(layout) as *mut GcArray<T> };
        unsafe {
            GcArray::init_at(NonNull::new_unchecked(ptr), capacity);
        }
        (ptr, layout)
    }

    fn free_test_array<T>(ptr: *mut GcArray<T>, layout: Layout) {
        unsafe {
            // Drop the array properly
            std::ptr::drop_in_place(ptr);
            std::alloc::dealloc(ptr as *mut u8, layout);
        }
    }

    #[test]
    fn test_layout_calculation() {
        // u64 array
        let (layout, offset) = GcArray::<u64>::layout_for_capacity(10);
        assert!(layout.size() >= std::mem::size_of::<GcArray<u64>>() + 10 * 8);
        assert!(offset >= std::mem::size_of::<GcArray<u64>>());
        assert_eq!(offset % 8, 0); // Properly aligned for u64
    }

    #[test]
    fn test_create_and_destroy() {
        let (ptr, layout) = alloc_test_array::<u64>(10);
        assert!(!ptr.is_null());

        unsafe {
            assert_eq!((*ptr).capacity(), 10);
            assert_eq!((*ptr).len(), 0);
            assert!((*ptr).is_empty());
        }

        free_test_array(ptr, layout);
    }

    #[test]
    fn test_push_pop() {
        let (ptr, layout) = alloc_test_array::<u64>(5);

        unsafe {
            let array = &mut *ptr;

            assert!(array.push(10));
            assert!(array.push(20));
            assert!(array.push(30));
            assert_eq!(array.len(), 3);

            assert_eq!(array.pop(), Some(30));
            assert_eq!(array.pop(), Some(20));
            assert_eq!(array.len(), 1);
        }

        free_test_array(ptr, layout);
    }

    #[test]
    fn test_capacity_limit() {
        let (ptr, layout) = alloc_test_array::<u64>(3);

        unsafe {
            let array = &mut *ptr;

            assert!(array.push(1));
            assert!(array.push(2));
            assert!(array.push(3));
            assert!(array.is_full());
            assert!(!array.push(4)); // Should fail
        }

        free_test_array(ptr, layout);
    }

    #[test]
    fn test_get_set() {
        let (ptr, layout) = alloc_test_array::<u64>(5);

        unsafe {
            let array = &mut *ptr;

            array.push(100);
            array.push(200);
            array.push(300);

            assert_eq!(*array.get(0).unwrap(), 100);
            assert_eq!(*array.get(1).unwrap(), 200);
            assert_eq!(*array.get(2).unwrap(), 300);
            assert!(array.get(3).is_none());

            array.set(1, 999);
            assert_eq!(*array.get(1).unwrap(), 999);
        }

        free_test_array(ptr, layout);
    }

    #[test]
    fn test_iteration() {
        let (ptr, layout) = alloc_test_array::<u64>(5);

        unsafe {
            let array = &mut *ptr;

            array.push(1);
            array.push(2);
            array.push(3);

            let sum: u64 = array.iter().sum();
            assert_eq!(sum, 6);
        }

        free_test_array(ptr, layout);
    }

    #[test]
    fn test_slice_access() {
        let (ptr, layout) = alloc_test_array::<u64>(5);

        unsafe {
            let array = &mut *ptr;

            array.push(10);
            array.push(20);
            array.push(30);

            let slice = array.as_slice();
            assert_eq!(slice, &[10, 20, 30]);

            array.as_slice_mut()[1] = 999;
            assert_eq!(array.as_slice(), &[10, 999, 30]);
        }

        free_test_array(ptr, layout);
    }

    #[test]
    fn test_clear() {
        let (ptr, layout) = alloc_test_array::<u64>(5);

        unsafe {
            let array = &mut *ptr;

            array.push(1);
            array.push(2);
            array.push(3);
            assert_eq!(array.len(), 3);

            array.clear();
            assert_eq!(array.len(), 0);
            assert!(array.is_empty());
        }

        free_test_array(ptr, layout);
    }

    #[test]
    fn test_copy_from_slice() {
        let (ptr, layout) = alloc_test_array::<u64>(10);

        unsafe {
            let array = &mut *ptr;

            let source = [1u64, 2, 3, 4, 5];
            let copied = array.copy_from_slice(&source);
            assert_eq!(copied, 5);
            assert_eq!(array.as_slice(), &[1, 2, 3, 4, 5]);
        }

        free_test_array(ptr, layout);
    }

    #[test]
    fn test_extend_from_slice() {
        let (ptr, layout) = alloc_test_array::<String>(5);

        unsafe {
            let array = &mut *ptr;

            let source = ["hello".to_string(), "world".to_string()];
            let extended = array.extend_from_slice(&source);
            assert_eq!(extended, 2);
            assert_eq!(array.len(), 2);
            assert_eq!(array.get(0).unwrap(), "hello");
            assert_eq!(array.get(1).unwrap(), "world");
        }

        free_test_array(ptr, layout);
    }

    #[test]
    fn test_string_drop() {
        // Ensure strings are properly dropped to avoid memory leaks
        let (ptr, layout) = alloc_test_array::<String>(3);

        unsafe {
            let array = &mut *ptr;

            array.push("one".to_string());
            array.push("two".to_string());
            array.push("three".to_string());
        }

        // This should properly drop all strings
        free_test_array(ptr, layout);
    }

    #[test]
    fn test_allocation_size() {
        // Check that allocation size is reasonable
        let size = GcArray::<u64>::allocation_size(100);
        assert!(size >= 100 * 8); // At least element storage
        assert!(size < 100 * 8 + 1024); // Not too much overhead
    }
}
