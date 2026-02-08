//! Write barriers for generational GC.
//!
//! Write barriers track old→young references to enable efficient
//! minor collection. Without barriers, we'd have to scan the entire
//! old generation to find references into the nursery.

mod card_table;
mod remembered_set;
pub mod satb_buffer;

#[cfg(test)]
mod satb_tests;

pub use card_table::CardTable;
pub use remembered_set::{RememberedEntry, RememberedSet};
pub use satb_buffer::{
    satb_capture_unconditional, satb_write_barrier, MarkingPhase, SatbBuffer, SatbMarkingState,
    SatbQueue,
};

use crate::heap::GcHeap;
use prism_core::Value;

/// Write barrier for pointer stores.
///
/// Call this after storing a reference into a heap object.
/// The barrier tracks old→young references in a card table.
///
/// # Arguments
///
/// * `heap` - The GC heap
/// * `holder` - Pointer to the object containing the field
/// * `new_value` - The value being stored
///
/// # Performance
///
/// This is called on every pointer store, so it must be fast.
/// The fast path is a single comparison + conditional store.
///
/// # Example
///
/// ```ignore
/// // When storing a reference:
/// obj.field = new_value;
/// write_barrier(heap, obj as *const (), new_value);
/// ```
#[inline(always)]
pub fn write_barrier(heap: &GcHeap, holder: *const (), new_value: Value) {
    // Only check object pointers
    if let Some(new_ptr) = new_value.as_object_ptr() {
        write_barrier_ptr(heap, holder, new_ptr);
    }
}

/// Write barrier for raw pointer stores.
#[inline(always)]
pub fn write_barrier_ptr(heap: &GcHeap, holder: *const (), new_ptr: *const ()) {
    // Fast path: if holder is young, no barrier needed
    if heap.is_young(holder) {
        return;
    }

    // Check if new_ptr points to young generation
    if heap.is_young(new_ptr) {
        // Old→Young reference: record in remembered set
        heap.remembered_set().insert(holder);
    }
}

/// Unconditional write barrier that always marks the card.
///
/// Used when we can't easily determine the value's generation.
#[inline(always)]
pub fn write_barrier_unconditional(holder: *const (), card_table: &CardTable) {
    card_table.mark(holder);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::GcConfig;

    #[test]
    fn test_write_barrier_no_panic() {
        let heap = GcHeap::new(GcConfig::default());

        // Should not panic with null pointers
        write_barrier(&heap, std::ptr::null(), Value::none());
        write_barrier(&heap, std::ptr::null(), Value::int(42).unwrap());
    }
}
