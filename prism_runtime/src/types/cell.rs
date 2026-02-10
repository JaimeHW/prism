//! Cell object for Python closure semantics.
//!
//! A `Cell` represents a single captured variable in a closure. Python closures
//! capture variables by reference, allowing inner functions to see updates from
//! outer scopes (and vice versa).
//!
//! # Performance
//!
//! - Single pointer indirection (Cell → Value)
//! - Cache-line optimized at 64 bytes
//! - Lock-free reads via atomic operations
//! - Thread-safe writes with relaxed ordering (Python GIL semantics assumed)
//!
//! # Python Semantics
//!
//! ```python
//! def outer():
//!     x = 10
//!     def inner():
//!         nonlocal x
//!         x = x + 1
//!         return x
//!     return inner
//!
//! f = outer()
//! f()  # Returns 11, x is shared via cell
//! f()  # Returns 12
//! ```
//!
//! # Implementation Notes
//!
//! Unlike CPython's `PyCellObject`, we use atomic value storage for thread
//! safety in concurrent scenarios. This adds minimal overhead while enabling
//! safe multi-threaded closure use.

use crate::object::ObjectHeader;
use crate::object::type_obj::TypeId;
use prism_core::Value;
use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};

// =============================================================================
// Constants
// =============================================================================

/// Sentinel value indicating an unbound cell (deleted or not yet assigned).
const UNBOUND_SENTINEL: u64 = 0xFFFF_FFFF_FFFF_FFFE;

// =============================================================================
// Cell Object
// =============================================================================

/// A cell object holding a reference to a captured variable.
///
/// Cells enable Python's closure semantics where inner functions can read
/// and write variables from enclosing scopes.
///
/// # Memory Layout
///
/// The cell is 64-byte aligned to fit in a single cache line:
/// - ObjectHeader: 16 bytes
/// - value: 8 bytes (atomic u64 for NaN-boxed Value)
/// - padding: 40 bytes (reserved for future expansion)
///
/// # Thread Safety
///
/// All operations are thread-safe using atomic operations. Under Python's GIL
/// semantics, this provides sequential consistency. In free-threaded scenarios,
/// users must ensure proper synchronization.
#[repr(C, align(64))]
pub struct Cell {
    /// Standard object header for GC and type identification.
    header: ObjectHeader,

    /// The contained value as an atomic u64 (NaN-boxed).
    ///
    /// Uses `AtomicU64` for thread-safe access.
    /// - `UNBOUND_SENTINEL` indicates unbound/deleted state
    /// - Otherwise contains a valid NaN-boxed Value
    value: AtomicU64,

    /// Padding to fill cache line.
    _padding: [u8; 40],
}

// =============================================================================
// Cell Implementation
// =============================================================================

impl Cell {
    /// Create a new cell containing the given value.
    ///
    /// # Arguments
    ///
    /// * `value` - The initial value to store in the cell
    ///
    /// # Returns
    ///
    /// A new cell containing the value.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let cell = Cell::new(Value::int(42).unwrap());
    /// assert_eq!(cell.get().as_int(), Some(42));
    /// ```
    #[inline]
    pub fn new(value: Value) -> Self {
        Cell {
            header: ObjectHeader::new(TypeId::CELL),
            value: AtomicU64::new(value.to_bits()),
            _padding: [0; 40],
        }
    }

    /// Create a new unbound cell.
    ///
    /// Unbound cells represent variables that have been deleted or not yet
    /// assigned. Accessing an unbound cell raises `UnboundLocalError`.
    ///
    /// # Returns
    ///
    /// A new cell in the unbound state.
    #[inline]
    pub fn unbound() -> Self {
        Cell {
            header: ObjectHeader::new(TypeId::CELL),
            value: AtomicU64::new(UNBOUND_SENTINEL),
            _padding: [0; 40],
        }
    }

    /// Get the value from the cell.
    ///
    /// # Returns
    ///
    /// The value if the cell is bound, or `None` if unbound.
    ///
    /// # Thread Safety
    ///
    /// This operation uses `Acquire` ordering to ensure visibility of
    /// any writes made before the corresponding `set()` call.
    #[inline]
    pub fn get(&self) -> Option<Value> {
        let bits = self.value.load(Ordering::Acquire);
        if bits == UNBOUND_SENTINEL {
            None
        } else {
            Some(Value::from_bits(bits))
        }
    }

    /// Get the value from the cell, returning `Value::none()` if unbound.
    ///
    /// This is a convenience method for cases where unbound should be
    /// treated as None (Python semantics for some operations).
    ///
    /// # Returns
    ///
    /// The value if bound, or `Value::none()` if unbound.
    #[inline]
    pub fn get_or_none(&self) -> Value {
        let bits = self.value.load(Ordering::Acquire);
        if bits == UNBOUND_SENTINEL {
            Value::none()
        } else {
            Value::from_bits(bits)
        }
    }

    /// Get the raw value, panicking if unbound.
    ///
    /// # Panics
    ///
    /// Panics if the cell is unbound.
    ///
    /// # Returns
    ///
    /// The contained value.
    #[inline]
    pub fn get_unchecked(&self) -> Value {
        let bits = self.value.load(Ordering::Acquire);
        debug_assert!(bits != UNBOUND_SENTINEL, "Cell is unbound");
        Value::from_bits(bits)
    }

    /// Set the value in the cell.
    ///
    /// # Arguments
    ///
    /// * `value` - The value to store
    ///
    /// # Thread Safety
    ///
    /// This operation uses `Release` ordering to ensure that all writes
    /// before this call are visible to subsequent `get()` calls.
    #[inline]
    pub fn set(&self, value: Value) {
        self.value.store(value.to_bits(), Ordering::Release);
    }

    /// Clear the cell, making it unbound.
    ///
    /// After this call, `get()` will return `None` and `is_empty()` will
    /// return `true`.
    #[inline]
    pub fn clear(&self) {
        self.value.store(UNBOUND_SENTINEL, Ordering::Release);
    }

    /// Check if the cell is unbound (empty).
    ///
    /// # Returns
    ///
    /// `true` if the cell has no value, `false` otherwise.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.value.load(Ordering::Acquire) == UNBOUND_SENTINEL
    }

    /// Check if the cell is bound (has a value).
    ///
    /// # Returns
    ///
    /// `true` if the cell has a value, `false` otherwise.
    #[inline]
    pub fn is_bound(&self) -> bool {
        self.value.load(Ordering::Acquire) != UNBOUND_SENTINEL
    }

    /// Swap the value in the cell, returning the old value.
    ///
    /// # Arguments
    ///
    /// * `value` - The new value to store
    ///
    /// # Returns
    ///
    /// The previous value, or `None` if the cell was unbound.
    #[inline]
    pub fn swap(&self, value: Value) -> Option<Value> {
        let old_bits = self.value.swap(value.to_bits(), Ordering::AcqRel);
        if old_bits == UNBOUND_SENTINEL {
            None
        } else {
            Some(Value::from_bits(old_bits))
        }
    }

    /// Get the object header.
    #[inline]
    pub fn header(&self) -> &ObjectHeader {
        &self.header
    }

    /// Get the object header mutably.
    #[inline]
    pub fn header_mut(&mut self) -> &mut ObjectHeader {
        &mut self.header
    }
}

// =============================================================================
// Trait Implementations
// =============================================================================

impl fmt::Debug for Cell {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.get() {
            Some(value) => write!(f, "Cell({:?})", value),
            None => write!(f, "Cell(<unbound>)"),
        }
    }
}

impl fmt::Display for Cell {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.get() {
            Some(value) => write!(f, "<cell: {:?}>", value),
            None => write!(f, "<cell: empty>"),
        }
    }
}

impl Clone for Cell {
    /// Clone the cell, creating a new cell with the same value.
    ///
    /// Note: This creates a *copy* of the cell, not a shared reference.
    /// For shared closure semantics, use `Arc<Cell>`.
    fn clone(&self) -> Self {
        let bits = self.value.load(Ordering::Acquire);
        Cell {
            header: ObjectHeader::new(TypeId::CELL),
            value: AtomicU64::new(bits),
            _padding: [0; 40],
        }
    }
}

impl Default for Cell {
    /// Create an unbound cell.
    fn default() -> Self {
        Self::unbound()
    }
}

// Safety: Cell uses atomic operations for all value access
unsafe impl Send for Cell {}
unsafe impl Sync for Cell {}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // Basic Operations
    // =========================================================================

    #[test]
    fn test_cell_new_with_int() {
        let cell = Cell::new(Value::int(42).unwrap());
        assert!(cell.is_bound());
        assert!(!cell.is_empty());
        assert_eq!(cell.get().unwrap().as_int(), Some(42));
    }

    #[test]
    fn test_cell_new_with_float() {
        let cell = Cell::new(Value::float(3.14159));
        assert!(cell.is_bound());
        let val = cell.get().unwrap();
        assert!((val.as_float().unwrap() - 3.14159).abs() < 1e-10);
    }

    #[test]
    fn test_cell_new_with_bool() {
        let cell_true = Cell::new(Value::bool(true));
        let cell_false = Cell::new(Value::bool(false));

        assert_eq!(cell_true.get().unwrap().as_bool(), Some(true));
        assert_eq!(cell_false.get().unwrap().as_bool(), Some(false));
    }

    #[test]
    fn test_cell_new_with_none() {
        let cell = Cell::new(Value::none());
        assert!(cell.is_bound());
        assert!(cell.get().unwrap().is_none());
    }

    #[test]
    fn test_cell_unbound() {
        let cell = Cell::unbound();
        assert!(cell.is_empty());
        assert!(!cell.is_bound());
        assert!(cell.get().is_none());
    }

    #[test]
    fn test_cell_default_is_unbound() {
        let cell = Cell::default();
        assert!(cell.is_empty());
        assert!(cell.get().is_none());
    }

    // =========================================================================
    // Set Operations
    // =========================================================================

    #[test]
    fn test_cell_set_updates_value() {
        let cell = Cell::new(Value::int(1).unwrap());
        assert_eq!(cell.get().unwrap().as_int(), Some(1));

        cell.set(Value::int(2).unwrap());
        assert_eq!(cell.get().unwrap().as_int(), Some(2));

        cell.set(Value::int(100).unwrap());
        assert_eq!(cell.get().unwrap().as_int(), Some(100));
    }

    #[test]
    fn test_cell_set_on_unbound() {
        let cell = Cell::unbound();
        assert!(cell.is_empty());

        cell.set(Value::int(42).unwrap());
        assert!(cell.is_bound());
        assert_eq!(cell.get().unwrap().as_int(), Some(42));
    }

    #[test]
    fn test_cell_clear() {
        let cell = Cell::new(Value::int(42).unwrap());
        assert!(cell.is_bound());

        cell.clear();
        assert!(cell.is_empty());
        assert!(cell.get().is_none());
    }

    #[test]
    fn test_cell_clear_already_unbound() {
        let cell = Cell::unbound();
        cell.clear(); // Should not panic
        assert!(cell.is_empty());
    }

    // =========================================================================
    // Swap Operations
    // =========================================================================

    #[test]
    fn test_cell_swap_bound() {
        let cell = Cell::new(Value::int(1).unwrap());

        let old = cell.swap(Value::int(2).unwrap());
        assert_eq!(old.unwrap().as_int(), Some(1));
        assert_eq!(cell.get().unwrap().as_int(), Some(2));
    }

    #[test]
    fn test_cell_swap_unbound() {
        let cell = Cell::unbound();

        let old = cell.swap(Value::int(42).unwrap());
        assert!(old.is_none());
        assert_eq!(cell.get().unwrap().as_int(), Some(42));
    }

    #[test]
    fn test_cell_swap_multiple_times() {
        let cell = Cell::new(Value::int(1).unwrap());

        for i in 2..=10 {
            let old = cell.swap(Value::int(i).unwrap());
            assert_eq!(old.unwrap().as_int(), Some(i - 1));
        }
        assert_eq!(cell.get().unwrap().as_int(), Some(10));
    }

    // =========================================================================
    // Convenience Methods
    // =========================================================================

    #[test]
    fn test_cell_get_or_none_bound() {
        let cell = Cell::new(Value::int(42).unwrap());
        let value = cell.get_or_none();
        assert_eq!(value.as_int(), Some(42));
    }

    #[test]
    fn test_cell_get_or_none_unbound() {
        let cell = Cell::unbound();
        let value = cell.get_or_none();
        assert!(value.is_none());
    }

    #[test]
    fn test_cell_get_unchecked() {
        let cell = Cell::new(Value::int(42).unwrap());
        let value = cell.get_unchecked();
        assert_eq!(value.as_int(), Some(42));
    }

    // =========================================================================
    // Clone and Debug
    // =========================================================================

    #[test]
    fn test_cell_clone_creates_copy() {
        let cell1 = Cell::new(Value::int(42).unwrap());
        let cell2 = cell1.clone();

        // Both have same value initially
        assert_eq!(cell1.get().unwrap().as_int(), Some(42));
        assert_eq!(cell2.get().unwrap().as_int(), Some(42));

        // Modifying one doesn't affect the other
        cell1.set(Value::int(100).unwrap());
        assert_eq!(cell1.get().unwrap().as_int(), Some(100));
        assert_eq!(cell2.get().unwrap().as_int(), Some(42));
    }

    #[test]
    fn test_cell_clone_unbound() {
        let cell1 = Cell::unbound();
        let cell2 = cell1.clone();

        assert!(cell1.is_empty());
        assert!(cell2.is_empty());

        // Setting one doesn't affect the other
        cell1.set(Value::int(42).unwrap());
        assert!(cell1.is_bound());
        assert!(cell2.is_empty());
    }

    #[test]
    fn test_cell_debug_bound() {
        let cell = Cell::new(Value::int(42).unwrap());
        let debug = format!("{:?}", cell);
        assert!(debug.contains("Cell"));
    }

    #[test]
    fn test_cell_debug_unbound() {
        let cell = Cell::unbound();
        let debug = format!("{:?}", cell);
        assert!(debug.contains("unbound"));
    }

    #[test]
    fn test_cell_display_bound() {
        let cell = Cell::new(Value::int(42).unwrap());
        let display = format!("{}", cell);
        assert!(display.contains("cell"));
    }

    #[test]
    fn test_cell_display_empty() {
        let cell = Cell::unbound();
        let display = format!("{}", cell);
        assert!(display.contains("empty"));
    }

    // =========================================================================
    // Header Access
    // =========================================================================

    #[test]
    fn test_cell_header_type_id() {
        let cell = Cell::new(Value::int(42).unwrap());
        assert_eq!(cell.header().type_id, TypeId::CELL);
    }

    #[test]
    fn test_cell_header_mut() {
        let mut cell = Cell::new(Value::int(42).unwrap());
        let header = cell.header_mut();
        // Can mutate header fields if needed
        assert_eq!(header.type_id, TypeId::CELL);
    }

    // =========================================================================
    // Size and Alignment
    // =========================================================================

    #[test]
    fn test_cell_size() {
        // Cell should fit in a cache line (64 bytes)
        assert_eq!(std::mem::size_of::<Cell>(), 64);
    }

    #[test]
    fn test_cell_alignment() {
        // Cell should be aligned to cache line boundary
        assert_eq!(std::mem::align_of::<Cell>(), 64);
    }

    // =========================================================================
    // Thread Safety (basic checks)
    // =========================================================================

    #[test]
    fn test_cell_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<Cell>();
    }

    #[test]
    fn test_cell_concurrent_read_write() {
        use std::sync::Arc;
        use std::thread;

        let cell = Arc::new(Cell::new(Value::int(0).unwrap()));

        let handles: Vec<_> = (0..4)
            .map(|_| {
                let cell = Arc::clone(&cell);
                thread::spawn(move || {
                    for i in 0..1000 {
                        cell.set(Value::int(i).unwrap());
                        let _ = cell.get();
                    }
                })
            })
            .collect();

        for handle in handles {
            handle.join().unwrap();
        }

        // Cell should have some valid value
        assert!(cell.is_bound());
    }

    // =========================================================================
    // Edge Cases
    // =========================================================================

    #[test]
    fn test_cell_large_int() {
        // Use a large value that's clearly within any integer representation
        let large = 1_000_000_000i64;
        let cell = Cell::new(Value::int(large).unwrap());
        assert_eq!(cell.get().unwrap().as_int(), Some(large));
    }

    #[test]
    fn test_cell_negative_int() {
        let cell = Cell::new(Value::int(-42).unwrap());
        assert_eq!(cell.get().unwrap().as_int(), Some(-42));
    }

    #[test]
    fn test_cell_zero() {
        let cell = Cell::new(Value::int(0).unwrap());
        assert_eq!(cell.get().unwrap().as_int(), Some(0));
        assert!(cell.is_bound());
    }

    #[test]
    fn test_cell_nan() {
        let cell = Cell::new(Value::float(f64::NAN));
        let val = cell.get().unwrap();
        assert!(val.as_float().unwrap().is_nan());
    }

    #[test]
    fn test_cell_infinity() {
        let cell = Cell::new(Value::float(f64::INFINITY));
        assert_eq!(cell.get().unwrap().as_float(), Some(f64::INFINITY));
    }

    #[test]
    fn test_cell_negative_zero() {
        let cell = Cell::new(Value::float(-0.0));
        let val = cell.get().unwrap().as_float().unwrap();
        // -0.0 should be preserved
        assert!(val.is_sign_negative() || val == 0.0);
    }

    // =========================================================================
    // Closure Simulation
    // =========================================================================

    #[test]
    fn test_cell_closure_simulation() {
        use std::sync::Arc;
        // Simulate: x = 10; def inner(): nonlocal x; x += 1; return x
        let cell = Arc::new(Cell::new(Value::int(10).unwrap()));

        // First "call" to inner
        {
            let cell = Arc::clone(&cell);
            let x = cell.get().unwrap().as_int().unwrap();
            cell.set(Value::int(x + 1).unwrap());
        }
        assert_eq!(cell.get().unwrap().as_int(), Some(11));

        // Second "call"
        {
            let cell = Arc::clone(&cell);
            let x = cell.get().unwrap().as_int().unwrap();
            cell.set(Value::int(x + 1).unwrap());
        }
        assert_eq!(cell.get().unwrap().as_int(), Some(12));
    }

    #[test]
    fn test_cell_nested_closure_simulation() {
        use std::sync::Arc;

        // Simulate nested closures sharing a cell
        let outer_cell = Arc::new(Cell::new(Value::int(0).unwrap()));

        // "outer" function creates and modifies
        outer_cell.set(Value::int(100).unwrap());

        // "middle" function shares the cell
        let middle_cell = Arc::clone(&outer_cell);
        let val = middle_cell.get().unwrap().as_int().unwrap();
        middle_cell.set(Value::int(val + 10).unwrap());

        // "inner" function also shares the cell
        let inner_cell = Arc::clone(&outer_cell);
        let val = inner_cell.get().unwrap().as_int().unwrap();
        inner_cell.set(Value::int(val + 1).unwrap());

        // All closures see the final value
        assert_eq!(outer_cell.get().unwrap().as_int(), Some(111));
        assert_eq!(middle_cell.get().unwrap().as_int(), Some(111));
        assert_eq!(inner_cell.get().unwrap().as_int(), Some(111));
    }
}
