//! Cached property descriptor implementation.
//!
//! `cached_property(func)` is a descriptor that converts a method into a
//! property whose value is computed once and then cached as an instance
//! attribute.
//!
//! # Performance
//!
//! | Operation | Time |
//! |-----------|------|
//! | First access | O(cost of func) |
//! | Subsequent access | O(1) |
//! | Delete/invalidate | O(1) |
//!
//! # Architecture
//!
//! The `CachedProperty` stores:
//! - The getter function (as a Value)
//! - The attribute name (for storing the cached result in `__dict__`)
//! - A local cache for cases where instance dict isn't available
//!
//! In full Python integration, the cached value would be stored in the
//! instance's `__dict__`, making subsequent accesses bypass the descriptor
//! entirely. Here we provide the core caching logic.

use prism_core::Value;

// =============================================================================
// CachedProperty
// =============================================================================

/// A cached property descriptor.
///
/// On first access, calls the getter function and caches the result.
/// Subsequent accesses return the cached value without calling the getter.
///
/// # Thread Safety
///
/// This implementation is NOT thread-safe (matching Python's semantics).
/// The VM should ensure single-threaded access to descriptors.
#[derive(Debug, Clone)]
pub struct CachedProperty {
    /// The getter function.
    func: Value,
    /// The attribute name (for __dict__ storage).
    attr_name: Option<String>,
    /// Local cache (used when instance dict isn't available).
    cached_value: Option<Value>,
    /// The docstring, if any.
    doc: Option<Value>,
}

impl CachedProperty {
    /// Create a new cached property with a getter function.
    #[inline]
    pub fn new(func: Value) -> Self {
        Self {
            func,
            attr_name: None,
            cached_value: None,
            doc: None,
        }
    }

    /// Create a cached property with a name.
    pub fn with_name(func: Value, name: String) -> Self {
        Self {
            func,
            attr_name: Some(name),
            cached_value: None,
            doc: None,
        }
    }

    /// Create a cached property with a name and docstring.
    pub fn with_doc(func: Value, name: String, doc: Value) -> Self {
        Self {
            func,
            attr_name: Some(name),
            cached_value: None,
            doc: Some(doc),
        }
    }

    // =========================================================================
    // Descriptor Protocol
    // =========================================================================

    /// Get the cached value, or compute and cache it.
    ///
    /// The `compute` closure is called with `&self.func` and should
    /// invoke the getter function on the instance, returning the result.
    ///
    /// Returns the cached value (either from cache or newly computed).
    pub fn get_or_compute<F>(&mut self, compute: F) -> Value
    where
        F: FnOnce(&Value) -> Value,
    {
        if let Some(ref cached) = self.cached_value {
            return cached.clone();
        }

        let value = compute(&self.func);
        self.cached_value = Some(value.clone());
        value
    }

    /// Get the cached value without computing.
    ///
    /// Returns `None` if no value has been cached yet.
    #[inline]
    pub fn get_cached(&self) -> Option<&Value> {
        self.cached_value.as_ref()
    }

    /// Check if a value has been cached.
    #[inline]
    pub fn is_cached(&self) -> bool {
        self.cached_value.is_some()
    }

    /// Delete (invalidate) the cached value.
    ///
    /// The next access will recompute the value by calling the getter.
    #[inline]
    pub fn invalidate(&mut self) {
        self.cached_value = None;
    }

    /// Set the cached value directly (for `__set__` support).
    #[inline]
    pub fn set_cached(&mut self, value: Value) {
        self.cached_value = Some(value);
    }

    // =========================================================================
    // Attribute Access
    // =========================================================================

    /// Get the getter function.
    #[inline]
    pub fn func(&self) -> &Value {
        &self.func
    }

    /// Get the attribute name.
    #[inline]
    pub fn attr_name(&self) -> Option<&str> {
        self.attr_name.as_deref()
    }

    /// Get the docstring.
    #[inline]
    pub fn doc(&self) -> Option<&Value> {
        self.doc.as_ref()
    }

    /// Set the attribute name (called by `__set_name__`).
    pub fn set_name(&mut self, name: String) {
        self.attr_name = Some(name);
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod cached_property_tests {
    use super::*;
    use prism_core::intern::intern;

    fn int(i: i64) -> Value {
        Value::int_unchecked(i)
    }

    fn str_val(s: &str) -> Value {
        Value::string(intern(s))
    }

    // =========================================================================
    // Construction tests
    // =========================================================================

    #[test]
    fn test_new() {
        let cp = CachedProperty::new(int(42));
        assert_eq!(cp.func().as_int(), Some(42));
        assert!(!cp.is_cached());
        assert!(cp.attr_name().is_none());
    }

    #[test]
    fn test_with_name() {
        let cp = CachedProperty::with_name(int(0), "my_prop".into());
        assert_eq!(cp.attr_name(), Some("my_prop"));
    }

    #[test]
    fn test_with_doc() {
        let cp = CachedProperty::with_doc(int(0), "prop".into(), str_val("A property."));
        assert!(cp.doc().is_some());
    }

    // =========================================================================
    // Caching behavior tests
    // =========================================================================

    #[test]
    fn test_get_or_compute_first_call() {
        let mut cp = CachedProperty::new(int(0));

        let result = cp.get_or_compute(|_func| int(42));
        assert_eq!(result.as_int(), Some(42));
        assert!(cp.is_cached());
    }

    #[test]
    fn test_get_or_compute_returns_cached() {
        let mut cp = CachedProperty::new(int(0));

        // First call
        cp.get_or_compute(|_| int(42));

        // Second call — should NOT call the closure
        let mut called = false;
        let result = cp.get_or_compute(|_| {
            called = true;
            int(999)
        });

        // Should return cached value, not new value
        assert_eq!(result.as_int(), Some(42));
        assert!(!called);
    }

    #[test]
    fn test_get_cached_none() {
        let cp = CachedProperty::new(int(0));
        assert!(cp.get_cached().is_none());
    }

    #[test]
    fn test_get_cached_some() {
        let mut cp = CachedProperty::new(int(0));
        cp.get_or_compute(|_| int(100));
        assert_eq!(cp.get_cached().unwrap().as_int(), Some(100));
    }

    // =========================================================================
    // Invalidation tests
    // =========================================================================

    #[test]
    fn test_invalidate() {
        let mut cp = CachedProperty::new(int(0));
        cp.get_or_compute(|_| int(42));
        assert!(cp.is_cached());

        cp.invalidate();
        assert!(!cp.is_cached());
    }

    #[test]
    fn test_invalidate_recomputes() {
        let mut cp = CachedProperty::new(int(0));
        let mut counter = 0;

        // First computation
        cp.get_or_compute(|_| {
            counter += 1;
            int(counter)
        });
        assert_eq!(cp.get_cached().unwrap().as_int(), Some(1));

        // Invalidate
        cp.invalidate();

        // Recompute
        cp.get_or_compute(|_| {
            counter += 1;
            int(counter)
        });
        assert_eq!(cp.get_cached().unwrap().as_int(), Some(2));
    }

    #[test]
    fn test_invalidate_uncached_noop() {
        let mut cp = CachedProperty::new(int(0));
        cp.invalidate(); // Should not panic
        assert!(!cp.is_cached());
    }

    // =========================================================================
    // Direct set tests
    // =========================================================================

    #[test]
    fn test_set_cached() {
        let mut cp = CachedProperty::new(int(0));
        cp.set_cached(int(99));
        assert!(cp.is_cached());
        assert_eq!(cp.get_cached().unwrap().as_int(), Some(99));
    }

    #[test]
    fn test_set_cached_overwrites_computed() {
        let mut cp = CachedProperty::new(int(0));
        cp.get_or_compute(|_| int(42));
        cp.set_cached(int(99));
        assert_eq!(cp.get_cached().unwrap().as_int(), Some(99));
    }

    // =========================================================================
    // set_name tests
    // =========================================================================

    #[test]
    fn test_set_name() {
        let mut cp = CachedProperty::new(int(0));
        assert!(cp.attr_name().is_none());
        cp.set_name("my_property".into());
        assert_eq!(cp.attr_name(), Some("my_property"));
    }

    // =========================================================================
    // Clone tests
    // =========================================================================

    #[test]
    fn test_clone_uncached() {
        let cp = CachedProperty::with_name(int(0), "prop".into());
        let clone = cp.clone();
        assert!(!clone.is_cached());
        assert_eq!(clone.attr_name(), Some("prop"));
    }

    #[test]
    fn test_clone_cached() {
        let mut cp = CachedProperty::new(int(0));
        cp.get_or_compute(|_| int(42));

        let clone = cp.clone();
        assert!(clone.is_cached());
        assert_eq!(clone.get_cached().unwrap().as_int(), Some(42));
    }

    // =========================================================================
    // Real-world pattern tests
    // =========================================================================

    #[test]
    fn test_expensive_computation_cached() {
        let mut cp = CachedProperty::new(int(0));
        let mut call_count = 0;

        // Simulate expensive computation
        for _ in 0..100 {
            cp.get_or_compute(|_| {
                call_count += 1;
                // "Expensive" computation
                int(42 * 42)
            });
        }

        // Should only compute once
        assert_eq!(call_count, 1);
        assert_eq!(cp.get_cached().unwrap().as_int(), Some(1764));
    }

    #[test]
    fn test_cached_property_with_string_value() {
        let mut cp = CachedProperty::new(int(0));
        let result = cp.get_or_compute(|_| str_val("computed_value"));
        assert!(result.is_string());
    }

    #[test]
    fn test_cached_property_with_none_value() {
        let mut cp = CachedProperty::new(int(0));
        let result = cp.get_or_compute(|_| Value::none());
        assert!(result.is_none());
        // None should still count as cached
        assert!(cp.is_cached());
    }

    #[test]
    fn test_cached_property_with_bool_value() {
        let mut cp = CachedProperty::new(int(0));
        let result = cp.get_or_compute(|_| Value::bool(true));
        assert_eq!(result.as_bool(), Some(true));
    }
}
