//! High-performance `partial` implementation.
//!
//! `partial(func, *args, **kwargs)` freezes some portion of a function's
//! arguments and/or keywords, producing a new callable with a simplified
//! signature.
//!
//! # Performance Characteristics
//!
//! | Operation | Time | Space |
//! |-----------|------|-------|
//! | Construction | O(a + k) | Inline for ≤8 args |
//! | Call | O(a + k + n) | Merged result vec |
//! | Attribute access | O(1) | - |
//!
//! Where a = frozen args, k = frozen kwargs, n = call-time args.
//!
//! # Memory Optimization
//!
//! Uses `SmallVec<[Value; 8]>` for frozen positional arguments, avoiding
//! heap allocation for the common case of ≤8 frozen arguments.

use prism_core::Value;
use smallvec::SmallVec;
use std::collections::HashMap;

use crate::stdlib::collections::counter::HashableValue;

// =============================================================================
// Partial
// =============================================================================

/// A partially-applied function object.
///
/// Stores a reference to the wrapped function along with pre-bound
/// positional arguments and keyword arguments. When called, the frozen
/// arguments are prepended to (or merged with) the call-time arguments.
///
/// # Nested Partial Flattening
///
/// When a `Partial` wraps another `Partial`, the frozen arguments are
/// automatically flattened to avoid unnecessary indirection.
#[derive(Debug, Clone)]
pub struct Partial {
    /// The wrapped callable's identifier or representation.
    func: Value,

    /// Pre-bound positional arguments.
    /// Uses SmallVec to avoid heap allocation for ≤8 args.
    args: SmallVec<[Value; 8]>,

    /// Pre-bound keyword arguments.
    keywords: HashMap<HashableValue, Value>,
}

impl Partial {
    /// Create a new Partial with only positional arguments.
    #[inline]
    pub fn new(func: Value, args: Vec<Value>) -> Self {
        Self {
            func,
            args: SmallVec::from_vec(args),
            keywords: HashMap::new(),
        }
    }

    /// Create a new Partial with positional and keyword arguments.
    pub fn with_kwargs(
        func: Value,
        args: Vec<Value>,
        kwargs: HashMap<HashableValue, Value>,
    ) -> Self {
        Self {
            func,
            args: SmallVec::from_vec(args),
            keywords: kwargs,
        }
    }

    /// Create from SmallVec directly (avoids allocation).
    #[inline]
    pub fn from_smallvec(
        func: Value,
        args: SmallVec<[Value; 8]>,
        keywords: HashMap<HashableValue, Value>,
    ) -> Self {
        Self {
            func,
            args,
            keywords,
        }
    }

    // =========================================================================
    // Attribute Access
    // =========================================================================

    /// Get the wrapped function.
    #[inline]
    pub fn func(&self) -> &Value {
        &self.func
    }

    /// Get the frozen positional arguments.
    #[inline]
    pub fn args(&self) -> &[Value] {
        &self.args
    }

    /// Get the frozen keyword arguments.
    #[inline]
    pub fn keywords(&self) -> &HashMap<HashableValue, Value> {
        &self.keywords
    }

    /// Get the number of frozen positional arguments.
    #[inline]
    pub fn num_args(&self) -> usize {
        self.args.len()
    }

    /// Get the number of frozen keyword arguments.
    #[inline]
    pub fn num_keywords(&self) -> usize {
        self.keywords.len()
    }

    /// Check if any arguments are frozen.
    #[inline]
    pub fn has_frozen_args(&self) -> bool {
        !self.args.is_empty() || !self.keywords.is_empty()
    }

    // =========================================================================
    // Call Support
    // =========================================================================

    /// Build the merged positional argument list for a call.
    ///
    /// Returns `frozen_args + call_args` as a single vector.
    ///
    /// # Performance
    ///
    /// Pre-allocates the exact required capacity.
    pub fn merge_args(&self, call_args: &[Value]) -> Vec<Value> {
        let mut merged = Vec::with_capacity(self.args.len() + call_args.len());
        merged.extend_from_slice(&self.args);
        merged.extend_from_slice(call_args);
        merged
    }

    /// Build the merged keyword argument map for a call.
    ///
    /// Call-time kwargs override frozen kwargs (matching Python semantics).
    pub fn merge_kwargs(
        &self,
        call_kwargs: &HashMap<HashableValue, Value>,
    ) -> HashMap<HashableValue, Value> {
        let mut merged = self.keywords.clone();
        // Call-time kwargs override frozen kwargs
        for (k, v) in call_kwargs {
            merged.insert(k.clone(), v.clone());
        }
        merged
    }

    /// Build merged kwargs from an iterator (avoids allocating input HashMap).
    pub fn merge_kwargs_iter<I>(&self, call_kwargs: I) -> HashMap<HashableValue, Value>
    where
        I: IntoIterator<Item = (HashableValue, Value)>,
    {
        let mut merged = self.keywords.clone();
        for (k, v) in call_kwargs {
            merged.insert(k, v);
        }
        merged
    }

    // =========================================================================
    // Nested Partial Flattening
    // =========================================================================

    /// Flatten nested partials into a single partial.
    ///
    /// If `self.func` is itself a Partial, extract the inner function
    /// and concatenate the argument chains.
    ///
    /// This returns the resolved function and the complete argument chain.
    pub fn flatten_args(&self, inner: &Partial) -> (Value, Vec<Value>) {
        // Inner partial's args come first, then our args
        let mut merged = Vec::with_capacity(inner.args.len() + self.args.len());
        merged.extend_from_slice(&inner.args);
        merged.extend_from_slice(&self.args);
        (inner.func.clone(), merged)
    }

    /// Flatten nested partial kwargs.
    pub fn flatten_kwargs(&self, inner: &Partial) -> HashMap<HashableValue, Value> {
        let mut merged = inner.keywords.clone();
        // Outer kwargs override inner kwargs
        for (k, v) in &self.keywords {
            merged.insert(k.clone(), v.clone());
        }
        merged
    }

    // =========================================================================
    // Mutation (for update_wrapper pattern)
    // =========================================================================

    /// Add a positional argument.
    #[inline]
    pub fn push_arg(&mut self, arg: Value) {
        self.args.push(arg);
    }

    /// Add a keyword argument.
    #[inline]
    pub fn set_keyword(&mut self, key: HashableValue, value: Value) {
        self.keywords.insert(key, value);
    }

    /// Remove a keyword argument.
    #[inline]
    pub fn remove_keyword(&mut self, key: &HashableValue) -> Option<Value> {
        self.keywords.remove(key)
    }

    /// Clear all frozen arguments.
    pub fn clear_args(&mut self) {
        self.args.clear();
        self.keywords.clear();
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod partial_tests {
    use super::*;
    use prism_core::intern::intern;

    fn int(i: i64) -> Value {
        Value::int_unchecked(i)
    }

    fn str_val(s: &str) -> Value {
        Value::string(intern(s))
    }

    fn hv(v: Value) -> HashableValue {
        HashableValue(v)
    }

    // =========================================================================
    // Construction tests
    // =========================================================================

    #[test]
    fn test_new_creates_partial() {
        let p = Partial::new(int(0), vec![int(1), int(2)]);
        assert_eq!(p.num_args(), 2);
        assert_eq!(p.num_keywords(), 0);
    }

    #[test]
    fn test_new_empty_args() {
        let p = Partial::new(int(0), vec![]);
        assert_eq!(p.num_args(), 0);
        assert!(!p.has_frozen_args());
    }

    #[test]
    fn test_with_kwargs() {
        let mut kwargs = HashMap::new();
        kwargs.insert(hv(str_val("x")), int(10));
        kwargs.insert(hv(str_val("y")), int(20));

        let p = Partial::with_kwargs(int(0), vec![int(1)], kwargs);
        assert_eq!(p.num_args(), 1);
        assert_eq!(p.num_keywords(), 2);
        assert!(p.has_frozen_args());
    }

    #[test]
    fn test_from_smallvec() {
        let args: SmallVec<[Value; 8]> = smallvec::smallvec![int(1), int(2), int(3)];
        let p = Partial::from_smallvec(int(0), args, HashMap::new());
        assert_eq!(p.num_args(), 3);
    }

    // =========================================================================
    // Attribute access tests
    // =========================================================================

    #[test]
    fn test_func_access() {
        let func = int(42);
        let p = Partial::new(func, vec![int(1)]);
        assert_eq!(p.func().as_int(), Some(42));
    }

    #[test]
    fn test_args_access() {
        let p = Partial::new(int(0), vec![int(10), int(20), int(30)]);
        let args = p.args();
        assert_eq!(args.len(), 3);
        assert_eq!(args[0].as_int(), Some(10));
        assert_eq!(args[1].as_int(), Some(20));
        assert_eq!(args[2].as_int(), Some(30));
    }

    #[test]
    fn test_keywords_access() {
        let mut kwargs = HashMap::new();
        kwargs.insert(hv(str_val("key")), int(99));

        let p = Partial::with_kwargs(int(0), vec![], kwargs);
        assert_eq!(p.keywords().len(), 1);
        assert_eq!(
            p.keywords().get(&hv(str_val("key"))).unwrap().as_int(),
            Some(99)
        );
    }

    // =========================================================================
    // Merge args tests
    // =========================================================================

    #[test]
    fn test_merge_args_prepends_frozen() {
        let p = Partial::new(int(0), vec![int(1), int(2)]);
        let merged = p.merge_args(&[int(3), int(4)]);

        let ints: Vec<i64> = merged.iter().filter_map(|v| v.as_int()).collect();
        assert_eq!(ints, vec![1, 2, 3, 4]);
    }

    #[test]
    fn test_merge_args_no_frozen() {
        let p = Partial::new(int(0), vec![]);
        let merged = p.merge_args(&[int(5), int(6)]);

        let ints: Vec<i64> = merged.iter().filter_map(|v| v.as_int()).collect();
        assert_eq!(ints, vec![5, 6]);
    }

    #[test]
    fn test_merge_args_no_call_args() {
        let p = Partial::new(int(0), vec![int(1), int(2)]);
        let merged = p.merge_args(&[]);

        let ints: Vec<i64> = merged.iter().filter_map(|v| v.as_int()).collect();
        assert_eq!(ints, vec![1, 2]);
    }

    #[test]
    fn test_merge_args_both_empty() {
        let p = Partial::new(int(0), vec![]);
        let merged = p.merge_args(&[]);
        assert!(merged.is_empty());
    }

    // =========================================================================
    // Merge kwargs tests
    // =========================================================================

    #[test]
    fn test_merge_kwargs_call_overrides_frozen() {
        let mut frozen_kwargs = HashMap::new();
        frozen_kwargs.insert(hv(str_val("x")), int(1));
        frozen_kwargs.insert(hv(str_val("y")), int(2));

        let p = Partial::with_kwargs(int(0), vec![], frozen_kwargs);

        let mut call_kwargs = HashMap::new();
        call_kwargs.insert(hv(str_val("x")), int(10)); // Override x
        call_kwargs.insert(hv(str_val("z")), int(3)); // New key

        let merged = p.merge_kwargs(&call_kwargs);
        assert_eq!(merged.get(&hv(str_val("x"))).unwrap().as_int(), Some(10)); // Overridden
        assert_eq!(merged.get(&hv(str_val("y"))).unwrap().as_int(), Some(2)); // Kept
        assert_eq!(merged.get(&hv(str_val("z"))).unwrap().as_int(), Some(3)); // New
    }

    #[test]
    fn test_merge_kwargs_no_frozen() {
        let p = Partial::new(int(0), vec![]);

        let mut call_kwargs = HashMap::new();
        call_kwargs.insert(hv(str_val("a")), int(1));

        let merged = p.merge_kwargs(&call_kwargs);
        assert_eq!(merged.len(), 1);
    }

    #[test]
    fn test_merge_kwargs_no_call_kwargs() {
        let mut frozen = HashMap::new();
        frozen.insert(hv(str_val("a")), int(1));

        let p = Partial::with_kwargs(int(0), vec![], frozen);
        let merged = p.merge_kwargs(&HashMap::new());
        assert_eq!(merged.len(), 1);
    }

    #[test]
    fn test_merge_kwargs_iter() {
        let mut frozen = HashMap::new();
        frozen.insert(hv(str_val("x")), int(1));

        let p = Partial::with_kwargs(int(0), vec![], frozen);

        let call_iter = vec![(hv(str_val("y")), int(2))];
        let merged = p.merge_kwargs_iter(call_iter);
        assert_eq!(merged.len(), 2);
    }

    // =========================================================================
    // Nested flattening tests
    // =========================================================================

    #[test]
    fn test_flatten_args() {
        let inner = Partial::new(int(99), vec![int(1), int(2)]);
        let outer = Partial::new(int(0), vec![int(3), int(4)]);

        let (func, merged) = outer.flatten_args(&inner);
        assert_eq!(func.as_int(), Some(99)); // Inner function

        let ints: Vec<i64> = merged.iter().filter_map(|v| v.as_int()).collect();
        assert_eq!(ints, vec![1, 2, 3, 4]); // Inner args + outer args
    }

    #[test]
    fn test_flatten_kwargs() {
        let mut inner_kw = HashMap::new();
        inner_kw.insert(hv(str_val("a")), int(1));
        inner_kw.insert(hv(str_val("b")), int(2));
        let inner = Partial::with_kwargs(int(99), vec![], inner_kw);

        let mut outer_kw = HashMap::new();
        outer_kw.insert(hv(str_val("b")), int(20)); // Override b
        outer_kw.insert(hv(str_val("c")), int(3)); // New
        let outer = Partial::with_kwargs(int(0), vec![], outer_kw);

        let merged = outer.flatten_kwargs(&inner);
        assert_eq!(merged.get(&hv(str_val("a"))).unwrap().as_int(), Some(1));
        assert_eq!(merged.get(&hv(str_val("b"))).unwrap().as_int(), Some(20)); // Overridden
        assert_eq!(merged.get(&hv(str_val("c"))).unwrap().as_int(), Some(3));
    }

    // =========================================================================
    // Mutation tests
    // =========================================================================

    #[test]
    fn test_push_arg() {
        let mut p = Partial::new(int(0), vec![int(1)]);
        p.push_arg(int(2));
        assert_eq!(p.num_args(), 2);
        assert_eq!(p.args()[1].as_int(), Some(2));
    }

    #[test]
    fn test_set_keyword() {
        let mut p = Partial::new(int(0), vec![]);
        p.set_keyword(hv(str_val("x")), int(10));
        assert_eq!(p.num_keywords(), 1);
    }

    #[test]
    fn test_remove_keyword() {
        let mut kwargs = HashMap::new();
        kwargs.insert(hv(str_val("x")), int(10));
        let mut p = Partial::with_kwargs(int(0), vec![], kwargs);

        let removed = p.remove_keyword(&hv(str_val("x")));
        assert_eq!(removed.unwrap().as_int(), Some(10));
        assert_eq!(p.num_keywords(), 0);
    }

    #[test]
    fn test_remove_nonexistent_keyword() {
        let mut p = Partial::new(int(0), vec![]);
        let removed = p.remove_keyword(&hv(str_val("missing")));
        assert!(removed.is_none());
    }

    #[test]
    fn test_clear_args() {
        let mut kwargs = HashMap::new();
        kwargs.insert(hv(str_val("x")), int(10));
        let mut p = Partial::with_kwargs(int(0), vec![int(1), int(2)], kwargs);

        p.clear_args();
        assert_eq!(p.num_args(), 0);
        assert_eq!(p.num_keywords(), 0);
        assert!(!p.has_frozen_args());
    }

    // =========================================================================
    // SmallVec optimization tests
    // =========================================================================

    #[test]
    fn test_smallvec_inline_storage() {
        // ≤8 args should be stored inline (no heap allocation)
        let p = Partial::new(
            int(0),
            vec![
                int(1),
                int(2),
                int(3),
                int(4),
                int(5),
                int(6),
                int(7),
                int(8),
            ],
        );
        assert_eq!(p.num_args(), 8);
        // SmallVec should use inline storage for 8 elements
        assert!(!p.args.spilled());
    }

    #[test]
    fn test_smallvec_spills_to_heap() {
        // >8 args should spill to heap
        let args: Vec<Value> = (0..20).map(|i| int(i)).collect();
        let p = Partial::new(int(0), args);
        assert_eq!(p.num_args(), 20);
        assert!(p.args.spilled());
    }

    // =========================================================================
    // Stress tests
    // =========================================================================

    #[test]
    fn test_merge_many_args() {
        let frozen: Vec<Value> = (0..100).map(|i| int(i)).collect();
        let call_args: Vec<Value> = (100..200).map(|i| int(i)).collect();

        let p = Partial::new(int(0), frozen);
        let merged = p.merge_args(&call_args);

        assert_eq!(merged.len(), 200);
        for (i, v) in merged.iter().enumerate() {
            assert_eq!(v.as_int(), Some(i as i64));
        }
    }

    #[test]
    fn test_merge_many_kwargs() {
        let mut frozen = HashMap::new();
        for i in 0..50 {
            frozen.insert(hv(int(i)), int(i * 10));
        }

        let p = Partial::with_kwargs(int(0), vec![], frozen);

        let mut call_kw = HashMap::new();
        for i in 25..75 {
            call_kw.insert(hv(int(i)), int(i * 100));
        }

        let merged = p.merge_kwargs(&call_kw);
        // 0..24 from frozen, 25..74 from call (overrides 25..49)
        assert_eq!(merged.len(), 75);
    }

    // =========================================================================
    // Edge case tests
    // =========================================================================

    #[test]
    fn test_partial_with_none_func() {
        let p = Partial::new(Value::none(), vec![int(1)]);
        assert!(p.func().is_none());
    }

    #[test]
    fn test_partial_with_bool_args() {
        let p = Partial::new(int(0), vec![Value::bool(true), Value::bool(false)]);
        assert_eq!(p.args()[0].as_bool(), Some(true));
        assert_eq!(p.args()[1].as_bool(), Some(false));
    }

    #[test]
    fn test_partial_with_float_args() {
        let p = Partial::new(int(0), vec![Value::float(3.14), Value::float(2.71)]);
        assert_eq!(p.args()[0].as_float(), Some(3.14));
        assert_eq!(p.args()[1].as_float(), Some(2.71));
    }

    #[test]
    fn test_partial_clone() {
        let mut kwargs = HashMap::new();
        kwargs.insert(hv(str_val("x")), int(10));
        let p = Partial::with_kwargs(int(0), vec![int(1), int(2)], kwargs);

        let p2 = p.clone();
        assert_eq!(p2.num_args(), 2);
        assert_eq!(p2.num_keywords(), 1);
    }

    #[test]
    fn test_partial_with_string_args() {
        let p = Partial::new(int(0), vec![str_val("hello"), str_val("world")]);
        assert!(p.args()[0].is_string());
        assert!(p.args()[1].is_string());
    }
}
