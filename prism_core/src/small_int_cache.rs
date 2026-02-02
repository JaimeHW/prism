//! Small Integer Cache for Python semantics.
//!
//! Python caches small integers in the range [-5, 256] so that:
//! - `a = 5; b = 5; a is b` returns True
//! - Common integers are pre-computed for performance
//!
//! # Performance Benefits
//!
//! 1. **Identity semantics**: Python's `is` operator works correctly for cached integers
//! 2. **Zero computation**: Cached integers avoid bit manipulation at runtime
//! 3. **Cache-friendly**: Hot integers are stored contiguously
//!
//! # Implementation
//!
//! We use a const-initialized static array containing pre-computed Value bitpatterns.
//! The cache is accessed via `SmallIntCache::get(i)` which performs a single bounds
//! check and array lookup.

use crate::Value;

/// Minimum cached small integer (inclusive).
pub const SMALL_INT_CACHE_MIN: i64 = -5;

/// Maximum cached small integer (inclusive).
pub const SMALL_INT_CACHE_MAX: i64 = 256;

/// Total number of cached small integers.
pub const SMALL_INT_CACHE_SIZE: usize = (SMALL_INT_CACHE_MAX - SMALL_INT_CACHE_MIN + 1) as usize;

/// Static array of pre-computed small integer Values.
///
/// Index 0 = -5, Index 5 = 0, Index 261 = 256
static SMALL_INT_CACHE: [Value; SMALL_INT_CACHE_SIZE] = {
    let mut cache = [Value::none(); SMALL_INT_CACHE_SIZE];
    let mut i = 0;
    while i < SMALL_INT_CACHE_SIZE {
        let val = SMALL_INT_CACHE_MIN + i as i64;
        // Safe because all values in [-5, 256] fit in small int range
        cache[i] = Value::int_unchecked(val);
        i += 1;
    }
    cache
};

/// Small integer cache for fast integer Value creation.
///
/// Provides O(1) lookup for integers in the range [-5, 256].
pub struct SmallIntCache;

impl SmallIntCache {
    /// Get a cached small integer, or None if out of range.
    ///
    /// # Performance
    ///
    /// This is a single bounds check + array index, approximately:
    /// - 2-3 instructions on the hot path
    /// - Always returns a reference to static memory (no allocation)
    ///
    /// # Example
    ///
    /// ```
    /// use prism_core::small_int_cache::SmallIntCache;
    ///
    /// // Cached (fast path)
    /// let five = SmallIntCache::get(5);
    /// assert!(five.is_some());
    ///
    /// // Not cached (returns None)
    /// let big = SmallIntCache::get(1000);
    /// assert!(big.is_none());
    /// ```
    #[inline]
    pub fn get(value: i64) -> Option<Value> {
        if value >= SMALL_INT_CACHE_MIN && value <= SMALL_INT_CACHE_MAX {
            let index = (value - SMALL_INT_CACHE_MIN) as usize;
            Some(SMALL_INT_CACHE[index])
        } else {
            None
        }
    }

    /// Get a cached small integer, panicking if out of range.
    ///
    /// # Panics
    ///
    /// Panics if value is not in [-5, 256].
    #[inline]
    pub fn get_unchecked(value: i64) -> Value {
        debug_assert!(
            value >= SMALL_INT_CACHE_MIN && value <= SMALL_INT_CACHE_MAX,
            "Integer {} is outside small int cache range [{}, {}]",
            value,
            SMALL_INT_CACHE_MIN,
            SMALL_INT_CACHE_MAX
        );
        let index = (value - SMALL_INT_CACHE_MIN) as usize;
        SMALL_INT_CACHE[index]
    }

    /// Check if a value is in the cached range.
    #[inline]
    pub const fn is_cached(value: i64) -> bool {
        value >= SMALL_INT_CACHE_MIN && value <= SMALL_INT_CACHE_MAX
    }

    /// Get the cache index for a value (assuming it's in range).
    #[inline]
    pub const fn index_of(value: i64) -> usize {
        debug_assert!(Self::is_cached(value));
        (value - SMALL_INT_CACHE_MIN) as usize
    }

    /// Get the minimum cached value.
    #[inline]
    pub const fn min() -> i64 {
        SMALL_INT_CACHE_MIN
    }

    /// Get the maximum cached value.
    #[inline]
    pub const fn max() -> i64 {
        SMALL_INT_CACHE_MAX
    }

    /// Get the number of cached values.
    #[inline]
    pub const fn size() -> usize {
        SMALL_INT_CACHE_SIZE
    }

    /// Get a direct reference to the cache array (for JIT integration).
    ///
    /// # Safety
    ///
    /// The returned pointer is valid for the lifetime of the program.
    /// Only access indices in [0, SMALL_INT_CACHE_SIZE).
    #[inline]
    pub fn cache_ptr() -> *const Value {
        SMALL_INT_CACHE.as_ptr()
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cache_constants() {
        assert_eq!(SMALL_INT_CACHE_MIN, -5);
        assert_eq!(SMALL_INT_CACHE_MAX, 256);
        assert_eq!(SMALL_INT_CACHE_SIZE, 262);
    }

    #[test]
    fn test_cache_zero() {
        let cached = SmallIntCache::get(0).unwrap();
        assert!(cached.is_int());
        assert_eq!(cached.as_int(), Some(0));
    }

    #[test]
    fn test_cache_positive() {
        for i in 0..=256 {
            let cached = SmallIntCache::get(i).expect(&format!("Should cache {}", i));
            assert_eq!(cached.as_int(), Some(i), "Mismatch for {}", i);
        }
    }

    #[test]
    fn test_cache_negative() {
        for i in -5..0 {
            let cached = SmallIntCache::get(i).expect(&format!("Should cache {}", i));
            assert_eq!(cached.as_int(), Some(i), "Mismatch for {}", i);
        }
    }

    #[test]
    fn test_cache_boundaries() {
        // Min boundary
        assert!(SmallIntCache::get(-5).is_some());
        assert!(SmallIntCache::get(-6).is_none());

        // Max boundary
        assert!(SmallIntCache::get(256).is_some());
        assert!(SmallIntCache::get(257).is_none());
    }

    #[test]
    fn test_cache_miss() {
        assert!(SmallIntCache::get(-100).is_none());
        assert!(SmallIntCache::get(1000).is_none());
        assert!(SmallIntCache::get(i64::MAX).is_none());
        assert!(SmallIntCache::get(i64::MIN).is_none());
    }

    #[test]
    fn test_is_cached() {
        assert!(SmallIntCache::is_cached(0));
        assert!(SmallIntCache::is_cached(-5));
        assert!(SmallIntCache::is_cached(256));
        assert!(!SmallIntCache::is_cached(-6));
        assert!(!SmallIntCache::is_cached(257));
    }

    #[test]
    fn test_index_of() {
        assert_eq!(SmallIntCache::index_of(-5), 0);
        assert_eq!(SmallIntCache::index_of(0), 5);
        assert_eq!(SmallIntCache::index_of(1), 6);
        assert_eq!(SmallIntCache::index_of(256), 261);
    }

    #[test]
    fn test_get_unchecked() {
        for i in -5..=256 {
            let cached = SmallIntCache::get_unchecked(i);
            assert_eq!(cached.as_int(), Some(i));
        }
    }

    #[test]
    fn test_cache_identity() {
        // Verify that cached values have identical bit patterns
        let a = SmallIntCache::get(42).unwrap();
        let b = SmallIntCache::get(42).unwrap();
        assert_eq!(a.to_bits(), b.to_bits());
    }

    #[test]
    fn test_cache_vs_direct() {
        // Cached values should equal directly constructed values
        for i in -5..=256 {
            let cached = SmallIntCache::get(i).unwrap();
            let direct = Value::int(i).unwrap();
            assert_eq!(cached, direct, "Mismatch for {}", i);
            assert_eq!(
                cached.to_bits(),
                direct.to_bits(),
                "Bits mismatch for {}",
                i
            );
        }
    }

    #[test]
    fn test_cache_ptr() {
        let ptr = SmallIntCache::cache_ptr();
        assert!(!ptr.is_null());

        // Verify we can read through the pointer
        unsafe {
            let zero = *ptr.add(5); // index 5 = value 0
            assert_eq!(zero.as_int(), Some(0));
        }
    }

    #[test]
    fn test_cache_static_initialization() {
        // Verify all cached values are correctly initialized
        for i in 0..SMALL_INT_CACHE_SIZE {
            let expected_value = SMALL_INT_CACHE_MIN + i as i64;
            assert_eq!(
                SMALL_INT_CACHE[i].as_int(),
                Some(expected_value),
                "Index {} should contain {}",
                i,
                expected_value
            );
        }
    }

    #[test]
    fn test_python_common_values() {
        // Verify commonly used Python integers are cached
        let common_values = [
            -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, // small numbers
            100, 200, 255, 256, // byte-related
        ];

        for &val in &common_values {
            assert!(
                SmallIntCache::get(val).is_some(),
                "Common value {} should be cached",
                val
            );
        }
    }

    #[test]
    fn test_loop_counter_values() {
        // Loop counters and common ranges should be cached
        for i in 0..100 {
            assert!(
                SmallIntCache::get(i).is_some(),
                "Loop counter value {} should be cached",
                i
            );
        }
    }

    // =========================================================================
    // Performance-focused tests
    // =========================================================================

    #[test]
    fn test_cache_lookup_equivalent_to_construction() {
        // Both paths should produce identical results
        for i in -5..=256 {
            let from_cache = SmallIntCache::get(i).unwrap();
            let from_constructor = Value::int(i).unwrap();

            // Value equality
            assert_eq!(from_cache, from_constructor);

            // Bit-level equality (important for identity semantics)
            assert_eq!(from_cache.to_bits(), from_constructor.to_bits());
        }
    }

    #[test]
    fn test_helpers() {
        assert_eq!(SmallIntCache::min(), -5);
        assert_eq!(SmallIntCache::max(), 256);
        assert_eq!(SmallIntCache::size(), 262);
    }
}
