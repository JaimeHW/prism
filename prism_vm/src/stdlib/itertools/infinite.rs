//! Infinite iterator constructors.
//!
//! Provides `count`, `cycle`, and `repeat` — the three infinite iterators
//! from Python's `itertools` module.
//!
//! # Performance Characteristics
//!
//! | Iterator | Time per `next()` | Space |
//! |----------|-------------------|-------|
//! | `Count`  | O(1) — single add | O(1) inline |
//! | `Cycle`  | O(1) amortized    | O(n) pool clone |
//! | `Repeat` | O(1) — clone      | O(1) inline |
//!
//! All iterators are fused and provide accurate `size_hint()` where applicable.

use prism_core::Value;

// =============================================================================
// Count
// =============================================================================

/// Infinite counter starting at `start` with a given `step`.
///
/// Equivalent to Python's `itertools.count(start=0, step=1)`.
///
/// # Performance
///
/// Each `next()` call is a single integer addition — O(1) time, O(1) space.
/// For integer-only counts, uses `i64` arithmetic (no float conversion).
/// When either `start` or `step` is a float, promotes to `f64` arithmetic.
#[derive(Debug, Clone)]
pub struct Count {
    current: CountState,
    step: CountStep,
}

/// Internal state: either pure integer or float arithmetic.
#[derive(Debug, Clone)]
enum CountState {
    Int(i64),
    Float(f64),
}

/// Step value for count iterator.
#[derive(Debug, Clone, Copy)]
enum CountStep {
    Int(i64),
    Float(f64),
}

impl Count {
    /// Create a new integer counter.
    ///
    /// # Examples
    /// ```ignore
    /// let c = Count::new(0, 1);
    /// // yields: 0, 1, 2, 3, ...
    /// ```
    #[inline]
    pub fn new(start: i64, step: i64) -> Self {
        Self {
            current: CountState::Int(start),
            step: CountStep::Int(step),
        }
    }

    /// Create a float counter.
    ///
    /// # Examples
    /// ```ignore
    /// let c = Count::new_float(0.5, 0.1);
    /// // yields: 0.5, 0.6, 0.7, ...
    /// ```
    #[inline]
    pub fn new_float(start: f64, step: f64) -> Self {
        Self {
            current: CountState::Float(start),
            step: CountStep::Float(step),
        }
    }

    /// Create a counter from Value arguments.
    ///
    /// Automatically promotes to float arithmetic if either argument is a float.
    #[inline]
    pub fn from_values(start: &Value, step: &Value) -> Option<Self> {
        // Try pure integer first
        if let (Some(s), Some(st)) = (start.as_int(), step.as_int()) {
            return Some(Self::new(s, st));
        }
        // Fall back to float coercion
        let s = start
            .as_float()
            .or_else(|| start.as_int().map(|i| i as f64))?;
        let st = step
            .as_float()
            .or_else(|| step.as_int().map(|i| i as f64))?;
        Some(Self::new_float(s, st))
    }
}

impl Iterator for Count {
    type Item = Value;

    #[inline]
    fn next(&mut self) -> Option<Value> {
        let step = self.step;
        match &mut self.current {
            CountState::Int(curr) => match step {
                CountStep::Int(s) => {
                    let val = *curr;
                    *curr = curr.wrapping_add(s);
                    Some(Value::int_unchecked(val))
                }
                CountStep::Float(s) => {
                    let val = *curr as f64;
                    self.current = CountState::Float(val + s);
                    Some(Value::float(val))
                }
            },
            CountState::Float(curr) => {
                let val = *curr;
                match step {
                    CountStep::Float(s) => *curr += s,
                    CountStep::Int(s) => *curr += s as f64,
                }
                Some(Value::float(val))
            }
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        (usize::MAX, None) // infinite
    }
}

impl std::iter::FusedIterator for Count {}

// =============================================================================
// Cycle
// =============================================================================

/// Infinitely repeats the elements of a saved pool.
///
/// Equivalent to Python's `itertools.cycle(iterable)`.
///
/// # Performance
///
/// - First pass: O(1) per element (consuming the source iterator + saving)
/// - Subsequent passes: O(1) per element (index into pool)
/// - Space: O(n) where n is the number of elements in the source
///
/// The pool is pre-allocated with the source iterator's `size_hint` for
/// zero-reallocation in the common case.
#[derive(Debug, Clone)]
pub struct Cycle {
    /// Saved elements for replay.
    pool: Vec<Value>,
    /// Current index into the pool during replay.
    index: usize,
    /// Whether we've finished consuming the source.
    exhausted: bool,
    /// Source iterator (consumed on first pass, then dropped conceptually).
    source: Vec<Value>,
    /// Position in source during first pass.
    source_pos: usize,
}

impl Cycle {
    /// Create from an iterable, consuming it into a pool.
    ///
    /// The pool is eagerly built for maximum replay performance.
    pub fn new(iterable: impl IntoIterator<Item = Value>) -> Self {
        let iter = iterable.into_iter();
        let (hint, _) = iter.size_hint();
        let mut pool = Vec::with_capacity(hint);
        let source: Vec<Value> = iter.collect();
        Self {
            pool,
            index: 0,
            exhausted: false,
            source,
            source_pos: 0,
        }
    }

    /// Create from a pre-built pool (avoids double collection).
    #[inline]
    pub fn from_pool(pool: Vec<Value>) -> Self {
        Self {
            pool,
            index: 0,
            exhausted: true,
            source: Vec::new(),
            source_pos: 0,
        }
    }

    /// Returns the number of elements in the cycle pool.
    #[inline]
    pub fn pool_len(&self) -> usize {
        if self.exhausted {
            self.pool.len()
        } else {
            self.source.len()
        }
    }
}

impl Iterator for Cycle {
    type Item = Value;

    #[inline]
    fn next(&mut self) -> Option<Value> {
        if !self.exhausted {
            // First pass: consume from source
            if self.source_pos < self.source.len() {
                let val = self.source[self.source_pos].clone();
                self.pool.push(val.clone());
                self.source_pos += 1;
                return Some(val);
            }
            // Source exhausted
            self.exhausted = true;
            // Free the source
            self.source = Vec::new();
            self.index = 0;
        }

        // Replay from pool
        if self.pool.is_empty() {
            return None; // Empty source = no cycle
        }

        let val = self.pool[self.index].clone();
        self.index = (self.index + 1) % self.pool.len();
        Some(val)
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        if !self.exhausted && self.source.is_empty() && self.pool.is_empty() {
            (0, Some(0))
        } else {
            (usize::MAX, None)
        }
    }
}

impl std::iter::FusedIterator for Cycle {}

// =============================================================================
// Repeat
// =============================================================================

/// Repeats a single value, either infinitely or a fixed number of times.
///
/// Equivalent to Python's `itertools.repeat(object[, times])`.
///
/// # Performance
///
/// - O(1) per `next()` (clone of the stored value)
/// - O(1) space (stores one value + optional counter)
/// - Implements `ExactSizeIterator` when bounded
#[derive(Debug, Clone)]
pub struct Repeat {
    value: Value,
    remaining: Option<usize>,
}

impl Repeat {
    /// Create an infinite repeater.
    #[inline]
    pub fn forever(value: Value) -> Self {
        Self {
            value,
            remaining: None,
        }
    }

    /// Create a bounded repeater.
    #[inline]
    pub fn times(value: Value, n: usize) -> Self {
        Self {
            value,
            remaining: Some(n),
        }
    }

    /// Check if this repeater is bounded.
    #[inline]
    pub fn is_bounded(&self) -> bool {
        self.remaining.is_some()
    }

    /// Get remaining count (None if infinite).
    #[inline]
    pub fn remaining(&self) -> Option<usize> {
        self.remaining
    }
}

impl Iterator for Repeat {
    type Item = Value;

    #[inline]
    fn next(&mut self) -> Option<Value> {
        match &mut self.remaining {
            Some(0) => None,
            Some(n) => {
                *n -= 1;
                Some(self.value.clone())
            }
            None => Some(self.value.clone()),
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        match self.remaining {
            Some(n) => (n, Some(n)),
            None => (usize::MAX, None),
        }
    }
}

impl ExactSizeIterator for Repeat {
    fn len(&self) -> usize {
        self.remaining.unwrap_or(usize::MAX)
    }
}

impl std::iter::FusedIterator for Repeat {}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use prism_core::Value;

    fn int(i: i64) -> Value {
        Value::int_unchecked(i)
    }

    // =========================================================================
    // Count tests
    // =========================================================================

    #[test]
    fn test_count_default() {
        let vals: Vec<i64> = Count::new(0, 1)
            .take(5)
            .filter_map(|v| v.as_int())
            .collect();
        assert_eq!(vals, vec![0, 1, 2, 3, 4]);
    }

    #[test]
    fn test_count_custom_start() {
        let vals: Vec<i64> = Count::new(10, 1)
            .take(4)
            .filter_map(|v| v.as_int())
            .collect();
        assert_eq!(vals, vec![10, 11, 12, 13]);
    }

    #[test]
    fn test_count_custom_step() {
        let vals: Vec<i64> = Count::new(0, 3)
            .take(5)
            .filter_map(|v| v.as_int())
            .collect();
        assert_eq!(vals, vec![0, 3, 6, 9, 12]);
    }

    #[test]
    fn test_count_negative_step() {
        let vals: Vec<i64> = Count::new(10, -2)
            .take(5)
            .filter_map(|v| v.as_int())
            .collect();
        assert_eq!(vals, vec![10, 8, 6, 4, 2]);
    }

    #[test]
    fn test_count_zero_step() {
        let vals: Vec<i64> = Count::new(42, 0)
            .take(3)
            .filter_map(|v| v.as_int())
            .collect();
        assert_eq!(vals, vec![42, 42, 42]);
    }

    #[test]
    fn test_count_float() {
        let vals: Vec<f64> = Count::new_float(0.0, 0.5)
            .take(5)
            .filter_map(|v| v.as_float())
            .collect();
        assert_eq!(vals, vec![0.0, 0.5, 1.0, 1.5, 2.0]);
    }

    #[test]
    fn test_count_float_negative_step() {
        let vals: Vec<f64> = Count::new_float(1.0, -0.25)
            .take(5)
            .filter_map(|v| v.as_float())
            .collect();
        assert_eq!(vals, vec![1.0, 0.75, 0.5, 0.25, 0.0]);
    }

    #[test]
    fn test_count_large_start() {
        // Use a large but NaN-boxing-safe value (2^47 - 5 fits within the 48-bit payload)
        let big: i64 = (1 << 47) - 5;
        let vals: Vec<i64> = Count::new(big, 1)
            .take(3)
            .filter_map(|v| v.as_int())
            .collect();
        assert_eq!(vals, vec![big, big + 1, big + 2]);
    }

    #[test]
    fn test_count_from_values_int() {
        let c = Count::from_values(&int(5), &int(2)).unwrap();
        let vals: Vec<i64> = c.take(3).filter_map(|v| v.as_int()).collect();
        assert_eq!(vals, vec![5, 7, 9]);
    }

    #[test]
    fn test_count_from_values_float() {
        let c = Count::from_values(&Value::float(1.0), &Value::float(0.1)).unwrap();
        let vals: Vec<f64> = c.take(3).filter_map(|v| v.as_float()).collect();
        assert!((vals[0] - 1.0).abs() < 1e-10);
        assert!((vals[1] - 1.1).abs() < 1e-10);
        assert!((vals[2] - 1.2).abs() < 1e-10);
    }

    #[test]
    fn test_count_from_values_mixed() {
        // int start + float step → promote to float
        let c = Count::from_values(&int(0), &Value::float(0.5)).unwrap();
        let vals: Vec<f64> = c.take(3).filter_map(|v| v.as_float()).collect();
        assert!((vals[0] - 0.0).abs() < 1e-10);
        assert!((vals[1] - 0.5).abs() < 1e-10);
        assert!((vals[2] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_count_size_hint_infinite() {
        let c = Count::new(0, 1);
        let (lo, hi) = c.size_hint();
        assert_eq!(lo, usize::MAX);
        assert!(hi.is_none());
    }

    #[test]
    fn test_count_is_fused() {
        // FusedIterator: Count never returns None, so it's trivially fused.
        let mut c = Count::new(0, 1);
        for _ in 0..1000 {
            assert!(c.next().is_some());
        }
    }

    #[test]
    fn test_count_clone() {
        let c1 = Count::new(0, 1);
        let mut c2 = c1.clone();
        let vals: Vec<i64> = c2.by_ref().take(3).filter_map(|v| v.as_int()).collect();
        assert_eq!(vals, vec![0, 1, 2]);
        // Original should still start from 0
        let mut c1 = c1;
        assert_eq!(c1.next().unwrap().as_int(), Some(0));
    }

    #[test]
    fn test_count_step_1_sequence() {
        // Verify count(1, 1) matches 1, 2, 3, ...
        let vals: Vec<i64> = Count::new(1, 1)
            .take(100)
            .filter_map(|v| v.as_int())
            .collect();
        for (i, v) in vals.iter().enumerate() {
            assert_eq!(*v, (i + 1) as i64);
        }
    }

    // =========================================================================
    // Cycle tests
    // =========================================================================

    #[test]
    fn test_cycle_basic() {
        let vals: Vec<i64> = Cycle::new(vec![int(1), int(2), int(3)])
            .take(9)
            .filter_map(|v| v.as_int())
            .collect();
        assert_eq!(vals, vec![1, 2, 3, 1, 2, 3, 1, 2, 3]);
    }

    #[test]
    fn test_cycle_single_element() {
        let vals: Vec<i64> = Cycle::new(vec![int(42)])
            .take(5)
            .filter_map(|v| v.as_int())
            .collect();
        assert_eq!(vals, vec![42, 42, 42, 42, 42]);
    }

    #[test]
    fn test_cycle_empty() {
        let mut c = Cycle::new(Vec::<Value>::new());
        assert!(c.next().is_none());
        assert!(c.next().is_none()); // fused
    }

    #[test]
    fn test_cycle_two_elements() {
        let vals: Vec<i64> = Cycle::new(vec![int(0), int(1)])
            .take(6)
            .filter_map(|v| v.as_int())
            .collect();
        assert_eq!(vals, vec![0, 1, 0, 1, 0, 1]);
    }

    #[test]
    fn test_cycle_from_pool() {
        let pool = vec![int(10), int(20)];
        let vals: Vec<i64> = Cycle::from_pool(pool)
            .take(5)
            .filter_map(|v| v.as_int())
            .collect();
        assert_eq!(vals, vec![10, 20, 10, 20, 10]);
    }

    #[test]
    fn test_cycle_pool_len() {
        let c = Cycle::new(vec![int(1), int(2), int(3)]);
        assert_eq!(c.pool_len(), 3);
    }

    #[test]
    fn test_cycle_pool_len_from_pool() {
        let c = Cycle::from_pool(vec![int(1), int(2)]);
        assert_eq!(c.pool_len(), 2);
    }

    #[test]
    fn test_cycle_size_hint_nonempty() {
        let c = Cycle::new(vec![int(1)]);
        let (lo, hi) = c.size_hint();
        assert_eq!(lo, usize::MAX);
        assert!(hi.is_none());
    }

    #[test]
    fn test_cycle_size_hint_empty() {
        let c = Cycle::new(Vec::<Value>::new());
        let (lo, hi) = c.size_hint();
        assert_eq!(lo, 0);
        assert_eq!(hi, Some(0));
    }

    #[test]
    fn test_cycle_preserves_order() {
        let vals: Vec<i64> = Cycle::new(vec![int(5), int(4), int(3), int(2), int(1)])
            .take(10)
            .filter_map(|v| v.as_int())
            .collect();
        assert_eq!(vals, vec![5, 4, 3, 2, 1, 5, 4, 3, 2, 1]);
    }

    #[test]
    fn test_cycle_clone() {
        let c1 = Cycle::from_pool(vec![int(1), int(2)]);
        let mut c2 = c1.clone();
        let v: Vec<i64> = c2.by_ref().take(3).filter_map(|v| v.as_int()).collect();
        assert_eq!(v, vec![1, 2, 1]);
    }

    #[test]
    fn test_cycle_large_pool() {
        let pool: Vec<Value> = (0..100).map(|i| int(i)).collect();
        let vals: Vec<i64> = Cycle::from_pool(pool)
            .take(250)
            .filter_map(|v| v.as_int())
            .collect();
        assert_eq!(vals.len(), 250);
        for (i, v) in vals.iter().enumerate() {
            assert_eq!(*v, (i % 100) as i64);
        }
    }

    // =========================================================================
    // Repeat tests
    // =========================================================================

    #[test]
    fn test_repeat_forever_take() {
        let vals: Vec<i64> = Repeat::forever(int(7))
            .take(5)
            .filter_map(|v| v.as_int())
            .collect();
        assert_eq!(vals, vec![7, 7, 7, 7, 7]);
    }

    #[test]
    fn test_repeat_bounded() {
        let vals: Vec<i64> = Repeat::times(int(3), 4)
            .filter_map(|v| v.as_int())
            .collect();
        assert_eq!(vals, vec![3, 3, 3, 3]);
    }

    #[test]
    fn test_repeat_zero_times() {
        let vals: Vec<Value> = Repeat::times(int(99), 0).collect();
        assert!(vals.is_empty());
    }

    #[test]
    fn test_repeat_one_time() {
        let vals: Vec<i64> = Repeat::times(int(42), 1)
            .filter_map(|v| v.as_int())
            .collect();
        assert_eq!(vals, vec![42]);
    }

    #[test]
    fn test_repeat_is_bounded() {
        assert!(!Repeat::forever(int(1)).is_bounded());
        assert!(Repeat::times(int(1), 5).is_bounded());
    }

    #[test]
    fn test_repeat_remaining() {
        let mut r = Repeat::times(int(1), 3);
        assert_eq!(r.remaining(), Some(3));
        r.next();
        assert_eq!(r.remaining(), Some(2));
        r.next();
        assert_eq!(r.remaining(), Some(1));
        r.next();
        assert_eq!(r.remaining(), Some(0));
    }

    #[test]
    fn test_repeat_remaining_infinite() {
        let r = Repeat::forever(int(1));
        assert_eq!(r.remaining(), None);
    }

    #[test]
    fn test_repeat_size_hint_bounded() {
        let r = Repeat::times(int(1), 10);
        assert_eq!(r.size_hint(), (10, Some(10)));
    }

    #[test]
    fn test_repeat_size_hint_infinite() {
        let r = Repeat::forever(int(1));
        let (lo, hi) = r.size_hint();
        assert_eq!(lo, usize::MAX);
        assert!(hi.is_none());
    }

    #[test]
    fn test_repeat_exact_size_bounded() {
        let r = Repeat::times(int(1), 5);
        assert_eq!(r.len(), 5);
    }

    #[test]
    fn test_repeat_fused_after_exhaustion() {
        let mut r = Repeat::times(int(1), 2);
        assert!(r.next().is_some());
        assert!(r.next().is_some());
        assert!(r.next().is_none());
        assert!(r.next().is_none()); // fused
    }

    #[test]
    fn test_repeat_with_none_value() {
        let vals: Vec<Value> = Repeat::times(Value::none(), 3).collect();
        assert_eq!(vals.len(), 3);
        for v in &vals {
            assert!(v.is_none());
        }
    }

    #[test]
    fn test_repeat_with_bool_value() {
        let vals: Vec<Value> = Repeat::times(Value::bool(true), 2).collect();
        assert_eq!(vals.len(), 2);
        for v in &vals {
            assert_eq!(v.as_bool(), Some(true));
        }
    }

    #[test]
    fn test_repeat_clone() {
        let r1 = Repeat::times(int(5), 3);
        let mut r2 = r1.clone();
        assert_eq!(r2.next().unwrap().as_int(), Some(5));
        // r1 unaffected
        let vals: Vec<i64> = r1.filter_map(|v| v.as_int()).collect();
        assert_eq!(vals, vec![5, 5, 5]);
    }

    #[test]
    fn test_repeat_stress_large_count() {
        let r = Repeat::times(int(1), 100_000);
        assert_eq!(r.count(), 100_000);
    }

    // =========================================================================
    // Cross-iterator tests
    // =========================================================================

    #[test]
    fn test_count_zipped_with_repeat() {
        // zip(count(0), repeat(1, 5)) → [(0,1), (1,1), (2,1), (3,1), (4,1)]
        let count_iter = Count::new(0, 1);
        let repeat_iter = Repeat::times(int(1), 5);
        let pairs: Vec<(i64, i64)> = count_iter
            .zip(repeat_iter)
            .map(|(a, b)| (a.as_int().unwrap(), b.as_int().unwrap()))
            .collect();
        assert_eq!(pairs, vec![(0, 1), (1, 1), (2, 1), (3, 1), (4, 1)]);
    }

    #[test]
    fn test_cycle_truncated_by_count() {
        // Take from cycle using a bounded repeat as the length limiter
        let cycle_iter = Cycle::new(vec![int(10), int(20)]);
        let vals: Vec<i64> = cycle_iter.take(7).filter_map(|v| v.as_int()).collect();
        assert_eq!(vals, vec![10, 20, 10, 20, 10, 20, 10]);
    }

    #[test]
    fn test_count_negative_start() {
        let vals: Vec<i64> = Count::new(-5, 1)
            .take(10)
            .filter_map(|v| v.as_int())
            .collect();
        assert_eq!(vals, vec![-5, -4, -3, -2, -1, 0, 1, 2, 3, 4]);
    }
}
