//! Itertools recipes — common patterns from Python's itertools documentation.
//!
//! These are utility iterators and functions built on top of the core
//! itertools primitives, matching the "recipes" section of Python's
//! itertools documentation.
//!
//! # Functions
//!
//! | Function | Description |
//! |----------|-------------|
//! | `flatten` | Flatten one level of nesting |
//! | `unique_everseen` | Unique elements preserving order |
//! | `unique_justseen` | Remove consecutive duplicates |
//! | `sliding_window` | Generic n-width sliding window |
//! | `roundrobin` | Interleave multiple iterators |
//! | `accumulate` | Running totals/reductions |
//! | `partition` | Split by predicate |
//! | `quantify` | Count items matching predicate |

use prism_core::Value;
use std::collections::HashSet;
use std::collections::VecDeque;

// =============================================================================
// Flatten
// =============================================================================

/// Flattens one level of nesting from an iterator of Vec<Value>.
///
/// Equivalent to Python's `itertools.chain.from_iterable(iterable)`.
///
/// # Performance
///
/// - O(1) per element (amortized)
/// - Lazy: processes one inner Vec at a time
#[derive(Debug, Clone)]
pub struct Flatten<I> {
    outer: I,
    inner: std::vec::IntoIter<Value>,
}

impl<I> Flatten<I>
where
    I: Iterator<Item = Vec<Value>>,
{
    /// Create a new flatten iterator.
    #[inline]
    pub fn new(iter: I) -> Self {
        Self {
            outer: iter,
            inner: Vec::new().into_iter(),
        }
    }
}

impl<I> Iterator for Flatten<I>
where
    I: Iterator<Item = Vec<Value>>,
{
    type Item = Value;

    #[inline]
    fn next(&mut self) -> Option<Value> {
        loop {
            if let Some(val) = self.inner.next() {
                return Some(val);
            }
            let next_vec = self.outer.next()?;
            self.inner = next_vec.into_iter();
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let (_, outer_hi) = self.outer.size_hint();
        let inner_remaining = self.inner.len();
        (
            inner_remaining,
            outer_hi.map(|h| h * 1024 + inner_remaining),
        ) // rough upper bound
    }
}

impl<I> std::iter::FusedIterator for Flatten<I> where I: Iterator<Item = Vec<Value>> {}

// =============================================================================
// UniqueEverseen
// =============================================================================

/// Yields unique elements, preserving first-seen order.
///
/// Equivalent to Python's `more_itertools.unique_everseen()`.
///
/// Uses a `HashSet<u64>` of value bit patterns for O(1) lookup.
///
/// # Performance
///
/// - O(1) amortized per `next()` (hash probe)
/// - O(k) space where k = number of unique elements
///
/// Note: Uses bit-pattern hashing, which works correctly for interned
/// strings, ints, bools, and None. For floats, different NaN bit patterns
/// may not deduplicate (matching Python's behavior for unhashable types).
#[derive(Debug, Clone)]
pub struct UniqueEverseen<I> {
    iter: I,
    seen: HashSet<u64>,
}

impl<I> UniqueEverseen<I>
where
    I: Iterator<Item = Value>,
{
    /// Create a new unique_everseen iterator.
    #[inline]
    pub fn new(iter: I) -> Self {
        let (hint, _) = iter.size_hint();
        Self {
            iter,
            seen: HashSet::with_capacity(hint.min(1024)),
        }
    }

    /// Create with a pre-sized capacity.
    #[inline]
    pub fn with_capacity(iter: I, cap: usize) -> Self {
        Self {
            iter,
            seen: HashSet::with_capacity(cap),
        }
    }
}

impl<I> Iterator for UniqueEverseen<I>
where
    I: Iterator<Item = Value>,
{
    type Item = Value;

    #[inline]
    fn next(&mut self) -> Option<Value> {
        loop {
            let val = self.iter.next()?;
            let bits = val.to_bits();
            if self.seen.insert(bits) {
                return Some(val);
            }
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        (0, self.iter.size_hint().1)
    }
}

impl<I> std::iter::FusedIterator for UniqueEverseen<I> where I: Iterator<Item = Value> {}

// =============================================================================
// UniqueJustseen
// =============================================================================

/// Removes consecutive duplicate elements.
///
/// Equivalent to the `unique_justseen` recipe from Python docs.
///
/// # Performance
///
/// - O(1) per `next()` — one comparison
/// - O(1) space — stores one previous value
#[derive(Debug, Clone)]
pub struct UniqueJustseen<I> {
    iter: I,
    prev_bits: Option<u64>,
}

impl<I> UniqueJustseen<I>
where
    I: Iterator<Item = Value>,
{
    /// Create a new unique_justseen iterator.
    #[inline]
    pub fn new(iter: I) -> Self {
        Self {
            iter,
            prev_bits: None,
        }
    }
}

impl<I> Iterator for UniqueJustseen<I>
where
    I: Iterator<Item = Value>,
{
    type Item = Value;

    #[inline]
    fn next(&mut self) -> Option<Value> {
        loop {
            let val = self.iter.next()?;
            let bits = val.to_bits();
            if self.prev_bits != Some(bits) {
                self.prev_bits = Some(bits);
                return Some(val);
            }
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let (lo, hi) = self.iter.size_hint();
        // At least 1 if source has elements, at most same as source
        (lo.min(1), hi)
    }
}

impl<I> std::iter::FusedIterator for UniqueJustseen<I> where I: Iterator<Item = Value> {}

// =============================================================================
// SlidingWindow
// =============================================================================

/// Generic sliding window of size `n`.
///
/// Equivalent to the `sliding_window` recipe from Python 3.12+ itertools.
///
/// `sliding_window([1,2,3,4,5], 3)` → `[1,2,3], [2,3,4], [3,4,5]`
///
/// # Performance
///
/// - O(1) per `next()` after initial fill (VecDeque push/pop)
/// - O(n) space for the window
#[derive(Debug, Clone)]
pub struct SlidingWindow<I> {
    iter: I,
    window: VecDeque<Value>,
    window_size: usize,
    filled: bool,
}

impl<I> SlidingWindow<I>
where
    I: Iterator<Item = Value>,
{
    /// Create a new sliding window of size `n`.
    ///
    /// # Panics
    ///
    /// Panics if `window_size` is 0.
    pub fn new(iter: I, window_size: usize) -> Self {
        assert!(window_size > 0, "sliding_window size must be >= 1");
        Self {
            iter,
            window: VecDeque::with_capacity(window_size),
            window_size,
            filled: false,
        }
    }
}

impl<I> Iterator for SlidingWindow<I>
where
    I: Iterator<Item = Value>,
{
    type Item = Vec<Value>;

    fn next(&mut self) -> Option<Vec<Value>> {
        if !self.filled {
            // Fill the initial window
            while self.window.len() < self.window_size {
                match self.iter.next() {
                    Some(val) => self.window.push_back(val),
                    None => return None, // Not enough elements
                }
            }
            self.filled = true;
            return Some(self.window.iter().cloned().collect());
        }

        // Slide: pop front, push back
        let next_val = self.iter.next()?;
        self.window.pop_front();
        self.window.push_back(next_val);
        Some(self.window.iter().cloned().collect())
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let (lo, hi) = self.iter.size_hint();
        if !self.filled {
            (0, hi)
        } else {
            (lo, hi.map(|h| h + 1))
        }
    }
}

impl<I> std::iter::FusedIterator for SlidingWindow<I> where I: Iterator<Item = Value> {}

// =============================================================================
// RoundRobin
// =============================================================================

/// Interleaves elements from multiple iterators.
///
/// Equivalent to the `roundrobin` recipe from Python docs.
///
/// `roundrobin([1,2,3], [4,5], [6,7,8,9])` → `1, 4, 6, 2, 5, 7, 3, 8, 9`
///
/// When an iterator is exhausted, it is removed; the remaining iterators
/// continue to contribute in round-robin order.
///
/// # Performance
///
/// - O(1) per element (amortized)
/// - O(k) space where k = number of active iterators
#[derive(Debug)]
pub struct RoundRobin {
    iterators: VecDeque<std::vec::IntoIter<Value>>,
}

impl RoundRobin {
    /// Create from multiple Vec<Value> iterables.
    pub fn new(iterables: Vec<Vec<Value>>) -> Self {
        let mut iters = VecDeque::with_capacity(iterables.len());
        for iterable in iterables {
            if !iterable.is_empty() {
                iters.push_back(iterable.into_iter());
            }
        }
        Self { iterators: iters }
    }
}

impl Iterator for RoundRobin {
    type Item = Value;

    #[inline]
    fn next(&mut self) -> Option<Value> {
        loop {
            let mut iter = self.iterators.pop_front()?;
            if let Some(val) = iter.next() {
                self.iterators.push_back(iter);
                return Some(val);
            }
            // Iterator exhausted, drop it and try next
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let total: usize = self.iterators.iter().map(|i| i.len()).sum();
        (total, Some(total))
    }
}

impl ExactSizeIterator for RoundRobin {
    fn len(&self) -> usize {
        self.iterators.iter().map(|i| i.len()).sum()
    }
}

impl std::iter::FusedIterator for RoundRobin {}

// =============================================================================
// Accumulate
// =============================================================================

/// Running accumulation / scan with a binary function.
///
/// Equivalent to Python's `itertools.accumulate(iterable, func, *, initial=None)`.
///
/// The default function is addition. If `initial` is provided, it is placed
/// before the first element.
///
/// # Performance
///
/// - O(1) per `next()` — single function call
/// - O(1) space — stores the running accumulator
#[derive(Clone)]
pub struct Accumulate<I, F> {
    iter: I,
    func: F,
    acc: Option<Value>,
    yielded_initial: bool,
    initial: Option<Value>,
}

impl<I, F> Accumulate<I, F>
where
    I: Iterator<Item = Value>,
    F: FnMut(&Value, &Value) -> Value,
{
    /// Create accumulate with a binary function.
    #[inline]
    pub fn new(iter: I, func: F) -> Self {
        Self {
            iter,
            func,
            acc: None,
            yielded_initial: false,
            initial: None,
        }
    }

    /// Create accumulate with an initial value.
    #[inline]
    pub fn with_initial(iter: I, func: F, initial: Value) -> Self {
        Self {
            iter,
            func,
            acc: None,
            yielded_initial: false,
            initial: Some(initial),
        }
    }
}

impl<I, F> Iterator for Accumulate<I, F>
where
    I: Iterator<Item = Value>,
    F: FnMut(&Value, &Value) -> Value,
{
    type Item = Value;

    #[inline]
    fn next(&mut self) -> Option<Value> {
        // Yield initial value first
        if !self.yielded_initial {
            self.yielded_initial = true;
            if let Some(initial) = self.initial.take() {
                self.acc = Some(initial.clone());
                return Some(initial);
            }
        }

        let val = self.iter.next()?;
        let result = match self.acc.take() {
            Some(acc) => (self.func)(&acc, &val),
            None => val, // first element (no initial)
        };
        self.acc = Some(result.clone());
        Some(result)
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let (lo, hi) = self.iter.size_hint();
        let extra = if !self.yielded_initial && self.initial.is_some() {
            1
        } else {
            0
        };
        (lo + extra, hi.map(|h| h + extra))
    }
}

// =============================================================================
// Standalone functions
// =============================================================================

/// Split elements by predicate into two Vecs.
///
/// `partition(pred, iterable)` → `(falses, trues)`
///
/// Returns `(Vec<Value>, Vec<Value>)` where the first contains elements
/// where the predicate is false, and the second where it's true.
///
/// # Performance
///
/// - O(n) time, O(n) space
pub fn partition<I, P>(iterable: I, mut pred: P) -> (Vec<Value>, Vec<Value>)
where
    I: Iterator<Item = Value>,
    P: FnMut(&Value) -> bool,
{
    let (hint, _) = iterable.size_hint();
    let half = hint / 2;
    let mut falses = Vec::with_capacity(half);
    let mut trues = Vec::with_capacity(half);

    for item in iterable {
        if pred(&item) {
            trues.push(item);
        } else {
            falses.push(item);
        }
    }

    (falses, trues)
}

/// Count how many items in the iterable match the predicate.
///
/// Equivalent to `sum(1 for x in iterable if pred(x))`.
///
/// # Performance
///
/// - O(n) time, O(1) space
#[inline]
pub fn quantify<I, P>(iterable: I, mut pred: P) -> usize
where
    I: Iterator<Item = Value>,
    P: FnMut(&Value) -> bool,
{
    iterable.filter(|v| pred(v)).count()
}

/// Consume an iterator and return the first `n` and last `n` elements.
///
/// Useful for preview/summary of large iterators.
///
/// # Performance
///
/// - O(total) time, O(n) space for the tail buffer
pub fn head_tail<I>(iterable: I, n: usize) -> (Vec<Value>, Vec<Value>)
where
    I: Iterator<Item = Value>,
{
    if n == 0 {
        // Consume the iterator but return nothing
        let _ = iterable.count();
        return (Vec::new(), Vec::new());
    }

    let mut head = Vec::with_capacity(n);
    let mut tail: VecDeque<Value> = VecDeque::with_capacity(n);
    let mut count = 0;

    for item in iterable {
        if count < n {
            head.push(item.clone());
        }
        if tail.len() >= n {
            tail.pop_front();
        }
        tail.push_back(item);
        count += 1;
    }

    (head, tail.into_iter().collect())
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn int(i: i64) -> Value {
        Value::int_unchecked(i)
    }

    fn vals(ints: &[i64]) -> Vec<Value> {
        ints.iter().map(|&i| int(i)).collect()
    }

    fn to_ints(v: Vec<Value>) -> Vec<i64> {
        v.into_iter().filter_map(|v| v.as_int()).collect()
    }

    // =========================================================================
    // Flatten tests
    // =========================================================================

    #[test]
    fn test_flatten_basic() {
        let data = vec![vals(&[1, 2]), vals(&[3, 4]), vals(&[5])];
        let result = to_ints(Flatten::new(data.into_iter()).collect());
        assert_eq!(result, vec![1, 2, 3, 4, 5]);
    }

    #[test]
    fn test_flatten_empty_inner() {
        let data: Vec<Vec<Value>> = vec![vec![], vals(&[1]), vec![], vals(&[2])];
        let result = to_ints(Flatten::new(data.into_iter()).collect());
        assert_eq!(result, vec![1, 2]);
    }

    #[test]
    fn test_flatten_empty_outer() {
        let data: Vec<Vec<Value>> = vec![];
        let result: Vec<Value> = Flatten::new(data.into_iter()).collect();
        assert!(result.is_empty());
    }

    #[test]
    fn test_flatten_all_empty_inner() {
        let data: Vec<Vec<Value>> = vec![vec![], vec![], vec![]];
        let result: Vec<Value> = Flatten::new(data.into_iter()).collect();
        assert!(result.is_empty());
    }

    #[test]
    fn test_flatten_single() {
        let data = vec![vals(&[1, 2, 3])];
        let result = to_ints(Flatten::new(data.into_iter()).collect());
        assert_eq!(result, vec![1, 2, 3]);
    }

    #[test]
    fn test_flatten_stress() {
        let data: Vec<Vec<Value>> = (0..100).map(|i| vec![int(i)]).collect();
        let result = to_ints(Flatten::new(data.into_iter()).collect());
        assert_eq!(result.len(), 100);
    }

    // =========================================================================
    // UniqueEverseen tests
    // =========================================================================

    #[test]
    fn test_unique_everseen_basic() {
        let data = vals(&[1, 2, 3, 1, 2, 4, 3, 5]);
        let result = to_ints(UniqueEverseen::new(data.into_iter()).collect());
        assert_eq!(result, vec![1, 2, 3, 4, 5]);
    }

    #[test]
    fn test_unique_everseen_all_unique() {
        let data = vals(&[1, 2, 3, 4, 5]);
        let result = to_ints(UniqueEverseen::new(data.into_iter()).collect());
        assert_eq!(result, vec![1, 2, 3, 4, 5]);
    }

    #[test]
    fn test_unique_everseen_all_same() {
        let data = vals(&[1, 1, 1, 1]);
        let result = to_ints(UniqueEverseen::new(data.into_iter()).collect());
        assert_eq!(result, vec![1]);
    }

    #[test]
    fn test_unique_everseen_empty() {
        let data: Vec<Value> = vec![];
        let result: Vec<Value> = UniqueEverseen::new(data.into_iter()).collect();
        assert!(result.is_empty());
    }

    #[test]
    fn test_unique_everseen_preserves_first_seen_order() {
        let data = vals(&[5, 3, 1, 3, 5, 2, 1, 4]);
        let result = to_ints(UniqueEverseen::new(data.into_iter()).collect());
        assert_eq!(result, vec![5, 3, 1, 2, 4]);
    }

    #[test]
    fn test_unique_everseen_with_none() {
        let data = vec![Value::none(), int(1), Value::none(), int(2)];
        let result: Vec<Value> = UniqueEverseen::new(data.into_iter()).collect();
        assert_eq!(result.len(), 3); // None, 1, 2
    }

    #[test]
    fn test_unique_everseen_stress() {
        // 10k elements with lots of duplicates
        let data: Vec<Value> = (0..10_000).map(|i| int(i % 100)).collect();
        let result = to_ints(UniqueEverseen::new(data.into_iter()).collect());
        assert_eq!(result.len(), 100);
    }

    // =========================================================================
    // UniqueJustseen tests
    // =========================================================================

    #[test]
    fn test_unique_justseen_basic() {
        let data = vals(&[1, 1, 2, 2, 2, 3, 1, 1]);
        let result = to_ints(UniqueJustseen::new(data.into_iter()).collect());
        assert_eq!(result, vec![1, 2, 3, 1]); // note: 1 appears twice (non-consecutive)
    }

    #[test]
    fn test_unique_justseen_all_unique() {
        let data = vals(&[1, 2, 3, 4, 5]);
        let result = to_ints(UniqueJustseen::new(data.into_iter()).collect());
        assert_eq!(result, vec![1, 2, 3, 4, 5]);
    }

    #[test]
    fn test_unique_justseen_all_same() {
        let data = vals(&[7, 7, 7, 7, 7]);
        let result = to_ints(UniqueJustseen::new(data.into_iter()).collect());
        assert_eq!(result, vec![7]);
    }

    #[test]
    fn test_unique_justseen_empty() {
        let data: Vec<Value> = vec![];
        let result: Vec<Value> = UniqueJustseen::new(data.into_iter()).collect();
        assert!(result.is_empty());
    }

    #[test]
    fn test_unique_justseen_single() {
        let data = vals(&[42]);
        let result = to_ints(UniqueJustseen::new(data.into_iter()).collect());
        assert_eq!(result, vec![42]);
    }

    #[test]
    fn test_unique_justseen_alternating() {
        let data = vals(&[1, 2, 1, 2, 1, 2]);
        let result = to_ints(UniqueJustseen::new(data.into_iter()).collect());
        assert_eq!(result, vec![1, 2, 1, 2, 1, 2]); // no consecutive dupes
    }

    // =========================================================================
    // SlidingWindow tests
    // =========================================================================

    #[test]
    fn test_sliding_window_size_3() {
        let data = vals(&[1, 2, 3, 4, 5]);
        let result: Vec<Vec<i64>> = SlidingWindow::new(data.into_iter(), 3)
            .map(|w| to_ints(w))
            .collect();
        assert_eq!(result, vec![vec![1, 2, 3], vec![2, 3, 4], vec![3, 4, 5]]);
    }

    #[test]
    fn test_sliding_window_size_1() {
        let data = vals(&[1, 2, 3]);
        let result: Vec<Vec<i64>> = SlidingWindow::new(data.into_iter(), 1)
            .map(|w| to_ints(w))
            .collect();
        assert_eq!(result, vec![vec![1], vec![2], vec![3]]);
    }

    #[test]
    fn test_sliding_window_size_equals_length() {
        let data = vals(&[1, 2, 3]);
        let result: Vec<Vec<i64>> = SlidingWindow::new(data.into_iter(), 3)
            .map(|w| to_ints(w))
            .collect();
        assert_eq!(result, vec![vec![1, 2, 3]]);
    }

    #[test]
    fn test_sliding_window_size_exceeds_length() {
        let data = vals(&[1, 2]);
        let result: Vec<Vec<Value>> = SlidingWindow::new(data.into_iter(), 5).collect();
        assert!(result.is_empty()); // not enough elements
    }

    #[test]
    fn test_sliding_window_empty() {
        let data: Vec<Value> = vec![];
        let result: Vec<Vec<Value>> = SlidingWindow::new(data.into_iter(), 3).collect();
        assert!(result.is_empty());
    }

    #[test]
    #[should_panic(expected = "size must be >= 1")]
    fn test_sliding_window_zero_panics() {
        let data = vals(&[1]);
        let _ = SlidingWindow::new(data.into_iter(), 0);
    }

    #[test]
    fn test_sliding_window_size_2_matches_pairwise() {
        // sliding_window(data, 2) should produce same as pairwise
        let data1 = vals(&[1, 2, 3, 4, 5]);
        let data2 = data1.clone();

        let sw: Vec<Vec<i64>> = SlidingWindow::new(data1.into_iter(), 2)
            .map(|w| to_ints(w))
            .collect();

        let pw: Vec<Vec<i64>> =
            crate::stdlib::itertools::grouping::Pairwise::new(data2.into_iter())
                .map(|(a, b)| vec![a.as_int().unwrap(), b.as_int().unwrap()])
                .collect();

        assert_eq!(sw, pw);
    }

    #[test]
    fn test_sliding_window_stress() {
        let data: Vec<Value> = (0..1000).map(|i| int(i)).collect();
        let result: Vec<Vec<Value>> = SlidingWindow::new(data.into_iter(), 10).collect();
        assert_eq!(result.len(), 991); // 1000 - 10 + 1
    }

    // =========================================================================
    // RoundRobin tests
    // =========================================================================

    #[test]
    fn test_roundrobin_equal_length() {
        let iters = vec![vals(&[1, 2, 3]), vals(&[4, 5, 6])];
        let result = to_ints(RoundRobin::new(iters).collect());
        assert_eq!(result, vec![1, 4, 2, 5, 3, 6]);
    }

    #[test]
    fn test_roundrobin_unequal_length() {
        let iters = vec![vals(&[1, 2, 3]), vals(&[4, 5]), vals(&[6, 7, 8, 9])];
        let result = to_ints(RoundRobin::new(iters).collect());
        // 1, 4, 6, 2, 5, 7, 3, 8, 9
        assert_eq!(result, vec![1, 4, 6, 2, 5, 7, 3, 8, 9]);
    }

    #[test]
    fn test_roundrobin_single_iter() {
        let iters = vec![vals(&[1, 2, 3])];
        let result = to_ints(RoundRobin::new(iters).collect());
        assert_eq!(result, vec![1, 2, 3]);
    }

    #[test]
    fn test_roundrobin_empty() {
        let iters: Vec<Vec<Value>> = vec![];
        let result: Vec<Value> = RoundRobin::new(iters).collect();
        assert!(result.is_empty());
    }

    #[test]
    fn test_roundrobin_all_empty() {
        let iters: Vec<Vec<Value>> = vec![vec![], vec![], vec![]];
        let result: Vec<Value> = RoundRobin::new(iters).collect();
        assert!(result.is_empty());
    }

    #[test]
    fn test_roundrobin_one_empty() {
        let iters = vec![vals(&[1, 2]), vec![], vals(&[3, 4])];
        let result = to_ints(RoundRobin::new(iters).collect());
        assert_eq!(result, vec![1, 3, 2, 4]);
    }

    #[test]
    fn test_roundrobin_exact_size() {
        let iters = vec![vals(&[1, 2, 3]), vals(&[4, 5])];
        let rr = RoundRobin::new(iters);
        assert_eq!(rr.len(), 5);
    }

    // =========================================================================
    // Accumulate tests
    // =========================================================================

    #[test]
    fn test_accumulate_sum() {
        let data = vals(&[1, 2, 3, 4, 5]);
        let result = to_ints(
            Accumulate::new(data.into_iter(), |a, b| {
                int(a.as_int().unwrap() + b.as_int().unwrap())
            })
            .collect(),
        );
        assert_eq!(result, vec![1, 3, 6, 10, 15]);
    }

    #[test]
    fn test_accumulate_product() {
        let data = vals(&[1, 2, 3, 4, 5]);
        let result = to_ints(
            Accumulate::new(data.into_iter(), |a, b| {
                int(a.as_int().unwrap() * b.as_int().unwrap())
            })
            .collect(),
        );
        assert_eq!(result, vec![1, 2, 6, 24, 120]);
    }

    #[test]
    fn test_accumulate_with_initial() {
        let data = vals(&[1, 2, 3]);
        let result = to_ints(
            Accumulate::with_initial(
                data.into_iter(),
                |a, b| int(a.as_int().unwrap() + b.as_int().unwrap()),
                int(100),
            )
            .collect(),
        );
        assert_eq!(result, vec![100, 101, 103, 106]);
    }

    #[test]
    fn test_accumulate_empty() {
        let data: Vec<Value> = vec![];
        let result: Vec<Value> = Accumulate::new(data.into_iter(), |a, b| {
            int(a.as_int().unwrap() + b.as_int().unwrap())
        })
        .collect();
        assert!(result.is_empty());
    }

    #[test]
    fn test_accumulate_single() {
        let data = vals(&[42]);
        let result = to_ints(
            Accumulate::new(data.into_iter(), |a, b| {
                int(a.as_int().unwrap() + b.as_int().unwrap())
            })
            .collect(),
        );
        assert_eq!(result, vec![42]);
    }

    #[test]
    fn test_accumulate_max() {
        let data = vals(&[3, 1, 4, 1, 5, 9, 2, 6]);
        let result = to_ints(
            Accumulate::new(data.into_iter(), |a, b| {
                int(a.as_int().unwrap().max(b.as_int().unwrap()))
            })
            .collect(),
        );
        assert_eq!(result, vec![3, 3, 4, 4, 5, 9, 9, 9]);
    }

    #[test]
    fn test_accumulate_with_initial_empty() {
        let data: Vec<Value> = vec![];
        let result = to_ints(
            Accumulate::with_initial(
                data.into_iter(),
                |a, b| int(a.as_int().unwrap() + b.as_int().unwrap()),
                int(0),
            )
            .collect(),
        );
        assert_eq!(result, vec![0]); // just the initial
    }

    // =========================================================================
    // partition tests
    // =========================================================================

    #[test]
    fn test_partition_basic() {
        let data = vals(&[1, 2, 3, 4, 5, 6]);
        let (odds, evens) = partition(data.into_iter(), |v| v.as_int().unwrap() % 2 == 0);
        assert_eq!(to_ints(odds), vec![1, 3, 5]); // falses (not even)
        assert_eq!(to_ints(evens), vec![2, 4, 6]); // trues (even)
    }

    #[test]
    fn test_partition_all_true() {
        let data = vals(&[2, 4, 6]);
        let (falses, trues) = partition(data.into_iter(), |v| v.as_int().unwrap() % 2 == 0);
        assert!(falses.is_empty());
        assert_eq!(to_ints(trues), vec![2, 4, 6]);
    }

    #[test]
    fn test_partition_all_false() {
        let data = vals(&[1, 3, 5]);
        let (falses, trues) = partition(data.into_iter(), |v| v.as_int().unwrap() % 2 == 0);
        assert_eq!(to_ints(falses), vec![1, 3, 5]);
        assert!(trues.is_empty());
    }

    #[test]
    fn test_partition_empty() {
        let data: Vec<Value> = vec![];
        let (falses, trues) = partition(data.into_iter(), |_| true);
        assert!(falses.is_empty());
        assert!(trues.is_empty());
    }

    #[test]
    fn test_partition_preserves_order() {
        let data = vals(&[5, 3, 1, 4, 2]);
        let (small, big) = partition(data.into_iter(), |v| v.as_int().unwrap() > 3);
        assert_eq!(to_ints(small), vec![3, 1, 2]);
        assert_eq!(to_ints(big), vec![5, 4]);
    }

    // =========================================================================
    // quantify tests
    // =========================================================================

    #[test]
    fn test_quantify_basic() {
        let data = vals(&[1, 2, 3, 4, 5, 6]);
        let count = quantify(data.into_iter(), |v| v.as_int().unwrap() % 2 == 0);
        assert_eq!(count, 3);
    }

    #[test]
    fn test_quantify_all_true() {
        let data = vals(&[2, 4, 6]);
        let count = quantify(data.into_iter(), |v| v.as_int().unwrap() % 2 == 0);
        assert_eq!(count, 3);
    }

    #[test]
    fn test_quantify_none_true() {
        let data = vals(&[1, 3, 5]);
        let count = quantify(data.into_iter(), |v| v.as_int().unwrap() % 2 == 0);
        assert_eq!(count, 0);
    }

    #[test]
    fn test_quantify_empty() {
        let data: Vec<Value> = vec![];
        let count = quantify(data.into_iter(), |_| true);
        assert_eq!(count, 0);
    }

    // =========================================================================
    // head_tail tests
    // =========================================================================

    #[test]
    fn test_head_tail_basic() {
        let data: Vec<Value> = (0..10).map(|i| int(i)).collect();
        let (head, tail) = head_tail(data.into_iter(), 3);
        assert_eq!(to_ints(head), vec![0, 1, 2]);
        assert_eq!(to_ints(tail), vec![7, 8, 9]);
    }

    #[test]
    fn test_head_tail_short_input() {
        let data = vals(&[1, 2]);
        let (head, tail) = head_tail(data.into_iter(), 5);
        assert_eq!(to_ints(head), vec![1, 2]);
        assert_eq!(to_ints(tail), vec![1, 2]);
    }

    #[test]
    fn test_head_tail_empty() {
        let data: Vec<Value> = vec![];
        let (head, tail) = head_tail(data.into_iter(), 3);
        assert!(head.is_empty());
        assert!(tail.is_empty());
    }

    #[test]
    fn test_head_tail_n_is_zero() {
        let data = vals(&[1, 2, 3]);
        let (head, tail) = head_tail(data.into_iter(), 0);
        assert!(head.is_empty());
        assert!(tail.is_empty());
    }

    // =========================================================================
    // Stress tests
    // =========================================================================

    #[test]
    fn test_unique_everseen_1m_small_alphabet() {
        // 100k elements from alphabet of 50
        let data: Vec<Value> = (0..100_000).map(|i| int(i % 50)).collect();
        let result = to_ints(UniqueEverseen::new(data.into_iter()).collect());
        assert_eq!(result.len(), 50);
    }

    #[test]
    fn test_accumulate_stress() {
        let data: Vec<Value> = (1..=1000).map(|i| int(i)).collect();
        let result = to_ints(
            Accumulate::new(data.into_iter(), |a, b| {
                int(a.as_int().unwrap() + b.as_int().unwrap())
            })
            .collect(),
        );
        assert_eq!(result.len(), 1000);
        // Last element should be sum(1..=1000) = 500500
        assert_eq!(*result.last().unwrap(), 500_500);
    }
}
