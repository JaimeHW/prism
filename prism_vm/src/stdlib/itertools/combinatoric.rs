//! Combinatoric iterator constructors.
//!
//! Provides `product`, `permutations`, `combinations`, and
//! `combinations_with_replacement` — producing all possible arrangements
//! from input pools.
//!
//! # Performance Characteristics
//!
//! | Iterator | Output Size | Space |
//! |----------|-------------|-------|
//! | `Product` (n pools, each k) | k^n | O(n) indices + O(k*n) pools |
//! | `Permutations(n, r)` | n!/(n-r)! | O(n) indices + O(n) pool |
//! | `Combinations(n, r)` | C(n,r) | O(r) indices + O(n) pool |
//! | `CombWithRepl(n, r)` | C(n+r-1,r) | O(r) indices + O(n) pool |
//!
//! All use SmallVec for index arrays to avoid heap allocation for small `r`.

use prism_core::Value;
use smallvec::SmallVec;

/// Stack-allocated index threshold. For r <= 8, indices live on the stack.
const SMALL_INDEX: usize = 8;

type IndexVec = SmallVec<[usize; SMALL_INDEX]>;

// =============================================================================
// Product
// =============================================================================

/// Cartesian product of input iterables.
///
/// Equivalent to Python's `itertools.product(*iterables, repeat=1)`.
///
/// # Algorithm
///
/// Uses odometer-style index advancement: the rightmost index increments
/// first, cascading left on overflow. This produces lexicographic order
/// matching CPython exactly.
///
/// # Performance
///
/// - O(1) per `next()` (amortized — index cascade is O(n) worst case
///   but averaged over all outputs is O(1))
/// - O(n) space for the index vector where n = number of pools
#[derive(Debug, Clone)]
pub struct Product {
    /// The pools of values to combine.
    pools: Vec<Vec<Value>>,
    /// Current index into each pool.
    indices: IndexVec,
    /// Whether we've finished.
    done: bool,
    /// Whether this is the first call to next().
    first: bool,
}

impl Product {
    /// Create a Cartesian product from multiple pools.
    pub fn new(pools: Vec<Vec<Value>>) -> Self {
        // If any pool is empty, the product is empty
        let done = pools.iter().any(|p| p.is_empty());
        let n = pools.len();
        let mut indices = IndexVec::with_capacity(n);
        indices.resize(n, 0);

        Self {
            pools,
            indices,
            done,
            first: true,
        }
    }

    /// Create with repeat (Python's `repeat` parameter).
    ///
    /// `product(pool, repeat=3)` = `product(pool, pool, pool)`
    pub fn with_repeat(pool: Vec<Value>, repeat: usize) -> Self {
        let pools = vec![pool; repeat];
        Self::new(pools)
    }

    /// Get the total number of elements in the product.
    pub fn total_size(&self) -> usize {
        if self.pools.is_empty() {
            return 1; // empty product yields one empty tuple
        }
        self.pools.iter().map(|p| p.len()).product()
    }

    /// Build the current tuple from indices.
    #[inline]
    fn current_tuple(&self) -> Vec<Value> {
        self.indices
            .iter()
            .zip(self.pools.iter())
            .map(|(&idx, pool)| pool[idx].clone())
            .collect()
    }

    /// Advance the odometer indices (rightmost first).
    #[inline]
    fn advance(&mut self) -> bool {
        // Increment from the right, cascading left on overflow
        for i in (0..self.indices.len()).rev() {
            self.indices[i] += 1;
            if self.indices[i] < self.pools[i].len() {
                return true;
            }
            self.indices[i] = 0;
        }
        false // all indices wrapped around → done
    }
}

impl Iterator for Product {
    type Item = Vec<Value>;

    #[inline]
    fn next(&mut self) -> Option<Vec<Value>> {
        if self.done {
            return None;
        }

        if self.first {
            self.first = false;
            if self.pools.is_empty() {
                self.done = true;
                return Some(Vec::new()); // single empty tuple
            }
            return Some(self.current_tuple());
        }

        if self.advance() {
            Some(self.current_tuple())
        } else {
            self.done = true;
            None
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        if self.done {
            (0, Some(0))
        } else {
            let total = self.total_size();
            (total, Some(total))
        }
    }
}

impl std::iter::FusedIterator for Product {}

// =============================================================================
// Permutations
// =============================================================================

/// Successive r-length permutations of elements from the pool.
///
/// Equivalent to Python's `itertools.permutations(iterable, r=None)`.
///
/// # Algorithm
///
/// Uses the lexicographic permutation generation algorithm. Indices cycle through
/// all possible orderings without repetition.
///
/// # Performance
///
/// - O(r) per `next()` for tuple construction
/// - O(n) space for pool + indices + cycles
#[derive(Debug, Clone)]
pub struct Permutations {
    pool: Vec<Value>,
    indices: Vec<usize>,
    cycles: Vec<usize>,
    r: usize,
    first: bool,
    done: bool,
}

impl Permutations {
    /// Create permutations of length `r` from the pool.
    pub fn new(pool: Vec<Value>, r: usize) -> Self {
        let n = pool.len();
        if r > n {
            return Self {
                pool,
                indices: Vec::new(),
                cycles: Vec::new(),
                r,
                first: true,
                done: true,
            };
        }

        let indices: Vec<usize> = (0..n).collect();
        let cycles: Vec<usize> = (n - r + 1..=n).rev().collect();

        Self {
            pool,
            indices,
            cycles,
            r,
            first: true,
            done: false,
        }
    }

    /// Create full-length permutations.
    pub fn full(pool: Vec<Value>) -> Self {
        let r = pool.len();
        Self::new(pool, r)
    }

    /// Get the current permutation tuple.
    #[inline]
    fn current_tuple(&self) -> Vec<Value> {
        self.indices[..self.r]
            .iter()
            .map(|&i| self.pool[i].clone())
            .collect()
    }
}

impl Iterator for Permutations {
    type Item = Vec<Value>;

    fn next(&mut self) -> Option<Vec<Value>> {
        if self.done {
            return None;
        }

        if self.first {
            self.first = false;
            return Some(self.current_tuple());
        }

        let n = self.pool.len();

        // Advance using the cycles algorithm (identical to CPython)
        for i in (0..self.r).rev() {
            self.cycles[i] -= 1;
            if self.cycles[i] == 0 {
                // Rotate indices[i..n] left by 1
                let saved = self.indices[i];
                for j in i..n - 1 {
                    self.indices[j] = self.indices[j + 1];
                }
                self.indices[n - 1] = saved;
                self.cycles[i] = n - i;
            } else {
                let j = n - self.cycles[i];
                self.indices.swap(i, j);
                return Some(self.current_tuple());
            }
        }

        self.done = true;
        None
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        if self.done {
            return (0, Some(0));
        }
        // n! / (n-r)! but we don't track position, so approximate
        (0, None)
    }
}

impl std::iter::FusedIterator for Permutations {}

// =============================================================================
// Combinations
// =============================================================================

/// Successive r-length combinations of elements from the pool.
///
/// Equivalent to Python's `itertools.combinations(iterable, r)`.
///
/// # Algorithm
///
/// Index-based lexicographic generation. Each combination is represented
/// by `r` indices `i_0 < i_1 < ... < i_{r-1}` into the pool.
///
/// # Performance
///
/// - O(r) per `next()` for advance + tuple construction
/// - O(r) space for indices (SmallVec-backed)
#[derive(Debug, Clone)]
pub struct Combinations {
    pool: Vec<Value>,
    indices: IndexVec,
    r: usize,
    first: bool,
    done: bool,
}

impl Combinations {
    /// Create r-length combinations from the pool.
    pub fn new(pool: Vec<Value>, r: usize) -> Self {
        let n = pool.len();
        if r > n {
            return Self {
                pool,
                indices: IndexVec::new(),
                r,
                first: true,
                done: true,
            };
        }

        let mut indices = IndexVec::with_capacity(r);
        for i in 0..r {
            indices.push(i);
        }

        Self {
            pool,
            indices,
            r,
            first: true,
            done: false,
        }
    }

    /// Get the current combination tuple.
    #[inline]
    fn current_tuple(&self) -> Vec<Value> {
        self.indices.iter().map(|&i| self.pool[i].clone()).collect()
    }
}

impl Iterator for Combinations {
    type Item = Vec<Value>;

    fn next(&mut self) -> Option<Vec<Value>> {
        if self.done {
            return None;
        }

        if self.first {
            self.first = false;
            if self.r == 0 {
                self.done = true;
                return Some(Vec::new()); // single empty combination
            }
            return Some(self.current_tuple());
        }

        let n = self.pool.len();

        // Find the rightmost index that can be incremented
        let mut i = self.r;
        loop {
            if i == 0 {
                self.done = true;
                return None;
            }
            i -= 1;
            if self.indices[i] != i + n - self.r {
                break;
            }
        }

        // Increment it and reset all indices to its right
        self.indices[i] += 1;
        for j in (i + 1)..self.r {
            self.indices[j] = self.indices[j - 1] + 1;
        }

        Some(self.current_tuple())
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        if self.done { (0, Some(0)) } else { (0, None) }
    }
}

impl std::iter::FusedIterator for Combinations {}

// =============================================================================
// CombinationsWithReplacement
// =============================================================================

/// Successive r-length combinations with replacement (repetition allowed).
///
/// Equivalent to Python's `itertools.combinations_with_replacement(iterable, r)`.
///
/// # Algorithm
///
/// Uses rising index vector: indices satisfy `i_0 <= i_1 <= ... <= i_{r-1}`.
///
/// # Performance
///
/// - O(r) per `next()` for advance + tuple construction
/// - O(r) space for indices (SmallVec-backed)
#[derive(Debug, Clone)]
pub struct CombinationsWithReplacement {
    pool: Vec<Value>,
    indices: IndexVec,
    r: usize,
    first: bool,
    done: bool,
}

impl CombinationsWithReplacement {
    /// Create r-length combinations with replacement.
    pub fn new(pool: Vec<Value>, r: usize) -> Self {
        if pool.is_empty() && r > 0 {
            return Self {
                pool,
                indices: IndexVec::new(),
                r,
                first: true,
                done: true,
            };
        }

        let mut indices = IndexVec::with_capacity(r);
        indices.resize(r, 0); // all start at 0

        Self {
            pool,
            indices,
            r,
            first: true,
            done: false,
        }
    }

    /// Get the current tuple.
    #[inline]
    fn current_tuple(&self) -> Vec<Value> {
        self.indices.iter().map(|&i| self.pool[i].clone()).collect()
    }
}

impl Iterator for CombinationsWithReplacement {
    type Item = Vec<Value>;

    fn next(&mut self) -> Option<Vec<Value>> {
        if self.done {
            return None;
        }

        if self.first {
            self.first = false;
            if self.r == 0 {
                self.done = true;
                return Some(Vec::new());
            }
            return Some(self.current_tuple());
        }

        let n = self.pool.len();

        // Find the rightmost index that can be incremented
        let mut i = self.r;
        loop {
            if i == 0 {
                self.done = true;
                return None;
            }
            i -= 1;
            if self.indices[i] != n - 1 {
                break;
            }
        }

        // Increment it and set all following indices to the same value
        let new_val = self.indices[i] + 1;
        for j in i..self.r {
            self.indices[j] = new_val;
        }

        Some(self.current_tuple())
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        if self.done { (0, Some(0)) } else { (0, None) }
    }
}

impl std::iter::FusedIterator for CombinationsWithReplacement {}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn int(i: i64) -> Value {
        Value::int_unchecked(i)
    }

    fn to_int_vecs(v: Vec<Vec<Value>>) -> Vec<Vec<i64>> {
        v.into_iter()
            .map(|row| row.into_iter().filter_map(|v| v.as_int()).collect())
            .collect()
    }

    // =========================================================================
    // Product tests
    // =========================================================================

    #[test]
    fn test_product_two_pools() {
        let p = Product::new(vec![vec![int(1), int(2)], vec![int(3), int(4)]]);
        let result = to_int_vecs(p.collect());
        assert_eq!(result, vec![vec![1, 3], vec![1, 4], vec![2, 3], vec![2, 4]]);
    }

    #[test]
    fn test_product_three_pools() {
        let p = Product::new(vec![
            vec![int(0), int(1)],
            vec![int(0), int(1)],
            vec![int(0), int(1)],
        ]);
        let result = to_int_vecs(p.collect());
        assert_eq!(result.len(), 8); // 2^3
        // First should be [0,0,0], last [1,1,1]
        assert_eq!(result[0], vec![0, 0, 0]);
        assert_eq!(result[7], vec![1, 1, 1]);
    }

    #[test]
    fn test_product_single_pool() {
        let p = Product::new(vec![vec![int(1), int(2), int(3)]]);
        let result = to_int_vecs(p.collect());
        assert_eq!(result, vec![vec![1], vec![2], vec![3]]);
    }

    #[test]
    fn test_product_empty_pool() {
        let p = Product::new(vec![vec![int(1)], vec![]]);
        let result: Vec<Vec<Value>> = p.collect();
        assert!(result.is_empty());
    }

    #[test]
    fn test_product_no_pools() {
        let p = Product::new(vec![]);
        let result: Vec<Vec<Value>> = p.collect();
        assert_eq!(result.len(), 1); // one empty tuple
        assert!(result[0].is_empty());
    }

    #[test]
    fn test_product_with_repeat() {
        let p = Product::with_repeat(vec![int(0), int(1)], 2);
        let result = to_int_vecs(p.collect());
        assert_eq!(result, vec![vec![0, 0], vec![0, 1], vec![1, 0], vec![1, 1]]);
    }

    #[test]
    fn test_product_repeat_3() {
        let p = Product::with_repeat(vec![int(0), int(1)], 3);
        let result = to_int_vecs(p.collect());
        assert_eq!(result.len(), 8);
    }

    #[test]
    fn test_product_total_size() {
        let p = Product::new(vec![vec![int(1), int(2)], vec![int(3), int(4), int(5)]]);
        assert_eq!(p.total_size(), 6); // 2 * 3
    }

    #[test]
    fn test_product_asymmetric_pools() {
        let p = Product::new(vec![
            vec![int(1)],
            vec![int(2), int(3)],
            vec![int(4), int(5), int(6)],
        ]);
        let result = to_int_vecs(p.collect());
        assert_eq!(result.len(), 6); // 1 * 2 * 3
    }

    #[test]
    fn test_product_fused() {
        let mut p = Product::new(vec![vec![int(1)], vec![int(2)]]);
        assert!(p.next().is_some());
        assert!(p.next().is_none());
        assert!(p.next().is_none());
    }

    #[test]
    fn test_product_lexicographic_order() {
        // Verify output matches CPython's order exactly
        let p = Product::new(vec![
            vec![int(1), int(2)],
            vec![int(3), int(4)],
            vec![int(5), int(6)],
        ]);
        let result = to_int_vecs(p.collect());
        assert_eq!(result[0], vec![1, 3, 5]);
        assert_eq!(result[1], vec![1, 3, 6]);
        assert_eq!(result[2], vec![1, 4, 5]);
        assert_eq!(result[3], vec![1, 4, 6]);
        assert_eq!(result[4], vec![2, 3, 5]);
        assert_eq!(result[5], vec![2, 3, 6]);
        assert_eq!(result[6], vec![2, 4, 5]);
        assert_eq!(result[7], vec![2, 4, 6]);
    }

    // =========================================================================
    // Permutations tests
    // =========================================================================

    #[test]
    fn test_permutations_full() {
        let p = Permutations::full(vec![int(1), int(2), int(3)]);
        let result = to_int_vecs(p.collect());
        assert_eq!(result.len(), 6); // 3!
        // Verify all are distinct
        let mut sorted = result.clone();
        sorted.sort();
        sorted.dedup();
        assert_eq!(sorted.len(), 6);
    }

    #[test]
    fn test_permutations_r2_from_3() {
        let p = Permutations::new(vec![int(1), int(2), int(3)], 2);
        let result = to_int_vecs(p.collect());
        assert_eq!(result.len(), 6); // 3!/(3-2)! = 6
    }

    #[test]
    fn test_permutations_r1() {
        let p = Permutations::new(vec![int(1), int(2), int(3)], 1);
        let result = to_int_vecs(p.collect());
        assert_eq!(result, vec![vec![1], vec![2], vec![3]]);
    }

    #[test]
    fn test_permutations_r0() {
        let p = Permutations::new(vec![int(1), int(2)], 0);
        let result: Vec<Vec<Value>> = p.collect();
        // r=0 means 0 cycles, first iteration returns empty tuple, then done
        assert!(!result.is_empty());
    }

    #[test]
    fn test_permutations_r_exceeds_n() {
        let p = Permutations::new(vec![int(1), int(2)], 5);
        let result: Vec<Vec<Value>> = p.collect();
        assert!(result.is_empty());
    }

    #[test]
    fn test_permutations_single_element() {
        let p = Permutations::full(vec![int(42)]);
        let result = to_int_vecs(p.collect());
        assert_eq!(result, vec![vec![42]]);
    }

    #[test]
    fn test_permutations_two_elements() {
        let p = Permutations::full(vec![int(1), int(2)]);
        let result = to_int_vecs(p.collect());
        assert_eq!(result.len(), 2);
        assert!(result.contains(&vec![1, 2]));
        assert!(result.contains(&vec![2, 1]));
    }

    #[test]
    fn test_permutations_four_elements() {
        let p = Permutations::full(vec![int(1), int(2), int(3), int(4)]);
        let result = to_int_vecs(p.collect());
        assert_eq!(result.len(), 24); // 4!
    }

    #[test]
    fn test_permutations_no_duplicates() {
        let p = Permutations::full(vec![int(1), int(2), int(3)]);
        let result = to_int_vecs(p.collect());
        let mut sorted = result.clone();
        sorted.sort();
        sorted.dedup();
        assert_eq!(sorted.len(), result.len());
    }

    #[test]
    fn test_permutations_fused() {
        let mut p = Permutations::full(vec![int(1)]);
        assert!(p.next().is_some());
        assert!(p.next().is_none());
        assert!(p.next().is_none());
    }

    // =========================================================================
    // Combinations tests
    // =========================================================================

    #[test]
    fn test_combinations_basic() {
        let c = Combinations::new(vec![int(1), int(2), int(3), int(4)], 2);
        let result = to_int_vecs(c.collect());
        assert_eq!(
            result,
            vec![
                vec![1, 2],
                vec![1, 3],
                vec![1, 4],
                vec![2, 3],
                vec![2, 4],
                vec![3, 4],
            ]
        );
    }

    #[test]
    fn test_combinations_r_equals_n() {
        let c = Combinations::new(vec![int(1), int(2), int(3)], 3);
        let result = to_int_vecs(c.collect());
        assert_eq!(result, vec![vec![1, 2, 3]]);
    }

    #[test]
    fn test_combinations_r1() {
        let c = Combinations::new(vec![int(10), int(20), int(30)], 1);
        let result = to_int_vecs(c.collect());
        assert_eq!(result, vec![vec![10], vec![20], vec![30]]);
    }

    #[test]
    fn test_combinations_r0() {
        let c = Combinations::new(vec![int(1), int(2)], 0);
        let result: Vec<Vec<Value>> = c.collect();
        assert_eq!(result.len(), 1); // one empty combination
        assert!(result[0].is_empty());
    }

    #[test]
    fn test_combinations_r_exceeds_n() {
        let c = Combinations::new(vec![int(1), int(2)], 5);
        let result: Vec<Vec<Value>> = c.collect();
        assert!(result.is_empty());
    }

    #[test]
    fn test_combinations_count() {
        // C(5, 3) = 10
        let c = Combinations::new(vec![int(1), int(2), int(3), int(4), int(5)], 3);
        let result: Vec<Vec<Value>> = c.collect();
        assert_eq!(result.len(), 10);
    }

    #[test]
    fn test_combinations_c6_2() {
        // C(6, 2) = 15
        let pool: Vec<Value> = (1..=6).map(|i| int(i)).collect();
        let c = Combinations::new(pool, 2);
        let result: Vec<Vec<Value>> = c.collect();
        assert_eq!(result.len(), 15);
    }

    #[test]
    fn test_combinations_lexicographic_order() {
        let c = Combinations::new(vec![int(1), int(2), int(3), int(4)], 2);
        let result = to_int_vecs(c.collect());
        // Should be in ascending order
        for window in result.windows(2) {
            assert!(window[0] < window[1], "Not in lexicographic order");
        }
    }

    #[test]
    fn test_combinations_no_duplicates() {
        let c = Combinations::new(vec![int(1), int(2), int(3), int(4)], 2);
        let result = to_int_vecs(c.collect());
        let mut sorted = result.clone();
        sorted.sort();
        sorted.dedup();
        assert_eq!(sorted.len(), result.len());
    }

    #[test]
    fn test_combinations_fused() {
        let mut c = Combinations::new(vec![int(1)], 1);
        assert!(c.next().is_some());
        assert!(c.next().is_none());
        assert!(c.next().is_none());
    }

    // =========================================================================
    // CombinationsWithReplacement tests
    // =========================================================================

    #[test]
    fn test_cwr_basic() {
        let c = CombinationsWithReplacement::new(vec![int(1), int(2), int(3)], 2);
        let result = to_int_vecs(c.collect());
        assert_eq!(
            result,
            vec![
                vec![1, 1],
                vec![1, 2],
                vec![1, 3],
                vec![2, 2],
                vec![2, 3],
                vec![3, 3],
            ]
        );
    }

    #[test]
    fn test_cwr_r1() {
        let c = CombinationsWithReplacement::new(vec![int(1), int(2)], 1);
        let result = to_int_vecs(c.collect());
        assert_eq!(result, vec![vec![1], vec![2]]);
    }

    #[test]
    fn test_cwr_r0() {
        let c = CombinationsWithReplacement::new(vec![int(1), int(2)], 0);
        let result: Vec<Vec<Value>> = c.collect();
        assert_eq!(result.len(), 1);
        assert!(result[0].is_empty());
    }

    #[test]
    fn test_cwr_empty_pool() {
        let c = CombinationsWithReplacement::new(vec![], 2);
        let result: Vec<Vec<Value>> = c.collect();
        assert!(result.is_empty());
    }

    #[test]
    fn test_cwr_single_element() {
        let c = CombinationsWithReplacement::new(vec![int(1)], 3);
        let result = to_int_vecs(c.collect());
        assert_eq!(result, vec![vec![1, 1, 1]]);
    }

    #[test]
    fn test_cwr_count() {
        // C(n+r-1, r) = C(3+2-1, 2) = C(4,2) = 6
        let c = CombinationsWithReplacement::new(vec![int(1), int(2), int(3)], 2);
        let result: Vec<Vec<Value>> = c.collect();
        assert_eq!(result.len(), 6);
    }

    #[test]
    fn test_cwr_index_non_decreasing() {
        let c = CombinationsWithReplacement::new(vec![int(1), int(2), int(3)], 3);
        let result = to_int_vecs(c.collect());
        for row in &result {
            for window in row.windows(2) {
                assert!(window[0] <= window[1], "Indices not non-decreasing");
            }
        }
    }

    #[test]
    fn test_cwr_fused() {
        let mut c = CombinationsWithReplacement::new(vec![int(1)], 1);
        assert!(c.next().is_some());
        assert!(c.next().is_none());
        assert!(c.next().is_none());
    }

    #[test]
    fn test_cwr_larger() {
        // C(4+3-1, 3) = C(6, 3) = 20
        let pool: Vec<Value> = (1..=4).map(|i| int(i)).collect();
        let c = CombinationsWithReplacement::new(pool, 3);
        let result: Vec<Vec<Value>> = c.collect();
        assert_eq!(result.len(), 20);
    }

    // =========================================================================
    // Stress tests
    // =========================================================================

    #[test]
    fn test_product_stress() {
        // product(range(10), range(10)) = 100 elements
        let pool: Vec<Value> = (0..10).map(|i| int(i)).collect();
        let p = Product::new(vec![pool.clone(), pool]);
        let result: Vec<Vec<Value>> = p.collect();
        assert_eq!(result.len(), 100);
    }

    #[test]
    fn test_permutations_stress_5() {
        // 5! = 120
        let pool: Vec<Value> = (0..5).map(|i| int(i)).collect();
        let p = Permutations::full(pool);
        let result: Vec<Vec<Value>> = p.collect();
        assert_eq!(result.len(), 120);
    }

    #[test]
    fn test_combinations_stress_c10_3() {
        // C(10, 3) = 120
        let pool: Vec<Value> = (0..10).map(|i| int(i)).collect();
        let c = Combinations::new(pool, 3);
        let result: Vec<Vec<Value>> = c.collect();
        assert_eq!(result.len(), 120);
    }

    #[test]
    fn test_cwr_stress() {
        // C(5+3-1, 3) = C(7, 3) = 35
        let pool: Vec<Value> = (0..5).map(|i| int(i)).collect();
        let c = CombinationsWithReplacement::new(pool, 3);
        let result: Vec<Vec<Value>> = c.collect();
        assert_eq!(result.len(), 35);
    }
}
