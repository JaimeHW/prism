//! Integration tests for the `itertools` module.
//!
//! These tests verify cross-component interactions and complex patterns
//! that span multiple iterator types.

use prism_core::Value;

use super::combinatoric::*;
use super::grouping::*;
use super::infinite::*;
use super::recipes::*;
use super::terminating::*;

fn int(i: i64) -> Value {
    Value::int_unchecked(i)
}

fn vals(ints: &[i64]) -> Vec<Value> {
    ints.iter().map(|&i| int(i)).collect()
}

fn to_ints(v: Vec<Value>) -> Vec<i64> {
    v.into_iter().filter_map(|v| v.as_int()).collect()
}

// =============================================================================
// Chain + other iterators
// =============================================================================

#[test]
fn test_chain_then_filter() {
    let a = vals(&[1, 2, 3]);
    let b = vals(&[4, 5, 6]);
    let result = to_ints(
        Chain::new(a.into_iter(), b.into_iter())
            .filter(|v| v.as_int().unwrap() % 2 == 0)
            .collect(),
    );
    assert_eq!(result, vec![2, 4, 6]);
}

#[test]
fn test_chain_then_unique() {
    let a = vals(&[1, 2, 3]);
    let b = vals(&[2, 3, 4]);
    let chained = Chain::new(a.into_iter(), b.into_iter());
    let result = to_ints(UniqueEverseen::new(chained).collect());
    assert_eq!(result, vec![1, 2, 3, 4]);
}

// =============================================================================
// Count + TakeWhile / ISlice
// =============================================================================

#[test]
fn test_count_takewhile() {
    let result = to_ints(TakeWhile::new(Count::new(0, 1), |v| v.as_int().unwrap() < 5).collect());
    assert_eq!(result, vec![0, 1, 2, 3, 4]);
}

#[test]
fn test_count_islice() {
    let result = to_ints(ISlice::new(Count::new(0, 1), 10, Some(20), 3).collect());
    assert_eq!(result, vec![10, 13, 16, 19]);
}

#[test]
fn test_count_islice_simple() {
    let result = to_ints(ISlice::new_stop(Count::new(100, 1), 5).collect());
    assert_eq!(result, vec![100, 101, 102, 103, 104]);
}

// =============================================================================
// Cycle + ISlice
// =============================================================================

#[test]
fn test_cycle_islice() {
    let c = Cycle::from_pool(vals(&[1, 2, 3]));
    let result = to_ints(ISlice::new_stop(c, 7).collect());
    assert_eq!(result, vec![1, 2, 3, 1, 2, 3, 1]);
}

// =============================================================================
// Repeat + Compress
// =============================================================================

#[test]
fn test_repeat_as_selector() {
    let data = vals(&[10, 20, 30, 40, 50]);
    let selectors = Repeat::times(Value::bool(true), 3); // only first 3 are true
    // Compress stops when shorter iterator (selectors) ends
    let result = to_ints(Compress::new(data.into_iter(), selectors).collect());
    assert_eq!(result, vec![10, 20, 30]);
}

// =============================================================================
// Product + FilterFalse
// =============================================================================

#[test]
fn test_product_filter_diagonal() {
    // Product of [0,1,2] x [0,1,2], filter out diagonal (i == j)
    let pool = vals(&[0, 1, 2]);
    let p = Product::with_repeat(pool, 2);
    let result: Vec<Vec<i64>> = p
        .filter(|v| {
            let a = v[0].as_int().unwrap();
            let b = v[1].as_int().unwrap();
            a != b
        })
        .map(|v| v.into_iter().filter_map(|x| x.as_int()).collect())
        .collect();
    assert_eq!(result.len(), 6); // 9 - 3 diagonal
}

// =============================================================================
// GroupBy + accumulate pattern
// =============================================================================

#[test]
fn test_groupby_then_count_groups() {
    let data = vals(&[1, 1, 2, 2, 2, 3, 3, 1]);
    let groups: Vec<(i64, usize)> = GroupBy::new(data.into_iter(), |v| v.clone())
        .map(|(k, g)| (k.as_int().unwrap(), g.len()))
        .collect();
    assert_eq!(groups, vec![(1, 2), (2, 3), (3, 2), (1, 1)]);
}

// =============================================================================
// Pairwise + accumulate
// =============================================================================

#[test]
fn test_pairwise_differences() {
    // Compute first differences using pairwise
    let data = vals(&[1, 3, 6, 10, 15]);
    let diffs: Vec<i64> = Pairwise::new(data.into_iter())
        .map(|(a, b)| b.as_int().unwrap() - a.as_int().unwrap())
        .collect();
    assert_eq!(diffs, vec![2, 3, 4, 5]);
}

// =============================================================================
// Combinations + quantify
// =============================================================================

#[test]
fn test_combinations_count_with_sum() {
    // How many pairs from [1,2,3,4,5] sum to >= 7?
    let pool = vals(&[1, 2, 3, 4, 5]);
    let count = Combinations::new(pool, 2)
        .filter(|c| {
            let sum: i64 = c.iter().filter_map(|v| v.as_int()).sum();
            sum >= 7
        })
        .count();
    assert_eq!(count, 4); // (2,5), (3,4), (3,5), (4,5)
}

// =============================================================================
// Batched + flatten roundtrip
// =============================================================================

#[test]
fn test_batched_flatten_roundtrip() {
    let data = vals(&[1, 2, 3, 4, 5, 6, 7, 8, 9]);
    let batched: Vec<Vec<Value>> = Batched::new(data.clone().into_iter(), 3).collect();
    let flattened = to_ints(Flatten::new(batched.into_iter()).collect());
    assert_eq!(flattened, vec![1, 2, 3, 4, 5, 6, 7, 8, 9]);
}

// =============================================================================
// Accumulate running sum verification
// =============================================================================

#[test]
fn test_accumulate_running_sum_matches_manual() {
    let data = vals(&[3, 1, 4, 1, 5, 9, 2, 6, 5, 3]);
    let result = to_ints(
        Accumulate::new(data.into_iter(), |a, b| {
            int(a.as_int().unwrap() + b.as_int().unwrap())
        })
        .collect(),
    );

    let expected = vec![3, 4, 8, 9, 14, 23, 25, 31, 36, 39];
    assert_eq!(result, expected);
}

// =============================================================================
// Partition + quantify consistency
// =============================================================================

#[test]
fn test_partition_quantify_consistent() {
    let data = vals(&[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
    let pred = |v: &Value| v.as_int().unwrap() > 5;

    let count = quantify(data.clone().into_iter(), pred);
    let (falses, trues) = partition(data.into_iter(), pred);

    assert_eq!(count, trues.len());
    assert_eq!(falses.len() + trues.len(), 10);
}

// =============================================================================
// ZipLongest + chain
// =============================================================================

#[test]
fn test_zip_longest_fill_with_chain() {
    let a = vals(&[1, 2, 3]);
    let b = vals(&[10, 20]);
    let result: Vec<(i64, i64)> = ZipLongest::new(a.into_iter(), b.into_iter())
        .map(|eob| {
            let (l, r) = eob.into_tuple(int(0), int(0));
            (l.as_int().unwrap(), r.as_int().unwrap())
        })
        .collect();
    assert_eq!(result, vec![(1, 10), (2, 20), (3, 0)]);
}

// =============================================================================
// Sliding window on count
// =============================================================================

#[test]
fn test_sliding_window_on_count() {
    // Take 5 sliding windows of size 3 from count(0)
    let windows: Vec<Vec<i64>> = SlidingWindow::new(ISlice::new_stop(Count::new(0, 1), 7), 3)
        .map(|w| to_ints(w))
        .collect();
    assert_eq!(
        windows,
        vec![
            vec![0, 1, 2],
            vec![1, 2, 3],
            vec![2, 3, 4],
            vec![3, 4, 5],
            vec![4, 5, 6],
        ]
    );
}

// =============================================================================
// RoundRobin preserves all elements
// =============================================================================

#[test]
fn test_roundrobin_preserves_all_elements() {
    let iters = vec![vals(&[1, 4, 7]), vals(&[2, 5]), vals(&[3, 6, 8, 9])];
    let total: usize = iters.iter().map(|v| v.len()).sum();
    let result = to_ints(RoundRobin::new(iters).collect());
    assert_eq!(result.len(), total);

    // All elements present (sorted)
    let mut sorted = result.clone();
    sorted.sort();
    assert_eq!(sorted, vec![1, 2, 3, 4, 5, 6, 7, 8, 9]);
}

// =============================================================================
// Permutations + unique verification
// =============================================================================

#[test]
fn test_permutations_all_unique_elements() {
    let pool = vals(&[1, 2, 3, 4]);
    let perms = Permutations::full(pool);
    let results: Vec<Vec<i64>> = perms
        .map(|p| p.into_iter().filter_map(|v| v.as_int()).collect())
        .collect();

    // All should be unique
    let mut sorted = results.clone();
    sorted.sort();
    sorted.dedup();
    assert_eq!(sorted.len(), results.len());
    assert_eq!(results.len(), 24); // 4!
}

// =============================================================================
// DropWhile + TakeWhile = extracting a range
// =============================================================================

#[test]
fn test_dropwhile_takewhile_range_extraction() {
    // Extract elements between 3 and 7 inclusive
    let data = vals(&[1, 2, 3, 4, 5, 6, 7, 8, 9]);
    let result = to_ints(
        TakeWhile::new(
            DropWhile::new(data.into_iter(), |v| v.as_int().unwrap() < 3),
            |v| v.as_int().unwrap() <= 7,
        )
        .collect(),
    );
    assert_eq!(result, vec![3, 4, 5, 6, 7]);
}

// =============================================================================
// Stress tests
// =============================================================================

#[test]
fn test_chain_compress_stress() {
    let data: Vec<Value> = (0..10_000).map(|i| int(i)).collect();
    let selectors: Vec<Value> = (0..10_000).map(|i| Value::bool(i % 3 == 0)).collect();
    let result = to_ints(Compress::new(data.into_iter(), selectors.into_iter()).collect());
    assert_eq!(result.len(), 3334); // ceil(10000/3)
}

#[test]
fn test_unique_everseen_on_cycle() {
    // Cycle [0,1,2] and take unique from first 100 elements
    let c = Cycle::from_pool(vals(&[0, 1, 2]));
    let first100: Vec<Value> = c.take(100).collect();
    let unique = to_ints(UniqueEverseen::new(first100.into_iter()).collect());
    assert_eq!(unique, vec![0, 1, 2]);
}

#[test]
fn test_complex_pipeline() {
    // count(0,2) → take(50) → batch(5) → flatten → filter(>30) → unique
    let pipeline: Vec<i64> = ISlice::new_stop(Count::new(0, 2), 50)
        .filter(|v| v.as_int().unwrap() > 30)
        .filter_map(|v| v.as_int())
        .collect();

    // count(0,2) gives 0,2,4,...,98
    // > 30 gives 32,34,...,98
    assert_eq!(pipeline.len(), 34); // 34 values from 32 to 98
    assert_eq!(pipeline[0], 32);
    assert_eq!(*pipeline.last().unwrap(), 98);
}
