//! Comprehensive membership operator tests for `in` / `not in`.
//!
//! Tests cover:
//! - String containment (SIMD substring search)
//! - Range containment (O(1) arithmetic)
//!
//! These tests focus on the underlying type methods that the VM's
//! `contains_value` function dispatches to.

use prism_runtime::types::range::RangeObject;
use prism_runtime::types::string::StringObject;

// =============================================================================
// String Containment Tests (SIMD-accelerated)
// =============================================================================

#[test]
fn test_string_contains_substring() {
    let haystack = StringObject::new("hello world");
    assert!(haystack.contains("world"));
}

#[test]
fn test_string_not_contains_substring() {
    let haystack = StringObject::new("hello world");
    assert!(!haystack.contains("xyz"));
}

#[test]
fn test_string_contains_empty() {
    let haystack = StringObject::new("hello");
    assert!(haystack.contains(""));
}

#[test]
fn test_string_contains_full_match() {
    let haystack = StringObject::new("exact");
    assert!(haystack.contains("exact"));
}

#[test]
fn test_string_contains_single_char() {
    let haystack = StringObject::new("hello");
    assert!(haystack.contains("e"));
    assert!(!haystack.contains("x"));
}

#[test]
fn test_string_contains_at_start() {
    let haystack = StringObject::new("hello world");
    assert!(haystack.contains("hello"));
}

#[test]
fn test_string_contains_at_end() {
    let haystack = StringObject::new("hello world");
    assert!(haystack.contains("world"));
}

#[test]
fn test_string_contains_longer_than_haystack() {
    let haystack = StringObject::new("hi");
    assert!(!haystack.contains("hello world"));
}

#[test]
fn test_string_contains_unicode() {
    let haystack = StringObject::new("hello 世界 🌍");
    assert!(haystack.contains("世界"));
    assert!(haystack.contains("🌍"));
}

#[test]
fn test_string_empty_haystack() {
    let haystack = StringObject::new("");
    assert!(!haystack.contains("a"));
    assert!(haystack.contains(""));
}

#[test]
fn test_string_contains_repeated_pattern() {
    let haystack = StringObject::new("ababababab");
    assert!(haystack.contains("abab"));
    assert!(haystack.contains("bab"));
}

#[test]
fn test_string_contains_partial_match_no_full() {
    let haystack = StringObject::new("abcd");
    assert!(!haystack.contains("abce"));
    assert!(haystack.contains("abc"));
}

#[test]
fn test_string_contains_large_haystack() {
    // Test SIMD path with larger strings
    let haystack = StringObject::new(&"a".repeat(10000));
    assert!(haystack.contains("aaaa"));
    assert!(!haystack.contains("b"));
}

#[test]
fn test_string_contains_needle_at_boundary() {
    // Test 16-byte boundary (SSE register size)
    let haystack = StringObject::new("0123456789abcdef_needle_here");
    assert!(haystack.contains("needle"));
}

// =============================================================================
// Range Containment Tests (O(1) arithmetic)
// =============================================================================

#[test]
fn test_range_contains_basic() {
    let range = RangeObject::from_stop(10);
    assert!(range.contains(0));
    assert!(range.contains(5));
    assert!(range.contains(9));
    assert!(!range.contains(10));
    assert!(!range.contains(-1));
}

#[test]
fn test_range_contains_with_start() {
    let range = RangeObject::from_start_stop(5, 10);
    assert!(!range.contains(4));
    assert!(range.contains(5));
    assert!(range.contains(9));
    assert!(!range.contains(10));
}

#[test]
fn test_range_contains_with_step() {
    // range(0, 10, 2) → 0, 2, 4, 6, 8
    let range = RangeObject::new(0, 10, 2);
    assert!(range.contains(0));
    assert!(range.contains(2));
    assert!(range.contains(8));
    assert!(!range.contains(1)); // Not aligned with step
    assert!(!range.contains(3));
    assert!(!range.contains(10)); // Out of bounds
}

#[test]
fn test_range_contains_negative_step() {
    // range(10, 0, -1) → 10, 9, 8, ..., 1
    let range = RangeObject::new(10, 0, -1);
    assert!(range.contains(10));
    assert!(range.contains(5));
    assert!(range.contains(1));
    assert!(!range.contains(0)); // stop is exclusive
    assert!(!range.contains(11));
}

#[test]
fn test_range_contains_negative_step_2() {
    // range(10, 0, -2) → 10, 8, 6, 4, 2
    let range = RangeObject::new(10, 0, -2);
    assert!(range.contains(10));
    assert!(range.contains(8));
    assert!(range.contains(2));
    assert!(!range.contains(9)); // Not aligned
    assert!(!range.contains(0));
}

#[test]
fn test_range_empty() {
    // range(5, 5) is empty
    let range = RangeObject::from_start_stop(5, 5);
    assert!(!range.contains(5));
    assert!(range.is_empty());
}

#[test]
fn test_range_empty_reverse() {
    // range(0, 10, -1) is empty (can't go from 0 to 10 with negative step)
    let range = RangeObject::new(0, 10, -1);
    assert!(range.is_empty());
}

#[test]
fn test_range_large_values() {
    // Test with large integers
    let range = RangeObject::from_start_stop(i64::MAX - 100, i64::MAX);
    assert!(range.contains(i64::MAX - 50));
    assert!(!range.contains(i64::MAX));
}

#[test]
fn test_range_negative_values() {
    let range = RangeObject::from_start_stop(-10, -5);
    assert!(range.contains(-10));
    assert!(range.contains(-6));
    assert!(!range.contains(-5));
    assert!(!range.contains(0));
}

#[test]
fn test_range_step_3() {
    // range(0, 15, 3) → 0, 3, 6, 9, 12
    let range = RangeObject::new(0, 15, 3);
    assert!(range.contains(0));
    assert!(range.contains(3));
    assert!(range.contains(12));
    assert!(!range.contains(15));
    assert!(!range.contains(1));
    assert!(!range.contains(2));
}

#[test]
fn test_range_step_negative_3() {
    // range(12, 0, -3) → 12, 9, 6, 3
    let range = RangeObject::new(12, 0, -3);
    assert!(range.contains(12));
    assert!(range.contains(9));
    assert!(range.contains(3));
    assert!(!range.contains(0));
    assert!(!range.contains(10));
}

#[test]
fn test_range_single_element() {
    // range(5, 6) → just [5]
    let range = RangeObject::from_start_stop(5, 6);
    assert!(range.contains(5));
    assert!(!range.contains(4));
    assert!(!range.contains(6));
    assert_eq!(range.len(), 1);
}

#[test]
fn test_range_crossing_zero() {
    let range = RangeObject::from_start_stop(-5, 5);
    assert!(range.contains(-5));
    assert!(range.contains(0));
    assert!(range.contains(4));
    assert!(!range.contains(5));
}

#[test]
fn test_range_len() {
    assert_eq!(RangeObject::from_stop(10).len(), 10);
    assert_eq!(RangeObject::from_start_stop(5, 10).len(), 5);
    assert_eq!(RangeObject::new(0, 10, 2).len(), 5);
    assert_eq!(RangeObject::new(10, 0, -1).len(), 10);
    assert_eq!(RangeObject::new(10, 0, -2).len(), 5);
}

#[test]
fn test_range_iter() {
    let range = RangeObject::new(0, 10, 2);
    let values: Vec<i64> = range.iter().collect();
    assert_eq!(values, vec![0, 2, 4, 6, 8]);
}

#[test]
fn test_range_iter_reverse() {
    let range = RangeObject::new(10, 0, -2);
    let values: Vec<i64> = range.iter().collect();
    assert_eq!(values, vec![10, 8, 6, 4, 2]);
}

#[test]
fn test_range_get_indexed() {
    let range = RangeObject::new(0, 10, 2);
    assert_eq!(range.get(0), Some(0));
    assert_eq!(range.get(1), Some(2));
    assert_eq!(range.get(4), Some(8));
    assert_eq!(range.get(5), None);
}

#[test]
fn test_range_get_negative_indexed() {
    let range = RangeObject::new(0, 10, 2);
    assert_eq!(range.get(-1), Some(8));
    assert_eq!(range.get(-2), Some(6));
    assert_eq!(range.get(-5), Some(0));
    assert_eq!(range.get(-6), None);
}
