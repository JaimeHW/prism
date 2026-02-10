//! Comprehensive unpacking and slicing tests.
//!
//! Tests cover:
//! - SliceValue compact optional representation
//! - SliceObject construction patterns  
//! - SliceIndices resolution algorithm
//! - Slice iterator behavior
//! - Tuple/List basic operations

use prism_core::Value;
#[allow(unused_imports)]
use prism_runtime::types::slice::SliceIndices;
use prism_runtime::types::{ListObject, SliceObject, SliceValue, TupleObject};

// =============================================================================
// SliceValue Tests
// =============================================================================

#[test]
fn test_slice_value_none_construction() {
    let v = SliceValue::none();
    assert!(v.is_none());
    assert!(!v.is_some());
    assert_eq!(v.get(), None);
}

#[test]
fn test_slice_value_some_construction() {
    let v = SliceValue::some(42);
    assert!(v.is_some());
    assert!(!v.is_none());
    assert_eq!(v.get(), Some(42));
}

#[test]
fn test_slice_value_zero() {
    let v = SliceValue::some(0);
    assert!(v.is_some());
    assert_eq!(v.get(), Some(0));
}

#[test]
fn test_slice_value_negative() {
    let v = SliceValue::some(-10);
    assert!(v.is_some());
    assert_eq!(v.get(), Some(-10));
}

#[test]
fn test_slice_value_unwrap_or() {
    let none = SliceValue::none();
    let some = SliceValue::some(5);

    assert_eq!(none.unwrap_or(100), 100);
    assert_eq!(some.unwrap_or(100), 5);
}

#[test]
fn test_slice_value_from_option() {
    let v1: SliceValue = Some(10i64).into();
    let v2: SliceValue = None.into();

    assert_eq!(v1.get(), Some(10));
    assert!(v2.is_none());
}

// =============================================================================
// SliceObject Construction Tests
// =============================================================================

#[test]
fn test_slice_new_full() {
    let s = SliceObject::new(Some(1), Some(10), Some(2));
    assert_eq!(s.start(), Some(1));
    assert_eq!(s.stop(), Some(10));
    assert_eq!(s.step(), Some(2));
}

#[test]
fn test_slice_new_all_none() {
    let s = SliceObject::new(None, None, None);
    assert_eq!(s.start(), None);
    assert_eq!(s.stop(), None);
    assert_eq!(s.step(), None);
}

#[test]
fn test_slice_stop_only() {
    let s = SliceObject::stop_only(5);
    assert_eq!(s.start(), None);
    assert_eq!(s.stop(), Some(5));
    assert_eq!(s.step(), None);
}

#[test]
fn test_slice_start_stop() {
    let s = SliceObject::start_stop(2, 8);
    assert_eq!(s.start(), Some(2));
    assert_eq!(s.stop(), Some(8));
    assert_eq!(s.step(), None);
}

#[test]
fn test_slice_full() {
    let s = SliceObject::full(0, 10, 2);
    assert_eq!(s.start(), Some(0));
    assert_eq!(s.stop(), Some(10));
    assert_eq!(s.step(), Some(2));
}

#[test]
#[should_panic(expected = "slice step cannot be zero")]
fn test_slice_zero_step_panics() {
    SliceObject::new(Some(0), Some(10), Some(0));
}

// =============================================================================
// Slice Index Resolution Tests
// =============================================================================

#[test]
fn test_indices_simple_forward() {
    // [1:5] on length-10
    let s = SliceObject::start_stop(1, 5);
    let idx = s.indices(10);
    assert_eq!(idx.start, 1);
    assert_eq!(idx.stop, 5);
    assert_eq!(idx.step, 1);
    assert_eq!(idx.length, 4);
}

#[test]
fn test_indices_full_slice() {
    // [:] on length-5
    let s = SliceObject::new(None, None, None);
    let idx = s.indices(5);
    assert_eq!(idx.start, 0);
    assert_eq!(idx.stop, 5);
    assert_eq!(idx.step, 1);
    assert_eq!(idx.length, 5);
}

#[test]
fn test_indices_negative_start() {
    // [-3:] on length-5 = [2:]
    let s = SliceObject::new(Some(-3), None, None);
    let idx = s.indices(5);
    assert_eq!(idx.start, 2);
    assert_eq!(idx.stop, 5);
    assert_eq!(idx.length, 3);
}

#[test]
fn test_indices_negative_stop() {
    // [:-2] on length-5 = [:3]
    let s = SliceObject::new(None, Some(-2), None);
    let idx = s.indices(5);
    assert_eq!(idx.start, 0);
    assert_eq!(idx.stop, 3);
    assert_eq!(idx.length, 3);
}

#[test]
fn test_indices_negative_both() {
    // [-4:-1] on length-5 = [1:4]
    let s = SliceObject::start_stop(-4, -1);
    let idx = s.indices(5);
    assert_eq!(idx.start, 1);
    assert_eq!(idx.stop, 4);
    assert_eq!(idx.length, 3);
}

#[test]
fn test_indices_with_step() {
    // [0:10:2] on length-10
    let s = SliceObject::full(0, 10, 2);
    let idx = s.indices(10);
    assert_eq!(idx.length, 5); // 0, 2, 4, 6, 8
}

#[test]
fn test_indices_reverse() {
    // [::-1] on length-5
    let s = SliceObject::new(None, None, Some(-1));
    let idx = s.indices(5);
    assert_eq!(idx.start, 4);
    assert_eq!(idx.step, -1);
    assert_eq!(idx.length, 5);
}

#[test]
fn test_indices_reverse_with_bounds() {
    // [4:1:-1] on length-5
    let s = SliceObject::full(4, 1, -1);
    let idx = s.indices(5);
    assert_eq!(idx.length, 3); // 4, 3, 2
}

#[test]
fn test_indices_empty_forward() {
    // [5:3] = empty (start > stop with positive step)
    let s = SliceObject::start_stop(5, 3);
    let idx = s.indices(10);
    assert_eq!(idx.length, 0);
}

#[test]
fn test_indices_empty_reverse() {
    // [3:5:-1] = empty (start < stop with negative step)
    let s = SliceObject::full(3, 5, -1);
    let idx = s.indices(10);
    assert_eq!(idx.length, 0);
}

#[test]
fn test_indices_out_of_bounds_clamped() {
    // [0:100] on length-5 = [0:5]
    let s = SliceObject::start_stop(0, 100);
    let idx = s.indices(5);
    assert_eq!(idx.stop, 5);
    assert_eq!(idx.length, 5);
}

#[test]
fn test_indices_negative_out_of_bounds() {
    // [-100:3] on length-5 = [0:3]
    let s = SliceObject::start_stop(-100, 3);
    let idx = s.indices(5);
    assert_eq!(idx.start, 0);
    assert_eq!(idx.length, 3);
}

#[test]
fn test_indices_empty_sequence() {
    let s = SliceObject::new(None, None, None);
    let idx = s.indices(0);
    assert_eq!(idx.length, 0);
}

#[test]
fn test_indices_step_2() {
    // [::2] on length-7
    let s = SliceObject::new(None, None, Some(2));
    let idx = s.indices(7);
    assert_eq!(idx.length, 4); // 0, 2, 4, 6
}

#[test]
fn test_indices_step_3() {
    // [1:10:3] on length-10
    let s = SliceObject::full(1, 10, 3);
    let idx = s.indices(10);
    assert_eq!(idx.length, 3); // 1, 4, 7
}

// =============================================================================
// Slice Iterator Tests
// =============================================================================

#[test]
fn test_slice_iter_forward() {
    let s = SliceObject::start_stop(1, 5);
    let indices: Vec<usize> = s.indices(10).iter().collect();
    assert_eq!(indices, vec![1, 2, 3, 4]);
}

#[test]
fn test_slice_iter_step() {
    let s = SliceObject::full(0, 10, 2);
    let indices: Vec<usize> = s.indices(10).iter().collect();
    assert_eq!(indices, vec![0, 2, 4, 6, 8]);
}

#[test]
fn test_slice_iter_reverse() {
    let s = SliceObject::new(None, None, Some(-1));
    let indices: Vec<usize> = s.indices(5).iter().collect();
    assert_eq!(indices, vec![4, 3, 2, 1, 0]);
}

#[test]
fn test_slice_iter_reverse_step() {
    let s = SliceObject::full(8, 0, -2);
    let indices: Vec<usize> = s.indices(10).iter().collect();
    assert_eq!(indices, vec![8, 6, 4, 2]);
}

#[test]
fn test_slice_iter_size_hint() {
    let s = SliceObject::start_stop(0, 5);
    let iter = s.indices(10).iter();
    let (lower, upper) = iter.size_hint();
    assert_eq!(lower, 5);
    assert_eq!(upper, Some(5));
}

// =============================================================================
// Slice Equality Tests
// =============================================================================

#[test]
fn test_slice_equality() {
    let s1 = SliceObject::start_stop(1, 5);
    let s2 = SliceObject::start_stop(1, 5);
    let s3 = SliceObject::start_stop(1, 6);

    assert_eq!(s1, s2);
    assert_ne!(s1, s3);
}

#[test]
fn test_slice_clone() {
    let s1 = SliceObject::full(1, 10, 2);
    let s2 = s1.clone();
    assert_eq!(s1, s2);
}

// =============================================================================
// Tuple Basic Tests
// =============================================================================

#[test]
fn test_tuple_basic_creation() {
    let tuple = TupleObject::from_vec(vec![Value::int_unchecked(1), Value::int_unchecked(2)]);
    assert_eq!(tuple.len(), 2);
}

#[test]
fn test_tuple_get_positive() {
    let tuple = TupleObject::from_vec(vec![
        Value::int_unchecked(1),
        Value::int_unchecked(2),
        Value::int_unchecked(3),
    ]);
    let v0 = tuple.get(0).expect("element at 0");
    let v1 = tuple.get(1).expect("element at 1");
    let v2 = tuple.get(2).expect("element at 2");
    assert_eq!(v0.as_int(), Some(1));
    assert_eq!(v1.as_int(), Some(2));
    assert_eq!(v2.as_int(), Some(3));
}

#[test]
fn test_tuple_get_negative() {
    let tuple = TupleObject::from_vec(vec![
        Value::int_unchecked(1),
        Value::int_unchecked(2),
        Value::int_unchecked(3),
    ]);
    let v_neg1 = tuple.get(-1).expect("element at -1");
    let v_neg2 = tuple.get(-2).expect("element at -2");
    let v_neg3 = tuple.get(-3).expect("element at -3");
    assert_eq!(v_neg1.as_int(), Some(3));
    assert_eq!(v_neg2.as_int(), Some(2));
    assert_eq!(v_neg3.as_int(), Some(1));
}

#[test]
fn test_tuple_empty() {
    let tuple = TupleObject::empty();
    assert_eq!(tuple.len(), 0);
    assert!(tuple.is_empty());
}

#[test]
fn test_tuple_get_out_of_bounds() {
    let tuple = TupleObject::from_vec(vec![Value::int_unchecked(1)]);
    assert!(tuple.get(5).is_none());
    assert!(tuple.get(-5).is_none());
}

// =============================================================================
// List Basic Tests
// =============================================================================

#[test]
fn test_list_basic_creation() {
    let list = ListObject::from_iter(vec![Value::int_unchecked(1), Value::int_unchecked(2)]);
    assert_eq!(list.len(), 2);
}

#[test]
fn test_list_get_positive() {
    let list = ListObject::from_iter(vec![
        Value::int_unchecked(10),
        Value::int_unchecked(20),
        Value::int_unchecked(30),
    ]);
    let v0 = list.get(0).expect("element at 0");
    let v1 = list.get(1).expect("element at 1");
    let v2 = list.get(2).expect("element at 2");
    assert_eq!(v0.as_int(), Some(10));
    assert_eq!(v1.as_int(), Some(20));
    assert_eq!(v2.as_int(), Some(30));
}

#[test]
fn test_list_get_negative() {
    let list = ListObject::from_iter(vec![
        Value::int_unchecked(1),
        Value::int_unchecked(2),
        Value::int_unchecked(3),
    ]);
    let v_neg1 = list.get(-1).expect("element at -1");
    let v_neg2 = list.get(-2).expect("element at -2");
    assert_eq!(v_neg1.as_int(), Some(3));
    assert_eq!(v_neg2.as_int(), Some(2));
}

#[test]
fn test_list_empty() {
    let list = ListObject::new();
    assert_eq!(list.len(), 0);
    assert!(list.is_empty());
}

// =============================================================================
// Edge Case Tests
// =============================================================================

#[test]
fn test_slice_step_negative_2() {
    // [10:0:-2] on length-11
    let s = SliceObject::full(10, 0, -2);
    let indices: Vec<usize> = s.indices(11).iter().collect();
    assert_eq!(indices, vec![10, 8, 6, 4, 2]);
}

#[test]
fn test_slice_single_element() {
    // [5:6] on length-10 = single element at index 5
    let s = SliceObject::start_stop(5, 6);
    let idx = s.indices(10);
    assert_eq!(idx.length, 1);
    assert_eq!(idx.start, 5);
}

#[test]
fn test_slice_step_larger_than_range() {
    // [0:5:10] on length-10 = just index 0
    let s = SliceObject::full(0, 5, 10);
    let idx = s.indices(10);
    assert_eq!(idx.length, 1);
}

#[test]
fn test_large_step_reverse() {
    // [9:0:-100] = just index 9
    let s = SliceObject::full(9, 0, -100);
    let idx = s.indices(10);
    assert_eq!(idx.length, 1);
}
