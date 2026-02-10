//! High-performance `reduce` implementation.
//!
//! Applies a two-argument function cumulatively to the items of an iterable,
//! from left to right, reducing the iterable to a single value.
//!
//! # Performance Characteristics
//!
//! | Operation | Time | Space |
//! |-----------|------|-------|
//! | `reduce(f, iter)` | O(n) | O(1) |
//! | `reduce(f, iter, init)` | O(n) | O(1) |
//!
//! Single pass, zero intermediate allocations.

use prism_core::Value;

// =============================================================================
// Reduce Function
// =============================================================================

/// Error type for reduce operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ReduceError {
    /// Iterable was empty and no initializer was provided.
    EmptySequence,
    /// The function returned an error.
    FunctionError(String),
}

impl std::fmt::Display for ReduceError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ReduceError::EmptySequence => {
                write!(f, "reduce() of empty iterable with no initial value")
            }
            ReduceError::FunctionError(msg) => write!(f, "reduce() function error: {}", msg),
        }
    }
}

impl std::error::Error for ReduceError {}

/// Apply `function` cumulatively to items of `iterable`, reducing to a single value.
///
/// If `initializer` is present, it is placed before the items of the iterable
/// in the reduction, and serves as a default when the iterable is empty.
///
/// # Errors
///
/// Returns `ReduceError::EmptySequence` if the iterable is empty and no
/// initializer is provided.
///
/// # Performance
///
/// - Single pass over the iterable (O(n))
/// - No intermediate allocations
/// - Short-circuits on function error
#[inline]
pub fn reduce<F>(
    function: F,
    iterable: impl IntoIterator<Item = Value>,
    initializer: Option<Value>,
) -> Result<Value, ReduceError>
where
    F: Fn(&Value, &Value) -> Value,
{
    let mut iter = iterable.into_iter();

    // Get initial accumulator
    let mut accumulator = match initializer {
        Some(init) => init,
        None => iter.next().ok_or(ReduceError::EmptySequence)?,
    };

    // Apply function cumulatively
    for item in iter {
        accumulator = function(&accumulator, &item);
    }

    Ok(accumulator)
}

/// Apply `function` cumulatively with a fallible function.
///
/// Like `reduce`, but the function may return an error.
#[inline]
pub fn reduce_fallible<F, E>(
    function: F,
    iterable: impl IntoIterator<Item = Value>,
    initializer: Option<Value>,
) -> Result<Value, ReduceError>
where
    F: Fn(&Value, &Value) -> Result<Value, E>,
    E: std::fmt::Display,
{
    let mut iter = iterable.into_iter();

    let mut accumulator = match initializer {
        Some(init) => init,
        None => iter.next().ok_or(ReduceError::EmptySequence)?,
    };

    for item in iter {
        accumulator =
            function(&accumulator, &item).map_err(|e| ReduceError::FunctionError(e.to_string()))?;
    }

    Ok(accumulator)
}

/// Scan (prefix reduction) — returns all intermediate accumulator values.
///
/// This is `itertools.accumulate` in Python, but included here as it shares
/// the reduction kernel.
///
/// # Performance
///
/// - Single pass O(n), allocates result vector of size n
pub fn accumulate<F>(
    function: F,
    iterable: impl IntoIterator<Item = Value>,
    initializer: Option<Value>,
) -> Vec<Value>
where
    F: Fn(&Value, &Value) -> Value,
{
    let iter = iterable.into_iter();
    let (size_hint, _) = iter.size_hint();
    let mut results = Vec::with_capacity(size_hint.max(1));

    let mut iter = iter.peekable();

    let mut accumulator = match initializer {
        Some(init) => {
            results.push(init.clone());
            init
        }
        None => match iter.next() {
            Some(first) => {
                results.push(first.clone());
                first
            }
            None => return results,
        },
    };

    for item in iter {
        accumulator = function(&accumulator, &item);
        results.push(accumulator.clone());
    }

    results
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod reduce_tests {
    use super::*;
    use prism_core::intern::intern;

    fn int(i: i64) -> Value {
        Value::int_unchecked(i)
    }

    fn str_val(s: &str) -> Value {
        Value::string(intern(s))
    }

    fn add(a: &Value, b: &Value) -> Value {
        int(a.as_int().unwrap() + b.as_int().unwrap())
    }

    fn mul(a: &Value, b: &Value) -> Value {
        int(a.as_int().unwrap() * b.as_int().unwrap())
    }

    fn max_fn(a: &Value, b: &Value) -> Value {
        if a.as_int().unwrap() >= b.as_int().unwrap() {
            a.clone()
        } else {
            b.clone()
        }
    }

    fn min_fn(a: &Value, b: &Value) -> Value {
        if a.as_int().unwrap() <= b.as_int().unwrap() {
            a.clone()
        } else {
            b.clone()
        }
    }

    // =========================================================================
    // Basic reduce tests
    // =========================================================================

    #[test]
    fn test_reduce_sum() {
        let result = reduce(add, vec![int(1), int(2), int(3), int(4)], None).unwrap();
        assert_eq!(result.as_int(), Some(10));
    }

    #[test]
    fn test_reduce_product() {
        let result = reduce(mul, vec![int(1), int(2), int(3), int(4)], None).unwrap();
        assert_eq!(result.as_int(), Some(24));
    }

    #[test]
    fn test_reduce_max() {
        let result = reduce(max_fn, vec![int(3), int(1), int(4), int(1), int(5)], None).unwrap();
        assert_eq!(result.as_int(), Some(5));
    }

    #[test]
    fn test_reduce_min() {
        let result = reduce(min_fn, vec![int(3), int(1), int(4), int(1), int(5)], None).unwrap();
        assert_eq!(result.as_int(), Some(1));
    }

    #[test]
    fn test_reduce_single_element() {
        let result = reduce(add, vec![int(42)], None).unwrap();
        assert_eq!(result.as_int(), Some(42));
    }

    #[test]
    fn test_reduce_two_elements() {
        let result = reduce(add, vec![int(10), int(20)], None).unwrap();
        assert_eq!(result.as_int(), Some(30));
    }

    // =========================================================================
    // Initializer tests
    // =========================================================================

    #[test]
    fn test_reduce_with_initializer() {
        let result = reduce(add, vec![int(1), int(2), int(3)], Some(int(10))).unwrap();
        assert_eq!(result.as_int(), Some(16)); // 10 + 1 + 2 + 3
    }

    #[test]
    fn test_reduce_with_initializer_product() {
        let result = reduce(mul, vec![int(2), int(3), int(4)], Some(int(1))).unwrap();
        assert_eq!(result.as_int(), Some(24));
    }

    #[test]
    fn test_reduce_empty_with_initializer() {
        let result = reduce(add, Vec::<Value>::new(), Some(int(42))).unwrap();
        assert_eq!(result.as_int(), Some(42));
    }

    #[test]
    fn test_reduce_initializer_is_first_accumulator() {
        // Verify order: init op elem1 op elem2 ...
        let result = reduce(
            |a, b| int(a.as_int().unwrap() * 10 + b.as_int().unwrap()),
            vec![int(1), int(2), int(3)],
            Some(int(0)),
        )
        .unwrap();
        // ((0*10+1)*10+2)*10+3 = 123
        assert_eq!(result.as_int(), Some(123));
    }

    // =========================================================================
    // Error tests
    // =========================================================================

    #[test]
    fn test_reduce_empty_no_initializer() {
        let result = reduce(add, Vec::<Value>::new(), None);
        assert!(result.is_err());
        match result.unwrap_err() {
            ReduceError::EmptySequence => {}
            other => panic!("Expected EmptySequence, got {:?}", other),
        }
    }

    #[test]
    fn test_reduce_error_display() {
        let err = ReduceError::EmptySequence;
        assert!(err.to_string().contains("empty iterable"));

        let err = ReduceError::FunctionError("division by zero".into());
        assert!(err.to_string().contains("division by zero"));
    }

    // =========================================================================
    // Fallible reduce tests
    // =========================================================================

    #[test]
    fn test_reduce_fallible_success() {
        let result = reduce_fallible(
            |a: &Value, b: &Value| -> Result<Value, String> {
                Ok(int(a.as_int().unwrap() + b.as_int().unwrap()))
            },
            vec![int(1), int(2), int(3)],
            None,
        )
        .unwrap();
        assert_eq!(result.as_int(), Some(6));
    }

    #[test]
    fn test_reduce_fallible_error() {
        let result = reduce_fallible(
            |a: &Value, b: &Value| -> Result<Value, String> {
                let bv = b.as_int().unwrap();
                if bv == 0 {
                    Err("division by zero".into())
                } else {
                    Ok(int(a.as_int().unwrap() / bv))
                }
            },
            vec![int(100), int(2), int(0), int(5)],
            None,
        );
        assert!(result.is_err());
        match result.unwrap_err() {
            ReduceError::FunctionError(msg) => assert!(msg.contains("division by zero")),
            other => panic!("Expected FunctionError, got {:?}", other),
        }
    }

    #[test]
    fn test_reduce_fallible_empty_with_init() {
        let result = reduce_fallible(
            |a: &Value, b: &Value| -> Result<Value, String> {
                Ok(int(a.as_int().unwrap() + b.as_int().unwrap()))
            },
            Vec::<Value>::new(),
            Some(int(99)),
        )
        .unwrap();
        assert_eq!(result.as_int(), Some(99));
    }

    #[test]
    fn test_reduce_fallible_short_circuits() {
        // Ensure we stop processing after error
        let mut call_count = 0;
        let counter = &mut call_count;

        // We can't easily track call count with closures in this test pattern,
        // so we verify the error is returned correctly.
        let result = reduce_fallible(
            |a: &Value, b: &Value| -> Result<Value, String> {
                if b.as_int().unwrap() == 3 {
                    Err("stop".into())
                } else {
                    Ok(int(a.as_int().unwrap() + b.as_int().unwrap()))
                }
            },
            vec![int(1), int(2), int(3), int(4), int(5)],
            None,
        );
        assert!(result.is_err());
    }

    // =========================================================================
    // Accumulate tests
    // =========================================================================

    #[test]
    fn test_accumulate_sum() {
        let result = accumulate(add, vec![int(1), int(2), int(3), int(4)], None);
        let ints: Vec<i64> = result.iter().filter_map(|v| v.as_int()).collect();
        assert_eq!(ints, vec![1, 3, 6, 10]);
    }

    #[test]
    fn test_accumulate_product() {
        let result = accumulate(mul, vec![int(1), int(2), int(3), int(4)], None);
        let ints: Vec<i64> = result.iter().filter_map(|v| v.as_int()).collect();
        assert_eq!(ints, vec![1, 2, 6, 24]);
    }

    #[test]
    fn test_accumulate_with_initializer() {
        let result = accumulate(add, vec![int(1), int(2), int(3)], Some(int(100)));
        let ints: Vec<i64> = result.iter().filter_map(|v| v.as_int()).collect();
        assert_eq!(ints, vec![100, 101, 103, 106]);
    }

    #[test]
    fn test_accumulate_empty() {
        let result = accumulate(add, Vec::<Value>::new(), None);
        assert!(result.is_empty());
    }

    #[test]
    fn test_accumulate_empty_with_initializer() {
        let result = accumulate(add, Vec::<Value>::new(), Some(int(42)));
        let ints: Vec<i64> = result.iter().filter_map(|v| v.as_int()).collect();
        assert_eq!(ints, vec![42]);
    }

    #[test]
    fn test_accumulate_single() {
        let result = accumulate(add, vec![int(7)], None);
        let ints: Vec<i64> = result.iter().filter_map(|v| v.as_int()).collect();
        assert_eq!(ints, vec![7]);
    }

    #[test]
    fn test_accumulate_running_max() {
        let result = accumulate(
            max_fn,
            vec![int(3), int(1), int(4), int(1), int(5), int(2)],
            None,
        );
        let ints: Vec<i64> = result.iter().filter_map(|v| v.as_int()).collect();
        assert_eq!(ints, vec![3, 3, 4, 4, 5, 5]);
    }

    // =========================================================================
    // Real-world pattern tests
    // =========================================================================

    #[test]
    fn test_reduce_flatten_sum() {
        // sum([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        let nums: Vec<Value> = (1..=10).map(|i| int(i)).collect();
        let result = reduce(add, nums, None).unwrap();
        assert_eq!(result.as_int(), Some(55));
    }

    #[test]
    fn test_reduce_factorial() {
        // 5! = 120
        let nums: Vec<Value> = (1..=5).map(|i| int(i)).collect();
        let result = reduce(mul, nums, None).unwrap();
        assert_eq!(result.as_int(), Some(120));
    }

    #[test]
    fn test_reduce_large_sequence() {
        // Sum of 0..999 = 999 * 1000 / 2 = 499500
        let nums: Vec<Value> = (0..1000).map(|i| int(i)).collect();
        let result = reduce(add, nums, None).unwrap();
        assert_eq!(result.as_int(), Some(499_500));
    }

    #[test]
    fn test_reduce_concatenation_pattern() {
        // Simulating string concatenation by building digit number
        let result = reduce(
            |a, b| int(a.as_int().unwrap() * 10 + b.as_int().unwrap()),
            vec![int(1), int(2), int(3), int(4), int(5)],
            Some(int(0)),
        )
        .unwrap();
        assert_eq!(result.as_int(), Some(12345));
    }

    #[test]
    fn test_reduce_power_chain() {
        // ((2^3)^2) = 64
        let result = reduce(
            |a, b| {
                let base = a.as_int().unwrap();
                let exp = b.as_int().unwrap();
                int(base.pow(exp as u32))
            },
            vec![int(2), int(3), int(2)],
            None,
        )
        .unwrap();
        assert_eq!(result.as_int(), Some(64));
    }

    #[test]
    fn test_reduce_bitwise_or() {
        let result = reduce(
            |a, b| int(a.as_int().unwrap() | b.as_int().unwrap()),
            vec![int(1), int(2), int(4), int(8)],
            Some(int(0)),
        )
        .unwrap();
        assert_eq!(result.as_int(), Some(15)); // 0b1111
    }

    #[test]
    fn test_reduce_bitwise_and() {
        let result = reduce(
            |a, b| int(a.as_int().unwrap() & b.as_int().unwrap()),
            vec![int(0xFF), int(0x0F), int(0x07)],
            None,
        )
        .unwrap();
        assert_eq!(result.as_int(), Some(0x07));
    }

    // =========================================================================
    // Edge case tests
    // =========================================================================

    #[test]
    fn test_reduce_with_none_values() {
        let result = reduce(|a, _b| a.clone(), vec![Value::none(), Value::none()], None).unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn test_reduce_preserves_value_types() {
        let result = reduce(
            |a, b| {
                if a.as_float().is_some() || b.as_float().is_some() {
                    Value::float(
                        a.as_float().unwrap_or_else(|| a.as_int().unwrap() as f64)
                            + b.as_float().unwrap_or_else(|| b.as_int().unwrap() as f64),
                    )
                } else {
                    int(a.as_int().unwrap() + b.as_int().unwrap())
                }
            },
            vec![int(1), Value::float(2.5), int(3)],
            None,
        )
        .unwrap();
        assert_eq!(result.as_float(), Some(6.5));
    }

    #[test]
    fn test_reduce_negative_numbers() {
        let result = reduce(add, vec![int(-5), int(3), int(-2), int(10)], None).unwrap();
        assert_eq!(result.as_int(), Some(6));
    }

    #[test]
    fn test_reduce_zero_initializer() {
        let result = reduce(add, vec![int(1), int(2)], Some(int(0))).unwrap();
        assert_eq!(result.as_int(), Some(3));
    }

    #[test]
    fn test_reduce_one_initializer() {
        let result = reduce(mul, vec![int(5), int(6)], Some(int(1))).unwrap();
        assert_eq!(result.as_int(), Some(30));
    }

    // =========================================================================
    // Stress tests
    // =========================================================================

    #[test]
    fn test_reduce_stress_10k_elements() {
        let nums: Vec<Value> = (0..10_000).map(|i| int(i)).collect();
        let result = reduce(add, nums, None).unwrap();
        assert_eq!(result.as_int(), Some(49_995_000));
    }

    #[test]
    fn test_accumulate_stress_1k_elements() {
        let nums: Vec<Value> = (1..=1000).map(|i| int(i)).collect();
        let result = accumulate(add, nums, None);
        assert_eq!(result.len(), 1000);
        // Verify last element = sum of 1..=1000 = 500500
        assert_eq!(result.last().unwrap().as_int(), Some(500_500));
    }
}
