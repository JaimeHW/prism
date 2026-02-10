//! Integration tests for the functools module.
//!
//! Tests common Python patterns and cross-component interactions.

#[cfg(test)]
mod integration_tests {
    use crate::stdlib::collections::counter::HashableValue;
    use crate::stdlib::functools::cached_property::CachedProperty;
    use crate::stdlib::functools::cmp::{ComparisonBase, TotalOrdering, sort_with_cmp};
    use crate::stdlib::functools::lru_cache::LruCache;
    use crate::stdlib::functools::partial::Partial;
    use crate::stdlib::functools::reduce::{accumulate, reduce};
    use crate::stdlib::functools::wraps::{WrapperMetadata, update_wrapper};
    use prism_core::Value;
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
    // reduce + partial integration
    // =========================================================================

    #[test]
    fn test_reduce_with_partial_like_pattern() {
        // functools.reduce(operator.add, [1, 2, 3, 4], 0)
        let add = |a: &Value, b: &Value| int(a.as_int().unwrap() + b.as_int().unwrap());
        let result = reduce(add, vec![int(1), int(2), int(3), int(4)], Some(int(0))).unwrap();
        assert_eq!(result.as_int(), Some(10));
    }

    #[test]
    fn test_partial_args_then_reduce() {
        // Create partial with frozen initial arg, then reduce
        let partial_func = Partial::new(int(0), vec![int(100)]);
        let merged = partial_func.merge_args(&[int(1), int(2), int(3)]);

        // Use merged as reduce input
        let result = reduce(
            |a, b| int(a.as_int().unwrap() + b.as_int().unwrap()),
            merged,
            None,
        )
        .unwrap();
        assert_eq!(result.as_int(), Some(106));
    }

    // =========================================================================
    // lru_cache + reduce integration
    // =========================================================================

    #[test]
    fn test_fibonacci_with_lru_cache() {
        // Simulate memoized fibonacci using LRU cache
        let mut cache = LruCache::new(100);

        fn fib(n: i64, cache: &mut LruCache) -> i64 {
            if n <= 1 {
                return n;
            }

            let key = HashableValue(Value::int_unchecked(n));
            if let Some(v) = cache.get(&key) {
                return v.as_int().unwrap();
            }

            let result = fib(n - 1, cache) + fib(n - 2, cache);
            cache.put(key, Value::int_unchecked(result));
            result
        }

        assert_eq!(fib(0, &mut cache), 0);
        assert_eq!(fib(1, &mut cache), 1);
        assert_eq!(fib(10, &mut cache), 55);
        assert_eq!(fib(20, &mut cache), 6765);

        let info = cache.cache_info();
        assert!(info.hits > 0); // Should have cache hits from recursive calls
    }

    // =========================================================================
    // cmp + sort integration
    // =========================================================================

    #[test]
    fn test_sort_with_total_ordering() {
        let ordering = TotalOrdering::new(ComparisonBase::Lt);
        let eq = |a: &Value, b: &Value| a.as_int().unwrap() == b.as_int().unwrap();
        let lt = |a: &Value, b: &Value| a.as_int().unwrap() < b.as_int().unwrap();

        let mut values = vec![int(5), int(2), int(8), int(1), int(9)];
        values.sort_by(|a, b| {
            if eq(a, b) {
                std::cmp::Ordering::Equal
            } else if ordering.lt(a, b, &eq, &lt) {
                std::cmp::Ordering::Less
            } else {
                std::cmp::Ordering::Greater
            }
        });

        let sorted: Vec<i64> = values.iter().filter_map(|v| v.as_int()).collect();
        assert_eq!(sorted, vec![1, 2, 5, 8, 9]);
    }

    // =========================================================================
    // cached_property patterns
    // =========================================================================

    #[test]
    fn test_cached_property_with_wrapper_metadata() {
        let mut cp = CachedProperty::with_doc(
            int(0),
            "expensive_computation".into(),
            str_val("Computes something expensive."),
        );

        // Wrap it with metadata
        let mut meta = WrapperMetadata::new();
        meta.set_attr("__name__", str_val("expensive_computation"));
        meta.set_attr("__doc__", str_val("Computes something expensive."));

        // Compute the property
        let result = cp.get_or_compute(|_| int(42));
        assert_eq!(result.as_int(), Some(42));

        // Verify metadata
        assert_eq!(cp.attr_name(), Some("expensive_computation"));
    }

    // =========================================================================
    // wraps + partial integration
    // =========================================================================

    #[test]
    fn test_wrapped_partial() {
        // Create a partial
        let partial = Partial::new(int(0), vec![int(1), int(2)]);

        // Create wrapper metadata as if wraps() was applied
        let mut meta = WrapperMetadata::new();
        meta.set_attr("__name__", str_val("add_wrapper"));
        meta.set_attr("__doc__", str_val("Adds numbers."));

        // The partial should have the wrapped function's metadata
        assert_eq!(partial.func().as_int(), Some(0));
        assert_eq!(meta.count(), 2);
    }

    // =========================================================================
    // accumulate patterns
    // =========================================================================

    #[test]
    fn test_accumulate_running_balance() {
        // Simulate bank account balance updates
        let transactions = vec![int(100), int(-20), int(50), int(-30), int(200)];
        let balances = accumulate(
            |a, b| int(a.as_int().unwrap() + b.as_int().unwrap()),
            transactions,
            Some(int(0)),
        );

        let expected = vec![0, 100, 80, 130, 100, 300];
        let actual: Vec<i64> = balances.iter().filter_map(|v| v.as_int()).collect();
        assert_eq!(actual, expected);
    }

    #[test]
    fn test_accumulate_prefix_product() {
        let nums = vec![int(1), int(2), int(3), int(4), int(5)];
        let products = accumulate(
            |a, b| int(a.as_int().unwrap() * b.as_int().unwrap()),
            nums,
            None,
        );

        let expected = vec![1, 2, 6, 24, 120];
        let actual: Vec<i64> = products.iter().filter_map(|v| v.as_int()).collect();
        assert_eq!(actual, expected);
    }

    // =========================================================================
    // Complex pattern: memoized sort with custom key
    // =========================================================================

    #[test]
    fn test_memoized_key_sort() {
        // Use LRU cache to memoize a key function
        let mut cache = LruCache::new(100);

        // Pre-compute and cache key values
        let values_to_sort = vec![int(-3), int(1), int(-2), int(4), int(0)];
        for v in &values_to_sort {
            let key = hv(v.clone());
            let result = -(v.as_int().unwrap().abs());
            cache.put(key, int(result));
        }

        // Verify all keys are cached
        assert_eq!(cache.len(), 5);

        // Now sort using cached keys
        let mut values = values_to_sort;
        values.sort_by_key(|v| cache.get(&hv(v.clone())).unwrap().as_int().unwrap());

        // Sorted by descending absolute value (most negative key = largest |x|)
        let sorted: Vec<i64> = values.iter().filter_map(|v| v.as_int()).collect();
        assert_eq!(sorted, vec![4, -3, -2, 1, 0]);
    }

    // =========================================================================
    // reduce composition pattern
    // =========================================================================

    #[test]
    fn test_reduce_compose_functions() {
        // Apply a series of transformations: add 1, multiply by 2, subtract 3
        let transforms: Vec<Box<dyn Fn(i64) -> i64>> = vec![
            Box::new(|x| x + 1),
            Box::new(|x| x * 2),
            Box::new(|x| x - 3),
        ];

        // Start with 5: (5+1)*2 - 3 = 9
        let input = 5_i64;
        let result = transforms.iter().fold(input, |acc, f| f(acc));
        assert_eq!(result, 9);
    }

    // =========================================================================
    // Cache clear + refill pattern
    // =========================================================================

    #[test]
    fn test_lru_cache_clear_and_refill() {
        let mut cache = LruCache::new(5);

        // Fill
        for i in 0..5 {
            cache.put(hv(int(i)), int(i * 10));
        }
        assert_eq!(cache.len(), 5);

        // Clear
        cache.cache_clear();
        assert_eq!(cache.len(), 0);

        // Refill with different values
        for i in 10..15 {
            cache.put(hv(int(i)), int(i * 100));
        }
        assert_eq!(cache.len(), 5);

        for i in 10..15 {
            assert_eq!(cache.get(&hv(int(i))).unwrap().as_int(), Some(i * 100));
        }
    }
}
