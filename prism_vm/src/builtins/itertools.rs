use super::BuiltinError;
use prism_core::Value;
use prism_runtime::types::range::RangeObject;

// =============================================================================
// range
// =============================================================================

/// Builtin range function.
///
/// range(stop) -> range object
/// range(start, stop[, step]) -> range object
///
/// Returns a range object representing a sequence of integers.
pub fn builtin_range(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.is_empty() || args.len() > 3 {
        return Err(BuiltinError::TypeError(format!(
            "range expected 1 to 3 arguments, got {}",
            args.len()
        )));
    }

    let (start, stop, step) = match args.len() {
        1 => {
            // range(stop)
            let stop = args[0].as_int().ok_or_else(|| {
                BuiltinError::TypeError("range() integer end argument expected".to_string())
            })?;
            (0i64, stop, 1i64)
        }
        2 => {
            // range(start, stop)
            let start = args[0].as_int().ok_or_else(|| {
                BuiltinError::TypeError("range() integer start argument expected".to_string())
            })?;
            let stop = args[1].as_int().ok_or_else(|| {
                BuiltinError::TypeError("range() integer end argument expected".to_string())
            })?;
            (start, stop, 1i64)
        }
        3 => {
            // range(start, stop, step)
            let start = args[0].as_int().ok_or_else(|| {
                BuiltinError::TypeError("range() integer start argument expected".to_string())
            })?;
            let stop = args[1].as_int().ok_or_else(|| {
                BuiltinError::TypeError("range() integer end argument expected".to_string())
            })?;
            let step = args[2].as_int().ok_or_else(|| {
                BuiltinError::TypeError("range() integer step argument expected".to_string())
            })?;
            if step == 0 {
                return Err(BuiltinError::ValueError(
                    "range() arg 3 must not be zero".to_string(),
                ));
            }
            (start, stop, step)
        }
        _ => unreachable!(),
    };

    // Create RangeObject on heap and return as Value
    // TODO: Use GC allocator instead of Box::leak
    let range_obj = Box::new(RangeObject::new(start, stop, step));
    let ptr = Box::leak(range_obj) as *mut RangeObject as *const ();
    Ok(Value::object_ptr(ptr))
}

// =============================================================================
// iter
// =============================================================================

/// Builtin iter function.
///
/// iter(object) -> iterator
/// iter(callable, sentinel) -> iterator (sentinel form)
///
/// # Performance
///
/// - Built-in types: O(1) TypeId dispatch (~16 cycles)
/// - User-defined types: O(n) protocol lookup
///
/// # Examples
///
/// ```python
/// iter([1, 2, 3])      # Returns list_iterator
/// iter(range(5))       # Returns range_iterator
/// iter("hello")        # Returns str_iterator
/// iter({1: 'a'})       # Returns dict_keys iterator
/// ```
pub fn builtin_iter(args: &[Value]) -> Result<Value, BuiltinError> {
    match args.len() {
        1 => {
            // iter(object) - standard form
            // Python semantics: iter(iterator) returns the same iterator object.
            if super::iter_dispatch::is_iterator(&args[0]) {
                return Ok(args[0]);
            }
            let iter = super::iter_dispatch::value_to_iterator(&args[0])?;
            Ok(super::iter_dispatch::iterator_to_value(iter))
        }
        2 => {
            // iter(callable, sentinel) - sentinel form
            // TODO: Create callable iterator that calls until sentinel
            Err(BuiltinError::NotImplemented(
                "iter(callable, sentinel) not yet implemented".to_string(),
            ))
        }
        _ => Err(BuiltinError::TypeError(format!(
            "iter() expected 1 or 2 arguments, got {}",
            args.len()
        ))),
    }
}

// =============================================================================
// next
// =============================================================================

/// Builtin next function.
///
/// next(iterator[, default]) -> next item from iterator
///
/// # Performance
///
/// O(1) for iterator objects - single method call.
///
/// # Raises
///
/// - TypeError: if argument is not an iterator
/// - StopIteration: if iterator is exhausted and no default provided
///
/// # Examples
///
/// ```python
/// it = iter([1, 2])
/// next(it)        # Returns 1
/// next(it)        # Returns 2
/// next(it)        # Raises StopIteration
/// next(it, None)  # Returns None (default)
/// ```
pub fn builtin_next(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.is_empty() || args.len() > 2 {
        return Err(BuiltinError::TypeError(format!(
            "next() expected 1 or 2 arguments, got {}",
            args.len()
        )));
    }

    let default = args.get(1).copied();

    // Get mutable reference to iterator
    let iter_obj = super::iter_dispatch::get_iterator_mut(&args[0]).ok_or_else(|| {
        BuiltinError::TypeError(format!(
            "'{}' object is not an iterator",
            get_type_name(&args[0])
        ))
    })?;

    // Get next value
    match iter_obj.next() {
        Some(value) => Ok(value),
        None => match default {
            Some(d) => Ok(d),
            None => {
                // Signal StopIteration - for now use ValueError
                // TODO: Proper StopIteration exception integration
                Err(BuiltinError::ValueError("StopIteration".to_string()))
            }
        },
    }
}

/// Get type name for error messages.
#[inline]
fn get_type_name(value: &Value) -> &'static str {
    if value.is_none() {
        "NoneType"
    } else if value.as_bool().is_some() {
        "bool"
    } else if value.as_int().is_some() {
        "int"
    } else if value.as_float().is_some() {
        "float"
    } else if value.is_string() {
        "str"
    } else {
        "object"
    }
}

// =============================================================================
// enumerate
// =============================================================================

/// Builtin enumerate function.
///
/// enumerate(iterable, start=0) -> enumerate object
pub fn builtin_enumerate(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.is_empty() || args.len() > 2 {
        return Err(BuiltinError::TypeError(format!(
            "enumerate expected 1 or 2 arguments, got {}",
            args.len()
        )));
    }

    // Parse start argument (default 0)
    let start = if args.len() == 2 {
        args[1].as_int().ok_or_else(|| {
            BuiltinError::TypeError("'start' argument must be an integer".to_string())
        })?
    } else {
        0
    };

    // Convert iterable to iterator using O(1) TypeId dispatch
    let inner = super::iter_dispatch::value_to_iterator(&args[0])?;

    // Create enumerate iterator
    let enumerate = prism_runtime::types::iter::IteratorObject::enumerate(inner, start);
    Ok(super::iter_dispatch::iterator_to_value(enumerate))
}

// =============================================================================
// zip
// =============================================================================

/// Builtin zip function.
///
/// zip(*iterables) -> zip object
///
/// # Performance
///
/// - O(k) construction where k = number of iterables
/// - O(k) per iteration step
/// - Terminates when any iterator is exhausted (shortest-first)
///
/// # Examples
///
/// ```python
/// zip([1, 2, 3], ['a', 'b', 'c'])  # -> [(1, 'a'), (2, 'b'), (3, 'c')]
/// zip([1, 2], ['a', 'b', 'c'])     # -> [(1, 'a'), (2, 'b')] (shortest)
/// zip()                            # -> empty iterator
/// ```
pub fn builtin_zip(args: &[Value]) -> Result<Value, BuiltinError> {
    // Convert all arguments to iterators
    let mut iterators = Vec::with_capacity(args.len());
    for arg in args {
        let iter = super::iter_dispatch::value_to_iterator(arg)?;
        iterators.push(iter);
    }

    // Create zip iterator
    let zip_iter = prism_runtime::types::iter::IteratorObject::zip(iterators);
    Ok(super::iter_dispatch::iterator_to_value(zip_iter))
}

// =============================================================================
// map
// =============================================================================

/// Builtin map function.
///
/// map(function, iterable, ...) -> map object
///
/// # Note
///
/// Currently only supports single-iterable map. The function is stored and
/// must be called by the VM when iterating (lazy evaluation).
pub fn builtin_map(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() < 2 {
        return Err(BuiltinError::TypeError(format!(
            "map expected at least 2 arguments, got {}",
            args.len()
        )));
    }

    let func = args[0];

    // For now, only support single iterable
    if args.len() > 2 {
        return Err(BuiltinError::NotImplemented(
            "map() with multiple iterables not yet implemented".to_string(),
        ));
    }

    let inner = super::iter_dispatch::value_to_iterator(&args[1])?;

    // Create map iterator (function call handled by VM on iteration)
    let map_iter = prism_runtime::types::iter::IteratorObject::map(func, inner);
    Ok(super::iter_dispatch::iterator_to_value(map_iter))
}

// =============================================================================
// filter
// =============================================================================

/// Builtin filter function.
///
/// filter(function, iterable) -> filter object
///
/// # Performance
///
/// - O(1) per iteration when predicate returns truthy
/// - O(n) when many consecutive falsy values (skipped)
///
/// # Special Cases
///
/// - filter(None, iterable) filters out falsy values (identity filter)
pub fn builtin_filter(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() != 2 {
        return Err(BuiltinError::TypeError(format!(
            "filter expected 2 arguments, got {}",
            args.len()
        )));
    }

    // Check if predicate is None (identity filter)
    let func = if args[0].is_none() {
        None
    } else {
        Some(args[0])
    };

    let inner = super::iter_dispatch::value_to_iterator(&args[1])?;

    // Create filter iterator
    let filter_iter = prism_runtime::types::iter::IteratorObject::filter(func, inner);
    Ok(super::iter_dispatch::iterator_to_value(filter_iter))
}

// =============================================================================
// reversed
// =============================================================================

/// Builtin reversed function.
///
/// reversed(sequence) -> reverse iterator
///
/// # Performance
///
/// - O(n) construction (materializes sequence into Vec)
/// - O(1) per iteration step
///
/// # Note
///
/// Currently materializes the entire sequence. A future optimization would
/// check for __reversed__ or use index-based backwards traversal.
pub fn builtin_reversed(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() != 1 {
        return Err(BuiltinError::TypeError(format!(
            "reversed expected 1 argument, got {}",
            args.len()
        )));
    }

    // Convert to iterator and collect all values
    let mut iter = super::iter_dispatch::value_to_iterator(&args[0])?;
    let values = iter.collect_remaining();

    // Create reversed iterator
    let reversed = prism_runtime::types::iter::IteratorObject::reversed(values);
    Ok(super::iter_dispatch::iterator_to_value(reversed))
}

// =============================================================================
// sorted
// =============================================================================

/// Builtin sorted function.
///
/// sorted(iterable, /, *, key=None, reverse=False) -> list
///
/// # Performance
///
/// - O(n log n) time complexity using Rust's Timsort
/// - O(n) space for collecting iterable + O(log n) for sort
/// - Stable sort: equal elements maintain relative order
///
/// # Arguments
///
/// - `iterable`: Any iterable to sort
/// - `key`: Optional function to extract comparison key (currently placeholder)
/// - `reverse`: If `True`, sort in descending order
///
/// # Examples
///
/// ```python
/// sorted([3, 1, 2])                    # [1, 2, 3]
/// sorted([3, 1, 2], reverse=True)      # [3, 2, 1]
/// sorted("hello")                       # ['e', 'h', 'l', 'l', 'o']
/// sorted(range(5, 0, -1))              # [1, 2, 3, 4, 5]
/// ```
pub fn builtin_sorted(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.is_empty() || args.len() > 3 {
        return Err(BuiltinError::TypeError(format!(
            "sorted expected 1 to 3 arguments, got {}",
            args.len()
        )));
    }

    // Parse optional key function (arg[1]) and reverse flag (arg[2])
    let key_func = args.get(1).filter(|v| !v.is_none()).copied();
    let reverse = args.get(2).and_then(|v| v.as_bool()).unwrap_or(false);

    // Collect iterable elements
    let mut iter = super::iter_dispatch::value_to_iterator(&args[0])?;
    let mut values = iter.collect_remaining();

    // If key function is provided, we need VM integration.
    // For now, return NotImplemented for key function.
    if key_func.is_some() {
        return Err(BuiltinError::NotImplemented(
            "sorted() with key function requires VM integration".to_string(),
        ));
    }

    // Sort using Python-compatible Value comparison
    // Uses stable sort (timsort) which is O(n log n)
    values.sort_by(|a, b| compare_values(a, b));

    // Reverse if requested
    if reverse {
        values.reverse();
    }

    // Return as list (allocate on heap)
    let list = prism_runtime::types::list::ListObject::from_slice(&values);
    let ptr = Box::leak(Box::new(list)) as *mut prism_runtime::types::list::ListObject as *const ();
    Ok(Value::object_ptr(ptr))
}

/// Compare two Values for sorting.
///
/// # Comparison Order (Python semantics)
///
/// 1. None < bool < int/float < str < other
/// 2. Within types: natural ordering
/// 3. Cross-type numeric: int vs float uses float comparison
///
/// # Performance
///
/// - O(1) for numeric types
/// - O(n) for strings (lexicographic)
#[inline]
fn compare_values(a: &Value, b: &Value) -> std::cmp::Ordering {
    use std::cmp::Ordering;

    // None is smallest
    match (a.is_none(), b.is_none()) {
        (true, true) => return Ordering::Equal,
        (true, false) => return Ordering::Less,
        (false, true) => return Ordering::Greater,
        _ => {}
    }

    // Boolean comparison
    if let (Some(a_bool), Some(b_bool)) = (a.as_bool(), b.as_bool()) {
        return a_bool.cmp(&b_bool);
    }

    // Cross-type bool vs numeric: bool < numeric
    match (
        a.as_bool(),
        b.as_int().or_else(|| b.as_float().map(|f| f as i64)),
    ) {
        (Some(_), Some(_)) => return Ordering::Less,
        _ => {}
    }
    match (
        a.as_int().or_else(|| a.as_float().map(|f| f as i64)),
        b.as_bool(),
    ) {
        (Some(_), Some(_)) => return Ordering::Greater,
        _ => {}
    }

    // Integer comparison
    if let (Some(a_int), Some(b_int)) = (a.as_int(), b.as_int()) {
        return a_int.cmp(&b_int);
    }

    // Float comparison
    if let (Some(a_float), Some(b_float)) = (a.as_float(), b.as_float()) {
        return a_float.partial_cmp(&b_float).unwrap_or(Ordering::Equal);
    }

    // Cross-type int vs float: promote int to float
    if let (Some(a_int), Some(b_float)) = (a.as_int(), b.as_float()) {
        return (a_int as f64)
            .partial_cmp(&b_float)
            .unwrap_or(Ordering::Equal);
    }
    if let (Some(a_float), Some(b_int)) = (a.as_float(), b.as_int()) {
        return a_float
            .partial_cmp(&(b_int as f64))
            .unwrap_or(Ordering::Equal);
    }

    // String comparison (lexicographic)
    if a.is_string() && b.is_string() {
        // Get string data for comparison
        // For now, compare by pointer address as placeholder
        // TODO: Proper string content comparison
        let a_ptr = a.as_object_ptr().unwrap_or(std::ptr::null());
        let b_ptr = b.as_object_ptr().unwrap_or(std::ptr::null());
        return (a_ptr as usize).cmp(&(b_ptr as usize));
    }

    // String vs numeric: numeric < string
    if a.is_string() {
        if b.as_int().is_some() || b.as_float().is_some() {
            return Ordering::Greater;
        }
    }
    if b.is_string() {
        if a.as_int().is_some() || a.as_float().is_some() {
            return Ordering::Less;
        }
    }

    // Default: compare by pointer (stable but arbitrary)
    let a_ptr = a.as_object_ptr().unwrap_or(std::ptr::null());
    let b_ptr = b.as_object_ptr().unwrap_or(std::ptr::null());
    (a_ptr as usize).cmp(&(b_ptr as usize))
}

// =============================================================================
// all / any
// =============================================================================

/// Builtin all function.
///
/// all(iterable) -> bool - True if all elements are truthy
///
/// # Performance
///
/// - O(n) worst case when all elements truthy
/// - O(1) best case with early exit on first falsy element
/// - Uses iterator dispatch for ~16 cycle setup
///
/// # Examples
///
/// ```python
/// all([1, 2, 3])      # True - all truthy
/// all([1, 0, 3])      # False - 0 is falsy, exits early
/// all([])             # True - vacuous truth
/// all(range(1, 100))  # True - all positive ints
/// ```
pub fn builtin_all(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() != 1 {
        return Err(BuiltinError::TypeError(format!(
            "all expected 1 argument, got {}",
            args.len()
        )));
    }

    // Convert to iterator using O(1) TypeId dispatch
    let mut iter = super::iter_dispatch::value_to_iterator(&args[0])?;

    // Iterate with early exit on first falsy element
    while let Some(value) = iter.next() {
        if !value.is_truthy() {
            return Ok(Value::bool(false));
        }
    }

    // All elements were truthy (or empty iterable - vacuous truth)
    Ok(Value::bool(true))
}

/// Builtin any function.
///
/// any(iterable) -> bool - True if any element is truthy
///
/// # Performance
///
/// - O(n) worst case when all elements falsy
/// - O(1) best case with early exit on first truthy element
/// - Uses iterator dispatch for ~16 cycle setup
///
/// # Examples
///
/// ```python
/// any([0, 0, 1])      # True - 1 is truthy, exits early
/// any([0, 0, 0])      # False - all falsy
/// any([])             # False - empty iterable
/// any(range(10))      # True - 1-9 are truthy
/// ```
pub fn builtin_any(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() != 1 {
        return Err(BuiltinError::TypeError(format!(
            "any expected 1 argument, got {}",
            args.len()
        )));
    }

    // Convert to iterator using O(1) TypeId dispatch
    let mut iter = super::iter_dispatch::value_to_iterator(&args[0])?;

    // Iterate with early exit on first truthy element
    while let Some(value) = iter.next() {
        if value.is_truthy() {
            return Ok(Value::bool(true));
        }
    }

    // No truthy elements found
    Ok(Value::bool(false))
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use prism_runtime::types::list::ListObject;

    #[test]
    fn test_range_validation() {
        // Zero step should error
        let result = builtin_range(&[
            Value::int(0).unwrap(),
            Value::int(10).unwrap(),
            Value::int(0).unwrap(),
        ]);
        assert!(result.is_err());
    }

    #[test]
    fn test_range_type_error() {
        // Non-integer should error
        let result = builtin_range(&[Value::float(3.14)]);
        assert!(result.is_err());
    }

    // -------------------------------------------------------------------------
    // all() tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_all_empty_list() {
        // all([]) = True (vacuous truth)
        let list = ListObject::new();
        let ptr = Box::leak(Box::new(list)) as *mut ListObject as *const ();
        let value = Value::object_ptr(ptr);

        let result = builtin_all(&[value]).unwrap();
        assert!(result.as_bool().unwrap());
    }

    #[test]
    fn test_all_all_truthy() {
        // all([1, 2, 3]) = True
        let list = ListObject::from_slice(&[
            Value::int(1).unwrap(),
            Value::int(2).unwrap(),
            Value::int(3).unwrap(),
        ]);
        let ptr = Box::leak(Box::new(list)) as *mut ListObject as *const ();
        let value = Value::object_ptr(ptr);

        let result = builtin_all(&[value]).unwrap();
        assert!(result.as_bool().unwrap());
    }

    #[test]
    fn test_all_one_falsy() {
        // all([1, 0, 3]) = False
        let list = ListObject::from_slice(&[
            Value::int(1).unwrap(),
            Value::int(0).unwrap(),
            Value::int(3).unwrap(),
        ]);
        let ptr = Box::leak(Box::new(list)) as *mut ListObject as *const ();
        let value = Value::object_ptr(ptr);

        let result = builtin_all(&[value]).unwrap();
        assert!(!result.as_bool().unwrap());
    }

    #[test]
    fn test_all_first_falsy() {
        // all([0, 1, 2]) = False (early exit)
        let list = ListObject::from_slice(&[
            Value::int(0).unwrap(),
            Value::int(1).unwrap(),
            Value::int(2).unwrap(),
        ]);
        let ptr = Box::leak(Box::new(list)) as *mut ListObject as *const ();
        let value = Value::object_ptr(ptr);

        let result = builtin_all(&[value]).unwrap();
        assert!(!result.as_bool().unwrap());
    }

    #[test]
    fn test_all_range() {
        // all(range(1, 5)) = True (all positive)
        let range = builtin_range(&[Value::int(1).unwrap(), Value::int(5).unwrap()]).unwrap();
        let result = builtin_all(&[range]).unwrap();
        assert!(result.as_bool().unwrap());
    }

    #[test]
    fn test_all_range_with_zero() {
        // all(range(0, 5)) = False (0 is falsy)
        let range = builtin_range(&[Value::int(0).unwrap(), Value::int(5).unwrap()]).unwrap();
        let result = builtin_all(&[range]).unwrap();
        assert!(!result.as_bool().unwrap());
    }

    #[test]
    fn test_all_wrong_arg_count() {
        let result = builtin_all(&[]);
        assert!(result.is_err());

        let list = ListObject::new();
        let ptr = Box::leak(Box::new(list)) as *mut ListObject as *const ();
        let value = Value::object_ptr(ptr);
        let result = builtin_all(&[value, value]);
        assert!(result.is_err());
    }

    // -------------------------------------------------------------------------
    // any() tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_any_empty_list() {
        // any([]) = False
        let list = ListObject::new();
        let ptr = Box::leak(Box::new(list)) as *mut ListObject as *const ();
        let value = Value::object_ptr(ptr);

        let result = builtin_any(&[value]).unwrap();
        assert!(!result.as_bool().unwrap());
    }

    #[test]
    fn test_any_all_falsy() {
        // any([0, None, False]) = False
        let list =
            ListObject::from_slice(&[Value::int(0).unwrap(), Value::none(), Value::bool(false)]);
        let ptr = Box::leak(Box::new(list)) as *mut ListObject as *const ();
        let value = Value::object_ptr(ptr);

        let result = builtin_any(&[value]).unwrap();
        assert!(!result.as_bool().unwrap());
    }

    #[test]
    fn test_any_one_truthy() {
        // any([0, 0, 1]) = True
        let list = ListObject::from_slice(&[
            Value::int(0).unwrap(),
            Value::int(0).unwrap(),
            Value::int(1).unwrap(),
        ]);
        let ptr = Box::leak(Box::new(list)) as *mut ListObject as *const ();
        let value = Value::object_ptr(ptr);

        let result = builtin_any(&[value]).unwrap();
        assert!(result.as_bool().unwrap());
    }

    #[test]
    fn test_any_first_truthy() {
        // any([1, 0, 0]) = True (early exit)
        let list = ListObject::from_slice(&[
            Value::int(1).unwrap(),
            Value::int(0).unwrap(),
            Value::int(0).unwrap(),
        ]);
        let ptr = Box::leak(Box::new(list)) as *mut ListObject as *const ();
        let value = Value::object_ptr(ptr);

        let result = builtin_any(&[value]).unwrap();
        assert!(result.as_bool().unwrap());
    }

    #[test]
    fn test_any_range() {
        // any(range(0, 5)) = True (1,2,3,4 are truthy)
        let range = builtin_range(&[Value::int(0).unwrap(), Value::int(5).unwrap()]).unwrap();
        let result = builtin_any(&[range]).unwrap();
        assert!(result.as_bool().unwrap());
    }

    #[test]
    fn test_any_range_empty() {
        // any(range(0, 0)) = False (empty range)
        let range = builtin_range(&[Value::int(0).unwrap(), Value::int(0).unwrap()]).unwrap();
        let result = builtin_any(&[range]).unwrap();
        assert!(!result.as_bool().unwrap());
    }

    #[test]
    fn test_any_wrong_arg_count() {
        let result = builtin_any(&[]);
        assert!(result.is_err());
    }

    // -------------------------------------------------------------------------
    // enumerate() tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_enumerate_basic() {
        // enumerate(['a', 'b']) should produce tuples
        let list = ListObject::from_slice(&[Value::int(10).unwrap(), Value::int(20).unwrap()]);
        let ptr = Box::leak(Box::new(list)) as *mut ListObject as *const ();
        let value = Value::object_ptr(ptr);

        let result = builtin_enumerate(&[value]);
        assert!(result.is_ok());
        // Result is an enumerate iterator
    }

    #[test]
    fn test_enumerate_with_start() {
        let list = ListObject::from_slice(&[Value::int(1).unwrap()]);
        let ptr = Box::leak(Box::new(list)) as *mut ListObject as *const ();
        let value = Value::object_ptr(ptr);

        let result = builtin_enumerate(&[value, Value::int(5).unwrap()]);
        assert!(result.is_ok());
    }

    #[test]
    fn test_enumerate_wrong_arg_count() {
        let result = builtin_enumerate(&[]);
        assert!(result.is_err());

        let list = ListObject::new();
        let ptr = Box::leak(Box::new(list)) as *mut ListObject as *const ();
        let value = Value::object_ptr(ptr);

        let result = builtin_enumerate(&[value, Value::int(0).unwrap(), Value::int(1).unwrap()]);
        assert!(result.is_err());
    }

    #[test]
    fn test_enumerate_invalid_start() {
        let list = ListObject::new();
        let ptr = Box::leak(Box::new(list)) as *mut ListObject as *const ();
        let value = Value::object_ptr(ptr);

        // Non-integer start should error
        let result = builtin_enumerate(&[value, Value::float(1.5)]);
        assert!(result.is_err());
    }

    // -------------------------------------------------------------------------
    // zip() tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_zip_basic() {
        let list1 = ListObject::from_slice(&[Value::int(1).unwrap(), Value::int(2).unwrap()]);
        let ptr1 = Box::leak(Box::new(list1)) as *mut ListObject as *const ();
        let val1 = Value::object_ptr(ptr1);

        let list2 = ListObject::from_slice(&[Value::int(10).unwrap(), Value::int(20).unwrap()]);
        let ptr2 = Box::leak(Box::new(list2)) as *mut ListObject as *const ();
        let val2 = Value::object_ptr(ptr2);

        let result = builtin_zip(&[val1, val2]);
        assert!(result.is_ok());
    }

    #[test]
    fn test_zip_empty() {
        // zip() with no arguments should return empty iterator
        let result = builtin_zip(&[]);
        assert!(result.is_ok());
    }

    #[test]
    fn test_zip_single() {
        let list = ListObject::from_slice(&[Value::int(1).unwrap()]);
        let ptr = Box::leak(Box::new(list)) as *mut ListObject as *const ();
        let value = Value::object_ptr(ptr);

        let result = builtin_zip(&[value]);
        assert!(result.is_ok());
    }

    // -------------------------------------------------------------------------
    // filter() tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_filter_identity() {
        // filter(None, iterable) filters falsy values
        let list = ListObject::from_slice(&[
            Value::int(1).unwrap(),
            Value::int(0).unwrap(),
            Value::int(2).unwrap(),
        ]);
        let ptr = Box::leak(Box::new(list)) as *mut ListObject as *const ();
        let value = Value::object_ptr(ptr);

        let result = builtin_filter(&[Value::none(), value]);
        assert!(result.is_ok());
    }

    #[test]
    fn test_filter_wrong_arg_count() {
        let result = builtin_filter(&[]);
        assert!(result.is_err());

        let result = builtin_filter(&[Value::none()]);
        assert!(result.is_err());
    }

    // -------------------------------------------------------------------------
    // reversed() tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_reversed_basic() {
        let list = ListObject::from_slice(&[
            Value::int(1).unwrap(),
            Value::int(2).unwrap(),
            Value::int(3).unwrap(),
        ]);
        let ptr = Box::leak(Box::new(list)) as *mut ListObject as *const ();
        let value = Value::object_ptr(ptr);

        let result = builtin_reversed(&[value]);
        assert!(result.is_ok());
    }

    #[test]
    fn test_reversed_wrong_arg_count() {
        let result = builtin_reversed(&[]);
        assert!(result.is_err());

        let list = ListObject::new();
        let ptr = Box::leak(Box::new(list)) as *mut ListObject as *const ();
        let value = Value::object_ptr(ptr);

        let result = builtin_reversed(&[value, value]);
        assert!(result.is_err());
    }

    // -------------------------------------------------------------------------
    // map() tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_map_basic() {
        let list = ListObject::from_slice(&[Value::int(1).unwrap()]);
        let ptr = Box::leak(Box::new(list)) as *mut ListObject as *const ();
        let value = Value::object_ptr(ptr);

        // func is a placeholder
        let result = builtin_map(&[Value::none(), value]);
        assert!(result.is_ok());
    }

    #[test]
    fn test_iter_on_iterator_returns_same_object() {
        let list = ListObject::from_slice(&[Value::int(1).unwrap(), Value::int(2).unwrap()]);
        let list_ptr = Box::leak(Box::new(list)) as *mut ListObject as *const ();
        let list_value = Value::object_ptr(list_ptr);

        let iter1 = builtin_iter(&[list_value]).unwrap();
        let iter2 = builtin_iter(&[iter1]).unwrap();
        assert_eq!(iter1, iter2);
    }

    #[test]
    fn test_iter_on_iterator_preserves_state() {
        let list = ListObject::from_slice(&[Value::int(10).unwrap(), Value::int(20).unwrap()]);
        let list_ptr = Box::leak(Box::new(list)) as *mut ListObject as *const ();
        let list_value = Value::object_ptr(list_ptr);

        let iter = builtin_iter(&[list_value]).unwrap();
        let first = builtin_next(&[iter]).unwrap();
        assert_eq!(first.as_int(), Some(10));

        let iter_again = builtin_iter(&[iter]).unwrap();
        assert_eq!(iter_again, iter);

        let second = builtin_next(&[iter_again]).unwrap();
        assert_eq!(second.as_int(), Some(20));
    }

    #[test]
    fn test_map_wrong_arg_count() {
        let result = builtin_map(&[]);
        assert!(result.is_err());

        let result = builtin_map(&[Value::none()]);
        assert!(result.is_err());
    }

    // -------------------------------------------------------------------------
    // sorted() tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_sorted_integers_ascending() {
        // sorted([3, 1, 2]) should return [1, 2, 3]
        let list = ListObject::from_slice(&[
            Value::int(3).unwrap(),
            Value::int(1).unwrap(),
            Value::int(2).unwrap(),
        ]);
        let ptr = Box::leak(Box::new(list)) as *mut ListObject as *const ();
        let value = Value::object_ptr(ptr);

        let result = builtin_sorted(&[value]).unwrap();

        // Result should be a list containing [1, 2, 3]
        let result_ptr = result.as_object_ptr().unwrap();
        let sorted_list = unsafe { &*(result_ptr as *const ListObject) };
        assert_eq!(sorted_list.len(), 3);
        assert_eq!(sorted_list.get(0).unwrap().as_int().unwrap(), 1);
        assert_eq!(sorted_list.get(1).unwrap().as_int().unwrap(), 2);
        assert_eq!(sorted_list.get(2).unwrap().as_int().unwrap(), 3);
    }

    #[test]
    fn test_sorted_integers_descending() {
        // sorted([1, 2, 3], reverse=True) should return [3, 2, 1]
        let list = ListObject::from_slice(&[
            Value::int(1).unwrap(),
            Value::int(2).unwrap(),
            Value::int(3).unwrap(),
        ]);
        let ptr = Box::leak(Box::new(list)) as *mut ListObject as *const ();
        let value = Value::object_ptr(ptr);

        let result = builtin_sorted(&[value, Value::none(), Value::bool(true)]).unwrap();

        let result_ptr = result.as_object_ptr().unwrap();
        let sorted_list = unsafe { &*(result_ptr as *const ListObject) };
        assert_eq!(sorted_list.len(), 3);
        assert_eq!(sorted_list.get(0).unwrap().as_int().unwrap(), 3);
        assert_eq!(sorted_list.get(1).unwrap().as_int().unwrap(), 2);
        assert_eq!(sorted_list.get(2).unwrap().as_int().unwrap(), 1);
    }

    #[test]
    fn test_sorted_range() {
        // sorted(range(5, 0, -1)) should return [1, 2, 3, 4, 5]
        let range = builtin_range(&[
            Value::int(5).unwrap(),
            Value::int(0).unwrap(),
            Value::int(-1).unwrap(),
        ])
        .unwrap();

        let result = builtin_sorted(&[range]).unwrap();

        let result_ptr = result.as_object_ptr().unwrap();
        let sorted_list = unsafe { &*(result_ptr as *const ListObject) };
        assert_eq!(sorted_list.len(), 5);
        assert_eq!(sorted_list.get(0).unwrap().as_int().unwrap(), 1);
        assert_eq!(sorted_list.get(4).unwrap().as_int().unwrap(), 5);
    }

    #[test]
    fn test_sorted_empty() {
        // sorted([]) should return []
        let list = ListObject::new();
        let ptr = Box::leak(Box::new(list)) as *mut ListObject as *const ();
        let value = Value::object_ptr(ptr);

        let result = builtin_sorted(&[value]).unwrap();

        let result_ptr = result.as_object_ptr().unwrap();
        let sorted_list = unsafe { &*(result_ptr as *const ListObject) };
        assert_eq!(sorted_list.len(), 0);
    }

    #[test]
    fn test_sorted_single_element() {
        let list = ListObject::from_slice(&[Value::int(42).unwrap()]);
        let ptr = Box::leak(Box::new(list)) as *mut ListObject as *const ();
        let value = Value::object_ptr(ptr);

        let result = builtin_sorted(&[value]).unwrap();

        let result_ptr = result.as_object_ptr().unwrap();
        let sorted_list = unsafe { &*(result_ptr as *const ListObject) };
        assert_eq!(sorted_list.len(), 1);
        assert_eq!(sorted_list.get(0).unwrap().as_int().unwrap(), 42);
    }

    #[test]
    fn test_sorted_already_sorted() {
        let list = ListObject::from_slice(&[
            Value::int(1).unwrap(),
            Value::int(2).unwrap(),
            Value::int(3).unwrap(),
        ]);
        let ptr = Box::leak(Box::new(list)) as *mut ListObject as *const ();
        let value = Value::object_ptr(ptr);

        let result = builtin_sorted(&[value]).unwrap();

        let result_ptr = result.as_object_ptr().unwrap();
        let sorted_list = unsafe { &*(result_ptr as *const ListObject) };
        assert_eq!(sorted_list.get(0).unwrap().as_int().unwrap(), 1);
        assert_eq!(sorted_list.get(1).unwrap().as_int().unwrap(), 2);
        assert_eq!(sorted_list.get(2).unwrap().as_int().unwrap(), 3);
    }

    #[test]
    fn test_sorted_wrong_arg_count() {
        let result = builtin_sorted(&[]);
        assert!(result.is_err());

        let list = ListObject::new();
        let ptr = Box::leak(Box::new(list)) as *mut ListObject as *const ();
        let value = Value::object_ptr(ptr);

        // Too many arguments
        let result = builtin_sorted(&[
            value,
            Value::none(),
            Value::bool(false),
            Value::int(1).unwrap(),
        ]);
        assert!(result.is_err());
    }

    #[test]
    fn test_sorted_key_not_implemented() {
        // Key function requires VM integration
        let list = ListObject::from_slice(&[Value::int(1).unwrap()]);
        let ptr = Box::leak(Box::new(list)) as *mut ListObject as *const ();
        let value = Value::object_ptr(ptr);

        // Pass a non-None key function
        let result = builtin_sorted(&[value, Value::int(42).unwrap()]);
        assert!(result.is_err()); // Should be NotImplemented
    }

    // -------------------------------------------------------------------------
    // compare_values() tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_compare_values_integers() {
        use std::cmp::Ordering;

        let a = Value::int(1).unwrap();
        let b = Value::int(2).unwrap();
        let c = Value::int(1).unwrap();

        assert_eq!(compare_values(&a, &b), Ordering::Less);
        assert_eq!(compare_values(&b, &a), Ordering::Greater);
        assert_eq!(compare_values(&a, &c), Ordering::Equal);
    }

    #[test]
    fn test_compare_values_floats() {
        use std::cmp::Ordering;

        let a = Value::float(1.5);
        let b = Value::float(2.5);
        let c = Value::float(1.5);

        assert_eq!(compare_values(&a, &b), Ordering::Less);
        assert_eq!(compare_values(&b, &a), Ordering::Greater);
        assert_eq!(compare_values(&a, &c), Ordering::Equal);
    }

    #[test]
    fn test_compare_values_int_float() {
        use std::cmp::Ordering;

        let a = Value::int(2).unwrap();
        let b = Value::float(2.5);

        assert_eq!(compare_values(&a, &b), Ordering::Less);
        assert_eq!(compare_values(&b, &a), Ordering::Greater);
    }

    #[test]
    fn test_compare_values_none() {
        use std::cmp::Ordering;

        let none = Value::none();
        let one = Value::int(1).unwrap();

        // None is smallest
        assert_eq!(compare_values(&none, &one), Ordering::Less);
        assert_eq!(compare_values(&one, &none), Ordering::Greater);
        assert_eq!(compare_values(&none, &none), Ordering::Equal);
    }

    #[test]
    fn test_compare_values_booleans() {
        use std::cmp::Ordering;

        let f = Value::bool(false);
        let t = Value::bool(true);

        assert_eq!(compare_values(&f, &t), Ordering::Less);
        assert_eq!(compare_values(&t, &f), Ordering::Greater);
        assert_eq!(compare_values(&t, &t), Ordering::Equal);
    }
}
