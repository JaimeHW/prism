//! GC Allocation Integration Tests
//!
//! Comprehensive test suite verifying that all GC-managed heap allocations
//! in the VM function correctly. Tests cover:
//!
//! - Container allocations (list, tuple, dict, set, slice)
//! - Subscript operations (list/tuple integer indexing)
//! - Function and closure allocations
//! - Stress tests for allocation pressure scenarios
//!
//! These tests validate the GC migration from Box::new/into_raw to
//! vm.allocator().alloc() across all opcode handlers.
//!
//! Compatibility edge cases that were previously ignored are kept active here
//! so regressions are caught in normal CI runs.

use prism_compiler::Compiler;
use prism_core::Value;
use prism_parser::parse;
use prism_vm::VirtualMachine;
use std::sync::Arc;

// =============================================================================
// Test Infrastructure
// =============================================================================

/// Execute Python source and return the result.
fn execute(source: &str) -> Result<Value, String> {
    let module = parse(source).map_err(|e| format!("Parse error: {:?}", e))?;
    let code = Compiler::compile_module(&module, "<gc_test>")
        .map_err(|e| format!("Compile error: {:?}", e))?;
    let mut vm = VirtualMachine::new();
    vm.execute(Arc::new(code))
        .map_err(|e| format!("Runtime error: {:?}", e))
}

/// Execute Python source and verify it doesn't crash.
fn assert_executes(source: &str, test_name: &str) {
    let result = execute(source);
    assert!(result.is_ok(), "{}: Failed with {:?}", test_name, result);
}

fn comma_joined<F>(count: usize, mut f: F) -> String
where
    F: FnMut(usize) -> String,
{
    (0..count).map(&mut f).collect::<Vec<_>>().join(", ")
}

// =============================================================================
// Container Allocation Tests - Lists
// =============================================================================

#[test]
fn test_gc_list_empty() {
    assert_executes("[]", "empty_list");
}

#[test]
fn test_gc_list_single_element() {
    assert_executes("[42]", "single_element_list");
}

#[test]
fn test_gc_list_multiple_elements() {
    assert_executes("[1, 2, 3, 4, 5]", "multiple_element_list");
}

#[test]
fn test_gc_list_nested() {
    assert_executes("[[1, 2], [3, 4], [5, 6]]", "nested_list");
}

#[test]
fn test_gc_list_deeply_nested() {
    assert_executes("[[[1]], [[2]], [[3]]]", "deeply_nested_list");
}

#[test]
fn test_gc_list_with_operations() {
    assert_executes(
        r#"
x = [1, 2, 3]
y = x[0] + x[1] + x[2]
"#,
        "list_with_operations",
    );
}

#[test]
fn test_gc_list_reassignment() {
    assert_executes(
        r#"
x = [1, 2, 3]
x = [4, 5, 6]
y = x[0]
"#,
        "list_reassignment",
    );
}

#[test]
fn test_gc_list_in_loop() {
    assert_executes(
        r#"
result = 0
for i in [1, 2, 3, 4, 5]:
    result = result + i
"#,
        "list_in_loop",
    );
}

#[test]
fn test_gc_list_multiple_allocations() {
    assert_executes(
        r#"
a = [1, 2, 3]
b = [4, 5, 6]
c = [7, 8, 9]
d = a[0] + b[0] + c[0]
"#,
        "multiple_list_allocations",
    );
}

#[test]
fn test_gc_list_large() {
    assert_executes(
        r#"
x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
result = x[0] + x[15]
"#,
        "list_large",
    );
}

// =============================================================================
// Container Allocation Tests - Tuples
// =============================================================================

#[test]
fn test_gc_tuple_empty() {
    assert_executes("()", "empty_tuple");
}

#[test]
fn test_gc_tuple_single_element() {
    assert_executes("(42,)", "single_element_tuple");
}

#[test]
fn test_gc_tuple_multiple_elements() {
    assert_executes("(1, 2, 3, 4, 5)", "multiple_element_tuple");
}

#[test]
fn test_gc_tuple_nested() {
    assert_executes("((1, 2), (3, 4))", "nested_tuple");
}

#[test]
fn test_gc_tuple_with_operations() {
    assert_executes(
        r#"
t = (10, 20, 30)
x = t[0] + t[1]
"#,
        "tuple_with_operations",
    );
}

#[test]
fn test_gc_tuple_in_loop() {
    assert_executes(
        r#"
result = 0
for val in (1, 2, 3, 4, 5):
    result = result + val
"#,
        "tuple_in_loop",
    );
}

#[test]
fn test_gc_tuple_large() {
    assert_executes(
        r#"
t = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
result = t[0] + t[9]
"#,
        "tuple_large",
    );
}

// =============================================================================
// Container Allocation Tests - Dicts
// =============================================================================

#[test]
fn test_gc_dict_empty() {
    assert_executes("{}", "empty_dict");
}

#[test]
fn test_gc_dict_single_entry() {
    assert_executes("{\"a\": 1}", "single_entry_dict");
}

#[test]
fn test_gc_dict_multiple_entries() {
    assert_executes("{\"a\": 1, \"b\": 2, \"c\": 3}", "multiple_entry_dict");
}

#[test]
fn test_gc_dict_multiple_allocations() {
    assert_executes(
        r#"
d1 = {"a": 1}
d2 = {"b": 2}
d3 = {"c": 3}
"#,
        "multiple_dict_allocations",
    );
}

#[test]
fn test_gc_dict_nested() {
    assert_executes(
        r#"
d = {"outer": {"inner": 1}}
"#,
        "dict_nested",
    );
}

// =============================================================================
// Container Allocation Tests - Sets
// =============================================================================

#[test]
fn test_gc_set_single_element() {
    assert_executes("{1}", "single_element_set");
}

#[test]
fn test_gc_set_multiple_elements() {
    assert_executes("{1, 2, 3, 4, 5}", "multiple_element_set");
}

#[test]
fn test_gc_set_with_duplicates() {
    assert_executes("{1, 1, 2, 2, 3}", "set_with_duplicates");
}

#[test]
fn test_gc_set_multiple_allocations() {
    assert_executes(
        r#"
s1 = {1, 2, 3}
s2 = {4, 5, 6}
s3 = {7, 8, 9}
"#,
        "multiple_set_allocations",
    );
}

#[test]
fn test_gc_set_large() {
    assert_executes("{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}", "set_large");
}

// =============================================================================
// Function Allocation Tests
// =============================================================================

#[test]
fn test_gc_function_simple_def() {
    assert_executes(
        r#"
def add(a, b):
    return a + b
result = add(1, 2)
"#,
        "function_simple_def",
    );
}

#[test]
fn test_gc_function_multiple_definitions() {
    assert_executes(
        r#"
def func1():
    return 1

def func2():
    return 2

def func3():
    return 3

result = func1() + func2() + func3()
"#,
        "function_multiple_definitions",
    );
}

#[test]
fn test_gc_function_nested_calls() {
    assert_executes(
        r#"
def double(x):
    return x * 2

def quadruple(x):
    return double(double(x))

result = quadruple(5)
"#,
        "function_nested_calls",
    );
}

#[test]
fn test_gc_function_with_local_containers() {
    assert_executes(
        r#"
def sum_list(items):
    total = 0
    for item in items:
        total = total + item
    return total

result = sum_list([1, 2, 3, 4, 5])
"#,
        "function_with_local_containers",
    );
}

#[test]
fn test_gc_function_returning_container() {
    assert_executes(
        r#"
def make_list():
    return [1, 2, 3]

x = make_list()
result = x[0] + x[1] + x[2]
"#,
        "function_returning_container",
    );
}

#[test]
fn test_gc_function_in_loop() {
    assert_executes(
        r#"
def square(n):
    return n * n

result = 0
for i in range(5):
    result = result + square(i)
"#,
        "function_in_loop",
    );
}

#[test]
fn test_gc_function_with_default_args() {
    assert_executes(
        r#"
def greet(name, times=1):
    return times

assert greet("world") == 1
assert greet("world", 4) == 4
assert greet(name="world") == 1
"#,
        "function_with_default_args",
    );
}

#[test]
fn test_gc_function_with_kwonly_default_args() {
    assert_executes(
        r#"
def add(a, *, b=10):
    return a + b

assert add(5) == 15
assert add(5, b=7) == 12
"#,
        "function_with_kwonly_default_args",
    );
}

#[test]
fn test_gc_lambda_with_default_args() {
    assert_executes(
        r#"
f = lambda x, y=3: x + y
assert f(4) == 7
assert f(4, 5) == 9
"#,
        "lambda_with_default_args",
    );
}

#[test]
fn test_gc_function_many_params() {
    assert_executes(
        r#"
def add_all(a, b, c, d, e):
    return a + b + c + d + e

result = add_all(1, 2, 3, 4, 5)
"#,
        "function_many_params",
    );
}

#[test]
fn test_gc_function_many_params_keyword_binding_over_u64_bitmap() {
    let param_count = 66usize;
    let params = comma_joined(param_count, |i| format!("p{}", i));
    let positional_args = comma_joined(param_count - 1, |i| (i + 1).to_string());

    let source = format!(
        r#"
def f({params}):
    return p0 + p64 + p65

assert f({positional_args}, p65=66) == 132
"#
    );

    assert_executes(
        &source,
        "function_many_params_keyword_binding_over_u64_bitmap",
    );
}

#[test]
fn test_gc_function_defaults_over_u64_bitmap() {
    let param_count = 66usize;
    let mut params = (0..(param_count - 1))
        .map(|i| format!("p{}", i))
        .collect::<Vec<_>>();
    params.push("p65=99".to_string());
    let params = params.join(", ");
    let positional_args = comma_joined(param_count - 1, |i| (i + 1).to_string());

    let source = format!(
        r#"
def f({params}):
    return p65

assert f({positional_args}) == 99
"#
    );

    assert_executes(&source, "function_defaults_over_u64_bitmap");
}

// =============================================================================
// Unpacking Allocation Tests
// =============================================================================

#[test]
fn test_gc_unpack_list_simple() {
    assert_executes(
        r#"
x = [1, 2, 3]
a = x[0]
b = x[1]
c = x[2]
result = a + b + c
"#,
        "unpack_list_simple",
    );
}

#[test]
fn test_gc_unpack_tuple_simple() {
    assert_executes(
        r#"
t = (10, 20, 30)
a = t[0]
b = t[1]
result = a + b
"#,
        "unpack_tuple_simple",
    );
}

// =============================================================================
// Stress Tests - Allocation Pressure
// =============================================================================

#[test]
fn test_gc_stress_many_lists() {
    assert_executes(
        r#"
result = 0
for i in range(100):
    x = [i, i+1, i+2]
    result = result + x[0]
"#,
        "stress_many_lists",
    );
}

#[test]
fn test_gc_stress_many_tuples() {
    assert_executes(
        r#"
result = 0
for i in range(100):
    t = (i, i*2, i*3)
    result = result + t[0]
"#,
        "stress_many_tuples",
    );
}

#[test]
fn test_gc_stress_many_dicts() {
    assert_executes(
        r#"
result = 0
for i in range(50):
    d = {"a": i, "b": i*2}
    result = result + 1
"#,
        "stress_many_dicts",
    );
}

#[test]
fn test_gc_stress_many_function_calls() {
    assert_executes(
        r#"
def add_one(n):
    return n + 1

result = 0
for i in range(100):
    result = add_one(result)
"#,
        "stress_many_function_calls",
    );
}

#[test]
fn test_gc_stress_nested_containers() {
    assert_executes(
        r#"
result = 0
for i in range(20):
    x = [[i], [i+1], [i+2]]
    result = result + x[0][0]
"#,
        "stress_nested_containers",
    );
}

#[test]
fn test_gc_stress_mixed_containers() {
    assert_executes(
        r#"
result = 0
for i in range(30):
    lst = [i, i+1]
    tup = (i, i+2)
    result = result + lst[0] + tup[0]
"#,
        "stress_mixed_containers",
    );
}

#[test]
fn test_gc_stress_rapid_reassignment() {
    assert_executes(
        r#"
for i in range(50):
    x = [1, 2, 3]
    x = [4, 5, 6]
    x = [7, 8, 9]
    x = [10, 11, 12]
"#,
        "stress_rapid_reassignment",
    );
}

// =============================================================================
// Mixed Allocation Scenarios
// =============================================================================

#[test]
fn test_gc_mixed_containers_and_functions() {
    assert_executes(
        r#"
def sum_items(items):
    total = 0
    for item in items:
        total = total + item
    return total

data = [1, 2, 3, 4, 5]
result = sum_items(data)
"#,
        "mixed_containers_and_functions",
    );
}

#[test]
fn test_gc_function_creates_multiple_containers() {
    assert_executes(
        r#"
def make_containers():
    a = [1, 2, 3]
    b = (4, 5, 6)
    c = {7, 8, 9}
    return a[0] + b[0]

result = make_containers()
"#,
        "function_creates_multiple_containers",
    );
}

#[test]
fn test_gc_nested_function_calls_with_containers() {
    assert_executes(
        r#"
def outer(items):
    def inner(x):
        return x * 2
    total = 0
    for item in items:
        total = total + inner(item)
    return total

result = outer([1, 2, 3])
"#,
        "nested_function_calls_with_containers",
    );
}

// =============================================================================
// Edge Cases
// =============================================================================

#[test]
fn test_gc_empty_containers_loop() {
    assert_executes(
        r#"
for i in range(10):
    x = []
    y = ()
    z = {}
"#,
        "empty_containers_loop",
    );
}

#[test]
fn test_gc_single_element_containers_loop() {
    assert_executes(
        r#"
for i in range(10):
    x = [i]
    y = (i,)
    z = {"a": i}
"#,
        "single_element_containers_loop",
    );
}

#[test]
fn test_gc_immediate_reassignment() {
    assert_executes(
        r#"
x = [1, 2, 3]
x = [4, 5, 6]
x = [7, 8, 9]
result = x[0]
"#,
        "immediate_reassignment",
    );
}

#[test]
fn test_gc_container_as_function_result() {
    assert_executes(
        r#"
def returns_list():
    return [1, 2, 3]

def uses_list():
    x = returns_list()
    return x[0] + x[1] + x[2]

result = uses_list()
"#,
        "container_as_function_result",
    );
}

#[test]
fn test_gc_function_returning_tuple() {
    assert_executes(
        r#"
def get_pair():
    return (10, 20)

t = get_pair()
result = t[0] + t[1]
"#,
        "function_returning_tuple",
    );
}

// =============================================================================
// Range Object Integration
// =============================================================================

#[test]
fn test_gc_range_iteration() {
    assert_executes(
        r#"
result = 0
for i in range(10):
    result = result + i
"#,
        "range_iteration",
    );
}

#[test]
fn test_gc_range_with_start_stop() {
    assert_executes(
        r#"
result = 0
for i in range(5, 10):
    result = result + i
"#,
        "range_with_start_stop",
    );
}

#[test]
fn test_gc_range_with_step() {
    assert_executes(
        r#"
result = 0
for i in range(0, 10, 2):
    result = result + i
"#,
        "range_with_step",
    );
}

#[test]
fn test_gc_nested_ranges() {
    assert_executes(
        r#"
result = 0
for i in range(5):
    for j in range(5):
        result = result + 1
"#,
        "nested_ranges",
    );
}

#[test]
fn test_gc_range_large() {
    assert_executes(
        r#"
result = 0
for i in range(1000):
    result = result + 1
"#,
        "range_large",
    );
}

// =============================================================================
// Complex Programs - GC Allocation Intensive
// =============================================================================

#[test]
fn test_gc_fibonacci_iterative() {
    assert_executes(
        r#"
def fib(n):
    a = 0
    b = 1
    for i in range(n):
        temp = a + b
        a = b
        b = temp
    return a

result = fib(10)
"#,
        "fibonacci_iterative",
    );
}

#[test]
fn test_gc_sum_squares() {
    assert_executes(
        r#"
def square(x):
    return x * x

result = 0
for i in range(10):
    result = result + square(i)
"#,
        "sum_squares",
    );
}

#[test]
fn test_gc_filter_pattern() {
    assert_executes(
        r#"
def is_even(n):
    return n % 2 == 0

result = 0
for i in range(20):
    if is_even(i):
        result = result + i
"#,
        "filter_pattern",
    );
}

#[test]
fn test_gc_nested_list_access() {
    assert_executes(
        r#"
matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
result = matrix[0][0] + matrix[1][1] + matrix[2][2]
"#,
        "nested_list_access",
    );
}

#[test]
fn test_gc_list_of_tuples() {
    assert_executes(
        r#"
pairs = [(1, 2), (3, 4), (5, 6)]
result = 0
for pair in pairs:
    result = result + pair[0] + pair[1]
"#,
        "list_of_tuples",
    );
}

#[test]
fn test_gc_multiple_return_containers() {
    assert_executes(
        r#"
def make_data():
    return [1, 2, 3]

x = make_data()
y = make_data()
z = make_data()
result = x[0] + y[0] + z[0]
"#,
        "multiple_return_containers",
    );
}

// =============================================================================
// Compatibility Regression Tests
// =============================================================================

#[test]
fn test_gc_closure_simple() {
    assert_executes(
        r#"
def make_adder(n):
    def adder(x):
        return x + n
    return adder

add5 = make_adder(5)
result = add5(10)
"#,
        "closure_simple",
    );
}

#[test]
fn test_gc_dict_string_keys() {
    assert_executes(
        r#"
d = {"key1": 100, "key2": 200}
x = d["key1"]
"#,
        "dict_string_keys",
    );
}

#[test]
fn test_gc_slice_with_step() {
    assert_executes(
        r#"
x = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
y = x[::2]
"#,
        "slice_with_step",
    );
}

#[test]
fn test_gc_slice_negative_indices() {
    assert_executes(
        r#"
x = [1, 2, 3, 4, 5]
y = x[-3:-1]
"#,
        "slice_negative_indices",
    );
}

#[test]
fn test_gc_unpack_starred() {
    assert_executes(
        r#"
first, *rest = [1, 2, 3, 4, 5]
result = first
"#,
        "unpack_starred",
    );
}

#[test]
fn test_gc_varargs_iteration() {
    assert_executes(
        r#"
def sum_all(*args):
    result = 0
    for arg in args:
        result = result + arg
    return result

x = sum_all(1, 2, 3, 4, 5)
"#,
        "varargs_iteration",
    );
}

#[test]
fn test_gc_kwargs_len() {
    assert_executes(
        r#"
def check_kwargs(**kwargs):
    return len(kwargs)

x = check_kwargs(a=1, b=2, c=3)
"#,
        "kwargs_len",
    );
}

#[test]
fn test_gc_function_recursion() {
    assert_executes(
        r#"
def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n - 1)
result = factorial(5)
"#,
        "function_recursion",
    );
}

#[test]
fn test_gc_tuple_unpacking() {
    assert_executes(
        r#"
a, b, c = (1, 2, 3)
result = a + b + c
"#,
        "tuple_unpacking",
    );
}
