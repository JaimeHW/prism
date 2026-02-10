//! Comprehensive keyword arguments and call-site unpacking compilation tests.
//!
//! This module provides exhaustive test coverage for `*args` and `**kwargs` semantics,
//! both at function definition (parameter collection) and call sites (argument unpacking).
//!
//! # Test Categories
//!
//! 1. **Basic *args Collection** - Variadic positional parameters (`*args`)
//! 2. **Basic **kwargs Collection** - Variadic keyword parameters (`**kwargs`)
//! 3. **Mixed Parameters** - Combinations of regular, *args, and **kwargs
//! 4. **Keyword-Only Parameters** - Parameters after `*` or `*args`
//! 5. **Call-Site *args Unpacking** - `func(*iterable)` patterns
//! 6. **Call-Site **kwargs Unpacking** - `func(**mapping)` patterns
//! 7. **Mixed Call-Site Unpacking** - `func(*args, **kwargs)` combinations
//! 8. **Default Arguments** - Default values with *args/**kwargs
//! 9. **Nested Functions** - Closures with variadic parameters
//! 10. **Edge Cases** - Complex and unusual patterns

use prism_compiler::{Compiler, Opcode};
use prism_parser::parse;

// =============================================================================
// Test Helpers
// =============================================================================

/// Parse and compile Python source, returning the compiled code object.
fn compile(source: &str) -> prism_compiler::CodeObject {
    let module = parse(source).expect("Failed to parse");
    Compiler::compile_module(&module, "test.py").expect("Failed to compile")
}

/// Check that compilation succeeds without panicking.
fn assert_compiles(source: &str) {
    compile(source);
}

/// Helper to find instructions of a specific opcode in bytecode.
fn find_opcodes(code: &prism_compiler::CodeObject, opcode: Opcode) -> Vec<usize> {
    code.instructions
        .iter()
        .enumerate()
        .filter(|(_, instr)| instr.opcode() == opcode as u8)
        .map(|(i, _)| i)
        .collect()
}

/// Check that specific opcodes appear in the compiled bytecode.
fn assert_contains_opcodes(source: &str, expected_opcodes: &[Opcode]) {
    let code = compile(source);
    for opcode in expected_opcodes {
        let found = find_opcodes(&code, *opcode);
        assert!(
            !found.is_empty(),
            "Expected opcode {:?} not found in bytecode",
            opcode
        );
    }
}

// =============================================================================
// A. Basic *args Collection (10 tests)
// =============================================================================

#[test]
fn test_args_simple_function() {
    let source = r#"
def f(*args):
    return args
"#;
    assert_compiles(source);
}

#[test]
fn test_args_with_positional_before() {
    let source = r#"
def f(a, b, *args):
    return (a, b, args)
"#;
    assert_compiles(source);
}

#[test]
fn test_args_with_single_positional() {
    let source = r#"
def f(x, *args):
    return x
"#;
    assert_compiles(source);
}

#[test]
fn test_args_empty_call() {
    let source = r#"
def f(*args):
    pass
f()
"#;
    assert_compiles(source);
}

#[test]
fn test_args_single_value_call() {
    let source = r#"
def f(*args):
    pass
f(1)
"#;
    assert_compiles(source);
}

#[test]
fn test_args_multiple_values_call() {
    let source = r#"
def f(*args):
    pass
f(1, 2, 3, 4, 5)
"#;
    assert_compiles(source);
}

#[test]
fn test_args_mixed_call() {
    let source = r#"
def f(a, *args):
    pass
f(1, 2, 3, 4)
"#;
    assert_compiles(source);
}

#[test]
fn test_args_tuple_access() {
    let source = r#"
def f(*args):
    return len(args)
"#;
    assert_compiles(source);
}

#[test]
fn test_args_iteration() {
    let source = r#"
def sum_all(*args):
    total = 0
    for x in args:
        total = total + x
    return total
"#;
    assert_compiles(source);
}

#[test]
fn test_args_indexing() {
    let source = r#"
def first(*args):
    return args[0]
"#;
    assert_compiles(source);
}

// =============================================================================
// B. Basic **kwargs Collection (10 tests)
// =============================================================================

#[test]
fn test_kwargs_simple_function() {
    let source = r#"
def f(**kwargs):
    return kwargs
"#;
    assert_compiles(source);
}

#[test]
fn test_kwargs_with_positional_before() {
    let source = r#"
def f(a, b, **kwargs):
    return kwargs
"#;
    assert_compiles(source);
}

#[test]
fn test_kwargs_empty_call() {
    let source = r#"
def f(**kwargs):
    pass
f()
"#;
    assert_compiles(source);
}

#[test]
fn test_kwargs_single_kw_call() {
    let source = r#"
def f(**kwargs):
    pass
f(x=1)
"#;
    assert_compiles(source);
}

#[test]
fn test_kwargs_multiple_kw_call() {
    let source = r#"
def f(**kwargs):
    pass
f(a=1, b=2, c=3)
"#;
    assert_compiles(source);
}

#[test]
fn test_kwargs_mixed_call() {
    let source = r#"
def f(a, **kwargs):
    pass
f(1, x=2, y=3)
"#;
    assert_compiles(source);
}

#[test]
fn test_kwargs_dict_access() {
    let source = r#"
def f(**kwargs):
    return len(kwargs)
"#;
    assert_compiles(source);
}

#[test]
fn test_kwargs_iteration() {
    let source = r#"
def print_kwargs(**kwargs):
    for key in kwargs:
        val = kwargs[key]
"#;
    assert_compiles(source);
}

#[test]
fn test_kwargs_get_method() {
    let source = r#"
def get_or_default(**kwargs):
    return kwargs.get("key", "default")
"#;
    assert_compiles(source);
}

#[test]
fn test_kwargs_keys_values_items() {
    let source = r#"
def inspect(**kwargs):
    k = list(kwargs.keys())
    v = list(kwargs.values())
    i = list(kwargs.items())
"#;
    assert_compiles(source);
}

// =============================================================================
// C. Mixed Parameters (8 tests)
// =============================================================================

#[test]
fn test_mixed_args_kwargs() {
    let source = r#"
def f(*args, **kwargs):
    return (args, kwargs)
"#;
    assert_compiles(source);
}

#[test]
fn test_mixed_positional_args_kwargs() {
    let source = r#"
def f(a, b, *args, **kwargs):
    return (a, b, args, kwargs)
"#;
    assert_compiles(source);
}

#[test]
fn test_mixed_call_positional_keyword() {
    let source = r#"
def f(a, *args, **kwargs):
    pass
f(1, 2, 3, x=4, y=5)
"#;
    assert_compiles(source);
}

#[test]
fn test_mixed_many_positional_many_keyword() {
    let source = r#"
def f(*args, **kwargs):
    pass
f(1, 2, 3, 4, 5, a=1, b=2, c=3, d=4)
"#;
    assert_compiles(source);
}

#[test]
fn test_mixed_only_positional() {
    let source = r#"
def f(*args, **kwargs):
    pass
f(1, 2, 3)
"#;
    assert_compiles(source);
}

#[test]
fn test_mixed_only_keyword() {
    let source = r#"
def f(*args, **kwargs):
    pass
f(a=1, b=2)
"#;
    assert_compiles(source);
}

#[test]
fn test_mixed_empty_args_with_kwargs() {
    let source = r#"
def f(x, *args, **kwargs):
    pass
f(1, a=2)
"#;
    assert_compiles(source);
}

#[test]
fn test_mixed_complex_signature() {
    let source = r#"
def complex_func(a, b, c=10, *args, d=20, e=30, **kwargs):
    return (a, b, c, args, d, e, kwargs)
"#;
    assert_compiles(source);
}

// =============================================================================
// D. Keyword-Only Parameters (7 tests)
// =============================================================================

#[test]
fn test_kwonly_after_star() {
    let source = r#"
def f(*, kw):
    return kw
"#;
    assert_compiles(source);
}

#[test]
fn test_kwonly_after_args() {
    let source = r#"
def f(*args, kw):
    return kw
"#;
    assert_compiles(source);
}

#[test]
fn test_kwonly_with_default() {
    let source = r#"
def f(*, kw=10):
    return kw
"#;
    assert_compiles(source);
}

#[test]
fn test_kwonly_multiple() {
    let source = r#"
def f(*, a, b, c=10):
    return (a, b, c)
"#;
    assert_compiles(source);
}

#[test]
fn test_kwonly_mixed_with_positional() {
    let source = r#"
def f(x, y, *, kw1, kw2=10):
    return (x, y, kw1, kw2)
"#;
    assert_compiles(source);
}

#[test]
fn test_kwonly_call() {
    let source = r#"
def f(*, a, b):
    pass
f(a=1, b=2)
"#;
    assert_compiles(source);
}

#[test]
fn test_kwonly_with_kwargs() {
    let source = r#"
def f(*args, kw, **kwargs):
    pass
f(1, 2, 3, kw=10, extra=20)
"#;
    assert_compiles(source);
}

// =============================================================================
// E. Call-Site *args Unpacking (7 tests)
// =============================================================================

#[test]
fn test_unpack_list_to_args() {
    let source = r#"
def f(a, b, c):
    pass
args = [1, 2, 3]
f(*args)
"#;
    assert_compiles(source);
}

#[test]
fn test_unpack_tuple_to_args() {
    let source = r#"
def f(a, b, c):
    pass
args = (1, 2, 3)
f(*args)
"#;
    assert_compiles(source);
}

#[test]
fn test_unpack_generator_to_args() {
    let source = r#"
def f(*args):
    pass
f(*range(10))
"#;
    assert_compiles(source);
}

#[test]
fn test_unpack_mixed_positional_starred() {
    let source = r#"
def f(a, b, c, d):
    pass
rest = [3, 4]
f(1, 2, *rest)
"#;
    assert_compiles(source);
}

#[test]
fn test_unpack_multiple_starred() {
    let source = r#"
def f(*args):
    pass
a = [1, 2]
b = [3, 4]
f(*a, *b)
"#;
    assert_compiles(source);
}

#[test]
fn test_unpack_with_regular_args_after() {
    let source = r#"
def f(*args):
    pass
lst = [1, 2]
f(*lst, 3, 4)
"#;
    assert_compiles(source);
}

#[test]
fn test_unpack_empty_iterable() {
    let source = r#"
def f(*args):
    pass
empty = []
f(*empty)
"#;
    assert_compiles(source);
}

// =============================================================================
// F. Call-Site **kwargs Unpacking (7 tests)
// =============================================================================

#[test]
fn test_unpack_dict_to_kwargs() {
    let source = r#"
def f(a, b, c):
    pass
kw = {"a": 1, "b": 2, "c": 3}
f(**kw)
"#;
    assert_compiles(source);
}

#[test]
fn test_unpack_mixed_keyword_double_starred() {
    let source = r#"
def f(a, b, **kwargs):
    pass
extra = {"c": 3, "d": 4}
f(a=1, b=2, **extra)
"#;
    assert_compiles(source);
}

#[test]
fn test_unpack_multiple_double_starred() {
    let source = r#"
def f(**kwargs):
    pass
d1 = {"a": 1}
d2 = {"b": 2}
f(**d1, **d2)
"#;
    assert_compiles(source);
}

#[test]
fn test_unpack_empty_dict() {
    let source = r#"
def f(**kwargs):
    pass
empty = {}
f(**empty)
"#;
    assert_compiles(source);
}

#[test]
fn test_unpack_dict_literal_double_star() {
    let source = r#"
def f(a, b):
    pass
f(**{"a": 1, "b": 2})
"#;
    assert_compiles(source);
}

#[test]
fn test_unpack_dict_comprehension() {
    let source = r#"
def f(**kwargs):
    pass
f(**{x: x*2 for x in range(3)})
"#;
    assert_compiles(source);
}

#[test]
fn test_unpack_kwargs_before_keywords() {
    let source = r#"
def f(a, b, c):
    pass
d = {"a": 1}
f(**d, b=2, c=3)
"#;
    assert_compiles(source);
}

// =============================================================================
// G. Mixed Call-Site Unpacking (8 tests)
// =============================================================================

#[test]
fn test_unpack_both_args_kwargs() {
    let source = r#"
def f(*args, **kwargs):
    pass
a = [1, 2]
k = {"x": 10}
f(*a, **k)
"#;
    assert_compiles(source);
}

#[test]
fn test_unpack_positional_starred_keyword_double_starred() {
    let source = r#"
def f(a, b, c, d):
    pass
rest = [2, 3]
kw = {"d": 4}
f(1, *rest, **kw)
"#;
    assert_compiles(source);
}

#[test]
fn test_unpack_complex_mix() {
    let source = r#"
def f(a, b, *args, c, d, **kwargs):
    pass
pos = [3, 4, 5]
kw = {"e": 6, "f": 7}
f(1, *pos, c=10, **kw, d=20)
"#;
    assert_compiles(source);
}

#[test]
fn test_unpack_multiple_starred_and_double_starred() {
    let source = r#"
def f(*args, **kwargs):
    pass
a1 = [1, 2]
a2 = [3, 4]
k1 = {"x": 10}
k2 = {"y": 20}
f(*a1, *a2, **k1, **k2)
"#;
    assert_compiles(source);
}

#[test]
fn test_unpack_starred_between_positional() {
    let source = r#"
def f(*args):
    pass
lst = [2, 3]
f(1, *lst, 4, 5)
"#;
    assert_compiles(source);
}

#[test]
fn test_unpack_double_starred_between_keywords() {
    let source = r#"
def f(**kwargs):
    pass
d = {"b": 2}
f(a=1, **d, c=3)
"#;
    assert_compiles(source);
}

#[test]
fn test_unpack_empty_both() {
    let source = r#"
def f(*args, **kwargs):
    pass
f(*[], **{})
"#;
    assert_compiles(source);
}

#[test]
fn test_unpack_nested_function_call() {
    let source = r#"
def inner(*args, **kwargs):
    return (args, kwargs)

def outer(*args, **kwargs):
    return inner(*args, **kwargs)
"#;
    assert_compiles(source);
}

// =============================================================================
// H. Default Arguments with *args/**kwargs (6 tests)
// =============================================================================

#[test]
fn test_default_before_args() {
    let source = r#"
def f(a, b=10, *args):
    return (a, b, args)
"#;
    assert_compiles(source);
}

#[test]
fn test_default_with_kwargs() {
    let source = r#"
def f(a, b=10, **kwargs):
    return (a, b, kwargs)
"#;
    assert_compiles(source);
}

#[test]
fn test_default_with_kwonly() {
    let source = r#"
def f(a, *args, b=10, c=20):
    return (a, args, b, c)
"#;
    assert_compiles(source);
}

#[test]
fn test_default_call_override() {
    let source = r#"
def f(a, b=10, c=20, *args):
    pass
f(1, 2)
"#;
    assert_compiles(source);
}

#[test]
fn test_default_call_positional_override_and_extra() {
    let source = r#"
def f(a, b=10, *args):
    pass
f(1, 2, 3, 4, 5)
"#;
    assert_compiles(source);
}

#[test]
fn test_default_kwonly_override() {
    let source = r#"
def f(*args, a=1, b=2):
    pass
f(1, 2, a=10)
"#;
    assert_compiles(source);
}

// =============================================================================
// I. Nested Functions with Variadic Parameters (5 tests)
// =============================================================================

#[test]
fn test_nested_closure_with_args() {
    let source = r#"
def outer():
    x = 10
    def inner(*args):
        return x + len(args)
    return inner
"#;
    assert_compiles(source);
}

#[test]
fn test_nested_closure_with_kwargs() {
    let source = r#"
def outer():
    x = 10
    def inner(**kwargs):
        return x + len(kwargs)
    return inner
"#;
    assert_compiles(source);
}

#[test]
fn test_nested_closure_capturing_args() {
    let source = r#"
def make_printer(*args):
    def printer():
        for a in args:
            print(a)
    return printer
"#;
    assert_compiles(source);
}

#[test]
fn test_nested_passing_through() {
    let source = r#"
def decorator(func):
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper
"#;
    assert_compiles(source);
}

#[test]
fn test_deeply_nested_variadic() {
    let source = r#"
def level1(*a1, **k1):
    def level2(*a2, **k2):
        def level3(*a3, **k3):
            return (a1, a2, a3, k1, k2, k3)
        return level3
    return level2
"#;
    assert_compiles(source);
}

// =============================================================================
// J. Edge Cases and Complex Patterns (7 tests)
// =============================================================================

#[test]
fn test_builtin_call_with_unpack() {
    let source = r#"
args = [1, 2, 3]
result = max(*args)
"#;
    assert_compiles(source);
}

#[test]
fn test_method_call_with_unpack() {
    let source = r#"
class C:
    def method(self, *args, **kwargs):
        pass

obj = C()
a = [1, 2]
k = {"x": 10}
obj.method(*a, **k)
"#;
    assert_compiles(source);
}

#[test]
fn test_lambda_with_args() {
    let source = r#"
f = lambda *args: sum(args)
"#;
    assert_compiles(source);
}

#[test]
fn test_lambda_with_kwargs() {
    let source = r#"
f = lambda **kwargs: len(kwargs)
"#;
    assert_compiles(source);
}

#[test]
fn test_lambda_mixed() {
    let source = r#"
f = lambda a, *args, b=10, **kwargs: (a, args, b, kwargs)
"#;
    assert_compiles(source);
}

#[test]
fn test_recursive_with_variadic() {
    let source = r#"
def recursive_sum(*args):
    if len(args) == 0:
        return 0
    elif len(args) == 1:
        return args[0]
    else:
        return args[0] + recursive_sum(*args[1:])
"#;
    assert_compiles(source);
}

#[test]
fn test_super_call_with_unpack() {
    let source = r#"
class Base:
    def __init__(self, *args, **kwargs):
        pass

class Derived(Base):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
"#;
    assert_compiles(source);
}

// =============================================================================
// K. Additional Comprehensive Tests (5 tests)
// =============================================================================

#[test]
fn test_positional_only_with_args() {
    let source = r#"
def f(a, b, /, *args):
    return (a, b, args)
"#;
    assert_compiles(source);
}

#[test]
fn test_full_signature() {
    let source = r#"
def f(pos_only, /, regular, *args, kw_only, kw_default=10, **kwargs):
    return (pos_only, regular, args, kw_only, kw_default, kwargs)
"#;
    assert_compiles(source);
}

#[test]
fn test_full_call() {
    let source = r#"
def f(a, b, c, *args, d, e=10, **kwargs):
    pass

extra_pos = [4, 5]
extra_kw = {"f": 20, "g": 30}
f(1, 2, 3, *extra_pos, d=100, **extra_kw, e=200)
"#;
    assert_compiles(source);
}

#[test]
fn test_decorator_pattern() {
    let source = r#"
def logging_decorator(func):
    def wrapper(*args, **kwargs):
        print("Calling function")
        result = func(*args, **kwargs)
        print("Function returned")
        return result
    return wrapper

@logging_decorator
def my_function(a, b, c=10):
    return a + b + c
"#;
    assert_compiles(source);
}

#[test]
fn test_partial_application_pattern() {
    let source = r#"
def partial(func, *fixed_args, **fixed_kwargs):
    def wrapper(*args, **kwargs):
        all_args = fixed_args + args
        all_kwargs = {**fixed_kwargs, **kwargs}
        return func(*all_args, **all_kwargs)
    return wrapper

def add(a, b, c):
    return a + b + c

add5 = partial(add, 5)
"#;
    assert_compiles(source);
}
