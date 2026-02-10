//! Integration tests for closure compilation.
//!
//! These tests verify that the compiler correctly emits closure-aware bytecode
//! for various closure patterns, from simple captures to complex multi-level closures.

use prism_compiler::{Compiler, Opcode, VarLocation};
use prism_parser::parse;

// =============================================================================
// Test Helpers
// =============================================================================

/// Parse and compile Python source, returning the compiled code object.
fn compile(source: &str) -> prism_compiler::CodeObject {
    let module = parse(source).expect("Failed to parse");
    Compiler::compile_module(&module, "test.py").expect("Failed to compile")
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

// =============================================================================
// Module Level Variable Tests
// =============================================================================

#[test]
fn test_module_variable_assignment() {
    let code = compile("x = 42");

    // Module level should use StoreGlobal
    let stores = find_opcodes(&code, Opcode::StoreGlobal);
    assert!(
        !stores.is_empty(),
        "Should have StoreGlobal for module-level assignment"
    );
}

#[test]
fn test_module_variable_read() {
    let code = compile("print(x)");

    // Module level reads should use LoadGlobal
    let loads = find_opcodes(&code, Opcode::LoadGlobal);
    assert!(
        loads.len() >= 1,
        "Should have LoadGlobal for module-level read"
    );
}

#[test]
fn test_module_multiple_variables() {
    let code = compile("a = 1\nb = 2\nc = a + b");

    let store_globals = find_opcodes(&code, Opcode::StoreGlobal);
    let load_globals = find_opcodes(&code, Opcode::LoadGlobal);

    // 3 stores (a, b, c)
    assert_eq!(store_globals.len(), 3, "Should store a, b, c as globals");

    // 2 loads (a, b for the c = a + b expression)
    assert_eq!(load_globals.len(), 2, "Should load a and b");
}

// =============================================================================
// Function Local Variable Tests
// =============================================================================

#[test]
fn test_function_local_variables() {
    let code = compile(
        r#"
def func():
    x = 10
    y = x + 5
    return y
"#,
    );

    // Function should be compiled but we don't execute it
    // Just verify the module compiles correctly
    assert!(!code.instructions.is_empty());
}

#[test]
fn test_function_parameter_access() {
    let code = compile(
        r#"
def add(a, b):
    return a + b
"#,
    );

    assert!(!code.instructions.is_empty());
}

// =============================================================================
// Simple Closure Tests
// =============================================================================

#[test]
fn test_simple_closure_definition() {
    // Outer function defines variable, inner function captures it
    let code = compile(
        r#"
def make_adder(n):
    def add(x):
        return x + n
    return add
"#,
    );

    // Module should compile successfully
    assert!(!code.instructions.is_empty());
}

#[test]
fn test_counter_closure() {
    // Classic counter closure pattern
    let code = compile(
        r#"
def make_counter():
    count = 0
    def increment():
        nonlocal count
        count = count + 1
        return count
    return increment
"#,
    );

    assert!(!code.instructions.is_empty());
}

#[test]
fn test_multiple_closures_same_variable() {
    // Multiple inner functions capture the same variable
    let code = compile(
        r#"
def make_ops(x):
    def get():
        return x
    def set(val):
        nonlocal x
        x = val
    return (get, set)
"#,
    );

    assert!(!code.instructions.is_empty());
}

// =============================================================================
// Nested Closure Tests (Multi-Level)
// =============================================================================

#[test]
fn test_two_level_closure() {
    // Grandparent -> Parent -> Child capture chain
    let code = compile(
        r#"
def grandparent():
    x = 10
    def parent():
        def child():
            return x
        return child
    return parent
"#,
    );

    assert!(!code.instructions.is_empty());
}

#[test]
fn test_three_level_closure_with_modification() {
    // Three levels with variable modification
    let code = compile(
        r#"
def level0():
    value = 0
    def level1():
        def level2():
            nonlocal value
            value = value + 1
            return value
        return level2
    return level1
"#,
    );

    assert!(!code.instructions.is_empty());
}

// =============================================================================
// Mixed Variable Types Tests
// =============================================================================

#[test]
fn test_local_and_captured_variables() {
    // Function has both local and captured variables
    let code = compile(
        r#"
def outer(x):
    local = 42
    def inner():
        return x + local
    return inner
"#,
    );

    assert!(!code.instructions.is_empty());
}

#[test]
fn test_global_local_and_closure() {
    // All three variable types in play
    let code = compile(
        r#"
GLOBAL = 100

def outer(param):
    local = 50
    def inner():
        return GLOBAL + param + local
    return inner
"#,
    );

    assert!(!code.instructions.is_empty());
}

#[test]
fn test_shadowed_variables() {
    // Variable shadowing in closures
    let code = compile(
        r#"
def outer():
    x = 1
    def inner():
        x = 2  # This shadows, not captures
        return x
    return inner
"#,
    );

    assert!(!code.instructions.is_empty());
}

// =============================================================================
// Lambda Closure Tests
// =============================================================================

#[test]
fn test_lambda_closure() {
    let code = compile(
        r#"
def make_multiplier(n):
    return lambda x: x * n
"#,
    );

    assert!(!code.instructions.is_empty());
}

#[test]
fn test_lambda_in_list_comprehension() {
    let code = compile(
        r#"
def make_funcs():
    return [lambda x, n=n: x + n for n in range(5)]
"#,
    );

    assert!(!code.instructions.is_empty());
}

// =============================================================================
// Class Method Closure Tests
// =============================================================================

#[test]
fn test_method_captures_class_variable() {
    // Method defined inside another function captures outer variable
    let code = compile(
        r#"
def make_class(value):
    class MyClass:
        def get(self):
            return value
    return MyClass
"#,
    );

    assert!(!code.instructions.is_empty());
}

// =============================================================================
// Edge Cases
// =============================================================================

#[test]
fn test_empty_closure() {
    // Closure that doesn't actually capture anything
    let code = compile(
        r#"
def outer():
    x = 10
    def inner():
        return 42  # Doesn't use x
    return inner
"#,
    );

    assert!(!code.instructions.is_empty());
}

#[test]
fn test_closure_only_reads() {
    // Closure only reads, never writes
    let code = compile(
        r#"
def outer():
    x = 10
    def inner():
        return x  # Read only
    return inner
"#,
    );

    assert!(!code.instructions.is_empty());
}

#[test]
fn test_closure_only_writes() {
    // Closure only writes (unusual but valid)
    let code = compile(
        r#"
def outer():
    x = 0
    def setter(val):
        nonlocal x
        x = val
    return setter
"#,
    );

    assert!(!code.instructions.is_empty());
}

#[test]
fn test_deeply_nested_closure() {
    // 5 levels deep
    let code = compile(
        r#"
def level0():
    x = 0
    def level1():
        def level2():
            def level3():
                def level4():
                    return x
                return level4
            return level3
        return level2
    return level1
"#,
    );

    assert!(!code.instructions.is_empty());
}

#[test]
fn test_multiple_captured_variables() {
    // Closure captures many variables
    let code = compile(
        r#"
def outer(a, b, c, d, e):
    def inner():
        return a + b + c + d + e
    return inner
"#,
    );

    assert!(!code.instructions.is_empty());
}

// =============================================================================
// Practical Patterns Tests
// =============================================================================

#[test]
fn test_decorator_pattern() {
    let code = compile(
        r#"
def decorator(func):
    def wrapper(*args):
        return func(*args)
    return wrapper
"#,
    );

    assert!(!code.instructions.is_empty());
}

#[test]
fn test_memoization_pattern() {
    let code = compile(
        r#"
def memoize(func):
    cache = {}
    def wrapper(n):
        if n not in cache:
            cache[n] = func(n)
        return cache[n]
    return wrapper
"#,
    );

    assert!(!code.instructions.is_empty());
}

#[test]
fn test_callback_pattern() {
    let code = compile(
        r#"
def register_callback(callback):
    handlers = []
    def add(f):
        handlers.append(f)
    def fire():
        for h in handlers:
            h()
    return (add, fire)
"#,
    );

    assert!(!code.instructions.is_empty());
}

// =============================================================================
// VarLocation Resolution Tests
// =============================================================================

#[test]
fn test_var_location_types() {
    // Verify VarLocation enum variants work correctly
    let local = VarLocation::Local(0);
    let closure = VarLocation::Closure(1);
    let global = VarLocation::Global;

    assert_ne!(local, closure);
    assert_ne!(local, global);
    assert_ne!(closure, global);

    // Same values should be equal
    assert_eq!(VarLocation::Local(5), VarLocation::Local(5));
    assert_eq!(VarLocation::Closure(3), VarLocation::Closure(3));
    assert_eq!(VarLocation::Global, VarLocation::Global);
}
