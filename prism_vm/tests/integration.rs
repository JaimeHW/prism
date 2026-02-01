//! End-to-end integration tests for Prism VM.
//!
//! These tests verify the complete parse → compile → execute pipeline.

use prism_compiler::Compiler;
use prism_core::Value;
use prism_parser::parse;
use prism_vm::VirtualMachine;
use std::sync::Arc;

/// Helper to run Python source code and return result.
fn execute(source: &str) -> Result<Value, String> {
    // Parse
    let module = parse(source).map_err(|e| format!("Parse error: {:?}", e))?;

    // Compile
    let code = Compiler::compile_module(&module, "<test>")
        .map_err(|e| format!("Compile error: {:?}", e))?;

    // Execute
    let mut vm = VirtualMachine::new();
    vm.execute(Arc::new(code))
        .map_err(|e| format!("Runtime error: {:?}", e))
}

// =============================================================================
// VM Builtin Tests
// =============================================================================

#[test]
fn test_vm_builtins_are_initialized() {
    let vm = VirtualMachine::new();

    // Verify len builtin exists and is an object_ptr
    let len_val = vm.builtins.get("len").expect("len should exist");
    assert!(
        len_val.as_object_ptr().is_some(),
        "len should be object_ptr, got: bits = {:#x}",
        unsafe { std::mem::transmute::<Value, u64>(len_val) }
    );

    // Verify range builtin
    let range_val = vm.builtins.get("range").expect("range should exist");
    assert!(
        range_val.as_object_ptr().is_some(),
        "range should be object_ptr"
    );
}

// =============================================================================
// Arithmetic Tests
// =============================================================================

#[test]
fn test_integer_addition() {
    let result = execute("1 + 2");
    // Module execution returns None (last statement is expression, not returned)
    // For now, just verify it doesn't crash
    assert!(result.is_ok(), "Failed: {:?}", result);
}

#[test]
fn test_integer_multiplication() {
    let result = execute("3 * 4");
    assert!(result.is_ok(), "Failed: {:?}", result);
}

#[test]
fn test_compound_arithmetic() {
    let result = execute("2 + 3 * 4");
    assert!(result.is_ok(), "Failed: {:?}", result);
}

#[test]
fn test_float_division() {
    let result = execute("10.0 / 3.0");
    assert!(result.is_ok(), "Failed: {:?}", result);
}

// =============================================================================
// Variable Tests
// =============================================================================

#[test]
fn test_assignment() {
    let result = execute("x = 42");
    assert!(result.is_ok(), "Failed: {:?}", result);
}

#[test]
fn test_assignment_and_use() {
    let result = execute(
        r#"
x = 10
y = 20
z = x + y
"#,
    );
    assert!(result.is_ok(), "Failed: {:?}", result);
}

#[test]
fn test_augmented_assignment() {
    let result = execute(
        r#"
x = 5
x += 3
"#,
    );
    assert!(result.is_ok(), "Failed: {:?}", result);
}

// =============================================================================
// Control Flow Tests
// =============================================================================

#[test]
fn test_if_statement_true() {
    let result = execute(
        r#"
x = 10
if x > 5:
    y = 1
"#,
    );
    assert!(result.is_ok(), "Failed: {:?}", result);
}

#[test]
fn test_if_else_statement() {
    let result = execute(
        r#"
x = 3
if x > 5:
    y = 1
else:
    y = 0
"#,
    );
    assert!(result.is_ok(), "Failed: {:?}", result);
}

#[test]
fn test_while_loop() {
    let result = execute(
        r#"
x = 0
while x < 5:
    x = x + 1
"#,
    );
    assert!(result.is_ok(), "Failed: {:?}", result);
}

// =============================================================================
// Container Tests
// =============================================================================

#[test]
fn test_list_literal() {
    let result = execute("[1, 2, 3]");
    assert!(result.is_ok(), "Failed: {:?}", result);
}

#[test]
fn test_dict_literal() {
    let result = execute("{\"a\": 1, \"b\": 2}");
    assert!(result.is_ok(), "Failed: {:?}", result);
}

#[test]
fn test_tuple_literal() {
    let result = execute("(1, 2, 3)");
    assert!(result.is_ok(), "Failed: {:?}", result);
}

#[test]
fn test_set_literal() {
    let result = execute("{1, 2, 3}");
    assert!(result.is_ok(), "Failed: {:?}", result);
}

// =============================================================================
// Comparison Tests
// =============================================================================

#[test]
fn test_comparison_operators() {
    let sources = ["1 < 2", "2 <= 2", "3 > 2", "3 >= 3", "1 == 1", "1 != 2"];

    for source in sources {
        let result = execute(source);
        assert!(result.is_ok(), "Failed for '{}': {:?}", source, result);
    }
}

#[test]
fn test_boolean_operators() {
    let sources = ["True and True", "True or False", "not False"];

    for source in sources {
        let result = execute(source);
        assert!(result.is_ok(), "Failed for '{}': {:?}", source, result);
    }
}

// =============================================================================
// Expression Tests
// =============================================================================

#[test]
fn test_parenthesized_expression() {
    let result = execute("(2 + 3) * 4");
    assert!(result.is_ok(), "Failed: {:?}", result);
}

#[test]
fn test_unary_operators() {
    let sources = ["-5", "+5", "not True", "~15"];

    for source in sources {
        let result = execute(source);
        assert!(result.is_ok(), "Failed for '{}': {:?}", source, result);
    }
}

// =============================================================================
// String Tests
// =============================================================================

#[test]
fn test_string_literal() {
    let result = execute("\"hello\"");
    assert!(result.is_ok(), "Failed: {:?}", result);
}

#[test]
fn test_string_assignment() {
    let result = execute("s = \"world\"");
    assert!(result.is_ok(), "Failed: {:?}", result);
}

// =============================================================================
// None and Boolean Tests
// =============================================================================

#[test]
fn test_none_literal() {
    let result = execute("None");
    assert!(result.is_ok(), "Failed: {:?}", result);
}

#[test]
fn test_boolean_literals() {
    let result = execute("True");
    assert!(result.is_ok(), "True failed: {:?}", result);

    let result = execute("False");
    assert!(result.is_ok(), "False failed: {:?}", result);
}

// =============================================================================
// Complex Programs
// =============================================================================

#[test]
fn test_sum_to_n() {
    let result = execute(
        r#"
total = 0
n = 10
i = 1
while i <= n:
    total = total + i
    i = i + 1
"#,
    );
    assert!(result.is_ok(), "Failed: {:?}", result);
}

#[test]
fn test_fibonacci() {
    let result = execute(
        r#"
a = 0
b = 1
n = 10
i = 0
while i < n:
    temp = a + b
    a = b
    b = temp
    i = i + 1
"#,
    );
    assert!(result.is_ok(), "Failed: {:?}", result);
}

#[test]
fn test_nested_loops() {
    let result = execute(
        r#"
result = 0
i = 0
while i < 3:
    j = 0
    while j < 3:
        result = result + 1
        j = j + 1
    i = i + 1
"#,
    );
    assert!(result.is_ok(), "Failed: {:?}", result);
}

// =============================================================================
// Builtin Call Tests
// =============================================================================

#[test]
fn test_builtin_len_call() {
    // Test calling len() builtin
    let result = execute("x = len([1, 2, 3])");
    assert!(result.is_ok(), "Failed: {:?}", result);
}

#[test]
fn test_builtin_abs_call() {
    // Test calling abs() builtin
    let result = execute("x = abs(-5)");
    assert!(result.is_ok(), "Failed: {:?}", result);
}

// =============================================================================
// For-Loop Tests
// =============================================================================

#[test]
fn test_for_loop_range() {
    // Test for-loop over range
    let result = execute(
        r#"
total = 0
for i in range(5):
    total = total + i
"#,
    );
    assert!(result.is_ok(), "Failed: {:?}", result);
}

#[test]
fn test_for_loop_list() {
    // Test for-loop over list
    let result = execute(
        r#"
total = 0
nums = [1, 2, 3, 4, 5]
for n in nums:
    total = total + n
"#,
    );
    assert!(result.is_ok(), "Failed: {:?}", result);
}

#[test]
fn test_nested_for_loops() {
    // Test nested for-loops
    let result = execute(
        r#"
result = 0
for i in range(3):
    for j in range(3):
        result = result + 1
"#,
    );
    assert!(result.is_ok(), "Failed: {:?}", result);
}
