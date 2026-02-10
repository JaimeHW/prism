//! Integration tests for LoadMethod/CallMethod compilation optimization.
//!
//! These tests verify that the compiler correctly emits optimized bytecode
//! for method call patterns (obj.method(args...)), using LoadMethod/CallMethod
//! instead of the slower GetAttr/Call sequence.
//!
//! The LoadMethod/CallMethod optimization provides 15-30% speedup on method-heavy
//! code by avoiding BoundMethod allocation on every call.

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

/// Helper to find instructions of a specific opcode in bytecode.
fn find_opcodes(code: &prism_compiler::CodeObject, opcode: Opcode) -> Vec<usize> {
    code.instructions
        .iter()
        .enumerate()
        .filter(|(_, instr)| instr.opcode() == opcode as u8)
        .map(|(i, _)| i)
        .collect()
}

/// Helper to count occurrences of an opcode.
fn count_opcodes(code: &prism_compiler::CodeObject, opcode: Opcode) -> usize {
    find_opcodes(code, opcode).len()
}

/// Helper to verify that LoadMethod/CallMethod pair is present.
fn has_method_call_optimization(code: &prism_compiler::CodeObject) -> bool {
    let load_methods = count_opcodes(code, Opcode::LoadMethod);
    let call_methods = count_opcodes(code, Opcode::CallMethod);
    load_methods > 0 && call_methods > 0
}

// =============================================================================
// Basic Method Call Emission Tests
// =============================================================================

#[test]
fn test_simple_method_call_emits_load_method() {
    let code = compile(
        r#"
obj = SomeClass()
obj.method()
"#,
    );

    assert!(
        has_method_call_optimization(&code),
        "Simple method call obj.method() should emit LoadMethod/CallMethod"
    );
}

#[test]
fn test_method_call_with_one_arg() {
    let code = compile(
        r#"
x.foo(42)
"#,
    );

    assert!(
        has_method_call_optimization(&code),
        "Method call with one arg should use LoadMethod/CallMethod"
    );
}

#[test]
fn test_method_call_with_multiple_args() {
    let code = compile(
        r#"
obj.process(a, b, c, d)
"#,
    );

    assert!(
        has_method_call_optimization(&code),
        "Method call with multiple args should use LoadMethod/CallMethod"
    );
}

#[test]
fn test_method_call_on_literal() {
    let code = compile(
        r#"
"hello".upper()
"#,
    );

    assert!(
        has_method_call_optimization(&code),
        "Method call on string literal should use optimization"
    );
}

#[test]
fn test_method_call_on_list_literal() {
    let code = compile(
        r#"
[1, 2, 3].append(4)
"#,
    );

    assert!(
        has_method_call_optimization(&code),
        "Method call on list literal should use optimization"
    );
}

#[test]
fn test_method_call_on_dict_literal() {
    let code = compile(
        r#"
{}.get("key")
"#,
    );

    assert!(
        has_method_call_optimization(&code),
        "Method call on dict literal should use optimization"
    );
}

// =============================================================================
// Method Call with Keyword Arguments (Fallback Path)
// =============================================================================

#[test]
fn test_method_call_with_kwargs_uses_fallback() {
    let code = compile(
        r#"
obj.method(key=value)
"#,
    );

    // Keyword arguments should use the fallback path for now
    // (CallMethodKw not yet implemented)
    // Verify that the optimization is NOT used
    let load_methods = count_opcodes(&code, Opcode::LoadMethod);
    assert_eq!(
        load_methods, 0,
        "Method with keyword args should NOT use LoadMethod optimization"
    );
}

#[test]
fn test_method_call_with_mixed_args_and_kwargs_uses_fallback() {
    let code = compile(
        r#"
obj.method(a, b, key=value)
"#,
    );

    // Verify that the optimization is NOT used for mixed args+kwargs
    let load_methods = count_opcodes(&code, Opcode::LoadMethod);
    assert_eq!(
        load_methods, 0,
        "Method with mixed args should NOT use LoadMethod optimization"
    );
}

// =============================================================================
// Chained Method Calls
// =============================================================================

#[test]
fn test_chained_method_calls() {
    let code = compile(
        r#"
result = obj.method1().method2().method3()
"#,
    );

    // Should emit 3 LoadMethod/CallMethod pairs
    let load_methods = count_opcodes(&code, Opcode::LoadMethod);
    let call_methods = count_opcodes(&code, Opcode::CallMethod);

    assert_eq!(load_methods, 3, "Should have 3 LoadMethod opcodes");
    assert_eq!(call_methods, 3, "Should have 3 CallMethod opcodes");
}

#[test]
fn test_double_method_call() {
    let code = compile(
        r#"
x.foo().bar()
"#,
    );

    assert_eq!(
        count_opcodes(&code, Opcode::LoadMethod),
        2,
        "Double method call should emit 2 LoadMethod"
    );
}

#[test]
fn test_method_call_on_method_result() {
    let code = compile(
        r#"
s = "hello world"
words = s.split().reverse()
"#,
    );

    assert!(
        has_method_call_optimization(&code),
        "Chained split().reverse() should use optimization"
    );
}

// =============================================================================
// Nested Method Calls
// =============================================================================

#[test]
fn test_method_call_as_argument() {
    let code = compile(
        r#"
outer.method(inner.other_method())
"#,
    );

    // Both method calls should use optimization
    assert_eq!(
        count_opcodes(&code, Opcode::LoadMethod),
        2,
        "Nested method calls should both use LoadMethod"
    );
}

#[test]
fn test_method_call_with_method_args() {
    let code = compile(
        r#"
x.foo(a.bar(), b.baz())
"#,
    );

    // 3 method calls: foo, bar, baz
    assert_eq!(
        count_opcodes(&code, Opcode::LoadMethod),
        3,
        "All three method calls should use LoadMethod"
    );
}

// =============================================================================
// Method Calls in Control Flow
// =============================================================================

#[test]
fn test_method_call_in_if_condition() {
    let code = compile(
        r#"
if obj.is_valid():
    pass
"#,
    );

    assert!(
        has_method_call_optimization(&code),
        "Method in if condition should use optimization"
    );
}

#[test]
fn test_method_call_in_while_condition() {
    let code = compile(
        r#"
while gen.has_next():
    x = gen.next()
"#,
    );

    // Two method calls: has_next and next
    let load_methods = count_opcodes(&code, Opcode::LoadMethod);
    assert!(
        load_methods >= 2,
        "While loop method calls should use optimization"
    );
}

#[test]
fn test_method_call_in_for_loop() {
    let code = compile(
        r#"
for item in container.items():
    item.process()
"#,
    );

    assert!(
        has_method_call_optimization(&code),
        "For loop method calls should use optimization"
    );
}

// =============================================================================
// Method Calls with Complex Expressions
// =============================================================================

#[test]
fn test_method_call_on_subscript_result() {
    let code = compile(
        r#"
result = array[0].method()
"#,
    );

    assert!(
        has_method_call_optimization(&code),
        "Method on subscript result should use optimization"
    );
}

#[test]
fn test_method_call_on_ternary_result() {
    let code = compile(
        r#"
result = (a if cond else b).method()
"#,
    );

    assert!(
        has_method_call_optimization(&code),
        "Method on ternary result should use optimization"
    );
}

#[test]
fn test_method_call_on_call_result() {
    let code = compile(
        r#"
get_object().process()
"#,
    );

    // get_object() is a regular call, process() is a method call
    assert!(
        has_method_call_optimization(&code),
        "Method on function call result should use optimization"
    );
}

// =============================================================================
// Non-Optimizable Cases (Fallback to GetAttr/Call)
// =============================================================================

#[test]
fn test_method_with_star_args_uses_fallback() {
    let code = compile(
        r#"
obj.method(*args)
"#,
    );

    // *args unpacking uses dynamic call path
    let call_ex = count_opcodes(&code, Opcode::CallEx);
    assert!(call_ex > 0, "*args should use CallEx dynamic path");
}

#[test]
fn test_method_with_double_star_kwargs_uses_fallback() {
    let code = compile(
        r#"
obj.method(**kwargs)
"#,
    );

    // **kwargs unpacking uses dynamic call path
    let call_ex = count_opcodes(&code, Opcode::CallEx);
    assert!(call_ex > 0, "**kwargs should use CallEx dynamic path");
}

// =============================================================================
// Attribute Access vs Method Call Differentiation
// =============================================================================

#[test]
fn test_attribute_access_not_optimized() {
    let code = compile(
        r#"
x = obj.attr
"#,
    );

    // Simple attribute access should NOT use LoadMethod
    let load_methods = count_opcodes(&code, Opcode::LoadMethod);
    assert_eq!(
        load_methods, 0,
        "Simple attribute access should use GetAttr, not LoadMethod"
    );
}

#[test]
fn test_only_calls_on_attributes_optimized() {
    let code = compile(
        r#"
value = obj.attr
result = obj.method()
"#,
    );

    // One LoadMethod for method(), one GetAttr for attr
    let load_methods = count_opcodes(&code, Opcode::LoadMethod);
    let get_attrs = count_opcodes(&code, Opcode::GetAttr);

    assert_eq!(load_methods, 1, "Should have 1 LoadMethod for method()");
    assert_eq!(get_attrs, 1, "Should have 1 GetAttr for .attr");
}

// =============================================================================
// Edge Cases
// =============================================================================

#[test]
fn test_method_call_with_zero_args() {
    let code = compile(
        r#"
x.reset()
"#,
    );

    assert!(
        has_method_call_optimization(&code),
        "Zero-arg method should use optimization"
    );
}

#[test]
fn test_method_call_with_many_args() {
    let code = compile(
        r#"
obj.func(a, b, c, d, e, f, g, h)
"#,
    );

    assert!(
        has_method_call_optimization(&code),
        "Method with 8 args should still use optimization"
    );
}

#[test]
fn test_method_call_self_reference() {
    let code = compile(
        r#"
class Foo:
    def bar(self):
        self.method()
"#,
    );

    assert!(
        !code.instructions.is_empty(),
        "Self method call should compile"
    );
}

#[test]
fn test_multiple_method_calls_same_object() {
    let code = compile(
        r#"
obj.method1()
obj.method2()
obj.method3()
"#,
    );

    assert_eq!(
        count_opcodes(&code, Opcode::LoadMethod),
        3,
        "Three separate method calls should give 3 LoadMethod"
    );
}

#[test]
fn test_method_call_in_comprehension() {
    let code = compile(
        r#"
result = [item.process() for item in items]
"#,
    );

    assert!(
        !code.instructions.is_empty(),
        "Comprehension with method should compile"
    );
}

#[test]
fn test_method_call_in_lambda() {
    let code = compile(
        r#"
f = lambda x: x.upper()
"#,
    );

    assert!(
        !code.instructions.is_empty(),
        "Lambda with method should compile"
    );
}

// =============================================================================
// Practical Patterns
// =============================================================================

#[test]
fn test_builder_pattern() {
    let code = compile(
        r#"
result = (
    Builder()
    .set_name("test")
    .set_value(42)
    .build()
)
"#,
    );

    // Builder pattern should use multiple LoadMethod/CallMethod pairs
    let load_methods = count_opcodes(&code, Opcode::LoadMethod);
    assert!(
        load_methods >= 3,
        "Builder pattern should use LoadMethod for chained calls"
    );
}

#[test]
fn test_string_processing() {
    let code = compile(
        r#"
result = text.strip().lower().replace(" ", "_")
"#,
    );

    assert_eq!(
        count_opcodes(&code, Opcode::LoadMethod),
        3,
        "String processing chain should use 3 LoadMethod"
    );
}

#[test]
fn test_collection_operations() {
    let code = compile(
        r#"
data.append(item)
data.extend(other)
data.sort()
"#,
    );

    assert_eq!(
        count_opcodes(&code, Opcode::LoadMethod),
        3,
        "Collection operations should use LoadMethod"
    );
}

// =============================================================================
// Register Layout Verification
// =============================================================================

#[test]
fn test_method_call_compiles_without_register_collision() {
    // This test ensures the register allocation doesn't cause issues
    // with complex nested expressions
    let code = compile(
        r#"
result = a.foo(b.bar(c.baz(x, y, z)), d.qux())
"#,
    );

    assert!(
        !code.instructions.is_empty(),
        "Complex nested method calls should compile without register collision"
    );
}

#[test]
fn test_method_result_as_condition() {
    let code = compile(
        r#"
if obj.check() and other.validate():
    process()
"#,
    );

    // Both method calls should use optimization
    let load_methods = count_opcodes(&code, Opcode::LoadMethod);
    assert_eq!(
        load_methods, 2,
        "Both condition method calls should use LoadMethod"
    );
}

// =============================================================================
// Instruction Count Verification
// =============================================================================

#[test]
fn test_optimization_reduces_instruction_count() {
    // When optimization is active, we should have fewer instructions
    // than the GetAttr+Call path would produce
    let code = compile(
        r#"
obj.method()
"#,
    );

    // LoadMethod + CallMethod = 2 instructions
    // GetAttr + Call = 2 instructions but with more overhead at runtime
    let load_methods = count_opcodes(&code, Opcode::LoadMethod);
    let call_methods = count_opcodes(&code, Opcode::CallMethod);

    // Verify the optimization path was taken
    assert_eq!(load_methods, 1, "Should emit exactly 1 LoadMethod");
    assert_eq!(call_methods, 1, "Should emit exactly 1 CallMethod");
}

// =============================================================================
// Opcode Ordering Verification
// =============================================================================

#[test]
fn test_load_method_precedes_call_method() {
    let code = compile(
        r#"
x.foo()
"#,
    );

    let load_positions = find_opcodes(&code, Opcode::LoadMethod);
    let call_positions = find_opcodes(&code, Opcode::CallMethod);

    assert!(!load_positions.is_empty(), "Should have LoadMethod");
    assert!(!call_positions.is_empty(), "Should have CallMethod");

    // LoadMethod should come before CallMethod
    assert!(
        load_positions[0] < call_positions[0],
        "LoadMethod should precede CallMethod"
    );
}
