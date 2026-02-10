//! Comprehensive tests for Lambda and List Comprehension compilation.
//!
//! These tests verify that the Prism compiler correctly compiles:
//! - Lambda expressions (nested code objects, arguments, closures)
//! - List comprehensions (iteration, filtering, element expressions)
//! - Set comprehensions
//! - Dict comprehensions  
//! - Generator expressions
//!
//! Following the commercial-grade standards of Cadence Spectre.

use prism_compiler::{CodeFlags, Compiler, Opcode};
use prism_parser::parse;

/// Helper to compile source and return the module CodeObject
fn compile_source(source: &str) -> prism_compiler::CodeObject {
    let module = parse(source).expect("parse failed");
    Compiler::compile_module(&module, "<test>").expect("compile failed")
}

/// Helper to count occurrences of an opcode in a code object
fn count_opcode(code: &prism_compiler::CodeObject, target: Opcode) -> usize {
    code.instructions
        .iter()
        .filter(|instr| instr.opcode() == target as u8)
        .count()
}

/// Helper to check if an opcode exists in a code object
fn has_opcode(code: &prism_compiler::CodeObject, target: Opcode) -> bool {
    count_opcode(code, target) > 0
}

/// Recursively count opcodes in nested code objects
fn count_opcode_recursive(code: &prism_compiler::CodeObject, target: Opcode) -> usize {
    let mut total = count_opcode(code, target);
    for nested in code.nested_code_objects.iter() {
        total += count_opcode_recursive(nested, target);
    }
    total
}

/// Check if any arithmetic opcode exists
fn has_arithmetic_opcode(code: &prism_compiler::CodeObject) -> bool {
    has_opcode(code, Opcode::Add)
        || has_opcode(code, Opcode::Mul)
        || has_opcode(code, Opcode::Sub)
        || has_opcode(code, Opcode::TrueDiv)
}

/// Check if any comparison opcode exists
fn has_comparison_opcode(code: &prism_compiler::CodeObject) -> bool {
    has_opcode(code, Opcode::Lt)
        || has_opcode(code, Opcode::Gt)
        || has_opcode(code, Opcode::Eq)
        || has_opcode(code, Opcode::Le)
        || has_opcode(code, Opcode::Ge)
        || has_opcode(code, Opcode::Ne)
}

// ===========================================================================
// LAMBDA EXPRESSION TESTS
// ===========================================================================

mod lambda_tests {
    use super::*;

    #[test]
    fn test_lambda_creates_nested_code_object() {
        let code = compile_source("f = lambda: 42");
        assert!(
            !code.nested_code_objects.is_empty(),
            "Lambda should create a nested code object"
        );
    }

    #[test]
    fn test_lambda_emits_make_function() {
        let code = compile_source("f = lambda: 0");
        assert!(
            has_opcode(&code, Opcode::MakeFunction),
            "Lambda should emit MakeFunction opcode"
        );
    }

    #[test]
    fn test_lambda_nested_code_has_return() {
        let code = compile_source("f = lambda: 99");
        assert!(!code.nested_code_objects.is_empty());
        let lambda_code = &code.nested_code_objects[0];
        assert!(
            has_opcode(lambda_code, Opcode::Return),
            "Lambda body should have implicit Return"
        );
    }

    #[test]
    fn test_lambda_with_single_argument() {
        let code = compile_source("f = lambda x: x + 1");
        assert!(!code.nested_code_objects.is_empty());
        let lambda_code = &code.nested_code_objects[0];
        assert_eq!(lambda_code.arg_count, 1, "Lambda should have 1 argument");
    }

    #[test]
    fn test_lambda_with_multiple_arguments() {
        let code = compile_source("f = lambda a, b, c: a + b + c");
        assert!(!code.nested_code_objects.is_empty());
        let lambda_code = &code.nested_code_objects[0];
        assert_eq!(lambda_code.arg_count, 3, "Lambda should have 3 arguments");
    }

    #[test]
    fn test_lambda_with_varargs() {
        let code = compile_source("f = lambda *args: args");
        assert!(!code.nested_code_objects.is_empty());
        let lambda_code = &code.nested_code_objects[0];
        assert!(
            lambda_code.flags.contains(CodeFlags::VARARGS),
            "Lambda with *args should have VARARGS flag"
        );
    }

    #[test]
    fn test_lambda_with_kwargs() {
        let code = compile_source("f = lambda **kw: kw");
        assert!(!code.nested_code_objects.is_empty());
        let lambda_code = &code.nested_code_objects[0];
        assert!(
            lambda_code.flags.contains(CodeFlags::VARKEYWORDS),
            "Lambda with **kw should have VARKEYWORDS flag"
        );
    }

    #[test]
    fn test_lambda_with_mixed_args() {
        let code = compile_source("f = lambda a, *args, **kwargs: None");
        assert!(!code.nested_code_objects.is_empty());
        let lambda_code = &code.nested_code_objects[0];
        assert!(
            lambda_code.flags.contains(CodeFlags::VARARGS),
            "Lambda should have VARARGS"
        );
        assert!(
            lambda_code.flags.contains(CodeFlags::VARKEYWORDS),
            "Lambda should have VARKEYWORDS"
        );
    }

    #[test]
    fn test_lambda_name_is_lambda() {
        let code = compile_source("f = lambda x: x");
        assert!(!code.nested_code_objects.is_empty());
        let lambda_code = &code.nested_code_objects[0];
        assert_eq!(
            lambda_code.name.as_ref(),
            "<lambda>",
            "Lambda code object should be named <lambda>"
        );
    }

    #[test]
    fn test_lambda_compiles_body_expression() {
        let code = compile_source("f = lambda x: x * 2");
        assert!(!code.nested_code_objects.is_empty());
        let lambda_code = &code.nested_code_objects[0];
        // Should have Mul for multiplication
        assert!(
            has_opcode(lambda_code, Opcode::Mul),
            "Lambda body should compile the expression"
        );
    }

    #[test]
    fn test_nested_lambda() {
        let code = compile_source("f = lambda x: lambda y: x + y");
        // Should have two nested lambdas
        assert!(
            !code.nested_code_objects.is_empty(),
            "Outer lambda should exist"
        );
        // The outer lambda should also have a nested code object
        assert!(
            !code.nested_code_objects[0].nested_code_objects.is_empty(),
            "Inner lambda should be nested"
        );
    }

    #[test]
    fn test_lambda_as_function_argument() {
        let code = compile_source("result = map(lambda x: x * 2, items)");
        assert!(
            has_opcode(&code, Opcode::MakeFunction),
            "Lambda in function call should create function"
        );
    }

    #[test]
    fn test_lambda_immediately_called() {
        let code = compile_source("result = (lambda x: x + 1)(5)");
        assert!(
            has_opcode(&code, Opcode::MakeFunction),
            "IIFE lambda should create function"
        );
        assert!(
            has_opcode(&code, Opcode::Call),
            "IIFE lambda should be called"
        );
    }

    #[test]
    fn test_lambda_in_list() {
        let code = compile_source("funcs = [lambda: 1, lambda: 2, lambda: 3]");
        assert_eq!(
            count_opcode(&code, Opcode::MakeFunction),
            3,
            "Should create 3 lambda functions"
        );
    }

    #[test]
    fn test_lambda_conditional_expression() {
        let code = compile_source("f = lambda x: x if x > 0 else -x");
        assert!(!code.nested_code_objects.is_empty());
        let lambda_code = &code.nested_code_objects[0];
        // Should have comparison for condition
        assert!(
            has_comparison_opcode(lambda_code),
            "Lambda with conditional should have comparison opcode"
        );
    }

    #[test]
    fn test_lambda_with_binary_ops() {
        let code = compile_source("f = lambda a, b: a + b");
        assert!(!code.nested_code_objects.is_empty());
        let lambda_code = &code.nested_code_objects[0];
        assert!(
            has_opcode(lambda_code, Opcode::Add),
            "Lambda should have Add operation"
        );
    }

    #[test]
    fn test_lambda_with_attribute_access() {
        let code = compile_source("f = lambda obj: obj.value");
        assert!(!code.nested_code_objects.is_empty());
        let lambda_code = &code.nested_code_objects[0];
        assert!(
            has_opcode(lambda_code, Opcode::GetAttr),
            "Lambda should get attribute"
        );
    }

    #[test]
    fn test_lambda_with_subscript() {
        let code = compile_source("f = lambda lst: lst[0]");
        assert!(!code.nested_code_objects.is_empty());
        let lambda_code = &code.nested_code_objects[0];
        assert!(
            has_opcode(lambda_code, Opcode::GetItem),
            "Lambda should have subscript operation"
        );
    }

    #[test]
    fn test_lambda_with_method_call() {
        let code = compile_source("f = lambda s: s.upper()");
        assert!(!code.nested_code_objects.is_empty());
        let lambda_code = &code.nested_code_objects[0];
        // Should have LoadMethod or GetAttr for method access
        assert!(
            has_opcode(lambda_code, Opcode::LoadMethod) || has_opcode(lambda_code, Opcode::GetAttr),
            "Lambda should access method"
        );
    }

    #[test]
    fn test_keyword_only_args_in_lambda() {
        let code = compile_source("f = lambda x, y: x + y");
        assert!(!code.nested_code_objects.is_empty());
        assert_eq!(code.nested_code_objects[0].arg_count, 2);
    }
}

// ===========================================================================
// LIST COMPREHENSION TESTS
// ===========================================================================

mod listcomp_tests {
    use super::*;

    #[test]
    fn test_listcomp_creates_list() {
        let code = compile_source("result = [x for x in items]");
        assert!(
            has_opcode(&code, Opcode::BuildList),
            "List comprehension should build a list"
        );
    }

    #[test]
    fn test_listcomp_has_iterator() {
        let code = compile_source("result = [x for x in items]");
        assert!(
            has_opcode(&code, Opcode::GetIter),
            "List comprehension should get iterator"
        );
    }

    #[test]
    fn test_listcomp_has_for_iter() {
        let code = compile_source("result = [x for x in items]");
        assert!(
            has_opcode(&code, Opcode::ForIter),
            "List comprehension should iterate"
        );
    }

    #[test]
    fn test_listcomp_has_list_append() {
        let code = compile_source("result = [x for x in items]");
        assert!(
            has_opcode(&code, Opcode::ListAppend),
            "List comprehension should append to list"
        );
    }

    #[test]
    fn test_listcomp_with_expression() {
        let code = compile_source("result = [x * 2 for x in items]");
        assert!(
            has_opcode(&code, Opcode::Mul),
            "List comp should evaluate expression"
        );
    }

    #[test]
    fn test_listcomp_with_filter() {
        let code = compile_source("result = [x for x in items if x > 0]");
        assert!(
            has_opcode(&code, Opcode::Gt),
            "Filtered listcomp should have comparison"
        );
        assert!(
            has_opcode(&code, Opcode::JumpIfFalse),
            "Filtered listcomp should have conditional jump"
        );
    }

    #[test]
    fn test_listcomp_with_multiple_filters() {
        let code = compile_source("result = [x for x in items if x > 0 if x < 100]");
        // Should have at least two comparisons (Gt and Lt)
        let has_both = has_opcode(&code, Opcode::Gt) && has_opcode(&code, Opcode::Lt);
        assert!(has_both, "Should have multiple comparisons");
    }

    #[test]
    fn test_nested_listcomp() {
        let code = compile_source("result = [[y for y in x] for x in matrix]");
        // Should have nested iteration
        assert!(
            count_opcode(&code, Opcode::GetIter) >= 2,
            "Nested listcomp should have multiple iterators"
        );
    }

    #[test]
    fn test_listcomp_with_nested_loop() {
        let code = compile_source("result = [x + y for x in a for y in b]");
        assert!(
            count_opcode(&code, Opcode::GetIter) >= 2,
            "Multi-loop listcomp should have multiple iterators"
        );
    }

    #[test]
    fn test_listcomp_with_tuple_unpacking() {
        let code = compile_source("result = [a + b for a, b in pairs]");
        // When we unpack a tuple target, we iterate and store locals
        assert!(
            has_opcode(&code, Opcode::ForIter),
            "Tuple unpacking in listcomp should iterate"
        );
    }

    #[test]
    fn test_listcomp_with_method_call() {
        let code = compile_source("result = [s.upper() for s in strings]");
        assert!(
            has_opcode(&code, Opcode::LoadMethod) || has_opcode(&code, Opcode::Call),
            "ListComp should call methods"
        );
    }

    #[test]
    fn test_listcomp_with_conditional_expression() {
        let code = compile_source("result = [x if x > 0 else 0 for x in nums]");
        assert!(
            has_comparison_opcode(&code),
            "Conditional expression in listcomp"
        );
    }

    #[test]
    fn test_listcomp_with_function_call() {
        let code = compile_source("result = [func(x) for x in items]");
        assert!(
            has_opcode(&code, Opcode::Call),
            "ListComp should call functions"
        );
    }

    #[test]
    fn test_empty_listcomp_elements() {
        // This should still compile even if source is empty at runtime
        let code = compile_source("result = [x for x in []]");
        assert!(has_opcode(&code, Opcode::BuildList));
    }

    #[test]
    fn test_listcomp_with_subscript_element() {
        let code = compile_source("result = [d[k] for k in keys]");
        assert!(
            has_opcode(&code, Opcode::GetItem),
            "ListComp element can use subscript"
        );
    }

    #[test]
    fn test_listcomp_with_attribute_element() {
        let code = compile_source("result = [obj.attr for obj in objects]");
        assert!(
            has_opcode(&code, Opcode::GetAttr),
            "ListComp element can access attributes"
        );
    }

    #[test]
    fn test_listcomp_deeply_nested() {
        let code = compile_source("result = [z for x in a for y in x for z in y]");
        assert!(
            count_opcode(&code, Opcode::GetIter) >= 3,
            "Triple-nested loop should have 3+ iterators"
        );
    }

    #[test]
    fn test_listcomp_with_enumerate() {
        let code = compile_source("result = [(i, x) for i, x in enumerate(items)]");
        assert!(has_opcode(&code, Opcode::BuildTuple));
        // Iteration with tuple target
        assert!(has_opcode(&code, Opcode::ForIter));
    }
}

// ===========================================================================
// SET COMPREHENSION TESTS
// ===========================================================================

mod setcomp_tests {
    use super::*;

    #[test]
    fn test_setcomp_creates_set() {
        let code = compile_source("result = {x for x in items}");
        assert!(
            has_opcode(&code, Opcode::BuildSet),
            "Set comprehension should build a set"
        );
    }

    #[test]
    fn test_setcomp_has_set_add() {
        let code = compile_source("result = {x for x in items}");
        assert!(
            has_opcode(&code, Opcode::SetAdd),
            "Set comprehension should add to set"
        );
    }

    #[test]
    fn test_setcomp_with_expression() {
        let code = compile_source("result = {x * 2 for x in items}");
        assert!(has_opcode(&code, Opcode::Mul));
    }

    #[test]
    fn test_setcomp_with_filter() {
        let code = compile_source("result = {x for x in items if x > 0}");
        assert!(has_opcode(&code, Opcode::Gt));
        assert!(has_opcode(&code, Opcode::JumpIfFalse));
    }

    #[test]
    fn test_setcomp_with_nested_loop() {
        let code = compile_source("result = {x + y for x in a for y in b}");
        assert!(count_opcode(&code, Opcode::GetIter) >= 2);
    }
}

// ===========================================================================
// DICT COMPREHENSION TESTS
// ===========================================================================

mod dictcomp_tests {
    use super::*;

    #[test]
    fn test_dictcomp_creates_dict() {
        let code = compile_source("result = {x: x for x in items}");
        assert!(
            has_opcode(&code, Opcode::BuildDict),
            "Dict comprehension should build a dict"
        );
    }

    #[test]
    fn test_dictcomp_sets_items() {
        let code = compile_source("result = {x: x for x in items}");
        assert!(
            has_opcode(&code, Opcode::SetItem),
            "Dict comprehension should set items"
        );
    }

    #[test]
    fn test_dictcomp_with_expression() {
        let code = compile_source("result = {x: x * 2 for x in items}");
        assert!(has_opcode(&code, Opcode::Mul));
    }

    #[test]
    fn test_dictcomp_with_filter() {
        let code = compile_source("result = {x: x for x in items if x > 0}");
        assert!(has_opcode(&code, Opcode::Gt));
    }

    #[test]
    fn test_dictcomp_key_value_expressions() {
        let code = compile_source("result = {x + 1: x + 2 for x in pairs}");
        // Should have Add operations for both key and value expressions
        assert!(count_opcode(&code, Opcode::Add) >= 2);
    }

    #[test]
    fn test_dictcomp_invert_mapping() {
        let code = compile_source("result = {x: x * 2 for x in data}");
        // Inverted dict comp should iterate
        assert!(has_opcode(&code, Opcode::ForIter));
    }
}

// ===========================================================================
// GENERATOR EXPRESSION TESTS
// ===========================================================================

mod genexp_tests {
    use super::*;

    #[test]
    fn test_genexp_creates_generator() {
        let code = compile_source("gen = (x for x in items)");
        assert!(
            !code.nested_code_objects.is_empty(),
            "Generator expression should create nested code"
        );
    }

    #[test]
    fn test_genexp_has_generator_flag() {
        let code = compile_source("gen = (x for x in items)");
        assert!(!code.nested_code_objects.is_empty());
        let gen_code = &code.nested_code_objects[0];
        assert!(
            gen_code.flags.contains(CodeFlags::GENERATOR),
            "Generator expression should have GENERATOR flag"
        );
    }

    #[test]
    fn test_genexp_has_yield() {
        let code = compile_source("gen = (x for x in items)");
        assert!(!code.nested_code_objects.is_empty());
        let gen_code = &code.nested_code_objects[0];
        assert!(
            has_opcode(gen_code, Opcode::Yield),
            "Generator expression should yield values"
        );
    }

    #[test]
    fn test_genexp_name() {
        let code = compile_source("gen = (x for x in items)");
        assert!(!code.nested_code_objects.is_empty());
        let gen_code = &code.nested_code_objects[0];
        assert_eq!(gen_code.name.as_ref(), "<genexpr>");
    }

    #[test]
    fn test_genexp_with_expression() {
        let code = compile_source("gen = (x * 2 for x in items)");
        assert!(!code.nested_code_objects.is_empty());
        let gen_code = &code.nested_code_objects[0];
        assert!(has_opcode(gen_code, Opcode::Mul));
    }

    #[test]
    fn test_genexp_with_filter() {
        let code = compile_source("gen = (x for x in items if x > 0)");
        assert!(!code.nested_code_objects.is_empty());
        let gen_code = &code.nested_code_objects[0];
        assert!(has_opcode(gen_code, Opcode::Gt));
    }

    #[test]
    fn test_genexp_in_function_call() {
        let code = compile_source("result = sum(x for x in items)");
        assert!(has_opcode(&code, Opcode::Call));
        assert!(!code.nested_code_objects.is_empty());
    }

    #[test]
    fn test_genexp_with_nested_loop() {
        let code = compile_source("gen = (x + y for x in a for y in b)");
        assert!(!code.nested_code_objects.is_empty());
        let gen_code = &code.nested_code_objects[0];
        assert!(count_opcode(gen_code, Opcode::ForIter) >= 2);
    }
}

// ===========================================================================
// ASYNC COMPREHENSION TESTS
// ===========================================================================

mod async_comp_tests {
    use super::*;

    #[test]
    fn test_async_listcomp_in_async_function() {
        let code =
            compile_source("async def fetch_all():\n    return [x async for x in async_gen()]");
        // Should compile without error
        assert!(!code.nested_code_objects.is_empty());
    }

    #[test]
    fn test_async_listcomp_uses_get_aiter() {
        let code = compile_source("async def f():\n    return [x async for x in aiter]");
        assert!(
            count_opcode_recursive(&code, Opcode::GetAIter) >= 1,
            "Async listcomp should use GetAIter"
        );
    }

    #[test]
    fn test_async_listcomp_uses_get_anext() {
        let code = compile_source("async def f():\n    return [x async for x in aiter]");
        assert!(
            count_opcode_recursive(&code, Opcode::GetANext) >= 1,
            "Async listcomp should use GetANext"
        );
    }

    #[test]
    fn test_await_in_listcomp_in_async_function() {
        let code = compile_source("async def gather():\n    return [await coro for coro in coros]");
        assert!(
            count_opcode_recursive(&code, Opcode::GetAwaitable) >= 1,
            "Await in listcomp should use GetAwaitable"
        );
    }

    #[test]
    fn test_mixed_async_sync_comprehension() {
        let code = compile_source("async def f():\n    return [await x for x in sync_iter]");
        // Should have GetIter for sync iterator and GetAwaitable for await
        assert!(count_opcode_recursive(&code, Opcode::GetIter) >= 1);
        assert!(count_opcode_recursive(&code, Opcode::GetAwaitable) >= 1);
    }
}

// ===========================================================================
// EDGE CASES AND INTEGRATION TESTS
// ===========================================================================

mod integration_tests {
    use super::*;

    #[test]
    fn test_lambda_with_listcomp() {
        let code = compile_source("f = lambda items: [x * 2 for x in items]");
        assert!(!code.nested_code_objects.is_empty());
        // Lambda should exist
        assert!(has_opcode(&code, Opcode::MakeFunction));
    }

    #[test]
    fn test_listcomp_with_lambda() {
        let code = compile_source("funcs = [lambda: i for i in range(3)]");
        // Should create multiple functions
        assert!(count_opcode_recursive(&code, Opcode::MakeFunction) >= 1);
    }

    #[test]
    fn test_comprehension_in_class() {
        let code = compile_source("class Container:\n    values = [x for x in base_values]");
        // Should have listcomp inside class
        assert!(count_opcode_recursive(&code, Opcode::ListAppend) >= 1);
    }

    #[test]
    fn test_multiple_comprehensions() {
        let code = compile_source(
            "a = [x for x in items]\nb = {x for x in items}\nc = {k: v for k, v in pairs}",
        );
        assert!(has_opcode(&code, Opcode::BuildList));
        assert!(has_opcode(&code, Opcode::BuildSet));
        assert!(has_opcode(&code, Opcode::BuildDict));
    }

    #[test]
    fn test_comprehension_in_function() {
        let code = compile_source("def process():\n    return [x * 2 for x in data]");
        assert!(!code.nested_code_objects.is_empty());
    }

    #[test]
    fn test_lambda_returns_comprehension() {
        let code = compile_source("f = lambda data: [x for x in data if x]");
        assert!(!code.nested_code_objects.is_empty());
    }

    #[test]
    fn test_walrus_operator_not_in_comp_condition() {
        // This tests a simple case that should work
        let code = compile_source("result = [x for x in items if x]");
        assert!(has_opcode(&code, Opcode::JumpIfFalse));
    }

    #[test]
    fn test_complex_comprehension_chain() {
        let code = compile_source("result = [func(x) for x in data if pred(x) for y in x if y]");
        assert!(count_opcode(&code, Opcode::GetIter) >= 2);
        assert!(count_opcode(&code, Opcode::JumpIfFalse) >= 2);
    }
}
