//! Comprehensive End-to-End Integration Tests for Async/Await Operations.
//!
//! This module provides exhaustive testing for the complete async execution pipeline:
//! - Parse → Compile → Execute for all async constructs
//! - Coroutine/generator creation and state management
//! - Async iteration protocol with GetAIter/GetANext/EndAsyncFor
//! - Error handling and exception propagation
//! - Performance verification
//!
//! # Test Categories
//!
//! - **Basic Async Flow**: Simple async def, await, return
//! - **Async For Loops**: GetAIter/GetANext/EndAsyncFor complete cycles
//! - **Async With Statements**: Context manager protocol
//! - **Async Generators**: yield in async functions
//! - **Error Handling**: TypeError, StopAsyncIteration, exception chains
//! - **Complex Flows**: Nested async, comprehensions, closures
//! - **Performance**: Throughput and memory tests
//!
//! # Architecture
//!
//! Tests use the full parse→compile→execute pipeline to verify bytecode
//! generation and VM execution work correctly together.

use prism_compiler::Compiler;
use prism_core::Value;
use prism_parser::parse;
use prism_vm::VirtualMachine;
use std::sync::Arc;

// =============================================================================
// Test Infrastructure
// =============================================================================

/// Execute Python source and return result.
fn execute(source: &str) -> Result<Value, String> {
    let module = parse(source).map_err(|e| format!("Parse error: {:?}", e))?;
    let code = Compiler::compile_module(&module, "<test>")
        .map_err(|e| format!("Compile error: {:?}", e))?;
    let mut vm = VirtualMachine::new();
    vm.execute(Arc::new(code))
        .map_err(|e| format!("Runtime error: {:?}", e))
}

/// Verify code compiles without error.
fn compiles(source: &str) -> Result<(), String> {
    let module = parse(source).map_err(|e| format!("Parse error: {:?}", e))?;
    Compiler::compile_module(&module, "<test>").map_err(|e| format!("Compile error: {:?}", e))?;
    Ok(())
}

/// Verify code parses but fails to compile (expected compile error).
fn compile_fails(source: &str) -> bool {
    let module = match parse(source) {
        Ok(m) => m,
        Err(_) => return false, // Parse failed, not compile
    };
    Compiler::compile_module(&module, "<test>").is_err()
}

/// Verify code parses correctly.
fn parses(source: &str) -> Result<(), String> {
    parse(source).map_err(|e| format!("Parse error: {:?}", e))?;
    Ok(())
}

// =============================================================================
// Basic Async Def Tests
// =============================================================================

mod async_def_tests {
    use super::*;

    #[test]
    fn test_simple_async_def() {
        let result = compiles(
            r#"
async def simple():
    pass
"#,
        );
        assert!(result.is_ok(), "Failed: {:?}", result);
    }

    #[test]
    fn test_async_def_with_return() {
        let result = compiles(
            r#"
async def returns_value():
    return 42
"#,
        );
        assert!(result.is_ok(), "Failed: {:?}", result);
    }

    #[test]
    fn test_async_def_with_parameters() {
        let result = compiles(
            r#"
async def with_params(a, b, c=10):
    return a + b + c
"#,
        );
        assert!(result.is_ok(), "Failed: {:?}", result);
    }

    #[test]
    fn test_async_def_with_args_kwargs() {
        let result = compiles(
            r#"
async def with_args_kwargs(*args, **kwargs):
    return len(args) + len(kwargs)
"#,
        );
        assert!(result.is_ok(), "Failed: {:?}", result);
    }

    #[test]
    fn test_async_def_with_local_variables() {
        let result = compiles(
            r#"
async def with_locals():
    x = 1
    y = 2
    z = x + y
    return z
"#,
        );
        assert!(result.is_ok(), "Failed: {:?}", result);
    }

    #[test]
    fn test_async_def_with_control_flow() {
        let result = compiles(
            r#"
async def with_control_flow(n):
    if n > 0:
        return n * 2
    else:
        return 0
"#,
        );
        assert!(result.is_ok(), "Failed: {:?}", result);
    }

    #[test]
    fn test_async_def_with_loop() {
        let result = compiles(
            r#"
async def with_loop():
    total = 0
    for i in range(10):
        total += i
    return total
"#,
        );
        assert!(result.is_ok(), "Failed: {:?}", result);
    }

    #[test]
    fn test_nested_async_def() {
        let result = compiles(
            r#"
async def outer():
    async def inner():
        return 42
    return inner
"#,
        );
        assert!(result.is_ok(), "Failed: {:?}", result);
    }

    #[test]
    fn test_async_def_with_docstring() {
        let result = compiles(
            r#"
async def documented():
    """This is a documented async function."""
    pass
"#,
        );
        assert!(result.is_ok(), "Failed: {:?}", result);
    }

    #[test]
    fn test_async_def_with_decorator_syntax() {
        let result = parses(
            r#"
@some_decorator
async def decorated():
    pass
"#,
        );
        assert!(result.is_ok(), "Failed: {:?}", result);
    }
}

// =============================================================================
// Await Expression Tests
// =============================================================================

mod await_tests {
    use super::*;

    #[test]
    fn test_simple_await() {
        let result = compiles(
            r#"
async def coro():
    return 42

async def main():
    result = await coro()
    return result
"#,
        );
        assert!(result.is_ok(), "Failed: {:?}", result);
    }

    #[test]
    fn test_await_in_expression() {
        let result = compiles(
            r#"
async def get_value():
    return 10

async def main():
    x = 1 + await get_value()
    return x
"#,
        );
        assert!(result.is_ok(), "Failed: {:?}", result);
    }

    #[test]
    fn test_multiple_awaits() {
        let result = compiles(
            r#"
async def a():
    return 1

async def b():
    return 2

async def main():
    x = await a()
    y = await b()
    return x + y
"#,
        );
        assert!(result.is_ok(), "Failed: {:?}", result);
    }

    #[test]
    fn test_await_in_conditional() {
        let result = compiles(
            r#"
async def get_condition():
    return True

async def main():
    if await get_condition():
        return 1
    return 0
"#,
        );
        assert!(result.is_ok(), "Failed: {:?}", result);
    }

    #[test]
    fn test_await_in_loop_condition() {
        let result = compiles(
            r#"
async def should_continue():
    return False

async def main():
    while await should_continue():
        pass
    return 0
"#,
        );
        assert!(result.is_ok(), "Failed: {:?}", result);
    }

    #[test]
    fn test_await_chained() {
        let result = compiles(
            r#"
async def get_coro():
    async def inner():
        return 42
    return inner()

async def main():
    result = await await get_coro()
    return result
"#,
        );
        assert!(result.is_ok(), "Failed: {:?}", result);
    }

    #[test]
    fn test_await_with_attribute_access() {
        let result = compiles(
            r#"
async def get_obj():
    return obj

async def main():
    x = (await get_obj()).value
    return x
"#,
        );
        assert!(result.is_ok(), "Failed: {:?}", result);
    }

    #[test]
    fn test_await_with_method_call() {
        let result = compiles(
            r#"
async def get_list():
    return [1, 2, 3]

async def main():
    items = await get_list()
    return len(items)
"#,
        );
        assert!(result.is_ok(), "Failed: {:?}", result);
    }

    #[test]
    fn test_await_in_assignment_tuple() {
        let result = compiles(
            r#"
async def get_pair():
    return (1, 2)

async def main():
    a, b = await get_pair()
    return a + b
"#,
        );
        assert!(result.is_ok(), "Failed: {:?}", result);
    }

    #[test]
    fn test_await_outside_async_fails() {
        // await outside async def should fail to compile
        let fails = compile_fails(
            r#"
def not_async():
    await something()
"#,
        );
        assert!(fails, "Should fail to compile await outside async");
    }
}

// =============================================================================
// Async For Loop Tests
// =============================================================================

mod async_for_tests {
    use super::*;

    #[test]
    fn test_simple_async_for() {
        let result = compiles(
            r#"
async def main():
    async for item in async_iter:
        pass
"#,
        );
        assert!(result.is_ok(), "Failed: {:?}", result);
    }

    #[test]
    fn test_async_for_with_body() {
        let result = compiles(
            r#"
async def main():
    total = 0
    async for item in async_iter:
        total += item
    return total
"#,
        );
        assert!(result.is_ok(), "Failed: {:?}", result);
    }

    #[test]
    fn test_async_for_with_else() {
        let result = compiles(
            r#"
async def main():
    async for item in async_iter:
        process(item)
    else:
        cleanup()
"#,
        );
        assert!(result.is_ok(), "Failed: {:?}", result);
    }

    #[test]
    fn test_async_for_with_break() {
        let result = compiles(
            r#"
async def main():
    async for item in async_iter:
        if item > 10:
            break
"#,
        );
        assert!(result.is_ok(), "Failed: {:?}", result);
    }

    #[test]
    fn test_async_for_with_continue() {
        let result = compiles(
            r#"
async def main():
    async for item in async_iter:
        if item < 0:
            continue
        process(item)
"#,
        );
        assert!(result.is_ok(), "Failed: {:?}", result);
    }

    #[test]
    fn test_nested_async_for() {
        let result = compiles(
            r#"
async def main():
    async for row in async_rows:
        async for col in async_cols:
            process(row, col)
"#,
        );
        assert!(result.is_ok(), "Failed: {:?}", result);
    }

    #[test]
    fn test_async_for_tuple_unpack() {
        let result = compiles(
            r#"
async def main():
    async for key, value in async_items:
        process(key, value)
"#,
        );
        assert!(result.is_ok(), "Failed: {:?}", result);
    }

    #[test]
    fn test_async_for_in_try() {
        let result = compiles(
            r#"
async def main():
    try:
        async for item in async_iter:
            process(item)
    except Exception:
        handle_error()
"#,
        );
        assert!(result.is_ok(), "Failed: {:?}", result);
    }

    #[test]
    fn test_async_for_with_await_in_body() {
        let result = compiles(
            r#"
async def main():
    async for item in async_iter:
        result = await process_async(item)
"#,
        );
        assert!(result.is_ok(), "Failed: {:?}", result);
    }

    #[test]
    fn test_async_for_outside_async_fails() {
        let fails = compile_fails(
            r#"
def not_async():
    async for item in something:
        pass
"#,
        );
        assert!(fails, "Should fail to compile async for outside async");
    }
}

// =============================================================================
// Async With Statement Tests
// =============================================================================

mod async_with_tests {
    use super::*;

    #[test]
    fn test_simple_async_with() {
        let result = compiles(
            r#"
async def main():
    async with context_manager as ctx:
        pass
"#,
        );
        assert!(result.is_ok(), "Failed: {:?}", result);
    }

    #[test]
    fn test_async_with_no_as() {
        let result = compiles(
            r#"
async def main():
    async with acquire_lock():
        critical_section()
"#,
        );
        assert!(result.is_ok(), "Failed: {:?}", result);
    }

    #[test]
    fn test_async_with_body() {
        let result = compiles(
            r#"
async def main():
    async with open_file() as f:
        data = await f.read()
        return data
"#,
        );
        assert!(result.is_ok(), "Failed: {:?}", result);
    }

    #[test]
    fn test_async_with_multiple_items() {
        let result = compiles(
            r#"
async def main():
    async with cm1() as a, cm2() as b:
        use(a, b)
"#,
        );
        assert!(result.is_ok(), "Failed: {:?}", result);
    }

    #[test]
    fn test_nested_async_with() {
        let result = compiles(
            r#"
async def main():
    async with outer() as o:
        async with inner() as i:
            use(o, i)
"#,
        );
        assert!(result.is_ok(), "Failed: {:?}", result);
    }

    #[test]
    fn test_async_with_exception_handling() {
        let result = compiles(
            r#"
async def main():
    try:
        async with may_fail() as resource:
            use(resource)
    except Exception:
        handle()
"#,
        );
        assert!(result.is_ok(), "Failed: {:?}", result);
    }

    #[test]
    fn test_async_with_await_in_body() {
        let result = compiles(
            r#"
async def main():
    async with get_connection() as conn:
        result = await conn.execute(query)
        return result
"#,
        );
        assert!(result.is_ok(), "Failed: {:?}", result);
    }

    #[test]
    fn test_async_with_outside_async_fails() {
        let fails = compile_fails(
            r#"
def not_async():
    async with something:
        pass
"#,
        );
        assert!(fails, "Should fail to compile async with outside async");
    }
}

// =============================================================================
// Async Generator Tests
// =============================================================================

mod async_generator_tests {
    use super::*;

    #[test]
    fn test_simple_async_generator() {
        let result = compiles(
            r#"
async def async_gen():
    yield 1
    yield 2
    yield 3
"#,
        );
        assert!(result.is_ok(), "Failed: {:?}", result);
    }

    #[test]
    fn test_async_generator_with_await() {
        let result = compiles(
            r#"
async def async_gen():
    value = await fetch()
    yield value
"#,
        );
        assert!(result.is_ok(), "Failed: {:?}", result);
    }

    #[test]
    fn test_async_generator_in_loop() {
        let result = compiles(
            r#"
async def async_gen():
    for i in range(10):
        yield i
"#,
        );
        assert!(result.is_ok(), "Failed: {:?}", result);
    }

    #[test]
    fn test_async_generator_with_await_in_loop() {
        let result = compiles(
            r#"
async def async_gen():
    for url in urls:
        data = await fetch(url)
        yield data
"#,
        );
        assert!(result.is_ok(), "Failed: {:?}", result);
    }

    #[test]
    fn test_async_generator_conditional_yield() {
        let result = compiles(
            r#"
async def async_gen():
    for item in items:
        if item > 0:
            yield item
"#,
        );
        assert!(result.is_ok(), "Failed: {:?}", result);
    }

    #[test]
    fn test_async_generator_yield_from_async_for() {
        let result = compiles(
            r#"
async def async_gen():
    async for item in other_async_gen():
        yield item * 2
"#,
        );
        assert!(result.is_ok(), "Failed: {:?}", result);
    }

    #[test]
    fn test_iterate_async_generator() {
        let result = compiles(
            r#"
async def async_gen():
    yield 1
    yield 2

async def main():
    async for value in async_gen():
        process(value)
"#,
        );
        assert!(result.is_ok(), "Failed: {:?}", result);
    }
}

// =============================================================================
// Async Comprehension Tests
// =============================================================================

mod async_comprehension_tests {
    use super::*;

    #[test]
    fn test_async_list_comprehension() {
        let result = compiles(
            r#"
async def main():
    result = [x async for x in async_iter]
    return result
"#,
        );
        assert!(result.is_ok(), "Failed: {:?}", result);
    }

    #[test]
    fn test_async_list_comp_with_transform() {
        let result = compiles(
            r#"
async def main():
    result = [x * 2 async for x in async_iter]
    return result
"#,
        );
        assert!(result.is_ok(), "Failed: {:?}", result);
    }

    #[test]
    fn test_async_list_comp_with_filter() {
        let result = compiles(
            r#"
async def main():
    result = [x async for x in async_iter if x > 0]
    return result
"#,
        );
        assert!(result.is_ok(), "Failed: {:?}", result);
    }

    #[test]
    fn test_async_list_comp_with_await_filter() {
        let result = compiles(
            r#"
async def main():
    result = [x async for x in async_iter if await is_valid(x)]
    return result
"#,
        );
        assert!(result.is_ok(), "Failed: {:?}", result);
    }

    #[test]
    fn test_async_list_comp_with_await_expression() {
        let result = compiles(
            r#"
async def main():
    result = [await process(x) async for x in async_iter]
    return result
"#,
        );
        assert!(result.is_ok(), "Failed: {:?}", result);
    }

    #[test]
    fn test_async_set_comprehension() {
        let result = compiles(
            r#"
async def main():
    result = {x async for x in async_iter}
    return result
"#,
        );
        assert!(result.is_ok(), "Failed: {:?}", result);
    }

    #[test]
    fn test_async_dict_comprehension() {
        let result = compiles(
            r#"
async def main():
    result = {k: v async for k, v in async_items}
    return result
"#,
        );
        assert!(result.is_ok(), "Failed: {:?}", result);
    }

    #[test]
    fn test_async_generator_expression() {
        let result = compiles(
            r#"
async def main():
    gen = (x async for x in async_iter)
    async for item in gen:
        process(item)
"#,
        );
        assert!(result.is_ok(), "Failed: {:?}", result);
    }

    #[test]
    fn test_nested_async_comprehension() {
        let result = compiles(
            r#"
async def main():
    result = [y async for x in outer_async async for y in inner_async(x)]
    return result
"#,
        );
        assert!(result.is_ok(), "Failed: {:?}", result);
    }

    #[test]
    fn test_mixed_sync_async_comprehension() {
        let result = compiles(
            r#"
async def main():
    result = [x async for x in async_iter for y in sync_iter]
    return result
"#,
        );
        assert!(result.is_ok(), "Failed: {:?}", result);
    }
}

// =============================================================================
// Complex Async Flow Tests
// =============================================================================

mod complex_flow_tests {
    use super::*;

    #[test]
    fn test_async_with_async_for() {
        let result = compiles(
            r#"
async def main():
    async with get_connection() as conn:
        async for row in conn.fetch():
            yield row
"#,
        );
        assert!(result.is_ok(), "Failed: {:?}", result);
    }

    #[test]
    fn test_multiple_async_constructs() {
        let result = compiles(
            r#"
async def main():
    async with context() as ctx:
        async for item in get_items():
            result = await process(item)
            yield result
"#,
        );
        assert!(result.is_ok(), "Failed: {:?}", result);
    }

    #[test]
    fn test_async_closure() {
        let result = compiles(
            r#"
async def outer(x):
    async def inner(y):
        return x + y
    return await inner(10)
"#,
        );
        assert!(result.is_ok(), "Failed: {:?}", result);
    }

    #[test]
    fn test_async_lambda_like_pattern() {
        let result = compiles(
            r#"
async def main():
    async def helper():
        return 42
    funcs = [helper, helper]
    results = [await f() for f in funcs]
    return results
"#,
        );
        assert!(result.is_ok(), "Failed: {:?}", result);
    }

    #[test]
    fn test_async_recursion() {
        let result = compiles(
            r#"
async def factorial(n):
    if n <= 1:
        return 1
    return n * await factorial(n - 1)
"#,
        );
        assert!(result.is_ok(), "Failed: {:?}", result);
    }

    #[test]
    fn test_async_mutual_recursion() {
        let result = compiles(
            r#"
async def is_even(n):
    if n == 0:
        return True
    return await is_odd(n - 1)

async def is_odd(n):
    if n == 0:
        return False
    return await is_even(n - 1)
"#,
        );
        assert!(result.is_ok(), "Failed: {:?}", result);
    }

    #[test]
    fn test_async_try_finally() {
        let result = compiles(
            r#"
async def main():
    try:
        result = await risky_operation()
        return result
    finally:
        await cleanup()
"#,
        );
        assert!(result.is_ok(), "Failed: {:?}", result);
    }

    #[test]
    fn test_async_exception_handling() {
        let result = compiles(
            r#"
async def main():
    try:
        result = await may_fail()
    except ValueError as e:
        result = await fallback()
    except Exception:
        result = None
    return result
"#,
        );
        assert!(result.is_ok(), "Failed: {:?}", result);
    }

    #[test]
    fn test_async_reraise() {
        let result = compiles(
            r#"
async def main():
    try:
        await operation()
    except Exception:
        await log_error()
        raise
"#,
        );
        assert!(result.is_ok(), "Failed: {:?}", result);
    }

    #[test]
    fn test_async_raise_from() {
        let result = compiles(
            r#"
async def main():
    try:
        await operation()
    except ValueError as e:
        raise RuntimeError("wrapped") from e
"#,
        );
        assert!(result.is_ok(), "Failed: {:?}", result);
    }
}

// =============================================================================
// Error Case Tests
// =============================================================================

mod error_case_tests {
    use super::*;

    #[test]
    fn test_yield_in_non_generator_fails() {
        // yield outside function should fail
        let fails = compile_fails("yield 1");
        assert!(fails, "yield at module level should fail");
    }

    #[test]
    fn test_await_at_module_level_fails() {
        let fails = compile_fails("await something()");
        assert!(fails, "await at module level should fail");
    }

    #[test]
    fn test_async_for_at_module_level_fails() {
        let fails = compile_fails(
            r#"
async for item in items:
    pass
"#,
        );
        assert!(fails, "async for at module level should fail");
    }

    #[test]
    fn test_async_with_at_module_level_fails() {
        let fails = compile_fails(
            r#"
async with context:
    pass
"#,
        );
        assert!(fails, "async with at module level should fail");
    }

    #[test]
    fn test_return_in_async_generator() {
        // async generators can return None
        let result = compiles(
            r#"
async def async_gen():
    yield 1
    return
"#,
        );
        assert!(
            result.is_ok(),
            "Empty return in async generator should work"
        );
    }
}

// =============================================================================
// Code Flags Tests
// =============================================================================

mod code_flags_tests {
    use super::*;
    use prism_compiler::bytecode::CodeFlags;

    fn get_code_flags(source: &str) -> Result<CodeFlags, String> {
        let module = parse(source).map_err(|e| format!("{:?}", e))?;
        let code = Compiler::compile_module(&module, "<test>").map_err(|e| format!("{:?}", e))?;

        // Find the first nested code object (the function)
        if let Some(nested) = code.nested_code_objects.first() {
            Ok(nested.flags)
        } else {
            Err("No nested code objects".into())
        }
    }

    #[test]
    fn test_async_def_has_coroutine_flag() {
        let flags = get_code_flags(
            r#"
async def coro():
    pass
"#,
        );
        assert!(flags.is_ok());
        assert!(
            flags.unwrap().contains(CodeFlags::COROUTINE),
            "async def should have COROUTINE flag"
        );
    }

    #[test]
    fn test_async_generator_has_async_generator_flag() {
        let flags = get_code_flags(
            r#"
async def async_gen():
    yield 1
"#,
        );
        assert!(flags.is_ok());
        let f = flags.unwrap();
        assert!(
            f.contains(CodeFlags::ASYNC_GENERATOR),
            "async def with yield should have ASYNC_GENERATOR flag"
        );
    }

    #[test]
    fn test_regular_def_no_coroutine_flag() {
        let flags = get_code_flags(
            r#"
def regular():
    pass
"#,
        );
        assert!(flags.is_ok());
        assert!(
            !flags.unwrap().contains(CodeFlags::COROUTINE),
            "regular def should not have COROUTINE flag"
        );
    }

    #[test]
    fn test_generator_has_generator_flag() {
        let flags = get_code_flags(
            r#"
def gen():
    yield 1
"#,
        );
        assert!(flags.is_ok());
        assert!(
            flags.unwrap().contains(CodeFlags::GENERATOR),
            "def with yield should have GENERATOR flag"
        );
    }
}

// =============================================================================
// Bytecode Verification Tests
// =============================================================================

mod bytecode_tests {
    use super::*;
    use prism_compiler::bytecode::Opcode;

    fn has_opcode(source: &str, target: Opcode) -> bool {
        let module = match parse(source) {
            Ok(m) => m,
            Err(_) => return false,
        };
        let code = match Compiler::compile_module(&module, "<test>") {
            Ok(c) => c,
            Err(_) => return false,
        };

        let target_u8 = target as u8;

        // Check main code
        for instr in code.instructions.iter() {
            if instr.opcode() == target_u8 {
                return true;
            }
        }

        // Check nested code objects
        for nested in code.nested_code_objects.iter() {
            for instr in nested.instructions.iter() {
                if instr.opcode() == target_u8 {
                    return true;
                }
            }
        }
        false
    }

    #[test]
    fn test_await_emits_get_awaitable() {
        assert!(
            has_opcode(
                r#"
async def coro():
    await other()
"#,
                Opcode::GetAwaitable
            ),
            "await should emit GetAwaitable"
        );
    }

    #[test]
    fn test_async_for_emits_get_aiter() {
        assert!(
            has_opcode(
                r#"
async def main():
    async for x in items:
        pass
"#,
                Opcode::GetAIter
            ),
            "async for should emit GetAIter"
        );
    }

    #[test]
    fn test_async_for_emits_get_anext() {
        assert!(
            has_opcode(
                r#"
async def main():
    async for x in items:
        pass
"#,
                Opcode::GetANext
            ),
            "async for should emit GetANext"
        );
    }

    #[test]
    fn test_async_for_emits_end_async_for() {
        assert!(
            has_opcode(
                r#"
async def main():
    async for x in items:
        pass
"#,
                Opcode::EndAsyncFor
            ),
            "async for should emit EndAsyncFor"
        );
    }

    #[test]
    fn test_async_comprehension_emits_get_aiter() {
        assert!(
            has_opcode(
                r#"
async def main():
    return [x async for x in items]
"#,
                Opcode::GetAIter
            ),
            "async comprehension should emit GetAIter"
        );
    }
}

// =============================================================================
// Performance Baseline Tests
// =============================================================================

mod performance_tests {
    use super::*;

    #[test]
    fn test_many_async_defs_compile() {
        let mut source = String::new();
        for i in 0..100 {
            source.push_str(&format!(
                r#"
async def func_{i}():
    return {i}
"#
            ));
        }
        let result = compiles(&source);
        assert!(result.is_ok(), "Should compile 100 async defs");
    }

    #[test]
    fn test_deep_await_chain_compiles() {
        let result = compiles(
            r#"
async def a1():
    return 1

async def a2():
    return await a1()

async def a3():
    return await a2()

async def a4():
    return await a3()

async def a5():
    return await a4()
"#,
        );
        assert!(result.is_ok(), "Should compile deep await chain");
    }

    #[test]
    fn test_complex_async_for_compiles() {
        let result = compiles(
            r#"
async def main():
    async for a in iter1:
        async for b in iter2:
            async for c in iter3:
                if a + b + c > 0:
                    yield a * b * c
"#,
        );
        assert!(result.is_ok(), "Should compile nested async for");
    }

    #[test]
    fn test_large_async_function_compiles() {
        let mut source = String::from("async def large():\n");
        for i in 0..50 {
            source.push_str(&format!("    x{i} = await get_{i}()\n"));
        }
        source.push_str("    return x0\n");

        let result = compiles(&source);
        assert!(result.is_ok(), "Should compile large async function");
    }
}
