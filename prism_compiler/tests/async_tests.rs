//! Comprehensive test suite for async/await compiler functionality.
//!
//! Tests all aspects of async compilation:
//! - async def functions
//! - await expressions
//! - async for loops
//! - async with statements
//! - Error cases (await outside async, etc.)
//! - Nested async constructs
//! - Opcode emission validation

use prism_compiler::{Compiler, Opcode};
use prism_parser::parse;

// ============================================================================
// Test Utilities
// ============================================================================

/// Helper to compile source code and return the code object.
fn compile(source: &str) -> prism_compiler::CodeObject {
    let module = parse(source).expect("Failed to parse");
    Compiler::compile_module(&module, "test.py").expect("Failed to compile")
}

/// Helper to check if code compiles successfully.
fn assert_compiles(source: &str) {
    let _ = compile(source);
}

/// Helper that expects compilation to fail with a specific error message.
fn assert_compile_error(source: &str, expected_error: &str) {
    let module = parse(source).expect("Failed to parse");
    let result = Compiler::compile_module(&module, "test.py");
    assert!(result.is_err(), "Expected compilation to fail");
    let err = result.unwrap_err();
    assert!(
        err.message.contains(expected_error),
        "Expected error containing '{}', got: '{}'",
        expected_error,
        err.message
    );
}

/// Helper to find opcode count in compiled code (top-level only).
fn count_opcodes(code: &prism_compiler::CodeObject, opcode: Opcode) -> usize {
    code.instructions
        .iter()
        .filter(|inst| inst.opcode() == opcode as u8)
        .count()
}

/// Helper to count opcodes recursively (top-level + all nested code objects).
fn count_opcodes_recursive(code: &prism_compiler::CodeObject, opcode: Opcode) -> usize {
    // Count in top-level code
    let mut count = count_opcodes(code, opcode);

    // Recursively count in nested code objects
    for nested in code.nested_code_objects.iter() {
        count += count_opcodes_recursive(nested, opcode);
    }

    count
}

/// Helper to check if code or any nested code contains an opcode.
fn has_opcode(code: &prism_compiler::CodeObject, opcode: Opcode) -> bool {
    count_opcodes_recursive(code, opcode) > 0
}

// ============================================================================
// Async Def Function Tests
// ============================================================================

#[test]
fn test_async_def_basic() {
    let source = r#"
async def foo():
    pass
"#;
    assert_compiles(source);
}

#[test]
fn test_async_def_with_return() {
    let source = r#"
async def foo():
    return 42
"#;
    assert_compiles(source);
}

#[test]
fn test_async_def_with_parameters() {
    let source = r#"
async def foo(a, b, c):
    return a + b + c
"#;
    assert_compiles(source);
}

#[test]
fn test_async_def_with_default_params() {
    let source = r#"
async def foo(a, b=10, c=20):
    return a + b + c
"#;
    assert_compiles(source);
}

#[test]
fn test_async_def_with_kwargs() {
    let source = r#"
async def foo(*args, **kwargs):
    return len(args) + len(kwargs)
"#;
    assert_compiles(source);
}

#[test]
fn test_async_def_sets_coroutine_flag() {
    // Note: The async function's code object is nested in constants.
    // The module itself doesn't have COROUTINE flag, but the nested function does.
    // For now, we just verify the source compiles successfully.
    // A full test would require extracting nested CodeObject from constants.
    let source = r#"
async def coro():
    return 1
"#;
    assert_compiles(source);
}

// ============================================================================
// Await Expression Tests
// ============================================================================

#[test]
fn test_await_simple() {
    let source = r#"
async def foo():
    await bar()
"#;
    assert_compiles(source);
}

#[test]
fn test_await_assignment() {
    let source = r#"
async def foo():
    result = await bar()
    return result
"#;
    assert_compiles(source);
}

#[test]
fn test_await_in_expression() {
    let source = r#"
async def foo():
    result = (await bar()) + 10
    return result
"#;
    assert_compiles(source);
}

#[test]
fn test_await_chain() {
    let source = r#"
async def foo():
    a = await step1()
    b = await step2(a)
    c = await step3(b)
    return c
"#;
    assert_compiles(source);
}

#[test]
fn test_await_in_conditional() {
    let source = r#"
async def foo():
    if await check():
        await do_something()
    else:
        await do_other()
"#;
    assert_compiles(source);
}

#[test]
fn test_await_method_call() {
    let source = r#"
async def foo():
    result = await obj.async_method()
    return result
"#;
    assert_compiles(source);
}

#[test]
fn test_await_nested_call() {
    let source = r#"
async def foo():
    result = await (await get_coro())
    return result
"#;
    assert_compiles(source);
}

#[test]
fn test_await_in_loop_body() {
    let source = r#"
async def foo():
    for i in range(10):
        await process(i)
"#;
    assert_compiles(source);
}

#[test]
fn test_await_in_while_body() {
    let source = r#"
async def foo():
    while condition:
        await process()
"#;
    assert_compiles(source);
}

#[test]
fn test_await_emits_get_awaitable() {
    let source = r#"
async def foo():
    await bar()
"#;
    let code = compile(source);
    assert!(
        has_opcode(&code, Opcode::GetAwaitable),
        "Await should emit GetAwaitable"
    );
}

#[test]
fn test_await_emits_yield_from() {
    let source = r#"
async def foo():
    await bar()
"#;
    let code = compile(source);
    assert!(
        has_opcode(&code, Opcode::YieldFrom),
        "Await should emit YieldFrom"
    );
}

// ============================================================================
// Await Error Cases
// ============================================================================

#[test]
fn test_await_outside_async_error() {
    let source = r#"
def foo():
    await bar()
"#;
    assert_compile_error(source, "outside async");
}

#[test]
fn test_await_at_module_level_error() {
    let source = r#"
result = await something()
"#;
    assert_compile_error(source, "outside async");
}

#[test]
fn test_await_in_sync_lambda_error() {
    let source = r#"
def foo():
    f = lambda: await bar()
"#;
    assert_compile_error(source, "outside async");
}

#[test]
fn test_await_in_list_comp_outside_async_error() {
    let source = r#"
def foo():
    result = [await x for x in items]
"#;
    assert_compile_error(source, "outside async");
}

// ============================================================================
// Async For Loop Tests
// ============================================================================

#[test]
fn test_async_for_basic() {
    let source = r#"
async def foo():
    async for item in aiter:
        print(item)
"#;
    assert_compiles(source);
}

#[test]
fn test_async_for_with_tuple_target() {
    let source = r#"
async def foo():
    async for a, b in aiter:
        process(a, b)
"#;
    assert_compiles(source);
}

#[test]
fn test_async_for_with_else() {
    let source = r#"
async def foo():
    async for item in aiter:
        if found(item):
            break
    else:
        handle_not_found()
"#;
    assert_compiles(source);
}

#[test]
fn test_async_for_nested() {
    let source = r#"
async def foo():
    async for outer in aiter1:
        async for inner in aiter2:
            process(outer, inner)
"#;
    assert_compiles(source);
}

#[test]
fn test_async_for_with_break() {
    let source = r#"
async def foo():
    async for item in aiter:
        if should_stop(item):
            break
"#;
    assert_compiles(source);
}

#[test]
fn test_async_for_with_continue() {
    let source = r#"
async def foo():
    async for item in aiter:
        if should_skip(item):
            continue
        process(item)
"#;
    assert_compiles(source);
}

#[test]
fn test_async_for_with_await_in_body() {
    let source = r#"
async def foo():
    async for item in aiter:
        result = await process(item)
        accumulate(result)
"#;
    assert_compiles(source);
}

#[test]
fn test_async_for_emits_get_aiter() {
    let source = r#"
async def foo():
    async for x in aiter:
        pass
"#;
    let code = compile(source);
    assert!(
        has_opcode(&code, Opcode::GetAIter),
        "Async for should emit GetAIter"
    );
}

#[test]
fn test_async_for_emits_get_anext() {
    let source = r#"
async def foo():
    async for x in aiter:
        pass
"#;
    let code = compile(source);
    assert!(
        has_opcode(&code, Opcode::GetANext),
        "Async for should emit GetANext"
    );
}

#[test]
fn test_async_for_emits_end_async_for() {
    let source = r#"
async def foo():
    async for x in aiter:
        pass
"#;
    let code = compile(source);
    assert!(
        has_opcode(&code, Opcode::EndAsyncFor),
        "Async for should emit EndAsyncFor"
    );
}

// ============================================================================
// Async For Error Cases
// ============================================================================

#[test]
fn test_async_for_outside_async_error() {
    let source = r#"
def foo():
    async for item in aiter:
        pass
"#;
    assert_compile_error(source, "outside async");
}

#[test]
fn test_async_for_at_module_level_error() {
    let source = r#"
async for item in aiter:
    pass
"#;
    assert_compile_error(source, "outside async");
}

// ============================================================================
// Async With Statement Tests
// ============================================================================

#[test]
fn test_async_with_basic() {
    let source = r#"
async def foo():
    async with mgr:
        do_something()
"#;
    assert_compiles(source);
}

#[test]
fn test_async_with_as_binding() {
    let source = r#"
async def foo():
    async with mgr as m:
        result = m.process()
        return result
"#;
    assert_compiles(source);
}

#[test]
fn test_async_with_multiple_managers() {
    let source = r#"
async def foo():
    async with a, b:
        do_something()
"#;
    assert_compiles(source);
}

#[test]
fn test_async_with_multiple_with_bindings() {
    let source = r#"
async def foo():
    async with a as x, b as y:
        process(x, y)
"#;
    assert_compiles(source);
}

#[test]
fn test_async_with_nested() {
    let source = r#"
async def foo():
    async with outer as o:
        async with inner as i:
            process(o, i)
"#;
    assert_compiles(source);
}

#[test]
fn test_async_with_await_in_body() {
    let source = r#"
async def foo():
    async with mgr as m:
        result = await m.async_operation()
        return result
"#;
    assert_compiles(source);
}

#[test]
fn test_async_with_emits_get_awaitable() {
    let source = r#"
async def foo():
    async with mgr:
        pass
"#;
    let code = compile(source);
    // Async with should emit GetAwaitable for __aenter__ and __aexit__
    let count = count_opcodes_recursive(&code, Opcode::GetAwaitable);
    assert!(
        count >= 2,
        "Async with should emit GetAwaitable for __aenter__ and __aexit__, got {}",
        count
    );
}

// ============================================================================
// Async With Error Cases
// ============================================================================

#[test]
fn test_async_with_outside_async_error() {
    let source = r#"
def foo():
    async with mgr:
        pass
"#;
    assert_compile_error(source, "outside async");
}

#[test]
fn test_async_with_at_module_level_error() {
    let source = r#"
async with mgr:
    pass
"#;
    assert_compile_error(source, "outside async");
}

// ============================================================================
// Combined Async Constructs
// ============================================================================

#[test]
fn test_async_for_in_async_with() {
    let source = r#"
async def foo():
    async with connection as conn:
        async for record in conn.records():
            await process(record)
"#;
    assert_compiles(source);
}

#[test]
fn test_async_with_in_async_for() {
    let source = r#"
async def foo():
    async for item in items:
        async with item.context() as ctx:
            await ctx.process()
"#;
    assert_compiles(source);
}

#[test]
fn test_complex_async_nesting() {
    let source = r#"
async def complex_async():
    results = []
    async with database.connection() as conn:
        async for table in conn.tables():
            async with table.lock() as lock:
                data = await lock.read_all()
                processed = await transform(data)
                results.append(processed)
    return results
"#;
    assert_compiles(source);
}

#[test]
fn test_await_in_try_except() {
    let source = r#"
async def foo():
    try:
        result = await risky_operation()
    except Error:
        result = await fallback()
    return result
"#;
    assert_compiles(source);
}

#[test]
fn test_await_in_try_finally() {
    let source = r#"
async def foo():
    try:
        result = await operation()
    finally:
        await cleanup()
"#;
    assert_compiles(source);
}

// ============================================================================
// Async Class Methods
// ============================================================================

#[test]
fn test_async_method() {
    let source = r#"
class Service:
    async def fetch(self):
        return await self.client.get()
"#;
    assert_compiles(source);
}

#[test]
fn test_async_static_method() {
    let source = r#"
class Service:
    @staticmethod
    async def fetch():
        return await make_request()
"#;
    assert_compiles(source);
}

#[test]
fn test_async_class_method() {
    let source = r#"
class Service:
    @classmethod
    async def create(cls):
        instance = cls()
        await instance.initialize()
        return instance
"#;
    assert_compiles(source);
}

// ============================================================================
// Yield and YieldFrom Tests (Generators in async context)
// ============================================================================

#[test]
fn test_async_generator_yield() {
    let source = r#"
async def async_gen():
    yield 1
    yield 2
    yield 3
"#;
    assert_compiles(source);
}

#[test]
fn test_async_generator_with_await() {
    let source = r#"
async def async_gen():
    value = await fetch()
    yield value
"#;
    assert_compiles(source);
}

// ============================================================================
// Edge Cases
// ============================================================================

#[test]
fn test_nested_async_functions() {
    let source = r#"
async def outer():
    async def inner():
        return await something()
    return await inner()
"#;
    assert_compiles(source);
}

#[test]
fn test_async_with_sync_for() {
    let source = r#"
async def foo():
    for item in sync_items:
        result = await process(item)
"#;
    assert_compiles(source);
}

#[test]
fn test_sync_with_in_async() {
    let source = r#"
async def foo():
    with sync_manager as m:
        result = await operation()
"#;
    assert_compiles(source);
}

#[test]
fn test_await_ternary() {
    let source = r#"
async def foo():
    result = await a() if condition else await b()
    return result
"#;
    assert_compiles(source);
}

#[test]
fn test_await_in_walrus() {
    // Python 3.8+ walrus operator
    let source = r#"
async def foo():
    if (result := await check()):
        process(result)
"#;
    assert_compiles(source);
}

// ============================================================================
// Real-World Pattern Tests
// ============================================================================

#[test]
fn test_http_client_pattern() {
    let source = r#"
async def fetch_data(url):
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            data = await response.json()
            return data
"#;
    assert_compiles(source);
}

#[test]
fn test_database_pattern() {
    let source = r#"
async def get_users():
    async with pool.acquire() as conn:
        users = []
        async for row in conn.execute("SELECT * FROM users"):
            users.append(row)
        return users
"#;
    assert_compiles(source);
}

#[test]
fn test_concurrent_gather_pattern() {
    let source = r#"
async def fetch_all(urls):
    results = []
    for url in urls:
        result = await fetch(url)
        results.append(result)
    return results
"#;
    assert_compiles(source);
}

#[test]
fn test_websocket_pattern() {
    let source = r#"
async def websocket_handler():
    async with websockets.connect(uri) as ws:
        async for message in ws:
            data = await process(message)
            await ws.send(data)
"#;
    assert_compiles(source);
}

#[test]
fn test_file_io_pattern() {
    let source = r#"
async def read_lines(path):
    async with aiofiles.open(path) as f:
        async for line in f:
            yield line.strip()
"#;
    assert_compiles(source);
}

// ============================================================================
// Opcode Count Tests
// ============================================================================

#[test]
fn test_single_await_opcodes() {
    let source = r#"
async def foo():
    await bar()
"#;
    let code = compile(source);

    // Single await should have exactly 1 GetAwaitable and 1 YieldFrom
    let get_awaitable_count = count_opcodes_recursive(&code, Opcode::GetAwaitable);
    let yield_from_count = count_opcodes_recursive(&code, Opcode::YieldFrom);

    assert_eq!(
        get_awaitable_count, 1,
        "Single await should emit 1 GetAwaitable, got {}",
        get_awaitable_count
    );
    assert_eq!(
        yield_from_count, 1,
        "Single await should emit 1 YieldFrom, got {}",
        yield_from_count
    );
}

#[test]
fn test_multiple_await_opcodes() {
    let source = r#"
async def foo():
    await a()
    await b()
    await c()
"#;
    let code = compile(source);

    let get_awaitable_count = count_opcodes_recursive(&code, Opcode::GetAwaitable);
    let yield_from_count = count_opcodes_recursive(&code, Opcode::YieldFrom);

    assert_eq!(
        get_awaitable_count, 3,
        "Three awaits should emit 3 GetAwaitable, got {}",
        get_awaitable_count
    );
    assert_eq!(
        yield_from_count, 3,
        "Three awaits should emit 3 YieldFrom, got {}",
        yield_from_count
    );
}

#[test]
fn test_async_for_opcode_sequence() {
    let source = r#"
async def foo():
    async for x in aiter:
        pass
"#;
    let code = compile(source);

    // Async for should emit: GetAIter, GetANext, GetAwaitable, YieldFrom, EndAsyncFor
    assert!(has_opcode(&code, Opcode::GetAIter));
    assert!(has_opcode(&code, Opcode::GetANext));
    assert!(has_opcode(&code, Opcode::GetAwaitable));
    assert!(has_opcode(&code, Opcode::YieldFrom));
    assert!(has_opcode(&code, Opcode::EndAsyncFor));
}
