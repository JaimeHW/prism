//! Comprehensive test suite for with statements (context managers).
//!
//! Tests all aspects of context manager compilation:
//! - Basic with statement
//! - As binding
//! - Multiple context managers
//! - Nested with statements
//! - Exception propagation
//! - __exit__ return value handling

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

/// Helper to assert that code compiles successfully.
fn assert_compiles(source: &str) {
    let _ = compile(source);
}

/// Helper to find opcode count in compiled code.
fn count_opcodes(code: &prism_compiler::CodeObject, opcode: Opcode) -> usize {
    code.instructions
        .iter()
        .filter(|inst| inst.opcode() == opcode as u8)
        .count()
}

/// Helper to check if code contains an opcode.
fn has_opcode(code: &prism_compiler::CodeObject, opcode: Opcode) -> bool {
    count_opcodes(code, opcode) > 0
}

// ============================================================================
// Basic With Statement Tests
// ============================================================================

#[test]
fn test_simple_with() {
    let source = r#"
with mgr:
    x = 1
"#;
    assert_compiles(source);
}

#[test]
fn test_with_as_binding() {
    let source = r#"
with mgr as value:
    x = value
"#;
    assert_compiles(source);
}

#[test]
fn test_with_function_call() {
    let source = r#"
with open("file.txt") as f:
    data = f.read()
"#;
    assert_compiles(source);
}

#[test]
fn test_with_method_call() {
    let source = r#"
with obj.get_context() as ctx:
    ctx.do_something()
"#;
    assert_compiles(source);
}

#[test]
fn test_with_expression() {
    let source = r#"
with (a or b):
    pass
"#;
    assert_compiles(source);
}

#[test]
fn test_with_complex_expression() {
    let source = r#"
with factory.create_manager(option=True):
    process()
"#;
    assert_compiles(source);
}

// ============================================================================
// Multiple Context Managers Tests
// ============================================================================

#[test]
fn test_two_context_managers() {
    let source = r#"
with a, b:
    pass
"#;
    assert_compiles(source);
}

#[test]
fn test_two_context_managers_with_bindings() {
    let source = r#"
with a as x, b as y:
    z = x + y
"#;
    assert_compiles(source);
}

#[test]
fn test_three_context_managers() {
    let source = r#"
with a, b, c:
    do_something()
"#;
    assert_compiles(source);
}

#[test]
fn test_mixed_binding_context_managers() {
    let source = r#"
with a as x, b, c as z:
    use(x, z)
"#;
    assert_compiles(source);
}

#[test]
fn test_many_context_managers() {
    let source = r#"
with a, b, c, d, e:
    pass
"#;
    assert_compiles(source);
}

// ============================================================================
// Nested With Statement Tests
// ============================================================================

#[test]
fn test_nested_with_explicit() {
    let source = r#"
with outer:
    with inner:
        pass
"#;
    assert_compiles(source);
}

#[test]
fn test_deeply_nested_with() {
    let source = r#"
with a:
    with b:
        with c:
            with d:
                value = process()
"#;
    assert_compiles(source);
}

#[test]
fn test_nested_with_mixed_bindings() {
    let source = r#"
with outer as o:
    with inner as i:
        result = o.combine(i)
"#;
    assert_compiles(source);
}

// ============================================================================
// With Statement Body Tests
// ============================================================================

#[test]
fn test_with_empty_body() {
    let source = r#"
with mgr:
    pass
"#;
    assert_compiles(source);
}

#[test]
fn test_with_multiple_statements() {
    let source = r#"
with mgr as m:
    x = 1
    y = 2
    z = x + y
    result = m.process(z)
"#;
    assert_compiles(source);
}

#[test]
fn test_with_return_in_body() {
    let source = r#"
def func():
    with mgr as f:
        return f.read()
"#;
    assert_compiles(source);
}

#[test]
fn test_with_break_in_loop() {
    let source = r#"
for i in items:
    with mgr:
        if condition:
            break
"#;
    assert_compiles(source);
}

#[test]
fn test_with_continue_in_loop() {
    let source = r#"
while True:
    with mgr:
        if skip:
            continue
        process()
"#;
    assert_compiles(source);
}

// ============================================================================
// With Statement and Control Flow Tests
// ============================================================================

#[test]
fn test_with_if_statement() {
    let source = r#"
with mgr as m:
    if m.is_valid():
        do_something()
    else:
        do_other()
"#;
    assert_compiles(source);
}

#[test]
fn test_with_for_loop() {
    let source = r#"
with mgr as items:
    for item in items:
        process(item)
"#;
    assert_compiles(source);
}

#[test]
fn test_with_while_loop() {
    let source = r#"
with mgr as m:
    while m.has_more():
        item = m.next()
        process(item)
"#;
    assert_compiles(source);
}

// ============================================================================
// With Statement and Exception Handling Tests
// ============================================================================

#[test]
fn test_with_try_inside() {
    let source = r#"
with mgr:
    try:
        risky_operation()
    except Error as e:
        handle(e)
"#;
    assert_compiles(source);
}

#[test]
fn test_try_with_inside() {
    let source = r#"
try:
    with mgr:
        risky_operation()
except Error:
    handle_error()
"#;
    assert_compiles(source);
}

#[test]
fn test_with_try_finally_inside() {
    let source = r#"
with mgr:
    try:
        work()
    finally:
        cleanup()
"#;
    assert_compiles(source);
}

// ============================================================================
// With Statement Exception Table Tests
// ============================================================================

#[test]
fn test_with_generates_exception_entry() {
    let source = r#"
with mgr:
    x = 1
"#;
    let code = compile(source);

    // Should have at least one exception table entry for cleanup
    assert!(
        !code.exception_table.is_empty(),
        "With statement should generate exception table entries"
    );
}

#[test]
fn test_multiple_with_generates_multiple_entries() {
    let source = r#"
with a, b, c:
    x = 1
"#;
    let code = compile(source);

    // Should have exception entries for each context manager
    assert!(
        code.exception_table.len() >= 3,
        "Multiple context managers should generate multiple exception entries"
    );
}

// ============================================================================
// With Statement Method Call Tests
// ============================================================================

#[test]
fn test_with_emits_load_method() {
    let source = r#"
with mgr:
    pass
"#;
    let code = compile(source);

    // Should emit LoadMethod for __enter__ and __exit__
    let load_method_count = count_opcodes(&code, Opcode::LoadMethod);
    assert!(
        load_method_count >= 2,
        "With should emit LoadMethod for __enter__ and __exit__, got {}",
        load_method_count
    );
}

#[test]
fn test_with_emits_call() {
    let source = r#"
with mgr:
    pass
"#;
    let code = compile(source);

    // Should emit Call for __enter__() and __exit__(...)
    assert!(
        has_opcode(&code, Opcode::Call),
        "With should emit Call opcodes for method invocations"
    );
}

// ============================================================================
// With Statement Edge Cases
// ============================================================================

#[test]
fn test_with_attribute_expression() {
    let source = r#"
with module.submodule.Manager() as m:
    m.work()
"#;
    assert_compiles(source);
}

#[test]
fn test_with_index_expression() {
    let source = r#"
with managers[0] as m:
    m.work()
"#;
    assert_compiles(source);
}

#[test]
fn test_with_call_chain() {
    let source = r#"
with get_factory().create_manager().configure(opt=1) as m:
    m.execute()
"#;
    assert_compiles(source);
}

#[test]
fn test_with_ternary_expression() {
    let source = r#"
with (a if condition else b) as m:
    m.work()
"#;
    assert_compiles(source);
}

// ============================================================================
// Async With Statement Tests
// ============================================================================

#[test]
fn test_async_with_basic() {
    let source = r#"
async def func():
    async with mgr:
        await operation()
"#;
    assert_compiles(source);
}

#[test]
fn test_async_with_as_binding() {
    let source = r#"
async def func():
    async with mgr as m:
        result = await m.read()
        return result
"#;
    assert_compiles(source);
}

#[test]
fn test_async_with_multiple_managers() {
    let source = r#"
async def func():
    async with a as x, b as y:
        data = await x.fetch()
        await y.process(data)
"#;
    assert_compiles(source);
}

// ============================================================================
// Real-World Pattern Tests
// ============================================================================

#[test]
fn test_file_reading_pattern() {
    let source = r#"
with open("input.txt") as f:
    data = f.read()
print(data)
"#;
    assert_compiles(source);
}

#[test]
fn test_file_writing_pattern() {
    let source = r#"
with open("output.txt", "w") as f:
    f.write("Hello, World!")
"#;
    assert_compiles(source);
}

#[test]
fn test_lock_pattern() {
    let source = r#"
with lock:
    shared_resource += 1
"#;
    assert_compiles(source);
}

#[test]
fn test_database_transaction_pattern() {
    let source = r#"
with db.transaction() as tx:
    tx.execute("INSERT INTO users VALUES (?)", (name,))
    tx.execute("INSERT INTO logs VALUES (?)", (action,))
"#;
    assert_compiles(source);
}

#[test]
fn test_temp_directory_pattern() {
    let source = r#"
with tempfile.TemporaryDirectory() as tmpdir:
    path = tmpdir + "/file.txt"
    with open(path, "w") as f:
        f.write(data)
"#;
    assert_compiles(source);
}

#[test]
fn test_socket_pattern() {
    let source = r#"
with socket.create_connection(("localhost", 8080)) as sock:
    sock.sendall(request)
    response = sock.recv(1024)
"#;
    assert_compiles(source);
}

// ============================================================================
// With Statement in Function Tests
// ============================================================================

#[test]
fn test_with_in_function() {
    let source = r#"
def process_file(path):
    with open(path) as f:
        return f.read()
"#;
    assert_compiles(source);
}

#[test]
fn test_with_in_method() {
    let source = r#"
class FileProcessor:
    def process(self, path):
        with open(path) as f:
            self.data = f.read()
"#;
    assert_compiles(source);
}

#[test]
fn test_with_in_lambda_body() {
    // Note: lambdas don't support statements, so this uses a workaround
    let source = r#"
def use_manager(mgr, func):
    with mgr as m:
        return func(m)
        
# Usage: use_manager(open("file"), lambda f: f.read())
"#;
    assert_compiles(source);
}

// ============================================================================
// With Statement Generator Tests
// ============================================================================

#[test]
fn test_with_yield_in_body() {
    let source = r#"
def gen():
    with mgr as m:
        yield m.get_value()
        yield m.get_next()
"#;
    assert_compiles(source);
}

#[test]
fn test_with_yield_from() {
    let source = r#"
def gen():
    with mgr as items:
        yield from items
"#;
    assert_compiles(source);
}

// ============================================================================
// With Statement Class Tests
// ============================================================================

#[test]
fn test_with_in_class_body() {
    let source = r#"
class Config:
    with ConfigLoader() as loader:
        settings = loader.load_defaults()
"#;
    assert_compiles(source);
}

// ============================================================================
// Complex Combined Tests
// ============================================================================

#[test]
fn test_complex_with_scenario() {
    let source = r#"
def complex_processing(path):
    results = []
    with open(path) as f:
        for line in f:
            with process_context() as ctx:
                try:
                    result = ctx.process(line)
                    results.append(result)
                except ProcessError as e:
                    ctx.log_error(e)
    return results
"#;
    assert_compiles(source);
}

#[test]
fn test_nested_with_try_finally() {
    let source = r#"
with outer:
    try:
        with inner:
            might_fail()
    finally:
        cleanup()
"#;
    assert_compiles(source);
}

#[test]
fn test_with_comprehension_inside() {
    let source = r#"
with mgr as items:
    result = [x * 2 for x in items if x > 0]
"#;
    assert_compiles(source);
}

#[test]
fn test_with_dict_comprehension() {
    let source = r#"
with mgr as data:
    mapping = {k: v for k, v in data.items()}
"#;
    assert_compiles(source);
}
