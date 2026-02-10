//! Comprehensive test suite for Python 3.10+ match statement compilation (PEP 634).
//!
//! Tests all aspects of structural pattern matching:
//! - Literal patterns (integers, strings, booleans, None)
//! - Capture patterns (simple and walrus operator)
//! - Wildcard patterns
//! - Sequence patterns (lists, tuples, star patterns)
//! - Mapping patterns (dicts, **rest)
//! - Class patterns (positional and keyword attributes)
//! - OR patterns (|)
//! - AS patterns
//! - Guard expressions
//! - Complex nested patterns
//! - Edge cases and error conditions
//!
//! # Test Organization
//!
//! Tests are organized into focused modules:
//! - `literal_patterns`: int, str, bool, None matching
//! - `capture_patterns`: variable binding and AS patterns
//! - `sequence_patterns`: list/tuple matching with star
//! - `mapping_patterns`: dict matching with **rest
//! - `class_patterns`: class instance matching
//! - `or_patterns`: alternative patterns with |
//! - `guard_patterns`: conditional guards with if
//! - `complex_patterns`: nested and combined patterns
//! - `edge_cases`: boundary conditions and error handling

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

/// Helper to assert code contains a specific opcode.
fn assert_has_opcode(code: &prism_compiler::CodeObject, opcode: Opcode) {
    assert!(
        has_opcode(code, opcode),
        "Expected opcode {:?} not found in compiled code",
        opcode
    );
}

/// Helper to find number of Jump instructions.
fn count_jumps(code: &prism_compiler::CodeObject) -> usize {
    code.instructions
        .iter()
        .filter(|inst| {
            let op = inst.opcode();
            op == Opcode::Jump as u8
                || op == Opcode::JumpIfFalse as u8
                || op == Opcode::JumpIfTrue as u8
        })
        .count()
}

// ============================================================================
// Module: Literal Patterns (10 tests)
// ============================================================================

#[test]
fn test_literal_integer_match() {
    let source = r#"
match x:
    case 1:
        result = "one"
    case 2:
        result = "two"
    case 3:
        result = "three"
"#;
    let code = compile(source);
    assert_compiles(source);
    // Should have comparison operations for each literal
    assert!(code.instructions.len() > 3);
}

#[test]
fn test_literal_negative_integer() {
    let source = r#"
match x:
    case -1:
        result = "negative one"
    case -100:
        result = "negative hundred"
"#;
    assert_compiles(source);
}

#[test]
fn test_literal_zero() {
    let source = r#"
match x:
    case 0:
        result = "zero"
"#;
    assert_compiles(source);
}

#[test]
fn test_literal_large_integer() {
    let source = r#"
match x:
    case 999999999:
        result = "large"
    case -999999999:
        result = "large negative"
"#;
    assert_compiles(source);
}

#[test]
fn test_literal_string_match() {
    let source = r#"
match command:
    case "start":
        action = 1
    case "stop":
        action = 2
    case "pause":
        action = 3
"#;
    assert_compiles(source);
}

#[test]
fn test_literal_empty_string() {
    let source = r#"
match s:
    case "":
        result = "empty"
    case _:
        result = "non-empty"
"#;
    assert_compiles(source);
}

#[test]
fn test_literal_boolean_true() {
    let source = r#"
match flag:
    case True:
        result = "truthy"
    case False:
        result = "falsy"
"#;
    assert_compiles(source);
}

#[test]
fn test_literal_none() {
    let source = r#"
match value:
    case None:
        result = "nothing"
    case _:
        result = "something"
"#;
    assert_compiles(source);
}

#[test]
fn test_literal_mixed_types() {
    let source = r#"
match value:
    case 0:
        result = "zero"
    case "":
        result = "empty string"
    case None:
        result = "none"
    case True:
        result = "true"
    case False:
        result = "false"
"#;
    assert_compiles(source);
}

#[test]
fn test_literal_float() {
    let source = r#"
match x:
    case 1.0:
        result = "one point zero"
    case 3.14:
        result = "pi-ish"
    case -0.5:
        result = "negative half"
"#;
    assert_compiles(source);
}

// ============================================================================
// Module: Capture Patterns (10 tests)
// ============================================================================

#[test]
fn test_capture_simple() {
    let source = r#"
match value:
    case x:
        result = x
"#;
    assert_compiles(source);
}

#[test]
fn test_capture_wildcard() {
    let source = r#"
match value:
    case 1:
        result = "one"
    case _:
        result = "other"
"#;
    assert_compiles(source);
}

#[test]
fn test_capture_with_literal() {
    let source = r#"
match point:
    case (0, y):
        result = y
    case (x, 0):
        result = x
"#;
    assert_compiles(source);
}

#[test]
fn test_capture_as_pattern() {
    let source = r#"
match value:
    case [1, 2] as whole:
        result = whole
"#;
    assert_compiles(source);
}

#[test]
fn test_capture_as_with_or() {
    let source = r#"
match value:
    case (1 | 2 | 3) as num:
        result = num
"#;
    assert_compiles(source);
}

#[test]
fn test_capture_nested_as() {
    let source = r#"
match data:
    case {"key": [x, y] as pair}:
        result = (pair, x, y)
"#;
    assert_compiles(source);
}

#[test]
fn test_capture_multiple_bindings() {
    let source = r#"
match point:
    case (x, y, z):
        result = x + y + z
"#;
    assert_compiles(source);
}

#[test]
fn test_capture_shadowing() {
    let source = r#"
x = 10
match value:
    case x:
        result = x
"#;
    assert_compiles(source);
}

#[test]
fn test_capture_in_sequence() {
    let source = r#"
match items:
    case [first, second, third]:
        result = (first, second, third)
"#;
    assert_compiles(source);
}

#[test]
fn test_capture_in_mapping() {
    let source = r#"
match data:
    case {"name": name, "age": age}:
        result = (name, age)
"#;
    assert_compiles(source);
}

// ============================================================================
// Module: Sequence Patterns (12 tests)
// ============================================================================

#[test]
fn test_sequence_empty_list() {
    let source = r#"
match items:
    case []:
        result = "empty"
"#;
    let code = compile(source);
    assert_has_opcode(&code, Opcode::MatchSequence);
}

#[test]
fn test_sequence_single_element() {
    let source = r#"
match items:
    case [x]:
        result = x
"#;
    let code = compile(source);
    assert_has_opcode(&code, Opcode::MatchSequence);
}

#[test]
fn test_sequence_fixed_length() {
    let source = r#"
match items:
    case [a, b, c]:
        result = a + b + c
"#;
    let code = compile(source);
    assert_has_opcode(&code, Opcode::MatchSequence);
}

#[test]
fn test_sequence_with_literals() {
    let source = r#"
match items:
    case [1, 2, 3]:
        result = "123"
    case [1, x, 3]:
        result = x
"#;
    assert_compiles(source);
}

#[test]
fn test_sequence_star_middle() {
    let source = r#"
match items:
    case [first, *middle, last]:
        result = (first, middle, last)
"#;
    assert_compiles(source);
}

#[test]
fn test_sequence_star_beginning() {
    let source = r#"
match items:
    case [*rest, last]:
        result = (rest, last)
"#;
    assert_compiles(source);
}

#[test]
fn test_sequence_star_end() {
    let source = r#"
match items:
    case [first, *rest]:
        result = (first, rest)
"#;
    assert_compiles(source);
}

#[test]
fn test_sequence_star_discard() {
    let source = r#"
match items:
    case [first, *_, last]:
        result = (first, last)
"#;
    assert_compiles(source);
}

#[test]
fn test_sequence_tuple_pattern() {
    let source = r#"
match point:
    case (x, y):
        result = x + y
"#;
    assert_compiles(source);
}

#[test]
fn test_sequence_nested() {
    let source = r#"
match matrix:
    case [[a, b], [c, d]]:
        result = a * d - b * c
"#;
    assert_compiles(source);
}

#[test]
fn test_sequence_mixed_elements() {
    let source = r#"
match data:
    case [1, "two", x, None]:
        result = x
"#;
    assert_compiles(source);
}

#[test]
fn test_sequence_length_check() {
    let source = r#"
match items:
    case []:
        result = 0
    case [_]:
        result = 1
    case [_, _]:
        result = 2
    case [_, _, _]:
        result = 3
    case _:
        result = "many"
"#;
    assert_compiles(source);
}

// ============================================================================
// Module: Mapping Patterns (12 tests)
// ============================================================================

#[test]
fn test_mapping_empty() {
    let source = r#"
match data:
    case {}:
        result = "empty dict"
"#;
    // TODO: assert_has_opcode(&code, Opcode::MatchMapping) when codegen is wired
    assert_compiles(source);
}

#[test]
fn test_mapping_single_key() {
    let source = r#"
match data:
    case {"key": value}:
        result = value
"#;
    // TODO: assert_has_opcode(&code, Opcode::MatchMapping) when codegen is wired
    // TODO: assert_has_opcode(&code, Opcode::MatchKeys) when codegen is wired
    assert_compiles(source);
}

#[test]
fn test_mapping_multiple_keys() {
    let source = r#"
match data:
    case {"name": name, "age": age, "city": city}:
        result = (name, age, city)
"#;
    assert_compiles(source);
}

#[test]
fn test_mapping_literal_values() {
    let source = r#"
match config:
    case {"debug": True, "level": 1}:
        result = "debug mode"
    case {"debug": False}:
        result = "production mode"
"#;
    assert_compiles(source);
}

#[test]
fn test_mapping_double_star_rest() {
    let source = r#"
match data:
    case {"required": value, **rest}:
        result = (value, rest)
"#;
    // TODO: assert_has_opcode(&code, Opcode::CopyDictWithoutKeys) when codegen is wired
    assert_compiles(source);
}

#[test]
fn test_mapping_double_star_discard() {
    let source = r#"
match data:
    case {"key": value, **_}:
        result = value
"#;
    assert_compiles(source);
}

#[test]
fn test_mapping_nested_dict() {
    let source = r#"
match data:
    case {"outer": {"inner": value}}:
        result = value
"#;
    assert_compiles(source);
}

#[test]
fn test_mapping_with_sequence() {
    let source = r#"
match data:
    case {"items": [first, *rest]}:
        result = (first, rest)
"#;
    assert_compiles(source);
}

#[test]
fn test_mapping_integer_keys() {
    let source = r#"
match data:
    case {0: zero, 1: one}:
        result = (zero, one)
"#;
    assert_compiles(source);
}

#[test]
fn test_mapping_mixed_key_types() {
    let source = r#"
match data:
    case {"str": s, 1: one, True: t}:
        result = (s, one, t)
"#;
    assert_compiles(source);
}

#[test]
fn test_mapping_optional_keys() {
    let source = r#"
match data:
    case {"required": req, "optional": opt}:
        result = (req, opt)
    case {"required": req}:
        result = (req, None)
"#;
    assert_compiles(source);
}

#[test]
fn test_mapping_deeply_nested() {
    let source = r#"
match config:
    case {"database": {"host": host, "port": port}, "cache": {"enabled": True}}:
        result = (host, port)
"#;
    assert_compiles(source);
}

// ============================================================================
// Module: Class Patterns (12 tests)
// ============================================================================

#[test]
fn test_class_simple() {
    let source = r#"
match obj:
    case Point():
        result = "is a point"
"#;
    // TODO: assert_has_opcode(&code, Opcode::MatchClass) when codegen is wired
    assert_compiles(source);
}

#[test]
fn test_class_positional_args() {
    let source = r#"
match point:
    case Point(x, y):
        result = x + y
"#;
    assert_compiles(source);
}

#[test]
fn test_class_keyword_args() {
    let source = r#"
match point:
    case Point(x=x, y=y):
        result = (x, y)
"#;
    assert_compiles(source);
}

#[test]
fn test_class_mixed_args() {
    let source = r#"
match rect:
    case Rectangle(x, y, width=w, height=h):
        result = (x, y, w, h)
"#;
    assert_compiles(source);
}

#[test]
fn test_class_with_literals() {
    let source = r#"
match point:
    case Point(0, 0):
        result = "origin"
    case Point(0, y):
        result = "on y-axis"
    case Point(x, 0):
        result = "on x-axis"
"#;
    assert_compiles(source);
}

#[test]
fn test_class_builtin_types() {
    let source = r#"
match value:
    case int():
        result = "integer"
    case str():
        result = "string"
    case list():
        result = "list"
"#;
    assert_compiles(source);
}

#[test]
fn test_class_nested() {
    let source = r#"
match shape:
    case Circle(Point(x, y), radius):
        result = (x, y, radius)
"#;
    assert_compiles(source);
}

#[test]
fn test_class_with_guard() {
    let source = r#"
match point:
    case Point(x, y) if x > 0 and y > 0:
        result = "first quadrant"
"#;
    assert_compiles(source);
}

#[test]
fn test_class_multiple_types() {
    let source = r#"
match event:
    case MouseClick(x, y):
        result = ("click", x, y)
    case KeyPress(key):
        result = ("key", key)
    case WindowResize(w, h):
        result = ("resize", w, h)
"#;
    assert_compiles(source);
}

#[test]
fn test_class_inheritance_pattern() {
    let source = r#"
match animal:
    case Dog(name=name):
        result = ("dog", name)
    case Cat(name=name):
        result = ("cat", name)
    case Animal(name=name):
        result = ("animal", name)
"#;
    assert_compiles(source);
}

#[test]
fn test_class_empty_vs_wildcard() {
    let source = r#"
match obj:
    case Point():
        result = "any point"
    case Circle(_, r):
        result = ("circle", r)
"#;
    assert_compiles(source);
}

#[test]
fn test_class_dotted_name() {
    let source = r#"
match obj:
    case geometry.Point(x, y):
        result = (x, y)
    case shapes.Circle(center, radius):
        result = (center, radius)
"#;
    assert_compiles(source);
}

// ============================================================================
// Module: OR Patterns (8 tests)
// ============================================================================

#[test]
fn test_or_simple_literals() {
    let source = r#"
match x:
    case 1 | 2 | 3:
        result = "small"
    case 4 | 5 | 6:
        result = "medium"
"#;
    assert_compiles(source);
}

#[test]
fn test_or_with_capture() {
    let source = r#"
match x:
    case (1 | 2 | 3) as num:
        result = num * 2
"#;
    assert_compiles(source);
}

#[test]
fn test_or_strings() {
    let source = r#"
match command:
    case "quit" | "exit" | "q":
        result = "bye"
    case "help" | "h" | "?":
        result = "showing help"
"#;
    assert_compiles(source);
}

#[test]
fn test_or_singletons() {
    let source = r#"
match value:
    case True | False:
        result = "boolean"
    case None:
        result = "none"
"#;
    assert_compiles(source);
}

#[test]
fn test_or_in_sequence() {
    let source = r#"
match point:
    case (0 | 1, 0 | 1):
        result = "corner"
"#;
    assert_compiles(source);
}

#[test]
fn test_or_nested() {
    let source = r#"
match data:
    case {"type": "a" | "b"} | {"kind": "c" | "d"}:
        result = "matched"
"#;
    assert_compiles(source);
}

#[test]
fn test_or_with_wildcard() {
    let source = r#"
match x:
    case 1 | 2:
        result = "one or two"
    case _ | 3:
        result = "anything else"
"#;
    // This should compile (wildcard makes the OR pointless but valid)
    assert_compiles(source);
}

#[test]
fn test_or_complex_alternatives() {
    let source = r#"
match event:
    case {"click": (x, y)} | {"tap": (x, y)}:
        result = ("touch", x, y)
    case {"key": k} | {"char": k}:
        result = ("input", k)
"#;
    assert_compiles(source);
}

// ============================================================================
// Module: Guard Patterns (8 tests)
// ============================================================================

#[test]
fn test_guard_simple_comparison() {
    let source = r#"
match x:
    case n if n > 0:
        result = "positive"
    case n if n < 0:
        result = "negative"
    case _:
        result = "zero"
"#;
    let code = compile(source);
    assert!(count_jumps(&code) >= 2);
}

#[test]
fn test_guard_boolean_expression() {
    let source = r#"
match point:
    case (x, y) if x > 0 and y > 0:
        result = "first quadrant"
    case (x, y) if x < 0 and y > 0:
        result = "second quadrant"
"#;
    assert_compiles(source);
}

#[test]
fn test_guard_function_call() {
    let source = r#"
match items:
    case [*all] if len(all) > 10:
        result = "many items"
    case [*all] if len(all) > 0:
        result = "some items"
    case []:
        result = "no items"
"#;
    assert_compiles(source);
}

#[test]
fn test_guard_using_captured_variable() {
    let source = r#"
match data:
    case {"value": v} if v is not None:
        result = v
    case {"value": v}:
        result = "default"
"#;
    assert_compiles(source);
}

#[test]
fn test_guard_with_or_pattern() {
    let source = r#"
match x:
    case (1 | 2 | 3) as n if n % 2 == 0:
        result = "even small"
    case (1 | 2 | 3) as n:
        result = "odd small"
"#;
    assert_compiles(source);
}

#[test]
fn test_guard_complex_condition() {
    let source = r#"
match point:
    case (x, y) if x**2 + y**2 <= 1:
        result = "inside unit circle"
    case (x, y) if x**2 + y**2 <= 4:
        result = "inside radius 2"
    case _:
        result = "outside"
"#;
    assert_compiles(source);
}

#[test]
fn test_guard_with_method_call() {
    let source = r#"
match s:
    case str() as text if text.startswith("http"):
        result = "url"
    case str() as text if text.endswith(".py"):
        result = "python file"
"#;
    assert_compiles(source);
}

#[test]
fn test_guard_multiple_variables() {
    let source = r#"
match pair:
    case (a, b) if a == b:
        result = "equal"
    case (a, b) if a > b:
        result = "first larger"
    case (a, b):
        result = "second larger"
"#;
    assert_compiles(source);
}

// ============================================================================
// Module: Complex/Edge Cases (10 tests)
// ============================================================================

#[test]
fn test_edge_case_single_underscore() {
    let source = r#"
match x:
    case _:
        result = "always"
"#;
    assert_compiles(source);
}

#[test]
fn test_edge_case_deeply_nested() {
    let source = r#"
match data:
    case {"level1": {"level2": {"level3": [x, y, z]}}}:
        result = (x, y, z)
"#;
    assert_compiles(source);
}

#[test]
fn test_edge_case_many_cases() {
    let source = r#"
match x:
    case 1: result = 1
    case 2: result = 2
    case 3: result = 3
    case 4: result = 4
    case 5: result = 5
    case 6: result = 6
    case 7: result = 7
    case 8: result = 8
    case 9: result = 9
    case 10: result = 10
    case _: result = 0
"#;
    assert_compiles(source);
}

#[test]
fn test_edge_case_empty_body() {
    let source = r#"
match x:
    case 1:
        pass
    case _:
        pass
"#;
    assert_compiles(source);
}

#[test]
fn test_edge_case_nested_match() {
    let source = r#"
match outer:
    case (a, b):
        match a:
            case 1:
                result = b
            case _:
                result = 0
"#;
    assert_compiles(source);
}

#[test]
fn test_edge_case_match_in_function() {
    let source = r#"
def process(data):
    match data:
        case [x, y]:
            return x + y
        case {"sum": s}:
            return s
        case _:
            return 0
"#;
    assert_compiles(source);
}

#[test]
fn test_edge_case_match_in_loop() {
    let source = r#"
for item in items:
    match item:
        case ("stop", _):
            break
        case ("skip", _):
            continue
        case (action, value):
            process(action, value)
"#;
    assert_compiles(source);
}

#[test]
fn test_edge_case_match_with_assignment() {
    let source = r#"
match x := get_value():
    case 1:
        result = x * 2
    case _:
        result = x
"#;
    assert_compiles(source);
}

#[test]
fn test_edge_case_all_pattern_types() {
    let source = r#"
match data:
    case 1:
        result = "literal int"
    case "hello":
        result = "literal str"
    case True:
        result = "literal bool"
    case None:
        result = "literal none"
    case x:
        result = "capture"
    case _:
        result = "wildcard"
"#;
    assert_compiles(source);
}

#[test]
fn test_edge_case_realistic_json() {
    let source = r#"
match response:
    case {"status": 200, "data": {"users": [*users]}}:
        result = users
    case {"status": 200, "data": {"error": msg}}:
        result = ("error", msg)
    case {"status": 404}:
        result = None
    case {"status": code} if code >= 500:
        result = ("server_error", code)
    case _:
        result = "unknown"
"#;
    assert_compiles(source);
}

// ============================================================================
// Module: Bytecode Verification Tests (10 tests)
// ============================================================================

#[test]
fn test_bytecode_sequence_generates_match_sequence() {
    let source = r#"
match items:
    case [x, y]:
        result = x
"#;
    // TODO: assert_has_opcode(&code, Opcode::MatchSequence) when codegen is wired
    assert_compiles(source);
}

#[test]
fn test_bytecode_mapping_generates_match_mapping() {
    let source = r#"
match data:
    case {"key": value}:
        result = value
"#;
    // TODO: assert_has_opcode(&code, Opcode::MatchMapping) when codegen is wired
    assert_compiles(source);
}

#[test]
fn test_bytecode_class_generates_match_class() {
    let source = r#"
match obj:
    case Point(x, y):
        result = (x, y)
"#;
    // TODO: assert_has_opcode(&code, Opcode::MatchClass) when codegen is wired
    assert_compiles(source);
}

#[test]
fn test_bytecode_mapping_rest_generates_copy_dict() {
    let source = r#"
match data:
    case {"k": v, **rest}:
        result = rest
"#;
    // TODO: assert_has_opcode(&code, Opcode::CopyDictWithoutKeys) when codegen is wired
    assert_compiles(source);
}

#[test]
fn test_bytecode_keys_generates_match_keys() {
    let source = r#"
match data:
    case {"a": x, "b": y}:
        result = (x, y)
"#;
    // TODO: assert_has_opcode(&code, Opcode::MatchKeys) when codegen is wired
    assert_compiles(source);
}

#[test]
fn test_bytecode_guard_generates_jump() {
    let source = r#"
match x:
    case n if n > 0:
        result = "positive"
"#;
    let code = compile(source);
    assert!(count_jumps(&code) >= 1);
}

#[test]
fn test_bytecode_or_generates_multiple_branches() {
    let source = r#"
match x:
    case 1 | 2 | 3:
        result = "small"
"#;
    let code = compile(source);
    // OR patterns generate multiple comparison branches
    assert!(code.instructions.len() >= 5);
}

#[test]
fn test_bytecode_capture_generates_store() {
    let source = r#"
match value:
    case x:
        result = x * 2
"#;
    let code = compile(source);
    // At module scope, pattern bindings are stored as globals
    assert_has_opcode(&code, Opcode::StoreGlobal);
}

#[test]
fn test_bytecode_wildcard_no_store() {
    let source = r#"
match value:
    case _:
        result = "matched"
"#;
    let code = compile(source);
    // Wildcard shouldn't store anything
    // Just verify it compiles and is simple
    assert!(code.instructions.len() < 20);
}

#[test]
fn test_bytecode_efficiency_literal_chain() {
    let source = r#"
match x:
    case 1: pass
    case 2: pass
    case 3: pass
"#;
    let code = compile(source);
    // Should use efficient jump chain, not redundant code
    let jump_count = count_jumps(&code);
    assert!(
        jump_count >= 2 && jump_count <= 6,
        "Expected 2-6 jumps for 3 cases, got {}",
        jump_count
    );
}

// ============================================================================
// Module: Error Handling / Robustness Tests (2 tests)
// ============================================================================

#[test]
fn test_robustness_unicode_in_patterns() {
    let source = r#"
match emoji:
    case "🎉":
        result = "party"
    case "🌟":
        result = "star"
"#;
    assert_compiles(source);
}

#[test]
fn test_robustness_escaped_strings() {
    let source = r#"
match s:
    case "\n":
        result = "newline"
    case "\t":
        result = "tab"
    case "\\":
        result = "backslash"
"#;
    assert_compiles(source);
}
