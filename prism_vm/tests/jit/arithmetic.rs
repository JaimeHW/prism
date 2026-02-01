//! JIT Integration Tests: Arithmetic Operations
//!
//! Tests for integer, float, and generic arithmetic through the JIT pipeline.

use super::test_utils::*;
use prism_core::Value;

// =============================================================================
// Integer Arithmetic
// =============================================================================

#[test]
fn test_jit_int_add() {
    let mut vm = create_test_vm();

    let code = create_code_object(
        "int_add",
        vec![
            load_const(0, 0), // r0 = 10
            load_const(1, 1), // r1 = 32
            add(2, 0, 1),     // r2 = r0 + r1
            return_reg(2),
        ],
        vec![Value::int(10).unwrap(), Value::int(32).unwrap()],
        3,
        0,
    );

    let result = execute_with_tierup(&mut vm, code, 15);
    assert_eq!(result.as_int(), Some(42));
}

#[test]
fn test_jit_int_sub() {
    let mut vm = create_test_vm();

    let code = create_code_object(
        "int_sub",
        vec![
            load_const(0, 0), // r0 = 100
            load_const(1, 1), // r1 = 58
            sub(2, 0, 1),     // r2 = r0 - r1
            return_reg(2),
        ],
        vec![Value::int(100).unwrap(), Value::int(58).unwrap()],
        3,
        0,
    );

    let result = execute_with_tierup(&mut vm, code, 15);
    assert_eq!(result.as_int(), Some(42));
}

#[test]
fn test_jit_int_mul() {
    let mut vm = create_test_vm();

    let code = create_code_object(
        "int_mul",
        vec![
            load_const(0, 0), // r0 = 6
            load_const(1, 1), // r1 = 7
            mul(2, 0, 1),     // r2 = r0 * r1
            return_reg(2),
        ],
        vec![Value::int(6).unwrap(), Value::int(7).unwrap()],
        3,
        0,
    );

    let result = execute_with_tierup(&mut vm, code, 15);
    assert_eq!(result.as_int(), Some(42));
}

#[test]
fn test_jit_int_floor_div() {
    let mut vm = create_test_vm();

    let code = create_code_object(
        "int_floor_div",
        vec![
            load_const(0, 0),   // r0 = 85
            load_const(1, 1),   // r1 = 2
            floor_div(2, 0, 1), // r2 = r0 // r1
            return_reg(2),
        ],
        vec![Value::int(85).unwrap(), Value::int(2).unwrap()],
        3,
        0,
    );

    let result = execute_with_tierup(&mut vm, code, 15);
    assert_eq!(result.as_int(), Some(42));
}

#[test]
fn test_jit_int_modulo() {
    let mut vm = create_test_vm();

    let code = create_code_object(
        "int_mod",
        vec![
            load_const(0, 0), // r0 = 142
            load_const(1, 1), // r1 = 100
            modulo(2, 0, 1),  // r2 = r0 % r1
            return_reg(2),
        ],
        vec![Value::int(142).unwrap(), Value::int(100).unwrap()],
        3,
        0,
    );

    let result = execute_with_tierup(&mut vm, code, 15);
    assert_eq!(result.as_int(), Some(42));
}

#[test]
fn test_jit_chained_arithmetic() {
    let mut vm = create_test_vm();

    // (2 + 3) * (10 - 2) = 5 * 8 = 40
    let code = create_code_object(
        "chained",
        vec![
            load_const(0, 0), // r0 = 2
            load_const(1, 1), // r1 = 3
            load_const(2, 2), // r2 = 10
            load_const(3, 3), // r3 = 2
            add(4, 0, 1),     // r4 = r0 + r1 = 5
            sub(5, 2, 3),     // r5 = r2 - r3 = 8
            mul(6, 4, 5),     // r6 = r4 * r5 = 40
            return_reg(6),
        ],
        vec![
            Value::int(2).unwrap(),
            Value::int(3).unwrap(),
            Value::int(10).unwrap(),
            Value::int(2).unwrap(),
        ],
        7,
        0,
    );

    let result = execute_with_tierup(&mut vm, code, 15);
    assert_eq!(result.as_int(), Some(40));
}

// =============================================================================
// Float Arithmetic
// =============================================================================

#[test]
fn test_jit_float_add() {
    let mut vm = create_test_vm();

    let code = create_code_object(
        "float_add",
        vec![
            load_const(0, 0), // r0 = 3.14
            load_const(1, 1), // r1 = 2.86
            add(2, 0, 1),     // r2 = r0 + r1
            return_reg(2),
        ],
        vec![Value::float(3.14), Value::float(2.86)],
        3,
        0,
    );

    let result = execute_with_tierup(&mut vm, code, 15);
    let f = result.as_float().expect("Expected float");
    assert!((f - 6.0).abs() < 0.0001);
}

#[test]
fn test_jit_float_mul() {
    let mut vm = create_test_vm();

    let code = create_code_object(
        "float_mul",
        vec![
            load_const(0, 0), // r0 = 2.5
            load_const(1, 1), // r1 = 4.0
            mul(2, 0, 1),     // r2 = r0 * r1
            return_reg(2),
        ],
        vec![Value::float(2.5), Value::float(4.0)],
        3,
        0,
    );

    let result = execute_with_tierup(&mut vm, code, 15);
    let f = result.as_float().expect("Expected float");
    assert!((f - 10.0).abs() < 0.0001);
}

#[test]
fn test_jit_float_div() {
    let mut vm = create_test_vm();

    let code = create_code_object(
        "float_div",
        vec![
            load_const(0, 0),  // r0 = 10.0
            load_const(1, 1),  // r1 = 4.0
            true_div(2, 0, 1), // r2 = r0 / r1
            return_reg(2),
        ],
        vec![Value::float(10.0), Value::float(4.0)],
        3,
        0,
    );

    let result = execute_with_tierup(&mut vm, code, 15);
    let f = result.as_float().expect("Expected float");
    assert!((f - 2.5).abs() < 0.0001);
}

// =============================================================================
// Comparisons
// =============================================================================

#[test]
fn test_jit_comparison_lt() {
    let mut vm = create_test_vm();

    let code = create_code_object(
        "cmp_lt",
        vec![
            load_const(0, 0), // r0 = 5
            load_const(1, 1), // r1 = 10
            lt(2, 0, 1),      // r2 = r0 < r1 (True)
            return_reg(2),
        ],
        vec![Value::int(5).unwrap(), Value::int(10).unwrap()],
        3,
        0,
    );

    let result = execute_with_tierup(&mut vm, code, 15);
    assert_eq!(result.as_bool(), Some(true));
}

#[test]
fn test_jit_comparison_ge() {
    let mut vm = create_test_vm();

    let code = create_code_object(
        "cmp_ge",
        vec![
            load_const(0, 0), // r0 = 10
            load_const(1, 1), // r1 = 10
            ge(2, 0, 1),      // r2 = r0 >= r1 (True)
            return_reg(2),
        ],
        vec![Value::int(10).unwrap(), Value::int(10).unwrap()],
        3,
        0,
    );

    let result = execute_with_tierup(&mut vm, code, 15);
    assert_eq!(result.as_bool(), Some(true));
}

#[test]
fn test_jit_comparison_eq() {
    let mut vm = create_test_vm();

    let code = create_code_object(
        "cmp_eq",
        vec![
            load_const(0, 0), // r0 = 42
            load_const(1, 1), // r1 = 42
            eq(2, 0, 1),      // r2 = r0 == r1 (True)
            return_reg(2),
        ],
        vec![Value::int(42).unwrap(), Value::int(42).unwrap()],
        3,
        0,
    );

    let result = execute_with_tierup(&mut vm, code, 15);
    assert_eq!(result.as_bool(), Some(true));
}

#[test]
fn test_jit_comparison_ne() {
    let mut vm = create_test_vm();

    let code = create_code_object(
        "cmp_ne",
        vec![
            load_const(0, 0), // r0 = 10
            load_const(1, 1), // r1 = 20
            ne(2, 0, 1),      // r2 = r0 != r1 (True)
            return_reg(2),
        ],
        vec![Value::int(10).unwrap(), Value::int(20).unwrap()],
        3,
        0,
    );

    let result = execute_with_tierup(&mut vm, code, 15);
    assert_eq!(result.as_bool(), Some(true));
}

// =============================================================================
// Edge Cases
// =============================================================================

#[test]
fn test_jit_large_int() {
    let mut vm = create_test_vm();

    // Use large but representable integers (within tagged int range)
    let a = 1_000_000_000i64;
    let b = 999_999_999i64;

    let code = create_code_object(
        "large_int",
        vec![
            load_const(0, 0),
            load_const(1, 1),
            add(2, 0, 1),
            return_reg(2),
        ],
        vec![Value::int(a).unwrap(), Value::int(b).unwrap()],
        3,
        0,
    );

    let result = execute_with_tierup(&mut vm, code, 15);
    assert_eq!(result.as_int(), Some(a + b));
}

#[test]
fn test_jit_negative_int() {
    let mut vm = create_test_vm();

    let code = create_code_object(
        "negative_int",
        vec![
            load_const(0, 0), // r0 = -10
            load_const(1, 1), // r1 = -32
            add(2, 0, 1),     // r2 = -42
            return_reg(2),
        ],
        vec![Value::int(-10).unwrap(), Value::int(-32).unwrap()],
        3,
        0,
    );

    let result = execute_with_tierup(&mut vm, code, 15);
    assert_eq!(result.as_int(), Some(-42));
}

#[test]
fn test_jit_zero_operations() {
    let mut vm = create_test_vm();

    let code = create_code_object(
        "zero_ops",
        vec![
            load_const(0, 0), // r0 = 0
            load_const(1, 1), // r1 = 42
            add(2, 0, 1),     // r2 = 0 + 42 = 42
            mul(3, 0, 1),     // r3 = 0 * 42 = 0
            add(4, 2, 3),     // r4 = 42 + 0 = 42
            return_reg(4),
        ],
        vec![Value::int(0).unwrap(), Value::int(42).unwrap()],
        5,
        0,
    );

    let result = execute_with_tierup(&mut vm, code, 15);
    assert_eq!(result.as_int(), Some(42));
}
