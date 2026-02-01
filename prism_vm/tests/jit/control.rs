//! JIT Integration Tests: Control Flow
//!
//! Tests for loops, branches, and control flow through the JIT pipeline.

use super::test_utils::*;
use prism_core::Value;

// =============================================================================
// Simple Loops
// =============================================================================

#[test]
fn test_jit_simple_loop_sum() {
    let mut vm = create_test_vm();

    // sum = 0; i = 0
    // while i < 10:
    //     sum += i
    //     i += 1
    // return sum
    //
    // Result: 0 + 1 + 2 + ... + 9 = 45

    let instructions = vec![
        // 0: r0 = 0 (sum)
        load_const(0, 0),
        // 1: r1 = 0 (i)
        load_const(1, 0),
        // 2: r2 = 10 (limit)
        load_const(2, 1),
        // 3: r3 = 1 (increment)
        load_const(3, 2),
        // -- Loop header (offset 4) --
        // 4: r4 = i < limit
        lt(4, 1, 2),
        // 5: if !r4 goto exit (offset 9, rel = 9-6 = 3)
        jump_if_false(4, 3),
        // 6: sum = sum + i
        add(0, 0, 1),
        // 7: i = i + 1
        add(1, 1, 3),
        // 8: goto loop header (offset 4, rel = 4-9 = -5)
        jump(-5),
        // -- Exit (offset 9) --
        // 9: return sum
        return_reg(0),
    ];

    let code = create_code_object(
        "loop_sum",
        instructions,
        vec![
            Value::int(0).unwrap(),
            Value::int(10).unwrap(),
            Value::int(1).unwrap(),
        ],
        5,
        0,
    );

    let result = execute_with_tierup(&mut vm, code, 15);
    assert_eq!(result.as_int(), Some(45));
}

#[test]
fn test_jit_countdown_loop() {
    let mut vm = create_test_vm();

    // n = 100
    // while n > 0:
    //     n -= 1
    // return n
    //
    // Result: 0

    let instructions = vec![
        // 0: r0 = 100 (n)
        load_const(0, 0),
        // 1: r1 = 0 (zero)
        load_const(1, 1),
        // 2: r2 = 1 (decrement)
        load_const(2, 2),
        // -- Loop header (offset 3) --
        // 3: r3 = n > 0
        gt(3, 0, 1),
        // 4: if !r3 goto exit (offset 7, rel = 7-5 = 2)
        jump_if_false(3, 2),
        // 5: n = n - 1
        sub(0, 0, 2),
        // 6: goto loop header (offset 3, rel = 3-7 = -4)
        jump(-4),
        // -- Exit (offset 7) --
        // 7: return n
        return_reg(0),
    ];

    let code = create_code_object(
        "countdown",
        instructions,
        vec![
            Value::int(100).unwrap(),
            Value::int(0).unwrap(),
            Value::int(1).unwrap(),
        ],
        4,
        0,
    );

    let result = execute_with_tierup(&mut vm, code, 15);
    assert_eq!(result.as_int(), Some(0));
}

#[test]
fn test_jit_loop_multiply() {
    let mut vm = create_test_vm();

    // Multiply by repeated addition: 7 * 6 = 42
    // result = 0; i = 0
    // while i < 6:
    //     result += 7
    //     i += 1
    // return result

    let instructions = vec![
        // 0: r0 = 0 (result)
        load_const(0, 0),
        // 1: r1 = 0 (i)
        load_const(1, 0),
        // 2: r2 = 6 (limit)
        load_const(2, 1),
        // 3: r3 = 7 (addend)
        load_const(3, 2),
        // 4: r4 = 1 (increment)
        load_const(4, 3),
        // -- Loop header (offset 5) --
        // 5: r5 = i < limit
        lt(5, 1, 2),
        // 6: if !r5 goto exit (offset 10, rel = 10-7 = 3)
        jump_if_false(5, 3),
        // 7: result = result + addend
        add(0, 0, 3),
        // 8: i = i + 1
        add(1, 1, 4),
        // 9: goto loop header (offset 5, rel = 5-10 = -5)
        jump(-5),
        // -- Exit (offset 10) --
        // 10: return result
        return_reg(0),
    ];

    let code = create_code_object(
        "loop_multiply",
        instructions,
        vec![
            Value::int(0).unwrap(),
            Value::int(6).unwrap(),
            Value::int(7).unwrap(),
            Value::int(1).unwrap(),
        ],
        6,
        0,
    );

    let result = execute_with_tierup(&mut vm, code, 15);
    assert_eq!(result.as_int(), Some(42));
}

// =============================================================================
// Conditional Branches
// =============================================================================

#[test]
fn test_jit_simple_if_true() {
    let mut vm = create_test_vm();

    // if True:
    //     return 42
    // return 0

    let instructions = vec![
        // 0: r0 = True
        load_const(0, 0),
        // 1: if !r0 goto else (offset 4, rel = 4-2 = 2)
        jump_if_false(0, 2),
        // 2: r1 = 42
        load_const(1, 1),
        // 3: return r1
        return_reg(1),
        // -- Else (offset 4) --
        // 4: r1 = 0
        load_const(1, 2),
        // 5: return r1
        return_reg(1),
    ];

    let code = create_code_object(
        "if_true",
        instructions,
        vec![
            Value::bool(true),
            Value::int(42).unwrap(),
            Value::int(0).unwrap(),
        ],
        2,
        0,
    );

    let result = execute_with_tierup(&mut vm, code, 15);
    assert_eq!(result.as_int(), Some(42));
}

#[test]
fn test_jit_simple_if_false() {
    let mut vm = create_test_vm();

    // if False:
    //     return 42
    // return 0

    let instructions = vec![
        // 0: r0 = False
        load_const(0, 0),
        // 1: if !r0 goto else (offset 4, rel = 4-2 = 2)
        jump_if_false(0, 2),
        // 2: r1 = 42
        load_const(1, 1),
        // 3: return r1
        return_reg(1),
        // -- Else (offset 4) --
        // 4: r1 = 0
        load_const(1, 2),
        // 5: return r1
        return_reg(1),
    ];

    let code = create_code_object(
        "if_false",
        instructions,
        vec![
            Value::bool(false),
            Value::int(42).unwrap(),
            Value::int(0).unwrap(),
        ],
        2,
        0,
    );

    let result = execute_with_tierup(&mut vm, code, 15);
    assert_eq!(result.as_int(), Some(0));
}

#[test]
fn test_jit_comparison_branch() {
    let mut vm = create_test_vm();

    // a = 10; b = 5
    // if a > b:
    //     return a
    // return b

    let instructions = vec![
        // 0: r0 = 10 (a)
        load_const(0, 0),
        // 1: r1 = 5 (b)
        load_const(1, 1),
        // 2: r2 = a > b
        gt(2, 0, 1),
        // 3: if !r2 goto else (offset 5, rel = 5-4 = 1)
        jump_if_false(2, 1),
        // 4: return a
        return_reg(0),
        // -- Else (offset 5) --
        // 5: return b
        return_reg(1),
    ];

    let code = create_code_object(
        "cmp_branch",
        instructions,
        vec![Value::int(10).unwrap(), Value::int(5).unwrap()],
        3,
        0,
    );

    let result = execute_with_tierup(&mut vm, code, 15);
    assert_eq!(result.as_int(), Some(10));
}

// =============================================================================
// Nested Loops
// =============================================================================

#[test]
fn test_jit_nested_loop() {
    let mut vm = create_test_vm();

    // Compute 3 * 4 = 12 using nested loops
    // result = 0
    // for i in range(3):
    //     for j in range(4):
    //         result += 1
    // return result

    let instructions = vec![
        // 0: r0 = 0 (result)
        load_const(0, 0),
        // 1: r1 = 0 (i)
        load_const(1, 0),
        // 2: r2 = 3 (outer limit)
        load_const(2, 1),
        // 3: r5 = 4 (inner limit)
        load_const(5, 2),
        // 4: r6 = 1 (increment)
        load_const(6, 3),
        // -- Outer loop header (offset 5) --
        // 5: r7 = i < 3
        lt(7, 1, 2),
        // 6: if !r7 goto exit (offset 15, rel = 15-7 = 8)
        jump_if_false(7, 8),
        // 7: r3 = 0 (j) -- reset inner counter
        load_const(3, 0),
        // -- Inner loop header (offset 8) --
        // 8: r4 = j < 4
        lt(4, 3, 5),
        // 9: if !r4 goto outer continue (offset 13, rel = 13-10 = 3)
        jump_if_false(4, 3),
        // 10: result += 1
        add(0, 0, 6),
        // 11: j += 1
        add(3, 3, 6),
        // 12: goto inner header (offset 8, rel = 8-13 = -5)
        jump(-5),
        // -- Outer continue (offset 13) --
        // 13: i += 1
        add(1, 1, 6),
        // 14: goto outer header (offset 5, rel = 5-15 = -10)
        jump(-10),
        // -- Exit (offset 15) --
        // 15: return result
        return_reg(0),
    ];

    let code = create_code_object(
        "nested_loop",
        instructions,
        vec![
            Value::int(0).unwrap(),
            Value::int(3).unwrap(),
            Value::int(4).unwrap(),
            Value::int(1).unwrap(),
        ],
        8,
        0,
    );

    let result = execute_with_tierup(&mut vm, code, 15);
    assert_eq!(result.as_int(), Some(12));
}

// =============================================================================
// Loop with Early Exit
// =============================================================================

#[test]
fn test_jit_loop_early_exit() {
    let mut vm = create_test_vm();

    // Find first i where i * 7 >= 42
    // i = 0
    // while True:
    //     if i * 7 >= 42:
    //         return i
    //     i += 1

    let instructions = vec![
        // 0: r0 = 0 (i)
        load_const(0, 0),
        // 1: r1 = 7
        load_const(1, 1),
        // 2: r2 = 42
        load_const(2, 2),
        // 3: r3 = 1 (increment)
        load_const(3, 3),
        // -- Loop header (offset 4) --
        // 4: r4 = i * 7
        mul(4, 0, 1),
        // 5: r5 = r4 >= 42
        ge(5, 4, 2),
        // 6: if r5 goto exit (offset 9, rel = 9-7 = 2)
        jump_if_true(5, 2),
        // 7: i += 1
        add(0, 0, 3),
        // 8: goto loop header (offset 4, rel = 4-9 = -5)
        jump(-5),
        // -- Exit (offset 9) --
        // 9: return i
        return_reg(0),
    ];

    let code = create_code_object(
        "early_exit",
        instructions,
        vec![
            Value::int(0).unwrap(),
            Value::int(7).unwrap(),
            Value::int(42).unwrap(),
            Value::int(1).unwrap(),
        ],
        6,
        0,
    );

    let result = execute_with_tierup(&mut vm, code, 15);
    assert_eq!(result.as_int(), Some(6)); // 6 * 7 = 42
}
