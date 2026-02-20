//! End-to-End Speculation Pipeline Integration Tests
//!
//! Tests the complete speculation pipeline from bytecode through interpreter
//! to speculative execution. Focuses on correctness of operations.

use prism_compiler::bytecode::{
    CodeFlags, CodeObject, ExceptionEntry, Instruction, Opcode, Register,
};
use prism_core::Value;
use prism_vm::{JitConfig, VirtualMachine};
use std::sync::Arc;

// =============================================================================
// Test Helpers
// =============================================================================

/// Create a code object for integer addition: a + b
fn create_int_add_code(a: i64, b: i64) -> Arc<CodeObject> {
    let instructions = vec![
        Instruction::op_di(Opcode::LoadConst, Register(0), 0),
        Instruction::op_di(Opcode::LoadConst, Register(1), 1),
        Instruction::op_dss(Opcode::Add, Register(2), Register(0), Register(1)),
        Instruction::op_d(Opcode::Return, Register(2)),
    ];

    let constants = vec![Value::int(a).unwrap(), Value::int(b).unwrap()];

    Arc::new(CodeObject {
        name: "test_int_add".into(),
        register_count: 3,
        arg_count: 0,
        posonlyarg_count: 0,
        kwonlyarg_count: 0,
        instructions: instructions.into_boxed_slice(),
        constants: constants.into_boxed_slice(),
        names: Box::new([]),
        locals: Box::new([]),
        freevars: Box::new([]),
        cellvars: Box::new([]),
        line_table: Box::new([]),
        exception_table: Box::new([]),
        filename: "test.py".into(),
        qualname: "test_int_add".into(),
        flags: CodeFlags::NONE,
        first_lineno: 1,
        nested_code_objects: Box::new([]),
    })
}

/// Create a code object for float addition
fn create_float_add_code(a: f64, b: f64) -> Arc<CodeObject> {
    let instructions = vec![
        Instruction::op_di(Opcode::LoadConst, Register(0), 0),
        Instruction::op_di(Opcode::LoadConst, Register(1), 1),
        Instruction::op_dss(Opcode::Add, Register(2), Register(0), Register(1)),
        Instruction::op_d(Opcode::Return, Register(2)),
    ];

    let constants = vec![Value::float(a), Value::float(b)];

    Arc::new(CodeObject {
        name: "test_float_add".into(),
        register_count: 3,
        arg_count: 0,
        posonlyarg_count: 0,
        kwonlyarg_count: 0,
        instructions: instructions.into_boxed_slice(),
        constants: constants.into_boxed_slice(),
        names: Box::new([]),
        locals: Box::new([]),
        freevars: Box::new([]),
        cellvars: Box::new([]),
        line_table: Box::new([]),
        exception_table: Box::new([]),
        filename: "test.py".into(),
        qualname: "test_float_add".into(),
        flags: CodeFlags::NONE,
        first_lineno: 1,
        nested_code_objects: Box::new([]),
    })
}

/// Create a code object for comparison: a < b
fn create_lt_code(a: i64, b: i64) -> Arc<CodeObject> {
    let instructions = vec![
        Instruction::op_di(Opcode::LoadConst, Register(0), 0),
        Instruction::op_di(Opcode::LoadConst, Register(1), 1),
        Instruction::op_dss(Opcode::Lt, Register(2), Register(0), Register(1)),
        Instruction::op_d(Opcode::Return, Register(2)),
    ];

    let constants = vec![Value::int(a).unwrap(), Value::int(b).unwrap()];

    Arc::new(CodeObject {
        name: "test_lt".into(),
        register_count: 3,
        arg_count: 0,
        posonlyarg_count: 0,
        kwonlyarg_count: 0,
        instructions: instructions.into_boxed_slice(),
        constants: constants.into_boxed_slice(),
        names: Box::new([]),
        locals: Box::new([]),
        freevars: Box::new([]),
        cellvars: Box::new([]),
        line_table: Box::new([]),
        exception_table: Box::new([]),
        filename: "test.py".into(),
        qualname: "test_lt".into(),
        flags: CodeFlags::NONE,
        first_lineno: 1,
        nested_code_objects: Box::new([]),
    })
}

/// Create a multiplication code object
fn create_mul_code(a: i64, b: i64) -> Arc<CodeObject> {
    let instructions = vec![
        Instruction::op_di(Opcode::LoadConst, Register(0), 0),
        Instruction::op_di(Opcode::LoadConst, Register(1), 1),
        Instruction::op_dss(Opcode::Mul, Register(2), Register(0), Register(1)),
        Instruction::op_d(Opcode::Return, Register(2)),
    ];

    let constants = vec![Value::int(a).unwrap(), Value::int(b).unwrap()];

    Arc::new(CodeObject {
        name: "test_mul".into(),
        register_count: 3,
        arg_count: 0,
        posonlyarg_count: 0,
        kwonlyarg_count: 0,
        instructions: instructions.into_boxed_slice(),
        constants: constants.into_boxed_slice(),
        names: Box::new([]),
        locals: Box::new([]),
        freevars: Box::new([]),
        cellvars: Box::new([]),
        line_table: Box::new([]),
        exception_table: Box::new([]),
        filename: "test.py".into(),
        qualname: "test_mul".into(),
        flags: CodeFlags::NONE,
        first_lineno: 1,
        nested_code_objects: Box::new([]),
    })
}

/// Create a hot loop with N iterations
fn create_hot_loop_code(n: i64) -> Arc<CodeObject> {
    let mut instructions = Vec::new();
    let mut constants = Vec::new();

    // LOAD_CONST 0 -> r0 (counter = 0)
    constants.push(Value::int(0).unwrap());
    instructions.push(Instruction::op_di(Opcode::LoadConst, Register(0), 0));

    // LOAD_CONST 1 -> r1 (limit = n)
    constants.push(Value::int(n).unwrap());
    instructions.push(Instruction::op_di(Opcode::LoadConst, Register(1), 1));

    // LOAD_CONST 2 -> r2 (increment = 1)
    constants.push(Value::int(1).unwrap());
    instructions.push(Instruction::op_di(Opcode::LoadConst, Register(2), 2));

    // Loop header
    let loop_header = instructions.len();

    // LT r0, r1 -> r3
    instructions.push(Instruction::op_dss(
        Opcode::Lt,
        Register(3),
        Register(0),
        Register(1),
    ));

    // JUMP_IF_FALSE r3, exit
    let jump_if_false_idx = instructions.len();
    instructions.push(Instruction::op_di(Opcode::JumpIfFalse, Register(3), 0));

    // ADD r0, r2 -> r0
    instructions.push(Instruction::op_dss(
        Opcode::Add,
        Register(0),
        Register(0),
        Register(2),
    ));

    // JUMP loop_header
    let jump_back_idx = instructions.len();
    instructions.push(Instruction::op_di(Opcode::Jump, Register(0), 0));

    // Fixup backward jump
    let next_ip = jump_back_idx + 1;
    let offset = (loop_header as i32) - (next_ip as i32);
    instructions[jump_back_idx] =
        Instruction::op_di(Opcode::Jump, Register(0), offset as i16 as u16);

    // Exit
    let exit_idx = instructions.len();

    // Fixup forward jump
    let next_ip_fwd = jump_if_false_idx + 1;
    let fwd_offset = (exit_idx as i32) - (next_ip_fwd as i32);
    instructions[jump_if_false_idx] =
        Instruction::op_di(Opcode::JumpIfFalse, Register(3), fwd_offset as i16 as u16);

    // RETURN r0
    instructions.push(Instruction::op_d(Opcode::Return, Register(0)));

    Arc::new(CodeObject {
        name: "test_hot_loop".into(),
        register_count: 4,
        arg_count: 0,
        posonlyarg_count: 0,
        kwonlyarg_count: 0,
        instructions: instructions.into_boxed_slice(),
        constants: constants.into_boxed_slice(),
        names: Box::new([]),
        locals: Box::new([]),
        freevars: Box::new([]),
        cellvars: Box::new([]),
        line_table: Box::new([]),
        exception_table: Box::new([]),
        filename: "test.py".into(),
        qualname: "test_hot_loop".into(),
        flags: CodeFlags::NONE,
        first_lineno: 1,
        nested_code_objects: Box::new([]),
    })
}

// =============================================================================
// Integer Arithmetic Tests
// =============================================================================

#[test]
fn test_e2e_integer_addition_correctness() {
    let config = JitConfig::for_testing();
    let mut vm = VirtualMachine::with_jit_config(config);

    let test_cases = [
        (0, 0, 0),
        (1, 2, 3),
        (-5, 5, 0),
        (100, 200, 300),
        (1_000_000, 1_000_000, 2_000_000),
    ];

    for (a, b, expected) in test_cases {
        let code = create_int_add_code(a, b);
        let result = vm.execute(code).unwrap();
        assert_eq!(
            result.as_int(),
            Some(expected),
            "Failed for {} + {} = {}",
            a,
            b,
            expected
        );
    }
}

// =============================================================================
// Float Arithmetic Tests
// =============================================================================

#[test]
fn test_e2e_float_addition_correctness() {
    let config = JitConfig::for_testing();
    let mut vm = VirtualMachine::with_jit_config(config);

    let test_cases = [
        (0.0, 0.0, 0.0),
        (1.5, 2.5, 4.0),
        (-3.14, 3.14, 0.0),
        (1e10, 1e10, 2e10),
    ];

    for (a, b, expected) in test_cases {
        let code = create_float_add_code(a, b);
        let result = vm.execute(code).unwrap();
        let actual = result.as_float().expect("Expected float result");
        assert!(
            (actual - expected).abs() < 1e-10,
            "Failed for {} + {} = {}, got {}",
            a,
            b,
            expected,
            actual
        );
    }
}

// =============================================================================
// Comparison Tests
// =============================================================================

#[test]
fn test_e2e_comparison_speculation() {
    let config = JitConfig::for_testing();
    let mut vm = VirtualMachine::with_jit_config(config);

    // Test less-than comparisons
    let code_true = create_lt_code(5, 10);
    let result = vm.execute(code_true).unwrap();
    assert_eq!(result.as_bool(), Some(true));

    let code_false = create_lt_code(10, 5);
    let result = vm.execute(code_false).unwrap();
    assert_eq!(result.as_bool(), Some(false));

    let code_equal = create_lt_code(5, 5);
    let result = vm.execute(code_equal).unwrap();
    assert_eq!(result.as_bool(), Some(false)); // 5 < 5 is false
}

// =============================================================================
// Multiplication Tests
// =============================================================================

#[test]
fn test_e2e_multiplication_correctness() {
    let config = JitConfig::for_testing();
    let mut vm = VirtualMachine::with_jit_config(config);

    let test_cases = [
        (0, 100, 0),
        (1, 1, 1),
        (7, 8, 56),
        (-3, 4, -12),
        (100, 100, 10000),
    ];

    for (a, b, expected) in test_cases {
        let code = create_mul_code(a, b);
        let result = vm.execute(code).unwrap();
        assert_eq!(
            result.as_int(),
            Some(expected),
            "Failed for {} * {} = {}",
            a,
            b,
            expected
        );
    }
}

// =============================================================================
// Hot Loop and OSR Tests
// =============================================================================

#[test]
fn test_e2e_hot_loop_correctness() {
    let config = JitConfig::for_testing();
    let mut vm = VirtualMachine::with_jit_config(config);

    let code = create_hot_loop_code(50);
    let result = vm.execute(code).unwrap();
    assert_eq!(result.as_int(), Some(50));
}

#[test]
fn test_e2e_hot_loop_large() {
    let config = JitConfig::for_testing();
    let mut vm = VirtualMachine::with_jit_config(config);

    let code = create_hot_loop_code(10000);
    let result = vm.execute(code).unwrap();
    assert_eq!(result.as_int(), Some(10000));
}

#[test]
fn test_e2e_repeated_execution() {
    let config = JitConfig::for_testing();
    let mut vm = VirtualMachine::with_jit_config(config);

    let code = create_int_add_code(10, 20);

    // Execute many times to exercise speculation
    for i in 0..100 {
        let result = vm.execute(Arc::clone(&code)).unwrap();
        assert_eq!(result.as_int(), Some(30), "Failed on iteration {}", i);
    }
}

#[test]
fn test_e2e_repeated_execution_past_tier2_threshold() {
    let config = JitConfig::for_testing();
    let mut vm = VirtualMachine::with_jit_config(config);

    let code = create_int_add_code(10, 20);

    // Exercise well beyond for_testing Tier2 threshold (100)
    // to verify ABI behavior remains correct after tier-up.
    for i in 0..250 {
        let result = vm.execute(Arc::clone(&code)).unwrap();
        assert_eq!(result.as_int(), Some(30), "Failed on iteration {}", i);
    }
}

// =============================================================================
// Nested Loops Test
// =============================================================================

#[test]
fn test_e2e_nested_loops() {
    let mut instructions = Vec::new();
    let mut constants = Vec::new();

    // LOAD_CONST 0 -> r0 (outer = 0)
    constants.push(Value::int(0).unwrap());
    instructions.push(Instruction::op_di(Opcode::LoadConst, Register(0), 0));

    // LOAD_CONST 0 -> r1 (i = 0)
    instructions.push(Instruction::op_di(Opcode::LoadConst, Register(1), 0));

    // LOAD_CONST 1 -> r3 (increment)
    constants.push(Value::int(1).unwrap());
    instructions.push(Instruction::op_di(Opcode::LoadConst, Register(3), 1));

    // LOAD_CONST 2 -> r4 (limit = 10)
    constants.push(Value::int(10).unwrap());
    instructions.push(Instruction::op_di(Opcode::LoadConst, Register(4), 2));

    // Outer loop header
    let outer_header = instructions.len();

    // LT r1, r4 -> r5
    instructions.push(Instruction::op_dss(
        Opcode::Lt,
        Register(5),
        Register(1),
        Register(4),
    ));

    // JUMP_IF_FALSE r5, outer_exit
    let outer_jump_idx = instructions.len();
    instructions.push(Instruction::op_di(Opcode::JumpIfFalse, Register(5), 0));

    // r2 = 0 (j = 0)
    instructions.push(Instruction::op_di(Opcode::LoadConst, Register(2), 0));

    // Inner loop header
    let inner_header = instructions.len();

    // LT r2, r4 -> r5
    instructions.push(Instruction::op_dss(
        Opcode::Lt,
        Register(5),
        Register(2),
        Register(4),
    ));

    // JUMP_IF_FALSE r5, inner_exit
    let inner_jump_idx = instructions.len();
    instructions.push(Instruction::op_di(Opcode::JumpIfFalse, Register(5), 0));

    // ADD r0, r3 -> r0 (outer += 1)
    instructions.push(Instruction::op_dss(
        Opcode::Add,
        Register(0),
        Register(0),
        Register(3),
    ));

    // ADD r2, r3 -> r2 (j += 1)
    instructions.push(Instruction::op_dss(
        Opcode::Add,
        Register(2),
        Register(2),
        Register(3),
    ));

    // JUMP inner_header
    let inner_back_idx = instructions.len();
    instructions.push(Instruction::op_di(Opcode::Jump, Register(0), 0));

    // Inner exit
    let inner_exit = instructions.len();

    // ADD r1, r3 -> r1 (i += 1)
    instructions.push(Instruction::op_dss(
        Opcode::Add,
        Register(1),
        Register(1),
        Register(3),
    ));

    // JUMP outer_header
    let outer_back_idx = instructions.len();
    instructions.push(Instruction::op_di(Opcode::Jump, Register(0), 0));

    // Outer exit
    let outer_exit = instructions.len();

    // RETURN r0
    instructions.push(Instruction::op_d(Opcode::Return, Register(0)));

    // Fixup jumps
    let inner_back_offset = (inner_header as i32) - (inner_back_idx as i32 + 1);
    instructions[inner_back_idx] =
        Instruction::op_di(Opcode::Jump, Register(0), inner_back_offset as i16 as u16);

    let inner_exit_offset = (inner_exit as i32) - (inner_jump_idx as i32 + 1);
    instructions[inner_jump_idx] = Instruction::op_di(
        Opcode::JumpIfFalse,
        Register(5),
        inner_exit_offset as i16 as u16,
    );

    let outer_back_offset = (outer_header as i32) - (outer_back_idx as i32 + 1);
    instructions[outer_back_idx] =
        Instruction::op_di(Opcode::Jump, Register(0), outer_back_offset as i16 as u16);

    let outer_exit_offset = (outer_exit as i32) - (outer_jump_idx as i32 + 1);
    instructions[outer_jump_idx] = Instruction::op_di(
        Opcode::JumpIfFalse,
        Register(5),
        outer_exit_offset as i16 as u16,
    );

    let code = Arc::new(CodeObject {
        name: "test_nested_loops".into(),
        register_count: 6,
        arg_count: 0,
        posonlyarg_count: 0,
        kwonlyarg_count: 0,
        instructions: instructions.into_boxed_slice(),
        constants: constants.into_boxed_slice(),
        names: Box::new([]),
        locals: Box::new([]),
        freevars: Box::new([]),
        cellvars: Box::new([]),
        line_table: Box::new([]),
        exception_table: Box::new([]),
        filename: "test.py".into(),
        qualname: "test_nested_loops".into(),
        flags: CodeFlags::NONE,
        first_lineno: 1,
        nested_code_objects: Box::new([]),
    });

    let config = JitConfig::for_testing();
    let mut vm = VirtualMachine::with_jit_config(config);

    let result = vm.execute(code).unwrap();
    assert_eq!(result.as_int(), Some(100)); // 10 * 10 = 100
}

// =============================================================================
// Edge Case Tests
// =============================================================================

#[test]
fn test_e2e_zero_iteration_loop() {
    let config = JitConfig::for_testing();
    let mut vm = VirtualMachine::with_jit_config(config);

    let code = create_hot_loop_code(0);
    let result = vm.execute(code).unwrap();
    assert_eq!(result.as_int(), Some(0));
}

#[test]
fn test_e2e_single_iteration_loop() {
    let config = JitConfig::for_testing();
    let mut vm = VirtualMachine::with_jit_config(config);

    let code = create_hot_loop_code(1);
    let result = vm.execute(code).unwrap();
    assert_eq!(result.as_int(), Some(1));
}

#[test]
fn test_e2e_negative_addition() {
    let config = JitConfig::for_testing();
    let mut vm = VirtualMachine::with_jit_config(config);

    let code = create_int_add_code(-50, -50);
    let result = vm.execute(code).unwrap();
    assert_eq!(result.as_int(), Some(-100));
}

#[test]
fn test_e2e_float_special_values() {
    let config = JitConfig::for_testing();
    let mut vm = VirtualMachine::with_jit_config(config);

    // Test with very small values
    let code = create_float_add_code(1e-300, 1e-300);
    let result = vm.execute(code).unwrap();
    let actual = result.as_float().expect("Expected float result");
    assert!((actual - 2e-300).abs() < 1e-310);

    // Test with very large values
    let code = create_float_add_code(1e100, 1e100);
    let result = vm.execute(code).unwrap();
    let actual = result.as_float().expect("Expected float result");
    assert!((actual - 2e100).abs() / 2e100 < 1e-10);
}
