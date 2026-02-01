//! Shared test utilities for JIT integration tests.

use prism_compiler::bytecode::{CodeFlags, CodeObject, Instruction, Opcode, Register};
use prism_core::Value;
use prism_vm::{JitConfig, VirtualMachine};
use std::sync::Arc;

/// Helper to create a basic code object with given instructions and constants.
pub fn create_code_object(
    name: &str,
    instructions: Vec<Instruction>,
    constants: Vec<Value>,
    register_count: u16,
    arg_count: u16,
) -> Arc<CodeObject> {
    Arc::new(CodeObject {
        name: name.into(),
        register_count,
        arg_count,
        posonlyarg_count: 0,
        kwonlyarg_count: 0,
        instructions: instructions.into_boxed_slice(),
        constants: constants.into_boxed_slice(),
        names: Box::new([]),
        locals: Box::new([]),
        freevars: Box::new([]),
        cellvars: Box::new([]),
        line_table: Box::new([]),
        filename: "jit_test.py".into(),
        qualname: name.into(),
        flags: CodeFlags::NONE,
        first_lineno: 1,
    })
}

/// Create a VM configured for JIT testing with low thresholds.
pub fn create_test_vm() -> VirtualMachine {
    VirtualMachine::with_jit_config(JitConfig::for_testing())
}

/// Execute code multiple times to trigger tier-up.
pub fn execute_with_tierup(
    vm: &mut VirtualMachine,
    code: Arc<CodeObject>,
    iterations: usize,
) -> Value {
    let mut result = Value::none();
    for _ in 0..iterations {
        result = vm.execute(Arc::clone(&code)).expect("Execution failed");
    }
    result
}

// =============================================================================
// Instruction Builders
// =============================================================================

/// Load a constant into a register.
pub fn load_const(dest: u8, const_idx: u16) -> Instruction {
    Instruction::op_di(Opcode::LoadConst, Register(dest), const_idx)
}

/// Return the value in a register.
pub fn return_reg(src: u8) -> Instruction {
    Instruction::op_d(Opcode::Return, Register(src))
}

/// Add two registers, storing in dest.
pub fn add(dest: u8, src1: u8, src2: u8) -> Instruction {
    Instruction::op_dss(Opcode::Add, Register(dest), Register(src1), Register(src2))
}

/// Subtract two registers, storing in dest.
pub fn sub(dest: u8, src1: u8, src2: u8) -> Instruction {
    Instruction::op_dss(Opcode::Sub, Register(dest), Register(src1), Register(src2))
}

/// Multiply two registers, storing in dest.
pub fn mul(dest: u8, src1: u8, src2: u8) -> Instruction {
    Instruction::op_dss(Opcode::Mul, Register(dest), Register(src1), Register(src2))
}

/// Integer division of two registers, storing in dest.
pub fn floor_div(dest: u8, src1: u8, src2: u8) -> Instruction {
    Instruction::op_dss(
        Opcode::FloorDiv,
        Register(dest),
        Register(src1),
        Register(src2),
    )
}

/// True division of two registers, storing in dest.
pub fn true_div(dest: u8, src1: u8, src2: u8) -> Instruction {
    Instruction::op_dss(
        Opcode::TrueDiv,
        Register(dest),
        Register(src1),
        Register(src2),
    )
}

/// Modulo of two registers, storing in dest.
pub fn modulo(dest: u8, src1: u8, src2: u8) -> Instruction {
    Instruction::op_dss(Opcode::Mod, Register(dest), Register(src1), Register(src2))
}

/// Less-than comparison.
pub fn lt(dest: u8, src1: u8, src2: u8) -> Instruction {
    Instruction::op_dss(Opcode::Lt, Register(dest), Register(src1), Register(src2))
}

/// Less-than-or-equal comparison.
pub fn le(dest: u8, src1: u8, src2: u8) -> Instruction {
    Instruction::op_dss(Opcode::Le, Register(dest), Register(src1), Register(src2))
}

/// Greater-than comparison.
pub fn gt(dest: u8, src1: u8, src2: u8) -> Instruction {
    Instruction::op_dss(Opcode::Gt, Register(dest), Register(src1), Register(src2))
}

/// Greater-than-or-equal comparison.
pub fn ge(dest: u8, src1: u8, src2: u8) -> Instruction {
    Instruction::op_dss(Opcode::Ge, Register(dest), Register(src1), Register(src2))
}

/// Equal comparison.
pub fn eq(dest: u8, src1: u8, src2: u8) -> Instruction {
    Instruction::op_dss(Opcode::Eq, Register(dest), Register(src1), Register(src2))
}

/// Not-equal comparison.
pub fn ne(dest: u8, src1: u8, src2: u8) -> Instruction {
    Instruction::op_dss(Opcode::Ne, Register(dest), Register(src1), Register(src2))
}

/// Unconditional jump with relative offset.
pub fn jump(offset: i16) -> Instruction {
    Instruction::op_di(Opcode::Jump, Register(0), offset as u16)
}

/// Conditional jump if register is false.
pub fn jump_if_false(cond: u8, offset: i16) -> Instruction {
    Instruction::op_di(Opcode::JumpIfFalse, Register(cond), offset as u16)
}

/// Conditional jump if register is true.
pub fn jump_if_true(cond: u8, offset: i16) -> Instruction {
    Instruction::op_di(Opcode::JumpIfTrue, Register(cond), offset as u16)
}

/// Move/copy register value.
pub fn move_reg(dest: u8, src: u8) -> Instruction {
    Instruction::op_ds(Opcode::Move, Register(dest), Register(src))
}
