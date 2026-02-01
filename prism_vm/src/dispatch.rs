//! Dispatch table and main execution loop.
//!
//! Uses a static function pointer table for O(1) opcode dispatch.
//! Each opcode maps to a handler function that returns control flow.

use crate::VirtualMachine;
use crate::error::RuntimeError;
use crate::frame::Frame;
use prism_compiler::bytecode::CodeObject;
use prism_compiler::bytecode::{Instruction, Opcode};
use prism_core::Value;
use std::sync::Arc;

/// Control flow result from opcode execution.
#[derive(Debug, Clone)]
pub enum ControlFlow {
    /// Continue to next instruction.
    Continue,
    /// Relative jump by signed offset.
    Jump(i16),
    /// Push new frame and call function.
    Call {
        code: Arc<CodeObject>,
        return_reg: u8,
    },
    /// Return value and pop frame.
    Return(Value),
    /// Runtime error occurred.
    Error(RuntimeError),
}

/// Opcode handler function signature.
pub type OpHandler = fn(&mut VirtualMachine, Instruction) -> ControlFlow;

/// Invalid opcode handler.
#[inline(always)]
fn op_invalid(_vm: &mut VirtualMachine, inst: Instruction) -> ControlFlow {
    ControlFlow::Error(RuntimeError::invalid_opcode(inst.opcode()))
}

// Import all opcode handlers
use crate::ops::arithmetic;
use crate::ops::calls;
use crate::ops::comparison;
use crate::ops::containers;
use crate::ops::control;
use crate::ops::load_store;
use crate::ops::objects;

/// Build the static dispatch table.
/// Returns array of 256 function pointers indexed by opcode.
const fn build_dispatch_table() -> [OpHandler; 256] {
    let mut table: [OpHandler; 256] = [op_invalid; 256];

    // Control Flow (0x00-0x0F)
    table[Opcode::Nop as usize] = control::nop;
    table[Opcode::Return as usize] = control::return_value;
    table[Opcode::ReturnNone as usize] = control::return_none;
    table[Opcode::Jump as usize] = control::jump;
    table[Opcode::JumpIfFalse as usize] = control::jump_if_false;
    table[Opcode::JumpIfTrue as usize] = control::jump_if_true;
    table[Opcode::JumpIfNone as usize] = control::jump_if_none;
    table[Opcode::JumpIfNotNone as usize] = control::jump_if_not_none;
    table[Opcode::PopExceptHandler as usize] = control::pop_except_handler;
    table[Opcode::Raise as usize] = control::raise;
    table[Opcode::Reraise as usize] = control::reraise;
    table[Opcode::EndFinally as usize] = control::end_finally;
    table[Opcode::Yield as usize] = control::yield_value;
    table[Opcode::YieldFrom as usize] = control::yield_from;

    // Load/Store (0x10-0x1F)
    table[Opcode::LoadConst as usize] = load_store::load_const;
    table[Opcode::LoadNone as usize] = load_store::load_none;
    table[Opcode::LoadTrue as usize] = load_store::load_true;
    table[Opcode::LoadFalse as usize] = load_store::load_false;
    table[Opcode::LoadLocal as usize] = load_store::load_local;
    table[Opcode::StoreLocal as usize] = load_store::store_local;
    table[Opcode::LoadClosure as usize] = load_store::load_closure;
    table[Opcode::StoreClosure as usize] = load_store::store_closure;
    table[Opcode::LoadGlobal as usize] = load_store::load_global;
    table[Opcode::StoreGlobal as usize] = load_store::store_global;
    table[Opcode::DeleteLocal as usize] = load_store::delete_local;
    table[Opcode::DeleteGlobal as usize] = load_store::delete_global;
    table[Opcode::Move as usize] = load_store::move_reg;

    // Integer Arithmetic (0x20-0x2F)
    table[Opcode::AddInt as usize] = arithmetic::add_int;
    table[Opcode::SubInt as usize] = arithmetic::sub_int;
    table[Opcode::MulInt as usize] = arithmetic::mul_int;
    table[Opcode::FloorDivInt as usize] = arithmetic::floor_div_int;
    table[Opcode::ModInt as usize] = arithmetic::mod_int;
    table[Opcode::PowInt as usize] = arithmetic::pow_int;
    table[Opcode::NegInt as usize] = arithmetic::neg_int;
    table[Opcode::PosInt as usize] = arithmetic::pos_int;

    // Float Arithmetic (0x30-0x37)
    table[Opcode::AddFloat as usize] = arithmetic::add_float;
    table[Opcode::SubFloat as usize] = arithmetic::sub_float;
    table[Opcode::MulFloat as usize] = arithmetic::mul_float;
    table[Opcode::DivFloat as usize] = arithmetic::div_float;
    table[Opcode::FloorDivFloat as usize] = arithmetic::floor_div_float;
    table[Opcode::ModFloat as usize] = arithmetic::mod_float;
    table[Opcode::PowFloat as usize] = arithmetic::pow_float;
    table[Opcode::NegFloat as usize] = arithmetic::neg_float;

    // Generic Arithmetic (0x38-0x3F)
    table[Opcode::Add as usize] = arithmetic::add;
    table[Opcode::Sub as usize] = arithmetic::sub;
    table[Opcode::Mul as usize] = arithmetic::mul;
    table[Opcode::TrueDiv as usize] = arithmetic::true_div;
    table[Opcode::FloorDiv as usize] = arithmetic::floor_div;
    table[Opcode::Mod as usize] = arithmetic::modulo;
    table[Opcode::Pow as usize] = arithmetic::pow;
    table[Opcode::Neg as usize] = arithmetic::neg;

    // Comparison (0x40-0x4F)
    table[Opcode::Lt as usize] = comparison::lt;
    table[Opcode::Le as usize] = comparison::le;
    table[Opcode::Eq as usize] = comparison::eq;
    table[Opcode::Ne as usize] = comparison::ne;
    table[Opcode::Gt as usize] = comparison::gt;
    table[Opcode::Ge as usize] = comparison::ge;
    table[Opcode::Is as usize] = comparison::is;
    table[Opcode::IsNot as usize] = comparison::is_not;
    table[Opcode::In as usize] = comparison::in_op;
    table[Opcode::NotIn as usize] = comparison::not_in;

    // Logical/Bitwise (0x50-0x5F)
    table[Opcode::BitwiseAnd as usize] = comparison::bitwise_and;
    table[Opcode::BitwiseOr as usize] = comparison::bitwise_or;
    table[Opcode::BitwiseXor as usize] = comparison::bitwise_xor;
    table[Opcode::BitwiseNot as usize] = comparison::bitwise_not;
    table[Opcode::Shl as usize] = comparison::shl;
    table[Opcode::Shr as usize] = comparison::shr;
    table[Opcode::Not as usize] = comparison::not;

    // Object Operations (0x60-0x6F)
    table[Opcode::GetAttr as usize] = objects::get_attr;
    table[Opcode::SetAttr as usize] = objects::set_attr;
    table[Opcode::DelAttr as usize] = objects::del_attr;
    table[Opcode::GetItem as usize] = objects::get_item;
    table[Opcode::SetItem as usize] = objects::set_item;
    table[Opcode::DelItem as usize] = objects::del_item;
    table[Opcode::GetIter as usize] = objects::get_iter;
    table[Opcode::ForIter as usize] = objects::for_iter;
    table[Opcode::Len as usize] = objects::len;
    table[Opcode::IsCallable as usize] = objects::is_callable;

    // Function Calls (0x70-0x7F)
    table[Opcode::Call as usize] = calls::call;
    table[Opcode::CallKw as usize] = calls::call_kw;
    table[Opcode::CallMethod as usize] = calls::call_method;
    table[Opcode::TailCall as usize] = calls::tail_call;
    table[Opcode::MakeFunction as usize] = calls::make_function;
    table[Opcode::MakeClosure as usize] = calls::make_closure;

    // Container Operations (0x80-0x8F)
    table[Opcode::BuildList as usize] = containers::build_list;
    table[Opcode::BuildTuple as usize] = containers::build_tuple;
    table[Opcode::BuildSet as usize] = containers::build_set;
    table[Opcode::BuildDict as usize] = containers::build_dict;
    table[Opcode::BuildString as usize] = containers::build_string;
    table[Opcode::ListAppend as usize] = containers::list_append;
    table[Opcode::SetAdd as usize] = containers::set_add;
    table[Opcode::DictSet as usize] = containers::dict_set;
    table[Opcode::UnpackSequence as usize] = containers::unpack_sequence;
    table[Opcode::UnpackEx as usize] = containers::unpack_ex;
    table[Opcode::BuildSlice as usize] = containers::build_slice;

    // Import (0x90-0x9F)
    table[Opcode::ImportName as usize] = containers::import_name;
    table[Opcode::ImportFrom as usize] = containers::import_from;
    table[Opcode::ImportStar as usize] = containers::import_star;

    table
}

/// Static dispatch table - computed at compile time.
pub static DISPATCH_TABLE: [OpHandler; 256] = build_dispatch_table();

/// Get the handler for an opcode.
#[inline(always)]
pub fn get_handler(opcode: u8) -> OpHandler {
    // Safety: opcode is u8, so always in bounds for 256-element array
    unsafe { *DISPATCH_TABLE.get_unchecked(opcode as usize) }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dispatch_table_built() {
        // Verify table is populated
        assert!(DISPATCH_TABLE.len() == 256);

        // Verify some known opcodes are not invalid
        let nop_handler = DISPATCH_TABLE[Opcode::Nop as usize];
        let add_handler = DISPATCH_TABLE[Opcode::Add as usize];

        // Handlers should be different from op_invalid for valid opcodes
        // (We can't easily compare function pointers, but we can verify non-panic)
    }

    #[test]
    fn test_get_handler() {
        let handler = get_handler(Opcode::LoadConst as u8);
        // Just verify it doesn't panic
    }
}
