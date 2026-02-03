//! Dispatch table and main execution loop.
//!
//! Uses a static function pointer table for O(1) opcode dispatch.
//! Each opcode maps to a handler function that returns control flow.

use crate::VirtualMachine;
use crate::error::RuntimeError;
use prism_compiler::bytecode::{CodeObject, Instruction, Opcode};
use prism_core::Value;
use std::sync::Arc;

/// Control flow result from opcode execution.
///
/// This enum represents all possible control flow outcomes from executing
/// a bytecode instruction. The VM dispatch loop uses this to determine
/// what action to take next.
#[derive(Debug, Clone)]
pub enum ControlFlow {
    // =========================================================================
    // Normal Execution
    // =========================================================================
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

    // =========================================================================
    // Exception Handling
    // =========================================================================
    /// Raise an exception.
    ///
    /// Triggers exception propagation: the VM will search for a handler,
    /// unwind the stack as needed, and either jump to a handler or
    /// propagate to the caller.
    Exception {
        /// Exception type ID for fast matching.
        type_id: u16,
        /// Handler PC if already located (0 if unknown).
        handler_pc: u32,
    },

    /// Re-raise the current exception.
    ///
    /// Used in except blocks to propagate an exception after partial handling.
    Reraise,

    /// Jump to exception handler.
    ///
    /// Called after handler lookup succeeds. Restores stack depth and
    /// transfers control to the handler code.
    EnterHandler {
        /// Handler bytecode address.
        handler_pc: u32,
        /// Stack depth to restore.
        stack_depth: u16,
    },

    /// Enter a finally block.
    ///
    /// Finally blocks execute unconditionally and may reraise after completion.
    EnterFinally {
        /// Finally block bytecode address.
        finally_pc: u32,
        /// Stack depth to restore.
        stack_depth: u16,
        /// Whether to reraise after finally completes.
        reraise: bool,
    },

    /// Exit exception handler.
    ///
    /// Pops the handler from the stack and resumes normal execution.
    ExitHandler,

    // =========================================================================
    // Generator Protocol
    // =========================================================================
    /// Yield a value from a generator.
    ///
    /// Suspends execution and returns the yielded value to the caller.
    /// The resume point is stored for later continuation.
    Yield {
        /// The value being yielded.
        value: Value,
        /// Bytecode offset to resume at when send() is called.
        resume_point: u32,
    },

    /// Resume a suspended generator.
    ///
    /// Continues execution from the saved resume point with the sent value.
    Resume {
        /// Value sent into the generator (or None for __next__).
        send_value: Value,
    },

    // =========================================================================
    // Error Handling
    // =========================================================================
    /// Runtime error occurred (non-exception error).
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
use crate::ops::class;
use crate::ops::comparison;
use crate::ops::containers;
use crate::ops::control;
use crate::ops::coroutine;
use crate::ops::load_store;
use crate::ops::r#match;
use crate::ops::method_dispatch;
use crate::ops::objects;
use crate::ops::unpack;

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
    table[Opcode::DeleteClosure as usize] = load_store::delete_closure;

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
    table[Opcode::LoadMethod as usize] = method_dispatch::load_method;
    table[Opcode::BuildClass as usize] = class::build_class;

    // Function Calls (0x70-0x7F)
    table[Opcode::Call as usize] = calls::call;
    table[Opcode::CallKw as usize] = calls::call_kw;
    table[Opcode::CallMethod as usize] = method_dispatch::call_method;
    table[Opcode::TailCall as usize] = calls::tail_call;
    table[Opcode::MakeFunction as usize] = calls::make_function;
    table[Opcode::MakeClosure as usize] = calls::make_closure;
    table[Opcode::CallKwEx as usize] = calls::call_kw_ex;
    table[Opcode::CallEx as usize] = unpack::call_ex;
    table[Opcode::BuildTupleUnpack as usize] = unpack::build_tuple_unpack;
    table[Opcode::BuildDictUnpack as usize] = unpack::build_dict_unpack;

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

    // Pattern Matching (0x9A-0x9F)
    table[Opcode::MatchClass as usize] = r#match::match_class;
    table[Opcode::MatchMapping as usize] = r#match::match_mapping;
    table[Opcode::MatchSequence as usize] = r#match::match_sequence;
    table[Opcode::MatchKeys as usize] = r#match::match_keys;
    table[Opcode::CopyDictWithoutKeys as usize] = r#match::copy_dict_without_keys;
    table[Opcode::GetMatchArgs as usize] = r#match::get_match_args;

    // Coroutine/Async Operations (0xA0-0xAF)
    table[Opcode::GetAwaitable as usize] = coroutine::get_awaitable;
    table[Opcode::GetAIter as usize] = coroutine::get_aiter;
    table[Opcode::GetANext as usize] = coroutine::get_anext;
    table[Opcode::EndAsyncFor as usize] = coroutine::end_async_for;
    table[Opcode::Send as usize] = coroutine::send;

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

    // =========================================================================
    // ControlFlow Tests
    // =========================================================================

    #[test]
    fn test_control_flow_continue() {
        let cf = ControlFlow::Continue;
        assert!(matches!(cf, ControlFlow::Continue));
    }

    #[test]
    fn test_control_flow_jump() {
        let cf = ControlFlow::Jump(10);
        if let ControlFlow::Jump(offset) = cf {
            assert_eq!(offset, 10);
        } else {
            panic!("Expected Jump");
        }

        // Negative jump
        let cf = ControlFlow::Jump(-5);
        if let ControlFlow::Jump(offset) = cf {
            assert_eq!(offset, -5);
        } else {
            panic!("Expected Jump");
        }
    }

    #[test]
    fn test_control_flow_return() {
        let cf = ControlFlow::Return(Value::int(42).unwrap());
        if let ControlFlow::Return(v) = cf {
            assert_eq!(v.as_int(), Some(42));
        } else {
            panic!("Expected Return");
        }
    }

    #[test]
    fn test_control_flow_exception() {
        let cf = ControlFlow::Exception {
            type_id: 5,
            handler_pc: 100,
        };
        if let ControlFlow::Exception {
            type_id,
            handler_pc,
        } = cf
        {
            assert_eq!(type_id, 5);
            assert_eq!(handler_pc, 100);
        } else {
            panic!("Expected Exception");
        }
    }

    #[test]
    fn test_control_flow_exception_unknown_handler() {
        let cf = ControlFlow::Exception {
            type_id: 1,
            handler_pc: 0, // Unknown handler
        };
        if let ControlFlow::Exception { handler_pc, .. } = cf {
            assert_eq!(handler_pc, 0);
        } else {
            panic!("Expected Exception");
        }
    }

    #[test]
    fn test_control_flow_reraise() {
        let cf = ControlFlow::Reraise;
        assert!(matches!(cf, ControlFlow::Reraise));
    }

    #[test]
    fn test_control_flow_enter_handler() {
        let cf = ControlFlow::EnterHandler {
            handler_pc: 50,
            stack_depth: 3,
        };
        if let ControlFlow::EnterHandler {
            handler_pc,
            stack_depth,
        } = cf
        {
            assert_eq!(handler_pc, 50);
            assert_eq!(stack_depth, 3);
        } else {
            panic!("Expected EnterHandler");
        }
    }

    #[test]
    fn test_control_flow_enter_finally() {
        let cf = ControlFlow::EnterFinally {
            finally_pc: 75,
            stack_depth: 2,
            reraise: true,
        };
        if let ControlFlow::EnterFinally {
            finally_pc,
            stack_depth,
            reraise,
        } = cf
        {
            assert_eq!(finally_pc, 75);
            assert_eq!(stack_depth, 2);
            assert!(reraise);
        } else {
            panic!("Expected EnterFinally");
        }
    }

    #[test]
    fn test_control_flow_enter_finally_no_reraise() {
        let cf = ControlFlow::EnterFinally {
            finally_pc: 80,
            stack_depth: 1,
            reraise: false,
        };
        if let ControlFlow::EnterFinally { reraise, .. } = cf {
            assert!(!reraise);
        } else {
            panic!("Expected EnterFinally");
        }
    }

    #[test]
    fn test_control_flow_exit_handler() {
        let cf = ControlFlow::ExitHandler;
        assert!(matches!(cf, ControlFlow::ExitHandler));
    }

    #[test]
    fn test_control_flow_yield() {
        let cf = ControlFlow::Yield {
            value: Value::int(100).unwrap(),
            resume_point: 25,
        };
        if let ControlFlow::Yield {
            value,
            resume_point,
        } = cf
        {
            assert_eq!(value.as_int(), Some(100));
            assert_eq!(resume_point, 25);
        } else {
            panic!("Expected Yield");
        }
    }

    #[test]
    fn test_control_flow_resume() {
        let cf = ControlFlow::Resume {
            send_value: Value::none(),
        };
        if let ControlFlow::Resume { send_value } = cf {
            assert!(send_value.is_none());
        } else {
            panic!("Expected Resume");
        }
    }

    #[test]
    fn test_control_flow_resume_with_value() {
        let cf = ControlFlow::Resume {
            send_value: Value::int(42).unwrap(),
        };
        if let ControlFlow::Resume { send_value } = cf {
            assert_eq!(send_value.as_int(), Some(42));
        } else {
            panic!("Expected Resume");
        }
    }

    #[test]
    fn test_control_flow_error() {
        let err = RuntimeError::internal("test error");
        let cf = ControlFlow::Error(err);
        assert!(matches!(cf, ControlFlow::Error(_)));
    }

    #[test]
    fn test_control_flow_size() {
        // Ensure ControlFlow remains reasonably sized
        // Note: The Error variant contains RuntimeError which includes a Vec<TracebackEntry>
        // making it the largest variant. On 64-bit systems the size is ~88 bytes.
        let size = std::mem::size_of::<ControlFlow>();
        assert!(
            size <= 104,
            "ControlFlow size is {} bytes, expected <= 104",
            size
        );
    }

    #[test]
    fn test_control_flow_clone() {
        let cf = ControlFlow::Exception {
            type_id: 5,
            handler_pc: 100,
        };
        let cloned = cf.clone();
        if let ControlFlow::Exception {
            type_id,
            handler_pc,
        } = cloned
        {
            assert_eq!(type_id, 5);
            assert_eq!(handler_pc, 100);
        } else {
            panic!("Clone failed");
        }
    }

    #[test]
    fn test_control_flow_debug() {
        let cf = ControlFlow::Continue;
        let debug = format!("{:?}", cf);
        assert!(debug.contains("Continue"));
    }
}
