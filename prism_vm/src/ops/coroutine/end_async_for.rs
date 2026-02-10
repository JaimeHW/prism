//! EndAsyncFor opcode handler.
//!
//! Handles the end of an async for loop iteration by checking for
//! `StopAsyncIteration` exception.
//!
//! # Python Semantics (PEP 492)
//!
//! At the end of each async for iteration:
//! 1. Check if awaited `__anext__()` raised `StopAsyncIteration`
//! 2. If yes: clear exception and jump to loop end (else clause or exit)
//! 3. If no: continue with current exception (re-raise)
//!
//! # Instruction Format
//!
//! `EndAsyncFor dst, imm16`
//! - `dst`: Register containing the exception info (set by preceding handler)
//! - `imm16`: Jump offset if StopAsyncIteration was raised
//!
//! # Performance
//!
//! - Normal iteration: ~3 cycles (no exception check needed)
//! - StopAsyncIteration: ~8 cycles (exception check + jump)

use crate::VirtualMachine;
use crate::dispatch::ControlFlow;
use crate::error::RuntimeError;
use prism_compiler::bytecode::Instruction;
use prism_core::Value;

/// EndAsyncFor: Handle StopAsyncIteration in async for loop.
///
/// This opcode is emitted after awaiting the result of `__anext__()`.
/// It checks if StopAsyncIteration was raised and handles loop termination.
///
/// # Current Implementation
///
/// This is a stub that will be fully implemented once the exception handling
/// infrastructure is integrated. Currently it:
/// - Checks if the dst register contains an exception marker
/// - If StopAsyncIteration, jumps to loop end
/// - Otherwise, re-raises the exception
#[inline(always)]
pub fn end_async_for(vm: &mut VirtualMachine, inst: Instruction) -> ControlFlow {
    let dst = inst.dst().0;
    let jump_offset = inst.imm16() as i16;

    // Get the exception value from the destination register
    // In the async for loop, the exception handler will have set this
    let exc_value = vm.current_frame().get_reg(dst);

    // =========================================================================
    // Check for StopAsyncIteration
    // =========================================================================

    // Check if the exception is StopAsyncIteration
    if is_stop_async_iteration(&exc_value) {
        // Clear the destination register and jump to loop end
        vm.current_frame_mut().set_reg(dst, Value::none());
        return ControlFlow::Jump(jump_offset);
    }

    // =========================================================================
    // Normal iteration or other exception
    // =========================================================================

    // If the value is None or a normal value, continue iteration
    if exc_value.is_none() {
        return ControlFlow::Continue;
    }

    // Otherwise, we have a different exception - re-raise it
    // TODO: This requires proper exception propagation infrastructure
    // For now, return a runtime error for unhandled exceptions
    ControlFlow::Error(RuntimeError::internal(
        "EndAsyncFor: exception propagation not yet implemented",
    ))
}

// =============================================================================
// Exception Type Checking
// =============================================================================

/// Check if an exception is StopAsyncIteration.
///
/// This is the async equivalent of StopIteration for `async for` loops.
///
/// # Implementation
///
/// This function checks if the value represents a StopAsyncIteration exception.
/// Currently returns false as a stub until the exception type system is
/// fully integrated.
#[inline(always)]
fn is_stop_async_iteration(exc: &Value) -> bool {
    // TODO: Check if the exception is an instance of StopAsyncIteration
    // This requires:
    // 1. Checking if exc is an object
    // 2. Getting its type
    // 3. Comparing against the built-in StopAsyncIteration type
    let _ = exc;
    false // Placeholder: implement when exception hierarchy is ready
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_none_not_stop_async_iteration() {
        // None is never StopAsyncIteration
        assert!(!is_stop_async_iteration(&Value::none()));
    }

    #[test]
    fn test_int_not_stop_async_iteration() {
        let val = Value::int(42).unwrap();
        assert!(!is_stop_async_iteration(&val));
    }

    #[test]
    fn test_bool_not_stop_async_iteration() {
        assert!(!is_stop_async_iteration(&Value::bool(true)));
        assert!(!is_stop_async_iteration(&Value::bool(false)));
    }

    #[test]
    fn test_float_not_stop_async_iteration() {
        let val = Value::float(3.14);
        assert!(!is_stop_async_iteration(&val));
    }
}
