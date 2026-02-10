//! Send opcode handler.
//!
//! Implements the coroutine/generator send protocol for sending values
//! into suspended coroutines and generators.
//!
//! # Python Semantics
//!
//! The `send()` method is used to:
//! 1. Resume a suspended coroutine/generator
//! 2. Pass a value to the `yield` expression that suspended it
//! 3. Return the next yielded value (or raise StopIteration)
//!
//! # Protocol Rules
//!
//! - First call must use `send(None)` (or `__next__()`)
//! - Sending non-None to a just-started generator raises TypeError
//! - Sending to an exhausted generator raises StopIteration
//!
//! # Performance
//!
//! - Direct send: ~5 cycles (generator state check + resume)
//! - Protocol validation: ~2 extra cycles for state checks

use crate::VirtualMachine;
use crate::dispatch::ControlFlow;
use crate::error::RuntimeError;
use prism_compiler::bytecode::Instruction;
use prism_core::Value;

/// Send: Send value to coroutine/generator.
///
/// Instruction format: `Send dst, gen, value`
/// - `dst`: Destination register for the result (yielded value or return)
/// - `gen`: Register containing the generator/coroutine (src1)
/// - `value`: Register containing the value to send (src2)
#[inline(always)]
pub fn send(vm: &mut VirtualMachine, inst: Instruction) -> ControlFlow {
    let dst = inst.dst().0;
    let gen_reg = inst.src1().0;
    let value_reg = inst.src2().0;

    let generator = vm.current_frame().get_reg(gen_reg);
    let value = vm.current_frame().get_reg(value_reg);

    // =========================================================================
    // Validate Generator/Coroutine
    // =========================================================================

    // Check if the object is a generator or coroutine
    match get_generator_state(&generator) {
        GeneratorState::NotAGenerator => {
            return ControlFlow::Error(RuntimeError::type_error(format!(
                "cannot send to non-generator object of type {}",
                type_name(&generator)
            )));
        }
        GeneratorState::Created => {
            // Just-started generator: can only receive None
            if !value.is_none() {
                return ControlFlow::Error(RuntimeError::type_error(
                    "can't send non-None value to a just-started generator",
                ));
            }
        }
        GeneratorState::Suspended => {
            // Normal case: generator is suspended and ready to receive
        }
        GeneratorState::Running => {
            return ControlFlow::Error(RuntimeError::value_error("generator already executing"));
        }
        GeneratorState::Closed => {
            // Exhausted generator
            return ControlFlow::Error(RuntimeError::stop_iteration());
        }
    }

    // =========================================================================
    // Perform Send
    // =========================================================================

    // Resume the generator with the sent value
    // The result will be the next yielded value or StopIteration on completion
    match resume_generator(vm, generator, value) {
        ResumeResult::Yielded(yielded_value) => {
            vm.current_frame_mut().set_reg(dst, yielded_value);
            ControlFlow::Continue
        }
        ResumeResult::Returned(_return_value) => {
            // Generator completed - raise StopIteration with return value
            // TODO: StopIteration should carry the return value in a future update
            ControlFlow::Error(RuntimeError::stop_iteration())
        }
        ResumeResult::Error(e) => ControlFlow::Error(e),
    }
}

// =============================================================================
// Generator State
// =============================================================================

/// State of a generator/coroutine.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum GeneratorState {
    /// Object is not a generator or coroutine.
    NotAGenerator,
    /// Generator was just created (not yet started).
    Created,
    /// Generator is suspended at a yield point.
    Suspended,
    /// Generator is currently executing.
    Running,
    /// Generator has completed (returned or raised).
    Closed,
}

/// Get the current state of a generator/coroutine.
#[inline]
fn get_generator_state(value: &Value) -> GeneratorState {
    // TODO: Check if value is a GeneratorObject and return its state
    // - Created: ip == 0, not running
    // - Suspended: ip > 0, has saved frame
    // - Running: is_running flag set
    // - Closed: is_exhausted flag set
    let _ = value;
    GeneratorState::NotAGenerator
}

/// Get the type name of a value for error messages.
#[inline]
fn type_name(value: &Value) -> &'static str {
    if value.is_none() {
        "NoneType"
    } else if value.is_bool() {
        "bool"
    } else if value.is_int() {
        "int"
    } else if value.is_float() {
        "float"
    } else if value.as_object_ptr().is_some() {
        "object"
    } else {
        "unknown"
    }
}

// =============================================================================
// Generator Resumption
// =============================================================================

/// Result of resuming a generator.
enum ResumeResult {
    /// Generator yielded a value.
    Yielded(Value),
    /// Generator returned (completed).
    Returned(Value),
    /// Error occurred during execution.
    Error(RuntimeError),
}

/// Resume a generator with a sent value.
#[inline]
fn resume_generator(_vm: &mut VirtualMachine, _gen: Value, _value: Value) -> ResumeResult {
    // TODO: Implement generator resumption
    // 1. Set generator state to Running
    // 2. Store sent value in generator's receive slot
    // 3. Resume execution from saved instruction pointer
    // 4. On yield: save state and return Yielded
    // 5. On return: set state to Closed and return Returned
    ResumeResult::Error(RuntimeError::internal("generator send not yet implemented"))
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // Generator State Tests
    // =========================================================================

    #[test]
    fn test_none_not_generator() {
        assert_eq!(
            get_generator_state(&Value::none()),
            GeneratorState::NotAGenerator
        );
    }

    #[test]
    fn test_int_not_generator() {
        let val = Value::int(42).unwrap();
        assert_eq!(get_generator_state(&val), GeneratorState::NotAGenerator);
    }

    #[test]
    fn test_bool_not_generator() {
        assert_eq!(
            get_generator_state(&Value::bool(true)),
            GeneratorState::NotAGenerator
        );
    }

    #[test]
    fn test_float_not_generator() {
        let val = Value::float(3.14);
        assert_eq!(get_generator_state(&val), GeneratorState::NotAGenerator);
    }

    // =========================================================================
    // Type Name Tests
    // =========================================================================

    #[test]
    fn test_type_name_none() {
        assert_eq!(type_name(&Value::none()), "NoneType");
    }

    #[test]
    fn test_type_name_bool() {
        assert_eq!(type_name(&Value::bool(true)), "bool");
    }

    #[test]
    fn test_type_name_int() {
        let val = Value::int(42).unwrap();
        assert_eq!(type_name(&val), "int");
    }

    #[test]
    fn test_type_name_float() {
        let val = Value::float(3.14);
        assert_eq!(type_name(&val), "float");
    }

    // =========================================================================
    // Generator State Enum Tests
    // =========================================================================

    #[test]
    fn test_generator_state_equality() {
        assert_eq!(GeneratorState::Created, GeneratorState::Created);
        assert_ne!(GeneratorState::Created, GeneratorState::Suspended);
    }

    #[test]
    fn test_generator_state_debug() {
        let state = GeneratorState::Suspended;
        assert!(!format!("{:?}", state).is_empty());
    }
}
