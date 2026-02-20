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
use crate::builtins::{create_exception, create_exception_with_args};
use crate::dispatch::ControlFlow;
use crate::error::RuntimeError;
use crate::stdlib::exceptions::ExceptionTypeId;
use crate::stdlib::generators::{GeneratorObject, GeneratorState as RuntimeGeneratorState};
use crate::vm::GeneratorResumeOutcome;
use prism_compiler::bytecode::Instruction;
use prism_core::Value;

use super::protocol::type_name;

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
        ResumeResult::Returned(return_value) => {
            // Generator completed: raise StopIteration with the return value payload.
            let stop_iteration_value = if return_value.is_none() {
                create_exception(ExceptionTypeId::StopIteration, None)
            } else {
                create_exception_with_args(
                    ExceptionTypeId::StopIteration,
                    None,
                    vec![return_value].into_boxed_slice(),
                )
            };
            vm.set_active_exception_with_type(
                stop_iteration_value,
                ExceptionTypeId::StopIteration as u16,
            );
            ControlFlow::Exception {
                type_id: ExceptionTypeId::StopIteration as u16,
                handler_pc: 0,
            }
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
    let Some(generator) = GeneratorObject::from_value(*value) else {
        return GeneratorState::NotAGenerator;
    };

    match generator.state() {
        RuntimeGeneratorState::Created => GeneratorState::Created,
        RuntimeGeneratorState::Suspended => GeneratorState::Suspended,
        RuntimeGeneratorState::Running => GeneratorState::Running,
        RuntimeGeneratorState::Exhausted => GeneratorState::Closed,
    }
}

/// Get the type name of a value for error messages.
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
fn resume_generator(vm: &mut VirtualMachine, gen_value: Value, send_value: Value) -> ResumeResult {
    let Some(generator) = GeneratorObject::from_value_mut(gen_value) else {
        return ResumeResult::Error(RuntimeError::type_error(
            "send target is not a generator object",
        ));
    };

    match vm.resume_generator_for_send(generator, send_value) {
        Ok(GeneratorResumeOutcome::Yielded(value)) => ResumeResult::Yielded(value),
        Ok(GeneratorResumeOutcome::Returned(value)) => ResumeResult::Returned(value),
        Err(e) => ResumeResult::Error(e),
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builtins::ExceptionValue;
    use crate::error::RuntimeErrorKind;
    use crate::stdlib::exceptions::ExceptionTypeId;
    use crate::stdlib::generators::LivenessMap;
    use prism_compiler::bytecode::{
        CodeFlags, CodeObject, ExceptionEntry, Instruction, Opcode, Register,
    };
    use std::sync::Arc;

    fn generator_value_for_state(state: GeneratorState) -> Value {
        let code = Arc::new(CodeObject::new("test_send", "<test>"));
        let mut generator = GeneratorObject::new(code);
        let regs = [Value::none(); 256];

        match state {
            GeneratorState::Created => {}
            GeneratorState::Suspended => {
                generator.try_start();
                generator.suspend(10, 1, &regs, LivenessMap::from_bits(0b1));
            }
            GeneratorState::Running => {
                generator.try_start();
            }
            GeneratorState::Closed => {
                generator.try_start();
                generator.exhaust();
            }
            GeneratorState::NotAGenerator => {}
        }

        let ptr = Box::into_raw(Box::new(generator)) as *const ();
        Value::object_ptr(ptr)
    }

    fn push_caller_frame(vm: &mut VirtualMachine) {
        let mut caller = CodeObject::new("send_caller", "<test>");
        caller.register_count = 16;
        vm.push_frame(Arc::new(caller), 0)
            .expect("failed to push caller frame");
    }

    fn runtime_send_generator() -> Value {
        let mut code = CodeObject::new("runtime_send_generator", "<test>");
        code.flags = CodeFlags::GENERATOR;
        code.register_count = 8;
        code.constants = vec![Value::int(1).unwrap()].into_boxed_slice();
        code.instructions = vec![
            // r2 = 1
            Instruction::op_di(Opcode::LoadConst, Register::new(2), 0),
            // first yield: yields r2, sent value lands in r1
            Instruction::op_ds(Opcode::Yield, Register::new(1), Register::new(2)),
            // second yield: yields last sent value from r1
            Instruction::op_ds(Opcode::Yield, Register::new(1), Register::new(1)),
            // stop
            Instruction::op(Opcode::ReturnNone),
        ]
        .into_boxed_slice();

        let generator = GeneratorObject::from_code(Arc::new(code));
        let ptr = Box::into_raw(Box::new(generator)) as *const ();
        Value::object_ptr(ptr)
    }

    fn runtime_raise_generator() -> Value {
        let mut code = CodeObject::new("runtime_raise_generator", "<test>");
        code.flags = CodeFlags::GENERATOR;
        code.register_count = 8;
        code.constants = vec![create_exception(
            ExceptionTypeId::TypeError,
            Some(Arc::from("boom from generator")),
        )]
        .into_boxed_slice();
        code.instructions = vec![
            Instruction::op_di(Opcode::LoadConst, Register::new(2), 0),
            Instruction::op_di(
                Opcode::Raise,
                Register::new(2),
                ExceptionTypeId::TypeError as u16,
            ),
            Instruction::op(Opcode::ReturnNone),
        ]
        .into_boxed_slice();

        let generator = GeneratorObject::from_code(Arc::new(code));
        let ptr = Box::into_raw(Box::new(generator)) as *const ();
        Value::object_ptr(ptr)
    }

    fn runtime_handled_raise_generator() -> Value {
        let mut code = CodeObject::new("runtime_handled_raise_generator", "<test>");
        code.flags = CodeFlags::GENERATOR;
        code.register_count = 8;
        code.constants = vec![
            create_exception(
                ExceptionTypeId::TypeError,
                Some(Arc::from("caught in generator")),
            ),
            Value::int(9).unwrap(),
        ]
        .into_boxed_slice();
        code.instructions = vec![
            // Raise TypeError from try-range [pc=1, pc=2)
            Instruction::op_di(Opcode::LoadConst, Register::new(2), 0),
            Instruction::op_di(
                Opcode::Raise,
                Register::new(2),
                ExceptionTypeId::TypeError as u16,
            ),
            Instruction::op(Opcode::ReturnNone),
            // Exception handler target: yield sentinel 9 and keep normal generator protocol.
            Instruction::op_di(Opcode::LoadConst, Register::new(3), 1),
            Instruction::op_ds(Opcode::Yield, Register::new(1), Register::new(3)),
            Instruction::op(Opcode::ReturnNone),
        ]
        .into_boxed_slice();
        code.exception_table = vec![ExceptionEntry {
            start_pc: 1,
            end_pc: 2,
            handler_pc: 3,
            finally_pc: u32::MAX,
            depth: 0,
            exception_type_idx: u16::MAX,
        }]
        .into_boxed_slice();

        let generator = GeneratorObject::from_code(Arc::new(code));
        let ptr = Box::into_raw(Box::new(generator)) as *const ();
        Value::object_ptr(ptr)
    }

    fn runtime_return_value_generator() -> Value {
        let mut code = CodeObject::new("runtime_return_value_generator", "<test>");
        code.flags = CodeFlags::GENERATOR;
        code.register_count = 8;
        code.constants = vec![Value::int(42).unwrap()].into_boxed_slice();
        code.instructions = vec![
            Instruction::op_di(Opcode::LoadConst, Register::new(2), 0),
            Instruction::op_d(Opcode::Return, Register::new(2)),
        ]
        .into_boxed_slice();

        let generator = GeneratorObject::from_code(Arc::new(code));
        let ptr = Box::into_raw(Box::new(generator)) as *const ();
        Value::object_ptr(ptr)
    }

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

    #[test]
    fn test_generator_created_state_detected() {
        let generator_value = generator_value_for_state(GeneratorState::Created);
        assert_eq!(
            get_generator_state(&generator_value),
            GeneratorState::Created
        );
    }

    #[test]
    fn test_generator_suspended_state_detected() {
        let generator_value = generator_value_for_state(GeneratorState::Suspended);
        assert_eq!(
            get_generator_state(&generator_value),
            GeneratorState::Suspended
        );
    }

    #[test]
    fn test_generator_running_state_detected() {
        let generator_value = generator_value_for_state(GeneratorState::Running);
        assert_eq!(
            get_generator_state(&generator_value),
            GeneratorState::Running
        );
    }

    #[test]
    fn test_generator_closed_state_detected() {
        let generator_value = generator_value_for_state(GeneratorState::Closed);
        assert_eq!(
            get_generator_state(&generator_value),
            GeneratorState::Closed
        );
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

    #[test]
    fn test_type_name_generator() {
        let generator_value = generator_value_for_state(GeneratorState::Created);
        assert_eq!(type_name(&generator_value), "generator");
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

    #[test]
    fn test_send_resumes_generator_and_yields_sent_value() {
        let mut vm = VirtualMachine::new();
        push_caller_frame(&mut vm);

        let generator = runtime_send_generator();
        vm.current_frame_mut().set_reg(1, generator);

        // First resume must send None and should yield constant 1.
        vm.current_frame_mut().set_reg(2, Value::none());
        let inst = Instruction::new(Opcode::Send, 0, 1, 2);
        let control = send(&mut vm, inst);
        assert!(matches!(control, ControlFlow::Continue));
        assert_eq!(vm.current_frame().get_reg(0).as_int(), Some(1));
        assert_eq!(
            GeneratorObject::from_value(generator)
                .expect("generator")
                .state(),
            RuntimeGeneratorState::Suspended
        );

        // Second resume sends 77 and generator immediately yields it back.
        vm.current_frame_mut().set_reg(2, Value::int(77).unwrap());
        let control = send(&mut vm, inst);
        assert!(matches!(control, ControlFlow::Continue));
        assert_eq!(vm.current_frame().get_reg(0).as_int(), Some(77));
        assert_eq!(
            GeneratorObject::from_value(generator)
                .expect("generator")
                .state(),
            RuntimeGeneratorState::Suspended
        );

        // Third resume reaches return and raises StopIteration.
        vm.current_frame_mut().set_reg(2, Value::none());
        let control = send(&mut vm, inst);
        match control {
            ControlFlow::Exception { type_id, .. } => {
                assert_eq!(type_id, ExceptionTypeId::StopIteration as u16);
            }
            other => panic!("expected StopIteration, got {other:?}"),
        }
        assert_eq!(
            GeneratorObject::from_value(generator)
                .expect("generator")
                .state(),
            RuntimeGeneratorState::Exhausted
        );
    }

    #[test]
    fn test_send_rejects_non_none_on_initial_resume() {
        let mut vm = VirtualMachine::new();
        push_caller_frame(&mut vm);

        let generator = runtime_send_generator();
        vm.current_frame_mut().set_reg(1, generator);
        vm.current_frame_mut().set_reg(2, Value::int(5).unwrap());

        let inst = Instruction::new(Opcode::Send, 0, 1, 2);
        let control = send(&mut vm, inst);
        match control {
            ControlFlow::Error(err) => {
                assert!(err.to_string().contains("can't send non-None"));
            }
            other => panic!("expected TypeError, got {other:?}"),
        }

        // Generator should remain in created state after protocol violation.
        assert_eq!(
            GeneratorObject::from_value(generator)
                .expect("generator")
                .state(),
            RuntimeGeneratorState::Created
        );
    }

    #[test]
    fn test_send_propagates_uncaught_generator_exception() {
        let mut vm = VirtualMachine::new();
        push_caller_frame(&mut vm);

        let generator = runtime_raise_generator();
        vm.current_frame_mut().set_reg(1, generator);
        vm.current_frame_mut().set_reg(2, Value::none());

        let inst = Instruction::new(Opcode::Send, 0, 1, 2);
        let control = send(&mut vm, inst);
        match control {
            ControlFlow::Error(err) => {
                assert!(matches!(
                    err.kind,
                    RuntimeErrorKind::Exception { type_id, .. }
                        if type_id == ExceptionTypeId::TypeError as u16
                ));
            }
            other => panic!("expected propagated exception, got {other:?}"),
        }
        assert_eq!(
            GeneratorObject::from_value(generator)
                .expect("generator")
                .state(),
            RuntimeGeneratorState::Exhausted
        );
    }

    #[test]
    fn test_send_handles_generator_exception_table_path() {
        let mut vm = VirtualMachine::new();
        push_caller_frame(&mut vm);

        let generator = runtime_handled_raise_generator();
        vm.current_frame_mut().set_reg(1, generator);
        vm.current_frame_mut().set_reg(2, Value::none());
        let inst = Instruction::new(Opcode::Send, 0, 1, 2);

        // First send triggers raise, catches in generator exception table, then yields 9.
        let control = send(&mut vm, inst);
        assert!(matches!(control, ControlFlow::Continue));
        assert_eq!(vm.current_frame().get_reg(0).as_int(), Some(9));
        assert_eq!(
            GeneratorObject::from_value(generator)
                .expect("generator")
                .state(),
            RuntimeGeneratorState::Suspended
        );

        // Next send cleanly completes generator.
        vm.current_frame_mut().set_reg(2, Value::none());
        let control = send(&mut vm, inst);
        match control {
            ControlFlow::Exception { type_id, .. } => {
                assert_eq!(type_id, ExceptionTypeId::StopIteration as u16);
            }
            other => panic!("expected StopIteration, got {other:?}"),
        }
        assert_eq!(
            GeneratorObject::from_value(generator)
                .expect("generator")
                .state(),
            RuntimeGeneratorState::Exhausted
        );
    }

    #[test]
    fn test_send_sets_stop_iteration_value_from_generator_return() {
        let mut vm = VirtualMachine::new();
        push_caller_frame(&mut vm);

        let generator = runtime_return_value_generator();
        vm.current_frame_mut().set_reg(1, generator);
        vm.current_frame_mut().set_reg(2, Value::none());
        let inst = Instruction::new(Opcode::Send, 0, 1, 2);

        let control = send(&mut vm, inst);
        match control {
            ControlFlow::Exception { type_id, .. } => {
                assert_eq!(type_id, ExceptionTypeId::StopIteration as u16);
            }
            other => panic!("expected StopIteration exception flow, got {other:?}"),
        }

        let exc_value = vm
            .get_active_exception()
            .copied()
            .expect("stop iteration should be active");
        let exc = unsafe {
            ExceptionValue::from_value(exc_value)
                .expect("active exception should be an ExceptionValue object")
        };
        let args = exc
            .args
            .as_ref()
            .expect("StopIteration should carry return value in args");
        assert_eq!(args.len(), 1);
        assert_eq!(args[0].as_int(), Some(42));

        assert_eq!(
            GeneratorObject::from_value(generator)
                .expect("generator")
                .state(),
            RuntimeGeneratorState::Exhausted
        );
    }
}
