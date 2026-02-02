//! Generator method implementations (send, throw, close).
//!
//! This module provides the coroutine protocol methods:
//!
//! - `send(value)` - Send a value into the generator
//! - `throw(type, value, traceback)` - Throw an exception into the generator
//! - `close()` - Close the generator by throwing GeneratorExit
//!
//! # Python Semantics
//!
//! ```python
//! def coroutine():
//!     received = yield 1
//!     if received:
//!         yield received * 2
//!     else:
//!         yield 0
//!
//! gen = coroutine()
//! next(gen)       # Start: yields 1
//! gen.send(10)    # Sends 10, yields 20
//! gen.close()     # Throws GeneratorExit, generator stops
//! ```

use prism_core::Value;

use super::iterator::{GeneratorError, GeneratorException};
use super::object::GeneratorObject;
use super::state::GeneratorState;
use super::storage::LivenessMap;

// ============================================================================
// Send Protocol
// ============================================================================

/// Result of a send operation.
#[derive(Debug, Clone, PartialEq)]
pub enum SendResult {
    /// Generator yielded a value.
    Yielded(Value),
    /// Generator returned (completed).
    Returned(Value),
    /// Operation failed.
    Error(GeneratorError),
}

impl SendResult {
    /// Returns the yielded value if present.
    #[inline]
    pub fn yielded(&self) -> Option<Value> {
        match self {
            Self::Yielded(v) => Some(*v),
            _ => None,
        }
    }

    /// Returns the returned value if present.
    #[inline]
    pub fn returned(&self) -> Option<Value> {
        match self {
            Self::Returned(v) => Some(*v),
            _ => None,
        }
    }

    /// Returns true if the generator yielded.
    #[inline]
    pub fn is_yielded(&self) -> bool {
        matches!(self, Self::Yielded(_))
    }

    /// Returns true if the generator returned.
    #[inline]
    pub fn is_done(&self) -> bool {
        matches!(self, Self::Returned(_) | Self::Error(_))
    }
}

/// Prepares a generator for a send operation.
///
/// This validates the generator state and sets the send value.
/// Returns the previous state if successful.
pub fn prepare_send(
    generator: &mut GeneratorObject,
    value: Value,
) -> Result<GeneratorState, GeneratorError> {
    match generator.state() {
        GeneratorState::Created => {
            // Can only send None to a just-created generator
            if !value.is_none() {
                return Err(GeneratorError::CantSendNonNone);
            }
            generator.set_send_value(value);
            generator.try_start().ok_or(GeneratorError::Exhausted)
        }
        GeneratorState::Suspended => {
            // Can send any value to a suspended generator
            generator.set_send_value(value);
            generator.try_start().ok_or(GeneratorError::Exhausted)
        }
        GeneratorState::Running => Err(GeneratorError::AlreadyRunning),
        GeneratorState::Exhausted => Err(GeneratorError::Exhausted),
    }
}

/// Completes a send operation after the generator has yielded.
///
/// Call this after the VM executes the generator and it yields.
pub fn complete_send_yielded(generator: &mut GeneratorObject, yielded_value: Value) -> SendResult {
    // The generator should now be suspended by the VM
    // We return the yielded value
    SendResult::Yielded(yielded_value)
}

/// Completes a send operation after the generator has returned.
///
/// Call this after the VM executes the generator and it returns.
pub fn complete_send_returned(generator: &mut GeneratorObject, return_value: Value) -> SendResult {
    generator.exhaust();
    SendResult::Returned(return_value)
}

// ============================================================================
// Throw Protocol
// ============================================================================

/// Result of a throw operation.
#[derive(Debug, Clone, PartialEq)]
pub enum ThrowResult {
    /// Generator caught the exception and yielded.
    Yielded(Value),
    /// Generator caught the exception and returned.
    Returned(Value),
    /// Exception propagated out of the generator.
    Propagated(GeneratorException),
    /// Operation failed (generator not in valid state).
    Error(GeneratorError),
}

impl ThrowResult {
    /// Returns true if the exception was caught and the generator yielded.
    #[inline]
    pub fn is_yielded(&self) -> bool {
        matches!(self, Self::Yielded(_))
    }

    /// Returns true if the generator is now exhausted.
    #[inline]
    pub fn is_done(&self) -> bool {
        matches!(
            self,
            Self::Returned(_) | Self::Propagated(_) | Self::Error(_)
        )
    }
}

/// Prepares a generator for a throw operation.
///
/// This sets up the exception to be thrown when the generator resumes.
pub fn prepare_throw(
    generator: &mut GeneratorObject,
    exception: GeneratorException,
) -> Result<GeneratorState, GeneratorError> {
    match generator.state() {
        GeneratorState::Created => {
            // Throwing into a never-started generator exhausts it
            generator.try_start();
            generator.exhaust();
            Err(GeneratorError::ThrownException(exception))
        }
        GeneratorState::Suspended => {
            // Throw exception into suspended generator
            generator.try_start().ok_or(GeneratorError::Exhausted)
        }
        GeneratorState::Running => Err(GeneratorError::AlreadyRunning),
        GeneratorState::Exhausted => Err(GeneratorError::Exhausted),
    }
}

/// Completes a throw operation after the generator has handled it.
pub fn complete_throw_yielded(
    generator: &mut GeneratorObject,
    yielded_value: Value,
) -> ThrowResult {
    ThrowResult::Yielded(yielded_value)
}

/// Completes a throw operation after the generator has returned.
pub fn complete_throw_returned(
    generator: &mut GeneratorObject,
    return_value: Value,
) -> ThrowResult {
    generator.exhaust();
    ThrowResult::Returned(return_value)
}

/// Completes a throw operation when the exception propagated.
pub fn complete_throw_propagated(
    generator: &mut GeneratorObject,
    exception: GeneratorException,
) -> ThrowResult {
    generator.exhaust();
    ThrowResult::Propagated(exception)
}

// ============================================================================
// Close Protocol
// ============================================================================

/// Result of a close operation.
#[derive(Debug, Clone, PartialEq)]
pub enum CloseResult {
    /// Generator closed successfully.
    Closed,
    /// Generator raised an exception other than GeneratorExit.
    RuntimeError(GeneratorException),
    /// Generator yielded a value (which is an error).
    YieldedInFinally(Value),
}

impl CloseResult {
    /// Returns true if the generator closed successfully.
    #[inline]
    pub fn is_ok(&self) -> bool {
        matches!(self, Self::Closed)
    }

    /// Returns true if there was an error.
    #[inline]
    pub fn is_err(&self) -> bool {
        !self.is_ok()
    }
}

/// Prepares a generator for a close operation.
///
/// This throws GeneratorExit into the generator.
pub fn prepare_close(generator: &mut GeneratorObject) -> CloseResult {
    match generator.state() {
        GeneratorState::Created => {
            // Never-started generator can just be marked exhausted
            generator.try_start();
            generator.exhaust();
            CloseResult::Closed
        }
        GeneratorState::Exhausted => {
            // Already closed
            CloseResult::Closed
        }
        GeneratorState::Running => {
            // Can't close a running generator
            CloseResult::RuntimeError(GeneratorException::new(
                "ValueError",
                "generator already executing",
            ))
        }
        GeneratorState::Suspended => {
            // Need to throw GeneratorExit
            if generator.try_start().is_none() {
                CloseResult::Closed
            } else {
                // VM will handle throwing GeneratorExit
                CloseResult::Closed
            }
        }
    }
}

/// Completes a close operation after the generator has stopped.
pub fn complete_close_caught(generator: &mut GeneratorObject) -> CloseResult {
    generator.exhaust();
    CloseResult::Closed
}

/// Completes a close operation if the generator yielded (error case).
pub fn complete_close_yielded(generator: &mut GeneratorObject, value: Value) -> CloseResult {
    generator.exhaust();
    CloseResult::YieldedInFinally(value)
}

/// Completes a close operation if an exception propagated.
pub fn complete_close_exception(
    generator: &mut GeneratorObject,
    exception: GeneratorException,
) -> CloseResult {
    generator.exhaust();

    // GeneratorExit is expected and counts as success
    if exception.type_name == "GeneratorExit" {
        CloseResult::Closed
    } else {
        CloseResult::RuntimeError(exception)
    }
}

// ============================================================================
// Utility Functions
// ============================================================================

/// Creates a GeneratorExit exception for closing generators.
#[inline]
pub fn generator_exit() -> GeneratorException {
    GeneratorException::new("GeneratorExit", "generator exit")
}

/// Creates a StopIteration exception.
#[inline]
pub fn stop_iteration(value: Option<Value>) -> GeneratorException {
    match value {
        Some(v) => GeneratorException::with_value("StopIteration", "", v),
        None => GeneratorException::new("StopIteration", ""),
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use prism_compiler::bytecode::CodeObject;
    use std::sync::Arc;

    fn test_code() -> Arc<CodeObject> {
        Arc::new(CodeObject::new("test_generator", "<test>"))
    }

    // ════════════════════════════════════════════════════════════════════════
    // SendResult Tests
    // ════════════════════════════════════════════════════════════════════════

    #[test]
    fn test_send_result_yielded() {
        let result = SendResult::Yielded(Value::int(42).unwrap());
        assert!(result.is_yielded());
        assert!(!result.is_done());
        assert_eq!(result.yielded().unwrap().as_int().unwrap(), 42);
        assert!(result.returned().is_none());
    }

    #[test]
    fn test_send_result_returned() {
        let result = SendResult::Returned(Value::none());
        assert!(!result.is_yielded());
        assert!(result.is_done());
        assert!(result.yielded().is_none());
        assert!(result.returned().is_some());
    }

    #[test]
    fn test_send_result_error() {
        let result = SendResult::Error(GeneratorError::Exhausted);
        assert!(!result.is_yielded());
        assert!(result.is_done());
    }

    // ════════════════════════════════════════════════════════════════════════
    // Send Protocol Tests
    // ════════════════════════════════════════════════════════════════════════

    #[test]
    fn test_prepare_send_none_to_created() {
        let code = test_code();
        let mut generator = GeneratorObject::new(code);

        let result = prepare_send(&mut generator, Value::none());
        assert_eq!(result, Ok(GeneratorState::Created));
        assert!(generator.is_running());
    }

    #[test]
    fn test_prepare_send_value_to_created_fails() {
        let code = test_code();
        let mut generator = GeneratorObject::new(code);

        let result = prepare_send(&mut generator, Value::int(42).unwrap());
        assert_eq!(result, Err(GeneratorError::CantSendNonNone));
    }

    #[test]
    fn test_prepare_send_to_suspended() {
        let code = test_code();
        let mut generator = GeneratorObject::new(code);

        // Start and suspend
        generator.try_start();
        let regs = [Value::none(); 256];
        generator.suspend(10, 0, &regs, LivenessMap::from_bits(0));

        let result = prepare_send(&mut generator, Value::int(42).unwrap());
        assert_eq!(result, Ok(GeneratorState::Suspended));
        assert!(generator.is_running());
        assert_eq!(
            generator.peek_receive_value().unwrap().as_int().unwrap(),
            42
        );
    }

    #[test]
    fn test_prepare_send_to_running() {
        let code = test_code();
        let mut generator = GeneratorObject::new(code);
        generator.try_start();

        let result = prepare_send(&mut generator, Value::none());
        assert_eq!(result, Err(GeneratorError::AlreadyRunning));
    }

    #[test]
    fn test_prepare_send_to_exhausted() {
        let code = test_code();
        let mut generator = GeneratorObject::new(code);
        generator.try_start();
        generator.exhaust();

        let result = prepare_send(&mut generator, Value::none());
        assert_eq!(result, Err(GeneratorError::Exhausted));
    }

    #[test]
    fn test_complete_send_yielded() {
        let code = test_code();
        let mut generator = GeneratorObject::new(code);

        let result = complete_send_yielded(&mut generator, Value::int(100).unwrap());
        assert!(result.is_yielded());
        assert_eq!(result.yielded().unwrap().as_int().unwrap(), 100);
    }

    #[test]
    fn test_complete_send_returned() {
        let code = test_code();
        let mut generator = GeneratorObject::new(code);
        generator.try_start();

        let result = complete_send_returned(&mut generator, Value::int(200).unwrap());
        assert!(result.is_done());
        assert_eq!(result.returned().unwrap().as_int().unwrap(), 200);
        assert!(generator.is_exhausted());
    }

    // ════════════════════════════════════════════════════════════════════════
    // ThrowResult Tests
    // ════════════════════════════════════════════════════════════════════════

    #[test]
    fn test_throw_result_yielded() {
        let result = ThrowResult::Yielded(Value::int(42).unwrap());
        assert!(result.is_yielded());
        assert!(!result.is_done());
    }

    #[test]
    fn test_throw_result_returned() {
        let result = ThrowResult::Returned(Value::none());
        assert!(!result.is_yielded());
        assert!(result.is_done());
    }

    #[test]
    fn test_throw_result_propagated() {
        let exc = GeneratorException::new("ValueError", "test");
        let result = ThrowResult::Propagated(exc);
        assert!(!result.is_yielded());
        assert!(result.is_done());
    }

    // ════════════════════════════════════════════════════════════════════════
    // Throw Protocol Tests
    // ════════════════════════════════════════════════════════════════════════

    #[test]
    fn test_prepare_throw_to_created() {
        let code = test_code();
        let mut generator = GeneratorObject::new(code);

        let exc = GeneratorException::new("ValueError", "test");
        let result = prepare_throw(&mut generator, exc.clone());

        assert!(matches!(result, Err(GeneratorError::ThrownException(_))));
        assert!(generator.is_exhausted());
    }

    #[test]
    fn test_prepare_throw_to_suspended() {
        let code = test_code();
        let mut generator = GeneratorObject::new(code);

        // Start and suspend
        generator.try_start();
        let regs = [Value::none(); 256];
        generator.suspend(10, 0, &regs, LivenessMap::from_bits(0));

        let exc = GeneratorException::new("ValueError", "test");
        let result = prepare_throw(&mut generator, exc);

        assert_eq!(result, Ok(GeneratorState::Suspended));
        assert!(generator.is_running());
    }

    #[test]
    fn test_complete_throw_yielded() {
        let code = test_code();
        let mut generator = GeneratorObject::new(code);

        let result = complete_throw_yielded(&mut generator, Value::int(50).unwrap());
        assert!(result.is_yielded());
    }

    #[test]
    fn test_complete_throw_returned() {
        let code = test_code();
        let mut generator = GeneratorObject::new(code);
        generator.try_start();

        let result = complete_throw_returned(&mut generator, Value::none());
        assert!(result.is_done());
        assert!(generator.is_exhausted());
    }

    #[test]
    fn test_complete_throw_propagated() {
        let code = test_code();
        let mut generator = GeneratorObject::new(code);
        generator.try_start();

        let exc = GeneratorException::new("RuntimeError", "unhandled");
        let result = complete_throw_propagated(&mut generator, exc);

        assert!(result.is_done());
        assert!(generator.is_exhausted());
    }

    // ════════════════════════════════════════════════════════════════════════
    // CloseResult Tests
    // ════════════════════════════════════════════════════════════════════════

    #[test]
    fn test_close_result_closed() {
        let result = CloseResult::Closed;
        assert!(result.is_ok());
        assert!(!result.is_err());
    }

    #[test]
    fn test_close_result_error() {
        let result = CloseResult::RuntimeError(GeneratorException::new("ValueError", "test"));
        assert!(!result.is_ok());
        assert!(result.is_err());
    }

    #[test]
    fn test_close_result_yielded_in_finally() {
        let result = CloseResult::YieldedInFinally(Value::int(42).unwrap());
        assert!(!result.is_ok());
        assert!(result.is_err());
    }

    // ════════════════════════════════════════════════════════════════════════
    // Close Protocol Tests
    // ════════════════════════════════════════════════════════════════════════

    #[test]
    fn test_prepare_close_created() {
        let code = test_code();
        let mut generator = GeneratorObject::new(code);

        let result = prepare_close(&mut generator);
        assert!(result.is_ok());
        assert!(generator.is_exhausted());
    }

    #[test]
    fn test_prepare_close_exhausted() {
        let code = test_code();
        let mut generator = GeneratorObject::new(code);
        generator.try_start();
        generator.exhaust();

        let result = prepare_close(&mut generator);
        assert!(result.is_ok());
    }

    #[test]
    fn test_prepare_close_running() {
        let code = test_code();
        let mut generator = GeneratorObject::new(code);
        generator.try_start();

        let result = prepare_close(&mut generator);
        assert!(result.is_err());
    }

    #[test]
    fn test_prepare_close_suspended() {
        let code = test_code();
        let mut generator = GeneratorObject::new(code);

        // Start and suspend
        generator.try_start();
        let regs = [Value::none(); 256];
        generator.suspend(10, 0, &regs, LivenessMap::from_bits(0));

        let result = prepare_close(&mut generator);
        // For suspended, we prepare for GeneratorExit throw
        assert!(result.is_ok());
    }

    #[test]
    fn test_complete_close_caught() {
        let code = test_code();
        let mut generator = GeneratorObject::new(code);
        generator.try_start();

        let result = complete_close_caught(&mut generator);
        assert!(result.is_ok());
        assert!(generator.is_exhausted());
    }

    #[test]
    fn test_complete_close_yielded() {
        let code = test_code();
        let mut generator = GeneratorObject::new(code);
        generator.try_start();

        let result = complete_close_yielded(&mut generator, Value::int(42).unwrap());
        assert!(result.is_err());
        assert!(matches!(result, CloseResult::YieldedInFinally(_)));
    }

    #[test]
    fn test_complete_close_generator_exit() {
        let code = test_code();
        let mut generator = GeneratorObject::new(code);
        generator.try_start();

        let exc = generator_exit();
        let result = complete_close_exception(&mut generator, exc);
        assert!(result.is_ok());
    }

    #[test]
    fn test_complete_close_other_exception() {
        let code = test_code();
        let mut generator = GeneratorObject::new(code);
        generator.try_start();

        let exc = GeneratorException::new("RuntimeError", "something else");
        let result = complete_close_exception(&mut generator, exc);
        assert!(result.is_err());
    }

    // ════════════════════════════════════════════════════════════════════════
    // Utility Tests
    // ════════════════════════════════════════════════════════════════════════

    #[test]
    fn test_generator_exit() {
        let exc = generator_exit();
        assert_eq!(exc.type_name, "GeneratorExit");
    }

    #[test]
    fn test_stop_iteration() {
        let exc = stop_iteration(None);
        assert_eq!(exc.type_name, "StopIteration");
        assert!(exc.value.is_none());

        let exc_with_value = stop_iteration(Some(Value::int(42).unwrap()));
        assert!(exc_with_value.value.is_some());
    }
}
