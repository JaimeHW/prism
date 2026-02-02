//! Iterator protocol implementation for generators.
//!
//! This module provides the `__iter__` and `__next__` methods for generators,
//! enabling them to be used in for-loops and other iterator contexts.
//!
//! # Python Semantics
//!
//! ```python
//! def my_generator():
//!     yield 1
//!     yield 2
//!     yield 3
//!
//! gen = my_generator()
//! iter(gen) is gen  # True - generators are their own iterators
//!
//! next(gen)  # 1
//! next(gen)  # 2
//! next(gen)  # 3
//! next(gen)  # raises StopIteration
//! ```
//!
//! # Thread Safety
//!
//! Generators are NOT thread-safe. Concurrent access to a generator from
//! multiple threads leads to undefined behavior. The caller must ensure
//! exclusive access during iteration.

use prism_core::Value;

use super::object::GeneratorObject;
use super::state::GeneratorState;

// ============================================================================
// Iterator Result
// ============================================================================

/// Result of a generator iteration step.
#[derive(Debug, Clone, PartialEq)]
pub enum IterResult {
    /// Generator yielded a value.
    Yielded(Value),
    /// Generator returned a value (completed normally).
    Returned(Value),
    /// Generator raised an exception.
    Raised(GeneratorError),
}

impl IterResult {
    /// Returns true if the generator yielded a value.
    #[inline]
    pub fn is_yielded(&self) -> bool {
        matches!(self, Self::Yielded(_))
    }

    /// Returns true if the generator returned (completed).
    #[inline]
    pub fn is_returned(&self) -> bool {
        matches!(self, Self::Returned(_))
    }

    /// Returns true if the generator raised an exception.
    #[inline]
    pub fn is_raised(&self) -> bool {
        matches!(self, Self::Raised(_))
    }

    /// Extracts the yielded value, if any.
    #[inline]
    pub fn yielded_value(&self) -> Option<Value> {
        match self {
            Self::Yielded(v) => Some(*v),
            _ => None,
        }
    }

    /// Extracts the returned value, if any.
    #[inline]
    pub fn returned_value(&self) -> Option<Value> {
        match self {
            Self::Returned(v) => Some(*v),
            _ => None,
        }
    }

    /// Converts to a Python-compatible result similar to StopIteration.value.
    #[inline]
    pub fn into_stop_iteration_value(self) -> Option<Value> {
        match self {
            Self::Returned(v) => Some(v),
            _ => None,
        }
    }
}

// ============================================================================
// Generator Error Types
// ============================================================================

/// Errors that can occur during generator execution.
#[derive(Debug, Clone, PartialEq)]
pub enum GeneratorError {
    /// Generator is already running (recursive call).
    AlreadyRunning,
    /// Generator is exhausted (already finished).
    Exhausted,
    /// Generator was never started but send() was called with non-None value.
    CantSendNonNone,
    /// An exception was thrown into the generator.
    ThrownException(GeneratorException),
    /// Generator raised StopIteration explicitly.
    StopIteration(Option<Value>),
    /// Runtime error during execution.
    RuntimeError(String),
}

impl GeneratorError {
    /// Creates a runtime error.
    pub fn runtime<S: Into<String>>(msg: S) -> Self {
        Self::RuntimeError(msg.into())
    }

    /// Returns true if the generator is simply exhausted.
    #[inline]
    pub fn is_exhausted(&self) -> bool {
        matches!(self, Self::Exhausted | Self::StopIteration(_))
    }

    /// Returns true if this is a fatal error.
    #[inline]
    pub fn is_fatal(&self) -> bool {
        matches!(self, Self::RuntimeError(_))
    }
}

impl std::fmt::Display for GeneratorError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::AlreadyRunning => write!(f, "generator already executing"),
            Self::Exhausted => write!(f, "StopIteration"),
            Self::CantSendNonNone => {
                write!(f, "can't send non-None value to a just-started generator")
            }
            Self::ThrownException(exc) => write!(f, "{}", exc),
            Self::StopIteration(None) => write!(f, "StopIteration"),
            Self::StopIteration(Some(_)) => write!(f, "StopIteration (with value)"),
            Self::RuntimeError(msg) => write!(f, "RuntimeError: {}", msg),
        }
    }
}

impl std::error::Error for GeneratorError {}

/// An exception thrown into a generator.
#[derive(Debug, Clone, PartialEq)]
pub struct GeneratorException {
    /// Exception type name (e.g., "ValueError")
    pub type_name: String,
    /// Exception message.
    pub message: String,
    /// Optional exception value.
    pub value: Option<Value>,
}

impl GeneratorException {
    /// Creates a new exception.
    pub fn new<T: Into<String>, M: Into<String>>(type_name: T, message: M) -> Self {
        Self {
            type_name: type_name.into(),
            message: message.into(),
            value: None,
        }
    }

    /// Creates a new exception with a value.
    pub fn with_value<T: Into<String>, M: Into<String>>(
        type_name: T,
        message: M,
        value: Value,
    ) -> Self {
        Self {
            type_name: type_name.into(),
            message: message.into(),
            value: Some(value),
        }
    }
}

impl std::fmt::Display for GeneratorException {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}: {}", self.type_name, self.message)
    }
}

// ============================================================================
// Generator Iterator
// ============================================================================

/// Iterator adapter for generators.
///
/// This wraps a `GeneratorObject` and provides a standard Rust iterator
/// interface for generators that only need simple iteration (no send/throw).
pub struct GeneratorIterator<'a> {
    /// The generator being iterated.
    generator: &'a mut GeneratorObject,
    /// Whether StopIteration has been raised.
    exhausted: bool,
}

impl<'a> GeneratorIterator<'a> {
    /// Creates a new iterator over a generator.
    #[inline]
    pub fn new(generator: &'a mut GeneratorObject) -> Self {
        Self {
            generator,
            exhausted: false,
        }
    }

    /// Returns a reference to the underlying generator.
    #[inline]
    pub fn generator(&self) -> &GeneratorObject {
        self.generator
    }

    /// Returns a mutable reference to the underlying generator.
    #[inline]
    pub fn generator_mut(&mut self) -> &mut GeneratorObject {
        self.generator
    }

    /// Checks if the generator is currently running.
    #[inline]
    pub fn is_running(&self) -> bool {
        self.generator.is_running()
    }

    /// Checks if the generator is exhausted.
    #[inline]
    pub fn is_exhausted(&self) -> bool {
        self.exhausted || self.generator.is_exhausted()
    }
}

// ============================================================================
// Iteration Helpers
// ============================================================================

/// Validates that a generator can be started or resumed.
///
/// Returns the previous state if the generator can be resumed,
/// or an appropriate error otherwise.
#[inline]
pub fn validate_resumable(generator: &GeneratorObject) -> Result<GeneratorState, GeneratorError> {
    match generator.state() {
        GeneratorState::Created => Ok(GeneratorState::Created),
        GeneratorState::Suspended => Ok(GeneratorState::Suspended),
        GeneratorState::Running => Err(GeneratorError::AlreadyRunning),
        GeneratorState::Exhausted => Err(GeneratorError::Exhausted),
    }
}

/// Validates that a send value is appropriate for the generator state.
///
/// You can only send a non-None value to a generator that has been started.
#[inline]
pub fn validate_send_value(
    generator: &GeneratorObject,
    value: Option<Value>,
) -> Result<(), GeneratorError> {
    match (generator.state(), value) {
        // Can always send None or to a suspended generator
        (GeneratorState::Suspended, _) => Ok(()),
        (GeneratorState::Created, None) => Ok(()),
        (GeneratorState::Created, Some(v)) if v.is_none() => Ok(()),
        // Cannot send non-None to a just-started generator
        (GeneratorState::Created, Some(_)) => Err(GeneratorError::CantSendNonNone),
        // Cannot send to running or exhausted generators
        (GeneratorState::Running, _) => Err(GeneratorError::AlreadyRunning),
        (GeneratorState::Exhausted, _) => Err(GeneratorError::Exhausted),
    }
}

/// Prepares a generator for the next iteration step.
///
/// This function:
/// 1. Validates that the generator can be resumed
/// 2. Optionally validates the send value
/// 3. Sets the send value if provided
/// 4. Starts the generator
///
/// Returns the previous state if successful.
pub fn prepare_iteration(
    generator: &mut GeneratorObject,
    send_value: Option<Value>,
) -> Result<GeneratorState, GeneratorError> {
    // Validate state
    let prev_state = validate_resumable(generator)?;

    // Validate and set send value
    if let Some(value) = send_value {
        validate_send_value(generator, Some(value))?;
        generator.set_send_value(value);
    }

    // Try to start the generator
    match generator.try_start() {
        Some(state) => Ok(state),
        None => {
            // This shouldn't happen if validation passed
            if generator.is_running() {
                Err(GeneratorError::AlreadyRunning)
            } else {
                Err(GeneratorError::Exhausted)
            }
        }
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
    // IterResult Tests
    // ════════════════════════════════════════════════════════════════════════

    #[test]
    fn test_iter_result_yielded() {
        let result = IterResult::Yielded(Value::int(42).unwrap());
        assert!(result.is_yielded());
        assert!(!result.is_returned());
        assert!(!result.is_raised());
        assert_eq!(result.yielded_value().unwrap().as_int().unwrap(), 42);
    }

    #[test]
    fn test_iter_result_returned() {
        let result = IterResult::Returned(Value::none());
        assert!(!result.is_yielded());
        assert!(result.is_returned());
        assert!(!result.is_raised());
        assert!(result.returned_value().unwrap().is_none());
    }

    #[test]
    fn test_iter_result_raised() {
        let result = IterResult::Raised(GeneratorError::Exhausted);
        assert!(!result.is_yielded());
        assert!(!result.is_returned());
        assert!(result.is_raised());
    }

    #[test]
    fn test_stop_iteration_value() {
        let returned = IterResult::Returned(Value::int(100).unwrap());
        let stop_value = returned.into_stop_iteration_value();
        assert!(stop_value.is_some());
        assert_eq!(stop_value.unwrap().as_int().unwrap(), 100);

        let yielded = IterResult::Yielded(Value::int(50).unwrap());
        assert!(yielded.into_stop_iteration_value().is_none());
    }

    // ════════════════════════════════════════════════════════════════════════
    // GeneratorError Tests
    // ════════════════════════════════════════════════════════════════════════

    #[test]
    fn test_error_already_running() {
        let err = GeneratorError::AlreadyRunning;
        assert!(!err.is_exhausted());
        assert!(!err.is_fatal());
        assert!(err.to_string().contains("already executing"));
    }

    #[test]
    fn test_error_exhausted() {
        let err = GeneratorError::Exhausted;
        assert!(err.is_exhausted());
        assert!(!err.is_fatal());
        assert!(err.to_string().contains("StopIteration"));
    }

    #[test]
    fn test_error_cant_send_non_none() {
        let err = GeneratorError::CantSendNonNone;
        assert!(!err.is_exhausted());
        assert!(!err.is_fatal());
        assert!(err.to_string().contains("non-None"));
    }

    #[test]
    fn test_error_runtime() {
        let err = GeneratorError::runtime("test error");
        assert!(!err.is_exhausted());
        assert!(err.is_fatal());
        assert!(err.to_string().contains("test error"));
    }

    #[test]
    fn test_error_stop_iteration_with_value() {
        let err = GeneratorError::StopIteration(Some(Value::int(42).unwrap()));
        assert!(err.is_exhausted());
        assert!(err.to_string().contains("with value"));
    }

    // ════════════════════════════════════════════════════════════════════════
    // GeneratorException Tests
    // ════════════════════════════════════════════════════════════════════════

    #[test]
    fn test_exception_new() {
        let exc = GeneratorException::new("ValueError", "invalid value");
        assert_eq!(exc.type_name, "ValueError");
        assert_eq!(exc.message, "invalid value");
        assert!(exc.value.is_none());
        assert!(exc.to_string().contains("ValueError"));
        assert!(exc.to_string().contains("invalid value"));
    }

    #[test]
    fn test_exception_with_value() {
        let exc =
            GeneratorException::with_value("TypeError", "wrong type", Value::int(99).unwrap());
        assert_eq!(exc.type_name, "TypeError");
        assert!(exc.value.is_some());
        assert_eq!(exc.value.unwrap().as_int().unwrap(), 99);
    }

    // ════════════════════════════════════════════════════════════════════════
    // Validation Tests
    // ════════════════════════════════════════════════════════════════════════

    #[test]
    fn test_validate_resumable_created() {
        let code = test_code();
        let generator = GeneratorObject::new(code);
        assert_eq!(validate_resumable(&generator), Ok(GeneratorState::Created));
    }

    #[test]
    fn test_validate_resumable_running() {
        let code = test_code();
        let generator = GeneratorObject::new(code);
        generator.try_start();
        assert_eq!(
            validate_resumable(&generator),
            Err(GeneratorError::AlreadyRunning)
        );
    }

    #[test]
    fn test_validate_resumable_exhausted() {
        let code = test_code();
        let generator = GeneratorObject::new(code);
        generator.try_start();
        generator.exhaust();
        assert_eq!(
            validate_resumable(&generator),
            Err(GeneratorError::Exhausted)
        );
    }

    #[test]
    fn test_validate_send_none_to_created() {
        let code = test_code();
        let generator = GeneratorObject::new(code);
        assert!(validate_send_value(&generator, None).is_ok());
        assert!(validate_send_value(&generator, Some(Value::none())).is_ok());
    }

    #[test]
    fn test_validate_send_value_to_created() {
        let code = test_code();
        let generator = GeneratorObject::new(code);
        assert_eq!(
            validate_send_value(&generator, Some(Value::int(42).unwrap())),
            Err(GeneratorError::CantSendNonNone)
        );
    }

    #[test]
    fn test_validate_send_to_exhausted() {
        let code = test_code();
        let generator = GeneratorObject::new(code);
        generator.try_start();
        generator.exhaust();
        assert_eq!(
            validate_send_value(&generator, None),
            Err(GeneratorError::Exhausted)
        );
    }

    // ════════════════════════════════════════════════════════════════════════
    // Prepare Iteration Tests
    // ════════════════════════════════════════════════════════════════════════

    #[test]
    fn test_prepare_first_iteration() {
        let code = test_code();
        let mut generator = GeneratorObject::new(code);

        let result = prepare_iteration(&mut generator, None);
        assert_eq!(result, Ok(GeneratorState::Created));
        assert!(generator.is_running());
    }

    #[test]
    fn test_prepare_with_send_value() {
        let code = test_code();
        let mut generator = GeneratorObject::new(code);

        // First iteration must be None
        prepare_iteration(&mut generator, None).unwrap();

        // Simulate suspend
        let regs = [Value::none(); 256];
        generator.suspend(
            10,
            0,
            &regs,
            super::super::storage::LivenessMap::from_bits(0),
        );

        // Now we can send a value
        let mut generator_new = GeneratorObject::new(test_code());
        generator_new.try_start();
        generator_new.suspend(
            10,
            0,
            &regs,
            super::super::storage::LivenessMap::from_bits(0),
        );

        let result = prepare_iteration(&mut generator_new, Some(Value::int(42).unwrap()));
        assert_eq!(result, Ok(GeneratorState::Suspended));
        assert_eq!(
            generator_new
                .peek_receive_value()
                .unwrap()
                .as_int()
                .unwrap(),
            42
        );
    }

    #[test]
    fn test_prepare_iteration_already_running() {
        let code = test_code();
        let mut generator = GeneratorObject::new(code);
        generator.try_start();

        let result = prepare_iteration(&mut generator, None);
        assert_eq!(result, Err(GeneratorError::AlreadyRunning));
    }

    #[test]
    fn test_prepare_iteration_exhausted() {
        let code = test_code();
        let mut generator = GeneratorObject::new(code);
        generator.try_start();
        generator.exhaust();

        let result = prepare_iteration(&mut generator, None);
        assert_eq!(result, Err(GeneratorError::Exhausted));
    }

    // ════════════════════════════════════════════════════════════════════════
    // GeneratorIterator Tests
    // ════════════════════════════════════════════════════════════════════════

    #[test]
    fn test_generator_iterator_new() {
        let code = test_code();
        let mut generator = GeneratorObject::new(code);
        let iter = GeneratorIterator::new(&mut generator);

        assert!(!iter.is_running());
        assert!(!iter.is_exhausted());
    }

    #[test]
    fn test_generator_iterator_running() {
        let code = test_code();
        let mut generator = GeneratorObject::new(code);
        generator.try_start();

        let iter = GeneratorIterator::new(&mut generator);
        assert!(iter.is_running());
    }

    #[test]
    fn test_generator_iterator_exhausted() {
        let code = test_code();
        let mut generator = GeneratorObject::new(code);
        generator.try_start();
        generator.exhaust();

        let iter = GeneratorIterator::new(&mut generator);
        assert!(iter.is_exhausted());
    }

    #[test]
    fn test_generator_iterator_access() {
        let code = test_code();
        let mut generator = GeneratorObject::new(code);

        let mut iter = GeneratorIterator::new(&mut generator);

        // Can access generator through iterator
        assert_eq!(iter.generator().state(), GeneratorState::Created);
        iter.generator_mut().try_start();
        assert!(iter.is_running());
    }
}
