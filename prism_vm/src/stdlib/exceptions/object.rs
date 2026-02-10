//! Exception object implementation.
//!
//! This module provides the core `ExceptionObject` type that represents
//! a Python exception instance. The design follows CPython 3.11+ patterns
//! with lazy allocation of args and tracebacks.
//!
//! # Performance Design
//!
//! - **Lazy args**: Exception args are only allocated when accessed
//! - **Lazy traceback**: Traceback is built incrementally as exception propagates
//! - **Compact header**: Type ID + flags = 2 bytes
//! - **OnceCell for lazy init**: Thread-safe lazy initialization

use super::flags::ExceptionFlags;
use super::traceback::TracebackObject;
use super::types::ExceptionTypeId;
use prism_core::Value;
use std::fmt;
use std::sync::Arc;
use std::sync::OnceLock;

// ============================================================================
// Exception Arguments
// ============================================================================

/// Lazy exception arguments.
///
/// Exception args are stored lazily to avoid allocation for exceptions
/// that don't need their args (e.g., control-flow exceptions).
#[derive(Clone)]
pub struct ExceptionArgs {
    /// The args tuple contents.
    values: Box<[Value]>,
}

impl ExceptionArgs {
    /// Creates empty args (no values).
    pub fn empty() -> Self {
        Self {
            values: Box::new([]),
        }
    }

    /// Creates args from a single value.
    pub fn single(value: Value) -> Self {
        Self {
            values: Box::new([value]),
        }
    }

    /// Creates args from a slice of values.
    pub fn from_slice(values: &[Value]) -> Self {
        Self {
            values: values.into(),
        }
    }

    /// Creates args from an iterator.
    pub fn from_iter(iter: impl IntoIterator<Item = Value>) -> Self {
        Self {
            values: iter.into_iter().collect(),
        }
    }

    /// Returns true if there are no args.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    /// Returns the number of args.
    #[inline]
    pub fn len(&self) -> usize {
        self.values.len()
    }

    /// Returns the args as a slice.
    #[inline]
    pub fn as_slice(&self) -> &[Value] {
        &self.values
    }

    /// Returns the first arg, if any.
    #[inline]
    pub fn first(&self) -> Option<&Value> {
        self.values.first()
    }
}

impl Default for ExceptionArgs {
    fn default() -> Self {
        Self::empty()
    }
}

impl fmt::Debug for ExceptionArgs {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("ExceptionArgs")
            .field(&self.values.len())
            .finish()
    }
}

// ============================================================================
// Exception Reference
// ============================================================================

/// Reference to an exception object.
///
/// This is the handle used for exception chaining (__cause__, __context__).
pub type ExceptionRef = Arc<ExceptionObject>;

// ============================================================================
// Exception Object
// ============================================================================

/// Python exception object.
///
/// Represents an active exception instance with lazy args and traceback.
/// The design minimizes allocations for common cases like StopIteration.
pub struct ExceptionObject {
    // ═══════════════════════════════════════════════════════════════════════
    // Compact Header (2 bytes total)
    // ═══════════════════════════════════════════════════════════════════════
    /// Exception type identifier.
    type_id: ExceptionTypeId,

    /// Exception state flags.
    flags: ExceptionFlags,

    // ═══════════════════════════════════════════════════════════════════════
    // Lazy Fields (allocated on demand)
    // ═══════════════════════════════════════════════════════════════════════
    /// Lazy exception arguments (the args tuple).
    args: OnceLock<ExceptionArgs>,

    /// Lazy traceback.
    traceback: OnceLock<TracebackObject>,

    /// Lazy message (first arg as string, or formatted).
    message: OnceLock<Arc<str>>,

    // ═══════════════════════════════════════════════════════════════════════
    // Exception Chaining (PEP 3134)
    // ═══════════════════════════════════════════════════════════════════════
    /// Explicit cause (`raise X from Y`).
    cause: Option<ExceptionRef>,

    /// Implicit context (exception that was being handled).
    context: Option<ExceptionRef>,
}

impl ExceptionObject {
    // ════════════════════════════════════════════════════════════════════════
    // Constructors
    // ════════════════════════════════════════════════════════════════════════

    /// Creates a new exception with just a type (no args, no traceback).
    ///
    /// This is the minimal allocation path for control-flow exceptions.
    #[inline]
    pub fn new(type_id: ExceptionTypeId) -> Self {
        Self {
            type_id,
            flags: ExceptionFlags::new_exception(),
            args: OnceLock::new(),
            traceback: OnceLock::new(),
            message: OnceLock::new(),
            cause: None,
            context: None,
        }
    }

    /// Creates an exception with a message.
    pub fn with_message(type_id: ExceptionTypeId, message: impl Into<Arc<str>>) -> Self {
        let msg = message.into();
        let mut exc = Self::new(type_id);
        exc.flags = exc.flags.set_normalized().set_has_args();
        exc.message.set(Arc::clone(&msg)).ok();
        exc.args.set(ExceptionArgs::single(Value::none())).ok(); // TODO: proper string value
        exc
    }

    /// Creates an exception with args.
    pub fn with_args(type_id: ExceptionTypeId, args: ExceptionArgs) -> Self {
        let mut exc = Self::new(type_id);
        exc.flags = exc.flags.set_normalized().set_has_args();
        exc.args.set(args).ok();
        exc
    }

    /// Creates a flyweight exception (singleton, no args).
    ///
    /// Used for StopIteration, GeneratorExit, etc.
    pub(super) fn flyweight(type_id: ExceptionTypeId) -> Self {
        Self {
            type_id,
            flags: ExceptionFlags::flyweight(),
            args: OnceLock::new(),
            traceback: OnceLock::new(),
            message: OnceLock::new(),
            cause: None,
            context: None,
        }
    }

    // ════════════════════════════════════════════════════════════════════════
    // Type Information
    // ════════════════════════════════════════════════════════════════════════

    /// Returns the exception type ID.
    #[inline(always)]
    pub const fn type_id(&self) -> ExceptionTypeId {
        self.type_id
    }

    /// Returns the exception type name.
    #[inline]
    pub const fn type_name(&self) -> &'static str {
        self.type_id.name()
    }

    /// Returns true if this exception is an instance of the given type.
    #[inline]
    pub fn is_instance(&self, type_id: ExceptionTypeId) -> bool {
        self.type_id == type_id
    }

    /// Returns true if this exception is a subclass of the given type.
    #[inline]
    pub fn is_subclass(&self, base: ExceptionTypeId) -> bool {
        self.type_id.is_subclass_of(base)
    }

    // ════════════════════════════════════════════════════════════════════════
    // Flags
    // ════════════════════════════════════════════════════════════════════════

    /// Returns the exception flags.
    #[inline(always)]
    pub const fn flags(&self) -> ExceptionFlags {
        self.flags
    }

    /// Returns true if this is a flyweight exception.
    #[inline]
    pub const fn is_flyweight(&self) -> bool {
        self.flags.is_flyweight()
    }

    /// Returns true if the exception has been normalized.
    #[inline]
    pub const fn is_normalized(&self) -> bool {
        self.flags.is_normalized()
    }

    // ════════════════════════════════════════════════════════════════════════
    // Args Access (Lazy)
    // ════════════════════════════════════════════════════════════════════════

    /// Returns the exception args, if set.
    #[inline]
    pub fn args(&self) -> Option<&ExceptionArgs> {
        self.args.get()
    }

    /// Returns the exception args, initializing to empty if needed.
    pub fn args_or_empty(&self) -> &ExceptionArgs {
        self.args.get_or_init(ExceptionArgs::empty)
    }

    /// Sets the exception args.
    ///
    /// Returns Err if args were already set.
    pub fn set_args(&self, args: ExceptionArgs) -> Result<(), ExceptionArgs> {
        self.args.set(args)
    }

    // ════════════════════════════════════════════════════════════════════════
    // Traceback Access (Lazy)
    // ════════════════════════════════════════════════════════════════════════

    /// Returns the traceback, if set.
    #[inline]
    pub fn traceback(&self) -> Option<&TracebackObject> {
        self.traceback.get()
    }

    /// Returns a mutable reference to the traceback, if set.
    #[inline]
    pub fn traceback_mut(&mut self) -> Option<&mut TracebackObject> {
        self.traceback.get_mut()
    }

    /// Returns the traceback, initializing to empty if needed.
    pub fn traceback_or_empty(&self) -> &TracebackObject {
        self.traceback.get_or_init(TracebackObject::empty)
    }

    /// Sets the traceback.
    pub fn set_traceback(&self, tb: TracebackObject) -> Result<(), TracebackObject> {
        self.traceback.set(tb)
    }

    /// Returns true if the exception has a traceback.
    #[inline]
    pub fn has_traceback(&self) -> bool {
        self.traceback.get().is_some()
    }

    // ════════════════════════════════════════════════════════════════════════
    // Message Access (Lazy)
    // ════════════════════════════════════════════════════════════════════════

    /// Returns the exception message.
    pub fn message(&self) -> Arc<str> {
        self.message
            .get_or_init(|| {
                // Try to get message from first arg
                if let Some(args) = self.args.get() {
                    if let Some(_first) = args.first() {
                        // TODO: Convert first arg to string
                        return Arc::from("");
                    }
                }
                Arc::from("")
            })
            .clone()
    }

    // ════════════════════════════════════════════════════════════════════════
    // Exception Chaining (PEP 3134)
    // ════════════════════════════════════════════════════════════════════════

    /// Returns the explicit cause (__cause__).
    #[inline]
    pub fn cause(&self) -> Option<&ExceptionRef> {
        self.cause.as_ref()
    }

    /// Sets the explicit cause (`raise X from Y`).
    pub fn set_cause(&mut self, cause: ExceptionRef) {
        self.cause = Some(cause);
        self.flags = self.flags.set_has_cause().set_suppress_context();
    }

    /// Clears the explicit cause.
    pub fn clear_cause(&mut self) {
        self.cause = None;
        self.flags = self.flags.clear_has_cause().clear_suppress_context();
    }

    /// Returns the implicit context (__context__).
    #[inline]
    pub fn context(&self) -> Option<&ExceptionRef> {
        self.context.as_ref()
    }

    /// Sets the implicit context.
    pub fn set_context(&mut self, context: ExceptionRef) {
        self.context = Some(context);
        self.flags = self.flags.set_has_context();
    }

    /// Returns true if __suppress_context__ is True.
    #[inline]
    pub const fn suppress_context(&self) -> bool {
        self.flags.suppress_context()
    }

    // ════════════════════════════════════════════════════════════════════════
    // Formatting
    // ════════════════════════════════════════════════════════════════════════

    /// Formats the exception as a string.
    pub fn format(&self) -> String {
        let msg = self.message();
        if msg.is_empty() {
            self.type_name().to_string()
        } else {
            format!("{}: {}", self.type_name(), msg)
        }
    }

    /// Formats the full exception with traceback.
    pub fn format_with_traceback(&mut self) -> String {
        let mut output = String::new();

        if let Some(tb) = self.traceback_mut() {
            output.push_str(&tb.format());
        }

        output.push_str(&self.format());
        output.push('\n');
        output
    }
}

impl fmt::Debug for ExceptionObject {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ExceptionObject")
            .field("type_id", &self.type_id)
            .field("flags", &self.flags)
            .field("has_args", &self.args.get().is_some())
            .field("has_traceback", &self.traceback.get().is_some())
            .field("has_cause", &self.cause.is_some())
            .field("has_context", &self.context.is_some())
            .finish()
    }
}

impl fmt::Display for ExceptionObject {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.format())
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::stdlib::exceptions::traceback::FrameInfo;

    // ════════════════════════════════════════════════════════════════════════
    // ExceptionArgs Tests
    // ════════════════════════════════════════════════════════════════════════

    #[test]
    fn test_args_empty() {
        let args = ExceptionArgs::empty();
        assert!(args.is_empty());
        assert_eq!(args.len(), 0);
    }

    #[test]
    fn test_args_single() {
        let args = ExceptionArgs::single(Value::none());
        assert!(!args.is_empty());
        assert_eq!(args.len(), 1);
        assert!(args.first().is_some());
    }

    #[test]
    fn test_args_from_slice() {
        let values = [Value::none(), Value::none()];
        let args = ExceptionArgs::from_slice(&values);
        assert_eq!(args.len(), 2);
    }

    #[test]
    fn test_args_from_iter() {
        let args = ExceptionArgs::from_iter(vec![Value::none(), Value::none(), Value::none()]);
        assert_eq!(args.len(), 3);
    }

    #[test]
    fn test_args_as_slice() {
        let args = ExceptionArgs::from_slice(&[Value::none(), Value::none()]);
        assert_eq!(args.as_slice().len(), 2);
    }

    #[test]
    fn test_args_default() {
        let args = ExceptionArgs::default();
        assert!(args.is_empty());
    }

    // ════════════════════════════════════════════════════════════════════════
    // ExceptionObject Creation Tests
    // ════════════════════════════════════════════════════════════════════════

    #[test]
    fn test_exception_new() {
        let exc = ExceptionObject::new(ExceptionTypeId::TypeError);
        assert_eq!(exc.type_id(), ExceptionTypeId::TypeError);
        assert!(!exc.is_normalized());
        assert!(!exc.is_flyweight());
    }

    #[test]
    fn test_exception_with_message() {
        let exc = ExceptionObject::with_message(ExceptionTypeId::ValueError, "invalid value");
        assert_eq!(exc.type_id(), ExceptionTypeId::ValueError);
        assert!(exc.is_normalized());
        assert!(exc.args().is_some());
    }

    #[test]
    fn test_exception_with_args() {
        let args = ExceptionArgs::from_slice(&[Value::none()]);
        let exc = ExceptionObject::with_args(ExceptionTypeId::KeyError, args);
        assert!(exc.is_normalized());
        assert!(!exc.args_or_empty().is_empty());
    }

    #[test]
    fn test_exception_flyweight() {
        let exc = ExceptionObject::flyweight(ExceptionTypeId::StopIteration);
        assert!(exc.is_flyweight());
        assert!(exc.is_normalized());
    }

    // ════════════════════════════════════════════════════════════════════════
    // Type Information Tests
    // ════════════════════════════════════════════════════════════════════════

    #[test]
    fn test_exception_type_name() {
        let exc = ExceptionObject::new(ExceptionTypeId::IndexError);
        assert_eq!(exc.type_name(), "IndexError");
    }

    #[test]
    fn test_exception_is_instance() {
        let exc = ExceptionObject::new(ExceptionTypeId::TypeError);
        assert!(exc.is_instance(ExceptionTypeId::TypeError));
        assert!(!exc.is_instance(ExceptionTypeId::ValueError));
    }

    #[test]
    fn test_exception_is_subclass() {
        let exc = ExceptionObject::new(ExceptionTypeId::FileNotFoundError);
        assert!(exc.is_subclass(ExceptionTypeId::OSError));
        assert!(exc.is_subclass(ExceptionTypeId::Exception));
        assert!(!exc.is_subclass(ExceptionTypeId::TypeError));
    }

    // ════════════════════════════════════════════════════════════════════════
    // Args Access Tests
    // ════════════════════════════════════════════════════════════════════════

    #[test]
    fn test_exception_args_none() {
        let exc = ExceptionObject::new(ExceptionTypeId::TypeError);
        assert!(exc.args().is_none());
    }

    #[test]
    fn test_exception_args_or_empty() {
        let exc = ExceptionObject::new(ExceptionTypeId::TypeError);
        let args = exc.args_or_empty();
        assert!(args.is_empty());
    }

    #[test]
    fn test_exception_set_args() {
        let exc = ExceptionObject::new(ExceptionTypeId::TypeError);
        let args = ExceptionArgs::single(Value::none());
        assert!(exc.set_args(args).is_ok());
        assert!(exc.args().is_some());
    }

    #[test]
    fn test_exception_set_args_twice_fails() {
        let exc = ExceptionObject::new(ExceptionTypeId::TypeError);
        let args1 = ExceptionArgs::single(Value::none());
        let args2 = ExceptionArgs::single(Value::none());

        assert!(exc.set_args(args1).is_ok());
        assert!(exc.set_args(args2).is_err());
    }

    // ════════════════════════════════════════════════════════════════════════
    // Traceback Access Tests
    // ════════════════════════════════════════════════════════════════════════

    #[test]
    fn test_exception_traceback_none() {
        let exc = ExceptionObject::new(ExceptionTypeId::TypeError);
        assert!(exc.traceback().is_none());
        assert!(!exc.has_traceback());
    }

    #[test]
    fn test_exception_traceback_or_empty() {
        let exc = ExceptionObject::new(ExceptionTypeId::TypeError);
        let tb = exc.traceback_or_empty();
        assert!(tb.is_empty());
    }

    #[test]
    fn test_exception_set_traceback() {
        let exc = ExceptionObject::new(ExceptionTypeId::TypeError);
        let tb =
            TracebackObject::single(FrameInfo::new(Arc::from("test"), Arc::from("test.py"), 1));

        assert!(exc.set_traceback(tb).is_ok());
        assert!(exc.has_traceback());
    }

    #[test]
    fn test_exception_traceback_mut() {
        let mut exc = ExceptionObject::new(ExceptionTypeId::TypeError);
        let tb =
            TracebackObject::single(FrameInfo::new(Arc::from("test"), Arc::from("test.py"), 1));
        exc.set_traceback(tb).ok();

        // Should be able to mutate
        let tb_mut = exc.traceback_mut().unwrap();
        tb_mut.push(FrameInfo::new(Arc::from("test2"), Arc::from("test2.py"), 2));

        assert_eq!(exc.traceback().unwrap().len(), 2);
    }

    // ════════════════════════════════════════════════════════════════════════
    // Exception Chaining Tests
    // ════════════════════════════════════════════════════════════════════════

    #[test]
    fn test_exception_no_chaining() {
        let exc = ExceptionObject::new(ExceptionTypeId::TypeError);
        assert!(exc.cause().is_none());
        assert!(exc.context().is_none());
    }

    #[test]
    fn test_exception_set_cause() {
        let mut exc = ExceptionObject::new(ExceptionTypeId::RuntimeError);
        let cause = Arc::new(ExceptionObject::new(ExceptionTypeId::ValueError));

        exc.set_cause(cause);
        assert!(exc.cause().is_some());
        assert!(exc.flags().has_cause());
        assert!(exc.suppress_context()); // Should be set when cause is set
    }

    #[test]
    fn test_exception_clear_cause() {
        let mut exc = ExceptionObject::new(ExceptionTypeId::RuntimeError);
        let cause = Arc::new(ExceptionObject::new(ExceptionTypeId::ValueError));

        exc.set_cause(cause);
        assert!(exc.cause().is_some());

        exc.clear_cause();
        assert!(exc.cause().is_none());
        assert!(!exc.flags().has_cause());
    }

    #[test]
    fn test_exception_set_context() {
        let mut exc = ExceptionObject::new(ExceptionTypeId::RuntimeError);
        let context = Arc::new(ExceptionObject::new(ExceptionTypeId::ValueError));

        exc.set_context(context);
        assert!(exc.context().is_some());
        assert!(exc.flags().has_context());
    }

    // ════════════════════════════════════════════════════════════════════════
    // Formatting Tests
    // ════════════════════════════════════════════════════════════════════════

    #[test]
    fn test_exception_format_no_message() {
        let exc = ExceptionObject::new(ExceptionTypeId::TypeError);
        let formatted = exc.format();
        assert_eq!(formatted, "TypeError");
    }

    #[test]
    fn test_exception_display() {
        let exc = ExceptionObject::new(ExceptionTypeId::ValueError);
        let display = format!("{}", exc);
        assert!(display.contains("ValueError"));
    }

    #[test]
    fn test_exception_debug() {
        let exc = ExceptionObject::new(ExceptionTypeId::KeyError);
        let debug = format!("{:?}", exc);
        assert!(debug.contains("ExceptionObject"));
        assert!(debug.contains("KeyError"));
    }

    #[test]
    fn test_exception_format_with_traceback() {
        let mut exc = ExceptionObject::new(ExceptionTypeId::TypeError);
        let tb =
            TracebackObject::single(FrameInfo::new(Arc::from("test"), Arc::from("test.py"), 42));
        exc.set_traceback(tb).ok();

        let formatted = exc.format_with_traceback();
        assert!(formatted.contains("Traceback"));
        assert!(formatted.contains("test.py"));
        assert!(formatted.contains("TypeError"));
    }

    // ════════════════════════════════════════════════════════════════════════
    // Memory Layout Tests
    // ════════════════════════════════════════════════════════════════════════

    #[test]
    fn test_exception_header_size() {
        // Type ID + Flags = 2 bytes
        assert_eq!(
            std::mem::size_of::<ExceptionTypeId>() + std::mem::size_of::<ExceptionFlags>(),
            2
        );
    }

    // ════════════════════════════════════════════════════════════════════════
    // Control Flow Exception Tests
    // ════════════════════════════════════════════════════════════════════════

    #[test]
    fn test_stop_iteration_is_control_flow() {
        let exc = ExceptionObject::new(ExceptionTypeId::StopIteration);
        assert!(exc.type_id().is_control_flow());
    }

    #[test]
    fn test_generator_exit_is_control_flow() {
        let exc = ExceptionObject::new(ExceptionTypeId::GeneratorExit);
        assert!(exc.type_id().is_control_flow());
    }

    // ════════════════════════════════════════════════════════════════════════
    // Arc Support Tests
    // ════════════════════════════════════════════════════════════════════════

    #[test]
    fn test_exception_ref() {
        let exc: ExceptionRef = Arc::new(ExceptionObject::new(ExceptionTypeId::TypeError));
        assert_eq!(exc.type_id(), ExceptionTypeId::TypeError);
    }

    #[test]
    fn test_exception_ref_clone() {
        let exc: ExceptionRef = Arc::new(ExceptionObject::new(ExceptionTypeId::TypeError));
        let cloned = Arc::clone(&exc);
        assert!(Arc::ptr_eq(&exc, &cloned));
    }
}
