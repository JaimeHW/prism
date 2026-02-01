//! Error types and result definitions for Prism.
//!
//! This module provides a comprehensive error hierarchy covering all phases of execution:
//! - Lexical errors (tokenization failures)
//! - Syntax errors (parsing failures)
//! - Compilation errors (bytecode generation failures)
//! - Runtime errors (execution failures)
//! - Type errors (dynamic type mismatches)

use crate::span::Span;
use std::fmt;
use thiserror::Error;

/// The unified result type used throughout Prism.
pub type PrismResult<T> = Result<T, PrismError>;

/// Comprehensive error type covering all Prism error conditions.
#[derive(Error, Debug, Clone)]
pub enum PrismError {
    /// Lexical analysis error.
    #[error("SyntaxError: {message}")]
    LexError {
        /// Error description.
        message: String,
        /// Source location.
        span: Span,
    },

    /// Syntax/parsing error.
    #[error("SyntaxError: {message}")]
    SyntaxError {
        /// Error description.
        message: String,
        /// Source location.
        span: Span,
    },

    /// Compilation error.
    #[error("CompileError: {message}")]
    CompileError {
        /// Error description.
        message: String,
        /// Source location.
        span: Option<Span>,
    },

    /// Runtime error during execution.
    #[error("{kind}: {message}")]
    RuntimeError {
        /// The Python exception type name.
        kind: RuntimeErrorKind,
        /// Error description.
        message: String,
    },

    /// Name not found in scope.
    #[error("NameError: name '{name}' is not defined")]
    NameError {
        /// The undefined name.
        name: String,
    },

    /// Type mismatch error.
    #[error("TypeError: {message}")]
    TypeError {
        /// Error description.
        message: String,
    },

    /// Value error.
    #[error("ValueError: {message}")]
    ValueError {
        /// Error description.
        message: String,
    },

    /// Attribute access error.
    #[error("AttributeError: {message}")]
    AttributeError {
        /// Error description.
        message: String,
    },

    /// Index out of bounds.
    #[error("IndexError: {message}")]
    IndexError {
        /// Error description.
        message: String,
    },

    /// Key not found in mapping.
    #[error("KeyError: {key}")]
    KeyError {
        /// The missing key representation.
        key: String,
    },

    /// Zero division.
    #[error("ZeroDivisionError: {message}")]
    ZeroDivisionError {
        /// Error description.
        message: String,
    },

    /// Import failure.
    #[error("ImportError: {message}")]
    ImportError {
        /// Error description.
        message: String,
    },

    /// Assertion failure.
    #[error("AssertionError: {message}")]
    AssertionError {
        /// Error description.
        message: String,
    },

    /// Stop iteration signal (not actually an error in normal flow).
    #[error("StopIteration")]
    StopIteration,

    /// Overflow error.
    #[error("OverflowError: {message}")]
    OverflowError {
        /// Error description.
        message: String,
    },

    /// Recursion limit exceeded.
    #[error("RecursionError: maximum recursion depth exceeded")]
    RecursionError,

    /// Memory allocation failure.
    #[error("MemoryError: {message}")]
    MemoryError {
        /// Error description.
        message: String,
    },

    /// Internal VM error (should never occur in correct implementation).
    #[error("InternalError: {message}")]
    InternalError {
        /// Error description.
        message: String,
    },
}

impl PrismError {
    /// Create a lex error with location.
    #[must_use]
    pub fn lex(message: impl Into<String>, span: Span) -> Self {
        Self::LexError {
            message: message.into(),
            span,
        }
    }

    /// Create a syntax error with location.
    #[must_use]
    pub fn syntax(message: impl Into<String>, span: Span) -> Self {
        Self::SyntaxError {
            message: message.into(),
            span,
        }
    }

    /// Create a compile error.
    #[must_use]
    pub fn compile(message: impl Into<String>, span: Option<Span>) -> Self {
        Self::CompileError {
            message: message.into(),
            span,
        }
    }

    /// Create a runtime error with kind.
    #[must_use]
    pub fn runtime(kind: RuntimeErrorKind, message: impl Into<String>) -> Self {
        Self::RuntimeError {
            kind,
            message: message.into(),
        }
    }

    /// Create a name error.
    #[must_use]
    pub fn name(name: impl Into<String>) -> Self {
        Self::NameError { name: name.into() }
    }

    /// Create a type error.
    #[must_use]
    pub fn type_error(message: impl Into<String>) -> Self {
        Self::TypeError {
            message: message.into(),
        }
    }

    /// Create a value error.
    #[must_use]
    pub fn value_error(message: impl Into<String>) -> Self {
        Self::ValueError {
            message: message.into(),
        }
    }

    /// Create an attribute error.
    #[must_use]
    pub fn attribute(message: impl Into<String>) -> Self {
        Self::AttributeError {
            message: message.into(),
        }
    }

    /// Create an index error.
    #[must_use]
    pub fn index(message: impl Into<String>) -> Self {
        Self::IndexError {
            message: message.into(),
        }
    }

    /// Create a key error.
    #[must_use]
    pub fn key(key: impl Into<String>) -> Self {
        Self::KeyError { key: key.into() }
    }

    /// Create a zero division error.
    #[must_use]
    pub fn zero_division(message: impl Into<String>) -> Self {
        Self::ZeroDivisionError {
            message: message.into(),
        }
    }

    /// Create an import error.
    #[must_use]
    pub fn import(message: impl Into<String>) -> Self {
        Self::ImportError {
            message: message.into(),
        }
    }

    /// Create an assertion error.
    #[must_use]
    pub fn assertion(message: impl Into<String>) -> Self {
        Self::AssertionError {
            message: message.into(),
        }
    }

    /// Create an internal error.
    #[must_use]
    pub fn internal(message: impl Into<String>) -> Self {
        Self::InternalError {
            message: message.into(),
        }
    }

    /// Get the Python exception type name.
    #[must_use]
    pub fn exception_type(&self) -> &'static str {
        match self {
            Self::LexError { .. } | Self::SyntaxError { .. } => "SyntaxError",
            Self::CompileError { .. } => "SyntaxError",
            Self::RuntimeError { kind, .. } => kind.as_str(),
            Self::NameError { .. } => "NameError",
            Self::TypeError { .. } => "TypeError",
            Self::ValueError { .. } => "ValueError",
            Self::AttributeError { .. } => "AttributeError",
            Self::IndexError { .. } => "IndexError",
            Self::KeyError { .. } => "KeyError",
            Self::ZeroDivisionError { .. } => "ZeroDivisionError",
            Self::ImportError { .. } => "ImportError",
            Self::AssertionError { .. } => "AssertionError",
            Self::StopIteration => "StopIteration",
            Self::OverflowError { .. } => "OverflowError",
            Self::RecursionError => "RecursionError",
            Self::MemoryError { .. } => "MemoryError",
            Self::InternalError { .. } => "SystemError",
        }
    }
}

/// Runtime error classification matching Python's exception hierarchy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RuntimeErrorKind {
    /// Generic runtime error.
    Runtime,
    /// Exception raised by user code.
    Exception,
    /// System exit requested.
    SystemExit,
    /// Keyboard interrupt.
    KeyboardInterrupt,
    /// Generator exit.
    GeneratorExit,
}

impl RuntimeErrorKind {
    /// Get the Python exception type name.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Runtime => "RuntimeError",
            Self::Exception => "Exception",
            Self::SystemExit => "SystemExit",
            Self::KeyboardInterrupt => "KeyboardInterrupt",
            Self::GeneratorExit => "GeneratorExit",
        }
    }
}

impl fmt::Display for RuntimeErrorKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lex_error_creation() {
        let span = Span::new(0, 10);
        let err = PrismError::lex("unexpected character", span);

        match &err {
            PrismError::LexError { message, span: s } => {
                assert_eq!(message, "unexpected character");
                assert_eq!(s.start, 0);
                assert_eq!(s.end, 10);
            }
            _ => panic!("Expected LexError"),
        }

        assert_eq!(err.exception_type(), "SyntaxError");
        assert_eq!(err.to_string(), "SyntaxError: unexpected character");
    }

    #[test]
    fn test_syntax_error_creation() {
        let span = Span::new(5, 15);
        let err = PrismError::syntax("invalid syntax", span);

        assert_eq!(err.exception_type(), "SyntaxError");
        assert_eq!(err.to_string(), "SyntaxError: invalid syntax");
    }

    #[test]
    fn test_name_error_creation() {
        let err = PrismError::name("undefined_var");

        match &err {
            PrismError::NameError { name } => {
                assert_eq!(name, "undefined_var");
            }
            _ => panic!("Expected NameError"),
        }

        assert_eq!(err.exception_type(), "NameError");
        assert_eq!(
            err.to_string(),
            "NameError: name 'undefined_var' is not defined"
        );
    }

    #[test]
    fn test_type_error_creation() {
        let err = PrismError::type_error("unsupported operand type(s) for +: 'int' and 'str'");

        assert_eq!(err.exception_type(), "TypeError");
        assert!(err.to_string().contains("unsupported operand type"));
    }

    #[test]
    fn test_value_error_creation() {
        let err = PrismError::value_error("invalid literal for int()");

        assert_eq!(err.exception_type(), "ValueError");
        assert!(err.to_string().contains("invalid literal"));
    }

    #[test]
    fn test_attribute_error_creation() {
        let err = PrismError::attribute("'int' object has no attribute 'foo'");

        assert_eq!(err.exception_type(), "AttributeError");
    }

    #[test]
    fn test_index_error_creation() {
        let err = PrismError::index("list index out of range");

        assert_eq!(err.exception_type(), "IndexError");
        assert_eq!(err.to_string(), "IndexError: list index out of range");
    }

    #[test]
    fn test_key_error_creation() {
        let err = PrismError::key("missing_key");

        assert_eq!(err.exception_type(), "KeyError");
        assert_eq!(err.to_string(), "KeyError: missing_key");
    }

    #[test]
    fn test_zero_division_error_creation() {
        let err = PrismError::zero_division("division by zero");

        assert_eq!(err.exception_type(), "ZeroDivisionError");
        assert_eq!(err.to_string(), "ZeroDivisionError: division by zero");
    }

    #[test]
    fn test_import_error_creation() {
        let err = PrismError::import("No module named 'nonexistent'");

        assert_eq!(err.exception_type(), "ImportError");
    }

    #[test]
    fn test_assertion_error_creation() {
        let err = PrismError::assertion("expected true");

        assert_eq!(err.exception_type(), "AssertionError");
        assert_eq!(err.to_string(), "AssertionError: expected true");
    }

    #[test]
    fn test_stop_iteration() {
        let err = PrismError::StopIteration;

        assert_eq!(err.exception_type(), "StopIteration");
        assert_eq!(err.to_string(), "StopIteration");
    }

    #[test]
    fn test_recursion_error() {
        let err = PrismError::RecursionError;

        assert_eq!(err.exception_type(), "RecursionError");
        assert!(err.to_string().contains("recursion depth"));
    }

    #[test]
    fn test_overflow_error() {
        let err = PrismError::OverflowError {
            message: "int too large".into(),
        };

        assert_eq!(err.exception_type(), "OverflowError");
    }

    #[test]
    fn test_memory_error() {
        let err = PrismError::MemoryError {
            message: "allocation failed".into(),
        };

        assert_eq!(err.exception_type(), "MemoryError");
    }

    #[test]
    fn test_internal_error_creation() {
        let err = PrismError::internal("VM stack corruption");

        assert_eq!(err.exception_type(), "SystemError");
        assert_eq!(err.to_string(), "InternalError: VM stack corruption");
    }

    #[test]
    fn test_compile_error_with_span() {
        let span = Span::new(100, 120);
        let err = PrismError::compile("cannot assign to literal", Some(span));

        match &err {
            PrismError::CompileError { message, span: s } => {
                assert_eq!(message, "cannot assign to literal");
                assert!(s.is_some());
                let s = s.unwrap();
                assert_eq!(s.start, 100);
                assert_eq!(s.end, 120);
            }
            _ => panic!("Expected CompileError"),
        }
    }

    #[test]
    fn test_compile_error_without_span() {
        let err = PrismError::compile("unknown error", None);

        match &err {
            PrismError::CompileError { span, .. } => {
                assert!(span.is_none());
            }
            _ => panic!("Expected CompileError"),
        }
    }

    #[test]
    fn test_runtime_error_kind_display() {
        assert_eq!(RuntimeErrorKind::Runtime.to_string(), "RuntimeError");
        assert_eq!(RuntimeErrorKind::Exception.to_string(), "Exception");
        assert_eq!(RuntimeErrorKind::SystemExit.to_string(), "SystemExit");
        assert_eq!(
            RuntimeErrorKind::KeyboardInterrupt.to_string(),
            "KeyboardInterrupt"
        );
        assert_eq!(RuntimeErrorKind::GeneratorExit.to_string(), "GeneratorExit");
    }

    #[test]
    fn test_runtime_error_with_kind() {
        let err = PrismError::runtime(RuntimeErrorKind::SystemExit, "exit code 1");

        assert_eq!(err.exception_type(), "SystemExit");
        assert_eq!(err.to_string(), "SystemExit: exit code 1");
    }

    #[test]
    fn test_error_is_clone() {
        let original = PrismError::name("test");
        let cloned = original.clone();

        match (&original, &cloned) {
            (PrismError::NameError { name: n1 }, PrismError::NameError { name: n2 }) => {
                assert_eq!(n1, n2);
            }
            _ => panic!("Clone mismatch"),
        }
    }

    #[test]
    fn test_error_is_debug() {
        let err = PrismError::type_error("test");
        let debug_str = format!("{:?}", err);
        assert!(debug_str.contains("TypeError"));
    }

    #[test]
    fn test_prism_result_ok() {
        let result: PrismResult<i32> = Ok(42);
        assert_eq!(result.unwrap(), 42);
    }

    #[test]
    fn test_prism_result_err() {
        let result: PrismResult<i32> = Err(PrismError::name("x"));
        assert!(result.is_err());
    }

    #[test]
    fn test_runtime_error_kind_equality() {
        assert_eq!(RuntimeErrorKind::Runtime, RuntimeErrorKind::Runtime);
        assert_ne!(RuntimeErrorKind::Runtime, RuntimeErrorKind::Exception);
    }

    #[test]
    fn test_runtime_error_kind_hash() {
        use std::collections::HashSet;
        let mut set = HashSet::new();
        set.insert(RuntimeErrorKind::Runtime);
        set.insert(RuntimeErrorKind::Exception);
        assert_eq!(set.len(), 2);
        assert!(set.contains(&RuntimeErrorKind::Runtime));
    }
}
