//! Runtime error types for the virtual machine.
//!
//! This module defines all error types that can occur during bytecode execution.
//! Errors are designed for minimal allocation and fast construction on hot paths.

use prism_core::{PrismError, Span};
use std::fmt;
use std::sync::Arc;

/// Runtime error during bytecode execution.
#[derive(Debug, Clone)]
pub struct RuntimeError {
    /// Error kind
    pub kind: RuntimeErrorKind,
    /// Source location (if available)
    pub span: Option<Span>,
    /// Traceback frames
    pub traceback: Vec<TracebackEntry>,
}

/// Specific runtime error types.
#[derive(Debug, Clone)]
pub enum RuntimeErrorKind {
    // =========================================================================
    // Type Errors
    // =========================================================================
    /// Operation not supported for types
    TypeError { message: Arc<str> },
    /// Unsupported operand types for binary operation
    UnsupportedOperandTypes {
        op: &'static str,
        left: Arc<str>,
        right: Arc<str>,
    },
    /// Object is not callable
    NotCallable { type_name: Arc<str> },
    /// Object is not iterable
    NotIterable { type_name: Arc<str> },
    /// Object is not subscriptable
    NotSubscriptable { type_name: Arc<str> },

    // =========================================================================
    // Name/Attribute Errors
    // =========================================================================
    /// Name not defined
    NameError { name: Arc<str> },
    /// Attribute not found
    AttributeError { type_name: Arc<str>, attr: Arc<str> },
    /// Local variable referenced before assignment
    UnboundLocalError { name: Arc<str> },

    // =========================================================================
    // Index/Key Errors
    // =========================================================================
    /// Index out of range
    IndexError { index: i64, length: usize },
    /// Key not found in dict
    KeyError { key: Arc<str> },

    // =========================================================================
    // Value Errors
    // =========================================================================
    /// Invalid value for operation
    ValueError { message: Arc<str> },
    /// Division by zero
    ZeroDivisionError,
    /// Overflow in arithmetic
    OverflowError { message: Arc<str> },

    // =========================================================================
    // Control Flow
    // =========================================================================
    /// StopIteration (not really an error)
    StopIteration,
    /// Generator return
    GeneratorExit,
    /// Assertion failed
    AssertionError { message: Option<Arc<str>> },

    // =========================================================================
    // System Errors
    // =========================================================================
    /// Recursion limit exceeded
    RecursionError { depth: usize },
    /// Invalid opcode
    InvalidOpcode { opcode: u8 },
    /// Internal VM error (should not happen)
    InternalError { message: Arc<str> },
    /// Import error
    ImportError { module: Arc<str>, message: Arc<str> },
    /// Python exception raised
    Exception { type_id: u16, message: Arc<str> },
}

/// A single entry in the traceback.
#[derive(Debug, Clone)]
pub struct TracebackEntry {
    /// Function name
    pub func_name: Arc<str>,
    /// File name
    pub filename: Arc<str>,
    /// Line number
    pub line: u32,
}

impl RuntimeError {
    /// Create a new runtime error.
    #[inline]
    pub fn new(kind: RuntimeErrorKind) -> Self {
        Self {
            kind,
            span: None,
            traceback: Vec::new(),
        }
    }

    /// Create with span information.
    #[inline]
    pub fn with_span(kind: RuntimeErrorKind, span: Span) -> Self {
        Self {
            kind,
            span: Some(span),
            traceback: Vec::new(),
        }
    }

    /// Add a traceback entry.
    #[inline]
    pub fn add_traceback(&mut self, entry: TracebackEntry) {
        self.traceback.push(entry);
    }

    // =========================================================================
    // Convenience Constructors
    // =========================================================================

    #[inline]
    pub fn type_error(message: impl Into<Arc<str>>) -> Self {
        Self::new(RuntimeErrorKind::TypeError {
            message: message.into(),
        })
    }

    #[inline]
    pub fn unsupported_operand(op: &'static str, left: &str, right: &str) -> Self {
        Self::new(RuntimeErrorKind::UnsupportedOperandTypes {
            op,
            left: left.into(),
            right: right.into(),
        })
    }

    #[inline]
    pub fn not_callable(type_name: impl Into<Arc<str>>) -> Self {
        Self::new(RuntimeErrorKind::NotCallable {
            type_name: type_name.into(),
        })
    }

    #[inline]
    pub fn name_error(name: impl Into<Arc<str>>) -> Self {
        Self::new(RuntimeErrorKind::NameError { name: name.into() })
    }

    #[inline]
    pub fn attribute_error(type_name: impl Into<Arc<str>>, attr: impl Into<Arc<str>>) -> Self {
        Self::new(RuntimeErrorKind::AttributeError {
            type_name: type_name.into(),
            attr: attr.into(),
        })
    }

    #[inline]
    pub fn index_error(index: i64, length: usize) -> Self {
        Self::new(RuntimeErrorKind::IndexError { index, length })
    }

    #[inline]
    pub fn key_error(key: impl Into<Arc<str>>) -> Self {
        Self::new(RuntimeErrorKind::KeyError { key: key.into() })
    }

    #[inline]
    pub fn value_error(message: impl Into<Arc<str>>) -> Self {
        Self::new(RuntimeErrorKind::ValueError {
            message: message.into(),
        })
    }

    #[inline]
    pub fn zero_division() -> Self {
        Self::new(RuntimeErrorKind::ZeroDivisionError)
    }

    #[inline]
    pub fn recursion_error(depth: usize) -> Self {
        Self::new(RuntimeErrorKind::RecursionError { depth })
    }

    #[inline]
    pub fn stop_iteration() -> Self {
        Self::new(RuntimeErrorKind::StopIteration)
    }

    #[inline]
    pub fn invalid_opcode(opcode: u8) -> Self {
        Self::new(RuntimeErrorKind::InvalidOpcode { opcode })
    }

    #[inline]
    pub fn internal(message: impl Into<Arc<str>>) -> Self {
        Self::new(RuntimeErrorKind::InternalError {
            message: message.into(),
        })
    }

    /// Create an exception error for propagation.
    #[inline]
    pub fn exception(type_id: u16, message: impl Into<Arc<str>>) -> Self {
        Self::new(RuntimeErrorKind::Exception {
            type_id,
            message: message.into(),
        })
    }
}

impl fmt::Display for RuntimeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.kind {
            RuntimeErrorKind::TypeError { message } => write!(f, "TypeError: {}", message),
            RuntimeErrorKind::UnsupportedOperandTypes { op, left, right } => {
                write!(
                    f,
                    "TypeError: unsupported operand type(s) for {}: '{}' and '{}'",
                    op, left, right
                )
            }
            RuntimeErrorKind::NotCallable { type_name } => {
                write!(f, "TypeError: '{}' object is not callable", type_name)
            }
            RuntimeErrorKind::NotIterable { type_name } => {
                write!(f, "TypeError: '{}' object is not iterable", type_name)
            }
            RuntimeErrorKind::NotSubscriptable { type_name } => {
                write!(f, "TypeError: '{}' object is not subscriptable", type_name)
            }
            RuntimeErrorKind::NameError { name } => {
                write!(f, "NameError: name '{}' is not defined", name)
            }
            RuntimeErrorKind::AttributeError { type_name, attr } => {
                write!(
                    f,
                    "AttributeError: '{}' object has no attribute '{}'",
                    type_name, attr
                )
            }
            RuntimeErrorKind::UnboundLocalError { name } => {
                write!(
                    f,
                    "UnboundLocalError: local variable '{}' referenced before assignment",
                    name
                )
            }
            RuntimeErrorKind::IndexError { index, length } => {
                write!(
                    f,
                    "IndexError: index {} out of range for sequence of length {}",
                    index, length
                )
            }
            RuntimeErrorKind::KeyError { key } => write!(f, "KeyError: '{}'", key),
            RuntimeErrorKind::ValueError { message } => write!(f, "ValueError: {}", message),
            RuntimeErrorKind::ZeroDivisionError => {
                write!(f, "ZeroDivisionError: division by zero")
            }
            RuntimeErrorKind::OverflowError { message } => {
                write!(f, "OverflowError: {}", message)
            }
            RuntimeErrorKind::StopIteration => write!(f, "StopIteration"),
            RuntimeErrorKind::GeneratorExit => write!(f, "GeneratorExit"),
            RuntimeErrorKind::AssertionError { message } => {
                if let Some(msg) = message {
                    write!(f, "AssertionError: {}", msg)
                } else {
                    write!(f, "AssertionError")
                }
            }
            RuntimeErrorKind::RecursionError { depth } => {
                write!(
                    f,
                    "RecursionError: maximum recursion depth exceeded ({})",
                    depth
                )
            }
            RuntimeErrorKind::InvalidOpcode { opcode } => {
                write!(f, "InternalError: invalid opcode 0x{:02x}", opcode)
            }
            RuntimeErrorKind::InternalError { message } => {
                write!(f, "InternalError: {}", message)
            }
            RuntimeErrorKind::ImportError { module, message } => {
                write!(f, "ImportError: cannot import '{}': {}", module, message)
            }
            RuntimeErrorKind::Exception { type_id, message } => {
                write!(f, "Exception(type_id={}): {}", type_id, message)
            }
        }
    }
}

impl std::error::Error for RuntimeError {}

impl From<RuntimeError> for PrismError {
    fn from(err: RuntimeError) -> Self {
        match &err.kind {
            RuntimeErrorKind::TypeError { message } => PrismError::type_error(&**message),
            RuntimeErrorKind::UnsupportedOperandTypes { .. } => {
                PrismError::type_error(err.to_string())
            }
            RuntimeErrorKind::NotCallable { .. } => PrismError::type_error(err.to_string()),
            RuntimeErrorKind::NotIterable { .. } => PrismError::type_error(err.to_string()),
            RuntimeErrorKind::NotSubscriptable { .. } => PrismError::type_error(err.to_string()),
            RuntimeErrorKind::NameError { name } => PrismError::name(&**name),
            RuntimeErrorKind::AttributeError { .. } => PrismError::attribute(err.to_string()),
            RuntimeErrorKind::UnboundLocalError { name } => PrismError::name(&**name),
            RuntimeErrorKind::IndexError { .. } => PrismError::index(err.to_string()),
            RuntimeErrorKind::KeyError { key } => PrismError::key(&**key),
            RuntimeErrorKind::ValueError { message } => PrismError::value_error(&**message),
            RuntimeErrorKind::ZeroDivisionError => PrismError::zero_division("division by zero"),
            RuntimeErrorKind::OverflowError { message } => PrismError::value_error(&**message),
            RuntimeErrorKind::StopIteration => PrismError::internal("StopIteration"),
            RuntimeErrorKind::GeneratorExit => PrismError::internal("GeneratorExit"),
            RuntimeErrorKind::AssertionError { message } => {
                PrismError::assertion(message.as_ref().map(|s| &**s).unwrap_or(""))
            }
            RuntimeErrorKind::RecursionError { depth } => {
                PrismError::internal(format!("maximum recursion depth exceeded ({})", depth))
            }
            RuntimeErrorKind::InvalidOpcode { opcode } => {
                PrismError::internal(format!("invalid opcode 0x{:02x}", opcode))
            }
            RuntimeErrorKind::InternalError { message } => PrismError::internal(&**message),
            RuntimeErrorKind::ImportError { module, message } => {
                PrismError::import(format!("cannot import '{}': {}", module, message))
            }
            RuntimeErrorKind::Exception { message, .. } => PrismError::internal(&**message),
        }
    }
}

/// Result type for VM operations.
pub type VmResult<T> = Result<T, RuntimeError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_display() {
        let err = RuntimeError::type_error("expected int, got str");
        assert!(err.to_string().contains("TypeError"));
        assert!(err.to_string().contains("expected int"));
    }

    #[test]
    fn test_zero_division() {
        let err = RuntimeError::zero_division();
        assert!(err.to_string().contains("ZeroDivisionError"));
    }

    #[test]
    fn test_name_error() {
        let err = RuntimeError::name_error("undefined_var");
        assert!(err.to_string().contains("NameError"));
        assert!(err.to_string().contains("undefined_var"));
    }

    #[test]
    fn test_traceback() {
        let mut err = RuntimeError::type_error("test");
        err.add_traceback(TracebackEntry {
            func_name: "foo".into(),
            filename: "test.py".into(),
            line: 10,
        });
        assert_eq!(err.traceback.len(), 1);
        assert_eq!(&*err.traceback[0].func_name, "foo");
    }
}
