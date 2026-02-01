//! # Prism Core
//!
//! Core types, traits, and primitives for the Prism Python runtime.
//!
//! This crate provides the foundational building blocks shared across all Prism components:
//!
//! - **Value System**: Tagged union representation of Python values with NaN-boxing
//! - **Object Model**: GC-managed object references and type descriptors
//! - **Interning**: String and identifier interning for O(1) equality checks
//! - **Error Handling**: Result types and error definitions
//! - **Memory**: Arena allocators and allocation primitives

#![deny(unsafe_op_in_unsafe_fn)]
#![warn(missing_docs)]
#![warn(clippy::all)]
#![warn(clippy::pedantic)]
#![allow(clippy::module_name_repetitions)]

pub mod error;
pub mod intern;
pub mod span;
pub mod value;

pub use error::{PrismError, PrismResult};
pub use intern::{InternedString, StringInterner};
pub use span::Span;
pub use value::Value;

/// Prism runtime version.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Python language version this runtime targets.
pub const PYTHON_VERSION: (u8, u8, u8) = (3, 12, 0);
