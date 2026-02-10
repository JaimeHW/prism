//! Python `io` module implementation.
//!
//! High-performance I/O primitives providing:
//! - `StringIO` — In-memory text stream
//! - `BytesIO` — In-memory binary stream
//! - `FileMode` — Mode string parser
//! - Constants — `SEEK_SET`, `SEEK_CUR`, `SEEK_END`, `DEFAULT_BUFFER_SIZE`
//!
//! # Architecture
//!
//! ```text
//! IoModule (Module trait)
//! ├── StringIO  (in-memory text)
//! ├── BytesIO   (in-memory binary)
//! ├── FileMode  (mode parsing)
//! └── Constants (SEEK_SET/CUR/END, DEFAULT_BUFFER_SIZE)
//! ```

pub mod bytes_io;
pub mod open_fn;
pub mod string_io;

#[cfg(test)]
mod tests;

pub use bytes_io::BytesIO;
pub use open_fn::{DEFAULT_BUFFER_SIZE, FileMode, SEEK_CUR, SEEK_END, SEEK_SET};
pub use string_io::{IoError, StringIO};

use super::{Module, ModuleError, ModuleResult};
use std::sync::Arc;

// =============================================================================
// IO Module
// =============================================================================

/// The `io` module implementation.
#[derive(Debug, Clone)]
pub struct IoModule {
    /// Cached attribute names for fast lookup.
    attrs: Vec<Arc<str>>,
}

impl IoModule {
    /// Create a new io module instance.
    pub fn new() -> Self {
        let attrs = vec![
            Arc::from("StringIO"),
            Arc::from("BytesIO"),
            Arc::from("open"),
            Arc::from("FileIO"),
            Arc::from("BufferedReader"),
            Arc::from("BufferedWriter"),
            Arc::from("BufferedRandom"),
            Arc::from("TextIOWrapper"),
            Arc::from("DEFAULT_BUFFER_SIZE"),
            Arc::from("SEEK_SET"),
            Arc::from("SEEK_CUR"),
            Arc::from("SEEK_END"),
            Arc::from("IOBase"),
            Arc::from("RawIOBase"),
            Arc::from("BufferedIOBase"),
            Arc::from("TextIOBase"),
            Arc::from("UnsupportedOperation"),
        ];

        Self { attrs }
    }
}

impl Default for IoModule {
    fn default() -> Self {
        Self::new()
    }
}

impl Module for IoModule {
    fn name(&self) -> &str {
        "io"
    }

    fn get_attr(&self, name: &str) -> ModuleResult {
        use prism_core::Value;
        match name {
            // Constants
            "DEFAULT_BUFFER_SIZE" => Ok(Value::int(DEFAULT_BUFFER_SIZE as i64).unwrap()),
            "SEEK_SET" => Ok(Value::int(SEEK_SET as i64).unwrap()),
            "SEEK_CUR" => Ok(Value::int(SEEK_CUR as i64).unwrap()),
            "SEEK_END" => Ok(Value::int(SEEK_END as i64).unwrap()),

            // Classes and functions — placeholders until callable system is ready
            "StringIO" | "BytesIO" | "open" | "FileIO" | "BufferedReader" | "BufferedWriter"
            | "BufferedRandom" | "TextIOWrapper" | "IOBase" | "RawIOBase" | "BufferedIOBase"
            | "TextIOBase" => Err(ModuleError::AttributeError(format!(
                "io.{} is not yet callable as an object",
                name
            ))),

            "UnsupportedOperation" => Err(ModuleError::AttributeError(
                "io.UnsupportedOperation is not yet accessible as a type".to_string(),
            )),

            _ => Err(ModuleError::AttributeError(format!(
                "module 'io' has no attribute '{}'",
                name
            ))),
        }
    }

    fn dir(&self) -> Vec<Arc<str>> {
        self.attrs.clone()
    }
}
