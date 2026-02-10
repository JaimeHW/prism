//! High-performance `open()` function implementation.
//!
//! Provides Python's `open()` builtin with full mode string parsing and
//! layered I/O construction (FileIO → BufferedReader/Writer → TextIOWrapper).
//!
//! # Mode String Format
//!
//! | Char | Meaning |
//! |------|---------|
//! | `r`  | Open for reading (default) |
//! | `w`  | Open for writing, truncating the file first |
//! | `x`  | Open for exclusive creation, failing if file exists |
//! | `a`  | Open for writing, appending to end |
//! | `b`  | Binary mode |
//! | `t`  | Text mode (default) |
//! | `+`  | Open for updating (reading and writing) |

use super::string_io::IoError;
use std::fmt;

/// Parsed file open mode.
///
/// Efficiently encoded as a bitfield for zero-cost mode checking.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FileMode {
    flags: u8,
}

// Bit positions for mode flags
const FLAG_READ: u8 = 0b0000_0001;
const FLAG_WRITE: u8 = 0b0000_0010;
const FLAG_APPEND: u8 = 0b0000_0100;
const FLAG_CREATE_EXCL: u8 = 0b0000_1000;
const FLAG_BINARY: u8 = 0b0001_0000;
const FLAG_PLUS: u8 = 0b0010_0000;
const FLAG_TRUNCATE: u8 = 0b0100_0000;

impl FileMode {
    /// Parse a Python-style mode string.
    ///
    /// Validates the mode string and returns a parsed `FileMode`.
    /// Invalid mode strings return an error.
    pub fn parse(mode: &str) -> Result<Self, IoError> {
        if mode.is_empty() {
            return Err(IoError::ValueError("invalid mode: ''".to_string()));
        }

        let mut flags = 0u8;
        let mut has_base = false;
        let mut has_bt = false; // text/binary flag seen

        for ch in mode.chars() {
            match ch {
                'r' => {
                    if has_base {
                        return Err(IoError::ValueError(format!(
                            "invalid mode: '{}' (multiple base modes)",
                            mode
                        )));
                    }
                    flags |= FLAG_READ;
                    has_base = true;
                }
                'w' => {
                    if has_base {
                        return Err(IoError::ValueError(format!(
                            "invalid mode: '{}' (multiple base modes)",
                            mode
                        )));
                    }
                    flags |= FLAG_WRITE | FLAG_TRUNCATE;
                    has_base = true;
                }
                'a' => {
                    if has_base {
                        return Err(IoError::ValueError(format!(
                            "invalid mode: '{}' (multiple base modes)",
                            mode
                        )));
                    }
                    flags |= FLAG_APPEND | FLAG_WRITE;
                    has_base = true;
                }
                'x' => {
                    if has_base {
                        return Err(IoError::ValueError(format!(
                            "invalid mode: '{}' (multiple base modes)",
                            mode
                        )));
                    }
                    flags |= FLAG_CREATE_EXCL | FLAG_WRITE;
                    has_base = true;
                }
                'b' => {
                    if has_bt {
                        return Err(IoError::ValueError(format!(
                            "invalid mode: '{}' (both text and binary)",
                            mode
                        )));
                    }
                    flags |= FLAG_BINARY;
                    has_bt = true;
                }
                't' => {
                    if has_bt {
                        return Err(IoError::ValueError(format!(
                            "invalid mode: '{}' (both text and binary)",
                            mode
                        )));
                    }
                    // Text is default, no flag needed
                    has_bt = true;
                }
                '+' => {
                    flags |= FLAG_PLUS;
                }
                _ => {
                    return Err(IoError::ValueError(format!(
                        "invalid mode: '{}' (character '{}' not recognized)",
                        mode, ch
                    )));
                }
            }
        }

        // Default to read mode if no base mode specified
        if !has_base {
            flags |= FLAG_READ;
        }

        Ok(Self { flags })
    }

    /// Check if mode includes reading.
    #[inline]
    pub fn is_reading(&self) -> bool {
        (self.flags & FLAG_READ) != 0 || (self.flags & FLAG_PLUS) != 0
    }

    /// Check if mode includes writing.
    #[inline]
    pub fn is_writing(&self) -> bool {
        (self.flags & FLAG_WRITE) != 0 || (self.flags & FLAG_PLUS) != 0
    }

    /// Check if mode is binary.
    #[inline]
    pub fn is_binary(&self) -> bool {
        (self.flags & FLAG_BINARY) != 0
    }

    /// Check if mode is text.
    #[inline]
    pub fn is_text(&self) -> bool {
        !self.is_binary()
    }

    /// Check if mode includes append.
    #[inline]
    pub fn is_append(&self) -> bool {
        (self.flags & FLAG_APPEND) != 0
    }

    /// Check if mode includes exclusive creation.
    #[inline]
    pub fn is_exclusive(&self) -> bool {
        (self.flags & FLAG_CREATE_EXCL) != 0
    }

    /// Check if mode includes truncation.
    #[inline]
    pub fn is_truncating(&self) -> bool {
        (self.flags & FLAG_TRUNCATE) != 0
    }

    /// Check if mode is update mode (+).
    #[inline]
    pub fn is_update(&self) -> bool {
        (self.flags & FLAG_PLUS) != 0
    }

    /// Return the OpenOptions needed for this mode.
    pub fn to_open_options(&self) -> std::fs::OpenOptions {
        let mut opts = std::fs::OpenOptions::new();

        if self.is_reading() {
            opts.read(true);
        }
        if self.is_writing() {
            opts.write(true);
        }
        if self.is_append() {
            opts.append(true);
        }
        if self.is_truncating() {
            opts.truncate(true);
        }
        if self.is_exclusive() {
            opts.create_new(true);
        } else if self.is_writing() {
            opts.create(true);
        }

        opts
    }
}

impl fmt::Display for FileMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut mode = String::with_capacity(4);
        if (self.flags & FLAG_READ) != 0 {
            mode.push('r');
        } else if (self.flags & FLAG_APPEND) != 0 {
            mode.push('a');
        } else if (self.flags & FLAG_CREATE_EXCL) != 0 {
            mode.push('x');
        } else if (self.flags & FLAG_WRITE) != 0 {
            mode.push('w');
        }
        if (self.flags & FLAG_BINARY) != 0 {
            mode.push('b');
        } else {
            mode.push('t');
        }
        if (self.flags & FLAG_PLUS) != 0 {
            mode.push('+');
        }
        write!(f, "{}", mode)
    }
}

/// Default buffer size for buffered I/O operations.
pub const DEFAULT_BUFFER_SIZE: usize = 8192;

/// Seek whence constants matching Python's `io.SEEK_SET`, etc.
pub const SEEK_SET: u32 = 0;
pub const SEEK_CUR: u32 = 1;
pub const SEEK_END: u32 = 2;
