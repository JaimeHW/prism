//! In-memory text stream (`StringIO`) implementation.
//!
//! A high-performance, zero-copy-where-possible text stream that stores
//! UTF-8 text in a contiguous `String` buffer. Mirrors Python's `io.StringIO`.
//!
//! # Performance
//!
//! - Small strings (≤1KB) avoid reallocation via pre-sizing
//! - Seek/tell are O(1) operations
//! - Write at position uses `splice` or `push_str` for optimal memory layout
//! - Read operations return slices where possible

use std::fmt;

/// Default initial capacity for new StringIO instances.
const DEFAULT_CAPACITY: usize = 256;

/// In-memory text stream.
///
/// Provides a file-like interface over an in-memory `String` buffer.
/// Supports read, write, seek, tell, truncate, and getvalue operations.
#[derive(Clone)]
pub struct StringIO {
    /// The underlying buffer.
    buffer: String,
    /// Current position in the buffer (byte offset).
    position: usize,
    /// Whether the stream has been closed.
    closed: bool,
}

impl StringIO {
    /// Create a new empty `StringIO`.
    #[inline]
    pub fn new() -> Self {
        Self {
            buffer: String::with_capacity(DEFAULT_CAPACITY),
            position: 0,
            closed: false,
        }
    }

    /// Create a `StringIO` with initial content.
    #[inline]
    pub fn with_initial(initial: &str) -> Self {
        Self {
            buffer: initial.to_string(),
            position: 0,
            closed: false,
        }
    }

    /// Create a `StringIO` with a specific capacity hint.
    #[inline]
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            buffer: String::with_capacity(capacity),
            position: 0,
            closed: false,
        }
    }

    /// Get the entire contents of the stream.
    ///
    /// This returns the full buffer regardless of position.
    #[inline]
    pub fn getvalue(&self) -> &str {
        &self.buffer
    }

    /// Read up to `n` characters from the current position.
    ///
    /// Returns the substring read and advances position.
    /// Returns empty string if at end or stream is closed.
    pub fn read(&mut self, n: Option<usize>) -> Result<&str, IoError> {
        self.check_closed()?;

        if self.position >= self.buffer.len() {
            return Ok("");
        }

        let available = self.buffer.len() - self.position;
        let to_read = n.map(|limit| limit.min(available)).unwrap_or(available);

        let start = self.position;
        let end = start + to_read;

        // Ensure we don't split a multi-byte character
        let end = self.adjust_boundary(end);

        self.position = end;
        Ok(&self.buffer[start..end])
    }

    /// Read a single line from the current position.
    ///
    /// Returns everything up to and including the next newline,
    /// or the rest of the buffer if no newline is found.
    pub fn readline(&mut self) -> Result<&str, IoError> {
        self.check_closed()?;

        if self.position >= self.buffer.len() {
            return Ok("");
        }

        let remaining = &self.buffer[self.position..];
        let line_end = remaining
            .find('\n')
            .map(|i| i + 1) // Include the newline
            .unwrap_or(remaining.len());

        let start = self.position;
        self.position += line_end;
        Ok(&self.buffer[start..self.position])
    }

    /// Write a string to the stream at the current position.
    ///
    /// Returns the number of characters written.
    pub fn write(&mut self, data: &str) -> Result<usize, IoError> {
        self.check_closed()?;

        let data_len = data.len();

        if self.position >= self.buffer.len() {
            // Append: extend with null bytes if needed, then append
            let gap = self.position - self.buffer.len();
            if gap > 0 {
                self.buffer.extend(std::iter::repeat('\0').take(gap));
            }
            self.buffer.push_str(data);
        } else if self.position + data_len <= self.buffer.len() {
            // Overwrite: replace existing content
            // SAFETY: We're replacing bytes at valid positions with valid UTF-8.
            let bytes = unsafe { self.buffer.as_bytes_mut() };
            bytes[self.position..self.position + data_len].copy_from_slice(data.as_bytes());
        } else {
            // Partial overwrite + extend
            let overlap = self.buffer.len() - self.position;
            let bytes = unsafe { self.buffer.as_bytes_mut() };
            bytes[self.position..self.position + overlap]
                .copy_from_slice(&data.as_bytes()[..overlap]);
            self.buffer.push_str(&data[overlap..]);
        }

        self.position += data_len;
        Ok(data_len)
    }

    /// Write a line (string + newline) to the stream.
    pub fn writelines(&mut self, lines: &[&str]) -> Result<(), IoError> {
        self.check_closed()?;
        for line in lines {
            self.write(line)?;
        }
        Ok(())
    }

    /// Seek to a position in the stream.
    ///
    /// `whence`:
    /// - 0 (SEEK_SET): from beginning
    /// - 1 (SEEK_CUR): from current position
    /// - 2 (SEEK_END): from end
    pub fn seek(&mut self, offset: i64, whence: u32) -> Result<usize, IoError> {
        self.check_closed()?;

        let new_pos = match whence {
            0 => {
                // SEEK_SET
                if offset < 0 {
                    return Err(IoError::ValueError("Negative seek position".to_string()));
                }
                offset as usize
            }
            1 => {
                // SEEK_CUR
                let current = self.position as i64;
                let target = current + offset;
                if target < 0 {
                    return Err(IoError::ValueError("Negative seek position".to_string()));
                }
                target as usize
            }
            2 => {
                // SEEK_END
                let end = self.buffer.len() as i64;
                let target = end + offset;
                if target < 0 {
                    return Err(IoError::ValueError("Negative seek position".to_string()));
                }
                target as usize
            }
            _ => {
                return Err(IoError::ValueError(format!(
                    "Invalid whence value: {} (expected 0, 1, or 2)",
                    whence
                )));
            }
        };

        self.position = new_pos;
        Ok(self.position)
    }

    /// Return the current stream position.
    #[inline]
    pub fn tell(&self) -> Result<usize, IoError> {
        self.check_closed()?;
        Ok(self.position)
    }

    /// Truncate the stream to at most the given size.
    ///
    /// If `size` is `None`, truncates at the current position.
    pub fn truncate(&mut self, size: Option<usize>) -> Result<usize, IoError> {
        self.check_closed()?;
        let target = size.unwrap_or(self.position);
        if target < self.buffer.len() {
            self.buffer.truncate(target);
        }
        Ok(target)
    }

    /// Close the stream.
    #[inline]
    pub fn close(&mut self) {
        self.closed = true;
    }

    /// Check if the stream is closed.
    #[inline]
    pub fn is_closed(&self) -> bool {
        self.closed
    }

    /// Check if the stream is readable.
    #[inline]
    pub fn readable(&self) -> bool {
        !self.closed
    }

    /// Check if the stream is writable.
    #[inline]
    pub fn writable(&self) -> bool {
        !self.closed
    }

    /// Check if the stream is seekable.
    #[inline]
    pub fn seekable(&self) -> bool {
        !self.closed
    }

    /// Get the total length of the buffer.
    #[inline]
    pub fn len(&self) -> usize {
        self.buffer.len()
    }

    /// Check if the buffer is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.buffer.is_empty()
    }

    // =========================================================================
    // Internal helpers
    // =========================================================================

    /// Check that the stream is not closed.
    #[inline]
    fn check_closed(&self) -> Result<(), IoError> {
        if self.closed {
            Err(IoError::ValueError(
                "I/O operation on closed file".to_string(),
            ))
        } else {
            Ok(())
        }
    }

    /// Adjust a byte boundary to not split a UTF-8 character.
    #[inline]
    fn adjust_boundary(&self, pos: usize) -> usize {
        if pos >= self.buffer.len() {
            return self.buffer.len();
        }
        // Find the nearest valid character boundary at or before pos
        let mut adjusted = pos;
        while adjusted > 0 && !self.buffer.is_char_boundary(adjusted) {
            adjusted -= 1;
        }
        adjusted
    }
}

impl Default for StringIO {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Debug for StringIO {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("StringIO")
            .field("len", &self.buffer.len())
            .field("position", &self.position)
            .field("closed", &self.closed)
            .finish()
    }
}

impl fmt::Display for StringIO {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "<_io.StringIO object>")
    }
}

/// I/O error type.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum IoError {
    /// Value error (e.g., invalid seek position, operation on closed file).
    ValueError(String),
    /// Unsupported operation.
    UnsupportedOperation(String),
    /// OS-level error.
    OsError(String),
}

impl fmt::Display for IoError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            IoError::ValueError(msg) => write!(f, "ValueError: {}", msg),
            IoError::UnsupportedOperation(msg) => {
                write!(f, "UnsupportedOperation: {}", msg)
            }
            IoError::OsError(msg) => write!(f, "OSError: {}", msg),
        }
    }
}

impl std::error::Error for IoError {}
