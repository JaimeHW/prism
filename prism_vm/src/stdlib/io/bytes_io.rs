//! In-memory binary stream (`BytesIO`) implementation.
//!
//! High-performance binary stream operating on raw bytes (`Vec<u8>`).
//! Mirrors Python's `io.BytesIO`.
//!
//! # Performance
//!
//! - All operations are O(1) or O(n) where n is the data size
//! - No encoding/decoding overhead (raw bytes)
//! - Write operations use `copy_from_slice` for maximum throughput
//! - Seek/tell are O(1)

use super::string_io::IoError;
use std::fmt;

/// Default initial capacity for new BytesIO instances.
const DEFAULT_CAPACITY: usize = 256;

/// In-memory binary stream.
///
/// Provides a file-like interface over an in-memory `Vec<u8>` buffer.
/// Supports read, write, seek, tell, truncate, and getvalue operations.
#[derive(Clone)]
pub struct BytesIO {
    /// The underlying byte buffer.
    buffer: Vec<u8>,
    /// Current position in the buffer.
    position: usize,
    /// Whether the stream has been closed.
    closed: bool,
}

impl BytesIO {
    /// Create a new empty `BytesIO`.
    #[inline]
    pub fn new() -> Self {
        Self {
            buffer: Vec::with_capacity(DEFAULT_CAPACITY),
            position: 0,
            closed: false,
        }
    }

    /// Create a `BytesIO` with initial content.
    #[inline]
    pub fn with_initial(initial: &[u8]) -> Self {
        Self {
            buffer: initial.to_vec(),
            position: 0,
            closed: false,
        }
    }

    /// Create a `BytesIO` with a specific capacity hint.
    #[inline]
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            buffer: Vec::with_capacity(capacity),
            position: 0,
            closed: false,
        }
    }

    /// Get the entire contents of the stream as bytes.
    ///
    /// This returns the full buffer regardless of position.
    #[inline]
    pub fn getvalue(&self) -> &[u8] {
        &self.buffer
    }

    /// Read up to `n` bytes from the current position.
    ///
    /// Returns the bytes read and advances position.
    pub fn read(&mut self, n: Option<usize>) -> Result<&[u8], IoError> {
        self.check_closed()?;

        if self.position >= self.buffer.len() {
            return Ok(&[]);
        }

        let available = self.buffer.len() - self.position;
        let to_read = n.map(|limit| limit.min(available)).unwrap_or(available);

        let start = self.position;
        self.position += to_read;
        Ok(&self.buffer[start..self.position])
    }

    /// Read exactly `n` bytes, or return an error if not enough data.
    pub fn readexactly(&mut self, n: usize) -> Result<&[u8], IoError> {
        self.check_closed()?;

        let available = self.buffer.len().saturating_sub(self.position);
        if available < n {
            return Err(IoError::ValueError(format!(
                "read beyond end of stream: requested {} bytes but only {} available",
                n, available
            )));
        }

        let start = self.position;
        self.position += n;
        Ok(&self.buffer[start..self.position])
    }

    /// Read a single line from the current position.
    ///
    /// Returns everything up to and including the next `\n` byte,
    /// or the rest of the buffer if no newline is found.
    pub fn readline(&mut self) -> Result<&[u8], IoError> {
        self.check_closed()?;

        if self.position >= self.buffer.len() {
            return Ok(&[]);
        }

        let remaining = &self.buffer[self.position..];
        let line_end = remaining
            .iter()
            .position(|&b| b == b'\n')
            .map(|i| i + 1) // Include the newline
            .unwrap_or(remaining.len());

        let start = self.position;
        self.position += line_end;
        Ok(&self.buffer[start..self.position])
    }

    /// Write bytes to the stream at the current position.
    ///
    /// Returns the number of bytes written.
    pub fn write(&mut self, data: &[u8]) -> Result<usize, IoError> {
        self.check_closed()?;

        let data_len = data.len();

        if self.position >= self.buffer.len() {
            // Append: extend with zeros if needed, then extend with data
            let gap = self.position - self.buffer.len();
            if gap > 0 {
                self.buffer.extend(std::iter::repeat(0u8).take(gap));
            }
            self.buffer.extend_from_slice(data);
        } else if self.position + data_len <= self.buffer.len() {
            // Overwrite: replace existing content
            self.buffer[self.position..self.position + data_len].copy_from_slice(data);
        } else {
            // Partial overwrite + extend
            let overlap = self.buffer.len() - self.position;
            self.buffer[self.position..self.position + overlap].copy_from_slice(&data[..overlap]);
            self.buffer.extend_from_slice(&data[overlap..]);
        }

        self.position += data_len;
        Ok(data_len)
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
                if offset < 0 {
                    return Err(IoError::ValueError("Negative seek position".to_string()));
                }
                offset as usize
            }
            1 => {
                let current = self.position as i64;
                let target = current + offset;
                if target < 0 {
                    return Err(IoError::ValueError("Negative seek position".to_string()));
                }
                target as usize
            }
            2 => {
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
}

impl Default for BytesIO {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Debug for BytesIO {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("BytesIO")
            .field("len", &self.buffer.len())
            .field("position", &self.position)
            .field("closed", &self.closed)
            .finish()
    }
}

impl fmt::Display for BytesIO {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "<_io.BytesIO object>")
    }
}
