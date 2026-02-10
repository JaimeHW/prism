//! Exhaustive tests for the `io` module.
//!
//! Tests are organized by component:
//! 1. StringIO tests
//! 2. BytesIO tests
//! 3. FileMode tests
//! 4. IoModule tests
//! 5. Cross-cutting tests

use super::*;

// =============================================================================
// StringIO Tests
// =============================================================================

mod string_io_tests {
    use super::*;

    // =========================================================================
    // Construction
    // =========================================================================

    #[test]
    fn test_new_empty() {
        let sio = StringIO::new();
        assert_eq!(sio.getvalue(), "");
        assert_eq!(sio.len(), 0);
        assert!(sio.is_empty());
        assert!(!sio.is_closed());
    }

    #[test]
    fn test_with_initial() {
        let sio = StringIO::with_initial("hello");
        assert_eq!(sio.getvalue(), "hello");
        assert_eq!(sio.len(), 5);
    }

    #[test]
    fn test_with_capacity() {
        let sio = StringIO::with_capacity(1024);
        assert_eq!(sio.len(), 0);
        assert!(sio.is_empty());
    }

    #[test]
    fn test_default() {
        let sio = StringIO::default();
        assert_eq!(sio.getvalue(), "");
    }

    // =========================================================================
    // Write
    // =========================================================================

    #[test]
    fn test_write_to_empty() {
        let mut sio = StringIO::new();
        let n = sio.write("hello").unwrap();
        assert_eq!(n, 5);
        assert_eq!(sio.getvalue(), "hello");
    }

    #[test]
    fn test_write_multiple() {
        let mut sio = StringIO::new();
        sio.write("hello").unwrap();
        sio.write(" world").unwrap();
        assert_eq!(sio.getvalue(), "hello world");
    }

    #[test]
    fn test_write_empty_string() {
        let mut sio = StringIO::new();
        let n = sio.write("").unwrap();
        assert_eq!(n, 0);
        assert_eq!(sio.getvalue(), "");
    }

    #[test]
    fn test_write_overwrite() {
        let mut sio = StringIO::with_initial("hello");
        sio.write("HEL").unwrap();
        assert_eq!(sio.getvalue(), "HELlo");
    }

    #[test]
    fn test_write_beyond_end() {
        let mut sio = StringIO::with_initial("hi");
        sio.seek(5, 0).unwrap();
        sio.write("!").unwrap();
        assert_eq!(sio.len(), 6);
    }

    #[test]
    fn test_write_closed_errors() {
        let mut sio = StringIO::new();
        sio.close();
        assert!(sio.write("test").is_err());
    }

    #[test]
    fn test_writelines() {
        let mut sio = StringIO::new();
        sio.writelines(&["hello\n", "world\n"]).unwrap();
        assert_eq!(sio.getvalue(), "hello\nworld\n");
    }

    #[test]
    fn test_writelines_empty() {
        let mut sio = StringIO::new();
        sio.writelines(&[]).unwrap();
        assert_eq!(sio.getvalue(), "");
    }

    // =========================================================================
    // Read
    // =========================================================================

    #[test]
    fn test_read_all() {
        let mut sio = StringIO::with_initial("hello world");
        let result = sio.read(None).unwrap();
        assert_eq!(result, "hello world");
    }

    #[test]
    fn test_read_n() {
        let mut sio = StringIO::with_initial("hello world");
        let result = sio.read(Some(5)).unwrap();
        assert_eq!(result, "hello");
    }

    #[test]
    fn test_read_sequential() {
        let mut sio = StringIO::with_initial("hello world");
        let r1 = sio.read(Some(5)).unwrap().to_string();
        let r2 = sio.read(Some(6)).unwrap().to_string();
        assert_eq!(r1, "hello");
        assert_eq!(r2, " world");
    }

    #[test]
    fn test_read_empty() {
        let mut sio = StringIO::new();
        let result = sio.read(None).unwrap();
        assert_eq!(result, "");
    }

    #[test]
    fn test_read_at_end() {
        let mut sio = StringIO::with_initial("hi");
        sio.read(None).unwrap();
        let result = sio.read(None).unwrap();
        assert_eq!(result, "");
    }

    #[test]
    fn test_read_zero() {
        let mut sio = StringIO::with_initial("hello");
        let result = sio.read(Some(0)).unwrap();
        assert_eq!(result, "");
    }

    #[test]
    fn test_read_more_than_available() {
        let mut sio = StringIO::with_initial("hi");
        let result = sio.read(Some(100)).unwrap();
        assert_eq!(result, "hi");
    }

    #[test]
    fn test_read_closed_errors() {
        let mut sio = StringIO::with_initial("hello");
        sio.close();
        assert!(sio.read(None).is_err());
    }

    // =========================================================================
    // Readline
    // =========================================================================

    #[test]
    fn test_readline_single_line() {
        let mut sio = StringIO::with_initial("hello\nworld\n");
        let line = sio.readline().unwrap().to_string();
        assert_eq!(line, "hello\n");
    }

    #[test]
    fn test_readline_last_line_no_newline() {
        let mut sio = StringIO::with_initial("hello");
        let line = sio.readline().unwrap().to_string();
        assert_eq!(line, "hello");
    }

    #[test]
    fn test_readline_sequential() {
        let mut sio = StringIO::with_initial("line1\nline2\nline3");
        let l1 = sio.readline().unwrap().to_string();
        let l2 = sio.readline().unwrap().to_string();
        let l3 = sio.readline().unwrap().to_string();
        assert_eq!(l1, "line1\n");
        assert_eq!(l2, "line2\n");
        assert_eq!(l3, "line3");
    }

    #[test]
    fn test_readline_empty() {
        let mut sio = StringIO::new();
        let line = sio.readline().unwrap();
        assert_eq!(line, "");
    }

    #[test]
    fn test_readline_at_end() {
        let mut sio = StringIO::with_initial("hi\n");
        let _ = sio.readline().unwrap();
        let line = sio.readline().unwrap();
        assert_eq!(line, "");
    }

    // =========================================================================
    // Seek and Tell
    // =========================================================================

    #[test]
    fn test_tell_initial() {
        let sio = StringIO::new();
        assert_eq!(sio.tell().unwrap(), 0);
    }

    #[test]
    fn test_tell_after_write() {
        let mut sio = StringIO::new();
        sio.write("hello").unwrap();
        assert_eq!(sio.tell().unwrap(), 5);
    }

    #[test]
    fn test_seek_set() {
        let mut sio = StringIO::with_initial("hello world");
        sio.read(None).unwrap();
        sio.seek(0, 0).unwrap();
        assert_eq!(sio.tell().unwrap(), 0);
    }

    #[test]
    fn test_seek_cur() {
        let mut sio = StringIO::with_initial("hello");
        sio.seek(2, 0).unwrap();
        sio.seek(1, 1).unwrap();
        assert_eq!(sio.tell().unwrap(), 3);
    }

    #[test]
    fn test_seek_end() {
        let mut sio = StringIO::with_initial("hello");
        sio.seek(0, 2).unwrap();
        assert_eq!(sio.tell().unwrap(), 5);
    }

    #[test]
    fn test_seek_end_negative() {
        let mut sio = StringIO::with_initial("hello");
        sio.seek(-2, 2).unwrap();
        assert_eq!(sio.tell().unwrap(), 3);
    }

    #[test]
    fn test_seek_negative_errors() {
        let mut sio = StringIO::with_initial("hello");
        assert!(sio.seek(-1, 0).is_err());
    }

    #[test]
    fn test_seek_invalid_whence() {
        let mut sio = StringIO::with_initial("hello");
        assert!(sio.seek(0, 3).is_err());
    }

    #[test]
    fn test_seek_then_read() {
        let mut sio = StringIO::with_initial("hello world");
        sio.seek(6, 0).unwrap();
        let result = sio.read(None).unwrap();
        assert_eq!(result, "world");
    }

    #[test]
    fn test_seek_then_write() {
        let mut sio = StringIO::with_initial("aaaaa");
        sio.seek(2, 0).unwrap();
        sio.write("BB").unwrap();
        assert_eq!(sio.getvalue(), "aaBBa");
    }

    // =========================================================================
    // Truncate
    // =========================================================================

    #[test]
    fn test_truncate_at_position() {
        let mut sio = StringIO::with_initial("hello world");
        sio.seek(5, 0).unwrap();
        sio.truncate(None).unwrap();
        assert_eq!(sio.getvalue(), "hello");
    }

    #[test]
    fn test_truncate_explicit_size() {
        let mut sio = StringIO::with_initial("hello world");
        sio.truncate(Some(5)).unwrap();
        assert_eq!(sio.getvalue(), "hello");
    }

    #[test]
    fn test_truncate_to_zero() {
        let mut sio = StringIO::with_initial("hello");
        sio.truncate(Some(0)).unwrap();
        assert_eq!(sio.getvalue(), "");
    }

    #[test]
    fn test_truncate_beyond_length() {
        let mut sio = StringIO::with_initial("hello");
        let result = sio.truncate(Some(100)).unwrap();
        assert_eq!(result, 100);
        assert_eq!(sio.getvalue(), "hello"); // No extension
    }

    // =========================================================================
    // Close
    // =========================================================================

    #[test]
    fn test_close() {
        let mut sio = StringIO::new();
        assert!(!sio.is_closed());
        sio.close();
        assert!(sio.is_closed());
    }

    #[test]
    fn test_operations_after_close() {
        let mut sio = StringIO::new();
        sio.close();
        assert!(sio.read(None).is_err());
        assert!(sio.write("test").is_err());
        assert!(sio.seek(0, 0).is_err());
        assert!(sio.tell().is_err());
        assert!(sio.truncate(None).is_err());
        assert!(sio.readline().is_err());
    }

    // =========================================================================
    // Capabilities
    // =========================================================================

    #[test]
    fn test_readable() {
        let sio = StringIO::new();
        assert!(sio.readable());
    }

    #[test]
    fn test_writable() {
        let sio = StringIO::new();
        assert!(sio.writable());
    }

    #[test]
    fn test_seekable() {
        let sio = StringIO::new();
        assert!(sio.seekable());
    }

    // =========================================================================
    // Display and Debug
    // =========================================================================

    #[test]
    fn test_display() {
        let sio = StringIO::new();
        assert_eq!(format!("{}", sio), "<_io.StringIO object>");
    }

    #[test]
    fn test_debug() {
        let sio = StringIO::new();
        let debug = format!("{:?}", sio);
        assert!(debug.contains("StringIO"));
        assert!(debug.contains("len"));
    }

    // =========================================================================
    // Unicode / Multi-byte
    // =========================================================================

    #[test]
    fn test_unicode_write_read() {
        let mut sio = StringIO::new();
        sio.write("héllo 世界 🌍").unwrap();
        sio.seek(0, 0).unwrap();
        let result = sio.read(None).unwrap();
        assert_eq!(result, "héllo 世界 🌍");
    }

    #[test]
    fn test_unicode_getvalue() {
        let sio = StringIO::with_initial("日本語テスト");
        assert_eq!(sio.getvalue(), "日本語テスト");
    }

    // =========================================================================
    // Edge cases
    // =========================================================================

    #[test]
    fn test_large_write() {
        let mut sio = StringIO::new();
        let data = "x".repeat(100_000);
        sio.write(&data).unwrap();
        assert_eq!(sio.len(), 100_000);
    }

    #[test]
    fn test_write_read_cycle() {
        let mut sio = StringIO::new();
        for i in 0..100 {
            sio.write(&format!("line{}\n", i)).unwrap();
        }
        sio.seek(0, 0).unwrap();
        let mut count = 0;
        loop {
            let line = sio.readline().unwrap().to_string();
            if line.is_empty() {
                break;
            }
            count += 1;
        }
        assert_eq!(count, 100);
    }
}

// =============================================================================
// BytesIO Tests
// =============================================================================

mod bytes_io_tests {
    use super::*;

    // =========================================================================
    // Construction
    // =========================================================================

    #[test]
    fn test_new_empty() {
        let bio = BytesIO::new();
        assert_eq!(bio.getvalue(), &[]);
        assert_eq!(bio.len(), 0);
        assert!(bio.is_empty());
    }

    #[test]
    fn test_with_initial() {
        let bio = BytesIO::with_initial(b"hello");
        assert_eq!(bio.getvalue(), b"hello");
        assert_eq!(bio.len(), 5);
    }

    #[test]
    fn test_with_capacity() {
        let bio = BytesIO::with_capacity(1024);
        assert_eq!(bio.len(), 0);
    }

    #[test]
    fn test_default() {
        let bio = BytesIO::default();
        assert_eq!(bio.getvalue(), &[]);
    }

    // =========================================================================
    // Write
    // =========================================================================

    #[test]
    fn test_write_to_empty() {
        let mut bio = BytesIO::new();
        let n = bio.write(b"hello").unwrap();
        assert_eq!(n, 5);
        assert_eq!(bio.getvalue(), b"hello");
    }

    #[test]
    fn test_write_multiple() {
        let mut bio = BytesIO::new();
        bio.write(b"hello").unwrap();
        bio.write(b" world").unwrap();
        assert_eq!(bio.getvalue(), b"hello world");
    }

    #[test]
    fn test_write_empty() {
        let mut bio = BytesIO::new();
        let n = bio.write(b"").unwrap();
        assert_eq!(n, 0);
    }

    #[test]
    fn test_write_overwrite() {
        let mut bio = BytesIO::with_initial(b"hello");
        bio.write(b"HEL").unwrap();
        assert_eq!(bio.getvalue(), b"HELlo");
    }

    #[test]
    fn test_write_beyond_end() {
        let mut bio = BytesIO::with_initial(b"hi");
        bio.seek(5, 0).unwrap();
        bio.write(b"!").unwrap();
        assert_eq!(bio.len(), 6);
        assert_eq!(bio.getvalue()[0..2], *b"hi");
        assert_eq!(bio.getvalue()[5], b'!');
    }

    #[test]
    fn test_write_closed_errors() {
        let mut bio = BytesIO::new();
        bio.close();
        assert!(bio.write(b"test").is_err());
    }

    // =========================================================================
    // Read
    // =========================================================================

    #[test]
    fn test_read_all() {
        let mut bio = BytesIO::with_initial(b"hello world");
        let result = bio.read(None).unwrap().to_vec();
        assert_eq!(result, b"hello world");
    }

    #[test]
    fn test_read_n() {
        let mut bio = BytesIO::with_initial(b"hello world");
        let result = bio.read(Some(5)).unwrap().to_vec();
        assert_eq!(result, b"hello");
    }

    #[test]
    fn test_read_sequential() {
        let mut bio = BytesIO::with_initial(b"hello world");
        let r1 = bio.read(Some(5)).unwrap().to_vec();
        let r2 = bio.read(Some(6)).unwrap().to_vec();
        assert_eq!(r1, b"hello");
        assert_eq!(r2, b" world");
    }

    #[test]
    fn test_read_at_end() {
        let mut bio = BytesIO::with_initial(b"hi");
        let _ = bio.read(None).unwrap();
        let result = bio.read(None).unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn test_read_zero() {
        let mut bio = BytesIO::with_initial(b"hello");
        let result = bio.read(Some(0)).unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn test_read_closed_errors() {
        let mut bio = BytesIO::with_initial(b"hello");
        bio.close();
        assert!(bio.read(None).is_err());
    }

    // =========================================================================
    // Readexactly
    // =========================================================================

    #[test]
    fn test_readexactly_success() {
        let mut bio = BytesIO::with_initial(b"hello world");
        let result = bio.readexactly(5).unwrap().to_vec();
        assert_eq!(result, b"hello");
    }

    #[test]
    fn test_readexactly_not_enough() {
        let mut bio = BytesIO::with_initial(b"hi");
        assert!(bio.readexactly(10).is_err());
    }

    #[test]
    fn test_readexactly_exact_match() {
        let mut bio = BytesIO::with_initial(b"hello");
        let result = bio.readexactly(5).unwrap().to_vec();
        assert_eq!(result, b"hello");
    }

    // =========================================================================
    // Readline
    // =========================================================================

    #[test]
    fn test_readline() {
        let mut bio = BytesIO::with_initial(b"hello\nworld\n");
        let line = bio.readline().unwrap().to_vec();
        assert_eq!(line, b"hello\n");
    }

    #[test]
    fn test_readline_no_newline() {
        let mut bio = BytesIO::with_initial(b"hello");
        let line = bio.readline().unwrap().to_vec();
        assert_eq!(line, b"hello");
    }

    #[test]
    fn test_readline_sequential() {
        let mut bio = BytesIO::with_initial(b"l1\nl2\nl3");
        let l1 = bio.readline().unwrap().to_vec();
        let l2 = bio.readline().unwrap().to_vec();
        let l3 = bio.readline().unwrap().to_vec();
        assert_eq!(l1, b"l1\n");
        assert_eq!(l2, b"l2\n");
        assert_eq!(l3, b"l3");
    }

    #[test]
    fn test_readline_at_end() {
        let mut bio = BytesIO::with_initial(b"hi\n");
        let _ = bio.readline().unwrap();
        let line = bio.readline().unwrap();
        assert!(line.is_empty());
    }

    // =========================================================================
    // Seek and Tell
    // =========================================================================

    #[test]
    fn test_tell_initial() {
        let bio = BytesIO::new();
        assert_eq!(bio.tell().unwrap(), 0);
    }

    #[test]
    fn test_tell_after_write() {
        let mut bio = BytesIO::new();
        bio.write(b"hello").unwrap();
        assert_eq!(bio.tell().unwrap(), 5);
    }

    #[test]
    fn test_seek_set() {
        let mut bio = BytesIO::with_initial(b"hello");
        bio.read(None).unwrap();
        bio.seek(0, 0).unwrap();
        assert_eq!(bio.tell().unwrap(), 0);
    }

    #[test]
    fn test_seek_cur() {
        let mut bio = BytesIO::with_initial(b"hello");
        bio.seek(2, 0).unwrap();
        bio.seek(1, 1).unwrap();
        assert_eq!(bio.tell().unwrap(), 3);
    }

    #[test]
    fn test_seek_end() {
        let mut bio = BytesIO::with_initial(b"hello");
        bio.seek(0, 2).unwrap();
        assert_eq!(bio.tell().unwrap(), 5);
    }

    #[test]
    fn test_seek_negative_errors() {
        let mut bio = BytesIO::with_initial(b"hello");
        assert!(bio.seek(-1, 0).is_err());
    }

    #[test]
    fn test_seek_invalid_whence() {
        let mut bio = BytesIO::with_initial(b"hello");
        assert!(bio.seek(0, 3).is_err());
    }

    #[test]
    fn test_seek_then_read() {
        let mut bio = BytesIO::with_initial(b"hello world");
        bio.seek(6, 0).unwrap();
        let result = bio.read(None).unwrap().to_vec();
        assert_eq!(result, b"world");
    }

    #[test]
    fn test_seek_then_write() {
        let mut bio = BytesIO::with_initial(b"aaaaa");
        bio.seek(2, 0).unwrap();
        bio.write(b"BB").unwrap();
        assert_eq!(bio.getvalue(), b"aaBBa");
    }

    // =========================================================================
    // Truncate
    // =========================================================================

    #[test]
    fn test_truncate_at_position() {
        let mut bio = BytesIO::with_initial(b"hello world");
        bio.seek(5, 0).unwrap();
        bio.truncate(None).unwrap();
        assert_eq!(bio.getvalue(), b"hello");
    }

    #[test]
    fn test_truncate_explicit_size() {
        let mut bio = BytesIO::with_initial(b"hello world");
        bio.truncate(Some(5)).unwrap();
        assert_eq!(bio.getvalue(), b"hello");
    }

    #[test]
    fn test_truncate_to_zero() {
        let mut bio = BytesIO::with_initial(b"hello");
        bio.truncate(Some(0)).unwrap();
        assert_eq!(bio.getvalue(), &[]);
    }

    // =========================================================================
    // Close
    // =========================================================================

    #[test]
    fn test_close() {
        let mut bio = BytesIO::new();
        assert!(!bio.is_closed());
        bio.close();
        assert!(bio.is_closed());
    }

    #[test]
    fn test_operations_after_close() {
        let mut bio = BytesIO::new();
        bio.close();
        assert!(bio.read(None).is_err());
        assert!(bio.write(b"test").is_err());
        assert!(bio.seek(0, 0).is_err());
        assert!(bio.tell().is_err());
        assert!(bio.truncate(None).is_err());
    }

    // =========================================================================
    // Capabilities
    // =========================================================================

    #[test]
    fn test_readable() {
        let bio = BytesIO::new();
        assert!(bio.readable());
    }

    #[test]
    fn test_writable() {
        let bio = BytesIO::new();
        assert!(bio.writable());
    }

    #[test]
    fn test_seekable() {
        let bio = BytesIO::new();
        assert!(bio.seekable());
    }

    // =========================================================================
    // Display and Debug
    // =========================================================================

    #[test]
    fn test_display() {
        let bio = BytesIO::new();
        assert_eq!(format!("{}", bio), "<_io.BytesIO object>");
    }

    #[test]
    fn test_debug() {
        let bio = BytesIO::new();
        let debug = format!("{:?}", bio);
        assert!(debug.contains("BytesIO"));
    }

    // =========================================================================
    // Binary data
    // =========================================================================

    #[test]
    fn test_binary_data_with_zeros() {
        let mut bio = BytesIO::new();
        bio.write(&[0x00, 0xFF, 0x00, 0xFF]).unwrap();
        assert_eq!(bio.getvalue(), &[0x00, 0xFF, 0x00, 0xFF]);
    }

    #[test]
    fn test_all_byte_values() {
        let mut bio = BytesIO::new();
        let all_bytes: Vec<u8> = (0..=255).collect();
        bio.write(&all_bytes).unwrap();
        assert_eq!(bio.getvalue(), &all_bytes[..]);
    }

    #[test]
    fn test_large_binary_write() {
        let mut bio = BytesIO::new();
        let data = vec![0xAB; 100_000];
        bio.write(&data).unwrap();
        assert_eq!(bio.len(), 100_000);
    }

    #[test]
    fn test_write_read_cycle() {
        let mut bio = BytesIO::new();
        for i in 0u8..100 {
            bio.write(&[i]).unwrap();
        }
        bio.seek(0, 0).unwrap();
        let result = bio.read(None).unwrap();
        assert_eq!(result.len(), 100);
        assert_eq!(result[0], 0);
        assert_eq!(result[99], 99);
    }
}

// =============================================================================
// FileMode Tests
// =============================================================================

mod file_mode_tests {
    use super::*;

    // =========================================================================
    // Parse - Basic modes
    // =========================================================================

    #[test]
    fn test_parse_read() {
        let mode = FileMode::parse("r").unwrap();
        assert!(mode.is_reading());
        assert!(!mode.is_writing());
        assert!(mode.is_text());
        assert!(!mode.is_binary());
    }

    #[test]
    fn test_parse_write() {
        let mode = FileMode::parse("w").unwrap();
        assert!(!mode.is_reading() || mode.is_update());
        assert!(mode.is_writing());
        assert!(mode.is_truncating());
    }

    #[test]
    fn test_parse_append() {
        let mode = FileMode::parse("a").unwrap();
        assert!(mode.is_append());
        assert!(mode.is_writing());
        assert!(!mode.is_truncating());
    }

    #[test]
    fn test_parse_exclusive() {
        let mode = FileMode::parse("x").unwrap();
        assert!(mode.is_exclusive());
        assert!(mode.is_writing());
    }

    // =========================================================================
    // Parse - Combined modes
    // =========================================================================

    #[test]
    fn test_parse_rb() {
        let mode = FileMode::parse("rb").unwrap();
        assert!(mode.is_reading());
        assert!(mode.is_binary());
    }

    #[test]
    fn test_parse_wb() {
        let mode = FileMode::parse("wb").unwrap();
        assert!(mode.is_writing());
        assert!(mode.is_binary());
    }

    #[test]
    fn test_parse_rt() {
        let mode = FileMode::parse("rt").unwrap();
        assert!(mode.is_reading());
        assert!(mode.is_text());
    }

    #[test]
    fn test_parse_rplus() {
        let mode = FileMode::parse("r+").unwrap();
        assert!(mode.is_reading());
        assert!(mode.is_writing());
        assert!(mode.is_update());
    }

    #[test]
    fn test_parse_wplus() {
        let mode = FileMode::parse("w+").unwrap();
        assert!(mode.is_writing());
        assert!(mode.is_reading());
        assert!(mode.is_truncating());
    }

    #[test]
    fn test_parse_rbplus() {
        let mode = FileMode::parse("rb+").unwrap();
        assert!(mode.is_reading());
        assert!(mode.is_writing());
        assert!(mode.is_binary());
    }

    #[test]
    fn test_parse_ab() {
        let mode = FileMode::parse("ab").unwrap();
        assert!(mode.is_append());
        assert!(mode.is_binary());
    }

    #[test]
    fn test_parse_xb() {
        let mode = FileMode::parse("xb").unwrap();
        assert!(mode.is_exclusive());
        assert!(mode.is_binary());
    }

    // =========================================================================
    // Parse - Error cases
    // =========================================================================

    #[test]
    fn test_parse_empty() {
        assert!(FileMode::parse("").is_err());
    }

    #[test]
    fn test_parse_invalid_char() {
        assert!(FileMode::parse("z").is_err());
    }

    #[test]
    fn test_parse_multiple_base_modes() {
        assert!(FileMode::parse("rw").is_err());
        assert!(FileMode::parse("ra").is_err());
        assert!(FileMode::parse("wa").is_err());
    }

    #[test]
    fn test_parse_text_and_binary() {
        assert!(FileMode::parse("rbt").is_err());
        assert!(FileMode::parse("rtb").is_err());
    }

    // =========================================================================
    // Display
    // =========================================================================

    #[test]
    fn test_display_read() {
        let mode = FileMode::parse("r").unwrap();
        assert_eq!(format!("{}", mode), "rt");
    }

    #[test]
    fn test_display_rb() {
        let mode = FileMode::parse("rb").unwrap();
        assert_eq!(format!("{}", mode), "rb");
    }

    #[test]
    fn test_display_wplus() {
        let mode = FileMode::parse("w+").unwrap();
        assert_eq!(format!("{}", mode), "wt+");
    }

    // =========================================================================
    // to_open_options
    // =========================================================================

    #[test]
    fn test_open_options_read() {
        let mode = FileMode::parse("r").unwrap();
        let _opts = mode.to_open_options(); // Just verify it doesn't panic
    }

    #[test]
    fn test_open_options_write() {
        let mode = FileMode::parse("w").unwrap();
        let _opts = mode.to_open_options();
    }

    #[test]
    fn test_open_options_append() {
        let mode = FileMode::parse("a").unwrap();
        let _opts = mode.to_open_options();
    }

    #[test]
    fn test_open_options_exclusive() {
        let mode = FileMode::parse("x").unwrap();
        let _opts = mode.to_open_options();
    }
}

// =============================================================================
// IoModule Tests
// =============================================================================

mod io_module_tests {
    use super::*;

    #[test]
    fn test_module_name() {
        let module = IoModule::new();
        assert_eq!(module.name(), "io");
    }

    #[test]
    fn test_default_buffer_size() {
        let module = IoModule::new();
        let result = module.get_attr("DEFAULT_BUFFER_SIZE");
        assert!(result.is_ok());
    }

    #[test]
    fn test_seek_constants() {
        let module = IoModule::new();
        assert!(module.get_attr("SEEK_SET").is_ok());
        assert!(module.get_attr("SEEK_CUR").is_ok());
        assert!(module.get_attr("SEEK_END").is_ok());
    }

    #[test]
    fn test_unknown_attr() {
        let module = IoModule::new();
        assert!(module.get_attr("nonexistent").is_err());
    }

    #[test]
    fn test_dir() {
        let module = IoModule::new();
        let attrs = module.dir();
        assert!(attrs.iter().any(|a| a.as_ref() == "StringIO"));
        assert!(attrs.iter().any(|a| a.as_ref() == "BytesIO"));
        assert!(attrs.iter().any(|a| a.as_ref() == "DEFAULT_BUFFER_SIZE"));
    }

    #[test]
    fn test_default_impl() {
        let module = IoModule::default();
        assert_eq!(module.name(), "io");
    }
}

// =============================================================================
// Constants Tests
// =============================================================================

mod constants_tests {
    use super::*;

    #[test]
    fn test_seek_set() {
        assert_eq!(SEEK_SET, 0);
    }

    #[test]
    fn test_seek_cur() {
        assert_eq!(SEEK_CUR, 1);
    }

    #[test]
    fn test_seek_end() {
        assert_eq!(SEEK_END, 2);
    }

    #[test]
    fn test_default_buffer_size() {
        assert_eq!(DEFAULT_BUFFER_SIZE, 8192);
    }

    #[test]
    fn test_seek_constants_distinct() {
        assert_ne!(SEEK_SET, SEEK_CUR);
        assert_ne!(SEEK_CUR, SEEK_END);
        assert_ne!(SEEK_SET, SEEK_END);
    }
}

// =============================================================================
// IoError Tests
// =============================================================================

mod io_error_tests {
    use super::*;

    #[test]
    fn test_value_error_display() {
        let e = IoError::ValueError("test".to_string());
        assert!(e.to_string().contains("ValueError"));
        assert!(e.to_string().contains("test"));
    }

    #[test]
    fn test_unsupported_display() {
        let e = IoError::UnsupportedOperation("not supported".to_string());
        assert!(e.to_string().contains("UnsupportedOperation"));
    }

    #[test]
    fn test_os_error_display() {
        let e = IoError::OsError("disk full".to_string());
        assert!(e.to_string().contains("OSError"));
    }

    #[test]
    fn test_error_equality() {
        let e1 = IoError::ValueError("test".to_string());
        let e2 = IoError::ValueError("test".to_string());
        assert_eq!(e1, e2);
    }

    #[test]
    fn test_error_inequality() {
        let e1 = IoError::ValueError("a".to_string());
        let e2 = IoError::ValueError("b".to_string());
        assert_ne!(e1, e2);
    }
}
