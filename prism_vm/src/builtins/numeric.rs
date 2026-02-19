//! Numeric formatting builtins (bin, hex, oct, complex).
//!
//! High-performance implementations with zero-allocation fast paths
//! for common integer sizes. All functions are Python 3.12 compatible.
//!
//! # Performance Characteristics
//!
//! - **bin/hex/oct**: Zero heap allocation for integers ≤64 bits
//! - **Lookup tables**: Pre-computed nibble→char mappings avoid branching
//! - **Exact sizing**: Pre-compute output length to avoid reallocations
//!
//! # Python Semantics
//!
//! These builtins accept any object with `__index__` protocol, but for
//! maximum performance, we fast-path native integers. The fallback
//! path handles objects with `__index__` when the object system is wired.

use super::BuiltinError;
use prism_core::Value;
use prism_core::intern::{intern, interned_by_ptr};
use prism_runtime::object::type_obj::TypeId;

// =============================================================================
// Lookup Tables (Zero-Branch Hex Conversion)
// =============================================================================

/// Lowercase hex digits lookup table.
/// Index by nibble value (0-15) to get ASCII char.
const HEX_CHARS_LOWER: [u8; 16] = *b"0123456789abcdef";

/// Uppercase hex digits lookup table (for potential future use).
#[allow(dead_code)]
const HEX_CHARS_UPPER: [u8; 16] = *b"0123456789ABCDEF";

/// Octal digits lookup table.
/// Index by 3-bit value (0-7) to get ASCII char.
const OCT_CHARS: [u8; 8] = *b"01234567";

// =============================================================================
// Stack Buffer for Zero-Allocation Formatting
// =============================================================================

/// Maximum buffer size needed for formatting.
/// - Binary: 2 (prefix) + 64 (bits) + 1 (sign) = 67 chars
/// - Hex: 2 (prefix) + 16 (nibbles) + 1 (sign) = 19 chars  
/// - Octal: 2 (prefix) + 22 (max for i64) + 1 (sign) = 25 chars
const FORMAT_BUFFER_SIZE: usize = 72;

/// Stack-allocated buffer for formatting operations.
/// Avoids heap allocation for all standard integer sizes.
struct FormatBuffer {
    /// Fixed-size buffer on the stack.
    data: [u8; FORMAT_BUFFER_SIZE],
    /// Current write position (we write right-to-left).
    pos: usize,
}

impl FormatBuffer {
    /// Create a new empty buffer positioned at the end.
    #[inline(always)]
    const fn new() -> Self {
        Self {
            data: [0u8; FORMAT_BUFFER_SIZE],
            pos: FORMAT_BUFFER_SIZE,
        }
    }

    /// Push a byte to the buffer (right-to-left).
    #[inline(always)]
    fn push(&mut self, byte: u8) {
        debug_assert!(self.pos > 0, "FormatBuffer overflow");
        self.pos -= 1;
        self.data[self.pos] = byte;
    }

    /// Push two bytes (prefix like "0b", "0x", "0o").
    #[inline(always)]
    fn push_prefix(&mut self, prefix: &[u8; 2]) {
        self.push(prefix[1]);
        self.push(prefix[0]);
    }

    /// Get the formatted string as a byte slice.
    #[inline(always)]
    fn as_bytes(&self) -> &[u8] {
        &self.data[self.pos..]
    }

    /// Get the formatted string as a str (unsafe but we only write ASCII).
    #[inline(always)]
    fn as_str(&self) -> &str {
        // SAFETY: We only write ASCII bytes
        unsafe { std::str::from_utf8_unchecked(self.as_bytes()) }
    }

    /// Get the length of the formatted content.
    #[inline(always)]
    fn len(&self) -> usize {
        FORMAT_BUFFER_SIZE - self.pos
    }
}

// =============================================================================
// bin() - Binary String Representation
// =============================================================================

/// Builtin bin(x) function.
///
/// Returns the binary representation of an integer as a string.
/// Output format: `"0b..."` for positive, `"-0b..."` for negative.
///
/// # Python Semantics
/// - `bin(0)` → `"0b0"`
/// - `bin(5)` → `"0b101"`  
/// - `bin(-5)` → `"-0b101"`
/// - Non-integers raise TypeError
///
/// # Performance
/// - Zero heap allocation for stack formatting
/// - Single pass bit extraction with shifts
/// - Exact output length pre-computed
pub fn builtin_bin(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() != 1 {
        return Err(BuiltinError::TypeError(format!(
            "bin() takes exactly one argument ({} given)",
            args.len()
        )));
    }

    // Fast path: native integer
    if let Some(n) = args[0].as_int() {
        return format_binary(n);
    }

    // Bool is valid via __index__ (True=1, False=0)
    if let Some(b) = args[0].as_bool() {
        return format_binary(if b { 1 } else { 0 });
    }

    // TODO: Handle objects with __index__ protocol
    Err(BuiltinError::TypeError(format!(
        "'{}' object cannot be interpreted as an integer",
        type_name_of(&args[0])
    )))
}

/// Format an i64 as binary string "0b...".
///
/// Uses zero-allocation stack buffer and right-to-left filling.
#[inline]
fn format_binary(n: i64) -> Result<Value, BuiltinError> {
    let mut buf = FormatBuffer::new();
    let negative = n < 0;

    // Work with absolute value (handle i64::MIN specially)
    let abs_val = if n == i64::MIN {
        // i64::MIN has no positive counterpart, work with u64
        i64::MIN as u64
    } else {
        n.unsigned_abs()
    };

    if abs_val == 0 {
        // Special case: "0b0"
        buf.push(b'0');
    } else {
        // Extract bits right-to-left
        let mut val = abs_val;
        while val > 0 {
            buf.push(b'0' + (val & 1) as u8);
            val >>= 1;
        }
    }

    // Add prefix "0b"
    buf.push_prefix(b"0b");

    // Add minus sign if negative
    if negative {
        buf.push(b'-');
    }

    Ok(interned_string_value(buf.as_str()))
}

// =============================================================================
// hex() - Hexadecimal String Representation
// =============================================================================

/// Builtin hex(x) function.
///
/// Returns the hexadecimal representation of an integer as a string.
/// Output format: `"0x..."` for positive, `"-0x..."` for negative.
/// Uses lowercase letters (a-f).
///
/// # Python Semantics
/// - `hex(0)` → `"0x0"`
/// - `hex(255)` → `"0xff"`
/// - `hex(-255)` → `"-0xff"`
/// - Non-integers raise TypeError
///
/// # Performance
/// - Lookup table for nibble→char (no branching)
/// - Zero heap allocation for formatting
/// - Single pass nibble extraction
pub fn builtin_hex(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() != 1 {
        return Err(BuiltinError::TypeError(format!(
            "hex() takes exactly one argument ({} given)",
            args.len()
        )));
    }

    // Fast path: native integer
    if let Some(n) = args[0].as_int() {
        return format_hex(n);
    }

    // Bool is valid via __index__
    if let Some(b) = args[0].as_bool() {
        return format_hex(if b { 1 } else { 0 });
    }

    Err(BuiltinError::TypeError(format!(
        "'{}' object cannot be interpreted as an integer",
        type_name_of(&args[0])
    )))
}

/// Format an i64 as hexadecimal string "0x...".
///
/// Uses lookup table for zero-branch nibble conversion.
#[inline]
fn format_hex(n: i64) -> Result<Value, BuiltinError> {
    let mut buf = FormatBuffer::new();
    let negative = n < 0;

    // Work with absolute value
    let abs_val = if n == i64::MIN {
        i64::MIN as u64
    } else {
        n.unsigned_abs()
    };

    if abs_val == 0 {
        buf.push(b'0');
    } else {
        let mut val = abs_val;
        while val > 0 {
            let nibble = (val & 0xF) as usize;
            buf.push(HEX_CHARS_LOWER[nibble]);
            val >>= 4;
        }
    }

    buf.push_prefix(b"0x");

    if negative {
        buf.push(b'-');
    }

    Ok(interned_string_value(buf.as_str()))
}

// =============================================================================
// oct() - Octal String Representation
// =============================================================================

/// Builtin oct(x) function.
///
/// Returns the octal representation of an integer as a string.
/// Output format: `"0o..."` for positive, `"-0o..."` for negative.
///
/// # Python Semantics
/// - `oct(0)` → `"0o0"`
/// - `oct(8)` → `"0o10"`
/// - `oct(-8)` → `"-0o10"`
/// - Non-integers raise TypeError
///
/// # Performance
/// - Lookup table for 3-bit→char
/// - Zero heap allocation for formatting
pub fn builtin_oct(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() != 1 {
        return Err(BuiltinError::TypeError(format!(
            "oct() takes exactly one argument ({} given)",
            args.len()
        )));
    }

    // Fast path: native integer
    if let Some(n) = args[0].as_int() {
        return format_oct(n);
    }

    // Bool is valid via __index__
    if let Some(b) = args[0].as_bool() {
        return format_oct(if b { 1 } else { 0 });
    }

    Err(BuiltinError::TypeError(format!(
        "'{}' object cannot be interpreted as an integer",
        type_name_of(&args[0])
    )))
}

/// Format an i64 as octal string "0o...".
#[inline]
fn format_oct(n: i64) -> Result<Value, BuiltinError> {
    let mut buf = FormatBuffer::new();
    let negative = n < 0;

    let abs_val = if n == i64::MIN {
        i64::MIN as u64
    } else {
        n.unsigned_abs()
    };

    if abs_val == 0 {
        buf.push(b'0');
    } else {
        let mut val = abs_val;
        while val > 0 {
            let digit = (val & 0x7) as usize;
            buf.push(OCT_CHARS[digit]);
            val >>= 3;
        }
    }

    buf.push_prefix(b"0o");

    if negative {
        buf.push(b'-');
    }

    Ok(interned_string_value(buf.as_str()))
}

#[inline]
fn interned_string_value(s: &str) -> Value {
    Value::string(intern(s))
}

// =============================================================================
// complex() - Complex Number Constructor
// =============================================================================

/// Builtin complex([real[, imag]]) function.
///
/// Creates a complex number from real and imaginary parts.
///
/// # Python Semantics
/// - `complex()` → `0j`
/// - `complex(1)` → `(1+0j)`
/// - `complex(1, 2)` → `(1+2j)`
/// - `complex("1+2j")` → `(1+2j)` (string parsing)
///
/// # Implementation Note
/// Complex numbers require a ComplexObject type in the runtime.
/// This is a placeholder until that type is implemented.
pub fn builtin_complex(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() > 2 {
        return Err(BuiltinError::TypeError(format!(
            "complex() takes at most 2 arguments ({} given)",
            args.len()
        )));
    }

    // Default values
    let (real, imag) = match args.len() {
        0 => (0.0_f64, 0.0_f64),
        1 => {
            let r = extract_float_for_complex(&args[0], "real")?;
            (r, 0.0)
        }
        2 => {
            let r = extract_float_for_complex(&args[0], "real")?;
            let i = extract_float_for_complex(&args[1], "imag")?;
            (r, i)
        }
        _ => unreachable!(),
    };

    // TODO: Create ComplexObject and return as Value
    // For now, store as a tuple-like representation or placeholder
    let _ = (real, imag);

    Err(BuiltinError::NotImplemented(
        "complex() requires ComplexObject type".to_string(),
    ))
}

/// Extract a float value for complex number construction.
///
/// Accepts int, float, or bool. Returns error for other types.
#[inline]
fn extract_float_for_complex(val: &Value, part_name: &str) -> Result<f64, BuiltinError> {
    if let Some(f) = val.as_float() {
        return Ok(f);
    }
    if let Some(i) = val.as_int() {
        return Ok(i as f64);
    }
    if let Some(b) = val.as_bool() {
        return Ok(if b { 1.0 } else { 0.0 });
    }

    Err(BuiltinError::TypeError(format!(
        "complex() {} part must be a number, not '{}'",
        part_name,
        type_name_of(val)
    )))
}

/// Get the type name of a value for error messages.
#[inline]
fn type_name_of(val: &Value) -> &'static str {
    if val.is_none() {
        "NoneType"
    } else if val.is_bool() {
        "bool"
    } else if val.is_int() {
        "int"
    } else if val.is_float() {
        "float"
    } else if val.is_object() {
        val.as_object_ptr()
            .map(crate::ops::objects::extract_type_id)
            .map(TypeId::name)
            .unwrap_or("object")
    } else {
        "unknown"
    }
}

// =============================================================================
// Internal Formatting Functions (Public for Testing)
// =============================================================================

/// Format an integer as binary and return the string.
///
/// This is the low-level implementation exposed for testing.
/// The actual builtin wraps this to return a Value.
pub fn format_binary_string(n: i64) -> String {
    let mut buf = FormatBuffer::new();
    let negative = n < 0;

    let abs_val = if n == i64::MIN {
        i64::MIN as u64
    } else {
        n.unsigned_abs()
    };

    if abs_val == 0 {
        buf.push(b'0');
    } else {
        let mut val = abs_val;
        while val > 0 {
            buf.push(b'0' + (val & 1) as u8);
            val >>= 1;
        }
    }

    buf.push_prefix(b"0b");

    if negative {
        buf.push(b'-');
    }

    buf.as_str().to_string()
}

/// Format an integer as hexadecimal and return the string.
pub fn format_hex_string(n: i64) -> String {
    let mut buf = FormatBuffer::new();
    let negative = n < 0;

    let abs_val = if n == i64::MIN {
        i64::MIN as u64
    } else {
        n.unsigned_abs()
    };

    if abs_val == 0 {
        buf.push(b'0');
    } else {
        let mut val = abs_val;
        while val > 0 {
            let nibble = (val & 0xF) as usize;
            buf.push(HEX_CHARS_LOWER[nibble]);
            val >>= 4;
        }
    }

    buf.push_prefix(b"0x");

    if negative {
        buf.push(b'-');
    }

    buf.as_str().to_string()
}

/// Format an integer as octal and return the string.
pub fn format_oct_string(n: i64) -> String {
    let mut buf = FormatBuffer::new();
    let negative = n < 0;

    let abs_val = if n == i64::MIN {
        i64::MIN as u64
    } else {
        n.unsigned_abs()
    };

    if abs_val == 0 {
        buf.push(b'0');
    } else {
        let mut val = abs_val;
        while val > 0 {
            let digit = (val & 0x7) as usize;
            buf.push(OCT_CHARS[digit]);
            val >>= 3;
        }
    }

    buf.push_prefix(b"0o");

    if negative {
        buf.push(b'-');
    }

    buf.as_str().to_string()
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn tagged_string_value_to_rust_string(value: Value) -> String {
        let ptr = value
            .as_string_object_ptr()
            .expect("expected interned string value");
        let interned =
            interned_by_ptr(ptr as *const u8).expect("interned string pointer should resolve");
        interned.as_str().to_string()
    }

    // =========================================================================
    // FormatBuffer Tests
    // =========================================================================

    #[test]
    fn test_format_buffer_new() {
        let buf = FormatBuffer::new();
        assert_eq!(buf.len(), 0);
        assert_eq!(buf.as_str(), "");
    }

    #[test]
    fn test_format_buffer_push_single() {
        let mut buf = FormatBuffer::new();
        buf.push(b'X');
        assert_eq!(buf.len(), 1);
        assert_eq!(buf.as_str(), "X");
    }

    #[test]
    fn test_format_buffer_push_multiple() {
        let mut buf = FormatBuffer::new();
        buf.push(b'C');
        buf.push(b'B');
        buf.push(b'A');
        assert_eq!(buf.as_str(), "ABC"); // Right-to-left, so reversed
    }

    #[test]
    fn test_format_buffer_push_prefix() {
        let mut buf = FormatBuffer::new();
        buf.push(b'1');
        buf.push_prefix(b"0x");
        assert_eq!(buf.as_str(), "0x1");
    }

    // =========================================================================
    // bin() Argument Validation Tests
    // =========================================================================

    #[test]
    fn test_bin_no_args() {
        let result = builtin_bin(&[]);
        assert!(result.is_err());
        match result {
            Err(BuiltinError::TypeError(msg)) => {
                assert!(msg.contains("exactly one argument"));
            }
            _ => panic!("Expected TypeError"),
        }
    }

    #[test]
    fn test_bin_too_many_args() {
        let result = builtin_bin(&[Value::int(1).unwrap(), Value::int(2).unwrap()]);
        assert!(result.is_err());
    }

    #[test]
    fn test_bin_float_error() {
        let result = builtin_bin(&[Value::float(3.14)]);
        assert!(result.is_err());
        match result {
            Err(BuiltinError::TypeError(msg)) => {
                assert!(msg.contains("float"));
            }
            _ => panic!("Expected TypeError"),
        }
    }

    #[test]
    fn test_bin_returns_tagged_string_value() {
        let result = builtin_bin(&[Value::int(13).unwrap()]).unwrap();
        assert_eq!(tagged_string_value_to_rust_string(result), "0b1101");
    }

    #[test]
    fn test_bin_none_error() {
        let result = builtin_bin(&[Value::none()]);
        assert!(result.is_err());
    }

    // =========================================================================
    // bin() Formatting Tests
    // =========================================================================

    #[test]
    fn test_bin_zero() {
        assert_eq!(format_binary_string(0), "0b0");
    }

    #[test]
    fn test_bin_one() {
        assert_eq!(format_binary_string(1), "0b1");
    }

    #[test]
    fn test_bin_two() {
        assert_eq!(format_binary_string(2), "0b10");
    }

    #[test]
    fn test_bin_three() {
        assert_eq!(format_binary_string(3), "0b11");
    }

    #[test]
    fn test_bin_five() {
        assert_eq!(format_binary_string(5), "0b101");
    }

    #[test]
    fn test_bin_255() {
        assert_eq!(format_binary_string(255), "0b11111111");
    }

    #[test]
    fn test_bin_256() {
        assert_eq!(format_binary_string(256), "0b100000000");
    }

    #[test]
    fn test_bin_negative_one() {
        assert_eq!(format_binary_string(-1), "-0b1");
    }

    #[test]
    fn test_bin_negative_five() {
        assert_eq!(format_binary_string(-5), "-0b101");
    }

    #[test]
    fn test_bin_negative_127() {
        assert_eq!(format_binary_string(-127), "-0b1111111");
    }

    #[test]
    fn test_bin_negative_128() {
        assert_eq!(format_binary_string(-128), "-0b10000000");
    }

    #[test]
    fn test_bin_i64_max() {
        assert_eq!(
            format_binary_string(i64::MAX),
            "0b111111111111111111111111111111111111111111111111111111111111111"
        );
    }

    #[test]
    fn test_bin_i64_min() {
        // i64::MIN = -9223372036854775808 = 1 followed by 63 zeros
        assert_eq!(
            format_binary_string(i64::MIN),
            "-0b1000000000000000000000000000000000000000000000000000000000000000"
        );
    }

    #[test]
    fn test_bin_powers_of_two() {
        assert_eq!(format_binary_string(8), "0b1000");
        assert_eq!(format_binary_string(16), "0b10000");
        assert_eq!(format_binary_string(32), "0b100000");
        assert_eq!(format_binary_string(64), "0b1000000");
        assert_eq!(format_binary_string(128), "0b10000000");
        assert_eq!(format_binary_string(1024), "0b10000000000");
    }

    // =========================================================================
    // hex() Argument Validation Tests
    // =========================================================================

    #[test]
    fn test_hex_no_args() {
        let result = builtin_hex(&[]);
        assert!(result.is_err());
    }

    #[test]
    fn test_hex_too_many_args() {
        let result = builtin_hex(&[Value::int(1).unwrap(), Value::int(2).unwrap()]);
        assert!(result.is_err());
    }

    #[test]
    fn test_hex_float_error() {
        let result = builtin_hex(&[Value::float(3.14)]);
        assert!(result.is_err());
    }

    #[test]
    fn test_hex_returns_tagged_string_value() {
        let result = builtin_hex(&[Value::int(255).unwrap()]).unwrap();
        assert_eq!(tagged_string_value_to_rust_string(result), "0xff");
    }

    // =========================================================================
    // hex() Formatting Tests
    // =========================================================================

    #[test]
    fn test_hex_zero() {
        assert_eq!(format_hex_string(0), "0x0");
    }

    #[test]
    fn test_hex_one() {
        assert_eq!(format_hex_string(1), "0x1");
    }

    #[test]
    fn test_hex_fifteen() {
        assert_eq!(format_hex_string(15), "0xf");
    }

    #[test]
    fn test_hex_sixteen() {
        assert_eq!(format_hex_string(16), "0x10");
    }

    #[test]
    fn test_hex_255() {
        assert_eq!(format_hex_string(255), "0xff");
    }

    #[test]
    fn test_hex_256() {
        assert_eq!(format_hex_string(256), "0x100");
    }

    #[test]
    fn test_hex_0xdeadbeef() {
        assert_eq!(format_hex_string(0xDEADBEEF), "0xdeadbeef");
    }

    #[test]
    fn test_hex_negative_one() {
        assert_eq!(format_hex_string(-1), "-0x1");
    }

    #[test]
    fn test_hex_negative_255() {
        assert_eq!(format_hex_string(-255), "-0xff");
    }

    #[test]
    fn test_hex_i64_max() {
        assert_eq!(format_hex_string(i64::MAX), "0x7fffffffffffffff");
    }

    #[test]
    fn test_hex_i64_min() {
        assert_eq!(format_hex_string(i64::MIN), "-0x8000000000000000");
    }

    #[test]
    fn test_hex_common_values() {
        assert_eq!(format_hex_string(10), "0xa");
        assert_eq!(format_hex_string(11), "0xb");
        assert_eq!(format_hex_string(12), "0xc");
        assert_eq!(format_hex_string(13), "0xd");
        assert_eq!(format_hex_string(14), "0xe");
    }

    // =========================================================================
    // oct() Argument Validation Tests
    // =========================================================================

    #[test]
    fn test_oct_no_args() {
        let result = builtin_oct(&[]);
        assert!(result.is_err());
    }

    #[test]
    fn test_oct_too_many_args() {
        let result = builtin_oct(&[Value::int(1).unwrap(), Value::int(2).unwrap()]);
        assert!(result.is_err());
    }

    #[test]
    fn test_oct_float_error() {
        let result = builtin_oct(&[Value::float(3.14)]);
        assert!(result.is_err());
    }

    #[test]
    fn test_oct_returns_tagged_string_value() {
        let result = builtin_oct(&[Value::int(9).unwrap()]).unwrap();
        assert_eq!(tagged_string_value_to_rust_string(result), "0o11");
    }

    // =========================================================================
    // oct() Formatting Tests
    // =========================================================================

    #[test]
    fn test_oct_zero() {
        assert_eq!(format_oct_string(0), "0o0");
    }

    #[test]
    fn test_oct_one() {
        assert_eq!(format_oct_string(1), "0o1");
    }

    #[test]
    fn test_oct_seven() {
        assert_eq!(format_oct_string(7), "0o7");
    }

    #[test]
    fn test_oct_eight() {
        assert_eq!(format_oct_string(8), "0o10");
    }

    #[test]
    fn test_oct_63() {
        assert_eq!(format_oct_string(63), "0o77");
    }

    #[test]
    fn test_oct_64() {
        assert_eq!(format_oct_string(64), "0o100");
    }

    #[test]
    fn test_oct_255() {
        assert_eq!(format_oct_string(255), "0o377");
    }

    #[test]
    fn test_oct_negative_one() {
        assert_eq!(format_oct_string(-1), "-0o1");
    }

    #[test]
    fn test_oct_negative_eight() {
        assert_eq!(format_oct_string(-8), "-0o10");
    }

    #[test]
    fn test_oct_i64_max() {
        assert_eq!(format_oct_string(i64::MAX), "0o777777777777777777777");
    }

    #[test]
    fn test_oct_i64_min() {
        assert_eq!(format_oct_string(i64::MIN), "-0o1000000000000000000000");
    }

    #[test]
    fn test_oct_powers_of_eight() {
        assert_eq!(format_oct_string(8), "0o10");
        assert_eq!(format_oct_string(64), "0o100");
        assert_eq!(format_oct_string(512), "0o1000");
        assert_eq!(format_oct_string(4096), "0o10000");
    }

    // =========================================================================
    // complex() Argument Validation Tests
    // =========================================================================

    #[test]
    fn test_complex_too_many_args() {
        let result = builtin_complex(&[
            Value::int(1).unwrap(),
            Value::int(2).unwrap(),
            Value::int(3).unwrap(),
        ]);
        assert!(result.is_err());
    }

    #[test]
    fn test_complex_none_error() {
        let result = builtin_complex(&[Value::none()]);
        assert!(result.is_err());
        match result {
            Err(BuiltinError::TypeError(msg)) => {
                assert!(msg.contains("NoneType"));
            }
            _ => panic!("Expected TypeError"),
        }
    }

    // =========================================================================
    // Boolean Input Tests (__index__ protocol)
    // =========================================================================

    #[test]
    fn test_bin_bool_true() {
        let result = builtin_bin(&[Value::bool(true)]).unwrap();
        assert_eq!(tagged_string_value_to_rust_string(result), "0b1");
    }

    #[test]
    fn test_bin_bool_false() {
        let result = builtin_bin(&[Value::bool(false)]).unwrap();
        assert_eq!(tagged_string_value_to_rust_string(result), "0b0");
    }

    #[test]
    fn test_hex_bool_true() {
        let result = builtin_hex(&[Value::bool(true)]).unwrap();
        assert_eq!(tagged_string_value_to_rust_string(result), "0x1");
    }

    #[test]
    fn test_oct_bool_true() {
        let result = builtin_oct(&[Value::bool(true)]).unwrap();
        assert_eq!(tagged_string_value_to_rust_string(result), "0o1");
    }

    // =========================================================================
    // Helper Function Tests
    // =========================================================================

    #[test]
    fn test_extract_float_from_int() {
        let val = Value::int(42).unwrap();
        let result = extract_float_for_complex(&val, "test");
        assert_eq!(result.unwrap(), 42.0);
    }

    #[test]
    fn test_extract_float_from_float() {
        let val = Value::float(3.14);
        let result = extract_float_for_complex(&val, "test");
        assert!((result.unwrap() - 3.14).abs() < 1e-10);
    }

    #[test]
    fn test_extract_float_from_bool() {
        let val = Value::bool(true);
        let result = extract_float_for_complex(&val, "test");
        assert_eq!(result.unwrap(), 1.0);

        let val = Value::bool(false);
        let result = extract_float_for_complex(&val, "test");
        assert_eq!(result.unwrap(), 0.0);
    }

    #[test]
    fn test_type_name_of() {
        assert_eq!(type_name_of(&Value::none()), "NoneType");
        assert_eq!(type_name_of(&Value::bool(true)), "bool");
        assert_eq!(type_name_of(&Value::int(1).unwrap()), "int");
        assert_eq!(type_name_of(&Value::float(1.0)), "float");
    }

    // =========================================================================
    // Cross-Format Consistency Tests
    // =========================================================================

    #[test]
    fn test_all_formats_for_zero() {
        assert_eq!(format_binary_string(0), "0b0");
        assert_eq!(format_hex_string(0), "0x0");
        assert_eq!(format_oct_string(0), "0o0");
    }

    #[test]
    fn test_all_formats_for_255() {
        assert_eq!(format_binary_string(255), "0b11111111");
        assert_eq!(format_hex_string(255), "0xff");
        assert_eq!(format_oct_string(255), "0o377");
    }

    #[test]
    fn test_all_formats_negative_255() {
        assert_eq!(format_binary_string(-255), "-0b11111111");
        assert_eq!(format_hex_string(-255), "-0xff");
        assert_eq!(format_oct_string(-255), "-0o377");
    }

    // =========================================================================
    // Lookup Table Integrity Tests
    // =========================================================================

    #[test]
    fn test_hex_lookup_table_integrity() {
        assert_eq!(HEX_CHARS_LOWER[0], b'0');
        assert_eq!(HEX_CHARS_LOWER[9], b'9');
        assert_eq!(HEX_CHARS_LOWER[10], b'a');
        assert_eq!(HEX_CHARS_LOWER[15], b'f');
    }

    #[test]
    fn test_oct_lookup_table_integrity() {
        assert_eq!(OCT_CHARS[0], b'0');
        assert_eq!(OCT_CHARS[7], b'7');
    }

    // =========================================================================
    // Edge Case and Boundary Tests
    // =========================================================================

    #[test]
    fn test_bin_byte_boundary() {
        // Values at byte boundaries
        assert_eq!(format_binary_string(127), "0b1111111");
        assert_eq!(format_binary_string(128), "0b10000000");
    }

    #[test]
    fn test_hex_nibble_boundaries() {
        for i in 0..=15 {
            let s = format_hex_string(i);
            assert!(s.starts_with("0x"));
            assert_eq!(s.len(), 3); // "0x" + 1 char
        }
    }

    #[test]
    fn test_oct_3bit_boundaries() {
        for i in 0..=7 {
            let s = format_oct_string(i);
            assert!(s.starts_with("0o"));
            assert_eq!(s.len(), 3); // "0o" + 1 char
        }
    }
}
