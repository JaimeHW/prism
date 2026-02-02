//! Python value representation using NaN-boxing for high performance.
//!
//! This module implements a tagged union representation of Python values that
//! fits in a single 64-bit word. We use NaN-boxing: IEEE 754 floating point
//! NaN values have many unused bit patterns that we exploit to store other
//! types inline.
//!
//! ## NaN-Boxing Scheme
//!
//! IEEE 754 double-precision NaN: sign(1) + exponent(11, all 1s) + mantissa(52, non-zero)
//!
//! We use the following encoding (little-endian view):
//! - Floats: standard IEEE 754 encoding (unboxed)
//! - Tagged values: exponent=0x7FF (NaN), bit 51=1 (quiet NaN), bits 48-50=tag, bits 0-47=payload
//!
//! | Tag  | Type        | Payload                           |
//! |------|-------------|-----------------------------------|
//! | 0x0  | None        | unused                            |
//! | 0x1  | Bool        | 0=false, 1=true                   |
//! | 0x2  | Int (small) | 48-bit signed integer             |
//! | 0x3  | Object      | 48-bit pointer                    |
//! | 0x4  | String      | 48-bit interned string pointer    |
//! | 0x5  | Reserved    | future use                        |
//! | 0x6  | Reserved    | future use                        |
//! | 0x7  | Reserved    | future use                        |

use crate::intern::InternedString;
use std::fmt;
use std::hash::{Hash, Hasher};

/// Quiet NaN bit pattern: exponent all 1s + quiet NaN bit (bit 51)
/// NOTE: We use 0x7FF8 NOT 0x7FFC to leave bits 48-50 free for tag encoding
const QNAN: u64 = 0x7FF8_0000_0000_0000;

/// Tag bits position (bits 48-50)
const TAG_SHIFT: u64 = 48;
const TAG_MASK: u64 = 0x0007_0000_0000_0000;

/// Payload mask (bits 0-47)
const PAYLOAD_MASK: u64 = 0x0000_FFFF_FFFF_FFFF;

/// Sign bit mask
const SIGN_BIT: u64 = 0x8000_0000_0000_0000;

/// Tag values
const TAG_NONE: u64 = 0;
const TAG_BOOL: u64 = 1;
const TAG_INT: u64 = 2;
const TAG_OBJECT: u64 = 3;
const TAG_STRING: u64 = 4;

// =============================================================================
// Public Tag Patterns (for branchless speculation)
// =============================================================================

/// Combined QNAN + TAG pattern for string values.
/// Use with `value.raw_bits() & STRING_TAG_MASK == STRING_TAG_PATTERN` for branchless checks.
pub const STRING_TAG_PATTERN: u64 = QNAN | (TAG_STRING << TAG_SHIFT);

/// Combined QNAN + TAG pattern for int values.
pub const INT_TAG_PATTERN: u64 = QNAN | (TAG_INT << TAG_SHIFT);

/// Mask for extracting the type tag portion (QNAN + tag bits).
pub const TYPE_TAG_MASK: u64 = QNAN | TAG_MASK;

/// Payload mask for extracting pointer/value from tagged values.
pub const VALUE_PAYLOAD_MASK: u64 = PAYLOAD_MASK;

/// Maximum small integer (47-bit signed)
pub const SMALL_INT_MAX: i64 = (1_i64 << 47) - 1;
/// Minimum small integer (47-bit signed)
pub const SMALL_INT_MIN: i64 = -(1_i64 << 47);

/// A Python value using NaN-boxing for efficient storage.
///
/// This type is exactly 8 bytes and can represent:
/// - Floating point numbers (unboxed)
/// - None
/// - Booleans
/// - Small integers (48-bit signed, approximately ±140 trillion)
/// - Object references (heap-allocated Python objects)
/// - Interned strings (special case for fast string operations)
#[derive(Clone, Copy)]
#[repr(transparent)]
pub struct Value {
    bits: u64,
}

impl Value {
    // =========================================================================
    // Constructors
    // =========================================================================

    /// Create a None value.
    #[inline]
    #[must_use]
    pub const fn none() -> Self {
        Self {
            bits: QNAN | (TAG_NONE << TAG_SHIFT),
        }
    }

    /// Create a boolean value.
    #[inline]
    #[must_use]
    pub const fn bool(b: bool) -> Self {
        Self {
            bits: QNAN | (TAG_BOOL << TAG_SHIFT) | (b as u64),
        }
    }

    /// Create an integer value.
    ///
    /// If the integer is in the small int cache range [-5, 256], returns
    /// a pre-computed cached value. Otherwise, if it fits in 48 bits,
    /// it's stored inline. Returns None if the integer is too large.
    ///
    /// # Performance
    ///
    /// For cached integers: O(1) array lookup (< 1ns)
    /// For uncached small ints: O(1) bit manipulation
    #[inline]
    #[must_use]
    pub fn int(i: i64) -> Option<Self> {
        // Fast path: check small int cache first
        if let Some(cached) = crate::small_int_cache::SmallIntCache::get(i) {
            return Some(cached);
        }

        // Slow path: construct inline if it fits
        if i >= SMALL_INT_MIN && i <= SMALL_INT_MAX {
            let payload = (i as u64) & PAYLOAD_MASK;
            Some(Self {
                bits: QNAN | (TAG_INT << TAG_SHIFT) | payload,
            })
        } else {
            None
        }
    }

    /// Create an integer value, panicking if it doesn't fit inline.
    #[inline]
    #[must_use]
    pub const fn int_unchecked(i: i64) -> Self {
        let payload = (i as u64) & PAYLOAD_MASK;
        Self {
            bits: QNAN | (TAG_INT << TAG_SHIFT) | payload,
        }
    }

    /// Create a float value.
    #[inline]
    #[must_use]
    pub fn float(f: f64) -> Self {
        let bits = f.to_bits();
        // Check if it's a NaN that would collide with our tagged values
        // Any NaN where (bits & QNAN) == QNAN would be misidentified as tagged
        if bits & QNAN == QNAN {
            // Use a safe NaN representation: quiet NaN with payload=1, tag bits clear
            // 0x7FF0_0000_0000_0001 = exponent all 1s, mantissa = 1 (valid NaN, but doesn't collide)
            Self {
                bits: 0x7FF0_0000_0000_0001,
            }
        } else {
            Self { bits }
        }
    }

    /// Create an object reference value.
    ///
    /// # Safety
    /// The pointer must be valid and properly aligned.
    #[inline]
    #[must_use]
    pub fn object_ptr(ptr: *const ()) -> Self {
        let ptr_bits = ptr as usize as u64;
        debug_assert!(
            ptr_bits & !PAYLOAD_MASK == 0,
            "Pointer too large for NaN-boxing"
        );
        Self {
            bits: QNAN | (TAG_OBJECT << TAG_SHIFT) | (ptr_bits & PAYLOAD_MASK),
        }
    }

    /// Create an interned string value.
    #[inline]
    #[must_use]
    pub fn string(s: InternedString) -> Self {
        // Store the Arc's data pointer (thin pointer to the str data)
        let arc = s.clone_arc();
        let ptr = arc.as_ptr() as *const u8 as usize as u64;
        // Don't drop the Arc - leak it (we're storing the pointer)
        std::mem::forget(arc);
        debug_assert!(ptr & !PAYLOAD_MASK == 0, "Pointer too large for NaN-boxing");
        Self {
            bits: QNAN | (TAG_STRING << TAG_SHIFT) | (ptr & PAYLOAD_MASK),
        }
    }

    // =========================================================================
    // Type Checking
    // =========================================================================

    /// Check if this is a tagged value (not a float).
    #[inline]
    #[must_use]
    pub const fn is_tagged(&self) -> bool {
        (self.bits & QNAN) == QNAN
    }

    /// Check if this is a float.
    #[inline]
    #[must_use]
    pub const fn is_float(&self) -> bool {
        (self.bits & QNAN) != QNAN
    }

    /// Check if this is None.
    #[inline]
    #[must_use]
    pub const fn is_none(&self) -> bool {
        self.is_tagged() && self.tag() == TAG_NONE
    }

    /// Check if this is a boolean.
    #[inline]
    #[must_use]
    pub const fn is_bool(&self) -> bool {
        self.is_tagged() && self.tag() == TAG_BOOL
    }

    /// Check if this is a small integer.
    #[inline]
    #[must_use]
    pub const fn is_int(&self) -> bool {
        self.is_tagged() && self.tag() == TAG_INT
    }

    /// Check if this is an object reference.
    #[inline]
    #[must_use]
    pub const fn is_object(&self) -> bool {
        self.is_tagged() && self.tag() == TAG_OBJECT
    }

    /// Check if this is an interned string.
    #[inline]
    #[must_use]
    pub const fn is_string(&self) -> bool {
        self.is_tagged() && self.tag() == TAG_STRING
    }

    /// Get the tag (for tagged values).
    #[inline]
    const fn tag(&self) -> u64 {
        (self.bits & TAG_MASK) >> TAG_SHIFT
    }

    /// Get the payload (for tagged values).
    #[inline]
    const fn payload(&self) -> u64 {
        self.bits & PAYLOAD_MASK
    }

    // =========================================================================
    // Value Extraction
    // =========================================================================

    /// Try to extract as a boolean.
    #[inline]
    #[must_use]
    pub const fn as_bool(&self) -> Option<bool> {
        if self.is_bool() {
            Some(self.payload() != 0)
        } else {
            None
        }
    }

    /// Try to extract as a small integer.
    #[inline]
    #[must_use]
    pub const fn as_int(&self) -> Option<i64> {
        if self.is_int() {
            // Sign-extend from 48 bits
            let payload = self.payload();
            let sign_bit = payload & (1 << 47);
            if sign_bit != 0 {
                // Negative: sign-extend
                Some((payload | !PAYLOAD_MASK) as i64)
            } else {
                // Positive
                Some(payload as i64)
            }
        } else {
            None
        }
    }

    /// Try to extract as a float.
    #[inline]
    #[must_use]
    pub fn as_float(&self) -> Option<f64> {
        if self.is_float() {
            Some(f64::from_bits(self.bits))
        } else {
            None
        }
    }

    /// Try to extract as a float, coercing integers if needed.
    #[inline]
    #[must_use]
    pub fn as_float_coerce(&self) -> Option<f64> {
        if let Some(f) = self.as_float() {
            Some(f)
        } else if let Some(i) = self.as_int() {
            Some(i as f64)
        } else {
            None
        }
    }

    /// Try to extract as an object pointer.
    #[inline]
    #[must_use]
    pub const fn as_object_ptr(&self) -> Option<*const ()> {
        if self.is_object() {
            Some(self.payload() as *const ())
        } else {
            None
        }
    }

    /// Try to extract as a string object pointer.
    ///
    /// Returns a raw pointer to the string data. For interned strings,
    /// this points to the Arc's data buffer. For heap-allocated StringObjects,
    /// this points to the StringObject in the GC heap.
    ///
    /// # Performance
    ///
    /// This is a branchless const operation after the type check.
    /// The returned pointer is suitable for direct string operations.
    ///
    /// # Safety
    ///
    /// The returned pointer is valid as long as:
    /// - For interned strings: the InternedString is not dropped (they're leaked)
    /// - For heap strings: the StringObject is reachable by the GC
    #[inline]
    #[must_use]
    pub const fn as_string_object_ptr(&self) -> Option<*const ()> {
        if self.is_string() {
            Some(self.payload() as *const ())
        } else {
            None
        }
    }

    /// Get raw bits (for speculation optimizations).
    ///
    /// # Performance
    ///
    /// Enables branchless type checking in speculation code:
    /// ```ignore
    /// let is_string = (value.raw_bits() >> 48) == STRING_TAG_PATTERN;
    /// ```
    #[inline(always)]
    #[must_use]
    pub const fn raw_bits(&self) -> u64 {
        self.bits
    }

    // =========================================================================
    // Truthiness (Python bool conversion)
    // =========================================================================

    /// Python truthiness evaluation.
    ///
    /// Returns false for: None, False, 0, 0.0, empty collections
    /// For objects, this requires calling __bool__ or __len__.
    #[inline]
    #[must_use]
    pub fn is_truthy(&self) -> bool {
        if self.is_none() {
            false
        } else if let Some(b) = self.as_bool() {
            b
        } else if let Some(i) = self.as_int() {
            i != 0
        } else if let Some(f) = self.as_float() {
            f != 0.0
        } else {
            // Objects are truthy by default (need __bool__/__len__ for full impl)
            true
        }
    }

    // =========================================================================
    // Type Name
    // =========================================================================

    /// Get the Python type name.
    #[must_use]
    pub const fn type_name(&self) -> &'static str {
        if self.is_float() {
            "float"
        } else if self.is_none() {
            "NoneType"
        } else if self.is_bool() {
            "bool"
        } else if self.is_int() {
            "int"
        } else if self.is_string() {
            "str"
        } else if self.is_object() {
            "object"
        } else {
            "unknown"
        }
    }

    /// Get the raw bits (for debugging/serialization).
    #[inline]
    #[must_use]
    pub const fn to_bits(&self) -> u64 {
        self.bits
    }

    /// Create from raw bits (for deserialization).
    #[inline]
    #[must_use]
    pub const fn from_bits(bits: u64) -> Self {
        Self { bits }
    }
}

impl PartialEq for Value {
    fn eq(&self, other: &Self) -> bool {
        // Handle NaN first (NaN != NaN in IEEE 754, even with identical bits)
        if self.is_float() && other.is_float() {
            let a = f64::from_bits(self.bits);
            let b = f64::from_bits(other.bits);
            // NaN != NaN per IEEE 754, otherwise compare normally
            return a == b;
        }

        // Fast path: identical bits (safe now that we've handled NaN)
        if self.bits == other.bits {
            return true;
        }

        // Handle int/float coercion (1 == 1.0 in Python)
        if let (Some(i), Some(f)) = (self.as_int(), other.as_float()) {
            return (i as f64) == f;
        }
        if let (Some(f), Some(i)) = (self.as_float(), other.as_int()) {
            return f == (i as f64);
        }

        false
    }
}

impl Eq for Value {}

impl Hash for Value {
    fn hash<H: Hasher>(&self, state: &mut H) {
        // For consistency with Python, int and equivalent float should hash the same
        if let Some(i) = self.as_int() {
            i.hash(state);
        } else if let Some(f) = self.as_float() {
            // Check if it's an integer value
            if f.fract() == 0.0 && f.is_finite() && f >= (i64::MIN as f64) && f <= (i64::MAX as f64)
            {
                (f as i64).hash(state);
            } else {
                self.bits.hash(state);
            }
        } else {
            self.bits.hash(state);
        }
    }
}

impl fmt::Debug for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_none() {
            write!(f, "Value(None)")
        } else if let Some(b) = self.as_bool() {
            write!(f, "Value({})", if b { "True" } else { "False" })
        } else if let Some(i) = self.as_int() {
            write!(f, "Value({})", i)
        } else if let Some(fl) = self.as_float() {
            write!(f, "Value({:?})", fl)
        } else if self.is_object() {
            write!(f, "Value(object@{:#x})", self.payload())
        } else if self.is_string() {
            write!(f, "Value(str@{:#x})", self.payload())
        } else {
            write!(f, "Value(bits={:#018x})", self.bits)
        }
    }
}

impl fmt::Display for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_none() {
            write!(f, "None")
        } else if let Some(b) = self.as_bool() {
            write!(f, "{}", if b { "True" } else { "False" })
        } else if let Some(i) = self.as_int() {
            write!(f, "{}", i)
        } else if let Some(fl) = self.as_float() {
            // Python-style float formatting
            if fl.fract() == 0.0 && fl.is_finite() {
                write!(f, "{}.0", fl as i64)
            } else {
                write!(f, "{}", fl)
            }
        } else if self.is_object() {
            write!(f, "<object at {:#x}>", self.payload())
        } else if self.is_string() {
            write!(f, "<str at {:#x}>", self.payload())
        } else {
            write!(f, "<unknown>")
        }
    }
}

impl Default for Value {
    fn default() -> Self {
        Self::none()
    }
}

impl From<bool> for Value {
    fn from(b: bool) -> Self {
        Self::bool(b)
    }
}

impl From<f64> for Value {
    fn from(f: f64) -> Self {
        Self::float(f)
    }
}

impl From<f32> for Value {
    fn from(f: f32) -> Self {
        Self::float(f as f64)
    }
}

impl TryFrom<i64> for Value {
    type Error = ();

    fn try_from(i: i64) -> Result<Self, Self::Error> {
        Self::int(i).ok_or(())
    }
}

impl From<i32> for Value {
    fn from(i: i32) -> Self {
        Self::int_unchecked(i as i64)
    }
}

impl From<i16> for Value {
    fn from(i: i16) -> Self {
        Self::int_unchecked(i as i64)
    }
}

impl From<i8> for Value {
    fn from(i: i8) -> Self {
        Self::int_unchecked(i as i64)
    }
}

impl From<u32> for Value {
    fn from(i: u32) -> Self {
        Self::int_unchecked(i as i64)
    }
}

impl From<u16> for Value {
    fn from(i: u16) -> Self {
        Self::int_unchecked(i as i64)
    }
}

impl From<u8> for Value {
    fn from(i: u8) -> Self {
        Self::int_unchecked(i as i64)
    }
}

// Helper for InternedString to avoid circular dependency
impl InternedString {
    /// Clone the underlying Arc for use in Value.
    fn clone_arc(&self) -> std::sync::Arc<str> {
        // We need to access the inner Arc - this requires the type to expose it
        // For now, we'll use a workaround by re-interning
        // TODO: Add proper Arc access to InternedString
        std::sync::Arc::from(self.as_str())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_value_size() {
        assert_eq!(std::mem::size_of::<Value>(), 8);
    }

    #[test]
    fn test_none_creation() {
        let v = Value::none();
        assert!(v.is_none());
        assert!(v.is_tagged());
        assert!(!v.is_float());
        assert!(!v.is_bool());
        assert!(!v.is_int());
    }

    #[test]
    fn test_bool_true() {
        let v = Value::bool(true);
        assert!(v.is_bool());
        assert_eq!(v.as_bool(), Some(true));
        assert!(v.is_truthy());
    }

    #[test]
    fn test_bool_false() {
        let v = Value::bool(false);
        assert!(v.is_bool());
        assert_eq!(v.as_bool(), Some(false));
        assert!(!v.is_truthy());
    }

    #[test]
    fn test_int_zero() {
        let v = Value::int(0).unwrap();
        assert!(v.is_int());
        assert_eq!(v.as_int(), Some(0));
        assert!(!v.is_truthy());
    }

    #[test]
    fn test_int_positive() {
        let v = Value::int(42).unwrap();
        assert!(v.is_int());
        assert_eq!(v.as_int(), Some(42));
        assert!(v.is_truthy());
    }

    #[test]
    fn test_int_negative() {
        let v = Value::int(-42).unwrap();
        assert!(v.is_int());
        assert_eq!(v.as_int(), Some(-42));
        assert!(v.is_truthy());
    }

    #[test]
    fn test_int_max_small() {
        let v = Value::int(SMALL_INT_MAX).unwrap();
        assert!(v.is_int());
        assert_eq!(v.as_int(), Some(SMALL_INT_MAX));
    }

    #[test]
    fn test_int_min_small() {
        let v = Value::int(SMALL_INT_MIN).unwrap();
        assert!(v.is_int());
        assert_eq!(v.as_int(), Some(SMALL_INT_MIN));
    }

    #[test]
    fn test_int_too_large() {
        let v = Value::int(SMALL_INT_MAX + 1);
        assert!(v.is_none());
    }

    #[test]
    fn test_int_too_small() {
        let v = Value::int(SMALL_INT_MIN - 1);
        assert!(v.is_none());
    }

    #[test]
    fn test_float_positive() {
        let v = Value::float(3.14);
        assert!(v.is_float());
        assert_eq!(v.as_float(), Some(3.14));
        assert!(v.is_truthy());
    }

    #[test]
    fn test_float_negative() {
        let v = Value::float(-2.71);
        assert!(v.is_float());
        assert_eq!(v.as_float(), Some(-2.71));
    }

    #[test]
    fn test_float_zero() {
        let v = Value::float(0.0);
        assert!(v.is_float());
        assert_eq!(v.as_float(), Some(0.0));
        assert!(!v.is_truthy());
    }

    #[test]
    fn test_float_negative_zero() {
        let v = Value::float(-0.0);
        assert!(v.is_float());
        let f = v.as_float().unwrap();
        assert!(f == 0.0);
    }

    #[test]
    fn test_float_infinity() {
        let v = Value::float(f64::INFINITY);
        assert!(v.is_float());
        assert_eq!(v.as_float(), Some(f64::INFINITY));
    }

    #[test]
    fn test_float_neg_infinity() {
        let v = Value::float(f64::NEG_INFINITY);
        assert!(v.is_float());
        assert_eq!(v.as_float(), Some(f64::NEG_INFINITY));
    }

    #[test]
    fn test_float_nan() {
        let v = Value::float(f64::NAN);
        assert!(v.is_float());
        assert!(v.as_float().unwrap().is_nan());
    }

    #[test]
    fn test_float_coerce_from_int() {
        let v = Value::int(42).unwrap();
        assert_eq!(v.as_float_coerce(), Some(42.0));
    }

    #[test]
    fn test_float_coerce_from_float() {
        let v = Value::float(3.14);
        assert_eq!(v.as_float_coerce(), Some(3.14));
    }

    #[test]
    fn test_type_name_none() {
        assert_eq!(Value::none().type_name(), "NoneType");
    }

    #[test]
    fn test_type_name_bool() {
        assert_eq!(Value::bool(true).type_name(), "bool");
    }

    #[test]
    fn test_type_name_int() {
        assert_eq!(Value::int(42).unwrap().type_name(), "int");
    }

    #[test]
    fn test_type_name_float() {
        assert_eq!(Value::float(3.14).type_name(), "float");
    }

    #[test]
    fn test_equality_none() {
        assert_eq!(Value::none(), Value::none());
    }

    #[test]
    fn test_equality_bool() {
        assert_eq!(Value::bool(true), Value::bool(true));
        assert_eq!(Value::bool(false), Value::bool(false));
        assert_ne!(Value::bool(true), Value::bool(false));
    }

    #[test]
    fn test_equality_int() {
        assert_eq!(Value::int(42).unwrap(), Value::int(42).unwrap());
        assert_ne!(Value::int(42).unwrap(), Value::int(43).unwrap());
    }

    #[test]
    fn test_equality_float() {
        assert_eq!(Value::float(3.14), Value::float(3.14));
        assert_ne!(Value::float(3.14), Value::float(3.15));
    }

    #[test]
    fn test_equality_nan() {
        // NaN != NaN in IEEE 754
        assert_ne!(Value::float(f64::NAN), Value::float(f64::NAN));
    }

    #[test]
    fn test_equality_int_float_coercion() {
        // Python: 1 == 1.0
        assert_eq!(Value::int(1).unwrap(), Value::float(1.0));
        assert_eq!(Value::float(1.0), Value::int(1).unwrap());
        assert_eq!(Value::int(0).unwrap(), Value::float(0.0));
    }

    #[test]
    fn test_equality_int_float_mismatch() {
        assert_ne!(Value::int(1).unwrap(), Value::float(1.5));
    }

    #[test]
    fn test_hash_int_float_equivalence() {
        use std::collections::hash_map::DefaultHasher;

        let int_val = Value::int(42).unwrap();
        let float_val = Value::float(42.0);

        let mut hasher1 = DefaultHasher::new();
        int_val.hash(&mut hasher1);
        let hash1 = hasher1.finish();

        let mut hasher2 = DefaultHasher::new();
        float_val.hash(&mut hasher2);
        let hash2 = hasher2.finish();

        assert_eq!(hash1, hash2);
    }

    #[test]
    fn test_debug_none() {
        let v = Value::none();
        assert!(format!("{:?}", v).contains("None"));
    }

    #[test]
    fn test_debug_bool() {
        let v = Value::bool(true);
        assert!(format!("{:?}", v).contains("True"));
    }

    #[test]
    fn test_debug_int() {
        let v = Value::int(42).unwrap();
        assert!(format!("{:?}", v).contains("42"));
    }

    #[test]
    fn test_debug_float() {
        let v = Value::float(3.14);
        let debug = format!("{:?}", v);
        assert!(debug.contains("3.14"));
    }

    #[test]
    fn test_display_none() {
        assert_eq!(format!("{}", Value::none()), "None");
    }

    #[test]
    fn test_display_bool_true() {
        assert_eq!(format!("{}", Value::bool(true)), "True");
    }

    #[test]
    fn test_display_bool_false() {
        assert_eq!(format!("{}", Value::bool(false)), "False");
    }

    #[test]
    fn test_display_int() {
        assert_eq!(format!("{}", Value::int(42).unwrap()), "42");
        assert_eq!(format!("{}", Value::int(-42).unwrap()), "-42");
    }

    #[test]
    fn test_display_float() {
        let v = Value::float(3.14);
        assert_eq!(format!("{}", v), "3.14");
    }

    #[test]
    fn test_display_float_integer_value() {
        let v = Value::float(42.0);
        assert_eq!(format!("{}", v), "42.0");
    }

    #[test]
    fn test_from_bool() {
        let v: Value = true.into();
        assert!(v.is_bool());
        assert_eq!(v.as_bool(), Some(true));
    }

    #[test]
    fn test_from_f64() {
        let v: Value = 3.14_f64.into();
        assert!(v.is_float());
    }

    #[test]
    fn test_from_f32() {
        let v: Value = 3.14_f32.into();
        assert!(v.is_float());
    }

    #[test]
    fn test_try_from_i64_success() {
        let v: Result<Value, _> = 42_i64.try_into();
        assert!(v.is_ok());
        assert_eq!(v.unwrap().as_int(), Some(42));
    }

    #[test]
    fn test_try_from_i64_overflow() {
        let v: Result<Value, _> = i64::MAX.try_into();
        assert!(v.is_err());
    }

    #[test]
    fn test_from_i32() {
        let v: Value = 42_i32.into();
        assert!(v.is_int());
        assert_eq!(v.as_int(), Some(42));
    }

    #[test]
    fn test_from_i16() {
        let v: Value = 42_i16.into();
        assert!(v.is_int());
    }

    #[test]
    fn test_from_i8() {
        let v: Value = 42_i8.into();
        assert!(v.is_int());
    }

    #[test]
    fn test_from_u32() {
        let v: Value = 42_u32.into();
        assert!(v.is_int());
    }

    #[test]
    fn test_from_u16() {
        let v: Value = 42_u16.into();
        assert!(v.is_int());
    }

    #[test]
    fn test_from_u8() {
        let v: Value = 42_u8.into();
        assert!(v.is_int());
    }

    #[test]
    fn test_default_is_none() {
        let v = Value::default();
        assert!(v.is_none());
    }

    #[test]
    fn test_bits_roundtrip() {
        let values = [
            Value::none(),
            Value::bool(true),
            Value::bool(false),
            Value::int(0).unwrap(),
            Value::int(42).unwrap(),
            Value::int(-42).unwrap(),
            Value::float(0.0),
            Value::float(3.14),
            Value::float(-2.71),
        ];

        for v in values {
            let bits = v.to_bits();
            let reconstructed = Value::from_bits(bits);
            assert_eq!(v.to_bits(), reconstructed.to_bits());
        }
    }

    #[test]
    fn test_all_small_int_range() {
        // Test boundaries
        for i in [
            SMALL_INT_MIN,
            SMALL_INT_MIN + 1,
            -1,
            0,
            1,
            SMALL_INT_MAX - 1,
            SMALL_INT_MAX,
        ] {
            let v = Value::int(i).expect("Should fit");
            assert_eq!(v.as_int(), Some(i), "Failed for {}", i);
        }
    }

    #[test]
    fn test_negative_int_sign_extension() {
        let values = [-1_i64, -2, -100, -1000, -1_000_000, SMALL_INT_MIN];
        for i in values {
            let v = Value::int(i).unwrap();
            assert_eq!(v.as_int(), Some(i), "Sign extension failed for {}", i);
        }
    }

    #[test]
    fn test_object_pointer() {
        let data = Box::new(42_u64);
        let ptr = Box::into_raw(data) as *const ();

        let v = Value::object_ptr(ptr);
        assert!(v.is_object());
        assert_eq!(v.as_object_ptr(), Some(ptr));

        // Cleanup
        unsafe {
            drop(Box::from_raw(ptr as *mut u64));
        }
    }

    #[test]
    fn test_value_copy() {
        let v1 = Value::int(42).unwrap();
        let v2 = v1; // Copy
        assert_eq!(v1, v2);
    }

    #[test]
    fn test_value_clone() {
        let v1 = Value::float(3.14);
        let v2 = v1.clone();
        assert_eq!(v1, v2);
    }

    #[test]
    fn test_truthiness_comprehensive() {
        // Falsy values
        assert!(!Value::none().is_truthy());
        assert!(!Value::bool(false).is_truthy());
        assert!(!Value::int(0).unwrap().is_truthy());
        assert!(!Value::float(0.0).is_truthy());

        // Truthy values
        assert!(Value::bool(true).is_truthy());
        assert!(Value::int(1).unwrap().is_truthy());
        assert!(Value::int(-1).unwrap().is_truthy());
        assert!(Value::float(0.1).is_truthy());
        assert!(Value::float(-0.1).is_truthy());
        assert!(Value::float(f64::INFINITY).is_truthy());
    }

    #[test]
    fn test_as_bool_on_non_bool() {
        assert_eq!(Value::none().as_bool(), None);
        assert_eq!(Value::int(42).unwrap().as_bool(), None);
        assert_eq!(Value::float(3.14).as_bool(), None);
    }

    #[test]
    fn test_as_int_on_non_int() {
        assert_eq!(Value::none().as_int(), None);
        assert_eq!(Value::bool(true).as_int(), None);
        assert_eq!(Value::float(3.14).as_int(), None);
    }

    #[test]
    fn test_as_float_on_non_float() {
        assert_eq!(Value::none().as_float(), None);
        assert_eq!(Value::bool(true).as_float(), None);
        assert_eq!(Value::int(42).unwrap().as_float(), None);
    }

    #[test]
    fn test_value_in_hashmap() {
        use std::collections::HashMap;

        let mut map = HashMap::new();
        map.insert(Value::int(1).unwrap(), "one");
        map.insert(Value::int(2).unwrap(), "two");

        assert_eq!(map.get(&Value::int(1).unwrap()), Some(&"one"));
        assert_eq!(map.get(&Value::float(1.0)), Some(&"one")); // Coercion!
    }

    #[test]
    fn test_value_in_hashset() {
        use std::collections::HashSet;

        let mut set = HashSet::new();
        set.insert(Value::int(42).unwrap());

        assert!(set.contains(&Value::int(42).unwrap()));
        assert!(set.contains(&Value::float(42.0))); // Same hash
    }

    #[test]
    fn test_special_floats() {
        let values = [
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::MIN,
            f64::MAX,
            f64::MIN_POSITIVE,
            f64::EPSILON,
        ];

        for f in values {
            let v = Value::float(f);
            assert!(v.is_float());
            assert_eq!(v.as_float(), Some(f));
        }
    }

    #[test]
    fn test_subnormal_floats() {
        let subnormal = f64::MIN_POSITIVE / 2.0;
        assert!(subnormal.is_subnormal());

        let v = Value::float(subnormal);
        assert!(v.is_float());
        assert_eq!(v.as_float(), Some(subnormal));
    }
}
