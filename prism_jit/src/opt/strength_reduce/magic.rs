//! Magic Number Division
//!
//! This module implements the "magic number" algorithm for replacing integer
//! division by a constant with multiplication followed by a shift. This is
//! based on the algorithms from "Hacker's Delight" by Henry S. Warren Jr.
//!
//! # Theory
//!
//! For a divisor `d`, we find a magic multiplier `m` and shift amount `s` such that:
//! ```text
//! x / d ≈ (x * m) >> s
//! ```
//!
//! This works because `m ≈ 2^s / d`, so `x * m >> s ≈ x * (2^s / d) / 2^s = x / d`.
//!
//! # Unsigned vs Signed
//!
//! Unsigned division is simpler. Signed division requires handling negative
//! dividends and divisors, potentially needing a correction step.
//!
//! # Performance
//!
//! Division: ~20-80 cycles on modern CPUs
//! Multiply + shift: ~3-5 cycles
//!
//! This optimization is always profitable when the divisor is a compile-time constant.

// =============================================================================
// Magic Number Structures
// =============================================================================

/// Magic numbers for unsigned 64-bit division.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct UnsignedMagic {
    /// The magic multiplier.
    pub multiplier: u64,
    /// Number of bits to shift right after multiplication.
    pub shift: u8,
    /// Whether to use add-shift sequence (for divisors where multiplier overflows).
    pub add: bool,
}

/// Magic numbers for signed 64-bit division.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SignedMagic {
    /// The magic multiplier (signed).
    pub multiplier: i64,
    /// Number of bits to shift right after multiplication.
    pub shift: u8,
    /// Whether to add the dividend to the high product.
    pub add: bool,
    /// Whether the original divisor was negative.
    pub negative_divisor: bool,
}

/// Magic numbers for unsigned 32-bit division.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct UnsignedMagic32 {
    /// The magic multiplier.
    pub multiplier: u32,
    /// Number of bits to shift right after multiplication.
    pub shift: u8,
    /// Whether to use add-shift sequence.
    pub add: bool,
}

/// Magic numbers for signed 32-bit division.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SignedMagic32 {
    /// The magic multiplier (signed).
    pub multiplier: i32,
    /// Number of bits to shift right after multiplication.
    pub shift: u8,
    /// Whether to add the dividend to the high product.
    pub add: bool,
    /// Whether the original divisor was negative.
    pub negative_divisor: bool,
}

// =============================================================================
// Unsigned 64-bit Magic Numbers
// =============================================================================

impl UnsignedMagic {
    /// Compute magic numbers for unsigned 64-bit division by `d`.
    ///
    /// Returns `None` if `d` is 0 or 1 (special cases).
    ///
    /// # Algorithm
    ///
    /// Based on "Hacker's Delight" Figure 10-1 (modified for 64-bit).
    ///
    /// The algorithm finds the smallest shift `p` such that:
    /// ```text
    /// 2^p mod d <= 2^p / (2^64 / d)
    /// ```
    ///
    /// Then computes `m = ceil(2^(64+p) / d) - 2^64`.
    pub fn compute(d: u64) -> Option<Self> {
        // Special cases
        if d == 0 {
            return None; // Division by zero
        }
        if d == 1 {
            return None; // No reduction needed (x / 1 = x)
        }

        // Check if d is a power of 2
        if d.is_power_of_two() {
            return Some(Self {
                multiplier: 0,
                shift: d.trailing_zeros() as u8,
                add: false,
            });
        }

        let two_64 = 1u128 << 64;

        // Find the minimal shift
        // Start with ceil(log2(d))
        let l = 64 - d.leading_zeros();

        // Try different shift amounts
        for p in l..=64 {
            // Calculate 2^(64+p) / d using 128-bit arithmetic
            let two_p = 1u128 << (64 + p);
            let m = (two_p + d as u128 - 1) / d as u128; // ceil(2^(64+p) / d)

            // Check if m fits in 64 bits (m < 2^64)
            if m < two_64 {
                return Some(Self {
                    multiplier: m as u64,
                    shift: p as u8,
                    add: false,
                });
            }

            // If m >= 2^64 but < 2^65, we need the add-shift sequence
            // Store m' = m - 2^64 (the low 64 bits)
            // The application formula accounts for the implicit high bit
            if m < two_64 * 2 {
                let adjusted = (m - two_64) as u64;
                return Some(Self {
                    multiplier: adjusted,
                    shift: (p - 1) as u8,
                    add: true,
                });
            }
        }

        // Fallback (should not be reached for valid divisors)
        None
    }

    /// Check if this represents a power-of-2 division (shift only).
    #[inline]
    pub fn is_power_of_two(&self) -> bool {
        self.multiplier == 0
    }

    /// Get the instruction sequence cost estimate.
    ///
    /// Lower is better. Used for cost-benefit analysis.
    #[inline]
    pub fn cost(&self) -> u32 {
        if self.is_power_of_two() {
            1 // Just a shift
        } else if self.add {
            4 // mulhi, sub, shr1, add, shr
        } else {
            2 // mulhi, shr
        }
    }
}

// =============================================================================
// Signed 64-bit Magic Numbers
// =============================================================================

impl SignedMagic {
    /// Compute magic numbers for signed 64-bit division by `d`.
    ///
    /// Returns `None` if `d` is 0, 1, or -1 (special cases).
    ///
    /// # Algorithm
    ///
    /// Based on libdivide's implementation which follows "Hacker's Delight".
    /// For signed division, we compute magic numbers for the absolute value
    /// of the divisor and track whether we need the add-shift sequence.
    pub fn compute(d: i64) -> Option<Self> {
        // Special cases
        if d == 0 {
            return None; // Division by zero
        }
        if d == 1 {
            return None; // x / 1 = x
        }
        if d == -1 {
            return None; // x / -1 = -x (special case, can overflow)
        }

        let ud = d as u64;
        let abs_d = if d < 0 { ud.wrapping_neg() } else { ud };

        // Check if |d| is a power of 2
        if (abs_d & (abs_d - 1)) == 0 {
            let shift = abs_d.trailing_zeros() as u8;
            // For signed division by power of 2, need sign correction
            return Some(Self {
                multiplier: 0,
                shift,
                add: false,
                negative_divisor: d < 0,
            });
        }

        let floor_log_2_d = 63 - abs_d.leading_zeros();

        // Compute 2^(floor_log_2_d + 63) / abs_d
        // Using 128-bit arithmetic: high word is (1 << (floor_log_2_d - 1)), low word is 0
        let mut proposed_m: u64;
        let remainder: u64;
        {
            let numerator_hi = 1u64 << (floor_log_2_d - 1);
            let numerator_lo = 0u64;
            // Compute (numerator_hi * 2^64 + numerator_lo) / abs_d
            let numerator = ((numerator_hi as u128) << 64) | (numerator_lo as u128);
            proposed_m = (numerator / abs_d as u128) as u64;
            remainder = (numerator % abs_d as u128) as u64;
        }

        let e = abs_d - remainder;

        // Check if we can use the smaller power
        let add_marker: bool;
        let shift: u8;

        if e < (1u64 << floor_log_2_d) {
            // This power works
            shift = (floor_log_2_d - 1) as u8;
            add_marker = false;
        } else {
            // We need to go one higher - double proposed_m
            proposed_m = proposed_m.wrapping_add(proposed_m);
            let twice_rem = remainder.wrapping_add(remainder);
            if twice_rem >= abs_d || twice_rem < remainder {
                proposed_m = proposed_m.wrapping_add(1);
            }
            shift = floor_log_2_d as u8;
            add_marker = true;
        }

        proposed_m = proposed_m.wrapping_add(1);
        let magic = proposed_m as i64;

        // Don't negate magic - we compute for abs(d) and use negative_divisor flag in application
        Some(Self {
            multiplier: magic,
            shift,
            add: add_marker,
            negative_divisor: d < 0,
        })
    }

    /// Check if this represents a power-of-2 division.
    #[inline]
    pub fn is_power_of_two(&self) -> bool {
        self.multiplier == 0
    }

    /// Check if divisor was negative.
    #[inline]
    pub fn is_negative_divisor(&self) -> bool {
        self.negative_divisor
    }

    /// Get the instruction sequence cost estimate.
    #[inline]
    pub fn cost(&self) -> u32 {
        if self.is_power_of_two() {
            3 // sra (sign extend), add, sra
        } else {
            4 // mulhi, conditional add, sra, sign correction
        }
    }
}

// =============================================================================
// Unsigned 32-bit Magic Numbers
// =============================================================================

impl UnsignedMagic32 {
    /// Compute magic numbers for unsigned 32-bit division by `d`.
    pub fn compute(d: u32) -> Option<Self> {
        if d == 0 {
            return None;
        }
        if d == 1 {
            return None;
        }

        if d.is_power_of_two() {
            return Some(Self {
                multiplier: 0,
                shift: d.trailing_zeros() as u8,
                add: false,
            });
        }

        let two_32 = 1u64 << 32;

        let l = 32 - d.leading_zeros();

        for p in l..=32 {
            let two_p = 1u64 << (32 + p);
            let m = (two_p + d as u64 - 1) / d as u64;

            if m < two_32 {
                return Some(Self {
                    multiplier: m as u32,
                    shift: p as u8,
                    add: false,
                });
            }

            // If m >= 2^32 but < 2^33, use add-shift sequence
            if m < two_32 * 2 {
                let adjusted = (m - two_32) as u32;
                return Some(Self {
                    multiplier: adjusted,
                    shift: (p - 1) as u8,
                    add: true,
                });
            }
        }

        None
    }

    /// Check if this represents a power-of-2 division.
    #[inline]
    pub fn is_power_of_two(&self) -> bool {
        self.multiplier == 0
    }

    /// Get the instruction sequence cost estimate.
    #[inline]
    pub fn cost(&self) -> u32 {
        if self.is_power_of_two() {
            1
        } else if self.add {
            4
        } else {
            2
        }
    }
}

// =============================================================================
// Signed 32-bit Magic Numbers
// =============================================================================

impl SignedMagic32 {
    /// Compute magic numbers for signed 32-bit division by `d`.
    ///
    /// Based on libdivide's implementation which follows "Hacker's Delight".
    pub fn compute(d: i32) -> Option<Self> {
        if d == 0 {
            return None;
        }
        if d == 1 {
            return None;
        }
        if d == -1 {
            return None;
        }

        let ud = d as u32;
        let abs_d = if d < 0 { ud.wrapping_neg() } else { ud };

        if (abs_d & (abs_d - 1)) == 0 {
            return Some(Self {
                multiplier: 0,
                shift: abs_d.trailing_zeros() as u8,
                add: false,
                negative_divisor: d < 0,
            });
        }

        let floor_log_2_d = 31 - abs_d.leading_zeros();

        // Compute 2^(floor_log_2_d + 31) / abs_d using 64-bit arithmetic
        let numerator = (1u64 << (floor_log_2_d - 1)) << 32;
        let mut proposed_m = (numerator / abs_d as u64) as u32;
        let remainder = (numerator % abs_d as u64) as u32;

        let e = abs_d - remainder;

        let shift: u8;
        let add_marker: bool;
        if e < (1u32 << floor_log_2_d) {
            shift = (floor_log_2_d - 1) as u8;
            add_marker = false;
        } else {
            // Double proposed_m
            proposed_m = proposed_m.wrapping_add(proposed_m);
            let twice_rem = remainder.wrapping_add(remainder);
            if twice_rem >= abs_d || twice_rem < remainder {
                proposed_m = proposed_m.wrapping_add(1);
            }
            shift = floor_log_2_d as u8;
            add_marker = true;
        }

        proposed_m = proposed_m.wrapping_add(1);
        let magic = proposed_m as i32;

        // Don't negate magic - use negative_divisor flag in application
        Some(Self {
            multiplier: magic,
            shift,
            add: add_marker,
            negative_divisor: d < 0,
        })
    }

    /// Check if this represents a power-of-2 division.
    #[inline]
    pub fn is_power_of_two(&self) -> bool {
        self.multiplier == 0
    }

    /// Check if divisor was negative.
    #[inline]
    pub fn is_negative_divisor(&self) -> bool {
        self.negative_divisor
    }

    /// Get the instruction sequence cost estimate.
    #[inline]
    pub fn cost(&self) -> u32 {
        if self.is_power_of_two() { 3 } else { 4 }
    }
}

// =============================================================================
// Helper Functions
// =============================================================================

/// Check if a value is a power of 2.
#[inline]
pub fn is_power_of_two_u64(n: u64) -> bool {
    n != 0 && (n & (n - 1)) == 0
}

/// Check if a signed value is a power of 2.
#[inline]
pub fn is_power_of_two_i64(n: i64) -> bool {
    n > 0 && is_power_of_two_u64(n as u64)
}

/// Get the log2 of a power of 2.
#[inline]
pub fn log2_u64(n: u64) -> u8 {
    debug_assert!(is_power_of_two_u64(n));
    n.trailing_zeros() as u8
}

/// Mask for power-of-2 modulo: n % (2^k) = n & (2^k - 1)
#[inline]
pub fn power_of_two_mask(shift: u8) -> u64 {
    (1u64 << shift) - 1
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // Unsigned 64-bit Tests
    // =========================================================================

    #[test]
    fn test_unsigned_magic_special_cases() {
        // Division by 0 not allowed
        assert!(UnsignedMagic::compute(0).is_none());

        // Division by 1 - identity
        assert!(UnsignedMagic::compute(1).is_none());
    }

    #[test]
    fn test_unsigned_magic_power_of_two() {
        for shift in 1..63 {
            let d = 1u64 << shift;
            let magic = UnsignedMagic::compute(d).unwrap();
            assert!(magic.is_power_of_two());
            assert_eq!(magic.shift, shift);
            assert!(!magic.add);
        }
    }

    #[test]
    fn test_unsigned_magic_small_divisors() {
        // Test divisors 2-20
        for d in 2u64..=20 {
            let magic = UnsignedMagic::compute(d);
            assert!(magic.is_some(), "Failed for divisor {}", d);

            let m = magic.unwrap();
            // Verify correctness for a sample of values
            for x in [0u64, 1, 10, 100, 1000, u64::MAX / 2, u64::MAX] {
                let expected = x / d;
                let actual = apply_unsigned_magic(x, &m);
                assert_eq!(
                    actual, expected,
                    "Failed for x={}, d={}: got {}, expected {}",
                    x, d, actual, expected
                );
            }
        }
    }

    #[test]
    fn test_unsigned_magic_large_divisors() {
        let test_divisors = [1000u64, 10000, 65537, 1_000_000, u64::MAX / 2];

        for d in test_divisors {
            let magic = UnsignedMagic::compute(d);
            assert!(magic.is_some(), "Failed for divisor {}", d);

            let m = magic.unwrap();
            for x in [0u64, 1, d - 1, d, d + 1, d * 2, u64::MAX] {
                let expected = x / d;
                let actual = apply_unsigned_magic(x, &m);
                assert_eq!(
                    actual, expected,
                    "Failed for x={}, d={}: got {}, expected {}",
                    x, d, actual, expected
                );
            }
        }
    }

    #[test]
    fn test_unsigned_magic_comprehensive() {
        // Test many divisors
        for d in 2u64..=1000 {
            let magic = UnsignedMagic::compute(d).unwrap();

            // Test boundary values
            let test_values = [0u64, 1, d - 1, d, d + 1, d * 2 - 1, d * 2, 1000 * d];

            for x in test_values {
                let expected = x / d;
                let actual = apply_unsigned_magic(x, &magic);
                assert_eq!(actual, expected, "Failed for x={}, d={}", x, d);
            }
        }
    }

    #[test]
    fn test_unsigned_magic_cost() {
        // Power of 2 should be cheapest
        let pow2 = UnsignedMagic::compute(4).unwrap();
        assert_eq!(pow2.cost(), 1);

        // Divisor 3 requires add-shift sequence for 64-bit
        let with_add = UnsignedMagic::compute(3).unwrap();
        assert!(with_add.add);
        assert_eq!(with_add.cost(), 4);

        // Some divisors work without add-shift (e.g., divisors where
        // a power of 2 is close to a multiple of d)
        // Divisor 5 needs add-shift at 64-bit too
        let five = UnsignedMagic::compute(5).unwrap();
        assert_eq!(five.cost(), if five.add { 4 } else { 2 });
    }

    // =========================================================================
    // Signed 64-bit Tests
    // =========================================================================

    #[test]
    fn test_signed_magic_special_cases() {
        assert!(SignedMagic::compute(0).is_none());
        assert!(SignedMagic::compute(1).is_none());
        assert!(SignedMagic::compute(-1).is_none());
    }

    #[test]
    fn test_signed_magic_power_of_two() {
        for shift in 1..62 {
            let d = 1i64 << shift;
            let magic = SignedMagic::compute(d).unwrap();
            assert!(magic.is_power_of_two());
            assert_eq!(magic.shift, shift);
        }
    }

    #[test]
    fn test_signed_magic_positive_divisors() {
        for d in 2i64..=20 {
            let magic = SignedMagic::compute(d);
            assert!(magic.is_some(), "Failed for divisor {}", d);

            let m = magic.unwrap();
            for x in [0i64, 1, -1, 10, -10, 100, -100, i64::MAX / 2, i64::MIN / 2] {
                let expected = x / d;
                let actual = apply_signed_magic(x, &m, d);
                assert_eq!(
                    actual, expected,
                    "Failed for x={}, d={}: got {}, expected {}",
                    x, d, actual, expected
                );
            }
        }
    }

    #[test]
    fn test_signed_magic_negative_divisors() {
        for d in [-2i64, -3, -5, -7, -10, -100] {
            let magic = SignedMagic::compute(d);
            assert!(magic.is_some(), "Failed for divisor {}", d);

            let m = magic.unwrap();
            for x in [0i64, 1, -1, 10, -10, 100, -100] {
                let expected = x / d;
                let actual = apply_signed_magic(x, &m, d);
                assert_eq!(
                    actual, expected,
                    "Failed for x={}, d={}: got {}, expected {}",
                    x, d, actual, expected
                );
            }
        }
    }

    #[test]
    fn test_signed_magic_comprehensive() {
        for d in 2i64..=100 {
            let magic = SignedMagic::compute(d).unwrap();

            let test_values = [
                0i64,
                1,
                -1,
                d - 1,
                d,
                d + 1,
                -(d - 1),
                -d,
                -(d + 1),
                1000 * d,
                -1000 * d,
            ];

            for x in test_values {
                let expected = x / d;
                let actual = apply_signed_magic(x, &magic, d);
                assert_eq!(actual, expected, "Failed for x={}, d={}", x, d);
            }
        }
    }

    // =========================================================================
    // Unsigned 32-bit Tests
    // =========================================================================

    #[test]
    fn test_unsigned32_magic_special_cases() {
        assert!(UnsignedMagic32::compute(0).is_none());
        assert!(UnsignedMagic32::compute(1).is_none());
    }

    #[test]
    fn test_unsigned32_magic_power_of_two() {
        for shift in 1..31 {
            let d = 1u32 << shift;
            let magic = UnsignedMagic32::compute(d).unwrap();
            assert!(magic.is_power_of_two());
            assert_eq!(magic.shift, shift);
        }
    }

    #[test]
    fn test_unsigned32_magic_correctness() {
        for d in 2u32..=100 {
            let magic = UnsignedMagic32::compute(d).unwrap();

            for x in [0u32, 1, d - 1, d, d + 1, 1000 * d, u32::MAX] {
                let expected = x / d;
                let actual = apply_unsigned_magic32(x, &magic);
                assert_eq!(actual, expected, "Failed for x={}, d={}", x, d);
            }
        }
    }

    // =========================================================================
    // Signed 32-bit Tests
    // =========================================================================

    #[test]
    fn test_signed32_magic_special_cases() {
        assert!(SignedMagic32::compute(0).is_none());
        assert!(SignedMagic32::compute(1).is_none());
        assert!(SignedMagic32::compute(-1).is_none());
    }

    #[test]
    fn test_signed32_magic_correctness() {
        for d in 2i32..=100 {
            let magic = SignedMagic32::compute(d).unwrap();

            for x in [0i32, 1, -1, d - 1, d, d + 1, -(d - 1), -d, i32::MAX / 2] {
                let expected = x / d;
                let actual = apply_signed_magic32(x, &magic, d);
                assert_eq!(actual, expected, "Failed for x={}, d={}", x, d);
            }
        }
    }

    // =========================================================================
    // Helper Functions Tests
    // =========================================================================

    #[test]
    fn test_is_power_of_two_u64() {
        assert!(!is_power_of_two_u64(0));
        assert!(is_power_of_two_u64(1));
        assert!(is_power_of_two_u64(2));
        assert!(!is_power_of_two_u64(3));
        assert!(is_power_of_two_u64(4));
        assert!(is_power_of_two_u64(1 << 63));
    }

    #[test]
    fn test_is_power_of_two_i64() {
        assert!(!is_power_of_two_i64(0));
        assert!(is_power_of_two_i64(1));
        assert!(is_power_of_two_i64(2));
        assert!(!is_power_of_two_i64(-1));
        assert!(!is_power_of_two_i64(-2));
    }

    #[test]
    fn test_log2_u64() {
        assert_eq!(log2_u64(1), 0);
        assert_eq!(log2_u64(2), 1);
        assert_eq!(log2_u64(4), 2);
        assert_eq!(log2_u64(1 << 63), 63);
    }

    #[test]
    fn test_power_of_two_mask() {
        assert_eq!(power_of_two_mask(1), 1);
        assert_eq!(power_of_two_mask(2), 3);
        assert_eq!(power_of_two_mask(3), 7);
        assert_eq!(power_of_two_mask(8), 255);
    }

    // =========================================================================
    // Test Helpers
    // =========================================================================

    /// Apply unsigned magic division (simulates generated code).
    fn apply_unsigned_magic(x: u64, magic: &UnsignedMagic) -> u64 {
        if magic.is_power_of_two() {
            return x >> magic.shift;
        }

        // mulhu: multiply and get high 64 bits
        let product = (x as u128) * (magic.multiplier as u128);
        let high = (product >> 64) as u64;

        if magic.add {
            // add-shift sequence: ((x - high) >> 1) + high) >> (shift - 1)
            let t = ((x - high) >> 1) + high;
            t >> magic.shift
        } else {
            high >> magic.shift
        }
    }

    /// Apply signed magic division (simulates generated code).
    fn apply_signed_magic(x: i64, magic: &SignedMagic, _d: i64) -> i64 {
        if magic.is_power_of_two() {
            // Signed power-of-2 division with rounding toward zero
            let shift = magic.shift;
            let sign_bit = x >> 63; // All 1s if negative, all 0s if positive
            let bias = (sign_bit as u64 >> (64 - shift)) as i64;
            let result = (x + bias) >> shift;
            // Negate if divisor was negative
            return if magic.negative_divisor {
                -result
            } else {
                result
            };
        }

        // mulhs: signed multiply high
        let product = (x as i128) * (magic.multiplier as i128);
        let mut q = (product >> 64) as i64;

        // Add x if we're using the add-shift sequence
        if magic.add {
            q = q.wrapping_add(x);
        }

        // Arithmetic shift right
        q >>= magic.shift;

        // Add 1 if result is negative (round toward zero)
        q += (q >> 63) as i64 & 1;

        // Negate if divisor was negative
        if magic.negative_divisor { -q } else { q }
    }

    /// Apply unsigned 32-bit magic division.
    fn apply_unsigned_magic32(x: u32, magic: &UnsignedMagic32) -> u32 {
        if magic.is_power_of_two() {
            return x >> magic.shift;
        }

        let product = (x as u64) * (magic.multiplier as u64);
        let high = (product >> 32) as u32;

        if magic.add {
            let t = ((x - high) >> 1) + high;
            t >> magic.shift
        } else {
            high >> magic.shift
        }
    }

    /// Apply signed 32-bit magic division.
    fn apply_signed_magic32(x: i32, magic: &SignedMagic32, _d: i32) -> i32 {
        if magic.is_power_of_two() {
            let shift = magic.shift;
            let sign_bit = x >> 31;
            let bias = (sign_bit as u32 >> (32 - shift)) as i32;
            let result = (x + bias) >> shift;
            return if magic.negative_divisor {
                -result
            } else {
                result
            };
        }

        let product = (x as i64) * (magic.multiplier as i64);
        let mut q = (product >> 32) as i32;

        if magic.add {
            q = q.wrapping_add(x);
        }

        q >>= magic.shift;
        q += (q >> 31) & 1;

        // Negate if divisor was negative
        if magic.negative_divisor { -q } else { q }
    }
}
