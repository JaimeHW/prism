//! Basic mathematical functions.
//!
//! These compile directly to hardware instructions on x64.
//! All functions are `#[inline]` for maximum performance.

use super::super::ModuleError;

// =============================================================================
// Rounding Functions
// =============================================================================

/// Return the ceiling of x, the smallest integer greater than or equal to x.
///
/// # Python Semantics
/// - `math.ceil(2.1)` → 3
/// - `math.ceil(-2.1)` → -2
/// - `math.ceil(inf)` → inf
/// - `math.ceil(nan)` → nan
#[inline]
pub fn ceil(x: f64) -> f64 {
    x.ceil()
}

/// Return the floor of x, the largest integer less than or equal to x.
///
/// # Python Semantics
/// - `math.floor(2.9)` → 2
/// - `math.floor(-2.9)` → -3
/// - `math.floor(inf)` → inf
/// - `math.floor(nan)` → nan
#[inline]
pub fn floor(x: f64) -> f64 {
    x.floor()
}

/// Return x with the fractional part removed.
///
/// # Python Semantics
/// - `math.trunc(2.9)` → 2
/// - `math.trunc(-2.9)` → -2
/// - `math.trunc(inf)` → inf
/// - `math.trunc(nan)` → nan
#[inline]
pub fn trunc(x: f64) -> f64 {
    x.trunc()
}

// =============================================================================
// Absolute Value and Sign
// =============================================================================

/// Return the absolute value of x.
///
/// # Performance
/// Compiles to single x64 instruction (ANDPS to clear sign bit).
#[inline]
pub fn fabs(x: f64) -> f64 {
    x.abs()
}

/// Return x with the sign of y.
///
/// # Python Semantics
/// - `math.copysign(1.0, -0.0)` → -1.0
/// - `math.copysign(inf, -1.0)` → -inf
/// - `math.copysign(nan, -1.0)` → nan (with negative sign)
#[inline]
pub fn copysign(x: f64, y: f64) -> f64 {
    x.copysign(y)
}

// =============================================================================
// Modulo and Remainder
// =============================================================================

/// Return x modulo y.
///
/// # Python Semantics
/// - `math.fmod(5.0, 3.0)` → 2.0
/// - `math.fmod(-5.0, 3.0)` → -2.0
/// - Result has same sign as x (unlike Python's % operator)
///
/// # Errors
/// - ValueError if y is 0
#[inline]
pub fn fmod(x: f64, y: f64) -> Result<f64, ModuleError> {
    if y == 0.0 {
        return Err(ModuleError::MathDomainError(
            "fmod() division by zero".to_string(),
        ));
    }
    Ok(x % y)
}

/// Return the fractional and integer parts of x.
///
/// Both results carry the sign of x.
///
/// # Python Semantics
/// - `math.modf(3.5)` → (0.5, 3.0)
/// - `math.modf(-3.5)` → (-0.5, -3.0)
///
/// # Returns
/// (fractional_part, integer_part)
#[inline]
pub fn modf(x: f64) -> (f64, f64) {
    let int_part = x.trunc();
    let frac_part = x - int_part;
    (frac_part, int_part)
}

/// Return IEEE 754-style remainder of x with respect to y.
///
/// # Python Semantics
/// - `math.remainder(5.0, 3.0)` → -1.0 (rounds to nearest)
/// - Unlike fmod, chooses closest value to 0
#[inline]
pub fn remainder(x: f64, y: f64) -> Result<f64, ModuleError> {
    if y == 0.0 {
        return Err(ModuleError::MathDomainError(
            "remainder() division by zero".to_string(),
        ));
    }
    if x.is_infinite() {
        return Err(ModuleError::MathDomainError(
            "remainder() not defined for infinite x".to_string(),
        ));
    }
    if y.is_nan() || x.is_nan() {
        return Ok(f64::NAN);
    }

    // IEEE 754 remainder: x - n*y where n = round(x/y)
    let n = (x / y).round();
    Ok(x - n * y)
}

/// Return (m, e) such that x = m * 2**e exactly.
///
/// m is normalized so 0.5 <= abs(m) < 1.0 (except for 0, inf, nan).
///
/// # Python Semantics
/// - `math.frexp(8.0)` → (0.5, 4)  because 8 = 0.5 * 2^4
/// - `math.frexp(0.0)` → (0.0, 0)
#[inline]
pub fn frexp(x: f64) -> (f64, i32) {
    if x == 0.0 || x.is_nan() || x.is_infinite() {
        return (x, 0);
    }

    // Use bit manipulation for maximum performance
    let bits = x.to_bits();
    let sign = bits & 0x8000_0000_0000_0000;
    let exponent = ((bits >> 52) & 0x7FF) as i32;
    let mantissa = bits & 0x000F_FFFF_FFFF_FFFF;

    if exponent == 0 {
        // Subnormal number - normalize first
        let normalized = x * (1u64 << 52) as f64;
        let (m, e) = frexp(normalized);
        return (m, e - 52);
    }

    // Reconstruct with exponent = -1 (so value is in [0.5, 1.0))
    let new_bits = sign | (0x3FE << 52) | mantissa;
    let m = f64::from_bits(new_bits);
    let e = exponent - 0x3FE;

    (m, e)
}

/// Return m * 2**e (inverse of frexp).
///
/// # Python Semantics
/// - `math.ldexp(0.5, 4)` → 8.0
/// - `math.ldexp(1.0, 1024)` → inf (overflow)
#[inline]
pub fn ldexp(m: f64, e: i32) -> f64 {
    // Use built-in for correctness with edge cases
    libm::ldexp(m, e)
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // ceil() Tests
    // =========================================================================

    #[test]
    fn test_ceil_positive() {
        assert_eq!(ceil(2.1), 3.0);
        assert_eq!(ceil(2.9), 3.0);
        assert_eq!(ceil(2.0), 2.0);
    }

    #[test]
    fn test_ceil_negative() {
        assert_eq!(ceil(-2.1), -2.0);
        assert_eq!(ceil(-2.9), -2.0);
        assert_eq!(ceil(-2.0), -2.0);
    }

    #[test]
    fn test_ceil_zero() {
        assert_eq!(ceil(0.0), 0.0);
        assert_eq!(ceil(-0.0), -0.0);
    }

    #[test]
    fn test_ceil_special() {
        assert!(ceil(f64::INFINITY).is_infinite());
        assert!(ceil(f64::NAN).is_nan());
    }

    // =========================================================================
    // floor() Tests
    // =========================================================================

    #[test]
    fn test_floor_positive() {
        assert_eq!(floor(2.1), 2.0);
        assert_eq!(floor(2.9), 2.0);
        assert_eq!(floor(2.0), 2.0);
    }

    #[test]
    fn test_floor_negative() {
        assert_eq!(floor(-2.1), -3.0);
        assert_eq!(floor(-2.9), -3.0);
        assert_eq!(floor(-2.0), -2.0);
    }

    #[test]
    fn test_floor_special() {
        assert!(floor(f64::INFINITY).is_infinite());
        assert!(floor(f64::NAN).is_nan());
    }

    // =========================================================================
    // trunc() Tests
    // =========================================================================

    #[test]
    fn test_trunc_positive() {
        assert_eq!(trunc(2.1), 2.0);
        assert_eq!(trunc(2.9), 2.0);
    }

    #[test]
    fn test_trunc_negative() {
        assert_eq!(trunc(-2.1), -2.0);
        assert_eq!(trunc(-2.9), -2.0);
    }

    #[test]
    fn test_trunc_special() {
        assert!(trunc(f64::INFINITY).is_infinite());
        assert!(trunc(f64::NAN).is_nan());
    }

    // =========================================================================
    // fabs() Tests
    // =========================================================================

    #[test]
    fn test_fabs_positive() {
        assert_eq!(fabs(3.14), 3.14);
    }

    #[test]
    fn test_fabs_negative() {
        assert_eq!(fabs(-3.14), 3.14);
    }

    #[test]
    fn test_fabs_zero() {
        assert_eq!(fabs(0.0), 0.0);
        assert_eq!(fabs(-0.0), 0.0);
    }

    #[test]
    fn test_fabs_infinity() {
        assert_eq!(fabs(f64::INFINITY), f64::INFINITY);
        assert_eq!(fabs(f64::NEG_INFINITY), f64::INFINITY);
    }

    #[test]
    fn test_fabs_nan() {
        assert!(fabs(f64::NAN).is_nan());
    }

    // =========================================================================
    // copysign() Tests
    // =========================================================================

    #[test]
    fn test_copysign_positive_to_positive() {
        assert_eq!(copysign(1.0, 1.0), 1.0);
    }

    #[test]
    fn test_copysign_positive_to_negative() {
        assert_eq!(copysign(1.0, -1.0), -1.0);
    }

    #[test]
    fn test_copysign_negative_to_positive() {
        assert_eq!(copysign(-1.0, 1.0), 1.0);
    }

    #[test]
    fn test_copysign_with_zero() {
        assert_eq!(copysign(1.0, 0.0), 1.0);
        assert_eq!(copysign(1.0, -0.0), -1.0);
    }

    #[test]
    fn test_copysign_infinity() {
        assert_eq!(copysign(f64::INFINITY, -1.0), f64::NEG_INFINITY);
    }

    #[test]
    fn test_copysign_nan() {
        let result = copysign(f64::NAN, -1.0);
        assert!(result.is_nan());
        assert!(result.is_sign_negative());
    }

    // =========================================================================
    // fmod() Tests
    // =========================================================================

    #[test]
    fn test_fmod_basic() {
        assert!((fmod(5.0, 3.0).unwrap() - 2.0).abs() < 1e-15);
        assert!((fmod(7.0, 4.0).unwrap() - 3.0).abs() < 1e-15);
    }

    #[test]
    fn test_fmod_negative() {
        assert!((fmod(-5.0, 3.0).unwrap() - (-2.0)).abs() < 1e-15);
        assert!((fmod(5.0, -3.0).unwrap() - 2.0).abs() < 1e-15);
    }

    #[test]
    fn test_fmod_division_by_zero() {
        assert!(fmod(5.0, 0.0).is_err());
    }

    // =========================================================================
    // modf() Tests
    // =========================================================================

    #[test]
    fn test_modf_positive() {
        let (frac, int) = modf(3.5);
        assert!((frac - 0.5).abs() < 1e-15);
        assert!((int - 3.0).abs() < 1e-15);
    }

    #[test]
    fn test_modf_negative() {
        let (frac, int) = modf(-3.5);
        assert!((frac - (-0.5)).abs() < 1e-15);
        assert!((int - (-3.0)).abs() < 1e-15);
    }

    #[test]
    fn test_modf_integer() {
        let (frac, int) = modf(5.0);
        assert!((frac).abs() < 1e-15);
        assert!((int - 5.0).abs() < 1e-15);
    }

    // =========================================================================
    // remainder() Tests
    // =========================================================================

    #[test]
    fn test_remainder_basic() {
        // 5 / 3 = 1.67, rounds to 2, so 5 - 2*3 = -1
        assert!((remainder(5.0, 3.0).unwrap() - (-1.0)).abs() < 1e-15);
    }

    #[test]
    fn test_remainder_division_by_zero() {
        assert!(remainder(5.0, 0.0).is_err());
    }

    #[test]
    fn test_remainder_infinite_x() {
        assert!(remainder(f64::INFINITY, 1.0).is_err());
    }

    // =========================================================================
    // frexp() Tests
    // =========================================================================

    #[test]
    fn test_frexp_power_of_two() {
        let (m, e) = frexp(8.0);
        assert!((m - 0.5).abs() < 1e-15);
        assert_eq!(e, 4);
    }

    #[test]
    fn test_frexp_non_power() {
        let (m, e) = frexp(3.0);
        assert!((m - 0.75).abs() < 1e-15);
        assert_eq!(e, 2);
    }

    #[test]
    fn test_frexp_zero() {
        let (m, e) = frexp(0.0);
        assert_eq!(m, 0.0);
        assert_eq!(e, 0);
    }

    #[test]
    fn test_frexp_negative() {
        let (m, e) = frexp(-4.0);
        assert!((m - (-0.5)).abs() < 1e-15);
        assert_eq!(e, 3);
    }

    // =========================================================================
    // ldexp() Tests
    // =========================================================================

    #[test]
    fn test_ldexp_basic() {
        assert!((ldexp(0.5, 4) - 8.0).abs() < 1e-15);
    }

    #[test]
    fn test_ldexp_negative_exp() {
        assert!((ldexp(2.0, -1) - 1.0).abs() < 1e-15);
    }

    #[test]
    fn test_ldexp_overflow() {
        assert!(ldexp(1.0, 1024).is_infinite());
    }

    // =========================================================================
    // Round-trip Tests
    // =========================================================================

    #[test]
    fn test_frexp_ldexp_roundtrip() {
        for x in [1.0, 2.0, 3.14159, 100.0, 0.001, -42.5] {
            let (m, e) = frexp(x);
            let result = ldexp(m, e);
            assert!(
                (result - x).abs() < 1e-15,
                "frexp/ldexp roundtrip failed for {}: got {}",
                x,
                result
            );
        }
    }
}
