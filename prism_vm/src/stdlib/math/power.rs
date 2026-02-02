//! Power and root functions.
//!
//! Includes optimized integer exponentiation and Newton-Raphson isqrt.

use super::super::ModuleError;

// =============================================================================
// Power Functions
// =============================================================================

/// Return x raised to the power y.
///
/// # Performance
/// Uses binary exponentiation for integer y, libm for fractional.
#[inline]
pub fn pow(x: f64, y: f64) -> Result<f64, ModuleError> {
    // Handle special cases
    if x == 0.0 && y < 0.0 {
        return Err(ModuleError::MathDomainError(
            "0.0 cannot be raised to a negative power".to_string(),
        ));
    }
    if x < 0.0 && y.fract() != 0.0 {
        return Err(ModuleError::MathDomainError(
            "negative number cannot be raised to a fractional power".to_string(),
        ));
    }

    // Integer exponent fast path
    if y.fract() == 0.0 && y.abs() <= i32::MAX as f64 {
        let exp = y as i32;
        return Ok(pow_int(x, exp));
    }

    Ok(x.powf(y))
}

/// Fast integer exponentiation using binary exponentiation.
///
/// O(log n) multiplications, no transcendental function calls.
#[inline]
fn pow_int(mut base: f64, mut exp: i32) -> f64 {
    if exp == 0 {
        return 1.0;
    }

    let negative = exp < 0;
    if negative {
        exp = -exp;
    }

    let mut result = 1.0;

    // Binary exponentiation: square and multiply
    while exp > 0 {
        if exp & 1 == 1 {
            result *= base;
        }
        base *= base;
        exp >>= 1;
    }

    if negative { 1.0 / result } else { result }
}

// =============================================================================
// Root Functions
// =============================================================================

/// Return the square root of x.
///
/// # Errors
/// - MathDomainError if x < 0
#[inline]
pub fn sqrt(x: f64) -> Result<f64, ModuleError> {
    if x < 0.0 {
        return Err(ModuleError::MathDomainError(
            "math domain error".to_string(),
        ));
    }
    Ok(x.sqrt())
}

/// Return the integer square root of n.
///
/// This is the floor of the exact square root.
///
/// # Algorithm
/// Uses Newton-Raphson with fast convergence (~5 iterations for 64-bit).
///
/// # Errors
/// - ValueError if n < 0
pub fn isqrt(n: i64) -> Result<i64, ModuleError> {
    if n < 0 {
        return Err(ModuleError::ValueError(
            "isqrt() argument must be nonnegative".to_string(),
        ));
    }

    if n == 0 {
        return Ok(0);
    }

    // Newton-Raphson: x_{n+1} = (x_n + n/x_n) / 2
    // Start with a good initial guess using bit length
    let mut x = 1i64 << ((64 - n.leading_zeros() + 1) / 2);

    loop {
        let x1 = (x + n / x) / 2;
        if x1 >= x {
            break;
        }
        x = x1;
    }

    Ok(x)
}

/// Return the Euclidean distance, sqrt(x² + y²).
///
/// Handles overflow by scaling.
#[inline]
pub fn hypot(x: f64, y: f64) -> f64 {
    x.hypot(y)
}

/// Return the cube root of x.
#[inline]
pub fn cbrt(x: f64) -> f64 {
    x.cbrt()
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    const EPSILON: f64 = 1e-10;

    // =========================================================================
    // pow() Tests
    // =========================================================================

    #[test]
    fn test_pow_zero_exp() {
        assert!((pow(5.0, 0.0).unwrap() - 1.0).abs() < EPSILON);
        assert!((pow(0.0, 0.0).unwrap() - 1.0).abs() < EPSILON); // Python convention
    }

    #[test]
    fn test_pow_positive_int() {
        assert!((pow(2.0, 3.0).unwrap() - 8.0).abs() < EPSILON);
        assert!((pow(3.0, 4.0).unwrap() - 81.0).abs() < EPSILON);
    }

    #[test]
    fn test_pow_negative_int() {
        assert!((pow(2.0, -1.0).unwrap() - 0.5).abs() < EPSILON);
        assert!((pow(2.0, -3.0).unwrap() - 0.125).abs() < EPSILON);
    }

    #[test]
    fn test_pow_fractional() {
        assert!((pow(4.0, 0.5).unwrap() - 2.0).abs() < EPSILON);
        assert!((pow(8.0, 1.0 / 3.0).unwrap() - 2.0).abs() < EPSILON);
    }

    #[test]
    fn test_pow_zero_base_positive_exp() {
        assert!((pow(0.0, 5.0).unwrap()).abs() < EPSILON);
    }

    #[test]
    fn test_pow_zero_base_negative_exp() {
        assert!(pow(0.0, -1.0).is_err());
    }

    #[test]
    fn test_pow_negative_base_fractional_exp() {
        assert!(pow(-2.0, 0.5).is_err());
    }

    #[test]
    fn test_pow_negative_base_integer_exp() {
        assert!((pow(-2.0, 3.0).unwrap() - (-8.0)).abs() < EPSILON);
        assert!((pow(-2.0, 4.0).unwrap() - 16.0).abs() < EPSILON);
    }

    #[test]
    fn test_pow_large_exponent() {
        assert!((pow(2.0, 10.0).unwrap() - 1024.0).abs() < EPSILON);
        assert!((pow(2.0, 20.0).unwrap() - 1048576.0).abs() < EPSILON);
    }

    // =========================================================================
    // pow_int() Tests (internal)
    // =========================================================================

    #[test]
    fn test_pow_int_basic() {
        assert!((pow_int(2.0, 0) - 1.0).abs() < EPSILON);
        assert!((pow_int(2.0, 1) - 2.0).abs() < EPSILON);
        assert!((pow_int(2.0, 10) - 1024.0).abs() < EPSILON);
    }

    #[test]
    fn test_pow_int_negative_exp() {
        assert!((pow_int(2.0, -1) - 0.5).abs() < EPSILON);
        assert!((pow_int(2.0, -2) - 0.25).abs() < EPSILON);
    }

    #[test]
    fn test_pow_int_large() {
        // 2^30 = 1073741824
        assert!((pow_int(2.0, 30) - 1073741824.0).abs() < EPSILON);
    }

    // =========================================================================
    // sqrt() Tests
    // =========================================================================

    #[test]
    fn test_sqrt_perfect_squares() {
        assert!((sqrt(0.0).unwrap()).abs() < EPSILON);
        assert!((sqrt(1.0).unwrap() - 1.0).abs() < EPSILON);
        assert!((sqrt(4.0).unwrap() - 2.0).abs() < EPSILON);
        assert!((sqrt(9.0).unwrap() - 3.0).abs() < EPSILON);
        assert!((sqrt(16.0).unwrap() - 4.0).abs() < EPSILON);
    }

    #[test]
    fn test_sqrt_non_perfect() {
        assert!((sqrt(2.0).unwrap() - std::f64::consts::SQRT_2).abs() < EPSILON);
    }

    #[test]
    fn test_sqrt_domain_error() {
        assert!(sqrt(-1.0).is_err());
        assert!(sqrt(-0.001).is_err());
    }

    #[test]
    fn test_sqrt_special() {
        assert!(sqrt(f64::INFINITY).unwrap().is_infinite());
    }

    // =========================================================================
    // isqrt() Tests
    // =========================================================================

    #[test]
    fn test_isqrt_zero() {
        assert_eq!(isqrt(0).unwrap(), 0);
    }

    #[test]
    fn test_isqrt_one() {
        assert_eq!(isqrt(1).unwrap(), 1);
    }

    #[test]
    fn test_isqrt_perfect_squares() {
        assert_eq!(isqrt(4).unwrap(), 2);
        assert_eq!(isqrt(9).unwrap(), 3);
        assert_eq!(isqrt(16).unwrap(), 4);
        assert_eq!(isqrt(100).unwrap(), 10);
        assert_eq!(isqrt(10000).unwrap(), 100);
    }

    #[test]
    fn test_isqrt_non_perfect() {
        assert_eq!(isqrt(2).unwrap(), 1);
        assert_eq!(isqrt(3).unwrap(), 1);
        assert_eq!(isqrt(5).unwrap(), 2);
        assert_eq!(isqrt(8).unwrap(), 2);
        assert_eq!(isqrt(15).unwrap(), 3);
        assert_eq!(isqrt(17).unwrap(), 4);
    }

    #[test]
    fn test_isqrt_large() {
        assert_eq!(isqrt(1000000).unwrap(), 1000);
        assert_eq!(isqrt(999999).unwrap(), 999);
        assert_eq!(isqrt(1000001).unwrap(), 1000);
    }

    #[test]
    fn test_isqrt_very_large() {
        // Near i64::MAX
        let n: i64 = 9_000_000_000_000_000_000;
        let result = isqrt(n).unwrap();
        assert!(result * result <= n);
        assert!((result + 1) * (result + 1) > n);
    }

    #[test]
    fn test_isqrt_negative() {
        assert!(isqrt(-1).is_err());
    }

    // =========================================================================
    // hypot() Tests
    // =========================================================================

    #[test]
    fn test_hypot_pythagorean_triples() {
        assert!((hypot(3.0, 4.0) - 5.0).abs() < EPSILON);
        assert!((hypot(5.0, 12.0) - 13.0).abs() < EPSILON);
        assert!((hypot(8.0, 15.0) - 17.0).abs() < EPSILON);
    }

    #[test]
    fn test_hypot_zero() {
        assert!((hypot(0.0, 0.0)).abs() < EPSILON);
        assert!((hypot(5.0, 0.0) - 5.0).abs() < EPSILON);
        assert!((hypot(0.0, 5.0) - 5.0).abs() < EPSILON);
    }

    #[test]
    fn test_hypot_negative() {
        assert!((hypot(-3.0, 4.0) - 5.0).abs() < EPSILON);
        assert!((hypot(3.0, -4.0) - 5.0).abs() < EPSILON);
        assert!((hypot(-3.0, -4.0) - 5.0).abs() < EPSILON);
    }

    #[test]
    fn test_hypot_infinity() {
        assert!(hypot(f64::INFINITY, 1.0).is_infinite());
        assert!(hypot(1.0, f64::INFINITY).is_infinite());
    }

    // =========================================================================
    // cbrt() Tests
    // =========================================================================

    #[test]
    fn test_cbrt_positive() {
        assert!((cbrt(8.0) - 2.0).abs() < EPSILON);
        assert!((cbrt(27.0) - 3.0).abs() < EPSILON);
        assert!((cbrt(64.0) - 4.0).abs() < EPSILON);
    }

    #[test]
    fn test_cbrt_negative() {
        assert!((cbrt(-8.0) - (-2.0)).abs() < EPSILON);
        assert!((cbrt(-27.0) - (-3.0)).abs() < EPSILON);
    }

    #[test]
    fn test_cbrt_zero() {
        assert!((cbrt(0.0)).abs() < EPSILON);
    }

    #[test]
    fn test_cbrt_one() {
        assert!((cbrt(1.0) - 1.0).abs() < EPSILON);
    }

    // =========================================================================
    // Identity Tests
    // =========================================================================

    #[test]
    fn test_sqrt_pow_identity() {
        for x in [1.0, 2.0, 4.0, 9.0, 16.0, 100.0] {
            let s = sqrt(x).unwrap();
            let p = pow(x, 0.5).unwrap();
            assert!(
                (s - p).abs() < EPSILON,
                "sqrt(x) ≠ pow(x, 0.5) for x = {}",
                x
            );
        }
    }

    #[test]
    fn test_pow_roundtrip() {
        for x in [2.0, 3.0, 5.0, 10.0] {
            for n in [2, 3, 5, 10] {
                let p = pow(x, n as f64).unwrap();
                let root = pow(p, 1.0 / n as f64).unwrap();
                assert!(
                    (root - x).abs() < EPSILON,
                    "pow(pow(x, n), 1/n) ≠ x for x = {}, n = {}",
                    x,
                    n
                );
            }
        }
    }
}
