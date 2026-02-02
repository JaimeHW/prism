//! Hyperbolic functions.
//!
//! All functions compile directly to hardware instructions or optimized libm.

use super::super::ModuleError;

// =============================================================================
// Hyperbolic Functions
// =============================================================================

/// Return the hyperbolic sine of x.
///
/// sinh(x) = (exp(x) - exp(-x)) / 2
#[inline]
pub fn sinh(x: f64) -> f64 {
    x.sinh()
}

/// Return the hyperbolic cosine of x.
///
/// cosh(x) = (exp(x) + exp(-x)) / 2
#[inline]
pub fn cosh(x: f64) -> f64 {
    x.cosh()
}

/// Return the hyperbolic tangent of x.
///
/// tanh(x) = sinh(x) / cosh(x)
#[inline]
pub fn tanh(x: f64) -> f64 {
    x.tanh()
}

// =============================================================================
// Inverse Hyperbolic Functions
// =============================================================================

/// Return the inverse hyperbolic sine of x.
///
/// asinh(x) = ln(x + sqrt(x² + 1))
#[inline]
pub fn asinh(x: f64) -> f64 {
    x.asinh()
}

/// Return the inverse hyperbolic cosine of x.
///
/// acosh(x) = ln(x + sqrt(x² - 1))
///
/// # Errors
/// - MathDomainError if x < 1
#[inline]
pub fn acosh(x: f64) -> Result<f64, ModuleError> {
    if x < 1.0 {
        return Err(ModuleError::MathDomainError(
            "math domain error".to_string(),
        ));
    }
    Ok(x.acosh())
}

/// Return the inverse hyperbolic tangent of x.
///
/// atanh(x) = 0.5 * ln((1 + x) / (1 - x))
///
/// # Errors
/// - MathDomainError if |x| >= 1
#[inline]
pub fn atanh(x: f64) -> Result<f64, ModuleError> {
    if x <= -1.0 || x >= 1.0 {
        return Err(ModuleError::MathDomainError(
            "math domain error".to_string(),
        ));
    }
    Ok(x.atanh())
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    const EPSILON: f64 = 1e-10;

    // =========================================================================
    // sinh() Tests
    // =========================================================================

    #[test]
    fn test_sinh_zero() {
        assert!(sinh(0.0).abs() < EPSILON);
    }

    #[test]
    fn test_sinh_positive() {
        assert!((sinh(1.0) - 1.1752011936438014).abs() < EPSILON);
    }

    #[test]
    fn test_sinh_negative() {
        assert!((sinh(-1.0) - (-1.1752011936438014)).abs() < EPSILON);
    }

    #[test]
    fn test_sinh_symmetry() {
        // sinh is odd: sinh(-x) = -sinh(x)
        for x in [0.5, 1.0, 2.0, 5.0] {
            assert!((sinh(-x) - (-sinh(x))).abs() < EPSILON);
        }
    }

    #[test]
    fn test_sinh_large() {
        assert!(sinh(710.0).is_finite()); // Near overflow
        assert!(sinh(711.0).is_infinite()); // Overflow
    }

    // =========================================================================
    // cosh() Tests
    // =========================================================================

    #[test]
    fn test_cosh_zero() {
        assert!((cosh(0.0) - 1.0).abs() < EPSILON);
    }

    #[test]
    fn test_cosh_positive() {
        assert!((cosh(1.0) - 1.5430806348152437).abs() < EPSILON);
    }

    #[test]
    fn test_cosh_symmetry() {
        // cosh is even: cosh(-x) = cosh(x)
        for x in [0.5, 1.0, 2.0, 5.0] {
            assert!((cosh(-x) - cosh(x)).abs() < EPSILON);
        }
    }

    #[test]
    fn test_cosh_ge_one() {
        // cosh(x) >= 1 for all x
        for x in [-10.0, -1.0, 0.0, 1.0, 10.0] {
            assert!(cosh(x) >= 1.0);
        }
    }

    // =========================================================================
    // tanh() Tests
    // =========================================================================

    #[test]
    fn test_tanh_zero() {
        assert!(tanh(0.0).abs() < EPSILON);
    }

    #[test]
    fn test_tanh_one() {
        assert!((tanh(1.0) - 0.7615941559557649).abs() < EPSILON);
    }

    #[test]
    fn test_tanh_symmetry() {
        // tanh is odd: tanh(-x) = -tanh(x)
        for x in [0.5, 1.0, 2.0, 5.0] {
            assert!((tanh(-x) - (-tanh(x))).abs() < EPSILON);
        }
    }

    #[test]
    fn test_tanh_bounds() {
        // -1 < tanh(x) < 1 for all finite x
        // For small values, strictly between -1 and 1
        for x in [-1.0, 0.0, 1.0] {
            let t = tanh(x);
            assert!(t > -1.0 && t < 1.0);
        }
        // For large values, approaches ±1 (may equal due to precision)
        for x in [-100.0, -10.0, 10.0, 100.0] {
            let t = tanh(x);
            assert!(t >= -1.0 && t <= 1.0);
        }
    }

    #[test]
    fn test_tanh_limits() {
        assert!((tanh(100.0) - 1.0).abs() < EPSILON);
        assert!((tanh(-100.0) - (-1.0)).abs() < EPSILON);
    }

    // =========================================================================
    // asinh() Tests
    // =========================================================================

    #[test]
    fn test_asinh_zero() {
        assert!(asinh(0.0).abs() < EPSILON);
    }

    #[test]
    fn test_asinh_one() {
        assert!((asinh(1.0) - 0.8813735870195430).abs() < EPSILON);
    }

    #[test]
    fn test_asinh_negative() {
        assert!((asinh(-1.0) - (-0.8813735870195430)).abs() < EPSILON);
    }

    #[test]
    fn test_asinh_roundtrip() {
        for x in [-10.0, -1.0, 0.0, 1.0, 10.0] {
            let result = sinh(asinh(x));
            assert!((result - x).abs() < EPSILON);
        }
    }

    // =========================================================================
    // acosh() Tests
    // =========================================================================

    #[test]
    fn test_acosh_one() {
        assert!(acosh(1.0).unwrap().abs() < EPSILON);
    }

    #[test]
    fn test_acosh_two() {
        assert!((acosh(2.0).unwrap() - 1.3169578969248166).abs() < EPSILON);
    }

    #[test]
    fn test_acosh_domain_error() {
        assert!(acosh(0.5).is_err());
        assert!(acosh(0.0).is_err());
        assert!(acosh(-1.0).is_err());
    }

    #[test]
    fn test_acosh_roundtrip() {
        for x in [1.0, 1.5, 2.0, 10.0] {
            let result = cosh(acosh(x).unwrap());
            assert!((result - x).abs() < EPSILON);
        }
    }

    // =========================================================================
    // atanh() Tests
    // =========================================================================

    #[test]
    fn test_atanh_zero() {
        assert!(atanh(0.0).unwrap().abs() < EPSILON);
    }

    #[test]
    fn test_atanh_half() {
        assert!((atanh(0.5).unwrap() - 0.5493061443340548).abs() < EPSILON);
    }

    #[test]
    fn test_atanh_domain_error() {
        assert!(atanh(1.0).is_err());
        assert!(atanh(-1.0).is_err());
        assert!(atanh(1.5).is_err());
    }

    #[test]
    fn test_atanh_roundtrip() {
        for x in [-0.9, -0.5, 0.0, 0.5, 0.9] {
            let result = tanh(atanh(x).unwrap());
            assert!((result - x).abs() < EPSILON);
        }
    }

    // =========================================================================
    // Identity Tests
    // =========================================================================

    #[test]
    fn test_hyperbolic_identity() {
        // cosh²(x) - sinh²(x) = 1
        for x in [0.0, 0.5, 1.0, 2.0, 5.0] {
            let c = cosh(x);
            let s = sinh(x);
            assert!(
                (c * c - s * s - 1.0).abs() < EPSILON,
                "cosh²(x) - sinh²(x) ≠ 1 for x = {}",
                x
            );
        }
    }

    #[test]
    fn test_tanh_equals_sinh_over_cosh() {
        for x in [0.1, 0.5, 1.0, 2.0] {
            let t = tanh(x);
            let ratio = sinh(x) / cosh(x);
            assert!(
                (t - ratio).abs() < EPSILON,
                "tanh(x) ≠ sinh(x)/cosh(x) for x = {}",
                x
            );
        }
    }
}
