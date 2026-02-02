//! Trigonometric functions.
//!
//! All functions compile to hardware sin/cos/tan instructions via libm.
//! Angles are in radians.

use super::super::ModuleError;

// =============================================================================
// Basic Trigonometric Functions
// =============================================================================

/// Return the sine of x (in radians).
///
/// # Performance
/// Compiles to FSIN instruction on x87 or calls optimized libm.
#[inline]
pub fn sin(x: f64) -> f64 {
    x.sin()
}

/// Return the cosine of x (in radians).
///
/// # Performance
/// Compiles to FCOS instruction on x87 or calls optimized libm.
#[inline]
pub fn cos(x: f64) -> f64 {
    x.cos()
}

/// Return the tangent of x (in radians).
///
/// # Performance
/// Compiles to FPTAN instruction on x87 or calls optimized libm.
#[inline]
pub fn tan(x: f64) -> f64 {
    x.tan()
}

// =============================================================================
// Inverse Trigonometric Functions
// =============================================================================

/// Return the arc sine of x, in radians.
///
/// Result is in [-π/2, π/2].
///
/// # Errors
/// - MathDomainError if x < -1 or x > 1
#[inline]
pub fn asin(x: f64) -> Result<f64, ModuleError> {
    if x < -1.0 || x > 1.0 {
        return Err(ModuleError::MathDomainError(
            "math domain error".to_string(),
        ));
    }
    Ok(x.asin())
}

/// Return the arc cosine of x, in radians.
///
/// Result is in [0, π].
///
/// # Errors
/// - MathDomainError if x < -1 or x > 1
#[inline]
pub fn acos(x: f64) -> Result<f64, ModuleError> {
    if x < -1.0 || x > 1.0 {
        return Err(ModuleError::MathDomainError(
            "math domain error".to_string(),
        ));
    }
    Ok(x.acos())
}

/// Return the arc tangent of x, in radians.
///
/// Result is in [-π/2, π/2].
#[inline]
pub fn atan(x: f64) -> f64 {
    x.atan()
}

/// Return atan(y/x), in radians.
///
/// Result is in [-π, π]. Unlike atan(y/x), atan2 gives correct
/// quadrant based on signs of both arguments.
///
/// # Python Semantics
/// - `math.atan2(1, 1)` → π/4
/// - `math.atan2(-1, -1)` → -3π/4
/// - `math.atan2(0, -1)` → π
/// - `math.atan2(0, 1)` → 0
#[inline]
pub fn atan2(y: f64, x: f64) -> f64 {
    y.atan2(x)
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    const EPSILON: f64 = 1e-10;

    // =========================================================================
    // sin() Tests
    // =========================================================================

    #[test]
    fn test_sin_zero() {
        assert!(sin(0.0).abs() < EPSILON);
    }

    #[test]
    fn test_sin_pi_half() {
        assert!((sin(PI / 2.0) - 1.0).abs() < EPSILON);
    }

    #[test]
    fn test_sin_pi() {
        assert!(sin(PI).abs() < EPSILON);
    }

    #[test]
    fn test_sin_negative() {
        assert!((sin(-PI / 2.0) - (-1.0)).abs() < EPSILON);
    }

    #[test]
    fn test_sin_special() {
        assert!(sin(f64::NAN).is_nan());
    }

    // =========================================================================
    // cos() Tests
    // =========================================================================

    #[test]
    fn test_cos_zero() {
        assert!((cos(0.0) - 1.0).abs() < EPSILON);
    }

    #[test]
    fn test_cos_pi_half() {
        assert!(cos(PI / 2.0).abs() < EPSILON);
    }

    #[test]
    fn test_cos_pi() {
        assert!((cos(PI) - (-1.0)).abs() < EPSILON);
    }

    #[test]
    fn test_cos_negative() {
        assert!((cos(-PI) - (-1.0)).abs() < EPSILON);
    }

    #[test]
    fn test_cos_special() {
        assert!(cos(f64::NAN).is_nan());
    }

    // =========================================================================
    // tan() Tests
    // =========================================================================

    #[test]
    fn test_tan_zero() {
        assert!(tan(0.0).abs() < EPSILON);
    }

    #[test]
    fn test_tan_pi_quarter() {
        assert!((tan(PI / 4.0) - 1.0).abs() < EPSILON);
    }

    #[test]
    fn test_tan_negative() {
        assert!((tan(-PI / 4.0) - (-1.0)).abs() < EPSILON);
    }

    #[test]
    fn test_tan_special() {
        assert!(tan(f64::NAN).is_nan());
    }

    // =========================================================================
    // asin() Tests
    // =========================================================================

    #[test]
    fn test_asin_zero() {
        assert!(asin(0.0).unwrap().abs() < EPSILON);
    }

    #[test]
    fn test_asin_one() {
        assert!((asin(1.0).unwrap() - PI / 2.0).abs() < EPSILON);
    }

    #[test]
    fn test_asin_neg_one() {
        assert!((asin(-1.0).unwrap() - (-PI / 2.0)).abs() < EPSILON);
    }

    #[test]
    fn test_asin_out_of_range() {
        assert!(asin(1.1).is_err());
        assert!(asin(-1.1).is_err());
    }

    // =========================================================================
    // acos() Tests
    // =========================================================================

    #[test]
    fn test_acos_one() {
        assert!(acos(1.0).unwrap().abs() < EPSILON);
    }

    #[test]
    fn test_acos_zero() {
        assert!((acos(0.0).unwrap() - PI / 2.0).abs() < EPSILON);
    }

    #[test]
    fn test_acos_neg_one() {
        assert!((acos(-1.0).unwrap() - PI).abs() < EPSILON);
    }

    #[test]
    fn test_acos_out_of_range() {
        assert!(acos(1.1).is_err());
        assert!(acos(-1.1).is_err());
    }

    // =========================================================================
    // atan() Tests
    // =========================================================================

    #[test]
    fn test_atan_zero() {
        assert!(atan(0.0).abs() < EPSILON);
    }

    #[test]
    fn test_atan_one() {
        assert!((atan(1.0) - PI / 4.0).abs() < EPSILON);
    }

    #[test]
    fn test_atan_negative() {
        assert!((atan(-1.0) - (-PI / 4.0)).abs() < EPSILON);
    }

    #[test]
    fn test_atan_infinity() {
        assert!((atan(f64::INFINITY) - PI / 2.0).abs() < EPSILON);
        assert!((atan(f64::NEG_INFINITY) - (-PI / 2.0)).abs() < EPSILON);
    }

    // =========================================================================
    // atan2() Tests
    // =========================================================================

    #[test]
    fn test_atan2_first_quadrant() {
        assert!((atan2(1.0, 1.0) - PI / 4.0).abs() < EPSILON);
    }

    #[test]
    fn test_atan2_second_quadrant() {
        let result = atan2(1.0, -1.0);
        assert!((result - 3.0 * PI / 4.0).abs() < EPSILON);
    }

    #[test]
    fn test_atan2_third_quadrant() {
        let result = atan2(-1.0, -1.0);
        assert!((result - (-3.0 * PI / 4.0)).abs() < EPSILON);
    }

    #[test]
    fn test_atan2_fourth_quadrant() {
        let result = atan2(-1.0, 1.0);
        assert!((result - (-PI / 4.0)).abs() < EPSILON);
    }

    #[test]
    fn test_atan2_axes() {
        assert!((atan2(0.0, 1.0)).abs() < EPSILON); // 0
        assert!((atan2(1.0, 0.0) - PI / 2.0).abs() < EPSILON); // π/2
        assert!((atan2(0.0, -1.0) - PI).abs() < EPSILON); // π
        assert!((atan2(-1.0, 0.0) - (-PI / 2.0)).abs() < EPSILON); // -π/2
    }

    // =========================================================================
    // Identity Tests
    // =========================================================================

    #[test]
    fn test_pythagorean_identity() {
        for angle in [0.0, 0.5, 1.0, PI / 4.0, PI / 2.0, PI] {
            let s = sin(angle);
            let c = cos(angle);
            assert!(
                (s * s + c * c - 1.0).abs() < EPSILON,
                "sin²(x) + cos²(x) ≠ 1 for x = {}",
                angle
            );
        }
    }

    #[test]
    fn test_tan_equals_sin_over_cos() {
        for angle in [0.1, 0.5, 1.0, PI / 4.0, PI / 3.0] {
            let t = tan(angle);
            let ratio = sin(angle) / cos(angle);
            assert!(
                (t - ratio).abs() < EPSILON,
                "tan(x) ≠ sin(x)/cos(x) for x = {}",
                angle
            );
        }
    }

    #[test]
    fn test_asin_sin_roundtrip() {
        for x in [-1.0, -0.5, 0.0, 0.5, 1.0] {
            let asin_x = asin(x).unwrap();
            let sin_asin_x = sin(asin_x);
            assert!(
                (sin_asin_x - x).abs() < EPSILON,
                "sin(asin(x)) ≠ x for x = {}",
                x
            );
        }
    }

    #[test]
    fn test_acos_cos_roundtrip() {
        for x in [-1.0, -0.5, 0.0, 0.5, 1.0] {
            let acos_x = acos(x).unwrap();
            let cos_acos_x = cos(acos_x);
            assert!(
                (cos_acos_x - x).abs() < EPSILON,
                "cos(acos(x)) ≠ x for x = {}",
                x
            );
        }
    }
}
