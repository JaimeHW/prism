//! Angular conversion functions.
//!
//! Simple multiply operations, inlined for zero overhead.

use super::constants::{DEGREES_PER_RADIAN, RADIANS_PER_DEGREE};

// =============================================================================
// Angular Conversion
// =============================================================================

/// Convert angle x from radians to degrees.
///
/// # Formula
/// degrees = radians * 180 / π
#[inline]
pub fn degrees(x: f64) -> f64 {
    x * DEGREES_PER_RADIAN
}

/// Convert angle x from degrees to radians.
///
/// # Formula
/// radians = degrees * π / 180
#[inline]
pub fn radians(x: f64) -> f64 {
    x * RADIANS_PER_DEGREE
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::super::constants::PI;
    use super::*;

    const EPSILON: f64 = 1e-10;

    // =========================================================================
    // degrees() Tests
    // =========================================================================

    #[test]
    fn test_degrees_zero() {
        assert!(degrees(0.0).abs() < EPSILON);
    }

    #[test]
    fn test_degrees_pi() {
        assert!((degrees(PI) - 180.0).abs() < EPSILON);
    }

    #[test]
    fn test_degrees_half_pi() {
        assert!((degrees(PI / 2.0) - 90.0).abs() < EPSILON);
    }

    #[test]
    fn test_degrees_two_pi() {
        assert!((degrees(2.0 * PI) - 360.0).abs() < EPSILON);
    }

    #[test]
    fn test_degrees_negative() {
        assert!((degrees(-PI) - (-180.0)).abs() < EPSILON);
    }

    #[test]
    fn test_degrees_small() {
        assert!((degrees(PI / 180.0) - 1.0).abs() < EPSILON);
    }

    // =========================================================================
    // radians() Tests
    // =========================================================================

    #[test]
    fn test_radians_zero() {
        assert!(radians(0.0).abs() < EPSILON);
    }

    #[test]
    fn test_radians_180() {
        assert!((radians(180.0) - PI).abs() < EPSILON);
    }

    #[test]
    fn test_radians_90() {
        assert!((radians(90.0) - PI / 2.0).abs() < EPSILON);
    }

    #[test]
    fn test_radians_360() {
        assert!((radians(360.0) - 2.0 * PI).abs() < EPSILON);
    }

    #[test]
    fn test_radians_negative() {
        assert!((radians(-180.0) - (-PI)).abs() < EPSILON);
    }

    #[test]
    fn test_radians_small() {
        assert!((radians(1.0) - PI / 180.0).abs() < EPSILON);
    }

    // =========================================================================
    // Round-trip Tests
    // =========================================================================

    #[test]
    fn test_degrees_radians_roundtrip() {
        for x in [0.0, 30.0, 45.0, 60.0, 90.0, 180.0, 360.0] {
            let result = degrees(radians(x));
            assert!(
                (result - x).abs() < EPSILON,
                "degrees(radians({})) = {} ≠ {}",
                x,
                result,
                x
            );
        }
    }

    #[test]
    fn test_radians_degrees_roundtrip() {
        for x in [0.0, PI / 6.0, PI / 4.0, PI / 3.0, PI / 2.0, PI, 2.0 * PI] {
            let result = radians(degrees(x));
            assert!(
                (result - x).abs() < EPSILON,
                "radians(degrees({})) = {} ≠ {}",
                x,
                result,
                x
            );
        }
    }

    // =========================================================================
    // Edge Cases
    // =========================================================================

    #[test]
    fn test_degrees_special() {
        assert!(degrees(f64::INFINITY).is_infinite());
        assert!(degrees(f64::NAN).is_nan());
    }

    #[test]
    fn test_radians_special() {
        assert!(radians(f64::INFINITY).is_infinite());
        assert!(radians(f64::NAN).is_nan());
    }

    // =========================================================================
    // Consistency with Trig Functions
    // =========================================================================

    #[test]
    fn test_sin_degrees() {
        // sin(30°) = 0.5
        let angle_deg = 30.0;
        let angle_rad = radians(angle_deg);
        assert!((angle_rad.sin() - 0.5).abs() < EPSILON);
    }

    #[test]
    fn test_cos_degrees() {
        // cos(60°) = 0.5
        let angle_deg = 60.0;
        let angle_rad = radians(angle_deg);
        assert!((angle_rad.cos() - 0.5).abs() < EPSILON);
    }

    #[test]
    fn test_tan_degrees() {
        // tan(45°) = 1
        let angle_deg = 45.0;
        let angle_rad = radians(angle_deg);
        assert!((angle_rad.tan() - 1.0).abs() < EPSILON);
    }
}
