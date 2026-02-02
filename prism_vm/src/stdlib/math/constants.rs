//! Mathematical constants.
//!
//! All constants are defined as `const` for zero-cost inlining.
//! Values match Python 3.12 `math` module exactly.

/// π (pi) - Ratio of circle circumference to diameter.
pub const PI: f64 = std::f64::consts::PI;

/// e (Euler's number) - Base of natural logarithm.
pub const E: f64 = std::f64::consts::E;

/// τ (tau) - Circle constant, 2π.
pub const TAU: f64 = std::f64::consts::TAU;

/// Positive infinity.
pub const INFINITY: f64 = f64::INFINITY;

/// Negative infinity.
pub const NEG_INFINITY: f64 = f64::NEG_INFINITY;

/// Not a Number (NaN).
pub const NAN: f64 = f64::NAN;

// =============================================================================
// Additional Constants for Internal Use
// =============================================================================

/// 180/π for degrees conversion.
pub(crate) const DEGREES_PER_RADIAN: f64 = 180.0 / PI;

/// π/180 for radians conversion.
pub(crate) const RADIANS_PER_DEGREE: f64 = PI / 180.0;

/// ln(2) for log2 conversion.
pub(crate) const LN_2: f64 = std::f64::consts::LN_2;

/// ln(10) for log10 conversion.
pub(crate) const LN_10: f64 = std::f64::consts::LN_10;

/// sqrt(2)
pub(crate) const SQRT_2: f64 = std::f64::consts::SQRT_2;

/// 1/sqrt(2)
pub(crate) const FRAC_1_SQRT_2: f64 = std::f64::consts::FRAC_1_SQRT_2;

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pi() {
        assert!((PI - 3.141592653589793).abs() < 1e-15);
    }

    #[test]
    fn test_e() {
        assert!((E - 2.718281828459045).abs() < 1e-15);
    }

    #[test]
    fn test_tau() {
        assert!((TAU - 6.283185307179586).abs() < 1e-15);
        assert!((TAU - 2.0 * PI).abs() < 1e-15);
    }

    #[test]
    fn test_infinity() {
        assert!(INFINITY.is_infinite());
        assert!(INFINITY.is_sign_positive());
        assert!(NEG_INFINITY.is_infinite());
        assert!(NEG_INFINITY.is_sign_negative());
    }

    #[test]
    fn test_nan() {
        assert!(NAN.is_nan());
    }

    #[test]
    fn test_degrees_per_radian() {
        assert!((DEGREES_PER_RADIAN - 57.29577951308232).abs() < 1e-10);
    }

    #[test]
    fn test_radians_per_degree() {
        assert!((RADIANS_PER_DEGREE - 0.017453292519943295).abs() < 1e-15);
    }

    #[test]
    fn test_ln_2() {
        assert!((LN_2 - 0.6931471805599453).abs() < 1e-15);
    }

    #[test]
    fn test_ln_10() {
        assert!((LN_10 - 2.302585092994046).abs() < 1e-15);
    }

    #[test]
    fn test_sqrt_2() {
        assert!((SQRT_2 - 1.4142135623730951).abs() < 1e-15);
    }

    #[test]
    fn test_frac_1_sqrt_2() {
        assert!((FRAC_1_SQRT_2 - 0.7071067811865476).abs() < 1e-15);
    }

    #[test]
    fn test_constant_relationships() {
        // τ = 2π
        assert!((TAU - 2.0 * PI).abs() < 1e-15);
        // 1/sqrt(2) = sqrt(2)/2
        assert!((FRAC_1_SQRT_2 - SQRT_2 / 2.0).abs() < 1e-15);
        // radians * degrees_per_radian = 180 for π
        assert!((PI * DEGREES_PER_RADIAN - 180.0).abs() < 1e-10);
    }
}
