//! Exponential and logarithmic functions.
//!
//! All functions compile directly to hardware instructions where available.
//! Handles special cases (inf, nan, domain errors) per Python 3.12.

use super::super::ModuleError;

// =============================================================================
// Exponential Functions
// =============================================================================

/// Return e raised to the power x.
///
/// # Performance
/// Compiles to optimized libm call.
#[inline]
pub fn exp(x: f64) -> f64 {
    x.exp()
}

/// Return 2 raised to the power x.
#[inline]
pub fn exp2(x: f64) -> f64 {
    x.exp2()
}

/// Return exp(x) - 1.
///
/// More accurate than `exp(x) - 1` for small x.
#[inline]
pub fn expm1(x: f64) -> f64 {
    x.exp_m1()
}

// =============================================================================
// Logarithmic Functions
// =============================================================================

/// Return the natural logarithm of x (base e).
///
/// # Errors
/// - MathDomainError if x <= 0
#[inline]
pub fn log(x: f64) -> Result<f64, ModuleError> {
    if x <= 0.0 {
        return Err(ModuleError::MathDomainError(
            "math domain error".to_string(),
        ));
    }
    Ok(x.ln())
}

/// Return the natural logarithm of x with optional base.
///
/// log(x, base) = ln(x) / ln(base)
#[inline]
pub fn log_base(x: f64, base: f64) -> Result<f64, ModuleError> {
    if x <= 0.0 || base <= 0.0 {
        return Err(ModuleError::MathDomainError(
            "math domain error".to_string(),
        ));
    }
    if base == 1.0 {
        return Err(ModuleError::MathDomainError(
            "math domain error".to_string(),
        ));
    }
    Ok(x.ln() / base.ln())
}

/// Return the base-2 logarithm of x.
///
/// # Errors
/// - MathDomainError if x <= 0
#[inline]
pub fn log2(x: f64) -> Result<f64, ModuleError> {
    if x <= 0.0 {
        return Err(ModuleError::MathDomainError(
            "math domain error".to_string(),
        ));
    }
    Ok(x.log2())
}

/// Return the base-10 logarithm of x.
///
/// # Errors
/// - MathDomainError if x <= 0
#[inline]
pub fn log10(x: f64) -> Result<f64, ModuleError> {
    if x <= 0.0 {
        return Err(ModuleError::MathDomainError(
            "math domain error".to_string(),
        ));
    }
    Ok(x.log10())
}

/// Return ln(1 + x).
///
/// More accurate than `log(1 + x)` for small x.
///
/// # Errors
/// - MathDomainError if x <= -1
#[inline]
pub fn log1p(x: f64) -> Result<f64, ModuleError> {
    if x <= -1.0 {
        return Err(ModuleError::MathDomainError(
            "math domain error".to_string(),
        ));
    }
    Ok(x.ln_1p())
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    const EPSILON: f64 = 1e-10;

    // =========================================================================
    // exp() Tests
    // =========================================================================

    #[test]
    fn test_exp_zero() {
        assert!((exp(0.0) - 1.0).abs() < EPSILON);
    }

    #[test]
    fn test_exp_one() {
        assert!((exp(1.0) - std::f64::consts::E).abs() < EPSILON);
    }

    #[test]
    fn test_exp_negative() {
        assert!((exp(-1.0) - 0.36787944117144233).abs() < EPSILON);
    }

    #[test]
    fn test_exp_large() {
        assert!(exp(709.0).is_finite());
        assert!(exp(710.0).is_infinite());
    }

    #[test]
    fn test_exp_neg_large() {
        assert!((exp(-1000.0)).abs() < EPSILON); // Underflow to 0
    }

    #[test]
    fn test_exp_special() {
        assert!(exp(f64::INFINITY).is_infinite());
        assert!(exp(f64::NEG_INFINITY).abs() < EPSILON);
        assert!(exp(f64::NAN).is_nan());
    }

    // =========================================================================
    // exp2() Tests
    // =========================================================================

    #[test]
    fn test_exp2_zero() {
        assert!((exp2(0.0) - 1.0).abs() < EPSILON);
    }

    #[test]
    fn test_exp2_integer() {
        assert!((exp2(3.0) - 8.0).abs() < EPSILON);
        assert!((exp2(10.0) - 1024.0).abs() < EPSILON);
    }

    #[test]
    fn test_exp2_fractional() {
        assert!((exp2(0.5) - std::f64::consts::SQRT_2).abs() < EPSILON);
    }

    #[test]
    fn test_exp2_negative() {
        assert!((exp2(-1.0) - 0.5).abs() < EPSILON);
        assert!((exp2(-2.0) - 0.25).abs() < EPSILON);
    }

    // =========================================================================
    // expm1() Tests
    // =========================================================================

    #[test]
    fn test_expm1_zero() {
        assert!(expm1(0.0).abs() < EPSILON);
    }

    #[test]
    fn test_expm1_small() {
        // For small x, expm1(x) ≈ x
        let x = 1e-10;
        assert!((expm1(x) - x).abs() < 1e-15);
    }

    #[test]
    fn test_expm1_one() {
        assert!((expm1(1.0) - (std::f64::consts::E - 1.0)).abs() < EPSILON);
    }

    // =========================================================================
    // log() Tests
    // =========================================================================

    #[test]
    fn test_log_one() {
        assert!(log(1.0).unwrap().abs() < EPSILON);
    }

    #[test]
    fn test_log_e() {
        assert!((log(std::f64::consts::E).unwrap() - 1.0).abs() < EPSILON);
    }

    #[test]
    fn test_log_e_squared() {
        let e2 = std::f64::consts::E * std::f64::consts::E;
        assert!((log(e2).unwrap() - 2.0).abs() < EPSILON);
    }

    #[test]
    fn test_log_domain_error() {
        assert!(log(0.0).is_err());
        assert!(log(-1.0).is_err());
    }

    // =========================================================================
    // log_base() Tests
    // =========================================================================

    #[test]
    fn test_log_base_2() {
        assert!((log_base(8.0, 2.0).unwrap() - 3.0).abs() < EPSILON);
    }

    #[test]
    fn test_log_base_10() {
        assert!((log_base(1000.0, 10.0).unwrap() - 3.0).abs() < EPSILON);
    }

    #[test]
    fn test_log_base_domain_error() {
        assert!(log_base(0.0, 2.0).is_err());
        assert!(log_base(1.0, 1.0).is_err()); // Base 1 is invalid
        assert!(log_base(-1.0, 2.0).is_err());
    }

    // =========================================================================
    // log2() Tests
    // =========================================================================

    #[test]
    fn test_log2_one() {
        assert!(log2(1.0).unwrap().abs() < EPSILON);
    }

    #[test]
    fn test_log2_powers() {
        assert!((log2(2.0).unwrap() - 1.0).abs() < EPSILON);
        assert!((log2(8.0).unwrap() - 3.0).abs() < EPSILON);
        assert!((log2(1024.0).unwrap() - 10.0).abs() < EPSILON);
    }

    #[test]
    fn test_log2_fractional() {
        assert!((log2(0.5).unwrap() - (-1.0)).abs() < EPSILON);
    }

    #[test]
    fn test_log2_domain_error() {
        assert!(log2(0.0).is_err());
        assert!(log2(-1.0).is_err());
    }

    // =========================================================================
    // log10() Tests
    // =========================================================================

    #[test]
    fn test_log10_one() {
        assert!(log10(1.0).unwrap().abs() < EPSILON);
    }

    #[test]
    fn test_log10_powers() {
        assert!((log10(10.0).unwrap() - 1.0).abs() < EPSILON);
        assert!((log10(100.0).unwrap() - 2.0).abs() < EPSILON);
        assert!((log10(1000.0).unwrap() - 3.0).abs() < EPSILON);
    }

    #[test]
    fn test_log10_fractional() {
        assert!((log10(0.1).unwrap() - (-1.0)).abs() < EPSILON);
    }

    #[test]
    fn test_log10_domain_error() {
        assert!(log10(0.0).is_err());
        assert!(log10(-1.0).is_err());
    }

    // =========================================================================
    // log1p() Tests
    // =========================================================================

    #[test]
    fn test_log1p_zero() {
        assert!(log1p(0.0).unwrap().abs() < EPSILON);
    }

    #[test]
    fn test_log1p_small() {
        // For small x, log1p(x) ≈ x
        let x = 1e-10;
        assert!((log1p(x).unwrap() - x).abs() < 1e-15);
    }

    #[test]
    fn test_log1p_e_minus_one() {
        let x = std::f64::consts::E - 1.0;
        assert!((log1p(x).unwrap() - 1.0).abs() < EPSILON);
    }

    #[test]
    fn test_log1p_domain_error() {
        assert!(log1p(-1.0).is_err());
        assert!(log1p(-2.0).is_err());
    }

    // =========================================================================
    // Identity Tests
    // =========================================================================

    #[test]
    fn test_exp_log_roundtrip() {
        for x in [0.1, 1.0, 2.0, 10.0, 100.0] {
            let result = log(exp(x)).unwrap();
            assert!(
                (result - x).abs() < EPSILON,
                "log(exp(x)) ≠ x for x = {}",
                x
            );
        }
    }

    #[test]
    fn test_log_exp_roundtrip() {
        for x in [0.1, 1.0, 2.0, 5.0] {
            let result = exp(log(x).unwrap());
            assert!(
                (result - x).abs() < EPSILON,
                "exp(log(x)) ≠ x for x = {}",
                x
            );
        }
    }

    #[test]
    fn test_exp2_log2_roundtrip() {
        for x in [0.1, 1.0, 2.0, 10.0] {
            let result = log2(exp2(x)).unwrap();
            assert!(
                (result - x).abs() < EPSILON,
                "log2(exp2(x)) ≠ x for x = {}",
                x
            );
        }
    }

    #[test]
    fn test_expm1_log1p_roundtrip() {
        for x in [-0.5, 0.0, 0.1, 1.0, 5.0] {
            let result = log1p(expm1(x)).unwrap();
            assert!(
                (result - x).abs() < EPSILON,
                "log1p(expm1(x)) ≠ x for x = {}",
                x
            );
        }
    }
}
