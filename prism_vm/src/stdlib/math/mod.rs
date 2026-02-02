//! Python `math` module implementation.
//!
//! Provides mathematical functions with maximum performance through:
//! - Direct hardware intrinsics (sin, cos, exp → SSE/x87 instructions)
//! - Branch-free classification (isinf, isnan via bit patterns)
//! - Const lookup tables (factorial 0-20)
//! - Binary algorithms (GCD, integer pow)
//!
//! # Performance Characteristics
//!
//! - Zero heap allocations for all operations
//! - All functions are `#[inline]` for call elimination
//! - Special value handling without branching where possible
//!
//! # Python 3.12 Compatibility
//!
//! All functions match Python 3.12 semantics including:
//! - Domain error handling (ValueError for invalid inputs)
//! - Special value propagation (inf, nan)
//! - Exact output format matching

mod angular;
mod basic;
mod classify;
mod combinatorics;
mod constants;
mod exp_log;
mod hyperbolic;
mod power;
mod special;
mod trig;

pub use angular::*;
pub use basic::*;
pub use classify::*;
pub use combinatorics::*;
pub use constants::*;
pub use exp_log::*;
pub use hyperbolic::*;
pub use power::*;
pub use special::*;
pub use trig::*;

use super::{Module, ModuleError, ModuleResult};
use prism_core::Value;
use std::sync::Arc;

/// The math module instance.
pub struct MathModule;

impl MathModule {
    /// Create a new math module.
    #[inline]
    pub const fn new() -> Self {
        Self
    }
}

impl Default for MathModule {
    fn default() -> Self {
        Self::new()
    }
}

impl Module for MathModule {
    fn name(&self) -> &str {
        "math"
    }

    fn get_attr(&self, name: &str) -> ModuleResult {
        match name {
            // Constants
            "pi" => Ok(Value::float(PI)),
            "e" => Ok(Value::float(E)),
            "tau" => Ok(Value::float(TAU)),
            "inf" => Ok(Value::float(INFINITY)),
            "nan" => Ok(Value::float(NAN)),

            // Functions are returned as None for now
            // Full implementation would return callable objects
            "ceil" | "floor" | "trunc" | "fabs" | "copysign" | "fmod" | "modf" | "remainder"
            | "sin" | "cos" | "tan" | "asin" | "acos" | "atan" | "atan2" | "sinh" | "cosh"
            | "tanh" | "asinh" | "acosh" | "atanh" | "exp" | "exp2" | "expm1" | "log" | "log2"
            | "log10" | "log1p" | "pow" | "sqrt" | "isqrt" | "hypot" | "factorial" | "gamma"
            | "lgamma" | "erf" | "erfc" | "comb" | "perm" | "gcd" | "lcm" | "degrees"
            | "radians" | "isinf" | "isnan" | "isfinite" => {
                // TODO: Return actual function objects when callable system is ready
                Err(ModuleError::AttributeError(format!(
                    "math.{} is not yet callable as an object",
                    name
                )))
            }

            _ => Err(ModuleError::AttributeError(format!(
                "module 'math' has no attribute '{}'",
                name
            ))),
        }
    }

    fn dir(&self) -> Vec<Arc<str>> {
        vec![
            // Constants
            Arc::from("pi"),
            Arc::from("e"),
            Arc::from("tau"),
            Arc::from("inf"),
            Arc::from("nan"),
            // Basic
            Arc::from("ceil"),
            Arc::from("floor"),
            Arc::from("trunc"),
            Arc::from("fabs"),
            Arc::from("copysign"),
            Arc::from("fmod"),
            Arc::from("modf"),
            Arc::from("remainder"),
            // Trig
            Arc::from("sin"),
            Arc::from("cos"),
            Arc::from("tan"),
            Arc::from("asin"),
            Arc::from("acos"),
            Arc::from("atan"),
            Arc::from("atan2"),
            // Hyperbolic
            Arc::from("sinh"),
            Arc::from("cosh"),
            Arc::from("tanh"),
            Arc::from("asinh"),
            Arc::from("acosh"),
            Arc::from("atanh"),
            // Exp/Log
            Arc::from("exp"),
            Arc::from("exp2"),
            Arc::from("expm1"),
            Arc::from("log"),
            Arc::from("log2"),
            Arc::from("log10"),
            Arc::from("log1p"),
            // Power
            Arc::from("pow"),
            Arc::from("sqrt"),
            Arc::from("isqrt"),
            Arc::from("hypot"),
            // Special
            Arc::from("factorial"),
            Arc::from("gamma"),
            Arc::from("lgamma"),
            Arc::from("erf"),
            Arc::from("erfc"),
            // Combinatorics
            Arc::from("comb"),
            Arc::from("perm"),
            Arc::from("gcd"),
            Arc::from("lcm"),
            // Angular
            Arc::from("degrees"),
            Arc::from("radians"),
            // Classification
            Arc::from("isinf"),
            Arc::from("isnan"),
            Arc::from("isfinite"),
        ]
    }
}

// =============================================================================
// Builtin Function Wrappers
// =============================================================================

/// Helper to extract a float from Value.
#[inline]
pub fn extract_float(value: &Value) -> Result<f64, ModuleError> {
    if let Some(f) = value.as_float() {
        Ok(f)
    } else if let Some(i) = value.as_int() {
        Ok(i as f64)
    } else if value.is_bool() {
        Ok(if value.as_bool().unwrap_or(false) {
            1.0
        } else {
            0.0
        })
    } else {
        Err(ModuleError::TypeError(
            "must be real number, not 'NoneType'".to_string(),
        ))
    }
}

/// Helper to extract an integer from Value.
#[inline]
pub fn extract_int(value: &Value) -> Result<i64, ModuleError> {
    if let Some(i) = value.as_int() {
        Ok(i)
    } else if value.is_bool() {
        Ok(if value.as_bool().unwrap_or(false) {
            1
        } else {
            0
        })
    } else if let Some(_f) = value.as_float() {
        Err(ModuleError::TypeError(format!(
            "'float' object cannot be interpreted as an integer"
        )))
    } else {
        Err(ModuleError::TypeError(
            "'NoneType' object cannot be interpreted as an integer".to_string(),
        ))
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_module_name() {
        let m = MathModule::new();
        assert_eq!(m.name(), "math");
    }

    #[test]
    fn test_get_pi() {
        let m = MathModule::new();
        let pi = m.get_attr("pi").unwrap();
        assert!(pi.is_float());
        assert!((pi.as_float().unwrap() - std::f64::consts::PI).abs() < 1e-15);
    }

    #[test]
    fn test_get_e() {
        let m = MathModule::new();
        let e = m.get_attr("e").unwrap();
        assert!(e.is_float());
        assert!((e.as_float().unwrap() - std::f64::consts::E).abs() < 1e-15);
    }

    #[test]
    fn test_get_tau() {
        let m = MathModule::new();
        let tau = m.get_attr("tau").unwrap();
        assert!(tau.is_float());
        assert!((tau.as_float().unwrap() - std::f64::consts::TAU).abs() < 1e-15);
    }

    #[test]
    fn test_get_inf() {
        let m = MathModule::new();
        let inf = m.get_attr("inf").unwrap();
        assert!(inf.is_float());
        assert!(inf.as_float().unwrap().is_infinite());
        assert!(inf.as_float().unwrap().is_sign_positive());
    }

    #[test]
    fn test_get_nan() {
        let m = MathModule::new();
        let nan = m.get_attr("nan").unwrap();
        assert!(nan.is_float());
        assert!(nan.as_float().unwrap().is_nan());
    }

    #[test]
    fn test_unknown_attr() {
        let m = MathModule::new();
        let result = m.get_attr("nonexistent");
        assert!(result.is_err());
        match result {
            Err(ModuleError::AttributeError(msg)) => {
                assert!(msg.contains("no attribute 'nonexistent'"));
            }
            _ => panic!("Expected AttributeError"),
        }
    }

    #[test]
    fn test_dir() {
        let m = MathModule::new();
        let attrs = m.dir();
        assert!(attrs.contains(&Arc::from("pi")));
        assert!(attrs.contains(&Arc::from("sin")));
        assert!(attrs.contains(&Arc::from("sqrt")));
        assert!(attrs.contains(&Arc::from("factorial")));
        assert!(attrs.len() >= 40); // We have 42+ functions
    }

    #[test]
    fn test_extract_float_from_float() {
        let v = Value::float(3.14);
        assert!((extract_float(&v).unwrap() - 3.14).abs() < 1e-15);
    }

    #[test]
    fn test_extract_float_from_int() {
        let v = Value::int(42).unwrap();
        assert!((extract_float(&v).unwrap() - 42.0).abs() < 1e-15);
    }

    #[test]
    fn test_extract_float_from_bool() {
        let t = Value::bool(true);
        let f = Value::bool(false);
        assert!((extract_float(&t).unwrap() - 1.0).abs() < 1e-15);
        assert!((extract_float(&f).unwrap()).abs() < 1e-15);
    }

    #[test]
    fn test_extract_float_from_none() {
        let v = Value::none();
        assert!(extract_float(&v).is_err());
    }

    #[test]
    fn test_extract_int_from_int() {
        let v = Value::int(42).unwrap();
        assert_eq!(extract_int(&v).unwrap(), 42);
    }

    #[test]
    fn test_extract_int_from_bool() {
        let t = Value::bool(true);
        let f = Value::bool(false);
        assert_eq!(extract_int(&t).unwrap(), 1);
        assert_eq!(extract_int(&f).unwrap(), 0);
    }

    #[test]
    fn test_extract_int_from_float_fails() {
        let v = Value::float(3.14);
        assert!(extract_int(&v).is_err());
    }
}
