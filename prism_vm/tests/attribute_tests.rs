//! Comprehensive tests for attribute mutation operations.
//!
//! Tests for GetAttr, SetAttr, DelAttr opcodes and getattr/setattr/hasattr/delattr builtins.
//!
//! Coverage:
//! - Builtin function semantics (argument validation, type errors)
//! - Error handling for non-object types

use prism_core::Value;

// =============================================================================
// Builtin Function Tests
// =============================================================================

mod builtin_tests {
    use super::*;
    use prism_vm::builtins::{
        BuiltinError, builtin_delattr, builtin_getattr, builtin_hasattr, builtin_setattr,
    };

    // =========================================================================
    // getattr() Argument Validation
    // =========================================================================

    #[test]
    fn test_getattr_too_few_args() {
        let result = builtin_getattr(&[Value::none()]);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(matches!(err, BuiltinError::TypeError(_)));
    }

    #[test]
    fn test_getattr_too_many_args() {
        let result = builtin_getattr(&[Value::none(), Value::none(), Value::none(), Value::none()]);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(matches!(err, BuiltinError::TypeError(_)));
    }

    #[test]
    fn test_getattr_non_string_name() {
        // Name must be a string - passing an int should fail with TypeError
        let result = builtin_getattr(&[Value::none(), Value::int(42).unwrap()]);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(matches!(err, BuiltinError::TypeError(_)));
    }

    #[test]
    fn test_getattr_non_string_name_with_default() {
        // Even with a default, non-string name should fail
        let result = builtin_getattr(&[Value::none(), Value::int(42).unwrap(), Value::bool(true)]);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(matches!(err, BuiltinError::TypeError(_)));
    }

    // =========================================================================
    // setattr() Argument Validation
    // =========================================================================

    #[test]
    fn test_setattr_too_few_args() {
        let result = builtin_setattr(&[Value::none(), Value::none()]);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(matches!(err, BuiltinError::TypeError(_)));
    }

    #[test]
    fn test_setattr_too_few_single_arg() {
        let result = builtin_setattr(&[Value::none()]);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(matches!(err, BuiltinError::TypeError(_)));
    }

    #[test]
    fn test_setattr_too_many_args() {
        let result = builtin_setattr(&[Value::none(), Value::none(), Value::none(), Value::none()]);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(matches!(err, BuiltinError::TypeError(_)));
    }

    #[test]
    fn test_setattr_non_string_name() {
        // Name must be a string
        let result = builtin_setattr(&[Value::none(), Value::int(42).unwrap(), Value::bool(true)]);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(matches!(err, BuiltinError::TypeError(_)));
    }

    // =========================================================================
    // hasattr() Argument Validation
    // =========================================================================

    #[test]
    fn test_hasattr_too_few_args() {
        let result = builtin_hasattr(&[Value::none()]);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(matches!(err, BuiltinError::TypeError(_)));
    }

    #[test]
    fn test_hasattr_no_args() {
        let result = builtin_hasattr(&[]);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(matches!(err, BuiltinError::TypeError(_)));
    }

    #[test]
    fn test_hasattr_too_many_args() {
        let result = builtin_hasattr(&[Value::none(), Value::none(), Value::none()]);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(matches!(err, BuiltinError::TypeError(_)));
    }

    #[test]
    fn test_hasattr_non_string_name() {
        // Name must be a string
        let result = builtin_hasattr(&[Value::none(), Value::int(42).unwrap()]);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(matches!(err, BuiltinError::TypeError(_)));
    }

    // =========================================================================
    // delattr() Argument Validation
    // =========================================================================

    #[test]
    fn test_delattr_too_few_args() {
        let result = builtin_delattr(&[Value::none()]);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(matches!(err, BuiltinError::TypeError(_)));
    }

    #[test]
    fn test_delattr_no_args() {
        let result = builtin_delattr(&[]);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(matches!(err, BuiltinError::TypeError(_)));
    }

    #[test]
    fn test_delattr_too_many_args() {
        let result = builtin_delattr(&[Value::none(), Value::none(), Value::none()]);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(matches!(err, BuiltinError::TypeError(_)));
    }

    #[test]
    fn test_delattr_non_string_name() {
        // Name must be a string
        let result = builtin_delattr(&[Value::none(), Value::int(42).unwrap()]);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(matches!(err, BuiltinError::TypeError(_)));
    }

    // =========================================================================
    // Cross-Type Error Scenarios
    // =========================================================================

    #[test]
    fn test_getattr_bool_name_fails() {
        let result = builtin_getattr(&[Value::int(100).unwrap(), Value::bool(true)]);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(matches!(err, BuiltinError::TypeError(_)));
    }

    #[test]
    fn test_setattr_float_name_fails() {
        let result =
            builtin_setattr(&[Value::int(100).unwrap(), Value::float(3.14), Value::none()]);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(matches!(err, BuiltinError::TypeError(_)));
    }

    #[test]
    fn test_hasattr_none_name_fails() {
        // None is not a valid attribute name
        let result = builtin_hasattr(&[Value::int(100).unwrap(), Value::none()]);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(matches!(err, BuiltinError::TypeError(_)));
    }

    #[test]
    fn test_delattr_none_name_fails() {
        // None is not a valid attribute name
        let result = builtin_delattr(&[Value::int(100).unwrap(), Value::none()]);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(matches!(err, BuiltinError::TypeError(_)));
    }
}

// =============================================================================
// Additional Integration Test Notes
// =============================================================================
//
// Full integration tests with ShapedObject require:
// 1. Creating StringObject values for attribute names
// 2. Creating ShapedObject values as test targets
//
// These would be best tested via the VM ops directly in the ops module tests,
// or via end-to-end Python execution tests.
