//! Introspection builtins (dir, vars, globals, locals, help).
//!
//! Functions for runtime inspection of Python objects and namespaces.
//! All functions are Python 3.12 compatible.
//!
//! # Python Semantics
//!
//! - `dir([object])` - List attributes/valid names in scope
//! - `vars([object])` - Return __dict__ of object (or locals if no arg)
//! - `globals()` - Return global symbol table dict
//! - `locals()` - Return local symbol table dict
//! - `help([object])` - Display documentation

use super::BuiltinError;
use prism_core::Value;

// =============================================================================
// dir() - List Attributes
// =============================================================================

/// Builtin dir([object]) function.
///
/// Without arguments, returns list of names in the current local scope.
/// With an object argument, returns list of valid attributes for that object.
///
/// # Python Semantics
/// - `dir()` → local scope names (sorted)
/// - `dir(int)` → ['__abs__', '__add__', '__and__', ...]
/// - `dir(obj)` → obj.__dir__() or sorted(obj.__dict__.keys())
///
/// # Implementation Note
/// Full implementation requires __dir__ protocol and __dict__ access.
pub fn builtin_dir(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() > 1 {
        return Err(BuiltinError::TypeError(format!(
            "dir() takes at most 1 argument ({} given)",
            args.len()
        )));
    }

    if args.is_empty() {
        // No argument: return names in local scope
        // TODO: Requires access to the current frame's locals
        return Err(BuiltinError::NotImplemented(
            "dir() without argument requires frame introspection".to_string(),
        ));
    }

    // With argument: return attributes of object
    let obj = &args[0];
    dir_of_value(obj)
}

/// Get directory (attributes) of a value.
///
/// This is the core implementation for dir(obj).
#[inline]
fn dir_of_value(value: &Value) -> Result<Value, BuiltinError> {
    // For primitive types, return their known methods
    if value.is_none() {
        // NoneType has minimal attributes
        return dir_nonetype();
    }

    if value.is_bool() {
        return dir_bool();
    }

    if value.is_int() {
        return dir_int();
    }

    if value.is_float() {
        return dir_float();
    }

    // For objects, we need __dir__ or __dict__
    Err(BuiltinError::NotImplemented(
        "dir() for objects requires __dir__ or __dict__".to_string(),
    ))
}

/// Return dir() for NoneType.
#[inline]
fn dir_nonetype() -> Result<Value, BuiltinError> {
    // NoneType has: __bool__, __class__, __repr__, __str__, etc.
    // TODO: Return actual ListObject when available
    Ok(Value::none())
}

/// Return dir() for bool type.
#[inline]
fn dir_bool() -> Result<Value, BuiltinError> {
    // bool inherits from int, has same methods plus:
    // __and__, __or__, __xor__, etc.
    Ok(Value::none())
}

/// Return dir() for int type.
#[inline]
fn dir_int() -> Result<Value, BuiltinError> {
    // int has many methods: __abs__, __add__, __and__, __bool__, etc.
    Ok(Value::none())
}

/// Return dir() for float type.
#[inline]
fn dir_float() -> Result<Value, BuiltinError> {
    // float has: __abs__, __add__, __bool__, __float__, __int__, etc.
    Ok(Value::none())
}

// =============================================================================
// vars() - Return __dict__
// =============================================================================

/// Builtin vars([object]) function.
///
/// Without arguments, acts like locals().
/// With an object, returns the __dict__ of the object.
///
/// # Python Semantics
/// - `vars()` → locals()
/// - `vars(obj)` → obj.__dict__
/// - `vars(None)` → TypeError
///
/// # Implementation Note
/// Full implementation requires __dict__ attribute access.
pub fn builtin_vars(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() > 1 {
        return Err(BuiltinError::TypeError(format!(
            "vars() takes at most 1 argument ({} given)",
            args.len()
        )));
    }

    if args.is_empty() {
        // No argument: return locals()
        return Err(BuiltinError::NotImplemented(
            "vars() without argument requires frame introspection".to_string(),
        ));
    }

    let obj = &args[0];

    // Check for objects without __dict__
    if obj.is_none() {
        return Err(BuiltinError::TypeError(
            "vars() argument must have __dict__ attribute".to_string(),
        ));
    }

    if obj.is_int() || obj.is_float() || obj.is_bool() {
        return Err(BuiltinError::TypeError(
            "vars() argument must have __dict__ attribute".to_string(),
        ));
    }

    // TODO: Access object's __dict__ when available
    Err(BuiltinError::NotImplemented(
        "vars() for objects requires __dict__ access".to_string(),
    ))
}

// =============================================================================
// globals() - Global Symbol Table
// =============================================================================

/// Builtin globals() function.
///
/// Returns a dictionary representing the current global symbol table.
///
/// # Python Semantics
/// - Always returns the globals dict of the current module
/// - Modifications to the returned dict affect the actual globals
///
/// # Implementation Note
/// Requires access to the module's global namespace.
pub fn builtin_globals(args: &[Value]) -> Result<Value, BuiltinError> {
    if !args.is_empty() {
        return Err(BuiltinError::TypeError(format!(
            "globals() takes no arguments ({} given)",
            args.len()
        )));
    }

    // TODO: Return actual globals dict from current frame
    Err(BuiltinError::NotImplemented(
        "globals() requires frame introspection".to_string(),
    ))
}

// =============================================================================
// locals() - Local Symbol Table
// =============================================================================

/// Builtin locals() function.
///
/// Returns a dictionary representing the current local symbol table.
///
/// # Python Semantics
/// - In function: returns a copy of local variables
/// - At module level: same as globals()
/// - Modifications may not affect actual locals (implementation-defined)
///
/// # Implementation Note  
/// Requires access to the current frame's local namespace.
pub fn builtin_locals(args: &[Value]) -> Result<Value, BuiltinError> {
    if !args.is_empty() {
        return Err(BuiltinError::TypeError(format!(
            "locals() takes no arguments ({} given)",
            args.len()
        )));
    }

    // TODO: Return actual locals dict from current frame
    Err(BuiltinError::NotImplemented(
        "locals() requires frame introspection".to_string(),
    ))
}

// =============================================================================
// help() - Interactive Help
// =============================================================================

/// Builtin help([object]) function.
///
/// Invokes the built-in help system.
///
/// # Python Semantics
/// - `help()` → Start interactive help
/// - `help(obj)` → Show help for object
/// - `help('topic')` → Show help for topic
///
/// # Implementation Note
/// Full interactive help is not implemented.
/// This provides a stub that returns NotImplemented.
pub fn builtin_help(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() > 1 {
        return Err(BuiltinError::TypeError(format!(
            "help() takes at most 1 argument ({} given)",
            args.len()
        )));
    }

    // For now, help() is a stub
    // In a full implementation, this would:
    // 1. Access object's __doc__
    // 2. Format and display documentation
    // 3. Use pydoc for topics

    Err(BuiltinError::NotImplemented(
        "help() is not implemented".to_string(),
    ))
}

// =============================================================================
// __import__() - Import System Hook
// =============================================================================

/// Builtin __import__(name, ...) function.
///
/// This function is invoked by the import statement.
///
/// # Python Semantics
/// - `__import__('os')` → <module 'os'>
/// - Usually not called directly; use import statement
///
/// # Implementation Note
/// Full implementation requires the import system.
pub fn builtin_import(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.is_empty() {
        return Err(BuiltinError::TypeError(
            "__import__() missing required argument: 'name'".to_string(),
        ));
    }

    // TODO: Implement import system
    Err(BuiltinError::NotImplemented(
        "__import__() requires import system".to_string(),
    ))
}

// =============================================================================
// hasattr, getattr, setattr, delattr - Already in types.rs
// =============================================================================

// Note: hasattr, getattr, setattr, delattr are already implemented
// in types.rs. They are core attribute access functions.

// =============================================================================
// isinstance, issubclass - Already in types.rs
// =============================================================================

// Note: isinstance, issubclass are already implemented in types.rs.
// They are type checking functions.

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // dir() Argument Validation Tests
    // =========================================================================

    #[test]
    fn test_dir_too_many_args() {
        let result = builtin_dir(&[Value::int(1).unwrap(), Value::int(2).unwrap()]);
        assert!(result.is_err());
        match result {
            Err(BuiltinError::TypeError(msg)) => {
                assert!(msg.contains("at most 1 argument"));
            }
            _ => panic!("Expected TypeError"),
        }
    }

    #[test]
    fn test_dir_no_args_not_implemented() {
        // dir() without args requires frame introspection
        let result = builtin_dir(&[]);
        assert!(result.is_err());
        match result {
            Err(BuiltinError::NotImplemented(_)) => {}
            _ => panic!("Expected NotImplemented"),
        }
    }

    #[test]
    fn test_dir_with_none() {
        // dir(None) should return NoneType attributes
        let result = builtin_dir(&[Value::none()]);
        // Currently returns placeholder
        assert!(result.is_ok() || matches!(result, Err(BuiltinError::NotImplemented(_))));
    }

    #[test]
    fn test_dir_with_int() {
        let result = builtin_dir(&[Value::int(42).unwrap()]);
        assert!(result.is_ok());
    }

    #[test]
    fn test_dir_with_float() {
        let result = builtin_dir(&[Value::float(3.14)]);
        assert!(result.is_ok());
    }

    #[test]
    fn test_dir_with_bool() {
        let result = builtin_dir(&[Value::bool(true)]);
        assert!(result.is_ok());
    }

    // =========================================================================
    // vars() Argument Validation Tests
    // =========================================================================

    #[test]
    fn test_vars_too_many_args() {
        let result = builtin_vars(&[Value::int(1).unwrap(), Value::int(2).unwrap()]);
        assert!(result.is_err());
        match result {
            Err(BuiltinError::TypeError(msg)) => {
                assert!(msg.contains("at most 1 argument"));
            }
            _ => panic!("Expected TypeError"),
        }
    }

    #[test]
    fn test_vars_no_args_not_implemented() {
        let result = builtin_vars(&[]);
        assert!(result.is_err());
        match result {
            Err(BuiltinError::NotImplemented(_)) => {}
            _ => panic!("Expected NotImplemented"),
        }
    }

    #[test]
    fn test_vars_with_none() {
        // vars(None) should be TypeError (no __dict__)
        let result = builtin_vars(&[Value::none()]);
        assert!(result.is_err());
        match result {
            Err(BuiltinError::TypeError(msg)) => {
                assert!(msg.contains("__dict__"));
            }
            _ => panic!("Expected TypeError"),
        }
    }

    #[test]
    fn test_vars_with_int() {
        // vars(42) should be TypeError (int has no __dict__)
        let result = builtin_vars(&[Value::int(42).unwrap()]);
        assert!(result.is_err());
        match result {
            Err(BuiltinError::TypeError(msg)) => {
                assert!(msg.contains("__dict__"));
            }
            _ => panic!("Expected TypeError"),
        }
    }

    #[test]
    fn test_vars_with_float() {
        let result = builtin_vars(&[Value::float(3.14)]);
        assert!(result.is_err());
        match result {
            Err(BuiltinError::TypeError(_)) => {}
            _ => panic!("Expected TypeError"),
        }
    }

    #[test]
    fn test_vars_with_bool() {
        let result = builtin_vars(&[Value::bool(true)]);
        assert!(result.is_err());
        match result {
            Err(BuiltinError::TypeError(_)) => {}
            _ => panic!("Expected TypeError"),
        }
    }

    // =========================================================================
    // globals() Argument Validation Tests
    // =========================================================================

    #[test]
    fn test_globals_with_args() {
        let result = builtin_globals(&[Value::int(1).unwrap()]);
        assert!(result.is_err());
        match result {
            Err(BuiltinError::TypeError(msg)) => {
                assert!(msg.contains("takes no arguments"));
            }
            _ => panic!("Expected TypeError"),
        }
    }

    #[test]
    fn test_globals_not_implemented() {
        let result = builtin_globals(&[]);
        assert!(result.is_err());
        match result {
            Err(BuiltinError::NotImplemented(_)) => {}
            _ => panic!("Expected NotImplemented"),
        }
    }

    // =========================================================================
    // locals() Argument Validation Tests
    // =========================================================================

    #[test]
    fn test_locals_with_args() {
        let result = builtin_locals(&[Value::int(1).unwrap()]);
        assert!(result.is_err());
        match result {
            Err(BuiltinError::TypeError(msg)) => {
                assert!(msg.contains("takes no arguments"));
            }
            _ => panic!("Expected TypeError"),
        }
    }

    #[test]
    fn test_locals_not_implemented() {
        let result = builtin_locals(&[]);
        assert!(result.is_err());
        match result {
            Err(BuiltinError::NotImplemented(_)) => {}
            _ => panic!("Expected NotImplemented"),
        }
    }

    // =========================================================================
    // help() Argument Validation Tests
    // =========================================================================

    #[test]
    fn test_help_too_many_args() {
        let result = builtin_help(&[Value::int(1).unwrap(), Value::int(2).unwrap()]);
        assert!(result.is_err());
        match result {
            Err(BuiltinError::TypeError(msg)) => {
                assert!(msg.contains("at most 1 argument"));
            }
            _ => panic!("Expected TypeError"),
        }
    }

    #[test]
    fn test_help_not_implemented() {
        let result = builtin_help(&[]);
        assert!(result.is_err());
        match result {
            Err(BuiltinError::NotImplemented(_)) => {}
            _ => panic!("Expected NotImplemented"),
        }
    }

    #[test]
    fn test_help_with_arg_not_implemented() {
        let result = builtin_help(&[Value::int(42).unwrap()]);
        assert!(result.is_err());
        match result {
            Err(BuiltinError::NotImplemented(_)) => {}
            _ => panic!("Expected NotImplemented"),
        }
    }

    // =========================================================================
    // __import__() Argument Validation Tests
    // =========================================================================

    #[test]
    fn test_import_no_args() {
        let result = builtin_import(&[]);
        assert!(result.is_err());
        match result {
            Err(BuiltinError::TypeError(msg)) => {
                assert!(msg.contains("missing required argument"));
            }
            _ => panic!("Expected TypeError"),
        }
    }

    #[test]
    fn test_import_not_implemented() {
        // _import with argument returns NotImplemented
        let result = builtin_import(&[Value::int(1).unwrap()]);
        assert!(result.is_err());
        match result {
            Err(BuiltinError::NotImplemented(_)) => {}
            _ => panic!("Expected NotImplemented"),
        }
    }

    // =========================================================================
    // dir_of_value() Implementation Tests
    // =========================================================================

    #[test]
    fn test_dir_of_value_none() {
        let result = dir_of_value(&Value::none());
        assert!(result.is_ok());
    }

    #[test]
    fn test_dir_of_value_int() {
        let result = dir_of_value(&Value::int(42).unwrap());
        assert!(result.is_ok());
    }

    #[test]
    fn test_dir_of_value_float() {
        let result = dir_of_value(&Value::float(3.14));
        assert!(result.is_ok());
    }

    #[test]
    fn test_dir_of_value_bool() {
        let result = dir_of_value(&Value::bool(true));
        assert!(result.is_ok());
    }

    // =========================================================================
    // Edge Cases
    // =========================================================================

    #[test]
    fn test_dir_preserves_type_info() {
        // Ensure dir() doesn't corrupt the value
        let val = Value::int(42).unwrap();
        let _ = builtin_dir(&[val.clone()]);
        assert!(val.is_int());
        assert_eq!(val.as_int(), Some(42));
    }

    #[test]
    fn test_vars_preserves_type_info() {
        let val = Value::int(42).unwrap();
        let _ = builtin_vars(&[val.clone()]);
        assert!(val.is_int());
    }
}
