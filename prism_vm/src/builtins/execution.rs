//! Execution builtins (exec, eval, compile).
//!
//! Functions for dynamic code execution and compilation.
//! All functions are Python 3.12 compatible.
//!
//! # Python Semantics
//!
//! - `exec(code, globals, locals)` - Execute code in given namespaces
//! - `eval(expression, globals, locals)` - Evaluate expression
//! - `compile(source, filename, mode)` - Compile source to code object
//!
//! # Security Note
//!
//! These functions execute arbitrary code. In production, they should
//! be used with caution and potentially sandboxed.

use super::BuiltinError;
use prism_core::Value;

// =============================================================================
// exec() - Execute Python Code
// =============================================================================

/// Builtin exec(code[, globals[, locals]]) function.
///
/// Execute the code in the optional globals and locals namespaces.
///
/// # Python Semantics
/// - `exec('x = 1')` → executes in current namespace
/// - `exec('x = 1', globals())` → executes in global namespace
/// - `exec('x = 1', globals(), locals())` → executes in local namespace
/// - Returns None
///
/// # Code Argument Types
/// - String: source code to compile and execute
/// - Code object: pre-compiled code to execute
///
/// # Implementation Note
/// Full implementation requires the compiler and frame system.
pub fn builtin_exec(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.is_empty() {
        return Err(BuiltinError::TypeError(
            "exec() missing required argument: 'source'".to_string(),
        ));
    }
    if args.len() > 3 {
        return Err(BuiltinError::TypeError(format!(
            "exec() takes at most 3 arguments ({} given)",
            args.len()
        )));
    }

    let source = &args[0];
    let _globals = args.get(1);
    let _locals = args.get(2);

    // Validate source type
    if source.is_none() {
        return Err(BuiltinError::TypeError(
            "exec() arg 1 must be a string, bytes or code object".to_string(),
        ));
    }

    // For primitive types, we can't execute them
    if source.is_int() || source.is_float() || source.is_bool() {
        return Err(BuiltinError::TypeError(
            "exec() arg 1 must be a string, bytes or code object".to_string(),
        ));
    }

    // TODO: Implement actual execution
    // 1. If source is a string, compile it with mode='exec'
    // 2. If source is a code object, use it directly
    // 3. Execute the code in the given namespaces
    Err(BuiltinError::NotImplemented(
        "exec() requires compiler integration".to_string(),
    ))
}

// =============================================================================
// eval() - Evaluate Python Expression
// =============================================================================

/// Builtin eval(expression[, globals[, locals]]) function.
///
/// Evaluate a Python expression and return the result.
///
/// # Python Semantics
/// - `eval('1 + 2')` → 3
/// - `eval('x', {'x': 10})` → 10
/// - Returns the result of the expression
///
/// # Expression Argument Types
/// - String: source code to compile and evaluate
/// - Code object: pre-compiled code to evaluate
///
/// # Implementation Note
/// Full implementation requires the compiler and frame system.
pub fn builtin_eval(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.is_empty() {
        return Err(BuiltinError::TypeError(
            "eval() missing required argument: 'source'".to_string(),
        ));
    }
    if args.len() > 3 {
        return Err(BuiltinError::TypeError(format!(
            "eval() takes at most 3 arguments ({} given)",
            args.len()
        )));
    }

    let source = &args[0];
    let _globals = args.get(1);
    let _locals = args.get(2);

    // Validate source type
    if source.is_none() {
        return Err(BuiltinError::TypeError(
            "eval() arg 1 must be a string, bytes or code object".to_string(),
        ));
    }

    if source.is_int() || source.is_float() || source.is_bool() {
        return Err(BuiltinError::TypeError(
            "eval() arg 1 must be a string, bytes or code object".to_string(),
        ));
    }

    // TODO: Implement actual evaluation
    // 1. If source is a string, compile it with mode='eval'
    // 2. If source is a code object, use it directly
    // 3. Evaluate the code and return the result
    Err(BuiltinError::NotImplemented(
        "eval() requires compiler integration".to_string(),
    ))
}

// =============================================================================
// compile() - Compile Source to Code Object
// =============================================================================

/// Compile mode for Python code.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompileMode {
    /// Compile as a module (sequence of statements)
    Exec,
    /// Compile as a single expression
    Eval,
    /// Compile as a single interactive statement
    Single,
}

impl CompileMode {
    /// Parse mode from string.
    pub fn from_str(s: &str) -> Option<Self> {
        match s {
            "exec" => Some(CompileMode::Exec),
            "eval" => Some(CompileMode::Eval),
            "single" => Some(CompileMode::Single),
            _ => None,
        }
    }
}

/// Builtin compile(source, filename, mode, ...) function.
///
/// Compile source code into a code object.
///
/// # Python Semantics
/// - `compile('x+1', '<string>', 'eval')` → code object
/// - `compile('x=1', '<string>', 'exec')` → code object
/// - `compile('x=1', '<string>', 'single')` → code object
///
/// # Arguments
/// - source: String, bytes, or AST object
/// - filename: Name for error messages
/// - mode: 'exec', 'eval', or 'single'
/// - flags: Optional compiler flags (default 0)
/// - dont_inherit: Don't inherit future flags (default False)
/// - optimize: Optimization level (default -1)
///
/// # Implementation Note
/// Full implementation requires the compiler.
pub fn builtin_compile(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() < 3 {
        return Err(BuiltinError::TypeError(format!(
            "compile() takes at least 3 arguments ({} given)",
            args.len()
        )));
    }
    if args.len() > 6 {
        return Err(BuiltinError::TypeError(format!(
            "compile() takes at most 6 arguments ({} given)",
            args.len()
        )));
    }

    let source = &args[0];
    let _filename = &args[1];
    let _mode = &args[2];

    // Validate source type
    if source.is_none() {
        return Err(BuiltinError::TypeError(
            "compile() source must be a string, bytes, or AST object".to_string(),
        ));
    }

    if source.is_int() || source.is_float() || source.is_bool() {
        return Err(BuiltinError::TypeError(
            "compile() source must be a string, bytes, or AST object".to_string(),
        ));
    }

    // TODO: Validate filename is a string
    // TODO: Validate mode is 'exec', 'eval', or 'single'
    // TODO: Validate optional arguments

    // TODO: Implement actual compilation
    // 1. Parse the source code
    // 2. Compile to bytecode
    // 3. Return a code object
    Err(BuiltinError::NotImplemented(
        "compile() requires compiler integration".to_string(),
    ))
}

// =============================================================================
// breakpoint() - Invoke Debugger
// =============================================================================

/// Builtin breakpoint(*args, **kws) function.
///
/// Calls sys.breakpointhook() to enter the debugger.
///
/// # Python Semantics
/// - `breakpoint()` → enters pdb debugger
/// - Can be customized via PYTHONBREAKPOINT env var
///
/// # Implementation Note
/// This is a stub for debugger integration.
pub fn builtin_breakpoint(args: &[Value]) -> Result<Value, BuiltinError> {
    // breakpoint() is typically a no-op if no debugger is attached
    // For now, we just ignore any arguments and return None
    let _ = args;

    // In a full implementation:
    // 1. Check PYTHONBREAKPOINT environment variable
    // 2. If set to '0', do nothing
    // 3. Otherwise, call the configured hook (default: pdb.set_trace)

    Ok(Value::none())
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // exec() Argument Validation Tests
    // =========================================================================

    #[test]
    fn test_exec_no_args() {
        let result = builtin_exec(&[]);
        assert!(result.is_err());
        match result {
            Err(BuiltinError::TypeError(msg)) => {
                assert!(msg.contains("missing required argument"));
            }
            _ => panic!("Expected TypeError"),
        }
    }

    #[test]
    fn test_exec_too_many_args() {
        let result = builtin_exec(&[
            Value::int(1).unwrap(),
            Value::int(2).unwrap(),
            Value::int(3).unwrap(),
            Value::int(4).unwrap(),
        ]);
        assert!(result.is_err());
        match result {
            Err(BuiltinError::TypeError(msg)) => {
                assert!(msg.contains("at most 3 arguments"));
            }
            _ => panic!("Expected TypeError"),
        }
    }

    #[test]
    fn test_exec_with_none() {
        let result = builtin_exec(&[Value::none()]);
        assert!(result.is_err());
        match result {
            Err(BuiltinError::TypeError(msg)) => {
                assert!(msg.contains("string, bytes or code object"));
            }
            _ => panic!("Expected TypeError"),
        }
    }

    #[test]
    fn test_exec_with_int() {
        let result = builtin_exec(&[Value::int(42).unwrap()]);
        assert!(result.is_err());
        match result {
            Err(BuiltinError::TypeError(msg)) => {
                assert!(msg.contains("string, bytes or code object"));
            }
            _ => panic!("Expected TypeError"),
        }
    }

    #[test]
    fn test_exec_with_float() {
        let result = builtin_exec(&[Value::float(3.14)]);
        assert!(result.is_err());
        match result {
            Err(BuiltinError::TypeError(_)) => {}
            _ => panic!("Expected TypeError"),
        }
    }

    #[test]
    fn test_exec_with_bool() {
        let result = builtin_exec(&[Value::bool(true)]);
        assert!(result.is_err());
        match result {
            Err(BuiltinError::TypeError(_)) => {}
            _ => panic!("Expected TypeError"),
        }
    }

    // =========================================================================
    // eval() Argument Validation Tests
    // =========================================================================

    #[test]
    fn test_eval_no_args() {
        let result = builtin_eval(&[]);
        assert!(result.is_err());
        match result {
            Err(BuiltinError::TypeError(msg)) => {
                assert!(msg.contains("missing required argument"));
            }
            _ => panic!("Expected TypeError"),
        }
    }

    #[test]
    fn test_eval_too_many_args() {
        let result = builtin_eval(&[
            Value::int(1).unwrap(),
            Value::int(2).unwrap(),
            Value::int(3).unwrap(),
            Value::int(4).unwrap(),
        ]);
        assert!(result.is_err());
        match result {
            Err(BuiltinError::TypeError(msg)) => {
                assert!(msg.contains("at most 3 arguments"));
            }
            _ => panic!("Expected TypeError"),
        }
    }

    #[test]
    fn test_eval_with_none() {
        let result = builtin_eval(&[Value::none()]);
        assert!(result.is_err());
        match result {
            Err(BuiltinError::TypeError(msg)) => {
                assert!(msg.contains("string, bytes or code object"));
            }
            _ => panic!("Expected TypeError"),
        }
    }

    #[test]
    fn test_eval_with_int() {
        let result = builtin_eval(&[Value::int(42).unwrap()]);
        assert!(result.is_err());
        match result {
            Err(BuiltinError::TypeError(_)) => {}
            _ => panic!("Expected TypeError"),
        }
    }

    #[test]
    fn test_eval_with_float() {
        let result = builtin_eval(&[Value::float(3.14)]);
        assert!(result.is_err());
        match result {
            Err(BuiltinError::TypeError(_)) => {}
            _ => panic!("Expected TypeError"),
        }
    }

    #[test]
    fn test_eval_with_bool() {
        let result = builtin_eval(&[Value::bool(true)]);
        assert!(result.is_err());
        match result {
            Err(BuiltinError::TypeError(_)) => {}
            _ => panic!("Expected TypeError"),
        }
    }

    // =========================================================================
    // compile() Argument Validation Tests
    // =========================================================================

    #[test]
    fn test_compile_too_few_args() {
        let result = builtin_compile(&[Value::int(1).unwrap(), Value::int(2).unwrap()]);
        assert!(result.is_err());
        match result {
            Err(BuiltinError::TypeError(msg)) => {
                assert!(msg.contains("at least 3 arguments"));
            }
            _ => panic!("Expected TypeError"),
        }
    }

    #[test]
    fn test_compile_too_many_args() {
        let result = builtin_compile(&[
            Value::int(1).unwrap(),
            Value::int(2).unwrap(),
            Value::int(3).unwrap(),
            Value::int(4).unwrap(),
            Value::int(5).unwrap(),
            Value::int(6).unwrap(),
            Value::int(7).unwrap(),
        ]);
        assert!(result.is_err());
        match result {
            Err(BuiltinError::TypeError(msg)) => {
                assert!(msg.contains("at most 6 arguments"));
            }
            _ => panic!("Expected TypeError"),
        }
    }

    #[test]
    fn test_compile_with_none_source() {
        let result = builtin_compile(&[
            Value::none(),
            Value::int(0).unwrap(),
            Value::int(0).unwrap(),
        ]);
        assert!(result.is_err());
        match result {
            Err(BuiltinError::TypeError(msg)) => {
                assert!(msg.contains("string, bytes, or AST"));
            }
            _ => panic!("Expected TypeError"),
        }
    }

    #[test]
    fn test_compile_with_int_source() {
        let result = builtin_compile(&[
            Value::int(42).unwrap(),
            Value::int(0).unwrap(),
            Value::int(0).unwrap(),
        ]);
        assert!(result.is_err());
        match result {
            Err(BuiltinError::TypeError(_)) => {}
            _ => panic!("Expected TypeError"),
        }
    }

    // =========================================================================
    // breakpoint() Tests
    // =========================================================================

    #[test]
    fn test_breakpoint_no_args() {
        let result = builtin_breakpoint(&[]);
        assert!(result.is_ok());
        assert!(result.unwrap().is_none());
    }

    #[test]
    fn test_breakpoint_with_args() {
        // breakpoint accepts arbitrary args and ignores them
        let result = builtin_breakpoint(&[Value::int(1).unwrap(), Value::int(2).unwrap()]);
        assert!(result.is_ok());
    }

    // =========================================================================
    // CompileMode Tests
    // =========================================================================

    #[test]
    fn test_compile_mode_from_str() {
        assert_eq!(CompileMode::from_str("exec"), Some(CompileMode::Exec));
        assert_eq!(CompileMode::from_str("eval"), Some(CompileMode::Eval));
        assert_eq!(CompileMode::from_str("single"), Some(CompileMode::Single));
        assert_eq!(CompileMode::from_str("invalid"), None);
        assert_eq!(CompileMode::from_str(""), None);
    }

    #[test]
    fn test_compile_mode_equality() {
        assert_eq!(CompileMode::Exec, CompileMode::Exec);
        assert_ne!(CompileMode::Exec, CompileMode::Eval);
        assert_ne!(CompileMode::Eval, CompileMode::Single);
    }

    // =========================================================================
    // Edge Cases
    // =========================================================================

    #[test]
    fn test_exec_preserves_input() {
        let val = Value::int(42).unwrap();
        let _ = builtin_exec(&[val.clone()]);
        assert!(val.is_int());
        assert_eq!(val.as_int(), Some(42));
    }

    #[test]
    fn test_eval_preserves_input() {
        let val = Value::float(3.14);
        let _ = builtin_eval(&[val.clone()]);
        assert!(val.is_float());
    }

    #[test]
    fn test_compile_preserves_input() {
        let val = Value::bool(true);
        let _ = builtin_compile(&[val.clone(), Value::int(0).unwrap(), Value::int(0).unwrap()]);
        assert!(val.is_bool());
    }
}
