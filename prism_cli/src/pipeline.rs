//! Execution pipeline: source → parse → compile → VM → result.
//!
//! Provides the core execution functions used by every CLI mode
//! (script, command string, stdin, REPL).

use crate::config::RuntimeConfig;
use crate::error;
use std::io::Read;
use std::path::Path;
use std::process::ExitCode;
use std::sync::Arc;

// =============================================================================
// Public Pipeline Functions
// =============================================================================

/// Run a Python source file.
///
/// Reads the file, parses, compiles, and executes it through the VM.
/// Returns the process exit code.
pub fn run_file(path: &Path, config: &RuntimeConfig) -> ExitCode {
    let filename = path.display().to_string();

    // Read source file.
    let source = match std::fs::read_to_string(path) {
        Ok(s) => s,
        Err(e) => {
            eprintln!(
                "prism: can't open file '{}': [Errno {}] {}",
                filename,
                e.raw_os_error().unwrap_or(0),
                e,
            );
            return ExitCode::from(error::EXIT_USAGE_ERROR);
        }
    };

    execute_source(&source, &filename, config)
}

/// Run a command string (from `-c` flag).
///
/// Parses and executes the string as if it were a module.
pub fn run_string(code: &str, config: &RuntimeConfig) -> ExitCode {
    execute_source(code, "<string>", config)
}

/// Run from stdin.
///
/// Reads all of stdin, then parses and executes.
pub fn run_stdin(config: &RuntimeConfig) -> ExitCode {
    let mut source = String::new();
    if let Err(e) = std::io::stdin().read_to_string(&mut source) {
        eprintln!("prism: error reading stdin: {}", e);
        return ExitCode::from(error::EXIT_ERROR);
    }
    execute_source(&source, "<stdin>", config)
}

// =============================================================================
// Core Execution
// =============================================================================

/// Execute source code through the full pipeline.
///
/// Pipeline: parse → compile → VM execute.
///
/// Returns the process exit code (0 on success, 1 on error).
fn execute_source(source: &str, filename: &str, config: &RuntimeConfig) -> ExitCode {
    // Phase 1: Parse.
    let module = match prism_parser::parse(source) {
        Ok(m) => m,
        Err(e) => {
            return error::format_prism_error(&e, Some(source), filename);
        }
    };

    // Phase 2: Compile.
    let code = match prism_compiler::Compiler::compile_module(&module, filename) {
        Ok(c) => c,
        Err(e) => {
            return error::format_compile_error(&e, Some(source), filename);
        }
    };

    // Phase 3: Execute.
    let code = Arc::new(code);
    let mut vm = if config.jit_enabled() {
        prism_vm::VirtualMachine::with_jit()
    } else {
        prism_vm::VirtualMachine::new()
    };

    match vm.execute(code) {
        Ok(_) => ExitCode::from(error::EXIT_SUCCESS),
        Err(e) => error::format_prism_error(&e, Some(source), filename),
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn execute_with_default_config(source: &str, filename: &str) -> ExitCode {
        let config = RuntimeConfig::from_args(&crate::args::PrismArgs::default());
        super::execute_source(source, filename, &config)
    }

    fn execute_with_config(source: &str, filename: &str, config: &RuntimeConfig) -> ExitCode {
        super::execute_source(source, filename, config)
    }

    // =========================================================================
    // Source Execution Tests
    // =========================================================================

    #[test]
    fn test_execute_empty_source() {
        let code = execute_source("", "<test>", &RuntimeConfig::from_args(&Default::default()));
        assert_eq!(code, ExitCode::from(0));
    }

    #[test]
    fn test_execute_simple_assignment() {
        let code = execute_source(
            "x = 42",
            "<test>",
            &RuntimeConfig::from_args(&Default::default()),
        );
        assert_eq!(code, ExitCode::from(0));
    }

    #[test]
    fn test_execute_simple_assignment_with_jit_disabled() {
        let args = crate::args::PrismArgs {
            x_options: vec!["nojit".to_string()],
            ..Default::default()
        };
        let config = RuntimeConfig::from_args(&args);
        let code = execute_with_config("x = 42", "<test>", &config);
        assert_eq!(code, ExitCode::from(0));
    }

    #[test]
    fn test_execute_multiple_statements() {
        let code = execute_source(
            "x = 1\ny = 2\nz = x + y",
            "<test>",
            &RuntimeConfig::from_args(&Default::default()),
        );
        assert_eq!(code, ExitCode::from(0));
    }

    #[test]
    fn test_execute_function_def_and_call() {
        let code = execute_source(
            "def add(a, b):\n    return a + b\nresult = add(1, 2)",
            "<test>",
            &RuntimeConfig::from_args(&Default::default()),
        );
        assert_eq!(code, ExitCode::from(0));
    }

    #[test]
    fn test_execute_if_statement() {
        let code = execute_source(
            "x = 10\nif x > 5:\n    y = True\nelse:\n    y = False",
            "<test>",
            &RuntimeConfig::from_args(&Default::default()),
        );
        assert_eq!(code, ExitCode::from(0));
    }

    #[test]
    fn test_execute_for_loop() {
        let code = execute_source(
            "total = 0\nfor i in range(10):\n    total = total + i",
            "<test>",
            &RuntimeConfig::from_args(&Default::default()),
        );
        assert_eq!(code, ExitCode::from(0));
    }

    #[test]
    fn test_execute_while_loop() {
        let code = execute_source(
            "x = 0\nwhile x < 5:\n    x = x + 1",
            "<test>",
            &RuntimeConfig::from_args(&Default::default()),
        );
        assert_eq!(code, ExitCode::from(0));
    }

    #[test]
    fn test_execute_list_operations() {
        let code = execute_source(
            "lst = [1, 2, 3]\nx = len(lst)",
            "<test>",
            &RuntimeConfig::from_args(&Default::default()),
        );
        assert_eq!(code, ExitCode::from(0));
    }

    #[test]
    fn test_execute_dict_operations() {
        let code = execute_source(
            "d = {'a': 1, 'b': 2}",
            "<test>",
            &RuntimeConfig::from_args(&Default::default()),
        );
        assert_eq!(code, ExitCode::from(0));
    }

    #[test]
    fn test_execute_string_operations() {
        let code = execute_with_default_config("s = 'hello'\nx = len(s)", "<test>");
        assert_eq!(code, ExitCode::from(0));
    }

    #[test]
    fn test_execute_print_call() {
        let code = execute_with_default_config("print('hello, world')", "<test>");
        assert_eq!(code, ExitCode::from(0));
    }

    #[test]
    fn test_execute_nested_function() {
        let code = execute_with_default_config(
            "def outer():\n    def inner():\n        return 42\n    return inner()\nresult = outer()",
            "<test>",
        );
        assert_eq!(code, ExitCode::from(0));
    }

    #[test]
    fn test_execute_class_definition() {
        let code = execute_with_default_config("class Foo:\n    x = 42", "<test>");
        assert_eq!(code, ExitCode::from(0));
    }

    #[test]
    fn test_execute_try_except() {
        let code = execute_with_default_config("try:\n    x = 1\nexcept:\n    x = 0", "<test>");
        assert_eq!(code, ExitCode::from(0));
    }

    #[test]
    fn test_execute_list_comprehension() {
        let code = execute_with_default_config("squares = [x * x for x in range(5)]", "<test>");
        assert_eq!(code, ExitCode::from(0));
    }

    // =========================================================================
    // Error Handling Tests
    // =========================================================================

    #[test]
    fn test_execute_syntax_error_returns_error_code() {
        let code = execute_with_default_config("def", "<test>");
        assert_ne!(code, ExitCode::from(0));
    }

    #[test]
    fn test_execute_name_error_returns_error_code() {
        let code = execute_with_default_config("print(undefined_variable)", "<test>");
        assert_ne!(code, ExitCode::from(0));
    }

    // =========================================================================
    // File Execution Tests
    // =========================================================================

    #[test]
    fn test_run_file_nonexistent() {
        let config = RuntimeConfig::from_args(&crate::args::PrismArgs::default());
        let code = run_file(Path::new("/nonexistent/path/test.py"), &config);
        assert_eq!(code, ExitCode::from(error::EXIT_USAGE_ERROR));
    }

    // =========================================================================
    // Edge Case Tests
    // =========================================================================

    #[test]
    fn test_execute_only_comments() {
        let code = execute_with_default_config("# just a comment\n# another comment", "<test>");
        assert_eq!(code, ExitCode::from(0));
    }

    #[test]
    fn test_execute_only_whitespace() {
        let code = execute_with_default_config("   \n\n   \n", "<test>");
        assert_eq!(code, ExitCode::from(0));
    }

    #[test]
    fn test_execute_pass_statement() {
        let code = execute_with_default_config("pass", "<test>");
        assert_eq!(code, ExitCode::from(0));
    }

    #[test]
    fn test_execute_multiline_string() {
        let code = execute_with_default_config("s = '''hello\nworld'''", "<test>");
        assert_eq!(code, ExitCode::from(0));
    }

    #[test]
    fn test_execute_boolean_operations() {
        let code = execute_with_default_config(
            "x = True and False\ny = True or False\nz = not True",
            "<test>",
        );
        assert_eq!(code, ExitCode::from(0));
    }

    #[test]
    fn test_execute_comparison_chain() {
        let code = execute_with_default_config("result = 1 < 2 < 3", "<test>");
        assert_eq!(code, ExitCode::from(0));
    }

    #[test]
    fn test_execute_tuple_unpacking() {
        let code = execute_with_default_config("a, b = 1, 2", "<test>");
        assert_eq!(code, ExitCode::from(0));
    }

    #[test]
    fn test_execute_augmented_assignment() {
        let code = execute_with_default_config("x = 1\nx += 2\nx *= 3", "<test>");
        assert_eq!(code, ExitCode::from(0));
    }

    #[test]
    fn test_execute_global_statement() {
        let code = execute_with_default_config(
            "x = 0\ndef inc():\n    global x\n    x = x + 1\ninc()\ninc()",
            "<test>",
        );
        assert_eq!(code, ExitCode::from(0));
    }

    #[test]
    fn test_execute_lambda() {
        let code = execute_with_default_config("f = lambda x: x * 2\nresult = f(21)", "<test>");
        assert_eq!(code, ExitCode::from(0));
    }

    #[test]
    fn test_execute_assert_passes() {
        let code = execute_with_default_config("assert True", "<test>");
        assert_eq!(code, ExitCode::from(0));
    }

    #[test]
    #[ignore = "compiler assert is a stub - does not yet raise AssertionError"]
    fn test_execute_assert_fails_returns_error() {
        let code = execute_with_default_config("assert False", "<test>");
        assert_ne!(code, ExitCode::from(0));
    }

    #[test]
    fn test_execute_delete_statement() {
        let code = execute_with_default_config("x = 1\ndel x", "<test>");
        assert_eq!(code, ExitCode::from(0));
    }

    #[test]
    fn test_run_string_simple() {
        let config = RuntimeConfig::from_args(&crate::args::PrismArgs::default());
        let code = run_string("x = 42", &config);
        assert_eq!(code, ExitCode::from(0));
    }

    #[test]
    fn test_run_string_error() {
        let config = RuntimeConfig::from_args(&crate::args::PrismArgs::default());
        let code = run_string("def", &config);
        assert_ne!(code, ExitCode::from(0));
    }
}
