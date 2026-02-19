//! Interactive REPL (Read-Eval-Print Loop) matching CPython's behavior.
//!
//! Provides `>>>` and `...` prompts, multiline statement support,
//! expression result display, and error handling without exit.

use crate::config::RuntimeConfig;
use std::io::{self, BufRead, Write};
use std::process::ExitCode;
use std::sync::Arc;

// =============================================================================
// REPL Entry Point
// =============================================================================

/// Start the interactive REPL.
///
/// Behavior matches CPython:
/// - `>>>` primary prompt
/// - `...` continuation prompt for multiline constructs
/// - Displays expression results (non-None)
/// - Errors are printed but don't exit the REPL
/// - `exit()` or `quit()` to exit
/// - Ctrl+D (EOF) exits cleanly
pub fn run_repl(config: &RuntimeConfig) -> ExitCode {
    // Print banner unless `-q` was specified.
    if !config.quiet {
        println!(
            "Prism {} (Python {}.{}.{} compatible)",
            prism_core::VERSION,
            prism_core::PYTHON_VERSION.0,
            prism_core::PYTHON_VERSION.1,
            prism_core::PYTHON_VERSION.2,
        );
        println!("Type \"help\", \"copyright\", \"credits\" or \"license\" for more information.");
    }

    // Create a persistent VM for the REPL session.
    let mut vm = if config.jit_enabled() {
        prism_vm::VirtualMachine::with_jit()
    } else {
        prism_vm::VirtualMachine::new()
    };

    let stdin = io::stdin();
    let mut reader = stdin.lock();
    let mut line_buf = String::new();

    loop {
        // Print prompt.
        print!(">>> ");
        if io::stdout().flush().is_err() {
            break;
        }

        // Read first line.
        line_buf.clear();
        match reader.read_line(&mut line_buf) {
            Ok(0) => {
                // EOF (Ctrl+D).
                println!();
                break;
            }
            Ok(_) => {}
            Err(_) => break,
        }

        let trimmed = line_buf.trim();

        // Handle exit commands.
        if trimmed == "exit()" || trimmed == "quit()" {
            break;
        }

        // Handle empty lines.
        if trimmed.is_empty() {
            continue;
        }

        // Collect multiline input for compound statements.
        let mut source = line_buf.clone();
        if needs_continuation(trimmed) {
            loop {
                print!("... ");
                if io::stdout().flush().is_err() {
                    break;
                }
                line_buf.clear();
                match reader.read_line(&mut line_buf) {
                    Ok(0) => break, // EOF
                    Ok(_) => {}
                    Err(_) => break,
                }
                // Empty line terminates multiline input.
                if line_buf.trim().is_empty() {
                    break;
                }
                source.push_str(&line_buf);
            }
        }

        // Execute the input.
        execute_repl_input(&source, &mut vm, config);
    }

    ExitCode::from(crate::error::EXIT_SUCCESS)
}

// =============================================================================
// REPL Execution
// =============================================================================

/// Execute a single REPL input, displaying results or errors.
fn execute_repl_input(source: &str, vm: &mut prism_vm::VirtualMachine, config: &RuntimeConfig) {
    // Parse.
    let module = match prism_parser::parse(source) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("{}", e);
            return;
        }
    };

    // Compile.
    let optimize = match config.optimize {
        crate::args::OptimizationLevel::None => prism_compiler::OptimizationLevel::None,
        crate::args::OptimizationLevel::Basic => prism_compiler::OptimizationLevel::Basic,
        crate::args::OptimizationLevel::Full => prism_compiler::OptimizationLevel::Full,
    };
    let code = match prism_compiler::Compiler::compile_module_with_optimization(
        &module, "<stdin>", optimize,
    ) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("SyntaxError: {}", e.message);
            return;
        }
    };

    // Execute.
    let code = Arc::new(code);
    match vm.execute(code) {
        Ok(result) => {
            // Display non-None results (matching CPython REPL behavior).
            if !result.is_none() {
                println!("{}", format_value(&result));
            }
        }
        Err(e) => {
            eprintln!("{}", e);
        }
    }
}

/// Check if a line needs continuation (starts a compound statement).
///
/// Compound statements that require `...` continuation:
/// - `def`, `class`, `if`, `elif`, `else`, `for`, `while`, `try`, `except`,
///   `finally`, `with`, `match`, `case`
/// - Trailing `:` at end of line
/// - Trailing `\` (explicit line continuation)
/// - Unclosed brackets/parens (simplified: just check for trailing `:`)
#[inline]
fn needs_continuation(line: &str) -> bool {
    let trimmed = line.trim();

    // Explicit line continuation.
    if trimmed.ends_with('\\') {
        return true;
    }

    // Compound statement keywords ending with `:`.
    if trimmed.ends_with(':') {
        let first_word = trimmed.split_whitespace().next().unwrap_or("");
        // Strip trailing `:` from the first word so `"else:"` matches `"else"`.
        let first_word = first_word.strip_suffix(':').unwrap_or(first_word);
        matches!(
            first_word,
            "def"
                | "class"
                | "if"
                | "elif"
                | "else"
                | "for"
                | "while"
                | "try"
                | "except"
                | "finally"
                | "with"
                | "match"
                | "case"
                | "async"
        )
    } else {
        false
    }
}

/// Format a Value for REPL display.
///
/// Matches CPython's `repr()` semantics.
fn format_value(value: &prism_core::Value) -> String {
    if let Some(i) = value.as_int() {
        i.to_string()
    } else if let Some(f) = value.as_float() {
        // Match CPython's float repr.
        if f.fract() == 0.0 && f.is_finite() {
            format!("{:.1}", f)
        } else {
            format!("{}", f)
        }
    } else if let Some(b) = value.as_bool() {
        if b { "True" } else { "False" }.to_string()
    } else if value.is_none() {
        // None is not displayed in REPL (handled by caller).
        "None".to_string()
    } else if value.is_string() {
        // Strings in REPL are shown with quotes (repr).
        format!("'{}'", value)
    } else {
        format!("{}", value)
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // Continuation Detection Tests
    // =========================================================================

    #[test]
    fn test_needs_continuation_def() {
        assert!(needs_continuation("def foo():"));
    }

    #[test]
    fn test_needs_continuation_class() {
        assert!(needs_continuation("class MyClass:"));
    }

    #[test]
    fn test_needs_continuation_if() {
        assert!(needs_continuation("if x > 0:"));
    }

    #[test]
    fn test_needs_continuation_elif() {
        assert!(needs_continuation("elif x < 0:"));
    }

    #[test]
    fn test_needs_continuation_else() {
        assert!(needs_continuation("else:"));
    }

    #[test]
    fn test_needs_continuation_for() {
        assert!(needs_continuation("for i in range(10):"));
    }

    #[test]
    fn test_needs_continuation_while() {
        assert!(needs_continuation("while True:"));
    }

    #[test]
    fn test_needs_continuation_try() {
        assert!(needs_continuation("try:"));
    }

    #[test]
    fn test_needs_continuation_except() {
        assert!(needs_continuation("except ValueError:"));
    }

    #[test]
    fn test_needs_continuation_finally() {
        assert!(needs_continuation("finally:"));
    }

    #[test]
    fn test_needs_continuation_with() {
        assert!(needs_continuation("with open('f') as f:"));
    }

    #[test]
    fn test_needs_continuation_match() {
        assert!(needs_continuation("match x:"));
    }

    #[test]
    fn test_needs_continuation_case() {
        assert!(needs_continuation("case 1:"));
    }

    #[test]
    fn test_needs_continuation_async() {
        assert!(needs_continuation("async def foo():"));
    }

    #[test]
    fn test_needs_continuation_backslash() {
        assert!(needs_continuation("x = 1 + \\"));
    }

    #[test]
    fn test_no_continuation_assignment() {
        assert!(!needs_continuation("x = 42"));
    }

    #[test]
    fn test_no_continuation_function_call() {
        assert!(!needs_continuation("print('hello')"));
    }

    #[test]
    fn test_no_continuation_colon_in_dict() {
        // Dict literal has `:` but doesn't start with a keyword.
        assert!(!needs_continuation("d = {'a': 1}"));
    }

    #[test]
    fn test_no_continuation_empty() {
        assert!(!needs_continuation(""));
    }

    #[test]
    fn test_no_continuation_comment() {
        assert!(!needs_continuation("# comment"));
    }

    // =========================================================================
    // Value Formatting Tests
    // =========================================================================

    #[test]
    fn test_format_value_int() {
        let v = prism_core::Value::int(42).unwrap();
        assert_eq!(format_value(&v), "42");
    }

    #[test]
    fn test_format_value_negative_int() {
        let v = prism_core::Value::int(-7).unwrap();
        assert_eq!(format_value(&v), "-7");
    }

    #[test]
    fn test_format_value_zero() {
        let v = prism_core::Value::int(0).unwrap();
        assert_eq!(format_value(&v), "0");
    }

    #[test]
    fn test_format_value_float() {
        let v = prism_core::Value::float(3.14);
        assert_eq!(format_value(&v), "3.14");
    }

    #[test]
    fn test_format_value_float_whole() {
        let v = prism_core::Value::float(42.0);
        assert_eq!(format_value(&v), "42.0");
    }

    #[test]
    fn test_format_value_bool_true() {
        let v = prism_core::Value::bool(true);
        assert_eq!(format_value(&v), "True");
    }

    #[test]
    fn test_format_value_bool_false() {
        let v = prism_core::Value::bool(false);
        assert_eq!(format_value(&v), "False");
    }

    #[test]
    fn test_format_value_none() {
        let v = prism_core::Value::none();
        assert_eq!(format_value(&v), "None");
    }

    // =========================================================================
    // REPL Execution Tests
    // =========================================================================

    #[test]
    fn test_execute_repl_assignment() {
        let mut vm = prism_vm::VirtualMachine::new();
        let config = RuntimeConfig::from_args(&crate::args::PrismArgs::default());
        // Should not panic.
        execute_repl_input("x = 42\n", &mut vm, &config);
    }

    #[test]
    fn test_execute_repl_syntax_error() {
        let mut vm = prism_vm::VirtualMachine::new();
        let config = RuntimeConfig::from_args(&crate::args::PrismArgs::default());
        // Should print error but not panic.
        execute_repl_input("def\n", &mut vm, &config);
    }

    #[test]
    fn test_execute_repl_print() {
        let mut vm = prism_vm::VirtualMachine::new();
        let config = RuntimeConfig::from_args(&crate::args::PrismArgs::default());
        execute_repl_input("print('hello')\n", &mut vm, &config);
    }

    #[test]
    fn test_execute_repl_multiline_function() {
        let mut vm = prism_vm::VirtualMachine::new();
        let config = RuntimeConfig::from_args(&crate::args::PrismArgs::default());
        execute_repl_input("def foo():\n    return 42\n\n", &mut vm, &config);
    }
}
