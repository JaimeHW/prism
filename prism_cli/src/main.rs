//! Prism — A hyper-performant Python runtime.
//!
//! This is the main CLI entry point that provides CPython-compatible
//! command-line interface for running Python programs.

mod args;
mod config;
mod diagnostics;
mod error;
mod pipeline;
mod repl;

use args::ExecutionMode;
use std::process::ExitCode;

fn main() -> ExitCode {
    // Parse CLI arguments (skip argv[0] = program name).
    let raw_args: Vec<String> = std::env::args().skip(1).collect();

    let parsed = match args::parse_args_vec(&raw_args) {
        Ok(a) => a,
        Err(e) => {
            eprintln!("prism: {}", e);
            return ExitCode::from(error::EXIT_USAGE_ERROR);
        }
    };

    // Handle immediate-exit modes before building config.
    match &parsed.mode {
        ExecutionMode::PrintVersion => {
            println!("{}", args::version_string());
            return ExitCode::SUCCESS;
        }
        ExecutionMode::PrintHelp => {
            println!("{}", args::help_text());
            return ExitCode::SUCCESS;
        }
        _ => {}
    }

    // Build runtime configuration from args + environment.
    let config = config::RuntimeConfig::from_args(&parsed);

    // Configure unbuffered output if requested.
    // Note: Rust's stdout is already line-buffered for terminals.

    // Dispatch to the appropriate execution mode.
    match &parsed.mode {
        ExecutionMode::Script(path) => {
            pipeline::run_file_with_args(path, &config, &parsed.script_args)
        }
        ExecutionMode::Command(cmd) => {
            pipeline::run_string_with_args(cmd, &config, &parsed.script_args)
        }
        ExecutionMode::Stdin => pipeline::run_stdin_with_args(&config, &parsed.script_args),
        ExecutionMode::Module(name) => {
            pipeline::run_module_with_args(name, &config, &parsed.script_args)
        }
        ExecutionMode::Repl => repl::run_repl(&config),
        ExecutionMode::PrintVersion | ExecutionMode::PrintHelp => {
            unreachable!("handled above")
        }
    }
}
