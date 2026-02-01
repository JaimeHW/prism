//! Bytecode compiler for Prism.
#![deny(unsafe_op_in_unsafe_fn)]
pub mod bytecode;
pub mod compiler;
pub use bytecode::*;
pub use compiler::Compiler;
