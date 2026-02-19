//! Bytecode compiler for Prism.
//!
//! This crate provides the compilation pipeline from AST to register-based bytecode.
//!
//! # Architecture
//!
//! ```text
//! AST → Scope Analysis → Bytecode Emission → CodeObject
//! ```
//!
//! # Key Types
//!
//! - [`CodeObject`] - Compiled function with bytecode and metadata
//! - [`FunctionBuilder`] - High-level API for bytecode construction
//! - [`Instruction`] - 32-bit register-based instruction
//! - [`Opcode`] - Bytecode operation enumeration
//!
//! # Example
//!
//! ```
//! use prism_compiler::{FunctionBuilder, LocalSlot, Register};
//!
//! let mut builder = FunctionBuilder::new("add");
//! builder.set_arg_count(2);
//!
//! let x = builder.alloc_register();
//! let y = builder.alloc_register();
//! let result = builder.alloc_register();
//!
//! builder.emit_load_local(x, LocalSlot::new(0));
//! builder.emit_load_local(y, LocalSlot::new(1));
//! builder.emit_add(result, x, y);
//! builder.emit_return(result);
//!
//! let code = builder.finish();
//! assert_eq!(code.instructions.len(), 4);
//! ```

#![deny(unsafe_op_in_unsafe_fn)]
#![warn(missing_docs)]

pub mod bytecode;
pub mod class_compiler;
pub mod compiler;
pub mod exception_compiler;
pub mod function_compiler;
pub mod match_compiler;
pub mod scope;

// Re-export main types
pub use bytecode::{
    CodeFlags, CodeObject, ConstIndex, ExceptionEntry, FunctionBuilder, Instruction,
    InstructionFormat, Label, LineTableEntry, LocalSlot, Opcode, Register, disassemble,
};
pub use class_compiler::{ClassCompiler, ClassInfo};
pub use compiler::{Compiler, OptimizationLevel};
pub use function_compiler::{
    ClosureInfo, FunctionContext, VarLocation, VariableEmitter, VariableResolver,
};
