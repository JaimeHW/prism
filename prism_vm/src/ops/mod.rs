//! Opcode handler modules.
//!
//! Organized by category for maintainability. Each handler takes
//! a reference to the VM and an instruction, returning a ControlFlow.

pub mod arithmetic;
pub mod calls;
pub mod class;
pub mod comparison;
pub mod containers;
pub mod control;
pub mod coroutine;
pub mod exception;
pub mod generator;
pub mod kw_binding;
pub mod load_store;
pub mod r#match;
pub mod method_dispatch;
pub mod objects;
pub mod unpack;

// Re-export common types for convenience
pub use super::dispatch::ControlFlow;
