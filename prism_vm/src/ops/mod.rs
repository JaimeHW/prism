//! Opcode handler modules.
//!
//! Organized by category for maintainability. Each handler takes
//! a reference to the VM and an instruction, returning a ControlFlow.

pub mod arithmetic;
pub mod calls;
pub mod comparison;
pub mod containers;
pub mod control;
pub mod exception;
pub mod generator;
pub mod load_store;
pub mod objects;

// Re-export common types for convenience
pub use super::dispatch::ControlFlow;
