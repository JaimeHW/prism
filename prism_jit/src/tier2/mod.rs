//! Tier 2 Optimizing JIT Compiler
//!
//! This module contains the final code generation stages that convert
//! the optimized Sea-of-Nodes IR into executable machine code.
//!
//! # Pipeline
//!
//! 1. **Instruction Selection** (`lower.rs`): Convert IR nodes to machine instructions
//! 2. **Code Emission** (`emit.rs`): Emit final machine code with register assignments
//! 3. **OSR** (`osr.rs`): On-stack replacement for tier transitions
//! 4. **State Points** (`state_points.rs`): Deoptimization state mapping
//!
//! # Usage
//!
//! ```ignore
//! // After optimization and register allocation
//! let mir = InstructionSelector::select(&optimized_graph, &alloc_map)?;
//! let code = CodeEmitter::emit(&mir)?;
//! ```

pub mod emit;
pub mod lower;
pub mod osr;
pub mod osr_stub;
pub mod safepoint_placement;
pub mod state_points;

pub use emit::CodeEmitter;
pub use lower::{InstructionSelector, MachineInst, MachineOp};
pub use osr::{DeoptInfo, DeoptReason, OsrCompiledCode, OsrEntry, ValueLocation};
pub use osr_stub::{OsrExitBuilder, OsrStubCache, OsrStubInfo};
pub use safepoint_placement::{SafepointAnalyzer, SafepointEmitter, SafepointPlacement};
pub use state_points::{StatePoint, StatePointBuilder, StatePointTable};
