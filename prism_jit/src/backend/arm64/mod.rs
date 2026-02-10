//! ARM64 (AArch64) backend modules.
//!
//! This module provides complete ARM64 code generation infrastructure:
//! - `registers`: GPR/SIMD register definitions and AAPCS64 calling conventions
//! - `encoder`: Low-level A64 instruction encoding
//! - `assembler`: High-level code emission with labels
//! - `neon`: NEON/ASIMD SIMD instruction encoding
//! - `cpuinfo`: ARM64 CPU feature detection

pub mod assembler;
pub mod encoder;
pub mod neon;
pub mod registers;

// Re-export commonly used types
pub use assembler::{Arm64Assembler, Label, Relocation, RelocationType};
pub use encoder::{Condition, EncodedInst, ShiftType};
pub use neon::{ArrangementSpec, Vreg, VregSet};
pub use registers::{AllocatableRegs, CallingConvention, Gpr, GprSet, MemOperand};
