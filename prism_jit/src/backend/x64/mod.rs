//! x64 backend modules.
//!
//! This module provides complete x64 code generation infrastructure:
//! - `registers`: GPR/XMM register definitions and calling conventions
//! - `memory`: Executable memory allocation and management
//! - `encoder`: Low-level instruction encoding
//! - `assembler`: High-level code emission with labels
//! - `cpuid`: CPU feature detection for runtime optimization
//! - `simd`: AVX/AVX2 SIMD instruction encoding (VEX prefix)
//! - `evex`: AVX-512 instruction encoding (EVEX prefix)
//! - `vector_spill`: Vector register spill/reload code emission
//! - `call_conv`: Vector calling convention abstractions

pub mod assembler;
pub mod call_conv;
pub mod cpuid;
pub mod encoder;
pub mod evex;
pub mod memory;
pub mod registers;
pub mod simd;
pub mod simd_integration;
pub mod vector_spill;

// Re-export commonly used types
pub use assembler::{
    Assembler, ConstantPool, ConstantPoolEntry, Label, Relocation, RelocationType,
};
pub use call_conv::{
    ArgClass, ArgLocationCalc, CallClobbers, Kreg, KregSet, RetClass, VectorCallingConvention,
};
pub use cpuid::{CpuFeatureFlags, CpuFeatures, CpuLevel, CpuVendor};
pub use encoder::{Condition, EncodedInst, Mod, Rex};
pub use evex::Evex;
pub use memory::{CodeCacheStats, CompiledCode, ExecutableBuffer, PAGE_SIZE};
pub use registers::{
    AllocatableRegs, CallingConvention, Gpr, GprSet, MemOperand, Scale, Xmm, XmmSet,
};
pub use simd::{Vex, Ymm, YmmSet, Zmm, ZmmSet};
pub use vector_spill::{DataKind, FrameBase, SpillError, SpillStats, VectorSpiller};
