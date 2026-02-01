//! Code Generation Infrastructure.
//!
//! This module provides the core code generation and patching infrastructure
//! for the Prism JIT. It includes:
//!
//! - **Code Patching**: W^X-safe runtime code modification for OSR and dispatch
//! - **OSR Patching**: Zero-cost On-Stack Replacement via loop header patching
//! - **Dispatch Patching**: Atomic handler replacement for JIT entry
//!
//! # Architecture
//!
//! The codegen module enables dynamic code modification without runtime overhead
//! on the hot path. By patching nop sequences to jumps at activation time,
//! we achieve zero-cost transitions from interpreter to JIT code.

pub mod dispatch_patcher;
pub mod osr_patcher;
pub mod patcher;

pub use patcher::{
    JMP_REL32_OPCODE, JMP_REL32_SIZE, NOP5_BYTES, NOP5_SIZE, PatchDescriptor, PatchStats,
    PatchType, Patcher, encode_jmp_rel32, verify_bytes,
};

pub use osr_patcher::{
    OsrEntryStub, OsrEntryStubBuilder, OsrPatchError, OsrPatchStats, Patchpoint,
    PatchpointRegistry, PatchpointState,
};

pub use dispatch_patcher::{
    DispatchTableStats, DynamicDispatchTable, HandlerEntry, JitCodeEntry, JitEntryRegistry,
    MAX_OPCODES,
};
