//! JVM/V8-tier JIT compiler for Prism.
//!
//! Custom high-performance JIT backend with:
//! - Sea-of-Nodes IR
//! - Advanced optimization passes
//! - Graph-coloring register allocation
//! - Native x64 code generation
//! - Precise GC integration with stack maps
//! - Runtime code caching and background compilation
#![deny(unsafe_op_in_unsafe_fn)]
pub mod backend;
pub mod gc;
pub mod ir;
pub mod opt;
pub mod regalloc;
pub mod runtime;
pub mod tier1;
pub mod tier2;
