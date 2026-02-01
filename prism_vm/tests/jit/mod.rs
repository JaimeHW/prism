//! JIT Integration Test Suite
//!
//! Comprehensive tests for the JIT compilation and execution pipeline.
//! Organized by category:
//! - arithmetic: Integer/float/generic arithmetic through JIT
//! - control: Loops, branches, and control flow
//! - containers: List/tuple/dict operations
//! - calls: Function and method calls
//! - tiering: Tier-up and OSR verification

pub mod arithmetic;
pub mod control;
pub mod test_utils;
