//! Match Statement Compiler
//!
//! Implements Python 3.10+ structural pattern matching (PEP 634).
//!
//! # Architecture
//!
//! This module provides Maranget's decision tree algorithm for optimal
//! pattern matching compilation:
//!
//! - `pattern` - Pattern normalization to flat representation
//! - `matrix` - Pattern matrix construction and specialization
//! - `decision_tree` - Maranget's algorithm for optimal tree generation
//! - `codegen` - Bytecode emission from decision trees
//!
//! The main entry point `compile_match` in `compiler.rs` can use either
//! the direct recursive matcher or the decision tree approach.

pub mod codegen;
pub mod decision_tree;
pub mod matrix;
pub mod pattern;

pub use codegen::{SubjectCache, emit_tree};
pub use codegen::{
    emit_copy_dict_without_keys, emit_match_class, emit_match_keys, emit_match_mapping,
    emit_match_sequence,
};
pub use decision_tree::{DecisionTree, SwitchCase, compile};
pub use matrix::{Constructor, PatternMatrix, PatternRow};
pub use pattern::{AccessPath, Binding, FlatPattern, LiteralKey, LiteralValue, flatten};
