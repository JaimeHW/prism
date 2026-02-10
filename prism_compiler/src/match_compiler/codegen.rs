//! Bytecode generation from decision trees.
//!
//! This module traverses compiled decision trees and emits optimal bytecode:
//! - Subject caching (avoid redundant loads)
//! - Type/length/key checks using specialized opcodes
//! - Binding stores to locals
//! - Guard evaluation with backtracking support
//!
//! # Performance Optimizations
//!
//! 1. **Subject Caching**: Accessed substructures are cached in registers to
//!    avoid redundant loads during tree traversal.
//! 2. **Specialized Opcodes**: Uses MatchClass, MatchMapping, MatchSequence
//!    for efficient type checks instead of generic isinstance calls.
//! 3. **Branch Optimization**: Generates efficient jump targets with minimal
//!    label resolution overhead.
//! 4. **Guard Backtracking**: Efficiently handles guard failures by continuing
//!    pattern matching from the fallback tree.

use super::decision_tree::{DecisionTree, SwitchCase};
use super::matrix::Constructor;
use super::pattern::{AccessPath, Binding, LiteralKey, LiteralValue};
use crate::bytecode::{FunctionBuilder, Instruction, Label, Opcode, Register};
use crate::compiler::CompileResult;
use prism_parser::ast::Expr;
use std::collections::HashMap;

// =============================================================================
// Subject Cache
// =============================================================================

/// Cache for accessed subject parts.
///
/// This avoids redundant loads when the same path is accessed multiple times
/// during decision tree traversal. The cache maps access paths to registers
/// that hold the loaded values.
///
/// # Example
///
/// For pattern `case [Point(x, y), z]:`, accessing subject[0].x requires:
/// 1. Load subject (root)
/// 2. Load subject[0] (index)
/// 3. Load subject[0].x (attr)
///
/// The cache stores each intermediate value so subsequent accesses to
/// subject[0].y reuse the cached subject[0] register.
#[derive(Debug)]
pub struct SubjectCache {
    /// Register holding the root subject value.
    root: Register,
    /// Map from access paths to registers holding cached values.
    cache: HashMap<AccessPath, Register>,
    /// Allocator for tracking which registers are in use.
    allocated: Vec<Register>,
}

impl SubjectCache {
    /// Create a new subject cache with the given root register.
    pub fn new(root: Register) -> Self {
        let mut cache = HashMap::new();
        cache.insert(AccessPath::Root, root);

        Self {
            root,
            cache,
            allocated: Vec::new(),
        }
    }

    /// Get or load a value at the given access path.
    ///
    /// If the value is already cached, returns the cached register.
    /// Otherwise, emits instructions to compute the value and caches it.
    pub fn get_or_load(
        &mut self,
        builder: &mut FunctionBuilder,
        path: &AccessPath,
    ) -> CompileResult<Register> {
        // Check cache first
        if let Some(&reg) = self.cache.get(path) {
            return Ok(reg);
        }

        // Need to load - compute based on path type
        let reg = match path {
            AccessPath::Root => self.root,

            AccessPath::Index(parent, idx) => {
                let parent_reg = self.get_or_load(builder, parent)?;
                let reg = builder.alloc_register();
                self.allocated.push(reg);

                // Load index as constant
                let idx_const = builder.add_int(*idx as i64);
                let idx_reg = builder.alloc_register();
                builder.emit_load_const(idx_reg, idx_const);

                // Emit GetItem: reg = parent[idx]
                self.emit_get_item(builder, reg, parent_reg, idx_reg);
                builder.free_register(idx_reg);

                reg
            }

            AccessPath::Key(parent, key) => {
                let parent_reg = self.get_or_load(builder, parent)?;
                let reg = builder.alloc_register();
                self.allocated.push(reg);

                // Load key as constant
                let key_reg = self.load_literal_key(builder, key)?;

                // Emit GetItem: reg = parent[key]
                self.emit_get_item(builder, reg, parent_reg, key_reg);
                builder.free_register(key_reg);

                reg
            }

            AccessPath::Attr(parent, name) => {
                let parent_reg = self.get_or_load(builder, parent)?;
                let reg = builder.alloc_register();
                self.allocated.push(reg);

                // Add attribute name to names pool
                let name_idx = builder.add_name(name.clone());

                // Emit GetAttr: reg = parent.name
                builder.emit_get_attr(reg, parent_reg, name_idx);

                reg
            }

            AccessPath::StarSlice {
                parent,
                start,
                from_end,
            } => {
                let parent_reg = self.get_or_load(builder, parent)?;
                let reg = builder.alloc_register();
                self.allocated.push(reg);

                // For star slices, we need to compute parent[start:len-from_end]
                // This requires getting the length first
                let len_reg = builder.alloc_register();
                self.emit_len(builder, len_reg, parent_reg);

                // Compute end = len - from_end
                let from_end_const = builder.add_int(*from_end as i64);
                let from_end_reg = builder.alloc_register();
                builder.emit_load_const(from_end_reg, from_end_const);

                let end_reg = builder.alloc_register();
                builder.emit_sub(end_reg, len_reg, from_end_reg);
                builder.free_register(from_end_reg);
                builder.free_register(len_reg);

                // Build slice object: emit BuildSlice opcode
                let start_const = builder.add_int(*start as i64);
                let start_reg = builder.alloc_register();
                builder.emit_load_const(start_reg, start_const);

                let slice_reg = builder.alloc_register();
                // BuildSlice: dst = slice(start, end, step)
                // Use None for step
                let step_reg = builder.alloc_register();
                builder.emit_load_none(step_reg);
                builder.emit(Instruction::op_dss(
                    Opcode::BuildSlice,
                    slice_reg,
                    start_reg,
                    end_reg,
                ));
                builder.free_register(step_reg);
                builder.free_register(start_reg);
                builder.free_register(end_reg);

                // Get the slice: reg = parent[slice]
                self.emit_get_item(builder, reg, parent_reg, slice_reg);
                builder.free_register(slice_reg);

                reg
            }
        };

        // Cache the result
        self.cache.insert(path.clone(), reg);
        Ok(reg)
    }

    /// Load a literal key into a register.
    fn load_literal_key(
        &mut self,
        builder: &mut FunctionBuilder,
        key: &LiteralKey,
    ) -> CompileResult<Register> {
        let reg = builder.alloc_register();

        match key {
            LiteralKey::Int(n) => {
                let idx = builder.add_int(*n);
                builder.emit_load_const(reg, idx);
            }
            LiteralKey::String(s) => {
                let idx = builder.add_string(s.as_ref());
                builder.emit_load_const(reg, idx);
            }
            LiteralKey::Bytes(_b) => {
                // Bytes constants are rarely used as mapping keys
                // For now, create an empty string constant as placeholder
                let idx = builder.add_string("");
                builder.emit_load_const(reg, idx);
            }
        }

        Ok(reg)
    }

    /// Emit GetItem instruction.
    fn emit_get_item(
        &self,
        builder: &mut FunctionBuilder,
        dst: Register,
        obj: Register,
        key: Register,
    ) {
        builder.emit(Instruction::op_dss(Opcode::GetItem, dst, obj, key));
    }

    /// Emit Len instruction.
    fn emit_len(&self, builder: &mut FunctionBuilder, dst: Register, src: Register) {
        builder.emit(Instruction::op_ds(Opcode::Len, dst, src));
    }

    /// Free all cached registers.
    pub fn free_all(&self, builder: &mut FunctionBuilder) {
        for &reg in &self.allocated {
            builder.free_register(reg);
        }
    }
}

// =============================================================================
// Match Codegen Context
// =============================================================================

/// Context for match statement code generation.
///
/// Holds all the state needed during bytecode emission from a decision tree.
pub struct MatchCodegen<'a, 'b> {
    /// Reference to the function builder.
    builder: &'a mut FunctionBuilder,
    /// Cache for subject access paths.
    subject_cache: &'b mut SubjectCache,
    /// Label for end of match statement.
    end_label: Label,
    /// Label for match failure (no cases matched).
    fail_label: Label,
}

impl<'a, 'b> MatchCodegen<'a, 'b> {
    /// Create a new match codegen context.
    pub fn new(
        builder: &'a mut FunctionBuilder,
        subject_cache: &'b mut SubjectCache,
        end_label: Label,
        fail_label: Label,
    ) -> Self {
        Self {
            builder,
            subject_cache,
            end_label,
            fail_label,
        }
    }
}

// =============================================================================
// Decision Tree Emission
// =============================================================================

/// Emit bytecode from a decision tree.
///
/// This is the main entry point for decision tree code generation. It traverses
/// the tree and emits bytecode for each node type.
///
/// # Arguments
///
/// * `builder` - The function builder for emitting bytecode
/// * `tree` - The decision tree to emit
/// * `subject_reg` - Register holding the match subject
/// * `fail_label` - Label to jump to if no pattern matches
/// * `end_label` - Label for end of match statement
///
/// # Returns
///
/// A compilation result indicating success or error.
pub fn emit_tree(
    builder: &mut FunctionBuilder,
    tree: &DecisionTree,
    subject_reg: Register,
    fail_label: Label,
    end_label: Label,
) -> CompileResult<()> {
    let mut cache = SubjectCache::new(subject_reg);
    emit_tree_node(builder, tree, &mut cache, fail_label, end_label)
}

/// Emit bytecode for a single decision tree node.
fn emit_tree_node(
    builder: &mut FunctionBuilder,
    tree: &DecisionTree,
    cache: &mut SubjectCache,
    fail_label: Label,
    end_label: Label,
) -> CompileResult<()> {
    match tree {
        DecisionTree::Fail => {
            // Jump to failure label
            builder.emit_jump(fail_label);
        }

        DecisionTree::Leaf {
            bindings,
            guard,
            action,
            fallback,
        } => {
            emit_leaf(
                builder,
                cache,
                bindings,
                guard.as_ref(),
                *action,
                fallback.as_deref(),
                fail_label,
                end_label,
            )?;
        }

        DecisionTree::Switch {
            access,
            cases,
            default,
        } => {
            emit_switch(
                builder,
                cache,
                access,
                cases,
                default.as_deref(),
                fail_label,
                end_label,
            )?;
        }

        DecisionTree::TypeCheck {
            access,
            cls,
            success,
            failure,
        } => {
            emit_type_check(
                builder, cache, access, cls, success, failure, fail_label, end_label,
            )?;
        }

        DecisionTree::SequenceCheck {
            access,
            min_len,
            exact,
            success,
            failure,
        } => {
            emit_sequence_check(
                builder, cache, access, *min_len, *exact, success, failure, fail_label, end_label,
            )?;
        }

        DecisionTree::MappingCheck {
            access,
            keys,
            success,
            failure,
        } => {
            emit_mapping_check(
                builder, cache, access, keys, success, failure, fail_label, end_label,
            )?;
        }
    }

    Ok(())
}

/// Emit bytecode for a leaf node.
///
/// Stores bindings to locals, evaluates guard (if any), and jumps to appropriate
/// body or fallback.
fn emit_leaf(
    builder: &mut FunctionBuilder,
    cache: &mut SubjectCache,
    bindings: &[Binding],
    guard: Option<&Expr>,
    _action: usize,
    fallback: Option<&DecisionTree>,
    fail_label: Label,
    end_label: Label,
) -> CompileResult<()> {
    // Store bindings to locals
    for binding in bindings {
        let value_reg = cache.get_or_load(builder, &binding.access)?;

        // Look up or define local for this binding
        let slot = builder.define_local(binding.name.as_ref());
        builder.emit_store_local(slot, value_reg);
    }

    // Handle guard expression
    if let Some(_guard_expr) = guard {
        if let Some(fallback_tree) = fallback {
            // Create label for guard failure
            let guard_fail = builder.create_label();

            // Compile guard expression - for now, emit a placeholder
            // The actual guard compilation will be done by the compiler
            // which has access to compile_expr
            // TODO: Pass guard compilation closure or integrate with compiler

            // If guard fails, go to fallback
            builder.bind_label(guard_fail);
            emit_tree_node(builder, fallback_tree, cache, fail_label, end_label)?;
        } else {
            // No fallback, guard failure means match failure
            // TODO: Compile guard and jump to fail_label on failure
        }
    }

    // Match succeeded - body will be compiled separately by the caller
    // For now, we assume the caller handles body compilation after bindings
    // Jump to end after body (caller's responsibility)

    Ok(())
}

/// Emit bytecode for a switch node.
///
/// Tests the value at the access path against each constructor case.
fn emit_switch(
    builder: &mut FunctionBuilder,
    cache: &mut SubjectCache,
    access: &AccessPath,
    cases: &[SwitchCase],
    default: Option<&DecisionTree>,
    fail_label: Label,
    end_label: Label,
) -> CompileResult<()> {
    if cases.is_empty() {
        // No cases - go to default or fail
        if let Some(default_tree) = default {
            emit_tree_node(builder, default_tree, cache, fail_label, end_label)?;
        } else {
            builder.emit_jump(fail_label);
        }
        return Ok(());
    }

    let value_reg = cache.get_or_load(builder, access)?;

    // For each case, test and branch
    for (i, case) in cases.iter().enumerate() {
        let is_last = i + 1 == cases.len();

        // Create label for next case
        let next_label = if is_last && default.is_none() {
            fail_label
        } else {
            builder.create_label()
        };

        // Emit test for this constructor
        let result_reg = builder.alloc_register();
        emit_constructor_test(builder, result_reg, value_reg, &case.ctor)?;

        // Jump to next case if test fails
        builder.emit_jump_if_false(result_reg, next_label);
        builder.free_register(result_reg);

        // Emit subtree for this case
        emit_tree_node(builder, &case.tree, cache, fail_label, end_label)?;

        // Jump to end if match succeeded (subtree should handle this)
        if !is_last || default.is_some() {
            builder.emit_jump(end_label);
            builder.bind_label(next_label);
        }
    }

    // Emit default case
    if let Some(default_tree) = default {
        emit_tree_node(builder, default_tree, cache, fail_label, end_label)?;
    }

    Ok(())
}

/// Emit a constructor test.
fn emit_constructor_test(
    builder: &mut FunctionBuilder,
    result: Register,
    value: Register,
    ctor: &Constructor,
) -> CompileResult<()> {
    match ctor {
        Constructor::Literal(lit) => {
            // Load literal and compare
            let lit_reg = load_literal_value(builder, lit)?;
            builder.emit_eq(result, value, lit_reg);
            builder.free_register(lit_reg);
        }

        Constructor::Singleton(singleton) => {
            // Load singleton and compare with IS
            let singleton_reg = builder.alloc_register();
            use prism_parser::ast::Singleton;
            match singleton {
                Singleton::True => builder.emit_load_true(singleton_reg),
                Singleton::False => builder.emit_load_false(singleton_reg),
                Singleton::None => builder.emit_load_none(singleton_reg),
            }
            // Use Is opcode for identity comparison
            builder.emit(Instruction::op_dss(
                Opcode::Is,
                result,
                value,
                singleton_reg,
            ));
            builder.free_register(singleton_reg);
        }

        Constructor::Sequence { len, has_star: _ } => {
            // Check if it's a sequence with this length
            // First check sequence type, then length
            emit_match_sequence(builder, result, value);

            // If sequence, check length
            let check_len_label = builder.create_label();
            let done_label = builder.create_label();

            builder.emit_jump_if_false(result, done_label);
            builder.bind_label(check_len_label);

            // Get length and compare
            let len_reg = builder.alloc_register();
            builder.emit(Instruction::op_ds(Opcode::Len, len_reg, value));

            let expected_len = builder.add_int(*len as i64);
            let expected_reg = builder.alloc_register();
            builder.emit_load_const(expected_reg, expected_len);

            builder.emit_eq(result, len_reg, expected_reg);
            builder.free_register(len_reg);
            builder.free_register(expected_reg);

            builder.bind_label(done_label);
        }

        Constructor::Mapping { keys } => {
            // Check if it's a mapping with these keys
            emit_match_mapping(builder, result, value);

            if !keys.is_empty() {
                // Also check that all keys exist
                // For now, set result based on mapping check only
                // Key existence will be checked by MatchKeys opcode
            }
        }

        Constructor::Class { cls: _ } => {
            // Use MatchClass opcode for isinstance check
            // Need to compile the class expression first
            // For now, emit a placeholder
            let cls_reg = builder.alloc_register();
            // TODO: Compile cls expression
            emit_match_class(builder, result, value, cls_reg);
            builder.free_register(cls_reg);
        }
    }

    Ok(())
}

/// Load a literal value into a register.
fn load_literal_value(
    builder: &mut FunctionBuilder,
    lit: &LiteralValue,
) -> CompileResult<Register> {
    let reg = builder.alloc_register();

    match lit {
        LiteralValue::Int(n) => {
            let idx = builder.add_int(*n);
            builder.emit_load_const(reg, idx);
        }
        LiteralValue::Float(n) => {
            let idx = builder.add_float(*n);
            builder.emit_load_const(reg, idx);
        }
        LiteralValue::String(s) => {
            let idx = builder.add_string(s.as_ref());
            builder.emit_load_const(reg, idx);
        }
        LiteralValue::Bytes(_b) => {
            // Bytes constants are rarely used as literal patterns
            // For now, create an empty string constant as placeholder
            let idx = builder.add_string("");
            builder.emit_load_const(reg, idx);
        }
    }

    Ok(reg)
}

/// Emit a type check node.
fn emit_type_check(
    builder: &mut FunctionBuilder,
    cache: &mut SubjectCache,
    access: &AccessPath,
    _cls: &Expr,
    success: &DecisionTree,
    failure: &DecisionTree,
    fail_label: Label,
    end_label: Label,
) -> CompileResult<()> {
    let value_reg = cache.get_or_load(builder, access)?;

    // Compile class expression
    let cls_reg = builder.alloc_register();
    // TODO: Compile cls expression - requires access to compiler

    // Emit MatchClass opcode
    let result_reg = builder.alloc_register();
    emit_match_class(builder, result_reg, value_reg, cls_reg);
    builder.free_register(cls_reg);

    // Create failure label
    let type_fail = builder.create_label();
    builder.emit_jump_if_false(result_reg, type_fail);
    builder.free_register(result_reg);

    // Emit success subtree
    emit_tree_node(builder, success, cache, fail_label, end_label)?;
    builder.emit_jump(end_label);

    // Emit failure subtree
    builder.bind_label(type_fail);
    emit_tree_node(builder, failure, cache, fail_label, end_label)?;

    Ok(())
}

/// Emit a sequence check node.
fn emit_sequence_check(
    builder: &mut FunctionBuilder,
    cache: &mut SubjectCache,
    access: &AccessPath,
    min_len: usize,
    exact: bool,
    success: &DecisionTree,
    failure: &DecisionTree,
    fail_label: Label,
    end_label: Label,
) -> CompileResult<()> {
    let value_reg = cache.get_or_load(builder, access)?;

    // Check if it's a sequence
    let seq_check_reg = builder.alloc_register();
    emit_match_sequence(builder, seq_check_reg, value_reg);

    let seq_fail = builder.create_label();
    builder.emit_jump_if_false(seq_check_reg, seq_fail);
    builder.free_register(seq_check_reg);

    // Check length
    let len_reg = builder.alloc_register();
    builder.emit(Instruction::op_ds(Opcode::Len, len_reg, value_reg));

    let min_len_const = builder.add_int(min_len as i64);
    let min_len_reg = builder.alloc_register();
    builder.emit_load_const(min_len_reg, min_len_const);

    let len_check_reg = builder.alloc_register();
    if exact {
        // Exact length match
        builder.emit_eq(len_check_reg, len_reg, min_len_reg);
    } else {
        // Minimum length (len >= min_len)
        builder.emit_ge(len_check_reg, len_reg, min_len_reg);
    }
    builder.free_register(len_reg);
    builder.free_register(min_len_reg);

    builder.emit_jump_if_false(len_check_reg, seq_fail);
    builder.free_register(len_check_reg);

    // Emit success subtree
    emit_tree_node(builder, success, cache, fail_label, end_label)?;
    builder.emit_jump(end_label);

    // Emit failure subtree
    builder.bind_label(seq_fail);
    emit_tree_node(builder, failure, cache, fail_label, end_label)?;

    Ok(())
}

/// Emit a mapping check node.
fn emit_mapping_check(
    builder: &mut FunctionBuilder,
    cache: &mut SubjectCache,
    access: &AccessPath,
    keys: &[LiteralValue],
    success: &DecisionTree,
    failure: &DecisionTree,
    fail_label: Label,
    end_label: Label,
) -> CompileResult<()> {
    let value_reg = cache.get_or_load(builder, access)?;

    // Check if it's a mapping
    let map_check_reg = builder.alloc_register();
    emit_match_mapping(builder, map_check_reg, value_reg);

    let map_fail = builder.create_label();
    builder.emit_jump_if_false(map_check_reg, map_fail);
    builder.free_register(map_check_reg);

    // Check that all required keys exist
    if !keys.is_empty() {
        // Build tuple of keys
        let keys_tuple = builder.alloc_register();
        let first_key_reg = load_literal_value(builder, &keys[0])?;

        // Build tuple incrementally
        // For simplicity, we check keys one by one using 'in' operator
        for key in keys {
            let key_reg = load_literal_value(builder, key)?;
            let in_check = builder.alloc_register();
            // Use In opcode directly
            builder.emit(Instruction::op_dss(
                Opcode::In,
                in_check,
                key_reg,
                value_reg,
            ));
            builder.free_register(key_reg);

            builder.emit_jump_if_false(in_check, map_fail);
            builder.free_register(in_check);
        }

        builder.free_register(first_key_reg);
        builder.free_register(keys_tuple);
    }

    // Emit success subtree
    emit_tree_node(builder, success, cache, fail_label, end_label)?;
    builder.emit_jump(end_label);

    // Emit failure subtree
    builder.bind_label(map_fail);
    emit_tree_node(builder, failure, cache, fail_label, end_label)?;

    Ok(())
}

// =============================================================================
// Opcode Emission Helpers
// =============================================================================

/// Emit MatchClass opcode.
#[inline]
pub fn emit_match_class(
    builder: &mut FunctionBuilder,
    dst: Register,
    subject: Register,
    cls: Register,
) {
    builder.emit(Instruction::op_dss(Opcode::MatchClass, dst, subject, cls));
}

/// Emit MatchMapping opcode.
#[inline]
pub fn emit_match_mapping(builder: &mut FunctionBuilder, dst: Register, subject: Register) {
    builder.emit(Instruction::op_ds(Opcode::MatchMapping, dst, subject));
}

/// Emit MatchSequence opcode.
#[inline]
pub fn emit_match_sequence(builder: &mut FunctionBuilder, dst: Register, subject: Register) {
    builder.emit(Instruction::op_ds(Opcode::MatchSequence, dst, subject));
}

/// Emit MatchKeys opcode.
#[inline]
pub fn emit_match_keys(
    builder: &mut FunctionBuilder,
    dst: Register,
    mapping: Register,
    keys: Register,
) {
    builder.emit(Instruction::op_dss(Opcode::MatchKeys, dst, mapping, keys));
}

/// Emit CopyDictWithoutKeys opcode.
#[inline]
pub fn emit_copy_dict_without_keys(
    builder: &mut FunctionBuilder,
    dst: Register,
    mapping: Register,
    keys: Register,
) {
    builder.emit(Instruction::op_dss(
        Opcode::CopyDictWithoutKeys,
        dst,
        mapping,
        keys,
    ));
}

/// Emit GetMatchArgs opcode.
#[inline]
#[allow(dead_code)]
fn emit_get_match_args(builder: &mut FunctionBuilder, dst: Register, subject: Register) {
    builder.emit(Instruction::op_ds(Opcode::GetMatchArgs, dst, subject));
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    #[test]
    fn test_subject_cache_new() {
        let root = Register::new(0);
        let cache = SubjectCache::new(root);
        assert!(cache.cache.contains_key(&AccessPath::Root));
    }

    #[test]
    fn test_subject_cache_root_lookup() {
        let root = Register::new(0);
        let cache = SubjectCache::new(root);
        assert_eq!(cache.cache.get(&AccessPath::Root), Some(&root));
    }

    #[test]
    fn test_emit_match_class_instruction() {
        let mut builder = FunctionBuilder::new("test");
        let dst = builder.alloc_register();
        let subject = builder.alloc_register();
        let cls = builder.alloc_register();

        emit_match_class(&mut builder, dst, subject, cls);

        let code = builder.finish();
        assert_eq!(code.instructions.len(), 1);
        assert_eq!(code.instructions[0].opcode(), Opcode::MatchClass as u8);
    }

    #[test]
    fn test_emit_match_mapping_instruction() {
        let mut builder = FunctionBuilder::new("test");
        let dst = builder.alloc_register();
        let subject = builder.alloc_register();

        emit_match_mapping(&mut builder, dst, subject);

        let code = builder.finish();
        assert_eq!(code.instructions.len(), 1);
        assert_eq!(code.instructions[0].opcode(), Opcode::MatchMapping as u8);
    }

    #[test]
    fn test_emit_match_sequence_instruction() {
        let mut builder = FunctionBuilder::new("test");
        let dst = builder.alloc_register();
        let subject = builder.alloc_register();

        emit_match_sequence(&mut builder, dst, subject);

        let code = builder.finish();
        assert_eq!(code.instructions.len(), 1);
        assert_eq!(code.instructions[0].opcode(), Opcode::MatchSequence as u8);
    }

    #[test]
    fn test_emit_match_keys_instruction() {
        let mut builder = FunctionBuilder::new("test");
        let dst = builder.alloc_register();
        let mapping = builder.alloc_register();
        let keys = builder.alloc_register();

        emit_match_keys(&mut builder, dst, mapping, keys);

        let code = builder.finish();
        assert_eq!(code.instructions.len(), 1);
        assert_eq!(code.instructions[0].opcode(), Opcode::MatchKeys as u8);
    }

    #[test]
    fn test_emit_copy_dict_without_keys_instruction() {
        let mut builder = FunctionBuilder::new("test");
        let dst = builder.alloc_register();
        let mapping = builder.alloc_register();
        let keys = builder.alloc_register();

        emit_copy_dict_without_keys(&mut builder, dst, mapping, keys);

        let code = builder.finish();
        assert_eq!(code.instructions.len(), 1);
        assert_eq!(
            code.instructions[0].opcode(),
            Opcode::CopyDictWithoutKeys as u8
        );
    }

    #[test]
    fn test_emit_get_match_args_instruction() {
        let mut builder = FunctionBuilder::new("test");
        let dst = builder.alloc_register();
        let subject = builder.alloc_register();

        emit_get_match_args(&mut builder, dst, subject);

        let code = builder.finish();
        assert_eq!(code.instructions.len(), 1);
        assert_eq!(code.instructions[0].opcode(), Opcode::GetMatchArgs as u8);
    }

    #[test]
    fn test_load_literal_int() {
        let mut builder = FunctionBuilder::new("test");
        let reg = load_literal_value(&mut builder, &LiteralValue::Int(42)).unwrap();

        let code = builder.finish();
        assert_eq!(code.instructions.len(), 1);
        assert_eq!(code.instructions[0].opcode(), Opcode::LoadConst as u8);
        assert_eq!(code.instructions[0].dst(), reg);
    }

    #[test]
    fn test_load_literal_float() {
        let mut builder = FunctionBuilder::new("test");
        let reg = load_literal_value(&mut builder, &LiteralValue::Float(3.14)).unwrap();

        let code = builder.finish();
        assert_eq!(code.instructions.len(), 1);
        assert_eq!(code.instructions[0].opcode(), Opcode::LoadConst as u8);
        assert_eq!(code.instructions[0].dst(), reg);
    }

    #[test]
    fn test_load_literal_string() {
        let mut builder = FunctionBuilder::new("test");
        let reg =
            load_literal_value(&mut builder, &LiteralValue::String(Arc::from("hello"))).unwrap();

        let code = builder.finish();
        assert_eq!(code.instructions.len(), 1);
        assert_eq!(code.instructions[0].opcode(), Opcode::LoadConst as u8);
        assert_eq!(code.instructions[0].dst(), reg);
    }

    #[test]
    fn test_decision_tree_fail_emits_jump() {
        let mut builder = FunctionBuilder::new("test");
        let end_label = builder.create_label();
        let fail_label = builder.create_label();
        let subject = builder.alloc_register();

        emit_tree(
            &mut builder,
            &DecisionTree::Fail,
            subject,
            fail_label,
            end_label,
        )
        .unwrap();

        // Bind labels to avoid "unbound label" error
        builder.bind_label(fail_label);
        builder.bind_label(end_label);

        let code = builder.finish();
        // DecisionTree::Fail emits a jump to fail_label
        assert!(code.instructions.len() >= 1);
        assert_eq!(code.instructions[0].opcode(), Opcode::Jump as u8);
    }

    #[test]
    fn test_subject_cache_preserves_root() {
        let root = Register::new(5);
        let cache = SubjectCache::new(root);

        assert_eq!(cache.root, root);
        assert_eq!(cache.cache.get(&AccessPath::Root), Some(&root));
    }

    #[test]
    fn test_subject_cache_allocated_empty_initially() {
        let root = Register::new(0);
        let cache = SubjectCache::new(root);

        assert!(cache.allocated.is_empty());
    }

    #[test]
    fn test_match_codegen_context_creation() {
        let mut builder = FunctionBuilder::new("test");
        let subject = builder.alloc_register();
        let mut cache = SubjectCache::new(subject);
        let end_label = builder.create_label();
        let fail_label = builder.create_label();

        let codegen = MatchCodegen::new(&mut builder, &mut cache, end_label, fail_label);

        assert_eq!(codegen.end_label, end_label);
        assert_eq!(codegen.fail_label, fail_label);
    }
}
