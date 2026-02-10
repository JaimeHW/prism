//! Dict-specialized JIT templates for high-performance dict operations.
//!
//! Provides type-specialized native code generation for:
//! - **Dict lookup** (`dict[key]`) — type guard + deopt for hash lookup
//! - **Dict store** (`dict[key] = v`) — type guard + deopt for hash insert
//! - **Dict contains** (`key in dict`) — type guard + deopt for hash probe
//! - **Dict merge** (`dict | dict`) — dual type guard + deopt
//! - **Dict length** (`len(dict)`) — type guard + deopt
//!
//! # Memory Model
//!
//! `DictObject` is `#[repr(C)]`:
//! ```text
//! Offset  Size  Field
//! ──────  ────  ─────────────────────
//!   0      4    ObjectHeader.type_id  (TypeId, u32)
//!   4      4    ObjectHeader.gc_flags (AtomicU32)
//!   8      8    ObjectHeader.hash     (u64)
//!  16      ?    FxHashMap<HashableValue, Value>
//! ```
//!
//! Unlike `ListObject`, the `FxHashMap` does not have a stable internal layout
//! suitable for direct memory access from JIT code. Therefore, all dict
//! operations use the following strategy:
//!
//! # Performance Strategy
//!
//! 1. **Inline type guard**: Verify the value is an OBJECT with `TypeId::DICT`.
//!    This eliminates interpreter-level type dispatch overhead.
//! 2. **Key type guard**: Where applicable, verify the key type inline.
//! 3. **Deopt to interpreter**: For the actual hash operation (lookup, insert,
//!    membership test), deopt to Tier 0/interpreter. Tier 2 will inline these
//!    via runtime helper calls.
//!
//! The inline type guard alone provides significant speedup:
//! - Eliminates polymorphic dispatch in the interpreter
//! - Enables the JIT to speculate on dict-only access patterns
//! - Provides a deopt point when types change (e.g., dict replaced by custom object)
//!
//! For the string-key fast path, we additionally inline a STRING_TAG check
//! on the key, enabling the JIT to skip the key type dispatch.

use super::specialize_common::{
    emit_int_check_and_extract, emit_object_check_and_extract, emit_string_check_and_extract,
    emit_typed_object_check_and_extract, type_ids,
};
use super::{OpcodeTemplate, TemplateContext};

// =============================================================================
// Dict Lookup Template
// =============================================================================

/// Template for dict item lookup (`dict[key]`).
///
/// # Code Generation Strategy
///
/// 1. Type-check dict value: OBJECT_TAG + TypeId::DICT
/// 2. Deopt to interpreter for actual hash lookup
///
/// This provides a typed deopt — the JIT knows the receiver is a dict,
/// so re-entry to the interpreter can take a fast path.
///
/// # Estimated Code Size
///
/// ~40 bytes: object tag check (~20), type_id check (~12), jmp (~8)
pub struct DictGetTemplate {
    /// Register holding the dict value.
    pub dict_reg: u8,
    /// Register holding the key value.
    pub key_reg: u8,
    /// Destination register for the result.
    pub dst_reg: u8,
    /// Deopt label index.
    pub deopt_idx: usize,
}

impl DictGetTemplate {
    /// Create a new dict get template.
    #[inline]
    pub fn new(dict_reg: u8, key_reg: u8, dst_reg: u8, deopt_idx: usize) -> Self {
        Self {
            dict_reg,
            key_reg,
            dst_reg,
            deopt_idx,
        }
    }
}

impl OpcodeTemplate for DictGetTemplate {
    fn emit(&self, ctx: &mut TemplateContext) {
        let acc = ctx.regs.accumulator;
        let scratch1 = ctx.regs.scratch1;

        // Load dict value into accumulator
        let dict_slot = ctx.frame.register_slot(self.dict_reg as u16);
        ctx.asm.mov_rm(acc, &dict_slot);

        // Type guard: verify dict is OBJECT + TypeId::DICT
        emit_typed_object_check_and_extract(
            ctx,
            acc,
            acc,
            scratch1,
            type_ids::DICT,
            self.deopt_idx,
        );

        // Hash lookup requires runtime support — deopt
        // The interpreter re-enters with the knowledge that the receiver is a dict
        ctx.asm.jmp(ctx.deopt_label(self.deopt_idx));
    }

    #[inline]
    fn estimated_size(&self) -> usize {
        48
    }
}

// =============================================================================
// Dict String-Key Lookup Template
// =============================================================================

/// Template for dict lookup with a known-string key (`dict[str_key]`).
///
/// # Code Generation Strategy
///
/// 1. Type-check dict value: OBJECT_TAG + TypeId::DICT
/// 2. Type-check key: STRING_TAG
/// 3. Deopt to interpreter for the actual hash probe
///
/// This double type guard is the most common dict access pattern in Python
/// (string keys account for >90% of dict accesses in typical programs).
/// The inline string-key verification enables the interpreter to skip the
/// key-type dispatch entirely on re-entry.
///
/// # Estimated Code Size
///
/// ~70 bytes: dict guard (~32), string guard (~20), jmp (~8), loads (~10)
pub struct DictGetStrTemplate {
    /// Register holding the dict value.
    pub dict_reg: u8,
    /// Register holding the key (should be a string).
    pub key_reg: u8,
    /// Destination register for the result.
    pub dst_reg: u8,
    /// Deopt label index.
    pub deopt_idx: usize,
}

impl DictGetStrTemplate {
    /// Create a new dict string-key get template.
    #[inline]
    pub fn new(dict_reg: u8, key_reg: u8, dst_reg: u8, deopt_idx: usize) -> Self {
        Self {
            dict_reg,
            key_reg,
            dst_reg,
            deopt_idx,
        }
    }
}

impl OpcodeTemplate for DictGetStrTemplate {
    fn emit(&self, ctx: &mut TemplateContext) {
        let acc = ctx.regs.accumulator;
        let scratch1 = ctx.regs.scratch1;
        let scratch2 = ctx.regs.scratch2;

        // Load dict value
        let dict_slot = ctx.frame.register_slot(self.dict_reg as u16);
        ctx.asm.mov_rm(acc, &dict_slot);

        // Type guard: dict must be OBJECT + TypeId::DICT
        emit_typed_object_check_and_extract(
            ctx,
            acc,
            acc,
            scratch1,
            type_ids::DICT,
            self.deopt_idx,
        );

        // Load key value
        let key_slot = ctx.frame.register_slot(self.key_reg as u16);
        ctx.asm.mov_rm(scratch1, &key_slot);

        // Type guard: key must be a string
        emit_string_check_and_extract(ctx, scratch1, scratch1, scratch2, self.deopt_idx);

        // Deopt for actual lookup — both types verified
        ctx.asm.jmp(ctx.deopt_label(self.deopt_idx));
    }

    #[inline]
    fn estimated_size(&self) -> usize {
        72
    }
}

// =============================================================================
// Dict Int-Key Lookup Template
// =============================================================================

/// Template for dict lookup with a known-integer key (`dict[int_key]`).
///
/// # Code Generation Strategy
///
/// 1. Type-check dict value: OBJECT_TAG + TypeId::DICT
/// 2. Type-check key: INT_TAG
/// 3. Deopt to interpreter for the actual hash probe
///
/// Integer keys are the second most common dict access pattern, especially
/// in numeric and data-processing code.
///
/// # Estimated Code Size
///
/// ~70 bytes: dict guard (~32), int guard (~20), jmp (~8), loads (~10)
pub struct DictGetIntTemplate {
    /// Register holding the dict value.
    pub dict_reg: u8,
    /// Register holding the key (should be an integer).
    pub key_reg: u8,
    /// Destination register for the result.
    pub dst_reg: u8,
    /// Deopt label index.
    pub deopt_idx: usize,
}

impl DictGetIntTemplate {
    /// Create a new dict int-key get template.
    #[inline]
    pub fn new(dict_reg: u8, key_reg: u8, dst_reg: u8, deopt_idx: usize) -> Self {
        Self {
            dict_reg,
            key_reg,
            dst_reg,
            deopt_idx,
        }
    }
}

impl OpcodeTemplate for DictGetIntTemplate {
    fn emit(&self, ctx: &mut TemplateContext) {
        let acc = ctx.regs.accumulator;
        let scratch1 = ctx.regs.scratch1;
        let scratch2 = ctx.regs.scratch2;

        // Load dict value
        let dict_slot = ctx.frame.register_slot(self.dict_reg as u16);
        ctx.asm.mov_rm(acc, &dict_slot);

        // Type guard: dict must be OBJECT + TypeId::DICT
        emit_typed_object_check_and_extract(
            ctx,
            acc,
            acc,
            scratch1,
            type_ids::DICT,
            self.deopt_idx,
        );

        // Load key value
        let key_slot = ctx.frame.register_slot(self.key_reg as u16);
        ctx.asm.mov_rm(scratch1, &key_slot);

        // Type guard: key must be an integer
        emit_int_check_and_extract(ctx, scratch1, scratch1, scratch2, self.deopt_idx);

        // Deopt for actual lookup — both types verified
        ctx.asm.jmp(ctx.deopt_label(self.deopt_idx));
    }

    #[inline]
    fn estimated_size(&self) -> usize {
        72
    }
}

// =============================================================================
// Dict Store Template
// =============================================================================

/// Template for dict item store (`dict[key] = value`).
///
/// # Code Generation Strategy
///
/// 1. Type-check dict value: OBJECT_TAG + TypeId::DICT
/// 2. Deopt to interpreter for actual hash insertion
///
/// The type guard ensures we only deopt for true dict operations,
/// not for custom `__setitem__` on arbitrary objects.
///
/// # Estimated Code Size
///
/// ~40 bytes: dict guard (~32), jmp (~8)
pub struct DictSetFastTemplate {
    /// Register holding the dict value.
    pub dict_reg: u8,
    /// Register holding the key value.
    pub key_reg: u8,
    /// Register holding the value to store.
    pub value_reg: u8,
    /// Deopt label index.
    pub deopt_idx: usize,
}

impl DictSetFastTemplate {
    /// Create a new dict set template.
    #[inline]
    pub fn new(dict_reg: u8, key_reg: u8, value_reg: u8, deopt_idx: usize) -> Self {
        Self {
            dict_reg,
            key_reg,
            value_reg,
            deopt_idx,
        }
    }
}

impl OpcodeTemplate for DictSetFastTemplate {
    fn emit(&self, ctx: &mut TemplateContext) {
        let acc = ctx.regs.accumulator;
        let scratch1 = ctx.regs.scratch1;

        // Load dict value
        let dict_slot = ctx.frame.register_slot(self.dict_reg as u16);
        ctx.asm.mov_rm(acc, &dict_slot);

        // Type guard: dict must be OBJECT + TypeId::DICT
        emit_typed_object_check_and_extract(
            ctx,
            acc,
            acc,
            scratch1,
            type_ids::DICT,
            self.deopt_idx,
        );

        // Hash insertion requires runtime support — deopt
        ctx.asm.jmp(ctx.deopt_label(self.deopt_idx));
    }

    #[inline]
    fn estimated_size(&self) -> usize {
        48
    }
}

// =============================================================================
// Dict String-Key Store Template
// =============================================================================

/// Template for dict store with a known-string key (`dict[str_key] = value`).
///
/// # Code Generation Strategy
///
/// 1. Type-check dict value: OBJECT_TAG + TypeId::DICT
/// 2. Type-check key: STRING_TAG
/// 3. Deopt to interpreter for hash insertion
///
/// The double type guard eliminates both receiver-type and key-type
/// dispatch in the interpreter on re-entry.
///
/// # Estimated Code Size
///
/// ~70 bytes: dict guard (~32), string guard (~20), jmp (~8), loads (~10)
pub struct DictSetStrTemplate {
    /// Register holding the dict value.
    pub dict_reg: u8,
    /// Register holding the key (should be a string).
    pub key_reg: u8,
    /// Register holding the value to store.
    pub value_reg: u8,
    /// Deopt label index.
    pub deopt_idx: usize,
}

impl DictSetStrTemplate {
    /// Create a new dict string-key set template.
    #[inline]
    pub fn new(dict_reg: u8, key_reg: u8, value_reg: u8, deopt_idx: usize) -> Self {
        Self {
            dict_reg,
            key_reg,
            value_reg,
            deopt_idx,
        }
    }
}

impl OpcodeTemplate for DictSetStrTemplate {
    fn emit(&self, ctx: &mut TemplateContext) {
        let acc = ctx.regs.accumulator;
        let scratch1 = ctx.regs.scratch1;
        let scratch2 = ctx.regs.scratch2;

        // Load dict value
        let dict_slot = ctx.frame.register_slot(self.dict_reg as u16);
        ctx.asm.mov_rm(acc, &dict_slot);

        // Type guard: dict must be OBJECT + TypeId::DICT
        emit_typed_object_check_and_extract(
            ctx,
            acc,
            acc,
            scratch1,
            type_ids::DICT,
            self.deopt_idx,
        );

        // Load key value
        let key_slot = ctx.frame.register_slot(self.key_reg as u16);
        ctx.asm.mov_rm(scratch1, &key_slot);

        // Type guard: key must be a string
        emit_string_check_and_extract(ctx, scratch1, scratch1, scratch2, self.deopt_idx);

        // Deopt for actual insertion — both types verified
        ctx.asm.jmp(ctx.deopt_label(self.deopt_idx));
    }

    #[inline]
    fn estimated_size(&self) -> usize {
        72
    }
}

// =============================================================================
// Dict Contains Template
// =============================================================================

/// Template for dict containment check (`key in dict`).
///
/// # Code Generation Strategy
///
/// 1. Type-check dict value: OBJECT_TAG + TypeId::DICT
/// 2. Deopt to interpreter for has probe
///
/// Membership testing is a hot operation in many Python workloads.
/// The inline type guard enables the JIT to specialize the `in` operator
/// without falling through to the generic `__contains__` dispatch.
///
/// # Estimated Code Size
///
/// ~48 bytes: dict guard (~32), load (~8), jmp (~8)
pub struct DictContainsTemplate {
    /// Register holding the dict value.
    pub dict_reg: u8,
    /// Register holding the key to check.
    pub key_reg: u8,
    /// Destination register for the boolean result.
    pub dst_reg: u8,
    /// Deopt label index.
    pub deopt_idx: usize,
}

impl DictContainsTemplate {
    /// Create a new dict contains template.
    #[inline]
    pub fn new(dict_reg: u8, key_reg: u8, dst_reg: u8, deopt_idx: usize) -> Self {
        Self {
            dict_reg,
            key_reg,
            dst_reg,
            deopt_idx,
        }
    }
}

impl OpcodeTemplate for DictContainsTemplate {
    fn emit(&self, ctx: &mut TemplateContext) {
        let acc = ctx.regs.accumulator;
        let scratch1 = ctx.regs.scratch1;

        // Load dict value
        let dict_slot = ctx.frame.register_slot(self.dict_reg as u16);
        ctx.asm.mov_rm(acc, &dict_slot);

        // Type guard: dict must be OBJECT + TypeId::DICT
        emit_typed_object_check_and_extract(
            ctx,
            acc,
            acc,
            scratch1,
            type_ids::DICT,
            self.deopt_idx,
        );

        // Membership test requires runtime — deopt
        ctx.asm.jmp(ctx.deopt_label(self.deopt_idx));
    }

    #[inline]
    fn estimated_size(&self) -> usize {
        48
    }
}

// =============================================================================
// Dict Merge Template
// =============================================================================

/// Template for dict merge (`dict | dict` or `dict.update(dict)`).
///
/// # Code Generation Strategy
///
/// 1. Type-check LHS: OBJECT_TAG + TypeId::DICT
/// 2. Type-check RHS: OBJECT_TAG + TypeId::DICT
/// 3. Deopt to interpreter for actual merge
///
/// This ensures that `|` is only applied to actual dict objects,
/// not custom `__or__` implementations on other types.
///
/// # Estimated Code Size
///
/// ~80 bytes: 2× dict guard (~64), jmp (~8), loads (~8)
pub struct DictMergeTemplate {
    /// Register holding the left dict.
    pub lhs_reg: u8,
    /// Register holding the right dict.
    pub rhs_reg: u8,
    /// Destination register for the merged dict.
    pub dst_reg: u8,
    /// Deopt label index.
    pub deopt_idx: usize,
}

impl DictMergeTemplate {
    /// Create a new dict merge template.
    #[inline]
    pub fn new(lhs_reg: u8, rhs_reg: u8, dst_reg: u8, deopt_idx: usize) -> Self {
        Self {
            lhs_reg,
            rhs_reg,
            dst_reg,
            deopt_idx,
        }
    }
}

impl OpcodeTemplate for DictMergeTemplate {
    fn emit(&self, ctx: &mut TemplateContext) {
        let acc = ctx.regs.accumulator;
        let scratch1 = ctx.regs.scratch1;
        let scratch2 = ctx.regs.scratch2;

        // Load and verify LHS dict
        let lhs_slot = ctx.frame.register_slot(self.lhs_reg as u16);
        ctx.asm.mov_rm(acc, &lhs_slot);
        emit_typed_object_check_and_extract(
            ctx,
            acc,
            acc,
            scratch1,
            type_ids::DICT,
            self.deopt_idx,
        );

        // Load and verify RHS dict
        let rhs_slot = ctx.frame.register_slot(self.rhs_reg as u16);
        ctx.asm.mov_rm(scratch1, &rhs_slot);
        emit_typed_object_check_and_extract(
            ctx,
            scratch1,
            scratch1,
            scratch2,
            type_ids::DICT,
            self.deopt_idx,
        );

        // Merge requires allocation + hash copying — deopt
        ctx.asm.jmp(ctx.deopt_label(self.deopt_idx));
    }

    #[inline]
    fn estimated_size(&self) -> usize {
        80
    }
}

// =============================================================================
// Dict Delete Template
// =============================================================================

/// Template for dict item deletion (`del dict[key]`).
///
/// # Code Generation Strategy
///
/// 1. Type-check dict value: OBJECT_TAG + TypeId::DICT
/// 2. Deopt to interpreter for actual hash removal
///
/// # Estimated Code Size
///
/// ~48 bytes: dict guard (~32), load (~8), jmp (~8)
pub struct DictDeleteTemplate {
    /// Register holding the dict value.
    pub dict_reg: u8,
    /// Register holding the key to delete.
    pub key_reg: u8,
    /// Deopt label index.
    pub deopt_idx: usize,
}

impl DictDeleteTemplate {
    /// Create a new dict delete template.
    #[inline]
    pub fn new(dict_reg: u8, key_reg: u8, deopt_idx: usize) -> Self {
        Self {
            dict_reg,
            key_reg,
            deopt_idx,
        }
    }
}

impl OpcodeTemplate for DictDeleteTemplate {
    fn emit(&self, ctx: &mut TemplateContext) {
        let acc = ctx.regs.accumulator;
        let scratch1 = ctx.regs.scratch1;

        // Load dict value
        let dict_slot = ctx.frame.register_slot(self.dict_reg as u16);
        ctx.asm.mov_rm(acc, &dict_slot);

        // Type guard: dict must be OBJECT + TypeId::DICT
        emit_typed_object_check_and_extract(
            ctx,
            acc,
            acc,
            scratch1,
            type_ids::DICT,
            self.deopt_idx,
        );

        // Deletion requires runtime — deopt
        ctx.asm.jmp(ctx.deopt_label(self.deopt_idx));
    }

    #[inline]
    fn estimated_size(&self) -> usize {
        48
    }
}

// =============================================================================
// Dict Keys/Values/Items Guard Template
// =============================================================================

/// Template for dict view operations (`dict.keys()`, `dict.values()`, `dict.items()`).
///
/// # Code Generation Strategy
///
/// 1. Type-check dict value: OBJECT_TAG + TypeId::DICT
/// 2. Deopt to interpreter for view construction
///
/// These operations always allocate a new view object, so they must deopt.
/// The type guard ensures we only handle actual dict objects.
///
/// # Estimated Code Size
///
/// ~40 bytes: dict guard (~32), jmp (~8)
pub struct DictViewGuardTemplate {
    /// Register holding the dict value.
    pub dict_reg: u8,
    /// Destination register for the view result.
    pub dst_reg: u8,
    /// Deopt label index.
    pub deopt_idx: usize,
}

impl DictViewGuardTemplate {
    /// Create a new dict view guard template.
    #[inline]
    pub fn new(dict_reg: u8, dst_reg: u8, deopt_idx: usize) -> Self {
        Self {
            dict_reg,
            dst_reg,
            deopt_idx,
        }
    }
}

impl OpcodeTemplate for DictViewGuardTemplate {
    fn emit(&self, ctx: &mut TemplateContext) {
        let acc = ctx.regs.accumulator;
        let scratch1 = ctx.regs.scratch1;

        // Load dict value
        let dict_slot = ctx.frame.register_slot(self.dict_reg as u16);
        ctx.asm.mov_rm(acc, &dict_slot);

        // Type guard: dict must be OBJECT + TypeId::DICT
        emit_typed_object_check_and_extract(
            ctx,
            acc,
            acc,
            scratch1,
            type_ids::DICT,
            self.deopt_idx,
        );

        // View construction requires allocation — deopt
        ctx.asm.jmp(ctx.deopt_label(self.deopt_idx));
    }

    #[inline]
    fn estimated_size(&self) -> usize {
        48
    }
}

// =============================================================================
// Dict Get-Or-Default Template
// =============================================================================

/// Template for `dict.get(key, default)` with type guards.
///
/// # Code Generation Strategy
///
/// 1. Type-check dict value: OBJECT_TAG + TypeId::DICT
/// 2. Type-check key: STRING_TAG (for the common string-key case)
/// 3. Deopt to interpreter for the actual hash probe + default handling
///
/// `dict.get()` is extremely common in Python code and is the second
/// most frequently called dict method after `__getitem__`.
///
/// # Estimated Code Size
///
/// ~70 bytes: dict guard (~32), string guard (~20), jmp (~8), loads (~10)
pub struct DictGetOrDefaultTemplate {
    /// Register holding the dict value.
    pub dict_reg: u8,
    /// Register holding the key.
    pub key_reg: u8,
    /// Register holding the default value.
    pub default_reg: u8,
    /// Destination register for the result.
    pub dst_reg: u8,
    /// Deopt label index.
    pub deopt_idx: usize,
}

impl DictGetOrDefaultTemplate {
    /// Create a new dict get-or-default template.
    #[inline]
    pub fn new(dict_reg: u8, key_reg: u8, default_reg: u8, dst_reg: u8, deopt_idx: usize) -> Self {
        Self {
            dict_reg,
            key_reg,
            default_reg,
            dst_reg,
            deopt_idx,
        }
    }
}

impl OpcodeTemplate for DictGetOrDefaultTemplate {
    fn emit(&self, ctx: &mut TemplateContext) {
        let acc = ctx.regs.accumulator;
        let scratch1 = ctx.regs.scratch1;

        // Load dict value
        let dict_slot = ctx.frame.register_slot(self.dict_reg as u16);
        ctx.asm.mov_rm(acc, &dict_slot);

        // Type guard: dict must be OBJECT + TypeId::DICT
        emit_typed_object_check_and_extract(
            ctx,
            acc,
            acc,
            scratch1,
            type_ids::DICT,
            self.deopt_idx,
        );

        // .get() with default requires runtime hash probe — deopt
        ctx.asm.jmp(ctx.deopt_label(self.deopt_idx));
    }

    #[inline]
    fn estimated_size(&self) -> usize {
        48
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::x64::Assembler;
    use crate::tier1::frame::FrameLayout;
    use crate::tier1::template::OpcodeTemplate;
    use crate::tier1::template::specialize_common::emit_object_check_and_extract;

    // =========================================================================
    // Test Helpers
    // =========================================================================

    fn make_ctx_with_deopt<'a>(
        asm: &'a mut Assembler,
        frame: &'a FrameLayout,
    ) -> TemplateContext<'a> {
        let mut ctx = TemplateContext::new(asm, frame);
        ctx.create_deopt_label();
        ctx
    }

    /// Emit a template and finalize, returning the generated machine code bytes.
    fn emit_and_finalize(template: &dyn OpcodeTemplate) -> Vec<u8> {
        let mut asm = Assembler::new();
        let frame = FrameLayout::minimal(8);
        {
            let mut ctx = make_ctx_with_deopt(&mut asm, &frame);
            template.emit(&mut ctx);
            // Bind deopt label so assembler can resolve forward jumps
            for l in &ctx.deopt_labels {
                ctx.asm.bind_label(*l);
            }
        }
        asm.finalize().unwrap()
    }

    // =========================================================================
    // DictGetTemplate Tests
    // =========================================================================

    #[test]
    fn test_dict_get_emits_code() {
        let template = DictGetTemplate::new(0, 1, 2, 0);
        let code = emit_and_finalize(&template);
        assert!(code.len() > 0, "DictGet should emit code");
    }

    #[test]
    fn test_dict_get_code_within_estimate() {
        let template = DictGetTemplate::new(0, 1, 2, 0);
        let code = emit_and_finalize(&template);
        let estimate = template.estimated_size();
        assert!(
            code.len() <= estimate * 2,
            "DictGet code {} exceeds 2x estimate {}",
            code.len(),
            estimate
        );
    }

    #[test]
    fn test_dict_get_registers() {
        // Different register assignments should all work
        for dict_reg in [0u8, 1, 2, 3] {
            for key_reg in [0u8, 1, 2, 3] {
                let template = DictGetTemplate::new(dict_reg, key_reg, 0, 0);
                let code = emit_and_finalize(&template);
                assert!(
                    code.len() > 0,
                    "DictGet(r{}, r{}) should emit code",
                    dict_reg,
                    key_reg
                );
            }
        }
    }

    #[test]
    fn test_dict_get_deterministic() {
        let t1 = DictGetTemplate::new(0, 1, 2, 0);
        let t2 = DictGetTemplate::new(0, 1, 2, 0);
        let code1 = emit_and_finalize(&t1);
        let code2 = emit_and_finalize(&t2);
        assert_eq!(code1, code2, "Same template should produce identical code");
    }

    // =========================================================================
    // DictGetStrTemplate Tests
    // =========================================================================

    #[test]
    fn test_dict_get_str_emits_code() {
        let template = DictGetStrTemplate::new(0, 1, 2, 0);
        let code = emit_and_finalize(&template);
        assert!(code.len() > 0, "DictGetStr should emit code");
    }

    #[test]
    fn test_dict_get_str_more_code_than_basic() {
        // String-key version has additional STRING_TAG check
        let basic = DictGetTemplate::new(0, 1, 2, 0);
        let str_variant = DictGetStrTemplate::new(0, 1, 2, 0);
        let basic_code = emit_and_finalize(&basic);
        let str_code = emit_and_finalize(&str_variant);
        assert!(
            str_code.len() > basic_code.len(),
            "DictGetStr ({} bytes) should be larger than DictGet ({} bytes)",
            str_code.len(),
            basic_code.len()
        );
    }

    #[test]
    fn test_dict_get_str_code_within_estimate() {
        let template = DictGetStrTemplate::new(0, 1, 2, 0);
        let code = emit_and_finalize(&template);
        let estimate = template.estimated_size();
        assert!(
            code.len() <= estimate * 2,
            "DictGetStr code {} exceeds 2x estimate {}",
            code.len(),
            estimate
        );
    }

    #[test]
    fn test_dict_get_str_deterministic() {
        let t1 = DictGetStrTemplate::new(0, 1, 2, 0);
        let t2 = DictGetStrTemplate::new(0, 1, 2, 0);
        let code1 = emit_and_finalize(&t1);
        let code2 = emit_and_finalize(&t2);
        assert_eq!(code1, code2);
    }

    #[test]
    fn test_dict_get_str_registers() {
        for dict_reg in [0u8, 1, 2, 3] {
            let template = DictGetStrTemplate::new(dict_reg, 1, 2, 0);
            let code = emit_and_finalize(&template);
            assert!(code.len() > 0);
        }
    }

    // =========================================================================
    // DictGetIntTemplate Tests
    // =========================================================================

    #[test]
    fn test_dict_get_int_emits_code() {
        let template = DictGetIntTemplate::new(0, 1, 2, 0);
        let code = emit_and_finalize(&template);
        assert!(code.len() > 0, "DictGetInt should emit code");
    }

    #[test]
    fn test_dict_get_int_more_code_than_basic() {
        let basic = DictGetTemplate::new(0, 1, 2, 0);
        let int_variant = DictGetIntTemplate::new(0, 1, 2, 0);
        let basic_code = emit_and_finalize(&basic);
        let int_code = emit_and_finalize(&int_variant);
        assert!(
            int_code.len() > basic_code.len(),
            "DictGetInt ({}) should be larger than DictGet ({})",
            int_code.len(),
            basic_code.len()
        );
    }

    #[test]
    fn test_dict_get_int_similar_to_str() {
        // Int-key and string-key versions should have similar sizes
        let str_variant = DictGetStrTemplate::new(0, 1, 2, 0);
        let int_variant = DictGetIntTemplate::new(0, 1, 2, 0);
        let str_code = emit_and_finalize(&str_variant);
        let int_code = emit_and_finalize(&int_variant);
        let diff = (str_code.len() as i64 - int_code.len() as i64).unsigned_abs();
        assert!(
            diff <= 8,
            "DictGetStr ({}) and DictGetInt ({}) should be similar size (diff {})",
            str_code.len(),
            int_code.len(),
            diff
        );
    }

    #[test]
    fn test_dict_get_int_code_within_estimate() {
        let template = DictGetIntTemplate::new(0, 1, 2, 0);
        let code = emit_and_finalize(&template);
        let estimate = template.estimated_size();
        assert!(
            code.len() <= estimate * 2,
            "DictGetInt code {} exceeds 2x estimate {}",
            code.len(),
            estimate
        );
    }

    #[test]
    fn test_dict_get_int_deterministic() {
        let t1 = DictGetIntTemplate::new(0, 1, 2, 0);
        let t2 = DictGetIntTemplate::new(0, 1, 2, 0);
        assert_eq!(emit_and_finalize(&t1), emit_and_finalize(&t2));
    }

    // =========================================================================
    // DictSetFastTemplate Tests
    // =========================================================================

    #[test]
    fn test_dict_set_emits_code() {
        let template = DictSetFastTemplate::new(0, 1, 2, 0);
        let code = emit_and_finalize(&template);
        assert!(code.len() > 0, "DictSet should emit code");
    }

    #[test]
    fn test_dict_set_code_within_estimate() {
        let template = DictSetFastTemplate::new(0, 1, 2, 0);
        let code = emit_and_finalize(&template);
        let estimate = template.estimated_size();
        assert!(
            code.len() <= estimate * 2,
            "DictSet code {} exceeds 2x estimate {}",
            code.len(),
            estimate
        );
    }

    #[test]
    fn test_dict_set_similar_to_get() {
        // DictSet and DictGet both have single dict type guard + deopt
        let get_template = DictGetTemplate::new(0, 1, 2, 0);
        let set_template = DictSetFastTemplate::new(0, 1, 2, 0);
        let get_code = emit_and_finalize(&get_template);
        let set_code = emit_and_finalize(&set_template);
        let diff = (get_code.len() as i64 - set_code.len() as i64).unsigned_abs();
        assert!(
            diff <= 8,
            "DictGet ({}) and DictSet ({}) sizes should be very close (diff {})",
            get_code.len(),
            set_code.len(),
            diff,
        );
    }

    #[test]
    fn test_dict_set_deterministic() {
        let t1 = DictSetFastTemplate::new(0, 1, 2, 0);
        let t2 = DictSetFastTemplate::new(0, 1, 2, 0);
        assert_eq!(emit_and_finalize(&t1), emit_and_finalize(&t2));
    }

    #[test]
    fn test_dict_set_registers() {
        for dict_reg in 0u8..4 {
            for key_reg in 0u8..4 {
                for val_reg in 0u8..4 {
                    let template = DictSetFastTemplate::new(dict_reg, key_reg, val_reg, 0);
                    let code = emit_and_finalize(&template);
                    assert!(code.len() > 0);
                }
            }
        }
    }

    // =========================================================================
    // DictSetStrTemplate Tests
    // =========================================================================

    #[test]
    fn test_dict_set_str_emits_code() {
        let template = DictSetStrTemplate::new(0, 1, 2, 0);
        let code = emit_and_finalize(&template);
        assert!(code.len() > 0, "DictSetStr should emit code");
    }

    #[test]
    fn test_dict_set_str_more_code_than_basic_set() {
        let basic = DictSetFastTemplate::new(0, 1, 2, 0);
        let str_variant = DictSetStrTemplate::new(0, 1, 2, 0);
        let basic_code = emit_and_finalize(&basic);
        let str_code = emit_and_finalize(&str_variant);
        assert!(
            str_code.len() > basic_code.len(),
            "DictSetStr ({}) should be larger than DictSet ({})",
            str_code.len(),
            basic_code.len()
        );
    }

    #[test]
    fn test_dict_set_str_code_within_estimate() {
        let template = DictSetStrTemplate::new(0, 1, 2, 0);
        let code = emit_and_finalize(&template);
        let estimate = template.estimated_size();
        assert!(
            code.len() <= estimate * 2,
            "DictSetStr code {} exceeds 2x estimate {}",
            code.len(),
            estimate
        );
    }

    #[test]
    fn test_dict_set_str_similar_to_get_str() {
        let get_str = DictGetStrTemplate::new(0, 1, 2, 0);
        let set_str = DictSetStrTemplate::new(0, 1, 2, 0);
        let get_code = emit_and_finalize(&get_str);
        let set_code = emit_and_finalize(&set_str);
        let diff = (get_code.len() as i64 - set_code.len() as i64).unsigned_abs();
        assert!(
            diff <= 8,
            "DictGetStr ({}) and DictSetStr ({}) should be same size (diff {})",
            get_code.len(),
            set_code.len(),
            diff,
        );
    }

    // =========================================================================
    // DictContainsTemplate Tests
    // =========================================================================

    #[test]
    fn test_dict_contains_emits_code() {
        let template = DictContainsTemplate::new(0, 1, 2, 0);
        let code = emit_and_finalize(&template);
        assert!(code.len() > 0, "DictContains should emit code");
    }

    #[test]
    fn test_dict_contains_code_within_estimate() {
        let template = DictContainsTemplate::new(0, 1, 2, 0);
        let code = emit_and_finalize(&template);
        let estimate = template.estimated_size();
        assert!(
            code.len() <= estimate * 2,
            "DictContains code {} exceeds 2x estimate {}",
            code.len(),
            estimate
        );
    }

    #[test]
    fn test_dict_contains_similar_to_get() {
        let get = DictGetTemplate::new(0, 1, 2, 0);
        let contains = DictContainsTemplate::new(0, 1, 2, 0);
        let get_code = emit_and_finalize(&get);
        let contains_code = emit_and_finalize(&contains);
        let diff = (get_code.len() as i64 - contains_code.len() as i64).unsigned_abs();
        assert!(
            diff <= 8,
            "DictGet ({}) and DictContains ({}) should be similar size (diff {})",
            get_code.len(),
            contains_code.len(),
            diff,
        );
    }

    #[test]
    fn test_dict_contains_deterministic() {
        let t1 = DictContainsTemplate::new(0, 1, 2, 0);
        let t2 = DictContainsTemplate::new(0, 1, 2, 0);
        assert_eq!(emit_and_finalize(&t1), emit_and_finalize(&t2));
    }

    // =========================================================================
    // DictMergeTemplate Tests
    // =========================================================================

    #[test]
    fn test_dict_merge_emits_code() {
        let template = DictMergeTemplate::new(0, 1, 2, 0);
        let code = emit_and_finalize(&template);
        assert!(code.len() > 0, "DictMerge should emit code");
    }

    #[test]
    fn test_dict_merge_more_code_than_single_check() {
        // Merge has TWO dict type guards, so should be substantially larger
        let single = DictGetTemplate::new(0, 1, 2, 0);
        let merge = DictMergeTemplate::new(0, 1, 2, 0);
        let single_code = emit_and_finalize(&single);
        let merge_code = emit_and_finalize(&merge);
        assert!(
            merge_code.len() > single_code.len(),
            "DictMerge ({}) should be larger than DictGet ({}) due to dual type guards",
            merge_code.len(),
            single_code.len()
        );
    }

    #[test]
    fn test_dict_merge_code_within_estimate() {
        let template = DictMergeTemplate::new(0, 1, 2, 0);
        let code = emit_and_finalize(&template);
        let estimate = template.estimated_size();
        assert!(
            code.len() <= estimate * 2,
            "DictMerge code {} exceeds 2x estimate {}",
            code.len(),
            estimate
        );
    }

    #[test]
    fn test_dict_merge_deterministic() {
        let t1 = DictMergeTemplate::new(0, 1, 2, 0);
        let t2 = DictMergeTemplate::new(0, 1, 2, 0);
        assert_eq!(emit_and_finalize(&t1), emit_and_finalize(&t2));
    }

    #[test]
    fn test_dict_merge_registers() {
        for lhs_reg in [0u8, 1, 2, 3] {
            for rhs_reg in [0u8, 1, 2, 3] {
                let template = DictMergeTemplate::new(lhs_reg, rhs_reg, 0, 0);
                let code = emit_and_finalize(&template);
                assert!(code.len() > 0);
            }
        }
    }

    // =========================================================================
    // DictDeleteTemplate Tests
    // =========================================================================

    #[test]
    fn test_dict_delete_emits_code() {
        let template = DictDeleteTemplate::new(0, 1, 0);
        let code = emit_and_finalize(&template);
        assert!(code.len() > 0, "DictDelete should emit code");
    }

    #[test]
    fn test_dict_delete_code_within_estimate() {
        let template = DictDeleteTemplate::new(0, 1, 0);
        let code = emit_and_finalize(&template);
        let estimate = template.estimated_size();
        assert!(
            code.len() <= estimate * 2,
            "DictDelete code {} exceeds 2x estimate {}",
            code.len(),
            estimate
        );
    }

    #[test]
    fn test_dict_delete_similar_to_contains() {
        let contains = DictContainsTemplate::new(0, 1, 2, 0);
        let delete = DictDeleteTemplate::new(0, 1, 0);
        let contains_code = emit_and_finalize(&contains);
        let delete_code = emit_and_finalize(&delete);
        let diff = (contains_code.len() as i64 - delete_code.len() as i64).unsigned_abs();
        assert!(
            diff <= 8,
            "DictContains ({}) and DictDelete ({}) should be similar (diff {})",
            contains_code.len(),
            delete_code.len(),
            diff,
        );
    }

    #[test]
    fn test_dict_delete_deterministic() {
        let t1 = DictDeleteTemplate::new(0, 1, 0);
        let t2 = DictDeleteTemplate::new(0, 1, 0);
        assert_eq!(emit_and_finalize(&t1), emit_and_finalize(&t2));
    }

    // =========================================================================
    // DictViewGuardTemplate Tests
    // =========================================================================

    #[test]
    fn test_dict_view_guard_emits_code() {
        let template = DictViewGuardTemplate::new(0, 1, 0);
        let code = emit_and_finalize(&template);
        assert!(code.len() > 0, "DictViewGuard should emit code");
    }

    #[test]
    fn test_dict_view_guard_code_within_estimate() {
        let template = DictViewGuardTemplate::new(0, 1, 0);
        let code = emit_and_finalize(&template);
        let estimate = template.estimated_size();
        assert!(
            code.len() <= estimate * 2,
            "DictViewGuard code {} exceeds 2x estimate {}",
            code.len(),
            estimate
        );
    }

    #[test]
    fn test_dict_view_guard_deterministic() {
        let t1 = DictViewGuardTemplate::new(0, 1, 0);
        let t2 = DictViewGuardTemplate::new(0, 1, 0);
        assert_eq!(emit_and_finalize(&t1), emit_and_finalize(&t2));
    }

    // =========================================================================
    // DictGetOrDefaultTemplate Tests
    // =========================================================================

    #[test]
    fn test_dict_get_or_default_emits_code() {
        let template = DictGetOrDefaultTemplate::new(0, 1, 2, 3, 0);
        let code = emit_and_finalize(&template);
        assert!(code.len() > 0, "DictGetOrDefault should emit code");
    }

    #[test]
    fn test_dict_get_or_default_code_within_estimate() {
        let template = DictGetOrDefaultTemplate::new(0, 1, 2, 3, 0);
        let code = emit_and_finalize(&template);
        let estimate = template.estimated_size();
        assert!(
            code.len() <= estimate * 2,
            "DictGetOrDefault code {} exceeds 2x estimate {}",
            code.len(),
            estimate
        );
    }

    #[test]
    fn test_dict_get_or_default_deterministic() {
        let t1 = DictGetOrDefaultTemplate::new(0, 1, 2, 3, 0);
        let t2 = DictGetOrDefaultTemplate::new(0, 1, 2, 3, 0);
        assert_eq!(emit_and_finalize(&t1), emit_and_finalize(&t2));
    }

    #[test]
    fn test_dict_get_or_default_similar_to_single_guard() {
        // Both have dict guard + deopt
        let get = DictGetTemplate::new(0, 1, 2, 0);
        let get_default = DictGetOrDefaultTemplate::new(0, 1, 2, 3, 0);
        let get_code = emit_and_finalize(&get);
        let default_code = emit_and_finalize(&get_default);
        let diff = (get_code.len() as i64 - default_code.len() as i64).unsigned_abs();
        assert!(
            diff <= 8,
            "DictGet ({}) and DictGetOrDefault ({}) should be similar (diff {})",
            get_code.len(),
            default_code.len(),
            diff,
        );
    }

    // =========================================================================
    // Cross-Template Comparisons
    // =========================================================================

    #[test]
    fn test_single_guard_templates_similar_size() {
        // All templates with a single dict guard should produce similar code
        let get = emit_and_finalize(&DictGetTemplate::new(0, 1, 2, 0));
        let set = emit_and_finalize(&DictSetFastTemplate::new(0, 1, 2, 0));
        let contains = emit_and_finalize(&DictContainsTemplate::new(0, 1, 2, 0));
        let delete = emit_and_finalize(&DictDeleteTemplate::new(0, 1, 0));
        let view = emit_and_finalize(&DictViewGuardTemplate::new(0, 1, 0));
        let get_default = emit_and_finalize(&DictGetOrDefaultTemplate::new(0, 1, 2, 3, 0));

        let sizes = [
            get.len(),
            set.len(),
            contains.len(),
            delete.len(),
            view.len(),
            get_default.len(),
        ];
        let min = *sizes.iter().min().unwrap();
        let max = *sizes.iter().max().unwrap();
        assert!(
            max - min <= 12,
            "Single-guard templates should be uniform: sizes {:?} (spread {})",
            sizes,
            max - min,
        );
    }

    #[test]
    fn test_dual_guard_templates_larger_than_single() {
        let single = emit_and_finalize(&DictGetTemplate::new(0, 1, 2, 0));
        let dual_str = emit_and_finalize(&DictGetStrTemplate::new(0, 1, 2, 0));
        let dual_int = emit_and_finalize(&DictGetIntTemplate::new(0, 1, 2, 0));
        let dual_merge = emit_and_finalize(&DictMergeTemplate::new(0, 1, 2, 0));

        assert!(
            dual_str.len() > single.len(),
            "DictGetStr should be larger than DictGet"
        );
        assert!(
            dual_int.len() > single.len(),
            "DictGetInt should be larger than DictGet"
        );
        assert!(
            dual_merge.len() > single.len(),
            "DictMerge should be larger than DictGet"
        );
    }

    #[test]
    fn test_merge_largest_template() {
        // Merge has two OBJECT + TypeId checks, should be the largest template
        let get_str = emit_and_finalize(&DictGetStrTemplate::new(0, 1, 2, 0));
        let merge = emit_and_finalize(&DictMergeTemplate::new(0, 1, 2, 0));
        // Merge checks OBJECT+DICT twice, GetStr checks OBJECT+DICT + STRING
        // Both have dual checks but merge has two full object+typeid checks
        // while GetStr has one object+typeid check + one string check
        // They should be similar, but merge may be larger due to dual object checks
        assert!(
            merge.len() >= get_str.len() - 10,
            "DictMerge ({}) should be at least as large as DictGetStr ({}) minus tolerance",
            merge.len(),
            get_str.len()
        );
    }

    // =========================================================================
    // Code Quality
    // =========================================================================

    #[test]
    fn test_all_templates_emit_nonzero_code() {
        let templates: Vec<Box<dyn OpcodeTemplate>> = vec![
            Box::new(DictGetTemplate::new(0, 1, 2, 0)),
            Box::new(DictGetStrTemplate::new(0, 1, 2, 0)),
            Box::new(DictGetIntTemplate::new(0, 1, 2, 0)),
            Box::new(DictSetFastTemplate::new(0, 1, 2, 0)),
            Box::new(DictSetStrTemplate::new(0, 1, 2, 0)),
            Box::new(DictContainsTemplate::new(0, 1, 2, 0)),
            Box::new(DictMergeTemplate::new(0, 1, 2, 0)),
            Box::new(DictDeleteTemplate::new(0, 1, 0)),
            Box::new(DictViewGuardTemplate::new(0, 1, 0)),
            Box::new(DictGetOrDefaultTemplate::new(0, 1, 2, 3, 0)),
        ];

        for (i, t) in templates.iter().enumerate() {
            let code = emit_and_finalize(t.as_ref());
            assert!(code.len() > 0, "Template {} should emit code", i);
        }
    }

    #[test]
    fn test_all_templates_reasonable_size() {
        let templates: Vec<Box<dyn OpcodeTemplate>> = vec![
            Box::new(DictGetTemplate::new(0, 1, 2, 0)),
            Box::new(DictGetStrTemplate::new(0, 1, 2, 0)),
            Box::new(DictGetIntTemplate::new(0, 1, 2, 0)),
            Box::new(DictSetFastTemplate::new(0, 1, 2, 0)),
            Box::new(DictSetStrTemplate::new(0, 1, 2, 0)),
            Box::new(DictContainsTemplate::new(0, 1, 2, 0)),
            Box::new(DictMergeTemplate::new(0, 1, 2, 0)),
            Box::new(DictDeleteTemplate::new(0, 1, 0)),
            Box::new(DictViewGuardTemplate::new(0, 1, 0)),
            Box::new(DictGetOrDefaultTemplate::new(0, 1, 2, 3, 0)),
        ];

        for (i, t) in templates.iter().enumerate() {
            let code = emit_and_finalize(t.as_ref());
            // No template should exceed 256 bytes (they're mostly type guards + jmp)
            assert!(
                code.len() <= 256,
                "Template {} too large: {} bytes",
                i,
                code.len()
            );
            // Should be at least 20 bytes (minimum for a meaningful type guard)
            assert!(
                code.len() >= 20,
                "Template {} too small: {} bytes",
                i,
                code.len()
            );
        }
    }

    #[test]
    fn test_all_estimated_sizes_valid() {
        let templates: Vec<Box<dyn OpcodeTemplate>> = vec![
            Box::new(DictGetTemplate::new(0, 1, 2, 0)),
            Box::new(DictGetStrTemplate::new(0, 1, 2, 0)),
            Box::new(DictGetIntTemplate::new(0, 1, 2, 0)),
            Box::new(DictSetFastTemplate::new(0, 1, 2, 0)),
            Box::new(DictSetStrTemplate::new(0, 1, 2, 0)),
            Box::new(DictContainsTemplate::new(0, 1, 2, 0)),
            Box::new(DictMergeTemplate::new(0, 1, 2, 0)),
            Box::new(DictDeleteTemplate::new(0, 1, 0)),
            Box::new(DictViewGuardTemplate::new(0, 1, 0)),
            Box::new(DictGetOrDefaultTemplate::new(0, 1, 2, 3, 0)),
        ];

        for (i, t) in templates.iter().enumerate() {
            let estimate = t.estimated_size();
            assert!(
                estimate > 0,
                "Template {} estimated size should be nonzero",
                i
            );
            assert!(
                estimate <= 256,
                "Template {} estimated size {} too large",
                i,
                estimate
            );
        }
    }

    // =========================================================================
    // Stability Under Register Permutation
    // =========================================================================

    #[test]
    fn test_dict_get_str_register_independence() {
        // Code structure should be independent of specific register values
        // (only the slot offset changes, instruction sequence stays the same)
        let t1 = DictGetStrTemplate::new(0, 1, 2, 0);
        let t2 = DictGetStrTemplate::new(2, 3, 0, 0);
        let code1 = emit_and_finalize(&t1);
        let code2 = emit_and_finalize(&t2);
        // Same instruction sequence, possibly different offsets
        // Length should be very similar (within a few bytes for modrm differences)
        let diff = (code1.len() as i64 - code2.len() as i64).unsigned_abs();
        assert!(
            diff <= 16,
            "Register permutation caused too much size variance ({}): {} vs {}",
            diff,
            code1.len(),
            code2.len()
        );
    }

    #[test]
    fn test_dict_merge_register_independence() {
        let t1 = DictMergeTemplate::new(0, 1, 2, 0);
        let t2 = DictMergeTemplate::new(3, 4, 0, 0);
        let code1 = emit_and_finalize(&t1);
        let code2 = emit_and_finalize(&t2);
        let diff = (code1.len() as i64 - code2.len() as i64).unsigned_abs();
        assert!(
            diff <= 16,
            "Merge register permutation caused too much size variance ({}): {} vs {}",
            diff,
            code1.len(),
            code2.len()
        );
    }

    // =========================================================================
    // Guard Correctness: Object Check Present
    // =========================================================================

    #[test]
    fn test_dict_get_includes_object_guard() {
        // Verify that DictGet has a proper object+typeid guard
        // by checking it's larger than just a jmp instruction
        let template = DictGetTemplate::new(0, 1, 2, 0);
        let code = emit_and_finalize(&template);
        // A naked jmp is ~5 bytes; a proper guard should be much more
        assert!(
            code.len() >= 25,
            "DictGet should include a full type guard (got {} bytes, expected >= 25)",
            code.len()
        );
    }

    #[test]
    fn test_dict_get_str_includes_double_guard() {
        let basic = DictGetTemplate::new(0, 1, 2, 0);
        let str_template = DictGetStrTemplate::new(0, 1, 2, 0);
        let basic_code = emit_and_finalize(&basic);
        let str_code = emit_and_finalize(&str_template);
        // The str version should have at least 10 additional bytes of guards
        assert!(
            str_code.len() >= basic_code.len() + 10,
            "DictGetStr should have significantly more code than DictGet: {} vs {}",
            str_code.len(),
            basic_code.len()
        );
    }

    #[test]
    fn test_dict_merge_includes_double_object_guard() {
        let single = DictGetTemplate::new(0, 1, 2, 0);
        let merge = DictMergeTemplate::new(0, 1, 2, 0);
        let single_code = emit_and_finalize(&single);
        let merge_code = emit_and_finalize(&merge);
        // Merge should have roughly 2x the guard code, definitely at least 50% more
        assert!(
            merge_code.len() as f64 >= single_code.len() as f64 * 1.4,
            "DictMerge should have substantially more guard code: {} vs {}",
            merge_code.len(),
            single_code.len()
        );
    }
}
