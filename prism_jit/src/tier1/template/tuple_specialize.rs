//! Tuple-specialized JIT templates for high-performance tuple operations.
//!
//! Provides type-specialized native code generation for:
//! - **Tuple indexing** (`tuple[i]`) — fully inline with negative index normalization
//! - **Tuple length** (`len(tuple)`) — fully inline, no runtime call
//! - **Tuple containment** (`x in tuple`) — guard + deopt for linear scan
//! - **Tuple concatenation** (`tuple + tuple`) — guard + deopt for allocation
//! - **Tuple repetition** (`tuple * n`) — guard + deopt for allocation
//!
//! # TupleObject Memory Layout
//!
//! `TupleObject` is `#[repr(C)]`:
//! ```text
//! Offset  Size  Field
//! ──────  ────  ─────────────────────
//!   0      4    type_id  (TypeId::TUPLE = 7)
//!   4      4    gc_flags (AtomicU32)
//!   8      8    hash     (u64)
//!  16      8    Box<[Value]>.ptr   (data pointer)
//!  24      8    Box<[Value]>.len   (element count)
//! ```
//!
//! Unlike `ListObject` (which uses `Vec<Value>` with ptr/len/cap),
//! `TupleObject` uses `Box<[Value]>` with only ptr/len — no capacity field,
//! since tuples are immutable and never grow.
//!
//! # Optimization Strategy
//!
//! 1. **Fully inlined** (`TupleIndex`, `TupleLen`): These are simple field
//!    loads and arithmetic. The JIT can emit all logic inline:
//!    - Index: type guard, extract index, normalize negative, bounds check, load
//!    - Len: type guard, load len, NaN-box as int
//!
//! 2. **Guard + deopt** (`TupleContains`, `TupleConcat`, `TupleRepeat`):
//!    The inline type guard eliminates interpreter-level type dispatch.
//!    The actual operation (linear scan, allocation) deopts to the interpreter.

use super::specialize_common::{
    emit_int_check_and_extract, emit_typed_object_check_and_extract, type_ids,
};
use super::{OpcodeTemplate, TemplateContext, value_tags};
use crate::backend::x64::Gpr;
use crate::backend::x64::registers::{MemOperand, Scale};

// =============================================================================
// Tuple Layout Constants
// =============================================================================

/// Memory layout constants for `TupleObject`.
///
/// These offsets correspond to the `#[repr(C)]` layout of `TupleObject`:
/// - `ObjectHeader` (16 bytes) at offset 0
/// - `Box<[Value]>` (ptr + len, no cap) starting at offset 16
pub mod tuple_layout {
    /// Offset of `Box<[Value]>.ptr` within TupleObject.
    ///
    /// This is `ObjectHeader::SIZE` = 16 bytes.
    pub const ITEMS_PTR_OFFSET: i32 = 16;

    /// Offset of `Box<[Value]>.len` within TupleObject.
    ///
    /// Immediately follows the data pointer: 16 + 8 = 24.
    pub const ITEMS_LEN_OFFSET: i32 = 24;

    /// Size of a single `Value` (u64 = 8 bytes).
    pub const VALUE_SIZE: i32 = 8;
}

// =============================================================================
// Helper: Emit tuple type check and extract
// =============================================================================

/// Emit code to verify a value is a tuple and extract the object pointer.
///
/// Performs:
/// 1. Check OBJECT_TAG in NaN-box
/// 2. Extract 48-bit pointer payload
/// 3. Load `type_id` from ObjectHeader and verify it equals `TypeId::TUPLE`
///
/// After this function, `dst` contains the raw `TupleObject` pointer.
#[inline]
fn emit_tuple_check_and_extract(
    ctx: &mut TemplateContext,
    src: Gpr,
    dst: Gpr,
    scratch: Gpr,
    deopt_idx: usize,
) {
    emit_typed_object_check_and_extract(ctx, src, dst, scratch, type_ids::TUPLE, deopt_idx);
}

// =============================================================================
// Helper: Emit integer NaN-box for length results
// =============================================================================

/// Emit code to NaN-box a non-negative integer value (e.g. tuple length).
///
/// Applies `PAYLOAD_MASK` then ORs with `(QNAN_BITS | INT_TAG)`.
/// The value register is modified in-place. `scratch` is clobbered.
#[inline]
fn emit_int_box(ctx: &mut TemplateContext, value: Gpr, scratch: Gpr) {
    ctx.asm.mov_ri64(scratch, value_tags::PAYLOAD_MASK as i64);
    ctx.asm.and_rr(value, scratch);
    let tag = (value_tags::QNAN_BITS | value_tags::INT_TAG) as i64;
    ctx.asm.mov_ri64(scratch, tag);
    ctx.asm.or_rr(value, scratch);
}

// =============================================================================
// Tuple Index Template
// =============================================================================

/// Template for tuple indexing (`tuple[i]`).
///
/// # Code Generation Strategy
///
/// 1. Type-check tuple: OBJECT_TAG + TypeId::TUPLE
/// 2. Type-check index: INT_TAG, extract signed 48-bit integer
/// 3. Load tuple length
/// 4. Normalize negative index: `if index < 0 { index += len }`
/// 5. Bounds check: unsigned `index >= len` → deopt (IndexError)
/// 6. Load items pointer, load `items[index * 8]`
/// 7. Store result to destination register
///
/// This is fully inlined — no runtime call needed. The negative index
/// normalization uses the same branchless pattern as `ListIndexTemplate`:
/// test-and-branch with `JGE` to skip the `ADD` for positive indices.
///
/// # Estimated Code Size
///
/// ~200 bytes: tuple guard (~32), int guard (~20), len load (~7),
/// negative normalization (~12), bounds check (~8), items load (~14),
/// indexed load (~4), store (~7), overhead (~40+)
pub struct TupleIndexTemplate {
    pub dst_reg: u8,
    pub tuple_reg: u8,
    pub index_reg: u8,
    pub deopt_idx: usize,
}

impl TupleIndexTemplate {
    /// Create a new tuple index template.
    #[inline]
    pub fn new(dst_reg: u8, tuple_reg: u8, index_reg: u8, deopt_idx: usize) -> Self {
        Self {
            dst_reg,
            tuple_reg,
            index_reg,
            deopt_idx,
        }
    }
}

impl OpcodeTemplate for TupleIndexTemplate {
    fn emit(&self, ctx: &mut TemplateContext) {
        let acc = ctx.regs.accumulator;
        let scratch1 = ctx.regs.scratch1;
        let scratch2 = ctx.regs.scratch2;

        // Load tuple value from frame
        let tuple_slot = ctx.frame.register_slot(self.tuple_reg as u16);
        ctx.asm.mov_rm(acc, &tuple_slot);

        // Type-check: OBJECT_TAG + TypeId::TUPLE
        emit_tuple_check_and_extract(ctx, acc, acc, scratch1, self.deopt_idx);
        // acc = TupleObject pointer

        // Load index value from frame
        let index_slot = ctx.frame.register_slot(self.index_reg as u16);
        ctx.asm.mov_rm(scratch1, &index_slot);

        // Type-check index: INT_TAG
        emit_int_check_and_extract(ctx, scratch1, scratch1, scratch2, self.deopt_idx);
        // scratch1 = sign-extended index value

        // Load length: scratch2 = tuple->items.len
        let len_mem = MemOperand {
            base: Some(acc),
            index: None,
            scale: Scale::X1,
            disp: tuple_layout::ITEMS_LEN_OFFSET,
        };
        ctx.asm.mov_rm(scratch2, &len_mem);

        // Handle negative index: if index < 0, index += len
        ctx.asm.cmp_ri(scratch1, 0);
        let positive_label = ctx.asm.create_label();
        ctx.asm.jge(positive_label);
        // Negative path: scratch1 += scratch2 (index += len)
        ctx.asm.add_rr(scratch1, scratch2);
        ctx.asm.bind_label(positive_label);

        // Bounds check: unsigned index >= len → deopt (IndexError)
        // After normalization, if still negative, unsigned cmp catches it
        ctx.asm.cmp_rr(scratch1, scratch2);
        ctx.asm.jae(ctx.deopt_label(self.deopt_idx));

        // Load items pointer: acc = tuple->items.ptr
        let items_mem = MemOperand {
            base: Some(acc),
            index: None,
            scale: Scale::X1,
            disp: tuple_layout::ITEMS_PTR_OFFSET,
        };
        ctx.asm.mov_rm(acc, &items_mem);

        // Load value: acc = items_ptr[index * 8]
        let item_mem = MemOperand {
            base: Some(acc),
            index: Some(scratch1),
            scale: Scale::X8,
            disp: 0,
        };
        ctx.asm.mov_rm(acc, &item_mem);

        // Store result to destination
        let dst_slot = ctx.frame.register_slot(self.dst_reg as u16);
        ctx.asm.mov_mr(&dst_slot, acc);
    }

    #[inline]
    fn estimated_size(&self) -> usize {
        200
    }
}

// =============================================================================
// Tuple Length Template
// =============================================================================

/// Template for tuple length (`len(tuple)`).
///
/// # Code Generation Strategy
///
/// 1. Type-check tuple: OBJECT_TAG + TypeId::TUPLE
/// 2. Load `items.len` field (offset 24)
/// 3. NaN-box as integer: `(QNAN_BITS | INT_TAG) | (len & PAYLOAD_MASK)`
/// 4. Store result
///
/// Fully inlined — trivial field load + integer boxing.
///
/// # Estimated Code Size
///
/// ~120 bytes: tuple guard (~32), len load (~7), int boxing (~24),
/// store (~7), overhead (~50)
pub struct TupleLenTemplate {
    pub dst_reg: u8,
    pub tuple_reg: u8,
    pub deopt_idx: usize,
}

impl TupleLenTemplate {
    /// Create a new tuple length template.
    #[inline]
    pub fn new(dst_reg: u8, tuple_reg: u8, deopt_idx: usize) -> Self {
        Self {
            dst_reg,
            tuple_reg,
            deopt_idx,
        }
    }
}

impl OpcodeTemplate for TupleLenTemplate {
    fn emit(&self, ctx: &mut TemplateContext) {
        let acc = ctx.regs.accumulator;
        let scratch1 = ctx.regs.scratch1;

        // Load tuple value from frame
        let tuple_slot = ctx.frame.register_slot(self.tuple_reg as u16);
        ctx.asm.mov_rm(acc, &tuple_slot);

        // Type-check: OBJECT_TAG + TypeId::TUPLE
        emit_tuple_check_and_extract(ctx, acc, acc, scratch1, self.deopt_idx);
        // acc = TupleObject pointer

        // Load length: acc = tuple->items.len
        let len_mem = MemOperand {
            base: Some(acc),
            index: None,
            scale: Scale::X1,
            disp: tuple_layout::ITEMS_LEN_OFFSET,
        };
        ctx.asm.mov_rm(acc, &len_mem);

        // NaN-box as integer
        emit_int_box(ctx, acc, scratch1);

        // Store result to destination
        let dst_slot = ctx.frame.register_slot(self.dst_reg as u16);
        ctx.asm.mov_mr(&dst_slot, acc);
    }

    #[inline]
    fn estimated_size(&self) -> usize {
        120
    }
}

// =============================================================================
// Tuple Contains Template
// =============================================================================

/// Template for tuple containment check (`x in tuple`).
///
/// # Code Generation Strategy
///
/// 1. Type-check tuple: OBJECT_TAG + TypeId::TUPLE
/// 2. Deopt to interpreter for linear scan
///
/// The inline type guard eliminates interpreter-level polymorphic dispatch.
/// The actual containment check requires a loop over items with value
/// comparison — too complex for Tier 1 inline code.
///
/// # Estimated Code Size
///
/// ~80 bytes: tuple guard (~32), value load (~7), jmp (~8), overhead (~33)
pub struct TupleContainsTemplate {
    pub tuple_reg: u8,
    pub value_reg: u8,
    pub dst_reg: u8,
    pub deopt_idx: usize,
}

impl TupleContainsTemplate {
    /// Create a new tuple contains template.
    #[inline]
    pub fn new(tuple_reg: u8, value_reg: u8, dst_reg: u8, deopt_idx: usize) -> Self {
        Self {
            tuple_reg,
            value_reg,
            dst_reg,
            deopt_idx,
        }
    }
}

impl OpcodeTemplate for TupleContainsTemplate {
    fn emit(&self, ctx: &mut TemplateContext) {
        let acc = ctx.regs.accumulator;
        let scratch1 = ctx.regs.scratch1;

        // Load tuple value from frame
        let tuple_slot = ctx.frame.register_slot(self.tuple_reg as u16);
        ctx.asm.mov_rm(acc, &tuple_slot);

        // Type-check: OBJECT_TAG + TypeId::TUPLE
        emit_tuple_check_and_extract(ctx, acc, acc, scratch1, self.deopt_idx);

        // Deopt to interpreter for linear scan
        ctx.asm.jmp(ctx.deopt_label(self.deopt_idx));
    }

    #[inline]
    fn estimated_size(&self) -> usize {
        80
    }
}

// =============================================================================
// Tuple Concat Template
// =============================================================================

/// Template for tuple concatenation (`tuple + tuple`).
///
/// # Code Generation Strategy
///
/// 1. Type-check LHS: OBJECT_TAG + TypeId::TUPLE
/// 2. Type-check RHS: OBJECT_TAG + TypeId::TUPLE
/// 3. Deopt to interpreter for allocation + copy
///
/// Both operands must be tuples. The double type guard eliminates both
/// type-dispatch checks on re-entry to the interpreter.
///
/// # Estimated Code Size
///
/// ~120 bytes: 2× tuple guard (~64), jmp (~8), loads (~14), overhead (~34)
pub struct TupleConcatTemplate {
    pub dst_reg: u8,
    pub lhs_reg: u8,
    pub rhs_reg: u8,
    pub deopt_idx: usize,
}

impl TupleConcatTemplate {
    /// Create a new tuple concat template.
    #[inline]
    pub fn new(dst_reg: u8, lhs_reg: u8, rhs_reg: u8, deopt_idx: usize) -> Self {
        Self {
            dst_reg,
            lhs_reg,
            rhs_reg,
            deopt_idx,
        }
    }
}

impl OpcodeTemplate for TupleConcatTemplate {
    fn emit(&self, ctx: &mut TemplateContext) {
        let acc = ctx.regs.accumulator;
        let scratch1 = ctx.regs.scratch1;

        // Load LHS tuple
        let lhs_slot = ctx.frame.register_slot(self.lhs_reg as u16);
        ctx.asm.mov_rm(acc, &lhs_slot);

        // Type-check LHS: OBJECT_TAG + TypeId::TUPLE
        emit_tuple_check_and_extract(ctx, acc, acc, scratch1, self.deopt_idx);

        // Load RHS tuple
        let rhs_slot = ctx.frame.register_slot(self.rhs_reg as u16);
        ctx.asm.mov_rm(scratch1, &rhs_slot);

        // Type-check RHS: OBJECT_TAG + TypeId::TUPLE
        // Use acc as scratch since we've consumed the LHS pointer already
        emit_tuple_check_and_extract(ctx, scratch1, scratch1, acc, self.deopt_idx);

        // Deopt to interpreter for allocation
        ctx.asm.jmp(ctx.deopt_label(self.deopt_idx));
    }

    #[inline]
    fn estimated_size(&self) -> usize {
        120
    }
}

// =============================================================================
// Tuple Repeat Template
// =============================================================================

/// Template for tuple repetition (`tuple * n`).
///
/// # Code Generation Strategy
///
/// 1. Type-check tuple: OBJECT_TAG + TypeId::TUPLE
/// 2. Type-check count: INT_TAG, extract signed integer
/// 3. Deopt to interpreter for allocation + repeated copy
///
/// The double type guard eliminates both receiver-type and count-type
/// dispatch in the interpreter on re-entry.
///
/// # Estimated Code Size
///
/// ~120 bytes: tuple guard (~32), int guard (~20), jmp (~8),
/// loads (~14), overhead (~46)
pub struct TupleRepeatTemplate {
    pub dst_reg: u8,
    pub tuple_reg: u8,
    pub count_reg: u8,
    pub deopt_idx: usize,
}

impl TupleRepeatTemplate {
    /// Create a new tuple repeat template.
    #[inline]
    pub fn new(dst_reg: u8, tuple_reg: u8, count_reg: u8, deopt_idx: usize) -> Self {
        Self {
            dst_reg,
            tuple_reg,
            count_reg,
            deopt_idx,
        }
    }
}

impl OpcodeTemplate for TupleRepeatTemplate {
    fn emit(&self, ctx: &mut TemplateContext) {
        let acc = ctx.regs.accumulator;
        let scratch1 = ctx.regs.scratch1;
        let scratch2 = ctx.regs.scratch2;

        // Load tuple value from frame
        let tuple_slot = ctx.frame.register_slot(self.tuple_reg as u16);
        ctx.asm.mov_rm(acc, &tuple_slot);

        // Type-check: OBJECT_TAG + TypeId::TUPLE
        emit_tuple_check_and_extract(ctx, acc, acc, scratch1, self.deopt_idx);

        // Load count value from frame
        let count_slot = ctx.frame.register_slot(self.count_reg as u16);
        ctx.asm.mov_rm(scratch1, &count_slot);

        // Type-check count: INT_TAG
        emit_int_check_and_extract(ctx, scratch1, scratch1, scratch2, self.deopt_idx);

        // Deopt to interpreter for allocation
        ctx.asm.jmp(ctx.deopt_label(self.deopt_idx));
    }

    #[inline]
    fn estimated_size(&self) -> usize {
        120
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

    // =========================================================================
    // Test Helpers
    // =========================================================================

    /// Emit a template and get the generated code bytes.
    fn emit_and_finalize(template: &dyn OpcodeTemplate) -> Vec<u8> {
        let mut asm = Assembler::new();
        let frame = FrameLayout::minimal(16);
        {
            let mut ctx = TemplateContext::new(&mut asm, &frame);
            ctx.create_deopt_label();
            ctx.create_deopt_label();
            template.emit(&mut ctx);
            for l in &ctx.deopt_labels {
                ctx.asm.bind_label(*l);
            }
        }
        asm.finalize().unwrap()
    }

    // =========================================================================
    // Tuple Layout Constants
    // =========================================================================

    #[test]
    fn test_tuple_items_ptr_offset_matches_header_size() {
        // items.ptr starts right after ObjectHeader (16 bytes)
        assert_eq!(tuple_layout::ITEMS_PTR_OFFSET, 16);
    }

    #[test]
    fn test_tuple_items_len_offset_follows_ptr() {
        // items.len is immediately after the pointer (16 + 8 = 24)
        assert_eq!(
            tuple_layout::ITEMS_LEN_OFFSET,
            tuple_layout::ITEMS_PTR_OFFSET + 8
        );
    }

    #[test]
    fn test_tuple_value_size_is_8() {
        assert_eq!(tuple_layout::VALUE_SIZE, 8);
    }

    #[test]
    fn test_tuple_no_capacity_field() {
        // Unlike ListObject (ptr at 16, len at 24, cap at 32),
        // TupleObject has no capacity — immutable, fixed-size.
        // items.len at 24 is the last field.
        assert_eq!(tuple_layout::ITEMS_LEN_OFFSET, 24);
        // No tuple_layout::ITEMS_CAP_OFFSET exists
    }

    #[test]
    fn test_tuple_layout_consistency_with_runtime() {
        // Verify layout offsets match the actual TupleObject struct.
        // TupleObject is #[repr(C)]:
        //   ObjectHeader (16) + Box<[Value]> (ptr=16, len=24)
        use super::super::specialize_common::object_layout;
        assert_eq!(
            tuple_layout::ITEMS_PTR_OFFSET,
            object_layout::HEADER_SIZE,
            "items.ptr must start at ObjectHeader boundary"
        );
    }

    #[test]
    fn test_tuple_type_id_matches_runtime() {
        // TypeId::TUPLE = 7, matching specialize_common::type_ids::TUPLE
        assert_eq!(type_ids::TUPLE, 7);
    }

    // =========================================================================
    // TupleIndexTemplate Tests
    // =========================================================================

    #[test]
    fn test_tuple_index_template_creation() {
        let t = TupleIndexTemplate::new(0, 1, 2, 0);
        assert_eq!(t.dst_reg, 0);
        assert_eq!(t.tuple_reg, 1);
        assert_eq!(t.index_reg, 2);
        assert_eq!(t.deopt_idx, 0);
    }

    #[test]
    fn test_tuple_index_template_estimated_size() {
        let t = TupleIndexTemplate::new(0, 1, 2, 0);
        assert!(
            t.estimated_size() >= 100,
            "TupleIndex estimated size {} too small",
            t.estimated_size()
        );
    }

    #[test]
    fn test_tuple_index_template_emits_code() {
        let t = TupleIndexTemplate::new(0, 1, 2, 0);
        let code = emit_and_finalize(&t);
        assert!(
            !code.is_empty(),
            "TupleIndexTemplate must emit non-zero code"
        );
    }

    #[test]
    fn test_tuple_index_code_size_within_estimate() {
        let t = TupleIndexTemplate::new(0, 1, 2, 0);
        let code = emit_and_finalize(&t);
        assert!(
            code.len() <= t.estimated_size(),
            "TupleIndex actual {} exceeds estimate {}",
            code.len(),
            t.estimated_size()
        );
    }

    #[test]
    fn test_tuple_index_different_registers() {
        // Verify code is emitted for various register combinations
        let combos: [(u8, u8, u8); 4] = [(0, 1, 2), (3, 4, 5), (0, 0, 1), (7, 8, 9)];
        for (dst, tup, idx) in combos {
            let t = TupleIndexTemplate::new(dst, tup, idx, 0);
            let code = emit_and_finalize(&t);
            assert!(
                !code.is_empty(),
                "No code for regs ({}, {}, {})",
                dst,
                tup,
                idx
            );
        }
    }

    #[test]
    fn test_tuple_index_register_sizes_consistent() {
        // Same template with different register combos should produce similar-sized code
        let t1 = TupleIndexTemplate::new(0, 1, 2, 0);
        let t2 = TupleIndexTemplate::new(3, 4, 5, 0);
        let c1 = emit_and_finalize(&t1);
        let c2 = emit_and_finalize(&t2);
        let diff = (c1.len() as i64 - c2.len() as i64).unsigned_abs();
        assert!(
            diff <= 20,
            "Register variation caused {} byte difference",
            diff
        );
    }

    #[test]
    fn test_tuple_index_contains_jge_for_negative_normalization() {
        // Must contain JGE (0x0F 0x8D or 0x7D) for the negative index branch
        let t = TupleIndexTemplate::new(0, 1, 2, 0);
        let code = emit_and_finalize(&t);
        let has_jge_short = code.windows(1).any(|w| w[0] == 0x7D);
        let has_jge_near = code.windows(2).any(|w| w[0] == 0x0F && w[1] == 0x8D);
        assert!(
            has_jge_short || has_jge_near,
            "TupleIndex must have JGE for negative index normalization"
        );
    }

    #[test]
    fn test_tuple_index_contains_jae_for_bounds_check() {
        // Must contain JAE (0x0F 0x83 or 0x73) for unsigned bounds check
        let t = TupleIndexTemplate::new(0, 1, 2, 0);
        let code = emit_and_finalize(&t);
        let has_jae_short = code.windows(1).any(|w| w[0] == 0x73);
        let has_jae_near = code.windows(2).any(|w| w[0] == 0x0F && w[1] == 0x83);
        assert!(
            has_jae_short || has_jae_near,
            "TupleIndex must have JAE for unsigned bounds check"
        );
    }

    #[test]
    fn test_tuple_index_high_register_indices() {
        // Verify encoding correctness with high register indices (REX prefix needed)
        let t = TupleIndexTemplate::new(10, 11, 12, 0);
        let code = emit_and_finalize(&t);
        assert!(
            !code.is_empty(),
            "High register indices must produce valid code"
        );
        assert!(
            code.len() <= t.estimated_size(),
            "High reg actual {} exceeds estimate {}",
            code.len(),
            t.estimated_size()
        );
    }

    #[test]
    fn test_tuple_index_deopt_idx_zero() {
        let t = TupleIndexTemplate::new(0, 1, 2, 0);
        let code = emit_and_finalize(&t);
        assert!(!code.is_empty());
    }

    #[test]
    fn test_tuple_index_deopt_idx_one() {
        let t = TupleIndexTemplate::new(0, 1, 2, 1);
        let code = emit_and_finalize(&t);
        assert!(!code.is_empty());
    }

    // =========================================================================
    // TupleLenTemplate Tests
    // =========================================================================

    #[test]
    fn test_tuple_len_template_creation() {
        let t = TupleLenTemplate::new(0, 1, 0);
        assert_eq!(t.dst_reg, 0);
        assert_eq!(t.tuple_reg, 1);
        assert_eq!(t.deopt_idx, 0);
    }

    #[test]
    fn test_tuple_len_template_estimated_size() {
        let t = TupleLenTemplate::new(0, 1, 0);
        assert!(
            t.estimated_size() >= 60,
            "TupleLen estimated size {} too small",
            t.estimated_size()
        );
    }

    #[test]
    fn test_tuple_len_template_emits_code() {
        let t = TupleLenTemplate::new(0, 1, 0);
        let code = emit_and_finalize(&t);
        assert!(!code.is_empty(), "TupleLenTemplate must emit non-zero code");
    }

    #[test]
    fn test_tuple_len_code_size_within_estimate() {
        let t = TupleLenTemplate::new(0, 1, 0);
        let code = emit_and_finalize(&t);
        assert!(
            code.len() <= t.estimated_size(),
            "TupleLen actual {} exceeds estimate {}",
            code.len(),
            t.estimated_size()
        );
    }

    #[test]
    fn test_tuple_len_different_registers() {
        for (dst, tup) in [(0, 1), (2, 3), (5, 6), (0, 0)] {
            let t = TupleLenTemplate::new(dst, tup, 0);
            let code = emit_and_finalize(&t);
            assert!(!code.is_empty(), "No code for regs ({}, {})", dst, tup);
        }
    }

    #[test]
    fn test_tuple_len_contains_int_tag_boxing() {
        // Must contain the INT_TAG pattern for NaN-boxing the length
        let t = TupleLenTemplate::new(0, 1, 0);
        let code = emit_and_finalize(&t);
        let int_tag_bytes = ((value_tags::QNAN_BITS | value_tags::INT_TAG) as i64).to_le_bytes();
        // At least part of the INT_TAG should appear in the code
        let has_tag = code.windows(8).any(|w| w == int_tag_bytes);
        assert!(
            has_tag,
            "TupleLen must contain INT_TAG immediate for boxing"
        );
    }

    #[test]
    fn test_tuple_len_smaller_than_tuple_index() {
        // Len is simpler than Index (no negative normalization, no bounds check)
        let len_tmpl = TupleLenTemplate::new(0, 1, 0);
        let idx_tmpl = TupleIndexTemplate::new(0, 1, 2, 0);
        let len_code = emit_and_finalize(&len_tmpl);
        let idx_code = emit_and_finalize(&idx_tmpl);
        assert!(
            len_code.len() < idx_code.len(),
            "TupleLen ({}) should be smaller than TupleIndex ({})",
            len_code.len(),
            idx_code.len()
        );
    }

    #[test]
    fn test_tuple_len_high_register_indices() {
        let t = TupleLenTemplate::new(10, 11, 0);
        let code = emit_and_finalize(&t);
        assert!(
            code.len() <= t.estimated_size(),
            "High reg actual {} exceeds estimate {}",
            code.len(),
            t.estimated_size()
        );
    }

    // =========================================================================
    // TupleContainsTemplate Tests
    // =========================================================================

    #[test]
    fn test_tuple_contains_template_creation() {
        let t = TupleContainsTemplate::new(1, 2, 0, 0);
        assert_eq!(t.tuple_reg, 1);
        assert_eq!(t.value_reg, 2);
        assert_eq!(t.dst_reg, 0);
        assert_eq!(t.deopt_idx, 0);
    }

    #[test]
    fn test_tuple_contains_template_estimated_size() {
        let t = TupleContainsTemplate::new(1, 2, 0, 0);
        assert!(
            t.estimated_size() >= 40,
            "TupleContains estimated size {} too small",
            t.estimated_size()
        );
    }

    #[test]
    fn test_tuple_contains_template_emits_code() {
        let t = TupleContainsTemplate::new(1, 2, 0, 0);
        let code = emit_and_finalize(&t);
        assert!(!code.is_empty());
    }

    #[test]
    fn test_tuple_contains_code_size_within_estimate() {
        let t = TupleContainsTemplate::new(1, 2, 0, 0);
        let code = emit_and_finalize(&t);
        assert!(
            code.len() <= t.estimated_size(),
            "TupleContains actual {} exceeds estimate {}",
            code.len(),
            t.estimated_size()
        );
    }

    #[test]
    fn test_tuple_contains_contains_jmp_for_deopt() {
        // Guard-only templates must have an unconditional JMP to deopt
        let t = TupleContainsTemplate::new(1, 2, 0, 0);
        let code = emit_and_finalize(&t);
        let has_jmp = code.contains(&0xE9) || code.contains(&0xEB);
        assert!(has_jmp, "TupleContains must have JMP for deopt path");
    }

    #[test]
    fn test_tuple_contains_different_registers() {
        for (tup, val, dst) in [(0, 1, 2), (3, 4, 5), (7, 8, 9)] {
            let t = TupleContainsTemplate::new(tup, val, dst, 0);
            let code = emit_and_finalize(&t);
            assert!(!code.is_empty());
        }
    }

    #[test]
    fn test_tuple_contains_smaller_than_tuple_index() {
        // Contains (guard only) should be notably smaller than Index (fully inline)
        let contains = TupleContainsTemplate::new(1, 2, 0, 0);
        let index = TupleIndexTemplate::new(0, 1, 2, 0);
        let c_code = emit_and_finalize(&contains);
        let i_code = emit_and_finalize(&index);
        assert!(
            c_code.len() < i_code.len(),
            "TupleContains ({}) should be smaller than TupleIndex ({})",
            c_code.len(),
            i_code.len()
        );
    }

    // =========================================================================
    // TupleConcatTemplate Tests
    // =========================================================================

    #[test]
    fn test_tuple_concat_template_creation() {
        let t = TupleConcatTemplate::new(0, 1, 2, 0);
        assert_eq!(t.dst_reg, 0);
        assert_eq!(t.lhs_reg, 1);
        assert_eq!(t.rhs_reg, 2);
        assert_eq!(t.deopt_idx, 0);
    }

    #[test]
    fn test_tuple_concat_template_estimated_size() {
        let t = TupleConcatTemplate::new(0, 1, 2, 0);
        assert!(
            t.estimated_size() >= 60,
            "TupleConcat estimated size {} too small",
            t.estimated_size()
        );
    }

    #[test]
    fn test_tuple_concat_template_emits_code() {
        let t = TupleConcatTemplate::new(0, 1, 2, 0);
        let code = emit_and_finalize(&t);
        assert!(!code.is_empty());
    }

    #[test]
    fn test_tuple_concat_code_size_within_estimate() {
        let t = TupleConcatTemplate::new(0, 1, 2, 0);
        let code = emit_and_finalize(&t);
        assert!(
            code.len() <= t.estimated_size(),
            "TupleConcat actual {} exceeds estimate {}",
            code.len(),
            t.estimated_size()
        );
    }

    #[test]
    fn test_tuple_concat_contains_jmp_for_deopt() {
        let t = TupleConcatTemplate::new(0, 1, 2, 0);
        let code = emit_and_finalize(&t);
        let has_jmp = code.contains(&0xE9) || code.contains(&0xEB);
        assert!(has_jmp, "TupleConcat must have JMP for deopt path");
    }

    #[test]
    fn test_tuple_concat_larger_than_contains() {
        // Concat has TWO type guards, Contains has ONE
        let concat = TupleConcatTemplate::new(0, 1, 2, 0);
        let contains = TupleContainsTemplate::new(1, 2, 0, 0);
        let concat_code = emit_and_finalize(&concat);
        let contains_code = emit_and_finalize(&contains);
        assert!(
            concat_code.len() > contains_code.len(),
            "TupleConcat ({}) should be larger than TupleContains ({})",
            concat_code.len(),
            contains_code.len()
        );
    }

    #[test]
    fn test_tuple_concat_different_registers() {
        for (dst, lhs, rhs) in [(0, 1, 2), (3, 4, 5), (6, 7, 8)] {
            let t = TupleConcatTemplate::new(dst, lhs, rhs, 0);
            let code = emit_and_finalize(&t);
            assert!(!code.is_empty());
        }
    }

    #[test]
    fn test_tuple_concat_high_register_indices() {
        let t = TupleConcatTemplate::new(10, 11, 12, 0);
        let code = emit_and_finalize(&t);
        assert!(
            code.len() <= t.estimated_size(),
            "High reg actual {} exceeds estimate {}",
            code.len(),
            t.estimated_size()
        );
    }

    // =========================================================================
    // TupleRepeatTemplate Tests
    // =========================================================================

    #[test]
    fn test_tuple_repeat_template_creation() {
        let t = TupleRepeatTemplate::new(0, 1, 2, 0);
        assert_eq!(t.dst_reg, 0);
        assert_eq!(t.tuple_reg, 1);
        assert_eq!(t.count_reg, 2);
        assert_eq!(t.deopt_idx, 0);
    }

    #[test]
    fn test_tuple_repeat_template_estimated_size() {
        let t = TupleRepeatTemplate::new(0, 1, 2, 0);
        assert!(
            t.estimated_size() >= 60,
            "TupleRepeat estimated size {} too small",
            t.estimated_size()
        );
    }

    #[test]
    fn test_tuple_repeat_template_emits_code() {
        let t = TupleRepeatTemplate::new(0, 1, 2, 0);
        let code = emit_and_finalize(&t);
        assert!(!code.is_empty());
    }

    #[test]
    fn test_tuple_repeat_code_size_within_estimate() {
        let t = TupleRepeatTemplate::new(0, 1, 2, 0);
        let code = emit_and_finalize(&t);
        assert!(
            code.len() <= t.estimated_size(),
            "TupleRepeat actual {} exceeds estimate {}",
            code.len(),
            t.estimated_size()
        );
    }

    #[test]
    fn test_tuple_repeat_contains_jmp_for_deopt() {
        let t = TupleRepeatTemplate::new(0, 1, 2, 0);
        let code = emit_and_finalize(&t);
        let has_jmp = code.contains(&0xE9) || code.contains(&0xEB);
        assert!(has_jmp, "TupleRepeat must have JMP for deopt path");
    }

    #[test]
    fn test_tuple_repeat_larger_than_contains() {
        // Repeat has tuple guard + int guard, Contains has tuple guard only
        let repeat = TupleRepeatTemplate::new(0, 1, 2, 0);
        let contains = TupleContainsTemplate::new(1, 2, 0, 0);
        let r_code = emit_and_finalize(&repeat);
        let c_code = emit_and_finalize(&contains);
        assert!(
            r_code.len() > c_code.len(),
            "TupleRepeat ({}) should be larger than TupleContains ({})",
            r_code.len(),
            c_code.len()
        );
    }

    #[test]
    fn test_tuple_repeat_different_registers() {
        for (dst, tup, cnt) in [(0, 1, 2), (3, 4, 5), (6, 7, 8)] {
            let t = TupleRepeatTemplate::new(dst, tup, cnt, 0);
            let code = emit_and_finalize(&t);
            assert!(!code.is_empty());
        }
    }

    #[test]
    fn test_tuple_repeat_high_register_indices() {
        let t = TupleRepeatTemplate::new(10, 11, 12, 0);
        let code = emit_and_finalize(&t);
        assert!(
            code.len() <= t.estimated_size(),
            "High reg actual {} exceeds estimate {}",
            code.len(),
            t.estimated_size()
        );
    }

    // =========================================================================
    // Cross-Template Tests
    // =========================================================================

    #[test]
    fn test_all_templates_compile() {
        // Smoke test: all 5 templates compile and emit non-zero code
        let templates: Vec<Box<dyn OpcodeTemplate>> = vec![
            Box::new(TupleIndexTemplate::new(0, 1, 2, 0)),
            Box::new(TupleLenTemplate::new(0, 1, 0)),
            Box::new(TupleContainsTemplate::new(1, 2, 0, 0)),
            Box::new(TupleConcatTemplate::new(0, 1, 2, 0)),
            Box::new(TupleRepeatTemplate::new(0, 1, 2, 0)),
        ];
        for (i, tmpl) in templates.iter().enumerate() {
            let code = emit_and_finalize(tmpl.as_ref());
            assert!(!code.is_empty(), "Template {} emitted empty code", i);
        }
    }

    #[test]
    fn test_all_estimates_are_conservative() {
        let templates: Vec<Box<dyn OpcodeTemplate>> = vec![
            Box::new(TupleIndexTemplate::new(0, 1, 2, 0)),
            Box::new(TupleLenTemplate::new(0, 1, 0)),
            Box::new(TupleContainsTemplate::new(1, 2, 0, 0)),
            Box::new(TupleConcatTemplate::new(0, 1, 2, 0)),
            Box::new(TupleRepeatTemplate::new(0, 1, 2, 0)),
        ];
        for (i, tmpl) in templates.iter().enumerate() {
            let code = emit_and_finalize(tmpl.as_ref());
            assert!(
                code.len() <= tmpl.estimated_size(),
                "Template {} actual {} exceeds estimate {}",
                i,
                code.len(),
                tmpl.estimated_size()
            );
        }
    }

    #[test]
    fn test_sequential_emission() {
        // Emit all templates into a single assembler buffer
        let mut asm = Assembler::new();
        let frame = FrameLayout::minimal(16);
        {
            let mut ctx = TemplateContext::new(&mut asm, &frame);
            ctx.create_deopt_label();
            ctx.create_deopt_label();

            TupleIndexTemplate::new(0, 1, 2, 0).emit(&mut ctx);
            let after_index = ctx.asm.offset();

            TupleLenTemplate::new(3, 4, 0).emit(&mut ctx);
            let after_len = ctx.asm.offset();

            TupleContainsTemplate::new(5, 6, 7, 0).emit(&mut ctx);
            let after_contains = ctx.asm.offset();

            TupleConcatTemplate::new(8, 9, 10, 0).emit(&mut ctx);
            let after_concat = ctx.asm.offset();

            TupleRepeatTemplate::new(11, 12, 13, 0).emit(&mut ctx);
            let after_repeat = ctx.asm.offset();

            // Each template should contribute non-zero bytes
            assert!(after_index > 0, "Index must emit code");
            assert!(after_len > after_index, "Len must add to buffer");
            assert!(after_contains > after_len, "Contains must add to buffer");
            assert!(after_concat > after_contains, "Concat must add to buffer");
            assert!(after_repeat > after_concat, "Repeat must add to buffer");

            for l in &ctx.deopt_labels {
                ctx.asm.bind_label(*l);
            }
        }
        let code = asm.finalize().unwrap();
        assert!(code.len() > 100, "Combined code {} too small", code.len());
    }

    #[test]
    fn test_fully_inline_templates_are_larger_than_guard_only() {
        // TupleIndex (fully inline) should be larger than all guard-only templates
        let index = TupleIndexTemplate::new(0, 1, 2, 0);
        let index_code = emit_and_finalize(&index);

        let guard_templates: Vec<Box<dyn OpcodeTemplate>> = vec![
            Box::new(TupleContainsTemplate::new(1, 2, 0, 0)),
            Box::new(TupleConcatTemplate::new(0, 1, 2, 0)),
            Box::new(TupleRepeatTemplate::new(0, 1, 2, 0)),
        ];

        for (i, tmpl) in guard_templates.iter().enumerate() {
            let code = emit_and_finalize(tmpl.as_ref());
            assert!(
                index_code.len() >= code.len(),
                "TupleIndex ({}) should be >= guard template {} ({})",
                index_code.len(),
                i,
                code.len()
            );
        }
    }

    #[test]
    fn test_zero_register_indices() {
        // Edge case: register 0 for all operands
        let t = TupleIndexTemplate::new(0, 0, 0, 0);
        let code = emit_and_finalize(&t);
        assert!(!code.is_empty(), "Zero registers must produce valid code");
    }

    #[test]
    fn test_adjacent_register_indices() {
        // Edge case: sequential registers
        let t = TupleIndexTemplate::new(0, 1, 2, 0);
        let code = emit_and_finalize(&t);
        assert!(
            !code.is_empty(),
            "Adjacent registers must produce valid code"
        );
    }

    #[test]
    fn test_multiple_deopt_labels() {
        // Test with multiple deopt labels in the context
        let mut asm = Assembler::new();
        let frame = FrameLayout::minimal(16);
        {
            let mut ctx = TemplateContext::new(&mut asm, &frame);
            // Create 4 deopt labels
            for _ in 0..4 {
                ctx.create_deopt_label();
            }

            // Use different deopt indices
            TupleIndexTemplate::new(0, 1, 2, 0).emit(&mut ctx);
            TupleLenTemplate::new(3, 4, 1).emit(&mut ctx);
            TupleContainsTemplate::new(5, 6, 7, 2).emit(&mut ctx);
            TupleConcatTemplate::new(8, 9, 10, 3).emit(&mut ctx);

            for l in &ctx.deopt_labels {
                ctx.asm.bind_label(*l);
            }
        }
        let code = asm.finalize().unwrap();
        assert!(!code.is_empty());
    }

    #[test]
    fn test_tuple_type_guard_uses_correct_type_id() {
        // Verify the tuple type guard emits code checking for TUPLE type ID (7)
        // by comparing code size against a known-good dict guard
        let mut asm_tuple = Assembler::new();
        let mut asm_dict = Assembler::new();
        let frame = FrameLayout::minimal(8);

        {
            let mut ctx = TemplateContext::new(&mut asm_tuple, &frame);
            ctx.create_deopt_label();
            let acc = ctx.regs.accumulator;
            let s1 = ctx.regs.scratch1;
            emit_tuple_check_and_extract(&mut ctx, acc, acc, s1, 0);
            for l in &ctx.deopt_labels {
                ctx.asm.bind_label(*l);
            }
        }

        {
            let mut ctx = TemplateContext::new(&mut asm_dict, &frame);
            ctx.create_deopt_label();
            let acc = ctx.regs.accumulator;
            let s1 = ctx.regs.scratch1;
            emit_typed_object_check_and_extract(&mut ctx, acc, acc, s1, type_ids::DICT, 0);
            for l in &ctx.deopt_labels {
                ctx.asm.bind_label(*l);
            }
        }

        let tuple_code = asm_tuple.finalize().unwrap();
        let dict_code = asm_dict.finalize().unwrap();

        // Same guard structure, same size (only the immediate type_id differs)
        assert_eq!(
            tuple_code.len(),
            dict_code.len(),
            "Tuple and dict guards should be same size"
        );
    }

    #[test]
    fn test_guard_only_templates_have_unconditional_deopt() {
        // All guard-only templates must end with an unconditional JMP to deopt
        let guard_templates: Vec<Box<dyn OpcodeTemplate>> = vec![
            Box::new(TupleContainsTemplate::new(1, 2, 0, 0)),
            Box::new(TupleConcatTemplate::new(0, 1, 2, 0)),
            Box::new(TupleRepeatTemplate::new(0, 1, 2, 0)),
        ];

        for (i, tmpl) in guard_templates.iter().enumerate() {
            let code = emit_and_finalize(tmpl.as_ref());
            // The code should contain either JMP rel8 (0xEB) or JMP rel32 (0xE9)
            let has_jmp_rel32 = code.contains(&0xE9);
            let has_jmp_rel8 = code.contains(&0xEB);
            assert!(
                has_jmp_rel32 || has_jmp_rel8,
                "Guard template {} missing unconditional JMP",
                i
            );
        }
    }

    #[test]
    fn test_tuple_len_uses_items_len_offset() {
        // Verify that TupleLen uses offset 24 (ITEMS_LEN_OFFSET) somewhere in the code.
        // offset 24 = 0x18 as a displacement byte in ModR/M encoding
        let t = TupleLenTemplate::new(0, 1, 0);
        let code = emit_and_finalize(&t);
        let has_disp_24 = code.contains(&0x18);
        assert!(
            has_disp_24,
            "TupleLen must reference displacement 24 (0x18) for items.len"
        );
    }

    #[test]
    fn test_tuple_index_uses_items_ptr_offset() {
        // TupleIndex loads items.ptr at offset 16 (0x10)
        let t = TupleIndexTemplate::new(0, 1, 2, 0);
        let code = emit_and_finalize(&t);
        let has_disp_16 = code.contains(&0x10);
        assert!(
            has_disp_16,
            "TupleIndex must reference displacement 16 (0x10) for items.ptr"
        );
    }

    #[test]
    fn test_tuple_index_uses_scale_x8() {
        // TupleIndex must use Scale::X8 (SIB byte scale field = 3) for
        // items[index * 8]. The SIB byte encodes: scale(2) | index(3) | base(3).
        // Scale::X8 = 0b11 in bits 7:6 of SIB, so SIB & 0xC0 == 0xC0.
        let t = TupleIndexTemplate::new(0, 1, 2, 0);
        let code = emit_and_finalize(&t);
        let has_scale_8 = code.windows(1).any(|w| (w[0] & 0xC0) == 0xC0);
        assert!(
            has_scale_8,
            "TupleIndex must use Scale::X8 for indexed access"
        );
    }
}
