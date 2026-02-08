//! Enhanced list operation templates for the Tier 1 JIT.
//!
//! Builds on the core list specialization in `list_specialize.rs` to add:
//! - **Pop last** (`list.pop()`) — fully inline: decrement len, load item
//! - **Length** (`len(list)`) — fully inline: load vec.len, box as int
//! - **Clear** (`list.clear()`) — fully inline: set vec.len = 0
//! - **Contains** (`x in list`) — guard-only: type check, deopt for scan
//! - **Insert** (`list.insert(i, v)`) — guard-only: type checks, deopt for memmove
//! - **Remove** (`list.remove(i)`) — guard-only: type checks, deopt for memmove
//!
//! All templates reuse helpers from `list_specialize.rs` via `pub(super)` visibility.

use super::list_specialize::{
    emit_int_check_and_extract_for_list, emit_list_check_and_extract, list_layout,
};
use super::{OpcodeTemplate, TemplateContext, value_tags};
use crate::backend::x64::registers::{MemOperand, Scale};

// =============================================================================
// ListPopLastTemplate
// =============================================================================

/// Template for inline pop from list end (`list.pop()` with no arguments).
///
/// # Code Generation Strategy
///
/// 1. Type-check list (OBJECT_TAG + TypeId::LIST)
/// 2. Load len from Vec header
/// 3. Guard: len > 0 (empty list → deopt to raise IndexError)
/// 4. Decrement len: `len -= 1`
/// 5. Write new len back to Vec header
/// 6. Load items pointer
/// 7. Load `items[new_len]` (the popped element)
/// 8. Store result to destination
///
/// # Performance
///
/// This is the optimal pop implementation: no memmove required since we pop
/// from the end. O(1) with only a bounds check and indexed load.
///
/// # Estimated Code Size
///
/// ~160 bytes: tag check (20), type_id check (16), len load (12),
/// empty guard (8), len decrement + store (16), items load (12),
/// indexed load (16), result store (8), overhead (52)
pub struct ListPopLastTemplate {
    /// Destination register slot for the popped value.
    pub dst_reg: u8,
    /// Register slot containing the list.
    pub list_reg: u8,
    /// Deoptimization label index.
    pub deopt_idx: usize,
}

impl ListPopLastTemplate {
    /// Create a new pop-last template.
    pub fn new(dst_reg: u8, list_reg: u8, deopt_idx: usize) -> Self {
        Self {
            dst_reg,
            list_reg,
            deopt_idx,
        }
    }
}

impl OpcodeTemplate for ListPopLastTemplate {
    fn emit(&self, ctx: &mut TemplateContext) {
        let acc = ctx.regs.accumulator;
        let scratch1 = ctx.regs.scratch1;
        let scratch2 = ctx.regs.scratch2;

        // Load list value from frame
        let list_slot = ctx.frame.register_slot(self.list_reg as u16);
        ctx.asm.mov_rm(acc, &list_slot);

        // Type-check: OBJECT_TAG + TypeId::LIST
        emit_list_check_and_extract(ctx, acc, acc, scratch1, self.deopt_idx);
        // acc = ListObject pointer

        // Load length: scratch1 = list->vec.len
        let len_mem = MemOperand {
            base: Some(acc),
            index: None,
            scale: Scale::X1,
            disp: list_layout::ITEMS_LEN_OFFSET,
        };
        ctx.asm.mov_rm(scratch1, &len_mem);

        // Guard: len > 0 (empty → deopt to raise IndexError)
        ctx.asm.cmp_ri(scratch1, 0);
        ctx.asm.je(ctx.deopt_label(self.deopt_idx));

        // Decrement length: scratch1 = len - 1
        ctx.asm.dec(scratch1);

        // Write new length back
        ctx.asm.mov_mr(&len_mem, scratch1);

        // Load items pointer: scratch2 = list->vec.ptr
        let items_mem = MemOperand {
            base: Some(acc),
            index: None,
            scale: Scale::X1,
            disp: list_layout::ITEMS_PTR_OFFSET,
        };
        ctx.asm.mov_rm(scratch2, &items_mem);

        // Load popped value: acc = items[new_len]
        let item_mem = MemOperand {
            base: Some(scratch2),
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
        160
    }
}

// =============================================================================
// ListLenTemplate
// =============================================================================

/// Template for inline list length (`len(list)`).
///
/// # Code Generation Strategy
///
/// 1. Type-check list (OBJECT_TAG + TypeId::LIST)
/// 2. Load len from Vec header
/// 3. NaN-box the length as INT_TAG
/// 4. Store result to destination
///
/// # Performance
///
/// Trivial operation: one memory load + int boxing. This eliminates the
/// overhead of `__len__` method dispatch entirely.
///
/// # Estimated Code Size
///
/// ~120 bytes: tag check (20), type_id check (16), len load (12),
/// int boxing (32), result store (8), overhead (32)
pub struct ListLenTemplate {
    /// Destination register slot for the length value.
    pub dst_reg: u8,
    /// Register slot containing the list.
    pub list_reg: u8,
    /// Deoptimization label index.
    pub deopt_idx: usize,
}

impl ListLenTemplate {
    /// Create a new list length template.
    pub fn new(dst_reg: u8, list_reg: u8, deopt_idx: usize) -> Self {
        Self {
            dst_reg,
            list_reg,
            deopt_idx,
        }
    }
}

impl OpcodeTemplate for ListLenTemplate {
    fn emit(&self, ctx: &mut TemplateContext) {
        let acc = ctx.regs.accumulator;
        let scratch1 = ctx.regs.scratch1;

        // Load list value from frame
        let list_slot = ctx.frame.register_slot(self.list_reg as u16);
        ctx.asm.mov_rm(acc, &list_slot);

        // Type-check: OBJECT_TAG + TypeId::LIST
        emit_list_check_and_extract(ctx, acc, acc, scratch1, self.deopt_idx);
        // acc = ListObject pointer

        // Load length: acc = list->vec.len
        let len_mem = MemOperand {
            base: Some(acc),
            index: None,
            scale: Scale::X1,
            disp: list_layout::ITEMS_LEN_OFFSET,
        };
        ctx.asm.mov_rm(acc, &len_mem);

        // NaN-box the length as integer
        // result = (QNAN_BITS | INT_TAG) | (len & PAYLOAD_MASK)
        ctx.asm.mov_ri64(scratch1, value_tags::PAYLOAD_MASK as i64);
        ctx.asm.and_rr(acc, scratch1);
        let tag = (value_tags::QNAN_BITS | value_tags::INT_TAG) as i64;
        ctx.asm.mov_ri64(scratch1, tag);
        ctx.asm.or_rr(acc, scratch1);

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
// ListClearTemplate
// =============================================================================

/// Template for inline list clear (`list.clear()`).
///
/// # Code Generation Strategy
///
/// 1. Type-check list (OBJECT_TAG + TypeId::LIST)
/// 2. Set vec.len = 0
///
/// This does NOT deallocate the backing array (same as Python's `list.clear()`).
/// The capacity is preserved for potential reuse, which is the optimal behavior
/// for lists that are cleared and refilled in loops.
///
/// # Performance
///
/// Single memory store after type guard. O(1). Note that unlike Python's
/// `list.clear()`, we don't need to drop individual elements since Values
/// are Copy types (NaN-boxed u64).
///
/// # Estimated Code Size
///
/// ~100 bytes: tag check (20), type_id check (16), zero store (16),
/// overhead (48)
pub struct ListClearTemplate {
    /// Register slot containing the list.
    pub list_reg: u8,
    /// Deoptimization label index.
    pub deopt_idx: usize,
}

impl ListClearTemplate {
    /// Create a new list clear template.
    pub fn new(list_reg: u8, deopt_idx: usize) -> Self {
        Self {
            list_reg,
            deopt_idx,
        }
    }
}

impl OpcodeTemplate for ListClearTemplate {
    fn emit(&self, ctx: &mut TemplateContext) {
        let acc = ctx.regs.accumulator;
        let scratch1 = ctx.regs.scratch1;

        // Load list value from frame
        let list_slot = ctx.frame.register_slot(self.list_reg as u16);
        ctx.asm.mov_rm(acc, &list_slot);

        // Type-check: OBJECT_TAG + TypeId::LIST
        emit_list_check_and_extract(ctx, acc, acc, scratch1, self.deopt_idx);
        // acc = ListObject pointer

        // Set vec.len = 0
        ctx.asm.xor_rr(scratch1, scratch1); // scratch1 = 0
        let len_mem = MemOperand {
            base: Some(acc),
            index: None,
            scale: Scale::X1,
            disp: list_layout::ITEMS_LEN_OFFSET,
        };
        ctx.asm.mov_mr(&len_mem, scratch1);
    }

    #[inline]
    fn estimated_size(&self) -> usize {
        100
    }
}

// =============================================================================
// ListContainsTemplate
// =============================================================================

/// Template for list containment check (`value in list`).
///
/// # Code Generation Strategy
///
/// 1. Type-check list operand (OBJECT_TAG + TypeId::LIST)
/// 2. Deopt to interpreter for actual linear scan
///
/// The inline type guard eliminates the overhead of generic `__contains__`
/// dispatch. The actual linear scan is deferred to the interpreter because
/// it requires a loop with per-element equality comparison, which would
/// generate too much code for Tier 1.
///
/// Tier 2 can inline the scan loop for homogeneous-type lists.
///
/// # Estimated Code Size
///
/// ~80 bytes: tag check (20), type_id check (16), deopt (8), overhead (36)
pub struct ListContainsTemplate {
    /// Destination register slot for the boolean result.
    pub dst_reg: u8,
    /// Register slot containing the list.
    pub list_reg: u8,
    /// Register slot containing the value to search for.
    pub value_reg: u8,
    /// Deoptimization label index.
    pub deopt_idx: usize,
}

impl ListContainsTemplate {
    /// Create a new list contains template.
    pub fn new(dst_reg: u8, list_reg: u8, value_reg: u8, deopt_idx: usize) -> Self {
        Self {
            dst_reg,
            list_reg,
            value_reg,
            deopt_idx,
        }
    }
}

impl OpcodeTemplate for ListContainsTemplate {
    fn emit(&self, ctx: &mut TemplateContext) {
        let acc = ctx.regs.accumulator;
        let scratch1 = ctx.regs.scratch1;

        // Type-check list operand
        let list_slot = ctx.frame.register_slot(self.list_reg as u16);
        ctx.asm.mov_rm(acc, &list_slot);
        emit_list_check_and_extract(ctx, acc, acc, scratch1, self.deopt_idx);

        // Deopt to interpreter for actual containment check
        // The type guard alone eliminates generic __contains__ dispatch overhead
        ctx.asm.jmp(ctx.deopt_label(self.deopt_idx));
    }

    #[inline]
    fn estimated_size(&self) -> usize {
        80
    }
}

// =============================================================================
// ListInsertTemplate
// =============================================================================

/// Template for list insertion (`list.insert(index, value)`).
///
/// # Code Generation Strategy
///
/// 1. Type-check list operand (OBJECT_TAG + TypeId::LIST)
/// 2. Type-check index operand (INT_TAG)
/// 3. Deopt to interpreter for actual insertion
///
/// Insert requires:
/// - Potential reallocation (if len == cap)
/// - memmove to shift elements right
/// - Index normalization (negative indices, clamping to [0, len])
///
/// These operations are too complex and code-size-expensive for Tier 1 inline
/// emission. The type guards still eliminate method dispatch overhead.
///
/// # Estimated Code Size
///
/// ~120 bytes: 2×tag checks (40), type_id check (16), deopt (8),
/// overhead (56)
pub struct ListInsertTemplate {
    /// Register slot containing the list.
    pub list_reg: u8,
    /// Register slot containing the index.
    pub index_reg: u8,
    /// Register slot containing the value to insert.
    pub value_reg: u8,
    /// Deoptimization label index.
    pub deopt_idx: usize,
}

impl ListInsertTemplate {
    /// Create a new list insert template.
    pub fn new(list_reg: u8, index_reg: u8, value_reg: u8, deopt_idx: usize) -> Self {
        Self {
            list_reg,
            index_reg,
            value_reg,
            deopt_idx,
        }
    }
}

impl OpcodeTemplate for ListInsertTemplate {
    fn emit(&self, ctx: &mut TemplateContext) {
        let acc = ctx.regs.accumulator;
        let scratch1 = ctx.regs.scratch1;
        let scratch2 = ctx.regs.scratch2;

        // Type-check list operand
        let list_slot = ctx.frame.register_slot(self.list_reg as u16);
        ctx.asm.mov_rm(acc, &list_slot);
        emit_list_check_and_extract(ctx, acc, acc, scratch1, self.deopt_idx);

        // Type-check index operand
        let index_slot = ctx.frame.register_slot(self.index_reg as u16);
        ctx.asm.mov_rm(scratch1, &index_slot);
        emit_int_check_and_extract_for_list(ctx, scratch1, scratch1, scratch2, self.deopt_idx);

        // Deopt to interpreter for memmove + potential realloc
        ctx.asm.jmp(ctx.deopt_label(self.deopt_idx));
    }

    #[inline]
    fn estimated_size(&self) -> usize {
        120
    }
}

// =============================================================================
// ListRemoveTemplate
// =============================================================================

/// Template for list removal by index (`list.pop(index)` / `del list[index]`).
///
/// # Code Generation Strategy
///
/// 1. Type-check list operand (OBJECT_TAG + TypeId::LIST)
/// 2. Type-check index operand (INT_TAG)
/// 3. Deopt to interpreter for actual removal
///
/// Remove requires memmove to shift elements left after the removal point,
/// which is too complex for Tier 1 inline emission. The type guards eliminate
/// method dispatch overhead.
///
/// # Estimated Code Size
///
/// ~120 bytes: 2×tag checks (40), type_id check (16), deopt (8),
/// overhead (56)
pub struct ListRemoveTemplate {
    /// Destination register slot for the removed value.
    pub dst_reg: u8,
    /// Register slot containing the list.
    pub list_reg: u8,
    /// Register slot containing the index.
    pub index_reg: u8,
    /// Deoptimization label index.
    pub deopt_idx: usize,
}

impl ListRemoveTemplate {
    /// Create a new list remove template.
    pub fn new(dst_reg: u8, list_reg: u8, index_reg: u8, deopt_idx: usize) -> Self {
        Self {
            dst_reg,
            list_reg,
            index_reg,
            deopt_idx,
        }
    }
}

impl OpcodeTemplate for ListRemoveTemplate {
    fn emit(&self, ctx: &mut TemplateContext) {
        let acc = ctx.regs.accumulator;
        let scratch1 = ctx.regs.scratch1;
        let scratch2 = ctx.regs.scratch2;

        // Type-check list operand
        let list_slot = ctx.frame.register_slot(self.list_reg as u16);
        ctx.asm.mov_rm(acc, &list_slot);
        emit_list_check_and_extract(ctx, acc, acc, scratch1, self.deopt_idx);

        // Type-check index operand
        let index_slot = ctx.frame.register_slot(self.index_reg as u16);
        ctx.asm.mov_rm(scratch1, &index_slot);
        emit_int_check_and_extract_for_list(ctx, scratch1, scratch1, scratch2, self.deopt_idx);

        // Deopt to interpreter for memmove
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
    use crate::tier1::template::list_specialize::list_layout;

    // =========================================================================
    // Test Helper
    // =========================================================================

    /// Emit a template into an assembler, bind all deopt labels, and finalize.
    fn emit_and_finalize(template: &dyn OpcodeTemplate) -> Vec<u8> {
        let mut asm = Assembler::new();
        let frame = FrameLayout::minimal(16);
        {
            let mut ctx = TemplateContext::new(&mut asm, &frame);
            let _deopt = ctx.create_deopt_label();
            template.emit(&mut ctx);
            // Bind all deopt labels to current position
            for label in &ctx.deopt_labels {
                ctx.asm.bind_label(*label);
            }
        }
        asm.finalize()
            .expect("assembler finalization should succeed")
    }

    /// Create a template context with a single deopt label for testing.
    fn make_ctx<'a>(asm: &'a mut Assembler, frame: &'a FrameLayout) -> TemplateContext<'a> {
        let mut ctx = TemplateContext::new(asm, frame);
        ctx.create_deopt_label();
        ctx
    }

    // =========================================================================
    // ListPopLastTemplate Tests
    // =========================================================================

    #[test]
    fn test_pop_last_creates() {
        let tmpl = ListPopLastTemplate::new(1, 0, 0);
        assert_eq!(tmpl.dst_reg, 1);
        assert_eq!(tmpl.list_reg, 0);
        assert_eq!(tmpl.deopt_idx, 0);
    }

    #[test]
    fn test_pop_last_estimated_size() {
        let tmpl = ListPopLastTemplate::new(1, 0, 0);
        assert_eq!(tmpl.estimated_size(), 160);
    }

    #[test]
    fn test_pop_last_emits_code() {
        let code = emit_and_finalize(&ListPopLastTemplate::new(1, 0, 0));
        assert!(code.len() > 30, "pop_last too short: {}", code.len());
    }

    #[test]
    fn test_pop_last_code_size_within_estimate() {
        let mut asm = Assembler::new();
        let frame = FrameLayout::minimal(16);
        let mut ctx = make_ctx(&mut asm, &frame);
        let tmpl = ListPopLastTemplate::new(1, 0, 0);
        tmpl.emit(&mut ctx);
        assert!(
            ctx.asm.offset() <= tmpl.estimated_size(),
            "pop_last code ({}) exceeds estimate ({})",
            ctx.asm.offset(),
            tmpl.estimated_size()
        );
    }

    #[test]
    fn test_pop_last_different_registers() {
        let code1 = emit_and_finalize(&ListPopLastTemplate::new(0, 1, 0));
        let code2 = emit_and_finalize(&ListPopLastTemplate::new(2, 3, 0));
        // Different register slots produce different code (different frame offsets)
        assert_ne!(code1, code2);
    }

    #[test]
    fn test_pop_last_self_alias() {
        // dst_reg == list_reg should still compile (overwrite is fine after loading)
        let code = emit_and_finalize(&ListPopLastTemplate::new(0, 0, 0));
        assert!(!code.is_empty());
    }

    #[test]
    fn test_pop_last_high_registers() {
        let code = emit_and_finalize(&ListPopLastTemplate::new(15, 14, 0));
        assert!(!code.is_empty());
    }

    #[test]
    fn test_pop_last_contains_dec_instruction() {
        let mut asm = Assembler::new();
        let frame = FrameLayout::minimal(16);
        let mut ctx = make_ctx(&mut asm, &frame);
        let tmpl = ListPopLastTemplate::new(1, 0, 0);
        tmpl.emit(&mut ctx);

        // DEC r64 encodes as REX.W FF /1 — look for 0xFF with mod=11, reg=1
        let code = ctx.asm.code();
        let has_dec = code.windows(2).any(|w| {
            // FF byte with next byte's reg field = 1 (DEC) and mod=11
            w[0] == 0xFF && (w[1] & 0xC8) == 0xC8
        });
        // DEC might be encoded differently depending on REX prefix presence
        // Also accept the REX.W prefix variant
        let has_dec_with_rex = code
            .windows(3)
            .any(|w| (w[0] & 0xF0) == 0x40 && w[1] == 0xFF && (w[2] & 0xC8) == 0xC8);
        assert!(
            has_dec || has_dec_with_rex,
            "DEC instruction not found in pop_last emitted code"
        );
    }

    #[test]
    fn test_pop_last_contains_je_for_empty_guard() {
        let mut asm = Assembler::new();
        let frame = FrameLayout::minimal(16);
        let mut ctx = make_ctx(&mut asm, &frame);
        let tmpl = ListPopLastTemplate::new(1, 0, 0);
        tmpl.emit(&mut ctx);

        // JE near is 0F 84 rel32 or JE short is 74 rel8
        let code = ctx.asm.code();
        let has_je_short = code.iter().any(|&b| b == 0x74);
        let has_je_near = code.windows(2).any(|w| w[0] == 0x0F && w[1] == 0x84);
        assert!(
            has_je_short || has_je_near,
            "JE (empty guard) not found in pop_last code"
        );
    }

    #[test]
    fn test_pop_last_larger_than_clear() {
        let pop_code = emit_and_finalize(&ListPopLastTemplate::new(1, 0, 0));
        let clear_code = emit_and_finalize(&ListClearTemplate::new(0, 0));
        assert!(
            pop_code.len() > clear_code.len(),
            "pop_last ({}) should be larger than clear ({})",
            pop_code.len(),
            clear_code.len()
        );
    }

    // =========================================================================
    // ListLenTemplate Tests
    // =========================================================================

    #[test]
    fn test_len_creates() {
        let tmpl = ListLenTemplate::new(1, 0, 0);
        assert_eq!(tmpl.dst_reg, 1);
        assert_eq!(tmpl.list_reg, 0);
        assert_eq!(tmpl.deopt_idx, 0);
    }

    #[test]
    fn test_len_estimated_size() {
        let tmpl = ListLenTemplate::new(1, 0, 0);
        assert_eq!(tmpl.estimated_size(), 120);
    }

    #[test]
    fn test_len_emits_code() {
        let code = emit_and_finalize(&ListLenTemplate::new(1, 0, 0));
        assert!(code.len() > 20, "len too short: {}", code.len());
    }

    #[test]
    fn test_len_code_size_within_estimate() {
        let mut asm = Assembler::new();
        let frame = FrameLayout::minimal(16);
        let mut ctx = make_ctx(&mut asm, &frame);
        let tmpl = ListLenTemplate::new(1, 0, 0);
        tmpl.emit(&mut ctx);
        assert!(
            ctx.asm.offset() <= tmpl.estimated_size(),
            "len code ({}) exceeds estimate ({})",
            ctx.asm.offset(),
            tmpl.estimated_size()
        );
    }

    #[test]
    fn test_len_different_registers() {
        let code1 = emit_and_finalize(&ListLenTemplate::new(0, 1, 0));
        let code2 = emit_and_finalize(&ListLenTemplate::new(3, 4, 0));
        assert_ne!(code1, code2);
    }

    #[test]
    fn test_len_self_alias() {
        let code = emit_and_finalize(&ListLenTemplate::new(0, 0, 0));
        assert!(!code.is_empty());
    }

    #[test]
    fn test_len_high_registers() {
        let code = emit_and_finalize(&ListLenTemplate::new(15, 14, 0));
        assert!(!code.is_empty());
    }

    #[test]
    fn test_len_contains_int_tag_boxing() {
        let mut asm = Assembler::new();
        let frame = FrameLayout::minimal(16);
        let mut ctx = make_ctx(&mut asm, &frame);
        let tmpl = ListLenTemplate::new(1, 0, 0);
        tmpl.emit(&mut ctx);

        // The INT_TAG constant should appear in the code as a movabs immediate
        let code = ctx.asm.code();
        let tag_bytes = ((value_tags::QNAN_BITS | value_tags::INT_TAG) as i64).to_le_bytes();
        let has_tag = code.windows(8).any(|w| w == tag_bytes);
        assert!(has_tag, "INT_TAG constant not found in len emitted code");
    }

    #[test]
    fn test_len_generates_substantial_code() {
        // Len produces substantial code due to int boxing (movabs for tag)
        let len_code = emit_and_finalize(&ListLenTemplate::new(1, 0, 0));
        assert!(
            len_code.len() > 50,
            "len should produce substantial code for type guard + int boxing, got {}",
            len_code.len()
        );
    }

    // =========================================================================
    // ListClearTemplate Tests
    // =========================================================================

    #[test]
    fn test_clear_creates() {
        let tmpl = ListClearTemplate::new(0, 0);
        assert_eq!(tmpl.list_reg, 0);
        assert_eq!(tmpl.deopt_idx, 0);
    }

    #[test]
    fn test_clear_estimated_size() {
        let tmpl = ListClearTemplate::new(0, 0);
        assert_eq!(tmpl.estimated_size(), 100);
    }

    #[test]
    fn test_clear_emits_code() {
        let code = emit_and_finalize(&ListClearTemplate::new(0, 0));
        assert!(code.len() > 15, "clear too short: {}", code.len());
    }

    #[test]
    fn test_clear_code_size_within_estimate() {
        let mut asm = Assembler::new();
        let frame = FrameLayout::minimal(16);
        let mut ctx = make_ctx(&mut asm, &frame);
        let tmpl = ListClearTemplate::new(0, 0);
        tmpl.emit(&mut ctx);
        assert!(
            ctx.asm.offset() <= tmpl.estimated_size(),
            "clear code ({}) exceeds estimate ({})",
            ctx.asm.offset(),
            tmpl.estimated_size()
        );
    }

    #[test]
    fn test_clear_different_registers() {
        let code1 = emit_and_finalize(&ListClearTemplate::new(0, 0));
        let code2 = emit_and_finalize(&ListClearTemplate::new(5, 0));
        assert_ne!(code1, code2);
    }

    #[test]
    fn test_clear_high_registers() {
        let code = emit_and_finalize(&ListClearTemplate::new(15, 0));
        assert!(!code.is_empty());
    }

    #[test]
    fn test_clear_contains_xor_for_zero() {
        let mut asm = Assembler::new();
        let frame = FrameLayout::minimal(16);
        let mut ctx = make_ctx(&mut asm, &frame);
        let tmpl = ListClearTemplate::new(0, 0);
        tmpl.emit(&mut ctx);

        // XOR r64, r64 can encode as:
        //   0x33 /r (XOR r64, r/m64) with mod=11
        //   0x31 /r (XOR r/m64, r64) with mod=11
        // Either form with same reg in both fields zeros the register
        let code = ctx.asm.code();
        let has_xor = code.windows(2).any(|w| {
            (w[0] == 0x33 || w[0] == 0x31)
                && (w[1] & 0xC0) == 0xC0
                && ((w[1] >> 3) & 7) == (w[1] & 7)
        });
        let has_xor_with_rex = code.windows(3).any(|w| {
            (w[0] & 0xF0) == 0x40 && (w[1] == 0x33 || w[1] == 0x31) && (w[2] & 0xC0) == 0xC0
        });
        assert!(
            has_xor || has_xor_with_rex,
            "XOR (zero-out) instruction not found in clear emitted code"
        );
    }

    #[test]
    fn test_clear_is_compact() {
        // Clear should produce relatively compact code: type guard + xor + store
        let clear_code = emit_and_finalize(&ListClearTemplate::new(0, 0));
        assert!(
            clear_code.len() <= 100,
            "clear code ({}) should be within estimated size (100)",
            clear_code.len()
        );
    }

    // =========================================================================
    // ListContainsTemplate Tests
    // =========================================================================

    #[test]
    fn test_contains_creates() {
        let tmpl = ListContainsTemplate::new(2, 0, 1, 0);
        assert_eq!(tmpl.dst_reg, 2);
        assert_eq!(tmpl.list_reg, 0);
        assert_eq!(tmpl.value_reg, 1);
        assert_eq!(tmpl.deopt_idx, 0);
    }

    #[test]
    fn test_contains_estimated_size() {
        let tmpl = ListContainsTemplate::new(2, 0, 1, 0);
        assert_eq!(tmpl.estimated_size(), 80);
    }

    #[test]
    fn test_contains_emits_code() {
        let code = emit_and_finalize(&ListContainsTemplate::new(2, 0, 1, 0));
        assert!(code.len() > 10, "contains too short: {}", code.len());
    }

    #[test]
    fn test_contains_code_size_within_estimate() {
        let mut asm = Assembler::new();
        let frame = FrameLayout::minimal(16);
        let mut ctx = make_ctx(&mut asm, &frame);
        let tmpl = ListContainsTemplate::new(2, 0, 1, 0);
        tmpl.emit(&mut ctx);
        assert!(
            ctx.asm.offset() <= tmpl.estimated_size(),
            "contains code ({}) exceeds estimate ({})",
            ctx.asm.offset(),
            tmpl.estimated_size()
        );
    }

    #[test]
    fn test_contains_different_registers() {
        let code1 = emit_and_finalize(&ListContainsTemplate::new(2, 0, 1, 0));
        let code2 = emit_and_finalize(&ListContainsTemplate::new(5, 3, 4, 0));
        assert_ne!(code1, code2);
    }

    #[test]
    fn test_contains_high_registers() {
        let code = emit_and_finalize(&ListContainsTemplate::new(15, 13, 14, 0));
        assert!(!code.is_empty());
    }

    #[test]
    fn test_contains_ends_with_jmp_deopt() {
        let mut asm = Assembler::new();
        let frame = FrameLayout::minimal(16);
        let mut ctx = make_ctx(&mut asm, &frame);
        let tmpl = ListContainsTemplate::new(2, 0, 1, 0);
        tmpl.emit(&mut ctx);

        // The last instruction should be a JMP (E9 or EB)
        let code = ctx.asm.code();
        let len = code.len();
        // JMP near is E9 rel32 (5 bytes), JMP short is EB rel8 (2 bytes)
        let ends_with_jmp =
            (len >= 5 && code[len - 5] == 0xE9) || (len >= 2 && code[len - 2] == 0xEB);
        assert!(
            ends_with_jmp,
            "contains template should end with JMP to deopt"
        );
    }

    // =========================================================================
    // ListInsertTemplate Tests
    // =========================================================================

    #[test]
    fn test_insert_creates() {
        let tmpl = ListInsertTemplate::new(0, 1, 2, 0);
        assert_eq!(tmpl.list_reg, 0);
        assert_eq!(tmpl.index_reg, 1);
        assert_eq!(tmpl.value_reg, 2);
        assert_eq!(tmpl.deopt_idx, 0);
    }

    #[test]
    fn test_insert_estimated_size() {
        let tmpl = ListInsertTemplate::new(0, 1, 2, 0);
        assert_eq!(tmpl.estimated_size(), 120);
    }

    #[test]
    fn test_insert_emits_code() {
        let code = emit_and_finalize(&ListInsertTemplate::new(0, 1, 2, 0));
        assert!(code.len() > 20, "insert too short: {}", code.len());
    }

    #[test]
    fn test_insert_code_size_within_estimate() {
        let mut asm = Assembler::new();
        let frame = FrameLayout::minimal(16);
        let mut ctx = make_ctx(&mut asm, &frame);
        let tmpl = ListInsertTemplate::new(0, 1, 2, 0);
        tmpl.emit(&mut ctx);
        assert!(
            ctx.asm.offset() <= tmpl.estimated_size(),
            "insert code ({}) exceeds estimate ({})",
            ctx.asm.offset(),
            tmpl.estimated_size()
        );
    }

    #[test]
    fn test_insert_different_registers() {
        let code1 = emit_and_finalize(&ListInsertTemplate::new(0, 1, 2, 0));
        let code2 = emit_and_finalize(&ListInsertTemplate::new(3, 4, 5, 0));
        assert_ne!(code1, code2);
    }

    #[test]
    fn test_insert_high_registers() {
        let code = emit_and_finalize(&ListInsertTemplate::new(13, 14, 15, 0));
        assert!(!code.is_empty());
    }

    #[test]
    fn test_insert_ends_with_jmp_deopt() {
        let mut asm = Assembler::new();
        let frame = FrameLayout::minimal(16);
        let mut ctx = make_ctx(&mut asm, &frame);
        let tmpl = ListInsertTemplate::new(0, 1, 2, 0);
        tmpl.emit(&mut ctx);

        let code = ctx.asm.code();
        let len = code.len();
        let ends_with_jmp =
            (len >= 5 && code[len - 5] == 0xE9) || (len >= 2 && code[len - 2] == 0xEB);
        assert!(
            ends_with_jmp,
            "insert template should end with JMP to deopt"
        );
    }

    #[test]
    fn test_insert_larger_than_contains() {
        let insert_code = emit_and_finalize(&ListInsertTemplate::new(0, 1, 2, 0));
        let contains_code = emit_and_finalize(&ListContainsTemplate::new(2, 0, 1, 0));
        assert!(
            insert_code.len() > contains_code.len(),
            "insert ({}) should be larger than contains ({}) due to 2 type checks",
            insert_code.len(),
            contains_code.len()
        );
    }

    // =========================================================================
    // ListRemoveTemplate Tests
    // =========================================================================

    #[test]
    fn test_remove_creates() {
        let tmpl = ListRemoveTemplate::new(2, 0, 1, 0);
        assert_eq!(tmpl.dst_reg, 2);
        assert_eq!(tmpl.list_reg, 0);
        assert_eq!(tmpl.index_reg, 1);
        assert_eq!(tmpl.deopt_idx, 0);
    }

    #[test]
    fn test_remove_estimated_size() {
        let tmpl = ListRemoveTemplate::new(2, 0, 1, 0);
        assert_eq!(tmpl.estimated_size(), 120);
    }

    #[test]
    fn test_remove_emits_code() {
        let code = emit_and_finalize(&ListRemoveTemplate::new(2, 0, 1, 0));
        assert!(code.len() > 20, "remove too short: {}", code.len());
    }

    #[test]
    fn test_remove_code_size_within_estimate() {
        let mut asm = Assembler::new();
        let frame = FrameLayout::minimal(16);
        let mut ctx = make_ctx(&mut asm, &frame);
        let tmpl = ListRemoveTemplate::new(2, 0, 1, 0);
        tmpl.emit(&mut ctx);
        assert!(
            ctx.asm.offset() <= tmpl.estimated_size(),
            "remove code ({}) exceeds estimate ({})",
            ctx.asm.offset(),
            tmpl.estimated_size()
        );
    }

    #[test]
    fn test_remove_different_registers() {
        let code1 = emit_and_finalize(&ListRemoveTemplate::new(2, 0, 1, 0));
        let code2 = emit_and_finalize(&ListRemoveTemplate::new(5, 3, 4, 0));
        assert_ne!(code1, code2);
    }

    #[test]
    fn test_remove_high_registers() {
        let code = emit_and_finalize(&ListRemoveTemplate::new(15, 13, 14, 0));
        assert!(!code.is_empty());
    }

    #[test]
    fn test_remove_ends_with_jmp_deopt() {
        let mut asm = Assembler::new();
        let frame = FrameLayout::minimal(16);
        let mut ctx = make_ctx(&mut asm, &frame);
        let tmpl = ListRemoveTemplate::new(2, 0, 1, 0);
        tmpl.emit(&mut ctx);

        let code = ctx.asm.code();
        let len = code.len();
        let ends_with_jmp =
            (len >= 5 && code[len - 5] == 0xE9) || (len >= 2 && code[len - 2] == 0xEB);
        assert!(
            ends_with_jmp,
            "remove template should end with JMP to deopt"
        );
    }

    #[test]
    fn test_remove_same_size_as_insert() {
        let insert_code = emit_and_finalize(&ListInsertTemplate::new(0, 1, 2, 0));
        let remove_code = emit_and_finalize(&ListRemoveTemplate::new(2, 0, 1, 0));
        // Both do list type check + int type check + deopt, so they should be
        // approximately the same size (±a few bytes for different register encoding)
        let diff = (insert_code.len() as i64 - remove_code.len() as i64).unsigned_abs();
        assert!(
            diff <= 20,
            "insert ({}) and remove ({}) should be within 20 bytes of each other",
            insert_code.len(),
            remove_code.len()
        );
    }

    // =========================================================================
    // Cross-Template Tests
    // =========================================================================

    #[test]
    fn test_all_templates_compile() {
        // Verify all 6 templates compile without panic
        emit_and_finalize(&ListPopLastTemplate::new(1, 0, 0));
        emit_and_finalize(&ListLenTemplate::new(1, 0, 0));
        emit_and_finalize(&ListClearTemplate::new(0, 0));
        emit_and_finalize(&ListContainsTemplate::new(2, 0, 1, 0));
        emit_and_finalize(&ListInsertTemplate::new(0, 1, 2, 0));
        emit_and_finalize(&ListRemoveTemplate::new(2, 0, 1, 0));
    }

    #[test]
    fn test_sequential_emission() {
        let mut asm = Assembler::new();
        let frame = FrameLayout::minimal(16);
        {
            let mut ctx = TemplateContext::new(&mut asm, &frame);
            let _deopt = ctx.create_deopt_label();

            let pop = ListPopLastTemplate::new(1, 0, 0);
            let len = ListLenTemplate::new(1, 0, 0);
            let clear = ListClearTemplate::new(0, 0);

            let pos0 = ctx.asm.offset();
            pop.emit(&mut ctx);
            let pos1 = ctx.asm.offset();
            len.emit(&mut ctx);
            let pos2 = ctx.asm.offset();
            clear.emit(&mut ctx);
            let pos3 = ctx.asm.offset();

            // Each template produces non-zero code
            assert!(pos1 > pos0, "pop should produce code");
            assert!(pos2 > pos1, "len should produce code");
            assert!(pos3 > pos2, "clear should produce code");

            // Bind deopt labels
            for label in &ctx.deopt_labels {
                ctx.asm.bind_label(*label);
            }
        }
        let code = asm.finalize().expect("finalization should succeed");
        assert!(!code.is_empty());
    }

    #[test]
    fn test_guard_only_templates_are_compact() {
        // Guard-only templates (contains, insert, remove) should be compact
        // since they just type-guard and immediately deopt.
        let contains = emit_and_finalize(&ListContainsTemplate::new(2, 0, 1, 0)).len();
        let insert = emit_and_finalize(&ListInsertTemplate::new(0, 1, 2, 0)).len();
        let remove = emit_and_finalize(&ListRemoveTemplate::new(2, 0, 1, 0)).len();

        // All guard-only templates should produce code
        assert!(
            contains > 30,
            "contains ({}) should be non-trivial",
            contains
        );
        assert!(insert > 30, "insert ({}) should be non-trivial", insert);
        assert!(remove > 30, "remove ({}) should be non-trivial", remove);

        // Insert and remove do 2 type checks; they should be larger than contains (1 check)
        assert!(
            insert > contains,
            "insert ({}) should be > contains ({}) due to extra int type check",
            insert,
            contains
        );
        assert!(
            remove > contains,
            "remove ({}) should be > contains ({}) due to extra int type check",
            remove,
            contains
        );
    }

    #[test]
    fn test_all_estimates_are_conservative() {
        let templates: Vec<Box<dyn OpcodeTemplate>> = vec![
            Box::new(ListPopLastTemplate::new(1, 0, 0)),
            Box::new(ListLenTemplate::new(1, 0, 0)),
            Box::new(ListClearTemplate::new(0, 0)),
            Box::new(ListContainsTemplate::new(2, 0, 1, 0)),
            Box::new(ListInsertTemplate::new(0, 1, 2, 0)),
            Box::new(ListRemoveTemplate::new(2, 0, 1, 0)),
        ];

        for (i, tmpl) in templates.iter().enumerate() {
            let mut asm = Assembler::new();
            let frame = FrameLayout::minimal(16);
            let mut ctx = make_ctx(&mut asm, &frame);
            tmpl.emit(&mut ctx);
            assert!(
                ctx.asm.offset() <= tmpl.estimated_size(),
                "Template {} actual size ({}) exceeds estimate ({})",
                i,
                ctx.asm.offset(),
                tmpl.estimated_size()
            );
        }
    }

    #[test]
    fn test_zero_register_indices() {
        // All templates with register index 0
        emit_and_finalize(&ListPopLastTemplate::new(0, 0, 0));
        emit_and_finalize(&ListLenTemplate::new(0, 0, 0));
        emit_and_finalize(&ListClearTemplate::new(0, 0));
        emit_and_finalize(&ListContainsTemplate::new(0, 0, 0, 0));
        emit_and_finalize(&ListInsertTemplate::new(0, 0, 0, 0));
        emit_and_finalize(&ListRemoveTemplate::new(0, 0, 0, 0));
    }

    #[test]
    fn test_adjacent_register_indices() {
        emit_and_finalize(&ListPopLastTemplate::new(1, 2, 0));
        emit_and_finalize(&ListLenTemplate::new(2, 3, 0));
        emit_and_finalize(&ListClearTemplate::new(4, 0));
        emit_and_finalize(&ListContainsTemplate::new(5, 6, 7, 0));
        emit_and_finalize(&ListInsertTemplate::new(8, 9, 10, 0));
        emit_and_finalize(&ListRemoveTemplate::new(11, 12, 13, 0));
    }

    #[test]
    fn test_multiple_deopt_labels() {
        let mut asm = Assembler::new();
        let frame = FrameLayout::minimal(16);
        {
            let mut ctx = TemplateContext::new(&mut asm, &frame);
            // Create multiple deopt labels
            let _d0 = ctx.create_deopt_label();
            let _d1 = ctx.create_deopt_label();
            let _d2 = ctx.create_deopt_label();

            ListPopLastTemplate::new(0, 1, 0).emit(&mut ctx);
            ListLenTemplate::new(2, 3, 1).emit(&mut ctx);
            ListClearTemplate::new(4, 2).emit(&mut ctx);

            for label in &ctx.deopt_labels {
                ctx.asm.bind_label(*label);
            }
        }
        let code = asm.finalize().expect("finalization should succeed");
        assert!(!code.is_empty());
    }

    // =========================================================================
    // Layout constant verification
    // =========================================================================

    #[test]
    fn test_reused_layout_constants_match() {
        // Verify that the layout constants we import are consistent
        assert_eq!(list_layout::TYPE_ID_OFFSET, 0);
        assert_eq!(list_layout::ITEMS_PTR_OFFSET, 16);
        assert_eq!(list_layout::ITEMS_LEN_OFFSET, 24);
        assert_eq!(list_layout::ITEMS_CAP_OFFSET, 32);
        assert_eq!(list_layout::VALUE_SIZE, 8);
        assert_eq!(list_layout::LIST_TYPE_ID, 6);
    }
}
