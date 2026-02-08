//! Object operation templates - attribute/item access, iterators, len.
//!
//! These templates handle Python object operations. GetAttr and SetAttr use
//! inline caching for fast property access on shaped objects.

use super::ic_helpers::{emit_get_attr_ic_fast_path, emit_set_attr_ic_fast_path};
use super::{OpcodeTemplate, TemplateContext};
use crate::backend::x64::Gpr;

// =============================================================================
// Attribute Access
// =============================================================================

/// Template for getting an attribute with inline caching.
///
/// The IC fast path:
/// 1. Load object's shape_id
/// 2. Compare against cached shape_id
/// 3. On match: load value from cached slot offset (O(1))
/// 4. On miss: jump to deopt stub for slow path
pub struct GetAttrTemplate {
    /// Destination register for the loaded value
    pub dst_reg: u8,
    /// Register containing the object pointer
    pub obj_reg: u8,
    /// Index into the name table for the attribute name
    pub name_idx: u16,
    /// Index into the deoptimization stub table
    pub deopt_idx: usize,
    /// Offset into the IC data array for this site (None = IC not enabled)
    pub ic_site_offset: Option<i32>,
}

impl OpcodeTemplate for GetAttrTemplate {
    fn emit(&self, ctx: &mut TemplateContext) {
        // If IC is enabled and we have scratch registers, use the fast path
        if let Some(ic_offset) = self.ic_site_offset {
            // Get the registers we need
            let obj_gpr = frame_slot_to_gpr(self.obj_reg, &ctx.regs);
            let dst_gpr = frame_slot_to_gpr(self.dst_reg, &ctx.regs);

            // Use caller-saved registers as scratch
            // R10 and R11 are typically scratch on both Windows and SysV
            let scratch1 = Gpr::R10;
            let scratch2 = Gpr::R11;

            // IC base is passed to JIT'd code in a dedicated register
            // R15 is typically used for IC data base
            let ic_base = Gpr::R15;

            // Create miss label that jumps to deopt
            let miss_label = ctx.asm.create_label();

            // Emit the IC fast path
            emit_get_attr_ic_fast_path(
                ctx.asm, dst_gpr, obj_gpr, ic_base, ic_offset, miss_label, scratch1, scratch2,
            );

            // Jump over the miss handler
            let done_label = ctx.asm.create_label();
            ctx.asm.jmp(done_label);

            // Miss handler: deoptimize
            ctx.asm.bind_label(miss_label);
            ctx.asm.jmp(ctx.deopt_label(self.deopt_idx));

            // Done label
            ctx.asm.bind_label(done_label);
        } else {
            // No IC site allocated, just deoptimize
            ctx.asm.jmp(ctx.deopt_label(self.deopt_idx));
        }
    }

    #[inline]
    fn estimated_size(&self) -> usize {
        if self.ic_site_offset.is_some() {
            // Shape guard + slot load + miss handler
            50
        } else {
            8
        }
    }
}

/// Template for setting an attribute with inline caching.
///
/// The IC fast path handles updates to existing properties:
/// 1. Load object's shape_id
/// 2. Compare against cached shape_id
/// 3. On match: store value to cached slot offset (O(1))
/// 4. On miss: deopt for slow path (may need shape transition)
pub struct SetAttrTemplate {
    /// Register containing the object pointer
    pub obj_reg: u8,
    /// Index into the name table for the attribute name
    pub name_idx: u16,
    /// Register containing the value to store
    pub value_reg: u8,
    /// Index into the deoptimization stub table
    pub deopt_idx: usize,
    /// Offset into the IC data array for this site (None = IC not enabled)
    pub ic_site_offset: Option<i32>,
}

impl OpcodeTemplate for SetAttrTemplate {
    fn emit(&self, ctx: &mut TemplateContext) {
        // If IC is enabled, use the fast path
        if let Some(ic_offset) = self.ic_site_offset {
            let obj_gpr = frame_slot_to_gpr(self.obj_reg, &ctx.regs);
            let value_gpr = frame_slot_to_gpr(self.value_reg, &ctx.regs);

            let scratch1 = Gpr::R10;
            let scratch2 = Gpr::R11;
            let ic_base = Gpr::R15;

            let miss_label = ctx.asm.create_label();

            // Emit the IC fast path
            emit_set_attr_ic_fast_path(
                ctx.asm, obj_gpr, value_gpr, ic_base, ic_offset, miss_label, scratch1, scratch2,
            );

            // Jump over the miss handler
            let done_label = ctx.asm.create_label();
            ctx.asm.jmp(done_label);

            // Miss handler: deoptimize
            ctx.asm.bind_label(miss_label);
            ctx.asm.jmp(ctx.deopt_label(self.deopt_idx));

            // Done label
            ctx.asm.bind_label(done_label);
        } else {
            ctx.asm.jmp(ctx.deopt_label(self.deopt_idx));
        }
    }

    #[inline]
    fn estimated_size(&self) -> usize {
        if self.ic_site_offset.is_some() { 50 } else { 8 }
    }
}

/// Template for deleting an attribute.
pub struct DelAttrTemplate {
    pub obj_reg: u8,
    pub name_idx: u16,
    pub deopt_idx: usize,
}

impl OpcodeTemplate for DelAttrTemplate {
    fn emit(&self, ctx: &mut TemplateContext) {
        // Delete always requires runtime - shape mutation
        ctx.asm.jmp(ctx.deopt_label(self.deopt_idx));
    }

    #[inline]
    fn estimated_size(&self) -> usize {
        8
    }
}

// =============================================================================
// Helper Functions
// =============================================================================

/// Map a frame slot index to a GPR.
///
/// This uses the JIT's register assignment to determine which GPR
/// corresponds to a given virtual register slot.
#[inline]
fn frame_slot_to_gpr(slot: u8, _regs: &crate::tier1::frame::RegisterAssignment) -> Gpr {
    // For Tier 1, we use a simple mapping based on slot index
    // The first few slots map to argument/callee-saved registers
    match slot {
        0 => Gpr::Rdi,  // First argument
        1 => Gpr::Rsi,  // Second argument
        2 => Gpr::Rdx,  // Third argument
        3 => Gpr::Rcx,  // Fourth argument
        4 => Gpr::R8,   // Fifth argument
        5 => Gpr::R9,   // Sixth argument
        6 => Gpr::Rax,  // Return value / scratch
        7 => Gpr::Rbx,  // Callee-saved
        8 => Gpr::R12,  // Callee-saved
        9 => Gpr::R13,  // Callee-saved
        10 => Gpr::R14, // Callee-saved
        _ => Gpr::Rax,  // Default for higher slots
    }
}

// =============================================================================
// Item Access
// =============================================================================

/// Template for getting an item (indexing).
pub struct GetItemTemplate {
    pub dst_reg: u8,
    pub obj_reg: u8,
    pub key_reg: u8,
    pub deopt_idx: usize,
}

impl OpcodeTemplate for GetItemTemplate {
    fn emit(&self, ctx: &mut TemplateContext) {
        // Item access requires type dispatch - deopt
        // Full impl would have fast paths for list/dict with int/str keys
        ctx.asm.jmp(ctx.deopt_label(self.deopt_idx));
    }

    #[inline]
    fn estimated_size(&self) -> usize {
        8
    }
}

/// Template for setting an item.
pub struct SetItemTemplate {
    pub obj_reg: u8,
    pub key_reg: u8,
    pub value_reg: u8,
    pub deopt_idx: usize,
}

impl OpcodeTemplate for SetItemTemplate {
    fn emit(&self, ctx: &mut TemplateContext) {
        ctx.asm.jmp(ctx.deopt_label(self.deopt_idx));
    }

    #[inline]
    fn estimated_size(&self) -> usize {
        8
    }
}

/// Template for deleting an item.
pub struct DelItemTemplate {
    pub obj_reg: u8,
    pub key_reg: u8,
    pub deopt_idx: usize,
}

impl OpcodeTemplate for DelItemTemplate {
    fn emit(&self, ctx: &mut TemplateContext) {
        ctx.asm.jmp(ctx.deopt_label(self.deopt_idx));
    }

    #[inline]
    fn estimated_size(&self) -> usize {
        8
    }
}

// =============================================================================
// Iterator Operations
// =============================================================================

/// Template for getting an iterator.
pub struct GetIterTemplate {
    pub dst_reg: u8,
    pub src_reg: u8,
    pub deopt_idx: usize,
}

impl OpcodeTemplate for GetIterTemplate {
    fn emit(&self, ctx: &mut TemplateContext) {
        // Iterator protocol requires runtime - deopt
        ctx.asm.jmp(ctx.deopt_label(self.deopt_idx));
    }

    #[inline]
    fn estimated_size(&self) -> usize {
        8
    }
}

/// Template for for-loop iteration.
pub struct ForIterTemplate {
    pub dst_reg: u8,
    pub iter_reg: u8,
    pub end_label: crate::backend::x64::Label,
    pub deopt_idx: usize,
}

impl OpcodeTemplate for ForIterTemplate {
    fn emit(&self, ctx: &mut TemplateContext) {
        // For iteration is complex - deopt
        ctx.asm.jmp(ctx.deopt_label(self.deopt_idx));
    }

    #[inline]
    fn estimated_size(&self) -> usize {
        8
    }
}

// =============================================================================
// Length and Callable Check
// =============================================================================

/// Template for getting length.
pub struct LenTemplate {
    pub dst_reg: u8,
    pub src_reg: u8,
    pub deopt_idx: usize,
}

impl OpcodeTemplate for LenTemplate {
    fn emit(&self, ctx: &mut TemplateContext) {
        // len() requires type dispatch - deopt
        ctx.asm.jmp(ctx.deopt_label(self.deopt_idx));
    }

    #[inline]
    fn estimated_size(&self) -> usize {
        8
    }
}

/// Template for checking if callable.
pub struct IsCallableTemplate {
    pub dst_reg: u8,
    pub src_reg: u8,
    pub deopt_idx: usize,
}

impl OpcodeTemplate for IsCallableTemplate {
    fn emit(&self, ctx: &mut TemplateContext) {
        // Callable check requires type inspection - deopt
        ctx.asm.jmp(ctx.deopt_label(self.deopt_idx));
    }

    #[inline]
    fn estimated_size(&self) -> usize {
        8
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

    fn make_ctx<'a>(asm: &'a mut Assembler, frame: &'a FrameLayout) -> TemplateContext<'a> {
        let mut ctx = TemplateContext::new(asm, frame);
        ctx.create_deopt_label();
        ctx
    }

    #[test]
    fn test_get_attr_template() {
        let mut asm = Assembler::new();
        let frame = FrameLayout::minimal(4);
        let mut ctx = make_ctx(&mut asm, &frame);

        let template = GetAttrTemplate {
            dst_reg: 1,
            obj_reg: 0,
            name_idx: 0,
            deopt_idx: 0,
            ic_site_offset: None, // No IC for basic test
        };
        template.emit(&mut ctx);

        assert!(ctx.asm.offset() > 0);
    }

    #[test]
    fn test_get_item_template() {
        let mut asm = Assembler::new();
        let frame = FrameLayout::minimal(4);
        let mut ctx = make_ctx(&mut asm, &frame);

        let template = GetItemTemplate {
            dst_reg: 2,
            obj_reg: 0,
            key_reg: 1,
            deopt_idx: 0,
        };
        template.emit(&mut ctx);

        assert!(ctx.asm.offset() > 0);
    }

    #[test]
    fn test_get_iter_template() {
        let mut asm = Assembler::new();
        let frame = FrameLayout::minimal(4);
        let mut ctx = make_ctx(&mut asm, &frame);

        let template = GetIterTemplate {
            dst_reg: 1,
            src_reg: 0,
            deopt_idx: 0,
        };
        template.emit(&mut ctx);

        assert!(ctx.asm.offset() > 0);
    }
}
