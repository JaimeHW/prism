//! Comparison templates - relational, equality, identity, membership.
//!
//! These templates generate code for comparison operations with type-specialized
//! fast paths for integers and floats.

use super::{OpcodeTemplate, TemplateContext, value_tags};
use crate::backend::x64::Condition;

// =============================================================================
// Helper: Integer Comparison
// =============================================================================

/// Emit integer comparison with type checks.
fn emit_int_compare(
    ctx: &mut TemplateContext,
    lhs_reg: u8,
    rhs_reg: u8,
    dst_reg: u8,
    condition: Condition,
    deopt_idx: usize,
) {
    let lhs_slot = ctx.frame.register_slot(lhs_reg as u16);
    let rhs_slot = ctx.frame.register_slot(rhs_reg as u16);
    let dst_slot = ctx.frame.register_slot(dst_reg as u16);

    let acc = ctx.regs.accumulator;
    let scratch1 = ctx.regs.scratch1;
    let scratch2 = ctx.regs.scratch2;

    // Load and check LHS type
    ctx.asm.mov_rm(acc, &lhs_slot);
    ctx.asm.mov_rr(scratch1, acc);
    ctx.asm.shr_ri(scratch1, 48);
    let expected_tag = value_tags::int_tag_check() as i32;
    ctx.asm.cmp_ri(scratch1, expected_tag);
    ctx.asm.jne(ctx.deopt_label(deopt_idx));

    // Load and check RHS type
    ctx.asm.mov_rm(scratch2, &rhs_slot);
    ctx.asm.mov_rr(scratch1, scratch2);
    ctx.asm.shr_ri(scratch1, 48);
    ctx.asm.cmp_ri(scratch1, expected_tag);
    ctx.asm.jne(ctx.deopt_label(deopt_idx));

    // Extract payloads (sign-extend from 48 bits)
    ctx.asm.shl_ri(acc, 16);
    ctx.asm.sar_ri(acc, 16);
    ctx.asm.shl_ri(scratch2, 16);
    ctx.asm.sar_ri(scratch2, 16);

    // Compare
    ctx.asm.cmp_rr(acc, scratch2);

    // Set result based on condition using setcc (sets lower byte)
    ctx.asm.xor_rr(scratch1, scratch1); // Zero scratch1
    ctx.asm.setcc(condition, scratch1); // Set low byte to 0 or 1

    // Convert 0/1 to False/True boxed boolean
    let true_val = value_tags::true_value() as i64;
    let false_val = value_tags::false_value() as i64;

    // If scratch1 == 0, result = FALSE, else result = TRUE
    ctx.asm.test_rr(scratch1, scratch1);
    ctx.asm.mov_ri64(acc, false_val);

    let done_label = ctx.asm.create_label();
    ctx.asm.jz(done_label); // If zero, keep false
    ctx.asm.mov_ri64(acc, true_val);
    ctx.asm.bind_label(done_label);

    // Store result
    ctx.asm.mov_mr(&dst_slot, acc);
}

// =============================================================================
// Integer Comparisons
// =============================================================================

/// Template for less than comparison.
pub struct LtTemplate {
    pub dst_reg: u8,
    pub lhs_reg: u8,
    pub rhs_reg: u8,
    pub deopt_idx: usize,
}

impl OpcodeTemplate for LtTemplate {
    fn emit(&self, ctx: &mut TemplateContext) {
        emit_int_compare(
            ctx,
            self.lhs_reg,
            self.rhs_reg,
            self.dst_reg,
            Condition::Less,
            self.deopt_idx,
        );
    }

    #[inline]
    fn estimated_size(&self) -> usize {
        100
    }
}

/// Template for less than or equal comparison.
pub struct LeTemplate {
    pub dst_reg: u8,
    pub lhs_reg: u8,
    pub rhs_reg: u8,
    pub deopt_idx: usize,
}

impl OpcodeTemplate for LeTemplate {
    fn emit(&self, ctx: &mut TemplateContext) {
        emit_int_compare(
            ctx,
            self.lhs_reg,
            self.rhs_reg,
            self.dst_reg,
            Condition::LessEqual,
            self.deopt_idx,
        );
    }

    #[inline]
    fn estimated_size(&self) -> usize {
        100
    }
}

/// Template for equality comparison.
pub struct EqTemplate {
    pub dst_reg: u8,
    pub lhs_reg: u8,
    pub rhs_reg: u8,
    pub deopt_idx: usize,
}

impl OpcodeTemplate for EqTemplate {
    fn emit(&self, ctx: &mut TemplateContext) {
        emit_int_compare(
            ctx,
            self.lhs_reg,
            self.rhs_reg,
            self.dst_reg,
            Condition::Equal,
            self.deopt_idx,
        );
    }

    #[inline]
    fn estimated_size(&self) -> usize {
        100
    }
}

/// Template for not equal comparison.
pub struct NeTemplate {
    pub dst_reg: u8,
    pub lhs_reg: u8,
    pub rhs_reg: u8,
    pub deopt_idx: usize,
}

impl OpcodeTemplate for NeTemplate {
    fn emit(&self, ctx: &mut TemplateContext) {
        emit_int_compare(
            ctx,
            self.lhs_reg,
            self.rhs_reg,
            self.dst_reg,
            Condition::NotEqual,
            self.deopt_idx,
        );
    }

    #[inline]
    fn estimated_size(&self) -> usize {
        100
    }
}

/// Template for greater than comparison.
pub struct GtTemplate {
    pub dst_reg: u8,
    pub lhs_reg: u8,
    pub rhs_reg: u8,
    pub deopt_idx: usize,
}

impl OpcodeTemplate for GtTemplate {
    fn emit(&self, ctx: &mut TemplateContext) {
        emit_int_compare(
            ctx,
            self.lhs_reg,
            self.rhs_reg,
            self.dst_reg,
            Condition::Greater,
            self.deopt_idx,
        );
    }

    #[inline]
    fn estimated_size(&self) -> usize {
        100
    }
}

/// Template for greater than or equal comparison.
pub struct GeTemplate {
    pub dst_reg: u8,
    pub lhs_reg: u8,
    pub rhs_reg: u8,
    pub deopt_idx: usize,
}

impl OpcodeTemplate for GeTemplate {
    fn emit(&self, ctx: &mut TemplateContext) {
        emit_int_compare(
            ctx,
            self.lhs_reg,
            self.rhs_reg,
            self.dst_reg,
            Condition::GreaterEqual,
            self.deopt_idx,
        );
    }

    #[inline]
    fn estimated_size(&self) -> usize {
        100
    }
}

// =============================================================================
// Identity Comparisons
// =============================================================================

/// Template for identity comparison (is).
pub struct IsTemplate {
    pub dst_reg: u8,
    pub lhs_reg: u8,
    pub rhs_reg: u8,
}

impl OpcodeTemplate for IsTemplate {
    fn emit(&self, ctx: &mut TemplateContext) {
        let lhs_slot = ctx.frame.register_slot(self.lhs_reg as u16);
        let rhs_slot = ctx.frame.register_slot(self.rhs_reg as u16);
        let dst_slot = ctx.frame.register_slot(self.dst_reg as u16);

        let acc = ctx.regs.accumulator;
        let scratch1 = ctx.regs.scratch1;

        // Load both values
        ctx.asm.mov_rm(acc, &lhs_slot);
        ctx.asm.mov_rm(scratch1, &rhs_slot);

        // Compare raw bits (identity check)
        ctx.asm.cmp_rr(acc, scratch1);

        // Set result using setcc + branch
        let true_val = value_tags::true_value() as i64;
        let false_val = value_tags::false_value() as i64;

        ctx.asm.mov_ri64(acc, false_val);
        let done_label = ctx.asm.create_label();
        ctx.asm.jne(done_label);
        ctx.asm.mov_ri64(acc, true_val);
        ctx.asm.bind_label(done_label);

        ctx.asm.mov_mr(&dst_slot, acc);
    }

    #[inline]
    fn estimated_size(&self) -> usize {
        48
    }
}

/// Template for not identity comparison (is not).
pub struct IsNotTemplate {
    pub dst_reg: u8,
    pub lhs_reg: u8,
    pub rhs_reg: u8,
}

impl OpcodeTemplate for IsNotTemplate {
    fn emit(&self, ctx: &mut TemplateContext) {
        let lhs_slot = ctx.frame.register_slot(self.lhs_reg as u16);
        let rhs_slot = ctx.frame.register_slot(self.rhs_reg as u16);
        let dst_slot = ctx.frame.register_slot(self.dst_reg as u16);

        let acc = ctx.regs.accumulator;
        let scratch1 = ctx.regs.scratch1;

        ctx.asm.mov_rm(acc, &lhs_slot);
        ctx.asm.mov_rm(scratch1, &rhs_slot);
        ctx.asm.cmp_rr(acc, scratch1);

        let true_val = value_tags::true_value() as i64;
        let false_val = value_tags::false_value() as i64;

        ctx.asm.mov_ri64(acc, false_val);
        let done_label = ctx.asm.create_label();
        ctx.asm.je(done_label);
        ctx.asm.mov_ri64(acc, true_val);
        ctx.asm.bind_label(done_label);

        ctx.asm.mov_mr(&dst_slot, acc);
    }

    #[inline]
    fn estimated_size(&self) -> usize {
        48
    }
}

// =============================================================================
// Float Comparisons
// =============================================================================

/// Emit float comparison using SSE2 ucomisd instruction.
///
/// The ucomisd instruction sets flags as follows:
/// - ZF=1, PF=0, CF=0: EQUAL
/// - ZF=0, PF=0, CF=1: LESS THAN
/// - ZF=0, PF=0, CF=0: GREATER THAN
/// - ZF=1, PF=1, CF=1: UNORDERED (one or both are NaN)
///
/// For NaN handling: Python comparisons with NaN always return False
/// (except != which returns True).
fn emit_float_compare(
    ctx: &mut TemplateContext,
    lhs_reg: u8,
    rhs_reg: u8,
    dst_reg: u8,
    condition: FloatCondition,
) {
    let lhs_slot = ctx.frame.register_slot(lhs_reg as u16);
    let rhs_slot = ctx.frame.register_slot(rhs_reg as u16);
    let dst_slot = ctx.frame.register_slot(dst_reg as u16);

    let xmm0 = ctx.regs.xmm0;
    let xmm1 = ctx.regs.xmm1;
    let acc = ctx.regs.accumulator;
    let scratch1 = ctx.regs.scratch1;

    // Load floats into XMM registers
    ctx.asm.movsd_rm(xmm0, &lhs_slot);
    ctx.asm.movsd_rm(xmm1, &rhs_slot);

    let true_val = value_tags::true_value() as i64;
    let false_val = value_tags::false_value() as i64;

    match condition {
        FloatCondition::Lt => {
            // a < b: Compare b to a, then check if b > a (CF=0 && ZF=0)
            // Using ucomisd xmm1, xmm0 (compare b, a)
            // If b > a, then a < b
            ctx.asm.ucomisd(xmm1, xmm0);
            // seta: Set if CF=0 AND ZF=0 (above)
            ctx.asm.xor_rr(scratch1, scratch1);
            ctx.asm.setcc(Condition::Above, scratch1);
        }
        FloatCondition::Le => {
            // a <= b: Compare b to a, check if b >= a
            ctx.asm.ucomisd(xmm1, xmm0);
            // setae: Set if CF=0 (above or equal)
            ctx.asm.xor_rr(scratch1, scratch1);
            ctx.asm.setcc(Condition::AboveEqual, scratch1);
        }
        FloatCondition::Gt => {
            // a > b: Compare a to b, check if a > b
            ctx.asm.ucomisd(xmm0, xmm1);
            ctx.asm.xor_rr(scratch1, scratch1);
            ctx.asm.setcc(Condition::Above, scratch1);
        }
        FloatCondition::Ge => {
            // a >= b: Compare a to b, check if a >= b
            ctx.asm.ucomisd(xmm0, xmm1);
            ctx.asm.xor_rr(scratch1, scratch1);
            ctx.asm.setcc(Condition::AboveEqual, scratch1);
        }
        FloatCondition::Eq => {
            // a == b: Must be ordered AND equal
            // ucomisd sets ZF=1 and PF=0 for equal
            ctx.asm.ucomisd(xmm0, xmm1);

            // Check for unordered first (NaN case) - if PF=1, result is false
            ctx.asm.xor_rr(scratch1, scratch1);
            let nan_label = ctx.asm.create_label();
            ctx.asm.jcc(Condition::Parity, nan_label); // Jump if parity (unordered/NaN)

            // Not unordered, check if equal
            ctx.asm.setcc(Condition::Equal, scratch1);
            ctx.asm.bind_label(nan_label);
        }
        FloatCondition::Ne => {
            // a != b: Unordered OR not equal
            // Returns true if PF=1 OR ZF=0
            ctx.asm.ucomisd(xmm0, xmm1);
            ctx.asm.xor_rr(scratch1, scratch1);

            // Check if unordered (PF=1) - if so, result is true
            let true_label = ctx.asm.create_label();
            ctx.asm.jcc(Condition::Parity, true_label); // Jump if parity (unordered/NaN)

            // Not unordered, check if not equal
            ctx.asm.setcc(Condition::NotEqual, scratch1);
            let done = ctx.asm.create_label();
            ctx.asm.jmp(done);

            ctx.asm.bind_label(true_label);
            ctx.asm.mov_ri32(scratch1, 1);
            ctx.asm.bind_label(done);
        }
    }

    // Convert 0/1 to False/True boxed boolean
    ctx.asm.test_rr(scratch1, scratch1);
    ctx.asm.mov_ri64(acc, false_val);
    let done_label = ctx.asm.create_label();
    ctx.asm.jz(done_label);
    ctx.asm.mov_ri64(acc, true_val);
    ctx.asm.bind_label(done_label);

    // Store result
    ctx.asm.mov_mr(&dst_slot, acc);
}

/// Float comparison condition.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum FloatCondition {
    Lt,
    Le,
    Gt,
    Ge,
    Eq,
    Ne,
}

/// Template for float less-than comparison.
pub struct FloatLtTemplate {
    pub dst_reg: u8,
    pub lhs_reg: u8,
    pub rhs_reg: u8,
}

impl OpcodeTemplate for FloatLtTemplate {
    fn emit(&self, ctx: &mut TemplateContext) {
        emit_float_compare(
            ctx,
            self.lhs_reg,
            self.rhs_reg,
            self.dst_reg,
            FloatCondition::Lt,
        );
    }

    #[inline]
    fn estimated_size(&self) -> usize {
        64
    }
}

/// Template for float less-than-or-equal comparison.
pub struct FloatLeTemplate {
    pub dst_reg: u8,
    pub lhs_reg: u8,
    pub rhs_reg: u8,
}

impl OpcodeTemplate for FloatLeTemplate {
    fn emit(&self, ctx: &mut TemplateContext) {
        emit_float_compare(
            ctx,
            self.lhs_reg,
            self.rhs_reg,
            self.dst_reg,
            FloatCondition::Le,
        );
    }

    #[inline]
    fn estimated_size(&self) -> usize {
        64
    }
}

/// Template for float greater-than comparison.
pub struct FloatGtTemplate {
    pub dst_reg: u8,
    pub lhs_reg: u8,
    pub rhs_reg: u8,
}

impl OpcodeTemplate for FloatGtTemplate {
    fn emit(&self, ctx: &mut TemplateContext) {
        emit_float_compare(
            ctx,
            self.lhs_reg,
            self.rhs_reg,
            self.dst_reg,
            FloatCondition::Gt,
        );
    }

    #[inline]
    fn estimated_size(&self) -> usize {
        64
    }
}

/// Template for float greater-than-or-equal comparison.
pub struct FloatGeTemplate {
    pub dst_reg: u8,
    pub lhs_reg: u8,
    pub rhs_reg: u8,
}

impl OpcodeTemplate for FloatGeTemplate {
    fn emit(&self, ctx: &mut TemplateContext) {
        emit_float_compare(
            ctx,
            self.lhs_reg,
            self.rhs_reg,
            self.dst_reg,
            FloatCondition::Ge,
        );
    }

    #[inline]
    fn estimated_size(&self) -> usize {
        64
    }
}

/// Template for float equality comparison.
pub struct FloatEqTemplate {
    pub dst_reg: u8,
    pub lhs_reg: u8,
    pub rhs_reg: u8,
}

impl OpcodeTemplate for FloatEqTemplate {
    fn emit(&self, ctx: &mut TemplateContext) {
        emit_float_compare(
            ctx,
            self.lhs_reg,
            self.rhs_reg,
            self.dst_reg,
            FloatCondition::Eq,
        );
    }

    #[inline]
    fn estimated_size(&self) -> usize {
        80 // Slightly larger due to NaN handling
    }
}

/// Template for float not-equal comparison.
pub struct FloatNeTemplate {
    pub dst_reg: u8,
    pub lhs_reg: u8,
    pub rhs_reg: u8,
}

impl OpcodeTemplate for FloatNeTemplate {
    fn emit(&self, ctx: &mut TemplateContext) {
        emit_float_compare(
            ctx,
            self.lhs_reg,
            self.rhs_reg,
            self.dst_reg,
            FloatCondition::Ne,
        );
    }

    #[inline]
    fn estimated_size(&self) -> usize {
        80 // Slightly larger due to NaN handling
    }
}

// =============================================================================
// Membership (in/not in) - Deopt to interpreter
// =============================================================================

/// Template for membership test (in).
pub struct InTemplate {
    pub dst_reg: u8,
    pub item_reg: u8,
    pub container_reg: u8,
    pub deopt_idx: usize,
}

impl OpcodeTemplate for InTemplate {
    fn emit(&self, ctx: &mut TemplateContext) {
        // Membership tests require container iteration - deopt
        ctx.asm.jmp(ctx.deopt_label(self.deopt_idx));
    }

    #[inline]
    fn estimated_size(&self) -> usize {
        8
    }
}

/// Template for not-in test.
pub struct NotInTemplate {
    pub dst_reg: u8,
    pub item_reg: u8,
    pub container_reg: u8,
    pub deopt_idx: usize,
}

impl OpcodeTemplate for NotInTemplate {
    fn emit(&self, ctx: &mut TemplateContext) {
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
    fn test_lt_template() {
        let mut asm = Assembler::new();
        let frame = FrameLayout::minimal(4);
        let mut ctx = make_ctx(&mut asm, &frame);

        let template = LtTemplate {
            dst_reg: 2,
            lhs_reg: 0,
            rhs_reg: 1,
            deopt_idx: 0,
        };
        template.emit(&mut ctx);

        assert!(ctx.asm.offset() > 50);
    }

    #[test]
    fn test_eq_template() {
        let mut asm = Assembler::new();
        let frame = FrameLayout::minimal(4);
        let mut ctx = make_ctx(&mut asm, &frame);

        let template = EqTemplate {
            dst_reg: 2,
            lhs_reg: 0,
            rhs_reg: 1,
            deopt_idx: 0,
        };
        template.emit(&mut ctx);

        assert!(ctx.asm.offset() > 50);
    }

    #[test]
    fn test_is_template() {
        let mut asm = Assembler::new();
        let frame = FrameLayout::minimal(4);
        let mut ctx = TemplateContext::new(&mut asm, &frame);

        let template = IsTemplate {
            dst_reg: 2,
            lhs_reg: 0,
            rhs_reg: 1,
        };
        template.emit(&mut ctx);

        assert!(ctx.asm.offset() > 20);
    }

    #[test]
    fn test_is_not_template() {
        let mut asm = Assembler::new();
        let frame = FrameLayout::minimal(4);
        let mut ctx = TemplateContext::new(&mut asm, &frame);

        let template = IsNotTemplate {
            dst_reg: 2,
            lhs_reg: 0,
            rhs_reg: 1,
        };
        template.emit(&mut ctx);

        assert!(ctx.asm.offset() > 20);
    }

    #[test]
    fn test_float_lt_template() {
        let mut asm = Assembler::new();
        let frame = FrameLayout::minimal(4);
        let mut ctx = TemplateContext::new(&mut asm, &frame);

        let template = FloatLtTemplate {
            dst_reg: 2,
            lhs_reg: 0,
            rhs_reg: 1,
        };
        template.emit(&mut ctx);

        assert!(ctx.asm.offset() > 20);
    }

    #[test]
    fn test_float_le_template() {
        let mut asm = Assembler::new();
        let frame = FrameLayout::minimal(4);
        let mut ctx = TemplateContext::new(&mut asm, &frame);

        let template = FloatLeTemplate {
            dst_reg: 2,
            lhs_reg: 0,
            rhs_reg: 1,
        };
        template.emit(&mut ctx);

        assert!(ctx.asm.offset() > 20);
    }

    #[test]
    fn test_float_gt_template() {
        let mut asm = Assembler::new();
        let frame = FrameLayout::minimal(4);
        let mut ctx = TemplateContext::new(&mut asm, &frame);

        let template = FloatGtTemplate {
            dst_reg: 2,
            lhs_reg: 0,
            rhs_reg: 1,
        };
        template.emit(&mut ctx);

        assert!(ctx.asm.offset() > 20);
    }

    #[test]
    fn test_float_ge_template() {
        let mut asm = Assembler::new();
        let frame = FrameLayout::minimal(4);
        let mut ctx = TemplateContext::new(&mut asm, &frame);

        let template = FloatGeTemplate {
            dst_reg: 2,
            lhs_reg: 0,
            rhs_reg: 1,
        };
        template.emit(&mut ctx);

        assert!(ctx.asm.offset() > 20);
    }

    #[test]
    fn test_float_eq_template() {
        let mut asm = Assembler::new();
        let frame = FrameLayout::minimal(4);
        let mut ctx = TemplateContext::new(&mut asm, &frame);

        let template = FloatEqTemplate {
            dst_reg: 2,
            lhs_reg: 0,
            rhs_reg: 1,
        };
        template.emit(&mut ctx);

        // Equality has extra NaN handling - should be larger
        assert!(ctx.asm.offset() > 30);
    }

    #[test]
    fn test_float_ne_template() {
        let mut asm = Assembler::new();
        let frame = FrameLayout::minimal(4);
        let mut ctx = TemplateContext::new(&mut asm, &frame);

        let template = FloatNeTemplate {
            dst_reg: 2,
            lhs_reg: 0,
            rhs_reg: 1,
        };
        template.emit(&mut ctx);

        // Not-equal has extra NaN handling - should be larger
        assert!(ctx.asm.offset() > 30);
    }
}
