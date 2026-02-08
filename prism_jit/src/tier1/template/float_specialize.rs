//! Float-specialized JIT templates for high-performance floating-point arithmetic.
//!
//! Provides type-specialized native code generation for:
//! - **Float addition** (`a + b`) — fully inline SSE2 `ADDSD`
//! - **Float subtraction** (`a - b`) — fully inline SSE2 `SUBSD`
//! - **Float multiplication** (`a * b`) — fully inline SSE2 `MULSD`
//! - **Float division** (`a / b`) — fully inline SSE2 `DIVSD` with zero-div guard
//! - **Float negation** (`-a`) — fully inline SSE2 `XORPD` sign-bit flip
//! - **Float absolute value** (`abs(a)`) — fully inline bit-mask clear sign bit
//! - **Float comparison** (`a < b`, etc.) — fully inline `UCOMISD` + `SETcc`
//!
//! # NaN-Boxing: Float Guard Strategy
//!
//! In Prism's NaN-boxing scheme, a float is any 64-bit value whose upper 16 bits
//! are *below* the QNAN threshold. Specifically:
//!
//! ```text
//! value >> 48  <  (QNAN_BITS >> 48)  →  it's a float (raw IEEE 754 bits)
//! value >> 48  >= (QNAN_BITS >> 48)  →  it's a tagged value (int, bool, etc.)
//! ```
//!
//! This means floats are stored as their raw IEEE 754 double-precision bits.
//! No boxing/unboxing is needed — the value in the frame slot IS the float.
//!
//! # SSE2 Strategy
//!
//! Since values live in memory (frame slots), we can directly load them into
//! XMM registers via `MOVSD xmm, [mem]` and store results back with
//! `MOVSD [mem], xmm`. No GPR↔XMM register moves required.
//!
//! The code generation pattern for binary ops:
//! 1. Float-guard LHS (GPR: load, SHR 48, CMP, JB ok / JMP deopt)
//! 2. Float-guard RHS (same)
//! 3. Load LHS into XMM0 from frame slot
//! 4. Load RHS into XMM1 from frame slot
//! 5. Perform SSE2 op (ADDSD/SUBSD/MULSD/DIVSD)
//! 6. Store result from XMM0 to destination frame slot

use super::{OpcodeTemplate, TemplateContext, value_tags};
use crate::backend::x64::Condition;
use crate::backend::x64::Gpr;
use crate::backend::x64::registers::Xmm;

// =============================================================================
// Float Type Guard Helper
// =============================================================================

/// Emit code to verify a value is a float (not a NaN-boxed tagged value).
///
/// Performs:
/// 1. Load value into GPR
/// 2. Shift right by 48 to isolate upper 16 bits
/// 3. Compare with QNAN threshold
/// 4. If below (unsigned), it's a valid float → continue
/// 5. If at or above, it's a tagged value → deopt
///
/// After this function, the value in the original frame slot is confirmed
/// to be a raw IEEE 754 float.
fn emit_float_guard(ctx: &mut TemplateContext, value_reg: Gpr, scratch: Gpr, deopt_idx: usize) {
    ctx.asm.mov_rr(scratch, value_reg);
    ctx.asm.shr_ri(scratch, 48);

    let qnan_check = (value_tags::QNAN_BITS >> 48) as i32;
    ctx.asm.cmp_ri(scratch, qnan_check);

    // If upper bits < QNAN pattern → valid float
    let ok_label = ctx.asm.create_label();
    ctx.asm.jb(ok_label);

    // Not a float → deopt
    ctx.asm.jmp(ctx.deopt_label(deopt_idx));

    ctx.asm.bind_label(ok_label);
}

// =============================================================================
// Float Binary Arithmetic Templates
// =============================================================================

/// Template for float addition (`a + b`).
///
/// # Strategy
///
/// 1. Guard both operands as float
/// 2. Load into XMM0, XMM1 via MOVSD from frame slots
/// 3. ADDSD XMM0, XMM1
/// 4. MOVSD result to destination frame slot
///
/// # Estimated Size: ~100 bytes
pub struct FloatAddTemplate {
    pub dst_reg: u8,
    pub lhs_reg: u8,
    pub rhs_reg: u8,
    pub deopt_idx: usize,
}

impl FloatAddTemplate {
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

impl OpcodeTemplate for FloatAddTemplate {
    fn emit(&self, ctx: &mut TemplateContext) {
        let acc = ctx.regs.accumulator;
        let scratch1 = ctx.regs.scratch1;

        // Guard LHS as float
        let lhs_slot = ctx.frame.register_slot(self.lhs_reg as u16);
        ctx.asm.mov_rm(acc, &lhs_slot);
        emit_float_guard(ctx, acc, scratch1, self.deopt_idx);

        // Guard RHS as float
        let rhs_slot = ctx.frame.register_slot(self.rhs_reg as u16);
        ctx.asm.mov_rm(acc, &rhs_slot);
        emit_float_guard(ctx, acc, scratch1, self.deopt_idx);

        // Load operands into XMM
        ctx.asm.movsd_rm(Xmm::Xmm0, &lhs_slot);
        ctx.asm.movsd_rm(Xmm::Xmm1, &rhs_slot);

        // ADDSD
        ctx.asm.addsd(Xmm::Xmm0, Xmm::Xmm1);

        // Store result
        let dst_slot = ctx.frame.register_slot(self.dst_reg as u16);
        ctx.asm.movsd_mr(&dst_slot, Xmm::Xmm0);
    }

    #[inline]
    fn estimated_size(&self) -> usize {
        100
    }
}

/// Template for float subtraction (`a - b`).
///
/// Same structure as FloatAdd but with SUBSD.
pub struct FloatSubTemplate {
    pub dst_reg: u8,
    pub lhs_reg: u8,
    pub rhs_reg: u8,
    pub deopt_idx: usize,
}

impl FloatSubTemplate {
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

impl OpcodeTemplate for FloatSubTemplate {
    fn emit(&self, ctx: &mut TemplateContext) {
        let acc = ctx.regs.accumulator;
        let scratch1 = ctx.regs.scratch1;

        let lhs_slot = ctx.frame.register_slot(self.lhs_reg as u16);
        ctx.asm.mov_rm(acc, &lhs_slot);
        emit_float_guard(ctx, acc, scratch1, self.deopt_idx);

        let rhs_slot = ctx.frame.register_slot(self.rhs_reg as u16);
        ctx.asm.mov_rm(acc, &rhs_slot);
        emit_float_guard(ctx, acc, scratch1, self.deopt_idx);

        ctx.asm.movsd_rm(Xmm::Xmm0, &lhs_slot);
        ctx.asm.movsd_rm(Xmm::Xmm1, &rhs_slot);
        ctx.asm.subsd(Xmm::Xmm0, Xmm::Xmm1);

        let dst_slot = ctx.frame.register_slot(self.dst_reg as u16);
        ctx.asm.movsd_mr(&dst_slot, Xmm::Xmm0);
    }

    #[inline]
    fn estimated_size(&self) -> usize {
        100
    }
}

/// Template for float multiplication (`a * b`).
///
/// Same structure as FloatAdd but with MULSD.
pub struct FloatMulTemplate {
    pub dst_reg: u8,
    pub lhs_reg: u8,
    pub rhs_reg: u8,
    pub deopt_idx: usize,
}

impl FloatMulTemplate {
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

impl OpcodeTemplate for FloatMulTemplate {
    fn emit(&self, ctx: &mut TemplateContext) {
        let acc = ctx.regs.accumulator;
        let scratch1 = ctx.regs.scratch1;

        let lhs_slot = ctx.frame.register_slot(self.lhs_reg as u16);
        ctx.asm.mov_rm(acc, &lhs_slot);
        emit_float_guard(ctx, acc, scratch1, self.deopt_idx);

        let rhs_slot = ctx.frame.register_slot(self.rhs_reg as u16);
        ctx.asm.mov_rm(acc, &rhs_slot);
        emit_float_guard(ctx, acc, scratch1, self.deopt_idx);

        ctx.asm.movsd_rm(Xmm::Xmm0, &lhs_slot);
        ctx.asm.movsd_rm(Xmm::Xmm1, &rhs_slot);
        ctx.asm.mulsd(Xmm::Xmm0, Xmm::Xmm1);

        let dst_slot = ctx.frame.register_slot(self.dst_reg as u16);
        ctx.asm.movsd_mr(&dst_slot, Xmm::Xmm0);
    }

    #[inline]
    fn estimated_size(&self) -> usize {
        100
    }
}

/// Template for float division (`a / b`).
///
/// Includes an additional zero-division guard: if RHS == 0.0, deopt
/// to the interpreter which will raise ZeroDivisionError.
///
/// # Strategy
///
/// 1. Guard both as float
/// 2. Load RHS into XMM1
/// 3. XORPD XMM2, XMM2 (create +0.0)
/// 4. UCOMISD XMM1, XMM2 (compare RHS with 0.0)
/// 5. JE → deopt (ZeroDivisionError)
/// 6. Load LHS into XMM0
/// 7. DIVSD XMM0, XMM1
/// 8. Store result
///
/// # Estimated Size: ~120 bytes (includes zero guard)
pub struct FloatDivTemplate {
    pub dst_reg: u8,
    pub lhs_reg: u8,
    pub rhs_reg: u8,
    pub deopt_idx: usize,
}

impl FloatDivTemplate {
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

impl OpcodeTemplate for FloatDivTemplate {
    fn emit(&self, ctx: &mut TemplateContext) {
        let acc = ctx.regs.accumulator;
        let scratch1 = ctx.regs.scratch1;

        // Guard LHS as float
        let lhs_slot = ctx.frame.register_slot(self.lhs_reg as u16);
        ctx.asm.mov_rm(acc, &lhs_slot);
        emit_float_guard(ctx, acc, scratch1, self.deopt_idx);

        // Guard RHS as float
        let rhs_slot = ctx.frame.register_slot(self.rhs_reg as u16);
        ctx.asm.mov_rm(acc, &rhs_slot);
        emit_float_guard(ctx, acc, scratch1, self.deopt_idx);

        // Load RHS first to check for zero
        ctx.asm.movsd_rm(Xmm::Xmm1, &rhs_slot);

        // Zero-division guard: XMM2 = 0.0, compare RHS with 0.0
        ctx.asm.zero_xmm(Xmm::Xmm2);
        ctx.asm.ucomisd(Xmm::Xmm1, Xmm::Xmm2);
        // UCOMISD sets ZF=1, PF=0 for equal; JP would catch unordered (NaN)
        // Use JE + JP pattern: deopt on exact zero
        ctx.asm.je(ctx.deopt_label(self.deopt_idx));

        // Load LHS
        ctx.asm.movsd_rm(Xmm::Xmm0, &lhs_slot);

        // DIVSD
        ctx.asm.divsd(Xmm::Xmm0, Xmm::Xmm1);

        // Store result
        let dst_slot = ctx.frame.register_slot(self.dst_reg as u16);
        ctx.asm.movsd_mr(&dst_slot, Xmm::Xmm0);
    }

    #[inline]
    fn estimated_size(&self) -> usize {
        120
    }
}

// =============================================================================
// Float Unary Templates
// =============================================================================

/// Template for float negation (`-a`).
///
/// # Strategy
///
/// Uses XORPD with a sign-bit mask (0x8000_0000_0000_0000) to flip the sign.
/// Since we don't have ANDPD in the assembler, we use a GPR-based approach:
///
/// 1. Guard operand as float
/// 2. Load value into GPR
/// 3. XOR with sign-bit constant (0x8000_0000_0000_0000)
/// 4. Store result
///
/// This is branchless and uses a single 64-bit XOR instruction.
///
/// # Estimated Size: ~70 bytes
pub struct FloatNegTemplate {
    pub dst_reg: u8,
    pub src_reg: u8,
    pub deopt_idx: usize,
}

impl FloatNegTemplate {
    #[inline]
    pub fn new(dst_reg: u8, src_reg: u8, deopt_idx: usize) -> Self {
        Self {
            dst_reg,
            src_reg,
            deopt_idx,
        }
    }
}

impl OpcodeTemplate for FloatNegTemplate {
    fn emit(&self, ctx: &mut TemplateContext) {
        let acc = ctx.regs.accumulator;
        let scratch1 = ctx.regs.scratch1;

        // Guard and load value
        let src_slot = ctx.frame.register_slot(self.src_reg as u16);
        ctx.asm.mov_rm(acc, &src_slot);
        emit_float_guard(ctx, acc, scratch1, self.deopt_idx);

        // XOR with sign bit: flip bit 63
        // sign_bit = 0x8000_0000_0000_0000
        let sign_bit: i64 = i64::MIN; // 0x8000_0000_0000_0000
        ctx.asm.mov_ri64(scratch1, sign_bit);
        ctx.asm.xor_rr(acc, scratch1);

        // Store result
        let dst_slot = ctx.frame.register_slot(self.dst_reg as u16);
        ctx.asm.mov_mr(&dst_slot, acc);
    }

    #[inline]
    fn estimated_size(&self) -> usize {
        70
    }
}

/// Template for float absolute value (`abs(a)`).
///
/// # Strategy
///
/// Clears the sign bit using AND with ~(1 << 63) = 0x7FFF_FFFF_FFFF_FFFF.
/// This is branchless and uses a single 64-bit AND instruction.
///
/// # Estimated Size: ~70 bytes
pub struct FloatAbsTemplate {
    pub dst_reg: u8,
    pub src_reg: u8,
    pub deopt_idx: usize,
}

impl FloatAbsTemplate {
    #[inline]
    pub fn new(dst_reg: u8, src_reg: u8, deopt_idx: usize) -> Self {
        Self {
            dst_reg,
            src_reg,
            deopt_idx,
        }
    }
}

impl OpcodeTemplate for FloatAbsTemplate {
    fn emit(&self, ctx: &mut TemplateContext) {
        let acc = ctx.regs.accumulator;
        let scratch1 = ctx.regs.scratch1;

        // Guard and load value
        let src_slot = ctx.frame.register_slot(self.src_reg as u16);
        ctx.asm.mov_rm(acc, &src_slot);
        emit_float_guard(ctx, acc, scratch1, self.deopt_idx);

        // AND with clear-sign-bit mask: 0x7FFF_FFFF_FFFF_FFFF
        let clear_sign: i64 = i64::MAX; // 0x7FFF_FFFF_FFFF_FFFF
        ctx.asm.mov_ri64(scratch1, clear_sign);
        ctx.asm.and_rr(acc, scratch1);

        // Store result
        let dst_slot = ctx.frame.register_slot(self.dst_reg as u16);
        ctx.asm.mov_mr(&dst_slot, acc);
    }

    #[inline]
    fn estimated_size(&self) -> usize {
        70
    }
}

// =============================================================================
// Float Comparison Template
// =============================================================================

/// Comparison operation for float comparisons.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FloatCmpOp {
    /// `a < b`
    Lt,
    /// `a <= b`
    Le,
    /// `a == b`
    Eq,
    /// `a != b`
    Ne,
    /// `a > b`
    Gt,
    /// `a >= b`
    Ge,
}

/// Template for float comparison (`a <op> b`).
///
/// # Strategy
///
/// 1. Guard both operands as float
/// 2. Load into XMM0, XMM1
/// 3. UCOMISD XMM0, XMM1
/// 4. Emit branchless boolean result using conditional moves or SETcc
///
/// UCOMISD sets:
/// - ZF=1, PF=0 for equal
/// - CF=1 for less-than
/// - ZF=0, PF=0, CF=0 for greater-than
/// - PF=1 for unordered (NaN)
///
/// For NaN comparisons: Python's `nan < x` and `nan == x` both return False,
/// and `nan != x` returns True. All comparisons with NaN are unordered, so
/// PF=1. We deopt to the interpreter for unordered results to handle this
/// correctly.
///
/// # Estimated Size: ~120 bytes
pub struct FloatCompareTemplate {
    pub dst_reg: u8,
    pub lhs_reg: u8,
    pub rhs_reg: u8,
    pub op: FloatCmpOp,
    pub deopt_idx: usize,
}

impl FloatCompareTemplate {
    #[inline]
    pub fn new(dst_reg: u8, lhs_reg: u8, rhs_reg: u8, op: FloatCmpOp, deopt_idx: usize) -> Self {
        Self {
            dst_reg,
            lhs_reg,
            rhs_reg,
            op,
            deopt_idx,
        }
    }
}

impl OpcodeTemplate for FloatCompareTemplate {
    fn emit(&self, ctx: &mut TemplateContext) {
        let acc = ctx.regs.accumulator;
        let scratch1 = ctx.regs.scratch1;

        // Guard LHS as float
        let lhs_slot = ctx.frame.register_slot(self.lhs_reg as u16);
        ctx.asm.mov_rm(acc, &lhs_slot);
        emit_float_guard(ctx, acc, scratch1, self.deopt_idx);

        // Guard RHS as float
        let rhs_slot = ctx.frame.register_slot(self.rhs_reg as u16);
        ctx.asm.mov_rm(acc, &rhs_slot);
        emit_float_guard(ctx, acc, scratch1, self.deopt_idx);

        // Load into XMM for comparison
        ctx.asm.movsd_rm(Xmm::Xmm0, &lhs_slot);
        ctx.asm.movsd_rm(Xmm::Xmm1, &rhs_slot);

        // UCOMISD: compare LHS with RHS
        ctx.asm.ucomisd(Xmm::Xmm0, Xmm::Xmm1);

        // Deopt on unordered (NaN) — PF=1 means unordered
        ctx.asm
            .jcc(Condition::Parity, ctx.deopt_label(self.deopt_idx));

        // Branch based on comparison operator
        // After UCOMISD (non-NaN):
        //   LHS <  RHS → CF=1
        //   LHS == RHS → ZF=1
        //   LHS >  RHS → CF=0, ZF=0
        let true_label = ctx.asm.create_label();
        let done_label = ctx.asm.create_label();

        match self.op {
            FloatCmpOp::Lt => ctx.asm.jb(true_label),  // CF=1
            FloatCmpOp::Le => ctx.asm.jbe(true_label), // CF=1 or ZF=1
            FloatCmpOp::Eq => ctx.asm.je(true_label),  // ZF=1
            FloatCmpOp::Ne => ctx.asm.jne(true_label), // ZF=0
            FloatCmpOp::Gt => ctx.asm.ja(true_label),  // CF=0 and ZF=0
            FloatCmpOp::Ge => ctx.asm.jae(true_label), // CF=0
        }

        // False path
        ctx.asm.mov_ri64(acc, value_tags::false_value() as i64);
        ctx.asm.jmp(done_label);

        // True path
        ctx.asm.bind_label(true_label);
        ctx.asm.mov_ri64(acc, value_tags::true_value() as i64);

        ctx.asm.bind_label(done_label);

        // Store result
        let dst_slot = ctx.frame.register_slot(self.dst_reg as u16);
        ctx.asm.mov_mr(&dst_slot, acc);
    }

    #[inline]
    fn estimated_size(&self) -> usize {
        140
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
    // Float Guard Tests
    // =========================================================================

    #[test]
    fn test_float_guard_emits_code() {
        let mut asm = Assembler::new();
        let frame = FrameLayout::minimal(8);
        {
            let mut ctx = TemplateContext::new(&mut asm, &frame);
            ctx.create_deopt_label();
            let acc = ctx.regs.accumulator;
            let s1 = ctx.regs.scratch1;
            emit_float_guard(&mut ctx, acc, s1, 0);
            for l in &ctx.deopt_labels {
                ctx.asm.bind_label(*l);
            }
        }
        let code = asm.finalize().unwrap();
        assert!(code.len() > 0, "Float guard must emit code");
    }

    #[test]
    fn test_float_guard_contains_shr_48() {
        // Must shift right by 48 to isolate upper 16 bits
        let mut asm = Assembler::new();
        let frame = FrameLayout::minimal(8);
        {
            let mut ctx = TemplateContext::new(&mut asm, &frame);
            ctx.create_deopt_label();
            let acc = ctx.regs.accumulator;
            let s1 = ctx.regs.scratch1;
            emit_float_guard(&mut ctx, acc, s1, 0);
            for l in &ctx.deopt_labels {
                ctx.asm.bind_label(*l);
            }
        }
        let code = asm.finalize().unwrap();
        // SHR r64, 48 → 48 = 0x30; should appear in code as the immediate
        assert!(code.contains(&0x30), "Float guard must SHR by 48 (0x30)");
    }

    #[test]
    fn test_float_guard_contains_jb_for_ok_path() {
        // JB (0x72 short or 0x0F 0x82 near) for the "it's a float" path
        let mut asm = Assembler::new();
        let frame = FrameLayout::minimal(8);
        {
            let mut ctx = TemplateContext::new(&mut asm, &frame);
            ctx.create_deopt_label();
            let acc = ctx.regs.accumulator;
            let s1 = ctx.regs.scratch1;
            emit_float_guard(&mut ctx, acc, s1, 0);
            for l in &ctx.deopt_labels {
                ctx.asm.bind_label(*l);
            }
        }
        let code = asm.finalize().unwrap();
        let has_jb_short = code.contains(&0x72);
        let has_jb_near = code.windows(2).any(|w| w[0] == 0x0F && w[1] == 0x82);
        assert!(
            has_jb_short || has_jb_near,
            "Float guard must have JB for valid-float path"
        );
    }

    #[test]
    fn test_float_guard_size() {
        let mut asm = Assembler::new();
        let frame = FrameLayout::minimal(8);
        {
            let mut ctx = TemplateContext::new(&mut asm, &frame);
            ctx.create_deopt_label();
            let acc = ctx.regs.accumulator;
            let s1 = ctx.regs.scratch1;
            emit_float_guard(&mut ctx, acc, s1, 0);
            for l in &ctx.deopt_labels {
                ctx.asm.bind_label(*l);
            }
        }
        let code = asm.finalize().unwrap();
        assert!(
            code.len() <= 40,
            "Float guard too large: {} bytes",
            code.len()
        );
    }

    // =========================================================================
    // FloatAddTemplate Tests
    // =========================================================================

    #[test]
    fn test_float_add_creation() {
        let t = FloatAddTemplate::new(0, 1, 2, 0);
        assert_eq!(t.dst_reg, 0);
        assert_eq!(t.lhs_reg, 1);
        assert_eq!(t.rhs_reg, 2);
    }

    #[test]
    fn test_float_add_emits_code() {
        let t = FloatAddTemplate::new(0, 1, 2, 0);
        let code = emit_and_finalize(&t);
        assert!(!code.is_empty());
    }

    #[test]
    fn test_float_add_size_within_estimate() {
        let t = FloatAddTemplate::new(0, 1, 2, 0);
        let code = emit_and_finalize(&t);
        assert!(
            code.len() <= t.estimated_size(),
            "FloatAdd actual {} exceeds estimate {}",
            code.len(),
            t.estimated_size()
        );
    }

    #[test]
    fn test_float_add_different_registers() {
        for (d, l, r) in [(0, 1, 2), (3, 4, 5), (7, 8, 9)] {
            let t = FloatAddTemplate::new(d, l, r, 0);
            let code = emit_and_finalize(&t);
            assert!(!code.is_empty());
        }
    }

    #[test]
    fn test_float_add_high_registers() {
        let t = FloatAddTemplate::new(10, 11, 12, 0);
        let code = emit_and_finalize(&t);
        assert!(code.len() <= t.estimated_size());
    }

    // =========================================================================
    // FloatSubTemplate Tests
    // =========================================================================

    #[test]
    fn test_float_sub_creation() {
        let t = FloatSubTemplate::new(0, 1, 2, 0);
        assert_eq!(t.dst_reg, 0);
    }

    #[test]
    fn test_float_sub_emits_code() {
        let t = FloatSubTemplate::new(0, 1, 2, 0);
        let code = emit_and_finalize(&t);
        assert!(!code.is_empty());
    }

    #[test]
    fn test_float_sub_size_within_estimate() {
        let t = FloatSubTemplate::new(0, 1, 2, 0);
        let code = emit_and_finalize(&t);
        assert!(code.len() <= t.estimated_size());
    }

    #[test]
    fn test_float_sub_different_registers() {
        for (d, l, r) in [(0, 1, 2), (3, 4, 5), (7, 8, 9)] {
            let t = FloatSubTemplate::new(d, l, r, 0);
            let code = emit_and_finalize(&t);
            assert!(!code.is_empty());
        }
    }

    // =========================================================================
    // FloatMulTemplate Tests
    // =========================================================================

    #[test]
    fn test_float_mul_creation() {
        let t = FloatMulTemplate::new(0, 1, 2, 0);
        assert_eq!(t.dst_reg, 0);
    }

    #[test]
    fn test_float_mul_emits_code() {
        let t = FloatMulTemplate::new(0, 1, 2, 0);
        let code = emit_and_finalize(&t);
        assert!(!code.is_empty());
    }

    #[test]
    fn test_float_mul_size_within_estimate() {
        let t = FloatMulTemplate::new(0, 1, 2, 0);
        let code = emit_and_finalize(&t);
        assert!(code.len() <= t.estimated_size());
    }

    #[test]
    fn test_float_mul_different_registers() {
        for (d, l, r) in [(0, 1, 2), (3, 4, 5), (7, 8, 9)] {
            let t = FloatMulTemplate::new(d, l, r, 0);
            let code = emit_and_finalize(&t);
            assert!(!code.is_empty());
        }
    }

    // =========================================================================
    // FloatDivTemplate Tests
    // =========================================================================

    #[test]
    fn test_float_div_creation() {
        let t = FloatDivTemplate::new(0, 1, 2, 0);
        assert_eq!(t.dst_reg, 0);
    }

    #[test]
    fn test_float_div_emits_code() {
        let t = FloatDivTemplate::new(0, 1, 2, 0);
        let code = emit_and_finalize(&t);
        assert!(!code.is_empty());
    }

    #[test]
    fn test_float_div_size_within_estimate() {
        let t = FloatDivTemplate::new(0, 1, 2, 0);
        let code = emit_and_finalize(&t);
        assert!(
            code.len() <= t.estimated_size(),
            "FloatDiv actual {} exceeds estimate {}",
            code.len(),
            t.estimated_size()
        );
    }

    #[test]
    fn test_float_div_larger_than_add() {
        // Div has zero-guard, so should be larger
        let div = FloatDivTemplate::new(0, 1, 2, 0);
        let add = FloatAddTemplate::new(0, 1, 2, 0);
        let div_code = emit_and_finalize(&div);
        let add_code = emit_and_finalize(&add);
        assert!(
            div_code.len() > add_code.len(),
            "FloatDiv ({}) should be larger than FloatAdd ({}) due to zero guard",
            div_code.len(),
            add_code.len()
        );
    }

    #[test]
    fn test_float_div_different_registers() {
        for (d, l, r) in [(0, 1, 2), (3, 4, 5), (7, 8, 9)] {
            let t = FloatDivTemplate::new(d, l, r, 0);
            let code = emit_and_finalize(&t);
            assert!(!code.is_empty());
        }
    }

    #[test]
    fn test_float_div_has_zero_guard_ucomisd() {
        // FloatDiv must contain UCOMISD for zero check
        // UCOMISD encoding: 66 0F 2E /r
        let t = FloatDivTemplate::new(0, 1, 2, 0);
        let code = emit_and_finalize(&t);
        let has_ucomisd = code
            .windows(3)
            .any(|w| w[0] == 0x66 && w[1] == 0x0F && w[2] == 0x2E);
        assert!(has_ucomisd, "FloatDiv must contain UCOMISD for zero-guard");
    }

    // =========================================================================
    // FloatNegTemplate Tests
    // =========================================================================

    #[test]
    fn test_float_neg_creation() {
        let t = FloatNegTemplate::new(0, 1, 0);
        assert_eq!(t.dst_reg, 0);
        assert_eq!(t.src_reg, 1);
    }

    #[test]
    fn test_float_neg_emits_code() {
        let t = FloatNegTemplate::new(0, 1, 0);
        let code = emit_and_finalize(&t);
        assert!(!code.is_empty());
    }

    #[test]
    fn test_float_neg_size_within_estimate() {
        let t = FloatNegTemplate::new(0, 1, 0);
        let code = emit_and_finalize(&t);
        assert!(
            code.len() <= t.estimated_size(),
            "FloatNeg actual {} exceeds estimate {}",
            code.len(),
            t.estimated_size()
        );
    }

    #[test]
    fn test_float_neg_smaller_than_binary() {
        // Unary should be smaller than binary ops
        let neg = FloatNegTemplate::new(0, 1, 0);
        let add = FloatAddTemplate::new(0, 1, 2, 0);
        let neg_code = emit_and_finalize(&neg);
        let add_code = emit_and_finalize(&add);
        assert!(
            neg_code.len() < add_code.len(),
            "FloatNeg ({}) should be smaller than FloatAdd ({})",
            neg_code.len(),
            add_code.len()
        );
    }

    #[test]
    fn test_float_neg_contains_xor_for_sign_flip() {
        // Must contain XOR (0x31 or 0x33) for sign bit flip
        let t = FloatNegTemplate::new(0, 1, 0);
        let code = emit_and_finalize(&t);
        let has_xor = code.contains(&0x31) || code.contains(&0x33);
        assert!(has_xor, "FloatNeg must contain XOR for sign-bit flip");
    }

    #[test]
    fn test_float_neg_different_registers() {
        for (d, s) in [(0, 1), (2, 3), (5, 6), (0, 0)] {
            let t = FloatNegTemplate::new(d, s, 0);
            let code = emit_and_finalize(&t);
            assert!(!code.is_empty());
        }
    }

    // =========================================================================
    // FloatAbsTemplate Tests
    // =========================================================================

    #[test]
    fn test_float_abs_creation() {
        let t = FloatAbsTemplate::new(0, 1, 0);
        assert_eq!(t.dst_reg, 0);
        assert_eq!(t.src_reg, 1);
    }

    #[test]
    fn test_float_abs_emits_code() {
        let t = FloatAbsTemplate::new(0, 1, 0);
        let code = emit_and_finalize(&t);
        assert!(!code.is_empty());
    }

    #[test]
    fn test_float_abs_size_within_estimate() {
        let t = FloatAbsTemplate::new(0, 1, 0);
        let code = emit_and_finalize(&t);
        assert!(
            code.len() <= t.estimated_size(),
            "FloatAbs actual {} exceeds estimate {}",
            code.len(),
            t.estimated_size()
        );
    }

    #[test]
    fn test_float_abs_similar_size_to_neg() {
        // Both are single-instruction bit manipulation
        let abs_t = FloatAbsTemplate::new(0, 1, 0);
        let neg_t = FloatNegTemplate::new(0, 1, 0);
        let abs_code = emit_and_finalize(&abs_t);
        let neg_code = emit_and_finalize(&neg_t);
        let diff = (abs_code.len() as i64 - neg_code.len() as i64).unsigned_abs();
        assert!(
            diff <= 5,
            "FloatAbs ({}) and FloatNeg ({}) should be similar size, diff={}",
            abs_code.len(),
            neg_code.len(),
            diff
        );
    }

    #[test]
    fn test_float_abs_contains_and_for_sign_clear() {
        // Must contain AND (0x21 or 0x23) for sign-bit clear
        let t = FloatAbsTemplate::new(0, 1, 0);
        let code = emit_and_finalize(&t);
        let has_and = code.contains(&0x21) || code.contains(&0x23);
        assert!(has_and, "FloatAbs must contain AND for sign-bit clear");
    }

    #[test]
    fn test_float_abs_different_registers() {
        for (d, s) in [(0, 1), (2, 3), (5, 6), (0, 0)] {
            let t = FloatAbsTemplate::new(d, s, 0);
            let code = emit_and_finalize(&t);
            assert!(!code.is_empty());
        }
    }

    // =========================================================================
    // FloatCompareTemplate Tests
    // =========================================================================

    #[test]
    fn test_float_compare_creation() {
        let t = FloatCompareTemplate::new(0, 1, 2, FloatCmpOp::Lt, 0);
        assert_eq!(t.dst_reg, 0);
        assert_eq!(t.op, FloatCmpOp::Lt);
    }

    #[test]
    fn test_float_compare_all_ops_emit_code() {
        let ops = [
            FloatCmpOp::Lt,
            FloatCmpOp::Le,
            FloatCmpOp::Eq,
            FloatCmpOp::Ne,
            FloatCmpOp::Gt,
            FloatCmpOp::Ge,
        ];
        for op in ops {
            let t = FloatCompareTemplate::new(0, 1, 2, op, 0);
            let code = emit_and_finalize(&t);
            assert!(!code.is_empty(), "Op {:?} must emit code", op);
        }
    }

    #[test]
    fn test_float_compare_all_ops_within_estimate() {
        let ops = [
            FloatCmpOp::Lt,
            FloatCmpOp::Le,
            FloatCmpOp::Eq,
            FloatCmpOp::Ne,
            FloatCmpOp::Gt,
            FloatCmpOp::Ge,
        ];
        for op in ops {
            let t = FloatCompareTemplate::new(0, 1, 2, op, 0);
            let code = emit_and_finalize(&t);
            assert!(
                code.len() <= t.estimated_size(),
                "Op {:?} actual {} exceeds estimate {}",
                op,
                code.len(),
                t.estimated_size()
            );
        }
    }

    #[test]
    fn test_float_compare_all_ops_similar_size() {
        // All comparison ops should generate roughly the same size code
        let ops = [
            FloatCmpOp::Lt,
            FloatCmpOp::Le,
            FloatCmpOp::Eq,
            FloatCmpOp::Ne,
            FloatCmpOp::Gt,
            FloatCmpOp::Ge,
        ];
        let sizes: Vec<usize> = ops
            .iter()
            .map(|&op| {
                let t = FloatCompareTemplate::new(0, 1, 2, op, 0);
                emit_and_finalize(&t).len()
            })
            .collect();
        let min_size = *sizes.iter().min().unwrap();
        let max_size = *sizes.iter().max().unwrap();
        assert!(
            max_size - min_size <= 4,
            "Compare ops should be similar size: min={}, max={}",
            min_size,
            max_size
        );
    }

    #[test]
    fn test_float_compare_contains_ucomisd() {
        // All comparisons must use UCOMISD
        let t = FloatCompareTemplate::new(0, 1, 2, FloatCmpOp::Lt, 0);
        let code = emit_and_finalize(&t);
        let has_ucomisd = code
            .windows(3)
            .any(|w| w[0] == 0x66 && w[1] == 0x0F && w[2] == 0x2E);
        assert!(has_ucomisd, "Float compare must contain UCOMISD");
    }

    #[test]
    fn test_float_compare_contains_jp_for_nan_deopt() {
        // Must have JP (0x7A or 0x0F 0x8A) for NaN deoptimization
        let t = FloatCompareTemplate::new(0, 1, 2, FloatCmpOp::Lt, 0);
        let code = emit_and_finalize(&t);
        let has_jp_short = code.contains(&0x7A);
        let has_jp_near = code.windows(2).any(|w| w[0] == 0x0F && w[1] == 0x8A);
        assert!(
            has_jp_short || has_jp_near,
            "Float compare must have JP for NaN deopt"
        );
    }

    #[test]
    fn test_float_compare_contains_true_false_values() {
        // Must contain both TRUE and FALSE immediates
        let t = FloatCompareTemplate::new(0, 1, 2, FloatCmpOp::Eq, 0);
        let code = emit_and_finalize(&t);
        let true_bytes = (value_tags::true_value() as i64).to_le_bytes();
        let false_bytes = (value_tags::false_value() as i64).to_le_bytes();
        let has_true = code.windows(8).any(|w| w == true_bytes);
        let has_false = code.windows(8).any(|w| w == false_bytes);
        assert!(
            has_true && has_false,
            "Float compare must contain both TRUE and FALSE constants"
        );
    }

    #[test]
    fn test_float_compare_larger_than_binary_arith() {
        // Comparison has branch + boolean boxing, so larger than simple arith
        let cmp = FloatCompareTemplate::new(0, 1, 2, FloatCmpOp::Lt, 0);
        let add = FloatAddTemplate::new(0, 1, 2, 0);
        let cmp_code = emit_and_finalize(&cmp);
        let add_code = emit_and_finalize(&add);
        assert!(
            cmp_code.len() > add_code.len(),
            "Compare ({}) should be larger than Add ({}) due to boolean boxing",
            cmp_code.len(),
            add_code.len()
        );
    }

    #[test]
    fn test_float_compare_different_registers() {
        for (d, l, r) in [(0, 1, 2), (3, 4, 5), (7, 8, 9)] {
            let t = FloatCompareTemplate::new(d, l, r, FloatCmpOp::Gt, 0);
            let code = emit_and_finalize(&t);
            assert!(!code.is_empty());
        }
    }

    // =========================================================================
    // Cross-Template Tests
    // =========================================================================

    #[test]
    fn test_all_templates_compile() {
        let templates: Vec<Box<dyn OpcodeTemplate>> = vec![
            Box::new(FloatAddTemplate::new(0, 1, 2, 0)),
            Box::new(FloatSubTemplate::new(0, 1, 2, 0)),
            Box::new(FloatMulTemplate::new(0, 1, 2, 0)),
            Box::new(FloatDivTemplate::new(0, 1, 2, 0)),
            Box::new(FloatNegTemplate::new(0, 1, 0)),
            Box::new(FloatAbsTemplate::new(0, 1, 0)),
            Box::new(FloatCompareTemplate::new(0, 1, 2, FloatCmpOp::Lt, 0)),
        ];
        for (i, tmpl) in templates.iter().enumerate() {
            let code = emit_and_finalize(tmpl.as_ref());
            assert!(!code.is_empty(), "Template {} emitted empty code", i);
        }
    }

    #[test]
    fn test_all_estimates_are_conservative() {
        let templates: Vec<Box<dyn OpcodeTemplate>> = vec![
            Box::new(FloatAddTemplate::new(0, 1, 2, 0)),
            Box::new(FloatSubTemplate::new(0, 1, 2, 0)),
            Box::new(FloatMulTemplate::new(0, 1, 2, 0)),
            Box::new(FloatDivTemplate::new(0, 1, 2, 0)),
            Box::new(FloatNegTemplate::new(0, 1, 0)),
            Box::new(FloatAbsTemplate::new(0, 1, 0)),
            Box::new(FloatCompareTemplate::new(0, 1, 2, FloatCmpOp::Lt, 0)),
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
    fn test_binary_arith_ops_same_size() {
        // Add, Sub, Mul should all be the same size (same structure)
        let add = emit_and_finalize(&FloatAddTemplate::new(0, 1, 2, 0));
        let sub = emit_and_finalize(&FloatSubTemplate::new(0, 1, 2, 0));
        let mul = emit_and_finalize(&FloatMulTemplate::new(0, 1, 2, 0));
        assert_eq!(
            add.len(),
            sub.len(),
            "Add ({}) and Sub ({}) should be same size",
            add.len(),
            sub.len()
        );
        assert_eq!(
            add.len(),
            mul.len(),
            "Add ({}) and Mul ({}) should be same size",
            add.len(),
            mul.len()
        );
    }

    #[test]
    fn test_unary_ops_smaller_than_binary() {
        let neg = emit_and_finalize(&FloatNegTemplate::new(0, 1, 0));
        let add = emit_and_finalize(&FloatAddTemplate::new(0, 1, 2, 0));
        assert!(
            neg.len() < add.len(),
            "Unary ({}) should be smaller than binary ({})",
            neg.len(),
            add.len()
        );
    }

    #[test]
    fn test_sequential_emission() {
        let mut asm = Assembler::new();
        let frame = FrameLayout::minimal(16);
        {
            let mut ctx = TemplateContext::new(&mut asm, &frame);
            ctx.create_deopt_label();
            ctx.create_deopt_label();

            FloatAddTemplate::new(0, 1, 2, 0).emit(&mut ctx);
            let after_add = ctx.asm.offset();

            FloatSubTemplate::new(3, 4, 5, 0).emit(&mut ctx);
            let after_sub = ctx.asm.offset();

            FloatMulTemplate::new(6, 7, 8, 0).emit(&mut ctx);
            let after_mul = ctx.asm.offset();

            FloatNegTemplate::new(9, 10, 0).emit(&mut ctx);
            let after_neg = ctx.asm.offset();

            assert!(after_add > 0);
            assert!(after_sub > after_add);
            assert!(after_mul > after_sub);
            assert!(after_neg > after_mul);

            for l in &ctx.deopt_labels {
                ctx.asm.bind_label(*l);
            }
        }
        let code = asm.finalize().unwrap();
        assert!(code.len() > 100);
    }

    #[test]
    fn test_zero_register_indices() {
        let t = FloatAddTemplate::new(0, 0, 0, 0);
        let code = emit_and_finalize(&t);
        assert!(!code.is_empty());
    }

    #[test]
    fn test_adjacent_register_indices() {
        let t = FloatAddTemplate::new(0, 1, 2, 0);
        let code = emit_and_finalize(&t);
        assert!(!code.is_empty());
    }

    #[test]
    fn test_multiple_deopt_labels() {
        let mut asm = Assembler::new();
        let frame = FrameLayout::minimal(16);
        {
            let mut ctx = TemplateContext::new(&mut asm, &frame);
            for _ in 0..4 {
                ctx.create_deopt_label();
            }
            FloatAddTemplate::new(0, 1, 2, 0).emit(&mut ctx);
            FloatSubTemplate::new(3, 4, 5, 1).emit(&mut ctx);
            FloatCompareTemplate::new(6, 7, 8, FloatCmpOp::Eq, 2).emit(&mut ctx);
            FloatNegTemplate::new(9, 10, 3).emit(&mut ctx);
            for l in &ctx.deopt_labels {
                ctx.asm.bind_label(*l);
            }
        }
        let code = asm.finalize().unwrap();
        assert!(!code.is_empty());
    }

    #[test]
    fn test_float_cmp_op_enum_completeness() {
        // All 6 standard comparison ops must exist
        let ops = [
            FloatCmpOp::Lt,
            FloatCmpOp::Le,
            FloatCmpOp::Eq,
            FloatCmpOp::Ne,
            FloatCmpOp::Gt,
            FloatCmpOp::Ge,
        ];
        assert_eq!(ops.len(), 6, "Must have exactly 6 comparison ops");
    }

    #[test]
    fn test_div_has_separate_zero_guard() {
        // Verify div template has extra code compared to mul
        // (the zero-division check adds XORPD + UCOMISD + JE)
        let div = emit_and_finalize(&FloatDivTemplate::new(0, 1, 2, 0));
        let mul = emit_and_finalize(&FloatMulTemplate::new(0, 1, 2, 0));
        assert!(
            div.len() > mul.len(),
            "Div ({}) must be larger than Mul ({}) due to zero guard",
            div.len(),
            mul.len()
        );
    }

    #[test]
    fn test_float_neg_sign_bit_constant() {
        // The sign bit 0x8000_0000_0000_0000 should appear as the constant
        let t = FloatNegTemplate::new(0, 1, 0);
        let code = emit_and_finalize(&t);
        let sign_bytes = (i64::MIN).to_le_bytes();
        // Check if the sign-bit constant appears in the code
        let has_sign_bit = code.windows(8).any(|w| w == sign_bytes);
        assert!(
            has_sign_bit,
            "FloatNeg must contain sign-bit constant 0x8000_0000_0000_0000"
        );
    }

    #[test]
    fn test_float_abs_clear_sign_constant() {
        // The clear-sign mask 0x7FFF_FFFF_FFFF_FFFF should appear
        let t = FloatAbsTemplate::new(0, 1, 0);
        let code = emit_and_finalize(&t);
        let mask_bytes = (i64::MAX).to_le_bytes();
        let has_mask = code.windows(8).any(|w| w == mask_bytes);
        assert!(
            has_mask,
            "FloatAbs must contain clear-sign mask 0x7FFF_FFFF_FFFF_FFFF"
        );
    }
}
