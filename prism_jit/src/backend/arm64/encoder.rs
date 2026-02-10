//! ARM64 A64 instruction encoder.
//!
//! This module provides low-level encoding for ARM64 instructions.
//! All A64 instructions are exactly 32 bits (4 bytes), little-endian.

use super::registers::{Extend, Gpr, MemOffset, MemOperand, SP_ENCODING, ZR_ENCODING};

// =============================================================================
// Condition Codes
// =============================================================================

/// ARM64 condition codes for conditional branches and selects.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum Condition {
    /// Equal (Z == 1)
    Eq = 0b0000,
    /// Not equal (Z == 0)
    Ne = 0b0001,
    /// Carry set / unsigned higher or same (C == 1)
    Cs = 0b0010,
    /// Carry clear / unsigned lower (C == 0)
    Cc = 0b0011,
    /// Minus / negative (N == 1)
    Mi = 0b0100,
    /// Plus / positive or zero (N == 0)
    Pl = 0b0101,
    /// Overflow (V == 1)
    Vs = 0b0110,
    /// No overflow (V == 0)
    Vc = 0b0111,
    /// Unsigned higher (C == 1 && Z == 0)
    Hi = 0b1000,
    /// Unsigned lower or same (C == 0 || Z == 1)
    Ls = 0b1001,
    /// Signed greater or equal (N == V)
    Ge = 0b1010,
    /// Signed less than (N != V)
    Lt = 0b1011,
    /// Signed greater than (Z == 0 && N == V)
    Gt = 0b1100,
    /// Signed less than or equal (Z == 1 || N != V)
    Le = 0b1101,
    /// Always (unconditional)
    Al = 0b1110,
    /// Never (reserved, but encodes as always)
    Nv = 0b1111,
}

impl Condition {
    /// Get the 4-bit encoding.
    #[inline(always)]
    pub const fn encoding(self) -> u32 {
        self as u32
    }

    /// Get the inverted condition.
    #[inline]
    pub const fn invert(self) -> Self {
        match self {
            Condition::Eq => Condition::Ne,
            Condition::Ne => Condition::Eq,
            Condition::Cs => Condition::Cc,
            Condition::Cc => Condition::Cs,
            Condition::Mi => Condition::Pl,
            Condition::Pl => Condition::Mi,
            Condition::Vs => Condition::Vc,
            Condition::Vc => Condition::Vs,
            Condition::Hi => Condition::Ls,
            Condition::Ls => Condition::Hi,
            Condition::Ge => Condition::Lt,
            Condition::Lt => Condition::Ge,
            Condition::Gt => Condition::Le,
            Condition::Le => Condition::Gt,
            Condition::Al => Condition::Nv,
            Condition::Nv => Condition::Al,
        }
    }
}

// Aliases for common conditions
impl Condition {
    /// Alias: Unsigned higher or same
    pub const Hs: Condition = Condition::Cs;
    /// Alias: Unsigned lower
    pub const Lo: Condition = Condition::Cc;
}

// =============================================================================
// Shift Types
// =============================================================================

/// Shift type for data processing instructions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum ShiftType {
    /// Logical shift left
    Lsl = 0b00,
    /// Logical shift right
    Lsr = 0b01,
    /// Arithmetic shift right
    Asr = 0b10,
    /// Rotate right
    Ror = 0b11,
}

impl ShiftType {
    /// Get the 2-bit encoding.
    #[inline(always)]
    pub const fn encoding(self) -> u32 {
        self as u32
    }
}

// =============================================================================
// Encoded Instruction
// =============================================================================

/// An encoded ARM64 instruction (always 32 bits).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EncodedInst(u32);

impl EncodedInst {
    /// Create from raw bits.
    #[inline(always)]
    pub const fn new(bits: u32) -> Self {
        Self(bits)
    }

    /// Get the raw bits.
    #[inline(always)]
    pub const fn bits(self) -> u32 {
        self.0
    }

    /// Get as little-endian bytes.
    #[inline(always)]
    pub const fn to_le_bytes(self) -> [u8; 4] {
        self.0.to_le_bytes()
    }
}

impl Default for EncodedInst {
    fn default() -> Self {
        Self(0)
    }
}

// =============================================================================
// Data Processing - Immediate
// =============================================================================

/// ADD Xd, Xn, #imm12 (64-bit)
#[inline]
pub fn encode_add_imm(dst: Gpr, src: Gpr, imm: u16, shift_12: bool) -> EncodedInst {
    debug_assert!(imm < 4096);
    let sf = 1u32; // 64-bit
    let op = 0u32; // ADD (not SUB)
    let s = 0u32; // Don't set flags
    let sh = shift_12 as u32;
    EncodedInst::new(
        (sf << 31)
            | (op << 30)
            | (s << 29)
            | (0b100010 << 23)
            | (sh << 22)
            | ((imm as u32) << 10)
            | (src.field() << 5)
            | dst.field(),
    )
}

/// ADDS Xd, Xn, #imm12 (64-bit, sets flags)
#[inline]
pub fn encode_adds_imm(dst: Gpr, src: Gpr, imm: u16, shift_12: bool) -> EncodedInst {
    debug_assert!(imm < 4096);
    let sf = 1u32;
    let op = 0u32;
    let s = 1u32; // Set flags
    let sh = shift_12 as u32;
    EncodedInst::new(
        (sf << 31)
            | (op << 30)
            | (s << 29)
            | (0b100010 << 23)
            | (sh << 22)
            | ((imm as u32) << 10)
            | (src.field() << 5)
            | dst.field(),
    )
}

/// SUB Xd, Xn, #imm12 (64-bit)
#[inline]
pub fn encode_sub_imm(dst: Gpr, src: Gpr, imm: u16, shift_12: bool) -> EncodedInst {
    debug_assert!(imm < 4096);
    let sf = 1u32;
    let op = 1u32; // SUB
    let s = 0u32;
    let sh = shift_12 as u32;
    EncodedInst::new(
        (sf << 31)
            | (op << 30)
            | (s << 29)
            | (0b100010 << 23)
            | (sh << 22)
            | ((imm as u32) << 10)
            | (src.field() << 5)
            | dst.field(),
    )
}

/// SUBS Xd, Xn, #imm12 (64-bit, sets flags)
#[inline]
pub fn encode_subs_imm(dst: Gpr, src: Gpr, imm: u16, shift_12: bool) -> EncodedInst {
    debug_assert!(imm < 4096);
    let sf = 1u32;
    let op = 1u32;
    let s = 1u32;
    let sh = shift_12 as u32;
    EncodedInst::new(
        (sf << 31)
            | (op << 30)
            | (s << 29)
            | (0b100010 << 23)
            | (sh << 22)
            | ((imm as u32) << 10)
            | (src.field() << 5)
            | dst.field(),
    )
}

// =============================================================================
// Data Processing - Register
// =============================================================================

/// ADD Xd, Xn, Xm (64-bit)
#[inline]
pub fn encode_add_reg(dst: Gpr, src1: Gpr, src2: Gpr) -> EncodedInst {
    encode_add_shifted(dst, src1, src2, ShiftType::Lsl, 0)
}

/// ADD Xd, Xn, Xm, shift #amount
#[inline]
pub fn encode_add_shifted(
    dst: Gpr,
    src1: Gpr,
    src2: Gpr,
    shift: ShiftType,
    amount: u8,
) -> EncodedInst {
    debug_assert!(amount < 64);
    let sf = 1u32;
    let op = 0u32;
    let s = 0u32;
    EncodedInst::new(
        (sf << 31)
            | (op << 30)
            | (s << 29)
            | (0b01011 << 24)
            | (shift.encoding() << 22)
            | (0 << 21)
            | (src2.field() << 16)
            | ((amount as u32) << 10)
            | (src1.field() << 5)
            | dst.field(),
    )
}

/// SUB Xd, Xn, Xm (64-bit)
#[inline]
pub fn encode_sub_reg(dst: Gpr, src1: Gpr, src2: Gpr) -> EncodedInst {
    encode_sub_shifted(dst, src1, src2, ShiftType::Lsl, 0)
}

/// SUB Xd, Xn, Xm, shift #amount
#[inline]
pub fn encode_sub_shifted(
    dst: Gpr,
    src1: Gpr,
    src2: Gpr,
    shift: ShiftType,
    amount: u8,
) -> EncodedInst {
    debug_assert!(amount < 64);
    let sf = 1u32;
    let op = 1u32;
    let s = 0u32;
    EncodedInst::new(
        (sf << 31)
            | (op << 30)
            | (s << 29)
            | (0b01011 << 24)
            | (shift.encoding() << 22)
            | (0 << 21)
            | (src2.field() << 16)
            | ((amount as u32) << 10)
            | (src1.field() << 5)
            | dst.field(),
    )
}

/// NEG Xd, Xm (alias for SUB Xd, XZR, Xm)
#[inline]
pub fn encode_neg(dst: Gpr, src: Gpr) -> EncodedInst {
    // NEG is SUB Xd, XZR, Xm
    let sf = 1u32;
    let op = 1u32;
    let s = 0u32;
    EncodedInst::new(
        (sf << 31)
            | (op << 30)
            | (s << 29)
            | (0b01011 << 24)
            | (0 << 22)
            | (0 << 21)
            | (src.field() << 16)
            | (0 << 10)
            | (ZR_ENCODING << 5)
            | dst.field(),
    )
}

// =============================================================================
// Multiply/Divide
// =============================================================================

/// MUL Xd, Xn, Xm (alias for MADD Xd, Xn, Xm, XZR)
#[inline]
pub fn encode_mul(dst: Gpr, src1: Gpr, src2: Gpr) -> EncodedInst {
    encode_madd(dst, src1, src2, None)
}

/// MADD Xd, Xn, Xm, Xa (Xd = Xa + Xn * Xm)
#[inline]
pub fn encode_madd(dst: Gpr, n: Gpr, m: Gpr, a: Option<Gpr>) -> EncodedInst {
    let sf = 1u32;
    let ra = a.map(|r| r.field()).unwrap_or(ZR_ENCODING);
    EncodedInst::new(
        (sf << 31)
            | (0b0011011000 << 21)
            | (m.field() << 16)
            | (0 << 15)
            | (ra << 10)
            | (n.field() << 5)
            | dst.field(),
    )
}

/// MSUB Xd, Xn, Xm, Xa (Xd = Xa - Xn * Xm)
#[inline]
pub fn encode_msub(dst: Gpr, n: Gpr, m: Gpr, a: Gpr) -> EncodedInst {
    let sf = 1u32;
    EncodedInst::new(
        (sf << 31)
            | (0b0011011000 << 21)
            | (m.field() << 16)
            | (1 << 15)
            | (a.field() << 10)
            | (n.field() << 5)
            | dst.field(),
    )
}

/// SDIV Xd, Xn, Xm (signed divide)
#[inline]
pub fn encode_sdiv(dst: Gpr, n: Gpr, m: Gpr) -> EncodedInst {
    let sf = 1u32;
    EncodedInst::new(
        (sf << 31)
            | (0b0011010110 << 21)
            | (m.field() << 16)
            | (0b000011 << 10)
            | (n.field() << 5)
            | dst.field(),
    )
}

/// UDIV Xd, Xn, Xm (unsigned divide)
#[inline]
pub fn encode_udiv(dst: Gpr, n: Gpr, m: Gpr) -> EncodedInst {
    let sf = 1u32;
    EncodedInst::new(
        (sf << 31)
            | (0b0011010110 << 21)
            | (m.field() << 16)
            | (0b000010 << 10)
            | (n.field() << 5)
            | dst.field(),
    )
}

// =============================================================================
// Logical Operations
// =============================================================================

/// AND Xd, Xn, Xm
#[inline]
pub fn encode_and_reg(dst: Gpr, src1: Gpr, src2: Gpr) -> EncodedInst {
    let sf = 1u32;
    let opc = 0b00u32; // AND
    EncodedInst::new(
        (sf << 31)
            | (opc << 29)
            | (0b01010 << 24)
            | (0 << 22)
            | (0 << 21)
            | (src2.field() << 16)
            | (0 << 10)
            | (src1.field() << 5)
            | dst.field(),
    )
}

/// ORR Xd, Xn, Xm
#[inline]
pub fn encode_orr_reg(dst: Gpr, src1: Gpr, src2: Gpr) -> EncodedInst {
    let sf = 1u32;
    let opc = 0b01u32; // ORR
    EncodedInst::new(
        (sf << 31)
            | (opc << 29)
            | (0b01010 << 24)
            | (0 << 22)
            | (0 << 21)
            | (src2.field() << 16)
            | (0 << 10)
            | (src1.field() << 5)
            | dst.field(),
    )
}

/// EOR Xd, Xn, Xm
#[inline]
pub fn encode_eor_reg(dst: Gpr, src1: Gpr, src2: Gpr) -> EncodedInst {
    let sf = 1u32;
    let opc = 0b10u32; // EOR
    EncodedInst::new(
        (sf << 31)
            | (opc << 29)
            | (0b01010 << 24)
            | (0 << 22)
            | (0 << 21)
            | (src2.field() << 16)
            | (0 << 10)
            | (src1.field() << 5)
            | dst.field(),
    )
}

/// MVN Xd, Xm (alias for ORN Xd, XZR, Xm)
#[inline]
pub fn encode_mvn(dst: Gpr, src: Gpr) -> EncodedInst {
    let sf = 1u32;
    let opc = 0b01u32;
    let n = 1u32; // NOT
    EncodedInst::new(
        (sf << 31)
            | (opc << 29)
            | (0b01010 << 24)
            | (0 << 22)
            | (n << 21)
            | (src.field() << 16)
            | (0 << 10)
            | (ZR_ENCODING << 5)
            | dst.field(),
    )
}

// =============================================================================
// Shifts
// =============================================================================

/// LSL Xd, Xn, Xm (logical shift left by register)
#[inline]
pub fn encode_lsl_reg(dst: Gpr, n: Gpr, m: Gpr) -> EncodedInst {
    let sf = 1u32;
    EncodedInst::new(
        (sf << 31)
            | (0b0011010110 << 21)
            | (m.field() << 16)
            | (0b001000 << 10)
            | (n.field() << 5)
            | dst.field(),
    )
}

/// LSR Xd, Xn, Xm (logical shift right by register)
#[inline]
pub fn encode_lsr_reg(dst: Gpr, n: Gpr, m: Gpr) -> EncodedInst {
    let sf = 1u32;
    EncodedInst::new(
        (sf << 31)
            | (0b0011010110 << 21)
            | (m.field() << 16)
            | (0b001001 << 10)
            | (n.field() << 5)
            | dst.field(),
    )
}

/// ASR Xd, Xn, Xm (arithmetic shift right by register)
#[inline]
pub fn encode_asr_reg(dst: Gpr, n: Gpr, m: Gpr) -> EncodedInst {
    let sf = 1u32;
    EncodedInst::new(
        (sf << 31)
            | (0b0011010110 << 21)
            | (m.field() << 16)
            | (0b001010 << 10)
            | (n.field() << 5)
            | dst.field(),
    )
}

// =============================================================================
// Move Instructions
// =============================================================================

/// MOVZ Xd, #imm16, LSL #pos (move wide with zero)
#[inline]
pub fn encode_movz(dst: Gpr, imm: u16, pos: u8) -> EncodedInst {
    debug_assert!(pos == 0 || pos == 16 || pos == 32 || pos == 48);
    let sf = 1u32;
    let opc = 0b10u32;
    let hw = (pos / 16) as u32;
    EncodedInst::new(
        (sf << 31)
            | (opc << 29)
            | (0b100101 << 23)
            | (hw << 21)
            | ((imm as u32) << 5)
            | dst.field(),
    )
}

/// MOVN Xd, #imm16, LSL #pos (move wide with NOT)
#[inline]
pub fn encode_movn(dst: Gpr, imm: u16, pos: u8) -> EncodedInst {
    debug_assert!(pos == 0 || pos == 16 || pos == 32 || pos == 48);
    let sf = 1u32;
    let opc = 0b00u32;
    let hw = (pos / 16) as u32;
    EncodedInst::new(
        (sf << 31)
            | (opc << 29)
            | (0b100101 << 23)
            | (hw << 21)
            | ((imm as u32) << 5)
            | dst.field(),
    )
}

/// MOVK Xd, #imm16, LSL #pos (move wide with keep)
#[inline]
pub fn encode_movk(dst: Gpr, imm: u16, pos: u8) -> EncodedInst {
    debug_assert!(pos == 0 || pos == 16 || pos == 32 || pos == 48);
    let sf = 1u32;
    let opc = 0b11u32;
    let hw = (pos / 16) as u32;
    EncodedInst::new(
        (sf << 31)
            | (opc << 29)
            | (0b100101 << 23)
            | (hw << 21)
            | ((imm as u32) << 5)
            | dst.field(),
    )
}

/// MOV Xd, Xm (alias for ORR Xd, XZR, Xm)
#[inline]
pub fn encode_mov_reg(dst: Gpr, src: Gpr) -> EncodedInst {
    encode_orr_reg(dst, Gpr::X0, src) // Using X0 as placeholder for XZR
}

// =============================================================================
// PC-Relative Address
// =============================================================================

/// ADR Xd, label (PC-relative, ±1MB)
#[inline]
pub fn encode_adr(dst: Gpr, offset: i32) -> EncodedInst {
    let immlo = (offset & 0x3) as u32;
    let immhi = ((offset >> 2) & 0x7FFFF) as u32;
    EncodedInst::new((0 << 31) | (immlo << 29) | (0b10000 << 24) | (immhi << 5) | dst.field())
}

/// ADRP Xd, label (PC-relative page, ±4GB)
#[inline]
pub fn encode_adrp(dst: Gpr, offset: i32) -> EncodedInst {
    let immlo = (offset & 0x3) as u32;
    let immhi = ((offset >> 2) & 0x7FFFF) as u32;
    EncodedInst::new((1 << 31) | (immlo << 29) | (0b10000 << 24) | (immhi << 5) | dst.field())
}

// =============================================================================
// Branch Instructions
// =============================================================================

/// B label (unconditional, ±128MB)
#[inline]
pub fn encode_b(offset: i32) -> EncodedInst {
    let imm26 = ((offset >> 2) as u32) & 0x3FF_FFFF;
    EncodedInst::new((0b000101 << 26) | imm26)
}

/// BL label (branch with link, ±128MB)
#[inline]
pub fn encode_bl(offset: i32) -> EncodedInst {
    let imm26 = ((offset >> 2) as u32) & 0x3FF_FFFF;
    EncodedInst::new((0b100101 << 26) | imm26)
}

/// B.cond label (conditional, ±1MB)
#[inline]
pub fn encode_bcond(cond: Condition, offset: i32) -> EncodedInst {
    let imm19 = ((offset >> 2) as u32) & 0x7FFFF;
    EncodedInst::new((0b01010100 << 24) | (imm19 << 5) | cond.encoding())
}

/// BR Xn (branch to register)
#[inline]
pub fn encode_br(target: Gpr) -> EncodedInst {
    EncodedInst::new(0xD61F0000 | (target.field() << 5))
}

/// BLR Xn (branch with link to register)
#[inline]
pub fn encode_blr(target: Gpr) -> EncodedInst {
    EncodedInst::new(0xD63F0000 | (target.field() << 5))
}

/// RET {Xn} (defaults to X30/LR)
#[inline]
pub fn encode_ret(target: Option<Gpr>) -> EncodedInst {
    let rn = target.map(|r| r.field()).unwrap_or(30);
    EncodedInst::new(0xD65F0000 | (rn << 5))
}

// =============================================================================
// Compare and Branch
// =============================================================================

/// CBZ Xn, label (compare and branch if zero)
#[inline]
pub fn encode_cbz(src: Gpr, offset: i32) -> EncodedInst {
    let sf = 1u32;
    let imm19 = ((offset >> 2) as u32) & 0x7FFFF;
    EncodedInst::new((sf << 31) | (0b011010 << 25) | (0 << 24) | (imm19 << 5) | src.field())
}

/// CBNZ Xn, label (compare and branch if not zero)
#[inline]
pub fn encode_cbnz(src: Gpr, offset: i32) -> EncodedInst {
    let sf = 1u32;
    let imm19 = ((offset >> 2) as u32) & 0x7FFFF;
    EncodedInst::new((sf << 31) | (0b011010 << 25) | (1 << 24) | (imm19 << 5) | src.field())
}

// =============================================================================
// Compare Instructions
// =============================================================================

/// CMP Xn, Xm (alias for SUBS XZR, Xn, Xm)
#[inline]
pub fn encode_cmp_reg(n: Gpr, m: Gpr) -> EncodedInst {
    let sf = 1u32;
    EncodedInst::new(
        (sf << 31)
            | (1 << 30)
            | (1 << 29)
            | (0b01011 << 24)
            | (0 << 22)
            | (0 << 21)
            | (m.field() << 16)
            | (0 << 10)
            | (n.field() << 5)
            | ZR_ENCODING,
    )
}

/// CMP Xn, #imm12 (alias for SUBS XZR, Xn, #imm12)
#[inline]
pub fn encode_cmp_imm(n: Gpr, imm: u16) -> EncodedInst {
    debug_assert!(imm < 4096);
    let sf = 1u32;
    EncodedInst::new(
        (sf << 31)
            | (1 << 30)
            | (1 << 29)
            | (0b100010 << 23)
            | (0 << 22)
            | ((imm as u32) << 10)
            | (n.field() << 5)
            | ZR_ENCODING,
    )
}

// =============================================================================
// Conditional Select
// =============================================================================

/// CSEL Xd, Xn, Xm, cond
#[inline]
pub fn encode_csel(dst: Gpr, n: Gpr, m: Gpr, cond: Condition) -> EncodedInst {
    let sf = 1u32;
    EncodedInst::new(
        (sf << 31)
            | (0 << 30)
            | (0 << 29)
            | (0b11010100 << 21)
            | (m.field() << 16)
            | (cond.encoding() << 12)
            | (0 << 10)
            | (n.field() << 5)
            | dst.field(),
    )
}

/// CSINC Xd, Xn, Xm, cond (Xd = cond ? Xn : Xm + 1)
#[inline]
pub fn encode_csinc(dst: Gpr, n: Gpr, m: Gpr, cond: Condition) -> EncodedInst {
    let sf = 1u32;
    EncodedInst::new(
        (sf << 31)
            | (0 << 30)
            | (0 << 29)
            | (0b11010100 << 21)
            | (m.field() << 16)
            | (cond.encoding() << 12)
            | (1 << 10)
            | (n.field() << 5)
            | dst.field(),
    )
}

// =============================================================================
// Load/Store Instructions
// =============================================================================

/// LDR Xd, [Xn, #imm] (unsigned offset, 64-bit)
#[inline]
pub fn encode_ldr_imm(dst: Gpr, base: Gpr, offset: i16) -> EncodedInst {
    debug_assert!(offset >= 0 && offset % 8 == 0);
    let imm12 = (offset as u32) / 8;
    debug_assert!(imm12 < 4096);
    EncodedInst::new(0xF9400000 | (imm12 << 10) | (base.field() << 5) | dst.field())
}

/// STR Xd, [Xn, #imm] (unsigned offset, 64-bit)
#[inline]
pub fn encode_str_imm(src: Gpr, base: Gpr, offset: i16) -> EncodedInst {
    debug_assert!(offset >= 0 && offset % 8 == 0);
    let imm12 = (offset as u32) / 8;
    debug_assert!(imm12 < 4096);
    EncodedInst::new(0xF9000000 | (imm12 << 10) | (base.field() << 5) | src.field())
}

/// LDP Xt1, Xt2, [Xn, #imm] (load pair)
#[inline]
pub fn encode_ldp(t1: Gpr, t2: Gpr, base: Gpr, offset: i16) -> EncodedInst {
    debug_assert!(offset % 8 == 0);
    let imm7 = ((offset / 8) as u32) & 0x7F;
    EncodedInst::new(
        0xA9400000 | (imm7 << 15) | (t2.field() << 10) | (base.field() << 5) | t1.field(),
    )
}

/// STP Xt1, Xt2, [Xn, #imm] (store pair)
#[inline]
pub fn encode_stp(t1: Gpr, t2: Gpr, base: Gpr, offset: i16) -> EncodedInst {
    debug_assert!(offset % 8 == 0);
    let imm7 = ((offset / 8) as u32) & 0x7F;
    EncodedInst::new(
        0xA9000000 | (imm7 << 15) | (t2.field() << 10) | (base.field() << 5) | t1.field(),
    )
}

/// STP Xt1, Xt2, [Xn, #imm]! (store pair, pre-index)
#[inline]
pub fn encode_stp_pre(t1: Gpr, t2: Gpr, base: Gpr, offset: i16) -> EncodedInst {
    debug_assert!(offset % 8 == 0);
    let imm7 = ((offset / 8) as u32) & 0x7F;
    EncodedInst::new(
        0xA9800000 | (imm7 << 15) | (t2.field() << 10) | (base.field() << 5) | t1.field(),
    )
}

/// LDP Xt1, Xt2, [Xn], #imm (load pair, post-index)
#[inline]
pub fn encode_ldp_post(t1: Gpr, t2: Gpr, base: Gpr, offset: i16) -> EncodedInst {
    debug_assert!(offset % 8 == 0);
    let imm7 = ((offset / 8) as u32) & 0x7F;
    EncodedInst::new(
        0xA8C00000 | (imm7 << 15) | (t2.field() << 10) | (base.field() << 5) | t1.field(),
    )
}

// =============================================================================
// System Instructions
// =============================================================================

/// NOP
#[inline]
pub const fn encode_nop() -> EncodedInst {
    EncodedInst::new(0xD503201F)
}

/// BRK #imm16 (breakpoint)
#[inline]
pub fn encode_brk(imm: u16) -> EncodedInst {
    EncodedInst::new(0xD4200000 | ((imm as u32) << 5))
}

/// DSB ISH (data synchronization barrier, inner shareable)
#[inline]
pub const fn encode_dsb_ish() -> EncodedInst {
    EncodedInst::new(0xD5033B9F)
}

/// ISB SY (instruction synchronization barrier)
#[inline]
pub const fn encode_isb() -> EncodedInst {
    EncodedInst::new(0xD5033FDF)
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_condition_encoding() {
        assert_eq!(Condition::Eq.encoding(), 0);
        assert_eq!(Condition::Ne.encoding(), 1);
        assert_eq!(Condition::Ge.encoding(), 10);
        assert_eq!(Condition::Lt.encoding(), 11);
    }

    #[test]
    fn test_condition_invert() {
        assert_eq!(Condition::Eq.invert(), Condition::Ne);
        assert_eq!(Condition::Ne.invert(), Condition::Eq);
        assert_eq!(Condition::Ge.invert(), Condition::Lt);
    }

    #[test]
    fn test_encoded_inst() {
        let inst = EncodedInst::new(0x12345678);
        assert_eq!(inst.bits(), 0x12345678);
        assert_eq!(inst.to_le_bytes(), [0x78, 0x56, 0x34, 0x12]);
    }

    #[test]
    fn test_nop() {
        assert_eq!(encode_nop().bits(), 0xD503201F);
    }

    #[test]
    fn test_ret() {
        assert_eq!(encode_ret(None).bits(), 0xD65F03C0);
    }

    #[test]
    fn test_b_encoding() {
        let b = encode_b(0);
        assert_eq!(b.bits() >> 26, 0b000101);
    }

    #[test]
    fn test_bl_encoding() {
        let bl = encode_bl(0);
        assert_eq!(bl.bits() >> 26, 0b100101);
    }

    #[test]
    fn test_add_imm() {
        let add = encode_add_imm(Gpr::X0, Gpr::X1, 42, false);
        assert_eq!(add.bits() & 0x1F, 0); // dst = X0
        assert_eq!((add.bits() >> 5) & 0x1F, 1); // src = X1
    }

    #[test]
    fn test_movz() {
        let movz = encode_movz(Gpr::X0, 0x1234, 0);
        assert_eq!(movz.bits() & 0x1F, 0);
        assert_eq!((movz.bits() >> 5) & 0xFFFF, 0x1234);
    }

    #[test]
    fn test_cmp_reg() {
        let cmp = encode_cmp_reg(Gpr::X1, Gpr::X2);
        // Destination should be XZR (31)
        assert_eq!(cmp.bits() & 0x1F, 31);
    }
}
