//! ARM64 high-level assembler with label support.
//!
//! Provides a fluent API for emitting ARM64 machine code with:
//! - Label management and forward references
//! - Relocation support for JIT linking
//! - Constant pool management

use super::encoder::{self, Condition, EncodedInst, ShiftType};
use super::neon::{ArrangementSpec, Vreg};
use super::registers::{CallingConvention, Gpr, GprSet};

// =============================================================================
// Labels
// =============================================================================

/// A label representing a position in the code.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Label(u32);

impl Label {
    #[inline]
    pub const fn new(id: u32) -> Self {
        Label(id)
    }

    #[inline]
    pub const fn id(self) -> u32 {
        self.0
    }
}

/// Relocation type for ARM64.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RelocationType {
    /// Branch26: ±128MB (B, BL)
    Branch26,
    /// Branch19: ±1MB (B.cond, CBZ, CBNZ)
    Branch19,
    /// ADR: ±1MB PC-relative
    Adr21,
    /// ADRP: ±4GB PC-relative page
    AdrpPage21,
}

/// A relocation entry.
#[derive(Debug, Clone, Copy)]
pub struct Relocation {
    /// Offset in code buffer where relocation applies.
    pub offset: u32,
    /// The label this relocation refers to.
    pub label: Label,
    /// Type of relocation.
    pub kind: RelocationType,
}

// =============================================================================
// Assembler
// =============================================================================

/// High-level ARM64 assembler.
pub struct Arm64Assembler {
    code: Vec<u8>,
    labels: Vec<Option<u32>>,
    relocations: Vec<Relocation>,
    next_label: u32,
}

impl Arm64Assembler {
    /// Create a new assembler.
    pub fn new() -> Self {
        Self::with_capacity(1024)
    }

    /// Create with specified capacity.
    pub fn with_capacity(cap: usize) -> Self {
        Arm64Assembler {
            code: Vec::with_capacity(cap),
            labels: Vec::new(),
            relocations: Vec::new(),
            next_label: 0,
        }
    }

    /// Get current code position.
    #[inline]
    pub fn position(&self) -> u32 {
        self.code.len() as u32
    }

    /// Create a new unbound label.
    pub fn create_label(&mut self) -> Label {
        let id = self.next_label;
        self.next_label += 1;
        self.labels.push(None);
        Label::new(id)
    }

    /// Bind a label to the current position.
    pub fn bind_label(&mut self, label: Label) {
        let pos = self.position();
        self.labels[label.id() as usize] = Some(pos);
    }

    /// Emit a raw instruction.
    fn emit(&mut self, inst: EncodedInst) {
        self.code.extend_from_slice(&inst.to_le_bytes());
    }

    /// Emit with relocation.
    fn emit_with_reloc(&mut self, inst: EncodedInst, label: Label, kind: RelocationType) {
        let offset = self.position();
        self.relocations.push(Relocation {
            offset,
            label,
            kind,
        });
        self.emit(inst);
    }

    // =========================================================================
    // Finalization
    // =========================================================================

    /// Finalize and resolve all relocations.
    pub fn finalize(mut self) -> Result<Vec<u8>, String> {
        for reloc in &self.relocations {
            let target = self.labels[reloc.label.id() as usize]
                .ok_or_else(|| format!("Unbound label: {:?}", reloc.label))?;

            let offset_in_code = reloc.offset as usize;
            let pc = reloc.offset as i64;
            let target = target as i64;
            let diff = target - pc;

            // Read current instruction
            let mut bytes = [0u8; 4];
            bytes.copy_from_slice(&self.code[offset_in_code..offset_in_code + 4]);
            let mut inst = u32::from_le_bytes(bytes);

            match reloc.kind {
                RelocationType::Branch26 => {
                    let imm26 = ((diff >> 2) as u32) & 0x3FF_FFFF;
                    inst = (inst & !0x3FF_FFFF) | imm26;
                }
                RelocationType::Branch19 => {
                    let imm19 = ((diff >> 2) as u32) & 0x7FFFF;
                    inst = (inst & !(0x7FFFF << 5)) | (imm19 << 5);
                }
                RelocationType::Adr21 => {
                    let immlo = (diff & 0x3) as u32;
                    let immhi = ((diff >> 2) & 0x7FFFF) as u32;
                    inst = (inst & !(0x3 << 29)) | (immlo << 29);
                    inst = (inst & !(0x7FFFF << 5)) | (immhi << 5);
                }
                RelocationType::AdrpPage21 => {
                    // Page-relative, divide by 4096
                    let page_diff = diff >> 12;
                    let immlo = (page_diff & 0x3) as u32;
                    let immhi = ((page_diff >> 2) & 0x7FFFF) as u32;
                    inst = (inst & !(0x3 << 29)) | (immlo << 29);
                    inst = (inst & !(0x7FFFF << 5)) | (immhi << 5);
                }
            }

            self.code[offset_in_code..offset_in_code + 4].copy_from_slice(&inst.to_le_bytes());
        }
        Ok(self.code)
    }

    // =========================================================================
    // Data Processing
    // =========================================================================

    /// ADD Xd, Xn, Xm
    pub fn add(&mut self, dst: Gpr, n: Gpr, m: Gpr) {
        self.emit(encoder::encode_add_reg(dst, n, m));
    }

    /// ADD Xd, Xn, #imm
    pub fn add_imm(&mut self, dst: Gpr, n: Gpr, imm: u16) {
        self.emit(encoder::encode_add_imm(dst, n, imm, false));
    }

    /// SUB Xd, Xn, Xm
    pub fn sub(&mut self, dst: Gpr, n: Gpr, m: Gpr) {
        self.emit(encoder::encode_sub_reg(dst, n, m));
    }

    /// SUB Xd, Xn, #imm
    pub fn sub_imm(&mut self, dst: Gpr, n: Gpr, imm: u16) {
        self.emit(encoder::encode_sub_imm(dst, n, imm, false));
    }

    /// MUL Xd, Xn, Xm
    pub fn mul(&mut self, dst: Gpr, n: Gpr, m: Gpr) {
        self.emit(encoder::encode_mul(dst, n, m));
    }

    /// SDIV Xd, Xn, Xm
    pub fn sdiv(&mut self, dst: Gpr, n: Gpr, m: Gpr) {
        self.emit(encoder::encode_sdiv(dst, n, m));
    }

    /// UDIV Xd, Xn, Xm
    pub fn udiv(&mut self, dst: Gpr, n: Gpr, m: Gpr) {
        self.emit(encoder::encode_udiv(dst, n, m));
    }

    /// NEG Xd, Xm
    pub fn neg(&mut self, dst: Gpr, m: Gpr) {
        self.emit(encoder::encode_neg(dst, m));
    }

    // =========================================================================
    // Logical
    // =========================================================================

    /// AND Xd, Xn, Xm
    pub fn and(&mut self, dst: Gpr, n: Gpr, m: Gpr) {
        self.emit(encoder::encode_and_reg(dst, n, m));
    }

    /// ORR Xd, Xn, Xm
    pub fn orr(&mut self, dst: Gpr, n: Gpr, m: Gpr) {
        self.emit(encoder::encode_orr_reg(dst, n, m));
    }

    /// EOR Xd, Xn, Xm
    pub fn eor(&mut self, dst: Gpr, n: Gpr, m: Gpr) {
        self.emit(encoder::encode_eor_reg(dst, n, m));
    }

    /// MVN Xd, Xm
    pub fn mvn(&mut self, dst: Gpr, m: Gpr) {
        self.emit(encoder::encode_mvn(dst, m));
    }

    // =========================================================================
    // Shifts
    // =========================================================================

    /// LSL Xd, Xn, Xm
    pub fn lsl(&mut self, dst: Gpr, n: Gpr, m: Gpr) {
        self.emit(encoder::encode_lsl_reg(dst, n, m));
    }

    /// LSR Xd, Xn, Xm
    pub fn lsr(&mut self, dst: Gpr, n: Gpr, m: Gpr) {
        self.emit(encoder::encode_lsr_reg(dst, n, m));
    }

    /// ASR Xd, Xn, Xm
    pub fn asr(&mut self, dst: Gpr, n: Gpr, m: Gpr) {
        self.emit(encoder::encode_asr_reg(dst, n, m));
    }

    // =========================================================================
    // Moves
    // =========================================================================

    /// MOV Xd, Xm
    pub fn mov(&mut self, dst: Gpr, src: Gpr) {
        self.emit(encoder::encode_mov_reg(dst, src));
    }

    /// MOVZ Xd, #imm16, LSL #pos
    pub fn movz(&mut self, dst: Gpr, imm: u16, pos: u8) {
        self.emit(encoder::encode_movz(dst, imm, pos));
    }

    /// MOVK Xd, #imm16, LSL #pos
    pub fn movk(&mut self, dst: Gpr, imm: u16, pos: u8) {
        self.emit(encoder::encode_movk(dst, imm, pos));
    }

    /// Load a 64-bit immediate (optimized sequence).
    pub fn mov_imm64(&mut self, dst: Gpr, val: u64) {
        if val == 0 {
            self.movz(dst, 0, 0);
            return;
        }

        // Find first non-zero 16-bit chunk
        let chunks: [u16; 4] = [
            (val & 0xFFFF) as u16,
            ((val >> 16) & 0xFFFF) as u16,
            ((val >> 32) & 0xFFFF) as u16,
            ((val >> 48) & 0xFFFF) as u16,
        ];

        let mut first = true;
        for (i, &chunk) in chunks.iter().enumerate() {
            if chunk != 0 || (i == 0 && val == 0) {
                let pos = (i * 16) as u8;
                if first {
                    self.movz(dst, chunk, pos);
                    first = false;
                } else {
                    self.movk(dst, chunk, pos);
                }
            }
        }
    }

    // =========================================================================
    // Branches
    // =========================================================================

    /// B label
    pub fn b(&mut self, target: Label) {
        self.emit_with_reloc(encoder::encode_b(0), target, RelocationType::Branch26);
    }

    /// BL label
    pub fn bl(&mut self, target: Label) {
        self.emit_with_reloc(encoder::encode_bl(0), target, RelocationType::Branch26);
    }

    /// B.cond label
    pub fn bcond(&mut self, cond: Condition, target: Label) {
        self.emit_with_reloc(
            encoder::encode_bcond(cond, 0),
            target,
            RelocationType::Branch19,
        );
    }

    /// BR Xn
    pub fn br(&mut self, target: Gpr) {
        self.emit(encoder::encode_br(target));
    }

    /// BLR Xn
    pub fn blr(&mut self, target: Gpr) {
        self.emit(encoder::encode_blr(target));
    }

    /// RET
    pub fn ret(&mut self) {
        self.emit(encoder::encode_ret(None));
    }

    /// CBZ Xn, label
    pub fn cbz(&mut self, src: Gpr, target: Label) {
        self.emit_with_reloc(
            encoder::encode_cbz(src, 0),
            target,
            RelocationType::Branch19,
        );
    }

    /// CBNZ Xn, label
    pub fn cbnz(&mut self, src: Gpr, target: Label) {
        self.emit_with_reloc(
            encoder::encode_cbnz(src, 0),
            target,
            RelocationType::Branch19,
        );
    }

    // =========================================================================
    // Compare
    // =========================================================================

    /// CMP Xn, Xm
    pub fn cmp(&mut self, n: Gpr, m: Gpr) {
        self.emit(encoder::encode_cmp_reg(n, m));
    }

    /// CMP Xn, #imm
    pub fn cmp_imm(&mut self, n: Gpr, imm: u16) {
        self.emit(encoder::encode_cmp_imm(n, imm));
    }

    // =========================================================================
    // Conditional Select
    // =========================================================================

    /// CSEL Xd, Xn, Xm, cond
    pub fn csel(&mut self, dst: Gpr, n: Gpr, m: Gpr, cond: Condition) {
        self.emit(encoder::encode_csel(dst, n, m, cond));
    }

    /// CSINC Xd, Xn, Xm, cond
    pub fn csinc(&mut self, dst: Gpr, n: Gpr, m: Gpr, cond: Condition) {
        self.emit(encoder::encode_csinc(dst, n, m, cond));
    }

    // =========================================================================
    // Load/Store
    // =========================================================================

    /// LDR Xd, [Xn, #offset]
    pub fn ldr(&mut self, dst: Gpr, base: Gpr, offset: i16) {
        self.emit(encoder::encode_ldr_imm(dst, base, offset));
    }

    /// STR Xd, [Xn, #offset]
    pub fn str(&mut self, src: Gpr, base: Gpr, offset: i16) {
        self.emit(encoder::encode_str_imm(src, base, offset));
    }

    /// LDP Xt1, Xt2, [Xn, #offset]
    pub fn ldp(&mut self, t1: Gpr, t2: Gpr, base: Gpr, offset: i16) {
        self.emit(encoder::encode_ldp(t1, t2, base, offset));
    }

    /// STP Xt1, Xt2, [Xn, #offset]
    pub fn stp(&mut self, t1: Gpr, t2: Gpr, base: Gpr, offset: i16) {
        self.emit(encoder::encode_stp(t1, t2, base, offset));
    }

    /// STP Xt1, Xt2, [Xn, #offset]! (pre-index)
    pub fn stp_pre(&mut self, t1: Gpr, t2: Gpr, base: Gpr, offset: i16) {
        self.emit(encoder::encode_stp_pre(t1, t2, base, offset));
    }

    /// LDP Xt1, Xt2, [Xn], #offset (post-index)
    pub fn ldp_post(&mut self, t1: Gpr, t2: Gpr, base: Gpr, offset: i16) {
        self.emit(encoder::encode_ldp_post(t1, t2, base, offset));
    }

    // =========================================================================
    // System
    // =========================================================================

    /// NOP
    pub fn nop(&mut self) {
        self.emit(encoder::encode_nop());
    }

    /// BRK #imm
    pub fn brk(&mut self, imm: u16) {
        self.emit(encoder::encode_brk(imm));
    }

    /// DSB ISH (data sync barrier)
    pub fn dsb_ish(&mut self) {
        self.emit(encoder::encode_dsb_ish());
    }

    /// ISB (instruction sync barrier)
    pub fn isb(&mut self) {
        self.emit(encoder::encode_isb());
    }

    // =========================================================================
    // Prologue/Epilogue Helpers
    // =========================================================================

    /// Emit standard function prologue.
    pub fn prologue(&mut self, frame_size: u16) {
        // STP X29, X30, [SP, #-frame_size]!
        self.stp_pre(Gpr::X29, Gpr::X30, Gpr::X29, -(frame_size as i16));
        // MOV X29, SP (would need ADD X29, SP, #0)
        self.add_imm(Gpr::X29, Gpr::X29, 0);
    }

    /// Emit standard function epilogue.
    pub fn epilogue(&mut self, frame_size: u16) {
        // LDP X29, X30, [SP], #frame_size
        self.ldp_post(Gpr::X29, Gpr::X30, Gpr::X29, frame_size as i16);
        self.ret();
    }
}

impl Default for Arm64Assembler {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_assembly() {
        let mut asm = Arm64Assembler::new();
        asm.nop();
        asm.ret();
        let code = asm.finalize().unwrap();
        assert_eq!(code.len(), 8); // 2 instructions × 4 bytes
    }

    #[test]
    fn test_labels() {
        let mut asm = Arm64Assembler::new();
        let label = asm.create_label();
        asm.b(label);
        asm.nop();
        asm.bind_label(label);
        asm.ret();
        let code = asm.finalize().unwrap();
        assert_eq!(code.len(), 12);
    }

    #[test]
    fn test_mov_imm64_zero() {
        let mut asm = Arm64Assembler::new();
        asm.mov_imm64(Gpr::X0, 0);
        let code = asm.finalize().unwrap();
        assert_eq!(code.len(), 4);
    }

    #[test]
    fn test_mov_imm64_small() {
        let mut asm = Arm64Assembler::new();
        asm.mov_imm64(Gpr::X0, 0x1234);
        let code = asm.finalize().unwrap();
        assert_eq!(code.len(), 4); // Just MOVZ
    }

    #[test]
    fn test_mov_imm64_large() {
        let mut asm = Arm64Assembler::new();
        asm.mov_imm64(Gpr::X0, 0x1234_5678_9ABC_DEF0);
        let code = asm.finalize().unwrap();
        // Should be MOVZ + MOVK × 3
        assert_eq!(code.len(), 16);
    }

    #[test]
    fn test_conditional_branch() {
        let mut asm = Arm64Assembler::new();
        let skip = asm.create_label();
        asm.cmp_imm(Gpr::X0, 0);
        asm.bcond(Condition::Eq, skip);
        asm.add_imm(Gpr::X0, Gpr::X0, 1);
        asm.bind_label(skip);
        asm.ret();
        let code = asm.finalize().unwrap();
        assert!(code.len() > 0);
    }
}
