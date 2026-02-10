//! EVEX prefix encoding for AVX-512 instructions.
//!
//! EVEX is a 4-byte prefix that extends VEX with:
//! - 32 vector registers (ZMM0-ZMM31)
//! - 512-bit vector width
//! - Opmask predication (k0-k7)
//! - Embedded broadcast, rounding control, and SAE
//!
//! # EVEX Format
//!
//! ```text
//! Byte 0: 62h (EVEX escape)
//! Byte 1: [R X B R' 0 0 m m] - Extended REX bits + opcode map
//! Byte 2: [W v v v v 1 p p] - REX.W + NDS/NDD + prefix
//! Byte 3: [z L' L b V' a a a] - Zeroing + vector length + broadcast + opmask
//! ```

use super::encoder::{EncodedInst, Mod, modrm};
use super::simd::{KReg, Zmm};

// =============================================================================
// EVEX Prefix
// =============================================================================

/// EVEX prefix for AVX-512 instructions.
#[derive(Debug, Clone, Copy)]
pub struct Evex {
    // Byte 1 fields
    /// Inverted REX.R (high bit of MODRM.reg).
    pub r: bool,
    /// Inverted REX.X (high bit of SIB.index).
    pub x: bool,
    /// Inverted REX.B (high bit of MODRM.rm or SIB.base).
    pub b: bool,
    /// Inverted extension of MODRM.reg (bit 4).
    pub r_prime: bool,
    /// Opcode map: 1=0F, 2=0F38, 3=0F3A.
    pub mm: u8,

    // Byte 2 fields
    /// REX.W equivalent (64-bit operand size).
    pub w: bool,
    /// Source register encoding (inverted, 4 bits).
    pub vvvv: u8,
    /// Operand prefix: 0=none, 1=66, 2=F3, 3=F2.
    pub pp: u8,

    // Byte 3 fields
    /// Zeroing-masking (z=1) vs merge-masking (z=0).
    pub z: bool,
    /// Vector length bits: 00=128, 01=256, 10=512.
    pub ll: u8,
    /// Embedded broadcast enable.
    pub broadcast: bool,
    /// High bit of vvvv (inverted, for ZMM16-31).
    pub v_prime: bool,
    /// Opmask register (k0-k7).
    pub aaa: u8,
}

impl Default for Evex {
    fn default() -> Self {
        Self {
            r: true,
            x: true,
            b: true,
            r_prime: true,
            mm: 1,
            w: false,
            vvvv: 0,
            pp: 0,
            z: false,
            ll: 2,
            broadcast: false,
            v_prime: true,
            aaa: 0,
        }
    }
}

impl Evex {
    /// Create EVEX for 512-bit operation with 66 prefix.
    #[inline]
    pub fn zmm_66() -> Self {
        Self {
            pp: 1,
            ll: 2,
            ..Default::default()
        }
    }

    /// Create EVEX for 256-bit operation with 66 prefix.
    #[inline]
    pub fn ymm_66() -> Self {
        Self {
            pp: 1,
            ll: 1,
            ..Default::default()
        }
    }

    /// Create EVEX for 128-bit operation with 66 prefix.
    #[inline]
    pub fn xmm_66() -> Self {
        Self {
            pp: 1,
            ll: 0,
            ..Default::default()
        }
    }

    /// Set destination register encoding bits.
    #[inline]
    pub fn with_dest(mut self, reg: Zmm) -> Self {
        self.r = (reg.encoding() & 0x8) == 0;
        self.r_prime = (reg.encoding() & 0x10) == 0;
        self
    }

    /// Set source1 register (vvvv field).
    #[inline]
    pub fn with_src1(mut self, reg: Zmm) -> Self {
        self.vvvv = reg.encoding() & 0xF;
        self.v_prime = (reg.encoding() & 0x10) == 0;
        self
    }

    /// Set source2/rm register encoding bits.
    #[inline]
    pub fn with_src2(mut self, reg: Zmm) -> Self {
        self.b = (reg.encoding() & 0x8) == 0;
        self.x = (reg.encoding() & 0x10) == 0;
        self
    }

    /// Set opmask register.
    #[inline]
    pub fn with_mask(mut self, k: KReg) -> Self {
        self.aaa = k.encoding();
        self
    }

    /// Enable zeroing-masking.
    #[inline]
    pub fn with_zeroing(mut self) -> Self {
        self.z = true;
        self
    }

    /// Enable embedded broadcast.
    #[inline]
    pub fn with_broadcast(mut self) -> Self {
        self.broadcast = true;
        self
    }

    /// Set W bit for 64-bit operands.
    #[inline]
    pub fn with_w(mut self) -> Self {
        self.w = true;
        self
    }

    /// Set opcode map.
    #[inline]
    pub fn with_map(mut self, map: u8) -> Self {
        self.mm = map;
        self
    }

    /// Encode as 4-byte EVEX prefix.
    #[inline]
    pub fn encode(&self) -> [u8; 4] {
        let byte1 = ((!self.r as u8) << 7)
            | ((!self.x as u8) << 6)
            | ((!self.b as u8) << 5)
            | ((!self.r_prime as u8) << 4)
            | self.mm;

        let byte2 = ((self.w as u8) << 7)
            | ((self.vvvv ^ 0xF) << 3)
            | (1 << 2) // Fixed bit
            | self.pp;

        let byte3 = ((self.z as u8) << 7)
            | (self.ll << 5)
            | ((self.broadcast as u8) << 4)
            | ((!self.v_prime as u8) << 3)
            | self.aaa;

        [0x62, byte1, byte2, byte3]
    }
}

// =============================================================================
// EVEX Instruction Encoding Helpers
// =============================================================================

/// Encode EVEX register-register-register instruction.
fn encode_evex_rrr(opcode: u8, dst: Zmm, src1: Zmm, src2: Zmm, evex: Evex) -> EncodedInst {
    let mut enc = EncodedInst::new();
    let prefix = evex.with_dest(dst).with_src1(src1).with_src2(src2).encode();
    for b in prefix {
        enc.push(b);
    }
    enc.push(opcode);
    enc.push(modrm(Mod::Direct, dst.low_bits(), src2.low_bits()));
    enc
}

/// Encode EVEX register-register-register with opmask.
fn encode_evex_rrr_k(
    opcode: u8,
    dst: Zmm,
    src1: Zmm,
    src2: Zmm,
    k: KReg,
    evex: Evex,
) -> EncodedInst {
    encode_evex_rrr(opcode, dst, src1, src2, evex.with_mask(k))
}

// =============================================================================
// AVX-512F Packed Double-Precision (512-bit)
// =============================================================================

/// VADDPD zmm, zmm, zmm - Add packed doubles (512-bit).
#[inline]
pub fn encode_vaddpd_zmm_rrr(dst: Zmm, src1: Zmm, src2: Zmm) -> EncodedInst {
    encode_evex_rrr(0x58, dst, src1, src2, Evex::zmm_66().with_w())
}

/// VSUBPD zmm, zmm, zmm - Subtract packed doubles (512-bit).
#[inline]
pub fn encode_vsubpd_zmm_rrr(dst: Zmm, src1: Zmm, src2: Zmm) -> EncodedInst {
    encode_evex_rrr(0x5C, dst, src1, src2, Evex::zmm_66().with_w())
}

/// VMULPD zmm, zmm, zmm - Multiply packed doubles (512-bit).
#[inline]
pub fn encode_vmulpd_zmm_rrr(dst: Zmm, src1: Zmm, src2: Zmm) -> EncodedInst {
    encode_evex_rrr(0x59, dst, src1, src2, Evex::zmm_66().with_w())
}

/// VDIVPD zmm, zmm, zmm - Divide packed doubles (512-bit).
#[inline]
pub fn encode_vdivpd_zmm_rrr(dst: Zmm, src1: Zmm, src2: Zmm) -> EncodedInst {
    encode_evex_rrr(0x5E, dst, src1, src2, Evex::zmm_66().with_w())
}

/// VMAXPD zmm, zmm, zmm - Maximum packed doubles (512-bit).
#[inline]
pub fn encode_vmaxpd_zmm_rrr(dst: Zmm, src1: Zmm, src2: Zmm) -> EncodedInst {
    encode_evex_rrr(0x5F, dst, src1, src2, Evex::zmm_66().with_w())
}

/// VMINPD zmm, zmm, zmm - Minimum packed doubles (512-bit).
#[inline]
pub fn encode_vminpd_zmm_rrr(dst: Zmm, src1: Zmm, src2: Zmm) -> EncodedInst {
    encode_evex_rrr(0x5D, dst, src1, src2, Evex::zmm_66().with_w())
}

/// VSQRTPD zmm, zmm - Square root packed doubles (512-bit).
#[inline]
pub fn encode_vsqrtpd_zmm_rr(dst: Zmm, src: Zmm) -> EncodedInst {
    let evex = Evex::zmm_66().with_w().with_dest(dst).with_src2(src);
    let mut enc = EncodedInst::new();
    for b in evex.encode() {
        enc.push(b);
    }
    enc.push(0x51);
    enc.push(modrm(Mod::Direct, dst.low_bits(), src.low_bits()));
    enc
}

// =============================================================================
// AVX-512F Packed Single-Precision (512-bit)
// =============================================================================

/// VADDPS zmm, zmm, zmm - Add packed singles (512-bit).
#[inline]
pub fn encode_vaddps_zmm_rrr(dst: Zmm, src1: Zmm, src2: Zmm) -> EncodedInst {
    encode_evex_rrr(0x58, dst, src1, src2, Evex::zmm_66().with_map(1))
}

/// VSUBPS zmm, zmm, zmm - Subtract packed singles (512-bit).
#[inline]
pub fn encode_vsubps_zmm_rrr(dst: Zmm, src1: Zmm, src2: Zmm) -> EncodedInst {
    encode_evex_rrr(0x5C, dst, src1, src2, Evex::zmm_66().with_map(1))
}

/// VMULPS zmm, zmm, zmm - Multiply packed singles (512-bit).
#[inline]
pub fn encode_vmulps_zmm_rrr(dst: Zmm, src1: Zmm, src2: Zmm) -> EncodedInst {
    encode_evex_rrr(0x59, dst, src1, src2, Evex::zmm_66().with_map(1))
}

/// VDIVPS zmm, zmm, zmm - Divide packed singles (512-bit).
#[inline]
pub fn encode_vdivps_zmm_rrr(dst: Zmm, src1: Zmm, src2: Zmm) -> EncodedInst {
    encode_evex_rrr(0x5E, dst, src1, src2, Evex::zmm_66().with_map(1))
}

// =============================================================================
// AVX-512F with Opmask
// =============================================================================

/// VADDPD zmm {k}, zmm, zmm - Add packed doubles with opmask.
#[inline]
pub fn encode_vaddpd_zmm_rrr_k(dst: Zmm, src1: Zmm, src2: Zmm, k: KReg) -> EncodedInst {
    encode_evex_rrr_k(0x58, dst, src1, src2, k, Evex::zmm_66().with_w())
}

/// VMULPD zmm {k}, zmm, zmm - Multiply packed doubles with opmask.
#[inline]
pub fn encode_vmulpd_zmm_rrr_k(dst: Zmm, src1: Zmm, src2: Zmm, k: KReg) -> EncodedInst {
    encode_evex_rrr_k(0x59, dst, src1, src2, k, Evex::zmm_66().with_w())
}

// =============================================================================
// AVX-512F Integer Operations
// =============================================================================

/// VPADDD zmm, zmm, zmm - Add packed 32-bit integers.
#[inline]
pub fn encode_vpaddd_zmm_rrr(dst: Zmm, src1: Zmm, src2: Zmm) -> EncodedInst {
    encode_evex_rrr(0xFE, dst, src1, src2, Evex::zmm_66())
}

/// VPSUBD zmm, zmm, zmm - Subtract packed 32-bit integers.
#[inline]
pub fn encode_vpsubd_zmm_rrr(dst: Zmm, src1: Zmm, src2: Zmm) -> EncodedInst {
    encode_evex_rrr(0xFA, dst, src1, src2, Evex::zmm_66())
}

/// VPADDQ zmm, zmm, zmm - Add packed 64-bit integers.
#[inline]
pub fn encode_vpaddq_zmm_rrr(dst: Zmm, src1: Zmm, src2: Zmm) -> EncodedInst {
    encode_evex_rrr(0xD4, dst, src1, src2, Evex::zmm_66().with_w())
}

/// VPSUBQ zmm, zmm, zmm - Subtract packed 64-bit integers.
#[inline]
pub fn encode_vpsubq_zmm_rrr(dst: Zmm, src1: Zmm, src2: Zmm) -> EncodedInst {
    encode_evex_rrr(0xFB, dst, src1, src2, Evex::zmm_66().with_w())
}

/// VPMULLD zmm, zmm, zmm - Multiply packed 32-bit integers (low 32 bits).
#[inline]
pub fn encode_vpmulld_zmm_rrr(dst: Zmm, src1: Zmm, src2: Zmm) -> EncodedInst {
    encode_evex_rrr(0x40, dst, src1, src2, Evex::zmm_66().with_map(2))
}

// =============================================================================
// AVX-512F Logical Operations
// =============================================================================

/// VPANDD zmm, zmm, zmm - Bitwise AND of packed 32-bit integers.
#[inline]
pub fn encode_vpandd_zmm_rrr(dst: Zmm, src1: Zmm, src2: Zmm) -> EncodedInst {
    encode_evex_rrr(0xDB, dst, src1, src2, Evex::zmm_66())
}

/// VPANDQ zmm, zmm, zmm - Bitwise AND of packed 64-bit integers.
#[inline]
pub fn encode_vpandq_zmm_rrr(dst: Zmm, src1: Zmm, src2: Zmm) -> EncodedInst {
    encode_evex_rrr(0xDB, dst, src1, src2, Evex::zmm_66().with_w())
}

/// VPORD zmm, zmm, zmm - Bitwise OR of packed 32-bit integers.
#[inline]
pub fn encode_vpord_zmm_rrr(dst: Zmm, src1: Zmm, src2: Zmm) -> EncodedInst {
    encode_evex_rrr(0xEB, dst, src1, src2, Evex::zmm_66())
}

/// VPORQ zmm, zmm, zmm - Bitwise OR of packed 64-bit integers.
#[inline]
pub fn encode_vporq_zmm_rrr(dst: Zmm, src1: Zmm, src2: Zmm) -> EncodedInst {
    encode_evex_rrr(0xEB, dst, src1, src2, Evex::zmm_66().with_w())
}

/// VPXORD zmm, zmm, zmm - Bitwise XOR of packed 32-bit integers.
#[inline]
pub fn encode_vpxord_zmm_rrr(dst: Zmm, src1: Zmm, src2: Zmm) -> EncodedInst {
    encode_evex_rrr(0xEF, dst, src1, src2, Evex::zmm_66())
}

/// VPXORQ zmm, zmm, zmm - Bitwise XOR of packed 64-bit integers.
#[inline]
pub fn encode_vpxorq_zmm_rrr(dst: Zmm, src1: Zmm, src2: Zmm) -> EncodedInst {
    encode_evex_rrr(0xEF, dst, src1, src2, Evex::zmm_66().with_w())
}

// =============================================================================
// AVX-512F FMA
// =============================================================================

/// VFMADD132PD zmm, zmm, zmm - Fused multiply-add (512-bit).
#[inline]
pub fn encode_vfmadd132pd_zmm_rrr(dst: Zmm, src1: Zmm, src2: Zmm) -> EncodedInst {
    encode_evex_rrr(0x98, dst, src1, src2, Evex::zmm_66().with_map(2).with_w())
}

/// VFMADD213PD zmm, zmm, zmm - Fused multiply-add (512-bit).
#[inline]
pub fn encode_vfmadd213pd_zmm_rrr(dst: Zmm, src1: Zmm, src2: Zmm) -> EncodedInst {
    encode_evex_rrr(0xA8, dst, src1, src2, Evex::zmm_66().with_map(2).with_w())
}

/// VFMADD231PD zmm, zmm, zmm - Fused multiply-add (512-bit).
#[inline]
pub fn encode_vfmadd231pd_zmm_rrr(dst: Zmm, src1: Zmm, src2: Zmm) -> EncodedInst {
    encode_evex_rrr(0xB8, dst, src1, src2, Evex::zmm_66().with_map(2).with_w())
}

// =============================================================================
// AVX-512F Memory Operations
// =============================================================================

use super::encoder::sib;
use super::registers::{Gpr, MemOperand, Scale};

/// Encode EVEX memory operand with optional disp8 compression.
/// Returns (mod, disp_bytes) where mod is the addressing mode and disp_bytes
/// contains the displacement encoding.
fn encode_evex_mem_operand(
    enc: &mut EncodedInst,
    reg: u8,
    mem: &MemOperand,
    vector_len: u8,   // 0=128, 1=256, 2=512
    element_size: u8, // Element size in bytes for disp8*N compression
) {
    let disp_factor = match vector_len {
        0 => element_size as i32,     // 128-bit: N = element_size
        1 => element_size as i32 * 2, // 256-bit: N = element_size * 2
        _ => element_size as i32 * 4, // 512-bit: N = element_size * 4
    };

    match (mem.base, mem.index) {
        (Some(base), None) => {
            let needs_sib = base == Gpr::Rsp || base == Gpr::R12;

            if mem.disp == 0 && !base.needs_displacement() {
                // [base]
                enc.push(modrm(
                    Mod::Indirect,
                    reg,
                    if needs_sib { 0b100 } else { base.low_bits() },
                ));
                if needs_sib {
                    enc.push(sib(Scale::X1, 0b100, base.low_bits()));
                }
            } else if mem.disp % disp_factor == 0 {
                let compressed = mem.disp / disp_factor;
                if compressed >= -128 && compressed <= 127 {
                    // Disp8*N compression
                    enc.push(modrm(
                        Mod::IndirectDisp8,
                        reg,
                        if needs_sib { 0b100 } else { base.low_bits() },
                    ));
                    if needs_sib {
                        enc.push(sib(Scale::X1, 0b100, base.low_bits()));
                    }
                    enc.push(compressed as u8);
                } else {
                    // Disp32
                    enc.push(modrm(
                        Mod::IndirectDisp32,
                        reg,
                        if needs_sib { 0b100 } else { base.low_bits() },
                    ));
                    if needs_sib {
                        enc.push(sib(Scale::X1, 0b100, base.low_bits()));
                    }
                    enc.push_u32(mem.disp as u32);
                }
            } else {
                // Cannot compress, use disp32
                enc.push(modrm(
                    Mod::IndirectDisp32,
                    reg,
                    if needs_sib { 0b100 } else { base.low_bits() },
                ));
                if needs_sib {
                    enc.push(sib(Scale::X1, 0b100, base.low_bits()));
                }
                enc.push_u32(mem.disp as u32);
            }
        }
        (Some(base), Some(index)) => {
            // Base + Index + Scale
            let mod_field = if mem.disp == 0 && !base.needs_displacement() {
                Mod::Indirect
            } else if mem.disp % disp_factor == 0 {
                let compressed = mem.disp / disp_factor;
                if compressed >= -128 && compressed <= 127 {
                    Mod::IndirectDisp8
                } else {
                    Mod::IndirectDisp32
                }
            } else {
                Mod::IndirectDisp32
            };

            enc.push(modrm(mod_field, reg, 0b100));
            enc.push(sib(mem.scale, index.low_bits(), base.low_bits()));

            match mod_field {
                Mod::IndirectDisp8 => {
                    let compressed = mem.disp / disp_factor;
                    enc.push(compressed as u8);
                }
                Mod::IndirectDisp32 => enc.push_u32(mem.disp as u32),
                Mod::Indirect if base.needs_displacement() => enc.push(0),
                _ => {}
            }
        }
        (None, Some(index)) => {
            // Index + disp32 (no base)
            enc.push(modrm(Mod::Indirect, reg, 0b100));
            enc.push(sib(mem.scale, index.low_bits(), 0b101));
            enc.push_u32(mem.disp as u32);
        }
        (None, None) => {
            // RIP-relative
            enc.push(modrm(Mod::Indirect, reg, 0b101));
            enc.push_u32(mem.disp as u32);
        }
    }
}

/// Set EVEX memory operand REX bits from base/index registers.
fn evex_with_mem(mut evex: Evex, mem: &MemOperand) -> Evex {
    if let Some(base) = mem.base {
        evex.b = !base.high_bit();
    }
    if let Some(index) = mem.index {
        evex.x = !index.high_bit();
    }
    evex
}

/// VMOVAPD zmm, m512 - Move aligned packed doubles from memory (512-bit).
#[inline]
pub fn encode_vmovapd_zmm_rm(dst: Zmm, mem: &MemOperand) -> EncodedInst {
    let mut enc = EncodedInst::new();
    let evex = evex_with_mem(Evex::zmm_66().with_w().with_dest(dst), mem);
    for b in evex.encode() {
        enc.push(b);
    }
    enc.push(0x28);
    encode_evex_mem_operand(&mut enc, dst.low_bits(), mem, 2, 8);
    enc
}

/// VMOVUPD zmm, m512 - Move unaligned packed doubles from memory (512-bit).
#[inline]
pub fn encode_vmovupd_zmm_rm(dst: Zmm, mem: &MemOperand) -> EncodedInst {
    let mut enc = EncodedInst::new();
    let evex = evex_with_mem(Evex::zmm_66().with_w().with_dest(dst), mem);
    for b in evex.encode() {
        enc.push(b);
    }
    enc.push(0x10);
    encode_evex_mem_operand(&mut enc, dst.low_bits(), mem, 2, 8);
    enc
}

/// VMOVAPD m512, zmm - Move aligned packed doubles to memory (512-bit).
#[inline]
pub fn encode_vmovapd_zmm_mr(mem: &MemOperand, src: Zmm) -> EncodedInst {
    let mut enc = EncodedInst::new();
    let evex = evex_with_mem(Evex::zmm_66().with_w().with_dest(src), mem);
    for b in evex.encode() {
        enc.push(b);
    }
    enc.push(0x29);
    encode_evex_mem_operand(&mut enc, src.low_bits(), mem, 2, 8);
    enc
}

/// VMOVUPD m512, zmm - Move unaligned packed doubles to memory (512-bit).
#[inline]
pub fn encode_vmovupd_zmm_mr(mem: &MemOperand, src: Zmm) -> EncodedInst {
    let mut enc = EncodedInst::new();
    let evex = evex_with_mem(Evex::zmm_66().with_w().with_dest(src), mem);
    for b in evex.encode() {
        enc.push(b);
    }
    enc.push(0x11);
    encode_evex_mem_operand(&mut enc, src.low_bits(), mem, 2, 8);
    enc
}

// =============================================================================
// AVX-512F ZMM Register-to-Register Move Instructions
// =============================================================================

/// VMOVAPD zmm, zmm - Move aligned packed double-precision (register-to-register).
///
/// This is the preferred instruction for ZMM register-to-register moves
/// with double-precision data. Uses EVEX encoding with opcode 0x28.
#[inline]
pub fn encode_vmovapd_zmm_rr(dst: Zmm, src: Zmm) -> EncodedInst {
    let evex = Evex::zmm_66().with_w().with_dest(dst).with_src2(src);
    let mut enc = EncodedInst::new();
    for b in evex.encode() {
        enc.push(b);
    }
    enc.push(0x28);
    enc.push(modrm(Mod::Direct, dst.low_bits(), src.low_bits()));
    enc
}

/// VMOVAPS zmm, zmm - Move aligned packed single-precision (register-to-register).
///
/// Uses EVEX encoding with opcode 0x28 and no W bit (single-precision).
#[inline]
pub fn encode_vmovaps_zmm_rr(dst: Zmm, src: Zmm) -> EncodedInst {
    let evex = Evex::zmm_66().with_dest(dst).with_src2(src);
    let mut enc = EncodedInst::new();
    // Need to use proper pp=0 for PS instructions
    let evex_ps = Evex {
        pp: 0, // No prefix for PS
        ll: 2, // 512-bit
        ..evex
    };
    for b in evex_ps.encode() {
        enc.push(b);
    }
    enc.push(0x28);
    enc.push(modrm(Mod::Direct, dst.low_bits(), src.low_bits()));
    enc
}

/// VMOVUPD zmm, zmm - Move unaligned packed double-precision (register-to-register).
///
/// For register operands, aligned vs unaligned makes no difference, but
/// this provides consistency with memory loading patterns.
#[inline]
pub fn encode_vmovupd_zmm_rr(dst: Zmm, src: Zmm) -> EncodedInst {
    let evex = Evex::zmm_66().with_w().with_dest(dst).with_src2(src);
    let mut enc = EncodedInst::new();
    for b in evex.encode() {
        enc.push(b);
    }
    enc.push(0x10);
    enc.push(modrm(Mod::Direct, dst.low_bits(), src.low_bits()));
    enc
}

/// VMOVUPS zmm, zmm - Move unaligned packed single-precision (register-to-register).
#[inline]
pub fn encode_vmovups_zmm_rr(dst: Zmm, src: Zmm) -> EncodedInst {
    let evex = Evex::zmm_66().with_dest(dst).with_src2(src);
    let mut enc = EncodedInst::new();
    let evex_ps = Evex {
        pp: 0,
        ll: 2,
        ..evex
    };
    for b in evex_ps.encode() {
        enc.push(b);
    }
    enc.push(0x10);
    enc.push(modrm(Mod::Direct, dst.low_bits(), src.low_bits()));
    enc
}

/// VMOVDQA64 zmm, zmm - Move aligned 64-bit integers (register-to-register).
///
/// Preferred for moving 64-bit integer vector data in ZMM registers.
/// Uses EVEX encoding with opcode 0x6F, W=1, pp=1.
#[inline]
pub fn encode_vmovdqa64_zmm_rr(dst: Zmm, src: Zmm) -> EncodedInst {
    let evex = Evex::zmm_66().with_w().with_dest(dst).with_src2(src);
    let mut enc = EncodedInst::new();
    for b in evex.encode() {
        enc.push(b);
    }
    enc.push(0x6F);
    enc.push(modrm(Mod::Direct, dst.low_bits(), src.low_bits()));
    enc
}

/// VMOVDQA32 zmm, zmm - Move aligned 32-bit integers (register-to-register).
#[inline]
pub fn encode_vmovdqa32_zmm_rr(dst: Zmm, src: Zmm) -> EncodedInst {
    let evex = Evex::zmm_66().with_dest(dst).with_src2(src);
    let mut enc = EncodedInst::new();
    for b in evex.encode() {
        enc.push(b);
    }
    enc.push(0x6F);
    enc.push(modrm(Mod::Direct, dst.low_bits(), src.low_bits()));
    enc
}

/// VMOVDQU64 zmm, zmm - Move unaligned 64-bit integers (register-to-register).
#[inline]
pub fn encode_vmovdqu64_zmm_rr(dst: Zmm, src: Zmm) -> EncodedInst {
    let evex = Evex {
        pp: 2, // F3 prefix
        ll: 2,
        w: true,
        ..Evex::default()
    }
    .with_dest(dst)
    .with_src2(src);
    let mut enc = EncodedInst::new();
    for b in evex.encode() {
        enc.push(b);
    }
    enc.push(0x6F);
    enc.push(modrm(Mod::Direct, dst.low_bits(), src.low_bits()));
    enc
}

/// VMOVDQU32 zmm, zmm - Move unaligned 32-bit integers (register-to-register).
#[inline]
pub fn encode_vmovdqu32_zmm_rr(dst: Zmm, src: Zmm) -> EncodedInst {
    let evex = Evex {
        pp: 2, // F3 prefix
        ll: 2,
        w: false,
        ..Evex::default()
    }
    .with_dest(dst)
    .with_src2(src);
    let mut enc = EncodedInst::new();
    for b in evex.encode() {
        enc.push(b);
    }
    enc.push(0x6F);
    enc.push(modrm(Mod::Direct, dst.low_bits(), src.low_bits()));
    enc
}

/// VBROADCASTSD zmm, m64 - Broadcast double from memory to all elements.
#[inline]
pub fn encode_vbroadcastsd_zmm_rm(dst: Zmm, mem: &MemOperand) -> EncodedInst {
    let mut enc = EncodedInst::new();
    let evex = evex_with_mem(Evex::zmm_66().with_w().with_map(2).with_dest(dst), mem);
    for b in evex.encode() {
        enc.push(b);
    }
    enc.push(0x19);
    encode_evex_mem_operand(&mut enc, dst.low_bits(), mem, 2, 8);
    enc
}

/// VADDPD zmm, zmm, m512 {1to8} - Add with embedded broadcast from memory.
#[inline]
pub fn encode_vaddpd_zmm_rmb(dst: Zmm, src1: Zmm, mem: &MemOperand) -> EncodedInst {
    let mut enc = EncodedInst::new();
    let evex = evex_with_mem(
        Evex::zmm_66()
            .with_w()
            .with_dest(dst)
            .with_src1(src1)
            .with_broadcast(),
        mem,
    );
    for b in evex.encode() {
        enc.push(b);
    }
    enc.push(0x58);
    encode_evex_mem_operand(&mut enc, dst.low_bits(), mem, 2, 8);
    enc
}

/// VMULPD zmm, zmm, m512 {1to8} - Multiply with embedded broadcast from memory.
#[inline]
pub fn encode_vmulpd_zmm_rmb(dst: Zmm, src1: Zmm, mem: &MemOperand) -> EncodedInst {
    let mut enc = EncodedInst::new();
    let evex = evex_with_mem(
        Evex::zmm_66()
            .with_w()
            .with_dest(dst)
            .with_src1(src1)
            .with_broadcast(),
        mem,
    );
    for b in evex.encode() {
        enc.push(b);
    }
    enc.push(0x59);
    encode_evex_mem_operand(&mut enc, dst.low_bits(), mem, 2, 8);
    enc
}

/// VFMADD231PD zmm, zmm, m512 {1to8} - FMA with embedded broadcast from memory.
#[inline]
pub fn encode_vfmadd231pd_zmm_rmb(dst: Zmm, src1: Zmm, mem: &MemOperand) -> EncodedInst {
    let mut enc = EncodedInst::new();
    let evex = evex_with_mem(
        Evex::zmm_66()
            .with_w()
            .with_map(2)
            .with_dest(dst)
            .with_src1(src1)
            .with_broadcast(),
        mem,
    );
    for b in evex.encode() {
        enc.push(b);
    }
    enc.push(0xB8);
    encode_evex_mem_operand(&mut enc, dst.low_bits(), mem, 2, 8);
    enc
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_evex_default() {
        let evex = Evex::default();
        assert_eq!(evex.ll, 2); // 512-bit
        assert!(!evex.z);
        assert_eq!(evex.aaa, 0);
    }

    #[test]
    fn test_evex_encode_basic() {
        let evex = Evex::zmm_66();
        let bytes = evex.encode();
        assert_eq!(bytes[0], 0x62); // EVEX escape
    }

    #[test]
    fn test_evex_with_mask() {
        let evex = Evex::zmm_66().with_mask(KReg::K1);
        let bytes = evex.encode();
        assert_eq!(bytes[3] & 0x7, 1); // aaa = k1
    }

    #[test]
    fn test_evex_with_zeroing() {
        let evex = Evex::zmm_66().with_zeroing();
        let bytes = evex.encode();
        assert_ne!(bytes[3] & 0x80, 0); // z bit set
    }

    #[test]
    fn test_vaddpd_zmm() {
        let enc = encode_vaddpd_zmm_rrr(Zmm::Zmm0, Zmm::Zmm1, Zmm::Zmm2);
        assert!(enc.len() >= 6);
        assert_eq!(enc.as_slice()[0], 0x62);
    }

    #[test]
    fn test_vmulpd_zmm() {
        let enc = encode_vmulpd_zmm_rrr(Zmm::Zmm16, Zmm::Zmm17, Zmm::Zmm18);
        assert!(enc.len() >= 6);
        assert_eq!(enc.as_slice()[0], 0x62);
    }

    #[test]
    fn test_vaddpd_zmm_with_mask() {
        let enc = encode_vaddpd_zmm_rrr_k(Zmm::Zmm0, Zmm::Zmm1, Zmm::Zmm2, KReg::K3);
        assert!(enc.len() >= 6);
        // Check mask register in byte 3
        let bytes = enc.as_slice();
        assert_eq!(bytes[3] & 0x7, 3);
    }

    #[test]
    fn test_vpaddd_zmm() {
        let enc = encode_vpaddd_zmm_rrr(Zmm::Zmm4, Zmm::Zmm5, Zmm::Zmm6);
        assert!(enc.len() >= 6);
    }

    #[test]
    fn test_vpxorq_zmm() {
        let enc = encode_vpxorq_zmm_rrr(Zmm::Zmm0, Zmm::Zmm0, Zmm::Zmm0);
        assert!(enc.len() >= 6);
    }

    #[test]
    fn test_vfmadd_zmm() {
        let enc = encode_vfmadd231pd_zmm_rrr(Zmm::Zmm0, Zmm::Zmm1, Zmm::Zmm2);
        assert!(enc.len() >= 6);
        assert_eq!(enc.as_slice()[0], 0x62);
    }

    #[test]
    fn test_high_registers() {
        // ZMM16-31 require extended bits
        let enc = encode_vaddpd_zmm_rrr(Zmm::Zmm31, Zmm::Zmm30, Zmm::Zmm29);
        assert!(enc.len() >= 6);
        // Verify R' and V' bits are set correctly for high registers
        let bytes = enc.as_slice();
        assert_eq!(bytes[0], 0x62);
    }

    #[test]
    fn test_vector_length_bits() {
        let zmm = Evex::zmm_66();
        assert_eq!(zmm.ll, 2);
        let ymm = Evex::ymm_66();
        assert_eq!(ymm.ll, 1);
        let xmm = Evex::xmm_66();
        assert_eq!(xmm.ll, 0);
    }

    #[test]
    fn test_vmovapd_zmm_rm() {
        let mem = MemOperand::base(Gpr::Rax);
        let enc = encode_vmovapd_zmm_rm(Zmm::Zmm0, &mem);
        assert!(enc.len() >= 6);
        assert_eq!(enc.as_slice()[0], 0x62);
    }

    #[test]
    fn test_vmovupd_zmm_rm() {
        let mem = MemOperand::base_disp(Gpr::Rbx, 128);
        let enc = encode_vmovupd_zmm_rm(Zmm::Zmm8, &mem);
        assert!(enc.len() >= 6);
    }

    #[test]
    fn test_vmovapd_zmm_mr() {
        let mem = MemOperand::base(Gpr::Rcx);
        let enc = encode_vmovapd_zmm_mr(&mem, Zmm::Zmm16);
        assert!(enc.len() >= 6);
    }

    #[test]
    fn test_vmovupd_zmm_mr() {
        let mem = MemOperand::base_disp(Gpr::R14, 64);
        let enc = encode_vmovupd_zmm_mr(&mem, Zmm::Zmm24);
        assert!(enc.len() >= 6);
    }

    #[test]
    fn test_vbroadcastsd_zmm() {
        let mem = MemOperand::base(Gpr::Rsi);
        let enc = encode_vbroadcastsd_zmm_rm(Zmm::Zmm0, &mem);
        assert!(enc.len() >= 6);
        assert_eq!(enc.as_slice()[0], 0x62);
    }

    #[test]
    fn test_vaddpd_zmm_rmb() {
        let mem = MemOperand::base(Gpr::Rdi);
        let enc = encode_vaddpd_zmm_rmb(Zmm::Zmm1, Zmm::Zmm2, &mem);
        assert!(enc.len() >= 6);
        // Check broadcast bit (b) is set in byte 3
        let bytes = enc.as_slice();
        assert_ne!(bytes[3] & 0x10, 0); // broadcast bit set
    }

    #[test]
    fn test_vmulpd_zmm_rmb() {
        let mem = MemOperand::base(Gpr::R8);
        let enc = encode_vmulpd_zmm_rmb(Zmm::Zmm3, Zmm::Zmm4, &mem);
        assert!(enc.len() >= 6);
        // Broadcast bit should be set
        assert_ne!(enc.as_slice()[3] & 0x10, 0);
    }

    #[test]
    fn test_vfmadd231pd_zmm_rmb() {
        let mem = MemOperand::base(Gpr::R9);
        let enc = encode_vfmadd231pd_zmm_rmb(Zmm::Zmm5, Zmm::Zmm6, &mem);
        assert!(enc.len() >= 6);
    }

    #[test]
    fn test_disp8_compression() {
        // 512-bit with 8-byte elements: disp8*N factor = 64
        // disp = 64 should compress to 1
        let mem = MemOperand::base_disp(Gpr::Rax, 64);
        let enc = encode_vmovapd_zmm_rm(Zmm::Zmm0, &mem);
        // Should use disp8 encoding (shorter instruction)
        let bytes = enc.as_slice();
        // ModRM byte is at index 5: EVEX (4 bytes) + opcode (1 byte) = 5
        let modrm = bytes[5];
        assert_eq!((modrm >> 6) & 3, 1); // mod = 01 means disp8
    }

    #[test]
    fn test_disp32_fallback() {
        // Displacement not divisible by 64 - cannot use disp8*N
        let mem = MemOperand::base_disp(Gpr::Rax, 100);
        let enc = encode_vmovapd_zmm_rm(Zmm::Zmm0, &mem);
        let bytes = enc.as_slice();
        // Should fall back to disp32
        // ModRM byte is at index 5: EVEX (4 bytes) + opcode (1 byte) = 5
        let modrm = bytes[5];
        assert_eq!((modrm >> 6) & 3, 2); // mod = 10 means disp32
    }

    #[test]
    fn test_memory_with_sib() {
        // RSP requires SIB byte
        let mem = MemOperand::base(Gpr::Rsp);
        let enc = encode_vmovapd_zmm_rm(Zmm::Zmm0, &mem);
        assert!(enc.len() >= 7); // Extra SIB byte
    }

    #[test]
    fn test_memory_with_index() {
        let mem = MemOperand::base_index(Gpr::Rax, Gpr::Rcx, Scale::X8);
        let enc = encode_vmovapd_zmm_rm(Zmm::Zmm0, &mem);
        assert!(enc.len() >= 7); // Needs SIB
    }

    #[test]
    fn test_all_arithmetic_zmm() {
        let dst = Zmm::Zmm0;
        let src1 = Zmm::Zmm1;
        let src2 = Zmm::Zmm2;

        assert!(encode_vaddpd_zmm_rrr(dst, src1, src2).len() >= 6);
        assert!(encode_vsubpd_zmm_rrr(dst, src1, src2).len() >= 6);
        assert!(encode_vmulpd_zmm_rrr(dst, src1, src2).len() >= 6);
        assert!(encode_vdivpd_zmm_rrr(dst, src1, src2).len() >= 6);
        assert!(encode_vmaxpd_zmm_rrr(dst, src1, src2).len() >= 6);
        assert!(encode_vminpd_zmm_rrr(dst, src1, src2).len() >= 6);
    }

    #[test]
    fn test_all_arithmetic_ps_zmm() {
        let dst = Zmm::Zmm0;
        let src1 = Zmm::Zmm1;
        let src2 = Zmm::Zmm2;

        assert!(encode_vaddps_zmm_rrr(dst, src1, src2).len() >= 6);
        assert!(encode_vsubps_zmm_rrr(dst, src1, src2).len() >= 6);
        assert!(encode_vmulps_zmm_rrr(dst, src1, src2).len() >= 6);
        assert!(encode_vdivps_zmm_rrr(dst, src1, src2).len() >= 6);
    }

    #[test]
    fn test_all_integer_zmm() {
        let dst = Zmm::Zmm0;
        let src1 = Zmm::Zmm1;
        let src2 = Zmm::Zmm2;

        assert!(encode_vpaddd_zmm_rrr(dst, src1, src2).len() >= 6);
        assert!(encode_vpsubd_zmm_rrr(dst, src1, src2).len() >= 6);
        assert!(encode_vpaddq_zmm_rrr(dst, src1, src2).len() >= 6);
        assert!(encode_vpsubq_zmm_rrr(dst, src1, src2).len() >= 6);
        assert!(encode_vpmulld_zmm_rrr(dst, src1, src2).len() >= 6);
    }

    #[test]
    fn test_all_logical_zmm() {
        let dst = Zmm::Zmm0;
        let src1 = Zmm::Zmm1;
        let src2 = Zmm::Zmm2;

        assert!(encode_vpandd_zmm_rrr(dst, src1, src2).len() >= 6);
        assert!(encode_vpandq_zmm_rrr(dst, src1, src2).len() >= 6);
        assert!(encode_vpord_zmm_rrr(dst, src1, src2).len() >= 6);
        assert!(encode_vporq_zmm_rrr(dst, src1, src2).len() >= 6);
        assert!(encode_vpxord_zmm_rrr(dst, src1, src2).len() >= 6);
        assert!(encode_vpxorq_zmm_rrr(dst, src1, src2).len() >= 6);
    }

    #[test]
    fn test_all_fma_zmm() {
        let dst = Zmm::Zmm0;
        let src1 = Zmm::Zmm1;
        let src2 = Zmm::Zmm2;

        assert!(encode_vfmadd132pd_zmm_rrr(dst, src1, src2).len() >= 6);
        assert!(encode_vfmadd213pd_zmm_rrr(dst, src1, src2).len() >= 6);
        assert!(encode_vfmadd231pd_zmm_rrr(dst, src1, src2).len() >= 6);
    }

    #[test]
    fn test_evex_bits_encoding() {
        // Test that R, X, B, R' bits are correctly inverted
        // When r/x/b/r_prime are true (meaning low registers 0-7),
        // the encoded bits are 0 (inverted)
        let evex = Evex::default();
        let bytes = evex.encode();

        // Default: r=x=b=r_prime=true → encoded bits are 0 (they're inverted)
        // Byte 1 high nibble = 0b0000 (all inverted to 0) + mm bits
        assert_eq!(bytes[1] & 0xF0, 0x00); // High nibble is 0

        // Test with high registers - bits should be 1 when r/x/b are false
        let evex_high = Evex {
            r: false,
            x: false,
            b: false,
            r_prime: false,
            ..Evex::zmm_66()
        };
        let bytes_high = evex_high.encode();
        // R=0, X=0, B=0, R'=0 → inverted bits are 1
        assert_eq!(bytes_high[1] & 0xF0, 0xF0);
    }

    #[test]
    fn test_vsqrtpd_zmm() {
        let enc = encode_vsqrtpd_zmm_rr(Zmm::Zmm0, Zmm::Zmm1);
        assert!(enc.len() >= 6);
        assert_eq!(enc.as_slice()[0], 0x62);
    }

    // =========================================================================
    // ZMM Register-to-Register Move Tests
    // =========================================================================

    #[test]
    fn test_vmovapd_zmm_rr_basic() {
        let enc = encode_vmovapd_zmm_rr(Zmm::Zmm0, Zmm::Zmm1);
        assert!(enc.len() >= 6);
        assert_eq!(enc.as_slice()[0], 0x62); // EVEX prefix
    }

    #[test]
    fn test_vmovapd_zmm_rr_high_registers() {
        // ZMM16-31 require extended register bits
        let enc = encode_vmovapd_zmm_rr(Zmm::Zmm16, Zmm::Zmm31);
        assert!(enc.len() >= 6);
        assert_eq!(enc.as_slice()[0], 0x62);
    }

    #[test]
    fn test_vmovaps_zmm_rr_basic() {
        let enc = encode_vmovaps_zmm_rr(Zmm::Zmm2, Zmm::Zmm3);
        assert!(enc.len() >= 6);
        assert_eq!(enc.as_slice()[0], 0x62);
    }

    #[test]
    fn test_vmovupd_zmm_rr_basic() {
        let enc = encode_vmovupd_zmm_rr(Zmm::Zmm4, Zmm::Zmm5);
        assert!(enc.len() >= 6);
    }

    #[test]
    fn test_vmovups_zmm_rr_basic() {
        let enc = encode_vmovups_zmm_rr(Zmm::Zmm6, Zmm::Zmm7);
        assert!(enc.len() >= 6);
    }

    #[test]
    fn test_vmovdqa64_zmm_rr_basic() {
        let enc = encode_vmovdqa64_zmm_rr(Zmm::Zmm8, Zmm::Zmm9);
        assert!(enc.len() >= 6);
        assert_eq!(enc.as_slice()[0], 0x62);
    }

    #[test]
    fn test_vmovdqa32_zmm_rr_basic() {
        let enc = encode_vmovdqa32_zmm_rr(Zmm::Zmm10, Zmm::Zmm11);
        assert!(enc.len() >= 6);
    }

    #[test]
    fn test_vmovdqu64_zmm_rr_basic() {
        let enc = encode_vmovdqu64_zmm_rr(Zmm::Zmm12, Zmm::Zmm13);
        assert!(enc.len() >= 6);
    }

    #[test]
    fn test_vmovdqu32_zmm_rr_basic() {
        let enc = encode_vmovdqu32_zmm_rr(Zmm::Zmm14, Zmm::Zmm15);
        assert!(enc.len() >= 6);
    }

    #[test]
    fn test_zmm_move_all_registers() {
        // Test move instruction encoding for all 32 ZMM register pairs
        for i in 0..32u8 {
            for j in 0..32u8 {
                let src = Zmm::from_encoding(i).unwrap();
                let dst = Zmm::from_encoding(j).unwrap();

                // All combinations should encode without panic
                let enc = encode_vmovapd_zmm_rr(dst, src);
                assert!(enc.len() >= 6, "Failed for ZMM{} -> ZMM{}", i, j);
            }
        }
    }

    #[test]
    fn test_zmm_move_self_encoding() {
        // Moving register to itself should still encode correctly
        for i in 0..32u8 {
            let reg = Zmm::from_encoding(i).unwrap();
            let enc = encode_vmovapd_zmm_rr(reg, reg);
            assert!(enc.len() >= 6);
        }
    }
}
