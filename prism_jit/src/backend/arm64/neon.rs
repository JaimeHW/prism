//! ARM64 NEON/ASIMD SIMD register and instruction support.
//!
//! NEON (Advanced SIMD) provides 32 128-bit vector registers (V0-V31).
//! Each can be viewed as different arrangements of elements.

use std::fmt;

// =============================================================================
// Vector Registers
// =============================================================================

/// ARM64 SIMD/FP vector register (V0-V31).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum Vreg {
    V0 = 0,
    V1,
    V2,
    V3,
    V4,
    V5,
    V6,
    V7,
    V8,
    V9,
    V10,
    V11,
    V12,
    V13,
    V14,
    V15,
    V16,
    V17,
    V18,
    V19,
    V20,
    V21,
    V22,
    V23,
    V24,
    V25,
    V26,
    V27,
    V28,
    V29,
    V30,
    V31,
}

impl Vreg {
    /// All 32 vector registers.
    pub const ALL: [Vreg; 32] = [
        Vreg::V0,
        Vreg::V1,
        Vreg::V2,
        Vreg::V3,
        Vreg::V4,
        Vreg::V5,
        Vreg::V6,
        Vreg::V7,
        Vreg::V8,
        Vreg::V9,
        Vreg::V10,
        Vreg::V11,
        Vreg::V12,
        Vreg::V13,
        Vreg::V14,
        Vreg::V15,
        Vreg::V16,
        Vreg::V17,
        Vreg::V18,
        Vreg::V19,
        Vreg::V20,
        Vreg::V21,
        Vreg::V22,
        Vreg::V23,
        Vreg::V24,
        Vreg::V25,
        Vreg::V26,
        Vreg::V27,
        Vreg::V28,
        Vreg::V29,
        Vreg::V30,
        Vreg::V31,
    ];

    /// Get the hardware encoding (0-31).
    #[inline(always)]
    pub const fn encoding(self) -> u8 {
        self as u8
    }

    /// Get the 5-bit field value for instruction encoding.
    #[inline(always)]
    pub const fn field(self) -> u32 {
        self as u32
    }

    /// Convert from encoding value if valid.
    #[inline]
    pub const fn from_encoding(enc: u8) -> Option<Vreg> {
        if enc < 32 {
            Some(unsafe { std::mem::transmute(enc) })
        } else {
            None
        }
    }

    /// Check if caller-saved (V0-V7 for args, V16-V31).
    pub const fn is_caller_saved(self) -> bool {
        let e = self as u8;
        e <= 7 || e >= 16
    }

    /// Check if callee-saved (V8-V15, lower 64 bits only).
    pub const fn is_callee_saved(self) -> bool {
        let e = self as u8;
        e >= 8 && e <= 15
    }
}

impl fmt::Display for Vreg {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "v{}", self.encoding())
    }
}

// =============================================================================
// Vector Register Set
// =============================================================================

/// A set of SIMD registers using a 32-bit bitfield.
#[derive(Clone, Copy, PartialEq, Eq, Default)]
pub struct VregSet(u32);

impl VregSet {
    pub const EMPTY: VregSet = VregSet(0);
    pub const ALL: VregSet = VregSet(0xFFFF_FFFF);
    pub const CALLEE_SAVED: VregSet = VregSet(0x0000_FF00);

    #[inline(always)]
    pub const fn singleton(reg: Vreg) -> Self {
        VregSet(1 << reg.encoding())
    }

    #[inline(always)]
    pub const fn contains(self, reg: Vreg) -> bool {
        (self.0 & (1 << reg.encoding())) != 0
    }

    #[inline(always)]
    pub const fn insert(self, reg: Vreg) -> Self {
        VregSet(self.0 | (1 << reg.encoding()))
    }

    #[inline(always)]
    pub const fn remove(self, reg: Vreg) -> Self {
        VregSet(self.0 & !(1 << reg.encoding()))
    }

    #[inline(always)]
    pub const fn count(self) -> u32 {
        self.0.count_ones()
    }

    pub fn iter(self) -> impl Iterator<Item = Vreg> {
        (0..32).filter_map(move |i| {
            if (self.0 & (1 << i)) != 0 {
                Vreg::from_encoding(i)
            } else {
                None
            }
        })
    }
}

impl fmt::Debug for VregSet {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "VregSet{{")?;
        let mut first = true;
        for reg in self.iter() {
            if !first {
                write!(f, ", ")?;
            }
            write!(f, "{}", reg)?;
            first = false;
        }
        write!(f, "}}")
    }
}

// =============================================================================
// Arrangement Specifiers
// =============================================================================

/// Vector element arrangement specifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArrangementSpec {
    /// 8 bytes: 8B (8 × 8-bit)
    B8,
    /// 16 bytes: 16B (16 × 8-bit)
    B16,
    /// 4 halfwords: 4H (4 × 16-bit)
    H4,
    /// 8 halfwords: 8H (8 × 16-bit)
    H8,
    /// 2 words: 2S (2 × 32-bit)
    S2,
    /// 4 words: 4S (4 × 32-bit)
    S4,
    /// 1 doubleword: 1D (1 × 64-bit)
    D1,
    /// 2 doublewords: 2D (2 × 64-bit)
    D2,
}

impl ArrangementSpec {
    /// Get the Q bit (0 = 64-bit, 1 = 128-bit).
    pub const fn q_bit(self) -> u32 {
        match self {
            ArrangementSpec::B8
            | ArrangementSpec::H4
            | ArrangementSpec::S2
            | ArrangementSpec::D1 => 0,
            ArrangementSpec::B16
            | ArrangementSpec::H8
            | ArrangementSpec::S4
            | ArrangementSpec::D2 => 1,
        }
    }

    /// Get the size bits (2-bit encoding).
    pub const fn size_bits(self) -> u32 {
        match self {
            ArrangementSpec::B8 | ArrangementSpec::B16 => 0b00,
            ArrangementSpec::H4 | ArrangementSpec::H8 => 0b01,
            ArrangementSpec::S2 | ArrangementSpec::S4 => 0b10,
            ArrangementSpec::D1 | ArrangementSpec::D2 => 0b11,
        }
    }

    /// Get element count.
    pub const fn count(self) -> u8 {
        match self {
            ArrangementSpec::D1 => 1,
            ArrangementSpec::S2 | ArrangementSpec::D2 => 2,
            ArrangementSpec::H4 | ArrangementSpec::S4 => 4,
            ArrangementSpec::B8 | ArrangementSpec::H8 => 8,
            ArrangementSpec::B16 => 16,
        }
    }

    /// Get element width in bits.
    pub const fn element_bits(self) -> u8 {
        match self {
            ArrangementSpec::B8 | ArrangementSpec::B16 => 8,
            ArrangementSpec::H4 | ArrangementSpec::H8 => 16,
            ArrangementSpec::S2 | ArrangementSpec::S4 => 32,
            ArrangementSpec::D1 | ArrangementSpec::D2 => 64,
        }
    }
}

impl fmt::Display for ArrangementSpec {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ArrangementSpec::B8 => write!(f, "8B"),
            ArrangementSpec::B16 => write!(f, "16B"),
            ArrangementSpec::H4 => write!(f, "4H"),
            ArrangementSpec::H8 => write!(f, "8H"),
            ArrangementSpec::S2 => write!(f, "2S"),
            ArrangementSpec::S4 => write!(f, "4S"),
            ArrangementSpec::D1 => write!(f, "1D"),
            ArrangementSpec::D2 => write!(f, "2D"),
        }
    }
}

// =============================================================================
// NEON Instruction Encoding
// =============================================================================

use super::encoder::EncodedInst;

/// FADD Vd.T, Vn.T, Vm.T (floating-point add)
#[inline]
pub fn encode_fadd(dst: Vreg, n: Vreg, m: Vreg, arr: ArrangementSpec) -> EncodedInst {
    debug_assert!(matches!(
        arr,
        ArrangementSpec::S2 | ArrangementSpec::S4 | ArrangementSpec::D2
    ));
    let q = arr.q_bit();
    let sz = if matches!(arr, ArrangementSpec::D2) {
        1
    } else {
        0
    };
    EncodedInst::new(
        (0 << 31)
            | (q << 30)
            | (0 << 29)
            | (0b01110 << 24)
            | (sz << 22)
            | (1 << 21)
            | (m.field() << 16)
            | (0b110101 << 10)
            | (n.field() << 5)
            | dst.field(),
    )
}

/// FSUB Vd.T, Vn.T, Vm.T
#[inline]
pub fn encode_fsub(dst: Vreg, n: Vreg, m: Vreg, arr: ArrangementSpec) -> EncodedInst {
    debug_assert!(matches!(
        arr,
        ArrangementSpec::S2 | ArrangementSpec::S4 | ArrangementSpec::D2
    ));
    let q = arr.q_bit();
    let sz = if matches!(arr, ArrangementSpec::D2) {
        1
    } else {
        0
    };
    EncodedInst::new(
        (0 << 31)
            | (q << 30)
            | (0 << 29)
            | (0b01110 << 24)
            | (1 << 23)
            | (sz << 22)
            | (1 << 21)
            | (m.field() << 16)
            | (0b110101 << 10)
            | (n.field() << 5)
            | dst.field(),
    )
}

/// FMUL Vd.T, Vn.T, Vm.T
#[inline]
pub fn encode_fmul(dst: Vreg, n: Vreg, m: Vreg, arr: ArrangementSpec) -> EncodedInst {
    debug_assert!(matches!(
        arr,
        ArrangementSpec::S2 | ArrangementSpec::S4 | ArrangementSpec::D2
    ));
    let q = arr.q_bit();
    let sz = if matches!(arr, ArrangementSpec::D2) {
        1
    } else {
        0
    };
    EncodedInst::new(
        (0 << 31)
            | (q << 30)
            | (1 << 29)
            | (0b01110 << 24)
            | (sz << 22)
            | (1 << 21)
            | (m.field() << 16)
            | (0b110111 << 10)
            | (n.field() << 5)
            | dst.field(),
    )
}

/// FMLA Vd.T, Vn.T, Vm.T (fused multiply-add: Vd += Vn * Vm)
#[inline]
pub fn encode_fmla(dst: Vreg, n: Vreg, m: Vreg, arr: ArrangementSpec) -> EncodedInst {
    let q = arr.q_bit();
    let sz = if matches!(arr, ArrangementSpec::D2) {
        1
    } else {
        0
    };
    EncodedInst::new(
        (0 << 31)
            | (q << 30)
            | (0 << 29)
            | (0b01110 << 24)
            | (sz << 22)
            | (1 << 21)
            | (m.field() << 16)
            | (0b110011 << 10)
            | (n.field() << 5)
            | dst.field(),
    )
}

/// ADD Vd.T, Vn.T, Vm.T (integer add)
#[inline]
pub fn encode_add_vec(dst: Vreg, n: Vreg, m: Vreg, arr: ArrangementSpec) -> EncodedInst {
    let q = arr.q_bit();
    let size = arr.size_bits();
    EncodedInst::new(
        (0 << 31)
            | (q << 30)
            | (0 << 29)
            | (0b01110 << 24)
            | (size << 22)
            | (1 << 21)
            | (m.field() << 16)
            | (0b100001 << 10)
            | (n.field() << 5)
            | dst.field(),
    )
}

/// SUB Vd.T, Vn.T, Vm.T (integer sub)
#[inline]
pub fn encode_sub_vec(dst: Vreg, n: Vreg, m: Vreg, arr: ArrangementSpec) -> EncodedInst {
    let q = arr.q_bit();
    let size = arr.size_bits();
    EncodedInst::new(
        (0 << 31)
            | (q << 30)
            | (1 << 29)
            | (0b01110 << 24)
            | (size << 22)
            | (1 << 21)
            | (m.field() << 16)
            | (0b100001 << 10)
            | (n.field() << 5)
            | dst.field(),
    )
}

/// AND Vd.16B, Vn.16B, Vm.16B
#[inline]
pub fn encode_and_vec(dst: Vreg, n: Vreg, m: Vreg) -> EncodedInst {
    EncodedInst::new(
        (0 << 31)
            | (1 << 30)
            | (0 << 29)
            | (0b01110 << 24)
            | (0 << 22)
            | (1 << 21)
            | (m.field() << 16)
            | (0b000111 << 10)
            | (n.field() << 5)
            | dst.field(),
    )
}

/// ORR Vd.16B, Vn.16B, Vm.16B
#[inline]
pub fn encode_orr_vec(dst: Vreg, n: Vreg, m: Vreg) -> EncodedInst {
    EncodedInst::new(
        (0 << 31)
            | (1 << 30)
            | (0 << 29)
            | (0b01110 << 24)
            | (0b10 << 22)
            | (1 << 21)
            | (m.field() << 16)
            | (0b000111 << 10)
            | (n.field() << 5)
            | dst.field(),
    )
}

/// EOR Vd.16B, Vn.16B, Vm.16B
#[inline]
pub fn encode_eor_vec(dst: Vreg, n: Vreg, m: Vreg) -> EncodedInst {
    EncodedInst::new(
        (0 << 31)
            | (1 << 30)
            | (1 << 29)
            | (0b01110 << 24)
            | (0 << 22)
            | (1 << 21)
            | (m.field() << 16)
            | (0b000111 << 10)
            | (n.field() << 5)
            | dst.field(),
    )
}

/// BSL Vd.16B, Vn.16B, Vm.16B (bitwise select: Vd = (Vd & Vn) | (~Vd & Vm))
#[inline]
pub fn encode_bsl(dst: Vreg, n: Vreg, m: Vreg) -> EncodedInst {
    EncodedInst::new(
        (0 << 31)
            | (1 << 30)
            | (1 << 29)
            | (0b01110 << 24)
            | (0b01 << 22)
            | (1 << 21)
            | (m.field() << 16)
            | (0b000111 << 10)
            | (n.field() << 5)
            | dst.field(),
    )
}

/// LD1 {Vt.T}, [Xn] (load single structure)
#[inline]
pub fn encode_ld1(vt: Vreg, base: super::registers::Gpr, arr: ArrangementSpec) -> EncodedInst {
    let q = arr.q_bit();
    let size = arr.size_bits();
    EncodedInst::new(
        (0 << 31)
            | (q << 30)
            | (0b001100010 << 21)
            | (0 << 16)
            | (0b0111 << 12)
            | (size << 10)
            | (base.field() << 5)
            | vt.field(),
    )
}

/// ST1 {Vt.T}, [Xn] (store single structure)
#[inline]
pub fn encode_st1(vt: Vreg, base: super::registers::Gpr, arr: ArrangementSpec) -> EncodedInst {
    let q = arr.q_bit();
    let size = arr.size_bits();
    EncodedInst::new(
        (0 << 31)
            | (q << 30)
            | (0b001100000 << 21)
            | (0 << 16)
            | (0b0111 << 12)
            | (size << 10)
            | (base.field() << 5)
            | vt.field(),
    )
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vreg_encoding() {
        assert_eq!(Vreg::V0.encoding(), 0);
        assert_eq!(Vreg::V31.encoding(), 31);
    }

    #[test]
    fn test_vreg_set() {
        let set = VregSet::singleton(Vreg::V5).insert(Vreg::V10);
        assert!(set.contains(Vreg::V5));
        assert!(set.contains(Vreg::V10));
        assert!(!set.contains(Vreg::V0));
        assert_eq!(set.count(), 2);
    }

    #[test]
    fn test_arrangement() {
        assert_eq!(ArrangementSpec::S4.q_bit(), 1);
        assert_eq!(ArrangementSpec::S2.q_bit(), 0);
        assert_eq!(ArrangementSpec::S4.count(), 4);
        assert_eq!(ArrangementSpec::D2.element_bits(), 64);
    }

    #[test]
    fn test_callee_saved() {
        assert!(Vreg::V8.is_callee_saved());
        assert!(Vreg::V15.is_callee_saved());
        assert!(!Vreg::V0.is_callee_saved());
        assert!(!Vreg::V16.is_callee_saved());
    }
}
