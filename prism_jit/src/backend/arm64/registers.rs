//! ARM64 register definitions and AAPCS64 calling conventions.
//!
//! This module provides:
//! - General-purpose register (GPR) definitions with proper encoding
//! - SIMD/FP register definitions for NEON operations
//! - AAPCS64 calling convention abstractions
//!
//! # Performance Considerations
//! - All register types are `Copy` with `#[repr(u8)]` for zero-cost encoding
//! - Register sets use 64-bit bitfields for O(1) membership testing (31 GPRs)
//! - Calling conventions are const-evaluated where possible
//!
//! # ARM64 Register Architecture
//! - 31 general-purpose registers (X0-X30)
//! - XZR/WZR: Zero register (reads as zero, writes are discarded)
//! - SP: Stack pointer (separate from GPRs)
//! - 32 SIMD/FP registers (V0-V31, each 128-bit)

use std::fmt;

// =============================================================================
// General-Purpose Registers (GPR)
// =============================================================================

/// ARM64 general-purpose register with proper hardware encoding.
///
/// ARM64 has 31 general-purpose 64-bit registers (X0-X30).
/// The encoding (0-30) maps directly to the instruction format.
/// Note: X31 is context-dependent - it can be either SP or XZR.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum Gpr {
    // Argument/result registers (caller-saved)
    X0 = 0,
    X1 = 1,
    X2 = 2,
    X3 = 3,
    X4 = 4,
    X5 = 5,
    X6 = 6,
    X7 = 7,

    // Indirect result location (caller-saved)
    X8 = 8,

    // Temporary registers (caller-saved)
    X9 = 9,
    X10 = 10,
    X11 = 11,
    X12 = 12,
    X13 = 13,
    X14 = 14,
    X15 = 15,

    // Intra-procedure-call scratch registers (caller-saved)
    // These are used by the linker for veneers and PLT entries
    X16 = 16, // IP0
    X17 = 17, // IP1

    // Platform-reserved register - DO NOT USE
    // On Apple platforms this is reserved for the OS
    // On Linux it may be used for TLS
    X18 = 18,

    // Callee-saved registers
    X19 = 19,
    X20 = 20,
    X21 = 21,
    X22 = 22,
    X23 = 23,
    X24 = 24,
    X25 = 25,
    X26 = 26,
    X27 = 27,
    X28 = 28,

    // Frame pointer (callee-saved)
    X29 = 29, // FP

    // Link register (holds return address)
    X30 = 30, // LR
}

impl Gpr {
    /// All 31 general-purpose registers in encoding order.
    pub const ALL: [Gpr; 31] = [
        Gpr::X0,
        Gpr::X1,
        Gpr::X2,
        Gpr::X3,
        Gpr::X4,
        Gpr::X5,
        Gpr::X6,
        Gpr::X7,
        Gpr::X8,
        Gpr::X9,
        Gpr::X10,
        Gpr::X11,
        Gpr::X12,
        Gpr::X13,
        Gpr::X14,
        Gpr::X15,
        Gpr::X16,
        Gpr::X17,
        Gpr::X18,
        Gpr::X19,
        Gpr::X20,
        Gpr::X21,
        Gpr::X22,
        Gpr::X23,
        Gpr::X24,
        Gpr::X25,
        Gpr::X26,
        Gpr::X27,
        Gpr::X28,
        Gpr::X29,
        Gpr::X30,
    ];

    /// Argument registers (X0-X7).
    pub const ARGS: [Gpr; 8] = [
        Gpr::X0,
        Gpr::X1,
        Gpr::X2,
        Gpr::X3,
        Gpr::X4,
        Gpr::X5,
        Gpr::X6,
        Gpr::X7,
    ];

    /// Temporary/scratch registers (X9-X15).
    pub const TEMPS: [Gpr; 7] = [
        Gpr::X9,
        Gpr::X10,
        Gpr::X11,
        Gpr::X12,
        Gpr::X13,
        Gpr::X14,
        Gpr::X15,
    ];

    /// Callee-saved registers (X19-X28).
    pub const CALLEE_SAVED: [Gpr; 10] = [
        Gpr::X19,
        Gpr::X20,
        Gpr::X21,
        Gpr::X22,
        Gpr::X23,
        Gpr::X24,
        Gpr::X25,
        Gpr::X26,
        Gpr::X27,
        Gpr::X28,
    ];

    /// Frame pointer register.
    pub const FP: Gpr = Gpr::X29;

    /// Link register (return address).
    pub const LR: Gpr = Gpr::X30;

    /// Get the hardware encoding (0-30).
    #[inline(always)]
    pub const fn encoding(self) -> u8 {
        self as u8
    }

    /// Get the 5-bit field value for instruction encoding.
    /// For X0-X30, this is the register number.
    #[inline(always)]
    pub const fn field(self) -> u32 {
        self as u32
    }

    /// Check if this is an argument register (X0-X7).
    #[inline(always)]
    pub const fn is_arg(self) -> bool {
        (self as u8) <= 7
    }

    /// Check if this is a temporary register (X9-X15).
    #[inline(always)]
    pub const fn is_temp(self) -> bool {
        let enc = self as u8;
        enc >= 9 && enc <= 15
    }

    /// Check if this is a callee-saved register (X19-X28).
    #[inline(always)]
    pub const fn is_callee_saved(self) -> bool {
        let enc = self as u8;
        enc >= 19 && enc <= 28
    }

    /// Check if this is the platform-reserved register (X18).
    #[inline(always)]
    pub const fn is_reserved(self) -> bool {
        self as u8 == 18
    }

    /// Check if this is the frame pointer (X29).
    #[inline(always)]
    pub const fn is_fp(self) -> bool {
        self as u8 == 29
    }

    /// Check if this is the link register (X30).
    #[inline(always)]
    pub const fn is_lr(self) -> bool {
        self as u8 == 30
    }

    /// Convert from encoding value if valid.
    #[inline]
    pub const fn from_encoding(enc: u8) -> Option<Gpr> {
        match enc {
            0 => Some(Gpr::X0),
            1 => Some(Gpr::X1),
            2 => Some(Gpr::X2),
            3 => Some(Gpr::X3),
            4 => Some(Gpr::X4),
            5 => Some(Gpr::X5),
            6 => Some(Gpr::X6),
            7 => Some(Gpr::X7),
            8 => Some(Gpr::X8),
            9 => Some(Gpr::X9),
            10 => Some(Gpr::X10),
            11 => Some(Gpr::X11),
            12 => Some(Gpr::X12),
            13 => Some(Gpr::X13),
            14 => Some(Gpr::X14),
            15 => Some(Gpr::X15),
            16 => Some(Gpr::X16),
            17 => Some(Gpr::X17),
            18 => Some(Gpr::X18),
            19 => Some(Gpr::X19),
            20 => Some(Gpr::X20),
            21 => Some(Gpr::X21),
            22 => Some(Gpr::X22),
            23 => Some(Gpr::X23),
            24 => Some(Gpr::X24),
            25 => Some(Gpr::X25),
            26 => Some(Gpr::X26),
            27 => Some(Gpr::X27),
            28 => Some(Gpr::X28),
            29 => Some(Gpr::X29),
            30 => Some(Gpr::X30),
            _ => None,
        }
    }

    /// Get the 64-bit register name (Xn).
    pub const fn name_64(self) -> &'static str {
        match self {
            Gpr::X0 => "x0",
            Gpr::X1 => "x1",
            Gpr::X2 => "x2",
            Gpr::X3 => "x3",
            Gpr::X4 => "x4",
            Gpr::X5 => "x5",
            Gpr::X6 => "x6",
            Gpr::X7 => "x7",
            Gpr::X8 => "x8",
            Gpr::X9 => "x9",
            Gpr::X10 => "x10",
            Gpr::X11 => "x11",
            Gpr::X12 => "x12",
            Gpr::X13 => "x13",
            Gpr::X14 => "x14",
            Gpr::X15 => "x15",
            Gpr::X16 => "x16",
            Gpr::X17 => "x17",
            Gpr::X18 => "x18",
            Gpr::X19 => "x19",
            Gpr::X20 => "x20",
            Gpr::X21 => "x21",
            Gpr::X22 => "x22",
            Gpr::X23 => "x23",
            Gpr::X24 => "x24",
            Gpr::X25 => "x25",
            Gpr::X26 => "x26",
            Gpr::X27 => "x27",
            Gpr::X28 => "x28",
            Gpr::X29 => "fp",
            Gpr::X30 => "lr",
        }
    }

    /// Get the 32-bit register name (Wn).
    pub const fn name_32(self) -> &'static str {
        match self {
            Gpr::X0 => "w0",
            Gpr::X1 => "w1",
            Gpr::X2 => "w2",
            Gpr::X3 => "w3",
            Gpr::X4 => "w4",
            Gpr::X5 => "w5",
            Gpr::X6 => "w6",
            Gpr::X7 => "w7",
            Gpr::X8 => "w8",
            Gpr::X9 => "w9",
            Gpr::X10 => "w10",
            Gpr::X11 => "w11",
            Gpr::X12 => "w12",
            Gpr::X13 => "w13",
            Gpr::X14 => "w14",
            Gpr::X15 => "w15",
            Gpr::X16 => "w16",
            Gpr::X17 => "w17",
            Gpr::X18 => "w18",
            Gpr::X19 => "w19",
            Gpr::X20 => "w20",
            Gpr::X21 => "w21",
            Gpr::X22 => "w22",
            Gpr::X23 => "w23",
            Gpr::X24 => "w24",
            Gpr::X25 => "w25",
            Gpr::X26 => "w26",
            Gpr::X27 => "w27",
            Gpr::X28 => "w28",
            Gpr::X29 => "w29",
            Gpr::X30 => "w30",
        }
    }
}

impl fmt::Display for Gpr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.name_64())
    }
}

// =============================================================================
// Special Registers
// =============================================================================

/// Constant for the stack pointer encoding (31).
/// When encoding field 31 is used in contexts that accept SP.
pub const SP_ENCODING: u32 = 31;

/// Constant for the zero register encoding (31).
/// When encoding field 31 is used in contexts that accept XZR/WZR.
pub const ZR_ENCODING: u32 = 31;

// =============================================================================
// Register Sets (Bitfield)
// =============================================================================

/// A set of GPR registers using a 32-bit bitfield for O(1) operations.
/// Supports all 31 ARM64 GPRs (X0-X30).
#[derive(Clone, Copy, PartialEq, Eq, Default)]
pub struct GprSet(u32);

impl GprSet {
    /// Empty register set.
    pub const EMPTY: GprSet = GprSet(0);

    /// All 31 registers (X0-X30).
    pub const ALL: GprSet = GprSet(0x7FFF_FFFF);

    /// Argument registers (X0-X7).
    pub const ARGS: GprSet = GprSet(0x00FF);

    /// Temporary registers (X9-X15).
    pub const TEMPS: GprSet = GprSet(0xFE00);

    /// Callee-saved registers (X19-X28).
    pub const CALLEE_SAVED: GprSet = GprSet(0x1FF8_0000);

    /// Caller-saved registers (X0-X18, X30).
    pub const CALLER_SAVED: GprSet = GprSet(0x4007_FFFF);

    /// Create a set containing a single register.
    #[inline(always)]
    pub const fn singleton(reg: Gpr) -> Self {
        GprSet(1 << reg.encoding())
    }

    /// Create from a raw bitmask.
    #[inline(always)]
    pub const fn from_bits(bits: u32) -> Self {
        GprSet(bits & 0x7FFF_FFFF) // Mask to 31 bits
    }

    /// Get the raw bitmask.
    #[inline(always)]
    pub const fn bits(self) -> u32 {
        self.0
    }

    /// Check if the set contains a register.
    #[inline(always)]
    pub const fn contains(self, reg: Gpr) -> bool {
        (self.0 & (1 << reg.encoding())) != 0
    }

    /// Add a register to the set.
    #[inline(always)]
    pub const fn insert(self, reg: Gpr) -> Self {
        GprSet(self.0 | (1 << reg.encoding()))
    }

    /// Remove a register from the set.
    #[inline(always)]
    pub const fn remove(self, reg: Gpr) -> Self {
        GprSet(self.0 & !(1 << reg.encoding()))
    }

    /// Union of two sets.
    #[inline(always)]
    pub const fn union(self, other: GprSet) -> Self {
        GprSet(self.0 | other.0)
    }

    /// Intersection of two sets.
    #[inline(always)]
    pub const fn intersection(self, other: GprSet) -> Self {
        GprSet(self.0 & other.0)
    }

    /// Difference (self - other).
    #[inline(always)]
    pub const fn difference(self, other: GprSet) -> Self {
        GprSet(self.0 & !other.0)
    }

    /// Check if the set is empty.
    #[inline(always)]
    pub const fn is_empty(self) -> bool {
        self.0 == 0
    }

    /// Count the number of registers in the set.
    #[inline(always)]
    pub const fn count(self) -> u32 {
        self.0.count_ones()
    }

    /// Get the first register (lowest encoding) in the set, if any.
    #[inline]
    pub const fn first(self) -> Option<Gpr> {
        if self.0 == 0 {
            None
        } else {
            Gpr::from_encoding(self.0.trailing_zeros() as u8)
        }
    }

    /// Iterate over registers in the set (ascending order).
    pub fn iter(self) -> impl Iterator<Item = Gpr> {
        (0..31).filter_map(move |i| {
            if (self.0 & (1 << i)) != 0 {
                Gpr::from_encoding(i)
            } else {
                None
            }
        })
    }
}

impl fmt::Debug for GprSet {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "GprSet{{")?;
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
// AAPCS64 Calling Convention
// =============================================================================

/// AAPCS64 calling convention definition.
///
/// The ARM Architecture Procedure Call Standard for AArch64 defines
/// how functions pass arguments and return values.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CallingConvention;

impl CallingConvention {
    /// Integer argument registers in order.
    pub const INT_ARG_REGS: [Gpr; 8] = [
        Gpr::X0,
        Gpr::X1,
        Gpr::X2,
        Gpr::X3,
        Gpr::X4,
        Gpr::X5,
        Gpr::X6,
        Gpr::X7,
    ];

    /// Integer return register.
    pub const INT_RETURN_REG: Gpr = Gpr::X0;

    /// Secondary integer return register (for 128-bit returns).
    pub const INT_RETURN_REG_2: Gpr = Gpr::X1;

    /// Indirect result location register.
    pub const INDIRECT_RESULT_REG: Gpr = Gpr::X8;

    /// Intra-procedure-call scratch registers.
    pub const IPC_SCRATCH: [Gpr; 2] = [Gpr::X16, Gpr::X17];

    /// Frame pointer register.
    pub const FRAME_POINTER: Gpr = Gpr::X29;

    /// Link register (return address).
    pub const LINK_REGISTER: Gpr = Gpr::X30;

    /// Get volatile (caller-saved) GPRs.
    /// These are: X0-X18, X30 (LR is caller-saved)
    pub const fn volatile_gprs() -> GprSet {
        GprSet::CALLER_SAVED
    }

    /// Get non-volatile (callee-saved) GPRs.
    /// These are: X19-X28, X29 (FP)
    pub const fn callee_saved_gprs() -> GprSet {
        GprSet::CALLEE_SAVED.insert(Gpr::X29)
    }

    /// Get the stack alignment requirement in bytes.
    /// AAPCS64 requires 16-byte stack alignment at all times.
    pub const fn stack_alignment() -> usize {
        16
    }

    /// Get the number of argument registers.
    pub const fn num_arg_regs() -> usize {
        8
    }

    /// Get the n-th argument register, if available.
    pub const fn arg_reg(n: usize) -> Option<Gpr> {
        if n < 8 {
            Some(Self::INT_ARG_REGS[n])
        } else {
            None
        }
    }
}

// =============================================================================
// Allocatable Registers for JIT
// =============================================================================

/// Registers available for allocation by the JIT compiler.
///
/// Excludes SP, X18 (platform-reserved), and designated scratch registers.
pub struct AllocatableRegs {
    /// GPRs available for general allocation.
    pub gprs: GprSet,
    /// Scratch register (always available, not allocated to user code).
    pub scratch_gpr: Gpr,
    /// Secondary scratch register.
    pub scratch_gpr_2: Gpr,
}

impl AllocatableRegs {
    /// Get allocatable registers for AAPCS64.
    pub const fn new() -> Self {
        // Exclude:
        // - X18: Platform-reserved (Apple uses for TLS)
        // - X16, X17: Used as JIT scratch registers (IP0, IP1)
        // - X29 (FP): Frame pointer (could be optional, but safer to exclude)
        // - X30 (LR): Link register (must be preserved)
        //
        // Note: SP is not in GprSet at all (it's separate)
        let all_gprs = GprSet::ALL
            .remove(Gpr::X16)
            .remove(Gpr::X17)
            .remove(Gpr::X18)
            .remove(Gpr::X29)
            .remove(Gpr::X30);

        AllocatableRegs {
            gprs: all_gprs,
            scratch_gpr: Gpr::X16,
            scratch_gpr_2: Gpr::X17,
        }
    }

    /// Get the number of allocatable GPRs.
    pub const fn gpr_count(&self) -> u32 {
        self.gprs.count()
    }
}

impl Default for AllocatableRegs {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// Memory Operands
// =============================================================================

/// Extend operation for register offsets.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum Extend {
    /// Unsigned extend byte (8 to 64)
    Uxtb = 0,
    /// Unsigned extend halfword (16 to 64)
    Uxth = 1,
    /// Unsigned extend word (32 to 64)
    Uxtw = 2,
    /// Unsigned extend doubleword (64 to 64, effectively none)
    Uxtx = 3,
    /// Signed extend byte (8 to 64)
    Sxtb = 4,
    /// Signed extend halfword (16 to 64)
    Sxth = 5,
    /// Signed extend word (32 to 64)
    Sxtw = 6,
    /// Signed extend doubleword (64 to 64, effectively none)
    Sxtx = 7,
    /// Left shift (LSL) - used with register offsets
    Lsl = 8, // Special case, encoded differently
}

impl Extend {
    /// Get the 3-bit encoding for extend operations.
    pub const fn encoding(self) -> u32 {
        match self {
            Extend::Uxtb => 0b000,
            Extend::Uxth => 0b001,
            Extend::Uxtw => 0b010,
            Extend::Uxtx => 0b011,
            Extend::Sxtb => 0b100,
            Extend::Sxth => 0b101,
            Extend::Sxtw => 0b110,
            Extend::Sxtx => 0b111,
            Extend::Lsl => 0b011, // Same encoding as UXTX for 64-bit
        }
    }
}

/// A memory operand for ARM64 addressing.
#[derive(Debug, Clone, Copy)]
pub struct MemOperand {
    /// Base register.
    pub base: Gpr,
    /// Offset type.
    pub offset: MemOffset,
}

/// Memory offset types for ARM64 addressing modes.
#[derive(Debug, Clone, Copy)]
pub enum MemOffset {
    /// No offset: [Xn]
    None,
    /// Unsigned immediate offset: [Xn, #imm]
    /// The immediate is scaled by the access size.
    Immediate(i64),
    /// Register offset: [Xn, Xm] or [Xn, Xm, LSL #shift]
    Register {
        index: Gpr,
        extend: Extend,
        shift: u8,
    },
    /// Pre-indexed: [Xn, #imm]!
    /// Updates base register before access.
    PreIndex(i64),
    /// Post-indexed: [Xn], #imm
    /// Updates base register after access.
    PostIndex(i64),
}

impl MemOperand {
    /// Create a simple [base] addressing mode.
    #[inline]
    pub const fn base(reg: Gpr) -> Self {
        MemOperand {
            base: reg,
            offset: MemOffset::None,
        }
    }

    /// Create a [base, #imm] addressing mode.
    #[inline]
    pub const fn base_offset(base: Gpr, offset: i64) -> Self {
        MemOperand {
            base,
            offset: MemOffset::Immediate(offset),
        }
    }

    /// Create a [base, Xm] register offset addressing mode.
    #[inline]
    pub const fn base_index(base: Gpr, index: Gpr) -> Self {
        MemOperand {
            base,
            offset: MemOffset::Register {
                index,
                extend: Extend::Lsl,
                shift: 0,
            },
        }
    }

    /// Create a [base, Xm, LSL #shift] addressing mode.
    #[inline]
    pub const fn base_index_shift(base: Gpr, index: Gpr, shift: u8) -> Self {
        MemOperand {
            base,
            offset: MemOffset::Register {
                index,
                extend: Extend::Lsl,
                shift,
            },
        }
    }

    /// Create a [base, Wm, SXTW #shift] addressing mode for 32-bit index.
    #[inline]
    pub const fn base_index_sxtw(base: Gpr, index: Gpr, shift: u8) -> Self {
        MemOperand {
            base,
            offset: MemOffset::Register {
                index,
                extend: Extend::Sxtw,
                shift,
            },
        }
    }

    /// Create a pre-indexed [base, #imm]! addressing mode.
    #[inline]
    pub const fn pre_index(base: Gpr, offset: i64) -> Self {
        MemOperand {
            base,
            offset: MemOffset::PreIndex(offset),
        }
    }

    /// Create a post-indexed [base], #imm addressing mode.
    #[inline]
    pub const fn post_index(base: Gpr, offset: i64) -> Self {
        MemOperand {
            base,
            offset: MemOffset::PostIndex(offset),
        }
    }

    /// Check if this is a simple base-only addressing mode.
    #[inline]
    pub const fn is_base_only(&self) -> bool {
        matches!(self.offset, MemOffset::None)
    }

    /// Check if this uses an immediate offset.
    #[inline]
    pub const fn is_immediate(&self) -> bool {
        matches!(self.offset, MemOffset::Immediate(_))
    }

    /// Check if this uses a register offset.
    #[inline]
    pub const fn is_register_offset(&self) -> bool {
        matches!(self.offset, MemOffset::Register { .. })
    }

    /// Check if this is pre-indexed.
    #[inline]
    pub const fn is_pre_index(&self) -> bool {
        matches!(self.offset, MemOffset::PreIndex(_))
    }

    /// Check if this is post-indexed.
    #[inline]
    pub const fn is_post_index(&self) -> bool {
        matches!(self.offset, MemOffset::PostIndex(_))
    }
}

impl fmt::Display for MemOperand {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.offset {
            MemOffset::None => write!(f, "[{}]", self.base),
            MemOffset::Immediate(0) => write!(f, "[{}]", self.base),
            MemOffset::Immediate(off) => write!(f, "[{}, #{}]", self.base, off),
            MemOffset::Register {
                index,
                extend,
                shift,
            } => {
                write!(f, "[{}, {}", self.base, index)?;
                if *shift > 0 || *extend != Extend::Lsl {
                    write!(f, ", {:?} #{}", extend, shift)?;
                }
                write!(f, "]")
            }
            MemOffset::PreIndex(off) => write!(f, "[{}, #{}]!", self.base, off),
            MemOffset::PostIndex(off) => write!(f, "[{}], #{}", self.base, off),
        }
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gpr_encoding() {
        assert_eq!(Gpr::X0.encoding(), 0);
        assert_eq!(Gpr::X1.encoding(), 1);
        assert_eq!(Gpr::X15.encoding(), 15);
        assert_eq!(Gpr::X30.encoding(), 30);
    }

    #[test]
    fn test_gpr_all_count() {
        assert_eq!(Gpr::ALL.len(), 31);
    }

    #[test]
    fn test_gpr_categories() {
        assert!(Gpr::X0.is_arg());
        assert!(Gpr::X7.is_arg());
        assert!(!Gpr::X8.is_arg());

        assert!(Gpr::X9.is_temp());
        assert!(Gpr::X15.is_temp());
        assert!(!Gpr::X8.is_temp());
        assert!(!Gpr::X16.is_temp());

        assert!(Gpr::X19.is_callee_saved());
        assert!(Gpr::X28.is_callee_saved());
        assert!(!Gpr::X29.is_callee_saved()); // FP, handled specially

        assert!(Gpr::X18.is_reserved());
        assert!(!Gpr::X17.is_reserved());

        assert!(Gpr::X29.is_fp());
        assert!(Gpr::X30.is_lr());
    }

    #[test]
    fn test_gpr_from_encoding() {
        for i in 0..31u8 {
            let reg = Gpr::from_encoding(i).unwrap();
            assert_eq!(reg.encoding(), i);
        }
        assert!(Gpr::from_encoding(31).is_none());
        assert!(Gpr::from_encoding(32).is_none());
    }

    #[test]
    fn test_gpr_set_singleton() {
        let set = GprSet::singleton(Gpr::X5);
        assert!(set.contains(Gpr::X5));
        assert!(!set.contains(Gpr::X4));
        assert!(!set.contains(Gpr::X6));
        assert_eq!(set.count(), 1);
    }

    #[test]
    fn test_gpr_set_operations() {
        let set1 = GprSet::singleton(Gpr::X0).insert(Gpr::X1);
        let set2 = GprSet::singleton(Gpr::X1).insert(Gpr::X2);

        let union = set1.union(set2);
        assert!(union.contains(Gpr::X0));
        assert!(union.contains(Gpr::X1));
        assert!(union.contains(Gpr::X2));
        assert_eq!(union.count(), 3);

        let intersection = set1.intersection(set2);
        assert!(!intersection.contains(Gpr::X0));
        assert!(intersection.contains(Gpr::X1));
        assert!(!intersection.contains(Gpr::X2));
        assert_eq!(intersection.count(), 1);

        let diff = set1.difference(set2);
        assert!(diff.contains(Gpr::X0));
        assert!(!diff.contains(Gpr::X1));
        assert!(!diff.contains(Gpr::X2));
        assert_eq!(diff.count(), 1);
    }

    #[test]
    fn test_gpr_set_all() {
        assert_eq!(GprSet::ALL.count(), 31);
        for i in 0..31 {
            let reg = Gpr::from_encoding(i).unwrap();
            assert!(GprSet::ALL.contains(reg));
        }
    }

    #[test]
    fn test_gpr_set_iter() {
        let set = GprSet::singleton(Gpr::X3).insert(Gpr::X7).insert(Gpr::X15);

        let regs: Vec<_> = set.iter().collect();
        assert_eq!(regs, vec![Gpr::X3, Gpr::X7, Gpr::X15]);
    }

    #[test]
    fn test_gpr_set_first() {
        let empty = GprSet::EMPTY;
        assert!(empty.first().is_none());

        let set = GprSet::singleton(Gpr::X5).insert(Gpr::X10);
        assert_eq!(set.first(), Some(Gpr::X5));
    }

    #[test]
    fn test_calling_convention_arg_regs() {
        for i in 0..8 {
            let reg = CallingConvention::arg_reg(i).unwrap();
            assert_eq!(reg.encoding() as usize, i);
        }
        assert!(CallingConvention::arg_reg(8).is_none());
    }

    #[test]
    fn test_calling_convention_callee_saved() {
        let callee_saved = CallingConvention::callee_saved_gprs();

        // X19-X28 should be callee-saved
        for i in 19..29 {
            let reg = Gpr::from_encoding(i).unwrap();
            assert!(callee_saved.contains(reg), "X{} should be callee-saved", i);
        }

        // X29 (FP) should also be callee-saved
        assert!(callee_saved.contains(Gpr::X29));

        // X0-X18 and X30 should NOT be callee-saved
        for i in 0..19 {
            let reg = Gpr::from_encoding(i).unwrap();
            assert!(
                !callee_saved.contains(reg),
                "X{} should NOT be callee-saved",
                i
            );
        }
        assert!(!callee_saved.contains(Gpr::X30));
    }

    #[test]
    fn test_allocatable_regs() {
        let alloc = AllocatableRegs::new();

        // Should have 26 allocatable GPRs (31 - 5: X16, X17, X18, X29, X30)
        assert_eq!(alloc.gpr_count(), 26);

        // Scratch registers should not be in allocatable set
        assert!(!alloc.gprs.contains(Gpr::X16));
        assert!(!alloc.gprs.contains(Gpr::X17));

        // Reserved register should not be allocatable
        assert!(!alloc.gprs.contains(Gpr::X18));

        // FP and LR should not be allocatable
        assert!(!alloc.gprs.contains(Gpr::X29));
        assert!(!alloc.gprs.contains(Gpr::X30));

        // Argument registers should be allocatable
        for i in 0..8 {
            let reg = Gpr::from_encoding(i).unwrap();
            assert!(alloc.gprs.contains(reg));
        }
    }

    #[test]
    fn test_mem_operand_display() {
        let base_only = MemOperand::base(Gpr::X0);
        assert_eq!(format!("{}", base_only), "[x0]");

        let base_offset = MemOperand::base_offset(Gpr::X1, 16);
        assert_eq!(format!("{}", base_offset), "[x1, #16]");

        let pre_index = MemOperand::pre_index(Gpr::X2, -16);
        assert_eq!(format!("{}", pre_index), "[x2, #-16]!");

        let post_index = MemOperand::post_index(Gpr::X3, 8);
        assert_eq!(format!("{}", post_index), "[x3], #8");
    }

    #[test]
    fn test_extend_encoding() {
        assert_eq!(Extend::Uxtb.encoding(), 0b000);
        assert_eq!(Extend::Uxth.encoding(), 0b001);
        assert_eq!(Extend::Uxtw.encoding(), 0b010);
        assert_eq!(Extend::Uxtx.encoding(), 0b011);
        assert_eq!(Extend::Sxtb.encoding(), 0b100);
        assert_eq!(Extend::Sxth.encoding(), 0b101);
        assert_eq!(Extend::Sxtw.encoding(), 0b110);
        assert_eq!(Extend::Sxtx.encoding(), 0b111);
    }

    #[test]
    fn test_stack_alignment() {
        assert_eq!(CallingConvention::stack_alignment(), 16);
    }

    #[test]
    fn test_special_registers() {
        assert_eq!(SP_ENCODING, 31);
        assert_eq!(ZR_ENCODING, 31);
    }

    #[test]
    fn test_gpr_names() {
        assert_eq!(Gpr::X0.name_64(), "x0");
        assert_eq!(Gpr::X0.name_32(), "w0");
        assert_eq!(Gpr::X29.name_64(), "fp");
        assert_eq!(Gpr::X30.name_64(), "lr");
        assert_eq!(Gpr::X15.name_64(), "x15");
        assert_eq!(Gpr::X15.name_32(), "w15");
    }
}
