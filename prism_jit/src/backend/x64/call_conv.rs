//! x64 Calling Convention Abstractions for Vector Registers
//!
//! This module provides comprehensive calling convention support for x64 vector
//! operations, extending the base `CallingConvention` with YMM (256-bit), ZMM
//! (512-bit), and AVX-512 mask register handling.
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────┐
//! │                    VectorCallingConvention                       │
//! ├─────────────────────────────────────────────────────────────────┤
//! │ • YMM/ZMM argument passing                                       │
//! │ • Vector return value registers                                  │
//! │ • Caller-saved/callee-saved vector register sets                │
//! │ • AVX-512 mask register (k0-k7) handling                        │
//! │ • Call-site clobber set generation                               │
//! └─────────────────────────────────────────────────────────────────┘
//! ```
//!
//! # x64 ABI Vector Register Conventions
//!
//! ## Windows x64 ABI
//! - XMM0-XMM3: Used for float/vector arguments (first 4)
//! - XMM0: Float/vector return value
//! - XMM0-XMM5: Caller-saved (volatile)
//! - XMM6-XMM15: Callee-saved (non-volatile)
//! - YMM/ZMM upper bits: Always caller-saved (volatile)
//! - k0: Reserved (always full mask), k1-k7: Caller-saved
//!
//! ## System V AMD64 ABI
//! - XMM0-XMM7: Used for float/vector arguments (up to 8)
//! - XMM0-XMM1: Float/vector return values
//! - XMM0-XMM15: All caller-saved (volatile)
//! - YMM/ZMM: Follow XMM rules (all volatile)
//! - k0: Reserved, k1-k7: Caller-saved
//!
//! # Usage
//!
//! ```ignore
//! use prism_jit::backend::x64::call_conv::VectorCallingConvention;
//!
//! let vcc = VectorCallingConvention::for_host();
//!
//! // Get all registers clobbered by a call
//! let clobbers = vcc.call_clobbers();
//!
//! // Get vector argument registers
//! let ymm_args = vcc.ymm_arg_regs();
//! ```

use super::registers::{CallingConvention, Gpr, GprSet, Xmm, XmmSet};
use super::simd::{Ymm, YmmSet, Zmm, ZmmSet};
use crate::regalloc::PReg;

// =============================================================================
// AVX-512 Mask Registers
// =============================================================================

/// AVX-512 opmask register (k0-k7).
///
/// In AVX-512, these 64-bit registers are used for predicated operations.
/// k0 is special: it always means "no masking" when used as a write mask.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum Kreg {
    K0 = 0,
    K1 = 1,
    K2 = 2,
    K3 = 3,
    K4 = 4,
    K5 = 5,
    K6 = 6,
    K7 = 7,
}

impl Kreg {
    /// All 8 mask registers.
    pub const ALL: [Kreg; 8] = [
        Kreg::K0,
        Kreg::K1,
        Kreg::K2,
        Kreg::K3,
        Kreg::K4,
        Kreg::K5,
        Kreg::K6,
        Kreg::K7,
    ];

    /// Allocatable mask registers (excludes k0 which is reserved).
    pub const ALLOCATABLE: [Kreg; 7] = [
        Kreg::K1,
        Kreg::K2,
        Kreg::K3,
        Kreg::K4,
        Kreg::K5,
        Kreg::K6,
        Kreg::K7,
    ];

    /// Get the hardware encoding.
    #[inline(always)]
    pub const fn encoding(self) -> u8 {
        self as u8
    }

    /// Get bits 0-2 for EVEX encoding.
    #[inline(always)]
    pub const fn low_bits(self) -> u8 {
        self.encoding() & 0x7
    }

    /// Convert from encoding value if valid.
    #[inline]
    pub const fn from_encoding(enc: u8) -> Option<Kreg> {
        match enc {
            0 => Some(Kreg::K0),
            1 => Some(Kreg::K1),
            2 => Some(Kreg::K2),
            3 => Some(Kreg::K3),
            4 => Some(Kreg::K4),
            5 => Some(Kreg::K5),
            6 => Some(Kreg::K6),
            7 => Some(Kreg::K7),
            _ => None,
        }
    }

    /// Check if this is k0 (the reserved "no masking" register).
    #[inline(always)]
    pub const fn is_reserved(self) -> bool {
        matches!(self, Kreg::K0)
    }
}

impl std::fmt::Display for Kreg {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "k{}", self.encoding())
    }
}

// =============================================================================
// Mask Register Set
// =============================================================================

/// A set of mask registers using an 8-bit bitfield for O(1) operations.
#[derive(Clone, Copy, PartialEq, Eq, Default)]
pub struct KregSet(u8);

impl KregSet {
    /// Empty register set.
    pub const EMPTY: KregSet = KregSet(0);

    /// All 8 registers (k0-k7).
    pub const ALL: KregSet = KregSet(0xFF);

    /// Allocatable registers (k1-k7, excludes k0).
    pub const ALLOCATABLE: KregSet = KregSet(0xFE);

    /// Create a set containing a single register.
    #[inline(always)]
    pub const fn singleton(reg: Kreg) -> Self {
        KregSet(1 << reg.encoding())
    }

    /// Create from a raw bitmask.
    #[inline(always)]
    pub const fn from_bits(bits: u8) -> Self {
        KregSet(bits)
    }

    /// Get the raw bitmask.
    #[inline(always)]
    pub const fn bits(self) -> u8 {
        self.0
    }

    /// Check if the set contains a register.
    #[inline(always)]
    pub const fn contains(self, reg: Kreg) -> bool {
        (self.0 & (1 << reg.encoding())) != 0
    }

    /// Check if the set is empty.
    #[inline(always)]
    pub const fn is_empty(self) -> bool {
        self.0 == 0
    }

    /// Add a register to the set.
    #[inline(always)]
    pub const fn insert(self, reg: Kreg) -> Self {
        KregSet(self.0 | (1 << reg.encoding()))
    }

    /// Remove a register from the set.
    #[inline(always)]
    pub const fn remove(self, reg: Kreg) -> Self {
        KregSet(self.0 & !(1 << reg.encoding()))
    }

    /// Union of two sets.
    #[inline(always)]
    pub const fn union(self, other: KregSet) -> Self {
        KregSet(self.0 | other.0)
    }

    /// Intersection of two sets.
    #[inline(always)]
    pub const fn intersection(self, other: KregSet) -> Self {
        KregSet(self.0 & other.0)
    }

    /// Difference (self - other).
    #[inline(always)]
    pub const fn difference(self, other: KregSet) -> Self {
        KregSet(self.0 & !other.0)
    }

    /// Count the number of registers in the set.
    #[inline(always)]
    pub const fn count(self) -> u32 {
        self.0.count_ones()
    }

    /// Get the first register (lowest encoding) in the set, if any.
    #[inline]
    pub const fn first(self) -> Option<Kreg> {
        if self.0 == 0 {
            None
        } else {
            Kreg::from_encoding(self.0.trailing_zeros() as u8)
        }
    }

    /// Iterate over registers in the set (ascending order).
    pub fn iter(self) -> impl Iterator<Item = Kreg> {
        (0..8).filter_map(move |i| {
            if (self.0 & (1 << i)) != 0 {
                Kreg::from_encoding(i)
            } else {
                None
            }
        })
    }
}

impl std::fmt::Debug for KregSet {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "KregSet{{")?;
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
// Vector Calling Convention
// =============================================================================

/// Comprehensive calling convention for x64 vector operations.
///
/// This structure extends the base `CallingConvention` with full support for:
/// - YMM (256-bit AVX/AVX2) registers
/// - ZMM (512-bit AVX-512) registers
/// - AVX-512 mask registers (k0-k7)
/// - Call-site clobber generation for register allocation
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VectorCallingConvention {
    /// Base calling convention (Windows x64 or System V).
    pub base: CallingConvention,
}

impl VectorCallingConvention {
    /// Windows x64 calling convention.
    pub const WINDOWS: VectorCallingConvention = VectorCallingConvention {
        base: CallingConvention::WindowsX64,
    };

    /// System V (Linux/macOS) calling convention.
    pub const SYSV: VectorCallingConvention = VectorCallingConvention {
        base: CallingConvention::SystemV,
    };

    /// Get the calling convention for the current platform.
    #[cfg(target_os = "windows")]
    pub const fn host() -> Self {
        Self::WINDOWS
    }

    #[cfg(not(target_os = "windows"))]
    pub const fn host() -> Self {
        Self::SYSV
    }

    /// Create from a base calling convention.
    #[inline]
    pub const fn from_base(base: CallingConvention) -> Self {
        VectorCallingConvention { base }
    }

    // =========================================================================
    // Argument Passing
    // =========================================================================

    /// Get integer argument registers (delegates to base).
    #[inline]
    pub const fn int_arg_regs(&self) -> &'static [Gpr] {
        self.base.int_arg_regs()
    }

    /// Get XMM (128-bit) argument registers (delegates to base).
    #[inline]
    pub const fn xmm_arg_regs(&self) -> &'static [Xmm] {
        self.base.float_arg_regs()
    }

    /// Get YMM (256-bit) argument registers.
    ///
    /// YMM registers alias XMM registers, so the same registers are used.
    /// The number of vector arguments is limited by the calling convention.
    pub const fn ymm_arg_regs(&self) -> &'static [Ymm] {
        match self.base {
            CallingConvention::WindowsX64 => {
                // Windows: First 4 vector args in YMM0-YMM3 (parallel with XMM)
                &[Ymm::Ymm0, Ymm::Ymm1, Ymm::Ymm2, Ymm::Ymm3]
            }
            CallingConvention::SystemV => {
                // System V: Up to 8 vector args in YMM0-YMM7
                &[
                    Ymm::Ymm0,
                    Ymm::Ymm1,
                    Ymm::Ymm2,
                    Ymm::Ymm3,
                    Ymm::Ymm4,
                    Ymm::Ymm5,
                    Ymm::Ymm6,
                    Ymm::Ymm7,
                ]
            }
        }
    }

    /// Get ZMM (512-bit) argument registers.
    ///
    /// ZMM registers alias XMM/YMM registers, using the same positions.
    pub const fn zmm_arg_regs(&self) -> &'static [Zmm] {
        match self.base {
            CallingConvention::WindowsX64 => {
                // Windows: First 4 vector args in ZMM0-ZMM3
                &[Zmm::Zmm0, Zmm::Zmm1, Zmm::Zmm2, Zmm::Zmm3]
            }
            CallingConvention::SystemV => {
                // System V: Up to 8 vector args in ZMM0-ZMM7
                &[
                    Zmm::Zmm0,
                    Zmm::Zmm1,
                    Zmm::Zmm2,
                    Zmm::Zmm3,
                    Zmm::Zmm4,
                    Zmm::Zmm5,
                    Zmm::Zmm6,
                    Zmm::Zmm7,
                ]
            }
        }
    }

    /// Get the maximum number of vector arguments passed in registers.
    #[inline]
    pub const fn max_vector_args(&self) -> usize {
        match self.base {
            CallingConvention::WindowsX64 => 4,
            CallingConvention::SystemV => 8,
        }
    }

    // =========================================================================
    // Return Values
    // =========================================================================

    /// Get the integer return register.
    #[inline]
    pub const fn int_return_reg(&self) -> Gpr {
        self.base.int_return_reg()
    }

    /// Get the XMM return register for scalar floats.
    #[inline]
    pub const fn xmm_return_reg(&self) -> Xmm {
        Xmm::Xmm0
    }

    /// Get the YMM return register for 256-bit vectors.
    #[inline]
    pub const fn ymm_return_reg(&self) -> Ymm {
        Ymm::Ymm0
    }

    /// Get the ZMM return register for 512-bit vectors.
    #[inline]
    pub const fn zmm_return_reg(&self) -> Zmm {
        Zmm::Zmm0
    }

    /// Get all XMM registers used for return values (System V can use 2).
    pub const fn xmm_return_regs(&self) -> &'static [Xmm] {
        match self.base {
            CallingConvention::WindowsX64 => &[Xmm::Xmm0],
            CallingConvention::SystemV => &[Xmm::Xmm0, Xmm::Xmm1],
        }
    }

    /// Get all YMM registers used for return values.
    pub const fn ymm_return_regs(&self) -> &'static [Ymm] {
        match self.base {
            CallingConvention::WindowsX64 => &[Ymm::Ymm0],
            CallingConvention::SystemV => &[Ymm::Ymm0, Ymm::Ymm1],
        }
    }

    // =========================================================================
    // Volatile (Caller-Saved) Registers
    // =========================================================================

    /// Get volatile (caller-saved) GPRs.
    #[inline]
    pub const fn volatile_gprs(&self) -> GprSet {
        self.base.volatile_gprs()
    }

    /// Get volatile (caller-saved) XMM registers.
    #[inline]
    pub const fn volatile_xmms(&self) -> XmmSet {
        self.base.volatile_xmms()
    }

    /// Get volatile (caller-saved) YMM registers.
    ///
    /// On Windows: YMM0-YMM5 have their lower 128 bits preserved (XMM portion),
    /// but the upper 128 bits are always volatile. YMM6-YMM15's upper bits
    /// are also volatile.
    ///
    /// On System V: All YMM registers are volatile.
    pub const fn volatile_ymms(&self) -> YmmSet {
        match self.base {
            // Windows: upper bits of all YMMs are volatile
            // But XMM6-15 lower bits are preserved, making full YMM6-15 volatile
            CallingConvention::WindowsX64 => YmmSet::ALL,
            CallingConvention::SystemV => YmmSet::ALL,
        }
    }

    /// Get volatile (caller-saved) ZMM registers.
    ///
    /// All upper bits beyond XMM/YMM are always volatile on both ABIs.
    /// Additionally, ZMM16-ZMM31 (extended registers) are always volatile.
    pub const fn volatile_zmms(&self) -> ZmmSet {
        // All ZMM registers are volatile (both ABIs)
        // The upper 384 bits are always clobbered
        ZmmSet::ALL
    }

    /// Get volatile (caller-saved) mask registers.
    ///
    /// k0 is reserved (not truly caller-saved, always "no mask").
    /// k1-k7 are all caller-saved on both ABIs.
    pub const fn volatile_kregs(&self) -> KregSet {
        // k1-k7 are volatile (k0 is reserved/special)
        KregSet::ALLOCATABLE
    }

    // =========================================================================
    // Callee-Saved (Non-Volatile) Registers
    // =========================================================================

    /// Get callee-saved GPRs.
    #[inline]
    pub const fn callee_saved_gprs(&self) -> GprSet {
        self.base.callee_saved_gprs()
    }

    /// Get callee-saved XMM registers.
    #[inline]
    pub const fn callee_saved_xmms(&self) -> XmmSet {
        self.base.callee_saved_xmms()
    }

    /// Get callee-saved YMM registers.
    ///
    /// Windows: The lower 128 bits (XMM portion) of XMM6-XMM15 are preserved.
    ///          The upper 128 bits are NOT preserved (volatile).
    ///          Therefore, NO YMM registers are fully callee-saved.
    ///          
    /// System V: No vector registers are callee-saved.
    pub const fn callee_saved_ymms(&self) -> YmmSet {
        // No YMM registers are fully callee-saved on any ABI
        // Windows only preserves the lower XMM portion
        YmmSet::EMPTY
    }

    /// Get callee-saved ZMM registers.
    pub const fn callee_saved_zmms(&self) -> ZmmSet {
        // No ZMM registers are callee-saved
        ZmmSet::EMPTY
    }

    /// Get callee-saved mask registers.
    pub const fn callee_saved_kregs(&self) -> KregSet {
        // No mask registers are callee-saved
        KregSet::EMPTY
    }

    // =========================================================================
    // Call Clobber Sets
    // =========================================================================

    /// Generate the complete set of registers clobbered by a call.
    ///
    /// This is the primary API for the register allocator to determine which
    /// values must be spilled before a call instruction.
    pub fn call_clobbers(&self) -> CallClobbers {
        CallClobbers {
            gprs: self.volatile_gprs(),
            xmms: self.volatile_xmms(),
            ymms: self.volatile_ymms(),
            zmms: self.volatile_zmms(),
            kregs: self.volatile_kregs(),
        }
    }

    /// Generate clobbers as a vector of PReg for the constraint system.
    ///
    /// This allocates but is convenient for one-shot constraint building.
    pub fn call_clobbers_as_pregs(&self) -> Vec<PReg> {
        let clobbers = self.call_clobbers();
        let mut result = Vec::with_capacity(64);

        // GPRs
        for gpr in clobbers.gprs.iter() {
            result.push(PReg::Gpr(gpr));
        }

        // XMMs
        for xmm in clobbers.xmms.iter() {
            result.push(PReg::Xmm(xmm));
        }

        // YMMs (these overlap with XMMs, so we might skip to avoid redundancy)
        // However, for width-aware allocation, we include them
        for ymm in clobbers.ymms.iter() {
            result.push(PReg::Ymm(ymm));
        }

        // ZMMs
        for zmm in clobbers.zmms.iter() {
            result.push(PReg::Zmm(zmm));
        }

        result
    }

    /// Count the total number of caller-saved vector registers.
    #[inline]
    pub const fn volatile_vector_count(&self) -> u32 {
        self.volatile_xmms().count()
            + self.volatile_ymms().count()
            + self.volatile_kregs().count() as u32
    }

    // =========================================================================
    // Stack Parameters
    // =========================================================================

    /// Get the stack alignment requirement in bytes.
    #[inline]
    pub const fn stack_alignment(&self) -> usize {
        self.base.stack_alignment()
    }

    /// Get the shadow space size (Windows only).
    #[inline]
    pub const fn shadow_space(&self) -> usize {
        self.base.shadow_space()
    }

    /// Get the red zone size (System V only).
    #[inline]
    pub const fn red_zone(&self) -> usize {
        self.base.red_zone()
    }

    /// Get the alignment for 256-bit vector arguments on stack.
    #[inline]
    pub const fn ymm_stack_alignment(&self) -> usize {
        32 // YMM requires 32-byte alignment
    }

    /// Get the alignment for 512-bit vector arguments on stack.
    #[inline]
    pub const fn zmm_stack_alignment(&self) -> usize {
        64 // ZMM requires 64-byte alignment
    }

    // =========================================================================
    // Register Width Queries
    // =========================================================================

    /// Check if the upper 128 bits of YMM registers need saving on Windows.
    ///
    /// This is useful for functions that want to preserve full YMM state
    /// across calls on Windows (which only preserves XMM portion).
    #[inline]
    pub const fn needs_ymm_upper_save(&self) -> bool {
        match self.base {
            CallingConvention::WindowsX64 => true,
            CallingConvention::SystemV => false, // All volatile anyway
        }
    }

    /// Get the number of XMM registers that need saving on Windows.
    #[inline]
    pub const fn windows_xmm_save_count(&self) -> usize {
        match self.base {
            CallingConvention::WindowsX64 => 10, // XMM6-XMM15
            CallingConvention::SystemV => 0,
        }
    }
}

// =============================================================================
// Call Clobbers
// =============================================================================

/// Complete set of registers clobbered by a function call.
///
/// This is used by the register allocator to ensure values are properly
/// spilled before call instructions.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CallClobbers {
    /// Clobbered GPRs.
    pub gprs: GprSet,
    /// Clobbered XMM registers.
    pub xmms: XmmSet,
    /// Clobbered YMM registers.
    pub ymms: YmmSet,
    /// Clobbered ZMM registers.
    pub zmms: ZmmSet,
    /// Clobbered mask registers.
    pub kregs: KregSet,
}

impl CallClobbers {
    /// Create an empty clobber set.
    pub const fn empty() -> Self {
        CallClobbers {
            gprs: GprSet::EMPTY,
            xmms: XmmSet::EMPTY,
            ymms: YmmSet::EMPTY,
            zmms: ZmmSet::EMPTY,
            kregs: KregSet::EMPTY,
        }
    }

    /// Check if a GPR is clobbered.
    #[inline]
    pub const fn clobbers_gpr(&self, gpr: Gpr) -> bool {
        self.gprs.contains(gpr)
    }

    /// Check if an XMM register is clobbered.
    #[inline]
    pub const fn clobbers_xmm(&self, xmm: Xmm) -> bool {
        self.xmms.contains(xmm)
    }

    /// Check if a YMM register is clobbered.
    #[inline]
    pub const fn clobbers_ymm(&self, ymm: Ymm) -> bool {
        self.ymms.contains(ymm)
    }

    /// Check if a ZMM register is clobbered.
    #[inline]
    pub const fn clobbers_zmm(&self, zmm: Zmm) -> bool {
        self.zmms.contains(zmm)
    }

    /// Check if a mask register is clobbered.
    #[inline]
    pub const fn clobbers_kreg(&self, kreg: Kreg) -> bool {
        self.kregs.contains(kreg)
    }

    /// Check if any portion of a physical register is clobbered.
    #[inline]
    pub fn clobbers_preg(&self, preg: PReg) -> bool {
        match preg {
            PReg::Gpr(g) => self.clobbers_gpr(g),
            PReg::Xmm(x) => self.clobbers_xmm(x),
            PReg::Ymm(y) => self.clobbers_ymm(y),
            PReg::Zmm(z) => self.clobbers_zmm(z),
        }
    }

    /// Get the total number of clobbered registers.
    pub const fn total_count(&self) -> u32 {
        self.gprs.count()
            + self.xmms.count()
            + self.ymms.count()
            + self.zmms.count()
            + self.kregs.count() as u32
    }

    /// Iterate over all clobbered GPRs.
    pub fn iter_gprs(&self) -> impl Iterator<Item = Gpr> + '_ {
        self.gprs.iter()
    }

    /// Iterate over all clobbered XMM registers.
    pub fn iter_xmms(&self) -> impl Iterator<Item = Xmm> + '_ {
        self.xmms.iter()
    }

    /// Iterate over all clobbered YMM registers.
    pub fn iter_ymms(&self) -> impl Iterator<Item = Ymm> + '_ {
        self.ymms.iter()
    }

    /// Iterate over all clobbered ZMM registers.
    pub fn iter_zmms(&self) -> impl Iterator<Item = Zmm> + '_ {
        self.zmms.iter()
    }
}

// =============================================================================
// Argument Slot Classification
// =============================================================================

/// Classification for how a function argument should be passed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArgClass {
    /// Pass in a GPR.
    Gpr(Gpr),
    /// Pass in an XMM register (scalar float or 128-bit vector).
    Xmm(Xmm),
    /// Pass in a YMM register (256-bit vector).
    Ymm(Ymm),
    /// Pass in a ZMM register (512-bit vector).
    Zmm(Zmm),
    /// Pass on the stack at the given offset from RSP.
    Stack(i32),
}

/// Classification for a function's return value.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RetClass {
    /// Return in a GPR.
    Gpr(Gpr),
    /// Return in XMM0 (or XMM0:XMM1 for two-register returns).
    Xmm(Xmm),
    /// Return in YMM0.
    Ymm(Ymm),
    /// Return in ZMM0.
    Zmm(Zmm),
    /// Return via pointer in first argument (struct returns).
    Memory,
}

// =============================================================================
// Argument Location Calculator
// =============================================================================

/// Calculator for determining argument locations per the calling convention.
#[derive(Debug, Clone)]
pub struct ArgLocationCalc {
    /// The calling convention to use.
    cc: VectorCallingConvention,
    /// Current GPR argument index.
    gpr_idx: usize,
    /// Current vector register argument index.
    vec_idx: usize,
    /// Current stack offset for stack arguments.
    stack_offset: i32,
}

impl ArgLocationCalc {
    /// Create a new calculator for the host calling convention.
    pub fn new() -> Self {
        Self::with_convention(VectorCallingConvention::host())
    }

    /// Create a new calculator with a specific calling convention.
    pub fn with_convention(cc: VectorCallingConvention) -> Self {
        ArgLocationCalc {
            cc,
            gpr_idx: 0,
            vec_idx: 0,
            stack_offset: 0,
        }
    }

    /// Reset the calculator for reuse.
    pub fn reset(&mut self) {
        self.gpr_idx = 0;
        self.vec_idx = 0;
        self.stack_offset = 0;
    }

    /// Get the location for an integer argument.
    pub fn next_int(&mut self) -> ArgClass {
        let regs = self.cc.int_arg_regs();
        if self.gpr_idx < regs.len() {
            let reg = regs[self.gpr_idx];
            self.gpr_idx += 1;
            // On Windows, GPR and vector args share slots
            if matches!(self.cc.base, CallingConvention::WindowsX64) {
                self.vec_idx = self.gpr_idx;
            }
            ArgClass::Gpr(reg)
        } else {
            let offset = self.stack_offset;
            self.stack_offset += 8;
            ArgClass::Stack(offset)
        }
    }

    /// Get the location for a 64-bit float argument.
    pub fn next_f64(&mut self) -> ArgClass {
        let regs = self.cc.xmm_arg_regs();
        if self.vec_idx < regs.len() {
            let reg = regs[self.vec_idx];
            self.vec_idx += 1;
            // On Windows, GPR and vector args share slots
            if matches!(self.cc.base, CallingConvention::WindowsX64) {
                self.gpr_idx = self.vec_idx;
            }
            ArgClass::Xmm(reg)
        } else {
            let offset = self.stack_offset;
            self.stack_offset += 8;
            ArgClass::Stack(offset)
        }
    }

    /// Get the location for a 256-bit vector argument.
    pub fn next_v256(&mut self) -> ArgClass {
        let regs = self.cc.ymm_arg_regs();
        if self.vec_idx < regs.len() {
            let reg = regs[self.vec_idx];
            self.vec_idx += 1;
            if matches!(self.cc.base, CallingConvention::WindowsX64) {
                self.gpr_idx = self.vec_idx;
            }
            ArgClass::Ymm(reg)
        } else {
            // YMM on stack needs 32-byte alignment
            self.stack_offset = (self.stack_offset + 31) & !31;
            let offset = self.stack_offset;
            self.stack_offset += 32;
            ArgClass::Stack(offset)
        }
    }

    /// Get the location for a 512-bit vector argument.
    pub fn next_v512(&mut self) -> ArgClass {
        let regs = self.cc.zmm_arg_regs();
        if self.vec_idx < regs.len() {
            let reg = regs[self.vec_idx];
            self.vec_idx += 1;
            if matches!(self.cc.base, CallingConvention::WindowsX64) {
                self.gpr_idx = self.vec_idx;
            }
            ArgClass::Zmm(reg)
        } else {
            // ZMM on stack needs 64-byte alignment
            self.stack_offset = (self.stack_offset + 63) & !63;
            let offset = self.stack_offset;
            self.stack_offset += 64;
            ArgClass::Stack(offset)
        }
    }

    /// Get the total stack space needed for arguments.
    pub const fn stack_size(&self) -> i32 {
        // Align to 16 bytes as required by ABI
        (self.stack_offset + 15) & !15
    }
}

impl Default for ArgLocationCalc {
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

    // =========================================================================
    // Kreg Tests
    // =========================================================================

    #[test]
    fn test_kreg_encoding() {
        assert_eq!(Kreg::K0.encoding(), 0);
        assert_eq!(Kreg::K1.encoding(), 1);
        assert_eq!(Kreg::K7.encoding(), 7);
    }

    #[test]
    fn test_kreg_from_encoding() {
        assert_eq!(Kreg::from_encoding(0), Some(Kreg::K0));
        assert_eq!(Kreg::from_encoding(7), Some(Kreg::K7));
        assert_eq!(Kreg::from_encoding(8), None);
    }

    #[test]
    fn test_kreg_is_reserved() {
        assert!(Kreg::K0.is_reserved());
        assert!(!Kreg::K1.is_reserved());
        assert!(!Kreg::K7.is_reserved());
    }

    #[test]
    fn test_kreg_display() {
        assert_eq!(format!("{}", Kreg::K0), "k0");
        assert_eq!(format!("{}", Kreg::K5), "k5");
    }

    // =========================================================================
    // KregSet Tests
    // =========================================================================

    #[test]
    fn test_kreg_set_empty() {
        let set = KregSet::EMPTY;
        assert!(set.is_empty());
        assert_eq!(set.count(), 0);
        assert!(!set.contains(Kreg::K0));
    }

    #[test]
    fn test_kreg_set_all() {
        let set = KregSet::ALL;
        assert!(!set.is_empty());
        assert_eq!(set.count(), 8);
        for i in 0..8 {
            assert!(set.contains(Kreg::from_encoding(i).unwrap()));
        }
    }

    #[test]
    fn test_kreg_set_allocatable() {
        let set = KregSet::ALLOCATABLE;
        assert_eq!(set.count(), 7);
        assert!(!set.contains(Kreg::K0));
        for i in 1..8 {
            assert!(set.contains(Kreg::from_encoding(i).unwrap()));
        }
    }

    #[test]
    fn test_kreg_set_singleton() {
        let set = KregSet::singleton(Kreg::K3);
        assert_eq!(set.count(), 1);
        assert!(set.contains(Kreg::K3));
        assert!(!set.contains(Kreg::K2));
    }

    #[test]
    fn test_kreg_set_insert_remove() {
        let set = KregSet::EMPTY.insert(Kreg::K1).insert(Kreg::K5);
        assert_eq!(set.count(), 2);
        assert!(set.contains(Kreg::K1));
        assert!(set.contains(Kreg::K5));

        let removed = set.remove(Kreg::K1);
        assert_eq!(removed.count(), 1);
        assert!(!removed.contains(Kreg::K1));
        assert!(removed.contains(Kreg::K5));
    }

    #[test]
    fn test_kreg_set_union_intersection() {
        let a = KregSet::EMPTY.insert(Kreg::K1).insert(Kreg::K2);
        let b = KregSet::EMPTY.insert(Kreg::K2).insert(Kreg::K3);

        let union = a.union(b);
        assert_eq!(union.count(), 3);
        assert!(union.contains(Kreg::K1));
        assert!(union.contains(Kreg::K2));
        assert!(union.contains(Kreg::K3));

        let intersection = a.intersection(b);
        assert_eq!(intersection.count(), 1);
        assert!(intersection.contains(Kreg::K2));
    }

    #[test]
    fn test_kreg_set_iter() {
        let set = KregSet::EMPTY
            .insert(Kreg::K1)
            .insert(Kreg::K3)
            .insert(Kreg::K7);
        let regs: Vec<_> = set.iter().collect();
        assert_eq!(regs, vec![Kreg::K1, Kreg::K3, Kreg::K7]);
    }

    #[test]
    fn test_kreg_set_first() {
        let set = KregSet::EMPTY.insert(Kreg::K3).insert(Kreg::K1);
        assert_eq!(set.first(), Some(Kreg::K1));
        assert_eq!(KregSet::EMPTY.first(), None);
    }

    // =========================================================================
    // VectorCallingConvention Tests
    // =========================================================================

    #[test]
    fn test_vcc_host() {
        let vcc = VectorCallingConvention::host();
        // Should have valid argument registers
        assert!(!vcc.int_arg_regs().is_empty());
        assert!(!vcc.xmm_arg_regs().is_empty());
        assert!(!vcc.ymm_arg_regs().is_empty());
        assert!(!vcc.zmm_arg_regs().is_empty());
    }

    #[test]
    fn test_vcc_windows() {
        let vcc = VectorCallingConvention::WINDOWS;

        // Integer args: RCX, RDX, R8, R9
        assert_eq!(vcc.int_arg_regs().len(), 4);
        assert_eq!(vcc.int_arg_regs()[0], Gpr::Rcx);

        // Vector args: 4 registers
        assert_eq!(vcc.max_vector_args(), 4);
        assert_eq!(vcc.ymm_arg_regs()[0], Ymm::Ymm0);
        assert_eq!(vcc.zmm_arg_regs()[0], Zmm::Zmm0);

        // Shadow space
        assert_eq!(vcc.shadow_space(), 32);
        assert_eq!(vcc.red_zone(), 0);
    }

    #[test]
    fn test_vcc_sysv() {
        let vcc = VectorCallingConvention::SYSV;

        // Integer args: RDI, RSI, RDX, RCX, R8, R9
        assert_eq!(vcc.int_arg_regs().len(), 6);
        assert_eq!(vcc.int_arg_regs()[0], Gpr::Rdi);

        // Vector args: 8 registers
        assert_eq!(vcc.max_vector_args(), 8);
        assert_eq!(vcc.ymm_arg_regs().len(), 8);
        assert_eq!(vcc.zmm_arg_regs().len(), 8);

        // No shadow space, 128-byte red zone
        assert_eq!(vcc.shadow_space(), 0);
        assert_eq!(vcc.red_zone(), 128);
    }

    #[test]
    fn test_vcc_return_regs() {
        let vcc = VectorCallingConvention::host();

        assert_eq!(vcc.int_return_reg(), Gpr::Rax);
        assert_eq!(vcc.xmm_return_reg(), Xmm::Xmm0);
        assert_eq!(vcc.ymm_return_reg(), Ymm::Ymm0);
        assert_eq!(vcc.zmm_return_reg(), Zmm::Zmm0);
    }

    #[test]
    fn test_vcc_windows_return_regs() {
        let vcc = VectorCallingConvention::WINDOWS;
        // Windows returns in single register
        assert_eq!(vcc.xmm_return_regs().len(), 1);
        assert_eq!(vcc.ymm_return_regs().len(), 1);
    }

    #[test]
    fn test_vcc_sysv_return_regs() {
        let vcc = VectorCallingConvention::SYSV;
        // SysV can return in up to 2 registers
        assert_eq!(vcc.xmm_return_regs().len(), 2);
        assert_eq!(vcc.ymm_return_regs().len(), 2);
    }

    #[test]
    fn test_vcc_volatile_ymm_all() {
        // YMM upper bits are always volatile
        let windows = VectorCallingConvention::WINDOWS;
        let sysv = VectorCallingConvention::SYSV;

        assert_eq!(windows.volatile_ymms(), YmmSet::ALL);
        assert_eq!(sysv.volatile_ymms(), YmmSet::ALL);
    }

    #[test]
    fn test_vcc_volatile_zmm_all() {
        // ZMM registers are always volatile
        let windows = VectorCallingConvention::WINDOWS;
        let sysv = VectorCallingConvention::SYSV;

        assert_eq!(windows.volatile_zmms(), ZmmSet::ALL);
        assert_eq!(sysv.volatile_zmms(), ZmmSet::ALL);
    }

    #[test]
    fn test_vcc_no_callee_saved_ymm() {
        let windows = VectorCallingConvention::WINDOWS;
        let sysv = VectorCallingConvention::SYSV;

        assert_eq!(windows.callee_saved_ymms(), YmmSet::EMPTY);
        assert_eq!(sysv.callee_saved_ymms(), YmmSet::EMPTY);
    }

    #[test]
    fn test_vcc_volatile_kregs() {
        let vcc = VectorCallingConvention::host();
        let kregs = vcc.volatile_kregs();

        // k1-k7 are volatile, k0 is not counted
        assert_eq!(kregs.count(), 7);
        assert!(!kregs.contains(Kreg::K0));
        for i in 1..8 {
            assert!(kregs.contains(Kreg::from_encoding(i).unwrap()));
        }
    }

    #[test]
    fn test_vcc_stack_alignment() {
        let vcc = VectorCallingConvention::host();

        assert_eq!(vcc.stack_alignment(), 16);
        assert_eq!(vcc.ymm_stack_alignment(), 32);
        assert_eq!(vcc.zmm_stack_alignment(), 64);
    }

    #[test]
    fn test_vcc_windows_needs_ymm_upper_save() {
        let windows = VectorCallingConvention::WINDOWS;
        let sysv = VectorCallingConvention::SYSV;

        assert!(windows.needs_ymm_upper_save());
        assert!(!sysv.needs_ymm_upper_save());
    }

    // =========================================================================
    // CallClobbers Tests
    // =========================================================================

    #[test]
    fn test_call_clobbers_windows() {
        let vcc = VectorCallingConvention::WINDOWS;
        let clobbers = vcc.call_clobbers();

        // Windows volatile GPRs: RAX, RCX, RDX, R8-R11
        assert!(clobbers.clobbers_gpr(Gpr::Rax));
        assert!(clobbers.clobbers_gpr(Gpr::Rcx));
        assert!(clobbers.clobbers_gpr(Gpr::R11));
        assert!(!clobbers.clobbers_gpr(Gpr::Rbx)); // callee-saved
        assert!(!clobbers.clobbers_gpr(Gpr::R12)); // callee-saved

        // Windows volatile XMMs: XMM0-XMM5
        assert!(clobbers.clobbers_xmm(Xmm::Xmm0));
        assert!(clobbers.clobbers_xmm(Xmm::Xmm5));
        assert!(!clobbers.clobbers_xmm(Xmm::Xmm6)); // callee-saved on Windows

        // All YMMs clobbered (upper bits)
        assert!(clobbers.clobbers_ymm(Ymm::Ymm0));
        assert!(clobbers.clobbers_ymm(Ymm::Ymm15));

        // All ZMMs clobbered
        assert!(clobbers.clobbers_zmm(Zmm::Zmm0));
        assert!(clobbers.clobbers_zmm(Zmm::Zmm31));
    }

    #[test]
    fn test_call_clobbers_sysv() {
        let vcc = VectorCallingConvention::SYSV;
        let clobbers = vcc.call_clobbers();

        // SysV volatile GPRs: RAX, RCX, RDX, RSI, RDI, R8-R11
        assert!(clobbers.clobbers_gpr(Gpr::Rax));
        assert!(clobbers.clobbers_gpr(Gpr::Rdi));
        assert!(clobbers.clobbers_gpr(Gpr::Rsi));
        assert!(!clobbers.clobbers_gpr(Gpr::Rbx)); // callee-saved
        assert!(!clobbers.clobbers_gpr(Gpr::R12)); // callee-saved

        // SysV: ALL XMMs are volatile
        for i in 0..16 {
            assert!(clobbers.clobbers_xmm(Xmm::from_encoding(i).unwrap()));
        }
    }

    #[test]
    fn test_call_clobbers_empty() {
        let clobbers = CallClobbers::empty();
        assert_eq!(clobbers.total_count(), 0);
        assert!(!clobbers.clobbers_gpr(Gpr::Rax));
        assert!(!clobbers.clobbers_xmm(Xmm::Xmm0));
    }

    #[test]
    fn test_call_clobbers_preg() {
        let vcc = VectorCallingConvention::host();
        let clobbers = vcc.call_clobbers();

        // Test PReg queries
        assert!(clobbers.clobbers_preg(PReg::Gpr(Gpr::Rax)));
        assert!(clobbers.clobbers_preg(PReg::Xmm(Xmm::Xmm0)));
        assert!(clobbers.clobbers_preg(PReg::Ymm(Ymm::Ymm0)));
        assert!(clobbers.clobbers_preg(PReg::Zmm(Zmm::Zmm0)));
    }

    #[test]
    fn test_call_clobbers_as_pregs() {
        let vcc = VectorCallingConvention::host();
        let pregs = vcc.call_clobbers_as_pregs();

        // Should have a reasonable number of clobbers
        assert!(!pregs.is_empty());
        assert!(pregs.len() > 10); // At least some GPRs + XMMs + YMMs + ZMMs
    }

    // =========================================================================
    // ArgLocationCalc Tests
    // =========================================================================

    #[test]
    fn test_arg_calc_windows_ints() {
        let mut calc = ArgLocationCalc::with_convention(VectorCallingConvention::WINDOWS);

        // First 4 ints go in registers
        assert_eq!(calc.next_int(), ArgClass::Gpr(Gpr::Rcx));
        assert_eq!(calc.next_int(), ArgClass::Gpr(Gpr::Rdx));
        assert_eq!(calc.next_int(), ArgClass::Gpr(Gpr::R8));
        assert_eq!(calc.next_int(), ArgClass::Gpr(Gpr::R9));

        // 5th goes on stack
        if let ArgClass::Stack(offset) = calc.next_int() {
            assert!(offset >= 0);
        } else {
            panic!("Expected stack argument");
        }
    }

    #[test]
    fn test_arg_calc_sysv_ints() {
        let mut calc = ArgLocationCalc::with_convention(VectorCallingConvention::SYSV);

        // First 6 ints go in registers
        assert_eq!(calc.next_int(), ArgClass::Gpr(Gpr::Rdi));
        assert_eq!(calc.next_int(), ArgClass::Gpr(Gpr::Rsi));
        assert_eq!(calc.next_int(), ArgClass::Gpr(Gpr::Rdx));
        assert_eq!(calc.next_int(), ArgClass::Gpr(Gpr::Rcx));
        assert_eq!(calc.next_int(), ArgClass::Gpr(Gpr::R8));
        assert_eq!(calc.next_int(), ArgClass::Gpr(Gpr::R9));

        // 7th goes on stack
        assert!(matches!(calc.next_int(), ArgClass::Stack(_)));
    }

    #[test]
    fn test_arg_calc_vectors() {
        let mut calc = ArgLocationCalc::with_convention(VectorCallingConvention::SYSV);

        // 256-bit vector in YMM
        assert_eq!(calc.next_v256(), ArgClass::Ymm(Ymm::Ymm0));
        assert_eq!(calc.next_v256(), ArgClass::Ymm(Ymm::Ymm1));

        // 512-bit vector in ZMM (continues from where we left off)
        assert_eq!(calc.next_v512(), ArgClass::Zmm(Zmm::Zmm2));
    }

    #[test]
    fn test_arg_calc_mixed_windows() {
        let mut calc = ArgLocationCalc::with_convention(VectorCallingConvention::WINDOWS);

        // Windows: int and float args share slots
        assert_eq!(calc.next_int(), ArgClass::Gpr(Gpr::Rcx));
        assert_eq!(calc.next_f64(), ArgClass::Xmm(Xmm::Xmm1)); // Uses slot 2
        assert_eq!(calc.next_int(), ArgClass::Gpr(Gpr::R8)); // Uses slot 3
    }

    #[test]
    fn test_arg_calc_stack_alignment() {
        let mut calc = ArgLocationCalc::with_convention(VectorCallingConvention::SYSV);

        // Use up all vector regs
        for _ in 0..8 {
            calc.next_f64();
        }

        // Add a small stack arg first
        let _small = calc.next_int();

        // Reset and test YMM alignment
        calc.reset();
        for _ in 0..8 {
            calc.next_v256();
        }

        // Next YMM should be on stack with alignment
        if let ArgClass::Stack(offset) = calc.next_v256() {
            assert!(offset % 32 == 0, "YMM stack arg not 32-byte aligned");
        }
    }

    #[test]
    fn test_arg_calc_zmm_alignment() {
        let mut calc = ArgLocationCalc::with_convention(VectorCallingConvention::SYSV);

        // Use up all vector regs
        for _ in 0..8 {
            calc.next_v512();
        }

        // ZMM on stack should be 64-byte aligned
        if let ArgClass::Stack(offset) = calc.next_v512() {
            assert!(offset % 64 == 0, "ZMM stack arg not 64-byte aligned");
        }
    }

    #[test]
    fn test_arg_calc_reset() {
        let mut calc = ArgLocationCalc::new();

        calc.next_int();
        calc.next_f64();
        calc.reset();

        // After reset, should start from beginning
        assert_eq!(calc.gpr_idx, 0);
        assert_eq!(calc.vec_idx, 0);
        assert_eq!(calc.stack_offset, 0);
    }

    #[test]
    fn test_arg_calc_stack_size_aligned() {
        let mut calc = ArgLocationCalc::with_convention(VectorCallingConvention::SYSV);

        // Force some stack arguments
        for _ in 0..8 {
            calc.next_int();
        }
        calc.next_int(); // 7th goes on stack (8 bytes)

        // Stack size should be 16-byte aligned
        let size = calc.stack_size();
        assert!(size % 16 == 0, "Stack size {} not 16-byte aligned", size);
    }

    // =========================================================================
    // RetClass Tests
    // =========================================================================

    #[test]
    fn test_ret_class_variants() {
        // Just ensure all variants work
        let _gpr = RetClass::Gpr(Gpr::Rax);
        let _xmm = RetClass::Xmm(Xmm::Xmm0);
        let _ymm = RetClass::Ymm(Ymm::Ymm0);
        let _zmm = RetClass::Zmm(Zmm::Zmm0);
        let _mem = RetClass::Memory;
    }

    // =========================================================================
    // Edge Case Tests
    // =========================================================================

    #[test]
    fn test_vcc_volatile_vector_count() {
        let vcc = VectorCallingConvention::host();
        let count = vcc.volatile_vector_count();

        // Should include XMMs + YMMs + kregs
        assert!(count > 20);
    }

    #[test]
    fn test_call_clobbers_iter() {
        let vcc = VectorCallingConvention::host();
        let clobbers = vcc.call_clobbers();

        // Ensure iterators work
        let gpr_count: usize = clobbers.iter_gprs().count();
        let xmm_count: usize = clobbers.iter_xmms().count();
        let ymm_count: usize = clobbers.iter_ymms().count();
        let zmm_count: usize = clobbers.iter_zmms().count();

        assert!(gpr_count > 0);
        assert!(xmm_count > 0);
        assert!(ymm_count > 0);
        assert!(zmm_count > 0);
    }

    #[test]
    fn test_kreg_set_debug_format() {
        let set = KregSet::EMPTY.insert(Kreg::K1).insert(Kreg::K3);
        let debug_str = format!("{:?}", set);
        assert!(debug_str.contains("k1"));
        assert!(debug_str.contains("k3"));
    }
}
