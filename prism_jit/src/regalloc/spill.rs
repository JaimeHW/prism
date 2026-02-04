//! Spill Code Generation
//!
//! Handles spilling virtual registers to the stack when
//! there are not enough physical registers.
//!
//! # Design
//!
//! - Stack slots are allocated with width-aware sizing:
//!   - GPR: 8 bytes, 8-byte aligned
//!   - XMM: 16 bytes, 16-byte aligned  
//!   - YMM: 32 bytes, 32-byte aligned
//!   - ZMM: 64 bytes, 64-byte aligned
//! - Spill slots are relative to the frame pointer
//! - Reload/spill instructions are inserted at use/def points

use crate::backend::x64::registers::{Gpr, Xmm};
use crate::backend::x64::simd::{Ymm, Zmm};
use std::fmt;

// =============================================================================
// Spill Slot Width
// =============================================================================

/// Width of a spill slot, determining size and alignment.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum SpillWidth {
    /// 8 bytes (GPR, 64-bit)
    W8 = 8,
    /// 16 bytes (XMM, 128-bit)
    W16 = 16,
    /// 32 bytes (YMM, 256-bit)
    W32 = 32,
    /// 64 bytes (ZMM, 512-bit)
    W64 = 64,
}

impl SpillWidth {
    /// Get the size in bytes.
    #[inline]
    pub const fn size(self) -> u32 {
        self as u32
    }

    /// Get the required alignment in bytes.
    #[inline]
    pub const fn alignment(self) -> u32 {
        // Vector spills require alignment equal to their size
        self as u32
    }

    /// Get the width for a register class.
    #[inline]
    pub const fn from_reg_class(class: super::RegClass) -> Self {
        match class {
            super::RegClass::Int | super::RegClass::Any => SpillWidth::W8,
            super::RegClass::Float => SpillWidth::W16,
            super::RegClass::Vec256 => SpillWidth::W32,
            super::RegClass::Vec512 => SpillWidth::W64,
        }
    }
}

// =============================================================================
// Spill Slot
// =============================================================================

/// A stack slot for storing spilled values.
///
/// Supports variable-width slots for different register classes:
/// - GPR: 8 bytes at 8-byte alignment
/// - XMM: 16 bytes at 16-byte alignment
/// - YMM: 32 bytes at 32-byte alignment
/// - ZMM: 64 bytes at 64-byte alignment
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SpillSlot {
    /// Byte offset from frame pointer (negative, grows downward).
    offset: i32,
    /// Width of this slot.
    width: SpillWidth,
}

impl SpillSlot {
    /// Create a new spill slot with explicit offset and width.
    #[inline]
    pub const fn new_with_offset(offset: i32, width: SpillWidth) -> Self {
        SpillSlot { offset, width }
    }

    /// Create a legacy 8-byte spill slot (for backward compatibility).
    #[inline]
    pub const fn new(index: u32) -> Self {
        SpillSlot {
            offset: -((index as i32 + 1) * 8),
            width: SpillWidth::W8,
        }
    }

    /// Get the slot index (only valid for legacy 8-byte slots).
    #[inline]
    pub const fn index(self) -> u32 {
        // Reverse the offset calculation for legacy compatibility
        ((-self.offset / 8) - 1) as u32
    }

    /// Get the stack offset from the frame pointer.
    /// Spill slots grow downward from RBP (negative offsets).
    #[inline]
    pub const fn offset(self) -> i32 {
        self.offset
    }

    /// Get the width of this spill slot.
    #[inline]
    pub const fn width(self) -> SpillWidth {
        self.width
    }

    /// Get the size of this spill slot in bytes.
    #[inline]
    pub const fn size(self) -> u32 {
        self.width.size()
    }

    /// Get the required alignment for this slot.
    #[inline]
    pub const fn alignment(self) -> u32 {
        self.width.alignment()
    }

    /// Check if this is a vector spill slot (16+ bytes).
    #[inline]
    pub const fn is_vector(self) -> bool {
        self.width.size() >= 16
    }

    /// Check if this is a wide vector slot (32+ bytes).
    #[inline]
    pub const fn is_wide_vector(self) -> bool {
        self.width.size() >= 32
    }
}

impl fmt::Display for SpillSlot {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let width_suffix = match self.width {
            SpillWidth::W8 => "",
            SpillWidth::W16 => " (xmm)",
            SpillWidth::W32 => " (ymm)",
            SpillWidth::W64 => " (zmm)",
        };
        write!(f, "[rbp{}]{}", self.offset, width_suffix)
    }
}

// =============================================================================
// Spill Slot Allocator
// =============================================================================

/// Allocates spill slots with proper alignment for different widths.
///
/// Uses a bump allocator strategy that grows downward from RBP.
/// Automatically aligns slots to their required alignment.
#[derive(Debug, Clone, Default)]
pub struct SpillSlotAllocator {
    /// Current stack offset (negative, grows downward).
    current_offset: i32,
    /// Number of 8-byte slots allocated.
    num_gpr_slots: u32,
    /// Number of 16-byte slots allocated.
    num_xmm_slots: u32,
    /// Number of 32-byte slots allocated.
    num_ymm_slots: u32,
    /// Number of 64-byte slots allocated.
    num_zmm_slots: u32,
}

impl SpillSlotAllocator {
    /// Create a new spill slot allocator.
    pub const fn new() -> Self {
        SpillSlotAllocator {
            current_offset: 0,
            num_gpr_slots: 0,
            num_xmm_slots: 0,
            num_ymm_slots: 0,
            num_zmm_slots: 0,
        }
    }

    /// Allocate a spill slot with the specified width.
    ///
    /// Returns a slot with the correct size and alignment.
    /// Alignment is automatically handled by adjusting the offset.
    pub fn alloc(&mut self, width: SpillWidth) -> SpillSlot {
        let size = width.size() as i32;
        let align = width.alignment() as i32;

        // Compute new offset (more negative = further down stack)
        let new_offset = self.current_offset - size;

        // Align by masking off low bits (makes more negative if needed)
        // For negative offsets, we need to align toward more negative values
        let aligned_offset = new_offset & !(align - 1);

        self.current_offset = aligned_offset;

        // Track counts by width
        match width {
            SpillWidth::W8 => self.num_gpr_slots += 1,
            SpillWidth::W16 => self.num_xmm_slots += 1,
            SpillWidth::W32 => self.num_ymm_slots += 1,
            SpillWidth::W64 => self.num_zmm_slots += 1,
        }

        SpillSlot::new_with_offset(aligned_offset, width)
    }

    /// Allocate a legacy 8-byte slot.
    #[inline]
    pub fn alloc_gpr(&mut self) -> SpillSlot {
        self.alloc(SpillWidth::W8)
    }

    /// Allocate a 16-byte XMM slot.
    #[inline]
    pub fn alloc_xmm(&mut self) -> SpillSlot {
        self.alloc(SpillWidth::W16)
    }

    /// Allocate a 32-byte YMM slot.
    #[inline]
    pub fn alloc_ymm(&mut self) -> SpillSlot {
        self.alloc(SpillWidth::W32)
    }

    /// Allocate a 64-byte ZMM slot.
    #[inline]
    pub fn alloc_zmm(&mut self) -> SpillSlot {
        self.alloc(SpillWidth::W64)
    }

    /// Allocate a slot for a register class.
    #[inline]
    pub fn alloc_for_class(&mut self, class: super::RegClass) -> SpillSlot {
        self.alloc(SpillWidth::from_reg_class(class))
    }

    /// Get the total stack space used (positive value).
    #[inline]
    pub const fn total_size(&self) -> u32 {
        (-self.current_offset) as u32
    }

    /// Get the number of GPR slots allocated.
    #[inline]
    pub const fn num_gpr_slots(&self) -> u32 {
        self.num_gpr_slots
    }

    /// Get the number of XMM slots allocated.
    #[inline]
    pub const fn num_xmm_slots(&self) -> u32 {
        self.num_xmm_slots
    }

    /// Get the number of YMM slots allocated.
    #[inline]
    pub const fn num_ymm_slots(&self) -> u32 {
        self.num_ymm_slots
    }

    /// Get the number of ZMM slots allocated.
    #[inline]
    pub const fn num_zmm_slots(&self) -> u32 {
        self.num_zmm_slots
    }

    /// Get total number of slots allocated.
    #[inline]
    pub const fn total_slots(&self) -> u32 {
        self.num_gpr_slots + self.num_xmm_slots + self.num_ymm_slots + self.num_zmm_slots
    }

    /// Reset the allocator to initial state.
    pub fn reset(&mut self) {
        self.current_offset = 0;
        self.num_gpr_slots = 0;
        self.num_xmm_slots = 0;
        self.num_ymm_slots = 0;
        self.num_zmm_slots = 0;
    }
}

// =============================================================================
// Spill Location
// =============================================================================

/// Location where a spill/reload should occur.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpillLocation {
    /// Before an instruction.
    Before(u32),
    /// After an instruction.
    After(u32),
}

// =============================================================================
// Spill Code
// =============================================================================

/// Spill code to be inserted.
#[derive(Debug, Clone)]
pub struct SpillCode {
    /// Spill operations (store to stack).
    spills: Vec<SpillOp>,
    /// Reload operations (load from stack).
    reloads: Vec<ReloadOp>,
    /// Total stack space needed for spill slots.
    total_spill_size: u32,
}

/// A spill operation (store value to stack).
#[derive(Debug, Clone)]
pub struct SpillOp {
    /// Where to insert the spill.
    pub location: SpillLocation,
    /// The spill slot to store to.
    pub slot: SpillSlot,
    /// Data source (register or virtual register).
    pub source: SpillSource,
}

/// A reload operation (load value from stack).
#[derive(Debug, Clone)]
pub struct ReloadOp {
    /// Where to insert the reload.
    pub location: SpillLocation,
    /// The spill slot to load from.
    pub slot: SpillSlot,
    /// Destination register.
    pub dest: ReloadDest,
}

/// Source of data for a spill.
#[derive(Debug, Clone, Copy)]
pub enum SpillSource {
    /// From a GPR (8 bytes).
    Gpr(Gpr),
    /// From an XMM register (16 bytes).
    Xmm(Xmm),
    /// From a YMM register (32 bytes).
    Ymm(Ymm),
    /// From a ZMM register (64 bytes).
    Zmm(Zmm),
}

impl SpillSource {
    /// Get the width of this spill source.
    #[inline]
    pub const fn width(self) -> SpillWidth {
        match self {
            SpillSource::Gpr(_) => SpillWidth::W8,
            SpillSource::Xmm(_) => SpillWidth::W16,
            SpillSource::Ymm(_) => SpillWidth::W32,
            SpillSource::Zmm(_) => SpillWidth::W64,
        }
    }
}

/// Destination for a reload.
#[derive(Debug, Clone, Copy)]
pub enum ReloadDest {
    /// To a GPR (8 bytes).
    Gpr(Gpr),
    /// To an XMM register (16 bytes).
    Xmm(Xmm),
    /// To a YMM register (32 bytes).
    Ymm(Ymm),
    /// To a ZMM register (64 bytes).
    Zmm(Zmm),
}

impl ReloadDest {
    /// Get the width of this reload destination.
    #[inline]
    pub const fn width(self) -> SpillWidth {
        match self {
            ReloadDest::Gpr(_) => SpillWidth::W8,
            ReloadDest::Xmm(_) => SpillWidth::W16,
            ReloadDest::Ymm(_) => SpillWidth::W32,
            ReloadDest::Zmm(_) => SpillWidth::W64,
        }
    }
}

impl SpillCode {
    /// Create a new empty spill code container.
    pub fn new() -> Self {
        SpillCode {
            spills: Vec::new(),
            reloads: Vec::new(),
            total_spill_size: 0,
        }
    }

    /// Add a spill operation.
    pub fn add_spill(&mut self, location: SpillLocation, slot: SpillSlot, source: SpillSource) {
        // Track maximum offset for size calculation
        let slot_end = ((-slot.offset()) as u32) + slot.size();
        if slot_end > self.total_spill_size {
            self.total_spill_size = slot_end;
        }

        self.spills.push(SpillOp {
            location,
            slot,
            source,
        });
    }

    /// Add a reload operation.
    pub fn add_reload(&mut self, location: SpillLocation, slot: SpillSlot, dest: ReloadDest) {
        let slot_end = ((-slot.offset()) as u32) + slot.size();
        if slot_end > self.total_spill_size {
            self.total_spill_size = slot_end;
        }

        self.reloads.push(ReloadOp {
            location,
            slot,
            dest,
        });
    }

    /// Get all spill operations.
    pub fn spills(&self) -> &[SpillOp] {
        &self.spills
    }

    /// Get all reload operations.
    pub fn reloads(&self) -> &[ReloadOp] {
        &self.reloads
    }

    /// Get the total stack space needed for spill slots.
    pub fn total_spill_size(&self) -> u32 {
        self.total_spill_size
    }

    /// Check if there are any spill/reload operations.
    pub fn is_empty(&self) -> bool {
        self.spills.is_empty() && self.reloads.is_empty()
    }

    /// Get the number of spill operations.
    pub fn num_spills(&self) -> usize {
        self.spills.len()
    }

    /// Get the number of reload operations.
    pub fn num_reloads(&self) -> usize {
        self.reloads.len()
    }
}

impl Default for SpillCode {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// Spill Weight Calculator
// =============================================================================

/// Calculates spill weights for intervals.
pub struct SpillWeightCalculator {
    /// Weight for definitions.
    def_weight: f32,
    /// Weight for uses.
    use_weight: f32,
    /// Multiplier for loop nesting.
    loop_weight: f32,
}

impl SpillWeightCalculator {
    /// Create a new calculator with default weights.
    pub fn new() -> Self {
        SpillWeightCalculator {
            def_weight: 1.0,
            use_weight: 1.0,
            loop_weight: 10.0,
        }
    }

    /// Create with custom loop weight.
    pub fn with_loop_weight(loop_weight: f32) -> Self {
        SpillWeightCalculator {
            def_weight: 1.0,
            use_weight: 1.0,
            loop_weight,
        }
    }

    /// Calculate spill weight for an interval.
    /// Higher weight = less desirable to spill.
    pub fn calculate(&self, interval: &super::interval::LiveInterval, loop_depths: &[u8]) -> f32 {
        let mut weight = 0.0f32;

        for use_pos in interval.uses() {
            let inst_idx = use_pos.pos.inst_index() as usize;
            let loop_depth = loop_depths.get(inst_idx).copied().unwrap_or(0);
            let loop_factor = self.loop_weight.powi(loop_depth as i32);

            match use_pos.kind {
                super::interval::UseKind::Def | super::interval::UseKind::PhiOutput => {
                    weight += self.def_weight * loop_factor;
                }
                super::interval::UseKind::Use
                | super::interval::UseKind::DefUse
                | super::interval::UseKind::PhiInput => {
                    weight += self.use_weight * loop_factor;
                }
            }
        }

        // Normalize by interval length to favor spilling long-lived values
        let length = interval.total_len().max(1) as f32;
        weight / length
    }
}

impl Default for SpillWeightCalculator {
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
    fn test_spill_slot_offset_legacy() {
        assert_eq!(SpillSlot::new(0).offset(), -8);
        assert_eq!(SpillSlot::new(1).offset(), -16);
        assert_eq!(SpillSlot::new(2).offset(), -24);
    }

    #[test]
    fn test_spill_slot_width() {
        let gpr_slot = SpillSlot::new_with_offset(-8, SpillWidth::W8);
        let xmm_slot = SpillSlot::new_with_offset(-16, SpillWidth::W16);
        let ymm_slot = SpillSlot::new_with_offset(-32, SpillWidth::W32);
        let zmm_slot = SpillSlot::new_with_offset(-64, SpillWidth::W64);

        assert_eq!(gpr_slot.size(), 8);
        assert_eq!(xmm_slot.size(), 16);
        assert_eq!(ymm_slot.size(), 32);
        assert_eq!(zmm_slot.size(), 64);

        assert!(!gpr_slot.is_vector());
        assert!(xmm_slot.is_vector());
        assert!(ymm_slot.is_vector());
        assert!(zmm_slot.is_vector());

        assert!(!gpr_slot.is_wide_vector());
        assert!(!xmm_slot.is_wide_vector());
        assert!(ymm_slot.is_wide_vector());
        assert!(zmm_slot.is_wide_vector());
    }

    #[test]
    fn test_spill_slot_allocator_basic() {
        let mut alloc = SpillSlotAllocator::new();

        let s1 = alloc.alloc_gpr();
        let s2 = alloc.alloc_gpr();
        let s3 = alloc.alloc_gpr();

        assert_eq!(s1.offset(), -8);
        assert_eq!(s2.offset(), -16);
        assert_eq!(s3.offset(), -24);
        assert_eq!(alloc.total_size(), 24);
        assert_eq!(alloc.num_gpr_slots(), 3);
    }

    #[test]
    fn test_spill_slot_allocator_ymm_alignment() {
        let mut alloc = SpillSlotAllocator::new();

        // Allocate a GPR first (offset becomes -8)
        let _gpr = alloc.alloc_gpr();

        // Now allocate YMM - must be 32-byte aligned
        let ymm = alloc.alloc_ymm();

        // YMM slot should be at offset that's a multiple of 32
        assert_eq!(ymm.offset() % 32, 0);
        assert!(ymm.offset() < -8); // Below the GPR slot
        assert_eq!(ymm.size(), 32);
    }

    #[test]
    fn test_spill_slot_allocator_zmm_alignment() {
        let mut alloc = SpillSlotAllocator::new();

        // Allocate a GPR first
        let _gpr = alloc.alloc_gpr();

        // Now allocate ZMM - must be 64-byte aligned
        let zmm = alloc.alloc_zmm();

        // ZMM slot should be at offset that's a multiple of 64
        assert_eq!(zmm.offset() % 64, 0);
        assert!(zmm.offset() < -8);
        assert_eq!(zmm.size(), 64);
    }

    #[test]
    fn test_spill_slot_allocator_mixed_widths() {
        let mut alloc = SpillSlotAllocator::new();

        let gpr1 = alloc.alloc_gpr(); // -8
        let xmm1 = alloc.alloc_xmm(); // -16 aligned, so -16 - 16 = -32 -> aligned to -32
        let gpr2 = alloc.alloc_gpr(); // -32 - 8 = -40
        let ymm1 = alloc.alloc_ymm(); // Must be 32-aligned
        let zmm1 = alloc.alloc_zmm(); // Must be 64-aligned

        // Verify all slots are within bounds
        assert!(gpr1.offset() < 0);
        assert!(xmm1.offset() < gpr1.offset());
        assert!(gpr2.offset() < xmm1.offset());
        assert!(ymm1.offset() < gpr2.offset());
        assert!(zmm1.offset() < ymm1.offset());

        // Verify alignments
        assert_eq!(xmm1.offset() % 16, 0);
        assert_eq!(ymm1.offset() % 32, 0);
        assert_eq!(zmm1.offset() % 64, 0);

        assert_eq!(alloc.num_gpr_slots(), 2);
        assert_eq!(alloc.num_xmm_slots(), 1);
        assert_eq!(alloc.num_ymm_slots(), 1);
        assert_eq!(alloc.num_zmm_slots(), 1);
    }

    #[test]
    fn test_spill_slot_allocator_for_class() {
        let mut alloc = SpillSlotAllocator::new();

        let s1 = alloc.alloc_for_class(super::super::RegClass::Int);
        let s2 = alloc.alloc_for_class(super::super::RegClass::Float);
        let s3 = alloc.alloc_for_class(super::super::RegClass::Vec256);
        let s4 = alloc.alloc_for_class(super::super::RegClass::Vec512);

        assert_eq!(s1.width(), SpillWidth::W8);
        assert_eq!(s2.width(), SpillWidth::W16);
        assert_eq!(s3.width(), SpillWidth::W32);
        assert_eq!(s4.width(), SpillWidth::W64);
    }

    #[test]
    fn test_spill_source_width() {
        assert_eq!(SpillSource::Gpr(Gpr::Rax).width(), SpillWidth::W8);
        assert_eq!(SpillSource::Xmm(Xmm::Xmm0).width(), SpillWidth::W16);
        assert_eq!(SpillSource::Ymm(Ymm::Ymm0).width(), SpillWidth::W32);
        assert_eq!(SpillSource::Zmm(Zmm::Zmm0).width(), SpillWidth::W64);
    }

    #[test]
    fn test_reload_dest_width() {
        assert_eq!(ReloadDest::Gpr(Gpr::Rax).width(), SpillWidth::W8);
        assert_eq!(ReloadDest::Xmm(Xmm::Xmm0).width(), SpillWidth::W16);
        assert_eq!(ReloadDest::Ymm(Ymm::Ymm0).width(), SpillWidth::W32);
        assert_eq!(ReloadDest::Zmm(Zmm::Zmm0).width(), SpillWidth::W64);
    }

    #[test]
    fn test_spill_code() {
        let mut code = SpillCode::new();
        let mut alloc = SpillSlotAllocator::new();

        let gpr_slot = alloc.alloc_gpr();
        let ymm_slot = alloc.alloc_ymm();

        code.add_spill(
            SpillLocation::After(5),
            gpr_slot,
            SpillSource::Gpr(Gpr::Rax),
        );
        code.add_reload(
            SpillLocation::Before(10),
            gpr_slot,
            ReloadDest::Gpr(Gpr::Rcx),
        );
        code.add_spill(
            SpillLocation::After(15),
            ymm_slot,
            SpillSource::Ymm(Ymm::Ymm0),
        );

        assert_eq!(code.num_spills(), 2);
        assert_eq!(code.num_reloads(), 1);
        assert!(code.total_spill_size() >= 40); // At least 8 + 32 bytes
    }

    #[test]
    fn test_spill_weight_basic() {
        use super::super::RegClass;
        use super::super::VReg;
        use super::super::interval::{LiveInterval, LiveRange, ProgPoint, UsePosition};

        let mut interval = LiveInterval::new(VReg::new(0), RegClass::Int);
        interval.add_range(LiveRange::new(ProgPoint::before(0), ProgPoint::before(10)));
        interval.add_use(UsePosition::def(ProgPoint::after(0)));
        interval.add_use(UsePosition::use_pos(ProgPoint::before(5)));
        interval.add_use(UsePosition::use_pos(ProgPoint::before(9)));

        let calc = SpillWeightCalculator::new();
        let loop_depths = vec![0u8; 20];
        let weight = calc.calculate(&interval, &loop_depths);

        assert!(weight > 0.0);
    }

    #[test]
    fn test_spill_weight_loop_matters() {
        use super::super::RegClass;
        use super::super::VReg;
        use super::super::interval::{LiveInterval, LiveRange, ProgPoint, UsePosition};

        // Interval A: used outside loops
        let mut interval_a = LiveInterval::new(VReg::new(0), RegClass::Int);
        interval_a.add_range(LiveRange::new(ProgPoint::before(0), ProgPoint::before(10)));
        interval_a.add_use(UsePosition::use_pos(ProgPoint::before(5)));

        // Interval B: used inside a loop
        let mut interval_b = LiveInterval::new(VReg::new(1), RegClass::Int);
        interval_b.add_range(LiveRange::new(ProgPoint::before(0), ProgPoint::before(10)));
        interval_b.add_use(UsePosition::use_pos(ProgPoint::before(5)));

        let calc = SpillWeightCalculator::new();

        let depths_no_loop = vec![0u8; 10];
        let mut depths_with_loop = vec![0u8; 10];
        depths_with_loop[5] = 1; // Position 5 is in a loop

        let weight_a = calc.calculate(&interval_a, &depths_no_loop);
        let weight_b = calc.calculate(&interval_b, &depths_with_loop);

        // B should have higher weight (less desirable to spill)
        assert!(weight_b > weight_a);
    }

    #[test]
    fn test_spill_width_from_reg_class() {
        use super::super::RegClass;

        assert_eq!(SpillWidth::from_reg_class(RegClass::Int), SpillWidth::W8);
        assert_eq!(SpillWidth::from_reg_class(RegClass::Any), SpillWidth::W8);
        assert_eq!(SpillWidth::from_reg_class(RegClass::Float), SpillWidth::W16);
        assert_eq!(
            SpillWidth::from_reg_class(RegClass::Vec256),
            SpillWidth::W32
        );
        assert_eq!(
            SpillWidth::from_reg_class(RegClass::Vec512),
            SpillWidth::W64
        );
    }

    #[test]
    fn test_spill_slot_allocator_reset() {
        let mut alloc = SpillSlotAllocator::new();

        alloc.alloc_gpr();
        alloc.alloc_ymm();
        alloc.alloc_zmm();

        assert!(alloc.total_size() > 0);
        assert!(alloc.total_slots() > 0);

        alloc.reset();

        assert_eq!(alloc.total_size(), 0);
        assert_eq!(alloc.total_slots(), 0);
    }
}
