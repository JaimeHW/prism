//! Linear Scan Register Allocator
//!
//! A fast register allocator using the linear scan algorithm.
//! Provides O(n log n) allocation suitable for JIT compilation.
//!
//! # Algorithm Overview
//!
//! 1. Sort intervals by start position
//! 2. Maintain active set of currently live intervals
//! 3. At each interval start, expire old intervals and allocate
//! 4. If no register available, spill the interval with furthest use
//!
//! # Vector Register Support
//!
//! This allocator handles aliasing between register widths:
//! - XMM0-15 (128-bit) aliases with YMM0-15 (256-bit) and ZMM0-15 (512-bit)
//! - YMM0-15 aliases with ZMM0-15
//! - ZMM16-31 have NO aliases and are preferred for allocation
//!
//! # References
//!
//! - Poletto & Sarkar, "Linear Scan Register Allocation" (1999)
//! - Wimmer & Franz, "Linear Scan Register Allocation on SSA Form" (2010)

use super::interval::{LiveInterval, ProgPoint};
use super::spill::{SpillSlotAllocator, SpillWidth};
use super::{Allocation, AllocationMap, AllocatorConfig, AllocatorStats, PReg, RegClass, VReg};
use crate::backend::x64::registers::{Gpr, GprSet, Xmm, XmmSet};
use crate::backend::x64::simd::{Ymm, YmmSet, Zmm, ZmmSet};
use std::collections::BinaryHeap;

// =============================================================================
// Active Interval
// =============================================================================

/// An interval in the active set, ordered by end position.
#[derive(Debug, Clone)]
struct ActiveInterval {
    vreg: VReg,
    end: ProgPoint,
    reg: PReg,
    reg_class: RegClass,
}

impl PartialEq for ActiveInterval {
    fn eq(&self, other: &Self) -> bool {
        self.end == other.end
    }
}

impl Eq for ActiveInterval {}

impl PartialOrd for ActiveInterval {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for ActiveInterval {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Reverse order for min-heap (earliest end first)
        other.end.cmp(&self.end)
    }
}

// =============================================================================
// Register Aliasing
// =============================================================================

/// Tracks register aliasing between XMM/YMM/ZMM register widths.
///
/// On x64, the lower 128 bits of YMM0 is XMM0, and the lower 256 bits
/// of ZMM0 is YMM0 (and thus XMM0). Writing to a wider register clobbers
/// the narrower alias.
#[derive(Debug, Clone)]
struct AliasingTracker {
    /// XMM registers blocked because a wider YMM/ZMM is in use.
    /// xmm_blocked[i] is true if XMM{i} is blocked.
    xmm_blocked_by_wider: [bool; 16],
    /// YMM registers blocked because a wider ZMM is in use.
    ymm_blocked_by_zmm: [bool; 16],
    /// XMM registers actually in use (not blocked, actively allocated).
    xmm_in_use: [bool; 16],
    /// YMM registers actually in use.
    ymm_in_use: [bool; 16],
    /// ZMM registers actually in use (0-31).
    zmm_in_use: [bool; 32],
}

impl Default for AliasingTracker {
    fn default() -> Self {
        Self::new()
    }
}

impl AliasingTracker {
    const fn new() -> Self {
        AliasingTracker {
            xmm_blocked_by_wider: [false; 16],
            ymm_blocked_by_zmm: [false; 16],
            xmm_in_use: [false; 16],
            ymm_in_use: [false; 16],
            zmm_in_use: [false; 32],
        }
    }

    /// Mark an XMM register as in use.
    fn use_xmm(&mut self, idx: u8) {
        debug_assert!(idx < 16);
        self.xmm_in_use[idx as usize] = true;
    }

    /// Mark a YMM register as in use (blocks corresponding XMM).
    fn use_ymm(&mut self, idx: u8) {
        debug_assert!(idx < 16);
        self.ymm_in_use[idx as usize] = true;
        self.xmm_blocked_by_wider[idx as usize] = true;
    }

    /// Mark a ZMM register as in use (blocks corresponding XMM and YMM if idx < 16).
    fn use_zmm(&mut self, idx: u8) {
        debug_assert!(idx < 32);
        self.zmm_in_use[idx as usize] = true;
        if idx < 16 {
            self.xmm_blocked_by_wider[idx as usize] = true;
            self.ymm_blocked_by_zmm[idx as usize] = true;
        }
    }

    /// Release an XMM register.
    fn release_xmm(&mut self, idx: u8) {
        debug_assert!(idx < 16);
        self.xmm_in_use[idx as usize] = false;
    }

    /// Release a YMM register (unblocks XMM if no ZMM blocking it).
    fn release_ymm(&mut self, idx: u8) {
        debug_assert!(idx < 16);
        self.ymm_in_use[idx as usize] = false;
        // Only unblock XMM if ZMM isn't also using this slot
        if !self.zmm_in_use[idx as usize] {
            self.xmm_blocked_by_wider[idx as usize] = false;
        }
    }

    /// Release a ZMM register (unblocks XMM/YMM if no longer needed).
    fn release_zmm(&mut self, idx: u8) {
        debug_assert!(idx < 32);
        self.zmm_in_use[idx as usize] = false;
        if idx < 16 {
            self.ymm_blocked_by_zmm[idx as usize] = false;
            // Only unblock XMM if YMM isn't also using this slot
            if !self.ymm_in_use[idx as usize] {
                self.xmm_blocked_by_wider[idx as usize] = false;
            }
        }
    }

    /// Check if an XMM register is available (not in use, not blocked).
    fn is_xmm_available(&self, idx: u8) -> bool {
        debug_assert!(idx < 16);
        !self.xmm_in_use[idx as usize] && !self.xmm_blocked_by_wider[idx as usize]
    }

    /// Check if a YMM register is available.
    fn is_ymm_available(&self, idx: u8) -> bool {
        debug_assert!(idx < 16);
        !self.ymm_in_use[idx as usize]
            && !self.ymm_blocked_by_zmm[idx as usize]
            && !self.xmm_in_use[idx as usize] // XMM must also be free
    }

    /// Check if a ZMM register is available.
    fn is_zmm_available(&self, idx: u8) -> bool {
        debug_assert!(idx < 32);
        if idx >= 16 {
            // ZMM16-31 have no aliases
            !self.zmm_in_use[idx as usize]
        } else {
            // ZMM0-15 alias with XMM and YMM
            !self.zmm_in_use[idx as usize]
                && !self.ymm_in_use[idx as usize]
                && !self.xmm_in_use[idx as usize]
        }
    }
}

// =============================================================================
// Linear Scan Allocator
// =============================================================================

/// The linear scan register allocator with full vector register support.
pub struct LinearScanAllocator {
    /// Configuration.
    config: AllocatorConfig,
    /// Allocation map being built.
    allocations: AllocationMap,
    /// Spill slot allocator.
    spill_allocator: SpillSlotAllocator,
    /// Active intervals for GPRs (sorted by end position).
    active_gprs: BinaryHeap<ActiveInterval>,
    /// Active intervals for XMMs.
    active_xmms: BinaryHeap<ActiveInterval>,
    /// Active intervals for YMMs.
    active_ymms: BinaryHeap<ActiveInterval>,
    /// Active intervals for ZMMs.
    active_zmms: BinaryHeap<ActiveInterval>,
    /// Available GPRs.
    free_gprs: GprSet,
    /// Available XMMs.
    free_xmms: XmmSet,
    /// Available YMMs.
    free_ymms: YmmSet,
    /// Available ZMMs.
    free_zmms: ZmmSet,
    /// Aliasing tracker for vector registers.
    aliasing: AliasingTracker,
    /// Statistics.
    stats: AllocatorStats,
}

impl LinearScanAllocator {
    /// Create a new allocator with the given configuration.
    pub fn new(config: AllocatorConfig) -> Self {
        let free_gprs = config.available_gprs;
        let free_xmms = config.available_xmms;
        let free_ymms = config.available_ymms;
        let free_zmms = config.available_zmms;

        LinearScanAllocator {
            config,
            allocations: AllocationMap::new(),
            spill_allocator: SpillSlotAllocator::new(),
            active_gprs: BinaryHeap::new(),
            active_xmms: BinaryHeap::new(),
            active_ymms: BinaryHeap::new(),
            active_zmms: BinaryHeap::new(),
            free_gprs,
            free_xmms,
            free_ymms,
            free_zmms,
            aliasing: AliasingTracker::new(),
            stats: AllocatorStats::default(),
        }
    }

    /// Allocate registers for a set of intervals.
    pub fn allocate(mut self, mut intervals: Vec<LiveInterval>) -> (AllocationMap, AllocatorStats) {
        self.stats.num_vregs = intervals.len();

        // Sort intervals by start position
        intervals.sort_by_key(|i| i.start());

        // Process each interval
        for interval in &intervals {
            if interval.is_empty() {
                continue;
            }

            let start = interval.start();

            // Expire old intervals (handles all register classes)
            self.expire_old_intervals(start);

            // Try to allocate
            if !self.try_allocate(interval) {
                // No register available - spill
                self.allocate_with_spill(interval);
            }
        }

        // Sync spill slot count from the external allocator
        self.allocations
            .set_spill_slot_count(self.spill_allocator.total_slots());

        (self.allocations, self.stats)
    }

    /// Expire intervals that end before the given position.
    fn expire_old_intervals(&mut self, pos: ProgPoint) {
        // Expire GPRs
        while let Some(active) = self.active_gprs.peek() {
            if active.end > pos {
                break;
            }
            let expired = self.active_gprs.pop().unwrap();
            if let Some(gpr) = expired.reg.as_gpr() {
                self.free_gprs = self.free_gprs.insert(gpr);
            }
        }

        // Expire XMMs
        while let Some(active) = self.active_xmms.peek() {
            if active.end > pos {
                break;
            }
            let expired = self.active_xmms.pop().unwrap();
            if let Some(xmm) = expired.reg.as_xmm() {
                self.free_xmms = self.free_xmms.insert(xmm);
                self.aliasing.release_xmm(xmm.encoding());
            }
        }

        // Expire YMMs
        while let Some(active) = self.active_ymms.peek() {
            if active.end > pos {
                break;
            }
            let expired = self.active_ymms.pop().unwrap();
            if let Some(ymm) = expired.reg.as_ymm() {
                self.free_ymms = self.free_ymms.insert(ymm);
                self.aliasing.release_ymm(ymm.encoding());
            }
        }

        // Expire ZMMs
        while let Some(active) = self.active_zmms.peek() {
            if active.end > pos {
                break;
            }
            let expired = self.active_zmms.pop().unwrap();
            if let Some(zmm) = expired.reg.as_zmm() {
                self.free_zmms = self.free_zmms.insert(zmm);
                self.aliasing.release_zmm(zmm.encoding());
            }
        }
    }

    /// Try to allocate a register for an interval.
    fn try_allocate(&mut self, interval: &LiveInterval) -> bool {
        match interval.reg_class {
            RegClass::Int => self.try_allocate_gpr(interval),
            RegClass::Float => self.try_allocate_xmm(interval),
            RegClass::Vec256 => self.try_allocate_ymm(interval),
            RegClass::Vec512 => self.try_allocate_zmm(interval),
            RegClass::Any => {
                // Try GPR first, then XMM
                self.try_allocate_gpr(interval) || self.try_allocate_xmm(interval)
            }
        }
    }

    /// Try to allocate a GPR.
    fn try_allocate_gpr(&mut self, interval: &LiveInterval) -> bool {
        if let Some(gpr) = self.free_gprs.first() {
            self.free_gprs = self.free_gprs.remove(gpr);

            let preg = PReg::Gpr(gpr);
            self.allocations
                .set(interval.vreg, Allocation::Register(preg));

            self.active_gprs.push(ActiveInterval {
                vreg: interval.vreg,
                end: interval.end(),
                reg: preg,
                reg_class: interval.reg_class,
            });

            self.stats.num_allocated += 1;
            true
        } else {
            false
        }
    }

    /// Try to allocate an XMM register with aliasing awareness.
    fn try_allocate_xmm(&mut self, interval: &LiveInterval) -> bool {
        // Find an XMM that's free AND not blocked by wider registers
        for i in 0..16u8 {
            if let Some(xmm) = Xmm::from_encoding(i) {
                if self.free_xmms.contains(xmm) && self.aliasing.is_xmm_available(i) {
                    self.free_xmms = self.free_xmms.remove(xmm);
                    self.aliasing.use_xmm(i);

                    let preg = PReg::Xmm(xmm);
                    self.allocations
                        .set(interval.vreg, Allocation::Register(preg));

                    self.active_xmms.push(ActiveInterval {
                        vreg: interval.vreg,
                        end: interval.end(),
                        reg: preg,
                        reg_class: interval.reg_class,
                    });

                    self.stats.num_allocated += 1;
                    return true;
                }
            }
        }
        false
    }

    /// Try to allocate a YMM register with aliasing awareness.
    fn try_allocate_ymm(&mut self, interval: &LiveInterval) -> bool {
        // Find a YMM that's free AND whose XMM alias is not in use
        for i in 0..16u8 {
            if let Some(ymm) = Ymm::from_encoding(i) {
                if self.free_ymms.contains(ymm) && self.aliasing.is_ymm_available(i) {
                    self.free_ymms = self.free_ymms.remove(ymm);
                    self.aliasing.use_ymm(i);

                    let preg = PReg::Ymm(ymm);
                    self.allocations
                        .set(interval.vreg, Allocation::Register(preg));

                    self.active_ymms.push(ActiveInterval {
                        vreg: interval.vreg,
                        end: interval.end(),
                        reg: preg,
                        reg_class: interval.reg_class,
                    });

                    self.stats.num_allocated += 1;
                    return true;
                }
            }
        }
        false
    }

    /// Try to allocate a ZMM register with aliasing awareness.
    /// Prefers ZMM16-31 (no aliases) over ZMM0-15.
    fn try_allocate_zmm(&mut self, interval: &LiveInterval) -> bool {
        // First try ZMM16-31 (no aliases - preferred)
        for i in 16..32u8 {
            if let Some(zmm) = Zmm::from_encoding(i) {
                if self.free_zmms.contains(zmm) && self.aliasing.is_zmm_available(i) {
                    self.free_zmms = self.free_zmms.remove(zmm);
                    self.aliasing.use_zmm(i);

                    let preg = PReg::Zmm(zmm);
                    self.allocations
                        .set(interval.vreg, Allocation::Register(preg));

                    self.active_zmms.push(ActiveInterval {
                        vreg: interval.vreg,
                        end: interval.end(),
                        reg: preg,
                        reg_class: interval.reg_class,
                    });

                    self.stats.num_allocated += 1;
                    return true;
                }
            }
        }

        // Then try ZMM0-15 (has aliases)
        for i in 0..16u8 {
            if let Some(zmm) = Zmm::from_encoding(i) {
                if self.free_zmms.contains(zmm) && self.aliasing.is_zmm_available(i) {
                    self.free_zmms = self.free_zmms.remove(zmm);
                    self.aliasing.use_zmm(i);

                    let preg = PReg::Zmm(zmm);
                    self.allocations
                        .set(interval.vreg, Allocation::Register(preg));

                    self.active_zmms.push(ActiveInterval {
                        vreg: interval.vreg,
                        end: interval.end(),
                        reg: preg,
                        reg_class: interval.reg_class,
                    });

                    self.stats.num_allocated += 1;
                    return true;
                }
            }
        }

        false
    }

    /// Allocate with spilling.
    fn allocate_with_spill(&mut self, interval: &LiveInterval) {
        match interval.reg_class {
            RegClass::Int => self.allocate_with_spill_gpr(interval),
            RegClass::Float => self.allocate_with_spill_xmm(interval),
            RegClass::Vec256 => self.allocate_with_spill_ymm(interval),
            RegClass::Vec512 => self.allocate_with_spill_zmm(interval),
            RegClass::Any => self.allocate_with_spill_gpr(interval),
        }
    }

    /// Allocate a GPR with potential spilling.
    fn allocate_with_spill_gpr(&mut self, interval: &LiveInterval) {
        // Find the interval with the furthest next use
        let mut spill_candidate: Option<(VReg, ProgPoint, PReg)> = None;

        // Check active intervals for best spill candidate
        for active in self.active_gprs.iter() {
            if active.end > interval.end() {
                match &spill_candidate {
                    None => spill_candidate = Some((active.vreg, active.end, active.reg)),
                    Some((_, furthest, _)) if active.end > *furthest => {
                        spill_candidate = Some((active.vreg, active.end, active.reg));
                    }
                    _ => {}
                }
            }
        }

        if let Some((spill_vreg, _, spill_reg)) = spill_candidate {
            // Spill the candidate
            let slot = self.spill_allocator.alloc(SpillWidth::W8);
            self.allocations
                .set(spill_vreg, Allocation::Spill(slot.into()));

            // Give its register to the current interval
            self.allocations
                .set(interval.vreg, Allocation::Register(spill_reg));

            // Update active set: remove spilled, add new
            self.active_gprs.retain(|a| a.vreg != spill_vreg);
            self.active_gprs.push(ActiveInterval {
                vreg: interval.vreg,
                end: interval.end(),
                reg: spill_reg,
                reg_class: interval.reg_class,
            });

            self.stats.num_spilled += 1;
            self.stats.num_allocated += 1;
        } else {
            // Spill the current interval
            let slot = self.spill_allocator.alloc(SpillWidth::W8);
            self.allocations
                .set(interval.vreg, Allocation::Spill(slot.into()));
            self.stats.num_spilled += 1;
        }
    }

    /// Allocate an XMM with potential spilling.
    fn allocate_with_spill_xmm(&mut self, interval: &LiveInterval) {
        let mut spill_candidate: Option<(VReg, ProgPoint, PReg)> = None;

        for active in self.active_xmms.iter() {
            if active.end > interval.end() {
                match &spill_candidate {
                    None => spill_candidate = Some((active.vreg, active.end, active.reg)),
                    Some((_, furthest, _)) if active.end > *furthest => {
                        spill_candidate = Some((active.vreg, active.end, active.reg));
                    }
                    _ => {}
                }
            }
        }

        if let Some((spill_vreg, _, spill_reg)) = spill_candidate {
            let slot = self.spill_allocator.alloc(SpillWidth::W16);
            self.allocations
                .set(spill_vreg, Allocation::Spill(slot.into()));
            self.allocations
                .set(interval.vreg, Allocation::Register(spill_reg));

            self.active_xmms.retain(|a| a.vreg != spill_vreg);
            self.active_xmms.push(ActiveInterval {
                vreg: interval.vreg,
                end: interval.end(),
                reg: spill_reg,
                reg_class: interval.reg_class,
            });

            self.stats.num_spilled += 1;
            self.stats.num_allocated += 1;
        } else {
            let slot = self.spill_allocator.alloc(SpillWidth::W16);
            self.allocations
                .set(interval.vreg, Allocation::Spill(slot.into()));
            self.stats.num_spilled += 1;
        }
    }

    /// Allocate a YMM with potential spilling.
    fn allocate_with_spill_ymm(&mut self, interval: &LiveInterval) {
        let mut spill_candidate: Option<(VReg, ProgPoint, PReg)> = None;

        for active in self.active_ymms.iter() {
            if active.end > interval.end() {
                match &spill_candidate {
                    None => spill_candidate = Some((active.vreg, active.end, active.reg)),
                    Some((_, furthest, _)) if active.end > *furthest => {
                        spill_candidate = Some((active.vreg, active.end, active.reg));
                    }
                    _ => {}
                }
            }
        }

        if let Some((spill_vreg, _, spill_reg)) = spill_candidate {
            let slot = self.spill_allocator.alloc(SpillWidth::W32);
            self.allocations
                .set(spill_vreg, Allocation::Spill(slot.into()));
            self.allocations
                .set(interval.vreg, Allocation::Register(spill_reg));

            // Release aliasing for spilled register
            if let Some(ymm) = spill_reg.as_ymm() {
                self.aliasing.release_ymm(ymm.encoding());
            }

            self.active_ymms.retain(|a| a.vreg != spill_vreg);
            self.active_ymms.push(ActiveInterval {
                vreg: interval.vreg,
                end: interval.end(),
                reg: spill_reg,
                reg_class: interval.reg_class,
            });

            // Re-mark aliasing for newly allocated register
            if let Some(ymm) = spill_reg.as_ymm() {
                self.aliasing.use_ymm(ymm.encoding());
            }

            self.stats.num_spilled += 1;
            self.stats.num_allocated += 1;
        } else {
            let slot = self.spill_allocator.alloc(SpillWidth::W32);
            self.allocations
                .set(interval.vreg, Allocation::Spill(slot.into()));
            self.stats.num_spilled += 1;
        }
    }

    /// Allocate a ZMM with potential spilling.
    fn allocate_with_spill_zmm(&mut self, interval: &LiveInterval) {
        let mut spill_candidate: Option<(VReg, ProgPoint, PReg)> = None;

        for active in self.active_zmms.iter() {
            if active.end > interval.end() {
                match &spill_candidate {
                    None => spill_candidate = Some((active.vreg, active.end, active.reg)),
                    Some((_, furthest, _)) if active.end > *furthest => {
                        spill_candidate = Some((active.vreg, active.end, active.reg));
                    }
                    _ => {}
                }
            }
        }

        if let Some((spill_vreg, _, spill_reg)) = spill_candidate {
            let slot = self.spill_allocator.alloc(SpillWidth::W64);
            self.allocations
                .set(spill_vreg, Allocation::Spill(slot.into()));
            self.allocations
                .set(interval.vreg, Allocation::Register(spill_reg));

            // Release aliasing for spilled register
            if let Some(zmm) = spill_reg.as_zmm() {
                self.aliasing.release_zmm(zmm.encoding());
            }

            self.active_zmms.retain(|a| a.vreg != spill_vreg);
            self.active_zmms.push(ActiveInterval {
                vreg: interval.vreg,
                end: interval.end(),
                reg: spill_reg,
                reg_class: interval.reg_class,
            });

            // Re-mark aliasing for newly allocated register
            if let Some(zmm) = spill_reg.as_zmm() {
                self.aliasing.use_zmm(zmm.encoding());
            }

            self.stats.num_spilled += 1;
            self.stats.num_allocated += 1;
        } else {
            let slot = self.spill_allocator.alloc(SpillWidth::W64);
            self.allocations
                .set(interval.vreg, Allocation::Spill(slot.into()));
            self.stats.num_spilled += 1;
        }
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::regalloc::interval::{LiveRange, ProgPoint};

    fn make_interval(vreg: u32, start: u32, end: u32, reg_class: RegClass) -> LiveInterval {
        let mut interval = LiveInterval::new(VReg::new(vreg), reg_class);
        interval.add_range(LiveRange::new(
            ProgPoint::before(start),
            ProgPoint::before(end),
        ));
        interval
    }

    fn make_gpr_interval(vreg: u32, start: u32, end: u32) -> LiveInterval {
        make_interval(vreg, start, end, RegClass::Int)
    }

    fn make_xmm_interval(vreg: u32, start: u32, end: u32) -> LiveInterval {
        make_interval(vreg, start, end, RegClass::Float)
    }

    fn make_ymm_interval(vreg: u32, start: u32, end: u32) -> LiveInterval {
        make_interval(vreg, start, end, RegClass::Vec256)
    }

    fn make_zmm_interval(vreg: u32, start: u32, end: u32) -> LiveInterval {
        make_interval(vreg, start, end, RegClass::Vec512)
    }

    // =========================================================================
    // Basic GPR Tests
    // =========================================================================

    #[test]
    fn test_simple_allocation() {
        let intervals = vec![
            make_gpr_interval(0, 0, 10),
            make_gpr_interval(1, 5, 15),
            make_gpr_interval(2, 20, 30),
        ];

        let config = AllocatorConfig::default();
        let allocator = LinearScanAllocator::new(config);
        let (map, stats) = allocator.allocate(intervals);

        assert!(map.get(VReg::new(0)).is_register());
        assert!(map.get(VReg::new(1)).is_register());
        assert!(map.get(VReg::new(2)).is_register());
        assert_eq!(stats.num_spilled, 0);
    }

    #[test]
    fn test_different_registers_for_overlapping() {
        let intervals = vec![make_gpr_interval(0, 0, 20), make_gpr_interval(1, 10, 30)];

        let config = AllocatorConfig::default();
        let allocator = LinearScanAllocator::new(config);
        let (map, _stats) = allocator.allocate(intervals);

        let r0 = map.get(VReg::new(0)).reg();
        let r1 = map.get(VReg::new(1)).reg();

        assert!(r0.is_some());
        assert!(r1.is_some());
        assert_ne!(r0, r1);
    }

    #[test]
    fn test_register_reuse() {
        let intervals = vec![make_gpr_interval(0, 0, 10), make_gpr_interval(1, 20, 30)];

        let config = AllocatorConfig::default();
        let allocator = LinearScanAllocator::new(config);
        let (map, _stats) = allocator.allocate(intervals);

        let r0 = map.get(VReg::new(0)).reg();
        let r1 = map.get(VReg::new(1)).reg();

        assert_eq!(r0, r1);
    }

    #[test]
    fn test_spill_when_needed() {
        let mut intervals = Vec::new();
        for i in 0..16 {
            intervals.push(make_gpr_interval(i, 0, 100));
        }

        let config = AllocatorConfig::default();
        let allocator = LinearScanAllocator::new(config);
        let (map, stats) = allocator.allocate(intervals);

        assert!(stats.num_spilled >= 2);
        assert!(map.spill_slot_count() >= 2);
    }

    // =========================================================================
    // YMM Vector Tests
    // =========================================================================

    #[test]
    fn test_ymm_simple_allocation() {
        let intervals = vec![make_ymm_interval(0, 0, 10)];

        let config = AllocatorConfig::default();
        let allocator = LinearScanAllocator::new(config);
        let (map, stats) = allocator.allocate(intervals);

        let alloc = map.get(VReg::new(0));
        assert!(alloc.is_register());
        assert!(alloc.reg().unwrap().as_ymm().is_some());
        assert_eq!(stats.num_spilled, 0);
        assert_eq!(stats.num_allocated, 1);
    }

    #[test]
    fn test_ymm_multiple_non_overlapping() {
        let intervals = vec![make_ymm_interval(0, 0, 10), make_ymm_interval(1, 20, 30)];

        let config = AllocatorConfig::default();
        let allocator = LinearScanAllocator::new(config);
        let (map, stats) = allocator.allocate(intervals);

        // Both should be allocated, potentially to the same register
        assert!(map.get(VReg::new(0)).is_register());
        assert!(map.get(VReg::new(1)).is_register());
        assert_eq!(stats.num_spilled, 0);
    }

    #[test]
    fn test_ymm_overlapping_different_registers() {
        let intervals = vec![make_ymm_interval(0, 0, 20), make_ymm_interval(1, 10, 30)];

        let config = AllocatorConfig::default();
        let allocator = LinearScanAllocator::new(config);
        let (map, _stats) = allocator.allocate(intervals);

        let r0 = map.get(VReg::new(0)).reg();
        let r1 = map.get(VReg::new(1)).reg();

        assert!(r0.is_some());
        assert!(r1.is_some());
        assert_ne!(r0, r1);
    }

    // =========================================================================
    // ZMM Vector Tests
    // =========================================================================

    #[test]
    fn test_zmm_simple_allocation() {
        let intervals = vec![make_zmm_interval(0, 0, 10)];

        let config = AllocatorConfig::default();
        let allocator = LinearScanAllocator::new(config);
        let (map, stats) = allocator.allocate(intervals);

        let alloc = map.get(VReg::new(0));
        assert!(alloc.is_register());
        assert!(alloc.reg().unwrap().as_zmm().is_some());
        assert_eq!(stats.num_spilled, 0);
    }

    #[test]
    fn test_zmm_prefers_upper_16() {
        // Allocate a single ZMM - should prefer ZMM16-31 (no aliases)
        let intervals = vec![make_zmm_interval(0, 0, 10)];

        let config = AllocatorConfig::default();
        let allocator = LinearScanAllocator::new(config);
        let (map, _stats) = allocator.allocate(intervals);

        let zmm = map.get(VReg::new(0)).reg().unwrap().as_zmm().unwrap();
        assert!(zmm.encoding() >= 16, "Should prefer ZMM16-31");
    }

    #[test]
    fn test_zmm_multiple_overlapping() {
        let mut intervals = Vec::new();
        for i in 0..5 {
            intervals.push(make_zmm_interval(i, 0, 100));
        }

        let config = AllocatorConfig::default();
        let allocator = LinearScanAllocator::new(config);
        let (map, stats) = allocator.allocate(intervals);

        // All should be allocated (we have 31 ZMMs available)
        for i in 0..5 {
            assert!(map.get(VReg::new(i)).is_register());
        }
        assert_eq!(stats.num_spilled, 0);
    }

    // =========================================================================
    // Aliasing Tests
    // =========================================================================

    #[test]
    fn test_xmm_ymm_aliasing_conflict() {
        // XMM0 and YMM0 cannot be used simultaneously
        // Allocate XMM first, then YMM - they must get different physical registers
        let intervals = vec![make_xmm_interval(0, 0, 50), make_ymm_interval(1, 0, 50)];

        let config = AllocatorConfig::default();
        let allocator = LinearScanAllocator::new(config);
        let (map, stats) = allocator.allocate(intervals);

        let xmm = map.get(VReg::new(0)).reg();
        let ymm = map.get(VReg::new(1)).reg();

        assert!(xmm.is_some());
        assert!(ymm.is_some());

        // They should NOT have the same encoding
        let xmm_enc = xmm.unwrap().as_xmm().unwrap().encoding();
        let ymm_enc = ymm.unwrap().as_ymm().unwrap().encoding();
        assert_ne!(xmm_enc, ymm_enc, "XMM and YMM should not alias");

        assert_eq!(stats.num_spilled, 0);
    }

    #[test]
    fn test_ymm_zmm_aliasing_conflict() {
        // YMM0-15 and ZMM0-15 cannot be used simultaneously
        let intervals = vec![make_ymm_interval(0, 0, 50), make_zmm_interval(1, 0, 50)];

        let config = AllocatorConfig::default();
        let allocator = LinearScanAllocator::new(config);
        let (map, stats) = allocator.allocate(intervals);

        let ymm = map.get(VReg::new(0)).reg();
        let zmm = map.get(VReg::new(1)).reg();

        assert!(ymm.is_some());
        assert!(zmm.is_some());

        let ymm_enc = ymm.unwrap().as_ymm().unwrap().encoding();
        let zmm_enc = zmm.unwrap().as_zmm().unwrap().encoding();

        // Either ZMM got a non-aliasing register (16-31) or they're different
        if zmm_enc < 16 {
            assert_ne!(ymm_enc, zmm_enc, "YMM and ZMM0-15 should not alias");
        }

        assert_eq!(stats.num_spilled, 0);
    }

    #[test]
    fn test_zmm16_31_no_aliasing() {
        // ZMM16-31 have no XMM/YMM aliases, so we can use them freely with XMM
        let intervals = vec![
            make_xmm_interval(0, 0, 50),
            make_xmm_interval(1, 0, 50),
            make_xmm_interval(2, 0, 50),
            make_zmm_interval(3, 0, 50), // Should get ZMM16+
        ];

        let config = AllocatorConfig::default();
        let allocator = LinearScanAllocator::new(config);
        let (map, stats) = allocator.allocate(intervals);

        // All should be allocated
        for i in 0..4 {
            assert!(map.get(VReg::new(i)).is_register());
        }

        // ZMM should be in the upper range
        let zmm = map.get(VReg::new(3)).reg().unwrap().as_zmm().unwrap();
        assert!(zmm.encoding() >= 16);

        assert_eq!(stats.num_spilled, 0);
    }

    #[test]
    fn test_aliasing_with_expiration() {
        // YMM expires, then XMM with same encoding should be usable
        let intervals = vec![
            make_ymm_interval(0, 0, 10),  // YMM, expires at 10
            make_xmm_interval(1, 20, 30), // XMM, starts after YMM expires
        ];

        let config = AllocatorConfig::default();
        let allocator = LinearScanAllocator::new(config);
        let (map, stats) = allocator.allocate(intervals);

        // Both should be allocated
        assert!(map.get(VReg::new(0)).is_register());
        assert!(map.get(VReg::new(1)).is_register());

        // They CAN have the same encoding since they don't overlap
        // (but don't require it)
        assert_eq!(stats.num_spilled, 0);
    }

    // =========================================================================
    // Mixed Width Tests
    // =========================================================================

    #[test]
    fn test_mixed_width_allocation() {
        let intervals = vec![
            make_gpr_interval(0, 0, 100),
            make_xmm_interval(1, 0, 100),
            make_ymm_interval(2, 0, 100),
            make_zmm_interval(3, 0, 100),
        ];

        let config = AllocatorConfig::default();
        let allocator = LinearScanAllocator::new(config);
        let (map, stats) = allocator.allocate(intervals);

        // All should be allocated to correct register types
        assert!(map.get(VReg::new(0)).reg().unwrap().as_gpr().is_some());
        assert!(map.get(VReg::new(1)).reg().unwrap().as_xmm().is_some());
        assert!(map.get(VReg::new(2)).reg().unwrap().as_ymm().is_some());
        assert!(map.get(VReg::new(3)).reg().unwrap().as_zmm().is_some());

        assert_eq!(stats.num_spilled, 0);
    }

    // =========================================================================
    // Spill Tests for Vectors
    // =========================================================================

    #[test]
    fn test_ymm_spill_when_exhausted() {
        // Create 17 overlapping YMM intervals (only 15 available)
        let mut intervals = Vec::new();
        for i in 0..17 {
            intervals.push(make_ymm_interval(i, 0, 100));
        }

        let config = AllocatorConfig::default();
        let allocator = LinearScanAllocator::new(config);
        let (map, stats) = allocator.allocate(intervals);

        // Should have at least 2 spills (17 - 15)
        assert!(stats.num_spilled >= 2);
        assert!(map.spill_slot_count() >= 2);
    }

    #[test]
    fn test_zmm_spill_when_exhausted() {
        // Create 33 overlapping ZMM intervals (only 31 available)
        let mut intervals = Vec::new();
        for i in 0..33 {
            intervals.push(make_zmm_interval(i, 0, 100));
        }

        let config = AllocatorConfig::default();
        let allocator = LinearScanAllocator::new(config);
        let (map, stats) = allocator.allocate(intervals);

        // Should have at least 2 spills (33 - 31)
        assert!(stats.num_spilled >= 2);
        assert!(map.spill_slot_count() >= 2);
    }

    // =========================================================================
    // Aliasing Tracker Unit Tests
    // =========================================================================

    #[test]
    fn test_aliasing_tracker_xmm() {
        let mut tracker = AliasingTracker::new();

        assert!(tracker.is_xmm_available(0));
        tracker.use_xmm(0);
        assert!(!tracker.is_xmm_available(0));

        tracker.release_xmm(0);
        assert!(tracker.is_xmm_available(0));
    }

    #[test]
    fn test_aliasing_tracker_ymm_blocks_xmm() {
        let mut tracker = AliasingTracker::new();

        assert!(tracker.is_xmm_available(5));
        assert!(tracker.is_ymm_available(5));

        tracker.use_ymm(5);

        // XMM5 should be blocked
        assert!(!tracker.is_xmm_available(5));
        assert!(!tracker.is_ymm_available(5));

        tracker.release_ymm(5);

        // Both should be available again
        assert!(tracker.is_xmm_available(5));
        assert!(tracker.is_ymm_available(5));
    }

    #[test]
    fn test_aliasing_tracker_zmm_blocks_both() {
        let mut tracker = AliasingTracker::new();

        assert!(tracker.is_xmm_available(3));
        assert!(tracker.is_ymm_available(3));
        assert!(tracker.is_zmm_available(3));

        tracker.use_zmm(3);

        // All should be blocked
        assert!(!tracker.is_xmm_available(3));
        assert!(!tracker.is_ymm_available(3));
        assert!(!tracker.is_zmm_available(3));

        tracker.release_zmm(3);

        // All should be available again
        assert!(tracker.is_xmm_available(3));
        assert!(tracker.is_ymm_available(3));
        assert!(tracker.is_zmm_available(3));
    }

    #[test]
    fn test_aliasing_tracker_zmm16_no_alias() {
        let mut tracker = AliasingTracker::new();

        // ZMM16 has no XMM/YMM alias
        assert!(tracker.is_zmm_available(16));

        // Using ZMM16 should NOT block any XMM/YMM
        tracker.use_zmm(16);

        assert!(!tracker.is_zmm_available(16));
        // XMM0-15 should still be available
        for i in 0..16 {
            assert!(tracker.is_xmm_available(i));
            assert!(tracker.is_ymm_available(i));
        }
    }

    #[test]
    fn test_aliasing_tracker_layered_blocking() {
        let mut tracker = AliasingTracker::new();

        // Use both YMM and ZMM for same slot
        tracker.use_ymm(0);
        tracker.use_zmm(0);

        // XMM0 is blocked by both
        assert!(!tracker.is_xmm_available(0));

        // Release YMM - XMM should still be blocked by ZMM
        tracker.release_ymm(0);
        assert!(!tracker.is_xmm_available(0));

        // Release ZMM - now XMM should be available
        tracker.release_zmm(0);
        assert!(tracker.is_xmm_available(0));
    }
}
