//! Register Allocation Module
//!
//! This module implements a production-grade register allocator for the Prism JIT.
//! It provides multiple allocation strategies:
//!
//! - **Linear Scan**: Fast O(n log n) allocation for quick compilation
//! - **Graph Coloring (Chaitin-Briggs)**: Optimal allocation with coalescing
//!
//! # Architecture
//!
//! The allocator consists of several components:
//!
//! - `liveness.rs`: Compute live ranges for virtual registers
//! - `interval.rs`: Live interval representation and manipulation
//! - `interference.rs`: Interference graph construction
//! - `linear_scan.rs`: Linear scan allocator
//! - `graph_coloring.rs`: Chaitin-Briggs graph coloring allocator
//! - `spill.rs`: Spill code generation
//! - `constraint.rs`: Instruction-level operand constraints
//!
//! # Usage
//!
//! ```ignore
//! let mut allocator = RegisterAllocator::new(RegisterAllocatorConfig::default());
//! let result = allocator.allocate(&func)?;
//! ```

pub mod constraint;
pub mod interference;
pub mod interval;
pub mod linear_scan;
pub mod liveness;
pub mod spill;

use crate::backend::x64::registers::{Gpr, GprSet, Xmm, XmmSet};
use crate::backend::x64::simd::{Ymm, YmmSet, Zmm, ZmmSet};
use crate::ir::node::NodeId;
use std::collections::HashMap;

// Re-export key types
pub use constraint::{
    ConstraintDatabase, InstructionConstraint, OperandConstraint, OperandDescriptor, OperandRole,
};
pub use interference::InterferenceGraph;
pub use interval::{LiveInterval, LiveRange};
pub use linear_scan::LinearScanAllocator;
pub use liveness::LivenessAnalysis;
pub use spill::{SpillCode, SpillSlot};

// =============================================================================
// Virtual Register
// =============================================================================

/// A virtual register representing an SSA value.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct VReg(u32);

impl VReg {
    /// Create a new virtual register from a raw index.
    #[inline]
    pub const fn new(index: u32) -> Self {
        VReg(index)
    }

    /// Create a virtual register from a node ID.
    #[inline]
    pub fn from_node(id: NodeId) -> Self {
        VReg(id.index())
    }

    /// Get the raw index.
    #[inline]
    pub const fn index(self) -> u32 {
        self.0
    }
}

impl std::fmt::Display for VReg {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "v{}", self.0)
    }
}

// =============================================================================
// Physical Register
// =============================================================================

/// A physical machine register.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PReg {
    /// General-purpose register (64-bit).
    Gpr(Gpr),
    /// XMM register (128-bit SSE/AVX scalar).
    Xmm(Xmm),
    /// YMM register (256-bit AVX/AVX2 vector).
    Ymm(Ymm),
    /// ZMM register (512-bit AVX-512 vector).
    Zmm(Zmm),
}

impl PReg {
    /// Get the GPR if this is a GPR.
    #[inline]
    pub fn as_gpr(self) -> Option<Gpr> {
        match self {
            PReg::Gpr(g) => Some(g),
            _ => None,
        }
    }

    /// Get the XMM register if this is an XMM.
    #[inline]
    pub fn as_xmm(self) -> Option<Xmm> {
        match self {
            PReg::Xmm(x) => Some(x),
            _ => None,
        }
    }

    /// Get the YMM register if this is a YMM.
    #[inline]
    pub fn as_ymm(self) -> Option<Ymm> {
        match self {
            PReg::Ymm(y) => Some(y),
            _ => None,
        }
    }

    /// Get the ZMM register if this is a ZMM.
    #[inline]
    pub fn as_zmm(self) -> Option<Zmm> {
        match self {
            PReg::Zmm(z) => Some(z),
            _ => None,
        }
    }

    /// Get the register width in bits.
    #[inline]
    pub fn width(self) -> u16 {
        match self {
            PReg::Gpr(_) => 64,
            PReg::Xmm(_) => 128,
            PReg::Ymm(_) => 256,
            PReg::Zmm(_) => 512,
        }
    }

    /// Check if this is a vector register (XMM, YMM, or ZMM).
    #[inline]
    pub fn is_vector(self) -> bool {
        !matches!(self, PReg::Gpr(_))
    }

    /// Check if this is a SIMD register (YMM or ZMM, not scalar XMM).
    #[inline]
    pub fn is_wide_vector(self) -> bool {
        matches!(self, PReg::Ymm(_) | PReg::Zmm(_))
    }

    /// Get the hardware encoding (0-15 for GPR/XMM/YMM, 0-31 for ZMM).
    #[inline]
    pub fn encoding(self) -> u8 {
        match self {
            PReg::Gpr(g) => g.encoding(),
            PReg::Xmm(x) => x.encoding(),
            PReg::Ymm(y) => y.encoding(),
            PReg::Zmm(z) => z.encoding(),
        }
    }
}

impl std::fmt::Display for PReg {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PReg::Gpr(g) => write!(f, "{}", g),
            PReg::Xmm(x) => write!(f, "{}", x),
            PReg::Ymm(y) => write!(f, "{}", y),
            PReg::Zmm(z) => write!(f, "{}", z),
        }
    }
}

// =============================================================================
// Register Class
// =============================================================================

/// Register class for allocation constraints.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RegClass {
    /// General-purpose integer registers (64-bit).
    Int,
    /// Floating-point XMM registers (128-bit scalar/vector).
    Float,
    /// 256-bit vector registers (YMM, AVX/AVX2).
    Vec256,
    /// 512-bit vector registers (ZMM, AVX-512).
    Vec512,
    /// Any register (for moves, etc.).
    Any,
}

impl RegClass {
    /// Get the register width in bits.
    #[inline]
    pub const fn width(self) -> u16 {
        match self {
            RegClass::Int => 64,
            RegClass::Float => 128,
            RegClass::Vec256 => 256,
            RegClass::Vec512 => 512,
            RegClass::Any => 64, // Conservative default
        }
    }

    /// Get the spill slot size in bytes.
    #[inline]
    pub const fn spill_size(self) -> u32 {
        match self {
            RegClass::Int => 8,
            RegClass::Float => 16,
            RegClass::Vec256 => 32,
            RegClass::Vec512 => 64,
            RegClass::Any => 8,
        }
    }

    /// Get required alignment for spill slots.
    #[inline]
    pub const fn spill_alignment(self) -> u32 {
        // Vector spills require alignment equal to their size
        self.spill_size()
    }

    /// Check if this is a vector register class.
    #[inline]
    pub const fn is_vector(self) -> bool {
        matches!(self, RegClass::Float | RegClass::Vec256 | RegClass::Vec512)
    }

    /// Check if this is a wide vector class (256+ bits).
    #[inline]
    pub const fn is_wide_vector(self) -> bool {
        matches!(self, RegClass::Vec256 | RegClass::Vec512)
    }
}

// =============================================================================
// Allocation Result
// =============================================================================

/// The result of register allocation for a single virtual register.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Allocation {
    /// Allocated to a physical register.
    Register(PReg),
    /// Spilled to a stack slot.
    Spill(SpillSlot),
    /// Not yet allocated.
    Unassigned,
}

impl Allocation {
    /// Check if this is a register allocation.
    #[inline]
    pub fn is_register(self) -> bool {
        matches!(self, Allocation::Register(_))
    }

    /// Check if this is spilled.
    #[inline]
    pub fn is_spill(self) -> bool {
        matches!(self, Allocation::Spill(_))
    }

    /// Get the register if allocated.
    #[inline]
    pub fn reg(self) -> Option<PReg> {
        match self {
            Allocation::Register(r) => Some(r),
            _ => None,
        }
    }

    /// Get the spill slot if spilled.
    #[inline]
    pub fn spill_slot(self) -> Option<SpillSlot> {
        match self {
            Allocation::Spill(s) => Some(s),
            _ => None,
        }
    }
}

// =============================================================================
// Allocation Map
// =============================================================================

/// Maps virtual registers to their allocations.
#[derive(Debug, Clone, Default)]
pub struct AllocationMap {
    /// Allocation for each virtual register.
    allocations: HashMap<VReg, Allocation>,
    /// Number of spill slots used.
    num_spill_slots: u32,
    /// Move pairs needed for resolution.
    moves: Vec<(VReg, VReg)>,
}

impl AllocationMap {
    /// Create a new empty allocation map.
    pub fn new() -> Self {
        AllocationMap {
            allocations: HashMap::new(),
            num_spill_slots: 0,
            moves: Vec::new(),
        }
    }

    /// Set the allocation for a virtual register.
    pub fn set(&mut self, vreg: VReg, alloc: Allocation) {
        self.allocations.insert(vreg, alloc);
    }

    /// Get the allocation for a virtual register.
    pub fn get(&self, vreg: VReg) -> Allocation {
        self.allocations
            .get(&vreg)
            .copied()
            .unwrap_or(Allocation::Unassigned)
    }

    /// Allocate a new spill slot.
    pub fn alloc_spill_slot(&mut self) -> SpillSlot {
        let slot = SpillSlot::new(self.num_spill_slots);
        self.num_spill_slots += 1;
        slot
    }

    /// Get the number of spill slots used.
    pub fn spill_slot_count(&self) -> u32 {
        self.num_spill_slots
    }

    /// Set the number of spill slots (used to sync with external allocators).
    pub fn set_spill_slot_count(&mut self, count: u32) {
        self.num_spill_slots = count;
    }

    /// Add a move pair for later resolution.
    pub fn add_move(&mut self, from: VReg, to: VReg) {
        self.moves.push((from, to));
    }

    /// Get the list of moves.
    pub fn moves(&self) -> &[(VReg, VReg)] {
        &self.moves
    }

    /// Iterate over all allocations.
    pub fn iter(&self) -> impl Iterator<Item = (VReg, Allocation)> + '_ {
        self.allocations.iter().map(|(&v, &a)| (v, a))
    }
}

// =============================================================================
// Allocator Configuration
// =============================================================================

/// Configuration for the register allocator.
#[derive(Debug, Clone)]
pub struct AllocatorConfig {
    /// Available GPRs for allocation.
    pub available_gprs: GprSet,
    /// Available XMM registers for allocation.
    pub available_xmms: XmmSet,
    /// Available YMM registers for 256-bit vectors.
    pub available_ymms: YmmSet,
    /// Available ZMM registers for 512-bit vectors.
    pub available_zmms: ZmmSet,
    /// Reserved GPR (scratch).
    pub scratch_gpr: Gpr,
    /// Reserved XMM (scratch).
    pub scratch_xmm: Xmm,
    /// Reserved YMM (scratch for 256-bit ops).
    pub scratch_ymm: Ymm,
    /// Reserved ZMM (scratch for 512-bit ops).
    pub scratch_zmm: Zmm,
    /// Enable coalescing (reduces moves).
    pub enable_coalescing: bool,
    /// Enable live range splitting.
    pub enable_splitting: bool,
    /// Spill weight multiplier for loop nesting.
    pub loop_weight: f32,
}

impl Default for AllocatorConfig {
    fn default() -> Self {
        // Default: allocate all registers except RSP and R11 (scratch)
        let gprs = GprSet::ALL
            .remove(Gpr::Rsp) // Stack pointer
            .remove(Gpr::R11); // Scratch

        AllocatorConfig {
            available_gprs: gprs,
            available_xmms: XmmSet::ALL.remove(Xmm::Xmm15),
            available_ymms: YmmSet::ALL.remove(Ymm::Ymm15),
            available_zmms: ZmmSet::ALL.remove(Zmm::Zmm31),
            scratch_gpr: Gpr::R11,
            scratch_xmm: Xmm::Xmm15,
            scratch_ymm: Ymm::Ymm15,
            scratch_zmm: Zmm::Zmm31,
            enable_coalescing: true,
            enable_splitting: true,
            loop_weight: 10.0,
        }
    }
}

// =============================================================================
// Allocation Statistics
// =============================================================================

/// Statistics from register allocation.
#[derive(Debug, Clone, Default)]
pub struct AllocatorStats {
    /// Number of virtual registers.
    pub num_vregs: usize,
    /// Number of virtual registers assigned to physical registers.
    pub num_allocated: usize,
    /// Number of virtual registers spilled.
    pub num_spilled: usize,
    /// Number of moves coalesced.
    pub num_coalesced: usize,
    /// Number of splits performed.
    pub num_splits: usize,
    /// Time spent in microseconds.
    pub time_us: u64,
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vreg_creation() {
        let v1 = VReg::new(0);
        let v2 = VReg::new(1);
        assert_ne!(v1, v2);
        assert_eq!(v1.index(), 0);
        assert_eq!(v2.index(), 1);
    }

    #[test]
    fn test_preg_display() {
        let gpr = PReg::Gpr(Gpr::Rax);
        let xmm = PReg::Xmm(Xmm::Xmm0);
        assert_eq!(format!("{}", gpr), "rax");
        assert_eq!(format!("{}", xmm), "xmm0");
    }

    #[test]
    fn test_allocation_map() {
        let mut map = AllocationMap::new();
        let v1 = VReg::new(0);
        let v2 = VReg::new(1);

        map.set(v1, Allocation::Register(PReg::Gpr(Gpr::Rax)));
        let slot = map.alloc_spill_slot();
        map.set(v2, Allocation::Spill(slot));

        assert!(map.get(v1).is_register());
        assert!(map.get(v2).is_spill());
        assert_eq!(map.spill_slot_count(), 1);
    }

    #[test]
    fn test_default_config() {
        let config = AllocatorConfig::default();
        // Should have 14 GPRs (16 - RSP - R11)
        assert_eq!(config.available_gprs.count(), 14);
        // Should have 15 XMMs (16 - XMM15)
        assert_eq!(config.available_xmms.count(), 15);
    }
}
