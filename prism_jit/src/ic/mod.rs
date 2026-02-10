//! Inline Caching (IC) System
//!
//! A V8-style inline caching implementation for O(1) property access and method dispatch.
//!
//! # Architecture
//!
//! The IC system optimizes dynamic property access by caching type feedback at call sites.
//! Each IC site progresses through states:
//!
//! ```text
//! Uninitialized → Monomorphic → Polymorphic → Megamorphic
//!                     ↑              ↑              ↑
//!                 1 shape       2-4 shapes      5+ shapes
//! ```
//!
//! ## Components
//!
//! - **PropertyIc**: Caches GetAttr/SetAttr operations using Shape-based dispatch
//! - **CallIc**: Caches function/method calls for direct invocation
//! - **IcManager**: Per-function IC allocation and lifecycle management
//! - **Invalidation**: Shape-based dependency tracking and batch invalidation
//!
//! ## Performance Targets
//!
//! | State        | Overhead     | Implementation                    |
//! |--------------|--------------|-----------------------------------|
//! | Monomorphic  | <5 cycles    | Shape check + direct slot load    |
//! | Polymorphic  | <20 cycles   | Linear scan 4 entries             |
//! | Megamorphic  | <100 cycles  | Hash table lookup                 |

mod call_ic;
pub mod hooks;
mod ic_runtime;
mod invalidation;
pub mod jit_listener;
mod mega_ic;
mod poly_ic;
mod property_ic;

#[cfg(test)]
mod integration_tests;

pub use call_ic::{CallIc, CallIcData, CallIcState};
pub use hooks::{
    CountingListener, NullListener, RecordedEvent, RecordingListener, ShapeHookStats,
    ShapeHookStatsSnapshot, ShapeTransitionListener, notify_accessor_installed,
    notify_batch_transitions, notify_property_delete, notify_prototype_change,
    notify_shape_transition, register_shape_listener, register_shape_listener_boxed,
};
pub use ic_runtime::{IcMissHandler, ic_call_miss, ic_get_property_miss, ic_set_property_miss};
pub use invalidation::{IcDependency, IcInvalidator, InvalidationReason, ShapeVersion};
pub use jit_listener::{
    JitListenerStats, JitListenerStatsSnapshot, JitShapeListener, global_jit_listener,
    init_jit_listener,
};
pub use mega_ic::{MegaIcEntry, MegamorphicCache};
pub use poly_ic::{PolyIc, PolyIcEntry};
pub use property_ic::{PropertyIc, PropertyIcData, SlotInfo};

use std::sync::atomic::{AtomicU8, AtomicU64, Ordering};

// =============================================================================
// IC State
// =============================================================================

/// The state of an inline cache site.
///
/// IC sites progress through states based on observed type diversity.
/// Once megamorphic, a site stays megamorphic (no state regression).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum IcState {
    /// No type information recorded yet.
    /// First access will transition to Monomorphic.
    Uninitialized = 0,

    /// Single shape observed - optimal case.
    /// Direct shape check + slot access with no dispatch overhead.
    Monomorphic = 1,

    /// 2-4 shapes observed.
    /// Linear scan through small inline array (cache-friendly).
    Polymorphic = 2,

    /// 5+ shapes observed.
    /// Falls back to global hash table lookup.
    Megamorphic = 3,
}

impl IcState {
    /// Convert from raw u8 value.
    #[inline]
    pub const fn from_u8(v: u8) -> Self {
        match v {
            0 => Self::Uninitialized,
            1 => Self::Monomorphic,
            2 => Self::Polymorphic,
            3 => Self::Megamorphic,
            _ => Self::Megamorphic, // Invalid values become megamorphic
        }
    }

    /// Check if this state has type feedback.
    #[inline]
    pub const fn has_feedback(self) -> bool {
        !matches!(self, Self::Uninitialized)
    }

    /// Check if this is the optimal monomorphic state.
    #[inline]
    pub const fn is_monomorphic(self) -> bool {
        matches!(self, Self::Monomorphic)
    }

    /// Check if IC can still specialize (not megamorphic).
    #[inline]
    pub const fn can_specialize(self) -> bool {
        !matches!(self, Self::Megamorphic)
    }
}

impl Default for IcState {
    #[inline]
    fn default() -> Self {
        Self::Uninitialized
    }
}

// =============================================================================
// IC Kind
// =============================================================================

/// The kind of operation an IC caches.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum IcKind {
    /// Property read: `obj.attr`
    GetProperty = 0,
    /// Property write: `obj.attr = value`
    SetProperty = 1,
    /// Property deletion: `del obj.attr`
    DelProperty = 2,
    /// Item read: `obj[key]`
    GetItem = 3,
    /// Item write: `obj[key] = value`
    SetItem = 4,
    /// Function/method call
    Call = 5,
    /// Binary operation: `a + b`
    BinaryOp = 6,
    /// Comparison: `a == b`
    Compare = 7,
}

impl IcKind {
    /// Check if this is a property operation.
    #[inline]
    pub const fn is_property_op(self) -> bool {
        matches!(
            self,
            Self::GetProperty | Self::SetProperty | Self::DelProperty
        )
    }

    /// Check if this is an item operation.
    #[inline]
    pub const fn is_item_op(self) -> bool {
        matches!(self, Self::GetItem | Self::SetItem)
    }
}

// =============================================================================
// IC Site Header
// =============================================================================

/// Compact header for an IC site (16 bytes).
///
/// Contains all metadata needed to locate and manage an IC.
/// The actual cached data is stored separately based on IC kind.
#[repr(C, align(16))]
#[derive(Debug)]
pub struct IcSiteHeader {
    /// Bytecode offset where this IC is located.
    pub bytecode_offset: u32,

    /// Kind of operation being cached.
    pub kind: IcKind,

    /// Current IC state (atomic for lock-free transitions).
    state: AtomicU8,

    /// Miss count (used for state transition decisions).
    miss_count: AtomicU8,

    /// Reserved for future use / padding.
    _reserved: u8,

    /// Index into the IC data array for this kind.
    pub data_index: u32,
}

impl IcSiteHeader {
    /// Create a new uninitialized IC site.
    #[inline]
    pub fn new(bytecode_offset: u32, kind: IcKind, data_index: u32) -> Self {
        Self {
            bytecode_offset,
            kind,
            state: AtomicU8::new(IcState::Uninitialized as u8),
            miss_count: AtomicU8::new(0),
            _reserved: 0,
            data_index,
        }
    }

    /// Get current IC state.
    #[inline]
    pub fn state(&self) -> IcState {
        IcState::from_u8(self.state.load(Ordering::Acquire))
    }

    /// Atomically transition to a new state.
    ///
    /// Returns true if transition succeeded, false if state was already higher.
    #[inline]
    pub fn transition_to(&self, new_state: IcState) -> bool {
        let new = new_state as u8;
        let mut current = self.state.load(Ordering::Acquire);

        loop {
            // Only allow forward transitions (no regression)
            if current >= new {
                return false;
            }

            match self.state.compare_exchange_weak(
                current,
                new,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => return true,
                Err(actual) => current = actual,
            }
        }
    }

    /// Increment miss count and return new value.
    #[inline]
    pub fn record_miss(&self) -> u8 {
        // Use CAS loop for proper saturation (fetch_add wraps on overflow)
        loop {
            let current = self.miss_count.load(Ordering::Relaxed);
            if current == u8::MAX {
                return current;
            }
            match self.miss_count.compare_exchange_weak(
                current,
                current + 1,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => return current.saturating_add(1),
                Err(_) => continue,
            }
        }
    }

    /// Get current miss count.
    #[inline]
    pub fn miss_count(&self) -> u8 {
        self.miss_count.load(Ordering::Relaxed)
    }

    /// Reset IC to uninitialized state (for invalidation).
    #[inline]
    pub fn reset(&self) {
        self.state
            .store(IcState::Uninitialized as u8, Ordering::Release);
        self.miss_count.store(0, Ordering::Relaxed);
    }
}

// =============================================================================
// IC Site
// =============================================================================

/// A complete IC site with header and cached data.
///
/// This is the primary interface for IC operations.
#[derive(Debug)]
pub struct IcSite {
    /// Header with metadata.
    pub header: IcSiteHeader,

    /// Cached property IC data (if property operation).
    property_ic: Option<PropertyIc>,

    /// Cached call IC data (if call operation).
    call_ic: Option<CallIc>,
}

impl IcSite {
    /// Create a new property IC site.
    #[inline]
    pub fn new_property(bytecode_offset: u32, kind: IcKind, data_index: u32) -> Self {
        debug_assert!(kind.is_property_op());
        Self {
            header: IcSiteHeader::new(bytecode_offset, kind, data_index),
            property_ic: Some(PropertyIc::new()),
            call_ic: None,
        }
    }

    /// Create a new call IC site.
    #[inline]
    pub fn new_call(bytecode_offset: u32, data_index: u32) -> Self {
        Self {
            header: IcSiteHeader::new(bytecode_offset, IcKind::Call, data_index),
            property_ic: None,
            call_ic: Some(CallIc::new()),
        }
    }

    /// Get property IC if this is a property operation.
    #[inline]
    pub fn property_ic(&self) -> Option<&PropertyIc> {
        self.property_ic.as_ref()
    }

    /// Get mutable property IC.
    #[inline]
    pub fn property_ic_mut(&mut self) -> Option<&mut PropertyIc> {
        self.property_ic.as_mut()
    }

    /// Get call IC if this is a call operation.
    #[inline]
    pub fn call_ic(&self) -> Option<&CallIc> {
        self.call_ic.as_ref()
    }

    /// Get mutable call IC.
    #[inline]
    pub fn call_ic_mut(&mut self) -> Option<&mut CallIc> {
        self.call_ic.as_mut()
    }

    /// Reset this IC site (for invalidation).
    #[inline]
    pub fn reset(&mut self) {
        self.header.reset();
        if let Some(ref mut prop_ic) = self.property_ic {
            prop_ic.reset();
        }
        if let Some(ref mut call_ic) = self.call_ic {
            call_ic.reset();
        }
    }
}

// =============================================================================
// IC Manager
// =============================================================================

/// Maximum number of IC sites per function.
///
/// Most functions have far fewer, but we cap to prevent memory bloat.
pub const MAX_IC_SITES_PER_FUNCTION: usize = 256;

/// Threshold to transition from polymorphic to megamorphic.
pub const POLY_TO_MEGA_THRESHOLD: u8 = 5;

/// Number of entries in polymorphic IC (power of 2 for alignment).
pub const POLY_IC_ENTRIES: usize = 4;

/// Statistics for IC usage.
#[derive(Debug, Clone, Copy, Default)]
pub struct IcStats {
    /// Total number of IC sites.
    pub total_sites: u32,
    /// Sites in monomorphic state.
    pub monomorphic: u32,
    /// Sites in polymorphic state.
    pub polymorphic: u32,
    /// Sites in megamorphic state.
    pub megamorphic: u32,
    /// Total cache hits.
    pub hits: u64,
    /// Total cache misses.
    pub misses: u64,
}

impl IcStats {
    /// Calculate hit rate.
    #[inline]
    pub fn hit_rate(&self) -> f64 {
        let total = self.hits + self.misses;
        if total == 0 {
            0.0
        } else {
            self.hits as f64 / total as f64
        }
    }
}

/// Manages IC sites for a single compiled function.
///
/// Provides allocation and lookup of IC sites, tracks statistics,
/// and coordinates invalidation.
#[derive(Debug)]
pub struct IcManager {
    /// IC sites indexed by their data_index.
    sites: Vec<IcSite>,

    /// Statistics counters.
    hits: AtomicU64,
    misses: AtomicU64,

    /// Shape version when this manager was created.
    /// Used to detect stale ICs after shape invalidation.
    shape_version: ShapeVersion,
}

impl IcManager {
    /// Create a new IC manager.
    #[inline]
    pub fn new(shape_version: ShapeVersion) -> Self {
        Self {
            sites: Vec::new(),
            hits: AtomicU64::new(0),
            misses: AtomicU64::new(0),
            shape_version,
        }
    }

    /// Create with pre-allocated capacity.
    #[inline]
    pub fn with_capacity(capacity: usize, shape_version: ShapeVersion) -> Self {
        Self {
            sites: Vec::with_capacity(capacity.min(MAX_IC_SITES_PER_FUNCTION)),
            hits: AtomicU64::new(0),
            misses: AtomicU64::new(0),
            shape_version,
        }
    }

    /// Allocate a new property IC site.
    ///
    /// Returns the index of the new site, or None if at capacity.
    pub fn alloc_property_ic(&mut self, bytecode_offset: u32, kind: IcKind) -> Option<u32> {
        if self.sites.len() >= MAX_IC_SITES_PER_FUNCTION {
            return None;
        }

        let index = self.sites.len() as u32;
        self.sites
            .push(IcSite::new_property(bytecode_offset, kind, index));
        Some(index)
    }

    /// Allocate a new call IC site.
    pub fn alloc_call_ic(&mut self, bytecode_offset: u32) -> Option<u32> {
        if self.sites.len() >= MAX_IC_SITES_PER_FUNCTION {
            return None;
        }

        let index = self.sites.len() as u32;
        self.sites.push(IcSite::new_call(bytecode_offset, index));
        Some(index)
    }

    /// Get an IC site by index.
    #[inline]
    pub fn get(&self, index: u32) -> Option<&IcSite> {
        self.sites.get(index as usize)
    }

    /// Get a mutable IC site by index.
    #[inline]
    pub fn get_mut(&mut self, index: u32) -> Option<&mut IcSite> {
        self.sites.get_mut(index as usize)
    }

    /// Record a cache hit.
    #[inline]
    pub fn record_hit(&self) {
        self.hits.fetch_add(1, Ordering::Relaxed);
    }

    /// Record a cache miss.
    #[inline]
    pub fn record_miss(&self) {
        self.misses.fetch_add(1, Ordering::Relaxed);
    }

    /// Get the number of IC sites.
    #[inline]
    pub fn len(&self) -> usize {
        self.sites.len()
    }

    /// Check if empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.sites.is_empty()
    }

    /// Get statistics snapshot.
    pub fn stats(&self) -> IcStats {
        let mut stats = IcStats {
            total_sites: self.sites.len() as u32,
            hits: self.hits.load(Ordering::Relaxed),
            misses: self.misses.load(Ordering::Relaxed),
            ..Default::default()
        };

        for site in &self.sites {
            match site.header.state() {
                IcState::Monomorphic => stats.monomorphic += 1,
                IcState::Polymorphic => stats.polymorphic += 1,
                IcState::Megamorphic => stats.megamorphic += 1,
                IcState::Uninitialized => {}
            }
        }

        stats
    }

    /// Get shape version.
    #[inline]
    pub fn shape_version(&self) -> ShapeVersion {
        self.shape_version
    }

    /// Reset all IC sites (after shape invalidation).
    pub fn reset_all(&mut self, new_version: ShapeVersion) {
        for site in &mut self.sites {
            site.reset();
        }
        self.shape_version = new_version;
        self.hits.store(0, Ordering::Relaxed);
        self.misses.store(0, Ordering::Relaxed);
    }

    /// Iterate over all IC sites.
    #[inline]
    pub fn iter(&self) -> impl Iterator<Item = &IcSite> {
        self.sites.iter()
    }

    /// Iterate mutably over all IC sites.
    #[inline]
    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut IcSite> {
        self.sites.iter_mut()
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -------------------------------------------------------------------------
    // IcState Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_ic_state_default() {
        assert_eq!(IcState::default(), IcState::Uninitialized);
    }

    #[test]
    fn test_ic_state_from_u8() {
        assert_eq!(IcState::from_u8(0), IcState::Uninitialized);
        assert_eq!(IcState::from_u8(1), IcState::Monomorphic);
        assert_eq!(IcState::from_u8(2), IcState::Polymorphic);
        assert_eq!(IcState::from_u8(3), IcState::Megamorphic);
        assert_eq!(IcState::from_u8(255), IcState::Megamorphic); // Invalid → mega
    }

    #[test]
    fn test_ic_state_has_feedback() {
        assert!(!IcState::Uninitialized.has_feedback());
        assert!(IcState::Monomorphic.has_feedback());
        assert!(IcState::Polymorphic.has_feedback());
        assert!(IcState::Megamorphic.has_feedback());
    }

    #[test]
    fn test_ic_state_is_monomorphic() {
        assert!(!IcState::Uninitialized.is_monomorphic());
        assert!(IcState::Monomorphic.is_monomorphic());
        assert!(!IcState::Polymorphic.is_monomorphic());
        assert!(!IcState::Megamorphic.is_monomorphic());
    }

    #[test]
    fn test_ic_state_can_specialize() {
        assert!(IcState::Uninitialized.can_specialize());
        assert!(IcState::Monomorphic.can_specialize());
        assert!(IcState::Polymorphic.can_specialize());
        assert!(!IcState::Megamorphic.can_specialize());
    }

    // -------------------------------------------------------------------------
    // IcKind Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_ic_kind_is_property_op() {
        assert!(IcKind::GetProperty.is_property_op());
        assert!(IcKind::SetProperty.is_property_op());
        assert!(IcKind::DelProperty.is_property_op());
        assert!(!IcKind::GetItem.is_property_op());
        assert!(!IcKind::Call.is_property_op());
    }

    #[test]
    fn test_ic_kind_is_item_op() {
        assert!(IcKind::GetItem.is_item_op());
        assert!(IcKind::SetItem.is_item_op());
        assert!(!IcKind::GetProperty.is_item_op());
        assert!(!IcKind::Call.is_item_op());
    }

    // -------------------------------------------------------------------------
    // IcSiteHeader Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_ic_site_header_new() {
        let header = IcSiteHeader::new(100, IcKind::GetProperty, 0);
        assert_eq!(header.bytecode_offset, 100);
        assert_eq!(header.kind, IcKind::GetProperty);
        assert_eq!(header.state(), IcState::Uninitialized);
        assert_eq!(header.miss_count(), 0);
        assert_eq!(header.data_index, 0);
    }

    #[test]
    fn test_ic_site_header_size() {
        // Ensure header is exactly 16 bytes for cache alignment
        assert_eq!(std::mem::size_of::<IcSiteHeader>(), 16);
        assert_eq!(std::mem::align_of::<IcSiteHeader>(), 16);
    }

    #[test]
    fn test_ic_site_header_transition() {
        let header = IcSiteHeader::new(0, IcKind::GetProperty, 0);

        // Forward transitions should succeed
        assert!(header.transition_to(IcState::Monomorphic));
        assert_eq!(header.state(), IcState::Monomorphic);

        assert!(header.transition_to(IcState::Polymorphic));
        assert_eq!(header.state(), IcState::Polymorphic);

        assert!(header.transition_to(IcState::Megamorphic));
        assert_eq!(header.state(), IcState::Megamorphic);

        // Backward transitions should fail
        assert!(!header.transition_to(IcState::Polymorphic));
        assert_eq!(header.state(), IcState::Megamorphic);

        assert!(!header.transition_to(IcState::Uninitialized));
        assert_eq!(header.state(), IcState::Megamorphic);
    }

    #[test]
    fn test_ic_site_header_skip_transition() {
        let header = IcSiteHeader::new(0, IcKind::GetProperty, 0);

        // Can skip directly to megamorphic
        assert!(header.transition_to(IcState::Megamorphic));
        assert_eq!(header.state(), IcState::Megamorphic);
    }

    #[test]
    fn test_ic_site_header_record_miss() {
        let header = IcSiteHeader::new(0, IcKind::GetProperty, 0);

        assert_eq!(header.miss_count(), 0);
        assert_eq!(header.record_miss(), 1);
        assert_eq!(header.miss_count(), 1);
        assert_eq!(header.record_miss(), 2);
        assert_eq!(header.miss_count(), 2);
    }

    #[test]
    fn test_ic_site_header_miss_saturation() {
        let header = IcSiteHeader::new(0, IcKind::GetProperty, 0);

        // Miss count should saturate at 255
        for _ in 0..300 {
            header.record_miss();
        }
        assert_eq!(header.miss_count(), 255);
    }

    #[test]
    fn test_ic_site_header_reset() {
        let header = IcSiteHeader::new(0, IcKind::GetProperty, 0);

        header.transition_to(IcState::Monomorphic);
        header.record_miss();
        header.record_miss();

        header.reset();

        assert_eq!(header.state(), IcState::Uninitialized);
        assert_eq!(header.miss_count(), 0);
    }

    // -------------------------------------------------------------------------
    // IcManager Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_ic_manager_new() {
        let mgr = IcManager::new(ShapeVersion::new(0));
        assert!(mgr.is_empty());
        assert_eq!(mgr.len(), 0);
    }

    #[test]
    fn test_ic_manager_with_capacity() {
        let mgr = IcManager::with_capacity(16, ShapeVersion::new(0));
        assert!(mgr.is_empty());
    }

    #[test]
    fn test_ic_manager_alloc_property() {
        let mut mgr = IcManager::new(ShapeVersion::new(0));

        let idx = mgr.alloc_property_ic(100, IcKind::GetProperty);
        assert_eq!(idx, Some(0));
        assert_eq!(mgr.len(), 1);

        let site = mgr.get(0).unwrap();
        assert_eq!(site.header.bytecode_offset, 100);
        assert_eq!(site.header.kind, IcKind::GetProperty);
        assert!(site.property_ic().is_some());
        assert!(site.call_ic().is_none());
    }

    #[test]
    fn test_ic_manager_alloc_call() {
        let mut mgr = IcManager::new(ShapeVersion::new(0));

        let idx = mgr.alloc_call_ic(200);
        assert_eq!(idx, Some(0));
        assert_eq!(mgr.len(), 1);

        let site = mgr.get(0).unwrap();
        assert_eq!(site.header.bytecode_offset, 200);
        assert_eq!(site.header.kind, IcKind::Call);
        assert!(site.property_ic().is_none());
        assert!(site.call_ic().is_some());
    }

    #[test]
    fn test_ic_manager_multiple_allocs() {
        let mut mgr = IcManager::new(ShapeVersion::new(0));

        let idx1 = mgr.alloc_property_ic(10, IcKind::GetProperty);
        let idx2 = mgr.alloc_property_ic(20, IcKind::SetProperty);
        let idx3 = mgr.alloc_call_ic(30);

        assert_eq!(idx1, Some(0));
        assert_eq!(idx2, Some(1));
        assert_eq!(idx3, Some(2));
        assert_eq!(mgr.len(), 3);
    }

    #[test]
    fn test_ic_manager_get_out_of_bounds() {
        let mgr = IcManager::new(ShapeVersion::new(0));
        assert!(mgr.get(0).is_none());
        assert!(mgr.get(100).is_none());
    }

    #[test]
    fn test_ic_manager_hit_miss_counters() {
        let mgr = IcManager::new(ShapeVersion::new(0));

        mgr.record_hit();
        mgr.record_hit();
        mgr.record_miss();

        let stats = mgr.stats();
        assert_eq!(stats.hits, 2);
        assert_eq!(stats.misses, 1);
    }

    #[test]
    fn test_ic_manager_stats() {
        let mut mgr = IcManager::new(ShapeVersion::new(0));

        mgr.alloc_property_ic(10, IcKind::GetProperty);
        mgr.alloc_property_ic(20, IcKind::SetProperty);

        // Transition one to monomorphic
        mgr.get_mut(0)
            .unwrap()
            .header
            .transition_to(IcState::Monomorphic);

        let stats = mgr.stats();
        assert_eq!(stats.total_sites, 2);
        assert_eq!(stats.monomorphic, 1);
        assert_eq!(stats.polymorphic, 0);
        assert_eq!(stats.megamorphic, 0);
    }

    #[test]
    fn test_ic_manager_reset_all() {
        let mut mgr = IcManager::new(ShapeVersion::new(0));
        mgr.alloc_property_ic(10, IcKind::GetProperty);
        mgr.alloc_property_ic(20, IcKind::SetProperty);

        // Transition to various states
        mgr.get_mut(0)
            .unwrap()
            .header
            .transition_to(IcState::Monomorphic);
        mgr.get_mut(1)
            .unwrap()
            .header
            .transition_to(IcState::Polymorphic);

        mgr.record_hit();
        mgr.record_miss();

        // Reset all
        mgr.reset_all(ShapeVersion::new(1));

        let stats = mgr.stats();
        assert_eq!(stats.monomorphic, 0);
        assert_eq!(stats.polymorphic, 0);
        assert_eq!(stats.hits, 0);
        assert_eq!(stats.misses, 0);
        assert_eq!(mgr.shape_version().value(), 1);

        // All sites should be uninitialized
        for site in mgr.iter() {
            assert_eq!(site.header.state(), IcState::Uninitialized);
        }
    }

    #[test]
    fn test_ic_stats_hit_rate() {
        let stats = IcStats {
            hits: 80,
            misses: 20,
            ..Default::default()
        };
        assert!((stats.hit_rate() - 0.8).abs() < 0.001);

        let empty_stats = IcStats::default();
        assert_eq!(empty_stats.hit_rate(), 0.0);
    }

    #[test]
    fn test_ic_manager_capacity_limit() {
        let mut mgr = IcManager::new(ShapeVersion::new(0));

        // Allocate up to limit
        for i in 0..MAX_IC_SITES_PER_FUNCTION {
            let result = mgr.alloc_property_ic(i as u32, IcKind::GetProperty);
            assert!(result.is_some(), "Failed at index {}", i);
        }

        // Next allocation should fail
        assert!(mgr.alloc_property_ic(999, IcKind::GetProperty).is_none());
        assert!(mgr.alloc_call_ic(999).is_none());
    }

    #[test]
    fn test_ic_site_reset() {
        let mut mgr = IcManager::new(ShapeVersion::new(0));
        mgr.alloc_property_ic(10, IcKind::GetProperty);

        let site = mgr.get_mut(0).unwrap();
        site.header.transition_to(IcState::Monomorphic);
        site.header.record_miss();

        site.reset();

        assert_eq!(site.header.state(), IcState::Uninitialized);
        assert_eq!(site.header.miss_count(), 0);
    }

    // -------------------------------------------------------------------------
    // Concurrent Access Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_ic_header_concurrent_transitions() {
        use std::sync::Arc;
        use std::thread;

        let header = Arc::new(IcSiteHeader::new(0, IcKind::GetProperty, 0));
        let mut handles = vec![];

        // Spawn threads that all try to transition
        for _ in 0..10 {
            let h = Arc::clone(&header);
            handles.push(thread::spawn(move || {
                h.transition_to(IcState::Monomorphic);
                h.transition_to(IcState::Polymorphic);
                h.transition_to(IcState::Megamorphic);
            }));
        }

        for handle in handles {
            handle.join().unwrap();
        }

        // Should end up at Megamorphic
        assert_eq!(header.state(), IcState::Megamorphic);
    }

    #[test]
    fn test_ic_manager_concurrent_hit_miss() {
        use std::sync::Arc;
        use std::thread;

        let mgr = Arc::new(IcManager::new(ShapeVersion::new(0)));
        let mut handles = vec![];

        for _ in 0..10 {
            let m = Arc::clone(&mgr);
            handles.push(thread::spawn(move || {
                for _ in 0..1000 {
                    m.record_hit();
                    m.record_miss();
                }
            }));
        }

        for handle in handles {
            handle.join().unwrap();
        }

        let stats = mgr.stats();
        assert_eq!(stats.hits, 10 * 1000);
        assert_eq!(stats.misses, 10 * 1000);
    }
}
