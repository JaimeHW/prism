//! Call Inline Cache
//!
//! Caches function/method call targets for direct invocation without dispatch overhead.
//!
//! # Design
//!
//! Call ICs handle:
//! - Direct function calls: `func(args)` → cache function pointer
//! - Method calls: `obj.method(args)` → cache bound method
//! - Constructor calls: `MyClass(args)` → cache __init__ method
//!
//! The cache stores the expected callee identity and provides a fast path
//! when the same function is called repeatedly.

use super::IcState;
use prism_runtime::object::shape::ShapeId;

// =============================================================================
// Call IC State
// =============================================================================

/// State of a call inline cache.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum CallIcState {
    /// No callee cached yet.
    Uninitialized = 0,
    /// Single callee cached - direct call.
    Monomorphic = 1,
    /// Multiple callees - dispatch table.
    Polymorphic = 2,
    /// Too many callees - no caching.
    Megamorphic = 3,
}

impl From<IcState> for CallIcState {
    fn from(s: IcState) -> Self {
        match s {
            IcState::Uninitialized => CallIcState::Uninitialized,
            IcState::Monomorphic => CallIcState::Monomorphic,
            IcState::Polymorphic => CallIcState::Polymorphic,
            IcState::Megamorphic => CallIcState::Megamorphic,
        }
    }
}

impl Default for CallIcState {
    fn default() -> Self {
        Self::Uninitialized
    }
}

// =============================================================================
// Call IC Data
// =============================================================================

/// Cached data for a function call.
///
/// Stores enough information to make a direct call without lookup.
#[derive(Debug, Clone)]
pub struct CallIcData {
    /// Expected function identity (pointer or unique ID).
    pub callee_id: u64,

    /// Direct code pointer (entry point address).
    pub code_ptr: usize,

    /// Expected argument count (for fast arity check).
    pub expected_argc: u16,

    /// Whether this is a bound method (needs implicit self).
    pub is_method: bool,

    /// Shape of receiver for method calls.
    pub receiver_shape: Option<ShapeId>,
}

impl CallIcData {
    /// Create new call data for a function.
    #[inline]
    pub fn function(callee_id: u64, code_ptr: usize, argc: u16) -> Self {
        Self {
            callee_id,
            code_ptr,
            expected_argc: argc,
            is_method: false,
            receiver_shape: None,
        }
    }

    /// Create new call data for a method.
    #[inline]
    pub fn method(callee_id: u64, code_ptr: usize, argc: u16, receiver_shape: ShapeId) -> Self {
        Self {
            callee_id,
            code_ptr,
            expected_argc: argc,
            is_method: true,
            receiver_shape: Some(receiver_shape),
        }
    }

    /// Check if this data matches a callee.
    #[inline]
    pub fn matches_function(&self, callee_id: u64) -> bool {
        self.callee_id == callee_id && !self.is_method
    }

    /// Check if this data matches a method on a shape.
    #[inline]
    pub fn matches_method(&self, callee_id: u64, receiver_shape: ShapeId) -> bool {
        self.callee_id == callee_id && self.is_method && self.receiver_shape == Some(receiver_shape)
    }
}

impl Default for CallIcData {
    fn default() -> Self {
        Self {
            callee_id: 0,
            code_ptr: 0,
            expected_argc: 0,
            is_method: false,
            receiver_shape: None,
        }
    }
}

// =============================================================================
// Polymorphic Call Entry
// =============================================================================

/// Entry in a polymorphic call cache.
#[derive(Debug, Clone, Default)]
pub struct PolyCallEntry {
    /// Cached call data.
    pub data: CallIcData,
    /// Access count for LRU.
    pub access_count: u32,
    /// Whether this entry is valid.
    pub valid: bool,
}

impl PolyCallEntry {
    /// Create a new entry.
    #[inline]
    pub fn new(data: CallIcData) -> Self {
        Self {
            data,
            access_count: 0,
            valid: true,
        }
    }

    /// Create an empty entry.
    #[inline]
    pub fn empty() -> Self {
        Self {
            data: CallIcData::default(),
            access_count: 0,
            valid: false,
        }
    }

    /// Touch for LRU tracking.
    #[inline]
    pub fn touch(&mut self) {
        self.access_count = self.access_count.saturating_add(1);
    }
}

// =============================================================================
// Call IC
// =============================================================================

/// Number of entries in polymorphic call IC.
const POLY_CALL_ENTRIES: usize = 4;

/// Complete call inline cache.
#[derive(Debug)]
pub struct CallIc {
    /// Current state.
    state: CallIcState,

    /// Monomorphic data.
    mono: CallIcData,

    /// Polymorphic entries.
    poly: [PolyCallEntry; POLY_CALL_ENTRIES],

    /// Number of valid poly entries.
    poly_count: u8,

    /// Statistics.
    hits: u64,
    misses: u64,
}

impl CallIc {
    /// Create a new uninitialized call IC.
    #[inline]
    pub fn new() -> Self {
        Self {
            state: CallIcState::Uninitialized,
            mono: CallIcData::default(),
            poly: std::array::from_fn(|_| PolyCallEntry::empty()),
            poly_count: 0,
            hits: 0,
            misses: 0,
        }
    }

    /// Get current state.
    #[inline]
    pub const fn state(&self) -> CallIcState {
        self.state
    }

    /// Lookup for a function call.
    ///
    /// Returns code pointer if hit.
    #[inline]
    pub fn lookup_function(&mut self, callee_id: u64) -> Option<usize> {
        match self.state {
            CallIcState::Uninitialized => {
                self.misses += 1;
                None
            }

            CallIcState::Monomorphic => {
                if self.mono.matches_function(callee_id) {
                    self.hits += 1;
                    Some(self.mono.code_ptr)
                } else {
                    self.misses += 1;
                    None
                }
            }

            CallIcState::Polymorphic => {
                for i in 0..(self.poly_count as usize) {
                    if self.poly[i].valid && self.poly[i].data.matches_function(callee_id) {
                        self.poly[i].touch();
                        self.hits += 1;
                        return Some(self.poly[i].data.code_ptr);
                    }
                }
                self.misses += 1;
                None
            }

            CallIcState::Megamorphic => {
                self.misses += 1;
                None
            }
        }
    }

    /// Lookup for a method call.
    ///
    /// Returns code pointer if hit.
    #[inline]
    pub fn lookup_method(&mut self, callee_id: u64, receiver_shape: ShapeId) -> Option<usize> {
        match self.state {
            CallIcState::Uninitialized => {
                self.misses += 1;
                None
            }

            CallIcState::Monomorphic => {
                if self.mono.matches_method(callee_id, receiver_shape) {
                    self.hits += 1;
                    Some(self.mono.code_ptr)
                } else {
                    self.misses += 1;
                    None
                }
            }

            CallIcState::Polymorphic => {
                for i in 0..(self.poly_count as usize) {
                    if self.poly[i].valid
                        && self.poly[i].data.matches_method(callee_id, receiver_shape)
                    {
                        self.poly[i].touch();
                        self.hits += 1;
                        return Some(self.poly[i].data.code_ptr);
                    }
                }
                self.misses += 1;
                None
            }

            CallIcState::Megamorphic => {
                self.misses += 1;
                None
            }
        }
    }

    /// Update with function call data.
    pub fn update_function(&mut self, callee_id: u64, code_ptr: usize, argc: u16) {
        let data = CallIcData::function(callee_id, code_ptr, argc);
        self.update(data);
    }

    /// Update with method call data.
    pub fn update_method(
        &mut self,
        callee_id: u64,
        code_ptr: usize,
        argc: u16,
        receiver_shape: ShapeId,
    ) {
        let data = CallIcData::method(callee_id, code_ptr, argc, receiver_shape);
        self.update(data);
    }

    /// Update the IC with new call data.
    fn update(&mut self, data: CallIcData) {
        match self.state {
            CallIcState::Uninitialized => {
                self.mono = data;
                self.state = CallIcState::Monomorphic;
            }

            CallIcState::Monomorphic => {
                // Different callee - transition to polymorphic
                if self.mono.callee_id != data.callee_id {
                    // Copy mono to poly[0]
                    self.poly[0] = PolyCallEntry::new(self.mono.clone());
                    self.poly[1] = PolyCallEntry::new(data);
                    self.poly_count = 2;
                    self.state = CallIcState::Polymorphic;
                }
            }

            CallIcState::Polymorphic => {
                // Check if already present
                for i in 0..(self.poly_count as usize) {
                    if self.poly[i].valid && self.poly[i].data.callee_id == data.callee_id {
                        return; // Already cached
                    }
                }

                // Try to add
                if (self.poly_count as usize) < POLY_CALL_ENTRIES {
                    let idx = self.poly_count as usize;
                    self.poly[idx] = PolyCallEntry::new(data);
                    self.poly_count += 1;
                } else {
                    // Cache full - go megamorphic
                    self.state = CallIcState::Megamorphic;
                }
            }

            CallIcState::Megamorphic => {
                // No caching
            }
        }
    }

    /// Reset the IC.
    pub fn reset(&mut self) {
        self.state = CallIcState::Uninitialized;
        self.mono = CallIcData::default();
        for entry in &mut self.poly {
            *entry = PolyCallEntry::empty();
        }
        self.poly_count = 0;
        self.hits = 0;
        self.misses = 0;
    }

    /// Get hit count.
    #[inline]
    pub const fn hits(&self) -> u64 {
        self.hits
    }

    /// Get miss count.
    #[inline]
    pub const fn misses(&self) -> u64 {
        self.misses
    }

    /// Get hit rate.
    pub fn hit_rate(&self) -> f64 {
        let total = self.hits + self.misses;
        if total == 0 {
            0.0
        } else {
            self.hits as f64 / total as f64
        }
    }
}

impl Default for CallIc {
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

    // -------------------------------------------------------------------------
    // CallIcData Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_call_ic_data_function() {
        let data = CallIcData::function(100, 0x1000, 2);
        assert_eq!(data.callee_id, 100);
        assert_eq!(data.code_ptr, 0x1000);
        assert_eq!(data.expected_argc, 2);
        assert!(!data.is_method);
        assert!(data.receiver_shape.is_none());
    }

    #[test]
    fn test_call_ic_data_method() {
        let data = CallIcData::method(100, 0x2000, 3, ShapeId(42));
        assert_eq!(data.callee_id, 100);
        assert_eq!(data.code_ptr, 0x2000);
        assert_eq!(data.expected_argc, 3);
        assert!(data.is_method);
        assert_eq!(data.receiver_shape, Some(ShapeId(42)));
    }

    #[test]
    fn test_call_ic_data_matches_function() {
        let data = CallIcData::function(100, 0x1000, 2);
        assert!(data.matches_function(100));
        assert!(!data.matches_function(101));
    }

    #[test]
    fn test_call_ic_data_matches_method() {
        let data = CallIcData::method(100, 0x1000, 2, ShapeId(42));
        assert!(data.matches_method(100, ShapeId(42)));
        assert!(!data.matches_method(100, ShapeId(43)));
        assert!(!data.matches_method(101, ShapeId(42)));
    }

    // -------------------------------------------------------------------------
    // CallIc Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_call_ic_new() {
        let ic = CallIc::new();
        assert_eq!(ic.state(), CallIcState::Uninitialized);
        assert_eq!(ic.hits(), 0);
        assert_eq!(ic.misses(), 0);
    }

    #[test]
    fn test_call_ic_lookup_uninitialized() {
        let mut ic = CallIc::new();
        assert!(ic.lookup_function(100).is_none());
        assert_eq!(ic.misses(), 1);
    }

    #[test]
    fn test_call_ic_update_to_mono() {
        let mut ic = CallIc::new();
        ic.update_function(100, 0x1000, 2);

        assert_eq!(ic.state(), CallIcState::Monomorphic);

        let result = ic.lookup_function(100);
        assert_eq!(result, Some(0x1000));
        assert_eq!(ic.hits(), 1);
    }

    #[test]
    fn test_call_ic_mono_miss() {
        let mut ic = CallIc::new();
        ic.update_function(100, 0x1000, 2);

        assert!(ic.lookup_function(101).is_none());
        assert_eq!(ic.misses(), 1);
    }

    #[test]
    fn test_call_ic_mono_to_poly() {
        let mut ic = CallIc::new();

        ic.update_function(100, 0x1000, 2);
        assert_eq!(ic.state(), CallIcState::Monomorphic);

        ic.update_function(200, 0x2000, 3);
        assert_eq!(ic.state(), CallIcState::Polymorphic);

        // Both should be accessible
        assert_eq!(ic.lookup_function(100), Some(0x1000));
        assert_eq!(ic.lookup_function(200), Some(0x2000));
    }

    #[test]
    fn test_call_ic_poly_to_mega() {
        let mut ic = CallIc::new();

        // Fill beyond capacity
        for i in 0..(POLY_CALL_ENTRIES + 2) {
            ic.update_function(i as u64, 0x1000 + i * 0x100, 0);
        }

        assert_eq!(ic.state(), CallIcState::Megamorphic);
    }

    #[test]
    fn test_call_ic_method_lookup() {
        let mut ic = CallIc::new();
        ic.update_method(100, 0x1000, 2, ShapeId(42));

        let result = ic.lookup_method(100, ShapeId(42));
        assert_eq!(result, Some(0x1000));
        assert_eq!(ic.hits(), 1);

        // Wrong shape
        assert!(ic.lookup_method(100, ShapeId(43)).is_none());
    }

    #[test]
    fn test_call_ic_reset() {
        let mut ic = CallIc::new();
        ic.update_function(100, 0x1000, 2);
        ic.lookup_function(100);
        ic.lookup_function(200);

        ic.reset();

        assert_eq!(ic.state(), CallIcState::Uninitialized);
        assert_eq!(ic.hits(), 0);
        assert_eq!(ic.misses(), 0);
    }

    #[test]
    fn test_call_ic_hit_rate() {
        let mut ic = CallIc::new();
        ic.update_function(100, 0x1000, 0);

        ic.lookup_function(100); // hit
        ic.lookup_function(100); // hit
        ic.lookup_function(100); // hit
        ic.lookup_function(200); // miss

        assert!((ic.hit_rate() - 0.75).abs() < 0.001);
    }

    #[test]
    fn test_call_ic_same_callee_no_transition() {
        let mut ic = CallIc::new();

        ic.update_function(100, 0x1000, 2);
        assert_eq!(ic.state(), CallIcState::Monomorphic);

        // Same callee shouldn't transition
        ic.update_function(100, 0x1000, 2);
        assert_eq!(ic.state(), CallIcState::Monomorphic);
    }

    #[test]
    fn test_call_ic_poly_dedup() {
        let mut ic = CallIc::new();

        ic.update_function(100, 0x1000, 0);
        ic.update_function(200, 0x2000, 0);
        assert_eq!(ic.poly_count, 2);

        // Same callee shouldn't add new entry
        ic.update_function(200, 0x2000, 0);
        assert_eq!(ic.poly_count, 2);
    }

    #[test]
    fn test_call_ic_state_from_ic_state() {
        assert_eq!(
            CallIcState::from(IcState::Uninitialized),
            CallIcState::Uninitialized
        );
        assert_eq!(
            CallIcState::from(IcState::Monomorphic),
            CallIcState::Monomorphic
        );
        assert_eq!(
            CallIcState::from(IcState::Polymorphic),
            CallIcState::Polymorphic
        );
        assert_eq!(
            CallIcState::from(IcState::Megamorphic),
            CallIcState::Megamorphic
        );
    }

    #[test]
    fn test_poly_call_entry() {
        let data = CallIcData::function(100, 0x1000, 2);
        let mut entry = PolyCallEntry::new(data);

        assert!(entry.valid);
        assert_eq!(entry.access_count, 0);

        entry.touch();
        assert_eq!(entry.access_count, 1);
    }

    #[test]
    fn test_poly_call_entry_empty() {
        let entry = PolyCallEntry::empty();
        assert!(!entry.valid);
    }
}
