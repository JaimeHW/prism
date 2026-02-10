//! Exception flags bitfield.
//!
//! This module defines `ExceptionFlags`, a compact bitfield that tracks
//! the state of an exception object without wasting memory on separate booleans.
//!
//! # Performance Design
//!
//! - **Single byte**: All flags packed into 1 byte
//! - **Branchless checks**: Flag testing is a single AND instruction
//! - **Lazy initialization tracking**: Enables deferred allocation of args/traceback

use std::fmt;

/// Compact flags for exception state.
///
/// Packed into a single byte for optimal memory layout.
/// Each flag can be tested with a single AND instruction.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct ExceptionFlags(u8);

impl ExceptionFlags {
    // ════════════════════════════════════════════════════════════════════════
    // Flag Bit Positions
    // ════════════════════════════════════════════════════════════════════════

    /// Exception has been normalized (args set).
    const NORMALIZED: u8 = 1 << 0;

    /// Exception has args tuple allocated.
    const HAS_ARGS: u8 = 1 << 1;

    /// Exception has traceback attached.
    const HAS_TRACEBACK: u8 = 1 << 2;

    /// Exception has __cause__ set (explicit chaining).
    const HAS_CAUSE: u8 = 1 << 3;

    /// Exception has __context__ set (implicit chaining).
    const HAS_CONTEXT: u8 = 1 << 4;

    /// Exception's __suppress_context__ is True.
    const SUPPRESS_CONTEXT: u8 = 1 << 5;

    /// Exception is a flyweight (singleton, not heap-allocated).
    const FLYWEIGHT: u8 = 1 << 6;

    /// Exception was re-raised (traceback was extended).
    const RERAISED: u8 = 1 << 7;

    // ════════════════════════════════════════════════════════════════════════
    // Constructors
    // ════════════════════════════════════════════════════════════════════════

    /// Creates empty flags (no bits set).
    #[inline(always)]
    pub const fn empty() -> Self {
        Self(0)
    }

    /// Creates flags with all bits set.
    #[inline(always)]
    pub const fn all() -> Self {
        Self(0xFF)
    }

    /// Creates flags from raw byte value.
    #[inline(always)]
    pub const fn from_raw(value: u8) -> Self {
        Self(value)
    }

    /// Creates flags for a flyweight exception.
    #[inline(always)]
    pub const fn flyweight() -> Self {
        Self(Self::FLYWEIGHT | Self::NORMALIZED)
    }

    /// Creates flags for a newly created exception.
    #[inline(always)]
    pub const fn new_exception() -> Self {
        Self(0)
    }

    /// Creates flags for an exception with args.
    #[inline(always)]
    pub const fn with_args() -> Self {
        Self(Self::HAS_ARGS | Self::NORMALIZED)
    }

    // ════════════════════════════════════════════════════════════════════════
    // Flag Getters (Branchless)
    // ════════════════════════════════════════════════════════════════════════

    /// Returns the raw byte value.
    #[inline(always)]
    pub const fn as_raw(self) -> u8 {
        self.0
    }

    /// Returns true if no flags are set.
    #[inline(always)]
    pub const fn is_empty(self) -> bool {
        self.0 == 0
    }

    /// Returns true if the exception has been normalized.
    #[inline(always)]
    pub const fn is_normalized(self) -> bool {
        self.0 & Self::NORMALIZED != 0
    }

    /// Returns true if the exception has args allocated.
    #[inline(always)]
    pub const fn has_args(self) -> bool {
        self.0 & Self::HAS_ARGS != 0
    }

    /// Returns true if the exception has a traceback.
    #[inline(always)]
    pub const fn has_traceback(self) -> bool {
        self.0 & Self::HAS_TRACEBACK != 0
    }

    /// Returns true if the exception has __cause__ set.
    #[inline(always)]
    pub const fn has_cause(self) -> bool {
        self.0 & Self::HAS_CAUSE != 0
    }

    /// Returns true if the exception has __context__ set.
    #[inline(always)]
    pub const fn has_context(self) -> bool {
        self.0 & Self::HAS_CONTEXT != 0
    }

    /// Returns true if __suppress_context__ is True.
    #[inline(always)]
    pub const fn suppress_context(self) -> bool {
        self.0 & Self::SUPPRESS_CONTEXT != 0
    }

    /// Returns true if this is a flyweight exception.
    #[inline(always)]
    pub const fn is_flyweight(self) -> bool {
        self.0 & Self::FLYWEIGHT != 0
    }

    /// Returns true if the exception was re-raised.
    #[inline(always)]
    pub const fn was_reraised(self) -> bool {
        self.0 & Self::RERAISED != 0
    }

    // ════════════════════════════════════════════════════════════════════════
    // Flag Setters (Return new value, don't mutate)
    // ════════════════════════════════════════════════════════════════════════

    /// Returns flags with normalized bit set.
    #[inline(always)]
    pub const fn set_normalized(self) -> Self {
        Self(self.0 | Self::NORMALIZED)
    }

    /// Returns flags with has_args bit set.
    #[inline(always)]
    pub const fn set_has_args(self) -> Self {
        Self(self.0 | Self::HAS_ARGS)
    }

    /// Returns flags with has_traceback bit set.
    #[inline(always)]
    pub const fn set_has_traceback(self) -> Self {
        Self(self.0 | Self::HAS_TRACEBACK)
    }

    /// Returns flags with has_cause bit set.
    #[inline(always)]
    pub const fn set_has_cause(self) -> Self {
        Self(self.0 | Self::HAS_CAUSE)
    }

    /// Returns flags with has_context bit set.
    #[inline(always)]
    pub const fn set_has_context(self) -> Self {
        Self(self.0 | Self::HAS_CONTEXT)
    }

    /// Returns flags with suppress_context bit set.
    #[inline(always)]
    pub const fn set_suppress_context(self) -> Self {
        Self(self.0 | Self::SUPPRESS_CONTEXT)
    }

    /// Returns flags with flyweight bit set.
    #[inline(always)]
    pub const fn set_flyweight(self) -> Self {
        Self(self.0 | Self::FLYWEIGHT)
    }

    /// Returns flags with reraised bit set.
    #[inline(always)]
    pub const fn set_reraised(self) -> Self {
        Self(self.0 | Self::RERAISED)
    }

    // ════════════════════════════════════════════════════════════════════════
    // Flag Clearers (Return new value, don't mutate)
    // ════════════════════════════════════════════════════════════════════════

    /// Returns flags with has_traceback bit cleared.
    #[inline(always)]
    pub const fn clear_has_traceback(self) -> Self {
        Self(self.0 & !Self::HAS_TRACEBACK)
    }

    /// Returns flags with has_cause bit cleared.
    #[inline(always)]
    pub const fn clear_has_cause(self) -> Self {
        Self(self.0 & !Self::HAS_CAUSE)
    }

    /// Returns flags with has_context bit cleared.
    #[inline(always)]
    pub const fn clear_has_context(self) -> Self {
        Self(self.0 & !Self::HAS_CONTEXT)
    }

    /// Returns flags with suppress_context bit cleared.
    #[inline(always)]
    pub const fn clear_suppress_context(self) -> Self {
        Self(self.0 & !Self::SUPPRESS_CONTEXT)
    }

    /// Returns flags with reraised bit cleared.
    #[inline(always)]
    pub const fn clear_reraised(self) -> Self {
        Self(self.0 & !Self::RERAISED)
    }

    // ════════════════════════════════════════════════════════════════════════
    // Mutation Methods (For when you have &mut self)
    // ════════════════════════════════════════════════════════════════════════

    /// Sets the normalized bit in place.
    #[inline(always)]
    pub fn mark_normalized(&mut self) {
        self.0 |= Self::NORMALIZED;
    }

    /// Sets the has_args bit in place.
    #[inline(always)]
    pub fn mark_has_args(&mut self) {
        self.0 |= Self::HAS_ARGS;
    }

    /// Sets the has_traceback bit in place.
    #[inline(always)]
    pub fn mark_has_traceback(&mut self) {
        self.0 |= Self::HAS_TRACEBACK;
    }

    /// Sets the has_cause bit in place.
    #[inline(always)]
    pub fn mark_has_cause(&mut self) {
        self.0 |= Self::HAS_CAUSE;
    }

    /// Sets the has_context bit in place.
    #[inline(always)]
    pub fn mark_has_context(&mut self) {
        self.0 |= Self::HAS_CONTEXT;
    }

    /// Sets the suppress_context bit in place.
    #[inline(always)]
    pub fn mark_suppress_context(&mut self) {
        self.0 |= Self::SUPPRESS_CONTEXT;
    }

    /// Sets the reraised bit in place.
    #[inline(always)]
    pub fn mark_reraised(&mut self) {
        self.0 |= Self::RERAISED;
    }

    // ════════════════════════════════════════════════════════════════════════
    // Bitwise Operations
    // ════════════════════════════════════════════════════════════════════════

    /// Returns flags with specified bits ORed in.
    #[inline(always)]
    pub const fn with(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }

    /// Returns flags with specified bits ANDed out.
    #[inline(always)]
    pub const fn without(self, mask: Self) -> Self {
        Self(self.0 & !mask.0)
    }

    /// Returns true if all bits in mask are set.
    #[inline(always)]
    pub const fn contains(self, mask: Self) -> bool {
        self.0 & mask.0 == mask.0
    }

    /// Returns true if any bits in mask are set.
    #[inline(always)]
    pub const fn intersects(self, mask: Self) -> bool {
        self.0 & mask.0 != 0
    }

    /// Counts the number of set bits.
    #[inline(always)]
    pub const fn count_ones(self) -> u32 {
        self.0.count_ones()
    }
}

impl fmt::Debug for ExceptionFlags {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut flags = Vec::new();

        if self.is_normalized() {
            flags.push("NORMALIZED");
        }
        if self.has_args() {
            flags.push("HAS_ARGS");
        }
        if self.has_traceback() {
            flags.push("HAS_TRACEBACK");
        }
        if self.has_cause() {
            flags.push("HAS_CAUSE");
        }
        if self.has_context() {
            flags.push("HAS_CONTEXT");
        }
        if self.suppress_context() {
            flags.push("SUPPRESS_CONTEXT");
        }
        if self.is_flyweight() {
            flags.push("FLYWEIGHT");
        }
        if self.was_reraised() {
            flags.push("RERAISED");
        }

        if flags.is_empty() {
            write!(f, "ExceptionFlags(empty)")
        } else {
            write!(f, "ExceptionFlags({})", flags.join(" | "))
        }
    }
}

impl std::ops::BitOr for ExceptionFlags {
    type Output = Self;

    #[inline(always)]
    fn bitor(self, rhs: Self) -> Self {
        Self(self.0 | rhs.0)
    }
}

impl std::ops::BitOrAssign for ExceptionFlags {
    #[inline(always)]
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0;
    }
}

impl std::ops::BitAnd for ExceptionFlags {
    type Output = Self;

    #[inline(always)]
    fn bitand(self, rhs: Self) -> Self {
        Self(self.0 & rhs.0)
    }
}

impl std::ops::BitAndAssign for ExceptionFlags {
    #[inline(always)]
    fn bitand_assign(&mut self, rhs: Self) {
        self.0 &= rhs.0;
    }
}

impl std::ops::Not for ExceptionFlags {
    type Output = Self;

    #[inline(always)]
    fn not(self) -> Self {
        Self(!self.0)
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ════════════════════════════════════════════════════════════════════════
    // Constructor Tests
    // ════════════════════════════════════════════════════════════════════════

    #[test]
    fn test_empty_flags() {
        let flags = ExceptionFlags::empty();
        assert!(flags.is_empty());
        assert_eq!(flags.as_raw(), 0);
    }

    #[test]
    fn test_all_flags() {
        let flags = ExceptionFlags::all();
        assert!(!flags.is_empty());
        assert_eq!(flags.as_raw(), 0xFF);
    }

    #[test]
    fn test_from_raw() {
        let flags = ExceptionFlags::from_raw(0b00000011);
        assert!(flags.is_normalized());
        assert!(flags.has_args());
        assert!(!flags.has_traceback());
    }

    #[test]
    fn test_flyweight_preset() {
        let flags = ExceptionFlags::flyweight();
        assert!(flags.is_flyweight());
        assert!(flags.is_normalized());
        assert!(!flags.has_args());
    }

    #[test]
    fn test_new_exception_preset() {
        let flags = ExceptionFlags::new_exception();
        assert!(flags.is_empty());
        assert!(!flags.is_normalized());
    }

    #[test]
    fn test_with_args_preset() {
        let flags = ExceptionFlags::with_args();
        assert!(flags.has_args());
        assert!(flags.is_normalized());
    }

    // ════════════════════════════════════════════════════════════════════════
    // Individual Flag Tests
    // ════════════════════════════════════════════════════════════════════════

    #[test]
    fn test_normalized_flag() {
        let flags = ExceptionFlags::empty().set_normalized();
        assert!(flags.is_normalized());
        assert!(!flags.has_args());
    }

    #[test]
    fn test_has_args_flag() {
        let flags = ExceptionFlags::empty().set_has_args();
        assert!(flags.has_args());
        assert!(!flags.is_normalized());
    }

    #[test]
    fn test_has_traceback_flag() {
        let flags = ExceptionFlags::empty().set_has_traceback();
        assert!(flags.has_traceback());
        assert!(!flags.has_args());
    }

    #[test]
    fn test_has_cause_flag() {
        let flags = ExceptionFlags::empty().set_has_cause();
        assert!(flags.has_cause());
        assert!(!flags.has_context());
    }

    #[test]
    fn test_has_context_flag() {
        let flags = ExceptionFlags::empty().set_has_context();
        assert!(flags.has_context());
        assert!(!flags.has_cause());
    }

    #[test]
    fn test_suppress_context_flag() {
        let flags = ExceptionFlags::empty().set_suppress_context();
        assert!(flags.suppress_context());
    }

    #[test]
    fn test_flyweight_flag() {
        let flags = ExceptionFlags::empty().set_flyweight();
        assert!(flags.is_flyweight());
    }

    #[test]
    fn test_reraised_flag() {
        let flags = ExceptionFlags::empty().set_reraised();
        assert!(flags.was_reraised());
    }

    // ════════════════════════════════════════════════════════════════════════
    // Clear Flag Tests
    // ════════════════════════════════════════════════════════════════════════

    #[test]
    fn test_clear_has_traceback() {
        let flags = ExceptionFlags::all().clear_has_traceback();
        assert!(!flags.has_traceback());
        assert!(flags.has_args()); // Other flags preserved
    }

    #[test]
    fn test_clear_has_cause() {
        let flags = ExceptionFlags::all().clear_has_cause();
        assert!(!flags.has_cause());
        assert!(flags.has_context()); // Other flags preserved
    }

    #[test]
    fn test_clear_has_context() {
        let flags = ExceptionFlags::all().clear_has_context();
        assert!(!flags.has_context());
        assert!(flags.has_cause()); // Other flags preserved
    }

    #[test]
    fn test_clear_suppress_context() {
        let flags = ExceptionFlags::all().clear_suppress_context();
        assert!(!flags.suppress_context());
    }

    #[test]
    fn test_clear_reraised() {
        let flags = ExceptionFlags::all().clear_reraised();
        assert!(!flags.was_reraised());
    }

    // ════════════════════════════════════════════════════════════════════════
    // Mutation Tests
    // ════════════════════════════════════════════════════════════════════════

    #[test]
    fn test_mark_normalized() {
        let mut flags = ExceptionFlags::empty();
        flags.mark_normalized();
        assert!(flags.is_normalized());
    }

    #[test]
    fn test_mark_has_args() {
        let mut flags = ExceptionFlags::empty();
        flags.mark_has_args();
        assert!(flags.has_args());
    }

    #[test]
    fn test_mark_has_traceback() {
        let mut flags = ExceptionFlags::empty();
        flags.mark_has_traceback();
        assert!(flags.has_traceback());
    }

    #[test]
    fn test_mark_has_cause() {
        let mut flags = ExceptionFlags::empty();
        flags.mark_has_cause();
        assert!(flags.has_cause());
    }

    #[test]
    fn test_mark_has_context() {
        let mut flags = ExceptionFlags::empty();
        flags.mark_has_context();
        assert!(flags.has_context());
    }

    #[test]
    fn test_mark_suppress_context() {
        let mut flags = ExceptionFlags::empty();
        flags.mark_suppress_context();
        assert!(flags.suppress_context());
    }

    #[test]
    fn test_mark_reraised() {
        let mut flags = ExceptionFlags::empty();
        flags.mark_reraised();
        assert!(flags.was_reraised());
    }

    // ════════════════════════════════════════════════════════════════════════
    // Combination Tests
    // ════════════════════════════════════════════════════════════════════════

    #[test]
    fn test_multiple_flags() {
        let flags = ExceptionFlags::empty()
            .set_normalized()
            .set_has_args()
            .set_has_traceback();

        assert!(flags.is_normalized());
        assert!(flags.has_args());
        assert!(flags.has_traceback());
        assert!(!flags.has_cause());
    }

    #[test]
    fn test_chained_exception_flags() {
        // Simulating a chained exception
        let flags = ExceptionFlags::empty()
            .set_normalized()
            .set_has_args()
            .set_has_context()
            .set_has_traceback();

        assert!(flags.is_normalized());
        assert!(flags.has_context());
        assert!(!flags.has_cause()); // Implicit chaining only
    }

    #[test]
    fn test_explicit_chained_exception() {
        // Simulating `raise X from Y`
        let flags = ExceptionFlags::empty()
            .set_normalized()
            .set_has_args()
            .set_has_cause()
            .set_suppress_context();

        assert!(flags.has_cause());
        assert!(flags.suppress_context());
    }

    // ════════════════════════════════════════════════════════════════════════
    // Bitwise Operation Tests
    // ════════════════════════════════════════════════════════════════════════

    #[test]
    fn test_with_operation() {
        let a = ExceptionFlags::empty().set_normalized();
        let b = ExceptionFlags::empty().set_has_args();
        let c = a.with(b);

        assert!(c.is_normalized());
        assert!(c.has_args());
    }

    #[test]
    fn test_without_operation() {
        let flags = ExceptionFlags::all();
        let mask =
            ExceptionFlags::from_raw(ExceptionFlags::HAS_ARGS | ExceptionFlags::HAS_TRACEBACK);
        let result = flags.without(mask);

        assert!(!result.has_args());
        assert!(!result.has_traceback());
        assert!(result.is_normalized()); // Other flags preserved
    }

    #[test]
    fn test_contains() {
        let flags = ExceptionFlags::empty()
            .set_normalized()
            .set_has_args()
            .set_has_traceback();

        let mask = ExceptionFlags::empty().set_normalized().set_has_args();

        assert!(flags.contains(mask));
    }

    #[test]
    fn test_contains_fails() {
        let flags = ExceptionFlags::empty().set_normalized();
        let mask = ExceptionFlags::empty().set_normalized().set_has_args();

        assert!(!flags.contains(mask));
    }

    #[test]
    fn test_intersects() {
        let a = ExceptionFlags::empty().set_normalized().set_has_args();
        let b = ExceptionFlags::empty().set_has_args().set_has_traceback();

        assert!(a.intersects(b)); // Both have HAS_ARGS
    }

    #[test]
    fn test_intersects_fails() {
        let a = ExceptionFlags::empty().set_normalized();
        let b = ExceptionFlags::empty().set_has_args();

        assert!(!a.intersects(b)); // No common flags
    }

    #[test]
    fn test_count_ones() {
        assert_eq!(ExceptionFlags::empty().count_ones(), 0);
        assert_eq!(ExceptionFlags::empty().set_normalized().count_ones(), 1);
        assert_eq!(
            ExceptionFlags::empty()
                .set_normalized()
                .set_has_args()
                .count_ones(),
            2
        );
        assert_eq!(ExceptionFlags::all().count_ones(), 8);
    }

    // ════════════════════════════════════════════════════════════════════════
    // Operator Tests
    // ════════════════════════════════════════════════════════════════════════

    #[test]
    fn test_bitor_operator() {
        let a = ExceptionFlags::from_raw(0b00000001);
        let b = ExceptionFlags::from_raw(0b00000010);
        let c = a | b;

        assert_eq!(c.as_raw(), 0b00000011);
    }

    #[test]
    fn test_bitor_assign_operator() {
        let mut flags = ExceptionFlags::from_raw(0b00000001);
        flags |= ExceptionFlags::from_raw(0b00000010);

        assert_eq!(flags.as_raw(), 0b00000011);
    }

    #[test]
    fn test_bitand_operator() {
        let a = ExceptionFlags::from_raw(0b00000011);
        let b = ExceptionFlags::from_raw(0b00000010);
        let c = a & b;

        assert_eq!(c.as_raw(), 0b00000010);
    }

    #[test]
    fn test_bitand_assign_operator() {
        let mut flags = ExceptionFlags::from_raw(0b00000011);
        flags &= ExceptionFlags::from_raw(0b00000010);

        assert_eq!(flags.as_raw(), 0b00000010);
    }

    #[test]
    fn test_not_operator() {
        let flags = ExceptionFlags::from_raw(0b00001111);
        let inverted = !flags;

        assert_eq!(inverted.as_raw(), 0b11110000);
    }

    // ════════════════════════════════════════════════════════════════════════
    // Debug Format Tests
    // ════════════════════════════════════════════════════════════════════════

    #[test]
    fn test_debug_empty() {
        let flags = ExceptionFlags::empty();
        let debug = format!("{:?}", flags);
        assert_eq!(debug, "ExceptionFlags(empty)");
    }

    #[test]
    fn test_debug_single_flag() {
        let flags = ExceptionFlags::empty().set_normalized();
        let debug = format!("{:?}", flags);
        assert_eq!(debug, "ExceptionFlags(NORMALIZED)");
    }

    #[test]
    fn test_debug_multiple_flags() {
        let flags = ExceptionFlags::empty().set_normalized().set_has_args();
        let debug = format!("{:?}", flags);
        assert_eq!(debug, "ExceptionFlags(NORMALIZED | HAS_ARGS)");
    }

    // ════════════════════════════════════════════════════════════════════════
    // Trait Tests
    // ════════════════════════════════════════════════════════════════════════

    #[test]
    fn test_default() {
        let flags = ExceptionFlags::default();
        assert!(flags.is_empty());
    }

    #[test]
    fn test_clone() {
        let original = ExceptionFlags::empty().set_normalized();
        let cloned = original.clone();
        assert_eq!(original, cloned);
    }

    #[test]
    fn test_copy() {
        let original = ExceptionFlags::empty().set_normalized();
        let copied: ExceptionFlags = original;
        assert_eq!(original, copied);
    }

    #[test]
    fn test_equality() {
        let a = ExceptionFlags::empty().set_normalized();
        let b = ExceptionFlags::empty().set_normalized();
        let c = ExceptionFlags::empty().set_has_args();

        assert_eq!(a, b);
        assert_ne!(a, c);
    }

    #[test]
    fn test_hash() {
        use std::collections::HashSet;

        let mut set = HashSet::new();
        set.insert(ExceptionFlags::empty().set_normalized());
        set.insert(ExceptionFlags::empty().set_has_args());
        set.insert(ExceptionFlags::empty().set_normalized()); // Duplicate

        assert_eq!(set.len(), 2);
    }

    // ════════════════════════════════════════════════════════════════════════
    // Size Tests (Performance Verification)
    // ════════════════════════════════════════════════════════════════════════

    #[test]
    fn test_size_is_one_byte() {
        assert_eq!(std::mem::size_of::<ExceptionFlags>(), 1);
    }

    #[test]
    fn test_alignment_is_one_byte() {
        assert_eq!(std::mem::align_of::<ExceptionFlags>(), 1);
    }

    // ════════════════════════════════════════════════════════════════════════
    // Edge Case Tests
    // ════════════════════════════════════════════════════════════════════════

    #[test]
    fn test_set_same_flag_twice() {
        let flags = ExceptionFlags::empty().set_normalized().set_normalized();

        assert!(flags.is_normalized());
        assert_eq!(flags.count_ones(), 1);
    }

    #[test]
    fn test_clear_unset_flag() {
        let flags = ExceptionFlags::empty().clear_has_traceback();
        assert!(!flags.has_traceback());
        assert!(flags.is_empty());
    }

    #[test]
    fn test_all_flags_are_independent() {
        // Each flag should be independently settable
        let flag_setters: [fn(ExceptionFlags) -> ExceptionFlags; 8] = [
            ExceptionFlags::set_normalized,
            ExceptionFlags::set_has_args,
            ExceptionFlags::set_has_traceback,
            ExceptionFlags::set_has_cause,
            ExceptionFlags::set_has_context,
            ExceptionFlags::set_suppress_context,
            ExceptionFlags::set_flyweight,
            ExceptionFlags::set_reraised,
        ];

        for (i, setter) in flag_setters.iter().enumerate() {
            let flags = setter(ExceptionFlags::empty());
            assert_eq!(flags.count_ones(), 1, "Flag {} should set exactly 1 bit", i);
        }
    }
}
