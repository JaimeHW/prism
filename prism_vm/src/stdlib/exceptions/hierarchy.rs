//! Exception hierarchy support.
//!
//! This module provides efficient lookup tables for Python's exception
//! inheritance hierarchy. The tables enable O(1) subclass checking in
//! JIT-compiled code for common cases.
//!
//! # Performance Design
//!
//! - **Static lookup table**: Pre-computed at compile time
//! - **Bitsets for descendants**: O(1) check if A is subclass of B
//! - **Depth-first numbering**: Range checks for subtree membership

use super::types::ExceptionTypeId;

/// Bitset representing a set of exception types.
///
/// With 64 built-in exception types, we use a single u64.
/// This enables O(1) subclass testing via bit operations.
#[derive(Clone, Copy, PartialEq, Eq, Default)]
pub struct ExceptionTypeSet(u64);

impl ExceptionTypeSet {
    /// Empty set (no exception types).
    #[inline(always)]
    pub const fn empty() -> Self {
        Self(0)
    }

    /// Set containing all built-in exception types.
    #[inline(always)]
    pub const fn all() -> Self {
        // All 64 bits set
        Self(u64::MAX)
    }

    /// Creates a set containing a single exception type.
    #[inline(always)]
    pub const fn singleton(type_id: ExceptionTypeId) -> Self {
        Self(1u64 << type_id.as_u8())
    }

    /// Creates a set from a raw bitmask.
    #[inline(always)]
    pub const fn from_raw(bits: u64) -> Self {
        Self(bits)
    }

    /// Returns the raw bitmask.
    #[inline(always)]
    pub const fn as_raw(self) -> u64 {
        self.0
    }

    /// Returns true if the set is empty.
    #[inline(always)]
    pub const fn is_empty(self) -> bool {
        self.0 == 0
    }

    /// Returns true if the set contains the given exception type.
    #[inline(always)]
    pub const fn contains(self, type_id: ExceptionTypeId) -> bool {
        (self.0 >> type_id.as_u8()) & 1 != 0
    }

    /// Returns the union of two sets.
    #[inline(always)]
    pub const fn union(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }

    /// Returns the intersection of two sets.
    #[inline(always)]
    pub const fn intersection(self, other: Self) -> Self {
        Self(self.0 & other.0)
    }

    /// Returns this set with the given type added.
    #[inline(always)]
    pub const fn with(self, type_id: ExceptionTypeId) -> Self {
        Self(self.0 | (1u64 << type_id.as_u8()))
    }

    /// Returns this set with the given type removed.
    #[inline(always)]
    pub const fn without(self, type_id: ExceptionTypeId) -> Self {
        Self(self.0 & !(1u64 << type_id.as_u8()))
    }

    /// Returns the number of types in the set.
    #[inline(always)]
    pub const fn len(self) -> u32 {
        self.0.count_ones()
    }

    /// Adds a type to this set (mutating).
    #[inline(always)]
    pub fn insert(&mut self, type_id: ExceptionTypeId) {
        self.0 |= 1u64 << type_id.as_u8();
    }

    /// Removes a type from this set (mutating).
    #[inline(always)]
    pub fn remove(&mut self, type_id: ExceptionTypeId) {
        self.0 &= !(1u64 << type_id.as_u8());
    }
}

impl std::fmt::Debug for ExceptionTypeSet {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut list = f.debug_set();
        for id in 0..64u8 {
            if let Some(type_id) = ExceptionTypeId::from_u8(id) {
                if self.contains(type_id) {
                    list.entry(&type_id);
                }
            }
        }
        list.finish()
    }
}

// ============================================================================
// Static Hierarchy Tables
// ============================================================================

/// Returns the set of all direct and indirect subclasses of the given type.
///
/// This is a compile-time computed lookup table that enables O(1)
/// subclass checking for `except BaseType:` clauses.
#[inline]
pub const fn descendants(type_id: ExceptionTypeId) -> ExceptionTypeSet {
    // Pre-computed descendant sets for each exception type
    // These are computed based on the parent() relationships in types.rs
    match type_id {
        // BaseException: All exception types are descendants
        ExceptionTypeId::BaseException => ExceptionTypeSet::all(),

        // Exception: All except SystemExit, KeyboardInterrupt, GeneratorExit
        ExceptionTypeId::Exception => ExceptionTypeSet::from_raw(
            !((1u64 << ExceptionTypeId::BaseException.as_u8())
                | (1u64 << ExceptionTypeId::SystemExit.as_u8())
                | (1u64 << ExceptionTypeId::KeyboardInterrupt.as_u8())
                | (1u64 << ExceptionTypeId::GeneratorExit.as_u8())),
        ),

        // LookupError: IndexError, KeyError
        ExceptionTypeId::LookupError => ExceptionTypeSet::from_raw(
            (1u64 << ExceptionTypeId::LookupError.as_u8())
                | (1u64 << ExceptionTypeId::IndexError.as_u8())
                | (1u64 << ExceptionTypeId::KeyError.as_u8()),
        ),

        // ArithmeticError: OverflowError, ZeroDivisionError, FloatingPointError
        ExceptionTypeId::ArithmeticError => ExceptionTypeSet::from_raw(
            (1u64 << ExceptionTypeId::ArithmeticError.as_u8())
                | (1u64 << ExceptionTypeId::OverflowError.as_u8())
                | (1u64 << ExceptionTypeId::ZeroDivisionError.as_u8())
                | (1u64 << ExceptionTypeId::FloatingPointError.as_u8()),
        ),

        // OSError: All OS-related errors (IDs 24-39)
        ExceptionTypeId::OSError => ExceptionTypeSet::from_raw(
            (1u64 << 24)
                | (1u64 << 25)
                | (1u64 << 26)
                | (1u64 << 27)
                | (1u64 << 28)
                | (1u64 << 29)
                | (1u64 << 30)
                | (1u64 << 31)
                | (1u64 << 32)
                | (1u64 << 33)
                | (1u64 << 34)
                | (1u64 << 35)
                | (1u64 << 36)
                | (1u64 << 37)
                | (1u64 << 38)
                | (1u64 << 39),
        ),

        // ConnectionError: ConnectionRefused, ConnectionReset, ConnectionAborted, BrokenPipe
        ExceptionTypeId::ConnectionError => ExceptionTypeSet::from_raw(
            (1u64 << ExceptionTypeId::ConnectionError.as_u8())
                | (1u64 << ExceptionTypeId::ConnectionRefusedError.as_u8())
                | (1u64 << ExceptionTypeId::ConnectionResetError.as_u8())
                | (1u64 << ExceptionTypeId::ConnectionAbortedError.as_u8())
                | (1u64 << ExceptionTypeId::BrokenPipeError.as_u8()),
        ),

        // RuntimeError: RecursionError, NotImplementedError
        ExceptionTypeId::RuntimeError => ExceptionTypeSet::from_raw(
            (1u64 << ExceptionTypeId::RuntimeError.as_u8())
                | (1u64 << ExceptionTypeId::RecursionError.as_u8())
                | (1u64 << ExceptionTypeId::NotImplementedError.as_u8()),
        ),

        // SyntaxError: IndentationError, TabError
        ExceptionTypeId::SyntaxError => ExceptionTypeSet::from_raw(
            (1u64 << ExceptionTypeId::SyntaxError.as_u8())
                | (1u64 << ExceptionTypeId::IndentationError.as_u8())
                | (1u64 << ExceptionTypeId::TabError.as_u8()),
        ),

        // IndentationError: TabError
        ExceptionTypeId::IndentationError => ExceptionTypeSet::from_raw(
            (1u64 << ExceptionTypeId::IndentationError.as_u8())
                | (1u64 << ExceptionTypeId::TabError.as_u8()),
        ),

        // ValueError: UnicodeError and subtypes
        ExceptionTypeId::ValueError => ExceptionTypeSet::from_raw(
            (1u64 << ExceptionTypeId::ValueError.as_u8())
                | (1u64 << ExceptionTypeId::UnicodeError.as_u8())
                | (1u64 << ExceptionTypeId::UnicodeDecodeError.as_u8())
                | (1u64 << ExceptionTypeId::UnicodeEncodeError.as_u8())
                | (1u64 << ExceptionTypeId::UnicodeTranslateError.as_u8()),
        ),

        // UnicodeError: Decode, Encode, Translate variants
        ExceptionTypeId::UnicodeError => ExceptionTypeSet::from_raw(
            (1u64 << ExceptionTypeId::UnicodeError.as_u8())
                | (1u64 << ExceptionTypeId::UnicodeDecodeError.as_u8())
                | (1u64 << ExceptionTypeId::UnicodeEncodeError.as_u8())
                | (1u64 << ExceptionTypeId::UnicodeTranslateError.as_u8()),
        ),

        // NameError: UnboundLocalError
        ExceptionTypeId::NameError => ExceptionTypeSet::from_raw(
            (1u64 << ExceptionTypeId::NameError.as_u8())
                | (1u64 << ExceptionTypeId::UnboundLocalError.as_u8()),
        ),

        // ImportError: ModuleNotFoundError
        ExceptionTypeId::ImportError => ExceptionTypeSet::from_raw(
            (1u64 << ExceptionTypeId::ImportError.as_u8())
                | (1u64 << ExceptionTypeId::ModuleNotFoundError.as_u8()),
        ),

        // Warning: All warning subtypes
        ExceptionTypeId::Warning => ExceptionTypeSet::from_raw(
            (1u64 << ExceptionTypeId::Warning.as_u8())
                | (1u64 << ExceptionTypeId::DeprecationWarning.as_u8())
                | (1u64 << ExceptionTypeId::PendingDeprecationWarning.as_u8())
                | (1u64 << ExceptionTypeId::RuntimeWarning.as_u8())
                | (1u64 << ExceptionTypeId::SyntaxWarning.as_u8())
                | (1u64 << ExceptionTypeId::UserWarning.as_u8()),
        ),

        // Leaf types (no descendants) - return singleton
        _ => ExceptionTypeSet::singleton(type_id),
    }
}

/// Fast check if `exception_type` is a subclass of `base_type`.
///
/// This uses the pre-computed descendant sets for O(1) lookup.
#[inline(always)]
pub const fn is_subclass(exception_type: ExceptionTypeId, base_type: ExceptionTypeId) -> bool {
    descendants(base_type).contains(exception_type)
}

/// Returns the common ancestor of two exception types.
///
/// This is useful for determining the catch type for multiple exceptions.
pub fn common_ancestor(a: ExceptionTypeId, b: ExceptionTypeId) -> ExceptionTypeId {
    if a == b {
        return a;
    }

    // Walk up both hierarchies to find common ancestor
    let mut ancestors_a = ExceptionTypeSet::empty();
    let mut current = a;
    loop {
        ancestors_a.insert(current);
        match current.parent() {
            Some(parent) => current = parent,
            None => break,
        }
    }

    // Walk up b's hierarchy until we find a common ancestor
    current = b;
    loop {
        if ancestors_a.contains(current) {
            return current;
        }
        match current.parent() {
            Some(parent) => current = parent,
            None => break,
        }
    }

    // Should never happen for valid types (all trace to BaseException)
    ExceptionTypeId::BaseException
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ════════════════════════════════════════════════════════════════════════
    // ExceptionTypeSet Tests
    // ════════════════════════════════════════════════════════════════════════

    #[test]
    fn test_empty_set() {
        let set = ExceptionTypeSet::empty();
        assert!(set.is_empty());
        assert_eq!(set.len(), 0);
    }

    #[test]
    fn test_all_set() {
        let set = ExceptionTypeSet::all();
        assert!(!set.is_empty());
        assert_eq!(set.len(), 64);
    }

    #[test]
    fn test_singleton() {
        let set = ExceptionTypeSet::singleton(ExceptionTypeId::TypeError);
        assert!(set.contains(ExceptionTypeId::TypeError));
        assert!(!set.contains(ExceptionTypeId::ValueError));
        assert_eq!(set.len(), 1);
    }

    #[test]
    fn test_contains() {
        let set = ExceptionTypeSet::singleton(ExceptionTypeId::TypeError)
            .with(ExceptionTypeId::ValueError);

        assert!(set.contains(ExceptionTypeId::TypeError));
        assert!(set.contains(ExceptionTypeId::ValueError));
        assert!(!set.contains(ExceptionTypeId::KeyError));
    }

    #[test]
    fn test_union() {
        let a = ExceptionTypeSet::singleton(ExceptionTypeId::TypeError);
        let b = ExceptionTypeSet::singleton(ExceptionTypeId::ValueError);
        let c = a.union(b);

        assert!(c.contains(ExceptionTypeId::TypeError));
        assert!(c.contains(ExceptionTypeId::ValueError));
        assert_eq!(c.len(), 2);
    }

    #[test]
    fn test_intersection() {
        let a = ExceptionTypeSet::singleton(ExceptionTypeId::TypeError)
            .with(ExceptionTypeId::ValueError);
        let b = ExceptionTypeSet::singleton(ExceptionTypeId::ValueError)
            .with(ExceptionTypeId::KeyError);
        let c = a.intersection(b);

        assert!(!c.contains(ExceptionTypeId::TypeError));
        assert!(c.contains(ExceptionTypeId::ValueError));
        assert!(!c.contains(ExceptionTypeId::KeyError));
        assert_eq!(c.len(), 1);
    }

    #[test]
    fn test_with() {
        let set = ExceptionTypeSet::empty()
            .with(ExceptionTypeId::TypeError)
            .with(ExceptionTypeId::ValueError);

        assert!(set.contains(ExceptionTypeId::TypeError));
        assert!(set.contains(ExceptionTypeId::ValueError));
    }

    #[test]
    fn test_without() {
        let set = ExceptionTypeSet::singleton(ExceptionTypeId::TypeError)
            .with(ExceptionTypeId::ValueError)
            .without(ExceptionTypeId::TypeError);

        assert!(!set.contains(ExceptionTypeId::TypeError));
        assert!(set.contains(ExceptionTypeId::ValueError));
    }

    #[test]
    fn test_insert_remove() {
        let mut set = ExceptionTypeSet::empty();
        set.insert(ExceptionTypeId::TypeError);
        assert!(set.contains(ExceptionTypeId::TypeError));

        set.remove(ExceptionTypeId::TypeError);
        assert!(!set.contains(ExceptionTypeId::TypeError));
    }

    #[test]
    fn test_set_size() {
        // u64 is 8 bytes
        assert_eq!(std::mem::size_of::<ExceptionTypeSet>(), 8);
    }

    // ════════════════════════════════════════════════════════════════════════
    // Descendants Tests
    // ════════════════════════════════════════════════════════════════════════

    #[test]
    fn test_base_exception_descendants_contain_all() {
        let desc = descendants(ExceptionTypeId::BaseException);

        // All exception types should be descendants of BaseException
        assert!(desc.contains(ExceptionTypeId::Exception));
        assert!(desc.contains(ExceptionTypeId::TypeError));
        assert!(desc.contains(ExceptionTypeId::SystemExit));
        assert!(desc.contains(ExceptionTypeId::KeyboardInterrupt));
    }

    #[test]
    fn test_exception_descendants() {
        let desc = descendants(ExceptionTypeId::Exception);

        // Should contain Exception subclasses
        assert!(desc.contains(ExceptionTypeId::TypeError));
        assert!(desc.contains(ExceptionTypeId::ValueError));
        assert!(desc.contains(ExceptionTypeId::KeyError));

        // Should NOT contain non-Exception types
        assert!(!desc.contains(ExceptionTypeId::BaseException));
        assert!(!desc.contains(ExceptionTypeId::SystemExit));
        assert!(!desc.contains(ExceptionTypeId::KeyboardInterrupt));
        assert!(!desc.contains(ExceptionTypeId::GeneratorExit));
    }

    #[test]
    fn test_lookup_error_descendants() {
        let desc = descendants(ExceptionTypeId::LookupError);

        assert!(desc.contains(ExceptionTypeId::LookupError));
        assert!(desc.contains(ExceptionTypeId::IndexError));
        assert!(desc.contains(ExceptionTypeId::KeyError));

        assert!(!desc.contains(ExceptionTypeId::TypeError));
        assert!(!desc.contains(ExceptionTypeId::ValueError));
    }

    #[test]
    fn test_arithmetic_error_descendants() {
        let desc = descendants(ExceptionTypeId::ArithmeticError);

        assert!(desc.contains(ExceptionTypeId::ArithmeticError));
        assert!(desc.contains(ExceptionTypeId::OverflowError));
        assert!(desc.contains(ExceptionTypeId::ZeroDivisionError));
        assert!(desc.contains(ExceptionTypeId::FloatingPointError));

        assert!(!desc.contains(ExceptionTypeId::TypeError));
    }

    #[test]
    fn test_os_error_descendants() {
        let desc = descendants(ExceptionTypeId::OSError);

        assert!(desc.contains(ExceptionTypeId::OSError));
        assert!(desc.contains(ExceptionTypeId::FileNotFoundError));
        assert!(desc.contains(ExceptionTypeId::PermissionError));
        assert!(desc.contains(ExceptionTypeId::ConnectionError));
        assert!(desc.contains(ExceptionTypeId::ConnectionRefusedError));

        assert!(!desc.contains(ExceptionTypeId::TypeError));
    }

    #[test]
    fn test_syntax_error_descendants() {
        let desc = descendants(ExceptionTypeId::SyntaxError);

        assert!(desc.contains(ExceptionTypeId::SyntaxError));
        assert!(desc.contains(ExceptionTypeId::IndentationError));
        assert!(desc.contains(ExceptionTypeId::TabError));

        assert!(!desc.contains(ExceptionTypeId::TypeError));
    }

    #[test]
    fn test_value_error_descendants() {
        let desc = descendants(ExceptionTypeId::ValueError);

        assert!(desc.contains(ExceptionTypeId::ValueError));
        assert!(desc.contains(ExceptionTypeId::UnicodeError));
        assert!(desc.contains(ExceptionTypeId::UnicodeDecodeError));
        assert!(desc.contains(ExceptionTypeId::UnicodeEncodeError));

        assert!(!desc.contains(ExceptionTypeId::TypeError));
    }

    #[test]
    fn test_warning_descendants() {
        let desc = descendants(ExceptionTypeId::Warning);

        assert!(desc.contains(ExceptionTypeId::Warning));
        assert!(desc.contains(ExceptionTypeId::DeprecationWarning));
        assert!(desc.contains(ExceptionTypeId::RuntimeWarning));
        assert!(desc.contains(ExceptionTypeId::SyntaxWarning));

        assert!(!desc.contains(ExceptionTypeId::TypeError));
    }

    #[test]
    fn test_leaf_type_descendants() {
        // Leaf types should only contain themselves
        let desc = descendants(ExceptionTypeId::TypeError);
        assert!(desc.contains(ExceptionTypeId::TypeError));
        assert_eq!(desc.len(), 1);

        let desc = descendants(ExceptionTypeId::KeyError);
        assert!(desc.contains(ExceptionTypeId::KeyError));
        assert_eq!(desc.len(), 1);
    }

    // ════════════════════════════════════════════════════════════════════════
    // is_subclass Tests
    // ════════════════════════════════════════════════════════════════════════

    #[test]
    fn test_is_subclass_self() {
        assert!(is_subclass(
            ExceptionTypeId::TypeError,
            ExceptionTypeId::TypeError
        ));
        assert!(is_subclass(
            ExceptionTypeId::Exception,
            ExceptionTypeId::Exception
        ));
    }

    #[test]
    fn test_is_subclass_direct_parent() {
        assert!(is_subclass(
            ExceptionTypeId::TypeError,
            ExceptionTypeId::Exception
        ));
        assert!(is_subclass(
            ExceptionTypeId::IndexError,
            ExceptionTypeId::LookupError
        ));
    }

    #[test]
    fn test_is_subclass_indirect_parent() {
        assert!(is_subclass(
            ExceptionTypeId::TypeError,
            ExceptionTypeId::BaseException
        ));
        assert!(is_subclass(
            ExceptionTypeId::TabError,
            ExceptionTypeId::Exception
        ));
        assert!(is_subclass(
            ExceptionTypeId::ConnectionRefusedError,
            ExceptionTypeId::Exception
        ));
    }

    #[test]
    fn test_is_not_subclass() {
        assert!(!is_subclass(
            ExceptionTypeId::TypeError,
            ExceptionTypeId::ValueError
        ));
        assert!(!is_subclass(
            ExceptionTypeId::Exception,
            ExceptionTypeId::TypeError
        ));
        assert!(!is_subclass(
            ExceptionTypeId::SystemExit,
            ExceptionTypeId::Exception
        ));
    }

    // ════════════════════════════════════════════════════════════════════════
    // common_ancestor Tests
    // ════════════════════════════════════════════════════════════════════════

    #[test]
    fn test_common_ancestor_same_type() {
        let ancestor = common_ancestor(ExceptionTypeId::TypeError, ExceptionTypeId::TypeError);
        assert_eq!(ancestor, ExceptionTypeId::TypeError);
    }

    #[test]
    fn test_common_ancestor_parent_child() {
        let ancestor = common_ancestor(ExceptionTypeId::TypeError, ExceptionTypeId::Exception);
        assert_eq!(ancestor, ExceptionTypeId::Exception);

        let ancestor = common_ancestor(ExceptionTypeId::Exception, ExceptionTypeId::TypeError);
        assert_eq!(ancestor, ExceptionTypeId::Exception);
    }

    #[test]
    fn test_common_ancestor_siblings() {
        let ancestor = common_ancestor(ExceptionTypeId::TypeError, ExceptionTypeId::ValueError);
        assert_eq!(ancestor, ExceptionTypeId::Exception);
    }

    #[test]
    fn test_common_ancestor_cousins() {
        let ancestor = common_ancestor(ExceptionTypeId::IndexError, ExceptionTypeId::KeyError);
        assert_eq!(ancestor, ExceptionTypeId::LookupError);
    }

    #[test]
    fn test_common_ancestor_distant() {
        let ancestor = common_ancestor(
            ExceptionTypeId::TabError,
            ExceptionTypeId::ConnectionRefusedError,
        );
        assert_eq!(ancestor, ExceptionTypeId::Exception);
    }

    #[test]
    fn test_common_ancestor_system_exit() {
        // SystemExit and TypeError only share BaseException
        let ancestor = common_ancestor(ExceptionTypeId::SystemExit, ExceptionTypeId::TypeError);
        assert_eq!(ancestor, ExceptionTypeId::BaseException);
    }

    // ════════════════════════════════════════════════════════════════════════
    // Debug Format Tests
    // ════════════════════════════════════════════════════════════════════════

    #[test]
    fn test_debug_empty_set() {
        let set = ExceptionTypeSet::empty();
        let debug = format!("{:?}", set);
        assert_eq!(debug, "{}");
    }

    #[test]
    fn test_debug_singleton_set() {
        let set = ExceptionTypeSet::singleton(ExceptionTypeId::TypeError);
        let debug = format!("{:?}", set);
        assert!(debug.contains("TypeError"));
    }
}
