//! Exception info stack for CPython 3.11+ semantics.
//!
//! This module provides `ExcInfoStack`, which maintains a stack of exception
//! contexts for proper handling of nested try/except/finally blocks.
//!
//! # CPython 3.11+ Semantics
//!
//! In Python 3.11+, exception contexts are preserved on a stack:
//! - `PushExcInfo`: Saves current exception state before entering handler
//! - `PopExcInfo`: Restores previous exception state after handler exits
//!
//! This enables proper exception chaining and `sys.exc_info()` behavior.
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────┐
//! │                      ExcInfoStack                                │
//! ├─────────────────────────────────────────────────────────────────┤
//! │                                                                  │
//! │  ┌─────────────────┐                                            │
//! │  │  ExcInfoEntry   │ ◄── Top (most recent)                      │
//! │  │  ├─ type_id     │                                            │
//! │  │  ├─ value       │                                            │
//! │  │  └─ tb_ref      │                                            │
//! │  ├─────────────────┤                                            │
//! │  │  ExcInfoEntry   │                                            │
//! │  ├─────────────────┤                                            │
//! │  │       ...       │                                            │
//! │  └─────────────────┘ ◄── Bottom (oldest)                        │
//! │                                                                  │
//! │  Inline capacity: 4 entries (typical nesting depth)             │
//! │  Max depth: 255 (enforced)                                      │
//! └─────────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Performance
//!
//! | Operation | Complexity | Notes |
//! |-----------|------------|-------|
//! | push | O(1) amortized | Inline for depth ≤ 4 |
//! | pop | O(1) | Direct stack pop |
//! | peek | O(1) | No allocation |
//! | clear | O(N) | N = current depth |

use prism_core::Value;
use smallvec::SmallVec;
use std::fmt;

// =============================================================================
// Constants
// =============================================================================

/// Inline capacity for exception info stack.
/// Most programs have shallow exception nesting (try { try { } } is depth 2).
const INLINE_CAPACITY: usize = 4;

/// Maximum exception info stack depth (prevent unbounded growth).
const MAX_DEPTH: usize = 255;

// =============================================================================
// ExcInfoEntry
// =============================================================================

/// A single exception info entry preserving exception context.
///
/// # Memory Layout (48 bytes on 64-bit)
///
/// ```text
/// ┌────────────────────────────────────────────┐
/// │ type_id: u16                    (2 bytes)  │
/// │ flags: EntryFlags               (1 byte)   │
/// │ _pad: [u8; 5]                   (5 bytes)  │
/// │ value: Option<Value>            (16 bytes) │
/// │ traceback_id: u32               (4 bytes)  │
/// │ frame_id: u32                   (4 bytes)  │
/// │ pc: u32                         (4 bytes)  │
/// │ _reserved: [u8; 12]             (12 bytes) │
/// └────────────────────────────────────────────┘
/// ```
#[derive(Clone)]
#[repr(C)]
pub struct ExcInfoEntry {
    /// Exception type ID (ExceptionTypeId discriminant).
    type_id: u16,

    /// Entry flags.
    flags: EntryFlags,

    /// Padding for alignment.
    _pad: [u8; 5],

    /// Exception value (cloned for preservation).
    value: Option<Value>,

    /// Traceback reference ID (index into traceback table).
    traceback_id: u32,

    /// Frame ID where exception was raised.
    frame_id: u32,

    /// Program counter where exception was raised.
    pc: u32,

    /// Reserved for future use.
    _reserved: [u8; 12],
}

/// Entry flags for ExcInfoEntry.
#[derive(Clone, Copy, Default, PartialEq, Eq)]
#[repr(transparent)]
pub struct EntryFlags(u8);

impl EntryFlags {
    /// No flags set.
    pub const EMPTY: Self = Self(0);

    /// Exception was explicitly raised (vs. implicit from runtime).
    pub const EXPLICIT: u8 = 1 << 0;

    /// Exception has a chained cause (__cause__ is set).
    pub const HAS_CAUSE: u8 = 1 << 1;

    /// Context was suppressed (__suppress_context__ = True).
    pub const SUPPRESS_CONTEXT: u8 = 1 << 2;

    /// Exception is currently being handled.
    pub const HANDLING: u8 = 1 << 3;

    /// Exception came from a `raise ... from ...` statement.
    pub const FROM_RAISE_FROM: u8 = 1 << 4;

    /// Creates flags from raw value.
    #[inline]
    pub const fn from_raw(value: u8) -> Self {
        Self(value)
    }

    /// Returns the raw value.
    #[inline]
    pub const fn as_raw(self) -> u8 {
        self.0
    }

    /// Sets a flag.
    #[inline]
    pub fn set(&mut self, flag: u8) {
        self.0 |= flag;
    }

    /// Clears a flag.
    #[inline]
    pub fn clear(&mut self, flag: u8) {
        self.0 &= !flag;
    }

    /// Checks if a flag is set.
    #[inline]
    pub const fn has(self, flag: u8) -> bool {
        (self.0 & flag) != 0
    }

    /// Returns true if exception was explicitly raised.
    #[inline]
    pub const fn is_explicit(self) -> bool {
        self.has(Self::EXPLICIT)
    }

    /// Returns true if exception has a chained cause.
    #[inline]
    pub const fn has_cause(self) -> bool {
        self.has(Self::HAS_CAUSE)
    }

    /// Returns true if context is suppressed.
    #[inline]
    pub const fn is_context_suppressed(self) -> bool {
        self.has(Self::SUPPRESS_CONTEXT)
    }
}

impl fmt::Debug for EntryFlags {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut flags = Vec::new();
        if self.has(Self::EXPLICIT) {
            flags.push("EXPLICIT");
        }
        if self.has(Self::HAS_CAUSE) {
            flags.push("HAS_CAUSE");
        }
        if self.has(Self::SUPPRESS_CONTEXT) {
            flags.push("SUPPRESS_CONTEXT");
        }
        if self.has(Self::HANDLING) {
            flags.push("HANDLING");
        }
        if self.has(Self::FROM_RAISE_FROM) {
            flags.push("FROM_RAISE_FROM");
        }
        write!(f, "EntryFlags({})", flags.join("|"))
    }
}

impl ExcInfoEntry {
    /// Creates a new exception info entry.
    #[inline]
    pub fn new(type_id: u16, value: Option<Value>) -> Self {
        Self {
            type_id,
            flags: EntryFlags::EMPTY,
            _pad: [0; 5],
            value,
            traceback_id: 0,
            frame_id: 0,
            pc: 0,
            _reserved: [0; 12],
        }
    }

    /// Creates an entry with full context.
    #[inline]
    pub fn with_context(
        type_id: u16,
        value: Option<Value>,
        traceback_id: u32,
        frame_id: u32,
        pc: u32,
    ) -> Self {
        Self {
            type_id,
            flags: EntryFlags::EMPTY,
            _pad: [0; 5],
            value,
            traceback_id,
            frame_id,
            pc,
            _reserved: [0; 12],
        }
    }

    /// Creates an empty/sentinel entry.
    #[inline]
    pub const fn empty() -> Self {
        Self {
            type_id: 0,
            flags: EntryFlags::EMPTY,
            _pad: [0; 5],
            value: None,
            traceback_id: 0,
            frame_id: 0,
            pc: 0,
            _reserved: [0; 12],
        }
    }

    /// Returns true if this entry represents an active exception.
    #[inline]
    pub fn is_active(&self) -> bool {
        self.type_id != 0 || self.value.is_some()
    }

    /// Returns the exception type ID.
    #[inline]
    pub const fn type_id(&self) -> u16 {
        self.type_id
    }

    /// Returns a reference to the exception value.
    #[inline]
    pub fn value(&self) -> Option<&Value> {
        self.value.as_ref()
    }

    /// Returns a cloned exception value.
    #[inline]
    pub fn value_cloned(&self) -> Option<Value> {
        self.value.clone()
    }

    /// Sets the exception value.
    #[inline]
    pub fn set_value(&mut self, value: Option<Value>) {
        self.value = value;
    }

    /// Returns the traceback ID.
    #[inline]
    pub const fn traceback_id(&self) -> u32 {
        self.traceback_id
    }

    /// Sets the traceback ID.
    #[inline]
    pub fn set_traceback_id(&mut self, id: u32) {
        self.traceback_id = id;
    }

    /// Returns the frame ID where exception was raised.
    #[inline]
    pub const fn frame_id(&self) -> u32 {
        self.frame_id
    }

    /// Returns the program counter where exception was raised.
    #[inline]
    pub const fn pc(&self) -> u32 {
        self.pc
    }

    /// Returns the entry flags.
    #[inline]
    pub const fn flags(&self) -> EntryFlags {
        self.flags
    }

    /// Returns a mutable reference to the entry flags.
    #[inline]
    pub fn flags_mut(&mut self) -> &mut EntryFlags {
        &mut self.flags
    }

    /// Marks this entry as having a chained cause.
    #[inline]
    pub fn set_has_cause(&mut self) {
        self.flags.set(EntryFlags::HAS_CAUSE);
    }

    /// Marks this entry as suppressing context.
    #[inline]
    pub fn set_suppress_context(&mut self) {
        self.flags.set(EntryFlags::SUPPRESS_CONTEXT);
    }

    /// Marks this entry as from a `raise ... from ...` statement.
    #[inline]
    pub fn set_from_raise_from(&mut self) {
        self.flags.set(EntryFlags::FROM_RAISE_FROM);
        self.flags.set(EntryFlags::HAS_CAUSE);
        self.flags.set(EntryFlags::SUPPRESS_CONTEXT);
    }
}

impl Default for ExcInfoEntry {
    fn default() -> Self {
        Self::empty()
    }
}

impl fmt::Debug for ExcInfoEntry {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ExcInfoEntry")
            .field("type_id", &self.type_id)
            .field("flags", &self.flags)
            .field("has_value", &self.value.is_some())
            .field("traceback_id", &self.traceback_id)
            .field("frame_id", &self.frame_id)
            .field("pc", &self.pc)
            .finish()
    }
}

// =============================================================================
// ExcInfoStack
// =============================================================================

/// Stack of exception info entries for nested exception handling.
///
/// This implements CPython 3.11+ semantics where exception contexts are
/// preserved on a stack during nested try/except/finally blocks.
///
/// # Usage
///
/// ```ignore
/// let mut stack = ExcInfoStack::new();
///
/// // Entering an except handler - preserve current exception
/// stack.push(ExcInfoEntry::new(exc_type_id, Some(exc_value)));
///
/// // Handle exception...
///
/// // Exiting handler - restore previous exception state
/// if let Some(entry) = stack.pop() {
///     // Restore previous exception context
/// }
/// ```
#[derive(Clone)]
pub struct ExcInfoStack {
    /// Stack entries (inline for common case).
    entries: SmallVec<[ExcInfoEntry; INLINE_CAPACITY]>,

    /// Statistics for monitoring.
    stats: ExcInfoStackStats,
}

/// Statistics for ExcInfoStack operations.
#[derive(Clone, Copy, Default)]
pub struct ExcInfoStackStats {
    /// Total push operations.
    pub pushes: u32,

    /// Total pop operations.
    pub pops: u32,

    /// Peak stack depth reached.
    pub peak_depth: u32,

    /// Number of overflow rejections.
    pub overflows: u32,
}

impl ExcInfoStack {
    /// Creates a new empty exception info stack.
    #[inline]
    pub fn new() -> Self {
        Self {
            entries: SmallVec::new(),
            stats: ExcInfoStackStats::default(),
        }
    }

    /// Creates a new stack with pre-allocated capacity.
    #[inline]
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            entries: SmallVec::with_capacity(capacity),
            stats: ExcInfoStackStats::default(),
        }
    }

    /// Returns true if the stack is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Returns the current stack depth.
    #[inline]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Returns the maximum allowed depth.
    #[inline]
    pub const fn max_depth() -> usize {
        MAX_DEPTH
    }

    /// Pushes an exception info entry onto the stack.
    ///
    /// Returns `true` if successful, `false` if stack overflow.
    #[inline]
    pub fn push(&mut self, entry: ExcInfoEntry) -> bool {
        if self.entries.len() >= MAX_DEPTH {
            self.stats.overflows += 1;
            return false;
        }

        self.entries.push(entry);
        self.stats.pushes += 1;

        let depth = self.entries.len() as u32;
        if depth > self.stats.peak_depth {
            self.stats.peak_depth = depth;
        }

        true
    }

    /// Pops the top exception info entry from the stack.
    ///
    /// Returns `None` if the stack is empty.
    #[inline]
    pub fn pop(&mut self) -> Option<ExcInfoEntry> {
        let entry = self.entries.pop();
        if entry.is_some() {
            self.stats.pops += 1;
        }
        entry
    }

    /// Returns a reference to the top entry without removing it.
    #[inline]
    pub fn peek(&self) -> Option<&ExcInfoEntry> {
        self.entries.last()
    }

    /// Returns a mutable reference to the top entry.
    #[inline]
    pub fn peek_mut(&mut self) -> Option<&mut ExcInfoEntry> {
        self.entries.last_mut()
    }

    /// Returns a reference to the entry at the given index (0 = bottom).
    #[inline]
    pub fn get(&self, index: usize) -> Option<&ExcInfoEntry> {
        self.entries.get(index)
    }

    /// Clears all entries from the stack.
    #[inline]
    pub fn clear(&mut self) {
        self.entries.clear();
    }

    /// Returns an iterator over entries from top to bottom.
    #[inline]
    pub fn iter(&self) -> impl Iterator<Item = &ExcInfoEntry> + DoubleEndedIterator {
        self.entries.iter().rev()
    }

    /// Returns an iterator over entries from bottom to top.
    #[inline]
    pub fn iter_bottom_up(&self) -> impl Iterator<Item = &ExcInfoEntry> {
        self.entries.iter()
    }

    /// Returns the stack statistics.
    #[inline]
    pub const fn stats(&self) -> &ExcInfoStackStats {
        &self.stats
    }

    /// Resets the statistics.
    #[inline]
    pub fn reset_stats(&mut self) {
        self.stats = ExcInfoStackStats::default();
    }

    /// Returns the current exception info (top of stack).
    ///
    /// This implements `sys.exc_info()` semantics.
    #[inline]
    pub fn current_exc_info(&self) -> (Option<u16>, Option<Value>, Option<u32>) {
        match self.peek() {
            Some(entry) => (
                Some(entry.type_id),
                entry.value_cloned(),
                if entry.traceback_id != 0 {
                    Some(entry.traceback_id)
                } else {
                    None
                },
            ),
            None => (None, None, None),
        }
    }

    /// Finds the first active exception entry from the top.
    #[inline]
    pub fn find_active(&self) -> Option<&ExcInfoEntry> {
        self.entries.iter().rev().find(|e| e.is_active())
    }

    /// Truncates the stack to the given depth.
    #[inline]
    pub fn truncate(&mut self, depth: usize) {
        self.entries.truncate(depth);
    }
}

impl Default for ExcInfoStack {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Debug for ExcInfoStack {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ExcInfoStack")
            .field("depth", &self.entries.len())
            .field("entries", &self.entries)
            .field("stats", &self.stats)
            .finish()
    }
}

impl fmt::Debug for ExcInfoStackStats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ExcInfoStackStats")
            .field("pushes", &self.pushes)
            .field("pops", &self.pops)
            .field("peak_depth", &self.peak_depth)
            .field("overflows", &self.overflows)
            .finish()
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // EntryFlags Tests
    // =========================================================================

    #[test]
    fn test_entry_flags_empty() {
        let flags = EntryFlags::EMPTY;
        assert_eq!(flags.as_raw(), 0);
        assert!(!flags.is_explicit());
        assert!(!flags.has_cause());
        assert!(!flags.is_context_suppressed());
    }

    #[test]
    fn test_entry_flags_set_and_check() {
        let mut flags = EntryFlags::EMPTY;

        flags.set(EntryFlags::EXPLICIT);
        assert!(flags.is_explicit());
        assert!(!flags.has_cause());

        flags.set(EntryFlags::HAS_CAUSE);
        assert!(flags.is_explicit());
        assert!(flags.has_cause());
    }

    #[test]
    fn test_entry_flags_clear() {
        let mut flags = EntryFlags::from_raw(0xFF);

        flags.clear(EntryFlags::EXPLICIT);
        assert!(!flags.is_explicit());
        assert!(flags.has_cause()); // Other flags still set
    }

    #[test]
    fn test_entry_flags_from_raw() {
        let flags = EntryFlags::from_raw(EntryFlags::EXPLICIT | EntryFlags::HAS_CAUSE);
        assert!(flags.is_explicit());
        assert!(flags.has_cause());
        assert!(!flags.is_context_suppressed());
    }

    #[test]
    fn test_entry_flags_debug() {
        let mut flags = EntryFlags::EMPTY;
        flags.set(EntryFlags::EXPLICIT);
        flags.set(EntryFlags::HAS_CAUSE);

        let debug = format!("{:?}", flags);
        assert!(debug.contains("EXPLICIT"));
        assert!(debug.contains("HAS_CAUSE"));
    }

    // =========================================================================
    // ExcInfoEntry Tests
    // =========================================================================

    #[test]
    fn test_exc_info_entry_new() {
        let entry = ExcInfoEntry::new(24, None); // 24 = TypeError
        assert_eq!(entry.type_id(), 24);
        assert!(entry.value().is_none());
        assert_eq!(entry.traceback_id(), 0);
        assert_eq!(entry.frame_id(), 0);
        assert_eq!(entry.pc(), 0);
    }

    #[test]
    fn test_exc_info_entry_with_value() {
        let value = Value::int(42).unwrap();
        let entry = ExcInfoEntry::new(5, Some(value.clone())); // 5 = StopIteration

        assert_eq!(entry.type_id(), 5);
        assert!(entry.value().is_some());
        assert_eq!(entry.value_cloned().unwrap().as_int(), Some(42));
    }

    #[test]
    fn test_exc_info_entry_with_context() {
        let value = Value::none();
        let entry = ExcInfoEntry::with_context(4, Some(value), 100, 1, 50);

        assert_eq!(entry.type_id(), 4);
        assert_eq!(entry.traceback_id(), 100);
        assert_eq!(entry.frame_id(), 1);
        assert_eq!(entry.pc(), 50);
    }

    #[test]
    fn test_exc_info_entry_empty() {
        let entry = ExcInfoEntry::empty();
        assert_eq!(entry.type_id(), 0);
        assert!(entry.value().is_none());
        assert!(!entry.is_active());
    }

    #[test]
    fn test_exc_info_entry_is_active() {
        let empty = ExcInfoEntry::empty();
        assert!(!empty.is_active());

        let with_type = ExcInfoEntry::new(24, None);
        assert!(with_type.is_active());

        let with_value = ExcInfoEntry::new(0, Some(Value::none()));
        assert!(with_value.is_active());
    }

    #[test]
    fn test_exc_info_entry_set_value() {
        let mut entry = ExcInfoEntry::new(24, None);
        assert!(entry.value().is_none());

        entry.set_value(Some(Value::int(100).unwrap()));
        assert!(entry.value().is_some());
    }

    #[test]
    fn test_exc_info_entry_set_traceback_id() {
        let mut entry = ExcInfoEntry::new(24, None);
        assert_eq!(entry.traceback_id(), 0);

        entry.set_traceback_id(500);
        assert_eq!(entry.traceback_id(), 500);
    }

    #[test]
    fn test_exc_info_entry_set_from_raise_from() {
        let mut entry = ExcInfoEntry::new(24, None);
        entry.set_from_raise_from();

        assert!(entry.flags().has(EntryFlags::FROM_RAISE_FROM));
        assert!(entry.flags().has_cause());
        assert!(entry.flags().is_context_suppressed());
    }

    #[test]
    fn test_exc_info_entry_flags_mut() {
        let mut entry = ExcInfoEntry::new(24, None);
        entry.flags_mut().set(EntryFlags::HANDLING);

        assert!(entry.flags().has(EntryFlags::HANDLING));
    }

    #[test]
    fn test_exc_info_entry_default() {
        let entry = ExcInfoEntry::default();
        assert!(!entry.is_active());
        assert_eq!(entry.type_id(), 0);
    }

    #[test]
    fn test_exc_info_entry_debug() {
        let entry = ExcInfoEntry::new(24, Some(Value::none()));
        let debug = format!("{:?}", entry);

        assert!(debug.contains("ExcInfoEntry"));
        assert!(debug.contains("type_id: 24"));
        assert!(debug.contains("has_value: true"));
    }

    #[test]
    fn test_exc_info_entry_clone() {
        let entry = ExcInfoEntry::with_context(24, Some(Value::int(42).unwrap()), 100, 1, 50);
        let cloned = entry.clone();

        assert_eq!(cloned.type_id(), 24);
        assert_eq!(cloned.traceback_id(), 100);
        assert_eq!(cloned.frame_id(), 1);
        assert_eq!(cloned.pc(), 50);
    }

    // =========================================================================
    // ExcInfoStack Basic Tests
    // =========================================================================

    #[test]
    fn test_exc_info_stack_new() {
        let stack = ExcInfoStack::new();
        assert!(stack.is_empty());
        assert_eq!(stack.len(), 0);
    }

    #[test]
    fn test_exc_info_stack_with_capacity() {
        let stack = ExcInfoStack::with_capacity(16);
        assert!(stack.is_empty());
    }

    #[test]
    fn test_exc_info_stack_push_pop() {
        let mut stack = ExcInfoStack::new();

        let entry = ExcInfoEntry::new(24, None);
        assert!(stack.push(entry));
        assert_eq!(stack.len(), 1);

        let popped = stack.pop();
        assert!(popped.is_some());
        assert_eq!(popped.unwrap().type_id(), 24);
        assert!(stack.is_empty());
    }

    #[test]
    fn test_exc_info_stack_peek() {
        let mut stack = ExcInfoStack::new();
        assert!(stack.peek().is_none());

        stack.push(ExcInfoEntry::new(24, None));
        stack.push(ExcInfoEntry::new(5, None));

        let top = stack.peek();
        assert!(top.is_some());
        assert_eq!(top.unwrap().type_id(), 5);
        assert_eq!(stack.len(), 2); // Peek doesn't remove
    }

    #[test]
    fn test_exc_info_stack_peek_mut() {
        let mut stack = ExcInfoStack::new();
        stack.push(ExcInfoEntry::new(24, None));

        if let Some(entry) = stack.peek_mut() {
            entry.set_traceback_id(999);
        }

        assert_eq!(stack.peek().unwrap().traceback_id(), 999);
    }

    #[test]
    fn test_exc_info_stack_lifo_order() {
        let mut stack = ExcInfoStack::new();

        stack.push(ExcInfoEntry::new(1, None));
        stack.push(ExcInfoEntry::new(2, None));
        stack.push(ExcInfoEntry::new(3, None));

        assert_eq!(stack.pop().unwrap().type_id(), 3);
        assert_eq!(stack.pop().unwrap().type_id(), 2);
        assert_eq!(stack.pop().unwrap().type_id(), 1);
        assert!(stack.pop().is_none());
    }

    #[test]
    fn test_exc_info_stack_get() {
        let mut stack = ExcInfoStack::new();

        stack.push(ExcInfoEntry::new(1, None));
        stack.push(ExcInfoEntry::new(2, None));
        stack.push(ExcInfoEntry::new(3, None));

        assert_eq!(stack.get(0).unwrap().type_id(), 1); // Bottom
        assert_eq!(stack.get(2).unwrap().type_id(), 3); // Top
        assert!(stack.get(10).is_none());
    }

    #[test]
    fn test_exc_info_stack_clear() {
        let mut stack = ExcInfoStack::new();

        stack.push(ExcInfoEntry::new(1, None));
        stack.push(ExcInfoEntry::new(2, None));
        assert_eq!(stack.len(), 2);

        stack.clear();
        assert!(stack.is_empty());
    }

    #[test]
    fn test_exc_info_stack_truncate() {
        let mut stack = ExcInfoStack::new();

        stack.push(ExcInfoEntry::new(1, None));
        stack.push(ExcInfoEntry::new(2, None));
        stack.push(ExcInfoEntry::new(3, None));
        stack.push(ExcInfoEntry::new(4, None));

        stack.truncate(2);
        assert_eq!(stack.len(), 2);
        assert_eq!(stack.peek().unwrap().type_id(), 2);
    }

    // =========================================================================
    // ExcInfoStack Iterator Tests
    // =========================================================================

    #[test]
    fn test_exc_info_stack_iter_top_to_bottom() {
        let mut stack = ExcInfoStack::new();

        stack.push(ExcInfoEntry::new(1, None));
        stack.push(ExcInfoEntry::new(2, None));
        stack.push(ExcInfoEntry::new(3, None));

        let type_ids: Vec<u16> = stack.iter().map(|e| e.type_id()).collect();
        assert_eq!(type_ids, vec![3, 2, 1]); // Top to bottom
    }

    #[test]
    fn test_exc_info_stack_iter_bottom_to_top() {
        let mut stack = ExcInfoStack::new();

        stack.push(ExcInfoEntry::new(1, None));
        stack.push(ExcInfoEntry::new(2, None));
        stack.push(ExcInfoEntry::new(3, None));

        let type_ids: Vec<u16> = stack.iter_bottom_up().map(|e| e.type_id()).collect();
        assert_eq!(type_ids, vec![1, 2, 3]); // Bottom to top
    }

    // =========================================================================
    // ExcInfoStack Overflow Tests
    // =========================================================================

    #[test]
    fn test_exc_info_stack_max_depth() {
        assert_eq!(ExcInfoStack::max_depth(), 255);
    }

    #[test]
    fn test_exc_info_stack_overflow_protection() {
        let mut stack = ExcInfoStack::new();

        // Fill to max
        for i in 0..MAX_DEPTH {
            assert!(stack.push(ExcInfoEntry::new(i as u16, None)));
        }

        assert_eq!(stack.len(), MAX_DEPTH);

        // Next push should fail
        assert!(!stack.push(ExcInfoEntry::new(999, None)));
        assert_eq!(stack.len(), MAX_DEPTH);
        assert_eq!(stack.stats().overflows, 1);
    }

    // =========================================================================
    // ExcInfoStack Statistics Tests
    // =========================================================================

    #[test]
    fn test_exc_info_stack_stats_pushes() {
        let mut stack = ExcInfoStack::new();

        stack.push(ExcInfoEntry::new(1, None));
        stack.push(ExcInfoEntry::new(2, None));
        stack.push(ExcInfoEntry::new(3, None));

        assert_eq!(stack.stats().pushes, 3);
    }

    #[test]
    fn test_exc_info_stack_stats_pops() {
        let mut stack = ExcInfoStack::new();

        stack.push(ExcInfoEntry::new(1, None));
        stack.push(ExcInfoEntry::new(2, None));

        stack.pop();
        stack.pop();
        stack.pop(); // Extra pop on empty stack

        assert_eq!(stack.stats().pops, 2); // Only successful pops counted
    }

    #[test]
    fn test_exc_info_stack_stats_peak_depth() {
        let mut stack = ExcInfoStack::new();

        stack.push(ExcInfoEntry::new(1, None));
        stack.push(ExcInfoEntry::new(2, None));
        stack.push(ExcInfoEntry::new(3, None));
        stack.pop();
        stack.pop();
        stack.push(ExcInfoEntry::new(4, None));

        assert_eq!(stack.len(), 2);
        assert_eq!(stack.stats().peak_depth, 3); // Peak was 3
    }

    #[test]
    fn test_exc_info_stack_reset_stats() {
        let mut stack = ExcInfoStack::new();

        stack.push(ExcInfoEntry::new(1, None));
        stack.pop();

        assert_eq!(stack.stats().pushes, 1);
        assert_eq!(stack.stats().pops, 1);

        stack.reset_stats();

        assert_eq!(stack.stats().pushes, 0);
        assert_eq!(stack.stats().pops, 0);
        assert_eq!(stack.stats().peak_depth, 0);
    }

    // =========================================================================
    // ExcInfoStack Current Exc Info Tests
    // =========================================================================

    #[test]
    fn test_exc_info_stack_current_exc_info_empty() {
        let stack = ExcInfoStack::new();
        let (type_id, value, tb) = stack.current_exc_info();

        assert!(type_id.is_none());
        assert!(value.is_none());
        assert!(tb.is_none());
    }

    #[test]
    fn test_exc_info_stack_current_exc_info_with_entry() {
        let mut stack = ExcInfoStack::new();

        let mut entry = ExcInfoEntry::new(24, Some(Value::int(42).unwrap()));
        entry.set_traceback_id(100);
        stack.push(entry);

        let (type_id, value, tb) = stack.current_exc_info();

        assert_eq!(type_id, Some(24));
        assert!(value.is_some());
        assert_eq!(tb, Some(100));
    }

    #[test]
    fn test_exc_info_stack_current_exc_info_no_traceback() {
        let mut stack = ExcInfoStack::new();

        stack.push(ExcInfoEntry::new(24, Some(Value::none())));

        let (type_id, value, tb) = stack.current_exc_info();

        assert_eq!(type_id, Some(24));
        assert!(value.is_some());
        assert!(tb.is_none()); // traceback_id is 0
    }

    // =========================================================================
    // ExcInfoStack Find Active Tests
    // =========================================================================

    #[test]
    fn test_exc_info_stack_find_active_empty() {
        let stack = ExcInfoStack::new();
        assert!(stack.find_active().is_none());
    }

    #[test]
    fn test_exc_info_stack_find_active_all_empty_entries() {
        let mut stack = ExcInfoStack::new();

        stack.push(ExcInfoEntry::empty());
        stack.push(ExcInfoEntry::empty());

        assert!(stack.find_active().is_none());
    }

    #[test]
    fn test_exc_info_stack_find_active_returns_top() {
        let mut stack = ExcInfoStack::new();

        stack.push(ExcInfoEntry::new(1, None)); // Active
        stack.push(ExcInfoEntry::empty()); // Empty
        stack.push(ExcInfoEntry::new(3, None)); // Active - top

        let active = stack.find_active();
        assert!(active.is_some());
        assert_eq!(active.unwrap().type_id(), 3);
    }

    #[test]
    fn test_exc_info_stack_find_active_skips_empty() {
        let mut stack = ExcInfoStack::new();

        stack.push(ExcInfoEntry::new(1, None)); // Active
        stack.push(ExcInfoEntry::empty()); // Empty - on top

        let active = stack.find_active();
        assert!(active.is_some());
        assert_eq!(active.unwrap().type_id(), 1);
    }

    // =========================================================================
    // ExcInfoStack Default Tests
    // =========================================================================

    #[test]
    fn test_exc_info_stack_default() {
        let stack = ExcInfoStack::default();
        assert!(stack.is_empty());
    }

    #[test]
    fn test_exc_info_stack_debug() {
        let mut stack = ExcInfoStack::new();
        stack.push(ExcInfoEntry::new(24, None));

        let debug = format!("{:?}", stack);
        assert!(debug.contains("ExcInfoStack"));
        assert!(debug.contains("depth: 1"));
    }

    #[test]
    fn test_exc_info_stack_stats_debug() {
        let stats = ExcInfoStackStats {
            pushes: 10,
            pops: 5,
            peak_depth: 3,
            overflows: 0,
        };

        let debug = format!("{:?}", stats);
        assert!(debug.contains("pushes: 10"));
        assert!(debug.contains("pops: 5"));
        assert!(debug.contains("peak_depth: 3"));
    }

    // =========================================================================
    // Memory Layout Tests
    // =========================================================================

    #[test]
    fn test_entry_flags_size() {
        assert_eq!(std::mem::size_of::<EntryFlags>(), 1);
    }

    #[test]
    fn test_exc_info_stack_inline_capacity() {
        assert_eq!(INLINE_CAPACITY, 4);
    }

    // =========================================================================
    // Integration Scenario Tests
    // =========================================================================

    #[test]
    fn test_nested_try_except_scenario() {
        let mut stack = ExcInfoStack::new();

        // Outer try block enters except handler
        stack.push(ExcInfoEntry::new(24, Some(Value::int(1).unwrap())));

        // Inner try block enters except handler
        stack.push(ExcInfoEntry::new(5, Some(Value::int(2).unwrap())));

        // Inner handler exits - restore outer exception
        let inner = stack.pop().unwrap();
        assert_eq!(inner.type_id(), 5);

        // Outer exception is now current
        let current = stack.peek().unwrap();
        assert_eq!(current.type_id(), 24);

        // Outer handler exits
        stack.pop();
        assert!(stack.is_empty());
    }

    #[test]
    fn test_raise_from_scenario() {
        let mut stack = ExcInfoStack::new();

        // Original exception
        let mut original = ExcInfoEntry::new(24, Some(Value::int(1).unwrap()));
        stack.push(original);

        // raise NewException from original_exception
        let mut chained = ExcInfoEntry::new(5, Some(Value::int(2).unwrap()));
        chained.set_from_raise_from();
        stack.push(chained);

        let top = stack.peek().unwrap();
        assert!(top.flags().has(EntryFlags::FROM_RAISE_FROM));
        assert!(top.flags().has_cause());
        assert!(top.flags().is_context_suppressed());
    }

    #[test]
    fn test_finally_block_scenario() {
        let mut stack = ExcInfoStack::new();

        // Exception raised
        stack.push(ExcInfoEntry::new(24, Some(Value::int(1).unwrap())));

        // Entering finally block - save exception state
        let saved_depth = stack.len();
        assert_eq!(saved_depth, 1);

        // Finally block runs...
        // (exception state preserved on stack)

        // Exiting finally - exception still there for potential reraise
        assert!(stack.peek().is_some());
        assert_eq!(stack.peek().unwrap().type_id(), 24);
    }
}
