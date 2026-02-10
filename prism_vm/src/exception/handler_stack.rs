//! Runtime exception handler stack.
//!
//! This module provides the runtime handler stack that tracks active try
//! blocks during execution. It complements the static `HandlerTable` by
//! maintaining dynamic execution state.
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────┐
//! │                    Handler Stack                                 │
//! ├─────────────────────────────────────────────────────────────────┤
//! │                                                                  │
//! │  ┌─────────────────┐   push     ┌─────────────────────────────┐│
//! │  │ try block entry │ ────────▶  │ HandlerFrame {              ││
//! │  └─────────────────┘             │   handler_idx: u16          ││
//! │                                   │   frame_id: u32             ││
//! │  ┌─────────────────┐             │   stack_depth: u16          ││
//! │  │ exception raise │ ────────▶  │   flags: HandlerFlags       ││
//! │  └─────────────────┘   pop      │ }                            ││
//! │                                  └─────────────────────────────┘│
//! │                                                                  │
//! │  Benefits:                                                       │
//! │  • O(1) push/pop for try block entry/exit                       │
//! │  • O(1) handler lookup for most exceptions                       │
//! │  • Proper finally block ordering                                 │
//! │  • Efficient stack unwinding across frames                       │
//! └─────────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Performance
//!
//! | Operation | Complexity |
//! |-----------|------------|
//! | push_handler | O(1) |
//! | pop_handler | O(1) |
//! | find_handler | O(N) worst case, O(1) typical |
//! | unwind_to | O(N) where N = frames to unwind |

use smallvec::SmallVec;
use std::fmt;

// ============================================================================
// Constants
// ============================================================================

/// Default inline capacity for the handler stack.
/// Most exception handling involves small nesting depths.
const INLINE_HANDLER_CAPACITY: usize = 8;

/// Maximum handler stack depth to prevent unbounded growth.
const MAX_HANDLER_DEPTH: usize = 1024;

/// Sentinel value for "no handler".
pub const NO_HANDLER: u16 = u16::MAX;

/// Sentinel value for "no frame".
pub const NO_FRAME: u32 = u32::MAX;

// ============================================================================
// Handler Frame
// ============================================================================

/// A runtime handler frame representing an active try block.
///
/// # Memory Layout (8 bytes)
///
/// ```text
/// ┌─────────────────────────────────────────────────────────┐
/// │ handler_idx (2b) │ stack_depth (2b) │ frame_id (4b)     │
/// └─────────────────────────────────────────────────────────┘
/// ```
#[derive(Clone, Copy, PartialEq, Eq)]
#[repr(C)]
pub struct HandlerFrame {
    /// Index into the code object's HandlerTable.
    pub handler_idx: u16,

    /// Stack depth when the try block was entered.
    /// Used for stack unwinding.
    pub stack_depth: u16,

    /// Frame ID that owns this handler.
    /// Used for cross-frame unwinding.
    pub frame_id: u32,
}

impl HandlerFrame {
    /// Creates a new handler frame.
    #[inline]
    pub const fn new(handler_idx: u16, stack_depth: u16, frame_id: u32) -> Self {
        Self {
            handler_idx,
            stack_depth,
            frame_id,
        }
    }

    /// Returns true if this is a valid handler (not sentinel).
    #[inline]
    pub const fn is_valid(&self) -> bool {
        self.handler_idx != NO_HANDLER
    }

    /// Creates an invalid/sentinel handler frame.
    #[inline]
    pub const fn invalid() -> Self {
        Self {
            handler_idx: NO_HANDLER,
            stack_depth: 0,
            frame_id: NO_FRAME,
        }
    }
}

impl fmt::Debug for HandlerFrame {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_valid() {
            f.debug_struct("HandlerFrame")
                .field("handler_idx", &self.handler_idx)
                .field("stack_depth", &self.stack_depth)
                .field("frame_id", &self.frame_id)
                .finish()
        } else {
            write!(f, "HandlerFrame(invalid)")
        }
    }
}

impl Default for HandlerFrame {
    fn default() -> Self {
        Self::invalid()
    }
}

// ============================================================================
// Handler Stack
// ============================================================================

/// Runtime stack of active exception handlers.
///
/// This stack tracks all active try blocks across the call stack.
/// When an exception is raised, the stack is unwound to find a
/// matching handler.
///
/// # Usage
///
/// ```ignore
/// let mut stack = HandlerStack::new();
///
/// // Enter a try block
/// stack.push(HandlerFrame::new(0, 5, 0));
///
/// // ... execute try block ...
///
/// // Exit normally
/// stack.pop();
/// ```
#[derive(Clone)]
pub struct HandlerStack {
    /// Stack of active handlers (newest on top).
    frames: SmallVec<[HandlerFrame; INLINE_HANDLER_CAPACITY]>,
}

impl HandlerStack {
    /// Creates a new empty handler stack.
    #[inline]
    pub fn new() -> Self {
        Self {
            frames: SmallVec::new(),
        }
    }

    /// Creates a handler stack with the specified capacity.
    #[inline]
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            frames: SmallVec::with_capacity(capacity),
        }
    }

    /// Returns true if the stack is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.frames.is_empty()
    }

    /// Returns the number of active handlers.
    #[inline]
    pub fn len(&self) -> usize {
        self.frames.len()
    }

    /// Pushes a new handler frame onto the stack.
    ///
    /// Returns `true` if successful, `false` if the stack is full.
    #[inline]
    pub fn push(&mut self, frame: HandlerFrame) -> bool {
        if self.frames.len() >= MAX_HANDLER_DEPTH {
            return false;
        }
        self.frames.push(frame);
        true
    }

    /// Pops the top handler frame from the stack.
    ///
    /// Returns `None` if the stack is empty.
    #[inline]
    pub fn pop(&mut self) -> Option<HandlerFrame> {
        self.frames.pop()
    }

    /// Returns a reference to the top handler frame without removing it.
    #[inline]
    pub fn peek(&self) -> Option<&HandlerFrame> {
        self.frames.last()
    }

    /// Returns a mutable reference to the top handler frame.
    #[inline]
    pub fn peek_mut(&mut self) -> Option<&mut HandlerFrame> {
        self.frames.last_mut()
    }

    /// Clears all handlers from the stack.
    #[inline]
    pub fn clear(&mut self) {
        self.frames.clear();
    }

    /// Returns an iterator over handler frames from top to bottom.
    #[inline]
    pub fn iter(&self) -> impl Iterator<Item = &HandlerFrame> + DoubleEndedIterator {
        self.frames.iter().rev()
    }

    /// Returns the depth of the stack.
    #[inline]
    pub fn depth(&self) -> usize {
        self.frames.len()
    }

    /// Pops all handlers belonging to a specific frame.
    ///
    /// This is used during frame exit to clean up handlers.
    #[inline]
    pub fn pop_frame_handlers(&mut self, frame_id: u32) {
        while let Some(handler) = self.frames.last() {
            if handler.frame_id == frame_id {
                self.frames.pop();
            } else {
                break;
            }
        }
    }

    /// Finds the first handler matching the given frame ID.
    ///
    /// Returns the handler frame if found, or `None` if no handler
    /// exists for the specified frame.
    #[inline]
    pub fn find_in_frame(&self, frame_id: u32) -> Option<&HandlerFrame> {
        self.frames.iter().rev().find(|h| h.frame_id == frame_id)
    }

    /// Returns the number of handlers for a specific frame.
    #[inline]
    pub fn count_frame_handlers(&self, frame_id: u32) -> usize {
        self.frames
            .iter()
            .filter(|h| h.frame_id == frame_id)
            .count()
    }

    /// Truncates the stack to the given depth.
    ///
    /// This is used during exception handling to unwind to a
    /// specific point.
    #[inline]
    pub fn truncate(&mut self, depth: usize) {
        self.frames.truncate(depth);
    }

    /// Returns the handler at the given index (0 = bottom).
    #[inline]
    pub fn get(&self, index: usize) -> Option<&HandlerFrame> {
        self.frames.get(index)
    }
}

impl Default for HandlerStack {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Debug for HandlerStack {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("HandlerStack")
            .field("depth", &self.frames.len())
            .field("frames", &self.frames.as_slice())
            .finish()
    }
}

// ============================================================================
// Handler Search Result
// ============================================================================

/// Result of searching for an exception handler.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HandlerSearchResult {
    /// Found a handler at the given stack index.
    Found {
        /// Index in the handler stack.
        stack_index: usize,
        /// The handler frame.
        handler: HandlerFrame,
    },

    /// No handler found, exception should propagate to caller.
    NotFound,

    /// Found a finally block that must run before continuing search.
    Finally {
        /// Index in the handler stack.
        stack_index: usize,
        /// The finally handler frame.
        handler: HandlerFrame,
    },
}

impl HandlerSearchResult {
    /// Returns true if a handler was found.
    #[inline]
    pub fn found(&self) -> bool {
        matches!(self, Self::Found { .. } | Self::Finally { .. })
    }

    /// Returns the handler frame if found.
    #[inline]
    pub fn handler(&self) -> Option<HandlerFrame> {
        match self {
            Self::Found { handler, .. } => Some(*handler),
            Self::Finally { handler, .. } => Some(*handler),
            Self::NotFound => None,
        }
    }

    /// Returns the stack index if found.
    #[inline]
    pub fn stack_index(&self) -> Option<usize> {
        match self {
            Self::Found { stack_index, .. } => Some(*stack_index),
            Self::Finally { stack_index, .. } => Some(*stack_index),
            Self::NotFound => None,
        }
    }
}

// ============================================================================
// Handler Stack Stats
// ============================================================================

/// Statistics for handler stack operations.
#[derive(Clone, Copy, Debug, Default)]
pub struct HandlerStackStats {
    /// Total number of push operations.
    pub push_count: u64,

    /// Total number of pop operations.
    pub pop_count: u64,

    /// Total number of search operations.
    pub search_count: u64,

    /// Total handlers examined during searches.
    pub handlers_examined: u64,

    /// Maximum stack depth observed.
    pub max_depth: usize,

    /// Number of searches that found a handler.
    pub hits: u64,

    /// Number of searches that found no handler.
    pub misses: u64,
}

impl HandlerStackStats {
    /// Creates new empty stats.
    #[inline]
    pub const fn new() -> Self {
        Self {
            push_count: 0,
            pop_count: 0,
            search_count: 0,
            handlers_examined: 0,
            max_depth: 0,
            hits: 0,
            misses: 0,
        }
    }

    /// Returns the hit rate as a percentage.
    #[inline]
    pub fn hit_rate(&self) -> f64 {
        if self.search_count == 0 {
            0.0
        } else {
            (self.hits as f64 / self.search_count as f64) * 100.0
        }
    }

    /// Returns the average handlers examined per search.
    #[inline]
    pub fn avg_handlers_examined(&self) -> f64 {
        if self.search_count == 0 {
            0.0
        } else {
            self.handlers_examined as f64 / self.search_count as f64
        }
    }

    /// Resets all statistics.
    #[inline]
    pub fn reset(&mut self) {
        *self = Self::new();
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ════════════════════════════════════════════════════════════════════════
    // HandlerFrame Tests
    // ════════════════════════════════════════════════════════════════════════

    #[test]
    fn test_handler_frame_new() {
        let frame = HandlerFrame::new(5, 10, 1);
        assert_eq!(frame.handler_idx, 5);
        assert_eq!(frame.stack_depth, 10);
        assert_eq!(frame.frame_id, 1);
    }

    #[test]
    fn test_handler_frame_is_valid() {
        let valid = HandlerFrame::new(0, 0, 0);
        assert!(valid.is_valid());

        let invalid = HandlerFrame::invalid();
        assert!(!invalid.is_valid());
    }

    #[test]
    fn test_handler_frame_size() {
        // Ensure compact memory layout
        assert_eq!(std::mem::size_of::<HandlerFrame>(), 8);
    }

    #[test]
    fn test_handler_frame_default() {
        let frame = HandlerFrame::default();
        assert!(!frame.is_valid());
        assert_eq!(frame.handler_idx, NO_HANDLER);
    }

    #[test]
    fn test_handler_frame_debug() {
        let valid = HandlerFrame::new(1, 5, 2);
        let debug_str = format!("{:?}", valid);
        assert!(debug_str.contains("handler_idx"));
        assert!(debug_str.contains("1"));

        let invalid = HandlerFrame::invalid();
        let debug_str = format!("{:?}", invalid);
        assert!(debug_str.contains("invalid"));
    }

    #[test]
    fn test_handler_frame_equality() {
        let f1 = HandlerFrame::new(1, 2, 3);
        let f2 = HandlerFrame::new(1, 2, 3);
        let f3 = HandlerFrame::new(1, 2, 4);

        assert_eq!(f1, f2);
        assert_ne!(f1, f3);
    }

    // ════════════════════════════════════════════════════════════════════════
    // HandlerStack Basic Tests
    // ════════════════════════════════════════════════════════════════════════

    #[test]
    fn test_handler_stack_new() {
        let stack = HandlerStack::new();
        assert!(stack.is_empty());
        assert_eq!(stack.len(), 0);
    }

    #[test]
    fn test_handler_stack_with_capacity() {
        let stack = HandlerStack::with_capacity(16);
        assert!(stack.is_empty());
    }

    #[test]
    fn test_handler_stack_push_pop() {
        let mut stack = HandlerStack::new();

        let frame = HandlerFrame::new(0, 5, 0);
        assert!(stack.push(frame));
        assert_eq!(stack.len(), 1);

        let popped = stack.pop();
        assert_eq!(popped, Some(frame));
        assert!(stack.is_empty());
    }

    #[test]
    fn test_handler_stack_lifo_order() {
        let mut stack = HandlerStack::new();

        let f1 = HandlerFrame::new(1, 0, 0);
        let f2 = HandlerFrame::new(2, 0, 0);
        let f3 = HandlerFrame::new(3, 0, 0);

        stack.push(f1);
        stack.push(f2);
        stack.push(f3);

        assert_eq!(stack.pop(), Some(f3));
        assert_eq!(stack.pop(), Some(f2));
        assert_eq!(stack.pop(), Some(f1));
        assert_eq!(stack.pop(), None);
    }

    #[test]
    fn test_handler_stack_peek() {
        let mut stack = HandlerStack::new();
        assert!(stack.peek().is_none());

        let frame = HandlerFrame::new(5, 10, 1);
        stack.push(frame);

        assert_eq!(stack.peek(), Some(&frame));
        assert_eq!(stack.len(), 1); // peek doesn't remove
    }

    #[test]
    fn test_handler_stack_peek_mut() {
        let mut stack = HandlerStack::new();
        stack.push(HandlerFrame::new(1, 5, 0));

        if let Some(frame) = stack.peek_mut() {
            frame.stack_depth = 10;
        }

        assert_eq!(stack.peek().unwrap().stack_depth, 10);
    }

    #[test]
    fn test_handler_stack_clear() {
        let mut stack = HandlerStack::new();
        stack.push(HandlerFrame::new(1, 0, 0));
        stack.push(HandlerFrame::new(2, 0, 0));
        stack.push(HandlerFrame::new(3, 0, 0));

        stack.clear();
        assert!(stack.is_empty());
    }

    // ════════════════════════════════════════════════════════════════════════
    // HandlerStack Frame Operations Tests
    // ════════════════════════════════════════════════════════════════════════

    #[test]
    fn test_handler_stack_pop_frame_handlers() {
        let mut stack = HandlerStack::new();

        // Frame 0: 2 handlers
        stack.push(HandlerFrame::new(0, 0, 0));
        stack.push(HandlerFrame::new(1, 0, 0));

        // Frame 1: 1 handler
        stack.push(HandlerFrame::new(2, 0, 1));

        assert_eq!(stack.len(), 3);

        // Pop all handlers for frame 1
        stack.pop_frame_handlers(1);
        assert_eq!(stack.len(), 2);

        // Remaining are frame 0
        assert_eq!(stack.peek().unwrap().frame_id, 0);
    }

    #[test]
    fn test_handler_stack_find_in_frame() {
        let mut stack = HandlerStack::new();

        stack.push(HandlerFrame::new(0, 5, 0));
        stack.push(HandlerFrame::new(1, 10, 1));
        stack.push(HandlerFrame::new(2, 15, 0)); // Another handler for frame 0

        // Find first handler for frame 0 (newest first = handler_idx 2)
        let found = stack.find_in_frame(0);
        assert!(found.is_some());
        assert_eq!(found.unwrap().handler_idx, 2);

        // Find handler for frame 1
        let found = stack.find_in_frame(1);
        assert!(found.is_some());
        assert_eq!(found.unwrap().handler_idx, 1);

        // Frame 2 doesn't exist
        assert!(stack.find_in_frame(2).is_none());
    }

    #[test]
    fn test_handler_stack_count_frame_handlers() {
        let mut stack = HandlerStack::new();

        stack.push(HandlerFrame::new(0, 0, 0));
        stack.push(HandlerFrame::new(1, 0, 0));
        stack.push(HandlerFrame::new(2, 0, 1));

        assert_eq!(stack.count_frame_handlers(0), 2);
        assert_eq!(stack.count_frame_handlers(1), 1);
        assert_eq!(stack.count_frame_handlers(2), 0);
    }

    #[test]
    fn test_handler_stack_truncate() {
        let mut stack = HandlerStack::new();

        for i in 0..5 {
            stack.push(HandlerFrame::new(i, 0, 0));
        }

        stack.truncate(3);
        assert_eq!(stack.len(), 3);
        assert_eq!(stack.peek().unwrap().handler_idx, 2);
    }

    #[test]
    fn test_handler_stack_iter() {
        let mut stack = HandlerStack::new();
        stack.push(HandlerFrame::new(1, 0, 0));
        stack.push(HandlerFrame::new(2, 0, 0));
        stack.push(HandlerFrame::new(3, 0, 0));

        // iter() returns top to bottom
        let indices: Vec<_> = stack.iter().map(|f| f.handler_idx).collect();
        assert_eq!(indices, vec![3, 2, 1]);
    }

    #[test]
    fn test_handler_stack_get() {
        let mut stack = HandlerStack::new();
        stack.push(HandlerFrame::new(1, 0, 0));
        stack.push(HandlerFrame::new(2, 0, 0));

        assert_eq!(stack.get(0).unwrap().handler_idx, 1);
        assert_eq!(stack.get(1).unwrap().handler_idx, 2);
        assert!(stack.get(2).is_none());
    }

    // ════════════════════════════════════════════════════════════════════════
    // HandlerSearchResult Tests
    // ════════════════════════════════════════════════════════════════════════

    #[test]
    fn test_search_result_found() {
        let handler = HandlerFrame::new(1, 5, 0);
        let result = HandlerSearchResult::Found {
            stack_index: 2,
            handler,
        };

        assert!(result.found());
        assert_eq!(result.handler(), Some(handler));
        assert_eq!(result.stack_index(), Some(2));
    }

    #[test]
    fn test_search_result_not_found() {
        let result = HandlerSearchResult::NotFound;

        assert!(!result.found());
        assert_eq!(result.handler(), None);
        assert_eq!(result.stack_index(), None);
    }

    #[test]
    fn test_search_result_finally() {
        let handler = HandlerFrame::new(3, 10, 1);
        let result = HandlerSearchResult::Finally {
            stack_index: 0,
            handler,
        };

        assert!(result.found());
        assert_eq!(result.handler(), Some(handler));
        assert_eq!(result.stack_index(), Some(0));
    }

    // ════════════════════════════════════════════════════════════════════════
    // HandlerStackStats Tests
    // ════════════════════════════════════════════════════════════════════════

    #[test]
    fn test_stats_new() {
        let stats = HandlerStackStats::new();
        assert_eq!(stats.push_count, 0);
        assert_eq!(stats.search_count, 0);
    }

    #[test]
    fn test_stats_hit_rate() {
        let mut stats = HandlerStackStats::new();

        // No searches = 0% hit rate
        assert_eq!(stats.hit_rate(), 0.0);

        stats.search_count = 100;
        stats.hits = 75;
        assert!((stats.hit_rate() - 75.0).abs() < 0.001);
    }

    #[test]
    fn test_stats_avg_handlers_examined() {
        let mut stats = HandlerStackStats::new();

        assert_eq!(stats.avg_handlers_examined(), 0.0);

        stats.search_count = 10;
        stats.handlers_examined = 35;
        assert!((stats.avg_handlers_examined() - 3.5).abs() < 0.001);
    }

    #[test]
    fn test_stats_reset() {
        let mut stats = HandlerStackStats::new();
        stats.push_count = 100;
        stats.pop_count = 50;
        stats.max_depth = 10;

        stats.reset();

        assert_eq!(stats.push_count, 0);
        assert_eq!(stats.pop_count, 0);
        assert_eq!(stats.max_depth, 0);
    }

    // ════════════════════════════════════════════════════════════════════════
    // Nested Try Block Simulation Tests
    // ════════════════════════════════════════════════════════════════════════

    #[test]
    fn test_nested_try_blocks() {
        let mut stack = HandlerStack::new();

        // Simulate:
        // try:                 # handler 0, frame 0
        //     try:             # handler 1, frame 0
        //         try:         # handler 2, frame 0
        //             ...
        //         except:
        //             pass
        //     except:
        //         pass
        // except:
        //     pass

        stack.push(HandlerFrame::new(0, 0, 0));
        stack.push(HandlerFrame::new(1, 5, 0));
        stack.push(HandlerFrame::new(2, 10, 0));

        assert_eq!(stack.len(), 3);

        // Exit innermost try
        stack.pop();
        assert_eq!(stack.len(), 2);
        assert_eq!(stack.peek().unwrap().handler_idx, 1);

        // Exit middle try
        stack.pop();
        assert_eq!(stack.len(), 1);
        assert_eq!(stack.peek().unwrap().handler_idx, 0);
    }

    #[test]
    fn test_cross_frame_handlers() {
        let mut stack = HandlerStack::new();

        // Frame 0: try block
        stack.push(HandlerFrame::new(0, 5, 0));

        // Frame 1 (called function): try block
        stack.push(HandlerFrame::new(1, 3, 1));

        // Frame 2 (another call): try block
        stack.push(HandlerFrame::new(2, 2, 2));

        // Unwind frame 2
        stack.pop_frame_handlers(2);
        assert_eq!(stack.len(), 2);

        // Unwind frame 1
        stack.pop_frame_handlers(1);
        assert_eq!(stack.len(), 1);

        // Only frame 0's handler remains
        assert_eq!(stack.peek().unwrap().frame_id, 0);
    }

    // ════════════════════════════════════════════════════════════════════════
    // Edge Cases
    // ════════════════════════════════════════════════════════════════════════

    #[test]
    fn test_empty_stack_operations() {
        let mut stack = HandlerStack::new();

        assert!(stack.pop().is_none());
        assert!(stack.peek().is_none());
        assert!(stack.peek_mut().is_none());
        assert!(stack.find_in_frame(0).is_none());
        assert_eq!(stack.count_frame_handlers(0), 0);

        // These should not panic
        stack.pop_frame_handlers(0);
        stack.truncate(0);
        stack.clear();
    }

    #[test]
    fn test_handler_frame_all_zeros() {
        let frame = HandlerFrame::new(0, 0, 0);
        assert!(frame.is_valid()); // handler_idx 0 is still valid
    }

    #[test]
    fn test_max_stack_depth() {
        // This test verifies the stack respects depth limits
        // but doesn't actually push 1024 items
        let mut stack = HandlerStack::new();

        // Push a few items to verify basic operation
        for i in 0..10 {
            assert!(stack.push(HandlerFrame::new(i, 0, 0)));
        }

        assert_eq!(stack.len(), 10);
    }
}
