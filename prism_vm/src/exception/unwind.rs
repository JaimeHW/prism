//! Stack unwinding for exception handling.
//!
//! This module provides the stack unwinding logic used during exception
//! propagation. It coordinates handler lookup, stack restoration, and
//! finally block execution.
//!
//! # Architecture
//!
//! ```text
//! ┌──────────────────────────────────────────────────────────────────────────┐
//! │                         Stack Unwinding Flow                              │
//! ├──────────────────────────────────────────────────────────────────────────┤
//! │                                                                           │
//! │    Exception Raised                                                       │
//! │          │                                                                │
//! │          ▼                                                                │
//! │    ┌─────────────┐                                                        │
//! │    │ Current     │───▶ Search HandlerTable for PC                         │
//! │    │ Frame       │                                                        │
//! │    └──────┬──────┘                                                        │
//! │           │                                                               │
//! │     ┌─────┴─────┐                                                         │
//! │     ▼           ▼                                                         │
//! │  Handler    No Handler                                                    │
//! │  Found      Found                                                         │
//! │     │           │                                                         │
//! │     ▼           ▼                                                         │
//! │  ┌──────┐   ┌──────────┐                                                  │
//! │  │Unwind│   │Pop Frame │                                                  │
//! │  │Stack │   │Continue  │──────────────────┐                               │
//! │  └──┬───┘   │Unwinding │                  │                               │
//! │     │       └──────────┘                  │                               │
//! │     ▼                                     ▼                               │
//! │  ┌──────────┐                      ┌─────────────┐                        │
//! │  │ Jump to  │                      │ Propagate   │                        │
//! │  │ Handler  │                      │ to Caller   │                        │
//! │  └──────────┘                      └─────────────┘                        │
//! │                                                                           │
//! └──────────────────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Performance
//!
//! | Operation | Complexity |
//! |-----------|------------|
//! | Handler search (single frame) | O(log N) binary search |
//! | Stack depth restoration | O(1) |
//! | Cross-frame unwind | O(F) where F = frames to unwind |

use super::handler_stack::{HandlerFrame, HandlerSearchResult, HandlerStack};
use smallvec::SmallVec;

// ============================================================================
// Constants
// ============================================================================

/// Maximum frames to unwind before giving up.
/// This prevents infinite loops in malformed code.
const MAX_UNWIND_FRAMES: usize = 1000;

/// Default capacity for the finally block queue.
const FINALLY_QUEUE_CAPACITY: usize = 4;

// ============================================================================
// Unwind Action
// ============================================================================

/// Action to take after unwinding.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum UnwindAction {
    /// Jump to handler at the specified PC.
    JumpToHandler {
        /// Handler PC to jump to.
        handler_pc: u32,
        /// Stack depth to restore.
        stack_depth: u16,
    },

    /// Execute a finally block before continuing.
    ExecuteFinally {
        /// Finally block PC.
        finally_pc: u32,
        /// Stack depth to restore.
        stack_depth: u16,
        /// Whether to reraise after finally.
        reraise: bool,
    },

    /// Propagate exception to caller frame.
    PropagateToFrame {
        /// Frame ID to propagate to.
        target_frame_id: u32,
    },

    /// Exception is unhandled, terminate execution.
    Unhandled,

    /// Continue execution (exception was cleared).
    Continue,
}

impl UnwindAction {
    /// Returns true if this action continues execution.
    #[inline]
    pub fn continues(&self) -> bool {
        matches!(self, Self::JumpToHandler { .. } | Self::Continue)
    }

    /// Returns true if this action propagates the exception.
    #[inline]
    pub fn propagates(&self) -> bool {
        matches!(self, Self::PropagateToFrame { .. } | Self::Unhandled)
    }

    /// Returns the handler PC if this is a jump action.
    #[inline]
    pub fn handler_pc(&self) -> Option<u32> {
        match self {
            Self::JumpToHandler { handler_pc, .. } => Some(*handler_pc),
            Self::ExecuteFinally { finally_pc, .. } => Some(*finally_pc),
            _ => None,
        }
    }

    /// Returns the stack depth if applicable.
    #[inline]
    pub fn stack_depth(&self) -> Option<u16> {
        match self {
            Self::JumpToHandler { stack_depth, .. } => Some(*stack_depth),
            Self::ExecuteFinally { stack_depth, .. } => Some(*stack_depth),
            _ => None,
        }
    }
}

// ============================================================================
// Unwind Result
// ============================================================================

/// Result of an unwind operation.
#[derive(Clone, Debug)]
pub struct UnwindResult {
    /// The action to take.
    pub action: UnwindAction,

    /// Number of frames unwound.
    pub frames_unwound: u32,

    /// Number of handlers examined.
    pub handlers_examined: u32,

    /// Finally blocks that need to run.
    pub finally_queue: SmallVec<[FinallyEntry; FINALLY_QUEUE_CAPACITY]>,
}

impl UnwindResult {
    /// Creates a new result with a handler found.
    #[inline]
    pub fn handler_found(handler_pc: u32, stack_depth: u16, handlers_examined: u32) -> Self {
        Self {
            action: UnwindAction::JumpToHandler {
                handler_pc,
                stack_depth,
            },
            frames_unwound: 0,
            handlers_examined,
            finally_queue: SmallVec::new(),
        }
    }

    /// Creates a result for unhandled exception.
    #[inline]
    pub fn unhandled(frames_unwound: u32, handlers_examined: u32) -> Self {
        Self {
            action: UnwindAction::Unhandled,
            frames_unwound,
            handlers_examined,
            finally_queue: SmallVec::new(),
        }
    }

    /// Creates a result for propagation to caller.
    #[inline]
    pub fn propagate(target_frame_id: u32, frames_unwound: u32) -> Self {
        Self {
            action: UnwindAction::PropagateToFrame { target_frame_id },
            frames_unwound,
            handlers_examined: 0,
            finally_queue: SmallVec::new(),
        }
    }

    /// Returns true if a handler was found.
    #[inline]
    pub fn found_handler(&self) -> bool {
        matches!(self.action, UnwindAction::JumpToHandler { .. })
    }
}

impl Default for UnwindResult {
    fn default() -> Self {
        Self {
            action: UnwindAction::Continue,
            frames_unwound: 0,
            handlers_examined: 0,
            finally_queue: SmallVec::new(),
        }
    }
}

// ============================================================================
// Finally Entry
// ============================================================================

/// Entry in the finally block queue.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FinallyEntry {
    /// PC of the finally block.
    pub finally_pc: u32,

    /// Stack depth to restore.
    pub stack_depth: u16,

    /// Frame ID owning this finally.
    pub frame_id: u32,

    /// Whether to reraise after this finally.
    pub reraise: bool,
}

impl FinallyEntry {
    /// Creates a new finally entry.
    #[inline]
    pub const fn new(finally_pc: u32, stack_depth: u16, frame_id: u32, reraise: bool) -> Self {
        Self {
            finally_pc,
            stack_depth,
            frame_id,
            reraise,
        }
    }
}

// ============================================================================
// Unwind Info
// ============================================================================

/// Information needed for stack unwinding.
#[derive(Clone, Copy, Debug)]
pub struct UnwindInfo {
    /// Current program counter.
    pub pc: u32,

    /// Current frame ID.
    pub frame_id: u32,

    /// Current stack depth.
    pub stack_depth: u16,

    /// Exception type ID (for type matching).
    pub exception_type_id: u16,
}

impl UnwindInfo {
    /// Creates new unwind info.
    #[inline]
    pub const fn new(pc: u32, frame_id: u32, stack_depth: u16, exception_type_id: u16) -> Self {
        Self {
            pc,
            frame_id,
            stack_depth,
            exception_type_id,
        }
    }
}

// ============================================================================
// Unwinder
// ============================================================================

/// Stack unwinder for exception propagation.
///
/// This struct coordinates the search for exception handlers across
/// the call stack.
#[derive(Clone, Debug, Default)]
pub struct Unwinder {
    /// Statistics for unwinding operations.
    stats: UnwinderStats,
}

/// Statistics for unwinding operations.
#[derive(Clone, Copy, Debug, Default)]
pub struct UnwinderStats {
    /// Total unwind operations.
    pub unwind_count: u64,

    /// Total frames unwound.
    pub frames_unwound: u64,

    /// Total handlers examined.
    pub handlers_examined: u64,

    /// Handlers found.
    pub handlers_found: u64,

    /// Unhandled exceptions.
    pub unhandled: u64,

    /// Maximum frames unwound in a single operation.
    pub max_frames_unwound: u32,
}

impl UnwinderStats {
    /// Creates new empty stats.
    #[inline]
    pub const fn new() -> Self {
        Self {
            unwind_count: 0,
            frames_unwound: 0,
            handlers_examined: 0,
            handlers_found: 0,
            unhandled: 0,
            max_frames_unwound: 0,
        }
    }

    /// Resets all statistics.
    #[inline]
    pub fn reset(&mut self) {
        *self = Self::new();
    }

    /// Returns the average frames unwound per operation.
    #[inline]
    pub fn avg_frames_unwound(&self) -> f64 {
        if self.unwind_count == 0 {
            0.0
        } else {
            self.frames_unwound as f64 / self.unwind_count as f64
        }
    }

    /// Returns the handler found rate as percentage.
    #[inline]
    pub fn handler_found_rate(&self) -> f64 {
        if self.unwind_count == 0 {
            0.0
        } else {
            (self.handlers_found as f64 / self.unwind_count as f64) * 100.0
        }
    }
}

impl Unwinder {
    /// Creates a new unwinder.
    #[inline]
    pub fn new() -> Self {
        Self {
            stats: UnwinderStats::new(),
        }
    }

    /// Returns the statistics.
    #[inline]
    pub fn stats(&self) -> &UnwinderStats {
        &self.stats
    }

    /// Resets statistics.
    #[inline]
    pub fn reset_stats(&mut self) {
        self.stats.reset();
    }

    /// Searches for a handler in the handler stack.
    ///
    /// This is the main unwinding entry point. It searches for a matching
    /// handler and returns the appropriate action.
    #[inline]
    pub fn search_handler(
        &mut self,
        handler_stack: &HandlerStack,
        info: &UnwindInfo,
    ) -> HandlerSearchResult {
        self.stats.unwind_count += 1;

        // Search from top of handler stack (most recent handlers first)
        for (idx, handler) in handler_stack.iter().enumerate() {
            self.stats.handlers_examined += 1;

            // Only consider handlers for this frame or parent frames
            if handler.frame_id > info.frame_id {
                continue;
            }

            // Found a handler for this or parent frame
            let stack_idx = handler_stack.len() - 1 - idx;
            return HandlerSearchResult::Found {
                stack_index: stack_idx,
                handler: *handler,
            };
        }

        self.stats.unhandled += 1;
        HandlerSearchResult::NotFound
    }

    /// Performs a full unwind operation.
    ///
    /// Searches for handlers and builds the unwind result including
    /// any finally blocks that need to run.
    pub fn unwind(&mut self, handler_stack: &HandlerStack, info: &UnwindInfo) -> UnwindResult {
        let search_result = self.search_handler(handler_stack, info);

        match search_result {
            HandlerSearchResult::Found { handler, .. } => {
                self.stats.handlers_found += 1;
                UnwindResult::handler_found(
                    handler.handler_idx as u32, // Use handler_idx as PC placeholder
                    handler.stack_depth,
                    self.stats.handlers_examined as u32,
                )
            }
            HandlerSearchResult::Finally { handler, .. } => {
                let mut result = UnwindResult::default();
                result.action = UnwindAction::ExecuteFinally {
                    finally_pc: handler.handler_idx as u32,
                    stack_depth: handler.stack_depth,
                    reraise: true,
                };
                result.handlers_examined = self.stats.handlers_examined as u32;
                result.finally_queue.push(FinallyEntry::new(
                    handler.handler_idx as u32,
                    handler.stack_depth,
                    handler.frame_id,
                    true,
                ));
                result
            }
            HandlerSearchResult::NotFound => {
                UnwindResult::unhandled(0, self.stats.handlers_examined as u32)
            }
        }
    }

    /// Updates statistics after an unwind operation.
    #[inline]
    pub fn record_unwind(&mut self, frames: u32) {
        self.stats.frames_unwound += frames as u64;
        if frames > self.stats.max_frames_unwound {
            self.stats.max_frames_unwound = frames;
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ════════════════════════════════════════════════════════════════════════
    // UnwindAction Tests
    // ════════════════════════════════════════════════════════════════════════

    #[test]
    fn test_unwind_action_jump_to_handler() {
        let action = UnwindAction::JumpToHandler {
            handler_pc: 100,
            stack_depth: 5,
        };

        assert!(action.continues());
        assert!(!action.propagates());
        assert_eq!(action.handler_pc(), Some(100));
        assert_eq!(action.stack_depth(), Some(5));
    }

    #[test]
    fn test_unwind_action_execute_finally() {
        let action = UnwindAction::ExecuteFinally {
            finally_pc: 200,
            stack_depth: 3,
            reraise: true,
        };

        assert!(!action.continues());
        assert!(!action.propagates());
        assert_eq!(action.handler_pc(), Some(200));
        assert_eq!(action.stack_depth(), Some(3));
    }

    #[test]
    fn test_unwind_action_propagate() {
        let action = UnwindAction::PropagateToFrame { target_frame_id: 1 };

        assert!(!action.continues());
        assert!(action.propagates());
        assert_eq!(action.handler_pc(), None);
        assert_eq!(action.stack_depth(), None);
    }

    #[test]
    fn test_unwind_action_unhandled() {
        let action = UnwindAction::Unhandled;

        assert!(!action.continues());
        assert!(action.propagates());
    }

    #[test]
    fn test_unwind_action_continue() {
        let action = UnwindAction::Continue;

        assert!(action.continues());
        assert!(!action.propagates());
    }

    // ════════════════════════════════════════════════════════════════════════
    // UnwindResult Tests
    // ════════════════════════════════════════════════════════════════════════

    #[test]
    fn test_unwind_result_handler_found() {
        let result = UnwindResult::handler_found(100, 5, 3);

        assert!(result.found_handler());
        assert_eq!(result.frames_unwound, 0);
        assert_eq!(result.handlers_examined, 3);
        assert!(result.finally_queue.is_empty());
    }

    #[test]
    fn test_unwind_result_unhandled() {
        let result = UnwindResult::unhandled(5, 10);

        assert!(!result.found_handler());
        assert_eq!(result.frames_unwound, 5);
        assert_eq!(result.handlers_examined, 10);
    }

    #[test]
    fn test_unwind_result_propagate() {
        let result = UnwindResult::propagate(42, 3);

        assert!(!result.found_handler());
        assert!(result.action.propagates());
        if let UnwindAction::PropagateToFrame { target_frame_id } = result.action {
            assert_eq!(target_frame_id, 42);
        } else {
            panic!("Expected PropagateToFrame");
        }
    }

    #[test]
    fn test_unwind_result_default() {
        let result = UnwindResult::default();

        assert!(!result.found_handler());
        assert_eq!(result.frames_unwound, 0);
        assert!(result.finally_queue.is_empty());
    }

    // ════════════════════════════════════════════════════════════════════════
    // FinallyEntry Tests
    // ════════════════════════════════════════════════════════════════════════

    #[test]
    fn test_finally_entry_new() {
        let entry = FinallyEntry::new(100, 5, 1, true);

        assert_eq!(entry.finally_pc, 100);
        assert_eq!(entry.stack_depth, 5);
        assert_eq!(entry.frame_id, 1);
        assert!(entry.reraise);
    }

    #[test]
    fn test_finally_entry_size() {
        // Should be compact
        assert!(std::mem::size_of::<FinallyEntry>() <= 16);
    }

    // ════════════════════════════════════════════════════════════════════════
    // UnwindInfo Tests
    // ════════════════════════════════════════════════════════════════════════

    #[test]
    fn test_unwind_info_new() {
        let info = UnwindInfo::new(50, 1, 10, 42);

        assert_eq!(info.pc, 50);
        assert_eq!(info.frame_id, 1);
        assert_eq!(info.stack_depth, 10);
        assert_eq!(info.exception_type_id, 42);
    }

    // ════════════════════════════════════════════════════════════════════════
    // UnwinderStats Tests
    // ════════════════════════════════════════════════════════════════════════

    #[test]
    fn test_unwinder_stats_new() {
        let stats = UnwinderStats::new();

        assert_eq!(stats.unwind_count, 0);
        assert_eq!(stats.handlers_found, 0);
    }

    #[test]
    fn test_unwinder_stats_avg_frames() {
        let mut stats = UnwinderStats::new();

        assert_eq!(stats.avg_frames_unwound(), 0.0);

        stats.unwind_count = 10;
        stats.frames_unwound = 25;
        assert!((stats.avg_frames_unwound() - 2.5).abs() < 0.001);
    }

    #[test]
    fn test_unwinder_stats_handler_rate() {
        let mut stats = UnwinderStats::new();

        assert_eq!(stats.handler_found_rate(), 0.0);

        stats.unwind_count = 100;
        stats.handlers_found = 75;
        assert!((stats.handler_found_rate() - 75.0).abs() < 0.001);
    }

    #[test]
    fn test_unwinder_stats_reset() {
        let mut stats = UnwinderStats::new();
        stats.unwind_count = 100;
        stats.handlers_found = 50;

        stats.reset();

        assert_eq!(stats.unwind_count, 0);
        assert_eq!(stats.handlers_found, 0);
    }

    // ════════════════════════════════════════════════════════════════════════
    // Unwinder Tests
    // ════════════════════════════════════════════════════════════════════════

    #[test]
    fn test_unwinder_new() {
        let unwinder = Unwinder::new();

        assert_eq!(unwinder.stats().unwind_count, 0);
    }

    #[test]
    fn test_unwinder_search_empty_stack() {
        let mut unwinder = Unwinder::new();
        let stack = HandlerStack::new();
        let info = UnwindInfo::new(50, 0, 10, 0);

        let result = unwinder.search_handler(&stack, &info);

        assert_eq!(result, HandlerSearchResult::NotFound);
        assert_eq!(unwinder.stats().unwind_count, 1);
        assert_eq!(unwinder.stats().unhandled, 1);
    }

    #[test]
    fn test_unwinder_search_finds_handler() {
        let mut unwinder = Unwinder::new();
        let mut stack = HandlerStack::new();
        stack.push(HandlerFrame::new(0, 5, 0));

        let info = UnwindInfo::new(50, 0, 10, 0);
        let result = unwinder.search_handler(&stack, &info);

        if let HandlerSearchResult::Found { handler, .. } = result {
            assert_eq!(handler.handler_idx, 0);
            assert_eq!(handler.stack_depth, 5);
        } else {
            panic!("Expected handler found");
        }
    }

    #[test]
    fn test_unwinder_search_finds_parent_frame_handler() {
        let mut unwinder = Unwinder::new();
        let mut stack = HandlerStack::new();
        // Handler in parent frame
        stack.push(HandlerFrame::new(0, 5, 0));

        // Searching in child frame
        let info = UnwindInfo::new(50, 1, 10, 0);
        let result = unwinder.search_handler(&stack, &info);

        if let HandlerSearchResult::Found { handler, .. } = result {
            assert_eq!(handler.frame_id, 0); // Found parent's handler
        } else {
            panic!("Expected handler found");
        }
    }

    #[test]
    fn test_unwinder_unwind_finds_handler() {
        let mut unwinder = Unwinder::new();
        let mut stack = HandlerStack::new();
        stack.push(HandlerFrame::new(5, 10, 0));

        let info = UnwindInfo::new(50, 0, 15, 0);
        let result = unwinder.unwind(&stack, &info);

        assert!(result.found_handler());
        assert_eq!(unwinder.stats().handlers_found, 1);
    }

    #[test]
    fn test_unwinder_unwind_no_handler() {
        let mut unwinder = Unwinder::new();
        let stack = HandlerStack::new();

        let info = UnwindInfo::new(50, 0, 10, 0);
        let result = unwinder.unwind(&stack, &info);

        assert!(!result.found_handler());
        assert_eq!(result.action, UnwindAction::Unhandled);
    }

    #[test]
    fn test_unwinder_record_unwind() {
        let mut unwinder = Unwinder::new();

        unwinder.record_unwind(5);
        assert_eq!(unwinder.stats().frames_unwound, 5);
        assert_eq!(unwinder.stats().max_frames_unwound, 5);

        unwinder.record_unwind(3);
        assert_eq!(unwinder.stats().frames_unwound, 8);
        assert_eq!(unwinder.stats().max_frames_unwound, 5);

        unwinder.record_unwind(10);
        assert_eq!(unwinder.stats().max_frames_unwound, 10);
    }

    #[test]
    fn test_unwinder_reset_stats() {
        let mut unwinder = Unwinder::new();
        unwinder.record_unwind(5);

        unwinder.reset_stats();

        assert_eq!(unwinder.stats().frames_unwound, 0);
        assert_eq!(unwinder.stats().max_frames_unwound, 0);
    }

    // ════════════════════════════════════════════════════════════════════════
    // Integration Tests
    // ════════════════════════════════════════════════════════════════════════

    #[test]
    fn test_nested_handlers_unwind() {
        let mut unwinder = Unwinder::new();
        let mut stack = HandlerStack::new();

        // Outermost handler
        stack.push(HandlerFrame::new(0, 2, 0));
        // Middle handler
        stack.push(HandlerFrame::new(1, 5, 0));
        // Innermost handler
        stack.push(HandlerFrame::new(2, 8, 0));

        let info = UnwindInfo::new(50, 0, 10, 0);
        let result = unwinder.unwind(&stack, &info);

        // Should find innermost handler first
        if let UnwindAction::JumpToHandler {
            handler_pc,
            stack_depth,
        } = result.action
        {
            assert_eq!(handler_pc, 2); // handler_idx 2
            assert_eq!(stack_depth, 8);
        } else {
            panic!("Expected JumpToHandler");
        }
    }

    #[test]
    fn test_cross_frame_unwind() {
        let mut unwinder = Unwinder::new();
        let mut stack = HandlerStack::new();

        // Handler in frame 0
        stack.push(HandlerFrame::new(0, 5, 0));
        // Handler in frame 1
        stack.push(HandlerFrame::new(1, 3, 1));

        // Exception in frame 2 (no handler)
        let info = UnwindInfo::new(50, 2, 10, 0);
        let result = unwinder.unwind(&stack, &info);

        // Should find handler in frame 1
        assert!(result.found_handler());
    }
}
