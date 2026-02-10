//! Snapshot-At-The-Beginning (SATB) write barrier buffers.
//!
//! During concurrent marking, the mutator may overwrite references to objects
//! that the marker hasn't yet visited. The SATB barrier captures the **old**
//! value on each pointer store, ensuring the marker sees all objects that were
//! reachable at the snapshot point (start of marking).
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────────────┐
//! │  SATB System                                                            │
//! │                                                                         │
//! │  Thread 1           Thread 2           Thread N                         │
//! │  ┌───────────┐     ┌───────────┐     ┌───────────┐                    │
//! │  │ SatbBuffer │     │ SatbBuffer │     │ SatbBuffer │                    │
//! │  │ [old vals] │     │ [old vals] │     │ [old vals] │                    │
//! │  └─────┬─────┘     └─────┬─────┘     └─────┬─────┘                    │
//! │        │ flush            │ flush            │ flush                    │
//! │        └──────────────────┼──────────────────┘                         │
//! │                           ▼                                             │
//! │                  ┌────────────────┐                                    │
//! │                  │   SatbQueue    │  (global, Mutex-protected)          │
//! │                  │ [buffers...]   │                                     │
//! │                  └────────┬───────┘                                    │
//! │                           │ drain                                      │
//! │                           ▼                                             │
//! │                  ┌────────────────┐                                    │
//! │                  │ ConcurrentMark │  (re-grays old values)             │
//! │                  └────────────────┘                                    │
//! │                                                                         │
//! │  SatbMarkingState: Idle ──► Marking ──► Remarking ──► Idle             │
//! └─────────────────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Performance
//!
//! - **Barrier fast path**: single atomic load + comparison (branching on `Idle`)
//! - **Buffer**: stack-allocated array, fixed capacity → no allocation on push
//! - **Flush**: amortized over `BUFFER_CAPACITY` stores

use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::Mutex;

// =============================================================================
// Constants
// =============================================================================

/// Capacity of each thread-local SATB buffer.
///
/// Chosen to balance memory usage per thread against flush frequency.
/// 512 entries × 8 bytes per pointer = 4KB per buffer.
const BUFFER_CAPACITY: usize = 512;

// =============================================================================
// SatbMarkingState
// =============================================================================

/// Global marking state for SATB barrier activation.
///
/// Controls whether the SATB write barrier captures old values.
/// Uses an `AtomicU8` for wait-free state checks.
///
/// State transitions:
/// ```text
/// Idle ──start_marking──► Marking ──start_remark──► Remarking ──finish──► Idle
/// ```
pub struct SatbMarkingState {
    state: AtomicU8,
}

/// Marking phase identifiers.
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MarkingPhase {
    /// No concurrent marking in progress. Barrier is inactive.
    Idle = 0,
    /// Concurrent marking in progress. Barrier captures old values.
    Marking = 1,
    /// Final remark phase. Barrier still active, draining remaining buffers.
    Remarking = 2,
}

impl SatbMarkingState {
    /// Create a new marking state (initially Idle).
    #[inline]
    pub fn new() -> Self {
        Self {
            state: AtomicU8::new(MarkingPhase::Idle as u8),
        }
    }

    /// Check if the SATB barrier should be active.
    ///
    /// This is the hot path for the write barrier — must be as fast as possible.
    /// Single atomic load with `Relaxed` ordering (sufficient for a boolean check).
    #[inline(always)]
    pub fn is_active(&self) -> bool {
        self.state.load(Ordering::Relaxed) != MarkingPhase::Idle as u8
    }

    /// Get the current marking phase.
    #[inline]
    pub fn phase(&self) -> MarkingPhase {
        match self.state.load(Ordering::Acquire) {
            0 => MarkingPhase::Idle,
            1 => MarkingPhase::Marking,
            2 => MarkingPhase::Remarking,
            _ => unreachable!("Invalid marking phase"),
        }
    }

    /// Transition to marking phase.
    ///
    /// # Panics
    ///
    /// Panics if not currently Idle.
    pub fn start_marking(&self) {
        let prev = self.state.compare_exchange(
            MarkingPhase::Idle as u8,
            MarkingPhase::Marking as u8,
            Ordering::AcqRel,
            Ordering::Acquire,
        );
        assert_eq!(
            prev,
            Ok(MarkingPhase::Idle as u8),
            "Can only start marking from Idle state"
        );
    }

    /// Transition to remark phase.
    ///
    /// # Panics
    ///
    /// Panics if not currently Marking.
    pub fn start_remark(&self) {
        let prev = self.state.compare_exchange(
            MarkingPhase::Marking as u8,
            MarkingPhase::Remarking as u8,
            Ordering::AcqRel,
            Ordering::Acquire,
        );
        assert_eq!(
            prev,
            Ok(MarkingPhase::Marking as u8),
            "Can only start remark from Marking state"
        );
    }

    /// Transition back to Idle.
    ///
    /// Callable from either Marking or Remarking state.
    pub fn finish(&self) {
        self.state
            .store(MarkingPhase::Idle as u8, Ordering::Release);
    }

    /// Reset to idle (unconditional, for error recovery).
    pub fn reset(&self) {
        self.state
            .store(MarkingPhase::Idle as u8, Ordering::Release);
    }
}

impl Default for SatbMarkingState {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// SatbBuffer
// =============================================================================

/// Thread-local SATB buffer for capturing old pointer values.
///
/// Each mutator thread maintains one of these. When a pointer store occurs
/// and the SATB barrier is active, the old value is pushed into this buffer.
/// When the buffer is full, it is flushed to the global `SatbQueue`.
///
/// # Memory Layout
///
/// Fixed-size array on the stack/heap — no dynamic allocation on push.
/// 512 entries × 8 bytes = 4KB per buffer.
pub struct SatbBuffer {
    /// Buffer storage for old pointer values.
    entries: Vec<*const ()>,
}

impl SatbBuffer {
    /// Create a new empty SATB buffer.
    #[inline]
    pub fn new() -> Self {
        Self {
            entries: Vec::with_capacity(BUFFER_CAPACITY),
        }
    }

    /// Record an old pointer value.
    ///
    /// If the buffer is full after this push, returns `true` to signal
    /// that the caller should flush to the global queue.
    ///
    /// # Performance
    ///
    /// Single push + length check. No allocation (pre-allocated capacity).
    #[inline]
    pub fn push(&mut self, old_ptr: *const ()) -> bool {
        self.entries.push(old_ptr);
        self.entries.len() >= BUFFER_CAPACITY
    }

    /// Check if the buffer is full.
    #[inline]
    pub fn is_full(&self) -> bool {
        self.entries.len() >= BUFFER_CAPACITY
    }

    /// Check if the buffer is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Get the number of entries in the buffer.
    #[inline]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Drain all entries from the buffer, returning them.
    ///
    /// Resets the buffer to empty without deallocating the underlying storage.
    #[inline]
    pub fn drain(&mut self) -> Vec<*const ()> {
        let mut drained = Vec::with_capacity(self.entries.len());
        drained.append(&mut self.entries);
        drained
    }

    /// Flush the buffer into a `SatbQueue`.
    ///
    /// Moves all entries to the global queue and resets the buffer.
    #[inline]
    pub fn flush(&mut self, queue: &SatbQueue) {
        if !self.entries.is_empty() {
            let entries = self.drain();
            queue.push_buffer(entries);
        }
    }

    /// Get an iterator over the buffer entries (for testing/debugging).
    pub fn iter(&self) -> impl Iterator<Item = &*const ()> {
        self.entries.iter()
    }

    /// Buffer capacity constant (for testing).
    #[inline]
    pub fn capacity() -> usize {
        BUFFER_CAPACITY
    }
}

impl Default for SatbBuffer {
    fn default() -> Self {
        Self::new()
    }
}

// Safety: SatbBuffer contains raw pointers but is single-threaded (thread-local).
// Only the owning thread accesses its entries.
unsafe impl Send for SatbBuffer {}

// =============================================================================
// SatbQueue
// =============================================================================

/// Global concurrent queue for collecting flushed SATB buffers.
///
/// Marker threads drain this queue to re-gray objects whose references
/// were overwritten by the mutator during concurrent marking.
///
/// Uses a `Mutex<Vec<Vec<*const ()>>>` — the lock is only held briefly
/// during flush/drain operations, and contention is amortized over
/// `BUFFER_CAPACITY` stores.
pub struct SatbQueue {
    /// Collected buffers from all threads.
    buffers: Mutex<Vec<Vec<*const ()>>>,
}

impl SatbQueue {
    /// Create a new empty SATB queue.
    pub fn new() -> Self {
        Self {
            buffers: Mutex::new(Vec::new()),
        }
    }

    /// Push a flushed buffer into the global queue.
    ///
    /// Called when a thread-local `SatbBuffer` is full.
    pub fn push_buffer(&self, buffer: Vec<*const ()>) {
        if !buffer.is_empty() {
            self.buffers.lock().unwrap().push(buffer);
        }
    }

    /// Drain all buffered entries, returning them as a flat list.
    ///
    /// Called by the marker during remark to process all captured old values.
    pub fn drain_all(&self) -> Vec<*const ()> {
        let mut guard = self.buffers.lock().unwrap();
        let total: usize = guard.iter().map(|b| b.len()).sum();
        let mut result = Vec::with_capacity(total);
        for buffer in guard.drain(..) {
            result.extend(buffer);
        }
        result
    }

    /// Drain all buffers without flattening (preserves buffer boundaries).
    ///
    /// More efficient when the marker will process each buffer individually.
    pub fn drain_buffers(&self) -> Vec<Vec<*const ()>> {
        let mut guard = self.buffers.lock().unwrap();
        std::mem::take(&mut *guard)
    }

    /// Check if the queue is empty.
    pub fn is_empty(&self) -> bool {
        self.buffers.lock().unwrap().is_empty()
    }

    /// Get the total number of entries across all buffers.
    pub fn total_entries(&self) -> usize {
        self.buffers.lock().unwrap().iter().map(|b| b.len()).sum()
    }

    /// Get the number of pending buffers.
    pub fn buffer_count(&self) -> usize {
        self.buffers.lock().unwrap().len()
    }
}

impl Default for SatbQueue {
    fn default() -> Self {
        Self::new()
    }
}

// Safety: SatbQueue uses Mutex internally for synchronization.
unsafe impl Send for SatbQueue {}
unsafe impl Sync for SatbQueue {}

// =============================================================================
// SATB Write Barrier Function
// =============================================================================

/// SATB write barrier — captures old pointer value before overwrite.
///
/// This is called **before** a pointer store to an object field.
/// The old value is recorded so the concurrent marker can still
/// discover objects that were live at the snapshot point.
///
/// # Fast Path
///
/// When marking is not active (`SatbMarkingState::Idle`), this reduces
/// to a single atomic load + branch — negligible overhead.
///
/// # Arguments
///
/// - `old_ptr`: The old pointer value being overwritten
/// - `state`: Global marking state
/// - `buffer`: Thread-local SATB buffer
/// - `queue`: Global SATB queue (for flushing)
///
/// # Usage
///
/// ```ignore
/// // Before overwriting a field:
/// let old = obj.field;
/// satb_write_barrier(old, &marking_state, &mut thread_buffer, &global_queue);
/// obj.field = new_value;
/// ```
#[inline(always)]
pub fn satb_write_barrier(
    old_ptr: *const (),
    state: &SatbMarkingState,
    buffer: &mut SatbBuffer,
    queue: &SatbQueue,
) {
    // Fast path: if not marking, do nothing (single atomic load)
    if !state.is_active() {
        return;
    }

    // Filter null pointers — they don't need to be traced
    if old_ptr.is_null() {
        return;
    }

    // Push old value into thread-local buffer
    let needs_flush = buffer.push(old_ptr);

    // If buffer is full, flush to global queue
    if needs_flush {
        buffer.flush(queue);
    }
}

/// Unconditional SATB capture — always records the old value.
///
/// Used in remark phases or when the caller has already checked `is_active()`.
#[inline]
pub fn satb_capture_unconditional(old_ptr: *const (), buffer: &mut SatbBuffer, queue: &SatbQueue) {
    if old_ptr.is_null() {
        return;
    }

    let needs_flush = buffer.push(old_ptr);
    if needs_flush {
        buffer.flush(queue);
    }
}
