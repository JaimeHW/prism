//! Concurrent marker with work-stealing deque.
//!
//! Orchestrates multi-threaded marking using:
//! - `AtomicMarkBitmap` for O(1) concurrent mark checks
//! - Chase-Lev work-stealing deques for load balancing
//! - `SatbQueue` for processing SATB-captured old values
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────────────┐
//! │  Concurrent Marking System                                              │
//! │                                                                         │
//! │  Master Thread                                                          │
//! │  ┌──────────────────────────────────────┐                              │
//! │  │ 1. Scan roots → initial worklist      │                              │
//! │  │ 2. Spawn marker threads               │                              │
//! │  │ 3. Wait for termination               │                              │
//! │  │ 4. Remark: drain SATB + re-scan roots │                              │
//! │  │ 5. Return statistics                  │                              │
//! │  └──────────────────────────────────────┘                              │
//! │                                                                         │
//! │  Marker Thread N                                                        │
//! │  ┌──────────────────────────────────────┐                              │
//! │  │ loop {                                │                              │
//! │  │   pop from local deque                │                              │
//! │  │   if empty → steal from others        │                              │
//! │  │   if nothing to steal → done          │                              │
//! │  │   mark children → push to local       │                              │
//! │  │ }                                     │                              │
//! │  └──────────────────────────────────────┘                              │
//! │                                                                         │
//! │  Shared State                                                           │
//! │  ┌────────────────┐  ┌────────────────┐  ┌─────────────────┐          │
//! │  │AtomicMarkBitmap│  │   SatbQueue    │  │  Termination   │          │
//! │  │ (mark bits)    │  │ (SATB buffers) │  │  (AtomicBool)  │          │
//! │  └────────────────┘  └────────────────┘  └─────────────────┘          │
//! └─────────────────────────────────────────────────────────────────────────┘
//! ```

use crate::barrier::satb_buffer::SatbQueue;
use crate::collector::mark_bitmap::AtomicMarkBitmap;
use crossbeam_deque::{Injector, Stealer, Worker};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;

// =============================================================================
// MarkPtr — Send-safe raw pointer wrapper
// =============================================================================

/// A raw pointer wrapper that implements `Send` for use in crossbeam deques.
///
/// # Safety
///
/// This is safe because:
/// - The GC owns all objects in the heap
/// - During concurrent marking, objects are not moved or deallocated
/// - The pointer is only used for address-based bitmap lookups and tracing
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(transparent)]
pub struct MarkPtr(*const ());

impl MarkPtr {
    /// Create a new MarkPtr from a raw pointer.
    #[inline]
    pub fn new(ptr: *const ()) -> Self {
        Self(ptr)
    }

    /// Get the raw pointer.
    #[inline]
    pub fn as_ptr(self) -> *const () {
        self.0
    }

    /// Get the address as usize.
    #[inline]
    pub fn addr(self) -> usize {
        self.0 as usize
    }
}

// SAFETY: MarkPtr is used exclusively during concurrent marking phases.
// During these phases, the GC guarantees that the referenced objects are
// not moved or deallocated (they are pinned in old-gen/LOS). The pointer
// is used only for:
// 1. Address-based bitmap lookups (AtomicMarkBitmap::mark/is_marked)
// 2. Passing to ObjectTracer::trace_object for child discovery
// Both operations are safe to perform from any thread.
unsafe impl Send for MarkPtr {}

// =============================================================================
// Configuration
// =============================================================================

/// Configuration for the concurrent marker.
#[derive(Debug, Clone)]
pub struct ConcurrentMarkerConfig {
    /// Number of marker threads (0 = single-threaded marking).
    pub marker_threads: usize,
    /// Maximum items to process before checking for termination.
    pub work_chunk_size: usize,
    /// Minimum items to trigger work distribution to other threads.
    pub distribute_threshold: usize,
}

impl Default for ConcurrentMarkerConfig {
    fn default() -> Self {
        Self {
            marker_threads: 2,
            work_chunk_size: 256,
            distribute_threshold: 64,
        }
    }
}

// =============================================================================
// Marking Statistics
// =============================================================================

/// Statistics from a concurrent marking phase.
#[derive(Debug, Default)]
pub struct MarkingStats {
    /// Total objects marked across all threads.
    pub objects_marked: AtomicUsize,
    /// Total objects traced (children scanned).
    pub objects_traced: AtomicUsize,
    /// Total SATB entries processed during remark.
    pub satb_entries_processed: AtomicUsize,
    /// Number of steal attempts.
    pub steal_attempts: AtomicUsize,
    /// Number of successful steals.
    pub steal_successes: AtomicUsize,
}

impl MarkingStats {
    /// Create new zeroed stats.
    pub fn new() -> Self {
        Self::default()
    }

    /// Record a mark.
    #[inline]
    pub fn record_mark(&self) {
        self.objects_marked.fetch_add(1, Ordering::Relaxed);
    }

    /// Record a trace.
    #[inline]
    pub fn record_trace(&self) {
        self.objects_traced.fetch_add(1, Ordering::Relaxed);
    }

    /// Record SATB processing.
    #[inline]
    pub fn record_satb(&self, count: usize) {
        self.satb_entries_processed
            .fetch_add(count, Ordering::Relaxed);
    }

    /// Record a steal attempt.
    #[inline]
    pub fn record_steal_attempt(&self) {
        self.steal_attempts.fetch_add(1, Ordering::Relaxed);
    }

    /// Record a successful steal.
    #[inline]
    pub fn record_steal_success(&self) {
        self.steal_successes.fetch_add(1, Ordering::Relaxed);
    }

    /// Get snapshot of current stats.
    pub fn snapshot(&self) -> MarkingStatsSnapshot {
        MarkingStatsSnapshot {
            objects_marked: self.objects_marked.load(Ordering::Relaxed),
            objects_traced: self.objects_traced.load(Ordering::Relaxed),
            satb_entries_processed: self.satb_entries_processed.load(Ordering::Relaxed),
            steal_attempts: self.steal_attempts.load(Ordering::Relaxed),
            steal_successes: self.steal_successes.load(Ordering::Relaxed),
        }
    }
}

/// Non-atomic snapshot of marking stats.
#[derive(Debug, Clone)]
pub struct MarkingStatsSnapshot {
    /// Objects marked.
    pub objects_marked: usize,
    /// Objects traced.
    pub objects_traced: usize,
    /// SATB entries processed.
    pub satb_entries_processed: usize,
    /// Steal attempts.
    pub steal_attempts: usize,
    /// Successful steals.
    pub steal_successes: usize,
}

// =============================================================================
// MarkingWorklist
// =============================================================================

/// Work-stealing worklist for concurrent marking.
///
/// Uses crossbeam's Chase-Lev deque:
/// - Each marker thread has a local `Worker<MarkPtr>` deque
/// - A global `Injector<MarkPtr>` for initial work distribution
/// - Threads steal from each other when their local deque is empty
pub struct MarkingWorklist {
    /// Global injector for distributing work across all threads.
    injector: Injector<MarkPtr>,
    /// Stealers for each worker thread.
    stealers: Vec<Stealer<MarkPtr>>,
}

impl MarkingWorklist {
    /// Create a new marking worklist with the specified number of workers.
    pub fn new(num_workers: usize) -> (Self, Vec<Worker<MarkPtr>>) {
        let injector = Injector::new();
        let mut workers = Vec::with_capacity(num_workers);
        let mut stealers = Vec::with_capacity(num_workers);

        for _ in 0..num_workers {
            let worker = Worker::new_fifo();
            stealers.push(worker.stealer());
            workers.push(worker);
        }

        (Self { injector, stealers }, workers)
    }

    /// Inject work into the global queue.
    ///
    /// Used for root scanning results and SATB remark items.
    #[inline]
    pub fn inject(&self, ptr: MarkPtr) {
        self.injector.push(ptr);
    }

    /// Inject a raw pointer into the global queue.
    #[inline]
    pub fn inject_raw(&self, ptr: *const ()) {
        self.injector.push(MarkPtr::new(ptr));
    }

    /// Inject multiple items into the global queue.
    pub fn inject_batch(&self, ptrs: &[MarkPtr]) {
        for &ptr in ptrs {
            self.injector.push(ptr);
        }
    }

    /// Try to pop work from the global injector or steal from another worker.
    ///
    /// This is the core work-stealing logic:
    /// 1. Try the global injector first
    /// 2. If empty, try to steal from each worker
    pub fn steal(&self, stats: &MarkingStats) -> Option<MarkPtr> {
        // Try global injector
        loop {
            match self.injector.steal() {
                crossbeam_deque::Steal::Success(item) => return Some(item),
                crossbeam_deque::Steal::Empty => break,
                crossbeam_deque::Steal::Retry => continue,
            }
        }

        // Try stealing from other workers
        for stealer in &self.stealers {
            stats.record_steal_attempt();
            loop {
                match stealer.steal() {
                    crossbeam_deque::Steal::Success(item) => {
                        stats.record_steal_success();
                        return Some(item);
                    }
                    crossbeam_deque::Steal::Empty => break,
                    crossbeam_deque::Steal::Retry => continue,
                }
            }
        }

        None
    }

    /// Check if the global injector is empty.
    pub fn is_injector_empty(&self) -> bool {
        self.injector.is_empty()
    }
}

// =============================================================================
// ConcurrentMarker
// =============================================================================

/// Orchestrates concurrent marking.
///
/// The concurrent marker manages the lifecycle of a concurrent marking phase:
/// 1. **Initial marking**: Roots are scanned (STW pause) and pushed to worklist
/// 2. **Concurrent marking**: Marker threads drain the worklist concurrently
/// 3. **Remark**: SATB buffers are drained and re-processed (brief STW pause)
pub struct ConcurrentMarker {
    /// Configuration.
    config: ConcurrentMarkerConfig,
    /// Mark bitmap for the old generation.
    bitmap: Arc<AtomicMarkBitmap>,
    /// SATB queue for barrier-captured old values.
    satb_queue: Arc<SatbQueue>,
    /// Global statistics.
    stats: Arc<MarkingStats>,
    /// Termination flag.
    terminated: Arc<AtomicBool>,
}

impl ConcurrentMarker {
    /// Create a new concurrent marker.
    ///
    /// # Arguments
    ///
    /// - `config`: Marker configuration
    /// - `bitmap`: Shared mark bitmap for the heap region
    /// - `satb_queue`: Global SATB queue for barrier-captured old values
    pub fn new(
        config: ConcurrentMarkerConfig,
        bitmap: Arc<AtomicMarkBitmap>,
        satb_queue: Arc<SatbQueue>,
    ) -> Self {
        Self {
            config,
            bitmap,
            satb_queue,
            stats: Arc::new(MarkingStats::new()),
            terminated: Arc::new(AtomicBool::new(false)),
        }
    }

    /// Get the mark bitmap.
    #[inline]
    pub fn bitmap(&self) -> &Arc<AtomicMarkBitmap> {
        &self.bitmap
    }

    /// Get the SATB queue.
    #[inline]
    pub fn satb_queue(&self) -> &Arc<SatbQueue> {
        &self.satb_queue
    }

    /// Get current marking statistics.
    pub fn stats(&self) -> MarkingStatsSnapshot {
        self.stats.snapshot()
    }

    /// Mark an address atomically.
    ///
    /// Returns `true` if the object was newly marked (first time).
    #[inline]
    pub fn mark(&self, addr: usize) -> bool {
        match self.bitmap.mark(addr) {
            Some(true) => {
                self.stats.record_mark();
                true
            }
            _ => false,
        }
    }

    /// Check if an address is marked.
    #[inline]
    pub fn is_marked(&self, addr: usize) -> bool {
        self.bitmap.is_marked(addr)
    }

    /// Process a chunk of work from a local deque.
    ///
    /// This is the core marking loop for each worker thread.
    /// The `trace_fn` callback is called for each object pointer to
    /// discover its children. It receives the object address and a
    /// push function to enqueue newly-discovered children.
    ///
    /// Returns the number of objects traced.
    pub fn process_local<F>(
        &self,
        worker: &Worker<MarkPtr>,
        worklist: &MarkingWorklist,
        mut trace_fn: F,
    ) -> usize
    where
        F: FnMut(MarkPtr, &dyn Fn(MarkPtr)),
    {
        let mut traced = 0;

        loop {
            // Try local deque first
            let ptr = worker.pop().or_else(|| worklist.steal(&self.stats));

            match ptr {
                Some(obj_ptr) => {
                    traced += 1;
                    self.stats.record_trace();

                    // Trace object's children — callback marks and pushes new work
                    let push_fn = |child_ptr: MarkPtr| {
                        if self.mark(child_ptr.addr()) {
                            worker.push(child_ptr);
                        }
                    };
                    trace_fn(obj_ptr, &push_fn);
                }
                None => {
                    // No more work — this worker is done
                    break;
                }
            }
        }

        traced
    }

    /// Process SATB entries accumulated during concurrent marking.
    ///
    /// Called during the remark pause to ensure all objects that were
    /// reachable at the snapshot point are marked.
    pub fn process_satb(&self) -> usize {
        let entries = self.satb_queue.drain_all();
        let count = entries.len();

        for ptr in &entries {
            let addr = *ptr as usize;
            self.mark(addr);
        }

        self.stats.record_satb(count);
        count
    }

    /// Clear all marks (prepare for next cycle).
    pub fn clear(&self) {
        self.bitmap.clear_all();
        self.terminated.store(false, Ordering::Release);
    }

    /// Mark the concurrent marking phase as terminated.
    pub fn terminate(&self) {
        self.terminated.store(true, Ordering::Release);
    }

    /// Check if terminated.
    pub fn is_terminated(&self) -> bool {
        self.terminated.load(Ordering::Acquire)
    }

    /// Get the number of marker threads.
    pub fn num_threads(&self) -> usize {
        self.config.marker_threads
    }

    /// Get work chunk size.
    pub fn work_chunk_size(&self) -> usize {
        self.config.work_chunk_size
    }
}

// Safety: ConcurrentMarker is designed for multi-threaded use.
// - AtomicMarkBitmap is lock-free and thread-safe
// - SatbQueue uses Mutex internally
// - Stats use atomic counters
// - terminated is AtomicBool
unsafe impl Send for ConcurrentMarker {}
unsafe impl Sync for ConcurrentMarker {}
