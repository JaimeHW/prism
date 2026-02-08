//! Concurrent major garbage collection.
//!
//! This module implements a three-phase concurrent mark-sweep collector
//! that can perform most marking work concurrently with mutator threads:
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────────┐
//! │  THREE-PHASE CONCURRENT MARK-SWEEP                                  │
//! │                                                                     │
//! │  Phase 1: Initial Mark (STW)                                        │
//! │  ┌──────────────────────────────────────────────────────────────┐   │
//! │  │ • Scan roots → inject into work-stealing worklist           │   │
//! │  │ • Activate SATB write barrier                                │   │
//! │  │ • Mark root-reachable objects in AtomicMarkBitmap            │   │
//! │  └──────────────────────────────────────────────────────────────┘   │
//! │                                                                     │
//! │  Phase 2: Concurrent Mark                                           │
//! │  ┌──────────────────────────────────────────────────────────────┐   │
//! │  │ • Worker threads drain worklist via work-stealing            │   │
//! │  │ • Each worker: pop → trace children → mark → push new work  │   │
//! │  │ • Mutator runs in parallel; SATB barrier captures overwrites│   │
//! │  └──────────────────────────────────────────────────────────────┘   │
//! │                                                                     │
//! │  Phase 3: Remark + Sweep (STW)                                      │
//! │  ┌──────────────────────────────────────────────────────────────┐   │
//! │  │ • Drain all SATB buffers → mark captured references         │   │
//! │  │ • Re-scan roots for new references added during marking     │   │
//! │  │ • Deactivate SATB barrier                                    │   │
//! │  │ • Sweep: free unmarked objects in old-space + LOS            │   │
//! │  └──────────────────────────────────────────────────────────────┘   │
//! └─────────────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Design Principles
//!
//! - **AtomicMarkBitmap** replaces `FxHashSet<usize>` for O(1) lock-free marking
//! - **Chase-Lev work-stealing** replaces `VecDeque` for multi-threaded load balancing
//! - **SATB write barrier** ensures no live objects are missed during concurrent marking
//! - **MarkPtr** wrapper provides `Send`-safe raw pointer transfer between threads

use crate::barrier::{SatbMarkingState, SatbQueue};
use crate::collector::concurrent_marker::{
    ConcurrentMarker, ConcurrentMarkerConfig, MarkPtr, MarkingWorklist,
};
use crate::collector::mark_bitmap::AtomicMarkBitmap;
use crate::heap::GcHeap;
use crate::roots::RootSet;
use crate::trace::{ObjectTracer, Tracer};
use std::sync::Arc;
use std::time::{Duration, Instant};

// =============================================================================
// Configuration
// =============================================================================

/// Configuration for the concurrent major collector.
#[derive(Debug, Clone)]
pub struct ConcurrentMajorConfig {
    /// Number of marker worker threads.
    pub marker_threads: usize,
    /// Work chunk size for distribution.
    pub work_chunk_size: usize,
    /// Maximum bitmap coverage in bytes.
    pub bitmap_coverage: usize,
}

impl Default for ConcurrentMajorConfig {
    #[inline]
    fn default() -> Self {
        Self {
            marker_threads: 4,
            work_chunk_size: 64,
            // 4 GB coverage — sufficient for most workloads
            bitmap_coverage: 4 * 1024 * 1024 * 1024,
        }
    }
}

impl ConcurrentMajorConfig {
    /// Create a config for testing with a single thread and small bitmap.
    #[inline]
    pub fn for_testing() -> Self {
        Self {
            marker_threads: 1,
            work_chunk_size: 16,
            bitmap_coverage: 64 * 1024 * 1024, // 64 MB
        }
    }
}

// =============================================================================
// Phase tracking
// =============================================================================

/// Current phase of the concurrent major collection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConcurrentGcPhase {
    /// No collection in progress.
    Idle,
    /// Phase 1: Stop-the-world initial mark.
    InitialMark,
    /// Phase 2: Concurrent marking (mutators running).
    ConcurrentMark,
    /// Phase 3: Stop-the-world remark + sweep.
    Remark,
    /// Sweeping dead objects.
    Sweep,
}

// =============================================================================
// Result
// =============================================================================

/// Result of a concurrent major collection.
#[derive(Debug, Default, Clone)]
pub struct ConcurrentMajorResult {
    /// Bytes freed during sweep.
    pub bytes_freed: usize,
    /// Objects freed during sweep.
    pub objects_freed: usize,
    /// Live bytes after collection.
    pub live_bytes: usize,
    /// Objects marked during initial mark + concurrent mark.
    pub objects_marked: usize,
    /// Objects traced (children scanned).
    pub objects_traced: usize,
    /// SATB entries processed during remark.
    pub satb_entries_processed: usize,
    /// Time spent in initial mark (STW).
    pub initial_mark_duration: Duration,
    /// Time spent in concurrent marking.
    pub concurrent_mark_duration: Duration,
    /// Time spent in remark + sweep (STW).
    pub remark_sweep_duration: Duration,
    /// Total collection duration.
    pub total_duration: Duration,
}

// =============================================================================
// ConcurrentMajorCollector
// =============================================================================

/// Three-phase concurrent mark-sweep collector.
///
/// Orchestrates the `ConcurrentMarker`, `AtomicMarkBitmap`, and SATB
/// infrastructure to perform major collections with minimal STW pauses.
pub struct ConcurrentMajorCollector {
    /// Configuration.
    config: ConcurrentMajorConfig,
    /// Lock-free mark bitmap for O(1) concurrent marking.
    bitmap: Arc<AtomicMarkBitmap>,
    /// SATB marking state (Idle, Marking, Remarking).
    marking_state: Arc<SatbMarkingState>,
    /// Global SATB queue for flushed thread-local buffers.
    satb_queue: Arc<SatbQueue>,
    /// Current GC phase.
    phase: ConcurrentGcPhase,
}

impl ConcurrentMajorCollector {
    /// Create a new concurrent major collector.
    #[inline]
    pub fn new() -> Self {
        Self::with_config(ConcurrentMajorConfig::default())
    }

    /// Create a concurrent major collector with custom configuration.
    pub fn with_config(config: ConcurrentMajorConfig) -> Self {
        let bitmap = Arc::new(AtomicMarkBitmap::new(0, config.bitmap_coverage));
        Self {
            config,
            bitmap,
            marking_state: Arc::new(SatbMarkingState::new()),
            satb_queue: Arc::new(SatbQueue::new()),
            phase: ConcurrentGcPhase::Idle,
        }
    }

    /// Create a collector configured for testing.
    #[inline]
    pub fn for_testing() -> Self {
        Self::with_config(ConcurrentMajorConfig::for_testing())
    }

    // =========================================================================
    // Accessors
    // =========================================================================

    /// Get the SATB marking state (shared with write barriers).
    #[inline]
    pub fn marking_state(&self) -> &Arc<SatbMarkingState> {
        &self.marking_state
    }

    /// Get the SATB queue (shared with thread-local buffers).
    #[inline]
    pub fn satb_queue(&self) -> &Arc<SatbQueue> {
        &self.satb_queue
    }

    /// Get the current GC phase.
    #[inline]
    pub fn phase(&self) -> ConcurrentGcPhase {
        self.phase
    }

    /// Get the mark bitmap (for inspection/testing).
    #[inline]
    pub fn bitmap(&self) -> &Arc<AtomicMarkBitmap> {
        &self.bitmap
    }

    /// Get the configuration.
    #[inline]
    pub fn config(&self) -> &ConcurrentMajorConfig {
        &self.config
    }

    // =========================================================================
    // Full Collection (3-Phase)
    // =========================================================================

    /// Perform a complete concurrent major collection.
    ///
    /// Executes all three phases sequentially:
    /// 1. Initial mark (STW)
    /// 2. Concurrent mark
    /// 3. Remark + sweep (STW)
    pub fn collect<T: ObjectTracer>(
        &mut self,
        heap: &mut GcHeap,
        roots: &RootSet,
        object_tracer: &T,
    ) -> ConcurrentMajorResult {
        let total_start = Instant::now();
        let mut result = ConcurrentMajorResult::default();

        // Phase 1: Initial Mark (STW)
        let initial_start = Instant::now();
        let initial_roots = self.initial_mark(heap, roots);
        result.initial_mark_duration = initial_start.elapsed();

        // Phase 2: Concurrent Mark
        let concurrent_start = Instant::now();
        let mark_stats = self.concurrent_mark(initial_roots, object_tracer);
        result.concurrent_mark_duration = concurrent_start.elapsed();

        result.objects_marked = mark_stats.objects_marked;
        result.objects_traced = mark_stats.objects_traced;

        // Phase 3: Remark + Sweep (STW)
        let remark_start = Instant::now();
        let (satb_processed, sweep_freed, sweep_objects) =
            self.remark_and_sweep(heap, roots, object_tracer);
        result.remark_sweep_duration = remark_start.elapsed();

        result.satb_entries_processed = satb_processed;
        result.bytes_freed = sweep_freed;
        result.objects_freed = sweep_objects;
        result.live_bytes = heap.old_space().usage() + heap.large_objects().usage();
        result.total_duration = total_start.elapsed();

        result
    }

    // =========================================================================
    // Phase 1: Initial Mark (STW)
    // =========================================================================

    /// Phase 1: Scan roots and activate SATB barrier.
    ///
    /// Returns the list of root pointers injected into the worklist.
    /// This must be called during a stop-the-world pause.
    fn initial_mark(&mut self, heap: &GcHeap, roots: &RootSet) -> Vec<MarkPtr> {
        self.phase = ConcurrentGcPhase::InitialMark;

        // Recreate bitmap (clear all bits from previous collection)
        self.bitmap = Arc::new(AtomicMarkBitmap::new(0, self.config.bitmap_coverage));

        // Clear LOS marks
        heap.large_objects().clear_marks();

        // Activate SATB write barrier
        self.marking_state.start_marking();

        // Scan roots — collect all old-gen pointers
        let mut root_tracer = InitialMarkTracer {
            bitmap: &self.bitmap,
            heap,
            roots_discovered: Vec::with_capacity(256),
        };
        roots.trace(&mut root_tracer);

        self.phase = ConcurrentGcPhase::ConcurrentMark;

        root_tracer.roots_discovered
    }

    // =========================================================================
    // Phase 2: Concurrent Mark
    // =========================================================================

    /// Phase 2: Multi-threaded concurrent marking.
    ///
    /// Worker threads drain the worklist using work-stealing deques.
    /// Returns marking statistics.
    fn concurrent_mark<T: ObjectTracer>(
        &mut self,
        initial_roots: Vec<MarkPtr>,
        object_tracer: &T,
    ) -> ConcurrentMarkStats {
        let marker_config = ConcurrentMarkerConfig {
            marker_threads: self.config.marker_threads,
            work_chunk_size: self.config.work_chunk_size,
            ..Default::default()
        };

        let marker = ConcurrentMarker::new(
            marker_config,
            Arc::clone(&self.bitmap),
            Arc::clone(&self.satb_queue),
        );

        // Create worklist and workers
        let num_workers = self.config.marker_threads.max(1);
        let (worklist, workers) = MarkingWorklist::new(num_workers);

        // Inject all root pointers into the global injector
        for root_ptr in &initial_roots {
            worklist.inject(*root_ptr);
        }

        // Process with worker 0 (single-threaded for now, multi-threaded in future).
        // process_local handles the steal loop internally — it tries local deque
        // first, then steals from the global injector + other workers.
        let mut total_traced = 0;
        if !workers.is_empty() {
            let worker = &workers[0];

            total_traced = marker.process_local(worker, &worklist, |obj_ptr, push_fn| {
                // Trace the object's children
                let mut tracer = ConcurrentObjectTracer {
                    bitmap: &self.bitmap,
                    push_fn,
                };
                // SAFETY: obj_ptr points to a valid old-gen object that is pinned
                // during concurrent marking. The bitmap mark prevents double-tracing.
                unsafe {
                    object_tracer.trace_object(obj_ptr.as_ptr(), &mut tracer);
                }
            });
        }

        // Get marking stats from the marker
        let stats = marker.stats();

        ConcurrentMarkStats {
            objects_marked: stats.objects_marked,
            objects_traced: total_traced,
        }
    }

    // =========================================================================
    // Phase 3: Remark + Sweep (STW)
    // =========================================================================

    /// Phase 3: Drain SATB, re-scan roots, sweep.
    ///
    /// Must be called during a stop-the-world pause.
    /// Returns (satb_entries_processed, bytes_freed, objects_freed).
    fn remark_and_sweep<T: ObjectTracer>(
        &mut self,
        heap: &mut GcHeap,
        roots: &RootSet,
        object_tracer: &T,
    ) -> (usize, usize, usize) {
        self.phase = ConcurrentGcPhase::Remark;

        // Transition SATB state to remarking
        self.marking_state.start_remark();

        // Drain SATB queue — mark all captured old-values
        let satb_processed = self.process_satb_queue(object_tracer);

        // Re-scan roots for references created during concurrent marking
        let rescan_roots = self.rescan_roots(heap, roots);

        // Process any newly discovered root pointers
        if !rescan_roots.is_empty() {
            self.trace_additional_roots(rescan_roots, object_tracer);
        }

        // Deactivate SATB barrier
        self.marking_state.finish();

        // Sweep phase
        self.phase = ConcurrentGcPhase::Sweep;
        let (bytes_freed, objects_freed) = self.sweep(heap);

        self.phase = ConcurrentGcPhase::Idle;

        (satb_processed, bytes_freed, objects_freed)
    }

    /// Process all entries in the SATB queue.
    ///
    /// Each captured old-value is marked in the bitmap and traced
    /// to discover any children that might have been missed.
    fn process_satb_queue<T: ObjectTracer>(&mut self, object_tracer: &T) -> usize {
        let mut processed = 0;

        // Drain all flushed SATB buffers
        let buffers = self.satb_queue.drain_buffers();
        for buffer in buffers {
            for &ptr in &buffer {
                if ptr.is_null() {
                    continue;
                }

                // Mark in bitmap if not already marked
                let addr = ptr as usize;
                if self.bitmap.mark(addr) == Some(true) {
                    processed += 1;

                    // Trace children of newly discovered object
                    let mut tracer = SatbRemarkTracer {
                        bitmap: &self.bitmap,
                    };
                    unsafe {
                        object_tracer.trace_object(ptr, &mut tracer);
                    }
                }
            }
        }

        processed
    }

    /// Re-scan roots during remark to find new references.
    fn rescan_roots(&self, heap: &GcHeap, roots: &RootSet) -> Vec<MarkPtr> {
        let mut tracer = InitialMarkTracer {
            bitmap: &self.bitmap,
            heap,
            roots_discovered: Vec::new(),
        };
        roots.trace(&mut tracer);
        tracer.roots_discovered
    }

    /// Trace additional roots discovered during remark.
    fn trace_additional_roots<T: ObjectTracer>(&mut self, roots: Vec<MarkPtr>, object_tracer: &T) {
        // Simple recursive tracing for remark (STW, so no concurrency needed)
        let mut worklist: Vec<MarkPtr> = roots;

        while let Some(obj_ptr) = worklist.pop() {
            let mut tracer = RemarkTracer {
                bitmap: &self.bitmap,
                worklist: &mut worklist,
            };
            unsafe {
                object_tracer.trace_object(obj_ptr.as_ptr(), &mut tracer);
            }
        }
    }

    /// Sweep unmarked objects from old-space and LOS.
    fn sweep(&self, heap: &mut GcHeap) -> (usize, usize) {
        let mut bytes_freed = 0;
        let mut objects_freed = 0;

        // Sweep old space
        let (old_freed, old_objects) = heap.old_space_mut().sweep();
        bytes_freed += old_freed;
        objects_freed += old_objects;

        // Sweep large object space
        let (los_freed, los_objects) = heap.large_objects().sweep();
        bytes_freed += los_freed;
        objects_freed += los_objects;

        (bytes_freed, objects_freed)
    }

    // =========================================================================
    // Bitmap Queries
    // =========================================================================

    /// Check if an address is marked in the bitmap.
    #[inline]
    pub fn is_marked(&self, addr: usize) -> bool {
        self.bitmap.is_marked(addr)
    }
}

impl Default for ConcurrentMajorCollector {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// Internal Tracers
// =============================================================================

/// Stats from the concurrent mark phase.
#[derive(Debug, Default)]
struct ConcurrentMarkStats {
    objects_marked: usize,
    objects_traced: usize,
}

/// Tracer for Phase 1: Initial Mark (root scanning).
///
/// Marks old-gen objects in the bitmap and collects them for the worklist.
struct InitialMarkTracer<'a> {
    bitmap: &'a AtomicMarkBitmap,
    heap: &'a GcHeap,
    roots_discovered: Vec<MarkPtr>,
}

impl<'a> Tracer for InitialMarkTracer<'a> {
    #[inline]
    fn trace_value(&mut self, value: prism_core::Value) {
        if let Some(ptr) = value.as_object_ptr() {
            self.trace_ptr(ptr);
        }
    }

    #[inline]
    fn trace_ptr(&mut self, ptr: *const ()) {
        if ptr.is_null() {
            return;
        }

        // Only process old-gen objects
        if !self.heap.is_old(ptr) {
            return;
        }

        // Mark in LOS if applicable
        if self.heap.large_objects().contains(ptr) {
            self.heap.large_objects().mark(ptr);
        }

        // Mark in bitmap — if newly marked, add to worklist
        let addr = ptr as usize;
        if self.bitmap.mark(addr) == Some(true) {
            self.roots_discovered.push(MarkPtr::new(ptr));
        }
    }
}

/// Tracer for Phase 2: Concurrent mark (object child tracing).
///
/// Used inside `process_local` to discover children of marked objects.
/// Marks children in the bitmap and pushes newly discovered ones to the worker.
///
/// Uses `&dyn Fn(MarkPtr)` to match the `process_local` callback signature,
/// which provides a dynamically-dispatched push function per worker.
struct ConcurrentObjectTracer<'a> {
    bitmap: &'a AtomicMarkBitmap,
    push_fn: &'a dyn Fn(MarkPtr),
}

impl<'a> Tracer for ConcurrentObjectTracer<'a> {
    #[inline]
    fn trace_value(&mut self, value: prism_core::Value) {
        if let Some(ptr) = value.as_object_ptr() {
            self.trace_ptr(ptr);
        }
    }

    #[inline]
    fn trace_ptr(&mut self, ptr: *const ()) {
        if ptr.is_null() {
            return;
        }

        let addr = ptr as usize;
        // Mark in bitmap — if newly marked, push to work-stealing deque
        if self.bitmap.mark(addr) == Some(true) {
            (self.push_fn)(MarkPtr::new(ptr));
        }
    }
}

/// Tracer for SATB remark — marks children during SATB processing.
struct SatbRemarkTracer<'a> {
    bitmap: &'a AtomicMarkBitmap,
}

impl<'a> Tracer for SatbRemarkTracer<'a> {
    #[inline]
    fn trace_value(&mut self, value: prism_core::Value) {
        if let Some(ptr) = value.as_object_ptr() {
            self.trace_ptr(ptr);
        }
    }

    #[inline]
    fn trace_ptr(&mut self, ptr: *const ()) {
        if ptr.is_null() {
            return;
        }
        // Mark children in bitmap (no further tracing needed for SATB —
        // they'll be picked up if the worklist is still active, or
        // they're already marked from the concurrent phase)
        let _ = self.bitmap.mark(ptr as usize);
    }
}

/// Tracer for remark phase — marks children and pushes to local worklist.
struct RemarkTracer<'a> {
    bitmap: &'a AtomicMarkBitmap,
    worklist: &'a mut Vec<MarkPtr>,
}

impl<'a> Tracer for RemarkTracer<'a> {
    #[inline]
    fn trace_value(&mut self, value: prism_core::Value) {
        if let Some(ptr) = value.as_object_ptr() {
            self.trace_ptr(ptr);
        }
    }

    #[inline]
    fn trace_ptr(&mut self, ptr: *const ()) {
        if ptr.is_null() {
            return;
        }

        let addr = ptr as usize;
        if self.bitmap.mark(addr) == Some(true) {
            self.worklist.push(MarkPtr::new(ptr));
        }
    }
}
