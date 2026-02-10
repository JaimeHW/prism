//! Tests for the concurrent marker infrastructure.

use super::concurrent_marker::*;
use super::mark_bitmap::{AtomicMarkBitmap, MIN_OBJECT_ALIGNMENT};
use crate::barrier::satb_buffer::SatbQueue;
use std::sync::Arc;

// =============================================================================
// Configuration Tests
// =============================================================================

#[test]
fn test_default_config() {
    let config = ConcurrentMarkerConfig::default();
    assert_eq!(config.marker_threads, 2);
    assert_eq!(config.work_chunk_size, 256);
    assert_eq!(config.distribute_threshold, 64);
}

#[test]
fn test_custom_config() {
    let config = ConcurrentMarkerConfig {
        marker_threads: 4,
        work_chunk_size: 512,
        distribute_threshold: 128,
    };
    assert_eq!(config.marker_threads, 4);
    assert_eq!(config.work_chunk_size, 512);
    assert_eq!(config.distribute_threshold, 128);
}

// =============================================================================
// MarkPtr Tests
// =============================================================================

#[test]
fn test_mark_ptr_creation() {
    let ptr = MarkPtr::new(0x1000 as *const ());
    assert_eq!(ptr.addr(), 0x1000);
    assert_eq!(ptr.as_ptr(), 0x1000 as *const ());
}

#[test]
fn test_mark_ptr_null() {
    let ptr = MarkPtr::new(std::ptr::null());
    assert_eq!(ptr.addr(), 0);
    assert!(ptr.as_ptr().is_null());
}

#[test]
fn test_mark_ptr_equality() {
    let a = MarkPtr::new(0x1000 as *const ());
    let b = MarkPtr::new(0x1000 as *const ());
    let c = MarkPtr::new(0x2000 as *const ());
    assert_eq!(a, b);
    assert_ne!(a, c);
}

#[test]
fn test_mark_ptr_copy() {
    let a = MarkPtr::new(0x1000 as *const ());
    let b = a; // Copy
    assert_eq!(a, b);
}

// =============================================================================
// MarkingStats Tests
// =============================================================================

#[test]
fn test_stats_initial_zero() {
    let stats = MarkingStats::new();
    let snap = stats.snapshot();
    assert_eq!(snap.objects_marked, 0);
    assert_eq!(snap.objects_traced, 0);
    assert_eq!(snap.satb_entries_processed, 0);
    assert_eq!(snap.steal_attempts, 0);
    assert_eq!(snap.steal_successes, 0);
}

#[test]
fn test_stats_record_mark() {
    let stats = MarkingStats::new();
    stats.record_mark();
    stats.record_mark();
    assert_eq!(stats.snapshot().objects_marked, 2);
}

#[test]
fn test_stats_record_trace() {
    let stats = MarkingStats::new();
    for _ in 0..10 {
        stats.record_trace();
    }
    assert_eq!(stats.snapshot().objects_traced, 10);
}

#[test]
fn test_stats_record_satb() {
    let stats = MarkingStats::new();
    stats.record_satb(100);
    stats.record_satb(50);
    assert_eq!(stats.snapshot().satb_entries_processed, 150);
}

#[test]
fn test_stats_record_steal() {
    let stats = MarkingStats::new();
    stats.record_steal_attempt();
    stats.record_steal_attempt();
    stats.record_steal_success();
    let snap = stats.snapshot();
    assert_eq!(snap.steal_attempts, 2);
    assert_eq!(snap.steal_successes, 1);
}

#[test]
fn test_stats_concurrent_increments() {
    let stats = Arc::new(MarkingStats::new());
    let num_threads = 8;
    let increments = 1000;

    let handles: Vec<_> = (0..num_threads)
        .map(|_| {
            let stats = Arc::clone(&stats);
            std::thread::spawn(move || {
                for _ in 0..increments {
                    stats.record_mark();
                    stats.record_trace();
                }
            })
        })
        .collect();

    for h in handles {
        h.join().unwrap();
    }

    let snap = stats.snapshot();
    assert_eq!(snap.objects_marked, num_threads * increments);
    assert_eq!(snap.objects_traced, num_threads * increments);
}

// =============================================================================
// MarkingWorklist Tests
// =============================================================================

#[test]
fn test_worklist_creation() {
    let (worklist, workers) = MarkingWorklist::new(4);
    assert_eq!(workers.len(), 4);
    assert!(worklist.is_injector_empty());
}

#[test]
fn test_worklist_inject_and_steal() {
    let stats = MarkingStats::new();
    let (worklist, _workers) = MarkingWorklist::new(2);

    worklist.inject(MarkPtr::new(0x1000 as *const ()));
    assert!(!worklist.is_injector_empty());

    let stolen = worklist.steal(&stats);
    assert!(stolen.is_some());
    assert_eq!(stolen.unwrap().addr(), 0x1000);
}

#[test]
fn test_worklist_inject_raw() {
    let stats = MarkingStats::new();
    let (worklist, _workers) = MarkingWorklist::new(2);

    worklist.inject_raw(0x2000 as *const ());
    let stolen = worklist.steal(&stats);
    assert!(stolen.is_some());
    assert_eq!(stolen.unwrap().addr(), 0x2000);
}

#[test]
fn test_worklist_inject_batch() {
    let stats = MarkingStats::new();
    let (worklist, _workers) = MarkingWorklist::new(2);
    let batch: Vec<MarkPtr> = (1..=5)
        .map(|i| MarkPtr::new((i * 0x1000) as *const ()))
        .collect();

    worklist.inject_batch(&batch);

    let mut stolen = Vec::new();
    while let Some(ptr) = worklist.steal(&stats) {
        stolen.push(ptr);
    }
    assert_eq!(stolen.len(), 5);
}

#[test]
fn test_worklist_steal_from_worker() {
    let stats = MarkingStats::new();
    let (worklist, workers) = MarkingWorklist::new(2);

    // Push to worker 0
    workers[0].push(MarkPtr::new(0x2000 as *const ()));
    workers[0].push(MarkPtr::new(0x3000 as *const ()));

    // Worker 0 pops its own work
    let local = workers[0].pop();
    assert!(local.is_some());

    // Steal should find work from worker 0's remaining entry
    let stolen = worklist.steal(&stats);
    assert!(stolen.is_some());
}

#[test]
fn test_worklist_steal_empty() {
    let stats = MarkingStats::new();
    let (worklist, _workers) = MarkingWorklist::new(2);

    let stolen = worklist.steal(&stats);
    assert!(stolen.is_none());
}

// =============================================================================
// ConcurrentMarker Tests
// =============================================================================

fn make_marker(size: usize) -> ConcurrentMarker {
    let bitmap = Arc::new(AtomicMarkBitmap::new(0, size));
    let satb = Arc::new(SatbQueue::new());
    ConcurrentMarker::new(ConcurrentMarkerConfig::default(), bitmap, satb)
}

#[test]
fn test_marker_creation() {
    let marker = make_marker(0x100000);
    assert_eq!(marker.num_threads(), 2);
    assert_eq!(marker.work_chunk_size(), 256);
    assert!(!marker.is_terminated());
}

#[test]
fn test_marker_mark_single() {
    let marker = make_marker(0x10000);

    assert!(!marker.is_marked(0));
    assert!(marker.mark(0)); // Newly marked
    assert!(marker.is_marked(0));
    assert!(!marker.mark(0)); // Already marked
}

#[test]
fn test_marker_mark_multiple() {
    let marker = make_marker(0x10000);

    for i in 0..100 {
        assert!(marker.mark(i * MIN_OBJECT_ALIGNMENT));
    }
    assert_eq!(marker.stats().objects_marked, 100);
}

#[test]
fn test_marker_clear() {
    let marker = make_marker(0x10000);

    marker.mark(0);
    marker.mark(MIN_OBJECT_ALIGNMENT);
    assert!(marker.is_marked(0));

    marker.clear();
    assert!(!marker.is_marked(0));
    assert!(!marker.is_terminated());
}

#[test]
fn test_marker_terminate() {
    let marker = make_marker(0x10000);

    assert!(!marker.is_terminated());
    marker.terminate();
    assert!(marker.is_terminated());

    // Clear resets termination
    marker.clear();
    assert!(!marker.is_terminated());
}

#[test]
fn test_marker_stats_tracking() {
    let marker = make_marker(0x10000);

    marker.mark(0);
    marker.mark(MIN_OBJECT_ALIGNMENT);
    marker.mark(0); // Already marked — no increment

    let snap = marker.stats();
    assert_eq!(snap.objects_marked, 2);
}

// =============================================================================
// SATB Processing Tests
// =============================================================================

#[test]
fn test_marker_process_satb_empty() {
    let marker = make_marker(0x10000);
    let count = marker.process_satb();
    assert_eq!(count, 0);
}

#[test]
fn test_marker_process_satb_entries() {
    let bitmap = Arc::new(AtomicMarkBitmap::new(0, 0x10000));
    let satb = Arc::new(SatbQueue::new());

    // Pre-populate SATB queue with old values
    let ptrs: Vec<*const ()> = (0..10)
        .map(|i| (i * MIN_OBJECT_ALIGNMENT) as *const ())
        .collect();
    satb.push_buffer(ptrs);

    let marker =
        ConcurrentMarker::new(ConcurrentMarkerConfig::default(), bitmap, Arc::clone(&satb));

    let count = marker.process_satb();
    assert_eq!(count, 10);

    // All should be marked now
    for i in 0..10 {
        assert!(marker.is_marked(i * MIN_OBJECT_ALIGNMENT));
    }
    assert_eq!(marker.stats().satb_entries_processed, 10);
}

// =============================================================================
// process_local Tests
// =============================================================================

#[test]
fn test_marker_process_local_simple() {
    let marker = make_marker(0x100000);
    let (worklist, workers) = MarkingWorklist::new(1);

    // Inject initial work
    let addr = 8 * MIN_OBJECT_ALIGNMENT;
    marker.mark(addr);
    worklist.inject(MarkPtr::new(addr as *const ()));

    // Process with a trace function that discovers children
    let child1 = 16 * MIN_OBJECT_ALIGNMENT;
    let child2 = 24 * MIN_OBJECT_ALIGNMENT;

    let traced = marker.process_local(&workers[0], &worklist, |ptr, push_fn| {
        let a = ptr.addr();
        if a == 8 * MIN_OBJECT_ALIGNMENT {
            // This object has two children
            push_fn(MarkPtr::new(child1 as *const ()));
            push_fn(MarkPtr::new(child2 as *const ()));
        }
        // Other objects have no children
    });

    // Should have traced 3 objects: root + 2 children
    assert_eq!(traced, 3);
    assert!(marker.is_marked(addr));
    assert!(marker.is_marked(child1));
    assert!(marker.is_marked(child2));
}

#[test]
fn test_marker_process_local_empty() {
    let marker = make_marker(0x10000);
    let (worklist, workers) = MarkingWorklist::new(1);

    let traced = marker.process_local(&workers[0], &worklist, |_ptr, _push_fn| {
        unreachable!("Should not be called with empty worklist");
    });

    assert_eq!(traced, 0);
}

#[test]
fn test_marker_process_local_chain() {
    let marker = make_marker(0x100000);
    let (worklist, workers) = MarkingWorklist::new(1);

    // Create a chain: A → B → C → D (no children)
    let a = 8 * MIN_OBJECT_ALIGNMENT;
    let b = 16 * MIN_OBJECT_ALIGNMENT;
    let c = 24 * MIN_OBJECT_ALIGNMENT;
    let d = 32 * MIN_OBJECT_ALIGNMENT;

    marker.mark(a);
    worklist.inject(MarkPtr::new(a as *const ()));

    let traced = marker.process_local(&workers[0], &worklist, |ptr, push_fn| {
        let addr = ptr.addr();
        match addr {
            x if x == a => push_fn(MarkPtr::new(b as *const ())),
            x if x == b => push_fn(MarkPtr::new(c as *const ())),
            x if x == c => push_fn(MarkPtr::new(d as *const ())),
            _ => {} // d has no children
        }
    });

    assert_eq!(traced, 4);
    assert!(marker.is_marked(a));
    assert!(marker.is_marked(b));
    assert!(marker.is_marked(c));
    assert!(marker.is_marked(d));
}

#[test]
fn test_marker_process_local_cycle() {
    let marker = make_marker(0x100000);
    let (worklist, workers) = MarkingWorklist::new(1);

    // Create a cycle: A → B → C → A
    let a = 8 * MIN_OBJECT_ALIGNMENT;
    let b = 16 * MIN_OBJECT_ALIGNMENT;
    let c = 24 * MIN_OBJECT_ALIGNMENT;

    marker.mark(a);
    worklist.inject(MarkPtr::new(a as *const ()));

    let traced = marker.process_local(&workers[0], &worklist, |ptr, push_fn| {
        let addr = ptr.addr();
        match addr {
            x if x == a => push_fn(MarkPtr::new(b as *const ())),
            x if x == b => push_fn(MarkPtr::new(c as *const ())),
            x if x == c => push_fn(MarkPtr::new(a as *const ())), // Cycle back → already marked
            _ => {}
        }
    });

    // Should terminate — the bitmap prevents re-pushing A
    assert_eq!(traced, 3);
    assert!(marker.is_marked(a));
    assert!(marker.is_marked(b));
    assert!(marker.is_marked(c));
}

#[test]
fn test_marker_process_local_dag() {
    let marker = make_marker(0x100000);
    let (worklist, workers) = MarkingWorklist::new(1);

    // DAG: A → {B, C}, B → {D}, C → {D}
    // D is shared — should only be traced once
    let a = 8 * MIN_OBJECT_ALIGNMENT;
    let b = 16 * MIN_OBJECT_ALIGNMENT;
    let c = 24 * MIN_OBJECT_ALIGNMENT;
    let d = 32 * MIN_OBJECT_ALIGNMENT;

    marker.mark(a);
    worklist.inject(MarkPtr::new(a as *const ()));

    let traced = marker.process_local(&workers[0], &worklist, |ptr, push_fn| {
        let addr = ptr.addr();
        match addr {
            x if x == a => {
                push_fn(MarkPtr::new(b as *const ()));
                push_fn(MarkPtr::new(c as *const ()));
            }
            x if x == b => push_fn(MarkPtr::new(d as *const ())),
            x if x == c => push_fn(MarkPtr::new(d as *const ())), // D already marked by B's trace
            _ => {}
        }
    });

    // 4 objects total, but D is shared so only traced once from whichever
    // path reaches it first. The second push for D is a no-op (already marked).
    assert_eq!(traced, 4);
    assert!(marker.is_marked(d));
}

// =============================================================================
// Concurrent Marking Tests
// =============================================================================

#[test]
fn test_concurrent_marking_multi_worker() {
    let marker = Arc::new(make_marker(0x100000));
    let (worklist, workers) = MarkingWorklist::new(4);
    let worklist = Arc::new(worklist);

    // Inject lots of initial work (no children — just test work distribution)
    for i in 0..1000 {
        let addr = i * MIN_OBJECT_ALIGNMENT;
        marker.mark(addr);
        worklist.inject(MarkPtr::new(addr as *const ()));
    }

    // Spawn 4 worker threads
    let handles: Vec<_> = workers
        .into_iter()
        .map(|worker| {
            let marker = Arc::clone(&marker);
            let worklist = Arc::clone(&worklist);
            std::thread::spawn(move || {
                marker.process_local(&worker, &worklist, |_ptr, _push_fn| {
                    // No children — each item is a leaf
                })
            })
        })
        .collect();

    let total_traced: usize = handles.into_iter().map(|h| h.join().unwrap()).sum();

    // All 1000 items should be traced (some by each worker)
    assert_eq!(total_traced, 1000);
    assert_eq!(marker.stats().objects_traced, 1000);
}

#[test]
fn test_concurrent_marking_with_children() {
    let marker = Arc::new(make_marker(0x100000));
    let (worklist, workers) = MarkingWorklist::new(2);
    let worklist = Arc::new(worklist);

    // Root objects at evenly spaced intervals
    // Each root discovers 10 children at adjacent addresses
    let num_roots = 100;
    let children_per_root = 10;

    for r in 0..num_roots {
        let addr = (r * (children_per_root + 1)) * MIN_OBJECT_ALIGNMENT;
        marker.mark(addr);
        worklist.inject(MarkPtr::new(addr as *const ()));
    }

    let handles: Vec<_> = workers
        .into_iter()
        .map(|worker| {
            let marker = Arc::clone(&marker);
            let worklist = Arc::clone(&worklist);
            std::thread::spawn(move || {
                marker.process_local(&worker, &worklist, |ptr, push_fn| {
                    let addr = ptr.addr();
                    let granule = addr / MIN_OBJECT_ALIGNMENT;

                    // If this is a root (offset is a multiple of children_per_root + 1)
                    if granule % (children_per_root + 1) == 0 {
                        let base = granule;
                        for c in 1..=children_per_root {
                            push_fn(MarkPtr::new(
                                ((base + c) * MIN_OBJECT_ALIGNMENT) as *const (),
                            ));
                        }
                    }
                    // Children have no further children
                })
            })
        })
        .collect();

    let total_traced: usize = handles.into_iter().map(|h| h.join().unwrap()).sum();

    let expected = num_roots * (1 + children_per_root);
    assert_eq!(total_traced, expected);
}

#[test]
fn test_concurrent_mark_with_satb_remark() {
    let bitmap = Arc::new(AtomicMarkBitmap::new(0, 0x100000));
    let satb = Arc::new(SatbQueue::new());

    let marker = Arc::new(ConcurrentMarker::new(
        ConcurrentMarkerConfig::default(),
        bitmap,
        Arc::clone(&satb),
    ));

    // Simulate concurrent marking phase
    for i in 0..100 {
        marker.mark(i * MIN_OBJECT_ALIGNMENT);
    }

    // Simulate SATB entries accumulated during marking
    let satb_ptrs: Vec<*const ()> = (100..150)
        .map(|i| (i * MIN_OBJECT_ALIGNMENT) as *const ())
        .collect();
    satb.push_buffer(satb_ptrs);

    // Remark phase
    let satb_count = marker.process_satb();
    assert_eq!(satb_count, 50);

    // All 150 should be marked
    let snap = marker.stats();
    assert_eq!(snap.objects_marked, 150);
    assert_eq!(snap.satb_entries_processed, 50);
}

#[test]
fn test_marker_accessors() {
    let bitmap = Arc::new(AtomicMarkBitmap::new(0, 0x10000));
    let satb = Arc::new(SatbQueue::new());
    let marker = ConcurrentMarker::new(
        ConcurrentMarkerConfig {
            marker_threads: 6,
            work_chunk_size: 128,
            distribute_threshold: 32,
        },
        Arc::clone(&bitmap),
        Arc::clone(&satb),
    );

    assert_eq!(marker.num_threads(), 6);
    assert_eq!(marker.work_chunk_size(), 128);
    // Check Arc references are shared
    assert!(Arc::ptr_eq(marker.bitmap(), &bitmap));
    assert!(Arc::ptr_eq(marker.satb_queue(), &satb));
}

// =============================================================================
// Stress Tests
// =============================================================================

#[test]
fn test_high_contention_marking() {
    // All threads try to mark overlapping address ranges
    let marker = Arc::new(make_marker(0x100000));
    let num_threads = 8;
    let marks_per_thread = 500;

    let handles: Vec<_> = (0..num_threads)
        .map(|_| {
            let marker = Arc::clone(&marker);
            std::thread::spawn(move || {
                let mut newly_marked = 0usize;
                for i in 0..marks_per_thread {
                    let addr = i * MIN_OBJECT_ALIGNMENT;
                    if marker.mark(addr) {
                        newly_marked += 1;
                    }
                }
                newly_marked
            })
        })
        .collect();

    let total_new: usize = handles.into_iter().map(|h| h.join().unwrap()).sum();

    // Each address marked exactly once across all threads
    assert_eq!(total_new, marks_per_thread);
    assert_eq!(marker.stats().objects_marked, marks_per_thread);
}

#[test]
fn test_worklist_concurrent_inject_and_steal() {
    let (worklist, _workers) = MarkingWorklist::new(4);
    let worklist = Arc::new(worklist);
    let stats = Arc::new(MarkingStats::new());

    // Inject from multiple threads, steal from one
    let inject_handles: Vec<_> = (0..4)
        .map(|t| {
            let worklist = Arc::clone(&worklist);
            std::thread::spawn(move || {
                for i in 0..100 {
                    worklist.inject(MarkPtr::new(((t * 100 + i) * 8 + 0x1000) as *const ()));
                }
            })
        })
        .collect();

    let steal_worklist = Arc::clone(&worklist);
    let steal_stats = Arc::clone(&stats);
    let steal_handle = std::thread::spawn(move || {
        let mut stolen = 0;
        // Keep stealing until injectors are done and queue is empty
        loop {
            if let Some(_) = steal_worklist.steal(&steal_stats) {
                stolen += 1;
            } else {
                // Brief yield, then try again
                std::thread::yield_now();
                // Try a few more times after getting empty
                let mut empty_runs = 0;
                loop {
                    if let Some(_) = steal_worklist.steal(&steal_stats) {
                        stolen += 1;
                        empty_runs = 0;
                    } else {
                        empty_runs += 1;
                        if empty_runs > 100 {
                            return stolen;
                        }
                        std::thread::yield_now();
                    }
                }
            }
        }
    });

    for h in inject_handles {
        h.join().unwrap();
    }
    let stolen = steal_handle.join().unwrap();

    // Stealer should have gotten some (possibly all) items
    assert!(stolen > 0, "Should have stolen at least some items");
}

#[test]
fn test_full_concurrent_pipeline() {
    // Integration test: initial mark → concurrent trace → SATB remark
    let bitmap = Arc::new(AtomicMarkBitmap::new(0, 0x100000));
    let satb = Arc::new(SatbQueue::new());
    let marker = Arc::new(ConcurrentMarker::new(
        ConcurrentMarkerConfig {
            marker_threads: 2,
            work_chunk_size: 64,
            distribute_threshold: 16,
        },
        bitmap,
        Arc::clone(&satb),
    ));

    // Phase 1: Initial mark (roots)
    let (worklist, workers) = MarkingWorklist::new(2);
    let worklist = Arc::new(worklist);

    let root_count = 50;
    for r in 0..root_count {
        let addr = r * 16 * MIN_OBJECT_ALIGNMENT;
        marker.mark(addr);
        worklist.inject(MarkPtr::new(addr as *const ()));
    }

    // Phase 2: Concurrent marking with 2 workers
    // Each root has 15 children (at adjacent granules)
    let handles: Vec<_> = workers
        .into_iter()
        .map(|worker| {
            let marker = Arc::clone(&marker);
            let worklist = Arc::clone(&worklist);
            std::thread::spawn(move || {
                marker.process_local(&worker, &worklist, |ptr, push_fn| {
                    let addr = ptr.addr();
                    let granule = addr / MIN_OBJECT_ALIGNMENT;
                    // Root objects are at granules 0, 16, 32, ...
                    if granule % 16 == 0 {
                        for c in 1..=15 {
                            push_fn(MarkPtr::new(
                                ((granule + c) * MIN_OBJECT_ALIGNMENT) as *const (),
                            ));
                        }
                    }
                })
            })
        })
        .collect();

    let total_traced: usize = handles.into_iter().map(|h| h.join().unwrap()).sum();

    // Phase 3: Simulate SATB captured during concurrent phase
    let satb_ptrs: Vec<*const ()> = (800..810)
        .map(|i| (i * MIN_OBJECT_ALIGNMENT) as *const ())
        .collect();
    satb.push_buffer(satb_ptrs);

    let satb_count = marker.process_satb();
    assert_eq!(satb_count, 10);

    // Verify totals
    let expected_traced = root_count * 16; // 50 roots + 50*15 children
    assert_eq!(total_traced, expected_traced);

    let snap = marker.stats();
    // 50 roots + 750 children + 10 SATB = 810 total marks
    assert_eq!(snap.objects_marked, expected_traced + 10);
    assert_eq!(snap.satb_entries_processed, 10);
}
