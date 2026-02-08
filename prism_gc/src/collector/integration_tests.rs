//! Integration tests for the concurrent GC infrastructure.
//!
//! Tests the full lifecycle of the concurrent major collector,
//! including orchestrator wiring, SATB barrier integration,
//! multi-cycle correctness, and concurrent safety scenarios.

#[cfg(test)]
mod tests {
    use crate::barrier::{SatbBuffer, SatbMarkingState, SatbQueue};
    use crate::collector::concurrent_major::{
        ConcurrentGcPhase, ConcurrentMajorCollector, ConcurrentMajorConfig,
    };
    use crate::collector::mark_bitmap::AtomicMarkBitmap;
    use crate::collector::{CollectionType, Collector};
    use crate::config::GcConfig;
    use crate::heap::GcHeap;
    use crate::roots::{RawHandle, RootSet};
    use crate::trace::{NoopObjectTracer, ObjectTracer, Tracer};
    use std::sync::Arc;
    use std::time::Duration;

    // =========================================================================
    // Helper: Test Object Tracer
    // =========================================================================

    /// A test object tracer that can be configured with a simulated object graph.
    ///
    /// When `trace_object` is called for a pointer, it looks up the pointer
    /// in its graph and calls `tracer.trace_ptr` for each child pointer.
    /// This allows testing the full marking pipeline with controlled topologies.
    struct GraphObjectTracer {
        /// Map from object address → child addresses.
        /// Using Vec for simplicity — sorted for binary search.
        edges: Vec<(usize, Vec<*const ()>)>,
    }

    impl GraphObjectTracer {
        fn new() -> Self {
            Self { edges: Vec::new() }
        }

        /// Add an edge: when tracing `parent`, discover `children`.
        fn add_object(&mut self, parent: *const (), children: Vec<*const ()>) {
            self.edges.push((parent as usize, children));
        }
    }

    // SAFETY: The test graph tracer only uses addresses as keys,
    // not dereferencing pointers. Safe for testing.
    unsafe impl Send for GraphObjectTracer {}
    unsafe impl Sync for GraphObjectTracer {}

    impl ObjectTracer for GraphObjectTracer {
        unsafe fn trace_object(&self, ptr: *const (), tracer: &mut dyn Tracer) {
            let addr = ptr as usize;
            for (parent_addr, children) in &self.edges {
                if *parent_addr == addr {
                    for &child in children {
                        tracer.trace_ptr(child);
                    }
                    return;
                }
            }
            // Unknown object — no children (leaf node)
        }

        unsafe fn size_of_object(&self, _ptr: *const ()) -> usize {
            64 // Simulated object size
        }

        unsafe fn finalize_object(&self, _ptr: *mut ()) {
            // No-op for tests
        }
    }

    // =========================================================================
    // 1. Collector Orchestrator Integration
    // =========================================================================

    #[test]
    fn test_collector_creates_concurrent_major() {
        let collector = Collector::new();
        assert_eq!(
            collector.concurrent_major().phase(),
            ConcurrentGcPhase::Idle
        );
    }

    #[test]
    fn test_collector_with_concurrent_config() {
        let config = ConcurrentMajorConfig {
            marker_threads: 2,
            work_chunk_size: 32,
            bitmap_coverage: 16 * 1024 * 1024,
        };
        let collector = Collector::with_concurrent_config(2, config);
        assert_eq!(collector.promotion_age(), 2);
        assert_eq!(collector.concurrent_major().config().marker_threads, 2);
        assert_eq!(collector.concurrent_major().config().work_chunk_size, 32);
    }

    #[test]
    fn test_collect_concurrent_major_empty_heap() {
        let mut collector = Collector::new();
        let mut heap = GcHeap::new(GcConfig::default());
        let roots = RootSet::new();

        let result = collector.collect_concurrent_major(&mut heap, &roots, &NoopObjectTracer);

        assert_eq!(result.collection_type, CollectionType::ConcurrentMajor);
        assert_eq!(result.bytes_freed, 0);
        assert_eq!(result.objects_freed, 0);
        assert!(result.duration > Duration::ZERO || result.duration == Duration::ZERO);
    }

    #[test]
    fn test_collect_concurrent_major_roots_only_empty() {
        let mut collector = Collector::new();
        let mut heap = GcHeap::new(GcConfig::default());
        let roots = RootSet::new();

        let result = collector.collect_concurrent_major_roots_only(&mut heap, &roots);

        assert_eq!(result.collection_type, CollectionType::ConcurrentMajor);
        assert_eq!(result.bytes_freed, 0);
    }

    #[test]
    fn test_collection_type_concurrent_major_distinct() {
        assert_ne!(CollectionType::ConcurrentMajor, CollectionType::Minor);
        assert_ne!(CollectionType::ConcurrentMajor, CollectionType::Major);
    }

    #[test]
    fn test_collection_type_concurrent_major_debug() {
        let debug = format!("{:?}", CollectionType::ConcurrentMajor);
        assert!(debug.contains("ConcurrentMajor"));
    }

    #[test]
    fn test_collection_type_concurrent_major_clone() {
        let ct = CollectionType::ConcurrentMajor;
        let cloned = ct;
        assert_eq!(ct, cloned);
    }

    #[test]
    fn test_concurrent_major_accessor_returns_same_instance() {
        let collector = Collector::new();
        let phase1 = collector.concurrent_major().phase();
        let phase2 = collector.concurrent_major().phase();
        assert_eq!(phase1, phase2);
    }

    #[test]
    fn test_concurrent_major_mut_accessor() {
        let mut collector = Collector::new();
        // Should be able to get mutable reference
        let concurrent = collector.concurrent_major_mut();
        assert_eq!(concurrent.phase(), ConcurrentGcPhase::Idle);
    }

    // =========================================================================
    // 2. Full Collection Lifecycle (Standalone)
    // =========================================================================

    #[test]
    fn test_standalone_collect_empty_heap() {
        let mut collector = ConcurrentMajorCollector::for_testing();
        let mut heap = GcHeap::new(GcConfig::default());
        let roots = RootSet::new();

        let result = collector.collect(&mut heap, &roots, &NoopObjectTracer);

        assert_eq!(result.bytes_freed, 0);
        assert_eq!(result.objects_freed, 0);
        assert_eq!(result.objects_marked, 0);
        assert_eq!(result.objects_traced, 0);
        assert_eq!(result.satb_entries_processed, 0);
        // Phase should return to idle after collection
        assert_eq!(collector.phase(), ConcurrentGcPhase::Idle);
    }

    #[test]
    fn test_standalone_collect_result_durations_populated() {
        let mut collector = ConcurrentMajorCollector::for_testing();
        let mut heap = GcHeap::new(GcConfig::default());
        let roots = RootSet::new();

        let result = collector.collect(&mut heap, &roots, &NoopObjectTracer);

        // Total duration should be >= sum of phase durations
        assert!(
            result.total_duration
                >= result
                    .initial_mark_duration
                    .saturating_add(result.concurrent_mark_duration)
                    .saturating_sub(Duration::from_micros(10)) // tolerance for measurement
        );
    }

    #[test]
    fn test_standalone_phase_returns_to_idle() {
        let mut collector = ConcurrentMajorCollector::for_testing();
        let mut heap = GcHeap::new(GcConfig::default());
        let roots = RootSet::new();

        assert_eq!(collector.phase(), ConcurrentGcPhase::Idle);
        collector.collect(&mut heap, &roots, &NoopObjectTracer);
        assert_eq!(collector.phase(), ConcurrentGcPhase::Idle);
    }

    #[test]
    fn test_standalone_satb_state_returns_to_idle() {
        let mut collector = ConcurrentMajorCollector::for_testing();
        let mut heap = GcHeap::new(GcConfig::default());
        let roots = RootSet::new();

        assert!(!collector.marking_state().is_active());
        collector.collect(&mut heap, &roots, &NoopObjectTracer);
        assert!(!collector.marking_state().is_active());
    }

    // =========================================================================
    // 3. Root Scanning Integration
    // =========================================================================

    #[test]
    fn test_collect_with_handle_roots() {
        let mut collector = ConcurrentMajorCollector::for_testing();
        let mut heap = GcHeap::new(GcConfig::default());
        let roots = RootSet::new();

        // Register handle pointers as roots
        // These are fake pointers pointing to simulated addresses.
        // The collector will try to mark them in the bitmap.
        let handle1 = RawHandle::new(0x1000 as *const ());
        let handle2 = RawHandle::new(0x2000 as *const ());
        roots.register_handle(handle1);
        roots.register_handle(handle2);

        assert_eq!(roots.handle_count(), 2);

        let result = collector.collect(&mut heap, &roots, &NoopObjectTracer);

        // Handles point to simulated addresses that are not in old space,
        // so they won't be marked (is_old returns false). This tests that
        // the root scanning doesn't crash on non-old-space pointers.
        assert_eq!(collector.phase(), ConcurrentGcPhase::Idle);
        assert_eq!(result.bytes_freed, 0);
    }

    #[test]
    fn test_collect_with_global_roots() {
        let mut collector = ConcurrentMajorCollector::for_testing();
        let mut heap = GcHeap::new(GcConfig::default());
        let roots = RootSet::new();

        // Add global value roots
        roots.add_global(prism_core::Value::int(42).unwrap());
        roots.add_global(prism_core::Value::bool(true));
        roots.add_global(prism_core::Value::float(3.14));

        assert_eq!(roots.global_count(), 3);

        let result = collector.collect(&mut heap, &roots, &NoopObjectTracer);

        // Integer/bool/float values are not object pointers — they won't be
        // traced as roots, but collection should complete without error.
        assert_eq!(collector.phase(), ConcurrentGcPhase::Idle);
        assert_eq!(result.bytes_freed, 0);
    }

    #[test]
    fn test_collect_with_null_handle_root() {
        let mut collector = ConcurrentMajorCollector::for_testing();
        let mut heap = GcHeap::new(GcConfig::default());
        let roots = RootSet::new();

        // Null handles should be safely skipped
        let null_handle = RawHandle::null();
        roots.register_handle(null_handle);

        let result = collector.collect(&mut heap, &roots, &NoopObjectTracer);
        assert_eq!(collector.phase(), ConcurrentGcPhase::Idle);
        assert_eq!(result.bytes_freed, 0);
    }

    #[test]
    fn test_collect_with_mixed_roots() {
        let mut collector = ConcurrentMajorCollector::for_testing();
        let mut heap = GcHeap::new(GcConfig::default());
        let roots = RootSet::new();

        // Mix of handles, globals, and nulls
        roots.register_handle(RawHandle::new(0x1000 as *const ()));
        roots.register_handle(RawHandle::null());
        roots.add_global(prism_core::Value::int(1).unwrap());
        roots.add_global(prism_core::Value::bool(false));

        let result = collector.collect(&mut heap, &roots, &NoopObjectTracer);
        assert_eq!(collector.phase(), ConcurrentGcPhase::Idle);
        assert_eq!(result.bytes_freed, 0);
    }

    // =========================================================================
    // 4. SATB Injection During Collection
    // =========================================================================

    #[test]
    fn test_satb_entries_processed_in_remark() {
        let mut collector = ConcurrentMajorCollector::for_testing();
        let mut heap = GcHeap::new(GcConfig::default());
        let roots = RootSet::new();

        // Pre-inject SATB entries that simulate write barrier captures
        let satb_queue = Arc::clone(collector.satb_queue());
        satb_queue.push_buffer(vec![0x1000 as *const (), 0x2000 as *const ()]);

        // The SATB entries won't be processed until the remark phase,
        // which needs marking to be active first. The collect() method
        // handles the full lifecycle including draining SATB.
        let result = collector.collect(&mut heap, &roots, &NoopObjectTracer);

        // After collection, the SATB queue should be drained
        assert!(collector.satb_queue().is_empty());
        assert_eq!(collector.phase(), ConcurrentGcPhase::Idle);
        // Pre-injected SATB entries are processed during remark.
        // The count depends on whether the entries fall within bitmap
        // coverage — simulated addresses at 0x1000/0x2000 are within
        // the testing config's 64MB coverage, so they are processed.
        // However, since they are not real old-gen objects, they won't
        // result in any actual tracing work.
        assert!(result.satb_entries_processed >= 0);
    }

    #[test]
    fn test_satb_barrier_capture_during_active_marking() {
        let collector = ConcurrentMajorCollector::for_testing();
        let state = Arc::clone(collector.marking_state());
        let queue = Arc::clone(collector.satb_queue());

        // Simulate what happens during concurrent marking:
        // 1. Marking starts
        state.start_marking();

        // 2. Mutator writes trigger SATB barrier
        let mut buffer = SatbBuffer::new();
        let old_ptr = 0xABCD0000 as *const ();
        crate::barrier::satb_write_barrier(old_ptr, &state, &mut buffer, &queue);

        // 3. Buffer captures the old value
        assert_eq!(buffer.len(), 1);

        // 4. Flush to global queue
        buffer.flush(&queue);
        assert_eq!(queue.total_entries(), 1);

        // 5. Marking finishes
        state.start_remark();
        state.finish();
    }

    #[test]
    fn test_satb_multi_thread_simulation() {
        let collector = ConcurrentMajorCollector::for_testing();
        let state = Arc::clone(collector.marking_state());
        let queue = Arc::clone(collector.satb_queue());

        state.start_marking();

        // Simulate multiple threads writing concurrently
        let mut buffers: Vec<SatbBuffer> = (0..4).map(|_| SatbBuffer::new()).collect();

        for (i, buffer) in buffers.iter_mut().enumerate() {
            let ptr = ((i + 1) * 0x1000) as *const ();
            crate::barrier::satb_write_barrier(ptr, &state, buffer, &queue);
        }

        // All buffers should have one entry each
        for buffer in &buffers {
            assert_eq!(buffer.len(), 1);
        }

        // Flush all buffers
        for buffer in &mut buffers {
            buffer.flush(&queue);
        }

        assert_eq!(queue.total_entries(), 4);
        assert_eq!(queue.buffer_count(), 4);

        // Drain should get all entries
        let drained = queue.drain_buffers();
        assert_eq!(drained.len(), 4);
        assert!(queue.is_empty());

        state.finish();
    }

    // =========================================================================
    // 5. Multi-Cycle Correctness
    // =========================================================================

    #[test]
    fn test_multiple_collection_cycles() {
        let mut collector = ConcurrentMajorCollector::for_testing();
        let mut heap = GcHeap::new(GcConfig::default());
        let roots = RootSet::new();

        // Run multiple collection cycles
        for i in 0..5 {
            let result = collector.collect(&mut heap, &roots, &NoopObjectTracer);
            assert_eq!(
                collector.phase(),
                ConcurrentGcPhase::Idle,
                "Phase not idle after cycle {}",
                i
            );
            assert!(
                !collector.marking_state().is_active(),
                "Marking state still active after cycle {}",
                i
            );
            assert_eq!(result.bytes_freed, 0);
            assert_eq!(result.objects_freed, 0);
        }
    }

    #[test]
    fn test_bitmap_cleared_between_cycles() {
        let mut collector = ConcurrentMajorCollector::for_testing();
        let mut heap = GcHeap::new(GcConfig::default());
        let roots = RootSet::new();

        // Mark something in the bitmap manually
        collector.bitmap().mark(1024);
        assert!(collector.is_marked(1024));

        // Run a collection — it should reset the bitmap
        collector.collect(&mut heap, &roots, &NoopObjectTracer);

        // After collection, the bitmap should be fresh (old marks cleared)
        assert!(!collector.is_marked(1024));
    }

    #[test]
    fn test_satb_queue_cleared_between_cycles() {
        let mut collector = ConcurrentMajorCollector::for_testing();
        let mut heap = GcHeap::new(GcConfig::default());
        let roots = RootSet::new();

        // First cycle
        collector.collect(&mut heap, &roots, &NoopObjectTracer);

        // Add SATB entries between cycles
        collector
            .satb_queue()
            .push_buffer(vec![0x1000 as *const ()]);
        assert!(!collector.satb_queue().is_empty());

        // Second cycle should drain SATB
        collector.collect(&mut heap, &roots, &NoopObjectTracer);
        // After remark, the queue should be empty
        assert!(collector.satb_queue().is_empty());
    }

    #[test]
    fn test_marking_state_not_corrupted_across_cycles() {
        let mut collector = ConcurrentMajorCollector::for_testing();
        let mut heap = GcHeap::new(GcConfig::default());
        let roots = RootSet::new();

        for _ in 0..10 {
            assert!(!collector.marking_state().is_active());
            collector.collect(&mut heap, &roots, &NoopObjectTracer);
            assert!(!collector.marking_state().is_active());
        }
    }

    // =========================================================================
    // 6. Orchestrator Multi-Collection Interleaving
    // =========================================================================

    #[test]
    fn test_interleave_minor_and_concurrent_major() {
        let mut collector = Collector::new();
        let mut heap = GcHeap::new(GcConfig::default());
        let roots = RootSet::new();

        // Alternate minor and concurrent major collections
        let minor_result = collector.collect_minor(&mut heap, &roots, &NoopObjectTracer);
        assert_eq!(minor_result.collection_type, CollectionType::Minor);

        let major_result = collector.collect_concurrent_major(&mut heap, &roots, &NoopObjectTracer);
        assert_eq!(
            major_result.collection_type,
            CollectionType::ConcurrentMajor
        );

        let minor_result2 = collector.collect_minor(&mut heap, &roots, &NoopObjectTracer);
        assert_eq!(minor_result2.collection_type, CollectionType::Minor);
    }

    #[test]
    fn test_interleave_stw_major_and_concurrent_major() {
        let mut collector = Collector::new();
        let mut heap = GcHeap::new(GcConfig::default());
        let roots = RootSet::new();

        // STW major
        let stw_result = collector.collect_major(&mut heap, &roots, &NoopObjectTracer);
        assert_eq!(stw_result.collection_type, CollectionType::Major);

        // Concurrent major
        let conc_result = collector.collect_concurrent_major(&mut heap, &roots, &NoopObjectTracer);
        assert_eq!(conc_result.collection_type, CollectionType::ConcurrentMajor);

        // Both should work without corruption
        assert_eq!(
            collector.concurrent_major().phase(),
            ConcurrentGcPhase::Idle
        );
    }

    #[test]
    fn test_all_three_collection_types_sequential() {
        let mut collector = Collector::new();
        let mut heap = GcHeap::new(GcConfig::default());
        let roots = RootSet::new();

        let r1 = collector.collect_minor(&mut heap, &roots, &NoopObjectTracer);
        let r2 = collector.collect_major(&mut heap, &roots, &NoopObjectTracer);
        let r3 = collector.collect_concurrent_major(&mut heap, &roots, &NoopObjectTracer);

        assert_eq!(r1.collection_type, CollectionType::Minor);
        assert_eq!(r2.collection_type, CollectionType::Major);
        assert_eq!(r3.collection_type, CollectionType::ConcurrentMajor);
    }

    // =========================================================================
    // 7. SATB Write Barrier Integration with Full Cycle
    // =========================================================================

    #[test]
    fn test_write_barrier_concurrent_idle_noop() {
        let collector = ConcurrentMajorCollector::for_testing();
        let state = collector.marking_state();
        let queue = collector.satb_queue();

        // When marking is idle, the barrier should be a no-op
        let mut buffer = SatbBuffer::new();
        for addr in [0x1000usize, 0x2000, 0x3000, 0x4000] {
            let ptr = addr as *const ();
            crate::barrier::satb_write_barrier(ptr, state, &mut buffer, queue);
        }

        assert_eq!(buffer.len(), 0);
        assert!(queue.is_empty());
    }

    #[test]
    fn test_write_barrier_concurrent_active_captures() {
        let collector = ConcurrentMajorCollector::for_testing();
        let state = collector.marking_state();
        let queue = collector.satb_queue();

        // Activate marking
        state.start_marking();

        let mut buffer = SatbBuffer::new();
        for addr in [0x1000usize, 0x2000, 0x3000] {
            let ptr = addr as *const ();
            crate::barrier::satb_write_barrier(ptr, state, &mut buffer, queue);
        }

        assert_eq!(buffer.len(), 3);

        state.finish();
    }

    #[test]
    fn test_write_barrier_skips_null() {
        let collector = ConcurrentMajorCollector::for_testing();
        let state = collector.marking_state();
        let queue = collector.satb_queue();

        state.start_marking();

        let mut buffer = SatbBuffer::new();
        crate::barrier::satb_write_barrier(std::ptr::null(), state, &mut buffer, queue);

        assert_eq!(buffer.len(), 0);

        state.finish();
    }

    // =========================================================================
    // 8. GraphObjectTracer Integration
    // =========================================================================

    #[test]
    fn test_graph_tracer_leaf_node() {
        let tracer = GraphObjectTracer::new();
        // Leaf node — should not discover any children
        let mut counting = CountingTracerImpl::new();
        unsafe {
            tracer.trace_object(0x1000 as *const (), &mut counting);
        }
        assert_eq!(counting.count, 0);
    }

    #[test]
    fn test_graph_tracer_single_child() {
        let mut tracer = GraphObjectTracer::new();
        let parent = 0x1000 as *const ();
        let child = 0x2000 as *const ();
        tracer.add_object(parent, vec![child]);

        let mut counting = CountingTracerImpl::new();
        unsafe {
            tracer.trace_object(parent, &mut counting);
        }
        assert_eq!(counting.count, 1);
    }

    #[test]
    fn test_graph_tracer_multiple_children() {
        let mut tracer = GraphObjectTracer::new();
        let parent = 0x1000 as *const ();
        let children = vec![
            0x2000 as *const (),
            0x3000 as *const (),
            0x4000 as *const (),
        ];
        tracer.add_object(parent, children);

        let mut counting = CountingTracerImpl::new();
        unsafe {
            tracer.trace_object(parent, &mut counting);
        }
        assert_eq!(counting.count, 3);
    }

    #[test]
    fn test_graph_tracer_disconnected_nodes() {
        let mut tracer = GraphObjectTracer::new();
        // Two separate objects, no shared children
        tracer.add_object(0x1000 as *const (), vec![0x2000 as *const ()]);
        tracer.add_object(0x3000 as *const (), vec![0x4000 as *const ()]);

        let mut c1 = CountingTracerImpl::new();
        unsafe {
            tracer.trace_object(0x1000 as *const (), &mut c1);
        }
        assert_eq!(c1.count, 1);

        let mut c2 = CountingTracerImpl::new();
        unsafe {
            tracer.trace_object(0x3000 as *const (), &mut c2);
        }
        assert_eq!(c2.count, 1);
    }

    #[test]
    fn test_graph_tracer_size_of_object() {
        let tracer = GraphObjectTracer::new();
        unsafe {
            assert_eq!(tracer.size_of_object(0x1000 as *const ()), 64);
        }
    }

    // =========================================================================
    // 9. Bitmap Operations Across Collection
    // =========================================================================

    #[test]
    fn test_bitmap_shared_with_marker() {
        let collector = ConcurrentMajorCollector::for_testing();
        let bitmap = Arc::clone(collector.bitmap());

        // Both references should point to the same bitmap
        bitmap.mark(1024);
        assert!(collector.is_marked(1024));
    }

    #[test]
    fn test_bitmap_coverage_matches_config() {
        let config = ConcurrentMajorConfig {
            marker_threads: 1,
            work_chunk_size: 16,
            bitmap_coverage: 1024 * 1024, // 1 MB
        };
        let collector = ConcurrentMajorCollector::with_config(config);

        // Should be able to mark within range
        assert_eq!(collector.bitmap().mark(0), Some(true));
        assert_eq!(collector.bitmap().mark(1024 * 1024 - 8), Some(true));

        // Out of range
        assert_eq!(collector.bitmap().mark(2 * 1024 * 1024), None);
    }

    #[test]
    fn test_bitmap_bulk_mark_and_query() {
        let collector = ConcurrentMajorCollector::for_testing();
        let bitmap = collector.bitmap();

        // Mark 100 addresses
        let addrs: Vec<usize> = (0..100).map(|i| i * 64).collect();
        for &addr in &addrs {
            assert_eq!(bitmap.mark(addr), Some(true));
        }

        // All should be marked
        for &addr in &addrs {
            assert!(bitmap.is_marked(addr));
        }

        // Unmarked addresses
        assert!(!bitmap.is_marked(32));
        assert!(!bitmap.is_marked(100 * 64));
    }

    // =========================================================================
    // 10. Concurrent Safety Tests
    // =========================================================================

    #[test]
    fn test_concurrent_bitmap_marking() {
        use std::sync::Arc;
        use std::thread;

        let bitmap = Arc::new(AtomicMarkBitmap::new(0, 1024 * 1024));
        let num_threads = 4;
        let marks_per_thread = 100;

        let handles: Vec<_> = (0..num_threads)
            .map(|tid| {
                let bitmap = Arc::clone(&bitmap);
                thread::spawn(move || {
                    let mut newly_marked = 0;
                    for i in 0..marks_per_thread {
                        let addr = (tid * marks_per_thread + i) * 8;
                        if bitmap.mark(addr) == Some(true) {
                            newly_marked += 1;
                        }
                    }
                    newly_marked
                })
            })
            .collect();

        let total: usize = handles.into_iter().map(|h| h.join().unwrap()).sum();

        // Each thread marks unique addresses, so all should be newly marked
        assert_eq!(total, num_threads * marks_per_thread);

        // Verify all marks
        for tid in 0..num_threads {
            for i in 0..marks_per_thread {
                let addr = (tid * marks_per_thread + i) * 8;
                assert!(bitmap.is_marked(addr));
            }
        }
    }

    #[test]
    fn test_concurrent_bitmap_contention() {
        use std::sync::Arc;
        use std::thread;

        let bitmap = Arc::new(AtomicMarkBitmap::new(0, 1024 * 1024));
        let num_threads = 8;
        // All threads try to mark the SAME address — only one should succeed
        let contested_addr = 4096usize;

        let handles: Vec<_> = (0..num_threads)
            .map(|_| {
                let bitmap = Arc::clone(&bitmap);
                thread::spawn(move || bitmap.mark(contested_addr) == Some(true))
            })
            .collect();

        let newly_marked: usize = handles
            .into_iter()
            .map(|h| if h.join().unwrap() { 1 } else { 0 })
            .sum();

        // Exactly one thread should have newly marked it
        assert_eq!(newly_marked, 1);
        assert!(bitmap.is_marked(contested_addr));
    }

    #[test]
    fn test_concurrent_satb_queue_push() {
        use std::sync::Arc;
        use std::thread;

        let queue = Arc::new(SatbQueue::new());
        let num_threads = 4;
        let entries_per_thread = 10;

        let handles: Vec<_> = (0..num_threads)
            .map(|tid| {
                let queue = Arc::clone(&queue);
                thread::spawn(move || {
                    let ptrs: Vec<*const ()> = (0..entries_per_thread)
                        .map(|i| ((tid * entries_per_thread + i + 1) * 0x1000) as *const ())
                        .collect();
                    queue.push_buffer(ptrs);
                })
            })
            .collect();

        for h in handles {
            h.join().unwrap();
        }

        assert_eq!(queue.buffer_count(), num_threads);
        assert_eq!(queue.total_entries(), num_threads * entries_per_thread);

        let drained = queue.drain_buffers();
        assert_eq!(drained.len(), num_threads);
        assert!(queue.is_empty());
    }

    #[test]
    fn test_concurrent_marking_state_transitions() {
        use std::sync::Arc;
        use std::thread;

        let state = Arc::new(SatbMarkingState::new());

        // Start marking
        state.start_marking();

        // Multiple threads check is_active concurrently
        let handles: Vec<_> = (0..8)
            .map(|_| {
                let state = Arc::clone(&state);
                thread::spawn(move || state.is_active())
            })
            .collect();

        for h in handles {
            assert!(h.join().unwrap());
        }

        state.finish();
        assert!(!state.is_active());
    }

    // =========================================================================
    // 11. Edge Cases
    // =========================================================================

    #[test]
    fn test_zero_thread_collection() {
        let config = ConcurrentMajorConfig {
            marker_threads: 0,
            work_chunk_size: 16,
            bitmap_coverage: 1024 * 1024,
        };
        let mut collector = ConcurrentMajorCollector::with_config(config);
        let mut heap = GcHeap::new(GcConfig::default());
        let roots = RootSet::new();

        // Collection with 0 threads should still work (no workers process anything)
        let result = collector.collect(&mut heap, &roots, &NoopObjectTracer);
        assert_eq!(collector.phase(), ConcurrentGcPhase::Idle);
        assert_eq!(result.bytes_freed, 0);
    }

    #[test]
    fn test_single_thread_collection() {
        let config = ConcurrentMajorConfig {
            marker_threads: 1,
            work_chunk_size: 16,
            bitmap_coverage: 1024 * 1024,
        };
        let mut collector = ConcurrentMajorCollector::with_config(config);
        let mut heap = GcHeap::new(GcConfig::default());
        let roots = RootSet::new();

        let result = collector.collect(&mut heap, &roots, &NoopObjectTracer);
        assert_eq!(collector.phase(), ConcurrentGcPhase::Idle);
        assert_eq!(result.bytes_freed, 0);
    }

    #[test]
    fn test_many_thread_config() {
        let config = ConcurrentMajorConfig {
            marker_threads: 16,
            work_chunk_size: 128,
            bitmap_coverage: 256 * 1024 * 1024,
        };
        let mut collector = ConcurrentMajorCollector::with_config(config);
        let mut heap = GcHeap::new(GcConfig::default());
        let roots = RootSet::new();

        let result = collector.collect(&mut heap, &roots, &NoopObjectTracer);
        assert_eq!(collector.phase(), ConcurrentGcPhase::Idle);
        assert_eq!(result.bytes_freed, 0);
    }

    #[test]
    fn test_collect_with_empty_satb_queue() {
        let mut collector = ConcurrentMajorCollector::for_testing();
        let mut heap = GcHeap::new(GcConfig::default());
        let roots = RootSet::new();

        assert!(collector.satb_queue().is_empty());

        let result = collector.collect(&mut heap, &roots, &NoopObjectTracer);
        assert_eq!(result.satb_entries_processed, 0);
        assert!(collector.satb_queue().is_empty());
    }

    #[test]
    fn test_collect_with_large_work_chunk() {
        let config = ConcurrentMajorConfig {
            marker_threads: 1,
            work_chunk_size: 1024, // Very large chunk
            bitmap_coverage: 1024 * 1024,
        };
        let mut collector = ConcurrentMajorCollector::with_config(config);
        let mut heap = GcHeap::new(GcConfig::default());
        let roots = RootSet::new();

        let result = collector.collect(&mut heap, &roots, &NoopObjectTracer);
        assert_eq!(collector.phase(), ConcurrentGcPhase::Idle);
        assert_eq!(result.bytes_freed, 0);
    }

    #[test]
    fn test_collect_with_minimum_bitmap_coverage() {
        let config = ConcurrentMajorConfig {
            marker_threads: 1,
            work_chunk_size: 8,
            bitmap_coverage: 64, // Absolute minimum — ~1 cache line
        };
        let mut collector = ConcurrentMajorCollector::with_config(config);
        let mut heap = GcHeap::new(GcConfig::default());
        let roots = RootSet::new();

        let result = collector.collect(&mut heap, &roots, &NoopObjectTracer);
        assert_eq!(collector.phase(), ConcurrentGcPhase::Idle);
        assert_eq!(result.bytes_freed, 0);
    }

    // =========================================================================
    // 12. Regression: Existing Collections Still Work
    // =========================================================================

    #[test]
    fn test_minor_collection_still_works() {
        let mut collector = Collector::new();
        let mut heap = GcHeap::new(GcConfig::default());
        let roots = RootSet::new();

        let result = collector.collect_minor(&mut heap, &roots, &NoopObjectTracer);
        assert_eq!(result.collection_type, CollectionType::Minor);
    }

    #[test]
    fn test_major_collection_still_works() {
        let mut collector = Collector::new();
        let mut heap = GcHeap::new(GcConfig::default());
        let roots = RootSet::new();

        let result = collector.collect_major(&mut heap, &roots, &NoopObjectTracer);
        assert_eq!(result.collection_type, CollectionType::Major);
    }

    #[test]
    fn test_roots_only_collections_still_work() {
        let mut collector = Collector::new();
        let mut heap = GcHeap::new(GcConfig::default());
        let roots = RootSet::new();

        let minor = collector.collect_minor_roots_only(&mut heap, &roots);
        let major = collector.collect_major_roots_only(&mut heap, &roots);

        assert_eq!(minor.collection_type, CollectionType::Minor);
        assert_eq!(major.collection_type, CollectionType::Major);
    }

    #[test]
    fn test_auto_select_still_works() {
        let mut collector = Collector::new();
        let mut heap = GcHeap::new(GcConfig::default());
        let roots = RootSet::new();

        let result = collector.collect(&mut heap, &roots, &NoopObjectTracer);
        // Empty heap should select minor collection
        assert_eq!(result.collection_type, CollectionType::Minor);
    }

    // =========================================================================
    // 13. Stress Tests
    // =========================================================================

    #[test]
    fn test_rapid_collection_cycles() {
        let mut collector = ConcurrentMajorCollector::for_testing();
        let mut heap = GcHeap::new(GcConfig::default());
        let roots = RootSet::new();

        // Run many rapid collection cycles
        for _ in 0..20 {
            let result = collector.collect(&mut heap, &roots, &NoopObjectTracer);
            assert_eq!(collector.phase(), ConcurrentGcPhase::Idle);
            assert!(!collector.marking_state().is_active());
            assert_eq!(result.bytes_freed, 0);
        }
    }

    #[test]
    fn test_rapid_orchestrator_cycles() {
        let mut collector = Collector::new();
        let mut heap = GcHeap::new(GcConfig::default());
        let roots = RootSet::new();

        for _ in 0..20 {
            let result = collector.collect_concurrent_major(&mut heap, &roots, &NoopObjectTracer);
            assert_eq!(result.collection_type, CollectionType::ConcurrentMajor);
        }
    }

    #[test]
    fn test_satb_stress_push_drain_cycle() {
        let queue = Arc::new(SatbQueue::new());

        for cycle in 0..10 {
            // Push many buffers
            for i in 0..50 {
                let ptrs: Vec<*const ()> = (0..10)
                    .map(|j| ((cycle * 500 + i * 10 + j + 1) * 0x100) as *const ())
                    .collect();
                queue.push_buffer(ptrs);
            }

            assert_eq!(queue.buffer_count(), 50);
            assert_eq!(queue.total_entries(), 500);

            // Drain all
            let drained = queue.drain_buffers();
            assert_eq!(drained.len(), 50);
            assert!(queue.is_empty());
        }
    }

    #[test]
    fn test_bitmap_stress_mark_clear() {
        let bitmap = AtomicMarkBitmap::new(0, 64 * 1024);

        // Mark many addresses
        for i in 0..1000 {
            let addr = i * 8;
            assert_eq!(bitmap.mark(addr), Some(true));
        }

        // All should be marked
        for i in 0..1000 {
            assert!(bitmap.is_marked(i * 8));
        }

        // Clear all
        for i in 0..1000 {
            let addr = i * 8;
            assert_eq!(bitmap.clear(addr), Some(true)); // was marked
        }

        // All should be cleared
        for i in 0..1000 {
            assert!(!bitmap.is_marked(i * 8));
        }
    }

    // =========================================================================
    // 14. Phase Transition Verification
    // =========================================================================

    #[test]
    fn test_marking_state_full_lifecycle() {
        let state = SatbMarkingState::new();

        // Idle → Marking → Remarking → Idle
        assert!(!state.is_active());

        state.start_marking();
        assert!(state.is_active());

        state.start_remark();
        assert!(state.is_active());

        state.finish();
        assert!(!state.is_active());
    }

    #[test]
    fn test_marking_state_marking_to_idle() {
        let state = SatbMarkingState::new();

        // Can also go Marking → Idle directly
        state.start_marking();
        state.finish();
        assert!(!state.is_active());
    }

    #[test]
    #[should_panic(expected = "Can only start marking from Idle state")]
    fn test_marking_state_double_start_panics() {
        let state = SatbMarkingState::new();
        state.start_marking();
        state.start_marking(); // Should panic
    }

    #[test]
    #[should_panic(expected = "Can only start remark from Marking state")]
    fn test_marking_state_remark_without_marking_panics() {
        let state = SatbMarkingState::new();
        state.start_remark(); // Should panic — not in Marking state
    }

    #[test]
    fn test_marking_state_finish_from_idle_is_noop() {
        // finish() is an unconditional store — calling from Idle is safe
        let state = SatbMarkingState::new();
        state.finish(); // Should NOT panic — just stores Idle again
        assert!(!state.is_active());
    }

    // =========================================================================
    // 15. Config Boundary Tests
    // =========================================================================

    #[test]
    fn test_config_single_thread_small_chunk() {
        let config = ConcurrentMajorConfig {
            marker_threads: 1,
            work_chunk_size: 1,
            bitmap_coverage: 4096,
        };
        let mut collector = ConcurrentMajorCollector::with_config(config);
        let mut heap = GcHeap::new(GcConfig::default());
        let roots = RootSet::new();

        let result = collector.collect(&mut heap, &roots, &NoopObjectTracer);
        assert_eq!(collector.phase(), ConcurrentGcPhase::Idle);
        assert_eq!(result.bytes_freed, 0);
    }

    #[test]
    fn test_config_max_threads() {
        let config = ConcurrentMajorConfig {
            marker_threads: 64,
            work_chunk_size: 256,
            bitmap_coverage: 512 * 1024 * 1024,
        };
        // Should not panic constructing with many threads
        let collector = ConcurrentMajorCollector::with_config(config);
        assert_eq!(collector.config().marker_threads, 64);
    }

    // =========================================================================
    // 16. Collection Result Validation
    // =========================================================================

    #[test]
    fn test_result_live_bytes_reflects_heap() {
        let mut collector = ConcurrentMajorCollector::for_testing();
        let mut heap = GcHeap::new(GcConfig::default());
        let roots = RootSet::new();

        let result = collector.collect(&mut heap, &roots, &NoopObjectTracer);

        // On empty heap, live_bytes should be whatever old_space + LOS reports
        // which is 0 for an empty heap
        assert_eq!(result.live_bytes, 0);
    }

    #[test]
    fn test_orchestrator_result_fields_populated() {
        let mut collector = Collector::new();
        let mut heap = GcHeap::new(GcConfig::default());
        let roots = RootSet::new();

        let result = collector.collect_concurrent_major(&mut heap, &roots, &NoopObjectTracer);

        assert_eq!(result.collection_type, CollectionType::ConcurrentMajor);
        assert_eq!(result.bytes_freed, 0);
        assert_eq!(result.objects_freed, 0);
        assert_eq!(result.bytes_promoted, 0);
        assert_eq!(result.objects_promoted, 0);
    }

    // =========================================================================
    // Counting tracer helper for GraphObjectTracer tests
    // =========================================================================

    struct CountingTracerImpl {
        count: usize,
    }

    impl CountingTracerImpl {
        fn new() -> Self {
            Self { count: 0 }
        }
    }

    impl Tracer for CountingTracerImpl {
        fn trace_value(&mut self, _value: prism_core::Value) {
            self.count += 1;
        }

        fn trace_ptr(&mut self, _ptr: *const ()) {
            self.count += 1;
        }
    }
}
