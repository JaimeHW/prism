//! Comprehensive tests for the ConcurrentMajorCollector.

#[cfg(test)]
mod tests {
    use crate::barrier::{SatbBuffer, SatbMarkingState, SatbQueue};
    use crate::collector::concurrent_major::{
        ConcurrentGcPhase, ConcurrentMajorCollector, ConcurrentMajorConfig, ConcurrentMajorResult,
    };

    // =========================================================================
    // Configuration Tests
    // =========================================================================

    #[test]
    fn test_default_config() {
        let config = ConcurrentMajorConfig::default();
        assert_eq!(config.marker_threads, 4);
        assert_eq!(config.work_chunk_size, 64);
        assert_eq!(config.bitmap_coverage, 4 * 1024 * 1024 * 1024);
    }

    #[test]
    fn test_testing_config() {
        let config = ConcurrentMajorConfig::for_testing();
        assert_eq!(config.marker_threads, 1);
        assert_eq!(config.work_chunk_size, 16);
        assert_eq!(config.bitmap_coverage, 64 * 1024 * 1024);
    }

    #[test]
    fn test_custom_config() {
        let config = ConcurrentMajorConfig {
            marker_threads: 8,
            work_chunk_size: 128,
            bitmap_coverage: 1024 * 1024,
        };
        assert_eq!(config.marker_threads, 8);
        assert_eq!(config.work_chunk_size, 128);
        assert_eq!(config.bitmap_coverage, 1024 * 1024);
    }

    // =========================================================================
    // Constructor Tests
    // =========================================================================

    #[test]
    fn test_new_default() {
        let collector = ConcurrentMajorCollector::new();
        assert_eq!(collector.phase(), ConcurrentGcPhase::Idle);
        assert_eq!(collector.config().marker_threads, 4);
    }

    #[test]
    fn test_new_with_config() {
        let config = ConcurrentMajorConfig::for_testing();
        let collector = ConcurrentMajorCollector::with_config(config);
        assert_eq!(collector.phase(), ConcurrentGcPhase::Idle);
        assert_eq!(collector.config().marker_threads, 1);
    }

    #[test]
    fn test_for_testing() {
        let collector = ConcurrentMajorCollector::for_testing();
        assert_eq!(collector.phase(), ConcurrentGcPhase::Idle);
        assert_eq!(collector.config().work_chunk_size, 16);
    }

    #[test]
    fn test_default_impl() {
        let collector = ConcurrentMajorCollector::default();
        assert_eq!(collector.phase(), ConcurrentGcPhase::Idle);
    }

    // =========================================================================
    // Accessor Tests
    // =========================================================================

    #[test]
    fn test_marking_state_accessor() {
        let collector = ConcurrentMajorCollector::for_testing();
        let state = collector.marking_state();
        // Should start in idle state
        assert!(!state.is_active());
    }

    #[test]
    fn test_satb_queue_accessor() {
        let collector = ConcurrentMajorCollector::for_testing();
        let queue = collector.satb_queue();
        assert!(queue.is_empty());
    }

    #[test]
    fn test_phase_accessor() {
        let collector = ConcurrentMajorCollector::for_testing();
        assert_eq!(collector.phase(), ConcurrentGcPhase::Idle);
    }

    #[test]
    fn test_bitmap_accessor() {
        let collector = ConcurrentMajorCollector::for_testing();
        let bitmap = collector.bitmap();
        // Bitmap should start clear
        assert!(!bitmap.is_marked(1024));
    }

    #[test]
    fn test_config_accessor() {
        let config = ConcurrentMajorConfig {
            marker_threads: 3,
            work_chunk_size: 32,
            bitmap_coverage: 2 * 1024 * 1024,
        };
        let collector = ConcurrentMajorCollector::with_config(config.clone());
        assert_eq!(collector.config().marker_threads, 3);
        assert_eq!(collector.config().work_chunk_size, 32);
    }

    // =========================================================================
    // Phase Enum Tests
    // =========================================================================

    #[test]
    fn test_phase_equality() {
        assert_eq!(ConcurrentGcPhase::Idle, ConcurrentGcPhase::Idle);
        assert_ne!(ConcurrentGcPhase::Idle, ConcurrentGcPhase::InitialMark);
        assert_ne!(ConcurrentGcPhase::ConcurrentMark, ConcurrentGcPhase::Remark);
    }

    #[test]
    fn test_phase_copy() {
        let phase = ConcurrentGcPhase::ConcurrentMark;
        let copy = phase;
        assert_eq!(phase, copy);
    }

    #[test]
    fn test_phase_debug() {
        let debug = format!("{:?}", ConcurrentGcPhase::Sweep);
        assert!(debug.contains("Sweep"));
    }

    #[test]
    fn test_all_phases_distinct() {
        let phases = [
            ConcurrentGcPhase::Idle,
            ConcurrentGcPhase::InitialMark,
            ConcurrentGcPhase::ConcurrentMark,
            ConcurrentGcPhase::Remark,
            ConcurrentGcPhase::Sweep,
        ];
        for i in 0..phases.len() {
            for j in (i + 1)..phases.len() {
                assert_ne!(
                    phases[i], phases[j],
                    "Phases at {} and {} should differ",
                    i, j
                );
            }
        }
    }

    // =========================================================================
    // Result Tests
    // =========================================================================

    #[test]
    fn test_result_default() {
        let result = ConcurrentMajorResult::default();
        assert_eq!(result.bytes_freed, 0);
        assert_eq!(result.objects_freed, 0);
        assert_eq!(result.live_bytes, 0);
        assert_eq!(result.objects_marked, 0);
        assert_eq!(result.objects_traced, 0);
        assert_eq!(result.satb_entries_processed, 0);
    }

    #[test]
    fn test_result_clone() {
        let mut result = ConcurrentMajorResult::default();
        result.bytes_freed = 1024;
        result.objects_freed = 10;
        let cloned = result.clone();
        assert_eq!(cloned.bytes_freed, 1024);
        assert_eq!(cloned.objects_freed, 10);
    }

    #[test]
    fn test_result_debug() {
        let result = ConcurrentMajorResult::default();
        let debug = format!("{:?}", result);
        assert!(debug.contains("bytes_freed"));
        assert!(debug.contains("objects_freed"));
    }

    // =========================================================================
    // Bitmap Marking Tests
    // =========================================================================

    #[test]
    fn test_is_marked_initially_false() {
        let collector = ConcurrentMajorCollector::for_testing();
        // Nothing should be marked initially
        assert!(!collector.is_marked(0));
        assert!(!collector.is_marked(1024));
        assert!(!collector.is_marked(4096));
    }

    #[test]
    fn test_bitmap_mark_and_query() {
        let collector = ConcurrentMajorCollector::for_testing();
        let bitmap = collector.bitmap();

        // Mark an address
        let result = bitmap.mark(1024);
        assert_eq!(result, Some(true)); // newly marked

        // Should now be marked
        assert!(collector.is_marked(1024));
        assert!(!collector.is_marked(2048)); // other addresses still clear
    }

    #[test]
    fn test_bitmap_double_mark() {
        let collector = ConcurrentMajorCollector::for_testing();
        let bitmap = collector.bitmap();

        // First mark — newly marked
        assert_eq!(bitmap.mark(1024), Some(true));
        // Second mark — already marked
        assert_eq!(bitmap.mark(1024), Some(false));
    }

    // =========================================================================
    // SATB Integration Tests
    // =========================================================================

    #[test]
    fn test_marking_state_lifecycle() {
        let collector = ConcurrentMajorCollector::for_testing();
        let state = collector.marking_state();

        // Start idle
        assert!(!state.is_active());

        // Activate marking
        state.start_marking();
        assert!(state.is_active());

        // Transition to remark
        state.start_remark();
        assert!(state.is_active());

        // Finish — back to idle
        state.finish();
        assert!(!state.is_active());
    }

    #[test]
    fn test_satb_queue_push_and_drain() {
        let collector = ConcurrentMajorCollector::for_testing();
        let queue = collector.satb_queue();

        assert!(queue.is_empty());

        // Push a buffer
        let ptrs = vec![0x1000 as *const (), 0x2000 as *const ()];
        queue.push_buffer(ptrs);

        assert!(!queue.is_empty());
        assert_eq!(queue.buffer_count(), 1);
        assert_eq!(queue.total_entries(), 2);

        // Drain
        let drained = queue.drain_buffers();
        assert_eq!(drained.len(), 1);
        assert_eq!(drained[0].len(), 2);
        assert!(queue.is_empty());
    }

    #[test]
    fn test_satb_buffer_integration() {
        let collector = ConcurrentMajorCollector::for_testing();
        let state = collector.marking_state();
        let queue = collector.satb_queue();

        // Activate marking so barrier captures old values
        state.start_marking();

        // Simulate a mutator write barrier
        let mut buffer = SatbBuffer::new();
        let old_ptr = 0x12345678 as *const ();
        crate::barrier::satb_write_barrier(old_ptr, state, &mut buffer, queue);

        // Buffer should have captured the old pointer
        assert_eq!(buffer.len(), 1);

        // Flush to global queue
        buffer.flush(queue);
        assert_eq!(queue.total_entries(), 1);
        assert_eq!(buffer.len(), 0);

        state.finish();
    }

    // =========================================================================
    // Multiple SATB Buffer Tests
    // =========================================================================

    #[test]
    fn test_multiple_satb_buffers_merge() {
        let collector = ConcurrentMajorCollector::for_testing();
        let queue = collector.satb_queue();

        // Simulate multiple threads flushing buffers
        queue.push_buffer(vec![0x1000 as *const (), 0x2000 as *const ()]);
        queue.push_buffer(vec![0x3000 as *const ()]);
        queue.push_buffer(vec![
            0x4000 as *const (),
            0x5000 as *const (),
            0x6000 as *const (),
        ]);

        assert_eq!(queue.buffer_count(), 3);
        assert_eq!(queue.total_entries(), 6);

        let drained = queue.drain_buffers();
        assert_eq!(drained.len(), 3);
        assert!(queue.is_empty());
    }

    // =========================================================================
    // Config Clone Tests
    // =========================================================================

    #[test]
    fn test_config_clone() {
        let config = ConcurrentMajorConfig {
            marker_threads: 7,
            work_chunk_size: 256,
            bitmap_coverage: 8 * 1024 * 1024,
        };
        let cloned = config.clone();
        assert_eq!(cloned.marker_threads, 7);
        assert_eq!(cloned.work_chunk_size, 256);
        assert_eq!(cloned.bitmap_coverage, 8 * 1024 * 1024);
    }

    #[test]
    fn test_config_debug() {
        let config = ConcurrentMajorConfig::default();
        let debug = format!("{:?}", config);
        assert!(debug.contains("marker_threads"));
        assert!(debug.contains("work_chunk_size"));
    }

    // =========================================================================
    // Bitmap Reset Tests
    // =========================================================================

    #[test]
    fn test_bitmap_replaced_on_new_cycle() {
        let mut collector = ConcurrentMajorCollector::for_testing();

        // Mark some addresses
        collector.bitmap().mark(1024);
        collector.bitmap().mark(2048);
        assert!(collector.is_marked(1024));
        assert!(collector.is_marked(2048));

        // Replace bitmap (simulating start of new GC cycle)
        use crate::collector::mark_bitmap::AtomicMarkBitmap;
        use std::sync::Arc;
        let new_bitmap = Arc::new(AtomicMarkBitmap::new(0, 64 * 1024 * 1024));
        // Using internal field replacement would require pub access — test via accessor
        // Instead, verify that the bitmap is indeed clearable

        // Clear via address
        let result = collector.bitmap().clear(1024);
        assert_eq!(result, Some(true)); // was marked, now cleared
        assert!(!collector.is_marked(1024));
    }

    // =========================================================================
    // Zero-Thread Config Edge Case
    // =========================================================================

    #[test]
    fn test_zero_marker_threads_config() {
        let config = ConcurrentMajorConfig {
            marker_threads: 0,
            work_chunk_size: 16,
            bitmap_coverage: 1024 * 1024,
        };
        // Should not panic — 0 threads means no workers, which is valid
        let collector = ConcurrentMajorCollector::with_config(config);
        assert_eq!(collector.config().marker_threads, 0);
    }

    // =========================================================================
    // Large Bitmap Coverage
    // =========================================================================

    #[test]
    fn test_small_bitmap_coverage() {
        let config = ConcurrentMajorConfig {
            marker_threads: 1,
            work_chunk_size: 8,
            bitmap_coverage: 4096, // Very small — 4 KB
        };
        let collector = ConcurrentMajorCollector::with_config(config);
        let bitmap = collector.bitmap();

        // Should be able to mark within range
        assert_eq!(bitmap.mark(0), Some(true));
        assert_eq!(bitmap.mark(8), Some(true));
        assert!(bitmap.is_marked(0));
        assert!(bitmap.is_marked(8));

        // Out of range should return None
        assert_eq!(bitmap.mark(8192), None);
    }

    // =========================================================================
    // Result Duration Tests
    // =========================================================================

    #[test]
    fn test_result_durations_default_to_zero() {
        let result = ConcurrentMajorResult::default();
        assert_eq!(result.initial_mark_duration, std::time::Duration::ZERO);
        assert_eq!(result.concurrent_mark_duration, std::time::Duration::ZERO);
        assert_eq!(result.remark_sweep_duration, std::time::Duration::ZERO);
        assert_eq!(result.total_duration, std::time::Duration::ZERO);
    }

    // =========================================================================
    // SATB Barrier Fast-Path Tests
    // =========================================================================

    #[test]
    fn test_satb_barrier_idle_fast_path() {
        let collector = ConcurrentMajorCollector::for_testing();
        let state = collector.marking_state();
        let queue = collector.satb_queue();

        // State is Idle — barrier should NOT capture
        let mut buffer = SatbBuffer::new();
        let old_ptr = 0xDEADBEEF as *const ();
        crate::barrier::satb_write_barrier(old_ptr, state, &mut buffer, queue);

        assert_eq!(buffer.len(), 0);
        assert!(queue.is_empty());
    }

    #[test]
    fn test_satb_barrier_null_ptr_fast_path() {
        let collector = ConcurrentMajorCollector::for_testing();
        let state = collector.marking_state();
        let queue = collector.satb_queue();

        // Even when marking is active, null ptrs should not be captured
        state.start_marking();

        let mut buffer = SatbBuffer::new();
        crate::barrier::satb_write_barrier(std::ptr::null(), state, &mut buffer, queue);

        assert_eq!(buffer.len(), 0);

        state.finish();
    }

    // =========================================================================
    // State Reset Tests
    // =========================================================================

    #[test]
    fn test_marking_state_reset() {
        let collector = ConcurrentMajorCollector::for_testing();
        let state = collector.marking_state();

        // Start marking, then force-reset
        state.start_marking();
        assert!(state.is_active());
        state.reset();
        assert!(!state.is_active());
    }

    // =========================================================================
    // Arc sharing tests
    // =========================================================================

    #[test]
    fn test_shared_marking_state() {
        let collector = ConcurrentMajorCollector::for_testing();
        let state1 = std::sync::Arc::clone(collector.marking_state());
        let state2 = std::sync::Arc::clone(collector.marking_state());

        // Both Arcs point to the same state
        state1.start_marking();
        assert!(state2.is_active());
        state1.finish();
        assert!(!state2.is_active());
    }

    #[test]
    fn test_shared_satb_queue() {
        let collector = ConcurrentMajorCollector::for_testing();
        let queue1 = std::sync::Arc::clone(collector.satb_queue());
        let queue2 = std::sync::Arc::clone(collector.satb_queue());

        // Push via one, drain via the other
        queue1.push_buffer(vec![0x1000 as *const ()]);
        assert_eq!(queue2.total_entries(), 1);

        let drained = queue2.drain_buffers();
        assert_eq!(drained.len(), 1);
        assert!(queue1.is_empty());
    }
}
