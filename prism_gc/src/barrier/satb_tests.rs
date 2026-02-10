//! Tests for SATB buffer infrastructure.

use super::satb_buffer::*;
use std::sync::Arc;

// =============================================================================
// SatbMarkingState Tests
// =============================================================================

#[test]
fn test_initial_state_is_idle() {
    let state = SatbMarkingState::new();
    assert_eq!(state.phase(), MarkingPhase::Idle);
    assert!(!state.is_active());
}

#[test]
fn test_start_marking_transition() {
    let state = SatbMarkingState::new();
    state.start_marking();
    assert_eq!(state.phase(), MarkingPhase::Marking);
    assert!(state.is_active());
}

#[test]
fn test_start_remark_transition() {
    let state = SatbMarkingState::new();
    state.start_marking();
    state.start_remark();
    assert_eq!(state.phase(), MarkingPhase::Remarking);
    assert!(state.is_active());
}

#[test]
fn test_finish_from_marking() {
    let state = SatbMarkingState::new();
    state.start_marking();
    state.finish();
    assert_eq!(state.phase(), MarkingPhase::Idle);
    assert!(!state.is_active());
}

#[test]
fn test_finish_from_remarking() {
    let state = SatbMarkingState::new();
    state.start_marking();
    state.start_remark();
    state.finish();
    assert_eq!(state.phase(), MarkingPhase::Idle);
    assert!(!state.is_active());
}

#[test]
fn test_full_cycle() {
    let state = SatbMarkingState::new();

    // Cycle 1
    assert!(!state.is_active());
    state.start_marking();
    assert!(state.is_active());
    state.start_remark();
    assert!(state.is_active());
    state.finish();
    assert!(!state.is_active());

    // Cycle 2 — can restart
    state.start_marking();
    assert!(state.is_active());
    state.finish();
    assert!(!state.is_active());
}

#[test]
fn test_reset() {
    let state = SatbMarkingState::new();
    state.start_marking();
    state.reset();
    assert_eq!(state.phase(), MarkingPhase::Idle);
    assert!(!state.is_active());
}

#[test]
#[should_panic(expected = "Can only start marking from Idle state")]
fn test_double_start_marking_panics() {
    let state = SatbMarkingState::new();
    state.start_marking();
    state.start_marking(); // Should panic
}

#[test]
#[should_panic(expected = "Can only start remark from Marking state")]
fn test_remark_from_idle_panics() {
    let state = SatbMarkingState::new();
    state.start_remark(); // Should panic
}

#[test]
#[should_panic(expected = "Can only start remark from Marking state")]
fn test_double_remark_panics() {
    let state = SatbMarkingState::new();
    state.start_marking();
    state.start_remark();
    state.start_remark(); // Should panic
}

#[test]
fn test_default_trait() {
    let state = SatbMarkingState::default();
    assert_eq!(state.phase(), MarkingPhase::Idle);
}

// =============================================================================
// SatbBuffer Tests
// =============================================================================

#[test]
fn test_buffer_creation() {
    let buffer = SatbBuffer::new();
    assert!(buffer.is_empty());
    assert_eq!(buffer.len(), 0);
    assert!(!buffer.is_full());
}

#[test]
fn test_buffer_push_single() {
    let mut buffer = SatbBuffer::new();
    let ptr = 0x1000 as *const ();
    let needs_flush = buffer.push(ptr);
    assert!(!needs_flush);
    assert_eq!(buffer.len(), 1);
    assert!(!buffer.is_empty());
}

#[test]
fn test_buffer_push_until_full() {
    let mut buffer = SatbBuffer::new();
    let capacity = SatbBuffer::capacity();

    for i in 0..capacity - 1 {
        let ptr = (i * 8 + 0x1000) as *const ();
        let needs_flush = buffer.push(ptr);
        assert!(!needs_flush, "Should not need flush at {}", i);
    }

    // Last push should signal flush
    let ptr = ((capacity - 1) * 8 + 0x1000) as *const ();
    let needs_flush = buffer.push(ptr);
    assert!(needs_flush, "Should need flush when full");
    assert!(buffer.is_full());
    assert_eq!(buffer.len(), capacity);
}

#[test]
fn test_buffer_drain() {
    let mut buffer = SatbBuffer::new();
    let addrs: Vec<*const ()> = (0..10).map(|i| (i * 8 + 0x1000) as *const ()).collect();

    for &ptr in &addrs {
        buffer.push(ptr);
    }

    let drained = buffer.drain();
    assert_eq!(drained.len(), 10);
    assert!(buffer.is_empty());

    // Verify contents
    for (i, &ptr) in drained.iter().enumerate() {
        assert_eq!(ptr, addrs[i]);
    }
}

#[test]
fn test_buffer_drain_reusable() {
    let mut buffer = SatbBuffer::new();

    // First fill + drain
    for i in 0..5 {
        buffer.push((i * 8) as *const ());
    }
    let d1 = buffer.drain();
    assert_eq!(d1.len(), 5);

    // Second fill + drain
    for i in 0..3 {
        buffer.push(((i + 100) * 8) as *const ());
    }
    let d2 = buffer.drain();
    assert_eq!(d2.len(), 3);
}

#[test]
fn test_buffer_iter() {
    let mut buffer = SatbBuffer::new();
    let ptrs: Vec<*const ()> = (0..5).map(|i| (i * 8 + 0x2000) as *const ()).collect();

    for &ptr in &ptrs {
        buffer.push(ptr);
    }

    let collected: Vec<*const ()> = buffer.iter().copied().collect();
    assert_eq!(collected, ptrs);
}

#[test]
fn test_buffer_default_trait() {
    let buffer = SatbBuffer::default();
    assert!(buffer.is_empty());
}

// =============================================================================
// SatbQueue Tests
// =============================================================================

#[test]
fn test_queue_creation() {
    let queue = SatbQueue::new();
    assert!(queue.is_empty());
    assert_eq!(queue.total_entries(), 0);
    assert_eq!(queue.buffer_count(), 0);
}

#[test]
fn test_queue_push_buffer() {
    let queue = SatbQueue::new();
    let buffer = vec![
        0x1000 as *const (),
        0x2000 as *const (),
        0x3000 as *const (),
    ];
    queue.push_buffer(buffer);

    assert!(!queue.is_empty());
    assert_eq!(queue.total_entries(), 3);
    assert_eq!(queue.buffer_count(), 1);
}

#[test]
fn test_queue_push_multiple_buffers() {
    let queue = SatbQueue::new();

    queue.push_buffer(vec![0x1000 as *const (), 0x2000 as *const ()]);
    queue.push_buffer(vec![0x3000 as *const ()]);
    queue.push_buffer(vec![
        0x4000 as *const (),
        0x5000 as *const (),
        0x6000 as *const (),
    ]);

    assert_eq!(queue.buffer_count(), 3);
    assert_eq!(queue.total_entries(), 6);
}

#[test]
fn test_queue_push_empty_buffer_ignored() {
    let queue = SatbQueue::new();
    queue.push_buffer(vec![]);
    assert!(queue.is_empty());
}

#[test]
fn test_queue_drain_all() {
    let queue = SatbQueue::new();
    queue.push_buffer(vec![0x1000 as *const (), 0x2000 as *const ()]);
    queue.push_buffer(vec![0x3000 as *const ()]);

    let drained = queue.drain_all();
    assert_eq!(drained.len(), 3);
    assert!(queue.is_empty());
}

#[test]
fn test_queue_drain_buffers() {
    let queue = SatbQueue::new();
    queue.push_buffer(vec![0x1000 as *const (), 0x2000 as *const ()]);
    queue.push_buffer(vec![0x3000 as *const ()]);

    let buffers = queue.drain_buffers();
    assert_eq!(buffers.len(), 2);
    assert_eq!(buffers[0].len(), 2);
    assert_eq!(buffers[1].len(), 1);
    assert!(queue.is_empty());
}

#[test]
fn test_queue_drain_all_empty() {
    let queue = SatbQueue::new();
    let drained = queue.drain_all();
    assert!(drained.is_empty());
}

#[test]
fn test_queue_default_trait() {
    let queue = SatbQueue::default();
    assert!(queue.is_empty());
}

// =============================================================================
// SatbBuffer + SatbQueue Integration
// =============================================================================

#[test]
fn test_buffer_flush_to_queue() {
    let queue = SatbQueue::new();
    let mut buffer = SatbBuffer::new();

    for i in 0..10 {
        buffer.push((i * 8 + 0x1000) as *const ());
    }

    buffer.flush(&queue);
    assert!(buffer.is_empty());
    assert_eq!(queue.total_entries(), 10);
}

#[test]
fn test_buffer_flush_empty_noop() {
    let queue = SatbQueue::new();
    let mut buffer = SatbBuffer::new();

    buffer.flush(&queue);
    assert!(queue.is_empty()); // No buffer pushed
}

#[test]
fn test_multiple_buffers_flush_to_same_queue() {
    let queue = SatbQueue::new();

    let mut buf1 = SatbBuffer::new();
    let mut buf2 = SatbBuffer::new();

    for i in 0..5 {
        buf1.push((i * 8) as *const ());
    }
    for i in 0..3 {
        buf2.push(((i + 100) * 8) as *const ());
    }

    buf1.flush(&queue);
    buf2.flush(&queue);

    assert_eq!(queue.buffer_count(), 2);
    assert_eq!(queue.total_entries(), 8);
}

// =============================================================================
// SATB Write Barrier Tests
// =============================================================================

#[test]
fn test_barrier_inactive_when_idle() {
    let state = SatbMarkingState::new();
    let mut buffer = SatbBuffer::new();
    let queue = SatbQueue::new();

    // Should be a no-op when idle
    satb_write_barrier(0x1000 as *const (), &state, &mut buffer, &queue);
    assert!(buffer.is_empty());
}

#[test]
fn test_barrier_active_during_marking() {
    let state = SatbMarkingState::new();
    let mut buffer = SatbBuffer::new();
    let queue = SatbQueue::new();

    state.start_marking();

    satb_write_barrier(0x1000 as *const (), &state, &mut buffer, &queue);
    assert_eq!(buffer.len(), 1);
}

#[test]
fn test_barrier_active_during_remarking() {
    let state = SatbMarkingState::new();
    let mut buffer = SatbBuffer::new();
    let queue = SatbQueue::new();

    state.start_marking();
    state.start_remark();

    satb_write_barrier(0x1000 as *const (), &state, &mut buffer, &queue);
    assert_eq!(buffer.len(), 1);
}

#[test]
fn test_barrier_skips_null() {
    let state = SatbMarkingState::new();
    let mut buffer = SatbBuffer::new();
    let queue = SatbQueue::new();

    state.start_marking();

    satb_write_barrier(std::ptr::null(), &state, &mut buffer, &queue);
    assert!(buffer.is_empty());
}

#[test]
fn test_barrier_auto_flush_on_full() {
    let state = SatbMarkingState::new();
    let mut buffer = SatbBuffer::new();
    let queue = SatbQueue::new();

    state.start_marking();

    // Push exactly BUFFER_CAPACITY values through barrier
    for i in 0..SatbBuffer::capacity() {
        satb_write_barrier(((i + 1) * 8) as *const (), &state, &mut buffer, &queue);
    }

    // Buffer should have auto-flushed to queue
    assert!(buffer.is_empty());
    assert_eq!(queue.total_entries(), SatbBuffer::capacity());
}

#[test]
fn test_barrier_deactivation() {
    let state = SatbMarkingState::new();
    let mut buffer = SatbBuffer::new();
    let queue = SatbQueue::new();

    state.start_marking();
    satb_write_barrier(0x1000 as *const (), &state, &mut buffer, &queue);
    assert_eq!(buffer.len(), 1);

    state.finish();
    satb_write_barrier(0x2000 as *const (), &state, &mut buffer, &queue);
    assert_eq!(buffer.len(), 1); // No new entry when idle
}

#[test]
fn test_barrier_unconditional() {
    let mut buffer = SatbBuffer::new();
    let queue = SatbQueue::new();

    satb_capture_unconditional(0x1000 as *const (), &mut buffer, &queue);
    assert_eq!(buffer.len(), 1);

    // Null is still skipped
    satb_capture_unconditional(std::ptr::null(), &mut buffer, &queue);
    assert_eq!(buffer.len(), 1);
}

// =============================================================================
// Concurrent SATB Tests
// =============================================================================

#[test]
fn test_concurrent_state_visibility() {
    let state = Arc::new(SatbMarkingState::new());
    let num_threads = 8;

    state.start_marking();

    let handles: Vec<_> = (0..num_threads)
        .map(|_| {
            let state = Arc::clone(&state);
            std::thread::spawn(move || {
                // All threads should see marking as active
                assert!(state.is_active());
                state.phase()
            })
        })
        .collect();

    for h in handles {
        let phase = h.join().unwrap();
        assert_eq!(phase, MarkingPhase::Marking);
    }
}

#[test]
fn test_concurrent_queue_flush() {
    let queue = Arc::new(SatbQueue::new());
    let num_threads = 8;
    let entries_per_thread = 100;

    let handles: Vec<_> = (0..num_threads)
        .map(|t| {
            let queue = Arc::clone(&queue);
            std::thread::spawn(move || {
                let mut buffer = SatbBuffer::new();
                for i in 0..entries_per_thread {
                    buffer.push(((t * entries_per_thread + i) * 8 + 0x1000) as *const ());
                }
                buffer.flush(&queue);
            })
        })
        .collect();

    for h in handles {
        h.join().unwrap();
    }

    assert_eq!(queue.total_entries(), num_threads * entries_per_thread);
    assert_eq!(queue.buffer_count(), num_threads);
}

#[test]
fn test_concurrent_barrier_with_marking() {
    let state = Arc::new(SatbMarkingState::new());
    let queue = Arc::new(SatbQueue::new());
    let num_threads = 4;
    let ops_per_thread = 200;

    state.start_marking();

    let handles: Vec<_> = (0..num_threads)
        .map(|t| {
            let state = Arc::clone(&state);
            let queue = Arc::clone(&queue);
            std::thread::spawn(move || {
                let mut buffer = SatbBuffer::new();
                for i in 0..ops_per_thread {
                    satb_write_barrier(
                        ((t * ops_per_thread + i) * 8 + 0x1000) as *const (),
                        &state,
                        &mut buffer,
                        &queue,
                    );
                }
                // Flush remaining
                buffer.flush(&queue);
            })
        })
        .collect();

    for h in handles {
        h.join().unwrap();
    }

    assert_eq!(queue.total_entries(), num_threads * ops_per_thread);
}

#[test]
fn test_concurrent_queue_drain_while_pushing() {
    let queue = Arc::new(SatbQueue::new());

    // Pre-populate
    for i in 0..10 {
        queue.push_buffer(vec![(i * 8 + 0x1000) as *const ()]);
    }

    let queue_push = Arc::clone(&queue);
    let pusher = std::thread::spawn(move || {
        for i in 0..100 {
            queue_push.push_buffer(vec![((i + 100) * 8) as *const ()]);
        }
    });

    let queue_drain = Arc::clone(&queue);
    let drainer = std::thread::spawn(move || {
        let mut total_drained = 0;
        for _ in 0..5 {
            let drained = queue_drain.drain_all();
            total_drained += drained.len();
            std::thread::yield_now();
        }
        total_drained
    });

    pusher.join().unwrap();
    let drained = drainer.join().unwrap();

    // Between pusher and drainer, all 110 entries were handled
    let remaining = queue.total_entries();
    assert_eq!(
        drained + remaining,
        110,
        "drained={} remaining={}",
        drained,
        remaining
    );
}

// =============================================================================
// MarkingPhase Enum Tests
// =============================================================================

#[test]
fn test_marking_phase_debug() {
    assert_eq!(format!("{:?}", MarkingPhase::Idle), "Idle");
    assert_eq!(format!("{:?}", MarkingPhase::Marking), "Marking");
    assert_eq!(format!("{:?}", MarkingPhase::Remarking), "Remarking");
}

#[test]
fn test_marking_phase_clone_eq() {
    let phase = MarkingPhase::Marking;
    let cloned = phase;
    assert_eq!(phase, cloned);
    assert_ne!(MarkingPhase::Idle, MarkingPhase::Marking);
}
