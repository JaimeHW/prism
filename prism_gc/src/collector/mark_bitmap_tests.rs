//! Tests for the atomic mark bitmap.

use super::mark_bitmap::*;
use std::sync::Arc;

// =============================================================================
// Construction
// =============================================================================

#[test]
fn test_bitmap_creation() {
    let bm = AtomicMarkBitmap::new(0x1000, 0x10000);
    assert_eq!(bm.base(), 0x1000);
    assert_eq!(bm.granularity(), MIN_OBJECT_ALIGNMENT);
    assert_eq!(bm.marked_count(), 0);
    assert!(bm.is_empty());
}

#[test]
fn test_bitmap_custom_granularity() {
    let bm = AtomicMarkBitmap::with_granularity(0x1000, 0x10000, 16);
    assert_eq!(bm.granularity(), 16);
    // 0x10000 / 16 = 4096 granules → 4096/64 = 64 words
    assert_eq!(bm.capacity(), 4096);
    assert_eq!(bm.word_count(), 64);
}

#[test]
fn test_bitmap_coverage() {
    let bm = AtomicMarkBitmap::new(0x1000, 0x8000);
    // Coverage should be at least the requested size
    assert!(bm.coverage() >= 0x8000);
    assert!(bm.contains(0x1000));
    assert!(bm.contains(0x8FFF));
    assert!(!bm.contains(0x0FFF));
}

#[test]
fn test_bitmap_memory_usage() {
    let bm = AtomicMarkBitmap::new(0, 1024 * 1024); // 1MB
                                                    // 1MB / 8 bytes = 131072 granules → 131072 / 64 = 2048 words → 16KB
    assert_eq!(bm.memory_usage(), 2048 * 8);
}

#[test]
#[should_panic(expected = "Bitmap size must be > 0")]
fn test_bitmap_zero_size_panics() {
    let _ = AtomicMarkBitmap::new(0, 0);
}

#[test]
#[should_panic(expected = "Granularity must be a power of two >= 8")]
fn test_bitmap_invalid_granularity_non_power_of_two() {
    let _ = AtomicMarkBitmap::with_granularity(0, 1024, 7);
}

#[test]
#[should_panic(expected = "Granularity must be a power of two >= 8")]
fn test_bitmap_invalid_granularity_too_small() {
    let _ = AtomicMarkBitmap::with_granularity(0, 1024, 4);
}

// =============================================================================
// Single-Thread Mark / Test
// =============================================================================

#[test]
fn test_mark_single_address() {
    let bm = AtomicMarkBitmap::new(0x1000, 0x10000);
    let addr = 0x1000;

    assert!(!bm.is_marked(addr));
    assert_eq!(bm.mark(addr), Some(true)); // Newly marked
    assert!(bm.is_marked(addr));
    assert_eq!(bm.marked_count(), 1);
}

#[test]
fn test_mark_idempotent() {
    let bm = AtomicMarkBitmap::new(0x1000, 0x10000);
    let addr = 0x1000;

    assert_eq!(bm.mark(addr), Some(true)); // First time → newly marked
    assert_eq!(bm.mark(addr), Some(false)); // Second time → already marked
    assert_eq!(bm.mark(addr), Some(false)); // Third time → still already marked
    assert_eq!(bm.marked_count(), 1); // Count should still be 1
}

#[test]
fn test_mark_multiple_addresses() {
    let bm = AtomicMarkBitmap::new(0x1000, 0x10000);

    // Mark addresses at granularity boundaries
    for i in 0..100 {
        let addr = 0x1000 + i * MIN_OBJECT_ALIGNMENT;
        assert_eq!(bm.mark(addr), Some(true));
    }
    assert_eq!(bm.marked_count(), 100);
}

#[test]
fn test_mark_out_of_range_returns_none() {
    let bm = AtomicMarkBitmap::new(0x1000, 0x1000); // [0x1000, 0x2000)

    assert_eq!(bm.mark(0x0FFF), None); // Below range
    assert_eq!(bm.mark(0x2000), None); // At end (exclusive)
    assert_eq!(bm.mark(0x3000), None); // Above range
    assert_eq!(bm.mark(0), None); // Zero
    assert_eq!(bm.marked_count(), 0);
}

#[test]
fn test_is_marked_out_of_range_returns_false() {
    let bm = AtomicMarkBitmap::new(0x1000, 0x1000);

    assert!(!bm.is_marked(0x0FFF));
    assert!(!bm.is_marked(0x2000));
    assert!(!bm.is_marked(0));
}

#[test]
fn test_mark_different_words() {
    let bm = AtomicMarkBitmap::new(0, 0x10000);

    // Mark one address per word (64 granules per word × 8 bytes = 512 bytes)
    let addrs: Vec<usize> = (0..10).map(|w| w * 64 * MIN_OBJECT_ALIGNMENT).collect();

    for &addr in &addrs {
        assert_eq!(bm.mark(addr), Some(true));
    }
    assert_eq!(bm.marked_count(), 10);

    for &addr in &addrs {
        assert!(bm.is_marked(addr));
    }
}

#[test]
fn test_mark_all_bits_in_word() {
    let bm = AtomicMarkBitmap::new(0, 0x10000);

    // Mark all 64 bits in the first word
    for bit in 0..64 {
        let addr = bit * MIN_OBJECT_ALIGNMENT;
        assert_eq!(bm.mark(addr), Some(true));
    }
    assert_eq!(bm.marked_count(), 64);

    // Verify all marked
    for bit in 0..64 {
        let addr = bit * MIN_OBJECT_ALIGNMENT;
        assert!(bm.is_marked(addr));
    }
}

#[test]
fn test_mark_sparse_across_words() {
    let bm = AtomicMarkBitmap::new(0, 0x100000); // 1MB

    // Mark bit 0 of every word (stride = 64 granules × 8 = 512 bytes)
    let stride = 64 * MIN_OBJECT_ALIGNMENT;
    let count = 100;
    for i in 0..count {
        let addr = i * stride;
        assert_eq!(bm.mark(addr), Some(true));
    }
    assert_eq!(bm.marked_count(), count);
}

// =============================================================================
// Clear Operations
// =============================================================================

#[test]
fn test_clear_all() {
    let bm = AtomicMarkBitmap::new(0x1000, 0x10000);

    // Mark several addresses
    for i in 0..50 {
        bm.mark(0x1000 + i * MIN_OBJECT_ALIGNMENT);
    }
    assert_eq!(bm.marked_count(), 50);

    bm.clear_all();
    assert_eq!(bm.marked_count(), 0);
    assert!(bm.is_empty());
}

#[test]
fn test_clear_single() {
    let bm = AtomicMarkBitmap::new(0x1000, 0x10000);
    let addr = 0x1000;

    bm.mark(addr);
    assert!(bm.is_marked(addr));

    assert_eq!(bm.clear(addr), Some(true)); // Was marked → true
    assert!(!bm.is_marked(addr));
    assert_eq!(bm.marked_count(), 0);
}

#[test]
fn test_clear_unmarked_returns_false() {
    let bm = AtomicMarkBitmap::new(0x1000, 0x10000);
    assert_eq!(bm.clear(0x1000), Some(false)); // Was not marked → false
}

#[test]
fn test_clear_out_of_range_returns_none() {
    let bm = AtomicMarkBitmap::new(0x1000, 0x1000);
    assert_eq!(bm.clear(0x0FFF), None);
    assert_eq!(bm.clear(0x2000), None);
}

#[test]
fn test_mark_after_clear() {
    let bm = AtomicMarkBitmap::new(0x1000, 0x10000);
    let addr = 0x1000;

    bm.mark(addr);
    bm.clear_all();
    assert!(!bm.is_marked(addr));

    // Re-mark should work
    assert_eq!(bm.mark(addr), Some(true));
    assert!(bm.is_marked(addr));
}

// =============================================================================
// Iteration
// =============================================================================

#[test]
fn test_for_each_marked_empty() {
    let bm = AtomicMarkBitmap::new(0x1000, 0x10000);
    let mut count = 0;
    bm.for_each_marked(|_| count += 1);
    assert_eq!(count, 0);
}

#[test]
fn test_for_each_marked_ordering() {
    let bm = AtomicMarkBitmap::new(0, 0x10000);

    // Mark in reverse order
    let addrs = vec![
        80 * MIN_OBJECT_ALIGNMENT,
        16 * MIN_OBJECT_ALIGNMENT,
        200 * MIN_OBJECT_ALIGNMENT,
        0,
    ];
    for &addr in &addrs {
        bm.mark(addr);
    }

    // Iteration should be in ascending order
    let mut visited = Vec::new();
    bm.for_each_marked(|addr| visited.push(addr));

    assert_eq!(visited.len(), 4);
    for i in 1..visited.len() {
        assert!(
            visited[i] > visited[i - 1],
            "Not in ascending order: {} <= {}",
            visited[i],
            visited[i - 1]
        );
    }
}

#[test]
fn test_for_each_marked_covers_all() {
    let bm = AtomicMarkBitmap::new(0, 0x10000);
    let expected: Vec<usize> = (0..200).map(|i| i * MIN_OBJECT_ALIGNMENT).collect();

    for &addr in &expected {
        bm.mark(addr);
    }

    let mut visited = Vec::new();
    bm.for_each_marked(|addr| visited.push(addr));

    assert_eq!(visited, expected);
}

#[test]
fn test_for_each_marked_in_range() {
    let bm = AtomicMarkBitmap::new(0, 0x10000);

    // Mark addresses 0, 8, 16, ..., 792 (100 addresses)
    for i in 0..100 {
        bm.mark(i * MIN_OBJECT_ALIGNMENT);
    }

    // Query range [200, 600) → granules 25..75
    let mut visited = Vec::new();
    bm.for_each_marked_in_range(200, 600, |addr| visited.push(addr));

    // Should get addresses from 200 to 592 (step 8)
    for &addr in &visited {
        assert!(addr >= 200 && addr < 600);
    }
    assert_eq!(visited.len(), 50);
}

#[test]
fn test_for_each_marked_in_range_empty() {
    let bm = AtomicMarkBitmap::new(0, 0x10000);
    bm.mark(0);
    bm.mark(0x1000);

    let mut visited = Vec::new();
    bm.for_each_marked_in_range(0x100, 0x200, |addr| visited.push(addr));
    assert!(visited.is_empty());
}

// =============================================================================
// Edge Cases
// =============================================================================

#[test]
fn test_base_address_not_zero() {
    let base = 0x7FFF_0000_0000usize;
    let bm = AtomicMarkBitmap::new(base, 0x10000);

    let addr = base + 64 * MIN_OBJECT_ALIGNMENT;
    assert_eq!(bm.mark(addr), Some(true));
    assert!(bm.is_marked(addr));
    assert_eq!(bm.marked_count(), 1);
}

#[test]
fn test_boundary_addresses() {
    let bm = AtomicMarkBitmap::new(0x1000, 0x1000); // [0x1000, 0x2000)

    // First valid address
    assert_eq!(bm.mark(0x1000), Some(true));
    assert!(bm.is_marked(0x1000));

    // Last valid address (must be aligned)
    let last_aligned = 0x1000 + ((0x1000 / MIN_OBJECT_ALIGNMENT) - 1) * MIN_OBJECT_ALIGNMENT;
    assert_eq!(bm.mark(last_aligned), Some(true));
    assert!(bm.is_marked(last_aligned));
}

#[test]
fn test_single_granule_bitmap() {
    let bm = AtomicMarkBitmap::new(0, MIN_OBJECT_ALIGNMENT);
    assert_eq!(bm.capacity(), 1);
    assert_eq!(bm.word_count(), 1);

    assert_eq!(bm.mark(0), Some(true));
    assert!(bm.is_marked(0));
    assert_eq!(bm.marked_count(), 1);
}

#[test]
fn test_exactly_one_word() {
    let size = 64 * MIN_OBJECT_ALIGNMENT; // Exactly 64 granules = 1 word
    let bm = AtomicMarkBitmap::new(0, size);
    assert_eq!(bm.word_count(), 1);
    assert_eq!(bm.capacity(), 64);

    // Mark all 64 bits
    for i in 0..64 {
        assert_eq!(bm.mark(i * MIN_OBJECT_ALIGNMENT), Some(true));
    }
    assert_eq!(bm.marked_count(), 64);
}

#[test]
fn test_word_boundary_crossing() {
    let bm = AtomicMarkBitmap::new(0, 0x10000);

    // Mark bit 63 of word 0 and bit 0 of word 1
    let addr_63 = 63 * MIN_OBJECT_ALIGNMENT;
    let addr_64 = 64 * MIN_OBJECT_ALIGNMENT;

    assert_eq!(bm.mark(addr_63), Some(true));
    assert_eq!(bm.mark(addr_64), Some(true));

    assert!(bm.is_marked(addr_63));
    assert!(bm.is_marked(addr_64));
    assert!(!bm.is_marked(62 * MIN_OBJECT_ALIGNMENT));
    assert!(!bm.is_marked(65 * MIN_OBJECT_ALIGNMENT));
}

#[test]
fn test_contains() {
    let bm = AtomicMarkBitmap::new(0x1000, 0x1000);
    assert!(bm.contains(0x1000));
    assert!(bm.contains(0x1FFF));
    assert!(!bm.contains(0x0FFF));
    assert!(!bm.contains(0x2000));
}

#[test]
fn test_debug_format() {
    let bm = AtomicMarkBitmap::new(0x1000, 0x10000);
    bm.mark(0x1000);
    let debug = format!("{:?}", bm);
    assert!(debug.contains("AtomicMarkBitmap"));
    assert!(debug.contains("0x1000"));
    assert!(debug.contains("marked: 1"));
}

// =============================================================================
// Concurrent Stress Tests
// =============================================================================

#[test]
fn test_concurrent_mark_different_addresses() {
    let bm = Arc::new(AtomicMarkBitmap::new(0, 0x100000)); // 1MB
    let num_threads = 8;
    let marks_per_thread = 1000;

    let handles: Vec<_> = (0..num_threads)
        .map(|t| {
            let bm = Arc::clone(&bm);
            std::thread::spawn(move || {
                let mut newly_marked = 0;
                for i in 0..marks_per_thread {
                    // Each thread marks non-overlapping addresses
                    let addr = (t * marks_per_thread + i) * MIN_OBJECT_ALIGNMENT;
                    if let Some(true) = bm.mark(addr) {
                        newly_marked += 1;
                    }
                }
                newly_marked
            })
        })
        .collect();

    let total_new: usize = handles.into_iter().map(|h| h.join().unwrap()).sum();
    assert_eq!(total_new, num_threads * marks_per_thread);
    assert_eq!(bm.marked_count(), num_threads * marks_per_thread);
}

#[test]
fn test_concurrent_mark_same_address() {
    let bm = Arc::new(AtomicMarkBitmap::new(0, 0x10000));
    let num_threads = 16;
    let addr = 0x100;

    let handles: Vec<_> = (0..num_threads)
        .map(|_| {
            let bm = Arc::clone(&bm);
            std::thread::spawn(move || {
                // All threads try to mark the same address
                bm.mark(addr).unwrap_or(false)
            })
        })
        .collect();

    let results: Vec<bool> = handles.into_iter().map(|h| h.join().unwrap()).collect();

    // Exactly one thread should have gotten `true` (newly marked)
    let new_marks: usize = results.iter().filter(|&&r| r).count();
    assert_eq!(new_marks, 1, "Exactly one thread should win the mark race");
    assert_eq!(bm.marked_count(), 1);
}

#[test]
fn test_concurrent_mark_and_read() {
    let bm = Arc::new(AtomicMarkBitmap::new(0, 0x100000));
    let num_markers = 4;
    let num_readers = 4;
    let marks_per_thread = 2000;

    let mut handles: Vec<std::thread::JoinHandle<usize>> = Vec::new();

    // Marker threads
    for t in 0..num_markers {
        let bm = Arc::clone(&bm);
        handles.push(std::thread::spawn(move || {
            let mut count = 0;
            for i in 0..marks_per_thread {
                let addr = (t * marks_per_thread + i) * MIN_OBJECT_ALIGNMENT;
                bm.mark(addr);
                count += 1;
            }
            count
        }));
    }

    // Reader threads
    for t in 0..num_readers {
        let bm = Arc::clone(&bm);
        handles.push(std::thread::spawn(move || {
            let mut found = 0usize;
            for i in 0..marks_per_thread {
                let addr = (t * marks_per_thread + i) * MIN_OBJECT_ALIGNMENT;
                if bm.is_marked(addr) {
                    found += 1;
                }
            }
            found // Just verify no panics, count is non-deterministic
        }));
    }

    for h in handles {
        let _ = h.join().unwrap();
    }

    assert_eq!(bm.marked_count(), num_markers * marks_per_thread);
}

#[test]
fn test_concurrent_mark_contention_on_word() {
    // All threads mark bits within the same word → high contention
    let bm = Arc::new(AtomicMarkBitmap::new(0, 0x10000));
    let num_threads = 64; // One per bit position in the first word

    let handles: Vec<_> = (0..num_threads.min(64))
        .map(|bit| {
            let bm = Arc::clone(&bm);
            std::thread::spawn(move || {
                let addr = bit * MIN_OBJECT_ALIGNMENT;
                bm.mark(addr)
            })
        })
        .collect();

    for h in handles {
        let result = h.join().unwrap();
        assert!(result.is_some()); // All should be in range
    }

    assert_eq!(bm.marked_count(), num_threads.min(64));
}

#[test]
fn test_concurrent_mark_then_clear_all() {
    let bm = Arc::new(AtomicMarkBitmap::new(0, 0x100000));

    // Phase 1: Concurrent marking
    let handles: Vec<_> = (0..4)
        .map(|t| {
            let bm = Arc::clone(&bm);
            std::thread::spawn(move || {
                for i in 0..1000 {
                    bm.mark((t * 1000 + i) * MIN_OBJECT_ALIGNMENT);
                }
            })
        })
        .collect();

    for h in handles {
        h.join().unwrap();
    }
    assert_eq!(bm.marked_count(), 4000);

    // Phase 2: Clear (single-threaded, as spec'd)
    bm.clear_all();
    assert_eq!(bm.marked_count(), 0);
    assert!(bm.is_empty());
}

#[test]
fn test_concurrent_for_each_marked_during_marking() {
    let bm = Arc::new(AtomicMarkBitmap::new(0, 0x100000));

    // Pre-mark some addresses
    for i in 0..500 {
        bm.mark(i * MIN_OBJECT_ALIGNMENT);
    }

    // Concurrent: one thread iterates while another marks more
    let bm_marker = Arc::clone(&bm);
    let marker = std::thread::spawn(move || {
        for i in 500..1000 {
            bm_marker.mark(i * MIN_OBJECT_ALIGNMENT);
        }
    });

    let bm_reader = Arc::clone(&bm);
    let reader = std::thread::spawn(move || {
        let mut count = 0;
        bm_reader.for_each_marked(|_| count += 1);
        count // Non-deterministic, but should see at least the pre-marked 500
    });

    marker.join().unwrap();
    let seen = reader.join().unwrap();

    // Reader should see at least the 500 pre-marked (may see more)
    assert!(seen >= 500, "Should see at least 500, saw {}", seen);
    assert_eq!(bm.marked_count(), 1000);
}

// =============================================================================
// Performance Characteristics
// =============================================================================

#[test]
fn test_bitmap_scales_correctly() {
    // 1GB heap with 8-byte granularity = 128M granules = 2M words = 16MB bitmap
    let bm = AtomicMarkBitmap::new(0, 1024 * 1024 * 1024);
    assert_eq!(bm.capacity(), 1024 * 1024 * 1024 / MIN_OBJECT_ALIGNMENT);
    assert_eq!(
        bm.memory_usage(),
        (1024 * 1024 * 1024 / MIN_OBJECT_ALIGNMENT / 64) * 8
    );
}

#[test]
fn test_has_any_marked() {
    let bm = AtomicMarkBitmap::new(0, 0x10000);
    assert!(!bm.has_any_marked());

    bm.mark(0);
    assert!(bm.has_any_marked());

    bm.clear_all();
    assert!(!bm.has_any_marked());
}
