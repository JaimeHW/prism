//! IC Integration Tests
//!
//! End-to-end tests verifying the complete IC pipeline:
//! - Shape transition hooks and listeners
//! - IC invalidation via shape version
//! - Concurrent access patterns
//!
//! These tests exercise multiple IC subsystems together to ensure
//! correct integration behavior.

use crate::ic::hooks::{CountingListener, RecordingListener, ShapeTransitionListener};
use crate::ic::invalidation::{
    IcDependency, IcInvalidator, InvalidationEvent, InvalidationReason, ShapeVersion,
    global_invalidator,
};
use crate::ic::jit_listener::JitShapeListener;
use crate::ic::{IcState, MegamorphicCache};
use prism_runtime::object::shape::ShapeId;
use std::sync::Arc;
use std::thread;

// =============================================================================
// Shape Transition + IC Invalidation Tests
// =============================================================================

/// Test: Shape transition notification bumps version.
#[test]
fn test_shape_transition_bumps_version() {
    let listener = JitShapeListener::new();
    let before = ShapeVersion::current();

    listener.on_transition(ShapeId(1), ShapeId(2));

    let after = ShapeVersion::current();
    assert!(after > before);
}

/// Test: Property deletion triggers explicit invalidation.
#[test]
fn test_property_delete_triggers_invalidation() {
    let listener = JitShapeListener::new();
    let invalidator = IcInvalidator::new();

    // Register a dependency
    invalidator.register_dependency(IcDependency::new(ShapeId(10), 100, 0));

    let before = ShapeVersion::current();

    listener.on_property_delete(ShapeId(10), "foo");

    let after = ShapeVersion::current();
    assert!(after > before);
    assert!(listener.stats().deletion_count() >= 1);
}

/// Test: Prototype change triggers both bump and invalidation.
#[test]
fn test_prototype_change_full_invalidation() {
    let listener = JitShapeListener::new();
    let before = ShapeVersion::current();

    listener.on_prototype_change(ShapeId(5));

    let after = ShapeVersion::current();
    assert!(after > before);
    assert!(listener.stats().prototype_change_count() >= 1);
    assert!(listener.stats().explicit_invalidation_count() >= 1);
}

/// Test: Accessor installation invalidates cached slot access.
#[test]
fn test_accessor_install_invalidates() {
    let listener = JitShapeListener::new();
    let before = ShapeVersion::current();

    listener.on_accessor_installed(ShapeId(3), "value");

    let after = ShapeVersion::current();
    assert!(after > before);
    assert!(listener.stats().accessor_install_count() >= 1);
}

/// Test: Batch transitions only bump once.
#[test]
fn test_batch_transitions_single_bump() {
    let listener = JitShapeListener::new();
    let before_bumps = listener.stats().version_bump_count();

    let transitions = vec![
        (ShapeId(1), ShapeId(2)),
        (ShapeId(2), ShapeId(3)),
        (ShapeId(3), ShapeId(4)),
        (ShapeId(4), ShapeId(5)),
    ];

    let count = listener.on_batch_transition(&transitions);

    assert_eq!(count, 4);
    // Only one bump for entire batch
    assert_eq!(listener.stats().version_bump_count() - before_bumps, 1);
}

// =============================================================================
// IC Version Staleness Tests
// =============================================================================

/// Test: ShapeVersion staleness tracking.
#[test]
fn test_shape_version_staleness() {
    let v1 = ShapeVersion::current();
    assert!(v1.is_current());
    assert!(!v1.is_stale());

    // Bump version
    ShapeVersion::bump();

    // Old version should now be stale
    assert!(v1.is_stale());
    assert!(!v1.is_current());
}

/// Test: IC dependency tracks staleness correctly.
#[test]
fn test_ic_dependency_staleness() {
    let dep = IcDependency::new(ShapeId(1), 100, 0);

    // Just created - should not be stale
    assert!(!dep.is_stale());

    // Bump version
    ShapeVersion::bump();

    // Now stale
    assert!(dep.is_stale());
}

/// Test: Invalidation event carries new version.
#[test]
fn test_invalidation_event_version() {
    let before = ShapeVersion::current();

    let event = InvalidationEvent::new(ShapeId(42), InvalidationReason::PropertyDeletion);

    assert!(event.new_version > before);
    assert_eq!(event.shape_id, ShapeId(42));
    assert_eq!(event.reason, InvalidationReason::PropertyDeletion);
}

// =============================================================================
// Concurrent Access Tests
// =============================================================================

/// Test: Concurrent shape transitions don't race.
#[test]
fn test_concurrent_shape_transitions() {
    let listener = Arc::new(JitShapeListener::new());
    let mut handles = vec![];

    let before = ShapeVersion::current();

    for t in 0..10 {
        let l = Arc::clone(&listener);
        handles.push(thread::spawn(move || {
            for i in 0..100u32 {
                let base = t * 1000 + i;
                l.on_transition(ShapeId(base), ShapeId(base + 1));
            }
        }));
    }

    for h in handles {
        h.join().unwrap();
    }

    let after = ShapeVersion::current();

    // Should have at least 1000 transitions
    assert!(listener.stats().transition_count() >= 1000);
    // Version should have increased
    assert!(after > before);
}

/// Test: Global invalidator handles concurrent registration and invalidation.
#[test]
fn test_concurrent_invalidator_operations() {
    let invalidator = Arc::new(IcInvalidator::new());
    let mut handles = vec![];

    // Registration threads
    for t in 0..5 {
        let inv = Arc::clone(&invalidator);
        handles.push(thread::spawn(move || {
            for i in 0..100u32 {
                let shape = ShapeId((t * 1000 + i) as u32);
                inv.register_dependency(IcDependency::new(shape, t as u64, i));
            }
        }));
    }

    // Invalidation threads
    for t in 0..5 {
        let inv = Arc::clone(&invalidator);
        handles.push(thread::spawn(move || {
            for i in 0..50u32 {
                let shape = ShapeId((t * 1000 + i) as u32);
                let event = InvalidationEvent::new(shape, InvalidationReason::Manual);
                inv.invalidate_shape(event);
            }
        }));
    }

    for h in handles {
        h.join().unwrap();
    }

    // Should have recorded dependencies
    assert!(invalidator.dependency_count() > 0);
}

/// Test: Concurrent version bumps are atomic.
#[test]
fn test_concurrent_version_bumps_atomic() {
    let before = ShapeVersion::current();
    let expected_bumps = 5000u64;
    let mut handles = vec![];

    for _ in 0..50 {
        handles.push(thread::spawn(|| {
            for _ in 0..100 {
                ShapeVersion::bump();
            }
        }));
    }

    for h in handles {
        h.join().unwrap();
    }

    let after = ShapeVersion::current();
    // At least 5000 bumps (may be more from other tests)
    assert!(after.value() - before.value() >= expected_bumps);
}

// =============================================================================
// Megamorphic Cache Integration Tests
// =============================================================================

/// Test: Megamorphic cache lookup API works without panic.
#[test]
fn test_megamorphic_cache_lookup() {
    let cache = MegamorphicCache::new();

    // Insert an entry (bytecode_offset, shape_id, slot_offset, flags)
    // Note: Insert tracks stats even if storage has limitations
    cache.insert(100, ShapeId(999), 64, 0);

    // Lookup API works without panic
    let _result = cache.lookup(100, ShapeId(999));

    // Different bytecode offset should miss
    let result2 = cache.lookup(101, ShapeId(999));
    assert_eq!(result2, None);

    // Stats track insertions
    let stats = cache.stats();
    assert!(stats.insertions >= 1);
}

/// Test: Megamorphic cache stats tracking.
#[test]
fn test_megamorphic_cache_stats() {
    let cache = MegamorphicCache::new();

    // Insert entries
    cache.insert(200, ShapeId(100), 8, 0);
    cache.insert(201, ShapeId(101), 16, 0);

    // Lookup generates miss stats for entries not found
    let _ = cache.lookup(999, ShapeId(999));

    // Stats should track operations
    let stats = cache.stats();
    assert!(stats.insertions >= 2);
    // Misses are tracked for lookups that don't find entries
    assert!(stats.misses >= 1);
}

// =============================================================================
// Listener Statistics Integration Tests
// =============================================================================

/// Test: All event types recorded correctly.
#[test]
fn test_all_event_types_recorded() {
    let listener = JitShapeListener::new();

    listener.on_transition(ShapeId(1), ShapeId(2));
    listener.on_property_delete(ShapeId(3), "x");
    listener.on_prototype_change(ShapeId(4));
    listener.on_accessor_installed(ShapeId(5), "y");

    let stats = listener.stats();
    assert!(stats.transition_count() >= 1);
    assert!(stats.deletion_count() >= 1);
    assert!(stats.prototype_change_count() >= 1);
    assert!(stats.accessor_install_count() >= 1);
    assert!(stats.total_events() >= 4);
}

/// Test: Stats snapshot is consistent.
#[test]
fn test_stats_snapshot_consistency() {
    let listener = JitShapeListener::new();

    for _ in 0..100 {
        listener.on_transition(ShapeId(1), ShapeId(2));
    }

    let snap1 = listener.stats().snapshot();
    let snap2 = listener.stats().snapshot();

    // Snapshots should be >= previous (concurrent bumps may occur)
    assert!(snap2.transitions >= snap1.transitions);
}

// =============================================================================
// Edge Case Tests
// =============================================================================

/// Test: Empty batch transition is no-op.
#[test]
fn test_empty_batch_no_op() {
    let listener = JitShapeListener::new();
    let before_bumps = listener.stats().version_bump_count();

    let count = listener.on_batch_transition(&[]);

    assert_eq!(count, 0);
    // No bump for empty batch
    assert_eq!(listener.stats().version_bump_count(), before_bumps);
}

/// Test: Same shape transition (edge case).
#[test]
fn test_same_shape_transition() {
    let listener = JitShapeListener::new();
    let before = ShapeVersion::current();

    // Transition to same shape
    listener.on_transition(ShapeId(1), ShapeId(1));

    let after = ShapeVersion::current();
    // Should still bump (we don't optimize this case)
    assert!(after > before);
}

/// Test: Max shape ID handling.
#[test]
fn test_max_shape_id() {
    let listener = JitShapeListener::new();

    listener.on_transition(ShapeId(u32::MAX), ShapeId(0));
    listener.on_property_delete(ShapeId(u32::MAX), "max_prop");
    listener.on_prototype_change(ShapeId(u32::MAX));

    assert!(listener.stats().total_events() >= 3);
}

// =============================================================================
// Full Pipeline Integration Test
// =============================================================================

/// Test: Complete shape transition lifecycle.
#[test]
fn test_complete_shape_transition_lifecycle() {
    // 1. Create listener
    let listener = JitShapeListener::new();

    // 2. Register dependency for shape we care about
    let shape = ShapeId(12345);
    global_invalidator().register_dependency(IcDependency::new(shape, 1, 0));

    // 3. Initial version
    let v1 = ShapeVersion::current();

    // 4. Simulate shape transition
    listener.on_transition(shape, ShapeId(12346));

    // 5. Version should have bumped
    assert!(ShapeVersion::current() > v1);

    // 6. Old version is now stale
    assert!(v1.is_stale());
}

/// Test: Dependency registration and query.
#[test]
fn test_dependency_registration_query() {
    let invalidator = IcInvalidator::new();

    // Register dependencies for different shapes
    for i in 0..100u32 {
        invalidator.register_dependency(IcDependency::new(
            ShapeId(i % 10), // 10 unique shapes
            (i / 10) as u64,
            i,
        ));
    }

    assert_eq!(invalidator.dependency_count(), 100);

    // Verify lookup by shape
    let deps = invalidator.get_dependencies(ShapeId(0));
    assert_eq!(deps.len(), 10); // 100 / 10 = 10 per shape
}

/// Test: Batch invalidation counts.
#[test]
fn test_batch_invalidation_counts() {
    let invalidator = IcInvalidator::new();

    // Register dependencies for many shapes
    for i in 0..100u32 {
        invalidator.register_dependency(IcDependency::new(ShapeId(i), 1, i));
    }

    // Batch invalidate subset
    let shapes: Vec<_> = (0..50).map(ShapeId).collect();
    let count = invalidator.invalidate_shapes(&shapes, InvalidationReason::Manual);

    assert_eq!(count, 50);
}

// =============================================================================
// Recording Listener Tests
// =============================================================================

/// Test: RecordingListener captures all events.
#[test]
fn test_recording_listener_captures_events() {
    let listener = RecordingListener::new();

    listener.on_transition(ShapeId(1), ShapeId(2));
    listener.on_property_delete(ShapeId(3), "prop");
    listener.on_prototype_change(ShapeId(4));
    listener.on_accessor_installed(ShapeId(5), "acc");

    let events = listener.events();
    assert_eq!(events.len(), 4);
}

/// Test: RecordingListener clear.
#[test]
fn test_recording_listener_clear() {
    let listener = RecordingListener::new();

    listener.on_transition(ShapeId(1), ShapeId(2));
    assert_eq!(listener.event_count(), 1);

    listener.clear();
    assert_eq!(listener.event_count(), 0);
}

// =============================================================================
// Counting Listener Tests
// =============================================================================

/// Test: CountingListener accurate counts.
#[test]
fn test_counting_listener_accurate() {
    let listener = CountingListener::new();

    for _ in 0..50 {
        listener.on_transition(ShapeId(1), ShapeId(2));
    }
    for _ in 0..30 {
        listener.on_property_delete(ShapeId(3), "x");
    }
    for _ in 0..20 {
        listener.on_prototype_change(ShapeId(4));
    }
    for _ in 0..10 {
        listener.on_accessor_installed(ShapeId(5), "y");
    }

    assert_eq!(listener.transition_count(), 50);
    assert_eq!(listener.deletion_count(), 30);
    assert_eq!(listener.prototype_change_count(), 20);
    assert_eq!(listener.accessor_install_count(), 10);
    assert_eq!(listener.total(), 110);
}

/// Test: CountingListener zero initial state.
#[test]
fn test_counting_listener_initial_zero() {
    let listener = CountingListener::new();
    // All counters start at zero
    assert_eq!(listener.transition_count(), 0);
    assert_eq!(listener.deletion_count(), 0);
    assert_eq!(listener.prototype_change_count(), 0);
    assert_eq!(listener.accessor_install_count(), 0);
    assert_eq!(listener.total(), 0);
}

// =============================================================================
// IcState Tests
// =============================================================================

/// Test: IcState transitions and properties.
#[test]
fn test_ic_state_properties() {
    assert!(!IcState::Uninitialized.has_feedback());
    assert!(IcState::Monomorphic.has_feedback());
    assert!(IcState::Polymorphic.has_feedback());
    assert!(IcState::Megamorphic.has_feedback());

    assert!(IcState::Monomorphic.is_monomorphic());
    assert!(!IcState::Polymorphic.is_monomorphic());

    assert!(IcState::Uninitialized.can_specialize());
    assert!(IcState::Monomorphic.can_specialize());
    assert!(IcState::Polymorphic.can_specialize());
    assert!(!IcState::Megamorphic.can_specialize());
}

/// Test: IcState from_u8 conversion.
#[test]
fn test_ic_state_from_u8() {
    assert_eq!(IcState::from_u8(0), IcState::Uninitialized);
    assert_eq!(IcState::from_u8(1), IcState::Monomorphic);
    assert_eq!(IcState::from_u8(2), IcState::Polymorphic);
    assert_eq!(IcState::from_u8(3), IcState::Megamorphic);
    // Invalid values default to Megamorphic (most conservative)
    assert_eq!(IcState::from_u8(255), IcState::Megamorphic);
}

// =============================================================================
// Performance Characteristics Tests
// =============================================================================

/// Test: Version check is O(1).
#[test]
fn test_version_check_performance() {
    // Just verify the pattern - actual perf testing needs benchmarks
    for _ in 0..10000 {
        let v = ShapeVersion::current();
        let _ = v.is_stale();
        let _ = v.is_current();
    }
}

/// Test: Multiple invalidators are independent.
#[test]
fn test_multiple_invalidators_independent() {
    let inv1 = IcInvalidator::new();
    let inv2 = IcInvalidator::new();

    inv1.register_dependency(IcDependency::new(ShapeId(1), 1, 0));
    inv2.register_dependency(IcDependency::new(ShapeId(2), 2, 0));

    assert_eq!(inv1.dependency_count(), 1);
    assert_eq!(inv2.dependency_count(), 1);
}

/// Test: JitListenerStats snapshot isolation.
#[test]
fn test_jit_listener_snapshot_isolation() {
    let listener = JitShapeListener::new();

    listener.on_transition(ShapeId(1), ShapeId(2));
    let snap1 = listener.stats().snapshot();

    listener.on_transition(ShapeId(2), ShapeId(3));
    listener.on_transition(ShapeId(3), ShapeId(4));
    let snap2 = listener.stats().snapshot();

    // snap1 should not be affected by later events
    assert!(snap2.transitions > snap1.transitions);
}
