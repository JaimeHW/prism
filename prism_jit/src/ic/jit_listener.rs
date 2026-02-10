//! JIT Shape Transition Listener
//!
//! Implements `ShapeTransitionListener` for the JIT compiler.
//! Reacts to shape changes by bumping the global shape version (lazy invalidation)
//! and explicitly invalidating IC sites for destructive changes.
//!
//! # Architecture
//!
//! The listener uses a two-tier invalidation strategy:
//!
//! 1. **Lazy invalidation** (shape transitions): Bump global version, ICs check on access
//! 2. **Explicit invalidation** (deletions, prototype changes): Mark specific shapes invalid
//!
//! This balances performance (minimal work on common transitions) with correctness
//! (guaranteed invalidation for destructive changes).
//!
//! # Thread Safety
//!
//! The listener is fully thread-safe:
//! - `ShapeVersion::bump()` is atomic
//! - `IcInvalidator` uses internal locking
//! - Statistics use atomic counters

use crate::ic::hooks::ShapeTransitionListener;
use crate::ic::invalidation::{
    InvalidationEvent, InvalidationReason, ShapeVersion, global_invalidator,
};
use prism_runtime::object::shape::ShapeId;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

// =============================================================================
// JIT Shape Listener
// =============================================================================

/// JIT compiler's shape transition listener.
///
/// This is the production implementation of `ShapeTransitionListener`.
/// It integrates with the IC invalidation system to maintain cache validity.
///
/// # Strategy
///
/// - **Transitions**: Bump global version (O(1), lazy check on IC access)
/// - **Property deletions**: Explicit invalidation of affected shapes
/// - **Prototype changes**: Explicit invalidation + version bump
/// - **Accessor installs**: Explicit invalidation of affected shapes
#[derive(Debug)]
pub struct JitShapeListener {
    /// Statistics for monitoring
    stats: JitListenerStats,
    /// Whether to also invalidate explicitly on transitions
    /// (more aggressive, useful for debugging)
    explicit_on_transition: AtomicBool,
}

impl JitShapeListener {
    /// Create a new JIT shape listener.
    pub fn new() -> Self {
        Self {
            stats: JitListenerStats::new(),
            explicit_on_transition: AtomicBool::new(false),
        }
    }

    /// Create a listener with explicit invalidation on all events.
    ///
    /// This is more aggressive than the default lazy strategy.
    /// Useful for debugging or when lazy invalidation causes issues.
    pub fn with_explicit_invalidation() -> Self {
        Self {
            stats: JitListenerStats::new(),
            explicit_on_transition: AtomicBool::new(true),
        }
    }

    /// Enable or disable explicit invalidation on transitions.
    pub fn set_explicit_on_transition(&self, enabled: bool) {
        self.explicit_on_transition
            .store(enabled, Ordering::Relaxed);
    }

    /// Check if explicit invalidation on transition is enabled.
    pub fn explicit_on_transition(&self) -> bool {
        self.explicit_on_transition.load(Ordering::Relaxed)
    }

    /// Get statistics.
    pub fn stats(&self) -> &JitListenerStats {
        &self.stats
    }
}

impl Default for JitShapeListener {
    fn default() -> Self {
        Self::new()
    }
}

impl ShapeTransitionListener for JitShapeListener {
    /// Handle shape transition.
    ///
    /// Default strategy: bump global version for lazy invalidation.
    /// If `explicit_on_transition` is true, also invalidates explicitly.
    fn on_transition(&self, old_shape: ShapeId, _new_shape: ShapeId) {
        self.stats.record_transition();

        // Always bump version for lazy invalidation
        ShapeVersion::bump();
        self.stats.record_version_bump();

        // Optionally do explicit invalidation too
        if self.explicit_on_transition.load(Ordering::Relaxed) {
            let event = InvalidationEvent::new(old_shape, InvalidationReason::ShapeTransition);
            let count = global_invalidator().invalidate_shape(event);
            self.stats.record_explicit_invalidation(count as u64);
        }
    }

    /// Handle property deletion.
    ///
    /// Property deletions are destructive - cached slot offsets become invalid.
    /// We do both version bump AND explicit invalidation for safety.
    fn on_property_delete(&self, shape: ShapeId, _property: &str) {
        self.stats.record_deletion();

        // Bump version
        ShapeVersion::bump();
        self.stats.record_version_bump();

        // Explicit invalidation for this shape
        let event = InvalidationEvent::new(shape, InvalidationReason::PropertyDeletion);
        let count = global_invalidator().invalidate_shape(event);
        self.stats.record_explicit_invalidation(count as u64);
    }

    /// Handle prototype change.
    ///
    /// Prototype changes affect method resolution and may invalidate
    /// cached call targets. We do both version bump AND explicit invalidation.
    fn on_prototype_change(&self, shape: ShapeId) {
        self.stats.record_prototype_change();

        // Bump version
        ShapeVersion::bump();
        self.stats.record_version_bump();

        // Explicit invalidation for this shape
        let event = InvalidationEvent::new(shape, InvalidationReason::PrototypeChange);
        let count = global_invalidator().invalidate_shape(event);
        self.stats.record_explicit_invalidation(count as u64);
    }

    /// Handle accessor installation.
    ///
    /// Accessor installation changes how property access works.
    /// Direct slot loads become invalid - must call getter/setter.
    fn on_accessor_installed(&self, shape: ShapeId, _property: &str) {
        self.stats.record_accessor_install();

        // Bump version
        ShapeVersion::bump();
        self.stats.record_version_bump();

        // Explicit invalidation for this shape
        let event = InvalidationEvent::new(shape, InvalidationReason::AccessorInstalled);
        let count = global_invalidator().invalidate_shape(event);
        self.stats.record_explicit_invalidation(count as u64);
    }

    /// Handle batch transitions.
    ///
    /// Optimized to do a single version bump for all transitions.
    fn on_batch_transition(&self, transitions: &[(ShapeId, ShapeId)]) -> usize {
        if transitions.is_empty() {
            return 0;
        }

        self.stats.record_batch_transition(transitions.len() as u64);

        // Single version bump for all
        ShapeVersion::bump();
        self.stats.record_version_bump();

        // If explicit mode, invalidate all old shapes
        if self.explicit_on_transition.load(Ordering::Relaxed) {
            let old_shapes: Vec<ShapeId> = transitions.iter().map(|(old, _)| *old).collect();
            let count = global_invalidator()
                .invalidate_shapes(&old_shapes, InvalidationReason::ShapeTransition);
            self.stats.record_explicit_invalidation(count as u64);
        }

        transitions.len()
    }
}

// =============================================================================
// JIT Listener Statistics
// =============================================================================

/// Statistics for the JIT shape listener.
#[derive(Debug)]
pub struct JitListenerStats {
    /// Total transitions received
    transitions: AtomicU64,
    /// Total property deletions received
    deletions: AtomicU64,
    /// Total prototype changes received
    prototype_changes: AtomicU64,
    /// Total accessor installations received
    accessor_installs: AtomicU64,
    /// Total batch transitions received
    batch_transitions: AtomicU64,
    /// Total version bumps performed
    version_bumps: AtomicU64,
    /// Total explicit invalidations performed
    explicit_invalidations: AtomicU64,
    /// Total ICs invalidated
    ics_invalidated: AtomicU64,
}

impl JitListenerStats {
    /// Create new zeroed stats.
    pub const fn new() -> Self {
        Self {
            transitions: AtomicU64::new(0),
            deletions: AtomicU64::new(0),
            prototype_changes: AtomicU64::new(0),
            accessor_installs: AtomicU64::new(0),
            batch_transitions: AtomicU64::new(0),
            version_bumps: AtomicU64::new(0),
            explicit_invalidations: AtomicU64::new(0),
            ics_invalidated: AtomicU64::new(0),
        }
    }

    /// Record a transition.
    #[inline]
    fn record_transition(&self) {
        self.transitions.fetch_add(1, Ordering::Relaxed);
    }

    /// Record a property deletion.
    #[inline]
    fn record_deletion(&self) {
        self.deletions.fetch_add(1, Ordering::Relaxed);
    }

    /// Record a prototype change.
    #[inline]
    fn record_prototype_change(&self) {
        self.prototype_changes.fetch_add(1, Ordering::Relaxed);
    }

    /// Record an accessor installation.
    #[inline]
    fn record_accessor_install(&self) {
        self.accessor_installs.fetch_add(1, Ordering::Relaxed);
    }

    /// Record batch transitions.
    #[inline]
    fn record_batch_transition(&self, count: u64) {
        self.batch_transitions.fetch_add(count, Ordering::Relaxed);
    }

    /// Record a version bump.
    #[inline]
    fn record_version_bump(&self) {
        self.version_bumps.fetch_add(1, Ordering::Relaxed);
    }

    /// Record explicit invalidations.
    #[inline]
    fn record_explicit_invalidation(&self, count: u64) {
        self.explicit_invalidations.fetch_add(1, Ordering::Relaxed);
        self.ics_invalidated.fetch_add(count, Ordering::Relaxed);
    }

    /// Get total transitions.
    #[inline]
    pub fn transition_count(&self) -> u64 {
        self.transitions.load(Ordering::Relaxed)
    }

    /// Get total deletions.
    #[inline]
    pub fn deletion_count(&self) -> u64 {
        self.deletions.load(Ordering::Relaxed)
    }

    /// Get total prototype changes.
    #[inline]
    pub fn prototype_change_count(&self) -> u64 {
        self.prototype_changes.load(Ordering::Relaxed)
    }

    /// Get total accessor installations.
    #[inline]
    pub fn accessor_install_count(&self) -> u64 {
        self.accessor_installs.load(Ordering::Relaxed)
    }

    /// Get total batch transitions.
    #[inline]
    pub fn batch_transition_count(&self) -> u64 {
        self.batch_transitions.load(Ordering::Relaxed)
    }

    /// Get total version bumps.
    #[inline]
    pub fn version_bump_count(&self) -> u64 {
        self.version_bumps.load(Ordering::Relaxed)
    }

    /// Get total explicit invalidation calls.
    #[inline]
    pub fn explicit_invalidation_count(&self) -> u64 {
        self.explicit_invalidations.load(Ordering::Relaxed)
    }

    /// Get total ICs invalidated.
    #[inline]
    pub fn ics_invalidated_count(&self) -> u64 {
        self.ics_invalidated.load(Ordering::Relaxed)
    }

    /// Get total events received.
    #[inline]
    pub fn total_events(&self) -> u64 {
        self.transition_count()
            + self.deletion_count()
            + self.prototype_change_count()
            + self.accessor_install_count()
    }

    /// Get a snapshot of all statistics.
    pub fn snapshot(&self) -> JitListenerStatsSnapshot {
        JitListenerStatsSnapshot {
            transitions: self.transition_count(),
            deletions: self.deletion_count(),
            prototype_changes: self.prototype_change_count(),
            accessor_installs: self.accessor_install_count(),
            batch_transitions: self.batch_transition_count(),
            version_bumps: self.version_bump_count(),
            explicit_invalidations: self.explicit_invalidation_count(),
            ics_invalidated: self.ics_invalidated_count(),
        }
    }
}

impl Default for JitListenerStats {
    fn default() -> Self {
        Self::new()
    }
}

/// Immutable snapshot of JIT listener statistics.
#[derive(Debug, Clone, Copy, Default)]
pub struct JitListenerStatsSnapshot {
    /// Total transitions received
    pub transitions: u64,
    /// Total property deletions received
    pub deletions: u64,
    /// Total prototype changes received
    pub prototype_changes: u64,
    /// Total accessor installations received
    pub accessor_installs: u64,
    /// Total batch transitions received
    pub batch_transitions: u64,
    /// Total version bumps performed
    pub version_bumps: u64,
    /// Total explicit invalidation calls
    pub explicit_invalidations: u64,
    /// Total ICs invalidated
    pub ics_invalidated: u64,
}

// =============================================================================
// Global JIT Listener
// =============================================================================

use std::sync::OnceLock;

/// Global JIT listener singleton.
static GLOBAL_JIT_LISTENER: OnceLock<JitShapeListener> = OnceLock::new();

/// Get or create the global JIT shape listener.
///
/// This is the listener that should be registered with `register_shape_listener()`.
#[inline]
pub fn global_jit_listener() -> &'static JitShapeListener {
    GLOBAL_JIT_LISTENER.get_or_init(JitShapeListener::new)
}

/// Initialize and register the JIT shape listener.
///
/// This should be called once at JIT initialization.
/// Returns true if registration succeeded, false if already registered.
pub fn init_jit_listener() -> bool {
    use crate::ic::hooks::register_shape_listener_boxed;

    // Ensure global listener exists
    let _ = global_jit_listener();

    // Create a new listener for registration (since we can't register a static ref)
    register_shape_listener_boxed(Box::new(JitShapeListener::new()))
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ic::invalidation::IcDependency;
    use std::sync::Arc;
    use std::thread;

    // -------------------------------------------------------------------------
    // JitShapeListener Construction Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_jit_listener_new() {
        let listener = JitShapeListener::new();
        assert!(!listener.explicit_on_transition());
        assert_eq!(listener.stats().transition_count(), 0);
    }

    #[test]
    fn test_jit_listener_with_explicit_invalidation() {
        let listener = JitShapeListener::with_explicit_invalidation();
        assert!(listener.explicit_on_transition());
    }

    #[test]
    fn test_jit_listener_default() {
        let listener = JitShapeListener::default();
        assert!(!listener.explicit_on_transition());
    }

    #[test]
    fn test_jit_listener_set_explicit() {
        let listener = JitShapeListener::new();
        assert!(!listener.explicit_on_transition());

        listener.set_explicit_on_transition(true);
        assert!(listener.explicit_on_transition());

        listener.set_explicit_on_transition(false);
        assert!(!listener.explicit_on_transition());
    }

    // -------------------------------------------------------------------------
    // Transition Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_jit_listener_transition_bumps_version() {
        let listener = JitShapeListener::new();
        let before = ShapeVersion::current();

        listener.on_transition(ShapeId(1), ShapeId(2));

        let after = ShapeVersion::current();
        assert!(after > before);
        assert_eq!(listener.stats().transition_count(), 1);
        assert_eq!(listener.stats().version_bump_count(), 1);
    }

    #[test]
    fn test_jit_listener_transition_explicit_mode() {
        let listener = JitShapeListener::with_explicit_invalidation();
        let before = ShapeVersion::current();

        listener.on_transition(ShapeId(1), ShapeId(2));

        let after = ShapeVersion::current();
        assert!(after > before);
        assert_eq!(listener.stats().transition_count(), 1);
        assert_eq!(listener.stats().explicit_invalidation_count(), 1);
    }

    #[test]
    fn test_jit_listener_multiple_transitions() {
        let listener = JitShapeListener::new();

        for i in 0..10 {
            listener.on_transition(ShapeId(i), ShapeId(i + 1));
        }

        assert_eq!(listener.stats().transition_count(), 10);
        assert_eq!(listener.stats().version_bump_count(), 10);
    }

    // -------------------------------------------------------------------------
    // Property Deletion Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_jit_listener_property_delete_bumps_version() {
        let listener = JitShapeListener::new();
        let before = ShapeVersion::current();

        listener.on_property_delete(ShapeId(1), "foo");

        let after = ShapeVersion::current();
        assert!(after > before);
        assert_eq!(listener.stats().deletion_count(), 1);
        assert_eq!(listener.stats().version_bump_count(), 1);
        assert_eq!(listener.stats().explicit_invalidation_count(), 1);
    }

    #[test]
    fn test_jit_listener_property_delete_invalidates() {
        let listener = JitShapeListener::new();

        // Register a dependency to be invalidated
        let dep = IcDependency::new(ShapeId(42), 100, 0);
        global_invalidator().register_dependency(dep);

        listener.on_property_delete(ShapeId(42), "prop");

        assert_eq!(listener.stats().deletion_count(), 1);
        // IC should have been invalidated
        assert!(listener.stats().ics_invalidated_count() >= 1);
    }

    // -------------------------------------------------------------------------
    // Prototype Change Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_jit_listener_prototype_change_bumps_version() {
        let listener = JitShapeListener::new();
        let before = ShapeVersion::current();

        listener.on_prototype_change(ShapeId(1));

        let after = ShapeVersion::current();
        assert!(after > before);
        assert_eq!(listener.stats().prototype_change_count(), 1);
        assert_eq!(listener.stats().version_bump_count(), 1);
        assert_eq!(listener.stats().explicit_invalidation_count(), 1);
    }

    #[test]
    fn test_jit_listener_prototype_change_invalidates() {
        let listener = JitShapeListener::new();

        // Register a dependency
        let dep = IcDependency::new(ShapeId(99), 200, 5);
        global_invalidator().register_dependency(dep);

        listener.on_prototype_change(ShapeId(99));

        assert_eq!(listener.stats().prototype_change_count(), 1);
    }

    // -------------------------------------------------------------------------
    // Accessor Installation Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_jit_listener_accessor_installed_bumps_version() {
        let listener = JitShapeListener::new();
        let before = ShapeVersion::current();

        listener.on_accessor_installed(ShapeId(1), "getValue");

        let after = ShapeVersion::current();
        assert!(after > before);
        assert_eq!(listener.stats().accessor_install_count(), 1);
        assert_eq!(listener.stats().version_bump_count(), 1);
        assert_eq!(listener.stats().explicit_invalidation_count(), 1);
    }

    // -------------------------------------------------------------------------
    // Batch Transition Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_jit_listener_batch_transition_empty() {
        let listener = JitShapeListener::new();
        let before = ShapeVersion::current();

        let count = listener.on_batch_transition(&[]);

        let after = ShapeVersion::current();
        assert_eq!(count, 0);
        // No version bump for empty batch
        assert_eq!(before, after);
    }

    #[test]
    fn test_jit_listener_batch_transition_single_bump() {
        let listener = JitShapeListener::new();
        let before = ShapeVersion::current();

        let transitions = vec![
            (ShapeId(1), ShapeId(2)),
            (ShapeId(2), ShapeId(3)),
            (ShapeId(3), ShapeId(4)),
        ];

        let count = listener.on_batch_transition(&transitions);

        let after = ShapeVersion::current();
        assert_eq!(count, 3);
        // Single version bump for all
        assert_eq!(listener.stats().version_bump_count(), 1);
        assert_eq!(listener.stats().batch_transition_count(), 3);
        assert!(after > before);
    }

    #[test]
    fn test_jit_listener_batch_transition_explicit_mode() {
        let listener = JitShapeListener::with_explicit_invalidation();

        let transitions = vec![(ShapeId(1), ShapeId(2)), (ShapeId(2), ShapeId(3))];

        let count = listener.on_batch_transition(&transitions);

        assert_eq!(count, 2);
        // Should have done explicit invalidation
        assert_eq!(listener.stats().explicit_invalidation_count(), 1);
    }

    // -------------------------------------------------------------------------
    // Statistics Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_jit_listener_stats_new() {
        let stats = JitListenerStats::new();
        assert_eq!(stats.transition_count(), 0);
        assert_eq!(stats.deletion_count(), 0);
        assert_eq!(stats.prototype_change_count(), 0);
        assert_eq!(stats.accessor_install_count(), 0);
        assert_eq!(stats.batch_transition_count(), 0);
        assert_eq!(stats.version_bump_count(), 0);
        assert_eq!(stats.explicit_invalidation_count(), 0);
        assert_eq!(stats.ics_invalidated_count(), 0);
        assert_eq!(stats.total_events(), 0);
    }

    #[test]
    fn test_jit_listener_stats_snapshot() {
        let listener = JitShapeListener::new();

        listener.on_transition(ShapeId(1), ShapeId(2));
        listener.on_property_delete(ShapeId(2), "x");
        listener.on_prototype_change(ShapeId(3));
        listener.on_accessor_installed(ShapeId(4), "y");

        let snap = listener.stats().snapshot();
        assert_eq!(snap.transitions, 1);
        assert_eq!(snap.deletions, 1);
        assert_eq!(snap.prototype_changes, 1);
        assert_eq!(snap.accessor_installs, 1);
        assert!(snap.version_bumps >= 4);
        assert_eq!(snap.explicit_invalidations, 3); // delete + proto + accessor
    }

    #[test]
    fn test_jit_listener_stats_total_events() {
        let listener = JitShapeListener::new();

        listener.on_transition(ShapeId(1), ShapeId(2));
        listener.on_transition(ShapeId(2), ShapeId(3));
        listener.on_property_delete(ShapeId(3), "a");
        listener.on_prototype_change(ShapeId(4));
        listener.on_accessor_installed(ShapeId(5), "b");

        assert_eq!(listener.stats().total_events(), 5);
    }

    // -------------------------------------------------------------------------
    // Thread Safety Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_jit_listener_concurrent_transitions() {
        let listener = Arc::new(JitShapeListener::new());
        let mut handles = vec![];

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

        assert_eq!(listener.stats().transition_count(), 1000);
        assert_eq!(listener.stats().version_bump_count(), 1000);
    }

    #[test]
    fn test_jit_listener_concurrent_mixed_events() {
        let listener = Arc::new(JitShapeListener::new());
        let mut handles = vec![];

        // Transition threads
        for t in 0..5 {
            let l = Arc::clone(&listener);
            handles.push(thread::spawn(move || {
                for i in 0..50u32 {
                    l.on_transition(ShapeId(t * 100 + i), ShapeId(t * 100 + i + 1));
                }
            }));
        }

        // Deletion threads
        for t in 0..3 {
            let l = Arc::clone(&listener);
            handles.push(thread::spawn(move || {
                for i in 0..30u32 {
                    l.on_property_delete(ShapeId(t * 100 + i), "prop");
                }
            }));
        }

        // Prototype change threads
        for t in 0..2 {
            let l = Arc::clone(&listener);
            handles.push(thread::spawn(move || {
                for i in 0..20u32 {
                    l.on_prototype_change(ShapeId(t * 100 + i));
                }
            }));
        }

        for h in handles {
            h.join().unwrap();
        }

        // 5*50 = 250 transitions
        assert_eq!(listener.stats().transition_count(), 250);
        // 3*30 = 90 deletions
        assert_eq!(listener.stats().deletion_count(), 90);
        // 2*20 = 40 prototype changes
        assert_eq!(listener.stats().prototype_change_count(), 40);
        // Total: 380 events
        assert_eq!(listener.stats().total_events(), 380);
    }

    // -------------------------------------------------------------------------
    // Global Listener Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_global_jit_listener_accessible() {
        let listener = global_jit_listener();
        // Should be accessible
        let _ = listener.stats().snapshot();
    }

    #[test]
    fn test_global_jit_listener_singleton() {
        let l1 = global_jit_listener();
        let l2 = global_jit_listener();
        // Same instance
        assert!(std::ptr::eq(l1, l2));
    }

    // -------------------------------------------------------------------------
    // Snapshot Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_stats_snapshot_default() {
        let snap = JitListenerStatsSnapshot::default();
        assert_eq!(snap.transitions, 0);
        assert_eq!(snap.deletions, 0);
        assert_eq!(snap.prototype_changes, 0);
        assert_eq!(snap.accessor_installs, 0);
        assert_eq!(snap.batch_transitions, 0);
        assert_eq!(snap.version_bumps, 0);
        assert_eq!(snap.explicit_invalidations, 0);
        assert_eq!(snap.ics_invalidated, 0);
    }

    #[test]
    fn test_stats_snapshot_copy() {
        let listener = JitShapeListener::new();
        listener.on_transition(ShapeId(1), ShapeId(2));

        let snap1 = listener.stats().snapshot();
        let snap2 = snap1; // Copy

        assert_eq!(snap1.transitions, snap2.transitions);
        assert_eq!(snap1.version_bumps, snap2.version_bumps);
    }

    // -------------------------------------------------------------------------
    // Edge Case Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_jit_listener_same_shape_transition() {
        let listener = JitShapeListener::new();
        let before = ShapeVersion::current();

        // Transition to same shape (edge case)
        listener.on_transition(ShapeId(1), ShapeId(1));

        let after = ShapeVersion::current();
        // Should still bump version
        assert!(after > before);
    }

    #[test]
    fn test_jit_listener_max_shape_id() {
        let listener = JitShapeListener::new();

        listener.on_transition(ShapeId(u32::MAX), ShapeId(0));
        listener.on_property_delete(ShapeId(u32::MAX), "x");
        listener.on_prototype_change(ShapeId(u32::MAX));

        assert_eq!(listener.stats().total_events(), 3);
    }

    #[test]
    fn test_jit_listener_empty_property_name() {
        let listener = JitShapeListener::new();

        listener.on_property_delete(ShapeId(1), "");
        listener.on_accessor_installed(ShapeId(1), "");

        assert_eq!(listener.stats().deletion_count(), 1);
        assert_eq!(listener.stats().accessor_install_count(), 1);
    }

    #[test]
    fn test_jit_listener_unicode_property_name() {
        let listener = JitShapeListener::new();

        listener.on_property_delete(ShapeId(1), "日本語");
        listener.on_accessor_installed(ShapeId(1), "𝓤𝓷𝓲𝓬𝓸𝓭𝓮");

        assert_eq!(listener.stats().deletion_count(), 1);
        assert_eq!(listener.stats().accessor_install_count(), 1);
    }

    #[test]
    fn test_jit_listener_large_batch() {
        let listener = JitShapeListener::new();

        let transitions: Vec<_> = (0..1000u32).map(|i| (ShapeId(i), ShapeId(i + 1))).collect();

        let count = listener.on_batch_transition(&transitions);

        assert_eq!(count, 1000);
        assert_eq!(listener.stats().batch_transition_count(), 1000);
        // Only one version bump for entire batch
        assert_eq!(listener.stats().version_bump_count(), 1);
    }
}
