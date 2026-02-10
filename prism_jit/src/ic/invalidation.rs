//! IC Invalidation System
//!
//! Manages shape-based invalidation of inline caches. When a shape transition
//! occurs that could affect cached slot offsets, all dependent ICs must be
//! invalidated.
//!
//! # Architecture
//!
//! - **ShapeVersion**: Monotonic epoch counter for tracking shape changes
//! - **IcDependency**: Links an IC site to the shapes it depends on
//! - **IcInvalidator**: Coordinates batch invalidation across IC managers
//!
//! # Invalidation Strategy
//!
//! Rather than eagerly invalidating ICs on every shape change, we use lazy
//! invalidation:
//!
//! 1. Shape changes bump the global shape version
//! 2. ICs record their creation version
//! 3. On IC access, if version mismatch → invalidate and re-populate
//!
//! This amortizes invalidation cost across accesses.

use prism_runtime::object::shape::ShapeId;
use std::collections::HashMap;
use std::sync::RwLock;
use std::sync::atomic::{AtomicU64, Ordering};

// =============================================================================
// Shape Version
// =============================================================================

/// Global shape version counter.
///
/// Incremented on any shape transition that could affect IC validity.
/// ICs record their creation version and check against global on access.
static GLOBAL_SHAPE_VERSION: AtomicU64 = AtomicU64::new(0);

/// A shape version stamp.
///
/// Used to detect stale ICs that need invalidation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(transparent)]
pub struct ShapeVersion(u64);

impl ShapeVersion {
    /// Create a new version from a raw value.
    #[inline]
    pub const fn new(value: u64) -> Self {
        Self(value)
    }

    /// Get the current global shape version.
    #[inline]
    pub fn current() -> Self {
        Self(GLOBAL_SHAPE_VERSION.load(Ordering::Acquire))
    }

    /// Increment the global shape version and return the new value.
    ///
    /// Call this when a shape transition occurs.
    #[inline]
    pub fn bump() -> Self {
        let new = GLOBAL_SHAPE_VERSION.fetch_add(1, Ordering::AcqRel) + 1;
        Self(new)
    }

    /// Get the raw version value.
    #[inline]
    pub const fn value(self) -> u64 {
        self.0
    }

    /// Check if this version is stale compared to the current global version.
    #[inline]
    pub fn is_stale(self) -> bool {
        self.0 < GLOBAL_SHAPE_VERSION.load(Ordering::Acquire)
    }

    /// Check if this version matches the current global version.
    #[inline]
    pub fn is_current(self) -> bool {
        self.0 == GLOBAL_SHAPE_VERSION.load(Ordering::Acquire)
    }
}

impl Default for ShapeVersion {
    #[inline]
    fn default() -> Self {
        Self::current()
    }
}

// =============================================================================
// IC Dependency
// =============================================================================

/// Describes a dependency between an IC site and shapes.
///
/// When any of the watched shapes transitions, the IC must be invalidated.
#[derive(Debug, Clone)]
pub struct IcDependency {
    /// The shape this IC depends on.
    pub shape_id: ShapeId,

    /// The IC manager owning this dependency.
    /// Weak reference to avoid cycles (manager owns dependencies).
    pub manager_id: u64,

    /// Index of the IC site within the manager.
    pub site_index: u32,

    /// Version when this dependency was created.
    pub version: ShapeVersion,
}

impl IcDependency {
    /// Create a new dependency.
    #[inline]
    pub fn new(shape_id: ShapeId, manager_id: u64, site_index: u32) -> Self {
        Self {
            shape_id,
            manager_id,
            site_index,
            version: ShapeVersion::current(),
        }
    }

    /// Check if this dependency is stale.
    #[inline]
    pub fn is_stale(&self) -> bool {
        self.version.is_stale()
    }
}

// =============================================================================
// Invalidation Event
// =============================================================================

/// An event that triggers IC invalidation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InvalidationReason {
    /// A new shape was created (transition from existing shape).
    ShapeTransition,
    /// A property was deleted.
    PropertyDeletion,
    /// An accessor (getter/setter) was installed.
    AccessorInstalled,
    /// The prototype chain was modified.
    PrototypeChange,
    /// Manual invalidation requested.
    Manual,
}

/// An invalidation event.
#[derive(Debug, Clone)]
pub struct InvalidationEvent {
    /// The shape that changed.
    pub shape_id: ShapeId,
    /// Reason for invalidation.
    pub reason: InvalidationReason,
    /// New version after this event.
    pub new_version: ShapeVersion,
}

impl InvalidationEvent {
    /// Create a new event and bump the global version.
    #[inline]
    pub fn new(shape_id: ShapeId, reason: InvalidationReason) -> Self {
        Self {
            shape_id,
            reason,
            new_version: ShapeVersion::bump(),
        }
    }
}

// =============================================================================
// IC Invalidator
// =============================================================================

/// Callback type for IC invalidation.
pub type InvalidationCallback = Box<dyn Fn(&InvalidationEvent) + Send + Sync>;

/// Coordinates IC invalidation across the runtime.
///
/// Maintains a registry of dependencies and handles batch invalidation.
/// Thread-safe via internal locking.
#[derive(Debug)]
pub struct IcInvalidator {
    /// Dependencies indexed by shape ID.
    /// Multiple ICs can depend on the same shape.
    dependencies: RwLock<HashMap<ShapeId, Vec<IcDependency>>>,

    /// Total number of dependencies tracked.
    dependency_count: AtomicU64,

    /// Total invalidation events processed.
    invalidation_count: AtomicU64,
}

impl IcInvalidator {
    /// Create a new invalidator.
    #[inline]
    pub fn new() -> Self {
        Self {
            dependencies: RwLock::new(HashMap::new()),
            dependency_count: AtomicU64::new(0),
            invalidation_count: AtomicU64::new(0),
        }
    }

    /// Register a dependency between an IC site and a shape.
    pub fn register_dependency(&self, dep: IcDependency) {
        let mut deps = self.dependencies.write().expect("Lock poisoned");
        deps.entry(dep.shape_id).or_insert_with(Vec::new).push(dep);
        self.dependency_count.fetch_add(1, Ordering::Relaxed);
    }

    /// Register multiple dependencies atomically.
    pub fn register_dependencies(&self, deps: impl IntoIterator<Item = IcDependency>) {
        let mut guard = self.dependencies.write().expect("Lock poisoned");
        let mut count = 0u64;
        for dep in deps {
            guard.entry(dep.shape_id).or_insert_with(Vec::new).push(dep);
            count += 1;
        }
        self.dependency_count.fetch_add(count, Ordering::Relaxed);
    }

    /// Get all dependencies for a shape.
    pub fn get_dependencies(&self, shape_id: ShapeId) -> Vec<IcDependency> {
        let deps = self.dependencies.read().expect("Lock poisoned");
        deps.get(&shape_id).cloned().unwrap_or_default()
    }

    /// Remove dependencies for a specific manager.
    ///
    /// Call this when an IcManager is destroyed.
    pub fn remove_manager_dependencies(&self, manager_id: u64) {
        let mut deps = self.dependencies.write().expect("Lock poisoned");
        let mut removed = 0u64;

        deps.retain(|_, v| {
            let before = v.len();
            v.retain(|d| d.manager_id != manager_id);
            removed += (before - v.len()) as u64;
            !v.is_empty()
        });

        if removed > 0 {
            // Saturating sub to handle race conditions
            let current = self.dependency_count.load(Ordering::Relaxed);
            self.dependency_count
                .store(current.saturating_sub(removed), Ordering::Relaxed);
        }
    }

    /// Invalidate all ICs depending on a shape.
    ///
    /// Returns the number of ICs invalidated.
    pub fn invalidate_shape(&self, event: InvalidationEvent) -> usize {
        self.invalidation_count.fetch_add(1, Ordering::Relaxed);

        let deps = self.dependencies.read().expect("Lock poisoned");
        let dependents = deps.get(&event.shape_id);

        if let Some(deps) = dependents {
            deps.len()
        } else {
            0
        }
    }

    /// Invalidate all ICs depending on any of the given shapes.
    ///
    /// Returns total number of ICs affected.
    pub fn invalidate_shapes(&self, shape_ids: &[ShapeId], reason: InvalidationReason) -> usize {
        if shape_ids.is_empty() {
            return 0;
        }

        let new_version = ShapeVersion::bump();
        let deps = self.dependencies.read().expect("Lock poisoned");
        let mut total = 0;

        for &shape_id in shape_ids {
            if let Some(shape_deps) = deps.get(&shape_id) {
                total += shape_deps.len();
            }
        }

        if total > 0 {
            self.invalidation_count.fetch_add(1, Ordering::Relaxed);
        }

        total
    }

    /// Prune stale dependencies.
    ///
    /// Removes dependencies whose version is older than `min_version`.
    /// Call periodically to prevent unbounded growth.
    pub fn prune_stale(&self, min_version: ShapeVersion) {
        let mut deps = self.dependencies.write().expect("Lock poisoned");
        let mut removed = 0u64;

        deps.retain(|_, v| {
            let before = v.len();
            v.retain(|d| d.version >= min_version);
            removed += (before - v.len()) as u64;
            !v.is_empty()
        });

        if removed > 0 {
            let current = self.dependency_count.load(Ordering::Relaxed);
            self.dependency_count
                .store(current.saturating_sub(removed), Ordering::Relaxed);
        }
    }

    /// Clear all dependencies.
    pub fn clear(&self) {
        let mut deps = self.dependencies.write().expect("Lock poisoned");
        deps.clear();
        self.dependency_count.store(0, Ordering::Relaxed);
    }

    /// Get total number of tracked dependencies.
    #[inline]
    pub fn dependency_count(&self) -> u64 {
        self.dependency_count.load(Ordering::Relaxed)
    }

    /// Get total number of invalidation events.
    #[inline]
    pub fn invalidation_count(&self) -> u64 {
        self.invalidation_count.load(Ordering::Relaxed)
    }

    /// Get statistics snapshot.
    pub fn stats(&self) -> InvalidatorStats {
        let deps = self.dependencies.read().expect("Lock poisoned");
        InvalidatorStats {
            dependency_count: self.dependency_count.load(Ordering::Relaxed),
            unique_shapes: deps.len() as u64,
            invalidation_count: self.invalidation_count.load(Ordering::Relaxed),
        }
    }
}

impl Default for IcInvalidator {
    fn default() -> Self {
        Self::new()
    }
}

/// Statistics about the invalidator.
#[derive(Debug, Clone, Copy, Default)]
pub struct InvalidatorStats {
    /// Total dependencies tracked.
    pub dependency_count: u64,
    /// Number of unique shapes with dependencies.
    pub unique_shapes: u64,
    /// Total invalidation events processed.
    pub invalidation_count: u64,
}

// =============================================================================
// Global Invalidator
// =============================================================================

use std::sync::OnceLock;

/// Global invalidator instance.
static GLOBAL_INVALIDATOR: OnceLock<IcInvalidator> = OnceLock::new();

/// Get the global IC invalidator.
#[inline]
pub fn global_invalidator() -> &'static IcInvalidator {
    GLOBAL_INVALIDATOR.get_or_init(IcInvalidator::new)
}

/// Initialize the global invalidator (call at startup).
pub fn init_invalidator() {
    let _ = global_invalidator();
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -------------------------------------------------------------------------
    // ShapeVersion Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_shape_version_new() {
        let v = ShapeVersion::new(42);
        assert_eq!(v.value(), 42);
    }

    #[test]
    fn test_shape_version_ordering() {
        let v1 = ShapeVersion::new(1);
        let v2 = ShapeVersion::new(2);

        assert!(v1 < v2);
        assert!(v2 > v1);
        assert_eq!(v1, ShapeVersion::new(1));
    }

    #[test]
    fn test_shape_version_bump() {
        let before = ShapeVersion::current();
        let after = ShapeVersion::bump();
        assert!(after > before);
    }

    // -------------------------------------------------------------------------
    // IcDependency Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_ic_dependency_new() {
        let dep = IcDependency::new(ShapeId(42), 1, 0);
        assert_eq!(dep.shape_id, ShapeId(42));
        assert_eq!(dep.manager_id, 1);
        assert_eq!(dep.site_index, 0);
    }

    #[test]
    fn test_ic_dependency_stale() {
        let dep = IcDependency::new(ShapeId(1), 1, 0);
        assert!(!dep.is_stale()); // Just created

        ShapeVersion::bump();
        assert!(dep.is_stale()); // Now stale
    }

    // -------------------------------------------------------------------------
    // InvalidationEvent Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_invalidation_event_new() {
        let before = ShapeVersion::current();
        let event = InvalidationEvent::new(ShapeId(1), InvalidationReason::ShapeTransition);

        assert_eq!(event.shape_id, ShapeId(1));
        assert_eq!(event.reason, InvalidationReason::ShapeTransition);
        assert!(event.new_version > before);
    }

    // -------------------------------------------------------------------------
    // IcInvalidator Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_invalidator_new() {
        let inv = IcInvalidator::new();
        assert_eq!(inv.dependency_count(), 0);
        assert_eq!(inv.invalidation_count(), 0);
    }

    #[test]
    fn test_invalidator_register_dependency() {
        let inv = IcInvalidator::new();
        let dep = IcDependency::new(ShapeId(1), 100, 0);

        inv.register_dependency(dep);
        assert_eq!(inv.dependency_count(), 1);
    }

    #[test]
    fn test_invalidator_register_multiple() {
        let inv = IcInvalidator::new();
        let deps = vec![
            IcDependency::new(ShapeId(1), 100, 0),
            IcDependency::new(ShapeId(1), 100, 1),
            IcDependency::new(ShapeId(2), 100, 2),
        ];

        inv.register_dependencies(deps);
        assert_eq!(inv.dependency_count(), 3);

        // Check grouped by shape
        assert_eq!(inv.get_dependencies(ShapeId(1)).len(), 2);
        assert_eq!(inv.get_dependencies(ShapeId(2)).len(), 1);
        assert!(inv.get_dependencies(ShapeId(99)).is_empty());
    }

    #[test]
    fn test_invalidator_remove_manager() {
        let inv = IcInvalidator::new();

        // Register deps for two managers
        inv.register_dependency(IcDependency::new(ShapeId(1), 100, 0));
        inv.register_dependency(IcDependency::new(ShapeId(1), 100, 1));
        inv.register_dependency(IcDependency::new(ShapeId(1), 200, 0));

        // Remove manager 100
        inv.remove_manager_dependencies(100);

        assert_eq!(inv.dependency_count(), 1);
        assert_eq!(inv.get_dependencies(ShapeId(1)).len(), 1);
    }

    #[test]
    fn test_invalidator_invalidate_shape() {
        let inv = IcInvalidator::new();

        inv.register_dependency(IcDependency::new(ShapeId(1), 100, 0));
        inv.register_dependency(IcDependency::new(ShapeId(1), 100, 1));
        inv.register_dependency(IcDependency::new(ShapeId(2), 100, 2));

        let event = InvalidationEvent::new(ShapeId(1), InvalidationReason::PropertyDeletion);
        let count = inv.invalidate_shape(event);

        assert_eq!(count, 2);
        assert_eq!(inv.invalidation_count(), 1);
    }

    #[test]
    fn test_invalidator_invalidate_shapes_batch() {
        let inv = IcInvalidator::new();

        inv.register_dependency(IcDependency::new(ShapeId(1), 100, 0));
        inv.register_dependency(IcDependency::new(ShapeId(2), 100, 1));
        inv.register_dependency(IcDependency::new(ShapeId(3), 100, 2));

        let count = inv.invalidate_shapes(
            &[ShapeId(1), ShapeId(3)],
            InvalidationReason::PrototypeChange,
        );

        assert_eq!(count, 2);
    }

    #[test]
    fn test_invalidator_prune_stale() {
        let inv = IcInvalidator::new();

        // Create deps at current version
        let v1 = ShapeVersion::current();
        inv.register_dependency(IcDependency::new(ShapeId(1), 100, 0));

        // Bump version and create more deps
        ShapeVersion::bump();
        let v2 = ShapeVersion::current();
        inv.register_dependency(IcDependency::new(ShapeId(2), 100, 1));

        // Prune older than v2
        inv.prune_stale(v2);

        // Should only have the newer dep
        assert_eq!(inv.dependency_count(), 1);
        assert!(inv.get_dependencies(ShapeId(1)).is_empty());
        assert_eq!(inv.get_dependencies(ShapeId(2)).len(), 1);
    }

    #[test]
    fn test_invalidator_clear() {
        let inv = IcInvalidator::new();

        inv.register_dependency(IcDependency::new(ShapeId(1), 100, 0));
        inv.register_dependency(IcDependency::new(ShapeId(2), 100, 1));

        inv.clear();

        assert_eq!(inv.dependency_count(), 0);
        assert!(inv.get_dependencies(ShapeId(1)).is_empty());
    }

    #[test]
    fn test_invalidator_stats() {
        let inv = IcInvalidator::new();

        inv.register_dependency(IcDependency::new(ShapeId(1), 100, 0));
        inv.register_dependency(IcDependency::new(ShapeId(1), 100, 1));
        inv.register_dependency(IcDependency::new(ShapeId(2), 100, 2));

        let event = InvalidationEvent::new(ShapeId(1), InvalidationReason::Manual);
        inv.invalidate_shape(event);

        let stats = inv.stats();
        assert_eq!(stats.dependency_count, 3);
        assert_eq!(stats.unique_shapes, 2);
        assert_eq!(stats.invalidation_count, 1);
    }

    // -------------------------------------------------------------------------
    // Concurrent Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_invalidator_concurrent_register() {
        use std::sync::Arc;
        use std::thread;

        let inv = Arc::new(IcInvalidator::new());
        let mut handles = vec![];

        for manager_id in 0..10u64 {
            let i = Arc::clone(&inv);
            handles.push(thread::spawn(move || {
                for site_idx in 0..100u32 {
                    let shape_id = ShapeId((manager_id * 100 + site_idx as u64) as u32);
                    i.register_dependency(IcDependency::new(shape_id, manager_id, site_idx));
                }
            }));
        }

        for handle in handles {
            handle.join().unwrap();
        }

        assert_eq!(inv.dependency_count(), 1000);
    }

    #[test]
    fn test_shape_version_concurrent_bump() {
        use std::sync::Arc;
        use std::thread;

        let before = ShapeVersion::current();
        let mut handles = vec![];

        for _ in 0..100 {
            handles.push(thread::spawn(|| {
                for _ in 0..100 {
                    ShapeVersion::bump();
                }
            }));
        }

        for handle in handles {
            handle.join().unwrap();
        }

        let after = ShapeVersion::current();
        // Each bump is atomic, so we should have at least 10000 increments
        // (may be more if other concurrent tests also bumped the version)
        assert!(
            after.value() - before.value() >= 10000,
            "Expected at least 10000 bumps, got {}",
            after.value() - before.value()
        );
    }

    // -------------------------------------------------------------------------
    // Global Invalidator Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_global_invalidator() {
        init_invalidator();
        let inv = global_invalidator();

        // Should be the same instance
        assert!(std::ptr::eq(inv, global_invalidator()));
    }
}
