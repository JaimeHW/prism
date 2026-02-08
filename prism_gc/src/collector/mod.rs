//! Garbage collection algorithms.
//!
//! This module contains the core collection algorithms:
//! - Minor collection (scavenge): Copying collection of the nursery
//! - Major collection: Mark-sweep of the old generation
//!
//! # Collection Strategy
//!
//! The collector uses a generational hypothesis: most objects die young.
//! This allows efficient collection by focusing on the nursery (minor GC)
//! and only occasionally scanning the entire heap (major GC).
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────────────┐
//! │  GENERATIONAL GC                                                         │
//! │                                                                         │
//! │  ┌──────────────────────┐    ┌──────────────────────┐                  │
//! │  │     NURSERY          │    │    OLD GENERATION     │                  │
//! │  │  (Minor GC Target)   │───▶│   (Major GC Target)   │                  │
//! │  │                      │    │                       │                  │
//! │  │  • Fast allocation   │    │  • Slower allocation  │                  │
//! │  │  • Frequent GC       │    │  • Infrequent GC      │                  │
//! │  │  • Copying collector │    │  • Mark-sweep         │                  │
//! │  └──────────────────────┘    └──────────────────────┘                  │
//! │           │                            │                                │
//! │           ▼ promotion                  ▼ sweep                          │
//! │   Objects surviving               Unreachable objects                   │
//! │   N minor GCs                     freed                                 │
//! └─────────────────────────────────────────────────────────────────────────┘
//! ```

pub mod concurrent_major;
pub mod concurrent_marker;
mod major;
pub mod mark_bitmap;
mod minor;

#[cfg(test)]
mod concurrent_major_tests;
#[cfg(test)]
mod concurrent_marker_tests;
#[cfg(test)]
mod mark_bitmap_tests;

pub use concurrent_major::{
    ConcurrentGcPhase, ConcurrentMajorCollector, ConcurrentMajorConfig, ConcurrentMajorResult,
};
pub use concurrent_marker::{
    ConcurrentMarker, ConcurrentMarkerConfig, MarkPtr, MarkingStats, MarkingStatsSnapshot,
    MarkingWorklist,
};
pub use major::{MajorCollector, MajorResult};
pub use mark_bitmap::AtomicMarkBitmap;
pub use minor::{MinorCollector, MinorResult};

use crate::heap::GcHeap;
use crate::roots::RootSet;
use crate::trace::ObjectTracer;
use std::time::{Duration, Instant};

/// Result of a garbage collection.
#[derive(Debug, Clone)]
pub struct CollectionResult {
    /// Type of collection performed.
    pub collection_type: CollectionType,
    /// Time spent in collection.
    pub duration: Duration,
    /// Bytes freed.
    pub bytes_freed: usize,
    /// Objects freed.
    pub objects_freed: usize,
    /// Bytes promoted (for minor GC).
    pub bytes_promoted: usize,
    /// Objects promoted (for minor GC).
    pub objects_promoted: usize,
    /// Live bytes after collection.
    pub live_bytes: usize,
}

/// Type of collection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CollectionType {
    /// Minor (young generation) collection.
    Minor,
    /// Major (full) collection.
    Major,
}

/// Collector state machine.
///
/// Orchestrates minor and major collections, providing a unified interface
/// for the runtime to trigger garbage collection.
pub struct Collector {
    /// Minor collector for nursery.
    minor: MinorCollector,
    /// Major collector for full heap.
    major: MajorCollector,
}

impl Collector {
    /// Create a new collector.
    #[inline]
    pub fn new() -> Self {
        Self {
            minor: MinorCollector::new(),
            major: MajorCollector::new(),
        }
    }

    /// Create a collector with custom promotion age.
    ///
    /// Objects surviving `promotion_age` minor collections are promoted
    /// to the tenured generation.
    #[inline]
    pub fn with_promotion_age(promotion_age: u8) -> Self {
        Self {
            minor: MinorCollector::with_promotion_age(promotion_age),
            major: MajorCollector::new(),
        }
    }

    /// Perform a minor (nursery) collection.
    ///
    /// Copies live objects from nursery to survivor/tenured space.
    ///
    /// # Arguments
    /// - `heap`: The GC heap
    /// - `roots`: Root set containing stack roots, globals, etc.
    /// - `object_tracer`: Object tracer for type-aware tracing
    pub fn collect_minor<T: ObjectTracer>(
        &mut self,
        heap: &mut GcHeap,
        roots: &RootSet,
        object_tracer: &T,
    ) -> CollectionResult {
        let start = Instant::now();

        let result = self.minor.collect(heap, roots, object_tracer);

        let duration = start.elapsed();
        heap.stats().record_minor_gc(duration);
        heap.reset_gc_counter();

        CollectionResult {
            collection_type: CollectionType::Minor,
            duration,
            bytes_freed: result.bytes_freed,
            objects_freed: result.objects_freed,
            bytes_promoted: result.bytes_promoted,
            objects_promoted: result.objects_promoted,
            live_bytes: result.live_bytes,
        }
    }

    /// Perform a major (full) collection.
    ///
    /// Mark-sweep of the entire heap.
    ///
    /// # Arguments
    /// - `heap`: The GC heap
    /// - `roots`: Root set containing stack roots, globals, etc.
    /// - `object_tracer`: Object tracer for type-aware tracing
    pub fn collect_major<T: ObjectTracer>(
        &mut self,
        heap: &mut GcHeap,
        roots: &RootSet,
        object_tracer: &T,
    ) -> CollectionResult {
        let start = Instant::now();

        let result = self.major.collect(heap, roots, object_tracer);

        let duration = start.elapsed();
        heap.stats().record_major_gc(duration);
        heap.reset_gc_counter();

        CollectionResult {
            collection_type: CollectionType::Major,
            duration,
            bytes_freed: result.bytes_freed,
            objects_freed: result.objects_freed,
            bytes_promoted: 0,
            objects_promoted: 0,
            live_bytes: result.live_bytes,
        }
    }

    /// Perform collection based on heap state.
    ///
    /// Chooses minor or major based on current conditions.
    ///
    /// # Arguments
    /// - `heap`: The GC heap
    /// - `roots`: Root set containing stack roots, globals, etc.
    /// - `object_tracer`: Object tracer for type-aware tracing
    pub fn collect<T: ObjectTracer>(
        &mut self,
        heap: &mut GcHeap,
        roots: &RootSet,
        object_tracer: &T,
    ) -> CollectionResult {
        if heap.should_major_collect() {
            self.collect_major(heap, roots, object_tracer)
        } else {
            self.collect_minor(heap, roots, object_tracer)
        }
    }

    /// Perform minor collection without object tracing.
    ///
    /// Useful for tests or when runtime is unavailable.
    pub fn collect_minor_roots_only(
        &mut self,
        heap: &mut GcHeap,
        roots: &RootSet,
    ) -> CollectionResult {
        let start = Instant::now();

        let result = self.minor.collect_roots_only(heap, roots);

        let duration = start.elapsed();
        heap.stats().record_minor_gc(duration);
        heap.reset_gc_counter();

        CollectionResult {
            collection_type: CollectionType::Minor,
            duration,
            bytes_freed: result.bytes_freed,
            objects_freed: result.objects_freed,
            bytes_promoted: result.bytes_promoted,
            objects_promoted: result.objects_promoted,
            live_bytes: result.live_bytes,
        }
    }

    /// Perform major collection without object tracing.
    ///
    /// Useful for tests or when runtime is unavailable.
    pub fn collect_major_roots_only(
        &mut self,
        heap: &mut GcHeap,
        roots: &RootSet,
    ) -> CollectionResult {
        let start = Instant::now();

        let result = self.major.collect_roots_only(heap, roots);

        let duration = start.elapsed();
        heap.stats().record_major_gc(duration);
        heap.reset_gc_counter();

        CollectionResult {
            collection_type: CollectionType::Major,
            duration,
            bytes_freed: result.bytes_freed,
            objects_freed: result.objects_freed,
            bytes_promoted: 0,
            objects_promoted: 0,
            live_bytes: result.live_bytes,
        }
    }

    /// Get the minor collector's promotion age.
    #[inline]
    pub fn promotion_age(&self) -> u8 {
        self.minor.promotion_age()
    }
}

impl Default for Collector {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::GcConfig;
    use crate::trace::NoopObjectTracer;

    #[test]
    fn test_collector_creation() {
        let collector = Collector::new();
        assert_eq!(collector.promotion_age(), 2);
    }

    #[test]
    fn test_collector_custom_promotion_age() {
        let collector = Collector::with_promotion_age(5);
        assert_eq!(collector.promotion_age(), 5);
    }

    #[test]
    fn test_collect_minor_empty() {
        let mut collector = Collector::new();
        let mut heap = GcHeap::new(GcConfig::default());
        let roots = RootSet::new();

        let result = collector.collect_minor(&mut heap, &roots, &NoopObjectTracer);

        assert_eq!(result.collection_type, CollectionType::Minor);
        assert_eq!(result.bytes_freed, 0);
    }

    #[test]
    fn test_collect_major_empty() {
        let mut collector = Collector::new();
        let mut heap = GcHeap::new(GcConfig::default());
        let roots = RootSet::new();

        let result = collector.collect_major(&mut heap, &roots, &NoopObjectTracer);

        assert_eq!(result.collection_type, CollectionType::Major);
        assert_eq!(result.bytes_freed, 0);
    }

    #[test]
    fn test_collect_roots_only() {
        let mut collector = Collector::new();
        let mut heap = GcHeap::new(GcConfig::default());
        let roots = RootSet::new();

        let minor_result = collector.collect_minor_roots_only(&mut heap, &roots);
        let major_result = collector.collect_major_roots_only(&mut heap, &roots);

        assert_eq!(minor_result.collection_type, CollectionType::Minor);
        assert_eq!(major_result.collection_type, CollectionType::Major);
    }

    #[test]
    fn test_auto_select_collection() {
        let mut collector = Collector::new();
        let mut heap = GcHeap::new(GcConfig::default());
        let roots = RootSet::new();

        // With empty heap, should choose minor
        let result = collector.collect(&mut heap, &roots, &NoopObjectTracer);
        assert_eq!(result.collection_type, CollectionType::Minor);
    }
}
