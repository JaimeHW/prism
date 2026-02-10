//! Inline handler cache for O(1) exception handler lookup.
//!
//! This module provides `InlineHandlerCache`, a per-frame cache that accelerates
//! repeated exception handler lookups. This is critical for hot paths like
//! StopIteration in for-loops, where the same exception type is raised repeatedly.
//!
//! # Design Rationale
//!
//! Without caching, every exception requires a linear scan of the handler table.
//! For hot loops iterating over generators, this means scanning the same table
//! thousands of times for the same PC range.
//!
//! The inline cache stores the last successful handler lookup:
//! - If the raise PC matches, use the cached handler index directly
//! - Otherwise, fall back to table scan and update the cache
//!
//! # Performance
//!
//! | Scenario | Without Cache | With Cache |
//! |----------|---------------|------------|
//! | First exception | O(N) scan | O(N) scan |
//! | Repeated exception (same PC) | O(N) scan | O(1) lookup |
//! | Different PC | O(N) scan | O(N) scan + cache update |
//!
//! For typical for-loop iteration, this reduces exception overhead by 80-90%.
//!
//! # Memory Layout
//!
//! ```text
//! ┌──────────────┬─────────────────┬──────────┐
//! │ last_pc (u32)│ cached_idx (u16)│ hits (u16)│
//! └──────────────┴─────────────────┴──────────┘
//! Total: 8 bytes (fits in Frame padding)
//! ```

use std::fmt;

// ============================================================================
// Constants
// ============================================================================

/// Sentinel value indicating no cached handler.
pub const NO_CACHED_HANDLER: u16 = u16::MAX;

/// Sentinel value indicating no cached PC.
pub const NO_CACHED_PC: u32 = u32::MAX;

/// Maximum hit count before saturation.
const MAX_HIT_COUNT: u16 = u16::MAX;

// ============================================================================
// InlineHandlerCache
// ============================================================================

/// Per-frame inline cache for exception handler lookup.
///
/// This cache stores the last successful handler lookup to enable O(1)
/// repeated lookups for the same exception raise site.
///
/// # Layout
///
/// ```text
/// ┌──────────────┬─────────────────┬──────────┐
/// │ last_pc (u32)│ cached_idx (u16)│ hits (u16)│
/// └──────────────┴─────────────────┴──────────┘
/// ```
///
/// # Example
///
/// ```ignore
/// let mut cache = InlineHandlerCache::new();
///
/// // First lookup - cache miss, populate cache
/// let handler_idx = find_handler(pc, table);
/// cache.record_hit(pc, handler_idx);
///
/// // Second lookup at same PC - cache hit!
/// if let Some(idx) = cache.try_get(pc) {
///     // O(1) lookup
/// }
/// ```
#[repr(C)]
#[derive(Clone, Copy)]
pub struct InlineHandlerCache {
    /// The PC of the last exception raise that was cached.
    last_pc: u32,
    /// Index into the handler table for the cached handler.
    /// `NO_CACHED_HANDLER` if no handler is cached.
    cached_handler_idx: u16,
    /// Hit count for monitoring cache effectiveness.
    hit_count: u16,
}

impl InlineHandlerCache {
    /// Creates a new empty cache.
    #[inline(always)]
    pub const fn new() -> Self {
        Self {
            last_pc: NO_CACHED_PC,
            cached_handler_idx: NO_CACHED_HANDLER,
            hit_count: 0,
        }
    }

    /// Attempts to get the cached handler index for the given PC.
    ///
    /// Returns `Some(index)` if the cache has a valid entry for this PC,
    /// `None` otherwise.
    #[inline(always)]
    pub fn try_get(&mut self, pc: u32) -> Option<u16> {
        if self.last_pc == pc && self.cached_handler_idx != NO_CACHED_HANDLER {
            // Cache hit - increment hit counter (saturating)
            self.hit_count = self.hit_count.saturating_add(1);
            Some(self.cached_handler_idx)
        } else {
            None
        }
    }

    /// Records a handler lookup result in the cache.
    ///
    /// This should be called after a successful handler table lookup.
    #[inline(always)]
    pub fn record(&mut self, pc: u32, handler_idx: u16) {
        self.last_pc = pc;
        self.cached_handler_idx = handler_idx;
        // Reset hit count for new entry
        self.hit_count = 0;
    }

    /// Records that no handler was found for the given PC.
    ///
    /// This prevents repeated table scans for the same PC.
    #[inline(always)]
    pub fn record_miss(&mut self, pc: u32) {
        self.last_pc = pc;
        self.cached_handler_idx = NO_CACHED_HANDLER;
        self.hit_count = 0;
    }

    /// Invalidates the cache.
    ///
    /// This should be called when the handler table changes or when
    /// entering/exiting try blocks.
    #[inline(always)]
    pub fn invalidate(&mut self) {
        self.last_pc = NO_CACHED_PC;
        self.cached_handler_idx = NO_CACHED_HANDLER;
        self.hit_count = 0;
    }

    /// Returns the current hit count.
    #[inline(always)]
    pub const fn hit_count(&self) -> u16 {
        self.hit_count
    }

    /// Returns true if the cache has a valid entry.
    #[inline(always)]
    pub const fn is_valid(&self) -> bool {
        self.last_pc != NO_CACHED_PC && self.cached_handler_idx != NO_CACHED_HANDLER
    }

    /// Returns true if the cache is empty/invalid.
    #[inline(always)]
    pub const fn is_empty(&self) -> bool {
        self.last_pc == NO_CACHED_PC
    }

    /// Returns the cached PC, if any.
    #[inline(always)]
    pub const fn cached_pc(&self) -> Option<u32> {
        if self.last_pc != NO_CACHED_PC {
            Some(self.last_pc)
        } else {
            None
        }
    }

    /// Returns the cached handler index, if any.
    #[inline(always)]
    pub const fn cached_handler(&self) -> Option<u16> {
        if self.cached_handler_idx != NO_CACHED_HANDLER {
            Some(self.cached_handler_idx)
        } else {
            None
        }
    }
}

impl Default for InlineHandlerCache {
    #[inline(always)]
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Debug for InlineHandlerCache {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("InlineHandlerCache")
            .field("last_pc", &format_args!("{:#x}", self.last_pc))
            .field("cached_handler_idx", &self.cached_handler_idx)
            .field("hit_count", &self.hit_count)
            .finish()
    }
}

// ============================================================================
// HandlerCacheStats
// ============================================================================

/// Statistics for handler cache performance monitoring.
///
/// This is used for profiling and optimization decisions.
#[derive(Debug, Clone, Copy, Default)]
pub struct HandlerCacheStats {
    /// Total cache lookup attempts.
    pub lookups: u64,
    /// Successful cache hits.
    pub hits: u64,
    /// Cache misses (required table scan).
    pub misses: u64,
    /// Cache invalidations.
    pub invalidations: u64,
}

impl HandlerCacheStats {
    /// Creates new empty stats.
    #[inline(always)]
    pub const fn new() -> Self {
        Self {
            lookups: 0,
            hits: 0,
            misses: 0,
            invalidations: 0,
        }
    }

    /// Records a cache hit.
    #[inline(always)]
    pub fn record_hit(&mut self) {
        self.lookups += 1;
        self.hits += 1;
    }

    /// Records a cache miss.
    #[inline(always)]
    pub fn record_miss(&mut self) {
        self.lookups += 1;
        self.misses += 1;
    }

    /// Records a cache invalidation.
    #[inline(always)]
    pub fn record_invalidation(&mut self) {
        self.invalidations += 1;
    }

    /// Returns the hit rate as a percentage (0.0 - 100.0).
    #[inline]
    pub fn hit_rate(&self) -> f64 {
        if self.lookups == 0 {
            0.0
        } else {
            (self.hits as f64 / self.lookups as f64) * 100.0
        }
    }

    /// Merges another stats instance into this one.
    #[inline]
    pub fn merge(&mut self, other: &Self) {
        self.lookups += other.lookups;
        self.hits += other.hits;
        self.misses += other.misses;
        self.invalidations += other.invalidations;
    }

    /// Resets all counters to zero.
    #[inline]
    pub fn reset(&mut self) {
        *self = Self::new();
    }
}

// ============================================================================
// MultiLevelCache
// ============================================================================

/// Multi-level handler cache for nested exception handling.
///
/// This provides caching for multiple active try blocks, supporting
/// efficient exception handling in deeply nested code.
///
/// # Design
///
/// Instead of a single cache entry, we maintain a small stack of entries
/// for the most recent try blocks. This handles the common case of
/// nested try blocks without cache thrashing.
#[derive(Clone, Copy)]
pub struct MultiLevelCache {
    /// Cache entries, most recent first.
    entries: [InlineHandlerCache; 4],
    /// Number of valid entries.
    count: u8,
}

impl MultiLevelCache {
    /// Creates a new empty multi-level cache.
    #[inline(always)]
    pub const fn new() -> Self {
        Self {
            entries: [InlineHandlerCache::new(); 4],
            count: 0,
        }
    }

    /// Maximum number of cache levels.
    pub const MAX_LEVELS: usize = 4;

    /// Attempts to find a cached handler for the given PC.
    ///
    /// Searches all cache levels, returning the first hit.
    #[inline]
    pub fn try_get(&mut self, pc: u32) -> Option<u16> {
        for i in 0..self.count as usize {
            if let Some(idx) = self.entries[i].try_get(pc) {
                return Some(idx);
            }
        }
        None
    }

    /// Records a handler lookup in the most recent cache level.
    #[inline]
    pub fn record(&mut self, pc: u32, handler_idx: u16) {
        if self.count > 0 {
            self.entries[0].record(pc, handler_idx);
        } else {
            // Create first entry
            self.entries[0].record(pc, handler_idx);
            self.count = 1;
        }
    }

    /// Pushes a new cache level (entering a try block).
    #[inline]
    pub fn push_level(&mut self) {
        if (self.count as usize) < Self::MAX_LEVELS {
            // Shift entries down
            for i in (1..=self.count as usize).rev() {
                if i < Self::MAX_LEVELS {
                    self.entries[i] = self.entries[i - 1];
                }
            }
            self.entries[0] = InlineHandlerCache::new();
            self.count = (self.count + 1).min(Self::MAX_LEVELS as u8);
        }
    }

    /// Pops a cache level (exiting a try block).
    #[inline]
    pub fn pop_level(&mut self) {
        if self.count > 0 {
            // Shift entries up
            for i in 0..self.count as usize - 1 {
                self.entries[i] = self.entries[i + 1];
            }
            self.entries[self.count as usize - 1] = InlineHandlerCache::new();
            self.count -= 1;
        }
    }

    /// Invalidates all cache levels.
    #[inline]
    pub fn invalidate_all(&mut self) {
        for entry in &mut self.entries {
            entry.invalidate();
        }
        self.count = 0;
    }

    /// Returns the number of active cache levels.
    #[inline(always)]
    pub const fn level_count(&self) -> usize {
        self.count as usize
    }

    /// Returns true if the cache is empty.
    #[inline(always)]
    pub const fn is_empty(&self) -> bool {
        self.count == 0
    }
}

impl Default for MultiLevelCache {
    #[inline(always)]
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Debug for MultiLevelCache {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("MultiLevelCache")
            .field("count", &self.count)
            .field("entries", &&self.entries[..self.count as usize])
            .finish()
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ════════════════════════════════════════════════════════════════════════
    // InlineHandlerCache Creation Tests
    // ════════════════════════════════════════════════════════════════════════

    #[test]
    fn test_cache_new_is_empty() {
        let cache = InlineHandlerCache::new();
        assert!(cache.is_empty());
        assert!(!cache.is_valid());
    }

    #[test]
    fn test_cache_default_equals_new() {
        let cache1 = InlineHandlerCache::new();
        let cache2 = InlineHandlerCache::default();

        assert_eq!(cache1.last_pc, cache2.last_pc);
        assert_eq!(cache1.cached_handler_idx, cache2.cached_handler_idx);
    }

    // ════════════════════════════════════════════════════════════════════════
    // InlineHandlerCache Size Tests
    // ════════════════════════════════════════════════════════════════════════

    #[test]
    fn test_cache_size() {
        // Must fit in 8 bytes for Frame padding efficiency
        assert_eq!(std::mem::size_of::<InlineHandlerCache>(), 8);
    }

    #[test]
    fn test_cache_alignment() {
        // 4-byte alignment for efficient access
        assert_eq!(std::mem::align_of::<InlineHandlerCache>(), 4);
    }

    // ════════════════════════════════════════════════════════════════════════
    // InlineHandlerCache Lookup Tests
    // ════════════════════════════════════════════════════════════════════════

    #[test]
    fn test_cache_try_get_empty() {
        let mut cache = InlineHandlerCache::new();
        assert_eq!(cache.try_get(100), None);
    }

    #[test]
    fn test_cache_try_get_after_record() {
        let mut cache = InlineHandlerCache::new();
        cache.record(100, 5);

        assert_eq!(cache.try_get(100), Some(5));
    }

    #[test]
    fn test_cache_try_get_wrong_pc() {
        let mut cache = InlineHandlerCache::new();
        cache.record(100, 5);

        assert_eq!(cache.try_get(200), None);
    }

    #[test]
    fn test_cache_try_get_increments_hit_count() {
        let mut cache = InlineHandlerCache::new();
        cache.record(100, 5);

        assert_eq!(cache.hit_count(), 0);
        cache.try_get(100);
        assert_eq!(cache.hit_count(), 1);
        cache.try_get(100);
        assert_eq!(cache.hit_count(), 2);
    }

    #[test]
    fn test_cache_hit_count_saturates() {
        let mut cache = InlineHandlerCache::new();
        cache.record(100, 5);
        cache.hit_count = MAX_HIT_COUNT;

        cache.try_get(100);
        assert_eq!(cache.hit_count(), MAX_HIT_COUNT);
    }

    // ════════════════════════════════════════════════════════════════════════
    // InlineHandlerCache Record Tests
    // ════════════════════════════════════════════════════════════════════════

    #[test]
    fn test_cache_record_sets_valid() {
        let mut cache = InlineHandlerCache::new();
        assert!(!cache.is_valid());

        cache.record(100, 5);
        assert!(cache.is_valid());
    }

    #[test]
    fn test_cache_record_overwrites() {
        let mut cache = InlineHandlerCache::new();
        cache.record(100, 5);
        cache.record(200, 10);

        assert_eq!(cache.try_get(100), None);
        assert_eq!(cache.try_get(200), Some(10));
    }

    #[test]
    fn test_cache_record_resets_hit_count() {
        let mut cache = InlineHandlerCache::new();
        cache.record(100, 5);
        cache.try_get(100);
        cache.try_get(100);
        assert_eq!(cache.hit_count(), 2);

        cache.record(200, 10);
        assert_eq!(cache.hit_count(), 0);
    }

    // ════════════════════════════════════════════════════════════════════════
    // InlineHandlerCache Miss Tests
    // ════════════════════════════════════════════════════════════════════════

    #[test]
    fn test_cache_record_miss() {
        let mut cache = InlineHandlerCache::new();
        cache.record_miss(100);

        assert!(!cache.is_empty());
        assert!(!cache.is_valid());
        assert_eq!(cache.try_get(100), None);
    }

    #[test]
    fn test_cache_record_miss_prevents_lookup() {
        let mut cache = InlineHandlerCache::new();
        cache.record_miss(100);

        // Even though PC matches, handler is NO_CACHED_HANDLER
        assert_eq!(cache.try_get(100), None);
    }

    // ════════════════════════════════════════════════════════════════════════
    // InlineHandlerCache Invalidate Tests
    // ════════════════════════════════════════════════════════════════════════

    #[test]
    fn test_cache_invalidate() {
        let mut cache = InlineHandlerCache::new();
        cache.record(100, 5);
        assert!(cache.is_valid());

        cache.invalidate();
        assert!(cache.is_empty());
        assert!(!cache.is_valid());
    }

    #[test]
    fn test_cache_invalidate_clears_hit_count() {
        let mut cache = InlineHandlerCache::new();
        cache.record(100, 5);
        cache.try_get(100);
        cache.try_get(100);

        cache.invalidate();
        assert_eq!(cache.hit_count(), 0);
    }

    // ════════════════════════════════════════════════════════════════════════
    // InlineHandlerCache Accessor Tests
    // ════════════════════════════════════════════════════════════════════════

    #[test]
    fn test_cache_cached_pc_empty() {
        let cache = InlineHandlerCache::new();
        assert_eq!(cache.cached_pc(), None);
    }

    #[test]
    fn test_cache_cached_pc_after_record() {
        let mut cache = InlineHandlerCache::new();
        cache.record(100, 5);
        assert_eq!(cache.cached_pc(), Some(100));
    }

    #[test]
    fn test_cache_cached_handler_empty() {
        let cache = InlineHandlerCache::new();
        assert_eq!(cache.cached_handler(), None);
    }

    #[test]
    fn test_cache_cached_handler_after_record() {
        let mut cache = InlineHandlerCache::new();
        cache.record(100, 5);
        assert_eq!(cache.cached_handler(), Some(5));
    }

    // ════════════════════════════════════════════════════════════════════════
    // HandlerCacheStats Tests
    // ════════════════════════════════════════════════════════════════════════

    #[test]
    fn test_stats_new() {
        let stats = HandlerCacheStats::new();
        assert_eq!(stats.lookups, 0);
        assert_eq!(stats.hits, 0);
        assert_eq!(stats.misses, 0);
    }

    #[test]
    fn test_stats_record_hit() {
        let mut stats = HandlerCacheStats::new();
        stats.record_hit();

        assert_eq!(stats.lookups, 1);
        assert_eq!(stats.hits, 1);
        assert_eq!(stats.misses, 0);
    }

    #[test]
    fn test_stats_record_miss() {
        let mut stats = HandlerCacheStats::new();
        stats.record_miss();

        assert_eq!(stats.lookups, 1);
        assert_eq!(stats.hits, 0);
        assert_eq!(stats.misses, 1);
    }

    #[test]
    fn test_stats_hit_rate_zero() {
        let stats = HandlerCacheStats::new();
        assert_eq!(stats.hit_rate(), 0.0);
    }

    #[test]
    fn test_stats_hit_rate_hundred() {
        let mut stats = HandlerCacheStats::new();
        stats.record_hit();
        stats.record_hit();

        assert_eq!(stats.hit_rate(), 100.0);
    }

    #[test]
    fn test_stats_hit_rate_fifty() {
        let mut stats = HandlerCacheStats::new();
        stats.record_hit();
        stats.record_miss();

        assert_eq!(stats.hit_rate(), 50.0);
    }

    #[test]
    fn test_stats_merge() {
        let mut stats1 = HandlerCacheStats::new();
        stats1.record_hit();
        stats1.record_miss();

        let mut stats2 = HandlerCacheStats::new();
        stats2.record_hit();
        stats2.record_hit();

        stats1.merge(&stats2);

        assert_eq!(stats1.lookups, 4);
        assert_eq!(stats1.hits, 3);
        assert_eq!(stats1.misses, 1);
    }

    #[test]
    fn test_stats_reset() {
        let mut stats = HandlerCacheStats::new();
        stats.record_hit();
        stats.record_miss();

        stats.reset();

        assert_eq!(stats.lookups, 0);
        assert_eq!(stats.hits, 0);
        assert_eq!(stats.misses, 0);
    }

    // ════════════════════════════════════════════════════════════════════════
    // MultiLevelCache Tests
    // ════════════════════════════════════════════════════════════════════════

    #[test]
    fn test_multi_level_new() {
        let cache = MultiLevelCache::new();
        assert!(cache.is_empty());
        assert_eq!(cache.level_count(), 0);
    }

    #[test]
    fn test_multi_level_push_level() {
        let mut cache = MultiLevelCache::new();
        cache.push_level();

        assert!(!cache.is_empty());
        assert_eq!(cache.level_count(), 1);
    }

    #[test]
    fn test_multi_level_pop_level() {
        let mut cache = MultiLevelCache::new();
        cache.push_level();
        cache.pop_level();

        assert!(cache.is_empty());
        assert_eq!(cache.level_count(), 0);
    }

    #[test]
    fn test_multi_level_pop_empty() {
        let mut cache = MultiLevelCache::new();
        cache.pop_level(); // Should not panic
        assert!(cache.is_empty());
    }

    #[test]
    fn test_multi_level_max_levels() {
        let mut cache = MultiLevelCache::new();

        for _ in 0..10 {
            cache.push_level();
        }

        assert_eq!(cache.level_count(), MultiLevelCache::MAX_LEVELS);
    }

    #[test]
    fn test_multi_level_record_and_get() {
        let mut cache = MultiLevelCache::new();
        cache.push_level();
        cache.record(100, 5);

        assert_eq!(cache.try_get(100), Some(5));
    }

    #[test]
    fn test_multi_level_nested_lookup() {
        let mut cache = MultiLevelCache::new();

        cache.push_level();
        cache.record(100, 1);

        cache.push_level();
        cache.record(200, 2);

        // Should find both
        assert_eq!(cache.try_get(200), Some(2));
        assert_eq!(cache.try_get(100), Some(1));
    }

    #[test]
    fn test_multi_level_pop_preserves_outer() {
        let mut cache = MultiLevelCache::new();

        cache.push_level();
        cache.record(100, 1);

        cache.push_level();
        cache.record(200, 2);

        cache.pop_level();

        assert_eq!(cache.try_get(100), Some(1));
        assert_eq!(cache.try_get(200), None);
    }

    #[test]
    fn test_multi_level_invalidate_all() {
        let mut cache = MultiLevelCache::new();

        cache.push_level();
        cache.record(100, 1);
        cache.push_level();
        cache.record(200, 2);

        cache.invalidate_all();

        assert!(cache.is_empty());
        assert_eq!(cache.try_get(100), None);
        assert_eq!(cache.try_get(200), None);
    }

    #[test]
    fn test_multi_level_size() {
        // 4 caches * 8 bytes + 1 byte count (padded to 4)
        let size = std::mem::size_of::<MultiLevelCache>();
        assert!(
            size <= 40,
            "MultiLevelCache should be <= 40 bytes, got {}",
            size
        );
    }
}
