//! Code cache for storing and indexing compiled functions.
//!
//! The code cache provides:
//! - O(1) lookup of compiled code by function ID
//! - Memory management for compiled code regions
//! - Eviction policies for managing memory pressure
//! - Statistics and debugging support

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, RwLock};

use crate::tier2::osr::OsrCompiledCode;

// =============================================================================
// Compiled Entry
// =============================================================================

use crate::backend::x64::ExecutableBuffer;

/// An entry in the code cache representing compiled code for a function.
#[derive(Debug)]
pub struct CompiledEntry {
    /// Unique identifier for the function.
    pub code_id: u64,
    /// Owned buffer containing executable code (keeps memory alive).
    owned_buffer: Option<ExecutableBuffer>,
    /// Base address of the executable code.
    code_ptr: *const u8,
    /// Size of the code in bytes.
    code_size: usize,
    /// Entry point offset from code_ptr.
    entry_offset: usize,
    /// OSR entry points (if any).
    osr_entries: Option<Arc<OsrCompiledCode>>,
    /// Compilation tier (1 = baseline, 2 = optimized).
    tier: u8,
    /// ABI used by the compiled function's return path.
    return_abi: ReturnAbi,
    /// Number of times this code has been called.
    call_count: u64,
}

/// Return ABI for compiled code entry points.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReturnAbi {
    /// Return raw `Value` bits in RAX.
    RawValueBits,
    /// Return encoded exit reason/data in RAX.
    EncodedExitReason,
}

impl CompiledEntry {
    /// Create a new compiled entry with an external code pointer.
    ///
    /// # Safety
    /// The caller must ensure `code_ptr` points to memory that remains
    /// valid for the lifetime of this entry. For JIT-compiled code, prefer
    /// using `from_executable_buffer()` which takes ownership of the memory.
    pub fn new(code_id: u64, code_ptr: *const u8, code_size: usize) -> Self {
        Self {
            code_id,
            owned_buffer: None,
            code_ptr,
            code_size,
            entry_offset: 0,
            osr_entries: None,
            tier: 1,
            return_abi: ReturnAbi::RawValueBits,
            call_count: 0,
        }
    }

    /// Create a new compiled entry that owns its executable memory.
    ///
    /// This is the preferred constructor for JIT-compiled code as it
    /// ensures the executable memory remains valid for the entry's lifetime.
    pub fn from_executable_buffer(code_id: u64, buffer: ExecutableBuffer) -> Self {
        let code_ptr = buffer.as_ptr();
        let code_size = buffer.len();
        Self {
            code_id,
            owned_buffer: Some(buffer),
            code_ptr,
            code_size,
            entry_offset: 0,
            osr_entries: None,
            tier: 1,
            return_abi: ReturnAbi::RawValueBits,
            call_count: 0,
        }
    }

    /// Create with an entry offset.
    pub fn with_entry_offset(mut self, offset: usize) -> Self {
        self.entry_offset = offset;
        self
    }

    /// Set OSR entries.
    pub fn with_osr(mut self, osr: OsrCompiledCode) -> Self {
        self.osr_entries = Some(Arc::new(osr));
        self
    }

    /// Set the compilation tier.
    pub fn with_tier(mut self, tier: u8) -> Self {
        self.tier = tier;
        self
    }

    /// Set return ABI.
    pub fn with_return_abi(mut self, return_abi: ReturnAbi) -> Self {
        self.return_abi = return_abi;
        self
    }

    /// Get the code pointer.
    #[inline]
    pub fn code_ptr(&self) -> *const u8 {
        self.code_ptr
    }

    /// Get the entry point address.
    #[inline]
    pub fn entry_point(&self) -> *const u8 {
        unsafe { self.code_ptr.add(self.entry_offset) }
    }

    /// Get the code size.
    #[inline]
    pub fn code_size(&self) -> usize {
        self.code_size
    }

    /// Get the compilation tier.
    #[inline]
    pub fn tier(&self) -> u8 {
        self.tier
    }

    /// Get the return ABI.
    #[inline]
    pub fn return_abi(&self) -> ReturnAbi {
        self.return_abi
    }

    /// Get OSR entries.
    #[inline]
    pub fn osr_entries(&self) -> Option<&Arc<OsrCompiledCode>> {
        self.osr_entries.as_ref()
    }

    /// Increment call count and return the new value.
    #[inline]
    pub fn increment_call_count(&mut self) -> u64 {
        self.call_count += 1;
        self.call_count
    }

    /// Get call count.
    #[inline]
    pub fn call_count(&self) -> u64 {
        self.call_count
    }

    /// Check if this code contains a given address.
    #[inline]
    pub fn contains_address(&self, addr: usize) -> bool {
        let base = self.code_ptr as usize;
        addr >= base && addr < base + self.code_size
    }
}

// SAFETY: CompiledEntry can be sent between threads (code pointer is const)
unsafe impl Send for CompiledEntry {}
unsafe impl Sync for CompiledEntry {}

// =============================================================================
// Code Cache
// =============================================================================

/// A cache for storing compiled code entries.
///
/// Thread-safe via internal locking.
#[derive(Debug)]
pub struct CodeCache {
    /// Map from code ID to compiled entry.
    entries: RwLock<HashMap<u64, Arc<CompiledEntry>>>,
    /// Total code size in bytes.
    total_size: RwLock<usize>,
    /// Maximum allowed size.
    max_size: usize,
    /// Lookup hit counter.
    hits: AtomicU64,
    /// Lookup miss counter.
    misses: AtomicU64,
    /// Insert counter.
    insertions: AtomicU64,
    /// Eviction counter.
    evictions: AtomicU64,
}

impl CodeCache {
    /// Create a new code cache with the given maximum size.
    pub fn new(max_size: usize) -> Self {
        Self {
            entries: RwLock::new(HashMap::new()),
            total_size: RwLock::new(0),
            max_size,
            hits: AtomicU64::new(0),
            misses: AtomicU64::new(0),
            insertions: AtomicU64::new(0),
            evictions: AtomicU64::new(0),
        }
    }

    /// Look up compiled code by function ID.
    #[inline]
    pub fn lookup(&self, code_id: u64) -> Option<Arc<CompiledEntry>> {
        let entries = self.entries.read().unwrap();
        let result = entries.get(&code_id).cloned();
        drop(entries);

        // Update stats
        if result.is_some() {
            self.hits.fetch_add(1, Ordering::Relaxed);
        } else {
            self.misses.fetch_add(1, Ordering::Relaxed);
        }

        result
    }

    /// Insert compiled code into the cache.
    ///
    /// Returns the previous entry if one existed.
    pub fn insert(&self, entry: CompiledEntry) -> Option<Arc<CompiledEntry>> {
        let code_id = entry.code_id;
        let code_size = entry.code_size;
        let new_entry = Arc::new(entry);

        // Insert while enforcing memory limit.
        let mut entries = self.entries.write().unwrap();
        let mut total = self.total_size.write().unwrap();

        while *total + code_size > self.max_size && !entries.is_empty() {
            let victim_id = entries.keys().next().copied();
            if let Some(victim_id) = victim_id {
                if let Some(victim) = entries.remove(&victim_id) {
                    *total = total.saturating_sub(victim.code_size);
                    self.evictions.fetch_add(1, Ordering::Relaxed);
                }
            } else {
                break;
            }
        }

        let old = entries.insert(code_id, new_entry);

        // Update size tracking
        if let Some(ref prev) = old {
            *total = total.saturating_sub(prev.code_size);
        }
        *total += code_size;

        // Update stats
        self.insertions.fetch_add(1, Ordering::Relaxed);

        old
    }

    /// Remove compiled code from the cache.
    pub fn remove(&self, code_id: u64) -> Option<Arc<CompiledEntry>> {
        let mut entries = self.entries.write().unwrap();
        let removed = entries.remove(&code_id);

        if let Some(ref entry) = removed {
            let mut total = self.total_size.write().unwrap();
            *total -= entry.code_size;
        }

        removed
    }

    /// Find entry containing the given instruction pointer.
    pub fn find_by_ip(&self, ip: usize) -> Option<Arc<CompiledEntry>> {
        let entries = self.entries.read().unwrap();
        for entry in entries.values() {
            if entry.contains_address(ip) {
                return Some(Arc::clone(entry));
            }
        }
        None
    }

    /// Get the number of entries in the cache.
    #[inline]
    pub fn len(&self) -> usize {
        self.entries.read().unwrap().len()
    }

    /// Check if the cache is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get the total code size.
    #[inline]
    pub fn total_size(&self) -> usize {
        *self.total_size.read().unwrap()
    }

    /// Get cache statistics.
    #[inline]
    pub fn stats(&self) -> CodeCacheStats {
        CodeCacheStats {
            hits: self.hits.load(Ordering::Relaxed),
            misses: self.misses.load(Ordering::Relaxed),
            insertions: self.insertions.load(Ordering::Relaxed),
            evictions: self.evictions.load(Ordering::Relaxed),
        }
    }

    /// Clear the entire cache.
    pub fn clear(&self) {
        let mut entries = self.entries.write().unwrap();
        entries.clear();
        *self.total_size.write().unwrap() = 0;
    }

    /// Iterate over all entries (for debugging/profiling).
    pub fn entries(&self) -> Vec<Arc<CompiledEntry>> {
        self.entries.read().unwrap().values().cloned().collect()
    }
}

// =============================================================================
// Statistics
// =============================================================================

/// Statistics for the code cache.
#[derive(Debug, Default, Clone)]
pub struct CodeCacheStats {
    /// Number of cache hits.
    pub hits: u64,
    /// Number of cache misses.
    pub misses: u64,
    /// Number of insertions.
    pub insertions: u64,
    /// Number of evictions.
    pub evictions: u64,
}

impl CodeCacheStats {
    /// Calculate hit rate.
    #[inline]
    pub fn hit_rate(&self) -> f64 {
        let total = self.hits + self.misses;
        if total == 0 {
            0.0
        } else {
            self.hits as f64 / total as f64
        }
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn dummy_code_ptr() -> *const u8 {
        0x10000usize as *const u8
    }

    #[test]
    fn test_compiled_entry_creation() {
        let entry = CompiledEntry::new(1, dummy_code_ptr(), 100);
        assert_eq!(entry.code_id, 1);
        assert_eq!(entry.code_size(), 100);
        assert_eq!(entry.tier(), 1);
        assert_eq!(entry.return_abi(), ReturnAbi::RawValueBits);
    }

    #[test]
    fn test_compiled_entry_builder_pattern() {
        let entry = CompiledEntry::new(1, dummy_code_ptr(), 100)
            .with_entry_offset(16)
            .with_tier(2)
            .with_return_abi(ReturnAbi::EncodedExitReason);
        assert_eq!(entry.tier(), 2);
        assert_eq!(entry.return_abi(), ReturnAbi::EncodedExitReason);
        assert_eq!(entry.entry_point() as usize, dummy_code_ptr() as usize + 16);
    }

    #[test]
    fn test_code_cache_insert_lookup() {
        let cache = CodeCache::new(1024 * 1024);

        let entry = CompiledEntry::new(42, dummy_code_ptr(), 100);
        assert!(cache.insert(entry).is_none());

        let found = cache.lookup(42);
        assert!(found.is_some());
        assert_eq!(found.unwrap().code_id, 42);

        assert!(cache.lookup(999).is_none());
    }

    #[test]
    fn test_code_cache_remove() {
        let cache = CodeCache::new(1024 * 1024);

        let entry = CompiledEntry::new(42, dummy_code_ptr(), 100);
        cache.insert(entry);

        assert_eq!(cache.total_size(), 100);

        let removed = cache.remove(42);
        assert!(removed.is_some());
        assert_eq!(cache.total_size(), 0);
        assert!(cache.is_empty());
    }

    #[test]
    fn test_code_cache_stats() {
        let cache = CodeCache::new(1024 * 1024);

        // Miss
        cache.lookup(1);
        // Insert
        cache.insert(CompiledEntry::new(1, dummy_code_ptr(), 50));
        // Hit
        cache.lookup(1);
        // Miss
        cache.lookup(2);

        let stats = cache.stats();
        assert_eq!(stats.hits, 1);
        assert_eq!(stats.misses, 2);
        assert_eq!(stats.insertions, 1);
    }

    #[test]
    fn test_code_cache_evicts_when_capacity_exceeded() {
        let cache = CodeCache::new(150);

        cache.insert(CompiledEntry::new(1, dummy_code_ptr(), 100));
        cache.insert(CompiledEntry::new(2, (dummy_code_ptr() as usize + 64) as *const u8, 100));

        assert!(cache.total_size() <= 150);
        assert_eq!(cache.len(), 1);

        let stats = cache.stats();
        assert_eq!(stats.evictions, 1);
        assert_eq!(stats.insertions, 2);
    }

    #[test]
    fn test_find_by_ip() {
        let cache = CodeCache::new(1024 * 1024);

        let ptr = 0x20000usize as *const u8;
        cache.insert(CompiledEntry::new(1, ptr, 100));

        // Within range
        let found = cache.find_by_ip(0x20050);
        assert!(found.is_some());

        // Out of range
        assert!(cache.find_by_ip(0x10000).is_none());
        assert!(cache.find_by_ip(0x20100).is_none());
    }
}
