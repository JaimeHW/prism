//! Lock-free atomic mark bitmap for concurrent GC marking.
//!
//! Provides O(1) concurrent-safe mark testing and setting using atomic
//! bit operations. Used by the concurrent marker to track which objects
//! are reachable without requiring locks.
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────────────┐
//! │  AtomicMarkBitmap                                                        │
//! │                                                                         │
//! │  Heap address range: [base, base + coverage)                            │
//! │                                                                         │
//! │  ┌──────┬──────┬──────┬──────┬──────┬──────┬──────┬──────┐            │
//! │  │ W0   │ W1   │ W2   │ W3   │ W4   │ W5   │ W6   │ W7   │ ...       │
//! │  │[63:0]│[63:0]│[63:0]│[63:0]│[63:0]│[63:0]│[63:0]│[63:0]│            │
//! │  └──────┴──────┴──────┴──────┴──────┴──────┴──────┴──────┘            │
//! │                                                                         │
//! │  Each bit maps to one granule (MIN_OBJECT_ALIGNMENT bytes).            │
//! │  Bit N of word W → address: base + (W * 64 + N) * granularity         │
//! │                                                                         │
//! │  Operations:                                                            │
//! │    mark(addr)       → fetch_or (lock-free, returns previous state)     │
//! │    is_marked(addr)  → load + bit test (wait-free)                      │
//! │    clear_all()      → store 0 to all words                              │
//! └─────────────────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Performance
//!
//! - **Mark**: Single `fetch_or` — lock-free, no CAS retry loop
//! - **Test**: Single `load` + bit test — wait-free
//! - **Memory**: 1 bit per granule (e.g., 8-byte alignment → 16MB covers 1GB heap)
//! - **Cache**: Words are naturally 8-byte aligned; padding available for
//!   cache-line alignment if needed

use std::sync::atomic::{AtomicU64, Ordering};

// =============================================================================
// Constants
// =============================================================================

/// Minimum object alignment in bytes.
///
/// Objects are always aligned to at least this many bytes, so we can use
/// each bit to represent one alignment granule rather than one byte.
/// This reduces bitmap memory usage by a factor of `MIN_OBJECT_ALIGNMENT`.
pub const MIN_OBJECT_ALIGNMENT: usize = 8;

/// Bits per word (AtomicU64).
const BITS_PER_WORD: usize = 64;

/// Log2 of bits per word, for shift-based division.
const LOG2_BITS_PER_WORD: u32 = 6;

/// Mask for extracting bit index within a word.
const BIT_INDEX_MASK: usize = BITS_PER_WORD - 1;

// =============================================================================
// AtomicMarkBitmap
// =============================================================================

/// Lock-free atomic mark bitmap for concurrent GC marking.
///
/// Maps a contiguous heap address range to a dense bit vector stored in
/// `AtomicU64` words. Supports concurrent marking from multiple threads
/// without locks via `fetch_or`.
///
/// # Granularity
///
/// Each bit represents one [`MIN_OBJECT_ALIGNMENT`]-byte granule.
/// For 8-byte alignment this gives 16MB of bitmap per 1GB of heap.
///
/// # Thread Safety
///
/// All operations are safe to call concurrently:
/// - `mark()`: lock-free (single `fetch_or`)
/// - `is_marked()`: wait-free (single `load`)
/// - `clear_all()`: should only be called when no concurrent marking is active
pub struct AtomicMarkBitmap {
    /// The bitmap words.
    words: Box<[AtomicU64]>,

    /// Base address of the covered region.
    base: usize,

    /// Number of granules covered.
    granule_count: usize,

    /// Granularity in bytes (= MIN_OBJECT_ALIGNMENT).
    granularity: usize,

    /// Log2 of granularity for shift-based division.
    granularity_shift: u32,
}

impl AtomicMarkBitmap {
    /// Create a new bitmap covering the address range `[base, base + size)`.
    ///
    /// # Arguments
    ///
    /// - `base`: Start address of the covered heap region
    /// - `size`: Size of the covered region in bytes
    ///
    /// # Panics
    ///
    /// Panics if `size` is 0.
    pub fn new(base: usize, size: usize) -> Self {
        Self::with_granularity(base, size, MIN_OBJECT_ALIGNMENT)
    }

    /// Create a bitmap with a custom granularity.
    ///
    /// # Arguments
    ///
    /// - `base`: Start address of the covered heap region
    /// - `size`: Size of the covered region in bytes
    /// - `granularity`: Bytes per bit (must be a power of two, >= 8)
    ///
    /// # Panics
    ///
    /// Panics if `granularity` is not a power of two, is less than 8, or `size` is 0.
    pub fn with_granularity(base: usize, size: usize, granularity: usize) -> Self {
        assert!(size > 0, "Bitmap size must be > 0");
        assert!(
            granularity.is_power_of_two() && granularity >= 8,
            "Granularity must be a power of two >= 8"
        );

        let granularity_shift = granularity.trailing_zeros();
        let granule_count = (size + granularity - 1) >> granularity_shift;
        let word_count = (granule_count + BITS_PER_WORD - 1) >> LOG2_BITS_PER_WORD;

        // Allocate zeroed words
        let words: Vec<AtomicU64> = (0..word_count).map(|_| AtomicU64::new(0)).collect();

        Self {
            words: words.into_boxed_slice(),
            base,
            granule_count,
            granularity,
            granularity_shift,
        }
    }

    // =========================================================================
    // Core Operations
    // =========================================================================

    /// Mark the object at `addr` as live.
    ///
    /// Returns `true` if the object was **newly** marked (previously unmarked).
    /// Returns `false` if the object was already marked.
    ///
    /// # Lock-free
    ///
    /// Uses a single `fetch_or` with `Ordering::AcqRel` — no CAS retry loop.
    /// Multiple threads can mark concurrently without contention beyond the
    /// atomic operation itself.
    ///
    /// # Arguments
    ///
    /// - `addr`: Raw address of the object to mark
    ///
    /// # Returns
    ///
    /// - `Some(true)`: Newly marked
    /// - `Some(false)`: Already marked
    /// - `None`: Address out of range
    #[inline]
    pub fn mark(&self, addr: usize) -> Option<bool> {
        let (word_idx, bit_mask) = self.addr_to_indices(addr)?;

        // fetch_or: atomically sets the bit and returns the previous value.
        // AcqRel ensures:
        //   - Acquire: we see all writes from threads that previously marked this word
        //   - Release: our mark is visible to subsequent reads from other threads
        let prev = self.words[word_idx].fetch_or(bit_mask, Ordering::AcqRel);

        // If the bit was already set, the object was already marked
        Some((prev & bit_mask) == 0)
    }

    /// Test whether the object at `addr` is marked.
    ///
    /// # Wait-free
    ///
    /// Single atomic `load` + bit test. No retry, no blocking.
    #[inline]
    pub fn is_marked(&self, addr: usize) -> bool {
        match self.addr_to_indices(addr) {
            Some((word_idx, bit_mask)) => {
                (self.words[word_idx].load(Ordering::Acquire) & bit_mask) != 0
            }
            None => false,
        }
    }

    /// Clear all marks.
    ///
    /// # Safety Contract
    ///
    /// Must only be called when no concurrent marking is in progress.
    /// Typically called at the beginning of a new GC cycle.
    pub fn clear_all(&self) {
        for word in self.words.iter() {
            word.store(0, Ordering::Release);
        }
    }

    /// Clear the mark for a single address.
    ///
    /// Returns `true` if the address was previously marked.
    #[inline]
    pub fn clear(&self, addr: usize) -> Option<bool> {
        let (word_idx, bit_mask) = self.addr_to_indices(addr)?;
        let prev = self.words[word_idx].fetch_and(!bit_mask, Ordering::AcqRel);
        Some((prev & bit_mask) != 0)
    }

    // =========================================================================
    // Statistics
    // =========================================================================

    /// Count the total number of marked bits.
    ///
    /// Uses hardware `popcount` on each word for optimal performance.
    /// This is a snapshot — concurrent modifications may cause the count
    /// to be approximate.
    pub fn marked_count(&self) -> usize {
        self.words
            .iter()
            .map(|w| w.load(Ordering::Relaxed).count_ones() as usize)
            .sum()
    }

    /// Check if any bits are marked.
    pub fn has_any_marked(&self) -> bool {
        self.words.iter().any(|w| w.load(Ordering::Relaxed) != 0)
    }

    /// Check if all bits are cleared (no marks).
    #[inline]
    pub fn is_empty(&self) -> bool {
        !self.has_any_marked()
    }

    // =========================================================================
    // Iteration
    // =========================================================================

    /// Iterate over all marked addresses.
    ///
    /// Calls `f` with the address of each marked object. Iteration order
    /// is ascending by address. This is a snapshot — addresses marked
    /// concurrently during iteration may or may not be visited.
    pub fn for_each_marked<F>(&self, mut f: F)
    where
        F: FnMut(usize),
    {
        for (word_idx, word) in self.words.iter().enumerate() {
            let mut bits = word.load(Ordering::Relaxed);
            while bits != 0 {
                // Extract lowest set bit position
                let bit_pos = bits.trailing_zeros() as usize;
                let granule_idx = word_idx * BITS_PER_WORD + bit_pos;
                let addr = self.base + (granule_idx << self.granularity_shift);
                f(addr);
                // Clear lowest set bit
                bits &= bits - 1;
            }
        }
    }

    /// Iterate over marked addresses in a specific word range.
    ///
    /// Useful for parallel sweeping where each thread handles a chunk.
    pub fn for_each_marked_in_range<F>(&self, start_addr: usize, end_addr: usize, mut f: F)
    where
        F: FnMut(usize),
    {
        let start_granule = match start_addr.checked_sub(self.base) {
            Some(offset) => offset >> self.granularity_shift,
            None => 0,
        };
        let end_granule = match end_addr.checked_sub(self.base) {
            Some(offset) => {
                ((offset + self.granularity - 1) >> self.granularity_shift).min(self.granule_count)
            }
            None => return,
        };

        let start_word = start_granule >> LOG2_BITS_PER_WORD;
        let end_word =
            ((end_granule + BITS_PER_WORD - 1) >> LOG2_BITS_PER_WORD).min(self.words.len());

        for word_idx in start_word..end_word {
            let mut bits = self.words[word_idx].load(Ordering::Relaxed);
            while bits != 0 {
                let bit_pos = bits.trailing_zeros() as usize;
                let granule_idx = word_idx * BITS_PER_WORD + bit_pos;
                if granule_idx >= start_granule && granule_idx < end_granule {
                    let addr = self.base + (granule_idx << self.granularity_shift);
                    f(addr);
                }
                bits &= bits - 1;
            }
        }
    }

    // =========================================================================
    // Capacity / Info
    // =========================================================================

    /// Get the number of bits (granules) in the bitmap.
    #[inline]
    pub fn capacity(&self) -> usize {
        self.granule_count
    }

    /// Get the number of words in the underlying storage.
    #[inline]
    pub fn word_count(&self) -> usize {
        self.words.len()
    }

    /// Get the base address of the covered region.
    #[inline]
    pub fn base(&self) -> usize {
        self.base
    }

    /// Get the granularity in bytes.
    #[inline]
    pub fn granularity(&self) -> usize {
        self.granularity
    }

    /// Get the total coverage in bytes.
    #[inline]
    pub fn coverage(&self) -> usize {
        self.granule_count << self.granularity_shift
    }

    /// Check if an address is within the covered range.
    #[inline]
    pub fn contains(&self, addr: usize) -> bool {
        addr >= self.base && addr < self.base + self.coverage()
    }

    /// Memory used by the bitmap itself (in bytes).
    #[inline]
    pub fn memory_usage(&self) -> usize {
        self.words.len() * std::mem::size_of::<AtomicU64>()
    }

    // =========================================================================
    // Internal Helpers
    // =========================================================================

    /// Convert an address to (word_index, bit_mask).
    ///
    /// Returns `None` if the address is outside the covered range.
    #[inline(always)]
    fn addr_to_indices(&self, addr: usize) -> Option<(usize, u64)> {
        // Fast check: address must be at or above the base
        let offset = addr.checked_sub(self.base)?;

        // Convert byte offset to granule index
        let granule_idx = offset >> self.granularity_shift;

        // Bounds check
        if granule_idx >= self.granule_count {
            return None;
        }

        let word_idx = granule_idx >> LOG2_BITS_PER_WORD;
        let bit_idx = granule_idx & BIT_INDEX_MASK;
        let bit_mask = 1u64 << bit_idx;

        Some((word_idx, bit_mask))
    }
}

// Safety: AtomicMarkBitmap is Send + Sync because all its mutable state
// is behind AtomicU64 operations.
unsafe impl Send for AtomicMarkBitmap {}
unsafe impl Sync for AtomicMarkBitmap {}

impl std::fmt::Debug for AtomicMarkBitmap {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AtomicMarkBitmap")
            .field("base", &format_args!("0x{:x}", self.base))
            .field("coverage", &self.coverage())
            .field("granularity", &self.granularity)
            .field("words", &self.words.len())
            .field("marked", &self.marked_count())
            .finish()
    }
}
