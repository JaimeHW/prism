//! Frame storage for generators.
//!
//! This module provides union-based storage for suspended generator frames.
//! Small generators (≤8 values) use inline storage with zero allocation.
//! Large generators spill to heap-allocated boxed storage.
//!
//! # Liveness Bitmap
//!
//! At compile time, we determine which registers are live at each yield point.
//! Only live registers are captured/restored, minimizing memory and copy overhead.
//!
//! # Storage Layout
//!
//! ```text
//! Union FrameStorage {
//!     inline: [Value; 8]       // 64 bytes, zero allocation
//!     boxed: Box<[Value]>      // Heap for large generators
//! }
//! ```

use prism_core::Value;
use std::fmt;
use std::mem::MaybeUninit;

// ============================================================================
// Constants
// ============================================================================

/// Maximum inline values before spilling to heap.
pub const INLINE_CAPACITY: usize = 8;

/// Maximum yield points per generator (30-bit resume index).
pub const MAX_YIELD_POINTS: usize = (1 << 30) - 1;

// ============================================================================
// Liveness Map
// ============================================================================

/// Bitmap indicating which registers are live at a yield point.
///
/// Supports up to 64 registers with a single u64. For functions
/// with more than 64 locals, we use boxed storage.
///
/// # Usage
///
/// ```ignore
/// // Compiler generates: only r0, r2, r5 are live at this yield
/// const YIELD_0_LIVENESS: LivenessMap = LivenessMap::from_bits(0b00100101);
/// ```
#[derive(Clone, Copy, PartialEq, Eq, Default)]
pub struct LivenessMap {
    bits: u64,
}

impl LivenessMap {
    /// Empty liveness (no registers live).
    pub const EMPTY: Self = Self { bits: 0 };

    /// All registers live (conservative fallback).
    pub const ALL: Self = Self { bits: u64::MAX };

    /// Creates an empty liveness map (convenience function).
    #[inline(always)]
    pub const fn empty() -> Self {
        Self::EMPTY
    }

    /// Returns true if no registers are live.
    #[inline(always)]
    pub const fn is_empty(self) -> bool {
        self.bits == 0
    }

    /// Creates a liveness map from raw bits.
    #[inline(always)]
    pub const fn from_bits(bits: u64) -> Self {
        Self { bits }
    }

    /// Returns the raw bits.
    #[inline(always)]
    pub const fn bits(self) -> u64 {
        self.bits
    }

    /// Returns the number of live registers.
    #[inline(always)]
    pub const fn count(self) -> u32 {
        self.bits.count_ones()
    }

    /// Checks if a register is live.
    #[inline(always)]
    pub const fn is_live(self, reg: u8) -> bool {
        (self.bits & (1u64 << reg)) != 0
    }

    /// Sets a register as live.
    #[inline(always)]
    pub const fn with_live(self, reg: u8) -> Self {
        Self {
            bits: self.bits | (1u64 << reg),
        }
    }

    /// Clears a register.
    #[inline(always)]
    pub const fn without(self, reg: u8) -> Self {
        Self {
            bits: self.bits & !(1u64 << reg),
        }
    }

    /// Returns an iterator over live register indices.
    #[inline]
    pub fn iter(self) -> LivenessIter {
        LivenessIter { bits: self.bits }
    }

    /// Computes the compact index for a live register.
    /// This is the position in the stored values array.
    #[inline]
    pub const fn compact_index(self, reg: u8) -> usize {
        debug_assert!(self.is_live(reg));
        // Count bits below this register
        let mask = (1u64 << reg) - 1;
        (self.bits & mask).count_ones() as usize
    }

    /// Returns whether all values fit inline.
    #[inline(always)]
    pub const fn fits_inline(self) -> bool {
        self.count() as usize <= INLINE_CAPACITY
    }
}

impl fmt::Debug for LivenessMap {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "LivenessMap({:#018x}, {} live)", self.bits, self.count())
    }
}

/// Iterator over live register indices.
pub struct LivenessIter {
    bits: u64,
}

impl Iterator for LivenessIter {
    type Item = u8;

    #[inline]
    fn next(&mut self) -> Option<u8> {
        if self.bits == 0 {
            return None;
        }
        let idx = self.bits.trailing_zeros() as u8;
        self.bits &= self.bits - 1; // Clear lowest set bit
        Some(idx)
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let count = self.bits.count_ones() as usize;
        (count, Some(count))
    }
}

impl ExactSizeIterator for LivenessIter {}

// ============================================================================
// Frame Storage
// ============================================================================

/// Storage for suspended generator frame values.
///
/// Uses union-based storage: inline for small, boxed for large.
pub struct FrameStorage {
    /// Inline storage for small generators.
    inline: [MaybeUninit<Value>; INLINE_CAPACITY],
    /// Number of values currently stored.
    len: u8,
    /// Whether we're using boxed storage.
    is_boxed: bool,
    /// Boxed storage for large generators (only valid if is_boxed).
    boxed: Option<Box<[Value]>>,
}

impl FrameStorage {
    /// Creates empty inline storage.
    #[inline]
    pub fn new() -> Self {
        Self {
            inline: [MaybeUninit::uninit(); INLINE_CAPACITY],
            len: 0,
            is_boxed: false,
            boxed: None,
        }
    }

    /// Creates storage with the given capacity.
    #[inline]
    pub fn with_capacity(capacity: usize) -> Self {
        if capacity <= INLINE_CAPACITY {
            Self::new()
        } else {
            Self {
                inline: [MaybeUninit::uninit(); INLINE_CAPACITY],
                len: 0,
                is_boxed: true,
                boxed: Some(vec![Value::none(); capacity].into_boxed_slice()),
            }
        }
    }

    /// Returns the number of stored values.
    #[inline(always)]
    pub fn len(&self) -> usize {
        self.len as usize
    }

    /// Returns true if empty.
    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Returns true if using boxed storage.
    #[inline(always)]
    pub fn is_boxed(&self) -> bool {
        self.is_boxed
    }

    /// Returns the capacity.
    #[inline]
    pub fn capacity(&self) -> usize {
        if self.is_boxed {
            self.boxed.as_ref().map_or(0, |b| b.len())
        } else {
            INLINE_CAPACITY
        }
    }

    /// Gets a value by index.
    ///
    /// # Safety
    /// Index must be valid.
    #[inline]
    pub fn get(&self, index: usize) -> Value {
        debug_assert!(index < self.len());
        if self.is_boxed {
            self.boxed.as_ref().unwrap()[index]
        } else {
            // SAFETY: index < len, so value is initialized
            unsafe { self.inline[index].assume_init() }
        }
    }

    /// Sets a value by index.
    ///
    /// # Safety
    /// Index must be within capacity.
    #[inline]
    pub fn set(&mut self, index: usize, value: Value) {
        debug_assert!(index < self.capacity());
        if self.is_boxed {
            self.boxed.as_mut().unwrap()[index] = value;
        } else {
            self.inline[index] = MaybeUninit::new(value);
        }
        if index >= self.len() {
            self.len = (index + 1) as u8;
        }
    }

    /// Captures live values from a register file.
    #[inline]
    pub fn capture(&mut self, registers: &[Value; 256], liveness: LivenessMap) {
        let count = liveness.count() as usize;

        // Ensure capacity
        if count > INLINE_CAPACITY && !self.is_boxed {
            self.boxed = Some(vec![Value::none(); count].into_boxed_slice());
            self.is_boxed = true;
        }

        // Copy live values
        let mut idx = 0;
        for reg in liveness.iter() {
            self.set(idx, registers[reg as usize]);
            idx += 1;
        }
        self.len = count as u8;
    }

    /// Restores live values to a register file.
    #[inline]
    pub fn restore(&self, registers: &mut [Value; 256], liveness: LivenessMap) {
        let mut idx = 0;
        for reg in liveness.iter() {
            registers[reg as usize] = self.get(idx);
            idx += 1;
        }
    }

    /// Clears all stored values.
    #[inline]
    pub fn clear(&mut self) {
        self.len = 0;
    }
}

impl Default for FrameStorage {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

impl Clone for FrameStorage {
    fn clone(&self) -> Self {
        if self.is_boxed {
            Self {
                inline: [MaybeUninit::uninit(); INLINE_CAPACITY],
                len: self.len,
                is_boxed: true,
                boxed: self.boxed.clone(),
            }
        } else {
            let mut inline = [MaybeUninit::uninit(); INLINE_CAPACITY];
            for i in 0..self.len as usize {
                inline[i] = MaybeUninit::new(unsafe { self.inline[i].assume_init() });
            }
            Self {
                inline,
                len: self.len,
                is_boxed: false,
                boxed: None,
            }
        }
    }
}

impl fmt::Debug for FrameStorage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("FrameStorage")
            .field("len", &self.len)
            .field("is_boxed", &self.is_boxed)
            .field("capacity", &self.capacity())
            .finish()
    }
}

// SAFETY: FrameStorage contains Values which are Copy and thread-safe
unsafe impl Send for FrameStorage {}
unsafe impl Sync for FrameStorage {}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ════════════════════════════════════════════════════════════════════════
    // LivenessMap Tests
    // ════════════════════════════════════════════════════════════════════════

    #[test]
    fn test_liveness_size() {
        assert_eq!(std::mem::size_of::<LivenessMap>(), 8);
    }

    #[test]
    fn test_liveness_empty() {
        let map = LivenessMap::EMPTY;
        assert_eq!(map.count(), 0);
        assert!(!map.is_live(0));
        assert!(!map.is_live(63));
    }

    #[test]
    fn test_liveness_all() {
        let map = LivenessMap::ALL;
        assert_eq!(map.count(), 64);
        assert!(map.is_live(0));
        assert!(map.is_live(63));
    }

    #[test]
    fn test_liveness_from_bits() {
        let map = LivenessMap::from_bits(0b101);
        assert!(map.is_live(0));
        assert!(!map.is_live(1));
        assert!(map.is_live(2));
        assert_eq!(map.count(), 2);
    }

    #[test]
    fn test_liveness_with_live() {
        let map = LivenessMap::EMPTY.with_live(5).with_live(10);
        assert!(map.is_live(5));
        assert!(map.is_live(10));
        assert!(!map.is_live(0));
        assert_eq!(map.count(), 2);
    }

    #[test]
    fn test_liveness_without() {
        let map = LivenessMap::from_bits(0b111).without(1);
        assert!(map.is_live(0));
        assert!(!map.is_live(1));
        assert!(map.is_live(2));
    }

    #[test]
    fn test_liveness_iter() {
        let map = LivenessMap::from_bits(0b10101);
        let regs: Vec<_> = map.iter().collect();
        assert_eq!(regs, vec![0, 2, 4]);
    }

    #[test]
    fn test_liveness_iter_empty() {
        let map = LivenessMap::EMPTY;
        let regs: Vec<_> = map.iter().collect();
        assert!(regs.is_empty());
    }

    #[test]
    fn test_liveness_iter_exact_size() {
        let map = LivenessMap::from_bits(0b1111);
        let iter = map.iter();
        assert_eq!(iter.len(), 4);
    }

    #[test]
    fn test_liveness_compact_index() {
        let map = LivenessMap::from_bits(0b10101);
        // Live: 0, 2, 4 → compact indices 0, 1, 2
        assert_eq!(map.compact_index(0), 0);
        assert_eq!(map.compact_index(2), 1);
        assert_eq!(map.compact_index(4), 2);
    }

    #[test]
    fn test_liveness_fits_inline() {
        assert!(LivenessMap::from_bits(0b11111111).fits_inline()); // 8
        assert!(!LivenessMap::from_bits(0b111111111).fits_inline()); // 9
    }

    #[test]
    fn test_liveness_debug() {
        let map = LivenessMap::from_bits(0xFF);
        let debug = format!("{:?}", map);
        assert!(debug.contains("8 live"));
    }

    #[test]
    fn test_liveness_eq() {
        assert_eq!(LivenessMap::from_bits(0b101), LivenessMap::from_bits(0b101));
        assert_ne!(LivenessMap::from_bits(0b101), LivenessMap::from_bits(0b110));
    }

    #[test]
    fn test_liveness_high_bit_registers() {
        let map = LivenessMap::EMPTY.with_live(63);
        assert!(map.is_live(63));
        assert_eq!(map.count(), 1);
        assert_eq!(map.compact_index(63), 0);
    }

    // ════════════════════════════════════════════════════════════════════════
    // FrameStorage Tests
    // ════════════════════════════════════════════════════════════════════════

    #[test]
    fn test_storage_size() {
        // Should be reasonably small
        let size = std::mem::size_of::<FrameStorage>();
        assert!(size <= 128, "FrameStorage too large: {}", size);
    }

    #[test]
    fn test_storage_new() {
        let storage = FrameStorage::new();
        assert_eq!(storage.len(), 0);
        assert!(storage.is_empty());
        assert!(!storage.is_boxed());
        assert_eq!(storage.capacity(), INLINE_CAPACITY);
    }

    #[test]
    fn test_storage_with_capacity_small() {
        let storage = FrameStorage::with_capacity(4);
        assert!(!storage.is_boxed());
        assert_eq!(storage.capacity(), INLINE_CAPACITY);
    }

    #[test]
    fn test_storage_with_capacity_large() {
        let storage = FrameStorage::with_capacity(20);
        assert!(storage.is_boxed());
        assert_eq!(storage.capacity(), 20);
    }

    #[test]
    fn test_storage_set_get_inline() {
        let mut storage = FrameStorage::new();
        storage.set(0, Value::int(42).unwrap());
        storage.set(1, Value::float(3.14));

        assert_eq!(storage.len(), 2);
        assert_eq!(storage.get(0).as_int().unwrap(), 42);
        assert!((storage.get(1).as_float().unwrap() - 3.14).abs() < 1e-10);
    }

    #[test]
    fn test_storage_set_get_boxed() {
        let mut storage = FrameStorage::with_capacity(20);
        storage.set(0, Value::int(100).unwrap());
        storage.set(15, Value::int(200).unwrap());

        assert_eq!(storage.get(0).as_int().unwrap(), 100);
        assert_eq!(storage.get(15).as_int().unwrap(), 200);
    }

    #[test]
    fn test_storage_capture_inline() {
        let mut registers = [Value::none(); 256];
        registers[0] = Value::int(10).unwrap();
        registers[2] = Value::int(20).unwrap();
        registers[5] = Value::int(30).unwrap();

        let liveness = LivenessMap::from_bits(0b100101); // regs 0, 2, 5
        let mut storage = FrameStorage::new();
        storage.capture(&registers, liveness);

        assert_eq!(storage.len(), 3);
        assert_eq!(storage.get(0).as_int().unwrap(), 10);
        assert_eq!(storage.get(1).as_int().unwrap(), 20);
        assert_eq!(storage.get(2).as_int().unwrap(), 30);
    }

    #[test]
    fn test_storage_restore() {
        let mut storage = FrameStorage::new();
        storage.set(0, Value::int(100).unwrap());
        storage.set(1, Value::int(200).unwrap());
        storage.set(2, Value::int(300).unwrap());

        let liveness = LivenessMap::from_bits(0b100101); // regs 0, 2, 5
        let mut registers = [Value::none(); 256];
        storage.restore(&mut registers, liveness);

        assert_eq!(registers[0].as_int().unwrap(), 100);
        assert_eq!(registers[2].as_int().unwrap(), 200);
        assert_eq!(registers[5].as_int().unwrap(), 300);
    }

    #[test]
    fn test_storage_capture_restore_roundtrip() {
        let mut orig_registers = [Value::none(); 256];
        orig_registers[1] = Value::int(42).unwrap();
        orig_registers[3] = Value::float(2.718);
        orig_registers[7] = Value::bool(true);

        let liveness = LivenessMap::from_bits(0b10001010); // regs 1, 3, 7

        let mut storage = FrameStorage::new();
        storage.capture(&orig_registers, liveness);

        let mut new_registers = [Value::none(); 256];
        storage.restore(&mut new_registers, liveness);

        assert_eq!(new_registers[1].as_int().unwrap(), 42);
        assert!((new_registers[3].as_float().unwrap() - 2.718).abs() < 1e-10);
        assert!(new_registers[7].as_bool().unwrap());
    }

    #[test]
    fn test_storage_capture_spills_to_boxed() {
        let mut registers = [Value::none(); 256];
        for i in 0..16 {
            registers[i] = Value::int(i as i64).unwrap();
        }

        // 16 live registers - exceeds inline capacity
        let liveness = LivenessMap::from_bits(0xFFFF);
        let mut storage = FrameStorage::new();
        storage.capture(&registers, liveness);

        assert!(storage.is_boxed());
        assert_eq!(storage.len(), 16);

        for i in 0..16 {
            assert_eq!(storage.get(i).as_int().unwrap(), i as i64);
        }
    }

    #[test]
    fn test_storage_clear() {
        let mut storage = FrameStorage::new();
        storage.set(0, Value::int(1).unwrap());
        storage.set(1, Value::int(2).unwrap());
        assert_eq!(storage.len(), 2);

        storage.clear();
        assert_eq!(storage.len(), 0);
        assert!(storage.is_empty());
    }

    #[test]
    fn test_storage_clone_inline() {
        let mut storage = FrameStorage::new();
        storage.set(0, Value::int(42).unwrap());
        storage.set(1, Value::int(99).unwrap());

        let cloned = storage.clone();
        assert_eq!(cloned.len(), 2);
        assert_eq!(cloned.get(0).as_int().unwrap(), 42);
        assert_eq!(cloned.get(1).as_int().unwrap(), 99);
    }

    #[test]
    fn test_storage_clone_boxed() {
        let mut storage = FrameStorage::with_capacity(20);
        storage.set(0, Value::int(1).unwrap());
        storage.set(10, Value::int(2).unwrap());

        let cloned = storage.clone();
        assert!(cloned.is_boxed());
        assert_eq!(cloned.get(0).as_int().unwrap(), 1);
        assert_eq!(cloned.get(10).as_int().unwrap(), 2);
    }

    #[test]
    fn test_storage_debug() {
        let storage = FrameStorage::new();
        let debug = format!("{:?}", storage);
        assert!(debug.contains("FrameStorage"));
        assert!(debug.contains("len"));
    }

    #[test]
    fn test_storage_default() {
        let storage = FrameStorage::default();
        assert!(storage.is_empty());
        assert!(!storage.is_boxed());
    }

    // ════════════════════════════════════════════════════════════════════════
    // Edge Cases
    // ════════════════════════════════════════════════════════════════════════

    #[test]
    fn test_empty_liveness_capture() {
        let registers = [Value::none(); 256];
        let liveness = LivenessMap::EMPTY;
        let mut storage = FrameStorage::new();
        storage.capture(&registers, liveness);

        assert_eq!(storage.len(), 0);
    }

    #[test]
    fn test_single_register_liveness() {
        let mut registers = [Value::none(); 256];
        registers[42] = Value::int(12345).unwrap();

        let liveness = LivenessMap::EMPTY.with_live(42);
        let mut storage = FrameStorage::new();
        storage.capture(&registers, liveness);

        assert_eq!(storage.len(), 1);
        assert_eq!(storage.get(0).as_int().unwrap(), 12345);

        let mut restored = [Value::none(); 256];
        storage.restore(&mut restored, liveness);
        assert_eq!(restored[42].as_int().unwrap(), 12345);
    }

    #[test]
    fn test_max_inline_capacity() {
        let mut registers = [Value::none(); 256];
        for i in 0..8 {
            registers[i] = Value::int(i as i64 * 10).unwrap();
        }

        let liveness = LivenessMap::from_bits(0xFF); // Exactly 8 registers
        let mut storage = FrameStorage::new();
        storage.capture(&registers, liveness);

        assert!(!storage.is_boxed());
        assert_eq!(storage.len(), 8);
    }

    #[test]
    fn test_storage_overwrite() {
        let mut storage = FrameStorage::new();
        storage.set(0, Value::int(1).unwrap());
        storage.set(0, Value::int(2).unwrap());

        assert_eq!(storage.get(0).as_int().unwrap(), 2);
    }
}
