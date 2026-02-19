//! Execution profiler for tier-up decisions.
//!
//! Collects runtime statistics to determine when functions should
//! be promoted to higher JIT tiers.

use rustc_hash::FxHashMap;

/// Threshold for Template JIT (Tier 1) compilation.
pub const TIER1_THRESHOLD: u64 = 1_000;

/// Threshold for Optimizing JIT (Tier 2) compilation.
pub const TIER2_THRESHOLD: u64 = 10_000;

/// Unique identifier for a code object.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CodeId(pub u64);

impl CodeId {
    /// Create from a raw u64 value.
    #[inline]
    pub const fn new(id: u64) -> Self {
        CodeId(id)
    }

    /// Create from raw pointer (for fast creation from Arc).
    #[inline]
    pub fn from_ptr(ptr: *const ()) -> Self {
        CodeId(ptr as u64)
    }

    /// Get the underlying u64 value.
    #[inline(always)]
    pub const fn as_u64(self) -> u64 {
        self.0
    }
}

/// Type feedback for a value (observed types at runtime).
#[derive(Debug, Clone, Copy, Default)]
pub struct TypeFeedback {
    /// Bitmap of observed types.
    pub observed: TypeBitmap,
    /// Sample count.
    pub samples: u32,
}

/// Bitmap for type observations (up to 16 types).
#[derive(Debug, Clone, Copy, Default)]
pub struct TypeBitmap(pub u16);

impl TypeBitmap {
    pub const NONE: u16 = 1 << 0;
    pub const BOOL: u16 = 1 << 1;
    pub const INT: u16 = 1 << 2;
    pub const FLOAT: u16 = 1 << 3;
    pub const STRING: u16 = 1 << 4;
    pub const BYTES: u16 = 1 << 5;
    pub const LIST: u16 = 1 << 6;
    pub const TUPLE: u16 = 1 << 7;
    pub const DICT: u16 = 1 << 8;
    pub const SET: u16 = 1 << 9;
    pub const FUNCTION: u16 = 1 << 10;
    pub const OBJECT: u16 = 1 << 11;

    /// Record an observed type.
    #[inline]
    pub fn record(&mut self, type_bit: u16) {
        self.0 |= type_bit;
    }

    /// Check if only a single type was observed (monomorphic).
    #[inline]
    pub fn is_monomorphic(&self) -> bool {
        self.0.count_ones() == 1
    }

    /// Check if type is polymorphic (2-4 types).
    #[inline]
    pub fn is_polymorphic(&self) -> bool {
        let count = self.0.count_ones();
        count >= 2 && count <= 4
    }

    /// Check if megamorphic (too many types to optimize).
    #[inline]
    pub fn is_megamorphic(&self) -> bool {
        self.0.count_ones() > 4
    }

    /// Get the single observed type (if monomorphic).
    #[inline]
    pub fn single_type(&self) -> Option<u16> {
        if self.is_monomorphic() {
            Some(self.0)
        } else {
            None
        }
    }
}

impl TypeFeedback {
    /// Record an observation.
    #[inline]
    pub fn record(&mut self, type_bit: u16) {
        self.observed.record(type_bit);
        self.samples = self.samples.saturating_add(1);
    }
}

/// Execution profiler collecting runtime statistics.
#[derive(Debug, Default)]
pub struct Profiler {
    /// Call counts per function.
    call_counts: FxHashMap<CodeId, u64>,
    /// Loop iteration counts (code_id, loop_offset) → count.
    loop_counts: FxHashMap<(CodeId, u32), u64>,
    /// Type feedback (code_id, instruction_offset) → feedback.
    type_feedback: FxHashMap<(CodeId, u32), TypeFeedback>,
    /// Functions ready for Tier 1.
    tier1_candidates: Vec<CodeId>,
    /// Functions ready for Tier 2.
    tier2_candidates: Vec<CodeId>,
}

impl Profiler {
    /// Create a new profiler.
    pub fn new() -> Self {
        Self::default()
    }

    // =========================================================================
    // Call Counting
    // =========================================================================

    /// Record a function call and return the current call count.
    #[inline]
    pub fn record_call(&mut self, code_id: CodeId) -> u64 {
        let count = self.call_counts.entry(code_id).or_insert(0);
        *count += 1;
        *count
    }

    /// Get call count for a function.
    #[inline]
    pub fn call_count(&self, code_id: CodeId) -> u64 {
        self.call_counts.get(&code_id).copied().unwrap_or(0)
    }

    // =========================================================================
    // Loop Counting
    // =========================================================================

    /// Record a loop back-edge.
    #[inline]
    pub fn record_loop(&mut self, code_id: CodeId, loop_offset: u32) -> bool {
        let count = self.loop_counts.entry((code_id, loop_offset)).or_insert(0);
        *count += 1;

        // Hot loop detection
        *count >= TIER1_THRESHOLD
    }

    /// Get loop iteration count.
    #[inline]
    pub fn loop_count(&self, code_id: CodeId, loop_offset: u32) -> u64 {
        self.loop_counts
            .get(&(code_id, loop_offset))
            .copied()
            .unwrap_or(0)
    }

    // =========================================================================
    // Type Feedback
    // =========================================================================

    /// Record type at an instruction.
    #[inline]
    pub fn record_type(&mut self, code_id: CodeId, offset: u32, type_bit: u16) {
        let feedback = self.type_feedback.entry((code_id, offset)).or_default();
        feedback.record(type_bit);
    }

    /// Get type feedback for an instruction.
    #[inline]
    pub fn get_type_feedback(&self, code_id: CodeId, offset: u32) -> Option<&TypeFeedback> {
        self.type_feedback.get(&(code_id, offset))
    }

    // =========================================================================
    // Tier-Up Management
    // =========================================================================

    /// Get functions ready for Tier 1 compilation.
    pub fn drain_tier1_candidates(&mut self) -> Vec<CodeId> {
        std::mem::take(&mut self.tier1_candidates)
    }

    /// Get functions ready for Tier 2 compilation.
    pub fn drain_tier2_candidates(&mut self) -> Vec<CodeId> {
        std::mem::take(&mut self.tier2_candidates)
    }

    /// Check if a function is hot.
    #[inline]
    pub fn is_hot(&self, code_id: CodeId) -> bool {
        self.call_count(code_id) >= TIER1_THRESHOLD
    }

    /// Check if a function is very hot.
    #[inline]
    pub fn is_very_hot(&self, code_id: CodeId) -> bool {
        self.call_count(code_id) >= TIER2_THRESHOLD
    }

    // =========================================================================
    // Statistics
    // =========================================================================

    /// Get total call count across all functions.
    pub fn total_calls(&self) -> u64 {
        self.call_counts.values().sum()
    }

    /// Get number of tracked functions.
    pub fn function_count(&self) -> usize {
        self.call_counts.len()
    }

    /// Get number of tracked loops.
    pub fn loop_count_total(&self) -> usize {
        self.loop_counts.len()
    }

    /// Clear all profiling data.
    pub fn clear(&mut self) {
        self.call_counts.clear();
        self.loop_counts.clear();
        self.type_feedback.clear();
        self.tier1_candidates.clear();
        self.tier2_candidates.clear();
    }
}

/// Decision from profiler about tier-up.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TierUpDecision {
    /// No tier-up needed.
    None,
    /// Should compile to Tier 1 (Template JIT).
    Tier1,
    /// Should compile to Tier 2 (Optimizing JIT).
    Tier2,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_call_counting() {
        let mut profiler = Profiler::new();
        let code_id = CodeId(12345);

        // Call count should increase monotonically.
        for i in 1..=16 {
            assert_eq!(profiler.record_call(code_id), i);
        }

        assert_eq!(profiler.call_count(code_id), 16);
    }

    #[test]
    fn test_type_bitmap() {
        let mut bitmap = TypeBitmap::default();

        bitmap.record(TypeBitmap::INT);
        assert!(bitmap.is_monomorphic());
        assert_eq!(bitmap.single_type(), Some(TypeBitmap::INT));

        bitmap.record(TypeBitmap::FLOAT);
        assert!(!bitmap.is_monomorphic());
        assert!(bitmap.is_polymorphic());

        // Add more types to make megamorphic
        bitmap.record(TypeBitmap::STRING);
        bitmap.record(TypeBitmap::LIST);
        bitmap.record(TypeBitmap::DICT);
        assert!(bitmap.is_megamorphic());
    }

    #[test]
    fn test_loop_counting() {
        let mut profiler = Profiler::new();
        let code_id = CodeId(0);

        for i in 0..TIER1_THRESHOLD {
            let is_hot = profiler.record_loop(code_id, 10);
            if i < TIER1_THRESHOLD - 1 {
                assert!(!is_hot);
            } else {
                assert!(is_hot);
            }
        }
    }
}
