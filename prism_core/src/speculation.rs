//! Speculation hints for JIT compilation.
//!
//! This module defines type speculation hints that can be shared between
//! the VM (which collects type feedback) and the JIT (which uses it for
//! type-specialized code generation).
//!
//! # Architecture
//!
//! ```text
//! ┌──────────────┐     ┌──────────────────┐     ┌──────────────────┐
//! │  Interpreter  │────▶│ ProfileCollector  │────▶│ SpeculationProvider │
//! │  (feedback)   │     │ (atomic counters) │     │ (query interface)  │
//! └──────────────┘     └──────────────────┘     └──────────────────┘
//!                                                       │
//!                              ┌────────────────────────┤
//!                              ▼                        ▼
//!                        ┌──────────┐          ┌──────────────┐
//!                        │TypeHints │          │BranchHints   │
//!                        └──────────┘          │CallTargets   │
//!                                              └──────────────┘
//! ```

// =============================================================================
// Type Hint
// =============================================================================

/// Type speculation hint for binary operations.
///
/// These hints are derived from runtime type feedback and inform the JIT
/// about observed operand types to enable speculative compilation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[repr(u8)]
pub enum TypeHint {
    /// No type information available (cold or polymorphic site).
    #[default]
    None = 0,
    /// Both operands are integers.
    IntInt = 1,
    /// Both operands are floats.
    FloatFloat = 2,
    /// Left operand is int, right is float.
    IntFloat = 3,
    /// Left operand is float, right is int.
    FloatInt = 4,
    /// Both operands are strings.
    StrStr = 5,
    /// String and int (for repetition).
    StrInt = 6,
    /// Int and string (for repetition).
    IntStr = 7,
    /// Both operands are lists (concatenation).
    ListList = 8,
}

impl TypeHint {
    /// Check if this hint suggests integer operations.
    #[inline]
    pub const fn is_int(self) -> bool {
        matches!(self, Self::IntInt)
    }

    /// Check if this hint suggests float operations.
    #[inline]
    pub const fn is_float(self) -> bool {
        matches!(self, Self::FloatFloat | Self::IntFloat | Self::FloatInt)
    }

    /// Check if this hint suggests string operations.
    #[inline]
    pub const fn is_string(self) -> bool {
        matches!(self, Self::StrStr | Self::StrInt | Self::IntStr)
    }

    /// Check if this hint is valid (not None).
    #[inline]
    pub const fn is_valid(self) -> bool {
        !matches!(self, Self::None)
    }
}

// =============================================================================
// PGO Branch Hint
// =============================================================================

/// A branch prediction hint derived from profile-guided data.
///
/// Associates a bytecode offset with a measured branch-taken probability,
/// expressed as a fixed-point fraction (numerator over 2^32). This avoids
/// floating-point arithmetic on the critical path.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PgoBranchHint {
    /// Bytecode offset of the branch instruction.
    pub offset: u32,
    /// Probability that the branch is taken, as a numerator over 2^32.
    ///
    /// - `0` = never taken
    /// - `u32::MAX` = always taken
    /// - `u32::MAX / 2` ≈ 50/50
    pub taken_numer: u32,
}

impl PgoBranchHint {
    /// Denominator (2^32 represented as u64 for ratio calculations).
    const DENOM: u64 = 1u64 << 32;

    /// Create a new branch hint.
    #[inline]
    pub const fn new(offset: u32, taken_numer: u32) -> Self {
        Self {
            offset,
            taken_numer,
        }
    }

    /// Create from measured branch counts.
    ///
    /// Computes `taken / (taken + not_taken)` as a fixed-point fraction.
    #[inline]
    pub fn from_counts(offset: u32, taken: u64, not_taken: u64) -> Self {
        let total = taken.saturating_add(not_taken);
        let numer = if total == 0 {
            u32::MAX / 2 // Default to 50/50 for no data
        } else {
            ((taken as u128 * Self::DENOM as u128) / total as u128).min(u32::MAX as u128) as u32
        };
        Self {
            offset,
            taken_numer: numer,
        }
    }

    /// Create from a floating-point probability [0.0, 1.0].
    #[inline]
    pub fn from_f64(offset: u32, prob: f64) -> Self {
        let clamped = prob.clamp(0.0, 1.0);
        let numer = (clamped * Self::DENOM as f64) as u32;
        Self {
            offset,
            taken_numer: numer,
        }
    }

    /// Get the taken probability as a float.
    #[inline]
    pub fn taken_probability(&self) -> f64 {
        self.taken_numer as f64 / Self::DENOM as f64
    }

    /// Get the not-taken probability as a float.
    #[inline]
    pub fn not_taken_probability(&self) -> f64 {
        1.0 - self.taken_probability()
    }

    /// Whether the branch is biased (significantly away from 50/50).
    ///
    /// A branch is considered biased if its probability deviates more
    /// than 20 percentage points from 50%.
    #[inline]
    pub fn is_biased(&self) -> bool {
        let mid = u32::MAX / 2;
        let threshold = u32::MAX / 5; // ~20%
        self.taken_numer > mid.saturating_add(threshold)
            || self.taken_numer < mid.saturating_sub(threshold)
    }

    /// Whether the branch is likely taken (probability >= 80%).
    #[inline]
    pub fn is_likely_taken(&self) -> bool {
        self.taken_numer >= (u32::MAX as u64 * 4 / 5) as u32
    }

    /// Whether the branch is unlikely taken (probability <= 20%).
    #[inline]
    pub fn is_unlikely_taken(&self) -> bool {
        self.taken_numer <= (u32::MAX as u64 / 5) as u32
    }

    /// Complement: returns a hint with the opposite probability.
    #[inline]
    pub fn complement(&self) -> Self {
        Self {
            offset: self.offset,
            taken_numer: u32::MAX.wrapping_sub(self.taken_numer),
        }
    }
}

// =============================================================================
// PGO Call Target
// =============================================================================

/// A call target observed via profile-guided data.
///
/// Records which function was called at a particular bytecode offset
/// and how frequently. Used for PGO-guided inlining decisions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PgoCallTarget {
    /// Bytecode offset of the call instruction.
    pub offset: u32,
    /// Unique identifier for the called function's code unit.
    pub target_id: u32,
    /// Number of times this target was observed at this call site.
    pub frequency: u32,
}

impl PgoCallTarget {
    /// Create a new call target.
    #[inline]
    pub const fn new(offset: u32, target_id: u32, frequency: u32) -> Self {
        Self {
            offset,
            target_id,
            frequency,
        }
    }

    /// Whether this is a monomorphic call site (single target observed).
    ///
    /// This is a hint — the caller should check that there are no other
    /// targets at the same offset.
    #[inline]
    pub fn is_frequent(&self, threshold: u32) -> bool {
        self.frequency >= threshold
    }
}

// =============================================================================
// Call Site Profile
// =============================================================================

/// Aggregated call target profile for a single call site.
///
/// Summarizes all observed targets at a bytecode offset, including
/// polymorphism classification used for inlining heuristics.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CallSiteProfile {
    /// Bytecode offset of the call instruction.
    pub offset: u32,
    /// All observed call targets, sorted by frequency (descending).
    pub targets: Vec<PgoCallTarget>,
    /// Total call count at this site.
    pub total_calls: u64,
}

impl CallSiteProfile {
    /// Create a new call site profile.
    pub fn new(offset: u32) -> Self {
        Self {
            offset,
            targets: Vec::new(),
            total_calls: 0,
        }
    }

    /// Add a target observation.
    pub fn add_target(&mut self, target_id: u32, frequency: u32) {
        self.targets
            .push(PgoCallTarget::new(self.offset, target_id, frequency));
        self.total_calls += frequency as u64;
        // Maintain descending frequency order
        self.targets
            .sort_unstable_by(|a, b| b.frequency.cmp(&a.frequency));
    }

    /// The morphism classification of this call site.
    #[inline]
    pub fn morphism(&self) -> CallMorphism {
        match self.targets.len() {
            0 => CallMorphism::Cold,
            1 => CallMorphism::Monomorphic,
            2..=4 => CallMorphism::Polymorphic,
            _ => CallMorphism::Megamorphic,
        }
    }

    /// The most frequently called target, if any.
    #[inline]
    pub fn dominant_target(&self) -> Option<&PgoCallTarget> {
        self.targets.first()
    }

    /// Whether a single target dominates (>= threshold% of calls).
    pub fn has_dominant_target(&self, threshold_pct: u32) -> bool {
        if let Some(target) = self.dominant_target() {
            if self.total_calls == 0 {
                return false;
            }
            let pct = (target.frequency as u64 * 100) / self.total_calls;
            pct >= threshold_pct as u64
        } else {
            false
        }
    }

    /// Number of distinct targets observed.
    #[inline]
    pub fn target_count(&self) -> usize {
        self.targets.len()
    }
}

/// Call site morphism classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CallMorphism {
    /// No calls observed at this site.
    Cold,
    /// Single target — ideal for inlining.
    Monomorphic,
    /// Few targets (2-4) — may benefit from inline cache / dispatch table.
    Polymorphic,
    /// Many targets (5+) — unlikely to benefit from specialization.
    Megamorphic,
}

// =============================================================================
// Speculation Provider Trait
// =============================================================================

/// Provider of type speculation hints for JIT compilation.
///
/// This trait allows the JIT to query speculation information without
/// direct dependency on VM internals.
pub trait SpeculationProvider {
    /// Get the type hint for a specific bytecode site.
    ///
    /// # Arguments
    /// * `code_id` - Unique identifier for the compiled code unit
    /// * `bc_offset` - Bytecode offset within the code unit
    ///
    /// # Returns
    /// The observed type hint, or `TypeHint::None` if no information is available.
    fn get_type_hint(&self, code_id: u32, bc_offset: u32) -> TypeHint;

    /// Get the PGO branch hint for a specific branch site.
    ///
    /// Returns `None` if no profile data is available for this branch.
    fn get_branch_hint(&self, code_id: u32, bc_offset: u32) -> Option<PgoBranchHint> {
        let _ = (code_id, bc_offset);
        None
    }

    /// Get all branch hints for a code unit.
    ///
    /// Returns an empty vector if no profile data is available.
    fn get_all_branch_hints(&self, code_id: u32) -> Vec<PgoBranchHint> {
        let _ = code_id;
        Vec::new()
    }

    /// Get the call site profile for a specific call site.
    ///
    /// Returns `None` if no profile data is available for this call site.
    fn get_call_targets(&self, code_id: u32, bc_offset: u32) -> Option<CallSiteProfile> {
        let _ = (code_id, bc_offset);
        None
    }

    /// Whether profile-guided data is available for this code unit.
    fn has_profile_data(&self, code_id: u32) -> bool {
        let _ = code_id;
        false
    }

    /// Get the execution count for a code unit (0 if unknown).
    fn execution_count(&self, code_id: u32) -> u64 {
        let _ = code_id;
        0
    }
}

// =============================================================================
// No-Op Provider
// =============================================================================

/// No-op speculation provider that always returns None.
///
/// Useful for testing or when speculation data is not available.
#[derive(Debug, Default)]
pub struct NoSpeculation;

impl SpeculationProvider for NoSpeculation {
    #[inline]
    fn get_type_hint(&self, _code_id: u32, _bc_offset: u32) -> TypeHint {
        TypeHint::None
    }
}

// =============================================================================
// Static Speculation Provider
// =============================================================================

/// A simple speculation provider backed by pre-populated static data.
///
/// Useful for testing and benchmarking without a live interpreter.
#[derive(Debug, Default)]
pub struct StaticSpeculation {
    /// Type hints: (code_id, bc_offset) → TypeHint
    type_hints: Vec<(u32, u32, TypeHint)>,
    /// Branch hints: (code_id, bc_offset) → PgoBranchHint
    branch_hints: Vec<(u32, PgoBranchHint)>,
    /// Call targets: (code_id, bc_offset) → CallSiteProfile
    call_profiles: Vec<(u32, CallSiteProfile)>,
    /// Code units with profile data
    profiled_units: Vec<u32>,
    /// Execution counts: code_id → count
    exec_counts: Vec<(u32, u64)>,
}

impl StaticSpeculation {
    /// Create a new empty static speculation provider.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a type hint.
    pub fn add_type_hint(&mut self, code_id: u32, bc_offset: u32, hint: TypeHint) {
        self.type_hints.push((code_id, bc_offset, hint));
    }

    /// Add a branch hint.
    pub fn add_branch_hint(&mut self, code_id: u32, hint: PgoBranchHint) {
        if !self.profiled_units.contains(&code_id) {
            self.profiled_units.push(code_id);
        }
        self.branch_hints.push((code_id, hint));
    }

    /// Add a call site profile.
    pub fn add_call_profile(&mut self, code_id: u32, profile: CallSiteProfile) {
        if !self.profiled_units.contains(&code_id) {
            self.profiled_units.push(code_id);
        }
        self.call_profiles.push((code_id, profile));
    }

    /// Set the execution count for a code unit.
    pub fn set_execution_count(&mut self, code_id: u32, count: u64) {
        if !self.profiled_units.contains(&code_id) {
            self.profiled_units.push(code_id);
        }
        self.exec_counts.push((code_id, count));
    }
}

impl SpeculationProvider for StaticSpeculation {
    fn get_type_hint(&self, code_id: u32, bc_offset: u32) -> TypeHint {
        self.type_hints
            .iter()
            .find(|(cid, off, _)| *cid == code_id && *off == bc_offset)
            .map(|(_, _, hint)| *hint)
            .unwrap_or(TypeHint::None)
    }

    fn get_branch_hint(&self, code_id: u32, bc_offset: u32) -> Option<PgoBranchHint> {
        self.branch_hints
            .iter()
            .find(|(cid, hint)| *cid == code_id && hint.offset == bc_offset)
            .map(|(_, hint)| *hint)
    }

    fn get_all_branch_hints(&self, code_id: u32) -> Vec<PgoBranchHint> {
        self.branch_hints
            .iter()
            .filter(|(cid, _)| *cid == code_id)
            .map(|(_, hint)| *hint)
            .collect()
    }

    fn get_call_targets(&self, code_id: u32, bc_offset: u32) -> Option<CallSiteProfile> {
        self.call_profiles
            .iter()
            .find(|(cid, profile)| *cid == code_id && profile.offset == bc_offset)
            .map(|(_, profile)| profile.clone())
    }

    fn has_profile_data(&self, code_id: u32) -> bool {
        self.profiled_units.contains(&code_id)
    }

    fn execution_count(&self, code_id: u32) -> u64 {
        self.exec_counts
            .iter()
            .find(|(cid, _)| *cid == code_id)
            .map(|(_, count)| *count)
            .unwrap_or(0)
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // TypeHint Tests
    // =========================================================================

    #[test]
    fn test_type_hint_is_int() {
        assert!(TypeHint::IntInt.is_int());
        assert!(!TypeHint::FloatFloat.is_int());
        assert!(!TypeHint::StrStr.is_int());
        assert!(!TypeHint::None.is_int());
    }

    #[test]
    fn test_type_hint_is_float() {
        assert!(TypeHint::FloatFloat.is_float());
        assert!(TypeHint::IntFloat.is_float());
        assert!(TypeHint::FloatInt.is_float());
        assert!(!TypeHint::IntInt.is_float());
        assert!(!TypeHint::StrStr.is_float());
    }

    #[test]
    fn test_type_hint_is_string() {
        assert!(TypeHint::StrStr.is_string());
        assert!(TypeHint::StrInt.is_string());
        assert!(TypeHint::IntStr.is_string());
        assert!(!TypeHint::IntInt.is_string());
        assert!(!TypeHint::FloatFloat.is_string());
    }

    #[test]
    fn test_type_hint_is_valid() {
        assert!(!TypeHint::None.is_valid());
        assert!(TypeHint::IntInt.is_valid());
        assert!(TypeHint::FloatFloat.is_valid());
        assert!(TypeHint::StrStr.is_valid());
    }

    #[test]
    fn test_type_hint_default() {
        let hint: TypeHint = Default::default();
        assert_eq!(hint, TypeHint::None);
    }

    #[test]
    fn test_type_hint_repr() {
        assert_eq!(TypeHint::None as u8, 0);
        assert_eq!(TypeHint::IntInt as u8, 1);
        assert_eq!(TypeHint::ListList as u8, 8);
    }

    // =========================================================================
    // PgoBranchHint Tests
    // =========================================================================

    #[test]
    fn test_branch_hint_new() {
        let hint = PgoBranchHint::new(42, u32::MAX / 2);
        assert_eq!(hint.offset, 42);
        assert_eq!(hint.taken_numer, u32::MAX / 2);
    }

    #[test]
    fn test_branch_hint_from_counts_balanced() {
        let hint = PgoBranchHint::from_counts(10, 100, 100);
        let prob = hint.taken_probability();
        assert!((prob - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_branch_hint_from_counts_always_taken() {
        let hint = PgoBranchHint::from_counts(0, 1000, 0);
        assert!(hint.taken_probability() > 0.99);
    }

    #[test]
    fn test_branch_hint_from_counts_never_taken() {
        let hint = PgoBranchHint::from_counts(0, 0, 1000);
        assert!(hint.taken_probability() < 0.01);
    }

    #[test]
    fn test_branch_hint_from_counts_zero_total() {
        let hint = PgoBranchHint::from_counts(5, 0, 0);
        // Should default to ~50/50
        let prob = hint.taken_probability();
        assert!((prob - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_branch_hint_from_f64() {
        let hint = PgoBranchHint::from_f64(0, 0.75);
        let prob = hint.taken_probability();
        assert!((prob - 0.75).abs() < 0.01);
    }

    #[test]
    fn test_branch_hint_from_f64_clamped() {
        let hint_over = PgoBranchHint::from_f64(0, 1.5);
        assert!(hint_over.taken_probability() <= 1.0);

        let hint_under = PgoBranchHint::from_f64(0, -0.5);
        assert!(hint_under.taken_probability() >= 0.0);
    }

    #[test]
    fn test_branch_hint_not_taken_probability() {
        let hint = PgoBranchHint::from_f64(0, 0.3);
        let not_taken = hint.not_taken_probability();
        assert!((not_taken - 0.7).abs() < 0.01);
    }

    #[test]
    fn test_branch_hint_is_biased() {
        // Heavily biased
        let hint_biased = PgoBranchHint::from_f64(0, 0.95);
        assert!(hint_biased.is_biased());

        // Not biased
        let hint_even = PgoBranchHint::from_f64(0, 0.5);
        assert!(!hint_even.is_biased());

        // Slightly biased but within threshold
        let hint_slight = PgoBranchHint::from_f64(0, 0.6);
        assert!(!hint_slight.is_biased());
    }

    #[test]
    fn test_branch_hint_is_likely_taken() {
        let likely = PgoBranchHint::from_f64(0, 0.9);
        assert!(likely.is_likely_taken());

        let unlikely = PgoBranchHint::from_f64(0, 0.3);
        assert!(!unlikely.is_likely_taken());
    }

    #[test]
    fn test_branch_hint_is_unlikely_taken() {
        let unlikely = PgoBranchHint::from_f64(0, 0.1);
        assert!(unlikely.is_unlikely_taken());

        let likely = PgoBranchHint::from_f64(0, 0.7);
        assert!(!likely.is_unlikely_taken());
    }

    #[test]
    fn test_branch_hint_complement() {
        let hint = PgoBranchHint::from_f64(42, 0.8);
        let comp = hint.complement();
        assert_eq!(comp.offset, 42);
        let comp_prob = comp.taken_probability();
        assert!((comp_prob - 0.2).abs() < 0.01);
    }

    #[test]
    fn test_branch_hint_complement_roundtrip() {
        let hint = PgoBranchHint::new(0, 1000000);
        let double_comp = hint.complement().complement();
        // Not exact roundtrip due to wrapping arithmetic, but close
        assert!((hint.taken_numer as i64 - double_comp.taken_numer as i64).unsigned_abs() <= 1,);
    }

    #[test]
    fn test_branch_hint_from_counts_large_numbers() {
        let hint = PgoBranchHint::from_counts(0, u64::MAX / 2, u64::MAX / 2);
        let prob = hint.taken_probability();
        assert!((prob - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_branch_hint_from_counts_saturating_add() {
        // Very large counts that would overflow without saturating add
        let hint = PgoBranchHint::from_counts(0, u64::MAX, 1);
        // Should not panic, probability should be near 1.0
        assert!(hint.taken_probability() > 0.99);
    }

    #[test]
    fn test_branch_hint_extreme_probabilities() {
        let always = PgoBranchHint::new(0, u32::MAX);
        assert!(always.taken_probability() > 0.99);
        assert!(always.is_likely_taken());
        assert!(!always.is_unlikely_taken());

        let never = PgoBranchHint::new(0, 0);
        assert!(never.taken_probability() < 0.01);
        assert!(!never.is_likely_taken());
        assert!(never.is_unlikely_taken());
    }

    // =========================================================================
    // PgoCallTarget Tests
    // =========================================================================

    #[test]
    fn test_call_target_new() {
        let target = PgoCallTarget::new(10, 42, 100);
        assert_eq!(target.offset, 10);
        assert_eq!(target.target_id, 42);
        assert_eq!(target.frequency, 100);
    }

    #[test]
    fn test_call_target_is_frequent() {
        let target = PgoCallTarget::new(0, 1, 50);
        assert!(target.is_frequent(50));
        assert!(target.is_frequent(10));
        assert!(!target.is_frequent(51));
    }

    #[test]
    fn test_call_target_zero_frequency() {
        let target = PgoCallTarget::new(0, 1, 0);
        assert!(!target.is_frequent(1));
    }

    // =========================================================================
    // CallSiteProfile Tests
    // =========================================================================

    #[test]
    fn test_call_site_profile_new() {
        let profile = CallSiteProfile::new(10);
        assert_eq!(profile.offset, 10);
        assert_eq!(profile.target_count(), 0);
        assert_eq!(profile.total_calls, 0);
    }

    #[test]
    fn test_call_site_profile_add_target() {
        let mut profile = CallSiteProfile::new(10);
        profile.add_target(1, 100);
        assert_eq!(profile.target_count(), 1);
        assert_eq!(profile.total_calls, 100);
    }

    #[test]
    fn test_call_site_profile_sorted_by_frequency() {
        let mut profile = CallSiteProfile::new(10);
        profile.add_target(1, 10);
        profile.add_target(2, 100);
        profile.add_target(3, 50);

        assert_eq!(profile.targets[0].target_id, 2); // Most frequent
        assert_eq!(profile.targets[1].target_id, 3);
        assert_eq!(profile.targets[2].target_id, 1); // Least frequent
    }

    #[test]
    fn test_call_site_profile_morphism_cold() {
        let profile = CallSiteProfile::new(10);
        assert_eq!(profile.morphism(), CallMorphism::Cold);
    }

    #[test]
    fn test_call_site_profile_morphism_monomorphic() {
        let mut profile = CallSiteProfile::new(10);
        profile.add_target(1, 100);
        assert_eq!(profile.morphism(), CallMorphism::Monomorphic);
    }

    #[test]
    fn test_call_site_profile_morphism_polymorphic() {
        let mut profile = CallSiteProfile::new(10);
        profile.add_target(1, 100);
        profile.add_target(2, 50);
        assert_eq!(profile.morphism(), CallMorphism::Polymorphic);

        profile.add_target(3, 30);
        profile.add_target(4, 10);
        assert_eq!(profile.morphism(), CallMorphism::Polymorphic);
    }

    #[test]
    fn test_call_site_profile_morphism_megamorphic() {
        let mut profile = CallSiteProfile::new(10);
        for i in 0..5 {
            profile.add_target(i, 10);
        }
        assert_eq!(profile.morphism(), CallMorphism::Megamorphic);
    }

    #[test]
    fn test_call_site_profile_dominant_target() {
        let mut profile = CallSiteProfile::new(10);
        profile.add_target(1, 100);
        profile.add_target(2, 10);
        profile.add_target(3, 5);

        let dom = profile.dominant_target().unwrap();
        assert_eq!(dom.target_id, 1);
        assert_eq!(dom.frequency, 100);
    }

    #[test]
    fn test_call_site_profile_dominant_target_none() {
        let profile = CallSiteProfile::new(10);
        assert!(profile.dominant_target().is_none());
    }

    #[test]
    fn test_call_site_profile_has_dominant_target() {
        let mut profile = CallSiteProfile::new(10);
        profile.add_target(1, 90);
        profile.add_target(2, 10);

        assert!(profile.has_dominant_target(85)); // 90%
        assert!(profile.has_dominant_target(90));
        assert!(!profile.has_dominant_target(91));
    }

    #[test]
    fn test_call_site_profile_no_dominant_when_empty() {
        let profile = CallSiteProfile::new(10);
        assert!(!profile.has_dominant_target(50));
    }

    #[test]
    fn test_call_site_profile_total_calls_accumulates() {
        let mut profile = CallSiteProfile::new(10);
        profile.add_target(1, 100);
        profile.add_target(2, 200);
        profile.add_target(3, 300);
        assert_eq!(profile.total_calls, 600);
    }

    // =========================================================================
    // CallMorphism Tests
    // =========================================================================

    #[test]
    fn test_call_morphism_variants() {
        // Just ensure all variants exist and are distinct
        assert_ne!(CallMorphism::Cold, CallMorphism::Monomorphic);
        assert_ne!(CallMorphism::Polymorphic, CallMorphism::Megamorphic);
    }

    // =========================================================================
    // SpeculationProvider Tests
    // =========================================================================

    #[test]
    fn test_no_speculation_provider() {
        let provider = NoSpeculation;
        assert_eq!(provider.get_type_hint(1, 0), TypeHint::None);
        assert_eq!(provider.get_type_hint(100, 50), TypeHint::None);
        assert!(provider.get_branch_hint(1, 0).is_none());
        assert!(provider.get_all_branch_hints(1).is_empty());
        assert!(provider.get_call_targets(1, 0).is_none());
        assert!(!provider.has_profile_data(1));
        assert_eq!(provider.execution_count(1), 0);
    }

    #[test]
    fn test_no_speculation_default_trait_methods() {
        let provider = NoSpeculation;
        // Ensure all default methods return correct defaults
        assert!(provider.get_branch_hint(0, 0).is_none());
        assert!(provider.get_all_branch_hints(0).is_empty());
        assert!(provider.get_call_targets(0, 0).is_none());
        assert!(!provider.has_profile_data(0));
        assert_eq!(provider.execution_count(0), 0);
    }

    // =========================================================================
    // StaticSpeculation Tests
    // =========================================================================

    #[test]
    fn test_static_speculation_new() {
        let spec = StaticSpeculation::new();
        assert_eq!(spec.get_type_hint(1, 0), TypeHint::None);
        assert!(!spec.has_profile_data(1));
    }

    #[test]
    fn test_static_speculation_type_hints() {
        let mut spec = StaticSpeculation::new();
        spec.add_type_hint(1, 10, TypeHint::IntInt);
        spec.add_type_hint(1, 20, TypeHint::FloatFloat);
        spec.add_type_hint(2, 10, TypeHint::StrStr);

        assert_eq!(spec.get_type_hint(1, 10), TypeHint::IntInt);
        assert_eq!(spec.get_type_hint(1, 20), TypeHint::FloatFloat);
        assert_eq!(spec.get_type_hint(2, 10), TypeHint::StrStr);
        assert_eq!(spec.get_type_hint(1, 30), TypeHint::None); // Missing
        assert_eq!(spec.get_type_hint(3, 10), TypeHint::None); // Missing code_id
    }

    #[test]
    fn test_static_speculation_branch_hints() {
        let mut spec = StaticSpeculation::new();
        let hint1 = PgoBranchHint::from_f64(10, 0.9);
        let hint2 = PgoBranchHint::from_f64(20, 0.1);
        spec.add_branch_hint(1, hint1);
        spec.add_branch_hint(1, hint2);

        let retrieved = spec.get_branch_hint(1, 10).unwrap();
        assert!((retrieved.taken_probability() - 0.9).abs() < 0.01);

        let retrieved2 = spec.get_branch_hint(1, 20).unwrap();
        assert!((retrieved2.taken_probability() - 0.1).abs() < 0.01);

        assert!(spec.get_branch_hint(1, 30).is_none());
        assert!(spec.get_branch_hint(2, 10).is_none());
    }

    #[test]
    fn test_static_speculation_all_branch_hints() {
        let mut spec = StaticSpeculation::new();
        spec.add_branch_hint(1, PgoBranchHint::from_f64(10, 0.9));
        spec.add_branch_hint(1, PgoBranchHint::from_f64(20, 0.1));
        spec.add_branch_hint(2, PgoBranchHint::from_f64(10, 0.5));

        let hints = spec.get_all_branch_hints(1);
        assert_eq!(hints.len(), 2);

        let hints2 = spec.get_all_branch_hints(2);
        assert_eq!(hints2.len(), 1);

        let hints3 = spec.get_all_branch_hints(3);
        assert!(hints3.is_empty());
    }

    #[test]
    fn test_static_speculation_call_targets() {
        let mut spec = StaticSpeculation::new();
        let mut profile = CallSiteProfile::new(10);
        profile.add_target(42, 100);
        profile.add_target(43, 50);
        spec.add_call_profile(1, profile);

        let retrieved = spec.get_call_targets(1, 10).unwrap();
        assert_eq!(retrieved.target_count(), 2);
        assert_eq!(retrieved.dominant_target().unwrap().target_id, 42);

        assert!(spec.get_call_targets(1, 20).is_none());
        assert!(spec.get_call_targets(2, 10).is_none());
    }

    #[test]
    fn test_static_speculation_has_profile_data() {
        let mut spec = StaticSpeculation::new();
        assert!(!spec.has_profile_data(1));

        spec.add_branch_hint(1, PgoBranchHint::from_f64(10, 0.9));
        assert!(spec.has_profile_data(1));
        assert!(!spec.has_profile_data(2));
    }

    #[test]
    fn test_static_speculation_execution_count() {
        let mut spec = StaticSpeculation::new();
        assert_eq!(spec.execution_count(1), 0);

        spec.set_execution_count(1, 5000);
        assert_eq!(spec.execution_count(1), 5000);
        assert_eq!(spec.execution_count(2), 0);
    }

    #[test]
    fn test_static_speculation_mixed_data() {
        let mut spec = StaticSpeculation::new();

        // Add all types of data for code unit 1
        spec.add_type_hint(1, 10, TypeHint::IntInt);
        spec.add_branch_hint(1, PgoBranchHint::from_f64(20, 0.9));

        let mut call_profile = CallSiteProfile::new(30);
        call_profile.add_target(99, 500);
        spec.add_call_profile(1, call_profile);

        spec.set_execution_count(1, 10000);

        // Verify all data
        assert_eq!(spec.get_type_hint(1, 10), TypeHint::IntInt);
        assert!(spec.get_branch_hint(1, 20).is_some());
        assert!(spec.get_call_targets(1, 30).is_some());
        assert_eq!(spec.execution_count(1), 10000);
        assert!(spec.has_profile_data(1));
    }

    #[test]
    fn test_static_speculation_multiple_code_units() {
        let mut spec = StaticSpeculation::new();

        spec.add_branch_hint(1, PgoBranchHint::from_f64(10, 0.9));
        spec.add_branch_hint(2, PgoBranchHint::from_f64(10, 0.1));

        let h1 = spec.get_branch_hint(1, 10).unwrap();
        let h2 = spec.get_branch_hint(2, 10).unwrap();

        assert!(h1.is_likely_taken());
        assert!(h2.is_unlikely_taken());
    }

    #[test]
    fn test_static_speculation_marks_profiled() {
        let mut spec = StaticSpeculation::new();

        // Adding call profile should mark as profiled
        spec.add_call_profile(5, CallSiteProfile::new(0));
        assert!(spec.has_profile_data(5));

        // Setting exec count should also mark as profiled
        spec.set_execution_count(7, 100);
        assert!(spec.has_profile_data(7));
    }

    #[test]
    fn test_branch_hint_equality() {
        let a = PgoBranchHint::new(10, 500);
        let b = PgoBranchHint::new(10, 500);
        let c = PgoBranchHint::new(10, 501);
        assert_eq!(a, b);
        assert_ne!(a, c);
    }

    #[test]
    fn test_call_target_equality() {
        let a = PgoCallTarget::new(10, 42, 100);
        let b = PgoCallTarget::new(10, 42, 100);
        let c = PgoCallTarget::new(10, 43, 100);
        assert_eq!(a, b);
        assert_ne!(a, c);
    }

    #[test]
    fn test_branch_hint_hash() {
        use std::collections::HashSet;
        let mut set = HashSet::new();
        set.insert(PgoBranchHint::new(10, 500));
        set.insert(PgoBranchHint::new(10, 500)); // Duplicate
        set.insert(PgoBranchHint::new(20, 500));
        assert_eq!(set.len(), 2);
    }

    #[test]
    fn test_call_target_hash() {
        use std::collections::HashSet;
        let mut set = HashSet::new();
        set.insert(PgoCallTarget::new(10, 42, 100));
        set.insert(PgoCallTarget::new(10, 42, 100)); // Duplicate
        set.insert(PgoCallTarget::new(20, 42, 100));
        assert_eq!(set.len(), 2);
    }

    #[test]
    fn test_type_hint_all_variants_classified() {
        // Every variant should be classifiable
        let variants = [
            TypeHint::None,
            TypeHint::IntInt,
            TypeHint::FloatFloat,
            TypeHint::IntFloat,
            TypeHint::FloatInt,
            TypeHint::StrStr,
            TypeHint::StrInt,
            TypeHint::IntStr,
            TypeHint::ListList,
        ];

        for hint in &variants {
            // Each variant should either be valid or None
            if *hint == TypeHint::None {
                assert!(!hint.is_valid());
            } else {
                assert!(hint.is_valid());
            }
        }
    }

    #[test]
    fn test_branch_hint_probability_bounds() {
        // All probabilities should be in [0.0, 1.0]
        for numer in [0, 1, u32::MAX / 4, u32::MAX / 2, u32::MAX - 1, u32::MAX] {
            let hint = PgoBranchHint::new(0, numer);
            let prob = hint.taken_probability();
            assert!(prob >= 0.0, "probability {} should be >= 0", prob);
            assert!(prob <= 1.0, "probability {} should be <= 1", prob);
        }
    }

    #[test]
    fn test_branch_hint_from_counts_precision() {
        // 90% taken
        let hint = PgoBranchHint::from_counts(0, 900, 100);
        let prob = hint.taken_probability();
        assert!((prob - 0.9).abs() < 0.001, "Expected ~0.9, got {}", prob);

        // 10% taken
        let hint = PgoBranchHint::from_counts(0, 10, 90);
        let prob = hint.taken_probability();
        assert!((prob - 0.1).abs() < 0.001, "Expected ~0.1, got {}", prob);
    }

    #[test]
    fn test_call_site_profile_single_target_fully_dominant() {
        let mut profile = CallSiteProfile::new(0);
        profile.add_target(1, 1000);
        assert!(profile.has_dominant_target(100)); // 100%
    }

    #[test]
    fn test_call_site_profile_equal_targets_not_dominant() {
        let mut profile = CallSiteProfile::new(0);
        profile.add_target(1, 50);
        profile.add_target(2, 50);
        assert!(!profile.has_dominant_target(51)); // Each is 50%
        assert!(profile.has_dominant_target(50)); // 50% exactly
    }
}
