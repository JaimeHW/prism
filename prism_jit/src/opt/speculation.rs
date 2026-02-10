//! Speculation bridge for PGO-guided JIT compilation.
//!
//! Bridges raw `ProfileData` into type-safe speculation decisions consumed
//! by the optimization pipeline and bytecode translator. Provides the
//! `SpeculationProvider` trait and a concrete `PgoSpeculationProvider`.
//!
//! # Architecture
//!
//! ```text
//! ┌──────────────┐     ┌─────────────────────┐     ┌──────────────┐
//! │ ProfileData  │────▶│ PgoSpeculationProv.  │────▶│ Opt Pipeline │
//! │ (raw counts) │     │ (typed decisions)    │     │ (branch/call │
//! └──────────────┘     └─────────────────────┘     │  annotations)│
//!                                                   └──────────────┘
//! ```

use crate::opt::branch_probability::BranchProbability;
use crate::runtime::profile_data::{CallProfile, ProfileData, TypeProfile};

// =============================================================================
// PGO Branch Hint
// =============================================================================

/// PGO-derived branch probability hint for a single branch site.
///
/// Uses fixed-point representation (numerator over 2^32) for zero-allocation
/// branch probability queries in the compilation hot path.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PgoBranchHint {
    /// Bytecode offset of the branch instruction.
    pub offset: u32,
    /// Probability of the taken path as fixed-point numerator (denom = 2^32).
    pub taken_numer: u32,
}

impl PgoBranchHint {
    /// Create a new branch hint from measured counts.
    #[inline]
    pub fn from_counts(offset: u32, taken: u64, not_taken: u64) -> Self {
        let prob = BranchProbability::from_counts(taken, not_taken);
        Self {
            offset,
            taken_numer: prob.numerator(),
        }
    }

    /// Create from a pre-computed probability.
    #[inline]
    pub fn from_probability(offset: u32, prob: BranchProbability) -> Self {
        Self {
            offset,
            taken_numer: prob.numerator(),
        }
    }

    /// Convert to `BranchProbability`.
    #[inline]
    pub fn as_branch_probability(self) -> BranchProbability {
        BranchProbability::from_ratio(self.taken_numer as u64, u32::MAX as u64)
    }

    /// Whether the branch is likely taken (>= 80%).
    #[inline]
    pub fn is_likely_taken(self) -> bool {
        self.taken_numer >= (u32::MAX as u64 * 4 / 5) as u32
    }

    /// Whether the branch is likely not taken (<= 20%).
    #[inline]
    pub fn is_unlikely_taken(self) -> bool {
        self.taken_numer <= (u32::MAX as u64 / 5) as u32
    }

    /// Whether the hint is biased away from 50/50.
    #[inline]
    pub fn is_biased(self) -> bool {
        let mid = u32::MAX / 2;
        let diff = if self.taken_numer > mid {
            self.taken_numer - mid
        } else {
            mid - self.taken_numer
        };
        diff > (u32::MAX / 5)
    }

    /// Taken probability as f64 [0.0, 1.0].
    #[inline]
    pub fn taken_f64(self) -> f64 {
        self.taken_numer as f64 / u32::MAX as f64
    }
}

// =============================================================================
// PGO Call Target
// =============================================================================

/// A single observed call target at a call site.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PgoCallTarget {
    /// Bytecode offset of the call instruction.
    pub offset: u32,
    /// Unique identifier of the callee code unit.
    pub target_id: u32,
    /// Call frequency (raw count).
    pub frequency: u64,
}

// =============================================================================
// Call Site Profile
// =============================================================================

/// Aggregated call-site profile from PGO data.
///
/// Contains all observed call targets for a single bytecode offset,
/// sorted by frequency (most frequent first).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CallSiteProfile {
    /// Bytecode offset of the call instruction.
    pub offset: u32,
    /// Targets sorted by frequency (descending).
    pub targets: Vec<PgoCallTarget>,
    /// Total calls observed at this site.
    pub total_calls: u64,
}

/// Classification of a call site's polymorphism.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CallMorphism {
    /// No calls observed.
    Cold,
    /// Single target (>95% of calls).
    Monomorphic,
    /// 2-4 significant targets.
    Polymorphic,
    /// 5+ significant targets; too many to speculate on.
    Megamorphic,
}

impl CallSiteProfile {
    /// Create from a `CallProfile` at a given bytecode offset.
    pub fn from_call_profile(offset: u32, cp: &CallProfile) -> Self {
        let targets: Vec<PgoCallTarget> = cp
            .targets()
            .iter()
            .map(|t| PgoCallTarget {
                offset,
                target_id: t.target_id,
                frequency: t.count,
            })
            .collect();

        Self {
            offset,
            targets,
            total_calls: cp.total(),
        }
    }

    /// Classify the call site's morphism.
    #[inline]
    pub fn morphism(&self) -> CallMorphism {
        if self.total_calls == 0 {
            return CallMorphism::Cold;
        }

        let significant = self
            .targets
            .iter()
            .filter(|t| t.frequency as f64 / self.total_calls.max(1) as f64 > 0.01)
            .count();

        match significant {
            0 => CallMorphism::Cold,
            1 => {
                // Check >95% dominance for true monomorphism
                if let Some(primary) = self.targets.first() {
                    if primary.frequency as f64 / self.total_calls as f64 > 0.95 {
                        CallMorphism::Monomorphic
                    } else {
                        CallMorphism::Polymorphic
                    }
                } else {
                    CallMorphism::Cold
                }
            }
            2..=4 => CallMorphism::Polymorphic,
            _ => CallMorphism::Megamorphic,
        }
    }

    /// Get the dominant (most frequently called) target.
    #[inline]
    pub fn dominant_target(&self) -> Option<&PgoCallTarget> {
        self.targets.first()
    }

    /// Whether inline speculation is worthwhile for this site.
    #[inline]
    pub fn is_inline_candidate(&self) -> bool {
        matches!(self.morphism(), CallMorphism::Monomorphic)
    }
}

// =============================================================================
// Type Hint from PGO
// =============================================================================

/// PGO-derived type hint for an operation site.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PgoTypeHint {
    /// Monomorphic: single type observed >95% of the time.
    Monomorphic {
        /// The dominant type ID.
        type_id: u8,
    },
    /// Polymorphic: 2-4 types with significant frequency.
    Polymorphic,
    /// Megamorphic: 5+ types; too many to speculate on.
    Megamorphic,
    /// No profile data available for this site.
    NoData,
}

impl PgoTypeHint {
    /// Derive from a `TypeProfile`.
    pub fn from_type_profile(tp: &TypeProfile) -> Self {
        if tp.total() < 10 {
            return Self::NoData;
        }
        if tp.is_monomorphic() {
            if let Some(entry) = tp.dominant_type() {
                return Self::Monomorphic {
                    type_id: entry.type_id,
                };
            }
        }
        if tp.is_megamorphic() {
            return Self::Megamorphic;
        }
        if tp.is_polymorphic() {
            return Self::Polymorphic;
        }
        Self::NoData
    }

    /// Whether speculation on a single type is safe.
    #[inline]
    pub fn is_specialize_candidate(self) -> bool {
        matches!(self, Self::Monomorphic { .. })
    }
}

// =============================================================================
// Speculation Provider Trait
// =============================================================================

/// Central interface for querying PGO speculation information during
/// JIT compilation.
///
/// This trait abstracts over the source of profile data, enabling
/// both real PGO-backed providers and mock providers for testing.
pub trait SpeculationProvider {
    /// Get a branch probability hint for the given bytecode offset.
    fn get_branch_hint(&self, offset: u32) -> Option<PgoBranchHint>;

    /// Get call site profile for the given bytecode offset.
    fn get_call_targets(&self, offset: u32) -> Option<CallSiteProfile>;

    /// Get a type hint for the given bytecode offset.
    fn get_type_hint(&self, offset: u32) -> PgoTypeHint;

    /// Whether any profile data is available.
    fn has_profile_data(&self) -> bool;

    /// Total execution count of the code unit.
    fn execution_count(&self) -> u64;
}

// =============================================================================
// PGO-Backed Speculation Provider
// =============================================================================

/// Concrete `SpeculationProvider` backed by `ProfileData`.
///
/// This is the primary implementation used during Tier 2 JIT compilation.
/// It wraps a `ProfileData` snapshot and translates raw profile counts
/// into typed speculation decisions.
#[derive(Debug, Clone)]
pub struct PgoSpeculationProvider {
    /// The underlying profile data.
    profile: ProfileData,
}

impl PgoSpeculationProvider {
    /// Create a new provider from profile data.
    #[inline]
    pub fn new(profile: ProfileData) -> Self {
        Self { profile }
    }

    /// Borrow the underlying profile data.
    #[inline]
    pub fn profile(&self) -> &ProfileData {
        &self.profile
    }
}

impl SpeculationProvider for PgoSpeculationProvider {
    #[inline]
    fn get_branch_hint(&self, offset: u32) -> Option<PgoBranchHint> {
        self.profile
            .branch_at(offset)
            .map(|bp| PgoBranchHint::from_counts(offset, bp.taken, bp.not_taken))
    }

    fn get_call_targets(&self, offset: u32) -> Option<CallSiteProfile> {
        self.profile
            .call_at(offset)
            .map(|cp| CallSiteProfile::from_call_profile(offset, cp))
    }

    fn get_type_hint(&self, offset: u32) -> PgoTypeHint {
        self.profile
            .type_at(offset)
            .map_or(PgoTypeHint::NoData, |tp| PgoTypeHint::from_type_profile(tp))
    }

    #[inline]
    fn has_profile_data(&self) -> bool {
        true
    }

    #[inline]
    fn execution_count(&self) -> u64 {
        self.profile.execution_count()
    }
}

// =============================================================================
// Empty Speculation Provider (no PGO data)
// =============================================================================

/// A no-op speculation provider used when no profile data is available.
///
/// Returns `None`/`NoData` for all queries. Used for baseline compilation
/// or when PGO is disabled.
#[derive(Debug, Clone, Copy, Default)]
pub struct NoSpeculation;

impl SpeculationProvider for NoSpeculation {
    #[inline]
    fn get_branch_hint(&self, _offset: u32) -> Option<PgoBranchHint> {
        None
    }

    #[inline]
    fn get_call_targets(&self, _offset: u32) -> Option<CallSiteProfile> {
        None
    }

    #[inline]
    fn get_type_hint(&self, _offset: u32) -> PgoTypeHint {
        PgoTypeHint::NoData
    }

    #[inline]
    fn has_profile_data(&self) -> bool {
        false
    }

    #[inline]
    fn execution_count(&self) -> u64 {
        0
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::profile_data::{BranchProfile, CallProfile, ProfileData, TypeProfile};

    // =========================================================================
    // PgoBranchHint Tests
    // =========================================================================

    #[test]
    fn test_branch_hint_from_counts_biased_taken() {
        let hint = PgoBranchHint::from_counts(10, 900, 100);
        assert!(hint.is_likely_taken());
        assert!(!hint.is_unlikely_taken());
        assert!(hint.is_biased());
        assert!((hint.taken_f64() - 0.9).abs() < 0.01);
    }

    #[test]
    fn test_branch_hint_from_counts_biased_not_taken() {
        let hint = PgoBranchHint::from_counts(20, 50, 950);
        assert!(!hint.is_likely_taken());
        assert!(hint.is_unlikely_taken());
        assert!(hint.is_biased());
    }

    #[test]
    fn test_branch_hint_from_counts_even() {
        let hint = PgoBranchHint::from_counts(30, 500, 500);
        assert!(!hint.is_likely_taken());
        assert!(!hint.is_unlikely_taken());
        assert!(!hint.is_biased());
        assert!((hint.taken_f64() - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_branch_hint_from_counts_all_taken() {
        let hint = PgoBranchHint::from_counts(0, 1000, 0);
        assert!(hint.is_likely_taken());
        assert!(!hint.is_unlikely_taken());
        assert!(hint.is_biased());
        assert!((hint.taken_f64() - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_branch_hint_from_counts_none_taken() {
        let hint = PgoBranchHint::from_counts(0, 0, 1000);
        assert!(!hint.is_likely_taken());
        assert!(hint.is_unlikely_taken());
    }

    #[test]
    fn test_branch_hint_from_counts_zero_total() {
        let hint = PgoBranchHint::from_counts(0, 0, 0);
        // zero/zero → EVEN (50/50)
        assert!(!hint.is_likely_taken());
        assert!(!hint.is_unlikely_taken());
    }

    #[test]
    fn test_branch_hint_from_probability() {
        let prob = BranchProbability::LIKELY;
        let hint = PgoBranchHint::from_probability(42, prob);
        assert_eq!(hint.offset, 42);
        assert!(hint.is_likely_taken());
    }

    #[test]
    fn test_branch_hint_as_branch_probability_roundtrip() {
        let original = BranchProbability::from_f64(0.85);
        let hint = PgoBranchHint::from_probability(0, original);
        let recovered = hint.as_branch_probability();
        assert!((recovered.as_f64() - 0.85).abs() < 0.01);
    }

    #[test]
    fn test_branch_hint_boundary_80_percent() {
        // Exactly 80% should be "likely taken"
        let hint = PgoBranchHint::from_counts(0, 800, 200);
        assert!(hint.is_likely_taken());
    }

    #[test]
    fn test_branch_hint_boundary_20_percent() {
        // Exactly 20% should be "unlikely taken"
        let hint = PgoBranchHint::from_counts(0, 200, 800);
        assert!(hint.is_unlikely_taken());
    }

    // =========================================================================
    // CallSiteProfile Tests
    // =========================================================================

    #[test]
    fn test_call_site_profile_cold() {
        let cp = CallProfile::new();
        let csp = CallSiteProfile::from_call_profile(0, &cp);
        assert_eq!(csp.morphism(), CallMorphism::Cold);
        assert!(!csp.is_inline_candidate());
        assert!(csp.dominant_target().is_none());
    }

    #[test]
    fn test_call_site_profile_monomorphic() {
        let mut cp = CallProfile::new();
        for _ in 0..100 {
            cp.record(42);
        }
        let csp = CallSiteProfile::from_call_profile(10, &cp);
        assert_eq!(csp.morphism(), CallMorphism::Monomorphic);
        assert!(csp.is_inline_candidate());
        assert_eq!(csp.dominant_target().unwrap().target_id, 42);
        assert_eq!(csp.total_calls, 100);
    }

    #[test]
    fn test_call_site_profile_polymorphic() {
        let mut cp = CallProfile::new();
        for _ in 0..50 {
            cp.record(1);
        }
        for _ in 0..30 {
            cp.record(2);
        }
        for _ in 0..20 {
            cp.record(3);
        }
        let csp = CallSiteProfile::from_call_profile(20, &cp);
        assert_eq!(csp.morphism(), CallMorphism::Polymorphic);
        assert!(!csp.is_inline_candidate());
    }

    #[test]
    fn test_call_site_profile_megamorphic() {
        let mut cp = CallProfile::new();
        for target_id in 0..6u32 {
            for _ in 0..20 {
                cp.record(target_id);
            }
        }
        let csp = CallSiteProfile::from_call_profile(30, &cp);
        assert_eq!(csp.morphism(), CallMorphism::Megamorphic);
        assert!(!csp.is_inline_candidate());
    }

    #[test]
    fn test_call_site_profile_near_monomorphic() {
        // 94% one target: just under the 95% threshold → Polymorphic
        let mut cp = CallProfile::new();
        for _ in 0..94 {
            cp.record(1);
        }
        for _ in 0..6 {
            cp.record(2);
        }
        let csp = CallSiteProfile::from_call_profile(0, &cp);
        // 94% is below the 95% monomorphic threshold
        assert_eq!(csp.morphism(), CallMorphism::Polymorphic);
    }

    #[test]
    fn test_call_site_profile_dominant_target_order() {
        let mut cp = CallProfile::new();
        for _ in 0..10 {
            cp.record(1);
        }
        for _ in 0..90 {
            cp.record(2);
        }
        let csp = CallSiteProfile::from_call_profile(0, &cp);
        // Most frequent should be first
        assert_eq!(csp.dominant_target().unwrap().target_id, 2);
        assert_eq!(csp.dominant_target().unwrap().frequency, 90);
    }

    // =========================================================================
    // PgoTypeHint Tests
    // =========================================================================

    #[test]
    fn test_type_hint_no_data() {
        let tp = TypeProfile::new();
        assert_eq!(PgoTypeHint::from_type_profile(&tp), PgoTypeHint::NoData);
    }

    #[test]
    fn test_type_hint_insufficient_data() {
        let mut tp = TypeProfile::new();
        for _ in 0..5 {
            tp.record(1);
        }
        // Only 5 observations (threshold = 10)
        assert_eq!(PgoTypeHint::from_type_profile(&tp), PgoTypeHint::NoData);
    }

    #[test]
    fn test_type_hint_monomorphic() {
        let mut tp = TypeProfile::new();
        for _ in 0..100 {
            tp.record(3);
        }
        let hint = PgoTypeHint::from_type_profile(&tp);
        assert_eq!(hint, PgoTypeHint::Monomorphic { type_id: 3 });
        assert!(hint.is_specialize_candidate());
    }

    #[test]
    fn test_type_hint_polymorphic() {
        let mut tp = TypeProfile::new();
        for _ in 0..50 {
            tp.record(1);
        }
        for _ in 0..50 {
            tp.record(2);
        }
        let hint = PgoTypeHint::from_type_profile(&tp);
        assert_eq!(hint, PgoTypeHint::Polymorphic);
        assert!(!hint.is_specialize_candidate());
    }

    #[test]
    fn test_type_hint_megamorphic() {
        let mut tp = TypeProfile::new();
        for type_id in 0..6u8 {
            for _ in 0..20 {
                tp.record(type_id);
            }
        }
        let hint = PgoTypeHint::from_type_profile(&tp);
        assert_eq!(hint, PgoTypeHint::Megamorphic);
        assert!(!hint.is_specialize_candidate());
    }

    // =========================================================================
    // PgoSpeculationProvider Tests
    // =========================================================================

    fn make_profile_with_branch(offset: u32, taken: u64, not_taken: u64) -> ProfileData {
        let mut pd = ProfileData::new(1);
        for _ in 0..taken {
            pd.record_branch(offset, true);
        }
        for _ in 0..not_taken {
            pd.record_branch(offset, false);
        }
        pd
    }

    #[test]
    fn test_provider_has_profile_data() {
        let pd = ProfileData::new(1);
        let provider = PgoSpeculationProvider::new(pd);
        assert!(provider.has_profile_data());
    }

    #[test]
    fn test_provider_execution_count() {
        let mut pd = ProfileData::new(1);
        for _ in 0..42 {
            pd.record_execution();
        }
        let provider = PgoSpeculationProvider::new(pd);
        assert_eq!(provider.execution_count(), 42);
    }

    #[test]
    fn test_provider_branch_hint_present() {
        let pd = make_profile_with_branch(10, 900, 100);
        let provider = PgoSpeculationProvider::new(pd);
        let hint = provider.get_branch_hint(10);
        assert!(hint.is_some());
        let h = hint.unwrap();
        assert!(h.is_likely_taken());
        assert_eq!(h.offset, 10);
    }

    #[test]
    fn test_provider_branch_hint_absent() {
        let pd = ProfileData::new(1);
        let provider = PgoSpeculationProvider::new(pd);
        assert!(provider.get_branch_hint(99).is_none());
    }

    #[test]
    fn test_provider_call_targets_present() {
        let mut pd = ProfileData::new(1);
        for _ in 0..100 {
            pd.record_call(20, 55);
        }
        let provider = PgoSpeculationProvider::new(pd);
        let csp = provider.get_call_targets(20);
        assert!(csp.is_some());
        let profile = csp.unwrap();
        assert_eq!(profile.morphism(), CallMorphism::Monomorphic);
        assert_eq!(profile.dominant_target().unwrap().target_id, 55);
    }

    #[test]
    fn test_provider_call_targets_absent() {
        let pd = ProfileData::new(1);
        let provider = PgoSpeculationProvider::new(pd);
        assert!(provider.get_call_targets(99).is_none());
    }

    #[test]
    fn test_provider_type_hint_present() {
        let mut pd = ProfileData::new(1);
        for _ in 0..50 {
            pd.record_type(30, 7);
        }
        let provider = PgoSpeculationProvider::new(pd);
        let hint = provider.get_type_hint(30);
        assert_eq!(hint, PgoTypeHint::Monomorphic { type_id: 7 });
    }

    #[test]
    fn test_provider_type_hint_absent() {
        let pd = ProfileData::new(1);
        let provider = PgoSpeculationProvider::new(pd);
        assert_eq!(provider.get_type_hint(99), PgoTypeHint::NoData);
    }

    #[test]
    fn test_provider_profile_accessor() {
        let pd = ProfileData::new(42);
        let provider = PgoSpeculationProvider::new(pd);
        assert_eq!(provider.profile().code_id(), 42);
    }

    // =========================================================================
    // NoSpeculation Tests
    // =========================================================================

    #[test]
    fn test_no_speculation_branch() {
        let ns = NoSpeculation;
        assert!(ns.get_branch_hint(0).is_none());
    }

    #[test]
    fn test_no_speculation_call() {
        let ns = NoSpeculation;
        assert!(ns.get_call_targets(0).is_none());
    }

    #[test]
    fn test_no_speculation_type() {
        let ns = NoSpeculation;
        assert_eq!(ns.get_type_hint(0), PgoTypeHint::NoData);
    }

    #[test]
    fn test_no_speculation_has_profile() {
        let ns = NoSpeculation;
        assert!(!ns.has_profile_data());
    }

    #[test]
    fn test_no_speculation_execution_count() {
        let ns = NoSpeculation;
        assert_eq!(ns.execution_count(), 0);
    }

    // =========================================================================
    // Integration: Multiple branch sites
    // =========================================================================

    #[test]
    fn test_provider_multiple_branch_sites() {
        let mut pd = ProfileData::new(1);
        // Branch at offset 10: biased taken
        for _ in 0..900 {
            pd.record_branch(10, true);
        }
        for _ in 0..100 {
            pd.record_branch(10, false);
        }
        // Branch at offset 20: biased not taken
        for _ in 0..100 {
            pd.record_branch(20, true);
        }
        for _ in 0..900 {
            pd.record_branch(20, false);
        }
        // Branch at offset 30: even
        for _ in 0..500 {
            pd.record_branch(30, true);
        }
        for _ in 0..500 {
            pd.record_branch(30, false);
        }

        let provider = PgoSpeculationProvider::new(pd);

        let h10 = provider.get_branch_hint(10).unwrap();
        assert!(h10.is_likely_taken());
        assert!(h10.is_biased());

        let h20 = provider.get_branch_hint(20).unwrap();
        assert!(h20.is_unlikely_taken());
        assert!(h20.is_biased());

        let h30 = provider.get_branch_hint(30).unwrap();
        assert!(!h30.is_likely_taken());
        assert!(!h30.is_unlikely_taken());
        assert!(!h30.is_biased());
    }

    // =========================================================================
    // Integration: Mixed profile data
    // =========================================================================

    #[test]
    fn test_provider_mixed_profile() {
        let mut pd = ProfileData::new(1);
        pd.record_execution();
        pd.record_execution();
        pd.record_execution();

        // Branch data
        for _ in 0..95 {
            pd.record_branch(5, true);
        }
        for _ in 0..5 {
            pd.record_branch(5, false);
        }

        // Type data
        for _ in 0..100 {
            pd.record_type(8, 2);
        }

        // Call data
        for _ in 0..80 {
            pd.record_call(12, 99);
        }
        for _ in 0..20 {
            pd.record_call(12, 100);
        }

        let provider = PgoSpeculationProvider::new(pd);
        assert_eq!(provider.execution_count(), 3);

        let bh = provider.get_branch_hint(5).unwrap();
        assert!(bh.is_likely_taken());

        let th = provider.get_type_hint(8);
        assert_eq!(th, PgoTypeHint::Monomorphic { type_id: 2 });

        let ch = provider.get_call_targets(12).unwrap();
        assert_eq!(ch.morphism(), CallMorphism::Polymorphic); // 80/20 split
        assert_eq!(ch.dominant_target().unwrap().target_id, 99);
    }
}
