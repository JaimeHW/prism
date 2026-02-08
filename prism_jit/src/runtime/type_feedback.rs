//! Type Feedback Oracle for Speculative Optimization.
//!
//! Provides a high-level query interface over raw [`ProfileData`] type profiles
//! to produce speculation decisions for the Tier 2 optimizing JIT.
//!
//! # Architecture
//!
//! ```text
//! ┌──────────────┐     ┌──────────────────────┐     ┌──────────────────┐
//! │  ProfileData │────►│  TypeFeedbackOracle   │────►│  Graph Builder   │
//! │  (raw data)  │     │  (decision engine)    │     │  (speculative    │
//! │              │     │                       │     │   guards)        │
//! └──────────────┘     └──────────────────────┘     └──────────────────┘
//! ```
//!
//! # Decision Flow
//!
//! For each bytecode offset with type feedback:
//! 1. Retrieve `TypeProfile` from `ProfileData`
//! 2. Classify stability (Monomorphic/Polymorphic/Megamorphic/Unknown)
//! 3. Compute confidence based on sample count and dominant frequency
//! 4. Produce `SpeculationDecision` with appropriate guard type
//!
//! # Performance
//!
//! - All lookups are O(1) hash table access via `ProfileData`
//! - Decision computation is O(k) where k = number of observed types (typically ≤ 5)
//! - Oracle is stateless and can be shared across compilation units

use super::profile_data::{CallProfile, ProfileData, TypeProfile, TypeProfileEntry};

// =============================================================================
// Observed Types
// =============================================================================

/// Canonical type classification used by the oracle.
///
/// Maps from the raw `type_id: u8` stored in `TypeProfileEntry` to a
/// semantic type that the JIT can reason about. The discriminant values
/// match the `TypeHint` repr used by the interpreter's profile instrumentation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum ObservedType {
    /// Small integer (i64, NaN-boxed).
    Int = 0,
    /// Double-precision float (f64, NaN-boxed).
    Float = 1,
    /// Boolean value.
    Bool = 2,
    /// None/null sentinel.
    None = 3,
    /// Heap-allocated string.
    String = 4,
    /// List (dynamic array).
    List = 5,
    /// Dictionary (hash map).
    Dict = 6,
    /// Tuple (immutable sequence).
    Tuple = 7,
    /// Set.
    Set = 8,
    /// User-defined class instance.
    Object = 9,
    /// Callable (function, method, closure).
    Callable = 10,
    /// Unknown/unrecognized type.
    Unknown = 255,
}

impl ObservedType {
    /// Convert from the raw `type_id` stored in profile data.
    #[inline]
    pub fn from_type_id(id: u8) -> Self {
        match id {
            0 => Self::Int,
            1 => Self::Float,
            2 => Self::Bool,
            3 => Self::None,
            4 => Self::String,
            5 => Self::List,
            6 => Self::Dict,
            7 => Self::Tuple,
            8 => Self::Set,
            9 => Self::Object,
            10 => Self::Callable,
            _ => Self::Unknown,
        }
    }

    /// Whether this type can benefit from speculative integer operations.
    #[inline]
    pub fn is_numeric(self) -> bool {
        matches!(self, Self::Int | Self::Float)
    }

    /// Whether this type is a collection that benefits from inline length checks.
    #[inline]
    pub fn is_collection(self) -> bool {
        matches!(self, Self::List | Self::Dict | Self::Tuple | Self::Set)
    }

    /// Whether this type is a heap-allocated object (needs GC barriers).
    #[inline]
    pub fn is_heap_allocated(self) -> bool {
        matches!(
            self,
            Self::String
                | Self::List
                | Self::Dict
                | Self::Tuple
                | Self::Set
                | Self::Object
                | Self::Callable
        )
    }
}

// =============================================================================
// Type Stability Classification
// =============================================================================

/// Classification of type behavior at a given site.
///
/// Determines whether speculation is safe and what kind of guard to emit.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TypeStability {
    /// Single type observed >95% of the time. Safe for monomorphic guard.
    Monomorphic(ObservedType),
    /// 2-4 types observed with reasonable frequency. Suitable for type switch.
    Polymorphic,
    /// 5+ distinct types or no clear winner. Don't speculate.
    Megamorphic,
    /// Insufficient data (< minimum sample count). Don't speculate yet.
    Unstable,
}

impl TypeStability {
    /// Whether it's safe to emit speculative code for this site.
    #[inline]
    pub fn can_speculate(self) -> bool {
        matches!(self, Self::Monomorphic(_))
    }

    /// Whether a polymorphic type switch is worthwhile.
    #[inline]
    pub fn can_type_switch(self) -> bool {
        matches!(self, Self::Monomorphic(_) | Self::Polymorphic)
    }
}

// =============================================================================
// Speculation Decision
// =============================================================================

/// The oracle's recommendation for how to compile a given operation.
#[derive(Debug, Clone, PartialEq)]
pub enum SpeculationDecision {
    /// Emit a monomorphic guard + specialized code path.
    ///
    /// Fields:
    /// - `guard_type`: The type to guard for
    /// - `confidence`: How confident we are (0.0 - 1.0)
    /// - `deopt_count_estimate`: Expected deopt frequency (lower = better)
    Speculate {
        guard_type: ObservedType,
        confidence: f64,
        deopt_count_estimate: f64,
    },

    /// Emit a polymorphic type switch with multiple specialized paths.
    ///
    /// Fields:
    /// - `types`: Ordered list of (type, probability) pairs
    TypeSwitch { types: Vec<(ObservedType, f64)> },

    /// Don't speculate — emit generic code.
    Generic { reason: GenericReason },
}

/// Reason for falling back to generic code.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GenericReason {
    /// Not enough samples to be confident.
    InsufficientData,
    /// Too many types observed (megamorphic).
    Megamorphic,
    /// The operation doesn't benefit from specialization.
    NotSpecializable,
    /// No profile data available for this offset.
    NoProfile,
    /// Confidence below threshold.
    LowConfidence,
}

// =============================================================================
// Call Speculation Decision
// =============================================================================

/// The oracle's recommendation for a call site.
#[derive(Debug, Clone, PartialEq)]
pub enum CallSpeculation {
    /// Monomorphic call — inline or devirtualize.
    Monomorphic { target_id: u32, confidence: f64 },
    /// Polymorphic dispatch — type-switch with multiple targets.
    Polymorphic { targets: Vec<(u32, f64)> },
    /// Megamorphic — use generic dispatch.
    Megamorphic,
    /// No call profile data.
    Unknown,
}

// =============================================================================
// Oracle Configuration
// =============================================================================

/// Configuration for the type feedback oracle's decision thresholds.
#[derive(Debug, Clone)]
pub struct OracleConfig {
    /// Minimum number of observations before speculation (default: 30).
    ///
    /// Lower values enable earlier speculation but with higher deopt risk.
    /// V8 uses ~100, HotSpot uses ~10000. We use 30 for a balanced approach
    /// that works well with our 3-tier system.
    pub min_samples: u64,

    /// Confidence threshold for monomorphic speculation (default: 0.90).
    ///
    /// The dominant type must represent at least this fraction of observations.
    pub monomorphic_threshold: f64,

    /// Confidence threshold for polymorphic type switch (default: 0.70).
    ///
    /// The top-N types must collectively represent this fraction.
    pub polymorphic_threshold: f64,

    /// Maximum number of types in a polymorphic switch (default: 4).
    pub max_polymorphic_types: usize,

    /// Minimum confidence for call site devirtualization (default: 0.95).
    pub call_monomorphic_threshold: f64,
}

impl Default for OracleConfig {
    fn default() -> Self {
        Self {
            min_samples: 30,
            monomorphic_threshold: 0.90,
            polymorphic_threshold: 0.70,
            max_polymorphic_types: 4,
            call_monomorphic_threshold: 0.95,
        }
    }
}

impl OracleConfig {
    /// Conservative configuration — higher thresholds, fewer deopts.
    pub fn conservative() -> Self {
        Self {
            min_samples: 100,
            monomorphic_threshold: 0.98,
            polymorphic_threshold: 0.85,
            max_polymorphic_types: 3,
            call_monomorphic_threshold: 0.99,
        }
    }

    /// Aggressive configuration — lower thresholds, more speculation.
    pub fn aggressive() -> Self {
        Self {
            min_samples: 10,
            monomorphic_threshold: 0.80,
            polymorphic_threshold: 0.60,
            max_polymorphic_types: 6,
            call_monomorphic_threshold: 0.90,
        }
    }
}

// =============================================================================
// Type Feedback Oracle
// =============================================================================

/// The type feedback oracle — the central decision engine for speculative
/// optimization in the Tier 2 JIT.
///
/// Given a `ProfileData` reference and a bytecode offset, the oracle
/// determines whether to speculate, what type to guard for, and how
/// confident the prediction is.
///
/// # Thread Safety
///
/// The oracle borrows `ProfileData` immutably and is stateless — it can
/// safely be used from the compilation thread without synchronization.
///
/// # Example
///
/// ```ignore
/// let oracle = TypeFeedbackOracle::new(&profile_data);
///
/// match oracle.query_type(bc_offset) {
///     SpeculationDecision::Speculate { guard_type, .. } => {
///         // Emit type guard + specialized code
///         emit_type_guard(guard_type);
///         emit_specialized_op(guard_type);
///     }
///     SpeculationDecision::Generic { .. } => {
///         // Emit generic code path
///         emit_generic_op();
///     }
///     _ => { /* handle other cases */ }
/// }
/// ```
pub struct TypeFeedbackOracle<'a> {
    /// The profile data to query.
    profile: &'a ProfileData,
    /// Decision configuration.
    config: OracleConfig,
}

impl<'a> TypeFeedbackOracle<'a> {
    /// Create a new oracle with default configuration.
    #[inline]
    pub fn new(profile: &'a ProfileData) -> Self {
        Self {
            profile,
            config: OracleConfig::default(),
        }
    }

    /// Create a new oracle with custom configuration.
    #[inline]
    pub fn with_config(profile: &'a ProfileData, config: OracleConfig) -> Self {
        Self { profile, config }
    }

    /// Get the oracle's configuration.
    #[inline]
    pub fn config(&self) -> &OracleConfig {
        &self.config
    }

    // =========================================================================
    // Type Queries
    // =========================================================================

    /// Query the type feedback for a given bytecode offset.
    ///
    /// Returns a `SpeculationDecision` indicating whether to speculate
    /// and what type guard to emit.
    pub fn query_type(&self, offset: u32) -> SpeculationDecision {
        let type_profile = match self.profile.type_at(offset) {
            Some(tp) => tp,
            None => {
                return SpeculationDecision::Generic {
                    reason: GenericReason::NoProfile,
                };
            }
        };

        // Insufficient data — don't speculate yet
        if type_profile.total() < self.config.min_samples {
            return SpeculationDecision::Generic {
                reason: GenericReason::InsufficientData,
            };
        }

        let stability = self.classify_type_stability(type_profile);

        match stability {
            TypeStability::Monomorphic(observed) => {
                let dominant = type_profile.dominant_type().unwrap();
                let confidence = dominant.count as f64 / type_profile.total() as f64;

                if confidence >= self.config.monomorphic_threshold {
                    let deopt_estimate = 1.0 - confidence;
                    SpeculationDecision::Speculate {
                        guard_type: observed,
                        confidence,
                        deopt_count_estimate: deopt_estimate,
                    }
                } else {
                    // Confidence too low for monomorphic — try polymorphic
                    self.try_polymorphic_decision(type_profile)
                }
            }
            TypeStability::Polymorphic => self.try_polymorphic_decision(type_profile),
            TypeStability::Megamorphic => SpeculationDecision::Generic {
                reason: GenericReason::Megamorphic,
            },
            TypeStability::Unstable => SpeculationDecision::Generic {
                reason: GenericReason::InsufficientData,
            },
        }
    }

    /// Query the dominant type at a given offset (fast path for simple checks).
    ///
    /// Returns `None` if no profile data or insufficient samples.
    #[inline]
    pub fn dominant_type_at(&self, offset: u32) -> Option<ObservedType> {
        let tp = self.profile.type_at(offset)?;
        if tp.total() < self.config.min_samples {
            return None;
        }
        tp.dominant_type()
            .map(|entry| ObservedType::from_type_id(entry.type_id))
    }

    /// Check if a site is monomorphic with sufficient confidence.
    #[inline]
    pub fn is_monomorphic_at(&self, offset: u32) -> bool {
        self.profile.type_at(offset).map_or(false, |tp| {
            tp.is_monomorphic() && tp.total() >= self.config.min_samples
        })
    }

    /// Get the type stability classification at a given offset.
    pub fn stability_at(&self, offset: u32) -> TypeStability {
        match self.profile.type_at(offset) {
            Some(tp) if tp.total() >= self.config.min_samples => self.classify_type_stability(tp),
            _ => TypeStability::Unstable,
        }
    }

    /// Compute confidence score for the dominant type at an offset.
    ///
    /// Returns a value between 0.0 (no confidence) and 1.0 (perfect confidence).
    /// Returns 0.0 if no profile data or insufficient samples.
    pub fn confidence_at(&self, offset: u32) -> f64 {
        match self.profile.type_at(offset) {
            Some(tp) if tp.total() >= self.config.min_samples => tp
                .dominant_type()
                .map_or(0.0, |d| d.count as f64 / tp.total() as f64),
            _ => 0.0,
        }
    }

    // =========================================================================
    // Call Site Queries
    // =========================================================================

    /// Query call site feedback for speculation/devirtualization decisions.
    pub fn query_call(&self, offset: u32) -> CallSpeculation {
        let call_profile = match self.profile.call_at(offset) {
            Some(cp) => cp,
            None => return CallSpeculation::Unknown,
        };

        if call_profile.total() < self.config.min_samples {
            return CallSpeculation::Unknown;
        }

        self.classify_call_site(call_profile)
    }

    /// Check if a call site is monomorphic (single target).
    #[inline]
    pub fn is_monomorphic_call_at(&self, offset: u32) -> bool {
        self.profile.call_at(offset).map_or(false, |cp| {
            cp.is_monomorphic() && cp.total() >= self.config.min_samples
        })
    }

    // =========================================================================
    // Loop Queries
    // =========================================================================

    /// Get the iteration count for a loop header.
    #[inline]
    pub fn loop_trip_count(&self, header_offset: u32) -> u64 {
        self.profile.loop_count(header_offset)
    }

    /// Whether a loop is hot (iterated more than threshold).
    #[inline]
    pub fn is_hot_loop(&self, header_offset: u32, threshold: u64) -> bool {
        self.loop_trip_count(header_offset) >= threshold
    }

    // =========================================================================
    // Internal Classification Methods
    // =========================================================================

    /// Classify the type stability of a profile.
    fn classify_type_stability(&self, profile: &TypeProfile) -> TypeStability {
        if profile.total() < self.config.min_samples {
            return TypeStability::Unstable;
        }

        if profile.is_megamorphic() {
            return TypeStability::Megamorphic;
        }

        if let Some(dominant) = profile.dominant_type() {
            let ratio = dominant.count as f64 / profile.total() as f64;
            let observed = ObservedType::from_type_id(dominant.type_id);

            if ratio >= self.config.monomorphic_threshold {
                return TypeStability::Monomorphic(observed);
            }
        }

        if profile.is_polymorphic() {
            return TypeStability::Polymorphic;
        }

        // Has data but no clear pattern
        TypeStability::Megamorphic
    }

    /// Attempt to produce a polymorphic type switch decision.
    fn try_polymorphic_decision(&self, profile: &TypeProfile) -> SpeculationDecision {
        let total = profile.total() as f64;
        if total == 0.0 {
            return SpeculationDecision::Generic {
                reason: GenericReason::InsufficientData,
            };
        }

        // Collect entries sorted by frequency (descending)
        let mut entries: Vec<&TypeProfileEntry> = profile.entries().iter().collect();
        entries.sort_unstable_by(|a, b| b.count.cmp(&a.count));

        // Take top-N types that pass the significance threshold (> 1%)
        let significant: Vec<(ObservedType, f64)> = entries
            .iter()
            .take(self.config.max_polymorphic_types)
            .filter(|e| e.count as f64 / total > 0.01)
            .map(|e| {
                (
                    ObservedType::from_type_id(e.type_id),
                    e.count as f64 / total,
                )
            })
            .collect();

        if significant.is_empty() {
            return SpeculationDecision::Generic {
                reason: GenericReason::InsufficientData,
            };
        }

        // Check if collective coverage meets threshold
        let coverage: f64 = significant.iter().map(|(_, p)| p).sum();
        if coverage >= self.config.polymorphic_threshold {
            if significant.len() == 1 {
                // Actually monomorphic with lower confidence
                SpeculationDecision::Speculate {
                    guard_type: significant[0].0,
                    confidence: significant[0].1,
                    deopt_count_estimate: 1.0 - significant[0].1,
                }
            } else {
                SpeculationDecision::TypeSwitch { types: significant }
            }
        } else {
            SpeculationDecision::Generic {
                reason: GenericReason::LowConfidence,
            }
        }
    }

    /// Classify a call site for devirtualization decisions.
    fn classify_call_site(&self, profile: &CallProfile) -> CallSpeculation {
        let total = profile.total() as f64;
        if total == 0.0 {
            return CallSpeculation::Unknown;
        }

        let targets = profile.targets();
        if targets.is_empty() {
            return CallSpeculation::Unknown;
        }

        // Check if the primary target dominates
        let primary = &targets[0];
        let primary_confidence = primary.count as f64 / total;

        if primary_confidence >= self.config.call_monomorphic_threshold {
            return CallSpeculation::Monomorphic {
                target_id: primary.target_id,
                confidence: primary_confidence,
            };
        }

        // Check if top-N targets cover enough
        let top_targets: Vec<(u32, f64)> = targets
            .iter()
            .take(self.config.max_polymorphic_types)
            .filter(|t| t.count as f64 / total > 0.01)
            .map(|t| (t.target_id, t.count as f64 / total))
            .collect();

        let coverage: f64 = top_targets.iter().map(|(_, p)| p).sum();

        if coverage >= self.config.polymorphic_threshold && top_targets.len() <= 4 {
            CallSpeculation::Polymorphic {
                targets: top_targets,
            }
        } else {
            CallSpeculation::Megamorphic
        }
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // Helper
    // =========================================================================

    fn make_profile_with_types(offset: u32, type_samples: &[(u8, u64)]) -> ProfileData {
        let mut profile = ProfileData::new(1);
        for &(type_id, count) in type_samples {
            for _ in 0..count {
                profile.record_type(offset, type_id);
            }
        }
        profile
    }

    fn make_profile_with_calls(offset: u32, call_samples: &[(u32, u64)]) -> ProfileData {
        let mut profile = ProfileData::new(1);
        for &(target_id, count) in call_samples {
            for _ in 0..count {
                profile.record_call(offset, target_id);
            }
        }
        profile
    }

    // =========================================================================
    // ObservedType Tests
    // =========================================================================

    #[test]
    fn test_observed_type_from_type_id_valid() {
        assert_eq!(ObservedType::from_type_id(0), ObservedType::Int);
        assert_eq!(ObservedType::from_type_id(1), ObservedType::Float);
        assert_eq!(ObservedType::from_type_id(2), ObservedType::Bool);
        assert_eq!(ObservedType::from_type_id(3), ObservedType::None);
        assert_eq!(ObservedType::from_type_id(4), ObservedType::String);
        assert_eq!(ObservedType::from_type_id(5), ObservedType::List);
        assert_eq!(ObservedType::from_type_id(6), ObservedType::Dict);
        assert_eq!(ObservedType::from_type_id(7), ObservedType::Tuple);
        assert_eq!(ObservedType::from_type_id(8), ObservedType::Set);
        assert_eq!(ObservedType::from_type_id(9), ObservedType::Object);
        assert_eq!(ObservedType::from_type_id(10), ObservedType::Callable);
    }

    #[test]
    fn test_observed_type_from_type_id_unknown() {
        assert_eq!(ObservedType::from_type_id(11), ObservedType::Unknown);
        assert_eq!(ObservedType::from_type_id(100), ObservedType::Unknown);
        assert_eq!(ObservedType::from_type_id(255), ObservedType::Unknown);
    }

    #[test]
    fn test_observed_type_is_numeric() {
        assert!(ObservedType::Int.is_numeric());
        assert!(ObservedType::Float.is_numeric());
        assert!(!ObservedType::String.is_numeric());
        assert!(!ObservedType::Bool.is_numeric());
        assert!(!ObservedType::None.is_numeric());
        assert!(!ObservedType::List.is_numeric());
    }

    #[test]
    fn test_observed_type_is_collection() {
        assert!(ObservedType::List.is_collection());
        assert!(ObservedType::Dict.is_collection());
        assert!(ObservedType::Tuple.is_collection());
        assert!(ObservedType::Set.is_collection());
        assert!(!ObservedType::Int.is_collection());
        assert!(!ObservedType::String.is_collection());
        assert!(!ObservedType::Object.is_collection());
    }

    #[test]
    fn test_observed_type_is_heap_allocated() {
        assert!(ObservedType::String.is_heap_allocated());
        assert!(ObservedType::List.is_heap_allocated());
        assert!(ObservedType::Dict.is_heap_allocated());
        assert!(ObservedType::Tuple.is_heap_allocated());
        assert!(ObservedType::Set.is_heap_allocated());
        assert!(ObservedType::Object.is_heap_allocated());
        assert!(ObservedType::Callable.is_heap_allocated());
        assert!(!ObservedType::Int.is_heap_allocated());
        assert!(!ObservedType::Float.is_heap_allocated());
        assert!(!ObservedType::Bool.is_heap_allocated());
        assert!(!ObservedType::None.is_heap_allocated());
    }

    // =========================================================================
    // TypeStability Tests
    // =========================================================================

    #[test]
    fn test_type_stability_can_speculate() {
        assert!(TypeStability::Monomorphic(ObservedType::Int).can_speculate());
        assert!(!TypeStability::Polymorphic.can_speculate());
        assert!(!TypeStability::Megamorphic.can_speculate());
        assert!(!TypeStability::Unstable.can_speculate());
    }

    #[test]
    fn test_type_stability_can_type_switch() {
        assert!(TypeStability::Monomorphic(ObservedType::Int).can_type_switch());
        assert!(TypeStability::Polymorphic.can_type_switch());
        assert!(!TypeStability::Megamorphic.can_type_switch());
        assert!(!TypeStability::Unstable.can_type_switch());
    }

    // =========================================================================
    // OracleConfig Tests
    // =========================================================================

    #[test]
    fn test_oracle_config_default() {
        let config = OracleConfig::default();
        assert_eq!(config.min_samples, 30);
        assert!((config.monomorphic_threshold - 0.90).abs() < f64::EPSILON);
        assert!((config.polymorphic_threshold - 0.70).abs() < f64::EPSILON);
        assert_eq!(config.max_polymorphic_types, 4);
        assert!((config.call_monomorphic_threshold - 0.95).abs() < f64::EPSILON);
    }

    #[test]
    fn test_oracle_config_conservative() {
        let config = OracleConfig::conservative();
        assert_eq!(config.min_samples, 100);
        assert!(config.monomorphic_threshold > OracleConfig::default().monomorphic_threshold);
    }

    #[test]
    fn test_oracle_config_aggressive() {
        let config = OracleConfig::aggressive();
        assert_eq!(config.min_samples, 10);
        assert!(config.monomorphic_threshold < OracleConfig::default().monomorphic_threshold);
    }

    // =========================================================================
    // Oracle Type Query Tests
    // =========================================================================

    #[test]
    fn test_oracle_query_no_profile() {
        let profile = ProfileData::new(1);
        let oracle = TypeFeedbackOracle::new(&profile);
        let decision = oracle.query_type(42);
        assert_eq!(
            decision,
            SpeculationDecision::Generic {
                reason: GenericReason::NoProfile
            }
        );
    }

    #[test]
    fn test_oracle_query_insufficient_data() {
        // Only 5 samples (below default threshold of 30)
        let profile = make_profile_with_types(10, &[(0, 5)]);
        let oracle = TypeFeedbackOracle::new(&profile);
        let decision = oracle.query_type(10);
        assert_eq!(
            decision,
            SpeculationDecision::Generic {
                reason: GenericReason::InsufficientData
            }
        );
    }

    #[test]
    fn test_oracle_query_monomorphic_int() {
        // 95 ints, 5 floats = 95% int → monomorphic
        let profile = make_profile_with_types(10, &[(0, 95), (1, 5)]);
        let oracle = TypeFeedbackOracle::new(&profile);
        let decision = oracle.query_type(10);
        match decision {
            SpeculationDecision::Speculate {
                guard_type,
                confidence,
                deopt_count_estimate,
            } => {
                assert_eq!(guard_type, ObservedType::Int);
                assert!(confidence >= 0.90);
                assert!(deopt_count_estimate <= 0.10);
            }
            other => panic!("Expected Speculate, got {:?}", other),
        }
    }

    #[test]
    fn test_oracle_query_monomorphic_string() {
        // 100 strings → 100% monomorphic
        let profile = make_profile_with_types(5, &[(4, 100)]);
        let oracle = TypeFeedbackOracle::new(&profile);
        let decision = oracle.query_type(5);
        match decision {
            SpeculationDecision::Speculate {
                guard_type,
                confidence,
                ..
            } => {
                assert_eq!(guard_type, ObservedType::String);
                assert!((confidence - 1.0).abs() < f64::EPSILON);
            }
            other => panic!("Expected Speculate, got {:?}", other),
        }
    }

    #[test]
    fn test_oracle_query_polymorphic() {
        // 40 ints, 35 floats, 25 strings → polymorphic
        let profile = make_profile_with_types(10, &[(0, 40), (1, 35), (4, 25)]);
        let oracle = TypeFeedbackOracle::new(&profile);
        let decision = oracle.query_type(10);
        match decision {
            SpeculationDecision::TypeSwitch { types } => {
                assert!(types.len() >= 2);
                assert!(types.len() <= 4);
                // All types should be present
                let has_int = types.iter().any(|(t, _)| *t == ObservedType::Int);
                let has_float = types.iter().any(|(t, _)| *t == ObservedType::Float);
                assert!(has_int);
                assert!(has_float);
            }
            other => panic!("Expected TypeSwitch, got {:?}", other),
        }
    }

    #[test]
    fn test_oracle_query_megamorphic() {
        // 6 types with roughly equal distribution → megamorphic
        let profile =
            make_profile_with_types(10, &[(0, 20), (1, 18), (4, 16), (5, 14), (6, 12), (7, 10)]);
        let oracle = TypeFeedbackOracle::new(&profile);
        let decision = oracle.query_type(10);
        assert_eq!(
            decision,
            SpeculationDecision::Generic {
                reason: GenericReason::Megamorphic
            }
        );
    }

    #[test]
    fn test_oracle_dominant_type_at() {
        let profile = make_profile_with_types(7, &[(0, 100)]);
        let oracle = TypeFeedbackOracle::new(&profile);
        assert_eq!(oracle.dominant_type_at(7), Some(ObservedType::Int));
        assert_eq!(oracle.dominant_type_at(99), None); // no data
    }

    #[test]
    fn test_oracle_dominant_type_insufficient_data() {
        let profile = make_profile_with_types(7, &[(0, 5)]);
        let oracle = TypeFeedbackOracle::new(&profile);
        assert_eq!(oracle.dominant_type_at(7), None); // below threshold
    }

    #[test]
    fn test_oracle_is_monomorphic_at() {
        // 98% int → monomorphic
        let profile = make_profile_with_types(10, &[(0, 98), (1, 2)]);
        let oracle = TypeFeedbackOracle::new(&profile);
        assert!(oracle.is_monomorphic_at(10));
        assert!(!oracle.is_monomorphic_at(99)); // no data
    }

    #[test]
    fn test_oracle_stability_at() {
        let profile = make_profile_with_types(10, &[(0, 100)]);
        let oracle = TypeFeedbackOracle::new(&profile);
        assert_eq!(
            oracle.stability_at(10),
            TypeStability::Monomorphic(ObservedType::Int)
        );
        assert_eq!(oracle.stability_at(99), TypeStability::Unstable);
    }

    #[test]
    fn test_oracle_confidence_at() {
        let profile = make_profile_with_types(10, &[(0, 80), (1, 20)]);
        let oracle = TypeFeedbackOracle::new(&profile);
        let conf = oracle.confidence_at(10);
        assert!((conf - 0.80).abs() < f64::EPSILON);
        assert!((oracle.confidence_at(99) - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_oracle_conservative_config_raises_threshold() {
        // 92% int — passes default but not conservative threshold
        let profile = make_profile_with_types(10, &[(0, 92), (1, 8)]);

        let default_oracle = TypeFeedbackOracle::new(&profile);
        let conservative_oracle =
            TypeFeedbackOracle::with_config(&profile, OracleConfig::conservative());

        // Default should speculate (>90%)
        match default_oracle.query_type(10) {
            SpeculationDecision::Speculate { .. } => {} // expected
            other => panic!("Expected Speculate with default, got {:?}", other),
        }

        // Conservative should NOT speculate monomorphically (>98% needed)
        // but may produce TypeSwitch since confidence is > polymorphic_threshold
        match conservative_oracle.query_type(10) {
            SpeculationDecision::Speculate { .. } => {
                panic!("Conservative should not speculate at 92%")
            }
            _ => {} // any non-speculate is correct
        }
    }

    #[test]
    fn test_oracle_aggressive_config_lowers_threshold() {
        // 82% int — fails default but passes aggressive threshold
        let profile = make_profile_with_types(10, &[(0, 82), (1, 18)]);

        let aggressive_oracle =
            TypeFeedbackOracle::with_config(&profile, OracleConfig::aggressive());

        match aggressive_oracle.query_type(10) {
            SpeculationDecision::Speculate {
                guard_type,
                confidence,
                ..
            } => {
                assert_eq!(guard_type, ObservedType::Int);
                assert!(confidence >= 0.80);
            }
            other => panic!("Expected Speculate with aggressive, got {:?}", other),
        }
    }

    // =========================================================================
    // Call Site Query Tests
    // =========================================================================

    #[test]
    fn test_oracle_call_no_profile() {
        let profile = ProfileData::new(1);
        let oracle = TypeFeedbackOracle::new(&profile);
        assert_eq!(oracle.query_call(42), CallSpeculation::Unknown);
    }

    #[test]
    fn test_oracle_call_monomorphic() {
        // 98% calls to target 42
        let profile = make_profile_with_calls(10, &[(42, 98), (43, 2)]);
        let oracle = TypeFeedbackOracle::new(&profile);
        match oracle.query_call(10) {
            CallSpeculation::Monomorphic {
                target_id,
                confidence,
            } => {
                assert_eq!(target_id, 42);
                assert!(confidence >= 0.95);
            }
            other => panic!("Expected Monomorphic call, got {:?}", other),
        }
    }

    #[test]
    fn test_oracle_call_polymorphic() {
        // 60% target 1, 30% target 2, 10% target 3
        let profile = make_profile_with_calls(10, &[(1, 60), (2, 30), (3, 10)]);
        let oracle = TypeFeedbackOracle::new(&profile);
        match oracle.query_call(10) {
            CallSpeculation::Polymorphic { targets } => {
                assert!(targets.len() >= 2);
                assert!(targets.len() <= 4);
            }
            other => panic!("Expected Polymorphic call, got {:?}", other),
        }
    }

    #[test]
    fn test_oracle_call_megamorphic() {
        // Many targets with low individual frequency
        let profile =
            make_profile_with_calls(10, &[(1, 10), (2, 10), (3, 10), (4, 10), (5, 10), (6, 10)]);
        let oracle = TypeFeedbackOracle::new(&profile);
        assert_eq!(oracle.query_call(10), CallSpeculation::Megamorphic);
    }

    #[test]
    fn test_oracle_is_monomorphic_call_at() {
        let profile = make_profile_with_calls(10, &[(42, 100)]);
        let oracle = TypeFeedbackOracle::new(&profile);
        assert!(oracle.is_monomorphic_call_at(10));
        assert!(!oracle.is_monomorphic_call_at(99));
    }

    // =========================================================================
    // Loop Query Tests
    // =========================================================================

    #[test]
    fn test_oracle_loop_trip_count() {
        let mut profile = ProfileData::new(1);
        for _ in 0..500 {
            profile.record_loop_iteration(20);
        }
        let oracle = TypeFeedbackOracle::new(&profile);
        assert_eq!(oracle.loop_trip_count(20), 500);
        assert_eq!(oracle.loop_trip_count(99), 0);
    }

    #[test]
    fn test_oracle_is_hot_loop() {
        let mut profile = ProfileData::new(1);
        for _ in 0..1000 {
            profile.record_loop_iteration(20);
        }
        let oracle = TypeFeedbackOracle::new(&profile);
        assert!(oracle.is_hot_loop(20, 100));
        assert!(!oracle.is_hot_loop(20, 2000));
        assert!(!oracle.is_hot_loop(99, 1));
    }

    // =========================================================================
    // Edge Case Tests
    // =========================================================================

    #[test]
    fn test_oracle_query_exactly_at_threshold() {
        // Exactly 30 samples (min_samples default) — should be processed
        let profile = make_profile_with_types(10, &[(0, 30)]);
        let oracle = TypeFeedbackOracle::new(&profile);
        let decision = oracle.query_type(10);
        match decision {
            SpeculationDecision::Speculate { guard_type, .. } => {
                assert_eq!(guard_type, ObservedType::Int);
            }
            other => panic!("Expected Speculate at exact threshold, got {:?}", other),
        }
    }

    #[test]
    fn test_oracle_query_just_below_threshold() {
        // 29 samples — just below threshold
        let profile = make_profile_with_types(10, &[(0, 29)]);
        let oracle = TypeFeedbackOracle::new(&profile);
        let decision = oracle.query_type(10);
        assert_eq!(
            decision,
            SpeculationDecision::Generic {
                reason: GenericReason::InsufficientData
            }
        );
    }

    #[test]
    fn test_oracle_multiple_offsets_independent() {
        let mut profile = ProfileData::new(1);
        // Offset 10: 100% int
        for _ in 0..100 {
            profile.record_type(10, 0);
        }
        // Offset 20: 100% string
        for _ in 0..100 {
            profile.record_type(20, 4);
        }

        let oracle = TypeFeedbackOracle::new(&profile);

        assert_eq!(oracle.dominant_type_at(10), Some(ObservedType::Int));
        assert_eq!(oracle.dominant_type_at(20), Some(ObservedType::String));
    }

    #[test]
    fn test_oracle_polymorphic_exactly_two_types() {
        // 55% int, 45% float → polymorphic
        let profile = make_profile_with_types(10, &[(0, 55), (1, 45)]);
        let oracle = TypeFeedbackOracle::new(&profile);
        let decision = oracle.query_type(10);
        match decision {
            SpeculationDecision::TypeSwitch { types } => {
                assert_eq!(types.len(), 2);
            }
            other => panic!("Expected TypeSwitch with 2 types, got {:?}", other),
        }
    }

    #[test]
    fn test_oracle_type_switch_ordering() {
        // Types should be ordered by frequency (descending)
        let profile = make_profile_with_types(10, &[(1, 20), (0, 50), (4, 30)]);
        let oracle = TypeFeedbackOracle::new(&profile);
        let decision = oracle.query_type(10);
        match decision {
            SpeculationDecision::TypeSwitch { types } => {
                // First should be Int (50%), then String (30%), then Float (20%)
                assert_eq!(types[0].0, ObservedType::Int);
                assert_eq!(types[1].0, ObservedType::String);
                assert_eq!(types[2].0, ObservedType::Float);
            }
            other => panic!("Expected TypeSwitch, got {:?}", other),
        }
    }

    #[test]
    fn test_oracle_single_type_below_mono_threshold_as_speculate() {
        // 85% int — below monomorphic threshold (90%) but only 1 significant type
        // Should still produce a Speculate decision if the polymorphic path
        // detects only 1 significant type
        let profile = make_profile_with_types(10, &[(0, 85), (1, 15)]);
        let oracle = TypeFeedbackOracle::new(&profile);
        let decision = oracle.query_type(10);
        // With 85%, below mono threshold but polymorphic should still work
        // since 2 types are significant
        match decision {
            SpeculationDecision::TypeSwitch { types } => {
                assert!(types.len() >= 2);
            }
            SpeculationDecision::Speculate { .. } => {
                // Also acceptable if oracle determines single type is dominant
            }
            other => panic!("Expected TypeSwitch or Speculate, got {:?}", other),
        }
    }
}
