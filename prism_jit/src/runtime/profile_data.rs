//! Profile data for Profile-Guided Optimization (PGO).
//!
//! Provides compact, serializable profile data collected during interpretation
//! for use in JIT compilation decisions. Supports branch frequency, type feedback,
//! call target tracking, and hot method detection.
//!
//! # Binary Format (`.prf`)
//!
//! ```text
//! ┌────────────────────────────────────────┐
//! │ Header (16 bytes)                      │
//! │   magic: [u8; 4] = "PPRF"             │
//! │   version: u16                         │
//! │   flags: u16                           │
//! │   section_count: u32                   │
//! │   checksum: u32                        │
//! ├────────────────────────────────────────┤
//! │ Section Directory                      │
//! │   SectionEntry × section_count         │
//! ├────────────────────────────────────────┤
//! │ Branch Profiles                        │
//! │ Type Profiles                          │
//! │ Call Profiles                          │
//! │ Execution Counts                       │
//! └────────────────────────────────────────┘
//! ```

use rustc_hash::FxHashMap;
use std::sync::atomic::{AtomicU64, Ordering};

// =============================================================================
// Constants
// =============================================================================

/// Magic bytes for profile data files.
const PROFILE_MAGIC: [u8; 4] = *b"PPRF";

/// Current profile format version.
const PROFILE_VERSION: u16 = 1;

/// Section type identifiers.
const SECTION_BRANCH: u8 = 1;
const SECTION_TYPE: u8 = 2;
const SECTION_CALL: u8 = 3;
const SECTION_EXEC_COUNT: u8 = 4;

// =============================================================================
// Branch Profile
// =============================================================================

/// Branch execution profile for a single branch site.
///
/// Tracks taken vs not-taken counts for conditional branches.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BranchProfile {
    /// Number of times the branch was taken.
    pub taken: u64,
    /// Number of times the branch was not taken.
    pub not_taken: u64,
}

impl BranchProfile {
    /// Create a new empty branch profile.
    #[inline]
    pub const fn new() -> Self {
        Self {
            taken: 0,
            not_taken: 0,
        }
    }

    /// Create with initial counts.
    #[inline]
    pub const fn with_counts(taken: u64, not_taken: u64) -> Self {
        Self { taken, not_taken }
    }

    /// Total executions of this branch site.
    #[inline]
    pub const fn total(&self) -> u64 {
        self.taken + self.not_taken
    }

    /// Probability of the branch being taken (0.0 to 1.0).
    ///
    /// Returns 0.5 if no data is available (unbiased default).
    #[inline]
    pub fn taken_probability(&self) -> f64 {
        let total = self.total();
        if total == 0 {
            0.5
        } else {
            self.taken as f64 / total as f64
        }
    }

    /// Probability of the branch not being taken (0.0 to 1.0).
    #[inline]
    pub fn not_taken_probability(&self) -> f64 {
        1.0 - self.taken_probability()
    }

    /// Whether this branch is heavily biased (>90% one direction).
    #[inline]
    pub fn is_biased(&self) -> bool {
        let prob = self.taken_probability();
        prob > 0.9 || prob < 0.1
    }

    /// Whether this branch is effectively never-taken (<1%).
    #[inline]
    pub fn is_cold(&self) -> bool {
        let total = self.total();
        total > 100 && (self.taken as f64 / total as f64) < 0.01
    }

    /// Merge another profile into this one.
    #[inline]
    pub fn merge(&mut self, other: &BranchProfile) {
        self.taken = self.taken.saturating_add(other.taken);
        self.not_taken = self.not_taken.saturating_add(other.not_taken);
    }

    /// Scale counts by a factor (useful for profile aging).
    #[inline]
    pub fn scale(&mut self, factor: f64) {
        self.taken = (self.taken as f64 * factor) as u64;
        self.not_taken = (self.not_taken as f64 * factor) as u64;
    }
}

impl Default for BranchProfile {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// Type Profile
// =============================================================================

/// Type feedback profile for a single operation site.
///
/// Tracks the frequency of observed operand type combinations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TypeProfile {
    /// Type histogram: maps TypeHint discriminant → count.
    entries: Vec<TypeProfileEntry>,
    /// Total observations.
    total: u64,
}

/// A single entry in the type histogram.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TypeProfileEntry {
    /// Type hint discriminant (matches `TypeHint` repr).
    pub type_id: u8,
    /// Observation count.
    pub count: u64,
}

impl TypeProfile {
    /// Create an empty type profile.
    #[inline]
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
            total: 0,
        }
    }

    /// Record an observation of a type.
    pub fn record(&mut self, type_id: u8) {
        self.total += 1;
        if let Some(entry) = self.entries.iter_mut().find(|e| e.type_id == type_id) {
            entry.count += 1;
        } else {
            self.entries.push(TypeProfileEntry { type_id, count: 1 });
        }
    }

    /// Get the dominant type (most frequently observed).
    pub fn dominant_type(&self) -> Option<TypeProfileEntry> {
        self.entries.iter().max_by_key(|e| e.count).copied()
    }

    /// Whether this site is monomorphic (single type >95%).
    pub fn is_monomorphic(&self) -> bool {
        if self.total < 10 {
            return false;
        }
        self.dominant_type()
            .map_or(false, |d| d.count as f64 / self.total as f64 > 0.95)
    }

    /// Whether this site is polymorphic (2-4 types observed).
    pub fn is_polymorphic(&self) -> bool {
        let significant: usize = self
            .entries
            .iter()
            .filter(|e| e.count as f64 / self.total.max(1) as f64 > 0.01)
            .count();
        significant >= 2 && significant <= 4
    }

    /// Whether this site is megamorphic (5+ types).
    pub fn is_megamorphic(&self) -> bool {
        let significant: usize = self
            .entries
            .iter()
            .filter(|e| e.count as f64 / self.total.max(1) as f64 > 0.01)
            .count();
        significant >= 5
    }

    /// Total observation count.
    #[inline]
    pub fn total(&self) -> u64 {
        self.total
    }

    /// Get all entries.
    #[inline]
    pub fn entries(&self) -> &[TypeProfileEntry] {
        &self.entries
    }

    /// Merge another profile into this one.
    pub fn merge(&mut self, other: &TypeProfile) {
        for entry in &other.entries {
            if let Some(existing) = self.entries.iter_mut().find(|e| e.type_id == entry.type_id) {
                existing.count = existing.count.saturating_add(entry.count);
            } else {
                self.entries.push(*entry);
            }
        }
        self.total = self.total.saturating_add(other.total);
    }
}

impl Default for TypeProfile {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// Call Profile
// =============================================================================

/// Call target profile for a single call site.
///
/// Tracks which functions are called and how frequently.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CallProfile {
    /// Call targets ordered by frequency (descending).
    targets: Vec<CallTarget>,
    /// Total call count.
    total: u64,
}

/// A single call target entry.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CallTarget {
    /// Unique identifier for the callee.
    pub target_id: u32,
    /// Number of calls to this target.
    pub count: u64,
}

impl CallProfile {
    /// Create an empty call profile.
    #[inline]
    pub fn new() -> Self {
        Self {
            targets: Vec::new(),
            total: 0,
        }
    }

    /// Record a call to a target.
    pub fn record(&mut self, target_id: u32) {
        self.total += 1;
        if let Some(target) = self.targets.iter_mut().find(|t| t.target_id == target_id) {
            target.count += 1;
        } else {
            self.targets.push(CallTarget {
                target_id,
                count: 1,
            });
        }
        // Keep sorted by frequency for fast lookup
        self.targets.sort_unstable_by(|a, b| b.count.cmp(&a.count));
    }

    /// Get the most frequent call target.
    pub fn primary_target(&self) -> Option<CallTarget> {
        self.targets.first().copied()
    }

    /// Whether this call site is monomorphic (single target >95%).
    pub fn is_monomorphic(&self) -> bool {
        if self.total < 10 {
            return false;
        }
        self.primary_target()
            .map_or(false, |t| t.count as f64 / self.total as f64 > 0.95)
    }

    /// Total call count.
    #[inline]
    pub fn total(&self) -> u64 {
        self.total
    }

    /// Get all targets.
    #[inline]
    pub fn targets(&self) -> &[CallTarget] {
        &self.targets
    }

    /// Number of distinct targets observed.
    #[inline]
    pub fn target_count(&self) -> usize {
        self.targets.len()
    }

    /// Merge another profile into this one.
    pub fn merge(&mut self, other: &CallProfile) {
        for target in &other.targets {
            if let Some(existing) = self
                .targets
                .iter_mut()
                .find(|t| t.target_id == target.target_id)
            {
                existing.count = existing.count.saturating_add(target.count);
            } else {
                self.targets.push(*target);
            }
        }
        self.total = self.total.saturating_add(other.total);
        self.targets.sort_unstable_by(|a, b| b.count.cmp(&a.count));
    }
}

impl Default for CallProfile {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// Atomic Branch Counter (for lock-free runtime collection)
// =============================================================================

/// Lock-free branch counter for runtime profile collection.
///
/// Uses atomic operations to allow concurrent updates from
/// multiple execution threads without locking.
#[derive(Debug)]
pub struct AtomicBranchCounter {
    taken: AtomicU64,
    not_taken: AtomicU64,
}

impl AtomicBranchCounter {
    /// Create a new zero-initialized counter.
    #[inline]
    pub const fn new() -> Self {
        Self {
            taken: AtomicU64::new(0),
            not_taken: AtomicU64::new(0),
        }
    }

    /// Record a taken branch (lock-free).
    #[inline]
    pub fn record_taken(&self) {
        self.taken.fetch_add(1, Ordering::Relaxed);
    }

    /// Record a not-taken branch (lock-free).
    #[inline]
    pub fn record_not_taken(&self) {
        self.not_taken.fetch_add(1, Ordering::Relaxed);
    }

    /// Snapshot to a `BranchProfile`.
    #[inline]
    pub fn snapshot(&self) -> BranchProfile {
        BranchProfile {
            taken: self.taken.load(Ordering::Relaxed),
            not_taken: self.not_taken.load(Ordering::Relaxed),
        }
    }

    /// Reset counters to zero.
    #[inline]
    pub fn reset(&self) {
        self.taken.store(0, Ordering::Relaxed);
        self.not_taken.store(0, Ordering::Relaxed);
    }
}

impl Default for AtomicBranchCounter {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// Profile Data (aggregate per code unit)
// =============================================================================

/// Aggregate profile data for a single code unit (function / module).
///
/// Contains all collected feedback: branch frequencies, type observations,
/// call targets, and execution counts.
#[derive(Debug, Clone)]
pub struct ProfileData {
    /// Unique identifier for the code unit.
    code_id: u32,
    /// Branch profiles keyed by bytecode offset.
    branches: FxHashMap<u32, BranchProfile>,
    /// Type profiles keyed by bytecode offset.
    types: FxHashMap<u32, TypeProfile>,
    /// Call profiles keyed by bytecode offset.
    calls: FxHashMap<u32, CallProfile>,
    /// Total execution count of this code unit.
    execution_count: u64,
    /// Loop iteration counts keyed by loop header bytecode offset.
    loop_counts: FxHashMap<u32, u64>,
}

impl ProfileData {
    /// Create empty profile data for a code unit.
    #[inline]
    pub fn new(code_id: u32) -> Self {
        Self {
            code_id,
            branches: FxHashMap::default(),
            types: FxHashMap::default(),
            calls: FxHashMap::default(),
            execution_count: 0,
            loop_counts: FxHashMap::default(),
        }
    }

    /// Get the code unit identifier.
    #[inline]
    pub fn code_id(&self) -> u32 {
        self.code_id
    }

    /// Get execution count.
    #[inline]
    pub fn execution_count(&self) -> u64 {
        self.execution_count
    }

    /// Increment execution count.
    #[inline]
    pub fn record_execution(&mut self) {
        self.execution_count += 1;
    }

    /// Record a branch outcome.
    pub fn record_branch(&mut self, offset: u32, taken: bool) {
        let profile = self.branches.entry(offset).or_default();
        if taken {
            profile.taken += 1;
        } else {
            profile.not_taken += 1;
        }
    }

    /// Record a type observation.
    pub fn record_type(&mut self, offset: u32, type_id: u8) {
        self.types.entry(offset).or_default().record(type_id);
    }

    /// Record a call target.
    pub fn record_call(&mut self, offset: u32, target_id: u32) {
        self.calls.entry(offset).or_default().record(target_id);
    }

    /// Record a loop iteration.
    pub fn record_loop_iteration(&mut self, header_offset: u32) {
        *self.loop_counts.entry(header_offset).or_insert(0) += 1;
    }

    /// Get branch profile at offset.
    #[inline]
    pub fn branch_at(&self, offset: u32) -> Option<&BranchProfile> {
        self.branches.get(&offset)
    }

    /// Get type profile at offset.
    #[inline]
    pub fn type_at(&self, offset: u32) -> Option<&TypeProfile> {
        self.types.get(&offset)
    }

    /// Get call profile at offset.
    #[inline]
    pub fn call_at(&self, offset: u32) -> Option<&CallProfile> {
        self.calls.get(&offset)
    }

    /// Get loop iteration count for a header.
    #[inline]
    pub fn loop_count(&self, header_offset: u32) -> u64 {
        self.loop_counts.get(&header_offset).copied().unwrap_or(0)
    }

    /// Whether this code unit is considered "hot" (above threshold).
    #[inline]
    pub fn is_hot(&self, threshold: u64) -> bool {
        self.execution_count >= threshold
    }

    /// Number of branch sites profiled.
    #[inline]
    pub fn branch_count(&self) -> usize {
        self.branches.len()
    }

    /// Get all branch offsets.
    pub fn branch_offsets(&self) -> Vec<u32> {
        self.branches.keys().copied().collect()
    }

    /// Number of type sites profiled.
    #[inline]
    pub fn type_count(&self) -> usize {
        self.types.len()
    }

    /// Number of call sites profiled.
    #[inline]
    pub fn call_count(&self) -> usize {
        self.calls.len()
    }

    /// Merge another profile into this one.
    pub fn merge(&mut self, other: &ProfileData) {
        assert_eq!(self.code_id, other.code_id);
        self.execution_count = self.execution_count.saturating_add(other.execution_count);

        for (&offset, profile) in &other.branches {
            self.branches.entry(offset).or_default().merge(profile);
        }
        for (&offset, profile) in &other.types {
            self.types.entry(offset).or_default().merge(profile);
        }
        for (&offset, profile) in &other.calls {
            self.calls.entry(offset).or_default().merge(profile);
        }
        for (&offset, &count) in &other.loop_counts {
            *self.loop_counts.entry(offset).or_insert(0) = self
                .loop_counts
                .get(&offset)
                .unwrap_or(&0)
                .saturating_add(count);
        }
    }

    /// Scale all counts by a factor (for profile aging/decay).
    pub fn scale(&mut self, factor: f64) {
        self.execution_count = (self.execution_count as f64 * factor) as u64;
        for profile in self.branches.values_mut() {
            profile.scale(factor);
        }
        for count in self.loop_counts.values_mut() {
            *count = (*count as f64 * factor) as u64;
        }
    }

    // =========================================================================
    // Serialization
    // =========================================================================

    /// Serialize to compact binary format.
    pub fn serialize(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(256);

        // Header
        buf.extend_from_slice(&PROFILE_MAGIC);
        buf.extend_from_slice(&PROFILE_VERSION.to_le_bytes());
        buf.extend_from_slice(&0u16.to_le_bytes()); // flags
        let section_count = 4u32; // branch, type, call, exec
        buf.extend_from_slice(&section_count.to_le_bytes());
        // Placeholder for checksum (filled at end)
        let checksum_pos = buf.len();
        buf.extend_from_slice(&0u32.to_le_bytes());

        // Code ID
        buf.extend_from_slice(&self.code_id.to_le_bytes());

        // Section 1: Branches
        buf.push(SECTION_BRANCH);
        let count = self.branches.len() as u32;
        buf.extend_from_slice(&count.to_le_bytes());
        for (&offset, profile) in &self.branches {
            buf.extend_from_slice(&offset.to_le_bytes());
            buf.extend_from_slice(&profile.taken.to_le_bytes());
            buf.extend_from_slice(&profile.not_taken.to_le_bytes());
        }

        // Section 2: Types
        buf.push(SECTION_TYPE);
        let count = self.types.len() as u32;
        buf.extend_from_slice(&count.to_le_bytes());
        for (&offset, profile) in &self.types {
            buf.extend_from_slice(&offset.to_le_bytes());
            buf.extend_from_slice(&profile.total.to_le_bytes());
            let entry_count = profile.entries.len() as u16;
            buf.extend_from_slice(&entry_count.to_le_bytes());
            for entry in profile.entries() {
                buf.push(entry.type_id);
                buf.extend_from_slice(&entry.count.to_le_bytes());
            }
        }

        // Section 3: Calls
        buf.push(SECTION_CALL);
        let count = self.calls.len() as u32;
        buf.extend_from_slice(&count.to_le_bytes());
        for (&offset, profile) in &self.calls {
            buf.extend_from_slice(&offset.to_le_bytes());
            buf.extend_from_slice(&profile.total.to_le_bytes());
            let target_count = profile.targets.len() as u16;
            buf.extend_from_slice(&target_count.to_le_bytes());
            for target in profile.targets() {
                buf.extend_from_slice(&target.target_id.to_le_bytes());
                buf.extend_from_slice(&target.count.to_le_bytes());
            }
        }

        // Section 4: Execution counts
        buf.push(SECTION_EXEC_COUNT);
        buf.extend_from_slice(&self.execution_count.to_le_bytes());
        let loop_count = self.loop_counts.len() as u32;
        buf.extend_from_slice(&loop_count.to_le_bytes());
        for (&offset, &count) in &self.loop_counts {
            buf.extend_from_slice(&offset.to_le_bytes());
            buf.extend_from_slice(&count.to_le_bytes());
        }

        // Compute and write checksum (FNV-1a of payload after header)
        let checksum = fnv1a_hash(&buf[16..]);
        buf[checksum_pos..checksum_pos + 4].copy_from_slice(&checksum.to_le_bytes());

        buf
    }

    /// Deserialize from binary format.
    pub fn deserialize(data: &[u8]) -> Result<Self, ProfileError> {
        if data.len() < 20 {
            return Err(ProfileError::TooShort);
        }

        // Validate header
        if &data[0..4] != &PROFILE_MAGIC {
            return Err(ProfileError::BadMagic);
        }
        let version = u16::from_le_bytes([data[4], data[5]]);
        if version != PROFILE_VERSION {
            return Err(ProfileError::UnsupportedVersion(version));
        }
        let _flags = u16::from_le_bytes([data[6], data[7]]);
        let section_count = u32::from_le_bytes([data[8], data[9], data[10], data[11]]);
        let stored_checksum = u32::from_le_bytes([data[12], data[13], data[14], data[15]]);

        // Verify checksum
        let computed = fnv1a_hash(&data[16..]);
        if stored_checksum != computed {
            return Err(ProfileError::ChecksumMismatch);
        }

        let code_id = u32::from_le_bytes([data[16], data[17], data[18], data[19]]);
        let mut profile = ProfileData::new(code_id);
        let mut pos = 20;

        for _ in 0..section_count {
            if pos >= data.len() {
                return Err(ProfileError::TooShort);
            }
            let section_type = data[pos];
            pos += 1;

            match section_type {
                SECTION_BRANCH => {
                    let count = read_u32(data, &mut pos)?;
                    for _ in 0..count {
                        let offset = read_u32(data, &mut pos)?;
                        let taken = read_u64(data, &mut pos)?;
                        let not_taken = read_u64(data, &mut pos)?;
                        profile
                            .branches
                            .insert(offset, BranchProfile::with_counts(taken, not_taken));
                    }
                }
                SECTION_TYPE => {
                    let count = read_u32(data, &mut pos)?;
                    for _ in 0..count {
                        let offset = read_u32(data, &mut pos)?;
                        let total = read_u64(data, &mut pos)?;
                        let entry_count = read_u16(data, &mut pos)?;
                        let mut tp = TypeProfile::new();
                        tp.total = total;
                        for _ in 0..entry_count {
                            if pos >= data.len() {
                                return Err(ProfileError::TooShort);
                            }
                            let type_id = data[pos];
                            pos += 1;
                            let count = read_u64(data, &mut pos)?;
                            tp.entries.push(TypeProfileEntry { type_id, count });
                        }
                        profile.types.insert(offset, tp);
                    }
                }
                SECTION_CALL => {
                    let count = read_u32(data, &mut pos)?;
                    for _ in 0..count {
                        let offset = read_u32(data, &mut pos)?;
                        let total = read_u64(data, &mut pos)?;
                        let target_count = read_u16(data, &mut pos)?;
                        let mut cp = CallProfile::new();
                        cp.total = total;
                        for _ in 0..target_count {
                            let target_id = read_u32(data, &mut pos)?;
                            let count = read_u64(data, &mut pos)?;
                            cp.targets.push(CallTarget { target_id, count });
                        }
                        profile.calls.insert(offset, cp);
                    }
                }
                SECTION_EXEC_COUNT => {
                    profile.execution_count = read_u64(data, &mut pos)?;
                    let loop_count = read_u32(data, &mut pos)?;
                    for _ in 0..loop_count {
                        let offset = read_u32(data, &mut pos)?;
                        let count = read_u64(data, &mut pos)?;
                        profile.loop_counts.insert(offset, count);
                    }
                }
                _ => return Err(ProfileError::UnknownSection(section_type)),
            }
        }

        Ok(profile)
    }
}

// =============================================================================
// Errors
// =============================================================================

/// Errors during profile deserialization.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProfileError {
    /// Data is too short.
    TooShort,
    /// Invalid magic bytes.
    BadMagic,
    /// Unsupported format version.
    UnsupportedVersion(u16),
    /// Checksum mismatch (corrupted data).
    ChecksumMismatch,
    /// Unknown section type.
    UnknownSection(u8),
}

impl std::fmt::Display for ProfileError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::TooShort => write!(f, "profile data too short"),
            Self::BadMagic => write!(f, "invalid profile magic bytes"),
            Self::UnsupportedVersion(v) => write!(f, "unsupported profile version: {v}"),
            Self::ChecksumMismatch => write!(f, "profile checksum mismatch"),
            Self::UnknownSection(s) => write!(f, "unknown profile section: {s}"),
        }
    }
}

impl std::error::Error for ProfileError {}

// =============================================================================
// Helpers
// =============================================================================

/// FNV-1a hash (32-bit) for checksum.
fn fnv1a_hash(data: &[u8]) -> u32 {
    let mut hash: u32 = 0x811c_9dc5;
    for &byte in data {
        hash ^= byte as u32;
        hash = hash.wrapping_mul(0x0100_0193);
    }
    hash
}

fn read_u16(data: &[u8], pos: &mut usize) -> Result<u16, ProfileError> {
    if *pos + 2 > data.len() {
        return Err(ProfileError::TooShort);
    }
    let val = u16::from_le_bytes([data[*pos], data[*pos + 1]]);
    *pos += 2;
    Ok(val)
}

fn read_u32(data: &[u8], pos: &mut usize) -> Result<u32, ProfileError> {
    if *pos + 4 > data.len() {
        return Err(ProfileError::TooShort);
    }
    let val = u32::from_le_bytes([data[*pos], data[*pos + 1], data[*pos + 2], data[*pos + 3]]);
    *pos += 4;
    Ok(val)
}

fn read_u64(data: &[u8], pos: &mut usize) -> Result<u64, ProfileError> {
    if *pos + 8 > data.len() {
        return Err(ProfileError::TooShort);
    }
    let val = u64::from_le_bytes([
        data[*pos],
        data[*pos + 1],
        data[*pos + 2],
        data[*pos + 3],
        data[*pos + 4],
        data[*pos + 5],
        data[*pos + 6],
        data[*pos + 7],
    ]);
    *pos += 8;
    Ok(val)
}
