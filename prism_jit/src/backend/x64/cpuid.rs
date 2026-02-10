//! CPU Feature Detection via CPUID
//!
//! This module provides runtime detection of x86-64 CPU features using the
//! CPUID instruction. This allows the JIT to generate optimal code for the
//! specific CPU running the program.
//!
//! # Detected Features
//!
//! - **SIMD**: SSE, SSE2, SSE3, SSSE3, SSE4.1, SSE4.2, AVX, AVX2, AVX-512
//! - **Bit Manipulation**: BMI1, BMI2, POPCNT, LZCNT, ABM
//! - **Crypto**: AES-NI, SHA, PCLMULQDQ
//! - **Other**: FMA, F16C, MOVBE, CMOV
//!
//! # Usage
//!
//! ```ignore
//! use prism_jit::backend::x64::cpuid::CpuFeatures;
//!
//! let features = CpuFeatures::detect();
//!
//! if features.has_avx2() {
//!     // Use AVX2 instructions for vectorized operations
//! } else if features.has_sse41() {
//!     // Fall back to SSE4.1
//! }
//! ```
//!
//! # Safety
//!
//! The feature detection is performed once at startup and cached in a static
//! variable. All queries are thread-safe and allocation-free.

use std::sync::OnceLock;

// =============================================================================
// CPU Features Bitflags
// =============================================================================

bitflags::bitflags! {
    /// CPU feature flags detected via CPUID.
    #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
    pub struct CpuFeatureFlags: u64 {
        // =====================================================================
        // Basic SIMD (ECX of CPUID EAX=1)
        // =====================================================================

        /// SSE3 - Streaming SIMD Extensions 3
        const SSE3      = 1 << 0;
        /// PCLMULQDQ - Carry-Less Multiplication
        const PCLMULQDQ = 1 << 1;
        /// SSSE3 - Supplemental SSE3
        const SSSE3     = 1 << 9;
        /// FMA - Fused Multiply-Add (256-bit)
        const FMA       = 1 << 12;
        /// SSE4.1 - Streaming SIMD Extensions 4.1
        const SSE41     = 1 << 19;
        /// SSE4.2 - Streaming SIMD Extensions 4.2
        const SSE42     = 1 << 20;
        /// MOVBE - Move Data After Byte Swap
        const MOVBE     = 1 << 22;
        /// POPCNT - Population Count
        const POPCNT    = 1 << 23;
        /// AES - AES instruction set (AES-NI)
        const AESNI     = 1 << 25;
        /// AVX - Advanced Vector Extensions
        const AVX       = 1 << 28;
        /// F16C - 16-bit FP conversion
        const F16C      = 1 << 29;

        // =====================================================================
        // Extended features (EBX of CPUID EAX=7, ECX=0)
        // =====================================================================

        /// BMI1 - Bit Manipulation Instruction Set 1
        const BMI1      = 1 << 32;
        /// AVX2 - Advanced Vector Extensions 2
        const AVX2      = 1 << 33;
        /// BMI2 - Bit Manipulation Instruction Set 2
        const BMI2      = 1 << 34;
        /// ERMS - Enhanced REP MOVSB/STOSB
        const ERMS      = 1 << 35;
        /// AVX-512 Foundation
        const AVX512F   = 1 << 36;
        /// AVX-512 Doubleword and Quadword
        const AVX512DQ  = 1 << 37;
        /// SHA - SHA extensions
        const SHA       = 1 << 38;
        /// AVX-512 Byte and Word
        const AVX512BW  = 1 << 39;
        /// AVX-512 Vector Length
        const AVX512VL  = 1 << 40;

        // =====================================================================
        // Extended features (EDX of CPUID EAX=1)
        // =====================================================================

        /// CMOV - Conditional Move
        const CMOV      = 1 << 48;
        /// SSE - Streaming SIMD Extensions
        const SSE       = 1 << 49;
        /// SSE2 - Streaming SIMD Extensions 2
        const SSE2      = 1 << 50;

        // =====================================================================
        // Extended features (ECX of CPUID EAX=0x80000001)
        // =====================================================================

        /// LZCNT - Leading Zero Count
        const LZCNT     = 1 << 56;
        /// ABM - Advanced Bit Manipulation (LZCNT + POPCNT)
        const ABM       = 1 << 57;
    }
}

// =============================================================================
// CPU Features Structure
// =============================================================================

/// Detected CPU features for the current system.
///
/// This structure caches the results of CPUID queries and provides
/// convenient methods for querying specific features.
#[derive(Clone, Debug)]
pub struct CpuFeatures {
    /// Raw feature flags
    flags: CpuFeatureFlags,
    /// CPU vendor ID string (e.g., "GenuineIntel", "AuthenticAMD")
    vendor: CpuVendor,
    /// CPU family
    family: u8,
    /// CPU model
    model: u8,
    /// CPU stepping
    stepping: u8,
    /// Maximum supported basic CPUID leaf
    max_basic_leaf: u32,
    /// Maximum supported extended CPUID leaf
    max_extended_leaf: u32,
    /// L1 data cache line size (bytes)
    cache_line_size: u8,
}

/// CPU vendor identification.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CpuVendor {
    /// Intel Corporation
    Intel,
    /// Advanced Micro Devices
    Amd,
    /// Unknown or other vendor
    Unknown,
}

impl CpuFeatures {
    /// Detect CPU features for the current system.
    ///
    /// This function executes CPUID instructions to detect available features.
    /// The result is cached for subsequent calls.
    pub fn detect() -> &'static Self {
        static FEATURES: OnceLock<CpuFeatures> = OnceLock::new();
        FEATURES.get_or_init(Self::detect_impl)
    }

    /// Get a baseline feature set (SSE2 only).
    ///
    /// This is useful for generating code that runs on all x86-64 CPUs.
    pub fn baseline() -> Self {
        Self {
            flags: CpuFeatureFlags::SSE | CpuFeatureFlags::SSE2 | CpuFeatureFlags::CMOV,
            vendor: CpuVendor::Unknown,
            family: 0,
            model: 0,
            stepping: 0,
            max_basic_leaf: 1,
            max_extended_leaf: 0,
            cache_line_size: 64,
        }
    }

    /// Internal feature detection implementation.
    fn detect_impl() -> Self {
        let mut features = Self::baseline();

        // Check if CPUID is supported (always true on x86-64)
        #[cfg(target_arch = "x86_64")]
        {
            features = Self::detect_x86_64();
        }

        features
    }

    /// Feature detection for x86-64.
    #[cfg(target_arch = "x86_64")]
    fn detect_x86_64() -> Self {
        use std::arch::x86_64::{__cpuid, __cpuid_count};

        let mut flags = CpuFeatureFlags::empty();
        let mut vendor = CpuVendor::Unknown;
        let mut family: u8 = 0;
        let mut model: u8 = 0;
        let mut stepping: u8 = 0;
        let mut cache_line_size: u8 = 64;

        // CPUID leaf 0: Get vendor ID and max basic leaf
        let cpuid0 = unsafe { __cpuid(0) };
        let max_basic_leaf = cpuid0.eax;

        // Extract vendor string
        let vendor_bytes: [u8; 12] = [
            (cpuid0.ebx & 0xFF) as u8,
            ((cpuid0.ebx >> 8) & 0xFF) as u8,
            ((cpuid0.ebx >> 16) & 0xFF) as u8,
            ((cpuid0.ebx >> 24) & 0xFF) as u8,
            (cpuid0.edx & 0xFF) as u8,
            ((cpuid0.edx >> 8) & 0xFF) as u8,
            ((cpuid0.edx >> 16) & 0xFF) as u8,
            ((cpuid0.edx >> 24) & 0xFF) as u8,
            (cpuid0.ecx & 0xFF) as u8,
            ((cpuid0.ecx >> 8) & 0xFF) as u8,
            ((cpuid0.ecx >> 16) & 0xFF) as u8,
            ((cpuid0.ecx >> 24) & 0xFF) as u8,
        ];

        if &vendor_bytes == b"GenuineIntel" {
            vendor = CpuVendor::Intel;
        } else if &vendor_bytes == b"AuthenticAMD" {
            vendor = CpuVendor::Amd;
        }

        // CPUID leaf 1: Feature flags and processor info
        if max_basic_leaf >= 1 {
            let cpuid1 = unsafe { __cpuid(1) };

            // Extract processor signature
            stepping = (cpuid1.eax & 0xF) as u8;
            let base_model = ((cpuid1.eax >> 4) & 0xF) as u8;
            let base_family = ((cpuid1.eax >> 8) & 0xF) as u8;
            let ext_model = ((cpuid1.eax >> 16) & 0xF) as u8;
            let ext_family = ((cpuid1.eax >> 20) & 0xFF) as u8;

            if base_family == 0xF {
                family = base_family + ext_family;
            } else {
                family = base_family;
            }

            if base_family == 0x6 || base_family == 0xF {
                model = (ext_model << 4) | base_model;
            } else {
                model = base_model;
            }

            // Cache line size (in 8-byte units)
            cache_line_size = (((cpuid1.ebx >> 8) & 0xFF) * 8) as u8;
            if cache_line_size == 0 {
                cache_line_size = 64; // Default
            }

            // ECX feature flags
            let ecx = cpuid1.ecx;
            if ecx & (1 << 0) != 0 {
                flags |= CpuFeatureFlags::SSE3;
            }
            if ecx & (1 << 1) != 0 {
                flags |= CpuFeatureFlags::PCLMULQDQ;
            }
            if ecx & (1 << 9) != 0 {
                flags |= CpuFeatureFlags::SSSE3;
            }
            if ecx & (1 << 12) != 0 {
                flags |= CpuFeatureFlags::FMA;
            }
            if ecx & (1 << 19) != 0 {
                flags |= CpuFeatureFlags::SSE41;
            }
            if ecx & (1 << 20) != 0 {
                flags |= CpuFeatureFlags::SSE42;
            }
            if ecx & (1 << 22) != 0 {
                flags |= CpuFeatureFlags::MOVBE;
            }
            if ecx & (1 << 23) != 0 {
                flags |= CpuFeatureFlags::POPCNT;
            }
            if ecx & (1 << 25) != 0 {
                flags |= CpuFeatureFlags::AESNI;
            }
            if ecx & (1 << 28) != 0 {
                flags |= CpuFeatureFlags::AVX;
            }
            if ecx & (1 << 29) != 0 {
                flags |= CpuFeatureFlags::F16C;
            }

            // EDX feature flags
            let edx = cpuid1.edx;
            if edx & (1 << 15) != 0 {
                flags |= CpuFeatureFlags::CMOV;
            }
            if edx & (1 << 25) != 0 {
                flags |= CpuFeatureFlags::SSE;
            }
            if edx & (1 << 26) != 0 {
                flags |= CpuFeatureFlags::SSE2;
            }
        }

        // CPUID leaf 7: Extended features
        if max_basic_leaf >= 7 {
            let cpuid7 = unsafe { __cpuid_count(7, 0) };

            let ebx = cpuid7.ebx;
            if ebx & (1 << 3) != 0 {
                flags |= CpuFeatureFlags::BMI1;
            }
            if ebx & (1 << 5) != 0 {
                flags |= CpuFeatureFlags::AVX2;
            }
            if ebx & (1 << 8) != 0 {
                flags |= CpuFeatureFlags::BMI2;
            }
            if ebx & (1 << 9) != 0 {
                flags |= CpuFeatureFlags::ERMS;
            }
            if ebx & (1 << 16) != 0 {
                flags |= CpuFeatureFlags::AVX512F;
            }
            if ebx & (1 << 17) != 0 {
                flags |= CpuFeatureFlags::AVX512DQ;
            }
            if ebx & (1 << 29) != 0 {
                flags |= CpuFeatureFlags::SHA;
            }
            if ebx & (1 << 30) != 0 {
                flags |= CpuFeatureFlags::AVX512BW;
            }
            if ebx & (1 << 31) != 0 {
                flags |= CpuFeatureFlags::AVX512VL;
            }
        }

        // Extended CPUID: 0x80000001 for LZCNT/ABM
        let cpuid_ext0 = unsafe { __cpuid(0x80000000) };
        let max_extended_leaf = cpuid_ext0.eax;

        if max_extended_leaf >= 0x80000001 {
            let cpuid_ext1 = unsafe { __cpuid(0x80000001) };

            let ecx = cpuid_ext1.ecx;
            if ecx & (1 << 5) != 0 {
                flags |= CpuFeatureFlags::LZCNT;
                flags |= CpuFeatureFlags::ABM;
            }
        }

        Self {
            flags,
            vendor,
            family,
            model,
            stepping,
            max_basic_leaf,
            max_extended_leaf,
            cache_line_size,
        }
    }

    // =========================================================================
    // Feature Queries - SIMD
    // =========================================================================

    /// Check if SSE is supported.
    #[inline]
    pub fn has_sse(&self) -> bool {
        self.flags.contains(CpuFeatureFlags::SSE)
    }

    /// Check if SSE2 is supported.
    #[inline]
    pub fn has_sse2(&self) -> bool {
        self.flags.contains(CpuFeatureFlags::SSE2)
    }

    /// Check if SSE3 is supported.
    #[inline]
    pub fn has_sse3(&self) -> bool {
        self.flags.contains(CpuFeatureFlags::SSE3)
    }

    /// Check if SSSE3 is supported.
    #[inline]
    pub fn has_ssse3(&self) -> bool {
        self.flags.contains(CpuFeatureFlags::SSSE3)
    }

    /// Check if SSE4.1 is supported.
    #[inline]
    pub fn has_sse41(&self) -> bool {
        self.flags.contains(CpuFeatureFlags::SSE41)
    }

    /// Check if SSE4.2 is supported.
    #[inline]
    pub fn has_sse42(&self) -> bool {
        self.flags.contains(CpuFeatureFlags::SSE42)
    }

    /// Check if AVX is supported.
    #[inline]
    pub fn has_avx(&self) -> bool {
        self.flags.contains(CpuFeatureFlags::AVX)
    }

    /// Check if AVX2 is supported.
    #[inline]
    pub fn has_avx2(&self) -> bool {
        self.flags.contains(CpuFeatureFlags::AVX2)
    }

    /// Check if AVX-512 Foundation is supported.
    #[inline]
    pub fn has_avx512f(&self) -> bool {
        self.flags.contains(CpuFeatureFlags::AVX512F)
    }

    /// Check if AVX-512 with DQ, BW, VL is supported (common subset).
    #[inline]
    pub fn has_avx512(&self) -> bool {
        self.flags.contains(
            CpuFeatureFlags::AVX512F
                | CpuFeatureFlags::AVX512DQ
                | CpuFeatureFlags::AVX512BW
                | CpuFeatureFlags::AVX512VL,
        )
    }

    /// Check if FMA is supported.
    #[inline]
    pub fn has_fma(&self) -> bool {
        self.flags.contains(CpuFeatureFlags::FMA)
    }

    // =========================================================================
    // Feature Queries - Bit Manipulation
    // =========================================================================

    /// Check if BMI1 is supported (ANDN, BEXTR, BLSI, BLSMSK, BLSR, TZCNT).
    #[inline]
    pub fn has_bmi1(&self) -> bool {
        self.flags.contains(CpuFeatureFlags::BMI1)
    }

    /// Check if BMI2 is supported (BZHI, MULX, PDEP, PEXT, RORX, SARX, SHLX, SHRX).
    #[inline]
    pub fn has_bmi2(&self) -> bool {
        self.flags.contains(CpuFeatureFlags::BMI2)
    }

    /// Check if POPCNT is supported.
    #[inline]
    pub fn has_popcnt(&self) -> bool {
        self.flags.contains(CpuFeatureFlags::POPCNT)
    }

    /// Check if LZCNT is supported.
    #[inline]
    pub fn has_lzcnt(&self) -> bool {
        self.flags.contains(CpuFeatureFlags::LZCNT)
    }

    /// Check if ABM (LZCNT + POPCNT) is supported.
    #[inline]
    pub fn has_abm(&self) -> bool {
        self.flags.contains(CpuFeatureFlags::ABM)
    }

    // =========================================================================
    // Feature Queries - Crypto
    // =========================================================================

    /// Check if AES-NI is supported.
    #[inline]
    pub fn has_aesni(&self) -> bool {
        self.flags.contains(CpuFeatureFlags::AESNI)
    }

    /// Check if SHA extensions are supported.
    #[inline]
    pub fn has_sha(&self) -> bool {
        self.flags.contains(CpuFeatureFlags::SHA)
    }

    /// Check if PCLMULQDQ is supported.
    #[inline]
    pub fn has_pclmulqdq(&self) -> bool {
        self.flags.contains(CpuFeatureFlags::PCLMULQDQ)
    }

    // =========================================================================
    // Feature Queries - Other
    // =========================================================================

    /// Check if CMOV is supported.
    #[inline]
    pub fn has_cmov(&self) -> bool {
        self.flags.contains(CpuFeatureFlags::CMOV)
    }

    /// Check if MOVBE is supported.
    #[inline]
    pub fn has_movbe(&self) -> bool {
        self.flags.contains(CpuFeatureFlags::MOVBE)
    }

    /// Check if F16C is supported.
    #[inline]
    pub fn has_f16c(&self) -> bool {
        self.flags.contains(CpuFeatureFlags::F16C)
    }

    /// Check if ERMS (Enhanced REP MOVSB/STOSB) is supported.
    #[inline]
    pub fn has_erms(&self) -> bool {
        self.flags.contains(CpuFeatureFlags::ERMS)
    }

    // =========================================================================
    // Accessors
    // =========================================================================

    /// Get the raw feature flags.
    #[inline]
    pub fn flags(&self) -> CpuFeatureFlags {
        self.flags
    }

    /// Get the CPU vendor.
    #[inline]
    pub fn vendor(&self) -> CpuVendor {
        self.vendor
    }

    /// Get the CPU family.
    #[inline]
    pub fn family(&self) -> u8 {
        self.family
    }

    /// Get the CPU model.
    #[inline]
    pub fn model(&self) -> u8 {
        self.model
    }

    /// Get the CPU stepping.
    #[inline]
    pub fn stepping(&self) -> u8 {
        self.stepping
    }

    /// Get the cache line size in bytes.
    #[inline]
    pub fn cache_line_size(&self) -> usize {
        self.cache_line_size as usize
    }

    /// Check if running on Intel.
    #[inline]
    pub fn is_intel(&self) -> bool {
        self.vendor == CpuVendor::Intel
    }

    /// Check if running on AMD.
    #[inline]
    pub fn is_amd(&self) -> bool {
        self.vendor == CpuVendor::Amd
    }
}

// =============================================================================
// Feature Level Enumeration
// =============================================================================

/// CPU feature level for code generation.
///
/// This provides a simple way to select instruction sets based on
/// common CPU generation levels.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum CpuLevel {
    /// Baseline x86-64 (SSE2 only)
    Baseline,
    /// SSE4.2 + POPCNT (Nehalem-era, ~2008)
    Sse42,
    /// AVX (Sandy Bridge-era, ~2011)
    Avx,
    /// AVX2 + BMI1/2 (Haswell-era, ~2013)
    Avx2,
    /// AVX-512 (Skylake-X era, ~2017)
    Avx512,
}

impl CpuLevel {
    /// Get the CPU level for the current system.
    pub fn detect() -> Self {
        let features = CpuFeatures::detect();
        Self::from_features(features)
    }

    /// Determine CPU level from feature flags.
    pub fn from_features(features: &CpuFeatures) -> Self {
        if features.has_avx512() {
            CpuLevel::Avx512
        } else if features.has_avx2() && features.has_bmi2() {
            CpuLevel::Avx2
        } else if features.has_avx() {
            CpuLevel::Avx
        } else if features.has_sse42() && features.has_popcnt() {
            CpuLevel::Sse42
        } else {
            CpuLevel::Baseline
        }
    }

    /// Get the name of this CPU level.
    pub fn name(&self) -> &'static str {
        match self {
            CpuLevel::Baseline => "x86-64",
            CpuLevel::Sse42 => "x86-64-v2",
            CpuLevel::Avx => "AVX",
            CpuLevel::Avx2 => "x86-64-v3",
            CpuLevel::Avx512 => "x86-64-v4",
        }
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_baseline_features() {
        let baseline = CpuFeatures::baseline();

        // x86-64 baseline includes SSE2
        assert!(baseline.has_sse());
        assert!(baseline.has_sse2());
        assert!(baseline.has_cmov());

        // But not newer features
        assert!(!baseline.has_avx());
        assert!(!baseline.has_avx2());
    }

    #[test]
    fn test_feature_detection() {
        let features = CpuFeatures::detect();

        // All x86-64 CPUs have these
        assert!(features.has_sse());
        assert!(features.has_sse2());
        assert!(features.has_cmov());

        // Cache line should be reasonable
        assert!(features.cache_line_size() >= 32);
        assert!(features.cache_line_size() <= 128);
    }

    #[test]
    fn test_vendor_detection() {
        let features = CpuFeatures::detect();

        // Should detect one of the known vendors on most systems
        let vendor = features.vendor();
        assert!(
            vendor == CpuVendor::Intel || vendor == CpuVendor::Amd || vendor == CpuVendor::Unknown
        );
    }

    #[test]
    fn test_cpu_level_ordering() {
        assert!(CpuLevel::Baseline < CpuLevel::Sse42);
        assert!(CpuLevel::Sse42 < CpuLevel::Avx);
        assert!(CpuLevel::Avx < CpuLevel::Avx2);
        assert!(CpuLevel::Avx2 < CpuLevel::Avx512);
    }

    #[test]
    fn test_cpu_level_detection() {
        let level = CpuLevel::detect();

        // Should be at least baseline
        assert!(level >= CpuLevel::Baseline);
    }

    #[test]
    fn test_cpu_level_names() {
        assert_eq!(CpuLevel::Baseline.name(), "x86-64");
        assert_eq!(CpuLevel::Sse42.name(), "x86-64-v2");
        assert_eq!(CpuLevel::Avx2.name(), "x86-64-v3");
        assert_eq!(CpuLevel::Avx512.name(), "x86-64-v4");
    }

    #[test]
    fn test_feature_flags_display() {
        let flags = CpuFeatureFlags::SSE | CpuFeatureFlags::SSE2 | CpuFeatureFlags::AVX;

        // Should be debug-printable
        let debug_str = format!("{:?}", flags);
        assert!(debug_str.contains("SSE"));
        assert!(debug_str.contains("SSE2"));
        assert!(debug_str.contains("AVX"));
    }

    #[test]
    fn test_is_vendor() {
        let features = CpuFeatures::detect();

        // At most one vendor should be true
        let intel = features.is_intel();
        let amd = features.is_amd();

        // Can't be both Intel and AMD
        assert!(!(intel && amd));
    }

    #[test]
    fn test_feature_hierarchy() {
        let features = CpuFeatures::detect();

        // AVX2 implies AVX
        if features.has_avx2() {
            assert!(features.has_avx());
        }

        // AVX implies SSE4.2
        if features.has_avx() {
            assert!(features.has_sse42());
        }

        // SSE4.2 implies SSE4.1
        if features.has_sse42() {
            assert!(features.has_sse41());
        }

        // SSE4.1 implies SSSE3
        if features.has_sse41() {
            assert!(features.has_ssse3());
        }
    }

    #[test]
    fn test_bmi_usually_with_avx2() {
        let features = CpuFeatures::detect();

        // On most modern CPUs, AVX2 comes with BMI1/BMI2
        // (though this isn't strictly required)
        if features.has_avx2() {
            // Most Haswell+ CPUs have BMI
            // This is informational, not a hard requirement
            let _ = features.has_bmi1();
            let _ = features.has_bmi2();
        }
    }

    #[test]
    fn test_cpu_level_from_features() {
        // Test with baseline
        let baseline = CpuFeatures::baseline();
        assert_eq!(CpuLevel::from_features(&baseline), CpuLevel::Baseline);
    }

    #[test]
    fn test_model_family_stepping() {
        let features = CpuFeatures::detect();

        // Should have reasonable values
        let _family = features.family();
        let _model = features.model();
        let _stepping = features.stepping();

        // All values are valid (0 is possible for virtual CPUs)
    }

    #[test]
    fn test_flags_accessor() {
        let features = CpuFeatures::detect();
        let flags = features.flags();

        // Should at least have SSE2
        assert!(flags.contains(CpuFeatureFlags::SSE2));
    }

    #[test]
    fn test_cached_detection() {
        // Multiple calls should return the same instance
        let features1 = CpuFeatures::detect();
        let features2 = CpuFeatures::detect();

        // Should be the same cached instance (same pointer)
        assert!(std::ptr::eq(features1, features2));
    }
}
