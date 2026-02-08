//! JIT Runtime Integration Module
//!
//! Provides the runtime infrastructure for executing JIT-compiled code,
//! including code caching, background compilation, and entry/exit stubs.
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────────┐
//! │                        JIT Runtime                                  │
//! ├─────────────────────────────────────────────────────────────────────┤
//! │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐              │
//! │  │  CodeCache   │  │ CompilerTask │  │  EntryStub   │              │
//! │  │              │  │              │  │              │              │
//! │  │ Stores and   │  │ Background   │  │ Transitions  │              │
//! │  │ indexes all  │  │ compilation  │  │ between      │              │
//! │  │ compiled     │  │ of hot       │  │ interpreter  │              │
//! │  │ functions    │  │ functions    │  │ and JIT      │              │
//! │  └──────────────┘  └──────────────┘  └──────────────┘              │
//! └─────────────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Usage
//!
//! ```ignore
//! use prism_jit::runtime::{CodeCache, CompilerTask, RuntimeConfig};
//!
//! // Create a code cache
//! let config = RuntimeConfig::default();
//! let cache = CodeCache::new(config);
//!
//! // Check for compiled code
//! if let Some(entry) = cache.lookup(code_id) {
//!     // Execute via entry stub
//!     entry.call(args);
//! }
//! ```

pub mod code_cache;
pub mod compiler_thread;
pub mod entry_stub;
pub mod profile_collector;
pub mod profile_data;

#[cfg(test)]
mod profile_data_tests;

pub use code_cache::{CodeCache, CompiledEntry};
pub use compiler_thread::{CompilationRequest, CompilerThread};
pub use entry_stub::{EntryStub, ExitReason};
pub use profile_data::{
    AtomicBranchCounter, BranchProfile, CallProfile, CallTarget, ProfileData, ProfileError,
    TypeProfile, TypeProfileEntry,
};

// =============================================================================
// Runtime Configuration
// =============================================================================

/// Configuration for the JIT runtime.
#[derive(Debug, Clone)]
pub struct RuntimeConfig {
    /// Maximum size of code cache in bytes.
    pub max_code_size: usize,
    /// Number of compiler threads (0 = synchronous compilation).
    pub compiler_threads: usize,
    /// Compilation tier-up threshold (execution count).
    pub tier_up_threshold: u32,
    /// Enable OSR for hot loops.
    pub enable_osr: bool,
    /// Enable speculative optimizations.
    pub enable_speculation: bool,
}

impl Default for RuntimeConfig {
    fn default() -> Self {
        Self {
            max_code_size: 64 * 1024 * 1024, // 64MB
            compiler_threads: 1,
            tier_up_threshold: 1000,
            enable_osr: true,
            enable_speculation: true,
        }
    }
}

impl RuntimeConfig {
    /// Create a config for testing (smaller limits, synchronous compilation).
    pub fn for_testing() -> Self {
        Self {
            max_code_size: 1024 * 1024, // 1MB
            compiler_threads: 0,
            tier_up_threshold: 10,
            enable_osr: false,
            enable_speculation: false,
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
    fn test_default_config() {
        let config = RuntimeConfig::default();
        assert_eq!(config.max_code_size, 64 * 1024 * 1024);
        assert_eq!(config.compiler_threads, 1);
        assert!(config.enable_osr);
    }

    #[test]
    fn test_testing_config() {
        let config = RuntimeConfig::for_testing();
        assert_eq!(config.max_code_size, 1024 * 1024);
        assert_eq!(config.compiler_threads, 0);
        assert!(!config.enable_osr);
    }
}
