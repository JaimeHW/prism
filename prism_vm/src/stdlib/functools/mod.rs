//! Python `functools` module implementation.
//!
//! Higher-order functions and operations on callable objects.
//!
//! # Components
//!
//! | Function | Description | Module |
//! |----------|-------------|--------|
//! | `reduce` | Cumulative reduction | `reduce.rs` |
//! | `partial` | Partial function application | `partial.rs` |
//! | `lru_cache` | LRU memoization cache | `lru_cache.rs` |
//! | `cmp_to_key` | Comparison → key function adapter | `cmp.rs` |
//! | `total_ordering` | Derive rich comparisons | `cmp.rs` |
//! | `wraps` / `update_wrapper` | Copy function metadata | `wraps.rs` |
//! | `cached_property` | Lazy-computed cached descriptor | `cached_property.rs` |
//!
//! # Performance Highlights
//!
//! - **`reduce`**: Single-pass O(n), zero intermediate allocations
//! - **`partial`**: SmallVec<[Value; 8]> inline storage for ≤8 frozen args
//! - **`lru_cache`**: Arena-backed intrusive linked list for O(1) operations
//! - **`cmp_to_key`**: Zero-cost Ord wrapper via generics
//! - **`cached_property`**: O(1) cached access after first computation

pub mod cached_property;
pub mod cmp;
pub mod lru_cache;
pub mod partial;
pub mod reduce;
pub mod wraps;

#[cfg(test)]
mod tests;

use super::{Module, ModuleError, ModuleResult};
use std::sync::Arc;

// Re-export core types
pub use cached_property::CachedProperty;
pub use cmp::{CmpKey, ComparisonBase, TotalOrdering};
pub use lru_cache::{CacheInfo, LruCache};
pub use partial::Partial;
pub use reduce::{ReduceError, accumulate, reduce, reduce_fallible};
pub use wraps::{WRAPPER_ASSIGNMENTS, WRAPPER_UPDATES, WrapperMetadata};

// =============================================================================
// Functools Module
// =============================================================================

/// The functools module implementation.
pub struct FunctoolsModule {
    attrs: Vec<Arc<str>>,
}

impl FunctoolsModule {
    pub fn new() -> Self {
        Self {
            attrs: vec![
                Arc::from("reduce"),
                Arc::from("partial"),
                Arc::from("partialmethod"),
                Arc::from("lru_cache"),
                Arc::from("cache"),
                Arc::from("cached_property"),
                Arc::from("cmp_to_key"),
                Arc::from("total_ordering"),
                Arc::from("update_wrapper"),
                Arc::from("wraps"),
                Arc::from("WRAPPER_ASSIGNMENTS"),
                Arc::from("WRAPPER_UPDATES"),
                Arc::from("singledispatch"),
                Arc::from("singledispatchmethod"),
            ],
        }
    }
}

impl Default for FunctoolsModule {
    fn default() -> Self {
        Self::new()
    }
}

impl Module for FunctoolsModule {
    fn name(&self) -> &str {
        "functools"
    }

    fn get_attr(&self, name: &str) -> ModuleResult {
        match name {
            "reduce"
            | "partial"
            | "partialmethod"
            | "lru_cache"
            | "cache"
            | "cached_property"
            | "cmp_to_key"
            | "total_ordering"
            | "update_wrapper"
            | "wraps"
            | "singledispatch"
            | "singledispatchmethod" => {
                // TODO: Return actual callable objects when the VM function protocol is ready
                Err(ModuleError::AttributeError(format!(
                    "functools.{} is not yet available as a callable object",
                    name
                )))
            }
            "WRAPPER_ASSIGNMENTS" | "WRAPPER_UPDATES" => {
                // These are tuple constants in Python; return as None placeholder
                // TODO: Return actual tuples when the VM tuple type is ready
                Err(ModuleError::AttributeError(format!(
                    "functools.{} is not yet available as a constant",
                    name
                )))
            }
            _ => Err(ModuleError::AttributeError(format!(
                "module 'functools' has no attribute '{}'",
                name
            ))),
        }
    }

    fn dir(&self) -> Vec<Arc<str>> {
        self.attrs.clone()
    }
}

// =============================================================================
// Module Registration
// =============================================================================

/// Create a new functools module instance.
pub fn create_module() -> FunctoolsModule {
    FunctoolsModule::new()
}
