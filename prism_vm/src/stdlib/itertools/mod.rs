//! Python `itertools` module implementation.
//!
//! Provides a comprehensive, high-performance implementation of Python's
//! `itertools` standard library module, split into logical submodules:
//!
//! - [`infinite`] — `count`, `cycle`, `repeat`
//! - [`terminating`] — `chain`, `compress`, `dropwhile`, `takewhile`,
//!   `filterfalse`, `islice`, `starmap`, `zip_longest`
//! - [`combinatoric`] — `product`, `permutations`, `combinations`,
//!   `combinations_with_replacement`
//! - [`grouping`] — `groupby`, `pairwise`, `batched`, `triplewise`
//! - [`recipes`] — `flatten`, `unique_everseen`, `unique_justseen`,
//!   `sliding_window`, `roundrobin`, `accumulate`, `partition`, `quantify`
//!
//! # Performance
//!
//! All iterators are zero-allocation where possible, implement `FusedIterator`,
//! provide accurate `size_hint()`, and use `#[inline]` on hot paths. Combinatoric
//! iterators use `SmallVec` for stack-allocated index arrays.

pub mod combinatoric;
pub mod grouping;
pub mod infinite;
pub mod recipes;
pub mod terminating;

#[cfg(test)]
mod tests;

// Re-export all public types for convenience
pub use combinatoric::{Combinations, CombinationsWithReplacement, Permutations, Product};
pub use grouping::{Batched, GroupBy, Pairwise, Triplewise};
pub use infinite::{Count, Cycle, Repeat};
pub use recipes::{Accumulate, Flatten, RoundRobin, SlidingWindow, UniqueEverseen, UniqueJustseen};
pub use terminating::{
    Chain, Compress, DropWhile, EitherOrBoth, FilterFalse, ISlice, Starmap, ZipLongest,
    chain_from_iterable,
};

use crate::stdlib::{Module, ModuleResult};
use prism_core::Value;
use std::sync::Arc;

/// The `itertools` module implementation.
pub struct ItertoolsModule;

impl ItertoolsModule {
    /// Create a new itertools module.
    #[inline]
    pub fn new() -> Self {
        Self
    }
}

impl Module for ItertoolsModule {
    fn name(&self) -> &str {
        "itertools"
    }

    fn get_attr(&self, name: &str) -> ModuleResult {
        match name {
            // Module-level constants/info
            "__name__" => Ok(Value::none()), // placeholder for module name
            _ => Err(crate::stdlib::ModuleError::AttributeError(format!(
                "module 'itertools' has no attribute '{}'",
                name
            ))),
        }
    }

    fn dir(&self) -> Vec<Arc<str>> {
        vec![
            Arc::from("count"),
            Arc::from("cycle"),
            Arc::from("repeat"),
            Arc::from("accumulate"),
            Arc::from("chain"),
            Arc::from("compress"),
            Arc::from("dropwhile"),
            Arc::from("takewhile"),
            Arc::from("filterfalse"),
            Arc::from("islice"),
            Arc::from("starmap"),
            Arc::from("zip_longest"),
            Arc::from("product"),
            Arc::from("permutations"),
            Arc::from("combinations"),
            Arc::from("combinations_with_replacement"),
            Arc::from("groupby"),
            Arc::from("pairwise"),
            Arc::from("batched"),
        ]
    }
}

#[cfg(test)]
mod mod_tests {
    use super::*;

    #[test]
    fn test_module_name() {
        let m = ItertoolsModule::new();
        assert_eq!(m.name(), "itertools");
    }

    #[test]
    fn test_module_dir() {
        let m = ItertoolsModule::new();
        let attrs = m.dir();
        assert!(attrs.iter().any(|a| a.as_ref() == "count"));
        assert!(attrs.iter().any(|a| a.as_ref() == "chain"));
        assert!(attrs.iter().any(|a| a.as_ref() == "product"));
        assert!(attrs.iter().any(|a| a.as_ref() == "groupby"));
        assert!(attrs.iter().any(|a| a.as_ref() == "pairwise"));
        assert!(attrs.iter().any(|a| a.as_ref() == "batched"));
    }

    #[test]
    fn test_module_unknown_attr() {
        let m = ItertoolsModule::new();
        assert!(m.get_attr("nonexistent").is_err());
    }
}
