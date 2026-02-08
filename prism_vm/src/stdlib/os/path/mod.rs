//! os.path module - Path manipulation operations.
//!
//! High-performance implementation of Python's `os.path` module providing:
//! - Path joining and normalization (`join`, `abspath`, `normpath`, `realpath`)
//! - Path splitting (`basename`, `dirname`, `splitext`, `split`, `splitdrive`)
//! - Path queries (`exists`, `isfile`, `isdir`, `islink`, `isabs`, `lexists`, `ismount`)
//! - Path comparison (`commonpath`, `commonprefix`, `relpath`, `samefile`)
//! - Timestamp access (`getmtime`, `getatime`, `getctime`, `lgetmtime`)

mod compare;
mod join;
mod normalize;
mod query;
mod split;
mod time_access;

pub use compare::*;
pub use join::*;
pub use normalize::*;
pub use query::*;
pub use split::*;
pub use time_access::*;

use super::constants::SEP;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_exports() {
        // Ensure all submodules export correctly
        let _ = exists(".");
        let _ = join("a", "b");
        let _ = basename("/foo/bar");
        let _ = abspath(".");
    }
}
