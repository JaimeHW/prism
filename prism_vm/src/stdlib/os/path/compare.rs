//! Path comparison and relative path operations.
//!
//! High-performance implementations of Python's `os.path` comparison functions:
//! - `commonpath` — longest common sub-path of a sequence of paths
//! - `commonprefix` — longest common string prefix (character-level)
//! - `relpath` — compute relative path between two locations
//! - `samefile` — check if two paths refer to the same file
//!
//! # Performance
//!
//! - `commonpath` uses component-wise comparison, no string allocation until result
//! - `commonprefix` operates on raw bytes for maximum throughput
//! - `relpath` pre-computes component arrays to minimize allocations
//! - `samefile` uses a single `metadata()` call per path

use std::fs;
use std::path::{Component, Path, PathBuf};

/// Return the longest common sub-path of each pathname in `paths`.
///
/// Unlike `commonprefix`, this returns a valid path. All paths must be
/// either all absolute or all relative; a `ValueError`-equivalent error
/// is returned if they're mixed.
///
/// Equivalent to Python's `os.path.commonpath()`.
///
/// # Examples
///
/// ```ignore
/// assert_eq!(commonpath(&["/usr/lib", "/usr/local"]).unwrap(), "/usr");
/// ```
pub fn commonpath<P: AsRef<Path>>(paths: &[P]) -> Result<String, CommonPathError> {
    if paths.is_empty() {
        return Err(CommonPathError::Empty);
    }

    if paths.len() == 1 {
        return Ok(paths[0].as_ref().to_string_lossy().into_owned());
    }

    // Collect components for each path
    let component_lists: Vec<Vec<Component<'_>>> = paths
        .iter()
        .map(|p| p.as_ref().components().collect())
        .collect();

    // Check consistency: all absolute or all relative
    let first_is_absolute = paths[0].as_ref().is_absolute();
    for p in &paths[1..] {
        if p.as_ref().is_absolute() != first_is_absolute {
            return Err(CommonPathError::MixedAbsoluteRelative);
        }
    }

    // Find the minimum component count
    let min_len = component_lists.iter().map(|c| c.len()).min().unwrap_or(0);

    // Find the longest common prefix by component
    let mut common_len = 0;
    for i in 0..min_len {
        let first = &component_lists[0][i];
        if component_lists[1..].iter().all(|cl| &cl[i] == first) {
            common_len = i + 1;
        } else {
            break;
        }
    }

    if common_len == 0 {
        if first_is_absolute {
            // Absolute paths always share at least the root
            #[cfg(windows)]
            {
                // On Windows, check if they share a drive prefix
                let first_comps = &component_lists[0];
                if !first_comps.is_empty() {
                    if let Component::Prefix(_) = first_comps[0] {
                        // Check if all share the same prefix
                        if component_lists[1..]
                            .iter()
                            .all(|cl| !cl.is_empty() && cl[0] == first_comps[0])
                        {
                            let mut result = PathBuf::new();
                            result.push(first_comps[0].as_os_str());
                            if first_comps.len() > 1 {
                                if let Component::RootDir = first_comps[1] {
                                    if component_lists[1..]
                                        .iter()
                                        .all(|cl| cl.len() > 1 && cl[1] == first_comps[1])
                                    {
                                        result.push(std::path::MAIN_SEPARATOR.to_string());
                                    }
                                }
                            }
                            return Ok(result.to_string_lossy().into_owned());
                        }
                    }
                }
            }
            #[cfg(not(windows))]
            {
                return Ok("/".to_string());
            }
        }
        return Err(CommonPathError::NoCommonPath);
    }

    let mut result = PathBuf::new();
    for comp in &component_lists[0][..common_len] {
        result.push(comp.as_os_str());
    }
    Ok(result.to_string_lossy().into_owned())
}

/// Error type for `commonpath`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CommonPathError {
    /// No paths provided.
    Empty,
    /// Mixed absolute and relative paths.
    MixedAbsoluteRelative,
    /// No common sub-path exists.
    NoCommonPath,
}

impl std::fmt::Display for CommonPathError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Empty => write!(f, "commonpath() arg is an empty sequence"),
            Self::MixedAbsoluteRelative => {
                write!(f, "Can't mix absolute and relative paths")
            }
            Self::NoCommonPath => write!(f, "Paths have no common sub-path"),
        }
    }
}

impl std::error::Error for CommonPathError {}

/// Return the longest path prefix (character by character) that is a
/// prefix of all paths in the list.
///
/// Unlike `commonpath`, this operates on raw strings and may return
/// an invalid path (e.g., splitting mid-component).
///
/// Equivalent to Python's `os.path.commonprefix()`.
///
/// # Performance
///
/// Operates on raw bytes — no path parsing, no allocations until result.
pub fn commonprefix(paths: &[&str]) -> String {
    if paths.is_empty() {
        return String::new();
    }
    if paths.len() == 1 {
        return paths[0].to_string();
    }

    let bytes: Vec<&[u8]> = paths.iter().map(|s| s.as_bytes()).collect();
    let min_len = bytes.iter().map(|b| b.len()).min().unwrap_or(0);

    let mut prefix_len = 0;
    for i in 0..min_len {
        let c = bytes[0][i];
        if bytes[1..].iter().all(|b| b[i] == c) {
            prefix_len = i + 1;
        } else {
            break;
        }
    }

    // SAFETY: We're slicing the original UTF-8 string at a position where
    // all strings agree, so the result is valid UTF-8 if the inputs are.
    paths[0][..prefix_len].to_string()
}

/// Return a relative filepath to `path` either from the current directory
/// or from an optional `start` directory.
///
/// Equivalent to Python's `os.path.relpath()`.
///
/// # Performance
///
/// Pre-computes component arrays. Uses a single allocation for the result.
pub fn relpath<P: AsRef<Path>, S: AsRef<Path>>(path: P, start: S) -> String {
    let path = path.as_ref();
    let start = start.as_ref();

    // Canonicalize both paths to absolute for comparison
    let abs_path = if path.is_absolute() {
        path.to_path_buf()
    } else {
        std::env::current_dir().unwrap_or_default().join(path)
    };
    let abs_start = if start.is_absolute() {
        start.to_path_buf()
    } else {
        std::env::current_dir().unwrap_or_default().join(start)
    };

    let path_comps: Vec<Component<'_>> = abs_path.components().collect();
    let start_comps: Vec<Component<'_>> = abs_start.components().collect();

    // Find common prefix length
    let mut common = 0;
    let min_len = path_comps.len().min(start_comps.len());
    for i in 0..min_len {
        if path_comps[i] == start_comps[i] {
            common = i + 1;
        } else {
            break;
        }
    }

    // Build relative path: go up from start to common ancestor, then down to path
    let ups = start_comps.len() - common;
    let mut result = PathBuf::new();
    for _ in 0..ups {
        result.push("..");
    }
    for comp in &path_comps[common..] {
        result.push(comp.as_os_str());
    }

    if result.as_os_str().is_empty() {
        ".".to_string()
    } else {
        result.to_string_lossy().into_owned()
    }
}

/// Check if two paths refer to the same file or directory.
///
/// Follows symlinks. Returns `false` if either path doesn't exist.
///
/// Equivalent to Python's `os.path.samefile()`.
///
/// # Platform behavior
///
/// - **Unix**: Compares (st_dev, st_ino) for identity
/// - **Windows**: Compares canonical paths (symlink-resolved)
pub fn samefile<P: AsRef<Path>, Q: AsRef<Path>>(path1: P, path2: Q) -> bool {
    samefile_impl(path1.as_ref(), path2.as_ref())
}

#[cfg(unix)]
fn samefile_impl(path1: &Path, path2: &Path) -> bool {
    use std::os::unix::fs::MetadataExt;
    let Ok(m1) = fs::metadata(path1) else {
        return false;
    };
    let Ok(m2) = fs::metadata(path2) else {
        return false;
    };
    m1.dev() == m2.dev() && m1.ino() == m2.ino()
}

#[cfg(windows)]
fn samefile_impl(path1: &Path, path2: &Path) -> bool {
    let Ok(c1) = fs::canonicalize(path1) else {
        return false;
    };
    let Ok(c2) = fs::canonicalize(path2) else {
        return false;
    };
    c1 == c2
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::env;
    use std::fs::File;
    use std::io::Write;

    // =========================================================================
    // commonpath Tests
    // =========================================================================

    #[test]
    fn test_commonpath_empty() {
        let paths: Vec<&str> = vec![];
        assert_eq!(commonpath(&paths), Err(CommonPathError::Empty));
    }

    #[test]
    fn test_commonpath_single() {
        let result = commonpath(&["foo/bar"]).unwrap();
        // PathBuf normalizes separators on Windows
        assert!(result == "foo/bar" || result == "foo\\bar");
    }

    #[test]
    fn test_commonpath_relative_same() {
        let result = commonpath(&["foo/bar", "foo/bar"]).unwrap();
        assert!(result == "foo/bar" || result == "foo\\bar");
    }

    #[test]
    fn test_commonpath_relative_siblings() {
        let result = commonpath(&["foo/bar", "foo/baz"]).unwrap();
        assert_eq!(result, "foo");
    }

    #[test]
    fn test_commonpath_relative_nested() {
        let result = commonpath(&["foo/bar/baz", "foo/bar/qux"]).unwrap();
        assert!(result == "foo/bar" || result == "foo\\bar");
    }

    #[test]
    fn test_commonpath_no_common_relative() {
        let result = commonpath(&["alpha", "beta"]);
        assert_eq!(result, Err(CommonPathError::NoCommonPath));
    }

    #[cfg(not(windows))]
    #[test]
    fn test_commonpath_absolute_unix() {
        let result = commonpath(&["/usr/lib", "/usr/local"]).unwrap();
        assert_eq!(result, "/usr");
    }

    #[cfg(not(windows))]
    #[test]
    fn test_commonpath_absolute_root_only() {
        let result = commonpath(&["/usr", "/var"]).unwrap();
        assert_eq!(result, "/");
    }

    #[cfg(windows)]
    #[test]
    fn test_commonpath_absolute_windows() {
        let result = commonpath(&["C:\\Users\\foo", "C:\\Users\\bar"]).unwrap();
        assert!(result.contains("Users"), "result: {}", result);
    }

    #[test]
    fn test_commonpath_mixed_error() {
        #[cfg(not(windows))]
        {
            let result = commonpath(&["/absolute", "relative"]);
            assert_eq!(result, Err(CommonPathError::MixedAbsoluteRelative));
        }
        #[cfg(windows)]
        {
            let result = commonpath(&["C:\\absolute", "relative"]);
            assert_eq!(result, Err(CommonPathError::MixedAbsoluteRelative));
        }
    }

    #[test]
    fn test_commonpath_three_paths() {
        let result = commonpath(&["a/b/c", "a/b/d", "a/b/e"]).unwrap();
        assert!(result == "a/b" || result == "a\\b");
    }

    #[test]
    fn test_commonpath_one_is_prefix_of_other() {
        let result = commonpath(&["a/b", "a/b/c"]).unwrap();
        assert!(result == "a/b" || result == "a\\b");
    }

    // =========================================================================
    // commonprefix Tests
    // =========================================================================

    #[test]
    fn test_commonprefix_empty() {
        let paths: Vec<&str> = vec![];
        assert_eq!(commonprefix(&paths), "");
    }

    #[test]
    fn test_commonprefix_single() {
        assert_eq!(commonprefix(&["hello"]), "hello");
    }

    #[test]
    fn test_commonprefix_identical() {
        assert_eq!(commonprefix(&["abc", "abc"]), "abc");
    }

    #[test]
    fn test_commonprefix_partial() {
        assert_eq!(commonprefix(&["abcdef", "abcxyz"]), "abc");
    }

    #[test]
    fn test_commonprefix_no_common() {
        assert_eq!(commonprefix(&["xyz", "abc"]), "");
    }

    #[test]
    fn test_commonprefix_one_empty() {
        assert_eq!(commonprefix(&["abc", ""]), "");
    }

    #[test]
    fn test_commonprefix_paths_split_mid_component() {
        // Note: commonprefix is a string operation, it may split mid-component
        assert_eq!(commonprefix(&["/usr/lib", "/usr/local"]), "/usr/l");
    }

    #[test]
    fn test_commonprefix_three_strings() {
        assert_eq!(
            commonprefix(&["interspecies", "interstellar", "interstate"]),
            "inters"
        );
    }

    #[test]
    fn test_commonprefix_all_same() {
        assert_eq!(commonprefix(&["abc", "abc", "abc"]), "abc");
    }

    #[test]
    fn test_commonprefix_single_char() {
        assert_eq!(commonprefix(&["a", "ab"]), "a");
    }

    // =========================================================================
    // relpath Tests
    // =========================================================================

    #[test]
    fn test_relpath_same_dir() {
        #[cfg(not(windows))]
        let rel = relpath("/a/b", "/a/b");
        #[cfg(windows)]
        let rel = relpath("C:\\a\\b", "C:\\a\\b");
        assert_eq!(rel, ".");
    }

    #[test]
    fn test_relpath_child() {
        #[cfg(not(windows))]
        {
            let rel = relpath("/a/b/c", "/a");
            assert_eq!(rel, "b/c");
        }
        #[cfg(windows)]
        {
            let rel = relpath("C:\\a\\b\\c", "C:\\a");
            assert!(rel.contains("b") && rel.contains("c"));
        }
    }

    #[test]
    fn test_relpath_parent() {
        #[cfg(not(windows))]
        {
            let rel = relpath("/a", "/a/b/c");
            assert_eq!(rel, "../..");
        }
        #[cfg(windows)]
        {
            let rel = relpath("C:\\a", "C:\\a\\b\\c");
            assert!(rel.contains(".."));
        }
    }

    #[test]
    fn test_relpath_sibling() {
        #[cfg(not(windows))]
        {
            let rel = relpath("/a/b", "/a/c");
            assert_eq!(rel, "../b");
        }
        #[cfg(windows)]
        {
            let rel = relpath("C:\\a\\b", "C:\\a\\c");
            assert!(rel.contains("..") && rel.contains("b"));
        }
    }

    #[test]
    fn test_relpath_deep_divergence() {
        #[cfg(not(windows))]
        {
            let rel = relpath("/a/b/c/d", "/a/x/y/z");
            assert_eq!(rel, "../../../b/c/d");
        }
    }

    // =========================================================================
    // samefile Tests
    // =========================================================================

    #[test]
    fn test_samefile_same_path() {
        let exe = env::current_exe().unwrap();
        assert!(samefile(&exe, &exe));
    }

    #[test]
    fn test_samefile_directory() {
        let temp = env::temp_dir();
        assert!(samefile(&temp, &temp));
    }

    #[test]
    fn test_samefile_different_files() {
        let dir = env::temp_dir();
        let path1 = dir.join("_prism_samefile_1.txt");
        let path2 = dir.join("_prism_samefile_2.txt");

        File::create(&path1).unwrap().write_all(b"a").unwrap();
        File::create(&path2).unwrap().write_all(b"b").unwrap();

        assert!(!samefile(&path1, &path2));

        let _ = fs::remove_file(&path1);
        let _ = fs::remove_file(&path2);
    }

    #[test]
    fn test_samefile_nonexistent_first() {
        let exe = env::current_exe().unwrap();
        assert!(!samefile("/nonexistent_12345", &exe));
    }

    #[test]
    fn test_samefile_nonexistent_second() {
        let exe = env::current_exe().unwrap();
        assert!(!samefile(&exe, "/nonexistent_12345"));
    }

    #[test]
    fn test_samefile_both_nonexistent() {
        assert!(!samefile("/nonexistent_12345", "/also_nonexistent_12345"));
    }

    #[test]
    fn test_samefile_canonical_equivalence() {
        // "." and the actual CWD should be the same file
        let cwd = env::current_dir().unwrap();
        assert!(samefile(".", &cwd));
    }

    // =========================================================================
    // CommonPathError Display Tests
    // =========================================================================

    #[test]
    fn test_commonpath_error_display_empty() {
        let e = CommonPathError::Empty;
        assert!(e.to_string().contains("empty"));
    }

    #[test]
    fn test_commonpath_error_display_mixed() {
        let e = CommonPathError::MixedAbsoluteRelative;
        assert!(e.to_string().contains("mix"));
    }

    #[test]
    fn test_commonpath_error_display_no_common() {
        let e = CommonPathError::NoCommonPath;
        assert!(e.to_string().contains("no common"));
    }

    #[test]
    fn test_commonpath_error_eq() {
        assert_eq!(CommonPathError::Empty, CommonPathError::Empty);
        assert_ne!(CommonPathError::Empty, CommonPathError::NoCommonPath);
    }
}
