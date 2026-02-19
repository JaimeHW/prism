//! Module search path management.
//!
//! Manages sys.path for module importing with efficient
//! path manipulation and caching.

use prism_core::Value;
use prism_core::intern::intern;
use prism_runtime::types::list::ListObject;
use std::path::PathBuf;
use std::sync::Arc;

/// Module search paths container.
///
/// Implements Python's sys.path semantics with efficient
/// path storage and lookup.
#[derive(Debug, Clone)]
pub struct SysPaths {
    /// The path list.
    paths: Vec<Arc<str>>,
}

impl SysPaths {
    /// Create empty paths.
    #[inline]
    pub fn new() -> Self {
        Self { paths: Vec::new() }
    }

    /// Create with initial paths.
    #[inline]
    pub fn with_paths(paths: Vec<String>) -> Self {
        Self {
            paths: paths.into_iter().map(|s| s.into()).collect(),
        }
    }

    /// Create default paths from environment.
    pub fn from_env() -> Self {
        let mut paths = Vec::new();

        // Add current directory (empty string in Python)
        paths.push(Arc::from(""));

        // Add PYTHONPATH entries if set
        if let Ok(pythonpath) = std::env::var("PYTHONPATH") {
            for path in pythonpath.split(path_separator()) {
                if !path.is_empty() {
                    paths.push(Arc::from(path));
                }
            }
        }

        Self { paths }
    }

    /// Get number of paths.
    #[inline]
    pub fn len(&self) -> usize {
        self.paths.len()
    }

    /// Check if empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.paths.is_empty()
    }

    /// Get path by index.
    #[inline]
    pub fn get(&self, index: usize) -> Option<&Arc<str>> {
        self.paths.get(index)
    }

    /// Append a path.
    #[inline]
    pub fn append(&mut self, path: impl Into<Arc<str>>) {
        self.paths.push(path.into());
    }

    /// Insert a path at index.
    #[inline]
    pub fn insert(&mut self, index: usize, path: impl Into<Arc<str>>) {
        if index <= self.paths.len() {
            self.paths.insert(index, path.into());
        }
    }

    /// Remove path at index.
    #[inline]
    pub fn remove(&mut self, index: usize) -> Option<Arc<str>> {
        if index < self.paths.len() {
            Some(self.paths.remove(index))
        } else {
            None
        }
    }

    /// Clear all paths.
    #[inline]
    pub fn clear(&mut self) {
        self.paths.clear();
    }

    /// Check if path exists in list.
    #[inline]
    pub fn contains(&self, path: &str) -> bool {
        self.paths.iter().any(|p| p.as_ref() == path)
    }

    /// Iterate over paths.
    #[inline]
    pub fn iter(&self) -> impl Iterator<Item = &Arc<str>> {
        self.paths.iter()
    }

    /// Get as slice.
    #[inline]
    pub fn as_slice(&self) -> &[Arc<str>] {
        &self.paths
    }

    /// Convert sys.path to a Python list value (`list[str]`).
    pub fn to_value(&self) -> Value {
        let values: Vec<Value> = self
            .paths
            .iter()
            .map(|path| Value::string(intern(path.as_ref())))
            .collect();
        let list = ListObject::from_slice(&values);
        let ptr = Box::into_raw(Box::new(list)) as *const ();
        Value::object_ptr(ptr)
    }

    /// Resolve a module name to a path.
    ///
    /// Searches paths in order for a matching module file.
    pub fn resolve_module(&self, module_name: &str) -> Option<PathBuf> {
        let module_file = format!("{}.py", module_name.replace('.', "/"));
        let package_init = format!("{}/__init__.py", module_name.replace('.', "/"));

        for path in &self.paths {
            let base = if path.is_empty() {
                PathBuf::from(".")
            } else {
                PathBuf::from(path.as_ref())
            };

            // Check for module file
            let module_path = base.join(&module_file);
            if module_path.exists() {
                return Some(module_path);
            }

            // Check for package
            let package_path = base.join(&package_init);
            if package_path.exists() {
                return Some(package_path);
            }
        }

        None
    }
}

impl Default for SysPaths {
    fn default() -> Self {
        Self::new()
    }
}

impl<'a> IntoIterator for &'a SysPaths {
    type Item = &'a Arc<str>;
    type IntoIter = std::slice::Iter<'a, Arc<str>>;

    fn into_iter(self) -> Self::IntoIter {
        self.paths.iter()
    }
}

// =============================================================================
// Platform Path Separator
// =============================================================================

/// Get the platform-specific path separator for PATH variables.
#[inline]
const fn path_separator() -> char {
    #[cfg(target_os = "windows")]
    {
        ';'
    }
    #[cfg(not(target_os = "windows"))]
    {
        ':'
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use prism_core::intern::interned_by_ptr;

    // =========================================================================
    // Construction Tests
    // =========================================================================

    #[test]
    fn test_new_empty() {
        let paths = SysPaths::new();
        assert!(paths.is_empty());
        assert_eq!(paths.len(), 0);
    }

    #[test]
    fn test_with_paths() {
        let paths = SysPaths::with_paths(vec![
            "/usr/lib/python".to_string(),
            "/home/user/lib".to_string(),
        ]);
        assert_eq!(paths.len(), 2);
    }

    #[test]
    fn test_default() {
        let paths = SysPaths::default();
        assert!(paths.is_empty());
    }

    // =========================================================================
    // Append/Insert/Remove Tests
    // =========================================================================

    #[test]
    fn test_append() {
        let mut paths = SysPaths::new();
        paths.append("/path/one");
        paths.append("/path/two");
        assert_eq!(paths.len(), 2);
        assert_eq!(paths.get(0).map(|s| s.as_ref()), Some("/path/one"));
        assert_eq!(paths.get(1).map(|s| s.as_ref()), Some("/path/two"));
    }

    #[test]
    fn test_insert_beginning() {
        let mut paths = SysPaths::with_paths(vec!["/existing".to_string()]);
        paths.insert(0, "/first");
        assert_eq!(paths.get(0).map(|s| s.as_ref()), Some("/first"));
        assert_eq!(paths.get(1).map(|s| s.as_ref()), Some("/existing"));
    }

    #[test]
    fn test_insert_middle() {
        let mut paths = SysPaths::with_paths(vec!["a".to_string(), "c".to_string()]);
        paths.insert(1, "b");
        assert_eq!(paths.len(), 3);
        assert_eq!(paths.get(1).map(|s| s.as_ref()), Some("b"));
    }

    #[test]
    fn test_insert_end() {
        let mut paths = SysPaths::with_paths(vec!["a".to_string()]);
        paths.insert(1, "b");
        assert_eq!(paths.get(1).map(|s| s.as_ref()), Some("b"));
    }

    #[test]
    fn test_insert_out_of_bounds() {
        let mut paths = SysPaths::new();
        paths.insert(100, "ignored");
        // Should not panic, just be a no-op
        assert!(paths.is_empty());
    }

    #[test]
    fn test_remove() {
        let mut paths =
            SysPaths::with_paths(vec!["a".to_string(), "b".to_string(), "c".to_string()]);
        let removed = paths.remove(1);
        assert_eq!(removed.map(|s| s.to_string()), Some("b".to_string()));
        assert_eq!(paths.len(), 2);
    }

    #[test]
    fn test_remove_first() {
        let mut paths = SysPaths::with_paths(vec!["first".to_string(), "second".to_string()]);
        paths.remove(0);
        assert_eq!(paths.get(0).map(|s| s.as_ref()), Some("second"));
    }

    #[test]
    fn test_remove_out_of_bounds() {
        let mut paths = SysPaths::with_paths(vec!["a".to_string()]);
        let removed = paths.remove(10);
        assert!(removed.is_none());
    }

    #[test]
    fn test_clear() {
        let mut paths = SysPaths::with_paths(vec!["a".to_string(), "b".to_string()]);
        paths.clear();
        assert!(paths.is_empty());
    }

    // =========================================================================
    // Contains Tests
    // =========================================================================

    #[test]
    fn test_contains_found() {
        let paths = SysPaths::with_paths(vec!["/path/a".to_string(), "/path/b".to_string()]);
        assert!(paths.contains("/path/a"));
        assert!(paths.contains("/path/b"));
    }

    #[test]
    fn test_contains_not_found() {
        let paths = SysPaths::with_paths(vec!["/path/a".to_string()]);
        assert!(!paths.contains("/path/b"));
    }

    #[test]
    fn test_contains_empty() {
        let paths = SysPaths::new();
        assert!(!paths.contains("anything"));
    }

    // =========================================================================
    // Iteration Tests
    // =========================================================================

    #[test]
    fn test_iter() {
        let paths = SysPaths::with_paths(vec!["x".to_string(), "y".to_string()]);
        let collected: Vec<&str> = paths.iter().map(|s| s.as_ref()).collect();
        assert_eq!(collected, vec!["x", "y"]);
    }

    #[test]
    fn test_into_iter() {
        let paths = SysPaths::with_paths(vec!["a".to_string(), "b".to_string()]);
        let collected: Vec<&str> = (&paths).into_iter().map(|s| s.as_ref()).collect();
        assert_eq!(collected, vec!["a", "b"]);
    }

    // =========================================================================
    // As Slice Tests
    // =========================================================================

    #[test]
    fn test_as_slice() {
        let paths = SysPaths::with_paths(vec!["test".to_string()]);
        let slice = paths.as_slice();
        assert_eq!(slice.len(), 1);
    }

    // =========================================================================
    // Clone Tests
    // =========================================================================

    #[test]
    fn test_clone() {
        let paths = SysPaths::with_paths(vec!["/original".to_string()]);
        let cloned = paths.clone();
        assert_eq!(cloned.len(), 1);
    }

    // =========================================================================
    // Path Separator Tests
    // =========================================================================

    #[test]
    fn test_path_separator() {
        let sep = path_separator();
        #[cfg(target_os = "windows")]
        assert_eq!(sep, ';');
        #[cfg(not(target_os = "windows"))]
        assert_eq!(sep, ':');
    }

    // =========================================================================
    // Unicode Path Tests
    // =========================================================================

    #[test]
    fn test_unicode_paths() {
        let mut paths = SysPaths::new();
        paths.append("/home/用户/lib");
        paths.append("/data/データ");
        assert_eq!(paths.len(), 2);
        assert!(paths.contains("/home/用户/lib"));
    }

    // =========================================================================
    // Empty String Path Tests
    // =========================================================================

    #[test]
    fn test_empty_string_path() {
        let paths = SysPaths::with_paths(vec!["".to_string()]);
        assert_eq!(paths.len(), 1);
        assert!(paths.contains(""));
    }

    #[test]
    fn test_to_value_roundtrip_paths() {
        let paths = SysPaths::with_paths(vec!["/a".to_string(), "/b".to_string()]);
        let value = paths.to_value();
        let ptr = value
            .as_object_ptr()
            .expect("sys.path should convert to list object");
        let list = unsafe { &*(ptr as *const ListObject) };
        assert_eq!(list.len(), 2);

        let first = list.get(0).expect("sys.path[0] should exist");
        let second = list.get(1).expect("sys.path[1] should exist");
        let first_ptr = first
            .as_string_object_ptr()
            .expect("sys.path[0] should be string")
            as *const u8;
        let second_ptr = second
            .as_string_object_ptr()
            .expect("sys.path[1] should be string")
            as *const u8;

        assert_eq!(interned_by_ptr(first_ptr).expect("path[0] should resolve").as_ref(), "/a");
        assert_eq!(interned_by_ptr(second_ptr).expect("path[1] should resolve").as_ref(), "/b");
    }
}
