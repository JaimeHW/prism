//! Command-line argument handling.
//!
//! Provides zero-copy access to command-line arguments with
//! efficient sharing via Arc<str>.

use prism_core::Value;
use prism_core::intern::intern;
use prism_runtime::types::list::ListObject;
use std::sync::Arc;

/// Command-line arguments container.
///
/// Uses `Arc<str>` for individual arguments to enable zero-copy sharing
/// when arguments are accessed from Python.
#[derive(Debug, Clone)]
pub struct SysArgv {
    /// The argument list.
    args: Arc<[Arc<str>]>,
}

impl SysArgv {
    /// Create from a vector of strings.
    #[inline]
    pub fn new(args: Vec<String>) -> Self {
        let args: Vec<Arc<str>> = args.into_iter().map(|s| s.into()).collect();
        Self { args: args.into() }
    }

    /// Create from environment (std::env::args).
    #[inline]
    pub fn from_env() -> Self {
        let args: Vec<Arc<str>> = std::env::args().map(|s| s.into()).collect();
        Self { args: args.into() }
    }

    /// Create empty argv.
    #[inline]
    pub fn empty() -> Self {
        Self {
            args: Arc::from([]),
        }
    }

    /// Get the number of arguments.
    #[inline]
    pub fn len(&self) -> usize {
        self.args.len()
    }

    /// Check if empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.args.is_empty()
    }

    /// Get argument by index.
    #[inline]
    pub fn get(&self, index: usize) -> Option<&Arc<str>> {
        self.args.get(index)
    }

    /// Get the script name (first argument).
    #[inline]
    pub fn script(&self) -> Option<&Arc<str>> {
        self.args.first()
    }

    /// Iterate over arguments.
    #[inline]
    pub fn iter(&self) -> impl Iterator<Item = &Arc<str>> {
        self.args.iter()
    }

    /// Get the underlying slice.
    #[inline]
    pub fn as_slice(&self) -> &[Arc<str>] {
        &self.args
    }

    /// Convert argv to a Python list value (`list[str]`).
    ///
    /// This builds a list of interned string Values and returns it as an object Value.
    pub fn to_value(&self) -> Value {
        let values: Vec<Value> = self
            .args
            .iter()
            .map(|arg| Value::string(intern(arg.as_ref())))
            .collect();
        let list = ListObject::from_slice(&values);
        let ptr = Box::into_raw(Box::new(list)) as *const ();
        Value::object_ptr(ptr)
    }
}

impl Default for SysArgv {
    fn default() -> Self {
        Self::empty()
    }
}

impl<'a> IntoIterator for &'a SysArgv {
    type Item = &'a Arc<str>;
    type IntoIter = std::slice::Iter<'a, Arc<str>>;

    fn into_iter(self) -> Self::IntoIter {
        self.args.iter()
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
        let argv = SysArgv::new(vec![]);
        assert!(argv.is_empty());
        assert_eq!(argv.len(), 0);
    }

    #[test]
    fn test_new_single() {
        let argv = SysArgv::new(vec!["script.py".to_string()]);
        assert!(!argv.is_empty());
        assert_eq!(argv.len(), 1);
    }

    #[test]
    fn test_new_multiple() {
        let argv = SysArgv::new(vec![
            "script.py".to_string(),
            "--flag".to_string(),
            "value".to_string(),
        ]);
        assert_eq!(argv.len(), 3);
    }

    #[test]
    fn test_empty() {
        let argv = SysArgv::empty();
        assert!(argv.is_empty());
    }

    #[test]
    fn test_default() {
        let argv = SysArgv::default();
        assert!(argv.is_empty());
    }

    // =========================================================================
    // Access Tests
    // =========================================================================

    #[test]
    fn test_get_valid_index() {
        let argv = SysArgv::new(vec!["a".to_string(), "b".to_string(), "c".to_string()]);
        assert_eq!(argv.get(0).map(|s| s.as_ref()), Some("a"));
        assert_eq!(argv.get(1).map(|s| s.as_ref()), Some("b"));
        assert_eq!(argv.get(2).map(|s| s.as_ref()), Some("c"));
    }

    #[test]
    fn test_get_invalid_index() {
        let argv = SysArgv::new(vec!["a".to_string()]);
        assert!(argv.get(1).is_none());
        assert!(argv.get(100).is_none());
    }

    #[test]
    fn test_script() {
        let argv = SysArgv::new(vec!["script.py".to_string(), "--flag".to_string()]);
        assert_eq!(argv.script().map(|s| s.as_ref()), Some("script.py"));
    }

    #[test]
    fn test_script_empty() {
        let argv = SysArgv::empty();
        assert!(argv.script().is_none());
    }

    // =========================================================================
    // Iteration Tests
    // =========================================================================

    #[test]
    fn test_iter() {
        let argv = SysArgv::new(vec!["a".to_string(), "b".to_string()]);
        let collected: Vec<&str> = argv.iter().map(|s| s.as_ref()).collect();
        assert_eq!(collected, vec!["a", "b"]);
    }

    #[test]
    fn test_into_iter() {
        let argv = SysArgv::new(vec!["x".to_string(), "y".to_string()]);
        let collected: Vec<&str> = (&argv).into_iter().map(|s| s.as_ref()).collect();
        assert_eq!(collected, vec!["x", "y"]);
    }

    #[test]
    fn test_for_loop() {
        let argv = SysArgv::new(vec!["one".to_string(), "two".to_string()]);
        let mut count = 0;
        for _arg in &argv {
            count += 1;
        }
        assert_eq!(count, 2);
    }

    // =========================================================================
    // Slice Tests
    // =========================================================================

    #[test]
    fn test_as_slice() {
        let argv = SysArgv::new(vec!["test".to_string()]);
        let slice = argv.as_slice();
        assert_eq!(slice.len(), 1);
        assert_eq!(slice[0].as_ref(), "test");
    }

    // =========================================================================
    // Clone Tests
    // =========================================================================

    #[test]
    fn test_clone() {
        let argv = SysArgv::new(vec!["original".to_string()]);
        let cloned = argv.clone();
        assert_eq!(cloned.len(), 1);
        assert_eq!(cloned.get(0).map(|s| s.as_ref()), Some("original"));
    }

    #[test]
    fn test_clone_shares_data() {
        let argv = SysArgv::new(vec!["shared".to_string()]);
        let cloned = argv.clone();
        // Both should point to same Arc data
        assert!(Arc::ptr_eq(&argv.args, &cloned.args));
    }

    // =========================================================================
    // Unicode Tests
    // =========================================================================

    #[test]
    fn test_unicode_args() {
        let argv = SysArgv::new(vec![
            "日本語.py".to_string(),
            "αβγ".to_string(),
            "مرحبا".to_string(),
        ]);
        assert_eq!(argv.len(), 3);
        assert_eq!(argv.get(0).map(|s| s.as_ref()), Some("日本語.py"));
        assert_eq!(argv.get(1).map(|s| s.as_ref()), Some("αβγ"));
        assert_eq!(argv.get(2).map(|s| s.as_ref()), Some("مرحبا"));
    }

    #[test]
    fn test_emoji_args() {
        let argv = SysArgv::new(vec!["🐍.py".to_string(), "--🚀".to_string()]);
        assert_eq!(argv.len(), 2);
    }

    // =========================================================================
    // Edge Cases
    // =========================================================================

    #[test]
    fn test_empty_string_arg() {
        let argv = SysArgv::new(vec!["".to_string()]);
        assert_eq!(argv.len(), 1);
        assert_eq!(argv.get(0).map(|s| s.as_ref()), Some(""));
    }

    #[test]
    fn test_whitespace_args() {
        let argv = SysArgv::new(vec![" ".to_string(), "  ".to_string(), "\t".to_string()]);
        assert_eq!(argv.len(), 3);
    }

    #[test]
    fn test_special_chars() {
        let argv = SysArgv::new(vec![
            "--key=value".to_string(),
            "-x".to_string(),
            "path/to/file".to_string(),
            "C:\\Windows\\file".to_string(),
        ]);
        assert_eq!(argv.len(), 4);
    }

    #[test]
    fn test_to_value_roundtrip_strings() {
        let argv = SysArgv::new(vec![
            "script.py".to_string(),
            "--flag".to_string(),
            "value".to_string(),
        ]);

        let value = argv.to_value();
        let ptr = value.as_object_ptr().expect("argv should be a list object");
        let list = unsafe { &*(ptr as *const ListObject) };

        assert_eq!(list.len(), 3);

        let first = list.get(0).expect("argv[0] should exist");
        let second = list.get(1).expect("argv[1] should exist");
        let third = list.get(2).expect("argv[2] should exist");

        let first_ptr = first
            .as_string_object_ptr()
            .expect("argv[0] should be an interned string") as *const u8;
        let second_ptr = second
            .as_string_object_ptr()
            .expect("argv[1] should be an interned string") as *const u8;
        let third_ptr = third
            .as_string_object_ptr()
            .expect("argv[2] should be an interned string") as *const u8;

        assert_eq!(
            interned_by_ptr(first_ptr)
                .expect("argv[0] should resolve")
                .as_ref(),
            "script.py"
        );
        assert_eq!(
            interned_by_ptr(second_ptr)
                .expect("argv[1] should resolve")
                .as_ref(),
            "--flag"
        );
        assert_eq!(
            interned_by_ptr(third_ptr)
                .expect("argv[2] should resolve")
                .as_ref(),
            "value"
        );
    }
}
