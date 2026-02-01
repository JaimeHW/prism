//! String interning for O(1) string equality and reduced memory usage.
//!
//! This module provides a high-performance string interner that stores unique
//! copies of strings and returns lightweight handles. Interned strings can be
//! compared by pointer equality, making identifier comparison extremely fast.

use parking_lot::RwLock;
use rustc_hash::FxHashMap;
use std::fmt;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

/// A handle to an interned string.
///
/// `InternedString` is a thin wrapper around an `Arc<str>` that provides
/// O(1) equality comparison via pointer comparison. Two `InternedString`s
/// are equal if and only if they were interned by the same interner and
/// contain the same string content.
#[derive(Clone)]
pub struct InternedString {
    inner: Arc<str>,
}

impl InternedString {
    /// Create a new interned string (for testing/internal use).
    /// Prefer using `StringInterner::intern` for deduplication.
    #[inline]
    fn new(s: Arc<str>) -> Self {
        Self { inner: s }
    }

    /// Get the string content.
    #[inline]
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.inner
    }

    /// Get the length in bytes.
    #[inline]
    #[must_use]
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    /// Check if the string is empty.
    #[inline]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    /// Get the pointer address for identity comparison.
    #[inline]
    fn ptr(&self) -> *const u8 {
        self.inner.as_ptr()
    }
}

impl PartialEq for InternedString {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        // Pointer comparison for O(1) equality
        Arc::ptr_eq(&self.inner, &other.inner)
    }
}

impl Eq for InternedString {}

impl Hash for InternedString {
    #[inline]
    fn hash<H: Hasher>(&self, state: &mut H) {
        // Hash the pointer for consistency with Eq
        self.ptr().hash(state);
    }
}

impl fmt::Debug for InternedString {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "InternedString({:?})", self.as_str())
    }
}

impl fmt::Display for InternedString {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl AsRef<str> for InternedString {
    #[inline]
    fn as_ref(&self) -> &str {
        self.as_str()
    }
}

impl std::ops::Deref for InternedString {
    type Target = str;

    #[inline]
    fn deref(&self) -> &Self::Target {
        self.as_str()
    }
}

impl PartialEq<str> for InternedString {
    fn eq(&self, other: &str) -> bool {
        self.as_str() == other
    }
}

impl PartialEq<&str> for InternedString {
    fn eq(&self, other: &&str) -> bool {
        self.as_str() == *other
    }
}

impl PartialEq<String> for InternedString {
    fn eq(&self, other: &String) -> bool {
        self.as_str() == other
    }
}

/// Thread-safe string interner.
///
/// The interner maintains a set of unique strings and returns handles to them.
/// Interning the same string multiple times returns the same handle, enabling
/// O(1) equality comparison.
pub struct StringInterner {
    /// Map from string content to interned handle.
    strings: RwLock<FxHashMap<Arc<str>, InternedString>>,
}

impl StringInterner {
    /// Create a new, empty string interner.
    #[must_use]
    pub fn new() -> Self {
        Self {
            strings: RwLock::new(FxHashMap::default()),
        }
    }

    /// Create a new interner with preallocated capacity.
    #[must_use]
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            strings: RwLock::new(FxHashMap::with_capacity_and_hasher(
                capacity,
                Default::default(),
            )),
        }
    }

    /// Intern a string, returning a handle.
    ///
    /// If the string has been interned before, the same handle is returned.
    /// This method is thread-safe.
    pub fn intern(&self, s: &str) -> InternedString {
        // Fast path: check if already interned with read lock
        {
            let strings = self.strings.read();
            if let Some(interned) = strings.get(s) {
                return interned.clone();
            }
        }

        // Slow path: insert with write lock
        let mut strings = self.strings.write();

        // Double-check after acquiring write lock
        if let Some(interned) = strings.get(s) {
            return interned.clone();
        }

        // Create new interned string
        let arc: Arc<str> = s.into();
        let interned = InternedString::new(arc.clone());
        strings.insert(arc, interned.clone());
        interned
    }

    /// Intern a string from an owned String.
    ///
    /// This avoids an allocation if the string is not already interned.
    pub fn intern_owned(&self, s: String) -> InternedString {
        // Fast path: check if already interned with read lock
        {
            let strings = self.strings.read();
            if let Some(interned) = strings.get(s.as_str()) {
                return interned.clone();
            }
        }

        // Slow path: insert with write lock
        let mut strings = self.strings.write();

        // Double-check after acquiring write lock
        if let Some(interned) = strings.get(s.as_str()) {
            return interned.clone();
        }

        // Create new interned string from owned String
        let arc: Arc<str> = s.into();
        let interned = InternedString::new(arc.clone());
        strings.insert(arc, interned.clone());
        interned
    }

    /// Get an already-interned string without creating a new one.
    ///
    /// Returns `None` if the string has not been interned.
    #[must_use]
    pub fn get(&self, s: &str) -> Option<InternedString> {
        self.strings.read().get(s).cloned()
    }

    /// Check if a string has been interned.
    #[must_use]
    pub fn contains(&self, s: &str) -> bool {
        self.strings.read().contains_key(s)
    }

    /// Get the number of interned strings.
    #[must_use]
    pub fn len(&self) -> usize {
        self.strings.read().len()
    }

    /// Check if the interner is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.strings.read().is_empty()
    }

    /// Clear all interned strings.
    ///
    /// Existing `InternedString` handles remain valid but will no longer
    /// be deduplicated with newly interned strings.
    pub fn clear(&self) {
        self.strings.write().clear();
    }
}

impl Default for StringInterner {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Debug for StringInterner {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let strings = self.strings.read();
        f.debug_struct("StringInterner")
            .field("count", &strings.len())
            .finish()
    }
}

/// A global string interner for common identifiers.
///
/// This is useful for keywords and common identifiers that appear frequently.
pub static GLOBAL_INTERNER: std::sync::LazyLock<StringInterner> =
    std::sync::LazyLock::new(StringInterner::new);

/// Intern a string using the global interner.
#[inline]
pub fn intern(s: &str) -> InternedString {
    GLOBAL_INTERNER.intern(s)
}

/// Intern an owned string using the global interner.
#[inline]
pub fn intern_owned(s: String) -> InternedString {
    GLOBAL_INTERNER.intern_owned(s)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_intern_same_string_returns_same_handle() {
        let interner = StringInterner::new();
        let s1 = interner.intern("hello");
        let s2 = interner.intern("hello");

        assert!(Arc::ptr_eq(&s1.inner, &s2.inner));
        assert_eq!(s1, s2);
    }

    #[test]
    fn test_intern_different_strings_returns_different_handles() {
        let interner = StringInterner::new();
        let s1 = interner.intern("hello");
        let s2 = interner.intern("world");

        assert!(!Arc::ptr_eq(&s1.inner, &s2.inner));
        assert_ne!(s1, s2);
    }

    #[test]
    fn test_interned_string_as_str() {
        let interner = StringInterner::new();
        let s = interner.intern("test content");

        assert_eq!(s.as_str(), "test content");
    }

    #[test]
    fn test_interned_string_len() {
        let interner = StringInterner::new();
        let s = interner.intern("hello");

        assert_eq!(s.len(), 5);
    }

    #[test]
    fn test_interned_string_is_empty() {
        let interner = StringInterner::new();
        let empty = interner.intern("");
        let non_empty = interner.intern("x");

        assert!(empty.is_empty());
        assert!(!non_empty.is_empty());
    }

    #[test]
    fn test_interner_get_existing() {
        let interner = StringInterner::new();
        interner.intern("existing");

        let result = interner.get("existing");
        assert!(result.is_some());
        assert_eq!(result.unwrap().as_str(), "existing");
    }

    #[test]
    fn test_interner_get_non_existing() {
        let interner = StringInterner::new();
        let result = interner.get("non_existing");

        assert!(result.is_none());
    }

    #[test]
    fn test_interner_contains() {
        let interner = StringInterner::new();
        interner.intern("present");

        assert!(interner.contains("present"));
        assert!(!interner.contains("absent"));
    }

    #[test]
    fn test_interner_len() {
        let interner = StringInterner::new();
        assert_eq!(interner.len(), 0);

        interner.intern("one");
        assert_eq!(interner.len(), 1);

        interner.intern("two");
        assert_eq!(interner.len(), 2);

        interner.intern("one"); // Duplicate
        assert_eq!(interner.len(), 2);
    }

    #[test]
    fn test_interner_is_empty() {
        let interner = StringInterner::new();
        assert!(interner.is_empty());

        interner.intern("something");
        assert!(!interner.is_empty());
    }

    #[test]
    fn test_interner_clear() {
        let interner = StringInterner::new();
        let s1 = interner.intern("first");
        interner.intern("second");

        assert_eq!(interner.len(), 2);
        interner.clear();
        assert_eq!(interner.len(), 0);

        // Old handle still valid
        assert_eq!(s1.as_str(), "first");

        // New interned string is different handle
        let s1_new = interner.intern("first");
        assert_ne!(s1, s1_new); // Different Arc
    }

    #[test]
    fn test_interner_with_capacity() {
        let interner = StringInterner::with_capacity(100);
        assert!(interner.is_empty());
    }

    #[test]
    fn test_interned_string_hash() {
        use std::collections::HashMap;

        let interner = StringInterner::new();
        let s1 = interner.intern("key");
        let s2 = interner.intern("key");

        let mut map = HashMap::new();
        map.insert(s1.clone(), 42);

        assert_eq!(map.get(&s2), Some(&42));
    }

    #[test]
    fn test_interned_string_debug() {
        let interner = StringInterner::new();
        let s = interner.intern("debug_test");

        let debug_str = format!("{:?}", s);
        assert!(debug_str.contains("debug_test"));
    }

    #[test]
    fn test_interned_string_display() {
        let interner = StringInterner::new();
        let s = interner.intern("display_test");

        assert_eq!(format!("{}", s), "display_test");
    }

    #[test]
    fn test_interned_string_as_ref() {
        let interner = StringInterner::new();
        let s = interner.intern("ref_test");

        let s_ref: &str = s.as_ref();
        assert_eq!(s_ref, "ref_test");
    }

    #[test]
    fn test_interned_string_deref() {
        let interner = StringInterner::new();
        let s = interner.intern("deref_test");

        // Use deref to get &str
        let len = s.len(); // Uses Deref
        assert_eq!(len, 10);

        // String methods work
        assert!(s.starts_with("deref"));
    }

    #[test]
    fn test_interned_string_eq_str() {
        let interner = StringInterner::new();
        let s = interner.intern("compare");

        assert!(s == "compare");
        assert!(s == *"compare");
        assert!(s != "different");
    }

    #[test]
    fn test_interned_string_eq_string() {
        let interner = StringInterner::new();
        let s = interner.intern("compare");

        assert!(s == String::from("compare"));
        assert!(s != String::from("different"));
    }

    #[test]
    fn test_interned_string_clone() {
        let interner = StringInterner::new();
        let s1 = interner.intern("clone_test");
        let s2 = s1.clone();

        assert_eq!(s1, s2);
        assert!(Arc::ptr_eq(&s1.inner, &s2.inner));
    }

    #[test]
    fn test_intern_owned() {
        let interner = StringInterner::new();
        let s1 = interner.intern_owned(String::from("owned"));
        let s2 = interner.intern("owned");

        assert_eq!(s1, s2);
    }

    #[test]
    fn test_intern_owned_deduplication() {
        let interner = StringInterner::new();
        interner.intern("existing");

        let s = interner.intern_owned(String::from("existing"));
        assert_eq!(interner.len(), 1);
        assert_eq!(s.as_str(), "existing");
    }

    #[test]
    fn test_global_interner() {
        let s1 = intern("global_test");
        let s2 = intern("global_test");

        assert_eq!(s1, s2);
        assert!(Arc::ptr_eq(&s1.inner, &s2.inner));
    }

    #[test]
    fn test_global_intern_owned() {
        let s1 = intern_owned(String::from("global_owned"));
        let s2 = intern("global_owned");

        assert_eq!(s1, s2);
    }

    #[test]
    fn test_interner_debug() {
        let interner = StringInterner::new();
        interner.intern("a");
        interner.intern("b");

        let debug_str = format!("{:?}", interner);
        assert!(debug_str.contains("StringInterner"));
        assert!(debug_str.contains("count"));
    }

    #[test]
    fn test_unicode_strings() {
        let interner = StringInterner::new();
        let s1 = interner.intern("こんにちは");
        let s2 = interner.intern("こんにちは");
        let s3 = interner.intern("世界");

        assert_eq!(s1, s2);
        assert_ne!(s1, s3);
        assert_eq!(s1.as_str(), "こんにちは");
    }

    #[test]
    fn test_emoji_strings() {
        let interner = StringInterner::new();
        let s1 = interner.intern("🦀🐍");
        let s2 = interner.intern("🦀🐍");

        assert_eq!(s1, s2);
        assert_eq!(s1.as_str(), "🦀🐍");
    }

    #[test]
    fn test_whitespace_significant() {
        let interner = StringInterner::new();
        let s1 = interner.intern("hello");
        let s2 = interner.intern("hello ");
        let s3 = interner.intern(" hello");

        assert_ne!(s1, s2);
        assert_ne!(s1, s3);
        assert_ne!(s2, s3);
    }

    #[test]
    fn test_case_sensitive() {
        let interner = StringInterner::new();
        let s1 = interner.intern("Hello");
        let s2 = interner.intern("hello");
        let s3 = interner.intern("HELLO");

        assert_ne!(s1, s2);
        assert_ne!(s1, s3);
        assert_ne!(s2, s3);
    }

    #[test]
    fn test_empty_string() {
        let interner = StringInterner::new();
        let s1 = interner.intern("");
        let s2 = interner.intern("");

        assert_eq!(s1, s2);
        assert!(s1.is_empty());
        assert_eq!(s1.len(), 0);
    }

    #[test]
    fn test_long_string() {
        let interner = StringInterner::new();
        let long = "x".repeat(10000);
        let s1 = interner.intern(&long);
        let s2 = interner.intern(&long);

        assert_eq!(s1, s2);
        assert_eq!(s1.len(), 10000);
    }

    #[test]
    fn test_special_characters() {
        let interner = StringInterner::new();
        let s1 = interner.intern("line1\nline2\ttab");
        let s2 = interner.intern("line1\nline2\ttab");

        assert_eq!(s1, s2);
        assert!(s1.contains('\n'));
        assert!(s1.contains('\t'));
    }

    #[test]
    fn test_null_character() {
        let interner = StringInterner::new();
        let s1 = interner.intern("before\0after");
        let s2 = interner.intern("before\0after");

        assert_eq!(s1, s2);
        assert_eq!(s1.len(), 12);
    }

    #[test]
    fn test_concurrent_interning() {
        use std::thread;

        let interner = Arc::new(StringInterner::new());
        let mut handles = vec![];

        for i in 0..10 {
            let interner = Arc::clone(&interner);
            handles.push(thread::spawn(move || {
                let s = format!("thread_{}", i);
                for _ in 0..100 {
                    interner.intern(&s);
                }
                interner.intern(&s)
            }));
        }

        for handle in handles {
            let _ = handle.join().unwrap();
        }

        // Each thread should have created exactly one unique string
        assert_eq!(interner.len(), 10);
    }

    #[test]
    fn test_concurrent_same_string() {
        use std::thread;

        let interner = Arc::new(StringInterner::new());
        let mut handles = vec![];

        for _ in 0..10 {
            let interner = Arc::clone(&interner);
            handles.push(thread::spawn(move || {
                for _ in 0..100 {
                    interner.intern("shared_string");
                }
                interner.intern("shared_string")
            }));
        }

        let results: Vec<_> = handles.into_iter().map(|h| h.join().unwrap()).collect();

        // All threads should get the same handle
        for result in &results[1..] {
            assert_eq!(&results[0], result);
        }

        // Only one string should be interned
        assert_eq!(interner.len(), 1);
    }

    #[test]
    fn test_interner_default() {
        let interner = StringInterner::default();
        assert!(interner.is_empty());
    }
}
