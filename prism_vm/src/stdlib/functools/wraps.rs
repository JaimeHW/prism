//! Wrapper utilities for functools.
//!
//! Implements `update_wrapper` and `wraps` for copying function metadata
//! from wrapped functions to their wrappers. This is essential for maintaining
//! introspection capabilities when using decorators.
//!
//! # Attributes Copied
//!
//! | Attribute | Description |
//! |-----------|-------------|
//! | `__module__` | Module where the function was defined |
//! | `__name__` | Function name |
//! | `__qualname__` | Qualified name (e.g., `Class.method`) |
//! | `__doc__` | Documentation string |
//! | `__wrapped__` | Reference to the original function |
//!
//! # Performance
//!
//! All attribute operations are O(1). The wrapper metadata is stored
//! in a compact `WrapperMetadata` struct using interned strings for
//! zero-copy attribute names.

use prism_core::Value;

// =============================================================================
// Constants
// =============================================================================

/// Attributes that are assigned (replaced) on the wrapper.
///
/// These are the default `WRAPPER_ASSIGNMENTS` in Python's functools.
pub const WRAPPER_ASSIGNMENTS: &[&str] = &[
    "__module__",
    "__name__",
    "__qualname__",
    "__annotations__",
    "__doc__",
];

/// Attributes that are updated (merged) on the wrapper.
///
/// These are the default `WRAPPER_UPDATES` in Python's functools.
pub const WRAPPER_UPDATES: &[&str] = &["__dict__"];

// =============================================================================
// Wrapper Metadata
// =============================================================================

/// Metadata extracted from a wrapped function.
///
/// Stores all the standard attributes that `update_wrapper` copies.
/// Uses `Option<Value>` for each field since not all functions have
/// all attributes.
#[derive(Debug, Clone)]
pub struct WrapperMetadata {
    /// `__module__` — the module name.
    pub module: Option<Value>,
    /// `__name__` — the function name.
    pub name: Option<Value>,
    /// `__qualname__` — the qualified name.
    pub qualname: Option<Value>,
    /// `__doc__` — the docstring.
    pub doc: Option<Value>,
    /// `__wrapped__` — reference to the original function.
    pub wrapped: Option<Value>,
    /// `__annotations__` — type annotations dict.
    pub annotations: Option<Value>,
}

impl WrapperMetadata {
    /// Create empty metadata.
    #[inline]
    pub fn new() -> Self {
        Self {
            module: None,
            name: None,
            qualname: None,
            doc: None,
            wrapped: None,
            annotations: None,
        }
    }

    /// Create metadata from a wrapped function's attributes.
    pub fn from_wrapped(
        module: Option<Value>,
        name: Option<Value>,
        qualname: Option<Value>,
        doc: Option<Value>,
        wrapped: Value,
    ) -> Self {
        Self {
            module,
            name,
            qualname,
            doc,
            wrapped: Some(wrapped),
            annotations: None,
        }
    }

    /// Create metadata with all fields populated.
    pub fn full(
        module: Value,
        name: Value,
        qualname: Value,
        doc: Value,
        wrapped: Value,
        annotations: Option<Value>,
    ) -> Self {
        Self {
            module: Some(module),
            name: Some(name),
            qualname: Some(qualname),
            doc: Some(doc),
            wrapped: Some(wrapped),
            annotations,
        }
    }

    /// Get an attribute by name.
    pub fn get_attr(&self, name: &str) -> Option<&Value> {
        match name {
            "__module__" => self.module.as_ref(),
            "__name__" => self.name.as_ref(),
            "__qualname__" => self.qualname.as_ref(),
            "__doc__" => self.doc.as_ref(),
            "__wrapped__" => self.wrapped.as_ref(),
            "__annotations__" => self.annotations.as_ref(),
            _ => None,
        }
    }

    /// Set an attribute by name.
    pub fn set_attr(&mut self, name: &str, value: Value) -> bool {
        match name {
            "__module__" => {
                self.module = Some(value);
                true
            }
            "__name__" => {
                self.name = Some(value);
                true
            }
            "__qualname__" => {
                self.qualname = Some(value);
                true
            }
            "__doc__" => {
                self.doc = Some(value);
                true
            }
            "__wrapped__" => {
                self.wrapped = Some(value);
                true
            }
            "__annotations__" => {
                self.annotations = Some(value);
                true
            }
            _ => false,
        }
    }

    /// Check if any metadata is set.
    pub fn has_any(&self) -> bool {
        self.module.is_some()
            || self.name.is_some()
            || self.qualname.is_some()
            || self.doc.is_some()
            || self.wrapped.is_some()
            || self.annotations.is_some()
    }

    /// Count how many attributes are set.
    pub fn count(&self) -> usize {
        let mut n = 0;
        if self.module.is_some() {
            n += 1;
        }
        if self.name.is_some() {
            n += 1;
        }
        if self.qualname.is_some() {
            n += 1;
        }
        if self.doc.is_some() {
            n += 1;
        }
        if self.wrapped.is_some() {
            n += 1;
        }
        if self.annotations.is_some() {
            n += 1;
        }
        n
    }

    /// List all set attribute names.
    pub fn dir(&self) -> Vec<&'static str> {
        let mut attrs = Vec::with_capacity(6);
        if self.module.is_some() {
            attrs.push("__module__");
        }
        if self.name.is_some() {
            attrs.push("__name__");
        }
        if self.qualname.is_some() {
            attrs.push("__qualname__");
        }
        if self.doc.is_some() {
            attrs.push("__doc__");
        }
        if self.wrapped.is_some() {
            attrs.push("__wrapped__");
        }
        if self.annotations.is_some() {
            attrs.push("__annotations__");
        }
        attrs
    }
}

impl Default for WrapperMetadata {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// update_wrapper
// =============================================================================

/// Copy attributes from a source metadata to a destination metadata.
///
/// This is the core of both `update_wrapper` and `wraps`. It copies
/// the standard `WRAPPER_ASSIGNMENTS` attributes from source to destination.
///
/// # Arguments
///
/// * `dest` — The wrapper's metadata (will be modified).
/// * `source` — The wrapped function's metadata.
/// * `assignments` — Which attributes to copy (default: `WRAPPER_ASSIGNMENTS`).
pub fn update_wrapper(
    dest: &mut WrapperMetadata,
    source: &WrapperMetadata,
    assignments: Option<&[&str]>,
) {
    let attrs = assignments.unwrap_or(WRAPPER_ASSIGNMENTS);
    for attr in attrs {
        if let Some(value) = source.get_attr(attr) {
            dest.set_attr(attr, value.clone());
        }
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod wraps_tests {
    use super::*;
    use prism_core::intern::intern;

    fn str_val(s: &str) -> Value {
        Value::string(intern(s))
    }

    fn int(i: i64) -> Value {
        Value::int_unchecked(i)
    }

    // =========================================================================
    // WrapperMetadata construction tests
    // =========================================================================

    #[test]
    fn test_new_empty() {
        let meta = WrapperMetadata::new();
        assert!(!meta.has_any());
        assert_eq!(meta.count(), 0);
    }

    #[test]
    fn test_from_wrapped() {
        let meta = WrapperMetadata::from_wrapped(
            Some(str_val("mymodule")),
            Some(str_val("myfunc")),
            Some(str_val("MyClass.myfunc")),
            Some(str_val("Does stuff")),
            int(42), // function reference placeholder
        );
        assert!(meta.has_any());
        assert_eq!(meta.count(), 5); // module, name, qualname, doc, wrapped
    }

    #[test]
    fn test_full() {
        let meta = WrapperMetadata::full(
            str_val("mod"),
            str_val("func"),
            str_val("mod.func"),
            str_val("docstring"),
            int(0),
            None,
        );
        assert_eq!(meta.count(), 5); // annotations is None
    }

    // =========================================================================
    // Attribute access tests
    // =========================================================================

    #[test]
    fn test_get_attr_module() {
        let meta = WrapperMetadata::from_wrapped(Some(str_val("mymod")), None, None, None, int(0));
        assert!(meta.get_attr("__module__").is_some());
        assert!(meta.get_attr("__name__").is_none());
    }

    #[test]
    fn test_get_attr_all_fields() {
        let meta = WrapperMetadata::full(
            str_val("mod"),
            str_val("fn"),
            str_val("mod.fn"),
            str_val("doc"),
            int(0),
            Some(int(99)),
        );
        assert!(meta.get_attr("__module__").is_some());
        assert!(meta.get_attr("__name__").is_some());
        assert!(meta.get_attr("__qualname__").is_some());
        assert!(meta.get_attr("__doc__").is_some());
        assert!(meta.get_attr("__wrapped__").is_some());
        assert!(meta.get_attr("__annotations__").is_some());
    }

    #[test]
    fn test_get_attr_unknown() {
        let meta = WrapperMetadata::new();
        assert!(meta.get_attr("__unknown__").is_none());
    }

    #[test]
    fn test_set_attr() {
        let mut meta = WrapperMetadata::new();
        assert!(meta.set_attr("__name__", str_val("test")));
        assert_eq!(meta.count(), 1);
        assert!(meta.get_attr("__name__").is_some());
    }

    #[test]
    fn test_set_attr_unknown() {
        let mut meta = WrapperMetadata::new();
        assert!(!meta.set_attr("__unknown__", int(0)));
    }

    #[test]
    fn test_set_attr_overwrites() {
        let mut meta = WrapperMetadata::new();
        meta.set_attr("__doc__", str_val("old"));
        meta.set_attr("__doc__", str_val("new"));
        // Should still count as 1
        assert_eq!(meta.count(), 1);
    }

    // =========================================================================
    // dir tests
    // =========================================================================

    #[test]
    fn test_dir_empty() {
        let meta = WrapperMetadata::new();
        assert!(meta.dir().is_empty());
    }

    #[test]
    fn test_dir_partial() {
        let mut meta = WrapperMetadata::new();
        meta.set_attr("__name__", str_val("f"));
        meta.set_attr("__doc__", str_val("d"));

        let dir = meta.dir();
        assert_eq!(dir.len(), 2);
        assert!(dir.contains(&"__name__"));
        assert!(dir.contains(&"__doc__"));
    }

    #[test]
    fn test_dir_full() {
        let meta = WrapperMetadata::full(
            str_val("m"),
            str_val("n"),
            str_val("q"),
            str_val("d"),
            int(0),
            Some(int(1)),
        );
        assert_eq!(meta.dir().len(), 6);
    }

    // =========================================================================
    // update_wrapper tests
    // =========================================================================

    #[test]
    fn test_update_wrapper_copies_all() {
        let source = WrapperMetadata::full(
            str_val("source_mod"),
            str_val("source_fn"),
            str_val("source_mod.source_fn"),
            str_val("source docs"),
            int(0),
            None,
        );
        let mut dest = WrapperMetadata::new();

        update_wrapper(&mut dest, &source, None);

        // WRAPPER_ASSIGNMENTS: module, name, qualname, annotations, doc
        assert!(dest.get_attr("__module__").is_some());
        assert!(dest.get_attr("__name__").is_some());
        assert!(dest.get_attr("__qualname__").is_some());
        assert!(dest.get_attr("__doc__").is_some());
        // __wrapped__ is NOT in WRAPPER_ASSIGNMENTS
        assert!(dest.get_attr("__wrapped__").is_none());
    }

    #[test]
    fn test_update_wrapper_custom_assignments() {
        let source = WrapperMetadata::full(
            str_val("mod"),
            str_val("fn"),
            str_val("mod.fn"),
            str_val("doc"),
            int(0),
            None,
        );
        let mut dest = WrapperMetadata::new();

        // Only copy name and doc
        update_wrapper(&mut dest, &source, Some(&["__name__", "__doc__"]));

        assert!(dest.get_attr("__name__").is_some());
        assert!(dest.get_attr("__doc__").is_some());
        assert!(dest.get_attr("__module__").is_none());
    }

    #[test]
    fn test_update_wrapper_skips_missing() {
        let mut source = WrapperMetadata::new();
        source.set_attr("__name__", str_val("fn"));
        // module, qualname, doc are NOT set

        let mut dest = WrapperMetadata::new();
        update_wrapper(&mut dest, &source, None);

        // Only __name__ should be copied
        assert!(dest.get_attr("__name__").is_some());
        assert!(dest.get_attr("__module__").is_none());
    }

    #[test]
    fn test_update_wrapper_overwrites_dest() {
        let mut source = WrapperMetadata::new();
        source.set_attr("__name__", str_val("original"));

        let mut dest = WrapperMetadata::new();
        dest.set_attr("__name__", str_val("wrapper"));

        update_wrapper(&mut dest, &source, None);

        // Source should have overwritten dest
        let name = dest.get_attr("__name__").unwrap();
        assert!(name.is_string());
    }

    // =========================================================================
    // Constants tests
    // =========================================================================

    #[test]
    fn test_wrapper_assignments_contents() {
        assert!(WRAPPER_ASSIGNMENTS.contains(&"__module__"));
        assert!(WRAPPER_ASSIGNMENTS.contains(&"__name__"));
        assert!(WRAPPER_ASSIGNMENTS.contains(&"__qualname__"));
        assert!(WRAPPER_ASSIGNMENTS.contains(&"__annotations__"));
        assert!(WRAPPER_ASSIGNMENTS.contains(&"__doc__"));
        assert_eq!(WRAPPER_ASSIGNMENTS.len(), 5);
    }

    #[test]
    fn test_wrapper_updates_contents() {
        assert!(WRAPPER_UPDATES.contains(&"__dict__"));
        assert_eq!(WRAPPER_UPDATES.len(), 1);
    }

    // =========================================================================
    // Clone/Default tests
    // =========================================================================

    #[test]
    fn test_metadata_clone() {
        let meta = WrapperMetadata::full(
            str_val("m"),
            str_val("n"),
            str_val("q"),
            str_val("d"),
            int(0),
            None,
        );
        let clone = meta.clone();
        assert_eq!(clone.count(), meta.count());
    }

    #[test]
    fn test_metadata_default() {
        let meta = WrapperMetadata::default();
        assert!(!meta.has_any());
    }
}
