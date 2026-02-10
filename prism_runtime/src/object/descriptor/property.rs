//! Property descriptor implementation.
//!
//! Properties provide computed attribute access via getter, setter, and deleter
//! functions. This is the mechanism behind Python's `@property` decorator.
//!
//! # Example (Python)
//!
//! ```python
//! class Circle:
//!     def __init__(self, radius):
//!         self._radius = radius
//!
//!     @property
//!     def radius(self):
//!         return self._radius
//!
//!     @radius.setter
//!     def radius(self, value):
//!         if value < 0:
//!             raise ValueError("Radius cannot be negative")
//!         self._radius = value
//!
//!     @property
//!     def area(self):
//!         return 3.14159 * self._radius ** 2
//! ```
//!
//! # Performance
//!
//! - Flags are cached for fast data descriptor checks
//! - Function pointers stored directly (no indirection)
//! - Inline caching can specialize property access

use super::{Descriptor, DescriptorFlags, DescriptorKind};
use prism_core::{PrismError, PrismResult, Value};

// =============================================================================
// Property Flags
// =============================================================================

bitflags::bitflags! {
    /// Flags for property configuration.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub struct PropertyFlags: u8 {
        /// Property has a getter.
        const HAS_GETTER = 1 << 0;
        /// Property has a setter.
        const HAS_SETTER = 1 << 1;
        /// Property has a deleter.
        const HAS_DELETER = 1 << 2;
        /// Property has documentation.
        const HAS_DOC = 1 << 3;
    }
}

impl Default for PropertyFlags {
    fn default() -> Self {
        Self::empty()
    }
}

// =============================================================================
// Property Descriptor
// =============================================================================

/// Property descriptor with optional getter, setter, and deleter.
///
/// # Memory Layout
///
/// Properties store function values directly for the getter, setter, and deleter.
/// Each is optional, allowing read-only, write-only, or delete-only properties
/// (though read-only is by far the most common).
///
/// # Thread Safety
///
/// PropertyDescriptor is immutable after creation. The getter/setter/deleter
/// functions may internally have mutable state, but that's managed by the
/// function objects themselves.
#[derive(Debug)]
pub struct PropertyDescriptor {
    /// Property flags for fast checking.
    flags: PropertyFlags,
    /// The getter function (fget).
    getter: Option<Value>,
    /// The setter function (fset).
    setter: Option<Value>,
    /// The deleter function (fdel).
    deleter: Option<Value>,
    /// Documentation string.
    doc: Option<Value>,
}

impl PropertyDescriptor {
    /// Create a new property with only a getter (read-only).
    pub fn new_getter(getter: Value) -> Self {
        Self {
            flags: PropertyFlags::HAS_GETTER,
            getter: Some(getter),
            setter: None,
            deleter: None,
            doc: None,
        }
    }

    /// Create a new property with getter and setter.
    pub fn new_getter_setter(getter: Value, setter: Value) -> Self {
        Self {
            flags: PropertyFlags::HAS_GETTER | PropertyFlags::HAS_SETTER,
            getter: Some(getter),
            setter: Some(setter),
            deleter: None,
            doc: None,
        }
    }

    /// Create a new property with all three functions.
    pub fn new_full(
        getter: Option<Value>,
        setter: Option<Value>,
        deleter: Option<Value>,
        doc: Option<Value>,
    ) -> Self {
        let mut flags = PropertyFlags::empty();
        if getter.is_some() {
            flags |= PropertyFlags::HAS_GETTER;
        }
        if setter.is_some() {
            flags |= PropertyFlags::HAS_SETTER;
        }
        if deleter.is_some() {
            flags |= PropertyFlags::HAS_DELETER;
        }
        if doc.is_some() {
            flags |= PropertyFlags::HAS_DOC;
        }

        Self {
            flags,
            getter,
            setter,
            deleter,
            doc,
        }
    }

    /// Get property flags.
    #[inline]
    pub fn property_flags(&self) -> PropertyFlags {
        self.flags
    }

    /// Check if property has a getter.
    #[inline]
    pub fn has_getter(&self) -> bool {
        self.flags.contains(PropertyFlags::HAS_GETTER)
    }

    /// Check if property has a setter.
    #[inline]
    pub fn has_setter(&self) -> bool {
        self.flags.contains(PropertyFlags::HAS_SETTER)
    }

    /// Check if property has a deleter.
    #[inline]
    pub fn has_deleter(&self) -> bool {
        self.flags.contains(PropertyFlags::HAS_DELETER)
    }

    /// Get the getter function.
    pub fn getter(&self) -> Option<Value> {
        self.getter
    }

    /// Get the setter function.
    pub fn setter(&self) -> Option<Value> {
        self.setter
    }

    /// Get the deleter function.
    pub fn deleter(&self) -> Option<Value> {
        self.deleter
    }

    /// Get the documentation.
    pub fn doc(&self) -> Option<Value> {
        self.doc
    }

    /// Create a new property with a different getter.
    pub fn with_getter(&self, getter: Value) -> Self {
        Self {
            flags: self.flags | PropertyFlags::HAS_GETTER,
            getter: Some(getter),
            setter: self.setter,
            deleter: self.deleter,
            doc: self.doc,
        }
    }

    /// Create a new property with a different setter.
    pub fn with_setter(&self, setter: Value) -> Self {
        Self {
            flags: self.flags | PropertyFlags::HAS_SETTER,
            getter: self.getter,
            setter: Some(setter),
            deleter: self.deleter,
            doc: self.doc,
        }
    }

    /// Create a new property with a different deleter.
    pub fn with_deleter(&self, deleter: Value) -> Self {
        Self {
            flags: self.flags | PropertyFlags::HAS_DELETER,
            getter: self.getter,
            setter: self.setter,
            deleter: Some(deleter),
            doc: self.doc,
        }
    }
}

impl Descriptor for PropertyDescriptor {
    fn kind(&self) -> DescriptorKind {
        DescriptorKind::Property
    }

    fn flags(&self) -> DescriptorFlags {
        let mut flags = DescriptorFlags::empty();

        if self.has_getter() {
            flags |= DescriptorFlags::HAS_GET;
        }
        if self.has_setter() {
            flags |= DescriptorFlags::HAS_SET;
        }
        if self.has_deleter() {
            flags |= DescriptorFlags::HAS_DELETE;
        }

        // A property is a data descriptor only if it has __set__ or __delete__
        if self.has_setter() || self.has_deleter() {
            flags |= DescriptorFlags::DATA_DESCRIPTOR;
        }

        flags
    }

    fn get(&self, obj: Option<Value>, objtype: Value) -> PrismResult<Value> {
        // If accessed through class (obj is None), return the property itself
        if obj.is_none() {
            // In a full implementation, we'd return self as a Value
            // For now, return None to indicate class access
            return Ok(Value::none());
        }

        let obj = obj.unwrap();

        // Call the getter if we have one
        if let Some(_getter) = self.getter {
            // TODO: Actually call the getter function
            // For now, return a placeholder
            // In real implementation: call_function(getter, &[obj])
            let _ = obj;
            let _ = objtype;
            Ok(Value::none())
        } else {
            Err(PrismError::attribute("property has no getter"))
        }
    }

    fn set(&self, obj: Value, value: Value) -> PrismResult<()> {
        if let Some(_setter) = self.setter {
            // TODO: Actually call the setter function
            // In real implementation: call_function(setter, &[obj, value])
            let _ = (obj, value);
            Ok(())
        } else {
            Err(PrismError::attribute("property is read-only"))
        }
    }

    fn delete(&self, obj: Value) -> PrismResult<()> {
        if let Some(_deleter) = self.deleter {
            // TODO: Actually call the deleter function
            // In real implementation: call_function(deleter, &[obj])
            let _ = obj;
            Ok(())
        } else {
            Err(PrismError::attribute("property does not support deletion"))
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
    fn test_property_getter_only() {
        let getter = Value::int_unchecked(100); // Placeholder
        let prop = PropertyDescriptor::new_getter(getter);

        assert!(prop.has_getter());
        assert!(!prop.has_setter());
        assert!(!prop.has_deleter());
        assert_eq!(prop.kind(), DescriptorKind::Property);
        // A getter-only property is a NON-data descriptor per Python semantics
        assert!(!prop.is_data_descriptor());
    }

    #[test]
    fn test_property_getter_setter() {
        let getter = Value::int_unchecked(100);
        let setter = Value::int_unchecked(200);
        let prop = PropertyDescriptor::new_getter_setter(getter, setter);

        assert!(prop.has_getter());
        assert!(prop.has_setter());
        assert!(!prop.has_deleter());
    }

    #[test]
    fn test_property_full() {
        let getter = Value::int_unchecked(100);
        let setter = Value::int_unchecked(200);
        let deleter = Value::int_unchecked(300);
        let doc = Value::int_unchecked(400);

        let prop =
            PropertyDescriptor::new_full(Some(getter), Some(setter), Some(deleter), Some(doc));

        assert!(prop.has_getter());
        assert!(prop.has_setter());
        assert!(prop.has_deleter());
        assert!(prop.property_flags().contains(PropertyFlags::HAS_DOC));
    }

    #[test]
    fn test_property_flags() {
        let prop = PropertyDescriptor::new_getter(Value::int_unchecked(1));
        let flags = prop.flags();

        assert!(flags.contains(DescriptorFlags::HAS_GET));
        // Getter-only property is NOT a data descriptor
        assert!(!flags.contains(DescriptorFlags::DATA_DESCRIPTOR));
        assert!(!flags.contains(DescriptorFlags::HAS_SET));
    }

    #[test]
    fn test_property_with_setter() {
        let prop = PropertyDescriptor::new_getter(Value::int_unchecked(1));
        assert!(!prop.has_setter());

        let prop2 = prop.with_setter(Value::int_unchecked(2));
        assert!(prop2.has_getter());
        assert!(prop2.has_setter());
    }

    #[test]
    fn test_property_with_deleter() {
        let prop = PropertyDescriptor::new_getter(Value::int_unchecked(1));
        assert!(!prop.has_deleter());

        let prop2 = prop.with_deleter(Value::int_unchecked(3));
        assert!(prop2.has_getter());
        assert!(prop2.has_deleter());
    }

    #[test]
    fn test_property_get_class_access() {
        let prop = PropertyDescriptor::new_getter(Value::int_unchecked(1));
        // When accessed through class (obj=None), should return the property itself
        let result = prop.get(None, Value::none());
        assert!(result.is_ok());
    }

    #[test]
    fn test_property_set_readonly_error() {
        let prop = PropertyDescriptor::new_getter(Value::int_unchecked(1));
        let result = prop.set(Value::int_unchecked(0), Value::int_unchecked(42));
        assert!(result.is_err());
    }

    #[test]
    fn test_property_delete_unsupported_error() {
        let prop = PropertyDescriptor::new_getter(Value::int_unchecked(1));
        let result = prop.delete(Value::int_unchecked(0));
        assert!(result.is_err());
    }

    #[test]
    fn test_property_no_getter_error() {
        let prop = PropertyDescriptor::new_full(None, Some(Value::int_unchecked(1)), None, None);
        let result = prop.get(Some(Value::int_unchecked(0)), Value::none());
        assert!(result.is_err());
    }
}
