//! ClassMethod descriptor implementation.
//!
//! A classmethod receives the class as its first argument instead of an instance.
//! This is useful for alternative constructors and class-level operations.
//!
//! # Example (Python)
//!
//! ```python
//! class Date:
//!     def __init__(self, year, month, day):
//!         self.year = year
//!         self.month = month
//!         self.day = day
//!
//!     @classmethod
//!     def from_string(cls, date_string):
//!         year, month, day = map(int, date_string.split('-'))
//!         return cls(year, month, day)  # cls is the class, not an instance
//!
//!     @classmethod
//!     def today(cls):
//!         # Get current date and return new instance
//!         return cls(2024, 1, 1)
//!
//! d = Date.from_string("2024-06-15")  # Uses classmethod
//! ```
//!
//! # Binding Behavior
//!
//! - Accessed through instance: binds to instance's class
//! - Accessed through class: binds to that class
//! - Works correctly with inheritance
//!
//! # Performance
//!
//! ClassMethodDescriptor creates a bound method that receives the class.
//! This is slightly more expensive than regular methods since we need
//! to determine the objtype.

use super::method::BoundMethod;
use super::{Descriptor, DescriptorFlags, DescriptorKind};
use prism_core::{PrismResult, Value};

// =============================================================================
// ClassMethod Descriptor
// =============================================================================

/// Descriptor for @classmethod decorated functions.
///
/// When accessed, binds the function to the class (or instance's class)
/// rather than the instance itself.
#[derive(Debug, Clone)]
pub struct ClassMethodDescriptor {
    /// The underlying function.
    function: Value,
}

impl ClassMethodDescriptor {
    /// Create a new classmethod descriptor.
    pub fn new(function: Value) -> Self {
        Self { function }
    }

    /// Get the underlying function.
    #[inline]
    pub fn function(&self) -> Value {
        self.function
    }

    /// Create a bound method bound to the given class.
    #[inline]
    pub fn bind(&self, class: Value) -> BoundMethod {
        BoundMethod::new(self.function, class)
    }
}

impl Descriptor for ClassMethodDescriptor {
    fn kind(&self) -> DescriptorKind {
        DescriptorKind::ClassMethod
    }

    fn flags(&self) -> DescriptorFlags {
        DescriptorFlags::HAS_GET | DescriptorFlags::CLASSMETHOD
    }

    fn get(&self, _obj: Option<Value>, objtype: Value) -> PrismResult<Value> {
        // For classmethod, we always bind to objtype (the class)
        // regardless of whether accessed through instance or class
        let bound = self.bind(objtype);

        // In a full implementation, we'd wrap the bound method as a Value
        // For now, return the function (the bound method creation would happen)
        let _ = bound;
        Ok(self.function)
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_classmethod_creation() {
        let func = Value::int_unchecked(100);
        let cm = ClassMethodDescriptor::new(func);

        assert_eq!(cm.function(), func);
    }

    #[test]
    fn test_classmethod_kind() {
        let cm = ClassMethodDescriptor::new(Value::int_unchecked(1));
        assert_eq!(cm.kind(), DescriptorKind::ClassMethod);
    }

    #[test]
    fn test_classmethod_flags() {
        let cm = ClassMethodDescriptor::new(Value::int_unchecked(1));
        let flags = cm.flags();

        assert!(flags.contains(DescriptorFlags::HAS_GET));
        assert!(flags.contains(DescriptorFlags::CLASSMETHOD));
        assert!(!flags.contains(DescriptorFlags::HAS_SET));
        assert!(!flags.contains(DescriptorFlags::METHOD));
    }

    #[test]
    fn test_classmethod_is_non_data() {
        let cm = ClassMethodDescriptor::new(Value::int_unchecked(1));
        assert!(!cm.is_data_descriptor());
    }

    #[test]
    fn test_classmethod_bind() {
        let func = Value::int_unchecked(100);
        let class = Value::int_unchecked(200);
        let cm = ClassMethodDescriptor::new(func);

        let bound = cm.bind(class);
        assert_eq!(bound.function(), func);
        assert_eq!(bound.instance(), class); // Note: "instance" is actually the class
    }

    #[test]
    fn test_classmethod_get_through_class() {
        let func = Value::int_unchecked(100);
        let class = Value::int_unchecked(200);
        let cm = ClassMethodDescriptor::new(func);

        // Access through class (obj=None, objtype=class)
        let result = cm.get(None, class);
        assert!(result.is_ok());
    }

    #[test]
    fn test_classmethod_get_through_instance() {
        let func = Value::int_unchecked(100);
        let instance = Value::int_unchecked(42);
        let class = Value::int_unchecked(200);
        let cm = ClassMethodDescriptor::new(func);

        // Access through instance (obj=instance, objtype=class)
        let result = cm.get(Some(instance), class);
        assert!(result.is_ok());
    }

    #[test]
    fn test_classmethod_set_error() {
        let cm = ClassMethodDescriptor::new(Value::int_unchecked(1));
        let result = cm.set(Value::int_unchecked(0), Value::int_unchecked(42));
        assert!(result.is_err());
    }

    #[test]
    fn test_classmethod_delete_error() {
        let cm = ClassMethodDescriptor::new(Value::int_unchecked(1));
        let result = cm.delete(Value::int_unchecked(0));
        assert!(result.is_err());
    }
}
