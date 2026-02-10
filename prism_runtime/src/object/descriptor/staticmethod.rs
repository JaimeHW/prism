//! StaticMethod descriptor implementation.
//!
//! A staticmethod is a regular function that happens to live in a class namespace.
//! It doesn't receive any implicit first argument (no self, no cls).
//!
//! # Example (Python)
//!
//! ```python
//! class Math:
//!     @staticmethod
//!     def add(x, y):
//!         return x + y
//!
//!     @staticmethod
//!     def is_even(n):
//!         return n % 2 == 0
//!
//! # Both work the same:
//! Math.add(1, 2)      # 3
//! Math().add(1, 2)    # 3
//! ```
//!
//! # Use Cases
//!
//! - Utility functions that logically belong to a class
//! - Functions that don't need instance or class state
//! - Organizing code into namespaces
//!
//! # Performance
//!
//! StaticMethod is the fastest descriptor type:
//! - No binding needed (returns function directly)
//! - No allocation on access
//! - Identical to calling a regular function

use super::{Descriptor, DescriptorFlags, DescriptorKind};
use prism_core::{PrismResult, Value};

// =============================================================================
// StaticMethod Descriptor
// =============================================================================

/// Descriptor for @staticmethod decorated functions.
///
/// StaticMethods return the underlying function unchanged, regardless
/// of whether accessed through a class or instance.
#[derive(Debug, Clone)]
pub struct StaticMethodDescriptor {
    /// The underlying function.
    function: Value,
}

impl StaticMethodDescriptor {
    /// Create a new staticmethod descriptor.
    pub fn new(function: Value) -> Self {
        Self { function }
    }

    /// Get the underlying function.
    #[inline]
    pub fn function(&self) -> Value {
        self.function
    }
}

impl Descriptor for StaticMethodDescriptor {
    fn kind(&self) -> DescriptorKind {
        DescriptorKind::StaticMethod
    }

    fn flags(&self) -> DescriptorFlags {
        DescriptorFlags::HAS_GET | DescriptorFlags::STATICMETHOD
    }

    fn get(&self, _obj: Option<Value>, _objtype: Value) -> PrismResult<Value> {
        // StaticMethod always returns the function unchanged
        // No binding, no wrapping - just the raw function
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
    fn test_staticmethod_creation() {
        let func = Value::int_unchecked(100);
        let sm = StaticMethodDescriptor::new(func);

        assert_eq!(sm.function(), func);
    }

    #[test]
    fn test_staticmethod_kind() {
        let sm = StaticMethodDescriptor::new(Value::int_unchecked(1));
        assert_eq!(sm.kind(), DescriptorKind::StaticMethod);
    }

    #[test]
    fn test_staticmethod_flags() {
        let sm = StaticMethodDescriptor::new(Value::int_unchecked(1));
        let flags = sm.flags();

        assert!(flags.contains(DescriptorFlags::HAS_GET));
        assert!(flags.contains(DescriptorFlags::STATICMETHOD));
        assert!(!flags.contains(DescriptorFlags::HAS_SET));
        assert!(!flags.contains(DescriptorFlags::METHOD));
        assert!(!flags.contains(DescriptorFlags::CLASSMETHOD));
    }

    #[test]
    fn test_staticmethod_is_non_data() {
        let sm = StaticMethodDescriptor::new(Value::int_unchecked(1));
        assert!(!sm.is_data_descriptor());
    }

    #[test]
    fn test_staticmethod_get_through_class() {
        let func = Value::int_unchecked(100);
        let class = Value::int_unchecked(200);
        let sm = StaticMethodDescriptor::new(func);

        // Access through class (obj=None)
        let result = sm.get(None, class);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), func);
    }

    #[test]
    fn test_staticmethod_get_through_instance() {
        let func = Value::int_unchecked(100);
        let instance = Value::int_unchecked(42);
        let class = Value::int_unchecked(200);
        let sm = StaticMethodDescriptor::new(func);

        // Access through instance (obj=instance)
        let result = sm.get(Some(instance), class);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), func);
    }

    #[test]
    fn test_staticmethod_always_returns_same_function() {
        let func = Value::int_unchecked(100);
        let sm = StaticMethodDescriptor::new(func);

        // Access multiple times - should always return same function
        assert_eq!(sm.get(None, Value::none()).unwrap(), func);
        assert_eq!(
            sm.get(Some(Value::int_unchecked(1)), Value::none())
                .unwrap(),
            func
        );
        assert_eq!(
            sm.get(Some(Value::int_unchecked(2)), Value::none())
                .unwrap(),
            func
        );
    }

    #[test]
    fn test_staticmethod_set_error() {
        let sm = StaticMethodDescriptor::new(Value::int_unchecked(1));
        let result = sm.set(Value::int_unchecked(0), Value::int_unchecked(42));
        assert!(result.is_err());
    }

    #[test]
    fn test_staticmethod_delete_error() {
        let sm = StaticMethodDescriptor::new(Value::int_unchecked(1));
        let result = sm.delete(Value::int_unchecked(0));
        assert!(result.is_err());
    }

    #[test]
    fn test_staticmethod_size() {
        // StaticMethodDescriptor should be small (just a Value = 8 bytes)
        assert_eq!(std::mem::size_of::<StaticMethodDescriptor>(), 8);
    }
}
