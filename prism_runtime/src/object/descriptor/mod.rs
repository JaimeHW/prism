//! Python descriptor protocol implementation.
//!
//! The descriptor protocol is the mechanism behind Python's attribute access,
//! enabling properties, bound methods, class methods, static methods, and slots.
//!
//! # Descriptor Types
//!
//! Python defines three categories of descriptors:
//!
//! 1. **Data Descriptors**: Define both `__get__` and `__set__` (or `__delete__`).
//!    Data descriptors take precedence over instance attributes.
//!
//! 2. **Non-Data Descriptors**: Define only `__get__`.
//!    Instance attributes take precedence over non-data descriptors.
//!
//! 3. **Slots**: Optimized fixed-offset storage for instance attributes.
//!
//! # Attribute Lookup Order
//!
//! 1. Class `__mro__` for data descriptors
//! 2. Instance `__dict__` (if not a data descriptor)
//! 3. Class `__mro__` for non-data descriptors
//! 4. Raise `AttributeError`
//!
//! # Performance Optimizations
//!
//! - Slot-based descriptors use fixed offsets for O(1) access
//! - Descriptor type is encoded in flags for fast dispatch
//! - Method binding is cached via bound method objects
//! - Hot paths are inlined and branch-free where possible
//!
//! # Architecture
//!
//! ```text
//! Descriptor (trait)
//! ├── PropertyDescriptor    - @property, getter/setter/deleter
//! ├── SlotDescriptor        - __slots__ member access
//! ├── MethodDescriptor      - Bound method creation
//! ├── ClassMethodDescriptor - @classmethod
//! ├── StaticMethodDescriptor - @staticmethod
//! └── WrapperDescriptor     - Built-in type methods
//! ```

mod classmethod;
mod method;
mod property;
mod slot;
mod staticmethod;

pub use classmethod::ClassMethodDescriptor;
pub use method::{BoundMethod, MethodDescriptor};
pub use property::{PropertyDescriptor, PropertyFlags};
pub use slot::{SlotAccess, SlotDescriptor};
pub use staticmethod::StaticMethodDescriptor;

use prism_core::{PrismResult, Value};
use std::fmt::Debug;

// =============================================================================
// Descriptor Flags
// =============================================================================

bitflags::bitflags! {
    /// Flags describing descriptor capabilities.
    ///
    /// These flags enable fast dispatch without virtual calls in hot paths.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub struct DescriptorFlags: u8 {
        /// Descriptor has `__get__`.
        const HAS_GET = 1 << 0;
        /// Descriptor has `__set__`.
        const HAS_SET = 1 << 1;
        /// Descriptor has `__delete__`.
        const HAS_DELETE = 1 << 2;
        /// This is a data descriptor (HAS_SET or HAS_DELETE).
        const DATA_DESCRIPTOR = 1 << 3;
        /// This is a slot descriptor (optimized fixed offset).
        const SLOT = 1 << 4;
        /// This is a method (creates bound method on access).
        const METHOD = 1 << 5;
        /// This is a classmethod.
        const CLASSMETHOD = 1 << 6;
        /// This is a staticmethod.
        const STATICMETHOD = 1 << 7;
    }
}

impl Default for DescriptorFlags {
    fn default() -> Self {
        Self::empty()
    }
}

// =============================================================================
// Descriptor Kind
// =============================================================================

/// Enumeration of descriptor types for fast dispatch.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum DescriptorKind {
    /// Regular attribute (not a descriptor).
    None = 0,
    /// Property descriptor with getter/setter.
    Property = 1,
    /// Slot descriptor with fixed offset.
    Slot = 2,
    /// Function that creates bound method.
    Method = 3,
    /// Class method descriptor.
    ClassMethod = 4,
    /// Static method descriptor.
    StaticMethod = 5,
    /// Built-in wrapper descriptor.
    Wrapper = 6,
}

impl DescriptorKind {
    /// Check if this is a data descriptor.
    #[inline]
    pub const fn is_data_descriptor(self) -> bool {
        matches!(self, DescriptorKind::Property | DescriptorKind::Slot)
    }

    /// Check if this binds to instances.
    #[inline]
    pub const fn binds_to_instance(self) -> bool {
        matches!(self, DescriptorKind::Method | DescriptorKind::Property)
    }
}

// =============================================================================
// Descriptor Trait
// =============================================================================

/// Core descriptor protocol trait.
///
/// This trait defines the three fundamental descriptor operations that Python
/// uses for attribute access on class instances.
///
/// # Performance
///
/// Implementations should be as fast as possible since attribute access is
/// extremely common. The default implementations allow partial descriptor
/// implementations (e.g., read-only properties).
pub trait Descriptor: Debug + Send + Sync {
    /// Get the descriptor kind for fast dispatch.
    fn kind(&self) -> DescriptorKind;

    /// Get descriptor flags.
    fn flags(&self) -> DescriptorFlags;

    /// `__get__(self, obj, objtype)` - Called when attribute is accessed.
    ///
    /// # Arguments
    ///
    /// * `obj` - The instance the attribute was accessed through, or None if
    ///           accessed through the class.
    /// * `objtype` - The type of the instance, or the class if accessed through class.
    ///
    /// # Returns
    ///
    /// The attribute value, or an error (typically AttributeError).
    fn get(&self, obj: Option<Value>, objtype: Value) -> PrismResult<Value>;

    /// `__set__(self, obj, value)` - Called when attribute is assigned.
    ///
    /// # Arguments
    ///
    /// * `obj` - The instance the attribute was assigned on.
    /// * `value` - The value being assigned.
    ///
    /// # Returns
    ///
    /// Ok(()) on success, or an error (typically AttributeError for read-only).
    fn set(&self, obj: Value, value: Value) -> PrismResult<()> {
        let _ = (obj, value);
        Err(prism_core::PrismError::attribute("attribute is read-only"))
    }

    /// `__delete__(self, obj)` - Called when attribute is deleted.
    ///
    /// # Arguments
    ///
    /// * `obj` - The instance the attribute was deleted from.
    ///
    /// # Returns
    ///
    /// Ok(()) on success, or an error.
    fn delete(&self, obj: Value) -> PrismResult<()> {
        let _ = obj;
        Err(prism_core::PrismError::attribute(
            "attribute cannot be deleted",
        ))
    }

    /// Check if this is a data descriptor (has __set__ or __delete__).
    #[inline]
    fn is_data_descriptor(&self) -> bool {
        self.flags()
            .intersects(DescriptorFlags::HAS_SET | DescriptorFlags::HAS_DELETE)
    }
}

// =============================================================================
// Descriptor Result
// =============================================================================

/// Result of descriptor lookup during attribute resolution.
#[derive(Debug, Clone)]
pub enum DescriptorLookup<'a> {
    /// Found a data descriptor (takes precedence over instance dict).
    DataDescriptor(&'a dyn Descriptor),
    /// Found a non-data descriptor (instance dict takes precedence).
    NonDataDescriptor(&'a dyn Descriptor),
    /// Found a regular value (not a descriptor).
    Value(Value),
    /// Nothing found.
    NotFound,
}

// =============================================================================
// Descriptor Object Wrapper
// =============================================================================

/// Boxed descriptor for storage in class dictionaries.
///
/// This wrapper allows storing different descriptor types uniformly
/// while preserving fast access to descriptor flags.
#[derive(Debug)]
pub struct DescriptorObject {
    /// Cached flags for fast dispatch.
    flags: DescriptorFlags,
    /// The actual descriptor implementation.
    inner: Box<dyn Descriptor>,
}

impl DescriptorObject {
    /// Create a new descriptor object.
    pub fn new<D: Descriptor + 'static>(descriptor: D) -> Self {
        let flags = descriptor.flags();
        Self {
            flags,
            inner: Box::new(descriptor),
        }
    }

    /// Get cached flags for fast dispatch.
    #[inline]
    pub fn flags(&self) -> DescriptorFlags {
        self.flags
    }

    /// Get the descriptor kind.
    #[inline]
    pub fn kind(&self) -> DescriptorKind {
        self.inner.kind()
    }

    /// Check if this is a data descriptor.
    #[inline]
    pub fn is_data_descriptor(&self) -> bool {
        self.flags.contains(DescriptorFlags::DATA_DESCRIPTOR)
    }

    /// Get attribute value.
    #[inline]
    pub fn get(&self, obj: Option<Value>, objtype: Value) -> PrismResult<Value> {
        self.inner.get(obj, objtype)
    }

    /// Set attribute value.
    #[inline]
    pub fn set(&self, obj: Value, value: Value) -> PrismResult<()> {
        self.inner.set(obj, value)
    }

    /// Delete attribute.
    #[inline]
    pub fn delete(&self, obj: Value) -> PrismResult<()> {
        self.inner.delete(obj)
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_descriptor_flags_data_descriptor() {
        let flags = DescriptorFlags::HAS_GET | DescriptorFlags::HAS_SET;
        assert!(flags.contains(DescriptorFlags::HAS_GET));
        assert!(flags.contains(DescriptorFlags::HAS_SET));
        assert!(flags.intersects(DescriptorFlags::HAS_SET | DescriptorFlags::HAS_DELETE));
    }

    #[test]
    fn test_descriptor_flags_non_data_descriptor() {
        let flags = DescriptorFlags::HAS_GET;
        assert!(flags.contains(DescriptorFlags::HAS_GET));
        assert!(!flags.contains(DescriptorFlags::HAS_SET));
        assert!(!flags.intersects(DescriptorFlags::HAS_SET | DescriptorFlags::HAS_DELETE));
    }

    #[test]
    fn test_descriptor_kind_is_data() {
        assert!(DescriptorKind::Property.is_data_descriptor());
        assert!(DescriptorKind::Slot.is_data_descriptor());
        assert!(!DescriptorKind::Method.is_data_descriptor());
        assert!(!DescriptorKind::StaticMethod.is_data_descriptor());
    }

    #[test]
    fn test_descriptor_kind_binds() {
        assert!(DescriptorKind::Method.binds_to_instance());
        assert!(DescriptorKind::Property.binds_to_instance());
        assert!(!DescriptorKind::StaticMethod.binds_to_instance());
        assert!(!DescriptorKind::ClassMethod.binds_to_instance());
    }
}
