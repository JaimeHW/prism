//! ShapedObject - Python object with hidden class optimization.
//!
//! This module provides the core object type that uses Shape-based hidden classes
//! for O(1) property access via inline property storage.
//!
//! # Architecture
//!
//! ShapedObject combines:
//! - A shape pointer describing property layout
//! - Fixed inline slots for fast property access
//! - Optional overflow dictionary for properties beyond inline capacity
//! - Standard ObjectHeader for GC and type information
//!
//! # Performance
//!
//! Property access is O(1) when shapes match:
//! 1. Check shape ID matches cached shape
//! 2. Read/write directly from inline slot at known offset
//!
//! On shape miss, fallback to shape lookup (still fast for small objects).

use super::shape::{MAX_INLINE_SLOTS, PropertyFlags, Shape, ShapeId};
use super::{ObjectHeader, PyObject};
use crate::object::type_obj::TypeId;
use prism_core::Value;
use prism_core::intern::InternedString;
use rustc_hash::FxHashMap;
use std::sync::{Arc, LazyLock};

/// Marker payload used to represent a deleted inline attribute slot.
///
/// Deleted attributes must be distinguishable from attributes explicitly set
/// to `None`, so we store a private marker object in inline slots.
#[repr(C)]
struct DeletedPropertyMarker {
    header: ObjectHeader,
}

static DELETED_PROPERTY_MARKER_PTR: LazyLock<usize> = LazyLock::new(|| {
    let marker = DeletedPropertyMarker {
        header: ObjectHeader::new(TypeId::OBJECT),
    };
    Box::into_raw(Box::new(marker)) as usize
});

#[inline(always)]
fn deleted_property_value() -> Value {
    Value::object_ptr(*DELETED_PROPERTY_MARKER_PTR as *const ())
}

#[inline(always)]
fn is_deleted_property(value: Value) -> bool {
    value
        .as_object_ptr()
        .is_some_and(|ptr| ptr as usize == *DELETED_PROPERTY_MARKER_PTR)
}

// =============================================================================
// Inline Slots
// =============================================================================

/// Fixed-size inline storage for object properties.
///
/// Stores up to MAX_INLINE_SLOTS Values directly in the object,
/// avoiding dictionary overhead for common cases.
#[derive(Debug, Clone)]
pub struct InlineSlots {
    /// Fixed array of slots.
    slots: [Value; MAX_INLINE_SLOTS],
    /// Number of slots currently in use.
    used: u16,
}

impl Default for InlineSlots {
    fn default() -> Self {
        Self::new()
    }
}

impl InlineSlots {
    /// Create new empty inline slots.
    #[inline]
    pub fn new() -> Self {
        Self {
            // Initialize all slots to None
            slots: std::array::from_fn(|_| Value::none()),
            used: 0,
        }
    }

    /// Get a slot value by index.
    ///
    /// # Panics
    /// Panics if index >= MAX_INLINE_SLOTS.
    #[inline]
    pub fn get(&self, index: u16) -> Value {
        self.slots[index as usize]
    }

    /// Set a slot value by index.
    ///
    /// # Panics
    /// Panics if index >= MAX_INLINE_SLOTS.
    #[inline]
    pub fn set(&mut self, index: u16, value: Value) {
        let idx = index as usize;
        if idx >= self.used as usize {
            self.used = index + 1;
        }
        self.slots[idx] = value;
    }

    /// Get number of slots in use.
    #[inline]
    pub fn used(&self) -> u16 {
        self.used
    }

    /// Iterate over used slots.
    pub fn iter(&self) -> impl Iterator<Item = (u16, &Value)> {
        self.slots[..self.used as usize]
            .iter()
            .enumerate()
            .map(|(i, v)| (i as u16, v))
    }
}

// =============================================================================
// Overflow Storage
// =============================================================================

/// Storage for properties that exceed inline capacity.
///
/// Used when an object has more than MAX_INLINE_SLOTS properties.
/// Falls back to dictionary-based storage with O(1) amortized access.
#[derive(Debug, Clone, Default)]
pub struct OverflowStorage {
    /// Map from property name to value.
    properties: FxHashMap<InternedString, Value>,
}

impl OverflowStorage {
    /// Create new empty overflow storage.
    #[inline]
    pub fn new() -> Self {
        Self {
            properties: FxHashMap::default(),
        }
    }

    /// Get a property by name.
    #[inline]
    pub fn get(&self, name: &InternedString) -> Option<&Value> {
        self.properties.get(name)
    }

    /// Set a property.
    #[inline]
    pub fn set(&mut self, name: InternedString, value: Value) {
        self.properties.insert(name, value);
    }

    /// Check if a property exists.
    #[inline]
    pub fn contains(&self, name: &InternedString) -> bool {
        self.properties.contains_key(name)
    }

    /// Remove a property.
    #[inline]
    pub fn remove(&mut self, name: &InternedString) -> Option<Value> {
        self.properties.remove(name)
    }

    /// Get number of overflow properties.
    #[inline]
    pub fn len(&self) -> usize {
        self.properties.len()
    }

    /// Check if overflow is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.properties.is_empty()
    }

    /// Iterate over overflow properties.
    pub fn iter(&self) -> impl Iterator<Item = (&InternedString, &Value)> {
        self.properties.iter()
    }
}

// =============================================================================
// ShapedObject
// =============================================================================

/// A Python object with hidden class optimization.
///
/// Uses Shape-based property layout for O(1) access to inline properties.
/// Falls back to overflow dictionary for large objects.
#[repr(C)]
#[derive(Debug)]
pub struct ShapedObject {
    /// Standard object header for GC and type info.
    header: ObjectHeader,

    /// Current shape describing property layout.
    shape: Arc<Shape>,

    /// Inline property storage (fast path).
    inline_slots: InlineSlots,

    /// Overflow storage for properties beyond inline capacity.
    /// Lazily allocated only when needed.
    overflow: Option<Box<OverflowStorage>>,
}

impl ShapedObject {
    /// Create a new empty ShapedObject with the given type.
    #[inline]
    pub fn new(type_id: TypeId, empty_shape: Arc<Shape>) -> Self {
        Self {
            header: ObjectHeader::new(type_id),
            shape: empty_shape,
            inline_slots: InlineSlots::new(),
            overflow: None,
        }
    }

    /// Create a new ShapedObject with default OBJECT type.
    #[inline]
    pub fn with_empty_shape(empty_shape: Arc<Shape>) -> Self {
        Self::new(TypeId::OBJECT, empty_shape)
    }

    /// Get the current shape.
    #[inline]
    pub fn shape(&self) -> &Arc<Shape> {
        &self.shape
    }

    /// Get the shape ID for fast comparison.
    #[inline]
    pub fn shape_id(&self) -> ShapeId {
        self.shape.id()
    }

    /// Get a property by name.
    ///
    /// Returns None if the property doesn't exist.
    pub fn get_property(&self, name: &str) -> Option<Value> {
        let interned = prism_core::intern::intern(name);
        self.get_property_interned(&interned)
    }

    /// Get a property using an interned name (faster).
    #[inline]
    pub fn get_property_interned(&self, name: &InternedString) -> Option<Value> {
        // Fast path: shape lookup with interned name
        if let Some(slot_index) = self.shape.lookup_interned(name) {
            if (slot_index as usize) < MAX_INLINE_SLOTS {
                let value = self.inline_slots.get(slot_index);
                return if is_deleted_property(value) {
                    None
                } else {
                    Some(value)
                };
            }
            return self.overflow.as_ref().and_then(|overflow| overflow.get(name).copied());
        }

        // Check overflow storage
        if let Some(overflow) = &self.overflow {
            return overflow.get(name).copied();
        }

        None
    }

    /// Get a property by cached slot index (fastest).
    ///
    /// This is the IC fast path - caller must verify shape_id matches.
    #[inline]
    pub fn get_property_cached(&self, slot_index: u16) -> Value {
        debug_assert!(
            (slot_index as usize) < MAX_INLINE_SLOTS,
            "Cached access only for inline slots"
        );
        self.inline_slots.get(slot_index)
    }

    /// Set a property by name.
    ///
    /// This may transition to a new shape if the property is new.
    /// Returns the new shape if a transition occurred.
    pub fn set_property(
        &mut self,
        name: InternedString,
        value: Value,
        registry: &super::shape::ShapeRegistry,
    ) -> Option<Arc<Shape>> {
        // Check if property already exists in current shape
        if let Some(slot_index) = self.shape.lookup_interned(&name) {
            if (slot_index as usize) < MAX_INLINE_SLOTS {
                // Property exists in inline storage - just update
                self.inline_slots.set(slot_index, value);
                return None;
            }
            // Property exists in overflow storage for current shape.
            let overflow = self
                .overflow
                .get_or_insert_with(|| Box::new(OverflowStorage::new()));
            overflow.set(name, value);
            return None;
        }

        // Property is new - create transition
        let new_shape = registry.transition_default(&self.shape, name.clone());
        let slot_index = new_shape
            .property()
            .expect("New shape must have property")
            .slot_index;

        if (slot_index as usize) < MAX_INLINE_SLOTS {
            // Store in inline slot
            self.inline_slots.set(slot_index, value);
        } else {
            // Spill to overflow
            let overflow = self
                .overflow
                .get_or_insert_with(|| Box::new(OverflowStorage::new()));
            overflow.set(name, value);
        }

        self.shape = new_shape.clone();
        Some(new_shape)
    }

    /// Set a property with custom flags.
    pub fn set_property_with_flags(
        &mut self,
        name: InternedString,
        value: Value,
        flags: PropertyFlags,
        registry: &super::shape::ShapeRegistry,
    ) -> Option<Arc<Shape>> {
        // Check if property already exists
        if let Some(slot_index) = self.shape.lookup_interned(&name) {
            if (slot_index as usize) < MAX_INLINE_SLOTS {
                self.inline_slots.set(slot_index, value);
                return None;
            }
            let overflow = self
                .overflow
                .get_or_insert_with(|| Box::new(OverflowStorage::new()));
            overflow.set(name, value);
            return None;
        }

        // Create transition with custom flags
        let new_shape = registry.transition(&self.shape, name.clone(), flags);
        let slot_index = new_shape.property().unwrap().slot_index;

        if (slot_index as usize) < MAX_INLINE_SLOTS {
            self.inline_slots.set(slot_index, value);
        } else {
            let overflow = self
                .overflow
                .get_or_insert_with(|| Box::new(OverflowStorage::new()));
            overflow.set(name, value);
        }

        self.shape = new_shape.clone();
        Some(new_shape)
    }

    /// Set a property by cached slot index (fastest).
    ///
    /// Caller must verify shape_id matches and property is writable.
    #[inline]
    pub fn set_property_cached(&mut self, slot_index: u16, value: Value) {
        debug_assert!(
            (slot_index as usize) < MAX_INLINE_SLOTS,
            "Cached access only for inline slots"
        );
        self.inline_slots.set(slot_index, value);
    }

    /// Check if a property exists.
    pub fn has_property(&self, name: &str) -> bool {
        let interned = prism_core::intern::intern(name);
        self.has_property_interned(&interned)
    }

    /// Check if a property exists using an interned name.
    pub fn has_property_interned(&self, name: &InternedString) -> bool {
        if let Some(slot_index) = self.shape.lookup_interned(name) {
            if (slot_index as usize) < MAX_INLINE_SLOTS {
                return !is_deleted_property(self.inline_slots.get(slot_index));
            }
            return self
                .overflow
                .as_ref()
                .is_some_and(|overflow| overflow.contains(name));
        }
        self.overflow
            .as_ref()
            .is_some_and(|overflow| overflow.contains(name))
    }

    /// Delete a property.
    ///
    /// Note: This doesn't change the shape. Inline slots use a private tombstone
    /// marker so deletion is distinct from assigning `None`.
    /// A more sophisticated implementation could use "delete shapes" like V8.
    pub fn delete_property(&mut self, name: &str) -> bool {
        let interned = prism_core::intern::intern(name);
        self.delete_property_interned(&interned)
    }

    /// Delete a property by interned name.
    ///
    /// For inline slots we install a tombstone marker so deleted attributes are
    /// not confused with attributes explicitly set to `None`.
    pub fn delete_property_interned(&mut self, name: &InternedString) -> bool {
        if let Some(slot_index) = self.shape.lookup_interned(name) {
            if (slot_index as usize) < MAX_INLINE_SLOTS {
                let current = self.inline_slots.get(slot_index);
                if is_deleted_property(current) {
                    return false;
                }
                self.inline_slots
                    .set(slot_index, deleted_property_value());
                return true;
            }
            return self
                .overflow
                .as_mut()
                .is_some_and(|overflow| overflow.remove(name).is_some());
        }

        self.overflow
            .as_mut()
            .is_some_and(|overflow| overflow.remove(name).is_some())
    }

    /// Get all property names in definition order.
    pub fn property_names(&self) -> Vec<InternedString> {
        let mut names = Vec::new();
        for name in self.shape.property_names() {
            if let Some(slot_index) = self.shape.lookup_interned(&name) {
                if (slot_index as usize) < MAX_INLINE_SLOTS {
                    if !is_deleted_property(self.inline_slots.get(slot_index)) {
                        names.push(name);
                    }
                    continue;
                }
            }
            if self
                .overflow
                .as_ref()
                .is_some_and(|overflow| overflow.contains(&name))
            {
                names.push(name);
            }
        }

        if let Some(overflow) = &self.overflow {
            for (name, _) in overflow.iter() {
                // Only add if not already in shape (shouldn't happen, but defensive)
                if !names.iter().any(|n| n == name) {
                    names.push(name.clone());
                }
            }
        }

        names
    }

    /// Get total property count.
    pub fn property_count(&self) -> usize {
        let mut inline_count = 0usize;
        for name in self.shape.property_names() {
            if let Some(slot_index) = self.shape.lookup_interned(&name) {
                if (slot_index as usize) < MAX_INLINE_SLOTS
                    && !is_deleted_property(self.inline_slots.get(slot_index))
                {
                    inline_count += 1;
                }
            }
        }
        let overflow_count = self.overflow.as_ref().map_or(0, |o| o.len());
        inline_count + overflow_count
    }

    /// Check if object uses only inline storage.
    #[inline]
    pub fn is_inline_only(&self) -> bool {
        self.overflow.is_none() || self.overflow.as_ref().unwrap().is_empty()
    }

    /// Iterate over all properties.
    pub fn iter_properties(&self) -> impl Iterator<Item = (InternedString, Value)> + '_ {
        let shape_props = self.shape.property_names();
        let inline_iter = shape_props.into_iter().filter_map(|name| {
            if let Some(slot_index) = self.shape.lookup_interned(&name) {
                if (slot_index as usize) < MAX_INLINE_SLOTS {
                    let value = self.inline_slots.get(slot_index);
                    if !is_deleted_property(value) {
                        return Some((name, value));
                    }
                }
            }
            None
        });

        let overflow_iter = self
            .overflow
            .iter()
            .flat_map(|o| o.iter().map(|(k, v)| (k.clone(), *v)));

        inline_iter.chain(overflow_iter)
    }
}

impl PyObject for ShapedObject {
    fn header(&self) -> &ObjectHeader {
        &self.header
    }

    fn header_mut(&mut self) -> &mut ObjectHeader {
        &mut self.header
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::object::shape::ShapeRegistry;

    fn intern(s: &str) -> InternedString {
        prism_core::intern::intern(s)
    }

    fn val(i: i64) -> Value {
        Value::int(i).unwrap()
    }

    // -------------------------------------------------------------------------
    // InlineSlots Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_inline_slots_new() {
        let slots = InlineSlots::new();
        assert_eq!(slots.used(), 0);
    }

    #[test]
    fn test_inline_slots_get_set() {
        let mut slots = InlineSlots::new();
        slots.set(0, val(42));
        assert_eq!(slots.get(0), val(42));
        assert_eq!(slots.used(), 1);
    }

    #[test]
    fn test_inline_slots_multiple() {
        let mut slots = InlineSlots::new();
        slots.set(0, val(1));
        slots.set(1, val(2));
        slots.set(2, val(3));

        assert_eq!(slots.get(0), val(1));
        assert_eq!(slots.get(1), val(2));
        assert_eq!(slots.get(2), val(3));
        assert_eq!(slots.used(), 3);
    }

    #[test]
    fn test_inline_slots_overwrite() {
        let mut slots = InlineSlots::new();
        slots.set(0, val(1));
        slots.set(0, val(2));
        assert_eq!(slots.get(0), val(2));
    }

    #[test]
    fn test_inline_slots_iter() {
        let mut slots = InlineSlots::new();
        slots.set(0, val(10));
        slots.set(1, val(20));

        let collected: Vec<_> = slots.iter().collect();
        assert_eq!(collected.len(), 2);
        assert_eq!(collected[0], (0, &val(10)));
        assert_eq!(collected[1], (1, &val(20)));
    }

    // -------------------------------------------------------------------------
    // OverflowStorage Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_overflow_new() {
        let overflow = OverflowStorage::new();
        assert!(overflow.is_empty());
        assert_eq!(overflow.len(), 0);
    }

    #[test]
    fn test_overflow_get_set() {
        let mut overflow = OverflowStorage::new();
        let name = intern("prop");
        overflow.set(name.clone(), val(42));

        assert!(overflow.contains(&name));
        assert_eq!(overflow.get(&name), Some(&val(42)));
        assert_eq!(overflow.len(), 1);
    }

    #[test]
    fn test_overflow_remove() {
        let mut overflow = OverflowStorage::new();
        let name = intern("prop");
        overflow.set(name.clone(), val(42));

        let removed = overflow.remove(&name);
        assert_eq!(removed, Some(val(42)));
        assert!(!overflow.contains(&name));
    }

    // -------------------------------------------------------------------------
    // ShapedObject Tests - Basic
    // -------------------------------------------------------------------------

    #[test]
    fn test_shaped_object_new() {
        let registry = ShapeRegistry::new();
        let obj = ShapedObject::new(TypeId::OBJECT, registry.empty_shape());

        assert!(obj.shape().is_empty());
        assert!(obj.is_inline_only());
        assert_eq!(obj.property_count(), 0);
    }

    #[test]
    fn test_shaped_object_set_get() {
        let registry = ShapeRegistry::new();
        let mut obj = ShapedObject::new(TypeId::OBJECT, registry.empty_shape());

        obj.set_property(intern("x"), val(10), &registry);
        assert_eq!(obj.get_property("x"), Some(val(10)));
    }

    #[test]
    fn test_shaped_object_multiple_properties() {
        let registry = ShapeRegistry::new();
        let mut obj = ShapedObject::new(TypeId::OBJECT, registry.empty_shape());

        obj.set_property(intern("a"), val(1), &registry);
        obj.set_property(intern("b"), val(2), &registry);
        obj.set_property(intern("c"), val(3), &registry);

        assert_eq!(obj.get_property("a"), Some(val(1)));
        assert_eq!(obj.get_property("b"), Some(val(2)));
        assert_eq!(obj.get_property("c"), Some(val(3)));
        assert_eq!(obj.property_count(), 3);
    }

    #[test]
    fn test_shaped_object_update_property() {
        let registry = ShapeRegistry::new();
        let mut obj = ShapedObject::new(TypeId::OBJECT, registry.empty_shape());

        obj.set_property(intern("x"), val(10), &registry);
        let shape_before = obj.shape_id();

        obj.set_property(intern("x"), val(20), &registry);
        let shape_after = obj.shape_id();

        // Shape should not change when updating existing property
        assert_eq!(shape_before, shape_after);
        assert_eq!(obj.get_property("x"), Some(val(20)));
    }

    #[test]
    fn test_shaped_object_shape_transitions() {
        let registry = ShapeRegistry::new();
        let mut obj = ShapedObject::new(TypeId::OBJECT, registry.empty_shape());

        let shape0 = obj.shape_id();
        obj.set_property(intern("x"), val(1), &registry);
        let shape1 = obj.shape_id();
        obj.set_property(intern("y"), val(2), &registry);
        let shape2 = obj.shape_id();

        // Each new property should create a shape transition
        assert_ne!(shape0, shape1);
        assert_ne!(shape1, shape2);
    }

    // -------------------------------------------------------------------------
    // ShapedObject Tests - Inline Cache Path
    // -------------------------------------------------------------------------

    #[test]
    fn test_shaped_object_cached_access() {
        let registry = ShapeRegistry::new();
        let mut obj = ShapedObject::new(TypeId::OBJECT, registry.empty_shape());

        obj.set_property(intern("x"), val(42), &registry);

        // Simulate IC: lookup slot once, then use cached access
        let slot_index = obj.shape().lookup("x").unwrap();
        let cached_value = obj.get_property_cached(slot_index);

        assert_eq!(cached_value, val(42));
    }

    #[test]
    fn test_shaped_object_cached_set() {
        let registry = ShapeRegistry::new();
        let mut obj = ShapedObject::new(TypeId::OBJECT, registry.empty_shape());

        obj.set_property(intern("x"), val(1), &registry);
        let slot_index = obj.shape().lookup("x").unwrap();

        // Cached set (IC fast path)
        obj.set_property_cached(slot_index, val(99));

        assert_eq!(obj.get_property("x"), Some(val(99)));
    }

    #[test]
    fn test_shaped_object_interned_access() {
        let registry = ShapeRegistry::new();
        let mut obj = ShapedObject::new(TypeId::OBJECT, registry.empty_shape());

        let name = intern("prop");
        obj.set_property(name.clone(), val(100), &registry);

        assert_eq!(obj.get_property_interned(&name), Some(val(100)));
    }

    // -------------------------------------------------------------------------
    // ShapedObject Tests - Property Operations
    // -------------------------------------------------------------------------

    #[test]
    fn test_shaped_object_has_property() {
        let registry = ShapeRegistry::new();
        let mut obj = ShapedObject::new(TypeId::OBJECT, registry.empty_shape());

        assert!(!obj.has_property("x"));
        obj.set_property(intern("x"), val(1), &registry);
        assert!(obj.has_property("x"));
    }

    #[test]
    fn test_shaped_object_delete_property() {
        let registry = ShapeRegistry::new();
        let mut obj = ShapedObject::new(TypeId::OBJECT, registry.empty_shape());

        obj.set_property(intern("x"), val(1), &registry);
        assert!(obj.delete_property("x"));

        // Attribute must be absent after deletion.
        assert_eq!(obj.get_property("x"), None);
        assert!(!obj.has_property("x"));
    }

    #[test]
    fn test_shaped_object_delete_nonexistent() {
        let registry = ShapeRegistry::new();
        let mut obj = ShapedObject::new(TypeId::OBJECT, registry.empty_shape());

        assert!(!obj.delete_property("nonexistent"));
    }

    #[test]
    fn test_shaped_object_delete_twice_returns_false_second_time() {
        let registry = ShapeRegistry::new();
        let mut obj = ShapedObject::new(TypeId::OBJECT, registry.empty_shape());

        obj.set_property(intern("x"), val(1), &registry);
        assert!(obj.delete_property("x"));
        assert!(!obj.delete_property("x"));
    }

    #[test]
    fn test_shaped_object_none_value_is_not_deleted() {
        let registry = ShapeRegistry::new();
        let mut obj = ShapedObject::new(TypeId::OBJECT, registry.empty_shape());

        obj.set_property(intern("x"), Value::none(), &registry);
        assert!(obj.has_property("x"));
        assert_eq!(obj.get_property("x"), Some(Value::none()));
    }

    #[test]
    fn test_shaped_object_property_names() {
        let registry = ShapeRegistry::new();
        let mut obj = ShapedObject::new(TypeId::OBJECT, registry.empty_shape());

        obj.set_property(intern("first"), val(1), &registry);
        obj.set_property(intern("second"), val(2), &registry);
        obj.set_property(intern("third"), val(3), &registry);

        let names = obj.property_names();
        assert_eq!(names.len(), 3);
        assert_eq!(names[0].as_str(), "first");
        assert_eq!(names[1].as_str(), "second");
        assert_eq!(names[2].as_str(), "third");
    }

    #[test]
    fn test_shaped_object_property_names_exclude_deleted() {
        let registry = ShapeRegistry::new();
        let mut obj = ShapedObject::new(TypeId::OBJECT, registry.empty_shape());

        obj.set_property(intern("first"), val(1), &registry);
        obj.set_property(intern("second"), val(2), &registry);
        assert!(obj.delete_property("first"));

        let names = obj.property_names();
        assert_eq!(names.len(), 1);
        assert_eq!(names[0].as_str(), "second");
        assert_eq!(obj.property_count(), 1);
    }

    // -------------------------------------------------------------------------
    // ShapedObject Tests - Overflow Storage
    // -------------------------------------------------------------------------

    #[test]
    fn test_shaped_object_inline_limit() {
        let registry = ShapeRegistry::new();
        let mut obj = ShapedObject::new(TypeId::OBJECT, registry.empty_shape());

        // Fill inline slots
        for i in 0..MAX_INLINE_SLOTS {
            obj.set_property(intern(&format!("prop{}", i)), val(i as i64), &registry);
        }

        assert!(obj.is_inline_only());

        // Add one more to trigger overflow
        obj.set_property(
            intern(&format!("prop{}", MAX_INLINE_SLOTS)),
            val(MAX_INLINE_SLOTS as i64),
            &registry,
        );

        assert!(!obj.is_inline_only());
    }

    #[test]
    fn test_shaped_object_overflow_access() {
        let registry = ShapeRegistry::new();
        let mut obj = ShapedObject::new(TypeId::OBJECT, registry.empty_shape());

        // Fill inline + overflow
        for i in 0..(MAX_INLINE_SLOTS + 3) {
            obj.set_property(intern(&format!("prop{}", i)), val(i as i64), &registry);
        }

        // Verify all properties accessible
        for i in 0..(MAX_INLINE_SLOTS + 3) {
            assert_eq!(
                obj.get_property(&format!("prop{}", i)),
                Some(val(i as i64)),
                "Failed to get prop{}",
                i
            );
        }
    }

    #[test]
    fn test_shaped_object_readd_deleted_overflow_property_reuses_shape() {
        let registry = ShapeRegistry::new();
        let mut obj = ShapedObject::new(TypeId::OBJECT, registry.empty_shape());

        for i in 0..=MAX_INLINE_SLOTS {
            obj.set_property(intern(&format!("prop{}", i)), val(i as i64), &registry);
        }

        let overflow_name = format!("prop{}", MAX_INLINE_SLOTS);
        assert!(obj.delete_property(&overflow_name));
        assert_eq!(obj.get_property(&overflow_name), None);
        let shape_before = obj.shape_id();

        obj.set_property(intern(&overflow_name), val(999), &registry);
        let shape_after = obj.shape_id();

        assert_eq!(shape_before, shape_after);
        assert_eq!(obj.get_property(&overflow_name), Some(val(999)));
    }

    // -------------------------------------------------------------------------
    // ShapedObject Tests - Shape Sharing
    // -------------------------------------------------------------------------

    #[test]
    fn test_shaped_object_shape_sharing() {
        let registry = ShapeRegistry::new();
        let mut obj1 = ShapedObject::new(TypeId::OBJECT, registry.empty_shape());
        let mut obj2 = ShapedObject::new(TypeId::OBJECT, registry.empty_shape());

        // Add same properties in same order
        obj1.set_property(intern("x"), val(1), &registry);
        obj1.set_property(intern("y"), val(2), &registry);

        obj2.set_property(intern("x"), val(10), &registry);
        obj2.set_property(intern("y"), val(20), &registry);

        // Should share the same shape
        assert_eq!(obj1.shape_id(), obj2.shape_id());

        // But have different values
        assert_eq!(obj1.get_property("x"), Some(val(1)));
        assert_eq!(obj2.get_property("x"), Some(val(10)));
    }

    #[test]
    fn test_shaped_object_different_order_different_shape() {
        let registry = ShapeRegistry::new();
        let mut obj1 = ShapedObject::new(TypeId::OBJECT, registry.empty_shape());
        let mut obj2 = ShapedObject::new(TypeId::OBJECT, registry.empty_shape());

        // Add same properties but different order
        obj1.set_property(intern("x"), val(1), &registry);
        obj1.set_property(intern("y"), val(2), &registry);

        obj2.set_property(intern("y"), val(2), &registry);
        obj2.set_property(intern("x"), val(1), &registry);

        // Different shapes due to different order
        assert_ne!(obj1.shape_id(), obj2.shape_id());

        // But both properties work
        assert_eq!(obj1.get_property("x"), Some(val(1)));
        assert_eq!(obj2.get_property("x"), Some(val(1)));
    }

    // -------------------------------------------------------------------------
    // ShapedObject Tests - Property Flags
    // -------------------------------------------------------------------------

    #[test]
    fn test_shaped_object_property_flags() {
        let registry = ShapeRegistry::new();
        let mut obj = ShapedObject::new(TypeId::OBJECT, registry.empty_shape());

        obj.set_property_with_flags(
            intern("readonly"),
            val(42),
            PropertyFlags::read_only(),
            &registry,
        );

        // Property exists
        assert_eq!(obj.get_property("readonly"), Some(val(42)));

        // Check flags via shape
        let desc = obj.shape().get_descriptor("readonly").unwrap();
        assert!(!desc.is_writable());
        assert!(desc.is_enumerable());
    }

    #[test]
    fn test_shaped_object_hidden_property() {
        let registry = ShapeRegistry::new();
        let mut obj = ShapedObject::new(TypeId::OBJECT, registry.empty_shape());

        obj.set_property_with_flags(
            intern("_internal"),
            val(99),
            PropertyFlags::hidden(),
            &registry,
        );

        let desc = obj.shape().get_descriptor("_internal").unwrap();
        assert!(!desc.is_enumerable());
        assert!(desc.is_writable());
    }

    // -------------------------------------------------------------------------
    // ShapedObject Tests - Iterator
    // -------------------------------------------------------------------------

    #[test]
    fn test_shaped_object_iter_properties() {
        let registry = ShapeRegistry::new();
        let mut obj = ShapedObject::new(TypeId::OBJECT, registry.empty_shape());

        obj.set_property(intern("a"), val(1), &registry);
        obj.set_property(intern("b"), val(2), &registry);

        let props: Vec<_> = obj.iter_properties().collect();
        assert_eq!(props.len(), 2);
    }

    // -------------------------------------------------------------------------
    // ShapedObject Tests - Type Integration
    // -------------------------------------------------------------------------

    #[test]
    fn test_shaped_object_type_id() {
        let registry = ShapeRegistry::new();
        let obj = ShapedObject::new(TypeId::DICT, registry.empty_shape());

        assert_eq!(obj.header().type_id, TypeId::DICT);
    }

    #[test]
    fn test_shaped_object_pyobject_trait() {
        let registry = ShapeRegistry::new();
        let obj = ShapedObject::new(TypeId::OBJECT, registry.empty_shape());

        // Test PyObject trait
        assert_eq!(obj.type_id(), TypeId::OBJECT);
    }

    // -------------------------------------------------------------------------
    // Edge Cases
    // -------------------------------------------------------------------------

    #[test]
    fn test_shaped_object_empty_property_name() {
        let registry = ShapeRegistry::new();
        let mut obj = ShapedObject::new(TypeId::OBJECT, registry.empty_shape());

        obj.set_property(intern(""), val(42), &registry);
        assert_eq!(obj.get_property(""), Some(val(42)));
    }

    #[test]
    fn test_shaped_object_unicode_property() {
        let registry = ShapeRegistry::new();
        let mut obj = ShapedObject::new(TypeId::OBJECT, registry.empty_shape());

        obj.set_property(intern("名前"), val(1), &registry);
        obj.set_property(intern("🚀"), val(2), &registry);

        assert_eq!(obj.get_property("名前"), Some(val(1)));
        assert_eq!(obj.get_property("🚀"), Some(val(2)));
    }

    #[test]
    fn test_shaped_object_nonexistent_property() {
        let registry = ShapeRegistry::new();
        let obj = ShapedObject::new(TypeId::OBJECT, registry.empty_shape());

        assert_eq!(obj.get_property("nonexistent"), None);
    }
}
