//! Python metaclass support.
//!
//! This module provides the metaclass infrastructure for Python class creation:
//! - `TypeMetaclass` - The default `type` metaclass for all classes
//! - `Metaclass` trait - Interface for custom metaclass implementations
//! - `MetaclassCall` - Protocol for `cls(...)` invocation
//!
//! # Metaclass Protocol
//!
//! When a class is defined:
//! ```python
//! class MyClass(Base, metaclass=Meta):
//!     ...
//! ```
//!
//! The metaclass is called as:
//! ```python
//! MyClass = Meta('MyClass', (Base,), {'__module__': ...})
//! ```
//!
//! This is equivalent to:
//! ```python
//! __class = Meta.__call__('MyClass', (Base,), namespace)
//! # Which internally does:
//! #   __class = Meta.__new__(Meta, 'MyClass', (Base,), namespace)
//! #   Meta.__init__(__class, 'MyClass', (Base,), namespace)
//! ```
//!
//! # Performance Optimizations
//!
//! - TypeId-based metaclass identification for O(1) dispatch
//! - Cached metaclass call slots for JIT specialization
//! - Lock-free metaclass resolution in common cases

use crate::object::class::{ClassDict, ClassFlags, InstantiationHint, PyClassObject};
use crate::object::mro::{ClassId, Mro};
use crate::object::{ObjectHeader, TypeId};
use prism_core::intern::{InternedString, intern};
use rustc_hash::FxHashMap;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Arc, RwLock};

// =============================================================================
// Metaclass TypeIds
// =============================================================================

/// Reserved TypeId for the `type` metaclass.
pub const TYPE_METACLASS_ID: TypeId = TypeId::TYPE;

/// Reserved TypeId for abstract base classes.
pub const ABC_META_ID: u32 = TypeId::FIRST_USER_TYPE - 1;

// =============================================================================
// Metaclass Errors
// =============================================================================

/// Errors that can occur during metaclass operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MetaclassError {
    /// Metaclass conflict in bases.
    Conflict {
        /// First conflicting metaclass.
        meta1: TypeId,
        /// Second conflicting metaclass.
        meta2: TypeId,
    },

    /// Could not resolve metaclass from bases.
    ResolutionFailed {
        /// Error message.
        message: String,
    },

    /// Metaclass __new__ failed.
    NewFailed {
        /// Error message.
        message: String,
    },

    /// Metaclass __init__ failed.
    InitFailed {
        /// Error message.
        message: String,
    },

    /// Metaclass __call__ failed.
    CallFailed {
        /// Error message.
        message: String,
    },

    /// Invalid metaclass type.
    InvalidMetaclass {
        /// TypeId of the invalid metaclass.
        type_id: TypeId,
    },
}

impl std::fmt::Display for MetaclassError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Conflict { meta1, meta2 } => {
                write!(
                    f,
                    "metaclass conflict: {} is not a subclass of {}",
                    meta1.raw(),
                    meta2.raw()
                )
            }
            Self::ResolutionFailed { message } => {
                write!(f, "metaclass resolution failed: {}", message)
            }
            Self::NewFailed { message } => {
                write!(f, "metaclass __new__ failed: {}", message)
            }
            Self::InitFailed { message } => {
                write!(f, "metaclass __init__ failed: {}", message)
            }
            Self::CallFailed { message } => {
                write!(f, "metaclass __call__ failed: {}", message)
            }
            Self::InvalidMetaclass { type_id } => {
                write!(f, "invalid metaclass: TypeId({})", type_id.raw())
            }
        }
    }
}

impl std::error::Error for MetaclassError {}

/// Result type for metaclass operations.
pub type MetaclassResult<T> = Result<T, MetaclassError>;

// =============================================================================
// Metaclass Trait
// =============================================================================

/// Trait for metaclass implementations.
///
/// A metaclass controls class creation. The default metaclass is `type`.
/// Custom metaclasses can override `__new__`, `__init__`, and `__call__`.
///
/// # Thread Safety
///
/// Implementations must be Send + Sync for use across threads.
pub trait Metaclass: Send + Sync + std::fmt::Debug {
    /// Get the TypeId of this metaclass.
    fn type_id(&self) -> TypeId;

    /// Get the name of this metaclass.
    fn name(&self) -> InternedString;

    /// Create a new class using this metaclass.
    ///
    /// This is the main entry point called when a class is defined.
    /// Protocol: `__call__(name, bases, namespace)`.
    fn __call__(
        &self,
        name: InternedString,
        bases: &[ClassId],
        namespace: ClassDict,
        registry: &dyn MetaclassRegistry,
    ) -> MetaclassResult<Arc<PyClassObject>>;

    /// Allocate and return a new class object.
    ///
    /// This is called first to create the class object structure.
    /// The class is not yet initialized at this point.
    fn __new__(
        &self,
        name: InternedString,
        bases: &[ClassId],
        namespace: &ClassDict,
        registry: &dyn MetaclassRegistry,
    ) -> MetaclassResult<Arc<PyClassObject>>;

    /// Initialize the class after creation.
    ///
    /// This is called after `__new__` to populate the class with
    /// methods and attributes from the namespace.
    fn __init__(
        &self,
        class: &PyClassObject,
        namespace: ClassDict,
        registry: &dyn MetaclassRegistry,
    ) -> MetaclassResult<()>;

    /// Check if this metaclass is a subclass of another.
    ///
    /// Used for metaclass conflict resolution.
    fn is_subclass_of(&self, other: TypeId) -> bool;

    /// Get the instantiation hint for classes created by this metaclass.
    fn instantiation_hint(&self) -> InstantiationHint {
        InstantiationHint::Generic
    }
}

// =============================================================================
// Metaclass Registry Trait
// =============================================================================

/// Registry for looking up metaclasses and classes.
///
/// This is passed to metaclass methods to allow class lookups during
/// class creation (e.g., for MRO computation).
pub trait MetaclassRegistry: Send + Sync {
    /// Get a class by its ClassId.
    fn get_class(&self, class_id: ClassId) -> Option<Arc<PyClassObject>>;

    /// Get MRO for a class.
    fn get_mro(&self, class_id: ClassId) -> Option<Mro>;

    /// Get metaclass by TypeId.
    fn get_metaclass(&self, type_id: TypeId) -> Option<Arc<dyn Metaclass>>;

    /// Get the default metaclass (TypeMetaclass).
    fn get_type_metaclass(&self) -> Arc<dyn Metaclass>;

    /// Register a new class.
    fn register_class(&self, class: Arc<PyClassObject>);
}

// =============================================================================
// TypeMetaclass - The Default `type` Metaclass
// =============================================================================

/// The default `type` metaclass.
///
/// This is the metaclass used when no explicit metaclass is specified.
/// It implements the standard Python class creation protocol.
///
/// # Example
///
/// ```python
/// class MyClass:
///     pass
///
/// # Equivalent to:
/// # MyClass = type('MyClass', (), {})
/// ```
#[derive(Debug)]
pub struct TypeMetaclass {
    /// Object header (metaclasses are themselves objects).
    header: ObjectHeader,

    /// The `type` type's own TypeId.
    type_id: TypeId,

    /// Number of classes created by this metaclass.
    classes_created: AtomicU32,
}

impl TypeMetaclass {
    /// Well-known name for the type metaclass.
    pub const NAME: &'static str = "type";

    /// Create the singleton type metaclass.
    pub fn new() -> Self {
        Self {
            header: ObjectHeader::new(TYPE_METACLASS_ID),
            type_id: TYPE_METACLASS_ID,
            classes_created: AtomicU32::new(0),
        }
    }

    /// Get the number of classes created by this metaclass.
    pub fn classes_created(&self) -> u32 {
        self.classes_created.load(Ordering::Relaxed)
    }

    /// Compute flags from namespace.
    fn compute_flags(&self, namespace: &ClassDict) -> ClassFlags {
        let mut flags = ClassFlags::empty();

        // Check for special methods in namespace
        if namespace.contains(&intern("__new__")) {
            flags |= ClassFlags::HAS_NEW;
        }
        if namespace.contains(&intern("__init__")) {
            flags |= ClassFlags::HAS_INIT;
        }
        if namespace.contains(&intern("__del__")) {
            flags |= ClassFlags::HAS_FINALIZER;
        }
        if namespace.contains(&intern("__hash__")) {
            flags |= ClassFlags::HASHABLE;
        }
        if namespace.contains(&intern("__eq__")) {
            flags |= ClassFlags::HAS_EQ;
        }
        if namespace.contains(&intern("__slots__")) {
            flags |= ClassFlags::HAS_SLOTS;
        }
        // Note: HAS_DICT and HAS_WEAKREF flags could be added when needed

        flags
    }

    /// Determine instantiation hint from flags and namespace.
    fn compute_instantiation_hint(
        &self,
        flags: ClassFlags,
        _namespace: &ClassDict,
    ) -> InstantiationHint {
        let has_custom_new = flags.contains(ClassFlags::HAS_NEW);
        let has_custom_init = flags.contains(ClassFlags::HAS_INIT);
        let has_slots = flags.contains(ClassFlags::HAS_SLOTS);

        match (has_custom_new, has_custom_init, has_slots) {
            // Fast path: no custom methods, no __slots__
            (false, false, false) => InstantiationHint::InlineSlots,
            // Has __slots__ but no custom methods
            (false, false, true) => InstantiationHint::FixedSlots,
            // Has __init__ only (common case)
            (false, true, _) => InstantiationHint::DefaultInit,
            // Has custom __new__ (needs full protocol)
            (true, _, _) => InstantiationHint::Generic,
        }
    }
}

impl Default for TypeMetaclass {
    fn default() -> Self {
        Self::new()
    }
}

impl Metaclass for TypeMetaclass {
    fn type_id(&self) -> TypeId {
        self.type_id
    }

    fn name(&self) -> InternedString {
        intern(Self::NAME)
    }

    fn __call__(
        &self,
        name: InternedString,
        bases: &[ClassId],
        namespace: ClassDict,
        registry: &dyn MetaclassRegistry,
    ) -> MetaclassResult<Arc<PyClassObject>> {
        // 1. Call __new__ to create class object
        let class = self.__new__(name, bases, &namespace, registry)?;

        // 2. Call __init__ to initialize class
        self.__init__(&class, namespace, registry)?;

        // 3. Track statistics
        self.classes_created.fetch_add(1, Ordering::Relaxed);

        Ok(class)
    }

    fn __new__(
        &self,
        name: InternedString,
        bases: &[ClassId],
        namespace: &ClassDict,
        registry: &dyn MetaclassRegistry,
    ) -> MetaclassResult<Arc<PyClassObject>> {
        // Create mro_lookup closure
        let mro_lookup = |id: ClassId| -> Option<Mro> { registry.get_mro(id) };

        // Create the class object
        let class = PyClassObject::new(name.clone(), bases, mro_lookup).map_err(|e| {
            MetaclassError::NewFailed {
                message: format!(
                    "MRO computation failed for class '{}': {:?}",
                    name.as_str(),
                    e
                ),
            }
        })?;

        Ok(Arc::new(class))
    }

    fn __init__(
        &self,
        _class: &PyClassObject,
        namespace: ClassDict,
        _registry: &dyn MetaclassRegistry,
    ) -> MetaclassResult<()> {
        // Compute flags from namespace (for validation/analysis)
        let _flags = self.compute_flags(&namespace);

        // Note: The class is already constructed with the namespace by PyClassObject::new.
        // Additional initialization (setting flags, copying namespace) would be done
        // via PyClassObject methods that can be added later when needed.
        // For now, the class is fully usable after __new__.

        Ok(())
    }

    fn is_subclass_of(&self, other: TypeId) -> bool {
        // TypeMetaclass is only subclass of itself
        other == TYPE_METACLASS_ID
    }

    fn instantiation_hint(&self) -> InstantiationHint {
        InstantiationHint::Generic
    }
}

// =============================================================================
// Metaclass Resolver
// =============================================================================

/// Resolves the winning metaclass from bases.
///
/// When a class has multiple bases with different metaclasses,
/// the winning metaclass must be a subclass of all other metaclasses.
///
/// # Algorithm
///
/// 1. Start with the explicit metaclass (or `type` if none)
/// 2. For each base, check if its metaclass is more derived
/// 3. If more derived, use it as the winner
/// 4. If neither is a subclass of the other, raise conflict
#[derive(Debug)]
pub struct MetaclassResolver;

impl MetaclassResolver {
    /// Resolve the winning metaclass from bases.
    ///
    /// # Arguments
    ///
    /// * `explicit` - Explicitly specified metaclass (if any)
    /// * `bases` - Base classes
    /// * `get_metaclass` - Function to get metaclass for a class
    ///
    /// # Returns
    ///
    /// The TypeId of the winning metaclass.
    pub fn resolve<F>(
        explicit: Option<TypeId>,
        bases: &[ClassId],
        get_metaclass: F,
    ) -> MetaclassResult<TypeId>
    where
        F: Fn(ClassId) -> TypeId,
    {
        // Start with explicit or default to type
        let mut winner = explicit.unwrap_or(TYPE_METACLASS_ID);

        // Check each base's metaclass
        for &base in bases {
            let base_meta = get_metaclass(base);

            // Check if base_meta is more derived
            if base_meta != winner {
                // In a full implementation, we'd check subclass relations.
                // For now, require explicit metaclass to be compatible.
                if explicit.is_none() && base_meta != TYPE_METACLASS_ID {
                    // Base has custom metaclass, use it
                    winner = base_meta;
                } else if explicit.is_some() && base_meta != TYPE_METACLASS_ID {
                    // Potential conflict - would need subclass check
                    // For simplicity, we allow if base_meta == winner
                    if base_meta != winner {
                        return Err(MetaclassError::Conflict {
                            meta1: winner,
                            meta2: base_meta,
                        });
                    }
                }
            }
        }

        Ok(winner)
    }
}

// =============================================================================
// Metaclass Cache
// =============================================================================

/// Cache for metaclass instances.
///
/// Provides fast lookup of metaclasses by TypeId.
/// Thread-safe with RwLock for concurrent access.
#[derive(Debug, Default)]
pub struct MetaclassCache {
    /// Cached metaclass instances.
    cache: RwLock<FxHashMap<TypeId, Arc<dyn Metaclass>>>,

    /// Hit count for profiling.
    hits: AtomicU32,

    /// Miss count for profiling.
    misses: AtomicU32,
}

impl MetaclassCache {
    /// Create a new empty cache.
    pub fn new() -> Self {
        Self {
            cache: RwLock::new(FxHashMap::default()),
            hits: AtomicU32::new(0),
            misses: AtomicU32::new(0),
        }
    }

    /// Create a cache with the default type metaclass pre-registered.
    pub fn with_type_metaclass(type_meta: Arc<TypeMetaclass>) -> Self {
        let mut cache = FxHashMap::default();
        cache.insert(TYPE_METACLASS_ID, type_meta as Arc<dyn Metaclass>);
        Self {
            cache: RwLock::new(cache),
            hits: AtomicU32::new(0),
            misses: AtomicU32::new(0),
        }
    }

    /// Get a metaclass by TypeId.
    pub fn get(&self, type_id: TypeId) -> Option<Arc<dyn Metaclass>> {
        let cache = self.cache.read().unwrap();
        if let Some(meta) = cache.get(&type_id) {
            self.hits.fetch_add(1, Ordering::Relaxed);
            Some(Arc::clone(meta))
        } else {
            self.misses.fetch_add(1, Ordering::Relaxed);
            None
        }
    }

    /// Register a metaclass.
    pub fn register(&self, metaclass: Arc<dyn Metaclass>) {
        let mut cache = self.cache.write().unwrap();
        cache.insert(metaclass.type_id(), metaclass);
    }

    /// Check if a metaclass is registered.
    pub fn contains(&self, type_id: TypeId) -> bool {
        let cache = self.cache.read().unwrap();
        cache.contains_key(&type_id)
    }

    /// Get cache statistics.
    pub fn stats(&self) -> (u32, u32) {
        (
            self.hits.load(Ordering::Relaxed),
            self.misses.load(Ordering::Relaxed),
        )
    }

    /// Clear the cache.
    pub fn clear(&self) {
        let mut cache = self.cache.write().unwrap();
        cache.clear();
    }

    /// Number of cached metaclasses.
    pub fn len(&self) -> usize {
        let cache = self.cache.read().unwrap();
        cache.len()
    }

    /// Check if cache is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

// =============================================================================
// Class Factory
// =============================================================================

/// Factory for creating classes using the metaclass protocol.
///
/// This is the high-level API for class creation, handling metaclass
/// resolution and invocation.
#[derive(Debug)]
pub struct ClassFactory {
    /// Metaclass cache.
    metaclass_cache: MetaclassCache,

    /// The default type metaclass.
    type_metaclass: Arc<TypeMetaclass>,

    /// Classes created count.
    classes_created: AtomicU32,
}

impl ClassFactory {
    /// Create a new class factory.
    pub fn new() -> Self {
        let type_metaclass = Arc::new(TypeMetaclass::new());
        let metaclass_cache = MetaclassCache::with_type_metaclass(Arc::clone(&type_metaclass));
        Self {
            metaclass_cache,
            type_metaclass,
            classes_created: AtomicU32::new(0),
        }
    }

    /// Create a class using the default metaclass.
    ///
    /// This is the fast path for classes without explicit metaclass.
    pub fn create_class(
        &self,
        name: InternedString,
        bases: &[ClassId],
        namespace: ClassDict,
        registry: &dyn MetaclassRegistry,
    ) -> MetaclassResult<Arc<PyClassObject>> {
        let class = self
            .type_metaclass
            .__call__(name, bases, namespace, registry)?;
        self.classes_created.fetch_add(1, Ordering::Relaxed);
        Ok(class)
    }

    /// Create a class with explicit metaclass.
    pub fn create_class_with_metaclass(
        &self,
        name: InternedString,
        bases: &[ClassId],
        namespace: ClassDict,
        metaclass_id: TypeId,
        registry: &dyn MetaclassRegistry,
    ) -> MetaclassResult<Arc<PyClassObject>> {
        // Look up metaclass
        let metaclass =
            self.metaclass_cache
                .get(metaclass_id)
                .ok_or(MetaclassError::InvalidMetaclass {
                    type_id: metaclass_id,
                })?;

        let class = metaclass.__call__(name, bases, namespace, registry)?;
        self.classes_created.fetch_add(1, Ordering::Relaxed);
        Ok(class)
    }

    /// Get the default type metaclass.
    pub fn type_metaclass(&self) -> &Arc<TypeMetaclass> {
        &self.type_metaclass
    }

    /// Register a custom metaclass.
    pub fn register_metaclass(&self, metaclass: Arc<dyn Metaclass>) {
        self.metaclass_cache.register(metaclass);
    }

    /// Get a metaclass by TypeId.
    pub fn get_metaclass(&self, type_id: TypeId) -> Option<Arc<dyn Metaclass>> {
        self.metaclass_cache.get(type_id)
    }

    /// Get the number of classes created.
    pub fn classes_created(&self) -> u32 {
        self.classes_created.load(Ordering::Relaxed)
    }

    /// Get cache statistics.
    pub fn cache_stats(&self) -> (u32, u32) {
        self.metaclass_cache.stats()
    }
}

impl Default for ClassFactory {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use prism_core::Value;

    // =========================================================================
    // MetaclassError Tests
    // =========================================================================

    #[test]
    fn test_metaclass_error_display_conflict() {
        let err = MetaclassError::Conflict {
            meta1: TypeId::from_raw(100),
            meta2: TypeId::from_raw(200),
        };
        let msg = format!("{}", err);
        assert!(msg.contains("metaclass conflict"));
        assert!(msg.contains("100"));
        assert!(msg.contains("200"));
    }

    #[test]
    fn test_metaclass_error_display_resolution_failed() {
        let err = MetaclassError::ResolutionFailed {
            message: "test error".to_string(),
        };
        let msg = format!("{}", err);
        assert!(msg.contains("resolution failed"));
        assert!(msg.contains("test error"));
    }

    #[test]
    fn test_metaclass_error_display_new_failed() {
        let err = MetaclassError::NewFailed {
            message: "__new__ error".to_string(),
        };
        let msg = format!("{}", err);
        assert!(msg.contains("__new__ failed"));
    }

    #[test]
    fn test_metaclass_error_display_init_failed() {
        let err = MetaclassError::InitFailed {
            message: "__init__ error".to_string(),
        };
        let msg = format!("{}", err);
        assert!(msg.contains("__init__ failed"));
    }

    #[test]
    fn test_metaclass_error_display_call_failed() {
        let err = MetaclassError::CallFailed {
            message: "__call__ error".to_string(),
        };
        let msg = format!("{}", err);
        assert!(msg.contains("__call__ failed"));
    }

    #[test]
    fn test_metaclass_error_display_invalid() {
        let err = MetaclassError::InvalidMetaclass {
            type_id: TypeId::from_raw(42),
        };
        let msg = format!("{}", err);
        assert!(msg.contains("invalid metaclass"));
        assert!(msg.contains("42"));
    }

    // =========================================================================
    // TypeMetaclass Tests
    // =========================================================================

    #[test]
    fn test_type_metaclass_new() {
        let meta = TypeMetaclass::new();
        assert_eq!(meta.type_id(), TYPE_METACLASS_ID);
        assert_eq!(meta.classes_created(), 0);
    }

    #[test]
    fn test_type_metaclass_default() {
        let meta = TypeMetaclass::default();
        assert_eq!(meta.type_id(), TYPE_METACLASS_ID);
    }

    #[test]
    fn test_type_metaclass_name() {
        let meta = TypeMetaclass::new();
        assert_eq!(meta.name().as_str(), "type");
    }

    #[test]
    fn test_type_metaclass_is_subclass_of_self() {
        let meta = TypeMetaclass::new();
        assert!(meta.is_subclass_of(TYPE_METACLASS_ID));
    }

    #[test]
    fn test_type_metaclass_not_subclass_of_other() {
        let meta = TypeMetaclass::new();
        assert!(!meta.is_subclass_of(TypeId::from_raw(999)));
    }

    #[test]
    fn test_type_metaclass_compute_flags_empty() {
        let meta = TypeMetaclass::new();
        let namespace = ClassDict::new();
        let flags = meta.compute_flags(&namespace);
        assert!(flags.is_empty());
    }

    #[test]
    fn test_type_metaclass_compute_flags_with_init() {
        let meta = TypeMetaclass::new();
        let namespace = ClassDict::new();
        namespace.set(intern("__init__"), Value::int_unchecked(0));
        let flags = meta.compute_flags(&namespace);
        assert!(flags.contains(ClassFlags::HAS_INIT));
    }

    #[test]
    fn test_type_metaclass_compute_flags_with_new() {
        let meta = TypeMetaclass::new();
        let namespace = ClassDict::new();
        namespace.set(intern("__new__"), Value::int_unchecked(0));
        let flags = meta.compute_flags(&namespace);
        assert!(flags.contains(ClassFlags::HAS_NEW));
    }

    #[test]
    fn test_type_metaclass_compute_flags_with_slots() {
        let meta = TypeMetaclass::new();
        let namespace = ClassDict::new();
        namespace.set(intern("__slots__"), Value::int_unchecked(0));
        let flags = meta.compute_flags(&namespace);
        assert!(flags.contains(ClassFlags::HAS_SLOTS));
    }

    #[test]
    fn test_type_metaclass_compute_flags_with_multiple() {
        let meta = TypeMetaclass::new();
        let namespace = ClassDict::new();
        namespace.set(intern("__init__"), Value::int_unchecked(0));
        namespace.set(intern("__eq__"), Value::int_unchecked(0));
        namespace.set(intern("__hash__"), Value::int_unchecked(0));
        let flags = meta.compute_flags(&namespace);
        assert!(flags.contains(ClassFlags::HAS_INIT));
        assert!(flags.contains(ClassFlags::HAS_EQ));
        assert!(flags.contains(ClassFlags::HASHABLE));
    }

    #[test]
    fn test_type_metaclass_instantiation_hint_default() {
        let meta = TypeMetaclass::new();
        assert_eq!(meta.instantiation_hint(), InstantiationHint::Generic);
    }

    // =========================================================================
    // MetaclassResolver Tests
    // =========================================================================

    #[test]
    fn test_resolver_no_bases_no_explicit() {
        let result = MetaclassResolver::resolve(None, &[], |_| TYPE_METACLASS_ID);
        assert_eq!(result.unwrap(), TYPE_METACLASS_ID);
    }

    #[test]
    fn test_resolver_explicit_type() {
        let result =
            MetaclassResolver::resolve(Some(TYPE_METACLASS_ID), &[], |_| TYPE_METACLASS_ID);
        assert_eq!(result.unwrap(), TYPE_METACLASS_ID);
    }

    #[test]
    fn test_resolver_bases_with_type_metaclass() {
        let bases = [ClassId(100), ClassId(200)];
        let result = MetaclassResolver::resolve(None, &bases, |_| TYPE_METACLASS_ID);
        assert_eq!(result.unwrap(), TYPE_METACLASS_ID);
    }

    #[test]
    fn test_resolver_base_with_custom_metaclass() {
        let custom_meta = TypeId::from_raw(500);
        let bases = [ClassId(100)];
        let result = MetaclassResolver::resolve(None, &bases, |_| custom_meta);
        assert_eq!(result.unwrap(), custom_meta);
    }

    #[test]
    fn test_resolver_conflict() {
        let custom_meta1 = TypeId::from_raw(500);
        let custom_meta2 = TypeId::from_raw(600);
        let bases = [ClassId(100), ClassId(200)];

        // Different metaclasses with explicit should conflict
        let result = MetaclassResolver::resolve(Some(custom_meta1), &bases, |id| {
            if id == ClassId(100) {
                custom_meta2
            } else {
                TYPE_METACLASS_ID
            }
        });

        assert!(matches!(result, Err(MetaclassError::Conflict { .. })));
    }

    // =========================================================================
    // MetaclassCache Tests
    // =========================================================================

    #[test]
    fn test_metaclass_cache_new() {
        let cache = MetaclassCache::new();
        assert!(cache.is_empty());
        assert_eq!(cache.len(), 0);
    }

    #[test]
    fn test_metaclass_cache_with_type() {
        let type_meta = Arc::new(TypeMetaclass::new());
        let cache = MetaclassCache::with_type_metaclass(type_meta);
        assert!(!cache.is_empty());
        assert_eq!(cache.len(), 1);
        assert!(cache.contains(TYPE_METACLASS_ID));
    }

    #[test]
    fn test_metaclass_cache_get_existing() {
        let type_meta = Arc::new(TypeMetaclass::new());
        let cache = MetaclassCache::with_type_metaclass(type_meta);

        let result = cache.get(TYPE_METACLASS_ID);
        assert!(result.is_some());
        assert_eq!(result.unwrap().type_id(), TYPE_METACLASS_ID);
    }

    #[test]
    fn test_metaclass_cache_get_missing() {
        let cache = MetaclassCache::new();
        let result = cache.get(TYPE_METACLASS_ID);
        assert!(result.is_none());
    }

    #[test]
    fn test_metaclass_cache_register() {
        let cache = MetaclassCache::new();
        let type_meta = Arc::new(TypeMetaclass::new());
        cache.register(type_meta as Arc<dyn Metaclass>);

        assert!(cache.contains(TYPE_METACLASS_ID));
        assert_eq!(cache.len(), 1);
    }

    #[test]
    fn test_metaclass_cache_stats() {
        let type_meta = Arc::new(TypeMetaclass::new());
        let cache = MetaclassCache::with_type_metaclass(type_meta);

        // Hit
        let _ = cache.get(TYPE_METACLASS_ID);
        // Miss
        let _ = cache.get(TypeId::from_raw(999));

        let (hits, misses) = cache.stats();
        assert_eq!(hits, 1);
        assert_eq!(misses, 1);
    }

    #[test]
    fn test_metaclass_cache_clear() {
        let type_meta = Arc::new(TypeMetaclass::new());
        let cache = MetaclassCache::with_type_metaclass(type_meta);
        assert!(!cache.is_empty());

        cache.clear();
        assert!(cache.is_empty());
    }

    // =========================================================================
    // ClassFactory Tests
    // =========================================================================

    #[test]
    fn test_class_factory_new() {
        let factory = ClassFactory::new();
        assert_eq!(factory.classes_created(), 0);
    }

    #[test]
    fn test_class_factory_default() {
        let factory = ClassFactory::default();
        assert_eq!(factory.classes_created(), 0);
    }

    #[test]
    fn test_class_factory_type_metaclass() {
        let factory = ClassFactory::new();
        let meta = factory.type_metaclass();
        assert_eq!(meta.type_id(), TYPE_METACLASS_ID);
    }

    #[test]
    fn test_class_factory_get_metaclass() {
        let factory = ClassFactory::new();
        let result = factory.get_metaclass(TYPE_METACLASS_ID);
        assert!(result.is_some());
    }

    #[test]
    fn test_class_factory_get_unknown_metaclass() {
        let factory = ClassFactory::new();
        let result = factory.get_metaclass(TypeId::from_raw(999));
        assert!(result.is_none());
    }

    #[test]
    fn test_class_factory_cache_stats() {
        let factory = ClassFactory::new();
        let _ = factory.get_metaclass(TYPE_METACLASS_ID);
        let _ = factory.get_metaclass(TypeId::from_raw(999));

        let (hits, misses) = factory.cache_stats();
        assert_eq!(hits, 1);
        assert_eq!(misses, 1);
    }

    #[test]
    fn test_class_factory_register_metaclass() {
        let factory = ClassFactory::new();
        let custom_meta = Arc::new(TypeMetaclass::new());

        // Note: This works because TypeMetaclass implements Metaclass
        factory.register_metaclass(custom_meta as Arc<dyn Metaclass>);

        // Would need different TypeId to properly test, but demonstrates API
    }

    // =========================================================================
    // Instantiation Hint Tests
    // =========================================================================

    #[test]
    fn test_compute_instantiation_hint_inline() {
        let meta = TypeMetaclass::new();
        let namespace = ClassDict::new();
        let flags = ClassFlags::empty();

        let hint = meta.compute_instantiation_hint(flags, &namespace);
        assert_eq!(hint, InstantiationHint::InlineSlots);
    }

    #[test]
    fn test_compute_instantiation_hint_with_init_only() {
        let meta = TypeMetaclass::new();
        let namespace = ClassDict::new();
        let flags = ClassFlags::HAS_INIT;

        let hint = meta.compute_instantiation_hint(flags, &namespace);
        assert_eq!(hint, InstantiationHint::DefaultInit);
    }

    #[test]
    fn test_compute_instantiation_hint_with_new() {
        let meta = TypeMetaclass::new();
        let namespace = ClassDict::new();
        let flags = ClassFlags::HAS_NEW;

        let hint = meta.compute_instantiation_hint(flags, &namespace);
        assert_eq!(hint, InstantiationHint::Generic);
    }

    #[test]
    fn test_compute_instantiation_hint_with_slots() {
        let meta = TypeMetaclass::new();
        let namespace = ClassDict::new();
        namespace.set(intern("__slots__"), Value::int_unchecked(0));
        let flags = ClassFlags::HAS_SLOTS;

        let hint = meta.compute_instantiation_hint(flags, &namespace);
        assert_eq!(hint, InstantiationHint::FixedSlots);
    }

    // =========================================================================
    // Debug and Display Tests
    // =========================================================================

    #[test]
    fn test_type_metaclass_debug() {
        let meta = TypeMetaclass::new();
        let debug_str = format!("{:?}", meta);
        assert!(debug_str.contains("TypeMetaclass"));
    }

    #[test]
    fn test_metaclass_cache_debug() {
        let cache = MetaclassCache::new();
        let debug_str = format!("{:?}", cache);
        assert!(debug_str.contains("MetaclassCache"));
    }

    #[test]
    fn test_class_factory_debug() {
        let factory = ClassFactory::new();
        let debug_str = format!("{:?}", factory);
        assert!(debug_str.contains("ClassFactory"));
    }

    #[test]
    fn test_metaclass_resolver_debug() {
        let debug_str = format!("{:?}", MetaclassResolver);
        assert!(debug_str.contains("MetaclassResolver"));
    }
}
