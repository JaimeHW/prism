//! Flyweight exception reference for zero-copy exception propagation.
//!
//! This module provides `FlyweightExceptionRef`, a lightweight pointer wrapper that
//! enables efficient exception passing through the VM's control flow without
//! cloning the full `ExceptionObject`.
//!
//! # Design Rationale
//!
//! Exception handling in hot paths (like StopIteration in for-loops) must be
//! as fast as possible. By using a pointer-based reference instead of cloning,
//! we achieve:
//!
//! - **Zero allocation**: No heap allocation during exception propagation
//! - **Minimal copying**: Only 8 bytes (pointer) copied instead of full object
//! - **Cache efficiency**: Pointer fits in register, no memory traffic
//!
//! # Safety
//!
//! `FlyweightExceptionRef` is **not** a safe abstraction by itself. The VM must ensure:
//!
//! 1. The referenced `ExceptionObject` outlives the `FlyweightExceptionRef`
//! 2. No mutation occurs while `FlyweightExceptionRef` is active
//! 3. Proper ownership transfer when exception is caught
//!
//! The VM's exception handling loop maintains these invariants through its
//! structured unwinding process.
//!
//! # Performance Target
//!
//! | Operation | Cycles |
//! |-----------|--------|
//! | Create    | 1      |
//! | Clone     | 1      |
//! | Deref     | 1      |

use crate::stdlib::exceptions::ExceptionObject;
use std::fmt;
use std::hash::{Hash, Hasher};
use std::ptr::NonNull;

// ============================================================================
// FlyweightExceptionRef
// ============================================================================

/// A flyweight reference to an exception object.
///
/// This is a thin pointer wrapper that provides:
/// - Zero-copy exception propagation
/// - Single-register size (8 bytes on 64-bit)
/// - Efficient passing through `ControlFlow` variants
///
/// # Memory Layout
///
/// ```text
/// ┌────────────────────────────────────────┐
/// │ ptr: NonNull<ExceptionObject> (8 bytes)│
/// └────────────────────────────────────────┘
/// ```
///
/// # Example
///
/// ```ignore
/// let exc = ExceptionObject::new(ExceptionTypeId::StopIteration);
/// let exc_ref = FlyweightExceptionRef::new(&exc);
///
/// // Pass through control flow (zero copy)
/// return ControlFlow::Exception(exc_ref);
/// ```
#[repr(transparent)]
#[derive(Clone, Copy)]
pub struct FlyweightExceptionRef {
    /// Non-null pointer to the exception object.
    /// Using NonNull for niche optimization (Option<FlyweightExceptionRef> is 8 bytes).
    ptr: NonNull<ExceptionObject>,
}

impl FlyweightExceptionRef {
    /// Creates a new exception reference from a shared reference.
    ///
    /// # Safety
    ///
    /// The caller must ensure the `ExceptionObject` outlives this reference.
    #[inline(always)]
    pub fn new(exception: &ExceptionObject) -> Self {
        Self {
            ptr: NonNull::from(exception),
        }
    }

    /// Creates a new exception reference from a mutable reference.
    ///
    /// # Safety
    ///
    /// The caller must ensure the `ExceptionObject` outlives this reference
    /// and no other mutable references exist.
    #[inline(always)]
    pub fn from_mut(exception: &mut ExceptionObject) -> Self {
        Self {
            ptr: NonNull::from(exception),
        }
    }

    /// Creates an exception reference from a raw pointer.
    ///
    /// # Safety
    ///
    /// The pointer must be non-null and point to a valid `ExceptionObject`.
    #[inline(always)]
    pub const unsafe fn from_raw(ptr: *const ExceptionObject) -> Self {
        Self {
            // SAFETY: Caller ensures ptr is non-null
            ptr: unsafe { NonNull::new_unchecked(ptr as *mut ExceptionObject) },
        }
    }

    /// Returns the raw pointer to the exception object.
    #[inline(always)]
    pub const fn as_ptr(&self) -> *const ExceptionObject {
        self.ptr.as_ptr()
    }

    /// Returns the mutable raw pointer to the exception object.
    #[inline(always)]
    pub fn as_mut_ptr(&self) -> *mut ExceptionObject {
        self.ptr.as_ptr()
    }

    /// Dereferences to the underlying exception object.
    ///
    /// # Safety
    ///
    /// The pointer must still be valid (object not dropped).
    #[inline(always)]
    pub unsafe fn as_ref(&self) -> &ExceptionObject {
        // SAFETY: Caller ensures the pointer is valid
        unsafe { self.ptr.as_ref() }
    }

    /// Dereferences to a mutable reference.
    ///
    /// # Safety
    ///
    /// The pointer must be valid and no other references must exist.
    #[inline(always)]
    pub unsafe fn as_mut(&mut self) -> &mut ExceptionObject {
        // SAFETY: Caller ensures exclusive access
        unsafe { self.ptr.as_mut() }
    }

    /// Checks if two references point to the same exception object.
    ///
    /// This is an identity check (pointer equality), not semantic equality.
    #[inline(always)]
    pub fn ptr_eq(&self, other: &Self) -> bool {
        self.ptr == other.ptr
    }
}

// Note: to_owned() is not available because ExceptionObject doesn't implement Clone.
// Use the original object reference patterns instead.

impl fmt::Debug for FlyweightExceptionRef {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("FlyweightExceptionRef")
            .field("ptr", &self.ptr)
            .finish()
    }
}

impl fmt::Display for FlyweightExceptionRef {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // SAFETY: For display, we assume the pointer is still valid
        // This is only used for debugging/logging
        unsafe {
            let exc = self.ptr.as_ref();
            write!(f, "{:?}", exc.type_id())
        }
    }
}

impl PartialEq for FlyweightExceptionRef {
    #[inline(always)]
    fn eq(&self, other: &Self) -> bool {
        // Pointer equality for fast comparison
        self.ptr == other.ptr
    }
}

impl Eq for FlyweightExceptionRef {}

impl Hash for FlyweightExceptionRef {
    #[inline(always)]
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.ptr.hash(state);
    }
}

// SAFETY: FlyweightExceptionRef is Send if ExceptionObject is Send.
// The VM ensures proper synchronization.
unsafe impl Send for FlyweightExceptionRef {}

// SAFETY: FlyweightExceptionRef is Sync if ExceptionObject is Sync.
// The VM ensures proper synchronization.
unsafe impl Sync for FlyweightExceptionRef {}

// ============================================================================
// OwnedExceptionRef
// ============================================================================

/// An owned exception reference that manages lifetime.
///
/// Unlike `FlyweightExceptionRef`, this type owns a `Box<ExceptionObject>` and
/// provides safe access. Use this when the exception needs to persist
/// beyond the current stack frame (e.g., stored in VM state).
///
/// # Memory Layout
///
/// ```text
/// ┌────────────────────────────────────────┐
/// │ exception: Box<ExceptionObject> (8b)   │
/// └────────────────────────────────────────┘
/// ```
#[derive(Debug)]
pub struct OwnedExceptionRef {
    exception: Box<ExceptionObject>,
}

impl OwnedExceptionRef {
    /// Creates a new owned exception reference.
    #[inline]
    pub fn new(exception: ExceptionObject) -> Self {
        Self {
            exception: Box::new(exception),
        }
    }

    /// Creates from an existing boxed exception.
    #[inline(always)]
    pub fn from_boxed(exception: Box<ExceptionObject>) -> Self {
        Self { exception }
    }

    /// Returns a flyweight reference to this exception.
    ///
    /// The returned `FlyweightExceptionRef` is valid as long as this
    /// `OwnedExceptionRef` is not dropped.
    #[inline(always)]
    pub fn as_flyweight(&self) -> FlyweightExceptionRef {
        FlyweightExceptionRef::new(&self.exception)
    }

    /// Returns a shared reference to the exception.
    #[inline(always)]
    pub fn get(&self) -> &ExceptionObject {
        &self.exception
    }

    /// Returns a mutable reference to the exception.
    #[inline(always)]
    pub fn get_mut(&mut self) -> &mut ExceptionObject {
        &mut self.exception
    }

    /// Consumes this and returns the inner exception.
    #[inline(always)]
    pub fn into_inner(self) -> ExceptionObject {
        *self.exception
    }

    /// Consumes this and returns the boxed exception.
    #[inline(always)]
    pub fn into_boxed(self) -> Box<ExceptionObject> {
        self.exception
    }
}

impl From<ExceptionObject> for OwnedExceptionRef {
    #[inline]
    fn from(exception: ExceptionObject) -> Self {
        Self::new(exception)
    }
}

impl std::ops::Deref for OwnedExceptionRef {
    type Target = ExceptionObject;

    #[inline(always)]
    fn deref(&self) -> &Self::Target {
        &self.exception
    }
}

impl std::ops::DerefMut for OwnedExceptionRef {
    #[inline(always)]
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.exception
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::stdlib::exceptions::ExceptionTypeId;

    // ════════════════════════════════════════════════════════════════════════
    // FlyweightExceptionRef Creation Tests
    // ════════════════════════════════════════════════════════════════════════

    #[test]
    fn test_flyweight_ref_from_shared_ref() {
        let exc = ExceptionObject::new(ExceptionTypeId::ValueError);
        let exc_ref = FlyweightExceptionRef::new(&exc);

        assert_eq!(exc_ref.as_ptr(), &exc as *const _);
    }

    #[test]
    fn test_flyweight_ref_from_mut_ref() {
        let mut exc = ExceptionObject::new(ExceptionTypeId::TypeError);
        let exc_ref = FlyweightExceptionRef::from_mut(&mut exc);

        assert_eq!(exc_ref.as_ptr(), &exc as *const _);
    }

    #[test]
    fn test_flyweight_ref_from_raw() {
        let exc = ExceptionObject::new(ExceptionTypeId::KeyError);
        let ptr = &exc as *const ExceptionObject;

        let exc_ref = unsafe { FlyweightExceptionRef::from_raw(ptr) };
        assert_eq!(exc_ref.as_ptr(), ptr);
    }

    // ════════════════════════════════════════════════════════════════════════
    // FlyweightExceptionRef Access Tests
    // ════════════════════════════════════════════════════════════════════════

    #[test]
    fn test_flyweight_ref_as_ref() {
        let exc = ExceptionObject::new(ExceptionTypeId::IndexError);
        let exc_ref = FlyweightExceptionRef::new(&exc);

        unsafe {
            let deref = exc_ref.as_ref();
            assert_eq!(deref.type_id(), ExceptionTypeId::IndexError);
        }
    }

    #[test]
    fn test_flyweight_ref_as_mut() {
        let mut exc = ExceptionObject::new(ExceptionTypeId::AttributeError);
        let mut exc_ref = FlyweightExceptionRef::from_mut(&mut exc);

        unsafe {
            let deref = exc_ref.as_mut();
            // Verify we can access mutably
            assert_eq!(deref.type_id(), ExceptionTypeId::AttributeError);
        }
    }

    // ════════════════════════════════════════════════════════════════════════
    // FlyweightExceptionRef Comparison Tests
    // ════════════════════════════════════════════════════════════════════════

    #[test]
    fn test_flyweight_ref_ptr_eq_same() {
        let exc = ExceptionObject::new(ExceptionTypeId::StopIteration);
        let ref1 = FlyweightExceptionRef::new(&exc);
        let ref2 = FlyweightExceptionRef::new(&exc);

        assert!(ref1.ptr_eq(&ref2));
        assert_eq!(ref1, ref2);
    }

    #[test]
    fn test_flyweight_ref_ptr_eq_different() {
        let exc1 = ExceptionObject::new(ExceptionTypeId::StopIteration);
        let exc2 = ExceptionObject::new(ExceptionTypeId::StopIteration);

        let ref1 = FlyweightExceptionRef::new(&exc1);
        let ref2 = FlyweightExceptionRef::new(&exc2);

        assert!(!ref1.ptr_eq(&ref2));
        assert_ne!(ref1, ref2);
    }

    #[test]
    fn test_flyweight_ref_clone() {
        let exc = ExceptionObject::new(ExceptionTypeId::GeneratorExit);
        let ref1 = FlyweightExceptionRef::new(&exc);
        let ref2 = ref1; // Copy

        assert!(ref1.ptr_eq(&ref2));
    }

    // ════════════════════════════════════════════════════════════════════════
    // FlyweightExceptionRef Size Tests
    // ════════════════════════════════════════════════════════════════════════

    #[test]
    fn test_flyweight_ref_size() {
        // Must be pointer-sized for efficiency
        assert_eq!(std::mem::size_of::<FlyweightExceptionRef>(), 8);
    }

    #[test]
    fn test_flyweight_ref_option_size() {
        // NonNull enables niche optimization
        assert_eq!(
            std::mem::size_of::<Option<FlyweightExceptionRef>>(),
            std::mem::size_of::<FlyweightExceptionRef>()
        );
    }

    #[test]
    fn test_flyweight_ref_alignment() {
        // Must be pointer-aligned
        assert_eq!(std::mem::align_of::<FlyweightExceptionRef>(), 8);
    }

    // ════════════════════════════════════════════════════════════════════════
    // FlyweightExceptionRef Hash Tests
    // ════════════════════════════════════════════════════════════════════════

    #[test]
    fn test_flyweight_ref_hash_consistency() {
        use std::collections::hash_map::DefaultHasher;

        let exc = ExceptionObject::new(ExceptionTypeId::ValueError);
        let ref1 = FlyweightExceptionRef::new(&exc);
        let ref2 = FlyweightExceptionRef::new(&exc);

        let mut hasher1 = DefaultHasher::new();
        let mut hasher2 = DefaultHasher::new();

        ref1.hash(&mut hasher1);
        ref2.hash(&mut hasher2);

        assert_eq!(hasher1.finish(), hasher2.finish());
    }

    // ════════════════════════════════════════════════════════════════════════
    // OwnedExceptionRef Tests
    // ════════════════════════════════════════════════════════════════════════

    #[test]
    fn test_owned_exception_ref_new() {
        let exc = ExceptionObject::new(ExceptionTypeId::TypeError);
        let owned = OwnedExceptionRef::new(exc);

        assert_eq!(owned.type_id(), ExceptionTypeId::TypeError);
    }

    #[test]
    fn test_owned_exception_ref_from_boxed() {
        let boxed = Box::new(ExceptionObject::new(ExceptionTypeId::KeyError));
        let owned = OwnedExceptionRef::from_boxed(boxed);

        assert_eq!(owned.type_id(), ExceptionTypeId::KeyError);
    }

    #[test]
    fn test_owned_exception_ref_as_flyweight() {
        let exc = ExceptionObject::new(ExceptionTypeId::IndexError);
        let owned = OwnedExceptionRef::new(exc);
        let exc_ref = owned.as_flyweight();

        // The flyweight ref should point to the owned data
        assert_eq!(exc_ref.as_ptr(), owned.get() as *const _);
    }

    #[test]
    fn test_owned_exception_ref_deref() {
        let exc = ExceptionObject::new(ExceptionTypeId::AttributeError);
        let owned = OwnedExceptionRef::new(exc);

        // Test Deref trait
        assert_eq!(owned.type_id(), ExceptionTypeId::AttributeError);
    }

    #[test]
    fn test_owned_exception_ref_deref_mut() {
        let exc = ExceptionObject::new(ExceptionTypeId::NameError);
        let mut owned = OwnedExceptionRef::new(exc);

        // Test DerefMut trait - just verify we can get mutable access
        let _ = owned.get_mut();
    }

    #[test]
    fn test_owned_exception_ref_into_inner() {
        let exc = ExceptionObject::new(ExceptionTypeId::OSError);
        let owned = OwnedExceptionRef::new(exc);

        let inner = owned.into_inner();
        assert_eq!(inner.type_id(), ExceptionTypeId::OSError);
    }

    #[test]
    fn test_owned_exception_ref_into_boxed() {
        let exc = ExceptionObject::new(ExceptionTypeId::EOFError);
        let owned = OwnedExceptionRef::new(exc);

        let boxed = owned.into_boxed();
        assert_eq!(boxed.type_id(), ExceptionTypeId::EOFError);
    }

    #[test]
    fn test_owned_exception_ref_from_exception() {
        let exc = ExceptionObject::new(ExceptionTypeId::MemoryError);
        let owned: OwnedExceptionRef = exc.into();

        assert_eq!(owned.type_id(), ExceptionTypeId::MemoryError);
    }

    #[test]
    fn test_owned_exception_ref_size() {
        // Should be just a Box (pointer-sized)
        assert_eq!(std::mem::size_of::<OwnedExceptionRef>(), 8);
    }
}
