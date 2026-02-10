//! Flyweight exception pool.
//!
//! This module provides pre-allocated singleton exceptions for common
//! control-flow exceptions like StopIteration and GeneratorExit.
//! This eliminates allocation for the most frequently thrown exceptions.
//!
//! # Performance Design
//!
//! - **Static singletons**: No allocation when raising StopIteration
//! - **Thread-safe access**: Uses static references
//! - **~10 cycles**: Throwing a flyweight exception is nearly free

use super::object::ExceptionObject;
use super::types::ExceptionTypeId;
use std::sync::{Arc, LazyLock};

// ============================================================================
// Static Flyweight Instances
// ============================================================================

/// Pre-allocated StopIteration exception.
///
/// Used when an iterator is exhausted. This is thrown millions of times
/// in typical Python code, so eliminating allocation is critical.
static STOP_ITERATION: LazyLock<ExceptionObject> =
    LazyLock::new(|| ExceptionObject::flyweight(ExceptionTypeId::StopIteration));

/// Pre-allocated StopAsyncIteration exception.
static STOP_ASYNC_ITERATION: LazyLock<ExceptionObject> =
    LazyLock::new(|| ExceptionObject::flyweight(ExceptionTypeId::StopAsyncIteration));

/// Pre-allocated GeneratorExit exception.
///
/// Raised when a generator's close() method is called.
static GENERATOR_EXIT: LazyLock<ExceptionObject> =
    LazyLock::new(|| ExceptionObject::flyweight(ExceptionTypeId::GeneratorExit));

/// Pre-allocated KeyboardInterrupt exception.
///
/// Raised when the user presses Ctrl+C.
static KEYBOARD_INTERRUPT: LazyLock<ExceptionObject> =
    LazyLock::new(|| ExceptionObject::flyweight(ExceptionTypeId::KeyboardInterrupt));

/// Pre-allocated MemoryError exception.
///
/// Used when memory allocation fails.
static MEMORY_ERROR: LazyLock<ExceptionObject> =
    LazyLock::new(|| ExceptionObject::flyweight(ExceptionTypeId::MemoryError));

/// Pre-allocated RecursionError exception.
///
/// Raised when maximum recursion depth is exceeded.
static RECURSION_ERROR: LazyLock<ExceptionObject> =
    LazyLock::new(|| ExceptionObject::flyweight(ExceptionTypeId::RecursionError));

// ============================================================================
// Flyweight Pool
// ============================================================================

/// Pool of pre-allocated flyweight exceptions.
///
/// Provides zero-allocation access to common control-flow exceptions.
pub struct FlyweightPool;

impl FlyweightPool {
    /// Returns a reference to the pre-allocated StopIteration exception.
    #[inline(always)]
    pub fn stop_iteration() -> &'static ExceptionObject {
        &STOP_ITERATION
    }

    /// Returns a reference to the pre-allocated StopAsyncIteration exception.
    #[inline(always)]
    pub fn stop_async_iteration() -> &'static ExceptionObject {
        &STOP_ASYNC_ITERATION
    }

    /// Returns a reference to the pre-allocated GeneratorExit exception.
    #[inline(always)]
    pub fn generator_exit() -> &'static ExceptionObject {
        &GENERATOR_EXIT
    }

    /// Returns a reference to the pre-allocated KeyboardInterrupt exception.
    #[inline(always)]
    pub fn keyboard_interrupt() -> &'static ExceptionObject {
        &KEYBOARD_INTERRUPT
    }

    /// Returns a reference to the pre-allocated MemoryError exception.
    #[inline(always)]
    pub fn memory_error() -> &'static ExceptionObject {
        &MEMORY_ERROR
    }

    /// Returns a reference to the pre-allocated RecursionError exception.
    #[inline(always)]
    pub fn recursion_error() -> &'static ExceptionObject {
        &RECURSION_ERROR
    }

    /// Returns a flyweight exception for the given type, if available.
    ///
    /// Returns None if no flyweight exists for this type.
    #[inline]
    pub fn get(type_id: ExceptionTypeId) -> Option<&'static ExceptionObject> {
        match type_id {
            ExceptionTypeId::StopIteration => Some(&STOP_ITERATION),
            ExceptionTypeId::StopAsyncIteration => Some(&STOP_ASYNC_ITERATION),
            ExceptionTypeId::GeneratorExit => Some(&GENERATOR_EXIT),
            ExceptionTypeId::KeyboardInterrupt => Some(&KEYBOARD_INTERRUPT),
            ExceptionTypeId::MemoryError => Some(&MEMORY_ERROR),
            ExceptionTypeId::RecursionError => Some(&RECURSION_ERROR),
            _ => None,
        }
    }

    /// Returns true if a flyweight exists for the given type.
    #[inline]
    pub const fn has_flyweight(type_id: ExceptionTypeId) -> bool {
        matches!(
            type_id,
            ExceptionTypeId::StopIteration
                | ExceptionTypeId::StopAsyncIteration
                | ExceptionTypeId::GeneratorExit
                | ExceptionTypeId::KeyboardInterrupt
                | ExceptionTypeId::MemoryError
                | ExceptionTypeId::RecursionError
        )
    }
}

// ============================================================================
// Convenience Functions
// ============================================================================

/// Raises a StopIteration exception (zero-allocation).
#[inline(always)]
pub fn raise_stop_iteration() -> &'static ExceptionObject {
    FlyweightPool::stop_iteration()
}

/// Raises a GeneratorExit exception (zero-allocation).
#[inline(always)]
pub fn raise_generator_exit() -> &'static ExceptionObject {
    FlyweightPool::generator_exit()
}

/// Creates a StopIteration with a value.
///
/// Unlike the flyweight, this allocates because it needs args.
pub fn stop_iteration_with_value(value: prism_core::Value) -> Arc<ExceptionObject> {
    use super::object::ExceptionArgs;

    Arc::new(ExceptionObject::with_args(
        ExceptionTypeId::StopIteration,
        ExceptionArgs::single(value),
    ))
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ════════════════════════════════════════════════════════════════════════
    // Flyweight Access Tests
    // ════════════════════════════════════════════════════════════════════════

    #[test]
    fn test_stop_iteration() {
        let exc = FlyweightPool::stop_iteration();
        assert_eq!(exc.type_id(), ExceptionTypeId::StopIteration);
        assert!(exc.is_flyweight());
    }

    #[test]
    fn test_stop_async_iteration() {
        let exc = FlyweightPool::stop_async_iteration();
        assert_eq!(exc.type_id(), ExceptionTypeId::StopAsyncIteration);
        assert!(exc.is_flyweight());
    }

    #[test]
    fn test_generator_exit() {
        let exc = FlyweightPool::generator_exit();
        assert_eq!(exc.type_id(), ExceptionTypeId::GeneratorExit);
        assert!(exc.is_flyweight());
    }

    #[test]
    fn test_keyboard_interrupt() {
        let exc = FlyweightPool::keyboard_interrupt();
        assert_eq!(exc.type_id(), ExceptionTypeId::KeyboardInterrupt);
        assert!(exc.is_flyweight());
    }

    #[test]
    fn test_memory_error() {
        let exc = FlyweightPool::memory_error();
        assert_eq!(exc.type_id(), ExceptionTypeId::MemoryError);
        assert!(exc.is_flyweight());
    }

    #[test]
    fn test_recursion_error() {
        let exc = FlyweightPool::recursion_error();
        assert_eq!(exc.type_id(), ExceptionTypeId::RecursionError);
        assert!(exc.is_flyweight());
    }

    // ════════════════════════════════════════════════════════════════════════
    // Flyweight Get Tests
    // ════════════════════════════════════════════════════════════════════════

    #[test]
    fn test_get_stop_iteration() {
        let exc = FlyweightPool::get(ExceptionTypeId::StopIteration);
        assert!(exc.is_some());
        assert_eq!(exc.unwrap().type_id(), ExceptionTypeId::StopIteration);
    }

    #[test]
    fn test_get_generator_exit() {
        let exc = FlyweightPool::get(ExceptionTypeId::GeneratorExit);
        assert!(exc.is_some());
    }

    #[test]
    fn test_get_none_for_regular_exception() {
        assert!(FlyweightPool::get(ExceptionTypeId::TypeError).is_none());
        assert!(FlyweightPool::get(ExceptionTypeId::ValueError).is_none());
        assert!(FlyweightPool::get(ExceptionTypeId::KeyError).is_none());
    }

    // ════════════════════════════════════════════════════════════════════════
    // Has Flyweight Tests
    // ════════════════════════════════════════════════════════════════════════

    #[test]
    fn test_has_flyweight_true() {
        assert!(FlyweightPool::has_flyweight(ExceptionTypeId::StopIteration));
        assert!(FlyweightPool::has_flyweight(
            ExceptionTypeId::StopAsyncIteration
        ));
        assert!(FlyweightPool::has_flyweight(ExceptionTypeId::GeneratorExit));
        assert!(FlyweightPool::has_flyweight(
            ExceptionTypeId::KeyboardInterrupt
        ));
        assert!(FlyweightPool::has_flyweight(ExceptionTypeId::MemoryError));
        assert!(FlyweightPool::has_flyweight(
            ExceptionTypeId::RecursionError
        ));
    }

    #[test]
    fn test_has_flyweight_false() {
        assert!(!FlyweightPool::has_flyweight(ExceptionTypeId::TypeError));
        assert!(!FlyweightPool::has_flyweight(ExceptionTypeId::ValueError));
        assert!(!FlyweightPool::has_flyweight(ExceptionTypeId::OSError));
        assert!(!FlyweightPool::has_flyweight(ExceptionTypeId::RuntimeError));
    }

    // ════════════════════════════════════════════════════════════════════════
    // Convenience Function Tests
    // ════════════════════════════════════════════════════════════════════════

    #[test]
    fn test_raise_stop_iteration() {
        let exc = raise_stop_iteration();
        assert_eq!(exc.type_id(), ExceptionTypeId::StopIteration);
    }

    #[test]
    fn test_raise_generator_exit() {
        let exc = raise_generator_exit();
        assert_eq!(exc.type_id(), ExceptionTypeId::GeneratorExit);
    }

    #[test]
    fn test_stop_iteration_with_value() {
        let exc = stop_iteration_with_value(prism_core::Value::none());
        assert_eq!(exc.type_id(), ExceptionTypeId::StopIteration);
        assert!(!exc.is_flyweight()); // Has args, so not flyweight
        assert!(exc.args().is_some());
    }

    // ════════════════════════════════════════════════════════════════════════
    // Static Reference Stability Tests
    // ════════════════════════════════════════════════════════════════════════

    #[test]
    fn test_flyweight_same_instance() {
        // Multiple calls should return the same static reference
        let a = FlyweightPool::stop_iteration();
        let b = FlyweightPool::stop_iteration();
        assert!(std::ptr::eq(a, b));
    }

    #[test]
    fn test_flyweight_get_same_instance() {
        let a = FlyweightPool::stop_iteration();
        let b = FlyweightPool::get(ExceptionTypeId::StopIteration).unwrap();
        assert!(std::ptr::eq(a, b));
    }

    // ════════════════════════════════════════════════════════════════════════
    // Flyweight Properties Tests
    // ════════════════════════════════════════════════════════════════════════

    #[test]
    fn test_flyweight_is_normalized() {
        let exc = FlyweightPool::stop_iteration();
        assert!(exc.is_normalized());
    }

    #[test]
    fn test_flyweight_no_args() {
        let exc = FlyweightPool::stop_iteration();
        assert!(exc.args().is_none());
    }

    #[test]
    fn test_flyweight_no_traceback() {
        let exc = FlyweightPool::generator_exit();
        assert!(!exc.has_traceback());
    }

    #[test]
    fn test_flyweight_no_cause() {
        let exc = FlyweightPool::stop_iteration();
        assert!(exc.cause().is_none());
    }

    #[test]
    fn test_flyweight_no_context() {
        let exc = FlyweightPool::stop_iteration();
        assert!(exc.context().is_none());
    }

    // ════════════════════════════════════════════════════════════════════════
    // Control Flow Tests
    // ════════════════════════════════════════════════════════════════════════

    #[test]
    fn test_flyweight_control_flow_exceptions() {
        // All control-flow exceptions should have flyweights
        assert!(FlyweightPool::has_flyweight(ExceptionTypeId::StopIteration));
        assert!(FlyweightPool::has_flyweight(
            ExceptionTypeId::StopAsyncIteration
        ));
        assert!(FlyweightPool::has_flyweight(ExceptionTypeId::GeneratorExit));
    }

    // ════════════════════════════════════════════════════════════════════════
    // Thread Safety Tests
    // ════════════════════════════════════════════════════════════════════════

    #[test]
    fn test_flyweight_thread_safe_access() {
        use std::thread;

        let handles: Vec<_> = (0..4)
            .map(|_| {
                thread::spawn(|| {
                    let exc = FlyweightPool::stop_iteration();
                    exc.type_id()
                })
            })
            .collect();

        for handle in handles {
            assert_eq!(handle.join().unwrap(), ExceptionTypeId::StopIteration);
        }
    }
}
