//! GetAwaitable opcode handler.
//!
//! Converts an object to an awaitable for use in `await` expressions.
//!
//! # Python Semantics (PEP 492)
//!
//! The conversion follows this priority order:
//!
//! 1. **Native Coroutine** (`async def` function) → return as-is
//! 2. **Async Generator** (`async def` with `yield`) → return as-is  
//! 3. **Generator with CO_ITERABLE_COROUTINE** (legacy `@types.coroutine`) → return as-is
//! 4. **Object with `__await__`** → call `__await__()` and verify iterator result
//! 5. **Otherwise** → raise `TypeError`
//!
//! # Performance
//!
//! - Fast path (coroutine): ~5 cycles (single type check)
//! - Slow path (__await__): ~50-100 cycles (method lookup + call + verify)

use crate::VirtualMachine;
use crate::dispatch::ControlFlow;
use crate::error::RuntimeError;
use prism_compiler::bytecode::Instruction;
use prism_core::Value;

/// GetAwaitable: Convert object to awaitable.
///
/// Instruction format: `GetAwaitable dst, src`
/// - `dst`: Destination register for the awaitable
/// - `src`: Source register containing the object to convert
#[inline(always)]
pub fn get_awaitable(vm: &mut VirtualMachine, inst: Instruction) -> ControlFlow {
    let dst = inst.dst().0;
    let src = inst.src1().0;
    let obj = vm.current_frame().get_reg(src);

    // =========================================================================
    // Fast Path: Check if already an awaitable type
    // =========================================================================

    // Check for native awaitable types by examining object flags
    // This is the common case for well-written async code
    if is_native_awaitable(&obj) {
        vm.current_frame_mut().set_reg(dst, obj);
        return ControlFlow::Continue;
    }

    // =========================================================================
    // Medium Path: Check for generator with CO_ITERABLE_COROUTINE
    // =========================================================================

    // Legacy support for @types.coroutine decorated generators
    if is_iterable_coroutine(&obj) {
        vm.current_frame_mut().set_reg(dst, obj);
        return ControlFlow::Continue;
    }

    // =========================================================================
    // Slow Path: Check for __await__ method
    // =========================================================================

    // Look up __await__ method on the object's type
    match lookup_await_method(vm, obj) {
        AwaitLookup::Found(await_method) => {
            // Call __await__() with no arguments
            match call_await_method(vm, await_method, obj) {
                Ok(iterator) => {
                    // Verify the result is an iterator (has __next__)
                    if !is_iterator(&iterator) {
                        return ControlFlow::Error(RuntimeError::type_error(
                            "__await__() returned non-iterator of type '{}'",
                        ));
                    }
                    vm.current_frame_mut().set_reg(dst, iterator);
                    ControlFlow::Continue
                }
                Err(e) => ControlFlow::Error(e),
            }
        }
        AwaitLookup::NotFound => {
            // Object is not awaitable
            ControlFlow::Error(RuntimeError::type_error(format!(
                "object {} can't be used in 'await' expression",
                type_name(&obj)
            )))
        }
        AwaitLookup::Error(e) => ControlFlow::Error(e),
    }
}

// =============================================================================
// Type Checking Helpers
// =============================================================================

/// Check if value is a native coroutine or async generator.
///
/// These types are inherently awaitable and need no conversion.
#[inline(always)]
fn is_native_awaitable(value: &Value) -> bool {
    // Check Value tag for coroutine/async generator types
    // For now, we check if it's an object and has the coroutine flag
    if let Some(ptr) = value.as_object_ptr() {
        // TODO: When we have proper TypeId, check for:
        // - TypeId::COROUTINE
        // - TypeId::ASYNC_GENERATOR
        // For now, stub as false until GeneratorObject is wired up
        let _ = ptr;
        return false;
    }
    false
}

/// Check if value is a generator with CO_ITERABLE_COROUTINE flag.
///
/// This is set by the `@types.coroutine` decorator for backward compatibility.
#[inline(always)]
fn is_iterable_coroutine(value: &Value) -> bool {
    if let Some(ptr) = value.as_object_ptr() {
        // TODO: Check GeneratorObject.code().is_iterable_coroutine()
        let _ = ptr;
        return false;
    }
    false
}

/// Check if value is an iterator (has __next__ method).
#[inline(always)]
fn is_iterator(value: &Value) -> bool {
    // TODO: Check for __next__ method
    // For now, accept any value (will be validated on iteration)
    let _ = value;
    true
}

/// Get the type name of a value for error messages.
#[inline]
fn type_name(value: &Value) -> &'static str {
    if value.is_none() {
        "NoneType"
    } else if value.is_bool() {
        "bool"
    } else if value.is_int() {
        "int"
    } else if value.is_float() {
        "float"
    } else if value.as_object_ptr().is_some() {
        "object" // TODO: Get actual type name from object
    } else {
        "unknown"
    }
}

// =============================================================================
// __await__ Method Lookup
// =============================================================================

/// Result of looking up __await__ method.
enum AwaitLookup {
    /// Method found, ready to call.
    Found(Value),
    /// No __await__ method on the type.
    NotFound,
    /// Error during lookup.
    Error(RuntimeError),
}

/// Look up the __await__ method on an object's type.
#[inline]
fn lookup_await_method(_vm: &VirtualMachine, obj: Value) -> AwaitLookup {
    // TODO: Implement proper __await__ lookup via type's method table
    // This should use the method cache for performance
    //
    // For now, return NotFound for non-coroutine types
    let _ = obj;
    AwaitLookup::NotFound
}

/// Call the __await__ method on an object.
#[inline]
fn call_await_method(
    _vm: &mut VirtualMachine,
    _method: Value,
    _obj: Value,
) -> Result<Value, RuntimeError> {
    // TODO: Implement method call
    // This should call method(obj) with no additional arguments
    Err(RuntimeError::internal("__await__ call not yet implemented"))
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // Type Check Tests
    // =========================================================================

    #[test]
    fn test_none_not_awaitable() {
        assert!(!is_native_awaitable(&Value::none()));
    }

    #[test]
    fn test_int_not_awaitable() {
        let val = Value::int(42).unwrap();
        assert!(!is_native_awaitable(&val));
    }

    #[test]
    fn test_bool_not_awaitable() {
        assert!(!is_native_awaitable(&Value::bool(true)));
        assert!(!is_native_awaitable(&Value::bool(false)));
    }

    #[test]
    fn test_float_not_awaitable() {
        let val = Value::float(3.14);
        assert!(!is_native_awaitable(&val));
    }

    // =========================================================================
    // Type Name Tests
    // =========================================================================

    #[test]
    fn test_type_name_none() {
        assert_eq!(type_name(&Value::none()), "NoneType");
    }

    #[test]
    fn test_type_name_bool() {
        assert_eq!(type_name(&Value::bool(true)), "bool");
    }

    #[test]
    fn test_type_name_int() {
        let val = Value::int(42).unwrap();
        assert_eq!(type_name(&val), "int");
    }

    #[test]
    fn test_type_name_float() {
        let val = Value::float(3.14);
        assert_eq!(type_name(&val), "float");
    }

    // =========================================================================
    // Iterator Check Tests
    // =========================================================================

    #[test]
    fn test_is_iterator_placeholder() {
        // Current implementation accepts all values
        // This will be tightened once __next__ lookup is implemented
        assert!(is_iterator(&Value::none()));
    }
}
