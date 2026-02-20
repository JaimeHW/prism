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
use crate::stdlib::generators::{GeneratorFlags, GeneratorObject};
use prism_compiler::bytecode::Instruction;
use prism_core::Value;

use super::protocol::{call_unary_magic_method, is_iterator, lookup_magic_method, type_name};

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
    GeneratorObject::from_value(*value).is_some_and(|generator| {
        generator.flags().contains(GeneratorFlags::IS_COROUTINE)
            || generator.flags().contains(GeneratorFlags::IS_ASYNC)
    })
}

/// Check if value is a generator with CO_ITERABLE_COROUTINE flag.
///
/// This is set by the `@types.coroutine` decorator for backward compatibility.
#[inline(always)]
fn is_iterable_coroutine(value: &Value) -> bool {
    // Prism does not yet expose a distinct iterable-coroutine runtime flag.
    // Keep this path strict until legacy coroutine decoration metadata is wired.
    let _ = value;
    false
}

/// Check if value is an iterator (has __next__ method).
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
fn lookup_await_method(vm: &VirtualMachine, obj: Value) -> AwaitLookup {
    match lookup_magic_method(vm, obj, "__await__") {
        Ok(Some(method)) => AwaitLookup::Found(method),
        Ok(None) => AwaitLookup::NotFound,
        Err(e) => AwaitLookup::Error(e),
    }
}

/// Call the __await__ method on an object.
#[inline]
fn call_await_method(
    vm: &mut VirtualMachine,
    method: Value,
    obj: Value,
) -> Result<Value, RuntimeError> {
    call_unary_magic_method(vm, method, obj, "__await__")
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use prism_compiler::bytecode::CodeObject;
    use std::sync::Arc;

    fn generator_value(flags: GeneratorFlags) -> Value {
        let code = Arc::new(CodeObject::new("test_get_awaitable", "<test>"));
        let generator = GeneratorObject::with_flags(code, flags);
        let ptr = Box::into_raw(Box::new(generator)) as *const ();
        Value::object_ptr(ptr)
    }

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

    #[test]
    fn test_coroutine_generator_is_native_awaitable() {
        let value = generator_value(GeneratorFlags::IS_COROUTINE | GeneratorFlags::INLINE_STORAGE);
        assert!(is_native_awaitable(&value));
    }

    #[test]
    fn test_async_generator_is_native_awaitable() {
        let value = generator_value(GeneratorFlags::IS_ASYNC | GeneratorFlags::INLINE_STORAGE);
        assert!(is_native_awaitable(&value));
    }

    #[test]
    fn test_regular_generator_not_native_awaitable() {
        let value = generator_value(GeneratorFlags::INLINE_STORAGE);
        assert!(!is_native_awaitable(&value));
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
    fn test_is_iterator_rejects_none() {
        assert!(!is_iterator(&Value::none()));
    }

    #[test]
    fn test_generator_is_iterator() {
        let value = generator_value(GeneratorFlags::INLINE_STORAGE);
        assert!(is_iterator(&value));
    }
}
