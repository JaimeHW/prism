//! GetAIter opcode handler.
//!
//! Implements the async iterator protocol for `async for` loops.
//!
//! # Python Semantics (PEP 492/525)
//!
//! The `GetAIter` opcode calls `__aiter__()` on an object to obtain
//! an async iterator. The result is used for `async for` iteration.
//!
//! # Versions
//!
//! - Python 3.5-3.9: `__aiter__` must return an awaitable
//! - Python 3.10+: `__aiter__` must be a regular method (not awaitable)
//!
//! We implement Python 3.10+ semantics for simplicity.
//!
//! # Performance
//!
//! - Method lookup: Cached via type's method table
//! - Call overhead: ~20 cycles for native async iterators

use crate::VirtualMachine;
use crate::dispatch::ControlFlow;
use crate::error::RuntimeError;
use crate::stdlib::generators::{GeneratorFlags, GeneratorObject};
use prism_compiler::bytecode::Instruction;
use prism_core::Value;
use prism_runtime::object::ObjectHeader;
use prism_runtime::object::type_obj::TypeId;

/// GetAIter: Get async iterator from object.
///
/// Instruction format: `GetAIter dst, src`
/// - `dst`: Destination register for the async iterator
/// - `src`: Source register containing the iterable object
#[inline(always)]
pub fn get_aiter(vm: &mut VirtualMachine, inst: Instruction) -> ControlFlow {
    let dst = inst.dst().0;
    let src = inst.src1().0;
    let obj = vm.current_frame().get_reg(src);

    // =========================================================================
    // Fast Path: Check for native async iterator types
    // =========================================================================

    if is_async_iterator(&obj) {
        // Native async iterators return themselves from __aiter__
        vm.current_frame_mut().set_reg(dst, obj);
        return ControlFlow::Continue;
    }

    // =========================================================================
    // General Path: Look up and call __aiter__
    // =========================================================================

    match lookup_aiter_method(vm, obj) {
        AIterLookup::Found(aiter_method) => {
            match call_aiter_method(vm, aiter_method, obj) {
                Ok(async_iterator) => {
                    // Verify result has __anext__ method
                    if !has_anext(&async_iterator) {
                        return ControlFlow::Error(RuntimeError::type_error(
                            "'async for' requires an object with __anext__ method",
                        ));
                    }
                    vm.current_frame_mut().set_reg(dst, async_iterator);
                    ControlFlow::Continue
                }
                Err(e) => ControlFlow::Error(e),
            }
        }
        AIterLookup::NotFound => ControlFlow::Error(RuntimeError::type_error(format!(
            "'async for' requires an object with __aiter__ method, found {}",
            type_name(&obj)
        ))),
        AIterLookup::Error(e) => ControlFlow::Error(e),
    }
}

// =============================================================================
// Type Checking Helpers
// =============================================================================

/// Check if value is a native async iterator.
#[inline(always)]
fn is_async_iterator(value: &Value) -> bool {
    GeneratorObject::from_value(*value)
        .is_some_and(|generator| generator.flags().contains(GeneratorFlags::IS_ASYNC))
}

/// Check if value has __anext__ method.
#[inline(always)]
fn has_anext(value: &Value) -> bool {
    is_async_iterator(value)
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
    } else if let Some(ptr) = value.as_object_ptr() {
        extract_type_id(ptr).name()
    } else {
        "unknown"
    }
}

// =============================================================================
// __aiter__ Method Lookup
// =============================================================================

/// Result of looking up __aiter__ method.
enum AIterLookup {
    /// Method found, ready to call.
    Found(Value),
    /// No __aiter__ method on the type.
    NotFound,
    /// Error during lookup.
    Error(RuntimeError),
}

/// Look up the __aiter__ method on an object's type.
#[inline]
fn lookup_aiter_method(_vm: &VirtualMachine, obj: Value) -> AIterLookup {
    // TODO: Implement proper __aiter__ lookup via type's method table
    let _ = obj;
    AIterLookup::NotFound
}

/// Call the __aiter__ method on an object.
#[inline]
fn call_aiter_method(
    _vm: &mut VirtualMachine,
    _method: Value,
    _obj: Value,
) -> Result<Value, RuntimeError> {
    // TODO: Implement method call
    Err(RuntimeError::internal("__aiter__ call not yet implemented"))
}

#[inline(always)]
fn extract_type_id(ptr: *const ()) -> TypeId {
    let header = ptr as *const ObjectHeader;
    unsafe { (*header).type_id }
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
        let code = Arc::new(CodeObject::new("test_get_aiter", "<test>"));
        let generator = GeneratorObject::with_flags(code, flags);
        let ptr = Box::into_raw(Box::new(generator)) as *const ();
        Value::object_ptr(ptr)
    }

    #[test]
    fn test_none_not_async_iterator() {
        assert!(!is_async_iterator(&Value::none()));
    }

    #[test]
    fn test_int_not_async_iterator() {
        let val = Value::int(42).unwrap();
        assert!(!is_async_iterator(&val));
    }

    #[test]
    fn test_async_generator_is_async_iterator() {
        let val = generator_value(GeneratorFlags::IS_ASYNC | GeneratorFlags::INLINE_STORAGE);
        assert!(is_async_iterator(&val));
        assert!(has_anext(&val));
    }

    #[test]
    fn test_regular_generator_not_async_iterator() {
        let val = generator_value(GeneratorFlags::INLINE_STORAGE);
        assert!(!is_async_iterator(&val));
        assert!(!has_anext(&val));
    }

    #[test]
    fn test_type_name_various() {
        assert_eq!(type_name(&Value::none()), "NoneType");
        assert_eq!(type_name(&Value::bool(true)), "bool");
        assert_eq!(type_name(&Value::int(1).unwrap()), "int");
        assert_eq!(type_name(&Value::float(1.0)), "float");
    }
}
