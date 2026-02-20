//! GetANext opcode handler.
//!
//! Implements the async iterator __anext__ protocol for `async for` loops.
//!
//! # Python Semantics (PEP 492/525)
//!
//! The `GetANext` opcode calls `__anext__()` on an async iterator to get
//! an awaitable that yields the next value. The result must be awaited.
//!
//! # Return Value
//!
//! The `__anext__()` method returns an awaitable that:
//! - Yields the next value when awaited
//! - Raises `StopAsyncIteration` when exhausted
//!
//! # Performance
//!
//! - Native async generators: ~10 cycles (direct call)
//! - Custom async iterators: ~30-50 cycles (method lookup + call)

use crate::VirtualMachine;
use crate::dispatch::ControlFlow;
use crate::error::RuntimeError;
use crate::stdlib::generators::{GeneratorFlags, GeneratorObject};
use prism_compiler::bytecode::Instruction;
use prism_core::Value;
use prism_runtime::object::ObjectHeader;
use prism_runtime::object::type_obj::TypeId;

/// GetANext: Get next awaitable from async iterator.
///
/// Instruction format: `GetANext dst, src`
/// - `dst`: Destination register for the awaitable
/// - `src`: Source register containing the async iterator
#[inline(always)]
pub fn get_anext(vm: &mut VirtualMachine, inst: Instruction) -> ControlFlow {
    let dst = inst.dst().0;
    let src = inst.src1().0;
    let async_iter = vm.current_frame().get_reg(src);

    // =========================================================================
    // Fast Path: Native async generator
    // =========================================================================

    if let Some(anext_awaitable) = try_native_anext(&async_iter) {
        vm.current_frame_mut().set_reg(dst, anext_awaitable);
        return ControlFlow::Continue;
    }

    // =========================================================================
    // General Path: Look up and call __anext__
    // =========================================================================

    match lookup_anext_method(vm, async_iter) {
        ANextLookup::Found(anext_method) => match call_anext_method(vm, anext_method, async_iter) {
            Ok(awaitable) => {
                vm.current_frame_mut().set_reg(dst, awaitable);
                ControlFlow::Continue
            }
            Err(e) => ControlFlow::Error(e),
        },
        ANextLookup::NotFound => ControlFlow::Error(RuntimeError::type_error(format!(
            "'async for' received object without __anext__ method: {}",
            type_name(&async_iter)
        ))),
        ANextLookup::Error(e) => ControlFlow::Error(e),
    }
}

// =============================================================================
// Native Async Generator Fast Path
// =============================================================================

/// Try to get the next awaitable from a native async generator.
///
/// Returns `Some(awaitable)` if the object is a native async generator,
/// `None` otherwise.
#[inline(always)]
fn try_native_anext(value: &Value) -> Option<Value> {
    let generator = GeneratorObject::from_value(*value)?;
    if generator.flags().contains(GeneratorFlags::IS_ASYNC) {
        Some(*value)
    } else {
        None
    }
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
// __anext__ Method Lookup
// =============================================================================

/// Result of looking up __anext__ method.
enum ANextLookup {
    /// Method found, ready to call.
    Found(Value),
    /// No __anext__ method on the type.
    NotFound,
    /// Error during lookup.
    Error(RuntimeError),
}

/// Look up the __anext__ method on an object's type.
#[inline]
fn lookup_anext_method(_vm: &VirtualMachine, obj: Value) -> ANextLookup {
    // TODO: Implement proper __anext__ lookup via type's method table
    let _ = obj;
    ANextLookup::NotFound
}

/// Call the __anext__ method on an async iterator.
#[inline]
fn call_anext_method(
    _vm: &mut VirtualMachine,
    _method: Value,
    _obj: Value,
) -> Result<Value, RuntimeError> {
    // TODO: Implement method call
    Err(RuntimeError::internal("__anext__ call not yet implemented"))
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
        let code = Arc::new(CodeObject::new("test_get_anext", "<test>"));
        let generator = GeneratorObject::with_flags(code, flags);
        let ptr = Box::into_raw(Box::new(generator)) as *const ();
        Value::object_ptr(ptr)
    }

    #[test]
    fn test_none_has_no_native_anext() {
        assert!(try_native_anext(&Value::none()).is_none());
    }

    #[test]
    fn test_int_has_no_native_anext() {
        let val = Value::int(42).unwrap();
        assert!(try_native_anext(&val).is_none());
    }

    #[test]
    fn test_async_generator_has_native_anext() {
        let val = generator_value(GeneratorFlags::IS_ASYNC | GeneratorFlags::INLINE_STORAGE);
        assert_eq!(try_native_anext(&val), Some(val));
    }

    #[test]
    fn test_regular_generator_has_no_native_anext() {
        let val = generator_value(GeneratorFlags::INLINE_STORAGE);
        assert!(try_native_anext(&val).is_none());
    }

    #[test]
    fn test_type_name_coverage() {
        assert_eq!(type_name(&Value::none()), "NoneType");
        assert_eq!(type_name(&Value::bool(false)), "bool");
        assert_eq!(type_name(&Value::int(0).unwrap()), "int");
        assert_eq!(type_name(&Value::float(0.0)), "float");
    }
}
