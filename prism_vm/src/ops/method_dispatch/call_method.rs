//! CallMethod opcode: Optimized method invocation.
//!
//! # Encoding
//!
//! `CallMethod dst, method_reg, argc`
//! - `dst`: receives return value
//! - `method_reg`: register containing method (from LoadMethod)
//! - `method_reg+1`: contains self instance or `None` marker
//! - `argc`: number of explicit arguments (after method_reg+1)
//!
//! # Register Layout
//!
//! After LoadMethod, registers are arranged as:
//! ```text
//! [method_reg]:     method/function
//! [method_reg+1]:   self instance or `None` marker
//! [method_reg+2..]: explicit arguments
//! ```
//!
//! # Type-Specialized Dispatch
//!
//! - **FunctionObject**: Push frame, optionally set self as r0
//! - **BuiltinFunction**: Inline call with args
//! - **BoundMethod**: Extract function + instance, recurse
//! - **Closure**: Push frame with captured variables

use crate::VirtualMachine;
use crate::builtins::BuiltinFunctionObject;
use crate::dispatch::ControlFlow;
use crate::error::RuntimeError;
use prism_compiler::bytecode::Instruction;
use prism_core::Value;
use prism_runtime::object::ObjectHeader;
use prism_runtime::object::descriptor::BoundMethod;
use prism_runtime::object::type_obj::TypeId;
use prism_runtime::types::function::FunctionObject;
use smallvec::SmallVec;
use std::sync::Arc;

// =============================================================================
// CallMethod Handler
// =============================================================================

/// CallMethod: Optimized method invocation.
///
/// Avoids BoundMethod allocation by passing self explicitly in registers.
/// Uses TypeId dispatch for fast type-specific call paths.
///
/// # Performance
///
/// - User function: ~15 cycles (frame push + arg copy)
/// - Builtin function: ~10 cycles (inline call)
/// - BoundMethod: ~20 cycles (unpack + recurse)
#[inline(always)]
pub fn call_method(vm: &mut VirtualMachine, inst: Instruction) -> ControlFlow {
    let dst = inst.dst().0;
    let method_reg = inst.src1().0;
    let argc = inst.src2().0 as usize;

    let method = vm.current_frame().get_reg(method_reg);
    let self_slot = vm.current_frame().get_reg(method_reg + 1);
    let implicit_self = implicit_self_from_slot(self_slot);

    // Check if method is a callable object
    let Some(ptr) = method.as_object_ptr() else {
        return ControlFlow::Error(RuntimeError::type_error(format!(
            "'{}' object is not callable",
            method.type_name()
        )));
    };

    let type_id = extract_type_id(ptr);

    match type_id {
        TypeId::FUNCTION | TypeId::CLOSURE => {
            call_user_function(vm, ptr, implicit_self, dst, method_reg, argc)
        }
        TypeId::BUILTIN_FUNCTION => {
            call_builtin_function(vm, ptr, implicit_self, dst, method_reg, argc)
        }
        TypeId::METHOD => call_bound_method(vm, ptr, dst, method_reg, argc),
        _ => ControlFlow::Error(RuntimeError::type_error(format!(
            "'{}' object is not callable",
            type_id.name()
        ))),
    }
}

/// Convert LoadMethod's self slot into an optional implicit self argument.
#[inline(always)]
fn implicit_self_from_slot(self_slot: Value) -> Option<Value> {
    if self_slot.is_none() {
        None
    } else {
        Some(self_slot)
    }
}

// =============================================================================
// Type-Specific Call Paths
// =============================================================================

/// Call a user-defined function with optional implicit self.
///
/// # Register Layout After Push
///
/// New frame:
/// - With implicit self: r0=self, r1..rN=explicit args
/// - Without implicit self: r0..rN=explicit args
#[inline]
fn call_user_function(
    vm: &mut VirtualMachine,
    func_ptr: *const (),
    implicit_self: Option<Value>,
    dst: u8,
    method_reg: u8,
    argc: usize,
) -> ControlFlow {
    let func = unsafe { &*(func_ptr as *const FunctionObject) };
    let code = Arc::clone(&func.code);

    // Push new frame for function execution
    if let Err(e) = vm.push_frame(Arc::clone(&code), dst) {
        return ControlFlow::Error(e);
    }

    // Get caller frame index (now -2 since we pushed a new frame)
    let caller_frame_idx = vm.call_depth() - 2;

    let mut arg_dst = 0u8;
    if let Some(self_val) = implicit_self {
        vm.current_frame_mut().set_reg(0, self_val);
        arg_dst = 1;
    }

    // Copy explicit arguments.
    for i in 0..argc {
        let arg = vm.frames[caller_frame_idx].get_reg(method_reg + 2 + i as u8);
        vm.current_frame_mut().set_reg(arg_dst + i as u8, arg);
    }

    ControlFlow::Continue
}

/// Call a builtin function with optional implicit self + args.
///
/// Collects all arguments into a SmallVec and invokes the builtin directly.
/// No frame push required - builtins execute synchronously.
#[inline]
fn call_builtin_function(
    vm: &mut VirtualMachine,
    func_ptr: *const (),
    implicit_self: Option<Value>,
    dst: u8,
    method_reg: u8,
    argc: usize,
) -> ControlFlow {
    let builtin = unsafe { &*(func_ptr as *const BuiltinFunctionObject) };

    // Collect arguments: implicit self (if present) + explicit args.
    // Use SmallVec to avoid heap allocation for typical calls (≤8 args)
    let caller_frame = &vm.frames[vm.call_depth() - 1];
    let mut args: SmallVec<[Value; 8]> =
        SmallVec::with_capacity(argc + implicit_self.is_some() as usize);
    if let Some(self_val) = implicit_self {
        args.push(self_val);
    }

    // Collect explicit arguments
    for i in 0..argc {
        args.push(caller_frame.get_reg(method_reg + 2 + i as u8));
    }

    // Call the builtin function
    match builtin.call(&args) {
        Ok(result) => {
            vm.current_frame_mut().set_reg(dst, result);
            ControlFlow::Continue
        }
        Err(e) => ControlFlow::Error(RuntimeError::type_error(e.to_string())),
    }
}

/// Call a pre-bound method (BoundMethod object).
///
/// Extracts the underlying function and instance, then dispatches
/// to the appropriate call path.
#[inline]
fn call_bound_method(
    vm: &mut VirtualMachine,
    bound_ptr: *const (),
    dst: u8,
    method_reg: u8,
    argc: usize,
) -> ControlFlow {
    let bound = unsafe { &*(bound_ptr as *const BoundMethod) };

    // Extract function and instance from BoundMethod
    let func = bound.function();
    let instance = bound.instance();

    // Get the function pointer
    let Some(func_ptr) = func.as_object_ptr() else {
        return ControlFlow::Error(RuntimeError::type_error(
            "bound method has invalid function",
        ));
    };

    let type_id = extract_type_id(func_ptr);

    // Recurse with extracted function and instance
    match type_id {
        TypeId::FUNCTION | TypeId::CLOSURE => {
            call_user_function(vm, func_ptr, Some(instance), dst, method_reg, argc)
        }
        TypeId::BUILTIN_FUNCTION => {
            call_builtin_function(vm, func_ptr, Some(instance), dst, method_reg, argc)
        }
        _ => ControlFlow::Error(RuntimeError::type_error(format!(
            "bound method wraps non-callable '{}' object",
            type_id.name()
        ))),
    }
}

// =============================================================================
// Helpers
// =============================================================================

/// Extract TypeId from an object pointer.
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
    use crate::builtins::{BuiltinError, BuiltinFunctionObject};
    use prism_compiler::bytecode::{CodeObject, Instruction, Opcode};
    use prism_runtime::types::function::FunctionObject;
    use prism_runtime::types::list::ListObject;
    use std::sync::Arc;

    fn push_test_frame(vm: &mut VirtualMachine, name: &str) {
        let mut code = CodeObject::new(name, "<test>");
        code.register_count = 16;
        vm.push_frame(Arc::new(code), 0)
            .expect("failed to push test frame");
    }

    fn make_test_function_value(name: &str) -> (*mut FunctionObject, Value) {
        let mut code = CodeObject::new(name, "<test>");
        code.register_count = 16;
        let func = Box::new(FunctionObject::new(
            Arc::new(code),
            Arc::from(name),
            None,
            None,
        ));
        let ptr = Box::into_raw(func);
        (ptr, Value::object_ptr(ptr as *const ()))
    }

    fn builtin_arg_count(args: &[Value]) -> Result<Value, BuiltinError> {
        Ok(Value::int(args.len() as i64).expect("arg count should fit in tagged int"))
    }

    fn make_builtin_value(name: &str) -> (*mut BuiltinFunctionObject, Value) {
        let builtin = Box::new(BuiltinFunctionObject::new(
            Arc::from(name),
            builtin_arg_count,
        ));
        let ptr = Box::into_raw(builtin);
        (ptr, Value::object_ptr(ptr as *const ()))
    }

    #[test]
    fn test_extract_type_id() {
        let list = Box::new(ListObject::new());
        let ptr = Box::into_raw(list) as *const ();

        let type_id = extract_type_id(ptr);
        assert_eq!(type_id, TypeId::LIST);

        // Clean up
        unsafe {
            drop(Box::from_raw(ptr as *mut ListObject));
        }
    }

    #[test]
    fn test_type_id_dispatch_coverage() {
        // Verify all handled types compile correctly
        let types = [
            TypeId::FUNCTION,
            TypeId::CLOSURE,
            TypeId::BUILTIN_FUNCTION,
            TypeId::METHOD,
        ];

        for _t in types {
            // Just verify these are valid TypeId values
        }
    }

    #[test]
    fn test_implicit_self_from_slot_none_marker() {
        assert!(implicit_self_from_slot(Value::none()).is_none());
        let self_value = Value::int(7).unwrap();
        assert_eq!(implicit_self_from_slot(self_value), Some(self_value));
    }

    #[test]
    fn test_call_method_user_function_without_implicit_self() {
        let mut vm = VirtualMachine::new();
        push_test_frame(&mut vm, "caller");

        let (func_ptr, func_value) = make_test_function_value("callee");
        vm.current_frame_mut().set_reg(1, func_value);
        vm.current_frame_mut().set_reg(2, Value::none()); // None marker => no implicit self
        vm.current_frame_mut().set_reg(3, Value::int(42).unwrap());

        let inst = Instruction::new(Opcode::CallMethod, 0, 1, 1);
        let control = call_method(&mut vm, inst);
        assert!(matches!(control, ControlFlow::Continue));
        assert_eq!(vm.call_depth(), 2);
        assert_eq!(vm.current_frame().get_reg(0).as_int(), Some(42));
        assert!(vm.current_frame().get_reg(1).is_none());

        vm.clear_frames();
        unsafe {
            drop(Box::from_raw(func_ptr));
        }
    }

    #[test]
    fn test_call_method_user_function_with_implicit_self() {
        let mut vm = VirtualMachine::new();
        push_test_frame(&mut vm, "caller");

        let (func_ptr, func_value) = make_test_function_value("callee");
        vm.current_frame_mut().set_reg(1, func_value);
        vm.current_frame_mut().set_reg(2, Value::int(7).unwrap());
        vm.current_frame_mut().set_reg(3, Value::int(42).unwrap());

        let inst = Instruction::new(Opcode::CallMethod, 0, 1, 1);
        let control = call_method(&mut vm, inst);
        assert!(matches!(control, ControlFlow::Continue));
        assert_eq!(vm.call_depth(), 2);
        assert_eq!(vm.current_frame().get_reg(0).as_int(), Some(7));
        assert_eq!(vm.current_frame().get_reg(1).as_int(), Some(42));

        vm.clear_frames();
        unsafe {
            drop(Box::from_raw(func_ptr));
        }
    }

    #[test]
    fn test_call_method_builtin_respects_none_marker() {
        let mut vm = VirtualMachine::new();
        push_test_frame(&mut vm, "caller");

        let (builtin_ptr, builtin_value) = make_builtin_value("argc");
        vm.current_frame_mut().set_reg(1, builtin_value);
        vm.current_frame_mut().set_reg(2, Value::none()); // None marker => no implicit self
        vm.current_frame_mut().set_reg(3, Value::int(99).unwrap());

        let inst = Instruction::new(Opcode::CallMethod, 0, 1, 1);
        let control = call_method(&mut vm, inst);
        assert!(matches!(control, ControlFlow::Continue));
        assert_eq!(vm.call_depth(), 1);
        assert_eq!(vm.current_frame().get_reg(0).as_int(), Some(1));

        vm.clear_frames();
        unsafe {
            drop(Box::from_raw(builtin_ptr));
        }
    }

    #[test]
    fn test_call_method_builtin_includes_implicit_self_when_present() {
        let mut vm = VirtualMachine::new();
        push_test_frame(&mut vm, "caller");

        let (builtin_ptr, builtin_value) = make_builtin_value("argc");
        vm.current_frame_mut().set_reg(1, builtin_value);
        vm.current_frame_mut().set_reg(2, Value::int(1).unwrap());
        vm.current_frame_mut().set_reg(3, Value::int(99).unwrap());

        let inst = Instruction::new(Opcode::CallMethod, 0, 1, 1);
        let control = call_method(&mut vm, inst);
        assert!(matches!(control, ControlFlow::Continue));
        assert_eq!(vm.call_depth(), 1);
        assert_eq!(vm.current_frame().get_reg(0).as_int(), Some(2));

        vm.clear_frames();
        unsafe {
            drop(Box::from_raw(builtin_ptr));
        }
    }
}
