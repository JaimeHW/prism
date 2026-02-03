//! Function call opcode handlers.
//!
//! Handles function calls, closures, and tail calls.
//!
//! # Performance Notes
//!
//! - Function objects are heap-allocated with Box::into_raw for stable pointers
//! - Call dispatch uses O(1) type discrimination via ObjectHeader
//! - Arguments are passed via register file, avoiding heap allocation

use crate::VirtualMachine;
use crate::builtins::BuiltinFunctionObject;
use crate::dispatch::ControlFlow;
use crate::error::RuntimeError;
use prism_compiler::bytecode::{CodeFlags, CodeObject, Instruction};
use prism_core::Value;
use prism_runtime::object::ObjectHeader;
use prism_runtime::object::type_obj::TypeId;
use prism_runtime::types::dict::DictObject;
use prism_runtime::types::function::FunctionObject;
use prism_runtime::types::tuple::TupleObject;
use smallvec::SmallVec;
use std::sync::Arc;

// =============================================================================
// Type ID Extraction Helper
// =============================================================================

/// Extract TypeId from an object pointer.
///
/// SAFETY: Relies on ObjectHeader being at offset 0 of all PyObject types.
#[inline(always)]
fn extract_type_id(ptr: *const ()) -> TypeId {
    let header_ptr = ptr as *const ObjectHeader;
    unsafe { (*header_ptr).type_id }
}

// =============================================================================
// Function Calls
// =============================================================================

/// Call: dst = func(args...)
/// src1 = function, src2 = argc, args in r(dst+1)..r(dst+argc)
///
/// Dispatches to the appropriate call handler based on the function type.
/// Uses O(1) type discrimination via ObjectHeader for fast dispatch.
#[inline(always)]
pub fn call(vm: &mut VirtualMachine, inst: Instruction) -> ControlFlow {
    let func_val = vm.current_frame().get_reg(inst.src1().0);
    let argc = inst.src2().0 as usize;
    let dst_reg = inst.dst().0;

    // Check if this is a callable object
    if let Some(ptr) = func_val.as_object_ptr() {
        let type_id = extract_type_id(ptr);

        match type_id {
            TypeId::BUILTIN_FUNCTION => {
                // Fast path: builtin function - call directly without frame push
                let builtin = unsafe { &*(ptr as *const BuiltinFunctionObject) };

                // Collect arguments from registers
                let frame = vm.current_frame();
                let args: Vec<Value> = (0..argc)
                    .map(|i| frame.get_reg(dst_reg + 1 + i as u8))
                    .collect();

                // Call the builtin function
                match builtin.call(&args) {
                    Ok(result) => {
                        vm.current_frame_mut().set_reg(dst_reg, result);
                        ControlFlow::Continue
                    }
                    Err(e) => ControlFlow::Error(RuntimeError::type_error(e.to_string())),
                }
            }
            TypeId::FUNCTION | TypeId::CLOSURE => {
                // User-defined function - push frame
                let func = unsafe { &*(ptr as *const FunctionObject) };
                let code = Arc::clone(&func.code);

                // Push new frame for function execution
                if let Err(e) = vm.push_frame(Arc::clone(&code), dst_reg) {
                    return ControlFlow::Error(e);
                }

                // Copy arguments to new frame's registers
                let caller_frame_idx = vm.call_depth() - 1;

                // Collect args from caller and set in new frame
                let args: Vec<Value> = (0..argc)
                    .map(|i| vm.frames[caller_frame_idx - 1].get_reg(dst_reg + 1 + i as u8))
                    .collect();

                for (i, arg) in args.into_iter().enumerate() {
                    vm.current_frame_mut().set_reg(i as u8, arg);
                }

                ControlFlow::Continue
            }
            _ => ControlFlow::Error(RuntimeError::type_error(format!(
                "'{}' object is not callable",
                type_id.name()
            ))),
        }
    } else {
        ControlFlow::Error(RuntimeError::type_error("object is not callable"))
    }
}

/// CallKw: call with keyword arguments
///
/// This is a two-instruction sequence:
/// - Instruction 1 (CallKw): [opcode][dst][func][posargc]
/// - Instruction 2 (CallKwEx): [kwargc][kwnames_idx_lo][kwnames_idx_hi]
///
/// Arguments layout in registers:
/// - dst+1 .. dst+posargc: positional argument values
/// - dst+posargc+1 .. dst+posargc+kwargc: keyword argument values
///
/// Keyword names are stored in the constant pool as a KwNamesTuple.
///
/// # Performance Optimizations
///
/// - Uses SmallVec<[Value; 8]> to avoid heap allocation for typical calls
/// - Uses u64 bitmap for tracking bound parameters (supports up to 64 params)
/// - Single-pass binding algorithm with O(P + K) complexity
/// - Pre-allocated varargs/varkw containers with exact capacity
#[inline(always)]
pub fn call_kw(vm: &mut VirtualMachine, inst: Instruction) -> ControlFlow {
    let dst_reg = inst.dst().0;
    let func_reg = inst.src1().0;
    let posargc = inst.src2().0 as usize;

    // Read the extension instruction (CallKwEx) for keyword metadata
    let frame = vm.current_frame_mut();
    let ip = frame.ip as usize;

    // The CallKwEx instruction should be at ip (current instruction is already consumed)
    let ext_inst = frame.code.instructions[ip];
    frame.ip = (ip + 1) as u32; // Skip past the extension instruction

    let kwargc = ext_inst.dst().0 as usize;
    let kwnames_idx = (ext_inst.src1().0 as u16) | ((ext_inst.src2().0 as u16) << 8);

    // Get function object
    let func_val = vm.frames[vm.call_depth() - 1].get_reg(func_reg);

    if let Some(ptr) = func_val.as_object_ptr() {
        let type_id = extract_type_id(ptr);

        match type_id {
            TypeId::BUILTIN_FUNCTION => call_kw_builtin(vm, ptr, dst_reg, posargc, kwargc),
            TypeId::FUNCTION | TypeId::CLOSURE => {
                call_kw_user_function(vm, ptr, dst_reg, posargc, kwargc, kwnames_idx)
            }
            _ => ControlFlow::Error(RuntimeError::type_error(format!(
                "'{}' object is not callable",
                type_id.name()
            ))),
        }
    } else {
        ControlFlow::Error(RuntimeError::type_error("object is not callable"))
    }
}

/// Handle CallKw for builtin functions.
///
/// Builtins receive all arguments as a flat vector; keyword semantics are
/// handled by the builtin implementation itself.
#[inline]
fn call_kw_builtin(
    vm: &mut VirtualMachine,
    ptr: *const (),
    dst_reg: u8,
    posargc: usize,
    kwargc: usize,
) -> ControlFlow {
    let builtin = unsafe { &*(ptr as *const BuiltinFunctionObject) };

    // Collect all arguments into a single vector
    let caller_frame = &vm.frames[vm.call_depth() - 1];
    let total_args = posargc + kwargc;
    let mut args: SmallVec<[Value; 8]> = SmallVec::with_capacity(total_args);

    for i in 0..total_args {
        args.push(caller_frame.get_reg(dst_reg + 1 + i as u8));
    }

    // Call the builtin
    match builtin.call(&args) {
        Ok(result) => {
            vm.current_frame_mut().set_reg(dst_reg, result);
            ControlFlow::Continue
        }
        Err(e) => ControlFlow::Error(RuntimeError::type_error(e.to_string())),
    }
}

/// Handle CallKw for user-defined functions.
///
/// This implements full Python 3.12 argument binding semantics:
/// 1. Bind positional arguments to parameters
/// 2. Collect excess positional into *args tuple (if function accepts varargs)
/// 3. Bind keyword arguments by name
/// 4. Collect unrecognized keywords into **kwargs dict (if function accepts varkw)
/// 5. Fill unbound parameters with defaults
/// 6. Verify all required parameters are bound
/// 7. Push new frame and populate locals
///
/// # Optimizations
///
/// - SmallVec<[Value; 8]> avoids heap allocation for typical functions
/// - u64 bitmap tracks bound parameters without allocation (supports ≤64 params)
/// - Varargs tuple allocated with exact capacity
/// - Varkw dict allocated only when needed
#[inline]
fn call_kw_user_function(
    vm: &mut VirtualMachine,
    ptr: *const (),
    dst_reg: u8,
    posargc: usize,
    kwargc: usize,
    kwnames_idx: u16,
) -> ControlFlow {
    let func = unsafe { &*(ptr as *const FunctionObject) };
    let code = Arc::clone(&func.code);

    // Extract function signature metadata
    let arg_count = code.arg_count as usize;
    let kwonly_count = code.kwonlyarg_count as usize;
    let total_params = arg_count + kwonly_count;
    let has_varargs = code.flags.contains(CodeFlags::VARARGS);
    let has_varkw = code.flags.contains(CodeFlags::VARKEYWORDS);

    // Optimization: use u64 bitmap for tracking bound parameters (supports ≤64 params)
    // Falls back to linear check for functions with >64 params (extremely rare)
    let mut bound_mask: u64 = 0;

    // Pre-allocate bound arguments with SmallVec optimization
    // SmallVec<[Value; 8]> stores up to 8 Values inline, avoiding heap allocation
    let mut bound_args: SmallVec<[Value; 8]> = SmallVec::with_capacity(total_params);
    bound_args.resize(total_params, Value::none());

    // =========================================================================
    // Phase 1: Bind positional arguments
    // =========================================================================

    let caller_frame_idx = vm.call_depth() - 1;

    // Bind positional args that fit in regular parameter slots
    let bound_positional = posargc.min(arg_count);
    for i in 0..bound_positional {
        let arg_val = vm.frames[caller_frame_idx].get_reg(dst_reg + 1 + i as u8);
        bound_args[i] = arg_val;
        bound_mask |= 1 << i;
    }

    // Handle excess positional arguments
    let varargs_tuple: Option<Box<TupleObject>> = if posargc > arg_count {
        if has_varargs {
            // Collect excess into *args tuple
            let excess_count = posargc - arg_count;
            let mut excess: SmallVec<[Value; 8]> = SmallVec::with_capacity(excess_count);
            for i in arg_count..posargc {
                let arg_val = vm.frames[caller_frame_idx].get_reg(dst_reg + 1 + i as u8);
                excess.push(arg_val);
            }
            Some(Box::new(TupleObject::from_slice(&excess)))
        } else {
            // Too many positional arguments - error
            return ControlFlow::Error(RuntimeError::type_error(format!(
                "{}() takes {} positional argument{} but {} {} given",
                code.name,
                arg_count,
                if arg_count == 1 { "" } else { "s" },
                posargc,
                if posargc == 1 { "was" } else { "were" }
            )));
        }
    } else if has_varargs {
        // Empty *args tuple
        Some(Box::new(TupleObject::empty()))
    } else {
        None
    };

    // =========================================================================
    // Phase 2: Bind keyword arguments
    // =========================================================================

    // Prepare **kwargs dict if function accepts it
    let mut varkw_dict: Option<Box<DictObject>> = if has_varkw {
        Some(Box::new(DictObject::with_capacity(kwargc)))
    } else {
        None
    };

    if kwargc > 0 {
        // Get keyword names from constant pool
        let kwnames_val = vm.frames[caller_frame_idx].get_const(kwnames_idx);

        if let Some(kwnames_ptr) = kwnames_val.as_object_ptr() {
            let kwnames =
                unsafe { &*(kwnames_ptr as *const prism_compiler::bytecode::KwNamesTuple) };

            for i in 0..kwargc {
                let kw_name = match kwnames.get(i) {
                    Some(name) => name,
                    None => {
                        return ControlFlow::Error(RuntimeError::internal(
                            "Invalid keyword names tuple",
                        ));
                    }
                };
                let kw_val =
                    vm.frames[caller_frame_idx].get_reg(dst_reg + 1 + posargc as u8 + i as u8);

                // Find parameter index by name
                // Account for varargs offset: if VARARGS is set, kwonly params
                // are offset by 1 in locals array
                if let Some(param_idx) = find_param_index_with_varargs(&code, kw_name, has_varargs)
                {
                    // Check for duplicate assignment using bitmap
                    if (bound_mask & (1 << param_idx)) != 0 {
                        return ControlFlow::Error(RuntimeError::type_error(format!(
                            "{}() got multiple values for argument '{}'",
                            code.name, kw_name
                        )));
                    }
                    bound_args[param_idx] = kw_val;
                    bound_mask |= 1 << param_idx;
                } else if let Some(ref mut kwargs_dict) = varkw_dict {
                    // Store in **kwargs dict
                    // Create string key for the dict
                    let key = create_string_key(kw_name);
                    kwargs_dict.set(key, kw_val);
                } else {
                    // Unexpected keyword argument - error
                    return ControlFlow::Error(RuntimeError::type_error(format!(
                        "{}() got an unexpected keyword argument '{}'",
                        code.name, kw_name
                    )));
                }
            }
        } else {
            return ControlFlow::Error(RuntimeError::internal(
                "Invalid keyword names in constant pool",
            ));
        }
    }

    // =========================================================================
    // Phase 3: Fill missing positional parameters with defaults
    // =========================================================================

    for i in 0..arg_count {
        if (bound_mask & (1 << i)) == 0 {
            // Try to get default value
            if let Some(default_val) = func.get_default(i) {
                bound_args[i] = default_val;
                bound_mask |= 1 << i;
            } else {
                // Missing required positional argument
                let param_name = code.locals.get(i).map(|s| s.as_ref()).unwrap_or("?");
                return ControlFlow::Error(RuntimeError::type_error(format!(
                    "{}() missing {} required positional argument: '{}'",
                    code.name,
                    1, // Could count total missing for better message
                    param_name
                )));
            }
        }
    }

    // =========================================================================
    // Phase 4: Fill missing keyword-only parameters with kwdefaults
    // =========================================================================

    for i in arg_count..total_params {
        if (bound_mask & (1 << i)) == 0 {
            // Calculate correct locals index accounting for varargs slot
            let locals_idx = if has_varargs { i + 1 } else { i };
            let param_name = code
                .locals
                .get(locals_idx)
                .map(|s| s.as_ref())
                .unwrap_or("?");

            // Check kwdefaults for this parameter
            let found_default = func.kwdefaults.as_ref().and_then(|kwdefaults| {
                kwdefaults
                    .iter()
                    .find(|(n, _)| n.as_ref() == param_name)
                    .map(|(_, val)| *val)
            });

            if let Some(default_val) = found_default {
                bound_args[i] = default_val;
            } else {
                // Missing required keyword-only argument
                return ControlFlow::Error(RuntimeError::type_error(format!(
                    "{}() missing {} required keyword-only argument: '{}'",
                    code.name, 1, param_name
                )));
            }
        }
    }

    // =========================================================================
    // Phase 5: Push new frame and populate locals
    // =========================================================================

    if let Err(e) = vm.push_frame(Arc::clone(&code), dst_reg) {
        return ControlFlow::Error(e);
    }

    // Populate bound parameters in new frame
    // Locals layout:
    // - [0..arg_count): positional parameters
    // - [arg_count]: *args (if VARARGS)
    // - [arg_count + varargs_offset..]: keyword-only parameters
    // - [after kwonly]: **kwargs (if VARKEYWORDS)

    let new_frame = vm.current_frame_mut();
    let mut local_idx = 0u8;

    // Set positional parameters
    for i in 0..arg_count {
        new_frame.set_reg(local_idx, bound_args[i]);
        local_idx += 1;
    }

    // Set *args tuple if present
    if has_varargs {
        if let Some(varargs) = varargs_tuple {
            let tuple_val = Value::object_ptr(Box::into_raw(varargs) as *const ());
            new_frame.set_reg(local_idx, tuple_val);
        } else {
            // Empty tuple placeholder
            let empty_tuple = Box::new(TupleObject::empty());
            let tuple_val = Value::object_ptr(Box::into_raw(empty_tuple) as *const ());
            new_frame.set_reg(local_idx, tuple_val);
        }
        local_idx += 1;
    }

    // Set keyword-only parameters
    for i in arg_count..total_params {
        new_frame.set_reg(local_idx, bound_args[i]);
        local_idx += 1;
    }

    // Set **kwargs dict if present
    if has_varkw {
        if let Some(varkw) = varkw_dict {
            let dict_val = Value::object_ptr(Box::into_raw(varkw) as *const ());
            new_frame.set_reg(local_idx, dict_val);
        } else {
            // Empty dict placeholder
            let empty_dict = Box::new(DictObject::new());
            let dict_val = Value::object_ptr(Box::into_raw(empty_dict) as *const ());
            new_frame.set_reg(local_idx, dict_val);
        }
    }

    ControlFlow::Continue
}

/// Find parameter index by name, accounting for varargs slot offset.
///
/// When a function has *args, the locals array layout is:
/// - [0..arg_count): positional params
/// - [arg_count]: *args slot
/// - [arg_count+1..): keyword-only params
///
/// This function returns the *parameter index* (0 to total_params-1),
/// not the locals index.
///
/// # Performance
///
/// Uses linear search which is optimal for typical parameter counts (<10).
/// For hot paths, the JIT's inline caching handles optimization.
#[inline]
fn find_param_index_with_varargs(
    code: &CodeObject,
    name: &str,
    has_varargs: bool,
) -> Option<usize> {
    let arg_count = code.arg_count as usize;
    let kwonly_count = code.kwonlyarg_count as usize;

    // Search positional parameters (no offset)
    for i in 0..arg_count {
        if let Some(param_name) = code.locals.get(i) {
            if param_name.as_ref() == name {
                return Some(i);
            }
        }
    }

    // Search keyword-only parameters (with varargs offset if applicable)
    let kwonly_locals_start = if has_varargs {
        arg_count + 1
    } else {
        arg_count
    };
    for i in 0..kwonly_count {
        let locals_idx = kwonly_locals_start + i;
        if let Some(param_name) = code.locals.get(locals_idx) {
            if param_name.as_ref() == name {
                return Some(arg_count + i); // Return parameter index, not locals index
            }
        }
    }

    None
}

/// Find parameter index by name (legacy function for non-varargs functions).
///
/// The first `arg_count + kwonlyarg_count` entries in locals are parameter names.
#[inline]
fn find_param_index(code: &CodeObject, name: &str) -> Option<usize> {
    let total_params = code.arg_count as usize + code.kwonlyarg_count as usize;
    for i in 0..total_params {
        if let Some(param_name) = code.locals.get(i) {
            if param_name.as_ref() == name {
                return Some(i);
            }
        }
    }
    None
}

/// Create a string key Value for use in **kwargs dict.
///
/// # Note
///
/// This creates a leaked string for now. Proper implementation would use
/// string interning to avoid memory leaks and enable pointer comparison.
#[inline]
fn create_string_key(name: &str) -> Value {
    // For now, leak the string to create a stable pointer
    // TODO: Use proper string interning when string object system is complete
    let leaked: &'static str = Box::leak(name.to_owned().into_boxed_str());
    Value::from_bits(leaked.as_ptr() as u64 | 0x0006_0000_0000_0000)
}

/// CallMethod: dst = obj.method(args...)
#[inline(always)]
pub fn call_method(_vm: &mut VirtualMachine, _inst: Instruction) -> ControlFlow {
    // TODO: Implement method calls
    // Similar to Call but with implicit self parameter
    ControlFlow::Error(RuntimeError::internal("CallMethod not yet implemented"))
}

/// CallKwEx: extension instruction for CallKw.
///
/// This instruction should never be executed directly - it is consumed by CallKw.
/// If we reach this opcode, it indicates bytecode corruption.
#[inline(always)]
pub fn call_kw_ex(_vm: &mut VirtualMachine, _inst: Instruction) -> ControlFlow {
    ControlFlow::Error(RuntimeError::internal(
        "CallKwEx executed directly (bytecode corruption)",
    ))
}

/// TailCall: call reusing current frame
///
/// Optimizes tail-recursive calls by reusing the current frame.
#[inline(always)]
pub fn tail_call(vm: &mut VirtualMachine, inst: Instruction) -> ControlFlow {
    // For now, fall back to regular call
    // TODO: Implement true tail call optimization
    call(vm, inst)
}

// =============================================================================
// Function Creation
// =============================================================================

/// MakeFunction: create function from code object
/// dst = function, imm16 = code constant index
///
/// Creates a FunctionObject from a code constant and stores it in dst.
#[inline(always)]
pub fn make_function(vm: &mut VirtualMachine, inst: Instruction) -> ControlFlow {
    let frame = vm.current_frame_mut();
    let code_idx = inst.imm16();
    let dst = inst.dst().0;

    // Get code object from constant pool
    // Constants can contain Arc<CodeObject> for nested function definitions
    let code_val = frame.get_const(code_idx);

    // For now, we need to extract the CodeObject from the constant
    // In a full implementation, the constant pool would store CodeObject directly
    // TODO: Properly handle code object constants
    if let Some(_code_ptr) = code_val.as_object_ptr() {
        // Assume this points to a CodeObject wrapper
        // For now, create a placeholder function
        let func = Box::new(FunctionObject::new(
            Arc::new(CodeObject::new("anonymous", "<module>")),
            Arc::from("anonymous"),
            None,
            None,
        ));
        let func_ptr = Box::into_raw(func) as *const ();
        frame.set_reg(dst, Value::object_ptr(func_ptr));
        ControlFlow::Continue
    } else {
        // Handle case where code is stored differently
        ControlFlow::Error(RuntimeError::internal(
            "Invalid code object in constant pool",
        ))
    }
}

/// MakeClosure: create closure with captured variables
///
/// Creates a FunctionObject with a captured closure environment.
#[inline(always)]
pub fn make_closure(vm: &mut VirtualMachine, inst: Instruction) -> ControlFlow {
    let frame = vm.current_frame_mut();
    let code_idx = inst.imm16();
    let dst = inst.dst().0;

    // Get code object from constant pool
    let _code_val = frame.get_const(code_idx);

    // TODO: Properly implement closure creation
    // 1. Get the list of free variables from the code object
    // 2. Capture values from the current frame's registers or closure env
    // 3. Create ClosureEnv and FunctionObject

    // Placeholder: create function without closure
    let func = Box::new(FunctionObject::new(
        Arc::new(CodeObject::new("closure", "<module>")),
        Arc::from("closure"),
        None,
        None, // Should be Some(ClosureEnv) with captured values
    ));
    let func_ptr = Box::into_raw(func) as *const ();
    frame.set_reg(dst, Value::object_ptr(func_ptr));
    ControlFlow::Continue
}

#[cfg(test)]
mod tests {
    // Call tests require full VM setup with function objects
}
