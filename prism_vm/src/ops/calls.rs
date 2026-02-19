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
use crate::builtins::{EXCEPTION_TYPE_ID, ExceptionTypeObject};
use crate::dispatch::ControlFlow;
use crate::error::RuntimeError;
use crate::frame::ClosureEnv;
use prism_compiler::bytecode::{CodeFlags, CodeObject, Instruction};
use prism_core::intern::{intern, interned_by_ptr};
use prism_core::Value;
use prism_runtime::object::ObjectHeader;
use prism_runtime::object::type_obj::TypeId;
use prism_runtime::types::Cell;
use prism_runtime::types::dict::DictObject;
use prism_runtime::types::function::FunctionObject;
use prism_runtime::types::string::StringObject;
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
            _ if type_id == EXCEPTION_TYPE_ID => {
                // Exception type object - call to construct an exception instance
                let exc_type = unsafe { &*(ptr as *const ExceptionTypeObject) };

                // Collect arguments from registers
                let frame = vm.current_frame();
                let args: Vec<Value> = (0..argc)
                    .map(|i| frame.get_reg(dst_reg + 1 + i as u8))
                    .collect();

                // Call the exception type's call method
                match exc_type.call(&args) {
                    Ok(result) => {
                        vm.current_frame_mut().set_reg(dst_reg, result);
                        ControlFlow::Continue
                    }
                    Err(e) => ControlFlow::Error(RuntimeError::type_error(e.to_string())),
                }
            }
            TypeId::FUNCTION | TypeId::CLOSURE => {
                let func = unsafe { &*(ptr as *const FunctionObject) };
                let code = &func.code;

                // Fast path for exact-arity positional calls without advanced binding.
                let defaults_empty = func.defaults.as_ref().is_none_or(|d| d.is_empty());
                let kwdefaults_empty = func.kwdefaults.as_ref().is_none_or(|d| d.is_empty());
                let simple_positional = !code.flags.contains(CodeFlags::VARARGS)
                    && !code.flags.contains(CodeFlags::VARKEYWORDS)
                    && code.kwonlyarg_count == 0
                    && defaults_empty
                    && kwdefaults_empty
                    && argc == code.arg_count as usize;

                if simple_positional {
                    let closure = vm.lookup_function_closure(ptr);
                    let caller_frame_idx = vm.call_depth() - 1;

                    if let Err(e) =
                        vm.push_frame_with_closure(Arc::clone(code), dst_reg, closure)
                    {
                        return ControlFlow::Error(e);
                    }

                    for i in 0..argc {
                        let arg = vm.frames[caller_frame_idx].get_reg(dst_reg + 1 + i as u8);
                        vm.current_frame_mut().set_reg(i as u8, arg);
                    }

                    initialize_closure_cellvars_from_locals(vm.current_frame_mut(), argc);
                    ControlFlow::Continue
                } else {
                    // Use full binder for defaults, *args/**kwargs, and kw-only semantics.
                    call_kw_user_function(vm, ptr, dst_reg, argc, 0, 0)
                }
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
    let closure = vm.lookup_function_closure(ptr);

    // Extract function signature metadata
    let arg_count = code.arg_count as usize;
    let kwonly_count = code.kwonlyarg_count as usize;
    let total_params = arg_count + kwonly_count;
    let has_varargs = code.flags.contains(CodeFlags::VARARGS);
    let has_varkw = code.flags.contains(CodeFlags::VARKEYWORDS);

    // Optimization: use u64 bitmap for tracking bound parameters (supports ≤64 params)
    // Falls back to linear check for functions with >64 params (extremely rare)
    // Fast-path for common signatures with a zero-allocation bitmap.
    // Large signatures transparently fall back to heap flags.
    let mut bound = BoundArgs::new(total_params);

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
        bound.set_bound(i);
    }

    // Handle excess positional arguments
    // We defer allocation to Phase 5 when we have access to the allocator
    let varargs_values: Option<SmallVec<[Value; 8]>> = if posargc > arg_count {
        if has_varargs {
            // Collect excess for later allocation into *args tuple
            let excess_count = posargc - arg_count;
            let mut excess: SmallVec<[Value; 8]> = SmallVec::with_capacity(excess_count);
            for i in arg_count..posargc {
                let arg_val = vm.frames[caller_frame_idx].get_reg(dst_reg + 1 + i as u8);
                excess.push(arg_val);
            }
            Some(excess)
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
        // Empty *args tuple (will create empty tuple during Phase 5)
        Some(SmallVec::new())
    } else {
        None
    };

    // =========================================================================
    // Phase 2: Bind keyword arguments
    // =========================================================================

    // Prepare varkw entries - defer dict allocation to Phase 5
    let mut varkw_entries: Option<SmallVec<[(Value, Value); 4]>> = if has_varkw {
        Some(SmallVec::new())
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
                    if bound.is_bound(param_idx) {
                        return ControlFlow::Error(RuntimeError::type_error(format!(
                            "{}() got multiple values for argument '{}'",
                            code.name, kw_name
                        )));
                    }
                    bound_args[param_idx] = kw_val;
                    bound.set_bound(param_idx);
                } else if let Some(ref mut varkw_list) = varkw_entries {
                    // Store entry for later allocation into **kwargs dict
                    let key = create_string_key(kw_name);
                    varkw_list.push((key, kw_val));
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
        if !bound.is_bound(i) {
            // Try to get default value
            if let Some(default_val) = func.get_default(i) {
                bound_args[i] = default_val;
                bound.set_bound(i);
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
        if !bound.is_bound(i) {
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
                bound.set_bound(i);
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

    if let Err(e) = vm.push_frame_with_closure(Arc::clone(&code), dst_reg, closure) {
        return ControlFlow::Error(e);
    }

    // Allocate *args tuple on GC heap if needed
    let varargs_value = if has_varargs {
        let tuple = match &varargs_values {
            Some(vals) if !vals.is_empty() => TupleObject::from_slice(&vals),
            _ => TupleObject::empty(),
        };
        match vm.allocator().alloc(tuple) {
            Some(ptr) => Some(Value::object_ptr(ptr as *const ())),
            None => {
                return ControlFlow::Error(RuntimeError::internal(
                    "out of memory: failed to allocate varargs tuple",
                ));
            }
        }
    } else {
        None
    };

    // Allocate **kwargs dict on GC heap if needed
    let varkw_value = if has_varkw {
        let mut dict = DictObject::new();
        if let Some(entries) = &varkw_entries {
            for (key, val) in entries {
                dict.set(*key, *val);
            }
        }
        match vm.allocator().alloc(dict) {
            Some(ptr) => Some(Value::object_ptr(ptr as *const ())),
            None => {
                return ControlFlow::Error(RuntimeError::internal(
                    "out of memory: failed to allocate varkw dict",
                ));
            }
        }
    } else {
        None
    };

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
    if let Some(tuple_val) = varargs_value {
        new_frame.set_reg(local_idx, tuple_val);
        local_idx += 1;
    }

    // Set keyword-only parameters
    for i in arg_count..total_params {
        new_frame.set_reg(local_idx, bound_args[i]);
        local_idx += 1;
    }

    // Set **kwargs dict if present
    if let Some(dict_val) = varkw_value {
        new_frame.set_reg(local_idx, dict_val);
    }

    let initialized_local_count = local_idx as usize + usize::from(varkw_value.is_some());
    initialize_closure_cellvars_from_locals(new_frame, initialized_local_count);
    ControlFlow::Continue
}

/// Tracks which parameters have been bound during argument binding.
///
/// Uses a `u64` bitset for common small signatures and a heap-allocated
/// boolean array for large signatures.
enum BoundArgs {
    Inline(u64),
    Heap(Box<[bool]>),
}

impl BoundArgs {
    #[inline]
    fn new(total_params: usize) -> Self {
        if total_params <= u64::BITS as usize {
            Self::Inline(0)
        } else {
            Self::Heap(vec![false; total_params].into_boxed_slice())
        }
    }

    #[inline]
    fn is_bound(&self, index: usize) -> bool {
        match self {
            Self::Inline(mask) => {
                debug_assert!(index < u64::BITS as usize);
                (mask & (1u64 << index)) != 0
            }
            Self::Heap(flags) => *flags.get(index).unwrap_or(&false),
        }
    }

    #[inline]
    fn set_bound(&mut self, index: usize) {
        match self {
            Self::Inline(mask) => {
                debug_assert!(index < u64::BITS as usize);
                *mask |= 1u64 << index;
            }
            Self::Heap(flags) => {
                if let Some(slot) = flags.get_mut(index) {
                    *slot = true;
                }
            }
        }
    }
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

/// Create a string key Value for use in **kwargs dict.
#[inline]
fn create_string_key(name: &str) -> Value {
    Value::string(intern(name))
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

/// SetFunctionDefaults: attach positional/kw-only default metadata to a function object.
///
/// - dst: function register
/// - src1: positional defaults tuple register (or None)
/// - src2: keyword-only defaults dict register (or None)
#[inline(always)]
pub fn set_function_defaults(vm: &mut VirtualMachine, inst: Instruction) -> ControlFlow {
    let (func_val, pos_defaults_val, kw_defaults_val) = {
        let frame = vm.current_frame();
        (
            frame.get_reg(inst.dst().0),
            frame.get_reg(inst.src1().0),
            frame.get_reg(inst.src2().0),
        )
    };

    let defaults = match parse_positional_defaults(pos_defaults_val) {
        Ok(v) => v,
        Err(e) => return ControlFlow::Error(e),
    };
    let kwdefaults = match parse_kw_defaults(kw_defaults_val) {
        Ok(v) => v,
        Err(e) => return ControlFlow::Error(e),
    };

    let Some(func_ptr) = func_val.as_object_ptr() else {
        return ControlFlow::Error(RuntimeError::type_error(
            "SetFunctionDefaults target is not a function object",
        ));
    };
    let type_id = extract_type_id(func_ptr);
    if type_id != TypeId::FUNCTION && type_id != TypeId::CLOSURE {
        return ControlFlow::Error(RuntimeError::type_error(
            "SetFunctionDefaults target is not a function object",
        ));
    }

    let func = unsafe { &mut *(func_ptr as *mut FunctionObject) };
    func.defaults = defaults;
    func.kwdefaults = kwdefaults;
    ControlFlow::Continue
}

fn parse_positional_defaults(value: Value) -> Result<Option<Box<[Value]>>, RuntimeError> {
    if value.is_none() {
        return Ok(None);
    }

    let Some(ptr) = value.as_object_ptr() else {
        return Err(RuntimeError::type_error(
            "function positional defaults must be tuple or None",
        ));
    };
    if extract_type_id(ptr) != TypeId::TUPLE {
        return Err(RuntimeError::type_error(
            "function positional defaults must be tuple or None",
        ));
    }

    let tuple = unsafe { &*(ptr as *const TupleObject) };
    let values: Vec<Value> = tuple.iter().copied().collect();
    Ok(Some(values.into_boxed_slice()))
}

fn parse_kw_defaults(value: Value) -> Result<Option<Box<[(Arc<str>, Value)]>>, RuntimeError> {
    if value.is_none() {
        return Ok(None);
    }

    let Some(ptr) = value.as_object_ptr() else {
        return Err(RuntimeError::type_error(
            "function keyword defaults must be dict or None",
        ));
    };
    if extract_type_id(ptr) != TypeId::DICT {
        return Err(RuntimeError::type_error(
            "function keyword defaults must be dict or None",
        ));
    }

    let dict = unsafe { &*(ptr as *const DictObject) };
    let mut entries = Vec::with_capacity(dict.len());
    for (key, value) in dict.iter() {
        let key_name = kw_default_key_to_name(key)?;
        entries.push((key_name, value));
    }
    Ok(Some(entries.into_boxed_slice()))
}

fn kw_default_key_to_name(key: Value) -> Result<Arc<str>, RuntimeError> {
    if let Some(ptr) = key.as_string_object_ptr() {
        if let Some(interned) = interned_by_ptr(ptr as *const u8) {
            return Ok(interned.get_arc());
        }
        return Err(RuntimeError::type_error(
            "keyword defaults dict contains invalid interned string key",
        ));
    }

    if let Some(ptr) = key.as_object_ptr() {
        if extract_type_id(ptr) == TypeId::STR {
            let string = unsafe { &*(ptr as *const StringObject) };
            return Ok(Arc::from(string.as_str()));
        }
    }

    Err(RuntimeError::type_error(
        "function keyword defaults dict keys must be strings",
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
    let code_idx = inst.imm16();
    let dst = inst.dst().0;

    // Get code object from constant pool (release frame borrow immediately)
    let (code_clone, qualname) = {
        let frame = vm.current_frame();
        let code_val = frame.get_const(code_idx);

        if let Some(code_ptr) = code_val.as_object_ptr() {
            // Reconstruct Arc<CodeObject> from raw pointer
            // SAFETY: The compiler stored Arc::into_raw(Arc<CodeObject>) in the constant pool.
            // We clone the Arc here to increment reference count (the constant pool keeps its copy).
            let code_raw = code_ptr as *const CodeObject;
            // Clone the Arc without taking ownership (the constant pool still owns the original)
            let code = unsafe { Arc::from_raw(code_raw) };
            let code_clone = Arc::clone(&code);
            // Prevent dropping (the constant pool still owns this Arc)
            std::mem::forget(code);
            let qualname = Arc::from(frame.code.name.as_ref());
            (code_clone, qualname)
        } else {
            return ControlFlow::Error(RuntimeError::internal(
                "Invalid code object in constant pool",
            ));
        }
    };

    // Create FunctionObject
    let func = FunctionObject::new(
        code_clone, qualname, None, // defaults - TODO: handle function defaults
        None, // closure - TODO: handle captured variables (use MakeClosure for that)
    );

    // Allocate on GC heap
    let func_ptr = match vm.allocator().alloc(func) {
        Some(ptr) => ptr as *const (),
        None => {
            return ControlFlow::Error(RuntimeError::internal(
                "out of memory: failed to allocate function",
            ));
        }
    };

    vm.current_frame_mut()
        .set_reg(dst, Value::object_ptr(func_ptr));
    ControlFlow::Continue
}

/// MakeClosure: create closure with captured variables
///
/// Creates a FunctionObject with a captured closure environment.
#[inline(always)]
pub fn make_closure(vm: &mut VirtualMachine, inst: Instruction) -> ControlFlow {
    let code_idx = inst.imm16();
    let dst = inst.dst().0;

    // Get code object from constant pool (release frame borrow immediately)
    let (code_clone, qualname) = {
        let frame = vm.current_frame();
        let code_val = frame.get_const(code_idx);

        if let Some(code_ptr) = code_val.as_object_ptr() {
            // Reconstruct Arc<CodeObject> from raw pointer
            // SAFETY: The compiler stored Arc::into_raw(Arc<CodeObject>) in the constant pool.
            // We clone the Arc here to increment reference count (the constant pool keeps its copy).
            let code_raw = code_ptr as *const CodeObject;
            // Clone the Arc without taking ownership (the constant pool still owns the original)
            let code = unsafe { Arc::from_raw(code_raw) };
            let code_clone = Arc::clone(&code);
            // Prevent dropping (the constant pool still owns this Arc)
            std::mem::forget(code);
            let qualname = Arc::from(frame.code.name.as_ref());
            (code_clone, qualname)
        } else {
            return ControlFlow::Error(RuntimeError::internal(
                "Invalid code object in constant pool for closure",
            ));
        }
    };

    let captured_closure = match capture_closure_environment(vm.current_frame(), &code_clone) {
        Ok(env) => env,
        Err(err) => return ControlFlow::Error(err),
    };

    // Create FunctionObject
    let func = FunctionObject::new(
        code_clone, qualname, None, // defaults - TODO: handle function defaults
        None,
    );

    // Allocate on GC heap
    let func_ptr = match vm.allocator().alloc(func) {
        Some(ptr) => ptr as *const (),
        None => {
            return ControlFlow::Error(RuntimeError::internal(
                "out of memory: failed to allocate closure",
            ));
        }
    };

    vm.register_function_closure(func_ptr, Arc::clone(&captured_closure));
    vm.current_frame_mut().set_reg(dst, Value::object_ptr(func_ptr));
    ControlFlow::Continue
}

/// Capture closure cells for a new closure function object.
fn capture_closure_environment(
    frame: &crate::frame::Frame,
    code: &Arc<CodeObject>,
) -> Result<Arc<ClosureEnv>, RuntimeError> {
    let mut cells = Vec::with_capacity(code.cellvars.len() + code.freevars.len());

    // Child cellvars start as unbound cells and become bound when assigned.
    for _ in code.cellvars.iter() {
        cells.push(Arc::new(Cell::unbound()));
    }

    // Freevars capture existing cells from the parent frame when available.
    // If the parent variable is still a plain local, promote the current value
    // into a new captured cell.
    for freevar in code.freevars.iter() {
        if let Some(cell) = capture_parent_cell(frame, freevar.as_ref()) {
            cells.push(cell);
        } else {
            return Err(RuntimeError::name_error(Arc::clone(freevar)));
        }
    }

    Ok(Arc::new(ClosureEnv::new(cells)))
}

/// Resolve a free variable from the parent frame to a capture cell.
fn capture_parent_cell(frame: &crate::frame::Frame, name: &str) -> Option<Arc<Cell>> {
    if let Some(slot) = find_parent_closure_slot(&frame.code, name) {
        if let Some(env) = &frame.closure {
            if slot < env.len() {
                let cell = Arc::clone(env.get_cell(slot));
                // Parameter-backed cellvars may start unbound and are initialized in
                // registers first; sync once on first capture.
                if cell.get().is_none() {
                    if let Some(local_slot) = find_local_slot(&frame.code, name) {
                        if local_slot < parameter_local_count(&frame.code)
                            && local_slot <= u8::MAX as usize
                        {
                            cell.set(frame.get_reg(local_slot as u8));
                        }
                    }
                }
                return Some(cell);
            }
        }
    }

    find_local_slot(&frame.code, name).map(|slot| {
        let value = frame.get_reg(slot as u8);
        Arc::new(Cell::new(value))
    })
}

/// Find a local slot index for `name` in a code object.
fn find_local_slot(code: &CodeObject, name: &str) -> Option<usize> {
    code.locals.iter().position(|local| local.as_ref() == name)
}

/// Find closure slot index for `name` in the parent frame.
///
/// Closure slot layout is `[cellvars..., freevars...]`.
fn find_parent_closure_slot(code: &CodeObject, name: &str) -> Option<usize> {
    if let Some(idx) = code.cellvars.iter().position(|n| n.as_ref() == name) {
        return Some(idx);
    }
    code.freevars
        .iter()
        .position(|n| n.as_ref() == name)
        .map(|idx| code.cellvars.len() + idx)
}

/// Count initialized parameter slots in local layout.
#[inline]
fn parameter_local_count(code: &CodeObject) -> usize {
    let mut count = code.arg_count as usize + code.kwonlyarg_count as usize;
    if code.flags.contains(CodeFlags::VARARGS) {
        count += 1;
    }
    if code.flags.contains(CodeFlags::VARKEYWORDS) {
        count += 1;
    }
    count
}

/// Initialize closure cellvars from already-populated local slots.
///
/// Parameters are first written into local registers during call setup, while
/// cellvars are accessed through the closure environment. This bridge keeps
/// parameter-backed cells bound before closure capture/load occurs.
fn initialize_closure_cellvars_from_locals(
    frame: &mut crate::frame::Frame,
    initialized_locals: usize,
) {
    if initialized_locals == 0 || frame.code.cellvars.is_empty() || frame.code.locals.is_empty() {
        return;
    }
    let Some(env) = frame.closure.as_ref() else {
        return;
    };

    for (cell_idx, cell_name) in frame.code.cellvars.iter().enumerate() {
        let Some(local_slot) = frame
            .code
            .locals
            .iter()
            .position(|name| name.as_ref() == cell_name.as_ref())
        else {
            continue;
        };
        if local_slot >= initialized_locals || local_slot > u8::MAX as usize {
            continue;
        }
        let value = frame.get_reg(local_slot as u8);
        env.set(cell_idx, value);
    }
}

#[cfg(test)]
mod tests {
    use super::BoundArgs;

    #[test]
    fn test_bound_args_inline_tracks_edges() {
        let mut bound = BoundArgs::new(64);
        bound.set_bound(0);
        bound.set_bound(63);

        assert!(bound.is_bound(0));
        assert!(bound.is_bound(63));
        assert!(!bound.is_bound(1));
        assert!(!bound.is_bound(62));
    }

    #[test]
    fn test_bound_args_heap_handles_large_signatures() {
        let mut bound = BoundArgs::new(80);
        bound.set_bound(0);
        bound.set_bound(64);
        bound.set_bound(79);

        assert!(bound.is_bound(0));
        assert!(bound.is_bound(64));
        assert!(bound.is_bound(79));
        assert!(!bound.is_bound(1));
        assert!(!bound.is_bound(78));
    }

    #[test]
    fn test_bound_args_heap_out_of_range_is_safe() {
        let mut bound = BoundArgs::new(65);
        bound.set_bound(70);
        assert!(!bound.is_bound(70));
    }
}
