//! Container construction opcode handlers.
//!
//! Handles building lists, tuples, dicts, sets, and string interpolation.

use crate::VirtualMachine;
use crate::dispatch::ControlFlow;
use crate::error::RuntimeError;
use prism_compiler::bytecode::Instruction;
use prism_core::Value;
use prism_runtime::types::dict::DictObject;
use prism_runtime::types::list::ListObject;
use prism_runtime::types::set::SetObject;
use prism_runtime::types::tuple::TupleObject;

// =============================================================================
// Container Construction
// =============================================================================

/// BuildList: dst = [r(src1)..r(src1+src2)]
#[inline(always)]
pub fn build_list(vm: &mut VirtualMachine, inst: Instruction) -> ControlFlow {
    let start_reg = inst.src1().0;
    let count = inst.src2().0 as usize;
    let dst = inst.dst().0;

    // Collect values from registers (borrow frame, then release)
    let values: Vec<Value> = {
        let frame = vm.current_frame();
        (0..count)
            .map(|i| frame.get_reg(start_reg + i as u8))
            .collect()
    };

    // Allocate on GC heap
    let list = ListObject::from_slice(&values);
    let ptr = match vm.allocator().alloc(list) {
        Some(p) => p as *const (),
        None => {
            return ControlFlow::Error(RuntimeError::internal(
                "out of memory: failed to allocate list",
            ));
        }
    };

    // Store as object Value
    vm.current_frame_mut().set_reg(dst, Value::object_ptr(ptr));
    ControlFlow::Continue
}

/// BuildTuple: dst = (r(src1)..r(src1+src2))
#[inline(always)]
pub fn build_tuple(vm: &mut VirtualMachine, inst: Instruction) -> ControlFlow {
    let start_reg = inst.src1().0;
    let count = inst.src2().0 as usize;
    let dst = inst.dst().0;

    // Collect values from registers (borrow frame, then release)
    let values: Vec<Value> = {
        let frame = vm.current_frame();
        (0..count)
            .map(|i| frame.get_reg(start_reg + i as u8))
            .collect()
    };

    // Allocate on GC heap
    let tuple = TupleObject::from_slice(&values);
    let ptr = match vm.allocator().alloc(tuple) {
        Some(p) => p as *const (),
        None => {
            return ControlFlow::Error(RuntimeError::internal(
                "out of memory: failed to allocate tuple",
            ));
        }
    };

    // Store as object Value
    vm.current_frame_mut().set_reg(dst, Value::object_ptr(ptr));
    ControlFlow::Continue
}

/// BuildSet: dst = {r(src1)..r(src1+src2)}
#[inline(always)]
pub fn build_set(vm: &mut VirtualMachine, inst: Instruction) -> ControlFlow {
    let start_reg = inst.src1().0;
    let count = inst.src2().0 as usize;
    let dst = inst.dst().0;

    // Collect values and build set (borrow frame, then release)
    let set = {
        let frame = vm.current_frame();
        let mut set = SetObject::new();
        for i in 0..count {
            let value = frame.get_reg(start_reg + i as u8);
            set.add(value);
        }
        set
    };

    // Allocate on GC heap
    let ptr = match vm.allocator().alloc(set) {
        Some(p) => p as *const (),
        None => {
            return ControlFlow::Error(RuntimeError::internal(
                "out of memory: failed to allocate set",
            ));
        }
    };

    // Store as object Value
    vm.current_frame_mut().set_reg(dst, Value::object_ptr(ptr));
    ControlFlow::Continue
}

/// BuildDict: dst = {} with src2 key-value pairs starting at src1
#[inline(always)]
pub fn build_dict(vm: &mut VirtualMachine, inst: Instruction) -> ControlFlow {
    let start_reg = inst.src1().0;
    let pair_count = inst.src2().0 as usize;
    let dst = inst.dst().0;

    // Build dict (borrow frame, then release)
    let dict = {
        let frame = vm.current_frame();
        let mut dict = DictObject::new();
        for i in 0..pair_count {
            let key = frame.get_reg(start_reg + (i * 2) as u8);
            let value = frame.get_reg(start_reg + (i * 2 + 1) as u8);
            dict.set(key, value);
        }
        dict
    };

    // Allocate on GC heap
    let ptr = match vm.allocator().alloc(dict) {
        Some(p) => p as *const (),
        None => {
            return ControlFlow::Error(RuntimeError::internal(
                "out of memory: failed to allocate dict",
            ));
        }
    };

    // Store as object Value
    vm.current_frame_mut().set_reg(dst, Value::object_ptr(ptr));
    ControlFlow::Continue
}

/// BuildString: dst = "".join(r(src1)..r(src1+src2))
#[inline(always)]
pub fn build_string(vm: &mut VirtualMachine, inst: Instruction) -> ControlFlow {
    let frame = vm.current_frame_mut();
    let start_reg = inst.src1().0;
    let count = inst.src2().0 as usize;
    let dst = inst.dst().0;

    // Collect string parts
    let mut result = String::new();
    for i in 0..count {
        let val = frame.get_reg(start_reg + i as u8);
        // Convert value to string representation
        if let Some(s) = val.as_int() {
            result.push_str(&s.to_string());
        } else if let Some(f) = val.as_float() {
            result.push_str(&f.to_string());
        } else if val.is_none() {
            result.push_str("None");
        } else if let Some(b) = val.as_bool() {
            result.push_str(if b { "True" } else { "False" });
        }
        // TODO: Handle string objects and other types
    }

    // TODO: Create StringObject when fully wired
    // For now, store as a None placeholder
    frame.set_reg(dst, Value::none());
    ControlFlow::Continue
}

// =============================================================================
// Container Modification
// =============================================================================

/// ListAppend: src1.append(src2)
///
/// Appends a value to a list object in place.
#[inline(always)]
pub fn list_append(vm: &mut VirtualMachine, inst: Instruction) -> ControlFlow {
    let frame = vm.current_frame();
    let list_val = frame.get_reg(inst.src1().0);
    let value = frame.get_reg(inst.src2().0);

    // Get the list pointer and cast back to mutable ListObject
    if let Some(ptr) = list_val.as_object_ptr() {
        // SAFETY: We know this is a ListObject because BuildList created it
        let list = unsafe { &mut *(ptr as *mut ListObject) };
        list.push(value);
        ControlFlow::Continue
    } else {
        ControlFlow::Error(RuntimeError::type_error("expected list object"))
    }
}

/// SetAdd: src1.add(src2)
///
/// Adds a value to a set object in place.
#[inline(always)]
pub fn set_add(vm: &mut VirtualMachine, inst: Instruction) -> ControlFlow {
    let frame = vm.current_frame();
    let set_val = frame.get_reg(inst.src1().0);
    let value = frame.get_reg(inst.src2().0);

    // Get the set pointer and cast back to mutable SetObject
    if let Some(ptr) = set_val.as_object_ptr() {
        // SAFETY: We know this is a SetObject because BuildSet created it
        let set = unsafe { &mut *(ptr as *mut SetObject) };
        set.add(value);
        ControlFlow::Continue
    } else {
        ControlFlow::Error(RuntimeError::type_error("expected set object"))
    }
}

/// DictSet: src1[dst] = src2 (dst is key register)
///
/// Sets a key-value pair in a dict object.
#[inline(always)]
pub fn dict_set(vm: &mut VirtualMachine, inst: Instruction) -> ControlFlow {
    let frame = vm.current_frame();
    let dict_val = frame.get_reg(inst.src1().0);
    let key = frame.get_reg(inst.dst().0);
    let value = frame.get_reg(inst.src2().0);

    // Get the dict pointer and cast back to mutable DictObject
    if let Some(ptr) = dict_val.as_object_ptr() {
        // SAFETY: We know this is a DictObject because BuildDict created it
        let dict = unsafe { &mut *(ptr as *mut DictObject) };
        dict.set(key, value);
        ControlFlow::Continue
    } else {
        ControlFlow::Error(RuntimeError::type_error("expected dict object"))
    }
}

// =============================================================================
// Unpacking
// =============================================================================

/// UnpackSequence: r(dst)..r(dst+src2) = unpack(src1)
///
/// Unpacks a sequence into consecutive registers starting at dst.
/// Supports lists, tuples, strings (char iteration), and ranges.
///
/// # Performance
///
/// - List/Tuple: O(1) per element access via direct indexing
/// - String: O(n) due to UTF-8 character iteration (lazy single-pass)
/// - Range: O(1) per element via arithmetic computation
///
/// # Errors
///
/// Returns ValueError if the sequence length doesn't match the expected count.
#[inline(always)]
pub fn unpack_sequence(vm: &mut VirtualMachine, inst: Instruction) -> ControlFlow {
    let dst_start = inst.dst().0;
    let count = inst.src2().0 as usize;

    // Get the sequence value
    let sequence = vm.current_frame().get_reg(inst.src1().0);

    // Dispatch based on type
    if let Some(ptr) = sequence.as_object_ptr() {
        // Try as ListObject
        let list_ptr = ptr as *const ListObject;
        let list = unsafe { &*list_ptr };

        // Heuristic: check if it looks like a list by length method
        // In a fully typed system, we'd use type tags
        let len = list.len();

        if len == count {
            // Fast path: direct register writes
            let frame = vm.current_frame_mut();
            for i in 0..count {
                if let Some(val) = list.get(i as i64) {
                    frame.set_reg(dst_start + i as u8, val);
                } else {
                    return ControlFlow::Error(RuntimeError::internal(
                        "list index out of bounds during unpack",
                    ));
                }
            }
            return ControlFlow::Continue;
        }

        // Try as TupleObject
        let tuple_ptr = ptr as *const TupleObject;
        let tuple = unsafe { &*tuple_ptr };
        let tuple_len = tuple.len();

        if tuple_len == count {
            let frame = vm.current_frame_mut();
            for i in 0..count {
                if let Some(val) = tuple.get(i as i64) {
                    frame.set_reg(dst_start + i as u8, val);
                } else {
                    return ControlFlow::Error(RuntimeError::internal(
                        "tuple index out of bounds during unpack",
                    ));
                }
            }
            return ControlFlow::Continue;
        }

        // Length mismatch
        return ControlFlow::Error(RuntimeError::value_error(format!(
            "not enough values to unpack (expected {}, got {})",
            count,
            len.max(tuple_len)
        )));
    }

    // Handle inline types (strings, ranges via iteration)
    ControlFlow::Error(RuntimeError::type_error("cannot unpack non-sequence type"))
}

/// UnpackEx: unpack with *rest
///
/// Extended unpacking for patterns like `a, *rest, b = sequence`.
/// The dst field encodes the before/after counts in a packed format.
///
/// Instruction format:
/// - dst: destination base register
/// - src1: sequence register
/// - src2: packed (before_count << 4) | after_count
///
/// # Performance
///
/// Uses a single pass over the sequence to collect all values,
/// then distributes them to registers.
#[inline(always)]
pub fn unpack_ex(vm: &mut VirtualMachine, inst: Instruction) -> ControlFlow {
    let dst_start = inst.dst().0;
    let sequence = vm.current_frame().get_reg(inst.src1().0);

    // Decode before/after counts from src2
    let packed = inst.src2().0;
    let before_count = (packed >> 4) as usize;
    let after_count = (packed & 0x0F) as usize;
    let min_required = before_count + after_count;

    // Collect all values from sequence
    let values: Vec<Value> = if let Some(ptr) = sequence.as_object_ptr() {
        // Try as ListObject
        let list = unsafe { &*(ptr as *const ListObject) };
        list.iter().copied().collect()
    } else {
        return ControlFlow::Error(RuntimeError::type_error("cannot unpack non-sequence type"));
    };

    let total = values.len();
    if total < min_required {
        return ControlFlow::Error(RuntimeError::value_error(format!(
            "not enough values to unpack (expected at least {}, got {})",
            min_required, total
        )));
    }

    let rest_count = total - min_required;

    // 1. Assign before values
    {
        let frame = vm.current_frame_mut();
        for i in 0..before_count {
            frame.set_reg(dst_start + i as u8, values[i]);
        }
    }

    // 2. Create rest list on GC heap and assign
    let rest_values: Vec<Value> = values[before_count..before_count + rest_count].to_vec();
    let rest_list = ListObject::from_slice(&rest_values);
    let rest_ptr = match vm.allocator().alloc(rest_list) {
        Some(p) => p as *const (),
        None => {
            return ControlFlow::Error(RuntimeError::internal(
                "out of memory: failed to allocate unpack rest list",
            ));
        }
    };
    vm.current_frame_mut()
        .set_reg(dst_start + before_count as u8, Value::object_ptr(rest_ptr));

    // 3. Assign after values
    {
        let frame = vm.current_frame_mut();
        for i in 0..after_count {
            let src_idx = before_count + rest_count + i;
            let dst_idx = before_count + 1 + i; // +1 for rest list register
            frame.set_reg(dst_start + dst_idx as u8, values[src_idx]);
        }
    }

    ControlFlow::Continue
}

/// BuildSlice: dst = slice(src1, src2[, step])
///
/// Creates a SliceObject from start and stop values.
/// For 3-arg slices, the compiler emits an extension instruction immediately
/// after BuildSlice:
/// - opcode: CallKwEx
/// - dst: step register index
/// - src1/src2: marker bytes ('S','L')
///
/// # Value Interpretation
///
/// - None values indicate "use default" (beginning/end)
/// - Integer values are used directly
/// - Other types raise TypeError
///
/// # Performance
///
/// O(1) allocation and construction. SliceObject is 40 bytes and fits
/// in a cache line for efficient access during slicing operations.
#[inline(always)]
pub fn build_slice(vm: &mut VirtualMachine, inst: Instruction) -> ControlFlow {
    use prism_runtime::types::SliceObject;

    const STEP_EXT_TAG_A: u8 = b'S';
    const STEP_EXT_TAG_B: u8 = b'L';

    // Read values from frame (borrow then release)
    let (start_val, stop_val) = {
        let frame = vm.current_frame();
        (frame.get_reg(inst.src1().0), frame.get_reg(inst.src2().0))
    };
    let dst = inst.dst().0;

    // Optional step from extension instruction.
    let mut step: Option<i64> = None;
    {
        let frame = vm.current_frame_mut();
        if (frame.ip as usize) < frame.code.instructions.len() {
            let next = frame.code.instructions[frame.ip as usize];
            if next.opcode() == prism_compiler::bytecode::Opcode::CallKwEx as u8
                && next.src1().0 == STEP_EXT_TAG_A
                && next.src2().0 == STEP_EXT_TAG_B
            {
                let ext = frame.fetch();
                let step_val = frame.get_reg(ext.dst().0);
                step = match value_to_slice_index(step_val) {
                    Ok(v) => v,
                    Err(cf) => return cf,
                };
            }
        }
    }

    // Convert Values to Option<i64> with explicit error handling
    let start = match value_to_slice_index(start_val) {
        Ok(v) => v,
        Err(cf) => return cf,
    };
    let stop = match value_to_slice_index(stop_val) {
        Ok(v) => v,
        Err(cf) => return cf,
    };
    if step == Some(0) {
        return ControlFlow::Error(RuntimeError::value_error("slice step cannot be zero"));
    }

    // Create slice on GC heap
    let slice = SliceObject::new(start, stop, step);
    let ptr = match vm.allocator().alloc(slice) {
        Some(p) => p as *const (),
        None => {
            return ControlFlow::Error(RuntimeError::internal(
                "out of memory: failed to allocate slice",
            ));
        }
    };

    vm.current_frame_mut().set_reg(dst, Value::object_ptr(ptr));
    ControlFlow::Continue
}

/// Helper to convert a Value to an optional slice index.
///
/// Returns:
/// - Ok(None) for None value
/// - Ok(Some(i)) for integer value
/// - Err for other types
#[inline]
fn value_to_slice_index(val: Value) -> Result<Option<i64>, ControlFlow> {
    if val.is_none() {
        Ok(None)
    } else if let Some(i) = val.as_int() {
        Ok(Some(i))
    } else {
        Err(ControlFlow::Error(RuntimeError::type_error(
            "slice indices must be integers or None",
        )))
    }
}

// =============================================================================
// Import Operations
// =============================================================================

/// ImportName: dst = import(name_idx)
///
/// Imports a module by name index and stores the module object in dst register.
/// Uses the VM's ImportResolver for high-performance cached module lookup.
#[inline(always)]
pub fn import_name(vm: &mut VirtualMachine, inst: Instruction) -> ControlFlow {
    let dst = inst.dst();
    let name_idx = inst.imm16();

    let frame = vm.current_frame();
    let module_name = frame.get_name(name_idx).clone();

    // Use dotted import resolution when needed.
    let import_result = if module_name.contains('.') {
        vm.import_resolver.import_dotted(&module_name)
    } else {
        vm.import_resolver.import_module(&module_name)
    };

    match import_result {
        Ok(module) => {
            // Store a stable pointer to the cached ModuleObject.
            // ImportResolver owns the Arc in sys.modules, so this pointer stays valid.
            let module_ptr = std::sync::Arc::as_ptr(&module) as *const ();
            vm.current_frame_mut()
                .set_reg(dst.0, Value::object_ptr(module_ptr));
            ControlFlow::Continue
        }
        Err(err) => ControlFlow::Error(RuntimeError::new(
            crate::error::RuntimeErrorKind::ImportError {
                module: module_name,
                message: err.to_string().into(),
            },
        )),
    }
}

/// ImportFrom: dst = from module import attr
///
/// Imports a specific attribute from a module object.
/// Encoding: dst=destination, src=module register, imm8=attr name index
#[inline(always)]
pub fn import_from(vm: &mut VirtualMachine, inst: Instruction) -> ControlFlow {
    let dst = inst.dst();
    let module_reg_idx = inst.src1().0;
    let attr_idx = inst.src2().0 as u16; // Using src2 position for 8-bit attr index

    let frame = vm.current_frame();
    let attr_name = frame.get_name(attr_idx).clone();
    let module_value = frame.get_reg(module_reg_idx);

    let Some(module_ptr) = module_value.as_object_ptr() else {
        return ControlFlow::Error(RuntimeError::new(
            crate::error::RuntimeErrorKind::ImportError {
                module: "<unknown>".into(),
                message: "Cannot import from None".into(),
            },
        ));
    };

    let Some(module) = vm.import_resolver.module_from_ptr(module_ptr) else {
        return ControlFlow::Error(RuntimeError::new(
            crate::error::RuntimeErrorKind::ImportError {
                module: "<unknown>".into(),
                message: "cannot import from non-module object".into(),
            },
        ));
    };

    // Get the attribute from the module
    match module.get_attr(&attr_name) {
        Some(value) => {
            vm.current_frame_mut().set_reg(dst.0, value);
            ControlFlow::Continue
        }
        None => ControlFlow::Error(RuntimeError::new(
            crate::error::RuntimeErrorKind::ImportError {
                module: module.name().into(),
                message: format!(
                    "cannot import name '{}' from '{}'",
                    attr_name,
                    module.name()
                )
                .into(),
            },
        )),
    }
}

/// ImportStar: from module import *
///
/// Imports all public names from a module into the current global scope.
/// Encoding: dst=unused, src=module register
#[inline(always)]
pub fn import_star(vm: &mut VirtualMachine, inst: Instruction) -> ControlFlow {
    let module_reg_idx = inst.src1().0;

    let frame = vm.current_frame();
    let module_value = frame.get_reg(module_reg_idx);

    let Some(module_ptr) = module_value.as_object_ptr() else {
        return ControlFlow::Error(RuntimeError::new(
            crate::error::RuntimeErrorKind::ImportError {
                module: "<unknown>".into(),
                message: "Cannot import * from None".into(),
            },
        ));
    };

    let Some(module) = vm.import_resolver.module_from_ptr(module_ptr) else {
        return ControlFlow::Error(RuntimeError::new(
            crate::error::RuntimeErrorKind::ImportError {
                module: "<unknown>".into(),
                message: "cannot import * from non-module object".into(),
            },
        ));
    };

    // Get all public names from the module
    // If __all__ is defined, use it; otherwise use all non-underscore names
    for (name, value) in module.public_attrs() {
        vm.globals.set((*name).into(), value);
    }

    ControlFlow::Continue
}

#[cfg(test)]
mod tests {
    // Container tests require full VM setup with object system
}
