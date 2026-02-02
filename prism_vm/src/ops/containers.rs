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
    let frame = vm.current_frame_mut();
    let start_reg = inst.src1().0;
    let count = inst.src2().0 as usize;
    let dst = inst.dst().0;

    // Collect values from registers
    let mut values = Vec::with_capacity(count);
    for i in 0..count {
        values.push(frame.get_reg(start_reg + i as u8));
    }

    // Create list on heap and get stable pointer
    let list = Box::new(ListObject::from_slice(&values));
    let ptr = Box::into_raw(list) as *const ();

    // Store as object Value
    frame.set_reg(dst, Value::object_ptr(ptr));
    ControlFlow::Continue
}

/// BuildTuple: dst = (r(src1)..r(src1+src2))
#[inline(always)]
pub fn build_tuple(vm: &mut VirtualMachine, inst: Instruction) -> ControlFlow {
    let frame = vm.current_frame_mut();
    let start_reg = inst.src1().0;
    let count = inst.src2().0 as usize;
    let dst = inst.dst().0;

    // Collect values from registers
    let values: Vec<Value> = (0..count)
        .map(|i| frame.get_reg(start_reg + i as u8))
        .collect();

    // Create tuple on heap and get stable pointer
    let tuple = Box::new(TupleObject::from_slice(&values));
    let ptr = Box::into_raw(tuple) as *const ();

    // Store as object Value
    frame.set_reg(dst, Value::object_ptr(ptr));
    ControlFlow::Continue
}

/// BuildSet: dst = {r(src1)..r(src1+src2)}
#[inline(always)]
pub fn build_set(vm: &mut VirtualMachine, inst: Instruction) -> ControlFlow {
    let frame = vm.current_frame_mut();
    let start_reg = inst.src1().0;
    let count = inst.src2().0 as usize;
    let dst = inst.dst().0;

    // Create set and add elements
    let mut set = SetObject::new();
    for i in 0..count {
        let value = frame.get_reg(start_reg + i as u8);
        set.add(value);
    }

    // Store on heap
    let set_box = Box::new(set);
    let ptr = Box::into_raw(set_box) as *const ();
    frame.set_reg(dst, Value::object_ptr(ptr));
    ControlFlow::Continue
}

/// BuildDict: dst = {} with src2 key-value pairs starting at src1
#[inline(always)]
pub fn build_dict(vm: &mut VirtualMachine, inst: Instruction) -> ControlFlow {
    let frame = vm.current_frame_mut();
    let start_reg = inst.src1().0;
    let pair_count = inst.src2().0 as usize;
    let dst = inst.dst().0;

    // Create dict and insert key-value pairs
    let mut dict = DictObject::new();
    for i in 0..pair_count {
        let key = frame.get_reg(start_reg + (i * 2) as u8);
        let value = frame.get_reg(start_reg + (i * 2 + 1) as u8);
        dict.set(key, value);
    }

    // Store on heap
    let dict_box = Box::new(dict);
    let ptr = Box::into_raw(dict_box) as *const ();
    frame.set_reg(dst, Value::object_ptr(ptr));
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
#[inline(always)]
pub fn unpack_sequence(vm: &mut VirtualMachine, inst: Instruction) -> ControlFlow {
    let frame = vm.current_frame();
    let _sequence = frame.get_reg(inst.src1().0);
    let _count = inst.src2().0;
    let _dst_start = inst.dst().0;

    // TODO: Implement sequence unpacking
    ControlFlow::Error(RuntimeError::internal("UnpackSequence not yet implemented"))
}

/// UnpackEx: unpack with *rest
#[inline(always)]
pub fn unpack_ex(vm: &mut VirtualMachine, inst: Instruction) -> ControlFlow {
    let frame = vm.current_frame();
    let _sequence = frame.get_reg(inst.src1().0);

    // TODO: Implement extended unpacking with *rest
    ControlFlow::Error(RuntimeError::internal("UnpackEx not yet implemented"))
}

/// BuildSlice: dst = slice(src1, src2)
#[inline(always)]
pub fn build_slice(vm: &mut VirtualMachine, inst: Instruction) -> ControlFlow {
    let frame = vm.current_frame();
    let _start = frame.get_reg(inst.src1().0);
    let _stop = frame.get_reg(inst.src2().0);

    // TODO: Create slice object
    ControlFlow::Error(RuntimeError::internal("BuildSlice not yet implemented"))
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

    // Use the VM's import resolver to import the module
    match vm.import_resolver.import_module(&module_name) {
        Ok(module) => {
            // Create a Value from the module
            // For now, we'll store a placeholder - proper object representation TBD
            // TODO: Convert Arc<ModuleObject> to Value with proper heap allocation
            let module_ptr = std::sync::Arc::into_raw(module) as u64;
            let value = Value::from_bits(module_ptr | 0x0004_0000_0000_0000); // Object tag
            vm.current_frame_mut().set_reg(dst.0, value);
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

    // Get the module from the value
    // TODO: Proper object system integration
    let module_ptr =
        (module_value.to_bits() & 0x0000_FFFF_FFFF_FFFF) as *const crate::import::ModuleObject;

    if module_ptr.is_null() {
        return ControlFlow::Error(RuntimeError::new(
            crate::error::RuntimeErrorKind::ImportError {
                module: "<unknown>".into(),
                message: "Cannot import from None".into(),
            },
        ));
    }

    // Safety: We trust the module pointer from a previous ImportName
    let module = unsafe { &*module_ptr };

    // Get the attribute from the module
    match module.get_attr(&attr_name) {
        Some(value) => {
            vm.current_frame_mut().set_reg(dst.0, value);
            ControlFlow::Continue
        }
        None => ControlFlow::Error(RuntimeError::new(
            crate::error::RuntimeErrorKind::ImportError {
                module: module.name().to_string().into(),
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

    // Get the module from the value
    let module_ptr =
        (module_value.to_bits() & 0x0000_FFFF_FFFF_FFFF) as *const crate::import::ModuleObject;

    if module_ptr.is_null() {
        return ControlFlow::Error(RuntimeError::new(
            crate::error::RuntimeErrorKind::ImportError {
                module: "<unknown>".into(),
                message: "Cannot import * from None".into(),
            },
        ));
    }

    // Safety: We trust the module pointer from a previous ImportName
    let module = unsafe { &*module_ptr };

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
