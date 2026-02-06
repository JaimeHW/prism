//! Comparison opcode handlers.
//!
//! Handles all comparison operations: <, <=, ==, !=, >, >=, is, in.

use crate::VirtualMachine;
use crate::dispatch::ControlFlow;
use crate::error::RuntimeError;
use prism_compiler::bytecode::Instruction;
use prism_core::Value;

// =============================================================================
// Numeric Comparisons
// =============================================================================

/// Lt: dst = src1 < src2 (generic with speculative fast-path)
///
/// Uses O(1) speculation cache lookup for type-specialized comparison.
#[inline(always)]
pub fn lt(vm: &mut VirtualMachine, inst: Instruction) -> ControlFlow {
    use crate::ic_manager::ICSiteId;
    use crate::speculative::{SpecResult, Speculation, spec_lt_float, spec_lt_int};
    use crate::type_feedback::{BinaryOpFeedback, OperandPair};

    let frame = vm.current_frame_mut();
    let (a, b) = frame.get_regs2(inst.src1().0, inst.src2().0);
    let code_id = frame.code_id();
    let bc_offset = frame.ip.saturating_sub(1) as u32;
    let site = ICSiteId::new(code_id, bc_offset);

    // =========================================================================
    // Speculative Fast Path (O(1) cache lookup)
    // =========================================================================
    if let Some(spec) = vm.speculation_cache.get(site) {
        match spec {
            Speculation::IntInt => {
                let (result, value) = spec_lt_int(a, b);
                if result == SpecResult::Success {
                    let frame = vm.current_frame_mut();
                    frame.set_reg(inst.dst().0, value);
                    return ControlFlow::Continue;
                }
                vm.speculation_cache.invalidate(site);
            }
            Speculation::FloatFloat | Speculation::IntFloat | Speculation::FloatInt => {
                let (result, value) = spec_lt_float(a, b);
                if result == SpecResult::Success {
                    let frame = vm.current_frame_mut();
                    frame.set_reg(inst.dst().0, value);
                    return ControlFlow::Continue;
                }
                vm.speculation_cache.invalidate(site);
            }
            Speculation::None
            | Speculation::StrStr
            | Speculation::StrInt
            | Speculation::IntStr
            | Speculation::ListList => {}
        }
    }

    // =========================================================================
    // Slow Path: Full type check + feedback recording
    // =========================================================================
    let pair = OperandPair::from_values(a, b);
    let feedback = BinaryOpFeedback::new(code_id, bc_offset, a, b);
    feedback.record(&mut vm.ic_manager);

    let spec = Speculation::from_operand_pair(pair);
    if spec != Speculation::None {
        vm.speculation_cache.insert(site, spec);
    }

    let frame = vm.current_frame_mut();

    // Int comparison
    if let (Some(x), Some(y)) = (a.as_int(), b.as_int()) {
        frame.set_reg(inst.dst().0, Value::bool(x < y));
        return ControlFlow::Continue;
    }

    // Float comparison (including int/float mixed)
    let x = if let Some(f) = a.as_float() {
        Some(f)
    } else if let Some(i) = a.as_int() {
        Some(i as f64)
    } else {
        None
    };
    let y = if let Some(f) = b.as_float() {
        Some(f)
    } else if let Some(i) = b.as_int() {
        Some(i as f64)
    } else {
        None
    };

    match (x, y) {
        (Some(x), Some(y)) => {
            frame.set_reg(inst.dst().0, Value::bool(x < y));
            ControlFlow::Continue
        }
        _ => ControlFlow::Error(RuntimeError::unsupported_operand("<", "unknown", "unknown")),
    }
}

/// Le: dst = src1 <= src2 (generic with speculative fast-path)
#[inline(always)]
pub fn le(vm: &mut VirtualMachine, inst: Instruction) -> ControlFlow {
    use crate::ic_manager::ICSiteId;
    use crate::speculative::{SpecResult, Speculation, spec_le_float, spec_le_int};
    use crate::type_feedback::{BinaryOpFeedback, OperandPair};

    let frame = vm.current_frame_mut();
    let (a, b) = frame.get_regs2(inst.src1().0, inst.src2().0);
    let code_id = frame.code_id();
    let bc_offset = frame.ip.saturating_sub(1) as u32;
    let site = ICSiteId::new(code_id, bc_offset);

    // Speculative Fast Path
    if let Some(spec) = vm.speculation_cache.get(site) {
        match spec {
            Speculation::IntInt => {
                let (result, value) = spec_le_int(a, b);
                if result == SpecResult::Success {
                    let frame = vm.current_frame_mut();
                    frame.set_reg(inst.dst().0, value);
                    return ControlFlow::Continue;
                }
                vm.speculation_cache.invalidate(site);
            }
            Speculation::FloatFloat | Speculation::IntFloat | Speculation::FloatInt => {
                let (result, value) = spec_le_float(a, b);
                if result == SpecResult::Success {
                    let frame = vm.current_frame_mut();
                    frame.set_reg(inst.dst().0, value);
                    return ControlFlow::Continue;
                }
                vm.speculation_cache.invalidate(site);
            }
            Speculation::None
            | Speculation::StrStr
            | Speculation::StrInt
            | Speculation::IntStr
            | Speculation::ListList => {}
        }
    }

    // Slow Path
    let pair = OperandPair::from_values(a, b);
    let feedback = BinaryOpFeedback::new(code_id, bc_offset, a, b);
    feedback.record(&mut vm.ic_manager);
    let spec = Speculation::from_operand_pair(pair);
    if spec != Speculation::None {
        vm.speculation_cache.insert(site, spec);
    }

    let frame = vm.current_frame_mut();
    if let (Some(x), Some(y)) = (a.as_int(), b.as_int()) {
        frame.set_reg(inst.dst().0, Value::bool(x <= y));
        return ControlFlow::Continue;
    }

    let x = a.as_float().or_else(|| a.as_int().map(|i| i as f64));
    let y = b.as_float().or_else(|| b.as_int().map(|i| i as f64));
    match (x, y) {
        (Some(x), Some(y)) => {
            frame.set_reg(inst.dst().0, Value::bool(x <= y));
            ControlFlow::Continue
        }
        _ => ControlFlow::Error(RuntimeError::unsupported_operand(
            "<=", "unknown", "unknown",
        )),
    }
}

/// Gt: dst = src1 > src2 (generic with speculative fast-path)
#[inline(always)]
pub fn gt(vm: &mut VirtualMachine, inst: Instruction) -> ControlFlow {
    use crate::ic_manager::ICSiteId;
    use crate::speculative::{SpecResult, Speculation, spec_gt_float, spec_gt_int};
    use crate::type_feedback::{BinaryOpFeedback, OperandPair};

    let frame = vm.current_frame_mut();
    let (a, b) = frame.get_regs2(inst.src1().0, inst.src2().0);
    let code_id = frame.code_id();
    let bc_offset = frame.ip.saturating_sub(1) as u32;
    let site = ICSiteId::new(code_id, bc_offset);

    // Speculative Fast Path
    if let Some(spec) = vm.speculation_cache.get(site) {
        match spec {
            Speculation::IntInt => {
                let (result, value) = spec_gt_int(a, b);
                if result == SpecResult::Success {
                    let frame = vm.current_frame_mut();
                    frame.set_reg(inst.dst().0, value);
                    return ControlFlow::Continue;
                }
                vm.speculation_cache.invalidate(site);
            }
            Speculation::FloatFloat | Speculation::IntFloat | Speculation::FloatInt => {
                let (result, value) = spec_gt_float(a, b);
                if result == SpecResult::Success {
                    let frame = vm.current_frame_mut();
                    frame.set_reg(inst.dst().0, value);
                    return ControlFlow::Continue;
                }
                vm.speculation_cache.invalidate(site);
            }
            Speculation::None
            | Speculation::StrStr
            | Speculation::StrInt
            | Speculation::IntStr
            | Speculation::ListList => {}
        }
    }

    // Slow Path
    let pair = OperandPair::from_values(a, b);
    let feedback = BinaryOpFeedback::new(code_id, bc_offset, a, b);
    feedback.record(&mut vm.ic_manager);
    let spec = Speculation::from_operand_pair(pair);
    if spec != Speculation::None {
        vm.speculation_cache.insert(site, spec);
    }

    let frame = vm.current_frame_mut();
    if let (Some(x), Some(y)) = (a.as_int(), b.as_int()) {
        frame.set_reg(inst.dst().0, Value::bool(x > y));
        return ControlFlow::Continue;
    }

    let x = a.as_float().or_else(|| a.as_int().map(|i| i as f64));
    let y = b.as_float().or_else(|| b.as_int().map(|i| i as f64));
    match (x, y) {
        (Some(x), Some(y)) => {
            frame.set_reg(inst.dst().0, Value::bool(x > y));
            ControlFlow::Continue
        }
        _ => ControlFlow::Error(RuntimeError::unsupported_operand(">", "unknown", "unknown")),
    }
}

/// Ge: dst = src1 >= src2 (generic with speculative fast-path)
#[inline(always)]
pub fn ge(vm: &mut VirtualMachine, inst: Instruction) -> ControlFlow {
    use crate::ic_manager::ICSiteId;
    use crate::speculative::{SpecResult, Speculation, spec_ge_float, spec_ge_int};
    use crate::type_feedback::{BinaryOpFeedback, OperandPair};

    let frame = vm.current_frame_mut();
    let (a, b) = frame.get_regs2(inst.src1().0, inst.src2().0);
    let code_id = frame.code_id();
    let bc_offset = frame.ip.saturating_sub(1) as u32;
    let site = ICSiteId::new(code_id, bc_offset);

    // Speculative Fast Path
    if let Some(spec) = vm.speculation_cache.get(site) {
        match spec {
            Speculation::IntInt => {
                let (result, value) = spec_ge_int(a, b);
                if result == SpecResult::Success {
                    let frame = vm.current_frame_mut();
                    frame.set_reg(inst.dst().0, value);
                    return ControlFlow::Continue;
                }
                vm.speculation_cache.invalidate(site);
            }
            Speculation::FloatFloat | Speculation::IntFloat | Speculation::FloatInt => {
                let (result, value) = spec_ge_float(a, b);
                if result == SpecResult::Success {
                    let frame = vm.current_frame_mut();
                    frame.set_reg(inst.dst().0, value);
                    return ControlFlow::Continue;
                }
                vm.speculation_cache.invalidate(site);
            }
            Speculation::None
            | Speculation::StrStr
            | Speculation::StrInt
            | Speculation::IntStr
            | Speculation::ListList => {}
        }
    }

    // Slow Path
    let pair = OperandPair::from_values(a, b);
    let feedback = BinaryOpFeedback::new(code_id, bc_offset, a, b);
    feedback.record(&mut vm.ic_manager);
    let spec = Speculation::from_operand_pair(pair);
    if spec != Speculation::None {
        vm.speculation_cache.insert(site, spec);
    }

    let frame = vm.current_frame_mut();
    if let (Some(x), Some(y)) = (a.as_int(), b.as_int()) {
        frame.set_reg(inst.dst().0, Value::bool(x >= y));
        return ControlFlow::Continue;
    }

    let x = a.as_float().or_else(|| a.as_int().map(|i| i as f64));
    let y = b.as_float().or_else(|| b.as_int().map(|i| i as f64));
    match (x, y) {
        (Some(x), Some(y)) => {
            frame.set_reg(inst.dst().0, Value::bool(x >= y));
            ControlFlow::Continue
        }
        _ => ControlFlow::Error(RuntimeError::unsupported_operand(
            ">=", "unknown", "unknown",
        )),
    }
}

// =============================================================================
// Equality
// =============================================================================

/// Eq: dst = src1 == src2 (generic with speculative fast-path)
#[inline(always)]
pub fn eq(vm: &mut VirtualMachine, inst: Instruction) -> ControlFlow {
    use crate::ic_manager::ICSiteId;
    use crate::speculative::{SpecResult, Speculation, spec_eq_float, spec_eq_int};
    use crate::type_feedback::{BinaryOpFeedback, OperandPair};

    let frame = vm.current_frame_mut();
    let (a, b) = frame.get_regs2(inst.src1().0, inst.src2().0);
    let code_id = frame.code_id();
    let bc_offset = frame.ip.saturating_sub(1) as u32;
    let site = ICSiteId::new(code_id, bc_offset);

    // Speculative Fast Path
    if let Some(spec) = vm.speculation_cache.get(site) {
        match spec {
            Speculation::IntInt => {
                let (result, value) = spec_eq_int(a, b);
                if result == SpecResult::Success {
                    let frame = vm.current_frame_mut();
                    frame.set_reg(inst.dst().0, value);
                    return ControlFlow::Continue;
                }
                vm.speculation_cache.invalidate(site);
            }
            Speculation::FloatFloat | Speculation::IntFloat | Speculation::FloatInt => {
                let (result, value) = spec_eq_float(a, b);
                if result == SpecResult::Success {
                    let frame = vm.current_frame_mut();
                    frame.set_reg(inst.dst().0, value);
                    return ControlFlow::Continue;
                }
                vm.speculation_cache.invalidate(site);
            }
            Speculation::None
            | Speculation::StrStr
            | Speculation::StrInt
            | Speculation::IntStr
            | Speculation::ListList => {}
        }
    }

    // Slow Path
    let pair = OperandPair::from_values(a, b);
    let feedback = BinaryOpFeedback::new(code_id, bc_offset, a, b);
    feedback.record(&mut vm.ic_manager);
    let spec = Speculation::from_operand_pair(pair);
    if spec != Speculation::None {
        vm.speculation_cache.insert(site, spec);
    }

    let frame = vm.current_frame_mut();

    // Check identical None/bool values (fast path for special types)
    if a.is_none() && b.is_none() {
        frame.set_reg(inst.dst().0, Value::bool(true));
        return ControlFlow::Continue;
    }
    if a.is_bool() && b.is_bool() {
        frame.set_reg(inst.dst().0, Value::bool(a.as_bool() == b.as_bool()));
        return ControlFlow::Continue;
    }

    // Int comparison
    if let (Some(x), Some(y)) = (a.as_int(), b.as_int()) {
        frame.set_reg(inst.dst().0, Value::bool(x == y));
        return ControlFlow::Continue;
    }

    // Float comparison
    let x = a.as_float().or_else(|| a.as_int().map(|i| i as f64));
    let y = b.as_float().or_else(|| b.as_int().map(|i| i as f64));
    match (x, y) {
        (Some(x), Some(y)) => {
            frame.set_reg(inst.dst().0, Value::bool(x == y));
            ControlFlow::Continue
        }
        _ => {
            // Different types that aren't comparable = not equal
            frame.set_reg(inst.dst().0, Value::bool(false));
            ControlFlow::Continue
        }
    }
}

/// Ne: dst = src1 != src2 (generic with speculative fast-path)
#[inline(always)]
pub fn ne(vm: &mut VirtualMachine, inst: Instruction) -> ControlFlow {
    use crate::ic_manager::ICSiteId;
    use crate::speculative::{SpecResult, Speculation, spec_ne_float, spec_ne_int};
    use crate::type_feedback::{BinaryOpFeedback, OperandPair};

    let frame = vm.current_frame_mut();
    let (a, b) = frame.get_regs2(inst.src1().0, inst.src2().0);
    let code_id = frame.code_id();
    let bc_offset = frame.ip.saturating_sub(1) as u32;
    let site = ICSiteId::new(code_id, bc_offset);

    // Speculative Fast Path
    if let Some(spec) = vm.speculation_cache.get(site) {
        match spec {
            Speculation::IntInt => {
                let (result, value) = spec_ne_int(a, b);
                if result == SpecResult::Success {
                    let frame = vm.current_frame_mut();
                    frame.set_reg(inst.dst().0, value);
                    return ControlFlow::Continue;
                }
                vm.speculation_cache.invalidate(site);
            }
            Speculation::FloatFloat | Speculation::IntFloat | Speculation::FloatInt => {
                let (result, value) = spec_ne_float(a, b);
                if result == SpecResult::Success {
                    let frame = vm.current_frame_mut();
                    frame.set_reg(inst.dst().0, value);
                    return ControlFlow::Continue;
                }
                vm.speculation_cache.invalidate(site);
            }
            Speculation::None
            | Speculation::StrStr
            | Speculation::StrInt
            | Speculation::IntStr
            | Speculation::ListList => {}
        }
    }

    // Slow Path
    let pair = OperandPair::from_values(a, b);
    let feedback = BinaryOpFeedback::new(code_id, bc_offset, a, b);
    feedback.record(&mut vm.ic_manager);
    let spec = Speculation::from_operand_pair(pair);
    if spec != Speculation::None {
        vm.speculation_cache.insert(site, spec);
    }

    let frame = vm.current_frame_mut();

    // Check identical None values
    if a.is_none() && b.is_none() {
        frame.set_reg(inst.dst().0, Value::bool(false));
        return ControlFlow::Continue;
    }
    if a.is_bool() && b.is_bool() {
        frame.set_reg(inst.dst().0, Value::bool(a.as_bool() != b.as_bool()));
        return ControlFlow::Continue;
    }

    if let (Some(x), Some(y)) = (a.as_int(), b.as_int()) {
        frame.set_reg(inst.dst().0, Value::bool(x != y));
        return ControlFlow::Continue;
    }

    let x = a.as_float().or_else(|| a.as_int().map(|i| i as f64));
    let y = b.as_float().or_else(|| b.as_int().map(|i| i as f64));
    match (x, y) {
        (Some(x), Some(y)) => {
            frame.set_reg(inst.dst().0, Value::bool(x != y));
            ControlFlow::Continue
        }
        _ => {
            // Different types = not equal
            frame.set_reg(inst.dst().0, Value::bool(true));
            ControlFlow::Continue
        }
    }
}

// =============================================================================
// Identity
// =============================================================================

/// Is: dst = src1 is src2
#[inline(always)]
pub fn is(vm: &mut VirtualMachine, inst: Instruction) -> ControlFlow {
    let frame = vm.current_frame_mut();
    let (a, b) = frame.get_regs2(inst.src1().0, inst.src2().0);

    // Identity check: comparing types and values
    let same = if a.is_none() && b.is_none() {
        true
    } else if a.is_bool() && b.is_bool() {
        a.as_bool() == b.as_bool()
    } else if let (Some(x), Some(y)) = (a.as_int(), b.as_int()) {
        x == y
    } else if a.is_object() && b.is_object() {
        a.as_object_ptr() == b.as_object_ptr()
    } else {
        false
    };
    frame.set_reg(inst.dst().0, Value::bool(same));
    ControlFlow::Continue
}

/// IsNot: dst = src1 is not src2
#[inline(always)]
pub fn is_not(vm: &mut VirtualMachine, inst: Instruction) -> ControlFlow {
    let frame = vm.current_frame_mut();
    let (a, b) = frame.get_regs2(inst.src1().0, inst.src2().0);

    // Identity check: comparing types and values
    let same = if a.is_none() && b.is_none() {
        true
    } else if a.is_bool() && b.is_bool() {
        a.as_bool() == b.as_bool()
    } else if let (Some(x), Some(y)) = (a.as_int(), b.as_int()) {
        x == y
    } else if a.is_object() && b.is_object() {
        a.as_object_ptr() == b.as_object_ptr()
    } else {
        false
    };
    frame.set_reg(inst.dst().0, Value::bool(!same));
    ControlFlow::Continue
}

// =============================================================================
// Membership Testing
// =============================================================================

/// In: dst = src1 in src2
///
/// Tests if src1 is contained in src2. Dispatches based on container type:
/// - List/Tuple: O(n) linear search with value equality
/// - Set: O(1) hash-based lookup
/// - Dict: O(1) hash-based key lookup
/// - String: SIMD-accelerated substring search
/// - Range: O(1) arithmetic bounds check
///
/// # Performance Characteristics
///
/// - Set/Dict: O(1) average case via hash tables
/// - List/Tuple: O(n) with speculative fast paths for common types
/// - String: O(n*m) worst case, O(n) typical with SIMD
/// - Range: O(1) arithmetic (no iteration)
#[inline(always)]
pub fn in_op(vm: &mut VirtualMachine, inst: Instruction) -> ControlFlow {
    let frame = vm.current_frame();
    let needle = frame.get_reg(inst.src1().0);
    let container = frame.get_reg(inst.src2().0);

    match contains_value(needle, container) {
        Ok(result) => {
            let frame = vm.current_frame_mut();
            frame.set_reg(inst.dst().0, Value::bool(result));
            ControlFlow::Continue
        }
        Err(cf) => cf,
    }
}

/// NotIn: dst = src1 not in src2
///
/// Logical negation of the `in` operator.
#[inline(always)]
pub fn not_in(vm: &mut VirtualMachine, inst: Instruction) -> ControlFlow {
    let frame = vm.current_frame();
    let needle = frame.get_reg(inst.src1().0);
    let container = frame.get_reg(inst.src2().0);

    match contains_value(needle, container) {
        Ok(result) => {
            let frame = vm.current_frame_mut();
            frame.set_reg(inst.dst().0, Value::bool(!result));
            ControlFlow::Continue
        }
        Err(cf) => cf,
    }
}

/// Core containment check with type dispatch.
///
/// Returns Ok(bool) for valid containment checks,
/// Err(ControlFlow) for type errors.
///
/// # Type Dispatch
///
/// - List/Tuple: O(n) linear scan with value equality
/// - Set: O(1) hash-based lookup
/// - Dict: O(1) hash-based key lookup
/// - String: SIMD-accelerated substring search (~8-32 GB/s)
/// - Range: O(1) arithmetic containment check
///
/// # Performance
///
/// String containment uses SSE4.2 PCMPESTRI for needles ≤16 bytes,
/// AVX2 dual-byte filter for longer needles.
#[inline]
fn contains_value(needle: Value, container: Value) -> Result<bool, ControlFlow> {
    use prism_runtime::object::type_obj::TypeId;
    use prism_runtime::types::range::RangeObject;
    use prism_runtime::types::string::StringObject;
    use prism_runtime::types::tuple::TupleObject;
    use prism_runtime::types::{DictObject, ListObject, SetObject};

    // Fast path: check object pointer types
    if let Some(ptr) = container.as_object_ptr() {
        // Read the TypeId from the object header for safe dispatch
        // All Python objects start with ObjectHeader which contains type_id
        let header_ptr = ptr as *const prism_runtime::object::ObjectHeader;
        let type_id = unsafe { (*header_ptr).type_id };

        match type_id {
            // List: O(n) linear scan
            TypeId::LIST => {
                let list = unsafe { &*(ptr as *const ListObject) };
                for i in 0..list.len() {
                    if let Some(val) = list.get(i as i64) {
                        if values_equal(needle, val) {
                            return Ok(true);
                        }
                    }
                }
                return Ok(false);
            }

            // Tuple: O(n) linear scan
            TypeId::TUPLE => {
                let tuple = unsafe { &*(ptr as *const TupleObject) };
                for i in 0..tuple.len() {
                    if let Some(val) = tuple.get(i as i64) {
                        if values_equal(needle, val) {
                            return Ok(true);
                        }
                    }
                }
                return Ok(false);
            }

            // Set: O(1) hash-based lookup
            TypeId::SET => {
                let set = unsafe { &*(ptr as *const SetObject) };
                return Ok(set.contains(needle));
            }

            // Dict: O(1) hash-based key lookup
            TypeId::DICT => {
                let dict = unsafe { &*(ptr as *const DictObject) };
                return Ok(dict.get(needle).is_some());
            }

            // String: SIMD-accelerated substring search
            TypeId::STR => {
                let haystack = unsafe { &*(ptr as *const StringObject) };

                // Needle must also be a string for substring search
                if let Some(needle_ptr) = needle.as_object_ptr() {
                    let needle_header = needle_ptr as *const prism_runtime::object::ObjectHeader;
                    let needle_type = unsafe { (*needle_header).type_id };

                    if needle_type == TypeId::STR {
                        let needle_str = unsafe { &*(needle_ptr as *const StringObject) };
                        // Use SIMD-accelerated search (~8-32 GB/s)
                        return Ok(haystack.contains(needle_str.as_str()));
                    }
                }

                // Non-string needle in string container is always false
                return Err(ControlFlow::Error(RuntimeError::type_error(
                    "'in <string>' requires string as left operand",
                )));
            }

            // Range: O(1) arithmetic containment
            TypeId::RANGE => {
                let range = unsafe { &*(ptr as *const RangeObject) };

                // Only integers can be in a range
                if let Some(value) = needle.as_int() {
                    return Ok(range.contains(value));
                }

                // Float values check: Python allows 5.0 in range(10)
                if let Some(f) = needle.as_float() {
                    // Check if it's a whole number
                    if f.fract() == 0.0 && f >= i64::MIN as f64 && f <= i64::MAX as f64 {
                        return Ok(range.contains(f as i64));
                    }
                }

                // Non-numeric types are not in range
                return Ok(false);
            }

            // Other types: fall through to protocol lookup
            _ => {}
        }

        // TODO: Implement __contains__ protocol for custom objects
        // For now, return type error for unsupported containers
        return Err(ControlFlow::Error(RuntimeError::type_error(format!(
            "argument of type '{}' is not iterable",
            type_id.name()
        ))));
    }

    // Inline types: integers, floats, bools cannot be containers
    Err(ControlFlow::Error(RuntimeError::type_error(
        "argument of type is not iterable",
    )))
}

/// Value equality comparison with cross-type numerical equality.
///
/// Implements Python semantics:
/// - int == float if same value (1 == 1.0)
/// - bool == int (True == 1, False == 0)
/// - Object equality by reference (identity for objects)
#[inline]
fn values_equal(a: Value, b: Value) -> bool {
    // None equality
    if a.is_none() && b.is_none() {
        return true;
    }

    // None is not equal to anything else
    if a.is_none() || b.is_none() {
        return false;
    }

    // Bool equality
    if a.is_bool() && b.is_bool() {
        return a.as_bool() == b.as_bool();
    }

    // Integer equality
    if let (Some(x), Some(y)) = (a.as_int(), b.as_int()) {
        return x == y;
    }

    // Float equality
    if let (Some(x), Some(y)) = (a.as_float(), b.as_float()) {
        return x == y;
    }

    // Cross-type: int-float comparison (Python semantics: 1 == 1.0)
    if let (Some(i), Some(f)) = (a.as_int(), b.as_float()) {
        return (i as f64) == f;
    }
    if let (Some(f), Some(i)) = (a.as_float(), b.as_int()) {
        return f == (i as f64);
    }

    // Cross-type: bool-int comparison (Python semantics: True == 1)
    if a.is_bool() {
        if let Some(i) = b.as_int() {
            return (a.is_truthy() as i64) == i;
        }
    }
    if b.is_bool() {
        if let Some(i) = a.as_int() {
            return i == (b.is_truthy() as i64);
        }
    }

    // Object pointer equality (by identity)
    if let (Some(pa), Some(pb)) = (a.as_object_ptr(), b.as_object_ptr()) {
        return pa == pb;
    }

    false
}

// =============================================================================
// Logical/Bitwise
// =============================================================================

/// Not: dst = not src1
#[inline(always)]
pub fn not(vm: &mut VirtualMachine, inst: Instruction) -> ControlFlow {
    let frame = vm.current_frame_mut();
    let a = frame.get_reg(inst.src1().0);

    frame.set_reg(inst.dst().0, Value::bool(!a.is_truthy()));
    ControlFlow::Continue
}

/// BitwiseAnd: dst = src1 & src2
#[inline(always)]
pub fn bitwise_and(vm: &mut VirtualMachine, inst: Instruction) -> ControlFlow {
    let frame = vm.current_frame_mut();
    let (a, b) = frame.get_regs2(inst.src1().0, inst.src2().0);

    match (a.as_int(), b.as_int()) {
        (Some(x), Some(y)) => match Value::int(x & y) {
            Some(v) => {
                frame.set_reg(inst.dst().0, v);
                ControlFlow::Continue
            }
            None => ControlFlow::Error(RuntimeError::value_error("Integer too large")),
        },
        // Bool & Bool
        (None, None) if a.is_bool() && b.is_bool() => {
            let x = a.is_truthy();
            let y = b.is_truthy();
            frame.set_reg(inst.dst().0, Value::bool(x && y));
            ControlFlow::Continue
        }
        _ => ControlFlow::Error(RuntimeError::unsupported_operand("&", "unknown", "unknown")),
    }
}

/// BitwiseOr: dst = src1 | src2
#[inline(always)]
pub fn bitwise_or(vm: &mut VirtualMachine, inst: Instruction) -> ControlFlow {
    let frame = vm.current_frame_mut();
    let (a, b) = frame.get_regs2(inst.src1().0, inst.src2().0);

    match (a.as_int(), b.as_int()) {
        (Some(x), Some(y)) => match Value::int(x | y) {
            Some(v) => {
                frame.set_reg(inst.dst().0, v);
                ControlFlow::Continue
            }
            None => ControlFlow::Error(RuntimeError::value_error("Integer too large")),
        },
        (None, None) if a.is_bool() && b.is_bool() => {
            let x = a.is_truthy();
            let y = b.is_truthy();
            frame.set_reg(inst.dst().0, Value::bool(x || y));
            ControlFlow::Continue
        }
        _ => ControlFlow::Error(RuntimeError::unsupported_operand("|", "unknown", "unknown")),
    }
}

/// BitwiseXor: dst = src1 ^ src2
#[inline(always)]
pub fn bitwise_xor(vm: &mut VirtualMachine, inst: Instruction) -> ControlFlow {
    let frame = vm.current_frame_mut();
    let (a, b) = frame.get_regs2(inst.src1().0, inst.src2().0);

    match (a.as_int(), b.as_int()) {
        (Some(x), Some(y)) => match Value::int(x ^ y) {
            Some(v) => {
                frame.set_reg(inst.dst().0, v);
                ControlFlow::Continue
            }
            None => ControlFlow::Error(RuntimeError::value_error("Integer too large")),
        },
        (None, None) if a.is_bool() && b.is_bool() => {
            let x = a.is_truthy();
            let y = b.is_truthy();
            frame.set_reg(inst.dst().0, Value::bool(x != y));
            ControlFlow::Continue
        }
        _ => ControlFlow::Error(RuntimeError::unsupported_operand("^", "unknown", "unknown")),
    }
}

/// BitwiseNot: dst = ~src1
#[inline(always)]
pub fn bitwise_not(vm: &mut VirtualMachine, inst: Instruction) -> ControlFlow {
    let frame = vm.current_frame_mut();
    let a = frame.get_reg(inst.src1().0);

    match a.as_int() {
        Some(x) => match Value::int(!x) {
            Some(v) => {
                frame.set_reg(inst.dst().0, v);
                ControlFlow::Continue
            }
            None => ControlFlow::Error(RuntimeError::value_error("Integer too large")),
        },
        None => ControlFlow::Error(RuntimeError::type_error("bad operand type for unary ~")),
    }
}

/// Shl: dst = src1 << src2
#[inline(always)]
pub fn shl(vm: &mut VirtualMachine, inst: Instruction) -> ControlFlow {
    let frame = vm.current_frame_mut();
    let (a, b) = frame.get_regs2(inst.src1().0, inst.src2().0);

    match (a.as_int(), b.as_int()) {
        (Some(x), Some(y)) if y >= 0 && y < 64 => match Value::int(x << y) {
            Some(v) => {
                frame.set_reg(inst.dst().0, v);
                ControlFlow::Continue
            }
            None => ControlFlow::Error(RuntimeError::value_error("Integer too large")),
        },
        (Some(_), Some(y)) if y < 0 => {
            ControlFlow::Error(RuntimeError::value_error("negative shift count"))
        }
        _ => ControlFlow::Error(RuntimeError::unsupported_operand(
            "<<", "unknown", "unknown",
        )),
    }
}

/// Shr: dst = src1 >> src2
#[inline(always)]
pub fn shr(vm: &mut VirtualMachine, inst: Instruction) -> ControlFlow {
    let frame = vm.current_frame_mut();
    let (a, b) = frame.get_regs2(inst.src1().0, inst.src2().0);

    match (a.as_int(), b.as_int()) {
        (Some(x), Some(y)) if y >= 0 && y < 64 => match Value::int(x >> y) {
            Some(v) => {
                frame.set_reg(inst.dst().0, v);
                ControlFlow::Continue
            }
            None => ControlFlow::Error(RuntimeError::value_error("Integer too large")),
        },
        (Some(_), Some(y)) if y < 0 => {
            ControlFlow::Error(RuntimeError::value_error("negative shift count"))
        }
        _ => ControlFlow::Error(RuntimeError::unsupported_operand(
            ">>", "unknown", "unknown",
        )),
    }
}

#[cfg(test)]
mod tests {
    // Comparison tests require full VM setup
}
