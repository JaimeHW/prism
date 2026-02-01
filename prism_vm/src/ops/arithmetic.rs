//! Arithmetic opcode handlers.
//!
//! Provides type-specialized fast paths for int/float operations,
//! with fallback to generic polymorphic operations.
//!
//! # Type Feedback Integration
//!
//! Generic handlers collect type feedback via `BinaryOpFeedback` for JIT
//! specialization. After sufficient observations, the JIT can emit
//! specialized code paths.

use crate::VirtualMachine;
use crate::dispatch::ControlFlow;
use crate::error::RuntimeError;
use crate::type_feedback::BinaryOpFeedback;
use prism_compiler::bytecode::Instruction;
use prism_core::Value;

// =============================================================================
// Integer Arithmetic (Fast Path)
// =============================================================================

/// AddInt: dst = src1 + src2 (integers only)
#[inline(always)]
pub fn add_int(vm: &mut VirtualMachine, inst: Instruction) -> ControlFlow {
    let frame = vm.current_frame_mut();
    let (a, b) = frame.get_regs2(inst.src1().0, inst.src2().0);

    // Fast path: both are integers
    match (a.as_int(), b.as_int()) {
        (Some(x), Some(y)) => match x.checked_add(y) {
            Some(result) => match Value::int(result) {
                Some(v) => {
                    frame.set_reg(inst.dst().0, v);
                    ControlFlow::Continue
                }
                None => ControlFlow::Error(RuntimeError::value_error("Integer too large for i48")),
            },
            None => ControlFlow::Error(RuntimeError::value_error("Integer overflow")),
        },
        _ => ControlFlow::Error(RuntimeError::type_error("AddInt requires integers")),
    }
}

/// SubInt: dst = src1 - src2
#[inline(always)]
pub fn sub_int(vm: &mut VirtualMachine, inst: Instruction) -> ControlFlow {
    let frame = vm.current_frame_mut();
    let (a, b) = frame.get_regs2(inst.src1().0, inst.src2().0);

    match (a.as_int(), b.as_int()) {
        (Some(x), Some(y)) => match x.checked_sub(y) {
            Some(result) => match Value::int(result) {
                Some(v) => {
                    frame.set_reg(inst.dst().0, v);
                    ControlFlow::Continue
                }
                None => ControlFlow::Error(RuntimeError::value_error("Integer too large for i48")),
            },
            None => ControlFlow::Error(RuntimeError::value_error("Integer overflow")),
        },
        _ => ControlFlow::Error(RuntimeError::type_error("SubInt requires integers")),
    }
}

/// MulInt: dst = src1 * src2
#[inline(always)]
pub fn mul_int(vm: &mut VirtualMachine, inst: Instruction) -> ControlFlow {
    let frame = vm.current_frame_mut();
    let (a, b) = frame.get_regs2(inst.src1().0, inst.src2().0);

    match (a.as_int(), b.as_int()) {
        (Some(x), Some(y)) => match x.checked_mul(y) {
            Some(result) => match Value::int(result) {
                Some(v) => {
                    frame.set_reg(inst.dst().0, v);
                    ControlFlow::Continue
                }
                None => ControlFlow::Error(RuntimeError::value_error("Integer too large for i48")),
            },
            None => ControlFlow::Error(RuntimeError::value_error("Integer overflow")),
        },
        _ => ControlFlow::Error(RuntimeError::type_error("MulInt requires integers")),
    }
}

/// FloorDivInt: dst = src1 // src2
#[inline(always)]
pub fn floor_div_int(vm: &mut VirtualMachine, inst: Instruction) -> ControlFlow {
    let frame = vm.current_frame_mut();
    let (a, b) = frame.get_regs2(inst.src1().0, inst.src2().0);

    match (a.as_int(), b.as_int()) {
        (Some(_), Some(0)) => ControlFlow::Error(RuntimeError::zero_division()),
        (Some(x), Some(y)) => {
            // Python-style floor division
            let result = x.div_euclid(y);
            match Value::int(result) {
                Some(v) => {
                    frame.set_reg(inst.dst().0, v);
                    ControlFlow::Continue
                }
                None => ControlFlow::Error(RuntimeError::value_error("Integer too large for i48")),
            }
        }
        _ => ControlFlow::Error(RuntimeError::type_error("FloorDivInt requires integers")),
    }
}

/// ModInt: dst = src1 % src2
#[inline(always)]
pub fn mod_int(vm: &mut VirtualMachine, inst: Instruction) -> ControlFlow {
    let frame = vm.current_frame_mut();
    let (a, b) = frame.get_regs2(inst.src1().0, inst.src2().0);

    match (a.as_int(), b.as_int()) {
        (Some(_), Some(0)) => ControlFlow::Error(RuntimeError::zero_division()),
        (Some(x), Some(y)) => {
            // Python-style modulo
            let result = x.rem_euclid(y);
            match Value::int(result) {
                Some(v) => {
                    frame.set_reg(inst.dst().0, v);
                    ControlFlow::Continue
                }
                None => ControlFlow::Error(RuntimeError::value_error("Integer too large for i48")),
            }
        }
        _ => ControlFlow::Error(RuntimeError::type_error("ModInt requires integers")),
    }
}

/// PowInt: dst = src1 ** src2
#[inline(always)]
pub fn pow_int(vm: &mut VirtualMachine, inst: Instruction) -> ControlFlow {
    let frame = vm.current_frame_mut();
    let (a, b) = frame.get_regs2(inst.src1().0, inst.src2().0);

    match (a.as_int(), b.as_int()) {
        (Some(base), Some(exp)) => {
            if exp < 0 {
                // Negative exponent produces float
                let result = (base as f64).powi(exp as i32);
                frame.set_reg(inst.dst().0, Value::float(result));
                ControlFlow::Continue
            } else {
                match (base as i128).checked_pow(exp as u32) {
                    Some(result) if result >= i64::MIN as i128 && result <= i64::MAX as i128 => {
                        match Value::int(result as i64) {
                            Some(v) => {
                                frame.set_reg(inst.dst().0, v);
                                ControlFlow::Continue
                            }
                            None => ControlFlow::Error(RuntimeError::value_error(
                                "Integer too large for i48",
                            )),
                        }
                    }
                    _ => ControlFlow::Error(RuntimeError::value_error("Integer overflow")),
                }
            }
        }
        _ => ControlFlow::Error(RuntimeError::type_error("PowInt requires integers")),
    }
}

/// NegInt: dst = -src1
#[inline(always)]
pub fn neg_int(vm: &mut VirtualMachine, inst: Instruction) -> ControlFlow {
    let frame = vm.current_frame_mut();
    let a = frame.get_reg(inst.src1().0);

    match a.as_int() {
        Some(x) => match x.checked_neg() {
            Some(result) => match Value::int(result) {
                Some(v) => {
                    frame.set_reg(inst.dst().0, v);
                    ControlFlow::Continue
                }
                None => ControlFlow::Error(RuntimeError::value_error("Integer too large for i48")),
            },
            None => ControlFlow::Error(RuntimeError::value_error("Integer overflow")),
        },
        None => ControlFlow::Error(RuntimeError::type_error("NegInt requires integers")),
    }
}

/// PosInt: dst = +src1 (identity)
#[inline(always)]
pub fn pos_int(vm: &mut VirtualMachine, inst: Instruction) -> ControlFlow {
    let frame = vm.current_frame_mut();
    let value = frame.get_reg(inst.src1().0);
    frame.set_reg(inst.dst().0, value);
    ControlFlow::Continue
}

// =============================================================================
// Float Arithmetic (Fast Path)
// =============================================================================

/// AddFloat: dst = src1 + src2 (floats)
#[inline(always)]
pub fn add_float(vm: &mut VirtualMachine, inst: Instruction) -> ControlFlow {
    let frame = vm.current_frame_mut();
    let (a, b) = frame.get_regs2(inst.src1().0, inst.src2().0);

    match (a.as_float(), b.as_float()) {
        (Some(x), Some(y)) => {
            frame.set_reg(inst.dst().0, Value::float(x + y));
            ControlFlow::Continue
        }
        _ => ControlFlow::Error(RuntimeError::type_error("AddFloat requires floats")),
    }
}

/// SubFloat: dst = src1 - src2
#[inline(always)]
pub fn sub_float(vm: &mut VirtualMachine, inst: Instruction) -> ControlFlow {
    let frame = vm.current_frame_mut();
    let (a, b) = frame.get_regs2(inst.src1().0, inst.src2().0);

    match (a.as_float(), b.as_float()) {
        (Some(x), Some(y)) => {
            frame.set_reg(inst.dst().0, Value::float(x - y));
            ControlFlow::Continue
        }
        _ => ControlFlow::Error(RuntimeError::type_error("SubFloat requires floats")),
    }
}

/// MulFloat: dst = src1 * src2
#[inline(always)]
pub fn mul_float(vm: &mut VirtualMachine, inst: Instruction) -> ControlFlow {
    let frame = vm.current_frame_mut();
    let (a, b) = frame.get_regs2(inst.src1().0, inst.src2().0);

    match (a.as_float(), b.as_float()) {
        (Some(x), Some(y)) => {
            frame.set_reg(inst.dst().0, Value::float(x * y));
            ControlFlow::Continue
        }
        _ => ControlFlow::Error(RuntimeError::type_error("MulFloat requires floats")),
    }
}

/// DivFloat: dst = src1 / src2
#[inline(always)]
pub fn div_float(vm: &mut VirtualMachine, inst: Instruction) -> ControlFlow {
    let frame = vm.current_frame_mut();
    let (a, b) = frame.get_regs2(inst.src1().0, inst.src2().0);

    match (a.as_float(), b.as_float()) {
        (Some(_), Some(y)) if y == 0.0 => ControlFlow::Error(RuntimeError::zero_division()),
        (Some(x), Some(y)) => {
            frame.set_reg(inst.dst().0, Value::float(x / y));
            ControlFlow::Continue
        }
        _ => ControlFlow::Error(RuntimeError::type_error("DivFloat requires floats")),
    }
}

/// FloorDivFloat: dst = src1 // src2
#[inline(always)]
pub fn floor_div_float(vm: &mut VirtualMachine, inst: Instruction) -> ControlFlow {
    let frame = vm.current_frame_mut();
    let (a, b) = frame.get_regs2(inst.src1().0, inst.src2().0);

    match (a.as_float(), b.as_float()) {
        (Some(_), Some(y)) if y == 0.0 => ControlFlow::Error(RuntimeError::zero_division()),
        (Some(x), Some(y)) => {
            frame.set_reg(inst.dst().0, Value::float((x / y).floor()));
            ControlFlow::Continue
        }
        _ => ControlFlow::Error(RuntimeError::type_error("FloorDivFloat requires floats")),
    }
}

/// ModFloat: dst = src1 % src2
#[inline(always)]
pub fn mod_float(vm: &mut VirtualMachine, inst: Instruction) -> ControlFlow {
    let frame = vm.current_frame_mut();
    let (a, b) = frame.get_regs2(inst.src1().0, inst.src2().0);

    match (a.as_float(), b.as_float()) {
        (Some(_), Some(y)) if y == 0.0 => ControlFlow::Error(RuntimeError::zero_division()),
        (Some(x), Some(y)) => {
            // Python-style modulo
            let result = x - y * (x / y).floor();
            frame.set_reg(inst.dst().0, Value::float(result));
            ControlFlow::Continue
        }
        _ => ControlFlow::Error(RuntimeError::type_error("ModFloat requires floats")),
    }
}

/// PowFloat: dst = src1 ** src2
#[inline(always)]
pub fn pow_float(vm: &mut VirtualMachine, inst: Instruction) -> ControlFlow {
    let frame = vm.current_frame_mut();
    let (a, b) = frame.get_regs2(inst.src1().0, inst.src2().0);

    match (a.as_float(), b.as_float()) {
        (Some(x), Some(y)) => {
            frame.set_reg(inst.dst().0, Value::float(x.powf(y)));
            ControlFlow::Continue
        }
        _ => ControlFlow::Error(RuntimeError::type_error("PowFloat requires floats")),
    }
}

/// NegFloat: dst = -src1
#[inline(always)]
pub fn neg_float(vm: &mut VirtualMachine, inst: Instruction) -> ControlFlow {
    let frame = vm.current_frame_mut();
    let a = frame.get_reg(inst.src1().0);

    match a.as_float() {
        Some(x) => {
            frame.set_reg(inst.dst().0, Value::float(-x));
            ControlFlow::Continue
        }
        None => ControlFlow::Error(RuntimeError::type_error("NegFloat requires floats")),
    }
}

// =============================================================================
// Generic Arithmetic (Polymorphic - Slower)
// =============================================================================

/// Add: dst = src1 + src2 (generic with speculative fast-path)
///
/// Uses O(1) speculation cache lookup to select optimized code path.
/// Records type feedback on slow path for future specialization.
#[inline(always)]
pub fn add(vm: &mut VirtualMachine, inst: Instruction) -> ControlFlow {
    use crate::ic_manager::ICSiteId;
    use crate::speculative::{
        SpecResult, Speculation, spec_add_float, spec_add_int, spec_str_concat,
    };
    use crate::type_feedback::OperandPair;

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
                let (result, value) = spec_add_int(a, b);
                if result == SpecResult::Success {
                    let frame = vm.current_frame_mut();
                    frame.set_reg(inst.dst().0, value);
                    return ControlFlow::Continue;
                }
                // Deopt: invalidate cache and fall through to slow path
                vm.speculation_cache.invalidate(site);
            }
            Speculation::FloatFloat | Speculation::IntFloat | Speculation::FloatInt => {
                let (result, value) = spec_add_float(a, b);
                if result == SpecResult::Success {
                    let frame = vm.current_frame_mut();
                    frame.set_reg(inst.dst().0, value);
                    return ControlFlow::Continue;
                }
                vm.speculation_cache.invalidate(site);
            }
            Speculation::StrStr => {
                // String concatenation fast path
                let (result, value) = spec_str_concat(a, b);
                if result == SpecResult::Success {
                    let frame = vm.current_frame_mut();
                    frame.set_reg(inst.dst().0, value);
                    return ControlFlow::Continue;
                }
                vm.speculation_cache.invalidate(site);
            }
            Speculation::None | Speculation::StrInt | Speculation::IntStr => {
                // StrInt/IntStr don't apply to addition (only mul for repetition)
            }
        }
    }

    // =========================================================================
    // Slow Path: Full type check + feedback recording
    // =========================================================================

    // Record type feedback for future speculation
    let pair = OperandPair::from_values(a, b);
    let feedback = BinaryOpFeedback::new(code_id, bc_offset, a, b);
    feedback.record(&mut vm.ic_manager);

    // Update speculation cache for next time
    let spec = Speculation::from_operand_pair(pair);
    if spec != Speculation::None {
        vm.speculation_cache.insert(site, spec);
    }

    let frame = vm.current_frame_mut();

    // Try int + int
    if let (Some(x), Some(y)) = (a.as_int(), b.as_int()) {
        if let Some(result) = x.checked_add(y) {
            if let Some(v) = Value::int(result) {
                frame.set_reg(inst.dst().0, v);
                return ControlFlow::Continue;
            }
        }
        return ControlFlow::Error(RuntimeError::value_error("Integer overflow"));
    }

    // Try float + float or mixed int/float
    let x = if let Some(f) = a.as_float() {
        f
    } else if let Some(i) = a.as_int() {
        i as f64
    } else {
        return ControlFlow::Error(RuntimeError::unsupported_operand("+", "unknown", "unknown"));
    };
    let y = if let Some(f) = b.as_float() {
        f
    } else if let Some(i) = b.as_int() {
        i as f64
    } else {
        return ControlFlow::Error(RuntimeError::unsupported_operand("+", "unknown", "unknown"));
    };

    frame.set_reg(inst.dst().0, Value::float(x + y));
    ControlFlow::Continue
}

/// Sub: dst = src1 - src2 (generic with speculative fast-path)
///
/// Uses O(1) speculation cache lookup to select optimized code path.
/// Records type feedback on slow path for future specialization.
#[inline(always)]
pub fn sub(vm: &mut VirtualMachine, inst: Instruction) -> ControlFlow {
    use crate::ic_manager::ICSiteId;
    use crate::speculative::{SpecResult, Speculation, spec_sub_float, spec_sub_int};
    use crate::type_feedback::OperandPair;

    let frame = vm.current_frame_mut();
    let (a, b) = frame.get_regs2(inst.src1().0, inst.src2().0);
    let code_id = frame.code_id();
    let bc_offset = frame.ip.saturating_sub(1) as u32;
    let site = ICSiteId::new(code_id, bc_offset);

    // Speculative Fast Path
    if let Some(spec) = vm.speculation_cache.get(site) {
        match spec {
            Speculation::IntInt => {
                let (result, value) = spec_sub_int(a, b);
                if result == SpecResult::Success {
                    let frame = vm.current_frame_mut();
                    frame.set_reg(inst.dst().0, value);
                    return ControlFlow::Continue;
                }
                vm.speculation_cache.invalidate(site);
            }
            Speculation::FloatFloat | Speculation::IntFloat | Speculation::FloatInt => {
                let (result, value) = spec_sub_float(a, b);
                if result == SpecResult::Success {
                    let frame = vm.current_frame_mut();
                    frame.set_reg(inst.dst().0, value);
                    return ControlFlow::Continue;
                }
                vm.speculation_cache.invalidate(site);
            }
            Speculation::None | Speculation::StrStr | Speculation::StrInt | Speculation::IntStr => {
            }
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
        if let Some(result) = x.checked_sub(y) {
            if let Some(v) = Value::int(result) {
                frame.set_reg(inst.dst().0, v);
                return ControlFlow::Continue;
            }
        }
        return ControlFlow::Error(RuntimeError::value_error("Integer overflow"));
    }

    let x = if let Some(f) = a.as_float() {
        f
    } else if let Some(i) = a.as_int() {
        i as f64
    } else {
        return ControlFlow::Error(RuntimeError::unsupported_operand("-", "unknown", "unknown"));
    };
    let y = if let Some(f) = b.as_float() {
        f
    } else if let Some(i) = b.as_int() {
        i as f64
    } else {
        return ControlFlow::Error(RuntimeError::unsupported_operand("-", "unknown", "unknown"));
    };

    frame.set_reg(inst.dst().0, Value::float(x - y));
    ControlFlow::Continue
}

/// Mul: dst = src1 * src2 (generic with speculative fast-path)
///
/// Uses O(1) speculation cache lookup to select optimized code path.
#[inline(always)]
pub fn mul(vm: &mut VirtualMachine, inst: Instruction) -> ControlFlow {
    use crate::ic_manager::ICSiteId;
    use crate::speculative::{
        SpecResult, Speculation, spec_mul_float, spec_mul_int, spec_str_repeat,
    };
    use crate::type_feedback::OperandPair;

    let frame = vm.current_frame_mut();
    let (a, b) = frame.get_regs2(inst.src1().0, inst.src2().0);
    let code_id = frame.code_id();
    let bc_offset = frame.ip.saturating_sub(1) as u32;
    let site = ICSiteId::new(code_id, bc_offset);

    // Speculative Fast Path
    if let Some(spec) = vm.speculation_cache.get(site) {
        match spec {
            Speculation::IntInt => {
                let (result, value) = spec_mul_int(a, b);
                if result == SpecResult::Success {
                    let frame = vm.current_frame_mut();
                    frame.set_reg(inst.dst().0, value);
                    return ControlFlow::Continue;
                }
                vm.speculation_cache.invalidate(site);
            }
            Speculation::FloatFloat | Speculation::IntFloat | Speculation::FloatInt => {
                let (result, value) = spec_mul_float(a, b);
                if result == SpecResult::Success {
                    let frame = vm.current_frame_mut();
                    frame.set_reg(inst.dst().0, value);
                    return ControlFlow::Continue;
                }
                vm.speculation_cache.invalidate(site);
            }
            Speculation::StrInt | Speculation::IntStr => {
                // String repetition fast path (str * int or int * str)
                let (result, value) = spec_str_repeat(a, b);
                match result {
                    SpecResult::Success => {
                        let frame = vm.current_frame_mut();
                        frame.set_reg(inst.dst().0, value);
                        return ControlFlow::Continue;
                    }
                    SpecResult::Overflow => {
                        // Negative repetition count: Python returns empty string
                        // This is handled in spec_str_repeat, so we shouldn't reach here
                        // But for safety, fall through to slow path
                    }
                    SpecResult::Deopt => {
                        vm.speculation_cache.invalidate(site);
                    }
                }
            }
            Speculation::None | Speculation::StrStr => {
                // StrStr doesn't apply to multiplication
            }
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
        if let Some(result) = x.checked_mul(y) {
            if let Some(v) = Value::int(result) {
                frame.set_reg(inst.dst().0, v);
                return ControlFlow::Continue;
            }
        }
        return ControlFlow::Error(RuntimeError::value_error("Integer overflow"));
    }

    let x = if let Some(f) = a.as_float() {
        f
    } else if let Some(i) = a.as_int() {
        i as f64
    } else {
        return ControlFlow::Error(RuntimeError::unsupported_operand("*", "unknown", "unknown"));
    };
    let y = if let Some(f) = b.as_float() {
        f
    } else if let Some(i) = b.as_int() {
        i as f64
    } else {
        return ControlFlow::Error(RuntimeError::unsupported_operand("*", "unknown", "unknown"));
    };

    frame.set_reg(inst.dst().0, Value::float(x * y));
    ControlFlow::Continue
}

/// TrueDiv: dst = src1 / src2 (always returns float, with speculative fast-path)
///
/// Uses O(1) speculation cache lookup to select optimized code path.
#[inline(always)]
pub fn true_div(vm: &mut VirtualMachine, inst: Instruction) -> ControlFlow {
    use crate::ic_manager::ICSiteId;
    use crate::speculative::{SpecResult, Speculation, spec_div_float};
    use crate::type_feedback::OperandPair;

    let frame = vm.current_frame_mut();
    let (a, b) = frame.get_regs2(inst.src1().0, inst.src2().0);
    let code_id = frame.code_id();
    let bc_offset = frame.ip.saturating_sub(1) as u32;
    let site = ICSiteId::new(code_id, bc_offset);

    // Speculative Fast Path (true_div always returns float)
    if let Some(spec) = vm.speculation_cache.get(site) {
        if spec.is_float() || spec == Speculation::IntInt {
            let (result, value) = spec_div_float(a, b);
            if result == SpecResult::Success {
                let frame = vm.current_frame_mut();
                frame.set_reg(inst.dst().0, value);
                return ControlFlow::Continue;
            }
            // Division by zero or type mismatch
            if result == SpecResult::Overflow {
                return ControlFlow::Error(RuntimeError::zero_division());
            }
            vm.speculation_cache.invalidate(site);
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

    let x = if let Some(f) = a.as_float() {
        f
    } else if let Some(i) = a.as_int() {
        i as f64
    } else {
        return ControlFlow::Error(RuntimeError::unsupported_operand("/", "unknown", "unknown"));
    };
    let y = if let Some(f) = b.as_float() {
        f
    } else if let Some(i) = b.as_int() {
        i as f64
    } else {
        return ControlFlow::Error(RuntimeError::unsupported_operand("/", "unknown", "unknown"));
    };

    if y == 0.0 {
        return ControlFlow::Error(RuntimeError::zero_division());
    }

    frame.set_reg(inst.dst().0, Value::float(x / y));
    ControlFlow::Continue
}

/// FloorDiv: dst = src1 // src2 (generic with speculative fast-path)
///
/// Int // int returns int. Float // float returns float.
/// Uses O(1) speculation cache lookup to select optimized code path.
#[inline(always)]
pub fn floor_div(vm: &mut VirtualMachine, inst: Instruction) -> ControlFlow {
    use crate::ic_manager::ICSiteId;
    use crate::speculative::{SpecResult, Speculation, spec_floor_div_float, spec_floor_div_int};
    use crate::type_feedback::OperandPair;

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
                let (result, value) = spec_floor_div_int(a, b);
                match result {
                    SpecResult::Success => {
                        let frame = vm.current_frame_mut();
                        frame.set_reg(inst.dst().0, value);
                        return ControlFlow::Continue;
                    }
                    SpecResult::Overflow => {
                        // Division by zero
                        return ControlFlow::Error(RuntimeError::zero_division());
                    }
                    SpecResult::Deopt => {
                        vm.speculation_cache.invalidate(site);
                    }
                }
            }
            Speculation::FloatFloat | Speculation::IntFloat | Speculation::FloatInt => {
                let (result, value) = spec_floor_div_float(a, b);
                match result {
                    SpecResult::Success => {
                        let frame = vm.current_frame_mut();
                        frame.set_reg(inst.dst().0, value);
                        return ControlFlow::Continue;
                    }
                    SpecResult::Overflow => {
                        return ControlFlow::Error(RuntimeError::zero_division());
                    }
                    SpecResult::Deopt => {
                        vm.speculation_cache.invalidate(site);
                    }
                }
            }
            Speculation::None | Speculation::StrStr | Speculation::StrInt | Speculation::IntStr => {
            }
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

    // Int // int returns int
    if let (Some(x), Some(y)) = (a.as_int(), b.as_int()) {
        if y == 0 {
            return ControlFlow::Error(RuntimeError::zero_division());
        }
        if let Some(v) = Value::int(x.div_euclid(y)) {
            frame.set_reg(inst.dst().0, v);
            return ControlFlow::Continue;
        }
        return ControlFlow::Error(RuntimeError::value_error("Integer overflow"));
    }

    // Otherwise returns float
    let x = if let Some(f) = a.as_float() {
        f
    } else if let Some(i) = a.as_int() {
        i as f64
    } else {
        return ControlFlow::Error(RuntimeError::unsupported_operand(
            "//", "unknown", "unknown",
        ));
    };
    let y = if let Some(f) = b.as_float() {
        f
    } else if let Some(i) = b.as_int() {
        i as f64
    } else {
        return ControlFlow::Error(RuntimeError::unsupported_operand(
            "//", "unknown", "unknown",
        ));
    };

    if y == 0.0 {
        return ControlFlow::Error(RuntimeError::zero_division());
    }

    frame.set_reg(inst.dst().0, Value::float((x / y).floor()));
    ControlFlow::Continue
}

/// Mod: dst = src1 % src2 (generic with speculative fast-path)
///
/// Int % int returns int. Float % float returns float.
/// Uses O(1) speculation cache lookup to select optimized code path.
#[inline(always)]
pub fn modulo(vm: &mut VirtualMachine, inst: Instruction) -> ControlFlow {
    use crate::ic_manager::ICSiteId;
    use crate::speculative::{SpecResult, Speculation, spec_mod_float, spec_mod_int};
    use crate::type_feedback::OperandPair;

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
                let (result, value) = spec_mod_int(a, b);
                match result {
                    SpecResult::Success => {
                        let frame = vm.current_frame_mut();
                        frame.set_reg(inst.dst().0, value);
                        return ControlFlow::Continue;
                    }
                    SpecResult::Overflow => {
                        return ControlFlow::Error(RuntimeError::zero_division());
                    }
                    SpecResult::Deopt => {
                        vm.speculation_cache.invalidate(site);
                    }
                }
            }
            Speculation::FloatFloat | Speculation::IntFloat | Speculation::FloatInt => {
                let (result, value) = spec_mod_float(a, b);
                match result {
                    SpecResult::Success => {
                        let frame = vm.current_frame_mut();
                        frame.set_reg(inst.dst().0, value);
                        return ControlFlow::Continue;
                    }
                    SpecResult::Overflow => {
                        return ControlFlow::Error(RuntimeError::zero_division());
                    }
                    SpecResult::Deopt => {
                        vm.speculation_cache.invalidate(site);
                    }
                }
            }
            Speculation::None | Speculation::StrStr | Speculation::StrInt | Speculation::IntStr => {
            }
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

    if let (Some(x), Some(y)) = (a.as_int(), b.as_int()) {
        if y == 0 {
            return ControlFlow::Error(RuntimeError::zero_division());
        }
        if let Some(v) = Value::int(x.rem_euclid(y)) {
            frame.set_reg(inst.dst().0, v);
            return ControlFlow::Continue;
        }
        return ControlFlow::Error(RuntimeError::value_error("Integer overflow"));
    }

    let x = if let Some(f) = a.as_float() {
        f
    } else if let Some(i) = a.as_int() {
        i as f64
    } else {
        return ControlFlow::Error(RuntimeError::unsupported_operand("%", "unknown", "unknown"));
    };
    let y = if let Some(f) = b.as_float() {
        f
    } else if let Some(i) = b.as_int() {
        i as f64
    } else {
        return ControlFlow::Error(RuntimeError::unsupported_operand("%", "unknown", "unknown"));
    };

    if y == 0.0 {
        return ControlFlow::Error(RuntimeError::zero_division());
    }

    let result = x - y * (x / y).floor();
    frame.set_reg(inst.dst().0, Value::float(result));
    ControlFlow::Continue
}

/// Pow: dst = src1 ** src2 (generic with speculative fast-path)
///
/// Int ** positive int returns int (if no overflow). Otherwise float.
/// Uses O(1) speculation cache lookup to select optimized code path.
#[inline(always)]
pub fn pow(vm: &mut VirtualMachine, inst: Instruction) -> ControlFlow {
    use crate::ic_manager::ICSiteId;
    use crate::speculative::{SpecResult, Speculation, spec_pow_float, spec_pow_int};
    use crate::type_feedback::OperandPair;

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
                let (result, value) = spec_pow_int(a, b);
                match result {
                    SpecResult::Success => {
                        let frame = vm.current_frame_mut();
                        frame.set_reg(inst.dst().0, value);
                        return ControlFlow::Continue;
                    }
                    SpecResult::Overflow => {
                        // Overflow: fall through to slow path which converts to float
                        // Don't invalidate - this is expected behavior
                    }
                    SpecResult::Deopt => {
                        vm.speculation_cache.invalidate(site);
                    }
                }
            }
            Speculation::FloatFloat | Speculation::IntFloat | Speculation::FloatInt => {
                let (result, value) = spec_pow_float(a, b);
                if result == SpecResult::Success {
                    let frame = vm.current_frame_mut();
                    frame.set_reg(inst.dst().0, value);
                    return ControlFlow::Continue;
                }
                vm.speculation_cache.invalidate(site);
            }
            Speculation::None | Speculation::StrStr | Speculation::StrInt | Speculation::IntStr => {
            }
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

    // int ** positive int returns int
    if let (Some(base), Some(exp)) = (a.as_int(), b.as_int()) {
        if exp >= 0 && exp <= 63 {
            if let Some(result) = (base as i128).checked_pow(exp as u32) {
                if result >= i64::MIN as i128 && result <= i64::MAX as i128 {
                    if let Some(v) = Value::int(result as i64) {
                        frame.set_reg(inst.dst().0, v);
                        return ControlFlow::Continue;
                    }
                }
            }
        }
        // Fall through to float for large or negative exponents
    }

    let x = if let Some(f) = a.as_float() {
        f
    } else if let Some(i) = a.as_int() {
        i as f64
    } else {
        return ControlFlow::Error(RuntimeError::unsupported_operand(
            "**", "unknown", "unknown",
        ));
    };
    let y = if let Some(f) = b.as_float() {
        f
    } else if let Some(i) = b.as_int() {
        i as f64
    } else {
        return ControlFlow::Error(RuntimeError::unsupported_operand(
            "**", "unknown", "unknown",
        ));
    };

    frame.set_reg(inst.dst().0, Value::float(x.powf(y)));
    ControlFlow::Continue
}

/// Neg: dst = -src1 (generic)
#[inline(always)]
pub fn neg(vm: &mut VirtualMachine, inst: Instruction) -> ControlFlow {
    let frame = vm.current_frame_mut();
    let a = frame.get_reg(inst.src1().0);

    if let Some(x) = a.as_int() {
        if let Some(result) = x.checked_neg() {
            if let Some(v) = Value::int(result) {
                frame.set_reg(inst.dst().0, v);
                return ControlFlow::Continue;
            }
        }
        return ControlFlow::Error(RuntimeError::value_error("Integer overflow"));
    }

    if let Some(x) = a.as_float() {
        frame.set_reg(inst.dst().0, Value::float(-x));
        return ControlFlow::Continue;
    }

    ControlFlow::Error(RuntimeError::type_error("bad operand type for unary -"))
}

#[cfg(test)]
mod tests {
    // Arithmetic tests require full VM setup
}
