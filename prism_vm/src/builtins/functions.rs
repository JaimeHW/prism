//! Core builtin functions (len, abs, min, max, sum, pow, etc.).

use super::BuiltinError;
use prism_core::Value;
use prism_core::intern::{intern, interned_by_ptr, interned_len_by_ptr};
use prism_runtime::object::type_obj::TypeId;
use prism_runtime::types::dict::DictObject;
use prism_runtime::types::list::ListObject;
use prism_runtime::types::range::RangeObject;
use prism_runtime::types::set::SetObject;
use prism_runtime::types::string::StringObject;
use prism_runtime::types::tuple::TupleObject;
use std::hash::{Hash, Hasher};

// =============================================================================
// len
// =============================================================================

/// Builtin len function.
///
/// Returns the length of an object (list, tuple, string, dict, set, range).
pub fn builtin_len(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() != 1 {
        return Err(BuiltinError::TypeError(format!(
            "len() takes exactly one argument ({} given)",
            args.len()
        )));
    }

    let obj = args[0];

    // Tagged interned strings (Value::string) are not object pointers.
    if obj.is_string() {
        let ptr = obj
            .as_string_object_ptr()
            .ok_or_else(|| BuiltinError::TypeError("invalid interned string".to_string()))?;
        let len = interned_len_by_ptr(ptr as *const u8)
            .ok_or_else(|| BuiltinError::TypeError("invalid interned string".to_string()))?;
        return len_to_value(len, "str");
    }

    // Heap objects (list/tuple/dict/set/str/range).
    if let Some(ptr) = obj.as_object_ptr() {
        use crate::ops::objects::extract_type_id;
        let type_id = extract_type_id(ptr);

        return match type_id {
            TypeId::LIST => {
                let list = unsafe { &*(ptr as *const ListObject) };
                len_to_value(list.len(), "list")
            }
            TypeId::TUPLE => {
                let tuple = unsafe { &*(ptr as *const TupleObject) };
                len_to_value(tuple.len(), "tuple")
            }
            TypeId::DICT => {
                let dict = unsafe { &*(ptr as *const DictObject) };
                len_to_value(dict.len(), "dict")
            }
            TypeId::SET => {
                let set = unsafe { &*(ptr as *const SetObject) };
                len_to_value(set.len(), "set")
            }
            TypeId::STR => {
                let string = unsafe { &*(ptr as *const StringObject) };
                len_to_value(string.len(), "str")
            }
            TypeId::RANGE => {
                let range = unsafe { &*(ptr as *const RangeObject) };
                len_to_value(range.len(), "range")
            }
            _ => Err(BuiltinError::TypeError(format!(
                "object of type '{}' has no len()",
                type_id.name()
            ))),
        };
    }

    if obj.is_none() {
        return Err(BuiltinError::TypeError(
            "object of type 'NoneType' has no len()".to_string(),
        ));
    }
    if obj.is_int() {
        return Err(BuiltinError::TypeError(
            "object of type 'int' has no len()".to_string(),
        ));
    }
    if obj.is_float() {
        return Err(BuiltinError::TypeError(
            "object of type 'float' has no len()".to_string(),
        ));
    }
    if obj.is_bool() {
        return Err(BuiltinError::TypeError(
            "object of type 'bool' has no len()".to_string(),
        ));
    }

    Err(BuiltinError::TypeError("object has no len()".to_string()))
}

#[inline]
fn len_to_value(len: usize, type_name: &str) -> Result<Value, BuiltinError> {
    let len_i64 = i64::try_from(len)
        .map_err(|_| BuiltinError::OverflowError(format!("{} length overflow", type_name)))?;
    Value::int(len_i64)
        .ok_or_else(|| BuiltinError::OverflowError(format!("{} length overflow", type_name)))
}

// =============================================================================
// abs
// =============================================================================

/// Builtin abs function.
///
/// Returns the absolute value of a number.
pub fn builtin_abs(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() != 1 {
        return Err(BuiltinError::TypeError(format!(
            "abs() takes exactly one argument ({} given)",
            args.len()
        )));
    }

    let arg = args[0];

    if let Some(i) = arg.as_int() {
        return Value::int(i.abs()).ok_or_else(|| {
            BuiltinError::OverflowError("integer absolute value overflow".to_string())
        });
    }

    if let Some(f) = arg.as_float() {
        return Ok(Value::float(f.abs()));
    }

    Err(BuiltinError::TypeError(
        "bad operand type for abs(): expected number".to_string(),
    ))
}

// =============================================================================
// min / max
// =============================================================================

/// Builtin min function.
///
/// Returns the smallest item in an iterable or the smallest of two or more arguments.
pub fn builtin_min(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.is_empty() {
        return Err(BuiltinError::TypeError(
            "min expected at least 1 argument, got 0".to_string(),
        ));
    }

    let mut min_val = args[0];

    for arg in &args[1..] {
        // Compare integers
        if let (Some(a), Some(b)) = (arg.as_int(), min_val.as_int()) {
            if a < b {
                min_val = *arg;
            }
            continue;
        }

        // Compare floats
        if let (Some(a), Some(b)) = (arg.as_float_coerce(), min_val.as_float_coerce()) {
            if a < b {
                min_val = *arg;
            }
            continue;
        }

        // TODO: Support comparison protocol for objects
    }

    Ok(min_val)
}

/// Builtin max function.
///
/// Returns the largest item in an iterable or the largest of two or more arguments.
pub fn builtin_max(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.is_empty() {
        return Err(BuiltinError::TypeError(
            "max expected at least 1 argument, got 0".to_string(),
        ));
    }

    let mut max_val = args[0];

    for arg in &args[1..] {
        // Compare integers
        if let (Some(a), Some(b)) = (arg.as_int(), max_val.as_int()) {
            if a > b {
                max_val = *arg;
            }
            continue;
        }

        // Compare floats
        if let (Some(a), Some(b)) = (arg.as_float_coerce(), max_val.as_float_coerce()) {
            if a > b {
                max_val = *arg;
            }
            continue;
        }

        // TODO: Support comparison protocol for objects
    }

    Ok(max_val)
}

// =============================================================================
// sum
// =============================================================================

/// Builtin sum function.
///
/// Sums the items of an iterable, left to right, with optional start value.
pub fn builtin_sum(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.is_empty() || args.len() > 2 {
        return Err(BuiltinError::TypeError(format!(
            "sum expected 1 or 2 arguments, got {}",
            args.len()
        )));
    }

    let mut acc = if args.len() == 2 {
        NumericAccumulator::from_start(args[1])?
    } else {
        NumericAccumulator::Int(0)
    };

    if let Some(iter) = super::iter_dispatch::get_iterator_mut(&args[0]) {
        while let Some(item) = iter.next() {
            acc.add(item)?;
        }
    } else {
        let mut iter =
            super::iter_dispatch::value_to_iterator(&args[0]).map_err(BuiltinError::from)?;
        while let Some(item) = iter.next() {
            acc.add(item)?;
        }
    }

    acc.into_value()
}

#[derive(Clone, Copy)]
enum NumericAccumulator {
    Int(i64),
    Float(f64),
}

impl NumericAccumulator {
    #[inline]
    fn from_start(value: Value) -> Result<Self, BuiltinError> {
        if value.is_string() || is_str_object(value) {
            return Err(BuiltinError::TypeError(
                "sum() can't sum strings [use ''.join(seq) instead]".to_string(),
            ));
        }

        if let Some(i) = value.as_int() {
            return Ok(Self::Int(i));
        }
        if let Some(f) = value.as_float() {
            return Ok(Self::Float(f));
        }
        if let Some(b) = value.as_bool() {
            return Ok(Self::Int(if b { 1 } else { 0 }));
        }

        Err(BuiltinError::TypeError(format!(
            "sum() start value must be a number, not {}",
            value_type_name(value)
        )))
    }

    #[inline]
    fn add(&mut self, value: Value) -> Result<(), BuiltinError> {
        let rhs = if let Some(i) = value.as_int() {
            NumericAccumulator::Int(i)
        } else if let Some(f) = value.as_float() {
            NumericAccumulator::Float(f)
        } else if let Some(b) = value.as_bool() {
            NumericAccumulator::Int(if b { 1 } else { 0 })
        } else if value.is_string() || is_str_object(value) {
            return Err(BuiltinError::TypeError(
                "sum() can't sum strings [use ''.join(seq) instead]".to_string(),
            ));
        } else {
            return Err(BuiltinError::TypeError(format!(
                "unsupported operand type(s) for +: '{}' and '{}'",
                self.type_name(),
                value_type_name(value)
            )));
        };

        match (*self, rhs) {
            (NumericAccumulator::Int(lhs), NumericAccumulator::Int(rhs_i)) => {
                let sum = lhs.checked_add(rhs_i).ok_or_else(|| {
                    BuiltinError::OverflowError("integer overflow in sum".to_string())
                })?;
                if Value::int(sum).is_none() {
                    return Err(BuiltinError::OverflowError(
                        "integer overflow in sum".to_string(),
                    ));
                }
                *self = NumericAccumulator::Int(sum);
            }
            (NumericAccumulator::Int(lhs), NumericAccumulator::Float(rhs_f)) => {
                *self = NumericAccumulator::Float(lhs as f64 + rhs_f);
            }
            (NumericAccumulator::Float(lhs), NumericAccumulator::Int(rhs_i)) => {
                *self = NumericAccumulator::Float(lhs + rhs_i as f64);
            }
            (NumericAccumulator::Float(lhs), NumericAccumulator::Float(rhs_f)) => {
                *self = NumericAccumulator::Float(lhs + rhs_f);
            }
        }
        Ok(())
    }

    #[inline]
    fn type_name(self) -> &'static str {
        match self {
            NumericAccumulator::Int(_) => "int",
            NumericAccumulator::Float(_) => "float",
        }
    }

    #[inline]
    fn into_value(self) -> Result<Value, BuiltinError> {
        match self {
            NumericAccumulator::Int(i) => Value::int(i)
                .ok_or_else(|| BuiltinError::OverflowError("integer overflow in sum".to_string())),
            NumericAccumulator::Float(f) => Ok(Value::float(f)),
        }
    }
}

#[inline]
fn is_str_object(value: Value) -> bool {
    if let Some(ptr) = value.as_object_ptr() {
        return crate::ops::objects::extract_type_id(ptr) == TypeId::STR;
    }
    false
}

#[inline]
fn value_type_name(value: Value) -> &'static str {
    if value.is_none() {
        "NoneType"
    } else if value.as_bool().is_some() {
        "bool"
    } else if value.as_int().is_some() {
        "int"
    } else if value.as_float().is_some() {
        "float"
    } else if value.is_string() {
        "str"
    } else if let Some(ptr) = value.as_object_ptr() {
        crate::ops::objects::extract_type_id(ptr).name()
    } else {
        "object"
    }
}

// =============================================================================
// pow
// =============================================================================

/// Builtin pow function.
///
/// pow(base, exp[, mod]) - Compute base**exp, optionally modulo mod.
pub fn builtin_pow(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() < 2 || args.len() > 3 {
        return Err(BuiltinError::TypeError(format!(
            "pow expected 2 or 3 arguments, got {}",
            args.len()
        )));
    }

    let base = args[0];
    let exp = args[1];

    // Integer power
    if let (Some(b), Some(e)) = (base.as_int(), exp.as_int()) {
        if args.len() == 3 {
            // Modular exponentiation
            if let Some(m) = args[2].as_int() {
                if m == 0 {
                    return Err(BuiltinError::ValueError(
                        "pow() 3rd argument cannot be 0".to_string(),
                    ));
                }
                // Use modular exponentiation
                let result = mod_pow(b, e, m);
                return Value::int(result).ok_or_else(|| {
                    BuiltinError::OverflowError("integer overflow in pow".to_string())
                });
            }
        }

        // Simple integer power
        if e >= 0 && e <= 63 {
            if let Some(result) = b.checked_pow(e as u32) {
                return Value::int(result).ok_or_else(|| {
                    BuiltinError::OverflowError("integer overflow in pow".to_string())
                });
            }
        }

        // Fall back to float for large exponents
        let result = (b as f64).powf(e as f64);
        return Ok(Value::float(result));
    }

    // Float power
    if let (Some(b), Some(e)) = (base.as_float_coerce(), exp.as_float_coerce()) {
        let result = b.powf(e);
        return Ok(Value::float(result));
    }

    Err(BuiltinError::TypeError(
        "pow() arguments must be numeric".to_string(),
    ))
}

/// Modular exponentiation: (base^exp) mod modulus
#[inline]
fn mod_pow(mut base: i64, mut exp: i64, modulus: i64) -> i64 {
    if exp < 0 {
        // Negative exponent with modulus is not supported for integers
        return 0;
    }

    let mut result: i64 = 1;
    base = base.rem_euclid(modulus);

    while exp > 0 {
        if exp % 2 == 1 {
            result = (result * base).rem_euclid(modulus);
        }
        exp /= 2;
        base = (base * base).rem_euclid(modulus);
    }

    result
}

// =============================================================================
// round
// =============================================================================

/// Builtin round function.
///
/// round(number[, ndigits]) - Round a number to a given precision.
pub fn builtin_round(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.is_empty() || args.len() > 2 {
        return Err(BuiltinError::TypeError(format!(
            "round expected 1 or 2 arguments, got {}",
            args.len()
        )));
    }

    let ndigits = if args.len() == 2 {
        args[1].as_int().unwrap_or(0)
    } else {
        0
    };

    if let Some(i) = args[0].as_int() {
        // Rounding an integer with no ndigits returns the integer
        if ndigits >= 0 {
            return Ok(args[0]);
        }
        // Negative ndigits: round to nearest 10^(-ndigits)
        let factor = 10i64.pow((-ndigits) as u32);
        let rounded = ((i as f64 / factor as f64).round() * factor as f64) as i64;
        return Value::int(rounded)
            .ok_or_else(|| BuiltinError::OverflowError("integer overflow in round".to_string()));
    }

    if let Some(f) = args[0].as_float() {
        if ndigits == 0 {
            // Round to integer
            let rounded = f.round();
            if rounded >= i64::MIN as f64 && rounded <= i64::MAX as f64 {
                return Value::int(rounded as i64).ok_or_else(|| {
                    BuiltinError::OverflowError("integer overflow in round".to_string())
                });
            }
            return Ok(Value::float(rounded));
        }
        // Round to ndigits decimal places
        let factor = 10f64.powi(ndigits as i32);
        let rounded = (f * factor).round() / factor;
        return Ok(Value::float(rounded));
    }

    Err(BuiltinError::TypeError(
        "round() argument must be a number".to_string(),
    ))
}

// =============================================================================
// divmod
// =============================================================================

/// Builtin divmod function.
///
/// divmod(a, b) - Return (a // b, a % b) as a tuple.
pub fn builtin_divmod(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() != 2 {
        return Err(BuiltinError::TypeError(format!(
            "divmod expected 2 arguments, got {}",
            args.len()
        )));
    }

    // Integer divmod (includes bool as int subclass semantics).
    if args[0].as_float().is_none() && args[1].as_float().is_none() {
        let a = if let Some(i) = args[0].as_int() {
            i
        } else if let Some(b) = args[0].as_bool() {
            if b { 1 } else { 0 }
        } else {
            return Err(BuiltinError::TypeError(
                "divmod() arguments must be numeric".to_string(),
            ));
        };
        let b = if let Some(i) = args[1].as_int() {
            i
        } else if let Some(b) = args[1].as_bool() {
            if b { 1 } else { 0 }
        } else {
            return Err(BuiltinError::TypeError(
                "divmod() arguments must be numeric".to_string(),
            ));
        };

        if b == 0 {
            return Err(BuiltinError::ValueError(
                "integer division or modulo by zero".to_string(),
            ));
        }

        let mut quotient = a / b;
        let mut remainder = a % b;
        if remainder != 0 && (remainder < 0) != (b < 0) {
            quotient -= 1;
            remainder += b;
        }
        let q = Value::int(quotient)
            .ok_or_else(|| BuiltinError::OverflowError("integer overflow in divmod".to_string()))?;
        let r = Value::int(remainder)
            .ok_or_else(|| BuiltinError::OverflowError("integer overflow in divmod".to_string()))?;
        return Ok(make_tuple2(q, r));
    }

    let a = if let Some(f) = args[0].as_float() {
        f
    } else if let Some(i) = args[0].as_int() {
        i as f64
    } else if let Some(b) = args[0].as_bool() {
        if b { 1.0 } else { 0.0 }
    } else {
        return Err(BuiltinError::TypeError(
            "divmod() arguments must be numeric".to_string(),
        ));
    };
    let b = if let Some(f) = args[1].as_float() {
        f
    } else if let Some(i) = args[1].as_int() {
        i as f64
    } else if let Some(b) = args[1].as_bool() {
        if b { 1.0 } else { 0.0 }
    } else {
        return Err(BuiltinError::TypeError(
            "divmod() arguments must be numeric".to_string(),
        ));
    };

    if b == 0.0 {
        return Err(BuiltinError::ValueError(
            "float divmod() by zero".to_string(),
        ));
    }

    // Python semantics: floor division quotient and remainder with sign of divisor.
    let quotient = (a / b).floor();
    let remainder = a - quotient * b;
    return Ok(make_tuple2(Value::float(quotient), Value::float(remainder)));
}

#[inline]
fn make_tuple2(a: Value, b: Value) -> Value {
    let tuple = TupleObject::from_slice(&[a, b]);
    let ptr = Box::leak(Box::new(tuple)) as *mut TupleObject as *const ();
    Value::object_ptr(ptr)
}

// =============================================================================
// hash
// =============================================================================

/// Builtin hash function.
///
/// hash(object) - Return the hash value of the object.
pub fn builtin_hash(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() != 1 {
        return Err(BuiltinError::TypeError(format!(
            "hash() takes exactly one argument ({} given)",
            args.len()
        )));
    }

    let hash = hash_value(args[0])?;
    let mut signed = lower_48_signed(hash);
    if signed == -1 {
        signed = -2;
    }
    Value::int(signed).ok_or_else(|| BuiltinError::OverflowError("hash overflow".to_string()))
}

#[inline]
fn hash_value(value: Value) -> Result<u64, BuiltinError> {
    if value.is_none()
        || value.as_bool().is_some()
        || value.as_int().is_some()
        || value.as_float().is_some()
    {
        return Ok(hash_with_default_hasher(&value));
    }

    if value.is_string() {
        let ptr = value
            .as_string_object_ptr()
            .ok_or_else(|| BuiltinError::TypeError("invalid interned string".to_string()))?;
        let interned = interned_by_ptr(ptr as *const u8)
            .ok_or_else(|| BuiltinError::TypeError("invalid interned string".to_string()))?;
        return Ok(hash_with_default_hasher(&interned.as_str()));
    }

    let ptr = value
        .as_object_ptr()
        .ok_or_else(|| BuiltinError::TypeError("unhashable type".to_string()))?;
    let type_id = crate::ops::objects::extract_type_id(ptr);
    match type_id {
        TypeId::LIST | TypeId::DICT | TypeId::SET => Err(BuiltinError::TypeError(format!(
            "unhashable type: '{}'",
            type_id.name()
        ))),
        TypeId::STR => {
            let string = unsafe { &*(ptr as *const StringObject) };
            Ok(hash_with_default_hasher(&string.as_str()))
        }
        TypeId::TUPLE => {
            let tuple = unsafe { &*(ptr as *const TupleObject) };
            hash_tuple(tuple)
        }
        _ => Ok(hash_with_default_hasher(&(ptr as usize))),
    }
}

#[inline]
fn hash_tuple(tuple: &TupleObject) -> Result<u64, BuiltinError> {
    // CPython-inspired tuple hash combiner.
    let mut acc: u64 = 0x345678;
    let len = tuple.len();
    for (index, item) in tuple.iter().copied().enumerate() {
        let item_hash = hash_value(item)?;
        acc = (acc ^ item_hash).wrapping_mul(1_000_003);
        acc = acc.wrapping_add((index as u64).wrapping_mul(2).wrapping_add(82_520));
    }
    acc ^= len as u64;
    if acc == u64::MAX {
        acc = u64::MAX - 1;
    }
    Ok(acc)
}

#[inline]
fn hash_with_default_hasher<T: Hash>(value: &T) -> u64 {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    value.hash(&mut hasher);
    hasher.finish()
}

#[inline]
fn lower_48_signed(value: u64) -> i64 {
    let masked = value & ((1u64 << 48) - 1);
    if (masked & (1u64 << 47)) != 0 {
        masked as i64 - (1i64 << 48)
    } else {
        masked as i64
    }
}

// =============================================================================
// id
// =============================================================================

/// Builtin id function.
///
/// id(object) - Return the identity of an object (its memory address).
pub fn builtin_id(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() != 1 {
        return Err(BuiltinError::TypeError(format!(
            "id() takes exactly one argument ({} given)",
            args.len()
        )));
    }

    let obj = args[0];

    // For objects, return pointer address
    if let Some(ptr) = obj.as_object_ptr() {
        return Value::int(ptr as i64)
            .ok_or_else(|| BuiltinError::OverflowError("id overflow".to_string()));
    }

    // For primitives, compute a stable ID based on the actual value
    // Python semantics: identical primitive values may share ID (interning)
    if obj.is_none() {
        return Ok(Value::int(0x_DEAD_BEEF).unwrap());
    }
    if let Some(b) = obj.as_bool() {
        return Ok(Value::int(if b { 0x_1 } else { 0x_0 }).unwrap());
    }
    if let Some(i) = obj.as_int() {
        // Small integers get stable IDs (like CPython's -5 to 256 cache)
        return Value::int(i).ok_or_else(|| BuiltinError::OverflowError("id overflow".to_string()));
    }
    if let Some(f) = obj.as_float() {
        // Use float bits as ID
        return Value::int(f.to_bits() as i64)
            .ok_or_else(|| BuiltinError::OverflowError("id overflow".to_string()));
    }

    // Fallback
    Ok(Value::int(0).unwrap())
}

// =============================================================================
// callable
// =============================================================================

/// Builtin callable function.
///
/// callable(object) - Return True if the object appears callable.
pub fn builtin_callable(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() != 1 {
        return Err(BuiltinError::TypeError(format!(
            "callable() takes exactly one argument ({} given)",
            args.len()
        )));
    }

    let obj = args[0];

    // Primitives are not callable
    if obj.is_none() || obj.is_bool() || obj.is_int() || obj.is_float() {
        return Ok(Value::bool(false));
    }
    if obj.is_string() {
        return Ok(Value::bool(false));
    }

    if let Some(ptr) = obj.as_object_ptr() {
        let type_id = crate::ops::objects::extract_type_id(ptr);
        let is_callable = matches!(
            type_id,
            TypeId::FUNCTION
                | TypeId::METHOD
                | TypeId::CLOSURE
                | TypeId::TYPE
                | TypeId::BUILTIN_FUNCTION
                | TypeId::EXCEPTION_TYPE
        );
        return Ok(Value::bool(is_callable));
    }

    Ok(Value::bool(false))
}

// =============================================================================
// repr / ascii
// =============================================================================

/// Builtin repr function.
///
/// repr(object) - Return a string containing a printable representation.
pub fn builtin_repr(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() != 1 {
        return Err(BuiltinError::TypeError(format!(
            "repr() takes exactly one argument ({} given)",
            args.len()
        )));
    }

    let repr = repr_value(args[0], 0)?;
    Ok(Value::string(intern(&repr)))
}

/// Builtin ascii function.
///
/// ascii(object) - Like repr() but escape non-ASCII characters.
pub fn builtin_ascii(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() != 1 {
        return Err(BuiltinError::TypeError(format!(
            "ascii() takes exactly one argument ({} given)",
            args.len()
        )));
    }

    let repr = repr_value(args[0], 0)?;
    let ascii = escape_non_ascii(&repr);
    Ok(Value::string(intern(&ascii)))
}

const MAX_REPR_DEPTH: usize = 12;

fn repr_value(value: Value, depth: usize) -> Result<String, BuiltinError> {
    if depth >= MAX_REPR_DEPTH {
        return Ok("...".to_string());
    }

    if value.is_none() {
        return Ok("None".to_string());
    }
    if let Some(b) = value.as_bool() {
        return Ok(if b {
            "True".to_string()
        } else {
            "False".to_string()
        });
    }
    if let Some(i) = value.as_int() {
        return Ok(i.to_string());
    }
    if let Some(f) = value.as_float() {
        if f.fract() == 0.0 && f.is_finite() {
            return Ok(format!("{:.1}", f));
        }
        return Ok(f.to_string());
    }
    if value.is_string() {
        let ptr = value
            .as_string_object_ptr()
            .ok_or_else(|| BuiltinError::TypeError("invalid interned string".to_string()))?;
        let interned = interned_by_ptr(ptr as *const u8)
            .ok_or_else(|| BuiltinError::TypeError("invalid interned string".to_string()))?;
        return Ok(quote_python_string(interned.as_str()));
    }

    let ptr = value
        .as_object_ptr()
        .ok_or_else(|| BuiltinError::TypeError("invalid object reference".to_string()))?;
    let type_id = crate::ops::objects::extract_type_id(ptr);
    match type_id {
        TypeId::STR => {
            let string = unsafe { &*(ptr as *const StringObject) };
            Ok(quote_python_string(string.as_str()))
        }
        TypeId::LIST => {
            let list = unsafe { &*(ptr as *const ListObject) };
            let mut out = String::from("[");
            let mut first = true;
            for item in list.iter() {
                if !first {
                    out.push_str(", ");
                }
                first = false;
                out.push_str(&repr_value(*item, depth + 1)?);
            }
            out.push(']');
            Ok(out)
        }
        TypeId::TUPLE => {
            let tuple = unsafe { &*(ptr as *const TupleObject) };
            if tuple.is_empty() {
                return Ok("()".to_string());
            }
            let mut out = String::from("(");
            for (index, item) in tuple.iter().enumerate() {
                if index > 0 {
                    out.push_str(", ");
                }
                out.push_str(&repr_value(*item, depth + 1)?);
            }
            if tuple.len() == 1 {
                out.push(',');
            }
            out.push(')');
            Ok(out)
        }
        TypeId::DICT => {
            let dict = unsafe { &*(ptr as *const DictObject) };
            let mut out = String::from("{");
            let mut first = true;
            for (key, value) in dict.iter() {
                if !first {
                    out.push_str(", ");
                }
                first = false;
                out.push_str(&repr_value(key, depth + 1)?);
                out.push_str(": ");
                out.push_str(&repr_value(value, depth + 1)?);
            }
            out.push('}');
            Ok(out)
        }
        TypeId::SET => {
            let set = unsafe { &*(ptr as *const SetObject) };
            if set.is_empty() {
                return Ok("set()".to_string());
            }
            let mut out = String::from("{");
            let mut first = true;
            for value in set.iter() {
                if !first {
                    out.push_str(", ");
                }
                first = false;
                out.push_str(&repr_value(value, depth + 1)?);
            }
            out.push('}');
            Ok(out)
        }
        TypeId::RANGE => {
            let range = unsafe { &*(ptr as *const RangeObject) };
            if range.step == 1 {
                Ok(format!("range({}, {})", range.start, range.stop))
            } else {
                Ok(format!(
                    "range({}, {}, {})",
                    range.start, range.stop, range.step
                ))
            }
        }
        _ => Ok(format!(
            "<{} object at 0x{:x}>",
            type_id.name(),
            ptr as usize
        )),
    }
}

fn quote_python_string(input: &str) -> String {
    let mut out = String::with_capacity(input.len() + 2);
    out.push('\'');
    for ch in input.chars() {
        match ch {
            '\\' => out.push_str("\\\\"),
            '\'' => out.push_str("\\'"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if c.is_control() => out.push_str(&escape_char(c)),
            c => out.push(c),
        }
    }
    out.push('\'');
    out
}

#[inline]
fn escape_non_ascii(input: &str) -> String {
    let mut out = String::with_capacity(input.len());
    for ch in input.chars() {
        if ch.is_ascii() {
            out.push(ch);
        } else {
            out.push_str(&escape_char(ch));
        }
    }
    out
}

#[inline]
fn escape_char(ch: char) -> String {
    let code = ch as u32;
    if code <= 0xFF {
        format!("\\x{:02x}", code)
    } else if code <= 0xFFFF {
        format!("\\u{:04x}", code)
    } else {
        format!("\\U{:08x}", code)
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builtins::BuiltinFunctionObject;
    use crate::builtins::itertools::{builtin_iter, builtin_next};
    use prism_core::intern::intern;
    use prism_core::value::SMALL_INT_MAX;
    use prism_runtime::object::ObjectHeader;

    fn boxed_value<T>(obj: T) -> (Value, *mut T) {
        let ptr = Box::into_raw(Box::new(obj));
        (Value::object_ptr(ptr as *const ()), ptr)
    }

    unsafe fn drop_boxed<T>(ptr: *mut T) {
        drop(unsafe { Box::from_raw(ptr) });
    }

    fn tagged_string_value_to_rust_string(value: Value) -> String {
        assert!(value.is_string(), "expected tagged string, got {value:?}");
        let ptr = value
            .as_string_object_ptr()
            .expect("tagged string pointer missing") as *const u8;
        prism_core::intern::interned_by_ptr(ptr)
            .expect("tagged string pointer not interned")
            .as_str()
            .to_string()
    }

    #[test]
    fn test_len_tagged_string() {
        let value = Value::string(intern("hello"));
        let result = builtin_len(&[value]).unwrap();
        assert_eq!(result.as_int(), Some(5));
    }

    #[test]
    fn test_len_list_object() {
        let list = ListObject::from_slice(&[
            Value::int(1).unwrap(),
            Value::int(2).unwrap(),
            Value::int(3).unwrap(),
        ]);
        let (value, ptr) = boxed_value(list);
        let result = builtin_len(&[value]).unwrap();
        assert_eq!(result.as_int(), Some(3));
        unsafe { drop_boxed(ptr) };
    }

    #[test]
    fn test_len_tuple_object() {
        let tuple = TupleObject::from_slice(&[
            Value::int(10).unwrap(),
            Value::int(20).unwrap(),
            Value::int(30).unwrap(),
            Value::int(40).unwrap(),
        ]);
        let (value, ptr) = boxed_value(tuple);
        let result = builtin_len(&[value]).unwrap();
        assert_eq!(result.as_int(), Some(4));
        unsafe { drop_boxed(ptr) };
    }

    #[test]
    fn test_len_dict_object() {
        let mut dict = DictObject::new();
        dict.set(Value::int(1).unwrap(), Value::int(11).unwrap());
        dict.set(Value::int(2).unwrap(), Value::int(22).unwrap());
        let (value, ptr) = boxed_value(dict);
        let result = builtin_len(&[value]).unwrap();
        assert_eq!(result.as_int(), Some(2));
        unsafe { drop_boxed(ptr) };
    }

    #[test]
    fn test_len_set_object() {
        let set = SetObject::from_slice(&[
            Value::int(1).unwrap(),
            Value::int(2).unwrap(),
            Value::int(2).unwrap(),
            Value::int(3).unwrap(),
        ]);
        let (value, ptr) = boxed_value(set);
        let result = builtin_len(&[value]).unwrap();
        assert_eq!(result.as_int(), Some(3));
        unsafe { drop_boxed(ptr) };
    }

    #[test]
    fn test_len_range_object() {
        let range = RangeObject::new(0, 10, 2);
        let (value, ptr) = boxed_value(range);
        let result = builtin_len(&[value]).unwrap();
        assert_eq!(result.as_int(), Some(5));
        unsafe { drop_boxed(ptr) };
    }

    #[test]
    fn test_len_string_object() {
        let string = StringObject::new("runtime");
        let (value, ptr) = boxed_value(string);
        let result = builtin_len(&[value]).unwrap();
        assert_eq!(result.as_int(), Some(7));
        unsafe { drop_boxed(ptr) };
    }

    #[test]
    fn test_len_arity_error() {
        let result = builtin_len(&[]);
        assert!(matches!(result, Err(BuiltinError::TypeError(_))));
    }

    #[test]
    fn test_len_non_sized_type_error() {
        let result = builtin_len(&[Value::int(42).unwrap()]);
        assert!(matches!(result, Err(BuiltinError::TypeError(_))));
    }

    #[test]
    fn test_abs_int() {
        let result = builtin_abs(&[Value::int(-42).unwrap()]).unwrap();
        assert_eq!(result.as_int(), Some(42));

        let result = builtin_abs(&[Value::int(42).unwrap()]).unwrap();
        assert_eq!(result.as_int(), Some(42));
    }

    #[test]
    fn test_abs_float() {
        let result = builtin_abs(&[Value::float(-3.14)]).unwrap();
        assert_eq!(result.as_float(), Some(3.14));
    }

    #[test]
    fn test_abs_error() {
        let result = builtin_abs(&[Value::none()]);
        assert!(result.is_err());
    }

    #[test]
    fn test_min() {
        let result = builtin_min(&[
            Value::int(5).unwrap(),
            Value::int(3).unwrap(),
            Value::int(8).unwrap(),
        ])
        .unwrap();
        assert_eq!(result.as_int(), Some(3));
    }

    #[test]
    fn test_max() {
        let result = builtin_max(&[
            Value::int(5).unwrap(),
            Value::int(3).unwrap(),
            Value::int(8).unwrap(),
        ])
        .unwrap();
        assert_eq!(result.as_int(), Some(8));
    }

    #[test]
    fn test_sum_int_list() {
        let list = ListObject::from_slice(&[
            Value::int(1).unwrap(),
            Value::int(2).unwrap(),
            Value::int(3).unwrap(),
            Value::int(4).unwrap(),
        ]);
        let (value, ptr) = boxed_value(list);
        let result = builtin_sum(&[value]).unwrap();
        assert_eq!(result.as_int(), Some(10));
        unsafe { drop_boxed(ptr) };
    }

    #[test]
    fn test_sum_float_list() {
        let list =
            ListObject::from_slice(&[Value::float(1.5), Value::float(2.0), Value::float(3.5)]);
        let (value, ptr) = boxed_value(list);
        let result = builtin_sum(&[value]).unwrap();
        assert_eq!(result.as_float(), Some(7.0));
        unsafe { drop_boxed(ptr) };
    }

    #[test]
    fn test_sum_mixed_numeric_promotes_to_float() {
        let list = ListObject::from_slice(&[Value::int(1).unwrap(), Value::float(2.5)]);
        let (value, ptr) = boxed_value(list);
        let result = builtin_sum(&[value]).unwrap();
        assert_eq!(result.as_float(), Some(3.5));
        unsafe { drop_boxed(ptr) };
    }

    #[test]
    fn test_sum_bool_items() {
        let list =
            ListObject::from_slice(&[Value::bool(true), Value::bool(false), Value::bool(true)]);
        let (value, ptr) = boxed_value(list);
        let result = builtin_sum(&[value]).unwrap();
        assert_eq!(result.as_int(), Some(2));
        unsafe { drop_boxed(ptr) };
    }

    #[test]
    fn test_sum_with_int_start() {
        let list = ListObject::from_slice(&[Value::int(2).unwrap(), Value::int(3).unwrap()]);
        let (value, ptr) = boxed_value(list);
        let result = builtin_sum(&[value, Value::int(10).unwrap()]).unwrap();
        assert_eq!(result.as_int(), Some(15));
        unsafe { drop_boxed(ptr) };
    }

    #[test]
    fn test_sum_with_float_start() {
        let list = ListObject::from_slice(&[Value::int(1).unwrap(), Value::int(2).unwrap()]);
        let (value, ptr) = boxed_value(list);
        let result = builtin_sum(&[value, Value::float(0.5)]).unwrap();
        assert_eq!(result.as_float(), Some(3.5));
        unsafe { drop_boxed(ptr) };
    }

    #[test]
    fn test_sum_range() {
        let range = RangeObject::new(1, 5, 1);
        let (value, ptr) = boxed_value(range);
        let result = builtin_sum(&[value]).unwrap();
        assert_eq!(result.as_int(), Some(10));
        unsafe { drop_boxed(ptr) };
    }

    #[test]
    fn test_sum_iterator_consumes_iterator_state() {
        let list = ListObject::from_slice(&[
            Value::int(1).unwrap(),
            Value::int(2).unwrap(),
            Value::int(3).unwrap(),
        ]);
        let (list_value, list_ptr) = boxed_value(list);
        let iter = builtin_iter(&[list_value]).unwrap();

        let result = builtin_sum(&[iter]).unwrap();
        assert_eq!(result.as_int(), Some(6));

        // Iterator should be exhausted after sum consumes it.
        let next_result = builtin_next(&[iter]);
        assert!(next_result.is_err());
        unsafe { drop_boxed(list_ptr) };
    }

    #[test]
    fn test_sum_non_iterable_error() {
        let err = builtin_sum(&[Value::int(42).unwrap()]).unwrap_err();
        assert!(matches!(err, BuiltinError::TypeError(_)));
    }

    #[test]
    fn test_sum_non_numeric_start_error() {
        let list = ListObject::from_slice(&[Value::int(1).unwrap()]);
        let (value, ptr) = boxed_value(list);
        let err = builtin_sum(&[value, Value::none()]).unwrap_err();
        assert!(matches!(err, BuiltinError::TypeError(_)));
        unsafe { drop_boxed(ptr) };
    }

    #[test]
    fn test_sum_string_start_error() {
        let list = ListObject::from_slice(&[Value::int(1).unwrap()]);
        let (value, ptr) = boxed_value(list);
        let err = builtin_sum(&[value, Value::string(intern("x"))]).unwrap_err();
        assert!(matches!(err, BuiltinError::TypeError(_)));
        let msg = err.to_string();
        assert!(msg.contains("can't sum strings"));
        unsafe { drop_boxed(ptr) };
    }

    #[test]
    fn test_sum_string_item_error() {
        let list = ListObject::from_slice(&[Value::int(1).unwrap(), Value::string(intern("x"))]);
        let (value, ptr) = boxed_value(list);
        let err = builtin_sum(&[value]).unwrap_err();
        assert!(matches!(err, BuiltinError::TypeError(_)));
        let msg = err.to_string();
        assert!(msg.contains("can't sum strings"));
        unsafe { drop_boxed(ptr) };
    }

    #[test]
    fn test_sum_overflow_small_int_domain() {
        let list =
            ListObject::from_slice(&[Value::int(SMALL_INT_MAX).unwrap(), Value::int(1).unwrap()]);
        let (value, ptr) = boxed_value(list);
        let err = builtin_sum(&[value]).unwrap_err();
        assert!(matches!(err, BuiltinError::OverflowError(_)));
        unsafe { drop_boxed(ptr) };
    }

    #[test]
    fn test_pow_int() {
        let result = builtin_pow(&[Value::int(2).unwrap(), Value::int(10).unwrap()]).unwrap();
        assert_eq!(result.as_int(), Some(1024));
    }

    #[test]
    fn test_pow_mod() {
        let result = builtin_pow(&[
            Value::int(2).unwrap(),
            Value::int(10).unwrap(),
            Value::int(100).unwrap(),
        ])
        .unwrap();
        assert_eq!(result.as_int(), Some(24)); // 1024 % 100 = 24
    }

    #[test]
    fn test_round_int() {
        let result = builtin_round(&[Value::int(42).unwrap()]).unwrap();
        assert_eq!(result.as_int(), Some(42));
    }

    #[test]
    fn test_round_float() {
        let result = builtin_round(&[Value::float(3.7)]).unwrap();
        assert_eq!(result.as_int(), Some(4));
    }

    #[test]
    fn test_divmod_int_returns_tuple() {
        let result = builtin_divmod(&[Value::int(17).unwrap(), Value::int(5).unwrap()]).unwrap();
        let ptr = result.as_object_ptr().expect("divmod should return tuple");
        let tuple = unsafe { &*(ptr as *const TupleObject) };
        assert_eq!(tuple.len(), 2);
        assert_eq!(tuple.get(0).unwrap().as_int(), Some(3));
        assert_eq!(tuple.get(1).unwrap().as_int(), Some(2));
    }

    #[test]
    fn test_divmod_int_negative_divisor() {
        let result = builtin_divmod(&[Value::int(7).unwrap(), Value::int(-3).unwrap()]).unwrap();
        let ptr = result.as_object_ptr().expect("divmod should return tuple");
        let tuple = unsafe { &*(ptr as *const TupleObject) };
        assert_eq!(tuple.get(0).unwrap().as_int(), Some(-3));
        assert_eq!(tuple.get(1).unwrap().as_int(), Some(-2));
    }

    #[test]
    fn test_divmod_float_returns_tuple() {
        let result = builtin_divmod(&[Value::float(7.5), Value::float(2.0)]).unwrap();
        let ptr = result.as_object_ptr().expect("divmod should return tuple");
        let tuple = unsafe { &*(ptr as *const TupleObject) };
        assert_eq!(tuple.get(0).unwrap().as_float(), Some(3.0));
        assert_eq!(tuple.get(1).unwrap().as_float(), Some(1.5));
    }

    #[test]
    fn test_divmod_mixed_numeric() {
        let result = builtin_divmod(&[Value::int(7).unwrap(), Value::float(2.0)]).unwrap();
        let ptr = result.as_object_ptr().expect("divmod should return tuple");
        let tuple = unsafe { &*(ptr as *const TupleObject) };
        assert_eq!(tuple.get(0).unwrap().as_float(), Some(3.0));
        assert_eq!(tuple.get(1).unwrap().as_float(), Some(1.0));
    }

    #[test]
    fn test_divmod_zero_errors() {
        let int_err = builtin_divmod(&[Value::int(1).unwrap(), Value::int(0).unwrap()]);
        assert!(matches!(int_err, Err(BuiltinError::ValueError(_))));

        let float_err = builtin_divmod(&[Value::float(1.0), Value::float(0.0)]);
        assert!(matches!(float_err, Err(BuiltinError::ValueError(_))));
    }

    #[test]
    fn test_repr_primitives() {
        assert_eq!(
            tagged_string_value_to_rust_string(builtin_repr(&[Value::none()]).unwrap()),
            "None"
        );
        assert_eq!(
            tagged_string_value_to_rust_string(builtin_repr(&[Value::bool(true)]).unwrap()),
            "True"
        );
        assert_eq!(
            tagged_string_value_to_rust_string(builtin_repr(&[Value::int(42).unwrap()]).unwrap()),
            "42"
        );
        assert_eq!(
            tagged_string_value_to_rust_string(builtin_repr(&[Value::float(1.5)]).unwrap()),
            "1.5"
        );
    }

    #[test]
    fn test_repr_tagged_string_escaping() {
        let value = Value::string(intern("a'b\n"));
        let repr = tagged_string_value_to_rust_string(builtin_repr(&[value]).unwrap());
        assert_eq!(repr, "'a\\'b\\n'");
    }

    #[test]
    fn test_repr_runtime_string() {
        let (value, ptr) = boxed_value(StringObject::new("runtime"));
        let repr = tagged_string_value_to_rust_string(builtin_repr(&[value]).unwrap());
        assert_eq!(repr, "'runtime'");
        unsafe { drop_boxed(ptr) };
    }

    #[test]
    fn test_repr_containers() {
        let list = ListObject::from_slice(&[Value::int(1).unwrap(), Value::int(2).unwrap()]);
        let (list_value, list_ptr) = boxed_value(list);
        let list_repr = tagged_string_value_to_rust_string(builtin_repr(&[list_value]).unwrap());
        assert_eq!(list_repr, "[1, 2]");
        unsafe { drop_boxed(list_ptr) };

        let tuple = TupleObject::from_slice(&[Value::int(1).unwrap()]);
        let (tuple_value, tuple_ptr) = boxed_value(tuple);
        let tuple_repr = tagged_string_value_to_rust_string(builtin_repr(&[tuple_value]).unwrap());
        assert_eq!(tuple_repr, "(1,)");
        unsafe { drop_boxed(tuple_ptr) };

        let mut dict = DictObject::new();
        dict.set(Value::int(1).unwrap(), Value::int(2).unwrap());
        let (dict_value, dict_ptr) = boxed_value(dict);
        let dict_repr = tagged_string_value_to_rust_string(builtin_repr(&[dict_value]).unwrap());
        assert_eq!(dict_repr, "{1: 2}");
        unsafe { drop_boxed(dict_ptr) };

        let mut set = SetObject::new();
        set.add(Value::int(3).unwrap());
        let (set_value, set_ptr) = boxed_value(set);
        let set_repr = tagged_string_value_to_rust_string(builtin_repr(&[set_value]).unwrap());
        assert_eq!(set_repr, "{3}");
        unsafe { drop_boxed(set_ptr) };
    }

    #[test]
    fn test_repr_range_object() {
        let (range_value, range_ptr) = boxed_value(RangeObject::new(1, 6, 2));
        let repr = tagged_string_value_to_rust_string(builtin_repr(&[range_value]).unwrap());
        assert_eq!(repr, "range(1, 6, 2)");
        unsafe { drop_boxed(range_ptr) };
    }

    #[test]
    fn test_ascii_non_ascii_escaping() {
        let tagged = Value::string(intern("hé"));
        let tagged_ascii = tagged_string_value_to_rust_string(builtin_ascii(&[tagged]).unwrap());
        assert_eq!(tagged_ascii, "'h\\xe9'");

        let (runtime, ptr) = boxed_value(StringObject::new("漢"));
        let runtime_ascii = tagged_string_value_to_rust_string(builtin_ascii(&[runtime]).unwrap());
        assert_eq!(runtime_ascii, "'\\u6f22'");
        unsafe { drop_boxed(ptr) };
    }

    #[test]
    fn test_repr_ascii_arity_errors() {
        let repr_err = builtin_repr(&[]);
        assert!(matches!(repr_err, Err(BuiltinError::TypeError(_))));

        let ascii_err = builtin_ascii(&[]);
        assert!(matches!(ascii_err, Err(BuiltinError::TypeError(_))));
    }

    #[test]
    fn test_hash_int() {
        let result = builtin_hash(&[Value::int(42).unwrap()]).unwrap();
        assert!(result.as_int().is_some());
    }

    #[test]
    fn test_hash_int_float_equivalence() {
        let int_hash = builtin_hash(&[Value::int(42).unwrap()]).unwrap();
        let float_hash = builtin_hash(&[Value::float(42.0)]).unwrap();
        assert_eq!(int_hash.as_int(), float_hash.as_int());
    }

    #[test]
    fn test_hash_tagged_string_by_content() {
        let a = builtin_hash(&[Value::string(intern("alpha"))]).unwrap();
        let b = builtin_hash(&[Value::string(intern("alpha"))]).unwrap();
        assert_eq!(a.as_int(), b.as_int());
    }

    #[test]
    fn test_hash_runtime_string_matches_tagged_string() {
        let tagged = builtin_hash(&[Value::string(intern("runtime"))]).unwrap();
        let (runtime_value, ptr) = boxed_value(StringObject::new("runtime"));
        let runtime = builtin_hash(&[runtime_value]).unwrap();
        assert_eq!(tagged.as_int(), runtime.as_int());
        unsafe { drop_boxed(ptr) };
    }

    #[test]
    fn test_hash_tuple_by_contents() {
        let tuple1 = TupleObject::from_slice(&[Value::int(1).unwrap(), Value::int(2).unwrap()]);
        let (tuple1_value, tuple1_ptr) = boxed_value(tuple1);
        let hash1 = builtin_hash(&[tuple1_value]).unwrap();

        let tuple2 = TupleObject::from_slice(&[Value::int(1).unwrap(), Value::int(2).unwrap()]);
        let (tuple2_value, tuple2_ptr) = boxed_value(tuple2);
        let hash2 = builtin_hash(&[tuple2_value]).unwrap();

        assert_eq!(hash1.as_int(), hash2.as_int());
        unsafe { drop_boxed(tuple1_ptr) };
        unsafe { drop_boxed(tuple2_ptr) };
    }

    #[test]
    fn test_hash_tuple_unhashable_member_error() {
        let list = ListObject::from_slice(&[Value::int(1).unwrap()]);
        let (list_value, list_ptr) = boxed_value(list);
        let tuple = TupleObject::from_slice(&[list_value]);
        let (tuple_value, tuple_ptr) = boxed_value(tuple);

        let err = builtin_hash(&[tuple_value]).unwrap_err();
        assert!(matches!(err, BuiltinError::TypeError(_)));
        assert!(err.to_string().contains("unhashable type"));

        unsafe { drop_boxed(tuple_ptr) };
        unsafe { drop_boxed(list_ptr) };
    }

    #[test]
    fn test_hash_unhashable_containers_error() {
        let (list_value, list_ptr) = boxed_value(ListObject::new());
        let list_err = builtin_hash(&[list_value]).unwrap_err();
        assert!(list_err.to_string().contains("unhashable type: 'list'"));
        unsafe { drop_boxed(list_ptr) };

        let (dict_value, dict_ptr) = boxed_value(DictObject::new());
        let dict_err = builtin_hash(&[dict_value]).unwrap_err();
        assert!(dict_err.to_string().contains("unhashable type: 'dict'"));
        unsafe { drop_boxed(dict_ptr) };

        let (set_value, set_ptr) = boxed_value(SetObject::new());
        let set_err = builtin_hash(&[set_value]).unwrap_err();
        assert!(set_err.to_string().contains("unhashable type: 'set'"));
        unsafe { drop_boxed(set_ptr) };
    }

    #[test]
    fn test_callable() {
        let result = builtin_callable(&[Value::int(42).unwrap()]).unwrap();
        assert!(!result.is_truthy());

        let result = builtin_callable(&[Value::none()]).unwrap();
        assert!(!result.is_truthy());
    }

    fn dummy_builtin(_args: &[Value]) -> Result<Value, BuiltinError> {
        Ok(Value::none())
    }

    #[test]
    fn test_callable_builtin_function_true() {
        let builtin = BuiltinFunctionObject::new("dummy".into(), dummy_builtin);
        let (value, ptr) = boxed_value(builtin);
        let result = builtin_callable(&[value]).unwrap();
        assert!(result.is_truthy());
        unsafe { drop_boxed(ptr) };
    }

    #[test]
    fn test_callable_type_object_true() {
        #[repr(C)]
        struct DummyTypeObject {
            header: ObjectHeader,
        }

        let dummy = DummyTypeObject {
            header: ObjectHeader::new(TypeId::TYPE),
        };
        let (value, ptr) = boxed_value(dummy);
        let result = builtin_callable(&[value]).unwrap();
        assert!(result.is_truthy());
        unsafe { drop_boxed(ptr) };
    }

    #[test]
    fn test_callable_non_callable_object_false() {
        let (value, ptr) = boxed_value(ListObject::new());
        let result = builtin_callable(&[value]).unwrap();
        assert!(!result.is_truthy());
        unsafe { drop_boxed(ptr) };
    }

    #[test]
    fn test_callable_string_false() {
        let result = builtin_callable(&[Value::string(intern("name"))]).unwrap();
        assert!(!result.is_truthy());
    }
}
