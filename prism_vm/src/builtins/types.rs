//! Type constructor builtins (int, float, str, bool, list, dict, etc.).

use super::BuiltinError;
use prism_core::Value;

/// Builtin int constructor.
pub fn builtin_int(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.is_empty() {
        return Ok(Value::int(0).unwrap());
    }
    if args.len() > 2 {
        return Err(BuiltinError::TypeError(format!(
            "int() takes at most 2 arguments ({} given)",
            args.len()
        )));
    }
    let arg = args[0];
    if let Some(i) = arg.as_int() {
        return Ok(arg);
    }
    if let Some(f) = arg.as_float() {
        return Value::int(f as i64)
            .ok_or_else(|| BuiltinError::OverflowError("int too large".to_string()));
    }
    if let Some(b) = arg.as_bool() {
        return Ok(Value::int(if b { 1 } else { 0 }).unwrap());
    }
    Err(BuiltinError::TypeError(
        "int() argument must be a string or number".to_string(),
    ))
}

/// Builtin float constructor.
pub fn builtin_float(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.is_empty() {
        return Ok(Value::float(0.0));
    }
    if args.len() != 1 {
        return Err(BuiltinError::TypeError(format!(
            "float() takes at most 1 argument ({} given)",
            args.len()
        )));
    }
    let arg = args[0];
    if let Some(f) = arg.as_float() {
        return Ok(arg);
    }
    if let Some(i) = arg.as_int() {
        return Ok(Value::float(i as f64));
    }
    if let Some(b) = arg.as_bool() {
        return Ok(Value::float(if b { 1.0 } else { 0.0 }));
    }
    Err(BuiltinError::TypeError(
        "float() argument must be a string or number".to_string(),
    ))
}

/// Builtin str constructor.
pub fn builtin_str(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.is_empty() {
        // TODO: Return empty string Value
        return Ok(Value::none());
    }
    // TODO: Call __str__ on object
    Ok(Value::none())
}

/// Builtin bool constructor.
pub fn builtin_bool(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.is_empty() {
        return Ok(Value::bool(false));
    }
    if args.len() != 1 {
        return Err(BuiltinError::TypeError(format!(
            "bool() takes at most 1 argument ({} given)",
            args.len()
        )));
    }
    Ok(Value::bool(args[0].is_truthy()))
}

/// Builtin list constructor.
pub fn builtin_list(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() > 1 {
        return Err(BuiltinError::TypeError(format!(
            "list() takes at most 1 argument ({} given)",
            args.len()
        )));
    }
    // TODO: Create ListObject from iterable
    Ok(Value::none())
}

/// Builtin tuple constructor.
pub fn builtin_tuple(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() > 1 {
        return Err(BuiltinError::TypeError(format!(
            "tuple() takes at most 1 argument ({} given)",
            args.len()
        )));
    }
    // TODO: Create TupleObject from iterable
    Ok(Value::none())
}

/// Builtin dict constructor.
pub fn builtin_dict(args: &[Value]) -> Result<Value, BuiltinError> {
    // TODO: Create DictObject from iterable of pairs or kwargs
    let _ = args;
    Ok(Value::none())
}

/// Builtin set constructor.
pub fn builtin_set(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() > 1 {
        return Err(BuiltinError::TypeError(format!(
            "set() takes at most 1 argument ({} given)",
            args.len()
        )));
    }
    // TODO: Create SetObject from iterable
    Ok(Value::none())
}

/// Builtin frozenset constructor.
pub fn builtin_frozenset(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() > 1 {
        return Err(BuiltinError::TypeError(format!(
            "frozenset() takes at most 1 argument ({} given)",
            args.len()
        )));
    }
    // TODO: Create FrozenSetObject
    Err(BuiltinError::NotImplemented(
        "frozenset() not yet implemented".to_string(),
    ))
}

/// Builtin type function.
pub fn builtin_type(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.is_empty() || args.len() > 3 {
        return Err(BuiltinError::TypeError(format!(
            "type() takes 1 or 3 arguments ({} given)",
            args.len()
        )));
    }
    // TODO: Return TypeObject for the value
    Ok(Value::none())
}

/// Builtin isinstance function.
pub fn builtin_isinstance(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() != 2 {
        return Err(BuiltinError::TypeError(format!(
            "isinstance() takes 2 arguments ({} given)",
            args.len()
        )));
    }
    // TODO: Check type hierarchy
    Ok(Value::bool(false))
}

/// Builtin issubclass function.
pub fn builtin_issubclass(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() != 2 {
        return Err(BuiltinError::TypeError(format!(
            "issubclass() takes 2 arguments ({} given)",
            args.len()
        )));
    }
    // TODO: Check class hierarchy
    Ok(Value::bool(false))
}

/// Builtin object constructor.
pub fn builtin_object(args: &[Value]) -> Result<Value, BuiltinError> {
    let _ = args;
    // TODO: Create base object
    Ok(Value::none())
}

/// Builtin getattr(object, name[, default]) function.
///
/// Returns the value of the named attribute of object.
/// If the named attribute does not exist, default is returned if provided,
/// otherwise AttributeError is raised.
///
/// # Python Semantics
/// - `getattr(x, 'name')` → x.name
/// - `getattr(x, 'name', default)` → x.name if exists, else default
/// - Raises AttributeError if attribute not found and no default
pub fn builtin_getattr(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() < 2 || args.len() > 3 {
        return Err(BuiltinError::TypeError(format!(
            "getattr() takes 2 or 3 arguments ({} given)",
            args.len()
        )));
    }

    let obj = args[0];
    let name = args[1];
    let default = args.get(2).copied();

    // Validate name is a string
    let name_str = if let Some(ptr) = name.as_object_ptr() {
        use prism_runtime::object::ObjectHeader;
        use prism_runtime::object::type_obj::TypeId;
        use prism_runtime::types::string::StringObject;

        let type_id = unsafe { (*(ptr as *const ObjectHeader)).type_id };
        if type_id == TypeId::STR {
            let string_obj = unsafe { &*(ptr as *const StringObject) };
            string_obj.as_str().to_string()
        } else {
            return Err(BuiltinError::TypeError(
                "attribute name must be string".to_string(),
            ));
        }
    } else {
        return Err(BuiltinError::TypeError(
            "attribute name must be string".to_string(),
        ));
    };

    // Try to get the attribute
    if let Some(ptr) = obj.as_object_ptr() {
        use prism_runtime::object::ObjectHeader;
        use prism_runtime::object::shaped_object::ShapedObject;
        use prism_runtime::object::type_obj::TypeId;

        let type_id = unsafe { (*(ptr as *const ObjectHeader)).type_id };

        if type_id == TypeId::OBJECT {
            let shaped = unsafe { &*(ptr as *const ShapedObject) };
            if let Some(value) = shaped.get_property(&name_str) {
                return Ok(value);
            }
        }
    }

    // Attribute not found - return default or raise error
    match default {
        Some(d) => Ok(d),
        None => Err(BuiltinError::AttributeError(format!(
            "'{}' object has no attribute '{}'",
            get_type_name(obj),
            name_str
        ))),
    }
}

/// Builtin setattr(object, name, value) function.
///
/// Sets the value of the named attribute of object.
///
/// # Python Semantics
/// - `setattr(x, 'name', value)` → x.name = value
/// - Raises TypeError if the object doesn't support attribute assignment
pub fn builtin_setattr(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() != 3 {
        return Err(BuiltinError::TypeError(format!(
            "setattr() takes 3 arguments ({} given)",
            args.len()
        )));
    }

    let obj = args[0];
    let name = args[1];
    let value = args[2];

    // Validate name is a string
    let name_str = if let Some(ptr) = name.as_object_ptr() {
        use prism_runtime::object::ObjectHeader;
        use prism_runtime::object::type_obj::TypeId;
        use prism_runtime::types::string::StringObject;

        let type_id = unsafe { (*(ptr as *const ObjectHeader)).type_id };
        if type_id == TypeId::STR {
            let string_obj = unsafe { &*(ptr as *const StringObject) };
            string_obj.as_str().to_string()
        } else {
            return Err(BuiltinError::TypeError(
                "attribute name must be string".to_string(),
            ));
        }
    } else {
        return Err(BuiltinError::TypeError(
            "attribute name must be string".to_string(),
        ));
    };

    // Try to set the attribute
    if let Some(ptr) = obj.as_object_ptr() {
        use prism_runtime::object::ObjectHeader;
        use prism_runtime::object::shape::shape_registry;
        use prism_runtime::object::shaped_object::ShapedObject;
        use prism_runtime::object::type_obj::TypeId;

        let type_id = unsafe { (*(ptr as *const ObjectHeader)).type_id };

        if type_id == TypeId::OBJECT {
            let shaped = unsafe { &mut *(ptr as *mut ShapedObject) };
            let registry = shape_registry();
            let interned_name = prism_core::intern::intern(&name_str);
            shaped.set_property(interned_name, value, registry);
            return Ok(Value::none());
        } else {
            return Err(BuiltinError::TypeError(format!(
                "'{}' object has no attribute '{}'",
                type_id.name(),
                name_str
            )));
        }
    }

    // Primitive types don't support setattr
    Err(BuiltinError::TypeError(format!(
        "'{}' object has no attribute '{}'",
        get_type_name(obj),
        name_str
    )))
}

/// Builtin hasattr(object, name) function.
///
/// Returns True if the object has the named attribute, False otherwise.
///
/// # Python Semantics
/// - `hasattr(x, 'name')` → True if x.name exists
/// - Implemented by calling getattr and checking for AttributeError
pub fn builtin_hasattr(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() != 2 {
        return Err(BuiltinError::TypeError(format!(
            "hasattr() takes 2 arguments ({} given)",
            args.len()
        )));
    }

    let obj = args[0];
    let name = args[1];

    // Validate name is a string
    let name_str = if let Some(ptr) = name.as_object_ptr() {
        use prism_runtime::object::ObjectHeader;
        use prism_runtime::object::type_obj::TypeId;
        use prism_runtime::types::string::StringObject;

        let type_id = unsafe { (*(ptr as *const ObjectHeader)).type_id };
        if type_id == TypeId::STR {
            let string_obj = unsafe { &*(ptr as *const StringObject) };
            string_obj.as_str().to_string()
        } else {
            return Err(BuiltinError::TypeError(
                "attribute name must be string".to_string(),
            ));
        }
    } else {
        return Err(BuiltinError::TypeError(
            "attribute name must be string".to_string(),
        ));
    };

    // Check if the attribute exists
    if let Some(ptr) = obj.as_object_ptr() {
        use prism_runtime::object::ObjectHeader;
        use prism_runtime::object::shaped_object::ShapedObject;
        use prism_runtime::object::type_obj::TypeId;

        let type_id = unsafe { (*(ptr as *const ObjectHeader)).type_id };

        if type_id == TypeId::OBJECT {
            let shaped = unsafe { &*(ptr as *const ShapedObject) };
            return Ok(Value::bool(shaped.get_property(&name_str).is_some()));
        }
    }

    // For other types, always return False (no custom attributes)
    Ok(Value::bool(false))
}

/// Builtin delattr(object, name) function.
///
/// Deletes the named attribute from the object.
///
/// # Python Semantics
/// - `delattr(x, 'name')` → del x.name
/// - Raises AttributeError if attribute doesn't exist
pub fn builtin_delattr(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() != 2 {
        return Err(BuiltinError::TypeError(format!(
            "delattr() takes 2 arguments ({} given)",
            args.len()
        )));
    }

    let obj = args[0];
    let name = args[1];

    // Validate name is a string
    let name_str = if let Some(ptr) = name.as_object_ptr() {
        use prism_runtime::object::ObjectHeader;
        use prism_runtime::object::type_obj::TypeId;
        use prism_runtime::types::string::StringObject;

        let type_id = unsafe { (*(ptr as *const ObjectHeader)).type_id };
        if type_id == TypeId::STR {
            let string_obj = unsafe { &*(ptr as *const StringObject) };
            string_obj.as_str().to_string()
        } else {
            return Err(BuiltinError::TypeError(
                "attribute name must be string".to_string(),
            ));
        }
    } else {
        return Err(BuiltinError::TypeError(
            "attribute name must be string".to_string(),
        ));
    };

    // Try to delete the attribute
    if let Some(ptr) = obj.as_object_ptr() {
        use prism_runtime::object::ObjectHeader;
        use prism_runtime::object::shaped_object::ShapedObject;
        use prism_runtime::object::type_obj::TypeId;

        let type_id = unsafe { (*(ptr as *const ObjectHeader)).type_id };

        if type_id == TypeId::OBJECT {
            let shaped = unsafe { &mut *(ptr as *mut ShapedObject) };
            if shaped.delete_property(&name_str) {
                return Ok(Value::none());
            } else {
                return Err(BuiltinError::AttributeError(format!(
                    "'object' object has no attribute '{}'",
                    name_str
                )));
            }
        } else {
            return Err(BuiltinError::TypeError(format!(
                "'{}' object has no attribute '{}'",
                type_id.name(),
                name_str
            )));
        }
    }

    // Primitive types don't support delattr
    Err(BuiltinError::TypeError(format!(
        "'{}' object has no attribute '{}'",
        get_type_name(obj),
        name_str
    )))
}

/// Helper to get the type name of a value.
fn get_type_name(value: Value) -> &'static str {
    if value.is_none() {
        "NoneType"
    } else if value.is_bool() {
        "bool"
    } else if value.is_int() {
        "int"
    } else if value.is_float() {
        "float"
    } else if value.is_string() {
        "str"
    } else if let Some(ptr) = value.as_object_ptr() {
        use prism_runtime::object::ObjectHeader;
        let type_id = unsafe { (*(ptr as *const ObjectHeader)).type_id };
        type_id.name()
    } else {
        "unknown"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_int_from_int() {
        let result = builtin_int(&[Value::int(42).unwrap()]).unwrap();
        assert_eq!(result.as_int(), Some(42));
    }

    #[test]
    fn test_int_from_float() {
        let result = builtin_int(&[Value::float(3.9)]).unwrap();
        assert_eq!(result.as_int(), Some(3));
    }

    #[test]
    fn test_float_from_int() {
        let result = builtin_float(&[Value::int(42).unwrap()]).unwrap();
        assert_eq!(result.as_float(), Some(42.0));
    }

    #[test]
    fn test_bool_truthy() {
        let result = builtin_bool(&[Value::int(1).unwrap()]).unwrap();
        assert!(result.is_truthy());

        let result = builtin_bool(&[Value::int(0).unwrap()]).unwrap();
        assert!(!result.is_truthy());
    }
}
