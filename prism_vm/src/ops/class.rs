//! Class operation opcode handlers.
//!
//! Implements the BUILD_CLASS opcode for Python class creation.
//!
//! # Python Class Creation Protocol
//!
//! When Python executes a class statement like:
//! ```python
//! class MyClass(Base1, Base2):
//!     x = 1
//!     def method(self):
//!         pass
//! ```
//!
//! The compiler generates:
//! 1. Evaluate base classes into registers
//! 2. Create class body CodeObject
//! 3. Execute BUILD_CLASS opcode
//!
//! BUILD_CLASS then:
//! 1. Executes the class body CodeObject to populate namespace dict
//! 2. Calls the metaclass (default: `type`) with (name, bases, namespace)
//! 3. Returns the new class object
//!
//! # Performance Notes
//!
//! - Class creation is not on the hot path - happens once per class definition
//! - Focus is on correctness and CPython compatibility
//! - Future optimization: cache class objects for repeated module imports

use crate::VirtualMachine;
use crate::dispatch::ControlFlow;
use crate::error::RuntimeError;
use prism_compiler::bytecode::Instruction;
use prism_core::Value;
use prism_core::intern::{InternedString, intern, interned_by_ptr};
use prism_runtime::object::class::{ClassDict, PyClassObject};
use prism_runtime::object::mro::ClassId;
use prism_runtime::object::type_obj::TypeId;
use prism_runtime::types::string::StringObject;
use std::sync::Arc;

// =============================================================================
// BUILD_CLASS Handler
// =============================================================================

/// BuildClass: Create a new class from bases and body code object.
///
/// # Opcode Format (DstSrcSrc)
/// - dst: Destination register for the new class object
/// - src1: Register containing class name string
/// - src2: Number of base classes
/// - imm16: Constant pool index of the class body CodeObject
///
/// # Register Layout
/// ```text
/// r[dst]     <- result (new class object)
/// r[dst+1]   <- base class 0
/// r[dst+2]   <- base class 1
/// ...
/// r[dst+n]   <- base class n-1
/// ```
///
/// # Algorithm
/// 1. Load class body CodeObject from constants
/// 2. Execute body to create class namespace (attribute dict)
/// 3. Collect base classes from registers
/// 4. Create PyClassObject with name, bases, namespace
/// 5. Store result in destination register
///
/// # Error Conditions
/// - Invalid constant index
/// - Non-CodeObject in constant slot
/// - Invalid base class type
/// - MRO computation failure (diamond inheritance conflicts)
#[inline(always)]
pub fn build_class(vm: &mut VirtualMachine, inst: Instruction) -> ControlFlow {
    let dst_reg = inst.dst().0;
    let name_reg = inst.src1().0;
    let base_count = inst.src2().0 as usize;

    // Get class name from register
    let name_val = vm.current_frame().get_reg(name_reg);
    let class_name = match extract_string_name(name_val) {
        Some(name) => name,
        None => {
            return ControlFlow::Error(RuntimeError::type_error("class name must be a string"));
        }
    };

    // Get code object index from instruction (encoded after base count)
    // The code object index is stored in the upper bits or as immediate
    let code_idx = inst.imm16();

    // Load class body CodeObject from constants
    let frame = vm.current_frame();
    let code_const = frame.get_const(code_idx);

    // Collect base classes from registers
    let mut base_class_ids = Vec::with_capacity(base_count);
    for i in 0..base_count {
        let base_val = frame.get_reg(dst_reg + 1 + i as u8);
        match extract_class_id(base_val) {
            Some(class_id) => base_class_ids.push(class_id),
            None => {
                // If no bases provided, inherit from object
                if base_count == 0 {
                    break;
                }
                return ControlFlow::Error(RuntimeError::type_error(format!(
                    "base class {} is not a valid class",
                    i
                )));
            }
        }
    }

    // Create class namespace
    // In a full implementation, we would:
    // 1. Push a new frame with the class body CodeObject
    // 2. Execute the class body
    // 3. Collect locals as the class namespace
    //
    // For now, create an empty namespace (class body already compiled attributes)
    let namespace = ClassDict::new();

    // Create the class object
    // For simple classes (no complex inheritance), use new_simple
    let class = if base_class_ids.is_empty() {
        // No explicit bases - inherit from object implicitly
        PyClassObject::new_simple(class_name)
    } else {
        // Has bases - need MRO computation
        // For now, use simple path and improve later with proper MRO lookup
        match PyClassObject::new(class_name, &base_class_ids, |_| None) {
            Ok(class) => class,
            Err(e) => {
                return ControlFlow::Error(RuntimeError::type_error(format!(
                    "cannot create class: MRO error - {}",
                    e
                )));
            }
        }
    };

    // Box and convert to Value
    let class = Arc::new(class);
    let class_ptr = Arc::into_raw(class) as *const ();
    let class_val = Value::object_ptr(class_ptr);

    // Store result in destination register
    vm.current_frame_mut().set_reg(dst_reg, class_val);

    // For classes defined at module level, also store to globals
    // The compiler should emit a StoreGlobal after BuildClass if needed

    ControlFlow::Continue
}

// =============================================================================
// Helper Functions
// =============================================================================

/// Extract a string name from a Value.
///
/// Supports both interned strings (small) and heap-allocated strings.
#[inline]
fn extract_string_name(val: Value) -> Option<InternedString> {
    if val.is_string() {
        let ptr = val.as_string_object_ptr()?;
        return interned_by_ptr(ptr as *const u8);
    }

    let ptr = val.as_object_ptr()?;
    if extract_type_id(ptr) != TypeId::STR {
        return None;
    }

    let string = unsafe { &*(ptr as *const StringObject) };
    Some(intern(string.as_str()))
}

/// Extract ClassId from a class object Value.
#[inline]
fn extract_class_id(val: Value) -> Option<ClassId> {
    if let Some(ptr) = val.as_object_ptr() {
        // Check if this is a PyClassObject
        let type_id = extract_type_id(ptr);
        if type_id == TypeId::TYPE {
            // This is a class object - extract its class_id
            // SAFETY: We verified type_id is TYPE, so ptr points to PyClassObject
            let class_obj = unsafe { &*(ptr as *const PyClassObject) };
            return Some(class_obj.class_id());
        }
    }
    None
}

/// Extract TypeId from an object pointer.
///
/// # Safety
/// Pointer must point to a valid object with ObjectHeader at offset 0.
#[inline(always)]
fn extract_type_id(ptr: *const ()) -> TypeId {
    use prism_runtime::object::ObjectHeader;
    let header_ptr = ptr as *const ObjectHeader;
    // SAFETY: Caller guarantees ptr points to valid object with ObjectHeader at offset 0
    unsafe { (*header_ptr).type_id }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use prism_core::intern::intern;
    use prism_runtime::object::class::PyClassObject;

    // =========================================================================
    // Unit Tests for Helper Functions
    // =========================================================================

    #[test]
    fn test_class_object_has_type_id() {
        use prism_runtime::object::PyObject;
        use prism_runtime::object::type_obj::TypeId;

        // Create a class object and verify its header has TYPE type_id via the trait
        let class = PyClassObject::new_simple(intern("TestClass"));

        // Use the PyObject trait to get the type_id (this is the safe way)
        assert_eq!(
            class.header().type_id,
            TypeId::TYPE,
            "PyClassObject header should have TypeId::TYPE"
        );
    }

    #[test]
    fn test_class_id_extraction_direct() {
        // Create a class and verify we can get its class_id through the object
        let class = PyClassObject::new_simple(intern("DirectTestClass"));
        let expected_id = class.class_id();
        let class = Arc::new(class);
        let class_ptr = Arc::as_ptr(&class) as *const PyClassObject;

        // Directly read from the object
        let class_ref = unsafe { &*class_ptr };
        assert_eq!(class_ref.class_id(), expected_id);
    }

    #[test]
    fn test_extract_class_id_returns_none_for_non_objects() {
        // Non-object values should return None
        assert_eq!(extract_class_id(Value::none()), None);
        assert_eq!(extract_class_id(Value::bool(true)), None);
        assert_eq!(extract_class_id(Value::int_unchecked(42)), None);
    }

    #[test]
    fn test_extract_string_name_tagged() {
        let name = extract_string_name(Value::string(intern("TaggedClass")));
        assert_eq!(name.unwrap().as_ref(), "TaggedClass");
    }

    #[test]
    fn test_extract_string_name_heap_string() {
        let ptr = Box::into_raw(Box::new(StringObject::new("HeapClass")));
        let value = Value::object_ptr(ptr as *const ());
        let name = extract_string_name(value);
        assert_eq!(name.unwrap().as_ref(), "HeapClass");
        unsafe {
            drop(Box::from_raw(ptr));
        }
    }

    #[test]
    fn test_extract_string_name_invalid_returns_none() {
        let name = extract_string_name(Value::none());
        assert!(name.is_none());
    }

    // =========================================================================
    // Class Creation Tests
    // =========================================================================

    #[test]
    fn test_simple_class_creation() {
        let class = PyClassObject::new_simple(intern("SimpleClass"));
        assert_eq!(class.name().as_ref(), "SimpleClass");
        assert!(class.bases().is_empty());
        assert_eq!(class.mro().len(), 2); // [SimpleClass, object]
    }

    #[test]
    fn test_class_with_single_base() {
        // Create parent class
        let parent = Arc::new(PyClassObject::new_simple(intern("Parent")));
        let parent_id = parent.class_id();
        let parent_mro = parent.mro().to_vec();

        // Create child with parent as base
        let child = PyClassObject::new(intern("Child"), &[parent_id], |id| {
            if id == parent_id {
                Some(parent_mro.clone().into())
            } else {
                None
            }
        });

        assert!(child.is_ok());
        let child = child.unwrap();
        assert_eq!(child.bases().len(), 1);
        assert!(child.bases().contains(&parent_id));
    }

    #[test]
    fn test_class_type_id_uniqueness() {
        let class1 = PyClassObject::new_simple(intern("Class1"));
        let class2 = PyClassObject::new_simple(intern("Class2"));

        assert_ne!(class1.class_type_id(), class2.class_type_id());
        assert_ne!(class1.class_id(), class2.class_id());
    }

    #[test]
    fn test_class_dict_operations() {
        let namespace = ClassDict::new();

        // Initially empty
        assert!(namespace.is_empty());

        // Add attribute
        let attr_name = intern("my_method");
        namespace.set(attr_name.clone(), Value::int_unchecked(42));

        // Check attribute
        assert!(!namespace.is_empty());
        assert!(namespace.contains(&attr_name));
        assert_eq!(namespace.get(&attr_name), Some(Value::int_unchecked(42)));

        // Delete attribute
        let deleted = namespace.delete(&attr_name);
        assert_eq!(deleted, Some(Value::int_unchecked(42)));
        assert!(namespace.is_empty());
    }

    #[test]
    fn test_class_dict_multiple_attributes() {
        let namespace = ClassDict::new();

        // Add multiple attributes
        for i in 0..10 {
            let name = intern(&format!("attr_{}", i));
            namespace.set(name, Value::int_unchecked(i as i64));
        }

        assert_eq!(namespace.len(), 10);

        // Verify all attributes
        for i in 0..10 {
            let name = intern(&format!("attr_{}", i));
            assert!(namespace.contains(&name));
            assert_eq!(namespace.get(&name), Some(Value::int_unchecked(i as i64)));
        }
    }

    #[test]
    fn test_class_inherits_from_object() {
        let class = PyClassObject::new_simple(intern("Derived"));

        // MRO should end with object (ClassId::OBJECT)
        let mro = class.mro();
        assert_eq!(mro.len(), 2);
        assert_eq!(mro[0], class.class_id()); // Self first
        assert_eq!(mro[1], ClassId::OBJECT); // Object last
    }

    #[test]
    fn test_class_attribute_setting() {
        let class = PyClassObject::new_simple(intern("AttrTest"));

        // Set some attributes
        class.set_attr(intern("x"), Value::int_unchecked(10));
        class.set_attr(intern("y"), Value::int_unchecked(20));

        // Verify
        assert!(class.has_attr(&intern("x")));
        assert!(class.has_attr(&intern("y")));
        assert_eq!(class.get_attr(&intern("x")), Some(Value::int_unchecked(10)));
        assert_eq!(class.get_attr(&intern("y")), Some(Value::int_unchecked(20)));
    }

    #[test]
    fn test_class_flags_default() {
        let class = PyClassObject::new_simple(intern("FlagsTest"));

        assert!(!class.is_initialized());
        assert!(!class.has_slots());
        assert!(!class.is_final());
        assert!(!class.has_custom_new());
        assert!(!class.has_custom_init());
    }

    #[test]
    fn test_class_flags_modification() {
        let mut class = PyClassObject::new_simple(intern("FlagsModTest"));

        // Modify flags
        class.mark_initialized();
        class.mark_final();
        class.mark_has_init();

        assert!(class.is_initialized());
        assert!(class.is_final());
        assert!(class.has_custom_init());
    }

    #[test]
    fn test_class_slots_definition() {
        let mut class = PyClassObject::new_simple(intern("SlottedClass"));

        // Define __slots__
        let slots = vec![intern("x"), intern("y"), intern("z")];
        class.set_slots(slots);

        assert!(class.has_slots());
        assert_eq!(class.slot_names().unwrap().len(), 3);
    }

    #[test]
    fn test_instantiation_hint_no_slots() {
        let class = PyClassObject::new_simple(intern("NoSlots"));

        // Without __init__, hint is DefaultInit
        use prism_runtime::object::class::InstantiationHint;
        assert_eq!(class.instantiation_hint(), InstantiationHint::DefaultInit);
    }

    #[test]
    fn test_instantiation_hint_with_init() {
        let mut class = PyClassObject::new_simple(intern("WithInit"));
        class.mark_has_init();

        use prism_runtime::object::class::InstantiationHint;
        assert_eq!(class.instantiation_hint(), InstantiationHint::Generic);
    }

    #[test]
    fn test_instantiation_hint_inline_slots() {
        let mut class = PyClassObject::new_simple(intern("InlineSlots"));
        class.set_slots(vec![intern("x"), intern("y")]); // 2 slots, fits inline

        use prism_runtime::object::class::InstantiationHint;
        assert_eq!(class.instantiation_hint(), InstantiationHint::InlineSlots);
    }

    #[test]
    fn test_instantiation_hint_fixed_slots() {
        let mut class = PyClassObject::new_simple(intern("FixedSlots"));
        // More than 4 slots - needs fixed allocation
        class.set_slots(vec![
            intern("a"),
            intern("b"),
            intern("c"),
            intern("d"),
            intern("e"),
            intern("f"),
        ]);

        use prism_runtime::object::class::InstantiationHint;
        assert_eq!(class.instantiation_hint(), InstantiationHint::FixedSlots);
    }

    // =========================================================================
    // Multiple Inheritance Tests
    // =========================================================================

    #[test]
    fn test_diamond_inheritance_mro() {
        use std::collections::HashMap;

        // Diamond: D(B, C) where B(A) and C(A)
        // Create A
        let a = Arc::new(PyClassObject::new_simple(intern("A")));
        let a_id = a.class_id();
        let a_mro = a.mro().to_vec();

        let mut registry: HashMap<ClassId, Vec<ClassId>> = HashMap::new();
        registry.insert(a_id, a_mro.clone());

        // Create B(A)
        let b = Arc::new(
            PyClassObject::new(intern("B"), &[a_id], |id| {
                registry.get(&id).cloned().map(|v| v.into())
            })
            .unwrap(),
        );
        let b_id = b.class_id();
        let b_mro = b.mro().to_vec();
        registry.insert(b_id, b_mro.clone());

        // Create C(A)
        let c = Arc::new(
            PyClassObject::new(intern("C"), &[a_id], |id| {
                registry.get(&id).cloned().map(|v| v.into())
            })
            .unwrap(),
        );
        let c_id = c.class_id();
        let c_mro = c.mro().to_vec();
        registry.insert(c_id, c_mro.clone());

        // Create D(B, C)
        let d = PyClassObject::new(intern("D"), &[b_id, c_id], |id| {
            registry.get(&id).cloned().map(|v| v.into())
        })
        .unwrap();

        // D's MRO should be [D, B, C, A, object]
        let d_mro = d.mro();
        assert_eq!(d_mro.len(), 5);
        assert_eq!(d_mro[0], d.class_id()); // D
        assert_eq!(d_mro[1], b_id); // B
        assert_eq!(d_mro[2], c_id); // C
        assert_eq!(d_mro[3], a_id); // A
        assert_eq!(d_mro[4], ClassId::OBJECT); // object
    }

    // =========================================================================
    // Class Value Conversion Tests
    // =========================================================================

    #[test]
    fn test_class_to_value_roundtrip() {
        let original = PyClassObject::new_simple(intern("Roundtrip"));
        let original_id = original.class_type_id();
        let original_name = original.name().clone();

        // Convert to Arc and then to Value
        let class = Arc::new(original);
        let class_ptr = Arc::into_raw(class) as *const ();
        let class_val = Value::object_ptr(class_ptr);

        // Should be able to check it's an object
        assert!(class_val.as_object_ptr().is_some());

        // Extract and verify
        let extracted_ptr = class_val.as_object_ptr().unwrap();
        let extracted_class = unsafe { &*(extracted_ptr as *const PyClassObject) };

        assert_eq!(extracted_class.class_type_id(), original_id);
        assert_eq!(extracted_class.name(), &original_name);

        // Clean up
        unsafe { Arc::from_raw(extracted_ptr as *const PyClassObject) };
    }

    // =========================================================================
    // Thread Safety Tests
    // =========================================================================

    #[test]
    fn test_class_dict_concurrent_access() {
        use std::sync::Arc;
        use std::thread;

        let namespace = Arc::new(ClassDict::new());

        let handles: Vec<_> = (0..4)
            .map(|i| {
                let ns = namespace.clone();
                thread::spawn(move || {
                    // Each thread sets its own attribute
                    let name = intern(&format!("thread_attr_{}", i));
                    ns.set(name.clone(), Value::int_unchecked(i as i64));

                    // Verify it was set
                    assert!(ns.contains(&name));
                })
            })
            .collect();

        for handle in handles {
            handle.join().unwrap();
        }

        // All 4 attributes should exist
        assert_eq!(namespace.len(), 4);
    }

    #[test]
    fn test_class_object_concurrent_reads() {
        use std::sync::Arc;
        use std::thread;

        // Create class with some attributes
        let class = PyClassObject::new_simple(intern("ConcurrentRead"));
        class.set_attr(intern("x"), Value::int_unchecked(100));
        class.set_attr(intern("y"), Value::int_unchecked(200));
        let class = Arc::new(class);

        // Multiple threads reading simultaneously
        let handles: Vec<_> = (0..8)
            .map(|_| {
                let c = class.clone();
                thread::spawn(move || {
                    for _ in 0..100 {
                        let x = c.get_attr(&intern("x"));
                        let y = c.get_attr(&intern("y"));
                        assert_eq!(x, Some(Value::int_unchecked(100)));
                        assert_eq!(y, Some(Value::int_unchecked(200)));
                    }
                })
            })
            .collect();

        for handle in handles {
            handle.join().unwrap();
        }
    }
}
