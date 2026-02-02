//! Load and store opcode handlers.
//!
//! Handles loading constants, locals, globals, closures, and register moves.

use crate::VirtualMachine;
use crate::dispatch::ControlFlow;
use prism_compiler::bytecode::Instruction;
use prism_core::Value;

// =============================================================================
// Constants
// =============================================================================

/// LoadConst: dst = constants[imm16]
#[inline(always)]
pub fn load_const(vm: &mut VirtualMachine, inst: Instruction) -> ControlFlow {
    let frame = vm.current_frame_mut();
    let value = frame.get_const(inst.imm16());
    frame.set_reg(inst.dst().0, value);
    ControlFlow::Continue
}

/// LoadNone: dst = None
#[inline(always)]
pub fn load_none(vm: &mut VirtualMachine, inst: Instruction) -> ControlFlow {
    vm.current_frame_mut().set_reg(inst.dst().0, Value::none());
    ControlFlow::Continue
}

/// LoadTrue: dst = True
#[inline(always)]
pub fn load_true(vm: &mut VirtualMachine, inst: Instruction) -> ControlFlow {
    vm.current_frame_mut()
        .set_reg(inst.dst().0, Value::bool(true));
    ControlFlow::Continue
}

/// LoadFalse: dst = False
#[inline(always)]
pub fn load_false(vm: &mut VirtualMachine, inst: Instruction) -> ControlFlow {
    vm.current_frame_mut()
        .set_reg(inst.dst().0, Value::bool(false));
    ControlFlow::Continue
}

// =============================================================================
// Locals
// =============================================================================

/// LoadLocal: dst = frame.registers[imm16]
/// Note: In our register-based VM, locals ARE registers. This opcode
/// may be used for explicit local variable access semantics.
#[inline(always)]
pub fn load_local(vm: &mut VirtualMachine, inst: Instruction) -> ControlFlow {
    let frame = vm.current_frame_mut();
    let slot = inst.imm16() as u8;
    let value = frame.get_reg(slot);
    frame.set_reg(inst.dst().0, value);
    ControlFlow::Continue
}

/// StoreLocal: frame.registers[imm16] = src1
#[inline(always)]
pub fn store_local(vm: &mut VirtualMachine, inst: Instruction) -> ControlFlow {
    let frame = vm.current_frame_mut();
    let value = frame.get_reg(inst.src1().0);
    let slot = inst.imm16() as u8;
    frame.set_reg(slot, value);
    ControlFlow::Continue
}

/// DeleteLocal: frame.registers[imm16] = undefined
#[inline(always)]
pub fn delete_local(vm: &mut VirtualMachine, inst: Instruction) -> ControlFlow {
    let frame = vm.current_frame_mut();
    let slot = inst.imm16() as u8;
    frame.set_reg(slot, Value::none()); // Mark as unbound
    ControlFlow::Continue
}

// =============================================================================
// Globals
// =============================================================================

/// LoadGlobal: dst = globals[names[imm16]]
#[inline(always)]
pub fn load_global(vm: &mut VirtualMachine, inst: Instruction) -> ControlFlow {
    let frame = vm.current_frame();
    let name = frame.get_name(inst.imm16()).clone();

    match vm.globals.get_arc(&name) {
        Some(value) => {
            vm.current_frame_mut().set_reg(inst.dst().0, value);
            ControlFlow::Continue
        }
        None => {
            // Check builtins
            match vm.builtins.get(&name) {
                Some(value) => {
                    vm.current_frame_mut().set_reg(inst.dst().0, value);
                    ControlFlow::Continue
                }
                None => ControlFlow::Error(crate::error::RuntimeError::name_error(name)),
            }
        }
    }
}

/// StoreGlobal: globals[names[imm16]] = src1
#[inline(always)]
pub fn store_global(vm: &mut VirtualMachine, inst: Instruction) -> ControlFlow {
    let frame = vm.current_frame();
    let name = frame.get_name(inst.imm16()).clone();
    let value = frame.get_reg(inst.src1().0);

    vm.globals.set(name, value);
    ControlFlow::Continue
}

/// DeleteGlobal: del globals[names[imm16]]
#[inline(always)]
pub fn delete_global(vm: &mut VirtualMachine, inst: Instruction) -> ControlFlow {
    let frame = vm.current_frame();
    let name = frame.get_name(inst.imm16()).clone();

    match vm.globals.delete(&name) {
        Some(_) => ControlFlow::Continue,
        None => ControlFlow::Error(crate::error::RuntimeError::name_error(name)),
    }
}

// =============================================================================
// Closures
// =============================================================================

/// LoadClosure: dst = closure[imm16].get()
///
/// Loads the value from a cell in the closure environment.
/// Returns an error if the cell is unbound (UnboundLocalError).
#[inline(always)]
pub fn load_closure(vm: &mut VirtualMachine, inst: Instruction) -> ControlFlow {
    let frame = vm.current_frame();

    match &frame.closure {
        Some(env) => {
            let idx = inst.imm16() as usize;
            let cell = env.get_cell(idx);
            match cell.get() {
                Some(value) => {
                    vm.current_frame_mut().set_reg(inst.dst().0, value);
                    ControlFlow::Continue
                }
                None => {
                    // Cell is unbound - variable was deleted or never assigned
                    ControlFlow::Error(crate::error::RuntimeError::unbound_local_cell(idx))
                }
            }
        }
        None => ControlFlow::Error(crate::error::RuntimeError::internal(
            "LoadClosure without closure environment",
        )),
    }
}

/// StoreClosure: closure[imm16].set(src1)
///
/// Stores a value into a cell in the closure environment.
/// The cell uses interior mutability (atomic operations), so this works
/// despite the Arc<ClosureEnv> being immutable.
#[inline(always)]
pub fn store_closure(vm: &mut VirtualMachine, inst: Instruction) -> ControlFlow {
    let frame = vm.current_frame();
    let value = frame.get_reg(inst.src1().0);
    let idx = inst.imm16() as usize;

    match &frame.closure {
        Some(env) => {
            env.set(idx, value);
            ControlFlow::Continue
        }
        None => ControlFlow::Error(crate::error::RuntimeError::internal(
            "StoreClosure without closure environment",
        )),
    }
}

/// DeleteClosure: del closure[imm16]
///
/// Clears a cell in the closure environment, making it unbound.
/// Subsequent reads will raise UnboundLocalError.
#[inline(always)]
pub fn delete_closure(vm: &mut VirtualMachine, inst: Instruction) -> ControlFlow {
    let frame = vm.current_frame();
    let idx = inst.imm16() as usize;

    match &frame.closure {
        Some(env) => {
            env.get_cell(idx).clear();
            ControlFlow::Continue
        }
        None => ControlFlow::Error(crate::error::RuntimeError::internal(
            "DeleteClosure without closure environment",
        )),
    }
}

// =============================================================================
// Move
// =============================================================================

/// Move: dst = src1
#[inline(always)]
pub fn move_reg(vm: &mut VirtualMachine, inst: Instruction) -> ControlFlow {
    let frame = vm.current_frame_mut();
    let value = frame.get_reg(inst.src1().0);
    frame.set_reg(inst.dst().0, value);
    ControlFlow::Continue
}

#[cfg(test)]
mod tests {
    use super::*;
    use prism_compiler::bytecode::{Instruction, Opcode, Register};
    use prism_core::Value;
    use prism_runtime::types::Cell;
    use std::sync::Arc;

    // ==========================================================================
    // Cell Integration Tests
    // ==========================================================================

    #[test]
    fn test_cell_basic_get_set() {
        let cell = Cell::new(Value::int(42).unwrap());
        assert_eq!(cell.get(), Some(Value::int(42).unwrap()));

        cell.set(Value::int(100).unwrap());
        assert_eq!(cell.get(), Some(Value::int(100).unwrap()));
    }

    #[test]
    fn test_cell_unbound() {
        let cell = Cell::unbound();
        assert!(cell.is_empty());
        assert_eq!(cell.get(), None);
    }

    #[test]
    fn test_cell_clear_makes_unbound() {
        let cell = Cell::new(Value::int(42).unwrap());
        assert!(!cell.is_empty());

        cell.clear();
        assert!(cell.is_empty());
        assert_eq!(cell.get(), None);
    }

    #[test]
    fn test_cell_shared_mutation() {
        // Simulate two closures sharing the same cell
        let cell = Arc::new(Cell::new(Value::int(1).unwrap()));
        let cell_clone = Arc::clone(&cell);

        // Both see initial value
        assert_eq!(cell.get(), Some(Value::int(1).unwrap()));
        assert_eq!(cell_clone.get(), Some(Value::int(1).unwrap()));

        // Mutate through one reference
        cell.set(Value::int(2).unwrap());

        // Both see updated value
        assert_eq!(cell.get(), Some(Value::int(2).unwrap()));
        assert_eq!(cell_clone.get(), Some(Value::int(2).unwrap()));
    }

    #[test]
    fn test_cell_none_value_is_not_unbound() {
        // Python None is different from unbound (deleted)
        let cell = Cell::new(Value::none());
        assert!(!cell.is_empty());
        assert_eq!(cell.get(), Some(Value::none()));
    }

    #[test]
    fn test_cell_atomic_thread_safety() {
        use std::thread;

        let cell = Arc::new(Cell::new(Value::int(0).unwrap()));
        let mut handles = vec![];

        // Spawn multiple threads that read and write
        for i in 0..4 {
            let cell_clone = Arc::clone(&cell);
            handles.push(thread::spawn(move || {
                for _ in 0..100 {
                    cell_clone.set(Value::int(i).unwrap());
                    let _ = cell_clone.get();
                }
            }));
        }

        for handle in handles {
            handle.join().unwrap();
        }

        // Should not crash, and should have some valid value
        let final_val = cell.get();
        assert!(final_val.is_some());
    }

    // ==========================================================================
    // ClosureEnv Integration Tests
    // ==========================================================================

    #[test]
    fn test_closure_env_inline_storage() {
        use crate::frame::ClosureEnv;

        let cells = vec![
            Arc::new(Cell::new(Value::int(1).unwrap())),
            Arc::new(Cell::new(Value::int(2).unwrap())),
        ];

        let env = ClosureEnv::new(cells);
        assert_eq!(env.len(), 2);
        assert!(env.is_inline()); // <= 4 cells use inline storage

        assert_eq!(env.get(0), Value::int(1).unwrap());
        assert_eq!(env.get(1), Value::int(2).unwrap());
    }

    #[test]
    fn test_closure_env_overflow_storage() {
        use crate::frame::ClosureEnv;

        let cells: Vec<_> = (0..6)
            .map(|i| Arc::new(Cell::new(Value::int(i).unwrap())))
            .collect();

        let env = ClosureEnv::new(cells);
        assert_eq!(env.len(), 6);
        assert!(!env.is_inline()); // > 4 cells use overflow

        for i in 0..6 {
            assert_eq!(env.get(i), Value::int(i as i64).unwrap());
        }
    }

    #[test]
    fn test_closure_env_mutation() {
        use crate::frame::ClosureEnv;

        let cells = vec![Arc::new(Cell::new(Value::int(0).unwrap()))];
        let env = ClosureEnv::new(cells);

        assert_eq!(env.get(0), Value::int(0).unwrap());
        env.set(0, Value::int(42).unwrap());
        assert_eq!(env.get(0), Value::int(42).unwrap());
    }

    #[test]
    fn test_closure_env_get_cell_for_unbound_check() {
        use crate::frame::ClosureEnv;

        let cells = vec![Arc::new(Cell::unbound())];
        let env = ClosureEnv::new(cells);

        // get_cell returns the cell, allowing unbound check
        let cell = env.get_cell(0);
        assert!(cell.is_empty());
        assert_eq!(cell.get(), None);
    }

    // ==========================================================================
    // Error Type Tests
    // ==========================================================================

    #[test]
    fn test_unbound_local_cell_error_message() {
        let err = crate::error::RuntimeError::unbound_local_cell(3);
        let msg = err.to_string();
        assert!(msg.contains("UnboundLocalError"));
        assert!(msg.contains("cell 3"));
    }

    #[test]
    fn test_unbound_local_named_error_message() {
        let err = crate::error::RuntimeError::unbound_local("x");
        let msg = err.to_string();
        assert!(msg.contains("UnboundLocalError"));
        assert!(msg.contains("'x'"));
    }

    // ==========================================================================
    // Instruction Format Tests
    // ==========================================================================

    #[test]
    fn test_load_closure_instruction_format() {
        let inst = Instruction::op_di(Opcode::LoadClosure, Register::new(0), 5);
        assert_eq!(inst.opcode(), Opcode::LoadClosure as u8);
        assert_eq!(inst.dst().0, 0);
        assert_eq!(inst.imm16(), 5);
    }

    #[test]
    fn test_store_closure_instruction_format() {
        let inst = Instruction::op_di(Opcode::StoreClosure, Register::new(3), 7);
        assert_eq!(inst.opcode(), Opcode::StoreClosure as u8);
        assert_eq!(inst.dst().0, 3);
        assert_eq!(inst.imm16(), 7);
    }

    #[test]
    fn test_delete_closure_instruction_format() {
        let inst = Instruction::op_di(Opcode::DeleteClosure, Register::new(0), 2);
        assert_eq!(inst.opcode(), Opcode::DeleteClosure as u8);
        assert_eq!(inst.imm16(), 2);
    }

    // ==========================================================================
    // Opcode Registration Tests
    // ==========================================================================

    #[test]
    fn test_delete_closure_opcode_value() {
        assert_eq!(Opcode::DeleteClosure as u8, 0x1D);
    }

    #[test]
    fn test_closure_opcodes_in_load_store_range() {
        // All closure opcodes should be in the 0x10-0x1F range
        assert!(Opcode::LoadClosure as u8 >= 0x10);
        assert!(Opcode::LoadClosure as u8 <= 0x1F);
        assert!(Opcode::StoreClosure as u8 >= 0x10);
        assert!(Opcode::StoreClosure as u8 <= 0x1F);
        assert!(Opcode::DeleteClosure as u8 >= 0x10);
        assert!(Opcode::DeleteClosure as u8 <= 0x1F);
    }

    #[test]
    fn test_opcode_from_u8_roundtrip() {
        let opcodes = [
            Opcode::LoadClosure,
            Opcode::StoreClosure,
            Opcode::DeleteClosure,
        ];

        for op in opcodes {
            let byte = op as u8;
            let recovered = Opcode::from_u8(byte);
            assert_eq!(recovered, Some(op));
        }
    }
}
