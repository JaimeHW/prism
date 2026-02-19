//! JIT Executor - High-performance execution of compiled code.
//!
//! The JitExecutor handles:
//! - Invoking compiled machine code from interpreter context
//! - Frame transitions between interpreter and JIT
//! - Deoptimization recovery back to interpreter
//!
//! # Performance Considerations
//!
//! - Zero allocations on hot execution path
//! - Direct function pointer invocation
//! - Inline frame state conversion
//! - Branch-free deopt result checking

use std::sync::Arc;

use prism_core::Value;
use prism_jit::runtime::{CodeCache, CompiledEntry, ExitReason, ReturnAbi};

use crate::error::RuntimeError;
use crate::frame::Frame;

// =============================================================================
// Execution Result
// =============================================================================

/// Result of JIT code execution.
#[derive(Debug)]
pub enum ExecutionResult {
    /// Normal return with value.
    Return(Value),
    /// Deoptimization - resume interpreter at bytecode offset.
    Deopt {
        /// Bytecode offset to resume at.
        bc_offset: u32,
        /// Reason for deoptimization.
        reason: DeoptReason,
    },
    /// Exception occurred.
    Exception(RuntimeError),
    /// Tail call to another function.
    TailCall {
        /// Target function address.
        target: u64,
        /// Number of arguments.
        arg_count: u8,
    },
}

/// Reason for deoptimization.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum DeoptReason {
    /// Type guard failed.
    TypeGuard = 0,
    /// Overflow detected.
    Overflow = 1,
    /// Bounds check failed.
    BoundsCheck = 2,
    /// Inline cache miss.
    CacheMiss = 3,
    /// Unknown opcode.
    UnknownOp = 4,
    /// Division by zero.
    DivByZero = 5,
    /// Stack overflow.
    StackOverflow = 6,
    /// OSR exit.
    OsrExit = 7,
    /// Uncommon trap.
    UncommonTrap = 8,
}

impl DeoptReason {
    /// Convert from raw u8 value.
    #[inline]
    pub fn from_u8(value: u8) -> Option<Self> {
        match value {
            0 => Some(Self::TypeGuard),
            1 => Some(Self::Overflow),
            2 => Some(Self::BoundsCheck),
            3 => Some(Self::CacheMiss),
            4 => Some(Self::UnknownOp),
            5 => Some(Self::DivByZero),
            6 => Some(Self::StackOverflow),
            7 => Some(Self::OsrExit),
            8 => Some(Self::UncommonTrap),
            _ => None,
        }
    }
}

// =============================================================================
// JIT Frame State
// =============================================================================

/// JIT frame state for interpreter ↔ JIT transitions.
///
/// This is a compact representation passed to/from compiled code.
/// Layout matches the x64 calling convention for maximum efficiency.
#[repr(C)]
#[derive(Debug, Clone)]
pub struct JitFrameState {
    /// Base pointer for the JIT frame.
    pub frame_base: *mut u64,
    /// Number of virtual registers.
    pub num_registers: u16,
    /// Current bytecode offset.
    pub bc_offset: u32,
    /// Pointer to constant pool.
    pub const_pool: *const u64,
    /// Pointer to closure environment.
    pub closure_env: *const u64,
    /// Pointer to global scope.
    pub global_scope: *const u64,
}

// SAFETY: JitFrameState contains raw pointers but is only used within
// a single thread during execution.
unsafe impl Send for JitFrameState {}

impl Default for JitFrameState {
    fn default() -> Self {
        Self {
            frame_base: std::ptr::null_mut(),
            num_registers: 0,
            bc_offset: 0,
            const_pool: std::ptr::null(),
            closure_env: std::ptr::null(),
            global_scope: std::ptr::null(),
        }
    }
}

// =============================================================================
// JIT Calling Convention
// =============================================================================

/// Function pointer type for compiled JIT code.
///
/// Calling convention (System V AMD64 ABI):
/// - RDI: Pointer to JitFrameState
/// - Returns: Exit reason in RAX (low byte = reason, high bytes = data)
pub type JitEntryFn = unsafe extern "C" fn(*mut JitFrameState) -> u64;

/// Decode the JIT return value.
#[inline(always)]
fn decode_jit_result(result: u64) -> (ExitReason, u32) {
    let reason = ExitReason::from_u8((result & 0xFF) as u8).unwrap_or(ExitReason::Return);
    let data = (result >> 8) as u32;
    (reason, data)
}

// =============================================================================
// JIT Executor
// =============================================================================

/// Executor for JIT-compiled code.
///
/// Handles invocation of compiled functions and manages the
/// interpreter ↔ JIT frame transitions.
pub struct JitExecutor {
    /// Code cache for looking up compiled functions.
    code_cache: Arc<CodeCache>,
    /// Reusable frame state to avoid allocation.
    frame_state: JitFrameState,
    /// Deopt handler address for recovery.
    _deopt_handler: u64,
}

impl JitExecutor {
    /// Create a new JIT executor.
    pub fn new(code_cache: Arc<CodeCache>) -> Self {
        Self {
            code_cache,
            frame_state: JitFrameState::default(),
            _deopt_handler: 0, // Will be set during initialization
        }
    }

    /// Create executor with deopt handler.
    pub fn with_deopt_handler(code_cache: Arc<CodeCache>, deopt_handler: u64) -> Self {
        Self {
            code_cache,
            frame_state: JitFrameState::default(),
            _deopt_handler: deopt_handler,
        }
    }

    /// Look up compiled code for a function.
    #[inline]
    pub fn lookup(&self, code_id: u64) -> Option<Arc<CompiledEntry>> {
        self.code_cache.lookup(code_id)
    }

    /// Execute compiled code for a function.
    ///
    /// # Arguments
    ///
    /// * `entry` - The compiled code entry
    /// * `frame` - The interpreter frame to execute in
    ///
    /// # Returns
    ///
    /// The execution result (return, deopt, or exception).
    ///
    /// # Safety
    ///
    /// The caller must ensure the compiled code is valid and the frame
    /// is properly initialized.
    pub fn execute(&mut self, entry: &CompiledEntry, frame: &mut Frame) -> ExecutionResult {
        // Setup JIT frame state from interpreter frame
        self.setup_frame_state(frame);

        // Get the entry point
        let entry_fn: JitEntryFn = unsafe { std::mem::transmute(entry.entry_point()) };

        // Execute compiled code
        let result = unsafe { entry_fn(&mut self.frame_state) };

        // Some JIT tiers return raw Prism Value bits in RAX.
        // Handle those paths directly to avoid interpreting valid values as exit reasons.
        if entry.return_abi() == ReturnAbi::RawValueBits {
            self.restore_frame_state(frame);
            return ExecutionResult::Return(Value::from_bits(result));
        }

        // Decode result and update interpreter frame
        let (exit_reason, data) = decode_jit_result(result);

        match exit_reason {
            ExitReason::Return => {
                // Read return value from frame
                let return_value = self.read_return_value();
                self.restore_frame_state(frame);
                ExecutionResult::Return(return_value)
            }
            ExitReason::Deoptimize => {
                // Restore interpreter state and continue in interpreter
                self.restore_frame_state(frame);
                let reason =
                    DeoptReason::from_u8((data & 0xFF) as u8).unwrap_or(DeoptReason::UncommonTrap);
                let bc_offset = (data >> 8) & 0xFFFFFF;
                ExecutionResult::Deopt { bc_offset, reason }
            }
            ExitReason::Exception => {
                self.restore_frame_state(frame);
                ExecutionResult::Exception(RuntimeError::internal("JIT exception occurred"))
            }
            ExitReason::TailCall => {
                let target = self.frame_state.const_pool as u64; // Reused for target
                let arg_count = (data & 0xFF) as u8;
                ExecutionResult::TailCall { target, arg_count }
            }
            ExitReason::OsrExit => {
                self.restore_frame_state(frame);
                ExecutionResult::Deopt {
                    bc_offset: data,
                    reason: DeoptReason::OsrExit,
                }
            }
            ExitReason::StackOverflow => {
                ExecutionResult::Exception(RuntimeError::recursion_error(1000))
            }
        }
    }

    /// Execute at an OSR entry point.
    ///
    /// This is used for on-stack replacement when entering JIT code
    /// mid-execution (e.g., in a hot loop).
    pub fn execute_osr(
        &mut self,
        entry: &CompiledEntry,
        frame: &mut Frame,
        osr_bc_offset: u32,
    ) -> ExecutionResult {
        // Check for OSR entry at this offset
        let osr_entry = match entry.osr_entries() {
            Some(osr) => match osr.lookup_entry(osr_bc_offset) {
                Some(e) => e,
                None => {
                    return ExecutionResult::Deopt {
                        bc_offset: osr_bc_offset,
                        reason: DeoptReason::OsrExit,
                    };
                }
            },
            None => {
                return ExecutionResult::Deopt {
                    bc_offset: osr_bc_offset,
                    reason: DeoptReason::OsrExit,
                };
            }
        };

        // Get the OSR entry offset
        let osr_offset = osr_entry.jit_offset;

        // Setup frame state
        self.setup_frame_state(frame);
        self.frame_state.bc_offset = osr_bc_offset;

        // Calculate OSR entry point
        let osr_entry_fn: JitEntryFn = unsafe {
            let base = entry.code_ptr();
            std::mem::transmute(base.add(osr_offset as usize))
        };

        // Execute from OSR entry
        let result = unsafe { osr_entry_fn(&mut self.frame_state) };

        // Some JIT tiers return raw Prism Value bits in RAX.
        if entry.return_abi() == ReturnAbi::RawValueBits {
            self.restore_frame_state(frame);
            return ExecutionResult::Return(Value::from_bits(result));
        }

        // Handle result same as regular execution
        let (exit_reason, data) = decode_jit_result(result);

        match exit_reason {
            ExitReason::Return => {
                let return_value = self.read_return_value();
                self.restore_frame_state(frame);
                ExecutionResult::Return(return_value)
            }
            ExitReason::Deoptimize => {
                self.restore_frame_state(frame);
                let reason =
                    DeoptReason::from_u8((data & 0xFF) as u8).unwrap_or(DeoptReason::UncommonTrap);
                let bc_offset = (data >> 8) & 0xFFFFFF;
                ExecutionResult::Deopt { bc_offset, reason }
            }
            _ => {
                self.restore_frame_state(frame);
                ExecutionResult::Deopt {
                    bc_offset: osr_bc_offset,
                    reason: DeoptReason::UncommonTrap,
                }
            }
        }
    }

    /// Setup JIT frame state from interpreter frame.
    #[inline]
    fn setup_frame_state(&mut self, frame: &Frame) {
        self.frame_state.frame_base = frame.registers.as_ptr() as *mut u64;
        self.frame_state.num_registers = frame.code.register_count;
        self.frame_state.bc_offset = frame.ip;
        self.frame_state.const_pool = frame.code.constants.as_ptr() as *const u64;
        // TODO: Update JIT closure handling for new cell-based ClosureEnv.
        // The new ClosureEnv uses Arc<Cell> objects for shared mutation semantics.
        // JIT code needs to be updated to handle cell indirection.
        // For now, we pass null and let the JIT deoptimize on closure access.
        self.frame_state.closure_env = std::ptr::null();
        // Global scope would be passed via VM reference
        self.frame_state.global_scope = std::ptr::null();
    }

    /// Restore interpreter frame state from JIT frame.
    #[inline]
    fn restore_frame_state(&self, frame: &mut Frame) {
        frame.ip = self.frame_state.bc_offset;
        // Registers are modified in-place via frame_base pointer
    }

    /// Read the return value from the JIT frame.
    ///
    /// By convention, the return value is in register 0.
    #[inline]
    fn read_return_value(&self) -> Value {
        if self.frame_state.frame_base.is_null() {
            return Value::none();
        }
        unsafe {
            let raw = *self.frame_state.frame_base;
            Value::from_bits(raw)
        }
    }

    /// Get code cache reference.
    #[inline]
    pub fn code_cache(&self) -> &Arc<CodeCache> {
        &self.code_cache
    }
}

// =============================================================================
// Deopt Recovery
// =============================================================================

/// Deoptimization recovery state.
///
/// Contains all information needed to resume interpreter execution
/// after a deoptimization from JIT code.
#[derive(Debug)]
pub struct DeoptRecovery {
    /// Bytecode offset to resume at.
    pub bc_offset: u32,
    /// Reason for deoptimization.
    pub reason: DeoptReason,
}

impl DeoptRecovery {
    /// Create recovery state from execution result.
    pub fn from_result(result: &ExecutionResult) -> Option<Self> {
        match result {
            ExecutionResult::Deopt { bc_offset, reason } => Some(Self {
                bc_offset: *bc_offset,
                reason: *reason,
            }),
            _ => None,
        }
    }

    /// Create recovery with bytecode offset and reason.
    pub fn new(bc_offset: u32, reason: DeoptReason) -> Self {
        Self { bc_offset, reason }
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_deopt_reason_roundtrip() {
        for i in 0..=8 {
            let reason = DeoptReason::from_u8(i).unwrap();
            assert_eq!(reason as u8, i);
        }
        assert!(DeoptReason::from_u8(255).is_none());
    }

    #[test]
    fn test_decode_jit_result() {
        // Normal return
        let (reason, _data) = decode_jit_result(0);
        assert_eq!(reason, ExitReason::Return);

        // Deopt with bc_offset
        let result = 2 | (100 << 8); // Deoptimize, offset 100
        let (reason, data) = decode_jit_result(result);
        assert_eq!(reason, ExitReason::Deoptimize);
        assert_eq!(data, 100);
    }

    #[test]
    fn test_jit_frame_state_default() {
        let state = JitFrameState::default();
        assert!(state.frame_base.is_null());
        assert_eq!(state.num_registers, 0);
        assert_eq!(state.bc_offset, 0);
    }

    #[test]
    fn test_executor_creation() {
        let cache = Arc::new(CodeCache::new(1024 * 1024));
        let executor = JitExecutor::new(cache);
        assert!(executor.code_cache().is_empty());
    }

    #[test]
    fn test_deopt_recovery() {
        let result = ExecutionResult::Deopt {
            bc_offset: 42,
            reason: DeoptReason::TypeGuard,
        };
        let recovery = DeoptRecovery::from_result(&result).unwrap();
        assert_eq!(recovery.bc_offset, 42);
        assert_eq!(recovery.reason, DeoptReason::TypeGuard);
    }
}
