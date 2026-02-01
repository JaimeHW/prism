//! Virtual machine implementation.
//!
//! The VirtualMachine is the main execution engine for Prism bytecode.
//! It manages frames, globals, builtins, and the dispatch loop.

use crate::builtins::BuiltinRegistry;
use crate::dispatch::{ControlFlow, get_handler};
use crate::error::{RuntimeError, VmResult};
use crate::frame::{Frame, MAX_RECURSION_DEPTH};
use crate::globals::GlobalScope;
use crate::inline_cache::InlineCacheStore;
use crate::profiler::{CodeId, Profiler, TierUpDecision};
use prism_compiler::bytecode::CodeObject;
use prism_core::{PrismResult, Value};
use std::sync::Arc;

/// The Prism virtual machine.
///
/// Executes register-based bytecode with:
/// - Frame stack for function calls
/// - Global scope for module-level names
/// - Builtin registry for Python builtins
/// - Inline caching for attribute access
/// - Profiling for JIT tier-up decisions
pub struct VirtualMachine {
    /// Frame stack (limited by MAX_RECURSION_DEPTH).
    pub frames: Vec<Frame>,
    /// Current frame index (frames.len() - 1).
    current_frame_idx: usize,
    /// Global scope.
    pub globals: GlobalScope,
    /// Builtin functions and values.
    pub builtins: BuiltinRegistry,
    /// Inline cache storage.
    pub inline_caches: InlineCacheStore,
    /// Execution profiler.
    pub profiler: Profiler,
}

impl VirtualMachine {
    /// Create a new virtual machine.
    pub fn new() -> Self {
        Self {
            frames: Vec::with_capacity(64),
            current_frame_idx: 0,
            globals: GlobalScope::new(),
            builtins: BuiltinRegistry::with_standard_builtins(),
            inline_caches: InlineCacheStore::default(),
            profiler: Profiler::new(),
        }
    }

    /// Create with pre-populated globals.
    pub fn with_globals(globals: GlobalScope) -> Self {
        Self {
            frames: Vec::with_capacity(64),
            current_frame_idx: 0,
            globals,
            builtins: BuiltinRegistry::with_standard_builtins(),
            inline_caches: InlineCacheStore::default(),
            profiler: Profiler::new(),
        }
    }

    // =========================================================================
    // Execution
    // =========================================================================

    /// Execute a code object and return the result.
    pub fn execute(&mut self, code: Arc<CodeObject>) -> PrismResult<Value> {
        // Push the initial frame
        self.push_frame(code, 0)?;

        // Run the main dispatch loop
        self.run_loop()
    }

    /// Main dispatch loop.
    #[inline(never)] // Prevent inlining for better branch prediction
    fn run_loop(&mut self) -> PrismResult<Value> {
        loop {
            // Fetch instruction
            let inst = {
                let frame = &mut self.frames[self.current_frame_idx];

                // Check if we've reached the end of the code
                if frame.ip as usize >= frame.code.instructions.len() {
                    // Implicit return None at end of function
                    match self.pop_frame(Value::none())? {
                        Some(value) => return Ok(value),
                        None => continue,
                    }
                }

                frame.fetch()
            };

            // Dispatch to handler
            let handler = get_handler(inst.opcode());
            let control = handler(self, inst);

            // Handle control flow
            match control {
                ControlFlow::Continue => {}

                ControlFlow::Jump(offset) => {
                    // Apply relative jump
                    let frame = &mut self.frames[self.current_frame_idx];
                    let new_ip = (frame.ip as i32) + (offset as i32) - 1; // -1 because we already advanced
                    frame.ip = new_ip.max(0) as u32;
                }

                ControlFlow::Call { code, return_reg } => {
                    self.push_frame(code, return_reg)?;
                }

                ControlFlow::Return(value) => {
                    match self.pop_frame(value)? {
                        Some(result) => return Ok(result),
                        None => {} // Continue with caller frame
                    }
                }

                ControlFlow::Error(err) => {
                    return Err(err.into());
                }
            }
        }
    }

    // =========================================================================
    // Frame Management
    // =========================================================================

    /// Push a new frame for calling a function.
    pub fn push_frame(&mut self, code: Arc<CodeObject>, return_reg: u8) -> VmResult<()> {
        // Check recursion limit
        if self.frames.len() >= MAX_RECURSION_DEPTH {
            return Err(RuntimeError::recursion_error(self.frames.len()));
        }

        // Record call for profiling
        let code_id = CodeId::from_ptr(Arc::as_ptr(&code) as *const ());
        let tier_decision = self.profiler.record_call(code_id);

        // TODO: Handle tier-up decisions
        match tier_decision {
            TierUpDecision::Tier1 => {
                // Queue for template JIT compilation
            }
            TierUpDecision::Tier2 => {
                // Queue for optimizing JIT compilation
            }
            TierUpDecision::None => {}
        }

        // Create new frame
        let return_frame = if self.frames.is_empty() {
            None
        } else {
            Some(self.current_frame_idx as u32)
        };

        let frame = Frame::new(code, return_frame, return_reg);
        self.frames.push(frame);
        self.current_frame_idx = self.frames.len() - 1;

        Ok(())
    }

    /// Pop the current frame and return to caller.
    /// Returns Some(value) if this was the last frame, None otherwise.
    pub fn pop_frame(&mut self, return_value: Value) -> VmResult<Option<Value>> {
        let frame = self.frames.pop().expect("no frame to pop");

        if self.frames.is_empty() {
            // This was the last frame - return final value
            Ok(Some(return_value))
        } else {
            // Store return value in caller's register
            let return_frame_idx = frame.return_frame.unwrap_or(0) as usize;
            let return_reg = frame.return_reg;

            self.current_frame_idx = return_frame_idx;
            self.frames[return_frame_idx].set_reg(return_reg, return_value);

            Ok(None)
        }
    }

    /// Get reference to current frame.
    #[inline(always)]
    pub fn current_frame(&self) -> &Frame {
        &self.frames[self.current_frame_idx]
    }

    /// Get mutable reference to current frame.
    #[inline(always)]
    pub fn current_frame_mut(&mut self) -> &mut Frame {
        &mut self.frames[self.current_frame_idx]
    }

    // =========================================================================
    // State Access
    // =========================================================================

    /// Get the current call depth.
    #[inline]
    pub fn call_depth(&self) -> usize {
        self.frames.len()
    }

    /// Check if VM is idle (no frames).
    #[inline]
    pub fn is_idle(&self) -> bool {
        self.frames.is_empty()
    }

    /// Reset VM state for reuse.
    pub fn reset(&mut self) {
        self.frames.clear();
        self.current_frame_idx = 0;
        self.globals = GlobalScope::new();
        self.inline_caches = InlineCacheStore::default();
    }

    /// Clear only the frame stack (keep globals).
    pub fn clear_frames(&mut self) {
        self.frames.clear();
        self.current_frame_idx = 0;
    }
}

impl Default for VirtualMachine {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vm_creation() {
        let vm = VirtualMachine::new();
        assert!(vm.is_idle());
        assert_eq!(vm.call_depth(), 0);
    }

    #[test]
    fn test_vm_with_globals() {
        let mut globals = GlobalScope::new();
        globals.set("x".into(), Value::int(42).unwrap());

        let vm = VirtualMachine::with_globals(globals);
        assert_eq!(vm.globals.get("x").unwrap().as_int(), Some(42));
    }

    #[test]
    fn test_builtins_available() {
        let vm = VirtualMachine::new();
        assert!(vm.builtins.get("None").is_some());
        assert!(vm.builtins.get("True").is_some());
        assert!(vm.builtins.get("False").is_some());
    }
}
