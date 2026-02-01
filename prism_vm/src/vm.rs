//! Virtual machine implementation.
//!
//! The VirtualMachine is the main execution engine for Prism bytecode.
//! It manages frames, globals, builtins, and the dispatch loop.

use crate::builtins::BuiltinRegistry;
use crate::dispatch::{ControlFlow, get_handler};
use crate::error::{RuntimeError, VmResult};
use crate::frame::{Frame, MAX_RECURSION_DEPTH};
use crate::globals::GlobalScope;
use crate::ic_manager::ICManager;
use crate::inline_cache::InlineCacheStore;
use crate::jit_context::{JitConfig, JitContext};
use crate::jit_executor::ExecutionResult;
use crate::profiler::{CodeId, Profiler, TierUpDecision};
use crate::speculative::SpeculationCache;
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
/// - Optional JIT compilation and execution
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
    /// IC Manager for centralized type profiling.
    pub ic_manager: ICManager,
    /// Speculation cache for O(1) fast-path lookup.
    pub speculation_cache: SpeculationCache,
    /// JIT context (None when JIT is disabled).
    jit: Option<JitContext>,
    /// Temporary storage for JIT return value when root frame executes via JIT.
    jit_return_value: Option<Value>,
}

impl VirtualMachine {
    /// Create a new virtual machine (interpreter only, no JIT).
    pub fn new() -> Self {
        Self {
            frames: Vec::with_capacity(64),
            current_frame_idx: 0,
            globals: GlobalScope::new(),
            builtins: BuiltinRegistry::with_standard_builtins(),
            inline_caches: InlineCacheStore::default(),
            profiler: Profiler::new(),
            ic_manager: ICManager::new(),
            speculation_cache: SpeculationCache::new(),
            jit: None,
            jit_return_value: None,
        }
    }

    /// Create a new virtual machine with JIT compilation enabled.
    pub fn with_jit() -> Self {
        Self {
            frames: Vec::with_capacity(64),
            current_frame_idx: 0,
            globals: GlobalScope::new(),
            builtins: BuiltinRegistry::with_standard_builtins(),
            inline_caches: InlineCacheStore::default(),
            profiler: Profiler::new(),
            ic_manager: ICManager::new(),
            speculation_cache: SpeculationCache::new(),
            jit: Some(JitContext::with_defaults()),
            jit_return_value: None,
        }
    }

    /// Create a virtual machine with custom JIT configuration.
    pub fn with_jit_config(config: JitConfig) -> Self {
        let jit = if config.enabled {
            Some(JitContext::new(config))
        } else {
            None
        };
        Self {
            frames: Vec::with_capacity(64),
            current_frame_idx: 0,
            globals: GlobalScope::new(),
            builtins: BuiltinRegistry::with_standard_builtins(),
            inline_caches: InlineCacheStore::default(),
            profiler: Profiler::new(),
            ic_manager: ICManager::new(),
            speculation_cache: SpeculationCache::new(),
            jit,
            jit_return_value: None,
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
            ic_manager: ICManager::new(),
            speculation_cache: SpeculationCache::new(),
            jit: None,
            jit_return_value: None,
        }
    }

    // =========================================================================
    // Execution
    // =========================================================================

    /// Execute a code object and return the result.
    pub fn execute(&mut self, code: Arc<CodeObject>) -> PrismResult<Value> {
        // Push the initial frame (may execute via JIT)
        self.push_frame(code, 0)?;

        // If JIT handled everything and no frame was pushed, return the result
        // from the JIT return value register (which is stored in our temp location)
        if self.frames.is_empty() {
            // JIT executed successfully without pushing a frame
            // The return value was stored in jit_return_value by push_frame
            return Ok(self.jit_return_value.take().unwrap_or_else(Value::none));
        }

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
                    // Note: offset is computed by compiler relative to instruction after jump,
                    // and fetch() already advanced ip, so just add offset directly
                    let frame = &mut self.frames[self.current_frame_idx];
                    let new_ip = (frame.ip as i32) + (offset as i32);
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
    ///
    /// This method implements a JIT-first dispatch strategy:
    /// 1. Profile the call and handle tier-up decisions
    /// 2. If compiled code exists, execute it directly
    /// 3. On JIT return, propagate value to caller
    /// 4. On deopt, create frame and resume interpreter
    /// 5. On miss, fall through to interpreter
    pub fn push_frame(&mut self, code: Arc<CodeObject>, return_reg: u8) -> VmResult<()> {
        // Check recursion limit
        if self.frames.len() >= MAX_RECURSION_DEPTH {
            return Err(RuntimeError::recursion_error(self.frames.len()));
        }

        // Record call for profiling
        let code_id = CodeId::from_ptr(Arc::as_ptr(&code) as *const ());
        let tier_decision = self.profiler.record_call(code_id);

        // Handle JIT: check for compiled code, handle tier-up, and try execution
        if let Some(jit) = &mut self.jit {
            // Handle tier-up decision (may trigger compilation)
            if tier_decision != TierUpDecision::None {
                jit.handle_tier_up(&code, tier_decision);
            }

            // Get code pointer ID for cache lookup
            let code_ptr_id = Arc::as_ptr(&code) as u64;

            // Try to execute compiled code if available
            if jit.lookup(code_ptr_id).is_some() {
                // Create temporary frame for JIT execution
                let return_frame_idx = if self.frames.is_empty() {
                    None
                } else {
                    Some(self.current_frame_idx as u32)
                };
                let mut jit_frame = Frame::new(Arc::clone(&code), return_frame_idx, return_reg);

                // Execute compiled code
                match jit.try_execute(code_ptr_id, &mut jit_frame) {
                    Some(ExecutionResult::Return(value)) => {
                        // JIT completed successfully - handle return value
                        if self.frames.is_empty() {
                            // Root frame execution - store value for execute() to retrieve
                            self.jit_return_value = Some(value);
                        } else {
                            // Nested call - store return value in caller's register
                            self.frames[self.current_frame_idx].set_reg(return_reg, value);
                        }
                        // Don't push a frame - JIT handled everything
                        return Ok(());
                    }
                    Some(ExecutionResult::Deopt { bc_offset, reason }) => {
                        // Deoptimization - resume interpreter at bc_offset
                        jit.handle_deopt(code_ptr_id, reason);
                        jit_frame.ip = bc_offset;
                        self.frames.push(jit_frame);
                        self.current_frame_idx = self.frames.len() - 1;
                        return Ok(());
                    }
                    Some(ExecutionResult::Exception(err)) => {
                        return Err(err);
                    }
                    Some(ExecutionResult::TailCall { .. }) => {
                        // Tail call - fall through to interpreter for now
                        // TODO: Implement tail call optimization
                        jit.record_miss();
                    }
                    None => {
                        // Execution didn't happen - fall through
                        jit.record_miss();
                    }
                }
            } else {
                jit.record_miss();
            }
        }

        // Fall through to interpreter - push frame normally
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
