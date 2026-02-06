//! Virtual machine implementation.
//!
//! The VirtualMachine is the main execution engine for Prism bytecode.
//! It manages frames, globals, builtins, and the dispatch loop.

use crate::allocator::GcAllocator;
use crate::builtins::BuiltinRegistry;
use crate::dispatch::{ControlFlow, get_handler};
use crate::error::{RuntimeError, VmResult};
use crate::exception::{ExcInfoStack, ExceptionState, HandlerStack, InlineHandlerCache};
use crate::frame::{Frame, MAX_RECURSION_DEPTH};
use crate::gc_integration::ManagedHeap;
use crate::globals::GlobalScope;
use crate::ic_manager::ICManager;
use crate::import::ImportResolver;
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
/// - GC-managed heap for object allocation
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

    // =========================================================================
    // GC Integration
    // =========================================================================
    /// GC-managed heap for object allocation.
    /// All runtime objects (List, Tuple, Dict, etc.) are allocated from here.
    heap: ManagedHeap,

    // =========================================================================
    // Exception Handling State
    // =========================================================================
    /// Exception state machine for tracking exception propagation phases.
    exc_state: ExceptionState,
    /// Runtime handler stack for active try/except/finally blocks.
    handler_stack: HandlerStack,
    /// Inline handler cache for fast handler lookup by PC.
    handler_cache: InlineHandlerCache,
    /// Currently active exception (if any) being propagated.
    active_exception: Option<Value>,
    /// Type ID of the active exception for fast matching.
    active_exception_type_id: Option<u16>,
    /// Exception info stack for CPython 3.11+ semantics.
    exc_info_stack: ExcInfoStack,
    /// Import resolver for module imports.
    pub import_resolver: ImportResolver,
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
            heap: ManagedHeap::with_defaults(),
            exc_state: ExceptionState::default(),
            handler_stack: HandlerStack::new(),
            handler_cache: InlineHandlerCache::new(),
            active_exception: None,
            active_exception_type_id: None,
            exc_info_stack: ExcInfoStack::new(),
            import_resolver: ImportResolver::new(),
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
            heap: ManagedHeap::with_defaults(),
            exc_state: ExceptionState::default(),
            handler_stack: HandlerStack::new(),
            handler_cache: InlineHandlerCache::new(),
            active_exception: None,
            active_exception_type_id: None,
            exc_info_stack: ExcInfoStack::new(),
            import_resolver: ImportResolver::new(),
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
            heap: ManagedHeap::with_defaults(),
            exc_state: ExceptionState::default(),
            handler_stack: HandlerStack::new(),
            handler_cache: InlineHandlerCache::new(),
            active_exception: None,
            active_exception_type_id: None,
            exc_info_stack: ExcInfoStack::new(),
            import_resolver: ImportResolver::new(),
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
            heap: ManagedHeap::with_defaults(),
            exc_state: ExceptionState::default(),
            handler_stack: HandlerStack::new(),
            handler_cache: InlineHandlerCache::new(),
            active_exception: None,
            active_exception_type_id: None,
            exc_info_stack: ExcInfoStack::new(),
            import_resolver: ImportResolver::new(),
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
                    eprintln!("[DEBUG Call] Calling function, return_reg={}", return_reg);
                    self.push_frame(code, return_reg)?;
                }

                ControlFlow::Return(value) => {
                    match self.pop_frame(value)? {
                        Some(result) => return Ok(result),
                        None => {} // Continue with caller frame
                    }
                }

                // =========================================================
                // Exception Handling
                // =========================================================
                ControlFlow::Exception {
                    type_id,
                    handler_pc: hint_pc,
                } => {
                    // Store the active exception for handlers to access
                    // (handler_pc in the ControlFlow is a hint from raise instruction encoding)
                    let _ = hint_pc; // Compiler hint, actual PC comes from exception table

                    // Look up handler in current frame's exception table
                    if let Some(handler_entry) = self.find_exception_handler(type_id) {
                        // Found a matching handler - jump to it
                        let frame = &mut self.frames[self.current_frame_idx];
                        frame.ip = handler_entry;
                    } else {
                        // No handler in current frame - unwind to caller
                        loop {
                            // Pop current frame
                            if self.frames.len() <= 1 {
                                // No more frames - return as uncaught exception
                                return Err(RuntimeError::exception(
                                    type_id,
                                    format!("Uncaught exception (type_id={})", type_id),
                                )
                                .into());
                            }

                            self.frames.pop();
                            self.current_frame_idx = self.frames.len() - 1;

                            // Try to find handler in caller
                            if let Some(handler_entry) = self.find_exception_handler(type_id) {
                                let frame = &mut self.frames[self.current_frame_idx];
                                frame.ip = handler_entry;
                                break;
                            }
                        }
                    }
                }

                ControlFlow::Reraise => {
                    eprintln!("[DEBUG ControlFlow::Reraise] Entered handler");
                    // Re-raise the current active exception
                    // First check active_exception_type_id (for except handlers)
                    // then fall back to exc_info_stack (for finally blocks)
                    let type_id = if let Some(tid) = self.active_exception_type_id {
                        eprintln!("[DEBUG Reraise] type_id from active={}", tid);
                        tid
                    } else if let Some(exc_info) = self.exc_info_stack.peek() {
                        let tid = exc_info.type_id();
                        eprintln!("[DEBUG Reraise] type_id from exc_info={}", tid);
                        tid
                    } else {
                        eprintln!("[DEBUG Reraise] No type_id found!");
                        return Err(
                            RuntimeError::type_error("No active exception to re-raise").into()
                        );
                    };

                    if type_id == 0 {
                        return Err(RuntimeError::internal(
                            "Reraise without active exception type",
                        )
                        .into());
                    }

                    // Look for handler (same as Exception flow)
                    if let Some(handler_entry) = self.find_exception_handler(type_id) {
                        let frame = &mut self.frames[self.current_frame_idx];
                        frame.ip = handler_entry;
                    } else {
                        // Unwind and propagate
                        loop {
                            if self.frames.len() <= 1 {
                                return Err(RuntimeError::exception(
                                    type_id,
                                    "Uncaught re-raised exception".to_string(),
                                )
                                .into());
                            }

                            self.frames.pop();
                            self.current_frame_idx = self.frames.len() - 1;

                            eprintln!(
                                "[DEBUG Reraise unwinding] frames.len={}, current_frame_idx={}, frame.ip={}",
                                self.frames.len(),
                                self.current_frame_idx,
                                self.frames[self.current_frame_idx].ip
                            );

                            if let Some(handler_entry) = self.find_exception_handler(type_id) {
                                let frame = &mut self.frames[self.current_frame_idx];
                                frame.ip = handler_entry;
                                break;
                            }
                        }
                    }
                }

                ControlFlow::EnterHandler {
                    handler_pc,
                    stack_depth: _,
                } => {
                    // Jump to handler code
                    let frame = &mut self.frames[self.current_frame_idx];
                    frame.ip = handler_pc;
                }

                ControlFlow::EnterFinally {
                    finally_pc,
                    stack_depth: _,
                    reraise: _,
                } => {
                    // Jump to finally block
                    let frame = &mut self.frames[self.current_frame_idx];
                    frame.ip = finally_pc;
                }

                ControlFlow::ExitHandler => {
                    // Handler completed, resume normal execution
                    // TODO: Pop handler from handler stack
                }

                // =========================================================
                // Generator Protocol
                // =========================================================
                ControlFlow::Yield {
                    value,
                    resume_point,
                } => {
                    // Generator suspension: capture frame state and return value
                    //
                    // When a generator yields:
                    // 1. Store the resume point (which yield we're at)
                    // 2. Store the current IP for resumption
                    // 3. Return the yielded value to the caller
                    //
                    // The generator's frame state (registers) is already captured
                    // by the GeneratorObject via FrameStorage when the generator
                    // was created. The VM only needs to track the resume_point.

                    // Store resume information in the frame for later
                    let frame = &mut self.frames[self.current_frame_idx];
                    frame.set_yield_point(resume_point);

                    // Pop the generator frame and return the yielded value
                    // The caller (either user code or the iterator protocol)
                    // will receive this value
                    return Ok(value);
                }

                ControlFlow::Resume { send_value } => {
                    // Generator resumption: restore frame state and continue
                    //
                    // When a generator is resumed via next()/send():
                    // 1. The send_value becomes the result of the yield expression
                    // 2. Execution continues from the stored resume point
                    //
                    // For now, place the sent value in register 0 (result register)
                    // and continue execution from where we left off.

                    let frame = &mut self.frames[self.current_frame_idx];

                    // The sent value becomes the result of the yield expression
                    // Register 0 is the conventional result register for yield
                    frame.set_reg(0, send_value);

                    // Continue normal execution from current IP
                    // (The IP was preserved when we yielded)
                }

                ControlFlow::Error(err) => {
                    // Map RuntimeErrorKind to ExceptionTypeId for handler lookup
                    use crate::error::RuntimeErrorKind;
                    use crate::stdlib::exceptions::types::ExceptionTypeId;

                    eprintln!("[DEBUG ControlFlow::Error] kind={:?}", err.kind);

                    let type_id = match &err.kind {
                        RuntimeErrorKind::TypeError { .. } => {
                            ExceptionTypeId::TypeError.as_u8() as u16
                        }
                        RuntimeErrorKind::UnsupportedOperandTypes { .. } => {
                            ExceptionTypeId::TypeError.as_u8() as u16
                        }
                        RuntimeErrorKind::NotCallable { .. } => {
                            ExceptionTypeId::TypeError.as_u8() as u16
                        }
                        RuntimeErrorKind::NotIterable { .. } => {
                            ExceptionTypeId::TypeError.as_u8() as u16
                        }
                        RuntimeErrorKind::NotSubscriptable { .. } => {
                            ExceptionTypeId::TypeError.as_u8() as u16
                        }
                        RuntimeErrorKind::NameError { .. } => {
                            ExceptionTypeId::NameError.as_u8() as u16
                        }
                        RuntimeErrorKind::AttributeError { .. } => {
                            ExceptionTypeId::AttributeError.as_u8() as u16
                        }
                        RuntimeErrorKind::UnboundLocalError { .. } => {
                            ExceptionTypeId::UnboundLocalError.as_u8() as u16
                        }
                        RuntimeErrorKind::IndexError { .. } => {
                            ExceptionTypeId::IndexError.as_u8() as u16
                        }
                        RuntimeErrorKind::KeyError { .. } => {
                            ExceptionTypeId::KeyError.as_u8() as u16
                        }
                        RuntimeErrorKind::ValueError { .. } => {
                            ExceptionTypeId::ValueError.as_u8() as u16
                        }
                        RuntimeErrorKind::ZeroDivisionError => {
                            ExceptionTypeId::ZeroDivisionError.as_u8() as u16
                        }
                        RuntimeErrorKind::OverflowError { .. } => {
                            ExceptionTypeId::OverflowError.as_u8() as u16
                        }
                        RuntimeErrorKind::StopIteration => {
                            ExceptionTypeId::StopIteration.as_u8() as u16
                        }
                        RuntimeErrorKind::GeneratorExit => {
                            ExceptionTypeId::GeneratorExit.as_u8() as u16
                        }
                        RuntimeErrorKind::AssertionError { .. } => {
                            ExceptionTypeId::AssertionError.as_u8() as u16
                        }
                        RuntimeErrorKind::RecursionError { .. } => {
                            ExceptionTypeId::RecursionError.as_u8() as u16
                        }
                        RuntimeErrorKind::ImportError { .. } => {
                            ExceptionTypeId::ImportError.as_u8() as u16
                        }
                        RuntimeErrorKind::InvalidOpcode { .. } => {
                            ExceptionTypeId::SystemError.as_u8() as u16
                        }
                        RuntimeErrorKind::InternalError { .. } => {
                            ExceptionTypeId::RuntimeError.as_u8() as u16
                        }
                        RuntimeErrorKind::Exception { type_id, .. } => *type_id,
                    };
                    // Create an exception value from the error message for reraise support
                    let exc_type_id_enum = match ExceptionTypeId::from_u8(type_id as u8) {
                        Some(id) => id,
                        None => ExceptionTypeId::RuntimeError,
                    };
                    let error_message = err.to_string();
                    let exc_value = crate::builtins::create_exception(
                        exc_type_id_enum,
                        Some(std::sync::Arc::from(error_message.as_str())),
                    );

                    // Set the active exception (both value and type_id) for handler matching + reraise
                    self.set_active_exception_with_type(exc_value, type_id);

                    // Try to find a handler in current frame
                    if let Some(handler_entry) = self.find_exception_handler(type_id) {
                        let frame = &mut self.frames[self.current_frame_idx];
                        frame.ip = handler_entry;
                    } else {
                        // Unwind stack looking for handler
                        loop {
                            if self.frames.len() <= 1 {
                                // No handlers found - propagate the error
                                return Err(err.into());
                            }

                            self.frames.pop();
                            self.current_frame_idx = self.frames.len() - 1;

                            if let Some(handler_entry) = self.find_exception_handler(type_id) {
                                let frame = &mut self.frames[self.current_frame_idx];
                                frame.ip = handler_entry;
                                break;
                            }
                        }
                    }
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

    // =========================================================================
    // GC Integration
    // =========================================================================

    /// Get read-only access to the managed heap.
    ///
    /// Use this to query heap statistics, check collection thresholds,
    /// or read heap configuration.
    #[inline]
    pub fn heap(&self) -> &ManagedHeap {
        &self.heap
    }

    /// Get mutable access to the managed heap.
    ///
    /// Required for:
    /// - Triggering garbage collection
    /// - Updating root sets
    /// - Modifying heap configuration
    #[inline]
    pub fn heap_mut(&mut self) -> &mut ManagedHeap {
        &mut self.heap
    }

    /// Get a typed allocator for GC-managed object allocation.
    ///
    /// This provides a zero-cost typed interface for allocating objects
    /// on the GC heap. The allocator borrows the underlying GcHeap,
    /// ensuring type-safe allocation with proper Trace bounds.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let list = vm.allocator().alloc(ListObject::from_slice(&values))?;
    /// let value = Value::object_ptr(list as *const ());
    /// ```
    ///
    /// # Performance
    ///
    /// This method is `#[inline]` and creates a zero-cost wrapper.
    /// The allocator itself is stack-allocated and holds only a reference.
    #[inline]
    pub fn allocator(&self) -> GcAllocator<'_> {
        GcAllocator::new(self.heap.heap())
    }

    // =========================================================================
    // Exception Handling
    // =========================================================================

    /// Get the current frame's ID for exception handler tracking.
    #[inline]
    pub fn current_frame_id(&self) -> u32 {
        self.current_frame_idx as u32
    }

    /// Set the active exception being propagated.
    /// Uses generic Exception type (4) - prefer set_active_exception_with_type for proper matching.
    #[inline]
    pub fn set_active_exception(&mut self, exc: Value) {
        self.active_exception = Some(exc);
        self.active_exception_type_id = Some(4); // Generic Exception type
        self.exc_state = ExceptionState::Propagating;
    }

    /// Set the active exception with a specific type ID.
    /// This enables proper exception type matching in except handlers.
    #[inline]
    pub fn set_active_exception_with_type(&mut self, exc: Value, type_id: u16) {
        self.active_exception = Some(exc);
        self.active_exception_type_id = Some(type_id);
        self.exc_state = ExceptionState::Propagating;
    }

    /// Get the active exception if any.
    #[inline]
    pub fn get_active_exception(&self) -> Option<&Value> {
        self.active_exception.as_ref()
    }

    /// Check if there's an active exception.
    #[inline]
    pub fn has_active_exception(&self) -> bool {
        self.active_exception.is_some()
    }

    /// Clear the active exception.
    #[inline]
    pub fn clear_active_exception(&mut self) {
        self.active_exception = None;
        self.active_exception_type_id = None;
    }

    /// Get the type ID of the active exception.
    ///
    /// Returns the exception type ID for fast matching, or None if no
    /// active exception exists.
    #[inline]
    pub fn get_active_exception_type_id(&self) -> Option<u16> {
        self.active_exception_type_id
    }

    /// Push an exception handler onto the handler stack.
    ///
    /// Returns false if the stack is full.
    #[inline]
    pub fn push_exception_handler(&mut self, frame: crate::exception::HandlerFrame) -> bool {
        self.handler_stack.push(frame)
    }

    /// Pop an exception handler from the handler stack.
    #[inline]
    pub fn pop_exception_handler(&mut self) -> Option<crate::exception::HandlerFrame> {
        self.handler_stack.pop()
    }

    /// Check if we should reraise after a finally block.
    ///
    /// We should reraise if there's still an active exception after the finally
    /// body executes. Note: We can't rely on exc_state == Finally because
    /// PopExcInfo may have changed it during the finally execution.
    #[inline]
    pub fn should_reraise_after_finally(&self) -> bool {
        // If there's an active exception with a valid type, reraise it
        // This handles both:
        // 1. Normal exceptions that entered finally during propagation
        // 2. Pop'd exception info that was restored as active
        self.active_exception_type_id.is_some() && self.active_exception_type_id != Some(0)
    }

    /// Clear the reraise flag after handling.
    #[inline]
    pub fn clear_reraise_flag(&mut self) {
        // Transition state - exception will be preserved for reraise
    }

    /// Clear exception state (after successful handling).
    #[inline]
    pub fn clear_exception_state(&mut self) {
        self.exc_state = ExceptionState::Normal;
    }

    /// Get the current exception state.
    #[inline]
    pub fn exception_state(&self) -> ExceptionState {
        self.exc_state
    }

    /// Set the exception state directly.
    #[inline]
    pub fn set_exception_state(&mut self, state: ExceptionState) {
        self.exc_state = state;
    }

    /// Cache a handler lookup result for fast path.
    #[inline]
    pub fn cache_handler(&mut self, pc: u32, handler_idx: u16) {
        self.handler_cache.record(pc, handler_idx);
    }

    /// Look up a cached handler for a PC.
    #[inline]
    pub fn lookup_cached_handler(&mut self, pc: u32) -> Option<u16> {
        self.handler_cache.try_get(pc)
    }

    /// Get the handler stack depth.
    #[inline]
    pub fn handler_stack_depth(&self) -> usize {
        self.handler_stack.len()
    }

    /// Find an exception handler for the given exception type in the current frame.
    ///
    /// Searches the frame's exception table for a handler that:
    /// 1. Covers the current PC (start_pc <= pc < end_pc)
    /// 2. Matches the exception type (or is a catch-all with type_idx = 0xFFFF)
    ///
    /// Returns the handler PC if a matching handler is found.
    ///
    /// # Performance
    ///
    /// Exception tables are typically small (<10 entries) and sorted by start_pc.
    /// Linear scan with early termination is optimal for this size.
    #[inline]
    pub fn find_exception_handler(&self, type_id: u16) -> Option<u32> {
        let frame = &self.frames[self.current_frame_idx];
        let pc = frame.ip.saturating_sub(1); // PC is post-increment, so -1 for current instruction

        eprintln!(
            "[DEBUG find_handler] pc={}, type_id={}, table_len={}",
            pc,
            type_id,
            frame.code.exception_table.len()
        );

        // Search exception table for matching handler
        // The handler bytecode itself uses ExceptionMatch opcode to do type matching,
        // so we just need to find any handler covering this PC range.
        for entry in frame.code.exception_table.iter() {
            eprintln!(
                "[DEBUG find_handler]   entry: start={}, end={}, handler={}",
                entry.start_pc, entry.end_pc, entry.handler_pc
            );
            // Check if PC is in range
            if pc >= entry.start_pc && pc < entry.end_pc {
                eprintln!(
                    "[DEBUG find_handler]   FOUND handler at {}",
                    entry.handler_pc
                );
                // Jump to handler - handler bytecode will do type matching via ExceptionMatch
                return Some(entry.handler_pc);
            }
        }

        eprintln!("[DEBUG find_handler]   NO handler found");
        None
    }

    // =========================================================================
    // Exception Info Stack (CPython 3.11+ semantics)
    // =========================================================================

    /// Get a reference to the exception info stack.
    #[inline]
    pub fn exc_info_stack(&self) -> &ExcInfoStack {
        &self.exc_info_stack
    }

    /// Get a mutable reference to the exception info stack.
    #[inline]
    pub fn exc_info_stack_mut(&mut self) -> &mut ExcInfoStack {
        &mut self.exc_info_stack
    }

    /// Push current exception info onto the stack.
    /// Returns false if stack is full.
    #[inline]
    pub fn push_exc_info(&mut self) -> bool {
        use crate::exception::ExcInfoEntry;

        let type_id = self.get_active_exception_type_id().unwrap_or(0);
        let value = self.active_exception.clone();
        let entry = ExcInfoEntry::new(type_id, value);
        self.exc_info_stack.push(entry)
    }

    /// Pop exception info from the stack and restore it as active.
    #[inline]
    pub fn pop_exc_info(&mut self) -> bool {
        if let Some(entry) = self.exc_info_stack.pop() {
            if entry.is_active() {
                self.active_exception = entry.value_cloned();
                self.exc_state = ExceptionState::Propagating;
            } else {
                self.active_exception = None;
                self.exc_state = ExceptionState::Normal;
            }
            true
        } else {
            false
        }
    }

    /// Check if there's exception info on the stack.
    #[inline]
    pub fn has_exc_info(&self) -> bool {
        !self.exc_info_stack.is_empty() || self.active_exception.is_some()
    }

    /// Get current exception info as (type_id, value, traceback_id).
    #[inline]
    pub fn current_exc_info(&self) -> (Option<u16>, Option<Value>, Option<u32>) {
        self.exc_info_stack.current_exc_info()
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
