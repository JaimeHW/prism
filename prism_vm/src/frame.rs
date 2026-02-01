//! Call frame management with stack-based register file.
//!
//! The Frame struct is the core execution context for a function call.
//! It uses a stack-allocated register file for maximum performance.

use prism_compiler::bytecode::CodeObject;
use prism_core::Value;
use std::sync::Arc;

/// Maximum recursion depth before RecursionError.
pub const MAX_RECURSION_DEPTH: usize = 1000;

/// Number of registers per frame.
pub const REGISTER_COUNT: usize = 256;

/// A call frame representing a function invocation.
///
/// The frame contains:
/// - Reference to the code object being executed
/// - Instruction pointer (current position)
/// - Stack-based register file (256 values, ~2KB)
/// - Optional closure environment
/// - Return information for popping the frame
///
/// # Performance
///
/// The register file is stored inline (not boxed) to avoid heap allocation
/// and ensure L1 cache locality. At 256 * 8 = 2048 bytes, it fits comfortably
/// in modern L1 caches (typically 32-64KB).
#[repr(C)]
pub struct Frame {
    /// Code object being executed.
    pub code: Arc<CodeObject>,

    /// Instruction pointer (index into code.instructions).
    /// Using u32 for compact representation (4GB of instructions is plenty).
    pub ip: u32,

    /// Return address: index of caller frame in the frame stack.
    /// None for the top-level module frame.
    pub return_frame: Option<u32>,

    /// Register in the caller frame where the return value should be stored.
    pub return_reg: u8,

    /// Closure environment for captured variables.
    /// Only set for closures, None for regular functions.
    pub closure: Option<Arc<ClosureEnv>>,

    /// Stack-based register file.
    /// Registers r0-r255 used for local computation.
    /// Parameters are passed in r0, r1, r2, ...
    pub registers: [Value; REGISTER_COUNT],
}

/// Closure environment holding captured variables.
///
/// Closures share their environment with the enclosing scope,
/// allowing mutations to be visible across closure invocations.
#[derive(Debug, Clone)]
pub struct ClosureEnv {
    /// Captured values (indexed by closure slot).
    pub values: Box<[Value]>,
}

impl Frame {
    /// Create a new frame for executing a code object.
    ///
    /// # Arguments
    /// * `code` - The code object to execute
    /// * `return_frame` - Index of the caller frame (None for module level)
    /// * `return_reg` - Register in caller to store return value
    #[inline]
    pub fn new(code: Arc<CodeObject>, return_frame: Option<u32>, return_reg: u8) -> Self {
        Self {
            code,
            ip: 0,
            return_frame,
            return_reg,
            closure: None,
            // Initialize all registers to None for safety
            registers: [Value::none(); REGISTER_COUNT],
        }
    }

    /// Create a frame with a closure environment.
    #[inline]
    pub fn with_closure(
        code: Arc<CodeObject>,
        return_frame: Option<u32>,
        return_reg: u8,
        closure: Arc<ClosureEnv>,
    ) -> Self {
        Self {
            code,
            ip: 0,
            return_frame,
            return_reg,
            closure: Some(closure),
            registers: [Value::none(); REGISTER_COUNT],
        }
    }

    // =========================================================================
    // Register Access (Inlined for Performance)
    // =========================================================================

    /// Get a register value.
    #[inline(always)]
    pub fn get_reg(&self, reg: u8) -> Value {
        // Safety: reg is u8, so always in bounds for 256-element array
        unsafe { *self.registers.get_unchecked(reg as usize) }
    }

    /// Set a register value.
    #[inline(always)]
    pub fn set_reg(&mut self, reg: u8, value: Value) {
        // Safety: reg is u8, so always in bounds for 256-element array
        unsafe { *self.registers.get_unchecked_mut(reg as usize) = value }
    }

    /// Get two register values (common for binary ops).
    #[inline(always)]
    pub fn get_regs2(&self, r1: u8, r2: u8) -> (Value, Value) {
        unsafe {
            (
                *self.registers.get_unchecked(r1 as usize),
                *self.registers.get_unchecked(r2 as usize),
            )
        }
    }

    /// Get three register values (common for ternary ops).
    #[inline(always)]
    pub fn get_regs3(&self, r1: u8, r2: u8, r3: u8) -> (Value, Value, Value) {
        unsafe {
            (
                *self.registers.get_unchecked(r1 as usize),
                *self.registers.get_unchecked(r2 as usize),
                *self.registers.get_unchecked(r3 as usize),
            )
        }
    }

    // =========================================================================
    // Instruction Fetching
    // =========================================================================

    /// Fetch the current instruction and advance IP.
    #[inline(always)]
    pub fn fetch(&mut self) -> prism_compiler::bytecode::Instruction {
        let inst = unsafe { *self.code.instructions.get_unchecked(self.ip as usize) };
        self.ip += 1;
        inst
    }

    /// Peek at the current instruction without advancing.
    #[inline(always)]
    pub fn peek(&self) -> prism_compiler::bytecode::Instruction {
        unsafe { *self.code.instructions.get_unchecked(self.ip as usize) }
    }

    /// Check if execution is complete (IP past end of instructions).
    #[inline(always)]
    pub fn is_done(&self) -> bool {
        self.ip as usize >= self.code.instructions.len()
    }

    // =========================================================================
    // Constant/Name Access
    // =========================================================================

    /// Get a constant from the constant pool.
    #[inline(always)]
    pub fn get_const(&self, idx: u16) -> Value {
        unsafe { *self.code.constants.get_unchecked(idx as usize) }
    }

    /// Get a name from the name table.
    #[inline(always)]
    pub fn get_name(&self, idx: u16) -> &Arc<str> {
        unsafe { self.code.names.get_unchecked(idx as usize) }
    }

    /// Get a local variable name.
    #[inline(always)]
    pub fn get_local_name(&self, idx: u16) -> &Arc<str> {
        unsafe { self.code.locals.get_unchecked(idx as usize) }
    }
}

impl ClosureEnv {
    /// Create a new closure environment with the given capacity.
    pub fn new(size: usize) -> Self {
        Self {
            values: vec![Value::none(); size].into_boxed_slice(),
        }
    }

    /// Get a captured value.
    #[inline(always)]
    pub fn get(&self, idx: usize) -> Value {
        unsafe { *self.values.get_unchecked(idx) }
    }

    /// Set a captured value.
    #[inline(always)]
    pub fn set(&mut self, idx: usize, value: Value) {
        unsafe { *self.values.get_unchecked_mut(idx) = value }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_frame_size() {
        // Frame should be approximately 2KB + overhead
        let size = std::mem::size_of::<Frame>();
        // 256 * 8 (registers) + code arc + ip + return info + closure
        assert!(size >= REGISTER_COUNT * 8);
        assert!(size < 4096, "Frame too large: {} bytes", size);
    }

    #[test]
    fn test_register_access() {
        let code = Arc::new(CodeObject::new("test", "test.py"));
        let mut frame = Frame::new(code, None, 0);

        // Test set/get
        frame.set_reg(0, Value::int(42).unwrap());
        assert_eq!(frame.get_reg(0).as_int(), Some(42));

        // Test boundary registers
        frame.set_reg(255, Value::float(3.14));
        assert!((frame.get_reg(255).as_float().unwrap() - 3.14).abs() < 0.001);
    }

    #[test]
    fn test_register_multi_access() {
        let code = Arc::new(CodeObject::new("test", "test.py"));
        let mut frame = Frame::new(code, None, 0);

        frame.set_reg(1, Value::int(10).unwrap());
        frame.set_reg(2, Value::int(20).unwrap());
        frame.set_reg(3, Value::int(30).unwrap());

        let (a, b) = frame.get_regs2(1, 2);
        assert_eq!(a.as_int(), Some(10));
        assert_eq!(b.as_int(), Some(20));

        let (x, y, z) = frame.get_regs3(1, 2, 3);
        assert_eq!(x.as_int(), Some(10));
        assert_eq!(y.as_int(), Some(20));
        assert_eq!(z.as_int(), Some(30));
    }

    #[test]
    fn test_closure_env() {
        let mut env = ClosureEnv::new(3);

        env.set(0, Value::int(100).unwrap());
        env.set(1, Value::float(2.5));
        env.set(2, Value::none());

        assert_eq!(env.get(0).as_int(), Some(100));
        assert!((env.get(1).as_float().unwrap() - 2.5).abs() < 0.001);
        assert!(env.get(2).is_none());
    }
}
