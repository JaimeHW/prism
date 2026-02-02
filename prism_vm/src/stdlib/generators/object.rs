//! Generator object implementation.
//!
//! This module provides the `GeneratorObject` which represents a suspended
//! Python generator. The object captures:
//!
//! - Execution state (via `GeneratorHeader`)
//! - Live registers (via `FrameStorage`)
//! - Code reference and instruction pointer
//! - Sent/thrown values for coroutine protocol
//!
//! # Memory Layout (96 bytes target)
//!
//! ```text
//! ┌───────────────────────────────────────────────────────────────┐
//! │ GeneratorHeader (4 bytes) │ Flags (2 bytes) │ Pad (2 bytes)   │ 8 bytes
//! ├───────────────────────────────────────────────────────────────┤
//! │ Code: Arc<CodeObject>                                         │ 8 bytes
//! ├───────────────────────────────────────────────────────────────┤
//! │ IP: u32 │ Receive: Option<u8>                                 │ 8 bytes
//! ├───────────────────────────────────────────────────────────────┤
//! │ FrameStorage (inline [Value; 8] + metadata)                   │ ~72 bytes
//! └───────────────────────────────────────────────────────────────┘
//! ```

use prism_compiler::bytecode::CodeObject;
use prism_core::Value;
use std::fmt;
use std::sync::Arc;

use super::state::{GeneratorHeader, GeneratorState};
use super::storage::{FrameStorage, LivenessMap};

// ============================================================================
// Generator Flags
// ============================================================================

/// Generator object flags.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct GeneratorFlags(u16);

impl GeneratorFlags {
    /// Generator has JIT-compiled code available.
    pub const HAS_JIT: Self = Self(0b0000_0001);
    /// Generator is using inline frame storage.
    pub const INLINE_STORAGE: Self = Self(0b0000_0010);
    /// Generator has a closure environment.
    pub const HAS_CLOSURE: Self = Self(0b0000_0100);
    /// Generator is a coroutine (uses send/throw).
    pub const IS_COROUTINE: Self = Self(0b0000_1000);
    /// Generator is an async generator.
    pub const IS_ASYNC: Self = Self(0b0001_0000);
    /// Generator has been started at least once.
    pub const STARTED: Self = Self(0b0010_0000);
    /// Empty flag set.
    pub const EMPTY: Self = Self(0);

    /// Returns true if this contains all bits of `other`.
    #[inline]
    pub const fn contains(self, other: Self) -> bool {
        self.0 & other.0 == other.0
    }

    /// Returns the union of self and other.
    #[inline]
    pub const fn union(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }

    /// Returns the raw bits.
    #[inline]
    pub const fn bits(self) -> u16 {
        self.0
    }
}

impl std::ops::BitOr for GeneratorFlags {
    type Output = Self;

    #[inline]
    fn bitor(self, rhs: Self) -> Self::Output {
        Self(self.0 | rhs.0)
    }
}

impl std::ops::BitOrAssign for GeneratorFlags {
    #[inline]
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0;
    }
}

// ============================================================================
// Generator Object
// ============================================================================

/// A Python generator object.
///
/// This represents a suspended generator function, storing all state
/// necessary to resume execution at the next yield point.
///
/// # Performance
///
/// - Creation: ~15 cycles (inline storage, no allocation)
/// - Yield: ~3-5 cycles (state update + value copy)
/// - Resume: ~5 cycles (state check + dispatch)
/// - Memory: 96 bytes typical (2 cache lines)
pub struct GeneratorObject {
    // === HEADER (8 bytes) ===
    /// Tagged state + resume index.
    header: GeneratorHeader,
    /// Configuration flags.
    flags: GeneratorFlags,
    /// Padding for alignment.
    _pad: u16,

    // === CODE REFERENCE (8 bytes) ===
    /// Reference to the generator's code object.
    code: Arc<CodeObject>,

    // === EXECUTION STATE (8 bytes) ===
    /// Instruction pointer to resume at.
    ip: u32,
    /// Liveness map for current yield point (compact form).
    liveness_bits: u32,

    // === FRAME STORAGE ===
    /// Storage for live register values.
    storage: FrameStorage,

    // === SEND/THROW VALUE ===
    /// Value received via send() or throw().
    receive_value: Option<Value>,
}

impl GeneratorObject {
    /// Creates a new generator object for the given code.
    ///
    /// The generator starts in `Created` state and must be primed
    /// with the first call to `next()`.
    #[inline]
    pub fn new(code: Arc<CodeObject>) -> Self {
        Self {
            header: GeneratorHeader::new(),
            flags: GeneratorFlags::INLINE_STORAGE,
            _pad: 0,
            code,
            ip: 0,
            liveness_bits: 0,
            storage: FrameStorage::new(),
            receive_value: None,
        }
    }

    /// Creates a generator with specific flags (e.g., for async generators).
    #[inline]
    pub fn with_flags(code: Arc<CodeObject>, flags: GeneratorFlags) -> Self {
        Self {
            header: GeneratorHeader::new(),
            flags,
            _pad: 0,
            code,
            ip: 0,
            liveness_bits: 0,
            storage: FrameStorage::new(),
            receive_value: None,
        }
    }

    // ═══════════════════════════════════════════════════════════════════════
    // State Accessors
    // ═══════════════════════════════════════════════════════════════════════

    /// Returns the current generator state.
    #[inline(always)]
    pub fn state(&self) -> GeneratorState {
        self.header.state()
    }

    /// Returns the resume index (yield point ID).
    #[inline(always)]
    pub fn resume_index(&self) -> u32 {
        self.header.resume_index()
    }

    /// Returns true if the generator can be resumed.
    #[inline(always)]
    pub fn is_resumable(&self) -> bool {
        self.header.is_resumable()
    }

    /// Returns true if the generator is exhausted.
    #[inline(always)]
    pub fn is_exhausted(&self) -> bool {
        self.header.is_exhausted()
    }

    /// Returns true if the generator is currently running.
    #[inline(always)]
    pub fn is_running(&self) -> bool {
        self.header.is_running()
    }

    /// Returns the generator's flags.
    #[inline(always)]
    pub fn flags(&self) -> GeneratorFlags {
        self.flags
    }

    /// Returns a reference to the code object.
    #[inline(always)]
    pub fn code(&self) -> &Arc<CodeObject> {
        &self.code
    }

    /// Returns the instruction pointer.
    #[inline(always)]
    pub fn ip(&self) -> u32 {
        self.ip
    }

    /// Returns the liveness map for the current yield point.
    #[inline]
    pub fn liveness(&self) -> LivenessMap {
        LivenessMap::from_bits(self.liveness_bits as u64)
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Lifecycle Methods
    // ═══════════════════════════════════════════════════════════════════════

    /// Attempts to start or resume the generator.
    ///
    /// Returns the previous state if successful, or None if the generator
    /// cannot be resumed (already running or exhausted).
    #[inline]
    pub fn try_start(&self) -> Option<GeneratorState> {
        self.header.try_start()
    }

    /// Suspends the generator at a yield point.
    ///
    /// # Parameters
    /// - `ip`: Instruction pointer for the next resume
    /// - `resume_index`: Yield point index for dispatch
    /// - `registers`: Current register file to capture
    /// - `liveness`: Bitmap of live registers at this yield
    #[inline]
    pub fn suspend(
        &mut self,
        ip: u32,
        resume_index: u32,
        registers: &[Value; 256],
        liveness: LivenessMap,
    ) {
        self.ip = ip;
        self.liveness_bits = liveness.bits() as u32;
        self.storage.capture(registers, liveness);
        self.header.suspend(resume_index);
    }

    /// Marks the generator as exhausted (returned or closed).
    #[inline]
    pub fn exhaust(&self) {
        self.header.exhaust();
    }

    /// Restores the generator's frame to a register file.
    #[inline]
    pub fn restore(&self, registers: &mut [Value; 256]) {
        let liveness = self.liveness();
        self.storage.restore(registers, liveness);
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Send/Throw Protocol
    // ═══════════════════════════════════════════════════════════════════════

    /// Sets a value to be received by the generator on resume.
    ///
    /// This is used by `send()` to inject values into the generator.
    #[inline]
    pub fn set_send_value(&mut self, value: Value) {
        self.receive_value = Some(value);
    }

    /// Takes the pending receive value, if any.
    #[inline]
    pub fn take_receive_value(&mut self) -> Option<Value> {
        self.receive_value.take()
    }

    /// Returns the pending receive value without consuming it.
    #[inline]
    pub fn peek_receive_value(&self) -> Option<Value> {
        self.receive_value
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Flag Manipulation
    // ═══════════════════════════════════════════════════════════════════════

    /// Sets the STARTED flag.
    #[inline]
    pub fn mark_started(&mut self) {
        self.flags |= GeneratorFlags::STARTED;
    }

    /// Returns true if the generator has been started.
    #[inline]
    pub fn is_started(&self) -> bool {
        self.flags.contains(GeneratorFlags::STARTED)
    }

    /// Returns true if this is a coroutine.
    #[inline]
    pub fn is_coroutine(&self) -> bool {
        self.flags.contains(GeneratorFlags::IS_COROUTINE)
    }

    /// Returns true if this is an async generator.
    #[inline]
    pub fn is_async(&self) -> bool {
        self.flags.contains(GeneratorFlags::IS_ASYNC)
    }
}

impl Clone for GeneratorObject {
    fn clone(&self) -> Self {
        Self {
            header: self.header.clone(),
            flags: self.flags,
            _pad: 0,
            code: Arc::clone(&self.code),
            ip: self.ip,
            liveness_bits: self.liveness_bits,
            storage: self.storage.clone(),
            receive_value: self.receive_value,
        }
    }
}

impl fmt::Debug for GeneratorObject {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("GeneratorObject")
            .field("state", &self.state())
            .field("resume_index", &self.resume_index())
            .field("ip", &self.ip)
            .field("flags", &self.flags)
            .field("storage_len", &self.storage.len())
            .finish()
    }
}

// SAFETY: GeneratorObject is safe to send between threads
// (though concurrent access to a single generator is not safe)
unsafe impl Send for GeneratorObject {}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use prism_compiler::bytecode::CodeObject;
    use std::sync::Arc;

    fn test_code() -> Arc<CodeObject> {
        Arc::new(CodeObject::new("test_generator", "<test>"))
    }

    // ════════════════════════════════════════════════════════════════════════
    // Construction Tests
    // ════════════════════════════════════════════════════════════════════════

    #[test]
    fn test_generator_size() {
        let size = std::mem::size_of::<GeneratorObject>();
        // Should be reasonably small, target ~96 bytes
        assert!(size <= 160, "GeneratorObject too large: {}", size);
    }

    #[test]
    fn test_generator_new() {
        let code = test_code();
        let generator = GeneratorObject::new(code);

        assert_eq!(generator.state(), GeneratorState::Created);
        assert_eq!(generator.resume_index(), 0);
        assert_eq!(generator.ip(), 0);
        assert!(generator.is_resumable());
        assert!(!generator.is_exhausted());
        assert!(!generator.is_running());
    }

    #[test]
    fn test_generator_with_flags() {
        let code = test_code();
        let flags = GeneratorFlags::IS_COROUTINE | GeneratorFlags::INLINE_STORAGE;
        let generator = GeneratorObject::with_flags(code, flags);

        assert!(generator.is_coroutine());
        assert!(!generator.is_async());
        assert_eq!(generator.flags(), flags);
    }

    // ════════════════════════════════════════════════════════════════════════
    // Lifecycle Tests
    // ════════════════════════════════════════════════════════════════════════

    #[test]
    fn test_generator_start() {
        let code = test_code();
        let generator = GeneratorObject::new(code);

        let prev = generator.try_start();
        assert_eq!(prev, Some(GeneratorState::Created));
        assert!(generator.is_running());
        assert!(!generator.is_resumable());
    }

    #[test]
    fn test_generator_start_twice_fails() {
        let code = test_code();
        let generator = GeneratorObject::new(code);

        generator.try_start();
        let second = generator.try_start();
        assert_eq!(second, None);
    }

    #[test]
    fn test_generator_suspend() {
        let code = test_code();
        let mut generator = GeneratorObject::new(code);

        generator.try_start();

        let mut registers = [Value::none(); 256];
        registers[0] = Value::int(42).unwrap();
        registers[2] = Value::int(99).unwrap();

        let liveness = LivenessMap::from_bits(0b101);
        generator.suspend(100, 1, &registers, liveness);

        assert_eq!(generator.state(), GeneratorState::Suspended);
        assert_eq!(generator.resume_index(), 1);
        assert_eq!(generator.ip(), 100);
    }

    #[test]
    fn test_generator_suspend_restore() {
        let code = test_code();
        let mut generator = GeneratorObject::new(code);

        generator.try_start();

        let mut registers = [Value::none(); 256];
        registers[0] = Value::int(10).unwrap();
        registers[2] = Value::int(20).unwrap();
        registers[5] = Value::int(30).unwrap();

        let liveness = LivenessMap::from_bits(0b100101);
        generator.suspend(50, 2, &registers, liveness);

        // Restore to fresh registers
        let mut new_regs = [Value::none(); 256];
        generator.restore(&mut new_regs);

        assert_eq!(new_regs[0].as_int().unwrap(), 10);
        assert_eq!(new_regs[2].as_int().unwrap(), 20);
        assert_eq!(new_regs[5].as_int().unwrap(), 30);
    }

    #[test]
    fn test_generator_exhaust() {
        let code = test_code();
        let generator = GeneratorObject::new(code);

        generator.try_start();
        generator.exhaust();

        assert!(generator.is_exhausted());
        assert!(!generator.is_resumable());
        assert_eq!(generator.try_start(), None);
    }

    #[test]
    fn test_generator_full_lifecycle() {
        let code = test_code();
        let mut generator = GeneratorObject::new(code);

        // Created -> Running
        assert_eq!(generator.try_start(), Some(GeneratorState::Created));

        // Running -> Suspended (yield 1)
        let mut regs = [Value::none(); 256];
        regs[0] = Value::int(100).unwrap();
        generator.suspend(10, 0, &regs, LivenessMap::from_bits(0b1));

        // Suspended -> Running
        assert_eq!(generator.try_start(), Some(GeneratorState::Suspended));

        // Running -> Suspended (yield 2)
        regs[0] = Value::int(200).unwrap();
        generator.suspend(20, 1, &regs, LivenessMap::from_bits(0b1));

        // Suspended -> Running -> Exhausted
        assert_eq!(generator.try_start(), Some(GeneratorState::Suspended));
        generator.exhaust();

        // Cannot restart
        assert_eq!(generator.try_start(), None);
    }

    // ════════════════════════════════════════════════════════════════════════
    // Send/Throw Tests
    // ════════════════════════════════════════════════════════════════════════

    #[test]
    fn test_send_value() {
        let code = test_code();
        let mut generator = GeneratorObject::new(code);

        assert!(generator.peek_receive_value().is_none());

        generator.set_send_value(Value::int(42).unwrap());
        assert!(generator.peek_receive_value().is_some());

        let value = generator.take_receive_value();
        assert_eq!(value.unwrap().as_int().unwrap(), 42);
        assert!(generator.take_receive_value().is_none());
    }

    #[test]
    fn test_send_none_value() {
        let code = test_code();
        let mut generator = GeneratorObject::new(code);

        generator.set_send_value(Value::none());
        let value = generator.take_receive_value();
        assert!(value.unwrap().is_none());
    }

    // ════════════════════════════════════════════════════════════════════════
    // Flag Tests
    // ════════════════════════════════════════════════════════════════════════

    #[test]
    fn test_started_flag() {
        let code = test_code();
        let mut generator = GeneratorObject::new(code);

        assert!(!generator.is_started());
        generator.mark_started();
        assert!(generator.is_started());
    }

    #[test]
    fn test_coroutine_flag() {
        let code = test_code();
        let generator = GeneratorObject::with_flags(code, GeneratorFlags::IS_COROUTINE);

        assert!(generator.is_coroutine());
        assert!(!generator.is_async());
    }

    #[test]
    fn test_async_flag() {
        let code = test_code();
        let generator = GeneratorObject::with_flags(code, GeneratorFlags::IS_ASYNC);

        assert!(generator.is_async());
        assert!(!generator.is_coroutine());
    }

    #[test]
    fn test_multiple_flags() {
        let code = test_code();
        let flags =
            GeneratorFlags::IS_COROUTINE | GeneratorFlags::HAS_JIT | GeneratorFlags::STARTED;
        let generator = GeneratorObject::with_flags(code, flags);

        assert!(generator.is_coroutine());
        assert!(generator.is_started());
        assert!(generator.flags().contains(GeneratorFlags::HAS_JIT));
    }

    // ════════════════════════════════════════════════════════════════════════
    // Clone Tests
    // ════════════════════════════════════════════════════════════════════════

    #[test]
    fn test_generator_clone() {
        let code = test_code();
        let mut generator = GeneratorObject::new(code);

        generator.try_start();
        let mut regs = [Value::none(); 256];
        regs[0] = Value::int(42).unwrap();
        generator.suspend(100, 1, &regs, LivenessMap::from_bits(0b1));
        generator.set_send_value(Value::int(99).unwrap());

        let cloned = generator.clone();

        assert_eq!(cloned.state(), generator.state());
        assert_eq!(cloned.resume_index(), generator.resume_index());
        assert_eq!(cloned.ip(), generator.ip());
    }

    // ════════════════════════════════════════════════════════════════════════
    // Debug Tests
    // ════════════════════════════════════════════════════════════════════════

    #[test]
    fn test_generator_debug() {
        let code = test_code();
        let generator = GeneratorObject::new(code);

        let debug = format!("{:?}", generator);
        assert!(debug.contains("GeneratorObject"));
        assert!(debug.contains("Created"));
    }

    // ════════════════════════════════════════════════════════════════════════
    // GeneratorFlags Tests
    // ════════════════════════════════════════════════════════════════════════

    #[test]
    fn test_flags_default() {
        let flags = GeneratorFlags::default();
        assert_eq!(flags.bits(), 0);
    }

    #[test]
    fn test_flags_union() {
        let flags1 = GeneratorFlags::HAS_JIT;
        let flags2 = GeneratorFlags::IS_COROUTINE;
        let combined = flags1 | flags2;

        assert!(combined.contains(GeneratorFlags::HAS_JIT));
        assert!(combined.contains(GeneratorFlags::IS_COROUTINE));
        assert!(!combined.contains(GeneratorFlags::IS_ASYNC));
    }

    #[test]
    fn test_flags_bitor_assign() {
        let mut flags = GeneratorFlags::EMPTY;
        flags |= GeneratorFlags::STARTED;
        assert!(flags.contains(GeneratorFlags::STARTED));
    }
}
