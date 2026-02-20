//! Deoptimization support for Template JIT.
//!
//! Deoptimization ("deopt") allows the JIT to fall back to the interpreter
//! when assumptions are violated. This includes:
//! - Type guard failures
//! - Overflow on arithmetic
//! - Uncommon path execution
//!
//! # Deopt Flow
//! 1. Guard fails in JIT code
//! 2. Jump to deopt stub
//! 3. Save JIT state to DeoptInfo
//! 4. Reconstruct interpreter frame
//! 5. Resume interpreter execution

use super::frame::FrameLayout;
use crate::backend::x64::{Assembler, Gpr, Label};
use crate::runtime::ExitReason;

// =============================================================================
// Deoptimization Reason
// =============================================================================

/// Why deoptimization occurred.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum DeoptReason {
    /// Type guard failed (unexpected type).
    TypeGuardFailed = 0,
    /// Integer overflow in arithmetic.
    IntegerOverflow = 1,
    /// Division by zero.
    DivisionByZero = 2,
    /// Null pointer dereference.
    NullPointer = 3,
    /// Array bounds check failed.
    BoundsCheckFailed = 4,
    /// Inline cache miss.
    InlineCacheMiss = 5,
    /// Stack overflow.
    StackOverflow = 6,
    /// Uncommon trap (cold path).
    UncommonTrap = 7,
    /// Debugger breakpoint.
    Breakpoint = 8,
}

impl DeoptReason {
    /// Get a human-readable name.
    pub const fn as_str(self) -> &'static str {
        match self {
            DeoptReason::TypeGuardFailed => "TypeGuardFailed",
            DeoptReason::IntegerOverflow => "IntegerOverflow",
            DeoptReason::DivisionByZero => "DivisionByZero",
            DeoptReason::NullPointer => "NullPointer",
            DeoptReason::BoundsCheckFailed => "BoundsCheckFailed",
            DeoptReason::InlineCacheMiss => "InlineCacheMiss",
            DeoptReason::StackOverflow => "StackOverflow",
            DeoptReason::UncommonTrap => "UncommonTrap",
            DeoptReason::Breakpoint => "Breakpoint",
        }
    }
}

// =============================================================================
// Deoptimization Info
// =============================================================================

/// Information about a deoptimization point.
#[derive(Debug, Clone)]
pub struct DeoptInfo {
    /// Bytecode offset at deopt point.
    pub bc_offset: u32,
    /// Native code offset of the deopt stub.
    pub native_offset: u32,
    /// Reason for deoptimization.
    pub reason: DeoptReason,
    /// Index of the deopt stub in the code.
    pub stub_index: u16,
}

impl DeoptInfo {
    /// Create a new deopt info.
    pub fn new(bc_offset: u32, native_offset: u32, reason: DeoptReason, stub_index: u16) -> Self {
        DeoptInfo {
            bc_offset,
            native_offset,
            reason,
            stub_index,
        }
    }
}

// =============================================================================
// Deopt Stub Generator
// =============================================================================

/// Generates deoptimization stubs at the end of compiled code.
pub struct DeoptStubGenerator {
    /// Pending deopt points to generate stubs for.
    pending_deopts: Vec<PendingDeopt>,
}

/// A pending deoptimization point.
struct PendingDeopt {
    /// Label to bind for this deopt.
    label: Label,
    /// Bytecode offset.
    bc_offset: u32,
    /// Reason for deopt.
    reason: DeoptReason,
}

impl DeoptStubGenerator {
    /// Create a new deopt stub generator.
    pub fn new() -> Self {
        DeoptStubGenerator {
            pending_deopts: Vec::new(),
        }
    }

    /// Register a deoptimization point.
    pub fn register_deopt(&mut self, label: Label, bc_offset: u32, reason: DeoptReason) {
        self.pending_deopts.push(PendingDeopt {
            label,
            bc_offset,
            reason,
        });
    }

    /// Get the number of pending deopts.
    pub fn deopt_count(&self) -> usize {
        self.pending_deopts.len()
    }

    /// Emit all deopt stubs and return deopt info.
    ///
    /// This should be called after all other code generation is complete.
    /// Stubs are emitted at the end of the code buffer.
    pub fn emit_stubs(self, asm: &mut Assembler, frame: &FrameLayout) -> Vec<DeoptInfo> {
        let mut deopt_infos = Vec::with_capacity(self.pending_deopts.len());

        for (idx, deopt) in self.pending_deopts.into_iter().enumerate() {
            // Bind the label
            asm.bind_label(deopt.label);
            let native_offset = asm.offset() as u32;

            // Create deopt info
            deopt_infos.push(DeoptInfo::new(
                deopt.bc_offset,
                native_offset,
                deopt.reason,
                idx as u16,
            ));

            // Emit deopt return:
            // 1. Encode deopt metadata in RAX as ExitReason::Deoptimize payload.
            // 2. Restore the compiled frame (same as normal epilogue).
            // 3. Return to VM dispatcher.
            let encoded = encode_deopt_exit(deopt.bc_offset, deopt.reason);
            asm.mov_ri64(Gpr::Rax, encoded as i64);

            let frame_size = frame.frame_size();
            if frame_size > 0 {
                asm.add_ri(Gpr::Rsp, frame_size);
            }
            asm.pop(Gpr::Rbp);

            let saved_regs: Vec<Gpr> = frame.saved_regs.iter().collect();
            for reg in saved_regs.into_iter().rev() {
                asm.pop(reg);
            }

            asm.ret();
        }

        deopt_infos
    }
}

impl Default for DeoptStubGenerator {
    fn default() -> Self {
        Self::new()
    }
}

#[inline]
fn encode_deopt_exit(bc_offset: u32, reason: DeoptReason) -> u64 {
    // VM decodes:
    // - low 8 bits: ExitReason
    // - next 8 bits: DeoptReason
    // - next 24 bits: bytecode offset
    debug_assert!(bc_offset <= 0x00FF_FFFF);
    let payload = (((bc_offset as u64) & 0x00FF_FFFF) << 8) | (reason as u8 as u64);
    (ExitReason::Deoptimize as u64) | (payload << 8)
}

// =============================================================================
// Deopt State Reconstruction
// =============================================================================

/// State needed to reconstruct an interpreter frame from JIT state.
#[derive(Debug, Clone)]
pub struct DeoptState {
    /// Current bytecode offset to resume from.
    pub bc_offset: u32,
    /// Values of all bytecode registers at deopt point.
    /// Stored as raw u64 values (NaN-boxed).
    pub register_values: Vec<u64>,
    /// Whether we were in the middle of an exception handler.
    pub in_exception_handler: bool,
}

impl DeoptState {
    /// Create a new deopt state.
    pub fn new(bc_offset: u32, num_registers: usize) -> Self {
        DeoptState {
            bc_offset,
            register_values: vec![0; num_registers],
            in_exception_handler: false,
        }
    }

    /// Read register values from JIT frame.
    ///
    /// # Safety
    /// The frame pointer must be valid and point to an active JIT frame.
    pub unsafe fn read_from_frame(
        bc_offset: u32,
        frame_ptr: *const u8,
        layout: &FrameLayout,
    ) -> Self {
        let num_regs = layout.num_registers as usize;
        let mut state = DeoptState::new(bc_offset, num_regs);

        for i in 0..num_regs {
            let slot = layout.register_slot(i as u16);
            // Calculate actual address
            // SAFETY: caller guarantees frame_ptr is valid
            let addr = unsafe { frame_ptr.offset(slot.disp as isize) as *const u64 };
            state.register_values[i] = unsafe { std::ptr::read(addr) };
        }

        state
    }
}

// =============================================================================
// Deopt Counter (for preventing infinite deopt loops)
// =============================================================================

/// Tracks deoptimization counts per function for recompilation decisions.
#[derive(Debug, Default)]
pub struct DeoptCounter {
    /// Total number of deopts.
    pub total: u32,
    /// Counts per reason.
    pub by_reason: [u32; 9],
}

impl DeoptCounter {
    /// Maximum deopts before we stop recompiling.
    pub const MAX_DEOPTS: u32 = 10;

    /// Create a new counter.
    pub fn new() -> Self {
        Self::default()
    }

    /// Record a deopt and return whether we should recompile.
    pub fn record(&mut self, reason: DeoptReason) -> bool {
        self.total += 1;
        self.by_reason[reason as usize] += 1;

        // If we're deopting too much, stop recompiling
        self.total < Self::MAX_DEOPTS
    }

    /// Check if we should give up on JIT compilation for this function.
    pub fn should_abandon_jit(&self) -> bool {
        self.total >= Self::MAX_DEOPTS
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_deopt_reason() {
        assert_eq!(DeoptReason::TypeGuardFailed.as_str(), "TypeGuardFailed");
        assert_eq!(DeoptReason::IntegerOverflow as u8, 1);
    }

    #[test]
    fn test_deopt_info() {
        let info = DeoptInfo::new(100, 200, DeoptReason::TypeGuardFailed, 0);
        assert_eq!(info.bc_offset, 100);
        assert_eq!(info.native_offset, 200);
        assert_eq!(info.reason, DeoptReason::TypeGuardFailed);
    }

    #[test]
    fn test_deopt_stub_generator() {
        let mut generator = DeoptStubGenerator::new();
        // Use the assembler to create labels
        let mut asm = crate::backend::x64::Assembler::new();
        let label1 = asm.create_label();
        let label2 = asm.create_label();

        generator.register_deopt(label1, 42, DeoptReason::TypeGuardFailed);
        generator.register_deopt(label2, 84, DeoptReason::IntegerOverflow);

        assert_eq!(generator.deopt_count(), 2);

        let frame = FrameLayout::minimal(2);
        let infos = generator.emit_stubs(&mut asm, &frame);
        assert_eq!(infos.len(), 2);
        assert_eq!(infos[0].bc_offset, 42);
        assert_eq!(infos[1].bc_offset, 84);
    }

    #[test]
    fn test_encode_deopt_exit_layout() {
        let encoded = encode_deopt_exit(0x123456, DeoptReason::InlineCacheMiss);
        assert_eq!((encoded & 0xFF) as u8, ExitReason::Deoptimize as u8);
        assert_eq!(
            ((encoded >> 8) & 0xFF) as u8,
            DeoptReason::InlineCacheMiss as u8
        );
        assert_eq!(((encoded >> 16) & 0x00FF_FFFF) as u32, 0x123456);
    }

    #[test]
    fn test_deopt_counter() {
        let mut counter = DeoptCounter::new();

        // Should allow recompilation initially
        assert!(counter.record(DeoptReason::TypeGuardFailed));
        assert_eq!(counter.total, 1);

        // After many deopts, should give up
        for _ in 0..15 {
            counter.record(DeoptReason::TypeGuardFailed);
        }

        assert!(counter.should_abandon_jit());
    }

    #[test]
    fn test_deopt_state() {
        let state = DeoptState::new(100, 8);
        assert_eq!(state.bc_offset, 100);
        assert_eq!(state.register_values.len(), 8);
        assert!(!state.in_exception_handler);
    }
}
