//! Template JIT opcode templates - modular organization.
//!
//! Templates are organized by opcode category for maintainability:
//! - `values`: Load constants (None, True, False, integers, floats)
//! - `memory`: Load/store locals, globals, closures
//! - `arithmetic`: Integer, float, and generic arithmetic
//! - `comparison`: Comparison and identity operators
//! - `bitwise`: Bitwise and logical operations
//! - `control`: Jumps, branches, returns
//! - `objects`: Attribute and item access, iterators
//! - `calls`: Function calls and closures
//! - `containers`: List, tuple, dict, set construction
//!
//! Each template implements fast type-specialized code with deoptimization
//! guards for type mismatches.

pub mod arithmetic;
pub mod bitwise;
pub mod calls;
pub mod comparison;
pub mod containers;
pub mod control;
pub mod guards;
pub mod memory;
pub mod objects;
pub mod values;

use crate::backend::x64::{Assembler, Label};
use crate::tier1::frame::{FrameLayout, JitCallingConvention, RegisterAssignment};

// Re-export commonly used types
pub use arithmetic::*;
pub use bitwise::*;
pub use calls::*;
pub use comparison::*;
pub use containers::*;
pub use control::*;
pub use guards::*;
pub use memory::*;
pub use objects::*;
pub use values::*;

// =============================================================================
// Value Type Tags (matching NaN-boxing from prism_core)
// =============================================================================

/// Tag bits for NaN-boxed values (from prism_core::value).
pub mod value_tags {
    /// Mask for the quiet NaN bits (top 13 bits set for NaN).
    pub const QNAN_BITS: u64 = 0x7FF8_0000_0000_0000;

    /// Tag for integer values (48-bit signed).
    pub const INT_TAG: u64 = 0x0001_0000_0000_0000;

    /// Tag for object pointers.
    pub const OBJECT_TAG: u64 = 0x0002_0000_0000_0000;

    /// Tag for None value.
    pub const NONE_TAG: u64 = 0x0003_0000_0000_0000;

    /// Tag for True boolean.
    pub const TRUE_TAG: u64 = 0x0004_0000_0000_0000;

    /// Tag for False boolean.
    pub const FALSE_TAG: u64 = 0x0005_0000_0000_0000;

    /// Payload mask (lower 48 bits).
    pub const PAYLOAD_MASK: u64 = 0x0000_FFFF_FFFF_FFFF;

    /// Tag mask (upper 16 bits).
    pub const TAG_MASK: u64 = 0xFFFF_0000_0000_0000;

    /// Complete None value.
    #[inline]
    pub const fn none_value() -> u64 {
        QNAN_BITS | NONE_TAG
    }

    /// Complete True value.
    #[inline]
    pub const fn true_value() -> u64 {
        QNAN_BITS | TRUE_TAG
    }

    /// Complete False value.
    #[inline]
    pub const fn false_value() -> u64 {
        QNAN_BITS | FALSE_TAG
    }

    /// Tag check value for integers (upper 16 bits).
    #[inline]
    pub const fn int_tag_check() -> u16 {
        ((QNAN_BITS | INT_TAG) >> 48) as u16
    }

    /// Box an integer value.
    #[inline]
    pub const fn box_int(value: i64) -> u64 {
        (QNAN_BITS | INT_TAG) | ((value as u64) & PAYLOAD_MASK)
    }
}

// =============================================================================
// Template Context
// =============================================================================

/// Context passed to all templates during code generation.
pub struct TemplateContext<'a> {
    /// The assembler to emit code to.
    pub asm: &'a mut Assembler,
    /// Frame layout for the current function.
    pub frame: &'a FrameLayout,
    /// Register assignments.
    pub regs: RegisterAssignment,
    /// JIT calling convention.
    pub cc: JitCallingConvention,
    /// Labels for deoptimization stubs.
    pub deopt_labels: Vec<Label>,
    /// Current bytecode offset (for debugging/deopt).
    pub bc_offset: usize,
}

impl<'a> TemplateContext<'a> {
    /// Create a new template context.
    #[inline]
    pub fn new(asm: &'a mut Assembler, frame: &'a FrameLayout) -> Self {
        TemplateContext {
            asm,
            frame,
            regs: RegisterAssignment::host(),
            cc: JitCallingConvention::host(),
            deopt_labels: Vec::new(),
            bc_offset: 0,
        }
    }

    /// Set the current bytecode offset.
    #[inline]
    pub fn set_bc_offset(&mut self, offset: usize) {
        self.bc_offset = offset;
    }

    /// Create a deoptimization label and return its index.
    #[inline]
    pub fn create_deopt_label(&mut self) -> usize {
        let idx = self.deopt_labels.len();
        let label = self.asm.create_label();
        self.deopt_labels.push(label);
        idx
    }

    /// Get a deoptimization label by index.
    #[inline]
    pub fn deopt_label(&self, idx: usize) -> Label {
        self.deopt_labels[idx]
    }
}

// =============================================================================
// Template Trait
// =============================================================================

/// Trait for opcode templates.
///
/// Each template generates native code for a specific bytecode operation.
/// Templates are designed to be:
/// - Fast to generate (minimal analysis)
/// - Type-specialized where profitable
/// - Deoptimization-capable for type mismatches
pub trait OpcodeTemplate {
    /// Emit the native code for this opcode.
    fn emit(&self, ctx: &mut TemplateContext);

    /// Get the estimated code size for this template (for allocation).
    fn estimated_size(&self) -> usize {
        32 // Default estimate
    }
}

// =============================================================================
// Template Registry
// =============================================================================

/// Centralized registry of all opcode templates.
///
/// Provides O(1) dispatch to the appropriate template emitter.
#[derive(Default)]
pub struct TemplateRegistry {
    // Statistics for monitoring
    total_emitted: usize,
    bytes_generated: usize,
}

impl TemplateRegistry {
    /// Create a new template registry.
    #[inline]
    pub const fn new() -> Self {
        TemplateRegistry {
            total_emitted: 0,
            bytes_generated: 0,
        }
    }

    /// Get statistics about template usage.
    #[inline]
    pub fn stats(&self) -> (usize, usize) {
        (self.total_emitted, self.bytes_generated)
    }

    /// Record template emission for statistics.
    #[inline]
    pub fn record_emission(&mut self, bytes: usize) {
        self.total_emitted += 1;
        self.bytes_generated += bytes;
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::x64::Assembler;
    use crate::tier1::frame::FrameLayout;

    #[test]
    fn test_template_context_creation() {
        let mut asm = Assembler::new();
        let frame = FrameLayout::minimal(4);
        let ctx = TemplateContext::new(&mut asm, &frame);

        assert_eq!(ctx.bc_offset, 0);
        assert!(ctx.deopt_labels.is_empty());
    }

    #[test]
    fn test_deopt_label_creation() {
        let mut asm = Assembler::new();
        let frame = FrameLayout::minimal(4);
        let mut ctx = TemplateContext::new(&mut asm, &frame);

        let idx1 = ctx.create_deopt_label();
        let idx2 = ctx.create_deopt_label();

        assert_eq!(idx1, 0);
        assert_eq!(idx2, 1);
        assert_eq!(ctx.deopt_labels.len(), 2);
    }

    #[test]
    fn test_value_tags() {
        assert_eq!(value_tags::none_value(), 0x7FFB_0000_0000_0000);
        assert_eq!(value_tags::true_value(), 0x7FFC_0000_0000_0000);
        assert_eq!(value_tags::false_value(), 0x7FFD_0000_0000_0000);
        assert_eq!(value_tags::int_tag_check(), 0x7FF9);
    }

    #[test]
    fn test_box_int() {
        let boxed = value_tags::box_int(42);
        assert_eq!(boxed & value_tags::PAYLOAD_MASK, 42);
        assert_eq!((boxed >> 48) as u16, value_tags::int_tag_check());
    }

    #[test]
    fn test_template_registry() {
        let mut registry = TemplateRegistry::new();
        assert_eq!(registry.stats(), (0, 0));

        registry.record_emission(100);
        registry.record_emission(50);
        assert_eq!(registry.stats(), (2, 150));
    }
}
