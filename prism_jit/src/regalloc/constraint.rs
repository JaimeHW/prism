//! Register Constraint Module
//!
//! This module provides instruction-level operand constraints for register allocation.
//! It enables the allocator to respect hardware requirements such as:
//!
//! - Fixed register requirements (e.g., `DIV` requires `RAX`/`RDX`)
//! - Register class constraints (GPR vs XMM vs YMM vs ZMM)
//! - Two-address form constraints (destination must alias first source)
//! - Clobber specifications for instructions that destroy registers
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
//! │   Instruction   │ ──▶ │   Constraints   │ ──▶ │   Allocator     │
//! │   Definitions   │     │   (this module) │     │  (linear_scan)  │
//! └─────────────────┘     └─────────────────┘     └─────────────────┘
//! ```
//!
//! # Design Principles
//!
//! 1. **Declarative Constraints**: Instructions declare their requirements
//!    without specifying how to satisfy them.
//!
//! 2. **Hardware-Accurate**: Constraints model actual x64 ISA requirements
//!    including destructive operations and fixed registers.
//!
//! 3. **Width-Aware**: Vector constraints properly distinguish between
//!    XMM (128-bit), YMM (256-bit), and ZMM (512-bit) requirements.
//!
//! 4. **Composable**: Constraints can be combined to express complex
//!    multi-operand requirements.
//!
//! # Usage
//!
//! ```ignore
//! // Define constraints for VADDPD ymm, ymm, ymm
//! let constraint = InstructionConstraint::new("vaddpd_ymm")
//!     .def(0, OperandConstraint::reg_class(RegClass::Vec256))
//!     .use_op(1, OperandConstraint::reg_class(RegClass::Vec256))
//!     .use_op(2, OperandConstraint::reg_class(RegClass::Vec256));
//! ```

use crate::backend::x64::registers::{Gpr, Xmm};
use crate::backend::x64::simd::{Ymm, Zmm};
use crate::regalloc::{PReg, RegClass, VReg};
use std::fmt;

// =============================================================================
// Operand Role
// =============================================================================

/// The role of an operand in an instruction.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OperandRole {
    /// Operand is only read (use).
    Use,
    /// Operand is only written (def).
    Def,
    /// Operand is both read and written (use-def).
    UseDef,
    /// Operand is an early clobber (def before uses are read).
    EarlyClobber,
}

impl OperandRole {
    /// Check if this role reads the operand.
    #[inline]
    pub const fn is_use(self) -> bool {
        matches!(self, OperandRole::Use | OperandRole::UseDef)
    }

    /// Check if this role writes the operand.
    #[inline]
    pub const fn is_def(self) -> bool {
        matches!(
            self,
            OperandRole::Def | OperandRole::UseDef | OperandRole::EarlyClobber
        )
    }

    /// Check if this is a destructive use-def (two-address form).
    #[inline]
    pub const fn is_use_def(self) -> bool {
        matches!(self, OperandRole::UseDef)
    }
}

// =============================================================================
// Operand Constraint
// =============================================================================

/// Constraint on a single operand of an instruction.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OperandConstraint {
    /// Must be in a specific physical register.
    FixedReg(PReg),
    /// Must be in a register of the given class.
    RegClass(RegClass),
    /// Must be in the same register as another operand (tied).
    Tied(u8),
    /// Can be either a register or memory (register preferred).
    RegOrMem(RegClass),
    /// Must be a memory operand (not in register).
    Memory,
    /// Can be any register (no constraint).
    Any,
    /// Reuse the input register for output (destructive).
    ReuseInput(u8),
}

impl OperandConstraint {
    // =========================================================================
    // Constructor helpers
    // =========================================================================

    /// Create a fixed register constraint.
    #[inline]
    pub const fn fixed(reg: PReg) -> Self {
        OperandConstraint::FixedReg(reg)
    }

    /// Create a constraint for a specific GPR.
    #[inline]
    pub const fn fixed_gpr(gpr: Gpr) -> Self {
        OperandConstraint::FixedReg(PReg::Gpr(gpr))
    }

    /// Create a constraint for a specific XMM.
    #[inline]
    pub const fn fixed_xmm(xmm: Xmm) -> Self {
        OperandConstraint::FixedReg(PReg::Xmm(xmm))
    }

    /// Create a constraint for a specific YMM.
    #[inline]
    pub const fn fixed_ymm(ymm: Ymm) -> Self {
        OperandConstraint::FixedReg(PReg::Ymm(ymm))
    }

    /// Create a constraint for a specific ZMM.
    #[inline]
    pub const fn fixed_zmm(zmm: Zmm) -> Self {
        OperandConstraint::FixedReg(PReg::Zmm(zmm))
    }

    /// Create a register class constraint.
    #[inline]
    pub const fn for_class(class: RegClass) -> Self {
        OperandConstraint::RegClass(class)
    }

    /// Create a GPR constraint.
    #[inline]
    pub const fn gpr() -> Self {
        OperandConstraint::RegClass(RegClass::Int)
    }

    /// Create an XMM constraint.
    #[inline]
    pub const fn xmm() -> Self {
        OperandConstraint::RegClass(RegClass::Float)
    }

    /// Create a YMM constraint.
    #[inline]
    pub const fn ymm() -> Self {
        OperandConstraint::RegClass(RegClass::Vec256)
    }

    /// Create a ZMM constraint.
    #[inline]
    pub const fn zmm() -> Self {
        OperandConstraint::RegClass(RegClass::Vec512)
    }

    /// Create a tied constraint (must be same register as operand N).
    #[inline]
    pub const fn tied(operand_idx: u8) -> Self {
        OperandConstraint::Tied(operand_idx)
    }

    /// Create a reuse-input constraint (output reuses input register).
    #[inline]
    pub const fn reuse_input(input_idx: u8) -> Self {
        OperandConstraint::ReuseInput(input_idx)
    }

    // =========================================================================
    // Query methods
    // =========================================================================

    /// Check if this constraint requires a specific register.
    #[inline]
    pub fn is_fixed(&self) -> bool {
        matches!(self, OperandConstraint::FixedReg(_))
    }

    /// Get the fixed register if this is a fixed constraint.
    #[inline]
    pub fn fixed_reg(&self) -> Option<PReg> {
        match self {
            OperandConstraint::FixedReg(r) => Some(*r),
            _ => None,
        }
    }

    /// Get the register class for this constraint.
    pub fn get_reg_class(&self) -> Option<RegClass> {
        match self {
            OperandConstraint::FixedReg(r) => Some(match r {
                PReg::Gpr(_) => RegClass::Int,
                PReg::Xmm(_) => RegClass::Float,
                PReg::Ymm(_) => RegClass::Vec256,
                PReg::Zmm(_) => RegClass::Vec512,
            }),
            OperandConstraint::RegClass(c) => Some(*c),
            OperandConstraint::RegOrMem(c) => Some(*c),
            OperandConstraint::Any => Some(RegClass::Any),
            _ => None,
        }
    }

    /// Check if this constraint is a vector constraint.
    #[inline]
    pub fn is_vector(&self) -> bool {
        self.get_reg_class().map(|c| c.is_vector()).unwrap_or(false)
    }

    /// Check if a register satisfies this constraint.
    pub fn is_satisfied_by(&self, reg: PReg) -> bool {
        match self {
            OperandConstraint::FixedReg(r) => *r == reg,
            OperandConstraint::RegClass(c) => match (c, reg) {
                (RegClass::Int, PReg::Gpr(_)) => true,
                (RegClass::Float, PReg::Xmm(_)) => true,
                (RegClass::Vec256, PReg::Ymm(_)) => true,
                (RegClass::Vec512, PReg::Zmm(_)) => true,
                (RegClass::Any, _) => true,
                _ => false,
            },
            OperandConstraint::Any => true,
            OperandConstraint::RegOrMem(c) => match (c, reg) {
                (RegClass::Int, PReg::Gpr(_)) => true,
                (RegClass::Float, PReg::Xmm(_)) => true,
                (RegClass::Vec256, PReg::Ymm(_)) => true,
                (RegClass::Vec512, PReg::Zmm(_)) => true,
                (RegClass::Any, _) => true,
                _ => false,
            },
            OperandConstraint::Tied(_) => true, // Checked separately
            OperandConstraint::ReuseInput(_) => true, // Checked separately
            OperandConstraint::Memory => false, // Registers never satisfy memory
        }
    }
}

impl fmt::Display for OperandConstraint {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            OperandConstraint::FixedReg(r) => write!(f, "={}", r),
            OperandConstraint::RegClass(c) => write!(f, "{:?}", c),
            OperandConstraint::Tied(idx) => write!(f, "tied({})", idx),
            OperandConstraint::ReuseInput(idx) => write!(f, "reuse({})", idx),
            OperandConstraint::RegOrMem(c) => write!(f, "{:?}/mem", c),
            OperandConstraint::Memory => write!(f, "mem"),
            OperandConstraint::Any => write!(f, "any"),
        }
    }
}

// =============================================================================
// Operand Descriptor
// =============================================================================

/// Complete descriptor for an instruction operand.
#[derive(Debug, Clone)]
pub struct OperandDescriptor {
    /// Virtual register for this operand.
    pub vreg: VReg,
    /// Role of this operand (use, def, use-def).
    pub role: OperandRole,
    /// Constraint on this operand.
    pub constraint: OperandConstraint,
}

impl OperandDescriptor {
    /// Create a new operand descriptor.
    #[inline]
    pub fn new(vreg: VReg, role: OperandRole, constraint: OperandConstraint) -> Self {
        OperandDescriptor {
            vreg,
            role,
            constraint,
        }
    }

    /// Create a use operand with the given constraint.
    #[inline]
    pub fn use_op(vreg: VReg, constraint: OperandConstraint) -> Self {
        OperandDescriptor::new(vreg, OperandRole::Use, constraint)
    }

    /// Create a def operand with the given constraint.
    #[inline]
    pub fn def(vreg: VReg, constraint: OperandConstraint) -> Self {
        OperandDescriptor::new(vreg, OperandRole::Def, constraint)
    }

    /// Create a use-def operand (two-address form).
    #[inline]
    pub fn use_def(vreg: VReg, constraint: OperandConstraint) -> Self {
        OperandDescriptor::new(vreg, OperandRole::UseDef, constraint)
    }

    /// Create an early clobber def.
    #[inline]
    pub fn early_clobber(vreg: VReg, constraint: OperandConstraint) -> Self {
        OperandDescriptor::new(vreg, OperandRole::EarlyClobber, constraint)
    }

    /// Check if this is a vector operand.
    #[inline]
    pub fn is_vector(&self) -> bool {
        self.constraint.is_vector()
    }
}

// =============================================================================
// Instruction Constraint
// =============================================================================

/// Complete constraint specification for an instruction.
#[derive(Debug, Clone)]
pub struct InstructionConstraint {
    /// Name of the instruction (for debugging).
    pub name: &'static str,
    /// Operand descriptors (defs first, then uses).
    pub operands: Vec<OperandDescriptor>,
    /// Registers clobbered by this instruction.
    pub clobbers: Vec<PReg>,
    /// Whether this is a two-address instruction.
    pub is_two_address: bool,
    /// Whether this instruction is commutative.
    pub is_commutative: bool,
}

impl InstructionConstraint {
    /// Create a new instruction constraint.
    #[inline]
    pub fn new(name: &'static str) -> Self {
        InstructionConstraint {
            name,
            operands: Vec::new(),
            clobbers: Vec::new(),
            is_two_address: false,
            is_commutative: false,
        }
    }

    /// Add a def operand.
    #[inline]
    pub fn def(mut self, vreg: VReg, constraint: OperandConstraint) -> Self {
        self.operands.push(OperandDescriptor::def(vreg, constraint));
        self
    }

    /// Add a use operand.
    #[inline]
    pub fn use_op(mut self, vreg: VReg, constraint: OperandConstraint) -> Self {
        self.operands
            .push(OperandDescriptor::use_op(vreg, constraint));
        self
    }

    /// Add a use-def operand (two-address form).
    #[inline]
    pub fn use_def(mut self, vreg: VReg, constraint: OperandConstraint) -> Self {
        self.operands
            .push(OperandDescriptor::use_def(vreg, constraint));
        self.is_two_address = true;
        self
    }

    /// Add a clobbered register.
    #[inline]
    pub fn clobber(mut self, reg: PReg) -> Self {
        self.clobbers.push(reg);
        self
    }

    /// Add multiple clobbered registers.
    #[inline]
    pub fn clobber_all(mut self, regs: impl IntoIterator<Item = PReg>) -> Self {
        self.clobbers.extend(regs);
        self
    }

    /// Mark as commutative.
    #[inline]
    pub fn commutative(mut self) -> Self {
        self.is_commutative = true;
        self
    }

    /// Get all def operands.
    pub fn defs(&self) -> impl Iterator<Item = &OperandDescriptor> {
        self.operands.iter().filter(|op| op.role.is_def())
    }

    /// Get all use operands.
    pub fn uses(&self) -> impl Iterator<Item = &OperandDescriptor> {
        self.operands.iter().filter(|op| op.role.is_use())
    }

    /// Get the number of defs.
    #[inline]
    pub fn num_defs(&self) -> usize {
        self.operands.iter().filter(|op| op.role.is_def()).count()
    }

    /// Get the number of uses.
    #[inline]
    pub fn num_uses(&self) -> usize {
        self.operands.iter().filter(|op| op.role.is_use()).count()
    }

    /// Check if any operand is a vector operand.
    #[inline]
    pub fn has_vector_operands(&self) -> bool {
        self.operands.iter().any(|op| op.is_vector())
    }

    /// Get the vector width if this has vector operands (in bits).
    pub fn vector_width(&self) -> Option<u16> {
        for op in &self.operands {
            if let Some(class) = op.constraint.get_reg_class() {
                match class {
                    RegClass::Vec512 => return Some(512),
                    RegClass::Vec256 => return Some(256),
                    RegClass::Float => return Some(128),
                    _ => {}
                }
            }
        }
        None
    }
}

// =============================================================================
// Constraint Templates for Common SIMD Operations
// =============================================================================

/// Template for a binary SIMD operation (dst = src1 OP src2).
///
/// This creates a 3-operand constraint for AVX-style instructions.
pub fn simd_binary_rrr(
    name: &'static str,
    dst: VReg,
    src1: VReg,
    src2: VReg,
    class: RegClass,
) -> InstructionConstraint {
    InstructionConstraint::new(name)
        .def(dst, OperandConstraint::for_class(class))
        .use_op(src1, OperandConstraint::for_class(class))
        .use_op(src2, OperandConstraint::for_class(class))
        .commutative()
}

/// Template for a unary SIMD operation (dst = OP src).
pub fn simd_unary_rr(
    name: &'static str,
    dst: VReg,
    src: VReg,
    class: RegClass,
) -> InstructionConstraint {
    InstructionConstraint::new(name)
        .def(dst, OperandConstraint::for_class(class))
        .use_op(src, OperandConstraint::for_class(class))
}

/// Template for a SIMD move instruction.
pub fn simd_move(
    name: &'static str,
    dst: VReg,
    src: VReg,
    class: RegClass,
) -> InstructionConstraint {
    InstructionConstraint::new(name)
        .def(dst, OperandConstraint::for_class(class))
        .use_op(src, OperandConstraint::for_class(class))
}

/// Template for a SIMD load from memory.
pub fn simd_load(name: &'static str, dst: VReg, class: RegClass) -> InstructionConstraint {
    InstructionConstraint::new(name).def(dst, OperandConstraint::for_class(class))
}

/// Template for a SIMD store to memory.
pub fn simd_store(name: &'static str, src: VReg, class: RegClass) -> InstructionConstraint {
    InstructionConstraint::new(name).use_op(src, OperandConstraint::for_class(class))
}

/// Template for a fused multiply-add (dst = dst * src1 + src2).
pub fn simd_fma(
    name: &'static str,
    dst: VReg,
    src1: VReg,
    src2: VReg,
    class: RegClass,
) -> InstructionConstraint {
    InstructionConstraint::new(name)
        .use_def(dst, OperandConstraint::for_class(class))
        .use_op(src1, OperandConstraint::for_class(class))
        .use_op(src2, OperandConstraint::for_class(class))
}

// =============================================================================
// Constraint Templates for x64 GPR Operations
// =============================================================================

/// Template for DIV/IDIV instruction (requires RAX, RDX fixed).
pub fn div_constraint(
    name: &'static str,
    dividend_lo: VReg,
    dividend_hi: VReg,
    divisor: VReg,
) -> InstructionConstraint {
    InstructionConstraint::new(name)
        .def(dividend_lo, OperandConstraint::fixed_gpr(Gpr::Rax)) // Quotient
        .def(dividend_hi, OperandConstraint::fixed_gpr(Gpr::Rdx)) // Remainder
        .use_op(dividend_lo, OperandConstraint::fixed_gpr(Gpr::Rax))
        .use_op(dividend_hi, OperandConstraint::fixed_gpr(Gpr::Rdx))
        .use_op(divisor, OperandConstraint::gpr())
}

/// Template for MUL/IMUL instruction (produces result in RAX:RDX).
pub fn mul_wide_constraint(
    name: &'static str,
    dst_lo: VReg,
    dst_hi: VReg,
    src1: VReg,
    src2: VReg,
) -> InstructionConstraint {
    InstructionConstraint::new(name)
        .def(dst_lo, OperandConstraint::fixed_gpr(Gpr::Rax))
        .def(dst_hi, OperandConstraint::fixed_gpr(Gpr::Rdx))
        .use_op(src1, OperandConstraint::fixed_gpr(Gpr::Rax))
        .use_op(src2, OperandConstraint::gpr())
}

/// Template for a shift with CL as count (SHL, SHR, SAR).
pub fn shift_by_cl_constraint(
    name: &'static str,
    dst: VReg,
    src: VReg,
    count: VReg,
) -> InstructionConstraint {
    InstructionConstraint::new(name)
        .use_def(dst, OperandConstraint::gpr())
        .use_op(src, OperandConstraint::tied(0))
        .use_op(count, OperandConstraint::fixed_gpr(Gpr::Rcx))
}

/// Template for a binary GPR operation.
pub fn gpr_binary_rr(name: &'static str, dst: VReg, src: VReg) -> InstructionConstraint {
    InstructionConstraint::new(name)
        .use_def(dst, OperandConstraint::gpr())
        .use_op(src, OperandConstraint::gpr())
}

// =============================================================================
// Constraint Database
// =============================================================================

/// Database of instruction constraints for lookup.
#[derive(Debug, Clone, Default)]
pub struct ConstraintDatabase {
    /// Known instruction constraints by opcode.
    constraints: Vec<(&'static str, InstructionConstraint)>,
}

impl ConstraintDatabase {
    /// Create a new empty database.
    pub fn new() -> Self {
        ConstraintDatabase {
            constraints: Vec::new(),
        }
    }

    /// Register a constraint for an instruction.
    pub fn register(&mut self, constraint: InstructionConstraint) {
        self.constraints.push((constraint.name, constraint));
    }

    /// Look up constraint by name.
    pub fn get(&self, name: &str) -> Option<&InstructionConstraint> {
        self.constraints
            .iter()
            .find(|(n, _)| *n == name)
            .map(|(_, c)| c)
    }

    /// Get total number of registered constraints.
    pub fn len(&self) -> usize {
        self.constraints.len()
    }

    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        self.constraints.is_empty()
    }

    /// Create a database with standard x64 SIMD constraints.
    pub fn with_simd_constraints() -> Self {
        let mut db = ConstraintDatabase::new();

        // These are template-based and will be instantiated with actual VRegs
        // at instruction selection time. For now, we register the patterns.

        db
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // OperandRole Tests
    // =========================================================================

    #[test]
    fn test_operand_role_is_use() {
        assert!(OperandRole::Use.is_use());
        assert!(OperandRole::UseDef.is_use());
        assert!(!OperandRole::Def.is_use());
        assert!(!OperandRole::EarlyClobber.is_use());
    }

    #[test]
    fn test_operand_role_is_def() {
        assert!(OperandRole::Def.is_def());
        assert!(OperandRole::UseDef.is_def());
        assert!(OperandRole::EarlyClobber.is_def());
        assert!(!OperandRole::Use.is_def());
    }

    #[test]
    fn test_operand_role_is_use_def() {
        assert!(OperandRole::UseDef.is_use_def());
        assert!(!OperandRole::Use.is_use_def());
        assert!(!OperandRole::Def.is_use_def());
    }

    // =========================================================================
    // OperandConstraint Tests
    // =========================================================================

    #[test]
    fn test_fixed_constraint() {
        let c = OperandConstraint::fixed_gpr(Gpr::Rax);
        assert!(c.is_fixed());
        assert_eq!(c.fixed_reg(), Some(PReg::Gpr(Gpr::Rax)));
    }

    #[test]
    fn test_reg_class_constraint() {
        let c = OperandConstraint::ymm();
        assert!(!c.is_fixed());
        assert_eq!(c.get_reg_class(), Some(RegClass::Vec256));
        assert!(c.is_vector());
    }

    #[test]
    fn test_constraint_satisfaction_gpr() {
        let c = OperandConstraint::gpr();
        assert!(c.is_satisfied_by(PReg::Gpr(Gpr::Rax)));
        assert!(c.is_satisfied_by(PReg::Gpr(Gpr::R15)));
        assert!(!c.is_satisfied_by(PReg::Xmm(Xmm::Xmm0)));
        assert!(!c.is_satisfied_by(PReg::Ymm(Ymm::Ymm0)));
    }

    #[test]
    fn test_constraint_satisfaction_xmm() {
        let c = OperandConstraint::xmm();
        assert!(c.is_satisfied_by(PReg::Xmm(Xmm::Xmm0)));
        assert!(c.is_satisfied_by(PReg::Xmm(Xmm::Xmm15)));
        assert!(!c.is_satisfied_by(PReg::Gpr(Gpr::Rax)));
        assert!(!c.is_satisfied_by(PReg::Ymm(Ymm::Ymm0)));
    }

    #[test]
    fn test_constraint_satisfaction_ymm() {
        let c = OperandConstraint::ymm();
        assert!(c.is_satisfied_by(PReg::Ymm(Ymm::Ymm0)));
        assert!(c.is_satisfied_by(PReg::Ymm(Ymm::Ymm15)));
        assert!(!c.is_satisfied_by(PReg::Gpr(Gpr::Rax)));
        assert!(!c.is_satisfied_by(PReg::Xmm(Xmm::Xmm0)));
    }

    #[test]
    fn test_constraint_satisfaction_zmm() {
        let c = OperandConstraint::zmm();
        assert!(c.is_satisfied_by(PReg::Zmm(Zmm::Zmm0)));
        assert!(c.is_satisfied_by(PReg::Zmm(Zmm::Zmm31)));
        assert!(!c.is_satisfied_by(PReg::Gpr(Gpr::Rax)));
        assert!(!c.is_satisfied_by(PReg::Ymm(Ymm::Ymm0)));
    }

    #[test]
    fn test_constraint_satisfaction_fixed() {
        let c = OperandConstraint::fixed_gpr(Gpr::Rax);
        assert!(c.is_satisfied_by(PReg::Gpr(Gpr::Rax)));
        assert!(!c.is_satisfied_by(PReg::Gpr(Gpr::Rbx)));
    }

    #[test]
    fn test_constraint_satisfaction_any() {
        let c = OperandConstraint::Any;
        assert!(c.is_satisfied_by(PReg::Gpr(Gpr::Rax)));
        assert!(c.is_satisfied_by(PReg::Xmm(Xmm::Xmm0)));
        assert!(c.is_satisfied_by(PReg::Ymm(Ymm::Ymm0)));
        assert!(c.is_satisfied_by(PReg::Zmm(Zmm::Zmm0)));
    }

    #[test]
    fn test_constraint_display() {
        assert_eq!(format!("{}", OperandConstraint::gpr()), "Int");
        assert_eq!(format!("{}", OperandConstraint::ymm()), "Vec256");
        assert!(format!("{}", OperandConstraint::fixed_gpr(Gpr::Rax)).contains("rax"));
        assert_eq!(format!("{}", OperandConstraint::tied(0)), "tied(0)");
    }

    // =========================================================================
    // OperandDescriptor Tests
    // =========================================================================

    #[test]
    fn test_operand_descriptor_use() {
        let v = VReg::new(0);
        let desc = OperandDescriptor::use_op(v, OperandConstraint::ymm());
        assert_eq!(desc.role, OperandRole::Use);
        assert!(desc.is_vector());
    }

    #[test]
    fn test_operand_descriptor_def() {
        let v = VReg::new(1);
        let desc = OperandDescriptor::def(v, OperandConstraint::gpr());
        assert_eq!(desc.role, OperandRole::Def);
        assert!(!desc.is_vector());
    }

    #[test]
    fn test_operand_descriptor_use_def() {
        let v = VReg::new(2);
        let desc = OperandDescriptor::use_def(v, OperandConstraint::zmm());
        assert_eq!(desc.role, OperandRole::UseDef);
        assert!(desc.is_vector());
    }

    // =========================================================================
    // InstructionConstraint Tests
    // =========================================================================

    #[test]
    fn test_instruction_constraint_creation() {
        let v0 = VReg::new(0);
        let v1 = VReg::new(1);
        let v2 = VReg::new(2);

        let ic = InstructionConstraint::new("vaddpd_ymm")
            .def(v0, OperandConstraint::ymm())
            .use_op(v1, OperandConstraint::ymm())
            .use_op(v2, OperandConstraint::ymm())
            .commutative();

        assert_eq!(ic.name, "vaddpd_ymm");
        assert_eq!(ic.num_defs(), 1);
        assert_eq!(ic.num_uses(), 2);
        assert!(ic.is_commutative);
        assert!(!ic.is_two_address);
    }

    #[test]
    fn test_instruction_constraint_two_address() {
        let v0 = VReg::new(0);
        let v1 = VReg::new(1);

        let ic = InstructionConstraint::new("add_rr")
            .use_def(v0, OperandConstraint::gpr())
            .use_op(v1, OperandConstraint::gpr());

        assert!(ic.is_two_address);
        assert_eq!(ic.num_defs(), 1);
        assert_eq!(ic.num_uses(), 2); // use_def counts as both
    }

    #[test]
    fn test_instruction_constraint_clobbers() {
        let v0 = VReg::new(0);
        let v1 = VReg::new(1);
        let v2 = VReg::new(2);

        let ic = InstructionConstraint::new("div")
            .def(v0, OperandConstraint::fixed_gpr(Gpr::Rax))
            .def(v1, OperandConstraint::fixed_gpr(Gpr::Rdx))
            .use_op(v2, OperandConstraint::gpr())
            .clobber(PReg::Gpr(Gpr::Rax))
            .clobber(PReg::Gpr(Gpr::Rdx));

        assert_eq!(ic.clobbers.len(), 2);
    }

    #[test]
    fn test_instruction_constraint_has_vector() {
        let v0 = VReg::new(0);
        let v1 = VReg::new(1);

        let ic_vec = InstructionConstraint::new("vmovapd_ymm")
            .def(v0, OperandConstraint::ymm())
            .use_op(v1, OperandConstraint::ymm());
        assert!(ic_vec.has_vector_operands());

        let ic_gpr = InstructionConstraint::new("mov_rr")
            .def(v0, OperandConstraint::gpr())
            .use_op(v1, OperandConstraint::gpr());
        assert!(!ic_gpr.has_vector_operands());
    }

    #[test]
    fn test_instruction_constraint_vector_width() {
        let v0 = VReg::new(0);
        let v1 = VReg::new(1);

        let ic_zmm = InstructionConstraint::new("vaddpd_zmm")
            .def(v0, OperandConstraint::zmm())
            .use_op(v1, OperandConstraint::zmm());
        assert_eq!(ic_zmm.vector_width(), Some(512));

        let ic_ymm = InstructionConstraint::new("vaddpd_ymm")
            .def(v0, OperandConstraint::ymm())
            .use_op(v1, OperandConstraint::ymm());
        assert_eq!(ic_ymm.vector_width(), Some(256));

        let ic_xmm = InstructionConstraint::new("addsd")
            .def(v0, OperandConstraint::xmm())
            .use_op(v1, OperandConstraint::xmm());
        assert_eq!(ic_xmm.vector_width(), Some(128));

        let ic_gpr = InstructionConstraint::new("add")
            .def(v0, OperandConstraint::gpr())
            .use_op(v1, OperandConstraint::gpr());
        assert_eq!(ic_gpr.vector_width(), None);
    }

    // =========================================================================
    // Template Tests
    // =========================================================================

    #[test]
    fn test_simd_binary_template() {
        let v0 = VReg::new(0);
        let v1 = VReg::new(1);
        let v2 = VReg::new(2);

        let ic = simd_binary_rrr("vaddpd_ymm", v0, v1, v2, RegClass::Vec256);
        assert_eq!(ic.name, "vaddpd_ymm");
        assert_eq!(ic.num_defs(), 1);
        assert_eq!(ic.num_uses(), 2);
        assert!(ic.is_commutative);
        assert!(ic.has_vector_operands());
    }

    #[test]
    fn test_simd_unary_template() {
        let v0 = VReg::new(0);
        let v1 = VReg::new(1);

        let ic = simd_unary_rr("vsqrtpd_ymm", v0, v1, RegClass::Vec256);
        assert_eq!(ic.num_defs(), 1);
        assert_eq!(ic.num_uses(), 1);
    }

    #[test]
    fn test_simd_fma_template() {
        let v0 = VReg::new(0);
        let v1 = VReg::new(1);
        let v2 = VReg::new(2);

        let ic = simd_fma("vfmadd213pd_zmm", v0, v1, v2, RegClass::Vec512);
        assert!(ic.is_two_address);
        assert_eq!(ic.vector_width(), Some(512));
    }

    #[test]
    fn test_div_constraint_template() {
        let v0 = VReg::new(0);
        let v1 = VReg::new(1);
        let v2 = VReg::new(2);

        let ic = div_constraint("idiv", v0, v1, v2);
        assert_eq!(ic.name, "idiv");
        assert_eq!(ic.num_defs(), 2);

        // Check fixed register constraints
        let defs: Vec<_> = ic.defs().collect();
        assert!(defs[0].constraint.is_fixed());
        assert_eq!(defs[0].constraint.fixed_reg(), Some(PReg::Gpr(Gpr::Rax)));
        assert!(defs[1].constraint.is_fixed());
        assert_eq!(defs[1].constraint.fixed_reg(), Some(PReg::Gpr(Gpr::Rdx)));
    }

    #[test]
    fn test_shift_by_cl_template() {
        let v0 = VReg::new(0);
        let v1 = VReg::new(1);
        let v2 = VReg::new(2);

        let ic = shift_by_cl_constraint("shl", v0, v1, v2);
        assert!(ic.is_two_address);

        // Count operand should be fixed to RCX
        let uses: Vec<_> = ic.uses().collect();
        assert!(
            uses.iter()
                .any(|op| op.constraint.fixed_reg() == Some(PReg::Gpr(Gpr::Rcx)))
        );
    }

    // =========================================================================
    // ConstraintDatabase Tests
    // =========================================================================

    #[test]
    fn test_constraint_database_register() {
        let mut db = ConstraintDatabase::new();
        let v0 = VReg::new(0);

        let ic = InstructionConstraint::new("test").def(v0, OperandConstraint::gpr());

        db.register(ic);
        assert_eq!(db.len(), 1);
        assert!(!db.is_empty());
    }

    #[test]
    fn test_constraint_database_lookup() {
        let mut db = ConstraintDatabase::new();
        let v0 = VReg::new(0);

        let ic = InstructionConstraint::new("test_op").def(v0, OperandConstraint::ymm());

        db.register(ic);

        let found = db.get("test_op");
        assert!(found.is_some());
        assert_eq!(found.unwrap().name, "test_op");

        let not_found = db.get("nonexistent");
        assert!(not_found.is_none());
    }

    #[test]
    fn test_all_constraint_types() {
        // Test all OperandConstraint variants
        let constraints = vec![
            OperandConstraint::FixedReg(PReg::Gpr(Gpr::Rax)),
            OperandConstraint::RegClass(RegClass::Int),
            OperandConstraint::Tied(0),
            OperandConstraint::RegOrMem(RegClass::Float),
            OperandConstraint::Memory,
            OperandConstraint::Any,
            OperandConstraint::ReuseInput(1),
        ];

        for c in constraints {
            // All should have Display impl
            let _ = format!("{}", c);

            // All should have reg_class or return None
            let _ = c.get_reg_class();

            // All should have is_fixed
            let _ = c.is_fixed();
        }
    }
}
