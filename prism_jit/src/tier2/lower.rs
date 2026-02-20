//! Instruction Selection (Lowering)
//!
//! Converts Sea-of-Nodes IR to machine instructions using tree-pattern matching.
//!
//! # Algorithm
//!
//! Uses a simple covering algorithm:
//! 1. Traverse nodes in reverse postorder
//! 2. Match each node to a machine instruction pattern
//! 3. Handle complex patterns (address modes, fused multiply-add)
//!
//! # Architecture
//!
//! Machine instructions use virtual registers that will be resolved
//! to physical registers during emission.

use crate::backend::x64::registers::{Gpr, MemOperand, Xmm};
use crate::backend::x64::simd::{Ymm, Zmm};
use crate::ir::graph::Graph;
use crate::ir::node::NodeId;
use crate::ir::operators::{ArithOp, BitwiseOp, CmpOp, ControlOp, Operator};
use crate::regalloc::{Allocation, AllocationMap, PReg, VReg};
use std::collections::HashMap;

// =============================================================================
// Machine Operand
// =============================================================================

/// An operand for a machine instruction.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MachineOperand {
    /// A virtual register (to be resolved to physical).
    VReg(VReg),
    /// A physical register (already allocated).
    PReg(PReg),
    /// An immediate value.
    Imm(i64),
    /// A memory location (base + offset).
    Mem(MemOperand),
    /// A label reference (for branches).
    Label(u32),
    /// A stack slot (for spills).
    StackSlot(i32),
    /// Unassigned/invalid.
    None,
}

impl MachineOperand {
    /// Create a GPR operand.
    pub fn gpr(gpr: Gpr) -> Self {
        MachineOperand::PReg(PReg::Gpr(gpr))
    }

    /// Create an XMM operand.
    #[allow(dead_code)]
    pub fn xmm(xmm: Xmm) -> Self {
        MachineOperand::PReg(PReg::Xmm(xmm))
    }

    /// Create a YMM (256-bit) operand.
    #[allow(dead_code)]
    pub fn ymm(ymm: Ymm) -> Self {
        MachineOperand::PReg(PReg::Ymm(ymm))
    }

    /// Create a ZMM (512-bit) operand.
    #[allow(dead_code)]
    pub fn zmm(zmm: Zmm) -> Self {
        MachineOperand::PReg(PReg::Zmm(zmm))
    }

    /// Check if this is a vector register (XMM, YMM, or ZMM).
    #[allow(dead_code)]
    pub fn is_vector(&self) -> bool {
        match self {
            MachineOperand::PReg(p) => matches!(p, PReg::Xmm(_) | PReg::Ymm(_) | PReg::Zmm(_)),
            MachineOperand::VReg(_) => false, // Can't determine until allocation
            _ => false,
        }
    }

    /// Check if this is a register.
    pub fn is_reg(&self) -> bool {
        matches!(self, MachineOperand::VReg(_) | MachineOperand::PReg(_))
    }

    /// Check if this is an immediate.
    pub fn is_imm(&self) -> bool {
        matches!(self, MachineOperand::Imm(_))
    }

    /// Check if this is memory.
    #[allow(dead_code)]
    pub fn is_mem(&self) -> bool {
        matches!(self, MachineOperand::Mem(_))
    }
}

// =============================================================================
// Machine Op
// =============================================================================

/// Machine operation (x64 instruction mnemonic).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum MachineOp {
    // Data movement
    /// Move register/memory/immediate to register.
    Mov,
    /// Move with sign extension.
    #[allow(dead_code)]
    Movsx,
    /// Move with zero extension.
    #[allow(dead_code)]
    Movzx,
    /// Load effective address.
    #[allow(dead_code)]
    Lea,
    /// Push to stack.
    #[allow(dead_code)]
    Push,
    /// Pop from stack.
    #[allow(dead_code)]
    Pop,

    // Integer arithmetic
    /// Add.
    Add,
    /// Subtract.
    Sub,
    /// Signed multiply.
    Imul,
    /// Unsigned divide (divides RDX:RAX).
    #[allow(dead_code)]
    Div,
    /// Signed divide.
    Idiv,
    /// Negate.
    Neg,
    /// Increment.
    #[allow(dead_code)]
    Inc,
    /// Decrement.
    #[allow(dead_code)]
    Dec,
    /// Convert doubleword to quadword (sign extend EAX to RDX:RAX).
    #[allow(dead_code)]
    Cdq,
    /// Convert quadword to octaword (sign extend RAX to RDX:RAX).
    Cqo,

    // Bitwise operations
    /// Bitwise AND.
    And,
    /// Bitwise OR.
    Or,
    /// Bitwise XOR.
    Xor,
    /// Bitwise NOT.
    Not,
    /// Shift left.
    Shl,
    /// Shift right arithmetic.
    Sar,
    /// Shift right logical.
    #[allow(dead_code)]
    Shr,

    // Comparison
    /// Compare.
    Cmp,
    /// Test (AND without storing result).
    Test,
    /// Set byte on condition.
    Setcc,

    // Control flow
    /// Jump.
    Jmp,
    /// Conditional jump.
    Jcc,
    /// Call.
    #[allow(dead_code)]
    Call,
    /// Return.
    Ret,

    // SSE floating-point
    /// Move scalar double.
    #[allow(dead_code)]
    Movsd,
    /// Move scalar single.
    #[allow(dead_code)]
    Movss,
    /// Add scalar double.
    Addsd,
    /// Subtract scalar double.
    Subsd,
    /// Multiply scalar double.
    Mulsd,
    /// Divide scalar double.
    Divsd,
    /// Unordered compare scalar double.
    Ucomisd,
    /// Convert int to scalar double.
    #[allow(dead_code)]
    Cvtsi2sd,
    /// Convert scalar double to int (truncate).
    #[allow(dead_code)]
    Cvttsd2si,
    /// XOR packed double (for zeroing).
    #[allow(dead_code)]
    Xorpd,

    // =========================================================================
    // AVX/AVX2 Vector Operations (YMM - 256-bit)
    // =========================================================================

    // Vector Data Movement
    /// Move aligned packed double-precision (256-bit).
    #[allow(dead_code)]
    Vmovapd256,
    /// Move aligned packed single-precision (256-bit).
    #[allow(dead_code)]
    Vmovaps256,
    /// Move unaligned packed double-precision (256-bit).
    #[allow(dead_code)]
    Vmovupd256,
    /// Move unaligned packed single-precision (256-bit).
    #[allow(dead_code)]
    Vmovups256,
    /// Move aligned packed integer (256-bit).
    #[allow(dead_code)]
    Vmovdqa256,

    // Vector Floating-Point Arithmetic
    /// Add packed double-precision (256-bit).
    #[allow(dead_code)]
    Vaddpd256,
    /// Subtract packed double-precision (256-bit).
    #[allow(dead_code)]
    Vsubpd256,
    /// Multiply packed double-precision (256-bit).
    #[allow(dead_code)]
    Vmulpd256,
    /// Divide packed double-precision (256-bit).
    #[allow(dead_code)]
    Vdivpd256,
    /// Square root packed double-precision (256-bit).
    #[allow(dead_code)]
    Vsqrtpd256,
    /// Minimum packed double-precision (256-bit).
    #[allow(dead_code)]
    Vminpd256,
    /// Maximum packed double-precision (256-bit).
    #[allow(dead_code)]
    Vmaxpd256,

    // Vector Integer Arithmetic
    /// Add packed doublewords (256-bit).
    #[allow(dead_code)]
    Vpaddd256,
    /// Subtract packed doublewords (256-bit).
    #[allow(dead_code)]
    Vpsubd256,
    /// Multiply packed doublewords (256-bit).
    #[allow(dead_code)]
    Vpmulld256,
    /// Add packed quadwords (256-bit).
    #[allow(dead_code)]
    Vpaddq256,
    /// Subtract packed quadwords (256-bit).
    #[allow(dead_code)]
    Vpsubq256,

    // Vector Logical
    /// Bitwise AND packed double-precision (256-bit).
    #[allow(dead_code)]
    Vandpd256,
    /// Bitwise OR packed double-precision (256-bit).
    #[allow(dead_code)]
    Vorpd256,
    /// Bitwise XOR packed double-precision (256-bit).
    #[allow(dead_code)]
    Vxorpd256,
    /// Bitwise AND-NOT packed double-precision (256-bit).
    #[allow(dead_code)]
    Vandnpd256,

    // Vector Compare
    /// Compare packed double-precision (256-bit).
    #[allow(dead_code)]
    Vcmppd256,

    // Vector FMA (256-bit)
    /// Fused multiply-add packed double-precision (256-bit).
    #[allow(dead_code)]
    Vfmadd213pd256,
    /// Fused multiply-subtract packed double-precision (256-bit).
    #[allow(dead_code)]
    Vfmsub213pd256,

    // Vector Shuffle/Permute
    /// Permute packed double-precision (256-bit).
    #[allow(dead_code)]
    Vpermpd256,
    /// Shuffle packed double-precision (256-bit).
    #[allow(dead_code)]
    Vshufpd256,
    /// Blend packed double-precision (256-bit).
    #[allow(dead_code)]
    Vblendpd256,

    // Vector Broadcast
    /// Broadcast scalar double to all lanes (256-bit).
    #[allow(dead_code)]
    Vbroadcastsd256,
    /// Broadcast 128-bit to 256-bit.
    #[allow(dead_code)]
    Vbroadcastf128,

    // =========================================================================
    // AVX-512 Vector Operations (ZMM - 512-bit)
    // =========================================================================

    // Vector Data Movement
    /// Move aligned packed double-precision (512-bit).
    #[allow(dead_code)]
    Vmovapd512,
    /// Move aligned packed single-precision (512-bit).
    #[allow(dead_code)]
    Vmovaps512,
    /// Move unaligned packed double-precision (512-bit).
    #[allow(dead_code)]
    Vmovupd512,
    /// Move unaligned packed single-precision (512-bit).
    #[allow(dead_code)]
    Vmovups512,
    /// Move aligned packed 64-bit integer (512-bit).
    #[allow(dead_code)]
    Vmovdqa64,
    /// Move aligned packed 32-bit integer (512-bit).
    #[allow(dead_code)]
    Vmovdqa32,

    // Vector Floating-Point Arithmetic
    /// Add packed double-precision (512-bit).
    #[allow(dead_code)]
    Vaddpd512,
    /// Subtract packed double-precision (512-bit).
    #[allow(dead_code)]
    Vsubpd512,
    /// Multiply packed double-precision (512-bit).
    #[allow(dead_code)]
    Vmulpd512,
    /// Divide packed double-precision (512-bit).
    #[allow(dead_code)]
    Vdivpd512,
    /// Square root packed double-precision (512-bit).
    #[allow(dead_code)]
    Vsqrtpd512,
    /// Minimum packed double-precision (512-bit).
    #[allow(dead_code)]
    Vminpd512,
    /// Maximum packed double-precision (512-bit).
    #[allow(dead_code)]
    Vmaxpd512,

    // Vector Integer Arithmetic (512-bit)
    /// Add packed doublewords (512-bit).
    #[allow(dead_code)]
    Vpaddd512,
    /// Subtract packed doublewords (512-bit).
    #[allow(dead_code)]
    Vpsubd512,
    /// Multiply packed doublewords (512-bit).
    #[allow(dead_code)]
    Vpmulld512,
    /// Add packed quadwords (512-bit).
    #[allow(dead_code)]
    Vpaddq512,
    /// Subtract packed quadwords (512-bit).
    #[allow(dead_code)]
    Vpsubq512,

    // Vector Logical (512-bit)
    /// Bitwise AND packed 64-bit integers (512-bit).
    #[allow(dead_code)]
    Vpandq512,
    /// Bitwise OR packed 64-bit integers (512-bit).
    #[allow(dead_code)]
    Vporq512,
    /// Bitwise XOR packed 64-bit integers (512-bit).
    #[allow(dead_code)]
    Vpxorq512,
    /// Bitwise AND-NOT packed 64-bit integers (512-bit).
    #[allow(dead_code)]
    Vpandnq512,

    // Vector Compare (512-bit)
    /// Compare packed double-precision (512-bit).
    #[allow(dead_code)]
    Vcmppd512,

    // Vector FMA (512-bit)
    /// Fused multiply-add packed double-precision (512-bit).
    #[allow(dead_code)]
    Vfmadd213pd512,
    /// Fused multiply-subtract packed double-precision (512-bit).
    #[allow(dead_code)]
    Vfmsub213pd512,

    // Vector Shuffle/Permute (512-bit)
    /// Permute packed double-precision (512-bit).
    #[allow(dead_code)]
    Vpermpd512,
    /// Shuffle packed double-precision (512-bit).
    #[allow(dead_code)]
    Vshufpd512,

    // Vector Broadcast (512-bit)
    /// Broadcast scalar double to all lanes (512-bit).
    #[allow(dead_code)]
    Vbroadcastsd512,

    // Vector Gather (512-bit)
    /// Gather packed double-precision using dword indices (512-bit).
    #[allow(dead_code)]
    Vgatherdpd512,
    /// Gather packed double-precision using qword indices (512-bit).
    #[allow(dead_code)]
    Vgatherqpd512,

    // Vector Scatter (512-bit)
    /// Scatter packed double-precision using dword indices (512-bit).
    #[allow(dead_code)]
    Vscatterdpd512,
    /// Scatter packed double-precision using qword indices (512-bit).
    #[allow(dead_code)]
    Vscatterqpd512,

    // Vector Mask Operations
    /// Move mask to GPR.
    #[allow(dead_code)]
    Kmovw,
    /// AND mask registers.
    #[allow(dead_code)]
    Kandw,
    /// OR mask registers.
    #[allow(dead_code)]
    Korw,
    /// NOT mask register.
    #[allow(dead_code)]
    Knotw,

    // =========================================================================
    // Vector Spill/Reload Pseudo-instructions
    // =========================================================================
    /// Spill YMM register to stack (256-bit).
    #[allow(dead_code)]
    SpillYmm,
    /// Reload YMM register from stack (256-bit).
    #[allow(dead_code)]
    ReloadYmm,
    /// Spill ZMM register to stack (512-bit).
    #[allow(dead_code)]
    SpillZmm,
    /// Reload ZMM register from stack (512-bit).
    #[allow(dead_code)]
    ReloadZmm,

    // Special
    /// No operation.
    #[allow(dead_code)]
    Nop,
    /// Breakpoint (for debugging).
    #[allow(dead_code)]
    Int3,
    /// Label (pseudo-instruction).
    Label,
    /// Spill to stack.
    #[allow(dead_code)]
    Spill,
    /// Reload from stack.
    #[allow(dead_code)]
    Reload,
}

// =============================================================================
// Condition Code
// =============================================================================

/// x64 condition codes for conditional jumps and setcc.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum CondCode {
    // Signed comparisons
    /// Equal (ZF=1).
    E = 0x04,
    /// Not equal (ZF=0).
    Ne = 0x05,
    /// Less than (SF≠OF).
    L = 0x0C,
    /// Less or equal (ZF=1 or SF≠OF).
    Le = 0x0E,
    /// Greater than (ZF=0 and SF=OF).
    G = 0x0F,
    /// Greater or equal (SF=OF).
    Ge = 0x0D,

    // Unsigned comparisons
    /// Below (CF=1).
    B = 0x02,
    /// Below or equal (CF=1 or ZF=1).
    Be = 0x06,
    /// Above (CF=0 and ZF=0).
    A = 0x07,
    /// Above or equal (CF=0).
    Ae = 0x03,

    // Other
    /// Overflow (OF=1).
    #[allow(dead_code)]
    O = 0x00,
    /// No overflow (OF=0).
    #[allow(dead_code)]
    No = 0x01,
    /// Sign (SF=1).
    #[allow(dead_code)]
    S = 0x08,
    /// No sign (SF=0).
    #[allow(dead_code)]
    Ns = 0x09,
    /// Parity (PF=1).
    #[allow(dead_code)]
    P = 0x0A,
    /// No parity (PF=0).
    #[allow(dead_code)]
    Np = 0x0B,
}

impl CondCode {
    /// Get the encoding byte for this condition.
    #[inline]
    #[allow(dead_code)]
    pub const fn encoding(self) -> u8 {
        self as u8
    }

    /// Get the inverse condition.
    #[allow(dead_code)]
    pub const fn inverse(self) -> Self {
        match self {
            CondCode::E => CondCode::Ne,
            CondCode::Ne => CondCode::E,
            CondCode::L => CondCode::Ge,
            CondCode::Le => CondCode::G,
            CondCode::G => CondCode::Le,
            CondCode::Ge => CondCode::L,
            CondCode::B => CondCode::Ae,
            CondCode::Be => CondCode::A,
            CondCode::A => CondCode::Be,
            CondCode::Ae => CondCode::B,
            CondCode::O => CondCode::No,
            CondCode::No => CondCode::O,
            CondCode::S => CondCode::Ns,
            CondCode::Ns => CondCode::S,
            CondCode::P => CondCode::Np,
            CondCode::Np => CondCode::P,
        }
    }

    /// Convert from IR comparison op to condition code.
    pub fn from_cmp_op(op: CmpOp, signed: bool) -> Self {
        if signed {
            match op {
                CmpOp::Eq => CondCode::E,
                CmpOp::Ne => CondCode::Ne,
                CmpOp::Lt => CondCode::L,
                CmpOp::Le => CondCode::Le,
                CmpOp::Gt => CondCode::G,
                CmpOp::Ge => CondCode::Ge,
                CmpOp::Is => CondCode::E,
                CmpOp::IsNot => CondCode::Ne,
                CmpOp::In | CmpOp::NotIn => CondCode::E, // Handled by runtime call
            }
        } else {
            match op {
                CmpOp::Eq => CondCode::E,
                CmpOp::Ne => CondCode::Ne,
                CmpOp::Lt => CondCode::B,
                CmpOp::Le => CondCode::Be,
                CmpOp::Gt => CondCode::A,
                CmpOp::Ge => CondCode::Ae,
                CmpOp::Is => CondCode::E,
                CmpOp::IsNot => CondCode::Ne,
                CmpOp::In | CmpOp::NotIn => CondCode::E,
            }
        }
    }
}

// =============================================================================
// Machine Instruction
// =============================================================================

/// A single machine instruction with operands.
#[derive(Debug, Clone)]
pub struct MachineInst {
    /// The operation.
    pub op: MachineOp,
    /// Destination operand (if any).
    pub dst: MachineOperand,
    /// First source operand.
    pub src1: MachineOperand,
    /// Second source operand (if any).
    pub src2: MachineOperand,
    /// Condition code (for Jcc, Setcc).
    pub cc: Option<CondCode>,
    /// Original IR node (for debugging).
    pub origin: Option<NodeId>,
}

impl MachineInst {
    /// Create a new instruction with destination and source.
    pub fn new(op: MachineOp, dst: MachineOperand, src1: MachineOperand) -> Self {
        MachineInst {
            op,
            dst,
            src1,
            src2: MachineOperand::None,
            cc: None,
            origin: None,
        }
    }

    /// Create a binary instruction (dst = src1 op src2).
    pub fn binary(
        op: MachineOp,
        dst: MachineOperand,
        src1: MachineOperand,
        src2: MachineOperand,
    ) -> Self {
        MachineInst {
            op,
            dst,
            src1,
            src2,
            cc: None,
            origin: None,
        }
    }

    /// Create a nullary instruction (no operands).
    pub fn nullary(op: MachineOp) -> Self {
        MachineInst {
            op,
            dst: MachineOperand::None,
            src1: MachineOperand::None,
            src2: MachineOperand::None,
            cc: None,
            origin: None,
        }
    }

    /// Create a label pseudo-instruction.
    pub fn label(id: u32) -> Self {
        MachineInst {
            op: MachineOp::Label,
            dst: MachineOperand::Label(id),
            src1: MachineOperand::None,
            src2: MachineOperand::None,
            cc: None,
            origin: None,
        }
    }

    /// Create a conditional jump.
    pub fn jcc(cc: CondCode, label: u32) -> Self {
        MachineInst {
            op: MachineOp::Jcc,
            dst: MachineOperand::Label(label),
            src1: MachineOperand::None,
            src2: MachineOperand::None,
            cc: Some(cc),
            origin: None,
        }
    }

    /// Set the origin node.
    pub fn with_origin(mut self, node: NodeId) -> Self {
        self.origin = Some(node);
        self
    }
}

// =============================================================================
// Machine Function
// =============================================================================

/// A function in machine code representation.
#[derive(Debug)]
pub struct MachineFunction {
    /// The instructions.
    pub insts: Vec<MachineInst>,
    /// Next label ID.
    next_label: u32,
    /// Mapping from IR labels to machine labels.
    label_map: HashMap<u32, u32>,
    /// Stack frame size in bytes.
    pub frame_size: u32,
    /// Number of spill slots used.
    #[allow(dead_code)]
    pub spill_slots: u32,
}

impl MachineFunction {
    /// Create a new empty machine function.
    pub fn new() -> Self {
        MachineFunction {
            insts: Vec::new(),
            next_label: 0,
            label_map: HashMap::new(),
            frame_size: 0,
            spill_slots: 0,
        }
    }

    /// Allocate a new label.
    pub fn new_label(&mut self) -> u32 {
        let id = self.next_label;
        self.next_label += 1;
        id
    }

    /// Add an instruction.
    pub fn push(&mut self, inst: MachineInst) {
        self.insts.push(inst);
    }

    /// Add a label.
    pub fn add_label(&mut self, id: u32) {
        self.insts.push(MachineInst::label(id));
    }

    /// Get or create a label for an IR label.
    pub fn get_or_create_label(&mut self, ir_label: u32) -> u32 {
        if let Some(&label) = self.label_map.get(&ir_label) {
            label
        } else {
            let label = self.new_label();
            self.label_map.insert(ir_label, label);
            label
        }
    }
}

impl Default for MachineFunction {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// Instruction Selector
// =============================================================================

/// Instruction selector that converts IR to machine instructions.
pub struct InstructionSelector<'a> {
    /// The IR graph.
    graph: &'a Graph,
    /// Register allocation map.
    alloc_map: &'a AllocationMap,
    /// The output machine function.
    mfunc: MachineFunction,
    /// Mapping from NodeId to VReg.
    node_to_vreg: HashMap<NodeId, VReg>,
    /// Next VReg ID.
    next_vreg: u32,
}

/// Byte size of one VM register slot in the frame array.
const VM_REGISTER_SLOT_SIZE: i32 = 8;
/// Dedicated scratch register holding the frame-base pointer in Tier2 code.
const FRAME_BASE_SCRATCH_GPR: Gpr = Gpr::R11;
/// Temporary scratch register used for stack-slot parameter materialization.
const PARAM_STACK_TEMP_GPR: Gpr = Gpr::Rax;

impl<'a> InstructionSelector<'a> {
    /// Match liveness behavior: all non-control nodes produce allocatable values.
    #[inline]
    fn node_produces_value(op: Operator) -> bool {
        !matches!(op, Operator::Control(_))
    }

    /// Create a new instruction selector.
    pub fn new(graph: &'a Graph, alloc_map: &'a AllocationMap) -> Self {
        let spill_frame_size = alloc_map.spill_slot_count().saturating_mul(8);
        let mut node_to_vreg = HashMap::new();
        let mut next_vreg = 0u32;

        // Pre-seed vreg mapping to match liveness numbering exactly.
        for (id, node) in graph.iter() {
            if node.is_dead() || !Self::node_produces_value(node.op) {
                continue;
            }
            node_to_vreg.insert(id, VReg::new(next_vreg));
            next_vreg += 1;
        }

        InstructionSelector {
            graph,
            alloc_map,
            mfunc: MachineFunction {
                frame_size: spill_frame_size,
                ..MachineFunction::new()
            },
            node_to_vreg,
            next_vreg,
        }
    }

    /// Select instructions for the entire graph.
    pub fn select(
        graph: &'a Graph,
        alloc_map: &'a AllocationMap,
    ) -> Result<MachineFunction, String> {
        let mut selector = InstructionSelector::new(graph, alloc_map);
        selector.select_all()?;
        Ok(selector.mfunc)
    }

    /// Select instructions for all nodes.
    fn select_all(&mut self) -> Result<(), String> {
        self.materialize_parameters()?;

        // Process nodes in order (simplified - real impl uses RPO)
        for (id, node) in self.graph.iter() {
            if node.is_dead() {
                continue;
            }
            self.select_node(id)?;
        }
        Ok(())
    }

    /// Materialize parameter nodes from the VM frame into machine operands.
    ///
    /// Parameters are loaded from `frame_base[index]`, where `frame_base` is
    /// preloaded into `FRAME_BASE_SCRATCH_GPR` by the Tier2 prologue.
    fn materialize_parameters(&mut self) -> Result<(), String> {
        let mut stacked = Vec::new();
        let mut direct = Vec::new();

        for (id, node) in self.graph.iter() {
            if node.is_dead() {
                continue;
            }
            let Operator::Parameter(index) = node.op else {
                continue;
            };

            let dst = self.operand_for_node(id);
            match dst {
                MachineOperand::StackSlot(_) => stacked.push((id, index, dst)),
                MachineOperand::PReg(PReg::Gpr(gpr)) if gpr == FRAME_BASE_SCRATCH_GPR => {
                    return Err(format!(
                        "parameter node {:?} uses reserved scratch register {:?}",
                        id, FRAME_BASE_SCRATCH_GPR
                    ));
                }
                MachineOperand::PReg(PReg::Gpr(_)) | MachineOperand::VReg(_) => {
                    direct.push((id, index, dst));
                }
                _ => {
                    return Err(format!(
                        "parameter node {:?} lowered to unsupported destination operand {:?}",
                        id, dst
                    ));
                }
            }
        }

        // Handle spilled parameters first so the temporary register cannot clobber
        // already-materialized parameter values.
        stacked.sort_by_key(|(_, index, _)| *index);
        direct.sort_by_key(|(_, index, _)| *index);

        for (id, index, dst) in stacked.into_iter().chain(direct.into_iter()) {
            self.emit_parameter_materialization(id, index, dst)?;
        }

        Ok(())
    }

    fn emit_parameter_materialization(
        &mut self,
        node_id: NodeId,
        index: u16,
        dst: MachineOperand,
    ) -> Result<(), String> {
        let byte_offset = i32::from(index)
            .checked_mul(VM_REGISTER_SLOT_SIZE)
            .ok_or_else(|| {
                format!(
                    "parameter index {} overflow while lowering node {:?}",
                    index, node_id
                )
            })?;
        let src = MachineOperand::Mem(MemOperand::base_disp(FRAME_BASE_SCRATCH_GPR, byte_offset));

        match dst {
            MachineOperand::StackSlot(offset) => {
                self.mfunc.push(
                    MachineInst::new(
                        MachineOp::Mov,
                        MachineOperand::gpr(PARAM_STACK_TEMP_GPR),
                        src,
                    )
                    .with_origin(node_id),
                );
                self.mfunc.push(
                    MachineInst::new(
                        MachineOp::Mov,
                        MachineOperand::StackSlot(offset),
                        MachineOperand::gpr(PARAM_STACK_TEMP_GPR),
                    )
                    .with_origin(node_id),
                );
            }
            _ => {
                self.mfunc
                    .push(MachineInst::new(MachineOp::Mov, dst, src).with_origin(node_id));
            }
        }

        Ok(())
    }

    /// Get or create a VReg for a node.
    fn vreg_for_node(&mut self, node: NodeId) -> VReg {
        if let Some(&vreg) = self.node_to_vreg.get(&node) {
            vreg
        } else {
            let vreg = VReg::new(self.next_vreg);
            self.next_vreg += 1;
            self.node_to_vreg.insert(node, vreg);
            vreg
        }
    }

    /// Get the operand for a node.
    fn operand_for_node(&mut self, node: NodeId) -> MachineOperand {
        let vreg = self.vreg_for_node(node);

        // Check if we have an allocation
        match self.alloc_map.get(vreg) {
            Allocation::Register(preg) => MachineOperand::PReg(preg),
            Allocation::Spill(slot) => MachineOperand::StackSlot(slot.offset()),
            Allocation::Unassigned => MachineOperand::VReg(vreg),
        }
    }

    /// Get input at index from a node.
    fn get_input(&self, node_id: NodeId, index: usize) -> Option<NodeId> {
        self.graph.node(node_id).inputs.get(index)
    }

    /// Get operand for an input at index.
    fn operand_for_input(&mut self, node_id: NodeId, index: usize) -> Option<MachineOperand> {
        self.get_input(node_id, index)
            .map(|inp| self.operand_for_node(inp))
    }

    /// Get required operand for an input at index.
    fn required_input_operand(
        &mut self,
        node_id: NodeId,
        index: usize,
    ) -> Result<MachineOperand, String> {
        self.operand_for_input(node_id, index).ok_or_else(|| {
            format!(
                "missing input {} for node {:?} ({:?}) during instruction selection",
                index,
                node_id,
                self.graph.node(node_id).op
            )
        })
    }

    /// Select instructions for a single node.
    fn select_node(&mut self, node_id: NodeId) -> Result<(), String> {
        let node = self.graph.node(node_id);
        let op = node.op;

        match op {
            // Constants
            Operator::ConstInt(value) => {
                let dst = self.operand_for_node(node_id);
                self.mfunc.push(
                    MachineInst::new(MachineOp::Mov, dst, MachineOperand::Imm(value))
                        .with_origin(node_id),
                );
                Ok(())
            }

            Operator::ConstFloat(bits) => {
                // Load float from constant pool (simplified - use RIP-relative)
                let dst = self.operand_for_node(node_id);
                self.mfunc.push(
                    MachineInst::new(MachineOp::Mov, dst, MachineOperand::Imm(bits as i64))
                        .with_origin(node_id),
                );
                Ok(())
            }

            Operator::ConstBool(value) => {
                let dst = self.operand_for_node(node_id);
                self.mfunc.push(
                    MachineInst::new(
                        MachineOp::Mov,
                        dst,
                        MachineOperand::Imm(if value { 1 } else { 0 }),
                    )
                    .with_origin(node_id),
                );
                Ok(())
            }

            Operator::ConstNone => {
                let dst = self.operand_for_node(node_id);
                // None is encoded as 0 in our value representation
                self.mfunc.push(
                    MachineInst::new(MachineOp::Mov, dst, MachineOperand::Imm(0))
                        .with_origin(node_id),
                );
                Ok(())
            }

            // Arithmetic operations (note: IntOp, FloatOp, not IntArith/FloatArith)
            Operator::IntOp(arith_op) => self.select_int_arith(node_id, arith_op),

            Operator::FloatOp(arith_op) => self.select_float_arith(node_id, arith_op),

            // Comparisons
            Operator::IntCmp(cmp_op) => self.select_int_cmp(node_id, cmp_op),

            Operator::FloatCmp(cmp_op) => self.select_float_cmp(node_id, cmp_op),

            // Bitwise operations
            Operator::Bitwise(bitwise_op) => self.select_bitwise(node_id, bitwise_op),

            // Control flow
            Operator::Control(ctrl_op) => self.select_control(node_id, ctrl_op),

            // Parameters and Phi are handled specially
            Operator::Phi | Operator::LoopPhi => {
                // These don't generate code directly
                Ok(())
            }
            Operator::Parameter(_) => {
                // Parameter nodes are materialized in a dedicated pre-pass.
                Ok(())
            }
            Operator::Projection(index) => self.select_projection(node_id, index),

            // Other operators
            _ => Err(format!(
                "Tier2 instruction selection does not support operator {:?} at node {:?}",
                op, node_id
            )),
        }
    }

    fn select_projection(&mut self, node_id: NodeId, index: u8) -> Result<(), String> {
        if index > 1 {
            return Err(format!(
                "Tier2 lowering only supports control projections 0/1, got Projection({}) at node {:?}",
                index, node_id
            ));
        }

        let if_node = self.get_input(node_id, 0).ok_or_else(|| {
            format!(
                "projection node {:?} is missing its If input during instruction selection",
                node_id
            )
        })?;
        if !matches!(
            self.graph.node(if_node).op,
            Operator::Control(ControlOp::If)
        ) {
            return Err(format!(
                "Tier2 lowering only supports projections from If nodes, but projection {:?} input is {:?}",
                node_id,
                self.graph.node(if_node).op
            ));
        }

        // Projection nodes are pure control tokens; their target blocks carry labels.
        Ok(())
    }
    /// Select instructions for integer arithmetic.
    fn select_int_arith(&mut self, node_id: NodeId, op: ArithOp) -> Result<(), String> {
        let dst = self.operand_for_node(node_id);

        match op {
            ArithOp::Add => {
                let lhs = self.required_input_operand(node_id, 0)?;
                let rhs = self.required_input_operand(node_id, 1)?;
                self.emit_binary(MachineOp::Add, dst, lhs, rhs, node_id);
                Ok(())
            }

            ArithOp::Sub => {
                let lhs = self.required_input_operand(node_id, 0)?;
                let rhs = self.required_input_operand(node_id, 1)?;
                self.emit_binary(MachineOp::Sub, dst, lhs, rhs, node_id);
                Ok(())
            }

            ArithOp::Mul => {
                let lhs = self.required_input_operand(node_id, 0)?;
                let rhs = self.required_input_operand(node_id, 1)?;
                self.emit_binary(MachineOp::Imul, dst, lhs, rhs, node_id);
                Ok(())
            }

            ArithOp::TrueDiv | ArithOp::FloorDiv => {
                // Division is complex: RDX:RAX / divisor
                let lhs = self.required_input_operand(node_id, 0)?;
                let rhs = self.required_input_operand(node_id, 1)?;
                // Move dividend to RAX
                self.mfunc.push(MachineInst::new(
                    MachineOp::Mov,
                    MachineOperand::gpr(Gpr::Rax),
                    lhs,
                ));

                // Sign extend to RDX:RAX
                self.mfunc.push(MachineInst::nullary(MachineOp::Cqo));

                // Divide
                self.mfunc.push(
                    MachineInst::new(MachineOp::Idiv, MachineOperand::gpr(Gpr::Rax), rhs)
                        .with_origin(node_id),
                );

                // Move result from RAX to destination
                if dst != MachineOperand::gpr(Gpr::Rax) {
                    self.mfunc.push(MachineInst::new(
                        MachineOp::Mov,
                        dst,
                        MachineOperand::gpr(Gpr::Rax),
                    ));
                }
                Ok(())
            }

            ArithOp::Mod => {
                // Modulo: remainder is in RDX after IDIV
                let lhs = self.required_input_operand(node_id, 0)?;
                let rhs = self.required_input_operand(node_id, 1)?;
                self.mfunc.push(MachineInst::new(
                    MachineOp::Mov,
                    MachineOperand::gpr(Gpr::Rax),
                    lhs,
                ));
                self.mfunc.push(MachineInst::nullary(MachineOp::Cqo));
                self.mfunc.push(
                    MachineInst::new(MachineOp::Idiv, MachineOperand::gpr(Gpr::Rax), rhs)
                        .with_origin(node_id),
                );

                // Remainder is in RDX
                if dst != MachineOperand::gpr(Gpr::Rdx) {
                    self.mfunc.push(MachineInst::new(
                        MachineOp::Mov,
                        dst,
                        MachineOperand::gpr(Gpr::Rdx),
                    ));
                }
                Ok(())
            }

            ArithOp::Neg => {
                let src = self.required_input_operand(node_id, 0)?;
                // NEG is unary: dst = -dst
                if dst != src {
                    self.mfunc.push(MachineInst::new(MachineOp::Mov, dst, src));
                }
                self.mfunc
                    .push(MachineInst::new(MachineOp::Neg, dst, dst).with_origin(node_id));
                Ok(())
            }

            _ => Err(format!(
                "Tier2 integer lowering does not support {:?} at node {:?}",
                op, node_id
            )),
        }
    }

    /// Select instructions for floating-point arithmetic.
    fn select_float_arith(&mut self, node_id: NodeId, op: ArithOp) -> Result<(), String> {
        let dst = self.operand_for_node(node_id);

        let lhs = self.required_input_operand(node_id, 0)?;
        let rhs = self.required_input_operand(node_id, 1)?;
        let machine_op = match op {
            ArithOp::Add => MachineOp::Addsd,
            ArithOp::Sub => MachineOp::Subsd,
            ArithOp::Mul => MachineOp::Mulsd,
            ArithOp::TrueDiv => MachineOp::Divsd,
            _ => {
                return Err(format!(
                    "Tier2 float lowering does not support {:?} at node {:?}",
                    op, node_id
                ));
            }
        };

        self.emit_binary(machine_op, dst, lhs, rhs, node_id);
        Ok(())
    }

    /// Select instructions for integer comparison.
    fn select_int_cmp(&mut self, node_id: NodeId, cmp_op: CmpOp) -> Result<(), String> {
        let dst = self.operand_for_node(node_id);

        let lhs = self.required_input_operand(node_id, 0)?;
        let rhs = self.required_input_operand(node_id, 1)?;
        // Emit compare
        self.mfunc.push(
            MachineInst::binary(MachineOp::Cmp, MachineOperand::None, lhs, rhs)
                .with_origin(node_id),
        );

        // Emit setcc to get boolean result
        let cc = CondCode::from_cmp_op(cmp_op, true);
        self.mfunc.push(MachineInst {
            op: MachineOp::Setcc,
            dst,
            src1: MachineOperand::None,
            src2: MachineOperand::None,
            cc: Some(cc),
            origin: Some(node_id),
        });
        Ok(())
    }

    /// Select instructions for floating-point comparison.
    fn select_float_cmp(&mut self, node_id: NodeId, cmp_op: CmpOp) -> Result<(), String> {
        let dst = self.operand_for_node(node_id);

        let lhs = self.required_input_operand(node_id, 0)?;
        let rhs = self.required_input_operand(node_id, 1)?;
        // Use UCOMISD for float comparison
        self.mfunc.push(
            MachineInst::binary(MachineOp::Ucomisd, MachineOperand::None, lhs, rhs)
                .with_origin(node_id),
        );

        // Use unsigned compare codes for floats
        let cc = CondCode::from_cmp_op(cmp_op, false);
        self.mfunc.push(MachineInst {
            op: MachineOp::Setcc,
            dst,
            src1: MachineOperand::None,
            src2: MachineOperand::None,
            cc: Some(cc),
            origin: Some(node_id),
        });
        Ok(())
    }

    /// Select instructions for bitwise operations.
    fn select_bitwise(&mut self, node_id: NodeId, op: BitwiseOp) -> Result<(), String> {
        let dst = self.operand_for_node(node_id);

        match op {
            BitwiseOp::And => {
                let lhs = self.required_input_operand(node_id, 0)?;
                let rhs = self.required_input_operand(node_id, 1)?;
                self.emit_binary(MachineOp::And, dst, lhs, rhs, node_id);
                Ok(())
            }

            BitwiseOp::Or => {
                let lhs = self.required_input_operand(node_id, 0)?;
                let rhs = self.required_input_operand(node_id, 1)?;
                self.emit_binary(MachineOp::Or, dst, lhs, rhs, node_id);
                Ok(())
            }

            BitwiseOp::Xor => {
                let lhs = self.required_input_operand(node_id, 0)?;
                let rhs = self.required_input_operand(node_id, 1)?;
                self.emit_binary(MachineOp::Xor, dst, lhs, rhs, node_id);
                Ok(())
            }

            BitwiseOp::Not => {
                let src = self.required_input_operand(node_id, 0)?;
                if dst != src {
                    self.mfunc.push(MachineInst::new(MachineOp::Mov, dst, src));
                }
                self.mfunc
                    .push(MachineInst::new(MachineOp::Not, dst, dst).with_origin(node_id));
                Ok(())
            }

            BitwiseOp::Shl => {
                let lhs = self.required_input_operand(node_id, 0)?;
                let rhs = self.required_input_operand(node_id, 1)?;
                // Shift count must be in CL for variable shifts
                self.emit_shift(MachineOp::Shl, dst, lhs, rhs, node_id);
                Ok(())
            }

            BitwiseOp::Shr => {
                let lhs = self.required_input_operand(node_id, 0)?;
                let rhs = self.required_input_operand(node_id, 1)?;
                self.emit_shift(MachineOp::Sar, dst, lhs, rhs, node_id);
                Ok(())
            }
        }
    }
    /// Select instructions for control flow.
    fn select_control(&mut self, node_id: NodeId, op: ControlOp) -> Result<(), String> {
        if !matches!(op, ControlOp::Start | ControlOp::End) {
            let label = self.mfunc.get_or_create_label(node_id.index());
            self.mfunc.add_label(label);
        }

        match op {
            ControlOp::Start => {
                // Function entry - nothing to emit
                Ok(())
            }

            ControlOp::End => {
                // Function exit - nothing to emit
                Ok(())
            }

            ControlOp::Return => {
                // Return value should already be in RAX
                // inputs[0] is control, inputs[1] is return value
                let ret_val = self.required_input_operand(node_id, 1)?;
                self.mfunc.push(MachineInst::new(
                    MachineOp::Mov,
                    MachineOperand::gpr(Gpr::Rax),
                    ret_val,
                ));
                self.mfunc
                    .push(MachineInst::nullary(MachineOp::Ret).with_origin(node_id));
                Ok(())
            }

            ControlOp::If => {
                // inputs[0]: control, inputs[1]: condition
                let cond = self.required_input_operand(node_id, 1)?;
                let cond_for_test = match cond {
                    MachineOperand::PReg(PReg::Gpr(_)) | MachineOperand::VReg(_) => cond,
                    MachineOperand::StackSlot(offset) => {
                        let scratch = MachineOperand::gpr(FRAME_BASE_SCRATCH_GPR);
                        self.mfunc.push(
                            MachineInst::new(
                                MachineOp::Mov,
                                scratch,
                                MachineOperand::StackSlot(offset),
                            )
                            .with_origin(node_id),
                        );
                        scratch
                    }
                    MachineOperand::Imm(imm) => {
                        let scratch = MachineOperand::gpr(FRAME_BASE_SCRATCH_GPR);
                        self.mfunc.push(
                            MachineInst::new(MachineOp::Mov, scratch, MachineOperand::Imm(imm))
                                .with_origin(node_id),
                        );
                        scratch
                    }
                    other => {
                        return Err(format!(
                            "Tier2 branch condition at node {:?} must lower to GPR/stack/immediate, got {:?}",
                            node_id, other
                        ));
                    }
                };

                let mut true_proj = None;
                let mut false_proj = None;
                for &user in self.graph.uses(node_id) {
                    let user_node = self.graph.node(user);
                    if user_node.is_dead() {
                        continue;
                    }
                    match user_node.op {
                        Operator::Projection(0) => {
                            if true_proj.replace(user).is_some() {
                                return Err(format!(
                                    "If node {:?} has multiple live Projection(0) users",
                                    node_id
                                ));
                            }
                        }
                        Operator::Projection(1) => {
                            if false_proj.replace(user).is_some() {
                                return Err(format!(
                                    "If node {:?} has multiple live Projection(1) users",
                                    node_id
                                ));
                            }
                        }
                        _ => {}
                    }
                }
                let mut true_target = None;
                if let Some(true_proj) = true_proj {
                    for &user in self.graph.uses(true_proj) {
                        let user_node = self.graph.node(user);
                        if user_node.is_dead() || !matches!(user_node.op, Operator::Control(_)) {
                            continue;
                        }
                        if true_target.replace(user).is_some() {
                            return Err(format!(
                                "Projection(0) node {:?} has multiple live control users",
                                true_proj
                            ));
                        }
                    }
                    if true_target.is_none() {
                        return Err(format!(
                            "Projection(0) node {:?} has no live control target for If {:?}",
                            true_proj, node_id
                        ));
                    }
                }

                let mut false_target = None;
                if let Some(false_proj) = false_proj {
                    for &user in self.graph.uses(false_proj) {
                        let user_node = self.graph.node(user);
                        if user_node.is_dead() || !matches!(user_node.op, Operator::Control(_)) {
                            continue;
                        }
                        if false_target.replace(user).is_some() {
                            return Err(format!(
                                "Projection(1) node {:?} has multiple live control users",
                                false_proj
                            ));
                        }
                    }
                    if false_target.is_none() {
                        return Err(format!(
                            "Projection(1) node {:?} has no live control target for If {:?}",
                            false_proj, node_id
                        ));
                    }
                }

                if true_target.is_none() && false_target.is_none() {
                    return Err(format!(
                        "If node {:?} has no live projection-based control targets",
                        node_id
                    ));
                }

                self.mfunc.push(
                    MachineInst::binary(
                        MachineOp::Test,
                        MachineOperand::None,
                        cond_for_test,
                        cond_for_test,
                    )
                    .with_origin(node_id),
                );

                match (true_target, false_target) {
                    (Some(true_target), Some(false_target)) => {
                        let true_label = self.mfunc.get_or_create_label(true_target.index());
                        let false_label = self.mfunc.get_or_create_label(false_target.index());
                        self.mfunc
                            .push(MachineInst::jcc(CondCode::Ne, true_label).with_origin(node_id));
                        self.mfunc.push(
                            MachineInst::new(
                                MachineOp::Jmp,
                                MachineOperand::Label(false_label),
                                MachineOperand::None,
                            )
                            .with_origin(node_id),
                        );
                    }
                    (Some(true_target), None) => {
                        let true_label = self.mfunc.get_or_create_label(true_target.index());
                        self.mfunc
                            .push(MachineInst::jcc(CondCode::Ne, true_label).with_origin(node_id));
                    }
                    (None, Some(false_target)) => {
                        let continue_label = self.mfunc.new_label();
                        let false_label = self.mfunc.get_or_create_label(false_target.index());
                        self.mfunc.push(
                            MachineInst::jcc(CondCode::Ne, continue_label).with_origin(node_id),
                        );
                        self.mfunc.push(
                            MachineInst::new(
                                MachineOp::Jmp,
                                MachineOperand::Label(false_label),
                                MachineOperand::None,
                            )
                            .with_origin(node_id),
                        );
                        self.mfunc.add_label(continue_label);
                    }
                    (None, None) => unreachable!(),
                }
                Ok(())
            }

            ControlOp::Region | ControlOp::Loop => {
                // Merge points are block labels only.
                Ok(())
            }

            _ => Err(format!(
                "Tier2 control lowering does not support {:?} at node {:?}",
                op, node_id
            )),
        }
    }

    /// Emit a binary operation, handling x64's two-address constraint.
    fn emit_binary(
        &mut self,
        op: MachineOp,
        dst: MachineOperand,
        lhs: MachineOperand,
        rhs: MachineOperand,
        node_id: NodeId,
    ) {
        // x64 binary ops are: dst = dst op src
        // So we need: mov dst, lhs; op dst, rhs

        if dst != lhs {
            self.mfunc.push(MachineInst::new(MachineOp::Mov, dst, lhs));
        }
        self.mfunc
            .push(MachineInst::binary(op, dst, dst, rhs).with_origin(node_id));
    }

    /// Emit a shift operation (shift count must be in CL or immediate).
    fn emit_shift(
        &mut self,
        op: MachineOp,
        dst: MachineOperand,
        value: MachineOperand,
        count: MachineOperand,
        node_id: NodeId,
    ) {
        // Move value to destination
        if dst != value {
            self.mfunc
                .push(MachineInst::new(MachineOp::Mov, dst, value));
        }

        // If count is not immediate, move to RCX (CL is the low byte)
        if let MachineOperand::Imm(_) = count {
            self.mfunc
                .push(MachineInst::binary(op, dst, dst, count).with_origin(node_id));
        } else {
            self.mfunc.push(MachineInst::new(
                MachineOp::Mov,
                MachineOperand::gpr(Gpr::Rcx),
                count,
            ));
            self.mfunc.push(
                MachineInst::binary(op, dst, dst, MachineOperand::gpr(Gpr::Rcx))
                    .with_origin(node_id),
            );
        }
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::builder::{
        ArithmeticBuilder, ContainerBuilder, ControlBuilder, GraphBuilder, ObjectBuilder,
    };
    use crate::ir::node::InputList;

    #[test]
    fn test_condition_code_inverse() {
        assert_eq!(CondCode::E.inverse(), CondCode::Ne);
        assert_eq!(CondCode::L.inverse(), CondCode::Ge);
        assert_eq!(CondCode::G.inverse(), CondCode::Le);
    }

    #[test]
    fn test_machine_inst_creation() {
        let inst = MachineInst::new(
            MachineOp::Mov,
            MachineOperand::gpr(Gpr::Rax),
            MachineOperand::Imm(42),
        );
        assert_eq!(inst.op, MachineOp::Mov);
        assert!(inst.dst.is_reg());
        assert!(inst.src1.is_imm());
    }

    #[test]
    fn test_machine_function_labels() {
        let mut mfunc = MachineFunction::new();
        let l1 = mfunc.new_label();
        let l2 = mfunc.new_label();
        assert_ne!(l1, l2);
    }

    #[test]
    fn test_instruction_selection_simple() {
        let mut builder = GraphBuilder::new(4, 0);
        let const_1 = builder.const_int(1);
        let const_2 = builder.const_int(2);
        let sum = builder.int_add(const_1, const_2);
        let _ret = builder.return_value(sum);
        let graph = builder.finish();

        let alloc_map = AllocationMap::new();
        let mfunc = InstructionSelector::select(&graph, &alloc_map)
            .expect("instruction selection should succeed");

        // Should have generated some instructions
        assert!(!mfunc.insts.is_empty());
    }

    #[test]
    fn test_instruction_selection_materializes_parameters_from_frame_base() {
        let mut builder = GraphBuilder::new(2, 2);
        let p0 = builder.parameter(0).expect("parameter 0 should exist");
        let p1 = builder.parameter(1).expect("parameter 1 should exist");
        let _ret = builder.return_value(p1);
        let graph = builder.finish();

        let alloc_map = AllocationMap::new();
        let mfunc = InstructionSelector::select(&graph, &alloc_map)
            .expect("instruction selection should materialize parameters");

        let expected_p0_src = MachineOperand::Mem(MemOperand::base_disp(FRAME_BASE_SCRATCH_GPR, 0));
        let expected_p1_src = MachineOperand::Mem(MemOperand::base_disp(
            FRAME_BASE_SCRATCH_GPR,
            VM_REGISTER_SLOT_SIZE,
        ));

        assert!(mfunc.insts.iter().any(|inst| inst.origin == Some(p0)
            && inst.op == MachineOp::Mov
            && inst.src1 == expected_p0_src));
        assert!(mfunc.insts.iter().any(|inst| inst.origin == Some(p1)
            && inst.op == MachineOp::Mov
            && inst.src1 == expected_p1_src));
    }

    #[test]
    fn test_instruction_selection_materializes_spilled_parameters_before_direct_regs() {
        let mut builder = GraphBuilder::new(2, 2);
        let p0 = builder.parameter(0).expect("parameter 0 should exist");
        let p1 = builder.parameter(1).expect("parameter 1 should exist");
        let _ret = builder.return_value(p1);
        let graph = builder.finish();

        let mut alloc_map = AllocationMap::new();
        let spill = alloc_map.alloc_spill_slot();
        alloc_map.set(VReg::new(0), Allocation::Spill(spill));
        alloc_map.set(VReg::new(1), Allocation::Register(PReg::Gpr(Gpr::Rcx)));

        let mfunc = InstructionSelector::select(&graph, &alloc_map)
            .expect("instruction selection should materialize spilled parameters");

        let first_p0_inst = mfunc
            .insts
            .iter()
            .position(|inst| inst.origin == Some(p0))
            .expect("parameter 0 materialization should emit instructions");
        let first_p1_inst = mfunc
            .insts
            .iter()
            .position(|inst| inst.origin == Some(p1))
            .expect("parameter 1 materialization should emit instructions");
        assert!(
            first_p0_inst < first_p1_inst,
            "spilled parameters must be materialized before direct-register parameters"
        );

        assert!(mfunc.insts.iter().any(|inst| {
            inst.origin == Some(p0)
                && inst.op == MachineOp::Mov
                && inst.dst == MachineOperand::gpr(PARAM_STACK_TEMP_GPR)
                && inst.src1
                    == MachineOperand::Mem(MemOperand::base_disp(FRAME_BASE_SCRATCH_GPR, 0))
        }));
        assert!(mfunc.insts.iter().any(|inst| {
            inst.origin == Some(p0)
                && inst.op == MachineOp::Mov
                && inst.dst == MachineOperand::StackSlot(spill.offset())
                && inst.src1 == MachineOperand::gpr(PARAM_STACK_TEMP_GPR)
        }));
    }

    #[test]
    fn test_instruction_selection_rejects_unsupported_int_pow() {
        let mut graph = Graph::new();
        let c1 = graph.add_node(Operator::ConstInt(2), InputList::Single(graph.start));
        let c2 = graph.add_node(Operator::ConstInt(3), InputList::Single(graph.start));
        let pow = graph.add_node(
            Operator::IntOp(ArithOp::Pow),
            InputList::from_slice(&[c1, c2]),
        );
        let _ret = graph.add_node(
            Operator::Control(ControlOp::Return),
            InputList::from_slice(&[graph.start, pow]),
        );

        let alloc_map = AllocationMap::new();
        let err = InstructionSelector::select(&graph, &alloc_map)
            .expect_err("unsupported Pow lowering should return error");
        assert!(err.contains("does not support"));
    }

    #[test]
    fn test_instruction_selection_lowers_if_with_projection_targets() {
        let mut graph = Graph::new();
        let cond = graph.add_node(Operator::ConstBool(true), InputList::Single(graph.start));
        let if_node = graph.add_node(
            Operator::Control(ControlOp::If),
            InputList::from_slice(&[graph.start, cond]),
        );
        let true_proj = graph.add_node(Operator::Projection(0), InputList::Single(if_node));
        let false_proj = graph.add_node(Operator::Projection(1), InputList::Single(if_node));
        let true_val = graph.add_node(Operator::ConstInt(1), InputList::Empty);
        let false_val = graph.add_node(Operator::ConstInt(0), InputList::Empty);
        let true_ret = graph.add_node(
            Operator::Control(ControlOp::Return),
            InputList::from_slice(&[true_proj, true_val]),
        );
        let false_ret = graph.add_node(
            Operator::Control(ControlOp::Return),
            InputList::from_slice(&[false_proj, false_val]),
        );

        let alloc_map = AllocationMap::new();
        let mfunc = InstructionSelector::select(&graph, &alloc_map)
            .expect("If control with projections should lower to machine branches");

        assert!(mfunc.insts.iter().any(|inst| {
            inst.op == MachineOp::Test && inst.origin == Some(if_node) && inst.src1 == inst.src2
        }));
        assert!(mfunc.insts.iter().any(|inst| {
            inst.op == MachineOp::Jcc
                && inst.origin == Some(if_node)
                && inst.cc == Some(CondCode::Ne)
        }));
        assert!(mfunc.insts.iter().any(|inst| {
            inst.op == MachineOp::Jmp
                && inst.origin == Some(if_node)
                && matches!(inst.dst, MachineOperand::Label(_))
        }));
        assert!(
            mfunc
                .insts
                .iter()
                .any(|inst| inst.op == MachineOp::Label && inst.origin.is_none()),
            "branch targets should be materialized as labels",
        );
        assert!(
            mfunc
                .insts
                .iter()
                .any(|inst| inst.op == MachineOp::Ret && inst.origin == Some(true_ret))
        );
        assert!(
            mfunc
                .insts
                .iter()
                .any(|inst| inst.op == MachineOp::Ret && inst.origin == Some(false_ret))
        );
    }

    #[test]
    fn test_instruction_selection_rejects_projection_not_from_if() {
        let mut graph = Graph::new();
        let _bad_proj = graph.add_node(Operator::Projection(0), InputList::Single(graph.start));

        let alloc_map = AllocationMap::new();
        let err = InstructionSelector::select(&graph, &alloc_map)
            .expect_err("projection not sourced from If must fail lowering");
        assert!(err.contains("projections from If"));
    }

    #[test]
    fn test_instruction_selection_rejects_if_without_projection_edges() {
        let mut graph = Graph::new();
        let cond = graph.add_node(Operator::ConstBool(true), InputList::Single(graph.start));
        let _if_node = graph.add_node(
            Operator::Control(ControlOp::If),
            InputList::from_slice(&[graph.start, cond]),
        );

        let alloc_map = AllocationMap::new();
        let err = InstructionSelector::select(&graph, &alloc_map)
            .expect_err("If lowering must fail without any live projection targets");
        assert!(err.contains("projection-based control targets"));
    }

    #[test]
    fn test_cond_code_from_cmp_op() {
        assert_eq!(CondCode::from_cmp_op(CmpOp::Eq, true), CondCode::E);
        assert_eq!(CondCode::from_cmp_op(CmpOp::Lt, true), CondCode::L);
        assert_eq!(CondCode::from_cmp_op(CmpOp::Lt, false), CondCode::B);
    }
}
