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

impl<'a> InstructionSelector<'a> {
    /// Create a new instruction selector.
    pub fn new(graph: &'a Graph, alloc_map: &'a AllocationMap) -> Self {
        InstructionSelector {
            graph,
            alloc_map,
            mfunc: MachineFunction::new(),
            node_to_vreg: HashMap::new(),
            next_vreg: 0,
        }
    }

    /// Select instructions for the entire graph.
    pub fn select(graph: &'a Graph, alloc_map: &'a AllocationMap) -> MachineFunction {
        let mut selector = InstructionSelector::new(graph, alloc_map);
        selector.select_all();
        selector.mfunc
    }

    /// Select instructions for all nodes.
    fn select_all(&mut self) {
        // Process nodes in order (simplified - real impl uses RPO)
        for (id, node) in self.graph.iter() {
            if node.is_dead() {
                continue;
            }
            self.select_node(id);
        }
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

    /// Select instructions for a single node.
    fn select_node(&mut self, node_id: NodeId) {
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
            }

            Operator::ConstFloat(bits) => {
                // Load float from constant pool (simplified - use RIP-relative)
                let dst = self.operand_for_node(node_id);
                self.mfunc.push(
                    MachineInst::new(MachineOp::Mov, dst, MachineOperand::Imm(bits as i64))
                        .with_origin(node_id),
                );
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
            }

            Operator::ConstNone => {
                let dst = self.operand_for_node(node_id);
                // None is encoded as 0 in our value representation
                self.mfunc.push(
                    MachineInst::new(MachineOp::Mov, dst, MachineOperand::Imm(0))
                        .with_origin(node_id),
                );
            }

            // Arithmetic operations (note: IntOp, FloatOp, not IntArith/FloatArith)
            Operator::IntOp(arith_op) => {
                self.select_int_arith(node_id, arith_op);
            }

            Operator::FloatOp(arith_op) => {
                self.select_float_arith(node_id, arith_op);
            }

            // Comparisons
            Operator::IntCmp(cmp_op) => {
                self.select_int_cmp(node_id, cmp_op);
            }

            Operator::FloatCmp(cmp_op) => {
                self.select_float_cmp(node_id, cmp_op);
            }

            // Bitwise operations
            Operator::Bitwise(bitwise_op) => {
                self.select_bitwise(node_id, bitwise_op);
            }

            // Control flow
            Operator::Control(ctrl_op) => {
                self.select_control(node_id, ctrl_op);
            }

            // Parameters and Phi are handled specially
            Operator::Parameter(_) | Operator::Phi | Operator::LoopPhi => {
                // These don't generate code directly
            }

            // Other operators
            _ => {
                // Not yet implemented - would be runtime calls
            }
        }
    }
    /// Select instructions for integer arithmetic.
    fn select_int_arith(&mut self, node_id: NodeId, op: ArithOp) {
        let dst = self.operand_for_node(node_id);

        match op {
            ArithOp::Add => {
                if let (Some(lhs), Some(rhs)) = (
                    self.operand_for_input(node_id, 0),
                    self.operand_for_input(node_id, 1),
                ) {
                    self.emit_binary(MachineOp::Add, dst, lhs, rhs, node_id);
                }
            }

            ArithOp::Sub => {
                if let (Some(lhs), Some(rhs)) = (
                    self.operand_for_input(node_id, 0),
                    self.operand_for_input(node_id, 1),
                ) {
                    self.emit_binary(MachineOp::Sub, dst, lhs, rhs, node_id);
                }
            }

            ArithOp::Mul => {
                if let (Some(lhs), Some(rhs)) = (
                    self.operand_for_input(node_id, 0),
                    self.operand_for_input(node_id, 1),
                ) {
                    self.emit_binary(MachineOp::Imul, dst, lhs, rhs, node_id);
                }
            }

            ArithOp::TrueDiv | ArithOp::FloorDiv => {
                // Division is complex: RDX:RAX / divisor
                if let (Some(lhs), Some(rhs)) = (
                    self.operand_for_input(node_id, 0),
                    self.operand_for_input(node_id, 1),
                ) {
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
                }
            }

            ArithOp::Mod => {
                // Modulo: remainder is in RDX after IDIV
                if let (Some(lhs), Some(rhs)) = (
                    self.operand_for_input(node_id, 0),
                    self.operand_for_input(node_id, 1),
                ) {
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
                }
            }

            ArithOp::Neg => {
                if let Some(src) = self.operand_for_input(node_id, 0) {
                    // NEG is unary: dst = -dst
                    if dst != src {
                        self.mfunc.push(MachineInst::new(MachineOp::Mov, dst, src));
                    }
                    self.mfunc
                        .push(MachineInst::new(MachineOp::Neg, dst, dst).with_origin(node_id));
                }
            }

            _ => {
                // Other ops (Pow, MatMul, etc.) not yet implemented
            }
        }
    }

    /// Select instructions for floating-point arithmetic.
    fn select_float_arith(&mut self, node_id: NodeId, op: ArithOp) {
        let dst = self.operand_for_node(node_id);

        if let (Some(lhs), Some(rhs)) = (
            self.operand_for_input(node_id, 0),
            self.operand_for_input(node_id, 1),
        ) {
            let machine_op = match op {
                ArithOp::Add => MachineOp::Addsd,
                ArithOp::Sub => MachineOp::Subsd,
                ArithOp::Mul => MachineOp::Mulsd,
                ArithOp::TrueDiv => MachineOp::Divsd,
                _ => return, // Not supported
            };

            self.emit_binary(machine_op, dst, lhs, rhs, node_id);
        }
    }

    /// Select instructions for integer comparison.
    fn select_int_cmp(&mut self, node_id: NodeId, cmp_op: CmpOp) {
        let dst = self.operand_for_node(node_id);

        if let (Some(lhs), Some(rhs)) = (
            self.operand_for_input(node_id, 0),
            self.operand_for_input(node_id, 1),
        ) {
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
        }
    }

    /// Select instructions for floating-point comparison.
    fn select_float_cmp(&mut self, node_id: NodeId, cmp_op: CmpOp) {
        let dst = self.operand_for_node(node_id);

        if let (Some(lhs), Some(rhs)) = (
            self.operand_for_input(node_id, 0),
            self.operand_for_input(node_id, 1),
        ) {
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
        }
    }

    /// Select instructions for bitwise operations.
    fn select_bitwise(&mut self, node_id: NodeId, op: BitwiseOp) {
        let dst = self.operand_for_node(node_id);

        match op {
            BitwiseOp::And => {
                if let (Some(lhs), Some(rhs)) = (
                    self.operand_for_input(node_id, 0),
                    self.operand_for_input(node_id, 1),
                ) {
                    self.emit_binary(MachineOp::And, dst, lhs, rhs, node_id);
                }
            }

            BitwiseOp::Or => {
                if let (Some(lhs), Some(rhs)) = (
                    self.operand_for_input(node_id, 0),
                    self.operand_for_input(node_id, 1),
                ) {
                    self.emit_binary(MachineOp::Or, dst, lhs, rhs, node_id);
                }
            }

            BitwiseOp::Xor => {
                if let (Some(lhs), Some(rhs)) = (
                    self.operand_for_input(node_id, 0),
                    self.operand_for_input(node_id, 1),
                ) {
                    self.emit_binary(MachineOp::Xor, dst, lhs, rhs, node_id);
                }
            }

            BitwiseOp::Not => {
                if let Some(src) = self.operand_for_input(node_id, 0) {
                    if dst != src {
                        self.mfunc.push(MachineInst::new(MachineOp::Mov, dst, src));
                    }
                    self.mfunc
                        .push(MachineInst::new(MachineOp::Not, dst, dst).with_origin(node_id));
                }
            }

            BitwiseOp::Shl => {
                if let (Some(lhs), Some(rhs)) = (
                    self.operand_for_input(node_id, 0),
                    self.operand_for_input(node_id, 1),
                ) {
                    // Shift count must be in CL for variable shifts
                    self.emit_shift(MachineOp::Shl, dst, lhs, rhs, node_id);
                }
            }

            BitwiseOp::Shr => {
                if let (Some(lhs), Some(rhs)) = (
                    self.operand_for_input(node_id, 0),
                    self.operand_for_input(node_id, 1),
                ) {
                    self.emit_shift(MachineOp::Sar, dst, lhs, rhs, node_id);
                }
            }
        }
    }
    /// Select instructions for control flow.
    fn select_control(&mut self, node_id: NodeId, op: ControlOp) {
        match op {
            ControlOp::Start => {
                // Function entry - nothing to emit
            }

            ControlOp::End => {
                // Function exit - nothing to emit
            }

            ControlOp::Return => {
                // Return value should already be in RAX
                // inputs[0] is control, inputs[1] is return value
                if let Some(ret_val) = self.operand_for_input(node_id, 1) {
                    self.mfunc.push(MachineInst::new(
                        MachineOp::Mov,
                        MachineOperand::gpr(Gpr::Rax),
                        ret_val,
                    ));
                }
                self.mfunc
                    .push(MachineInst::nullary(MachineOp::Ret).with_origin(node_id));
            }

            ControlOp::If => {
                // Conditional branch (if-then-else)
                // inputs[0]: control, inputs[1]: condition
                if let Some(cond) = self.operand_for_input(node_id, 1) {
                    // Test condition
                    self.mfunc.push(MachineInst::binary(
                        MachineOp::Test,
                        MachineOperand::None,
                        cond,
                        cond,
                    ));

                    // Conditional jumps would be resolved later based on graph structure
                }
            }

            ControlOp::Region | ControlOp::Loop => {
                // Merge point - emit a label
                let label = self.mfunc.get_or_create_label(node_id.index());
                self.mfunc.add_label(label);
            }

            _ => {
                // Other control ops not yet implemented (Throw, Deopt)
            }
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
        let mut builder = GraphBuilder::new(4, 1);
        let p0 = builder.parameter(0).unwrap();
        let const_1 = builder.const_int(1);
        let sum = builder.int_add(p0, const_1);
        let _ret = builder.return_value(sum);
        let graph = builder.finish();

        let alloc_map = AllocationMap::new();
        let mfunc = InstructionSelector::select(&graph, &alloc_map);

        // Should have generated some instructions
        assert!(!mfunc.insts.is_empty());
    }

    #[test]
    fn test_cond_code_from_cmp_op() {
        assert_eq!(CondCode::from_cmp_op(CmpOp::Eq, true), CondCode::E);
        assert_eq!(CondCode::from_cmp_op(CmpOp::Lt, true), CondCode::L);
        assert_eq!(CondCode::from_cmp_op(CmpOp::Lt, false), CondCode::B);
    }
}
