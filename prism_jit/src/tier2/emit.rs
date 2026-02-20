//! Code Emission
//!
//! Final stage that converts machine instructions to executable bytes.
//!
//! # Responsibilities
//!
//! 1. Resolve virtual registers to physical registers
//! 2. Emit x64 machine code bytes
//! 3. Resolve labels/branches
//! 4. Generate stack maps for GC
//!
//! # Architecture
//!
//! The emitter takes a `MachineFunction` and produces an `ExecutableBuffer`
//! ready to be called.

use super::lower::{CondCode, MachineFunction, MachineInst, MachineOp, MachineOperand};
use super::safepoint_placement::{SafepointAnalyzer, SafepointEmitter};
use crate::backend::x64::encoder::Condition;
use crate::backend::x64::registers::{CallingConvention, Gpr, MemOperand, Xmm};
use crate::backend::x64::{Assembler, ExecutableBuffer, Label};
use crate::regalloc::PReg;
use std::collections::{HashMap, HashSet};

// =============================================================================
// Condition Code Conversion
// =============================================================================

impl CondCode {
    /// Convert to the encoder's Condition enum.
    pub fn to_condition(self) -> Condition {
        match self {
            CondCode::E => Condition::Equal,
            CondCode::Ne => Condition::NotEqual,
            CondCode::L => Condition::Less,
            CondCode::Le => Condition::LessEqual,
            CondCode::G => Condition::Greater,
            CondCode::Ge => Condition::GreaterEqual,
            CondCode::B => Condition::Below,
            CondCode::Be => Condition::BelowEqual,
            CondCode::A => Condition::Above,
            CondCode::Ae => Condition::AboveEqual,
            CondCode::O => Condition::Overflow,
            CondCode::No => Condition::NoOverflow,
            CondCode::S => Condition::Sign,
            CondCode::Ns => Condition::NoSign,
            CondCode::P => Condition::Parity,
            CondCode::Np => Condition::NoParity,
        }
    }
}

// =============================================================================
// Stack Map Entry
// =============================================================================

/// A stack map entry for GC safety.
#[derive(Debug, Clone)]
pub struct StackMapEntry {
    /// Offset in code where this safepoint occurs.
    pub code_offset: u32,
    /// Stack slots that contain object pointers.
    pub gc_slots: Vec<i32>,
    /// Registers that contain object pointers.
    pub gc_regs: Vec<Gpr>,
}

// =============================================================================
// Compiled Code
// =============================================================================

/// Result of code emission.
pub struct CompiledCode {
    /// The executable buffer.
    pub code: ExecutableBuffer,
    /// Stack maps for GC.
    pub stack_maps: Vec<StackMapEntry>,
    /// Frame size in bytes.
    pub frame_size: u32,
}

impl std::fmt::Debug for CompiledCode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CompiledCode")
            .field("code_len", &self.code.as_slice().len())
            .field("stack_maps", &self.stack_maps)
            .field("frame_size", &self.frame_size)
            .finish()
    }
}

// =============================================================================
// Code Emitter
// =============================================================================

/// Emits machine code from machine instructions.
pub struct CodeEmitter<'a> {
    /// The machine function to emit.
    mfunc: &'a MachineFunction,
    /// The assembler.
    asm: Assembler,
    /// Label mapping (machine label -> assembler label).
    labels: HashMap<u32, Label>,
    /// Stack maps generated.
    stack_maps: Vec<StackMapEntry>,
    /// Safepoint emitter for poll insertion.
    safepoint_emitter: Option<SafepointEmitter>,
    /// Set of instruction indices where polls are needed.
    poll_indices: HashSet<usize>,
}

/// Offset of `JitFrameState.frame_base` within the runtime frame state struct.
const JIT_FRAME_STATE_FRAME_BASE_OFFSET: i32 = 0;
/// Dedicated scratch register carrying VM frame-base pointer for Tier2 code.
const TIER2_FRAME_BASE_GPR: Gpr = Gpr::R11;

#[inline]
fn jit_state_arg_gpr() -> Gpr {
    CallingConvention::host()
        .int_arg_regs()
        .first()
        .copied()
        .unwrap_or(Gpr::Rdi)
}

impl<'a> CodeEmitter<'a> {
    /// Create a new emitter.
    pub fn new(mfunc: &'a MachineFunction) -> Self {
        CodeEmitter {
            mfunc,
            asm: Assembler::new(),
            labels: HashMap::new(),
            stack_maps: Vec::new(),
            safepoint_emitter: None,
            poll_indices: HashSet::new(),
        }
    }

    /// Emit code for the machine function.
    pub fn emit(mfunc: &'a MachineFunction) -> Result<CompiledCode, String> {
        Self::emit_with_safepoint(mfunc, None)
    }

    /// Emit code with safepoint support.
    ///
    /// If `safepoint_page_addr` is provided, safepoint polls will be emitted
    /// at loop back-edges and long straight-line code sections.
    pub fn emit_with_safepoint(
        mfunc: &'a MachineFunction,
        safepoint_page_addr: Option<usize>,
    ) -> Result<CompiledCode, String> {
        let mut emitter = CodeEmitter::new(mfunc);

        // Analyze and set up safepoint emission if page address provided
        if let Some(page_addr) = safepoint_page_addr {
            let analyzer = SafepointAnalyzer::new();
            let placement = analyzer.analyze(mfunc);
            if placement.needs_safepoint_register {
                emitter.poll_indices = placement.poll_indices.iter().copied().collect();
                emitter.safepoint_emitter = Some(SafepointEmitter::new(placement, page_addr));
            }
        }

        emitter.emit_all()?;
        emitter.finalize()
    }

    /// Pre-create all labels.
    fn create_labels(&mut self) {
        // Scan for label pseudo-instructions
        for inst in &self.mfunc.insts {
            if inst.op == MachineOp::Label {
                if let MachineOperand::Label(id) = inst.dst {
                    if !self.labels.contains_key(&id) {
                        let label = self.asm.create_label();
                        self.labels.insert(id, label);
                    }
                }
            }
        }

        // Also create labels for jump targets
        for inst in &self.mfunc.insts {
            match inst.op {
                MachineOp::Jmp | MachineOp::Jcc => {
                    if let MachineOperand::Label(id) = inst.dst {
                        if !self.labels.contains_key(&id) {
                            let label = self.asm.create_label();
                            self.labels.insert(id, label);
                        }
                    }
                }
                _ => {}
            }
        }
    }

    /// Emit all instructions.
    fn emit_all(&mut self) -> Result<(), String> {
        self.create_labels();
        self.emit_prologue();

        for i in 0..self.mfunc.insts.len() {
            self.emit_inst(&self.mfunc.insts[i])?;

            // Emit safepoint poll if needed after this instruction
            if self.poll_indices.contains(&i) {
                self.emit_safepoint_poll();
            }
        }

        Ok(())
    }

    /// Emit function prologue.
    fn emit_prologue(&mut self) {
        // Push RBP
        self.asm.push(Gpr::Rbp);
        // MOV RBP, RSP
        self.asm.mov_rr(Gpr::Rbp, Gpr::Rsp);

        // Allocate stack space for spills
        if self.mfunc.frame_size > 0 {
            // Align to 16 bytes
            let aligned_size = (self.mfunc.frame_size + 15) & !15;
            self.asm.sub_ri(Gpr::Rsp, aligned_size as i32);
        }

        // Cache `JitFrameState.frame_base` in the dedicated Tier2 scratch register.
        let state_arg = jit_state_arg_gpr();
        self.asm.mov_rm(
            TIER2_FRAME_BASE_GPR,
            &MemOperand::base_disp(state_arg, JIT_FRAME_STATE_FRAME_BASE_OFFSET),
        );

        // Load safepoint page address into R15 if needed
        if let Some(ref emitter) = self.safepoint_emitter {
            if emitter.needs_safepoint_register() {
                let page_addr = emitter.safepoint_page_addr();
                // R15 is the dedicated safepoint register
                self.asm.mov_ri64(Gpr::R15, page_addr as i64);
            }
        }
    }

    /// Emit function epilogue (called before RET).
    fn emit_epilogue(&mut self) {
        // MOV RSP, RBP
        self.asm.mov_rr(Gpr::Rsp, Gpr::Rbp);
        // POP RBP
        self.asm.pop(Gpr::Rbp);
    }

    /// Emit a single instruction.
    fn emit_inst(&mut self, inst: &MachineInst) -> Result<(), String> {
        match inst.op {
            // Labels
            MachineOp::Label => {
                if let MachineOperand::Label(id) = inst.dst {
                    if let Some(&label) = self.labels.get(&id) {
                        self.asm.bind_label(label);
                    } else {
                        return Err(format!("label {} was not created before emission", id));
                    }
                } else {
                    return Err(format!(
                        "LABEL instruction requires label operand, got {:?}",
                        inst.dst
                    ));
                }
            }

            // Data movement
            MachineOp::Mov => {
                self.emit_mov(&inst.dst, &inst.src1)?;
            }

            MachineOp::Push => {
                let gpr = self.resolve_gpr(&inst.src1)?;
                self.asm.push(gpr);
            }

            MachineOp::Pop => {
                let gpr = self.resolve_gpr(&inst.dst)?;
                self.asm.pop(gpr);
            }

            MachineOp::Lea => {
                let dst = self.resolve_gpr(&inst.dst)?;
                if let MachineOperand::Mem(mem) = &inst.src1 {
                    self.asm.lea(dst, mem);
                } else {
                    return Err(format!("LEA requires memory source, got {:?}", inst.src1));
                }
            }

            // Integer arithmetic
            MachineOp::Add => {
                self.emit_binary_arith(inst, |asm, dst, src| asm.add_rr(dst, src))?;
            }

            MachineOp::Sub => {
                self.emit_binary_arith(inst, |asm, dst, src| asm.sub_rr(dst, src))?;
            }

            MachineOp::Imul => {
                self.emit_binary_arith(inst, |asm, dst, src| asm.imul_rr(dst, src))?;
            }

            MachineOp::Neg => {
                let dst = self.resolve_gpr(&inst.dst)?;
                self.asm.neg(dst);
            }

            MachineOp::Inc => {
                let dst = self.resolve_gpr(&inst.dst)?;
                self.asm.inc(dst);
            }

            MachineOp::Dec => {
                let dst = self.resolve_gpr(&inst.dst)?;
                self.asm.dec(dst);
            }

            MachineOp::Idiv => {
                // IDIV uses RDX:RAX / src
                let src = self.resolve_gpr(&inst.src1)?;
                self.asm.idiv(src);
            }

            MachineOp::Cqo => {
                self.asm.cqo();
            }

            MachineOp::Cdq => {
                self.asm.cdq();
            }

            // Bitwise
            MachineOp::And => {
                self.emit_binary_arith(inst, |asm, dst, src| asm.and_rr(dst, src))?;
            }

            MachineOp::Or => {
                self.emit_binary_arith(inst, |asm, dst, src| asm.or_rr(dst, src))?;
            }

            MachineOp::Xor => {
                self.emit_binary_arith(inst, |asm, dst, src| asm.xor_rr(dst, src))?;
            }

            MachineOp::Not => {
                let dst = self.resolve_gpr(&inst.dst)?;
                self.asm.not(dst);
            }

            MachineOp::Shl => {
                let dst = self.resolve_gpr(&inst.dst)?;
                match &inst.src2 {
                    MachineOperand::Imm(imm) => {
                        self.asm.shl_ri(dst, *imm as u8);
                    }
                    _ => {
                        // Shift by CL
                        self.asm.shl_cl(dst);
                    }
                }
            }

            MachineOp::Sar => {
                let dst = self.resolve_gpr(&inst.dst)?;
                match &inst.src2 {
                    MachineOperand::Imm(imm) => {
                        self.asm.sar_ri(dst, *imm as u8);
                    }
                    _ => {
                        self.asm.sar_cl(dst);
                    }
                }
            }

            MachineOp::Shr => {
                let dst = self.resolve_gpr(&inst.dst)?;
                match &inst.src2 {
                    MachineOperand::Imm(imm) => {
                        self.asm.shr_ri(dst, *imm as u8);
                    }
                    _ => {
                        self.asm.shr_cl(dst);
                    }
                }
            }

            // Comparison
            MachineOp::Cmp => {
                let lhs = self.resolve_gpr(&inst.src1)?;
                if let MachineOperand::Imm(imm) = &inst.src2 {
                    self.asm.cmp_ri(lhs, *imm as i32);
                } else {
                    let rhs = self.resolve_gpr(&inst.src2)?;
                    self.asm.cmp_rr(lhs, rhs);
                }
            }

            MachineOp::Test => {
                let lhs = self.resolve_gpr(&inst.src1)?;
                if let MachineOperand::Imm(imm) = &inst.src2 {
                    self.asm.test_ri(lhs, *imm as i32);
                } else {
                    let rhs = self.resolve_gpr(&inst.src2)?;
                    self.asm.test_rr(lhs, rhs);
                }
            }

            MachineOp::Setcc => {
                let cc = inst
                    .cc
                    .ok_or_else(|| "SETcc missing condition code".to_string())?;
                let dst = self.resolve_gpr(&inst.dst)?;
                self.asm.setcc(cc.to_condition(), dst);
                // Zero-extend to full register
                self.asm.movzx_rb(dst, dst);
            }

            // Control flow
            MachineOp::Jmp => {
                if let MachineOperand::Label(id) = inst.dst {
                    if let Some(&label) = self.labels.get(&id) {
                        self.asm.jmp(label);
                    } else {
                        return Err(format!("JMP to unknown label {}", id));
                    }
                } else {
                    return Err(format!("JMP target must be label, got {:?}", inst.dst));
                }
            }

            MachineOp::Jcc => {
                if let (Some(cc), MachineOperand::Label(id)) = (inst.cc, inst.dst) {
                    if let Some(&label) = self.labels.get(&id) {
                        self.asm.jcc(cc.to_condition(), label);
                    } else {
                        return Err(format!("JCC to unknown label {}", id));
                    }
                } else {
                    return Err(format!(
                        "JCC requires condition code and label target, got cc={:?}, dst={:?}",
                        inst.cc, inst.dst
                    ));
                }
            }

            MachineOp::Call => {
                // Record safepoint for GC
                self.record_safepoint();

                if let MachineOperand::Imm(addr) = inst.dst {
                    // Call to absolute address
                    self.asm.mov_ri64(Gpr::R11, addr);
                    self.asm.call_r(Gpr::R11);
                } else if let Ok(target) = self.resolve_gpr(&inst.dst) {
                    self.asm.call_r(target);
                } else {
                    return Err(format!(
                        "CALL target must be immediate or GPR, got {:?}",
                        inst.dst
                    ));
                }
            }

            MachineOp::Ret => {
                self.emit_epilogue();
                self.asm.ret();
            }

            // SSE floating-point
            MachineOp::Movsd => {
                self.emit_movsd(inst)?;
            }

            MachineOp::Addsd => {
                self.emit_binary_sse(inst, |asm, dst, src| asm.addsd(dst, src))?;
            }

            MachineOp::Subsd => {
                self.emit_binary_sse(inst, |asm, dst, src| asm.subsd(dst, src))?;
            }

            MachineOp::Mulsd => {
                self.emit_binary_sse(inst, |asm, dst, src| asm.mulsd(dst, src))?;
            }

            MachineOp::Divsd => {
                self.emit_binary_sse(inst, |asm, dst, src| asm.divsd(dst, src))?;
            }

            MachineOp::Ucomisd => {
                let lhs = self.resolve_xmm(&inst.src1)?;
                let rhs = self.resolve_xmm(&inst.src2)?;
                self.asm.ucomisd(lhs, rhs);
            }

            MachineOp::Xorpd => {
                let dst = self.resolve_xmm(&inst.dst)?;
                self.asm.xorpd(dst, dst);
            }

            MachineOp::Cvtsi2sd => {
                let dst = self.resolve_xmm(&inst.dst)?;
                let src = self.resolve_gpr(&inst.src1)?;
                self.asm.cvtsi2sd(dst, src);
            }

            MachineOp::Cvttsd2si => {
                let dst = self.resolve_gpr(&inst.dst)?;
                let src = self.resolve_xmm(&inst.src1)?;
                self.asm.cvttsd2si(dst, src);
            }

            // Spill/reload (generated during regalloc)
            MachineOp::Spill => {
                let src = self.resolve_gpr(&inst.src1)?;
                if let MachineOperand::StackSlot(offset) = &inst.dst {
                    // MOV [RBP + offset], src
                    self.asm
                        .mov_mr(&MemOperand::base_disp(Gpr::Rbp, *offset), src);
                } else {
                    return Err(format!(
                        "invalid SPILL operands dst={:?}, src={:?}",
                        inst.dst, inst.src1
                    ));
                }
            }

            MachineOp::Reload => {
                let dst = self.resolve_gpr(&inst.dst)?;
                if let MachineOperand::StackSlot(offset) = &inst.src1 {
                    // MOV dst, [RBP + offset]
                    self.asm
                        .mov_rm(dst, &MemOperand::base_disp(Gpr::Rbp, *offset));
                } else {
                    return Err(format!(
                        "invalid RELOAD operands dst={:?}, src={:?}",
                        inst.dst, inst.src1
                    ));
                }
            }

            // Special
            MachineOp::Nop => {
                self.asm.nop();
            }

            MachineOp::Int3 => {
                self.asm.int3();
            }

            _ => {
                return Err(format!(
                    "machine op {:?} is not supported by x64 emitter",
                    inst.op
                ));
            }
        }

        Ok(())
    }

    /// Emit a MOV instruction.
    fn emit_mov(&mut self, dst: &MachineOperand, src: &MachineOperand) -> Result<(), String> {
        match (dst, src) {
            (
                MachineOperand::PReg(PReg::Gpr(dst_gpr)),
                MachineOperand::PReg(PReg::Gpr(src_gpr)),
            ) => {
                self.asm.mov_rr(*dst_gpr, *src_gpr);
            }
            (MachineOperand::PReg(PReg::Gpr(dst_gpr)), MachineOperand::Imm(imm)) => {
                if *imm >= 0 && *imm <= u32::MAX as i64 {
                    self.asm.mov_ri32(*dst_gpr, *imm as u32);
                } else {
                    self.asm.mov_ri64(*dst_gpr, *imm);
                }
            }
            (MachineOperand::PReg(PReg::Gpr(dst_gpr)), MachineOperand::Mem(mem)) => {
                self.asm.mov_rm(*dst_gpr, mem);
            }
            (MachineOperand::Mem(mem), MachineOperand::PReg(PReg::Gpr(src_gpr))) => {
                self.asm.mov_mr(mem, *src_gpr);
            }
            (MachineOperand::StackSlot(offset), MachineOperand::PReg(PReg::Gpr(src_gpr))) => {
                self.asm
                    .mov_mr(&MemOperand::base_disp(Gpr::Rbp, *offset), *src_gpr);
            }
            (MachineOperand::PReg(PReg::Gpr(dst_gpr)), MachineOperand::StackSlot(offset)) => {
                self.asm
                    .mov_rm(*dst_gpr, &MemOperand::base_disp(Gpr::Rbp, *offset));
            }
            // VReg cases - these should be resolved by regalloc
            (MachineOperand::VReg(_), _) | (_, MachineOperand::VReg(_)) => {
                return Err(format!(
                    "unresolved virtual register in MOV dst={:?}, src={:?}",
                    dst, src
                ));
            }
            _ => {
                return Err(format!(
                    "unsupported MOV operand combination dst={:?}, src={:?}",
                    dst, src
                ));
            }
        }
        Ok(())
    }

    /// Emit a MOVSD instruction.
    fn emit_movsd(&mut self, inst: &MachineInst) -> Result<(), String> {
        match (&inst.dst, &inst.src1) {
            (MachineOperand::PReg(PReg::Xmm(dst)), MachineOperand::PReg(PReg::Xmm(src))) => {
                self.asm.movsd_rr(*dst, *src);
            }
            (MachineOperand::PReg(PReg::Xmm(dst)), MachineOperand::Mem(mem)) => {
                self.asm.movsd_rm(*dst, mem);
            }
            (MachineOperand::Mem(mem), MachineOperand::PReg(PReg::Xmm(src))) => {
                self.asm.movsd_mr(mem, *src);
            }
            _ => {
                return Err(format!(
                    "unsupported MOVSD operands dst={:?}, src={:?}",
                    inst.dst, inst.src1
                ));
            }
        }
        Ok(())
    }

    /// Emit a binary GPR arithmetic instruction.
    fn emit_binary_arith<F>(&mut self, inst: &MachineInst, emit_fn: F) -> Result<(), String>
    where
        F: FnOnce(&mut Assembler, Gpr, Gpr),
    {
        let dst = self.resolve_gpr(&inst.dst)?;
        if let MachineOperand::Imm(imm) = &inst.src2 {
            // Need to handle immediate operands
            match inst.op {
                MachineOp::Add => self.asm.add_ri(dst, *imm as i32),
                MachineOp::Sub => self.asm.sub_ri(dst, *imm as i32),
                MachineOp::And => self.asm.and_ri(dst, *imm as i32),
                MachineOp::Or => self.asm.or_ri(dst, *imm as i32),
                MachineOp::Xor => self.asm.xor_ri(dst, *imm as i32),
                _ => {
                    return Err(format!(
                        "immediate form not supported for {:?} with operand {:?}",
                        inst.op, inst.src2
                    ));
                }
            }
        } else {
            let src = self.resolve_gpr(&inst.src2)?;
            emit_fn(&mut self.asm, dst, src);
        }
        Ok(())
    }

    /// Emit a binary SSE instruction.
    fn emit_binary_sse<F>(&mut self, inst: &MachineInst, emit_fn: F) -> Result<(), String>
    where
        F: FnOnce(&mut Assembler, Xmm, Xmm),
    {
        let dst = self.resolve_xmm(&inst.dst)?;
        let src = self.resolve_xmm(&inst.src2)?;
        emit_fn(&mut self.asm, dst, src);
        Ok(())
    }

    /// Resolve an operand to a GPR.
    fn resolve_gpr(&self, op: &MachineOperand) -> Result<Gpr, String> {
        match op {
            MachineOperand::PReg(PReg::Gpr(gpr)) => Ok(*gpr),
            MachineOperand::VReg(_) => Err(format!("unresolved virtual register operand {:?}", op)),
            _ => Err(format!("expected GPR operand, got {:?}", op)),
        }
    }

    /// Resolve an operand to an XMM register.
    fn resolve_xmm(&self, op: &MachineOperand) -> Result<Xmm, String> {
        match op {
            MachineOperand::PReg(PReg::Xmm(xmm)) => Ok(*xmm),
            MachineOperand::VReg(_) => Err(format!("unresolved virtual register operand {:?}", op)),
            _ => Err(format!("expected XMM operand, got {:?}", op)),
        }
    }

    /// Record a safepoint for GC.
    fn record_safepoint(&mut self) {
        let offset = self.asm.offset() as u32;
        self.stack_maps.push(StackMapEntry {
            code_offset: offset,
            gc_slots: self.mfunc.gc_roots.stack_slots.clone(),
            gc_regs: self.mfunc.gc_roots.regs.clone(),
        });
    }

    /// Emit a safepoint poll instruction.
    ///
    /// This emits `test [r15], al` which is a 3-byte memory read.
    /// When the safepoint page is protected, this will fault and trigger
    /// the signal/exception handler which will stop the thread for GC.
    fn emit_safepoint_poll(&mut self) {
        // test byte ptr [r15], al
        // Encoding: 41 84 07
        //   41 = REX.B prefix (R15 is an extended register)
        //   84 = TEST r/m8, r8 opcode
        //   07 = ModR/M: mod=00 (memory), reg=000 (AL), r/m=111 (R15)
        self.asm.emit_bytes(&[0x41, 0x84, 0x07]);

        // Record this as a safepoint for stack map generation
        self.record_safepoint();
    }

    /// Finalize and produce executable code.
    fn finalize(self) -> Result<CompiledCode, String> {
        let code = self.asm.finalize_executable()?;

        Ok(CompiledCode {
            code,
            stack_maps: self.stack_maps,
            frame_size: self.mfunc.frame_size,
        })
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
    use crate::regalloc::{AllocationMap, VReg};
    use crate::tier2::lower::InstructionSelector;

    #[test]
    fn test_code_emitter_empty() {
        let mfunc = MachineFunction::new();
        let result = CodeEmitter::emit(&mfunc);
        assert!(result.is_ok());
    }

    #[test]
    fn test_code_emitter_with_mov() {
        let mut mfunc = MachineFunction::new();
        mfunc.push(MachineInst::new(
            MachineOp::Mov,
            MachineOperand::gpr(Gpr::Rax),
            MachineOperand::Imm(42),
        ));
        mfunc.push(MachineInst::nullary(MachineOp::Ret));

        let result = CodeEmitter::emit(&mfunc);
        assert!(result.is_ok());

        let code = result.unwrap();
        assert!(!code.code.as_slice().is_empty());
    }

    #[test]
    fn test_code_emitter_with_arithmetic() {
        let mut mfunc = MachineFunction::new();

        // MOV RAX, 10
        mfunc.push(MachineInst::new(
            MachineOp::Mov,
            MachineOperand::gpr(Gpr::Rax),
            MachineOperand::Imm(10),
        ));

        // MOV RBX, 5
        mfunc.push(MachineInst::new(
            MachineOp::Mov,
            MachineOperand::gpr(Gpr::Rbx),
            MachineOperand::Imm(5),
        ));

        // ADD RAX, RBX
        mfunc.push(MachineInst::binary(
            MachineOp::Add,
            MachineOperand::gpr(Gpr::Rax),
            MachineOperand::gpr(Gpr::Rax),
            MachineOperand::gpr(Gpr::Rbx),
        ));

        // RET
        mfunc.push(MachineInst::nullary(MachineOp::Ret));

        let result = CodeEmitter::emit(&mfunc);
        assert!(result.is_ok());
    }

    #[test]
    fn test_stack_map_entry() {
        let entry = StackMapEntry {
            code_offset: 0x10,
            gc_slots: vec![-8, -16],
            gc_regs: vec![Gpr::Rax, Gpr::Rbx],
        };
        assert_eq!(entry.gc_slots.len(), 2);
        assert_eq!(entry.gc_regs.len(), 2);
    }

    #[test]
    fn test_call_safepoint_uses_machine_gc_roots() {
        let mut mfunc = MachineFunction::new();
        mfunc.gc_roots.stack_slots = vec![-24, -8];
        mfunc.gc_roots.regs = vec![Gpr::Rbx, Gpr::R12];
        mfunc.push(MachineInst::new(
            MachineOp::Call,
            MachineOperand::Imm(0x1234),
            MachineOperand::None,
        ));
        mfunc.push(MachineInst::nullary(MachineOp::Ret));

        let code = CodeEmitter::emit(&mfunc).expect("emission should succeed");
        assert_eq!(code.stack_maps.len(), 1);
        assert_eq!(code.stack_maps[0].gc_slots, vec![-24, -8]);
        assert_eq!(code.stack_maps[0].gc_regs, vec![Gpr::Rbx, Gpr::R12]);
    }

    #[test]
    fn test_poll_safepoint_uses_machine_gc_roots() {
        let mut mfunc = MachineFunction::new();
        let loop_label = mfunc.new_label();
        mfunc.gc_roots.stack_slots = vec![-32];
        mfunc.gc_roots.regs = vec![Gpr::R13];
        mfunc.add_label(loop_label);
        mfunc.push(MachineInst::nullary(MachineOp::Nop));
        mfunc.push(MachineInst::new(
            MachineOp::Jmp,
            MachineOperand::Label(loop_label),
            MachineOperand::None,
        ));

        let code = CodeEmitter::emit_with_safepoint(&mfunc, Some(0x1000))
            .expect("safepoint-enabled emission should succeed");
        assert!(
            !code.stack_maps.is_empty(),
            "loop back-edge should produce at least one safepoint poll"
        );
        assert!(code
            .stack_maps
            .iter()
            .all(|entry| entry.gc_slots == vec![-32] && entry.gc_regs == vec![Gpr::R13]));
    }

    #[test]
    fn test_code_emitter_rejects_unresolved_vreg_operand() {
        let mut mfunc = MachineFunction::new();
        mfunc.push(MachineInst::new(
            MachineOp::Mov,
            MachineOperand::gpr(Gpr::Rax),
            MachineOperand::VReg(VReg::new(0)),
        ));
        mfunc.push(MachineInst::nullary(MachineOp::Ret));

        let err =
            CodeEmitter::emit(&mfunc).expect_err("unresolved virtual registers must fail emission");
        assert!(err.contains("unresolved virtual register"));
    }

    #[test]
    fn test_code_emitter_rejects_unsupported_machine_op() {
        let mut mfunc = MachineFunction::new();
        mfunc.push(MachineInst::nullary(MachineOp::Vaddpd256));
        mfunc.push(MachineInst::nullary(MachineOp::Ret));

        let err = CodeEmitter::emit(&mfunc).expect_err("unsupported machine op must fail emission");
        assert!(err.contains("not supported"));
    }

    #[test]
    fn test_full_pipeline() {
        // Build IR
        let mut builder = GraphBuilder::new(4, 0);
        let const_1 = builder.const_int(1);
        let const_2 = builder.const_int(2);
        let sum = builder.int_add(const_1, const_2);
        let _ret = builder.return_value(sum);
        let graph = builder.finish();

        // Instruction selection
        let alloc_map = AllocationMap::new();
        let mfunc = InstructionSelector::select(&graph, &alloc_map)
            .expect("instruction selection should succeed");

        // Code emission (will have virtual registers)
        // This is a smoke test - full regalloc integration would resolve VRegs
        assert!(!mfunc.insts.is_empty());
    }

    #[test]
    fn test_cond_code_conversion() {
        assert_eq!(CondCode::E.to_condition(), Condition::Equal);
        assert_eq!(CondCode::L.to_condition(), Condition::Less);
        assert_eq!(CondCode::G.to_condition(), Condition::Greater);
    }
}
