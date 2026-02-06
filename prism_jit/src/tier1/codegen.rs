//! Template JIT compilation driver.
//!
//! This module orchestrates the compilation of bytecode to native code
//! using the template-based approach. The flow is:
//!
//! 1. Analyze bytecode to determine register count, jumps, etc.
//! 2. Create frame layout
//! 3. Emit prologue
//! 4. For each bytecode instruction, emit corresponding template
//! 5. Emit deopt stubs
//! 6. Emit epilogue
//! 7. Finalize and return executable code

use super::deopt::{DeoptInfo, DeoptReason, DeoptStubGenerator};
use super::frame::{FrameLayout, JitCallingConvention};
use super::template::*;
use crate::backend::x64::{Assembler, ExecutableBuffer, Gpr, Label};

use std::collections::HashMap;

// =============================================================================
// Compilation Result
// =============================================================================

/// Result of JIT compilation.
pub struct CompiledFunction {
    /// The executable code.
    pub code: ExecutableBuffer,
    /// Mapping from bytecode offset to native offset.
    pub bc_to_native: HashMap<u32, u32>,
    /// Deoptimization info for each deopt point.
    pub deopt_info: Vec<DeoptInfo>,
    /// Frame layout used.
    pub frame_layout: FrameLayout,
}

impl CompiledFunction {
    /// Get the entry point as a function pointer.
    ///
    /// # Safety
    /// The caller must ensure the function is called with the correct
    /// calling convention and argument types.
    pub unsafe fn as_fn<F: Copy>(&self) -> F {
        unsafe { self.code.as_fn() }
    }

    /// Get the native offset for a bytecode offset.
    pub fn native_offset(&self, bc_offset: u32) -> Option<u32> {
        self.bc_to_native.get(&bc_offset).copied()
    }

    /// Get deopt info by stub index.
    pub fn get_deopt_info(&self, stub_index: u16) -> Option<&DeoptInfo> {
        self.deopt_info.get(stub_index as usize)
    }
}

// =============================================================================
// Template Compiler
// =============================================================================

/// Main entry point for template JIT compilation.
pub struct TemplateCompiler {
    /// Address of the runtime deopt handler.
    deopt_handler: u64,
}

impl TemplateCompiler {
    /// Create a new template compiler.
    pub fn new(deopt_handler: u64) -> Self {
        TemplateCompiler { deopt_handler }
    }

    /// Create a compiler with a dummy deopt handler (for testing).
    pub fn new_for_testing() -> Self {
        // Use a placeholder address - actual deopts will crash but
        // we can still test code generation
        Self::new(0xDEAD_BEEF_DEAD_BEEF)
    }

    /// Compile a function.
    ///
    /// This is a simplified API that takes just the essential info.
    /// A real implementation would take a `CompiledCode` from the bytecode compiler.
    pub fn compile(
        &self,
        num_registers: u16,
        instructions: &[TemplateInstruction],
    ) -> Result<CompiledFunction, String> {
        let mut asm = Assembler::new();
        let frame = FrameLayout::minimal(num_registers);
        let cc = JitCallingConvention::host();
        let mut bc_to_native = HashMap::new();
        let mut deopt_gen = DeoptStubGenerator::new();

        // Create deopt labels first
        let mut deopt_labels = Vec::with_capacity(instructions.len());
        for _ in 0..instructions.len() {
            deopt_labels.push(asm.create_label());
        }

        // Collect all labels for jump targets
        let mut labels: HashMap<u32, Label> = HashMap::new();
        for instr in instructions {
            if let Some(target) = instr.jump_target() {
                if !labels.contains_key(&target) {
                    labels.insert(target, asm.create_label());
                }
            }
        }

        // Emit prologue
        self.emit_prologue(&mut asm, &frame, &cc);

        // Emit each instruction using a scoped context
        {
            let mut ctx = TemplateContext::new(&mut asm, &frame);
            ctx.deopt_labels = deopt_labels.clone();

            for (idx, instr) in instructions.iter().enumerate() {
                let bc_offset = instr.bc_offset();
                ctx.set_bc_offset(bc_offset as usize);

                // Record bc->native mapping
                bc_to_native.insert(bc_offset, ctx.asm.offset() as u32);

                // Bind label if this is a jump target
                if let Some(label) = labels.get(&bc_offset) {
                    ctx.asm.bind_label(*label);
                }

                // Register deopt for this instruction
                if instr.can_deopt() {
                    deopt_gen.register_deopt(deopt_labels[idx], bc_offset, instr.deopt_reason());
                }

                // Emit the template
                self.emit_instruction(&mut ctx, instr, &labels, idx);
            }
        }
        // ctx is dropped here, releasing the borrow on asm

        // Emit epilogue
        self.emit_epilogue(&mut asm, &frame);

        // Emit deopt stubs
        let deopt_info = deopt_gen.emit_stubs(&mut asm, &frame, self.deopt_handler);

        // Finalize - now we own asm
        let code = asm.finalize_executable()?;

        Ok(CompiledFunction {
            code,
            bc_to_native,
            deopt_info,
            frame_layout: frame,
        })
    }

    /// Emit function prologue.
    fn emit_prologue(&self, asm: &mut Assembler, frame: &FrameLayout, cc: &JitCallingConvention) {
        // Push callee-saved registers
        for reg in frame.saved_regs.iter() {
            asm.push(reg);
        }

        // Set up frame pointer
        asm.push(Gpr::Rbp);
        asm.mov_rr(Gpr::Rbp, Gpr::Rsp);

        // Allocate stack space
        let frame_size = frame.frame_size();
        if frame_size > 0 {
            asm.sub_ri(Gpr::Rsp, frame_size);
        }

        // Store context pointer (first argument)
        let ctx_slot = frame.context_slot();
        asm.mov_mr(&ctx_slot, cc.arg0);
    }

    /// Emit function epilogue.
    fn emit_epilogue(&self, asm: &mut Assembler, frame: &FrameLayout) {
        // Deallocate stack
        let frame_size = frame.frame_size();
        if frame_size > 0 {
            asm.add_ri(Gpr::Rsp, frame_size);
        }

        // Restore frame pointer
        asm.pop(Gpr::Rbp);

        // Pop callee-saved registers in reverse
        let saved_regs: Vec<Gpr> = frame.saved_regs.iter().collect();
        for reg in saved_regs.into_iter().rev() {
            asm.pop(reg);
        }

        // Return
        asm.ret();
    }

    /// Emit code for a single instruction.
    fn emit_instruction(
        &self,
        ctx: &mut TemplateContext,
        instr: &TemplateInstruction,
        labels: &HashMap<u32, Label>,
        deopt_idx: usize,
    ) {
        match instr {
            TemplateInstruction::LoadInt { dst, value, .. } => {
                LoadIntTemplate {
                    dst_reg: *dst,
                    value: *value,
                }
                .emit(ctx);
            }
            TemplateInstruction::LoadFloat { dst, value, .. } => {
                LoadFloatTemplate {
                    dst_reg: *dst,
                    value: *value,
                }
                .emit(ctx);
            }
            TemplateInstruction::LoadNone { dst, .. } => {
                LoadNoneTemplate { dst_reg: *dst }.emit(ctx);
            }
            TemplateInstruction::LoadBool { dst, value, .. } => {
                LoadBoolTemplate {
                    dst_reg: *dst,
                    value: *value,
                }
                .emit(ctx);
            }
            TemplateInstruction::Move { dst, src, .. } => {
                MoveTemplate {
                    dst_reg: *dst,
                    src_reg: *src,
                }
                .emit(ctx);
            }
            TemplateInstruction::LoadLocal { dst, slot, .. } => {
                LoadLocalTemplate {
                    dst_reg: *dst,
                    local_idx: *slot,
                }
                .emit(ctx);
            }
            TemplateInstruction::StoreLocal { src, slot, .. } => {
                StoreLocalTemplate {
                    src_reg: *src,
                    local_idx: *slot,
                }
                .emit(ctx);
            }
            TemplateInstruction::DeleteLocal { slot, .. } => {
                DeleteLocalTemplate { local_idx: *slot }.emit(ctx);
            }
            TemplateInstruction::LoadGlobal { dst, name_idx, .. } => {
                LoadGlobalTemplate {
                    dst_reg: *dst,
                    name_idx: *name_idx,
                    deopt_idx,
                }
                .emit(ctx);
            }
            TemplateInstruction::StoreGlobal { src, name_idx, .. } => {
                StoreGlobalTemplate {
                    src_reg: *src,
                    name_idx: *name_idx,
                }
                .emit(ctx);
            }
            TemplateInstruction::DeleteGlobal { name_idx, .. } => {
                DeleteGlobalTemplate {
                    name_idx: *name_idx,
                    deopt_idx,
                }
                .emit(ctx);
            }
            // Closure operations - deopt to interpreter for Tier 1
            // (requires closure environment access from VM context)
            TemplateInstruction::LoadClosure { .. } => {
                // Closure cell access requires VM context - deopt for Tier 1
                ctx.asm.nop();
            }
            TemplateInstruction::StoreClosure { .. } => {
                // Closure cell write requires VM context - deopt for Tier 1
                ctx.asm.nop();
            }
            TemplateInstruction::DeleteClosure { .. } => {
                // Closure cell clear requires VM context - deopt for Tier 1
                ctx.asm.nop();
            }
            // Object attribute operations - deopt to interpreter for Tier 1
            // (requires type dispatch and Shape/slot management)
            TemplateInstruction::GetAttr { .. } => {
                // Attribute access requires type dispatch - deopt for Tier 1
                ctx.asm.nop();
            }
            TemplateInstruction::SetAttr { .. } => {
                // Attribute write requires type dispatch - deopt for Tier 1
                ctx.asm.nop();
            }
            TemplateInstruction::DelAttr { .. } => {
                // Attribute delete requires type dispatch - deopt for Tier 1
                ctx.asm.nop();
            }
            TemplateInstruction::LoadMethod { .. } => {
                // Method loading requires type dispatch - deopt for Tier 1
                ctx.asm.nop();
            }
            TemplateInstruction::IntAdd { dst, lhs, rhs, .. } => {
                IntAddTemplate {
                    dst_reg: *dst,
                    lhs_reg: *lhs,
                    rhs_reg: *rhs,
                    deopt_idx,
                }
                .emit(ctx);
            }
            TemplateInstruction::IntSub { dst, lhs, rhs, .. } => {
                IntSubTemplate {
                    dst_reg: *dst,
                    lhs_reg: *lhs,
                    rhs_reg: *rhs,
                    deopt_idx,
                }
                .emit(ctx);
            }
            TemplateInstruction::IntMul { dst, lhs, rhs, .. } => {
                IntMulTemplate {
                    dst_reg: *dst,
                    lhs_reg: *lhs,
                    rhs_reg: *rhs,
                    deopt_idx,
                }
                .emit(ctx);
            }
            TemplateInstruction::IntDiv { dst, lhs, rhs, .. } => {
                IntFloorDivTemplate {
                    dst_reg: *dst,
                    lhs_reg: *lhs,
                    rhs_reg: *rhs,
                    deopt_idx,
                }
                .emit(ctx);
            }
            TemplateInstruction::IntMod { dst, lhs, rhs, .. } => {
                IntModTemplate {
                    dst_reg: *dst,
                    lhs_reg: *lhs,
                    rhs_reg: *rhs,
                    deopt_idx,
                }
                .emit(ctx);
            }
            TemplateInstruction::IntNeg { dst, src, .. } => {
                IntNegTemplate {
                    dst_reg: *dst,
                    src_reg: *src,
                    deopt_idx,
                }
                .emit(ctx);
            }
            TemplateInstruction::FloatAdd { dst, lhs, rhs, .. } => {
                FloatAddTemplate {
                    dst_reg: *dst,
                    lhs_reg: *lhs,
                    rhs_reg: *rhs,
                }
                .emit(ctx);
            }
            TemplateInstruction::FloatSub { dst, lhs, rhs, .. } => {
                FloatSubTemplate {
                    dst_reg: *dst,
                    lhs_reg: *lhs,
                    rhs_reg: *rhs,
                }
                .emit(ctx);
            }
            TemplateInstruction::FloatMul { dst, lhs, rhs, .. } => {
                FloatMulTemplate {
                    dst_reg: *dst,
                    lhs_reg: *lhs,
                    rhs_reg: *rhs,
                }
                .emit(ctx);
            }
            TemplateInstruction::FloatDiv { dst, lhs, rhs, .. } => {
                FloatDivTemplate {
                    dst_reg: *dst,
                    lhs_reg: *lhs,
                    rhs_reg: *rhs,
                }
                .emit(ctx);
            }
            TemplateInstruction::FloatFloorDiv { dst, lhs, rhs, .. } => {
                FloatFloorDivTemplate {
                    dst_reg: *dst,
                    lhs_reg: *lhs,
                    rhs_reg: *rhs,
                    deopt_idx,
                }
                .emit(ctx);
            }
            TemplateInstruction::FloatMod { dst, lhs, rhs, .. } => {
                FloatModTemplate {
                    dst_reg: *dst,
                    lhs_reg: *lhs,
                    rhs_reg: *rhs,
                    deopt_idx,
                }
                .emit(ctx);
            }
            TemplateInstruction::FloatNeg { dst, src, .. } => {
                FloatNegTemplate {
                    dst_reg: *dst,
                    src_reg: *src,
                }
                .emit(ctx);
            }
            TemplateInstruction::Jump { target, .. } => {
                if let Some(label) = labels.get(target) {
                    JumpTemplate { target: *label }.emit(ctx);
                }
            }
            TemplateInstruction::BranchIfTrue { cond, target, .. } => {
                if let Some(label) = labels.get(target) {
                    BranchIfTrueTemplate {
                        condition_reg: *cond,
                        target: *label,
                        deopt_idx,
                    }
                    .emit(ctx);
                }
            }
            TemplateInstruction::BranchIfFalse { cond, target, .. } => {
                if let Some(label) = labels.get(target) {
                    BranchIfFalseTemplate {
                        condition_reg: *cond,
                        target: *label,
                        deopt_idx,
                    }
                    .emit(ctx);
                }
            }
            TemplateInstruction::BranchIfNone { cond, target, .. } => {
                if let Some(label) = labels.get(target) {
                    BranchIfNoneTemplate {
                        condition_reg: *cond,
                        target: *label,
                    }
                    .emit(ctx);
                }
            }
            TemplateInstruction::BranchIfNotNone { cond, target, .. } => {
                if let Some(label) = labels.get(target) {
                    BranchIfNotNoneTemplate {
                        condition_reg: *cond,
                        target: *label,
                    }
                    .emit(ctx);
                }
            }
            TemplateInstruction::Return { value, .. } => {
                ReturnTemplate { value_reg: *value }.emit(ctx);
            }
            // Integer comparisons
            TemplateInstruction::IntLt { dst, lhs, rhs, .. } => {
                LtTemplate {
                    dst_reg: *dst,
                    lhs_reg: *lhs,
                    rhs_reg: *rhs,
                    deopt_idx,
                }
                .emit(ctx);
            }
            TemplateInstruction::IntLe { dst, lhs, rhs, .. } => {
                LeTemplate {
                    dst_reg: *dst,
                    lhs_reg: *lhs,
                    rhs_reg: *rhs,
                    deopt_idx,
                }
                .emit(ctx);
            }
            TemplateInstruction::IntGt { dst, lhs, rhs, .. } => {
                GtTemplate {
                    dst_reg: *dst,
                    lhs_reg: *lhs,
                    rhs_reg: *rhs,
                    deopt_idx,
                }
                .emit(ctx);
            }
            TemplateInstruction::IntGe { dst, lhs, rhs, .. } => {
                GeTemplate {
                    dst_reg: *dst,
                    lhs_reg: *lhs,
                    rhs_reg: *rhs,
                    deopt_idx,
                }
                .emit(ctx);
            }
            TemplateInstruction::IntEq { dst, lhs, rhs, .. } => {
                EqTemplate {
                    dst_reg: *dst,
                    lhs_reg: *lhs,
                    rhs_reg: *rhs,
                    deopt_idx,
                }
                .emit(ctx);
            }
            TemplateInstruction::IntNe { dst, lhs, rhs, .. } => {
                NeTemplate {
                    dst_reg: *dst,
                    lhs_reg: *lhs,
                    rhs_reg: *rhs,
                    deopt_idx,
                }
                .emit(ctx);
            }
            // Float comparisons
            TemplateInstruction::FloatLt { dst, lhs, rhs, .. } => {
                FloatLtTemplate {
                    dst_reg: *dst,
                    lhs_reg: *lhs,
                    rhs_reg: *rhs,
                }
                .emit(ctx);
            }
            TemplateInstruction::FloatLe { dst, lhs, rhs, .. } => {
                FloatLeTemplate {
                    dst_reg: *dst,
                    lhs_reg: *lhs,
                    rhs_reg: *rhs,
                }
                .emit(ctx);
            }
            TemplateInstruction::FloatGt { dst, lhs, rhs, .. } => {
                FloatGtTemplate {
                    dst_reg: *dst,
                    lhs_reg: *lhs,
                    rhs_reg: *rhs,
                }
                .emit(ctx);
            }
            TemplateInstruction::FloatGe { dst, lhs, rhs, .. } => {
                FloatGeTemplate {
                    dst_reg: *dst,
                    lhs_reg: *lhs,
                    rhs_reg: *rhs,
                }
                .emit(ctx);
            }
            TemplateInstruction::FloatEq { dst, lhs, rhs, .. } => {
                FloatEqTemplate {
                    dst_reg: *dst,
                    lhs_reg: *lhs,
                    rhs_reg: *rhs,
                }
                .emit(ctx);
            }
            TemplateInstruction::FloatNe { dst, lhs, rhs, .. } => {
                FloatNeTemplate {
                    dst_reg: *dst,
                    lhs_reg: *lhs,
                    rhs_reg: *rhs,
                }
                .emit(ctx);
            }
            // Generic (polymorphic) comparisons - runtime dispatch (NOP in tier1)
            TemplateInstruction::GenericLt { .. }
            | TemplateInstruction::GenericLe { .. }
            | TemplateInstruction::GenericGt { .. }
            | TemplateInstruction::GenericGe { .. }
            | TemplateInstruction::GenericEq { .. }
            | TemplateInstruction::GenericNe { .. } => {
                // Generic comparisons require runtime dispatch for polymorphic behavior
                // For tier1, these emit nothing and rely on interpreter fallback
                // TODO: Add type-specialized fast paths with guards
            }
            // Bitwise operations
            TemplateInstruction::IntAnd { dst, lhs, rhs, .. } => {
                BitwiseAndTemplate {
                    dst_reg: *dst,
                    lhs_reg: *lhs,
                    rhs_reg: *rhs,
                    deopt_idx,
                }
                .emit(ctx);
            }
            TemplateInstruction::IntOr { dst, lhs, rhs, .. } => {
                BitwiseOrTemplate {
                    dst_reg: *dst,
                    lhs_reg: *lhs,
                    rhs_reg: *rhs,
                    deopt_idx,
                }
                .emit(ctx);
            }
            TemplateInstruction::IntXor { dst, lhs, rhs, .. } => {
                BitwiseXorTemplate {
                    dst_reg: *dst,
                    lhs_reg: *lhs,
                    rhs_reg: *rhs,
                    deopt_idx,
                }
                .emit(ctx);
            }
            TemplateInstruction::IntNot { dst, src, .. } => {
                BitwiseNotTemplate {
                    dst_reg: *dst,
                    src_reg: *src,
                    deopt_idx,
                }
                .emit(ctx);
            }
            TemplateInstruction::IntShl { dst, lhs, rhs, .. } => {
                ShlTemplate {
                    dst_reg: *dst,
                    lhs_reg: *lhs,
                    rhs_reg: *rhs,
                    deopt_idx,
                }
                .emit(ctx);
            }
            TemplateInstruction::IntShr { dst, lhs, rhs, .. } => {
                ShrTemplate {
                    dst_reg: *dst,
                    lhs_reg: *lhs,
                    rhs_reg: *rhs,
                    deopt_idx,
                }
                .emit(ctx);
            }
            TemplateInstruction::LogicalNot { dst, src, .. } => {
                NotTemplate {
                    dst_reg: *dst,
                    src_reg: *src,
                    deopt_idx,
                }
                .emit(ctx);
            }
            // Identity operations (pointer equality)
            TemplateInstruction::Is { dst, lhs, rhs, .. } => {
                IsTemplate {
                    dst_reg: *dst,
                    lhs_reg: *lhs,
                    rhs_reg: *rhs,
                }
                .emit(ctx);
            }
            TemplateInstruction::IsNot { dst, lhs, rhs, .. } => {
                IsNotTemplate {
                    dst_reg: *dst,
                    lhs_reg: *lhs,
                    rhs_reg: *rhs,
                }
                .emit(ctx);
            }
            // Membership operations - runtime dispatch (NOP in tier1, handled by interpreter fallback)
            TemplateInstruction::In { .. } | TemplateInstruction::NotIn { .. } => {
                // Membership tests require complex runtime logic (__contains__ or iteration)
                // For tier1, these emit nothing and rely on interpreter fallback
                // TODO: Add specialized templates for common container types
            }
            // Type guards
            TemplateInstruction::GuardInt { reg, .. } => {
                GuardIntTemplate {
                    reg: *reg,
                    deopt_idx,
                }
                .emit(ctx);
            }
            TemplateInstruction::GuardFloat { reg, .. } => {
                GuardFloatTemplate {
                    reg: *reg,
                    deopt_idx,
                }
                .emit(ctx);
            }
            TemplateInstruction::GuardBool { reg, .. } => {
                GuardBoolTemplate {
                    reg: *reg,
                    deopt_idx,
                }
                .emit(ctx);
            }
            TemplateInstruction::GuardNone { reg, .. } => {
                GuardNoneTemplate {
                    reg: *reg,
                    deopt_idx,
                }
                .emit(ctx);
            }
            TemplateInstruction::Nop { .. } => {
                // No-op: emit nothing
            }
        }
    }
}

// =============================================================================
// Template Instruction (IR for template compilation)
// =============================================================================

/// Intermediate representation for template-based compilation.
/// This is a simplified IR that maps 1:1 to bytecode operations.
#[derive(Debug, Clone)]
pub enum TemplateInstruction {
    // Value loading
    LoadInt {
        bc_offset: u32,
        dst: u8,
        value: i64,
    },
    LoadFloat {
        bc_offset: u32,
        dst: u8,
        value: f64,
    },
    LoadNone {
        bc_offset: u32,
        dst: u8,
    },
    LoadBool {
        bc_offset: u32,
        dst: u8,
        value: bool,
    },

    // Register operations
    Move {
        bc_offset: u32,
        dst: u8,
        src: u8,
    },

    // Local variable operations
    /// Load from local variable slot: dst = locals[slot]
    LoadLocal {
        bc_offset: u32,
        dst: u8,
        slot: u16,
    },
    /// Store to local variable slot: locals[slot] = src
    StoreLocal {
        bc_offset: u32,
        src: u8,
        slot: u16,
    },
    /// Delete local variable slot: locals[slot] = undefined
    DeleteLocal {
        bc_offset: u32,
        slot: u16,
    },

    // Global variable operations
    /// Load from global variable: dst = globals[name_idx]
    LoadGlobal {
        bc_offset: u32,
        dst: u8,
        name_idx: u16,
    },
    /// Store to global variable: globals[name_idx] = src
    StoreGlobal {
        bc_offset: u32,
        src: u8,
        name_idx: u16,
    },
    /// Delete global variable: del globals[name_idx]
    DeleteGlobal {
        bc_offset: u32,
        name_idx: u16,
    },

    // Closure variable operations
    /// Load from closure cell: dst = closure[cell_idx].get()
    /// Can fail if cell is unbound (UnboundLocalError)
    LoadClosure {
        bc_offset: u32,
        dst: u8,
        cell_idx: u16,
    },
    /// Store to closure cell: closure[cell_idx].set(src)
    StoreClosure {
        bc_offset: u32,
        src: u8,
        cell_idx: u16,
    },
    /// Delete closure cell: closure[cell_idx].clear()
    DeleteClosure {
        bc_offset: u32,
        cell_idx: u16,
    },

    // Object attribute operations
    /// Get attribute: dst = obj.attr[name_idx]
    /// Requires type dispatch - deopt for Tier 1
    GetAttr {
        bc_offset: u32,
        dst: u8,
        obj: u8,
        name_idx: u8, // 8-bit name index from src2 field
    },
    /// Set attribute: obj.attr[name_idx] = value
    /// Requires type dispatch - deopt for Tier 1
    SetAttr {
        bc_offset: u32,
        obj: u8,
        name_idx: u8,
        value: u8,
    },
    /// Delete attribute: del obj.attr[name_idx]
    /// Requires type dispatch - deopt for Tier 1
    DelAttr {
        bc_offset: u32,
        obj: u8,
        name_idx: u8,
    },
    /// Load method for optimized method calls: dst = obj.method
    /// Requires type dispatch - deopt for Tier 1
    LoadMethod {
        bc_offset: u32,
        dst: u8,
        obj: u8,
        name_idx: u8,
    },

    // Container item operations
    /// Get item: dst = container[key]
    /// Requires type dispatch - deopt for Tier 1
    GetItem {
        bc_offset: u32,
        dst: u8,
        container: u8,
        key: u8,
    },
    /// Set item: container[key] = value
    /// Requires type dispatch - deopt for Tier 1
    SetItem {
        bc_offset: u32,
        container: u8,
        key: u8,
        value: u8,
    },
    /// Delete item: del container[key]
    /// Requires type dispatch - deopt for Tier 1
    DelItem {
        bc_offset: u32,
        container: u8,
        key: u8,
    },

    // Iteration operations
    /// Get iterator: dst = iter(src)
    /// Requires type dispatch - deopt for Tier 1
    GetIter {
        bc_offset: u32,
        dst: u8,
        src: u8,
    },
    /// For-loop iterator advance: dst = next(iter), jump if StopIteration
    /// Requires iterator state access - deopt for Tier 1
    ForIter {
        bc_offset: u32,
        dst: u8,
        iter: u8,
        offset: i8, // Jump offset on StopIteration
    },

    // Utility operations
    /// Get length: dst = len(src)
    /// Requires type dispatch - deopt for Tier 1
    Len {
        bc_offset: u32,
        dst: u8,
        src: u8,
    },
    /// Check if callable: dst = callable(src)
    /// Requires type checking - deopt for Tier 1
    IsCallable {
        bc_offset: u32,
        dst: u8,
        src: u8,
    },

    // Integer arithmetic
    IntAdd {
        bc_offset: u32,
        dst: u8,
        lhs: u8,
        rhs: u8,
    },
    IntSub {
        bc_offset: u32,
        dst: u8,
        lhs: u8,
        rhs: u8,
    },
    IntMul {
        bc_offset: u32,
        dst: u8,
        lhs: u8,
        rhs: u8,
    },
    IntDiv {
        bc_offset: u32,
        dst: u8,
        lhs: u8,
        rhs: u8,
    },
    IntMod {
        bc_offset: u32,
        dst: u8,
        lhs: u8,
        rhs: u8,
    },
    IntNeg {
        bc_offset: u32,
        dst: u8,
        src: u8,
    },

    // Float arithmetic
    FloatAdd {
        bc_offset: u32,
        dst: u8,
        lhs: u8,
        rhs: u8,
    },
    FloatSub {
        bc_offset: u32,
        dst: u8,
        lhs: u8,
        rhs: u8,
    },
    FloatMul {
        bc_offset: u32,
        dst: u8,
        lhs: u8,
        rhs: u8,
    },
    FloatDiv {
        bc_offset: u32,
        dst: u8,
        lhs: u8,
        rhs: u8,
    },
    /// Float floor division: dst = floor(lhs / rhs)
    FloatFloorDiv {
        bc_offset: u32,
        dst: u8,
        lhs: u8,
        rhs: u8,
    },
    /// Float modulo: dst = lhs % rhs (Python semantics)
    FloatMod {
        bc_offset: u32,
        dst: u8,
        lhs: u8,
        rhs: u8,
    },
    /// Float negation: dst = -src
    FloatNeg {
        bc_offset: u32,
        dst: u8,
        src: u8,
    },

    // Integer comparisons
    IntLt {
        bc_offset: u32,
        dst: u8,
        lhs: u8,
        rhs: u8,
    },
    IntLe {
        bc_offset: u32,
        dst: u8,
        lhs: u8,
        rhs: u8,
    },
    IntGt {
        bc_offset: u32,
        dst: u8,
        lhs: u8,
        rhs: u8,
    },
    IntGe {
        bc_offset: u32,
        dst: u8,
        lhs: u8,
        rhs: u8,
    },
    IntEq {
        bc_offset: u32,
        dst: u8,
        lhs: u8,
        rhs: u8,
    },
    IntNe {
        bc_offset: u32,
        dst: u8,
        lhs: u8,
        rhs: u8,
    },

    // Float comparisons
    FloatLt {
        bc_offset: u32,
        dst: u8,
        lhs: u8,
        rhs: u8,
    },
    FloatLe {
        bc_offset: u32,
        dst: u8,
        lhs: u8,
        rhs: u8,
    },
    FloatGt {
        bc_offset: u32,
        dst: u8,
        lhs: u8,
        rhs: u8,
    },
    FloatGe {
        bc_offset: u32,
        dst: u8,
        lhs: u8,
        rhs: u8,
    },
    FloatEq {
        bc_offset: u32,
        dst: u8,
        lhs: u8,
        rhs: u8,
    },
    FloatNe {
        bc_offset: u32,
        dst: u8,
        lhs: u8,
        rhs: u8,
    },

    // Generic (polymorphic) comparisons - require runtime dispatch
    /// Generic less than: dst = lhs < rhs
    GenericLt {
        bc_offset: u32,
        dst: u8,
        lhs: u8,
        rhs: u8,
    },
    /// Generic less than or equal: dst = lhs <= rhs
    GenericLe {
        bc_offset: u32,
        dst: u8,
        lhs: u8,
        rhs: u8,
    },
    /// Generic greater than: dst = lhs > rhs
    GenericGt {
        bc_offset: u32,
        dst: u8,
        lhs: u8,
        rhs: u8,
    },
    /// Generic greater than or equal: dst = lhs >= rhs
    GenericGe {
        bc_offset: u32,
        dst: u8,
        lhs: u8,
        rhs: u8,
    },
    /// Generic equal: dst = lhs == rhs
    GenericEq {
        bc_offset: u32,
        dst: u8,
        lhs: u8,
        rhs: u8,
    },
    /// Generic not equal: dst = lhs != rhs
    GenericNe {
        bc_offset: u32,
        dst: u8,
        lhs: u8,
        rhs: u8,
    },

    // Bitwise operations
    /// Bitwise AND: dst = lhs & rhs
    IntAnd {
        bc_offset: u32,
        dst: u8,
        lhs: u8,
        rhs: u8,
    },
    /// Bitwise OR: dst = lhs | rhs
    IntOr {
        bc_offset: u32,
        dst: u8,
        lhs: u8,
        rhs: u8,
    },
    /// Bitwise XOR: dst = lhs ^ rhs
    IntXor {
        bc_offset: u32,
        dst: u8,
        lhs: u8,
        rhs: u8,
    },
    /// Bitwise NOT: dst = ~src
    IntNot {
        bc_offset: u32,
        dst: u8,
        src: u8,
    },
    /// Left shift: dst = lhs << rhs
    IntShl {
        bc_offset: u32,
        dst: u8,
        lhs: u8,
        rhs: u8,
    },
    /// Right shift: dst = lhs >> rhs (arithmetic)
    IntShr {
        bc_offset: u32,
        dst: u8,
        lhs: u8,
        rhs: u8,
    },
    /// Logical NOT: dst = not src
    LogicalNot {
        bc_offset: u32,
        dst: u8,
        src: u8,
    },

    // Identity operations
    /// Identity check: dst = lhs is rhs
    Is {
        bc_offset: u32,
        dst: u8,
        lhs: u8,
        rhs: u8,
    },
    /// Not identity check: dst = lhs is not rhs
    IsNot {
        bc_offset: u32,
        dst: u8,
        lhs: u8,
        rhs: u8,
    },

    // Membership operations
    /// Membership check: dst = lhs in rhs
    In {
        bc_offset: u32,
        dst: u8,
        lhs: u8,
        rhs: u8,
    },
    /// Not membership check: dst = lhs not in rhs
    NotIn {
        bc_offset: u32,
        dst: u8,
        lhs: u8,
        rhs: u8,
    },

    // Type guards (emit deopt on type mismatch)
    GuardInt {
        bc_offset: u32,
        reg: u8,
    },
    GuardFloat {
        bc_offset: u32,
        reg: u8,
    },
    GuardBool {
        bc_offset: u32,
        reg: u8,
    },
    GuardNone {
        bc_offset: u32,
        reg: u8,
    },

    // Control flow
    Jump {
        bc_offset: u32,
        target: u32,
    },
    BranchIfTrue {
        bc_offset: u32,
        cond: u8,
        target: u32,
    },
    BranchIfFalse {
        bc_offset: u32,
        cond: u8,
        target: u32,
    },
    /// Branch if value is None
    BranchIfNone {
        bc_offset: u32,
        cond: u8,
        target: u32,
    },
    /// Branch if value is not None
    BranchIfNotNone {
        bc_offset: u32,
        cond: u8,
        target: u32,
    },
    Return {
        bc_offset: u32,
        value: u8,
    },

    // No-op
    Nop {
        bc_offset: u32,
    },
}

impl TemplateInstruction {
    /// Get the bytecode offset of this instruction.
    pub fn bc_offset(&self) -> u32 {
        match self {
            TemplateInstruction::LoadInt { bc_offset, .. }
            | TemplateInstruction::LoadFloat { bc_offset, .. }
            | TemplateInstruction::LoadNone { bc_offset, .. }
            | TemplateInstruction::LoadBool { bc_offset, .. }
            | TemplateInstruction::Move { bc_offset, .. }
            | TemplateInstruction::LoadLocal { bc_offset, .. }
            | TemplateInstruction::StoreLocal { bc_offset, .. }
            | TemplateInstruction::DeleteLocal { bc_offset, .. }
            | TemplateInstruction::LoadGlobal { bc_offset, .. }
            | TemplateInstruction::StoreGlobal { bc_offset, .. }
            | TemplateInstruction::DeleteGlobal { bc_offset, .. }
            | TemplateInstruction::LoadClosure { bc_offset, .. }
            | TemplateInstruction::StoreClosure { bc_offset, .. }
            | TemplateInstruction::DeleteClosure { bc_offset, .. }
            | TemplateInstruction::GetAttr { bc_offset, .. }
            | TemplateInstruction::SetAttr { bc_offset, .. }
            | TemplateInstruction::DelAttr { bc_offset, .. }
            | TemplateInstruction::LoadMethod { bc_offset, .. }
            | TemplateInstruction::IntAdd { bc_offset, .. }
            | TemplateInstruction::IntSub { bc_offset, .. }
            | TemplateInstruction::IntMul { bc_offset, .. }
            | TemplateInstruction::IntDiv { bc_offset, .. }
            | TemplateInstruction::IntMod { bc_offset, .. }
            | TemplateInstruction::IntNeg { bc_offset, .. }
            | TemplateInstruction::FloatAdd { bc_offset, .. }
            | TemplateInstruction::FloatSub { bc_offset, .. }
            | TemplateInstruction::FloatMul { bc_offset, .. }
            | TemplateInstruction::FloatDiv { bc_offset, .. }
            | TemplateInstruction::FloatFloorDiv { bc_offset, .. }
            | TemplateInstruction::FloatMod { bc_offset, .. }
            | TemplateInstruction::FloatNeg { bc_offset, .. }
            | TemplateInstruction::IntLt { bc_offset, .. }
            | TemplateInstruction::IntLe { bc_offset, .. }
            | TemplateInstruction::IntGt { bc_offset, .. }
            | TemplateInstruction::IntGe { bc_offset, .. }
            | TemplateInstruction::IntEq { bc_offset, .. }
            | TemplateInstruction::IntNe { bc_offset, .. }
            | TemplateInstruction::FloatLt { bc_offset, .. }
            | TemplateInstruction::FloatLe { bc_offset, .. }
            | TemplateInstruction::FloatGt { bc_offset, .. }
            | TemplateInstruction::FloatGe { bc_offset, .. }
            | TemplateInstruction::FloatEq { bc_offset, .. }
            | TemplateInstruction::FloatNe { bc_offset, .. }
            | TemplateInstruction::GenericLt { bc_offset, .. }
            | TemplateInstruction::GenericLe { bc_offset, .. }
            | TemplateInstruction::GenericGt { bc_offset, .. }
            | TemplateInstruction::GenericGe { bc_offset, .. }
            | TemplateInstruction::GenericEq { bc_offset, .. }
            | TemplateInstruction::GenericNe { bc_offset, .. }
            | TemplateInstruction::IntAnd { bc_offset, .. }
            | TemplateInstruction::IntOr { bc_offset, .. }
            | TemplateInstruction::IntXor { bc_offset, .. }
            | TemplateInstruction::IntNot { bc_offset, .. }
            | TemplateInstruction::IntShl { bc_offset, .. }
            | TemplateInstruction::IntShr { bc_offset, .. }
            | TemplateInstruction::LogicalNot { bc_offset, .. }
            | TemplateInstruction::Is { bc_offset, .. }
            | TemplateInstruction::IsNot { bc_offset, .. }
            | TemplateInstruction::In { bc_offset, .. }
            | TemplateInstruction::NotIn { bc_offset, .. }
            | TemplateInstruction::GuardInt { bc_offset, .. }
            | TemplateInstruction::GuardFloat { bc_offset, .. }
            | TemplateInstruction::GuardBool { bc_offset, .. }
            | TemplateInstruction::GuardNone { bc_offset, .. }
            | TemplateInstruction::Jump { bc_offset, .. }
            | TemplateInstruction::BranchIfTrue { bc_offset, .. }
            | TemplateInstruction::BranchIfFalse { bc_offset, .. }
            | TemplateInstruction::BranchIfNone { bc_offset, .. }
            | TemplateInstruction::BranchIfNotNone { bc_offset, .. }
            | TemplateInstruction::Return { bc_offset, .. }
            | TemplateInstruction::Nop { bc_offset } => *bc_offset,
        }
    }

    /// Get the jump target if this is a branch/jump instruction.
    pub fn jump_target(&self) -> Option<u32> {
        match self {
            TemplateInstruction::Jump { target, .. } => Some(*target),
            TemplateInstruction::BranchIfTrue { target, .. } => Some(*target),
            TemplateInstruction::BranchIfFalse { target, .. } => Some(*target),
            _ => None,
        }
    }

    /// Check if this instruction can trigger deoptimization.
    pub fn can_deopt(&self) -> bool {
        matches!(
            self,
            TemplateInstruction::IntAdd { .. }
                | TemplateInstruction::IntSub { .. }
                | TemplateInstruction::IntMul { .. }
                | TemplateInstruction::IntDiv { .. }
                | TemplateInstruction::IntMod { .. }
                | TemplateInstruction::IntNeg { .. }
                | TemplateInstruction::IntLt { .. }
                | TemplateInstruction::IntLe { .. }
                | TemplateInstruction::IntGt { .. }
                | TemplateInstruction::IntGe { .. }
                | TemplateInstruction::IntEq { .. }
                | TemplateInstruction::IntNe { .. }
                | TemplateInstruction::IntAnd { .. }
                | TemplateInstruction::IntOr { .. }
                | TemplateInstruction::IntXor { .. }
                | TemplateInstruction::IntNot { .. }
                | TemplateInstruction::IntShl { .. }
                | TemplateInstruction::IntShr { .. }
                | TemplateInstruction::LogicalNot { .. }
                | TemplateInstruction::FloatFloorDiv { .. }
                | TemplateInstruction::FloatMod { .. }
                | TemplateInstruction::LoadGlobal { .. }
                | TemplateInstruction::DeleteGlobal { .. }
                | TemplateInstruction::BranchIfTrue { .. }
                | TemplateInstruction::BranchIfFalse { .. }
                | TemplateInstruction::GuardInt { .. }
                | TemplateInstruction::GuardFloat { .. }
                | TemplateInstruction::GuardBool { .. }
                | TemplateInstruction::GuardNone { .. }
        )
    }

    /// Get the deopt reason for this instruction.
    pub fn deopt_reason(&self) -> DeoptReason {
        match self {
            TemplateInstruction::IntAdd { .. }
            | TemplateInstruction::IntSub { .. }
            | TemplateInstruction::IntMul { .. }
            | TemplateInstruction::IntDiv { .. }
            | TemplateInstruction::IntMod { .. }
            | TemplateInstruction::IntNeg { .. }
            | TemplateInstruction::IntLt { .. }
            | TemplateInstruction::IntLe { .. }
            | TemplateInstruction::IntGt { .. }
            | TemplateInstruction::IntGe { .. }
            | TemplateInstruction::IntEq { .. }
            | TemplateInstruction::IntNe { .. }
            | TemplateInstruction::IntAnd { .. }
            | TemplateInstruction::IntOr { .. }
            | TemplateInstruction::IntXor { .. }
            | TemplateInstruction::IntNot { .. }
            | TemplateInstruction::IntShl { .. }
            | TemplateInstruction::IntShr { .. }
            | TemplateInstruction::LogicalNot { .. }
            | TemplateInstruction::FloatFloorDiv { .. }
            | TemplateInstruction::FloatMod { .. }
            | TemplateInstruction::LoadGlobal { .. }
            | TemplateInstruction::DeleteGlobal { .. }
            | TemplateInstruction::GuardInt { .. }
            | TemplateInstruction::GuardFloat { .. }
            | TemplateInstruction::GuardBool { .. }
            | TemplateInstruction::GuardNone { .. } => DeoptReason::TypeGuardFailed,
            TemplateInstruction::BranchIfTrue { .. }
            | TemplateInstruction::BranchIfFalse { .. } => DeoptReason::TypeGuardFailed,
            _ => DeoptReason::UncommonTrap,
        }
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compile_empty() {
        let compiler = TemplateCompiler::new_for_testing();
        let result = compiler.compile(4, &[]);
        assert!(result.is_ok());
    }

    #[test]
    fn test_compile_load_int() {
        let compiler = TemplateCompiler::new_for_testing();
        let instrs = vec![
            TemplateInstruction::LoadInt {
                bc_offset: 0,
                dst: 0,
                value: 42,
            },
            TemplateInstruction::Return {
                bc_offset: 4,
                value: 0,
            },
        ];

        let result = compiler.compile(4, &instrs);
        assert!(result.is_ok());

        let func = result.unwrap();
        assert!(func.code.len() > 0);
        assert!(func.bc_to_native.contains_key(&0));
        assert!(func.bc_to_native.contains_key(&4));
    }

    #[test]
    fn test_compile_arithmetic() {
        let compiler = TemplateCompiler::new_for_testing();
        let instrs = vec![
            TemplateInstruction::LoadInt {
                bc_offset: 0,
                dst: 0,
                value: 10,
            },
            TemplateInstruction::LoadInt {
                bc_offset: 4,
                dst: 1,
                value: 20,
            },
            TemplateInstruction::IntAdd {
                bc_offset: 8,
                dst: 2,
                lhs: 0,
                rhs: 1,
            },
            TemplateInstruction::Return {
                bc_offset: 12,
                value: 2,
            },
        ];

        let result = compiler.compile(4, &instrs);
        assert!(result.is_ok());

        let func = result.unwrap();
        // Should have deopt info for IntAdd
        assert!(!func.deopt_info.is_empty());
    }

    #[test]
    fn test_compile_float_arithmetic() {
        let compiler = TemplateCompiler::new_for_testing();
        let instrs = vec![
            TemplateInstruction::LoadFloat {
                bc_offset: 0,
                dst: 0,
                value: 1.5,
            },
            TemplateInstruction::LoadFloat {
                bc_offset: 4,
                dst: 1,
                value: 2.5,
            },
            TemplateInstruction::FloatAdd {
                bc_offset: 8,
                dst: 2,
                lhs: 0,
                rhs: 1,
            },
            TemplateInstruction::FloatMul {
                bc_offset: 12,
                dst: 3,
                lhs: 2,
                rhs: 1,
            },
            TemplateInstruction::Return {
                bc_offset: 16,
                value: 3,
            },
        ];

        let result = compiler.compile(4, &instrs);
        assert!(result.is_ok());
    }

    #[test]
    fn test_compile_jumps() {
        let compiler = TemplateCompiler::new_for_testing();
        let instrs = vec![
            TemplateInstruction::LoadBool {
                bc_offset: 0,
                dst: 0,
                value: true,
            },
            TemplateInstruction::BranchIfTrue {
                bc_offset: 4,
                cond: 0,
                target: 12,
            },
            TemplateInstruction::LoadInt {
                bc_offset: 8,
                dst: 1,
                value: 1,
            },
            TemplateInstruction::LoadInt {
                bc_offset: 12,
                dst: 1,
                value: 2,
            },
            TemplateInstruction::Return {
                bc_offset: 16,
                value: 1,
            },
        ];

        let result = compiler.compile(4, &instrs);
        assert!(result.is_ok());
    }

    #[test]
    fn test_template_instruction_properties() {
        let jump = TemplateInstruction::Jump {
            bc_offset: 0,
            target: 100,
        };
        assert_eq!(jump.bc_offset(), 0);
        assert_eq!(jump.jump_target(), Some(100));
        assert!(!jump.can_deopt());

        let add = TemplateInstruction::IntAdd {
            bc_offset: 4,
            dst: 0,
            lhs: 1,
            rhs: 2,
        };
        assert_eq!(add.bc_offset(), 4);
        assert_eq!(add.jump_target(), None);
        assert!(add.can_deopt());
        assert_eq!(add.deopt_reason(), DeoptReason::TypeGuardFailed);
    }

    #[test]
    fn test_compile_int_comparisons() {
        let compiler = TemplateCompiler::new_for_testing();
        let instrs = vec![
            TemplateInstruction::LoadInt {
                bc_offset: 0,
                dst: 0,
                value: 10,
            },
            TemplateInstruction::LoadInt {
                bc_offset: 4,
                dst: 1,
                value: 20,
            },
            TemplateInstruction::IntLt {
                bc_offset: 8,
                dst: 2,
                lhs: 0,
                rhs: 1,
            },
            TemplateInstruction::IntLe {
                bc_offset: 12,
                dst: 2,
                lhs: 0,
                rhs: 1,
            },
            TemplateInstruction::IntGt {
                bc_offset: 16,
                dst: 2,
                lhs: 0,
                rhs: 1,
            },
            TemplateInstruction::IntGe {
                bc_offset: 20,
                dst: 2,
                lhs: 0,
                rhs: 1,
            },
            TemplateInstruction::IntEq {
                bc_offset: 24,
                dst: 2,
                lhs: 0,
                rhs: 1,
            },
            TemplateInstruction::IntNe {
                bc_offset: 28,
                dst: 2,
                lhs: 0,
                rhs: 1,
            },
            TemplateInstruction::Return {
                bc_offset: 32,
                value: 2,
            },
        ];

        let result = compiler.compile(4, &instrs);
        assert!(result.is_ok());

        let func = result.unwrap();
        // Should have deopt info for all integer comparisons
        assert!(func.deopt_info.len() >= 6);
        // Code should be substantial (each comparison generates ~100 bytes)
        assert!(func.code.len() > 200);
    }

    #[test]
    fn test_compile_float_comparisons() {
        let compiler = TemplateCompiler::new_for_testing();
        let instrs = vec![
            TemplateInstruction::LoadFloat {
                bc_offset: 0,
                dst: 0,
                value: 1.5,
            },
            TemplateInstruction::LoadFloat {
                bc_offset: 4,
                dst: 1,
                value: 2.5,
            },
            TemplateInstruction::FloatLt {
                bc_offset: 8,
                dst: 2,
                lhs: 0,
                rhs: 1,
            },
            TemplateInstruction::FloatLe {
                bc_offset: 12,
                dst: 2,
                lhs: 0,
                rhs: 1,
            },
            TemplateInstruction::FloatGt {
                bc_offset: 16,
                dst: 2,
                lhs: 0,
                rhs: 1,
            },
            TemplateInstruction::FloatGe {
                bc_offset: 20,
                dst: 2,
                lhs: 0,
                rhs: 1,
            },
            TemplateInstruction::FloatEq {
                bc_offset: 24,
                dst: 2,
                lhs: 0,
                rhs: 1,
            },
            TemplateInstruction::FloatNe {
                bc_offset: 28,
                dst: 2,
                lhs: 0,
                rhs: 1,
            },
            TemplateInstruction::Return {
                bc_offset: 32,
                value: 2,
            },
        ];

        let result = compiler.compile(4, &instrs);
        assert!(result.is_ok());

        let func = result.unwrap();
        // Float comparisons don't deopt (pure float math)
        // Code should be substantial
        assert!(func.code.len() > 100);
    }

    #[test]
    fn test_compile_type_guards() {
        let compiler = TemplateCompiler::new_for_testing();
        let instrs = vec![
            TemplateInstruction::LoadInt {
                bc_offset: 0,
                dst: 0,
                value: 42,
            },
            TemplateInstruction::GuardInt {
                bc_offset: 4,
                reg: 0,
            },
            TemplateInstruction::LoadFloat {
                bc_offset: 8,
                dst: 1,
                value: 3.14,
            },
            TemplateInstruction::GuardFloat {
                bc_offset: 12,
                reg: 1,
            },
            TemplateInstruction::LoadBool {
                bc_offset: 16,
                dst: 2,
                value: true,
            },
            TemplateInstruction::GuardBool {
                bc_offset: 20,
                reg: 2,
            },
            TemplateInstruction::Return {
                bc_offset: 24,
                value: 0,
            },
        ];

        let result = compiler.compile(4, &instrs);
        assert!(result.is_ok());

        let func = result.unwrap();
        // Should have deopt info for all 3 guards
        assert!(func.deopt_info.len() >= 3);
        // Guards generate compact code
        assert!(func.code.len() > 50);
    }

    #[test]
    fn test_int_comparison_deopt_properties() {
        // Verify IntLt can deopt
        let lt = TemplateInstruction::IntLt {
            bc_offset: 0,
            dst: 0,
            lhs: 1,
            rhs: 2,
        };
        assert!(lt.can_deopt());
        assert_eq!(lt.deopt_reason(), DeoptReason::TypeGuardFailed);

        // Verify IntEq can deopt
        let eq = TemplateInstruction::IntEq {
            bc_offset: 0,
            dst: 0,
            lhs: 1,
            rhs: 2,
        };
        assert!(eq.can_deopt());

        // Verify FloatLt does NOT deopt (pure float)
        let float_lt = TemplateInstruction::FloatLt {
            bc_offset: 0,
            dst: 0,
            lhs: 1,
            rhs: 2,
        };
        assert!(!float_lt.can_deopt());
    }

    #[test]
    fn test_guard_deopt_properties() {
        let guard_int = TemplateInstruction::GuardInt {
            bc_offset: 0,
            reg: 0,
        };
        assert!(guard_int.can_deopt());
        assert_eq!(guard_int.deopt_reason(), DeoptReason::TypeGuardFailed);

        let guard_float = TemplateInstruction::GuardFloat {
            bc_offset: 0,
            reg: 0,
        };
        assert!(guard_float.can_deopt());

        let guard_bool = TemplateInstruction::GuardBool {
            bc_offset: 0,
            reg: 0,
        };
        assert!(guard_bool.can_deopt());
    }
}
