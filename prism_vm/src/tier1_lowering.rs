//! Shared Tier 1 lowering utilities for VM-side JIT entry points.
//!
//! This module centralizes bytecode -> template lowering so sync and async
//! compilation paths use identical semantics.

use prism_compiler::bytecode::{CodeObject, Opcode};
use prism_jit::tier1::codegen::TemplateInstruction;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum KnownType {
    Unknown,
    None,
    Bool,
    Int,
    Float,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum NumericKind {
    Int,
    Float,
}

#[inline]
fn get_reg_type(reg_types: &[KnownType], reg: u8) -> KnownType {
    reg_types
        .get(reg as usize)
        .copied()
        .unwrap_or(KnownType::Unknown)
}

#[inline]
fn set_reg_type(reg_types: &mut [KnownType], reg: u8, ty: KnownType) {
    if let Some(slot) = reg_types.get_mut(reg as usize) {
        *slot = ty;
    }
}

fn infer_numeric_kind(
    op: Opcode,
    bc_offset: u32,
    lhs: u8,
    rhs: u8,
    reg_types: &[KnownType],
) -> Result<NumericKind, String> {
    let lhs_ty = get_reg_type(reg_types, lhs);
    let rhs_ty = get_reg_type(reg_types, rhs);
    match (lhs_ty, rhs_ty) {
        (KnownType::Int, KnownType::Int) => Ok(NumericKind::Int),
        (KnownType::Float, KnownType::Float) => Ok(NumericKind::Float),
        _ => Err(format!(
            "tier1 lowering requires monomorphic numeric operands for {:?} at bytecode offset {} (lhs r{}={:?}, rhs r{}={:?})",
            op, bc_offset, lhs, lhs_ty, rhs, rhs_ty
        )),
    }
}

/// Lower a code object to template IR for Tier 1 compilation.
///
/// This performs strict validation to prevent silent miscompilation and only
/// accepts a known-safe Tier 1 subset:
/// - every supported bytecode lowers to one concrete template
/// - unsupported bytecodes return an error (caller falls back to interpreter)
/// - every jump target must resolve to a valid bytecode offset
pub(crate) fn lower_code_to_templates(
    code: &CodeObject,
) -> Result<Vec<TemplateInstruction>, String> {
    let mut templates = Vec::with_capacity(code.instructions.len());
    let mut reg_types = vec![KnownType::Unknown; code.register_count as usize];

    for (index, inst) in code.instructions.iter().enumerate() {
        let bc_offset = (index as u32) * 4;
        let op = Opcode::from_u8(inst.opcode())
            .ok_or_else(|| format!("invalid opcode byte {} at index {}", inst.opcode(), index))?;

        let template = match op {
            Opcode::Nop => TemplateInstruction::Nop { bc_offset },

            Opcode::LoadConst => {
                let dst = inst.dst().0;
                let (template, ty) = load_const_template(code, bc_offset, dst, inst.imm16(), index)?;
                set_reg_type(&mut reg_types, dst, ty);
                template
            }
            Opcode::LoadNone => {
                let dst = inst.dst().0;
                set_reg_type(&mut reg_types, dst, KnownType::None);
                TemplateInstruction::LoadNone { bc_offset, dst }
            }
            Opcode::LoadTrue => {
                let dst = inst.dst().0;
                set_reg_type(&mut reg_types, dst, KnownType::Bool);
                TemplateInstruction::LoadBool {
                    bc_offset,
                    dst,
                    value: true,
                }
            }
            Opcode::LoadFalse => {
                let dst = inst.dst().0;
                set_reg_type(&mut reg_types, dst, KnownType::Bool);
                TemplateInstruction::LoadBool {
                    bc_offset,
                    dst,
                    value: false,
                }
            }
            Opcode::Move => {
                let dst = inst.dst().0;
                let src = inst.src1().0;
                let src_ty = get_reg_type(&reg_types, src);
                set_reg_type(&mut reg_types, dst, src_ty);
                TemplateInstruction::Move {
                    bc_offset,
                    dst,
                    src,
                }
            }

            Opcode::AddInt => {
                let dst = inst.dst().0;
                set_reg_type(&mut reg_types, dst, KnownType::Int);
                TemplateInstruction::IntAdd {
                    bc_offset,
                    dst,
                    lhs: inst.src1().0,
                    rhs: inst.src2().0,
                }
            }
            Opcode::Add => {
                let dst = inst.dst().0;
                let lhs = inst.src1().0;
                let rhs = inst.src2().0;
                match infer_numeric_kind(op, bc_offset, lhs, rhs, &reg_types)? {
                    NumericKind::Int => {
                        set_reg_type(&mut reg_types, dst, KnownType::Int);
                        TemplateInstruction::IntAdd {
                            bc_offset,
                            dst,
                            lhs,
                            rhs,
                        }
                    }
                    NumericKind::Float => {
                        set_reg_type(&mut reg_types, dst, KnownType::Float);
                        TemplateInstruction::FloatAdd {
                            bc_offset,
                            dst,
                            lhs,
                            rhs,
                        }
                    }
                }
            }
            Opcode::SubInt => {
                let dst = inst.dst().0;
                set_reg_type(&mut reg_types, dst, KnownType::Int);
                TemplateInstruction::IntSub {
                    bc_offset,
                    dst,
                    lhs: inst.src1().0,
                    rhs: inst.src2().0,
                }
            }
            Opcode::Sub => {
                let dst = inst.dst().0;
                let lhs = inst.src1().0;
                let rhs = inst.src2().0;
                match infer_numeric_kind(op, bc_offset, lhs, rhs, &reg_types)? {
                    NumericKind::Int => {
                        set_reg_type(&mut reg_types, dst, KnownType::Int);
                        TemplateInstruction::IntSub {
                            bc_offset,
                            dst,
                            lhs,
                            rhs,
                        }
                    }
                    NumericKind::Float => {
                        set_reg_type(&mut reg_types, dst, KnownType::Float);
                        TemplateInstruction::FloatSub {
                            bc_offset,
                            dst,
                            lhs,
                            rhs,
                        }
                    }
                }
            }
            Opcode::MulInt => {
                let dst = inst.dst().0;
                set_reg_type(&mut reg_types, dst, KnownType::Int);
                TemplateInstruction::IntMul {
                    bc_offset,
                    dst,
                    lhs: inst.src1().0,
                    rhs: inst.src2().0,
                }
            }
            Opcode::Mul => {
                let dst = inst.dst().0;
                let lhs = inst.src1().0;
                let rhs = inst.src2().0;
                match infer_numeric_kind(op, bc_offset, lhs, rhs, &reg_types)? {
                    NumericKind::Int => {
                        set_reg_type(&mut reg_types, dst, KnownType::Int);
                        TemplateInstruction::IntMul {
                            bc_offset,
                            dst,
                            lhs,
                            rhs,
                        }
                    }
                    NumericKind::Float => {
                        set_reg_type(&mut reg_types, dst, KnownType::Float);
                        TemplateInstruction::FloatMul {
                            bc_offset,
                            dst,
                            lhs,
                            rhs,
                        }
                    }
                }
            }
            Opcode::FloorDivInt => {
                let dst = inst.dst().0;
                set_reg_type(&mut reg_types, dst, KnownType::Int);
                TemplateInstruction::IntDiv {
                    bc_offset,
                    dst,
                    lhs: inst.src1().0,
                    rhs: inst.src2().0,
                }
            }
            Opcode::FloorDivFloat => {
                let dst = inst.dst().0;
                set_reg_type(&mut reg_types, dst, KnownType::Float);
                TemplateInstruction::FloatFloorDiv {
                    bc_offset,
                    dst,
                    lhs: inst.src1().0,
                    rhs: inst.src2().0,
                }
            }
            Opcode::FloorDiv => {
                let dst = inst.dst().0;
                let lhs = inst.src1().0;
                let rhs = inst.src2().0;
                match infer_numeric_kind(op, bc_offset, lhs, rhs, &reg_types)? {
                    NumericKind::Int => {
                        set_reg_type(&mut reg_types, dst, KnownType::Int);
                        TemplateInstruction::IntDiv {
                            bc_offset,
                            dst,
                            lhs,
                            rhs,
                        }
                    }
                    NumericKind::Float => {
                        set_reg_type(&mut reg_types, dst, KnownType::Float);
                        TemplateInstruction::FloatFloorDiv {
                            bc_offset,
                            dst,
                            lhs,
                            rhs,
                        }
                    }
                }
            }
            Opcode::ModInt => {
                let dst = inst.dst().0;
                set_reg_type(&mut reg_types, dst, KnownType::Int);
                TemplateInstruction::IntMod {
                    bc_offset,
                    dst,
                    lhs: inst.src1().0,
                    rhs: inst.src2().0,
                }
            }
            Opcode::ModFloat => {
                let dst = inst.dst().0;
                set_reg_type(&mut reg_types, dst, KnownType::Float);
                TemplateInstruction::FloatMod {
                    bc_offset,
                    dst,
                    lhs: inst.src1().0,
                    rhs: inst.src2().0,
                }
            }
            Opcode::Mod => {
                let dst = inst.dst().0;
                let lhs = inst.src1().0;
                let rhs = inst.src2().0;
                match infer_numeric_kind(op, bc_offset, lhs, rhs, &reg_types)? {
                    NumericKind::Int => {
                        set_reg_type(&mut reg_types, dst, KnownType::Int);
                        TemplateInstruction::IntMod {
                            bc_offset,
                            dst,
                            lhs,
                            rhs,
                        }
                    }
                    NumericKind::Float => {
                        set_reg_type(&mut reg_types, dst, KnownType::Float);
                        TemplateInstruction::FloatMod {
                            bc_offset,
                            dst,
                            lhs,
                            rhs,
                        }
                    }
                }
            }
            Opcode::NegInt => {
                let dst = inst.dst().0;
                set_reg_type(&mut reg_types, dst, KnownType::Int);
                TemplateInstruction::IntNeg {
                    bc_offset,
                    dst,
                    src: inst.src1().0,
                }
            }
            Opcode::AddFloat => {
                let dst = inst.dst().0;
                set_reg_type(&mut reg_types, dst, KnownType::Float);
                TemplateInstruction::FloatAdd {
                    bc_offset,
                    dst,
                    lhs: inst.src1().0,
                    rhs: inst.src2().0,
                }
            }
            Opcode::SubFloat => {
                let dst = inst.dst().0;
                set_reg_type(&mut reg_types, dst, KnownType::Float);
                TemplateInstruction::FloatSub {
                    bc_offset,
                    dst,
                    lhs: inst.src1().0,
                    rhs: inst.src2().0,
                }
            }
            Opcode::MulFloat => {
                let dst = inst.dst().0;
                set_reg_type(&mut reg_types, dst, KnownType::Float);
                TemplateInstruction::FloatMul {
                    bc_offset,
                    dst,
                    lhs: inst.src1().0,
                    rhs: inst.src2().0,
                }
            }
            Opcode::DivFloat => {
                let dst = inst.dst().0;
                set_reg_type(&mut reg_types, dst, KnownType::Float);
                TemplateInstruction::FloatDiv {
                    bc_offset,
                    dst,
                    lhs: inst.src1().0,
                    rhs: inst.src2().0,
                }
            }
            Opcode::TrueDiv => {
                let dst = inst.dst().0;
                let lhs = inst.src1().0;
                let rhs = inst.src2().0;
                match infer_numeric_kind(op, bc_offset, lhs, rhs, &reg_types)? {
                    NumericKind::Float => {
                        set_reg_type(&mut reg_types, dst, KnownType::Float);
                        TemplateInstruction::FloatDiv {
                            bc_offset,
                            dst,
                            lhs,
                            rhs,
                        }
                    }
                    NumericKind::Int => {
                        return Err(format!(
                            "tier1 lowering does not support int TrueDiv at bytecode offset {}",
                            bc_offset
                        ));
                    }
                }
            }
            Opcode::NegFloat => {
                let dst = inst.dst().0;
                set_reg_type(&mut reg_types, dst, KnownType::Float);
                TemplateInstruction::FloatNeg {
                    bc_offset,
                    dst,
                    src: inst.src1().0,
                }
            }

            Opcode::Lt => {
                let dst = inst.dst().0;
                let lhs = inst.src1().0;
                let rhs = inst.src2().0;
                set_reg_type(&mut reg_types, dst, KnownType::Bool);
                match infer_numeric_kind(op, bc_offset, lhs, rhs, &reg_types)? {
                    NumericKind::Int => TemplateInstruction::IntLt {
                        bc_offset,
                        dst,
                        lhs,
                        rhs,
                    },
                    NumericKind::Float => TemplateInstruction::FloatLt {
                        bc_offset,
                        dst,
                        lhs,
                        rhs,
                    },
                }
            }
            Opcode::Le => {
                let dst = inst.dst().0;
                let lhs = inst.src1().0;
                let rhs = inst.src2().0;
                set_reg_type(&mut reg_types, dst, KnownType::Bool);
                match infer_numeric_kind(op, bc_offset, lhs, rhs, &reg_types)? {
                    NumericKind::Int => TemplateInstruction::IntLe {
                        bc_offset,
                        dst,
                        lhs,
                        rhs,
                    },
                    NumericKind::Float => TemplateInstruction::FloatLe {
                        bc_offset,
                        dst,
                        lhs,
                        rhs,
                    },
                }
            }
            Opcode::Eq => {
                let dst = inst.dst().0;
                let lhs = inst.src1().0;
                let rhs = inst.src2().0;
                set_reg_type(&mut reg_types, dst, KnownType::Bool);
                match infer_numeric_kind(op, bc_offset, lhs, rhs, &reg_types)? {
                    NumericKind::Int => TemplateInstruction::IntEq {
                        bc_offset,
                        dst,
                        lhs,
                        rhs,
                    },
                    NumericKind::Float => TemplateInstruction::FloatEq {
                        bc_offset,
                        dst,
                        lhs,
                        rhs,
                    },
                }
            }
            Opcode::Ne => {
                let dst = inst.dst().0;
                let lhs = inst.src1().0;
                let rhs = inst.src2().0;
                set_reg_type(&mut reg_types, dst, KnownType::Bool);
                match infer_numeric_kind(op, bc_offset, lhs, rhs, &reg_types)? {
                    NumericKind::Int => TemplateInstruction::IntNe {
                        bc_offset,
                        dst,
                        lhs,
                        rhs,
                    },
                    NumericKind::Float => TemplateInstruction::FloatNe {
                        bc_offset,
                        dst,
                        lhs,
                        rhs,
                    },
                }
            }
            Opcode::Gt => {
                let dst = inst.dst().0;
                let lhs = inst.src1().0;
                let rhs = inst.src2().0;
                set_reg_type(&mut reg_types, dst, KnownType::Bool);
                match infer_numeric_kind(op, bc_offset, lhs, rhs, &reg_types)? {
                    NumericKind::Int => TemplateInstruction::IntGt {
                        bc_offset,
                        dst,
                        lhs,
                        rhs,
                    },
                    NumericKind::Float => TemplateInstruction::FloatGt {
                        bc_offset,
                        dst,
                        lhs,
                        rhs,
                    },
                }
            }
            Opcode::Ge => {
                let dst = inst.dst().0;
                let lhs = inst.src1().0;
                let rhs = inst.src2().0;
                set_reg_type(&mut reg_types, dst, KnownType::Bool);
                match infer_numeric_kind(op, bc_offset, lhs, rhs, &reg_types)? {
                    NumericKind::Int => TemplateInstruction::IntGe {
                        bc_offset,
                        dst,
                        lhs,
                        rhs,
                    },
                    NumericKind::Float => TemplateInstruction::FloatGe {
                        bc_offset,
                        dst,
                        lhs,
                        rhs,
                    },
                }
            }
            Opcode::Is => {
                let dst = inst.dst().0;
                set_reg_type(&mut reg_types, dst, KnownType::Bool);
                TemplateInstruction::Is {
                    bc_offset,
                    dst,
                    lhs: inst.src1().0,
                    rhs: inst.src2().0,
                }
            }
            Opcode::IsNot => {
                let dst = inst.dst().0;
                set_reg_type(&mut reg_types, dst, KnownType::Bool);
                TemplateInstruction::IsNot {
                    bc_offset,
                    dst,
                    lhs: inst.src1().0,
                    rhs: inst.src2().0,
                }
            }
            Opcode::BitwiseAnd => {
                let dst = inst.dst().0;
                set_reg_type(&mut reg_types, dst, KnownType::Int);
                TemplateInstruction::IntAnd {
                    bc_offset,
                    dst,
                    lhs: inst.src1().0,
                    rhs: inst.src2().0,
                }
            }
            Opcode::BitwiseOr => {
                let dst = inst.dst().0;
                set_reg_type(&mut reg_types, dst, KnownType::Int);
                TemplateInstruction::IntOr {
                    bc_offset,
                    dst,
                    lhs: inst.src1().0,
                    rhs: inst.src2().0,
                }
            }
            Opcode::BitwiseXor => {
                let dst = inst.dst().0;
                set_reg_type(&mut reg_types, dst, KnownType::Int);
                TemplateInstruction::IntXor {
                    bc_offset,
                    dst,
                    lhs: inst.src1().0,
                    rhs: inst.src2().0,
                }
            }
            Opcode::BitwiseNot => {
                let dst = inst.dst().0;
                set_reg_type(&mut reg_types, dst, KnownType::Int);
                TemplateInstruction::IntNot {
                    bc_offset,
                    dst,
                    src: inst.src1().0,
                }
            }
            Opcode::Shl => {
                let dst = inst.dst().0;
                set_reg_type(&mut reg_types, dst, KnownType::Int);
                TemplateInstruction::IntShl {
                    bc_offset,
                    dst,
                    lhs: inst.src1().0,
                    rhs: inst.src2().0,
                }
            }
            Opcode::Shr => {
                let dst = inst.dst().0;
                set_reg_type(&mut reg_types, dst, KnownType::Int);
                TemplateInstruction::IntShr {
                    bc_offset,
                    dst,
                    lhs: inst.src1().0,
                    rhs: inst.src2().0,
                }
            }
            Opcode::Not => {
                let dst = inst.dst().0;
                set_reg_type(&mut reg_types, dst, KnownType::Bool);
                TemplateInstruction::LogicalNot {
                    bc_offset,
                    dst,
                    src: inst.src1().0,
                }
            }

            Opcode::Jump => TemplateInstruction::Jump {
                bc_offset,
                target: calculate_jump_target(bc_offset, inst.imm16() as i16)?,
            },
            Opcode::JumpIfTrue => TemplateInstruction::BranchIfTrue {
                bc_offset,
                cond: inst.dst().0,
                target: calculate_jump_target(bc_offset, inst.imm16() as i16)?,
            },
            Opcode::JumpIfFalse => TemplateInstruction::BranchIfFalse {
                bc_offset,
                cond: inst.dst().0,
                target: calculate_jump_target(bc_offset, inst.imm16() as i16)?,
            },
            Opcode::JumpIfNone => TemplateInstruction::BranchIfNone {
                bc_offset,
                cond: inst.dst().0,
                target: calculate_jump_target(bc_offset, inst.imm16() as i16)?,
            },
            Opcode::JumpIfNotNone => TemplateInstruction::BranchIfNotNone {
                bc_offset,
                cond: inst.dst().0,
                target: calculate_jump_target(bc_offset, inst.imm16() as i16)?,
            },
            Opcode::Return => TemplateInstruction::Return {
                bc_offset,
                value: inst.dst().0,
            },
            Opcode::ReturnNone => TemplateInstruction::ReturnNone { bc_offset },

            _ => {
                return Err(format!(
                    "tier1 lowering unsupported opcode {:?} at bytecode offset {}",
                    op, bc_offset
                ));
            }
        };

        templates.push(template);
    }

    validate_jump_targets(code, &templates)?;
    Ok(templates)
}

fn load_const_template(
    code: &CodeObject,
    bc_offset: u32,
    dst: u8,
    const_idx: u16,
    inst_index: usize,
) -> Result<(TemplateInstruction, KnownType), String> {
    let idx = const_idx as usize;
    let value = code.constants.get(idx).copied().ok_or_else(|| {
        format!(
            "invalid constant index {} at instruction {} ({} constants)",
            const_idx,
            inst_index,
            code.constants.len()
        )
    })?;

    if value.is_none() {
        Ok((TemplateInstruction::LoadNone { bc_offset, dst }, KnownType::None))
    } else if let Some(b) = value.as_bool() {
        Ok((
            TemplateInstruction::LoadBool {
                bc_offset,
                dst,
                value: b,
            },
            KnownType::Bool,
        ))
    } else if let Some(i) = value.as_int() {
        Ok((
            TemplateInstruction::LoadInt {
                bc_offset,
                dst,
                value: i,
            },
            KnownType::Int,
        ))
    } else if let Some(f) = value.as_float() {
        Ok((
            TemplateInstruction::LoadFloat {
                bc_offset,
                dst,
                value: f,
            },
            KnownType::Float,
        ))
    } else {
        Err(format!(
            "tier1 lowering unsupported constant type at index {} for bytecode offset {}",
            const_idx, bc_offset
        ))
    }
}

fn calculate_jump_target(bc_offset: u32, relative: i16) -> Result<u32, String> {
    let target = bc_offset as i64 + 4 + (relative as i64 * 4);
    if !(0..=u32::MAX as i64).contains(&target) {
        return Err(format!(
            "jump target overflow from bytecode offset {} with delta {}",
            bc_offset, relative
        ));
    }
    Ok(target as u32)
}

fn validate_jump_targets(
    code: &CodeObject,
    templates: &[TemplateInstruction],
) -> Result<(), String> {
    let max_offset = (code.instructions.len() as u32) * 4;
    for template in templates {
        if let Some(target) = jump_target(template) {
            if target % 4 != 0 || target > max_offset {
                return Err(format!(
                    "invalid jump target {} from bytecode offset {} (max {})",
                    target,
                    template.bc_offset(),
                    max_offset
                ));
            }
        }
    }
    Ok(())
}

#[inline]
fn jump_target(template: &TemplateInstruction) -> Option<u32> {
    match template {
        TemplateInstruction::Jump { target, .. }
        | TemplateInstruction::BranchIfTrue { target, .. }
        | TemplateInstruction::BranchIfFalse { target, .. }
        | TemplateInstruction::BranchIfNone { target, .. }
        | TemplateInstruction::BranchIfNotNone { target, .. } => Some(*target),
        TemplateInstruction::EndAsyncFor { target, .. } => Some(*target as u32),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use prism_compiler::bytecode::{Instruction, Register};
    use prism_core::Value;

    #[test]
    fn test_lower_code_to_templates_rejects_invalid_jump_target() {
        let mut code = CodeObject::new("invalid_jump", "<test>");
        code.register_count = 1;
        code.instructions = vec![
            Instruction::op_di(Opcode::Jump, Register::new(0), 32_767),
            Instruction::op_d(Opcode::Return, Register::new(0)),
        ]
        .into_boxed_slice();

        let err = lower_code_to_templates(&code).unwrap_err();
        assert!(err.contains("invalid jump target"));
    }

    #[test]
    fn test_lower_code_to_templates_rejects_unlowered_opcode() {
        let mut code = CodeObject::new("unsupported", "<test>");
        code.register_count = 2;
        code.instructions = vec![
            Instruction::op_dss(
                Opcode::CallMethod,
                Register::new(0),
                Register::new(1),
                Register::new(0),
            ),
            Instruction::op_d(Opcode::Return, Register::new(0)),
        ]
        .into_boxed_slice();

        let err = lower_code_to_templates(&code).unwrap_err();
        assert!(err.contains("unsupported opcode"));
        assert!(err.contains("CallMethod"));
    }

    #[test]
    fn test_lower_code_to_templates_rejects_invalid_constant_index() {
        let mut code = CodeObject::new("bad_const_idx", "<test>");
        code.register_count = 1;
        code.constants = vec![Value::int(1).unwrap()].into_boxed_slice();
        code.instructions = vec![
            Instruction::op_di(Opcode::LoadConst, Register::new(0), 3),
            Instruction::op_d(Opcode::Return, Register::new(0)),
        ]
        .into_boxed_slice();

        let err = lower_code_to_templates(&code).unwrap_err();
        assert!(err.contains("invalid constant index"));
    }

    #[test]
    fn test_lower_code_to_templates_accepts_simple_function() {
        let mut code = CodeObject::new("simple", "<test>");
        code.register_count = 1;
        code.constants = vec![Value::int(7).unwrap()].into_boxed_slice();
        code.instructions = vec![
            Instruction::op_di(Opcode::LoadConst, Register::new(0), 0),
            Instruction::op_d(Opcode::Return, Register::new(0)),
        ]
        .into_boxed_slice();

        let templates = lower_code_to_templates(&code).unwrap();
        assert!(!templates.is_empty());
    }

    #[test]
    fn test_lower_code_to_templates_maps_return_none() {
        let mut code = CodeObject::new("ret_none", "<test>");
        code.register_count = 0;
        code.instructions = vec![Instruction::op(Opcode::ReturnNone)].into_boxed_slice();

        let templates = lower_code_to_templates(&code).unwrap();
        assert_eq!(templates.len(), 1);
        assert!(matches!(
            templates[0],
            TemplateInstruction::ReturnNone { bc_offset: 0 }
        ));
    }

    #[test]
    fn test_lower_code_to_templates_maps_primitive_load_consts() {
        let mut code = CodeObject::new("consts", "<test>");
        code.register_count = 4;
        code.constants = vec![
            Value::none(),
            Value::bool(true),
            Value::int(7).unwrap(),
            Value::float(3.5),
        ]
        .into_boxed_slice();
        code.instructions = vec![
            Instruction::op_di(Opcode::LoadConst, Register::new(0), 0),
            Instruction::op_di(Opcode::LoadConst, Register::new(1), 1),
            Instruction::op_di(Opcode::LoadConst, Register::new(2), 2),
            Instruction::op_di(Opcode::LoadConst, Register::new(3), 3),
            Instruction::op_d(Opcode::Return, Register::new(2)),
        ]
        .into_boxed_slice();

        let templates = lower_code_to_templates(&code).unwrap();
        assert!(matches!(
            templates[0],
            TemplateInstruction::LoadNone {
                bc_offset: 0,
                dst: 0
            }
        ));
        assert!(matches!(
            templates[1],
            TemplateInstruction::LoadBool {
                bc_offset: 4,
                dst: 1,
                value: true
            }
        ));
        assert!(matches!(
            templates[2],
            TemplateInstruction::LoadInt {
                bc_offset: 8,
                dst: 2,
                value: 7
            }
        ));
        assert!(matches!(
            templates[3],
            TemplateInstruction::LoadFloat {
                bc_offset: 12,
                dst: 3,
                value
            } if (value - 3.5).abs() < f64::EPSILON
        ));
    }

    #[test]
    fn test_lower_code_to_templates_computes_relative_jump_targets() {
        let mut code = CodeObject::new("jump_relative", "<test>");
        code.register_count = 1;
        code.instructions = vec![
            // offset 0 -> next_pc 4 + (1 * 4) => 8
            Instruction::op_di(Opcode::Jump, Register::new(0), 1),
            Instruction::op_d(Opcode::LoadNone, Register::new(0)),
            Instruction::op(Opcode::ReturnNone),
        ]
        .into_boxed_slice();

        let templates = lower_code_to_templates(&code).unwrap();
        assert!(matches!(
            templates[0],
            TemplateInstruction::Jump {
                bc_offset: 0,
                target: 8
            }
        ));
    }

    #[test]
    fn test_lower_code_to_templates_rejects_jump_underflow() {
        let mut code = CodeObject::new("jump_underflow", "<test>");
        code.register_count = 1;
        code.instructions = vec![
            // offset 0 -> next_pc 4 + (-2 * 4) => -4 (invalid)
            Instruction::op_di(Opcode::Jump, Register::new(0), (-2_i16) as u16),
            Instruction::op(Opcode::ReturnNone),
        ]
        .into_boxed_slice();

        let err = lower_code_to_templates(&code).unwrap_err();
        assert!(err.contains("jump target overflow"));
    }

    #[test]
    fn test_lower_code_to_templates_allows_jump_to_end_sentinel() {
        let mut code = CodeObject::new("jump_end", "<test>");
        code.register_count = 1;
        code.instructions = vec![
            // offset 0 -> next_pc 4 + (1 * 4) => 8 == max_offset
            Instruction::op_di(Opcode::Jump, Register::new(0), 1),
            Instruction::op(Opcode::ReturnNone),
        ]
        .into_boxed_slice();

        let templates = lower_code_to_templates(&code).unwrap();
        assert!(matches!(
            templates[0],
            TemplateInstruction::Jump {
                bc_offset: 0,
                target: 8
            }
        ));
    }

    #[test]
    fn test_lower_generic_add_int_specializes_to_int_add() {
        let mut code = CodeObject::new("add_int", "<test>");
        code.register_count = 3;
        code.constants = vec![Value::int(10).unwrap(), Value::int(32).unwrap()].into_boxed_slice();
        code.instructions = vec![
            Instruction::op_di(Opcode::LoadConst, Register::new(0), 0),
            Instruction::op_di(Opcode::LoadConst, Register::new(1), 1),
            Instruction::op_dss(
                Opcode::Add,
                Register::new(2),
                Register::new(0),
                Register::new(1),
            ),
            Instruction::op_d(Opcode::Return, Register::new(2)),
        ]
        .into_boxed_slice();

        let templates = lower_code_to_templates(&code).unwrap();
        assert!(matches!(
            templates[2],
            TemplateInstruction::IntAdd {
                bc_offset: 8,
                dst: 2,
                lhs: 0,
                rhs: 1
            }
        ));
    }

    #[test]
    fn test_lower_generic_add_float_specializes_to_float_add() {
        let mut code = CodeObject::new("add_float", "<test>");
        code.register_count = 3;
        code.constants = vec![Value::float(1.5), Value::float(2.5)].into_boxed_slice();
        code.instructions = vec![
            Instruction::op_di(Opcode::LoadConst, Register::new(0), 0),
            Instruction::op_di(Opcode::LoadConst, Register::new(1), 1),
            Instruction::op_dss(
                Opcode::Add,
                Register::new(2),
                Register::new(0),
                Register::new(1),
            ),
            Instruction::op_d(Opcode::Return, Register::new(2)),
        ]
        .into_boxed_slice();

        let templates = lower_code_to_templates(&code).unwrap();
        assert!(matches!(
            templates[2],
            TemplateInstruction::FloatAdd {
                bc_offset: 8,
                dst: 2,
                lhs: 0,
                rhs: 1
            }
        ));
    }

    #[test]
    fn test_lower_generic_add_rejects_mixed_numeric_types() {
        let mut code = CodeObject::new("add_mixed", "<test>");
        code.register_count = 3;
        code.constants = vec![Value::int(1).unwrap(), Value::float(2.0)].into_boxed_slice();
        code.instructions = vec![
            Instruction::op_di(Opcode::LoadConst, Register::new(0), 0),
            Instruction::op_di(Opcode::LoadConst, Register::new(1), 1),
            Instruction::op_dss(
                Opcode::Add,
                Register::new(2),
                Register::new(0),
                Register::new(1),
            ),
            Instruction::op_d(Opcode::Return, Register::new(2)),
        ]
        .into_boxed_slice();

        let err = lower_code_to_templates(&code).unwrap_err();
        assert!(err.contains("monomorphic numeric operands"));
        assert!(err.contains("Add"));
    }

    #[test]
    fn test_lower_generic_floor_div_int_specializes_to_int_div() {
        let mut code = CodeObject::new("floordiv_int", "<test>");
        code.register_count = 3;
        code.constants = vec![Value::int(9).unwrap(), Value::int(2).unwrap()].into_boxed_slice();
        code.instructions = vec![
            Instruction::op_di(Opcode::LoadConst, Register::new(0), 0),
            Instruction::op_di(Opcode::LoadConst, Register::new(1), 1),
            Instruction::op_dss(
                Opcode::FloorDiv,
                Register::new(2),
                Register::new(0),
                Register::new(1),
            ),
            Instruction::op_d(Opcode::Return, Register::new(2)),
        ]
        .into_boxed_slice();

        let templates = lower_code_to_templates(&code).unwrap();
        assert!(matches!(
            templates[2],
            TemplateInstruction::IntDiv {
                bc_offset: 8,
                dst: 2,
                lhs: 0,
                rhs: 1
            }
        ));
    }

    #[test]
    fn test_lower_generic_floor_div_float_specializes_to_float_floor_div() {
        let mut code = CodeObject::new("floordiv_float", "<test>");
        code.register_count = 3;
        code.constants = vec![Value::float(9.0), Value::float(2.0)].into_boxed_slice();
        code.instructions = vec![
            Instruction::op_di(Opcode::LoadConst, Register::new(0), 0),
            Instruction::op_di(Opcode::LoadConst, Register::new(1), 1),
            Instruction::op_dss(
                Opcode::FloorDiv,
                Register::new(2),
                Register::new(0),
                Register::new(1),
            ),
            Instruction::op_d(Opcode::Return, Register::new(2)),
        ]
        .into_boxed_slice();

        let templates = lower_code_to_templates(&code).unwrap();
        assert!(matches!(
            templates[2],
            TemplateInstruction::FloatFloorDiv {
                bc_offset: 8,
                dst: 2,
                lhs: 0,
                rhs: 1
            }
        ));
    }

    #[test]
    fn test_lower_generic_mod_float_specializes_to_float_mod() {
        let mut code = CodeObject::new("mod_float", "<test>");
        code.register_count = 3;
        code.constants = vec![Value::float(9.5), Value::float(2.0)].into_boxed_slice();
        code.instructions = vec![
            Instruction::op_di(Opcode::LoadConst, Register::new(0), 0),
            Instruction::op_di(Opcode::LoadConst, Register::new(1), 1),
            Instruction::op_dss(
                Opcode::Mod,
                Register::new(2),
                Register::new(0),
                Register::new(1),
            ),
            Instruction::op_d(Opcode::Return, Register::new(2)),
        ]
        .into_boxed_slice();

        let templates = lower_code_to_templates(&code).unwrap();
        assert!(matches!(
            templates[2],
            TemplateInstruction::FloatMod {
                bc_offset: 8,
                dst: 2,
                lhs: 0,
                rhs: 1
            }
        ));
    }

    #[test]
    fn test_lower_true_div_int_rejected() {
        let mut code = CodeObject::new("truediv_int", "<test>");
        code.register_count = 3;
        code.constants = vec![Value::int(7).unwrap(), Value::int(2).unwrap()].into_boxed_slice();
        code.instructions = vec![
            Instruction::op_di(Opcode::LoadConst, Register::new(0), 0),
            Instruction::op_di(Opcode::LoadConst, Register::new(1), 1),
            Instruction::op_dss(
                Opcode::TrueDiv,
                Register::new(2),
                Register::new(0),
                Register::new(1),
            ),
            Instruction::op_d(Opcode::Return, Register::new(2)),
        ]
        .into_boxed_slice();

        let err = lower_code_to_templates(&code).unwrap_err();
        assert!(err.contains("does not support int TrueDiv"));
    }

    #[test]
    fn test_lower_true_div_float_specializes_to_float_div() {
        let mut code = CodeObject::new("truediv_float", "<test>");
        code.register_count = 3;
        code.constants = vec![Value::float(7.0), Value::float(2.0)].into_boxed_slice();
        code.instructions = vec![
            Instruction::op_di(Opcode::LoadConst, Register::new(0), 0),
            Instruction::op_di(Opcode::LoadConst, Register::new(1), 1),
            Instruction::op_dss(
                Opcode::TrueDiv,
                Register::new(2),
                Register::new(0),
                Register::new(1),
            ),
            Instruction::op_d(Opcode::Return, Register::new(2)),
        ]
        .into_boxed_slice();

        let templates = lower_code_to_templates(&code).unwrap();
        assert!(matches!(
            templates[2],
            TemplateInstruction::FloatDiv {
                bc_offset: 8,
                dst: 2,
                lhs: 0,
                rhs: 1
            }
        ));
    }

    #[test]
    fn test_lower_generic_gt_float_specializes_to_float_gt() {
        let mut code = CodeObject::new("gt_float", "<test>");
        code.register_count = 3;
        code.constants = vec![Value::float(3.0), Value::float(2.0)].into_boxed_slice();
        code.instructions = vec![
            Instruction::op_di(Opcode::LoadConst, Register::new(0), 0),
            Instruction::op_di(Opcode::LoadConst, Register::new(1), 1),
            Instruction::op_dss(
                Opcode::Gt,
                Register::new(2),
                Register::new(0),
                Register::new(1),
            ),
            Instruction::op_d(Opcode::Return, Register::new(2)),
        ]
        .into_boxed_slice();

        let templates = lower_code_to_templates(&code).unwrap();
        assert!(matches!(
            templates[2],
            TemplateInstruction::FloatGt {
                bc_offset: 8,
                dst: 2,
                lhs: 0,
                rhs: 1
            }
        ));
    }

    #[test]
    fn test_lower_move_propagates_known_type() {
        let mut code = CodeObject::new("move_type_prop", "<test>");
        code.register_count = 4;
        code.constants = vec![Value::int(5).unwrap()].into_boxed_slice();
        code.instructions = vec![
            Instruction::op_di(Opcode::LoadConst, Register::new(0), 0),
            Instruction::op_ds(Opcode::Move, Register::new(1), Register::new(0)),
            Instruction::op_dss(
                Opcode::Add,
                Register::new(2),
                Register::new(1),
                Register::new(0),
            ),
            Instruction::op_d(Opcode::Return, Register::new(2)),
        ]
        .into_boxed_slice();

        let templates = lower_code_to_templates(&code).unwrap();
        assert!(matches!(
            templates[2],
            TemplateInstruction::IntAdd {
                bc_offset: 8,
                dst: 2,
                lhs: 1,
                rhs: 0
            }
        ));
    }

    #[test]
    fn test_lower_load_true_false_opcodes() {
        let mut code = CodeObject::new("bool_load_ops", "<test>");
        code.register_count = 2;
        code.instructions = vec![
            Instruction::op_d(Opcode::LoadTrue, Register::new(0)),
            Instruction::op_d(Opcode::LoadFalse, Register::new(1)),
            Instruction::op_d(Opcode::Return, Register::new(1)),
        ]
        .into_boxed_slice();

        let templates = lower_code_to_templates(&code).unwrap();
        assert!(matches!(
            templates[0],
            TemplateInstruction::LoadBool {
                bc_offset: 0,
                dst: 0,
                value: true
            }
        ));
        assert!(matches!(
            templates[1],
            TemplateInstruction::LoadBool {
                bc_offset: 4,
                dst: 1,
                value: false
            }
        ));
    }
}
