use crate::ir::builder::{
    ArithmeticBuilder, BitwiseBuilder, ContainerBuilder, ControlBuilder, GraphBuilder,
    ObjectBuilder,
};
use crate::ir::graph::Graph;
use crate::opt::speculation::SpeculationProvider;
use prism_compiler::bytecode::{CodeObject, Instruction, Opcode, Register};

/// Translator from Bytecode to Sea-of-Nodes IR.
pub struct BytecodeTranslator<'a> {
    builder: GraphBuilder,
    code: &'a CodeObject,
    /// Optional speculation provider for PGO-guided branch seeding.
    speculation: Option<Box<dyn SpeculationProvider>>,
}

impl<'a> BytecodeTranslator<'a> {
    /// Create a new translator.
    pub fn new(builder: GraphBuilder, code: &'a CodeObject) -> Self {
        BytecodeTranslator {
            builder,
            code,
            speculation: None,
        }
    }

    /// Attach a speculation provider for PGO-guided IR construction.
    pub fn with_speculation(mut self, provider: Box<dyn SpeculationProvider>) -> Self {
        self.speculation = Some(provider);
        self
    }

    /// Translate the bytecode to a Graph.
    pub fn translate(mut self) -> Result<Graph, String> {
        let instructions = &self.code.instructions;
        let len = instructions.len();
        let mut pc = 0;

        while pc < len {
            let offset = pc as u32;

            // Handle control flow merges
            self.builder.merge_state(offset);
            self.builder.set_bc_offset(offset);

            // Fetch instruction
            let inst = instructions[pc];

            // Dispatch instruction
            self.dispatch(inst, offset, len)?;

            pc += 1;
        }

        Ok(self.builder.finish())
    }

    /// Helper to get a value from a register.
    #[inline(always)]
    fn read_register(
        &mut self,
        reg: Register,
        offset: u32,
        op: Opcode,
    ) -> Result<crate::ir::node::NodeId, String> {
        let value = self.builder.get_register(reg.index() as u16);
        if value.is_valid() {
            Ok(value)
        } else {
            Err(format!(
                "read of uninitialized register r{} at instruction offset {} for opcode {:?}",
                reg.index(),
                offset,
                op
            ))
        }
    }

    /// Helper to set a value to a register.
    #[inline(always)]
    fn set_register(&mut self, reg: Register, node: crate::ir::node::NodeId) {
        self.builder.set_register(reg.index() as u16, node)
    }

    /// Resolve a signed relative jump target in instruction units.
    #[inline]
    fn resolve_jump_target(
        &self,
        offset: u32,
        relative: i16,
        instruction_count: usize,
    ) -> Result<u32, String> {
        let target = offset as i64 + 1 + relative as i64;
        let max = instruction_count as i64;
        if !(0..=max).contains(&target) {
            return Err(format!(
                "invalid jump target {target} at instruction offset {offset} (delta {relative}, max {max})"
            ));
        }
        Ok(target as u32)
    }

    /// Dispatch instruction to appropriate builder method.
    fn dispatch(
        &mut self,
        inst: Instruction,
        offset: u32,
        instruction_count: usize,
    ) -> Result<(), String> {
        let Some(op) = Opcode::from_u8(inst.opcode()) else {
            return Err(format!(
                "invalid opcode byte 0x{:02X} at instruction offset {}",
                inst.opcode(),
                offset
            ));
        };

        match op {
            Opcode::Nop => {}

            // Control Flow
            Opcode::Jump => {
                let target =
                    self.resolve_jump_target(offset, inst.imm16() as i16, instruction_count)?;
                self.builder.translate_jump(target);
            }
            Opcode::JumpIfFalse => {
                let cond = self.read_register(inst.dst(), offset, op)?;
                let target =
                    self.resolve_jump_target(offset, inst.imm16() as i16, instruction_count)?;
                // JumpIfFalse: if !cond goto target else goto fallthrough
                let fallthrough = offset + 1;
                self.builder.translate_branch(cond, fallthrough, target);
            }
            Opcode::JumpIfTrue => {
                let cond = self.read_register(inst.dst(), offset, op)?;
                let target =
                    self.resolve_jump_target(offset, inst.imm16() as i16, instruction_count)?;
                let fallthrough = offset + 1;
                self.builder.translate_branch(cond, target, fallthrough);
            }
            Opcode::JumpIfNone => {
                let value = self.read_register(inst.dst(), offset, op)?;
                let none = self.builder.const_none();
                let cond = self.builder.int_eq(value, none);
                let target =
                    self.resolve_jump_target(offset, inst.imm16() as i16, instruction_count)?;
                let fallthrough = offset + 1;
                self.builder.translate_branch(cond, target, fallthrough);
            }
            Opcode::JumpIfNotNone => {
                let value = self.read_register(inst.dst(), offset, op)?;
                let none = self.builder.const_none();
                let cond = self.builder.int_ne(value, none);
                let target =
                    self.resolve_jump_target(offset, inst.imm16() as i16, instruction_count)?;
                let fallthrough = offset + 1;
                self.builder.translate_branch(cond, target, fallthrough);
            }

            Opcode::Return => {
                let val = self.read_register(inst.dst(), offset, op)?;
                self.builder.return_value(val);
            }
            Opcode::ReturnNone => {
                let val = self.builder.const_none();
                self.builder.return_value(val);
            }

            // Constants / Loads
            Opcode::LoadConst => {
                let const_idx = inst.imm16() as usize;
                let val = self.code.constants.get(const_idx).ok_or_else(|| {
                    format!(
                        "invalid constant index {} at instruction offset {} ({} constants)",
                        const_idx,
                        offset,
                        self.code.constants.len()
                    )
                })?;

                let node = if let Some(i) = val.as_int() {
                    self.builder.const_int(i)
                } else if let Some(f) = val.as_float() {
                    self.builder.const_float(f)
                } else if let Some(b) = val.as_bool() {
                    self.builder.const_bool(b)
                } else if val.is_none() {
                    self.builder.const_none()
                } else {
                    return Err(format!(
                        "unsupported constant type at index {} for instruction offset {}",
                        const_idx, offset
                    ));
                };
                self.set_register(inst.dst(), node);
            }
            Opcode::LoadNone => {
                let val = self.builder.const_none();
                self.set_register(inst.dst(), val);
            }
            Opcode::LoadTrue => {
                let val = self.builder.const_bool(true);
                self.set_register(inst.dst(), val);
            }
            Opcode::LoadFalse => {
                let val = self.builder.const_bool(false);
                self.set_register(inst.dst(), val);
            }
            Opcode::Move => {
                let val = self.read_register(inst.src1(), offset, op)?;
                self.set_register(inst.dst(), val);
            }
            Opcode::LoadLocal => {
                let src = Register((inst.imm16() & 0xFF) as u8);
                let val = self.read_register(src, offset, op)?;
                self.set_register(inst.dst(), val);
            }
            Opcode::StoreLocal => {
                let src = self.read_register(inst.dst(), offset, op)?;
                let dst = Register((inst.imm16() & 0xFF) as u8);
                self.set_register(dst, src);
            }

            // Arithmetic
            Opcode::AddInt | Opcode::AddFloat | Opcode::Add => {
                let lhs = self.read_register(inst.src1(), offset, op)?;
                let rhs = self.read_register(inst.src2(), offset, op)?;
                let res = match op {
                    Opcode::AddInt => self.builder.int_add(lhs, rhs),
                    Opcode::AddFloat => self.builder.float_add(lhs, rhs),
                    _ => self.builder.generic_add(lhs, rhs),
                };
                self.set_register(inst.dst(), res);
            }
            Opcode::SubInt | Opcode::SubFloat | Opcode::Sub => {
                let lhs = self.read_register(inst.src1(), offset, op)?;
                let rhs = self.read_register(inst.src2(), offset, op)?;
                let res = match op {
                    Opcode::SubInt => self.builder.int_sub(lhs, rhs),
                    Opcode::SubFloat => self.builder.float_sub(lhs, rhs),
                    _ => self.builder.generic_sub(lhs, rhs),
                };
                self.set_register(inst.dst(), res);
            }
            Opcode::MulInt | Opcode::MulFloat | Opcode::Mul => {
                let lhs = self.read_register(inst.src1(), offset, op)?;
                let rhs = self.read_register(inst.src2(), offset, op)?;
                let res = match op {
                    Opcode::MulInt => self.builder.int_mul(lhs, rhs),
                    Opcode::MulFloat => self.builder.float_mul(lhs, rhs),
                    _ => self.builder.generic_mul(lhs, rhs),
                };
                self.set_register(inst.dst(), res);
            }
            Opcode::FloorDivInt => {
                let lhs = self.read_register(inst.src1(), offset, op)?;
                let rhs = self.read_register(inst.src2(), offset, op)?;
                let res = self.builder.int_div(lhs, rhs);
                self.set_register(inst.dst(), res);
            }
            Opcode::ModInt => {
                let lhs = self.read_register(inst.src1(), offset, op)?;
                let rhs = self.read_register(inst.src2(), offset, op)?;
                let res = self.builder.int_mod(lhs, rhs);
                self.set_register(inst.dst(), res);
            }
            Opcode::NegInt => {
                let src = self.read_register(inst.src1(), offset, op)?;
                let res = self.builder.int_neg(src);
                self.set_register(inst.dst(), res);
            }
            Opcode::PosInt => {
                let src = self.read_register(inst.src1(), offset, op)?;
                self.set_register(inst.dst(), src);
            }
            Opcode::DivFloat => {
                let lhs = self.read_register(inst.src1(), offset, op)?;
                let rhs = self.read_register(inst.src2(), offset, op)?;
                let res = self.builder.float_div(lhs, rhs);
                self.set_register(inst.dst(), res);
            }
            Opcode::FloorDivFloat | Opcode::ModFloat => {
                return Err(format!(
                    "unsupported opcode {:?} encountered at instruction offset {}",
                    op, offset
                ));
            }
            Opcode::NegFloat => {
                let src = self.read_register(inst.src1(), offset, op)?;
                let res = self.builder.float_neg(src);
                self.set_register(inst.dst(), res);
            }

            // Bitwise / logical
            Opcode::BitwiseAnd => {
                let lhs = self.read_register(inst.src1(), offset, op)?;
                let rhs = self.read_register(inst.src2(), offset, op)?;
                let res = self.builder.bitwise_and(lhs, rhs);
                self.set_register(inst.dst(), res);
            }
            Opcode::BitwiseOr => {
                let lhs = self.read_register(inst.src1(), offset, op)?;
                let rhs = self.read_register(inst.src2(), offset, op)?;
                let res = self.builder.bitwise_or(lhs, rhs);
                self.set_register(inst.dst(), res);
            }
            Opcode::BitwiseXor => {
                let lhs = self.read_register(inst.src1(), offset, op)?;
                let rhs = self.read_register(inst.src2(), offset, op)?;
                let res = self.builder.bitwise_xor(lhs, rhs);
                self.set_register(inst.dst(), res);
            }
            Opcode::BitwiseNot => {
                let src = self.read_register(inst.src1(), offset, op)?;
                let res = self.builder.bitwise_not(src);
                self.set_register(inst.dst(), res);
            }
            Opcode::Shl => {
                let lhs = self.read_register(inst.src1(), offset, op)?;
                let rhs = self.read_register(inst.src2(), offset, op)?;
                let res = self.builder.bitwise_shl(lhs, rhs);
                self.set_register(inst.dst(), res);
            }
            Opcode::Shr => {
                let lhs = self.read_register(inst.src1(), offset, op)?;
                let rhs = self.read_register(inst.src2(), offset, op)?;
                let res = self.builder.bitwise_shr(lhs, rhs);
                self.set_register(inst.dst(), res);
            }

            // Comparisons
            Opcode::Lt | Opcode::Le | Opcode::Eq | Opcode::Ne | Opcode::Gt | Opcode::Ge => {
                let lhs = self.read_register(inst.src1(), offset, op)?;
                let rhs = self.read_register(inst.src2(), offset, op)?;
                let res = match op {
                    Opcode::Lt => self.builder.generic_lt(lhs, rhs),
                    Opcode::Le => self.builder.generic_le(lhs, rhs),
                    Opcode::Eq => self.builder.generic_eq(lhs, rhs),
                    Opcode::Ne => self.builder.generic_ne(lhs, rhs),
                    Opcode::Gt => self.builder.generic_gt(lhs, rhs),
                    Opcode::Ge => self.builder.generic_ge(lhs, rhs),
                    _ => unreachable!(),
                };
                self.set_register(inst.dst(), res);
            }

            // Objects
            Opcode::GetAttr | Opcode::SetAttr | Opcode::GetItem | Opcode::SetItem => {
                self.dispatch_object_op(op, inst, offset)?;
            }

            // Calls
            Opcode::Call | Opcode::CallMethod => {
                self.dispatch_call_op(op, inst, offset)?;
            }

            // Containers
            Opcode::BuildList | Opcode::BuildTuple | Opcode::GetIter | Opcode::Len => {
                self.dispatch_container_op(op, inst, offset)?;
            }

            _ => {
                return Err(format!(
                    "unsupported opcode 0x{:02X} encountered at instruction offset {}",
                    inst.opcode(),
                    offset
                ));
            }
        }

        Ok(())
    }

    fn dispatch_object_op(
        &mut self,
        op: Opcode,
        inst: Instruction,
        offset: u32,
    ) -> Result<(), String> {
        match op {
            Opcode::GetAttr => {
                let obj = self.read_register(inst.src1(), offset, op)?;
                let name = self.read_register(inst.src2(), offset, op)?;
                let res = self.builder.get_attr(obj, name);
                self.set_register(inst.dst(), res);
            }
            Opcode::SetAttr => {
                let obj = self.read_register(inst.dst(), offset, op)?;
                let name = self.read_register(inst.src1(), offset, op)?;
                let val = self.read_register(inst.src2(), offset, op)?;
                self.builder.set_attr(obj, name, val);
            }
            Opcode::GetItem => {
                let obj = self.read_register(inst.src1(), offset, op)?;
                let key = self.read_register(inst.src2(), offset, op)?;
                let res = self.builder.get_item(obj, key);
                self.set_register(inst.dst(), res);
            }
            Opcode::SetItem => {
                let obj = self.read_register(inst.dst(), offset, op)?;
                let key = self.read_register(inst.src1(), offset, op)?;
                let val = self.read_register(inst.src2(), offset, op)?;
                self.builder.set_item(obj, key, val);
            }
            _ => {}
        }
        Ok(())
    }

    fn dispatch_container_op(
        &mut self,
        op: Opcode,
        inst: Instruction,
        offset: u32,
    ) -> Result<(), String> {
        match op {
            Opcode::BuildList => {
                let start_reg = inst.src1().index();
                let count = inst.src2().index();
                let mut elements = Vec::with_capacity(count as usize);
                for i in 0..count {
                    elements.push(self.read_register(Register(start_reg + i), offset, op)?);
                }
                let res = self.builder.build_list(&elements);
                self.set_register(inst.dst(), res);
            }
            Opcode::BuildTuple => {
                let start_reg = inst.src1().index();
                let count = inst.src2().index();
                let mut elements = Vec::with_capacity(count as usize);
                for i in 0..count {
                    elements.push(self.read_register(Register(start_reg + i), offset, op)?);
                }
                let res = self.builder.build_tuple(&elements);
                self.set_register(inst.dst(), res);
            }
            Opcode::GetIter => {
                let obj = self.read_register(inst.src1(), offset, op)?;
                let res = self.builder.get_iter(obj);
                self.set_register(inst.dst(), res);
            }
            Opcode::Len => {
                let obj = self.read_register(inst.src1(), offset, op)?;
                let res = self.builder.len(obj);
                self.set_register(inst.dst(), res);
            }
            _ => {}
        }
        Ok(())
    }

    fn dispatch_call_op(
        &mut self,
        op: Opcode,
        inst: Instruction,
        offset: u32,
    ) -> Result<(), String> {
        match op {
            Opcode::Call => {
                let func = self.read_register(inst.src1(), offset, op)?;
                let argc = inst.src2().index();
                let start_reg = inst.dst().index() + 1;

                let mut args = Vec::with_capacity(argc as usize);
                for i in 0..argc {
                    args.push(self.read_register(Register(start_reg + i), offset, op)?);
                }
                let res = self.builder.call(func, &args);
                self.set_register(inst.dst(), res);
            }
            Opcode::CallMethod => {
                // CallMethod encoding:
                // - src1: method register (loaded by LoadMethod)
                // - src1+1: implicit self slot (or None marker for unbound)
                // - src2: explicit argument count
                // - src1+2..: explicit arguments
                let method_reg = inst.src1().index();
                let argc = inst.src2().index() as u16;

                let self_reg_idx = method_reg as u16 + 1;
                if self_reg_idx > u8::MAX as u16 {
                    return Err(format!(
                        "CallMethod self register overflow at instruction offset {} (base r{})",
                        offset, method_reg
                    ));
                }

                let mut args = Vec::with_capacity(argc as usize + 1);
                args.push(self.read_register(Register(self_reg_idx as u8), offset, op)?);

                for i in 0..argc {
                    let arg_reg_idx = method_reg as u16 + 2 + i;
                    if arg_reg_idx > u8::MAX as u16 {
                        return Err(format!(
                            "CallMethod argument register overflow at instruction offset {} (base r{}, arg index {})",
                            offset, method_reg, i
                        ));
                    }
                    args.push(self.read_register(Register(arg_reg_idx as u8), offset, op)?);
                }

                let func = self.read_register(inst.src1(), offset, op)?;
                let res = self.builder.call(func, &args);
                self.set_register(inst.dst(), res);
            }
            _ => {}
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::operators::{ArithOp, BitwiseOp, CallKind, CmpOp, Operator};
    use prism_compiler::bytecode::Register;

    #[test]
    fn test_translate_simple_return_is_ok() {
        let mut code = CodeObject::new("simple", "<test>");
        code.register_count = 1;
        code.instructions =
            vec![Instruction::op_d(Opcode::ReturnNone, Register::new(0))].into_boxed_slice();

        let builder = GraphBuilder::new(1, 0);
        let translator = BytecodeTranslator::new(builder, &code);
        let graph = translator.translate();
        assert!(graph.is_ok());
    }

    #[test]
    fn test_translate_rejects_unsupported_opcode() {
        let mut code = CodeObject::new("unsupported", "<test>");
        code.register_count = 1;
        code.instructions =
            vec![Instruction::op_d(Opcode::BuildClass, Register::new(0))].into_boxed_slice();

        let builder = GraphBuilder::new(1, 0);
        let translator = BytecodeTranslator::new(builder, &code);
        let err = translator.translate().unwrap_err();
        assert!(err.contains("unsupported opcode"));
    }

    #[test]
    fn test_translate_rejects_invalid_constant_index() {
        let mut code = CodeObject::new("bad_const", "<test>");
        code.register_count = 1;
        code.instructions = vec![
            Instruction::op_di(Opcode::LoadConst, Register::new(0), 99),
            Instruction::op_d(Opcode::Return, Register::new(0)),
        ]
        .into_boxed_slice();

        let builder = GraphBuilder::new(1, 0);
        let translator = BytecodeTranslator::new(builder, &code);
        let err = translator.translate().unwrap_err();
        assert!(err.contains("invalid constant index"));
    }

    #[test]
    fn test_translate_rejects_jump_target_underflow() {
        let mut code = CodeObject::new("bad_jump_underflow", "<test>");
        code.register_count = 1;
        code.instructions = vec![
            Instruction::op_di(Opcode::Jump, Register::new(0), (-2_i16) as u16),
            Instruction::op(Opcode::ReturnNone),
        ]
        .into_boxed_slice();

        let builder = GraphBuilder::new(1, 0);
        let translator = BytecodeTranslator::new(builder, &code);
        let err = translator.translate().unwrap_err();
        assert!(err.contains("invalid jump target"));
    }

    #[test]
    fn test_translate_rejects_jump_target_overflow() {
        let mut code = CodeObject::new("bad_jump_overflow", "<test>");
        code.register_count = 1;
        code.instructions = vec![
            Instruction::op_di(Opcode::Jump, Register::new(0), 10),
            Instruction::op(Opcode::ReturnNone),
        ]
        .into_boxed_slice();

        let builder = GraphBuilder::new(1, 0);
        let translator = BytecodeTranslator::new(builder, &code);
        let err = translator.translate().unwrap_err();
        assert!(err.contains("invalid jump target"));
    }

    #[test]
    fn test_translate_accepts_jump_to_end() {
        let mut code = CodeObject::new("jump_to_end", "<test>");
        code.register_count = 1;
        code.instructions = vec![
            // offset 0 -> next is 1, rel 1 => target 2 (end sentinel)
            Instruction::op_di(Opcode::Jump, Register::new(0), 1),
            Instruction::op(Opcode::ReturnNone),
        ]
        .into_boxed_slice();

        let builder = GraphBuilder::new(1, 0);
        let translator = BytecodeTranslator::new(builder, &code);
        assert!(translator.translate().is_ok());
    }

    #[test]
    fn test_translate_rejects_invalid_opcode_byte() {
        let mut code = CodeObject::new("invalid_opcode", "<test>");
        code.register_count = 1;
        code.instructions = vec![Instruction::from_raw(0xFF00_0000)].into_boxed_slice();

        let builder = GraphBuilder::new(1, 0);
        let translator = BytecodeTranslator::new(builder, &code);
        let err = translator.translate().unwrap_err();
        assert!(err.contains("invalid opcode byte"));
    }

    #[test]
    fn test_translate_rejects_uninitialized_register_read() {
        let mut code = CodeObject::new("uninitialized_read", "<test>");
        code.register_count = 3;
        code.instructions = vec![
            Instruction::op_dss(
                Opcode::Add,
                Register::new(2),
                Register::new(0),
                Register::new(1),
            ),
            Instruction::op_d(Opcode::Return, Register::new(2)),
        ]
        .into_boxed_slice();

        let builder = GraphBuilder::new(3, 0);
        let translator = BytecodeTranslator::new(builder, &code);
        let err = translator.translate().unwrap_err();
        assert!(err.contains("uninitialized register"));
        assert!(err.contains("r0") || err.contains("r1"));
    }

    #[test]
    fn test_translate_reads_argument_registers_from_parameters() {
        let mut code = CodeObject::new("arg_add", "<test>");
        code.register_count = 2;
        code.arg_count = 2;
        code.instructions = vec![
            Instruction::op_dss(
                Opcode::Add,
                Register::new(0),
                Register::new(0),
                Register::new(1),
            ),
            Instruction::op_d(Opcode::Return, Register::new(0)),
        ]
        .into_boxed_slice();

        let builder = GraphBuilder::new(2, 2);
        let translator = BytecodeTranslator::new(builder, &code);
        let graph = translator
            .translate()
            .expect("argument registers should be seeded");

        let mut found_arg_add = false;
        for (_, node) in graph.iter() {
            if let Operator::GenericOp(ArithOp::Add) = node.op {
                let lhs = graph.node(node.inputs.get(0).unwrap()).op;
                let rhs = graph.node(node.inputs.get(1).unwrap()).op;
                assert!(matches!(lhs, Operator::Parameter(0)));
                assert!(matches!(rhs, Operator::Parameter(1)));
                found_arg_add = true;
                break;
            }
        }

        assert!(
            found_arg_add,
            "expected translated add node using parameter inputs"
        );
    }

    #[test]
    fn test_translate_call_method_uses_method_self_and_explicit_args() {
        let mut code = CodeObject::new("call_method", "<test>");
        code.register_count = 4;
        code.arg_count = 4; // r0..r3 seeded as parameters
        code.instructions = vec![
            // r0 = r1(r2, r3)
            Instruction::new(Opcode::CallMethod, 0, 1, 1),
            Instruction::op_d(Opcode::Return, Register::new(0)),
        ]
        .into_boxed_slice();

        let builder = GraphBuilder::new(4, 4);
        let translator = BytecodeTranslator::new(builder, &code);
        let graph = translator
            .translate()
            .expect("call method translation should succeed");

        let mut call_verified = false;
        for (_, node) in graph.iter() {
            if let Operator::Call(CallKind::Direct) = node.op {
                assert_eq!(node.inputs.len(), 3);
                let func_op = graph.node(node.inputs.get(0).unwrap()).op;
                let self_op = graph.node(node.inputs.get(1).unwrap()).op;
                let arg0_op = graph.node(node.inputs.get(2).unwrap()).op;
                assert!(matches!(func_op, Operator::Parameter(1)));
                assert!(matches!(self_op, Operator::Parameter(2)));
                assert!(matches!(arg0_op, Operator::Parameter(3)));
                call_verified = true;
                break;
            }
        }

        assert!(call_verified, "expected translated CallMethod call node");
    }

    #[test]
    fn test_translate_call_method_rejects_uninitialized_self_slot() {
        let mut code = CodeObject::new("call_method_uninit_self", "<test>");
        code.register_count = 4;
        code.arg_count = 2; // r2 (self slot) is uninitialized
        code.instructions = vec![
            Instruction::new(Opcode::CallMethod, 0, 1, 0),
            Instruction::op_d(Opcode::Return, Register::new(0)),
        ]
        .into_boxed_slice();

        let builder = GraphBuilder::new(4, 2);
        let translator = BytecodeTranslator::new(builder, &code);
        let err = translator.translate().unwrap_err();
        assert!(err.contains("uninitialized register"));
        assert!(err.contains("r2"));
    }

    #[test]
    fn test_translate_jump_if_none_and_not_none_use_int_cmp_against_none() {
        let mut code = CodeObject::new("jump_if_none", "<test>");
        code.register_count = 1;
        code.arg_count = 1;
        code.instructions = vec![
            Instruction::op_di(Opcode::JumpIfNone, Register::new(0), 1),
            Instruction::op_di(Opcode::JumpIfNotNone, Register::new(0), 1),
            Instruction::op_d(Opcode::Return, Register::new(0)),
            Instruction::op_d(Opcode::Return, Register::new(0)),
        ]
        .into_boxed_slice();

        let builder = GraphBuilder::new(1, 1);
        let translator = BytecodeTranslator::new(builder, &code);
        let graph = translator.translate().expect("translation should succeed");

        assert!(
            graph
                .iter()
                .any(|(_, node)| matches!(node.op, Operator::IntCmp(CmpOp::Eq))),
            "JumpIfNone should lower through IntCmp(Eq) against ConstNone"
        );
        assert!(
            graph
                .iter()
                .any(|(_, node)| matches!(node.op, Operator::IntCmp(CmpOp::Ne))),
            "JumpIfNotNone should lower through IntCmp(Ne) against ConstNone"
        );
        assert!(
            graph
                .iter()
                .any(|(_, node)| matches!(node.op, Operator::ConstNone)),
            "None checks should materialize a ConstNone node"
        );
    }

    #[test]
    fn test_translate_load_store_local_updates_register_aliases() {
        let mut code = CodeObject::new("local_alias", "<test>");
        code.register_count = 3;
        code.arg_count = 1;
        code.instructions = vec![
            Instruction::op_ds(Opcode::Move, Register::new(1), Register::new(0)),
            // Store r1 into local/register slot 2
            Instruction::op_di(Opcode::StoreLocal, Register::new(1), 2),
            // Load slot 2 back into r0
            Instruction::op_di(Opcode::LoadLocal, Register::new(0), 2),
            Instruction::op_d(Opcode::Return, Register::new(0)),
        ]
        .into_boxed_slice();

        let builder = GraphBuilder::new(3, 1);
        let translator = BytecodeTranslator::new(builder, &code);
        let graph = translator.translate().expect("translation should succeed");

        let mut saw_parameter_return = false;
        for (_, node) in graph.iter() {
            if let Operator::Control(crate::ir::operators::ControlOp::Return) = node.op {
                let value = node.inputs.get(1).expect("return must carry value input");
                if matches!(graph.node(value).op, Operator::Parameter(0)) {
                    saw_parameter_return = true;
                    break;
                }
            }
        }
        assert!(
            saw_parameter_return,
            "local load/store aliasing should preserve the underlying SSA value"
        );
    }

    #[test]
    fn test_translate_typed_arithmetic_and_bitwise_opcodes() {
        let mut code = CodeObject::new("typed_ops", "<test>");
        code.register_count = 12;
        code.arg_count = 2;
        code.instructions = vec![
            Instruction::op_dss(
                Opcode::FloorDivInt,
                Register::new(2),
                Register::new(0),
                Register::new(1),
            ),
            Instruction::op_dss(
                Opcode::ModInt,
                Register::new(3),
                Register::new(0),
                Register::new(1),
            ),
            Instruction::op_ds(Opcode::NegInt, Register::new(4), Register::new(0)),
            Instruction::op_ds(Opcode::PosInt, Register::new(5), Register::new(0)),
            Instruction::op_dss(
                Opcode::DivFloat,
                Register::new(6),
                Register::new(0),
                Register::new(1),
            ),
            Instruction::op_ds(Opcode::NegFloat, Register::new(7), Register::new(0)),
            Instruction::op_dss(
                Opcode::BitwiseAnd,
                Register::new(8),
                Register::new(0),
                Register::new(1),
            ),
            Instruction::op_dss(
                Opcode::BitwiseOr,
                Register::new(9),
                Register::new(0),
                Register::new(1),
            ),
            Instruction::op_dss(
                Opcode::BitwiseXor,
                Register::new(10),
                Register::new(0),
                Register::new(1),
            ),
            Instruction::op_ds(Opcode::BitwiseNot, Register::new(11), Register::new(0)),
            Instruction::op_dss(
                Opcode::Shl,
                Register::new(2),
                Register::new(2),
                Register::new(1),
            ),
            Instruction::op_dss(
                Opcode::Shr,
                Register::new(3),
                Register::new(3),
                Register::new(1),
            ),
            Instruction::op_d(Opcode::Return, Register::new(2)),
        ]
        .into_boxed_slice();

        let builder = GraphBuilder::new(12, 2);
        let translator = BytecodeTranslator::new(builder, &code);
        let graph = translator.translate().expect("typed opcode translation should succeed");

        let has_op = |pred: fn(Operator) -> bool| graph.iter().any(|(_, node)| pred(node.op));
        assert!(has_op(|op| op == Operator::IntOp(ArithOp::FloorDiv)));
        assert!(has_op(|op| op == Operator::IntOp(ArithOp::Mod)));
        assert!(has_op(|op| op == Operator::IntOp(ArithOp::Neg)));
        assert!(has_op(|op| op == Operator::FloatOp(ArithOp::TrueDiv)));
        assert!(has_op(|op| op == Operator::FloatOp(ArithOp::Neg)));
        assert!(has_op(|op| op == Operator::Bitwise(BitwiseOp::And)));
        assert!(has_op(|op| op == Operator::Bitwise(BitwiseOp::Or)));
        assert!(has_op(|op| op == Operator::Bitwise(BitwiseOp::Xor)));
        assert!(has_op(|op| op == Operator::Bitwise(BitwiseOp::Not)));
        assert!(has_op(|op| op == Operator::Bitwise(BitwiseOp::Shl)));
        assert!(has_op(|op| op == Operator::Bitwise(BitwiseOp::Shr)));
    }
}
