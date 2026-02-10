use crate::ir::builder::{
    ArithmeticBuilder, ContainerBuilder, ControlBuilder, GraphBuilder, ObjectBuilder,
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
    pub fn translate(mut self) -> Graph {
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
            self.dispatch(inst, offset);

            pc += 1;
        }

        self.builder.finish()
    }

    /// Helper to get a value from a register.
    #[inline(always)]
    fn get_register(&mut self, reg: Register) -> crate::ir::node::NodeId {
        self.builder.get_register(reg.index() as u16)
    }

    /// Helper to set a value to a register.
    #[inline(always)]
    fn set_register(&mut self, reg: Register, node: crate::ir::node::NodeId) {
        self.builder.set_register(reg.index() as u16, node)
    }

    /// Dispatch instruction to appropriate builder method.
    fn dispatch(&mut self, inst: Instruction, offset: u32) {
        if let Some(op) = Opcode::from_u8(inst.opcode()) {
            match op {
                // Control Flow
                Opcode::Jump => {
                    let rel = inst.imm16() as i16;
                    let target = (offset as i32 + rel as i32) as u32;
                    self.builder.translate_jump(target);
                }
                Opcode::JumpIfFalse => {
                    let cond = self.get_register(inst.dst());
                    let rel = inst.imm16() as i16;
                    let target = (offset as i32 + rel as i32) as u32;
                    // JumpIfFalse: if !cond goto target else goto fallthrough
                    // fallthrough is offset + 1 (next instruction)
                    let fallthrough = offset + 1;
                    self.builder.translate_branch(cond, fallthrough, target);
                }
                Opcode::JumpIfTrue => {
                    let cond = self.get_register(inst.dst());
                    let rel = inst.imm16() as i16;
                    let target = (offset as i32 + rel as i32) as u32;
                    let fallthrough = offset + 1;
                    self.builder.translate_branch(cond, target, fallthrough);
                }

                Opcode::Return => {
                    let val = self.get_register(inst.dst());
                    self.builder.return_value(val);
                }
                Opcode::ReturnNone => {
                    let val = self.builder.const_none();
                    self.builder.return_value(val);
                }

                // Constants / Loads
                Opcode::LoadConst => {
                    let const_idx = inst.imm16();
                    if let Some(val) = self.code.constants.get(const_idx as usize) {
                        let node = if let Some(i) = val.as_int() {
                            self.builder.const_int(i)
                        } else if let Some(f) = val.as_float() {
                            self.builder.const_float(f)
                        } else if let Some(b) = val.as_bool() {
                            self.builder.const_bool(b)
                        } else if val.is_none() {
                            self.builder.const_none()
                        } else {
                            self.builder.const_none()
                        };
                        self.set_register(inst.dst(), node);
                    }
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
                    let val = self.get_register(inst.src1());
                    self.set_register(inst.dst(), val);
                }

                // Arithmetic
                Opcode::AddInt | Opcode::AddFloat | Opcode::Add => {
                    let lhs = self.get_register(inst.src1());
                    let rhs = self.get_register(inst.src2());
                    let res = match op {
                        Opcode::AddInt => self.builder.int_add(lhs, rhs),
                        Opcode::AddFloat => self.builder.float_add(lhs, rhs),
                        _ => self.builder.generic_add(lhs, rhs),
                    };
                    self.set_register(inst.dst(), res);
                }
                Opcode::SubInt | Opcode::SubFloat | Opcode::Sub => {
                    let lhs = self.get_register(inst.src1());
                    let rhs = self.get_register(inst.src2());
                    let res = match op {
                        Opcode::SubInt => self.builder.int_sub(lhs, rhs),
                        Opcode::SubFloat => self.builder.float_sub(lhs, rhs),
                        _ => self.builder.generic_sub(lhs, rhs),
                    };
                    self.set_register(inst.dst(), res);
                }
                Opcode::MulInt | Opcode::MulFloat | Opcode::Mul => {
                    let lhs = self.get_register(inst.src1());
                    let rhs = self.get_register(inst.src2());
                    let res = match op {
                        Opcode::MulInt => self.builder.int_mul(lhs, rhs),
                        Opcode::MulFloat => self.builder.float_mul(lhs, rhs),
                        _ => self.builder.generic_mul(lhs, rhs),
                    };
                    self.set_register(inst.dst(), res);
                }
                // ... extend other arithmetic ...

                // Comparisons
                Opcode::Lt | Opcode::Le | Opcode::Eq | Opcode::Ne | Opcode::Gt | Opcode::Ge => {
                    let lhs = self.get_register(inst.src1());
                    let rhs = self.get_register(inst.src2());
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
                    self.dispatch_object_op(op, inst);
                }

                // Calls
                Opcode::Call | Opcode::CallMethod => {
                    self.dispatch_call_op(op, inst);
                }

                // Containers
                Opcode::BuildList | Opcode::BuildTuple | Opcode::GetIter | Opcode::Len => {
                    self.dispatch_container_op(op, inst);
                }

                _ => {
                    // Unimplemented opcodes
                }
            }
        }
    }

    fn dispatch_object_op(&mut self, op: Opcode, inst: Instruction) {
        match op {
            Opcode::GetAttr => {
                let obj = self.get_register(inst.src1());
                let name = self.get_register(inst.src2());
                let res = self.builder.get_attr(obj, name);
                self.set_register(inst.dst(), res);
            }
            Opcode::SetAttr => {
                let obj = self.get_register(inst.dst());
                let name = self.get_register(inst.src1());
                let val = self.get_register(inst.src2());
                self.builder.set_attr(obj, name, val);
            }
            Opcode::GetItem => {
                let obj = self.get_register(inst.src1());
                let key = self.get_register(inst.src2());
                let res = self.builder.get_item(obj, key);
                self.set_register(inst.dst(), res);
            }
            Opcode::SetItem => {
                let obj = self.get_register(inst.dst());
                let key = self.get_register(inst.src1());
                let val = self.get_register(inst.src2());
                self.builder.set_item(obj, key, val);
            }
            _ => {}
        }
    }

    fn dispatch_container_op(&mut self, op: Opcode, inst: Instruction) {
        match op {
            Opcode::BuildList => {
                let start_reg = inst.src1().index();
                let count = inst.src2().index();
                let mut elements = Vec::with_capacity(count as usize);
                for i in 0..count {
                    elements.push(self.get_register(Register(start_reg + i)));
                }
                let res = self.builder.build_list(&elements);
                self.set_register(inst.dst(), res);
            }
            Opcode::BuildTuple => {
                let start_reg = inst.src1().index();
                let count = inst.src2().index();
                let mut elements = Vec::with_capacity(count as usize);
                for i in 0..count {
                    elements.push(self.get_register(Register(start_reg + i)));
                }
                let res = self.builder.build_tuple(&elements);
                self.set_register(inst.dst(), res);
            }
            Opcode::GetIter => {
                let obj = self.get_register(inst.src1());
                let res = self.builder.get_iter(obj);
                self.set_register(inst.dst(), res);
            }
            Opcode::Len => {
                let obj = self.get_register(inst.src1());
                let res = self.builder.len(obj);
                self.set_register(inst.dst(), res);
            }
            _ => {}
        }
    }

    fn dispatch_call_op(&mut self, op: Opcode, inst: Instruction) {
        match op {
            Opcode::Call => {
                let func = self.get_register(inst.src1());
                let argc = inst.src2().index();
                let start_reg = inst.dst().index() + 1;

                let mut args = Vec::with_capacity(argc as usize);
                for i in 0..argc {
                    args.push(self.get_register(Register(start_reg + i)));
                }
                let res = self.builder.call(func, &args);
                self.set_register(inst.dst(), res);
            }
            Opcode::CallMethod => {
                // TODO: Determine CallMethod encoding.
            }
            _ => {}
        }
    }
}
