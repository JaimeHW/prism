//! AST to register-based bytecode compiler.
//!
//! The compiler transforms a parsed Python AST into executable bytecode
//! using a two-phase approach:
//!
//! 1. **Scope analysis**: Builds symbol tables and determines variable scopes
//! 2. **Code generation**: Emits register-based bytecode instructions

use crate::bytecode::{
    CodeFlags, CodeObject, FunctionBuilder, Instruction, LocalSlot, Opcode, Register,
};
use crate::scope::{ScopeAnalyzer, SymbolTable};
use prism_parser::ast::{
    AugOp, BinOp, BoolOp, CmpOp, Expr, ExprKind, Module, Stmt, StmtKind, UnaryOp,
};
use std::sync::Arc;

/// Compilation error.
#[derive(Debug, Clone)]
pub struct CompileError {
    /// Error message.
    pub message: String,
    /// Line number (1-indexed).
    pub line: u32,
    /// Column number (0-indexed).
    pub column: u32,
}

impl std::fmt::Display for CompileError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}:{}: {}", self.line, self.column, self.message)
    }
}

impl std::error::Error for CompileError {}

/// Result type for compilation.
pub type CompileResult<T> = Result<T, CompileError>;

/// Bytecode compiler.
pub struct Compiler {
    /// Current function builder.
    builder: FunctionBuilder,
    /// Symbol table from scope analysis.
    #[allow(dead_code)]
    symbol_table: SymbolTable,
    /// Source filename.
    #[allow(dead_code)]
    filename: Arc<str>,
}

impl Compiler {
    /// Create a new compiler for a module.
    pub fn new(filename: impl Into<Arc<str>>) -> Self {
        let filename = filename.into();
        Self {
            builder: FunctionBuilder::new("<module>"),
            symbol_table: SymbolTable::new("<module>"),
            filename,
        }
    }

    /// Compile a module to bytecode.
    pub fn compile_module(module: &Module, filename: &str) -> CompileResult<CodeObject> {
        // Phase 1: Scope analysis
        let symbol_table = ScopeAnalyzer::new().analyze(module, "<module>");

        // Phase 2: Code generation
        let mut compiler = Compiler {
            builder: FunctionBuilder::new("<module>"),
            symbol_table,
            filename: filename.into(),
        };

        compiler.builder.set_filename(filename);
        compiler.builder.add_flags(CodeFlags::MODULE);

        for stmt in &module.body {
            compiler.compile_stmt(stmt)?;
        }

        // Implicit return None at end of module
        compiler.builder.emit_return_none();

        Ok(compiler.builder.finish())
    }

    /// Compile a statement.
    fn compile_stmt(&mut self, stmt: &Stmt) -> CompileResult<()> {
        // Use span start as a line approximation (byte offset for now)
        self.builder.set_line(stmt.span.start);

        match &stmt.kind {
            StmtKind::Expr(value) => {
                // Expression statement - evaluate and discard
                let reg = self.compile_expr(value)?;
                self.builder.free_register(reg);
            }

            StmtKind::Assign { targets, value } => {
                let value_reg = self.compile_expr(value)?;

                for target in targets {
                    self.compile_store(target, value_reg)?;
                }

                self.builder.free_register(value_reg);
            }

            StmtKind::AugAssign { target, op, value } => {
                let left_reg = self.compile_expr(target)?;
                let right_reg = self.compile_expr(value)?;
                let result_reg = self.builder.alloc_register();

                self.emit_augop(*op, result_reg, left_reg, right_reg);

                self.compile_store(target, result_reg)?;

                self.builder.free_register(left_reg);
                self.builder.free_register(right_reg);
                self.builder.free_register(result_reg);
            }

            StmtKind::Return(value) => {
                if let Some(val) = value {
                    let reg = self.compile_expr(val)?;
                    self.builder.emit_return(reg);
                    self.builder.free_register(reg);
                } else {
                    self.builder.emit_return_none();
                }
            }

            StmtKind::Pass => {
                // No-op
            }

            StmtKind::Break => {
                // TODO: Emit jump to loop exit (requires loop context tracking)
                return Err(CompileError {
                    message: "break outside loop".to_string(),
                    line: stmt.span.start,
                    column: 0,
                });
            }

            StmtKind::Continue => {
                // TODO: Emit jump to loop start (requires loop context tracking)
                return Err(CompileError {
                    message: "continue outside loop".to_string(),
                    line: stmt.span.start,
                    column: 0,
                });
            }

            StmtKind::If { test, body, orelse } => {
                let cond_reg = self.compile_expr(test)?;
                let else_label = self.builder.create_label();
                let end_label = self.builder.create_label();

                self.builder.emit_jump_if_false(cond_reg, else_label);
                self.builder.free_register(cond_reg);

                // Then branch
                for s in body {
                    self.compile_stmt(s)?;
                }

                if !orelse.is_empty() {
                    self.builder.emit_jump(end_label);
                }

                self.builder.bind_label(else_label);

                // Else branch
                for s in orelse {
                    self.compile_stmt(s)?;
                }

                if !orelse.is_empty() {
                    self.builder.bind_label(end_label);
                }
            }

            StmtKind::While { test, body, orelse } => {
                let loop_start = self.builder.create_label();
                let loop_else = self.builder.create_label();
                let loop_end = self.builder.create_label();

                self.builder.bind_label(loop_start);

                let cond_reg = self.compile_expr(test)?;
                self.builder.emit_jump_if_false(cond_reg, loop_else);
                self.builder.free_register(cond_reg);

                for s in body {
                    self.compile_stmt(s)?;
                }

                self.builder.emit_jump(loop_start);
                self.builder.bind_label(loop_else);

                for s in orelse {
                    self.compile_stmt(s)?;
                }

                self.builder.bind_label(loop_end);
            }

            StmtKind::For {
                target,
                iter,
                body,
                orelse,
            } => {
                // Compile iterator
                let iter_reg = self.compile_expr(iter)?;
                let iterator_reg = self.builder.alloc_register();
                self.builder.emit_get_iter(iterator_reg, iter_reg);
                self.builder.free_register(iter_reg);

                let loop_start = self.builder.create_label();
                let loop_else = self.builder.create_label();
                let loop_end = self.builder.create_label();

                self.builder.bind_label(loop_start);

                // Get next item
                let item_reg = self.builder.alloc_register();
                self.builder.emit_for_iter(item_reg, loop_else);

                // Store to target
                self.compile_store(target, item_reg)?;
                self.builder.free_register(item_reg);

                // Loop body
                for s in body {
                    self.compile_stmt(s)?;
                }

                self.builder.emit_jump(loop_start);
                self.builder.bind_label(loop_else);

                // Else clause (if loop completed normally)
                for s in orelse {
                    self.compile_stmt(s)?;
                }

                self.builder.bind_label(loop_end);
                self.builder.free_register(iterator_reg);
            }

            StmtKind::FunctionDef { name, .. } => {
                // TODO: Compile nested function
                // For now, just define the name
                let name_idx = self.builder.add_name(name.clone());
                let _slot = self.builder.define_local(name.clone());

                // Placeholder: load None
                let reg = self.builder.alloc_register();
                self.builder.emit_load_none(reg);
                self.builder.emit_store_global(name_idx, reg);
                self.builder.free_register(reg);
            }

            StmtKind::AsyncFunctionDef { name, .. } => {
                let name_idx = self.builder.add_name(name.clone());
                let reg = self.builder.alloc_register();
                self.builder.emit_load_none(reg);
                self.builder.emit_store_global(name_idx, reg);
                self.builder.free_register(reg);
            }

            StmtKind::ClassDef { name, .. } => {
                // TODO: Compile class
                let name_idx = self.builder.add_name(name.clone());
                let reg = self.builder.alloc_register();
                self.builder.emit_load_none(reg);
                self.builder.emit_store_global(name_idx, reg);
                self.builder.free_register(reg);
            }

            StmtKind::Import(aliases) => {
                for alias in aliases {
                    let local_name = alias.asname.as_ref().unwrap_or(&alias.name);
                    let name_idx = self.builder.add_name(local_name.clone());

                    // TODO: Emit ImportName instruction
                    let reg = self.builder.alloc_register();
                    self.builder.emit_load_none(reg); // Placeholder
                    self.builder.emit_store_global(name_idx, reg);
                    self.builder.free_register(reg);
                }
            }

            StmtKind::ImportFrom { names, .. } => {
                for alias in names {
                    let local_name = alias.asname.as_ref().unwrap_or(&alias.name);
                    let name_idx = self.builder.add_name(local_name.clone());

                    // TODO: Emit ImportFrom instruction
                    let reg = self.builder.alloc_register();
                    self.builder.emit_load_none(reg); // Placeholder
                    self.builder.emit_store_global(name_idx, reg);
                    self.builder.free_register(reg);
                }
            }

            StmtKind::Global(_) | StmtKind::Nonlocal(_) => {
                // These are handled during scope analysis
            }

            StmtKind::Delete(targets) => {
                for target in targets {
                    match &target.kind {
                        ExprKind::Name(name) => {
                            let name_idx = self.builder.add_name(name.clone());
                            self.builder.emit(Instruction::op_di(
                                Opcode::DeleteGlobal,
                                Register::new(0),
                                name_idx,
                            ));
                        }
                        _ => {
                            // TODO: Handle delete subscript/attribute
                        }
                    }
                }
            }

            StmtKind::Assert { test, msg } => {
                let pass_label = self.builder.create_label();

                let cond_reg = self.compile_expr(test)?;
                self.builder.emit_jump_if_true(cond_reg, pass_label);
                self.builder.free_register(cond_reg);

                // Raise AssertionError
                if let Some(msg_expr) = msg {
                    let _msg_reg = self.compile_expr(msg_expr)?;
                    // TODO: Raise with message
                }
                // TODO: Actually raise AssertionError

                self.builder.bind_label(pass_label);
            }

            StmtKind::Raise { exc, cause: _ } => {
                if let Some(e) = exc {
                    let reg = self.compile_expr(e)?;
                    self.builder.emit(Instruction::op_d(Opcode::Raise, reg));
                    self.builder.free_register(reg);
                } else {
                    self.builder.emit(Instruction::op(Opcode::Reraise));
                }
            }

            // TODO: Implement remaining statements
            _ => {
                // Placeholder for unimplemented statements
            }
        }

        Ok(())
    }

    /// Compile an expression and return the register containing the result.
    fn compile_expr(&mut self, expr: &Expr) -> CompileResult<Register> {
        let reg = self.builder.alloc_register();

        match &expr.kind {
            ExprKind::Int(n) => {
                let idx = self.builder.add_int(*n);
                self.builder.emit_load_const(reg, idx);
            }

            ExprKind::Float(n) => {
                let idx = self.builder.add_float(*n);
                self.builder.emit_load_const(reg, idx);
            }

            ExprKind::Bool(b) => {
                if *b {
                    self.builder.emit_load_true(reg);
                } else {
                    self.builder.emit_load_false(reg);
                }
            }

            ExprKind::None => {
                self.builder.emit_load_none(reg);
            }

            ExprKind::String(s) => {
                // TODO: Implement proper string constant pool
                // For now, store as None placeholder and add string to names
                let _name_idx = self.builder.add_name(s.value.clone());
                self.builder.emit_load_none(reg);
            }

            ExprKind::Name(name) => {
                // Check if it's a local or global
                if let Some(slot) = self.builder.lookup_local(name) {
                    self.builder.emit_load_local(reg, slot);
                } else {
                    let name_idx = self.builder.add_name(name.clone());
                    self.builder.emit_load_global(reg, name_idx);
                }
            }

            ExprKind::BinOp { op, left, right } => {
                let left_reg = self.compile_expr(left)?;
                let right_reg = self.compile_expr(right)?;

                self.emit_binop(*op, reg, left_reg, right_reg);

                self.builder.free_register(left_reg);
                self.builder.free_register(right_reg);
            }

            ExprKind::UnaryOp { op, operand } => {
                let operand_reg = self.compile_expr(operand)?;

                match op {
                    UnaryOp::USub => self.builder.emit_neg(reg, operand_reg),
                    UnaryOp::UAdd => self.builder.emit_move(reg, operand_reg),
                    UnaryOp::Not => self.builder.emit_not(reg, operand_reg),
                    UnaryOp::Invert => self.builder.emit_bitwise_not(reg, operand_reg),
                }

                self.builder.free_register(operand_reg);
            }

            ExprKind::Compare {
                left,
                ops,
                comparators,
            } => {
                // Handle chained comparisons: a < b < c becomes (a < b) and (b < c)
                let left_reg = self.compile_expr(left)?;
                let result_reg = reg;

                if ops.len() == 1 {
                    // Simple case: single comparison
                    let right_reg = self.compile_expr(&comparators[0])?;
                    self.emit_cmpop(ops[0], result_reg, left_reg, right_reg);
                    self.builder.free_register(right_reg);
                } else {
                    // Chained comparisons
                    self.builder.emit_load_true(result_reg);
                    let mut prev_reg = left_reg;

                    for (op, comp) in ops.iter().zip(comparators.iter()) {
                        let next_reg = self.compile_expr(comp)?;
                        let cmp_reg = self.builder.alloc_register();

                        self.emit_cmpop(*op, cmp_reg, prev_reg, next_reg);

                        // result = result and cmp
                        self.builder
                            .emit_bitwise_and(result_reg, result_reg, cmp_reg);

                        self.builder.free_register(cmp_reg);
                        if prev_reg != left_reg {
                            self.builder.free_register(prev_reg);
                        }
                        prev_reg = next_reg;
                    }

                    self.builder.free_register(prev_reg);
                }

                self.builder.free_register(left_reg);
            }

            ExprKind::BoolOp { op, values } => {
                // Short-circuit evaluation
                let end_label = self.builder.create_label();

                let first_reg = self.compile_expr(&values[0])?;
                self.builder.emit_move(reg, first_reg);
                self.builder.free_register(first_reg);

                for value in &values[1..] {
                    match op {
                        BoolOp::And => {
                            self.builder.emit_jump_if_false(reg, end_label);
                        }
                        BoolOp::Or => {
                            self.builder.emit_jump_if_true(reg, end_label);
                        }
                    }

                    let next_reg = self.compile_expr(value)?;
                    self.builder.emit_move(reg, next_reg);
                    self.builder.free_register(next_reg);
                }

                self.builder.bind_label(end_label);
            }

            ExprKind::Call {
                func,
                args,
                keywords,
            } => {
                let func_reg = self.compile_expr(func)?;

                // Compile arguments into consecutive registers
                let first_arg = if !args.is_empty() {
                    self.builder.alloc_register()
                } else {
                    Register::new(0)
                };

                let mut arg_regs = Vec::with_capacity(args.len());
                for (i, arg) in args.iter().enumerate() {
                    let arg_reg = if i == 0 {
                        first_arg
                    } else {
                        self.builder.alloc_register()
                    };
                    let temp = self.compile_expr(arg)?;
                    self.builder.emit_move(arg_reg, temp);
                    self.builder.free_register(temp);
                    arg_regs.push(arg_reg);
                }

                // TODO: Handle keyword arguments
                if !keywords.is_empty() {
                    // For now, ignore keywords
                }

                // Emit call
                self.builder.emit_call(reg, func_reg, args.len() as u8);

                // Free argument registers
                for arg_reg in arg_regs {
                    self.builder.free_register(arg_reg);
                }
                self.builder.free_register(func_reg);
            }

            ExprKind::Attribute { value, attr, .. } => {
                let obj_reg = self.compile_expr(value)?;
                let name_idx = self.builder.add_name(attr.clone());
                self.builder.emit_get_attr(reg, obj_reg, name_idx);
                self.builder.free_register(obj_reg);
            }

            ExprKind::Subscript { value, slice, .. } => {
                let obj_reg = self.compile_expr(value)?;
                let key_reg = self.compile_expr(slice)?;
                self.builder.emit_get_item(reg, obj_reg, key_reg);
                self.builder.free_register(obj_reg);
                self.builder.free_register(key_reg);
            }

            ExprKind::List(elts) => {
                let first_elem = self.builder.alloc_register();
                let mut elem_regs = Vec::with_capacity(elts.len());

                for (i, elt) in elts.iter().enumerate() {
                    let elem_reg = if i == 0 {
                        first_elem
                    } else {
                        self.builder.alloc_register()
                    };
                    let temp = self.compile_expr(elt)?;
                    self.builder.emit_move(elem_reg, temp);
                    self.builder.free_register(temp);
                    elem_regs.push(elem_reg);
                }

                self.builder
                    .emit_build_list(reg, first_elem, elts.len() as u8);

                for elem_reg in elem_regs {
                    self.builder.free_register(elem_reg);
                }
            }

            ExprKind::Tuple(elts) => {
                let first_elem = self.builder.alloc_register();
                let mut elem_regs = Vec::with_capacity(elts.len());

                for (i, elt) in elts.iter().enumerate() {
                    let elem_reg = if i == 0 {
                        first_elem
                    } else {
                        self.builder.alloc_register()
                    };
                    let temp = self.compile_expr(elt)?;
                    self.builder.emit_move(elem_reg, temp);
                    self.builder.free_register(temp);
                    elem_regs.push(elem_reg);
                }

                self.builder
                    .emit_build_tuple(reg, first_elem, elts.len() as u8);

                for elem_reg in elem_regs {
                    self.builder.free_register(elem_reg);
                }
            }

            ExprKind::IfExp { test, body, orelse } => {
                let else_label = self.builder.create_label();
                let end_label = self.builder.create_label();

                let cond_reg = self.compile_expr(test)?;
                self.builder.emit_jump_if_false(cond_reg, else_label);
                self.builder.free_register(cond_reg);

                let body_reg = self.compile_expr(body)?;
                self.builder.emit_move(reg, body_reg);
                self.builder.free_register(body_reg);
                self.builder.emit_jump(end_label);

                self.builder.bind_label(else_label);
                let else_reg = self.compile_expr(orelse)?;
                self.builder.emit_move(reg, else_reg);
                self.builder.free_register(else_reg);

                self.builder.bind_label(end_label);
            }

            // TODO: Implement remaining expressions
            _ => {
                // Placeholder: load None for unimplemented expressions
                self.builder.emit_load_none(reg);
            }
        }

        Ok(reg)
    }

    /// Compile a store to a target.
    fn compile_store(&mut self, target: &Expr, value: Register) -> CompileResult<()> {
        match &target.kind {
            ExprKind::Name(name) => {
                // Check if local or global
                if let Some(slot) = self.builder.lookup_local(name) {
                    self.builder.emit_store_local(slot, value);
                } else {
                    // For module-level, store as global
                    let name_idx = self.builder.add_name(name.clone());
                    self.builder.emit_store_global(name_idx, value);
                    // Also define as local for future lookups
                    self.builder.define_local(name.clone());
                }
            }

            ExprKind::Attribute {
                value: obj, attr, ..
            } => {
                let obj_reg = self.compile_expr(obj)?;
                let _name_idx = self.builder.add_name(attr.clone());
                // TODO: Emit SetAttr
                self.builder.free_register(obj_reg);
            }

            ExprKind::Subscript {
                value: obj, slice, ..
            } => {
                let obj_reg = self.compile_expr(obj)?;
                let key_reg = self.compile_expr(slice)?;
                self.builder.emit_set_item(obj_reg, key_reg, value);
                self.builder.free_register(obj_reg);
                self.builder.free_register(key_reg);
            }

            ExprKind::Tuple(elts) | ExprKind::List(elts) => {
                // Unpack assignment
                for (i, elt) in elts.iter().enumerate() {
                    let item_reg = self.builder.alloc_register();
                    let idx = self.builder.add_int(i as i64);
                    let idx_reg = self.builder.alloc_register();
                    self.builder.emit_load_const(idx_reg, idx);
                    self.builder.emit_get_item(item_reg, value, idx_reg);
                    self.compile_store(elt, item_reg)?;
                    self.builder.free_register(item_reg);
                    self.builder.free_register(idx_reg);
                }
            }

            _ => {
                return Err(CompileError {
                    message: format!("cannot assign to {:?}", target.kind),
                    line: target.span.start,
                    column: 0,
                });
            }
        }

        Ok(())
    }

    /// Emit a binary operation.
    fn emit_binop(&mut self, op: BinOp, dst: Register, left: Register, right: Register) {
        match op {
            BinOp::Add => self.builder.emit_add(dst, left, right),
            BinOp::Sub => self.builder.emit_sub(dst, left, right),
            BinOp::Mult => self.builder.emit_mul(dst, left, right),
            BinOp::Div => self.builder.emit_div(dst, left, right),
            BinOp::FloorDiv => self.builder.emit_floor_div(dst, left, right),
            BinOp::Mod => self.builder.emit_mod(dst, left, right),
            BinOp::Pow => self.builder.emit_pow(dst, left, right),
            BinOp::LShift => self.builder.emit_shl(dst, left, right),
            BinOp::RShift => self.builder.emit_shr(dst, left, right),
            BinOp::BitAnd => self.builder.emit_bitwise_and(dst, left, right),
            BinOp::BitOr => self.builder.emit_bitwise_or(dst, left, right),
            BinOp::BitXor => self.builder.emit_bitwise_xor(dst, left, right),
            BinOp::MatMult => {
                // Matrix multiplication - use generic multiply for now
                self.builder.emit_mul(dst, left, right)
            }
        }
    }

    /// Emit an augmented assignment operation.
    fn emit_augop(&mut self, op: AugOp, dst: Register, left: Register, right: Register) {
        match op {
            AugOp::Add => self.builder.emit_add(dst, left, right),
            AugOp::Sub => self.builder.emit_sub(dst, left, right),
            AugOp::Mult => self.builder.emit_mul(dst, left, right),
            AugOp::Div => self.builder.emit_div(dst, left, right),
            AugOp::FloorDiv => self.builder.emit_floor_div(dst, left, right),
            AugOp::Mod => self.builder.emit_mod(dst, left, right),
            AugOp::Pow => self.builder.emit_pow(dst, left, right),
            AugOp::LShift => self.builder.emit_shl(dst, left, right),
            AugOp::RShift => self.builder.emit_shr(dst, left, right),
            AugOp::BitAnd => self.builder.emit_bitwise_and(dst, left, right),
            AugOp::BitOr => self.builder.emit_bitwise_or(dst, left, right),
            AugOp::BitXor => self.builder.emit_bitwise_xor(dst, left, right),
            AugOp::MatMult => self.builder.emit_mul(dst, left, right),
        }
    }

    /// Emit a comparison operation.
    fn emit_cmpop(&mut self, op: CmpOp, dst: Register, left: Register, right: Register) {
        match op {
            CmpOp::Lt => self.builder.emit_lt(dst, left, right),
            CmpOp::LtE => self.builder.emit_le(dst, left, right),
            CmpOp::Eq => self.builder.emit_eq(dst, left, right),
            CmpOp::NotEq => self.builder.emit_ne(dst, left, right),
            CmpOp::Gt => self.builder.emit_gt(dst, left, right),
            CmpOp::GtE => self.builder.emit_ge(dst, left, right),
            CmpOp::Is => self
                .builder
                .emit(Instruction::op_dss(Opcode::Is, dst, left, right)),
            CmpOp::IsNot => self
                .builder
                .emit(Instruction::op_dss(Opcode::IsNot, dst, left, right)),
            CmpOp::In => self
                .builder
                .emit(Instruction::op_dss(Opcode::In, dst, left, right)),
            CmpOp::NotIn => self
                .builder
                .emit(Instruction::op_dss(Opcode::NotIn, dst, left, right)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn compile(source: &str) -> CodeObject {
        let module = prism_parser::parse(source).expect("parse error");
        Compiler::compile_module(&module, "<test>").expect("compile error")
    }

    #[test]
    fn test_compile_simple_expr() {
        let code = compile("1 + 2");
        assert!(!code.instructions.is_empty());
    }

    #[test]
    fn test_compile_assignment() {
        let code = compile("x = 42");
        assert!(!code.instructions.is_empty());
    }

    #[test]
    fn test_compile_if() {
        let code = compile("if True:\n    pass");
        assert!(!code.instructions.is_empty());
    }

    #[test]
    fn test_compile_while() {
        let code = compile("x = 0\nwhile x < 10:\n    x = x + 1");
        assert!(!code.instructions.is_empty());
    }

    #[test]
    fn test_compile_function_call() {
        let code = compile("print(42)");
        assert!(!code.instructions.is_empty());
    }

    #[test]
    fn test_register_count() {
        let code = compile("a = 1\nb = 2\nc = a + b");
        // Should use some registers
        assert!(code.register_count > 0);
    }
}
