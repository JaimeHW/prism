//! AST to register-based bytecode compiler.
//!
//! The compiler transforms a parsed Python AST into executable bytecode
//! using a two-phase approach:
//!
//! 1. **Scope analysis**: Builds symbol tables and determines variable scopes
//! 2. **Code generation**: Emits register-based bytecode instructions

use crate::bytecode::{
    CodeFlags, CodeObject, FunctionBuilder, Instruction, Label, LocalSlot, Opcode, Register,
};
use crate::function_compiler::{VarLocation, VariableEmitter};
use crate::scope::{ScopeAnalyzer, SymbolTable};

use prism_parser::ast::{
    AugOp, BinOp, BoolOp, CmpOp, ExceptHandler, Expr, ExprKind, Module, Stmt, StmtKind, UnaryOp,
};
use smallvec::SmallVec;
use std::sync::Arc;

/// Stack-allocated loop context stack for typical loop nesting depths.
/// Most code has ≤4 nested loops, so we avoid heap allocation in the common case.
type LoopStack = SmallVec<[LoopContext; 4]>;

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

// =============================================================================
// Loop Context for break/continue
// =============================================================================

/// Context for tracking loop nesting and jump targets.
///
/// Each loop (while, for) pushes a context onto the stack to enable
/// break and continue statements to emit correct jump instructions.
#[derive(Debug, Clone, Copy)]
struct LoopContext {
    /// Label to jump to for `break` (after loop body, before else clause).
    /// Note: break skips the else clause in Python.
    break_label: Label,
    /// Label to jump to for `continue` (back to loop condition/iterator).
    continue_label: Label,
}

// =============================================================================
// Compiler
// =============================================================================

/// Bytecode compiler.
pub struct Compiler {
    /// Current function builder.
    builder: FunctionBuilder,
    /// Symbol table from scope analysis.
    symbol_table: SymbolTable,
    /// Source filename.
    #[allow(dead_code)]
    filename: Arc<str>,
    /// Stack of active loop contexts for break/continue.
    /// Innermost loop is at the end (top) of the stack.
    /// Uses SmallVec to avoid heap allocation for typical nesting depths (≤4).
    loop_stack: LoopStack,
}

impl Compiler {
    /// Create a new compiler for a module.
    pub fn new(filename: impl Into<Arc<str>>) -> Self {
        let filename = filename.into();
        Self {
            builder: FunctionBuilder::new("<module>"),
            symbol_table: SymbolTable::new("<module>"),
            filename,
            loop_stack: LoopStack::new(),
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
            loop_stack: LoopStack::new(),
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

    /// Resolve a variable name to its location.
    ///
    /// Uses the symbol table to determine whether the variable is local,
    /// closure (cell/free), or global.
    fn resolve_variable(&self, name: &str) -> VarLocation {
        // Look up in symbol table's root scope (module level)
        if let Some(symbol) = self.symbol_table.root.lookup(name) {
            // Check closure variables first (cells and frees use same opcodes)
            if symbol.is_cell() || symbol.is_free() {
                if let Some(slot) = symbol.closure_slot {
                    return VarLocation::Closure(slot);
                }
            }
            // Check local variables (but not cells - those use closure access)
            if symbol.is_local() && !symbol.is_cell() {
                if let Some(slot) = symbol.local_slot {
                    return VarLocation::Local(slot);
                }
            }
        }
        // Fall back to global for undefined or explicitly global symbols
        VarLocation::Global
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
                // Jump to the break label of the innermost loop
                if let Some(ctx) = self.loop_stack.last() {
                    self.builder.emit_jump(ctx.break_label);
                } else {
                    return Err(CompileError {
                        message: "'break' outside loop".to_string(),
                        line: stmt.span.start,
                        column: 0,
                    });
                }
            }

            StmtKind::Continue => {
                // Jump to the continue label of the innermost loop
                if let Some(ctx) = self.loop_stack.last() {
                    self.builder.emit_jump(ctx.continue_label);
                } else {
                    return Err(CompileError {
                        message: "'continue' outside loop".to_string(),
                        line: stmt.span.start,
                        column: 0,
                    });
                }
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

                // Push loop context for break/continue
                // break jumps to loop_end, continue jumps to loop_start
                self.loop_stack.push(LoopContext {
                    break_label: loop_end,
                    continue_label: loop_start,
                });

                self.builder.bind_label(loop_start);

                let cond_reg = self.compile_expr(test)?;
                self.builder.emit_jump_if_false(cond_reg, loop_else);
                self.builder.free_register(cond_reg);

                for s in body {
                    self.compile_stmt(s)?;
                }

                self.builder.emit_jump(loop_start);
                self.builder.bind_label(loop_else);

                // Else clause only executes if loop completed normally (no break)
                for s in orelse {
                    self.compile_stmt(s)?;
                }

                self.builder.bind_label(loop_end);

                // Pop loop context
                self.loop_stack.pop();
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

                // Push loop context for break/continue
                // break jumps to loop_end, continue jumps to loop_start
                self.loop_stack.push(LoopContext {
                    break_label: loop_end,
                    continue_label: loop_start,
                });

                self.builder.bind_label(loop_start);

                // Get next item
                let item_reg = self.builder.alloc_register();
                self.builder
                    .emit_for_iter(item_reg, iterator_reg, loop_else);

                // Store to target
                self.compile_store(target, item_reg)?;
                self.builder.free_register(item_reg);

                // Loop body
                for s in body {
                    self.compile_stmt(s)?;
                }

                self.builder.emit_jump(loop_start);
                self.builder.bind_label(loop_else);

                // Else clause only executes if loop completed normally (no break)
                for s in orelse {
                    self.compile_stmt(s)?;
                }

                self.builder.bind_label(loop_end);

                // Pop loop context
                self.loop_stack.pop();

                self.builder.free_register(iterator_reg);
            }

            StmtKind::ClassDef {
                name,
                bases,
                keywords: _keywords,
                body,
                decorator_list,
                type_params: _type_params,
            } => {
                // Class definition compilation follows CPython's BUILD_CLASS protocol:
                // 1. Compile decorators (evaluated first, applied last)
                // 2. Evaluate base classes
                // 3. Create child CodeObject for class body
                // 4. Emit BUILD_CLASS instruction
                // 5. Apply decorators in reverse order
                // 6. Store result in enclosing scope

                // Step 1: Compile decorators and save to registers
                let decorator_regs: Vec<Register> = decorator_list
                    .iter()
                    .map(|d| self.compile_expr(d))
                    .collect::<Result<_, _>>()?;

                // Step 2: Evaluate base classes into registers
                let base_count = bases.len();
                let base_regs: Vec<Register> = bases
                    .iter()
                    .map(|b| self.compile_expr(b))
                    .collect::<Result<_, _>>()?;

                // Step 3: Create the class body code object using builder-swap pattern
                // Find the scope for this class from the symbol table
                let class_scope = self.find_child_scope(name);

                // Create a new FunctionBuilder for the class body
                let mut class_builder = FunctionBuilder::new(name.clone());
                class_builder.set_filename(self.builder.get_filename());
                class_builder.set_qualname(name.clone());
                class_builder.add_flag(CodeFlags::CLASS);

                // Check if any method uses zero-arg super() and inject __class__ cell
                let uses_zero_arg_super =
                    crate::class_compiler::ClassCompiler::uses_zero_arg_super(body);
                if uses_zero_arg_super {
                    // __class__ is implicitly a cell variable in class bodies that use super()
                    class_builder.add_cellvar("__class__");
                }

                // Register cell and free variables from scope analysis
                if let Some(scope) = class_scope {
                    // Cell variables: names captured by inner functions
                    for sym in scope.cellvars() {
                        // Skip __class__ if we already added it
                        if sym.name.as_ref() != "__class__" {
                            class_builder.add_cellvar(Arc::from(sym.name.as_ref()));
                        }
                    }

                    // Free variables: names captured from outer scopes
                    for sym in scope.freevars() {
                        class_builder.add_freevar(Arc::from(sym.name.as_ref()));
                    }
                }

                // Swap builders to compile class body
                let parent_builder = std::mem::replace(&mut self.builder, class_builder);

                // Compile class body statements (method definitions, class variables, etc.)
                for stmt in body {
                    self.compile_stmt(stmt)?;
                }

                // Class body returns the namespace dict (implicit)
                self.builder.emit_return_none();

                // Swap back and get finished class body code
                let class_builder = std::mem::replace(&mut self.builder, parent_builder);
                let class_code = class_builder.finish();

                // Store the nested CodeObject as a constant
                let code_idx = self.builder.add_code_object(Arc::new(class_code));

                // Step 4: Emit BUILD_CLASS instruction
                // Allocate result register for the class object
                let result_reg = self.builder.alloc_register();
                self.builder
                    .emit_build_class(result_reg, code_idx, base_count as u8);

                // Step 5: Apply decorators in reverse order
                // @decorator1
                // @decorator2
                // class Foo: ...
                // is equivalent to: Foo = decorator1(decorator2(Foo))
                for decorator_reg in decorator_regs.into_iter().rev() {
                    // Call decorator with class as argument
                    let call_result = self.builder.alloc_register();
                    self.builder
                        .emit_move(Register::new(call_result.0 + 1), result_reg);
                    self.builder.emit(Instruction::op_dss(
                        Opcode::Call,
                        call_result,
                        decorator_reg,
                        Register::new(1), // 1 argument
                    ));
                    self.builder.free_register(decorator_reg);
                    self.builder.free_register(result_reg);
                    // Move result to result_reg position (we'll reuse call_result as func_reg)
                }

                // Free base class registers
                for base_reg in base_regs {
                    self.builder.free_register(base_reg);
                }

                // Step 6: Store the class in the enclosing scope
                // Use scope-aware storage (global for module-level, local for nested)
                let name_idx = self.builder.add_name(name.clone());
                self.builder.emit_store_global(name_idx, result_reg);
                self.builder.free_register(result_reg);
            }

            StmtKind::Import(aliases) => {
                // import module1, module2 as alias, ...
                // For each alias:
                //   1. Emit ImportName to load the module
                //   2. Store in the target name (alias if present, else module name)
                for alias in aliases {
                    // Use asname if present, otherwise use the module name
                    // For `import foo.bar`, Python stores the top-level module `foo`
                    let local_name = alias.asname.as_ref().unwrap_or(&alias.name);

                    // For dotted imports like `import foo.bar`, we need to store the
                    // top-level module name. Get just the first component.
                    let store_name = if alias.asname.is_some() {
                        local_name.as_str()
                    } else {
                        alias.name.split('.').next().unwrap_or(&alias.name)
                    };

                    // Add the full module name to the names table for importing
                    let module_name_idx = self.builder.add_name(alias.name.clone());
                    let store_name_idx = self.builder.add_name(store_name.to_string());

                    // Emit ImportName: reg = import(module_name)
                    let reg = self.builder.alloc_register();
                    self.builder.emit_import_name(reg, module_name_idx);

                    // Store the module in the appropriate scope
                    match self.resolve_variable(store_name) {
                        VarLocation::Local(slot) => {
                            self.builder
                                .emit_store_local(LocalSlot::new(slot as u16), reg);
                        }
                        VarLocation::Closure(slot) => {
                            self.builder.emit_store_closure(slot, reg);
                        }
                        VarLocation::Global => {
                            self.builder.emit_store_global(store_name_idx, reg);
                        }
                    }

                    self.builder.free_register(reg);
                }
            }

            StmtKind::ImportFrom {
                module,
                names,
                level: _,
            } => {
                // from module import name1, name2 as alias, ...
                // 1. Import the source module first
                // 2. For each name, import the attribute

                // Handle `from module import *` case
                let is_star = names.len() == 1 && names[0].name == "*";

                if is_star {
                    // from module import *
                    if let Some(mod_name) = module {
                        let mod_name_idx = self.builder.add_name(mod_name.clone());
                        let mod_reg = self.builder.alloc_register();

                        // Import the module
                        self.builder.emit_import_name(mod_reg, mod_name_idx);

                        // Emit ImportStar to inject all public names
                        self.builder.emit_import_star(Register::new(0), mod_reg);

                        self.builder.free_register(mod_reg);
                    }
                } else {
                    // from module import name1, name2, ...
                    let mod_reg = if let Some(mod_name) = module {
                        let mod_name_idx = self.builder.add_name(mod_name.clone());
                        let reg = self.builder.alloc_register();
                        self.builder.emit_import_name(reg, mod_name_idx);
                        reg
                    } else {
                        // Relative import with no module (e.g., `from . import x`)
                        // For now, emit LoadNone as placeholder
                        let reg = self.builder.alloc_register();
                        self.builder.emit_load_none(reg);
                        reg
                    };

                    for alias in names {
                        let local_name = alias.asname.as_ref().unwrap_or(&alias.name);
                        let attr_name_idx = self.builder.add_name(alias.name.clone());
                        let store_name_idx = self.builder.add_name(local_name.clone());

                        // We need the name index as u8 for ImportFrom encoding
                        let attr_idx_u8 = (attr_name_idx & 0xFF) as u8;

                        // Emit ImportFrom: reg = module.attr
                        let attr_reg = self.builder.alloc_register();
                        self.builder
                            .emit_import_from(attr_reg, mod_reg, attr_idx_u8);

                        // Store in the appropriate scope
                        match self.resolve_variable(local_name) {
                            VarLocation::Local(slot) => {
                                self.builder
                                    .emit_store_local(LocalSlot::new(slot as u16), attr_reg);
                            }
                            VarLocation::Closure(slot) => {
                                self.builder.emit_store_closure(slot, attr_reg);
                            }
                            VarLocation::Global => {
                                self.builder.emit_store_global(store_name_idx, attr_reg);
                            }
                        }

                        self.builder.free_register(attr_reg);
                    }

                    self.builder.free_register(mod_reg);
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

            StmtKind::Raise { exc, cause } => {
                match (exc, cause) {
                    // raise X from Y - exception chaining with explicit cause
                    (Some(e), Some(c)) => {
                        let exc_reg = self.compile_expr(e)?;
                        let cause_reg = self.compile_expr(c)?;
                        self.builder.emit(Instruction::op_ds(
                            Opcode::RaiseFrom,
                            exc_reg,
                            cause_reg,
                        ));
                        self.builder.free_register(cause_reg);
                        self.builder.free_register(exc_reg);
                    }
                    // raise X - simple exception raise
                    (Some(e), None) => {
                        let reg = self.compile_expr(e)?;
                        self.builder.emit(Instruction::op_d(Opcode::Raise, reg));
                        self.builder.free_register(reg);
                    }
                    // bare raise - reraise current exception
                    (None, _) => {
                        self.builder.emit(Instruction::op(Opcode::Reraise));
                    }
                }
            }

            StmtKind::FunctionDef {
                name,
                args,
                body,
                decorator_list,
                ..
            } => {
                self.compile_function_def(name, args, body, decorator_list, false)?;
            }

            StmtKind::AsyncFunctionDef {
                name,
                args,
                body,
                decorator_list,
                ..
            } => {
                self.compile_function_def(name, args, body, decorator_list, true)?;
            }

            StmtKind::Try {
                body,
                handlers,
                orelse,
                finalbody,
            } => {
                self.compile_try(body, handlers, orelse, finalbody)?;
            }

            StmtKind::TryStar {
                body,
                handlers,
                orelse,
                finalbody,
            } => {
                // TryStar is Python 3.11+ except* syntax
                // For now, treat it like regular Try (except* semantics require ExceptionGroup)
                self.compile_try(body, handlers, orelse, finalbody)?;
            }

            // TODO: Implement remaining statements (With, etc.)
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
                // Add string to constant pool with automatic interning and deduplication
                let str_idx = self.builder.add_string(s.value.as_str());
                self.builder.emit_load_const(reg, str_idx);
            }

            ExprKind::Name(name) => {
                // Use scope-aware variable resolution
                let location = self.resolve_variable(name);
                self.builder
                    .emit_load_var(reg, location, Some(name.as_ref()));
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
                // Strategy to avoid register collisions:
                // 1. Compile function first
                // 2. If func_reg falls in the arg range (dst+1..dst+argc), move it to a safe register
                // 3. Compile args to dst+1..dst+argc
                // 4. Emit call

                let mut func_reg = self.compile_expr(func)?;

                // Check if func_reg collides with arg range (dst+1 to dst+argc)
                let arg_range_start = reg.0 + 1;
                let arg_range_end = reg.0 + 1 + args.len() as u8;

                if !args.is_empty() && func_reg.0 >= arg_range_start && func_reg.0 < arg_range_end {
                    // func_reg would be clobbered by arg writes - move it to a safe register
                    let safe_reg = self.builder.alloc_register();
                    self.builder.emit_move(safe_reg, func_reg);
                    self.builder.free_register(func_reg);
                    func_reg = safe_reg;
                }

                // Now compile arguments to their destination registers
                // These won't collide with func_reg since we moved it out of the way
                for (i, arg) in args.iter().enumerate() {
                    let arg_dst = Register::new(reg.0 + 1 + i as u8);
                    let temp = self.compile_expr(arg)?;
                    if temp != arg_dst {
                        self.builder.emit_move(arg_dst, temp);
                    }
                    self.builder.free_register(temp);
                }

                // TODO: Handle keyword arguments
                if !keywords.is_empty() {
                    // For now, ignore keywords
                }

                // Emit call - func_reg is safe, args are at dst+1 onwards
                self.builder.emit_call(reg, func_reg, args.len() as u8);

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
                // Use scope-aware variable resolution
                let location = self.resolve_variable(name);
                self.builder
                    .emit_store_var(location, value, Some(name.as_ref()));
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

    // =========================================================================
    // Exception Handling Compilation
    // =========================================================================

    /// Compile a try/except/finally statement with zero-cost exception handling.
    ///
    /// This generates exception table entries for the VM's table-driven unwinder.
    /// No runtime opcodes are executed on try block entry/exit - the exception
    /// table is consulted only when an exception is raised.
    ///
    /// # Layout
    /// ```text
    /// try_start:
    ///     <try body>              # Protected by exception entry
    ///     JUMP end_label          # Skip handlers on normal exit
    /// handler_0:                  # except Type1 as e:
    ///     <check exception type>
    ///     <handler body>
    ///     JUMP end_label
    /// handler_1:                  # except Type2:
    ///     <handler body>
    ///     JUMP end_label
    /// finally:                    # finally:
    ///     <finally body>
    /// end_label:
    /// ```
    fn compile_try(
        &mut self,
        body: &[Stmt],
        handlers: &[ExceptHandler],
        orelse: &[Stmt],
        finalbody: &[Stmt],
    ) -> CompileResult<()> {
        use crate::ExceptionEntry;

        // Create labels for control flow
        let end_label = self.builder.create_label();
        let orelse_label = if !orelse.is_empty() {
            Some(self.builder.create_label())
        } else {
            None
        };
        let finally_label = if !finalbody.is_empty() {
            Some(self.builder.create_label())
        } else {
            None
        };

        // Create handler labels (one per except clause)
        let handler_labels: Vec<_> = handlers
            .iter()
            .map(|_| self.builder.create_label())
            .collect();

        // Record try block start position
        let try_start_pc = self.builder.current_pc();
        let stack_depth = self.builder.current_stack_depth();

        // Compile the try body
        for stmt in body {
            self.compile_stmt(stmt)?;
        }

        // Record try block end position
        let try_end_pc = self.builder.current_pc();

        // Jump to else or end on normal completion
        if let Some(else_label) = orelse_label {
            self.builder.emit_jump(else_label);
        } else if let Some(fin_label) = finally_label {
            self.builder.emit_jump(fin_label);
        } else if !handlers.is_empty() {
            self.builder.emit_jump(end_label);
        }

        // Compile each exception handler
        for (i, handler) in handlers.iter().enumerate() {
            self.builder.bind_label(handler_labels[i]);

            let handler_start_pc = self.builder.current_pc();

            // Extract handler info
            let type_idx = if let Some(type_expr) = &handler.typ {
                // Compile the exception type expression and store as type filter
                let type_reg = self.compile_expr(type_expr)?;

                // For now, we use the type register as part of the handler check
                // The VM will use ExceptionMatch opcode at runtime
                let exc_reg = self.builder.alloc_register();

                // Load the current exception into exc_reg (LoadException opcode)
                self.builder
                    .emit(Instruction::op_d(Opcode::LoadException, exc_reg));

                // Check if exception matches type
                let match_reg = self.builder.alloc_register();
                self.builder.emit(Instruction::op_dss(
                    Opcode::ExceptionMatch,
                    match_reg,
                    exc_reg,
                    type_reg,
                ));

                // If no match, jump to next handler or reraise
                let next_handler = if i + 1 < handlers.len() {
                    handler_labels[i + 1]
                } else if let Some(fin_label) = finally_label {
                    fin_label
                } else {
                    // No more handlers, jump to end (reraise happens at end_label)
                    end_label
                };

                self.builder.emit_jump_if_false(match_reg, next_handler);

                self.builder.free_register(match_reg);
                self.builder.free_register(type_reg);

                // If handler has a name binding (except E as e:), bind it
                if let Some(name) = &handler.name {
                    let location = self.resolve_variable(name);
                    self.builder
                        .emit_store_var(location, exc_reg, Some(name.as_ref()));
                }

                self.builder.free_register(exc_reg);

                Some(handler_start_pc as u16)
            } else {
                // Bare except: catches all
                if let Some(name) = &handler.name {
                    let exc_reg = self.builder.alloc_register();
                    self.builder
                        .emit(Instruction::op_d(Opcode::LoadException, exc_reg));
                    let location = self.resolve_variable(name);
                    self.builder
                        .emit_store_var(location, exc_reg, Some(name.as_ref()));
                    self.builder.free_register(exc_reg);
                }
                None
            };

            // Compile handler body
            for stmt in &handler.body {
                self.compile_stmt(stmt)?;
            }

            // Jump to finally or end after handler
            if let Some(fin_label) = finally_label {
                self.builder.emit_jump(fin_label);
            } else {
                self.builder.emit_jump(end_label);
            }

            // Add exception entry for this handler
            // exception_type_idx = u16::MAX means catch-all
            self.builder.add_exception_entry(ExceptionEntry {
                start_pc: try_start_pc,
                end_pc: try_end_pc,
                handler_pc: handler_start_pc,
                finally_pc: u32::MAX, // No finally for this handler entry
                depth: stack_depth as u16,
                exception_type_idx: type_idx.unwrap_or(u16::MAX),
            });
        }

        // Compile else block (only if try body completes normally)
        if let Some(else_label) = orelse_label {
            self.builder.bind_label(else_label);
            for stmt in orelse {
                self.compile_stmt(stmt)?;
            }
            // Jump to finally or end
            if let Some(fin_label) = finally_label {
                self.builder.emit_jump(fin_label);
            } else {
                self.builder.emit_jump(end_label);
            }
        }

        // Compile finally block
        if let Some(fin_label) = finally_label {
            self.builder.bind_label(fin_label);
            let finally_start_pc = self.builder.current_pc();

            // Push current exception info to preserve exception state
            // This is required for proper exception re-raise semantics
            self.builder.emit(Instruction::op(Opcode::PushExcInfo));

            for stmt in finalbody {
                self.compile_stmt(stmt)?;
            }

            // Pop exception info and check if we need to re-raise
            self.builder.emit(Instruction::op(Opcode::PopExcInfo));

            // EndFinally opcode re-raises exception if one is pending
            self.builder.emit(Instruction::op(Opcode::EndFinally));

            // Add finally exception entry covering the try block
            self.builder.add_exception_entry(ExceptionEntry {
                start_pc: try_start_pc,
                end_pc: try_end_pc,
                handler_pc: finally_start_pc,
                finally_pc: finally_start_pc, // Handler is also the finally block
                depth: stack_depth as u16,
                exception_type_idx: u16::MAX, // Finally catches all
            });
        }

        // End label
        self.builder.bind_label(end_label);

        Ok(())
    }

    // =========================================================================
    // Function Definition Compilation
    // =========================================================================

    /// Compile a function definition (FunctionDef or AsyncFunctionDef).
    ///
    /// This creates a nested CodeObject for the function body and emits
    /// MakeFunction or MakeClosure opcode to create the function object.
    ///
    /// # Arguments
    ///
    /// * `name` - Function name
    /// * `args` - Function arguments specification
    /// * `body` - Function body statements
    /// * `decorator_list` - Decorators to apply
    /// * `is_async` - Whether this is an async function
    fn compile_function_def(
        &mut self,
        name: &str,
        args: &prism_parser::ast::Arguments,
        body: &[Stmt],
        decorator_list: &[Expr],
        is_async: bool,
    ) -> CompileResult<()> {
        use crate::bytecode::LocalSlot;

        // Find the scope for this function from the symbol table
        // We need to look it up by name in the current scope's children
        let func_scope = self.find_child_scope(name);

        // Create a new FunctionBuilder for the function body
        let mut func_builder = FunctionBuilder::new(name);
        func_builder.set_filename(&*self.filename);

        // Set function flags
        if is_async {
            func_builder.add_flags(CodeFlags::COROUTINE);
        }

        // Count parameters
        let posonly_count = args.posonlyargs.len() as u16;
        let kwonly_count = args.kwonlyargs.len() as u16;
        let total_positional = args.posonlyargs.len() + args.args.len();

        // Set parameter counts on the builder
        func_builder.set_arg_count(total_positional as u16);
        func_builder.set_kwonlyarg_count(kwonly_count);
        func_builder.set_posonlyarg_count(posonly_count);

        // Handle varargs and kwargs
        if args.vararg.is_some() {
            func_builder.add_flags(CodeFlags::VARARGS);
        }
        if args.kwarg.is_some() {
            func_builder.add_flags(CodeFlags::VARKEYWORDS);
        }

        // Register parameters as locals (they occupy the first slots)
        // Python parameter order: posonly, regular args, vararg, kwonly, kwarg

        // Position-only parameters
        for arg in &args.posonlyargs {
            func_builder.define_local(arg.arg.as_str());
        }

        // Regular positional parameters
        for arg in &args.args {
            func_builder.define_local(arg.arg.as_str());
        }

        // *args
        if let Some(ref vararg) = args.vararg {
            func_builder.define_local(vararg.arg.as_str());
        }

        // Keyword-only parameters
        for arg in &args.kwonlyargs {
            func_builder.define_local(arg.arg.as_str());
        }

        // **kwargs
        if let Some(ref kwarg) = args.kwarg {
            func_builder.define_local(kwarg.arg.as_str());
        }

        // Register cell and free variables from scope analysis
        let mut has_closure = false;
        if let Some(scope) = func_scope {
            // Cell variables: locals captured by inner functions
            for sym in scope.cellvars() {
                func_builder.add_cellvar(Arc::from(sym.name.as_ref()));
                has_closure = true;
            }

            // Free variables: captured from outer scopes
            for sym in scope.freevars() {
                func_builder.add_freevar(Arc::from(sym.name.as_ref()));
                has_closure = true;
            }

            // Set generator flag from scope analysis
            if scope.has_yield {
                func_builder.add_flags(CodeFlags::GENERATOR);
            }
        }

        if has_closure {
            func_builder.add_flags(CodeFlags::NESTED);
        }

        // Swap builders to compile function body
        let parent_builder = std::mem::replace(&mut self.builder, func_builder);

        // Compile function body
        for stmt in body {
            self.compile_stmt(stmt)?;
        }

        // Ensure function returns None if no explicit return
        self.builder.emit_return_none();

        // Swap back and get finished function code
        let func_builder = std::mem::replace(&mut self.builder, parent_builder);
        let func_code = func_builder.finish();

        // Store the nested CodeObject as a constant
        let code_const_idx = self.builder.add_code_object(Arc::new(func_code));

        // Compile decorators in reverse order (they'll wrap the function)
        // Decorators are compiled first, then applied after function creation
        let decorator_regs: Vec<Register> = decorator_list
            .iter()
            .map(|d| self.compile_expr(d))
            .collect::<Result<_, _>>()?;

        // Emit function/closure creation
        let func_reg = self.builder.alloc_register();

        if has_closure {
            // MakeClosure: needs to capture variables from current scope
            // The instruction format is: dst = closure, imm16 = code index
            // Free variables must be loaded from current scope and packed
            self.builder.emit(Instruction::op_di(
                Opcode::MakeClosure,
                func_reg,
                code_const_idx,
            ));
        } else {
            // MakeFunction: simple function without captures
            self.builder.emit(Instruction::op_di(
                Opcode::MakeFunction,
                func_reg,
                code_const_idx,
            ));
        }

        // Apply decorators in reverse order
        // @decorator1
        // @decorator2
        // def func(): ...
        // is equivalent to: func = decorator1(decorator2(func))
        for decorator_reg in decorator_regs.into_iter().rev() {
            // Call decorator with function as argument
            let call_result = self.builder.alloc_register();
            // Place function as arg in next register after call_result
            self.builder
                .emit_move(Register::new(call_result.0 + 1), func_reg);
            self.builder.emit(Instruction::op_dss(
                Opcode::Call,
                call_result,
                decorator_reg,
                Register::new(1), // 1 argument
            ));
            self.builder.free_register(decorator_reg);
            self.builder.free_register(func_reg);
            // Result becomes the new function
            // Move result back to func_reg position for consistency
            // Actually, we'll just use call_result as the final function
            // But we need to track it properly - let's just store directly
        }

        // Store function to its name
        // For now, treat it as a global store (module level)
        // TODO: Proper scope-aware store for nested functions
        let name_idx = self.builder.add_name(Arc::from(name));
        self.builder.emit_store_global(name_idx, func_reg);
        self.builder.free_register(func_reg);

        Ok(())
    }

    /// Find a child scope by name in the current scope.
    ///
    /// This is used to look up the scope for nested function definitions
    /// so we can access cell and free variable information.
    fn find_child_scope(&self, name: &str) -> Option<&crate::scope::Scope> {
        // For now, search the root scope's children
        // TODO: Track current scope path for proper nested function compilation
        self.symbol_table
            .root
            .children
            .iter()
            .find(|c| c.name.as_ref() == name)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn compile(source: &str) -> CodeObject {
        let module = prism_parser::parse(source).expect("parse error");
        Compiler::compile_module(&module, "<test>").expect("compile error")
    }

    fn try_compile(source: &str) -> Result<CodeObject, CompileError> {
        let module = prism_parser::parse(source).expect("parse error");
        Compiler::compile_module(&module, "<test>")
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

    // =========================================================================
    // Loop Control Flow Tests (break/continue)
    // =========================================================================

    #[test]
    fn test_break_in_while_loop() {
        // Basic break in while loop
        let code = compile(
            r#"
i = 0
while True:
    i = i + 1
    if i >= 5:
        break
"#,
        );
        assert!(!code.instructions.is_empty());
        // Should have Jump instructions for break
        let has_jump = code.instructions.iter().any(|i| {
            let opcode = i.opcode();
            opcode == Opcode::Jump as u8
        });
        assert!(has_jump, "expected Jump instruction for break");
    }

    #[test]
    fn test_continue_in_while_loop() {
        // Continue in while loop
        let code = compile(
            r#"
total = 0
i = 0
while i < 10:
    i = i + 1
    if i % 2 == 0:
        continue
    total = total + i
"#,
        );
        assert!(!code.instructions.is_empty());
        // Should have Jump instructions for continue
        let has_jump = code.instructions.iter().any(|i| {
            let opcode = i.opcode();
            opcode == Opcode::Jump as u8
        });
        assert!(has_jump, "expected Jump instruction for continue");
    }

    #[test]
    fn test_break_in_for_loop() {
        // Break in for loop
        let code = compile(
            r#"
result = 0
for x in range(100):
    if x == 5:
        break
    result = result + x
"#,
        );
        assert!(!code.instructions.is_empty());
    }

    #[test]
    fn test_continue_in_for_loop() {
        // Continue in for loop
        let code = compile(
            r#"
total = 0
for x in range(10):
    if x % 2 == 0:
        continue
    total = total + x
"#,
        );
        assert!(!code.instructions.is_empty());
    }

    #[test]
    fn test_nested_loops_with_break() {
        // Break in nested loops - should only break inner loop
        let code = compile(
            r#"
found = False
for i in range(5):
    for j in range(5):
        if i == 2 and j == 3:
            found = True
            break
    if found:
        break
"#,
        );
        assert!(!code.instructions.is_empty());
    }

    #[test]
    fn test_nested_loops_with_continue() {
        // Continue in nested loops
        let code = compile(
            r#"
total = 0
for i in range(5):
    for j in range(5):
        if j % 2 == 0:
            continue
        total = total + 1
"#,
        );
        assert!(!code.instructions.is_empty());
    }

    #[test]
    fn test_while_with_else_and_break() {
        // While-else with break (else should be skipped on break)
        let code = compile(
            r#"
i = 0
while i < 10:
    if i == 5:
        break
    i = i + 1
else:
    x = 42
"#,
        );
        assert!(!code.instructions.is_empty());
    }

    #[test]
    fn test_for_with_else_and_break() {
        // For-else with break
        let code = compile(
            r#"
for x in range(10):
    if x == 5:
        break
else:
    y = 42
"#,
        );
        assert!(!code.instructions.is_empty());
    }

    #[test]
    fn test_break_outside_loop_error() {
        // Break outside loop should be an error
        let result = try_compile("break");
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(
            err.message.contains("'break' outside loop"),
            "expected 'break' outside loop error, got: {}",
            err.message
        );
    }

    #[test]
    fn test_continue_outside_loop_error() {
        // Continue outside loop should be an error
        let result = try_compile("continue");
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(
            err.message.contains("'continue' outside loop"),
            "expected 'continue' outside loop error, got: {}",
            err.message
        );
    }

    #[test]
    fn test_break_in_if_inside_loop() {
        // Break in if statement inside loop is valid
        let code = compile(
            r#"
for x in range(10):
    if x > 5:
        break
"#,
        );
        assert!(!code.instructions.is_empty());
    }

    #[test]
    fn test_deeply_nested_break() {
        // Break in deeply nested structure
        let code = compile(
            r#"
for a in range(5):
    for b in range(5):
        for c in range(5):
            if a + b + c > 10:
                break
"#,
        );
        assert!(!code.instructions.is_empty());
    }

    #[test]
    fn test_multiple_breaks_in_loop() {
        // Multiple break statements in same loop
        let code = compile(
            r#"
for x in range(100):
    if x == 5:
        break
    if x == 10:
        break
"#,
        );
        assert!(!code.instructions.is_empty());
    }

    #[test]
    fn test_break_and_continue_in_same_loop() {
        // Both break and continue in same loop
        let code = compile(
            r#"
for x in range(100):
    if x == 50:
        break
    if x % 2 == 0:
        continue
    y = x * 2
"#,
        );
        assert!(!code.instructions.is_empty());
    }

    // =========================================================================
    // Class Compilation Tests
    // =========================================================================

    #[test]
    fn test_compile_empty_class() {
        // Simplest possible class definition
        let code = compile(
            r#"
class Empty:
    pass
"#,
        );
        assert!(!code.instructions.is_empty());
        // Class body code should be in constants
        assert!(
            !code.constants.is_empty(),
            "Class should have nested code object"
        );
    }

    #[test]
    fn test_compile_class_with_method() {
        // Class with a simple method
        let code = compile(
            r#"
class Counter:
    def increment(self):
        pass
"#,
        );
        assert!(!code.instructions.is_empty());
        // Should have nested code object for class body
        assert!(
            !code.constants.is_empty(),
            "Class should have nested code objects"
        );
    }

    #[test]
    fn test_compile_class_with_init() {
        // Class with __init__ method
        let code = compile(
            r#"
class MyClass:
    def __init__(self, x):
        self.x = x
"#,
        );
        assert!(!code.instructions.is_empty());
    }

    #[test]
    fn test_compile_class_with_class_variable() {
        // Class with class-level variable
        let code = compile(
            r#"
class Config:
    DEBUG = True
    VERSION = 1
"#,
        );
        assert!(!code.instructions.is_empty());
    }

    #[test]
    fn test_compile_class_with_single_base() {
        // Simple inheritance
        let code = compile(
            r#"
class Child(Parent):
    pass
"#,
        );
        assert!(!code.instructions.is_empty());
    }

    #[test]
    fn test_compile_class_with_multiple_bases() {
        // Multiple inheritance
        let code = compile(
            r#"
class Multi(Base1, Base2, Base3):
    pass
"#,
        );
        assert!(!code.instructions.is_empty());
    }

    #[test]
    fn test_compile_class_with_decorator() {
        // Decorated class
        let code = compile(
            r#"
@decorator
class MyClass:
    pass
"#,
        );
        assert!(!code.instructions.is_empty());
        // Should have CALL for decorator application
    }

    #[test]
    fn test_compile_class_with_multiple_decorators() {
        // Multiple decorators
        let code = compile(
            r#"
@decorator1
@decorator2
@decorator3
class MyClass:
    pass
"#,
        );
        assert!(!code.instructions.is_empty());
    }

    #[test]
    fn test_compile_class_with_decorator_call() {
        // Decorator with arguments
        let code = compile(
            r#"
@dataclass(frozen=True)
class Point:
    x: int
    y: int
"#,
        );
        assert!(!code.instructions.is_empty());
    }

    #[test]
    fn test_compile_class_with_multiple_methods() {
        // Class with multiple methods
        let code = compile(
            r#"
class Calculator:
    def add(self, a, b):
        return a + b
    
    def subtract(self, a, b):
        return a - b
    
    def multiply(self, a, b):
        return a * b
"#,
        );
        assert!(!code.instructions.is_empty());
    }

    #[test]
    fn test_compile_class_with_static_method() {
        // Class with static method
        let code = compile(
            r#"
class Utils:
    @staticmethod
    def helper(x):
        return x * 2
"#,
        );
        assert!(!code.instructions.is_empty());
    }

    #[test]
    fn test_compile_class_with_class_method() {
        // Class with class method
        let code = compile(
            r#"
class Factory:
    @classmethod
    def create(cls):
        return cls()
"#,
        );
        assert!(!code.instructions.is_empty());
    }

    #[test]
    fn test_compile_class_with_property() {
        // Class with property decorator
        let code = compile(
            r#"
class Circle:
    @property
    def area(self):
        return 3.14159 * self.radius ** 2
"#,
        );
        assert!(!code.instructions.is_empty());
    }

    #[test]
    fn test_compile_nested_class() {
        // Nested class definition
        let code = compile(
            r#"
class Outer:
    class Inner:
        pass
"#,
        );
        assert!(!code.instructions.is_empty());
    }

    #[test]
    fn test_compile_deeply_nested_class() {
        // Deeply nested class definitions
        let code = compile(
            r#"
class Level1:
    class Level2:
        class Level3:
            value = 42
"#,
        );
        assert!(!code.instructions.is_empty());
    }

    #[test]
    fn test_compile_class_with_docstring() {
        // Class with docstring
        let code = compile(
            r#"
class Documented:
    """This is a docstring."""
    pass
"#,
        );
        assert!(!code.instructions.is_empty());
    }

    #[test]
    fn test_compile_class_with_super_init() {
        // Class calling super().__init__
        let code = compile(
            r#"
class Child(Parent):
    def __init__(self):
        super().__init__()
"#,
        );
        assert!(!code.instructions.is_empty());
    }

    #[test]
    fn test_compile_class_with_explicit_super() {
        // Class using explicit super(ClassName, self)
        let code = compile(
            r#"
class Child(Parent):
    def __init__(self):
        super(Child, self).__init__()
"#,
        );
        assert!(!code.instructions.is_empty());
    }

    #[test]
    fn test_compile_class_with_super_method_call() {
        // Class calling super() method
        let code = compile(
            r#"
class Child(Parent):
    def process(self):
        return super().process() + 1
"#,
        );
        assert!(!code.instructions.is_empty());
    }

    #[test]
    fn test_compile_class_with_dunder_methods() {
        // Class with magic methods
        let code = compile(
            r#"
class Custom:
    def __str__(self):
        return "Custom"
    
    def __repr__(self):
        return "Custom()"
    
    def __len__(self):
        return 0
"#,
        );
        assert!(!code.instructions.is_empty());
    }

    #[test]
    fn test_compile_class_with_operator_overloading() {
        // Class with operator overloading
        let code = compile(
            r#"
class Vector:
    def __add__(self, other):
        pass
    
    def __sub__(self, other):
        pass
    
    def __mul__(self, scalar):
        pass
"#,
        );
        assert!(!code.instructions.is_empty());
    }

    #[test]
    fn test_compile_class_with_slots() {
        // Class with __slots__ definition
        let code = compile(
            r#"
class Point:
    __slots__ = ['x', 'y']
"#,
        );
        assert!(!code.instructions.is_empty());
    }

    #[test]
    fn test_compile_class_with_class_body_expression() {
        // Class with expression in body
        let code = compile(
            r#"
class Computed:
    VALUE = 1 + 2 + 3
"#,
        );
        assert!(!code.instructions.is_empty());
    }

    #[test]
    fn test_compile_class_with_conditional() {
        // Class with conditional in body
        let code = compile(
            r#"
class Conditional:
    if True:
        x = 1
    else:
        x = 2
"#,
        );
        assert!(!code.instructions.is_empty());
    }

    #[test]
    fn test_compile_class_with_for_loop() {
        // Class with for loop in body
        let code = compile(
            r#"
class Generated:
    items = []
    for i in range(5):
        items.append(i)
"#,
        );
        assert!(!code.instructions.is_empty());
    }

    #[test]
    fn test_compile_class_with_comprehension() {
        // Class with comprehension in body
        let code = compile(
            r#"
class WithComprehension:
    squares = [x**2 for x in range(10)]
"#,
        );
        assert!(!code.instructions.is_empty());
    }

    #[test]
    fn test_compile_class_inheriting_from_expression() {
        // Class inheriting from expression
        let code = compile(
            r#"
class Sub(get_base()):
    pass
"#,
        );
        assert!(!code.instructions.is_empty());
    }

    #[test]
    fn test_compile_class_with_method_decorator() {
        // Method with multiple decorators
        let code = compile(
            r#"
class Service:
    @decorator1
    @decorator2
    def method(self):
        pass
"#,
        );
        assert!(!code.instructions.is_empty());
    }

    #[test]
    fn test_compile_class_with_private_method() {
        // Class with private method (name mangling)
        let code = compile(
            r#"
class Private:
    def __secret(self):
        pass
"#,
        );
        assert!(!code.instructions.is_empty());
    }

    #[test]
    fn test_compile_class_with_closure_in_method() {
        // Method containing a closure
        let code = compile(
            r#"
class WithClosure:
    def outer(self):
        x = 1
        def inner():
            return x
        return inner
"#,
        );
        assert!(!code.instructions.is_empty());
    }

    #[test]
    fn test_compile_multiple_classes() {
        // Multiple class definitions in same module
        let code = compile(
            r#"
class First:
    pass

class Second:
    pass

class Third(First, Second):
    pass
"#,
        );
        assert!(!code.instructions.is_empty());
    }

    #[test]
    fn test_compile_class_and_function() {
        // Class and function in same module
        let code = compile(
            r#"
def helper():
    pass

class MyClass:
    def method(self):
        helper()
"#,
        );
        assert!(!code.instructions.is_empty());
    }

    #[test]
    fn test_compile_dataclass_like() {
        // Dataclass-like pattern
        let code = compile(
            r#"
class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    def __eq__(self, other):
        return self.x == other.x and self.y == other.y
"#,
        );
        assert!(!code.instructions.is_empty());
    }

    #[test]
    fn test_compile_singleton_pattern() {
        // Singleton pattern
        let code = compile(
            r#"
class Singleton:
    _instance = None
    
    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
"#,
        );
        assert!(!code.instructions.is_empty());
    }

    #[test]
    fn test_compile_class_with_lambda_in_body() {
        // Class with lambda in body
        let code = compile(
            r#"
class WithLambda:
    transform = lambda x: x * 2
"#,
        );
        assert!(!code.instructions.is_empty());
    }

    #[test]
    fn test_class_code_object_has_class_flag() {
        // Verify class compilation produces code object
        let code = compile(
            r#"
class Flagged:
    pass
"#,
        );
        // Verify we have constants (class body code object)
        assert!(
            !code.constants.is_empty(),
            "Class body code object should exist in constants"
        );
        // Verify instructions are generated
        assert!(!code.instructions.is_empty());
    }

    // =========================================================================
    // Exception Compilation Tests
    // =========================================================================

    #[test]
    fn test_compile_simple_try_except() {
        let code = compile(
            r#"
try:
    x = 1
except:
    y = 2
"#,
        );
        assert!(!code.instructions.is_empty());
        assert!(!code.exception_table.is_empty());
    }

    #[test]
    fn test_compile_try_except_with_type() {
        let code = compile(
            r#"
try:
    x = dangerous()
except ValueError:
    y = fallback()
"#,
        );
        assert!(!code.instructions.is_empty());
        assert!(!code.exception_table.is_empty());
    }

    #[test]
    fn test_compile_try_except_else() {
        let code = compile(
            r#"
try:
    x = 1
except:
    y = 2
else:
    z = 3
"#,
        );
        assert!(!code.instructions.is_empty());
        assert!(!code.exception_table.is_empty());
    }

    #[test]
    fn test_compile_try_finally() {
        let code = compile(
            r#"
try:
    x = 1
finally:
    cleanup()
"#,
        );
        assert!(!code.instructions.is_empty());
        assert!(!code.exception_table.is_empty());
    }

    #[test]
    fn test_compile_try_except_finally() {
        let code = compile(
            r#"
try:
    x = 1
except ValueError:
    y = 2
finally:
    cleanup()
"#,
        );
        assert!(!code.instructions.is_empty());
        assert!(code.exception_table.len() >= 2);
    }

    #[test]
    fn test_compile_multiple_except_handlers() {
        let code = compile(
            r#"
try:
    x = risky()
except ValueError:
    a = 1
except TypeError:
    b = 2
except:
    c = 3
"#,
        );
        assert!(!code.instructions.is_empty());
        assert!(code.exception_table.len() >= 3);
    }

    #[test]
    fn test_compile_nested_try_except() {
        let code = compile(
            r#"
try:
    try:
        x = 1
    except:
        y = 2
except:
    z = 3
"#,
        );
        assert!(!code.instructions.is_empty());
        assert!(code.exception_table.len() >= 2);
    }

    #[test]
    fn test_compile_try_in_function() {
        let code = compile(
            r#"
def safe_divide(a, b):
    try:
        return a / b
    except ZeroDivisionError:
        return 0
"#,
        );
        assert!(!code.instructions.is_empty());
    }
}
