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
    /// Whether we are inside an async function.
    /// This is used to validate `await` expressions and `async for`/`async with` statements.
    in_async_context: bool,
    /// Whether we are inside a function/generator context.
    /// This is used to validate `yield` expressions.
    in_function_context: bool,
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
            in_async_context: false,
            in_function_context: false,
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
            in_async_context: false,
            in_function_context: false,
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
    ///
    /// For nested functions, also checks the builder's local map for
    /// parameters and locals defined via define_local().
    fn resolve_variable(&self, name: &str) -> VarLocation {
        // First, check if this name is defined as a local in the current builder.
        // This handles function parameters and body-level locals in nested functions.
        if let Some(slot) = self.builder.lookup_local(name) {
            return VarLocation::Local(slot.0);
        }

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

            // =========================================================================
            // Async For Loop (PEP 492)
            // =========================================================================
            // `async for target in iter:` compiles to:
            //   aiter = GetAIter(iter)            # Call __aiter__
            //   loop:
            //     try:
            //       anext_awaitable = GetANext(aiter)  # Call __anext__
            //       awaitable = GetAwaitable(anext_awaitable)
            //       result = YieldFrom(awaitable)  # Await the result
            //     except StopAsyncIteration:       # EndAsyncFor handles this
            //       goto else_block
            //     target = result
            //     <body>
            //     goto loop
            //   else_block:
            //     <orelse>
            //   end_block:
            // =========================================================================
            StmtKind::AsyncFor {
                target,
                iter,
                body,
                orelse,
            } => {
                // Validate async context
                if !self.in_async_context {
                    return Err(CompileError {
                        message: "'async for' outside async function".to_string(),
                        line: stmt.span.start,
                        column: 0,
                    });
                }

                // Step 1: Get async iterator
                let iter_expr = self.compile_expr(iter)?;
                let aiter_reg = self.builder.alloc_register();
                self.builder
                    .emit(Instruction::op_ds(Opcode::GetAIter, aiter_reg, iter_expr));
                self.builder.free_register(iter_expr);

                // Create labels
                let loop_start = self.builder.create_label();
                let loop_else = self.builder.create_label();
                let loop_end = self.builder.create_label();

                // Push loop context
                self.loop_stack.push(LoopContext {
                    break_label: loop_end,
                    continue_label: loop_start,
                });

                self.builder.bind_label(loop_start);

                // Step 2: Get next awaitable from async iterator
                let anext_reg = self.builder.alloc_register();
                self.builder
                    .emit(Instruction::op_ds(Opcode::GetANext, anext_reg, aiter_reg));

                // Step 3: Convert to awaitable and await
                self.builder.emit(Instruction::op_ds(
                    Opcode::GetAwaitable,
                    anext_reg,
                    anext_reg,
                ));
                self.builder
                    .emit(Instruction::op_ds(Opcode::YieldFrom, anext_reg, anext_reg));

                // Step 4: Check for StopAsyncIteration
                // EndAsyncFor checks if the awaited result indicates StopAsyncIteration
                // If so, clears the exception and jumps to the else label
                // Otherwise continues with the value
                self.builder.emit_end_async_for(anext_reg, loop_else);

                // Step 5: Store result to target
                self.compile_store(target, anext_reg)?;
                self.builder.free_register(anext_reg);

                // Step 6: Compile loop body
                for s in body {
                    self.compile_stmt(s)?;
                }

                // Loop back
                self.builder.emit_jump(loop_start);

                // Else clause
                self.builder.bind_label(loop_else);
                for s in orelse {
                    self.compile_stmt(s)?;
                }

                self.builder.bind_label(loop_end);

                // Cleanup
                self.loop_stack.pop();
                self.builder.free_register(aiter_reg);
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

                // Synthesize: raise AssertionError([message])
                // We lower through the regular call path so the raised value is a real
                // exception object (compatible with except AssertionError as e).
                let assertion_name_idx = self.builder.add_name(Arc::from("AssertionError"));
                let mut ctor_reg = self.builder.alloc_register();
                self.builder.emit_load_global(ctor_reg, assertion_name_idx);

                let arg_count = if msg.is_some() { 1u8 } else { 0u8 };

                let (call_dst, call_block, block_size) = if arg_count == 0 {
                    (ctor_reg, None, 0)
                } else {
                    // Reserve a fresh contiguous block [dst, arg0] for Call layout.
                    let block = self.builder.alloc_register_block(1 + arg_count);
                    (block, Some(block), 1 + arg_count)
                };

                // If the constructor register falls within the call block, move it
                // to a safe register to avoid clobbering.
                if let Some(block) = call_block {
                    let block_end = block.0 + block_size;
                    if ctor_reg.0 >= block.0 && ctor_reg.0 < block_end {
                        let safe_reg = self.builder.alloc_register();
                        self.builder.emit_move(safe_reg, ctor_reg);
                        self.builder.free_register(ctor_reg);
                        ctor_reg = safe_reg;
                    }
                }

                if let Some(msg_expr) = msg {
                    let arg_dst = Register::new(call_dst.0 + 1);
                    let temp = self.compile_expr(msg_expr)?;
                    if temp != arg_dst {
                        self.builder.emit_move(arg_dst, temp);
                    }
                    self.builder.free_register(temp);
                }

                self.builder.emit_call(call_dst, ctor_reg, arg_count);

                if let Some(block) = call_block {
                    if call_dst != ctor_reg {
                        self.builder.emit_move(ctor_reg, call_dst);
                    }
                    self.builder.free_register_block(block, block_size);
                }

                self.builder.emit(Instruction::op_d(Opcode::Raise, ctor_reg));
                self.builder.free_register(ctor_reg);

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

            StmtKind::With { items, body } => {
                self.compile_with(items, body)?;
            }

            StmtKind::AsyncWith { items, body } => {
                // Validate async context
                if !self.in_async_context {
                    return Err(CompileError {
                        message: "'async with' outside async function".to_string(),
                        line: stmt.span.start,
                        column: 0,
                    });
                }

                // Async with uses the same structure as sync with but awaits the
                // __aenter__ and __aexit__ calls. For now, we compile it using the
                // sync version's structure foundation but add await points.
                self.compile_async_with(items, body)?;
            }

            StmtKind::Match { subject, cases } => {
                self.compile_match(subject, cases)?;
            }

            // Remaining unimplemented statements
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
                // Check for call-site unpacking:
                // - *args: positional arg is ExprKind::Starred
                // - **kwargs: keyword with arg = None
                let has_star_unpack = args
                    .iter()
                    .any(|arg| matches!(&arg.kind, ExprKind::Starred(_)));
                let has_dstar_unpack = keywords.iter().any(|kw| kw.arg.is_none());

                if has_star_unpack || has_dstar_unpack {
                    // Dynamic call path: build tuple/dict and call with unpacking
                    return self.compile_dynamic_call(
                        func,
                        args,
                        keywords,
                        reg,
                        expr.span.start as u32,
                    );
                }

                // =====================================================================
                // OPTIMIZATION: Method call pattern detection
                // =====================================================================
                // If func is an Attribute expression (obj.method), use optimized
                // LoadMethod/CallMethod sequence instead of GetAttr + Call.
                // This avoids creating a bound method object on every call.
                if let ExprKind::Attribute { value, attr, .. } = &func.kind {
                    // Only use method call optimization for simple cases:
                    // - No keyword arguments (CallMethodKw not yet implemented)
                    // - No *args/**kwargs unpacking (already handled above)
                    if keywords.is_empty() {
                        return self.compile_method_call(value, attr, args, reg);
                    }
                    // TODO: Implement compile_method_call_kw for keyword arguments
                }

                // Strategy to avoid register collisions:
                // CRITICAL: The Call instruction uses consecutive registers [dst, dst+1, dst+2, ...].
                // The VM reads args from registers [dst+1, dst+2, ...]. If `reg` (allocated by compile_expr)
                // is from the free list at a low position, `reg+1` could clobber a live register
                // like list_reg in a list comprehension.
                //
                // SOLUTION: When there are arguments (argc > 0), allocate a fresh contiguous block
                // from next_register to avoid any collision. For zero-arg calls, use reg directly
                // since there are no consecutive arg writes to cause clobbering.

                let posargc = args.len();
                let kwargc = keywords.len();
                let total_argc = posargc + kwargc;

                // Compile function expression first (before allocating call block)
                let mut func_reg = self.compile_expr(func)?;

                // For calls WITH arguments, use fresh contiguous block to prevent clobbering.
                // For zero-arg calls, use `reg` directly (no arg writes to worry about).
                let (call_dst, call_block, block_size) = if total_argc > 0 {
                    // Allocate fresh contiguous block for [call_dst, arg0, arg1, ...]
                    let size = (1 + total_argc) as u8;
                    let block = self.builder.alloc_register_block(size);
                    (block, Some(block), size)
                } else {
                    // No arguments - use reg directly, no block needed
                    (reg, None, 0)
                };

                // Check if func_reg is inside our call block range (would be clobbered)
                if let Some(block) = call_block {
                    let block_end = block.0 + block_size;
                    if func_reg.0 >= block.0 && func_reg.0 < block_end {
                        // Move func to a safe register outside the block
                        let safe_reg = self.builder.alloc_register();
                        self.builder.emit_move(safe_reg, func_reg);
                        self.builder.free_register(func_reg);
                        func_reg = safe_reg;
                    }
                }

                // Compile positional arguments to call_dst+1..call_dst+posargc
                for (i, arg) in args.iter().enumerate() {
                    let arg_dst = Register::new(call_dst.0 + 1 + i as u8);
                    let temp = self.compile_expr(arg)?;
                    if temp != arg_dst {
                        self.builder.emit_move(arg_dst, temp);
                    }
                    self.builder.free_register(temp);
                }

                // Handle keyword arguments
                if keywords.is_empty() {
                    // No keywords - use simple Call instruction
                    self.builder.emit_call(call_dst, func_reg, posargc as u8);
                } else {
                    // Compile keyword argument values to consecutive registers
                    // after positional arguments
                    for (i, kw) in keywords.iter().enumerate() {
                        let kw_dst = Register::new(call_dst.0 + 1 + posargc as u8 + i as u8);
                        let temp = self.compile_expr(&kw.value)?;
                        if temp != kw_dst {
                            self.builder.emit_move(kw_dst, temp);
                        }
                        self.builder.free_register(temp);
                    }

                    // Build keyword names tuple for the constant pool
                    let kw_names: Vec<std::sync::Arc<str>> = keywords
                        .iter()
                        .map(|kw| {
                            // We already checked that arg is Some (no **kwargs unpacking)
                            std::sync::Arc::from(kw.arg.as_ref().unwrap().as_str())
                        })
                        .collect();
                    let kwnames_idx = self.builder.add_kwnames_tuple(kw_names);

                    // Emit CallKw instruction pair
                    self.builder.emit_call_kw(
                        call_dst,
                        func_reg,
                        posargc as u8,
                        kwargc as u8,
                        kwnames_idx,
                    );
                }

                self.builder.free_register(func_reg);

                // If we used a block, move result to expected destination and free block
                if let Some(block) = call_block {
                    if call_dst != reg {
                        self.builder.emit_move(reg, call_dst);
                    }
                    self.builder.free_register_block(block, block_size);
                }
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
                // CRITICAL: BuildTuple expects consecutive registers [first, first+1, ...].
                // Using alloc_register() individually can allocate non-contiguous registers
                // from the free list, breaking this invariant.
                // Use alloc_register_block to guarantee contiguity.
                if elts.is_empty() {
                    // Empty tuple - just build with no elements
                    self.builder.emit_build_tuple(reg, reg, 0);
                } else {
                    let count = elts.len() as u8;
                    let first_elem = self.builder.alloc_register_block(count);

                    for (i, elt) in elts.iter().enumerate() {
                        let elem_reg = Register::new(first_elem.0 + i as u8);
                        let temp = self.compile_expr(elt)?;
                        if temp != elem_reg {
                            self.builder.emit_move(elem_reg, temp);
                        }
                        self.builder.free_register(temp);
                    }

                    self.builder.emit_build_tuple(reg, first_elem, count);

                    // Free the element register block
                    self.builder.free_register_block(first_elem, count);
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

            // =========================================================================
            // Async/Await and Generator Expressions (PEP 492/255)
            // =========================================================================
            ExprKind::Await(value) => {
                // Validate we're in async context
                if !self.in_async_context {
                    return Err(CompileError {
                        message: "'await' outside async function".to_string(),
                        line: expr.span.start,
                        column: 0,
                    });
                }

                // Step 1: Compile the awaitable expression
                let awaitable_reg = self.compile_expr(value)?;

                // Step 2: GetAwaitable - Convert to awaitable (validates __await__)
                // This handles:
                //   - Coroutines → pass through
                //   - Async generators → pass through
                //   - Generators with CO_ITERABLE_COROUTINE → pass through
                //   - Objects with __await__ → call and verify
                self.builder
                    .emit(Instruction::op_ds(Opcode::GetAwaitable, reg, awaitable_reg));
                self.builder.free_register(awaitable_reg);

                // Step 3: YieldFrom - Delegate to the awaitable until completion
                // This suspends the coroutine and returns control to the event loop
                // The result ends up in reg when the awaitable completes
                self.builder
                    .emit(Instruction::op_ds(Opcode::YieldFrom, reg, reg));
            }

            ExprKind::Yield(value) => {
                // Yield must be inside a function (generator context)
                if !self.in_function_context {
                    return Err(CompileError {
                        message: "'yield' outside function".into(),
                        line: expr.span.start,
                        column: 0,
                    });
                }

                if let Some(val) = value {
                    // yield <expr>
                    let val_reg = self.compile_expr(val)?;
                    self.builder
                        .emit(Instruction::op_ds(Opcode::Yield, reg, val_reg));
                    self.builder.free_register(val_reg);
                } else {
                    // yield (no value - yields None)
                    let none_reg = self.builder.alloc_register();
                    self.builder.emit_load_none(none_reg);
                    self.builder
                        .emit(Instruction::op_ds(Opcode::Yield, reg, none_reg));
                    self.builder.free_register(none_reg);
                }
            }

            ExprKind::YieldFrom(value) => {
                // YieldFrom must be inside a function (generator context)
                if !self.in_function_context {
                    return Err(CompileError {
                        message: "'yield from' outside function".into(),
                        line: expr.span.start,
                        column: 0,
                    });
                }

                // Compile the iterable
                let iter_reg = self.compile_expr(value)?;

                // YieldFrom: dst = result, src = iterator
                self.builder
                    .emit(Instruction::op_ds(Opcode::YieldFrom, reg, iter_reg));
                self.builder.free_register(iter_reg);
            }

            ExprKind::Lambda { args, body } => {
                // Lambda expressions create nested code objects just like functions.
                // The key difference is:
                // 1. Body is a single expression (not statements)
                // 2. Result is implicitly returned
                // 3. Lambda inherits async context from enclosing scope
                return self.compile_lambda(args, body, reg);
            }

            ExprKind::ListComp { elt, generators } => {
                // List comprehensions create nested code objects for proper scoping.
                // This prevents loop variables from leaking into enclosing scope.
                return self.compile_listcomp(elt, generators, reg);
            }

            ExprKind::SetComp { elt, generators } => {
                // Set comprehensions follow same pattern as list comprehensions
                return self.compile_setcomp(elt, generators, reg);
            }

            ExprKind::DictComp {
                key,
                value,
                generators,
            } => {
                // Dict comprehensions create nested code for proper scoping
                return self.compile_dictcomp(key, value, generators, reg);
            }

            ExprKind::GeneratorExp { elt, generators } => {
                // Generator expressions are lazy - create generator function
                return self.compile_genexp(elt, generators, reg);
            }

            // TODO: Implement remaining expressions
            _ => {
                // Placeholder: load None for unimplemented expressions
                self.builder.emit_load_none(reg);
            }
        }

        Ok(reg)
    }

    /// Compile an optimized method call using LoadMethod/CallMethod.
    ///
    /// This is used when the call expression is of the form `obj.method(args...)`.
    /// Instead of:
    ///   1. GetAttr to load bound method
    ///   2. Call the bound method
    ///
    /// We emit:
    ///   1. LoadMethod: loads method and self into consecutive registers
    ///   2. CallMethod: calls with self already in place
    ///
    /// This optimization avoids allocating a BoundMethod object on every call,
    /// providing 15-30% speedup on method-heavy code.
    ///
    /// # Register Layout
    /// ```text
    /// [method_reg]:     method/function object
    /// [method_reg+1]:   self instance
    /// [method_reg+2..]: explicit arguments
    /// ```
    fn compile_method_call(
        &mut self,
        obj_expr: &Expr,
        method_name: &str,
        args: &[Expr],
        dst: Register,
    ) -> CompileResult<Register> {
        // Step 1: Compile the object expression
        let obj_reg = self.compile_expr(obj_expr)?;

        // Step 2: Allocate registers for method call
        // LoadMethod needs dst (for method) and dst+1 (for self)
        // Arguments go to dst+2, dst+3, etc.
        let method_reg = self.builder.alloc_register();
        // Reserve the next register for self (LoadMethod writes to both)
        let self_reg = self.builder.alloc_register();

        // Step 3: Emit LoadMethod - this populates method_reg and self_reg
        let name_idx = self.builder.add_name(method_name);
        self.builder.emit_load_method(method_reg, obj_reg, name_idx);

        // Free the object register since LoadMethod copies self to self_reg
        self.builder.free_register(obj_reg);

        // Step 4: Compile arguments into consecutive registers after self
        let arg_base = self_reg.0 + 1;
        let mut arg_regs = Vec::with_capacity(args.len());

        for (i, arg) in args.iter().enumerate() {
            let arg_dst = Register::new(arg_base + i as u8);
            // Reserve the register if it might be allocated elsewhere
            let temp = self.compile_expr(arg)?;
            if temp != arg_dst {
                self.builder.emit_move(arg_dst, temp);
                self.builder.free_register(temp);
            }
            arg_regs.push(arg_dst);
        }

        // Step 5: Emit CallMethod
        self.builder
            .emit_call_method(dst, method_reg, args.len() as u8);

        // Step 6: Cleanup - free method and self registers
        self.builder.free_register(method_reg);
        self.builder.free_register(self_reg);
        for arg_reg in arg_regs {
            self.builder.free_register(arg_reg);
        }

        Ok(dst)
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
    // Dynamic Call Compilation (with *args/**kwargs unpacking)
    // =========================================================================

    /// Compile a function call that contains *args or **kwargs unpacking.
    ///
    /// This builds a tuple for positional arguments and a dict for keyword arguments,
    /// then uses CallEx to invoke the function with the unpacked args.
    ///
    /// # Algorithm
    /// 1. Compile function expression
    /// 2. Build positional args into a tuple (merging any *iterables)
    /// 3. Build keyword args into a dict (merging any **mappings)
    /// 4. Emit CallEx(dst, func, args_tuple, kwargs_dict)
    fn compile_dynamic_call(
        &mut self,
        func: &Expr,
        args: &[Expr],
        keywords: &[prism_parser::ast::Keyword],
        dst: Register,
        _line: u32,
    ) -> CompileResult<Register> {
        // Step 1: Compile function
        let func_reg = self.compile_expr(func)?;

        // Step 2: Build positional args tuple
        // Separate regular args from starred args (for unpack flags)
        let args_tuple_reg = if args.is_empty() {
            // Empty tuple - use BuildTuple with 0 count
            let tuple_reg = self.builder.alloc_register();
            self.builder.emit_build_tuple(tuple_reg, tuple_reg, 0);
            tuple_reg
        } else {
            // Compile each arg and track which are starred
            let base_reg = self.builder.alloc_register();
            let mut unpack_flags: u32 = 0;

            for (i, arg) in args.iter().enumerate() {
                let arg_reg = Register::new(base_reg.0 + i as u8);

                match &arg.kind {
                    ExprKind::Starred(inner) => {
                        // This is a *iterable - compile the inner expression
                        let temp = self.compile_expr(inner)?;
                        if temp != arg_reg {
                            self.builder.emit_move(arg_reg, temp);
                        }
                        self.builder.free_register(temp);
                        // Mark this position for unpacking
                        unpack_flags |= 1 << i;
                    }
                    _ => {
                        // Regular arg - compile directly
                        let temp = self.compile_expr(arg)?;
                        if temp != arg_reg {
                            self.builder.emit_move(arg_reg, temp);
                        }
                        self.builder.free_register(temp);
                    }
                }
            }

            // Build tuple with unpacking (merges starred iterables)
            let tuple_reg = self.builder.alloc_register();
            self.builder.emit_build_tuple_unpack(
                tuple_reg,
                base_reg,
                args.len() as u8,
                unpack_flags,
            );

            // Free arg registers
            for i in 0..args.len() {
                self.builder
                    .free_register(Register::new(base_reg.0 + i as u8));
            }

            tuple_reg
        };

        // Step 3: Build keyword args dict (if any)
        let kwargs_dict_reg = if keywords.is_empty() {
            None
        } else {
            // Compile each keyword and track which are **dict
            let base_reg = self.builder.alloc_register();
            let mut unpack_flags: u32 = 0;
            let mut static_key_regs: Vec<(Register, Register)> = Vec::new(); // (key_reg, val_reg)

            for (i, kw) in keywords.iter().enumerate() {
                let val_reg = Register::new(base_reg.0 + i as u8);

                if kw.arg.is_none() {
                    // This is **dict - compile the mapping
                    let temp = self.compile_expr(&kw.value)?;
                    if temp != val_reg {
                        self.builder.emit_move(val_reg, temp);
                    }
                    self.builder.free_register(temp);
                    unpack_flags |= 1 << i;
                } else {
                    // Static keyword - we need both key and value
                    let key_name = kw.arg.as_ref().unwrap();
                    let key_idx = self.builder.add_string(key_name);
                    let key_reg = self.builder.alloc_register();
                    self.builder.emit_load_const(key_reg, key_idx);

                    let temp = self.compile_expr(&kw.value)?;
                    if temp != val_reg {
                        self.builder.emit_move(val_reg, temp);
                    }
                    self.builder.free_register(temp);

                    static_key_regs.push((key_reg, val_reg));
                }
            }

            // Build dict with potential unpacking
            let dict_reg = self.builder.alloc_register();
            self.builder.emit_build_dict_unpack(
                dict_reg,
                base_reg,
                keywords.len() as u8,
                unpack_flags,
            );

            // Free registers
            for (key_reg, _) in &static_key_regs {
                self.builder.free_register(*key_reg);
            }
            for i in 0..keywords.len() {
                self.builder
                    .free_register(Register::new(base_reg.0 + i as u8));
            }

            Some(dict_reg)
        };

        // Step 4: Emit CallEx
        self.builder
            .emit_call_ex(dst, func_reg, args_tuple_reg, kwargs_dict_reg);

        // Cleanup
        self.builder.free_register(func_reg);
        self.builder.free_register(args_tuple_reg);
        if let Some(kr) = kwargs_dict_reg {
            self.builder.free_register(kr);
        }

        Ok(dst)
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

        // =================================================================
        // Analysis Phase - Determine handler structure
        // =================================================================

        // Check if there's a bare except clause (catches all exceptions)
        let has_bare_except = handlers.iter().any(|h| h.typ.is_none());

        // Check if there are any typed handlers that need matching
        let has_typed_handlers = handlers.iter().any(|h| h.typ.is_some());

        // =================================================================
        // Label Creation Phase
        // =================================================================

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

        // Only create reraise label if we have typed handlers AND no bare except
        // (if there's a bare except, it will catch everything, so no reraise needed)
        let reraise_label = if has_typed_handlers && !has_bare_except {
            Some(self.builder.create_label())
        } else {
            None
        };

        // Create handler labels (one per except clause)
        let handler_labels: Vec<_> = handlers
            .iter()
            .map(|_| self.builder.create_label())
            .collect();

        // =================================================================
        // Try Body Compilation
        // =================================================================

        let try_start_pc = self.builder.current_pc();
        let stack_depth = self.builder.current_stack_depth();

        for stmt in body {
            self.compile_stmt(stmt)?;
        }

        let try_end_pc = self.builder.current_pc();

        // Jump to else/finally/end on normal completion (no exception)
        if let Some(else_label) = orelse_label {
            self.builder.emit_jump(else_label);
        } else if let Some(fin_label) = finally_label {
            self.builder.emit_jump(fin_label);
        } else if !handlers.is_empty() {
            self.builder.emit_jump(end_label);
        }

        // =================================================================
        // Exception Handler Compilation
        // =================================================================

        for (i, handler) in handlers.iter().enumerate() {
            self.builder.bind_label(handler_labels[i]);

            let handler_start_pc = self.builder.current_pc();

            // Compile handler match logic
            let type_idx = if let Some(type_expr) = &handler.typ {
                // -----------------------------------------------------------
                // Typed handler: except SomeException as e:
                // -----------------------------------------------------------

                // Compile the exception type expression to get the type class
                let type_reg = self.compile_expr(type_expr)?;

                // Load the current exception into a register for later binding
                let exc_reg = self.builder.alloc_register();
                self.builder
                    .emit(Instruction::op_d(Opcode::LoadException, exc_reg));

                // Check if exception matches type using dynamic matching
                // Note: ExceptionMatch reads src1 as the type, gets exception from VM state
                let match_reg = self.builder.alloc_register();
                self.builder.emit(Instruction::op_ds(
                    Opcode::ExceptionMatch,
                    match_reg,
                    type_reg,
                ));

                // Determine where to jump if no match
                let no_match_target = if i + 1 < handlers.len() {
                    // Try next handler
                    handler_labels[i + 1]
                } else if let Some(reraise_lbl) = reraise_label {
                    // No more handlers, reraise the exception
                    reraise_lbl
                } else if let Some(fin_label) = finally_label {
                    // No reraise needed, go to finally (bare except will catch)
                    fin_label
                } else {
                    // Should not happen if has_bare_except is true
                    end_label
                };

                self.builder.emit_jump_if_false(match_reg, no_match_target);

                self.builder.free_register(match_reg);
                self.builder.free_register(type_reg);

                // If handler has a name binding (except E as e:), store the exception
                if let Some(name) = &handler.name {
                    let location = self.resolve_variable(name);
                    self.builder
                        .emit_store_var(location, exc_reg, Some(name.as_ref()));
                }

                self.builder.free_register(exc_reg);

                Some(handler_start_pc as u16)
            } else {
                // -----------------------------------------------------------
                // Bare except: catches all exceptions
                // -----------------------------------------------------------

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

            // =============================================================
            // Handler Body Execution
            // =============================================================

            // NOTE: We do NOT clear the exception here at the start.
            // Bare `raise` inside the handler needs to access the exception.
            // We clear it AFTER the handler body completes successfully.

            // Compile handler body
            for stmt in &handler.body {
                self.compile_stmt(stmt)?;
            }

            // Clear exception state AFTER successful handler execution
            // If handler body contained bare `raise`, control flow never reaches here
            self.builder.emit(Instruction::op(Opcode::ClearException));

            // Jump to finally or end after successful handler execution
            if let Some(fin_label) = finally_label {
                self.builder.emit_jump(fin_label);
            } else {
                self.builder.emit_jump(end_label);
            }

            // Add exception entry for this handler
            self.builder.add_exception_entry(ExceptionEntry {
                start_pc: try_start_pc,
                end_pc: try_end_pc,
                handler_pc: handler_start_pc,
                finally_pc: u32::MAX,
                depth: stack_depth as u16,
                exception_type_idx: type_idx.unwrap_or(u16::MAX),
            });
        }

        // =================================================================
        // Else Block Compilation (runs only if no exception occurred)
        // =================================================================

        if let Some(else_label) = orelse_label {
            self.builder.bind_label(else_label);
            for stmt in orelse {
                self.compile_stmt(stmt)?;
            }
            if let Some(fin_label) = finally_label {
                self.builder.emit_jump(fin_label);
            } else {
                self.builder.emit_jump(end_label);
            }
        }

        // =================================================================
        // Reraise Path (only if typed handlers exist without bare except)
        // =================================================================

        if let Some(reraise_lbl) = reraise_label {
            self.builder.bind_label(reraise_lbl);

            if let Some(fin_label) = finally_label {
                // Execute finally before reraising
                self.builder.emit_jump(fin_label);
            } else {
                // No finally, reraise immediately
                self.builder.emit(Instruction::op(Opcode::Reraise));
            }
        }

        // =================================================================
        // Finally Block Compilation
        // =================================================================

        if let Some(fin_label) = finally_label {
            self.builder.bind_label(fin_label);
            let finally_start_pc = self.builder.current_pc();

            // Push exception info to preserve state during finally execution
            self.builder.emit(Instruction::op(Opcode::PushExcInfo));

            // Compile finally body
            for stmt in finalbody {
                self.compile_stmt(stmt)?;
            }

            // Pop exception info
            self.builder.emit(Instruction::op(Opcode::PopExcInfo));

            // EndFinally will reraise if there's a pending exception
            self.builder.emit(Instruction::op(Opcode::EndFinally));

            // Add finally exception entry
            self.builder.add_exception_entry(ExceptionEntry {
                start_pc: try_start_pc,
                end_pc: try_end_pc,
                handler_pc: finally_start_pc,
                finally_pc: finally_start_pc,
                depth: stack_depth as u16,
                exception_type_idx: u16::MAX,
            });
        }

        // =================================================================
        // End Label - Normal exit point
        // =================================================================

        self.builder.bind_label(end_label);

        Ok(())
    }

    // =========================================================================
    // With Statement (Context Manager) Compilation
    // =========================================================================

    /// Compile a with statement.
    ///
    /// The with statement implements the context manager protocol:
    ///
    /// ```python
    /// with expr as var:
    ///     body
    /// ```
    ///
    /// Is equivalent to:
    /// ```python
    /// mgr = expr
    /// value = mgr.__enter__()
    /// try:
    ///     var = value  # if as clause present
    ///     body
    /// except:
    ///     if not mgr.__exit__(*sys.exc_info()):
    ///         raise
    /// else:
    ///     mgr.__exit__(None, None, None)
    /// ```
    ///
    /// For multiple context managers, they are nested from left to right:
    /// ```python
    /// with a as x, b as y:
    ///     body
    /// # is equivalent to:
    /// with a as x:
    ///     with b as y:
    ///         body
    /// ```
    fn compile_with(
        &mut self,
        items: &[prism_parser::ast::WithItem],
        body: &[Stmt],
    ) -> CompileResult<()> {
        // Compile nested context managers recursively
        self.compile_with_items(items, body, 0)
    }

    /// Compile with statement items recursively for nested context managers.
    fn compile_with_items(
        &mut self,
        items: &[prism_parser::ast::WithItem],
        body: &[Stmt],
        depth: usize,
    ) -> CompileResult<()> {
        use crate::ExceptionEntry;

        if depth >= items.len() {
            // All context managers set up, compile the body
            for stmt in body {
                self.compile_stmt(stmt)?;
            }
            return Ok(());
        }

        let item = &items[depth];

        // Step 1: Evaluate context expression -> mgr
        let mgr_reg = self.compile_expr(&item.context_expr)?;

        // Step 2: Look up __enter__ and __exit__ methods
        let enter_name_idx = self.builder.add_name("__enter__");
        let exit_name_idx = self.builder.add_name("__exit__");

        // Step 3: Load __exit__ method (need to store for cleanup)
        // We store both the manager and __exit__ bound method for cleanup
        let exit_method_reg = self.builder.alloc_register();
        self.builder.emit(Instruction::new(
            Opcode::LoadMethod,
            exit_method_reg.0,
            mgr_reg.0,
            (exit_name_idx & 0xFF) as u8,
        ));

        // Step 4: Load __enter__ method
        let enter_method_reg = self.builder.alloc_register();
        self.builder.emit(Instruction::new(
            Opcode::LoadMethod,
            enter_method_reg.0,
            mgr_reg.0,
            (enter_name_idx & 0xFF) as u8,
        ));

        // Step 5: Call __enter__() -> value
        let value_reg = self.builder.alloc_register();
        self.builder.emit(Instruction::op_dss(
            Opcode::Call,
            value_reg,
            enter_method_reg,
            Register::new(0), // 0 arguments
        ));
        self.builder.free_register(enter_method_reg);

        // Step 6: If there's an as-clause, bind the value
        if let Some(optional_vars) = &item.optional_vars {
            self.compile_store(optional_vars, value_reg)?;
        }
        self.builder.free_register(value_reg);

        // Step 7: Set up exception handling for the body
        // Record try block start position
        let try_start_pc = self.builder.current_pc();
        let cleanup_label = self.builder.create_label();
        let end_label = self.builder.create_label();

        // Step 8: Compile nested items and body
        self.compile_with_items(items, body, depth + 1)?;

        // Step 9: Record try block end position (normal exit path)
        let try_end_pc = self.builder.current_pc();

        // Step 10: Normal exit - call __exit__(None, None, None)
        // Load three None values
        let none1_reg = self.builder.alloc_register();
        let none2_reg = self.builder.alloc_register();
        let none3_reg = self.builder.alloc_register();
        self.builder
            .emit(Instruction::op_d(Opcode::LoadNone, none1_reg));
        self.builder
            .emit(Instruction::op_d(Opcode::LoadNone, none2_reg));
        self.builder
            .emit(Instruction::op_d(Opcode::LoadNone, none3_reg));

        // Call exit_method(None, None, None)
        let exit_result_reg = self.builder.alloc_register();
        // We need to set up call with 3 arguments
        // Args go in registers following the result register
        self.builder.emit(Instruction::op_ds(
            Opcode::Move,
            Register::new(exit_result_reg.0 + 1),
            none1_reg,
        ));
        self.builder.emit(Instruction::op_ds(
            Opcode::Move,
            Register::new(exit_result_reg.0 + 2),
            none2_reg,
        ));
        self.builder.emit(Instruction::op_ds(
            Opcode::Move,
            Register::new(exit_result_reg.0 + 3),
            none3_reg,
        ));
        self.builder.emit(Instruction::op_dss(
            Opcode::Call,
            exit_result_reg,
            exit_method_reg,
            Register::new(3), // 3 arguments
        ));
        self.builder.free_register(none1_reg);
        self.builder.free_register(none2_reg);
        self.builder.free_register(none3_reg);
        self.builder.free_register(exit_result_reg);

        // Jump to end (skip exception path)
        self.builder.emit_jump(end_label);

        // Step 11: Exception cleanup path
        self.builder.bind_label(cleanup_label);
        let cleanup_start_pc = self.builder.current_pc();

        // Push exception info for cleanup
        self.builder.emit(Instruction::op(Opcode::PushExcInfo));

        // Load exception info registers
        let exc_type_reg = self.builder.alloc_register();
        let exc_val_reg = self.builder.alloc_register();
        let exc_tb_reg = self.builder.alloc_register();

        // LoadException gets the current exception value
        self.builder
            .emit(Instruction::op_d(Opcode::LoadException, exc_val_reg));

        // For now, we'll pass the exception value for all three
        // A full implementation would extract type and traceback
        self.builder
            .emit(Instruction::op_ds(Opcode::Move, exc_type_reg, exc_val_reg));
        self.builder
            .emit(Instruction::op_d(Opcode::LoadNone, exc_tb_reg));

        // Call __exit__(type, value, tb)
        let suppress_reg = self.builder.alloc_register();
        self.builder.emit(Instruction::op_ds(
            Opcode::Move,
            Register::new(suppress_reg.0 + 1),
            exc_type_reg,
        ));
        self.builder.emit(Instruction::op_ds(
            Opcode::Move,
            Register::new(suppress_reg.0 + 2),
            exc_val_reg,
        ));
        self.builder.emit(Instruction::op_ds(
            Opcode::Move,
            Register::new(suppress_reg.0 + 3),
            exc_tb_reg,
        ));
        self.builder.emit(Instruction::op_dss(
            Opcode::Call,
            suppress_reg,
            exit_method_reg,
            Register::new(3), // 3 arguments
        ));

        // Pop exception info
        self.builder.emit(Instruction::op(Opcode::PopExcInfo));

        // If __exit__ returns true, suppress the exception; otherwise reraise
        self.builder.emit_jump_if_true(suppress_reg, end_label);

        self.builder.free_register(exc_type_reg);
        self.builder.free_register(exc_val_reg);
        self.builder.free_register(exc_tb_reg);
        self.builder.free_register(suppress_reg);

        // Reraise the exception
        self.builder.emit(Instruction::op(Opcode::Reraise));

        // Step 12: End label
        self.builder.bind_label(end_label);

        // Free the stored method and manager registers
        self.builder.free_register(exit_method_reg);
        self.builder.free_register(mgr_reg);

        // Step 13: Add exception table entry for cleanup
        self.builder.add_exception_entry(ExceptionEntry {
            start_pc: try_start_pc,
            end_pc: try_end_pc,
            handler_pc: cleanup_start_pc,
            finally_pc: u32::MAX, // No separate finally, cleanup handles both
            depth: depth as u16,
            exception_type_idx: u16::MAX, // Catches all exceptions
        });

        Ok(())
    }

    // =========================================================================
    // Async With Statement Compilation
    // =========================================================================

    /// Compile async with statement with awaited __aenter__/__aexit__.
    ///
    /// `async with ctx as var:` compiles to roughly:
    ///   mgr = ctx
    ///   aexit = mgr.__aexit__
    ///   aenter = mgr.__aenter__
    ///   val = await aenter()
    ///   var = val
    ///   try:
    ///       <body>
    ///   except:
    ///       if not await aexit(type, val, tb): raise
    ///   else:
    ///       await aexit(None, None, None)
    fn compile_async_with(
        &mut self,
        items: &[prism_parser::ast::WithItem],
        body: &[Stmt],
    ) -> CompileResult<()> {
        self.compile_async_with_items(items, body, 0)
    }

    /// Compile async with statement items recursively for nested async context managers.
    fn compile_async_with_items(
        &mut self,
        items: &[prism_parser::ast::WithItem],
        body: &[Stmt],
        depth: usize,
    ) -> CompileResult<()> {
        use crate::ExceptionEntry;

        if depth >= items.len() {
            // All async context managers set up, compile the body
            for stmt in body {
                self.compile_stmt(stmt)?;
            }
            return Ok(());
        }

        let item = &items[depth];

        // Step 1: Evaluate context expression -> mgr
        let mgr_reg = self.compile_expr(&item.context_expr)?;

        // Step 2: Look up __aenter__ and __aexit__ methods
        let aenter_name_idx = self.builder.add_name("__aenter__");
        let aexit_name_idx = self.builder.add_name("__aexit__");

        // Step 3: Load __aexit__ method (need to store for cleanup)
        let aexit_method_reg = self.builder.alloc_register();
        self.builder.emit(Instruction::new(
            Opcode::LoadMethod,
            aexit_method_reg.0,
            mgr_reg.0,
            (aexit_name_idx & 0xFF) as u8,
        ));

        // Step 4: Load __aenter__ method
        let aenter_method_reg = self.builder.alloc_register();
        self.builder.emit(Instruction::new(
            Opcode::LoadMethod,
            aenter_method_reg.0,
            mgr_reg.0,
            (aenter_name_idx & 0xFF) as u8,
        ));

        // Step 5: Call __aenter__() and AWAIT the result
        let aenter_awaitable_reg = self.builder.alloc_register();
        self.builder.emit(Instruction::op_dss(
            Opcode::Call,
            aenter_awaitable_reg,
            aenter_method_reg,
            Register::new(0), // 0 arguments
        ));
        self.builder.free_register(aenter_method_reg);

        // Await the __aenter__ result: GetAwaitable + YieldFrom
        let value_reg = self.builder.alloc_register();
        self.builder.emit(Instruction::op_ds(
            Opcode::GetAwaitable,
            value_reg,
            aenter_awaitable_reg,
        ));
        self.builder.free_register(aenter_awaitable_reg);
        self.builder
            .emit(Instruction::op_ds(Opcode::YieldFrom, value_reg, value_reg));

        // Step 6: If there's an as-clause, bind the value
        if let Some(optional_vars) = &item.optional_vars {
            self.compile_store(optional_vars, value_reg)?;
        }
        self.builder.free_register(value_reg);

        // Step 7: Set up exception handling for the body
        let try_start_pc = self.builder.current_pc();
        let cleanup_label = self.builder.create_label();
        let end_label = self.builder.create_label();

        // Step 8: Compile nested items and body
        self.compile_async_with_items(items, body, depth + 1)?;

        // Step 9: Normal exit path
        let try_end_pc = self.builder.current_pc();

        // Load three None values for __aexit__(None, None, None)
        let none1_reg = self.builder.alloc_register();
        let none2_reg = self.builder.alloc_register();
        let none3_reg = self.builder.alloc_register();
        self.builder
            .emit(Instruction::op_d(Opcode::LoadNone, none1_reg));
        self.builder
            .emit(Instruction::op_d(Opcode::LoadNone, none2_reg));
        self.builder
            .emit(Instruction::op_d(Opcode::LoadNone, none3_reg));

        // Call __aexit__(None, None, None)
        let aexit_awaitable_reg = self.builder.alloc_register();
        self.builder.emit(Instruction::op_ds(
            Opcode::Move,
            Register::new(aexit_awaitable_reg.0 + 1),
            none1_reg,
        ));
        self.builder.emit(Instruction::op_ds(
            Opcode::Move,
            Register::new(aexit_awaitable_reg.0 + 2),
            none2_reg,
        ));
        self.builder.emit(Instruction::op_ds(
            Opcode::Move,
            Register::new(aexit_awaitable_reg.0 + 3),
            none3_reg,
        ));
        self.builder.emit(Instruction::op_dss(
            Opcode::Call,
            aexit_awaitable_reg,
            aexit_method_reg,
            Register::new(3),
        ));
        self.builder.free_register(none1_reg);
        self.builder.free_register(none2_reg);
        self.builder.free_register(none3_reg);

        // Await the __aexit__ result
        let aexit_result_reg = self.builder.alloc_register();
        self.builder.emit(Instruction::op_ds(
            Opcode::GetAwaitable,
            aexit_result_reg,
            aexit_awaitable_reg,
        ));
        self.builder.free_register(aexit_awaitable_reg);
        self.builder.emit(Instruction::op_ds(
            Opcode::YieldFrom,
            aexit_result_reg,
            aexit_result_reg,
        ));
        self.builder.free_register(aexit_result_reg);

        // Jump to end (skip exception path)
        self.builder.emit_jump(end_label);

        // Step 10: Exception cleanup path
        self.builder.bind_label(cleanup_label);
        let cleanup_start_pc = self.builder.current_pc();

        // Push exception info for cleanup
        self.builder.emit(Instruction::op(Opcode::PushExcInfo));

        // Load exception info registers
        let exc_type_reg = self.builder.alloc_register();
        let exc_val_reg = self.builder.alloc_register();
        let exc_tb_reg = self.builder.alloc_register();

        self.builder
            .emit(Instruction::op_d(Opcode::LoadException, exc_val_reg));
        self.builder
            .emit(Instruction::op_ds(Opcode::Move, exc_type_reg, exc_val_reg));
        self.builder
            .emit(Instruction::op_d(Opcode::LoadNone, exc_tb_reg));

        // Call __aexit__(type, value, tb)
        let suppress_awaitable_reg = self.builder.alloc_register();
        self.builder.emit(Instruction::op_ds(
            Opcode::Move,
            Register::new(suppress_awaitable_reg.0 + 1),
            exc_type_reg,
        ));
        self.builder.emit(Instruction::op_ds(
            Opcode::Move,
            Register::new(suppress_awaitable_reg.0 + 2),
            exc_val_reg,
        ));
        self.builder.emit(Instruction::op_ds(
            Opcode::Move,
            Register::new(suppress_awaitable_reg.0 + 3),
            exc_tb_reg,
        ));
        self.builder.emit(Instruction::op_dss(
            Opcode::Call,
            suppress_awaitable_reg,
            aexit_method_reg,
            Register::new(3),
        ));

        // Await the __aexit__ result for exception case
        let suppress_reg = self.builder.alloc_register();
        self.builder.emit(Instruction::op_ds(
            Opcode::GetAwaitable,
            suppress_reg,
            suppress_awaitable_reg,
        ));
        self.builder.free_register(suppress_awaitable_reg);
        self.builder.emit(Instruction::op_ds(
            Opcode::YieldFrom,
            suppress_reg,
            suppress_reg,
        ));

        // Pop exception info
        self.builder.emit(Instruction::op(Opcode::PopExcInfo));

        // If __aexit__ returns true, suppress the exception; otherwise reraise
        self.builder.emit_jump_if_true(suppress_reg, end_label);

        self.builder.free_register(exc_type_reg);
        self.builder.free_register(exc_val_reg);
        self.builder.free_register(exc_tb_reg);
        self.builder.free_register(suppress_reg);

        // Reraise the exception
        self.builder.emit(Instruction::op(Opcode::Reraise));

        // Step 11: End label
        self.builder.bind_label(end_label);

        // Free the stored method and manager registers
        self.builder.free_register(aexit_method_reg);
        self.builder.free_register(mgr_reg);

        // Step 12: Add exception table entry for cleanup
        self.builder.add_exception_entry(ExceptionEntry {
            start_pc: try_start_pc,
            end_pc: try_end_pc,
            handler_pc: cleanup_start_pc,
            finally_pc: u32::MAX,
            depth: depth as u16,
            exception_type_idx: u16::MAX,
        });

        Ok(())
    }

    // =========================================================================
    // Match Statement (Pattern Matching) Compilation
    // =========================================================================

    /// Compile a match statement using Maranget's decision tree algorithm.
    ///
    /// This implements Python 3.10+ structural pattern matching (PEP 634).
    /// The algorithm:
    /// 1. Evaluate the subject expression once
    /// 2. Build a pattern matrix from all cases
    /// 3. Generate a decision tree for optimal pattern testing
    /// 4. Emit bytecode that traverses the decision tree
    fn compile_match(
        &mut self,
        subject: &Expr,
        cases: &[prism_parser::ast::MatchCase],
    ) -> CompileResult<()> {
        // Step 1: Compile subject expression and store in register
        let subject_reg = self.compile_expr(subject)?;

        // Step 2: Create labels for each case and the end
        let end_label = self.builder.create_label();
        let case_labels: Vec<Label> = cases.iter().map(|_| self.builder.create_label()).collect();

        // Step 3: Compile pattern tests and bindings for each case
        // We compile cases in order, with fallthrough to next case on failure
        for (i, case) in cases.iter().enumerate() {
            let next_label = if i + 1 < cases.len() {
                case_labels[i + 1]
            } else {
                end_label
            };

            // Compile pattern match
            self.compile_pattern_match(&case.pattern, subject_reg, next_label)?;

            // Compile guard if present
            if let Some(guard) = &case.guard {
                let guard_reg = self.compile_expr(guard)?;
                self.builder.emit_jump_if_false(guard_reg, next_label);
                self.builder.free_register(guard_reg);
            }

            // Compile case body if pattern (and guard) matched
            for stmt in &case.body {
                self.compile_stmt(stmt)?;
            }

            // Jump to end after executing matched case
            self.builder.emit_jump(end_label);

            // Bind next case label for fallthrough
            if i + 1 < cases.len() {
                self.builder.bind_label(case_labels[i + 1]);
            }
        }

        // End label
        self.builder.bind_label(end_label);
        self.builder.free_register(subject_reg);

        Ok(())
    }

    /// Compile a pattern match test.
    ///
    /// On success, any bindings are stored to locals and execution continues.
    /// On failure, jumps to fail_label.
    fn compile_pattern_match(
        &mut self,
        pattern: &prism_parser::ast::Pattern,
        subject_reg: Register,
        fail_label: Label,
    ) -> CompileResult<()> {
        use prism_parser::ast::PatternKind;

        match &pattern.kind {
            PatternKind::MatchValue(expr) => {
                // Value pattern: subject == expr
                let value_reg = self.compile_expr(expr)?;
                let result_reg = self.builder.alloc_register();
                self.builder.emit_eq(result_reg, subject_reg, value_reg);
                self.builder.emit_jump_if_false(result_reg, fail_label);
                self.builder.free_register(result_reg);
                self.builder.free_register(value_reg);
            }

            PatternKind::MatchSingleton(singleton) => {
                // Singleton pattern: subject is True/False/None
                use prism_parser::ast::Singleton;
                let cmp_reg = self.builder.alloc_register();
                match singleton {
                    Singleton::True => self.builder.emit_load_true(cmp_reg),
                    Singleton::False => self.builder.emit_load_false(cmp_reg),
                    Singleton::None => self.builder.emit_load_none(cmp_reg),
                }
                let result_reg = self.builder.alloc_register();
                // Use IS comparison for singletons
                // TODO: emit_is for identity comparison; using emit_eq for now
                self.builder.emit_eq(result_reg, subject_reg, cmp_reg);
                self.builder.emit_jump_if_false(result_reg, fail_label);
                self.builder.free_register(result_reg);
                self.builder.free_register(cmp_reg);
            }

            PatternKind::MatchSequence(patterns) => {
                // Sequence pattern: [a, b, c]
                // First check if subject is a sequence type using MatchSequence opcode
                let is_seq_reg = self.builder.alloc_register();
                crate::match_compiler::emit_match_sequence(
                    &mut self.builder,
                    is_seq_reg,
                    subject_reg,
                );
                self.builder.emit_jump_if_false(is_seq_reg, fail_label);
                self.builder.free_register(is_seq_reg);

                // Check length
                let len_reg = self.builder.alloc_register();
                let len_name = self.builder.add_name(Arc::from("__len__"));
                let method_reg = self.builder.alloc_register();
                self.builder
                    .emit_get_attr(method_reg, subject_reg, len_name);
                self.builder.emit_call(len_reg, method_reg, 0);
                self.builder.free_register(method_reg);

                // Check length
                let expected_len = self.builder.add_int(patterns.len() as i64);
                let expected_reg = self.builder.alloc_register();
                self.builder.emit_load_const(expected_reg, expected_len);
                let cmp_reg = self.builder.alloc_register();
                self.builder.emit_eq(cmp_reg, len_reg, expected_reg);
                self.builder.emit_jump_if_false(cmp_reg, fail_label);
                self.builder.free_register(cmp_reg);
                self.builder.free_register(expected_reg);
                self.builder.free_register(len_reg);

                // Match each element
                for (idx, sub_pattern) in patterns.iter().enumerate() {
                    let idx_const = self.builder.add_int(idx as i64);
                    let idx_reg = self.builder.alloc_register();
                    self.builder.emit_load_const(idx_reg, idx_const);
                    let elem_reg = self.builder.alloc_register();
                    self.builder.emit_get_item(elem_reg, subject_reg, idx_reg);
                    self.compile_pattern_match(sub_pattern, elem_reg, fail_label)?;
                    self.builder.free_register(elem_reg);
                    self.builder.free_register(idx_reg);
                }
            }

            PatternKind::MatchMapping {
                keys,
                patterns,
                rest,
            } => {
                // Mapping pattern: {"key": value, ...}
                // First check if subject is a mapping type using MatchMapping opcode
                let is_map_reg = self.builder.alloc_register();
                crate::match_compiler::emit_match_mapping(
                    &mut self.builder,
                    is_map_reg,
                    subject_reg,
                );
                self.builder.emit_jump_if_false(is_map_reg, fail_label);
                self.builder.free_register(is_map_reg);

                // TODO: MatchKeys requires consecutive register allocation for key tuple.
                // For now, skip emitting MatchKeys and just check keys via GetItem.
                // The MatchMapping type check ensures we have a mapping type.

                // Match each key-value pair
                for (key, sub_pattern) in keys.iter().zip(patterns.iter()) {
                    let key_reg = self.compile_expr(key)?;
                    let value_reg = self.builder.alloc_register();
                    self.builder.emit_get_item(value_reg, subject_reg, key_reg);
                    self.compile_pattern_match(sub_pattern, value_reg, fail_label)?;
                    self.builder.free_register(value_reg);
                    self.builder.free_register(key_reg);
                }

                // Handle rest binding if present
                // TODO: CopyDictWithoutKeys requires consecutive register allocation for key tuple.
                // For now, skip emitting CopyDictWithoutKeys - rest binding requires runtime support.
                if rest.is_some() {
                    // Future: emit CopyDictWithoutKeys opcode here
                }
            }

            PatternKind::MatchClass {
                cls,
                patterns,
                kwd_attrs,
                kwd_patterns,
            } => {
                // Class pattern: ClassName(x, y, attr=z)
                // First get the class object and check isinstance
                let cls_reg = self.compile_expr(cls)?;

                // Use MatchClass opcode to check isinstance and get match result
                let match_result_reg = self.builder.alloc_register();
                crate::match_compiler::emit_match_class(
                    &mut self.builder,
                    match_result_reg,
                    subject_reg,
                    cls_reg,
                );
                self.builder
                    .emit_jump_if_false(match_result_reg, fail_label);
                self.builder.free_register(match_result_reg);
                self.builder.free_register(cls_reg);

                // Match positional patterns via __match_args__
                for (idx, sub_pattern) in patterns.iter().enumerate() {
                    // For now, just use index - proper implementation needs __match_args__
                    let idx_const = self.builder.add_int(idx as i64);
                    let idx_reg = self.builder.alloc_register();
                    self.builder.emit_load_const(idx_reg, idx_const);
                    let elem_reg = self.builder.alloc_register();
                    self.builder.emit_get_item(elem_reg, subject_reg, idx_reg);
                    self.compile_pattern_match(sub_pattern, elem_reg, fail_label)?;
                    self.builder.free_register(elem_reg);
                    self.builder.free_register(idx_reg);
                }

                // Match keyword patterns via attributes
                for (attr_name, sub_pattern) in kwd_attrs.iter().zip(kwd_patterns.iter()) {
                    let attr_idx = self.builder.add_name(Arc::from(attr_name.as_ref()));
                    let attr_reg = self.builder.alloc_register();
                    self.builder.emit_get_attr(attr_reg, subject_reg, attr_idx);
                    self.compile_pattern_match(sub_pattern, attr_reg, fail_label)?;
                    self.builder.free_register(attr_reg);
                }
            }

            PatternKind::MatchStar(_name) => {
                // Star pattern: *rest - captures remaining elements
                // This is only valid inside sequence patterns (handled there)
            }

            PatternKind::MatchAs { pattern, name } => {
                // As pattern: pattern as name, or just name (wildcard)
                // First match the inner pattern if any
                if let Some(inner) = pattern {
                    self.compile_pattern_match(inner, subject_reg, fail_label)?;
                }

                // Bind the name if present (None means wildcard _)
                if let Some(bound_name) = name {
                    // Use resolve_variable to properly handle scope
                    match self.resolve_variable(bound_name) {
                        VarLocation::Local(slot) => {
                            self.builder
                                .emit_store_local(LocalSlot::new(slot), subject_reg);
                        }
                        VarLocation::Global => {
                            let name_idx = self.builder.add_name(Arc::from(bound_name.as_ref()));
                            self.builder.emit_store_global(name_idx, subject_reg);
                        }
                        VarLocation::Closure(slot) => {
                            self.builder.emit_store_closure(slot, subject_reg);
                        }
                    }
                }
                // If name is None, it's just a wildcard _ which always matches
            }

            PatternKind::MatchOr(alternatives) => {
                // Or pattern: pattern1 | pattern2 | ...
                // Match succeeds if any alternative matches
                let success_label = self.builder.create_label();

                for (i, alt) in alternatives.iter().enumerate() {
                    let is_last = i + 1 == alternatives.len();

                    if is_last {
                        // Last alternative - fail to outer fail_label
                        self.compile_pattern_match(alt, subject_reg, fail_label)?;
                    } else {
                        // Not last - create temp fail label
                        let temp_fail = self.builder.create_label();
                        self.compile_pattern_match(alt, subject_reg, temp_fail)?;
                        // Match succeeded - jump to success
                        self.builder.emit_jump(success_label);
                        self.builder.bind_label(temp_fail);
                    }
                }

                self.builder.bind_label(success_label);
            }
        }

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
                if is_async {
                    // async def with yield = async generator
                    func_builder.add_flags(CodeFlags::ASYNC_GENERATOR);
                } else {
                    // regular generator
                    func_builder.add_flags(CodeFlags::GENERATOR);
                }
            }
        }

        if has_closure {
            func_builder.add_flags(CodeFlags::NESTED);
        }

        // Swap builders to compile function body
        let parent_builder = std::mem::replace(&mut self.builder, func_builder);

        // Save and update context for function body compilation
        let parent_async_context = self.in_async_context;
        let parent_function_context = self.in_function_context;
        self.in_async_context = is_async;
        self.in_function_context = true;

        // Compile function body
        for stmt in body {
            self.compile_stmt(stmt)?;
        }

        // Ensure function returns None if no explicit return
        self.builder.emit_return_none();

        // Restore contexts
        self.in_async_context = parent_async_context;
        self.in_function_context = parent_function_context;

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

    // =========================================================================
    // Lambda Expression Compilation
    // =========================================================================

    /// Compile a lambda expression.
    ///
    /// Lambda expressions create nested code objects like functions, but with:
    /// - Single expression body (not statements)
    /// - Implicit return of expression result
    /// - Anonymous name (`<lambda>`)
    /// - Inherits async context from enclosing scope
    ///
    /// # Performance Optimizations
    /// - Uses register-based evaluation for body expression
    /// - Direct return without intermediate storage
    /// - Closure handling only when capturing variables
    fn compile_lambda(
        &mut self,
        args: &prism_parser::ast::Arguments,
        body: &Expr,
        dst: Register,
    ) -> CompileResult<Register> {
        // Find lambda scope from symbol table (lambdas are named "<lambda>" in scope analysis)
        let lambda_scope = self.find_child_scope("<lambda>");

        // Create a new FunctionBuilder for the lambda body
        let mut lambda_builder = FunctionBuilder::new("<lambda>");
        lambda_builder.set_filename(&*self.filename);

        // Calculate argument counts
        let posonly_count = args.posonlyargs.len() as u16;
        let regular_args = args.args.len() as u16;
        let total_positional = posonly_count + regular_args;
        let kwonly_count = args.kwonlyargs.len() as u16;

        // Set parameter counts
        lambda_builder.set_arg_count(total_positional);
        lambda_builder.set_kwonlyarg_count(kwonly_count);
        lambda_builder.set_posonlyarg_count(posonly_count);

        // Handle varargs and kwargs
        if args.vararg.is_some() {
            lambda_builder.add_flags(CodeFlags::VARARGS);
        }
        if args.kwarg.is_some() {
            lambda_builder.add_flags(CodeFlags::VARKEYWORDS);
        }

        // Register parameters as locals
        for arg in &args.posonlyargs {
            lambda_builder.define_local(arg.arg.as_str());
        }
        for arg in &args.args {
            lambda_builder.define_local(arg.arg.as_str());
        }
        if let Some(ref vararg) = args.vararg {
            lambda_builder.define_local(vararg.arg.as_str());
        }
        for arg in &args.kwonlyargs {
            lambda_builder.define_local(arg.arg.as_str());
        }
        if let Some(ref kwarg) = args.kwarg {
            lambda_builder.define_local(kwarg.arg.as_str());
        }

        // Register cell and free variables from scope analysis
        let mut has_closure = false;
        if let Some(scope) = lambda_scope {
            for sym in scope.cellvars() {
                lambda_builder.add_cellvar(Arc::from(sym.name.as_ref()));
                has_closure = true;
            }
            for sym in scope.freevars() {
                lambda_builder.add_freevar(Arc::from(sym.name.as_ref()));
                has_closure = true;
            }
        }

        // Swap builders to compile lambda body
        let parent_builder = std::mem::replace(&mut self.builder, lambda_builder);
        let parent_async_context = self.in_async_context;
        let parent_function_context = self.in_function_context;
        // Lambda inherits async context from enclosing scope but sets function context
        self.in_function_context = true;

        // Compile the expression body
        let result_reg = self.compile_expr(body)?;

        // Emit implicit return of the expression result
        self.builder.emit_return(result_reg);

        // Restore parent contexts
        self.in_async_context = parent_async_context;
        self.in_function_context = parent_function_context;

        // Swap back and get finished lambda code
        let lambda_builder = std::mem::replace(&mut self.builder, parent_builder);
        let lambda_code = lambda_builder.finish();

        // Store the nested CodeObject as a constant
        let code_const_idx = self.builder.add_code_object(Arc::new(lambda_code));

        // Emit function/closure creation
        if has_closure {
            self.builder
                .emit(Instruction::op_di(Opcode::MakeClosure, dst, code_const_idx));
        } else {
            self.builder.emit(Instruction::op_di(
                Opcode::MakeFunction,
                dst,
                code_const_idx,
            ));
        }

        Ok(dst)
    }

    // =========================================================================
    // Comprehension Expression Compilation
    // =========================================================================

    /// Compile a list comprehension.
    ///
    /// List comprehensions create a nested scope (as a hidden function) to prevent
    /// loop variables from leaking into the enclosing scope. This matches Python 3
    /// semantics.
    ///
    /// # Bytecode Strategy
    /// 1. Create a hidden function containing the comprehension logic
    /// 2. Inside: create empty list, iterate with FOR_ITER, append elements
    /// 3. Call the hidden function with the first iterator
    /// 4. Result is the completed list
    ///
    /// # Performance Optimizations
    /// - Uses LIST_APPEND opcode for O(1) amortized append
    /// - Inlines filter conditions to avoid function call overhead
    /// - Pre-allocates result register for minimal register pressure
    fn compile_listcomp(
        &mut self,
        elt: &Expr,
        generators: &[prism_parser::ast::Comprehension],
        dst: Register,
    ) -> CompileResult<Register> {
        eprintln!("[DEBUG compile_listcomp] dst={}, entering", dst.0);

        // For now, compile comprehension inline for simplicity
        // Full implementation would create a nested scope

        // Create the result list
        // CRITICAL: Use alloc_register_block(1) instead of alloc_register() to prevent
        // the list register from being at a position that could be clobbered by
        // Call instruction's consecutive argument writes ([dst, dst+1, dst+2, ...]).
        // alloc_register_block allocates from next_register (guaranteed contiguous),
        // not from the free list which could give us a register adjacent to Call's dst.
        let list_reg = self.builder.alloc_register_block(1);
        eprintln!(
            "[DEBUG compile_listcomp] list_reg={} (will use for BuildList)",
            list_reg.0
        );

        // CRITICAL: Clear the free register list to prevent register reuse.
        // Call instructions use consecutive registers [dst, dst+1, ...], and if dst
        // is reused from the free list at a position before list_reg, then dst+1
        // could clobber list_reg. By clearing the free list, all subsequent allocations
        // (including Call's dst) will use fresh registers from next_register (which is
        // after list_reg), preventing any clobbering.
        self.builder.clear_free_registers();

        self.builder.emit_build_list(list_reg, list_reg, 0);

        // Compile generators (nested loops)
        self.compile_comprehension_generators(elt, generators, list_reg, ComprehensionKind::List)?;

        // Move result to destination
        eprintln!(
            "[DEBUG compile_listcomp] moving list_reg={} to dst={}",
            list_reg.0, dst.0
        );
        self.builder.emit_move(dst, list_reg);
        eprintln!("[DEBUG compile_listcomp] freeing list_reg={}", list_reg.0);
        self.builder.free_register(list_reg);

        Ok(dst)
    }

    /// Compile a set comprehension.
    fn compile_setcomp(
        &mut self,
        elt: &Expr,
        generators: &[prism_parser::ast::Comprehension],
        dst: Register,
    ) -> CompileResult<Register> {
        // Create empty set
        let set_reg = self.builder.alloc_register();
        self.builder
            .emit(Instruction::op_d(Opcode::BuildSet, set_reg));

        // Compile generators
        self.compile_comprehension_generators(elt, generators, set_reg, ComprehensionKind::Set)?;

        // Move result to destination
        self.builder.emit_move(dst, set_reg);
        self.builder.free_register(set_reg);

        Ok(dst)
    }

    /// Compile a dict comprehension.
    fn compile_dictcomp(
        &mut self,
        key: &Expr,
        value: &Expr,
        generators: &[prism_parser::ast::Comprehension],
        dst: Register,
    ) -> CompileResult<Register> {
        // Create empty dict
        let dict_reg = self.builder.alloc_register();
        self.builder
            .emit(Instruction::op_d(Opcode::BuildDict, dict_reg));

        // Compile generators with key-value pair
        self.compile_dict_comprehension_generators(key, value, generators, dict_reg)?;

        // Move result to destination
        self.builder.emit_move(dst, dict_reg);
        self.builder.free_register(dict_reg);

        Ok(dst)
    }

    /// Compile a generator expression.
    ///
    /// Generator expressions are lazy - they create a generator function that
    /// yields values on demand. This is more memory efficient for large sequences.
    fn compile_genexp(
        &mut self,
        elt: &Expr,
        generators: &[prism_parser::ast::Comprehension],
        dst: Register,
    ) -> CompileResult<Register> {
        // Create a generator function that yields each element
        let mut gen_builder = FunctionBuilder::new("<genexpr>");
        gen_builder.set_filename(&*self.filename);
        gen_builder.add_flags(CodeFlags::GENERATOR);

        // First iterator is passed as argument
        gen_builder.set_arg_count(1);
        gen_builder.define_local(".0"); // Hidden argument for first iterator

        // Swap builders
        let parent_builder = std::mem::replace(&mut self.builder, gen_builder);

        // Get the first iterator from argument
        let iter_reg = self.builder.alloc_register();
        self.builder
            .emit_load_local(iter_reg, crate::bytecode::LocalSlot::new(0));

        // Compile generator loops (yields instead of appending)
        self.compile_genexp_generators(elt, generators, 0, iter_reg)?;

        // Return None at end
        self.builder.emit_return_none();

        // Swap back
        let gen_builder = std::mem::replace(&mut self.builder, parent_builder);
        let gen_code = gen_builder.finish();

        // Store code object and create function
        let code_idx = self.builder.add_code_object(Arc::new(gen_code));
        let func_reg = self.builder.alloc_register();
        self.builder
            .emit(Instruction::op_di(Opcode::MakeFunction, func_reg, code_idx));

        // Compile first iterator and call generator function
        let first_iter = self.compile_expr(&generators[0].iter)?;
        let iter_result = self.builder.alloc_register();
        self.builder.emit_get_iter(iter_result, first_iter);
        self.builder.free_register(first_iter);

        // Call generator function with first iterator
        self.builder
            .emit_move(Register::new(dst.0 + 1), iter_result);
        self.builder.emit_call(dst, func_reg, 1);
        self.builder.free_register(func_reg);
        self.builder.free_register(iter_result);

        Ok(dst)
    }

    /// Helper to compile comprehension generators (list/set).
    fn compile_comprehension_generators(
        &mut self,
        elt: &Expr,
        generators: &[prism_parser::ast::Comprehension],
        result_reg: Register,
        kind: ComprehensionKind,
    ) -> CompileResult<()> {
        if generators.is_empty() {
            // Base case: compute element and add to collection
            let elem_reg = self.compile_expr(elt)?;
            match kind {
                ComprehensionKind::List => {
                    eprintln!(
                        "[DEBUG emit ListAppend] result_reg={}, elem_reg={}",
                        result_reg.0, elem_reg.0
                    );
                    // ListAppend: src1.append(src2) - list in src1, element in src2
                    self.builder.emit(Instruction::op_dss(
                        Opcode::ListAppend,
                        Register(0), // dst unused for ListAppend
                        result_reg,  // src1 = list
                        elem_reg,    // src2 = element
                    ));
                }
                ComprehensionKind::Set => {
                    // SetAdd: src1.add(src2) - set in src1, element in src2
                    self.builder.emit(Instruction::op_dss(
                        Opcode::SetAdd,
                        Register(0), // dst unused for SetAdd
                        result_reg,  // src1 = set
                        elem_reg,    // src2 = element
                    ));
                }
            }
            self.builder.free_register(elem_reg);
            return Ok(());
        }

        let comp_gen = &generators[0];
        let rest = &generators[1..];

        // Compile iterator
        let iter_expr_reg = self.compile_expr(&comp_gen.iter)?;
        let iter_reg = self.builder.alloc_register();

        // Get iterator (sync or async)
        if comp_gen.is_async {
            if !self.in_async_context {
                return Err(CompileError {
                    message: "asynchronous comprehension outside of an async function".to_string(),
                    line: 0,
                    column: 0,
                });
            }
            self.builder.emit(Instruction::op_ds(
                Opcode::GetAIter,
                iter_reg,
                iter_expr_reg,
            ));
        } else {
            self.builder.emit_get_iter(iter_reg, iter_expr_reg);
        }
        self.builder.free_register(iter_expr_reg);

        // Create loop labels
        let loop_start = self.builder.create_label();
        let loop_end = self.builder.create_label();

        self.builder.bind_label(loop_start);

        // Get next item
        let item_reg = self.builder.alloc_register();
        if comp_gen.is_async {
            self.builder
                .emit(Instruction::op_ds(Opcode::GetANext, item_reg, iter_reg));
            // await the result
            self.builder
                .emit(Instruction::op_ds(Opcode::GetAwaitable, item_reg, item_reg));
            self.builder
                .emit(Instruction::op_ds(Opcode::YieldFrom, item_reg, item_reg));
        } else {
            self.builder.emit_for_iter(item_reg, iter_reg, loop_end);
        }

        // Unpack target
        self.compile_store(&comp_gen.target, item_reg)?;

        // Compile filter conditions
        for if_expr in &comp_gen.ifs {
            let cond_reg = self.compile_expr(if_expr)?;
            self.builder.emit_jump_if_false(cond_reg, loop_start);
            self.builder.free_register(cond_reg);
        }

        // Recurse for nested generators or emit element
        self.compile_comprehension_generators(elt, rest, result_reg, kind)?;

        // Jump back to loop start
        self.builder.emit_jump(loop_start);

        self.builder.bind_label(loop_end);
        self.builder.free_register(iter_reg);
        self.builder.free_register(item_reg);

        Ok(())
    }

    /// Helper to compile dict comprehension generators.
    fn compile_dict_comprehension_generators(
        &mut self,
        key: &Expr,
        value: &Expr,
        generators: &[prism_parser::ast::Comprehension],
        result_reg: Register,
    ) -> CompileResult<()> {
        if generators.is_empty() {
            // Base case: compute key-value and add to dict
            let key_reg = self.compile_expr(key)?;
            let val_reg = self.compile_expr(value)?;
            self.builder.emit_set_item(result_reg, key_reg, val_reg);
            self.builder.free_register(key_reg);
            self.builder.free_register(val_reg);
            return Ok(());
        }

        let comp_gen = &generators[0];
        let rest = &generators[1..];

        // Compile iterator
        let iter_expr_reg = self.compile_expr(&comp_gen.iter)?;
        let iter_reg = self.builder.alloc_register();

        if comp_gen.is_async {
            if !self.in_async_context {
                return Err(CompileError {
                    message: "asynchronous comprehension outside of an async function".to_string(),
                    line: 0,
                    column: 0,
                });
            }
            self.builder.emit(Instruction::op_ds(
                Opcode::GetAIter,
                iter_reg,
                iter_expr_reg,
            ));
        } else {
            self.builder.emit_get_iter(iter_reg, iter_expr_reg);
        }
        self.builder.free_register(iter_expr_reg);

        // Create loop labels
        let loop_start = self.builder.create_label();
        let loop_end = self.builder.create_label();

        self.builder.bind_label(loop_start);

        // Get next item
        let item_reg = self.builder.alloc_register();
        if comp_gen.is_async {
            self.builder
                .emit(Instruction::op_ds(Opcode::GetANext, item_reg, iter_reg));
            self.builder
                .emit(Instruction::op_ds(Opcode::GetAwaitable, item_reg, item_reg));
            self.builder
                .emit(Instruction::op_ds(Opcode::YieldFrom, item_reg, item_reg));
        } else {
            self.builder.emit_for_iter(item_reg, iter_reg, loop_end);
        }

        // Unpack target
        self.compile_store(&comp_gen.target, item_reg)?;

        // Compile filter conditions
        for if_expr in &comp_gen.ifs {
            let cond_reg = self.compile_expr(if_expr)?;
            self.builder.emit_jump_if_false(cond_reg, loop_start);
            self.builder.free_register(cond_reg);
        }

        // Recurse for nested generators
        self.compile_dict_comprehension_generators(key, value, rest, result_reg)?;

        // Jump back to loop start
        self.builder.emit_jump(loop_start);

        self.builder.bind_label(loop_end);
        self.builder.free_register(iter_reg);
        self.builder.free_register(item_reg);

        Ok(())
    }

    /// Helper to compile generator expression generators.
    fn compile_genexp_generators(
        &mut self,
        elt: &Expr,
        generators: &[prism_parser::ast::Comprehension],
        depth: usize,
        iter_reg: Register,
    ) -> CompileResult<()> {
        if generators.is_empty() {
            // Base case: yield element
            let elem_reg = self.compile_expr(elt)?;
            let yield_result = self.builder.alloc_register();
            self.builder
                .emit(Instruction::op_ds(Opcode::Yield, yield_result, elem_reg));
            self.builder.free_register(yield_result);
            self.builder.free_register(elem_reg);
            return Ok(());
        }

        let comp_gen = &generators[0];
        let rest = &generators[1..];

        // For depth > 0, compile iterator; depth 0 uses passed-in iter_reg
        let actual_iter = if depth == 0 {
            iter_reg
        } else {
            let iter_expr_reg = self.compile_expr(&comp_gen.iter)?;
            let new_iter = self.builder.alloc_register();
            self.builder.emit_get_iter(new_iter, iter_expr_reg);
            self.builder.free_register(iter_expr_reg);
            new_iter
        };

        // Create loop labels
        let loop_start = self.builder.create_label();
        let loop_end = self.builder.create_label();

        self.builder.bind_label(loop_start);

        // Get next item
        let item_reg = self.builder.alloc_register();
        self.builder.emit_for_iter(item_reg, actual_iter, loop_end);

        // Unpack target
        self.compile_store(&comp_gen.target, item_reg)?;

        // Compile filter conditions
        for if_expr in &comp_gen.ifs {
            let cond_reg = self.compile_expr(if_expr)?;
            self.builder.emit_jump_if_false(cond_reg, loop_start);
            self.builder.free_register(cond_reg);
        }

        // Recurse for nested generators
        let next_iter = self.builder.alloc_register();
        if !rest.is_empty() {
            let iter_expr_reg = self.compile_expr(&rest[0].iter)?;
            self.builder.emit_get_iter(next_iter, iter_expr_reg);
            self.builder.free_register(iter_expr_reg);
        }
        self.compile_genexp_generators(elt, rest, depth + 1, next_iter)?;
        if !rest.is_empty() {
            self.builder.free_register(next_iter);
        }

        // Jump back to loop start
        self.builder.emit_jump(loop_start);

        self.builder.bind_label(loop_end);
        if depth > 0 {
            self.builder.free_register(actual_iter);
        }
        self.builder.free_register(item_reg);

        Ok(())
    }
}

/// Kind of comprehension being compiled.
#[derive(Debug, Clone, Copy)]
enum ComprehensionKind {
    List,
    Set,
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
    fn test_compile_assert_emits_raise_path() {
        let code = compile("assert False");
        let opcodes: Vec<u8> = code.instructions.iter().map(|inst| inst.opcode()).collect();

        assert!(
            opcodes.iter().any(|op| *op == Opcode::LoadGlobal as u8),
            "assert should load AssertionError constructor"
        );
        assert!(
            opcodes.iter().any(|op| *op == Opcode::Call as u8),
            "assert should call AssertionError constructor"
        );
        assert!(
            opcodes.iter().any(|op| *op == Opcode::Raise as u8),
            "assert should raise the constructed exception"
        );
    }

    #[test]
    fn test_compile_assert_with_message_emits_call_with_one_arg() {
        let code = compile("assert False, 42");

        let call = code
            .instructions
            .iter()
            .find(|inst| inst.opcode() == Opcode::Call as u8)
            .expect("assert with message should emit Call");
        assert_eq!(call.src2().0, 1, "assert message should be passed as 1 call arg");
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
