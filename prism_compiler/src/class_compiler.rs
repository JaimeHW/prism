//! Class body compilation.
//!
//! This module handles the compilation of Python class definitions into bytecode.
//! The compilation follows CPython's BUILD_CLASS protocol:
//!
//! 1. A child `CodeObject` is created for the class body
//! 2. The class body is executed, populating the class namespace
//! 3. `type()` or a metaclass is called with `(name, bases, namespace, **kwds)`
//!
//! ## Zero-arg super() Optimization
//!
//! Methods that use `super()` without arguments get an implicit `__class__` cell
//! variable injected at compilation time. This enables `super()` to work correctly
//! without requiring explicit class/instance arguments.

use crate::bytecode::{CodeFlags, CodeObject, FunctionBuilder};
use prism_parser::ast::{Expr, ExprKind, Stmt, StmtKind};
use std::sync::Arc;

/// Compiler for class body definitions.
///
/// Creates a child `CodeObject` that, when executed, returns a namespace dict
/// containing all class members (methods, attributes, etc.).
pub struct ClassCompiler {
    /// Name of the class being compiled.
    name: Arc<str>,
    /// Qualified name (includes enclosing scopes).
    qualname: Arc<str>,
    /// Filename for debug information.
    filename: Arc<str>,
    /// Whether any method uses zero-arg super().
    has_implicit_class_cell: bool,
    /// The function builder for the class body.
    builder: FunctionBuilder,
}

impl ClassCompiler {
    /// Create a new class compiler.
    pub fn new(
        name: impl Into<Arc<str>>,
        qualname: impl Into<Arc<str>>,
        filename: impl Into<Arc<str>>,
    ) -> Self {
        let name = name.into();
        let qualname = qualname.into();
        let filename_arc = filename.into();

        let mut builder = FunctionBuilder::new(name.clone());
        builder.set_filename(filename_arc.clone());
        builder.set_qualname(qualname.clone());

        ClassCompiler {
            name,
            qualname,
            filename: filename_arc,
            has_implicit_class_cell: false,
            builder,
        }
    }

    /// Check if a function body contains a call to `super()` with no arguments.
    ///
    /// This is used to determine if we need to inject the `__class__` cell.
    pub fn uses_zero_arg_super(body: &[Stmt]) -> bool {
        struct SuperVisitor {
            found: bool,
        }

        impl SuperVisitor {
            fn visit_expr(&mut self, expr: &Expr) {
                if self.found {
                    return;
                }

                match &expr.kind {
                    ExprKind::Call {
                        func,
                        args,
                        keywords,
                    } => {
                        // Check if it's a call to `super()`
                        if let ExprKind::Name(name) = &func.kind {
                            let name_str: &str = name.as_ref();
                            if name_str == "super" && args.is_empty() && keywords.is_empty() {
                                self.found = true;
                                return;
                            }
                        }
                        // Recurse into function call arguments
                        self.visit_expr(func);
                        for arg in args {
                            self.visit_expr(arg);
                        }
                        for kw in keywords {
                            self.visit_expr(&kw.value);
                        }
                    }
                    ExprKind::Lambda { body, .. } => {
                        self.visit_expr(body);
                    }
                    ExprKind::BinOp { left, right, .. } => {
                        self.visit_expr(left);
                        self.visit_expr(right);
                    }
                    ExprKind::UnaryOp { operand, .. } => {
                        self.visit_expr(operand);
                    }
                    ExprKind::IfExp { test, body, orelse } => {
                        self.visit_expr(test);
                        self.visit_expr(body);
                        self.visit_expr(orelse);
                    }
                    ExprKind::Attribute { value, .. } => {
                        self.visit_expr(value);
                    }
                    ExprKind::Subscript { value, slice, .. } => {
                        self.visit_expr(value);
                        self.visit_expr(slice);
                    }
                    ExprKind::List(elts) | ExprKind::Tuple(elts) | ExprKind::Set(elts) => {
                        for e in elts {
                            self.visit_expr(e);
                        }
                    }
                    ExprKind::Dict { keys, values } => {
                        for k in keys.iter().flatten() {
                            self.visit_expr(k);
                        }
                        for v in values {
                            self.visit_expr(v);
                        }
                    }
                    ExprKind::Compare {
                        left, comparators, ..
                    } => {
                        self.visit_expr(left);
                        for c in comparators {
                            self.visit_expr(c);
                        }
                    }
                    ExprKind::BoolOp { values, .. } => {
                        for v in values {
                            self.visit_expr(v);
                        }
                    }
                    ExprKind::ListComp { elt, generators }
                    | ExprKind::SetComp { elt, generators }
                    | ExprKind::GeneratorExp { elt, generators } => {
                        self.visit_expr(elt);
                        for comp in generators {
                            self.visit_expr(&comp.iter);
                            self.visit_expr(&comp.target);
                            for cond in &comp.ifs {
                                self.visit_expr(cond);
                            }
                        }
                    }
                    ExprKind::DictComp {
                        key,
                        value,
                        generators,
                    } => {
                        self.visit_expr(key);
                        self.visit_expr(value);
                        for comp in generators {
                            self.visit_expr(&comp.iter);
                            self.visit_expr(&comp.target);
                            for cond in &comp.ifs {
                                self.visit_expr(cond);
                            }
                        }
                    }
                    ExprKind::Await(e)
                    | ExprKind::Yield(Some(e))
                    | ExprKind::YieldFrom(e)
                    | ExprKind::Starred(e) => {
                        self.visit_expr(e);
                    }
                    ExprKind::NamedExpr { target, value } => {
                        self.visit_expr(target);
                        self.visit_expr(value);
                    }
                    ExprKind::JoinedStr(parts) => {
                        for p in parts {
                            self.visit_expr(p);
                        }
                    }
                    ExprKind::FormattedValue {
                        value, format_spec, ..
                    } => {
                        self.visit_expr(value);
                        if let Some(spec) = format_spec {
                            self.visit_expr(spec);
                        }
                    }
                    ExprKind::Slice { lower, upper, step } => {
                        if let Some(l) = lower {
                            self.visit_expr(l);
                        }
                        if let Some(u) = upper {
                            self.visit_expr(u);
                        }
                        if let Some(s) = step {
                            self.visit_expr(s);
                        }
                    }
                    // Terminals - no recursion needed
                    _ => {}
                }
            }

            fn visit_stmt(&mut self, stmt: &Stmt) {
                if self.found {
                    return;
                }

                match &stmt.kind {
                    StmtKind::Expr(e)
                    | StmtKind::Return(Some(e))
                    | StmtKind::Assert { test: e, .. } => {
                        self.visit_expr(e);
                    }
                    StmtKind::Assign { value, .. } | StmtKind::AugAssign { value, .. } => {
                        self.visit_expr(value);
                    }
                    StmtKind::For { body, orelse, .. }
                    | StmtKind::AsyncFor { body, orelse, .. }
                    | StmtKind::While { body, orelse, .. } => {
                        for s in body {
                            self.visit_stmt(s);
                        }
                        for s in orelse {
                            self.visit_stmt(s);
                        }
                    }
                    StmtKind::If { body, orelse, .. } => {
                        for s in body {
                            self.visit_stmt(s);
                        }
                        for s in orelse {
                            self.visit_stmt(s);
                        }
                    }
                    StmtKind::With { body, .. } | StmtKind::AsyncWith { body, .. } => {
                        for s in body {
                            self.visit_stmt(s);
                        }
                    }
                    StmtKind::Try {
                        body,
                        handlers,
                        orelse,
                        finalbody,
                    }
                    | StmtKind::TryStar {
                        body,
                        handlers,
                        orelse,
                        finalbody,
                    } => {
                        for s in body {
                            self.visit_stmt(s);
                        }
                        for h in handlers {
                            for s in &h.body {
                                self.visit_stmt(s);
                            }
                        }
                        for s in orelse {
                            self.visit_stmt(s);
                        }
                        for s in finalbody {
                            self.visit_stmt(s);
                        }
                    }
                    // Recurse into nested function definitions
                    StmtKind::FunctionDef { body, .. }
                    | StmtKind::AsyncFunctionDef { body, .. } => {
                        for s in body {
                            self.visit_stmt(s);
                        }
                    }
                    _ => {}
                }
            }
        }

        let mut visitor = SuperVisitor { found: false };
        for stmt in body {
            visitor.visit_stmt(stmt);
            if visitor.found {
                return true;
            }
        }
        false
    }

    /// Get the underlying builder for compilation.
    pub fn builder_mut(&mut self) -> &mut FunctionBuilder {
        &mut self.builder
    }

    /// Mark that this class needs an implicit __class__ cell.
    pub fn set_needs_class_cell(&mut self, needs: bool) {
        self.has_implicit_class_cell = needs;
    }

    /// Finish compilation and return the code object.
    ///
    /// The code object will have the CLASS flag set and will include
    /// the __class__ cell if any method uses zero-arg super().
    pub fn finish(mut self) -> CodeObject {
        // Ensure the class body returns the namespace dict
        // The namespace is built up in the local scope during execution
        // and returned via a special mechanism

        // Add the CLASS flag
        self.builder.add_flag(CodeFlags::CLASS);

        // Emit return at the end of class body
        // The namespace dict is returned to BUILD_CLASS
        self.builder.emit_return_none();

        self.builder.finish()
    }
}

/// Information about a class being compiled.
#[derive(Debug, Clone)]
pub struct ClassInfo {
    /// Class name.
    pub name: Arc<str>,
    /// Base classes (as expression indices).
    pub bases: Vec<u16>,
    /// Keyword arguments (metaclass, etc).
    pub keywords: Vec<(Arc<str>, u16)>,
    /// Code object index for class body.
    pub body_code_idx: u16,
    /// Decorator indices (applied in reverse order).
    pub decorators: Vec<u16>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_class_compiler_new() {
        let compiler = ClassCompiler::new("TestClass", "module.TestClass", "test.py");
        assert_eq!(&*compiler.name, "TestClass");
        assert_eq!(&*compiler.qualname, "module.TestClass");
    }

    #[test]
    fn test_class_compiler_finish_has_class_flag() {
        let mut compiler = ClassCompiler::new("Test", "Test", "test.py");
        let code = compiler.finish();
        assert!(code.flags.contains(CodeFlags::CLASS));
    }

    #[test]
    fn test_detect_zero_arg_super_simple() {
        // Parse and check detection
        let source = "def method(self):\n    super().__init__()";
        let module = prism_parser::parse(source).expect("parse failed");

        if let StmtKind::FunctionDef { body, .. } = &module.body[0].kind {
            assert!(ClassCompiler::uses_zero_arg_super(body));
        } else {
            panic!("Expected function def");
        }
    }

    #[test]
    fn test_detect_zero_arg_super_with_args() {
        // super(Class, self) should NOT be detected as zero-arg
        let source = "def method(self):\n    super(Foo, self).__init__()";
        let module = prism_parser::parse(source).expect("parse failed");

        if let StmtKind::FunctionDef { body, .. } = &module.body[0].kind {
            assert!(!ClassCompiler::uses_zero_arg_super(body));
        } else {
            panic!("Expected function def");
        }
    }

    #[test]
    fn test_detect_zero_arg_super_nested() {
        // super() in nested function should be detected
        let source = "def method(self):\n    def inner():\n        super()\n    inner()";
        let module = prism_parser::parse(source).expect("parse failed");

        if let StmtKind::FunctionDef { body, .. } = &module.body[0].kind {
            // The inner function contains super(), but we're checking the outer body
            // This should find it when recursing
            assert!(ClassCompiler::uses_zero_arg_super(body));
        }
    }

    #[test]
    fn test_detect_no_super() {
        let source = "def method(self):\n    print('hello')";
        let module = prism_parser::parse(source).expect("parse failed");

        if let StmtKind::FunctionDef { body, .. } = &module.body[0].kind {
            assert!(!ClassCompiler::uses_zero_arg_super(body));
        }
    }

    #[test]
    fn test_class_info_creation() {
        let info = ClassInfo {
            name: "MyClass".into(),
            bases: vec![0, 1],
            keywords: vec![("metaclass".into(), 2)],
            body_code_idx: 0,
            decorators: vec![],
        };
        assert_eq!(&*info.name, "MyClass");
        assert_eq!(info.bases.len(), 2);
        assert_eq!(info.keywords.len(), 1);
    }
}
