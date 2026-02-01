//! AST to bytecode compiler.
use crate::CodeObject;

/// Bytecode compiler.
pub struct Compiler;

impl Compiler {
    /// Compile to bytecode (placeholder).
    pub fn compile() -> CodeObject {
        CodeObject {
            code: vec![],
            consts: vec![],
            names: vec![],
        }
    }
}
