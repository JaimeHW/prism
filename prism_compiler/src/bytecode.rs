//! Bytecode definitions.

/// Opcode enumeration matching Python 3.12.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum Opcode {
    /// No operation.
    Nop = 0,
    /// Pop top of stack.
    PopTop = 1,
    /// Push None.
    PushNull = 2,
    /// Load constant.
    LoadConst = 100,
    /// Load name.
    LoadName = 101,
    /// Store name.
    StoreName = 106,
    /// Return value.
    ReturnValue = 83,
    /// Binary add.
    BinaryAdd = 23,
}

/// A code object containing compiled bytecode.
#[derive(Debug, Clone)]
pub struct CodeObject {
    /// Bytecode instructions.
    pub code: Vec<u8>,
    /// Constants pool.
    pub consts: Vec<prism_core::Value>,
    /// Names used.
    pub names: Vec<String>,
}
