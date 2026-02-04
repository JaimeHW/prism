//! Register-based bytecode instruction definitions.
//!
//! This module defines the core instruction format for Prism's register-based bytecode.
//! All instructions are 32 bits wide for cache efficiency and predictable decoding.
//!
//! # Instruction Format
//!
//! ```text
//! ┌─────────┬─────────┬─────────┬─────────┐
//! │ opcode  │   dst   │  src1   │  src2   │
//! │ (8 bit) │ (8 bit) │ (8 bit) │ (8 bit) │
//! └─────────┴─────────┴─────────┴─────────┘
//! ```
//!
//! - `opcode`: Operation to perform (256 max)
//! - `dst`: Destination register or slot index
//! - `src1`: First source register or immediate index
//! - `src2`: Second source register or flags

use std::fmt;

/// A register index (0-255).
///
/// Registers are virtual and will be mapped to physical CPU registers
/// during JIT compilation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[repr(transparent)]
pub struct Register(pub u8);

impl Register {
    /// The accumulator register (r0), used for return values.
    pub const ACC: Register = Register(0);

    /// Create a new register with the given index.
    #[inline]
    pub const fn new(index: u8) -> Self {
        Register(index)
    }

    /// Get the register index.
    #[inline]
    pub const fn index(self) -> u8 {
        self.0
    }
}

impl fmt::Display for Register {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "r{}", self.0)
    }
}

/// A constant pool index.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[repr(transparent)]
pub struct ConstIndex(pub u16);

impl ConstIndex {
    /// Create a new constant index.
    #[inline]
    pub const fn new(index: u16) -> Self {
        ConstIndex(index)
    }
}

/// A local variable slot index.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[repr(transparent)]
pub struct LocalSlot(pub u16);

impl LocalSlot {
    /// Create a new local slot.
    #[inline]
    pub const fn new(index: u16) -> Self {
        LocalSlot(index)
    }
}

/// A 32-bit bytecode instruction.
///
/// The instruction is stored as a packed 32-bit value for cache efficiency.
/// All fields are accessed through methods that extract the relevant bits.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Default)]
#[repr(transparent)]
pub struct Instruction(u32);

impl Instruction {
    /// Create an instruction from raw opcode and operands.
    #[inline]
    pub const fn new(opcode: Opcode, dst: u8, src1: u8, src2: u8) -> Self {
        Instruction(
            ((opcode as u32) << 24) | ((dst as u32) << 16) | ((src1 as u32) << 8) | (src2 as u32),
        )
    }

    /// Create an instruction with only an opcode (no operands).
    #[inline]
    pub const fn op(opcode: Opcode) -> Self {
        Self::new(opcode, 0, 0, 0)
    }

    /// Create an instruction with opcode and destination register.
    #[inline]
    pub const fn op_d(opcode: Opcode, dst: Register) -> Self {
        Self::new(opcode, dst.0, 0, 0)
    }

    /// Create an instruction with opcode, destination, and one source.
    #[inline]
    pub const fn op_ds(opcode: Opcode, dst: Register, src: Register) -> Self {
        Self::new(opcode, dst.0, src.0, 0)
    }

    /// Create an instruction with opcode, destination, and two sources.
    #[inline]
    pub const fn op_dss(opcode: Opcode, dst: Register, src1: Register, src2: Register) -> Self {
        Self::new(opcode, dst.0, src1.0, src2.0)
    }

    /// Create an instruction with a 16-bit immediate in src1:src2.
    #[inline]
    pub const fn op_di(opcode: Opcode, dst: Register, imm16: u16) -> Self {
        Self::new(opcode, dst.0, (imm16 >> 8) as u8, imm16 as u8)
    }

    /// Get the opcode.
    #[inline]
    pub const fn opcode(self) -> u8 {
        (self.0 >> 24) as u8
    }

    /// Get the destination register index.
    #[inline]
    pub const fn dst(self) -> Register {
        Register(((self.0 >> 16) & 0xFF) as u8)
    }

    /// Get the first source register index.
    #[inline]
    pub const fn src1(self) -> Register {
        Register(((self.0 >> 8) & 0xFF) as u8)
    }

    /// Get the second source register index.
    #[inline]
    pub const fn src2(self) -> Register {
        Register((self.0 & 0xFF) as u8)
    }

    /// Get a 16-bit immediate from src1:src2.
    #[inline]
    pub const fn imm16(self) -> u16 {
        (self.0 & 0xFFFF) as u16
    }

    /// Get the raw 32-bit value.
    #[inline]
    pub const fn raw(self) -> u32 {
        self.0
    }

    /// Create from raw 32-bit value.
    #[inline]
    pub const fn from_raw(raw: u32) -> Self {
        Instruction(raw)
    }
}

impl fmt::Debug for Instruction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Instruction({:02x}, {}, {}, {})",
            self.opcode(),
            self.dst(),
            self.src1(),
            self.src2()
        )
    }
}

impl fmt::Display for Instruction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Some(op) = Opcode::from_u8(self.opcode()) {
            write!(f, "{:?}", op)?;
            // Format based on opcode category
            match op.format() {
                InstructionFormat::NoOp => {}
                InstructionFormat::Dst => write!(f, " {}", self.dst())?,
                InstructionFormat::DstSrc => write!(f, " {}, {}", self.dst(), self.src1())?,
                InstructionFormat::DstSrcSrc => {
                    write!(f, " {}, {}, {}", self.dst(), self.src1(), self.src2())?
                }
                InstructionFormat::DstImm16 => write!(f, " {}, #{}", self.dst(), self.imm16())?,
                InstructionFormat::Imm16 => write!(f, " #{}", self.imm16())?,
            }
            Ok(())
        } else {
            write!(f, "INVALID({:08x})", self.0)
        }
    }
}

/// Instruction format categories for disassembly.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InstructionFormat {
    /// No operands (e.g., Nop, ReturnNone).
    NoOp,
    /// Destination register only (e.g., LoadNone).
    Dst,
    /// Destination and source register (e.g., Move, Negate).
    DstSrc,
    /// Destination and two source registers (e.g., Add, Sub).
    DstSrcSrc,
    /// Destination and 16-bit immediate (e.g., LoadConst).
    DstImm16,
    /// 16-bit immediate only (e.g., Jump).
    Imm16,
}

/// Bytecode opcodes for the register-based VM.
///
/// Opcodes are organized by category:
/// - 0x00-0x0F: Control flow
/// - 0x10-0x1F: Load/Store
/// - 0x20-0x2F: Integer arithmetic
/// - 0x30-0x3F: Float arithmetic
/// - 0x40-0x4F: Comparison
/// - 0x50-0x5F: Logical/bitwise
/// - 0x60-0x6F: Object operations
/// - 0x70-0x7F: Function calls
/// - 0x80-0x8F: Container operations
/// - 0x90-0xFF: Reserved for future use
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum Opcode {
    // =========================================================================
    // Control Flow (0x00-0x0F)
    // =========================================================================
    /// No operation.
    Nop = 0x00,
    /// Return value in register.
    Return = 0x01,
    /// Return None.
    ReturnNone = 0x02,
    /// Unconditional jump (signed 16-bit offset).
    Jump = 0x03,
    /// Jump if register is falsy.
    JumpIfFalse = 0x04,
    /// Jump if register is truthy.
    JumpIfTrue = 0x05,
    /// Jump if register is None.
    JumpIfNone = 0x06,
    /// Jump if register is not None.
    JumpIfNotNone = 0x07,
    /// Pop exception handler.
    PopExceptHandler = 0x08,
    /// Raise exception from register.
    Raise = 0x09,
    /// Re-raise current exception.
    Reraise = 0x0A,
    /// End finally block.
    EndFinally = 0x0B,
    /// Yield value (generator).
    Yield = 0x0C,
    /// Yield from (generator delegation).
    YieldFrom = 0x0D,
    /// Check if exception matches type: dst = isinstance(exc, type).
    ExceptionMatch = 0x0E,
    /// Load current exception into register: dst = current_exception.
    LoadException = 0x0F,

    // =========================================================================
    // Load/Store (0x10-0x1F)
    // =========================================================================
    /// Load constant: dst = consts[imm16].
    LoadConst = 0x10,
    /// Load None: dst = None.
    LoadNone = 0x11,
    /// Load True: dst = True.
    LoadTrue = 0x12,
    /// Load False: dst = False.
    LoadFalse = 0x13,
    /// Load local: dst = locals[imm16].
    LoadLocal = 0x14,
    /// Store local: locals[imm16] = src.
    StoreLocal = 0x15,
    /// Load closure variable: dst = closure[imm16].
    LoadClosure = 0x16,
    /// Store closure variable: closure[imm16] = src.
    StoreClosure = 0x17,
    /// Load global: dst = globals[name_idx].
    LoadGlobal = 0x18,
    /// Store global: globals[name_idx] = src.
    StoreGlobal = 0x19,
    /// Delete local: del locals[imm16].
    DeleteLocal = 0x1A,
    /// Delete global: del globals[name_idx].
    DeleteGlobal = 0x1B,
    /// Move register: dst = src.
    Move = 0x1C,
    /// Delete closure variable: del closure[imm16].
    DeleteClosure = 0x1D,

    // =========================================================================
    // Integer Arithmetic (0x20-0x2F)
    // =========================================================================
    /// Integer add: dst = src1 + src2.
    AddInt = 0x20,
    /// Integer subtract: dst = src1 - src2.
    SubInt = 0x21,
    /// Integer multiply: dst = src1 * src2.
    MulInt = 0x22,
    /// Floor divide: dst = src1 // src2.
    FloorDivInt = 0x23,
    /// Modulo: dst = src1 % src2.
    ModInt = 0x24,
    /// Power: dst = src1 ** src2.
    PowInt = 0x25,
    /// Negate: dst = -src.
    NegInt = 0x26,
    /// Positive: dst = +src.
    PosInt = 0x27,

    // =========================================================================
    // Float Arithmetic (0x30-0x3F)
    // =========================================================================
    /// Float add: dst = src1 + src2.
    AddFloat = 0x30,
    /// Float subtract: dst = src1 - src2.
    SubFloat = 0x31,
    /// Float multiply: dst = src1 * src2.
    MulFloat = 0x32,
    /// Float divide: dst = src1 / src2.
    DivFloat = 0x33,
    /// Float floor divide: dst = src1 // src2.
    FloorDivFloat = 0x34,
    /// Float modulo: dst = src1 % src2.
    ModFloat = 0x35,
    /// Float power: dst = src1 ** src2.
    PowFloat = 0x36,
    /// Float negate: dst = -src.
    NegFloat = 0x37,

    // =========================================================================
    // Generic Arithmetic (0x38-0x3F) - polymorphic, slower
    // =========================================================================
    /// Generic add: dst = src1 + src2.
    Add = 0x38,
    /// Generic subtract: dst = src1 - src2.
    Sub = 0x39,
    /// Generic multiply: dst = src1 * src2.
    Mul = 0x3A,
    /// Generic true divide: dst = src1 / src2.
    TrueDiv = 0x3B,
    /// Generic floor divide: dst = src1 // src2.
    FloorDiv = 0x3C,
    /// Generic modulo: dst = src1 % src2.
    Mod = 0x3D,
    /// Generic power: dst = src1 ** src2.
    Pow = 0x3E,
    /// Generic negate: dst = -src.
    Neg = 0x3F,

    // =========================================================================
    // Comparison (0x40-0x4F)
    // =========================================================================
    /// Less than: dst = src1 < src2.
    Lt = 0x40,
    /// Less than or equal: dst = src1 <= src2.
    Le = 0x41,
    /// Equal: dst = src1 == src2.
    Eq = 0x42,
    /// Not equal: dst = src1 != src2.
    Ne = 0x43,
    /// Greater than: dst = src1 > src2.
    Gt = 0x44,
    /// Greater than or equal: dst = src1 >= src2.
    Ge = 0x45,
    /// Identity: dst = src1 is src2.
    Is = 0x46,
    /// Not identity: dst = src1 is not src2.
    IsNot = 0x47,
    /// Membership: dst = src1 in src2.
    In = 0x48,
    /// Not membership: dst = src1 not in src2.
    NotIn = 0x49,

    // =========================================================================
    // Logical/Bitwise (0x50-0x5F)
    // =========================================================================
    /// Bitwise and: dst = src1 & src2.
    BitwiseAnd = 0x50,
    /// Bitwise or: dst = src1 | src2.
    BitwiseOr = 0x51,
    /// Bitwise xor: dst = src1 ^ src2.
    BitwiseXor = 0x52,
    /// Bitwise not: dst = ~src.
    BitwiseNot = 0x53,
    /// Left shift: dst = src1 << src2.
    Shl = 0x54,
    /// Right shift: dst = src1 >> src2.
    Shr = 0x55,
    /// Logical not: dst = not src.
    Not = 0x57,

    // =========================================================================
    // Object Operations (0x60-0x6F)
    // =========================================================================
    /// Get attribute: dst = src.attr[name_idx].
    GetAttr = 0x60,
    /// Set attribute: src1.attr[name_idx] = src2.
    SetAttr = 0x61,
    /// Delete attribute: del src.attr[name_idx].
    DelAttr = 0x62,
    /// Get item: dst = src1[src2].
    GetItem = 0x63,
    /// Set item: src1[src2] = src3 (dst=src1, src1=src2, src2=src3).
    SetItem = 0x64,
    /// Delete item: del src1[src2].
    DelItem = 0x65,
    /// Get iterator: dst = iter(src).
    GetIter = 0x66,
    /// Get next from iterator: dst = next(src), jumps on StopIteration.
    ForIter = 0x67,
    /// Get length: dst = len(src).
    Len = 0x68,
    /// Check if callable: dst = callable(src).
    IsCallable = 0x69,
    /// Build class: dst = class body in src1, name_idx in imm16.
    /// Bases are in registers starting at dst+1, count in src2.
    BuildClass = 0x6A,
    /// Load method for super(): dst = super().method lookup.
    LoadMethod = 0x6B,

    // =========================================================================
    // Function Calls (0x70-0x7F)
    // =========================================================================
    /// Call function: dst = src(args...).
    /// src1 = function, src2 = argc, args in r(dst+1)..r(dst+argc).
    Call = 0x70,
    /// Call with keyword args.
    /// src1 = function, src2 = argc, kwargs follow.
    CallKw = 0x71,
    /// Call method: dst = src1.method(args...).
    CallMethod = 0x72,
    /// Tail call (reuse frame).
    TailCall = 0x73,
    /// Make function from code object.
    MakeFunction = 0x74,
    /// Make closure with captured variables.
    MakeClosure = 0x75,
    /// CallKw extension: [kwargc][kwnames_idx_lo][kwnames_idx_hi].
    /// Always follows CallKw. Contains keyword count and constant pool index.
    CallKwEx = 0x76,
    /// Call function with unpacked args: dst = func(*args_tuple, **kwargs_dict).
    /// src1 = function, src2 = args_tuple register.
    /// Extension byte contains kwargs_dict register (or 0xFF for no kwargs).
    CallEx = 0x77,
    /// Build tuple from multiple iterables with unpacking.
    /// dst = (*src1, *src2, ...) for count iterables starting at src1.
    /// src2 = count of source registers. Extension byte has unpack flags.
    BuildTupleUnpack = 0x78,
    /// Build dict from multiple mappings with unpacking.
    /// dst = {**src1, **src2, ...} for count mappings starting at src1.
    /// src2 = count of source registers. Extension byte has merge flags.
    BuildDictUnpack = 0x79,

    // =========================================================================
    // Container Operations (0x80-0x8F)
    // =========================================================================
    /// Build list: dst = [r(src1)..r(src1+src2)].
    BuildList = 0x80,
    /// Build tuple: dst = (r(src1)..r(src1+src2)).
    BuildTuple = 0x81,
    /// Build set: dst = {r(src1)..r(src1+src2)}.
    BuildSet = 0x82,
    /// Build dict: dst = {} with src2 key-value pairs starting at src1.
    BuildDict = 0x83,
    /// Build string: dst = "".join(r(src1)..r(src1+src2)).
    BuildString = 0x84,
    /// List append: src1.append(src2).
    ListAppend = 0x85,
    /// Set add: src1.add(src2).
    SetAdd = 0x86,
    /// Dict set: src1[dst] = src2 (uses dst as key register).
    DictSet = 0x87,
    /// Unpack sequence: r(dst)..r(dst+src2) = unpack(src1).
    UnpackSequence = 0x88,
    /// Unpack with star: unpack with *rest.
    UnpackEx = 0x89,
    /// Build slice: dst = slice(src1, src2).
    BuildSlice = 0x8A,

    // =========================================================================
    // Import (0x90-0x9F)
    // =========================================================================
    /// Import module: dst = import(name_idx).
    ImportName = 0x90,
    /// Import from: dst = from module import name.
    ImportFrom = 0x91,
    /// Import star: from module import *.
    ImportStar = 0x92,

    // =========================================================================
    // Extended Exception Handling (0x93-0x9F)
    // =========================================================================
    /// Raise exception with chained cause: raise exc from cause.
    /// Format: DstSrc (dst=exc_reg, src=cause_reg).
    /// Sets exception.__cause__ = cause and __suppress_context__ = True.
    RaiseFrom = 0x93,
    /// Push current exception info to exc_info stack for finally preservation.
    /// Used at finally block entry to preserve pending exception state.
    PushExcInfo = 0x94,
    /// Pop exception info from exc_info stack.
    /// Restores exception state after finally block executes.
    PopExcInfo = 0x95,
    /// Check if there's a pending exception (for finally block logic).
    /// dst = True if exception is pending, False otherwise.
    HasExcInfo = 0x96,
    /// Clear exception state after handler processes exception successfully.
    /// Used when an except block catches an exception and handles it.
    /// Clears active exception and sets state to Normal.
    ClearException = 0xAB,

    // =========================================================================
    // With Statement / Context Managers (0x97-0x9F)
    // =========================================================================
    /// Prepare context manager: calls __enter__ and stores (mgr, __exit__).
    /// dst = __enter__() result, src = context manager expression.
    /// Internally stores (mgr, __exit__) for cleanup on corresponding stack slot.
    BeforeWith = 0x97,
    /// Normal exit from with block: calls __exit__(None, None, None).
    /// dst = result of __exit__, src = context manager slot.
    ExitWith = 0x98,
    /// Exception cleanup from with block: calls __exit__(exc_type, exc_val, exc_tb).
    /// dst = result of __exit__ (True suppresses exception), src = context manager slot.
    WithCleanup = 0x99,

    // =========================================================================
    // Pattern Matching (0x9A-0x9F) - PEP 634 Match Statement Support
    // =========================================================================
    /// Match class pattern: dst = isinstance(src1, src2).
    /// Used for class patterns like `case Point(x, y):`.
    /// Returns True if src1 is an instance of class src2.
    MatchClass = 0x9A,
    /// Match mapping pattern: dst = is_mapping(src).
    /// Returns True if src supports the mapping protocol (__getitem__ and keys()).
    /// Used for patterns like `case {"key": value}:`.
    MatchMapping = 0x9B,
    /// Match sequence pattern: dst = is_sequence(src).
    /// Returns True if src is a sequence (has __len__ and __getitem__) but NOT str/bytes.
    /// Used for patterns like `case [a, b, c]:`.
    MatchSequence = 0x9C,
    /// Match keys: Extract values from mapping by key tuple.
    /// dst = tuple of values, src1 = mapping, src2 = tuple of keys.
    /// Jumps to fail if any key is missing.
    MatchKeys = 0x9D,
    /// Copy dict without keys: dst = {k:v for k,v in src1.items() if k not in src2}.
    /// src1 = mapping, src2 = tuple of keys to exclude.
    /// Used for **rest capture in mapping patterns.
    CopyDictWithoutKeys = 0x9E,
    /// Get __match_args__: dst = getattr(type(src), '__match_args__', ()).
    /// Used to determine positional attributes for class patterns.
    GetMatchArgs = 0x9F,

    // =========================================================================
    // Coroutine/Async Operations (0xA0-0xAF) - PEP 492/525/530 Support
    // =========================================================================
    /// Convert object to awaitable for use in `await` expression.
    /// PEP 492 semantics:
    /// 1. If coroutine/async_generator → return as-is (fast path)
    /// 2. If generator with CO_ITERABLE_COROUTINE → return as-is
    /// 3. If has __await__ → call __await__() and verify iterator result
    /// 4. Otherwise → raise TypeError
    GetAwaitable = 0xA0,
    /// Get async iterator: dst = src.__aiter__().
    /// Used for `async for` loops to obtain the async iterator.
    /// Raises TypeError if __aiter__ is not defined.
    GetAIter = 0xA1,
    /// Get next from async iterator: dst = src.__anext__().
    /// Returns an awaitable that yields the next value.
    /// Used in `async for` loop iteration.
    GetANext = 0xA2,
    /// Handle StopAsyncIteration in async for loop.
    /// Checks if the current exception is StopAsyncIteration.
    /// If so, clears exception and jumps to offset in imm16.
    /// Otherwise, re-raises the exception.
    EndAsyncFor = 0xA3,
    /// Send value to coroutine/generator: dst = src1.send(src2).
    /// Implements the send protocol for coroutines and generators.
    /// src1 = generator/coroutine, src2 = value to send.
    Send = 0xA4,
}

impl Opcode {
    /// Convert from u8, returning None if invalid.
    #[inline]
    pub fn from_u8(value: u8) -> Option<Self> {
        // Safety: we check the value is valid
        match value {
            0x00 => Some(Opcode::Nop),
            0x01 => Some(Opcode::Return),
            0x02 => Some(Opcode::ReturnNone),
            0x03 => Some(Opcode::Jump),
            0x04 => Some(Opcode::JumpIfFalse),
            0x05 => Some(Opcode::JumpIfTrue),
            0x06 => Some(Opcode::JumpIfNone),
            0x07 => Some(Opcode::JumpIfNotNone),
            0x08 => Some(Opcode::PopExceptHandler),
            0x09 => Some(Opcode::Raise),
            0x0A => Some(Opcode::Reraise),
            0x0B => Some(Opcode::EndFinally),
            0x0C => Some(Opcode::Yield),
            0x0D => Some(Opcode::YieldFrom),
            0x0E => Some(Opcode::ExceptionMatch),
            0x0F => Some(Opcode::LoadException),

            0x10 => Some(Opcode::LoadConst),
            0x11 => Some(Opcode::LoadNone),
            0x12 => Some(Opcode::LoadTrue),
            0x13 => Some(Opcode::LoadFalse),
            0x14 => Some(Opcode::LoadLocal),
            0x15 => Some(Opcode::StoreLocal),
            0x16 => Some(Opcode::LoadClosure),
            0x17 => Some(Opcode::StoreClosure),
            0x18 => Some(Opcode::LoadGlobal),
            0x19 => Some(Opcode::StoreGlobal),
            0x1A => Some(Opcode::DeleteLocal),
            0x1B => Some(Opcode::DeleteGlobal),
            0x1C => Some(Opcode::Move),
            0x1D => Some(Opcode::DeleteClosure),

            0x20 => Some(Opcode::AddInt),
            0x21 => Some(Opcode::SubInt),
            0x22 => Some(Opcode::MulInt),
            0x23 => Some(Opcode::FloorDivInt),
            0x24 => Some(Opcode::ModInt),
            0x25 => Some(Opcode::PowInt),
            0x26 => Some(Opcode::NegInt),
            0x27 => Some(Opcode::PosInt),

            0x30 => Some(Opcode::AddFloat),
            0x31 => Some(Opcode::SubFloat),
            0x32 => Some(Opcode::MulFloat),
            0x33 => Some(Opcode::DivFloat),
            0x34 => Some(Opcode::FloorDivFloat),
            0x35 => Some(Opcode::ModFloat),
            0x36 => Some(Opcode::PowFloat),
            0x37 => Some(Opcode::NegFloat),

            0x38 => Some(Opcode::Add),
            0x39 => Some(Opcode::Sub),
            0x3A => Some(Opcode::Mul),
            0x3B => Some(Opcode::TrueDiv),
            0x3C => Some(Opcode::FloorDiv),
            0x3D => Some(Opcode::Mod),
            0x3E => Some(Opcode::Pow),
            0x3F => Some(Opcode::Neg),

            0x40 => Some(Opcode::Lt),
            0x41 => Some(Opcode::Le),
            0x42 => Some(Opcode::Eq),
            0x43 => Some(Opcode::Ne),
            0x44 => Some(Opcode::Gt),
            0x45 => Some(Opcode::Ge),
            0x46 => Some(Opcode::Is),
            0x47 => Some(Opcode::IsNot),
            0x48 => Some(Opcode::In),
            0x49 => Some(Opcode::NotIn),

            0x50 => Some(Opcode::BitwiseAnd),
            0x51 => Some(Opcode::BitwiseOr),
            0x52 => Some(Opcode::BitwiseXor),
            0x53 => Some(Opcode::BitwiseNot),
            0x54 => Some(Opcode::Shl),
            0x55 => Some(Opcode::Shr),
            0x57 => Some(Opcode::Not),

            0x60 => Some(Opcode::GetAttr),
            0x61 => Some(Opcode::SetAttr),
            0x62 => Some(Opcode::DelAttr),
            0x63 => Some(Opcode::GetItem),
            0x64 => Some(Opcode::SetItem),
            0x65 => Some(Opcode::DelItem),
            0x66 => Some(Opcode::GetIter),
            0x67 => Some(Opcode::ForIter),
            0x68 => Some(Opcode::Len),
            0x69 => Some(Opcode::IsCallable),
            0x6A => Some(Opcode::BuildClass),
            0x6B => Some(Opcode::LoadMethod),

            0x70 => Some(Opcode::Call),
            0x71 => Some(Opcode::CallKw),
            0x72 => Some(Opcode::CallMethod),
            0x73 => Some(Opcode::TailCall),
            0x74 => Some(Opcode::MakeFunction),
            0x75 => Some(Opcode::MakeClosure),
            0x76 => Some(Opcode::CallKwEx),
            0x77 => Some(Opcode::CallEx),
            0x78 => Some(Opcode::BuildTupleUnpack),
            0x79 => Some(Opcode::BuildDictUnpack),

            0x80 => Some(Opcode::BuildList),
            0x81 => Some(Opcode::BuildTuple),
            0x82 => Some(Opcode::BuildSet),
            0x83 => Some(Opcode::BuildDict),
            0x84 => Some(Opcode::BuildString),
            0x85 => Some(Opcode::ListAppend),
            0x86 => Some(Opcode::SetAdd),
            0x87 => Some(Opcode::DictSet),
            0x88 => Some(Opcode::UnpackSequence),
            0x89 => Some(Opcode::UnpackEx),
            0x8A => Some(Opcode::BuildSlice),

            0x90 => Some(Opcode::ImportName),
            0x91 => Some(Opcode::ImportFrom),
            0x92 => Some(Opcode::ImportStar),

            0x93 => Some(Opcode::RaiseFrom),
            0x94 => Some(Opcode::PushExcInfo),
            0x95 => Some(Opcode::PopExcInfo),
            0x96 => Some(Opcode::HasExcInfo),
            0xAB => Some(Opcode::ClearException),

            0x97 => Some(Opcode::BeforeWith),
            0x98 => Some(Opcode::ExitWith),
            0x99 => Some(Opcode::WithCleanup),

            0x9A => Some(Opcode::MatchClass),
            0x9B => Some(Opcode::MatchMapping),
            0x9C => Some(Opcode::MatchSequence),
            0x9D => Some(Opcode::MatchKeys),
            0x9E => Some(Opcode::CopyDictWithoutKeys),
            0x9F => Some(Opcode::GetMatchArgs),

            0xA0 => Some(Opcode::GetAwaitable),
            0xA1 => Some(Opcode::GetAIter),
            0xA2 => Some(Opcode::GetANext),
            0xA3 => Some(Opcode::EndAsyncFor),
            0xA4 => Some(Opcode::Send),

            _ => None,
        }
    }

    /// Get the instruction format for this opcode.
    #[inline]
    pub const fn format(self) -> InstructionFormat {
        use InstructionFormat::*;
        use Opcode::*;

        match self {
            // No operands
            Nop | ReturnNone | Reraise | EndFinally => NoOp,

            // Destination only
            LoadNone | LoadTrue | LoadFalse => Dst,

            // Jump/return with source
            Return | Raise | Yield | YieldFrom | ExceptionMatch => DstSrc,
            LoadException => Dst,

            // Jumps with 16-bit offset
            Jump => Imm16,
            JumpIfFalse | JumpIfTrue | JumpIfNone | JumpIfNotNone => DstImm16,

            // Load/store with 16-bit index
            LoadConst | LoadLocal | StoreLocal | LoadClosure | StoreClosure | LoadGlobal
            | StoreGlobal | DeleteLocal | DeleteGlobal | DeleteClosure => DstImm16,

            // Move
            Move => DstSrc,

            // Unary operations
            NegInt | PosInt | NegFloat | Neg | BitwiseNot | Not | GetIter | Len | IsCallable => {
                DstSrc
            }

            // Binary operations
            AddInt | SubInt | MulInt | FloorDivInt | ModInt | PowInt | AddFloat | SubFloat
            | MulFloat | DivFloat | FloorDivFloat | ModFloat | PowFloat | Add | Sub | Mul
            | TrueDiv | FloorDiv | Mod | Pow | Lt | Le | Eq | Ne | Gt | Ge | Is | IsNot | In
            | NotIn | BitwiseAnd | BitwiseOr | BitwiseXor | Shl | Shr | GetItem | SetItem
            | DelItem | BuildSlice => DstSrcSrc,

            // Object ops with name index
            GetAttr | SetAttr | DelAttr => DstSrcSrc,
            ForIter => DstImm16,
            PopExceptHandler => Imm16,

            // Calls
            Call | CallKw | CallMethod | TailCall => DstSrcSrc,
            MakeFunction | MakeClosure => DstImm16,
            CallKwEx => DstSrcSrc, // kwargc, kwnames_idx_lo, kwnames_idx_hi
            CallEx => DstSrcSrc,   // dst, func, args_tuple (extension: kwargs_dict)
            BuildTupleUnpack | BuildDictUnpack => DstSrcSrc, // dst, base_reg, count

            // Class operations
            BuildClass => DstSrcSrc, // dst = class, src1 = body code, src2 = base count
            LoadMethod => DstSrcSrc, // dst = method, src1 = object, src2 = name_idx

            // Container ops
            BuildList | BuildTuple | BuildSet | BuildDict | BuildString | UnpackSequence
            | UnpackEx => DstSrcSrc,
            ListAppend | SetAdd | DictSet => DstSrcSrc,

            // Import
            ImportName | ImportFrom => DstImm16,
            ImportStar => DstSrc,

            // Extended exception handling
            RaiseFrom => DstSrc, // dst = exc_reg, src = cause_reg
            PushExcInfo | PopExcInfo | ClearException => NoOp, // No operands, operates on exception stack
            HasExcInfo => Dst,                                 // dst = bool result

            // With statement / context managers
            BeforeWith | ExitWith | WithCleanup => DstSrc, // dst = result, src = context manager

            // Pattern matching (PEP 634)
            MatchClass => DstSrcSrc, // dst = bool, src1 = subject, src2 = class
            MatchMapping | MatchSequence => DstSrc, // dst = bool, src = subject
            MatchKeys => DstSrcSrc,  // dst = values tuple, src1 = mapping, src2 = keys tuple
            CopyDictWithoutKeys => DstSrcSrc, // dst = new dict, src1 = mapping, src2 = keys to exclude
            GetMatchArgs => DstSrc,           // dst = __match_args__ tuple, src = subject

            // Coroutine/async (PEP 492/525/530)
            GetAwaitable | GetAIter | GetANext => DstSrc, // dst = result, src = object
            EndAsyncFor => DstImm16,                      // dst = value, imm16 = jump offset
            Send => DstSrcSrc,                            // dst = result, src1 = gen, src2 = value
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_instruction_encoding() {
        let inst = Instruction::new(Opcode::Add, 5, 10, 15);
        assert_eq!(inst.opcode(), Opcode::Add as u8);
        assert_eq!(inst.dst(), Register(5));
        assert_eq!(inst.src1(), Register(10));
        assert_eq!(inst.src2(), Register(15));
    }

    #[test]
    fn test_instruction_imm16() {
        let inst = Instruction::op_di(Opcode::LoadConst, Register(3), 0x1234);
        assert_eq!(inst.opcode(), Opcode::LoadConst as u8);
        assert_eq!(inst.dst(), Register(3));
        assert_eq!(inst.imm16(), 0x1234);
    }

    #[test]
    fn test_instruction_size() {
        assert_eq!(std::mem::size_of::<Instruction>(), 4);
    }

    #[test]
    fn test_opcode_from_u8() {
        assert_eq!(Opcode::from_u8(0x00), Some(Opcode::Nop));
        assert_eq!(Opcode::from_u8(0x38), Some(Opcode::Add));
        assert_eq!(Opcode::from_u8(0xFF), None);
    }

    #[test]
    fn test_instruction_display() {
        let add = Instruction::op_dss(Opcode::Add, Register(0), Register(1), Register(2));
        assert!(add.to_string().contains("Add"));

        let load = Instruction::op_di(Opcode::LoadConst, Register(5), 42);
        assert!(load.to_string().contains("LoadConst"));
    }

    #[test]
    fn test_register_display() {
        assert_eq!(Register(0).to_string(), "r0");
        assert_eq!(Register(255).to_string(), "r255");
    }
}
