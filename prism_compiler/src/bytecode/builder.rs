//! Function builder for bytecode emission.
//!
//! The `FunctionBuilder` provides a high-level API for constructing bytecode
//! with automatic register allocation and label resolution.

use super::code_object::{CodeFlags, CodeObject, ExceptionEntry, LineTableEntry};
use super::instruction::{ConstIndex, Instruction, LocalSlot, Opcode, Register};
use prism_core::Value;
use std::collections::HashMap;
use std::sync::Arc;

/// A tuple of keyword argument names for CallKw instructions.
///
/// This is stored in the constant pool and referenced by CallKwEx.
/// Using a dedicated struct allows efficient lookup during argument binding.
#[derive(Debug, Clone)]
pub struct KwNamesTuple {
    /// Keyword argument names in call order.
    pub names: Box<[Arc<str>]>,
}

impl KwNamesTuple {
    /// Create a new keyword names tuple.
    pub fn new(names: Vec<Arc<str>>) -> Self {
        Self {
            names: names.into_boxed_slice(),
        }
    }

    /// Get the number of keyword arguments.
    #[inline]
    pub fn len(&self) -> usize {
        self.names.len()
    }

    /// Check if empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.names.is_empty()
    }

    /// Get a keyword name by index.
    #[inline]
    pub fn get(&self, index: usize) -> Option<&Arc<str>> {
        self.names.get(index)
    }

    /// Iterate over keyword names.
    #[inline]
    pub fn iter(&self) -> impl Iterator<Item = &Arc<str>> {
        self.names.iter()
    }
}

/// A label for jump targets.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Label(u32);

/// A forward reference to a label that needs patching.
#[derive(Debug)]
struct ForwardRef {
    /// Instruction index containing the jump.
    instruction_index: usize,
    /// The label being jumped to.
    label: Label,
}

/// Builder for constructing code objects.
///
/// This provides a high-level interface for:
/// - Emitting bytecode instructions
/// - Managing virtual registers
/// - Defining and resolving labels
/// - Managing constant and name pools
///
/// # Example
/// ```ignore
/// let mut builder = FunctionBuilder::new("add");
/// let r0 = builder.alloc_register(); // x
/// let r1 = builder.alloc_register(); // y
/// let r2 = builder.alloc_register(); // result
///
/// builder.emit_load_local(r0, 0); // load x
/// builder.emit_load_local(r1, 1); // load y
/// builder.emit_add(r2, r0, r1);   // r2 = x + y
/// builder.emit_return(r2);
///
/// let code = builder.finish();
/// ```
pub struct FunctionBuilder {
    /// Function name.
    name: Arc<str>,
    /// Qualified name.
    qualname: Arc<str>,
    /// Filename.
    filename: Arc<str>,
    /// First line number.
    first_lineno: u32,
    /// Current line number (for line table).
    current_line: u32,

    /// Emitted instructions.
    instructions: Vec<Instruction>,

    /// Constant pool.
    constants: Vec<Value>,
    /// Constant deduplication map.
    constant_map: HashMap<ConstantKey, ConstIndex>,

    /// Local variable names.
    locals: Vec<Arc<str>>,
    /// Local name to slot map.
    local_map: HashMap<Arc<str>, LocalSlot>,

    /// Global/attribute names.
    names: Vec<Arc<str>>,
    /// Name to index map.
    name_map: HashMap<Arc<str>, u16>,

    /// Free variable names.
    freevars: Vec<Arc<str>>,
    /// Cell variable names.
    cellvars: Vec<Arc<str>>,

    /// Number of parameters.
    arg_count: u16,
    posonlyarg_count: u16,
    kwonlyarg_count: u16,

    /// Code flags.
    flags: CodeFlags,

    /// Next register to allocate.
    next_register: u8,
    /// Maximum registers used (high water mark).
    max_registers: u8,
    /// Register free list for reuse.
    free_registers: Vec<Register>,

    /// Label counter.
    next_label: u32,
    /// Label to instruction index map.
    labels: HashMap<Label, usize>,
    /// Forward references that need patching.
    forward_refs: Vec<ForwardRef>,

    /// Line number table entries.
    line_table: Vec<LineTableEntry>,
    /// Start PC for current line.
    line_start_pc: u32,
    /// Exception table entries.
    exception_entries: Vec<ExceptionEntry>,
    /// Nested code objects (functions, classes defined within this code).
    nested_code_objects: Vec<Arc<CodeObject>>,
}

/// Key type for constant deduplication.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum ConstantKey {
    None,
    Bool(bool),
    Int(i64),
    /// Float bits for exact comparison.
    Float(u64),
    String(Arc<str>),
    /// Code object by name (simplified).
    Code(Arc<str>),
    /// Tuple of strings (for keyword argument names).
    KwNamesTuple(Box<[Arc<str>]>),
}

impl ConstantKey {
    fn from_value(value: &Value) -> Option<Self> {
        // We can only deduplicate simple types
        if value.is_none() {
            Some(ConstantKey::None)
        } else if let Some(b) = value.as_bool() {
            Some(ConstantKey::Bool(b))
        } else if let Some(i) = value.as_int() {
            Some(ConstantKey::Int(i))
        } else if let Some(f) = value.as_float() {
            Some(ConstantKey::Float(f.to_bits()))
        } else {
            // Can't deduplicate complex types (objects, lists, etc.)
            None
        }
    }
}

impl FunctionBuilder {
    /// Create a new function builder.
    pub fn new(name: impl Into<Arc<str>>) -> Self {
        let name = name.into();
        Self {
            qualname: name.clone(),
            name,
            filename: "<unknown>".into(),
            first_lineno: 1,
            current_line: 1,
            instructions: Vec::new(),
            constants: Vec::new(),
            constant_map: HashMap::new(),
            locals: Vec::new(),
            local_map: HashMap::new(),
            names: Vec::new(),
            name_map: HashMap::new(),
            freevars: Vec::new(),
            cellvars: Vec::new(),
            arg_count: 0,
            posonlyarg_count: 0,
            kwonlyarg_count: 0,
            flags: CodeFlags::NONE,
            next_register: 0,
            max_registers: 0,
            free_registers: Vec::new(),
            next_label: 0,
            labels: HashMap::new(),
            forward_refs: Vec::new(),
            line_table: Vec::new(),
            line_start_pc: 0,
            exception_entries: Vec::new(),
            nested_code_objects: Vec::new(),
        }
    }

    // =========================================================================
    // Configuration
    // =========================================================================

    /// Set the qualified name.
    pub fn set_qualname(&mut self, qualname: impl Into<Arc<str>>) {
        self.qualname = qualname.into();
    }

    /// Set the filename.
    pub fn set_filename(&mut self, filename: impl Into<Arc<str>>) {
        self.filename = filename.into();
    }

    /// Get the filename.
    #[inline]
    pub fn get_filename(&self) -> Arc<str> {
        self.filename.clone()
    }

    /// Set the first line number.
    pub fn set_first_lineno(&mut self, line: u32) {
        self.first_lineno = line;
        self.current_line = line;
    }

    /// Add a flag to the code object.
    pub fn add_flag(&mut self, flag: CodeFlags) {
        self.flags = self.flags | flag;
    }

    /// Set the current line number for subsequent instructions.
    pub fn set_line(&mut self, line: u32) {
        if line != self.current_line {
            // Record the previous line range
            let current_pc = self.instructions.len() as u32;
            if current_pc > self.line_start_pc {
                self.line_table.push(LineTableEntry {
                    start_pc: self.line_start_pc,
                    end_pc: current_pc,
                    line: self.current_line,
                });
            }
            self.current_line = line;
            self.line_start_pc = current_pc;
        }
    }

    /// Set the number of parameters.
    pub fn set_arg_count(&mut self, count: u16) {
        self.arg_count = count;
    }

    /// Set the number of positional-only parameters.
    pub fn set_posonlyarg_count(&mut self, count: u16) {
        self.posonlyarg_count = count;
    }

    /// Set the number of keyword-only parameters.
    pub fn set_kwonlyarg_count(&mut self, count: u16) {
        self.kwonlyarg_count = count;
    }

    /// Add code flags.
    pub fn add_flags(&mut self, flags: CodeFlags) {
        self.flags |= flags;
    }

    // =========================================================================
    // Register Management
    // =========================================================================

    /// Allocate a new virtual register.
    #[inline]
    pub fn alloc_register(&mut self) -> Register {
        // Try to reuse a freed register first
        if let Some(reg) = self.free_registers.pop() {
            return reg;
        }

        let reg = Register(self.next_register);
        self.next_register = self
            .next_register
            .checked_add(1)
            .expect("register overflow");
        self.max_registers = self.max_registers.max(self.next_register);
        reg
    }

    /// Free a register for reuse.
    #[inline]
    pub fn free_register(&mut self, reg: Register) {
        self.free_registers.push(reg);
    }

    /// Reserve registers for function parameters.
    pub fn reserve_parameters(&mut self, count: u16) {
        for _ in 0..count {
            self.alloc_register();
        }
    }

    /// Allocate a contiguous block of registers for function calls.
    ///
    /// This reserves `count` consecutive registers starting from a fresh base register
    /// (never from the free list) to avoid clobbering live registers.
    ///
    /// Call instructions use consecutive registers: [result, arg0, arg1, ...]
    /// This method ensures all those registers are properly reserved.
    ///
    /// # Returns
    /// The base register of the block. Registers [base..base+count) are reserved.
    #[inline]
    pub fn alloc_register_block(&mut self, count: u8) -> Register {
        // Allocate from `next_register` to guarantee contiguity
        // Do NOT reuse freed registers, as they may be scattered
        let base = Register(self.next_register);
        self.next_register = self
            .next_register
            .checked_add(count)
            .expect("register overflow");
        self.max_registers = self.max_registers.max(self.next_register);
        base
    }

    /// Free a contiguous block of registers (for cleanup after call).
    #[inline]
    pub fn free_register_block(&mut self, base: Register, count: u8) {
        // Add all registers in the block to the free list
        for i in 0..count {
            self.free_registers.push(Register(base.0 + i));
        }
    }

    /// Clear the free register list to prevent register reuse.
    ///
    /// This is used before compiling code that allocates a live register (like list_reg in comprehensions)
    /// followed by inner code that uses Call instructions. Call uses consecutive registers
    /// [dst, dst+1, ...], and if dst is reused from the free list at a low position,
    /// dst+1 could clobber the live register.
    ///
    /// By clearing the free list, we force all subsequent allocations to use fresh registers
    /// from `next_register`, which are guaranteed to be after any previously allocated live registers.
    #[inline]
    pub fn clear_free_registers(&mut self) {
        self.free_registers.clear();
    }

    // =========================================================================
    // Constant Pool
    // =========================================================================

    /// Add a constant and return its index.
    pub fn add_constant(&mut self, value: Value) -> ConstIndex {
        // Try deduplication for simple types
        if let Some(key) = ConstantKey::from_value(&value) {
            if let Some(&idx) = self.constant_map.get(&key) {
                return idx;
            }
            let idx = ConstIndex::new(self.constants.len() as u16);
            self.constants.push(value);
            self.constant_map.insert(key, idx);
            idx
        } else {
            // No deduplication for complex types
            let idx = ConstIndex::new(self.constants.len() as u16);
            self.constants.push(value);
            idx
        }
    }

    /// Add an integer constant.
    pub fn add_int(&mut self, value: i64) -> ConstIndex {
        self.add_constant(Value::int(value).unwrap_or_else(|| Value::none()))
    }

    /// Add a float constant.
    pub fn add_float(&mut self, value: f64) -> ConstIndex {
        self.add_constant(Value::float(value))
    }

    /// Add a string constant with automatic interning and deduplication.
    ///
    /// This method:
    /// 1. Interns the string using the global string interner for O(1) equality
    /// 2. Deduplicates identical strings in the constant pool
    /// 3. Returns an index suitable for LoadConst instruction
    ///
    /// # Performance
    ///
    /// - O(1) lookup for already-added strings via ConstantKey::String deduplication
    /// - Interned strings enable pointer equality at runtime
    ///
    /// # Example
    ///
    /// ```ignore
    /// let idx = builder.add_string("hello");
    /// builder.emit_load_const(dst, idx);
    /// ```
    pub fn add_string(&mut self, s: impl AsRef<str>) -> ConstIndex {
        let s_str = s.as_ref();
        let arc_str: Arc<str> = Arc::from(s_str);
        let key = ConstantKey::String(arc_str.clone());

        // Deduplication: return existing index if this string was already added
        if let Some(&idx) = self.constant_map.get(&key) {
            return idx;
        }

        // Intern the string for runtime O(1) equality checks
        let interned = prism_core::intern::intern(s_str);
        let value = Value::string(interned);

        let idx = ConstIndex::new(self.constants.len() as u16);
        self.constants.push(value);
        self.constant_map.insert(key, idx);
        idx
    }

    /// Add a nested code object constant.
    ///
    /// Code objects are stored in the constant pool for MakeFunction/MakeClosure
    /// to create function objects at runtime.
    ///
    /// Returns the constant index that can be used with MakeFunction/MakeClosure opcodes.
    pub fn add_code_object(&mut self, code: Arc<CodeObject>) -> u16 {
        // Store code object reference as a constant
        // We use the name for deduplication key
        let name = code.name.clone();
        let key = ConstantKey::Code(name);

        if let Some(&idx) = self.constant_map.get(&key) {
            return idx.0;
        }

        // Store the Arc<CodeObject> as an object pointer constant
        // At runtime, the VM will interpret this as a code object reference
        let code_ptr = Arc::into_raw(Arc::clone(&code)) as *const ();
        let idx = ConstIndex::new(self.constants.len() as u16);
        self.constants.push(Value::object_ptr(code_ptr));
        self.constant_map.insert(key, idx);

        // Store Arc in nested_code_objects for test accessibility
        self.nested_code_objects.push(code);

        idx.0
    }

    // =========================================================================
    // Local Variables
    // =========================================================================

    /// Define a local variable and return its slot.
    pub fn define_local(&mut self, name: impl Into<Arc<str>>) -> LocalSlot {
        let name = name.into();
        if let Some(&slot) = self.local_map.get(&name) {
            return slot;
        }
        let slot = LocalSlot::new(self.locals.len() as u16);
        self.local_map.insert(name.clone(), slot);
        self.locals.push(name);
        slot
    }

    /// Look up a local variable by name.
    pub fn lookup_local(&self, name: &str) -> Option<LocalSlot> {
        self.local_map.get(name).copied()
    }

    // =========================================================================
    // Names (globals, attributes)
    // =========================================================================

    /// Add a name and return its index.
    pub fn add_name(&mut self, name: impl Into<Arc<str>>) -> u16 {
        let name = name.into();
        if let Some(&idx) = self.name_map.get(&name) {
            return idx;
        }
        let idx = self.names.len() as u16;
        self.name_map.insert(name.clone(), idx);
        self.names.push(name);
        idx
    }

    // =========================================================================
    // Labels
    // =========================================================================

    /// Create a new label for a jump target.
    pub fn create_label(&mut self) -> Label {
        let label = Label(self.next_label);
        self.next_label += 1;
        label
    }

    /// Mark the current position as the target for a label.
    pub fn bind_label(&mut self, label: Label) {
        let pc = self.instructions.len();
        self.labels.insert(label, pc);
    }

    /// Get the current instruction offset (for relative jumps).
    pub fn current_offset(&self) -> usize {
        self.instructions.len()
    }

    // =========================================================================
    // Instruction Emission
    // =========================================================================

    /// Emit a raw instruction.
    #[inline]
    pub fn emit(&mut self, inst: Instruction) {
        self.instructions.push(inst);
    }

    /// Emit a NOP instruction.
    pub fn emit_nop(&mut self) {
        self.emit(Instruction::op(Opcode::Nop));
    }

    // --- Load/Store ---

    /// Load a constant into a register.
    pub fn emit_load_const(&mut self, dst: Register, idx: ConstIndex) {
        self.emit(Instruction::op_di(Opcode::LoadConst, dst, idx.0));
    }

    /// Load None into a register.
    pub fn emit_load_none(&mut self, dst: Register) {
        self.emit(Instruction::op_d(Opcode::LoadNone, dst));
    }

    /// Load True into a register.
    pub fn emit_load_true(&mut self, dst: Register) {
        self.emit(Instruction::op_d(Opcode::LoadTrue, dst));
    }

    /// Load False into a register.
    pub fn emit_load_false(&mut self, dst: Register) {
        self.emit(Instruction::op_d(Opcode::LoadFalse, dst));
    }

    /// Load a local variable into a register.
    pub fn emit_load_local(&mut self, dst: Register, slot: LocalSlot) {
        self.emit(Instruction::op_di(Opcode::LoadLocal, dst, slot.0));
    }

    /// Store a register into a local variable.
    pub fn emit_store_local(&mut self, slot: LocalSlot, src: Register) {
        self.emit(Instruction::op_di(Opcode::StoreLocal, src, slot.0));
    }

    /// Load a global variable into a register.
    pub fn emit_load_global(&mut self, dst: Register, name_idx: u16) {
        self.emit(Instruction::op_di(Opcode::LoadGlobal, dst, name_idx));
    }

    /// Store a register into a global variable.
    pub fn emit_store_global(&mut self, name_idx: u16, src: Register) {
        self.emit(Instruction::op_di(Opcode::StoreGlobal, src, name_idx));
    }

    // --- Closure Variables ---

    /// Load a closure variable (cell or free) into a register.
    ///
    /// Closure slot indices are determined by scope analysis:
    /// - Cell variables (captured by inner scopes) come first
    /// - Free variables (captured from outer scopes) follow
    #[inline]
    pub fn emit_load_closure(&mut self, dst: Register, slot: u16) {
        self.emit(Instruction::op_di(Opcode::LoadClosure, dst, slot));
    }

    /// Store a register into a closure variable.
    ///
    /// Uses Cell interior mutability for proper closure semantics.
    #[inline]
    pub fn emit_store_closure(&mut self, slot: u16, src: Register) {
        self.emit(Instruction::op_di(Opcode::StoreClosure, src, slot));
    }

    /// Delete (mark as unbound) a closure variable.
    ///
    /// Subsequent reads will raise UnboundLocalError.
    #[inline]
    pub fn emit_delete_closure(&mut self, slot: u16) {
        self.emit(Instruction::op_di(
            Opcode::DeleteClosure,
            Register::new(0),
            slot,
        ));
    }

    /// Add a cell variable (captured by inner scopes).
    ///
    /// Returns the closure slot index for this cell.
    pub fn add_cellvar(&mut self, name: impl Into<Arc<str>>) -> u16 {
        let name = name.into();
        let slot = self.cellvars.len() as u16;
        self.cellvars.push(name);
        slot
    }

    /// Add a free variable (captured from outer scope).
    ///
    /// Returns the closure slot index for this freevar.
    /// Note: Free variables are indexed after all cell variables.
    pub fn add_freevar(&mut self, name: impl Into<Arc<str>>) -> u16 {
        let name = name.into();
        // Free vars come after cell vars in the closure environment
        let slot = (self.cellvars.len() + self.freevars.len()) as u16;
        self.freevars.push(name);
        slot
    }

    /// Get the number of cell variables.
    #[inline]
    pub fn cellvar_count(&self) -> usize {
        self.cellvars.len()
    }

    /// Get the number of free variables.
    #[inline]
    pub fn freevar_count(&self) -> usize {
        self.freevars.len()
    }

    /// Check if this function has any closure variables.
    #[inline]
    pub fn has_closure(&self) -> bool {
        !self.cellvars.is_empty() || !self.freevars.is_empty()
    }

    // --- Class Construction ---

    /// Emit BUILD_CLASS instruction.
    ///
    /// Creates a class object from a class body code object and base classes.
    ///
    /// # Parameters
    /// - `dst`: Register to store the resulting class object
    /// - `code_idx`: Constant index of the class body code object
    /// - `base_count`: Number of base classes (stored in consecutive registers starting at dst+1)
    ///
    /// # Encoding
    /// Uses DstSrcSrc format: dst = class, src1 = code_idx as u8, src2 = base_count
    pub fn emit_build_class(&mut self, dst: Register, code_idx: u16, base_count: u8) {
        self.emit(Instruction::op_dss(
            Opcode::BuildClass,
            dst,
            Register::new(code_idx as u8), // code object index (low byte)
            Register::new(base_count),
        ));
    }

    /// Move value between registers.
    pub fn emit_move(&mut self, dst: Register, src: Register) {
        if dst != src {
            self.emit(Instruction::op_ds(Opcode::Move, dst, src));
        }
    }

    // --- Arithmetic ---

    /// Generic add: dst = src1 + src2.
    pub fn emit_add(&mut self, dst: Register, src1: Register, src2: Register) {
        self.emit(Instruction::op_dss(Opcode::Add, dst, src1, src2));
    }

    /// Generic subtract: dst = src1 - src2.
    pub fn emit_sub(&mut self, dst: Register, src1: Register, src2: Register) {
        self.emit(Instruction::op_dss(Opcode::Sub, dst, src1, src2));
    }

    /// Generic multiply: dst = src1 * src2.
    pub fn emit_mul(&mut self, dst: Register, src1: Register, src2: Register) {
        self.emit(Instruction::op_dss(Opcode::Mul, dst, src1, src2));
    }

    /// Generic true divide: dst = src1 / src2.
    pub fn emit_div(&mut self, dst: Register, src1: Register, src2: Register) {
        self.emit(Instruction::op_dss(Opcode::TrueDiv, dst, src1, src2));
    }

    /// Generic floor divide: dst = src1 // src2.
    pub fn emit_floor_div(&mut self, dst: Register, src1: Register, src2: Register) {
        self.emit(Instruction::op_dss(Opcode::FloorDiv, dst, src1, src2));
    }

    /// Generic modulo: dst = src1 % src2.
    pub fn emit_mod(&mut self, dst: Register, src1: Register, src2: Register) {
        self.emit(Instruction::op_dss(Opcode::Mod, dst, src1, src2));
    }

    /// Generic power: dst = src1 ** src2.
    pub fn emit_pow(&mut self, dst: Register, src1: Register, src2: Register) {
        self.emit(Instruction::op_dss(Opcode::Pow, dst, src1, src2));
    }

    /// Generic negate: dst = -src.
    pub fn emit_neg(&mut self, dst: Register, src: Register) {
        self.emit(Instruction::op_ds(Opcode::Neg, dst, src));
    }

    // --- Comparison ---

    /// Less than: dst = src1 < src2.
    pub fn emit_lt(&mut self, dst: Register, src1: Register, src2: Register) {
        self.emit(Instruction::op_dss(Opcode::Lt, dst, src1, src2));
    }

    /// Less than or equal: dst = src1 <= src2.
    pub fn emit_le(&mut self, dst: Register, src1: Register, src2: Register) {
        self.emit(Instruction::op_dss(Opcode::Le, dst, src1, src2));
    }

    /// Equal: dst = src1 == src2.
    pub fn emit_eq(&mut self, dst: Register, src1: Register, src2: Register) {
        self.emit(Instruction::op_dss(Opcode::Eq, dst, src1, src2));
    }

    /// Not equal: dst = src1 != src2.
    pub fn emit_ne(&mut self, dst: Register, src1: Register, src2: Register) {
        self.emit(Instruction::op_dss(Opcode::Ne, dst, src1, src2));
    }

    /// Greater than: dst = src1 > src2.
    pub fn emit_gt(&mut self, dst: Register, src1: Register, src2: Register) {
        self.emit(Instruction::op_dss(Opcode::Gt, dst, src1, src2));
    }

    /// Greater than or equal: dst = src1 >= src2.
    pub fn emit_ge(&mut self, dst: Register, src1: Register, src2: Register) {
        self.emit(Instruction::op_dss(Opcode::Ge, dst, src1, src2));
    }

    // --- Bitwise ---

    /// Bitwise and: dst = src1 & src2.
    pub fn emit_bitwise_and(&mut self, dst: Register, src1: Register, src2: Register) {
        self.emit(Instruction::op_dss(Opcode::BitwiseAnd, dst, src1, src2));
    }

    /// Bitwise or: dst = src1 | src2.
    pub fn emit_bitwise_or(&mut self, dst: Register, src1: Register, src2: Register) {
        self.emit(Instruction::op_dss(Opcode::BitwiseOr, dst, src1, src2));
    }

    /// Bitwise xor: dst = src1 ^ src2.
    pub fn emit_bitwise_xor(&mut self, dst: Register, src1: Register, src2: Register) {
        self.emit(Instruction::op_dss(Opcode::BitwiseXor, dst, src1, src2));
    }

    /// Bitwise not: dst = ~src.
    pub fn emit_bitwise_not(&mut self, dst: Register, src: Register) {
        self.emit(Instruction::op_ds(Opcode::BitwiseNot, dst, src));
    }

    /// Left shift: dst = src1 << src2.
    pub fn emit_shl(&mut self, dst: Register, src1: Register, src2: Register) {
        self.emit(Instruction::op_dss(Opcode::Shl, dst, src1, src2));
    }

    /// Right shift: dst = src1 >> src2.
    pub fn emit_shr(&mut self, dst: Register, src1: Register, src2: Register) {
        self.emit(Instruction::op_dss(Opcode::Shr, dst, src1, src2));
    }

    /// Logical not: dst = not src.
    pub fn emit_not(&mut self, dst: Register, src: Register) {
        self.emit(Instruction::op_ds(Opcode::Not, dst, src));
    }

    // --- Control Flow ---

    /// Return value from register.
    pub fn emit_return(&mut self, src: Register) {
        self.emit(Instruction::op_d(Opcode::Return, src));
    }

    /// Return None.
    pub fn emit_return_none(&mut self) {
        self.emit(Instruction::op(Opcode::ReturnNone));
    }

    /// Unconditional jump to label.
    pub fn emit_jump(&mut self, label: Label) {
        let inst_idx = self.instructions.len();
        // Emit placeholder, will be patched later
        self.emit(Instruction::op(Opcode::Jump));
        self.forward_refs.push(ForwardRef {
            instruction_index: inst_idx,
            label,
        });
    }

    /// Jump if register is falsy.
    pub fn emit_jump_if_false(&mut self, src: Register, label: Label) {
        let inst_idx = self.instructions.len();
        self.emit(Instruction::op_d(Opcode::JumpIfFalse, src));
        self.forward_refs.push(ForwardRef {
            instruction_index: inst_idx,
            label,
        });
    }

    /// Jump if register is truthy.
    pub fn emit_jump_if_true(&mut self, src: Register, label: Label) {
        let inst_idx = self.instructions.len();
        self.emit(Instruction::op_d(Opcode::JumpIfTrue, src));
        self.forward_refs.push(ForwardRef {
            instruction_index: inst_idx,
            label,
        });
    }

    // --- Object Operations ---

    /// Get attribute: dst = obj.attr.
    pub fn emit_get_attr(&mut self, dst: Register, obj: Register, name_idx: u16) {
        self.emit(Instruction::new(
            Opcode::GetAttr,
            dst.0,
            obj.0,
            (name_idx & 0xFF) as u8,
        ));
        // Note: GetAttr currently uses compact 8-bit name encoding.
        // Full 16-bit name indices require an extended instruction format.
    }

    /// Get item: dst = obj[key].
    pub fn emit_get_item(&mut self, dst: Register, obj: Register, key: Register) {
        self.emit(Instruction::op_dss(Opcode::GetItem, dst, obj, key));
    }

    /// Set item: obj[key] = value.
    pub fn emit_set_item(&mut self, obj: Register, key: Register, value: Register) {
        self.emit(Instruction::op_dss(Opcode::SetItem, obj, key, value));
    }

    // --- Function Calls ---

    /// Call function: dst = func(args...).
    /// Args should be in registers dst+1, dst+2, etc.
    pub fn emit_call(&mut self, dst: Register, func: Register, argc: u8) {
        self.emit(Instruction::new(Opcode::Call, dst.0, func.0, argc));
    }

    /// Load method for optimized method calls: dst = obj.method (with self).
    ///
    /// This is the first half of the LoadMethod/CallMethod optimization pair.
    /// It performs method lookup and stores the method in `dst` and `self` in `dst+1`.
    ///
    /// Register layout after LoadMethod:
    /// - [dst]: method/function object
    /// - [dst+1]: self instance (or NULL marker for unbound)
    ///
    /// # Arguments
    /// * `dst` - Register to store method (dst+1 gets self automatically)
    /// * `obj` - Register containing the object to look up method on
    /// * `name_idx` - Index into names table for the method name
    #[inline]
    pub fn emit_load_method(&mut self, dst: Register, obj: Register, name_idx: u16) {
        // LoadMethod uses DstSrcSrc format: dst, obj, name_idx (8-bit truncated)
        // For larger name indices, we'd need an extended encoding
        self.emit(Instruction::new(
            Opcode::LoadMethod,
            dst.0,
            obj.0,
            (name_idx & 0xFF) as u8,
        ));
    }

    /// Call method using result from LoadMethod: dst = method(self, args...).
    ///
    /// This is the second half of the LoadMethod/CallMethod optimization pair.
    /// It expects the method in `method_reg` and self in `method_reg+1`.
    ///
    /// Register layout expected (from LoadMethod):
    /// - [method_reg]: method/function object
    /// - [method_reg+1]: self instance
    /// - [method_reg+2..]: explicit arguments
    ///
    /// # Arguments
    /// * `dst` - Register to store the return value
    /// * `method_reg` - Register containing method (from LoadMethod)
    /// * `argc` - Number of explicit arguments (in method_reg+2, method_reg+3, etc.)
    #[inline]
    pub fn emit_call_method(&mut self, dst: Register, method_reg: Register, argc: u8) {
        self.emit(Instruction::new(
            Opcode::CallMethod,
            dst.0,
            method_reg.0,
            argc,
        ));
    }

    /// Call function with keyword arguments: dst = func(pos_args..., kw_args...).
    ///
    /// Uses a two-instruction sequence for encoding:
    /// - Instruction 1: [CallKw][dst][func][posargc]
    /// - Instruction 2: [CallKwEx][kwargc][kwnames_idx_lo][kwnames_idx_hi]
    ///
    /// Arguments layout in registers:
    /// - dst+1 .. dst+posargc: positional arguments
    /// - dst+posargc+1 .. dst+posargc+kwargc: keyword argument values
    ///
    /// Keyword names are stored as a tuple in the constant pool at kwnames_idx.
    pub fn emit_call_kw(
        &mut self,
        dst: Register,
        func: Register,
        posargc: u8,
        kwargc: u8,
        kwnames_idx: u16,
    ) {
        // First instruction: opcode, dst, func, posargc
        self.emit(Instruction::new(Opcode::CallKw, dst.0, func.0, posargc));
        // Second instruction: kwargc, kwnames_idx (split into two bytes)
        self.emit(Instruction::new(
            Opcode::CallKwEx,
            kwargc,
            (kwnames_idx & 0xFF) as u8,
            (kwnames_idx >> 8) as u8,
        ));
    }

    /// Add a tuple of keyword argument names to the constant pool.
    ///
    /// This is used for CallKw instructions to efficiently pass keyword names
    /// without dictionary allocation. Names are interned strings for fast comparison.
    ///
    /// Returns the constant pool index of the tuple.
    pub fn add_kwnames_tuple(&mut self, names: Vec<Arc<str>>) -> u16 {
        // Create deduplication key
        let key = ConstantKey::KwNamesTuple(names.clone().into_boxed_slice());

        if let Some(&idx) = self.constant_map.get(&key) {
            return idx.0;
        }

        // Create a tuple value containing the keyword names
        // For now, we use a packed representation stored as object pointer
        // The VM will interpret this as a keyword names tuple
        let tuple = KwNamesTuple::new(names);
        let tuple_ptr = Box::into_raw(Box::new(tuple)) as *const ();
        let idx = ConstIndex::new(self.constants.len() as u16);
        self.constants.push(Value::object_ptr(tuple_ptr));
        self.constant_map.insert(key, idx);
        idx.0
    }

    /// Call function with unpacked arguments: dst = func(*args_tuple, **kwargs_dict).
    ///
    /// Used when call site contains *args or **kwargs unpacking. Uses two instructions:
    /// - Instruction 1: [CallEx][dst][func][args_tuple_reg]
    /// - Instruction 2: [CallKwEx][kwargs_dict_reg][0][0] (or 0xFF for no kwargs)
    ///
    /// The args_tuple_reg should contain a tuple of positional arguments.
    /// The kwargs_dict_reg should contain a dict of keyword arguments (or None).
    pub fn emit_call_ex(
        &mut self,
        dst: Register,
        func: Register,
        args_tuple: Register,
        kwargs_dict: Option<Register>,
    ) {
        // First instruction: CallEx with args tuple register
        self.emit(Instruction::new(
            Opcode::CallEx,
            dst.0,
            func.0,
            args_tuple.0,
        ));
        // Second instruction as extension: kwargs dict register (0xFF = no kwargs)
        let kwargs_reg = kwargs_dict.map_or(0xFF, |r| r.0);
        self.emit(Instruction::new(Opcode::CallKwEx, kwargs_reg, 0, 0));
    }

    /// Build a tuple from multiple values/iterables with unpacking.
    ///
    /// This is used for combining static positional args with *args unpacking.
    /// The unpack_flags is a bitmap where bit i indicates src+i should be unpacked.
    /// Values are sourced from consecutive registers starting at `base`.
    ///
    /// Format: [BuildTupleUnpack][dst][base][count] + [extension with flags]
    pub fn emit_build_tuple_unpack(
        &mut self,
        dst: Register,
        base: Register,
        count: u8,
        unpack_flags: u32,
    ) {
        self.emit(Instruction::new(
            Opcode::BuildTupleUnpack,
            dst.0,
            base.0,
            count,
        ));
        // Extension instruction with unpack flags (lower 24 bits)
        self.emit(Instruction::new(
            Opcode::CallKwEx,
            (unpack_flags & 0xFF) as u8,
            ((unpack_flags >> 8) & 0xFF) as u8,
            ((unpack_flags >> 16) & 0xFF) as u8,
        ));
    }

    /// Build a dict from multiple values/mappings with unpacking.
    ///
    /// This is used for combining static keyword args with **kwargs unpacking.
    /// The unpack_flags is a bitmap where bit i indicates src+i should be unpacked.
    /// Values are sourced from consecutive registers starting at `base`.
    ///
    /// Format: [BuildDictUnpack][dst][base][count] + [extension with flags]
    pub fn emit_build_dict_unpack(
        &mut self,
        dst: Register,
        base: Register,
        count: u8,
        unpack_flags: u32,
    ) {
        self.emit(Instruction::new(
            Opcode::BuildDictUnpack,
            dst.0,
            base.0,
            count,
        ));
        // Extension instruction with unpack flags (lower 24 bits)
        self.emit(Instruction::new(
            Opcode::CallKwEx,
            (unpack_flags & 0xFF) as u8,
            ((unpack_flags >> 8) & 0xFF) as u8,
            ((unpack_flags >> 16) & 0xFF) as u8,
        ));
    }

    // --- Container Operations ---

    /// Build list from registers.
    pub fn emit_build_list(&mut self, dst: Register, start: Register, count: u8) {
        self.emit(Instruction::new(Opcode::BuildList, dst.0, start.0, count));
    }

    /// Build tuple from registers.
    pub fn emit_build_tuple(&mut self, dst: Register, start: Register, count: u8) {
        self.emit(Instruction::new(Opcode::BuildTuple, dst.0, start.0, count));
    }

    /// Get iterator: dst = iter(src).
    pub fn emit_get_iter(&mut self, dst: Register, src: Register) {
        self.emit(Instruction::op_ds(Opcode::GetIter, dst, src));
    }

    /// For iteration: dst = next(iter_src), jump to label on StopIteration.
    pub fn emit_for_iter(&mut self, dst: Register, iter_src: Register, label: Label) {
        let inst_idx = self.instructions.len();
        // Encode dst and iter_src; imm16 will be patched with jump offset
        self.emit(Instruction::op_ds(Opcode::ForIter, dst, iter_src));
        self.forward_refs.push(ForwardRef {
            instruction_index: inst_idx,
            label,
        });
    }

    /// End async for: check if register contains StopAsyncIteration, jump to label if so.
    ///
    /// This is used at the end of each async for iteration to check if the awaited
    /// result indicates StopAsyncIteration. If so, the exception is cleared and
    /// execution jumps to the label (typically the else clause or loop end).
    ///
    /// # Arguments
    /// * `src` - Register containing the awaited result to check
    /// * `label` - Label to jump to if StopAsyncIteration was raised
    pub fn emit_end_async_for(&mut self, src: Register, label: Label) {
        let inst_idx = self.instructions.len();
        // Emit EndAsyncFor with src register; imm16 will be patched with jump offset
        self.emit(Instruction::op_d(Opcode::EndAsyncFor, src));
        self.forward_refs.push(ForwardRef {
            instruction_index: inst_idx,
            label,
        });
    }

    // =========================================================================
    // Import Operations
    // =========================================================================

    /// Import a module by name index.
    ///
    /// Emits ImportName opcode: dst = import(names[name_idx])
    ///
    /// # Arguments
    /// * `dst` - Register to store the imported module object
    /// * `name_idx` - Index into the names table for the module name
    #[inline]
    pub fn emit_import_name(&mut self, dst: Register, name_idx: u16) {
        self.emit(Instruction::op_di(Opcode::ImportName, dst, name_idx));
    }

    /// Import an attribute from a module.
    ///
    /// Emits ImportFrom opcode: dst = from module import names[name_idx]
    ///
    /// # Arguments
    /// * `dst` - Register to store the imported attribute value
    /// * `module_reg` - Register containing the source module object
    /// * `name_idx` - Index into the names table for the attribute name
    ///
    /// # Instruction Encoding
    /// Uses extended format: opcode(8) | dst(8) | module_reg(8) | name_idx_lo(8)
    /// with name_idx_hi packed in a following instruction if needed.
    /// For simplicity, we use a compact encoding with src1=module_reg, imm8=name_idx_lo.
    #[inline]
    pub fn emit_import_from(&mut self, dst: Register, module_reg: Register, name_idx: u8) {
        // Compact encoding: dst | module_reg | name_idx (8-bit for now)
        self.emit(Instruction::new(
            Opcode::ImportFrom,
            dst.0,
            module_reg.0,
            name_idx,
        ));
    }

    /// Import all public names from a module.
    ///
    /// Emits ImportStar opcode: from module import *
    ///
    /// # Arguments
    /// * `dst` - Unused (set to 0), but required by instruction format
    /// * `module_reg` - Register containing the source module object
    ///
    /// Note: The VM handler will inject all public names from the module
    /// into the current scope's global namespace.
    #[inline]
    pub fn emit_import_star(&mut self, dst: Register, module_reg: Register) {
        self.emit(Instruction::op_ds(Opcode::ImportStar, dst, module_reg));
    }

    // =========================================================================
    // Exception Handling
    // =========================================================================

    /// Adds an exception entry to the exception table.
    ///
    /// This is used by the exception compiler to build the zero-cost exception
    /// table. Entries should be added in order of start_pc for efficient binary
    /// search during runtime exception handling.
    ///
    /// # Arguments
    /// * `entry` - The exception entry describing a try block and its handlers
    #[inline]
    pub fn add_exception_entry(&mut self, entry: ExceptionEntry) {
        self.exception_entries.push(entry);
    }

    /// Returns the current instruction count (program counter).
    ///
    /// This is used during exception compilation to record PC values for
    /// exception table entries.
    #[inline]
    pub fn current_pc(&self) -> u32 {
        self.instructions.len() as u32
    }

    /// Returns the current stack depth for exception handling.
    ///
    /// This is used to record the stack depth at try block entry for proper
    /// stack unwinding during exception handling.
    #[inline]
    pub fn current_stack_depth(&self) -> u8 {
        self.next_register
    }

    // =========================================================================
    // Finalization
    // =========================================================================

    /// Finish building and return the code object.
    pub fn finish(mut self) -> CodeObject {
        // Finalize line table
        let final_pc = self.instructions.len() as u32;
        if final_pc > self.line_start_pc {
            self.line_table.push(LineTableEntry {
                start_pc: self.line_start_pc,
                end_pc: final_pc,
                line: self.current_line,
            });
        }

        // Patch forward references
        for fwd in self.forward_refs {
            let target = self.labels.get(&fwd.label).expect("unbound label");
            let offset = (*target as i32) - (fwd.instruction_index as i32) - 1;

            // Replace instruction with patched version
            let old = self.instructions[fwd.instruction_index];
            let opcode = Opcode::from_u8(old.opcode()).unwrap();
            let dst = old.dst();

            // Encode offset as signed 16-bit
            let offset_u16 = offset as i16 as u16;

            // ForIter needs special handling: preserve src1 (iterator register)
            // Use the upper byte of offset_u16 for src1, lower byte for offset
            // Instruction format: [opcode:8][dst:8][src1:8][offset:8]
            let patched = if opcode == Opcode::ForIter {
                // ForIter encodes src1 in imm16 high byte, offset in low byte
                // For now, just encode the offset in src2 position (8 bits signed)
                let src1 = old.src1();
                let offset_i8 = offset as i8 as u8;
                Instruction::new(opcode, dst.0, src1.0, offset_i8)
            } else {
                // Standard jump: opcode, dst, imm16
                Instruction::op_di(opcode, dst, offset_u16)
            };

            self.instructions[fwd.instruction_index] = patched;
        }

        CodeObject {
            name: self.name,
            qualname: self.qualname,
            filename: self.filename,
            first_lineno: self.first_lineno,
            instructions: self.instructions.into_boxed_slice(),
            constants: self.constants.into_boxed_slice(),
            locals: self.locals.into_boxed_slice(),
            names: self.names.into_boxed_slice(),
            freevars: self.freevars.into_boxed_slice(),
            cellvars: self.cellvars.into_boxed_slice(),
            arg_count: self.arg_count,
            posonlyarg_count: self.posonlyarg_count,
            kwonlyarg_count: self.kwonlyarg_count,
            register_count: self.max_registers as u16,
            flags: self.flags,
            line_table: self.line_table.into_boxed_slice(),
            exception_table: self.exception_entries.into_boxed_slice(),
            nested_code_objects: self.nested_code_objects.into_boxed_slice(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_function() {
        let mut builder = FunctionBuilder::new("add");
        builder.set_filename("test.py");
        builder.set_arg_count(2);

        let x = builder.alloc_register();
        let y = builder.alloc_register();
        let result = builder.alloc_register();

        builder.emit_load_local(x, LocalSlot::new(0));
        builder.emit_load_local(y, LocalSlot::new(1));
        builder.emit_add(result, x, y);
        builder.emit_return(result);

        let code = builder.finish();

        assert_eq!(&*code.name, "add");
        assert_eq!(code.instructions.len(), 4);
        assert_eq!(code.register_count, 3);
    }

    #[test]
    fn test_constant_deduplication() {
        let mut builder = FunctionBuilder::new("test");

        let idx1 = builder.add_int(42);
        let idx2 = builder.add_int(42);
        let idx3 = builder.add_int(100);

        assert_eq!(idx1.0, idx2.0); // Same constant, same index
        assert_ne!(idx1.0, idx3.0); // Different constant, different index
    }

    #[test]
    fn test_labels() {
        let mut builder = FunctionBuilder::new("loop");

        let loop_start = builder.create_label();
        let loop_end = builder.create_label();

        let r0 = builder.alloc_register();

        builder.bind_label(loop_start);
        builder.emit_jump_if_false(r0, loop_end);
        builder.emit_nop();
        builder.emit_jump(loop_start);
        builder.bind_label(loop_end);
        builder.emit_return_none();

        let code = builder.finish();
        assert_eq!(code.instructions.len(), 4);
    }

    #[test]
    fn test_register_allocation() {
        let mut builder = FunctionBuilder::new("test");

        let r0 = builder.alloc_register();
        let r1 = builder.alloc_register();
        builder.free_register(r0);
        let r2 = builder.alloc_register(); // Should reuse r0

        assert_eq!(r0.0, 0);
        assert_eq!(r1.0, 1);
        assert_eq!(r2.0, 0); // Reused
    }

    #[test]
    fn test_emit_get_attr_uses_low_byte_of_name_index() {
        let mut builder = FunctionBuilder::new("attr");
        let dst = builder.alloc_register();
        let obj = builder.alloc_register();

        builder.emit_get_attr(dst, obj, 0x0123);
        let code = builder.finish();
        let inst = code.instructions[0];

        assert_eq!(inst.opcode(), Opcode::GetAttr as u8);
        assert_eq!(inst.src2().0, 0x23);
    }

    // =========================================================================
    // Closure Variable Tests
    // =========================================================================

    #[test]
    fn test_cellvar_registration() {
        let mut builder = FunctionBuilder::new("outer");

        assert!(!builder.has_closure());
        assert_eq!(builder.cellvar_count(), 0);

        let slot0 = builder.add_cellvar("x");
        let slot1 = builder.add_cellvar("y");
        let slot2 = builder.add_cellvar("z");

        assert_eq!(slot0, 0);
        assert_eq!(slot1, 1);
        assert_eq!(slot2, 2);
        assert_eq!(builder.cellvar_count(), 3);
        assert!(builder.has_closure());

        let code = builder.finish();
        assert_eq!(code.cellvars.len(), 3);
        assert_eq!(&*code.cellvars[0], "x");
        assert_eq!(&*code.cellvars[1], "y");
        assert_eq!(&*code.cellvars[2], "z");
    }

    #[test]
    fn test_freevar_registration() {
        let mut builder = FunctionBuilder::new("inner");

        let slot0 = builder.add_freevar("captured");
        let slot1 = builder.add_freevar("outer_var");

        assert_eq!(slot0, 0);
        assert_eq!(slot1, 1);
        assert_eq!(builder.freevar_count(), 2);
        assert!(builder.has_closure());

        let code = builder.finish();
        assert_eq!(code.freevars.len(), 2);
        assert_eq!(&*code.freevars[0], "captured");
        assert_eq!(&*code.freevars[1], "outer_var");
    }

    #[test]
    fn test_mixed_cell_and_free_vars() {
        let mut builder = FunctionBuilder::new("middle");

        let cell0 = builder.add_cellvar("local_captured");
        let cell1 = builder.add_cellvar("another");

        let free0 = builder.add_freevar("from_outer");
        let free1 = builder.add_freevar("also_outer");

        assert_eq!(cell0, 0);
        assert_eq!(cell1, 1);
        assert_eq!(free0, 2); // After 2 cell vars
        assert_eq!(free1, 3);

        assert_eq!(builder.cellvar_count(), 2);
        assert_eq!(builder.freevar_count(), 2);
    }

    #[test]
    fn test_emit_load_closure() {
        let mut builder = FunctionBuilder::new("test");
        let r0 = builder.alloc_register();

        builder.add_cellvar("x");
        builder.emit_load_closure(r0, 0);

        let code = builder.finish();
        assert_eq!(code.instructions.len(), 1);

        let inst = code.instructions[0];
        assert_eq!(inst.opcode(), Opcode::LoadClosure as u8);
        assert_eq!(inst.dst().0, r0.0);
        assert_eq!(inst.imm16(), 0);
    }

    #[test]
    fn test_emit_store_closure() {
        let mut builder = FunctionBuilder::new("test");
        let r0 = builder.alloc_register();

        builder.add_cellvar("x");
        builder.emit_store_closure(0, r0);

        let code = builder.finish();
        assert_eq!(code.instructions.len(), 1);

        let inst = code.instructions[0];
        assert_eq!(inst.opcode(), Opcode::StoreClosure as u8);
        assert_eq!(inst.imm16(), 0);
    }

    #[test]
    fn test_emit_delete_closure() {
        let mut builder = FunctionBuilder::new("test");

        builder.add_cellvar("x");
        builder.emit_delete_closure(0);

        let code = builder.finish();
        assert_eq!(code.instructions.len(), 1);

        let inst = code.instructions[0];
        assert_eq!(inst.opcode(), Opcode::DeleteClosure as u8);
        assert_eq!(inst.imm16(), 0);
    }

    #[test]
    fn test_closure_function_pattern() {
        let mut outer = FunctionBuilder::new("make_counter");
        outer.set_arg_count(1);

        let count_slot = outer.add_cellvar("count");
        assert_eq!(count_slot, 0);

        let r0 = outer.alloc_register();
        outer.emit_load_local(r0, LocalSlot::new(0));
        outer.emit_store_closure(count_slot, r0);
        outer.emit_return_none();

        let outer_code = outer.finish();
        assert_eq!(outer_code.cellvars.len(), 1);
        assert_eq!(&*outer_code.cellvars[0], "count");
        assert_eq!(outer_code.instructions.len(), 3);
    }

    #[test]
    fn test_closure_instruction_sequence() {
        let mut builder = FunctionBuilder::new("increment_closure");

        builder.add_cellvar("counter");
        let r0 = builder.alloc_register();
        let r1 = builder.alloc_register();

        builder.emit_load_closure(r0, 0);
        let one_idx = builder.add_int(1);
        builder.emit_load_const(r1, one_idx);
        builder.emit_add(r0, r0, r1);
        builder.emit_store_closure(0, r0);
        builder.emit_return(r0);

        let code = builder.finish();
        assert_eq!(code.instructions.len(), 5);

        assert_eq!(code.instructions[0].opcode(), Opcode::LoadClosure as u8);
        assert_eq!(code.instructions[1].opcode(), Opcode::LoadConst as u8);
        assert_eq!(code.instructions[2].opcode(), Opcode::Add as u8);
        assert_eq!(code.instructions[3].opcode(), Opcode::StoreClosure as u8);
        assert_eq!(code.instructions[4].opcode(), Opcode::Return as u8);
    }

    #[test]
    fn test_multiple_closure_slots() {
        let mut builder = FunctionBuilder::new("multi_closure");

        for i in 0..8 {
            let slot = builder.add_cellvar(format!("var{}", i));
            assert_eq!(slot, i as u16);
        }

        let r0 = builder.alloc_register();

        for i in 0..8u16 {
            builder.emit_load_closure(r0, i);
        }

        let code = builder.finish();
        assert_eq!(code.cellvars.len(), 8);
        assert_eq!(code.instructions.len(), 8);

        for i in 0..8 {
            let inst = code.instructions[i];
            assert_eq!(inst.opcode(), Opcode::LoadClosure as u8);
            assert_eq!(inst.imm16(), i as u16);
        }
    }

    #[test]
    fn test_no_closure_by_default() {
        let builder = FunctionBuilder::new("simple");
        assert!(!builder.has_closure());
        assert_eq!(builder.cellvar_count(), 0);
        assert_eq!(builder.freevar_count(), 0);

        let code = builder.finish();
        assert!(code.cellvars.is_empty());
        assert!(code.freevars.is_empty());
    }

    // =========================================================================
    // String Constant Tests
    // =========================================================================

    #[test]
    fn test_add_string_basic() {
        let mut builder = FunctionBuilder::new("test");
        let idx = builder.add_string("hello");
        assert_eq!(idx.0, 0);
    }

    #[test]
    fn test_add_string_deduplication() {
        let mut builder = FunctionBuilder::new("test");

        let idx1 = builder.add_string("hello");
        let idx2 = builder.add_string("hello");
        let idx3 = builder.add_string("world");

        // Same string should return same index
        assert_eq!(idx1.0, idx2.0);
        // Different strings should have different indices
        assert_ne!(idx1.0, idx3.0);
    }

    #[test]
    fn test_add_string_empty() {
        let mut builder = FunctionBuilder::new("test");
        let idx1 = builder.add_string("");
        let idx2 = builder.add_string("");

        assert_eq!(idx1.0, idx2.0);
    }

    #[test]
    fn test_add_string_unicode() {
        let mut builder = FunctionBuilder::new("test");

        let idx1 = builder.add_string("こんにちは");
        let idx2 = builder.add_string("こんにちは");
        let idx3 = builder.add_string("世界");

        assert_eq!(idx1.0, idx2.0);
        assert_ne!(idx1.0, idx3.0);
    }

    #[test]
    fn test_add_string_emoji() {
        let mut builder = FunctionBuilder::new("test");

        let idx1 = builder.add_string("🦀🐍💻");
        let idx2 = builder.add_string("🦀🐍💻");

        assert_eq!(idx1.0, idx2.0);
    }

    #[test]
    fn test_add_string_escape_sequences() {
        let mut builder = FunctionBuilder::new("test");

        let idx1 = builder.add_string("line1\nline2\ttab");
        let idx2 = builder.add_string("line1\nline2\ttab");
        let idx3 = builder.add_string("line1\\nline2\\ttab"); // Different: escaped

        assert_eq!(idx1.0, idx2.0);
        assert_ne!(idx1.0, idx3.0);
    }

    #[test]
    fn test_add_string_whitespace_significant() {
        let mut builder = FunctionBuilder::new("test");

        let idx1 = builder.add_string("hello");
        let idx2 = builder.add_string("hello ");
        let idx3 = builder.add_string(" hello");
        let idx4 = builder.add_string("hello");

        // Whitespace matters
        assert_ne!(idx1.0, idx2.0);
        assert_ne!(idx1.0, idx3.0);
        assert_ne!(idx2.0, idx3.0);
        // Same string should deduplicate
        assert_eq!(idx1.0, idx4.0);
    }

    #[test]
    fn test_add_string_case_sensitive() {
        let mut builder = FunctionBuilder::new("test");

        let idx1 = builder.add_string("Hello");
        let idx2 = builder.add_string("hello");
        let idx3 = builder.add_string("HELLO");

        assert_ne!(idx1.0, idx2.0);
        assert_ne!(idx1.0, idx3.0);
        assert_ne!(idx2.0, idx3.0);
    }

    #[test]
    fn test_add_string_long() {
        let mut builder = FunctionBuilder::new("test");

        let long_string = "x".repeat(10000);
        let idx1 = builder.add_string(&long_string);
        let idx2 = builder.add_string(&long_string);

        assert_eq!(idx1.0, idx2.0);
    }

    #[test]
    fn test_add_string_multiple_distinct() {
        let mut builder = FunctionBuilder::new("test");

        for i in 0..100 {
            let s = format!("string_{}", i);
            let idx = builder.add_string(&s);
            assert_eq!(idx.0, i as u16);
        }
    }

    #[test]
    fn test_add_string_with_load_const() {
        let mut builder = FunctionBuilder::new("test");
        let r0 = builder.alloc_register();

        let str_idx = builder.add_string("test_string");
        builder.emit_load_const(r0, str_idx);
        builder.emit_return_none();

        let code = builder.finish();
        assert_eq!(code.instructions.len(), 2);
        assert_eq!(code.constants.len(), 1);

        let inst = code.instructions[0];
        assert_eq!(inst.opcode(), Opcode::LoadConst as u8);
        assert_eq!(inst.dst().0, r0.0);
        assert_eq!(inst.imm16(), str_idx.0);
    }

    #[test]
    fn test_add_string_constant_value_is_string() {
        let mut builder = FunctionBuilder::new("test");
        builder.add_string("hello");

        let code = builder.finish();
        assert_eq!(code.constants.len(), 1);

        // The constant should be a string value
        let val = code.constants[0];
        assert!(val.is_string());
    }

    #[test]
    fn test_mixed_constant_types() {
        let mut builder = FunctionBuilder::new("test");

        let int_idx = builder.add_int(42);
        let str_idx = builder.add_string("hello");
        let float_idx = builder.add_float(3.14);
        let str_idx2 = builder.add_string("world");

        // All should have unique indices
        assert_ne!(int_idx.0, str_idx.0);
        assert_ne!(str_idx.0, float_idx.0);
        assert_ne!(float_idx.0, str_idx2.0);

        let code = builder.finish();
        assert_eq!(code.constants.len(), 4);
    }

    #[test]
    fn test_string_dedup_does_not_affect_other_types() {
        let mut builder = FunctionBuilder::new("test");

        // Add string "42" and int 42 - should not deduplicate
        let str_idx = builder.add_string("42");
        let int_idx = builder.add_int(42);

        assert_ne!(str_idx.0, int_idx.0);

        let code = builder.finish();
        assert_eq!(code.constants.len(), 2);
    }

    #[test]
    fn test_add_string_from_string_type() {
        let mut builder = FunctionBuilder::new("test");

        let owned_string = String::from("owned_string");
        let idx1 = builder.add_string(&owned_string);
        let idx2 = builder.add_string("owned_string");

        // Should deduplicate even when coming from different source types
        assert_eq!(idx1.0, idx2.0);
    }

    #[test]
    fn test_string_constant_pool_ordering() {
        let mut builder = FunctionBuilder::new("test");

        let idx_a = builder.add_string("aaa");
        let idx_b = builder.add_string("bbb");
        let idx_c = builder.add_string("ccc");

        assert_eq!(idx_a.0, 0);
        assert_eq!(idx_b.0, 1);
        assert_eq!(idx_c.0, 2);

        // Re-adding should return original indices
        assert_eq!(builder.add_string("bbb").0, 1);
        assert_eq!(builder.add_string("aaa").0, 0);
        assert_eq!(builder.add_string("ccc").0, 2);
    }
}
