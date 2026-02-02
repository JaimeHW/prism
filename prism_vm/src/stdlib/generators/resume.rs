//! Generator resume dispatch implementation.
//!
//! This module provides the computed-goto-style dispatch mechanism for
//! resuming generators at specific yield points. This is a critical
//! performance optimization that avoids switch statement overhead.
//!
//! # Architecture
//!
//! When a generator yields, it saves:
//! 1. The resume index (which yield point to return to)
//! 2. The live register values
//! 3. The instruction pointer
//!
//! When resumed, we:
//! 1. Restore the saved registers
//! 2. Look up the resume point in a dispatch table
//! 3. Jump directly to the appropriate code location
//!
//! # Performance Target
//!
//! Resume dispatch should take ~5 cycles on hot paths:
//! - 1 cycle: Load resume index
//! - 1 cycle: Bounds check
//! - 1 cycle: Table lookup
//! - 2 cycles: Indirect jump
//!
//! This is achieved through:
//! - Inline dispatch tables (no function calls)
//! - Branch prediction hints
//! - Careful memory layout

use super::object::GeneratorObject;
use super::state::GeneratorState;
use super::storage::LivenessMap;
use prism_core::Value;

// ============================================================================
// Resume Points
// ============================================================================

/// Maximum number of yield points in a single generator.
///
/// This limit exists because:
/// 1. Resume indices are stored as u32 in the header
/// 2. Larger tables hurt instruction cache locality
/// 3. Generators with more yield points are rare in practice
pub const MAX_RESUME_POINTS: usize = 65536;

/// A resume point descriptor.
///
/// This captures all information needed to resume execution at a
/// specific yield point in a generator.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ResumePoint {
    /// Instruction pointer to resume at.
    pub ip: u32,
    /// Liveness bitmap for this yield point.
    pub liveness: LivenessMap,
    /// Stack depth at this yield point (for frame restoration).
    pub stack_depth: u8,
    /// Whether this is from a `yield from` expression.
    pub is_yield_from: bool,
}

impl ResumePoint {
    /// Creates a new resume point.
    #[inline]
    pub const fn new(ip: u32, liveness: LivenessMap) -> Self {
        Self {
            ip,
            liveness,
            stack_depth: 0,
            is_yield_from: false,
        }
    }

    /// Creates a resume point with stack depth.
    #[inline]
    pub const fn with_depth(ip: u32, liveness: LivenessMap, stack_depth: u8) -> Self {
        Self {
            ip,
            liveness,
            stack_depth,
            is_yield_from: false,
        }
    }

    /// Creates a resume point for yield from.
    #[inline]
    pub const fn yield_from(ip: u32, liveness: LivenessMap) -> Self {
        Self {
            ip,
            liveness,
            stack_depth: 0,
            is_yield_from: true,
        }
    }
}

impl Default for ResumePoint {
    fn default() -> Self {
        Self::new(0, LivenessMap::empty())
    }
}

// ============================================================================
// Resume Table
// ============================================================================

/// A dispatch table for generator resume points.
///
/// This is compiled once per generator function and shared by all
/// instances of that generator. The table maps yield point indices
/// to resume point descriptors.
#[derive(Debug, Clone)]
pub struct ResumeTable {
    /// The resume points, indexed by yield point ID.
    points: Vec<ResumePoint>,
}

impl ResumeTable {
    /// Creates an empty resume table.
    #[inline]
    pub const fn new() -> Self {
        Self { points: Vec::new() }
    }

    /// Creates a resume table with the given capacity.
    #[inline]
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            points: Vec::with_capacity(capacity),
        }
    }

    /// Creates a resume table from a slice of points.
    pub fn from_points(points: &[ResumePoint]) -> Self {
        Self {
            points: points.to_vec(),
        }
    }

    /// Adds a resume point to the table.
    ///
    /// Returns the index of the added point.
    #[inline]
    pub fn add(&mut self, point: ResumePoint) -> u32 {
        let idx = self.points.len();
        debug_assert!(idx < MAX_RESUME_POINTS, "Too many resume points");
        self.points.push(point);
        idx as u32
    }

    /// Gets a resume point by index.
    #[inline]
    pub fn get(&self, index: u32) -> Option<&ResumePoint> {
        self.points.get(index as usize)
    }

    /// Gets a resume point by index without bounds checking.
    ///
    /// # Safety
    ///
    /// The caller must ensure that `index < self.len()`.
    #[inline]
    pub unsafe fn get_unchecked(&self, index: u32) -> &ResumePoint {
        // SAFETY: Caller ensures index is in bounds
        unsafe { self.points.get_unchecked(index as usize) }
    }

    /// Returns the number of resume points.
    #[inline]
    pub fn len(&self) -> usize {
        self.points.len()
    }

    /// Returns true if the table is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.points.is_empty()
    }
}

impl Default for ResumeTable {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Resume Action
// ============================================================================

/// The action to take after resuming a generator.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResumeAction {
    /// Execute from the given instruction pointer.
    Execute { ip: u32, liveness: LivenessMap },
    /// Generator is exhausted.
    Exhausted,
    /// Generator is already running.
    AlreadyRunning,
    /// Invalid resume index.
    InvalidIndex(u32),
}

impl ResumeAction {
    /// Returns true if execution should proceed.
    #[inline]
    pub fn should_execute(&self) -> bool {
        matches!(self, Self::Execute { .. })
    }

    /// Returns the instruction pointer if execution should proceed.
    #[inline]
    pub fn ip(&self) -> Option<u32> {
        match self {
            Self::Execute { ip, .. } => Some(*ip),
            _ => None,
        }
    }
}

// ============================================================================
// Resume Operations
// ============================================================================

/// Prepares a generator for resumption and returns the action to take.
///
/// This is the main entry point for the resume dispatch system.
/// It handles:
/// - State validation
/// - Resume table lookup
/// - Register restoration preparation
#[inline]
pub fn prepare_resume(generator: &GeneratorObject, table: &ResumeTable) -> ResumeAction {
    match generator.state() {
        GeneratorState::Created => {
            // First invocation - start from the beginning
            ResumeAction::Execute {
                ip: 0,
                liveness: LivenessMap::empty(),
            }
        }
        GeneratorState::Suspended => {
            // Resume at the saved yield point
            let resume_idx = generator.resume_index();

            match table.get(resume_idx) {
                Some(point) => ResumeAction::Execute {
                    ip: generator.ip(),
                    liveness: point.liveness,
                },
                None => ResumeAction::InvalidIndex(resume_idx),
            }
        }
        GeneratorState::Running => ResumeAction::AlreadyRunning,
        GeneratorState::Exhausted => ResumeAction::Exhausted,
    }
}

/// Restores a generator's saved state to a register file.
///
/// This should be called after `prepare_resume` returns `Execute`.
#[inline]
pub fn restore_generator_state(generator: &GeneratorObject, registers: &mut [Value; 256]) {
    generator.restore(registers);
}

/// Fast path for checking if a generator can be resumed.
///
/// Use this for quick validation before more expensive operations.
#[inline(always)]
pub fn can_resume(generator: &GeneratorObject) -> bool {
    generator.is_resumable()
}

/// Suspends a generator at a yield point.
///
/// This saves all necessary state for later resumption.
#[inline]
pub fn suspend_at_yield(
    generator: &mut GeneratorObject,
    ip: u32,
    resume_index: u32,
    registers: &[Value; 256],
    liveness: LivenessMap,
) {
    generator.suspend(ip, resume_index, registers, liveness);
}

/// Marks a generator as exhausted after it returns.
#[inline]
pub fn exhaust_generator(generator: &GeneratorObject) {
    generator.exhaust();
}

// ============================================================================
// Yield Point Builder
// ============================================================================

/// Builder for constructing resume tables during compilation.
///
/// This is used by the compiler to collect yield points and generate
/// the dispatch table.
#[derive(Debug)]
pub struct ResumeTableBuilder {
    table: ResumeTable,
}

impl ResumeTableBuilder {
    /// Creates a new builder.
    #[inline]
    pub fn new() -> Self {
        Self {
            table: ResumeTable::new(),
        }
    }

    /// Creates a builder with initial capacity.
    #[inline]
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            table: ResumeTable::with_capacity(capacity),
        }
    }

    /// Records a yield point.
    ///
    /// Returns the resume index to use for this yield.
    pub fn add_yield(&mut self, ip: u32, liveness: LivenessMap) -> u32 {
        self.table.add(ResumePoint::new(ip, liveness))
    }

    /// Records a yield point with stack depth.
    pub fn add_yield_with_depth(&mut self, ip: u32, liveness: LivenessMap, depth: u8) -> u32 {
        self.table.add(ResumePoint::with_depth(ip, liveness, depth))
    }

    /// Records a yield from point.
    pub fn add_yield_from(&mut self, ip: u32, liveness: LivenessMap) -> u32 {
        self.table.add(ResumePoint::yield_from(ip, liveness))
    }

    /// Finalizes and returns the resume table.
    #[inline]
    pub fn build(self) -> ResumeTable {
        self.table
    }

    /// Returns the current number of yield points.
    #[inline]
    pub fn len(&self) -> usize {
        self.table.len()
    }

    /// Returns true if no yield points have been recorded.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.table.is_empty()
    }
}

impl Default for ResumeTableBuilder {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use prism_compiler::bytecode::CodeObject;
    use std::sync::Arc;

    fn test_code() -> Arc<CodeObject> {
        Arc::new(CodeObject::new("test_generator", "<test>"))
    }

    // ════════════════════════════════════════════════════════════════════════
    // ResumePoint Tests
    // ════════════════════════════════════════════════════════════════════════

    #[test]
    fn test_resume_point_new() {
        let point = ResumePoint::new(100, LivenessMap::from_bits(0b101));
        assert_eq!(point.ip, 100);
        assert_eq!(point.liveness.bits(), 0b101);
        assert_eq!(point.stack_depth, 0);
        assert!(!point.is_yield_from);
    }

    #[test]
    fn test_resume_point_with_depth() {
        let point = ResumePoint::with_depth(200, LivenessMap::from_bits(0b11), 5);
        assert_eq!(point.ip, 200);
        assert_eq!(point.stack_depth, 5);
        assert!(!point.is_yield_from);
    }

    #[test]
    fn test_resume_point_yield_from() {
        let point = ResumePoint::yield_from(300, LivenessMap::from_bits(0b1));
        assert_eq!(point.ip, 300);
        assert!(point.is_yield_from);
    }

    #[test]
    fn test_resume_point_default() {
        let point = ResumePoint::default();
        assert_eq!(point.ip, 0);
        assert!(point.liveness.is_empty());
    }

    // ════════════════════════════════════════════════════════════════════════
    // ResumeTable Tests
    // ════════════════════════════════════════════════════════════════════════

    #[test]
    fn test_resume_table_add() {
        let mut table = ResumeTable::new();
        assert!(table.is_empty());

        let idx0 = table.add(ResumePoint::new(10, LivenessMap::from_bits(0b1)));
        let idx1 = table.add(ResumePoint::new(20, LivenessMap::from_bits(0b11)));

        assert_eq!(idx0, 0);
        assert_eq!(idx1, 1);
        assert_eq!(table.len(), 2);
    }

    #[test]
    fn test_resume_table_get() {
        let mut table = ResumeTable::new();
        table.add(ResumePoint::new(100, LivenessMap::from_bits(0b101)));
        table.add(ResumePoint::new(200, LivenessMap::from_bits(0b111)));

        assert_eq!(table.get(0).unwrap().ip, 100);
        assert_eq!(table.get(1).unwrap().ip, 200);
        assert!(table.get(2).is_none());
    }

    #[test]
    fn test_resume_table_get_unchecked() {
        let mut table = ResumeTable::new();
        table.add(ResumePoint::new(50, LivenessMap::from_bits(0)));

        unsafe {
            let point = table.get_unchecked(0);
            assert_eq!(point.ip, 50);
        }
    }

    #[test]
    fn test_resume_table_from_points() {
        let points = vec![
            ResumePoint::new(10, LivenessMap::from_bits(0b1)),
            ResumePoint::new(20, LivenessMap::from_bits(0b10)),
            ResumePoint::new(30, LivenessMap::from_bits(0b11)),
        ];

        let table = ResumeTable::from_points(&points);
        assert_eq!(table.len(), 3);
        assert_eq!(table.get(0).unwrap().ip, 10);
        assert_eq!(table.get(2).unwrap().ip, 30);
    }

    // ════════════════════════════════════════════════════════════════════════
    // ResumeAction Tests
    // ════════════════════════════════════════════════════════════════════════

    #[test]
    fn test_resume_action_execute() {
        let action = ResumeAction::Execute {
            ip: 100,
            liveness: LivenessMap::from_bits(0b101),
        };
        assert!(action.should_execute());
        assert_eq!(action.ip(), Some(100));
    }

    #[test]
    fn test_resume_action_exhausted() {
        let action = ResumeAction::Exhausted;
        assert!(!action.should_execute());
        assert!(action.ip().is_none());
    }

    #[test]
    fn test_resume_action_already_running() {
        let action = ResumeAction::AlreadyRunning;
        assert!(!action.should_execute());
    }

    #[test]
    fn test_resume_action_invalid_index() {
        let action = ResumeAction::InvalidIndex(99);
        assert!(!action.should_execute());
    }

    // ════════════════════════════════════════════════════════════════════════
    // Prepare Resume Tests
    // ════════════════════════════════════════════════════════════════════════

    #[test]
    fn test_prepare_resume_created() {
        let code = test_code();
        let generator = super::super::object::GeneratorObject::new(code);
        let table = ResumeTable::new();

        let action = prepare_resume(&generator, &table);
        assert!(action.should_execute());
        assert_eq!(action.ip(), Some(0));
    }

    #[test]
    fn test_prepare_resume_suspended() {
        let code = test_code();
        let mut generator = super::super::object::GeneratorObject::new(code);

        // Start the generator
        generator.try_start();

        // Suspend at yield point 0
        let liveness = LivenessMap::from_bits(0b1);
        let regs = [Value::none(); 256];
        generator.suspend(50, 0, &regs, liveness);

        // Create table with the resume point
        let mut table = ResumeTable::new();
        table.add(ResumePoint::new(50, liveness));

        let action = prepare_resume(&generator, &table);
        assert!(action.should_execute());
        assert_eq!(action.ip(), Some(50));
    }

    #[test]
    fn test_prepare_resume_running() {
        let code = test_code();
        let generator = super::super::object::GeneratorObject::new(code);
        generator.try_start();

        let table = ResumeTable::new();
        let action = prepare_resume(&generator, &table);
        assert_eq!(action, ResumeAction::AlreadyRunning);
    }

    #[test]
    fn test_prepare_resume_exhausted() {
        let code = test_code();
        let generator = super::super::object::GeneratorObject::new(code);
        generator.try_start();
        generator.exhaust();

        let table = ResumeTable::new();
        let action = prepare_resume(&generator, &table);
        assert_eq!(action, ResumeAction::Exhausted);
    }

    #[test]
    fn test_prepare_resume_invalid_index() {
        let code = test_code();
        let mut generator = super::super::object::GeneratorObject::new(code);

        // Start and suspend at index 5
        generator.try_start();
        let regs = [Value::none(); 256];
        generator.suspend(100, 5, &regs, LivenessMap::from_bits(0));

        // But table only has 2 points
        let mut table = ResumeTable::new();
        table.add(ResumePoint::new(10, LivenessMap::empty()));
        table.add(ResumePoint::new(20, LivenessMap::empty()));

        let action = prepare_resume(&generator, &table);
        assert_eq!(action, ResumeAction::InvalidIndex(5));
    }

    // ════════════════════════════════════════════════════════════════════════
    // ResumeTableBuilder Tests
    // ════════════════════════════════════════════════════════════════════════

    #[test]
    fn test_builder_add_yield() {
        let mut builder = ResumeTableBuilder::new();
        assert!(builder.is_empty());

        let idx0 = builder.add_yield(10, LivenessMap::from_bits(0b1));
        let idx1 = builder.add_yield(20, LivenessMap::from_bits(0b11));

        assert_eq!(idx0, 0);
        assert_eq!(idx1, 1);
        assert_eq!(builder.len(), 2);
    }

    #[test]
    fn test_builder_add_yield_with_depth() {
        let mut builder = ResumeTableBuilder::new();
        let idx = builder.add_yield_with_depth(100, LivenessMap::from_bits(0b101), 3);

        assert_eq!(idx, 0);

        let table = builder.build();
        assert_eq!(table.get(0).unwrap().stack_depth, 3);
    }

    #[test]
    fn test_builder_add_yield_from() {
        let mut builder = ResumeTableBuilder::new();
        let idx = builder.add_yield_from(50, LivenessMap::from_bits(0b1));

        assert_eq!(idx, 0);

        let table = builder.build();
        assert!(table.get(0).unwrap().is_yield_from);
    }

    #[test]
    fn test_builder_build() {
        let mut builder = ResumeTableBuilder::with_capacity(10);
        builder.add_yield(10, LivenessMap::from_bits(0b1));
        builder.add_yield_from(20, LivenessMap::from_bits(0b10));

        let table = builder.build();
        assert_eq!(table.len(), 2);
        assert!(!table.get(0).unwrap().is_yield_from);
        assert!(table.get(1).unwrap().is_yield_from);
    }

    // ════════════════════════════════════════════════════════════════════════
    // Helper Function Tests
    // ════════════════════════════════════════════════════════════════════════

    #[test]
    fn test_can_resume() {
        let code = test_code();
        let generator = super::super::object::GeneratorObject::new(code);

        assert!(can_resume(&generator));

        generator.try_start();
        assert!(!can_resume(&generator)); // Running

        generator.exhaust();
        assert!(!can_resume(&generator)); // Exhausted
    }

    #[test]
    fn test_suspend_at_yield() {
        let code = test_code();
        let mut generator = super::super::object::GeneratorObject::new(code);
        generator.try_start();

        let mut regs = [Value::none(); 256];
        regs[0] = Value::int(42).unwrap();
        regs[1] = Value::int(99).unwrap();

        let liveness = LivenessMap::from_bits(0b11);
        suspend_at_yield(&mut generator, 100, 1, &regs, liveness);

        assert_eq!(generator.state(), GeneratorState::Suspended);
        assert_eq!(generator.ip(), 100);
        assert_eq!(generator.resume_index(), 1);
    }

    #[test]
    fn test_exhaust_generator() {
        let code = test_code();
        let generator = super::super::object::GeneratorObject::new(code);
        generator.try_start();

        exhaust_generator(&generator);
        assert!(generator.is_exhausted());
    }

    #[test]
    fn test_restore_generator_state() {
        let code = test_code();
        let mut generator = super::super::object::GeneratorObject::new(code);
        generator.try_start();

        // Suspend with some values
        let mut regs = [Value::none(); 256];
        regs[0] = Value::int(42).unwrap();
        regs[2] = Value::int(100).unwrap();

        let liveness = LivenessMap::from_bits(0b101);
        suspend_at_yield(&mut generator, 50, 0, &regs, liveness);

        // Restore to new registers
        let mut new_regs = [Value::none(); 256];
        restore_generator_state(&generator, &mut new_regs);

        assert_eq!(new_regs[0].as_int().unwrap(), 42);
        assert_eq!(new_regs[2].as_int().unwrap(), 100);
    }
}
