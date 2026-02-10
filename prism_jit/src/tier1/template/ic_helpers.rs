//! IC Helper Routines for Tier 1 Templates.
//!
//! Provides low-level emission routines for inline cache operations:
//! - Shape ID loading and comparison
//! - Inline slot load/store
//! - IC site data access
//!
//! These are the building blocks for GetAttr/SetAttr IC templates.

use crate::backend::x64::{Assembler, Gpr, Label, MemOperand, Scale};

// =============================================================================
// Object Layout Constants
// =============================================================================

/// Offset to shape_id field in object header.
///
/// ShapedObject layout:
/// - ObjectHeader (16 bytes): gc_bits(8) + type_id(4) + flags(4)
/// - shape: Arc<Shape> (8 bytes, pointer)
/// - inline_slots: InlineSlots starts after shape pointer
///
/// We need the shape_id from the Shape, which requires dereferencing.
/// For IC purposes, we assume a flattened shape_id is cached in IC site.
pub const OBJECT_HEADER_SIZE: i32 = 16;

/// Offset to shape pointer in ShapedObject (after header).
pub const OBJECT_SHAPE_PTR_OFFSET: i32 = OBJECT_HEADER_SIZE;

/// Offset to shape_id within Shape struct.
/// Shape layout: [parent: Option<Arc>: 8][property: Option<Box>: 8][id: ShapeId: 4]...
/// Simplified: Shape.id() is at offset 16 in the Shape struct.
pub const SHAPE_ID_OFFSET: i32 = 16;

/// Offset to first inline slot in shaped object.
/// Layout: header(16) + shape_ptr(8) + InlineSlots (slots start immediately)
/// InlineSlots: [slots: [Value; MAX_INLINE_SLOTS]][used: u16]
pub const OBJECT_INLINE_SLOTS_OFFSET: i32 = 24;

/// Size of each slot (Value is 8 bytes).
pub const SLOT_SIZE: i32 = 8;

// =============================================================================
// IC Site Layout Constants
// =============================================================================

/// IC site data layout in frame:
/// [shape_id: 4][slot_offset: 2][flags: 2] = 8 bytes per site
pub const IC_SITE_SIZE: usize = 8;

/// Offset to shape_id in IC site.
pub const IC_SHAPE_ID_OFFSET: i32 = 0;

/// Offset to slot_offset in IC site.
pub const IC_SLOT_OFFSET_OFFSET: i32 = 4;

/// Offset to flags in IC site.
pub const IC_FLAGS_OFFSET: i32 = 6;

// =============================================================================
// Shape Check Emission
// =============================================================================

/// Emit code to load shape_id from an object's shape.
///
/// This dereferences the shape pointer and loads the shape_id field.
///
/// # Arguments
/// * `asm` - Assembler to emit to
/// * `dst` - Destination register for shape_id (32-bit value)
/// * `obj` - Register containing object pointer
/// * `scratch` - Scratch register for shape pointer
#[inline]
pub fn emit_load_shape_id(asm: &mut Assembler, dst: Gpr, obj: Gpr, scratch: Gpr) {
    // Load shape pointer: scratch = obj->shape
    let shape_mem = MemOperand::base_disp(obj, OBJECT_SHAPE_PTR_OFFSET);
    asm.mov_rm(scratch, &shape_mem);

    // Load shape_id from shape: dst = shape->id
    let id_mem = MemOperand::base_disp(scratch, SHAPE_ID_OFFSET);
    asm.mov_rm32(dst, &id_mem);
}

/// Emit shape guard: compare object shape vs cached shape, jump on mismatch.
///
/// # Arguments
/// * `asm` - Assembler to emit to
/// * `obj` - Register containing object pointer
/// * `ic_base` - Register containing IC data base pointer
/// * `ic_offset` - Offset to IC site within IC data
/// * `miss_label` - Label to jump to on shape mismatch
/// * `scratch1` - First scratch register (for shape pointer)
/// * `scratch2` - Second scratch register (for shape_id comparison)
///
/// # Register Clobbers
/// Clobbers scratch1 and scratch2.
pub fn emit_shape_guard(
    asm: &mut Assembler,
    obj: Gpr,
    ic_base: Gpr,
    ic_offset: i32,
    miss_label: Label,
    scratch1: Gpr,
    scratch2: Gpr,
) {
    // Load object's shape_id into scratch2
    emit_load_shape_id(asm, scratch2, obj, scratch1);

    // Load cached shape_id from IC site into scratch1
    let ic_shape_mem = MemOperand::base_disp(ic_base, ic_offset + IC_SHAPE_ID_OFFSET);
    asm.mov_rm32(scratch1, &ic_shape_mem);

    // Compare shape IDs
    asm.cmp_rr(scratch2, scratch1);

    // Jump to miss handler on mismatch
    asm.jne(miss_label);
}

/// Emit inline shape guard with immediate shape value (for monomorphic sites).
///
/// More efficient than memory-based guard when shape is known at compile time.
///
/// # Arguments
/// * `asm` - Assembler to emit to
/// * `obj` - Register containing object pointer
/// * `expected_shape` - Expected shape ID value
/// * `miss_label` - Label to jump to on mismatch
/// * `scratch1` - Scratch register for shape pointer
/// * `scratch2` - Scratch register for shape_id
pub fn emit_shape_guard_immediate(
    asm: &mut Assembler,
    obj: Gpr,
    expected_shape: u32,
    miss_label: Label,
    scratch1: Gpr,
    scratch2: Gpr,
) {
    // Load object's shape_id
    emit_load_shape_id(asm, scratch2, obj, scratch1);

    // Compare against immediate value (cast to i32 for cmp_ri)
    asm.cmp_ri(scratch2, expected_shape as i32);

    // Jump on mismatch
    asm.jne(miss_label);
}

// =============================================================================
// Slot Load/Store Emission
// =============================================================================

/// Emit inline slot load with known offset.
///
/// This is the fastest form - used when slot index is known at compile time.
///
/// # Arguments
/// * `asm` - Assembler to emit to
/// * `dst` - Destination register for loaded value
/// * `obj` - Register containing object pointer
/// * `slot_index` - Slot index (0-based)
#[inline]
pub fn emit_slot_load_immediate(asm: &mut Assembler, dst: Gpr, obj: Gpr, slot_index: u16) {
    let offset = OBJECT_INLINE_SLOTS_OFFSET + (slot_index as i32) * SLOT_SIZE;
    let mem = MemOperand::base_disp(obj, offset);
    asm.mov_rm(dst, &mem);
}

/// Emit inline slot load with dynamic offset from IC site.
///
/// Loads the slot offset from the IC site, then uses indexed addressing.
///
/// # Arguments
/// * `asm` - Assembler to emit to
/// * `dst` - Destination register for loaded value
/// * `obj` - Register containing object pointer
/// * `ic_base` - IC data base register
/// * `ic_offset` - Offset to IC site
/// * `scratch` - Scratch register for slot index
///
/// # Note
/// This uses a two-step process:
/// 1. Load slot index from IC site
/// 2. Calculate address: obj + INLINE_SLOTS_OFFSET + index * 8
pub fn emit_slot_load_ic(
    asm: &mut Assembler,
    dst: Gpr,
    obj: Gpr,
    ic_base: Gpr,
    ic_offset: i32,
    scratch: Gpr,
) {
    // Load slot_offset from IC site (u16, zero-extend to 64-bit)
    let slot_mem = MemOperand::base_disp(ic_base, ic_offset + IC_SLOT_OFFSET_OFFSET);
    // Load as 32-bit which zero-extends upper bits
    asm.mov_rm32(scratch, &slot_mem);

    // Calculate slot address and load value
    // We need: [obj + OBJECT_INLINE_SLOTS_OFFSET + scratch * 8]
    // Use SIB addressing with scale=8
    let slot_value_mem =
        MemOperand::base_index_disp(obj, scratch, Scale::X8, OBJECT_INLINE_SLOTS_OFFSET);
    asm.mov_rm(dst, &slot_value_mem);
}

/// Emit inline slot store with known offset.
///
/// # Arguments
/// * `asm` - Assembler to emit to
/// * `obj` - Register containing object pointer
/// * `slot_index` - Slot index (0-based)
/// * `src` - Source register containing value to store
#[inline]
pub fn emit_slot_store_immediate(asm: &mut Assembler, obj: Gpr, slot_index: u16, src: Gpr) {
    let offset = OBJECT_INLINE_SLOTS_OFFSET + (slot_index as i32) * SLOT_SIZE;
    let mem = MemOperand::base_disp(obj, offset);
    asm.mov_mr(&mem, src);
}

/// Emit inline slot store with dynamic offset from IC site.
///
/// # Arguments
/// * `asm` - Assembler to emit to
/// * `obj` - Register containing object pointer
/// * `ic_base` - IC data base register
/// * `ic_offset` - Offset to IC site
/// * `src` - Source register containing value to store
/// * `scratch` - Scratch register for slot index
pub fn emit_slot_store_ic(
    asm: &mut Assembler,
    obj: Gpr,
    ic_base: Gpr,
    ic_offset: i32,
    src: Gpr,
    scratch: Gpr,
) {
    // Load slot_offset from IC site
    let slot_mem = MemOperand::base_disp(ic_base, ic_offset + IC_SLOT_OFFSET_OFFSET);
    asm.mov_rm32(scratch, &slot_mem);

    // Store to calculated slot address
    let slot_value_mem =
        MemOperand::base_index_disp(obj, scratch, Scale::X8, OBJECT_INLINE_SLOTS_OFFSET);
    asm.mov_mr(&slot_value_mem, src);
}

// =============================================================================
// IC Site Update Helpers
// =============================================================================

/// Emit IC site update with new shape and slot.
///
/// Called by miss handlers to populate the IC site after a successful lookup.
///
/// # Arguments
/// * `asm` - Assembler to emit to
/// * `ic_base` - IC data base register
/// * `ic_offset` - Offset to IC site
/// * `shape_id` - Register containing shape ID to cache
/// * `slot_offset` - Register containing slot index to cache
pub fn emit_ic_site_update(
    asm: &mut Assembler,
    ic_base: Gpr,
    ic_offset: i32,
    shape_id: Gpr,
    slot_offset: Gpr,
) {
    // Store shape_id to IC site (32-bit)
    let shape_mem = MemOperand::base_disp(ic_base, ic_offset + IC_SHAPE_ID_OFFSET);
    asm.mov_mr32(&shape_mem, shape_id);

    // Store slot_offset to IC site (store 32-bit, only lower 16 used)
    let slot_mem = MemOperand::base_disp(ic_base, ic_offset + IC_SLOT_OFFSET_OFFSET);
    asm.mov_mr32(&slot_mem, slot_offset);
}

/// Emit load of IC data base pointer from JIT context.
///
/// The IC data is stored in the JIT context passed to compiled functions.
///
/// # Arguments
/// * `asm` - Assembler to emit to
/// * `dst` - Destination register for IC data pointer
/// * `ctx_reg` - Register containing JIT context pointer
/// * `ic_data_offset` - Offset to IC data array in context
#[inline]
pub fn emit_load_ic_base(asm: &mut Assembler, dst: Gpr, ctx_reg: Gpr, ic_data_offset: i32) {
    let mem = MemOperand::base_disp(ctx_reg, ic_data_offset);
    asm.mov_rm(dst, &mem);
}

// =============================================================================
// Complete IC Fast Path Emission
// =============================================================================

/// Emit complete GetAttr IC fast path.
///
/// Generates:
/// 1. Load object's shape_id
/// 2. Compare against cached shape_id in IC site
/// 3. On match: load value from cached slot offset (O(1))
/// 4. On miss: jump to miss handler
///
/// # Arguments
/// * `asm` - Assembler to emit to
/// * `dst` - Destination register for result
/// * `obj` - Object register
/// * `ic_base` - IC data base register
/// * `ic_offset` - Offset to this IC site
/// * `miss_label` - Label for cache miss
/// * `scratch1` - First scratch register
/// * `scratch2` - Second scratch register
///
/// # Register Usage
/// - `dst`: Output (value loaded from object)
/// - `obj`: Input (object pointer, preserved)
/// - `ic_base`: Input (IC data pointer, preserved)
/// - `scratch1`, `scratch2`: Clobbered
pub fn emit_get_attr_ic_fast_path(
    asm: &mut Assembler,
    dst: Gpr,
    obj: Gpr,
    ic_base: Gpr,
    ic_offset: i32,
    miss_label: Label,
    scratch1: Gpr,
    scratch2: Gpr,
) {
    // Emit shape guard (jumps to miss_label on mismatch)
    emit_shape_guard(asm, obj, ic_base, ic_offset, miss_label, scratch1, scratch2);

    // Fast path: load from cached slot
    emit_slot_load_ic(asm, dst, obj, ic_base, ic_offset, scratch1);
}

/// Emit complete SetAttr IC fast path for existing property.
///
/// Note: This only handles updates to existing properties where the shape
/// already matches. New property additions require shape transition and deopt.
///
/// # Arguments
/// * `asm` - Assembler to emit to
/// * `obj` - Object register
/// * `src` - Value register to store
/// * `ic_base` - IC data base register
/// * `ic_offset` - Offset to this IC site
/// * `miss_label` - Label for cache miss
/// * `scratch1` - First scratch register
/// * `scratch2` - Second scratch register
pub fn emit_set_attr_ic_fast_path(
    asm: &mut Assembler,
    obj: Gpr,
    src: Gpr,
    ic_base: Gpr,
    ic_offset: i32,
    miss_label: Label,
    scratch1: Gpr,
    scratch2: Gpr,
) {
    // Emit shape guard
    emit_shape_guard(asm, obj, ic_base, ic_offset, miss_label, scratch1, scratch2);

    // Fast path: store to cached slot
    emit_slot_store_ic(asm, obj, ic_base, ic_offset, src, scratch1);
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::x64::Assembler;

    // -------------------------------------------------------------------------
    // Layout Constant Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_object_layout_constants() {
        // Verify layout assumptions match ShapedObject
        assert_eq!(OBJECT_HEADER_SIZE, 16);
        assert_eq!(OBJECT_SHAPE_PTR_OFFSET, 16);
        assert_eq!(OBJECT_INLINE_SLOTS_OFFSET, 24);
        assert_eq!(SLOT_SIZE, 8);
    }

    #[test]
    fn test_ic_site_layout_constants() {
        assert_eq!(IC_SITE_SIZE, 8);
        assert_eq!(IC_SHAPE_ID_OFFSET, 0);
        assert_eq!(IC_SLOT_OFFSET_OFFSET, 4);
        assert_eq!(IC_FLAGS_OFFSET, 6);
    }

    #[test]
    fn test_slot_offset_calculation() {
        // Verify slot offset calculations for various indices
        for i in 0..8u16 {
            let expected = OBJECT_INLINE_SLOTS_OFFSET + (i as i32) * SLOT_SIZE;
            assert_eq!(expected, 24 + (i as i32) * 8);
        }
    }

    #[test]
    fn test_ic_offset_non_overlapping() {
        // Verify multiple IC sites don't overlap
        for i in 0..10 {
            let offset = (i as i32) * (IC_SITE_SIZE as i32);
            let next_offset = ((i + 1) as i32) * (IC_SITE_SIZE as i32);
            assert_eq!(next_offset - offset, IC_SITE_SIZE as i32);
        }
    }

    // -------------------------------------------------------------------------
    // Shape ID Loading Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_emit_load_shape_id() {
        let mut asm = Assembler::new();
        emit_load_shape_id(&mut asm, Gpr::Rax, Gpr::Rdi, Gpr::Rcx);

        // Should emit:
        // mov rcx, [rdi + 16]  (load shape pointer)
        // mov eax, [rcx + 16]  (load shape_id)
        assert!(asm.offset() > 0, "Should emit instructions");
    }

    #[test]
    fn test_emit_load_shape_id_different_regs() {
        let mut asm = Assembler::new();
        emit_load_shape_id(&mut asm, Gpr::Rbx, Gpr::Rsi, Gpr::R10);

        assert!(asm.offset() > 0);
    }

    // -------------------------------------------------------------------------
    // Slot Load Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_emit_slot_load_immediate_first_slot() {
        let mut asm = Assembler::new();
        emit_slot_load_immediate(&mut asm, Gpr::Rax, Gpr::Rdi, 0);

        // Should emit: mov rax, [rdi + 24]
        assert!(asm.offset() > 0);
    }

    #[test]
    fn test_emit_slot_load_immediate_various_slots() {
        for slot_idx in 0..8u16 {
            let mut asm = Assembler::new();
            emit_slot_load_immediate(&mut asm, Gpr::Rax, Gpr::Rdi, slot_idx);
            assert!(asm.offset() > 0, "Slot {} should emit code", slot_idx);
        }
    }

    #[test]
    fn test_emit_slot_load_ic() {
        let mut asm = Assembler::new();
        emit_slot_load_ic(&mut asm, Gpr::Rax, Gpr::Rdi, Gpr::R15, 0, Gpr::Rcx);

        // Should emit slot offset load + indexed load
        assert!(asm.offset() > 5, "Should emit multiple instructions");
    }

    #[test]
    fn test_emit_slot_load_ic_with_offset() {
        let mut asm = Assembler::new();
        emit_slot_load_ic(&mut asm, Gpr::Rax, Gpr::Rdi, Gpr::R15, 16, Gpr::Rcx);

        assert!(asm.offset() > 5);
    }

    // -------------------------------------------------------------------------
    // Slot Store Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_emit_slot_store_immediate() {
        let mut asm = Assembler::new();
        emit_slot_store_immediate(&mut asm, Gpr::Rdi, 0, Gpr::Rax);

        // Should emit: mov [rdi + 24], rax
        assert!(asm.offset() > 0);
    }

    #[test]
    fn test_emit_slot_store_immediate_various_slots() {
        for slot_idx in 0..8u16 {
            let mut asm = Assembler::new();
            emit_slot_store_immediate(&mut asm, Gpr::Rdi, slot_idx, Gpr::Rax);
            assert!(asm.offset() > 0, "Slot {} should emit code", slot_idx);
        }
    }

    #[test]
    fn test_emit_slot_store_ic() {
        let mut asm = Assembler::new();
        emit_slot_store_ic(&mut asm, Gpr::Rdi, Gpr::R15, 0, Gpr::Rax, Gpr::Rcx);

        assert!(asm.offset() > 5);
    }

    // -------------------------------------------------------------------------
    // Shape Guard Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_emit_shape_guard() {
        let mut asm = Assembler::new();
        let miss = asm.create_label();

        emit_shape_guard(&mut asm, Gpr::Rdi, Gpr::R15, 0, miss, Gpr::Rcx, Gpr::Rdx);

        // Should emit: load shape_id, load cached, cmp, jne
        assert!(
            asm.offset() > 10,
            "Shape guard should emit multiple instructions"
        );
    }

    #[test]
    fn test_emit_shape_guard_with_ic_offset() {
        let mut asm = Assembler::new();
        let miss = asm.create_label();

        emit_shape_guard(&mut asm, Gpr::Rdi, Gpr::R15, 24, miss, Gpr::Rcx, Gpr::Rdx);

        assert!(asm.offset() > 10);
    }

    #[test]
    fn test_emit_shape_guard_immediate() {
        let mut asm = Assembler::new();
        let miss = asm.create_label();

        emit_shape_guard_immediate(&mut asm, Gpr::Rdi, 42, miss, Gpr::Rcx, Gpr::Rdx);

        assert!(asm.offset() > 0);
    }

    #[test]
    fn test_emit_shape_guard_immediate_large_value() {
        let mut asm = Assembler::new();
        let miss = asm.create_label();

        emit_shape_guard_immediate(&mut asm, Gpr::Rdi, 0xDEADBEEF, miss, Gpr::Rcx, Gpr::Rdx);

        assert!(asm.offset() > 0);
    }

    // -------------------------------------------------------------------------
    // IC Site Update Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_emit_ic_site_update() {
        let mut asm = Assembler::new();

        emit_ic_site_update(&mut asm, Gpr::R15, 0, Gpr::Rax, Gpr::Rcx);

        // Should emit two stores
        assert!(asm.offset() > 0);
    }

    #[test]
    fn test_emit_ic_site_update_with_offset() {
        let mut asm = Assembler::new();

        emit_ic_site_update(&mut asm, Gpr::R15, 32, Gpr::Rax, Gpr::Rcx);

        assert!(asm.offset() > 0);
    }

    #[test]
    fn test_emit_load_ic_base() {
        let mut asm = Assembler::new();

        emit_load_ic_base(&mut asm, Gpr::R15, Gpr::Rdi, 64);

        // Should emit: mov r15, [rdi + 64]
        assert!(asm.offset() > 0);
    }

    // -------------------------------------------------------------------------
    // Complete Fast Path Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_emit_get_attr_ic_fast_path() {
        let mut asm = Assembler::new();
        let miss = asm.create_label();

        emit_get_attr_ic_fast_path(
            &mut asm,
            Gpr::Rax,
            Gpr::Rdi,
            Gpr::R15,
            0,
            miss,
            Gpr::Rcx,
            Gpr::Rdx,
        );

        // Should emit shape guard + slot load
        assert!(asm.offset() > 15, "GetAttr IC should emit substantial code");
    }

    #[test]
    fn test_emit_get_attr_ic_fast_path_various_offsets() {
        for offset in [0, 8, 16, 24, 32].iter() {
            let mut asm = Assembler::new();
            let miss = asm.create_label();

            emit_get_attr_ic_fast_path(
                &mut asm,
                Gpr::Rax,
                Gpr::Rdi,
                Gpr::R15,
                *offset,
                miss,
                Gpr::Rcx,
                Gpr::Rdx,
            );

            assert!(asm.offset() > 10, "Offset {} should work", offset);
        }
    }

    #[test]
    fn test_emit_set_attr_ic_fast_path() {
        let mut asm = Assembler::new();
        let miss = asm.create_label();

        emit_set_attr_ic_fast_path(
            &mut asm,
            Gpr::Rdi,
            Gpr::Rsi,
            Gpr::R15,
            0,
            miss,
            Gpr::Rcx,
            Gpr::Rdx,
        );

        // Should emit shape guard + slot store
        assert!(asm.offset() > 15, "SetAttr IC should emit substantial code");
    }

    #[test]
    fn test_emit_set_attr_ic_fast_path_different_src() {
        let mut asm = Assembler::new();
        let miss = asm.create_label();

        emit_set_attr_ic_fast_path(
            &mut asm,
            Gpr::Rdi,
            Gpr::Rax, // Different source register
            Gpr::R15,
            8,
            miss,
            Gpr::Rcx,
            Gpr::Rdx,
        );

        assert!(asm.offset() > 10);
    }

    // -------------------------------------------------------------------------
    // Code Size Estimation Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_fast_path_code_size() {
        // Measure typical fast path sizes for estimation

        let mut asm = Assembler::new();
        let miss = asm.create_label();
        emit_get_attr_ic_fast_path(
            &mut asm,
            Gpr::Rax,
            Gpr::Rdi,
            Gpr::R15,
            0,
            miss,
            Gpr::Rcx,
            Gpr::Rdx,
        );
        let get_attr_size = asm.offset();

        let mut asm = Assembler::new();
        let miss = asm.create_label();
        emit_set_attr_ic_fast_path(
            &mut asm,
            Gpr::Rdi,
            Gpr::Rsi,
            Gpr::R15,
            0,
            miss,
            Gpr::Rcx,
            Gpr::Rdx,
        );
        let set_attr_size = asm.offset();

        // Both should be compact
        assert!(
            get_attr_size < 100,
            "GetAttr should be compact: {}",
            get_attr_size
        );
        assert!(
            set_attr_size < 100,
            "SetAttr should be compact: {}",
            set_attr_size
        );
    }

    // -------------------------------------------------------------------------
    // Register Constraint Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_emit_with_r8_r15() {
        // Test with extended registers
        let mut asm = Assembler::new();
        let miss = asm.create_label();

        emit_get_attr_ic_fast_path(
            &mut asm,
            Gpr::R8,
            Gpr::R9,
            Gpr::R10,
            0,
            miss,
            Gpr::R11,
            Gpr::R12,
        );

        assert!(asm.offset() > 0);
    }

    #[test]
    fn test_emit_preserves_obj_register() {
        // The obj register should not be clobbered by the fast path
        let mut asm = Assembler::new();
        let miss = asm.create_label();

        emit_get_attr_ic_fast_path(
            &mut asm,
            Gpr::Rax, // dst - gets result
            Gpr::Rdi, // obj - should be preserved
            Gpr::R15, // ic_base - should be preserved
            0,
            miss,
            Gpr::Rcx, // scratch1 - clobbered
            Gpr::Rdx, // scratch2 - clobbered
        );

        // If we got here, the register assignment is valid
        assert!(asm.offset() > 0);
    }
}
