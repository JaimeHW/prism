//! String Operations Performance Benchmarks
//!
//! Measures performance of Python string operations to validate SSO (Small String
//! Optimization) and identity-based fast paths.
//!
//! # Benchmark Categories
//!
//! 1. **Concatenation**: String + String with varying sizes
//! 2. **Repetition**: String * N for building repeated patterns
//! 3. **Comparison**: Identity-based vs content-based equality
//! 4. **Interning**: Interned string lookup performance
//!
//! # Key Performance Targets
//!
//! - SSO strings (<= 23 bytes): No heap allocation
//! - Identity comparison: O(1) pointer equality
//! - Interned strings: Hash-consing for deduplication

use criterion::{BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main};
use prism_compiler::bytecode::{CodeFlags, CodeObject, Instruction, Opcode, Register};
use prism_core::Value;
use prism_core::intern::intern;
use prism_vm::{JitConfig, VirtualMachine};
use std::sync::Arc;

// =============================================================================
// String Benchmark Helpers
// =============================================================================

/// Create code that concatenates two string constants.
fn create_string_concat_code(a: &str, b: &str) -> Arc<CodeObject> {
    let instructions = vec![
        Instruction::op_di(Opcode::LoadConst, Register(0), 0),
        Instruction::op_di(Opcode::LoadConst, Register(1), 1),
        Instruction::op_dss(Opcode::Add, Register(2), Register(0), Register(1)),
        Instruction::op_d(Opcode::Return, Register(2)),
    ];

    let constants = vec![Value::string(intern(a)), Value::string(intern(b))];

    Arc::new(CodeObject {
        name: "bench_string_concat".into(),
        register_count: 3,
        arg_count: 0,
        posonlyarg_count: 0,
        kwonlyarg_count: 0,
        instructions: instructions.into_boxed_slice(),
        constants: constants.into_boxed_slice(),
        names: Box::new([]),
        locals: Box::new([]),
        freevars: Box::new([]),
        cellvars: Box::new([]),
        line_table: Box::new([]),
        filename: "bench.py".into(),
        qualname: "bench_string_concat".into(),
        flags: CodeFlags::NONE,
        first_lineno: 1,
    })
}

/// Create code that repeats a string N times (str * n).
fn create_string_repeat_code(s: &str, n: i64) -> Arc<CodeObject> {
    let instructions = vec![
        Instruction::op_di(Opcode::LoadConst, Register(0), 0),
        Instruction::op_di(Opcode::LoadConst, Register(1), 1),
        Instruction::op_dss(Opcode::Mul, Register(2), Register(0), Register(1)),
        Instruction::op_d(Opcode::Return, Register(2)),
    ];

    let constants = vec![Value::string(intern(s)), Value::int(n).unwrap()];

    Arc::new(CodeObject {
        name: "bench_string_repeat".into(),
        register_count: 3,
        arg_count: 0,
        posonlyarg_count: 0,
        kwonlyarg_count: 0,
        instructions: instructions.into_boxed_slice(),
        constants: constants.into_boxed_slice(),
        names: Box::new([]),
        locals: Box::new([]),
        freevars: Box::new([]),
        cellvars: Box::new([]),
        line_table: Box::new([]),
        filename: "bench.py".into(),
        qualname: "bench_string_repeat".into(),
        flags: CodeFlags::NONE,
        first_lineno: 1,
    })
}

/// Create code that compares two strings for equality.
fn create_string_compare_code(a: &str, b: &str) -> Arc<CodeObject> {
    let instructions = vec![
        Instruction::op_di(Opcode::LoadConst, Register(0), 0),
        Instruction::op_di(Opcode::LoadConst, Register(1), 1),
        Instruction::op_dss(Opcode::Eq, Register(2), Register(0), Register(1)),
        Instruction::op_d(Opcode::Return, Register(2)),
    ];

    let constants = vec![Value::string(intern(a)), Value::string(intern(b))];

    Arc::new(CodeObject {
        name: "bench_string_compare".into(),
        register_count: 3,
        arg_count: 0,
        posonlyarg_count: 0,
        kwonlyarg_count: 0,
        instructions: instructions.into_boxed_slice(),
        constants: constants.into_boxed_slice(),
        names: Box::new([]),
        locals: Box::new([]),
        freevars: Box::new([]),
        cellvars: Box::new([]),
        line_table: Box::new([]),
        filename: "bench.py".into(),
        qualname: "bench_string_compare".into(),
        flags: CodeFlags::NONE,
        first_lineno: 1,
    })
}

/// Create code that builds a large string through repeated concatenation.
fn create_string_build_loop(iterations: i64) -> Arc<CodeObject> {
    // Build: result = ""; for i in range(n): result = result + "x"
    let mut instructions = Vec::new();
    let mut constants = Vec::new();

    // r0 = "" (empty string accumulator)
    constants.push(Value::string(intern("")));
    instructions.push(Instruction::op_di(Opcode::LoadConst, Register(0), 0));

    // r1 = n (iterations)
    constants.push(Value::int(iterations).unwrap());
    instructions.push(Instruction::op_di(Opcode::LoadConst, Register(1), 1));

    // r2 = 0 (counter)
    constants.push(Value::int(0).unwrap());
    instructions.push(Instruction::op_di(Opcode::LoadConst, Register(2), 2));

    // r3 = 1 (increment)
    constants.push(Value::int(1).unwrap());
    instructions.push(Instruction::op_di(Opcode::LoadConst, Register(3), 3));

    // r4 = "x" (character to append)
    constants.push(Value::string(intern("x")));
    instructions.push(Instruction::op_di(Opcode::LoadConst, Register(4), 4));

    // loop_header:
    let loop_header = instructions.len();

    // r5 = r2 < r1
    instructions.push(Instruction::op_dss(
        Opcode::Lt,
        Register(5),
        Register(2),
        Register(1),
    ));

    // JumpIfFalse r5 -> exit
    let jump_if_false_idx = instructions.len();
    instructions.push(Instruction::op_di(Opcode::JumpIfFalse, Register(5), 0));

    // r0 = r0 + r4 (concatenate)
    instructions.push(Instruction::op_dss(
        Opcode::Add,
        Register(0),
        Register(0),
        Register(4),
    ));

    // r2 = r2 + r3 (increment counter)
    instructions.push(Instruction::op_dss(
        Opcode::Add,
        Register(2),
        Register(2),
        Register(3),
    ));

    // Jump -> loop_header
    let jump_back_idx = instructions.len();
    instructions.push(Instruction::op_di(Opcode::Jump, Register(0), 0));

    // Patch jump back
    let next_ip = jump_back_idx + 1;
    let offset = (loop_header as i32) - (next_ip as i32);
    instructions[jump_back_idx] =
        Instruction::op_di(Opcode::Jump, Register(0), offset as i16 as u16);

    // exit:
    let exit_idx = instructions.len();

    // Patch forward jump
    let next_ip_fwd = jump_if_false_idx + 1;
    let fwd_offset = (exit_idx as i32) - (next_ip_fwd as i32);
    instructions[jump_if_false_idx] =
        Instruction::op_di(Opcode::JumpIfFalse, Register(5), fwd_offset as i16 as u16);

    // Return r0
    instructions.push(Instruction::op_d(Opcode::Return, Register(0)));

    Arc::new(CodeObject {
        name: "bench_string_build_loop".into(),
        register_count: 6,
        arg_count: 0,
        posonlyarg_count: 0,
        kwonlyarg_count: 0,
        instructions: instructions.into_boxed_slice(),
        constants: constants.into_boxed_slice(),
        names: Box::new([]),
        locals: Box::new([]),
        freevars: Box::new([]),
        cellvars: Box::new([]),
        line_table: Box::new([]),
        filename: "bench.py".into(),
        qualname: "bench_string_build_loop".into(),
        flags: CodeFlags::NONE,
        first_lineno: 1,
    })
}

// =============================================================================
// Benchmarks
// =============================================================================

fn bench_string_concat(c: &mut Criterion) {
    let mut group = c.benchmark_group("string_concat");

    // SSO strings (short, fits in inline storage)
    group.bench_function("sso_short", |b| {
        let config = JitConfig::for_testing();
        let mut vm = VirtualMachine::with_jit_config(config);
        let code = create_string_concat_code("hello", "world");

        b.iter(|| black_box(vm.execute(Arc::clone(&code)).unwrap()));
    });

    // SSO boundary (near 23 bytes)
    group.bench_function("sso_boundary", |b| {
        let config = JitConfig::for_testing();
        let mut vm = VirtualMachine::with_jit_config(config);
        // 12 + 12 = 24 bytes (just over SSO threshold)
        let code = create_string_concat_code("0123456789ab", "0123456789ab");

        b.iter(|| black_box(vm.execute(Arc::clone(&code)).unwrap()));
    });

    // Heap strings (longer than SSO)
    group.bench_function("heap_medium", |b| {
        let config = JitConfig::for_testing();
        let mut vm = VirtualMachine::with_jit_config(config);
        let s = "x".repeat(50);
        let code = create_string_concat_code(&s, &s);

        b.iter(|| black_box(vm.execute(Arc::clone(&code)).unwrap()));
    });

    // Large heap strings
    group.bench_function("heap_large", |b| {
        let config = JitConfig::for_testing();
        let mut vm = VirtualMachine::with_jit_config(config);
        let s = "x".repeat(1000);
        let code = create_string_concat_code(&s, &s);

        b.iter(|| black_box(vm.execute(Arc::clone(&code)).unwrap()));
    });

    group.finish();
}

fn bench_string_repeat(c: &mut Criterion) {
    let mut group = c.benchmark_group("string_repeat");

    // Small repetitions (stays in SSO)
    group.bench_function("sso_small", |b| {
        let config = JitConfig::for_testing();
        let mut vm = VirtualMachine::with_jit_config(config);
        let code = create_string_repeat_code("abc", 5); // 15 bytes

        b.iter(|| black_box(vm.execute(Arc::clone(&code)).unwrap()));
    });

    // Medium repetitions
    for count in [10, 100, 1000].iter() {
        group.throughput(Throughput::Elements(*count as u64));
        group.bench_with_input(BenchmarkId::new("repeat_x", count), count, |b, &n| {
            let config = JitConfig::for_testing();
            let mut vm = VirtualMachine::with_jit_config(config);
            let code = create_string_repeat_code("x", n as i64);

            b.iter(|| black_box(vm.execute(Arc::clone(&code)).unwrap()));
        });
    }

    group.finish();
}

fn bench_string_compare(c: &mut Criterion) {
    let mut group = c.benchmark_group("string_compare");

    // Identity comparison (same interned string)
    group.bench_function("identity_match", |b| {
        let config = JitConfig::for_testing();
        let mut vm = VirtualMachine::with_jit_config(config);
        // Same string literal → should be identity-comparable
        let code = create_string_compare_code("hello", "hello");

        b.iter(|| black_box(vm.execute(Arc::clone(&code)).unwrap()));
    });

    // Content comparison (different strings, same content)
    group.bench_function("content_match_short", |b| {
        let config = JitConfig::for_testing();
        let mut vm = VirtualMachine::with_jit_config(config);
        let code = create_string_compare_code("hello", "hello");

        // Warm up to ensure speculation
        for _ in 0..20 {
            let _ = vm.execute(Arc::clone(&code));
        }

        b.iter(|| black_box(vm.execute(Arc::clone(&code)).unwrap()));
    });

    // Mismatch (early exit)
    group.bench_function("mismatch_early", |b| {
        let config = JitConfig::for_testing();
        let mut vm = VirtualMachine::with_jit_config(config);
        let code = create_string_compare_code("hello", "world");

        b.iter(|| black_box(vm.execute(Arc::clone(&code)).unwrap()));
    });

    // Long string comparison
    group.bench_function("content_match_long", |b| {
        let config = JitConfig::for_testing();
        let mut vm = VirtualMachine::with_jit_config(config);
        let s = "x".repeat(1000);
        let code = create_string_compare_code(&s, &s);

        b.iter(|| black_box(vm.execute(Arc::clone(&code)).unwrap()));
    });

    group.finish();
}

fn bench_string_build_loop(c: &mut Criterion) {
    let mut group = c.benchmark_group("string_build_loop");
    group.sample_size(50); // Reduce sample size for expensive benchmarks

    for iterations in [10, 50, 100].iter() {
        group.throughput(Throughput::Elements(*iterations as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(iterations),
            iterations,
            |b, &n| {
                let config = JitConfig::for_testing();
                let mut vm = VirtualMachine::with_jit_config(config);
                let code = create_string_build_loop(n as i64);

                b.iter(|| black_box(vm.execute(Arc::clone(&code)).unwrap()));
            },
        );
    }

    group.finish();
}

criterion_group!(
    string_benches,
    bench_string_concat,
    bench_string_repeat,
    bench_string_compare,
    bench_string_build_loop,
);

criterion_main!(string_benches);
