//! JIT Speedup Factor Benchmarks
//!
//! Measures the performance improvement of JIT-compiled code over interpreter
//! baseline for various workloads.
//!
//! # Benchmark Categories
//!
//! 1. **Fibonacci**: Recursive computation (tests call overhead, speculation)
//! 2. **Loop Sum**: Simple loop accumulator (tests loop optimization, OSR)
//! 3. **Matrix Multiply**: Nested loops with array access (tests memory patterns)
//! 4. **Guard Speculation**: Type-stable vs polymorphic code paths
//!
//! # Measurement Methodology
//!
//! Each benchmark is run in two configurations:
//! - **Interpreter**: JIT disabled, pure bytecode dispatch
//! - **JIT Enabled**: With speculation and optimization
//!
//! The speedup factor is calculated as: interpreter_time / jit_time

use criterion::{
    BenchmarkGroup, BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main,
    measurement::WallTime,
};
use prism_compiler::bytecode::{CodeFlags, CodeObject, Instruction, Opcode, Register};
use prism_core::Value;
use prism_vm::{JitConfig, VirtualMachine};
use std::sync::Arc;

// =============================================================================
// Benchmark Code Generators
// =============================================================================

/// Create iterative fibonacci code: fib(n)
///
/// Python equivalent:
/// ```python
/// def fib(n):
///     a, b = 0, 1
///     for _ in range(n):
///         a, b = b, a + b
///     return a
/// ```
fn create_fib_iterative(n: i64) -> Arc<CodeObject> {
    let mut instructions = Vec::new();
    let mut constants = Vec::new();

    // r0 = 0 (a)
    constants.push(Value::int(0).unwrap());
    instructions.push(Instruction::op_di(Opcode::LoadConst, Register(0), 0));

    // r1 = 1 (b)
    constants.push(Value::int(1).unwrap());
    instructions.push(Instruction::op_di(Opcode::LoadConst, Register(1), 1));

    // r2 = n (counter target)
    constants.push(Value::int(n).unwrap());
    instructions.push(Instruction::op_di(Opcode::LoadConst, Register(2), 2));

    // r3 = 0 (counter)
    instructions.push(Instruction::op_di(Opcode::LoadConst, Register(3), 0));

    // r4 = 1 (increment)
    instructions.push(Instruction::op_di(Opcode::LoadConst, Register(4), 1));

    // loop_header:
    let loop_header = instructions.len();

    // r5 = r3 < r2
    instructions.push(Instruction::op_dss(
        Opcode::Lt,
        Register(5),
        Register(3),
        Register(2),
    ));

    // JumpIfFalse r5 -> exit
    let jump_if_false_idx = instructions.len();
    instructions.push(Instruction::op_di(Opcode::JumpIfFalse, Register(5), 0));

    // r6 = r0 + r1 (temp = a + b)
    instructions.push(Instruction::op_dss(
        Opcode::Add,
        Register(6),
        Register(0),
        Register(1),
    ));

    // r0 = r1 (a = b)
    instructions.push(Instruction::op_ds(Opcode::Move, Register(0), Register(1)));

    // r1 = r6 (b = temp)
    instructions.push(Instruction::op_ds(Opcode::Move, Register(1), Register(6)));

    // r3 = r3 + r4 (counter++)
    instructions.push(Instruction::op_dss(
        Opcode::Add,
        Register(3),
        Register(3),
        Register(4),
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
        name: "fib_iterative".into(),
        register_count: 7,
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
        qualname: "fib_iterative".into(),
        flags: CodeFlags::NONE,
        first_lineno: 1,
    })
}

/// Create simple loop sum: sum(range(n))
fn create_loop_sum(n: i64) -> Arc<CodeObject> {
    let mut instructions = Vec::new();
    let mut constants = Vec::new();

    // r0 = 0 (sum)
    constants.push(Value::int(0).unwrap());
    instructions.push(Instruction::op_di(Opcode::LoadConst, Register(0), 0));

    // r1 = n (limit)
    constants.push(Value::int(n).unwrap());
    instructions.push(Instruction::op_di(Opcode::LoadConst, Register(1), 1));

    // r2 = 0 (i)
    instructions.push(Instruction::op_di(Opcode::LoadConst, Register(2), 0));

    // r3 = 1 (increment)
    constants.push(Value::int(1).unwrap());
    instructions.push(Instruction::op_di(Opcode::LoadConst, Register(3), 2));

    // loop:
    let loop_header = instructions.len();

    // r4 = r2 < r1
    instructions.push(Instruction::op_dss(
        Opcode::Lt,
        Register(4),
        Register(2),
        Register(1),
    ));

    // JumpIfFalse r4 -> exit
    let jump_if_false_idx = instructions.len();
    instructions.push(Instruction::op_di(Opcode::JumpIfFalse, Register(4), 0));

    // r0 = r0 + r2 (sum += i)
    instructions.push(Instruction::op_dss(
        Opcode::Add,
        Register(0),
        Register(0),
        Register(2),
    ));

    // r2 = r2 + r3 (i++)
    instructions.push(Instruction::op_dss(
        Opcode::Add,
        Register(2),
        Register(2),
        Register(3),
    ));

    // Jump -> loop
    let jump_back_idx = instructions.len();
    instructions.push(Instruction::op_di(Opcode::Jump, Register(0), 0));

    // Patch jumps
    let next_ip = jump_back_idx + 1;
    let offset = (loop_header as i32) - (next_ip as i32);
    instructions[jump_back_idx] =
        Instruction::op_di(Opcode::Jump, Register(0), offset as i16 as u16);

    let exit_idx = instructions.len();
    let next_ip_fwd = jump_if_false_idx + 1;
    let fwd_offset = (exit_idx as i32) - (next_ip_fwd as i32);
    instructions[jump_if_false_idx] =
        Instruction::op_di(Opcode::JumpIfFalse, Register(4), fwd_offset as i16 as u16);

    // Return r0
    instructions.push(Instruction::op_d(Opcode::Return, Register(0)));

    Arc::new(CodeObject {
        name: "loop_sum".into(),
        register_count: 5,
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
        qualname: "loop_sum".into(),
        flags: CodeFlags::NONE,
        first_lineno: 1,
    })
}

/// Create a nested loop workload (simulates simple matrix operations).
/// Computes: sum of i*j for i in range(n) for j in range(n)
fn create_nested_loop(n: i64) -> Arc<CodeObject> {
    let mut instructions = Vec::new();
    let mut constants = Vec::new();

    // r0 = 0 (total sum)
    constants.push(Value::int(0).unwrap());
    instructions.push(Instruction::op_di(Opcode::LoadConst, Register(0), 0));

    // r1 = n (outer limit)
    constants.push(Value::int(n).unwrap());
    instructions.push(Instruction::op_di(Opcode::LoadConst, Register(1), 1));

    // r2 = 0 (outer i)
    instructions.push(Instruction::op_di(Opcode::LoadConst, Register(2), 0));

    // r3 = 1 (increment)
    constants.push(Value::int(1).unwrap());
    instructions.push(Instruction::op_di(Opcode::LoadConst, Register(3), 2));

    // OUTER LOOP
    let outer_loop = instructions.len();

    // r4 = r2 < r1
    instructions.push(Instruction::op_dss(
        Opcode::Lt,
        Register(4),
        Register(2),
        Register(1),
    ));

    // JumpIfFalse r4 -> outer_exit
    let outer_jump_idx = instructions.len();
    instructions.push(Instruction::op_di(Opcode::JumpIfFalse, Register(4), 0));

    // r5 = 0 (inner j)
    instructions.push(Instruction::op_di(Opcode::LoadConst, Register(5), 0));

    // INNER LOOP
    let inner_loop = instructions.len();

    // r6 = r5 < r1
    instructions.push(Instruction::op_dss(
        Opcode::Lt,
        Register(6),
        Register(5),
        Register(1),
    ));

    // JumpIfFalse r6 -> inner_exit
    let inner_jump_idx = instructions.len();
    instructions.push(Instruction::op_di(Opcode::JumpIfFalse, Register(6), 0));

    // r7 = r2 * r5 (i * j)
    instructions.push(Instruction::op_dss(
        Opcode::Mul,
        Register(7),
        Register(2),
        Register(5),
    ));

    // r0 = r0 + r7 (sum += i*j)
    instructions.push(Instruction::op_dss(
        Opcode::Add,
        Register(0),
        Register(0),
        Register(7),
    ));

    // r5 = r5 + r3 (j++)
    instructions.push(Instruction::op_dss(
        Opcode::Add,
        Register(5),
        Register(5),
        Register(3),
    ));

    // Jump -> inner_loop
    let inner_back_idx = instructions.len();
    instructions.push(Instruction::op_di(Opcode::Jump, Register(0), 0));

    // Patch inner loop back
    let next_ip = inner_back_idx + 1;
    let offset = (inner_loop as i32) - (next_ip as i32);
    instructions[inner_back_idx] =
        Instruction::op_di(Opcode::Jump, Register(0), offset as i16 as u16);

    // inner_exit:
    let inner_exit = instructions.len();
    let inner_fwd = (inner_exit as i32) - ((inner_jump_idx + 1) as i32);
    instructions[inner_jump_idx] =
        Instruction::op_di(Opcode::JumpIfFalse, Register(6), inner_fwd as i16 as u16);

    // r2 = r2 + r3 (i++)
    instructions.push(Instruction::op_dss(
        Opcode::Add,
        Register(2),
        Register(2),
        Register(3),
    ));

    // Jump -> outer_loop
    let outer_back_idx = instructions.len();
    instructions.push(Instruction::op_di(Opcode::Jump, Register(0), 0));

    // Patch outer loop back
    let next_ip = outer_back_idx + 1;
    let offset = (outer_loop as i32) - (next_ip as i32);
    instructions[outer_back_idx] =
        Instruction::op_di(Opcode::Jump, Register(0), offset as i16 as u16);

    // outer_exit:
    let outer_exit = instructions.len();
    let outer_fwd = (outer_exit as i32) - ((outer_jump_idx + 1) as i32);
    instructions[outer_jump_idx] =
        Instruction::op_di(Opcode::JumpIfFalse, Register(4), outer_fwd as i16 as u16);

    // Return r0
    instructions.push(Instruction::op_d(Opcode::Return, Register(0)));

    Arc::new(CodeObject {
        name: "nested_loop".into(),
        register_count: 8,
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
        qualname: "nested_loop".into(),
        flags: CodeFlags::NONE,
        first_lineno: 1,
    })
}

// =============================================================================
// Benchmark Runner Helpers
// =============================================================================

fn run_with_jit(group: &mut BenchmarkGroup<'_, WallTime>, name: &str, code: Arc<CodeObject>) {
    group.bench_function(BenchmarkId::new("jit", name), |b| {
        let config = JitConfig::for_testing();
        let mut vm = VirtualMachine::with_jit_config(config);

        // Warm up to trigger JIT compilation
        for _ in 0..100 {
            let _ = vm.execute(Arc::clone(&code));
        }

        b.iter(|| black_box(vm.execute(Arc::clone(&code)).unwrap()));
    });
}

fn run_interpreter_only(
    group: &mut BenchmarkGroup<'_, WallTime>,
    name: &str,
    code: Arc<CodeObject>,
) {
    group.bench_function(BenchmarkId::new("interpreter", name), |b| {
        let config = JitConfig::disabled();
        let mut vm = VirtualMachine::with_jit_config(config);

        b.iter(|| black_box(vm.execute(Arc::clone(&code)).unwrap()));
    });
}

// =============================================================================
// Benchmarks
// =============================================================================

fn bench_fibonacci(c: &mut Criterion) {
    let mut group = c.benchmark_group("fibonacci");

    for n in [10, 20, 30, 40].iter() {
        let code = create_fib_iterative(*n as i64);

        group.throughput(Throughput::Elements(*n as u64));
        run_with_jit(&mut group, &format!("fib_{}", n), Arc::clone(&code));
        run_interpreter_only(&mut group, &format!("fib_{}", n), Arc::clone(&code));
    }

    group.finish();
}

fn bench_loop_sum(c: &mut Criterion) {
    let mut group = c.benchmark_group("loop_sum");

    for n in [100, 1000, 10000].iter() {
        let code = create_loop_sum(*n as i64);

        group.throughput(Throughput::Elements(*n as u64));
        run_with_jit(&mut group, &format!("n_{}", n), Arc::clone(&code));
        run_interpreter_only(&mut group, &format!("n_{}", n), Arc::clone(&code));
    }

    group.finish();
}

fn bench_nested_loops(c: &mut Criterion) {
    let mut group = c.benchmark_group("nested_loops");
    group.sample_size(20); // Fewer samples for expensive benchmarks

    for n in [10, 20, 50].iter() {
        let code = create_nested_loop(*n as i64);
        let ops = (*n as u64) * (*n as u64);

        group.throughput(Throughput::Elements(ops));
        run_with_jit(&mut group, &format!("{}x{}", n, n), Arc::clone(&code));
        run_interpreter_only(&mut group, &format!("{}x{}", n, n), Arc::clone(&code));
    }

    group.finish();
}

fn bench_cold_vs_warm(c: &mut Criterion) {
    let mut group = c.benchmark_group("cold_vs_warm");

    // Cold start: new VM each time
    group.bench_function("cold_start", |b| {
        let code = create_loop_sum(1000);

        b.iter(|| {
            let config = JitConfig::for_testing();
            let mut vm = VirtualMachine::with_jit_config(config);
            black_box(vm.execute(Arc::clone(&code)).unwrap())
        });
    });

    // Warm: reuse VM with JIT-compiled code
    group.bench_function("warm_jit", |b| {
        let config = JitConfig::for_testing();
        let mut vm = VirtualMachine::with_jit_config(config);
        let code = create_loop_sum(1000);

        // Warm up
        for _ in 0..100 {
            let _ = vm.execute(Arc::clone(&code));
        }

        b.iter(|| black_box(vm.execute(Arc::clone(&code)).unwrap()));
    });

    group.finish();
}

criterion_group!(
    jit_speedup_benches,
    bench_fibonacci,
    bench_loop_sum,
    bench_nested_loops,
    bench_cold_vs_warm,
);

criterion_main!(jit_speedup_benches);
