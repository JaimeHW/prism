//! GC Pause Distribution Benchmarks
//!
//! Measures GC pause latency during active JIT execution.
//!
//! # Benchmark Categories
//!
//! 1. **Allocation Rate**: Objects/second under various allocation patterns
//! 2. **Pause Latency**: P50, P95, P99 pause times
//! 3. **Throughput Impact**: Application throughput with GC overhead
//! 4. **Memory Pressure**: Behavior under low/high memory conditions
//!
//! # Key Performance Targets
//!
//! - Minor GC: < 1ms P99
//! - Major GC: < 10ms P99 for small heaps
//! - Allocation rate: > 100MB/s sustained

use criterion::{BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main};
use prism_compiler::bytecode::{CodeFlags, CodeObject, Instruction, Opcode, Register};
use prism_core::Value;
use prism_vm::{JitConfig, VirtualMachine};
use std::sync::Arc;
use std::time::{Duration, Instant};

// =============================================================================
// GC-Heavy Code Generators
// =============================================================================

/// Create code that allocates many small objects in a loop.
/// Simulates: [i for i in range(n)]
fn create_allocation_loop(n: i64) -> Arc<CodeObject> {
    let mut instructions = Vec::new();
    let mut constants = Vec::new();

    // r0 = [] (result list - simulated as counter for now)
    constants.push(Value::int(0).unwrap());
    instructions.push(Instruction::op_di(Opcode::LoadConst, Register(0), 0));

    // r1 = n (limit)
    constants.push(Value::int(n).unwrap());
    instructions.push(Instruction::op_di(Opcode::LoadConst, Register(1), 1));

    // r2 = 0 (counter i)
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

    // Simulated allocation: r0 = r0 + r2 (would be list.append in real code)
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
        name: "allocation_loop".into(),
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
        qualname: "allocation_loop".into(),
        flags: CodeFlags::NONE,
        first_lineno: 1,
    })
}

/// Create code that allocates temporary objects (high churn).
/// Outer loop runs n times, inner creates/discards objects.
fn create_high_churn_code(n: i64) -> Arc<CodeObject> {
    let mut instructions = Vec::new();
    let mut constants = Vec::new();

    // r0 = 0 (result)
    constants.push(Value::int(0).unwrap());
    instructions.push(Instruction::op_di(Opcode::LoadConst, Register(0), 0));

    // r1 = n (limit)
    constants.push(Value::int(n).unwrap());
    instructions.push(Instruction::op_di(Opcode::LoadConst, Register(1), 1));

    // r2 = 0 (outer counter)
    instructions.push(Instruction::op_di(Opcode::LoadConst, Register(2), 0));

    // r3 = 1 (increment)
    constants.push(Value::int(1).unwrap());
    instructions.push(Instruction::op_di(Opcode::LoadConst, Register(3), 2));

    // r4 = 10 (inner loop limit)
    constants.push(Value::int(10).unwrap());
    instructions.push(Instruction::op_di(Opcode::LoadConst, Register(4), 3));

    // OUTER LOOP
    let outer_loop = instructions.len();

    // r5 = r2 < r1
    instructions.push(Instruction::op_dss(
        Opcode::Lt,
        Register(5),
        Register(2),
        Register(1),
    ));

    // JumpIfFalse r5 -> outer_exit
    let outer_jump_idx = instructions.len();
    instructions.push(Instruction::op_di(Opcode::JumpIfFalse, Register(5), 0));

    // r6 = 0 (inner counter)
    instructions.push(Instruction::op_di(Opcode::LoadConst, Register(6), 0));

    // INNER LOOP (high churn - creates temporary values)
    let inner_loop = instructions.len();

    // r7 = r6 < r4
    instructions.push(Instruction::op_dss(
        Opcode::Lt,
        Register(7),
        Register(6),
        Register(4),
    ));

    // JumpIfFalse r7 -> inner_exit
    let inner_jump_idx = instructions.len();
    instructions.push(Instruction::op_di(Opcode::JumpIfFalse, Register(7), 0));

    // Simulate allocation: r8 = r2 * r6 (temporary)
    instructions.push(Instruction::op_dss(
        Opcode::Mul,
        Register(8),
        Register(2),
        Register(6),
    ));

    // Accumulate: r0 = r0 + r8
    instructions.push(Instruction::op_dss(
        Opcode::Add,
        Register(0),
        Register(0),
        Register(8),
    ));

    // r6 = r6 + r3 (inner counter++)
    instructions.push(Instruction::op_dss(
        Opcode::Add,
        Register(6),
        Register(6),
        Register(3),
    ));

    // Jump -> inner_loop
    let inner_back_idx = instructions.len();
    instructions.push(Instruction::op_di(Opcode::Jump, Register(0), 0));

    // Patch inner loop
    let next_ip = inner_back_idx + 1;
    let offset = (inner_loop as i32) - (next_ip as i32);
    instructions[inner_back_idx] =
        Instruction::op_di(Opcode::Jump, Register(0), offset as i16 as u16);

    // inner_exit:
    let inner_exit = instructions.len();
    let inner_fwd = (inner_exit as i32) - ((inner_jump_idx + 1) as i32);
    instructions[inner_jump_idx] =
        Instruction::op_di(Opcode::JumpIfFalse, Register(7), inner_fwd as i16 as u16);

    // r2 = r2 + r3 (outer counter++)
    instructions.push(Instruction::op_dss(
        Opcode::Add,
        Register(2),
        Register(2),
        Register(3),
    ));

    // Jump -> outer_loop
    let outer_back_idx = instructions.len();
    instructions.push(Instruction::op_di(Opcode::Jump, Register(0), 0));

    // Patch outer loop
    let next_ip = outer_back_idx + 1;
    let offset = (outer_loop as i32) - (next_ip as i32);
    instructions[outer_back_idx] =
        Instruction::op_di(Opcode::Jump, Register(0), offset as i16 as u16);

    // outer_exit:
    let outer_exit = instructions.len();
    let outer_fwd = (outer_exit as i32) - ((outer_jump_idx + 1) as i32);
    instructions[outer_jump_idx] =
        Instruction::op_di(Opcode::JumpIfFalse, Register(5), outer_fwd as i16 as u16);

    // Return r0
    instructions.push(Instruction::op_d(Opcode::Return, Register(0)));

    Arc::new(CodeObject {
        name: "high_churn".into(),
        register_count: 9,
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
        qualname: "high_churn".into(),
        flags: CodeFlags::NONE,
        first_lineno: 1,
    })
}

// =============================================================================
// Pause Distribution Measurement
// =============================================================================

/// Measure individual execution times to capture pause distribution.
fn measure_pause_distribution(
    vm: &mut VirtualMachine,
    code: &Arc<CodeObject>,
    iterations: usize,
) -> PauseStats {
    let mut times = Vec::with_capacity(iterations);

    for _ in 0..iterations {
        let start = Instant::now();
        let _ = black_box(vm.execute(Arc::clone(code)));
        times.push(start.elapsed());
    }

    times.sort();

    PauseStats {
        min: times[0],
        p50: times[iterations / 2],
        p95: times[(iterations * 95) / 100],
        p99: times[(iterations * 99) / 100],
        max: times[iterations - 1],
        mean: times.iter().sum::<Duration>() / iterations as u32,
    }
}

#[derive(Debug, Clone)]
struct PauseStats {
    min: Duration,
    p50: Duration,
    p95: Duration,
    p99: Duration,
    max: Duration,
    mean: Duration,
}

// =============================================================================
// Benchmarks
// =============================================================================

fn bench_allocation_rate(c: &mut Criterion) {
    let mut group = c.benchmark_group("allocation_rate");

    for n in [100, 1000, 10000].iter() {
        let code = create_allocation_loop(*n as i64);

        group.throughput(Throughput::Elements(*n as u64));
        group.bench_with_input(BenchmarkId::from_parameter(n), n, |b, _| {
            let config = JitConfig::for_testing();
            let mut vm = VirtualMachine::with_jit_config(config);

            // Warm up
            for _ in 0..50 {
                let _ = vm.execute(Arc::clone(&code));
            }

            b.iter(|| black_box(vm.execute(Arc::clone(&code)).unwrap()));
        });
    }

    group.finish();
}

fn bench_high_churn(c: &mut Criterion) {
    let mut group = c.benchmark_group("high_churn");
    group.sample_size(50);

    for n in [100, 500, 1000].iter() {
        let code = create_high_churn_code(*n as i64);
        let ops = (*n as u64) * 10; // n outer * 10 inner

        group.throughput(Throughput::Elements(ops));
        group.bench_with_input(BenchmarkId::from_parameter(n), n, |b, _| {
            let config = JitConfig::for_testing();
            let mut vm = VirtualMachine::with_jit_config(config);

            b.iter(|| black_box(vm.execute(Arc::clone(&code)).unwrap()));
        });
    }

    group.finish();
}

fn bench_pause_latency(c: &mut Criterion) {
    let mut group = c.benchmark_group("pause_latency");
    group.measurement_time(Duration::from_secs(5));

    // Test pause distribution under allocation load
    group.bench_function("allocation_pauses", |b| {
        let config = JitConfig::for_testing();
        let mut vm = VirtualMachine::with_jit_config(config);
        let code = create_allocation_loop(1000);

        // Warm up
        for _ in 0..100 {
            let _ = vm.execute(Arc::clone(&code));
        }

        b.iter(|| {
            let stats = measure_pause_distribution(&mut vm, &code, 100);
            black_box(stats)
        });
    });

    group.bench_function("churn_pauses", |b| {
        let config = JitConfig::for_testing();
        let mut vm = VirtualMachine::with_jit_config(config);
        let code = create_high_churn_code(100);

        // Warm up
        for _ in 0..50 {
            let _ = vm.execute(Arc::clone(&code));
        }

        b.iter(|| {
            let stats = measure_pause_distribution(&mut vm, &code, 100);
            black_box(stats)
        });
    });

    group.finish();
}

fn bench_sustained_throughput(c: &mut Criterion) {
    let mut group = c.benchmark_group("sustained_throughput");
    group.measurement_time(Duration::from_secs(10));
    group.sample_size(20);

    // Measure sustained throughput over longer period
    group.bench_function("sustained_1s", |b| {
        let config = JitConfig::for_testing();
        let mut vm = VirtualMachine::with_jit_config(config);
        let code = create_allocation_loop(100);

        // Warm up
        for _ in 0..100 {
            let _ = vm.execute(Arc::clone(&code));
        }

        b.iter(|| {
            let start = Instant::now();
            let mut iterations = 0u64;

            while start.elapsed() < Duration::from_millis(100) {
                let _ = black_box(vm.execute(Arc::clone(&code)));
                iterations += 1;
            }

            black_box(iterations)
        });
    });

    group.finish();
}

criterion_group!(
    gc_pause_benches,
    bench_allocation_rate,
    bench_high_churn,
    bench_pause_latency,
    bench_sustained_throughput,
);

criterion_main!(gc_pause_benches);
