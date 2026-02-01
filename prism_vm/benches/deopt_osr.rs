//! Deoptimization and OSR Entry Benchmarks
//!
//! Measures the latency overhead of deoptimization recovery and
//! On-Stack Replacement (OSR) entry/exit transitions.
//!
//! # Benchmark Categories
//!
//! 1. **Guard Failure Recovery**: Time to restore interpreter state after speculation failure
//! 2. **OSR Cold Entry**: Cost of first OSR into JIT-compiled loop
//! 3. **OSR Warm Entry**: Cost of subsequent OSR entries (hot path)
//! 4. **OSR Exit Recovery**: Cost of returning from optimized code
//!
//! # Performance Targets
//!
//! - Guard failure recovery: < 100μs
//! - OSR cold entry: < 1ms
//! - OSR warm entry: < 10μs

use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use prism_compiler::bytecode::{CodeFlags, CodeObject, Instruction, Opcode, Register};
use prism_core::Value;
use prism_vm::{JitConfig, VirtualMachine};
use std::sync::Arc;

// =============================================================================
// Guard/Deopt Test Code Generators
// =============================================================================

/// Create code that exercises type specialization.
/// When input types are stable, guards pass. When types change, deopt occurs.
fn create_type_stable_code() -> Arc<CodeObject> {
    let instructions = vec![
        Instruction::op_di(Opcode::LoadConst, Register(0), 0),
        Instruction::op_di(Opcode::LoadConst, Register(1), 1),
        Instruction::op_dss(Opcode::Add, Register(2), Register(0), Register(1)),
        Instruction::op_d(Opcode::Return, Register(2)),
    ];

    let constants = vec![Value::int(100).unwrap(), Value::int(200).unwrap()];

    Arc::new(CodeObject {
        name: "type_stable".into(),
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
        qualname: "type_stable".into(),
        flags: CodeFlags::NONE,
        first_lineno: 1,
    })
}

/// Create code that simulates polymorphic operations (type changes).
fn create_type_varying_code(use_float: bool) -> Arc<CodeObject> {
    let instructions = vec![
        Instruction::op_di(Opcode::LoadConst, Register(0), 0),
        Instruction::op_di(Opcode::LoadConst, Register(1), 1),
        Instruction::op_dss(Opcode::Add, Register(2), Register(0), Register(1)),
        Instruction::op_d(Opcode::Return, Register(2)),
    ];

    let constants = if use_float {
        vec![Value::float(100.5), Value::float(200.5)]
    } else {
        vec![Value::int(100).unwrap(), Value::int(200).unwrap()]
    };

    Arc::new(CodeObject {
        name: "type_varying".into(),
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
        qualname: "type_varying".into(),
        flags: CodeFlags::NONE,
        first_lineno: 1,
    })
}

/// Create a loop that is a candidate for OSR entry.
fn create_osr_candidate_loop(n: i64) -> Arc<CodeObject> {
    let mut instructions = Vec::new();
    let mut constants = Vec::new();

    // r0 = 0 (sum)
    constants.push(Value::int(0).unwrap());
    instructions.push(Instruction::op_di(Opcode::LoadConst, Register(0), 0));

    // r1 = n (limit)
    constants.push(Value::int(n).unwrap());
    instructions.push(Instruction::op_di(Opcode::LoadConst, Register(1), 1));

    // r2 = 0 (counter)
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
        name: "osr_loop".into(),
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
        qualname: "osr_loop".into(),
        flags: CodeFlags::NONE,
        first_lineno: 1,
    })
}

// =============================================================================
// Benchmarks
// =============================================================================

fn bench_guard_speculation(c: &mut Criterion) {
    let mut group = c.benchmark_group("guard_speculation");

    // Type-stable path (guards always pass)
    group.bench_function("stable_guards", |b| {
        let config = JitConfig::for_testing();
        let mut vm = VirtualMachine::with_jit_config(config);
        let code = create_type_stable_code();

        // Warm up to specialize
        for _ in 0..100 {
            let _ = vm.execute(Arc::clone(&code));
        }

        b.iter(|| black_box(vm.execute(Arc::clone(&code)).unwrap()));
    });

    // Measure first execution before specialization
    group.bench_function("first_execution", |b| {
        let code = create_type_stable_code();

        b.iter(|| {
            let config = JitConfig::for_testing();
            let mut vm = VirtualMachine::with_jit_config(config);
            black_box(vm.execute(Arc::clone(&code)).unwrap())
        });
    });

    group.finish();
}

fn bench_osr_entry(c: &mut Criterion) {
    let mut group = c.benchmark_group("osr_entry");

    // Cold OSR entry (first time entering JIT)
    group.bench_function("cold_entry", |b| {
        let code = create_osr_candidate_loop(10000);

        b.iter(|| {
            let config = JitConfig::for_testing();
            let mut vm = VirtualMachine::with_jit_config(config);
            black_box(vm.execute(Arc::clone(&code)).unwrap())
        });
    });

    // Warm OSR entry (JIT already compiled)
    group.bench_function("warm_entry", |b| {
        let config = JitConfig::for_testing();
        let mut vm = VirtualMachine::with_jit_config(config);
        let code = create_osr_candidate_loop(10000);

        // Warm up to trigger JIT compilation
        for _ in 0..20 {
            let _ = vm.execute(Arc::clone(&code));
        }

        b.iter(|| black_box(vm.execute(Arc::clone(&code)).unwrap()));
    });

    group.finish();
}

fn bench_osr_varying_sizes(c: &mut Criterion) {
    let mut group = c.benchmark_group("osr_loop_sizes");

    for n in [100, 1000, 10000, 100000].iter() {
        let code = create_osr_candidate_loop(*n as i64);

        // Cold (new VM each time)
        group.bench_with_input(BenchmarkId::new("cold", n), n, |b, _| {
            b.iter(|| {
                let config = JitConfig::for_testing();
                let mut vm = VirtualMachine::with_jit_config(config);
                black_box(vm.execute(Arc::clone(&code)).unwrap())
            });
        });

        // Warm (reuse VM)
        group.bench_with_input(BenchmarkId::new("warm", n), n, |b, _| {
            let config = JitConfig::for_testing();
            let mut vm = VirtualMachine::with_jit_config(config);

            // Warm up
            for _ in 0..20 {
                let _ = vm.execute(Arc::clone(&code));
            }

            b.iter(|| black_box(vm.execute(Arc::clone(&code)).unwrap()));
        });
    }

    group.finish();
}

fn bench_speculation_vs_interpreter(c: &mut Criterion) {
    let mut group = c.benchmark_group("speculation_vs_interpreter");

    let code = create_osr_candidate_loop(10000);

    // With JIT and speculation
    group.bench_function("jit_speculated", |b| {
        let config = JitConfig::for_testing();
        let mut vm = VirtualMachine::with_jit_config(config);

        // Warm up
        for _ in 0..50 {
            let _ = vm.execute(Arc::clone(&code));
        }

        b.iter(|| black_box(vm.execute(Arc::clone(&code)).unwrap()));
    });

    // Interpreter only (no JIT)
    group.bench_function("interpreter_only", |b| {
        let config = JitConfig::disabled();
        let mut vm = VirtualMachine::with_jit_config(config);

        b.iter(|| black_box(vm.execute(Arc::clone(&code)).unwrap()));
    });

    group.finish();
}

fn bench_warmup_iterations(c: &mut Criterion) {
    let mut group = c.benchmark_group("warmup_iterations");
    group.sample_size(20);

    // Measure how many executions until JIT kicks in
    for warmup_count in [1, 5, 10, 20, 50, 100].iter() {
        let code = create_osr_candidate_loop(1000);

        group.bench_with_input(
            BenchmarkId::from_parameter(warmup_count),
            warmup_count,
            |b, &n| {
                b.iter(|| {
                    let config = JitConfig::for_testing();
                    let mut vm = VirtualMachine::with_jit_config(config);

                    // Run warmup iterations
                    for _ in 0..n {
                        let _ = vm.execute(Arc::clone(&code));
                    }

                    // Measure post-warmup execution
                    black_box(vm.execute(Arc::clone(&code)).unwrap())
                });
            },
        );
    }

    group.finish();
}

criterion_group!(
    deopt_osr_benches,
    bench_guard_speculation,
    bench_osr_entry,
    bench_osr_varying_sizes,
    bench_speculation_vs_interpreter,
    bench_warmup_iterations,
);

criterion_main!(deopt_osr_benches);
