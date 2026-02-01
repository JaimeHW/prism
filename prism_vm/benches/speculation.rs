//! Speculation Pipeline Performance Benchmarks
//!
//! Measures execution performance of speculative operations.

use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use prism_compiler::bytecode::{CodeFlags, CodeObject, Instruction, Opcode, Register};
use prism_core::Value;
use prism_vm::{JitConfig, VirtualMachine};
use std::sync::Arc;

// =============================================================================
// Benchmark Helpers
// =============================================================================

fn create_int_add_code(a: i64, b: i64) -> Arc<CodeObject> {
    let instructions = vec![
        Instruction::op_di(Opcode::LoadConst, Register(0), 0),
        Instruction::op_di(Opcode::LoadConst, Register(1), 1),
        Instruction::op_dss(Opcode::Add, Register(2), Register(0), Register(1)),
        Instruction::op_d(Opcode::Return, Register(2)),
    ];

    let constants = vec![Value::int(a).unwrap(), Value::int(b).unwrap()];

    Arc::new(CodeObject {
        name: "bench_int_add".into(),
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
        qualname: "bench_int_add".into(),
        flags: CodeFlags::NONE,
        first_lineno: 1,
    })
}

fn create_float_add_code(a: f64, b: f64) -> Arc<CodeObject> {
    let instructions = vec![
        Instruction::op_di(Opcode::LoadConst, Register(0), 0),
        Instruction::op_di(Opcode::LoadConst, Register(1), 1),
        Instruction::op_dss(Opcode::Add, Register(2), Register(0), Register(1)),
        Instruction::op_d(Opcode::Return, Register(2)),
    ];

    let constants = vec![Value::float(a), Value::float(b)];

    Arc::new(CodeObject {
        name: "bench_float_add".into(),
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
        qualname: "bench_float_add".into(),
        flags: CodeFlags::NONE,
        first_lineno: 1,
    })
}

fn create_mul_code(a: i64, b: i64) -> Arc<CodeObject> {
    let instructions = vec![
        Instruction::op_di(Opcode::LoadConst, Register(0), 0),
        Instruction::op_di(Opcode::LoadConst, Register(1), 1),
        Instruction::op_dss(Opcode::Mul, Register(2), Register(0), Register(1)),
        Instruction::op_d(Opcode::Return, Register(2)),
    ];

    let constants = vec![Value::int(a).unwrap(), Value::int(b).unwrap()];

    Arc::new(CodeObject {
        name: "bench_mul".into(),
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
        qualname: "bench_mul".into(),
        flags: CodeFlags::NONE,
        first_lineno: 1,
    })
}

fn create_hot_loop_code(n: i64) -> Arc<CodeObject> {
    let mut instructions = Vec::new();
    let mut constants = Vec::new();

    constants.push(Value::int(0).unwrap());
    instructions.push(Instruction::op_di(Opcode::LoadConst, Register(0), 0));

    constants.push(Value::int(n).unwrap());
    instructions.push(Instruction::op_di(Opcode::LoadConst, Register(1), 1));

    constants.push(Value::int(1).unwrap());
    instructions.push(Instruction::op_di(Opcode::LoadConst, Register(2), 2));

    let loop_header = instructions.len();

    instructions.push(Instruction::op_dss(
        Opcode::Lt,
        Register(3),
        Register(0),
        Register(1),
    ));

    let jump_if_false_idx = instructions.len();
    instructions.push(Instruction::op_di(Opcode::JumpIfFalse, Register(3), 0));

    instructions.push(Instruction::op_dss(
        Opcode::Add,
        Register(0),
        Register(0),
        Register(2),
    ));

    let jump_back_idx = instructions.len();
    instructions.push(Instruction::op_di(Opcode::Jump, Register(0), 0));

    let next_ip = jump_back_idx + 1;
    let offset = (loop_header as i32) - (next_ip as i32);
    instructions[jump_back_idx] =
        Instruction::op_di(Opcode::Jump, Register(0), offset as i16 as u16);

    let exit_idx = instructions.len();

    let next_ip_fwd = jump_if_false_idx + 1;
    let fwd_offset = (exit_idx as i32) - (next_ip_fwd as i32);
    instructions[jump_if_false_idx] =
        Instruction::op_di(Opcode::JumpIfFalse, Register(3), fwd_offset as i16 as u16);

    instructions.push(Instruction::op_d(Opcode::Return, Register(0)));

    Arc::new(CodeObject {
        name: "bench_hot_loop".into(),
        register_count: 4,
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
        qualname: "bench_hot_loop".into(),
        flags: CodeFlags::NONE,
        first_lineno: 1,
    })
}

// =============================================================================
// Benchmarks
// =============================================================================

fn bench_integer_arithmetic(c: &mut Criterion) {
    let mut group = c.benchmark_group("integer_arithmetic");

    // Benchmark integer addition
    group.bench_function("add", |b| {
        let config = JitConfig::for_testing();
        let mut vm = VirtualMachine::with_jit_config(config);
        let code = create_int_add_code(100, 200);

        b.iter(|| black_box(vm.execute(Arc::clone(&code)).unwrap()));
    });

    // Benchmark integer multiplication
    group.bench_function("mul", |b| {
        let config = JitConfig::for_testing();
        let mut vm = VirtualMachine::with_jit_config(config);
        let code = create_mul_code(7, 8);

        b.iter(|| black_box(vm.execute(Arc::clone(&code)).unwrap()));
    });

    group.finish();
}

fn bench_float_arithmetic(c: &mut Criterion) {
    let mut group = c.benchmark_group("float_arithmetic");

    group.bench_function("add", |b| {
        let config = JitConfig::for_testing();
        let mut vm = VirtualMachine::with_jit_config(config);
        let code = create_float_add_code(3.14159, 2.71828);

        b.iter(|| black_box(vm.execute(Arc::clone(&code)).unwrap()));
    });

    group.finish();
}

fn bench_hot_loops(c: &mut Criterion) {
    let mut group = c.benchmark_group("hot_loops");

    for iterations in [100, 1000, 10000].iter() {
        group.bench_with_input(
            BenchmarkId::from_parameter(iterations),
            iterations,
            |b, &n| {
                let config = JitConfig::for_testing();
                let mut vm = VirtualMachine::with_jit_config(config);
                let code = create_hot_loop_code(n);

                b.iter(|| black_box(vm.execute(Arc::clone(&code)).unwrap()));
            },
        );
    }

    group.finish();
}

fn bench_speculation_warmup(c: &mut Criterion) {
    let mut group = c.benchmark_group("speculation_warmup");

    // Measure cost of first execution (cold) vs subsequent (warm)
    group.bench_function("cold_first_execution", |b| {
        let code = create_int_add_code(1, 2);

        b.iter(|| {
            let config = JitConfig::for_testing();
            let mut vm = VirtualMachine::with_jit_config(config);
            black_box(vm.execute(Arc::clone(&code)).unwrap())
        });
    });

    group.bench_function("warm_repeated_execution", |b| {
        let config = JitConfig::for_testing();
        let mut vm = VirtualMachine::with_jit_config(config);
        let code = create_int_add_code(1, 2);

        // Warm up
        for _ in 0..50 {
            let _ = vm.execute(Arc::clone(&code));
        }

        b.iter(|| black_box(vm.execute(Arc::clone(&code)).unwrap()));
    });

    group.finish();
}

fn bench_vm_overhead(c: &mut Criterion) {
    let mut group = c.benchmark_group("vm_overhead");

    // Minimal code: just return a constant
    group.bench_function("return_constant", |b| {
        let instructions = vec![
            Instruction::op_di(Opcode::LoadConst, Register(0), 0),
            Instruction::op_d(Opcode::Return, Register(0)),
        ];
        let constants = vec![Value::int(42).unwrap()];

        let code = Arc::new(CodeObject {
            name: "bench_return_const".into(),
            register_count: 1,
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
            qualname: "bench_return_const".into(),
            flags: CodeFlags::NONE,
            first_lineno: 1,
        });

        let config = JitConfig::for_testing();
        let mut vm = VirtualMachine::with_jit_config(config);

        b.iter(|| black_box(vm.execute(Arc::clone(&code)).unwrap()));
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_integer_arithmetic,
    bench_float_arithmetic,
    bench_hot_loops,
    bench_speculation_warmup,
    bench_vm_overhead,
);

criterion_main!(benches);
