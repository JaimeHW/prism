//! Small Integer Cache Benchmarks
//!
//! Measures the performance advantage of cached small integers vs direct construction.
//!
//! # Key Metrics
//!
//! - Cache lookup time: Target < 1ns
//! - Cache vs construction speedup: Target > 2x for hot integers
//! - Cache miss overhead: Minimal (single branch)

use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use prism_core::Value;
use prism_core::small_int_cache::SmallIntCache;

// =============================================================================
// Cache Lookup Benchmarks
// =============================================================================

fn bench_cache_lookup(c: &mut Criterion) {
    let mut group = c.benchmark_group("cache_lookup");

    // Single hot value (0)
    group.bench_function("zero", |b| b.iter(|| black_box(SmallIntCache::get(0))));

    // Common loop counter
    group.bench_function("ten", |b| b.iter(|| black_box(SmallIntCache::get(10))));

    // Boolean-like values
    group.bench_function("one", |b| b.iter(|| black_box(SmallIntCache::get(1))));

    // Negative value
    group.bench_function("negative_one", |b| {
        b.iter(|| black_box(SmallIntCache::get(-1)))
    });

    // Boundary value
    group.bench_function("max_cached", |b| {
        b.iter(|| black_box(SmallIntCache::get(256)))
    });

    group.finish();
}

// =============================================================================
// Cache vs Direct Construction
// =============================================================================

fn bench_cache_vs_direct(c: &mut Criterion) {
    let mut group = c.benchmark_group("cache_vs_direct");

    // Compare cache lookup vs direct Value::int() construction
    group.bench_function("cached_zero", |b| {
        b.iter(|| black_box(SmallIntCache::get(0).unwrap()))
    });

    group.bench_function("direct_zero", |b| {
        b.iter(|| black_box(Value::int(0).unwrap()))
    });

    group.bench_function("cached_42", |b| {
        b.iter(|| black_box(SmallIntCache::get(42).unwrap()))
    });

    group.bench_function("direct_42", |b| {
        b.iter(|| black_box(Value::int(42).unwrap()))
    });

    // Unchecked variants (no bounds check)
    group.bench_function("cached_unchecked_100", |b| {
        b.iter(|| black_box(SmallIntCache::get_unchecked(100)))
    });

    group.bench_function("direct_unchecked_100", |b| {
        b.iter(|| black_box(Value::int_unchecked(100)))
    });

    group.finish();
}

// =============================================================================
// Cache Miss Performance
// =============================================================================

fn bench_cache_miss(c: &mut Criterion) {
    let mut group = c.benchmark_group("cache_miss");

    // Value just outside cache range
    group.bench_function("just_outside_max", |b| {
        b.iter(|| black_box(SmallIntCache::get(257)))
    });

    group.bench_function("just_outside_min", |b| {
        b.iter(|| black_box(SmallIntCache::get(-6)))
    });

    // Large value
    group.bench_function("large_value", |b| {
        b.iter(|| black_box(SmallIntCache::get(1_000_000)))
    });

    group.finish();
}

// =============================================================================
// Batch Operations
// =============================================================================

fn bench_batch_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_operations");

    // Create many cached integers
    group.bench_function("create_0_to_255_cached", |b| {
        b.iter(|| {
            let mut sum = 0u64;
            for i in 0..256 {
                let v = SmallIntCache::get(i).unwrap();
                sum = sum.wrapping_add(v.raw_bits());
            }
            black_box(sum)
        })
    });

    group.bench_function("create_0_to_255_direct", |b| {
        b.iter(|| {
            let mut sum = 0u64;
            for i in 0..256 {
                let v = Value::int(i).unwrap();
                sum = sum.wrapping_add(v.raw_bits());
            }
            black_box(sum)
        })
    });

    // Loop iteration pattern (common in Python)
    group.bench_function("loop_counter_pattern", |b| {
        b.iter(|| {
            let mut values = Vec::with_capacity(100);
            for i in 0..100 {
                values.push(SmallIntCache::get(i).unwrap());
            }
            black_box(values)
        })
    });

    group.finish();
}

// =============================================================================
// JIT Integration (cache_ptr)
// =============================================================================

fn bench_jit_integration(c: &mut Criterion) {
    let mut group = c.benchmark_group("jit_integration");

    // Direct pointer access (simulating JIT code)
    group.bench_function("ptr_access", |b| {
        let ptr = SmallIntCache::cache_ptr();
        let index = 5; // value 0

        b.iter(|| {
            let value = unsafe { *ptr.add(index) };
            black_box(value)
        })
    });

    // Compare to SmallIntCache::get
    group.bench_function("api_access", |b| {
        b.iter(|| black_box(SmallIntCache::get(0).unwrap()))
    });

    group.finish();
}

// =============================================================================
// Scaling Tests
// =============================================================================

fn bench_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("scaling");

    // Different values across the cache range
    for value in [-5, 0, 50, 128, 200, 256].iter() {
        group.bench_with_input(BenchmarkId::new("cached", value), value, |b, &val| {
            b.iter(|| black_box(SmallIntCache::get(val)))
        });
    }

    group.finish();
}

// =============================================================================
// Criterion Groups
// =============================================================================

criterion_group!(
    small_int_benches,
    bench_cache_lookup,
    bench_cache_vs_direct,
    bench_cache_miss,
    bench_batch_operations,
    bench_jit_integration,
    bench_scaling,
);

criterion_main!(small_int_benches);
