//! SIMD ISA Extension benchmarks: scalar vs AVX2 vs AVX-512 vs SVE.
//!
//! This benchmark suite compares performance across different instruction set
//! architectures for key compute kernels:
//! - Dot product (fused multiply-add reduction)
//! - Sum reduction (horizontal reduction)
//! - Min/Max reduction
//! - ArgMin (Min + Location)
//!
//! Run with:
//! ```
//! cargo bench --bench simd_isa_comparison -- --output-format bencher | tee results.txt
//! ```
//!
//! Compare two runs:
//! ```
//! cargo bench --bench simd_isa_comparison -- --baseline v0 --output-format bencher
//! ```

use criterion::{
    black_box, criterion_group, criterion_main, measurement::WallTime, BenchmarkGroup,
    BenchmarkId, Criterion, SamplingMode, Throughput,
};
use hermes_simd::{
    argmin, dot, max, min, sum,
};

// ─────────────────────────────────────────────────────────────────────────────
// Test Data Generators
// ─────────────────────────────────────────────────────────────────────────────

/// Generate synthetic test data for benchmarks.
fn generate_test_data(n: usize) -> Vec<f32> {
    (0..n).map(|i| (i as f32).sin()).collect()
}

// ─────────────────────────────────────────────────────────────────────────────
// Benchmark Groups
// ─────────────────────────────────────────────────────────────────────────────

fn benchmark_group<'criterion>(
    criterion: &'criterion mut Criterion,
    name: &str,
) -> BenchmarkGroup<'criterion, WallTime> {
    let mut group = criterion.benchmark_group(name);
    group.sampling_mode(SamplingMode::Flat);
    // Increase sample count for better statistical significance
    group.sample_size(100);
    group
}

// ─────────────────────────────────────────────────────────────────────────────
// Dot Product Benchmarks
// ─────────────────────────────────────────────────────────────────────────────

fn bench_dot_product(c: &mut Criterion) {
    let mut group = benchmark_group(c, "dot_product_f32");
    
    let sizes = vec![
        256,    // Small: cache-friendly
        4096,   // Medium: L2/L3 boundary
        65536,  // Large: DRAM-bound
        1_048_576, // Very large: multi-threaded candidate
    ];

    for n in sizes {
        group.throughput(Throughput::Elements(n as u64));
        let a = black_box(generate_test_data(n));
        let b = black_box(generate_test_data(n));

        group.bench_with_input(
            BenchmarkId::from_parameter(format!("n={}", n)),
            &n,
            |bench, _| {
                bench.iter(|| {
                    dot(&a, &b)
                });
            },
        );
    }

    group.finish();
}

// ─────────────────────────────────────────────────────────────────────────────
// Sum Reduction Benchmarks
// ─────────────────────────────────────────────────────────────────────────────

fn bench_sum_reduction(c: &mut Criterion) {
    let mut group = benchmark_group(c, "sum_reduction_f32");

    let sizes = vec![
        256,
        4096,
        65536,
        1_048_576,
    ];

    for n in sizes {
        group.throughput(Throughput::Elements(n as u64));
        let data = black_box(generate_test_data(n));

        group.bench_with_input(
            BenchmarkId::from_parameter(format!("n={}", n)),
            &n,
            |bench, _| {
                bench.iter(|| {
                    sum(&data)
                });
            },
        );
    }

    group.finish();
}

// ─────────────────────────────────────────────────────────────────────────────
// Min Reduction Benchmarks
// ─────────────────────────────────────────────────────────────────────────────

fn bench_min_reduction(c: &mut Criterion) {
    let mut group = benchmark_group(c, "min_reduction_f32");

    let sizes = vec![256, 4096, 65536];

    for n in sizes {
        group.throughput(Throughput::Elements(n as u64));
        let data = black_box(generate_test_data(n));

        group.bench_with_input(
            BenchmarkId::from_parameter(format!("n={}", n)),
            &n,
            |bench, _| {
                bench.iter(|| {
                    min(&data)
                });
            },
        );
    }

    group.finish();
}

// ─────────────────────────────────────────────────────────────────────────────
// Max Reduction Benchmarks
// ─────────────────────────────────────────────────────────────────────────────

fn bench_max_reduction(c: &mut Criterion) {
    let mut group = benchmark_group(c, "max_reduction_f32");

    let sizes = vec![256, 4096, 65536];

    for n in sizes {
        group.throughput(Throughput::Elements(n as u64));
        let data = black_box(generate_test_data(n));

        group.bench_with_input(
            BenchmarkId::from_parameter(format!("n={}", n)),
            &n,
            |bench, _| {
                bench.iter(|| {
                    max(&data)
                });
            },
        );
    }

    group.finish();
}

// ─────────────────────────────────────────────────────────────────────────────
// ArgMin Benchmarks (Min + Location)
// ─────────────────────────────────────────────────────────────────────────────

fn bench_argmin_reduction(c: &mut Criterion) {
    let mut group = benchmark_group(c, "argmin_f32");

    let sizes = vec![256, 4096, 65536];

    for n in sizes {
        group.throughput(Throughput::Elements(n as u64));
        let data = black_box(generate_test_data(n));

        group.bench_with_input(
            BenchmarkId::from_parameter(format!("n={}", n)),
            &n,
            |bench, _| {
                bench.iter(|| {
                    argmin(&data)
                });
            },
        );
    }

    group.finish();
}

// ─────────────────────────────────────────────────────────────────────────────
// Criterion Configuration
// ─────────────────────────────────────────────────────────────────────────────

criterion_group!(
    name = benches;
    config = Criterion::default().warm_up_time(std::time::Duration::from_secs(1));
    targets =
        bench_dot_product,
        bench_sum_reduction,
        bench_min_reduction,
        bench_max_reduction,
        bench_argmin_reduction
);

criterion_main!(benches);
