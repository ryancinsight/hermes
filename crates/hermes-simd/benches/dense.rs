//! Criterion benchmarks for dense SIMD operations.
//!
//! Groups:
//! - `sum_f32`: horizontal sum over varying lengths.
//! - `dot_f32`: dot product (fused multiply-add) over varying lengths.
//! - `elementwise_mul_f32`: in-place elementwise multiplication.
//!
//! Throughput is reported in `elements/second`, enabling direct comparison
//! across sizes and SIMD widths.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use hermes_simd::{dot, elementwise_mul, sum};

const SIZES: &[usize] = &[256, 1024, 4096, 16384];

fn bench_sum_f32(c: &mut Criterion) {
    let mut group = c.benchmark_group("sum_f32");
    for &n in SIZES {
        group.throughput(Throughput::Elements(n as u64));
        let data: Vec<f32> = (0..n).map(|i| i as f32).collect();
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            b.iter(|| sum(black_box(&data)))
        });
    }
    group.finish();
}

fn bench_dot_f32(c: &mut Criterion) {
    let mut group = c.benchmark_group("dot_f32");
    for &n in SIZES {
        group.throughput(Throughput::Elements(n as u64));
        let a: Vec<f32> = (0..n).map(|i| i as f32).collect();
        let b: Vec<f32> = (0..n).map(|i| (n - i) as f32).collect();
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |bench, _| {
            bench.iter(|| dot(black_box(&a), black_box(&b)))
        });
    }
    group.finish();
}

fn bench_elementwise_mul_f32(c: &mut Criterion) {
    let mut group = c.benchmark_group("elementwise_mul_f32");
    for &n in SIZES {
        group.throughput(Throughput::Elements(n as u64));
        let a: Vec<f32> = (0..n).map(|i| i as f32).collect();
        let b: Vec<f32> = (0..n).map(|i| (n - i) as f32).collect();
        let mut out = vec![0.0f32; n];
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |bench, _| {
            bench.iter(|| elementwise_mul(black_box(&a), black_box(&b), black_box(&mut out)))
        });
    }
    group.finish();
}

criterion_group!(
    dense_benches,
    bench_sum_f32,
    bench_dot_f32,
    bench_elementwise_mul_f32
);
criterion_main!(dense_benches);
