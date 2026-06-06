//! Dense SIMD benchmarks: sum and dot product with backend comparison.
//!
//! Benchmark groups:
//! - `Dense Sum f32` — `sum::<f32>` dispatch vs scalar iterator, sizes 256–1M
//! - `Dense Dot f32` — `dot::<f32>` dispatch vs scalar iterator, sizes 256–65536
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use hermes_simd::{dot, sum};

#[inline(never)]
fn scalar_sum(data: &[f32]) -> f32 {
    data.iter().sum()
}

#[inline(never)]
fn scalar_dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

fn bench_sum(c: &mut Criterion) {
    let mut group = c.benchmark_group("Dense Sum f32");
    for &size in &[256usize, 1024, 16384, 65536, 1 << 20] {
        let data = vec![1.0f32; size];
        group.throughput(Throughput::Elements(size as u64));

        group.bench_with_input(BenchmarkId::new("scalar_iter", size), &size, |b, _| {
            b.iter(|| scalar_sum(&data))
        });
        group.bench_with_input(BenchmarkId::new("dispatch", size), &size, |b, _| {
            b.iter(|| sum::<f32>(&data))
        });
    }
    group.finish();
}

fn bench_dot(c: &mut Criterion) {
    let mut group = c.benchmark_group("Dense Dot f32");
    for &size in &[256usize, 1024, 16384, 65536] {
        let a = vec![1.0f32; size];
        let b = vec![2.0f32; size];
        group.throughput(Throughput::Elements(size as u64));

        group.bench_with_input(BenchmarkId::new("scalar_iter", size), &size, |bencher, _| {
            bencher.iter(|| scalar_dot(&a, &b))
        });
        group.bench_with_input(BenchmarkId::new("dispatch", size), &size, |bencher, _| {
            bencher.iter(|| dot::<f32>(&a, &b).unwrap())
        });
    }
    group.finish();
}

criterion_group!(benches, bench_sum, bench_dot);
criterion_main!(benches);