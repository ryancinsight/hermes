//! Shared dense SIMD benchmark families.

use criterion::{black_box, BenchmarkId, Criterion, Throughput};
use hermes_simd::{dot, sum};

#[inline(never)]
fn scalar_sum<T: Copy + core::ops::Add<Output = T>>(data: &[T], zero: T) -> T {
    data.iter()
        .copied()
        .fold(zero, |acc, x| black_box(acc + black_box(x)))
}

#[inline(never)]
fn scalar_dot<T>(a: &[T], b: &[T], zero: T) -> T
where
    T: Copy + core::ops::Add<Output = T> + core::ops::Mul<Output = T>,
{
    a.iter().zip(b.iter()).fold(zero, |acc, (&x, &y)| {
        black_box(acc + black_box(x) * black_box(y))
    })
}

pub fn bench_sum_f32(c: &mut Criterion) {
    let mut group = c.benchmark_group("Dense Sum f32");
    for &size in &[256usize, 1024, 16384, 65536, 1 << 20] {
        let data = vec![1.0f32; size];
        group.throughput(Throughput::Elements(size as u64));
        group.bench_with_input(BenchmarkId::new("scalar_iter", size), &size, |b, _| {
            b.iter(|| scalar_sum(black_box(&data), 0.0f32))
        });
        group.bench_with_input(BenchmarkId::new("dispatch", size), &size, |b, _| {
            b.iter(|| sum::<f32>(black_box(&data)))
        });
    }
    group.finish();
}

pub fn bench_sum_f64(c: &mut Criterion) {
    let mut group = c.benchmark_group("Dense Sum f64");
    for &size in &[256usize, 1024, 16384, 65536, 1 << 20] {
        let data = vec![1.0f64; size];
        group.throughput(Throughput::Elements(size as u64));
        group.bench_with_input(BenchmarkId::new("scalar_iter", size), &size, |b, _| {
            b.iter(|| scalar_sum(black_box(&data), 0.0f64))
        });
        group.bench_with_input(BenchmarkId::new("dispatch", size), &size, |b, _| {
            b.iter(|| sum::<f64>(black_box(&data)))
        });
    }
    group.finish();
}

pub fn bench_sum_i32(c: &mut Criterion) {
    let mut group = c.benchmark_group("Dense Sum i32");
    for &size in &[256usize, 1024, 16384, 65536, 1 << 20] {
        let data = vec![1i32; size];
        group.throughput(Throughput::Elements(size as u64));
        group.bench_with_input(BenchmarkId::new("scalar_iter", size), &size, |b, _| {
            b.iter(|| scalar_sum(black_box(&data), 0i32))
        });
        group.bench_with_input(BenchmarkId::new("dispatch", size), &size, |b, _| {
            b.iter(|| sum::<i32>(black_box(&data)))
        });
    }
    group.finish();
}

pub fn bench_dot_f32(c: &mut Criterion) {
    let mut group = c.benchmark_group("Dense Dot f32");
    for &size in &[256usize, 1024, 16384, 65536] {
        let a = vec![1.0f32; size];
        let b = vec![2.0f32; size];
        group.throughput(Throughput::Elements(size as u64));
        group.bench_with_input(BenchmarkId::new("scalar_iter", size), &size, |bencher, _| {
            bencher.iter(|| scalar_dot(black_box(&a), black_box(&b), 0.0f32))
        });
        group.bench_with_input(BenchmarkId::new("dispatch", size), &size, |bencher, _| {
            bencher.iter(|| dot::<f32>(black_box(&a), black_box(&b)).unwrap())
        });
    }
    group.finish();
}

pub fn bench_dot_f64(c: &mut Criterion) {
    let mut group = c.benchmark_group("Dense Dot f64");
    for &size in &[256usize, 1024, 16384, 65536] {
        let a = vec![1.0f64; size];
        let b = vec![2.0f64; size];
        group.throughput(Throughput::Elements(size as u64));
        group.bench_with_input(BenchmarkId::new("scalar_iter", size), &size, |bencher, _| {
            bencher.iter(|| scalar_dot(black_box(&a), black_box(&b), 0.0f64))
        });
        group.bench_with_input(BenchmarkId::new("dispatch", size), &size, |bencher, _| {
            bencher.iter(|| dot::<f64>(black_box(&a), black_box(&b)).unwrap())
        });
    }
    group.finish();
}

pub fn bench_dot_i32(c: &mut Criterion) {
    let mut group = c.benchmark_group("Dense Dot i32");
    for &size in &[256usize, 1024, 16384, 65536] {
        let a = vec![3i32; size];
        let b = vec![2i32; size];
        group.throughput(Throughput::Elements(size as u64));
        group.bench_with_input(BenchmarkId::new("scalar_iter", size), &size, |bencher, _| {
            bencher.iter(|| scalar_dot(black_box(&a), black_box(&b), 0i32))
        });
        group.bench_with_input(BenchmarkId::new("dispatch", size), &size, |bencher, _| {
            bencher.iter(|| dot::<i32>(black_box(&a), black_box(&b)).unwrap())
        });
    }
    group.finish();
}

pub fn bench_dot_reduced_precision(c: &mut Criterion) {
    let mut group = c.benchmark_group("Dense Dot f16");
    for &size in &[256usize, 16384, 65536] {
        let a = vec![eunomia::F16::from_f32(1.5); size];
        let b = vec![eunomia::F16::from_f32(0.5); size];
        group.throughput(Throughput::Elements(size as u64));
        group.bench_with_input(BenchmarkId::new("dispatch", size), &size, |bencher, _| {
            bencher.iter(|| dot::<eunomia::F16>(black_box(&a), black_box(&b)).unwrap())
        });
    }
    group.finish();

    let mut group = c.benchmark_group("Dense Dot bf16");
    for &size in &[256usize, 16384, 65536] {
        let a = vec![eunomia::Bf16::from_f32(1.5); size];
        let b = vec![eunomia::Bf16::from_f32(0.5); size];
        group.throughput(Throughput::Elements(size as u64));
        group.bench_with_input(BenchmarkId::new("dispatch", size), &size, |bencher, _| {
            bencher.iter(|| dot::<eunomia::Bf16>(black_box(&a), black_box(&b)).unwrap())
        });
    }
    group.finish();
}
