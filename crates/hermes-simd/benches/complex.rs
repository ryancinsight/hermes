//! Criterion benchmarks for interleaved complex kernels.
//!
//! Groups (runtime-dispatched backend vs always-available `Scalar` backend):
//! - `complex_dot`: interleaved complex dot product.
//! - `complex_mul_assign`: in-place interleaved complex multiply.
//!
//! Sizes are complex-pair counts; throughput is reported in complex pairs
//! per second. The scalar series doubles as a regression reference: a
//! runtime-dispatch series falling to scalar throughput indicates a broken
//! dispatch or kernel regression.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use hermes_simd::{
    interleaved_complex_dot, interleaved_complex_dot_runtime, interleaved_complex_mul_assign,
    interleaved_complex_mul_assign_runtime, Scalar,
};

const PAIR_SIZES: &[usize] = &[256, 1024, 4096, 16384];

fn make_inputs(pairs: usize) -> (Vec<f64>, Vec<f64>) {
    let a = (0..pairs * 2)
        .map(|i| ((i % 17) as f64) * 0.25 - 2.0)
        .collect();
    let b = (0..pairs * 2)
        .map(|i| ((i % 13) as f64) * 0.5 - 3.0)
        .collect();
    (a, b)
}

fn bench_complex_dot(c: &mut Criterion) {
    let mut group = c.benchmark_group("complex_dot");
    for &pairs in PAIR_SIZES {
        group.throughput(Throughput::Elements(pairs as u64));
        let (a, b) = make_inputs(pairs);
        group.bench_with_input(BenchmarkId::new("runtime", pairs), &pairs, |bench, _| {
            bench.iter(|| {
                interleaved_complex_dot_runtime::<f64, false>(black_box(&a), black_box(&b)).unwrap()
            });
        });
        group.bench_with_input(BenchmarkId::new("scalar", pairs), &pairs, |bench, _| {
            bench.iter(|| {
                interleaved_complex_dot::<f64, Scalar, false>(black_box(&a), black_box(&b)).unwrap()
            });
        });
    }
    group.finish();
}

fn bench_complex_mul_assign(c: &mut Criterion) {
    let mut group = c.benchmark_group("complex_mul_assign");
    for &pairs in PAIR_SIZES {
        group.throughput(Throughput::Elements(pairs as u64));
        let (a, b) = make_inputs(pairs);
        let mut buf = a.clone();
        group.bench_with_input(BenchmarkId::new("runtime", pairs), &pairs, |bench, _| {
            bench.iter(|| {
                buf.copy_from_slice(&a);
                interleaved_complex_mul_assign_runtime::<f64, true>(
                    black_box(&mut buf),
                    black_box(&b),
                )
                .unwrap();
            });
        });
        group.bench_with_input(BenchmarkId::new("scalar", pairs), &pairs, |bench, _| {
            bench.iter(|| {
                buf.copy_from_slice(&a);
                interleaved_complex_mul_assign::<f64, Scalar, true>(
                    black_box(&mut buf),
                    black_box(&b),
                )
                .unwrap();
            });
        });
    }
    group.finish();
}

criterion_group!(benches, bench_complex_dot, bench_complex_mul_assign);
criterion_main!(benches);
