//! Criterion benchmarks for interleaved complex kernels.
//!
//! Groups (runtime-dispatched backend vs always-available `Scalar` backend):
//! - `complex_dot`: interleaved complex dot product.
//! - `real_complex_dot`: real samples against interleaved complex weights.
//! - `complex_mul_assign`: in-place interleaved complex multiply.
//! - `real_mul_interleave`: real multiply into interleaved complex storage,
//!   compared with the prior copy/multiply/interleave materialization shape.
//!
//! Sizes are complex-pair counts; throughput is reported in complex pairs
//! per second. The scalar series doubles as a regression reference: a
//! runtime-dispatch series falling to scalar throughput indicates a broken
//! dispatch or kernel regression.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use hermes_simd::{
    elementwise_mul, interleaved_complex_dot, interleaved_complex_dot_runtime,
    interleaved_complex_mul_assign, interleaved_complex_mul_assign_runtime,
    real_interleaved_complex_dot_runtime, real_mul_to_interleaved_complex_runtime, Scalar,
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

fn make_real_input(pairs: usize) -> (Vec<f64>, Vec<f64>) {
    let real = (0..pairs)
        .map(|i| ((i % 17) as f64) * 0.25 - 2.0)
        .collect::<Vec<_>>();
    let mut interleaved = Vec::with_capacity(pairs * 2);
    for &value in &real {
        interleaved.push(value);
        interleaved.push(0.0);
    }
    (real, interleaved)
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

fn bench_real_complex_dot(c: &mut Criterion) {
    let mut group = c.benchmark_group("real_complex_dot");
    for &pairs in PAIR_SIZES {
        group.throughput(Throughput::Elements(pairs as u64));
        let (real, interleaved) = make_real_input(pairs);
        let (_, weights) = make_inputs(pairs);
        group.bench_with_input(
            BenchmarkId::new("materialized_runtime", pairs),
            &pairs,
            |bench, _| {
                bench.iter(|| {
                    interleaved_complex_dot_runtime::<f64, false>(
                        black_box(&interleaved),
                        black_box(&weights),
                    )
                    .unwrap()
                });
            },
        );
        group.bench_with_input(
            BenchmarkId::new("real_runtime", pairs),
            &pairs,
            |bench, _| {
                bench.iter(|| {
                    real_interleaved_complex_dot_runtime::<f64>(
                        black_box(&real),
                        black_box(&weights),
                    )
                    .unwrap()
                });
            },
        );
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

fn three_pass_real_mul_interleave(
    input: &[f64],
    factors: &[f64],
    copy: &mut [f64],
    products: &mut [f64],
    output: &mut [f64],
) {
    copy.copy_from_slice(input);
    elementwise_mul(copy, factors, products).unwrap();
    for (&product, pair) in products.iter().zip(output.chunks_exact_mut(2)) {
        pair[0] = product;
        pair[1] = 0.0;
    }
}

fn bench_real_mul_interleave(c: &mut Criterion) {
    let mut group = c.benchmark_group("real_mul_interleave_f64");
    for &len in &[64_usize, 256, 1_024, 4_096] {
        group.throughput(Throughput::Elements(len as u64));
        let input: Vec<f64> = (0..len)
            .map(|index| (index % 31) as f64 * 0.125 - 1.5)
            .collect();
        let factors: Vec<f64> = (0..len)
            .map(|index| (index % 17) as f64 * 0.0625 - 0.25)
            .collect();
        let mut copy = vec![0.0; len];
        let mut products = vec![0.0; len];
        let mut three_pass_output = vec![0.0; len * 2];
        let mut fused_output = vec![0.0; len * 2];

        three_pass_real_mul_interleave(
            &input,
            &factors,
            &mut copy,
            &mut products,
            &mut three_pass_output,
        );
        real_mul_to_interleaved_complex_runtime(&input, &factors, &mut fused_output).unwrap();
        assert_eq!(three_pass_output, fused_output);

        group.bench_with_input(BenchmarkId::new("three_pass", len), &len, |bench, _| {
            bench.iter(|| {
                three_pass_real_mul_interleave(
                    black_box(&input),
                    black_box(&factors),
                    black_box(&mut copy),
                    black_box(&mut products),
                    black_box(&mut three_pass_output),
                );
            });
        });
        group.bench_with_input(BenchmarkId::new("fused", len), &len, |bench, _| {
            bench.iter(|| {
                real_mul_to_interleaved_complex_runtime(
                    black_box(&input),
                    black_box(&factors),
                    black_box(&mut fused_output),
                )
                .unwrap();
            });
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_complex_dot,
    bench_real_complex_dot,
    bench_complex_mul_assign,
    bench_real_mul_interleave
);
criterion_main!(benches);
