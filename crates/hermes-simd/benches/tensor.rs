//! Criterion benchmarks for tensor operations.
//!
//! Groups:
//! - `matmul_f32`: square matrix multiplication at increasing sizes.
//! - `softmax_f32`: numerically stable softmax over flat f32 slices.
//!
//! `batch_matmul` is not included in this suite because it depends on a batch
//! dimension that complicates throughput annotation; it will be a separate group.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use hermes_simd::{TensorView, matmul, softmax_alloc};
use hermes_simd::{Scalar, Unaligned};

/// Square matmul dimensions.
const MATMUL_DIMS: &[usize] = &[32, 64, 128];

/// Flat softmax input lengths.
const SOFTMAX_SIZES: &[usize] = &[256, 1024, 4096];

fn bench_matmul_f32(c: &mut Criterion) {
    let mut group = c.benchmark_group("matmul_f32");
    for &m in MATMUL_DIMS {
        let n = m;
        let k = m;
        // Throughput: m*n output elements, each requiring k multiply-adds.
        group.throughput(Throughput::Elements((m * n * k) as u64));

        let a_data: Vec<f32> = (0..m * k).map(|i| i as f32 / (m as f32)).collect();
        let b_data: Vec<f32> = (0..k * n).map(|i| i as f32 / (k as f32)).collect();
        let mut c_data: Vec<f32> = vec![0.0f32; m * n];

        let a = TensorView::<f32, 2>::new(&a_data, [m, k]).unwrap();
        let b = TensorView::<f32, 2>::new(&b_data, [k, n]).unwrap();

        group.bench_with_input(
            BenchmarkId::new("scalar", m),
            &m,
            |bench, _| {
                bench.iter(|| {
                    let mut c_view = TensorView::<f32, 2>::new_mut(
                        black_box(&mut c_data),
                        [m, n],
                    ).unwrap();
                    matmul::<f32, Scalar, Unaligned>(
                        black_box(&a),
                        black_box(&b),
                        &mut c_view,
                    ).unwrap()
                })
            }
        );
    }
    group.finish();
}

fn bench_softmax_f32(c: &mut Criterion) {
    let mut group = c.benchmark_group("softmax_f32");
    for &n in SOFTMAX_SIZES {
        group.throughput(Throughput::Elements(n as u64));
        let data: Vec<f32> = (0..n).map(|i| i as f32 / (n as f32)).collect();
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |bench, _| {
            bench.iter(|| softmax_alloc::<f32, Scalar, Unaligned>(black_box(&data)))
        });
    }
    group.finish();
}

criterion_group!(tensor_benches, bench_matmul_f32, bench_softmax_f32);
criterion_main!(tensor_benches);
