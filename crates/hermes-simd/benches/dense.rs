//! Criterion benchmarks for dense SIMD operations.
//!
//! Groups:
//! - `sum_f32`: horizontal sum over varying lengths.
//! - `dot_f32`: dot product (fused multiply-add) over varying lengths.
//! - `elementwise_mul_f32`: in-place elementwise multiplication.
//! - `axpy_rows_batch_f32`: fused dense row-panel accumulation.
//!
//! Throughput is reported in `elements/second`, enabling direct comparison
//! across sizes and SIMD widths.

use criterion::{
    black_box, criterion_group, criterion_main, BatchSize, BenchmarkId, Criterion, Throughput,
};
use hermes_simd::{axpy_rows, axpy_rows_batch, dot, elementwise_mul, gemv, sum};

const SIZES: &[usize] = &[256, 1024, 4096, 16384];
const AXPY_BATCH_CASES: &[AxpyBatchCase] = &[
    AxpyBatchCase {
        rows: 8,
        depth: 8,
        cols: 256,
        row_stride: 272,
    },
    AxpyBatchCase {
        rows: 16,
        depth: 16,
        cols: 512,
        row_stride: 544,
    },
];

#[derive(Clone, Copy)]
struct AxpyBatchCase {
    rows: usize,
    depth: usize,
    cols: usize,
    row_stride: usize,
}

impl AxpyBatchCase {
    fn label(self) -> String {
        format!("rows_{}_depth_{}_cols_{}", self.rows, self.depth, self.cols)
    }

    fn work_items(self) -> u64 {
        (self.rows * self.depth * self.cols) as u64
    }
}

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

/// Register-blocked GEMV (`y = A·x`) vs a scalar row-by-row reference, over
/// square `n × n` matrices. Exposes the SIMD speedup and guards against
/// regression in the dispatched kernel (the instrument measures the production
/// code path, never a tuned body).
fn bench_gemv_f32(c: &mut Criterion) {
    let mut group = c.benchmark_group("gemv_f32");
    for &n in SIZES {
        group.throughput(Throughput::Elements((n as u64) * (n as u64)));
        let a: Vec<f32> = (0..n * n).map(|i| (i % 17) as f32 * 0.01 - 0.5).collect();
        let x: Vec<f32> = (0..n).map(|i| (i % 11) as f32 * 0.1 - 0.3).collect();

        group.bench_with_input(BenchmarkId::new("simd", n), &n, |bench, &n| {
            bench.iter(|| {
                let mut y = vec![0.0f32; n];
                gemv(black_box(&a), black_box(&x), black_box(&mut y), n, n)
                    .expect("invariant: benchmark extents are valid");
                black_box(y)
            })
        });

        group.bench_with_input(BenchmarkId::new("scalar_ref", n), &n, |bench, &n| {
            bench.iter(|| {
                let mut y = vec![0.0f32; n];
                for r in 0..n {
                    let row = &a[r * n..r * n + n];
                    let mut acc = 0.0f32;
                    for (&av, &xv) in row.iter().zip(x.iter()) {
                        acc += av * xv;
                    }
                    y[r] = acc;
                }
                black_box(y)
            })
        });
    }
    group.finish();
}

fn bench_axpy_rows_batch_f32(c: &mut Criterion) {
    let mut group = c.benchmark_group("axpy_rows_batch_f32");
    for &case in AXPY_BATCH_CASES {
        group.throughput(Throughput::Elements(case.work_items()));
        let label = case.label();
        let alphas: Vec<f32> = (0..case.rows * case.depth)
            .map(|idx| idx as f32 * 0.001 - 0.25)
            .collect();
        let x_panel: Vec<f32> = (0..case.depth * case.cols)
            .map(|idx| idx as f32 * 0.002 - 0.5)
            .collect();
        let initial_out: Vec<f32> = (0..case.rows * case.row_stride)
            .map(|idx| idx as f32 * 0.0005)
            .collect();

        group.bench_with_input(
            BenchmarkId::new("repeated_axpy_rows", &label),
            &case,
            |bench, case| {
                bench.iter_batched(
                    || initial_out.clone(),
                    |mut out| {
                        for shared in 0..case.depth {
                            let alpha_start = shared * case.rows;
                            let x_start = shared * case.cols;
                            axpy_rows(
                                black_box(&alphas[alpha_start..alpha_start + case.rows]),
                                black_box(&x_panel[x_start..x_start + case.cols]),
                                black_box(&mut out),
                                case.row_stride,
                                case.rows,
                                case.cols,
                            )
                            .expect("invariant: benchmark extents are valid");
                        }
                        black_box(out)
                    },
                    BatchSize::SmallInput,
                );
            },
        );

        group.bench_with_input(BenchmarkId::new("batch", &label), &case, |bench, case| {
            bench.iter_batched(
                || initial_out.clone(),
                |mut out| {
                    axpy_rows_batch(
                        black_box(&alphas),
                        black_box(&x_panel),
                        black_box(&mut out),
                        case.row_stride,
                        case.rows,
                        case.depth,
                        case.cols,
                    )
                    .expect("invariant: benchmark extents are valid");
                    black_box(out)
                },
                BatchSize::SmallInput,
            );
        });
    }
    group.finish();
}

criterion_group!(
    dense_benches,
    bench_sum_f32,
    bench_dot_f32,
    bench_elementwise_mul_f32,
    bench_gemv_f32,
    bench_axpy_rows_batch_f32
);
criterion_main!(dense_benches);
