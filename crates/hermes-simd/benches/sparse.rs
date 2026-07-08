//! Criterion benchmarks for sparse SIMD operations.
//!
//! Groups:
//! - `spmv_csr_f32`: CSR SpMV at varying matrix densities.
//!
//! A 512×512 sparse matrix is constructed in-memory at 1%, 5%, and 10% fill
//! density. Throughput is reported in non-zero floating-point multiply-adds
//! per second.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use hermes_simd::{spmv_csr, CsrData, ValidatedData};

/// Construct a random-ish CSR sparse matrix of shape `(nrows, ncols)` at `density` fill.
/// Uses a deterministic pseudo-random pattern (modular arithmetic, no external RNG dep).
fn build_csr(nrows: usize, ncols: usize, density: f64) -> (Vec<i32>, Vec<i32>, Vec<f32>) {
    let mut row_ptr = Vec::with_capacity(nrows + 1);
    let mut col_idx: Vec<i32> = Vec::new();
    let mut values: Vec<f32> = Vec::new();

    row_ptr.push(0i32);
    let nnz_per_row = ((ncols as f64) * density).ceil() as usize;
    let nnz_per_row = nnz_per_row.max(1);

    for row in 0..nrows {
        // Deterministic spread: pick `nnz_per_row` evenly distributed columns with jitter.
        let step = ncols / nnz_per_row;
        let step = step.max(1);
        let mut count = 0usize;
        let mut col = (row * 7 + 3) % ncols;
        for _ in 0..nnz_per_row {
            col_idx.push(col as i32);
            // Value: row+col to avoid trivial cancellation.
            values.push((row + col + 1) as f32);
            col = (col + step) % ncols;
            count += 1;
        }
        row_ptr.push(row_ptr.last().unwrap() + count as i32);
    }

    (row_ptr, col_idx, values)
}

const NROWS: usize = 512;
const NCOLS: usize = 512;
const DENSITIES: &[(&str, f64)] = &[("1pct", 0.01), ("5pct", 0.05), ("10pct", 0.10)];

fn bench_spmv_csr_f32(c: &mut Criterion) {
    let mut group = c.benchmark_group("spmv_csr_f32");

    for &(label, density) in DENSITIES {
        let (row_ptr, col_idx, values) = build_csr(NROWS, NCOLS, density);
        let nnz = values.len();
        // Throughput: one FMA per non-zero.
        group.throughput(Throughput::Elements(nnz as u64));

        let x: Vec<f32> = (0..NCOLS).map(|i| i as f32 / NCOLS as f32).collect();
        let mut y: Vec<f32> = vec![0.0f32; NROWS];
        let data = ValidatedData::new(CsrData::new(&values, &col_idx, &row_ptr, NROWS, NCOLS))
            .expect("benchmark CSR fixture must validate");

        group.bench_with_input(BenchmarkId::new("scalar", label), &label, |bench, _| {
            bench.iter(|| {
                // Reset output each iteration to get consistent results.
                y.iter_mut().for_each(|v| *v = 0.0);
                spmv_csr::<f32>(black_box(data.clone()), black_box(&x), black_box(&mut y))
            })
        });
    }

    group.finish();
}

criterion_group!(sparse_benches, bench_spmv_csr_f32);
criterion_main!(sparse_benches);
