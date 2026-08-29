//! Criterion benchmarks for sparse SIMD operations.
//!
//! Groups:
//! - `spmv_csr_f32`: CSR `SpMV` at varying matrix densities.
//! - `spmv_csr_gather_bound_f32`: CSR `SpMV` with a 64 MiB indirect operand.
//!
//! A 512×512 sparse matrix is constructed in-memory at 1%, 5%, and 10% fill
//! density. Throughput is reported in non-zero floating-point multiply-adds
//! per second. The gather-bound group uses 128 nonzeros per row and a
//! deterministic permutation of a 64 MiB dense operand so the indirect working
//! set exceeds the development host's last-level cache.

use std::time::Duration;

use criterion::{
    black_box, criterion_group, criterion_main, BenchmarkId, Criterion, SamplingMode, Throughput,
};
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

const GATHER_NCOLS: usize = 1 << 24;
const GATHER_NNZ_PER_ROW: usize = 128;
const GATHER_ROW_COUNTS: &[usize] = &[8_192, 16_384];
const GATHER_COLUMN_MULTIPLIER: usize = 0x9e37_79b1;
const GATHER_COLUMN_OFFSET: usize = 0x85eb_ca6b;

fn build_gather_bound_csr(nrows: usize) -> (Vec<i32>, Vec<i32>, Vec<f32>) {
    let nnz = nrows * GATHER_NNZ_PER_ROW;
    let row_ptr = (0..=nrows)
        .map(|row| {
            i32::try_from(row * GATHER_NNZ_PER_ROW)
                .expect("gather-bound fixture nonzero count must fit i32")
        })
        .collect();
    let col_idx = (0..nnz)
        .map(|linear| {
            let column = linear
                .wrapping_mul(GATHER_COLUMN_MULTIPLIER)
                .wrapping_add(GATHER_COLUMN_OFFSET)
                & (GATHER_NCOLS - 1);
            i32::try_from(column).expect("gather-bound fixture column must fit i32")
        })
        .collect();
    let values = vec![1.0f32; nnz];
    (row_ptr, col_idx, values)
}

fn build_gather_operand() -> Vec<f32> {
    (0..GATHER_NCOLS)
        .map(|index| {
            let byte = u8::try_from(index & 0xff)
                .expect("the gather operand pattern is limited to one byte");
            f32::from(byte) * 0.125
        })
        .collect()
}

fn gather_bound_reference(col_idx: &[i32], x: &[f32]) -> Vec<f32> {
    col_idx
        .chunks_exact(GATHER_NNZ_PER_ROW)
        .map(|row| {
            row.iter().fold(0.0f32, |sum, &column| {
                let column = usize::try_from(column)
                    .expect("validated gather-bound columns are nonnegative");
                sum + x[column]
            })
        })
        .collect()
}

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
                y.fill(0.0);
                spmv_csr::<f32>(black_box(data.clone()), black_box(&x), black_box(&mut y));
            });
        });
    }

    group.finish();
}

/// Measure CSR rows whose indirect operand footprint exceeds the host LLC.
fn bench_spmv_csr_gather_bound_f32(c: &mut Criterion) {
    let mut group = c.benchmark_group("spmv_csr_gather_bound_f32");
    group.sampling_mode(SamplingMode::Flat);
    group.warm_up_time(Duration::from_millis(100));
    group.measurement_time(Duration::from_millis(500));
    group.sample_size(10);

    for &nrows in GATHER_ROW_COUNTS {
        let (row_ptr, col_idx, values) = build_gather_bound_csr(nrows);
        let x = build_gather_operand();
        let expected = gather_bound_reference(&col_idx, &x);
        let mut y = vec![0.0f32; nrows];
        let data = ValidatedData::new(CsrData::new(
            &values,
            &col_idx,
            &row_ptr,
            nrows,
            GATHER_NCOLS,
        ))
        .expect("gather-bound CSR fixture must validate");

        spmv_csr::<f32>(data.clone(), &x, &mut y);
        assert_eq!(y, expected, "gather-bound CSR fixture reference mismatch");

        group.throughput(Throughput::Elements(values.len() as u64));
        group.bench_with_input(BenchmarkId::from_parameter(nrows), &nrows, |bench, _| {
            bench.iter(|| {
                y.fill(0.0);
                spmv_csr::<f32>(black_box(data.clone()), black_box(&x), black_box(&mut y));
                black_box(y.as_slice());
            });
        });
    }

    group.finish();
}

criterion_group!(
    sparse_benches,
    bench_spmv_csr_f32,
    bench_spmv_csr_gather_bound_f32
);
criterion_main!(sparse_benches);
