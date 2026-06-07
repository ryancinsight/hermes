#![allow(clippy::same_item_push)]
//! Sparse matrix-vector multiplication benchmarks.
//!
//! Benchmarks SpMV across formats and sparsity levels:
//! - `CSR SpMV` — CSR format at 0%, 50%, 90%, 99% sparsity on a 512x512 matrix
//! - `DenseWithMask SpMV` — same sparsity sweep, dense layout with per-element mask
//! - `Blocked-COO SpMV` — 4x4 and 8x8 block formats at 90% block sparsity
//!
//! Matrix size is fixed at 512x512 to keep benchmark runs short; the relative
//! throughput differences between formats are representative at larger sizes.
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use hermes_simd::{
    spmv_csr, spmv_dense_masked, spmv_bcoo4x4, spmv_bcoo8x8,
};
use hermes_simd_core::sparse::{BlockedCooData, CsrData, DenseWithMaskData};

/// Build a CSR matrix with the given sparsity (fraction of zeros) for an `nxn` matrix.
///
/// Uses a simple LCG for reproducible, allocation-minimal pseudo-random generation.
fn make_csr(n: usize, sparsity: f64) -> (Vec<f32>, Vec<i32>, Vec<i32>) {
    let mut values = Vec::new();
    let mut col_indices = Vec::new();
    let mut row_ptr = vec![0i32];

    let mut rng_state = 12345u64;
    let lcg = |s: &mut u64| -> f64 {
        *s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (*s >> 33) as f64 / (1u64 << 31) as f64
    };

    for _r in 0..n {
        for c in 0..n {
            if lcg(&mut rng_state) >= sparsity {
                values.push(1.0f32);
                col_indices.push(c as i32);
            }
        }
        row_ptr.push(values.len() as i32);
    }
    (values, col_indices, row_ptr)
}

/// Build a DenseWithMask matrix with the given sparsity.
fn make_dense_masked(n: usize, sparsity: f64) -> (Vec<f32>, Vec<bool>) {
    let mut rng = 99991u64;
    let lcg = |s: &mut u64| -> f64 {
        *s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (*s >> 33) as f64 / (1u64 << 31) as f64
    };
    let total = n * n;
    let values: Vec<f32> = (0..total).map(|_| 1.0f32).collect();
    let mask: Vec<bool> = (0..total).map(|_| lcg(&mut rng) >= sparsity).collect();
    (values, mask)
}

/// Build a Blocked-COO matrix at `sparsity` block-level sparsity for const block dims.
fn make_bcoo<const BM: usize, const BN: usize>(
    n: usize,
    sparsity: f64,
) -> (Vec<f32>, Vec<i32>, Vec<i32>, usize) {
    let n_block_rows = n / BM;
    let n_block_cols = n / BN;
    let block_size = BM * BN;

    let mut rng = 77771u64;
    let lcg = |s: &mut u64| -> f64 {
        *s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (*s >> 33) as f64 / (1u64 << 31) as f64
    };

    let mut blocks: Vec<f32> = Vec::new();
    let mut brow: Vec<i32> = Vec::new();
    let mut bcol: Vec<i32> = Vec::new();

    for br in 0..n_block_rows {
        for bc in 0..n_block_cols {
            if lcg(&mut rng) >= sparsity {
                for _ in 0..block_size {
                    blocks.push(1.0f32);
                }
                brow.push((br * BM) as i32);
                bcol.push((bc * BN) as i32);
            }
        }
    }
    let nblocks = brow.len();
    (blocks, brow, bcol, nblocks)
}

fn bench_csr_spmv(c: &mut Criterion) {
    let mut group = c.benchmark_group("CSR SpMV f32");
    let n = 512usize;
    let x = vec![1.0f32; n];

    for sparsity in [0.0f64, 0.5, 0.9, 0.99] {
        let (vals, cols, ptr) = make_csr(n, sparsity);
        let mut y = vec![0.0f32; n];
        let label = format!("sparsity_{:.0}pct", sparsity * 100.0);
        group.throughput(Throughput::Elements(n as u64 * n as u64));
        group.bench_with_input(
            BenchmarkId::new("csr", &label),
            &sparsity,
            |b, _| {
                b.iter(|| {
                    y.fill(0.0);
                    spmv_csr::<f32>(
                        CsrData::new(&vals[..], &cols[..], &ptr[..], n, n),
                        &x,
                        &mut y,
                    );
                })
            },
        );
    }
    group.finish();
}

fn bench_dense_masked_spmv(c: &mut Criterion) {
    let mut group = c.benchmark_group("DenseWithMask SpMV f32");
    let n = 512usize;
    let x = vec![1.0f32; n];

    for sparsity in [0.0f64, 0.5, 0.9, 0.99] {
        let (vals, mask) = make_dense_masked(n, sparsity);
        let mut y = vec![0.0f32; n];
        let label = format!("sparsity_{:.0}pct", sparsity * 100.0);
        group.throughput(Throughput::Elements(n as u64 * n as u64));
        group.bench_with_input(
            BenchmarkId::new("dense_masked", &label),
            &sparsity,
            |b, _| {
                b.iter(|| {
                    y.fill(0.0);
                    spmv_dense_masked::<f32>(
                        DenseWithMaskData::new(&vals[..], &mask[..], n, n),
                        &x,
                        &mut y,
                    );
                })
            },
        );
    }
    group.finish();
}

fn bench_bcoo_spmv(c: &mut Criterion) {
    let mut group = c.benchmark_group("Blocked-COO SpMV f32");
    let n = 512usize;
    let x = vec![1.0f32; n];
    let sparsity = 0.9f64;
    group.throughput(Throughput::Elements(n as u64 * n as u64));

    // 4x4 blocks
    {
        let (blocks, brow, bcol, nblocks) = make_bcoo::<4, 4>(n, sparsity);
        let mut y = vec![0.0f32; n];
        group.bench_function("bcoo_4x4", |b| {
            b.iter(|| {
                y.fill(0.0);
                spmv_bcoo4x4::<f32>(
                    BlockedCooData::new(&blocks[..], &brow[..], &bcol[..], nblocks, n, n),
                    &x,
                    &mut y,
                );
            })
        });
    }

    // 8x8 blocks
    {
        let (blocks, brow, bcol, nblocks) = make_bcoo::<8, 8>(n, sparsity);
        let mut y = vec![0.0f32; n];
        group.bench_function("bcoo_8x8", |b| {
            b.iter(|| {
                y.fill(0.0);
                spmv_bcoo8x8::<f32>(
                    BlockedCooData::new(&blocks[..], &brow[..], &bcol[..], nblocks, n, n),
                    &x,
                    &mut y,
                );
            })
        });
    }
    group.finish();
}

criterion_group!(benches, bench_csr_spmv, bench_dense_masked_spmv, bench_bcoo_spmv);
criterion_main!(benches);