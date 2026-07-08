//! Sparse matrix-vector multiplication benchmarks.
//!
//! The scalable sweep uses a fixed column count and varies row count plus
//! structural non-zero density. CSR, SELL-p, and Blocked-COO cover 1K, 10K,
//! and 100K rows. Dense-with-mask stores full dense values and masks, so it
//! is capped at 10K rows to keep local memory bounded.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use hermes_simd::{spmv_bcoo, spmv_csr, spmv_dense_masked, spmv_sellp};
use hermes_simd_core::sparse::{
    BlockedCooData, CsrData, DenseWithMaskData, SellPData, ValidatedData,
};

const NCOLS: usize = 1024;
const ROW_SWEEP: [usize; 3] = [1024, 10_000, 100_000];
const DENSE_ROW_SWEEP: [usize; 2] = [1024, 10_000];
const DENSITIES: [f64; 3] = [0.001, 0.01, 0.10];

#[derive(Clone, Copy)]
struct MatrixCase {
    nrows: usize,
    density: f64,
}

impl MatrixCase {
    fn label(self) -> String {
        format!("rows_{}_density_{:.1}pct", self.nrows, self.density * 100.0)
    }

    fn nnz_per_row(self) -> usize {
        ((NCOLS as f64 * self.density).round() as usize).clamp(1, NCOLS)
    }

    fn dense_len(self) -> usize {
        self.nrows * NCOLS
    }

    fn logical_elements(self) -> u64 {
        self.dense_len() as u64
    }
}

fn col_for(row: usize, ordinal: usize) -> usize {
    (row.wrapping_mul(1_103_515_245).wrapping_add(ordinal * 97)) & (NCOLS - 1)
}

fn make_csr(case: MatrixCase) -> (Vec<f32>, Vec<i32>, Vec<i32>) {
    let nnz_per_row = case.nnz_per_row();
    let nnz = case.nrows * nnz_per_row;
    let mut values = Vec::with_capacity(nnz);
    let mut col_indices = Vec::with_capacity(nnz);
    let mut row_ptr = Vec::with_capacity(case.nrows + 1);
    row_ptr.push(0);

    for row in 0..case.nrows {
        for ordinal in 0..nnz_per_row {
            values.push(1.0);
            col_indices.push(col_for(row, ordinal) as i32);
        }
        row_ptr.push(values.len() as i32);
    }
    (values, col_indices, row_ptr)
}

fn make_sellp<const C: usize>(case: MatrixCase) -> (Vec<f32>, Vec<i32>, Vec<i32>, Vec<i32>) {
    let nnz_per_row = case.nnz_per_row();
    let nslices = case.nrows.div_ceil(C);
    let padded_rows = nslices * C;
    let entries = padded_rows * nnz_per_row;
    let mut values = Vec::with_capacity(entries);
    let mut col_indices = Vec::with_capacity(entries);
    let mut slice_ptr = Vec::with_capacity(nslices + 1);
    let mut slice_col_count = Vec::with_capacity(nslices);

    slice_ptr.push(0);
    for slice in 0..nslices {
        slice_col_count.push(nnz_per_row as i32);
        for ordinal in 0..nnz_per_row {
            for lane in 0..C {
                let row = slice * C + lane;
                if row < case.nrows {
                    values.push(1.0);
                    col_indices.push(col_for(row, ordinal) as i32);
                } else {
                    values.push(0.0);
                    col_indices.push(0);
                }
            }
        }
        slice_ptr.push(values.len() as i32);
    }
    (values, col_indices, slice_ptr, slice_col_count)
}

fn make_bcoo<const BM: usize, const BN: usize>(
    case: MatrixCase,
) -> (Vec<f32>, Vec<i32>, Vec<i32>, usize) {
    let n_block_rows = case.nrows.div_ceil(BM);
    let n_block_cols = NCOLS / BN;
    let blocks_per_row =
        ((n_block_cols as f64 * case.density).round() as usize).clamp(1, n_block_cols);
    let nblocks = n_block_rows * blocks_per_row;
    let block_size = BM * BN;
    let mut blocks = Vec::with_capacity(nblocks * block_size);
    let mut block_rows = Vec::with_capacity(nblocks);
    let mut block_cols = Vec::with_capacity(nblocks);

    for block_row in 0..n_block_rows {
        for ordinal in 0..blocks_per_row {
            let block_col = (block_row.wrapping_mul(17).wrapping_add(ordinal * 13)) % n_block_cols;
            blocks.extend(std::iter::repeat_n(1.0, block_size));
            block_rows.push((block_row * BM) as i32);
            block_cols.push((block_col * BN) as i32);
        }
    }
    (blocks, block_rows, block_cols, nblocks)
}

fn make_dense_masked(case: MatrixCase) -> (Vec<f32>, Vec<bool>) {
    let values = vec![1.0; case.dense_len()];
    let active_period = (1.0 / case.density).round() as usize;
    let mask = (0..case.dense_len())
        .map(|idx| idx % active_period == 0)
        .collect();
    (values, mask)
}

fn bench_csr_spmv(c: &mut Criterion) {
    let mut group = c.benchmark_group("CSR SpMV f32 scalability");
    for nrows in ROW_SWEEP {
        let x = vec![1.0f32; NCOLS];
        for density in DENSITIES {
            let case = MatrixCase { nrows, density };
            let (values, cols, row_ptr) = make_csr(case);
            let data = ValidatedData::new(CsrData::new(&values, &cols, &row_ptr, nrows, NCOLS))
                .expect("benchmark CSR fixture must validate");
            let mut y = vec![0.0f32; nrows];
            group.throughput(Throughput::Elements(case.logical_elements()));
            group.bench_with_input(BenchmarkId::new("csr", case.label()), &case, |bench, _| {
                bench.iter(|| {
                    y.fill(0.0);
                    spmv_csr::<f32>(data.clone(), black_box(&x), black_box(&mut y));
                })
            });
        }
    }
    group.finish();
}

fn bench_sellp_spmv(c: &mut Criterion) {
    let mut group = c.benchmark_group("SELL-p SpMV f32 scalability");
    for nrows in ROW_SWEEP {
        let x = vec![1.0f32; NCOLS];
        for density in DENSITIES {
            let case = MatrixCase { nrows, density };
            let (values4, cols4, ptr4, count4) = make_sellp::<4>(case);
            let data4 = ValidatedData::new(SellPData::new(
                &values4, &cols4, &ptr4, &count4, nrows, NCOLS,
            ))
            .expect("benchmark SELL-p fixture must validate");
            let mut y4 = vec![0.0f32; nrows];
            group.throughput(Throughput::Elements(case.logical_elements()));
            group.bench_with_input(
                BenchmarkId::new("sellp4", case.label()),
                &case,
                |bench, _| {
                    bench.iter(|| {
                        y4.fill(0.0);
                        spmv_sellp::<f32, 4>(data4.clone(), black_box(&x), black_box(&mut y4));
                    })
                },
            );

            let (values8, cols8, ptr8, count8) = make_sellp::<8>(case);
            let data8 = ValidatedData::new(SellPData::new(
                &values8, &cols8, &ptr8, &count8, nrows, NCOLS,
            ))
            .expect("benchmark SELL-p fixture must validate");
            let mut y8 = vec![0.0f32; nrows];
            group.bench_with_input(
                BenchmarkId::new("sellp8", case.label()),
                &case,
                |bench, _| {
                    bench.iter(|| {
                        y8.fill(0.0);
                        spmv_sellp::<f32, 8>(data8.clone(), black_box(&x), black_box(&mut y8));
                    })
                },
            );
        }
    }
    group.finish();
}

fn bench_bcoo_spmv(c: &mut Criterion) {
    let mut group = c.benchmark_group("Blocked-COO SpMV f32 scalability");
    for nrows in ROW_SWEEP {
        let x = vec![1.0f32; NCOLS];
        for density in DENSITIES {
            let case = MatrixCase { nrows, density };
            let (blocks4, brow4, bcol4, nblocks4) = make_bcoo::<4, 4>(case);
            let data4 = ValidatedData::new(BlockedCooData::new(
                &blocks4, &brow4, &bcol4, nblocks4, nrows, NCOLS,
            ))
            .expect("benchmark Blocked-COO fixture must validate");
            let mut y4 = vec![0.0f32; nrows];
            group.throughput(Throughput::Elements(case.logical_elements()));
            group.bench_with_input(
                BenchmarkId::new("bcoo4x4", case.label()),
                &case,
                |bench, _| {
                    bench.iter(|| {
                        y4.fill(0.0);
                        spmv_bcoo::<f32, 4, 4>(data4.clone(), black_box(&x), black_box(&mut y4));
                    })
                },
            );

            let (blocks8, brow8, bcol8, nblocks8) = make_bcoo::<8, 8>(case);
            let data8 = ValidatedData::new(BlockedCooData::new(
                &blocks8, &brow8, &bcol8, nblocks8, nrows, NCOLS,
            ))
            .expect("benchmark Blocked-COO fixture must validate");
            let mut y8 = vec![0.0f32; nrows];
            group.bench_with_input(
                BenchmarkId::new("bcoo8x8", case.label()),
                &case,
                |bench, _| {
                    bench.iter(|| {
                        y8.fill(0.0);
                        spmv_bcoo::<f32, 8, 8>(data8.clone(), black_box(&x), black_box(&mut y8));
                    })
                },
            );
        }
    }
    group.finish();
}

fn bench_dense_masked_spmv(c: &mut Criterion) {
    let mut group = c.benchmark_group("DenseWithMask SpMV f32 scalability");
    for nrows in DENSE_ROW_SWEEP {
        let x = vec![1.0f32; NCOLS];
        for density in DENSITIES {
            let case = MatrixCase { nrows, density };
            let (values, mask) = make_dense_masked(case);
            let data = DenseWithMaskData::new(&values, &mask, nrows, NCOLS);
            let mut y = vec![0.0f32; nrows];
            group.throughput(Throughput::Elements(case.logical_elements()));
            group.bench_with_input(
                BenchmarkId::new("dense_masked", case.label()),
                &case,
                |bench, _| {
                    bench.iter(|| {
                        y.fill(0.0);
                        spmv_dense_masked::<f32>(data.clone(), black_box(&x), black_box(&mut y));
                    })
                },
            );
        }
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_csr_spmv,
    bench_sellp_spmv,
    bench_bcoo_spmv,
    bench_dense_masked_spmv
);
criterion_main!(benches);
