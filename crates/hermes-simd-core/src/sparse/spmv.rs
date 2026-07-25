//! Sparse matrix-vector multiplication (SpMV) kernels.
//!
//! # Safety
//!
//! Every kernel call below is `#[target_feature]`-gated and is therefore sound
//! only on a host implementing `Arch`. That holds by construction rather than by
//! inspection: [`SimdView::new`](crate::view::SimdView::new) returns `None` for
//! an architecture the host cannot execute, and the sparse and copy-on-write
//! constructors assert the same condition, so possessing one of these
//! arch-parameterized values *is* the proof. Per-site `SAFETY` comments record
//! only the obligations that go beyond it — pointer provenance, bounds, and
//! alignment.

use super::{BlockedCoo, Csr, DenseWithMask, SellP, SellPData, SparseView, Validated};
use crate::arch::SimdArch;
use crate::kernel::SimdKernel;
use crate::scalar::Scalar;

/// Unified trait for sparse matrix-vector multiplication.
pub trait SparseSpMv<T> {
    /// Perform matrix-vector multiplication: `y += A * x`.
    ///
    /// # Panics
    /// Panics if the dimensions of `x` or `y` are incompatible with the matrix.
    fn spmv(&self, x: &[T], y: &mut [T]);
}

/// Build an `Arch::IndexVector` from a slice of `i32` column indices.
///
/// # Safety
/// All implementations of `SimdKernel` in this workspace define `IndexVector`
/// with the layout `[i32; LANE_COUNT]`. This function reads `&[i32]` of length
/// `>= LANE_COUNT` as one `Arch::IndexVector` via an unaligned read (so element
/// alignment, not vector alignment, is the only requirement). The size half of
/// the layout invariant is enforced at compile time per backend by the
/// `const` assert below; the length contract is enforced at runtime.
#[inline(always)]
pub(crate) unsafe fn build_index_vector<T: Scalar, Arch: SimdKernel<T>>(
    cols: &[i32],
) -> Arch::IndexVector {
    // Compile-time guard binding the soundness condition the SAFETY note relies
    // on: a backend whose `IndexVector` is not `LANE_COUNT` packed `i32`s fails
    // to build rather than reading out of bounds / forming an invalid value.
    const {
        assert!(
            core::mem::size_of::<Arch::IndexVector>()
                == Arch::LANE_COUNT * core::mem::size_of::<i32>(),
            "IndexVector size must equal LANE_COUNT * size_of::<i32>()"
        )
    };
    assert!(
        cols.len() >= Arch::LANE_COUNT,
        "cols slice length {} is less than LANE_COUNT {}",
        cols.len(),
        Arch::LANE_COUNT
    );
    let ptr = cols.as_ptr() as *const Arch::IndexVector;
    core::ptr::read_unaligned(ptr)
}

#[inline(never)]
fn validate_spmv_sizes(x_len: usize, y_len: usize, ncols: usize, nrows: usize, format_name: &str) {
    assert!(
        x_len >= ncols,
        "x too short for {} ncols (got {}, expected >= {})",
        format_name,
        x_len,
        ncols
    );
    assert!(
        y_len >= nrows,
        "y too short for {} nrows (got {}, expected >= {})",
        format_name,
        y_len,
        nrows
    );
}

impl<'a, T, Arch> SparseSpMv<T> for SparseView<'a, T, Validated<Csr>, Arch>
where
    T: Scalar,
    Arch: SimdArch + SimdKernel<T>,
{
    #[inline]
    fn spmv(&self, x: &[T], y: &mut [T]) {
        let data = self.data.storage();
        validate_spmv_sizes(x.len(), y.len(), data.ncols, data.nrows, "CSR");

        let lane_count = Arch::LANE_COUNT;

        for r in 0..data.nrows {
            let start = data.row_ptr[r] as usize;
            let end = data.row_ptr[r + 1] as usize;
            let row_nnz = end - start;

            if row_nnz == 0 {
                continue;
            }

            let vals = &data.values[start..end];
            let cols = &data.col_indices[start..end];

            let simd_len = (row_nnz / lane_count) * lane_count;

            // Accumulate in vector registers first to avoid horizontal reductions in the inner loop.
            // SAFETY: `Arch::*` are the target-feature kernels covered by the
            // module invariant. Each lane-block below reads a `LANE_COUNT` window
            // of `vals`/`cols` at offset `j < simd_len <= row_nnz`, and both
            // slices have `row_nnz` elements, so every `load_unaligned` and every
            // `build_index_vector` (which requires `>= LANE_COUNT` inputs) stays
            // in bounds. `Validated<Csr>` proves every gathered `cols[k] < ncols`
            // and `validate_spmv_sizes` asserted `x.len() >= ncols`, so each
            // `Arch::gather` reads a live element of `x`.
            let acc_vec = unsafe {
                let mut acc_vec0 = Arch::zero();
                let mut acc_vec1 = Arch::zero();
                let mut acc_vec2 = Arch::zero();
                let mut acc_vec3 = Arch::zero();

                let unroll_len = (row_nnz / (lane_count * 4)) * (lane_count * 4);
                let mut j = 0usize;
                while j < unroll_len {
                    let idx0 = build_index_vector::<T, Arch>(&cols[j..j + lane_count]);
                    acc_vec0 = Arch::fmadd(
                        Arch::gather(x.as_ptr(), idx0),
                        Arch::load_unaligned(vals[j..].as_ptr()),
                        acc_vec0,
                    );

                    let idx1 =
                        build_index_vector::<T, Arch>(&cols[j + lane_count..j + lane_count * 2]);
                    acc_vec1 = Arch::fmadd(
                        Arch::gather(x.as_ptr(), idx1),
                        Arch::load_unaligned(vals[j + lane_count..].as_ptr()),
                        acc_vec1,
                    );

                    let idx2 = build_index_vector::<T, Arch>(
                        &cols[j + lane_count * 2..j + lane_count * 3],
                    );
                    acc_vec2 = Arch::fmadd(
                        Arch::gather(x.as_ptr(), idx2),
                        Arch::load_unaligned(vals[j + lane_count * 2..].as_ptr()),
                        acc_vec2,
                    );

                    let idx3 = build_index_vector::<T, Arch>(
                        &cols[j + lane_count * 3..j + lane_count * 4],
                    );
                    acc_vec3 = Arch::fmadd(
                        Arch::gather(x.as_ptr(), idx3),
                        Arch::load_unaligned(vals[j + lane_count * 3..].as_ptr()),
                        acc_vec3,
                    );

                    j += lane_count * 4;
                }

                let mut acc_vec =
                    Arch::add(Arch::add(acc_vec0, acc_vec1), Arch::add(acc_vec2, acc_vec3));

                while j < simd_len {
                    let idx = build_index_vector::<T, Arch>(&cols[j..j + lane_count]);
                    acc_vec = Arch::fmadd(
                        Arch::gather(x.as_ptr(), idx),
                        Arch::load_unaligned(vals[j..].as_ptr()),
                        acc_vec,
                    );
                    j += lane_count;
                }
                acc_vec
            };

            // SAFETY: target-feature kernel, covered by the module invariant.
            let mut acc = unsafe { Arch::sum_reduce(acc_vec) };

            let mut j = simd_len;
            while j < row_nnz {
                // SAFETY: `Validated<Csr>` proves every `col_indices[k] < ncols`,
                // and `validate_spmv_sizes` asserted `x.len() >= ncols`, so
                // `cols[j] < x.len()`. This is the same invariant the SIMD
                // `Arch::gather` above (and the SellP vectorized path) already
                // relies on; the scalar tail — the entire row when
                // `row_nnz < LANE_COUNT` — was inconsistently keeping a per-nonzero
                // bounds-check + panic branch on the gather.
                acc += vals[j] * unsafe { *x.get_unchecked(cols[j] as usize) };
                j += 1;
            }

            y[r] += acc;
        }
    }
}

impl<'a, T, Arch> SparseSpMv<T> for SparseView<'a, T, DenseWithMask, Arch>
where
    T: Scalar,
    Arch: SimdArch + SimdKernel<T>,
{
    #[inline]
    fn spmv(&self, x: &[T], y: &mut [T]) {
        let data = &self.data;
        validate_spmv_sizes(x.len(), y.len(), data.ncols, data.nrows, "DenseWithMask");

        let lane_count = Arch::LANE_COUNT;

        for r in 0..data.nrows {
            let row_offset = r * data.ncols;
            let vals = &data.values[row_offset..row_offset + data.ncols];
            let mask_bits = &data.mask[row_offset..row_offset + data.ncols];

            let simd_len = (data.ncols / lane_count) * lane_count;

            // SAFETY: `Arch::*` are the target-feature kernels covered by the
            // module invariant. Every windowed access below reads a `LANE_COUNT`
            // span of `vals`/`mask_bits`/`x` at offset `j < simd_len`; `vals` and
            // `mask_bits` hold `ncols` elements and `x.len() >= ncols` was
            // asserted, so `simd_len <= ncols` keeps each `load`, `mask_from_bools`,
            // and `masked_load_unaligned` in bounds. `zero_vec` — the masked-off
            // fill — is loop-invariant and hoisted here.
            let acc_vec = unsafe {
                let zero_vec = Arch::zero();
                let mut acc_vec0 = zero_vec;
                let mut acc_vec1 = zero_vec;
                let mut acc_vec2 = zero_vec;
                let mut acc_vec3 = zero_vec;

                let unroll_len = (data.ncols / (lane_count * 4)) * (lane_count * 4);
                let mut j = 0usize;
                while j < unroll_len {
                    let msk0 = Arch::mask_from_bools(&mask_bits[j..j + lane_count]);
                    acc_vec0 = Arch::masked_fmadd(
                        Arch::masked_load_unaligned(vals[j..].as_ptr(), msk0, zero_vec),
                        Arch::load_unaligned(x[j..].as_ptr()),
                        acc_vec0,
                        msk0,
                    );

                    let msk1 =
                        Arch::mask_from_bools(&mask_bits[j + lane_count..j + lane_count * 2]);
                    acc_vec1 = Arch::masked_fmadd(
                        Arch::masked_load_unaligned(
                            vals[j + lane_count..].as_ptr(),
                            msk1,
                            zero_vec,
                        ),
                        Arch::load_unaligned(x[j + lane_count..].as_ptr()),
                        acc_vec1,
                        msk1,
                    );

                    let msk2 =
                        Arch::mask_from_bools(&mask_bits[j + lane_count * 2..j + lane_count * 3]);
                    acc_vec2 = Arch::masked_fmadd(
                        Arch::masked_load_unaligned(
                            vals[j + lane_count * 2..].as_ptr(),
                            msk2,
                            zero_vec,
                        ),
                        Arch::load_unaligned(x[j + lane_count * 2..].as_ptr()),
                        acc_vec2,
                        msk2,
                    );

                    let msk3 =
                        Arch::mask_from_bools(&mask_bits[j + lane_count * 3..j + lane_count * 4]);
                    acc_vec3 = Arch::masked_fmadd(
                        Arch::masked_load_unaligned(
                            vals[j + lane_count * 3..].as_ptr(),
                            msk3,
                            zero_vec,
                        ),
                        Arch::load_unaligned(x[j + lane_count * 3..].as_ptr()),
                        acc_vec3,
                        msk3,
                    );

                    j += lane_count * 4;
                }

                let mut acc_vec =
                    Arch::add(Arch::add(acc_vec0, acc_vec1), Arch::add(acc_vec2, acc_vec3));

                while j < simd_len {
                    let msk = Arch::mask_from_bools(&mask_bits[j..j + lane_count]);
                    acc_vec = Arch::masked_fmadd(
                        Arch::masked_load_unaligned(vals[j..].as_ptr(), msk, zero_vec),
                        Arch::load_unaligned(x[j..].as_ptr()),
                        acc_vec,
                        msk,
                    );
                    j += lane_count;
                }
                acc_vec
            };

            // SAFETY: target-feature kernel, covered by the module invariant.
            let mut acc = unsafe { Arch::sum_reduce(acc_vec) };

            let mut j = simd_len;
            while j < data.ncols {
                if mask_bits[j] {
                    acc += vals[j] * x[j];
                }
                j += 1;
            }

            y[r] += acc;
        }
    }
}

impl<'a, T, const BM: usize, const BN: usize, Arch> SparseSpMv<T>
    for SparseView<'a, T, Validated<BlockedCoo<BM, BN>>, Arch>
where
    T: Scalar,
    Arch: SimdArch + SimdKernel<T>,
{
    #[inline]
    fn spmv(&self, x: &[T], y: &mut [T]) {
        let data = self.data.storage();
        validate_spmv_sizes(x.len(), y.len(), data.ncols, data.nrows, "BlockedCoo");

        let block_size = BM * BN;
        let lane_count = Arch::LANE_COUNT;

        if BN == lane_count {
            for b in 0..data.nblocks {
                let br = data.block_row[b] as usize;
                let bc = data.block_col[b] as usize;
                let block = &data.blocks[b * block_size..(b + 1) * block_size];

                // SAFETY: `Arch::*` are target-feature kernels (module invariant).
                // `BN == LANE_COUNT`, so each `LANE_COUNT`-wide load reads exactly
                // one block row `block[i*BN .. i*BN + BN]` (in bounds — `block` has
                // `BM*BN` elements) or the column window `x[bc .. bc + BN]`. A
                // `Validated<BlockedCoo>` guarantees `bc + BN <= ncols <= x.len()`.
                unsafe {
                    let x_vec = Arch::load_unaligned(x.as_ptr().add(bc));
                    for i in 0..BM {
                        let b_vec = Arch::load_unaligned(block.as_ptr().add(i * BN));
                        y[br + i] += Arch::sum_reduce(Arch::mul(b_vec, x_vec));
                    }
                }
            }
        } else if BN == lane_count * 2 {
            for b in 0..data.nblocks {
                let br = data.block_row[b] as usize;
                let bc = data.block_col[b] as usize;
                let block = &data.blocks[b * block_size..(b + 1) * block_size];

                // SAFETY: as above, with `BN == 2*LANE_COUNT` so each block row and
                // each `x` column window spans two `LANE_COUNT` loads; both halves
                // stay within `block` (`BM*BN` elements) and within
                // `x[bc .. bc + BN]` (`bc + BN <= ncols <= x.len()`).
                unsafe {
                    let x_vec0 = Arch::load_unaligned(x.as_ptr().add(bc));
                    let x_vec1 = Arch::load_unaligned(x.as_ptr().add(bc + lane_count));
                    for i in 0..BM {
                        let offset = i * BN;
                        let prod0 =
                            Arch::mul(Arch::load_unaligned(block.as_ptr().add(offset)), x_vec0);
                        let prod1 = Arch::mul(
                            Arch::load_unaligned(block.as_ptr().add(offset + lane_count)),
                            x_vec1,
                        );
                        y[br + i] += Arch::sum_reduce(Arch::add(prod0, prod1));
                    }
                }
            }
        } else {
            // SAFETY: `Validated<BlockedCoo>` guarantees all block coordinates
            // are in bounds; `validate_spmv_sizes` at function entry asserted
            // `x.len() >= ncols` and `y.len() >= nrows`. Raw pointers eliminate
            // redundant bounds checks from sub-slicing.
            unsafe {
                let x_ptr = x.as_ptr();
                let y_ptr = y.as_mut_ptr();
                for b in 0..data.nblocks {
                    let br = data.block_row[b] as usize;
                    let bc = data.block_col[b] as usize;
                    let block_ptr = data.blocks.as_ptr().add(b * block_size);

                    for i in 0..BM {
                        let row_ptr = block_ptr.add(i * BN);
                        let mut s = T::ZERO;
                        for k in 0..BN {
                            s = s + *row_ptr.add(k) * *x_ptr.add(bc + k);
                        }
                        *y_ptr.add(br + i) += s;
                    }
                }
            }
        }
    }
}

fn sellp_spmv_scalar<T, const C: usize>(data: &SellPData<'_, T, C>, x: &[T], y: &mut [T])
where
    T: Scalar,
{
    let nslices = data.nslices();
    for s in 0..nslices {
        let col_count = data.slice_col_count[s] as usize;
        let start_offset = data.slice_ptr[s] as usize;

        let mut row_acc = [T::ZERO; C];

        for col in 0..col_count {
            for row in 0..C {
                let idx = start_offset + col * C + row;
                let val = data.values[idx];
                let c_idx = data.col_indices[idx] as usize;
                // SAFETY: `Validated<SellP>` proves every `col_indices[k] < ncols`
                // and `validate_spmv_sizes` asserted `x.len() >= ncols`, so
                // `c_idx < x.len()`. The vectorized path gathers on exactly this
                // invariant (see its SAFETY note). The removed `if c_idx < x.len()`
                // guard was dead under that invariant — and had it ever been false
                // it would have *silently dropped* the term rather than surfacing
                // the violation, so this is also a correctness-honesty improvement.
                row_acc[row] += val * unsafe { *x.get_unchecked(c_idx) };
            }
        }

        for row in 0..C {
            let r_idx = s * C + row;
            if r_idx < y.len() {
                y[r_idx] += row_acc[row];
            }
        }
    }
}

/// Vectorized SELL-p SpMV for the case `Arch::LANE_COUNT == C`.
///
/// # Safety
/// - The host must implement `Arch` (its kernels are `#[target_feature]`-gated).
///   The caller establishes this by holding an arch-parameterized `SparseView`,
///   whose constructor asserts host support.
/// - `data` must be a `Validated<SellP<C>>` payload: every `col_indices[k]` is
///   `< ncols`, and `x.len() >= ncols`, so each gathered `x[col]` is in bounds.
///   Each slice `values[offset .. offset + C]` and `col_indices[offset ..
///   offset + C]` must lie within its buffer, which the SELL-p slice layout
///   guarantees for `offset = slice_ptr[s] + col*C`, `col < slice_col_count[s]`.
unsafe fn sellp_spmv_vectorized<T, const C: usize, Arch>(
    data: &SellPData<'_, T, C>,
    x: &[T],
    y: &mut [T],
) where
    T: Scalar,
    Arch: SimdArch + SimdKernel<T>,
{
    assert_eq!(
        Arch::LANE_COUNT,
        C,
        "sellp_spmv_vectorized requires Arch::LANE_COUNT == C"
    );
    let nslices = data.nslices();
    for s in 0..nslices {
        let col_count = data.slice_col_count[s] as usize;
        let start_offset = data.slice_ptr[s] as usize;

        let mut acc0 = Arch::zero();
        let mut acc1 = Arch::zero();
        let mut acc2 = Arch::zero();
        let mut acc3 = Arch::zero();

        let unroll = (col_count / 4) * 4;
        let mut col = 0;
        while col < unroll {
            // Unroll 0
            let offset = start_offset + col * C;
            let val_vec = Arch::load_unaligned(data.values[offset..].as_ptr());
            let idx_vec = build_index_vector::<T, Arch>(&data.col_indices[offset..offset + C]);
            let x_vec = Arch::gather(x.as_ptr(), idx_vec);
            acc0 = Arch::fmadd(val_vec, x_vec, acc0);

            // Unroll 1
            let offset = start_offset + (col + 1) * C;
            let val_vec = Arch::load_unaligned(data.values[offset..].as_ptr());
            let idx_vec = build_index_vector::<T, Arch>(&data.col_indices[offset..offset + C]);
            let x_vec = Arch::gather(x.as_ptr(), idx_vec);
            acc1 = Arch::fmadd(val_vec, x_vec, acc1);

            // Unroll 2
            let offset = start_offset + (col + 2) * C;
            let val_vec = Arch::load_unaligned(data.values[offset..].as_ptr());
            let idx_vec = build_index_vector::<T, Arch>(&data.col_indices[offset..offset + C]);
            let x_vec = Arch::gather(x.as_ptr(), idx_vec);
            acc2 = Arch::fmadd(val_vec, x_vec, acc2);

            // Unroll 3
            let offset = start_offset + (col + 3) * C;
            let val_vec = Arch::load_unaligned(data.values[offset..].as_ptr());
            let idx_vec = build_index_vector::<T, Arch>(&data.col_indices[offset..offset + C]);
            let x_vec = Arch::gather(x.as_ptr(), idx_vec);
            acc3 = Arch::fmadd(val_vec, x_vec, acc3);

            col += 4;
        }

        let mut acc = Arch::add(Arch::add(acc0, acc1), Arch::add(acc2, acc3));

        while col < col_count {
            let offset = start_offset + col * C;
            let val_vec = Arch::load_unaligned(data.values[offset..].as_ptr());
            let idx_vec = build_index_vector::<T, Arch>(&data.col_indices[offset..offset + C]);
            let x_vec = Arch::gather(x.as_ptr(), idx_vec);
            acc = Arch::fmadd(val_vec, x_vec, acc);
            col += 1;
        }

        let r_idx = s * C;
        if r_idx + C <= y.len() {
            let y_ptr = y.as_mut_ptr().add(r_idx);
            let y_vec = Arch::load_unaligned(y_ptr);
            let res_vec = Arch::add(y_vec, acc);
            Arch::store_unaligned(y_ptr, res_vec);
        } else {
            let mut temp = [T::ZERO; C];
            Arch::store_unaligned(temp.as_mut_ptr(), acc);
            for row in 0..y.len() - r_idx {
                y[r_idx + row] += temp[row];
            }
        }
    }
}

impl<'a, T, const C: usize, Arch> SparseSpMv<T> for SparseView<'a, T, Validated<SellP<C>>, Arch>
where
    T: Scalar,
    Arch: SimdArch + SimdKernel<T>,
{
    #[inline]
    fn spmv(&self, x: &[T], y: &mut [T]) {
        let data = self.data.storage();
        validate_spmv_sizes(x.len(), y.len(), data.ncols, data.nrows, "SellP");

        if Arch::LANE_COUNT == C {
            // SAFETY: `ValidatedData` proves every `col_indices[k] < ncols`
            // (so each gathered `x[col]` is in bounds given `x.len() >= ncols`)
            // and every slice load `values[offset..offset + C]` stays within
            // `values`, which are the unchecked-load preconditions.
            unsafe { sellp_spmv_vectorized::<T, C, Arch>(data, x, y) };
        } else {
            sellp_spmv_scalar::<T, C>(data, x, y);
        }
    }
}
