//! Sparse matrix-vector multiplication (SpMV) kernels.

use crate::scalar::Scalar;
use crate::arch::SimdArch;
use crate::kernel::SimdKernel;
use super::{
    SparseView, Csr, SellP, BlockedCoo, DenseWithMask,
    SellPData,
};

/// Build an `Arch::IndexVector` from a slice of `i32` column indices.
///
/// # Safety
/// All implementations of `SimdKernel` in this workspace define `IndexVector`
/// as `[i32; LANE_COUNT]`. This function transmutes `&[i32]` of length `LANE_COUNT`
/// to `Arch::IndexVector` — this is sound iff the workspace invariant holds.
/// `debug_assert!` validates the length contract.
#[inline(always)]
pub(crate) unsafe fn build_index_vector<T: Scalar, Arch: SimdKernel<T>>(
    cols: &[i32],
) -> Arch::IndexVector {
    debug_assert_eq!(cols.len(), Arch::LANE_COUNT);
    let ptr = cols.as_ptr() as *const Arch::IndexVector;
    core::ptr::read_unaligned(ptr)
}

#[inline(never)]
fn validate_spmv_sizes(x_len: usize, y_len: usize, ncols: usize, nrows: usize, format_name: &str) {
    assert!(x_len >= ncols, "x too short for {} ncols (got {}, expected >= {})", format_name, x_len, ncols);
    assert!(y_len >= nrows, "y too short for {} nrows (got {}, expected >= {})", format_name, y_len, nrows);
}

impl<'a, T, Arch> SparseView<'a, T, Csr, Arch>
where
    T: Scalar,
    Arch: SimdArch + SimdKernel<T>,
{
    /// Sparse matrix-vector multiply: `y += A * x`.
    ///
    /// # Panics
    /// Panics if `x.len() < ncols` or `y.len() < nrows`.
    pub fn spmv(&self, x: &[T], y: &mut [T]) {
        let data = &self.data;
        validate_spmv_sizes(x.len(), y.len(), data.ncols, data.nrows, "CSR");

        let lane_count = Arch::LANE_COUNT;

        for r in 0..data.nrows {
            let start   = data.row_ptr[r] as usize;
            let end     = data.row_ptr[r + 1] as usize;
            let row_nnz = end - start;

            if row_nnz == 0 { continue; }

            let vals = &data.values[start..end];
            let cols = &data.col_indices[start..end];

            let simd_len = (row_nnz / lane_count) * lane_count;
            
            // Accumulate in vector registers first to avoid horizontal reductions in the inner loop.
            let mut acc_vec0 = unsafe { Arch::zero() };
            let mut acc_vec1 = unsafe { Arch::zero() };
            let mut acc_vec2 = unsafe { Arch::zero() };
            let mut acc_vec3 = unsafe { Arch::zero() };

            let unroll_len = (row_nnz / (lane_count * 4)) * (lane_count * 4);
            let mut j = 0usize;
            while j < unroll_len {
                // 0
                let idx0 = unsafe { build_index_vector::<T, Arch>(&cols[j..j + lane_count]) };
                let x_vec0 = unsafe { Arch::gather(x.as_ptr(), idx0) };
                let v_vec0 = unsafe { Arch::load_unaligned(vals[j..].as_ptr()) };
                acc_vec0 = unsafe { Arch::fmadd(x_vec0, v_vec0, acc_vec0) };

                // 1
                let idx1 = unsafe { build_index_vector::<T, Arch>(&cols[j + lane_count..j + lane_count * 2]) };
                let x_vec1 = unsafe { Arch::gather(x.as_ptr(), idx1) };
                let v_vec1 = unsafe { Arch::load_unaligned(vals[j + lane_count..].as_ptr()) };
                acc_vec1 = unsafe { Arch::fmadd(x_vec1, v_vec1, acc_vec1) };

                // 2
                let idx2 = unsafe { build_index_vector::<T, Arch>(&cols[j + lane_count * 2..j + lane_count * 3]) };
                let x_vec2 = unsafe { Arch::gather(x.as_ptr(), idx2) };
                let v_vec2 = unsafe { Arch::load_unaligned(vals[j + lane_count * 2..].as_ptr()) };
                acc_vec2 = unsafe { Arch::fmadd(x_vec2, v_vec2, acc_vec2) };

                // 3
                let idx3 = unsafe { build_index_vector::<T, Arch>(&cols[j + lane_count * 3..j + lane_count * 4]) };
                let x_vec3 = unsafe { Arch::gather(x.as_ptr(), idx3) };
                let v_vec3 = unsafe { Arch::load_unaligned(vals[j + lane_count * 3..].as_ptr()) };
                acc_vec3 = unsafe { Arch::fmadd(x_vec3, v_vec3, acc_vec3) };

                j += lane_count * 4;
            }

            let mut acc_vec = unsafe { Arch::add(Arch::add(acc_vec0, acc_vec1), Arch::add(acc_vec2, acc_vec3)) };

            while j < simd_len {
                let idx   = unsafe { build_index_vector::<T, Arch>(&cols[j..j + lane_count]) };
                let x_vec = unsafe { Arch::gather(x.as_ptr(), idx) };
                let v_vec = unsafe { Arch::load_unaligned(vals[j..].as_ptr()) };
                acc_vec = unsafe { Arch::fmadd(x_vec, v_vec, acc_vec) };
                j += lane_count;
            }

            let mut acc = unsafe { Arch::sum_reduce(acc_vec) };

            while j < row_nnz {
                acc += vals[j] * x[cols[j] as usize];
                j += 1;
            }

            y[r] += acc;
        }
    }
}

impl<'a, T, Arch> SparseView<'a, T, DenseWithMask, Arch>
where
    T: Scalar,
    Arch: SimdArch + SimdKernel<T>,
{
    /// SpMV using masked SIMD on dense-with-mask format: `y += A * x`.
    ///
    /// # Panics
    /// Panics if `x.len() < ncols` or `y.len() < nrows`.
    pub fn spmv(&self, x: &[T], y: &mut [T]) {
        let data = &self.data;
        validate_spmv_sizes(x.len(), y.len(), data.ncols, data.nrows, "DenseWithMask");

        let lane_count = Arch::LANE_COUNT;

        for r in 0..data.nrows {
            let row_offset = r * data.ncols;
            let vals       = &data.values[row_offset..row_offset + data.ncols];
            let mask_bits  = &data.mask[row_offset..row_offset + data.ncols];

            let simd_len = (data.ncols / lane_count) * lane_count;

            let mut acc_vec0 = unsafe { Arch::zero() };
            let mut acc_vec1 = unsafe { Arch::zero() };
            let mut acc_vec2 = unsafe { Arch::zero() };
            let mut acc_vec3 = unsafe { Arch::zero() };

            let unroll_len = (data.ncols / (lane_count * 4)) * (lane_count * 4);
            let mut j = 0usize;
            while j < unroll_len {
                let zero_vec = unsafe { Arch::zero() };

                // 0
                let msk0 = unsafe { Arch::mask_from_bools(&mask_bits[j..j + lane_count]) };
                let v_vec0 = unsafe { Arch::masked_load_unaligned(vals[j..].as_ptr(), msk0, zero_vec) };
                let x_vec0 = unsafe { Arch::load_unaligned(x[j..].as_ptr()) };
                acc_vec0 = unsafe { Arch::masked_fmadd(v_vec0, x_vec0, acc_vec0, msk0) };

                // 1
                let msk1 = unsafe { Arch::mask_from_bools(&mask_bits[j + lane_count..j + lane_count * 2]) };
                let v_vec1 = unsafe { Arch::masked_load_unaligned(vals[j + lane_count..].as_ptr(), msk1, zero_vec) };
                let x_vec1 = unsafe { Arch::load_unaligned(x[j + lane_count..].as_ptr()) };
                acc_vec1 = unsafe { Arch::masked_fmadd(v_vec1, x_vec1, acc_vec1, msk1) };

                // 2
                let msk2 = unsafe { Arch::mask_from_bools(&mask_bits[j + lane_count * 2..j + lane_count * 3]) };
                let v_vec2 = unsafe { Arch::masked_load_unaligned(vals[j + lane_count * 2..].as_ptr(), msk2, zero_vec) };
                let x_vec2 = unsafe { Arch::load_unaligned(x[j + lane_count * 2..].as_ptr()) };
                acc_vec2 = unsafe { Arch::masked_fmadd(v_vec2, x_vec2, acc_vec2, msk2) };

                // 3
                let msk3 = unsafe { Arch::mask_from_bools(&mask_bits[j + lane_count * 3..j + lane_count * 4]) };
                let v_vec3 = unsafe { Arch::masked_load_unaligned(vals[j + lane_count * 3..].as_ptr(), msk3, zero_vec) };
                let x_vec3 = unsafe { Arch::load_unaligned(x[j + lane_count * 3..].as_ptr()) };
                acc_vec3 = unsafe { Arch::masked_fmadd(v_vec3, x_vec3, acc_vec3, msk3) };

                j += lane_count * 4;
            }

            let mut acc_vec = unsafe { Arch::add(Arch::add(acc_vec0, acc_vec1), Arch::add(acc_vec2, acc_vec3)) };

            while j < simd_len {
                let msk      = unsafe { Arch::mask_from_bools(&mask_bits[j..j + lane_count]) };
                let zero_vec = unsafe { Arch::zero() };
                let v_vec    = unsafe { Arch::masked_load_unaligned(vals[j..].as_ptr(), msk, zero_vec) };
                let x_vec    = unsafe { Arch::load_unaligned(x[j..].as_ptr()) };
                acc_vec = unsafe { Arch::masked_fmadd(v_vec, x_vec, acc_vec, msk) };
                j += lane_count;
            }

            let mut acc = unsafe { Arch::sum_reduce(acc_vec) };

            while j < data.ncols {
                if mask_bits[j] { acc += vals[j] * x[j]; }
                j += 1;
            }

            y[r] += acc;
        }
    }
}

impl<'a, T, const BM: usize, const BN: usize, Arch> SparseView<'a, T, BlockedCoo<BM, BN>, Arch>
where
    T: Scalar,
    Arch: SimdArch + SimdKernel<T>,
{
    /// SpMV on Blocked-COO: `y += A * x`.
    ///
    /// # Panics
    /// Panics if `x.len() < ncols` or `y.len() < nrows`.
    pub fn spmv(&self, x: &[T], y: &mut [T]) {
        let data = &self.data;
        validate_spmv_sizes(x.len(), y.len(), data.ncols, data.nrows, "BlockedCoo");

        let block_size = BM * BN;
        let lane_count = Arch::LANE_COUNT;

        if BN == lane_count {
            for b in 0..data.nblocks {
                let br    = data.block_row[b] as usize;
                let bc    = data.block_col[b] as usize;
                let block = &data.blocks[b * block_size..(b + 1) * block_size];

                let x_vec = unsafe { Arch::load_unaligned(x.as_ptr().add(bc)) };

                for i in 0..BM {
                    let b_vec = unsafe { Arch::load_unaligned(block.as_ptr().add(i * BN)) };
                    let prod = unsafe { Arch::mul(b_vec, x_vec) };
                    let s = unsafe { Arch::sum_reduce(prod) };
                    y[br + i] += s;
                }
            }
        } else if BN == lane_count * 2 {
            for b in 0..data.nblocks {
                let br    = data.block_row[b] as usize;
                let bc    = data.block_col[b] as usize;
                let block = &data.blocks[b * block_size..(b + 1) * block_size];

                let x_vec0 = unsafe { Arch::load_unaligned(x.as_ptr().add(bc)) };
                let x_vec1 = unsafe { Arch::load_unaligned(x.as_ptr().add(bc + lane_count)) };

                for i in 0..BM {
                    let offset = i * BN;
                    let b_vec0 = unsafe { Arch::load_unaligned(block.as_ptr().add(offset)) };
                    let b_vec1 = unsafe { Arch::load_unaligned(block.as_ptr().add(offset + lane_count)) };
                    let prod0 = unsafe { Arch::mul(b_vec0, x_vec0) };
                    let prod1 = unsafe { Arch::mul(b_vec1, x_vec1) };
                    let sum_vec = unsafe { Arch::add(prod0, prod1) };
                    let s = unsafe { Arch::sum_reduce(sum_vec) };
                    y[br + i] += s;
                }
            }
        } else {
            for b in 0..data.nblocks {
                let br    = data.block_row[b] as usize;
                let bc    = data.block_col[b] as usize;
                let block = &data.blocks[b * block_size..(b + 1) * block_size];

                for i in 0..BM {
                    let block_row_data = &block[i * BN..(i + 1) * BN];
                    let x_slice        = &x[bc..bc + BN];
                    let mut s = T::ZERO;
                    for k in 0..BN {
                        s += block_row_data[k] * x_slice[k];
                    }
                    y[br + i] += s;
                }
            }
        }
    }
}

fn sellp_spmv_scalar<T, const C: usize>(
    data: &SellPData<'_, T, C>,
    x: &[T],
    y: &mut [T],
) where
    T: Scalar,
{
    let nslices = data.nslices();
    for s in 0..nslices {
        let col_count    = data.slice_col_count[s] as usize;
        let start_offset = data.slice_ptr[s] as usize;

        let mut row_acc = [T::ZERO; C];

        for col in 0..col_count {
            for row in 0..C {
                let idx   = start_offset + col * C + row;
                let val   = data.values[idx];
                let c_idx = data.col_indices[idx] as usize;
                if c_idx < x.len() {
                    row_acc[row] += val * x[c_idx];
                }
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

unsafe fn sellp_spmv_vectorized<T, const C: usize, Arch>(
    data: &SellPData<'_, T, C>,
    x: &[T],
    y: &mut [T],
) where
    T: Scalar,
    Arch: SimdArch + SimdKernel<T>,
{
    let nslices = data.nslices();
    for s in 0..nslices {
        let col_count    = data.slice_col_count[s] as usize;
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
            let x_vec   = Arch::gather(x.as_ptr(), idx_vec);
            acc0 = Arch::fmadd(val_vec, x_vec, acc0);

            // Unroll 1
            let offset = start_offset + (col + 1) * C;
            let val_vec = Arch::load_unaligned(data.values[offset..].as_ptr());
            let idx_vec = build_index_vector::<T, Arch>(&data.col_indices[offset..offset + C]);
            let x_vec   = Arch::gather(x.as_ptr(), idx_vec);
            acc1 = Arch::fmadd(val_vec, x_vec, acc1);

            // Unroll 2
            let offset = start_offset + (col + 2) * C;
            let val_vec = Arch::load_unaligned(data.values[offset..].as_ptr());
            let idx_vec = build_index_vector::<T, Arch>(&data.col_indices[offset..offset + C]);
            let x_vec   = Arch::gather(x.as_ptr(), idx_vec);
            acc2 = Arch::fmadd(val_vec, x_vec, acc2);

            // Unroll 3
            let offset = start_offset + (col + 3) * C;
            let val_vec = Arch::load_unaligned(data.values[offset..].as_ptr());
            let idx_vec = build_index_vector::<T, Arch>(&data.col_indices[offset..offset + C]);
            let x_vec   = Arch::gather(x.as_ptr(), idx_vec);
            acc3 = Arch::fmadd(val_vec, x_vec, acc3);

            col += 4;
        }

        let mut acc = Arch::add(Arch::add(acc0, acc1), Arch::add(acc2, acc3));

        while col < col_count {
            let offset = start_offset + col * C;
            let val_vec = Arch::load_unaligned(data.values[offset..].as_ptr());
            let idx_vec = build_index_vector::<T, Arch>(&data.col_indices[offset..offset + C]);
            let x_vec   = Arch::gather(x.as_ptr(), idx_vec);
            acc = Arch::fmadd(val_vec, x_vec, acc);
            col += 1;
        }

        let mut temp = [T::ZERO; C];
        Arch::store_unaligned(temp.as_mut_ptr(), acc);
        for row in 0..C {
            let r_idx = s * C + row;
            if r_idx < y.len() {
                y[r_idx] += temp[row];
            }
        }
    }
}

impl<'a, T, const C: usize, Arch> SparseView<'a, T, SellP<C>, Arch>
where
    T: Scalar,
    Arch: SimdArch + SimdKernel<T>,
{
    /// Sliced ELLPACK SpMV: `y += A * x`.
    pub fn spmv(&self, x: &[T], y: &mut [T]) {
        validate_spmv_sizes(x.len(), y.len(), self.data.ncols, self.data.nrows, "SellP");

        if Arch::LANE_COUNT == C {
            unsafe { sellp_spmv_vectorized::<T, C, Arch>(&self.data, x, y) };
        } else {
            sellp_spmv_scalar::<T, C>(&self.data, x, y);
        }
    }
}
