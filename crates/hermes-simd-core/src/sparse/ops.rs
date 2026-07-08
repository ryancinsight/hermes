//! Elementwise operations and value sum/accumulate helpers.

use super::{BlockedCoo, Csr, DenseWithMask, SellP, SparseView};
use crate::arch::SimdArch;
use crate::kernel::SimdKernel;
use crate::scalar::Scalar;
use crate::sparse::spmv::build_index_vector;

/// Unified trait for elementwise and reduction operations on sparse matrices.
pub trait SparseOps<T> {
    /// Compute the sum of all elements stored in the sparse matrix.
    fn sum_values(&self) -> T;

    /// Elementwise multiply the sparse matrix values by corresponding entries
    /// in a dense matrix, writing the results to `out_values`.
    fn elementwise_mul_dense(&self, dense: &[T], out_values: &mut [T]);
}

impl<'a, T, Arch> SparseOps<T> for SparseView<'a, T, Csr, Arch>
where
    T: Scalar,
    Arch: SimdArch + SimdKernel<T>,
{
    #[inline]
    fn sum_values(&self) -> T {
        if let Some(view) =
            crate::view::SimdView::<T, Arch, crate::align::Unaligned>::new(self.data.values)
        {
            view.reduce(crate::ops::Sum)
        } else {
            T::ZERO
        }
    }

    #[inline]
    fn elementwise_mul_dense(&self, dense: &[T], out_values: &mut [T]) {
        let d = &self.data;
        let lane_count = Arch::LANE_COUNT;
        for r in 0..d.nrows {
            let start = d.row_ptr[r] as usize;
            let end = d.row_ptr[r + 1] as usize;
            let row_nnz = end - start;
            let vals = &d.values[start..end];
            let cols = &d.col_indices[start..end];
            let out = &mut out_values[start..end];

            let simd_len = (row_nnz / lane_count) * lane_count;
            let mut j = 0usize;
            while j < simd_len {
                let idx = unsafe { build_index_vector::<T, Arch>(&cols[j..j + lane_count]) };
                let dense_vec = unsafe { Arch::gather(dense.as_ptr(), idx) };
                let v_vec = unsafe { Arch::load_unaligned(vals[j..].as_ptr()) };
                let res_vec = unsafe { Arch::mul(v_vec, dense_vec) };
                unsafe { Arch::store_unaligned(out[j..].as_mut_ptr(), res_vec) };
                j += lane_count;
            }
            while j < row_nnz {
                let c = cols[j] as usize;
                out[j] = vals[j] * dense[c];
                j += 1;
            }
        }
    }
}

impl<'a, T, const C: usize, Arch> SparseOps<T> for SparseView<'a, T, SellP<C>, Arch>
where
    T: Scalar,
    Arch: SimdArch + SimdKernel<T>,
{
    #[inline]
    fn sum_values(&self) -> T {
        if let Some(view) =
            crate::view::SimdView::<T, Arch, crate::align::Unaligned>::new(self.data.values)
        {
            view.reduce(crate::ops::Sum)
        } else {
            T::ZERO
        }
    }

    #[inline]
    fn elementwise_mul_dense(&self, dense: &[T], out_values: &mut [T]) {
        let d = &self.data;
        let nslices = d.nslices();
        let lane_count = Arch::LANE_COUNT;

        if lane_count == C {
            // SOUNDNESS: the vectorized path loads `values[offset..]` and stores
            // `out_values[offset..]` as full `C`-lane vectors. Validate SELL-p
            // slice geometry via the SSOT checker (bounds `offset + C <=
            // values.len()`) and require the output to be at least as long as the
            // values array, so both unchecked accesses stay in bounds even for a
            // caller-constructed matrix with `pub` fields.
            use super::types::SparseValidate;
            d.validate()
                .expect("SELL-p matrix failed structural validation before vectorized kernel");
            assert!(
                out_values.len() >= d.values.len(),
                "SELL-p elementwise_mul_dense: out_values len {} < values len {}",
                out_values.len(),
                d.values.len()
            );
            for s in 0..nslices {
                let col_count = d.slice_col_count[s] as usize;
                let start_offset = d.slice_ptr[s] as usize;
                let slice_base_r = s * C;

                for col in 0..col_count {
                    let offset = start_offset + col * C;

                    let mut idx_arr = [0i32; 64];
                    let mut mask_arr = [false; 64];
                    for row in 0..C {
                        let r = slice_base_r + row;
                        let c = d.col_indices[offset + row] as usize;
                        let in_bounds = r < d.nrows && c < d.ncols;
                        mask_arr[row] = in_bounds;
                        if in_bounds {
                            idx_arr[row] = (r * d.ncols + c) as i32;
                        }
                    }

                    let idx = unsafe { build_index_vector::<T, Arch>(&idx_arr[..C]) };
                    let mask = unsafe { Arch::mask_from_bools(&mask_arr[..C]) };
                    let zero_vec = unsafe { Arch::zero() };

                    let dense_vec =
                        unsafe { Arch::gather_masked(dense.as_ptr(), idx, mask, zero_vec) };
                    let val_vec = unsafe { Arch::load_unaligned(d.values[offset..].as_ptr()) };
                    let res_vec = unsafe { Arch::mul(val_vec, dense_vec) };

                    unsafe {
                        Arch::masked_store_unaligned(
                            out_values[offset..].as_mut_ptr(),
                            mask,
                            res_vec,
                        )
                    };
                }
            }
        } else {
            for s in 0..nslices {
                let col_count = d.slice_col_count[s] as usize;
                let start_offset = d.slice_ptr[s] as usize;
                for col in 0..col_count {
                    for row in 0..C {
                        let idx = start_offset + col * C + row;
                        let c = d.col_indices[idx] as usize;
                        let r = s * C + row;
                        if r < d.nrows && c < d.ncols {
                            out_values[idx] = d.values[idx] * dense[r * d.ncols + c];
                        }
                    }
                }
            }
        }
    }
}

impl<'a, T, const BM: usize, const BN: usize, Arch> SparseOps<T>
    for SparseView<'a, T, BlockedCoo<BM, BN>, Arch>
where
    T: Scalar,
    Arch: SimdArch + SimdKernel<T>,
{
    #[inline]
    fn sum_values(&self) -> T {
        if let Some(view) =
            crate::view::SimdView::<T, Arch, crate::align::Unaligned>::new(self.data.blocks)
        {
            view.reduce(crate::ops::Sum)
        } else {
            T::ZERO
        }
    }

    #[inline]
    fn elementwise_mul_dense(&self, dense: &[T], out_values: &mut [T]) {
        let d = &self.data;
        let lane_count = Arch::LANE_COUNT;

        // Bounds for the unchecked SIMD loads/stores below: the dense matrix and
        // the block/output buffers must be large enough, and every block must lie
        // within the `nrows x ncols` dense extent so each `dense[(br+i)*ncols+bc
        // .. +BN]` read stays in bounds. O(nblocks), once per call.
        let block_elems = d.nblocks * BM * BN;
        assert!(
            dense.len() >= d.nrows * d.ncols,
            "dense buffer {} too small for {}x{}",
            dense.len(),
            d.nrows,
            d.ncols
        );
        assert!(
            out_values.len() >= block_elems && d.blocks.len() >= block_elems,
            "block/output buffers too small for {} block elements",
            block_elems
        );
        for b in 0..d.nblocks {
            let br = d.block_row[b] as usize;
            let bc = d.block_col[b] as usize;
            assert!(
                bc + BN <= d.ncols && br + BM <= d.nrows,
                "BlockedCoo block {b} (row {br}+{BM}, col {bc}+{BN}) exceeds {}x{}",
                d.nrows,
                d.ncols
            );
        }

        if BN == lane_count {
            for b in 0..d.nblocks {
                let br = d.block_row[b] as usize;
                let bc = d.block_col[b] as usize;
                for i in 0..BM {
                    let offset = b * (BM * BN) + i * BN;
                    let b_vec = unsafe { Arch::load_unaligned(d.blocks[offset..].as_ptr()) };
                    let dense_idx = (br + i) * d.ncols + bc;
                    let dense_vec = unsafe { Arch::load_unaligned(dense[dense_idx..].as_ptr()) };
                    let res_vec = unsafe { Arch::mul(b_vec, dense_vec) };
                    unsafe { Arch::store_unaligned(out_values[offset..].as_mut_ptr(), res_vec) };
                }
            }
        } else if BN == lane_count * 2 {
            for b in 0..d.nblocks {
                let br = d.block_row[b] as usize;
                let bc = d.block_col[b] as usize;
                for i in 0..BM {
                    let offset = b * (BM * BN) + i * BN;
                    let b_vec0 = unsafe { Arch::load_unaligned(d.blocks[offset..].as_ptr()) };
                    let b_vec1 =
                        unsafe { Arch::load_unaligned(d.blocks[offset + lane_count..].as_ptr()) };
                    let dense_idx = (br + i) * d.ncols + bc;
                    let dense_vec0 = unsafe { Arch::load_unaligned(dense[dense_idx..].as_ptr()) };
                    let dense_vec1 =
                        unsafe { Arch::load_unaligned(dense[dense_idx + lane_count..].as_ptr()) };
                    let res_vec0 = unsafe { Arch::mul(b_vec0, dense_vec0) };
                    let res_vec1 = unsafe { Arch::mul(b_vec1, dense_vec1) };
                    unsafe {
                        Arch::store_unaligned(out_values[offset..].as_mut_ptr(), res_vec0);
                        Arch::store_unaligned(
                            out_values[offset + lane_count..].as_mut_ptr(),
                            res_vec1,
                        );
                    }
                }
            }
        } else {
            for b in 0..d.nblocks {
                let br = d.block_row[b] as usize;
                let bc = d.block_col[b] as usize;
                for i in 0..BM {
                    for j in 0..BN {
                        let idx = b * (BM * BN) + i * BN + j;
                        out_values[idx] = d.blocks[idx] * dense[(br + i) * d.ncols + (bc + j)];
                    }
                }
            }
        }
    }
}

impl<'a, T, Arch> SparseOps<T> for SparseView<'a, T, DenseWithMask, Arch>
where
    T: Scalar,
    Arch: SimdArch + SimdKernel<T>,
{
    #[inline]
    fn sum_values(&self) -> T {
        let lane_count = Arch::LANE_COUNT;
        let len = self.data.values.len();
        let simd_len = (len / lane_count) * lane_count;
        let mut acc_vec = unsafe { Arch::zero() };
        let mut i = 0usize;
        while i < simd_len {
            let msk = unsafe { Arch::mask_from_bools(&self.data.mask[i..i + lane_count]) };
            let zero_vec = unsafe { Arch::zero() };
            let v_vec = unsafe {
                Arch::masked_load_unaligned(self.data.values[i..].as_ptr(), msk, zero_vec)
            };
            acc_vec = unsafe { Arch::add(acc_vec, v_vec) };
            i += lane_count;
        }
        let mut s = unsafe { Arch::sum_reduce(acc_vec) };
        while i < len {
            if self.data.mask[i] {
                s += self.data.values[i];
            }
            i += 1;
        }
        s
    }

    #[inline]
    fn elementwise_mul_dense(&self, dense: &[T], out_values: &mut [T]) {
        let d = &self.data;
        let len = d.values.len();
        let lane_count = Arch::LANE_COUNT;
        let simd_len = (len / lane_count) * lane_count;
        let mut i = 0usize;
        while i < simd_len {
            let msk = unsafe { Arch::mask_from_bools(&d.mask[i..i + lane_count]) };
            let v_vec = unsafe { Arch::load_unaligned(d.values[i..].as_ptr()) };
            let dense_vec = unsafe { Arch::load_unaligned(dense[i..].as_ptr()) };
            let zero_vec = unsafe { Arch::zero() };
            let res_vec = unsafe { Arch::masked_mul(v_vec, dense_vec, msk, zero_vec) };
            unsafe { Arch::store_unaligned(out_values[i..].as_mut_ptr(), res_vec) };
            i += lane_count;
        }
        while i < len {
            if d.mask[i] {
                out_values[i] = d.values[i] * dense[i];
            } else {
                out_values[i] = T::ZERO;
            }
            i += 1;
        }
    }
}
