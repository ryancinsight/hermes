//! Elementwise operations and value sum/accumulate helpers.

use crate::scalar::Scalar;
use crate::arch::SimdArch;
use super::{SparseView, Csr, SellP, BlockedCoo, DenseWithMask};

impl<'a, T: Scalar, Arch: SimdArch> SparseView<'a, T, Csr, Arch> {
    /// Sum of all non-zero values in CSR storage.
    pub fn sum_values(&self) -> T {
        self.data.values.iter().copied().fold(T::ZERO, |a, b| a + b)
    }

    /// Elementwise multiply non-zero values by corresponding entries in a dense matrix.
    pub fn elementwise_mul_dense(&self, dense: &[T], out_values: &mut [T]) {
        let d = &self.data;
        for r in 0..d.nrows {
            let start = d.row_ptr[r] as usize;
            let end   = d.row_ptr[r + 1] as usize;
            for j in start..end {
                let c = d.col_indices[j] as usize;
                out_values[j] = d.values[j] * dense[r * d.ncols + c];
            }
        }
    }
}

impl<'a, T: Scalar, const C: usize, Arch: SimdArch> SparseView<'a, T, SellP<C>, Arch> {
    /// Sum of all (padded) values in SELL-p storage.
    pub fn sum_values(&self) -> T {
        self.data.values.iter().copied().fold(T::ZERO, |a, b| a + b)
    }

    /// Elementwise multiply SELL-p values by corresponding dense-matrix entries.
    pub fn elementwise_mul_dense(&self, dense: &[T], out_values: &mut [T]) {
        let d = &self.data;
        let nslices = d.nslices();
        for s in 0..nslices {
            let col_count    = d.slice_col_count[s] as usize;
            let start_offset = d.slice_ptr[s] as usize;
            for col in 0..col_count {
                for row in 0..C {
                    let idx = start_offset + col * C + row;
                    let c   = d.col_indices[idx] as usize;
                    let r   = s * C + row;
                    if r < d.nrows && c < d.ncols {
                        out_values[idx] = d.values[idx] * dense[r * d.ncols + c];
                    }
                }
            }
        }
    }
}

impl<'a, T: Scalar, const BM: usize, const BN: usize, Arch: SimdArch>
    SparseView<'a, T, BlockedCoo<BM, BN>, Arch>
{
    /// Sum of all block values in Blocked-COO storage.
    pub fn sum_values(&self) -> T {
        self.data.blocks.iter().copied().fold(T::ZERO, |a, b| a + b)
    }

    /// Elementwise multiply Blocked-COO blocks by corresponding dense-matrix entries.
    pub fn elementwise_mul_dense(&self, dense: &[T], out_values: &mut [T]) {
        let d = &self.data;
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

impl<'a, T: Scalar, Arch: SimdArch> SparseView<'a, T, DenseWithMask, Arch> {
    /// Sum of masked non-zero values.
    pub fn sum_values(&self) -> T {
        let mut s = T::ZERO;
        for i in 0..self.data.values.len() {
            if self.data.mask[i] { s += self.data.values[i]; }
        }
        s
    }

    /// Elementwise multiply masked values by corresponding dense entries.
    pub fn elementwise_mul_dense(&self, dense: &[T], out_values: &mut [T]) {
        let d = &self.data;
        for i in 0..d.values.len() {
            if d.mask[i] {
                out_values[i] = d.values[i] * dense[i];
            } else {
                out_values[i] = T::ZERO;
            }
        }
    }
}
