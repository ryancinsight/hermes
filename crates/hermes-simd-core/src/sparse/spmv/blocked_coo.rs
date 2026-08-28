//! Blocked coordinate multiplication.

use super::{validate_spmv_sizes, SparseSpMv};
use crate::arch::SimdArch;
use crate::kernel::{SimdArith, SimdLoadStore, SimdReduce};
use crate::scalar::Scalar;
use crate::sparse::{BlockedCoo, SparseView, Validated};

impl<T, const BM: usize, const BN: usize, Arch> SparseSpMv<T>
    for SparseView<'_, T, Validated<BlockedCoo<BM, BN>>, Arch>
where
    T: Scalar,
    Arch: SimdArch + SimdLoadStore<T> + SimdArith<T> + SimdReduce<T>,
{
    #[inline]
    fn spmv(&self, x: &[T], y: &mut [T]) {
        let data = self.data.storage();
        validate_spmv_sizes(x.len(), y.len(), data.ncols, data.nrows, "BlockedCoo");

        let block_size = BM * BN;
        let lane_count = Arch::LANE_COUNT;

        if BN == lane_count {
            for b in 0..data.nblocks {
                let block_row = data.block_row[b] as usize;
                let block_col = data.block_col[b] as usize;
                let block = &data.blocks[b * block_size..(b + 1) * block_size];

                // SAFETY: BN equals LANE_COUNT, and validated block coordinates
                // keep both the block row and x window in bounds.
                unsafe {
                    let x_vec = Arch::load_unaligned(x.as_ptr().add(block_col));
                    for row in 0..BM {
                        let block_vec = Arch::load_unaligned(block.as_ptr().add(row * BN));
                        y[block_row + row] += Arch::sum_reduce(Arch::mul(block_vec, x_vec));
                    }
                }
            }
        } else if BN == lane_count * 2 {
            for b in 0..data.nblocks {
                let block_row = data.block_row[b] as usize;
                let block_col = data.block_col[b] as usize;
                let block = &data.blocks[b * block_size..(b + 1) * block_size];

                // SAFETY: BN equals two LANE_COUNT windows; validation keeps
                // both halves inside the block and x.
                unsafe {
                    let x_vec0 = Arch::load_unaligned(x.as_ptr().add(block_col));
                    let x_vec1 = Arch::load_unaligned(x.as_ptr().add(block_col + lane_count));
                    for row in 0..BM {
                        let offset = row * BN;
                        let product0 =
                            Arch::mul(Arch::load_unaligned(block.as_ptr().add(offset)), x_vec0);
                        let product1 = Arch::mul(
                            Arch::load_unaligned(block.as_ptr().add(offset + lane_count)),
                            x_vec1,
                        );
                        y[block_row + row] += Arch::sum_reduce(Arch::add(product0, product1));
                    }
                }
            }
        } else {
            // SAFETY: validated block coordinates plus the entry size checks
            // keep every raw-pointer access inside its source or destination.
            unsafe {
                let x_ptr = x.as_ptr();
                let y_ptr = y.as_mut_ptr();
                for b in 0..data.nblocks {
                    let block_row = data.block_row[b] as usize;
                    let block_col = data.block_col[b] as usize;
                    let block_ptr = data.blocks.as_ptr().add(b * block_size);

                    for row in 0..BM {
                        let row_ptr = block_ptr.add(row * BN);
                        let mut sum = T::ZERO;
                        for col in 0..BN {
                            sum += *row_ptr.add(col) * *x_ptr.add(block_col + col);
                        }
                        *y_ptr.add(block_row + row) += sum;
                    }
                }
            }
        }
    }
}
