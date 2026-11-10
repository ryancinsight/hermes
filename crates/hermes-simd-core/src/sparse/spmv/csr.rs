//! Compressed Sparse Row multiplication.

use super::{build_index_vector, validate_spmv_sizes, SparseScaledSpMv, SparseSpMv};
use crate::arch::SimdArch;
use crate::kernel::{SimdArith, SimdGather, SimdLoadStore, SimdReduce};
use crate::scalar::Scalar;
use crate::sparse::{Csr, SparseView, Validated};

impl<T, Arch> SparseSpMv<T> for SparseView<'_, T, Validated<Csr>, Arch>
where
    T: Scalar,
    Arch: SimdArch + SimdLoadStore<T> + SimdArith<T> + SimdGather<T> + SimdReduce<T>,
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

            // Accumulate in vector registers first to avoid horizontal
            // reductions in the inner loop.
            // SAFETY: each lane block is inside `vals` and `cols`.
            // `Validated<Csr>` proves every gathered column is below `ncols`,
            // and the entry assertion proves `x.len() >= ncols`.
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

            // SAFETY: the parent module's target-feature invariant applies.
            let mut acc = unsafe { Arch::sum_reduce(acc_vec) };
            let mut j = simd_len;
            while j < row_nnz {
                // SAFETY: validated columns and the entry size assertion prove
                // this index is in bounds.
                acc += vals[j] * unsafe { *x.get_unchecked(cols[j] as usize) };
                j += 1;
            }
            y[r] += acc;
        }
    }
}

impl<T, Arch> SparseScaledSpMv<T> for SparseView<'_, T, Validated<Csr>, Arch>
where
    T: Scalar,
    Arch: SimdArch + SimdLoadStore<T> + SimdArith<T> + SimdGather<T> + SimdReduce<T>,
{
    #[inline]
    fn spmv_scaled(&self, alpha: T, x: &[T], y: &mut [T]) {
        let data = self.data.storage();
        validate_spmv_sizes(x.len(), y.len(), data.ncols, data.nrows, "CSR");

        for r in 0..data.nrows {
            let start = data.row_ptr[r] as usize;
            let end = data.row_ptr[r + 1] as usize;
            spmv_scaled_csr_row::<T, Arch>(
                &data.values[start..end],
                &data.col_indices[start..end],
                alpha,
                x,
                // SAFETY: `validate_spmv_sizes` proves `y.len() >= nrows`, so
                // every row writes its own in-bounds `y[r]` exactly once.
                unsafe { y.get_unchecked_mut(r) },
            );
        }
    }
}

/// Fused SpMV-plus-`axpy`: `y += alpha * A · x` in one row pass.
///
/// A second non-zero-sized-type policy over the same validated CSR view: the
/// caller-visible effect is one traversal that scales the row dot product by
/// `alpha` before the single `y[r] += …` write, instead of two passes (one
/// `SpMV`, one `axpy`) each of which reloads the row. No second buffer, no
/// second row-pointer walk; the kernel body is the 4×-unrolled gather/FMA
/// loop above with the scalar multiply folded in before the accumulation.
///
/// Row independence is unchanged: every row writes exactly its own `y[r]`.
#[inline]
pub(crate) fn spmv_scaled_csr_row<T, Arch>(
    values: &[T],
    columns: &[i32],
    alpha: T,
    x: &[T],
    y_row: &mut T,
) where
    T: Scalar,
    Arch: SimdArch + SimdLoadStore<T> + SimdArith<T> + SimdGather<T> + SimdReduce<T>,
{
    let row_nnz = values.len();
    if row_nnz == 0 {
        return;
    }
    spmv_scaled_row_tail::<T, Arch>(values, columns, alpha, x, y_row, row_nnz, Arch::LANE_COUNT);
}

/// Fused-row tail: the row body after the 4×-unrolled prefix.
///
/// Split as its own function so the unrolled loop above (and any future
/// unroll factor) stays the hot path while the scalar drain shares one
/// implementation with the un-unrolled entry point.
fn spmv_scaled_row_tail<T, Arch>(
    values: &[T],
    columns: &[i32],
    alpha: T,
    x: &[T],
    y_row: &mut T,
    row_nnz: usize,
    lane_count: usize,
) where
    T: Scalar,
    Arch: SimdArch + SimdLoadStore<T> + SimdArith<T> + SimdGather<T> + SimdReduce<T>,
{
    // SAFETY: same validated-column + entry-size proof as `spmv` above.
    let acc_vec = unsafe {
        let mut acc0 = Arch::zero();
        let mut acc1 = Arch::zero();
        let mut acc2 = Arch::zero();
        let mut acc3 = Arch::zero();
        let unroll = (row_nnz / (lane_count * 4)) * (lane_count * 4);
        let mut j = 0usize;
        while j < unroll {
            let i0 = super::build_index_vector::<T, Arch>(&columns[j..j + lane_count]);
            acc0 = Arch::fmadd(
                Arch::gather(x.as_ptr(), i0),
                Arch::load_unaligned(values[j..].as_ptr()),
                acc0,
            );
            let i1 =
                super::build_index_vector::<T, Arch>(&columns[j + lane_count..j + lane_count * 2]);
            acc1 = Arch::fmadd(
                Arch::gather(x.as_ptr(), i1),
                Arch::load_unaligned(values[j + lane_count..].as_ptr()),
                acc1,
            );
            let i2 = super::build_index_vector::<T, Arch>(
                &columns[j + lane_count * 2..j + lane_count * 3],
            );
            acc2 = Arch::fmadd(
                Arch::gather(x.as_ptr(), i2),
                Arch::load_unaligned(values[j + lane_count * 2..].as_ptr()),
                acc2,
            );
            let i3 = super::build_index_vector::<T, Arch>(
                &columns[j + lane_count * 3..j + lane_count * 4],
            );
            acc3 = Arch::fmadd(
                Arch::gather(x.as_ptr(), i3),
                Arch::load_unaligned(values[j + lane_count * 3..].as_ptr()),
                acc3,
            );
            j += lane_count * 4;
        }
        let mut acc_vec = Arch::add(Arch::add(acc0, acc1), Arch::add(acc2, acc3));
        let simd_len = (row_nnz / lane_count) * lane_count;
        while j < simd_len {
            let idx = super::build_index_vector::<T, Arch>(&columns[j..j + lane_count]);
            acc_vec = Arch::fmadd(
                Arch::gather(x.as_ptr(), idx),
                Arch::load_unaligned(values[j..].as_ptr()),
                acc_vec,
            );
            j += lane_count;
        }
        acc_vec
    };
    // SAFETY: the parent module's target-feature invariant applies.
    let mut acc = unsafe { Arch::sum_reduce(acc_vec) };
    let mut j = unroll_tail_start(row_nnz, lane_count);
    while j < row_nnz {
        // SAFETY: validated columns and the entry size assertion prove
        // this index is in bounds.
        acc += values[j] * unsafe { *x.get_unchecked(columns[j] as usize) };
        j += 1;
    }
    *y_row += alpha * acc;
}

/// First scalar-drain index: rows shorter than one unrolled block start at
/// the SIMD-lane prefix, longer rows resume after the unrolled prefix.
///
/// The unrolled prefix consumed `(row_nnz / (lane*4)) * lane*4` elements; the
/// vector prefix consumed `(row_nnz / lane) * lane`. Their maximum is the
/// resume point, since both loops run in order on disjoint index ranges.
fn unroll_tail_start(row_nnz: usize, lane_count: usize) -> usize {
    let block = lane_count * 4;
    let after_unrolled = (row_nnz / block) * block;
    let after_vector = (row_nnz / lane_count) * lane_count;
    after_unrolled.max(after_vector)
}

#[cfg(test)]
mod tests {
    use super::unroll_tail_start;

    /// Resume index covers every element exactly once, regardless of row shape.
    #[test]
    fn tail_start_partitions_every_row_length() {
        for lane_count in [1usize, 4, 8] {
            for row_nnz in 0..257usize {
                let tail = unroll_tail_start(row_nnz, lane_count);
                let block = lane_count * 4;
                let after_unrolled = (row_nnz / block) * block;
                let after_vector = (row_nnz / lane_count) * lane_count;
                assert_eq!(tail, after_unrolled.max(after_vector));
                assert_eq!(tail % lane_count, 0);
                assert!(tail <= row_nnz);
            }
        }
    }
}
