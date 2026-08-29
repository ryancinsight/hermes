//! Compressed Sparse Row multiplication.

use super::{build_index_vector, validate_spmv_sizes, SparseSpMv};
use crate::arch::SimdArch;
use crate::kernel::{SimdArith, SimdGather, SimdLoadStore, SimdReduce};
use crate::scalar::Scalar;
use crate::sparse::{Csr, SparseView, Validated};

const PREFETCH_UNROLL_DISTANCE: usize = 1;

#[inline(always)]
unsafe fn prefetch_columns<T, Arch>(x: *const T, cols: *const i32, start: usize, len: usize)
where
    T: Scalar,
    Arch: SimdLoadStore<T>,
{
    let end = start + len;
    let mut cursor = start;
    while cursor < end {
        // SAFETY: the caller proves `[start, start + len)` lies in the current
        // validated CSR row, whose columns are nonnegative and below `ncols`.
        let column = unsafe { *cols.add(cursor) } as usize;
        // SAFETY: validated columns plus the SpMV entry length check prove the
        // addressed dense element exists.
        unsafe { Arch::prefetch_read(x.add(column)) };
        cursor += 1;
    }
}

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

                let unroll_width = lane_count * 4;
                let unroll_len = (row_nnz / unroll_width) * unroll_width;
                let prefetch_offset = unroll_width * PREFETCH_UNROLL_DISTANCE;
                macro_rules! accumulate_unroll {
                    ($j:expr) => {{
                        let j = $j;
                        let idx0 = build_index_vector::<T, Arch>(&cols[j..j + lane_count]);
                        acc_vec0 = Arch::fmadd(
                            Arch::gather(x.as_ptr(), idx0),
                            Arch::load_unaligned(vals[j..].as_ptr()),
                            acc_vec0,
                        );

                        let idx1 = build_index_vector::<T, Arch>(
                            &cols[j + lane_count..j + lane_count * 2],
                        );
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
                    }};
                }

                let prefetch_len = unroll_len.saturating_sub(prefetch_offset);
                let mut j = 0usize;
                if Arch::SUPPORTS_READ_PREFETCH {
                    while j < prefetch_len {
                        prefetch_columns::<T, Arch>(
                            x.as_ptr(),
                            cols.as_ptr(),
                            j + prefetch_offset,
                            unroll_width,
                        );
                        accumulate_unroll!(j);
                        j += unroll_width;
                    }
                    while j < unroll_len {
                        accumulate_unroll!(j);
                        j += unroll_width;
                    }
                } else {
                    while j < unroll_len {
                        accumulate_unroll!(j);
                        j += unroll_width;
                    }
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
