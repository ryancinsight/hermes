//! Dense-with-mask multiplication.

use super::{validate_spmv_sizes, SparseSpMv};
use crate::arch::SimdArch;
use crate::kernel::{SimdArith, SimdLoadStore, SimdMask, SimdReduce};
use crate::mask::PackedMask;
use crate::scalar::Scalar;
use crate::sparse::{DenseWithMask, DenseWithMaskData, SparseView};

#[inline(always)]
unsafe fn full_chunk_sum<T, Arch>(
    values: &[T],
    x: &[T],
    mask: &PackedMask<&[u64]>,
    row_offset: usize,
    simd_len: usize,
) -> Arch::Vector
where
    T: Scalar,
    Arch: SimdLoadStore<T> + SimdArith<T> + SimdMask<T>,
{
    let lane_count = Arch::LANE_COUNT;
    // SAFETY: the caller provides `values` and `x` with at least `simd_len`
    // elements, and `simd_len` is an exact multiple of LANE_COUNT. The mask
    // shape was validated at operation entry.
    unsafe {
        let zero_vec = Arch::zero();
        let mut acc_vec0 = zero_vec;
        let mut acc_vec1 = zero_vec;
        let mut acc_vec2 = zero_vec;
        let mut acc_vec3 = zero_vec;

        let unroll_len = (simd_len / (lane_count * 4)) * (lane_count * 4);
        let mut j = 0usize;
        while j < unroll_len {
            let mask0 =
                Arch::mask_from_bitmask(mask.lane_bits_in_bounds(row_offset + j, lane_count));
            acc_vec0 = Arch::masked_fmadd(
                Arch::masked_load_unaligned(values[j..].as_ptr(), mask0, zero_vec),
                Arch::load_unaligned(x[j..].as_ptr()),
                acc_vec0,
                mask0,
            );

            let mask1 = Arch::mask_from_bitmask(
                mask.lane_bits_in_bounds(row_offset + j + lane_count, lane_count),
            );
            acc_vec1 = Arch::masked_fmadd(
                Arch::masked_load_unaligned(values[j + lane_count..].as_ptr(), mask1, zero_vec),
                Arch::load_unaligned(x[j + lane_count..].as_ptr()),
                acc_vec1,
                mask1,
            );

            let mask2 = Arch::mask_from_bitmask(
                mask.lane_bits_in_bounds(row_offset + j + lane_count * 2, lane_count),
            );
            acc_vec2 = Arch::masked_fmadd(
                Arch::masked_load_unaligned(values[j + lane_count * 2..].as_ptr(), mask2, zero_vec),
                Arch::load_unaligned(x[j + lane_count * 2..].as_ptr()),
                acc_vec2,
                mask2,
            );

            let mask3 = Arch::mask_from_bitmask(
                mask.lane_bits_in_bounds(row_offset + j + lane_count * 3, lane_count),
            );
            acc_vec3 = Arch::masked_fmadd(
                Arch::masked_load_unaligned(values[j + lane_count * 3..].as_ptr(), mask3, zero_vec),
                Arch::load_unaligned(x[j + lane_count * 3..].as_ptr()),
                acc_vec3,
                mask3,
            );
            j += lane_count * 4;
        }

        let mut acc_vec = Arch::add(Arch::add(acc_vec0, acc_vec1), Arch::add(acc_vec2, acc_vec3));
        while j < simd_len {
            let chunk_mask =
                Arch::mask_from_bitmask(mask.lane_bits_in_bounds(row_offset + j, lane_count));
            acc_vec = Arch::masked_fmadd(
                Arch::masked_load_unaligned(values[j..].as_ptr(), chunk_mask, zero_vec),
                Arch::load_unaligned(x[j..].as_ptr()),
                acc_vec,
                chunk_mask,
            );
            j += lane_count;
        }
        acc_vec
    }
}

#[inline(always)]
fn scalar_rows<T: Scalar>(data: &DenseWithMaskData<'_, T>, x: &[T], y: &mut [T]) {
    for row in 0..data.nrows {
        let row_offset = row * data.ncols;
        let values = &data.values[row_offset..row_offset + data.ncols];
        let mut sum = T::ZERO;
        for col in 0..data.ncols {
            if data.mask.bit_in_bounds(row_offset + col) {
                sum += values[col] * x[col];
            }
        }
        y[row] += sum;
    }
}

/// Reduce one remainder through exact-prefix masked memory.
///
/// # Safety
/// The host must support `Arch`; `values` and `x` must each expose `tail`
/// elements, `tail < LANE_COUNT`, and `mask_bits` must not select a higher lane.
#[inline(always)]
unsafe fn tail_sum<T, Arch>(values: *const T, x: *const T, tail: usize, mask_bits: u64) -> T
where
    T: Scalar,
    Arch: SimdLoadStore<T> + SimdArith<T> + SimdMask<T> + SimdReduce<T>,
{
    // SAFETY: the caller limits both masks and pointer access to the live tail.
    let active = unsafe { Arch::mask_from_bitmask(mask_bits) };
    let prefix = unsafe { Arch::leading_k_mask(tail) };
    let tail_acc = unsafe {
        Arch::masked_fmadd(
            Arch::masked_load_partial(values, tail, active, Arch::zero()),
            Arch::masked_load_partial(x, tail, prefix, Arch::zero()),
            Arch::zero(),
            active,
        )
    };
    // SAFETY: the parent module's target-feature invariant applies.
    unsafe { Arch::sum_reduce(tail_acc) }
}

impl<T, Arch> SparseSpMv<T> for SparseView<'_, T, DenseWithMask, Arch>
where
    T: Scalar,
    Arch: SimdArch + SimdLoadStore<T> + SimdArith<T> + SimdMask<T> + SimdReduce<T>,
{
    #[inline]
    fn spmv(&self, x: &[T], y: &mut [T]) {
        let data = &self.data;
        validate_spmv_sizes(x.len(), y.len(), data.ncols, data.nrows, "DenseWithMask");
        let matrix_len = data.assert_valid_shape("spmv");
        debug_assert_eq!(matrix_len, data.values.len());

        let lane_count = Arch::LANE_COUNT;
        let masked_tail_threshold = lane_count.div_ceil(2);
        if data.ncols < masked_tail_threshold {
            scalar_rows(data, x, y);
            return;
        }

        let simd_len = (data.ncols / lane_count) * lane_count;
        let tail = data.ncols - simd_len;

        for row in 0..data.nrows {
            let row_offset = row * data.ncols;
            let values = &data.values[row_offset..row_offset + data.ncols];

            // SAFETY: full chunks are inside `values` and `x`; the validated
            // mask shape covers the complete row window.
            let acc_vec =
                unsafe { full_chunk_sum::<T, Arch>(values, x, &data.mask, row_offset, simd_len) };
            // SAFETY: the parent module's target-feature invariant applies.
            let mut acc = unsafe { Arch::sum_reduce(acc_vec) };

            // The measured eight-lane f32 crossover lies between three and
            // four live lanes. Preserve scalar evaluation below a
            // backend-relative half-register threshold.
            if tail < masked_tail_threshold {
                let mut col = simd_len;
                while col < data.ncols {
                    if data.mask.bit_in_bounds(row_offset + col) {
                        acc += values[col] * x[col];
                    }
                    col += 1;
                }
            } else if tail != 0 {
                // SAFETY: both slices expose the exact tail prefix and the
                // validated mask window cannot select a higher lane.
                acc += unsafe {
                    tail_sum::<T, Arch>(
                        values[simd_len..].as_ptr(),
                        x[simd_len..].as_ptr(),
                        tail,
                        data.mask.lane_bits_in_bounds(row_offset + simd_len, tail),
                    )
                };
            }
            y[row] += acc;
        }
    }
}
