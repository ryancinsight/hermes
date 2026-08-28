//! Generic runtime-dispatch in-place scale kernel.
//!
//! `scale_in_place<T>(data, scalar)` broadcasts `scalar` to all SIMD lanes
//! then multiplies each chunk of `data` by it. The final partial vector uses
//! provider-owned masked memory and multiplication operations without touching
//! storage beyond the live slice.

use hermes_simd_core::{
    arch::SimdArch,
    kernel::{SimdArith, SimdLoadStore, SimdMask},
    scalar::Scalar,
};
use hermes_simd_macros::runtime_dispatch;

/// Apply the final partial scale vector without reading or writing beyond the
/// live tail. The partial-memory contract guarantees that inactive lanes do not
/// access the allocation beyond the tail.
#[inline(always)]
unsafe fn scale_masked_tail<T, A>(data: *mut T, scalar: T, tail: usize)
where
    T: Scalar,
    A: SimdArch + SimdArith<T> + SimdLoadStore<T> + SimdMask<T>,
{
    debug_assert!(tail > 0 && tail < A::LANE_COUNT);

    let mask = A::leading_k_mask(tail);
    let loaded = A::masked_load_partial(data, tail, mask, A::zero());
    let value = A::masked_mul(loaded, A::splat(scalar), mask, loaded);
    A::masked_store_partial(data, tail, mask, value);
}

#[runtime_dispatch(avx512f, avx2, neon, scalar)]
pub(super) fn dispatch_scale_kernel<T, A>(data: &mut [T], scalar: T)
where
    T: Scalar,
    A: SimdArch + SimdArith<T> + SimdLoadStore<T> + SimdMask<T>,
{
    let len = data.len();
    if len == 0 {
        return;
    }
    let lane_count = A::LANE_COUNT;
    let unroll_factor = A::UNROLL_FACTOR;
    let chunk_size = lane_count * unroll_factor;
    let unrolled_simd_len = (len / chunk_size) * chunk_size;
    let ptr = data.as_mut_ptr();

    // SAFETY: the unrolled and full-vector bounds are derived from `len`, and
    // the dispatch wrapper proves the target feature required by `A`.
    unsafe {
        let vsplat = A::splat(scalar);

        // ── 4× unrolled SIMD loop — hides load/store latency ────────────────
        let mut i = 0usize;
        while i < unrolled_simd_len {
            let p0 = ptr.add(i);
            let p1 = ptr.add(i + lane_count);
            let p2 = ptr.add(i + lane_count * 2);
            let p3 = ptr.add(i + lane_count * 3);
            A::store_unaligned(p0, A::mul(A::load_unaligned(p0), vsplat));
            A::store_unaligned(p1, A::mul(A::load_unaligned(p1), vsplat));
            A::store_unaligned(p2, A::mul(A::load_unaligned(p2), vsplat));
            A::store_unaligned(p3, A::mul(A::load_unaligned(p3), vsplat));
            i += chunk_size;
        }

        // ── Remaining full SIMD vectors ─────────────────────────────────────
        let simd_len = (len / lane_count) * lane_count;
        while i < simd_len {
            let p = ptr.add(i);
            A::store_unaligned(p, A::mul(A::load_unaligned(p), vsplat));
            i += lane_count;
        }
    }

    let simd_len = (len / lane_count) * lane_count;
    let tail = len - simd_len;
    if tail != 0 {
        // SAFETY: `simd_len` is the start of the remaining in-bounds prefix and
        // `tail` is its exact length, strictly smaller than the lane width.
        unsafe { scale_masked_tail::<T, A>(data.as_mut_ptr().add(simd_len), scalar, tail) };
    }
}

#[cfg(test)]
mod tests {
    use super::super::scale;

    #[test]
    fn scale_matches_scalar_reference_across_tail_sizes() {
        for &len in &[0usize, 1, 3, 7, 8, 9, 15, 16, 17, 63, 64, 65, 1027] {
            let mut data: Vec<f64> = (0..len).map(|i| i as f64 * 0.5 - 3.0).collect();
            let expected: Vec<f64> = data.iter().map(|&value| value * 1.75).collect();

            scale(&mut data, 1.75);
            assert_eq!(data, expected, "len {len}");
        }
    }

    #[test]
    fn scale_single_precision_matches_reference_at_partial_lengths() {
        for &len in &[5usize, 9, 17, 33, 65] {
            let mut data: Vec<f32> = (0..len).map(|i| i as f32 + 0.125).collect();
            let expected: Vec<f32> = data.iter().map(|&value| value * -0.75).collect();

            scale(&mut data, -0.75);
            assert_eq!(data, expected, "len {len}");
        }
    }
}
