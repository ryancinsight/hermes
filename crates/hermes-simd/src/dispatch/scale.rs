//! Generic runtime-dispatch in-place scale kernel.
//!
//! `scale_in_place<T>(data, scalar)` broadcasts `scalar` to all SIMD lanes
//! then multiplies each chunk of `data` by it, covering the scalar tail
//! element-by-element.

use hermes_simd_core::{arch::SimdArch, kernel::SimdKernel, scalar::Scalar};
use hermes_simd_macros::runtime_dispatch;

#[runtime_dispatch(avx512f, avx2, neon, scalar)]
pub(super) fn dispatch_scale_kernel<T, A>(data: &mut [T], scalar: T)
where
    T: Scalar,
    A: SimdArch + SimdKernel<T>,
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

    // Scalar tail
    let simd_len = (len / lane_count) * lane_count;
    for i in simd_len..len {
        data[i] = data[i] * scalar;
    }
}
