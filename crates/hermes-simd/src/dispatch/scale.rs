//! Generic runtime-dispatch in-place scale kernel.
//!
//! `scale_in_place<T>(data, scalar)` broadcasts `scalar` to all SIMD lanes
//! then multiplies each chunk of `data` by it, covering the scalar tail
//! element-by-element.

use hermes_simd_core::{
    kernel::SimdKernel,
    scalar::Scalar,
    arch::SimdArch,
};
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
    let simd_len = (len / lane_count) * lane_count;
    let ptr = data.as_mut_ptr();

    unsafe {
        let vsplat = A::splat(scalar);
        let mut i = 0usize;
        while i < simd_len {
            let p = ptr.add(i);
            let v = A::load_unaligned(p);
            A::store_unaligned(p, A::mul(v, vsplat));
            i += lane_count;
        }
    }

    // Scalar tail
    for i in simd_len..len {
        data[i] = data[i] * scalar;
    }
}
