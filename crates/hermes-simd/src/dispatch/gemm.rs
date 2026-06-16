//! Generic runtime-dispatch tiled GEMM kernel.

use hermes_simd_core::{
    align::Unaligned,
    arch::SimdArch,
    execution::Unmasked,
    kernel::SimdKernel,
    scalar::Scalar,
    view::{SimdError, SimdView},
};
use hermes_simd_macros::runtime_dispatch;

#[runtime_dispatch(avx512f, avx2, neon, scalar)]
pub(super) fn dispatch_tiled_gemm_kernel<T, A>(
    a: &[T],
    b: &[T],
    c: &mut [T],
    m: usize,
    n: usize,
    k: usize,
) -> Result<(), SimdError>
where
    T: Scalar,
    A: SimdArch + SimdKernel<T>,
{
    match (
        SimdView::<T, A, Unaligned, Unmasked, &[T]>::new(a),
        SimdView::<T, A, Unaligned, Unmasked, &[T]>::new(b),
    ) {
        (Some(v1), Some(v2)) => {
            use hermes_simd_core::tiling::{TilingPolicy, TilingStrategy};

            let decision = crate::dispatcher::AdaptiveDispatcher::select_backend(
                m,
                n,
                k,
                a.as_ptr(),
                a.len(),
                b.as_ptr(),
                b.len(),
            );

            match decision {
                crate::dispatcher::DispatchDecision::Scalar => {
                    let is_too_small = (m < 16) || (n < 16) || (k < 32);
                    if A::LANE_COUNT > 8 && !is_too_small {
                        <TilingPolicy<6, 4> as TilingStrategy<T, A, Unaligned>>::gemm(
                            &v1, &v2, c, m, n, k,
                        )
                    } else if A::LANE_COUNT > 1 && !is_too_small {
                        <TilingPolicy<3, 4> as TilingStrategy<T, A, Unaligned>>::gemm(
                            &v1, &v2, c, m, n, k,
                        )
                    } else {
                        <TilingPolicy<1, 1> as TilingStrategy<T, A, Unaligned>>::gemm(
                            &v1, &v2, c, m, n, k,
                        )
                    }
                }
                _ => {
                    if A::LANE_COUNT > 8 {
                        <TilingPolicy<6, 4> as TilingStrategy<T, A, Unaligned>>::gemm(
                            &v1, &v2, c, m, n, k,
                        )
                    } else if A::LANE_COUNT > 1 {
                        <TilingPolicy<3, 4> as TilingStrategy<T, A, Unaligned>>::gemm(
                            &v1, &v2, c, m, n, k,
                        )
                    } else {
                        <TilingPolicy<1, 1> as TilingStrategy<T, A, Unaligned>>::gemm(
                            &v1, &v2, c, m, n, k,
                        )
                    }
                }
            }
        }
        _ => unsafe { core::hint::unreachable_unchecked() },
    }
}
