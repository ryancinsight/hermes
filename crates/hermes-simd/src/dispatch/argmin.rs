//! Generic runtime-dispatch argmin kernel.
//!
//! Returns the first minimum, or `None` for empty or NaN-containing slices.

use hermes_simd_core::{
    align::Unaligned, arch::SimdArch, execution::Unmasked, kernel::SimdKernel, scalar::Scalar,
    view::SimdView,
};
use hermes_simd_macros::runtime_dispatch;

#[expect(
    dead_code,
    reason = "The runtime-dispatch macro emits architecture-specific consumers"
)]
#[runtime_dispatch(avx512f, avx2, neon, scalar)]
pub(super) fn dispatch_argmin_kernel<T, A>(data: &[T]) -> Option<(usize, T)>
where
    T: Scalar,
    A: SimdArch + SimdKernel<T>,
{
    SimdView::<T, A, Unaligned, Unmasked, &[T]>::new(data)?.argmin()
}
