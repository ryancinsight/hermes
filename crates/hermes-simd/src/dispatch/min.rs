//! Generic runtime-dispatch min-reduction kernel.

use hermes_simd_core::{
    align::Unaligned, arch::SimdArch, execution::Unmasked, kernel::SimdKernel, ops::Min,
    scalar::Scalar, view::SimdView,
};
use hermes_simd_macros::runtime_dispatch;

#[runtime_dispatch(avx512f, avx2, neon, scalar)]
pub(super) fn dispatch_min_kernel<T, A>(data: &[T]) -> T
where
    T: Scalar,
    A: SimdArch + SimdKernel<T>,
{
    match SimdView::<T, A, Unaligned, Unmasked, &[T]>::new(data) {
        Some(v) => v.reduce(Min),
        None => unsafe { core::hint::unreachable_unchecked() },
    }
}
