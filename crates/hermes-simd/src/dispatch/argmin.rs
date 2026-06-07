//! Generic runtime-dispatch argmin kernel.
//!
//! Returns `None` for empty slices, `Some((index, value))` for the first minimum element.

use hermes_simd_core::{
    view::SimdView,
    align::Unaligned,
    execution::Unmasked,
    kernel::SimdKernel,
    scalar::Scalar,
    arch::SimdArch,
};
use hermes_simd_macros::runtime_dispatch;

#[allow(dead_code)]
#[runtime_dispatch(avx512f, avx2, neon, scalar)]
pub(super) fn dispatch_argmin_kernel<T, A>(data: &[T]) -> Option<(usize, T)>
where
    T: Scalar,
    A: SimdArch + SimdKernel<T>,
{
    match SimdView::<T, A, Unaligned, Unmasked, &[T]>::new(data) {
        Some(v) => v.argmin(),
        None => unsafe { core::hint::unreachable_unchecked() },
    }
}
