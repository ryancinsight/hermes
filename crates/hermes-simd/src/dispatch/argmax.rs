//! Generic runtime-dispatch argmax kernel.
//!
//! Returns the first maximum, or `None` for empty or NaN-containing slices.

use hermes_simd_core::{
    align::Unaligned,
    arch::SimdArch,
    execution::Unmasked,
    kernel::{SimdArith, SimdBitwise, SimdCompare, SimdLoadStore, SimdMask, SimdReduce},
    scalar::Scalar,
    view::SimdView,
};
use hermes_simd_macros::runtime_dispatch;

#[runtime_dispatch(avx512f, avx2, neon, scalar)]
pub(super) fn dispatch_argmax_kernel<T, A>(data: &[T]) -> Option<(usize, T)>
where
    T: Scalar,
    A: SimdArch
        + SimdLoadStore<T>
        + SimdArith<T>
        + SimdBitwise<T>
        + SimdCompare<T>
        + SimdMask<T>
        + SimdReduce<T>,
{
    SimdView::<T, A, Unaligned, Unmasked, &[T]>::new(data)?.argmax()
}
