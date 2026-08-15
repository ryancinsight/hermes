//! Generic runtime-dispatch dot product kernel.

use hermes_simd_core::{
    align::Unaligned,
    arch::SimdArch,
    execution::Unmasked,
    kernel::{SimdArith, SimdLoadStore, SimdMask, SimdReduce},
    scalar::Scalar,
    view::{SimdError, SimdView},
};
use hermes_simd_macros::runtime_dispatch;

#[runtime_dispatch(avx512f, avx2, neon, scalar)]
pub(super) fn dispatch_dot_kernel<T, A>(a: &[T], b: &[T]) -> Result<T, SimdError>
where
    T: Scalar,
    A: SimdArch + SimdLoadStore<T> + SimdArith<T> + SimdMask<T> + SimdReduce<T>,
{
    match (
        SimdView::<T, A, Unaligned, Unmasked, &[T]>::new(a),
        SimdView::<T, A, Unaligned, Unmasked, &[T]>::new(b),
    ) {
        (Some(v1), Some(v2)) => v1.dot(&v2),
        _ => unsafe { core::hint::unreachable_unchecked() },
    }
}
