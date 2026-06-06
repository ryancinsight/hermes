//! Generic runtime-dispatch elementwise multiplication kernel.

use hermes_simd_core::{
    view::{SimdView, SimdError},
    align::Unaligned,
    execution::Unmasked,
    kernel::SimdKernel,
    scalar::Scalar,
    arch::SimdArch,
};
use hermes_simd_macros::runtime_dispatch;

#[runtime_dispatch(avx512f, avx2, neon, scalar)]
pub(super) fn dispatch_elementwise_mul_kernel<T, A>(a: &[T], b: &[T], out: &mut [T]) -> Result<(), SimdError>
where
    T: Scalar,
    A: SimdArch + SimdKernel<T>,
{
    match (
        SimdView::<T, A, Unaligned, Unmasked, &[T]>::new(a),
        SimdView::<T, A, Unaligned, Unmasked, &[T]>::new(b),
    ) {
        (Some(v1), Some(v2)) => v1.elementwise_mul(&v2, out),
        _ => unsafe { core::hint::unreachable_unchecked() },
    }
}