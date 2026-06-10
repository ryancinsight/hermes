//! Generic runtime-dispatch elementwise binary kernel.
//!
//! A single SIMD loop (`SimdView::zip_into`) parameterized by an `ElementOp` ZST
//! marker (`Add`/`Sub`/`Mul`/`Div`) — the SIMD-effect SSOT for elementwise binary
//! operations. Each `SimdOps::elementwise_*` method selects the marker at the call
//! site; monomorphization erases the ZST and emits one specialized kernel per
//! `(T, Op, Arch)`.

use hermes_simd_core::{
    align::Unaligned,
    arch::SimdArch,
    execution::Unmasked,
    kernel::SimdKernel,
    scalar::Scalar,
    view::{SimdError, SimdView},
    ElementOp,
};
use hermes_simd_macros::runtime_dispatch;

#[runtime_dispatch(avx512f, avx2, neon, scalar)]
pub(super) fn dispatch_elementwise_binary_kernel<T, Op, A>(
    a: &[T],
    b: &[T],
    out: &mut [T],
    op: Op,
) -> Result<(), SimdError>
where
    T: Scalar,
    Op: ElementOp<T>,
    A: SimdArch + SimdKernel<T>,
{
    match (
        SimdView::<T, A, Unaligned, Unmasked, &[T]>::new(a),
        SimdView::<T, A, Unaligned, Unmasked, &[T]>::new(b),
    ) {
        (Some(v1), Some(v2)) => v1.zip_into(&v2, out, op),
        _ => unsafe { core::hint::unreachable_unchecked() },
    }
}
