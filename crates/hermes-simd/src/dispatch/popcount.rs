//! Generic runtime-dispatch population count and bitwise reduction kernels.
#![allow(missing_docs)]

use hermes_simd_core::{
    align::Unaligned,
    arch::SimdArch,
    execution::Unmasked,
    kernel::SimdKernel,
    scalar::Scalar,
    view::{SimdError, SimdView},
};
use hermes_simd_macros::runtime_dispatch;

/// Dispatch population count reduction over a slice.
#[allow(missing_docs)]
#[runtime_dispatch(avx512f, avx2, neon, scalar)]
pub fn dispatch_reduce_popcount_kernel<T, A>(data: &[T]) -> usize
where
    T: Scalar,
    A: SimdArch + SimdKernel<T>,
{
    match SimdView::<T, A, Unaligned, Unmasked, &[T]>::new(data) {
        Some(v) => v.reduce_popcount(),
        None => unsafe { core::hint::unreachable_unchecked() },
    }
}

/// Dispatch bitwise AND population count reduction over two slices.
#[allow(missing_docs)]
#[runtime_dispatch(avx512f, avx2, neon, scalar)]
pub fn dispatch_reduce_popcount_and_kernel<T, A>(a: &[T], b: &[T]) -> Result<usize, SimdError>
where
    T: Scalar,
    A: SimdArch + SimdKernel<T>,
{
    match (
        SimdView::<T, A, Unaligned, Unmasked, &[T]>::new(a),
        SimdView::<T, A, Unaligned, Unmasked, &[T]>::new(b),
    ) {
        (Some(v1), Some(v2)) => v1.reduce_popcount_and(&v2),
        _ => unsafe { core::hint::unreachable_unchecked() },
    }
}

/// Dispatch bitwise OR population count reduction over two slices.
#[allow(missing_docs)]
#[runtime_dispatch(avx512f, avx2, neon, scalar)]
pub fn dispatch_reduce_popcount_or_kernel<T, A>(a: &[T], b: &[T]) -> Result<usize, SimdError>
where
    T: Scalar,
    A: SimdArch + SimdKernel<T>,
{
    match (
        SimdView::<T, A, Unaligned, Unmasked, &[T]>::new(a),
        SimdView::<T, A, Unaligned, Unmasked, &[T]>::new(b),
    ) {
        (Some(v1), Some(v2)) => v1.reduce_popcount_or(&v2),
        _ => unsafe { core::hint::unreachable_unchecked() },
    }
}

/// Dispatch bitwise XOR population count reduction over two slices.
#[allow(missing_docs)]
#[runtime_dispatch(avx512f, avx2, neon, scalar)]
pub fn dispatch_reduce_popcount_xor_kernel<T, A>(a: &[T], b: &[T]) -> Result<usize, SimdError>
where
    T: Scalar,
    A: SimdArch + SimdKernel<T>,
{
    match (
        SimdView::<T, A, Unaligned, Unmasked, &[T]>::new(a),
        SimdView::<T, A, Unaligned, Unmasked, &[T]>::new(b),
    ) {
        (Some(v1), Some(v2)) => v1.reduce_popcount_xor(&v2),
        _ => unsafe { core::hint::unreachable_unchecked() },
    }
}
