//! Runtime-dispatched sparse matrix-vector multiplication.

use hermes_simd_core::sparse::{
    CsrData, DenseWithMaskData, SellPData,
    SparseView, Csr, DenseWithMask, SellP,
};
use hermes_simd_core::kernel::SimdKernel;
use hermes_simd_core::scalar::Scalar;
use hermes_simd_core::arch::SimdArch;
use hermes_simd_macros::runtime_dispatch;

#[runtime_dispatch(avx512f, avx2, neon, scalar)]
pub(super) fn dispatch_spmv_csr_kernel<T, A>(data: CsrData<'_, T>, x: &[T], y: &mut [T])
where
    T: Scalar,
    A: SimdArch + SimdKernel<T>,
{
    SparseView::<T, Csr, A>::from_csr(data).spmv(x, y);
}

#[runtime_dispatch(avx512f, avx2, neon, scalar)]
pub(super) fn dispatch_spmv_dense_masked_kernel<T, A>(data: DenseWithMaskData<'_, T>, x: &[T], y: &mut [T])
where
    T: Scalar,
    A: SimdArch + SimdKernel<T>,
{
    SparseView::<T, DenseWithMask, A>::from_dense_with_mask(data).spmv(x, y);
}

#[runtime_dispatch(avx512f, avx2, neon, scalar)]
pub(super) fn dispatch_spmv_sellp4_kernel<T, A>(data: SellPData<'_, T, 4>, x: &[T], y: &mut [T])
where
    T: Scalar,
    A: SimdArch + SimdKernel<T>,
{
    SparseView::<T, SellP<4>, A>::from_sellp4(data).spmv(x, y);
}

#[runtime_dispatch(avx512f, avx2, neon, scalar)]
pub(super) fn dispatch_spmv_sellp8_kernel<T, A>(data: SellPData<'_, T, 8>, x: &[T], y: &mut [T])
where
    T: Scalar,
    A: SimdArch + SimdKernel<T>,
{
    SparseView::<T, SellP<8>, A>::from_sellp8(data).spmv(x, y);
}