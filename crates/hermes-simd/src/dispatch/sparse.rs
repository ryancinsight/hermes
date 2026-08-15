//! Runtime-dispatched sparse matrix-vector multiplication.

use hermes_simd_core::arch::SimdArch;
use hermes_simd_core::kernel::{SimdArith, SimdGather, SimdLoadStore, SimdMask, SimdReduce};
use hermes_simd_core::scalar::Scalar;
use hermes_simd_core::sparse::{
    BlockedCoo, BlockedCooData, Csr, CsrData, DenseWithMask, DenseWithMaskData, SellP, SellPData,
    SparseSpMv, SparseView, Validated, ValidatedData,
};
use hermes_simd_macros::runtime_dispatch;

#[runtime_dispatch(avx512f, avx2, neon, scalar)]
pub(super) fn dispatch_spmv_csr_kernel<T, A>(
    data: ValidatedData<CsrData<'_, T>>,
    x: &[T],
    y: &mut [T],
) where
    T: Scalar,
    A: SimdArch + SimdLoadStore<T> + SimdArith<T> + SimdGather<T> + SimdReduce<T>,
{
    SparseView::<T, Validated<Csr>, A>::from_validated_csr(data).spmv(x, y);
}

#[runtime_dispatch(avx512f, avx2, neon, scalar)]
pub(super) fn dispatch_spmv_dense_masked_kernel<T, A>(
    data: DenseWithMaskData<'_, T>,
    x: &[T],
    y: &mut [T],
) where
    T: Scalar,
    A: SimdArch + SimdLoadStore<T> + SimdArith<T> + SimdMask<T> + SimdReduce<T>,
{
    SparseView::<T, DenseWithMask, A>::from_dense_with_mask(data).spmv(x, y);
}

#[runtime_dispatch(avx512f, avx2, neon, scalar)]
pub(super) fn dispatch_spmv_sellp_kernel<T, const C: usize, A>(
    data: ValidatedData<SellPData<'_, T, C>>,
    x: &[T],
    y: &mut [T],
) where
    T: Scalar,
    A: SimdArch + SimdLoadStore<T> + SimdArith<T> + SimdGather<T>,
{
    SparseView::<T, Validated<SellP<C>>, A>::from_validated_sellp(data).spmv(x, y);
}

#[runtime_dispatch(avx512f, avx2, neon, scalar)]
pub(super) fn dispatch_spmv_bcoo_kernel<T, const BM: usize, const BN: usize, A>(
    data: ValidatedData<BlockedCooData<'_, T, BM, BN>>,
    x: &[T],
    y: &mut [T],
) where
    T: Scalar,
    A: SimdArch + SimdLoadStore<T> + SimdArith<T> + SimdReduce<T>,
{
    SparseView::<T, Validated<BlockedCoo<BM, BN>>, A>::from_validated_blocked_coo(data).spmv(x, y);
}
