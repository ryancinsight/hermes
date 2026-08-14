use super::super::popcount::{
    dispatch_reduce_popcount, dispatch_reduce_popcount_and, dispatch_reduce_popcount_or,
    dispatch_reduce_popcount_xor,
};
use super::super::{
    abs_reduce, argmax, argmin, axpy, binary, complex, dot, gemm, gemv, gemv_strided,
    gemv_transpose, gemv_transpose_strided, masked, max, min, scale, sparse, sum,
};
use super::{private, SimdOps};
use hermes_simd_core::scalar::Scalar as ScalarTrait;
use hermes_simd_core::sparse::{
    BlockedCooData, CsrData, DenseWithMaskData, SellPData, ValidatedData,
};
use hermes_simd_core::view::SimdError;
use hermes_simd_core::{Add, Div, Mul, Sub};
#[cfg(not(any(target_arch = "x86", target_arch = "x86_64", target_arch = "aarch64")))]
use hermes_simd_intrinsics::Scalar as ScalarArch;
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[expect(
    unused_imports,
    reason = "The architecture markers are consumed by cfg-selected dispatch implementations"
)]
use hermes_simd_intrinsics::{Avx2, Avx512, Neon, Scalar as ScalarArch};
#[cfg(target_arch = "aarch64")]
use hermes_simd_intrinsics::{Neon, Scalar as ScalarArch};

/// Method bodies shared verbatim by the three target-gated `SimdOps`
/// blanket impls below, which differ only in the architecture-kernel
/// bound each `where` clause requires. Defining them once keeps the
/// dispatch facade DRY and behavior identical across targets.
macro_rules! impl_simd_ops_methods {
    () => {
        #[inline(always)]
        fn sum(data: &[Self]) -> Self {
            sum::dispatch_sum::<Self>(data)
        }
        #[inline(always)]
        fn abs_sum(data: &[Self]) -> Self {
            abs_reduce::dispatch_abs_sum::<Self>(data)
        }
        #[inline(always)]
        fn abs_max(data: &[Self]) -> Self {
            abs_reduce::dispatch_abs_max::<Self>(data)
        }
        #[inline(always)]
        fn min(data: &[Self]) -> Self {
            min::dispatch_min::<Self>(data)
        }
        #[inline(always)]
        fn max(data: &[Self]) -> Self {
            max::dispatch_max::<Self>(data)
        }
        #[inline(always)]
        fn scale(data: &mut [Self], scalar: Self) {
            scale::dispatch_scale::<Self>(data, scalar)
        }
        #[inline(always)]
        fn argmin(data: &[Self]) -> Option<(usize, Self)> {
            argmin::dispatch_argmin::<Self>(data)
        }
        #[inline(always)]
        fn argmax(data: &[Self]) -> Option<(usize, Self)> {
            argmax::dispatch_argmax::<Self>(data)
        }
        #[inline(always)]
        fn dot(a: &[Self], b: &[Self]) -> Result<Self, SimdError> {
            dot::dispatch_dot::<Self>(a, b)
        }
        #[inline(always)]
        fn axpy(alpha: Self, x: &[Self], out: &mut [Self]) -> Result<(), SimdError> {
            axpy::dispatch_axpy::<Self>(alpha, x, out)
        }
        #[inline(always)]
        fn axpy_mul(
            alpha: Self,
            a: &[Self],
            b: &[Self],
            out: &mut [Self],
        ) -> Result<(), SimdError> {
            axpy::dispatch_axpy_mul::<Self>(alpha, a, b, out)
        }
        #[inline(always)]
        fn axpy_rows(
            alphas: &[Self],
            x: &[Self],
            out: &mut [Self],
            row_stride: usize,
            rows: usize,
            cols: usize,
        ) -> Result<(), SimdError> {
            axpy::dispatch_axpy_rows::<Self>(alphas, x, out, row_stride, rows, cols)
        }
        #[inline(always)]
        fn axpy_rows_batch(
            alphas: &[Self],
            x_panel: &[Self],
            out: &mut [Self],
            row_stride: usize,
            rows: usize,
            depth: usize,
            cols: usize,
        ) -> Result<(), SimdError> {
            axpy::dispatch_axpy_rows_batch::<Self>(
                alphas, x_panel, out, row_stride, rows, depth, cols,
            )
        }
        #[inline(always)]
        fn elementwise_mul(a: &[Self], b: &[Self], out: &mut [Self]) -> Result<(), SimdError> {
            binary::dispatch_elementwise_binary::<Self, Mul>(a, b, out, Mul)
        }
        #[inline(always)]
        fn elementwise_add(a: &[Self], b: &[Self], out: &mut [Self]) -> Result<(), SimdError> {
            binary::dispatch_elementwise_binary::<Self, Add>(a, b, out, Add)
        }
        #[inline(always)]
        fn elementwise_sub(a: &[Self], b: &[Self], out: &mut [Self]) -> Result<(), SimdError> {
            binary::dispatch_elementwise_binary::<Self, Sub>(a, b, out, Sub)
        }
        #[inline(always)]
        fn elementwise_div(a: &[Self], b: &[Self], out: &mut [Self]) -> Result<(), SimdError> {
            binary::dispatch_elementwise_binary::<Self, Div>(a, b, out, Div)
        }
        #[inline(always)]
        fn masked_sum(data: &[Self], mask: &[bool]) -> Self {
            masked::dispatch_masked_sum::<Self>(data, mask)
        }
        #[inline(always)]
        fn masked_dot(a: &[Self], b: &[Self], mask: &[bool]) -> Result<Self, SimdError> {
            masked::dispatch_masked_dot::<Self>(a, b, mask)
        }
        #[inline(always)]
        fn masked_add(
            a: &[Self],
            b: &[Self],
            mask: &[bool],
            out: &mut [Self],
        ) -> Result<(), SimdError> {
            masked::dispatch_masked_add::<Self>(a, b, mask, out)
        }
        #[inline(always)]
        fn spmv_csr(data: ValidatedData<CsrData<'_, Self>>, x: &[Self], y: &mut [Self]) {
            sparse::dispatch_spmv_csr::<Self>(data, x, y)
        }
        #[inline(always)]
        fn spmv_bcoo<const BM: usize, const BN: usize>(
            data: ValidatedData<BlockedCooData<'_, Self, BM, BN>>,
            x: &[Self],
            y: &mut [Self],
        ) {
            // Runtime-dispatched like the other sparse kernels (was hardcoded to
            // ScalarArch, which left the SIMD BlockedCoo paths dead at runtime).
            sparse::dispatch_spmv_bcoo::<Self, BM, BN>(data, x, y)
        }
        #[inline(always)]
        fn spmv_dense_masked(data: DenseWithMaskData<'_, Self>, x: &[Self], y: &mut [Self]) {
            sparse::dispatch_spmv_dense_masked::<Self>(data, x, y)
        }
        #[inline(always)]
        fn spmv_sellp<const C: usize>(
            data: ValidatedData<SellPData<'_, Self, C>>,
            x: &[Self],
            y: &mut [Self],
        ) {
            sparse::dispatch_spmv_sellp::<Self, C>(data, x, y)
        }
        #[inline(always)]
        fn tiled_gemm(
            a: &[Self],
            b: &[Self],
            c: &mut [Self],
            m: usize,
            n: usize,
            k: usize,
        ) -> Result<(), SimdError> {
            gemm::dispatch_tiled_gemm::<Self>(a, b, c, m, n, k)
        }
        #[inline(always)]
        fn gemv(
            a: &[Self],
            x: &[Self],
            y: &mut [Self],
            nrows: usize,
            ncols: usize,
        ) -> Result<(), SimdError> {
            gemv::dispatch_gemv::<Self>(a, x, y, nrows, ncols)
        }
        #[inline(always)]
        fn gemv_transpose(
            a: &[Self],
            x: &[Self],
            y: &mut [Self],
            nrows: usize,
            ncols: usize,
        ) -> Result<(), SimdError> {
            gemv_transpose::dispatch_gemv_transpose::<Self>(a, x, y, nrows, ncols)
        }
        #[inline(always)]
        fn gemv_strided(
            a: &[Self],
            x: &[Self],
            y: &mut [Self],
            nrows: usize,
            ncols: usize,
            lda: usize,
        ) -> Result<(), SimdError> {
            gemv_strided::dispatch_gemv_strided::<Self>(a, x, y, nrows, ncols, lda)
        }
        #[inline(always)]
        fn gemv_transpose_strided(
            a: &[Self],
            x: &[Self],
            y: &mut [Self],
            nrows: usize,
            ncols: usize,
            lda: usize,
        ) -> Result<(), SimdError> {
            gemv_transpose_strided::dispatch_gemv_transpose_strided::<Self>(
                a, x, y, nrows, ncols, lda,
            )
        }
        #[inline(always)]
        fn interleaved_complex_mul_assign<const CONJ_B: bool>(
            a: &mut [Self],
            b: &[Self],
        ) -> Result<(), SimdError>
        where
            Self: core::ops::Neg<Output = Self>,
        {
            complex::dispatch_interleaved_complex_mul_assign::<Self, CONJ_B>(a, b)
        }
        #[inline(always)]
        fn interleaved_complex_dot<const CONJ_B: bool>(
            a: &[Self],
            b: &[Self],
        ) -> Result<(Self, Self), SimdError>
        where
            Self: core::ops::Neg<Output = Self>,
        {
            complex::dispatch_interleaved_complex_dot::<Self, CONJ_B>(a, b)
        }
        #[inline(always)]
        fn reduce_popcount(data: &[Self]) -> usize {
            dispatch_reduce_popcount::<Self>(data)
        }
        #[inline(always)]
        fn reduce_popcount_and(a: &[Self], b: &[Self]) -> Result<usize, SimdError> {
            dispatch_reduce_popcount_and::<Self>(a, b)
        }
        #[inline(always)]
        fn reduce_popcount_or(a: &[Self], b: &[Self]) -> Result<usize, SimdError> {
            dispatch_reduce_popcount_or::<Self>(a, b)
        }
        #[inline(always)]
        fn reduce_popcount_xor(a: &[Self], b: &[Self]) -> Result<usize, SimdError> {
            dispatch_reduce_popcount_xor::<Self>(a, b)
        }
    };
}

/// x86/x86_64 specialized generic implementation of SimdOps.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
impl<T> SimdOps for T
where
    T: ScalarTrait + private::Sealed,
    ScalarArch: hermes_simd_core::kernel::SimdKernel<T>,
    Avx2: hermes_simd_core::kernel::SimdKernel<T>,
    Avx512: hermes_simd_core::kernel::SimdKernel<T>,
{
    impl_simd_ops_methods!();
}

/// AArch64 specialized generic implementation of SimdOps.
#[cfg(target_arch = "aarch64")]
impl<T> SimdOps for T
where
    T: ScalarTrait + private::Sealed,
    ScalarArch: hermes_simd_core::kernel::SimdKernel<T>,
    Neon: hermes_simd_core::kernel::SimdKernel<T>,
{
    impl_simd_ops_methods!();
}

/// Fallback generic implementation of SimdOps.
#[cfg(not(any(target_arch = "x86", target_arch = "x86_64", target_arch = "aarch64")))]
impl<T> SimdOps for T
where
    T: ScalarTrait + private::Sealed,
    ScalarArch: hermes_simd_core::kernel::SimdKernel<T>,
{
    impl_simd_ops_methods!();
}
