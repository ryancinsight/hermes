//! Runtime-dispatched SIMD operations.
//!
//! # Monomorphization chain
//!
//! `sum::<f32>(data)` -> `f32::sum(data)` -> `sum::dispatch_sum::<f32>(data)` -> avx2 kernel.

mod abs_reduce;
pub mod argmax;
pub mod argmin;
mod axpy;
pub mod binary;
pub mod complex;
pub mod dot;
pub mod gemm;
pub mod gemv;
pub mod gemv_transpose;
pub mod masked;
pub mod max;
pub mod min;
pub mod modular;
pub mod scale;
pub mod sparse;
pub mod sum;

use hermes_simd_core::scalar::Scalar as ScalarTrait;
use hermes_simd_core::sparse::{
    BlockedCoo, BlockedCooData, CsrData, DenseWithMaskData, SellPData, SparseSpMv, SparseView,
};
use hermes_simd_core::view::SimdError;
use hermes_simd_core::{Add, Div, Mul, Sub};
#[cfg(not(any(target_arch = "x86", target_arch = "x86_64", target_arch = "aarch64")))]
use hermes_simd_intrinsics::Scalar as ScalarArch;
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[allow(unused_imports)]
use hermes_simd_intrinsics::{Avx2, Avx512, Neon, Scalar as ScalarArch};
#[cfg(target_arch = "aarch64")]
use hermes_simd_intrinsics::{Neon, Scalar as ScalarArch};

mod private {
    pub trait Sealed {}
}

impl private::Sealed for f32 {}
impl private::Sealed for f64 {}
impl private::Sealed for half::f16 {}
impl private::Sealed for half::bf16 {}
impl private::Sealed for i8 {}
impl private::Sealed for i16 {}
impl private::Sealed for i32 {}

impl private::Sealed for hermes_numeric::F16 {}
impl private::Sealed for hermes_numeric::F32 {}
impl private::Sealed for hermes_numeric::F64 {}
impl private::Sealed for hermes_numeric::Bf16 {}
impl private::Sealed for hermes_numeric::Bf8 {}
impl private::Sealed for hermes_numeric::Bf4 {}
impl private::Sealed for hermes_numeric::F8 {}
impl private::Sealed for hermes_numeric::F4 {}
impl private::Sealed for hermes_numeric::I8 {}
impl private::Sealed for hermes_numeric::I16 {}
impl private::Sealed for hermes_numeric::I32 {}

/// Sealed extension trait implementing dynamic runtime SIMD dispatch for any `T: Scalar`.
pub trait SimdOps: ScalarTrait + private::Sealed {
    /// Reduces the slice to its sum.
    fn sum(data: &[Self]) -> Self;
    /// Reduces the slice to `Σ |x|` (L1-norm accumulator); `T::ZERO` for empty.
    fn abs_sum(data: &[Self]) -> Self;
    /// Reduces the slice to `max |x|` (∞-norm accumulator); `T::ZERO` for empty.
    fn abs_max(data: &[Self]) -> Self;
    /// Reduces the slice to its minimum element.
    ///
    /// Returns `T::MAX_VALUE` for empty slices (the identity element for min).
    fn min(data: &[Self]) -> Self;
    /// Reduces the slice to its maximum element.
    ///
    /// Returns `T::MIN_VALUE` for empty slices (the identity element for max).
    fn max(data: &[Self]) -> Self;
    /// Multiplies every element by `scalar` in-place.
    fn scale(data: &mut [Self], scalar: Self);
    /// Returns `Some((index, value))` of the minimum element, or `None` for empty.
    fn argmin(data: &[Self]) -> Option<(usize, Self)>;
    /// Returns `Some((index, value))` of the maximum element, or `None` for empty.
    fn argmax(data: &[Self]) -> Option<(usize, Self)>;
    /// Computes the dot product of two slices.
    fn dot(a: &[Self], b: &[Self]) -> Result<Self, SimdError>;
    /// Fused row update `out[i] += alpha * x[i]` (AXPY) with no temporaries.
    fn axpy(alpha: Self, x: &[Self], out: &mut [Self]) -> Result<(), SimdError>;
    /// Fused multi-row update `out[row, i] += alphas[row] * x[i]`.
    fn axpy_rows(
        alphas: &[Self],
        x: &[Self],
        out: &mut [Self],
        row_stride: usize,
        rows: usize,
        cols: usize,
    ) -> Result<(), SimdError>;
    /// Fused batched multi-row update:
    /// `out[row, i] += sum_k alphas[k, row] * x_panel[k, i]`.
    fn axpy_rows_batch(
        alphas: &[Self],
        x_panel: &[Self],
        out: &mut [Self],
        row_stride: usize,
        rows: usize,
        depth: usize,
        cols: usize,
    ) -> Result<(), SimdError>;
    /// Computes the elementwise product and writes to `out`.
    fn elementwise_mul(a: &[Self], b: &[Self], out: &mut [Self]) -> Result<(), SimdError>;
    /// Computes the elementwise sum `a[i] + b[i]` and writes to `out`.
    fn elementwise_add(a: &[Self], b: &[Self], out: &mut [Self]) -> Result<(), SimdError>;
    /// Computes the elementwise difference `a[i] - b[i]` and writes to `out`.
    fn elementwise_sub(a: &[Self], b: &[Self], out: &mut [Self]) -> Result<(), SimdError>;
    /// Computes the elementwise quotient `a[i] / b[i]` and writes to `out`.
    fn elementwise_div(a: &[Self], b: &[Self], out: &mut [Self]) -> Result<(), SimdError>;
    /// Computes the sum of elements matching a boolean mask.
    fn masked_sum(data: &[Self], mask: &[bool]) -> Self;
    /// Computes the dot product of elements matching a boolean mask.
    fn masked_dot(a: &[Self], b: &[Self], mask: &[bool]) -> Result<Self, SimdError>;
    /// Computes the elementwise sum of elements matching a boolean mask.
    fn masked_add(a: &[Self], b: &[Self], mask: &[bool], out: &mut [Self])
        -> Result<(), SimdError>;
    /// Computes sparse SpMV using CSR.
    fn spmv_csr(data: CsrData<'_, Self>, x: &[Self], y: &mut [Self]);
    /// Computes sparse SpMV using const-generic Blocked-COO tiles.
    fn spmv_bcoo<const BM: usize, const BN: usize>(
        data: BlockedCooData<'_, Self, BM, BN>,
        x: &[Self],
        y: &mut [Self],
    );
    /// Computes sparse SpMV using Dense-with-Mask.
    fn spmv_dense_masked(data: DenseWithMaskData<'_, Self>, x: &[Self], y: &mut [Self]);
    /// Computes sparse SpMV using const-generic Sliced ELLPACK (SELL-p).
    fn spmv_sellp<const C: usize>(data: SellPData<'_, Self, C>, x: &[Self], y: &mut [Self]);
    /// Computes register-blocked tiled GEMM: `c += A * B`.
    fn tiled_gemm(
        a: &[Self],
        b: &[Self],
        c: &mut [Self],
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<(), SimdError>;
    /// Computes register-blocked GEMV: `y += A * x` (`A` row-major `nrows × ncols`).
    fn gemv(
        a: &[Self],
        x: &[Self],
        y: &mut [Self],
        nrows: usize,
        ncols: usize,
    ) -> Result<(), SimdError>;
    /// Computes register-blocked transposed GEMV: `y += Aᵀ * x`
    /// (`A` row-major `nrows × ncols`, `x` length `nrows`, `y` length `ncols`).
    fn gemv_transpose(
        a: &[Self],
        x: &[Self],
        y: &mut [Self],
        nrows: usize,
        ncols: usize,
    ) -> Result<(), SimdError>;
    /// Multiplies interleaved complex lanes in-place: `a[k] *= b[k]`
    /// (`a[k] *= conj(b[k])` when `CONJ_B`).
    fn interleaved_complex_mul_assign<const CONJ_B: bool>(
        a: &mut [Self],
        b: &[Self],
    ) -> Result<(), SimdError>;
    /// Computes the interleaved complex dot product `(re, im)` of `sum(a[k] * b[k])`
    /// (`sum(a[k] * conj(b[k]))` when `CONJ_B`).
    fn interleaved_complex_dot<const CONJ_B: bool>(
        a: &[Self],
        b: &[Self],
    ) -> Result<(Self, Self), SimdError>;
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
        axpy::dispatch_axpy_rows_batch::<Self>(alphas, x_panel, out, row_stride, rows, depth, cols)
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
    fn spmv_csr(data: CsrData<'_, Self>, x: &[Self], y: &mut [Self]) {
        sparse::dispatch_spmv_csr::<Self>(data, x, y)
    }
    #[inline(always)]
    fn spmv_bcoo<const BM: usize, const BN: usize>(
        data: BlockedCooData<'_, Self, BM, BN>,
        x: &[Self],
        y: &mut [Self],
    ) {
        SparseView::<Self, BlockedCoo<BM, BN>, ScalarArch>::from_blocked_coo(data).spmv(x, y);
    }
    #[inline(always)]
    fn spmv_dense_masked(data: DenseWithMaskData<'_, Self>, x: &[Self], y: &mut [Self]) {
        sparse::dispatch_spmv_dense_masked::<Self>(data, x, y)
    }
    #[inline(always)]
    fn spmv_sellp<const C: usize>(data: SellPData<'_, Self, C>, x: &[Self], y: &mut [Self]) {
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
    fn interleaved_complex_mul_assign<const CONJ_B: bool>(
        a: &mut [Self],
        b: &[Self],
    ) -> Result<(), SimdError> {
        complex::dispatch_interleaved_complex_mul_assign::<Self, CONJ_B>(a, b)
    }
    #[inline(always)]
    fn interleaved_complex_dot<const CONJ_B: bool>(
        a: &[Self],
        b: &[Self],
    ) -> Result<(Self, Self), SimdError> {
        complex::dispatch_interleaved_complex_dot::<Self, CONJ_B>(a, b)
    }
}

/// AArch64 specialized generic implementation of SimdOps.
#[cfg(target_arch = "aarch64")]
impl<T> SimdOps for T
where
    T: ScalarTrait + private::Sealed,
    ScalarArch: hermes_simd_core::kernel::SimdKernel<T>,
    Neon: hermes_simd_core::kernel::SimdKernel<T>,
{
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
        axpy::dispatch_axpy_rows_batch::<Self>(alphas, x_panel, out, row_stride, rows, depth, cols)
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
    fn spmv_csr(data: CsrData<'_, Self>, x: &[Self], y: &mut [Self]) {
        sparse::dispatch_spmv_csr::<Self>(data, x, y)
    }
    #[inline(always)]
    fn spmv_bcoo<const BM: usize, const BN: usize>(
        data: BlockedCooData<'_, Self, BM, BN>,
        x: &[Self],
        y: &mut [Self],
    ) {
        SparseView::<Self, BlockedCoo<BM, BN>, ScalarArch>::from_blocked_coo(data).spmv(x, y);
    }
    #[inline(always)]
    fn spmv_dense_masked(data: DenseWithMaskData<'_, Self>, x: &[Self], y: &mut [Self]) {
        sparse::dispatch_spmv_dense_masked::<Self>(data, x, y)
    }
    #[inline(always)]
    fn spmv_sellp<const C: usize>(data: SellPData<'_, Self, C>, x: &[Self], y: &mut [Self]) {
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
    fn interleaved_complex_mul_assign<const CONJ_B: bool>(
        a: &mut [Self],
        b: &[Self],
    ) -> Result<(), SimdError> {
        complex::dispatch_interleaved_complex_mul_assign::<Self, CONJ_B>(a, b)
    }
    #[inline(always)]
    fn interleaved_complex_dot<const CONJ_B: bool>(
        a: &[Self],
        b: &[Self],
    ) -> Result<(Self, Self), SimdError> {
        complex::dispatch_interleaved_complex_dot::<Self, CONJ_B>(a, b)
    }
}

/// Fallback generic implementation of SimdOps.
#[cfg(not(any(target_arch = "x86", target_arch = "x86_64", target_arch = "aarch64")))]
impl<T> SimdOps for T
where
    T: ScalarTrait + private::Sealed,
    ScalarArch: hermes_simd_core::kernel::SimdKernel<T>,
{
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
        axpy::dispatch_axpy_rows_batch::<Self>(alphas, x_panel, out, row_stride, rows, depth, cols)
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
    fn spmv_csr(data: CsrData<'_, Self>, x: &[Self], y: &mut [Self]) {
        sparse::dispatch_spmv_csr::<Self>(data, x, y)
    }
    #[inline(always)]
    fn spmv_bcoo<const BM: usize, const BN: usize>(
        data: BlockedCooData<'_, Self, BM, BN>,
        x: &[Self],
        y: &mut [Self],
    ) {
        SparseView::<Self, BlockedCoo<BM, BN>, ScalarArch>::from_blocked_coo(data).spmv(x, y);
    }
    #[inline(always)]
    fn spmv_dense_masked(data: DenseWithMaskData<'_, Self>, x: &[Self], y: &mut [Self]) {
        sparse::dispatch_spmv_dense_masked::<Self>(data, x, y)
    }
    #[inline(always)]
    fn spmv_sellp<const C: usize>(data: SellPData<'_, Self, C>, x: &[Self], y: &mut [Self]) {
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
    fn interleaved_complex_mul_assign<const CONJ_B: bool>(
        a: &mut [Self],
        b: &[Self],
    ) -> Result<(), SimdError> {
        complex::dispatch_interleaved_complex_mul_assign::<Self, CONJ_B>(a, b)
    }
    #[inline(always)]
    fn interleaved_complex_dot<const CONJ_B: bool>(
        a: &[Self],
        b: &[Self],
    ) -> Result<(Self, Self), SimdError> {
        complex::dispatch_interleaved_complex_dot::<Self, CONJ_B>(a, b)
    }
}

/// Computes the sum of elements in the slice using runtime-dispatched SIMD.
#[inline(always)]
pub fn sum<T: SimdOps>(data: &[T]) -> T {
    T::sum(data)
}

/// Computes the minimum element of the slice using runtime-dispatched SIMD.
///
/// Returns `T::MAX_VALUE` for empty slices.
#[inline(always)]
pub fn min<T: SimdOps>(data: &[T]) -> T {
    T::min(data)
}

/// Computes the maximum element of the slice using runtime-dispatched SIMD.
///
/// Returns `T::MIN_VALUE` for empty slices.
#[inline(always)]
pub fn max<T: SimdOps>(data: &[T]) -> T {
    T::max(data)
}

/// Reduces the slice to `Σ |x|` (L1-norm accumulator); `T::ZERO` for empty.
#[inline(always)]
pub fn abs_sum<T: SimdOps>(data: &[T]) -> T {
    T::abs_sum(data)
}

/// Reduces the slice to `max |x|` (∞-norm accumulator); `T::ZERO` for empty.
#[inline(always)]
pub fn abs_max<T: SimdOps>(data: &[T]) -> T {
    T::abs_max(data)
}

/// Multiplies every element of `data` by `scalar` in-place.
#[inline(always)]
pub fn scale<T: SimdOps>(data: &mut [T], scalar: T) {
    T::scale(data, scalar)
}

/// Returns `Some((index, value))` of the minimum element, or `None` for empty.
#[inline(always)]
pub fn argmin<T: SimdOps>(data: &[T]) -> Option<(usize, T)> {
    T::argmin(data)
}

/// Returns `Some((index, value))` of the maximum element, or `None` for empty.
#[inline(always)]
pub fn argmax<T: SimdOps>(data: &[T]) -> Option<(usize, T)> {
    T::argmax(data)
}

/// Computes the dot product of two slices using runtime-dispatched SIMD.
#[inline(always)]
pub fn dot<T: SimdOps>(a: &[T], b: &[T]) -> Result<T, SimdError> {
    T::dot(a, b)
}

/// Fused row update `out[i] += alpha * x[i]` (AXPY) via runtime-dispatched
/// SIMD with no temporary allocation. Errors on length mismatch.
#[inline(always)]
pub fn axpy<T: SimdOps>(alpha: T, x: &[T], out: &mut [T]) -> Result<(), SimdError> {
    T::axpy(alpha, x, out)
}

/// Fused multi-row update `out[row, i] += alphas[row] * x[i]` via one
/// runtime-dispatched SIMD kernel. `out` is a row-major strided window.
#[inline(always)]
pub fn axpy_rows<T: SimdOps>(
    alphas: &[T],
    x: &[T],
    out: &mut [T],
    row_stride: usize,
    rows: usize,
    cols: usize,
) -> Result<(), SimdError> {
    T::axpy_rows(alphas, x, out, row_stride, rows, cols)
}

/// Fused batched multi-row update:
/// `out[row, i] += sum_k alphas[k, row] * x_panel[k, i]` via one
/// runtime-dispatched SIMD kernel. `alphas` is depth-major with `rows`
/// elements per depth, `x_panel` is depth-major with `cols` elements per
/// depth, and `out` is a row-major strided window.
#[inline(always)]
pub fn axpy_rows_batch<T: SimdOps>(
    alphas: &[T],
    x_panel: &[T],
    out: &mut [T],
    row_stride: usize,
    rows: usize,
    depth: usize,
    cols: usize,
) -> Result<(), SimdError> {
    T::axpy_rows_batch(alphas, x_panel, out, row_stride, rows, depth, cols)
}

/// Computes the elementwise multiplication of two slices and writes to `out`.
#[inline(always)]
pub fn elementwise_mul<T: SimdOps>(a: &[T], b: &[T], out: &mut [T]) -> Result<(), SimdError> {
    T::elementwise_mul(a, b, out)
}

/// Computes the elementwise sum of two slices and writes to `out`.
#[inline(always)]
pub fn elementwise_add<T: SimdOps>(a: &[T], b: &[T], out: &mut [T]) -> Result<(), SimdError> {
    T::elementwise_add(a, b, out)
}

/// Computes the elementwise difference of two slices and writes to `out`.
#[inline(always)]
pub fn elementwise_sub<T: SimdOps>(a: &[T], b: &[T], out: &mut [T]) -> Result<(), SimdError> {
    T::elementwise_sub(a, b, out)
}

/// Computes the elementwise quotient of two slices and writes to `out`.
#[inline(always)]
pub fn elementwise_div<T: SimdOps>(a: &[T], b: &[T], out: &mut [T]) -> Result<(), SimdError> {
    T::elementwise_div(a, b, out)
}

/// Executes one exact modular radix-2 NTT butterfly stage over `u64` residues.
#[inline]
pub fn ntt_butterfly_stage_u64(
    data: &mut [u64],
    stage_len: usize,
    twiddles: &[u64],
    modulus: u64,
) -> Result<(), SimdError> {
    modular::ntt_butterfly_stage_u64(data, stage_len, twiddles, modulus)
}

/// Computes the sum of elements matching a boolean mask.
#[inline(always)]
pub fn masked_sum<T: SimdOps>(data: &[T], mask: &[bool]) -> T {
    T::masked_sum(data, mask)
}

/// Computes the dot product of elements matching a boolean mask.
#[inline(always)]
pub fn masked_dot<T: SimdOps>(a: &[T], b: &[T], mask: &[bool]) -> Result<T, SimdError> {
    T::masked_dot(a, b, mask)
}

/// Computes the elementwise sum of elements matching a boolean mask.
#[inline(always)]
pub fn masked_add<T: SimdOps>(
    a: &[T],
    b: &[T],
    mask: &[bool],
    out: &mut [T],
) -> Result<(), SimdError> {
    T::masked_add(a, b, mask, out)
}

/// Computes sparse SpMV using CSR.
#[inline(always)]
pub fn spmv_csr<T: SimdOps>(data: CsrData<'_, T>, x: &[T], y: &mut [T]) {
    T::spmv_csr(data, x, y)
}

/// Computes sparse SpMV using const-generic Blocked-COO tiles.
#[inline(always)]
pub fn spmv_bcoo<T: SimdOps, const BM: usize, const BN: usize>(
    data: BlockedCooData<'_, T, BM, BN>,
    x: &[T],
    y: &mut [T],
) {
    T::spmv_bcoo::<BM, BN>(data, x, y)
}

/// Computes sparse SpMV using Dense-with-Mask.
#[inline(always)]
pub fn spmv_dense_masked<T: SimdOps>(data: DenseWithMaskData<'_, T>, x: &[T], y: &mut [T]) {
    T::spmv_dense_masked(data, x, y)
}

/// Computes sparse SpMV using const-generic Sliced ELLPACK (SELL-p).
#[inline(always)]
pub fn spmv_sellp<T: SimdOps, const C: usize>(data: SellPData<'_, T, C>, x: &[T], y: &mut [T]) {
    T::spmv_sellp::<C>(data, x, y)
}

/// Computes register-blocked tiled GEMM: `c += A * B`.
#[inline(always)]
pub fn tiled_gemm<T: SimdOps>(
    a: &[T],
    b: &[T],
    c: &mut [T],
    m: usize,
    n: usize,
    k: usize,
) -> Result<(), SimdError> {
    T::tiled_gemm(a, b, c, m, n, k)
}

/// Computes register-blocked GEMV `y += A · x` with runtime backend selection.
///
/// `a` is row-major `nrows × ncols`; the product **accumulates** into `y`
/// (zero `y` first for `y = A·x`). See [`gemv`](crate::dispatch::gemv) for the
/// operand-reuse theorem.
///
/// # Errors
/// [`SimdError::LengthMismatch`] if `a.len() < nrows·ncols`, `x.len() < ncols`,
/// or `y.len() < nrows`.
#[inline(always)]
pub fn gemv<T: SimdOps>(
    a: &[T],
    x: &[T],
    y: &mut [T],
    nrows: usize,
    ncols: usize,
) -> Result<(), SimdError> {
    T::gemv(a, x, y, nrows, ncols)
}

/// Computes register-blocked transposed GEMV `y += Aᵀ · x` with runtime backend
/// selection — the complement of [`gemv`].
///
/// `a` is row-major `nrows × ncols`, `x` length `nrows`, `y` length `ncols`; the
/// product **accumulates** into `y` (zero `y` first for `y = Aᵀ·x`). See
/// [`gemv_transpose`](crate::dispatch::gemv_transpose) for the operand-reuse theorem.
///
/// # Errors
/// [`SimdError::LengthMismatch`] if `a.len() < nrows·ncols`, `x.len() < nrows`,
/// or `y.len() < ncols`.
#[inline(always)]
pub fn gemv_transpose<T: SimdOps>(
    a: &[T],
    x: &[T],
    y: &mut [T],
    nrows: usize,
    ncols: usize,
) -> Result<(), SimdError> {
    T::gemv_transpose(a, x, y, nrows, ncols)
}

/// Multiplies interleaved complex values in-place using a monomorphized SIMD architecture.
///
/// Inputs are primitive lane slices in `[re0, im0, re1, im1, ...]` order. `a`
/// is updated with `a[i] * b[i]`; when `CONJ_B` is true, the operation is
/// `a[i] * conj(b[i])`.
#[inline]
pub fn interleaved_complex_mul_assign<T, A, const CONJ_B: bool>(
    a: &mut [T],
    b: &[T],
) -> Result<(), SimdError>
where
    T: ScalarTrait,
    A: hermes_simd_core::arch::SimdArch + hermes_simd_core::kernel::SimdKernel<T>,
{
    complex::interleaved_complex_mul_assign::<T, A, CONJ_B>(a, b)
}

/// Computes an interleaved complex dot product using a monomorphized SIMD architecture.
///
/// Inputs are primitive lane slices in `[re0, im0, re1, im1, ...]` order. The
/// result is `(re, im)` for `sum(a[i] * b[i])`; when `CONJ_B` is true, the
/// operation is `sum(a[i] * conj(b[i]))`.
#[inline]
pub fn interleaved_complex_dot<T, A, const CONJ_B: bool>(
    a: &[T],
    b: &[T],
) -> Result<(T, T), SimdError>
where
    T: ScalarTrait,
    A: hermes_simd_core::arch::SimdArch + hermes_simd_core::kernel::SimdKernel<T>,
{
    complex::interleaved_complex_dot::<T, A, CONJ_B>(a, b)
}

/// Multiplies interleaved complex values in-place using Hermes runtime provider selection.
#[inline]
pub fn interleaved_complex_mul_assign_runtime<T, const CONJ_B: bool>(
    a: &mut [T],
    b: &[T],
) -> Result<(), SimdError>
where
    T: SimdOps,
{
    T::interleaved_complex_mul_assign::<CONJ_B>(a, b)
}

/// Computes an interleaved complex dot product using Hermes runtime provider selection.
#[inline]
pub fn interleaved_complex_dot_runtime<T, const CONJ_B: bool>(
    a: &[T],
    b: &[T],
) -> Result<(T, T), SimdError>
where
    T: SimdOps,
{
    T::interleaved_complex_dot::<CONJ_B>(a, b)
}
