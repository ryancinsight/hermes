//! Runtime-dispatched SIMD operations.
//!
//! # Monomorphization chain
//!
//! `sum::<f32>(data)` -> `f32::sum(data)` -> `sum::dispatch_sum::<f32>(data)` -> avx2 kernel.

pub mod sum;
pub mod dot;
pub mod binary;
pub mod masked;
pub mod sparse;
pub mod gemm;
pub mod min;
pub mod max;
pub mod scale;
pub mod argmin;
pub mod argmax;
pub mod complex;

use hermes_simd_core::view::SimdError;
use hermes_simd_core::sparse::{
    CsrData, BlockedCooData, DenseWithMaskData, SellPData,
    SparseView, BlockedCoo, SparseSpMv,
};
use hermes_simd_core::scalar::Scalar as ScalarTrait;
use hermes_simd_core::{Add, Sub, Mul, Div};
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[allow(unused_imports)]
use hermes_simd_intrinsics::{Scalar as ScalarArch, Avx2, Avx512, Neon};
#[cfg(target_arch = "aarch64")]
use hermes_simd_intrinsics::{Scalar as ScalarArch, Neon};
#[cfg(not(any(target_arch = "x86", target_arch = "x86_64", target_arch = "aarch64")))]
use hermes_simd_intrinsics::Scalar as ScalarArch;

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
    fn masked_add(a: &[Self], b: &[Self], mask: &[bool], out: &mut [Self]) -> Result<(), SimdError>;
    /// Computes sparse SpMV using CSR.
    fn spmv_csr(data: CsrData<'_, Self>, x: &[Self], y: &mut [Self]);
    /// Computes sparse SpMV using Blocked-COO 4x4.
    fn spmv_bcoo4x4(data: BlockedCooData<'_, Self, 4, 4>, x: &[Self], y: &mut [Self]);
    /// Computes sparse SpMV using Blocked-COO 8x8.
    fn spmv_bcoo8x8(data: BlockedCooData<'_, Self, 8, 8>, x: &[Self], y: &mut [Self]);
    /// Computes sparse SpMV using Dense-with-Mask.
    fn spmv_dense_masked(data: DenseWithMaskData<'_, Self>, x: &[Self], y: &mut [Self]);
    /// Computes sparse SpMV using Sliced ELLPACK (SELL-p) with C = 4.
    fn spmv_sellp4(data: SellPData<'_, Self, 4>, x: &[Self], y: &mut [Self]);
    /// Computes sparse SpMV using Sliced ELLPACK (SELL-p) with C = 8.
    fn spmv_sellp8(data: SellPData<'_, Self, 8>, x: &[Self], y: &mut [Self]);
    /// Computes register-blocked tiled GEMM: `c += A * B`.
    fn tiled_gemm(a: &[Self], b: &[Self], c: &mut [Self], m: usize, n: usize, k: usize) -> Result<(), SimdError>;
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
    fn sum(data: &[Self]) -> Self { sum::dispatch_sum::<Self>(data) }
    #[inline(always)]
    fn min(data: &[Self]) -> Self { min::dispatch_min::<Self>(data) }
    #[inline(always)]
    fn max(data: &[Self]) -> Self { max::dispatch_max::<Self>(data) }
    #[inline(always)]
    fn scale(data: &mut [Self], scalar: Self) { scale::dispatch_scale::<Self>(data, scalar) }
    #[inline(always)]
    fn argmin(data: &[Self]) -> Option<(usize, Self)> { argmin::dispatch_argmin::<Self>(data) }
    #[inline(always)]
    fn argmax(data: &[Self]) -> Option<(usize, Self)> { argmax::dispatch_argmax::<Self>(data) }
    #[inline(always)]
    fn dot(a: &[Self], b: &[Self]) -> Result<Self, SimdError> { dot::dispatch_dot::<Self>(a, b) }
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
    fn masked_sum(data: &[Self], mask: &[bool]) -> Self { masked::dispatch_masked_sum::<Self>(data, mask) }
    #[inline(always)]
    fn masked_dot(a: &[Self], b: &[Self], mask: &[bool]) -> Result<Self, SimdError> {
        masked::dispatch_masked_dot::<Self>(a, b, mask)
    }
    #[inline(always)]
    fn masked_add(a: &[Self], b: &[Self], mask: &[bool], out: &mut [Self]) -> Result<(), SimdError> {
        masked::dispatch_masked_add::<Self>(a, b, mask, out)
    }
    #[inline(always)]
    fn spmv_csr(data: CsrData<'_, Self>, x: &[Self], y: &mut [Self]) {
        sparse::dispatch_spmv_csr::<Self>(data, x, y)
    }
    #[inline(always)]
    fn spmv_bcoo4x4(data: BlockedCooData<'_, Self, 4, 4>, x: &[Self], y: &mut [Self]) {
        SparseView::<Self, BlockedCoo<4, 4>, ScalarArch>::from_blocked_coo_4x4(data).spmv(x, y);
    }
    #[inline(always)]
    fn spmv_bcoo8x8(data: BlockedCooData<'_, Self, 8, 8>, x: &[Self], y: &mut [Self]) {
        SparseView::<Self, BlockedCoo<8, 8>, ScalarArch>::from_blocked_coo_8x8(data).spmv(x, y);
    }
    #[inline(always)]
    fn spmv_dense_masked(data: DenseWithMaskData<'_, Self>, x: &[Self], y: &mut [Self]) {
        sparse::dispatch_spmv_dense_masked::<Self>(data, x, y)
    }
    #[inline(always)]
    fn spmv_sellp4(data: SellPData<'_, Self, 4>, x: &[Self], y: &mut [Self]) {
        sparse::dispatch_spmv_sellp4::<Self>(data, x, y)
    }
    #[inline(always)]
    fn spmv_sellp8(data: SellPData<'_, Self, 8>, x: &[Self], y: &mut [Self]) {
        sparse::dispatch_spmv_sellp8::<Self>(data, x, y)
    }
    #[inline(always)]
    fn tiled_gemm(a: &[Self], b: &[Self], c: &mut [Self], m: usize, n: usize, k: usize) -> Result<(), SimdError> {
        gemm::dispatch_tiled_gemm::<Self>(a, b, c, m, n, k)
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
    fn sum(data: &[Self]) -> Self { sum::dispatch_sum::<Self>(data) }
    #[inline(always)]
    fn min(data: &[Self]) -> Self { min::dispatch_min::<Self>(data) }
    #[inline(always)]
    fn max(data: &[Self]) -> Self { max::dispatch_max::<Self>(data) }
    #[inline(always)]
    fn scale(data: &mut [Self], scalar: Self) { scale::dispatch_scale::<Self>(data, scalar) }
    #[inline(always)]
    fn argmin(data: &[Self]) -> Option<(usize, Self)> { argmin::dispatch_argmin::<Self>(data) }
    #[inline(always)]
    fn argmax(data: &[Self]) -> Option<(usize, Self)> { argmax::dispatch_argmax::<Self>(data) }
    #[inline(always)]
    fn dot(a: &[Self], b: &[Self]) -> Result<Self, SimdError> { dot::dispatch_dot::<Self>(a, b) }
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
    fn masked_sum(data: &[Self], mask: &[bool]) -> Self { masked::dispatch_masked_sum::<Self>(data, mask) }
    #[inline(always)]
    fn masked_dot(a: &[Self], b: &[Self], mask: &[bool]) -> Result<Self, SimdError> {
        masked::dispatch_masked_dot::<Self>(a, b, mask)
    }
    #[inline(always)]
    fn masked_add(a: &[Self], b: &[Self], mask: &[bool], out: &mut [Self]) -> Result<(), SimdError> {
        masked::dispatch_masked_add::<Self>(a, b, mask, out)
    }
    #[inline(always)]
    fn spmv_csr(data: CsrData<'_, Self>, x: &[Self], y: &mut [Self]) {
        sparse::dispatch_spmv_csr::<Self>(data, x, y)
    }
    #[inline(always)]
    fn spmv_bcoo4x4(data: BlockedCooData<'_, Self, 4, 4>, x: &[Self], y: &mut [Self]) {
        SparseView::<Self, BlockedCoo<4, 4>, ScalarArch>::from_blocked_coo_4x4(data).spmv(x, y);
    }
    #[inline(always)]
    fn spmv_bcoo8x8(data: BlockedCooData<'_, Self, 8, 8>, x: &[Self], y: &mut [Self]) {
        SparseView::<Self, BlockedCoo<8, 8>, ScalarArch>::from_blocked_coo_8x8(data).spmv(x, y);
    }
    #[inline(always)]
    fn spmv_dense_masked(data: DenseWithMaskData<'_, Self>, x: &[Self], y: &mut [Self]) {
        sparse::dispatch_spmv_dense_masked::<Self>(data, x, y)
    }
    #[inline(always)]
    fn spmv_sellp4(data: SellPData<'_, Self, 4>, x: &[Self], y: &mut [Self]) {
        sparse::dispatch_spmv_sellp4::<Self>(data, x, y)
    }
    #[inline(always)]
    fn spmv_sellp8(data: SellPData<'_, Self, 8>, x: &[Self], y: &mut [Self]) {
        sparse::dispatch_spmv_sellp8::<Self>(data, x, y)
    }
    #[inline(always)]
    fn tiled_gemm(a: &[Self], b: &[Self], c: &mut [Self], m: usize, n: usize, k: usize) -> Result<(), SimdError> {
        gemm::dispatch_tiled_gemm::<Self>(a, b, c, m, n, k)
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
    fn sum(data: &[Self]) -> Self { sum::dispatch_sum::<Self>(data) }
    #[inline(always)]
    fn min(data: &[Self]) -> Self { min::dispatch_min::<Self>(data) }
    #[inline(always)]
    fn max(data: &[Self]) -> Self { max::dispatch_max::<Self>(data) }
    #[inline(always)]
    fn scale(data: &mut [Self], scalar: Self) { scale::dispatch_scale::<Self>(data, scalar) }
    #[inline(always)]
    fn argmin(data: &[Self]) -> Option<(usize, Self)> { argmin::dispatch_argmin::<Self>(data) }
    #[inline(always)]
    fn argmax(data: &[Self]) -> Option<(usize, Self)> { argmax::dispatch_argmax::<Self>(data) }
    #[inline(always)]
    fn dot(a: &[Self], b: &[Self]) -> Result<Self, SimdError> { dot::dispatch_dot::<Self>(a, b) }
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
    fn masked_sum(data: &[Self], mask: &[bool]) -> Self { masked::dispatch_masked_sum::<Self>(data, mask) }
    #[inline(always)]
    fn masked_dot(a: &[Self], b: &[Self], mask: &[bool]) -> Result<Self, SimdError> {
        masked::dispatch_masked_dot::<Self>(a, b, mask)
    }
    #[inline(always)]
    fn masked_add(a: &[Self], b: &[Self], mask: &[bool], out: &mut [Self]) -> Result<(), SimdError> {
        masked::dispatch_masked_add::<Self>(a, b, mask, out)
    }
    #[inline(always)]
    fn spmv_csr(data: CsrData<'_, Self>, x: &[Self], y: &mut [Self]) {
        sparse::dispatch_spmv_csr::<Self>(data, x, y)
    }
    #[inline(always)]
    fn spmv_bcoo4x4(data: BlockedCooData<'_, Self, 4, 4>, x: &[Self], y: &mut [Self]) {
        SparseView::<Self, BlockedCoo<4, 4>, ScalarArch>::from_blocked_coo_4x4(data).spmv(x, y);
    }
    #[inline(always)]
    fn spmv_bcoo8x8(data: BlockedCooData<'_, Self, 8, 8>, x: &[Self], y: &mut [Self]) {
        SparseView::<Self, BlockedCoo<8, 8>, ScalarArch>::from_blocked_coo_8x8(data).spmv(x, y);
    }
    #[inline(always)]
    fn spmv_dense_masked(data: DenseWithMaskData<'_, Self>, x: &[Self], y: &mut [Self]) {
        sparse::dispatch_spmv_dense_masked::<Self>(data, x, y)
    }
    #[inline(always)]
    fn spmv_sellp4(data: SellPData<'_, Self, 4>, x: &[Self], y: &mut [Self]) {
        sparse::dispatch_spmv_sellp4::<Self>(data, x, y)
    }
    #[inline(always)]
    fn spmv_sellp8(data: SellPData<'_, Self, 8>, x: &[Self], y: &mut [Self]) {
        sparse::dispatch_spmv_sellp8::<Self>(data, x, y)
    }
    #[inline(always)]
    fn tiled_gemm(a: &[Self], b: &[Self], c: &mut [Self], m: usize, n: usize, k: usize) -> Result<(), SimdError> {
        gemm::dispatch_tiled_gemm::<Self>(a, b, c, m, n, k)
    }
}

/// Computes the sum of elements in the slice using runtime-dispatched SIMD.
#[inline(always)]
pub fn sum<T: SimdOps>(data: &[T]) -> T { T::sum(data) }

/// Computes the minimum element of the slice using runtime-dispatched SIMD.
///
/// Returns `T::MAX_VALUE` for empty slices.
#[inline(always)]
pub fn min<T: SimdOps>(data: &[T]) -> T { T::min(data) }

/// Computes the maximum element of the slice using runtime-dispatched SIMD.
///
/// Returns `T::MIN_VALUE` for empty slices.
#[inline(always)]
pub fn max<T: SimdOps>(data: &[T]) -> T { T::max(data) }

/// Multiplies every element of `data` by `scalar` in-place.
#[inline(always)]
pub fn scale<T: SimdOps>(data: &mut [T], scalar: T) { T::scale(data, scalar) }

/// Returns `Some((index, value))` of the minimum element, or `None` for empty.
#[inline(always)]
pub fn argmin<T: SimdOps>(data: &[T]) -> Option<(usize, T)> { T::argmin(data) }

/// Returns `Some((index, value))` of the maximum element, or `None` for empty.
#[inline(always)]
pub fn argmax<T: SimdOps>(data: &[T]) -> Option<(usize, T)> { T::argmax(data) }

/// Computes the dot product of two slices using runtime-dispatched SIMD.
#[inline(always)]
pub fn dot<T: SimdOps>(a: &[T], b: &[T]) -> Result<T, SimdError> { T::dot(a, b) }

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

/// Computes the sum of elements matching a boolean mask.
#[inline(always)]
pub fn masked_sum<T: SimdOps>(data: &[T], mask: &[bool]) -> T { T::masked_sum(data, mask) }

/// Computes the dot product of elements matching a boolean mask.
#[inline(always)]
pub fn masked_dot<T: SimdOps>(a: &[T], b: &[T], mask: &[bool]) -> Result<T, SimdError> {
    T::masked_dot(a, b, mask)
}

/// Computes the elementwise sum of elements matching a boolean mask.
#[inline(always)]
pub fn masked_add<T: SimdOps>(a: &[T], b: &[T], mask: &[bool], out: &mut [T]) -> Result<(), SimdError> {
    T::masked_add(a, b, mask, out)
}

/// Computes sparse SpMV using CSR.
#[inline(always)]
pub fn spmv_csr<T: SimdOps>(data: CsrData<'_, T>, x: &[T], y: &mut [T]) {
    T::spmv_csr(data, x, y)
}

/// Computes sparse SpMV using Blocked-COO 4x4.
#[inline(always)]
pub fn spmv_bcoo4x4<T: SimdOps>(data: BlockedCooData<'_, T, 4, 4>, x: &[T], y: &mut [T]) {
    T::spmv_bcoo4x4(data, x, y)
}

/// Computes sparse SpMV using Blocked-COO 8x8.
#[inline(always)]
pub fn spmv_bcoo8x8<T: SimdOps>(data: BlockedCooData<'_, T, 8, 8>, x: &[T], y: &mut [T]) {
    T::spmv_bcoo8x8(data, x, y)
}

/// Computes sparse SpMV using Dense-with-Mask.
#[inline(always)]
pub fn spmv_dense_masked<T: SimdOps>(data: DenseWithMaskData<'_, T>, x: &[T], y: &mut [T]) {
    T::spmv_dense_masked(data, x, y)
}

/// Computes sparse SpMV using Sliced ELLPACK (SELL-p) with C = 4.
#[inline(always)]
pub fn spmv_sellp4<T: SimdOps>(data: SellPData<'_, T, 4>, x: &[T], y: &mut [T]) {
    T::spmv_sellp4(data, x, y)
}

/// Computes sparse SpMV using Sliced ELLPACK (SELL-p) with C = 8.
#[inline(always)]
pub fn spmv_sellp8<T: SimdOps>(data: SellPData<'_, T, 8>, x: &[T], y: &mut [T]) {
    T::spmv_sellp8(data, x, y)
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

/// Multiplies interleaved complex values in-place using Hermes runtime provider selection.
#[inline]
pub fn interleaved_complex_mul_assign_runtime<T, const CONJ_B: bool>(
    a: &mut [T],
    b: &[T],
) -> Result<(), SimdError>
where
    T: complex::InterleavedComplexLane,
{
    complex::interleaved_complex_mul_assign_runtime::<T, CONJ_B>(a, b)
}
