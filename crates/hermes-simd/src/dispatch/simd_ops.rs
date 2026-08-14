use super::popcount::{
    dispatch_reduce_popcount, dispatch_reduce_popcount_and, dispatch_reduce_popcount_or,
    dispatch_reduce_popcount_xor,
};
use super::{
    abs_reduce, argmax, argmin, axpy, binary, complex, dot, gemm, gemv, gemv_strided,
    gemv_transpose, gemv_transpose_strided, masked, max, min, scale, sparse, sum,
};
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

mod private {
    pub trait Sealed {}
}

impl private::Sealed for f32 {}
impl private::Sealed for f64 {}
impl private::Sealed for i8 {}
impl private::Sealed for i16 {}
impl private::Sealed for i32 {}

impl private::Sealed for eunomia::F16 {}
impl private::Sealed for eunomia::F32 {}
impl private::Sealed for eunomia::F64 {}
impl private::Sealed for eunomia::Bf16 {}
impl private::Sealed for eunomia::Bf8 {}
impl private::Sealed for eunomia::Bf4 {}
impl private::Sealed for eunomia::F8 {}
impl private::Sealed for eunomia::F4 {}
impl private::Sealed for eunomia::I8 {}
impl private::Sealed for eunomia::I16 {}
impl private::Sealed for eunomia::I32 {}

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
    ///
    /// # Errors
    ///
    /// Returns [`SimdError::LengthMismatch`] when the slices have different
    /// lengths.
    fn dot(a: &[Self], b: &[Self]) -> Result<Self, SimdError>;
    /// Fused row update `out[i] += alpha * x[i]` (AXPY) with no temporaries.
    ///
    /// # Errors
    ///
    /// Returns [`SimdError::LengthMismatch`] when `x` and `out` have different
    /// lengths.
    fn axpy(alpha: Self, x: &[Self], out: &mut [Self]) -> Result<(), SimdError>;
    /// Fused ternary update `out[i] += alpha * a[i] * b[i]` with no temporary.
    ///
    /// # Errors
    ///
    /// Returns [`SimdError::LengthMismatch`] when the input and output lengths
    /// do not match.
    fn axpy_mul(alpha: Self, a: &[Self], b: &[Self], out: &mut [Self]) -> Result<(), SimdError>;
    /// Fused multi-row update `out[row, i] += alphas[row] * x[i]`.
    ///
    /// # Errors
    ///
    /// Returns [`SimdError::LengthMismatch`] when an input is shorter than the
    /// requested shape, `row_stride < cols`, or `out` does not cover the rows.
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
    ///
    /// # Errors
    ///
    /// Returns [`SimdError::LengthMismatch`] when an input panel is shorter
    /// than the requested shape, `row_stride < cols`, or `out` does not cover
    /// the rows.
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
    ///
    /// # Errors
    ///
    /// Returns [`SimdError::LengthMismatch`] when the input and output lengths
    /// do not match.
    fn elementwise_mul(a: &[Self], b: &[Self], out: &mut [Self]) -> Result<(), SimdError>;
    /// Computes the elementwise sum `a[i] + b[i]` and writes to `out`.
    ///
    /// # Errors
    ///
    /// Returns [`SimdError::LengthMismatch`] when the input and output lengths
    /// do not match.
    fn elementwise_add(a: &[Self], b: &[Self], out: &mut [Self]) -> Result<(), SimdError>;
    /// Computes the elementwise difference `a[i] - b[i]` and writes to `out`.
    ///
    /// # Errors
    ///
    /// Returns [`SimdError::LengthMismatch`] when the input and output lengths
    /// do not match.
    fn elementwise_sub(a: &[Self], b: &[Self], out: &mut [Self]) -> Result<(), SimdError>;
    /// Computes the elementwise quotient `a[i] / b[i]` and writes to `out`.
    ///
    /// # Errors
    ///
    /// Returns [`SimdError::LengthMismatch`] when the input and output lengths
    /// do not match.
    fn elementwise_div(a: &[Self], b: &[Self], out: &mut [Self]) -> Result<(), SimdError>;
    /// Computes the sum of elements matching a boolean mask.
    fn masked_sum(data: &[Self], mask: &[bool]) -> Self;
    /// Computes the dot product of elements matching a boolean mask.
    ///
    /// # Errors
    ///
    /// Returns [`SimdError::LengthMismatch`] when `a`, `b`, and `mask` do not
    /// have equal lengths.
    fn masked_dot(a: &[Self], b: &[Self], mask: &[bool]) -> Result<Self, SimdError>;
    /// Computes the elementwise sum of elements matching a boolean mask.
    ///
    /// # Errors
    ///
    /// Returns [`SimdError::LengthMismatch`] when `a`, `b`, `mask`, and `out`
    /// do not have equal lengths.
    fn masked_add(a: &[Self], b: &[Self], mask: &[bool], out: &mut [Self])
        -> Result<(), SimdError>;
    /// Computes sparse SpMV using CSR.
    fn spmv_csr(data: ValidatedData<CsrData<'_, Self>>, x: &[Self], y: &mut [Self]);
    /// Computes sparse SpMV using const-generic Blocked-COO tiles.
    fn spmv_bcoo<const BM: usize, const BN: usize>(
        data: ValidatedData<BlockedCooData<'_, Self, BM, BN>>,
        x: &[Self],
        y: &mut [Self],
    );
    /// Computes sparse SpMV using Dense-with-Mask.
    fn spmv_dense_masked(data: DenseWithMaskData<'_, Self>, x: &[Self], y: &mut [Self]);
    /// Computes sparse SpMV using const-generic Sliced ELLPACK (SELL-p).
    fn spmv_sellp<const C: usize>(
        data: ValidatedData<SellPData<'_, Self, C>>,
        x: &[Self],
        y: &mut [Self],
    );
    /// Computes register-blocked tiled GEMM: `c += A * B`.
    ///
    /// # Errors
    ///
    /// Returns [`SimdError::LengthMismatch`] when the operand buffers do not
    /// cover the requested matrix dimensions.
    fn tiled_gemm(
        a: &[Self],
        b: &[Self],
        c: &mut [Self],
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<(), SimdError>;
    /// Computes register-blocked GEMV: `y += A * x` (`A` row-major `nrows × ncols`).
    ///
    /// # Errors
    ///
    /// Returns [`SimdError::LengthMismatch`] when an operand is shorter than
    /// the requested matrix or vector shape.
    fn gemv(
        a: &[Self],
        x: &[Self],
        y: &mut [Self],
        nrows: usize,
        ncols: usize,
    ) -> Result<(), SimdError>;
    /// Computes register-blocked transposed GEMV: `y += Aᵀ * x`
    /// (`A` row-major `nrows × ncols`, `x` length `nrows`, `y` length `ncols`).
    ///
    /// # Errors
    ///
    /// Returns [`SimdError::LengthMismatch`] when an operand is shorter than
    /// the requested matrix or vector shape.
    fn gemv_transpose(
        a: &[Self],
        x: &[Self],
        y: &mut [Self],
        nrows: usize,
        ncols: usize,
    ) -> Result<(), SimdError>;
    /// Computes register-blocked sub-matrix GEMV: `y += A * x` with row stride
    /// `lda ≥ ncols` (`lda = ncols` is the packed [`Self::gemv`]).
    ///
    /// # Errors
    ///
    /// Returns [`SimdError::LengthMismatch`] when `lda < ncols` or an operand
    /// is shorter than the requested strided shape.
    fn gemv_strided(
        a: &[Self],
        x: &[Self],
        y: &mut [Self],
        nrows: usize,
        ncols: usize,
        lda: usize,
    ) -> Result<(), SimdError>;
    /// Computes register-blocked transposed sub-matrix GEMV: `y += Aᵀ * x` with
    /// row stride `lda ≥ ncols` (`lda = ncols` is the packed [`Self::gemv_transpose`]).
    ///
    /// # Errors
    ///
    /// Returns [`SimdError::LengthMismatch`] when `lda < ncols` or an operand
    /// is shorter than the requested strided shape.
    fn gemv_transpose_strided(
        a: &[Self],
        x: &[Self],
        y: &mut [Self],
        nrows: usize,
        ncols: usize,
        lda: usize,
    ) -> Result<(), SimdError>;
    /// Multiplies interleaved complex lanes in-place: `a[k] *= b[k]`
    /// (`a[k] *= conj(b[k])` when `CONJ_B`).
    ///
    /// # Errors
    ///
    /// Returns [`SimdError::LengthMismatch`] when the slices have different
    /// lengths or their common length is odd.
    fn interleaved_complex_mul_assign<const CONJ_B: bool>(
        a: &mut [Self],
        b: &[Self],
    ) -> Result<(), SimdError>
    where
        Self: core::ops::Neg<Output = Self>;
    /// Computes the interleaved complex dot product `(re, im)` of `sum(a[k] * b[k])`
    /// (`sum(a[k] * conj(b[k]))` when `CONJ_B`).
    ///
    /// # Errors
    ///
    /// Returns [`SimdError::LengthMismatch`] when the slices have different
    /// lengths or their common length is odd.
    fn interleaved_complex_dot<const CONJ_B: bool>(
        a: &[Self],
        b: &[Self],
    ) -> Result<(Self, Self), SimdError>
    where
        Self: core::ops::Neg<Output = Self>;
    /// Computes the horizontal sum of population counts of all elements.
    fn reduce_popcount(data: &[Self]) -> usize;
    /// Computes the horizontal sum of population counts of `a[i] & b[i]`.
    ///
    /// # Errors
    ///
    /// Returns [`SimdError::LengthMismatch`] when the slices have different
    /// lengths.
    fn reduce_popcount_and(a: &[Self], b: &[Self]) -> Result<usize, SimdError>;
    /// Computes the horizontal sum of population counts of `a[i] | b[i]`.
    ///
    /// # Errors
    ///
    /// Returns [`SimdError::LengthMismatch`] when the slices have different
    /// lengths.
    fn reduce_popcount_or(a: &[Self], b: &[Self]) -> Result<usize, SimdError>;
    /// Computes the horizontal sum of population counts of `a[i] ^ b[i]` (Hamming distance).
    ///
    /// # Errors
    ///
    /// Returns [`SimdError::LengthMismatch`] when the slices have different
    /// lengths.
    fn reduce_popcount_xor(a: &[Self], b: &[Self]) -> Result<usize, SimdError>;
}

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
