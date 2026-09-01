use hermes_simd_core::scalar::Scalar as ScalarTrait;
use hermes_simd_core::sparse::{
    BlockedCooData, CsrData, DenseWithMaskData, SellPData, ValidatedData,
};
use hermes_simd_core::view::SimdError;

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
    /// Computes sparse `SpMV` using CSR.
    fn spmv_csr(data: ValidatedData<CsrData<'_, Self>>, x: &[Self], y: &mut [Self]);
    /// Computes sparse `SpMV` using const-generic Blocked-COO tiles.
    fn spmv_bcoo<const BM: usize, const BN: usize>(
        data: ValidatedData<BlockedCooData<'_, Self, BM, BN>>,
        x: &[Self],
        y: &mut [Self],
    );
    /// Computes sparse `SpMV` using Dense-with-Mask.
    fn spmv_dense_masked(data: DenseWithMaskData<'_, Self>, x: &[Self], y: &mut [Self]);
    /// Computes sparse `SpMV` using const-generic Sliced ELLPACK (SELL-p).
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
    /// Computes `sum(real[k] * weights[k])` for interleaved complex weights.
    ///
    /// # Errors
    ///
    /// Returns [`SimdError::LengthMismatch`] unless `weights.len()` is exactly
    /// twice `real.len()`.
    fn real_interleaved_complex_dot(
        real: &[Self],
        weights: &[Self],
    ) -> Result<(Self, Self), SimdError>;
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

mod blanket_impls;
