use super::{complex, modular, simd_ops::SimdOps};
use hermes_simd_core::scalar::Scalar as ScalarTrait;
use hermes_simd_core::sparse::{
    BlockedCooData, CsrData, DenseWithMaskData, SellPData, ValidatedData,
};
use hermes_simd_core::view::SimdError;

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
    T::scale(data, scalar);
}

/// Returns the first minimum, or `None` for empty or NaN-containing data.
#[inline(always)]
pub fn argmin<T: SimdOps>(data: &[T]) -> Option<(usize, T)> {
    T::argmin(data)
}

/// Returns the first maximum, or `None` for empty or NaN-containing data.
#[inline(always)]
pub fn argmax<T: SimdOps>(data: &[T]) -> Option<(usize, T)> {
    T::argmax(data)
}

/// Computes the dot product of two slices using runtime-dispatched SIMD.
///
/// # Errors
///
/// Returns [`SimdError::LengthMismatch`] when the slices have different
/// lengths.
#[inline(always)]
pub fn dot<T: SimdOps>(a: &[T], b: &[T]) -> Result<T, SimdError> {
    T::dot(a, b)
}

/// Fused row update `out[i] += alpha * x[i]` (AXPY) via runtime-dispatched
/// SIMD with no temporary allocation. Errors on length mismatch.
///
/// # Errors
///
/// Returns [`SimdError::LengthMismatch`] when `x` and `out` have different
/// lengths.
#[inline(always)]
pub fn axpy<T: SimdOps>(alpha: T, x: &[T], out: &mut [T]) -> Result<(), SimdError> {
    T::axpy(alpha, x, out)
}

/// Fused ternary update `out[i] += alpha * a[i] * b[i]` without a temporary.
///
/// # Errors
/// Returns [`SimdError::LengthMismatch`] when `a`, `b`, and `out` do not have
/// equal lengths.
#[inline(always)]
pub fn axpy_mul<T: SimdOps>(alpha: T, a: &[T], b: &[T], out: &mut [T]) -> Result<(), SimdError> {
    T::axpy_mul(alpha, a, b, out)
}

/// Fused multi-row update `out[row, i] += alphas[row] * x[i]` via one
/// runtime-dispatched SIMD kernel. `out` is a row-major strided window.
///
/// # Errors
///
/// Returns [`SimdError::LengthMismatch`] when `alphas` or `x` is shorter than
/// the requested shape, `row_stride < cols`, or `out` does not cover the
/// requested rows and columns.
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
///
/// # Errors
///
/// Returns [`SimdError::LengthMismatch`] when either input panel is shorter
/// than the requested shape, `row_stride < cols`, or `out` does not cover the
/// requested rows and columns.
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
///
/// # Errors
///
/// Returns [`SimdError::LengthMismatch`] when the input and output lengths do
/// not match.
#[inline(always)]
pub fn elementwise_mul<T: SimdOps>(a: &[T], b: &[T], out: &mut [T]) -> Result<(), SimdError> {
    T::elementwise_mul(a, b, out)
}

/// Computes the elementwise sum of two slices and writes to `out`.
///
/// # Errors
///
/// Returns [`SimdError::LengthMismatch`] when the input and output lengths do
/// not match.
#[inline(always)]
pub fn elementwise_add<T: SimdOps>(a: &[T], b: &[T], out: &mut [T]) -> Result<(), SimdError> {
    T::elementwise_add(a, b, out)
}

/// Computes the elementwise difference of two slices and writes to `out`.
///
/// # Errors
///
/// Returns [`SimdError::LengthMismatch`] when the input and output lengths do
/// not match.
#[inline(always)]
pub fn elementwise_sub<T: SimdOps>(a: &[T], b: &[T], out: &mut [T]) -> Result<(), SimdError> {
    T::elementwise_sub(a, b, out)
}

/// Computes the elementwise quotient of two slices and writes to `out`.
///
/// # Errors
///
/// Returns [`SimdError::LengthMismatch`] when the input and output lengths do
/// not match.
#[inline(always)]
pub fn elementwise_div<T: SimdOps>(a: &[T], b: &[T], out: &mut [T]) -> Result<(), SimdError> {
    T::elementwise_div(a, b, out)
}

/// Executes one exact modular radix-2 NTT butterfly stage over `u64` residues.
///
/// # Errors
///
/// Returns [`SimdError::LengthMismatch`] when the stage shape is invalid.
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
///
/// # Errors
///
/// Returns [`SimdError::LengthMismatch`] when `a`, `b`, and `mask` do not have
/// equal lengths.
#[inline(always)]
pub fn masked_dot<T: SimdOps>(a: &[T], b: &[T], mask: &[bool]) -> Result<T, SimdError> {
    T::masked_dot(a, b, mask)
}

/// Computes the elementwise sum of elements matching a boolean mask.
///
/// # Errors
///
/// Returns [`SimdError::LengthMismatch`] when `a`, `b`, `mask`, and `out` do
/// not have equal lengths.
#[inline(always)]
pub fn masked_add<T: SimdOps>(
    a: &[T],
    b: &[T],
    mask: &[bool],
    out: &mut [T],
) -> Result<(), SimdError> {
    T::masked_add(a, b, mask, out)
}

/// Computes sparse `SpMV` using CSR: `y += A · x`.
///
/// # Panics
/// Panics if `x.len() < ncols` or `y.len() < nrows`. Structural CSR validation
/// is performed by [`ValidatedData::new`] before this function can be called.
#[inline(always)]
pub fn spmv_csr<T: SimdOps>(data: ValidatedData<CsrData<'_, T>>, x: &[T], y: &mut [T]) {
    T::spmv_csr(data, x, y);
}

/// Computes sparse `SpMV` using const-generic Blocked-COO tiles.
///
/// # Panics
/// Panics if `x.len() < ncols` or `y.len() < nrows`. Structural Blocked-COO
/// validation is performed by [`ValidatedData::new`] before this function can be
/// called.
#[inline(always)]
pub fn spmv_bcoo<T: SimdOps, const BM: usize, const BN: usize>(
    data: ValidatedData<BlockedCooData<'_, T, BM, BN>>,
    x: &[T],
    y: &mut [T],
) {
    T::spmv_bcoo::<BM, BN>(data, x, y);
}

/// Computes sparse `SpMV` using Dense-with-Mask.
///
/// # Panics
/// Panics if `x.len() < ncols`, `y.len() < nrows`, the matrix dimensions
/// overflow, or the value and packed-mask lengths do not exactly match
/// `nrows * ncols`.
#[inline(always)]
pub fn spmv_dense_masked<T: SimdOps>(data: DenseWithMaskData<'_, T>, x: &[T], y: &mut [T]) {
    T::spmv_dense_masked(data, x, y);
}

/// Computes sparse `SpMV` using const-generic Sliced ELLPACK (SELL-p).
///
/// # Panics
/// Panics if `x.len() < ncols` or `y.len() < nrows`. Structural SELL-p
/// validation is performed by [`ValidatedData::new`] before this function can be
/// called.
#[inline(always)]
pub fn spmv_sellp<T: SimdOps, const C: usize>(
    data: ValidatedData<SellPData<'_, T, C>>,
    x: &[T],
    y: &mut [T],
) {
    T::spmv_sellp::<C>(data, x, y);
}

/// Computes register-blocked tiled GEMM: `c += A * B`.
///
/// # Errors
///
/// Returns [`SimdError::LengthMismatch`] when the operand buffers do not cover
/// the requested matrix dimensions.
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
/// (zero `y` first for `y = A·x`). See [`gemv()`] for the
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
/// selection — the complement of [`gemv()`].
///
/// `a` is row-major `nrows × ncols`, `x` length `nrows`, `y` length `ncols`; the
/// product **accumulates** into `y` (zero `y` first for `y = Aᵀ·x`). See
/// [`gemv_transpose()`] for the operand-reuse theorem.
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

/// Computes register-blocked sub-matrix GEMV `y += A · x` with row stride `lda`,
/// runtime backend selection. `A` is a row-major `nrows × ncols` block with
/// leading dimension `lda ≥ ncols`; `lda = ncols` is the packed [`gemv()`].
/// Accumulates into `y`.
///
/// # Errors
/// [`SimdError::LengthMismatch`] if `lda < ncols`, `a.len() < (nrows−1)·lda +
/// ncols`, `x.len() < ncols`, or `y.len() < nrows`.
#[inline(always)]
pub fn gemv_strided<T: SimdOps>(
    a: &[T],
    x: &[T],
    y: &mut [T],
    nrows: usize,
    ncols: usize,
    lda: usize,
) -> Result<(), SimdError> {
    T::gemv_strided(a, x, y, nrows, ncols, lda)
}

/// Computes register-blocked transposed sub-matrix GEMV `y += Aᵀ · x` with row
/// stride `lda`, runtime backend selection. `lda = ncols` is the packed
/// [`gemv_transpose()`]. Accumulates into `y`.
///
/// # Errors
/// [`SimdError::LengthMismatch`] if `lda < ncols`, `a.len() < (nrows−1)·lda +
/// ncols`, `x.len() < nrows`, or `y.len() < ncols`.
#[inline(always)]
pub fn gemv_transpose_strided<T: SimdOps>(
    a: &[T],
    x: &[T],
    y: &mut [T],
    nrows: usize,
    ncols: usize,
    lda: usize,
) -> Result<(), SimdError> {
    T::gemv_transpose_strided(a, x, y, nrows, ncols, lda)
}

/// Multiplies interleaved complex values in-place using a monomorphized SIMD architecture.
///
/// Inputs are primitive lane slices in `[re0, im0, re1, im1, ...]` order. `a`
/// is updated with `a[i] * b[i]`; when `CONJ_B` is true, the operation is
/// `a[i] * conj(b[i])`.
///
/// # Errors
///
/// Returns [`SimdError::LengthMismatch`] when the slices have different
/// lengths or their common length is odd.
#[inline]
pub fn interleaved_complex_mul_assign<T, A, const CONJ_B: bool>(
    a: &mut [T],
    b: &[T],
) -> Result<(), SimdError>
where
    T: ScalarTrait + core::ops::Neg<Output = T>,
    A: hermes_simd_core::arch::SimdArch
        + hermes_simd_core::kernel::SimdArith<T>
        + hermes_simd_core::kernel::SimdLoadStore<T>
        + hermes_simd_core::kernel::SimdPermute<T>,
{
    complex::interleaved_complex_mul_assign::<T, A, CONJ_B>(a, b)
}

/// Computes an interleaved complex dot product using a monomorphized SIMD architecture.
///
/// Inputs are primitive lane slices in `[re0, im0, re1, im1, ...]` order. The
/// result is `(re, im)` for `sum(a[i] * b[i])`; when `CONJ_B` is true, the
/// operation is `sum(a[i] * conj(b[i]))`.
///
/// # Errors
///
/// Returns [`SimdError::LengthMismatch`] when the slices have different
/// lengths or their common length is odd.
#[inline]
pub fn interleaved_complex_dot<T, A, const CONJ_B: bool>(
    a: &[T],
    b: &[T],
) -> Result<(T, T), SimdError>
where
    T: ScalarTrait + core::ops::Neg<Output = T>,
    A: hermes_simd_core::arch::SimdArch
        + hermes_simd_core::kernel::SimdArith<T>
        + hermes_simd_core::kernel::SimdLoadStore<T>
        + hermes_simd_core::kernel::SimdPermute<T>,
{
    complex::interleaved_complex_dot::<T, A, CONJ_B>(a, b)
}

/// Multiplies interleaved complex values in-place using Hermes runtime provider selection.
///
/// # Errors
///
/// Returns [`SimdError::LengthMismatch`] when the slices have different
/// lengths or their common length is odd.
#[inline]
pub fn interleaved_complex_mul_assign_runtime<T, const CONJ_B: bool>(
    a: &mut [T],
    b: &[T],
) -> Result<(), SimdError>
where
    T: SimdOps + core::ops::Neg<Output = T>,
{
    T::interleaved_complex_mul_assign::<CONJ_B>(a, b)
}

/// Computes an interleaved complex dot product using Hermes runtime provider selection.
///
/// # Errors
///
/// Returns [`SimdError::LengthMismatch`] when the slices have different
/// lengths or their common length is odd.
#[inline]
pub fn interleaved_complex_dot_runtime<T, const CONJ_B: bool>(
    a: &[T],
    b: &[T],
) -> Result<(T, T), SimdError>
where
    T: SimdOps + core::ops::Neg<Output = T>,
{
    T::interleaved_complex_dot::<CONJ_B>(a, b)
}

/// Computes a real-by-interleaved-complex dot product using a monomorphized
/// SIMD architecture.
///
/// # Errors
///
/// Returns [`SimdError::LengthMismatch`] unless `weights.len()` is exactly
/// twice `real.len()`.
#[inline]
pub fn real_interleaved_complex_dot<T, A>(real: &[T], weights: &[T]) -> Result<(T, T), SimdError>
where
    T: ScalarTrait,
    A: hermes_simd_core::arch::SimdArch
        + hermes_simd_core::kernel::SimdArith<T>
        + hermes_simd_core::kernel::SimdLoadStore<T>
        + hermes_simd_core::kernel::SimdPermute<T>,
{
    complex::real_interleaved_complex_dot::<T, A>(real, weights)
}

/// Computes a real-by-interleaved-complex dot product using Hermes runtime
/// provider selection.
///
/// # Errors
///
/// Returns [`SimdError::LengthMismatch`] unless `weights.len()` is exactly
/// twice `real.len()`.
#[inline]
pub fn real_interleaved_complex_dot_runtime<T>(
    real: &[T],
    weights: &[T],
) -> Result<(T, T), SimdError>
where
    T: SimdOps,
{
    T::real_interleaved_complex_dot(real, weights)
}

/// Computes the horizontal sum of population counts of all elements using runtime-dispatched SIMD.
#[inline(always)]
pub fn reduce_popcount<T: SimdOps>(data: &[T]) -> usize {
    T::reduce_popcount(data)
}

/// Computes the horizontal sum of population counts of `a[i] & b[i]` using runtime-dispatched SIMD.
///
/// # Errors
///
/// Returns [`SimdError::LengthMismatch`] when the slices have different
/// lengths.
#[inline(always)]
pub fn reduce_popcount_and<T: SimdOps>(a: &[T], b: &[T]) -> Result<usize, SimdError> {
    T::reduce_popcount_and(a, b)
}

/// Computes the horizontal sum of population counts of `a[i] | b[i]` using runtime-dispatched SIMD.
///
/// # Errors
///
/// Returns [`SimdError::LengthMismatch`] when the slices have different
/// lengths.
#[inline(always)]
pub fn reduce_popcount_or<T: SimdOps>(a: &[T], b: &[T]) -> Result<usize, SimdError> {
    T::reduce_popcount_or(a, b)
}

/// Computes the horizontal sum of population counts of `a[i] ^ b[i]` (Hamming distance) using runtime-dispatched SIMD.
///
/// # Errors
///
/// Returns [`SimdError::LengthMismatch`] when the slices have different
/// lengths.
#[inline(always)]
pub fn reduce_popcount_xor<T: SimdOps>(a: &[T], b: &[T]) -> Result<usize, SimdError> {
    T::reduce_popcount_xor(a, b)
}
