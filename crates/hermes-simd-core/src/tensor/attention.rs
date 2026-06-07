//! Scaled dot-product attention (SDPA).
//!
//! # Definition
//!
//! `Attention(Q, K, V) = softmax(Q·Kᵀ / √d_k) · V`
//!
//! - `Q` — query matrix, shape `[seq_q, d_k]`
//! - `K` — key matrix,   shape `[seq_k, d_k]`
//! - `V` — value matrix, shape `[seq_k, d_v]`
//! - Output shape: `[seq_q, d_v]`
//!
//! # Algorithm
//!
//! 1. `S = Q·Kᵀ` via tiled GEMM — shape `[seq_q, seq_k]`.
//! 2. Scale every element: `S[i][j] /= √d_k` — SIMD broadcast-scalar op.
//! 3. Row-wise softmax over `S`              — existing `softmax_inplace`.
//! 4. `Out = S·V` via tiled GEMM             — shape `[seq_q, d_v]`.
//!
//! # Zero-Allocation Contract
//!
//! `attention` allocates exactly two `AlignedVec`s:
//! - One for the `seq_q × seq_k` attention score matrix `S`.
//! - One for the `seq_q × d_v` output matrix.
//!
//! No intermediate per-row buffers are heap-allocated.
//!
//! # Monomorphization
//!
//! Parameterized over `(T, Arch, Align, TILE_M, TILE_N)`. All specializations
//! monomorphize to direct intrinsic sequences with zero virtual dispatch.

extern crate alloc;

use crate::align::{Alignment, Unaligned};
use crate::arch::SimdArch;
use crate::execution::Unmasked;
use crate::kernel::SimdKernel;
use crate::scalar::{Scalar, FloatElement};
use crate::tensor::softmax::softmax_inplace;
use crate::tiling::{TilingPolicy, TilingStrategy};
use crate::vec::AlignedVec;
use crate::view::{SimdError, SimdView};
use super::TensorView;

/// Scaled dot-product attention.
///
/// Computes `softmax(Q·Kᵀ / √d_k) · V`.
///
/// # Type Parameters
/// - `T`      — scalar element type (must be `FloatElement` for softmax).
/// - `Arch`   — SIMD architecture ZST marker.
/// - `Align`  — alignment typestate for output buffers.
/// - `TILE_M` — tiling row count; 4 is safe for AVX2, 8 for AVX-512.
/// - `TILE_N` — tiling column-vector count.
///
/// # Errors
/// - [`SimdError::LengthMismatch`] if `Q.shape[1] != K.shape[1]` (d_k mismatch),
///   `K.shape[0] != V.shape[0]` (seq_k mismatch), or any inner dimension is zero.
///
/// # Returns
/// An owned `AlignedVec<T, Align>` of length `seq_q * d_v` in row-major order.
#[inline]
pub fn attention<T, Arch, Align, const TILE_M: usize, const TILE_N: usize>(
    q: &TensorView<'_, T, 2>,
    k: &TensorView<'_, T, 2>,
    v: &TensorView<'_, T, 2>,
) -> Result<AlignedVec<T, Align>, SimdError>
where
    T: Scalar + FloatElement,
    Arch: SimdArch + SimdKernel<T>,
    Align: Alignment,
{
    let [seq_q, d_k] = q.shape();
    let [seq_k, d_k2] = k.shape();
    let [seq_k2, d_v] = v.shape();

    if d_k != d_k2 || seq_k != seq_k2 {
        return Err(SimdError::LengthMismatch);
    }
    if seq_q == 0 || d_k == 0 || seq_k == 0 || d_v == 0 {
        let out: AlignedVec<T, Align> = AlignedVec::with_capacity(0);
        return Ok(out);
    }

    // --- Step 1: S = Q · Kᵀ  (seq_q × seq_k) ---
    // We need Kᵀ: shape [d_k, seq_k]. Transpose K into a temporary buffer.
    let mut kt_buf: AlignedVec<T, Unaligned> = AlignedVec::with_capacity(seq_k * d_k);
    unsafe { kt_buf.set_len(seq_k * d_k); }
    transpose_into(k.as_slice(), seq_k, d_k, kt_buf.as_mut_slice());

    let kt_view = TensorView::<T, 2>::new(kt_buf.as_slice(), [d_k, seq_k])
        .map_err(|_| SimdError::LengthMismatch)?;

    let mut scores: AlignedVec<T, Align> = AlignedVec::with_capacity(seq_q * seq_k);
    unsafe { scores.set_len(seq_q * seq_k); }
    for x in scores.as_mut_slice().iter_mut() { *x = T::ZERO; }

    {
        let q_simd = SimdView::<'_, T, Arch, Unaligned, Unmasked, &[T]>::new(q.as_slice())
            .ok_or(SimdError::LengthMismatch)?;
        let kt_simd = SimdView::<'_, T, Arch, Unaligned, Unmasked, &[T]>::new(kt_view.as_slice())
            .ok_or(SimdError::LengthMismatch)?;

        <TilingPolicy<TILE_M, TILE_N> as TilingStrategy<T, Arch, Unaligned>>::gemm(
            &q_simd,
            &kt_simd,
            scores.as_mut_slice(),
            seq_q,
            seq_k,
            d_k,
        )?;
    }

    // --- Step 2: scale by 1/√d_k ---
    let inv_sqrt_dk = T::from_f32(1.0 / (d_k as f32).sqrt());
    for x in scores.as_mut_slice().iter_mut() {
        *x = *x * inv_sqrt_dk;
    }

    // --- Step 3: row-wise softmax over S ---
    let scores_slice = scores.as_mut_slice();
    for row in scores_slice.chunks_mut(seq_k) {
        softmax_inplace::<T, Arch>(row);
    }

    // --- Step 4: Out = S · V  (seq_q × d_v) ---
    let scores_view = TensorView::<T, 2>::new(scores.as_slice(), [seq_q, seq_k])
        .map_err(|_| SimdError::LengthMismatch)?;

    let mut out: AlignedVec<T, Align> = AlignedVec::with_capacity(seq_q * d_v);
    unsafe { out.set_len(seq_q * d_v); }
    for x in out.as_mut_slice().iter_mut() { *x = T::ZERO; }

    {
        let s_simd = SimdView::<'_, T, Arch, Unaligned, Unmasked, &[T]>::new(scores_view.as_slice())
            .ok_or(SimdError::LengthMismatch)?;
        let v_simd = SimdView::<'_, T, Arch, Unaligned, Unmasked, &[T]>::new(v.as_slice())
            .ok_or(SimdError::LengthMismatch)?;

        <TilingPolicy<TILE_M, TILE_N> as TilingStrategy<T, Arch, Unaligned>>::gemm(
            &s_simd,
            &v_simd,
            out.as_mut_slice(),
            seq_q,
            d_v,
            seq_k,
        )?;
    }

    Ok(out)
}

/// Batched scaled dot-product attention.
///
/// `Q` shape `[batch, seq_q, d_k]`, `K` shape `[batch, seq_k, d_k]`,
/// `V` shape `[batch, seq_k, d_v]`.
///
/// Returns an `AlignedVec<T, Align>` of length `batch * seq_q * d_v`.
///
/// # Errors
/// [`SimdError::LengthMismatch`] if batch sizes or inner dimensions mismatch.
#[inline]
pub fn batch_attention<T, Arch, Align, const TILE_M: usize, const TILE_N: usize>(
    q: &TensorView<'_, T, 3>,
    k: &TensorView<'_, T, 3>,
    v: &TensorView<'_, T, 3>,
) -> Result<AlignedVec<T, Align>, SimdError>
where
    T: Scalar + FloatElement,
    Arch: SimdArch + SimdKernel<T>,
    Align: Alignment,
{
    let [batch, seq_q, d_k] = q.shape();
    let [batch2, seq_k, d_k2] = k.shape();
    let [batch3, seq_k2, d_v] = v.shape();

    if batch != batch2 || batch != batch3 || d_k != d_k2 || seq_k != seq_k2 {
        return Err(SimdError::LengthMismatch);
    }

    let total = batch * seq_q * d_v;
    let mut out: AlignedVec<T, Align> = AlignedVec::with_capacity(total);
    unsafe { out.set_len(total); }
    for x in out.as_mut_slice().iter_mut() { *x = T::ZERO; }

    for b in 0..batch {
        let q_mat = q.matrix_at(b).map_err(|_| SimdError::LengthMismatch)?;
        let k_mat = k.matrix_at(b).map_err(|_| SimdError::LengthMismatch)?;
        let v_mat = v.matrix_at(b).map_err(|_| SimdError::LengthMismatch)?;

        let result = attention::<T, Arch, Align, TILE_M, TILE_N>(&q_mat, &k_mat, &v_mat)?;

        let o_start = b * seq_q * d_v;
        out.as_mut_slice()[o_start..o_start + seq_q * d_v]
            .copy_from_slice(result.as_slice());
    }

    Ok(out)
}

// ---------------------------------------------------------------------------
// Private helpers
// ---------------------------------------------------------------------------

/// Transpose an `rows × cols` row-major matrix into `out` (now `cols × rows`).
///
/// `out.len()` must be `rows * cols`. Fully scalar — no SIMD overhead for the
/// transpose itself, which is a non-critical setup path.
#[inline(never)]
fn transpose_into<T: Copy>(src: &[T], rows: usize, cols: usize, out: &mut [T]) {
    debug_assert_eq!(src.len(), rows * cols);
    debug_assert_eq!(out.len(), rows * cols);
    for r in 0..rows {
        for c in 0..cols {
            out[c * rows + r] = src[r * cols + c];
        }
    }
}

// Unit tests moved to integration tests in crates/hermes-simd/tests/tensor_tests.rs
