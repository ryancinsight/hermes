//! Generic runtime-dispatch AXPY kernel: `out[i] += alpha * x[i]`.
//!
//! The fused row-update primitives matmul-style consumers (leto) accumulate
//! with broadcast multipliers, streaming reads of RHS panels, and in-place
//! updates of `out`, with no temporary allocation. SIMD chunks use the fused
//! multiply-add primitive; scalar tails are covered element-by-element.

use hermes_simd_core::{arch::SimdArch, kernel::SimdKernel, scalar::Scalar, view::SimdError};
use hermes_simd_macros::runtime_dispatch;

/// Apply the final partial AXPY vector through the provider's masked arithmetic
/// seam without ever reading or writing beyond the live tail.
#[inline(always)]
unsafe fn axpy_masked_tail<T, A>(alpha: T, x: *const T, out: *mut T, tail: usize)
where
    T: Scalar,
    A: SimdArch + SimdKernel<T>,
{
    debug_assert!(tail > 0 && tail < A::LANE_COUNT);

    let mask = A::leading_k_mask(tail);
    let x_value = A::masked_load_partial(x, tail, mask, A::zero());
    let out_value = A::masked_load_partial(out, tail, mask, A::zero());
    let result = A::masked_fmadd(x_value, A::splat(alpha), out_value, mask);
    A::masked_store_partial(out, tail, mask, result);
}

/// Apply a SIMD AXPY tail through the active-prefix masked-memory seam.
///
/// The backend owns mask semantics, while this dispatch kernel owns the exact
/// accessible-prefix proof for each caller slice.

#[runtime_dispatch(avx512f, avx2, neon, scalar)]
pub(super) fn dispatch_axpy_kernel<T, A>(alpha: T, x: &[T], out: &mut [T]) -> Result<(), SimdError>
where
    T: Scalar,
    A: SimdArch + SimdKernel<T>,
{
    if x.len() != out.len() {
        return Err(SimdError::LengthMismatch);
    }
    let len = out.len();
    if len == 0 {
        return Ok(());
    }

    let lane_count = A::LANE_COUNT;
    let unroll_factor = A::UNROLL_FACTOR;
    let chunk_size = lane_count * unroll_factor;
    let unrolled_simd_len = (len / chunk_size) * chunk_size;
    let x_ptr = x.as_ptr();
    let out_ptr = out.as_mut_ptr();

    // SAFETY: `chunk_size | lane_count` bounds every pointer inside both slices.
    // The 4-accumulator loop hides store-to-load latency on modern micro-architectures
    // and matches the throughput model used by `dot()` and `scale()`.
    unsafe {
        let valpha = A::splat(alpha);
        let mut i = 0usize;

        // ── 4× unrolled FMA loop ─────────────────────────────────────────────
        while i < unrolled_simd_len {
            let px0 = x_ptr.add(i);
            let px1 = x_ptr.add(i + lane_count);
            let px2 = x_ptr.add(i + lane_count * 2);
            let px3 = x_ptr.add(i + lane_count * 3);
            let po0 = out_ptr.add(i);
            let po1 = out_ptr.add(i + lane_count);
            let po2 = out_ptr.add(i + lane_count * 2);
            let po3 = out_ptr.add(i + lane_count * 3);
            A::store_unaligned(
                po0,
                A::fmadd(A::load_unaligned(px0), valpha, A::load_unaligned(po0)),
            );
            A::store_unaligned(
                po1,
                A::fmadd(A::load_unaligned(px1), valpha, A::load_unaligned(po1)),
            );
            A::store_unaligned(
                po2,
                A::fmadd(A::load_unaligned(px2), valpha, A::load_unaligned(po2)),
            );
            A::store_unaligned(
                po3,
                A::fmadd(A::load_unaligned(px3), valpha, A::load_unaligned(po3)),
            );
            i += chunk_size;
        }

        // ── Remaining full SIMD vectors ──────────────────────────────────────
        let simd_len = (len / lane_count) * lane_count;
        while i < simd_len {
            let p = out_ptr.add(i);
            A::store_unaligned(
                p,
                A::fmadd(
                    A::load_unaligned(x_ptr.add(i)),
                    valpha,
                    A::load_unaligned(p),
                ),
            );
            i += lane_count;
        }
    }

    // The provider-owned partial-memory seam keeps the final SIMD operation
    // inside the live slice even on an allocation or page boundary.
    let simd_len = (len / lane_count) * lane_count;
    let tail = len - simd_len;
    if tail != 0 {
        // SAFETY: `simd_len` is the start of the remaining in-bounds prefix;
        // `tail` is its exact length and is smaller than the lane width.
        unsafe {
            axpy_masked_tail::<T, A>(
                alpha,
                x.as_ptr().add(simd_len),
                out.as_mut_ptr().add(simd_len),
                tail,
            );
        }
    }
    Ok(())
}

/// Apply the final partial fused ternary update without reading or writing
/// beyond the live tail.
#[inline(always)]
unsafe fn axpy_mul_masked_tail<T, A>(alpha: T, a: *const T, b: *const T, out: *mut T, tail: usize)
where
    T: Scalar,
    A: SimdArch + SimdKernel<T>,
{
    debug_assert!(tail > 0 && tail < A::LANE_COUNT);

    let mask = A::leading_k_mask(tail);
    let a_value = A::masked_load_partial(a, tail, mask, A::zero());
    let b_value = A::masked_load_partial(b, tail, mask, A::zero());
    let out_value = A::masked_load_partial(out, tail, mask, A::zero());
    let scaled_a = A::mul(a_value, A::splat(alpha));
    let result = A::masked_fmadd(scaled_a, b_value, out_value, mask);
    A::masked_store_partial(out, tail, mask, result);
}

/// Fused ternary update `out[i] += alpha * a[i] * b[i]` without a temporary.
///
/// The scaled first operand is multiplied in the SIMD register and accumulated
/// with `fmadd`, so the output is written once per lane. The scalar tail uses
/// the same `(alpha * a) * b + out` operation order as the vector path.
#[runtime_dispatch(avx512f, avx2, neon, scalar)]
pub(super) fn dispatch_axpy_mul_kernel<T, A>(
    alpha: T,
    a: &[T],
    b: &[T],
    out: &mut [T],
) -> Result<(), SimdError>
where
    T: Scalar,
    A: SimdArch + SimdKernel<T>,
{
    if a.len() != b.len() || a.len() != out.len() {
        return Err(SimdError::LengthMismatch);
    }
    let len = out.len();
    if len == 0 {
        return Ok(());
    }

    let lane_count = A::LANE_COUNT;
    let unroll_factor = A::UNROLL_FACTOR;
    let chunk_size = lane_count * unroll_factor;
    let unrolled_simd_len = (len / chunk_size) * chunk_size;
    let a_ptr = a.as_ptr();
    let b_ptr = b.as_ptr();
    let out_ptr = out.as_mut_ptr();

    // SAFETY: length validation proves every unrolled and vector load/store is
    // within its corresponding slice; the dispatch wrapper proves the target
    // feature required by `A` before entering this kernel.
    unsafe {
        let valpha = A::splat(alpha);
        let mut i = 0usize;
        while i < unrolled_simd_len {
            for offset in [0, lane_count, lane_count * 2, lane_count * 3] {
                let pa = a_ptr.add(i + offset);
                let pb = b_ptr.add(i + offset);
                let po = out_ptr.add(i + offset);
                let scaled_a = A::mul(A::load_unaligned(pa), valpha);
                A::store_unaligned(
                    po,
                    A::fmadd(scaled_a, A::load_unaligned(pb), A::load_unaligned(po)),
                );
            }
            i += chunk_size;
        }

        let simd_len = (len / lane_count) * lane_count;
        while i < simd_len {
            let po = out_ptr.add(i);
            let scaled_a = A::mul(A::load_unaligned(a_ptr.add(i)), valpha);
            A::store_unaligned(
                po,
                A::fmadd(
                    scaled_a,
                    A::load_unaligned(b_ptr.add(i)),
                    A::load_unaligned(po),
                ),
            );
            i += lane_count;
        }
    }

    let simd_len = (len / lane_count) * lane_count;
    let tail = len - simd_len;
    if tail != 0 {
        // SAFETY: `simd_len` is the start of the remaining in-bounds prefix and
        // `tail` is its exact length, strictly smaller than the lane width.
        unsafe {
            axpy_mul_masked_tail::<T, A>(
                alpha,
                a.as_ptr().add(simd_len),
                b.as_ptr().add(simd_len),
                out.as_mut_ptr().add(simd_len),
                tail,
            );
        }
    }
    Ok(())
}

#[runtime_dispatch(avx512f, avx2, neon, scalar)]
pub(super) fn dispatch_axpy_rows_kernel<T, A>(
    alphas: &[T],
    x: &[T],
    out: &mut [T],
    row_stride: usize,
    rows: usize,
    cols: usize,
) -> Result<(), SimdError>
where
    T: Scalar,
    A: SimdArch + SimdKernel<T>,
{
    if rows == 0 || cols == 0 {
        return Ok(());
    }
    let Some(last_row_offset) = rows
        .checked_sub(1)
        .and_then(|row| row.checked_mul(row_stride))
    else {
        return Err(SimdError::LengthMismatch);
    };
    let Some(required_out_len) = last_row_offset.checked_add(cols) else {
        return Err(SimdError::LengthMismatch);
    };
    if alphas.len() < rows || x.len() < cols || row_stride < cols || out.len() < required_out_len {
        return Err(SimdError::LengthMismatch);
    }

    let lane_count = A::LANE_COUNT;
    let simd_len = (cols / lane_count) * lane_count;
    let x_ptr = x.as_ptr();
    let out_ptr = out.as_mut_ptr();

    for (row, &alpha) in alphas.iter().take(rows).enumerate() {
        // SAFETY: validation above proves every row spans `cols` elements
        // inside `out` with `row_stride >= cols`, so rows are disjoint. The
        // RHS row spans `cols` elements inside `x`, and the vector loop stays
        // within `simd_len <= cols`.
        unsafe {
            let valpha = A::splat(alpha);
            let row_ptr = out_ptr.add(row * row_stride);
            let mut col = 0usize;
            while col < simd_len {
                let vx = A::load_unaligned(x_ptr.add(col));
                let vo = A::load_unaligned(row_ptr.add(col));
                A::store_unaligned(row_ptr.add(col), A::fmadd(vx, valpha, vo));
                col += lane_count;
            }

            let tail = cols - simd_len;
            if tail != 0 {
                // SAFETY: `simd_len` is the start of the validated row prefix;
                // the helper copies and writes only its exact live tail.
                axpy_masked_tail::<T, A>(alpha, x_ptr.add(simd_len), row_ptr.add(simd_len), tail);
            }
        }
    }

    Ok(())
}

/// Apply one row's final partial AXPY vector through partial masked memory.
///
/// The caller has already validated the row and tail bounds. Keeping the
/// boundary handling here reuses the same provider-owned masked FMA contract as
/// the one-row AXPY path without exposing a second public SIMD abstraction.
#[inline(always)]
unsafe fn axpy_rows_batch_masked_tail<T, A>(
    alphas: *const T,
    x_panel: *const T,
    out: *mut T,
    row: usize,
    rows: usize,
    depth: usize,
    cols: usize,
    simd_len: usize,
    tail: usize,
) where
    T: Scalar,
    A: SimdArch + SimdKernel<T>,
{
    debug_assert!(tail > 0 && tail < A::LANE_COUNT);

    let mask = A::leading_k_mask(tail);
    let mut acc = A::masked_load_partial(out, tail, mask, A::zero());
    for shared in 0..depth {
        let x_tail = x_panel.add(shared * cols + simd_len);
        let x_value = A::masked_load_partial(x_tail, tail, mask, A::zero());
        let alpha = *alphas.add(shared * rows + row);
        acc = A::masked_fmadd(x_value, A::splat(alpha), acc, mask);
    }
    A::masked_store_partial(out, tail, mask, acc);
}

/// Type-independent extent validation for `axpy_rows_batch`, extracted so it is
/// emitted once rather than re-monomorphized into every `(T, Arch)` instantiation
/// of the kernel. Returns `Ok(true)` when the problem is empty (caller returns
/// `Ok(())`), `Ok(false)` when the extents are valid, or a length error.
#[inline(never)]
fn check_axpy_rows_batch_extents(
    alphas_len: usize,
    x_panel_len: usize,
    out_len: usize,
    row_stride: usize,
    rows: usize,
    depth: usize,
    cols: usize,
) -> Result<bool, SimdError> {
    if rows == 0 || depth == 0 || cols == 0 {
        return Ok(true);
    }
    let alpha_len = rows.checked_mul(depth).ok_or(SimdError::LengthMismatch)?;
    let panel_len = depth.checked_mul(cols).ok_or(SimdError::LengthMismatch)?;
    let last_row_offset = rows
        .checked_sub(1)
        .and_then(|row| row.checked_mul(row_stride))
        .ok_or(SimdError::LengthMismatch)?;
    let required_out_len = last_row_offset
        .checked_add(cols)
        .ok_or(SimdError::LengthMismatch)?;
    if alphas_len < alpha_len
        || x_panel_len < panel_len
        || row_stride < cols
        || out_len < required_out_len
    {
        return Err(SimdError::LengthMismatch);
    }
    Ok(false)
}

#[runtime_dispatch(avx512f, avx2, neon, scalar)]
pub(super) fn dispatch_axpy_rows_batch_kernel<T, A>(
    alphas: &[T],
    x_panel: &[T],
    out: &mut [T],
    row_stride: usize,
    rows: usize,
    depth: usize,
    cols: usize,
) -> Result<(), SimdError>
where
    T: Scalar,
    A: SimdArch + SimdKernel<T>,
{
    if check_axpy_rows_batch_extents(
        alphas.len(),
        x_panel.len(),
        out.len(),
        row_stride,
        rows,
        depth,
        cols,
    )? {
        return Ok(());
    }

    let lane_count = A::LANE_COUNT;
    let simd_len = (cols / lane_count) * lane_count;
    let x_ptr = x_panel.as_ptr();
    let out_ptr = out.as_mut_ptr();

    for row in 0..rows {
        // SAFETY: validation above proves every RHS panel row has `cols`
        // elements, every output row spans `cols` elements inside `out` with
        // `row_stride >= cols`, and row windows are disjoint. The depth loop
        // uses checked `rows * depth` and `depth * cols` bounds validated
        // before pointer arithmetic.
        unsafe {
            let row_ptr = out_ptr.add(row * row_stride);
            let mut col = 0usize;
            while col < simd_len {
                let mut acc = A::load_unaligned(row_ptr.add(col));
                for shared in 0..depth {
                    let alpha = *alphas.get_unchecked(shared * rows + row);
                    let valpha = A::splat(alpha);
                    let x_row = x_ptr.add(shared * cols);
                    let vx = A::load_unaligned(x_row.add(col));
                    acc = A::fmadd(vx, valpha, acc);
                }
                A::store_unaligned(row_ptr.add(col), acc);
                col += lane_count;
            }

            let tail = cols - simd_len;
            if tail != 0 {
                // SAFETY: extent validation proves every panel and output row
                // contains this exact tail, and the helper writes only it.
                axpy_rows_batch_masked_tail::<T, A>(
                    alphas.as_ptr(),
                    x_ptr,
                    row_ptr.add(simd_len),
                    row,
                    rows,
                    depth,
                    cols,
                    simd_len,
                    tail,
                );
            }
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests;
