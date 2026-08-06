//! Generic runtime-dispatch AXPY kernel: `out[i] += alpha * x[i]`.
//!
//! The fused row-update primitives matmul-style consumers (leto) accumulate
//! with broadcast multipliers, streaming reads of RHS panels, and in-place
//! updates of `out`, with no temporary allocation. SIMD chunks use the fused
//! multiply-add primitive; scalar tails are covered element-by-element.

use hermes_simd_core::{arch::SimdArch, kernel::SimdKernel, scalar::Scalar, view::SimdError};
use hermes_simd_macros::runtime_dispatch;

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

    // Scalar tail.
    let simd_len = (len / lane_count) * lane_count;
    for i in simd_len..len {
        out[i] = out[i] + x[i] * alpha;
    }
    Ok(())
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
    for i in simd_len..len {
        out[i] = (alpha * a[i]).scalar_fmadd(b[i], out[i]);
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

            for col in simd_len..cols {
                let out_ref = row_ptr.add(col);
                *out_ref = *out_ref + *x_ptr.add(col) * alpha;
            }
        }
    }

    Ok(())
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

            for col in simd_len..cols {
                let out_ref = row_ptr.add(col);
                let mut acc = *out_ref;
                for shared in 0..depth {
                    let alpha = *alphas.get_unchecked(shared * rows + row);
                    let x_row = x_ptr.add(shared * cols);
                    acc = acc + *x_row.add(col) * alpha;
                }
                *out_ref = acc;
            }
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::super::{axpy, axpy_mul, axpy_rows, axpy_rows_batch};
    use hermes_simd_core::view::SimdError;

    #[test]
    fn axpy_matches_scalar_reference_across_tail_sizes() {
        // Sizes straddle every lane width incl. partial tails.
        for &len in &[0usize, 1, 3, 7, 8, 9, 15, 16, 17, 63, 64, 65, 1027] {
            let alpha = 1.75f64;
            let x: Vec<f64> = (0..len).map(|i| i as f64 * 0.5 - 3.0).collect();
            let mut out: Vec<f64> = (0..len).map(|i| 100.0 - i as f64).collect();
            let expected: Vec<f64> = out.iter().zip(&x).map(|(o, xv)| o + alpha * xv).collect();

            axpy(alpha, &x, &mut out).unwrap();
            assert_eq!(out, expected, "len {len}");
        }
    }

    #[test]
    fn axpy_mul_matches_scalar_fma_reference_across_tail_sizes() {
        for &len in &[0usize, 1, 3, 7, 8, 9, 15, 16, 17, 63, 64, 65, 1027] {
            let alpha = 1.75f64;
            let a: Vec<f64> = (0..len).map(|i| i as f64 * 0.5 - 3.0).collect();
            let b: Vec<f64> = (0..len).map(|i| i as f64 * 0.25 + 1.0).collect();
            let mut out: Vec<f64> = (0..len).map(|i| 100.0 - i as f64).collect();
            let expected: Vec<f64> = out
                .iter()
                .zip(&a)
                .zip(&b)
                .map(|((&out, &a), &b)| (alpha * a).mul_add(b, out))
                .collect();

            axpy_mul(alpha, &a, &b, &mut out).unwrap();
            assert_eq!(out, expected, "len {len}");
        }
    }

    #[test]
    fn axpy_mul_rejects_length_mismatch() {
        let mut out = [0.0f64; 2];
        assert_eq!(
            axpy_mul(1.0, &[1.0, 2.0], &[3.0], &mut out),
            Err(SimdError::LengthMismatch)
        );
        assert_eq!(
            axpy_mul(1.0, &[1.0], &[3.0], &mut out),
            Err(SimdError::LengthMismatch)
        );
    }

    #[test]
    fn axpy_single_precision_matches_reference() {
        let alpha = -0.5f32;
        let x: Vec<f32> = (0..133).map(|i| i as f32).collect();
        let mut out = vec![1.0f32; 133];
        let expected: Vec<f32> = x.iter().map(|xv| 1.0 + alpha * xv).collect();
        axpy(alpha, &x, &mut out).unwrap();
        assert_eq!(out, expected);
    }

    #[test]
    fn axpy_rejects_length_mismatch() {
        let x = [1.0f64, 2.0];
        let mut out = [0.0f64; 3];
        assert_eq!(axpy(1.0, &x, &mut out), Err(SimdError::LengthMismatch));
    }

    #[test]
    fn axpy_zero_alpha_is_identity() {
        let x: Vec<f64> = (0..50).map(|i| i as f64).collect();
        let mut out: Vec<f64> = (0..50).map(|i| i as f64 * 2.0).collect();
        let expected = out.clone();
        axpy(0.0, &x, &mut out).unwrap();
        assert_eq!(out, expected);
    }

    #[test]
    fn axpy_rows_matches_repeated_axpy_with_padding() {
        let rows = 5usize;
        let cols = 17usize;
        let row_stride = 23usize;
        let alphas: Vec<f64> = (0..rows).map(|row| row as f64 * 0.25 - 0.5).collect();
        let x: Vec<f64> = (0..cols).map(|col| col as f64 * 1.5 - 2.0).collect();
        let mut out: Vec<f64> = (0..rows * row_stride).map(|i| i as f64 * 0.125).collect();
        let mut expected = out.clone();

        for row in 0..rows {
            let start = row * row_stride;
            axpy(alphas[row], &x, &mut expected[start..start + cols]).unwrap();
        }

        axpy_rows(&alphas, &x, &mut out, row_stride, rows, cols).unwrap();
        assert_eq!(out, expected);
    }

    #[test]
    fn axpy_rows_rejects_invalid_extents() {
        let alphas = [1.0f64, 2.0];
        let x = [1.0f64, 2.0, 3.0];
        let mut out = [0.0f64; 5];

        assert_eq!(
            axpy_rows(&alphas, &x, &mut out, 2, 2, 3),
            Err(SimdError::LengthMismatch)
        );
        assert_eq!(
            axpy_rows(&alphas[..1], &x, &mut out, 3, 2, 3),
            Err(SimdError::LengthMismatch)
        );
        assert_eq!(
            axpy_rows(&alphas, &x[..2], &mut out, 3, 2, 3),
            Err(SimdError::LengthMismatch)
        );
    }

    #[test]
    fn axpy_rows_batch_matches_repeated_axpy_rows_with_padding() {
        let rows = 4usize;
        let depth = 3usize;
        let cols = 19usize;
        let row_stride = 23usize;
        let alphas: Vec<f64> = (0..rows * depth)
            .map(|idx| idx as f64 * 0.125 - 0.75)
            .collect();
        let x_panel: Vec<f64> = (0..depth * cols)
            .map(|idx| idx as f64 * 0.25 - 1.5)
            .collect();
        let mut out: Vec<f64> = (0..rows * row_stride).map(|i| i as f64 * 0.03125).collect();
        let mut expected = out.clone();

        for shared in 0..depth {
            let alpha_start = shared * rows;
            let x_start = shared * cols;
            axpy_rows(
                &alphas[alpha_start..alpha_start + rows],
                &x_panel[x_start..x_start + cols],
                &mut expected,
                row_stride,
                rows,
                cols,
            )
            .unwrap();
        }

        axpy_rows_batch(&alphas, &x_panel, &mut out, row_stride, rows, depth, cols).unwrap();
        assert_eq!(out, expected);
    }

    #[test]
    fn axpy_rows_batch_rejects_invalid_extents() {
        let alphas = [1.0f64, 2.0, 3.0, 4.0];
        let x_panel = [1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mut out = [0.0f64; 6];

        assert_eq!(
            axpy_rows_batch(&alphas, &x_panel, &mut out, 2, 2, 2, 3),
            Err(SimdError::LengthMismatch)
        );
        assert_eq!(
            axpy_rows_batch(&alphas[..3], &x_panel, &mut out, 3, 2, 2, 3),
            Err(SimdError::LengthMismatch)
        );
        assert_eq!(
            axpy_rows_batch(&alphas, &x_panel[..5], &mut out, 3, 2, 2, 3),
            Err(SimdError::LengthMismatch)
        );
    }
}
