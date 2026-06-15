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
    let simd_len = (len / lane_count) * lane_count;
    let x_ptr = x.as_ptr();
    let out_ptr = out.as_mut_ptr();

    // SAFETY: `simd_len <= len` bounds every vector load/store inside both
    // equal-length slices; lanes are processed in disjoint `lane_count`
    // chunks of the same index range, so the read-modify-write of `out`
    // never overlaps a pending store.
    unsafe {
        let valpha = A::splat(alpha);
        let mut i = 0usize;
        while i < simd_len {
            let vx = A::load_unaligned(x_ptr.add(i));
            let vo = A::load_unaligned(out_ptr.add(i));
            A::store_unaligned(out_ptr.add(i), A::fmadd(vx, valpha, vo));
            i += lane_count;
        }
    }

    // Scalar tail.
    for i in simd_len..len {
        out[i] = out[i] + x[i] * alpha;
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
    if rows == 0 || depth == 0 || cols == 0 {
        return Ok(());
    }
    let Some(alpha_len) = rows.checked_mul(depth) else {
        return Err(SimdError::LengthMismatch);
    };
    let Some(panel_len) = depth.checked_mul(cols) else {
        return Err(SimdError::LengthMismatch);
    };
    let Some(last_row_offset) = rows
        .checked_sub(1)
        .and_then(|row| row.checked_mul(row_stride))
    else {
        return Err(SimdError::LengthMismatch);
    };
    let Some(required_out_len) = last_row_offset.checked_add(cols) else {
        return Err(SimdError::LengthMismatch);
    };
    if alphas.len() < alpha_len
        || x_panel.len() < panel_len
        || row_stride < cols
        || out.len() < required_out_len
    {
        return Err(SimdError::LengthMismatch);
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
    use super::super::{axpy, axpy_rows, axpy_rows_batch};
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
        assert!(axpy(1.0, &x, &mut out).is_err());
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

        assert!(axpy_rows(&alphas, &x, &mut out, 2, 2, 3).is_err());
        assert!(axpy_rows(&alphas[..1], &x, &mut out, 3, 2, 3).is_err());
        assert!(axpy_rows(&alphas, &x[..2], &mut out, 3, 2, 3).is_err());
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
