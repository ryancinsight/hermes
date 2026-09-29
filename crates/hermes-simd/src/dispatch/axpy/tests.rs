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
fn axpy_tail_preserves_fused_operation_order() {
    let len = 9usize;
    let alpha = 1.0_f32 / 3.0;
    let x: Vec<f32> = (0..len).map(|i| i as f32 + 0.125).collect();
    let mut out: Vec<f32> = (0..len).map(|i| i as f32 * 0.75 + 0.2).collect();
    let expected: Vec<f32> = out
        .iter()
        .zip(&x)
        .map(|(&out, &x)| x.mul_add(alpha, out))
        .collect();

    axpy(alpha, &x, &mut out).unwrap();
    assert_eq!(out, expected);
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
fn axpy_mul_tail_preserves_fused_operation_order() {
    let len = 9usize;
    let alpha = 1.0_f32 / 3.0;
    let a: Vec<f32> = (0..len).map(|i| i as f32 + 0.125).collect();
    let b: Vec<f32> = (0..len).map(|i| 0.75 - i as f32 * 0.0625).collect();
    let mut out: Vec<f32> = (0..len).map(|i| i as f32 * 0.5 + 0.2).collect();
    let expected: Vec<f32> = out
        .iter()
        .zip(&a)
        .zip(&b)
        .map(|((&out, &a), &b)| (alpha * a).mul_add(b, out))
        .collect();

    axpy_mul(alpha, &a, &b, &mut out).unwrap();
    assert_eq!(out, expected);
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
    let x: Vec<f64> = (0..50).map(f64::from).collect();
    let mut out: Vec<f64> = (0..50).map(|i| f64::from(i) * 2.0).collect();
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
fn axpy_rows_tail_preserves_fused_operation_order() {
    let rows = 2usize;
    let cols = 9usize;
    let row_stride = 11usize;
    let alphas = [1.0_f32 / 3.0, -0.25];
    let x: Vec<f32> = (0..cols).map(|i| i as f32 + 0.125).collect();
    let mut out: Vec<f32> = (0..rows * row_stride)
        .map(|i| i as f32 * 0.5 + 0.2)
        .collect();
    let mut expected = out.clone();
    for row in 0..rows {
        let start = row * row_stride;
        for col in 0..cols {
            expected[start + col] = x[col].mul_add(alphas[row], expected[start + col]);
        }
    }

    axpy_rows(&alphas, &x, &mut out, row_stride, rows, cols).unwrap();
    assert_eq!(out, expected);
}

#[test]
fn axpy_rows_masked_tails_cover_multiple_widths_and_f64() {
    for &cols in &[1usize, 2, 3, 5, 9, 17, 65] {
        let rows = 3usize;
        let row_stride = cols + 3;
        let alphas: Vec<f64> = (0..rows).map(|row| row as f64 * 0.125 - 0.25).collect();
        let x: Vec<f64> = (0..cols).map(|col| col as f64 * 0.375 + 0.0625).collect();
        let mut out: Vec<f64> = (0..rows * row_stride)
            .map(|index| index as f64 * 0.25 - 0.5)
            .collect();
        let mut expected = out.clone();
        for row in 0..rows {
            let start = row * row_stride;
            for col in 0..cols {
                expected[start + col] = x[col].mul_add(alphas[row], expected[start + col]);
            }
        }

        axpy_rows(&alphas, &x, &mut out, row_stride, rows, cols).unwrap();
        assert_eq!(out, expected, "cols {cols}");
    }
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
fn axpy_rows_batch_tail_preserves_fused_operation_order() {
    let rows = 2usize;
    let depth = 2usize;
    let cols = 9usize;
    let row_stride = 11usize;
    let alphas = [1.0_f32 / 3.0, -0.25, 0.2, -1.0 / 7.0];
    let x_panel: Vec<f32> = (0..depth * cols)
        .map(|i| i as f32 * 0.125 + 0.0625)
        .collect();
    let mut out: Vec<f32> = (0..rows * row_stride)
        .map(|i| i as f32 * 0.5 + 0.2)
        .collect();
    let mut expected = out.clone();
    for row in 0..rows {
        let start = row * row_stride;
        for col in 0..cols {
            let mut value = expected[start + col];
            for shared in 0..depth {
                value =
                    x_panel[shared * cols + col].mul_add(alphas[shared * rows + row], value);
            }
            expected[start + col] = value;
        }
    }

    axpy_rows_batch(&alphas, &x_panel, &mut out, row_stride, rows, depth, cols).unwrap();
    assert_eq!(out, expected);
}

#[test]
fn axpy_rows_batch_masked_tails_cover_multiple_widths_and_depths() {
    for &(cols, depth) in &[(1usize, 1usize), (3, 2), (5, 3), (9, 4), (17, 4), (65, 3)] {
        let rows = 3usize;
        let row_stride = cols + 2;
        let alphas: Vec<f64> = (0..rows * depth)
            .map(|index| index as f64 * 0.125 - 0.375)
            .collect();
        let x_panel: Vec<f64> = (0..depth * cols)
            .map(|index| index as f64 * 0.25 + 0.0625)
            .collect();
        let mut out: Vec<f64> = (0..rows * row_stride)
            .map(|index| index as f64 * 0.125 - 0.25)
            .collect();
        let mut expected = out.clone();
        for row in 0..rows {
            let start = row * row_stride;
            for col in 0..cols {
                let mut value = expected[start + col];
                for shared in 0..depth {
                    value = x_panel[shared * cols + col]
                        .mul_add(alphas[shared * rows + row], value);
                }
                expected[start + col] = value;
            }
        }

        axpy_rows_batch(&alphas, &x_panel, &mut out, row_stride, rows, depth, cols).unwrap();
        assert_eq!(out, expected, "cols {cols}, depth {depth}");
    }
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
