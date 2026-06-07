//! Integration tests for TensorView, matmul, batch_matmul, softmax, and histogram.

use hermes_simd::*;

// ---------------------------------------------------------------------------
// TensorView construction and stride verification
// ---------------------------------------------------------------------------

#[test]
fn test_tensor_view_row_major_strides_2d() {
    let data: Vec<f32> = (0..12).map(|x| x as f32).collect();
    let t = TensorView::<f32, 2>::new(&data, [3, 4]).unwrap();
    // Row-major strides for [3, 4]: strides = [4, 1]
    assert_eq!(t.strides(), [4, 1]);
    assert_eq!(t.shape(), [3, 4]);
    assert_eq!(t.num_elements(), 12);
}

#[test]
fn test_tensor_view_row_major_strides_3d() {
    let data: Vec<i32> = (0..24).collect();
    let t = TensorView::<i32, 3>::new(&data, [2, 3, 4]).unwrap();
    // Strides for [2, 3, 4]: [12, 4, 1]
    assert_eq!(t.strides(), [12, 4, 1]);
    assert_eq!(t.num_elements(), 24);
}

// ---------------------------------------------------------------------------
// Bounds-checked element access
// ---------------------------------------------------------------------------

#[test]
fn test_tensor_view_get_2d() {
    let data: Vec<i32> = (0..12).collect();
    let t = TensorView::<i32, 2>::new(&data, [3, 4]).unwrap();
    // Element [0][0] = 0
    assert_eq!(t.get([0, 0]).unwrap(), 0);
    // Element [1][2] = 1*4 + 2 = 6
    assert_eq!(t.get([1, 2]).unwrap(), 6);
    // Element [2][3] = 2*4 + 3 = 11
    assert_eq!(t.get([2, 3]).unwrap(), 11);
}

#[test]
fn test_tensor_view_get_out_of_bounds() {
    let data: Vec<f32> = (0..6).map(|x| x as f32).collect();
    let t = TensorView::<f32, 2>::new(&data, [2, 3]).unwrap();
    assert!(matches!(t.get([2, 0]), Err(TensorError::IndexOutOfBounds)));
    assert!(matches!(t.get([0, 3]), Err(TensorError::IndexOutOfBounds)));
}

// ---------------------------------------------------------------------------
// Shape mismatch error
// ---------------------------------------------------------------------------

#[test]
fn test_tensor_view_shape_mismatch() {
    let data: Vec<f32> = vec![1.0; 5];
    // shape product = 12 > data.len() = 5
    let result = TensorView::<f32, 2>::new(&data, [3, 4]);
    assert!(matches!(result, Err(TensorError::ShapeMismatch)));
}

// ---------------------------------------------------------------------------
// Reshape
// ---------------------------------------------------------------------------

#[test]
fn test_tensor_reshape_2d_to_1d() {
    let data: Vec<i32> = (0..12).collect();
    let t2d = TensorView::<i32, 2>::new(&data, [3, 4]).unwrap();
    let t1d = t2d.reshape([12]).unwrap();
    assert_eq!(t1d.num_elements(), 12);
    assert_eq!(t1d.get([0]).unwrap(), 0);
    assert_eq!(t1d.get([11]).unwrap(), 11);
}

#[test]
fn test_tensor_reshape_element_count_mismatch() {
    let data: Vec<i32> = (0..12).collect();
    let t = TensorView::<i32, 2>::new(&data, [3, 4]).unwrap();
    // 13 ≠ 12 → ShapeMismatch
    assert!(matches!(t.reshape([13]), Err(TensorError::ShapeMismatch)));
}

// ---------------------------------------------------------------------------
// row_view and iter_rows on 2-D tensor
// ---------------------------------------------------------------------------

#[test]
fn test_tensor_row_view() {
    let data: Vec<f32> = (0..9).map(|x| x as f32).collect();
    let t = TensorView::<f32, 2>::new(&data, [3, 3]).unwrap();

    let row0 = t.row_view(0).unwrap();
    assert_eq!(row0.get([0]).unwrap(), 0.0);
    assert_eq!(row0.get([2]).unwrap(), 2.0);

    let row1 = t.row_view(1).unwrap();
    assert_eq!(row1.get([0]).unwrap(), 3.0);

    let row2 = t.row_view(2).unwrap();
    assert_eq!(row2.get([2]).unwrap(), 8.0);
}

#[test]
fn test_tensor_iter_rows() {
    let data: Vec<i32> = (0..6).collect();
    let t = TensorView::<i32, 2>::new(&data, [2, 3]).unwrap();
    let rows: Vec<&[i32]> = t.iter_rows().unwrap().collect();
    assert_eq!(rows.len(), 2);
    assert_eq!(rows[0], &[0, 1, 2]);
    assert_eq!(rows[1], &[3, 4, 5]);
}

// ---------------------------------------------------------------------------
// 3-D batch slice
// ---------------------------------------------------------------------------

#[test]
fn test_tensor_matrix_at() {
    let data: Vec<f32> = (0..24).map(|x| x as f32).collect();
    // Shape [2, 3, 4]: two 3×4 matrices
    let t3 = TensorView::<f32, 3>::new(&data, [2, 3, 4]).unwrap();

    let mat0 = t3.matrix_at(0).unwrap();
    assert_eq!(mat0.shape(), [3, 4]);
    assert_eq!(mat0.get([0, 0]).unwrap(), 0.0);
    assert_eq!(mat0.get([2, 3]).unwrap(), 11.0);

    let mat1 = t3.matrix_at(1).unwrap();
    assert_eq!(mat1.get([0, 0]).unwrap(), 12.0);
    assert_eq!(mat1.get([2, 3]).unwrap(), 23.0);
}

// ---------------------------------------------------------------------------
// 2-D matmul via TilingPolicy (Scalar architecture)
// ---------------------------------------------------------------------------

#[test]
fn test_tensor_matmul_f32_scalar() {
    // A = [[1,2],[3,4]]  B = [[5,6],[7,8]]
    // C = A * B = [[1*5+2*7, 1*6+2*8],[3*5+4*7, 3*6+4*8]]
    //           = [[19, 22], [43, 50]]
    let a_data = [1.0f32, 2.0, 3.0, 4.0];
    let b_data = [5.0f32, 6.0, 7.0, 8.0];

    let a = TensorView::<f32, 2>::new(&a_data, [2, 2]).unwrap();
    let b = TensorView::<f32, 2>::new(&b_data, [2, 2]).unwrap();

    let c: AlignedVec<f32, Unaligned> =
        matmul::<f32, Scalar, Unaligned, 1, 1>(&a, &b).unwrap();

    assert_eq!(c.as_slice().len(), 4);
    assert!((c.as_slice()[0] - 19.0).abs() < 1e-4, "c[0][0] = {}", c.as_slice()[0]);
    assert!((c.as_slice()[1] - 22.0).abs() < 1e-4, "c[0][1] = {}", c.as_slice()[1]);
    assert!((c.as_slice()[2] - 43.0).abs() < 1e-4, "c[1][0] = {}", c.as_slice()[2]);
    assert!((c.as_slice()[3] - 50.0).abs() < 1e-4, "c[1][1] = {}", c.as_slice()[3]);
}

#[test]
fn test_tensor_matmul_identity_4x4() {
    // A = 4×4 identity, B = random 4×4 → C = B
    let mut a_data = [0.0f32; 16];
    for i in 0..4 { a_data[i * 4 + i] = 1.0; }
    let b_data: Vec<f32> = (1..=16).map(|x| x as f32).collect();

    let a = TensorView::<f32, 2>::new(&a_data, [4, 4]).unwrap();
    let b = TensorView::<f32, 2>::new(&b_data, [4, 4]).unwrap();

    let c: AlignedVec<f32, Unaligned> =
        matmul::<f32, Scalar, Unaligned, 1, 1>(&a, &b).unwrap();

    for (i, (&actual, &expected)) in c.as_slice().iter().zip(b_data.iter()).enumerate() {
        assert!((actual - expected).abs() < 1e-4, "c[{i}] = {actual} expected {expected}");
    }
}

// ---------------------------------------------------------------------------
// Batched matmul
// ---------------------------------------------------------------------------

#[test]
fn test_tensor_batch_matmul_f32() {
    // Batch=2, each is a 2×2 identity * [[1,2],[3,4]]
    let identity = [1.0f32, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0]; // 2 matrices flattened
    let b_data   = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];  // 2 matrices flattened

    let a = TensorView::<f32, 3>::new(&identity, [2, 2, 2]).unwrap();
    let b = TensorView::<f32, 3>::new(&b_data,   [2, 2, 2]).unwrap();

    let c: AlignedVec<f32, Unaligned> =
        batch_matmul::<f32, Scalar, Unaligned, 1, 1>(&a, &b).unwrap();

    assert_eq!(c.as_slice().len(), 8);
    // First batch: I * [[1,2],[3,4]] = [[1,2],[3,4]]
    assert!((c.as_slice()[0] - 1.0).abs() < 1e-4);
    assert!((c.as_slice()[3] - 4.0).abs() < 1e-4);
    // Second batch: I * [[5,6],[7,8]] = [[5,6],[7,8]]
    assert!((c.as_slice()[4] - 5.0).abs() < 1e-4);
    assert!((c.as_slice()[7] - 8.0).abs() < 1e-4);
}

// ---------------------------------------------------------------------------
// Softmax (via hermes-simd free function)
// ---------------------------------------------------------------------------

#[test]
fn test_softmax_scalar_sums_to_one() {
    let logits = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let probs = softmax_alloc::<f32, Scalar>(&logits);
    let total: f32 = probs.iter().sum();
    assert!((total - 1.0).abs() < 1e-5, "softmax sum = {total}");
    for &p in &probs {
        assert!(p >= 0.0 && p <= 1.0, "out of [0,1]: {p}");
    }
}

#[test]
fn test_softmax_inplace_empty() {
    let mut empty: Vec<f32> = vec![];
    softmax_inplace::<f32, Scalar>(&mut empty);
    assert!(empty.is_empty());
}

#[test]
fn test_softmax_inplace_single_element() {
    let mut v = [2.0f32];
    softmax_inplace::<f32, Scalar>(&mut v);
    assert!((v[0] - 1.0).abs() < 1e-6, "single element softmax should be 1.0");
}

#[test]
fn test_softmax_uniform_logits() {
    // All equal logits → uniform distribution
    let logits = [1.0f32; 8];
    let probs = softmax_alloc::<f32, Scalar>(&logits);
    for &p in &probs {
        assert!((p - 0.125).abs() < 1e-5, "expected 0.125, got {p}");
    }
}

// ---------------------------------------------------------------------------
// Softmax via SimdCow::softmax_cow
// ---------------------------------------------------------------------------

#[test]
fn test_softmax_cow_f32() {
    let logits = vec![0.5f32, 1.0, 1.5, 2.0];
    let view = SimdView::<'_, f32, Scalar, Unaligned, Unmasked, &[f32]>::new(&logits).unwrap();
    let cow: SimdCow<'_, f32, Scalar, Unaligned> = SimdCow::Borrowed(view);
    let result = cow.softmax_cow();
    let total: f32 = result.as_ref().iter().sum();
    assert!((total - 1.0).abs() < 1e-5, "cow softmax sum = {total}");
}

// ---------------------------------------------------------------------------
// Histogram via SimdCow::histogram_cow
// ---------------------------------------------------------------------------

#[test]
fn test_histogram_cow_uniform() {
    // 100 values uniformly in [0, 10): should give 10 elements per bin
    let data: Vec<f32> = (0..100).map(|i| (i as f32 % 10.0)).collect();
    let view = SimdView::<'_, f32, Scalar, Unaligned, Unmasked, &[f32]>::new(&data).unwrap();
    let cow: SimdCow<'_, f32, Scalar, Unaligned> = SimdCow::Borrowed(view);
    let hist = cow.histogram_cow(10, 0.0, 10.0);

    assert_eq!(hist.len(), 10);
    let total: usize = hist.iter().sum();
    assert_eq!(total, 100, "total count should be 100, got {total}");
    for (i, &count) in hist.iter().enumerate() {
        assert_eq!(count, 10, "bin {i} should have 10 elements, got {count}");
    }
}

#[test]
fn test_histogram_cow_out_of_range_ignored() {
    let data = vec![-1.0f32, 0.0, 5.0, 10.0, 11.0];
    let view = SimdView::<'_, f32, Scalar, Unaligned, Unmasked, &[f32]>::new(&data).unwrap();
    let cow: SimdCow<'_, f32, Scalar, Unaligned> = SimdCow::Borrowed(view);
    let hist = cow.histogram_cow(10, 0.0, 10.0);

    // -1.0 < lo; 10.0 >= hi; 11.0 >= hi → only 0.0 and 5.0 are counted
    let total: usize = hist.iter().sum();
    assert_eq!(total, 2, "only 2 values should be counted, got {total}");
}

// ---------------------------------------------------------------------------
// Advanced TensorView mutable view, transpose, matmul_to, TensorCow, row-wise softmax
// ---------------------------------------------------------------------------

#[test]
fn test_mutable_tensor_view_and_slicing() {
    let mut data = [0.0f32; 12];
    let mut t = TensorView::new_mut(&mut data, [3, 4]).unwrap();
    t.set([1, 2], 5.0).unwrap();
    assert_eq!(t.get([1, 2]).unwrap(), 5.0);
    unsafe {
        t.set_unchecked([2, 3], 9.0);
        assert_eq!(t.get_unchecked([2, 3]), 9.0);
    }

    let mut row1 = t.row_view_mut(1).unwrap();
    assert_eq!(row1.get([2]).unwrap(), 5.0);
    row1.set([2], 25.0).unwrap();
    assert_eq!(t.get([1, 2]).unwrap(), 25.0);

    let rows: Vec<&mut [f32]> = t.iter_rows_mut().unwrap().collect();
    assert_eq!(rows.len(), 3);
    assert_eq!(rows[1][2], 25.0);
}

#[test]
fn test_tensor_view_transpose() {
    let data = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let t = TensorView::new(&data, [2, 3]).unwrap(); // RowMajor
    assert_eq!(t.strides(), [3, 1]);

    let t_trans = t.transpose(); // ColMajor
    assert_eq!(t_trans.shape(), [3, 2]);
    assert_eq!(t_trans.strides(), [1, 3]);

    // Original [1, 2] in RowMajor is index 1*3 + 2 = 5 (value 6.0)
    // In transposed ColMajor, this element is at [2, 1], index is 2*1 + 1*3 = 5 (value 6.0)
    assert_eq!(t_trans.get([2, 1]).unwrap(), 6.0);

    let t_back = t_trans.transpose(); // RowMajor
    assert_eq!(t_back.shape(), [2, 3]);
    assert_eq!(t_back.strides(), [3, 1]);
}

#[test]
fn test_tensor_matmul_to() {
    let a_data = [1.0f32, 2.0, 3.0, 4.0];
    let b_data = [5.0f32, 6.0, 7.0, 8.0];
    let mut c_data = [0.0f32; 4];

    let a = TensorView::new(&a_data, [2, 2]).unwrap();
    let b = TensorView::new(&b_data, [2, 2]).unwrap();
    let mut c = TensorView::new_mut(&mut c_data, [2, 2]).unwrap();

    matmul_to::<f32, Scalar, Unaligned, 1, 1>(&a, &b, &mut c).unwrap();

    assert_eq!(c_data, [19.0f32, 22.0, 43.0, 50.0]);
}

#[test]
fn test_tensor_cow_lazy_reshape() {
    let data = vec![1.0f32, 2.0, 3.0, 4.0];
    let view = TensorView::new(&data, [2, 2]).unwrap();
    let mut cow = TensorCow::<f32, 2>::borrowed(view);
    assert!(matches!(cow, TensorCow::Borrowed(_)));

    let reshaped = cow.clone().reshape([4]).unwrap();
    assert_eq!(reshaped.shape(), [4]);
    assert!(matches!(reshaped, TensorCow::Borrowed(_)));

    let mut_vec = cow.to_mut();
    mut_vec.as_mut_slice()[1] = 20.0;
    assert!(matches!(cow, TensorCow::Owned { .. }));
    assert_eq!(cow.as_view().get([0, 1]).unwrap(), 20.0);
}

#[test]
fn test_softmax_2d_rows() {
    let logits = [
        1.0f32, 2.0, 3.0,
        1.0, 1.0, 1.0,
    ];
    let t = TensorView::new(&logits, [2, 3]).unwrap();
    let probs = softmax_2d_rows::<f32, Scalar, RowMajor>(&t);

    // Row 1: softmax([1, 1, 1]) -> [1/3, 1/3, 1/3]
    assert!((probs[3] - 0.33333).abs() < 1e-4);
    assert!((probs[4] - 0.33333).abs() < 1e-4);
    assert!((probs[5] - 0.33333).abs() < 1e-4);

    let mut mut_logits = logits;
    let mut mut_tensor = TensorView::new_mut(&mut mut_logits, [2, 3]).unwrap();
    softmax_2d_rows_inplace::<f32, Scalar, RowMajor>(&mut mut_tensor);
    assert!((mut_logits[3] - 0.33333).abs() < 1e-4);
}

// ---------------------------------------------------------------------------
// Norm functions: L1, L2, L∞
// ---------------------------------------------------------------------------

#[test]
fn test_norm_l2_pythagorean() {
    let data = [3.0f32, 4.0];
    let n = norm_l2::<f32, Scalar>(&data);
    assert!((n - 5.0).abs() < 1e-5, "expected 5.0, got {n}");
}

#[test]
fn test_norm_l2_zero_vector() {
    let data = [0.0f32; 8];
    let n = norm_l2::<f32, Scalar>(&data);
    assert_eq!(n, 0.0);
}

#[test]
fn test_norm_l1_basic() {
    let data = [-1.0f32, 2.0, -3.0];
    let n = norm_l1::<f32, Scalar>(&data);
    assert!((n - 6.0).abs() < 1e-5, "expected 6.0, got {n}");
}

#[test]
fn test_norm_linf_basic() {
    let data = [1.0f32, -5.0, 3.0];
    let n = norm_linf::<f32, Scalar>(&data);
    assert!((n - 5.0).abs() < 1e-5, "expected 5.0, got {n}");
}

#[test]
fn test_norm_empty_slices() {
    let empty: [f32; 0] = [];
    assert_eq!(norm_l2::<f32, Scalar>(&empty), 0.0);
    assert_eq!(norm_l1::<f32, Scalar>(&empty), 0.0);
    assert_eq!(norm_linf::<f32, Scalar>(&empty), 0.0);
}

#[test]
fn test_normalize_l2_inplace_unit_norm() {
    let mut data = [3.0f32, 4.0];
    normalize_l2_inplace::<f32, Scalar>(&mut data);
    let n = norm_l2::<f32, Scalar>(&data);
    assert!((n - 1.0).abs() < 1e-5, "expected unit norm, got {n}");
}

#[test]
fn test_normalize_l2_inplace_zero_noop() {
    let mut data = [0.0f32, 0.0, 0.0];
    normalize_l2_inplace::<f32, Scalar>(&mut data);
    // Should not NaN — remains zero
    for &x in &data {
        assert!(x.is_finite(), "expected finite, got {x}");
    }
}

#[test]
fn test_row_norms_l2_basic() {
    // 2×3 matrix; rows = [3,4,0], [0,0,5]
    let data = [3.0f32, 4.0, 0.0, 0.0, 0.0, 5.0];
    let t = TensorView::<f32, 2>::new(&data, [2, 3]).unwrap();
    let norms = row_norms_l2::<f32, Scalar>(&t).unwrap();
    assert_eq!(norms.len(), 2);
    assert!((norms[0] - 5.0).abs() < 1e-5, "row 0 norm = {}", norms[0]);
    assert!((norms[1] - 5.0).abs() < 1e-5, "row 1 norm = {}", norms[1]);
}

// ---------------------------------------------------------------------------
// Layer normalization
// ---------------------------------------------------------------------------

#[test]
fn test_layer_norm_zero_mean_unit_var() {
    let data = [1.0f32, 2.0, 3.0, 4.0];
    let out = layer_norm::<f32, Scalar>(&data, 1e-5, None, None);
    let mean: f32 = out.iter().sum::<f32>() / out.len() as f32;
    let var: f32 = out.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / out.len() as f32;
    assert!(mean.abs() < 1e-4, "expected zero mean, got {mean}");
    assert!((var - 1.0).abs() < 0.02, "expected unit variance, got {var}");
}

#[test]
fn test_layer_norm_affine_scale() {
    let data = [1.0f32, 2.0, 3.0, 4.0];
    let gamma = [2.0f32; 4];
    let beta  = [1.0f32; 4];
    let out = layer_norm::<f32, Scalar>(&data, 1e-5, Some(&gamma), Some(&beta));
    // After LayerNorm: y = gamma * x_norm + beta
    // Sum of gamma*x_norm = 0 (zero mean) → sum = 4 * beta[0] = 4.0
    let total: f32 = out.iter().sum();
    assert!((total - 4.0).abs() < 1e-3, "expected sum=4.0 (beta contribution), got {total}");
}

#[test]
fn test_layer_norm_inplace_preserves_length() {
    let mut data = [0.5f32, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5];
    layer_norm_inplace::<f32, Scalar>(&mut data, 1e-5, None, None);
    let mean: f32 = data.iter().sum::<f32>() / data.len() as f32;
    assert!(mean.abs() < 1e-4, "expected zero mean, got {mean}");
}

#[test]
fn test_layer_norm_uniform_input() {
    // Uniform input → variance = 0 → output should be near zero (not NaN)
    let data = [3.0f32; 8];
    let out = layer_norm::<f32, Scalar>(&data, 1e-5, None, None);
    for &x in &out {
        assert!(x.is_finite(), "non-finite for uniform input: {x}");
    }
}

// ---------------------------------------------------------------------------
// New zero-copy TensorView methods: transpose_view, col_iter, diag_iter
// ---------------------------------------------------------------------------

#[test]
fn test_transpose_view_zero_copy() {
    let data = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let t = TensorView::<f32, 2>::new(&data, [2, 3]).unwrap(); // row-major [2,3]
    let tT = t.transpose_view(); // ColMajor, shape [3,2]

    assert_eq!(tT.shape(), [3, 2]);
    assert_eq!(tT.strides(), [1, 3]);

    // Original [0][1] = data[1] = 2.0
    // Transposed: [1][0] in ColMajor, offset = 1*1 + 0*3 = 1 → data[1] = 2.0
    assert_eq!(tT.get([1, 0]).unwrap(), 2.0);

    // Original [1][2] = data[5] = 6.0
    // Transposed: [2][1] in ColMajor, offset = 2*1 + 1*3 = 5 → data[5] = 6.0
    assert_eq!(tT.get([2, 1]).unwrap(), 6.0);
}

#[test]
fn test_col_iter_values() {
    let data = [1.0f32, 2.0, 3.0,
                4.0, 5.0, 6.0];
    let t = TensorView::<f32, 2>::new(&data, [2, 3]).unwrap();
    // Column 1: data[1] = 2.0, data[4] = 5.0
    let col: Vec<f32> = t.col_iter(1).unwrap().collect();
    assert_eq!(col, vec![2.0, 5.0]);

    // Column 0: data[0] = 1.0, data[3] = 4.0
    let col0: Vec<f32> = t.col_iter(0).unwrap().collect();
    assert_eq!(col0, vec![1.0, 4.0]);
}

#[test]
fn test_col_iter_out_of_bounds() {
    let data = [1.0f32; 6];
    let t = TensorView::<f32, 2>::new(&data, [2, 3]).unwrap();
    assert!(matches!(t.col_iter(3), Err(TensorError::IndexOutOfBounds)));
}

#[test]
fn test_diag_iter_square() {
    // 3×3 identity diagonal = [1, 1, 1]
    let data = [1.0f32, 0.0, 0.0,
                0.0, 1.0, 0.0,
                0.0, 0.0, 1.0];
    let t = TensorView::<f32, 2>::new(&data, [3, 3]).unwrap();
    let diag: Vec<f32> = t.diag_iter().collect();
    assert_eq!(diag, vec![1.0, 1.0, 1.0]);
}

#[test]
fn test_diag_iter_rectangular() {
    // 2×4 matrix: diagonal length = min(2, 4) = 2
    let data: Vec<f32> = (1..=8).map(|x| x as f32).collect();
    let t = TensorView::<f32, 2>::new(&data, [2, 4]).unwrap();
    let diag: Vec<f32> = t.diag_iter().collect();
    // diag[0] = data[0*5] = data[0] = 1.0
    // diag[1] = data[1*5] = data[5] = 6.0
    assert_eq!(diag, vec![1.0, 6.0]);
}
