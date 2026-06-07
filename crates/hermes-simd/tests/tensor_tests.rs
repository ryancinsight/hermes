//! Integration tests for `TensorView`/`TensorCow` layout primitives and histogram.

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
// Histogram via SimdCow::histogram_cow
// ---------------------------------------------------------------------------

#[test]
fn test_histogram_cow_uniform() {
    // 100 values uniformly in [0, 10): should give 10 elements per bin
    let data: Vec<f32> = (0..100).map(|i| i as f32 % 10.0).collect();
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
fn test_transpose_view_zero_copy() {
    let data = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let t = TensorView::<f32, 2>::new(&data, [2, 3]).unwrap(); // row-major [2,3]
    let t_t = t.transpose_view(); // ColMajor, shape [3,2]

    assert_eq!(t_t.shape(), [3, 2]);
    assert_eq!(t_t.strides(), [1, 3]);

    // Original [0][1] = data[1] = 2.0
    // Transposed: [1][0] in ColMajor, offset = 1*1 + 0*3 = 1 → data[1] = 2.0
    assert_eq!(t_t.get([1, 0]).unwrap(), 2.0);

    // Original [1][2] = data[5] = 6.0
    // Transposed: [2][1] in ColMajor, offset = 2*1 + 1*3 = 5 → data[5] = 6.0
    assert_eq!(t_t.get([2, 1]).unwrap(), 6.0);
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



