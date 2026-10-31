//! Public tensor layout contracts: reachable storage, logical rows and bridges.

use hermes_simd_core::align::Unaligned;
use hermes_simd_core::tensor::{ColMajor, RowMajor, TensorError, TensorView};
use hermes_simd_intrinsics::Scalar;

#[test]
fn explicit_strides_reject_out_of_storage_offsets() {
    let data = [10, 20, 30, 40];
    assert!(matches!(
        TensorView::<i32, 2>::with_strides(&data, [2, 2], [3, 1]),
        Err(TensorError::ShapeMismatch)
    ));
}

#[test]
fn mutable_explicit_strides_reject_out_of_storage_offsets() {
    let mut data = [10, 20, 30, 40];
    assert!(matches!(
        TensorView::<i32, 2, RowMajor, &mut [i32]>::with_strides_mut(&mut data, [2, 2], [3, 1]),
        Err(TensorError::ShapeMismatch)
    ));
    assert_eq!(data, [10, 20, 30, 40]);
}

#[test]
fn unreachable_stride_products_are_rejected_without_overflow() {
    assert!(matches!(
        TensorView::<i32, 2>::with_strides(&[1, 2, 3, 4], [2, 2], [usize::MAX, 1]),
        Err(TensorError::ShapeMismatch)
    ));
}

#[test]
fn oversized_shapes_return_shape_mismatch() {
    let shape = [usize::MAX / 2 + 1, 2];
    assert!(matches!(
        TensorView::<i32, 2>::new(&[], shape),
        Err(TensorError::ShapeMismatch)
    ));
    assert!(matches!(
        TensorView::<i32, 2, RowMajor, &mut [i32]>::new_mut(&mut [], shape),
        Err(TensorError::ShapeMismatch)
    ));
    assert!(matches!(
        TensorView::<i32, 2, ColMajor>::new_col_major(&[], shape),
        Err(TensorError::ShapeMismatch)
    ));
    assert!(matches!(
        TensorView::<i32, 2>::with_strides(&[], shape, [0, 0]),
        Err(TensorError::ShapeMismatch)
    ));
}

#[test]
fn reshape_rejects_cardinality_overflow() {
    let view = TensorView::<i32, 1>::new(&[], [0]).unwrap();
    assert!(matches!(
        view.reshape([usize::MAX / 2 + 1, 2]),
        Err(TensorError::ShapeMismatch)
    ));
}

#[test]
fn rows_preserve_column_major_element_positions() {
    let data = [10, 11, 20, 21, 30, 31];
    let view = TensorView::<i32, 2, ColMajor>::new_col_major(&data, [2, 3]).unwrap();
    let row = view.row_view(1).unwrap();
    assert_eq!(
        [
            row.get([0]).unwrap(),
            row.get([1]).unwrap(),
            row.get([2]).unwrap()
        ],
        [11, 21, 31]
    );
    assert_eq!(row.strides(), [2]);
}

#[test]
fn columns_preserve_column_major_element_positions() {
    let data = [10, 11, 20, 21, 30, 31];
    let view = TensorView::<i32, 2, ColMajor>::new_col_major(&data, [2, 3]).unwrap();
    assert_eq!(view.col_iter(2).unwrap().collect::<Vec<_>>(), [30, 31]);
}

#[test]
fn mutable_rows_update_only_the_selected_strided_elements() {
    let mut data = [10, 11, 20, 21, 30, 31];
    {
        let mut view =
            TensorView::<i32, 2, ColMajor, &mut [i32]>::with_strides_mut(&mut data, [2, 3], [1, 2])
                .unwrap();
        let mut row = view.row_view_mut(1).unwrap();
        row.set([1], 99).unwrap();
    }
    assert_eq!(data, [10, 11, 20, 99, 30, 31]);
}

#[test]
fn mutable_row_iterator_does_not_expose_trailing_storage() {
    let mut data = [0; 9];
    {
        let mut view = TensorView::new_mut(&mut data, [2, 3]).unwrap();
        for row in view.iter_rows_mut().unwrap() {
            row.fill(7);
        }
    }
    assert_eq!(data, [7, 7, 7, 7, 7, 7, 0, 0, 0]);
}

#[test]
fn zero_width_rows_preserve_the_logical_row_count() {
    let mut data: [i32; 0] = [];
    let mut view = TensorView::new_mut(&mut data, [3, 0]).unwrap();
    let rows: Vec<_> = view.iter_rows_mut().unwrap().collect();
    assert_eq!(rows.len(), 3);
    assert!(rows.iter().all(|row| row.is_empty()));
}

#[test]
fn empty_shapes_do_not_overflow_before_the_zero_axis() {
    let view = TensorView::<i32, 3>::new(&[], [usize::MAX, 2, 0]).unwrap();
    assert_eq!(view.num_elements(), 0);
    assert!(view.is_empty());
    let reshaped = view.reshape([usize::MAX, 0, 2]).unwrap();
    assert_eq!(reshaped.num_elements(), 0);
    assert!(matches!(
        reshaped.get([0, 0, 0]),
        Err(TensorError::IndexOutOfBounds)
    ));
}

#[test]
fn singleton_diagonal_does_not_add_unreachable_strides() {
    let view = TensorView::<i32, 2>::with_strides(&[42], [1, 1], [usize::MAX; 2]).unwrap();
    assert_eq!(view.diag_iter().collect::<Vec<_>>(), [42]);
}

#[test]
fn rows_preserve_zero_stride_aliases_of_a_valid_parent() {
    let data = [10, 20, 30, 40];
    let view = TensorView::<i32, 2>::with_strides(&data, [2, 2], [3, 0]).unwrap();
    let row = view.row_view(1).unwrap();
    assert_eq!([row.get([0]).unwrap(), row.get([1]).unwrap()], [40, 40]);
}

#[test]
fn simd_bridge_rejects_noncontiguous_rank_one_views() {
    let view = TensorView::<f32, 1>::with_strides(&[10.0, 99.0, 20.0, 99.0], [2], [2]).unwrap();
    assert!(view.into_simd_view::<Scalar, Unaligned>().is_none());
}

#[test]
fn simd_bridge_preserves_logical_extent() {
    let view = TensorView::<f32, 1>::new(&[10.0, 20.0, 99.0], [2]).unwrap();
    let simd = view.into_simd_view::<Scalar, Unaligned>().unwrap();
    assert_eq!(simd.len(), 2);
    assert_eq!(
        simd.as_slice()
            .iter()
            .map(|x| x.to_bits())
            .collect::<Vec<_>>(),
        [10.0_f32.to_bits(), 20.0_f32.to_bits()]
    );
}

#[test]
fn padded_layout_retains_value_semantics_and_bounds() {
    let data = [10, 11, 99, 20, 21, 99];
    let view = TensorView::<i32, 2>::with_strides(&data, [2, 2], [3, 1]).unwrap();
    assert_eq!(view.get([1, 1]).unwrap(), 21);
    assert_eq!(view.row_view(1).unwrap().get([1]).unwrap(), 21);
    assert_eq!(view.col_iter(1).unwrap().collect::<Vec<_>>(), [11, 21]);
    assert_eq!(view.diag_iter().collect::<Vec<_>>(), [10, 21]);
    assert!(matches!(
        view.get([2, 0]),
        Err(TensorError::IndexOutOfBounds)
    ));
    assert!(matches!(view.reshape([4]), Err(TensorError::NotContiguous)));
}

#[test]
fn scalar_rank_and_empty_projections_retain_their_cardinality() {
    let scalar = TensorView::<i32, 0>::new(&[42], []).unwrap();
    assert_eq!(scalar.num_elements(), 1);
    assert_eq!(scalar.get([]).unwrap(), 42);
    assert!(matches!(
        TensorView::<i32, 0>::new(&[], []),
        Err(TensorError::ShapeMismatch)
    ));

    let empty = TensorView::<i32, 2>::with_strides(&[], [3, 0], [usize::MAX; 2]).unwrap();
    let row = empty.row_view(2).unwrap();
    assert_eq!(row.shape(), [0]);
    assert_eq!(row.num_elements(), 0);
    assert!(row.as_slice().is_empty());

    let batched = TensorView::<i32, 3>::new(&[], [4, usize::MAX, 0]).unwrap();
    let matrix = batched.matrix_at(3).unwrap();
    assert_eq!(matrix.shape(), [usize::MAX, 0]);
    assert_eq!(matrix.num_elements(), 0);
    let mut data: [i32; 0] = [];
    let mut batched = TensorView::new_mut(&mut data, [4, usize::MAX, 0]).unwrap();
    let mut matrix = batched.matrix_at_mut(3).unwrap();
    assert_eq!(matrix.iter_rows_mut().unwrap().take(3).count(), 3);

    let empty = TensorView::<f32, 1>::new(&[42.0], [0]).unwrap();
    assert!(empty.into_simd_view::<Scalar, Unaligned>().is_none());
}

#[test]
fn small_layouts_match_enumerated_storage_coordinates() {
    let data = [10, 20, 30, 40, 50, 60, 70];
    let mut cases = 0;
    for rows in 0..=2 {
        for columns in 0..=2 {
            for row_stride in 0..=3 {
                for column_stride in 0..=3 {
                    let shape = [rows, columns];
                    let strides = [row_stride, column_stride];
                    let offsets: Vec<_> = (0..rows)
                        .flat_map(|i| (0..columns).map(move |j| i * row_stride + j * column_stride))
                        .collect();
                    for len in 0..=data.len() {
                        cases += 1;
                        let expected_valid = offsets.iter().all(|&offset| offset < len);
                        let result =
                            TensorView::<i32, 2>::with_strides(&data[..len], shape, strides);
                        if !expected_valid {
                            assert!(
                                matches!(result, Err(TensorError::ShapeMismatch)),
                                "shape={shape:?}, strides={strides:?}, len={len}"
                            );
                            let mut storage = data;
                            assert!(matches!(
                                TensorView::<i32, 2, RowMajor, &mut [i32]>::with_strides_mut(
                                    &mut storage[..len],
                                    shape,
                                    strides
                                ),
                                Err(TensorError::ShapeMismatch)
                            ));
                            assert_eq!(storage, data);
                            continue;
                        }
                        let view = result.unwrap();
                        assert_eq!(view.num_elements(), offsets.len());
                        for i in 0..rows {
                            let row = view.row_view(i).unwrap();
                            for j in 0..columns {
                                let expected = data[i * row_stride + j * column_stride];
                                assert_eq!(view.get([i, j]).unwrap(), expected);
                                assert_eq!(row.get([j]).unwrap(), expected);
                                assert_eq!(view.transpose_view().get([j, i]).unwrap(), expected);
                                // SAFETY: indices are in shape; the independent enumeration
                                // above also proved this exact offset lies inside storage.
                                assert_eq!(unsafe { view.get_unchecked([i, j]) }, expected);
                            }
                        }
                        for j in 0..columns {
                            let expected: Vec<_> = (0..rows)
                                .map(|i| data[i * row_stride + j * column_stride])
                                .collect();
                            assert_eq!(view.col_iter(j).unwrap().collect::<Vec<_>>(), expected);
                        }
                        let expected_diagonal: Vec<_> = (0..rows.min(columns))
                            .map(|i| data[i * row_stride + i * column_stride])
                            .collect();
                        assert_eq!(view.diag_iter().collect::<Vec<_>>(), expected_diagonal);
                        assert!(matches!(
                            view.row_view(rows),
                            Err(TensorError::IndexOutOfBounds)
                        ));
                        assert!(matches!(
                            view.col_iter(columns),
                            Err(TensorError::IndexOutOfBounds)
                        ));

                        let mut storage = data;
                        let mut expected = data;
                        {
                            let mut view =
                                TensorView::<i32, 2, RowMajor, &mut [i32]>::with_strides_mut(
                                    &mut storage[..len],
                                    shape,
                                    strides,
                                )
                                .unwrap();
                            for i in 0..rows {
                                let mut row = view.row_view_mut(i).unwrap();
                                for j in 0..columns {
                                    let value = 100 + i32::try_from(i * columns + j).unwrap();
                                    row.set([j], value).unwrap();
                                    expected[i * row_stride + j * column_stride] = value;
                                }
                            }
                        }
                        assert_eq!(storage, expected);
                    }
                }
            }
        }
    }
    assert_eq!(cases, 1_152);
}
