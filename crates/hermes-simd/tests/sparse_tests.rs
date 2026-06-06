use hermes_simd::*;

#[test]
fn test_spmv_csr_identity() {
    // 3x3 identity
    let values = [1.0f32, 1.0, 1.0];
    let col_indices = [0i32, 1, 2];
    let row_ptr = [0i32, 1, 2, 3];
    let data = CsrData {
        values: &values,
        col_indices: &col_indices,
        row_ptr: &row_ptr,
        nrows: 3,
        ncols: 3,
    };
    let x = [5.0f32, 7.0, 11.0];
    let mut y = [0.0f32; 3];
    spmv_csr::<f32>(data, &x, &mut y);
    assert_eq!(y, [5.0, 7.0, 11.0]);
}

#[test]
fn test_spmv_csr_accumulates() {
    // 2x2 all-ones matrix
    let values = [1.0f32, 1.0, 1.0, 1.0];
    let col_indices = [0i32, 1, 0, 1];
    let row_ptr = [0i32, 2, 4];
    let data = CsrData {
        values: &values,
        col_indices: &col_indices,
        row_ptr: &row_ptr,
        nrows: 2,
        ncols: 2,
    };
    let x = [3.0f32, 4.0];
    let mut y = [1.0f32; 2]; // y starts at 1.0 to test accumulation
    spmv_csr::<f32>(data, &x, &mut y);
    // y[0] = 1 + (1*3 + 1*4) = 8; y[1] = 1 + (1*3 + 1*4) = 8
    assert_eq!(y, [8.0, 8.0]);
}

#[test]
fn test_spmv_dense_masked() {
    let values = [1.0f32, 0.0, 0.0, 1.0]; // 2x2 identity
    let mask_bits = [true, false, false, true];
    let data = DenseWithMaskData {
        values: &values,
        mask: &mask_bits,
        nrows: 2,
        ncols: 2,
    };
    let x = [6.0f32, 9.0];
    let mut y = [0.0f32; 2];
    spmv_dense_masked::<f32>(data, &x, &mut y);
    assert_eq!(y, [6.0, 9.0]);
}

#[test]
fn test_blocked_coo_4x4_spmv() {
    // 4x4 identity matrix as one 4x4 block
    let block: Vec<f32> = vec![
        1.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0,
        0.0, 0.0, 0.0, 1.0,
    ];
    let block_row = [0i32];
    let block_col = [0i32];
    let data = BlockedCooData {
        blocks: &block,
        block_row: &block_row,
        block_col: &block_col,
        nblocks: 1,
        nrows: 4,
        ncols: 4,
    };
    let x = [2.0f32, 3.0, 5.0, 7.0];
    let mut y = [0.0f32; 4];
    spmv_bcoo4x4::<f32>(data, &x, &mut y);
    assert_eq!(y, x);
}

#[test]
fn test_sellp_spmv_correctness() {
    let values = [1.0f32, 2.0, 3.0, 4.0];
    let col_indices = [0i32, 1, 2, 3];
    let slice_ptr = [0i32, 4];
    let slice_col_count = [1i32];
    let data = SellPData {
        values: &values,
        col_indices: &col_indices,
        slice_ptr: &slice_ptr,
        slice_col_count: &slice_col_count,
        nrows: 4,
        ncols: 4,
    };
    let x = [10.0f32, 10.0, 10.0, 10.0];
    let mut y = [0.0f32; 4];
    
    let view = SparseView::<f32, SellP<4>, Scalar>::from_sellp4(data);
    view.spmv(&x, &mut y);
    
    assert_eq!(y, [10.0, 20.0, 30.0, 40.0]);
}

#[test]
fn test_sellp_spmv_dispatch() {
    let values = [1.0f32, 2.0, 3.0, 4.0];
    let col_indices = [0i32, 1, 2, 3];
    let slice_ptr = [0i32, 4];
    let slice_col_count = [1i32];
    let data = SellPData {
        values: &values,
        col_indices: &col_indices,
        slice_ptr: &slice_ptr,
        slice_col_count: &slice_col_count,
        nrows: 4,
        ncols: 4,
    };
    let x = [10.0f32, 10.0, 10.0, 10.0];
    let mut y = [0.0f32; 4];
    
    spmv_sellp4::<f32>(data.clone(), &x, &mut y);
    assert_eq!(y, [10.0, 20.0, 30.0, 40.0]);

    let mut y8 = [0.0f32; 4];
    let data8 = SellPData {
        values: &[1.0, 2.0, 3.0, 4.0, 0.0, 0.0, 0.0, 0.0],
        col_indices: &[0, 1, 2, 3, 0, 0, 0, 0],
        slice_ptr: &[0, 8],
        slice_col_count: &[1],
        nrows: 4,
        ncols: 4,
    };
    spmv_sellp8::<f32>(data8, &x, &mut y8);
    assert_eq!(y8, [10.0, 20.0, 30.0, 40.0]);
}

#[test]
fn test_unpack_int4() {
    let packed = [0xABu8];
    let mut unpacked = [0i8; 2];
    unpack_int4(&packed, &mut unpacked);
    assert_eq!(unpacked[0], -5);
    assert_eq!(unpacked[1], -6);
}
