use hermes_simd::*;

#[test]
fn test_spmv_csr_identity() {
    // 3x3 identity
    let values = [1.0f32, 1.0, 1.0];
    let col_indices = [0i32, 1, 2];
    let row_ptr = [0i32, 1, 2, 3];
    let data = CsrData::new(&values[..], &col_indices[..], &row_ptr[..], 3, 3);
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
    let data = CsrData::new(&values[..], &col_indices[..], &row_ptr[..], 2, 2);
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
    let data = DenseWithMaskData::new(&values[..], &mask_bits[..], 2, 2);
    let x = [6.0f32, 9.0];
    let mut y = [0.0f32; 2];
    spmv_dense_masked::<f32>(data, &x, &mut y);
    assert_eq!(y, [6.0, 9.0]);
}

#[test]
fn test_blocked_coo_4x4_spmv() {
    // 4x4 identity matrix as one 4x4 block
    let block: Vec<f32> = vec![
        1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
    ];
    let block_row = [0i32];
    let block_col = [0i32];
    let data = BlockedCooData::new(&block[..], &block_row[..], &block_col[..], 1, 4, 4);
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
    let data = SellPData::new(
        &values[..],
        &col_indices[..],
        &slice_ptr[..],
        &slice_col_count[..],
        4,
        4,
    );
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
    let data = SellPData::new(
        &values[..],
        &col_indices[..],
        &slice_ptr[..],
        &slice_col_count[..],
        4,
        4,
    );
    let x = [10.0f32, 10.0, 10.0, 10.0];
    let mut y = [0.0f32; 4];

    spmv_sellp4::<f32>(data.clone(), &x, &mut y);
    assert_eq!(y, [10.0, 20.0, 30.0, 40.0]);

    let mut y8 = [0.0f32; 4];
    let data8 = SellPData::new(
        &[1.0, 2.0, 3.0, 4.0, 0.0, 0.0, 0.0, 0.0][..],
        &[0, 1, 2, 3, 0, 0, 0, 0][..],
        &[0, 8][..],
        &[1][..],
        4,
        4,
    );
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

#[test]
fn test_unpack_int4_signed_nibble_domain() {
    let packed = [0x10u8, 0x32, 0x54, 0x76, 0x98, 0xBA, 0xDC, 0xFE];
    let mut unpacked = [0i8; 16];
    unpack_int4(&packed, &mut unpacked);
    assert_eq!(
        unpacked,
        [0, 1, 2, 3, 4, 5, 6, 7, -8, -7, -6, -5, -4, -3, -2, -1]
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// SparseCow tests
// ─────────────────────────────────────────────────────────────────────────────

fn make_csr_3x3_identity() -> (Vec<f32>, Vec<i32>, Vec<i32>) {
    (
        vec![1.0f32, 1.0, 1.0],
        vec![0i32, 1, 2],
        vec![0i32, 1, 2, 3],
    )
}

#[test]
fn test_csr_cow_borrowed_is_zero_alloc() {
    let (vals, cols, row_ptr) = make_csr_3x3_identity();
    let data = CsrData::new(&vals, &cols, &row_ptr, 3, 3);
    let cow: SparseCow<f32, Csr, Scalar> = SparseCow::borrowed(data);

    // Borrowed variant: no allocation, dimensions correct.
    assert!(cow.is_borrowed());
    assert!(!cow.is_owned());
    assert_eq!(cow.nrows(), 3);
    assert_eq!(cow.ncols(), 3);
}

#[test]
fn test_csr_cow_spmv_borrowed() {
    let (vals, cols, row_ptr) = make_csr_3x3_identity();
    let data = CsrData::new(&vals, &cols, &row_ptr, 3, 3);
    let cow: SparseCow<f32, Csr, Scalar> = SparseCow::borrowed(data);

    let x = [2.0f32, 3.0, 5.0];
    let mut y = [0.0f32; 3];
    cow.spmv(&x, &mut y);
    assert_eq!(y, [2.0, 3.0, 5.0]);
}

#[test]
fn test_csr_cow_spmv_owned() {
    let (vals, cols, row_ptr) = make_csr_3x3_identity();
    let cow = SparseCow::<f32, Csr, Scalar>::from_vecs(vals, cols, row_ptr, 3, 3);

    assert!(cow.is_owned());

    let x = [2.0f32, 3.0, 5.0];
    let mut y = [0.0f32; 3];
    cow.spmv(&x, &mut y);
    assert_eq!(y, [2.0, 3.0, 5.0]);
}

#[test]
fn test_csr_cow_to_owned_promotes_borrowed() {
    let (vals, cols, row_ptr) = make_csr_3x3_identity();
    let data = CsrData::new(&vals, &cols, &row_ptr, 3, 3);
    let mut cow: SparseCow<f32, Csr, Scalar> = SparseCow::borrowed(data);

    assert!(cow.is_borrowed());
    cow.to_owned();
    assert!(cow.is_owned());

    // SpMV still correct after promotion.
    let x = [7.0f32, 11.0, 13.0];
    let mut y = [0.0f32; 3];
    cow.spmv(&x, &mut y);
    assert_eq!(y, [7.0, 11.0, 13.0]);
}

#[test]
fn test_csr_cow_to_owned_noop_when_already_owned() {
    let (vals, cols, row_ptr) = make_csr_3x3_identity();
    let mut cow = SparseCow::<f32, Csr, Scalar>::from_vecs(vals, cols, row_ptr, 3, 3);
    assert!(cow.is_owned());
    cow.to_owned(); // must not panic or reallocate
    assert!(cow.is_owned());
}

#[test]
fn test_csr_cow_sum_values_borrowed() {
    let (vals, cols, row_ptr) = make_csr_3x3_identity();
    let data = CsrData::new(&vals, &cols, &row_ptr, 3, 3);
    let cow: SparseCow<f32, Csr, Scalar> = SparseCow::borrowed(data);
    // 3x1.0 = 3.0
    let s = cow.sum_values();
    assert!((s - 3.0f32).abs() < 1e-6);
}

#[test]
fn test_csr_cow_sum_values_owned() {
    let vals = vec![2.0f32, 5.0, 1.0];
    let cols = vec![0i32, 1, 2];
    let row_ptr = vec![0i32, 1, 2, 3];
    let cow = SparseCow::<f32, Csr, Scalar>::from_vecs(vals, cols, row_ptr, 3, 3);
    let s = cow.sum_values();
    assert!((s - 8.0f32).abs() < 1e-6);
}

#[test]
fn test_csr_cow_elementwise_mul_dense() {
    let vals = vec![2.0f32, 3.0, 4.0];
    let cols = vec![0i32, 1, 2];
    let row_ptr = vec![0i32, 1, 2, 3];
    let cow = SparseCow::<f32, Csr, Scalar>::from_vecs(vals, cols, row_ptr, 3, 3);
    // Dense: col 0 → 10, col 1 → 20, col 2 → 30
    let dense = [10.0f32, 20.0, 30.0];
    let mut out = [0.0f32; 3];
    cow.elementwise_mul_dense(&dense, &mut out);
    // out[i] = values[i] * dense[col_indices[i]]
    assert!((out[0] - 20.0f32).abs() < 1e-6); // 2.0 * 10
    assert!((out[1] - 60.0f32).abs() < 1e-6); // 3.0 * 20
    assert!((out[2] - 120.0f32).abs() < 1e-6); // 4.0 * 30
}

// ─────────────────────────────────────────────────────────────────────────────
// SellPCow tests
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_sellp_cow_borrowed_spmv() {
    let values = [1.0f32, 2.0, 3.0, 4.0];
    let col_indices = [0i32, 1, 2, 3];
    let slice_ptr = [0i32, 4];
    let slice_col_count = [1i32];
    let data = SellPData::new(&values, &col_indices, &slice_ptr, &slice_col_count, 4, 4);
    let cow: SparseCow<f32, SellP<4>, Scalar> = SparseCow::borrowed(data);

    assert!(cow.is_borrowed());
    let x = [10.0f32; 4];
    let mut y = [0.0f32; 4];
    cow.spmv(&x, &mut y);
    assert_eq!(y, [10.0, 20.0, 30.0, 40.0]);
}

#[test]
fn test_sellp_cow_owned_spmv() {
    let cow = SparseCow::<f32, SellP<4>, Scalar>::from_vecs(
        vec![1.0f32, 2.0, 3.0, 4.0],
        vec![0i32, 1, 2, 3],
        vec![0i32, 4],
        vec![1i32],
        4,
        4,
    );
    assert!(cow.is_owned());
    let x = [10.0f32; 4];
    let mut y = [0.0f32; 4];
    cow.spmv(&x, &mut y);
    assert_eq!(y, [10.0, 20.0, 30.0, 40.0]);
}

#[test]
fn test_sellp_cow_to_owned_promotes() {
    let values = [1.0f32, 2.0, 3.0, 4.0];
    let col_indices = [0i32, 1, 2, 3];
    let slice_ptr = [0i32, 4];
    let slice_col_count = [1i32];
    let data = SellPData::new(&values, &col_indices, &slice_ptr, &slice_col_count, 4, 4);
    let mut cow: SparseCow<f32, SellP<4>, Scalar> = SparseCow::borrowed(data);
    assert!(cow.is_borrowed());
    cow.to_owned();
    assert!(cow.is_owned());

    let x = [10.0f32; 4];
    let mut y = [0.0f32; 4];
    cow.spmv(&x, &mut y);
    assert_eq!(y, [10.0, 20.0, 30.0, 40.0]);
}

// ─────────────────────────────────────────────────────────────────────────────
// DenseWithMaskCow tests
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_dense_masked_cow_borrowed_spmv() {
    let values = [1.0f32, 0.0, 0.0, 1.0]; // 2x2 identity
    let mask = [true, false, false, true];
    let data = DenseWithMaskData::new(&values, &mask, 2, 2);
    let cow: SparseCow<f32, DenseWithMask, Scalar> = SparseCow::borrowed(data);

    assert!(cow.is_borrowed());
    let x = [6.0f32, 9.0];
    let mut y = [0.0f32; 2];
    cow.spmv(&x, &mut y);
    assert_eq!(y, [6.0, 9.0]);
}

#[test]
fn test_dense_masked_cow_owned_spmv() {
    let cow = SparseCow::<f32, DenseWithMask, Scalar>::from_vecs(
        vec![1.0f32, 0.0, 0.0, 1.0],
        vec![true, false, false, true],
        2,
        2,
    );
    assert!(cow.is_owned());
    let x = [6.0f32, 9.0];
    let mut y = [0.0f32; 2];
    cow.spmv(&x, &mut y);
    assert_eq!(y, [6.0, 9.0]);
}

#[test]
fn test_dense_masked_cow_sum_values() {
    let cow = SparseCow::<f32, DenseWithMask, Scalar>::from_vecs(
        vec![3.0f32, 0.0, 0.0, 5.0],
        vec![true, false, false, true],
        2,
        2,
    );
    let s = cow.sum_values();
    assert!((s - 8.0f32).abs() < 1e-6);
}

// ─────────────────────────────────────────────────────────────────────────────
// BlockedCooCow tests
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_bcoo_cow_borrowed_spmv() {
    let block: Vec<f32> = vec![
        1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
    ];
    let block_row = [0i32];
    let block_col = [0i32];
    let data = BlockedCooData::new(&block, &block_row, &block_col, 1, 4, 4);
    let cow: SparseCow<f32, BlockedCoo<4, 4>, Scalar> = SparseCow::borrowed(data);

    assert!(cow.is_borrowed());
    let x = [1.0f32, 2.0, 3.0, 4.0];
    let mut y = [0.0f32; 4];
    cow.spmv(&x, &mut y);
    assert_eq!(y, x);
}

#[test]
fn test_bcoo_cow_owned_spmv() {
    let cow = SparseCow::<f32, BlockedCoo<4, 4>, Scalar>::from_vecs(
        vec![
            1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
        ],
        vec![0i32],
        vec![0i32],
        1,
        4,
        4,
    );
    assert!(cow.is_owned());
    let x = [5.0f32, 6.0, 7.0, 8.0];
    let mut y = [0.0f32; 4];
    cow.spmv(&x, &mut y);
    assert_eq!(y, x);
}

#[test]
fn test_bcoo_cow_to_owned_promotes() {
    let block: Vec<f32> = vec![
        2.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 2.0,
    ];
    let block_row = [0i32];
    let block_col = [0i32];
    let data = BlockedCooData::new(&block, &block_row, &block_col, 1, 4, 4);
    let mut cow: SparseCow<f32, BlockedCoo<4, 4>, Scalar> = SparseCow::borrowed(data);
    assert!(cow.is_borrowed());
    cow.to_owned();
    assert!(cow.is_owned());

    let x = [1.0f32, 2.0, 3.0, 4.0];
    let mut y = [0.0f32; 4];
    cow.spmv(&x, &mut y);
    assert_eq!(y, [2.0, 4.0, 6.0, 8.0]);
}
