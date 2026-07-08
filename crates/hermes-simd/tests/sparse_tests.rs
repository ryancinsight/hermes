use hermes_simd::*;

fn validated<S>(data: S) -> ValidatedData<S>
where
    S: hermes_simd_core::sparse::types::SparseValidate,
{
    ValidatedData::new(data).expect("test sparse fixture must validate")
}

#[test]
fn test_spmv_csr_identity() {
    // 3x3 identity
    let values = [1.0f32, 1.0, 1.0];
    let col_indices = [0i32, 1, 2];
    let row_ptr = [0i32, 1, 2, 3];
    let data = CsrData::new(&values[..], &col_indices[..], &row_ptr[..], 3, 3);
    let x = [5.0f32, 7.0, 11.0];
    let mut y = [0.0f32; 3];
    spmv_csr::<f32>(validated(data), &x, &mut y);
    assert_eq!(y, [5.0, 7.0, 11.0]);
}

#[test]
fn test_spmv_csr_rejects_out_of_range_column() {
    // A column index of 3 is out of range for a 3-column matrix; the SIMD gather
    // would otherwise read out of bounds, so the kernel must reject it.
    let values = [1.0f32, 1.0, 1.0];
    let col_indices = [0i32, 3, 2]; // 3 >= ncols
    let row_ptr = [0i32, 1, 2, 3];
    let data = CsrData::new(&values[..], &col_indices[..], &row_ptr[..], 3, 3);
    assert_eq!(
        ValidatedData::new(data).err(),
        Some(SimdError::IndexOutOfBounds)
    );
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
    spmv_csr::<f32>(validated(data), &x, &mut y);
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
    spmv_bcoo::<f32, 4, 4>(validated(data), &x, &mut y);
    assert_eq!(y, x);
}

#[test]
fn test_blocked_coo_simd_branch_matches_scalar_reference() {
    // `BN = 8` triggers a SIMD branch on common hosts (AVX2 lane=8 → 1×; NEON /
    // scalar lane=4 → 2×), exercising the now-live runtime-dispatched BCOO SIMD
    // path (previously hardcoded to ScalarArch). Dyadic-exact entries keep the
    // SIMD reduction order bit-identical to the sequential reference.
    const BM: usize = 2;
    const BN: usize = 8;
    let nrows = 2usize;
    let ncols = 16usize;
    let nblocks = 2usize;
    // Two 2×8 blocks: block 0 at (row 0, col 0), block 1 at (row 0, col 8).
    let block: Vec<f32> = (0..nblocks * BM * BN)
        .map(|i| ((i % 7) as f32 - 3.0) * 0.5)
        .collect();
    let block_row = [0i32, 0i32];
    let block_col = [0i32, 8i32];
    let x: Vec<f32> = (0..ncols).map(|i| (i % 5) as f32 - 2.0).collect();

    let mut want = vec![0.0f32; nrows];
    for b in 0..nblocks {
        let br = block_row[b] as usize;
        let bc = block_col[b] as usize;
        for i in 0..BM {
            let mut acc = 0.0f32;
            for j in 0..BN {
                acc += block[b * BM * BN + i * BN + j] * x[bc + j];
            }
            want[br + i] += acc;
        }
    }

    let data = BlockedCooData::new(
        &block[..],
        &block_row[..],
        &block_col[..],
        nblocks,
        nrows,
        ncols,
    );
    let mut y = vec![0.0f32; nrows];
    spmv_bcoo::<f32, BM, BN>(validated(data), &x, &mut y);
    assert_eq!(y, want, "BCOO SIMD spmv must match scalar reference");
}

#[test]
fn test_blocked_coo_simd_double_lane_branch_matches_scalar_reference() {
    // `BN = 16` triggers the `BN == lane_count * 2` SIMD branch on an AVX2 host
    // (lane = 8 → 2×) and the `BN == lane_count` branch on an AVX-512 host
    // (lane = 16 → 1×) — the native AVX-512 lane width. Both are real
    // runtime-dispatched SIMD paths through `dispatch_spmv_bcoo`. Dyadic-exact
    // entries keep every partial sum exact in f32, so the SIMD reduction (two
    // half-width products summed lane-wise, then horizontally reduced) is
    // bit-identical to the sequential scalar reference.
    const BM: usize = 2;
    const BN: usize = 16;
    let nrows = 2usize;
    let ncols = 32usize;
    let nblocks = 2usize;
    // Two 2×16 blocks: block 0 at (row 0, col 0), block 1 at (row 0, col 16).
    let block: Vec<f32> = (0..nblocks * BM * BN)
        .map(|i| ((i % 7) as f32 - 3.0) * 0.5)
        .collect();
    let block_row = [0i32, 0i32];
    let block_col = [0i32, 16i32];
    let x: Vec<f32> = (0..ncols).map(|i| (i % 5) as f32 - 2.0).collect();

    let mut want = vec![0.0f32; nrows];
    for b in 0..nblocks {
        let br = block_row[b] as usize;
        let bc = block_col[b] as usize;
        for i in 0..BM {
            let mut acc = 0.0f32;
            for j in 0..BN {
                acc += block[b * BM * BN + i * BN + j] * x[bc + j];
            }
            want[br + i] += acc;
        }
    }

    let data = BlockedCooData::new(
        &block[..],
        &block_row[..],
        &block_col[..],
        nblocks,
        nrows,
        ncols,
    );
    let mut y = vec![0.0f32; nrows];
    spmv_bcoo::<f32, BM, BN>(validated(data), &x, &mut y);
    assert_eq!(
        y, want,
        "BCOO double-lane SIMD spmv must match scalar reference"
    );
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

    let view = SparseView::<f32, Validated<SellP<4>>, Scalar>::try_from_sellp(data).unwrap();
    view.spmv(&x, &mut y);

    assert_eq!(y, [10.0, 20.0, 30.0, 40.0]);
}

#[test]
fn test_sellp_spmv_rejects_out_of_range_column() {
    // Column index 4 is out of range for a 4-column matrix. The vectorized SELL-p
    // path (engaged because `Scalar::LANE_COUNT == 4 == C`) gathers `x[col]`
    // without a per-lane bounds check, so it must reject the matrix up front
    // rather than read out of bounds. Regression for the SELL-p gather OOB.
    let values = [1.0f32, 2.0, 3.0, 4.0];
    let col_indices_bad = [0i32, 1, 4, 3];
    let slice_ptr = [0i32, 4];
    let slice_col_count = [1i32];
    let data = SellPData::new(
        &values[..],
        &col_indices_bad[..],
        &slice_ptr[..],
        &slice_col_count[..],
        4,
        4,
    );
    assert_eq!(
        SparseView::<f32, Validated<SellP<4>>, Scalar>::try_from_sellp(data).err(),
        Some(SimdError::IndexOutOfBounds)
    );
}

#[test]
fn test_sellp_spmv_rejects_bad_slice_geometry() {
    // `slice_col_count = 2` with `C = 4` and only 4 values makes the second
    // column's load `values[4..8]` read past the 4-element array. The vectorized
    // load is full-width and unchecked, so the matrix must be rejected. Regression
    // for the SELL-p `values[offset..]` over-read.
    let values = [1.0f32, 2.0, 3.0, 4.0];
    let col_indices = [0i32, 1, 2, 3];
    let slice_ptr = [0i32, 4];
    let slice_col_count = [2i32]; // 0 + 2*4 = 8 > values.len() == 4
    let data = SellPData::new(
        &values[..],
        &col_indices[..],
        &slice_ptr[..],
        &slice_col_count[..],
        8,
        4,
    );
    assert_eq!(
        SparseView::<f32, Validated<SellP<4>>, Scalar>::try_from_sellp(data).err(),
        Some(SimdError::LengthMismatch)
    );
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

    spmv_sellp::<f32, 4>(validated(data.clone()), &x, &mut y);
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
    spmv_sellp::<f32, 8>(validated(data8), &x, &mut y8);
    assert_eq!(y8, [10.0, 20.0, 30.0, 40.0]);
}

/// Chunk-width-aware dispatch differential: a two-slice SELL-8 matrix with
/// per-slice padding and non-uniform values must match an independent dense
/// reference through the public dispatcher, whichever ISA the ladder picks
/// (on an AVX2 host C = 8 engages the AVX2 kernel; C = 4 routes to the
/// lane-matching scalar marker instead of the per-element fallback).
#[test]
fn test_sellp_spmv_dispatch_multislice_differential() {
    // 16 rows (2 slices of C = 8), 8 columns; row r has entries at columns
    // (r % 8) and ((r + 3) % 8) with values derived from r.
    const C: usize = 8;
    let nrows = 16usize;
    let ncols = 8usize;
    let cols_per_row = 2usize;

    // SELL-p layout: slice s columns stored column-major within the slice.
    let mut values = vec![0.0f32; nrows * cols_per_row];
    let mut col_indices = vec![0i32; nrows * cols_per_row];
    for s in 0..2 {
        for col in 0..cols_per_row {
            for row in 0..C {
                let r = s * C + row;
                let idx = s * (cols_per_row * C) + col * C + row;
                let c = (r + 3 * col) % ncols;
                values[idx] = (r * cols_per_row + col) as f32 * 0.5 - 3.0;
                col_indices[idx] = c as i32;
            }
        }
    }
    let slice_ptr = [
        0i32,
        (cols_per_row * C) as i32,
        (2 * cols_per_row * C) as i32,
    ];
    let slice_col_count = [cols_per_row as i32, cols_per_row as i32];
    let data = SellPData::<f32, C>::new(
        &values,
        &col_indices,
        &slice_ptr[..],
        &slice_col_count[..],
        nrows,
        ncols,
    );

    let x: Vec<f32> = (0..ncols).map(|i| i as f32 + 0.25).collect();
    let mut y = vec![1.0f32; nrows];
    spmv_sellp::<f32, C>(validated(data), &x, &mut y);

    // Independent dense reference: y_ref = 1 + Σ values[r,·]·x[col].
    let mut y_ref = vec![1.0f32; nrows];
    for (r, y_r) in y_ref.iter_mut().enumerate() {
        let s = r / C;
        let row = r % C;
        for col in 0..cols_per_row {
            let idx = s * (cols_per_row * C) + col * C + row;
            *y_r += values[idx] * x[col_indices[idx] as usize];
        }
    }
    assert_eq!(y, y_ref, "SELL-8 dispatch diverges from dense reference");
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
    let cow: SparseCow<f32, Validated<Csr>, Scalar> = SparseCow::try_borrowed(data).unwrap();

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
    let cow: SparseCow<f32, Validated<Csr>, Scalar> = SparseCow::try_borrowed(data).unwrap();

    let x = [2.0f32, 3.0, 5.0];
    let mut y = [0.0f32; 3];
    cow.spmv(&x, &mut y);
    assert_eq!(y, [2.0, 3.0, 5.0]);
}

#[test]
fn test_csr_cow_spmv_owned() {
    let (vals, cols, row_ptr) = make_csr_3x3_identity();
    let cow = SparseCow::<f32, Validated<Csr>, Scalar>::from_slices(&vals, &cols, &row_ptr, 3, 3)
        .unwrap();

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
    let mut cow: SparseCow<f32, Validated<Csr>, Scalar> = SparseCow::try_borrowed(data).unwrap();

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
    let mut cow =
        SparseCow::<f32, Validated<Csr>, Scalar>::from_slices(&vals, &cols, &row_ptr, 3, 3)
            .unwrap();
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
    let cow = SparseCow::<f32, Csr, Scalar>::from_slices(&vals, &cols, &row_ptr, 3, 3);
    let s = cow.sum_values();
    assert!((s - 8.0f32).abs() < 1e-6);
}

#[test]
fn test_csr_cow_elementwise_mul_dense() {
    let vals = vec![2.0f32, 3.0, 4.0];
    let cols = vec![0i32, 1, 2];
    let row_ptr = vec![0i32, 1, 2, 3];
    let cow = SparseCow::<f32, Csr, Scalar>::from_slices(&vals, &cols, &row_ptr, 3, 3);
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
    let cow: SparseCow<f32, Validated<SellP<4>>, Scalar> = SparseCow::try_borrowed(data).unwrap();

    assert!(cow.is_borrowed());
    let x = [10.0f32; 4];
    let mut y = [0.0f32; 4];
    cow.spmv(&x, &mut y);
    assert_eq!(y, [10.0, 20.0, 30.0, 40.0]);
}

#[test]
fn test_sellp_cow_owned_spmv() {
    let cow = SparseCow::<f32, Validated<SellP<4>>, Scalar>::from_slices(
        &[1.0f32, 2.0, 3.0, 4.0],
        &[0i32, 1, 2, 3],
        &[0i32, 4],
        &[1i32],
        4,
        4,
    )
    .unwrap();
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
    let mut cow: SparseCow<f32, Validated<SellP<4>>, Scalar> =
        SparseCow::try_borrowed(data).unwrap();
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
    let cow = SparseCow::<f32, DenseWithMask, Scalar>::from_slices(
        &[1.0f32, 0.0, 0.0, 1.0],
        &[true, false, false, true],
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
    let cow = SparseCow::<f32, DenseWithMask, Scalar>::from_slices(
        &[3.0f32, 0.0, 0.0, 5.0],
        &[true, false, false, true],
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
    let cow: SparseCow<f32, Validated<BlockedCoo<4, 4>>, Scalar> =
        SparseCow::try_borrowed(data).unwrap();

    assert!(cow.is_borrowed());
    let x = [1.0f32, 2.0, 3.0, 4.0];
    let mut y = [0.0f32; 4];
    cow.spmv(&x, &mut y);
    assert_eq!(y, x);
}

#[test]
fn test_bcoo_cow_owned_spmv() {
    let cow = SparseCow::<f32, Validated<BlockedCoo<4, 4>>, Scalar>::from_slices(
        &[
            1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
        ],
        &[0i32],
        &[0i32],
        1,
        4,
        4,
    )
    .unwrap();
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
    let mut cow: SparseCow<f32, Validated<BlockedCoo<4, 4>>, Scalar> =
        SparseCow::try_borrowed(data).unwrap();
    assert!(cow.is_borrowed());
    cow.to_owned();
    assert!(cow.is_owned());

    let x = [1.0f32, 2.0, 3.0, 4.0];
    let mut y = [0.0f32; 4];
    cow.spmv(&x, &mut y);
    assert_eq!(y, [2.0, 4.0, 6.0, 8.0]);
}

#[test]
fn test_sparse_validate_csr_bounds() {
    use hermes_simd_core::sparse::types::SparseValidate;

    // Normal valid CSR
    let values = [1.0f32, 2.0];
    let col_indices = [0i32, 1];
    let row_ptr = [0i32, 1, 2];
    let data = CsrData::new(&values[..], &col_indices[..], &row_ptr[..], 2, 2);
    assert!(data.validate().is_ok());

    // Out of bounds column index
    let col_indices_bad = [0i32, 2]; // ncols is 2, index 2 is out of bounds
    let data_bad_col = CsrData::new(&values[..], &col_indices_bad[..], &row_ptr[..], 2, 2);
    assert_eq!(data_bad_col.validate(), Err(SimdError::IndexOutOfBounds));

    // Negative column index
    let col_indices_neg = [-1i32, 1];
    let data_neg_col = CsrData::new(&values[..], &col_indices_neg[..], &row_ptr[..], 2, 2);
    assert_eq!(data_neg_col.validate(), Err(SimdError::IndexOutOfBounds));

    // Mismatched length between values and col_indices
    let col_indices_short = [0i32];
    let data_short_col = CsrData::new(&values[..], &col_indices_short[..], &row_ptr[..], 2, 2);
    assert_eq!(data_short_col.validate(), Err(SimdError::LengthMismatch));

    // Mismatched length of row_ptr (needs nrows + 1)
    let row_ptr_short = [0i32, 1];
    let data_short_row = CsrData::new(&values[..], &col_indices[..], &row_ptr_short[..], 2, 2);
    assert_eq!(data_short_row.validate(), Err(SimdError::LengthMismatch));

    // row_ptr does not start at 0
    let row_ptr_not_zero = [1i32, 1, 2];
    let data_not_zero_row =
        CsrData::new(&values[..], &col_indices[..], &row_ptr_not_zero[..], 2, 2);
    assert_eq!(
        data_not_zero_row.validate(),
        Err(SimdError::IndexOutOfBounds)
    );

    // last row_ptr does not match values length
    let row_ptr_bad_last = [0i32, 1, 3];
    let data_bad_last = CsrData::new(&values[..], &col_indices[..], &row_ptr_bad_last[..], 2, 2);
    assert_eq!(data_bad_last.validate(), Err(SimdError::LengthMismatch));
}

#[test]
fn test_sparse_validate_sellp_bounds() {
    use hermes_simd_core::sparse::types::SparseValidate;

    // Normal valid SELL-p
    let values = [1.0f32, 2.0, 3.0, 4.0];
    let col_indices = [0i32, 1, 2, 3];
    let slice_ptr = [0i32, 4];
    let slice_col_count = [1i32];
    let data = SellPData::<f32, 4>::new(
        &values[..],
        &col_indices[..],
        &slice_ptr[..],
        &slice_col_count[..],
        4,
        4,
    );
    assert!(data.validate().is_ok());

    // Bad col index
    let col_indices_bad = [0i32, 1, 4, 3]; // ncols is 4, 4 is out of bounds
    let data_bad_col = SellPData::<f32, 4>::new(
        &values[..],
        &col_indices_bad[..],
        &slice_ptr[..],
        &slice_col_count[..],
        4,
        4,
    );
    assert_eq!(data_bad_col.validate(), Err(SimdError::IndexOutOfBounds));
}

#[test]
fn test_sparse_validate_bcoo_bounds() {
    use hermes_simd_core::sparse::types::SparseValidate;

    let block: Vec<f32> = vec![
        1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
    ];
    let block_row = [0i32];
    let block_col = [0i32];
    let data = BlockedCooData::<f32, 4, 4>::new(&block, &block_row, &block_col, 1, 4, 4);
    assert!(data.validate().is_ok());

    // Out of bounds block row
    let block_row_bad = [2i32]; // 2 + BM (4) = 6 > nrows (4)
    let data_bad_row =
        BlockedCooData::<f32, 4, 4>::new(&block, &block_row_bad[..], &block_col, 1, 4, 4);
    assert_eq!(data_bad_row.validate(), Err(SimdError::IndexOutOfBounds));

    // Out of bounds block col
    let block_col_bad = [1i32]; // 1 + BN (4) = 5 > ncols (4)
    let data_bad_col =
        BlockedCooData::<f32, 4, 4>::new(&block, &block_row, &block_col_bad[..], 1, 4, 4);
    assert_eq!(data_bad_col.validate(), Err(SimdError::IndexOutOfBounds));
}

mod sparse_validated_properties {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #[test]
        fn prop_validated_csr_single_entry_spmv_matches_reference(
            ncols in 1usize..16,
            col in 0usize..16,
            value in -32i32..32,
            x_value in -32i32..32,
        ) {
            let col = col % ncols;
            let values = [value as f32];
            let cols = [col as i32];
            let row_ptr = [0i32, 1];
            let mut x = vec![0.0f32; ncols];
            x[col] = x_value as f32;
            let mut y = [0.0f32; 1];

            let data = CsrData::new(&values, &cols, &row_ptr, 1, ncols);
            spmv_csr::<f32>(validated(data), &x, &mut y);

            prop_assert_eq!(y[0], values[0] * x[col]);
        }

        #[test]
        fn prop_validated_csr_rejects_generated_bad_column(
            ncols in 1usize..16,
            extra in 0usize..16,
        ) {
            let values = [1.0f32];
            let cols = [(ncols + extra) as i32];
            let row_ptr = [0i32, 1];
            let data = CsrData::new(&values, &cols, &row_ptr, 1, ncols);

            prop_assert_eq!(ValidatedData::new(data).err(), Some(SimdError::IndexOutOfBounds));
        }

        #[test]
        fn prop_validated_sellp_single_slice_spmv_matches_reference(
            values_i in prop::array::uniform4(-16i32..16),
            x_i in prop::array::uniform4(-16i32..16),
            cols_raw in prop::array::uniform4(0usize..4),
        ) {
            let values = values_i.map(|v| v as f32);
            let x = x_i.map(|v| v as f32);
            let cols = cols_raw.map(|c| c as i32);
            let slice_ptr = [0i32, 4];
            let slice_col_count = [1i32];
            let data = SellPData::<f32, 4>::new(
                &values,
                &cols,
                &slice_ptr,
                &slice_col_count,
                4,
                4,
            );
            let mut y = [0.0f32; 4];

            spmv_sellp::<f32, 4>(validated(data), &x, &mut y);

            for row in 0..4 {
                prop_assert_eq!(y[row], values[row] * x[cols_raw[row]]);
            }
        }

        #[test]
        fn prop_validated_bcoo_single_block_spmv_matches_reference(
            block_i in prop::array::uniform4(-8i32..8),
            x_i in prop::array::uniform4(-8i32..8),
            br in 0usize..3,
            bc in 0usize..3,
        ) {
            let block = block_i.map(|v| v as f32);
            let x = x_i.map(|v| v as f32);
            let block_row = [br as i32];
            let block_col = [bc as i32];
            let data = BlockedCooData::<f32, 2, 2>::new(
                &block,
                &block_row,
                &block_col,
                1,
                4,
                4,
            );
            let mut y = [0.0f32; 4];

            spmv_bcoo::<f32, 2, 2>(validated(data), &x, &mut y);

            let mut want = [0.0f32; 4];
            for row in 0..2 {
                for col in 0..2 {
                    want[br + row] += block[row * 2 + col] * x[bc + col];
                }
            }
            prop_assert_eq!(y, want);
        }
    }
}
