//! Sparse SpMV through the validated-data typestate.
//!
//! A CSR matrix is built from raw parts, validated by `ValidatedData::new`
//! (structural checks only — index bounds are checked before any SIMD
//! gather can read out of bounds), and then consumed by `spmv_csr`.

use hermes_simd::{
    spmv_csr, spmv_dense_masked, CsrData, DenseWithMaskData, SimdError, ValidatedData,
};

fn main() {
    // ── CSR: 2×2 all-ones, accumulated into y ──
    let values = [1.0f32, 1.0, 1.0, 1.0];
    let col_indices = [0i32, 1, 0, 1];
    let row_ptr = [0i32, 2, 4];
    let data = CsrData::new(&values[..], &col_indices[..], &row_ptr[..], 2, 2);

    let x = [3.0f32, 4.0];
    let mut y = [1.0f32; 2]; // y starts at 1.0 to exercise the accumulate contract
    spmv_csr::<f32>(ValidatedData::new(data).expect("valid csr"), &x, &mut y);
    // y[0] = 1 + (1*3 + 1*4) = 8; y[1] = 1 + (1*3 + 1*4) = 8
    assert_eq!(y, [8.0, 8.0]);
    println!("spmv_csr accumulated y = {y:?}");

    // ── CSR: structural rejection before any SIMD gather ──
    let bad = CsrData::new(&[1.0f32][..], &[3i32][..], &[0i32, 1][..], 1, 3); // col 3 >= ncols
    match ValidatedData::new(bad) {
        Err(SimdError::IndexOutOfBounds) => {
            println!("out-of-range column rejected: IndexOutOfBounds")
        }
        Err(e) => panic!("unexpected error: {e:?}"),
        Ok(_) => panic!("malformed CSR validated"),
    }

    // ── Dense-with-mask: 2×2 identity ──
    // Dense rectangular storage has no structural index hazard, so the facade
    // takes `DenseWithMaskData` directly (no `ValidatedData` wrapper).
    let dense = DenseWithMaskData::new(
        &[1.0f32, 0.0, 0.0, 1.0][..],
        &[true, false, false, true][..],
        2,
        2,
    );
    let mut y2 = [0.0f32; 2];
    spmv_dense_masked::<f32>(dense, &[6.0f32, 9.0], &mut y2);
    assert_eq!(y2, [6.0, 9.0]);
    println!("spmv_dense_masked y = {y2:?}");

    println!("all sparse assertions passed");
}
