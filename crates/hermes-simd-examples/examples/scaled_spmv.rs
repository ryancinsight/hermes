//! Fused scaled sparse product `y += alpha * A * x` in one CSR pass.
//!
//! `spmv_scaled_csr` scales each row's dot product by `alpha` before the single
//! write to `y[r]`, so a scaled accumulation needs no second `axpy` pass over
//! `y`. The matrix is validated once by `ValidatedData::new`; the asserts check
//! the result against the same sum computed by hand.

#![expect(
    clippy::float_cmp,
    reason = "The runnable example asserts exact manufactured sparse outputs"
)]

use hermes_simd::{spmv_scaled_csr, CsrData, ValidatedData};

fn main() {
    // A = [[2, 0, 1],
    //      [0, 0, 0],
    //      [3, 4, 5]]; row 1 is empty and must leave y[1] untouched.
    let values = [2.0f64, 1.0, 3.0, 4.0, 5.0];
    let col_indices = [0i32, 2, 0, 1, 2];
    let row_ptr = [0i32, 2, 2, 5];
    let data = ValidatedData::new(CsrData::new(
        &values[..],
        &col_indices[..],
        &row_ptr[..],
        3,
        3,
    ))
    .expect("invariant: the example CSR matrix is structurally valid");

    let x = [1.0f64, 2.0, 4.0];
    let mut y = [10.0f64, 20.0, 30.0];
    let alpha = 0.5f64;
    spmv_scaled_csr::<f64>(data.clone(), alpha, &x, &mut y);

    // Every product and partial sum is a dyadic rational below 2^53, so the
    // arithmetic is exact in any SIMD reduction order.
    //   y[0] = 10 + 0.5 * (2*1 + 1*4)           = 13
    //   y[1] = 20                               (empty row)
    //   y[2] = 30 + 0.5 * (3*1 + 4*2 + 5*4)     = 45.5
    assert_eq!(y, [13.0, 20.0, 45.5]);

    // A zero scale leaves the accumulator unchanged.
    spmv_scaled_csr::<f64>(data, 0.0, &x, &mut y);
    assert_eq!(y, [13.0, 20.0, 45.5]);
}
