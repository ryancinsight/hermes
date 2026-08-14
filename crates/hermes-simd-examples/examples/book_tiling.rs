//! Register-blocked tiling: `tiled_dot` and `gemv` on `SimdView`.
//!
//! Both kernels hold `TILE_M` independent register accumulators to saturate
//! FMA throughput (`tiled_dot`) or to reuse each loaded `x` vector across
//! `TILE_M` rows (`gemv`, which is memory-bound). This example runs both paths
//! and cross-checks against the plain facade `dot`.

#![expect(
    clippy::float_cmp,
    reason = "The runnable example asserts exact manufactured tile outputs"
)]

use hermes_simd::{dot, gemv, tiled_dot, Scalar, SimdError, SimdView, Unaligned};

fn main() -> Result<(), SimdError> {
    // ── tiled_dot: TILE_M independent FMA accumulators ──
    let a = [1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let b = [2.0_f32, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0];

    let va = SimdView::<f32, Scalar, Unaligned>::new(&a).expect("slice fits");
    let vb = SimdView::<f32, Scalar, Unaligned>::new(&b).expect("slice fits");

    // TILE_M = 4 independent accumulators: 4*LANE_COUNT elements per iteration.
    let tiled = tiled_dot::<f32, Scalar, Unaligned, 4>(&va, &vb)?;
    let plain = dot::<f32>(&a, &b)?;

    println!("tiled_dot(TILE_M=4) = {tiled}");
    println!("dot                  = {plain}");
    assert!((tiled - 72.0).abs() < 1e-6);
    assert!((tiled - plain).abs() < 1e-6);
    assert!(
        dot::<f32>(&a[..3], &b[..5]).is_err(),
        "length mismatch is an error"
    );

    // ── gemv: register-blocked y += A·x ──
    // A is row-major 3×4; the product ACCUMULATES into y, so zero y for y = A·x.
    let a_mat = [
        1.0_f32, 0.0, 2.0, -1.0, // row 0
        0.0, 1.0, 0.0, 3.0, // row 1
        -2.0, 0.0, 1.0, 0.0, // row 2
    ];
    let x = [1.0_f32, 2.0, 3.0, 4.0];
    let mut y = [0.0_f32; 3];

    gemv(&a_mat, &x, &mut y, 3, 4)?;
    println!("gemv y = {y:?}"); // row0 = 1+6-4 = 3; row1 = 2+12 = 14; row2 = -2+3 = 1
    assert_eq!(y, [3.0, 14.0, 1.0]);

    // The accumulate convention: y += A·x is the contract, so a second call adds.
    gemv(&a_mat, &x, &mut y, 3, 4)?;
    assert_eq!(y, [6.0, 28.0, 2.0]);
    println!("gemv again accumulates: y = {y:?}");

    Ok(())
}
