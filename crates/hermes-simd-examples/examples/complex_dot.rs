//! Interleaved complex kernels: runtime-dispatched vs scalar backend.
//!
//! Demonstrates the `[re, im, ...]` interleaved complex API and reports the
//! throughput difference between the runtime-selected SIMD backend and the
//! always-available `Scalar` backend.

use std::time::Instant;

use hermes_simd::{
    interleaved_complex_dot, interleaved_complex_dot_runtime, interleaved_complex_mul_assign,
    interleaved_complex_mul_assign_runtime, Scalar,
};

const COMPLEX_LEN: usize = 1 << 16;
const ITERS: u32 = 2_000;

fn main() {
    // Dyadic-rational inputs: products are exactly representable, so the SIMD
    // and scalar paths must agree bitwise.
    let a: Vec<f64> = (0..COMPLEX_LEN * 2)
        .map(|i| ((i % 9) as f64) * 0.25 - 1.0)
        .collect();
    let b: Vec<f64> = (0..COMPLEX_LEN * 2)
        .map(|i| ((i % 7) as f64) * 0.5 - 1.5)
        .collect();

    let start = Instant::now();
    let mut dot_simd = (0.0, 0.0);
    for _ in 0..ITERS {
        dot_simd = interleaved_complex_dot_runtime::<f64, false>(&a, &b).unwrap();
    }
    let simd_time = start.elapsed();

    let start = Instant::now();
    let mut dot_scalar = (0.0, 0.0);
    for _ in 0..ITERS {
        dot_scalar = interleaved_complex_dot::<f64, Scalar, false>(&a, &b).unwrap();
    }
    let scalar_time = start.elapsed();

    assert_eq!(dot_simd, dot_scalar);
    println!("dot       runtime SIMD: {simd_time:>10.2?}  scalar: {scalar_time:>10.2?}");

    // Reset the destination each iteration: repeated in-place squaring would
    // overflow, and at non-finite magnitudes fused and unfused rounding may
    // legitimately diverge. Both timed loops carry the same copy cost.
    let mut a_simd = a.clone();
    let start = Instant::now();
    for _ in 0..ITERS {
        a_simd.copy_from_slice(&a);
        interleaved_complex_mul_assign_runtime::<f64, true>(&mut a_simd, &b).unwrap();
    }
    let simd_time = start.elapsed();

    let mut a_scalar = a.clone();
    let start = Instant::now();
    for _ in 0..ITERS {
        a_scalar.copy_from_slice(&a);
        interleaved_complex_mul_assign::<f64, Scalar, true>(&mut a_scalar, &b).unwrap();
    }
    let scalar_time = start.elapsed();

    assert_eq!(a_simd, a_scalar);
    println!("mul_assign runtime SIMD: {simd_time:>10.2?}  scalar: {scalar_time:>10.2?}");
}
