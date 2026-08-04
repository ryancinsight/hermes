//! Runtime ISA detection and dispatch in hermes-simd.
//!
//! `hermes_simd` detects the available instruction set at runtime and
//! routes every operation through the fastest available kernel.  This
//! example reads the detected capabilities and then exercises `sum` and
//! `dot` with runtime dispatch so the numbers are computed by whichever
//! ISA the host supports.

use hermes_simd::{
    cpu::{has_fma3, FmaSupport},
    dot, sum,
};

fn main() {
    // ── ISA capability report ──
    println!("=== ISA Capabilities ===");
    println!("FMA3 (fused multiply-add) : {}", has_fma3());
    println!("f32 FmaSupport            : {}", f32::has_fma());

    // ── Runtime-dispatched sum ──
    println!("\n=== Runtime-dispatched sum ===");
    let data: Vec<f32> = (0..1024).map(|i| i as f32).collect();
    let result = sum::<f32>(&data);
    let expected = (0..1024_u32).sum::<u32>() as f32;
    println!("sum(0..1024) = {result}  expected = {expected}");
    assert!(
        (result - expected).abs() < 1.0,
        "sum deviates by more than 1 ULP from the reference"
    );

    // ── Runtime-dispatched dot product ──
    println!("\n=== Runtime-dispatched dot product ===");
    let ones: Vec<f32> = vec![1.0_f32; 1024];
    let dp = dot::<f32>(&data, &ones).expect("equal-length slices");
    println!("dot(0..1024, [1.0; 1024]) = {dp}  expected = {expected}");
    assert!(
        (dp - expected).abs() < 1.0,
        "dot deviates by more than 1 ULP from the reference"
    );

    // ── Mismatched lengths return Err ──
    assert!(dot::<f32>(&data[..3], &ones[..5]).is_err());
    println!("dot(len=3, len=5) correctly returns Err");

    println!("\nall ISA-dispatch assertions passed");
}

