//! Dense SIMD benchmarks: sum and dot product with backend comparison.
//!
//! Benchmark groups:
//! - `Dense Sum f32`/`f64`/`i32` — dispatch vs scalar iterator, sizes 256–1M
//! - `Dense Dot f32`/`f64`/`i32` — dispatch vs scalar iterator, sizes 256–65536
//! - `Dense Dot f16`/`bf16` — reduced-precision dispatch, sizes 256–65536
//! - `Dense AXPY f32`/`f64` — dispatch tail cost, sizes 3–31

#[path = "simd/axpy.rs"]
mod axpy;
#[path = "simd/dispatch_dot.rs"]
mod dispatch_dot;
#[path = "simd/dot.rs"]
mod dot;
#[path = "simd/group.rs"]
mod group;
#[path = "simd/sum.rs"]
mod sum;

use criterion::{criterion_group, criterion_main, Criterion};
use std::time::Duration;

fn bench_sum(c: &mut Criterion) {
    sum::bench(c, "Dense Sum f32", 1.0f32, 0.0f32, hermes_simd::sum);
    sum::bench(c, "Dense Sum f64", 1.0f64, 0.0f64, hermes_simd::sum);
    sum::bench(c, "Dense Sum i32", 1i32, 0i32, hermes_simd::sum);
}

fn bench_dot(c: &mut Criterion) {
    dot::bench(c, "Dense Dot f32", 1.0f32, 2.0f32, 0.0f32, |a, b| {
        hermes_simd::dot(a, b).expect("invariant: benchmark vectors have equal length")
    });
    dot::bench(c, "Dense Dot f64", 1.0f64, 2.0f64, 0.0f64, |a, b| {
        hermes_simd::dot(a, b).expect("invariant: benchmark vectors have equal length")
    });
    dot::bench(c, "Dense Dot i32", 3i32, 2i32, 0i32, |a, b| {
        hermes_simd::dot(a, b).expect("invariant: benchmark vectors have equal length")
    });
    dispatch_dot::bench(
        c,
        "Dense Dot f16",
        eunomia::F16::from_f32(1.5),
        eunomia::F16::from_f32(0.5),
        |a, b| hermes_simd::dot(a, b).expect("invariant: benchmark vectors have equal length"),
    );
    dispatch_dot::bench(
        c,
        "Dense Dot bf16",
        eunomia::Bf16::from_f32(1.5),
        eunomia::Bf16::from_f32(0.5),
        |a, b| hermes_simd::dot(a, b).expect("invariant: benchmark vectors have equal length"),
    );
}

fn bench_axpy(c: &mut Criterion) {
    axpy::bench(
        c,
        "Dense AXPY f32",
        1.25_f32,
        0.75_f32,
        2.0_f32,
        |alpha, x, out| {
            hermes_simd::axpy(alpha, x, out)
                .expect("invariant: benchmark vectors have equal length");
        },
    );
    axpy::bench(
        c,
        "Dense AXPY f64",
        1.25_f64,
        0.75_f64,
        2.0_f64,
        |alpha, x, out| {
            hermes_simd::axpy(alpha, x, out)
                .expect("invariant: benchmark vectors have equal length");
        },
    );
}

fn benchmark_config() -> Criterion {
    // Sixty-eight benchmark IDs at ten flat samples require 40.8 seconds of scheduled
    // warm-up and measurement before analysis. This preserves every workload
    // while bounding the canonical binary below its 300-second CI limit.
    Criterion::default()
        .warm_up_time(Duration::from_millis(100))
        .measurement_time(Duration::from_millis(500))
        .sample_size(10)
}

criterion_group! {
    name = benches;
    config = benchmark_config();
    targets = bench_sum, bench_dot, bench_axpy
}
criterion_main!(benches);
