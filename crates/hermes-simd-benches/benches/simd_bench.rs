//! Dense SIMD benchmarks: sum and dot product with backend comparison.
//!
//! Benchmark groups:
//! - `Dense Sum f32`/`f64`/`i32` — dispatch vs scalar iterator, sizes 256–1M
//! - `Dense Dot f32`/`f64`/`i32` — dispatch vs scalar iterator, sizes 256–65536
//! - `Dense Dot f16`/`bf16` — reduced-precision dispatch, sizes 256–65536
//! - `Dense AXPY f32`/`f64` — dispatch tail cost, sizes 3–31
//! - `Exact Lane FMA f16x8`/`f32x4`/`f64x4` — `vectorize_lanes` scalar-fallback
//!   codegen, sizes 256–4096

#[path = "simd/axpy.rs"]
mod axpy;
#[path = "simd/dispatch_dot.rs"]
mod dispatch_dot;
#[path = "simd/dot.rs"]
mod dot;
#[path = "simd/group.rs"]
mod group;
#[path = "simd/lane_frame.rs"]
mod lane_frame;
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

fn bench_lane_frame(c: &mut Criterion) {
    // F16 at eight lanes is the changed path: AVX-512 carries 32 F16 lanes and
    // AVX2 carries 16, so eight matches neither and the request falls through
    // to the scalar backend.
    lane_frame::bench::<8, _>(
        c,
        "Exact Lane FMA f16x8",
        eunomia::F16::from_f32(1.5),
        eunomia::F16::from_f32(0.5),
    );
    // f32 at four lanes reaches the same fallback through the same arm and is
    // unaffected by any F16 change — the instrument's own control.
    lane_frame::bench::<4, _>(c, "Exact Lane FMA f32x4", 1.5_f32, 0.5_f32);
    // f64 at four lanes matches AVX2 exactly and never reaches the fallback,
    // so it measures host stability rather than this path at all.
    lane_frame::bench::<4, _>(c, "Exact Lane FMA f64x4", 1.5_f64, 0.5_f64);
}

fn benchmark_config() -> Criterion {
    // Seventy-seven benchmark IDs at ten flat samples require 46.2 seconds of
    // scheduled warm-up and measurement before analysis. This preserves every
    // workload while bounding the canonical binary below its 300-second CI
    // limit.
    Criterion::default()
        .warm_up_time(Duration::from_millis(100))
        .measurement_time(Duration::from_millis(500))
        .sample_size(10)
}

criterion_group! {
    name = benches;
    config = benchmark_config();
    targets = bench_sum, bench_dot, bench_axpy, bench_lane_frame
}
criterion_main!(benches);
