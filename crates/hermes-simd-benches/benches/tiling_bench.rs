//! Register-blocking / tiling benchmark: const-generic tile size sweep.
//!
//! Measures the effect of different `TILE_M` values on dot product throughput.
//! Goal: demonstrate that `TILE_M=4` (and `TILE_M=8` on AVX-512) outperform
//! `TILE_M=1` by eliminating loop-carried FMA dependency chains.
//!
//! Each tile accumulates into `TILE_M` independent accumulators, allowing the
//! CPU to exploit out-of-order execution across the FMA latency pipeline.
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use hermes_simd::{dot, tiled_dot, Scalar, SimdView, Unaligned, Unmasked};

fn bench_tiled_dot(c: &mut Criterion) {
    let mut group = c.benchmark_group("Tiled Dot f32");

    for &size in &[1024usize, 16384, 65536] {
        let a: Vec<f32> = (0..size).map(|i| i as f32 * 0.001).collect();
        let b_vec: Vec<f32> = (0..size).map(|i| (size - i) as f32 * 0.001).collect();

        let view_a = SimdView::<f32, Scalar, Unaligned, Unmasked, &[f32]>::new(&a).unwrap();
        let view_b = SimdView::<f32, Scalar, Unaligned, Unmasked, &[f32]>::new(&b_vec).unwrap();

        group.throughput(Throughput::Elements(size as u64));

        // Baseline: standard runtime-dispatched dot product
        group.bench_with_input(BenchmarkId::new("dispatch", size), &size, |bencher, _| {
            bencher.iter(|| dot::<f32>(&a, &b_vec).unwrap());
        });

        // TILE_M=1: degenerate single accumulator (same as a plain loop)
        group.bench_with_input(BenchmarkId::new("tile_1", size), &size, |bencher, _| {
            bencher.iter(|| tiled_dot::<f32, Scalar, Unaligned, 1>(&view_a, &view_b).unwrap());
        });

        // TILE_M=2
        group.bench_with_input(BenchmarkId::new("tile_2", size), &size, |bencher, _| {
            bencher.iter(|| tiled_dot::<f32, Scalar, Unaligned, 2>(&view_a, &view_b).unwrap());
        });

        // TILE_M=4: matches UNROLL_FACTOR for AVX2 (4 x 256-bit = 32 f32 lanes)
        group.bench_with_input(BenchmarkId::new("tile_4", size), &size, |bencher, _| {
            bencher.iter(|| tiled_dot::<f32, Scalar, Unaligned, 4>(&view_a, &view_b).unwrap());
        });

        // TILE_M=8: optimal for AVX-512 (8 x 512-bit = 128 f32 lanes)
        group.bench_with_input(BenchmarkId::new("tile_8", size), &size, |bencher, _| {
            bencher.iter(|| tiled_dot::<f32, Scalar, Unaligned, 8>(&view_a, &view_b).unwrap());
        });
    }
    group.finish();
}

criterion_group!(benches, bench_tiled_dot);
criterion_main!(benches);
