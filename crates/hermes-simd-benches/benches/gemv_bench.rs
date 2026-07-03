//! GEMV (`y += A · x`) benchmark, structured to isolate the column-tail cost.
//!
//! GEMV reads each `A` element exactly once for ~2 FLOP, so it is
//! memory-bandwidth-bound above cache and latency/throughput-bound while
//! resident. The `ncols % LANE_COUNT` trailing columns run a scalar tail; each
//! size class pairs a tail-free `ncols` (multiple of the AVX2 f32 lane width, 8)
//! with a tail-having `ncols` (one short of the next multiple, a 7-lane tail) at
//! matched total work, so the per-element throughput gap between the two rows is
//! the tail's marginal cost. Rows span a cache-resident class (where the tail's
//! extra scalar cycles are visible) and a DRAM-resident class (where memory
//! bandwidth may hide them) — the evidence gate for whether a masked-vector tail
//! is worth implementing.
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use hermes_simd::gemv;

fn bench_gemv_f32(c: &mut Criterion) {
    let mut group = c.benchmark_group("GEMV f32 tail");

    // (nrows, ncols): each pair is (tail-free, tail-having) at matched scale.
    // 256×256 ≈ 256 KiB A (L2-resident) is the compute-visible regime where the
    // tail's marginal cost shows; 3000×1504 ≈ 17 MiB A (DRAM, non-power-of-two
    // lda to avoid the cache-set-conflict pathology) is bandwidth-bound, where
    // the tail is hidden — both retained as regression rows.
    for &(nrows, ncols) in &[(256usize, 256usize), (256, 255), (3000, 1504), (3000, 1503)] {
        let a: Vec<f32> = (0..nrows * ncols).map(|i| (i % 17) as f32 * 0.01).collect();
        let x: Vec<f32> = (0..ncols).map(|i| (i % 13) as f32 * 0.1).collect();
        let mut y = vec![0.0f32; nrows];

        group.throughput(Throughput::Elements((nrows * ncols) as u64));
        let tag = if ncols % 8 == 0 { "aligned" } else { "tail7" };
        group.bench_with_input(
            BenchmarkId::new(tag, format!("{nrows}x{ncols}")),
            &(nrows, ncols),
            |bencher, &(nrows, ncols)| {
                bencher.iter(|| {
                    gemv::<f32>(black_box(&a), black_box(&x), &mut y, nrows, ncols).unwrap()
                })
            },
        );
    }
    group.finish();
}

criterion_group!(benches, bench_gemv_f32);
criterion_main!(benches);
