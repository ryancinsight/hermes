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
//! bandwidth may hide them).
//!
//! The gate this was built for has since been answered: `e08ab17` folded the
//! column tail into a masked vector accumulator, so there is no scalar tail
//! left to measure. What the rows now guard is regression in that masked
//! path.
//!
//! Measured 2026-08-04 on a quiet host: the cache-resident pair costs the
//! tail ~18% throughput (25.5 vs 21.0 Gelem/s). The DRAM pair, however,
//! reports the *tail-free* row ~2x SLOWER than the tail-having one (4.4 vs
//! 8.1 Gelem/s, reproduced at p = 0.27), which is not a tail effect at all:
//! 1504 * 4 B = 6016 B is exactly 94 cache lines, so every row of the
//! tail-free matrix starts at the same cache-set alignment, while
//! 1503 * 4 B = 6012 B does not. The comment's claim that a non-power-of-two
//! `lda` avoids the set-conflict pathology does not hold -- what matters is
//! that the byte stride not be a multiple of the line size. Until the sizes
//! are re-chosen (a tail-free `ncols` with `ncols % 16 != 0`, e.g. 1512
//! paired with 1511), read the DRAM row as a layout measurement, not as a
//! tail measurement. Tracked in atlas backlog.md.
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
                    gemv::<f32>(black_box(&a), black_box(&x), &mut y, nrows, ncols).unwrap();
                });
            },
        );
    }
    group.finish();
}

criterion_group!(benches, bench_gemv_f32);
criterion_main!(benches);
