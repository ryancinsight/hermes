//! Masked SIMD benchmarks: compare dense vs masked sum and dot product.
//!
//! Benchmark groups:
//! - `Masked Sum f32` — `masked_sum::<f32>` at various densities vs `sum::<f32>`
//! - `Masked Dot f32` — `masked_dot::<f32>` at various densities vs `dot::<f32>`
//!
//! Density is the fraction of `true` values in the mask (1.0 = all active,
//! 0.01 = 1% active). Lower densities stress the scalar tail and mask overhead.
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use hermes_simd::{dot, masked_dot, masked_sum, sum};

/// Generate a boolean mask with `density` fraction of `true` values.
///
/// Values are distributed uniformly: index `i` is active iff
/// `(i as f32 / len as f32) < density`, giving a deterministic, repeatable layout.
fn make_mask(len: usize, density: f32) -> Vec<bool> {
    (0..len)
        .map(|i| (i as f32 / len as f32) < density)
        .collect()
}

fn bench_masked_sum(c: &mut Criterion) {
    let mut group = c.benchmark_group("Masked Sum f32");
    let size = 65536usize;
    let data = vec![1.0f32; size];
    group.throughput(Throughput::Elements(size as u64));

    // Dense baseline
    group.bench_function("dense_sum", |b| b.iter(|| sum::<f32>(&data)));

    // Masked at various densities
    for density in [1.0f32, 0.5, 0.1, 0.01] {
        let mask = make_mask(size, density);
        let label = format!("density_{:.0}pct", density * 100.0);
        group.bench_with_input(
            BenchmarkId::new("masked_sum", &label),
            &density,
            |b, _| b.iter(|| masked_sum::<f32>(&data, &mask)),
        );
    }
    group.finish();
}

fn bench_masked_dot(c: &mut Criterion) {
    let mut group = c.benchmark_group("Masked Dot f32");
    let size = 65536usize;
    let a = vec![1.0f32; size];
    let b = vec![2.0f32; size];
    group.throughput(Throughput::Elements(size as u64));

    group.bench_function("dense_dot", |bencher| bencher.iter(|| dot::<f32>(&a, &b).unwrap()));

    for density in [1.0f32, 0.5, 0.1, 0.01] {
        let mask = make_mask(size, density);
        let label = format!("density_{:.0}pct", density * 100.0);
        group.bench_with_input(
            BenchmarkId::new("masked_dot", &label),
            &density,
            |bencher, _| bencher.iter(|| masked_dot::<f32>(&a, &b, &mask).unwrap()),
        );
    }
    group.finish();
}

criterion_group!(benches, bench_masked_sum, bench_masked_dot);
criterion_main!(benches);