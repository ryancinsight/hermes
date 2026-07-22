use criterion::{criterion_group, criterion_main};
#[path = "simd/sum.rs"]
mod sum_suite;

fn bench(c: &mut criterion::Criterion) {
    sum_suite::bench(c, "Dense Sum f64", 1.0f64, 0.0f64, hermes_simd::sum::<f64>);
}

criterion_group!(benches, bench);
criterion_main!(benches);
