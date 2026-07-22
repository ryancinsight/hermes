use criterion::{criterion_group, criterion_main};
#[path = "simd/sum.rs"]
mod sum_suite;

fn bench(c: &mut criterion::Criterion) {
    sum_suite::bench(c, "Dense Sum f32", 1.0f32, 0.0f32, hermes_simd::sum::<f32>);
}

criterion_group!(benches, bench);
criterion_main!(benches);
