use criterion::{criterion_group, criterion_main};
#[path = "simd/dot.rs"]
mod dot_suite;

fn bench(c: &mut criterion::Criterion) {
    dot_suite::bench(c, "Dense Dot f32", 1.0f32, 2.0f32, 0.0f32, |a, b| {
        hermes_simd::dot::<f32>(a, b).expect("invariant: equal benchmark input lengths")
    });
}

criterion_group!(benches, bench);
criterion_main!(benches);
