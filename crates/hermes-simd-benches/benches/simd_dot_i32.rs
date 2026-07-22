use criterion::{criterion_group, criterion_main};
#[path = "simd/dot.rs"]
mod dot_suite;

fn bench(c: &mut criterion::Criterion) {
    dot_suite::bench(c, "Dense Dot i32", 3i32, 2i32, 0i32, |a, b| {
        hermes_simd::dot::<i32>(a, b).expect("invariant: equal benchmark input lengths")
    });
}

criterion_group!(benches, bench);
criterion_main!(benches);
