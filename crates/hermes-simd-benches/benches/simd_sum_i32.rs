use criterion::{criterion_group, criterion_main};
#[path = "simd/sum.rs"]
mod sum_suite;

fn bench(c: &mut criterion::Criterion) {
    sum_suite::bench(c, "Dense Sum i32", 1i32, 0i32, hermes_simd::sum::<i32>);
}

criterion_group!(benches, bench);
criterion_main!(benches);
