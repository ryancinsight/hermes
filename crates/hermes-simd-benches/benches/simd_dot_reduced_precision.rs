use criterion::{criterion_group, criterion_main};
#[path = "simd/suite.rs"]
mod suite;

criterion_group!(benches, suite::bench_dot_reduced_precision);
criterion_main!(benches);
