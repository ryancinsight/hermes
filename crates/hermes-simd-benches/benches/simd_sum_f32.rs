use criterion::{criterion_group, criterion_main};
#[path = "simd/suite.rs"]
mod suite;

criterion_group!(benches, suite::bench_sum_f32);
criterion_main!(benches);
