//! Lane-boundary cost for consumer-shaped butterfly kernels.
//!
//! The planar group compares Hermes and `fearless_simd` at both native floating
//! precisions. The interleaved group isolates Hermes wrapper overhead by holding
//! arithmetic and dispatch constant across checked, view/chunk, and direct
//! backend paths.

use core::time::Duration;

use criterion::{criterion_group, criterion_main, Criterion};

#[path = "lane_throughput/comparison.rs"]
mod comparison;
#[path = "lane_throughput/interleaved.rs"]
mod interleaved;
#[path = "lane_throughput/planar.rs"]
mod planar;

const SCALAR_LENGTHS: &[usize] = &[256, 1_024, 4_096];

criterion_group! {
    name = benches;
    config = Criterion::default()
        .warm_up_time(Duration::from_millis(20))
        .measurement_time(Duration::from_millis(200))
        .sample_size(20);
    targets = planar::bench, interleaved::bench
}
criterion_main!(benches);
