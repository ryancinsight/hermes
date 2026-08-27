//! Lane and dispatch-boundary cost for consumer-shaped kernels.
//!
//! The dispatch group isolates public entry cost from kernel throughput. The
//! planar group compares Hermes and `fearless_simd` at both native floating
//! precisions. The permute groups compare the shared cross-lane operation
//! surface. The interleaved group isolates Hermes wrapper overhead across
//! checked, view/chunk, and direct backend paths, then compares `ComplexReg`
//! with the raw Hermes recipe and Fearless SIMD's public
//! deinterleave/planar/reinterleave composition.

use core::time::Duration;

use criterion::{criterion_group, criterion_main, Criterion};

#[path = "lane_throughput/comparison.rs"]
mod comparison;
#[path = "lane_throughput/dispatch.rs"]
mod dispatch;
#[path = "lane_throughput/interleaved.rs"]
mod interleaved;
#[path = "lane_throughput/permute.rs"]
mod permute;
#[path = "lane_throughput/planar.rs"]
mod planar;

const SCALAR_LENGTHS: &[usize] = &[256, 1_024, 4_096];

criterion_group! {
    name = benches;
    config = Criterion::default()
        .warm_up_time(Duration::from_millis(20))
        .measurement_time(Duration::from_millis(200))
        .sample_size(20);
    targets = dispatch::bench, planar::bench, permute::bench, interleaved::bench
}
criterion_main!(benches);
