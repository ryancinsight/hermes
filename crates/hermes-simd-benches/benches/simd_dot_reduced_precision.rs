use criterion::{criterion_group, criterion_main};
#[path = "simd/dispatch_dot.rs"]
mod dot_suite;

fn bench(c: &mut criterion::Criterion) {
    dot_suite::bench(
        c,
        "Dense Dot f16",
        eunomia::F16::from_f32(1.5),
        eunomia::F16::from_f32(0.5),
        |a, b| {
            hermes_simd::dot::<eunomia::F16>(a, b)
                .expect("invariant: equal benchmark input lengths")
        },
    );
    dot_suite::bench(
        c,
        "Dense Dot bf16",
        eunomia::Bf16::from_f32(1.5),
        eunomia::Bf16::from_f32(0.5),
        |a, b| {
            hermes_simd::dot::<eunomia::Bf16>(a, b)
                .expect("invariant: equal benchmark input lengths")
        },
    );
}

criterion_group!(benches, bench);
criterion_main!(benches);
