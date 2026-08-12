//! Criterion coverage for public `SimdView::compress`.
//!
//! Evidence tier: empirical validation. These rows measure the public view
//! compaction path whose scratch vector is hoisted out of the chunk loop.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
use hermes_simd::{Avx2, SimdArch};
use hermes_simd::{BitMask, Scalar, SimdView, Unaligned, Unmasked};
use hermes_simd_core::kernel::SimdKernel;

const LENGTHS: [usize; 3] = [1_024, 16_384, 262_144];

/// `SimdView::compress` asserts that the mask's lane count equals the backend's,
/// so the width is taken from the backend rather than written as a literal —
/// a hardcoded width silently rots the moment a backend's lane count changes.
const SCALAR_LANES: usize = <Scalar as SimdKernel<f32>>::LANE_COUNT;

fn data(len: usize) -> Vec<f32> {
    (0..len).map(|idx| idx as f32 * 0.25 + 1.0).collect()
}

fn mask_from<const LANES: usize>(active: impl Fn(usize) -> bool) -> BitMask<LANES> {
    let mut lanes = [false; LANES];
    for (lane, value) in lanes.iter_mut().enumerate() {
        *value = active(lane);
    }
    BitMask::from_bools(&lanes)
}

fn bench_scalar(c: &mut Criterion) {
    let mut group = c.benchmark_group("SimdView compress");
    let mask = mask_from::<SCALAR_LANES>(|_| true);

    for len in LENGTHS {
        let input = data(len);
        let view = SimdView::<f32, Scalar, Unaligned, Unmasked, &[f32]>::new(&input)
            .expect("invariant: unaligned scalar view accepts all slices");
        let mut out = vec![0.0f32; len];
        group.throughput(Throughput::Elements(len as u64));
        group.bench_with_input(BenchmarkId::new("scalar_all", len), &len, |b, _| {
            b.iter(|| {
                let written = black_box(view)
                    .compress(&mask, black_box(&mut out))
                    .expect("invariant: output length equals input length");
                black_box(written);
                black_box(&out[..written]);
            });
        });
    }

    group.finish();
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
fn bench_avx2(c: &mut Criterion) {
    if !Avx2::is_runtime_supported() {
        return;
    }

    let mut group = c.benchmark_group("SimdView compress");
    let masks = [
        ("avx2_all", mask_from::<8>(|_| true)),
        ("avx2_half", mask_from::<8>(|lane| lane % 2 == 0)),
        ("avx2_quarter", mask_from::<8>(|lane| lane % 4 == 0)),
    ];

    for len in LENGTHS {
        let input = data(len);
        let view = SimdView::<f32, Avx2, Unaligned, Unmasked, &[f32]>::new(&input)
            .expect("invariant: unaligned AVX2 view accepts all slices");

        for (name, mask) in masks {
            let mut out = vec![0.0f32; len];
            group.throughput(Throughput::Elements(len as u64));
            group.bench_with_input(BenchmarkId::new(name, len), &len, |b, _| {
                b.iter(|| {
                    let written = black_box(view)
                        .compress(&mask, black_box(&mut out))
                        .expect("invariant: output length equals input length");
                    black_box(written);
                    black_box(&out[..written]);
                });
            });
        }
    }

    group.finish();
}

#[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
fn bench_avx2(_c: &mut Criterion) {}

criterion_group!(benches, bench_scalar, bench_avx2);
criterion_main!(benches);
