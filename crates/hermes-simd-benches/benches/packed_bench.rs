//! Criterion coverage for packed 4-bit COW unpacking.
//!
//! Evidence tier: empirical validation. These benchmarks measure the public
//! facade path that delegates through the `Packable4` dispatcher, including
//! any selected ISA-specific unpack backend on the host.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use hermes_simd::{Bf4, Packed4CowExt, PackedBf4Cow, PackedF4Cow, Scalar, SimdCow, Unaligned, F4};

const LOGICAL_LENGTHS: [usize; 3] = [1024, 16_384, 262_144];

fn bytes_with<F>(len: usize, mut nibble: F) -> Vec<u8>
where
    F: FnMut(u8) -> u8,
{
    let mut packed = Vec::with_capacity(len.div_ceil(2));
    for byte_index in 0..len.div_ceil(2) {
        let lo = nibble((2 * byte_index) as u8);
        let hi = nibble((2 * byte_index + 1) as u8);
        packed.push((hi << 4) | lo);
    }
    packed
}

fn bench_packed_unpack(c: &mut Criterion) {
    let mut group = c.benchmark_group("packed4_cow_unpack");

    for len in LOGICAL_LENGTHS {
        let bytes = bytes_with(len, |value| Bf4(value & 0x0f).0);
        group.bench_with_input(BenchmarkId::new("bf4_to_wide", len), &bytes, |b, bytes| {
            let cow = PackedBf4Cow::from_packed_slice(bytes, len)
                .expect("invariant: benchmark bytes cover logical length");
            b.iter(|| {
                let unpacked: SimdCow<'static, _, Scalar, Unaligned> =
                    black_box(&cow).unpack_to_cow();
                black_box(unpacked);
            });
        });

        let bytes = bytes_with(len, |value| F4(value & 0x0f).0);
        group.bench_with_input(BenchmarkId::new("f4_to_wide", len), &bytes, |b, bytes| {
            let cow = PackedF4Cow::from_packed_slice(bytes, len)
                .expect("invariant: benchmark bytes cover logical length");
            b.iter(|| {
                let unpacked: SimdCow<'static, _, Scalar, Unaligned> =
                    black_box(&cow).unpack_to_cow();
                black_box(unpacked);
            });
        });
    }

    group.finish();
}

criterion_group!(benches, bench_packed_unpack);
criterion_main!(benches);
