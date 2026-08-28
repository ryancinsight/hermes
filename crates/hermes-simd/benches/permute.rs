//! Criterion benchmarks for the cross-lane permute family.
//!
//! Groups:
//! - `reverse_f32` / `reverse_f64`: whole-slice reversal, vector at a time.
//! - `interleave_f32`: two-vector zip over a slice pair.
//! - `deinterleave_f32`: the inverse unzip.
//! - `transpose_square`: one register-resident square tile per backend/scalar.
//!
//! Slice operations run through `PreferredArch`, as a consumer reaches them.
//! The transpose group names each admitted backend explicitly so a native build
//! can be compared with the same backend's forced generic default without
//! adding public API solely for measurement.
//!
//! Throughput is reported in elements/second so widths compare directly.

use criterion::{
    black_box, criterion_group, criterion_main, measurement::WallTime, BenchmarkGroup, BenchmarkId,
    Criterion, Throughput,
};
use hermes_simd_core::{
    kernel::{SimdLoadStore, SimdPermute, SimdStorage},
    Scalar,
};
use hermes_simd_types::PreferredArch;
use std::time::Duration;

#[cfg(target_arch = "aarch64")]
use hermes_simd_intrinsics::Neon;
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
use hermes_simd_intrinsics::{Avx2, Avx512};

const SIZES: &[usize] = &[1024, 16384];

/// Keeps the suite inside the committed per-binary runtime budget: four slice
/// groups at two sizes plus one fixed-tile group take a few seconds at these
/// settings, well under the 300s bound, while still averaging enough iterations
/// for a stable median on operations this small.
fn configure(group: &mut BenchmarkGroup<'_, WallTime>) {
    group.warm_up_time(Duration::from_millis(200));
    group.measurement_time(Duration::from_millis(600));
}

fn reverse_f32(c: &mut Criterion) {
    let mut group = c.benchmark_group("reverse_f32");
    configure(&mut group);
    let lanes = <PreferredArch as SimdStorage<f32>>::LANE_COUNT;

    for &n in SIZES {
        let src: Vec<f32> = (0..n).map(|i| i as f32).collect();
        let mut dst = vec![0.0f32; n];
        group.throughput(Throughput::Elements(n as u64));
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |bencher, _| {
            bencher.iter(|| {
                // SAFETY: `PreferredArch` is selected at compile time from the
                // enabled target features, and every load/store below covers
                // exactly one full vector inside the allocation.
                unsafe {
                    for chunk in 0..n / lanes {
                        let v = <PreferredArch as SimdLoadStore<f32>>::load_unaligned(
                            src.as_ptr().add(chunk * lanes),
                        );
                        let r = <PreferredArch as SimdPermute<f32>>::reverse(v);
                        <PreferredArch as SimdLoadStore<f32>>::store_unaligned(
                            dst.as_mut_ptr().add(chunk * lanes),
                            r,
                        );
                    }
                }
                // Returned so the whole loop cannot be treated as dead.
                black_box(dst[0])
            });
        });
    }
    group.finish();
}

fn reverse_f64(c: &mut Criterion) {
    let mut group = c.benchmark_group("reverse_f64");
    configure(&mut group);
    let lanes = <PreferredArch as SimdStorage<f64>>::LANE_COUNT;

    for &n in SIZES {
        let src: Vec<f64> = (0..n).map(|i| i as f64).collect();
        let mut dst = vec![0.0f64; n];
        group.throughput(Throughput::Elements(n as u64));
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |bencher, _| {
            bencher.iter(|| {
                // SAFETY: as in `reverse_f32`.
                unsafe {
                    for chunk in 0..n / lanes {
                        let v = <PreferredArch as SimdLoadStore<f64>>::load_unaligned(
                            src.as_ptr().add(chunk * lanes),
                        );
                        let r = <PreferredArch as SimdPermute<f64>>::reverse(v);
                        <PreferredArch as SimdLoadStore<f64>>::store_unaligned(
                            dst.as_mut_ptr().add(chunk * lanes),
                            r,
                        );
                    }
                }
                black_box(dst[0])
            });
        });
    }
    group.finish();
}

fn interleave_f32(c: &mut Criterion) {
    let mut group = c.benchmark_group("interleave_f32");
    configure(&mut group);
    let lanes = <PreferredArch as SimdStorage<f32>>::LANE_COUNT;

    for &n in SIZES {
        let a: Vec<f32> = (0..n).map(|i| i as f32).collect();
        let b: Vec<f32> = (0..n).map(|i| -(i as f32)).collect();
        let mut dst = vec![0.0f32; 2 * n];
        group.throughput(Throughput::Elements(2 * n as u64));
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |bencher, _| {
            bencher.iter(|| {
                // SAFETY: as in `reverse_f32`; `dst` holds `2 * n` elements, so
                // each chunk's two stores stay inside the allocation.
                unsafe {
                    for chunk in 0..n / lanes {
                        let va = <PreferredArch as SimdLoadStore<f32>>::load_unaligned(
                            a.as_ptr().add(chunk * lanes),
                        );
                        let vb = <PreferredArch as SimdLoadStore<f32>>::load_unaligned(
                            b.as_ptr().add(chunk * lanes),
                        );
                        let (lo, hi) = <PreferredArch as SimdPermute<f32>>::interleave(va, vb);
                        <PreferredArch as SimdLoadStore<f32>>::store_unaligned(
                            dst.as_mut_ptr().add(2 * chunk * lanes),
                            lo,
                        );
                        <PreferredArch as SimdLoadStore<f32>>::store_unaligned(
                            dst.as_mut_ptr().add((2 * chunk + 1) * lanes),
                            hi,
                        );
                    }
                }
                black_box(dst[0])
            });
        });
    }
    group.finish();
}

fn deinterleave_f32(c: &mut Criterion) {
    let mut group = c.benchmark_group("deinterleave_f32");
    configure(&mut group);
    let lanes = <PreferredArch as SimdStorage<f32>>::LANE_COUNT;

    for &n in SIZES {
        let a: Vec<f32> = (0..n).map(|i| i as f32).collect();
        let b: Vec<f32> = (0..n).map(|i| -(i as f32)).collect();
        let mut even = vec![0.0f32; n];
        let mut odd = vec![0.0f32; n];
        group.throughput(Throughput::Elements(2 * n as u64));
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |bencher, _| {
            bencher.iter(|| {
                // SAFETY: as in `reverse_f32`.
                unsafe {
                    for chunk in 0..n / lanes {
                        let va = <PreferredArch as SimdLoadStore<f32>>::load_unaligned(
                            a.as_ptr().add(chunk * lanes),
                        );
                        let vb = <PreferredArch as SimdLoadStore<f32>>::load_unaligned(
                            b.as_ptr().add(chunk * lanes),
                        );
                        let (e, o) = <PreferredArch as SimdPermute<f32>>::deinterleave(va, vb);
                        <PreferredArch as SimdLoadStore<f32>>::store_unaligned(
                            even.as_mut_ptr().add(chunk * lanes),
                            e,
                        );
                        <PreferredArch as SimdLoadStore<f32>>::store_unaligned(
                            odd.as_mut_ptr().add(chunk * lanes),
                            o,
                        );
                    }
                }
                black_box(even[0]) + black_box(odd[0])
            });
        });
    }
    group.finish();
}

fn measure_transpose<T, A>(group: &mut BenchmarkGroup<'_, WallTime>, backend: &str, scalar: &str)
where
    T: Scalar + PartialEq + core::fmt::Debug + From<u16>,
    A: SimdLoadStore<T> + SimdPermute<T> + SimdStorage<T>,
{
    let lanes = <A as SimdStorage<T>>::LANE_COUNT;
    let values: Vec<T> = (0..lanes * lanes)
        .map(|index| T::from(u16::try_from(index).expect("fixture index must fit in u16")))
        .collect();

    // SAFETY: callers admit only a backend whose target features are present;
    // every chunk contains exactly one complete register.
    let mut tile: Vec<<A as SimdStorage<T>>::Vector> = values
        .chunks_exact(lanes)
        .map(|row| unsafe { <A as SimdLoadStore<T>>::load_unaligned(row.as_ptr()) })
        .collect();

    // Validate the exact row/column mapping before timing, then transpose a
    // second time to restore the fixture for the first measured iteration.
    let mut observed = values.clone();
    unsafe {
        <A as SimdPermute<T>>::transpose_square(&mut tile);
        for (row, vector) in observed.chunks_exact_mut(lanes).zip(tile.iter().copied()) {
            <A as SimdLoadStore<T>>::store_unaligned(row.as_mut_ptr(), vector);
        }
        <A as SimdPermute<T>>::transpose_square(&mut tile);
    }
    for row in 0..lanes {
        for column in 0..lanes {
            assert_eq!(
                observed[row * lanes + column],
                values[column * lanes + row],
                "transpose fixture mismatch at ({row}, {column})"
            );
        }
    }

    group.throughput(Throughput::Elements((lanes * lanes) as u64));
    group.bench_with_input(BenchmarkId::new(backend, scalar), &lanes, |bencher, _| {
        bencher.iter(|| {
            // SAFETY: the backend admission and exact tile length are unchanged
            // from the checked setup above.
            unsafe { <A as SimdPermute<T>>::transpose_square(black_box(&mut tile)) };
        });
    });
}

fn transpose_square(c: &mut Criterion) {
    let mut group = c.benchmark_group("transpose_square");
    configure(&mut group);

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if std::is_x86_feature_detected!("avx2") {
            measure_transpose::<f32, Avx2>(&mut group, "avx2", "f32");
            measure_transpose::<f64, Avx2>(&mut group, "avx2", "f64");
        }
        if std::is_x86_feature_detected!("avx512f") {
            measure_transpose::<f32, Avx512>(&mut group, "avx512", "f32");
            measure_transpose::<f64, Avx512>(&mut group, "avx512", "f64");
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        measure_transpose::<f32, Neon>(&mut group, "neon", "f32");
        measure_transpose::<f64, Neon>(&mut group, "neon", "f64");
    }

    group.finish();
}

criterion_group!(
    permutes,
    reverse_f32,
    reverse_f64,
    interleave_f32,
    deinterleave_f32,
    transpose_square
);
criterion_main!(permutes);
