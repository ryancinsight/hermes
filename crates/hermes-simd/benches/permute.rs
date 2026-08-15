//! Criterion benchmarks for the cross-lane permute family.
//!
//! Groups:
//! - `reverse_f32` / `reverse_f64`: whole-slice reversal, vector at a time.
//! - `interleave_f32`: two-vector zip over a slice pair.
//! - `deinterleave_f32`: the inverse unzip.
//!
//! These measure the operations as a consumer reaches them, through
//! `PreferredArch`. They are the committed regression baseline for the native
//! overrides; the override-versus-generic-default comparison is a one-off
//! measurement recorded in the backlog, because the generic default lives in a
//! private module and no public API is added solely to benchmark it.
//!
//! Throughput is reported in elements/second so widths compare directly.

use criterion::{
    black_box, criterion_group, criterion_main, measurement::WallTime, BenchmarkGroup, BenchmarkId,
    Criterion, Throughput,
};
use hermes_simd_core::kernel::{SimdLoadStore, SimdPermute, SimdStorage};
use hermes_simd_types::PreferredArch;
use std::time::Duration;

const SIZES: &[usize] = &[1024, 16384];

/// Keeps the suite inside the committed per-binary runtime budget: four groups
/// times two sizes at these settings is a few seconds of measurement, well
/// under the 300s bound, while still averaging enough iterations for a stable
/// median on an operation this cheap.
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

criterion_group!(
    permutes,
    reverse_f32,
    reverse_f64,
    interleave_f32,
    deinterleave_f32
);
criterion_main!(permutes);
