//! Same-binary comparison-mask extraction against the native route and Fearless SIMD.

use core::marker::PhantomData;
use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, Throughput};
use fearless_simd::prelude::{SimdBase, SimdMask};
use fearless_simd::{Level, Simd as FearlessSimd};
use hermes_simd::{vectorize, LaneKernel, Simd, SimdArch, SimdKernel, Vector};

use super::comparison::{fearless_lane_count, hermes_lane_count, BenchmarkFloat};
use super::SCALAR_LENGTHS;

trait HermesMaskRoute {
    fn equal_bits<T, A>(a: Vector<T, A>, b: Vector<T, A>) -> u64
    where
        T: BenchmarkFloat,
        A: SimdArch + SimdKernel<T>;
}

struct PublicRoute;

impl HermesMaskRoute for PublicRoute {
    #[inline(always)]
    fn equal_bits<T, A>(a: Vector<T, A>, b: Vector<T, A>) -> u64
    where
        T: BenchmarkFloat,
        A: SimdArch + SimdKernel<T>,
    {
        a.cmp_eq_mask(b).to_bitmask().0
    }
}

struct DirectRoute;

impl HermesMaskRoute for DirectRoute {
    #[inline(always)]
    fn equal_bits<T, A>(a: Vector<T, A>, b: Vector<T, A>) -> u64
    where
        T: BenchmarkFloat,
        A: SimdArch + SimdKernel<T>,
    {
        // SAFETY: both vectors carry the host-support proof for `A`; the
        // comparison and mask conversions execute within that feature frame.
        unsafe { A::mask_to_bitmask(A::vector_to_mask(A::cmp_eq(a.raw, b.raw))) }
    }
}

struct HermesEqual<'a, T, Route> {
    a: &'a [T],
    b: &'a [T],
    route: PhantomData<Route>,
}

impl<T: BenchmarkFloat, Route: HermesMaskRoute> LaneKernel<T> for HermesEqual<'_, T, Route> {
    type Output = usize;

    #[inline(always)]
    fn call<A: SimdArch + SimdKernel<T>>(self, _simd: Simd<T, A>) -> Self::Output {
        let lanes = A::LANE_COUNT;
        let mut matches = 0usize;
        for (a, b) in self.a.chunks_exact(lanes).zip(self.b.chunks_exact(lanes)) {
            // SAFETY: each exact chunk holds one complete `A` vector and this
            // target-feature frame establishes host support for `A`.
            let a = unsafe { Vector::<T, A>::load_unaligned(a.as_ptr()) };
            let b = unsafe { Vector::<T, A>::load_unaligned(b.as_ptr()) };
            matches += Route::equal_bits(a, b).count_ones() as usize;
        }
        matches
    }
}

#[inline(always)]
fn fearless_equal<T: BenchmarkFloat, S: FearlessSimd>(simd: S, a: &[T], b: &[T]) -> usize {
    let lanes = T::Fearless::<S>::N;
    a.chunks_exact(lanes)
        .zip(b.chunks_exact(lanes))
        .map(|(a, b)| {
            T::Fearless::<S>::from_slice(simd, a)
                .simd_eq(T::Fearless::<S>::from_slice(simd, b))
                .to_bitmask()
                .count_ones() as usize
        })
        .sum()
}

fn inputs<T: BenchmarkFloat>(len: usize) -> (Vec<T>, Vec<T>) {
    let a = (0..len)
        .map(|index| {
            let value = u32::try_from(index % 97).expect("bounded index fits u32");
            T::from_f64(f64::from(value) - 48.0)
        })
        .collect::<Vec<_>>();
    let b = a
        .iter()
        .enumerate()
        .map(|(index, &value)| {
            if index % 3 == 0 || index % 11 == 0 {
                value
            } else {
                let replacement = u32::try_from((index + 1) % 97).expect("bounded index fits u32");
                T::from_f64(f64::from(replacement) - 48.0)
            }
        })
        .collect();
    (a, b)
}

fn bench_precision<T: BenchmarkFloat>(c: &mut Criterion) {
    let level = Level::new();
    let mut group = c.benchmark_group(format!("comparison_mask_{}", T::LABEL));
    for &len in SCALAR_LENGTHS {
        group.throughput(Throughput::Elements(
            u64::try_from(len).expect("benchmark length fits u64"),
        ));
        let (a, b) = inputs::<T>(len);
        let expected = a.iter().zip(&b).filter(|(a, b)| a == b).count();

        let hermes_lanes = hermes_lane_count::<T>();
        let fearless_lanes =
            fearless_simd::dispatch!(level, simd => fearless_lane_count::<T, _>(simd));
        assert_eq!(
            hermes_lanes, fearless_lanes,
            "the comparison requires equal native vector widths"
        );
        assert_eq!(
            len % hermes_lanes,
            0,
            "the comparison workload must contain only complete native vectors"
        );

        let public = vectorize(HermesEqual::<T, PublicRoute> {
            a: &a,
            b: &b,
            route: PhantomData,
        });
        let direct = vectorize(HermesEqual::<T, DirectRoute> {
            a: &a,
            b: &b,
            route: PhantomData,
        });
        let fearless =
            fearless_simd::dispatch!(level, simd => fearless_equal::<T, _>(simd, &a, &b));
        assert_eq!(public, expected);
        assert_eq!(direct, expected);
        assert_eq!(fearless, expected);

        group.bench_function(BenchmarkId::new("hermes_public", len), |bencher| {
            bencher.iter(|| {
                black_box(vectorize(HermesEqual::<T, PublicRoute> {
                    a: black_box(&a),
                    b: black_box(&b),
                    route: PhantomData,
                }))
            });
        });
        group.bench_function(BenchmarkId::new("hermes_direct", len), |bencher| {
            bencher.iter(|| {
                black_box(vectorize(HermesEqual::<T, DirectRoute> {
                    a: black_box(&a),
                    b: black_box(&b),
                    route: PhantomData,
                }))
            });
        });
        group.bench_function(BenchmarkId::new("fearless_simd", len), |bencher| {
            bencher.iter(|| {
                black_box(fearless_simd::dispatch!(level, simd =>
                    fearless_equal::<T, _>(simd, black_box(&a), black_box(&b))
                ))
            });
        });
    }
    group.finish();
}

pub(super) fn bench(c: &mut Criterion) {
    bench_precision::<f32>(c);
    bench_precision::<f64>(c);
}
