//! Same-binary cross-lane comparison against `fearless_simd`.

use core::marker::PhantomData;
use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, Throughput};
use fearless_simd::prelude::SimdBase;
use fearless_simd::{Level, Simd as FearlessSimd};
use hermes_simd::{vectorize, LaneKernel, Simd, SimdArch, SimdKernel, Vector};

use super::comparison::{fearless_lane_count, hermes_lane_count, BenchmarkFloat};
use super::SCALAR_LENGTHS;

trait PermuteOperation {
    const LABEL: &'static str;

    fn reference<T: BenchmarkFloat>(a: &[T], b: &[T], out_a: &mut [T], out_b: &mut [T]);

    fn hermes<T: BenchmarkFloat, A: SimdArch + SimdKernel<T>>(
        a: Vector<T, A>,
        b: Vector<T, A>,
    ) -> (Vector<T, A>, Vector<T, A>);

    fn fearless<T: BenchmarkFloat, S: FearlessSimd>(
        a: T::Fearless<S>,
        b: T::Fearless<S>,
    ) -> (T::Fearless<S>, T::Fearless<S>);
}

struct Interleave;

impl PermuteOperation for Interleave {
    const LABEL: &'static str = "interleave";

    fn reference<T: BenchmarkFloat>(a: &[T], b: &[T], out_a: &mut [T], out_b: &mut [T]) {
        out_a
            .iter_mut()
            .chain(out_b)
            .zip(a.iter().zip(b).flat_map(|(&a, &b)| [a, b]))
            .for_each(|(out, x)| *out = x);
    }

    #[inline(always)]
    fn hermes<T: BenchmarkFloat, A: SimdArch + SimdKernel<T>>(
        a: Vector<T, A>,
        b: Vector<T, A>,
    ) -> (Vector<T, A>, Vector<T, A>) {
        a.interleave(b)
    }

    #[inline(always)]
    fn fearless<T: BenchmarkFloat, S: FearlessSimd>(
        a: T::Fearless<S>,
        b: T::Fearless<S>,
    ) -> (T::Fearless<S>, T::Fearless<S>) {
        a.interleave(b)
    }
}

struct Deinterleave;

impl PermuteOperation for Deinterleave {
    const LABEL: &'static str = "deinterleave";

    fn reference<T: BenchmarkFloat>(a: &[T], b: &[T], out_a: &mut [T], out_b: &mut [T]) {
        out_a
            .iter_mut()
            .zip(a.iter().chain(b).step_by(2))
            .for_each(|(out, &x)| *out = x);
        out_b
            .iter_mut()
            .zip(a.iter().chain(b).skip(1).step_by(2))
            .for_each(|(out, &x)| *out = x);
    }

    #[inline(always)]
    fn hermes<T: BenchmarkFloat, A: SimdArch + SimdKernel<T>>(
        a: Vector<T, A>,
        b: Vector<T, A>,
    ) -> (Vector<T, A>, Vector<T, A>) {
        a.deinterleave(b)
    }

    #[inline(always)]
    fn fearless<T: BenchmarkFloat, S: FearlessSimd>(
        a: T::Fearless<S>,
        b: T::Fearless<S>,
    ) -> (T::Fearless<S>, T::Fearless<S>) {
        a.deinterleave(b)
    }
}

struct HermesPermute<'a, T, O> {
    a: &'a [T],
    b: &'a [T],
    out_a: &'a mut [T],
    out_b: &'a mut [T],
    operation: PhantomData<O>,
}

impl<T: BenchmarkFloat, O: PermuteOperation> LaneKernel<T> for HermesPermute<'_, T, O> {
    type Output = ();

    #[inline(always)]
    fn call<A: SimdArch + SimdKernel<T>>(self, simd: Simd<T, A>) {
        let mut chunks = simd.io_chunks([self.a, self.b], [self.out_a, self.out_b]);
        for ([a, b], [mut out_a, mut out_b]) in &mut chunks {
            let (a, b) = O::hermes(a.load(), b.load());
            out_a.store(a);
            out_b.store(b);
        }
        let (input_tails, output_tails) = chunks.into_remainders();
        debug_assert!(input_tails.iter().all(|tail| tail.is_empty()));
        debug_assert!(output_tails.iter().all(|tail| tail.is_empty()));
    }
}

#[inline(always)]
fn fearless_permute<T: BenchmarkFloat, O: PermuteOperation, S: FearlessSimd>(
    simd: S,
    a: &[T],
    b: &[T],
    out_a: &mut [T],
    out_b: &mut [T],
) {
    let lanes = T::Fearless::<S>::N;
    for (((a, b), out_a), out_b) in a
        .chunks_exact(lanes)
        .zip(b.chunks_exact(lanes))
        .zip(out_a.chunks_exact_mut(lanes))
        .zip(out_b.chunks_exact_mut(lanes))
    {
        let a = T::Fearless::<S>::from_slice(simd, a);
        let b = T::Fearless::<S>::from_slice(simd, b);
        let (a, b): (T::Fearless<S>, T::Fearless<S>) = O::fearless::<T, S>(a, b);
        a.store_slice(out_a);
        b.store_slice(out_b);
    }
}

fn reference<T: BenchmarkFloat, O: PermuteOperation>(
    a: &[T],
    b: &[T],
    out_a: &mut [T],
    out_b: &mut [T],
    lanes: usize,
) {
    for (((a, b), out_a), out_b) in a
        .chunks_exact(lanes)
        .zip(b.chunks_exact(lanes))
        .zip(out_a.chunks_exact_mut(lanes))
        .zip(out_b.chunks_exact_mut(lanes))
    {
        O::reference(a, b, out_a, out_b);
    }
}

fn inputs<T: BenchmarkFloat>(len: usize) -> (Vec<T>, Vec<T>) {
    let a = (0..len)
        .map(|index| T::from_f64(f64::from(u32::try_from(index).expect("length fits u32"))))
        .collect();
    let b = (0..len)
        .map(|index| {
            T::from_f64(-f64::from(
                u32::try_from(index + len).expect("length fits u32"),
            ))
        })
        .collect();
    (a, b)
}

fn bench_operation<T: BenchmarkFloat, O: PermuteOperation>(c: &mut Criterion) {
    let level = Level::new();
    let mut group = c.benchmark_group(format!("cross_lane_{}_{}", O::LABEL, T::LABEL));
    for &len in SCALAR_LENGTHS {
        group.throughput(Throughput::Elements(
            u64::try_from(2 * len).expect("benchmark length fits u64"),
        ));
        let (a, b) = inputs::<T>(len);
        let mut out_a = vec![T::ZERO; len];
        let mut out_b = vec![T::ZERO; len];
        let mut expected_a = vec![T::ZERO; len];
        let mut expected_b = vec![T::ZERO; len];

        let hermes_lanes = hermes_lane_count::<T>();
        let fearless_lanes =
            fearless_simd::dispatch!(level, simd => fearless_lane_count::<T, _>(simd));
        assert_eq!(
            hermes_lanes, fearless_lanes,
            "the comparison requires equal native vector widths"
        );
        reference::<T, O>(&a, &b, &mut expected_a, &mut expected_b, hermes_lanes);

        vectorize(HermesPermute::<T, O> {
            a: &a,
            b: &b,
            out_a: &mut out_a,
            out_b: &mut out_b,
            operation: PhantomData,
        });
        assert_eq!(out_a, expected_a);
        assert_eq!(out_b, expected_b);

        fearless_simd::dispatch!(level, simd => fearless_permute::<T, O, _>(
            simd,
            &a,
            &b,
            &mut out_a,
            &mut out_b,
        ));
        assert_eq!(out_a, expected_a);
        assert_eq!(out_b, expected_b);

        group.bench_function(BenchmarkId::new("hermes", len), |bencher| {
            bencher.iter(|| {
                vectorize(HermesPermute::<T, O> {
                    a: black_box(&a),
                    b: black_box(&b),
                    out_a: black_box(&mut out_a),
                    out_b: black_box(&mut out_b),
                    operation: PhantomData,
                });
                black_box((&out_a, &out_b));
            });
        });
        group.bench_function(BenchmarkId::new("fearless_simd", len), |bencher| {
            bencher.iter(|| {
                fearless_simd::dispatch!(level, simd => fearless_permute::<T, O, _>(
                    simd,
                    black_box(&a),
                    black_box(&b),
                    black_box(&mut out_a),
                    black_box(&mut out_b),
                ));
                black_box((&out_a, &out_b));
            });
        });
    }
    group.finish();
}

pub(super) fn bench(c: &mut Criterion) {
    bench_operation::<f32, Interleave>(c);
    bench_operation::<f64, Interleave>(c);
    bench_operation::<f32, Deinterleave>(c);
    bench_operation::<f64, Deinterleave>(c);
}
