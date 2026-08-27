//! Provider comparison for interleaved complex-register butterflies.

use core::marker::PhantomData;
use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, Throughput};
use fearless_simd::prelude::{SimdBase, SimdFloat};
use fearless_simd::{Level, Simd as FearlessSimd};
use hermes_simd::{
    vectorize, ComplexReg, LaneKernel, Simd, SimdArch, SimdChunk, SimdKernel, Unmasked, Vector,
};

use super::super::comparison::{
    assert_within_rounding, fearless_lane_count, hermes_lane_count, BenchmarkFloat,
};
use super::super::SCALAR_LENGTHS;
use super::{butterfly, inputs, scalar_reference};

trait HermesButterfly<T: BenchmarkFloat> {
    fn apply<A: SimdArch + SimdKernel<T>>(
        a: Vector<T, A>,
        b: Vector<T, A>,
        twiddle: Vector<T, A>,
    ) -> (Vector<T, A>, Vector<T, A>);
}

struct VectorRecipe;

impl<T: BenchmarkFloat> HermesButterfly<T> for VectorRecipe {
    #[inline(always)]
    fn apply<A: SimdArch + SimdKernel<T>>(
        a: Vector<T, A>,
        b: Vector<T, A>,
        twiddle: Vector<T, A>,
    ) -> (Vector<T, A>, Vector<T, A>) {
        butterfly(a, b, twiddle)
    }
}

struct ComplexRegisterRecipe;

impl<T: BenchmarkFloat> HermesButterfly<T> for ComplexRegisterRecipe {
    #[inline(always)]
    fn apply<A: SimdArch + SimdKernel<T>>(
        a: Vector<T, A>,
        b: Vector<T, A>,
        twiddle: Vector<T, A>,
    ) -> (Vector<T, A>, Vector<T, A>) {
        let a = ComplexReg::from_interleaved(a);
        let product = ComplexReg::from_interleaved(twiddle) * ComplexReg::from_interleaved(b);
        let (sum, difference) = a.butterfly(product);
        (sum.into_interleaved(), difference.into_interleaved())
    }
}

struct HermesPaired<'a, T, B> {
    a: &'a [T],
    b: &'a [T],
    twiddle: &'a [T],
    out_sum: &'a mut [T],
    out_difference: &'a mut [T],
    recipe: PhantomData<B>,
}

#[inline(always)]
fn process_hermes_chunk<'input, 'output, T, A, B>(
    [a, b, twiddle]: [SimdChunk<'input, T, A, Unmasked, &'input [T]>; 3],
    [mut out_sum, mut out_difference]: [SimdChunk<'output, T, A, Unmasked, &'output mut [T]>; 2],
) where
    T: BenchmarkFloat,
    A: SimdArch + SimdKernel<T>,
    B: HermesButterfly<T>,
{
    let (sum, difference) = B::apply(a.load(), b.load(), twiddle.load());
    out_sum.store(sum);
    out_difference.store(difference);
}

impl<T, B> LaneKernel<T> for HermesPaired<'_, T, B>
where
    T: BenchmarkFloat,
    B: HermesButterfly<T>,
{
    type Output = ();

    #[inline(always)]
    fn call<A: SimdArch + SimdKernel<T>>(self, simd: Simd<T, A>) {
        let mut chunks = simd.io_chunks(
            [self.a, self.b, self.twiddle],
            [self.out_sum, self.out_difference],
        );
        debug_assert_eq!(chunks.chunks_remaining() % 2, 0);
        while let Some(first) = chunks.next() {
            let second = chunks
                .next()
                .expect("benchmark input contains complete paired register groups");
            process_hermes_chunk::<T, A, B>(first.0, first.1);
            process_hermes_chunk::<T, A, B>(second.0, second.1);
        }
    }
}

#[inline(always)]
fn fearless_butterfly<T: BenchmarkFloat, S: FearlessSimd>(
    simd: S,
    a: &[T],
    b: &[T],
    twiddle: &[T],
    out_sum: &mut [T],
    out_difference: &mut [T],
) {
    let lanes = T::Fearless::<S>::N;
    let group_len = 2 * lanes;
    for ((((a, b), twiddle), out_sum), out_difference) in a
        .chunks_exact(group_len)
        .zip(b.chunks_exact(group_len))
        .zip(twiddle.chunks_exact(group_len))
        .zip(out_sum.chunks_exact_mut(group_len))
        .zip(out_difference.chunks_exact_mut(group_len))
    {
        let (a0, a1) = a.split_at(lanes);
        let (b0, b1) = b.split_at(lanes);
        let (twiddle0, twiddle1) = twiddle.split_at(lanes);
        let a0 = T::Fearless::<S>::from_slice(simd, a0);
        let a1 = T::Fearless::<S>::from_slice(simd, a1);
        let b0 = T::Fearless::<S>::from_slice(simd, b0);
        let b1 = T::Fearless::<S>::from_slice(simd, b1);
        let twiddle0 = T::Fearless::<S>::from_slice(simd, twiddle0);
        let twiddle1 = T::Fearless::<S>::from_slice(simd, twiddle1);

        let (a_re, a_im) = a0.deinterleave(a1);
        let (b_re, b_im) = b0.deinterleave(b1);
        let (twiddle_re, twiddle_im) = twiddle0.deinterleave(twiddle1);
        let product_re = twiddle_re.mul_add(b_re, -(twiddle_im * b_im));
        let product_im = twiddle_re.mul_add(b_im, twiddle_im * b_re);
        let (sum0, sum1) = (a_re + product_re).interleave(a_im + product_im);
        let (difference0, difference1) = (a_re - product_re).interleave(a_im - product_im);

        let (out_sum0, out_sum1) = out_sum.split_at_mut(lanes);
        let (out_difference0, out_difference1) = out_difference.split_at_mut(lanes);
        sum0.store_slice(out_sum0);
        sum1.store_slice(out_sum1);
        difference0.store_slice(out_difference0);
        difference1.store_slice(out_difference1);
    }
}

struct Comparison<'a, T> {
    a: &'a [T],
    b: &'a [T],
    twiddle: &'a [T],
    out_sum: &'a mut [T],
    out_difference: &'a mut [T],
    expected_sum: &'a [T],
    expected_difference: &'a [T],
}

fn bench_hermes_recipe<T, B>(
    group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>,
    label: &str,
    comparison: &mut Comparison<'_, T>,
) where
    T: BenchmarkFloat,
    B: HermesButterfly<T>,
{
    vectorize(HermesPaired::<T, B> {
        a: comparison.a,
        b: comparison.b,
        twiddle: comparison.twiddle,
        out_sum: comparison.out_sum,
        out_difference: comparison.out_difference,
        recipe: PhantomData,
    });
    assert_within_rounding(comparison.out_sum, comparison.expected_sum, T::EPSILON);
    assert_within_rounding(
        comparison.out_difference,
        comparison.expected_difference,
        T::EPSILON,
    );
    group.bench_function(BenchmarkId::new(label, comparison.a.len()), |bencher| {
        bencher.iter(|| {
            vectorize(HermesPaired::<T, B> {
                a: black_box(comparison.a),
                b: black_box(comparison.b),
                twiddle: black_box(comparison.twiddle),
                out_sum: black_box(&mut *comparison.out_sum),
                out_difference: black_box(&mut *comparison.out_difference),
                recipe: PhantomData,
            });
            black_box((&comparison.out_sum, &comparison.out_difference));
        });
    });
}

pub(super) fn bench<T: BenchmarkFloat>(c: &mut Criterion) {
    let level = Level::new();
    let mut group = c.benchmark_group(format!("complex_register_butterfly_{}", T::LABEL));
    for &len in SCALAR_LENGTHS {
        group.throughput(Throughput::Elements(
            u64::try_from(len / 2).expect("benchmark length fits u64"),
        ));
        let (a, b, twiddle) = inputs::<T>(len);
        let (expected_sum, expected_difference) = scalar_reference(&a, &b, &twiddle);
        let mut out_sum = vec![T::ZERO; len];
        let mut out_difference = vec![T::ZERO; len];

        let hermes_lanes = hermes_lane_count::<T>();
        let fearless_lanes =
            fearless_simd::dispatch!(level, simd => fearless_lane_count::<T, _>(simd));
        assert_eq!(
            hermes_lanes, fearless_lanes,
            "the comparison requires equal native vector widths"
        );
        assert_eq!(
            len % (2 * hermes_lanes),
            0,
            "the comparison requires complete paired register groups"
        );

        let mut comparison = Comparison {
            a: &a,
            b: &b,
            twiddle: &twiddle,
            out_sum: &mut out_sum,
            out_difference: &mut out_difference,
            expected_sum: &expected_sum,
            expected_difference: &expected_difference,
        };
        bench_hermes_recipe::<T, VectorRecipe>(&mut group, "hermes_vector", &mut comparison);
        bench_hermes_recipe::<T, ComplexRegisterRecipe>(
            &mut group,
            "hermes_complex_reg",
            &mut comparison,
        );

        fearless_simd::dispatch!(level, simd => fearless_butterfly::<T, _>(
            simd,
            &a,
            &b,
            &twiddle,
            &mut out_sum,
            &mut out_difference,
        ));
        assert_within_rounding(&out_sum, &expected_sum, T::EPSILON);
        assert_within_rounding(&out_difference, &expected_difference, T::EPSILON);
        group.bench_function(BenchmarkId::new("fearless_simd", len), |bencher| {
            bencher.iter(|| {
                fearless_simd::dispatch!(level, simd => fearless_butterfly::<T, _>(
                    simd,
                    black_box(&a),
                    black_box(&b),
                    black_box(&twiddle),
                    black_box(&mut out_sum),
                    black_box(&mut out_difference),
                ));
                black_box((&out_sum, &out_difference));
            });
        });
    }
    group.finish();
}
