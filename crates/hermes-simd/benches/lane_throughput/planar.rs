//! Same-binary planar comparison against `fearless_simd`.

use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, Throughput};
use eunomia::FloatElement;
use fearless_simd::prelude::{SimdBase, SimdFloat};
use fearless_simd::{Level, Simd as FearlessSimd};
use hermes_simd::{vectorize, LaneKernel, Simd, SimdArch, SimdKernel};

use super::comparison::{assert_within_rounding, BenchmarkFloat};
use super::SCALAR_LENGTHS;

struct PlanarInputs<T> {
    a_re: Vec<T>,
    a_im: Vec<T>,
    b_re: Vec<T>,
    b_im: Vec<T>,
    tw_re: Vec<T>,
    tw_im: Vec<T>,
}

struct PlanarOutputs<T> {
    out0_re: Vec<T>,
    out0_im: Vec<T>,
    out1_re: Vec<T>,
    out1_im: Vec<T>,
}

impl<T: BenchmarkFloat> PlanarInputs<T> {
    fn new(len: usize) -> Self {
        let tone = |scale: f64, phase: f64| {
            let scale = T::from_f64(scale);
            let phase = T::from_f64(phase);
            (0..len)
                .map(|index| {
                    let index = u32::try_from(index)
                        .expect("benchmark lengths fit the exact u32-to-f64 domain");
                    FloatElement::sin(scale * T::from_f64(f64::from(index)) + phase)
                })
                .collect()
        };
        Self {
            a_re: tone(0.017, 0.0),
            a_im: tone(0.031, 0.4),
            b_re: tone(0.023, 0.9),
            b_im: tone(0.041, 1.3),
            tw_re: tone(0.011, 2.1),
            tw_im: tone(0.037, 2.7),
        }
    }

    fn reference(&self) -> PlanarOutputs<T> {
        let mut out = PlanarOutputs::zeroed(self.a_re.len());
        let two = T::from_f64(2.0);
        for index in 0..self.a_re.len() {
            let out0_re = self.tw_im[index].scalar_fmadd(
                -self.b_im[index],
                self.tw_re[index].scalar_fmadd(self.b_re[index], self.a_re[index]),
            );
            let out0_im = self.tw_im[index].scalar_fmadd(
                self.b_re[index],
                self.tw_re[index].scalar_fmadd(self.b_im[index], self.a_im[index]),
            );
            out.out0_re[index] = out0_re;
            out.out0_im[index] = out0_im;
            out.out1_re[index] = two.scalar_fmadd(self.a_re[index], -out0_re);
            out.out1_im[index] = two.scalar_fmadd(self.a_im[index], -out0_im);
        }
        out
    }
}

impl<T: BenchmarkFloat> PlanarOutputs<T> {
    fn zeroed(len: usize) -> Self {
        Self {
            out0_re: vec![T::ZERO; len],
            out0_im: vec![T::ZERO; len],
            out1_re: vec![T::ZERO; len],
            out1_im: vec![T::ZERO; len],
        }
    }

    fn assert_matches(&self, expected: &Self) {
        assert_within_rounding(&self.out0_re, &expected.out0_re, T::EPSILON);
        assert_within_rounding(&self.out0_im, &expected.out0_im, T::EPSILON);
        assert_within_rounding(&self.out1_re, &expected.out1_re, T::EPSILON);
        assert_within_rounding(&self.out1_im, &expected.out1_im, T::EPSILON);
    }
}

struct HermesPlanar<'a, T> {
    input: &'a PlanarInputs<T>,
    output: &'a mut PlanarOutputs<T>,
}

impl<T: BenchmarkFloat> LaneKernel<T> for HermesPlanar<'_, T> {
    type Output = ();

    #[inline(always)]
    fn call<A: SimdArch + SimdKernel<T>>(self, simd: Simd<T, A>) {
        let two = simd.splat(T::from_f64(2.0));
        let mut chunks = simd.io_chunks(
            [
                self.input.a_re.as_slice(),
                self.input.a_im.as_slice(),
                self.input.b_re.as_slice(),
                self.input.b_im.as_slice(),
                self.input.tw_re.as_slice(),
                self.input.tw_im.as_slice(),
            ],
            [
                self.output.out0_re.as_mut_slice(),
                self.output.out0_im.as_mut_slice(),
                self.output.out1_re.as_mut_slice(),
                self.output.out1_im.as_mut_slice(),
            ],
        );

        for (
            [a_re, a_im, b_re, b_im, tw_re, tw_im],
            [mut out0_re, mut out0_im, mut out1_re, mut out1_im],
        ) in &mut chunks
        {
            let a_re = a_re.load();
            let a_im = a_im.load();
            let b_re = b_re.load();
            let b_im = b_im.load();
            let tw_re = tw_re.load();
            let tw_im = tw_im.load();
            let first_re = tw_im.mul_add(-b_im, tw_re.mul_add(b_re, a_re));
            let first_im = tw_im.mul_add(b_re, tw_re.mul_add(b_im, a_im));
            out0_re.store(first_re);
            out0_im.store(first_im);
            out1_re.store(two.mul_sub(a_re, first_re));
            out1_im.store(two.mul_sub(a_im, first_im));
        }
        let (input_tails, output_tails) = chunks.into_remainders();
        debug_assert!(input_tails.iter().all(|tail| tail.is_empty()));
        debug_assert!(output_tails.iter().all(|tail| tail.is_empty()));
    }
}

#[inline(always)]
fn fearless_planar<T: BenchmarkFloat, S: FearlessSimd>(
    simd: S,
    input: &PlanarInputs<T>,
    output: &mut PlanarOutputs<T>,
) {
    let lanes = T::Fearless::<S>::N;
    let two = T::Fearless::<S>::splat(simd, T::from_f64(2.0));
    let inputs = input
        .a_re
        .chunks_exact(lanes)
        .zip(input.a_im.chunks_exact(lanes))
        .zip(input.b_re.chunks_exact(lanes))
        .zip(input.b_im.chunks_exact(lanes))
        .zip(input.tw_re.chunks_exact(lanes))
        .zip(input.tw_im.chunks_exact(lanes));
    let outputs = output
        .out0_re
        .chunks_exact_mut(lanes)
        .zip(output.out0_im.chunks_exact_mut(lanes))
        .zip(output.out1_re.chunks_exact_mut(lanes))
        .zip(output.out1_im.chunks_exact_mut(lanes));

    for (
        (((((a_re, a_im), b_re), b_im), tw_re), tw_im),
        (((out0_re, out0_im), out1_re), out1_im),
    ) in inputs.zip(outputs)
    {
        let a_re = T::Fearless::<S>::from_slice(simd, a_re);
        let a_im = T::Fearless::<S>::from_slice(simd, a_im);
        let b_re = T::Fearless::<S>::from_slice(simd, b_re);
        let b_im = T::Fearless::<S>::from_slice(simd, b_im);
        let tw_re = T::Fearless::<S>::from_slice(simd, tw_re);
        let tw_im = T::Fearless::<S>::from_slice(simd, tw_im);
        let first_re = tw_im.mul_add(-b_im, tw_re.mul_add(b_re, a_re));
        let first_im = tw_im.mul_add(b_re, tw_re.mul_add(b_im, a_im));
        let second_re = two.mul_sub(a_re, first_re);
        let second_im = two.mul_sub(a_im, first_im);
        first_re.store_slice(out0_re);
        first_im.store_slice(out0_im);
        second_re.store_slice(out1_re);
        second_im.store_slice(out1_im);
    }
}

fn bench_precision<T: BenchmarkFloat>(c: &mut Criterion) {
    let level = Level::new();
    let mut group = c.benchmark_group(format!("planar_complex_butterfly_{}", T::LABEL));
    for &len in SCALAR_LENGTHS {
        group.throughput(Throughput::Elements(len as u64));
        let input = PlanarInputs::<T>::new(len);
        let expected = input.reference();

        // Both candidates reuse these exact output addresses. Separate Vec
        // allocations can map to different cache sets and confound a
        // same-operation substrate comparison even when codegen is identical.
        let mut output = PlanarOutputs::zeroed(len);
        vectorize(HermesPlanar {
            input: &input,
            output: &mut output,
        });
        output.assert_matches(&expected);
        group.bench_function(BenchmarkId::new("hermes", len), |bencher| {
            bencher.iter(|| {
                vectorize(HermesPlanar {
                    input: black_box(&input),
                    output: black_box(&mut output),
                });
                black_box(&output);
            });
        });

        fearless_simd::dispatch!(level, simd =>
            fearless_planar::<T, _>(simd, &input, &mut output)
        );
        output.assert_matches(&expected);
        group.bench_function(BenchmarkId::new("fearless_simd", len), |bencher| {
            bencher.iter(|| {
                fearless_simd::dispatch!(level, simd => fearless_planar::<T, _>(
                    simd,
                    black_box(&input),
                    black_box(&mut output),
                ));
                black_box(&output);
            });
        });
    }
    group.finish();
}

pub(super) fn bench(c: &mut Criterion) {
    bench_precision::<f32>(c);
    bench_precision::<f64>(c);
}
