//! Lane-boundary cost for FFT butterfly kernels.
//!
//! The primary group compares Hermes and `fearless_simd` on the same planar,
//! native-width f64 butterfly used by `PhastFT`. The secondary interleaved group
//! isolates Hermes wrapper overhead by holding its arithmetic and dispatch
//! constant across checked, view/chunk, and direct backend paths.

use core::time::Duration;
use std::hint::black_box;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use fearless_simd::prelude::{SimdBase, SimdFloat};
use fearless_simd::{Level, Simd as FearlessSimd};
use hermes_simd::{vectorize, LaneKernel, Simd, SimdArch, SimdKernel, SimdStorage, Vector};

const SCALAR_LENGTHS: &[usize] = &[256, 1_024, 4_096];

fn inputs(len: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let a = (0..len)
        .map(|index| ((index % 29) as f64).mul_add(0.03125, -0.375))
        .collect();
    let b = (0..len)
        .map(|index| ((index % 31) as f64).mul_add(-0.015_625, 0.625))
        .collect();
    let twiddle = (0..len / 2)
        .flat_map(|index| {
            let theta = -core::f64::consts::TAU * index as f64 / (len / 2) as f64;
            let (sin, cos) = theta.sin_cos();
            [cos, sin]
        })
        .collect();
    (a, b, twiddle)
}

fn scalar_reference(a: &[f64], b: &[f64], twiddle: &[f64]) -> (Vec<f64>, Vec<f64>) {
    let mut sum = Vec::with_capacity(a.len());
    let mut difference = Vec::with_capacity(a.len());
    for ((a_pair, b_pair), twiddle_pair) in a
        .chunks_exact(2)
        .zip(b.chunks_exact(2))
        .zip(twiddle.chunks_exact(2))
    {
        let product_real = twiddle_pair[0].mul_add(b_pair[0], -(twiddle_pair[1] * b_pair[1]));
        let product_imaginary = twiddle_pair[0].mul_add(b_pair[1], twiddle_pair[1] * b_pair[0]);
        sum.extend([a_pair[0] + product_real, a_pair[1] + product_imaginary]);
        difference.extend([a_pair[0] - product_real, a_pair[1] - product_imaginary]);
    }
    (sum, difference)
}

fn assert_matches_reference(actual: &[f64], expected: &[f64]) {
    for (&actual, &expected) in actual.iter().zip(expected) {
        // The butterfly has at most four rounded operations along either output
        // path. `8 * epsilon * max(1, |expected|)` conservatively covers that
        // depth plus backend FMA/sign-arrangement differences at this input scale.
        let bound = 8.0 * f64::EPSILON * expected.abs().max(1.0);
        assert!(
            (actual - expected).abs() <= bound,
            "butterfly result {actual} differs from {expected} by more than {bound}"
        );
    }
}

struct PlanarInputs {
    a_re: Vec<f64>,
    a_im: Vec<f64>,
    b_re: Vec<f64>,
    b_im: Vec<f64>,
    tw_re: Vec<f64>,
    tw_im: Vec<f64>,
}

struct PlanarOutputs {
    out0_re: Vec<f64>,
    out0_im: Vec<f64>,
    out1_re: Vec<f64>,
    out1_im: Vec<f64>,
}

impl PlanarInputs {
    fn new(len: usize) -> Self {
        let tone = |scale: f64, phase: f64| {
            (0..len)
                .map(|index| (scale.mul_add(index as f64, phase)).sin())
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

    fn reference(&self) -> PlanarOutputs {
        let mut out = PlanarOutputs::zeroed(self.a_re.len());
        for index in 0..self.a_re.len() {
            let out0_re = self.tw_im[index].mul_add(
                -self.b_im[index],
                self.tw_re[index].mul_add(self.b_re[index], self.a_re[index]),
            );
            let out0_im = self.tw_im[index].mul_add(
                self.b_re[index],
                self.tw_re[index].mul_add(self.b_im[index], self.a_im[index]),
            );
            out.out0_re[index] = out0_re;
            out.out0_im[index] = out0_im;
            out.out1_re[index] = 2.0f64.mul_add(self.a_re[index], -out0_re);
            out.out1_im[index] = 2.0f64.mul_add(self.a_im[index], -out0_im);
        }
        out
    }
}

impl PlanarOutputs {
    fn zeroed(len: usize) -> Self {
        Self {
            out0_re: vec![0.0; len],
            out0_im: vec![0.0; len],
            out1_re: vec![0.0; len],
            out1_im: vec![0.0; len],
        }
    }

    fn assert_matches(&self, expected: &Self) {
        assert_matches_reference(&self.out0_re, &expected.out0_re);
        assert_matches_reference(&self.out0_im, &expected.out0_im);
        assert_matches_reference(&self.out1_re, &expected.out1_re);
        assert_matches_reference(&self.out1_im, &expected.out1_im);
    }
}

struct HermesPlanar<'a> {
    input: &'a PlanarInputs,
    output: &'a mut PlanarOutputs,
}

impl LaneKernel<f64> for HermesPlanar<'_> {
    type Output = ();

    #[inline(always)]
    fn call<A: SimdArch + SimdKernel<f64>>(self, simd: Simd<f64, A>) {
        let two = simd.splat(2.0);
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
fn fearless_planar<S: FearlessSimd>(simd: S, input: &PlanarInputs, output: &mut PlanarOutputs) {
    let lanes = S::f64s::N;
    let two = S::f64s::splat(simd, 2.0);
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
        let a_re = S::f64s::from_slice(simd, a_re);
        let a_im = S::f64s::from_slice(simd, a_im);
        let b_re = S::f64s::from_slice(simd, b_re);
        let b_im = S::f64s::from_slice(simd, b_im);
        let tw_re = S::f64s::from_slice(simd, tw_re);
        let tw_im = S::f64s::from_slice(simd, tw_im);
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

fn bench_planar_comparison(c: &mut Criterion) {
    let level = Level::new();
    let mut group = c.benchmark_group("planar_complex_butterfly_f64");
    for &len in SCALAR_LENGTHS {
        group.throughput(Throughput::Elements(len as u64));
        let input = PlanarInputs::new(len);
        let expected = input.reference();

        // All candidates reuse these exact output addresses. Separate Vec
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
            fearless_planar(simd, &input, &mut output)
        );
        output.assert_matches(&expected);
        group.bench_function(BenchmarkId::new("fearless_simd", len), |bencher| {
            bencher.iter(|| {
                fearless_simd::dispatch!(level, simd => fearless_planar(
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

#[inline(always)]
fn butterfly<T, A>(
    a: Vector<T, A>,
    b: Vector<T, A>,
    twiddle: Vector<T, A>,
) -> (Vector<T, A>, Vector<T, A>)
where
    T: hermes_simd::SimdScalar,
    A: SimdArch + SimdKernel<T>,
{
    let product = twiddle
        .dup_even()
        .fmaddsub(b, twiddle.dup_odd() * b.swap_adjacent());
    (a + product, a - product)
}

struct Checked<'a> {
    a: &'a [f64],
    b: &'a [f64],
    twiddle: &'a [f64],
    out_sum: &'a mut [f64],
    out_difference: &'a mut [f64],
}

impl LaneKernel<f64> for Checked<'_> {
    type Output = ();

    #[inline(always)]
    fn call<A: SimdArch + SimdKernel<f64>>(self, _simd: Simd<f64, A>) {
        let lanes = <A as SimdStorage<f64>>::LANE_COUNT;
        for offset in (0..self.a.len()).step_by(lanes) {
            let end = offset + lanes;
            let a = Vector::<f64, A>::load_unaligned_from_slice(&self.a[offset..end])
                .expect("benchmark input contains complete lane groups");
            let b = Vector::<f64, A>::load_unaligned_from_slice(&self.b[offset..end])
                .expect("benchmark input contains complete lane groups");
            let twiddle = Vector::<f64, A>::load_unaligned_from_slice(&self.twiddle[offset..end])
                .expect("benchmark input contains complete lane groups");
            let (sum, difference) = butterfly(a, b, twiddle);
            sum.store_unaligned_to_slice(&mut self.out_sum[offset..end])
                .expect("benchmark output contains complete lane groups");
            difference
                .store_unaligned_to_slice(&mut self.out_difference[offset..end])
                .expect("benchmark output contains complete lane groups");
        }
    }
}

struct Viewed<'a> {
    a: &'a [f64],
    b: &'a [f64],
    twiddle: &'a [f64],
    out_sum: &'a mut [f64],
    out_difference: &'a mut [f64],
}

impl LaneKernel<f64> for Viewed<'_> {
    type Output = ();

    #[inline(always)]
    fn call<A: SimdArch + SimdKernel<f64>>(self, simd: Simd<f64, A>) {
        let a = simd.view(self.a);
        let b = simd.view(self.b);
        let twiddle = simd.view(self.twiddle);
        let out_sum = simd.view_mut(self.out_sum);
        let out_difference = simd.view_mut(self.out_difference);

        for ((((a, b), twiddle), mut out_sum), mut out_difference) in a
            .simd_chunks()
            .zip(b.simd_chunks())
            .zip(twiddle.simd_chunks())
            .zip(out_sum.simd_chunks_mut())
            .zip(out_difference.simd_chunks_mut())
        {
            let (sum, difference) = butterfly(a.load(), b.load(), twiddle.load());
            out_sum.store(sum);
            out_difference.store(difference);
        }
    }
}

struct Direct<'a> {
    a: &'a [f64],
    b: &'a [f64],
    twiddle: &'a [f64],
    out_sum: &'a mut [f64],
    out_difference: &'a mut [f64],
}

impl LaneKernel<f64> for Direct<'_> {
    type Output = ();

    #[inline(always)]
    fn call<A: SimdArch + SimdKernel<f64>>(self, _simd: Simd<f64, A>) {
        let lanes = <A as SimdStorage<f64>>::LANE_COUNT;
        for offset in (0..self.a.len()).step_by(lanes) {
            // SAFETY: `vectorize` enters a target-feature scope for `A`; every
            // benchmark slice has the same lane-multiple length, and `offset`
            // advances by one complete lane group.
            unsafe {
                let a = A::load_unaligned(self.a.as_ptr().add(offset));
                let b = A::load_unaligned(self.b.as_ptr().add(offset));
                let twiddle = A::load_unaligned(self.twiddle.as_ptr().add(offset));
                let product = A::fmaddsub(
                    A::dup_even(twiddle),
                    b,
                    A::mul(A::dup_odd(twiddle), A::swap_adjacent(b)),
                );
                A::store_unaligned(self.out_sum.as_mut_ptr().add(offset), A::add(a, product));
                A::store_unaligned(
                    self.out_difference.as_mut_ptr().add(offset),
                    A::sub(a, product),
                );
            }
        }
    }
}

fn bench_lane_boundary(c: &mut Criterion) {
    let mut group = c.benchmark_group("interleaved_complex_butterfly_f64");
    for &len in SCALAR_LENGTHS {
        group.throughput(Throughput::Elements((len / 2) as u64));
        let (a, b, twiddle) = inputs(len);
        let (expected_sum, expected_difference) = scalar_reference(&a, &b, &twiddle);
        // Reusing one pair keeps address and cache-set placement identical for
        // the diagnostic variants.
        let mut out_sum = vec![0.0; len];
        let mut out_difference = vec![0.0; len];

        for implementation in ["checked", "view", "direct"] {
            match implementation {
                "checked" => vectorize(Checked {
                    a: &a,
                    b: &b,
                    twiddle: &twiddle,
                    out_sum: &mut out_sum,
                    out_difference: &mut out_difference,
                }),
                "view" => vectorize(Viewed {
                    a: &a,
                    b: &b,
                    twiddle: &twiddle,
                    out_sum: &mut out_sum,
                    out_difference: &mut out_difference,
                }),
                "direct" => vectorize(Direct {
                    a: &a,
                    b: &b,
                    twiddle: &twiddle,
                    out_sum: &mut out_sum,
                    out_difference: &mut out_difference,
                }),
                _ => unreachable!("the benchmark declares exactly three variants"),
            }
            assert_matches_reference(&out_sum, &expected_sum);
            assert_matches_reference(&out_difference, &expected_difference);
            group.bench_with_input(
                BenchmarkId::new(implementation, len),
                &implementation,
                |bencher, implementation| {
                    bencher.iter(|| {
                        match *implementation {
                            "checked" => vectorize(Checked {
                                a: black_box(&a),
                                b: black_box(&b),
                                twiddle: black_box(&twiddle),
                                out_sum: black_box(&mut out_sum),
                                out_difference: black_box(&mut out_difference),
                            }),
                            "view" => vectorize(Viewed {
                                a: black_box(&a),
                                b: black_box(&b),
                                twiddle: black_box(&twiddle),
                                out_sum: black_box(&mut out_sum),
                                out_difference: black_box(&mut out_difference),
                            }),
                            "direct" => vectorize(Direct {
                                a: black_box(&a),
                                b: black_box(&b),
                                twiddle: black_box(&twiddle),
                                out_sum: black_box(&mut out_sum),
                                out_difference: black_box(&mut out_difference),
                            }),
                            _ => unreachable!("the benchmark declares exactly three variants"),
                        }
                        black_box((&out_sum, &out_difference));
                    });
                },
            );
        }
    }
    group.finish();
}

criterion_group! {
    name = benches;
    config = Criterion::default()
        .warm_up_time(Duration::from_millis(20))
        .measurement_time(Duration::from_millis(200))
        .sample_size(20);
    targets = bench_planar_comparison, bench_lane_boundary
}
criterion_main!(benches);
