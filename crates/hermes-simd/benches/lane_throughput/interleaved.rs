//! Interleaved complex diagnostics and provider comparison.

use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, Throughput};
use eunomia::FloatElement;
use hermes_simd::{vectorize, LaneKernel, Simd, SimdArch, SimdKernel, SimdStorage, Vector};

use super::comparison::{assert_within_rounding, BenchmarkFloat};
use super::SCALAR_LENGTHS;

#[path = "interleaved/provider.rs"]
mod provider;

fn inputs<T: BenchmarkFloat>(len: usize) -> (Vec<T>, Vec<T>, Vec<T>) {
    let a = (0..len)
        .map(|index| {
            T::from_f64(f64::from(
                u32::try_from(index % 29).expect("benchmark residue fits u32"),
            ))
            .scalar_fmadd(T::from_f64(0.03125), T::from_f64(-0.375))
        })
        .collect();
    let b = (0..len)
        .map(|index| {
            T::from_f64(f64::from(
                u32::try_from(index % 31).expect("benchmark residue fits u32"),
            ))
            .scalar_fmadd(T::from_f64(-0.015_625), T::from_f64(0.625))
        })
        .collect();
    let twiddle = (0..len / 2)
        .flat_map(|index| {
            let index = T::from_f64(f64::from(
                u32::try_from(index).expect("benchmark length fits u32"),
            ));
            let sample_count = T::from_f64(f64::from(
                u32::try_from(len / 2).expect("benchmark length fits u32"),
            ));
            let theta = -T::from_f64(core::f64::consts::TAU) * index / sample_count;
            [FloatElement::cos(theta), FloatElement::sin(theta)]
        })
        .collect();
    (a, b, twiddle)
}

fn scalar_reference<T: BenchmarkFloat>(a: &[T], b: &[T], twiddle: &[T]) -> (Vec<T>, Vec<T>) {
    let mut sum = Vec::with_capacity(a.len());
    let mut difference = Vec::with_capacity(a.len());
    for ((a_pair, b_pair), twiddle_pair) in a
        .chunks_exact(2)
        .zip(b.chunks_exact(2))
        .zip(twiddle.chunks_exact(2))
    {
        let product_real = twiddle_pair[0].scalar_fmadd(b_pair[0], -(twiddle_pair[1] * b_pair[1]));
        let product_imaginary =
            twiddle_pair[0].scalar_fmadd(b_pair[1], twiddle_pair[1] * b_pair[0]);
        sum.extend([a_pair[0] + product_real, a_pair[1] + product_imaginary]);
        difference.extend([a_pair[0] - product_real, a_pair[1] - product_imaginary]);
    }
    (sum, difference)
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

pub(super) fn bench(c: &mut Criterion) {
    let mut group = c.benchmark_group("interleaved_complex_butterfly_f64");
    for &len in SCALAR_LENGTHS {
        group.throughput(Throughput::Elements((len / 2) as u64));
        let (a, b, twiddle) = inputs::<f64>(len);
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
            assert_within_rounding(&out_sum, &expected_sum, f64::EPSILON);
            assert_within_rounding(&out_difference, &expected_difference, f64::EPSILON);
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

    provider::bench::<f32>(c);
    provider::bench::<f64>(c);
}
