//! Runtime-dispatch boundary cost independent of kernel throughput.

use core::marker::PhantomData;

use criterion::{black_box, Criterion};
use fearless_simd::prelude::SimdBase;
use fearless_simd::{Level, Simd as FearlessSimd};
use hermes_simd::{vectorize, LaneKernel, Simd, SimdArch, SimdKernel, SimdStorage};

use super::comparison::BenchmarkFloat;

const INPUT_CANARY: u64 = 1;

struct HermesDispatch<T> {
    input: u64,
    marker: PhantomData<T>,
}

impl<T: BenchmarkFloat> LaneKernel<T> for HermesDispatch<T> {
    type Output = (u64, usize);

    #[inline(always)]
    fn call<A: SimdArch + SimdKernel<T>>(self, _simd: Simd<T, A>) -> Self::Output {
        (self.input, <A as SimdStorage<T>>::LANE_COUNT)
    }
}

// Keep the public dispatch boundary inside each timed iteration. Without this
// frame LLVM may hoist the input-independent target decision out of Criterion's
// generated loop and reduce every provider to the direct tuple control.
#[inline(never)]
fn hermes_dispatch<T: BenchmarkFloat>(input: u64) -> (u64, usize) {
    vectorize(HermesDispatch::<T> {
        input,
        marker: PhantomData,
    })
}

#[inline(always)]
fn fearless_dispatch<T, S>(input: u64, _simd: S) -> (u64, usize)
where
    T: BenchmarkFloat,
    S: FearlessSimd,
{
    (input, T::Fearless::<S>::N)
}

#[inline(never)]
fn fearless_new<T: BenchmarkFloat>(input: u64) -> (u64, usize) {
    let level = Level::new();
    fearless_simd::dispatch!(level, simd => fearless_dispatch::<T, _>(input, simd))
}

#[inline(never)]
fn fearless_reused<T: BenchmarkFloat>(level: Level, input: u64) -> (u64, usize) {
    fearless_simd::dispatch!(level, simd => fearless_dispatch::<T, _>(input, simd))
}

#[inline(never)]
fn direct_control(input: u64, lanes: usize) -> (u64, usize) {
    (input, lanes)
}

fn bench_precision<T: BenchmarkFloat>(c: &mut Criterion) {
    let reused_level = Level::new();
    let expected = hermes_dispatch::<T>(INPUT_CANARY);
    let detected = fearless_reused::<T>(reused_level, INPUT_CANARY);
    assert_eq!(
        detected, expected,
        "providers selected different native widths"
    );

    let mut group = c.benchmark_group(format!("dispatch_boundary_{}", T::LABEL));
    group.bench_function("direct_control", |bencher| {
        bencher.iter(|| {
            black_box(direct_control(
                black_box(INPUT_CANARY),
                black_box(expected.1),
            ))
        });
    });
    group.bench_function("hermes", |bencher| {
        bencher.iter(|| black_box(hermes_dispatch::<T>(black_box(INPUT_CANARY))));
    });
    group.bench_function("fearless_new", |bencher| {
        bencher.iter(|| black_box(fearless_new::<T>(black_box(INPUT_CANARY))));
    });
    group.bench_function("fearless_reused", |bencher| {
        bencher.iter(|| {
            black_box(fearless_reused::<T>(
                black_box(reused_level),
                black_box(INPUT_CANARY),
            ))
        });
    });
    group.finish();
}

pub(super) fn bench(c: &mut Criterion) {
    bench_precision::<f32>(c);
    bench_precision::<f64>(c);
}
