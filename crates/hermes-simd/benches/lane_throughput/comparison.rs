//! Scalar, numerical, and native-width contracts shared by the benchmarks.

use core::marker::PhantomData;
use core::ops::Neg;

use eunomia::FloatElement;
use fearless_simd::prelude::{SimdBase, SimdFloat};
use fearless_simd::{Simd as FearlessSimd, SimdFloatElement};
use hermes_simd::{vectorize, LaneKernel, LaneScalar, Simd, SimdArch, SimdKernel, SimdStorage};

/// Floating lane precision exercised by the comparison instruments.
pub(super) trait BenchmarkFloat:
    LaneScalar + FloatElement + SimdFloatElement + Into<f64> + Neg<Output = Self>
{
    /// Native-width vector selected by a `fearless_simd` capability.
    type Fearless<S: FearlessSimd>: SimdFloat<S, Element = Self>;

    /// Stable label used in Criterion group names.
    const LABEL: &'static str;
    /// Machine epsilon expressed exactly in the f64 comparison domain.
    const EPSILON: f64;
}

impl BenchmarkFloat for f32 {
    type Fearless<S: FearlessSimd> = S::f32s;

    const LABEL: &'static str = "f32";
    const EPSILON: f64 = f32::EPSILON as f64;
}

impl BenchmarkFloat for f64 {
    type Fearless<S: FearlessSimd> = S::f64s;

    const LABEL: &'static str = "f64";
    const EPSILON: f64 = f64::EPSILON;
}

struct HermesLaneCount<T>(PhantomData<T>);

impl<T: BenchmarkFloat> LaneKernel<T> for HermesLaneCount<T> {
    type Output = usize;

    fn call<A: SimdArch + SimdKernel<T>>(self, _simd: Simd<T, A>) -> Self::Output {
        <A as SimdStorage<T>>::LANE_COUNT
    }
}

/// Native Hermes lane count selected on this host.
pub(super) fn hermes_lane_count<T: BenchmarkFloat>() -> usize {
    vectorize(HermesLaneCount::<T>(PhantomData))
}

/// Native Fearless lane count selected by one capability token.
pub(super) fn fearless_lane_count<T: BenchmarkFloat, S: FearlessSimd>(_simd: S) -> usize {
    T::Fearless::<S>::N
}

/// Assert a butterfly result against its scalar reference.
pub(super) fn assert_within_rounding<T>(actual: &[T], expected: &[T], epsilon: f64)
where
    T: Copy + Into<f64>,
{
    for (&actual, &expected) in actual.iter().zip(expected) {
        let actual = actual.into();
        let expected = expected.into();
        // The butterfly has at most four rounded operations along either output
        // path. `8 * epsilon * max(1, |expected|)` conservatively covers that
        // depth plus backend FMA/sign-arrangement differences at this input scale.
        let bound = 8.0 * epsilon * expected.abs().max(1.0);
        assert!(
            (actual - expected).abs() <= bound,
            "butterfly result {actual} differs from {expected} by more than {bound}"
        );
    }
}
