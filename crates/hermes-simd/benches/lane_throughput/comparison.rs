//! Scalar and numerical contracts shared by the benchmark families.

use core::ops::Neg;

use eunomia::FloatElement;
use fearless_simd::prelude::SimdFloat;
use fearless_simd::{Simd as FearlessSimd, SimdFloatElement};
use hermes_simd::LaneScalar;

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
