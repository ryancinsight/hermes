#![expect(
    clippy::float_cmp,
    reason = "These integration tests assert exact manufactured complex reference values"
)]
#![expect(
    clippy::needless_pass_by_value,
    reason = "The shared error helper consumes heterogeneous Result values to inspect their error variant"
)]

use hermes_simd::{
    interleaved_complex_dot, interleaved_complex_dot_runtime, interleaved_complex_mul_assign,
    interleaved_complex_mul_assign_runtime, PreferredArch, Scalar, SimdError,
};

fn reference_mul<const CONJ_B: bool>(a: &mut [f64], b: &[f64]) {
    for (x, y) in a.chunks_exact_mut(2).zip(b.chunks_exact(2)) {
        let ar = x[0];
        let ai = x[1];
        let br = y[0];
        let bi = if CONJ_B { -y[1] } else { y[1] };
        x[0] = ar * br - ai * bi;
        x[1] = ar * bi + ai * br;
    }
}

fn reference_dot<const CONJ_B: bool>(a: &[f64], b: &[f64]) -> (f64, f64) {
    let mut re = 0.0;
    let mut im = 0.0;
    for (x, y) in a.chunks_exact(2).zip(b.chunks_exact(2)) {
        let ar = x[0];
        let ai = x[1];
        let br = y[0];
        let bi = if CONJ_B { -y[1] } else { y[1] };
        re += ar * br - ai * bi;
        im += ar * bi + ai * br;
    }
    (re, im)
}

fn assert_simd_error<T>(result: Result<T, SimdError>, expected: SimdError) {
    match result {
        Err(actual) => assert_eq!(actual, expected),
        Ok(_) => panic!("expected {expected:?}"),
    }
}

#[test]
fn interleaved_complex_mul_assign_f64_matches_scalar_reference() {
    for &complex_len in &[0usize, 1, 2, 3, 4, 7, 8, 17, 64] {
        let mut data: Vec<f64> = (0..complex_len * 2)
            .map(|i| (i as f64 * 0.25) - 3.0)
            .collect();
        let rhs: Vec<f64> = (0..complex_len * 2)
            .map(|i| (i as f64 % 5.0) - 2.0)
            .collect();
        let mut expected = data.clone();
        reference_mul::<false>(&mut expected, &rhs);

        interleaved_complex_mul_assign::<f64, PreferredArch, false>(&mut data, &rhs).unwrap();

        assert_eq!(data, expected, "complex_len={complex_len}");
    }
}

#[test]
fn interleaved_complex_mul_assign_conjugated_f32_matches_scalar_reference() {
    for &complex_len in &[0usize, 1, 2, 3, 4, 9, 16, 65] {
        let mut data: Vec<f32> = (0..complex_len * 2)
            .map(|i| (i as f32 * 0.125) - 2.0)
            .collect();
        let rhs: Vec<f32> = (0..complex_len * 2)
            .map(|i| (i as f32 % 7.0) - 3.0)
            .collect();
        let mut expected = data.clone();
        for (x, y) in expected.chunks_exact_mut(2).zip(rhs.chunks_exact(2)) {
            let ar = x[0];
            let ai = x[1];
            let br = y[0];
            let bi = -y[1];
            x[0] = ar * br - ai * bi;
            x[1] = ar * bi + ai * br;
        }

        interleaved_complex_mul_assign::<f32, PreferredArch, true>(&mut data, &rhs).unwrap();

        assert_eq!(data, expected, "complex_len={complex_len}");
    }
}

#[test]
fn interleaved_complex_mul_assign_scalar_architecture_matches_preferred() {
    let rhs = [2.0f64, -1.0, 0.5, 3.0, -2.0, 4.0];
    let mut preferred = [1.0f64, 2.0, -3.0, 4.0, 5.0, -6.0];
    let mut scalar = preferred;

    interleaved_complex_mul_assign::<f64, PreferredArch, true>(&mut preferred, &rhs).unwrap();
    interleaved_complex_mul_assign::<f64, Scalar, true>(&mut scalar, &rhs).unwrap();

    assert_eq!(preferred, scalar);
}

#[test]
fn interleaved_complex_mul_assign_runtime_matches_provider_architecture() {
    for &complex_len in &[1usize, 2, 5, 16, 65] {
        let rhs: Vec<f64> = (0..complex_len * 2)
            .map(|i| (i as f64 % 11.0) - 5.0)
            .collect();
        let mut runtime: Vec<f64> = (0..complex_len * 2)
            .map(|i| (i as f64 * 0.375) - 4.0)
            .collect();
        let mut expected = runtime.clone();

        interleaved_complex_mul_assign_runtime::<f64, false>(&mut runtime, &rhs).unwrap();
        interleaved_complex_mul_assign::<f64, PreferredArch, false>(&mut expected, &rhs).unwrap();

        assert_eq!(runtime, expected, "complex_len={complex_len}");
    }
}

#[test]
fn interleaved_complex_dot_f64_matches_scalar_reference() {
    for &complex_len in &[0usize, 1, 2, 3, 4, 7, 8, 17, 64] {
        let lhs: Vec<f64> = (0..complex_len * 2)
            .map(|i| (i as f64 * 0.25) - 3.0)
            .collect();
        let rhs: Vec<f64> = (0..complex_len * 2)
            .map(|i| (i as f64 % 5.0) - 2.0)
            .collect();
        let expected = reference_dot::<false>(&lhs, &rhs);

        let actual = interleaved_complex_dot::<f64, PreferredArch, false>(&lhs, &rhs).unwrap();

        assert_eq!(actual.0, expected.0, "complex_len={complex_len}");
        assert_eq!(actual.1, expected.1, "complex_len={complex_len}");
    }
}

#[test]
fn interleaved_complex_dot_conjugated_runtime_matches_provider_architecture() {
    for &complex_len in &[1usize, 2, 5, 16, 65] {
        let lhs: Vec<f64> = (0..complex_len * 2)
            .map(|i| (i as f64 * 0.375) - 4.0)
            .collect();
        let rhs: Vec<f64> = (0..complex_len * 2)
            .map(|i| (i as f64 % 11.0) - 5.0)
            .collect();

        let runtime = interleaved_complex_dot_runtime::<f64, true>(&lhs, &rhs).unwrap();
        let expected = interleaved_complex_dot::<f64, PreferredArch, true>(&lhs, &rhs).unwrap();

        assert_eq!(runtime, expected, "complex_len={complex_len}");
    }
}

/// Differential verification of the vectorized kernels against the `Scalar`
/// backend for an explicit architecture marker.
///
/// Inputs are dyadic rationals with few mantissa bits, so every product and
/// partial sum is exactly representable in both `f32` and `f64`; fused and
/// unfused multiply-add paths therefore produce bitwise-identical results and
/// `assert_eq!` is exact, not tolerance-based.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
macro_rules! assert_arch_matches_scalar {
    ($t:ty, $arch:ty, $conj:expr) => {
        for &complex_len in &[1usize, 2, 3, 4, 7, 8, 9, 16, 17, 33, 64, 65] {
            let lhs: Vec<$t> = (0..complex_len * 2)
                .map(|i| ((i % 9) as $t) * 0.25 - 1.0)
                .collect();
            let rhs: Vec<$t> = (0..complex_len * 2)
                .map(|i| ((i % 7) as $t) * 0.5 - 1.5)
                .collect();

            let mut vectorized = lhs.clone();
            let mut scalar = lhs.clone();
            interleaved_complex_mul_assign::<$t, $arch, $conj>(&mut vectorized, &rhs).unwrap();
            interleaved_complex_mul_assign::<$t, Scalar, $conj>(&mut scalar, &rhs).unwrap();
            assert_eq!(vectorized, scalar, "mul complex_len={complex_len}");

            let vec_dot = interleaved_complex_dot::<$t, $arch, $conj>(&lhs, &rhs).unwrap();
            let scalar_dot = interleaved_complex_dot::<$t, Scalar, $conj>(&lhs, &rhs).unwrap();
            assert_eq!(vec_dot, scalar_dot, "dot complex_len={complex_len}");
        }
    };
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[test]
fn interleaved_complex_avx2_matches_scalar_backend() {
    if !(std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma")) {
        return;
    }
    assert_arch_matches_scalar!(f32, hermes_simd::Avx2, false);
    assert_arch_matches_scalar!(f32, hermes_simd::Avx2, true);
    assert_arch_matches_scalar!(f64, hermes_simd::Avx2, false);
    assert_arch_matches_scalar!(f64, hermes_simd::Avx2, true);
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[test]
fn interleaved_complex_avx512_matches_scalar_backend() {
    if !std::is_x86_feature_detected!("avx512f") {
        return;
    }
    assert_arch_matches_scalar!(f32, hermes_simd::Avx512, false);
    assert_arch_matches_scalar!(f32, hermes_simd::Avx512, true);
    assert_arch_matches_scalar!(f64, hermes_simd::Avx512, false);
    assert_arch_matches_scalar!(f64, hermes_simd::Avx512, true);
}

#[test]
fn interleaved_complex_runtime_conjugated_reduced_precision_lane_matches_scalar() {
    for &complex_len in &[1usize, 3, 8, 33] {
        let rhs: Vec<f32> = (0..complex_len * 2)
            .map(|i| ((i % 7) as f32) * 0.5 - 1.5)
            .collect();
        let mut runtime: Vec<f32> = (0..complex_len * 2)
            .map(|i| ((i % 9) as f32) * 0.25 - 1.0)
            .collect();
        let mut expected = runtime.clone();

        interleaved_complex_mul_assign_runtime::<f32, true>(&mut runtime, &rhs).unwrap();
        interleaved_complex_mul_assign::<f32, Scalar, true>(&mut expected, &rhs).unwrap();
        assert_eq!(runtime, expected, "complex_len={complex_len}");

        let runtime_dot = interleaved_complex_dot_runtime::<f32, true>(&runtime, &rhs).unwrap();
        let expected_dot = interleaved_complex_dot::<f32, Scalar, true>(&runtime, &rhs).unwrap();
        assert_eq!(runtime_dot, expected_dot, "dot complex_len={complex_len}");
    }
}

/// Reduced-precision lanes (`f16`, `bf16`).
///
/// Elementwise multiply: every compiled backend for these types is
/// lane-emulated with the identical per-lane operation sequence (product,
/// product, subtract/add in native `T` semantics), so the runtime dispatch
/// path and the `Scalar` backend must agree bitwise — `assert_eq!` is exact.
///
/// Dot: backends accumulate with different lane counts and unroll groupings,
/// which reorders the low-precision sum; the comparison uses the analytical
/// reordering bound `(n + 8)·ε_T·Σ (|ar|+|ai|)·(|br|+|bi|)` per component,
/// with `ε_f16 = 2⁻¹⁰` and `ε_bf16 = 2⁻⁷`.
mod reduced_precision_lanes {
    use super::*;

    macro_rules! check_lane_type {
        ($t:ty, $from:expr, $eps:expr) => {
            for &complex_len in &[1usize, 2, 3, 4, 8, 9, 17, 33] {
                let a: Vec<$t> = (0..complex_len * 2)
                    .map(|i| $from((i as f32 * 0.37) - 3.1))
                    .collect();
                let b: Vec<$t> = (0..complex_len * 2)
                    .map(|i| $from(((i % 13) as f32 * 0.71) - 4.2))
                    .collect();

                let mut runtime = a.clone();
                let mut scalar = a.clone();
                interleaved_complex_mul_assign_runtime::<$t, false>(&mut runtime, &b).unwrap();
                interleaved_complex_mul_assign::<$t, Scalar, false>(&mut scalar, &b).unwrap();
                assert_eq!(runtime, scalar, "mul complex_len={complex_len}");

                let mut runtime_conj = a.clone();
                let mut scalar_conj = a.clone();
                interleaved_complex_mul_assign_runtime::<$t, true>(&mut runtime_conj, &b).unwrap();
                interleaved_complex_mul_assign::<$t, Scalar, true>(&mut scalar_conj, &b).unwrap();
                assert_eq!(runtime_conj, scalar_conj, "conj mul complex_len={complex_len}");

                let dot_runtime = interleaved_complex_dot_runtime::<$t, true>(&a, &b).unwrap();
                let dot_scalar = interleaved_complex_dot::<$t, Scalar, true>(&a, &b).unwrap();
                let mag: f32 = (0..a.len())
                    .step_by(2)
                    .map(|k| {
                        (a[k].to_f32().abs() + a[k + 1].to_f32().abs())
                            * (b[k].to_f32().abs() + b[k + 1].to_f32().abs())
                    })
                    .sum();
                let tol = (complex_len as f32 + 8.0) * $eps * mag.max(1.0);
                assert!(
                    (dot_runtime.0.to_f32() - dot_scalar.0.to_f32()).abs() <= tol
                        && (dot_runtime.1.to_f32() - dot_scalar.1.to_f32()).abs() <= tol,
                    "dot complex_len={complex_len}: runtime={dot_runtime:?}, scalar={dot_scalar:?}, tol={tol}",
                );
            }
        };
    }

    #[test]
    fn interleaved_complex_half_lane_runtime_matches_scalar_backend() {
        check_lane_type!(eunomia::F16, eunomia::F16::from_f32, 2.0f32.powi(-10));
    }

    #[test]
    fn interleaved_complex_brain_lane_runtime_matches_scalar_backend() {
        check_lane_type!(eunomia::Bf16, eunomia::Bf16::from_f32, 2.0f32.powi(-7));
    }
}

mod complex_properties {
    use super::*;
    use proptest::prelude::*;

    /// Per-component error bound for one complex multiply.
    ///
    /// Each output component is a two-term sum of products; the SIMD path
    /// (fused multiply-add, alternating-sign form) and the scalar reference
    /// differ by at most 2 roundings per product/sum step. Bound:
    /// `4·ε·(|ar|+|ai|)·(|br|+|bi|)` per component, with a small absolute
    /// floor of `4·ε` for results near zero.
    fn mul_tolerance_f64(ar: f64, ai: f64, br: f64, bi: f64) -> f64 {
        4.0 * f64::EPSILON * ((ar.abs() + ai.abs()) * (br.abs() + bi.abs())).max(1.0)
    }

    fn paired_vecs(max: f64, max_pairs: usize) -> impl Strategy<Value = (Vec<f64>, Vec<f64>)> {
        prop::collection::vec(-max..max, 0..max_pairs)
            .prop_map(|v| {
                let even = (v.len() / 2) * 2;
                v[..even].to_vec()
            })
            .prop_flat_map(|a| {
                let len = a.len();
                (Just(a), prop::collection::vec(-1000.0f64..1000.0, len))
            })
    }

    proptest! {
        #[test]
        fn prop_complex_mul_assign_within_rounding_of_reference(
            (a, b) in paired_vecs(1000.0, 128),
            conj in any::<bool>(),
        ) {
            let mut actual = a.clone();
            let mut expected = a.clone();
            if conj {
                interleaved_complex_mul_assign_runtime::<f64, true>(&mut actual, &b).unwrap();
                reference_mul::<true>(&mut expected, &b);
            } else {
                interleaved_complex_mul_assign_runtime::<f64, false>(&mut actual, &b).unwrap();
                reference_mul::<false>(&mut expected, &b);
            }
            for k in (0..a.len()).step_by(2) {
                let tol = mul_tolerance_f64(a[k], a[k + 1], b[k], b[k + 1]);
                prop_assert!(
                    (actual[k] - expected[k]).abs() <= tol
                        && (actual[k + 1] - expected[k + 1]).abs() <= tol,
                    "pair {k}: actual=({}, {}), expected=({}, {}), tol={tol}",
                    actual[k], actual[k + 1], expected[k], expected[k + 1],
                );
            }
        }

        /// Dot-product error bound: an `n`-pair dot is a sum of `n` complex
        /// products. Reordering an `n`-term sum and fusing the products gives
        /// `|err| <= (n + 4)·ε·Σ (|ar|+|ai|)·(|br|+|bi|)` per component.
        #[test]
        fn prop_complex_dot_within_rounding_of_reference(
            (a, b) in paired_vecs(1000.0, 128),
            conj in any::<bool>(),
        ) {
            let (actual, expected) = if conj {
                (
                    interleaved_complex_dot_runtime::<f64, true>(&a, &b).unwrap(),
                    reference_dot::<true>(&a, &b),
                )
            } else {
                (
                    interleaved_complex_dot_runtime::<f64, false>(&a, &b).unwrap(),
                    reference_dot::<false>(&a, &b),
                )
            };
            let n = (a.len() / 2) as f64;
            let mag: f64 = (0..a.len())
                .step_by(2)
                .map(|k| (a[k].abs() + a[k + 1].abs()) * (b[k].abs() + b[k + 1].abs()))
                .sum();
            let tol = (n + 4.0) * f64::EPSILON * mag.max(1.0);
            prop_assert!(
                (actual.0 - expected.0).abs() <= tol && (actual.1 - expected.1).abs() <= tol,
                "actual=({}, {}), expected=({}, {}), tol={tol}",
                actual.0, actual.1, expected.0, expected.1,
            );
        }
    }
}

#[test]
fn interleaved_complex_mul_assign_rejects_invalid_shapes() {
    let mut odd = [1.0f32, 2.0, 3.0];
    assert_simd_error(
        interleaved_complex_mul_assign::<f32, PreferredArch, false>(&mut odd, &[1.0, 2.0, 3.0]),
        SimdError::LengthMismatch,
    );

    let mut lhs = [1.0f32, 2.0];
    assert_simd_error(
        interleaved_complex_mul_assign::<f32, PreferredArch, false>(&mut lhs, &[1.0]),
        SimdError::LengthMismatch,
    );
    assert_simd_error(
        interleaved_complex_dot::<f32, PreferredArch, false>(&lhs, &[1.0]),
        SimdError::LengthMismatch,
    );
}
