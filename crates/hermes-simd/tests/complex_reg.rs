//! Consumer-shaped conformance for [`hermes_simd::ComplexReg`].
//!
//! Written the way a downstream transform kernel would use the type, under
//! `forbid(unsafe_code)`: interleaved `[re, im, ...]` data loaded into
//! registers, complex arithmetic performed in-register, results stored back.
//!
//! ## Oracles
//!
//! Two tiers:
//!
//! - **Exact tier:** inputs chosen so every intermediate is exactly
//!   representable (small integers scaled by powers of two). Products, sums,
//!   and differences of such values round to nothing on any backend, so the
//!   assertion is `assert_eq!` on bits. Layout defects (a swapped lane, a
//!   wrong pair) cannot hide in a tolerance that does not exist.
//! - **Rounded tier:** irrational-ish inputs against a scalar reference
//!   written with `mul_add` in the same even/odd shape. The alternating-FMA
//!   combining step is fused (one rounding) on every backend — hardware
//!   `vfmaddsub` on x86, `scalar_fmadd` in the generic default — so the
//!   2 ULP bound is a conservative ceiling for any residual
//!   evaluation-shape difference, not an allowance for a twice-rounding
//!   fallback.
//!
//! `mul_i`, `mul_neg_i`, `swap_samples`, `splat`, and `butterfly` are pure
//! sign flips and permutations, exact on every backend, so they use only the
//! exact tier.

#![forbid(unsafe_code)]

use hermes_simd::{ComplexReg, LaneKernel, Simd, SimdArch, SimdKernel, SimdStorage};

/// Applies every `ComplexReg` operation to one register-load of interleaved
/// samples and returns the interleaved results in a fixed order.
struct ComplexOps<'a> {
    /// Interleaved input samples; exactly one register.
    a: &'a [f64],
    /// Interleaved second operand; exactly one register.
    w: &'a [f64],
}

/// One output block per operation, interleaved, in declaration order:
/// `mul`, `mul_conj`, `mul_i`, `mul_neg_i`, `swap_samples`, `add`, `sub`.
struct ComplexResults {
    lanes: usize,
    out: Vec<f64>,
}

impl LaneKernel<f64> for ComplexOps<'_> {
    type Output = ComplexResults;

    fn call<A: SimdArch + SimdKernel<f64>>(self, _simd: Simd<f64, A>) -> ComplexResults {
        let lanes = <A as SimdStorage<f64>>::LANE_COUNT;
        let a = ComplexReg::<f64, A>::from_interleaved(
            hermes_simd::Vector::load_unaligned_from_slice(&self.a[..lanes])
                .expect("input holds one register"),
        );
        let w = ComplexReg::<f64, A>::from_interleaved(
            hermes_simd::Vector::load_unaligned_from_slice(&self.w[..lanes])
                .expect("input holds one register"),
        );

        let results = [
            a * w,
            a.mul_conj(w),
            a.mul_i(),
            a.mul_neg_i(),
            a.swap_samples(),
            a + w,
            a - w,
        ];
        let mut out = vec![0.0f64; lanes * results.len()];
        for (block, r) in results.into_iter().enumerate() {
            r.into_interleaved()
                .store_unaligned_to_slice(&mut out[block * lanes..(block + 1) * lanes])
                .expect("output block holds one register");
        }
        ComplexResults { lanes, out }
    }
}

/// Scalar complex reference in the same even/odd `mul_add` shape the fused
/// backends compute, so the rounded tier compares against the tightest
/// legitimate answer.
fn reference(a: &[f64], w: &[f64], lanes: usize) -> Vec<Vec<f64>> {
    let mut blocks = vec![vec![0.0f64; lanes]; 7];
    for s in 0..lanes / 2 {
        let (ar, ai) = (a[2 * s], a[2 * s + 1]);
        let (wr, wi) = (w[2 * s], w[2 * s + 1]);
        // mul: (ar*wr - ai*wi, ar*wi + ai*wr), combining step fused.
        blocks[0][2 * s] = ar.mul_add(wr, -(ai * wi));
        blocks[0][2 * s + 1] = ar.mul_add(wi, ai * wr);
        // mul_conj: (ai*wi + ar*wr, ai*wr - ar*wi).
        blocks[1][2 * s] = ai.mul_add(wi, ar * wr);
        blocks[1][2 * s + 1] = ai.mul_add(wr, -(ar * wi));
        // mul_i / mul_neg_i: pure sign-and-swap.
        blocks[2][2 * s] = -ai;
        blocks[2][2 * s + 1] = ar;
        blocks[3][2 * s] = ai;
        blocks[3][2 * s + 1] = -ar;
        // add / sub.
        blocks[5][2 * s] = ar + wr;
        blocks[5][2 * s + 1] = ai + wi;
        blocks[6][2 * s] = ar - wr;
        blocks[6][2 * s + 1] = ai - wi;
    }
    // swap_samples: neighbouring samples exchange; a lone trailing pair stays.
    let samples = lanes / 2;
    for s in 0..samples {
        let src = if samples >= 2 {
            let partner = s ^ 1;
            if partner < samples {
                partner
            } else {
                s
            }
        } else {
            s
        };
        blocks[4][2 * s] = a[2 * src];
        blocks[4][2 * s + 1] = a[2 * src + 1];
    }
    blocks
}

fn ulps_apart(x: f64, y: f64) -> u64 {
    x.to_bits().abs_diff(y.to_bits())
}

const OP_NAMES: [&str; 7] = [
    "mul",
    "mul_conj",
    "mul_i",
    "mul_neg_i",
    "swap_samples",
    "add",
    "sub",
];

/// Which blocks are permutations and sign flips — exact on every backend.
const EXACT_ALWAYS: [bool; 7] = [false, false, true, true, true, false, false];

#[test]
fn exact_inputs_agree_bitwise_on_every_operation() {
    // Small integers over powers of two: every product and sum below is
    // exactly representable, so fused and unfused backends must agree on bits.
    let a: Vec<f64> = (0..64).map(|i| f64::from(i - 13) * 0.25).collect();
    let w: Vec<f64> = (0..64)
        .map(|i| f64::from((i * 7) % 23 - 11) * 0.5)
        .collect();

    let ComplexResults { lanes, out } = hermes_simd::vectorize(ComplexOps { a: &a, w: &w });
    let expected = reference(&a, &w, lanes);

    for (block, exp) in expected.iter().enumerate() {
        let got = &out[block * lanes..(block + 1) * lanes];
        for lane in 0..lanes {
            assert_eq!(
                got[lane].to_bits(),
                exp[lane].to_bits(),
                "{} lane {lane}: {} != {} on exact inputs",
                OP_NAMES[block],
                got[lane],
                exp[lane],
            );
        }
    }
}

#[test]
fn rounded_inputs_stay_within_two_ulps_of_the_fused_reference() {
    let a: Vec<f64> = (0..64)
        .map(|i| (0.017 * f64::from(i) + 0.3).sin())
        .collect();
    let w: Vec<f64> = (0..64)
        .map(|i| (0.031 * f64::from(i) + 1.1).cos())
        .collect();

    let ComplexResults { lanes, out } = hermes_simd::vectorize(ComplexOps { a: &a, w: &w });
    let expected = reference(&a, &w, lanes);

    for (block, exp) in expected.iter().enumerate() {
        let got = &out[block * lanes..(block + 1) * lanes];
        for lane in 0..lanes {
            if EXACT_ALWAYS[block] {
                assert_eq!(
                    got[lane].to_bits(),
                    exp[lane].to_bits(),
                    "{} lane {lane} must be exact: a permutation cannot round",
                    OP_NAMES[block],
                );
            } else {
                let d = ulps_apart(got[lane], exp[lane]);
                assert!(
                    d <= 2,
                    "{} lane {lane}: {} is {d} ulps from {}",
                    OP_NAMES[block],
                    got[lane],
                    exp[lane],
                );
            }
        }
    }
}

/// `splat` and `butterfly` exercised separately: splat produces one known
/// pattern, and the butterfly of exact inputs is exact.
struct SplatButterfly;

impl LaneKernel<f64> for SplatButterfly {
    type Output = (usize, Vec<f64>, Vec<f64>, Vec<f64>);

    fn call<A: SimdArch + SimdKernel<f64>>(self, _simd: Simd<f64, A>) -> Self::Output {
        let lanes = <A as SimdStorage<f64>>::LANE_COUNT;
        let tw = ComplexReg::<f64, A>::splat(eunomia::Complex64::new(3.0, -0.5));

        let a: Vec<f64> = (0..lanes).map(|i| f64::from(i as u32) + 1.0).collect();
        let b: Vec<f64> = (0..lanes)
            .map(|i| f64::from(i as u32) * 2.0 - 3.0)
            .collect();
        let av = ComplexReg::<f64, A>::from_interleaved(
            hermes_simd::Vector::load_unaligned_from_slice(&a).expect("one register"),
        );
        let bv = ComplexReg::<f64, A>::from_interleaved(
            hermes_simd::Vector::load_unaligned_from_slice(&b).expect("one register"),
        );
        let (sum, diff) = av.butterfly(bv);

        let mut splat_out = vec![0.0f64; lanes];
        let mut sum_out = vec![0.0f64; lanes];
        let mut diff_out = vec![0.0f64; lanes];
        tw.into_interleaved()
            .store_unaligned_to_slice(&mut splat_out)
            .expect("one register");
        sum.into_interleaved()
            .store_unaligned_to_slice(&mut sum_out)
            .expect("one register");
        diff.into_interleaved()
            .store_unaligned_to_slice(&mut diff_out)
            .expect("one register");
        (lanes, splat_out, sum_out, diff_out)
    }
}

#[test]
fn splat_repeats_the_sample_and_butterfly_is_sum_and_difference() {
    let (lanes, splat_out, sum_out, diff_out) = hermes_simd::vectorize(SplatButterfly);

    for s in 0..lanes / 2 {
        assert_eq!(
            splat_out[2 * s].to_bits(),
            3.0f64.to_bits(),
            "splat real, sample {s}"
        );
        assert_eq!(
            splat_out[2 * s + 1].to_bits(),
            (-0.5f64).to_bits(),
            "splat imaginary, sample {s}"
        );
    }
    for i in 0..lanes {
        let (a, b) = (i as f64 + 1.0, i as f64 * 2.0 - 3.0);
        assert_eq!(
            sum_out[i].to_bits(),
            (a + b).to_bits(),
            "butterfly sum, lane {i}"
        );
        assert_eq!(
            diff_out[i].to_bits(),
            (a - b).to_bits(),
            "butterfly difference, lane {i}"
        );
    }
}
