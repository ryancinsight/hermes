//! The NaN contract of `min` and `max`, pinned.
//!
//! Consumers fold these into layout-dependent routes (leto's strided axis
//! reduction against its contiguous one), so the two must agree, and they can
//! only agree if the provider's rule is stated and held. The rule is `fmin` /
//! `fmax`: NaN is ignored wherever it sits, and an all-NaN slice reduces to the
//! identity. Lengths span the scalar tail and full vectors so every dispatched
//! path is exercised.

fn with_nan_at(len: usize, at: usize) -> Vec<f64> {
    (0..len)
        .map(|i| {
            if i == at {
                f64::NAN
            } else {
                ((i * 7) % 13) as f64 - 6.0
            }
        })
        .collect()
}

// The identity is what an empty slice returns -- for floats, +/-infinity --
// so the reference folds from the same point the provider does.
/// Exact comparison: these are reductions over a fixed input and the answer
/// is one of the inputs or the identity, so bit equality is the contract.
fn assert_exact(actual: f64, expected: f64, context: &str) {
    assert!(
        actual.to_bits() == expected.to_bits(),
        "{context}: got {actual}, expected {expected}"
    );
}

fn reference_min(data: &[f64]) -> f64 {
    data.iter()
        .copied()
        .filter(|x| !x.is_nan())
        .fold(hermes_simd::min::<f64>(&[]), f64::min)
}

fn reference_max(data: &[f64]) -> f64 {
    data.iter()
        .copied()
        .filter(|x| !x.is_nan())
        .fold(hermes_simd::max::<f64>(&[]), f64::max)
}

#[test]
fn min_and_max_ignore_nan_wherever_it_sits() {
    for len in [1usize, 2, 3, 7, 8, 9, 15, 16, 17, 31, 32, 33, 64, 65, 129] {
        for at in [0, len / 2, len - 1] {
            let data = with_nan_at(len, at);
            let f32s: Vec<f32> = data.iter().map(|&x| x as f32).collect();
            assert_exact(
                hermes_simd::min::<f64>(&data),
                reference_min(&data),
                &format!("f64 min, len {len}, NaN at {at}"),
            );
            assert_exact(
                hermes_simd::max::<f64>(&data),
                reference_max(&data),
                &format!("f64 max, len {len}, NaN at {at}"),
            );
            assert_exact(
                f64::from(hermes_simd::min::<f32>(&f32s)),
                reference_min(&data),
                &format!("f32 min, len {len}, NaN at {at}"),
            );
            assert_exact(
                f64::from(hermes_simd::max::<f32>(&f32s)),
                reference_max(&data),
                &format!("f32 max, len {len}, NaN at {at}"),
            );
        }
    }
}

#[test]
fn an_all_nan_slice_reduces_to_the_identity() {
    // The identity is the empty-slice value; for floats that is infinity, not
    // the largest finite value, so both are asserted rather than assumed.
    assert_exact(hermes_simd::min::<f64>(&[]), f64::INFINITY, "empty min");
    assert_exact(hermes_simd::max::<f64>(&[]), f64::NEG_INFINITY, "empty max");
    for len in [1usize, 5, 16, 33] {
        let data = vec![f64::NAN; len];
        assert_exact(
            hermes_simd::min::<f64>(&data),
            f64::INFINITY,
            &format!("all-NaN min, len {len}"),
        );
        assert_exact(
            hermes_simd::max::<f64>(&data),
            f64::NEG_INFINITY,
            &format!("all-NaN max, len {len}"),
        );
        let f32s = vec![f32::NAN; len];
        assert_exact(
            f64::from(hermes_simd::min::<f32>(&f32s)),
            f64::INFINITY,
            &format!("all-NaN f32 min, len {len}"),
        );
        assert_exact(
            f64::from(hermes_simd::max::<f32>(&f32s)),
            f64::NEG_INFINITY,
            &format!("all-NaN f32 max, len {len}"),
        );
    }
}
