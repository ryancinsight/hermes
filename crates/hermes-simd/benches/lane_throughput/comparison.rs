//! Numerical comparison shared by the benchmark families.

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
