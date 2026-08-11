//! Generic runtime-dispatch absolute-value reductions: `Σ|x|` and `max|x|`.
//!
//! The L1 / ∞-norm accumulators consumers (leto norms) reduce with: one
//! lane-wise `abs` fused into the additive or max fold, no temporary buffer.
//! Both return `T::ZERO` for empty slices — the mathematically correct empty
//! norm, since every magnitude is non-negative.

use hermes_simd_core::{
    align::Unaligned,
    arch::SimdArch,
    execution::Unmasked,
    kernel::SimdKernel,
    ops::{AbsMax, AbsSum},
    scalar::Scalar,
    view::SimdView,
};
use hermes_simd_macros::runtime_dispatch;

#[runtime_dispatch(avx512f, avx2, neon, scalar)]
pub(super) fn dispatch_abs_sum_kernel<T, A>(data: &[T]) -> T
where
    T: Scalar,
    A: SimdArch + SimdKernel<T>,
{
    match SimdView::<T, A, Unaligned, Unmasked, &[T]>::new(data) {
        Some(v) => v.reduce(AbsSum),
        None => unsafe { core::hint::unreachable_unchecked() },
    }
}

#[runtime_dispatch(avx512f, avx2, neon, scalar)]
pub(super) fn dispatch_abs_max_kernel<T, A>(data: &[T]) -> T
where
    T: Scalar,
    A: SimdArch + SimdKernel<T>,
{
    match SimdView::<T, A, Unaligned, Unmasked, &[T]>::new(data) {
        Some(v) => v.reduce(AbsMax),
        None => unsafe { core::hint::unreachable_unchecked() },
    }
}

#[cfg(test)]
mod tests {
    use super::super::{abs_max, abs_sum};

    #[test]
    fn abs_sum_matches_scalar_reference_across_tail_sizes() {
        for &len in &[0usize, 1, 3, 7, 8, 9, 15, 16, 17, 63, 64, 65, 1027] {
            let data: Vec<f64> = (0..len)
                .map(|i| (i as f64 - len as f64 / 2.0) * 0.75)
                .collect();
            let expected: f64 = data.iter().map(|x| x.abs()).sum();
            assert_eq!(abs_sum(&data), expected, "len {len}");
        }
    }

    #[test]
    fn abs_max_matches_scalar_reference_across_tail_sizes() {
        for &len in &[0usize, 1, 3, 7, 8, 9, 15, 16, 17, 63, 64, 65, 1027] {
            let data: Vec<f64> = (0..len)
                .map(|i| (i as f64 - len as f64 / 2.0) * -1.25)
                .collect();
            let expected = data.iter().map(|x| x.abs()).fold(0.0f64, f64::max);
            assert_eq!(abs_max(&data), expected, "len {len}");
        }
    }

    #[test]
    fn abs_reductions_single_precision_match_reference() {
        let data: Vec<f32> = (0..133)
            .map(|i| (i as f32) * if i % 2 == 0 { 1.0 } else { -1.0 })
            .collect();
        let expected_sum: f32 = data.iter().map(|x| x.abs()).sum();
        let expected_max = data.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
        assert_eq!(abs_sum(&data), expected_sum);
        assert_eq!(abs_max(&data), expected_max);
    }

    #[test]
    fn abs_reductions_masked_tails_cover_multiple_widths_and_f32() {
        for &len in &[1usize, 2, 3, 5, 9, 17, 65, 133] {
            let data: Vec<f32> = (0..len)
                .map(|index| (index as f32 - 2.25) * if index % 2 == 0 { 0.75 } else { -0.5 })
                .collect();
            let expected_sum: f32 = data.iter().map(|value| value.abs()).sum();
            let expected_max = data.iter().map(|value| value.abs()).fold(0.0f32, f32::max);
            assert!(
                (abs_sum(&data) - expected_sum).abs() <= 2.0e-5,
                "sum len {len}"
            );
            assert_eq!(abs_max(&data), expected_max, "max len {len}");
        }
    }

    #[test]
    fn abs_reductions_masked_tails_cover_multiple_widths_and_f64() {
        for &len in &[1usize, 2, 3, 5, 9, 17, 65, 133] {
            let data: Vec<f64> = (0..len)
                .map(|index| (index as f64 - 3.5) * if index % 2 == 0 { -0.625 } else { 0.375 })
                .collect();
            let expected_sum: f64 = data.iter().map(|value| value.abs()).sum();
            let expected_max = data.iter().map(|value| value.abs()).fold(0.0f64, f64::max);
            assert!(
                (abs_sum(&data) - expected_sum).abs() <= 2.0e-12,
                "sum len {len}"
            );
            assert_eq!(abs_max(&data), expected_max, "max len {len}");
        }
    }

    #[test]
    fn abs_reductions_empty_are_zero() {
        let empty: [f64; 0] = [];
        assert_eq!(abs_sum(&empty), 0.0);
        assert_eq!(abs_max(&empty), 0.0);
    }

    #[test]
    fn abs_max_all_negative_returns_largest_magnitude() {
        let data = [-3.0f64, -7.5, -0.25, -7.25];
        assert_eq!(abs_max(&data), 7.5);
    }
}
