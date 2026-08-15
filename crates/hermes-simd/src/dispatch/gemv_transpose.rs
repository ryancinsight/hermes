//! Generic runtime-dispatch transposed matrix–vector product (`y += Aᵀ · x`).
//!
//! Plumbs the core register-blocked transposed-GEMV micro-kernel
//! ([`hermes_simd_core::tiling::TilingStrategy::gemv_transpose`]) through runtime
//! backend selection, the complement of [`super::gemv()`]. `A` is row-major
//! `nrows × ncols`, `x` length `nrows`, `y` length `ncols`; the result
//! **accumulates** into `y` (`y += Aᵀ·x`), so callers wanting `y = Aᵀ·x` zero
//! `y` first.
//!
//! # Theorem (output reuse, reduction-free)
//! `Aᵀx = Σᵢ xᵢ·A[i,:]` — a sum of the rows of `A` scaled by `x`. Each row is
//! contiguous, so the update vectorizes across the `ncols` output lanes with **no
//! horizontal reduction** (unlike `A·x`). Blocking `TILE_N` output lane-chunks of
//! `y` in registers reuses each accumulator across all `nrows` rows and breaks the
//! per-chunk FMA dependency chain; `TILE_N` scales with the register file. ∎

use hermes_simd_core::{
    align::Unaligned,
    arch::SimdArch,
    execution::Unmasked,
    kernel::{SimdArith, SimdLoadStore, SimdMask},
    scalar::Scalar,
    view::{SimdError, SimdView},
};
use hermes_simd_macros::runtime_dispatch;

#[runtime_dispatch(avx512f, avx2, neon, scalar)]
pub(super) fn dispatch_gemv_transpose_kernel<T, A>(
    a: &[T],
    x: &[T],
    y: &mut [T],
    nrows: usize,
    ncols: usize,
) -> Result<(), SimdError>
where
    T: Scalar,
    A: SimdArch + SimdLoadStore<T> + SimdArith<T> + SimdMask<T>,
{
    match (
        SimdView::<T, A, Unaligned, Unmasked, &[T]>::new(a),
        SimdView::<T, A, Unaligned, Unmasked, &[T]>::new(x),
    ) {
        (Some(va), Some(vx)) => {
            use hermes_simd_core::tiling::{TilingPolicy, TilingStrategy};
            // `gemv_transpose` blocks `TILE_N` output lane-chunks (TILE_M inert);
            // wider register files block more chunks before spilling.
            if A::LANE_COUNT > 8 {
                <TilingPolicy<1, 8> as TilingStrategy<T, A, Unaligned>>::gemv_transpose(
                    &va, &vx, y, nrows, ncols,
                )
            } else if A::LANE_COUNT > 1 {
                <TilingPolicy<1, 4> as TilingStrategy<T, A, Unaligned>>::gemv_transpose(
                    &va, &vx, y, nrows, ncols,
                )
            } else {
                <TilingPolicy<1, 1> as TilingStrategy<T, A, Unaligned>>::gemv_transpose(
                    &va, &vx, y, nrows, ncols,
                )
            }
        }
        // SAFETY: `Unaligned` skips the alignment check, so `SimdView::new` is
        // `Some` for every slice — this arm is unreachable (mirrors `super::gemv`).
        _ => unsafe { core::hint::unreachable_unchecked() },
    }
}

#[cfg(test)]
mod tests {
    use crate::dispatch::gemv_transpose;

    /// Naive reference `y = Aᵀ·x` (`A` row-major `nrows × ncols`).
    fn reference(a: &[f64], x: &[f64], nrows: usize, ncols: usize) -> Vec<f64> {
        let mut y = vec![0.0f64; ncols];
        for (i, &xi) in x.iter().enumerate().take(nrows) {
            for (j, yj) in y.iter_mut().enumerate() {
                *yj += a[i * ncols + j] * xi;
            }
        }
        y
    }

    fn run_case(nrows: usize, ncols: usize) {
        // Dyadic-exact entries: the column-accumulation order matches the
        // reference, so no rounding divergence (assert exact equality).
        let a: Vec<f64> = (0..nrows * ncols)
            .map(|i| ((i % 7) as f64 - 3.0) * 0.25)
            .collect();
        let x: Vec<f64> = (0..nrows).map(|i| ((i % 5) as f64 - 2.0) * 0.5).collect();
        let mut y = vec![0.0f64; ncols];
        gemv_transpose::dispatch_gemv_transpose::<f64>(&a, &x, &mut y, nrows, ncols).unwrap();
        let want = reference(&a, &x, nrows, ncols);
        assert_eq!(
            y, want,
            "gemv_transpose {nrows}x{ncols} mismatch vs reference"
        );
    }

    #[test]
    fn gemv_transpose_matches_reference_across_shapes() {
        // Shapes exercising the TILE_N output-chunk remainder and the column tail.
        for &(m, n) in &[
            (1, 1),
            (1, 33),
            (4, 3),
            (8, 8),
            (13, 9),
            (1, 64),
            (31, 17),
            (64, 33),
            (64, 64),
        ] {
            run_case(m, n);
        }
    }

    #[test]
    fn gemv_transpose_non_dyadic_tail_matches_within_f32_tolerance() {
        let (nrows, ncols) = (5usize, 13usize);
        let a: Vec<f32> = (0..nrows * ncols)
            .map(|i| ((i * 17 + 3) as f32) / 19.0 - 2.0)
            .collect();
        let x: Vec<f32> = (0..nrows).map(|i| (i as f32 + 0.25) / 7.0).collect();
        let mut y = vec![0.125f32; ncols];
        let initial = y.clone();
        gemv_transpose::dispatch_gemv_transpose::<f32>(&a, &x, &mut y, nrows, ncols).unwrap();

        for j in 0..ncols {
            let expected = initial[j] + (0..nrows).map(|i| a[i * ncols + j] * x[i]).sum::<f32>();
            assert!(
                (y[j] - expected).abs() <= 4.0e-6 * expected.abs().max(1.0),
                "column {j}: got {} expected {expected}",
                y[j]
            );
        }
    }

    #[test]
    fn gemv_transpose_accumulates_into_y() {
        let a = vec![1.0f64, 2.0, 3.0, 4.0]; // 2x2: rows [1,2], [3,4]
        let x = vec![1.0f64, 1.0];
        let mut y = vec![10.0f64, 20.0];
        gemv_transpose::dispatch_gemv_transpose::<f64>(&a, &x, &mut y, 2, 2).unwrap();
        // y += Aᵀ·x = [10 + (1+3), 20 + (2+4)] = [14, 26]
        assert_eq!(y, vec![14.0, 26.0]);
    }

    #[test]
    fn gemv_transpose_rejects_short_operands() {
        let a = vec![1.0f64; 4];
        let x = vec![1.0f64; 2];
        let mut y = vec![0.0f64; 1]; // too short for ncols=2
        assert!(gemv_transpose::dispatch_gemv_transpose::<f64>(&a, &x, &mut y, 2, 2).is_err());
    }
}
