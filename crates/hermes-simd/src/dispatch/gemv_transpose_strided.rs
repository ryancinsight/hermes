//! Generic runtime-dispatch transposed sub-matrix GEMV (`y += Aᵀ · x`, row
//! stride `lda`).
//!
//! Generalizes [`super::gemv_transpose()`] to a row-major **sub-matrix**:
//! `nrows × ncols` with leading dimension `lda ≥ ncols`. `lda = ncols` recovers
//! the packed transpose. Computes `Σᵢ xᵢ·A[i,:]` (sum of the strided rows scaled
//! by `x`), vectorizing across the `ncols` output lanes with no horizontal
//! reduction. Admits the `Aᵀ·x` reduction over a trailing/leading block of a
//! larger buffer — e.g. forming `Aw = Σⱼ wⱼ·colⱼ` in a reflector apply — without
//! copying it out. Accumulates into `y`.

use hermes_simd_core::{
    align::Unaligned,
    arch::SimdArch,
    execution::Unmasked,
    kernel::SimdKernel,
    scalar::Scalar,
    view::{SimdError, SimdView},
};
use hermes_simd_macros::runtime_dispatch;

#[runtime_dispatch(avx512f, avx2, neon, scalar)]
pub(super) fn dispatch_gemv_transpose_strided_kernel<T, A>(
    a: &[T],
    x: &[T],
    y: &mut [T],
    nrows: usize,
    ncols: usize,
    lda: usize,
) -> Result<(), SimdError>
where
    T: Scalar,
    A: SimdArch + SimdKernel<T>,
{
    match (
        SimdView::<T, A, Unaligned, Unmasked, &[T]>::new(a),
        SimdView::<T, A, Unaligned, Unmasked, &[T]>::new(x),
    ) {
        (Some(va), Some(vx)) => {
            use hermes_simd_core::tiling::{TilingPolicy, TilingStrategy};
            if A::LANE_COUNT > 8 {
                <TilingPolicy<1, 8> as TilingStrategy<T, A, Unaligned>>::gemv_transpose_strided(
                    &va, &vx, y, nrows, ncols, lda,
                )
            } else if A::LANE_COUNT > 1 {
                <TilingPolicy<1, 4> as TilingStrategy<T, A, Unaligned>>::gemv_transpose_strided(
                    &va, &vx, y, nrows, ncols, lda,
                )
            } else {
                <TilingPolicy<1, 1> as TilingStrategy<T, A, Unaligned>>::gemv_transpose_strided(
                    &va, &vx, y, nrows, ncols, lda,
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
    use crate::dispatch::{gemv_transpose, gemv_transpose_strided};

    /// Naive `y = Aᵀ·x` over a sub-matrix with row stride `lda`.
    fn reference(a: &[f64], x: &[f64], nrows: usize, ncols: usize, lda: usize) -> Vec<f64> {
        let mut y = vec![0.0f64; ncols];
        for (i, &xi) in x.iter().enumerate().take(nrows) {
            for (j, yj) in y.iter_mut().enumerate() {
                *yj += a[i * lda + j] * xi;
            }
        }
        y
    }

    #[test]
    fn gemv_transpose_strided_matches_reference_over_submatrix() {
        let lda = 11usize;
        let (nrows, ncols) = (5usize, 7usize);
        let a: Vec<f64> = (0..nrows * lda)
            .map(|i| ((i % 9) as f64 - 4.0) * 0.25)
            .collect();
        let x: Vec<f64> = (0..nrows).map(|i| ((i % 5) as f64 - 2.0) * 0.5).collect();
        let mut y = vec![0.0f64; ncols];
        gemv_transpose_strided::dispatch_gemv_transpose_strided::<f64>(
            &a, &x, &mut y, nrows, ncols, lda,
        )
        .unwrap();
        assert_eq!(y, reference(&a, &x, nrows, ncols, lda));
    }

    #[test]
    fn gemv_transpose_strided_packed_equals_gemv_transpose() {
        let (nrows, ncols) = (9usize, 13usize);
        let a: Vec<f64> = (0..nrows * ncols)
            .map(|i| ((i % 7) as f64 - 3.0) * 0.5)
            .collect();
        let x: Vec<f64> = (0..nrows).map(|i| ((i % 4) as f64 - 1.0) * 0.25).collect();
        let mut y_s = vec![0.0f64; ncols];
        let mut y_p = vec![0.0f64; ncols];
        gemv_transpose_strided::dispatch_gemv_transpose_strided::<f64>(
            &a, &x, &mut y_s, nrows, ncols, ncols,
        )
        .unwrap();
        gemv_transpose::dispatch_gemv_transpose::<f64>(&a, &x, &mut y_p, nrows, ncols).unwrap();
        assert_eq!(y_s, y_p);
    }

    #[test]
    fn gemv_transpose_strided_rejects_invalid() {
        let a = vec![1.0f64; 40];
        let x = vec![1.0f64; 5];
        let mut y = vec![0.0f64; 7];
        // lda < ncols invalid.
        assert!(
            gemv_transpose_strided::dispatch_gemv_transpose_strided::<f64>(&a, &x, &mut y, 5, 7, 6)
                .is_err()
        );
    }

    #[test]
    fn gemv_transpose_strided_rejects_dimension_overflow() {
        use hermes_simd_core::view::SimdError;
        // `(nrows-1)·lda + ncols` overflows `usize`; `x`/`y` are sized so only the
        // A-span fails. Unchecked the product wraps and admits an OOB SIMD load —
        // the checked span arithmetic rejects with the exact variant.
        let a = vec![1.0f64; 40];
        let x = vec![1.0f64; 2];
        let mut y = vec![0.0f64; 6];
        let r = gemv_transpose_strided::dispatch_gemv_transpose_strided::<f64>(
            &a,
            &x,
            &mut y,
            2,
            6,
            usize::MAX,
        );
        assert_eq!(r, Err(SimdError::LengthMismatch));
    }
}
