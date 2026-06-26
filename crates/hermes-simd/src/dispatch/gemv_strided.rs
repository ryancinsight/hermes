//! Generic runtime-dispatch register-blocked sub-matrix GEMV (`y += A · x`,
//! row stride `lda`).
//!
//! Generalizes [`super::gemv()`] to a row-major **sub-matrix**: `nrows × ncols`
//! with leading dimension `lda ≥ ncols` (rows contiguous over `ncols`, spaced
//! `lda` apart). `lda = ncols` recovers the packed `gemv`. This admits matvec
//! over a trailing/leading block of a larger buffer — e.g. the column-major
//! trailing block of a reflector apply, whose columns are contiguous but spaced
//! by the buffer's row count — without copying the block out. Result
//! **accumulates** into `y`.

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
pub(super) fn dispatch_gemv_strided_kernel<T, A>(
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
                <TilingPolicy<8, 1> as TilingStrategy<T, A, Unaligned>>::gemv_strided(
                    &va, &vx, y, nrows, ncols, lda,
                )
            } else if A::LANE_COUNT > 1 {
                <TilingPolicy<4, 1> as TilingStrategy<T, A, Unaligned>>::gemv_strided(
                    &va, &vx, y, nrows, ncols, lda,
                )
            } else {
                <TilingPolicy<1, 1> as TilingStrategy<T, A, Unaligned>>::gemv_strided(
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
    use crate::dispatch::{gemv, gemv_strided};

    /// Naive reference `y = A·x` over a sub-matrix with row stride `lda`.
    fn reference(a: &[f64], x: &[f64], nrows: usize, ncols: usize, lda: usize) -> Vec<f64> {
        (0..nrows)
            .map(|r| (0..ncols).map(|c| a[r * lda + c] * x[c]).sum())
            .collect()
    }

    #[test]
    fn gemv_strided_matches_reference_over_submatrix() {
        // A 6x10 backing buffer; operate on the 4x6 sub-matrix (lda=10, ncols=6).
        let lda = 10usize;
        let (nrows, ncols) = (4usize, 6usize);
        let a: Vec<f64> = (0..nrows * lda)
            .map(|i| ((i % 9) as f64 - 4.0) * 0.25)
            .collect();
        let x: Vec<f64> = (0..ncols).map(|i| ((i % 5) as f64 - 2.0) * 0.5).collect();
        let mut y = vec![0.0f64; nrows];
        gemv_strided::dispatch_gemv_strided::<f64>(&a, &x, &mut y, nrows, ncols, lda).unwrap();
        assert_eq!(y, reference(&a, &x, nrows, ncols, lda));
    }

    #[test]
    fn gemv_strided_packed_equals_gemv() {
        // lda == ncols must agree bit-for-bit with the packed gemv.
        let (nrows, ncols) = (9usize, 13usize);
        let a: Vec<f64> = (0..nrows * ncols)
            .map(|i| ((i % 7) as f64 - 3.0) * 0.5)
            .collect();
        let x: Vec<f64> = (0..ncols).map(|i| ((i % 4) as f64 - 1.0) * 0.25).collect();
        let mut y_strided = vec![0.0f64; nrows];
        let mut y_packed = vec![0.0f64; nrows];
        gemv_strided::dispatch_gemv_strided::<f64>(&a, &x, &mut y_strided, nrows, ncols, ncols)
            .unwrap();
        gemv::dispatch_gemv::<f64>(&a, &x, &mut y_packed, nrows, ncols).unwrap();
        assert_eq!(y_strided, y_packed);
    }

    #[test]
    fn gemv_strided_rejects_lda_below_ncols_and_short_spans() {
        let a = vec![1.0f64; 40];
        let x = vec![1.0f64; 6];
        let mut y = vec![0.0f64; 4];
        // lda < ncols is invalid.
        assert!(gemv_strided::dispatch_gemv_strided::<f64>(&a, &x, &mut y, 4, 6, 5).is_err());
        // a too short for the requested sub-matrix span.
        let short = vec![1.0f64; 10];
        assert!(gemv_strided::dispatch_gemv_strided::<f64>(&short, &x, &mut y, 4, 6, 10).is_err());
    }
}
