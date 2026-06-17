//! Generic runtime-dispatch register-blocked matrix–vector product (`y += A · x`).
//!
//! Plumbs the core register-blocked `gemv` micro-kernel
//! ([`hermes_simd_core::tiling::TilingStrategy::gemv`]) through runtime backend
//! selection, mirroring [`super::gemm`]. `A` is row-major `nrows × ncols`; the
//! result **accumulates** into `y` (`y += A·x`), so callers wanting `y = A·x`
//! zero `y` first — matching the `axpy`/GEMM accumulate convention.
//!
//! # Theorem (operand reuse — why register blocking helps)
//! GEMV performs `2·nrows·ncols` flops over `nrows·ncols` matrix elements:
//! arithmetic intensity ≈ 2 flops/element, so it is **memory-bound** and
//! throughput is governed by operand reuse, not FLOP rate. Blocking `TILE_M`
//! rows of `A` loads each `x[c..c+lane]` vector **once** and applies it to all
//! `TILE_M` rows held in independent register accumulators; this cuts `x`
//! traffic by `TILE_M×` and breaks the per-row FMA dependency chain (one live
//! accumulator per row). The `nrows mod TILE_M` remainder is handled by a
//! single-row cleanup, so any shape is supported. `TILE_M` scales with the
//! register file: wider ISAs block more rows before spilling. ∎

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
pub(super) fn dispatch_gemv_kernel<T, A>(
    a: &[T],
    x: &[T],
    y: &mut [T],
    nrows: usize,
    ncols: usize,
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
            // GEMV uses only `TILE_M` (row blocking); `TILE_N` is inert here.
            // Wider register files block more rows to amortize the shared `x`
            // load — `<1,1>` for scalar avoids any unused-accumulator overhead.
            if A::LANE_COUNT > 8 {
                <TilingPolicy<8, 1> as TilingStrategy<T, A, Unaligned>>::gemv(
                    &va, &vx, y, nrows, ncols,
                )
            } else if A::LANE_COUNT > 1 {
                <TilingPolicy<4, 1> as TilingStrategy<T, A, Unaligned>>::gemv(
                    &va, &vx, y, nrows, ncols,
                )
            } else {
                <TilingPolicy<1, 1> as TilingStrategy<T, A, Unaligned>>::gemv(
                    &va, &vx, y, nrows, ncols,
                )
            }
        }
        // SAFETY: `Unaligned::IS_ALIGNED` is false, so `SimdView::new` skips the
        // alignment check and is `Some` for every slice (including empty) — this
        // arm is unreachable. Mirrors `super::gemm::dispatch_tiled_gemm_kernel`.
        _ => unsafe { core::hint::unreachable_unchecked() },
    }
}

#[cfg(test)]
mod tests {
    use crate::dispatch::gemv;

    /// Naive reference `y = A·x` (`A` row-major `nrows × ncols`).
    fn reference(a: &[f64], x: &[f64], nrows: usize, ncols: usize) -> Vec<f64> {
        (0..nrows)
            .map(|r| (0..ncols).map(|c| a[r * ncols + c] * x[c]).sum())
            .collect()
    }

    fn run_case(nrows: usize, ncols: usize) {
        // Deterministic, dyadic-exact entries so the SIMD reduction order does
        // not change the result vs the reference (no rounding divergence).
        let a: Vec<f64> = (0..nrows * ncols)
            .map(|i| ((i % 7) as f64 - 3.0) * 0.25)
            .collect();
        let x: Vec<f64> = (0..ncols).map(|i| ((i % 5) as f64 - 2.0) * 0.5).collect();
        let mut y = vec![0.0f64; nrows];
        gemv::dispatch_gemv::<f64>(&a, &x, &mut y, nrows, ncols).unwrap();
        let want = reference(&a, &x, nrows, ncols);
        assert_eq!(y, want, "gemv {nrows}x{ncols} mismatch vs reference");
    }

    #[test]
    fn gemv_matches_reference_across_shapes() {
        // Includes shapes exercising the TILE_M row remainder and the column
        // SIMD tail (ncols not a multiple of any lane count).
        for &(m, n) in &[
            (1, 1),
            (1, 17),
            (3, 4),
            (8, 8),
            (9, 13),
            (16, 1),
            (17, 31),
            (33, 64),
            (64, 64),
        ] {
            run_case(m, n);
        }
    }

    #[test]
    fn gemv_accumulates_into_y() {
        let a = vec![1.0f64, 2.0, 3.0, 4.0]; // 2x2
        let x = vec![1.0f64, 1.0];
        let mut y = vec![10.0f64, 20.0];
        gemv::dispatch_gemv::<f64>(&a, &x, &mut y, 2, 2).unwrap();
        // y += A·x = [10+3, 20+7] = [13, 27]
        assert_eq!(y, vec![13.0, 27.0]);
    }

    #[test]
    fn gemv_rejects_short_operands() {
        let a = vec![1.0f64; 4];
        let x = vec![1.0f64; 2];
        let mut y = vec![0.0f64; 1]; // too short for nrows=2
        assert!(gemv::dispatch_gemv::<f64>(&a, &x, &mut y, 2, 2).is_err());
    }
}
