//! Generic runtime-dispatch tiled GEMM kernel.
//!
//! # Tiling policy selection
//!
//! The tiling policy is selected solely from the **runtime-dispatched
//! architecture** `A`, not from a secondary hardware-detection pass.  This
//! avoids a historical double-dispatch bug where `AdaptiveDispatcher` would
//! detect AVX-512 hardware and route the AVX2 kernel through `TilingPolicy<3,
//! 4>` (17 live registers → spill on AVX2's 16-register file, measured
//! 30–60 % slower at 256²).  The architecture parameter `A` already encodes
//! the correct register file width via `A::LANE_COUNT`.
//!
//! | `A::LANE_COUNT` | Register file    | Tiling policy | Live registers |
//! |-----------------|------------------|---------------|----------------|
//! | > 8             | AVX-512 (32 regs)| `<6, 4>`      | 24+4+1 = 29    |
//! | > 1             | AVX2 (16 regs)   | `<3, 3>`      | 9+3+1 = 13     |
//! | > 1             | NEON (32 regs)   | `<3, 3>`      | 9+3+1 = 13     |
//! | 1               | Scalar           | `<1, 1>`      | 1+1+1 = 3      |

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
pub(super) fn dispatch_tiled_gemm_kernel<T, A>(
    a: &[T],
    b: &[T],
    c: &mut [T],
    m: usize,
    n: usize,
    k: usize,
) -> Result<(), SimdError>
where
    T: Scalar,
    A: SimdArch + SimdKernel<T>,
{
    match (
        SimdView::<T, A, Unaligned, Unmasked, &[T]>::new(a),
        SimdView::<T, A, Unaligned, Unmasked, &[T]>::new(b),
    ) {
        (Some(v1), Some(v2)) => {
            use hermes_simd_core::tiling::{TilingPolicy, TilingStrategy};

            if A::LANE_COUNT > 8 && m >= 16 && n >= 16 && k >= 32 {
                // AVX-512 class (32 vector registers). `<6, 4>` holds 24
                // accumulators + 4 B-vectors + 1 broadcast A-scalar = 29
                // registers, fitting within the 32-register file with
                // headroom for loop temporaries.
                <TilingPolicy<6, 4> as TilingStrategy<T, A, Unaligned>>::gemm(&v1, &v2, c, m, n, k)
            } else if A::LANE_COUNT > 1 && m >= 16 && n >= 16 && k >= 32 {
                // AVX2/NEON class (16 vector registers). `<3, 3>` holds 9
                // accumulators + 3 B-vectors + 1 broadcast A-scalar = 13
                // registers, leaving headroom for loop temporaries (no
                // spill). `<3,4>` (12+4+1 = 17) and `<4,3>` (12+3+1 = 16,
                // zero headroom) both spill on a 16-register file and were
                // measured ~30-60% slower at 256².
                <TilingPolicy<3, 3> as TilingStrategy<T, A, Unaligned>>::gemm(&v1, &v2, c, m, n, k)
            } else {
                <TilingPolicy<1, 1> as TilingStrategy<T, A, Unaligned>>::gemm(&v1, &v2, c, m, n, k)
            }
        }
        _ => unsafe { core::hint::unreachable_unchecked() },
    }
}

#[cfg(test)]
mod tests {
    use crate::dispatch::gemm;
    use hermes_simd_core::view::SimdError;

    #[test]
    fn tiled_gemm_rejects_dimension_overflow() {
        // `m·k` overflows `usize`. Unchecked (release `overflow-checks = false`)
        // the product wraps, the `a_len < a_needed` guard passes, and the kernel
        // issues an OOB SIMD load/store. The checked area arithmetic must reject
        // with the exact variant. (Operand correctness is covered in
        // `tests/tiling_tests.rs`.)
        let a = vec![1.0f64; 16];
        let b = vec![1.0f64; 16];
        let mut c = vec![0.0f64; 16];
        let r = gemm::dispatch_tiled_gemm::<f64>(&a, &b, &mut c, 2, 8, usize::MAX);
        assert_eq!(r, Err(SimdError::LengthMismatch));
    }
}
