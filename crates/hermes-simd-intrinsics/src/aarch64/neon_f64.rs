//! AArch64 NEON f64 hardware kernel.
//!
//! 2-lane f64 (`float64x2_t`). Masked ops via `vbslq_f64`.
//! Gather is emulated via individual lane loads — NEON has no native gather.
//! Compress/expand are emulated via scalar loops.

use crate::Neon;
#[cfg(target_arch = "aarch64")]
use core::arch::aarch64::*;
use hermes_simd_core::kernel::BackendKernel;

/// Newtype over `float64x2_t` providing `Send + Sync`.
#[cfg(target_arch = "aarch64")]
#[repr(transparent)]
#[derive(Copy, Clone)]
pub struct NeonF64Vec(pub float64x2_t);

#[cfg(target_arch = "aarch64")]
unsafe impl Send for NeonF64Vec {}
#[cfg(target_arch = "aarch64")]
unsafe impl Sync for NeonF64Vec {}

/// NEON f64 mask: `uint64x2_t` used as a bitwise select mask.
///
/// Lane `i` is active iff bit 63 of `mask[i]` is set — the sign-bit
/// convention shared with `mask_to_bitmask`. Every constructor
/// (`mask_from_bools`, `leading_k_mask`, `vector_to_mask`) produces canonical
/// all-ones/all-zero lanes, which is what the bitwise `vbslq_f64` merges rely
/// on.
#[cfg(target_arch = "aarch64")]
#[repr(transparent)]
#[derive(Copy, Clone)]
pub struct NeonF64Mask(pub uint64x2_t);

#[cfg(target_arch = "aarch64")]
unsafe impl Send for NeonF64Mask {}
#[cfg(target_arch = "aarch64")]
unsafe impl Sync for NeonF64Mask {}

/// Per-lane active flags keyed on bit 63, the mask-active criterion every
/// other consumer (`mask_to_bitmask` included) uses — a plain nonzero test
/// would diverge on a non-canonical mask built through the `pub` field.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline]
fn lane_actives(mask: uint64x2_t) -> [bool; 2] {
    [
        vgetq_lane_u64::<0>(mask) >> 63 == 1,
        vgetq_lane_u64::<1>(mask) >> 63 == 1,
    ]
}

#[cfg(target_arch = "aarch64")]
impl BackendKernel<f64> for Neon {
    type Vector = NeonF64Vec;
    type Mask = NeonF64Mask;
    /// Scalar index array — NEON has no native gather; emulated lane-by-lane.
    type IndexVector = [i32; 2];
    const LANE_COUNT: usize = 2;
    const UNROLL_FACTOR: usize = 4;

    // -----------------------------------------------------------------------
    // Load / store
    // -----------------------------------------------------------------------

    // SAFETY: caller must ensure the target CPU supports `neon` (enforced by the `#[target_feature]` gate above plus `cfg(target_arch = "aarch64")` selection in the hermes-simd dispatcher; NEON is baseline-mandatory on AArch64); any pointer operands are valid for the 2-lane vector width within caller-validated bounds.
    #[target_feature(enable = "neon")]
    #[inline]
    unsafe fn load_aligned(ptr: *const f64) -> Self::Vector {
        NeonF64Vec(vld1q_f64(ptr))
    }

    // SAFETY: caller must ensure the target CPU supports `neon` (enforced by the `#[target_feature]` gate above plus `cfg(target_arch = "aarch64")` selection in the hermes-simd dispatcher; NEON is baseline-mandatory on AArch64); any pointer operands are valid for the 2-lane vector width within caller-validated bounds.
    #[target_feature(enable = "neon")]
    #[inline]
    unsafe fn load_unaligned(ptr: *const f64) -> Self::Vector {
        NeonF64Vec(vld1q_f64(ptr))
    }

    // SAFETY: caller must ensure the target CPU supports `neon` (enforced by the `#[target_feature]` gate above plus `cfg(target_arch = "aarch64")` selection in the hermes-simd dispatcher; NEON is baseline-mandatory on AArch64); any pointer operands are valid for the 2-lane vector width within caller-validated bounds.
    #[target_feature(enable = "neon")]
    #[inline]
    unsafe fn store_aligned(ptr: *mut f64, val: Self::Vector) {
        vst1q_f64(ptr, val.0);
    }

    // SAFETY: caller must ensure the target CPU supports `neon` (enforced by the `#[target_feature]` gate above plus `cfg(target_arch = "aarch64")` selection in the hermes-simd dispatcher; NEON is baseline-mandatory on AArch64); any pointer operands are valid for the 2-lane vector width within caller-validated bounds.
    #[target_feature(enable = "neon")]
    #[inline]
    unsafe fn store_unaligned(ptr: *mut f64, val: Self::Vector) {
        vst1q_f64(ptr, val.0);
    }

    // -----------------------------------------------------------------------
    // Arithmetic
    // -----------------------------------------------------------------------

    // SAFETY: caller must ensure the target CPU supports `neon` (enforced by the `#[target_feature]` gate above plus `cfg(target_arch = "aarch64")` selection in the hermes-simd dispatcher; NEON is baseline-mandatory on AArch64); any pointer operands are valid for the 2-lane vector width within caller-validated bounds.
    #[target_feature(enable = "neon")]
    #[inline]
    unsafe fn add(a: Self::Vector, b: Self::Vector) -> Self::Vector {
        NeonF64Vec(vaddq_f64(a.0, b.0))
    }

    // SAFETY: caller must ensure the target CPU supports `neon` (enforced by the `#[target_feature]` gate above plus `cfg(target_arch = "aarch64")` selection in the hermes-simd dispatcher; NEON is baseline-mandatory on AArch64); any pointer operands are valid for the 2-lane vector width within caller-validated bounds.
    #[target_feature(enable = "neon")]
    #[inline]
    unsafe fn mul(a: Self::Vector, b: Self::Vector) -> Self::Vector {
        NeonF64Vec(vmulq_f64(a.0, b.0))
    }

    // SAFETY: caller must ensure the target CPU supports `neon` (enforced by the `#[target_feature]` gate above plus `cfg(target_arch = "aarch64")` selection in the hermes-simd dispatcher; NEON is baseline-mandatory on AArch64); any pointer operands are valid for the 2-lane vector width within caller-validated bounds.
    #[target_feature(enable = "neon")]
    #[inline]
    unsafe fn sub(a: Self::Vector, b: Self::Vector) -> Self::Vector {
        NeonF64Vec(vsubq_f64(a.0, b.0))
    }

    // SAFETY: caller must ensure the target CPU supports `neon` (enforced by the `#[target_feature]` gate above plus `cfg(target_arch = "aarch64")` selection in the hermes-simd dispatcher; NEON is baseline-mandatory on AArch64); any pointer operands are valid for the 2-lane vector width within caller-validated bounds.
    #[target_feature(enable = "neon")]
    #[inline]
    unsafe fn div(a: Self::Vector, b: Self::Vector) -> Self::Vector {
        NeonF64Vec(vdivq_f64(a.0, b.0))
    }

    /// `vfmaq_f64(c, a, b)` computes `a*b + c`.
    // SAFETY: caller must ensure the target CPU supports `neon` (enforced by the `#[target_feature]` gate above plus `cfg(target_arch = "aarch64")` selection in the hermes-simd dispatcher; NEON is baseline-mandatory on AArch64); any pointer operands are valid for the 2-lane vector width within caller-validated bounds.
    #[target_feature(enable = "neon")]
    #[inline]
    unsafe fn fmadd(a: Self::Vector, b: Self::Vector, c: Self::Vector) -> Self::Vector {
        NeonF64Vec(vfmaq_f64(c.0, a.0, b.0))
    }

    // SAFETY: caller must ensure the target CPU supports `neon`; negation is
    // exact and `vfmaq_f64` performs the multiply-subtract with one rounding.
    #[target_feature(enable = "neon")]
    #[inline]
    unsafe fn fmsub(a: Self::Vector, b: Self::Vector, c: Self::Vector) -> Self::Vector {
        NeonF64Vec(vfmaq_f64(vnegq_f64(c.0), a.0, b.0))
    }

    /// `vextq_f64(v, v, 1)` rotates the two lanes: `[a0, a1] -> [a1, a0]`.
    // SAFETY: caller must ensure the target CPU supports `neon` (enforced by the `#[target_feature]` gate above plus `cfg(target_arch = "aarch64")` selection in the hermes-simd dispatcher; NEON is baseline-mandatory on AArch64); any pointer operands are valid for the 2-lane vector width within caller-validated bounds.
    #[target_feature(enable = "neon")]
    #[inline]
    unsafe fn swap_adjacent(v: Self::Vector) -> Self::Vector {
        NeonF64Vec(vextq_f64(v.0, v.0, 1))
    }

    // -----------------------------------------------------------------------
    // Cross-lane permutes (native `ext`, `zip`, `uzp`)
    // -----------------------------------------------------------------------
    //
    // At two lanes, reversing and swapping the adjacent pair are the same
    // operation, so `reverse` is the same `ext` rotation as `swap_adjacent`.
    // They are kept as separate methods because they diverge at every other
    // width, and a caller reaching for one must not silently get the other.

    // SAFETY: caller must ensure the target CPU supports `neon` (as above); operands are whole registers, so no pointer validity is involved.
    #[target_feature(enable = "neon")]
    #[inline]
    #[cfg(not(hermes_benchmark_generic_default))]
    unsafe fn interleave(a: Self::Vector, b: Self::Vector) -> (Self::Vector, Self::Vector) {
        (
            NeonF64Vec(vzip1q_f64(a.0, b.0)),
            NeonF64Vec(vzip2q_f64(a.0, b.0)),
        )
    }

    // SAFETY: caller must ensure the target CPU supports `neon` (as above); operands are whole registers, so no pointer validity is involved.
    #[target_feature(enable = "neon")]
    #[inline]
    #[cfg(not(hermes_benchmark_generic_default))]
    unsafe fn deinterleave(a: Self::Vector, b: Self::Vector) -> (Self::Vector, Self::Vector) {
        (
            NeonF64Vec(vuzp1q_f64(a.0, b.0)),
            NeonF64Vec(vuzp2q_f64(a.0, b.0)),
        )
    }

    // SAFETY: caller must ensure the target CPU supports `neon` (enforced by the `#[target_feature]` gate above plus `cfg(target_arch = "aarch64")` selection in the hermes-simd dispatcher; NEON is baseline-mandatory on AArch64); any pointer operands are valid for the 2-lane vector width within caller-validated bounds.
    #[target_feature(enable = "neon")]
    #[inline]
    unsafe fn deinterleave_pairs(a: Self::Vector, b: Self::Vector) -> (Self::Vector, Self::Vector) {
        // Two lanes are one pair: the even pair is `a` whole and the odd
        // pair is `b` whole.
        (a, b)
    }

    // SAFETY: caller must ensure the target CPU supports `neon` (enforced by the `#[target_feature]` gate above plus `cfg(target_arch = "aarch64")` selection in the hermes-simd dispatcher; NEON is baseline-mandatory on AArch64); any pointer operands are valid for the 2-lane vector width within caller-validated bounds.
    #[target_feature(enable = "neon")]
    #[inline]
    unsafe fn interleave_halves(a: Self::Vector, b: Self::Vector) -> (Self::Vector, Self::Vector) {
        // A half is one lane: the zips pair the low lanes, then the high lanes.
        (
            NeonF64Vec(vzip1q_f64(a.0, b.0)),
            NeonF64Vec(vzip2q_f64(a.0, b.0)),
        )
    }

    // SAFETY: caller must ensure the target CPU supports `neon` (enforced by the `#[target_feature]` gate above plus `cfg(target_arch = "aarch64")` selection in the hermes-simd dispatcher; NEON is baseline-mandatory on AArch64); any pointer operands are valid for the 2-lane vector width within caller-validated bounds.
    #[target_feature(enable = "neon")]
    #[inline]
    unsafe fn splat_pair(lo: f64, hi: f64) -> Self::Vector {
        // Two lanes hold exactly one pair.
        NeonF64Vec(vcombine_f64(vdup_n_f64(lo), vdup_n_f64(hi)))
    }

    // SAFETY: caller must ensure the target CPU supports `neon` (enforced by the `#[target_feature]` gate above plus `cfg(target_arch = "aarch64")` selection in the hermes-simd dispatcher; NEON is baseline-mandatory on AArch64); any pointer operands are valid for the 2-lane vector width within caller-validated bounds.
    #[target_feature(enable = "neon")]
    #[inline]
    unsafe fn blend_halves(a: Self::Vector, b: Self::Vector) -> Self::Vector {
        // Each half keeps its position: one combine of the two halves.
        NeonF64Vec(vcombine_f64(vget_low_f64(a.0), vget_high_f64(b.0)))
    }

    // SAFETY: caller must ensure the target CPU supports `neon` (enforced by the `#[target_feature]` gate above plus `cfg(target_arch = "aarch64")` selection in the hermes-simd dispatcher; NEON is baseline-mandatory on AArch64); any pointer operands are valid for the 2-lane vector width within caller-validated bounds.
    #[target_feature(enable = "neon")]
    #[inline]
    unsafe fn dup_even(v: Self::Vector) -> Self::Vector {
        NeonF64Vec(vtrn1q_f64(v.0, v.0))
    }

    // SAFETY: caller must ensure the target CPU supports `neon` (enforced by the `#[target_feature]` gate above plus `cfg(target_arch = "aarch64")` selection in the hermes-simd dispatcher; NEON is baseline-mandatory on AArch64); any pointer operands are valid for the 2-lane vector width within caller-validated bounds.
    #[target_feature(enable = "neon")]
    #[inline]
    unsafe fn dup_odd(v: Self::Vector) -> Self::Vector {
        NeonF64Vec(vtrn2q_f64(v.0, v.0))
    }

    /// NEON has no alternating-FMA instruction; sign-flip the even lane of
    /// `c` (one XOR) then fuse with `vfmaq_f64`, which is rounding-identical
    /// to a native `fmaddsub`.
    // SAFETY: caller must ensure the target CPU supports `neon` (enforced by the `#[target_feature]` gate above plus `cfg(target_arch = "aarch64")` selection in the hermes-simd dispatcher; NEON is baseline-mandatory on AArch64); any pointer operands are valid for the 2-lane vector width within caller-validated bounds.
    #[target_feature(enable = "neon")]
    #[inline]
    unsafe fn fmaddsub(a: Self::Vector, b: Self::Vector, c: Self::Vector) -> Self::Vector {
        const EVEN_SIGN: [u64; 2] = [0x8000_0000_0000_0000, 0];
        let flipped = vreinterpretq_f64_u64(veorq_u64(
            vreinterpretq_u64_f64(c.0),
            vld1q_u64(EVEN_SIGN.as_ptr()),
        ));
        NeonF64Vec(vfmaq_f64(flipped, a.0, b.0))
    }

    /// Sign-flips the odd lane of `c` then fuses; see [`Self::fmaddsub`].
    // SAFETY: caller must ensure the target CPU supports `neon` (enforced by the `#[target_feature]` gate above plus `cfg(target_arch = "aarch64")` selection in the hermes-simd dispatcher; NEON is baseline-mandatory on AArch64); any pointer operands are valid for the 2-lane vector width within caller-validated bounds.
    #[target_feature(enable = "neon")]
    #[inline]
    unsafe fn fmsubadd(a: Self::Vector, b: Self::Vector, c: Self::Vector) -> Self::Vector {
        const ODD_SIGN: [u64; 2] = [0, 0x8000_0000_0000_0000];
        let flipped = vreinterpretq_f64_u64(veorq_u64(
            vreinterpretq_u64_f64(c.0),
            vld1q_u64(ODD_SIGN.as_ptr()),
        ));
        NeonF64Vec(vfmaq_f64(flipped, a.0, b.0))
    }

    // SAFETY: caller must ensure the target CPU supports `neon` (enforced by the `#[target_feature]` gate above plus `cfg(target_arch = "aarch64")` selection in the hermes-simd dispatcher; NEON is baseline-mandatory on AArch64); any pointer operands are valid for the 2-lane vector width within caller-validated bounds.
    #[target_feature(enable = "neon")]
    #[inline]
    unsafe fn sum_reduce(v: Self::Vector) -> f64 {
        vaddvq_f64(v.0)
    }

    // SAFETY: caller must ensure the target CPU supports `neon` (enforced by the `#[target_feature]` gate above plus `cfg(target_arch = "aarch64")` selection in the hermes-simd dispatcher; NEON is baseline-mandatory on AArch64); any pointer operands are valid for the 2-lane vector width within caller-validated bounds.
    #[target_feature(enable = "neon")]
    #[inline]
    unsafe fn abs(a: Self::Vector) -> Self::Vector {
        NeonF64Vec(vabsq_f64(a.0))
    }

    // SAFETY: caller must ensure the target CPU supports `neon` (enforced by the `#[target_feature]` gate above plus `cfg(target_arch = "aarch64")` selection in the hermes-simd dispatcher; NEON is baseline-mandatory on AArch64); any pointer operands are valid for the 2-lane vector width within caller-validated bounds.
    #[target_feature(enable = "neon")]
    #[inline]
    unsafe fn min(a: Self::Vector, b: Self::Vector) -> Self::Vector {
        NeonF64Vec(vminq_f64(a.0, b.0))
    }

    // SAFETY: caller must ensure the target CPU supports `neon` (enforced by the `#[target_feature]` gate above plus `cfg(target_arch = "aarch64")` selection in the hermes-simd dispatcher; NEON is baseline-mandatory on AArch64); any pointer operands are valid for the 2-lane vector width within caller-validated bounds.
    #[target_feature(enable = "neon")]
    #[inline]
    unsafe fn max(a: Self::Vector, b: Self::Vector) -> Self::Vector {
        NeonF64Vec(vmaxq_f64(a.0, b.0))
    }

    // SAFETY: caller must ensure the target CPU supports `neon` (enforced by the `#[target_feature]` gate above plus `cfg(target_arch = "aarch64")` selection in the hermes-simd dispatcher; NEON is baseline-mandatory on AArch64); any pointer operands are valid for the 2-lane vector width within caller-validated bounds.
    #[target_feature(enable = "neon")]
    #[inline]
    unsafe fn sqrt(a: Self::Vector) -> Self::Vector {
        NeonF64Vec(vsqrtq_f64(a.0))
    }

    // SAFETY: caller must ensure the target CPU supports `neon` (enforced by the `#[target_feature]` gate above plus `cfg(target_arch = "aarch64")` selection in the hermes-simd dispatcher; NEON is baseline-mandatory on AArch64); any pointer operands are valid for the 2-lane vector width within caller-validated bounds.
    #[target_feature(enable = "neon")]
    #[inline]
    unsafe fn recip_sqrt(a: Self::Vector) -> Self::Vector {
        // `vrsqrteq_f64` seeds only ~8 bits; reaching f64's 52-bit mantissa by
        // Newton iteration would need ~3 steps. The correctly-rounded hardware
        // sqrt + divide gives full f64 precision (~1 ulp) directly, matching the
        // x86 and scalar paths.
        NeonF64Vec(vdivq_f64(vdupq_n_f64(1.0), vsqrtq_f64(a.0)))
    }

    // SAFETY: caller must ensure the target CPU supports `neon` (enforced by the `#[target_feature]` gate above plus `cfg(target_arch = "aarch64")` selection in the hermes-simd dispatcher; NEON is baseline-mandatory on AArch64); any pointer operands are valid for the 2-lane vector width within caller-validated bounds.
    #[target_feature(enable = "neon")]
    #[inline]
    unsafe fn floor(a: Self::Vector) -> Self::Vector {
        // `FRINTM`: round toward minus infinity.
        NeonF64Vec(vrndmq_f64(a.0))
    }

    // SAFETY: caller must ensure the target CPU supports `neon` (enforced by the `#[target_feature]` gate above plus `cfg(target_arch = "aarch64")` selection in the hermes-simd dispatcher; NEON is baseline-mandatory on AArch64); any pointer operands are valid for the 2-lane vector width within caller-validated bounds.
    #[target_feature(enable = "neon")]
    #[inline]
    unsafe fn ceil(a: Self::Vector) -> Self::Vector {
        // `FRINTP`: round toward plus infinity.
        NeonF64Vec(vrndpq_f64(a.0))
    }

    // SAFETY: caller must ensure the target CPU supports `neon` (enforced by the `#[target_feature]` gate above plus `cfg(target_arch = "aarch64")` selection in the hermes-simd dispatcher; NEON is baseline-mandatory on AArch64); any pointer operands are valid for the 2-lane vector width within caller-validated bounds.
    #[target_feature(enable = "neon")]
    #[inline]
    unsafe fn round(a: Self::Vector) -> Self::Vector {
        // `FRINTN`: round to nearest, ties to even — the `round_ties_even` contract.
        NeonF64Vec(vrndnq_f64(a.0))
    }

    // SAFETY: caller must ensure the target CPU supports `neon` (enforced by the `#[target_feature]` gate above plus `cfg(target_arch = "aarch64")` selection in the hermes-simd dispatcher; NEON is baseline-mandatory on AArch64); any pointer operands are valid for the 2-lane vector width within caller-validated bounds.
    #[target_feature(enable = "neon")]
    #[inline]
    unsafe fn trunc(a: Self::Vector) -> Self::Vector {
        // `FRINTZ`: round toward zero.
        NeonF64Vec(vrndq_f64(a.0))
    }

    // SAFETY: caller must ensure the target CPU supports `neon` (enforced by the `#[target_feature]` gate above plus `cfg(target_arch = "aarch64")` selection in the hermes-simd dispatcher; NEON is baseline-mandatory on AArch64); any pointer operands are valid for the 2-lane vector width within caller-validated bounds.
    #[target_feature(enable = "neon")]
    #[inline]
    unsafe fn popcount(a: Self::Vector) -> Self::Vector {
        let v_u8 = vreinterpretq_u8_f64(a.0);
        let pop_bytes = vcntq_u8(v_u8);
        let pop_u16 = vpaddlq_u8(pop_bytes);
        let pop_u32 = vpaddlq_u16(pop_u16);
        let pop_u64 = vpaddlq_u32(pop_u32);
        let pop_f64 = vcvtq_f64_u64(pop_u64);
        NeonF64Vec(pop_f64)
    }

    // SAFETY: caller must ensure the target CPU supports `neon` (enforced by the `#[target_feature]` gate above plus `cfg(target_arch = "aarch64")` selection in the hermes-simd dispatcher; NEON is baseline-mandatory on AArch64); any pointer operands are valid for the 2-lane vector width within caller-validated bounds.
    #[target_feature(enable = "neon")]
    #[inline]
    unsafe fn bitand(a: Self::Vector, b: Self::Vector) -> Self::Vector {
        NeonF64Vec(vreinterpretq_f64_u64(vandq_u64(
            vreinterpretq_u64_f64(a.0),
            vreinterpretq_u64_f64(b.0),
        )))
    }

    // SAFETY: caller must ensure the target CPU supports `neon` (enforced by the `#[target_feature]` gate above plus `cfg(target_arch = "aarch64")` selection in the hermes-simd dispatcher; NEON is baseline-mandatory on AArch64); any pointer operands are valid for the 2-lane vector width within caller-validated bounds.
    #[target_feature(enable = "neon")]
    #[inline]
    unsafe fn bitor(a: Self::Vector, b: Self::Vector) -> Self::Vector {
        NeonF64Vec(vreinterpretq_f64_u64(vorrq_u64(
            vreinterpretq_u64_f64(a.0),
            vreinterpretq_u64_f64(b.0),
        )))
    }

    // SAFETY: caller must ensure the target CPU supports `neon` (enforced by the `#[target_feature]` gate above plus `cfg(target_arch = "aarch64")` selection in the hermes-simd dispatcher; NEON is baseline-mandatory on AArch64); any pointer operands are valid for the 2-lane vector width within caller-validated bounds.
    #[target_feature(enable = "neon")]
    #[inline]
    unsafe fn bitxor(a: Self::Vector, b: Self::Vector) -> Self::Vector {
        NeonF64Vec(vreinterpretq_f64_u64(veorq_u64(
            vreinterpretq_u64_f64(a.0),
            vreinterpretq_u64_f64(b.0),
        )))
    }

    // SAFETY: caller must ensure the target CPU supports `neon` (enforced by the `#[target_feature]` gate above plus `cfg(target_arch = "aarch64")` selection in the hermes-simd dispatcher; NEON is baseline-mandatory on AArch64); any pointer operands are valid for the 2-lane vector width within caller-validated bounds.
    #[target_feature(enable = "neon")]
    #[inline]
    unsafe fn cmp_eq(a: Self::Vector, b: Self::Vector) -> Self::Vector {
        NeonF64Vec(vreinterpretq_f64_u64(vceqq_f64(a.0, b.0)))
    }

    // SAFETY: caller must ensure the target CPU supports `neon` (enforced by the `#[target_feature]` gate above plus `cfg(target_arch = "aarch64")` selection in the hermes-simd dispatcher; NEON is baseline-mandatory on AArch64); any pointer operands are valid for the 2-lane vector width within caller-validated bounds.
    #[target_feature(enable = "neon")]
    #[inline]
    unsafe fn cmp_ne(a: Self::Vector, b: Self::Vector) -> Self::Vector {
        NeonF64Vec(vreinterpretq_f64_u64(vreinterpretq_u64_u32(vmvnq_u32(
            vreinterpretq_u32_u64(vceqq_f64(a.0, b.0)),
        ))))
    }

    // SAFETY: caller must ensure the target CPU supports `neon` (enforced by the `#[target_feature]` gate above plus `cfg(target_arch = "aarch64")` selection in the hermes-simd dispatcher; NEON is baseline-mandatory on AArch64); any pointer operands are valid for the 2-lane vector width within caller-validated bounds.
    #[target_feature(enable = "neon")]
    #[inline]
    unsafe fn cmp_lt(a: Self::Vector, b: Self::Vector) -> Self::Vector {
        NeonF64Vec(vreinterpretq_f64_u64(vcltq_f64(a.0, b.0)))
    }

    // SAFETY: caller must ensure the target CPU supports `neon` (enforced by the `#[target_feature]` gate above plus `cfg(target_arch = "aarch64")` selection in the hermes-simd dispatcher; NEON is baseline-mandatory on AArch64); any pointer operands are valid for the 2-lane vector width within caller-validated bounds.
    #[target_feature(enable = "neon")]
    #[inline]
    unsafe fn cmp_le(a: Self::Vector, b: Self::Vector) -> Self::Vector {
        NeonF64Vec(vreinterpretq_f64_u64(vcleq_f64(a.0, b.0)))
    }

    // SAFETY: caller must ensure the target CPU supports `neon` (enforced by the `#[target_feature]` gate above plus `cfg(target_arch = "aarch64")` selection in the hermes-simd dispatcher; NEON is baseline-mandatory on AArch64); any pointer operands are valid for the 2-lane vector width within caller-validated bounds.
    #[target_feature(enable = "neon")]
    #[inline]
    unsafe fn cmp_gt(a: Self::Vector, b: Self::Vector) -> Self::Vector {
        NeonF64Vec(vreinterpretq_f64_u64(vcgtq_f64(a.0, b.0)))
    }

    // SAFETY: caller must ensure the target CPU supports `neon` (enforced by the `#[target_feature]` gate above plus `cfg(target_arch = "aarch64")` selection in the hermes-simd dispatcher; NEON is baseline-mandatory on AArch64); any pointer operands are valid for the 2-lane vector width within caller-validated bounds.
    #[target_feature(enable = "neon")]
    #[inline]
    unsafe fn cmp_ge(a: Self::Vector, b: Self::Vector) -> Self::Vector {
        NeonF64Vec(vreinterpretq_f64_u64(vcgeq_f64(a.0, b.0)))
    }

    // SAFETY: caller must ensure the target CPU supports `neon` (enforced by the `#[target_feature]` gate above plus `cfg(target_arch = "aarch64")` selection in the hermes-simd dispatcher; NEON is baseline-mandatory on AArch64); any pointer operands are valid for the 2-lane vector width within caller-validated bounds.
    #[target_feature(enable = "neon")]
    #[inline]
    unsafe fn blend(
        mask: Self::Vector,
        true_val: Self::Vector,
        false_val: Self::Vector,
    ) -> Self::Vector {
        // Selection is on the mask lane's sign bit, but `vbsl` is a *bitwise*
        // select that would splice the operands bit-by-bit on a non-canonical
        // mask. Broadcast each lane's sign across the lane first (arithmetic
        // shift right by 63), so `vbsl` sees all-ones or all-zeros per lane.
        let sign = vreinterpretq_u64_s64(vshrq_n_s64::<63>(vreinterpretq_s64_f64(mask.0)));
        NeonF64Vec(vbslq_f64(sign, true_val.0, false_val.0))
    }

    // -----------------------------------------------------------------------
    // Masked load / store
    // -----------------------------------------------------------------------

    // SAFETY: caller must ensure the target CPU supports `neon` (enforced by the `#[target_feature]` gate above plus `cfg(target_arch = "aarch64")` selection in the hermes-simd dispatcher; NEON is baseline-mandatory on AArch64); any pointer operands are valid for the 2-lane vector width within caller-validated bounds.
    #[target_feature(enable = "neon")]
    #[inline]
    unsafe fn masked_load_unaligned(
        ptr: *const f64,
        mask: Self::Mask,
        src: Self::Vector,
    ) -> Self::Vector {
        NeonF64Vec(vbslq_f64(mask.0, vld1q_f64(ptr), src.0))
    }

    // SAFETY: caller must ensure the target CPU supports `neon` (enforced by the `#[target_feature]` gate above plus `cfg(target_arch = "aarch64")` selection in the hermes-simd dispatcher; NEON is baseline-mandatory on AArch64); any pointer operands are valid for the 2-lane vector width within caller-validated bounds.
    #[target_feature(enable = "neon")]
    #[inline]
    unsafe fn masked_store_unaligned(ptr: *mut f64, mask: Self::Mask, val: Self::Vector) {
        let current = vld1q_f64(ptr);
        vst1q_f64(ptr, vbslq_f64(mask.0, val.0, current));
    }

    // -----------------------------------------------------------------------
    // Masked arithmetic
    // -----------------------------------------------------------------------

    // SAFETY: caller must ensure the target CPU supports `neon` (enforced by the `#[target_feature]` gate above plus `cfg(target_arch = "aarch64")` selection in the hermes-simd dispatcher; NEON is baseline-mandatory on AArch64); any pointer operands are valid for the 2-lane vector width within caller-validated bounds.
    #[target_feature(enable = "neon")]
    #[inline]
    unsafe fn masked_add(
        a: Self::Vector,
        b: Self::Vector,
        mask: Self::Mask,
        src: Self::Vector,
    ) -> Self::Vector {
        NeonF64Vec(vbslq_f64(mask.0, vaddq_f64(a.0, b.0), src.0))
    }

    // SAFETY: caller must ensure the target CPU supports `neon` (enforced by the `#[target_feature]` gate above plus `cfg(target_arch = "aarch64")` selection in the hermes-simd dispatcher; NEON is baseline-mandatory on AArch64); any pointer operands are valid for the 2-lane vector width within caller-validated bounds.
    #[target_feature(enable = "neon")]
    #[inline]
    unsafe fn masked_mul(
        a: Self::Vector,
        b: Self::Vector,
        mask: Self::Mask,
        src: Self::Vector,
    ) -> Self::Vector {
        NeonF64Vec(vbslq_f64(mask.0, vmulq_f64(a.0, b.0), src.0))
    }

    // SAFETY: caller must ensure the target CPU supports `neon` (enforced by the `#[target_feature]` gate above plus `cfg(target_arch = "aarch64")` selection in the hermes-simd dispatcher; NEON is baseline-mandatory on AArch64); any pointer operands are valid for the 2-lane vector width within caller-validated bounds.
    #[target_feature(enable = "neon")]
    #[inline]
    unsafe fn masked_fmadd(
        a: Self::Vector,
        b: Self::Vector,
        c: Self::Vector,
        mask: Self::Mask,
    ) -> Self::Vector {
        // Inactive lanes retain c (addend pass-through).
        NeonF64Vec(vbslq_f64(mask.0, vfmaq_f64(c.0, a.0, b.0), c.0))
    }

    // SAFETY: caller must ensure the target CPU supports `neon` (enforced by the `#[target_feature]` gate above plus `cfg(target_arch = "aarch64")` selection in the hermes-simd dispatcher; NEON is baseline-mandatory on AArch64); any pointer operands are valid for the 2-lane vector width within caller-validated bounds.
    #[target_feature(enable = "neon")]
    #[inline]
    unsafe fn masked_sum_reduce(v: Self::Vector, mask: Self::Mask) -> f64 {
        vaddvq_f64(vbslq_f64(mask.0, v.0, vdupq_n_f64(0.0)))
    }

    // -----------------------------------------------------------------------
    // Compress / Expand (emulated)
    // -----------------------------------------------------------------------

    // SAFETY: caller must ensure the target CPU supports `neon` (enforced by the `#[target_feature]` gate above plus `cfg(target_arch = "aarch64")` selection in the hermes-simd dispatcher; NEON is baseline-mandatory on AArch64); any pointer operands are valid for the 2-lane vector width within caller-validated bounds.
    #[target_feature(enable = "neon")]
    #[inline]
    unsafe fn compress(src: Self::Vector, mask: Self::Mask) -> Self::Vector {
        let mut arr = [0.0f64; 2];
        vst1q_f64(arr.as_mut_ptr(), src.0);
        let m = lane_actives(mask.0);
        let mut out = [0.0f64; 2];
        let mut k = 0usize;
        for i in 0..2 {
            if m[i] {
                out[k] = arr[i];
                k += 1;
            }
        }
        NeonF64Vec(vld1q_f64(out.as_ptr()))
    }

    // SAFETY: caller must ensure the target CPU supports `neon` (enforced by the `#[target_feature]` gate above plus `cfg(target_arch = "aarch64")` selection in the hermes-simd dispatcher; NEON is baseline-mandatory on AArch64); any pointer operands are valid for the 2-lane vector width within caller-validated bounds.
    #[target_feature(enable = "neon")]
    #[inline]
    unsafe fn expand(src: Self::Vector, mask: Self::Mask, fill: Self::Vector) -> Self::Vector {
        let mut src_arr = [0.0f64; 2];
        vst1q_f64(src_arr.as_mut_ptr(), src.0);
        let mut out_arr = [0.0f64; 2];
        vst1q_f64(out_arr.as_mut_ptr(), fill.0);
        let m = lane_actives(mask.0);
        let mut k = 0usize;
        for i in 0..2 {
            if m[i] {
                out_arr[i] = src_arr[k];
                k += 1;
            }
        }
        NeonF64Vec(vld1q_f64(out_arr.as_ptr()))
    }

    // -----------------------------------------------------------------------
    // Gather (emulated)
    // -----------------------------------------------------------------------

    // SAFETY: caller must ensure the target CPU supports `neon` (enforced by the `#[target_feature]` gate above plus `cfg(target_arch = "aarch64")` selection in the hermes-simd dispatcher; NEON is baseline-mandatory on AArch64); any pointer operands are valid for the 2-lane vector width within caller-validated bounds.
    #[target_feature(enable = "neon")]
    #[inline]
    unsafe fn gather(base: *const f64, indices: Self::IndexVector) -> Self::Vector {
        let arr = [
            *base.add(indices[0] as usize),
            *base.add(indices[1] as usize),
        ];
        NeonF64Vec(vld1q_f64(arr.as_ptr()))
    }

    // SAFETY: caller must ensure the target CPU supports `neon` (enforced by the `#[target_feature]` gate above plus `cfg(target_arch = "aarch64")` selection in the hermes-simd dispatcher; NEON is baseline-mandatory on AArch64); any pointer operands are valid for the 2-lane vector width within caller-validated bounds.
    #[target_feature(enable = "neon")]
    #[inline]
    unsafe fn gather_masked(
        base: *const f64,
        indices: Self::IndexVector,
        mask: Self::Mask,
        src: Self::Vector,
    ) -> Self::Vector {
        let mut src_arr = [0.0f64; 2];
        vst1q_f64(src_arr.as_mut_ptr(), src.0);
        let m = lane_actives(mask.0);
        let out = [
            if m[0] {
                *base.add(indices[0] as usize)
            } else {
                src_arr[0]
            },
            if m[1] {
                *base.add(indices[1] as usize)
            } else {
                src_arr[1]
            },
        ];
        NeonF64Vec(vld1q_f64(out.as_ptr()))
    }

    // -----------------------------------------------------------------------
    // Mask construction
    // -----------------------------------------------------------------------

    // SAFETY: caller must ensure the target CPU supports `neon` (enforced by the `#[target_feature]` gate above plus `cfg(target_arch = "aarch64")` selection in the hermes-simd dispatcher; NEON is baseline-mandatory on AArch64); any pointer operands are valid for the 2-lane vector width within caller-validated bounds.
    #[target_feature(enable = "neon")]
    #[inline]
    unsafe fn mask_from_bools(bits: &[bool]) -> Self::Mask {
        debug_assert_eq!(bits.len(), 2);
        let vals: [u64; 2] = [
            if bits[0] { !0u64 } else { 0 },
            if bits[1] { !0u64 } else { 0 },
        ];
        NeonF64Mask(vld1q_u64(vals.as_ptr()))
    }

    // SAFETY: caller must ensure the target CPU supports `neon` (enforced by the `#[target_feature]` gate above plus `cfg(target_arch = "aarch64")` selection in the hermes-simd dispatcher; NEON is baseline-mandatory on AArch64); any pointer operands are valid for the 2-lane vector width within caller-validated bounds.
    #[target_feature(enable = "neon")]
    #[inline]
    unsafe fn leading_k_mask(k: usize) -> Self::Mask {
        let vals: [u64; 2] = [if k > 0 { !0u64 } else { 0 }, if k > 1 { !0u64 } else { 0 }];
        NeonF64Mask(vld1q_u64(vals.as_ptr()))
    }

    // SAFETY: caller must ensure the target CPU supports `neon` (enforced by the `#[target_feature]` gate above plus `cfg(target_arch = "aarch64")` selection in the hermes-simd dispatcher; NEON is baseline-mandatory on AArch64); the only memory operand is the constant lane-bit table.
    #[target_feature(enable = "neon")]
    #[inline]
    unsafe fn mask_from_bitmask(bm: u64) -> Self::Mask {
        // `vtst` sets a lane to all-ones where `(bits & lane_bit) != 0` —
        // canonical expansion in one test instruction, replacing the generic
        // bool-array bounce (bits 2.. are ignored because only bits 0..2
        // appear in the table).
        let lane_bits: [u64; 2] = [1, 2];
        let bits = vdupq_n_u64(bm);
        NeonF64Mask(vtstq_u64(bits, vld1q_u64(lane_bits.as_ptr())))
    }

    // -----------------------------------------------------------------------
    // Broadcast / zero
    // -----------------------------------------------------------------------

    // SAFETY: caller must ensure the target CPU supports `neon` (enforced by the `#[target_feature]` gate above plus `cfg(target_arch = "aarch64")` selection in the hermes-simd dispatcher; NEON is baseline-mandatory on AArch64); any pointer operands are valid for the 2-lane vector width within caller-validated bounds.
    #[target_feature(enable = "neon")]
    #[inline]
    unsafe fn zero() -> Self::Vector {
        NeonF64Vec(vdupq_n_f64(0.0))
    }

    // SAFETY: caller must ensure the target CPU supports `neon` (enforced by the `#[target_feature]` gate above plus `cfg(target_arch = "aarch64")` selection in the hermes-simd dispatcher; NEON is baseline-mandatory on AArch64); any pointer operands are valid for the 2-lane vector width within caller-validated bounds.
    #[target_feature(enable = "neon")]
    #[inline]
    unsafe fn splat(val: f64) -> Self::Vector {
        NeonF64Vec(vdupq_n_f64(val))
    }

    // SAFETY: caller must ensure the target CPU supports `neon` (enforced by the `#[target_feature]` gate above plus `cfg(target_arch = "aarch64")` selection in the hermes-simd dispatcher; NEON is baseline-mandatory on AArch64); any pointer operands are valid for the 2-lane vector width within caller-validated bounds.
    #[target_feature(enable = "neon")]
    #[inline]
    unsafe fn mask_to_bitmask(mask: Self::Mask) -> u64 {
        let mut m = 0u64;
        m |= ((vgetq_lane_u64(mask.0, 0) >> 63) as u64) << 0;
        m |= ((vgetq_lane_u64(mask.0, 1) >> 63) as u64) << 1;
        m
    }

    // SAFETY: caller must ensure the target CPU supports `neon` (enforced by the `#[target_feature]` gate above plus `cfg(target_arch = "aarch64")` selection in the hermes-simd dispatcher; NEON is baseline-mandatory on AArch64); any pointer operands are valid for the 2-lane vector width within caller-validated bounds.
    #[target_feature(enable = "neon")]
    #[inline]
    unsafe fn mask_to_vector(mask: Self::Mask) -> Self::Vector {
        NeonF64Vec(vreinterpretq_f64_u64(mask.0))
    }

    // SAFETY: caller must ensure the target CPU supports `neon` (enforced by the `#[target_feature]` gate above plus runtime selection in the hermes-simd dispatcher); this is a register-to-register reinterpretation with no memory operands.
    #[target_feature(enable = "neon")]
    #[inline]
    unsafe fn vector_to_mask(v: Self::Vector) -> Self::Mask {
        // Canonicalize on entry: broadcast each lane's sign bit (the documented
        // active criterion) across the lane with an arithmetic shift right by
        // 63, so every `Mask` consumer — the bitwise `vbsl` merges included —
        // sees all-ones or all-zeros per lane regardless of the mask vector's
        // remaining bits. A bare reinterpretation kept non-canonical bits and
        // let `vbsl`-based masked ops splice operands bit-by-bit.
        NeonF64Mask(vreinterpretq_u64_s64(vshrq_n_s64::<63>(
            vreinterpretq_s64_f64(v.0),
        )))
    }
}
