//! AArch64 NEON f32 hardware kernel.
//!
//! 4-lane f32 (`float32x4_t`). Masked ops use `vbslq_f32` (bitwise select).
//! Gather is emulated via four individual lane loads — NEON has no native gather.
//! Compress/expand are emulated via scalar loops.

use crate::Neon;
#[cfg(target_arch = "aarch64")]
use core::arch::aarch64::*;
use hermes_simd_core::kernel::BackendKernel;

/// Newtype over `float32x4_t` providing `Send + Sync`.
#[cfg(target_arch = "aarch64")]
#[repr(transparent)]
#[derive(Copy, Clone)]
pub struct NeonF32Vec(pub float32x4_t);

#[cfg(target_arch = "aarch64")]
unsafe impl Send for NeonF32Vec {}
#[cfg(target_arch = "aarch64")]
unsafe impl Sync for NeonF32Vec {}

/// NEON f32 mask: `uint32x4_t` used as a bitwise select mask.
///
/// Lane `i` is active iff bit 31 of `mask[i]` is set — the sign-bit
/// convention shared with `mask_to_bitmask`. Every constructor
/// (`mask_from_bools`, `leading_k_mask`, `vector_to_mask`) produces canonical
/// all-ones/all-zero lanes, which is what the bitwise `vbslq_f32` merges rely
/// on (true lane selects from the first argument).
#[cfg(target_arch = "aarch64")]
#[repr(transparent)]
#[derive(Copy, Clone)]
pub struct NeonF32Mask(pub uint32x4_t);

#[cfg(target_arch = "aarch64")]
unsafe impl Send for NeonF32Mask {}
#[cfg(target_arch = "aarch64")]
unsafe impl Sync for NeonF32Mask {}

/// Per-lane active flags keyed on bit 31, the mask-active criterion every
/// other consumer (`mask_to_bitmask` included) uses — a plain nonzero test
/// would diverge on a non-canonical mask built through the `pub` field.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline]
fn lane_actives(mask: uint32x4_t) -> [bool; 4] {
    [
        vgetq_lane_u32::<0>(mask) >> 31 == 1,
        vgetq_lane_u32::<1>(mask) >> 31 == 1,
        vgetq_lane_u32::<2>(mask) >> 31 == 1,
        vgetq_lane_u32::<3>(mask) >> 31 == 1,
    ]
}

#[cfg(target_arch = "aarch64")]
impl BackendKernel<f32> for Neon {
    type Vector = NeonF32Vec;
    type Mask = NeonF32Mask;
    /// Scalar index array — NEON has no native gather; emulated lane-by-lane.
    type IndexVector = [i32; 4];
    const LANE_COUNT: usize = 4;
    const UNROLL_FACTOR: usize = 4;

    // -----------------------------------------------------------------------
    // Load / store
    // -----------------------------------------------------------------------

    // SAFETY: caller must ensure the target CPU supports `neon` (enforced by the `#[target_feature]` gate above plus `cfg(target_arch = "aarch64")` selection in the hermes-simd dispatcher; NEON is baseline-mandatory on AArch64); any pointer operands are valid for the 4-lane vector width within caller-validated bounds.
    #[target_feature(enable = "neon")]
    #[inline]
    unsafe fn load_aligned(ptr: *const f32) -> Self::Vector {
        NeonF32Vec(vld1q_f32(ptr))
    }

    // SAFETY: caller must ensure the target CPU supports `neon` (enforced by the `#[target_feature]` gate above plus `cfg(target_arch = "aarch64")` selection in the hermes-simd dispatcher; NEON is baseline-mandatory on AArch64); any pointer operands are valid for the 4-lane vector width within caller-validated bounds.
    #[target_feature(enable = "neon")]
    #[inline]
    unsafe fn load_unaligned(ptr: *const f32) -> Self::Vector {
        NeonF32Vec(vld1q_f32(ptr))
    }

    // SAFETY: caller must ensure the target CPU supports `neon` (enforced by the `#[target_feature]` gate above plus `cfg(target_arch = "aarch64")` selection in the hermes-simd dispatcher; NEON is baseline-mandatory on AArch64); any pointer operands are valid for the 4-lane vector width within caller-validated bounds.
    #[target_feature(enable = "neon")]
    #[inline]
    unsafe fn store_aligned(ptr: *mut f32, val: Self::Vector) {
        vst1q_f32(ptr, val.0);
    }

    // SAFETY: caller must ensure the target CPU supports `neon` (enforced by the `#[target_feature]` gate above plus `cfg(target_arch = "aarch64")` selection in the hermes-simd dispatcher; NEON is baseline-mandatory on AArch64); any pointer operands are valid for the 4-lane vector width within caller-validated bounds.
    #[target_feature(enable = "neon")]
    #[inline]
    unsafe fn store_unaligned(ptr: *mut f32, val: Self::Vector) {
        vst1q_f32(ptr, val.0);
    }

    // -----------------------------------------------------------------------
    // Arithmetic
    // -----------------------------------------------------------------------

    // SAFETY: caller must ensure the target CPU supports `neon` (enforced by the `#[target_feature]` gate above plus `cfg(target_arch = "aarch64")` selection in the hermes-simd dispatcher; NEON is baseline-mandatory on AArch64); any pointer operands are valid for the 4-lane vector width within caller-validated bounds.
    #[target_feature(enable = "neon")]
    #[inline]
    unsafe fn add(a: Self::Vector, b: Self::Vector) -> Self::Vector {
        NeonF32Vec(vaddq_f32(a.0, b.0))
    }

    // SAFETY: caller must ensure the target CPU supports `neon` (enforced by the `#[target_feature]` gate above plus `cfg(target_arch = "aarch64")` selection in the hermes-simd dispatcher; NEON is baseline-mandatory on AArch64); any pointer operands are valid for the 4-lane vector width within caller-validated bounds.
    #[target_feature(enable = "neon")]
    #[inline]
    unsafe fn mul(a: Self::Vector, b: Self::Vector) -> Self::Vector {
        NeonF32Vec(vmulq_f32(a.0, b.0))
    }

    // SAFETY: caller must ensure the target CPU supports `neon` (enforced by the `#[target_feature]` gate above plus `cfg(target_arch = "aarch64")` selection in the hermes-simd dispatcher; NEON is baseline-mandatory on AArch64); any pointer operands are valid for the 4-lane vector width within caller-validated bounds.
    #[target_feature(enable = "neon")]
    #[inline]
    unsafe fn sub(a: Self::Vector, b: Self::Vector) -> Self::Vector {
        NeonF32Vec(vsubq_f32(a.0, b.0))
    }

    // SAFETY: caller must ensure the target CPU supports `neon` (enforced by the `#[target_feature]` gate above plus `cfg(target_arch = "aarch64")` selection in the hermes-simd dispatcher; NEON is baseline-mandatory on AArch64); any pointer operands are valid for the 4-lane vector width within caller-validated bounds.
    #[target_feature(enable = "neon")]
    #[inline]
    unsafe fn div(a: Self::Vector, b: Self::Vector) -> Self::Vector {
        NeonF32Vec(vdivq_f32(a.0, b.0))
    }

    /// NEON FMA: `vfmaq_f32(c, a, b)` computes `a*b + c`.
    // SAFETY: caller must ensure the target CPU supports `neon` (enforced by the `#[target_feature]` gate above plus `cfg(target_arch = "aarch64")` selection in the hermes-simd dispatcher; NEON is baseline-mandatory on AArch64); any pointer operands are valid for the 4-lane vector width within caller-validated bounds.
    #[target_feature(enable = "neon")]
    #[inline]
    unsafe fn fmadd(a: Self::Vector, b: Self::Vector, c: Self::Vector) -> Self::Vector {
        NeonF32Vec(vfmaq_f32(c.0, a.0, b.0))
    }

    // SAFETY: caller must ensure the target CPU supports `neon`; negation is
    // exact and `vfmaq_f32` performs the multiply-subtract with one rounding.
    #[target_feature(enable = "neon")]
    #[inline]
    unsafe fn fmsub(a: Self::Vector, b: Self::Vector, c: Self::Vector) -> Self::Vector {
        NeonF32Vec(vfmaq_f32(vnegq_f32(c.0), a.0, b.0))
    }

    /// `vrev64q_f32` swaps 32-bit lanes within each 64-bit doubleword.
    // SAFETY: caller must ensure the target CPU supports `neon` (enforced by the `#[target_feature]` gate above plus `cfg(target_arch = "aarch64")` selection in the hermes-simd dispatcher; NEON is baseline-mandatory on AArch64); any pointer operands are valid for the 4-lane vector width within caller-validated bounds.
    #[target_feature(enable = "neon")]
    #[inline]
    unsafe fn swap_adjacent(v: Self::Vector) -> Self::Vector {
        NeonF32Vec(vrev64q_f32(v.0))
    }

    // SAFETY: caller must ensure the target CPU supports `neon` (enforced by the `#[target_feature]` gate above plus runtime feature selection in the hermes-simd dispatcher); operands are valid 4-lane vectors.
    #[target_feature(enable = "neon")]
    #[inline]
    unsafe fn swap_pairs(v: Self::Vector) -> Self::Vector {
        // Four f32 lanes hold two pairs; rotating by two lanes exchanges them.
        NeonF32Vec(vextq_f32::<2>(v.0, v.0))
    }

    // -----------------------------------------------------------------------
    // Cross-lane permutes (native `rev64` + `ext`, `zip`, `uzp`)
    // -----------------------------------------------------------------------
    //
    // NEON's zip/uzp are defined over the full 128-bit register rather than a
    // sub-lane, so at this width they *are* the flat interleave and
    // deinterleave the trait specifies — unlike x86 `unpack`, which works
    // within 128-bit halves of a wider register and needs a cross-half fixup.

    // SAFETY: caller must ensure the target CPU supports `neon` (as above); operands are whole registers, so no pointer validity is involved.
    #[target_feature(enable = "neon")]
    #[inline]
    #[cfg(not(hermes_benchmark_generic_default))]
    unsafe fn interleave(a: Self::Vector, b: Self::Vector) -> (Self::Vector, Self::Vector) {
        // zip1 = [a0, b0, a1, b1] and zip2 = [a2, b2, a3, b3] are exactly the
        // low and high halves of the flat 8-lane interleaving.
        (
            NeonF32Vec(vzip1q_f32(a.0, b.0)),
            NeonF32Vec(vzip2q_f32(a.0, b.0)),
        )
    }

    // SAFETY: caller must ensure the target CPU supports `neon` (as above); operands are whole registers, so no pointer validity is involved.
    #[target_feature(enable = "neon")]
    #[inline]
    #[cfg(not(hermes_benchmark_generic_default))]
    unsafe fn deinterleave(a: Self::Vector, b: Self::Vector) -> (Self::Vector, Self::Vector) {
        // uzp1 collects the even positions of `a || b` and uzp2 the odd ones.
        (
            NeonF32Vec(vuzp1q_f32(a.0, b.0)),
            NeonF32Vec(vuzp2q_f32(a.0, b.0)),
        )
    }

    // SAFETY: caller must ensure the target CPU supports `neon` (enforced by the `#[target_feature]` gate above plus `cfg(target_arch = "aarch64")` selection in the hermes-simd dispatcher; NEON is baseline-mandatory on AArch64); any pointer operands are valid for the 4-lane vector width within caller-validated bounds.
    #[target_feature(enable = "neon")]
    #[inline]
    unsafe fn deinterleave_pairs(a: Self::Vector, b: Self::Vector) -> (Self::Vector, Self::Vector) {
        // A lane pair is 64 bits, so the f64 unzips collect alternating pairs
        // with each pair's lanes kept adjacent.
        let a64 = vreinterpretq_f64_f32(a.0);
        let b64 = vreinterpretq_f64_f32(b.0);
        (
            NeonF32Vec(vreinterpretq_f32_f64(vuzp1q_f64(a64, b64))),
            NeonF32Vec(vreinterpretq_f32_f64(vuzp2q_f64(a64, b64))),
        )
    }

    #[inline]
    unsafe fn interleave_pairs(
        even: Self::Vector,
        odd: Self::Vector,
    ) -> (Self::Vector, Self::Vector) {
        // A lane pair is 64 bits, so the f64 zips rebuild each operand from
        // its even and odd pair with the pair's lanes kept adjacent.
        let even64 = vreinterpretq_f64_f32(even.0);
        let odd64 = vreinterpretq_f64_f32(odd.0);
        (
            NeonF32Vec(vreinterpretq_f32_f64(vzip1q_f64(even64, odd64))),
            NeonF32Vec(vreinterpretq_f32_f64(vzip2q_f64(even64, odd64))),
        )
    }

    // SAFETY: caller must ensure the target CPU supports `neon` (enforced by the `#[target_feature]` gate above plus `cfg(target_arch = "aarch64")` selection in the hermes-simd dispatcher; NEON is baseline-mandatory on AArch64); any pointer operands are valid for the 4-lane vector width within caller-validated bounds.
    #[target_feature(enable = "neon")]
    #[inline]
    #[cfg(not(hermes_benchmark_generic_default))]
    unsafe fn interleave_halves(a: Self::Vector, b: Self::Vector) -> (Self::Vector, Self::Vector) {
        // A half is a 64-bit `float32x2_t`: combine the low halves, then the high.
        (
            NeonF32Vec(vcombine_f32(vget_low_f32(a.0), vget_low_f32(b.0))),
            NeonF32Vec(vcombine_f32(vget_high_f32(a.0), vget_high_f32(b.0))),
        )
    }

    // SAFETY: caller must ensure the target CPU supports `neon` (enforced by the `#[target_feature]` gate above plus `cfg(target_arch = "aarch64")` selection in the hermes-simd dispatcher; NEON is baseline-mandatory on AArch64); any pointer operands are valid for the 4-lane vector width within caller-validated bounds.
    #[target_feature(enable = "neon")]
    #[inline]
    #[cfg(not(hermes_benchmark_generic_default))]
    unsafe fn splat_pair(lo: f32, hi: f32) -> Self::Vector {
        // The `(lo, hi)` f32 pair is one f64 lane; one `dup` fills both halves.
        let pair = f64::from_bits((u64::from(hi.to_bits()) << 32) | u64::from(lo.to_bits()));
        NeonF32Vec(vreinterpretq_f32_f64(vdupq_n_f64(pair)))
    }

    // SAFETY: caller must ensure the target CPU supports `neon` (enforced by the `#[target_feature]` gate above plus `cfg(target_arch = "aarch64")` selection in the hermes-simd dispatcher; NEON is baseline-mandatory on AArch64); any pointer operands are valid for the 4-lane vector width within caller-validated bounds.
    #[target_feature(enable = "neon")]
    #[inline]
    #[cfg(not(hermes_benchmark_generic_default))]
    unsafe fn blend_halves(a: Self::Vector, b: Self::Vector) -> Self::Vector {
        // Each half keeps its position: one combine of the two halves.
        NeonF32Vec(vcombine_f32(vget_low_f32(a.0), vget_high_f32(b.0)))
    }

    // SAFETY: caller must ensure the target CPU supports `neon` (enforced by the `#[target_feature]` gate above plus `cfg(target_arch = "aarch64")` selection in the hermes-simd dispatcher; NEON is baseline-mandatory on AArch64); any pointer operands are valid for the 4-lane vector width within caller-validated bounds.
    #[target_feature(enable = "neon")]
    #[inline]
    unsafe fn dup_even(v: Self::Vector) -> Self::Vector {
        NeonF32Vec(vtrn1q_f32(v.0, v.0))
    }

    // SAFETY: caller must ensure the target CPU supports `neon` (enforced by the `#[target_feature]` gate above plus `cfg(target_arch = "aarch64")` selection in the hermes-simd dispatcher; NEON is baseline-mandatory on AArch64); any pointer operands are valid for the 4-lane vector width within caller-validated bounds.
    #[target_feature(enable = "neon")]
    #[inline]
    unsafe fn dup_odd(v: Self::Vector) -> Self::Vector {
        NeonF32Vec(vtrn2q_f32(v.0, v.0))
    }

    // SAFETY: NEON is mandatory on aarch64 and `tile` must hold exactly four
    // rows; the safe vector wrapper enforces the length before entering here.
    #[target_feature(enable = "neon")]
    #[inline]
    #[cfg(not(hermes_benchmark_generic_default))]
    unsafe fn transpose_square(tile: &mut [Self::Vector]) {
        let tile: &mut [Self::Vector; 4] = tile
            .try_into()
            .expect("invariant: tile holds exactly LANE_COUNT rows");
        // The canonical eight-shuffle 4x4 network: `trn` weaves adjacent rows,
        // then 64-bit `zip` operations assemble the four complete columns.
        let t0 = vreinterpretq_u64_f32(vtrn1q_f32(tile[0].0, tile[1].0));
        let t1 = vreinterpretq_u64_f32(vtrn2q_f32(tile[0].0, tile[1].0));
        let t2 = vreinterpretq_u64_f32(vtrn1q_f32(tile[2].0, tile[3].0));
        let t3 = vreinterpretq_u64_f32(vtrn2q_f32(tile[2].0, tile[3].0));
        tile[0] = NeonF32Vec(vreinterpretq_f32_u64(vzip1q_u64(t0, t2)));
        tile[1] = NeonF32Vec(vreinterpretq_f32_u64(vzip1q_u64(t1, t3)));
        tile[2] = NeonF32Vec(vreinterpretq_f32_u64(vzip2q_u64(t0, t2)));
        tile[3] = NeonF32Vec(vreinterpretq_f32_u64(vzip2q_u64(t1, t3)));
    }

    /// NEON has no alternating-FMA instruction; sign-flip the even lanes of
    /// `c` (one XOR) then fuse with `vfmaq_f32`, which is rounding-identical
    /// to a native `fmaddsub`.
    // SAFETY: caller must ensure the target CPU supports `neon` (enforced by the `#[target_feature]` gate above plus `cfg(target_arch = "aarch64")` selection in the hermes-simd dispatcher; NEON is baseline-mandatory on AArch64); any pointer operands are valid for the 4-lane vector width within caller-validated bounds.
    #[target_feature(enable = "neon")]
    #[inline]
    unsafe fn fmaddsub(a: Self::Vector, b: Self::Vector, c: Self::Vector) -> Self::Vector {
        const EVEN_SIGN: [u32; 4] = [0x8000_0000, 0, 0x8000_0000, 0];
        let flipped = vreinterpretq_f32_u32(veorq_u32(
            vreinterpretq_u32_f32(c.0),
            vld1q_u32(EVEN_SIGN.as_ptr()),
        ));
        NeonF32Vec(vfmaq_f32(flipped, a.0, b.0))
    }

    /// Sign-flips the odd lanes of `c` then fuses; see [`Self::fmaddsub`].
    // SAFETY: caller must ensure the target CPU supports `neon` (enforced by the `#[target_feature]` gate above plus `cfg(target_arch = "aarch64")` selection in the hermes-simd dispatcher; NEON is baseline-mandatory on AArch64); any pointer operands are valid for the 4-lane vector width within caller-validated bounds.
    #[target_feature(enable = "neon")]
    #[inline]
    unsafe fn fmsubadd(a: Self::Vector, b: Self::Vector, c: Self::Vector) -> Self::Vector {
        const ODD_SIGN: [u32; 4] = [0, 0x8000_0000, 0, 0x8000_0000];
        let flipped = vreinterpretq_f32_u32(veorq_u32(
            vreinterpretq_u32_f32(c.0),
            vld1q_u32(ODD_SIGN.as_ptr()),
        ));
        NeonF32Vec(vfmaq_f32(flipped, a.0, b.0))
    }

    // SAFETY: caller must ensure the target CPU supports `neon` (enforced by the `#[target_feature]` gate above plus `cfg(target_arch = "aarch64")` selection in the hermes-simd dispatcher; NEON is baseline-mandatory on AArch64); any pointer operands are valid for the 4-lane vector width within caller-validated bounds.
    #[target_feature(enable = "neon")]
    #[inline]
    unsafe fn sum_reduce(v: Self::Vector) -> f32 {
        vaddvq_f32(v.0)
    }

    // SAFETY: caller must ensure the target CPU supports `neon` (enforced by the `#[target_feature]` gate above plus `cfg(target_arch = "aarch64")` selection in the hermes-simd dispatcher; NEON is baseline-mandatory on AArch64); any pointer operands are valid for the 4-lane vector width within caller-validated bounds.
    #[target_feature(enable = "neon")]
    #[inline]
    unsafe fn abs(a: Self::Vector) -> Self::Vector {
        NeonF32Vec(vabsq_f32(a.0))
    }

    // SAFETY: caller must ensure the target CPU supports `neon` (enforced by the `#[target_feature]` gate above plus `cfg(target_arch = "aarch64")` selection in the hermes-simd dispatcher; NEON is baseline-mandatory on AArch64); any pointer operands are valid for the 4-lane vector width within caller-validated bounds.
    #[target_feature(enable = "neon")]
    #[inline]
    unsafe fn min(a: Self::Vector, b: Self::Vector) -> Self::Vector {
        NeonF32Vec(vminq_f32(a.0, b.0))
    }

    // SAFETY: caller must ensure the target CPU supports `neon` (enforced by the `#[target_feature]` gate above plus `cfg(target_arch = "aarch64")` selection in the hermes-simd dispatcher; NEON is baseline-mandatory on AArch64); any pointer operands are valid for the 4-lane vector width within caller-validated bounds.
    #[target_feature(enable = "neon")]
    #[inline]
    unsafe fn max(a: Self::Vector, b: Self::Vector) -> Self::Vector {
        NeonF32Vec(vmaxq_f32(a.0, b.0))
    }

    // SAFETY: caller must ensure the target CPU supports `neon` (enforced by the `#[target_feature]` gate above plus `cfg(target_arch = "aarch64")` selection in the hermes-simd dispatcher; NEON is baseline-mandatory on AArch64); any pointer operands are valid for the 4-lane vector width within caller-validated bounds.
    #[target_feature(enable = "neon")]
    #[inline]
    unsafe fn sqrt(a: Self::Vector) -> Self::Vector {
        NeonF32Vec(vsqrtq_f32(a.0))
    }

    // SAFETY: caller must ensure the target CPU supports `neon` (enforced by the `#[target_feature]` gate above plus `cfg(target_arch = "aarch64")` selection in the hermes-simd dispatcher; NEON is baseline-mandatory on AArch64); any pointer operands are valid for the 4-lane vector width within caller-validated bounds.
    #[target_feature(enable = "neon")]
    #[inline]
    unsafe fn recip_sqrt(a: Self::Vector) -> Self::Vector {
        // `vrsqrteq_f32` seeds ~8 correct bits; each `vrsqrtsq`-refined Newton step
        // roughly doubles them. One step (~16 bits) is below f32's 23-bit mantissa,
        // so two steps (~32 bits) are required for full f32 precision (~1 ulp),
        // matching the x86 and scalar paths.
        let y0 = vrsqrteq_f32(a.0);
        let y1 = vmulq_f32(y0, vrsqrtsq_f32(a.0, vmulq_f32(y0, y0)));
        let y2 = vmulq_f32(y1, vrsqrtsq_f32(a.0, vmulq_f32(y1, y1)));
        NeonF32Vec(y2)
    }

    // SAFETY: caller must ensure the target CPU supports `neon` (enforced by the `#[target_feature]` gate above plus `cfg(target_arch = "aarch64")` selection in the hermes-simd dispatcher; NEON is baseline-mandatory on AArch64); any pointer operands are valid for the 4-lane vector width within caller-validated bounds.
    #[target_feature(enable = "neon")]
    #[inline]
    unsafe fn floor(a: Self::Vector) -> Self::Vector {
        // `FRINTM`: round toward minus infinity.
        NeonF32Vec(vrndmq_f32(a.0))
    }

    // SAFETY: caller must ensure the target CPU supports `neon` (enforced by the `#[target_feature]` gate above plus `cfg(target_arch = "aarch64")` selection in the hermes-simd dispatcher; NEON is baseline-mandatory on AArch64); any pointer operands are valid for the 4-lane vector width within caller-validated bounds.
    #[target_feature(enable = "neon")]
    #[inline]
    unsafe fn ceil(a: Self::Vector) -> Self::Vector {
        // `FRINTP`: round toward plus infinity.
        NeonF32Vec(vrndpq_f32(a.0))
    }

    // SAFETY: caller must ensure the target CPU supports `neon` (enforced by the `#[target_feature]` gate above plus `cfg(target_arch = "aarch64")` selection in the hermes-simd dispatcher; NEON is baseline-mandatory on AArch64); any pointer operands are valid for the 4-lane vector width within caller-validated bounds.
    #[target_feature(enable = "neon")]
    #[inline]
    unsafe fn round(a: Self::Vector) -> Self::Vector {
        // `FRINTN`: round to nearest, ties to even — the `round_ties_even` contract.
        NeonF32Vec(vrndnq_f32(a.0))
    }

    // SAFETY: caller must ensure the target CPU supports `neon` (enforced by the `#[target_feature]` gate above plus `cfg(target_arch = "aarch64")` selection in the hermes-simd dispatcher; NEON is baseline-mandatory on AArch64); any pointer operands are valid for the 4-lane vector width within caller-validated bounds.
    #[target_feature(enable = "neon")]
    #[inline]
    unsafe fn trunc(a: Self::Vector) -> Self::Vector {
        // `FRINTZ`: round toward zero.
        NeonF32Vec(vrndq_f32(a.0))
    }

    // SAFETY: caller must ensure the target CPU supports `neon` (enforced by the `#[target_feature]` gate above plus `cfg(target_arch = "aarch64")` selection in the hermes-simd dispatcher; NEON is baseline-mandatory on AArch64); any pointer operands are valid for the 4-lane vector width within caller-validated bounds.
    #[target_feature(enable = "neon")]
    #[inline]
    unsafe fn popcount(a: Self::Vector) -> Self::Vector {
        let v_u8 = vreinterpretq_u8_f32(a.0);
        let pop_bytes = vcntq_u8(v_u8);
        let pop_u16 = vpaddlq_u8(pop_bytes);
        let pop_u32 = vpaddlq_u16(pop_u16);
        let pop_f32 = vcvtq_f32_u32(pop_u32);
        NeonF32Vec(pop_f32)
    }

    // SAFETY: caller must ensure the target CPU supports `neon` (enforced by the `#[target_feature]` gate above plus `cfg(target_arch = "aarch64")` selection in the hermes-simd dispatcher; NEON is baseline-mandatory on AArch64); any pointer operands are valid for the 4-lane vector width within caller-validated bounds.
    #[target_feature(enable = "neon")]
    #[inline]
    unsafe fn bitand(a: Self::Vector, b: Self::Vector) -> Self::Vector {
        NeonF32Vec(vreinterpretq_f32_u32(vandq_u32(
            vreinterpretq_u32_f32(a.0),
            vreinterpretq_u32_f32(b.0),
        )))
    }

    // SAFETY: caller must ensure the target CPU supports `neon` (enforced by the `#[target_feature]` gate above plus `cfg(target_arch = "aarch64")` selection in the hermes-simd dispatcher; NEON is baseline-mandatory on AArch64); any pointer operands are valid for the 4-lane vector width within caller-validated bounds.
    #[target_feature(enable = "neon")]
    #[inline]
    unsafe fn bitor(a: Self::Vector, b: Self::Vector) -> Self::Vector {
        NeonF32Vec(vreinterpretq_f32_u32(vorrq_u32(
            vreinterpretq_u32_f32(a.0),
            vreinterpretq_u32_f32(b.0),
        )))
    }

    // SAFETY: caller must ensure the target CPU supports `neon` (enforced by the `#[target_feature]` gate above plus `cfg(target_arch = "aarch64")` selection in the hermes-simd dispatcher; NEON is baseline-mandatory on AArch64); any pointer operands are valid for the 4-lane vector width within caller-validated bounds.
    #[target_feature(enable = "neon")]
    #[inline]
    unsafe fn bitxor(a: Self::Vector, b: Self::Vector) -> Self::Vector {
        NeonF32Vec(vreinterpretq_f32_u32(veorq_u32(
            vreinterpretq_u32_f32(a.0),
            vreinterpretq_u32_f32(b.0),
        )))
    }

    // SAFETY: caller must ensure the target CPU supports `neon` (enforced by the `#[target_feature]` gate above plus `cfg(target_arch = "aarch64")` selection in the hermes-simd dispatcher; NEON is baseline-mandatory on AArch64); any pointer operands are valid for the 4-lane vector width within caller-validated bounds.
    #[target_feature(enable = "neon")]
    #[inline]
    unsafe fn cmp_eq(a: Self::Vector, b: Self::Vector) -> Self::Vector {
        NeonF32Vec(vreinterpretq_f32_u32(vceqq_f32(a.0, b.0)))
    }

    // SAFETY: caller must ensure the target CPU supports `neon` (enforced by the `#[target_feature]` gate above plus `cfg(target_arch = "aarch64")` selection in the hermes-simd dispatcher; NEON is baseline-mandatory on AArch64); any pointer operands are valid for the 4-lane vector width within caller-validated bounds.
    #[target_feature(enable = "neon")]
    #[inline]
    unsafe fn cmp_ne(a: Self::Vector, b: Self::Vector) -> Self::Vector {
        NeonF32Vec(vreinterpretq_f32_u32(vmvnq_u32(vceqq_f32(a.0, b.0))))
    }

    // SAFETY: caller must ensure the target CPU supports `neon` (enforced by the `#[target_feature]` gate above plus `cfg(target_arch = "aarch64")` selection in the hermes-simd dispatcher; NEON is baseline-mandatory on AArch64); any pointer operands are valid for the 4-lane vector width within caller-validated bounds.
    #[target_feature(enable = "neon")]
    #[inline]
    unsafe fn cmp_lt(a: Self::Vector, b: Self::Vector) -> Self::Vector {
        NeonF32Vec(vreinterpretq_f32_u32(vcltq_f32(a.0, b.0)))
    }

    // SAFETY: caller must ensure the target CPU supports `neon` (enforced by the `#[target_feature]` gate above plus `cfg(target_arch = "aarch64")` selection in the hermes-simd dispatcher; NEON is baseline-mandatory on AArch64); any pointer operands are valid for the 4-lane vector width within caller-validated bounds.
    #[target_feature(enable = "neon")]
    #[inline]
    unsafe fn cmp_le(a: Self::Vector, b: Self::Vector) -> Self::Vector {
        NeonF32Vec(vreinterpretq_f32_u32(vcleq_f32(a.0, b.0)))
    }

    // SAFETY: caller must ensure the target CPU supports `neon` (enforced by the `#[target_feature]` gate above plus `cfg(target_arch = "aarch64")` selection in the hermes-simd dispatcher; NEON is baseline-mandatory on AArch64); any pointer operands are valid for the 4-lane vector width within caller-validated bounds.
    #[target_feature(enable = "neon")]
    #[inline]
    unsafe fn cmp_gt(a: Self::Vector, b: Self::Vector) -> Self::Vector {
        NeonF32Vec(vreinterpretq_f32_u32(vcgtq_f32(a.0, b.0)))
    }

    // SAFETY: caller must ensure the target CPU supports `neon` (enforced by the `#[target_feature]` gate above plus `cfg(target_arch = "aarch64")` selection in the hermes-simd dispatcher; NEON is baseline-mandatory on AArch64); any pointer operands are valid for the 4-lane vector width within caller-validated bounds.
    #[target_feature(enable = "neon")]
    #[inline]
    unsafe fn cmp_ge(a: Self::Vector, b: Self::Vector) -> Self::Vector {
        NeonF32Vec(vreinterpretq_f32_u32(vcgeq_f32(a.0, b.0)))
    }

    // SAFETY: caller must ensure the target CPU supports `neon` (enforced by the `#[target_feature]` gate above plus `cfg(target_arch = "aarch64")` selection in the hermes-simd dispatcher; NEON is baseline-mandatory on AArch64); any pointer operands are valid for the 4-lane vector width within caller-validated bounds.
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
        // shift right by 31), so `vbsl` sees all-ones or all-zeros per lane.
        let sign = vreinterpretq_u32_s32(vshrq_n_s32::<31>(vreinterpretq_s32_f32(mask.0)));
        NeonF32Vec(vbslq_f32(sign, true_val.0, false_val.0))
    }

    // -----------------------------------------------------------------------
    // Masked load / store
    // -----------------------------------------------------------------------

    // SAFETY: caller must ensure the target CPU supports `neon` (enforced by the `#[target_feature]` gate above plus `cfg(target_arch = "aarch64")` selection in the hermes-simd dispatcher; NEON is baseline-mandatory on AArch64); any pointer operands are valid for the 4-lane vector width within caller-validated bounds.
    #[target_feature(enable = "neon")]
    #[inline]
    unsafe fn masked_load_unaligned(
        ptr: *const f32,
        mask: Self::Mask,
        src: Self::Vector,
    ) -> Self::Vector {
        // vbslq: for each bit, mask=1 → first arg, mask=0 → second arg.
        NeonF32Vec(vbslq_f32(mask.0, vld1q_f32(ptr), src.0))
    }

    // SAFETY: caller must ensure the target CPU supports `neon` (enforced by the `#[target_feature]` gate above plus `cfg(target_arch = "aarch64")` selection in the hermes-simd dispatcher; NEON is baseline-mandatory on AArch64); any pointer operands are valid for the 4-lane vector width within caller-validated bounds.
    #[target_feature(enable = "neon")]
    #[inline]
    unsafe fn masked_store_unaligned(ptr: *mut f32, mask: Self::Mask, val: Self::Vector) {
        // Blend val into current memory contents and store.
        let current = vld1q_f32(ptr);
        vst1q_f32(ptr, vbslq_f32(mask.0, val.0, current));
    }

    // -----------------------------------------------------------------------
    // Masked arithmetic
    // -----------------------------------------------------------------------

    // SAFETY: caller must ensure the target CPU supports `neon` (enforced by the `#[target_feature]` gate above plus `cfg(target_arch = "aarch64")` selection in the hermes-simd dispatcher; NEON is baseline-mandatory on AArch64); any pointer operands are valid for the 4-lane vector width within caller-validated bounds.
    #[target_feature(enable = "neon")]
    #[inline]
    unsafe fn masked_add(
        a: Self::Vector,
        b: Self::Vector,
        mask: Self::Mask,
        src: Self::Vector,
    ) -> Self::Vector {
        NeonF32Vec(vbslq_f32(mask.0, vaddq_f32(a.0, b.0), src.0))
    }

    // SAFETY: caller must ensure the target CPU supports `neon` (enforced by the `#[target_feature]` gate above plus `cfg(target_arch = "aarch64")` selection in the hermes-simd dispatcher; NEON is baseline-mandatory on AArch64); any pointer operands are valid for the 4-lane vector width within caller-validated bounds.
    #[target_feature(enable = "neon")]
    #[inline]
    unsafe fn masked_mul(
        a: Self::Vector,
        b: Self::Vector,
        mask: Self::Mask,
        src: Self::Vector,
    ) -> Self::Vector {
        NeonF32Vec(vbslq_f32(mask.0, vmulq_f32(a.0, b.0), src.0))
    }

    // SAFETY: caller must ensure the target CPU supports `neon` (enforced by the `#[target_feature]` gate above plus `cfg(target_arch = "aarch64")` selection in the hermes-simd dispatcher; NEON is baseline-mandatory on AArch64); any pointer operands are valid for the 4-lane vector width within caller-validated bounds.
    #[target_feature(enable = "neon")]
    #[inline]
    unsafe fn masked_fmadd(
        a: Self::Vector,
        b: Self::Vector,
        c: Self::Vector,
        mask: Self::Mask,
    ) -> Self::Vector {
        // Inactive lanes retain c (addend pass-through).
        NeonF32Vec(vbslq_f32(mask.0, vfmaq_f32(c.0, a.0, b.0), c.0))
    }

    // SAFETY: caller must ensure the target CPU supports `neon` (enforced by the `#[target_feature]` gate above plus `cfg(target_arch = "aarch64")` selection in the hermes-simd dispatcher; NEON is baseline-mandatory on AArch64); any pointer operands are valid for the 4-lane vector width within caller-validated bounds.
    #[target_feature(enable = "neon")]
    #[inline]
    unsafe fn masked_sum_reduce(v: Self::Vector, mask: Self::Mask) -> f32 {
        vaddvq_f32(vbslq_f32(mask.0, v.0, vdupq_n_f32(0.0)))
    }

    // -----------------------------------------------------------------------
    // Compress / Expand (emulated)
    // -----------------------------------------------------------------------

    // SAFETY: caller must ensure the target CPU supports `neon` (enforced by the `#[target_feature]` gate above plus `cfg(target_arch = "aarch64")` selection in the hermes-simd dispatcher; NEON is baseline-mandatory on AArch64); any pointer operands are valid for the 4-lane vector width within caller-validated bounds.
    #[target_feature(enable = "neon")]
    #[inline]
    unsafe fn compress(src: Self::Vector, mask: Self::Mask) -> Self::Vector {
        let mut arr = [0.0f32; 4];
        vst1q_f32(arr.as_mut_ptr(), src.0);
        let m = lane_actives(mask.0);
        let mut out = [0.0f32; 4];
        let mut k = 0usize;
        for i in 0..4 {
            if m[i] {
                out[k] = arr[i];
                k += 1;
            }
        }
        NeonF32Vec(vld1q_f32(out.as_ptr()))
    }

    // SAFETY: caller must ensure the target CPU supports `neon` (enforced by the `#[target_feature]` gate above plus `cfg(target_arch = "aarch64")` selection in the hermes-simd dispatcher; NEON is baseline-mandatory on AArch64); any pointer operands are valid for the 4-lane vector width within caller-validated bounds.
    #[target_feature(enable = "neon")]
    #[inline]
    unsafe fn expand(src: Self::Vector, mask: Self::Mask, fill: Self::Vector) -> Self::Vector {
        let mut src_arr = [0.0f32; 4];
        vst1q_f32(src_arr.as_mut_ptr(), src.0);
        let mut out_arr = [0.0f32; 4];
        vst1q_f32(out_arr.as_mut_ptr(), fill.0);
        let m = lane_actives(mask.0);
        let mut k = 0usize;
        for i in 0..4 {
            if m[i] {
                out_arr[i] = src_arr[k];
                k += 1;
            }
        }
        NeonF32Vec(vld1q_f32(out_arr.as_ptr()))
    }

    // -----------------------------------------------------------------------
    // Gather (emulated — NEON has no native scatter/gather)
    // -----------------------------------------------------------------------

    // SAFETY: caller must ensure the target CPU supports `neon` (enforced by the `#[target_feature]` gate above plus `cfg(target_arch = "aarch64")` selection in the hermes-simd dispatcher; NEON is baseline-mandatory on AArch64); any pointer operands are valid for the 4-lane vector width within caller-validated bounds.
    #[target_feature(enable = "neon")]
    #[inline]
    unsafe fn gather(base: *const f32, indices: Self::IndexVector) -> Self::Vector {
        let arr = [
            *base.add(indices[0] as usize),
            *base.add(indices[1] as usize),
            *base.add(indices[2] as usize),
            *base.add(indices[3] as usize),
        ];
        NeonF32Vec(vld1q_f32(arr.as_ptr()))
    }

    // SAFETY: caller must ensure the target CPU supports `neon` (enforced by the `#[target_feature]` gate above plus `cfg(target_arch = "aarch64")` selection in the hermes-simd dispatcher; NEON is baseline-mandatory on AArch64); any pointer operands are valid for the 4-lane vector width within caller-validated bounds.
    #[target_feature(enable = "neon")]
    #[inline]
    unsafe fn gather_masked(
        base: *const f32,
        indices: Self::IndexVector,
        mask: Self::Mask,
        src: Self::Vector,
    ) -> Self::Vector {
        let m = lane_actives(mask.0);
        let mut src_arr = [0.0f32; 4];
        vst1q_f32(src_arr.as_mut_ptr(), src.0);
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
            if m[2] {
                *base.add(indices[2] as usize)
            } else {
                src_arr[2]
            },
            if m[3] {
                *base.add(indices[3] as usize)
            } else {
                src_arr[3]
            },
        ];
        NeonF32Vec(vld1q_f32(out.as_ptr()))
    }

    // -----------------------------------------------------------------------
    // Mask construction
    // -----------------------------------------------------------------------

    // SAFETY: caller must ensure the target CPU supports `neon` (enforced by the `#[target_feature]` gate above plus `cfg(target_arch = "aarch64")` selection in the hermes-simd dispatcher; NEON is baseline-mandatory on AArch64); any pointer operands are valid for the 4-lane vector width within caller-validated bounds.
    #[target_feature(enable = "neon")]
    #[inline]
    unsafe fn mask_from_bools(bits: &[bool]) -> Self::Mask {
        debug_assert_eq!(bits.len(), 4);
        let vals: [u32; 4] = core::array::from_fn(|i| if bits[i] { 0xFFFF_FFFFu32 } else { 0u32 });
        NeonF32Mask(vld1q_u32(vals.as_ptr()))
    }

    // SAFETY: caller must ensure the target CPU supports `neon` (enforced by the `#[target_feature]` gate above plus `cfg(target_arch = "aarch64")` selection in the hermes-simd dispatcher; NEON is baseline-mandatory on AArch64); any pointer operands are valid for the 4-lane vector width within caller-validated bounds.
    #[target_feature(enable = "neon")]
    #[inline]
    unsafe fn leading_k_mask(k: usize) -> Self::Mask {
        let k = k.min(4);
        let vals: [u32; 4] = core::array::from_fn(|i| if i < k { 0xFFFF_FFFFu32 } else { 0u32 });
        NeonF32Mask(vld1q_u32(vals.as_ptr()))
    }

    // SAFETY: caller must ensure the target CPU supports `neon` (enforced by the `#[target_feature]` gate above plus `cfg(target_arch = "aarch64")` selection in the hermes-simd dispatcher; NEON is baseline-mandatory on AArch64); the only memory operand is the constant lane-bit table.
    #[target_feature(enable = "neon")]
    #[inline]
    unsafe fn mask_from_bitmask(bm: u64) -> Self::Mask {
        // `vtst` sets a lane to all-ones where `(bits & lane_bit) != 0` —
        // canonical expansion in one test instruction, replacing the generic
        // bool-array bounce (bits 4.. are ignored because only bits 0..4
        // appear in the table).
        let lane_bits: [u32; 4] = [1, 2, 4, 8];
        let bits = vdupq_n_u32(bm as u32);
        NeonF32Mask(vtstq_u32(bits, vld1q_u32(lane_bits.as_ptr())))
    }

    // -----------------------------------------------------------------------
    // Broadcast / zero
    // -----------------------------------------------------------------------

    // SAFETY: caller must ensure the target CPU supports `neon` (enforced by the `#[target_feature]` gate above plus `cfg(target_arch = "aarch64")` selection in the hermes-simd dispatcher; NEON is baseline-mandatory on AArch64); any pointer operands are valid for the 4-lane vector width within caller-validated bounds.
    #[target_feature(enable = "neon")]
    #[inline]
    unsafe fn zero() -> Self::Vector {
        NeonF32Vec(vdupq_n_f32(0.0))
    }

    // SAFETY: caller must ensure the target CPU supports `neon` (enforced by the `#[target_feature]` gate above plus `cfg(target_arch = "aarch64")` selection in the hermes-simd dispatcher; NEON is baseline-mandatory on AArch64); any pointer operands are valid for the 4-lane vector width within caller-validated bounds.
    #[target_feature(enable = "neon")]
    #[inline]
    unsafe fn splat(val: f32) -> Self::Vector {
        NeonF32Vec(vdupq_n_f32(val))
    }

    // SAFETY: caller must ensure the target CPU supports `neon` (enforced by the `#[target_feature]` gate above plus `cfg(target_arch = "aarch64")` selection in the hermes-simd dispatcher; NEON is baseline-mandatory on AArch64); any pointer operands are valid for the 4-lane vector width within caller-validated bounds.
    #[target_feature(enable = "neon")]
    #[inline]
    unsafe fn mask_to_bitmask(mask: Self::Mask) -> u64 {
        let mut m = 0u64;
        m |= ((vgetq_lane_u32::<0>(mask.0) >> 31) as u64) << 0;
        m |= ((vgetq_lane_u32::<1>(mask.0) >> 31) as u64) << 1;
        m |= ((vgetq_lane_u32::<2>(mask.0) >> 31) as u64) << 2;
        m |= ((vgetq_lane_u32::<3>(mask.0) >> 31) as u64) << 3;
        m
    }

    // SAFETY: caller must ensure the target CPU supports `neon` (enforced by the `#[target_feature]` gate above plus `cfg(target_arch = "aarch64")` selection in the hermes-simd dispatcher; NEON is baseline-mandatory on AArch64); any pointer operands are valid for the 4-lane vector width within caller-validated bounds.
    #[target_feature(enable = "neon")]
    #[inline]
    unsafe fn mask_to_vector(mask: Self::Mask) -> Self::Vector {
        NeonF32Vec(vreinterpretq_f32_u32(mask.0))
    }

    // SAFETY: caller must ensure the target CPU supports `neon` (enforced by the `#[target_feature]` gate above plus runtime selection in the hermes-simd dispatcher); this is a register-to-register reinterpretation with no memory operands.
    #[target_feature(enable = "neon")]
    #[inline]
    unsafe fn vector_to_mask(v: Self::Vector) -> Self::Mask {
        // Canonicalize on entry: broadcast each lane's sign bit (the documented
        // active criterion) across the lane with an arithmetic shift right by
        // 31, so every `Mask` consumer — the bitwise `vbsl` merges included —
        // sees all-ones or all-zeros per lane regardless of the mask vector's
        // remaining bits. A bare reinterpretation kept non-canonical bits and
        // let `vbsl`-based masked ops splice operands bit-by-bit.
        NeonF32Mask(vreinterpretq_u32_s32(vshrq_n_s32::<31>(
            vreinterpretq_s32_f32(v.0),
        )))
    }
}
