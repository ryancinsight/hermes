//! AArch64 NEON f64 hardware kernel.
//!
//! 2-lane f64 (`float64x2_t`). Masked ops via `vbslq_f64`.
//! Gather is emulated via individual lane loads — NEON has no native gather.
//! Compress/expand are emulated via scalar loops.

use crate::Neon;
#[cfg(target_arch = "aarch64")]
use core::arch::aarch64::*;
use hermes_simd_core::kernel::SimdKernel;

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
/// Lane `i` is active when `mask[i] == 0xFFFF_FFFF_FFFF_FFFF`.
#[cfg(target_arch = "aarch64")]
#[repr(transparent)]
#[derive(Copy, Clone)]
pub struct NeonF64Mask(pub uint64x2_t);

#[cfg(target_arch = "aarch64")]
unsafe impl Send for NeonF64Mask {}
#[cfg(target_arch = "aarch64")]
unsafe impl Sync for NeonF64Mask {}

#[cfg(target_arch = "aarch64")]
impl SimdKernel<f64> for Neon {
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

    /// `vextq_f64(v, v, 1)` rotates the two lanes: `[a0, a1] -> [a1, a0]`.
    // SAFETY: caller must ensure the target CPU supports `neon` (enforced by the `#[target_feature]` gate above plus `cfg(target_arch = "aarch64")` selection in the hermes-simd dispatcher; NEON is baseline-mandatory on AArch64); any pointer operands are valid for the 2-lane vector width within caller-validated bounds.
    #[target_feature(enable = "neon")]
    #[inline]
    unsafe fn swap_adjacent(v: Self::Vector) -> Self::Vector {
        NeonF64Vec(vextq_f64(v.0, v.0, 1))
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
        NeonF64Vec(vbslq_f64(
            vreinterpretq_u64_f64(mask.0),
            true_val.0,
            false_val.0,
        ))
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
        let m = [
            vgetq_lane_u64(mask.0, 0) != 0,
            vgetq_lane_u64(mask.0, 1) != 0,
        ];
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
        let m = [
            vgetq_lane_u64(mask.0, 0) != 0,
            vgetq_lane_u64(mask.0, 1) != 0,
        ];
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
        let m = [
            vgetq_lane_u64(mask.0, 0) != 0,
            vgetq_lane_u64(mask.0, 1) != 0,
        ];
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
}
