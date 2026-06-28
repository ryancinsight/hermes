//! AArch64 NEON f32 hardware kernel.
//!
//! 4-lane f32 (`float32x4_t`). Masked ops use `vbslq_f32` (bitwise select).
//! Gather is emulated via four individual lane loads — NEON has no native gather.
//! Compress/expand are emulated via scalar loops.

use crate::Neon;
#[cfg(target_arch = "aarch64")]
use core::arch::aarch64::*;
use hermes_simd_core::kernel::SimdKernel;

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
/// Lane `i` is active when `mask[i] == 0xFFFF_FFFF`. This matches the
/// `vbslq_f32` convention (true lane selects from the first argument).
#[cfg(target_arch = "aarch64")]
#[repr(transparent)]
#[derive(Copy, Clone)]
pub struct NeonF32Mask(pub uint32x4_t);

#[cfg(target_arch = "aarch64")]
unsafe impl Send for NeonF32Mask {}
#[cfg(target_arch = "aarch64")]
unsafe impl Sync for NeonF32Mask {}

#[cfg(target_arch = "aarch64")]
impl SimdKernel<f32> for Neon {
    type Vector = NeonF32Vec;
    type Mask = NeonF32Mask;
    /// Scalar index array — NEON has no native gather; emulated lane-by-lane.
    type IndexVector = [i32; 4];
    const LANE_COUNT: usize = 4;
    const UNROLL_FACTOR: usize = 4;

    // -----------------------------------------------------------------------
    // Load / store
    // -----------------------------------------------------------------------

    #[target_feature(enable = "neon")]
    #[inline]
    unsafe fn load_aligned(ptr: *const f32) -> Self::Vector {
        NeonF32Vec(vld1q_f32(ptr))
    }

    #[target_feature(enable = "neon")]
    #[inline]
    unsafe fn load_unaligned(ptr: *const f32) -> Self::Vector {
        NeonF32Vec(vld1q_f32(ptr))
    }

    #[target_feature(enable = "neon")]
    #[inline]
    unsafe fn store_aligned(ptr: *mut f32, val: Self::Vector) {
        vst1q_f32(ptr, val.0);
    }

    #[target_feature(enable = "neon")]
    #[inline]
    unsafe fn store_unaligned(ptr: *mut f32, val: Self::Vector) {
        vst1q_f32(ptr, val.0);
    }

    // -----------------------------------------------------------------------
    // Arithmetic
    // -----------------------------------------------------------------------

    #[target_feature(enable = "neon")]
    #[inline]
    unsafe fn add(a: Self::Vector, b: Self::Vector) -> Self::Vector {
        NeonF32Vec(vaddq_f32(a.0, b.0))
    }

    #[target_feature(enable = "neon")]
    #[inline]
    unsafe fn mul(a: Self::Vector, b: Self::Vector) -> Self::Vector {
        NeonF32Vec(vmulq_f32(a.0, b.0))
    }

    #[target_feature(enable = "neon")]
    #[inline]
    unsafe fn sub(a: Self::Vector, b: Self::Vector) -> Self::Vector {
        NeonF32Vec(vsubq_f32(a.0, b.0))
    }

    #[target_feature(enable = "neon")]
    #[inline]
    unsafe fn div(a: Self::Vector, b: Self::Vector) -> Self::Vector {
        NeonF32Vec(vdivq_f32(a.0, b.0))
    }

    /// NEON FMA: `vfmaq_f32(c, a, b)` computes `a*b + c`.
    #[target_feature(enable = "neon")]
    #[inline]
    unsafe fn fmadd(a: Self::Vector, b: Self::Vector, c: Self::Vector) -> Self::Vector {
        NeonF32Vec(vfmaq_f32(c.0, a.0, b.0))
    }

    /// `vrev64q_f32` swaps 32-bit lanes within each 64-bit doubleword.
    #[target_feature(enable = "neon")]
    #[inline]
    unsafe fn swap_adjacent(v: Self::Vector) -> Self::Vector {
        NeonF32Vec(vrev64q_f32(v.0))
    }

    #[target_feature(enable = "neon")]
    #[inline]
    unsafe fn dup_even(v: Self::Vector) -> Self::Vector {
        NeonF32Vec(vtrn1q_f32(v.0, v.0))
    }

    #[target_feature(enable = "neon")]
    #[inline]
    unsafe fn dup_odd(v: Self::Vector) -> Self::Vector {
        NeonF32Vec(vtrn2q_f32(v.0, v.0))
    }

    /// NEON has no alternating-FMA instruction; sign-flip the even lanes of
    /// `c` (one XOR) then fuse with `vfmaq_f32`, which is rounding-identical
    /// to a native `fmaddsub`.
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

    #[target_feature(enable = "neon")]
    #[inline]
    unsafe fn sum_reduce(v: Self::Vector) -> f32 {
        vaddvq_f32(v.0)
    }

    #[target_feature(enable = "neon")]
    #[inline]
    unsafe fn abs(a: Self::Vector) -> Self::Vector {
        NeonF32Vec(vabsq_f32(a.0))
    }

    #[target_feature(enable = "neon")]
    #[inline]
    unsafe fn min(a: Self::Vector, b: Self::Vector) -> Self::Vector {
        NeonF32Vec(vminq_f32(a.0, b.0))
    }

    #[target_feature(enable = "neon")]
    #[inline]
    unsafe fn max(a: Self::Vector, b: Self::Vector) -> Self::Vector {
        NeonF32Vec(vmaxq_f32(a.0, b.0))
    }

    #[target_feature(enable = "neon")]
    #[inline]
    unsafe fn sqrt(a: Self::Vector) -> Self::Vector {
        NeonF32Vec(vsqrtq_f32(a.0))
    }

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

    #[target_feature(enable = "neon")]
    #[inline]
    unsafe fn bitand(a: Self::Vector, b: Self::Vector) -> Self::Vector {
        NeonF32Vec(vreinterpretq_f32_u32(vandq_u32(
            vreinterpretq_u32_f32(a.0),
            vreinterpretq_u32_f32(b.0),
        )))
    }

    #[target_feature(enable = "neon")]
    #[inline]
    unsafe fn bitor(a: Self::Vector, b: Self::Vector) -> Self::Vector {
        NeonF32Vec(vreinterpretq_f32_u32(vorrq_u32(
            vreinterpretq_u32_f32(a.0),
            vreinterpretq_u32_f32(b.0),
        )))
    }

    #[target_feature(enable = "neon")]
    #[inline]
    unsafe fn bitxor(a: Self::Vector, b: Self::Vector) -> Self::Vector {
        NeonF32Vec(vreinterpretq_f32_u32(veorq_u32(
            vreinterpretq_u32_f32(a.0),
            vreinterpretq_u32_f32(b.0),
        )))
    }

    #[target_feature(enable = "neon")]
    #[inline]
    unsafe fn cmp_eq(a: Self::Vector, b: Self::Vector) -> Self::Vector {
        NeonF32Vec(vreinterpretq_f32_u32(vceqq_f32(a.0, b.0)))
    }

    #[target_feature(enable = "neon")]
    #[inline]
    unsafe fn cmp_ne(a: Self::Vector, b: Self::Vector) -> Self::Vector {
        NeonF32Vec(vreinterpretq_f32_u32(vmvnq_u32(vceqq_f32(a.0, b.0))))
    }

    #[target_feature(enable = "neon")]
    #[inline]
    unsafe fn cmp_lt(a: Self::Vector, b: Self::Vector) -> Self::Vector {
        NeonF32Vec(vreinterpretq_f32_u32(vcltq_f32(a.0, b.0)))
    }

    #[target_feature(enable = "neon")]
    #[inline]
    unsafe fn cmp_le(a: Self::Vector, b: Self::Vector) -> Self::Vector {
        NeonF32Vec(vreinterpretq_f32_u32(vcleq_f32(a.0, b.0)))
    }

    #[target_feature(enable = "neon")]
    #[inline]
    unsafe fn cmp_gt(a: Self::Vector, b: Self::Vector) -> Self::Vector {
        NeonF32Vec(vreinterpretq_f32_u32(vcgtq_f32(a.0, b.0)))
    }

    #[target_feature(enable = "neon")]
    #[inline]
    unsafe fn cmp_ge(a: Self::Vector, b: Self::Vector) -> Self::Vector {
        NeonF32Vec(vreinterpretq_f32_u32(vcgeq_f32(a.0, b.0)))
    }

    #[target_feature(enable = "neon")]
    #[inline]
    unsafe fn blend(
        mask: Self::Vector,
        true_val: Self::Vector,
        false_val: Self::Vector,
    ) -> Self::Vector {
        NeonF32Vec(vbslq_f32(
            vreinterpretq_u32_f32(mask.0),
            true_val.0,
            false_val.0,
        ))
    }

    // -----------------------------------------------------------------------
    // Masked load / store
    // -----------------------------------------------------------------------

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

    #[target_feature(enable = "neon")]
    #[inline]
    unsafe fn masked_sum_reduce(v: Self::Vector, mask: Self::Mask) -> f32 {
        vaddvq_f32(vbslq_f32(mask.0, v.0, vdupq_n_f32(0.0)))
    }

    // -----------------------------------------------------------------------
    // Compress / Expand (emulated)
    // -----------------------------------------------------------------------

    #[target_feature(enable = "neon")]
    #[inline]
    unsafe fn compress(src: Self::Vector, mask: Self::Mask) -> Self::Vector {
        let mut arr = [0.0f32; 4];
        vst1q_f32(arr.as_mut_ptr(), src.0);
        let m = [
            vgetq_lane_u32(mask.0, 0) != 0,
            vgetq_lane_u32(mask.0, 1) != 0,
            vgetq_lane_u32(mask.0, 2) != 0,
            vgetq_lane_u32(mask.0, 3) != 0,
        ];
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

    #[target_feature(enable = "neon")]
    #[inline]
    unsafe fn expand(src: Self::Vector, mask: Self::Mask, fill: Self::Vector) -> Self::Vector {
        let mut src_arr = [0.0f32; 4];
        vst1q_f32(src_arr.as_mut_ptr(), src.0);
        let mut out_arr = [0.0f32; 4];
        vst1q_f32(out_arr.as_mut_ptr(), fill.0);
        let m = [
            vgetq_lane_u32(mask.0, 0) != 0,
            vgetq_lane_u32(mask.0, 1) != 0,
            vgetq_lane_u32(mask.0, 2) != 0,
            vgetq_lane_u32(mask.0, 3) != 0,
        ];
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

    #[target_feature(enable = "neon")]
    #[inline]
    unsafe fn gather_masked(
        base: *const f32,
        indices: Self::IndexVector,
        mask: Self::Mask,
        src: Self::Vector,
    ) -> Self::Vector {
        let m = [
            vgetq_lane_u32(mask.0, 0) != 0,
            vgetq_lane_u32(mask.0, 1) != 0,
            vgetq_lane_u32(mask.0, 2) != 0,
            vgetq_lane_u32(mask.0, 3) != 0,
        ];
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

    #[target_feature(enable = "neon")]
    #[inline]
    unsafe fn mask_from_bools(bits: &[bool]) -> Self::Mask {
        debug_assert_eq!(bits.len(), 4);
        let vals: [u32; 4] = core::array::from_fn(|i| if bits[i] { 0xFFFF_FFFFu32 } else { 0u32 });
        NeonF32Mask(vld1q_u32(vals.as_ptr()))
    }

    #[target_feature(enable = "neon")]
    #[inline]
    unsafe fn leading_k_mask(k: usize) -> Self::Mask {
        let k = k.min(4);
        let vals: [u32; 4] = core::array::from_fn(|i| if i < k { 0xFFFF_FFFFu32 } else { 0u32 });
        NeonF32Mask(vld1q_u32(vals.as_ptr()))
    }

    // -----------------------------------------------------------------------
    // Broadcast / zero
    // -----------------------------------------------------------------------

    #[target_feature(enable = "neon")]
    #[inline]
    unsafe fn zero() -> Self::Vector {
        NeonF32Vec(vdupq_n_f32(0.0))
    }

    #[target_feature(enable = "neon")]
    #[inline]
    unsafe fn splat(val: f32) -> Self::Vector {
        NeonF32Vec(vdupq_n_f32(val))
    }

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

    #[target_feature(enable = "neon")]
    #[inline]
    unsafe fn mask_to_vector(mask: Self::Mask) -> Self::Vector {
        NeonF32Vec(vreinterpretq_f32_u32(mask.0))
    }
}
