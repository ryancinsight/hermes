//! AVX-512F f64 hardware kernel.
//!
//! 8-lane f64 (`__m512d`). Native AVX-512 predicated arithmetic.
//! Mask type is `__mmask8` (8-bit integer). IndexVector is `__m256i` (8 × i32).
//!
//! Native operations used:
//! - Masked load/store: `_mm512_mask_loadu_pd`, `_mm512_mask_storeu_pd`.
//! - Masked arithmetic: `_mm512_mask_add_pd`, `_mm512_mask_mul_pd`.
//! - Masked FMA: `_mm512_mask3_fmadd_pd` (inactive lanes retain `c`).
//! - Compress: `_mm512_maskz_compress_pd`.
//! - Expand: `_mm512_mask_expand_pd`.
//! - Gather: `_mm512_i32gather_pd`, `_mm512_mask_i32gather_pd`.

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
use core::arch::x86_64::*;
use hermes_simd_core::kernel::SimdKernel;
use crate::Avx512;

/// Newtype over `__m512d` providing `Send + Sync`.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[repr(transparent)]
#[derive(Copy, Clone)]
pub struct Avx512F64Vec(pub __m512d);

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
unsafe impl Send for Avx512F64Vec {}
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
unsafe impl Sync for Avx512F64Vec {}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
impl SimdKernel<f64> for Avx512 {
    type Vector = Avx512F64Vec;
    /// Native AVX-512 8-bit mask register. Bit `i` set → lane `i` active.
    type Mask = __mmask8;
    /// 8 × i32 index vector for gather (`__m256i`).
    type IndexVector = __m256i;
    const LANE_COUNT: usize = 8;
    const UNROLL_FACTOR: usize = 4;

    // -----------------------------------------------------------------------
    // Load / store
    // -----------------------------------------------------------------------

    #[target_feature(enable = "avx512f")]
    #[inline]
    unsafe fn load_aligned(ptr: *const f64) -> Self::Vector {
        Avx512F64Vec(_mm512_load_pd(ptr))
    }

    #[target_feature(enable = "avx512f")]
    #[inline]
    unsafe fn load_unaligned(ptr: *const f64) -> Self::Vector {
        Avx512F64Vec(_mm512_loadu_pd(ptr))
    }

    #[target_feature(enable = "avx512f")]
    #[inline]
    unsafe fn store_aligned(ptr: *mut f64, val: Self::Vector) {
        _mm512_store_pd(ptr, val.0);
    }

    #[target_feature(enable = "avx512f")]
    #[inline]
    unsafe fn store_unaligned(ptr: *mut f64, val: Self::Vector) {
        _mm512_storeu_pd(ptr, val.0);
    }

    // -----------------------------------------------------------------------
    // Arithmetic
    // -----------------------------------------------------------------------

    #[target_feature(enable = "avx512f")]
    #[inline]
    unsafe fn add(a: Self::Vector, b: Self::Vector) -> Self::Vector {
        Avx512F64Vec(_mm512_add_pd(a.0, b.0))
    }

    #[target_feature(enable = "avx512f")]
    #[inline]
    unsafe fn mul(a: Self::Vector, b: Self::Vector) -> Self::Vector {
        Avx512F64Vec(_mm512_mul_pd(a.0, b.0))
    }

    #[target_feature(enable = "avx512f")]
    #[inline]
    unsafe fn fmadd(a: Self::Vector, b: Self::Vector, c: Self::Vector) -> Self::Vector {
        Avx512F64Vec(_mm512_fmadd_pd(a.0, b.0, c.0))
    }

    #[target_feature(enable = "avx512f")]
    #[inline]
    unsafe fn sum_reduce(v: Self::Vector) -> f64 {
        _mm512_reduce_add_pd(v.0)
    }

    // -----------------------------------------------------------------------
    // Native masked load / store
    // -----------------------------------------------------------------------

    #[target_feature(enable = "avx512f")]
    #[inline]
    unsafe fn masked_load_unaligned(
        ptr: *const f64,
        mask: Self::Mask,
        src: Self::Vector,
    ) -> Self::Vector {
        Avx512F64Vec(_mm512_mask_loadu_pd(src.0, mask, ptr))
    }

    #[target_feature(enable = "avx512f")]
    #[inline]
    unsafe fn masked_store_unaligned(ptr: *mut f64, mask: Self::Mask, val: Self::Vector) {
        _mm512_mask_storeu_pd(ptr, mask, val.0);
    }

    // -----------------------------------------------------------------------
    // Native masked arithmetic
    // -----------------------------------------------------------------------

    #[target_feature(enable = "avx512f")]
    #[inline]
    unsafe fn masked_add(
        a: Self::Vector,
        b: Self::Vector,
        mask: Self::Mask,
        src: Self::Vector,
    ) -> Self::Vector {
        Avx512F64Vec(_mm512_mask_add_pd(src.0, mask, a.0, b.0))
    }

    #[target_feature(enable = "avx512f")]
    #[inline]
    unsafe fn masked_mul(
        a: Self::Vector,
        b: Self::Vector,
        mask: Self::Mask,
        src: Self::Vector,
    ) -> Self::Vector {
        Avx512F64Vec(_mm512_mask_mul_pd(src.0, mask, a.0, b.0))
    }

    /// `mask3_fmadd`: inactive lanes retain `c` (the addend / 3rd operand).
    #[target_feature(enable = "avx512f")]
    #[inline]
    unsafe fn masked_fmadd(
        a: Self::Vector,
        b: Self::Vector,
        c: Self::Vector,
        mask: Self::Mask,
    ) -> Self::Vector {
        Avx512F64Vec(_mm512_mask3_fmadd_pd(a.0, b.0, c.0, mask))
    }

    #[target_feature(enable = "avx512f")]
    #[inline]
    unsafe fn masked_sum_reduce(v: Self::Vector, mask: Self::Mask) -> f64 {
        let zero = _mm512_setzero_pd();
        _mm512_reduce_add_pd(_mm512_mask_mov_pd(zero, mask, v.0))
    }

    // -----------------------------------------------------------------------
    // Native compress / expand
    // -----------------------------------------------------------------------

    #[target_feature(enable = "avx512f")]
    #[inline]
    unsafe fn compress(src: Self::Vector, mask: Self::Mask) -> Self::Vector {
        Avx512F64Vec(_mm512_maskz_compress_pd(mask, src.0))
    }

    #[target_feature(enable = "avx512f")]
    #[inline]
    unsafe fn expand(src: Self::Vector, mask: Self::Mask, fill: Self::Vector) -> Self::Vector {
        Avx512F64Vec(_mm512_mask_expand_pd(fill.0, mask, src.0))
    }

    // -----------------------------------------------------------------------
    // Native gather (scale = 8 — byte stride of f64)
    // -----------------------------------------------------------------------

    #[target_feature(enable = "avx512f")]
    #[inline]
    unsafe fn gather(base: *const f64, indices: Self::IndexVector) -> Self::Vector {
        Avx512F64Vec(_mm512_i32gather_pd(indices, base as *const _, 8))
    }

    #[target_feature(enable = "avx512f")]
    #[inline]
    unsafe fn gather_masked(
        base: *const f64,
        indices: Self::IndexVector,
        mask: Self::Mask,
        src: Self::Vector,
    ) -> Self::Vector {
        Avx512F64Vec(_mm512_mask_i32gather_pd(src.0, mask, indices, base as *const _, 8))
    }

    // -----------------------------------------------------------------------
    // Mask construction
    // -----------------------------------------------------------------------

    #[target_feature(enable = "avx512f")]
    #[inline]
    unsafe fn mask_from_bools(bits: &[bool]) -> Self::Mask {
        debug_assert_eq!(bits.len(), 8);
        let mut m: u8 = 0;
        for (i, &b) in bits.iter().enumerate() {
            if b { m |= 1 << i; }
        }
        m
    }

    #[target_feature(enable = "avx512f")]
    #[inline]
    unsafe fn leading_k_mask(k: usize) -> Self::Mask {
        let k = k.min(8);
        if k == 8 { !0u8 } else { (1u8 << k).wrapping_sub(1) }
    }

    // -----------------------------------------------------------------------
    // Broadcast / zero
    // -----------------------------------------------------------------------

    #[target_feature(enable = "avx512f")]
    #[inline]
    unsafe fn zero() -> Self::Vector { Avx512F64Vec(_mm512_setzero_pd()) }

    #[target_feature(enable = "avx512f")]
    #[inline]
    unsafe fn splat(val: f64) -> Self::Vector { Avx512F64Vec(_mm512_set1_pd(val)) }

    #[target_feature(enable = "avx512f")]
    #[inline]
    unsafe fn div(a: Self::Vector, b: Self::Vector) -> Self::Vector {
        Avx512F64Vec(_mm512_div_pd(a.0, b.0))
    }

    #[target_feature(enable = "avx512f")]
    #[inline]
    unsafe fn bitand(a: Self::Vector, b: Self::Vector) -> Self::Vector {
        Avx512F64Vec(_mm512_and_pd(a.0, b.0))
    }

    #[target_feature(enable = "avx512f")]
    #[inline]
    unsafe fn bitor(a: Self::Vector, b: Self::Vector) -> Self::Vector {
        Avx512F64Vec(_mm512_or_pd(a.0, b.0))
    }

    #[target_feature(enable = "avx512f")]
    #[inline]
    unsafe fn bitxor(a: Self::Vector, b: Self::Vector) -> Self::Vector {
        Avx512F64Vec(_mm512_xor_pd(a.0, b.0))
    }

    #[target_feature(enable = "avx512f")]
    #[inline]
    unsafe fn abs(a: Self::Vector) -> Self::Vector {
        let sign_mask = _mm512_set1_pd(-0.0f64);
        Avx512F64Vec(_mm512_andnot_pd(sign_mask, a.0))
    }

    #[target_feature(enable = "avx512f")]
    #[inline]
    unsafe fn min(a: Self::Vector, b: Self::Vector) -> Self::Vector {
        Avx512F64Vec(_mm512_min_pd(a.0, b.0))
    }

    #[target_feature(enable = "avx512f")]
    #[inline]
    unsafe fn max(a: Self::Vector, b: Self::Vector) -> Self::Vector {
        Avx512F64Vec(_mm512_max_pd(a.0, b.0))
    }

    #[target_feature(enable = "avx512f")]
    #[inline]
    unsafe fn sqrt(a: Self::Vector) -> Self::Vector {
        Avx512F64Vec(_mm512_sqrt_pd(a.0))
    }

    #[target_feature(enable = "avx512f")]
    #[inline]
    unsafe fn cmp_eq(a: Self::Vector, b: Self::Vector) -> Self::Vector {
        let m = _mm512_cmp_pd_mask(a.0, b.0, _CMP_EQ_OQ);
        Avx512F64Vec(_mm512_mask_blend_pd(m, _mm512_setzero_pd(), _mm512_set1_pd(f64::from_bits(0xFFFF_FFFF_FFFF_FFFF))))
    }

    #[target_feature(enable = "avx512f")]
    #[inline]
    unsafe fn cmp_ne(a: Self::Vector, b: Self::Vector) -> Self::Vector {
        let m = _mm512_cmp_pd_mask(a.0, b.0, _CMP_NEQ_OQ);
        Avx512F64Vec(_mm512_mask_blend_pd(m, _mm512_setzero_pd(), _mm512_set1_pd(f64::from_bits(0xFFFF_FFFF_FFFF_FFFF))))
    }

    #[target_feature(enable = "avx512f")]
    #[inline]
    unsafe fn cmp_lt(a: Self::Vector, b: Self::Vector) -> Self::Vector {
        let m = _mm512_cmp_pd_mask(a.0, b.0, _CMP_LT_OQ);
        Avx512F64Vec(_mm512_mask_blend_pd(m, _mm512_setzero_pd(), _mm512_set1_pd(f64::from_bits(0xFFFF_FFFF_FFFF_FFFF))))
    }

    #[target_feature(enable = "avx512f")]
    #[inline]
    unsafe fn cmp_le(a: Self::Vector, b: Self::Vector) -> Self::Vector {
        let m = _mm512_cmp_pd_mask(a.0, b.0, _CMP_LE_OQ);
        Avx512F64Vec(_mm512_mask_blend_pd(m, _mm512_setzero_pd(), _mm512_set1_pd(f64::from_bits(0xFFFF_FFFF_FFFF_FFFF))))
    }

    #[target_feature(enable = "avx512f")]
    #[inline]
    unsafe fn cmp_gt(a: Self::Vector, b: Self::Vector) -> Self::Vector {
        let m = _mm512_cmp_pd_mask(a.0, b.0, _CMP_GT_OQ);
        Avx512F64Vec(_mm512_mask_blend_pd(m, _mm512_setzero_pd(), _mm512_set1_pd(f64::from_bits(0xFFFF_FFFF_FFFF_FFFF))))
    }

    #[target_feature(enable = "avx512f")]
    #[inline]
    unsafe fn cmp_ge(a: Self::Vector, b: Self::Vector) -> Self::Vector {
        let m = _mm512_cmp_pd_mask(a.0, b.0, _CMP_GE_OQ);
        Avx512F64Vec(_mm512_mask_blend_pd(m, _mm512_setzero_pd(), _mm512_set1_pd(f64::from_bits(0xFFFF_FFFF_FFFF_FFFF))))
    }

    #[target_feature(enable = "avx512f")]
    #[inline]
    unsafe fn blend(mask: Self::Vector, true_val: Self::Vector, false_val: Self::Vector) -> Self::Vector {
        let m = _mm512_cmp_pd_mask(mask.0, _mm512_setzero_pd(), _CMP_NEQ_OQ);
        Avx512F64Vec(_mm512_mask_blend_pd(m, false_val.0, true_val.0))
    }

    #[target_feature(enable = "avx512f")]
    #[inline]
    unsafe fn mask_to_bitmask(mask: Self::Mask) -> u64 {
        mask as u64
    }
}
