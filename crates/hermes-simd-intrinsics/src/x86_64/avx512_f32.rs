//! AVX-512F f32 hardware kernel.
//!
//! 16-lane f32 (`__m512`). Uses native AVX-512 predicated instructions:
//! - Masked load/store: `_mm512_mask_loadu_ps`, `_mm512_mask_storeu_ps`.
//! - Masked arithmetic: `_mm512_mask_add_ps`, `_mm512_mask_mul_ps`.
//! - Masked FMA: `_mm512_mask3_fmadd_ps` (inactive lanes retain `c`).
//! - Compress: `_mm512_maskz_compress_ps` (native — no emulation needed).
//! - Expand: `_mm512_mask_expand_ps` (native).
//! - Gather: `_mm512_i32gather_ps`, `_mm512_mask_i32gather_ps`.
//! - Mask register: `__mmask16` (16-bit integer).

use crate::Avx512;
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
use core::arch::x86_64::*;
use hermes_simd_core::kernel::SimdKernel;

/// Newtype over `__m512` providing `Send + Sync`.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[repr(transparent)]
#[derive(Copy, Clone)]
pub struct Avx512F32Vec(pub __m512);

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
unsafe impl Send for Avx512F32Vec {}
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
unsafe impl Sync for Avx512F32Vec {}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
impl SimdKernel<f32> for Avx512 {
    type Vector = Avx512F32Vec;
    /// Native AVX-512 16-bit mask register. Bit `i` set → lane `i` active.
    type Mask = __mmask16;
    /// 16 × i32 index vector for gather (`__m512i`).
    type IndexVector = __m512i;
    const LANE_COUNT: usize = 16;
    const UNROLL_FACTOR: usize = 4;

    // -----------------------------------------------------------------------
    // Load / store
    // -----------------------------------------------------------------------

    #[target_feature(enable = "avx512f")]
    #[inline]
    unsafe fn load_aligned(ptr: *const f32) -> Self::Vector {
        Avx512F32Vec(_mm512_load_ps(ptr))
    }

    #[target_feature(enable = "avx512f")]
    #[inline]
    unsafe fn load_unaligned(ptr: *const f32) -> Self::Vector {
        Avx512F32Vec(_mm512_loadu_ps(ptr))
    }

    #[target_feature(enable = "avx512f")]
    #[inline]
    unsafe fn store_aligned(ptr: *mut f32, val: Self::Vector) {
        _mm512_store_ps(ptr, val.0);
    }

    #[target_feature(enable = "avx512f")]
    #[inline]
    unsafe fn store_unaligned(ptr: *mut f32, val: Self::Vector) {
        _mm512_storeu_ps(ptr, val.0);
    }

    // -----------------------------------------------------------------------
    // Arithmetic
    // -----------------------------------------------------------------------

    #[target_feature(enable = "avx512f")]
    #[inline]
    unsafe fn add(a: Self::Vector, b: Self::Vector) -> Self::Vector {
        Avx512F32Vec(_mm512_add_ps(a.0, b.0))
    }

    #[target_feature(enable = "avx512f")]
    #[inline]
    unsafe fn mul(a: Self::Vector, b: Self::Vector) -> Self::Vector {
        Avx512F32Vec(_mm512_mul_ps(a.0, b.0))
    }

    #[target_feature(enable = "avx512f")]
    #[inline]
    unsafe fn sub(a: Self::Vector, b: Self::Vector) -> Self::Vector {
        Avx512F32Vec(_mm512_sub_ps(a.0, b.0))
    }

    #[target_feature(enable = "avx512f")]
    #[inline]
    unsafe fn fmadd(a: Self::Vector, b: Self::Vector, c: Self::Vector) -> Self::Vector {
        Avx512F32Vec(_mm512_fmadd_ps(a.0, b.0, c.0))
    }

    #[target_feature(enable = "avx512f")]
    #[inline]
    unsafe fn swap_adjacent(v: Self::Vector) -> Self::Vector {
        Avx512F32Vec(_mm512_permute_ps(v.0, 0b1011_0001))
    }

    #[target_feature(enable = "avx512f")]
    #[inline]
    unsafe fn dup_even(v: Self::Vector) -> Self::Vector {
        Avx512F32Vec(_mm512_moveldup_ps(v.0))
    }

    #[target_feature(enable = "avx512f")]
    #[inline]
    unsafe fn dup_odd(v: Self::Vector) -> Self::Vector {
        Avx512F32Vec(_mm512_movehdup_ps(v.0))
    }

    #[target_feature(enable = "avx512f")]
    #[inline]
    unsafe fn fmaddsub(a: Self::Vector, b: Self::Vector, c: Self::Vector) -> Self::Vector {
        Avx512F32Vec(_mm512_fmaddsub_ps(a.0, b.0, c.0))
    }

    #[target_feature(enable = "avx512f")]
    #[inline]
    unsafe fn fmsubadd(a: Self::Vector, b: Self::Vector, c: Self::Vector) -> Self::Vector {
        Avx512F32Vec(_mm512_fmsubadd_ps(a.0, b.0, c.0))
    }

    #[target_feature(enable = "avx512f")]
    #[inline]
    unsafe fn sum_reduce(v: Self::Vector) -> f32 {
        _mm512_reduce_add_ps(v.0)
    }

    // -----------------------------------------------------------------------
    // Native masked load / store
    // -----------------------------------------------------------------------

    #[target_feature(enable = "avx512f")]
    #[inline]
    unsafe fn masked_load_unaligned(
        ptr: *const f32,
        mask: Self::Mask,
        src: Self::Vector,
    ) -> Self::Vector {
        Avx512F32Vec(_mm512_mask_loadu_ps(src.0, mask, ptr))
    }

    #[target_feature(enable = "avx512f")]
    #[inline]
    unsafe fn masked_store_unaligned(ptr: *mut f32, mask: Self::Mask, val: Self::Vector) {
        _mm512_mask_storeu_ps(ptr, mask, val.0);
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
        Avx512F32Vec(_mm512_mask_add_ps(src.0, mask, a.0, b.0))
    }

    #[target_feature(enable = "avx512f")]
    #[inline]
    unsafe fn masked_mul(
        a: Self::Vector,
        b: Self::Vector,
        mask: Self::Mask,
        src: Self::Vector,
    ) -> Self::Vector {
        Avx512F32Vec(_mm512_mask_mul_ps(src.0, mask, a.0, b.0))
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
        Avx512F32Vec(_mm512_mask3_fmadd_ps(a.0, b.0, c.0, mask))
    }

    #[target_feature(enable = "avx512f")]
    #[inline]
    unsafe fn masked_sum_reduce(v: Self::Vector, mask: Self::Mask) -> f32 {
        let zero = _mm512_setzero_ps();
        _mm512_reduce_add_ps(_mm512_mask_mov_ps(zero, mask, v.0))
    }

    // -----------------------------------------------------------------------
    // Native compress / expand
    // -----------------------------------------------------------------------

    #[target_feature(enable = "avx512f")]
    #[inline]
    unsafe fn compress(src: Self::Vector, mask: Self::Mask) -> Self::Vector {
        Avx512F32Vec(_mm512_maskz_compress_ps(mask, src.0))
    }

    #[target_feature(enable = "avx512f")]
    #[inline]
    unsafe fn expand(src: Self::Vector, mask: Self::Mask, fill: Self::Vector) -> Self::Vector {
        Avx512F32Vec(_mm512_mask_expand_ps(fill.0, mask, src.0))
    }

    // -----------------------------------------------------------------------
    // Gather
    // -----------------------------------------------------------------------

    #[target_feature(enable = "avx512f")]
    #[inline]
    unsafe fn gather(base: *const f32, indices: Self::IndexVector) -> Self::Vector {
        Avx512F32Vec(_mm512_i32gather_ps(indices, base, 4))
    }

    #[target_feature(enable = "avx512f")]
    #[inline]
    unsafe fn gather_masked(
        base: *const f32,
        indices: Self::IndexVector,
        mask: Self::Mask,
        src: Self::Vector,
    ) -> Self::Vector {
        Avx512F32Vec(_mm512_mask_i32gather_ps(src.0, mask, indices, base, 4))
    }

    // -----------------------------------------------------------------------
    // Mask construction
    // -----------------------------------------------------------------------

    #[target_feature(enable = "avx512f")]
    #[inline]
    unsafe fn mask_from_bools(bits: &[bool]) -> Self::Mask {
        debug_assert_eq!(bits.len(), 16);
        let mut mask: u64 = 0;
        for i in 0..16 {
            if bits[i] {
                mask |= 1 << i;
            }
        }
        mask as Self::Mask
    }

    #[target_feature(enable = "avx512f")]
    #[inline]
    unsafe fn leading_k_mask(k: usize) -> Self::Mask {
        let k = k.min(16);
        if k == 0 {
            0 as Self::Mask
        } else {
            let mask = (1u64 << k) - 1;
            mask as Self::Mask
        }
    }

    // -----------------------------------------------------------------------
    // Broadcast / zero
    // -----------------------------------------------------------------------

    #[target_feature(enable = "avx512f")]
    #[inline]
    unsafe fn zero() -> Self::Vector {
        Avx512F32Vec(_mm512_setzero_ps())
    }

    #[target_feature(enable = "avx512f")]
    #[inline]
    unsafe fn splat(val: f32) -> Self::Vector {
        Avx512F32Vec(_mm512_set1_ps(val))
    }

    #[target_feature(enable = "avx512f")]
    #[inline]
    unsafe fn div(a: Self::Vector, b: Self::Vector) -> Self::Vector {
        Avx512F32Vec(_mm512_div_ps(a.0, b.0))
    }

    #[target_feature(enable = "avx512f")]
    #[inline]
    unsafe fn bitand(a: Self::Vector, b: Self::Vector) -> Self::Vector {
        Avx512F32Vec(_mm512_and_ps(a.0, b.0))
    }

    #[target_feature(enable = "avx512f")]
    #[inline]
    unsafe fn bitor(a: Self::Vector, b: Self::Vector) -> Self::Vector {
        Avx512F32Vec(_mm512_or_ps(a.0, b.0))
    }

    #[target_feature(enable = "avx512f")]
    #[inline]
    unsafe fn bitxor(a: Self::Vector, b: Self::Vector) -> Self::Vector {
        Avx512F32Vec(_mm512_xor_ps(a.0, b.0))
    }

    #[target_feature(enable = "avx512f")]
    #[inline]
    unsafe fn abs(a: Self::Vector) -> Self::Vector {
        let sign_mask = _mm512_set1_ps(-0.0);
        Avx512F32Vec(_mm512_andnot_ps(sign_mask, a.0))
    }

    #[target_feature(enable = "avx512f")]
    #[inline]
    unsafe fn min(a: Self::Vector, b: Self::Vector) -> Self::Vector {
        Avx512F32Vec(_mm512_min_ps(a.0, b.0))
    }

    #[target_feature(enable = "avx512f")]
    #[inline]
    unsafe fn max(a: Self::Vector, b: Self::Vector) -> Self::Vector {
        Avx512F32Vec(_mm512_max_ps(a.0, b.0))
    }

    #[target_feature(enable = "avx512f")]
    #[inline]
    unsafe fn sqrt(a: Self::Vector) -> Self::Vector {
        Avx512F32Vec(_mm512_sqrt_ps(a.0))
    }

    #[target_feature(enable = "avx512f")]
    #[inline]
    unsafe fn recip_sqrt(a: Self::Vector) -> Self::Vector {
        let y0 = _mm512_rsqrt14_ps(a.0);
        let y0_sq = _mm512_mul_ps(y0, y0);
        let half_y0 = _mm512_mul_ps(y0, _mm512_set1_ps(0.5));
        let term = _mm512_fnmadd_ps(a.0, y0_sq, _mm512_set1_ps(3.0));
        Avx512F32Vec(_mm512_mul_ps(half_y0, term))
    }

    #[target_feature(enable = "avx512f")]
    #[inline]
    unsafe fn popcount(a: Self::Vector) -> Self::Vector {
        use core::arch::x86_64::*;
        let v_si512 = _mm512_castps_si512(a.0);
        let lo = _mm512_extracti64x4_epi64(v_si512, 0);
        let hi = _mm512_extracti64x4_epi64(v_si512, 1);

        let low_mask = _mm256_set1_epi8(0x0F);
        let lookup = _mm256_setr_epi8(
            0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4, 0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2,
            3, 3, 4,
        );

        let run_avx2_popcount = |v: __m256i| -> __m256 {
            let v_lo = _mm256_and_si256(v, low_mask);
            let v_hi = _mm256_and_si256(_mm256_srli_epi16(v, 4), low_mask);
            let pop_lo = _mm256_shuffle_epi8(lookup, v_lo);
            let pop_hi = _mm256_shuffle_epi8(lookup, v_hi);
            let pop_bytes = _mm256_add_epi8(pop_lo, pop_hi);
            let pop_u16 = _mm256_maddubs_epi16(pop_bytes, _mm256_set1_epi8(1));
            let pop_u32 = _mm256_madd_epi16(pop_u16, _mm256_set1_epi16(1));
            _mm256_cvtepi32_ps(pop_u32)
        };

        let res_lo = run_avx2_popcount(lo);
        let res_hi = run_avx2_popcount(hi);

        let merged = _mm512_insertf32x8(_mm512_castps256_ps512(res_lo), res_hi, 1);
        Avx512F32Vec(merged)
    }

    #[target_feature(enable = "avx512f")]
    #[inline]
    unsafe fn cmp_eq(a: Self::Vector, b: Self::Vector) -> Self::Vector {
        let m = _mm512_cmp_ps_mask(a.0, b.0, _CMP_EQ_OQ);
        Avx512F32Vec(_mm512_mask_blend_ps(
            m,
            _mm512_setzero_ps(),
            _mm512_set1_ps(f32::from_bits(!0)),
        ))
    }

    #[target_feature(enable = "avx512f")]
    #[inline]
    unsafe fn cmp_ne(a: Self::Vector, b: Self::Vector) -> Self::Vector {
        let m = _mm512_cmp_ps_mask(a.0, b.0, _CMP_NEQ_OQ);
        Avx512F32Vec(_mm512_mask_blend_ps(
            m,
            _mm512_setzero_ps(),
            _mm512_set1_ps(f32::from_bits(!0)),
        ))
    }

    #[target_feature(enable = "avx512f")]
    #[inline]
    unsafe fn cmp_lt(a: Self::Vector, b: Self::Vector) -> Self::Vector {
        let m = _mm512_cmp_ps_mask(a.0, b.0, _CMP_LT_OQ);
        Avx512F32Vec(_mm512_mask_blend_ps(
            m,
            _mm512_setzero_ps(),
            _mm512_set1_ps(f32::from_bits(!0)),
        ))
    }

    #[target_feature(enable = "avx512f")]
    #[inline]
    unsafe fn cmp_le(a: Self::Vector, b: Self::Vector) -> Self::Vector {
        let m = _mm512_cmp_ps_mask(a.0, b.0, _CMP_LE_OQ);
        Avx512F32Vec(_mm512_mask_blend_ps(
            m,
            _mm512_setzero_ps(),
            _mm512_set1_ps(f32::from_bits(!0)),
        ))
    }

    #[target_feature(enable = "avx512f")]
    #[inline]
    unsafe fn cmp_gt(a: Self::Vector, b: Self::Vector) -> Self::Vector {
        let m = _mm512_cmp_ps_mask(a.0, b.0, _CMP_GT_OQ);
        Avx512F32Vec(_mm512_mask_blend_ps(
            m,
            _mm512_setzero_ps(),
            _mm512_set1_ps(f32::from_bits(!0)),
        ))
    }

    #[target_feature(enable = "avx512f")]
    #[inline]
    unsafe fn cmp_ge(a: Self::Vector, b: Self::Vector) -> Self::Vector {
        let m = _mm512_cmp_ps_mask(a.0, b.0, _CMP_GE_OQ);
        Avx512F32Vec(_mm512_mask_blend_ps(
            m,
            _mm512_setzero_ps(),
            _mm512_set1_ps(f32::from_bits(!0)),
        ))
    }

    #[target_feature(enable = "avx512f")]
    #[inline]
    unsafe fn blend(
        mask: Self::Vector,
        true_val: Self::Vector,
        false_val: Self::Vector,
    ) -> Self::Vector {
        let m = _mm512_cmp_ps_mask(mask.0, _mm512_setzero_ps(), _CMP_NEQ_OQ);
        Avx512F32Vec(_mm512_mask_blend_ps(m, false_val.0, true_val.0))
    }

    #[target_feature(enable = "avx512f")]
    #[inline]
    unsafe fn mask_to_bitmask(mask: Self::Mask) -> u64 {
        mask as u64
    }

    #[target_feature(enable = "avx512f")]
    #[inline]
    unsafe fn mask_to_vector(mask: Self::Mask) -> Self::Vector {
        Avx512F32Vec(_mm512_mask_blend_ps(
            mask,
            _mm512_setzero_ps(),
            _mm512_set1_ps(f32::from_bits(!0)),
        ))
    }
}
