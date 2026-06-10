//! AVX2 f32 hardware kernel.
//!
//! 8-lane f32 (`__m256`). Masked operations use `_mm256_blendv_ps`
//! (blend-by-sign-bit). Gather uses native `_mm256_i32gather_ps` (scale = 4).
//! Compress/expand are emulated via scalar loops — AVX2 has no native
//! `vcompress` instruction (that requires AVX-512F).

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
use core::arch::x86_64::*;
use hermes_simd_core::kernel::SimdKernel;
use crate::Avx2;

/// Newtype over `__m256` so `Send + Sync` can be implemented on the wrapper.
///
/// Raw SIMD register types are not `Send`/`Sync` by default; the newtype
/// carries those bounds because `__m256` contains no pointer indirection and
/// is safe to move across threads when treated as plain data.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[repr(transparent)]
#[derive(Copy, Clone)]
pub struct Avx2F32Vec(pub __m256);

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
unsafe impl Send for Avx2F32Vec {}
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
unsafe impl Sync for Avx2F32Vec {}

/// AVX2 f32 blend mask.
///
/// Stored as a `__m256` float register. Lane `i` is active when the sign bit
/// of `mask[i]` is set (all-ones pattern = `f32::from_bits(0xFFFF_FFFF)`).
/// This matches the `_mm256_blendv_ps` / `_mm256_maskstore_ps` conventions.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[repr(transparent)]
#[derive(Copy, Clone)]
pub struct Avx2F32Mask(pub __m256);

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
unsafe impl Send for Avx2F32Mask {}
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
unsafe impl Sync for Avx2F32Mask {}

/// AVX2 i32 index vector for f32 gather (8 × i32 in `__m256i`).
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[repr(transparent)]
#[derive(Copy, Clone)]
pub struct Avx2IdxI32(pub __m256i);

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
unsafe impl Send for Avx2IdxI32 {}
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
unsafe impl Sync for Avx2IdxI32 {}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
impl SimdKernel<f32> for Avx2 {
    type Vector = Avx2F32Vec;
    type Mask = Avx2F32Mask;
    type IndexVector = Avx2IdxI32;
    const LANE_COUNT: usize = 8;
    const UNROLL_FACTOR: usize = 4;

    // -----------------------------------------------------------------------
    // Load / store
    // -----------------------------------------------------------------------

    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn load_aligned(ptr: *const f32) -> Self::Vector {
        Avx2F32Vec(_mm256_load_ps(ptr))
    }

    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn load_unaligned(ptr: *const f32) -> Self::Vector {
        Avx2F32Vec(_mm256_loadu_ps(ptr))
    }

    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn store_aligned(ptr: *mut f32, val: Self::Vector) {
        _mm256_store_ps(ptr, val.0);
    }

    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn store_unaligned(ptr: *mut f32, val: Self::Vector) {
        _mm256_storeu_ps(ptr, val.0);
    }

    // -----------------------------------------------------------------------
    // Arithmetic
    // -----------------------------------------------------------------------

    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn add(a: Self::Vector, b: Self::Vector) -> Self::Vector {
        Avx2F32Vec(_mm256_add_ps(a.0, b.0))
    }

    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn mul(a: Self::Vector, b: Self::Vector) -> Self::Vector {
        Avx2F32Vec(_mm256_mul_ps(a.0, b.0))
    }

    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn sub(a: Self::Vector, b: Self::Vector) -> Self::Vector {
        Avx2F32Vec(_mm256_sub_ps(a.0, b.0))
    }

    /// FMA requires `avx2` + `fma` target features.
    #[target_feature(enable = "avx2,fma")]
    #[inline]
    unsafe fn fmadd(a: Self::Vector, b: Self::Vector, c: Self::Vector) -> Self::Vector {
        Avx2F32Vec(_mm256_fmadd_ps(a.0, b.0, c.0))
    }

    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn swap_adjacent(v: Self::Vector) -> Self::Vector {
        Avx2F32Vec(_mm256_permute_ps(v.0, 0b1011_0001))
    }

    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn dup_even(v: Self::Vector) -> Self::Vector {
        Avx2F32Vec(_mm256_moveldup_ps(v.0))
    }

    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn dup_odd(v: Self::Vector) -> Self::Vector {
        Avx2F32Vec(_mm256_movehdup_ps(v.0))
    }

    /// Alternating FMA requires `avx2` + `fma` target features.
    #[target_feature(enable = "avx2,fma")]
    #[inline]
    unsafe fn fmaddsub(a: Self::Vector, b: Self::Vector, c: Self::Vector) -> Self::Vector {
        Avx2F32Vec(_mm256_fmaddsub_ps(a.0, b.0, c.0))
    }

    /// Alternating FMA requires `avx2` + `fma` target features.
    #[target_feature(enable = "avx2,fma")]
    #[inline]
    unsafe fn fmsubadd(a: Self::Vector, b: Self::Vector, c: Self::Vector) -> Self::Vector {
        Avx2F32Vec(_mm256_fmsubadd_ps(a.0, b.0, c.0))
    }

    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn sum_reduce(v: Self::Vector) -> f32 {
        // Fold 8 → 4 → 2 → 1 using 128-bit halves.
        let hi_quad = _mm256_extractf128_ps(v.0, 1);
        let lo_quad = _mm256_castps256_ps128(v.0);
        let sum_quad = _mm_add_ps(lo_quad, hi_quad);
        let hi_dual = _mm_shuffle_ps(sum_quad, sum_quad, 0x4E);
        let sum_dual = _mm_add_ps(sum_quad, hi_dual);
        let hi_single = _mm_shuffle_ps(sum_dual, sum_dual, 0xB1);
        _mm_cvtss_f32(_mm_add_ps(sum_dual, hi_single))
    }

    // -----------------------------------------------------------------------
    // Masked load / store
    // -----------------------------------------------------------------------

    /// Merge-masked load via `blendv`: sign bit of mask selects loaded vs src.
    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn masked_load_unaligned(
        ptr: *const f32,
        mask: Self::Mask,
        src: Self::Vector,
    ) -> Self::Vector {
        let loaded = _mm256_loadu_ps(ptr);
        // blendv: mask sign bit 1 → loaded; 0 → src.
        Avx2F32Vec(_mm256_blendv_ps(src.0, loaded, mask.0))
    }

    /// Masked store via `_mm256_maskstore_ps`; only lanes with sign bit set are written.
    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn masked_store_unaligned(ptr: *mut f32, mask: Self::Mask, val: Self::Vector) {
        _mm256_maskstore_ps(ptr, _mm256_castps_si256(mask.0), val.0);
    }

    // -----------------------------------------------------------------------
    // Masked arithmetic
    // -----------------------------------------------------------------------

    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn masked_add(
        a: Self::Vector,
        b: Self::Vector,
        mask: Self::Mask,
        src: Self::Vector,
    ) -> Self::Vector {
        let result = _mm256_add_ps(a.0, b.0);
        Avx2F32Vec(_mm256_blendv_ps(src.0, result, mask.0))
    }

    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn masked_mul(
        a: Self::Vector,
        b: Self::Vector,
        mask: Self::Mask,
        src: Self::Vector,
    ) -> Self::Vector {
        let result = _mm256_mul_ps(a.0, b.0);
        Avx2F32Vec(_mm256_blendv_ps(src.0, result, mask.0))
    }

    #[target_feature(enable = "avx2,fma")]
    #[inline]
    unsafe fn masked_fmadd(
        a: Self::Vector,
        b: Self::Vector,
        c: Self::Vector,
        mask: Self::Mask,
    ) -> Self::Vector {
        let result = _mm256_fmadd_ps(a.0, b.0, c.0);
        // Inactive lanes retain c (addend pass-through, matching mask3_fmadd semantics).
        Avx2F32Vec(_mm256_blendv_ps(c.0, result, mask.0))
    }

    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn masked_sum_reduce(v: Self::Vector, mask: Self::Mask) -> f32 {
        let zero = _mm256_setzero_ps();
        let selected = _mm256_blendv_ps(zero, v.0, mask.0);
        Self::sum_reduce(Avx2F32Vec(selected))
    }

    // -----------------------------------------------------------------------
    // Compress / Expand (emulated — AVX2 has no native vcompress)
    // -----------------------------------------------------------------------

    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn compress(src: Self::Vector, mask: Self::Mask) -> Self::Vector {
        let mask_bits = _mm256_movemask_ps(mask.0) as u32;
        let mut arr = [0.0f32; 8];
        _mm256_storeu_ps(arr.as_mut_ptr(), src.0);
        let mut out = [0.0f32; 8];
        let mut k = 0usize;
        for i in 0..8usize {
            if (mask_bits >> i) & 1 != 0 { out[k] = arr[i]; k += 1; }
        }
        Avx2F32Vec(_mm256_loadu_ps(out.as_ptr()))
    }

    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn expand(src: Self::Vector, mask: Self::Mask, fill: Self::Vector) -> Self::Vector {
        let mask_bits = _mm256_movemask_ps(mask.0) as u32;
        let mut src_arr = [0.0f32; 8];
        _mm256_storeu_ps(src_arr.as_mut_ptr(), src.0);
        let mut out_arr = [0.0f32; 8];
        _mm256_storeu_ps(out_arr.as_mut_ptr(), fill.0);
        let mut k = 0usize;
        for i in 0..8usize {
            if (mask_bits >> i) & 1 != 0 { out_arr[i] = src_arr[k]; k += 1; }
        }
        Avx2F32Vec(_mm256_loadu_ps(out_arr.as_ptr()))
    }

    // -----------------------------------------------------------------------
    // Gather
    // -----------------------------------------------------------------------

    /// Native gather: scale = 4 (byte stride of f32).
    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn gather(base: *const f32, indices: Self::IndexVector) -> Self::Vector {
        Avx2F32Vec(_mm256_i32gather_ps(base, indices.0, 4))
    }

    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn gather_masked(
        base: *const f32,
        indices: Self::IndexVector,
        mask: Self::Mask,
        src: Self::Vector,
    ) -> Self::Vector {
        Avx2F32Vec(_mm256_mask_i32gather_ps(src.0, base, indices.0, mask.0, 4))
    }

    // -----------------------------------------------------------------------
    // Mask construction
    // -----------------------------------------------------------------------

    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn mask_from_bools(bits: &[bool]) -> Self::Mask {
        debug_assert_eq!(bits.len(), 8);
        // Active lane: all-ones bit pattern (sign bit set for blendv).
        let vals: [f32; 8] = core::array::from_fn(|i| {
            if bits[i] { f32::from_bits(0xFFFF_FFFF) } else { 0.0f32 }
        });
        Avx2F32Mask(_mm256_loadu_ps(vals.as_ptr()))
    }

    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn leading_k_mask(k: usize) -> Self::Mask {
        let k = k.min(8);
        let vals: [f32; 8] = core::array::from_fn(|i| {
            if i < k { f32::from_bits(0xFFFF_FFFF) } else { 0.0f32 }
        });
        Avx2F32Mask(_mm256_loadu_ps(vals.as_ptr()))
    }

    // -----------------------------------------------------------------------
    // Broadcast / zero
    // -----------------------------------------------------------------------

    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn zero() -> Self::Vector { Avx2F32Vec(_mm256_setzero_ps()) }

    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn splat(val: f32) -> Self::Vector { Avx2F32Vec(_mm256_set1_ps(val)) }

    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn div(a: Self::Vector, b: Self::Vector) -> Self::Vector {
        Avx2F32Vec(_mm256_div_ps(a.0, b.0))
    }

    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn bitand(a: Self::Vector, b: Self::Vector) -> Self::Vector {
        Avx2F32Vec(_mm256_and_ps(a.0, b.0))
    }

    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn bitor(a: Self::Vector, b: Self::Vector) -> Self::Vector {
        Avx2F32Vec(_mm256_or_ps(a.0, b.0))
    }

    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn bitxor(a: Self::Vector, b: Self::Vector) -> Self::Vector {
        Avx2F32Vec(_mm256_xor_ps(a.0, b.0))
    }

    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn abs(a: Self::Vector) -> Self::Vector {
        let sign_mask = _mm256_set1_ps(-0.0f32);
        Avx2F32Vec(_mm256_andnot_ps(sign_mask, a.0))
    }

    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn min(a: Self::Vector, b: Self::Vector) -> Self::Vector {
        Avx2F32Vec(_mm256_min_ps(a.0, b.0))
    }

    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn max(a: Self::Vector, b: Self::Vector) -> Self::Vector {
        Avx2F32Vec(_mm256_max_ps(a.0, b.0))
    }

    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn sqrt(a: Self::Vector) -> Self::Vector {
        Avx2F32Vec(_mm256_sqrt_ps(a.0))
    }

    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn cmp_eq(a: Self::Vector, b: Self::Vector) -> Self::Vector {
        Avx2F32Vec(_mm256_cmp_ps(a.0, b.0, _CMP_EQ_OQ))
    }

    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn cmp_ne(a: Self::Vector, b: Self::Vector) -> Self::Vector {
        Avx2F32Vec(_mm256_cmp_ps(a.0, b.0, _CMP_NEQ_OQ))
    }

    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn cmp_lt(a: Self::Vector, b: Self::Vector) -> Self::Vector {
        Avx2F32Vec(_mm256_cmp_ps(a.0, b.0, _CMP_LT_OQ))
    }

    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn cmp_le(a: Self::Vector, b: Self::Vector) -> Self::Vector {
        Avx2F32Vec(_mm256_cmp_ps(a.0, b.0, _CMP_LE_OQ))
    }

    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn cmp_gt(a: Self::Vector, b: Self::Vector) -> Self::Vector {
        Avx2F32Vec(_mm256_cmp_ps(a.0, b.0, _CMP_GT_OQ))
    }

    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn cmp_ge(a: Self::Vector, b: Self::Vector) -> Self::Vector {
        Avx2F32Vec(_mm256_cmp_ps(a.0, b.0, _CMP_GE_OQ))
    }

    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn blend(mask: Self::Vector, true_val: Self::Vector, false_val: Self::Vector) -> Self::Vector {
        Avx2F32Vec(_mm256_blendv_ps(false_val.0, true_val.0, mask.0))
    }

    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn mask_to_bitmask(mask: Self::Mask) -> u64 {
        _mm256_movemask_ps(mask.0) as u64
    }

    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn mask_to_vector(mask: Self::Mask) -> Self::Vector {
        Avx2F32Vec(mask.0)
    }
}
