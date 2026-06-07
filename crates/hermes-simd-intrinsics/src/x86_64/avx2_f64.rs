//! AVX2 f64 hardware kernel.
//!
//! 4-lane f64 (`__m256d`). Masked ops via `_mm256_blendv_pd` (blend-by-sign-bit).
//! Gather uses `_mm256_i32gather_pd` with a `__m128i` index vector (4 × i32).
//! Compress/expand are emulated via scalar loops — AVX2 has no native vcompress.

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
use core::arch::x86_64::*;
use hermes_simd_core::kernel::SimdKernel;
use crate::Avx2;

/// Newtype over `__m256d` providing `Send + Sync`.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[repr(transparent)]
#[derive(Copy, Clone)]
pub struct Avx2F64Vec(pub __m256d);

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
unsafe impl Send for Avx2F64Vec {}
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
unsafe impl Sync for Avx2F64Vec {}

/// AVX2 f64 blend mask (`__m256d`).
///
/// Lane `i` is active when the sign bit of `mask[i]` is set
/// (`f64::from_bits(0xFFFF_FFFF_FFFF_FFFF)` = all-ones).
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[repr(transparent)]
#[derive(Copy, Clone)]
pub struct Avx2F64Mask(pub __m256d);

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
unsafe impl Send for Avx2F64Mask {}
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
unsafe impl Sync for Avx2F64Mask {}

/// AVX2 f64 gather index vector: 4 × i32 packed into `__m128i`.
///
/// `_mm256_i32gather_pd` requires a 128-bit integer register holding four
/// 32-bit signed indices.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[repr(transparent)]
#[derive(Copy, Clone)]
pub struct Avx2F64Idx(pub __m128i);

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
unsafe impl Send for Avx2F64Idx {}
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
unsafe impl Sync for Avx2F64Idx {}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
impl SimdKernel<f64> for Avx2 {
    type Vector = Avx2F64Vec;
    type Mask = Avx2F64Mask;
    type IndexVector = Avx2F64Idx;
    const LANE_COUNT: usize = 4;
    const UNROLL_FACTOR: usize = 4;

    // -----------------------------------------------------------------------
    // Load / store
    // -----------------------------------------------------------------------

    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn load_aligned(ptr: *const f64) -> Self::Vector {
        Avx2F64Vec(_mm256_load_pd(ptr))
    }

    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn load_unaligned(ptr: *const f64) -> Self::Vector {
        Avx2F64Vec(_mm256_loadu_pd(ptr))
    }

    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn store_aligned(ptr: *mut f64, val: Self::Vector) {
        _mm256_store_pd(ptr, val.0);
    }

    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn store_unaligned(ptr: *mut f64, val: Self::Vector) {
        _mm256_storeu_pd(ptr, val.0);
    }

    // -----------------------------------------------------------------------
    // Arithmetic
    // -----------------------------------------------------------------------

    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn add(a: Self::Vector, b: Self::Vector) -> Self::Vector {
        Avx2F64Vec(_mm256_add_pd(a.0, b.0))
    }

    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn mul(a: Self::Vector, b: Self::Vector) -> Self::Vector {
        Avx2F64Vec(_mm256_mul_pd(a.0, b.0))
    }

    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn sub(a: Self::Vector, b: Self::Vector) -> Self::Vector {
        Avx2F64Vec(_mm256_sub_pd(a.0, b.0))
    }

    #[target_feature(enable = "avx2,fma")]
    #[inline]
    unsafe fn fmadd(a: Self::Vector, b: Self::Vector, c: Self::Vector) -> Self::Vector {
        Avx2F64Vec(_mm256_fmadd_pd(a.0, b.0, c.0))
    }

    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn sum_reduce(v: Self::Vector) -> f64 {
        // Fold 4 → 2 → 1 using 128-bit halves.
        let hi = _mm256_extractf128_pd(v.0, 1);
        let lo = _mm256_castpd256_pd128(v.0);
        let sum2 = _mm_add_pd(lo, hi);
        let hi1 = _mm_unpackhi_pd(sum2, sum2);
        _mm_cvtsd_f64(_mm_add_pd(sum2, hi1))
    }

    // -----------------------------------------------------------------------
    // Masked load / store
    // -----------------------------------------------------------------------

    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn masked_load_unaligned(
        ptr: *const f64,
        mask: Self::Mask,
        src: Self::Vector,
    ) -> Self::Vector {
        let loaded = _mm256_loadu_pd(ptr);
        Avx2F64Vec(_mm256_blendv_pd(src.0, loaded, mask.0))
    }

    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn masked_store_unaligned(ptr: *mut f64, mask: Self::Mask, val: Self::Vector) {
        _mm256_maskstore_pd(ptr, _mm256_castpd_si256(mask.0), val.0);
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
        Avx2F64Vec(_mm256_blendv_pd(src.0, _mm256_add_pd(a.0, b.0), mask.0))
    }

    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn masked_mul(
        a: Self::Vector,
        b: Self::Vector,
        mask: Self::Mask,
        src: Self::Vector,
    ) -> Self::Vector {
        Avx2F64Vec(_mm256_blendv_pd(src.0, _mm256_mul_pd(a.0, b.0), mask.0))
    }

    #[target_feature(enable = "avx2,fma")]
    #[inline]
    unsafe fn masked_fmadd(
        a: Self::Vector,
        b: Self::Vector,
        c: Self::Vector,
        mask: Self::Mask,
    ) -> Self::Vector {
        // Inactive lanes retain c (addend pass-through).
        Avx2F64Vec(_mm256_blendv_pd(c.0, _mm256_fmadd_pd(a.0, b.0, c.0), mask.0))
    }

    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn masked_sum_reduce(v: Self::Vector, mask: Self::Mask) -> f64 {
        let zero = _mm256_setzero_pd();
        Self::sum_reduce(Avx2F64Vec(_mm256_blendv_pd(zero, v.0, mask.0)))
    }

    // -----------------------------------------------------------------------
    // Compress / Expand (emulated)
    // -----------------------------------------------------------------------

    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn compress(src: Self::Vector, mask: Self::Mask) -> Self::Vector {
        let mask_bits = _mm256_movemask_pd(mask.0) as u32;
        let mut arr = [0.0f64; 4];
        _mm256_storeu_pd(arr.as_mut_ptr(), src.0);
        let mut out = [0.0f64; 4];
        let mut k = 0usize;
        for i in 0..4usize {
            if (mask_bits >> i) & 1 != 0 { out[k] = arr[i]; k += 1; }
        }
        Avx2F64Vec(_mm256_loadu_pd(out.as_ptr()))
    }

    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn expand(src: Self::Vector, mask: Self::Mask, fill: Self::Vector) -> Self::Vector {
        let mask_bits = _mm256_movemask_pd(mask.0) as u32;
        let mut src_arr = [0.0f64; 4];
        _mm256_storeu_pd(src_arr.as_mut_ptr(), src.0);
        let mut out_arr = [0.0f64; 4];
        _mm256_storeu_pd(out_arr.as_mut_ptr(), fill.0);
        let mut k = 0usize;
        for i in 0..4usize {
            if (mask_bits >> i) & 1 != 0 { out_arr[i] = src_arr[k]; k += 1; }
        }
        Avx2F64Vec(_mm256_loadu_pd(out_arr.as_ptr()))
    }

    // -----------------------------------------------------------------------
    // Gather (scale = 8 — byte stride of f64)
    // -----------------------------------------------------------------------

    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn gather(base: *const f64, indices: Self::IndexVector) -> Self::Vector {
        Avx2F64Vec(_mm256_i32gather_pd(base, indices.0, 8))
    }

    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn gather_masked(
        base: *const f64,
        indices: Self::IndexVector,
        mask: Self::Mask,
        src: Self::Vector,
    ) -> Self::Vector {
        Avx2F64Vec(_mm256_mask_i32gather_pd(src.0, base, indices.0, mask.0, 8))
    }

    // -----------------------------------------------------------------------
    // Mask construction
    // -----------------------------------------------------------------------

    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn mask_from_bools(bits: &[bool]) -> Self::Mask {
        debug_assert_eq!(bits.len(), 4);
        let vals: [f64; 4] = core::array::from_fn(|i| {
            if bits[i] { f64::from_bits(0xFFFF_FFFF_FFFF_FFFF) } else { 0.0f64 }
        });
        Avx2F64Mask(_mm256_loadu_pd(vals.as_ptr()))
    }

    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn leading_k_mask(k: usize) -> Self::Mask {
        let k = k.min(4);
        let vals: [f64; 4] = core::array::from_fn(|i| {
            if i < k { f64::from_bits(0xFFFF_FFFF_FFFF_FFFF) } else { 0.0f64 }
        });
        Avx2F64Mask(_mm256_loadu_pd(vals.as_ptr()))
    }

    // -----------------------------------------------------------------------
    // Broadcast / zero
    // -----------------------------------------------------------------------

    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn zero() -> Self::Vector { Avx2F64Vec(_mm256_setzero_pd()) }

    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn splat(val: f64) -> Self::Vector { Avx2F64Vec(_mm256_set1_pd(val)) }

    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn div(a: Self::Vector, b: Self::Vector) -> Self::Vector {
        Avx2F64Vec(_mm256_div_pd(a.0, b.0))
    }

    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn bitand(a: Self::Vector, b: Self::Vector) -> Self::Vector {
        Avx2F64Vec(_mm256_and_pd(a.0, b.0))
    }

    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn bitor(a: Self::Vector, b: Self::Vector) -> Self::Vector {
        Avx2F64Vec(_mm256_or_pd(a.0, b.0))
    }

    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn bitxor(a: Self::Vector, b: Self::Vector) -> Self::Vector {
        Avx2F64Vec(_mm256_xor_pd(a.0, b.0))
    }

    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn abs(a: Self::Vector) -> Self::Vector {
        let sign_mask = _mm256_set1_pd(-0.0f64);
        Avx2F64Vec(_mm256_andnot_pd(sign_mask, a.0))
    }

    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn min(a: Self::Vector, b: Self::Vector) -> Self::Vector {
        Avx2F64Vec(_mm256_min_pd(a.0, b.0))
    }

    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn max(a: Self::Vector, b: Self::Vector) -> Self::Vector {
        Avx2F64Vec(_mm256_max_pd(a.0, b.0))
    }

    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn sqrt(a: Self::Vector) -> Self::Vector {
        Avx2F64Vec(_mm256_sqrt_pd(a.0))
    }

    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn cmp_eq(a: Self::Vector, b: Self::Vector) -> Self::Vector {
        Avx2F64Vec(_mm256_cmp_pd(a.0, b.0, _CMP_EQ_OQ))
    }

    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn cmp_ne(a: Self::Vector, b: Self::Vector) -> Self::Vector {
        Avx2F64Vec(_mm256_cmp_pd(a.0, b.0, _CMP_NEQ_OQ))
    }

    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn cmp_lt(a: Self::Vector, b: Self::Vector) -> Self::Vector {
        Avx2F64Vec(_mm256_cmp_pd(a.0, b.0, _CMP_LT_OQ))
    }

    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn cmp_le(a: Self::Vector, b: Self::Vector) -> Self::Vector {
        Avx2F64Vec(_mm256_cmp_pd(a.0, b.0, _CMP_LE_OQ))
    }

    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn cmp_gt(a: Self::Vector, b: Self::Vector) -> Self::Vector {
        Avx2F64Vec(_mm256_cmp_pd(a.0, b.0, _CMP_GT_OQ))
    }

    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn cmp_ge(a: Self::Vector, b: Self::Vector) -> Self::Vector {
        Avx2F64Vec(_mm256_cmp_pd(a.0, b.0, _CMP_GE_OQ))
    }

    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn blend(mask: Self::Vector, true_val: Self::Vector, false_val: Self::Vector) -> Self::Vector {
        Avx2F64Vec(_mm256_blendv_pd(false_val.0, true_val.0, mask.0))
    }

    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn mask_to_bitmask(mask: Self::Mask) -> u64 {
        _mm256_movemask_pd(mask.0) as u64
    }

    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn mask_to_vector(mask: Self::Mask) -> Self::Vector {
        Avx2F64Vec(mask.0)
    }
}
