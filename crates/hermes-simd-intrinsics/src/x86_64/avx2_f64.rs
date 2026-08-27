//! AVX2 f64 hardware kernel.
//!
//! 4-lane f64 (`__m256d`). Masked operations use `_mm256_blendv_pd`
//! (blend-by-sign-bit). Gather uses native `_mm256_i32gather_pd` (scale = 8).
//! Compress/expand are emulated via scalar loops — AVX2 has no native
//! `vcompress` instruction (that requires AVX-512F).

use crate::Avx2;
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
use core::arch::x86_64::{
    __m128i, __m256d, _mm256_add_pd, _mm256_and_pd, _mm256_and_si256, _mm256_andnot_pd,
    _mm256_blendv_pd,
    _mm256_castpd256_pd128, _mm256_castpd_si256, _mm256_castsi256_pd, _mm256_ceil_pd,
    _mm256_cmp_pd, _mm256_cmpeq_epi64, _mm256_div_pd,
    _mm256_extractf128_pd, _mm256_floor_pd, _mm256_fmadd_pd, _mm256_fmaddsub_pd, _mm256_fmsub_pd,
    _mm256_fmsubadd_pd, _mm256_i32gather_pd, _mm256_load_pd, _mm256_loadu_pd,
    _mm256_mask_i32gather_pd, _mm256_maskstore_pd, _mm256_max_pd, _mm256_min_pd, _mm256_movedup_pd,
    _mm256_movemask_pd, _mm256_mul_pd, _mm256_or_pd, _mm256_permute2f128_pd, _mm256_permute4x64_pd,
    _mm256_permute_pd, _mm256_round_pd, _mm256_set1_epi64x, _mm256_set1_pd, _mm256_setr_epi64x,
    _mm256_setzero_pd, _mm256_sqrt_pd,
    _mm256_store_pd, _mm256_storeu_pd, _mm256_stream_pd, _mm256_sub_pd, _mm256_xor_pd, _mm_add_pd,
    _mm_cvtsd_f64, _mm_unpackhi_pd, _CMP_EQ_OQ, _CMP_GE_OQ, _CMP_GT_OQ, _CMP_LE_OQ, _CMP_LT_OQ,
    _CMP_NEQ_UQ, _MM_FROUND_NO_EXC, _MM_FROUND_TO_NEAREST_INT, _MM_FROUND_TO_ZERO,
};
#[cfg(all(
    any(target_arch = "x86", target_arch = "x86_64"),
    not(hermes_benchmark_generic_default)
))]
use core::arch::x86_64::{_mm256_unpackhi_pd, _mm256_unpacklo_pd};
use hermes_simd_core::kernel::BackendKernel;

/// Newtype over `__m256d` so `Send + Sync` can be implemented on the wrapper.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[repr(transparent)]
#[derive(Copy, Clone)]
pub struct Avx2F64Vec(pub __m256d);

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
unsafe impl Send for Avx2F64Vec {}
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
unsafe impl Sync for Avx2F64Vec {}

/// AVX2 f64 blend mask.
///
/// Stored as a `__m256d` register. Lane `i` is active when the sign bit
/// of `mask[i]` is set.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[repr(transparent)]
#[derive(Copy, Clone)]
pub struct Avx2F64Mask(pub __m256d);

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
unsafe impl Send for Avx2F64Mask {}
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
unsafe impl Sync for Avx2F64Mask {}

/// AVX2 gather index vector.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[repr(transparent)]
#[derive(Copy, Clone)]
pub struct Avx2F64Idx(pub __m128i);

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
unsafe impl Send for Avx2F64Idx {}
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
unsafe impl Sync for Avx2F64Idx {}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
impl BackendKernel<f64> for Avx2 {
    type Vector = Avx2F64Vec;
    type Mask = Avx2F64Mask;
    type IndexVector = Avx2F64Idx;
    const LANE_COUNT: usize = 4;
    const UNROLL_FACTOR: usize = 4;

    // -----------------------------------------------------------------------
    // Load / store
    // -----------------------------------------------------------------------

    // SAFETY: caller must ensure the target CPU supports `avx2` (enforced by the `#[target_feature]` gate above plus runtime `is_x86_feature_detected!` selection in the hermes-simd dispatcher (`target.rs`/`lib.rs`)); any pointer operands are valid for the 4-lane vector width within caller-validated bounds.
    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn load_aligned(ptr: *const f64) -> Self::Vector {
        Avx2F64Vec(_mm256_load_pd(ptr))
    }

    // SAFETY: caller must ensure the target CPU supports `avx2` (enforced by the `#[target_feature]` gate above plus runtime `is_x86_feature_detected!` selection in the hermes-simd dispatcher (`target.rs`/`lib.rs`)); any pointer operands are valid for the 4-lane vector width within caller-validated bounds.
    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn load_unaligned(ptr: *const f64) -> Self::Vector {
        Avx2F64Vec(_mm256_loadu_pd(ptr))
    }

    // SAFETY: caller must ensure the target CPU supports `avx2` (enforced by the `#[target_feature]` gate above plus runtime `is_x86_feature_detected!` selection in the hermes-simd dispatcher (`target.rs`/`lib.rs`)); any pointer operands are valid for the 4-lane vector width within caller-validated bounds.
    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn store_aligned(ptr: *mut f64, val: Self::Vector) {
        _mm256_store_pd(ptr, val.0);
    }

    // SAFETY: caller must ensure the target CPU supports `avx2` (enforced by the `#[target_feature]` gate above plus runtime `is_x86_feature_detected!` selection in the hermes-simd dispatcher (`target.rs`/`lib.rs`)); any pointer operands are valid for the 4-lane vector width within caller-validated bounds.
    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn store_unaligned(ptr: *mut f64, val: Self::Vector) {
        _mm256_storeu_pd(ptr, val.0);
    }

    const SUPPORTS_NT_STORE: bool = true;

    // SAFETY: caller must ensure the target CPU supports `avx2` (as above) and that `ptr` is aligned to the 4-lane width; `_mm256_stream_pd` (`vmovntps`/`vmovntpd`) faults on misalignment.
    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn store_streaming(ptr: *mut f64, val: Self::Vector) {
        _mm256_stream_pd(ptr, val.0);
    }

    #[inline]
    fn stream_write_barrier() {
        // SAFETY: `_mm_sfence` (SSE) is unconditionally available on x86_64.
        unsafe { core::arch::x86_64::_mm_sfence() };
    }

    // -----------------------------------------------------------------------
    // Arithmetic
    // -----------------------------------------------------------------------

    // SAFETY: caller must ensure the target CPU supports `avx2` (enforced by the `#[target_feature]` gate above plus runtime `is_x86_feature_detected!` selection in the hermes-simd dispatcher (`target.rs`/`lib.rs`)); any pointer operands are valid for the 4-lane vector width within caller-validated bounds.
    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn add(a: Self::Vector, b: Self::Vector) -> Self::Vector {
        Avx2F64Vec(_mm256_add_pd(a.0, b.0))
    }

    // SAFETY: caller must ensure the target CPU supports `avx2` (enforced by the `#[target_feature]` gate above plus runtime `is_x86_feature_detected!` selection in the hermes-simd dispatcher (`target.rs`/`lib.rs`)); any pointer operands are valid for the 4-lane vector width within caller-validated bounds.
    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn mul(a: Self::Vector, b: Self::Vector) -> Self::Vector {
        Avx2F64Vec(_mm256_mul_pd(a.0, b.0))
    }

    // SAFETY: caller must ensure the target CPU supports `avx2` (enforced by the `#[target_feature]` gate above plus runtime `is_x86_feature_detected!` selection in the hermes-simd dispatcher (`target.rs`/`lib.rs`)); any pointer operands are valid for the 4-lane vector width within caller-validated bounds.
    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn sub(a: Self::Vector, b: Self::Vector) -> Self::Vector {
        Avx2F64Vec(_mm256_sub_pd(a.0, b.0))
    }

    /// FMA requires `avx2` + `fma` target features.
    // SAFETY: caller must ensure the target CPU supports `avx2,fma` (enforced by the `#[target_feature]` gate above plus runtime `is_x86_feature_detected!` selection in the hermes-simd dispatcher (`target.rs`/`lib.rs`)); any pointer operands are valid for the 4-lane vector width within caller-validated bounds.
    #[target_feature(enable = "avx2,fma")]
    #[inline]
    unsafe fn fmadd(a: Self::Vector, b: Self::Vector, c: Self::Vector) -> Self::Vector {
        Avx2F64Vec(_mm256_fmadd_pd(a.0, b.0, c.0))
    }

    // SAFETY: caller must ensure the target CPU supports `avx2,fma`; operands
    // are registers of this backend and require no pointer validation.
    #[target_feature(enable = "avx2,fma")]
    #[inline]
    unsafe fn fmsub(a: Self::Vector, b: Self::Vector, c: Self::Vector) -> Self::Vector {
        // SAFETY: this function enables AVX2 and FMA, and all operands are
        // valid registers of the matching backend.
        Avx2F64Vec(_mm256_fmsub_pd(a.0, b.0, c.0))
    }

    // SAFETY: caller must ensure the target CPU supports `avx2` (enforced by the `#[target_feature]` gate above plus runtime `is_x86_feature_detected!` selection in the hermes-simd dispatcher (`target.rs`/`lib.rs`)); any pointer operands are valid for the 4-lane vector width within caller-validated bounds.
    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn swap_adjacent(v: Self::Vector) -> Self::Vector {
        Avx2F64Vec(_mm256_permute_pd(v.0, 0b0101))
    }

    // SAFETY: caller must ensure the target CPU supports `avx2` (enforced by the `#[target_feature]` gate above plus runtime `is_x86_feature_detected!` selection in the hermes-simd dispatcher (`target.rs`/`lib.rs`)); any pointer operands are valid for the 4-lane vector width within caller-validated bounds.
    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn swap_pairs(v: Self::Vector) -> Self::Vector {
        // Four f64 lanes hold two pairs, one per 128-bit half, so the pair
        // swap is exactly the half exchange.
        Avx2F64Vec(_mm256_permute2f128_pd(v.0, v.0, 0x01))
    }

    // SAFETY: caller must ensure the target CPU supports `avx2` (enforced by the `#[target_feature]` gate above plus runtime `is_x86_feature_detected!` selection in the hermes-simd dispatcher (`target.rs`/`lib.rs`)); any pointer operands are valid for the 4-lane vector width within caller-validated bounds.
    #[target_feature(enable = "avx2")]
    #[inline]
    #[cfg(not(hermes_benchmark_generic_default))]
    unsafe fn transpose_square(tile: &mut [Self::Vector]) {
        debug_assert_eq!(tile.len(), 4, "tile must hold LANE_COUNT rows");
        // The canonical eight-shuffle 4x4 network: unpacks weave within
        // 128-bit halves, the cross-half permutes assemble whole columns.
        let t0 = _mm256_unpacklo_pd(tile[0].0, tile[1].0);
        let t1 = _mm256_unpackhi_pd(tile[0].0, tile[1].0);
        let t2 = _mm256_unpacklo_pd(tile[2].0, tile[3].0);
        let t3 = _mm256_unpackhi_pd(tile[2].0, tile[3].0);
        tile[0] = Avx2F64Vec(_mm256_permute2f128_pd::<0x20>(t0, t2));
        tile[1] = Avx2F64Vec(_mm256_permute2f128_pd::<0x20>(t1, t3));
        tile[2] = Avx2F64Vec(_mm256_permute2f128_pd::<0x31>(t0, t2));
        tile[3] = Avx2F64Vec(_mm256_permute2f128_pd::<0x31>(t1, t3));
    }

    // SAFETY: caller must ensure the target CPU supports `avx2` (enforced by the `#[target_feature]` gate above plus runtime `is_x86_feature_detected!` selection in the hermes-simd dispatcher (`target.rs`/`lib.rs`)); any pointer operands are valid for the 4-lane vector width within caller-validated bounds.
    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn reverse(v: Self::Vector) -> Self::Vector {
        // `vpermpd` with an immediate selector crosses the 128-bit halves, so
        // one instruction expresses the flat reversal. Each 2-bit field of the
        // immediate picks the source lane for one destination lane, low field
        // first: lane0<-3, lane1<-2, lane2<-1, lane3<-0 is 0b00_01_10_11.
        Avx2F64Vec(_mm256_permute4x64_pd::<0b00_01_10_11>(v.0))
    }

    // SAFETY: caller must ensure the target CPU supports `avx2` (enforced by the `#[target_feature]` gate above plus runtime `is_x86_feature_detected!` selection in the hermes-simd dispatcher (`target.rs`/`lib.rs`)); any pointer operands are valid for the 4-lane vector width within caller-validated bounds.
    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn dup_even(v: Self::Vector) -> Self::Vector {
        Avx2F64Vec(_mm256_movedup_pd(v.0))
    }

    // SAFETY: caller must ensure the target CPU supports `avx2` (enforced by the `#[target_feature]` gate above plus runtime `is_x86_feature_detected!` selection in the hermes-simd dispatcher (`target.rs`/`lib.rs`)); any pointer operands are valid for the 4-lane vector width within caller-validated bounds.
    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn dup_odd(v: Self::Vector) -> Self::Vector {
        Avx2F64Vec(_mm256_permute_pd(v.0, 0b1111))
    }

    // SAFETY: caller must ensure the target CPU supports `avx2` (enforced by the `#[target_feature]` gate above plus runtime `is_x86_feature_detected!` selection in the hermes-simd dispatcher (`target.rs`/`lib.rs`)); any pointer operands are valid for the 4-lane vector width within caller-validated bounds.
    #[target_feature(enable = "avx2")]
    #[inline]
    #[cfg(not(hermes_benchmark_generic_default))]
    unsafe fn interleave(a: Self::Vector, b: Self::Vector) -> (Self::Vector, Self::Vector) {
        // `unpacklo/hi_pd` weave within 128-bit halves, so pre-permute each
        // source to [0, 2, 1, 3] order; the unpacks then produce the flat
        // interleave of the 8-lane sequence.
        let ta = _mm256_permute4x64_pd::<0b11_01_10_00>(a.0);
        let tb = _mm256_permute4x64_pd::<0b11_01_10_00>(b.0);
        (
            Avx2F64Vec(_mm256_unpacklo_pd(ta, tb)),
            Avx2F64Vec(_mm256_unpackhi_pd(ta, tb)),
        )
    }

    // SAFETY: caller must ensure the target CPU supports `avx2` (enforced by the `#[target_feature]` gate above plus runtime `is_x86_feature_detected!` selection in the hermes-simd dispatcher (`target.rs`/`lib.rs`)); any pointer operands are valid for the 4-lane vector width within caller-validated bounds.
    #[target_feature(enable = "avx2")]
    #[inline]
    #[cfg(not(hermes_benchmark_generic_default))]
    unsafe fn deinterleave(a: Self::Vector, b: Self::Vector) -> (Self::Vector, Self::Vector) {
        // `unpacklo/hi_pd` split evens/odds within 128-bit halves in
        // [a, b | a, b] block order; the closing [0, 2, 1, 3] permute
        // restores flat lane order.
        let t0 = _mm256_unpacklo_pd(a.0, b.0);
        let t1 = _mm256_unpackhi_pd(a.0, b.0);
        (
            Avx2F64Vec(_mm256_permute4x64_pd::<0b11_01_10_00>(t0)),
            Avx2F64Vec(_mm256_permute4x64_pd::<0b11_01_10_00>(t1)),
        )
    }

    /// Alternating FMA requires `avx2` + `fma` target features.
    // SAFETY: caller must ensure the target CPU supports `avx2,fma` (enforced by the `#[target_feature]` gate above plus runtime `is_x86_feature_detected!` selection in the hermes-simd dispatcher (`target.rs`/`lib.rs`)); any pointer operands are valid for the 4-lane vector width within caller-validated bounds.
    #[target_feature(enable = "avx2,fma")]
    #[inline]
    unsafe fn fmaddsub(a: Self::Vector, b: Self::Vector, c: Self::Vector) -> Self::Vector {
        Avx2F64Vec(_mm256_fmaddsub_pd(a.0, b.0, c.0))
    }

    /// Alternating FMA requires `avx2` + `fma` target features.
    // SAFETY: caller must ensure the target CPU supports `avx2,fma` (enforced by the `#[target_feature]` gate above plus runtime `is_x86_feature_detected!` selection in the hermes-simd dispatcher (`target.rs`/`lib.rs`)); any pointer operands are valid for the 4-lane vector width within caller-validated bounds.
    #[target_feature(enable = "avx2,fma")]
    #[inline]
    unsafe fn fmsubadd(a: Self::Vector, b: Self::Vector, c: Self::Vector) -> Self::Vector {
        Avx2F64Vec(_mm256_fmsubadd_pd(a.0, b.0, c.0))
    }

    // SAFETY: caller must ensure the target CPU supports `avx2` (enforced by the `#[target_feature]` gate above plus runtime `is_x86_feature_detected!` selection in the hermes-simd dispatcher (`target.rs`/`lib.rs`)); any pointer operands are valid for the 4-lane vector width within caller-validated bounds.
    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn sum_reduce(v: Self::Vector) -> f64 {
        let hi = _mm256_extractf128_pd(v.0, 1);
        let lo = _mm256_castpd256_pd128(v.0);
        let sum_128 = _mm_add_pd(lo, hi);
        let hi_lane = _mm_unpackhi_pd(sum_128, sum_128);
        _mm_cvtsd_f64(_mm_add_pd(sum_128, hi_lane))
    }

    // -----------------------------------------------------------------------
    // Masked load / store
    // -----------------------------------------------------------------------

    /// Merge-masked load via `blendv`: sign bit of mask selects loaded vs src.
    // SAFETY: caller must ensure the target CPU supports `avx2` (enforced by the `#[target_feature]` gate above plus runtime `is_x86_feature_detected!` selection in the hermes-simd dispatcher (`target.rs`/`lib.rs`)); any pointer operands are valid for the 4-lane vector width within caller-validated bounds.
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

    /// Masked store via maskstore; only lanes with sign bit set are written.
    // SAFETY: caller must ensure the target CPU supports `avx2` (enforced by the `#[target_feature]` gate above plus runtime `is_x86_feature_detected!` selection in the hermes-simd dispatcher (`target.rs`/`lib.rs`)); any pointer operands are valid for the 4-lane vector width within caller-validated bounds.
    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn masked_store_unaligned(ptr: *mut f64, mask: Self::Mask, val: Self::Vector) {
        _mm256_maskstore_pd(ptr, _mm256_castpd_si256(mask.0), val.0);
    }

    // -----------------------------------------------------------------------
    // Masked arithmetic
    // -----------------------------------------------------------------------

    // SAFETY: caller must ensure the target CPU supports `avx2` (enforced by the `#[target_feature]` gate above plus runtime `is_x86_feature_detected!` selection in the hermes-simd dispatcher (`target.rs`/`lib.rs`)); any pointer operands are valid for the 4-lane vector width within caller-validated bounds.
    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn masked_add(
        a: Self::Vector,
        b: Self::Vector,
        mask: Self::Mask,
        src: Self::Vector,
    ) -> Self::Vector {
        let result = _mm256_add_pd(a.0, b.0);
        Avx2F64Vec(_mm256_blendv_pd(src.0, result, mask.0))
    }

    // SAFETY: caller must ensure the target CPU supports `avx2` (enforced by the `#[target_feature]` gate above plus runtime `is_x86_feature_detected!` selection in the hermes-simd dispatcher (`target.rs`/`lib.rs`)); any pointer operands are valid for the 4-lane vector width within caller-validated bounds.
    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn masked_mul(
        a: Self::Vector,
        b: Self::Vector,
        mask: Self::Mask,
        src: Self::Vector,
    ) -> Self::Vector {
        let result = _mm256_mul_pd(a.0, b.0);
        Avx2F64Vec(_mm256_blendv_pd(src.0, result, mask.0))
    }

    // SAFETY: caller must ensure the target CPU supports `avx2,fma` (enforced by the `#[target_feature]` gate above plus runtime `is_x86_feature_detected!` selection in the hermes-simd dispatcher (`target.rs`/`lib.rs`)); any pointer operands are valid for the 4-lane vector width within caller-validated bounds.
    #[target_feature(enable = "avx2,fma")]
    #[inline]
    unsafe fn masked_fmadd(
        a: Self::Vector,
        b: Self::Vector,
        c: Self::Vector,
        mask: Self::Mask,
    ) -> Self::Vector {
        let result = _mm256_fmadd_pd(a.0, b.0, c.0);
        Avx2F64Vec(_mm256_blendv_pd(c.0, result, mask.0))
    }

    // SAFETY: caller must ensure the target CPU supports `avx2` (enforced by the `#[target_feature]` gate above plus runtime `is_x86_feature_detected!` selection in the hermes-simd dispatcher (`target.rs`/`lib.rs`)); any pointer operands are valid for the 4-lane vector width within caller-validated bounds.
    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn masked_sum_reduce(v: Self::Vector, mask: Self::Mask) -> f64 {
        let zero = _mm256_setzero_pd();
        let selected = _mm256_blendv_pd(zero, v.0, mask.0);
        Self::sum_reduce(Avx2F64Vec(selected))
    }

    // -----------------------------------------------------------------------
    // Compress / Expand (emulated)
    // -----------------------------------------------------------------------

    // SAFETY: caller must ensure the target CPU supports `avx2` (enforced by the `#[target_feature]` gate above plus runtime `is_x86_feature_detected!` selection in the hermes-simd dispatcher (`target.rs`/`lib.rs`)); any pointer operands are valid for the 4-lane vector width within caller-validated bounds.
    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn compress(src: Self::Vector, mask: Self::Mask) -> Self::Vector {
        let mask_bits = _mm256_movemask_pd(mask.0) as u32;
        let mut arr = [0.0f64; 4];
        _mm256_storeu_pd(arr.as_mut_ptr(), src.0);
        let mut out = [0.0f64; 4];
        let mut k = 0usize;
        for i in 0..4 {
            if (mask_bits >> i) & 1 != 0 {
                out[k] = arr[i];
                k += 1;
            }
        }
        Avx2F64Vec(_mm256_loadu_pd(out.as_ptr()))
    }

    // SAFETY: caller must ensure the target CPU supports `avx2` (enforced by the `#[target_feature]` gate above plus runtime `is_x86_feature_detected!` selection in the hermes-simd dispatcher (`target.rs`/`lib.rs`)); any pointer operands are valid for the 4-lane vector width within caller-validated bounds.
    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn expand(src: Self::Vector, mask: Self::Mask, fill: Self::Vector) -> Self::Vector {
        let mask_bits = _mm256_movemask_pd(mask.0) as u32;
        let mut src_arr = [0.0f64; 4];
        _mm256_storeu_pd(src_arr.as_mut_ptr(), src.0);
        let mut out_arr = [0.0f64; 4];
        _mm256_storeu_pd(out_arr.as_mut_ptr(), fill.0);
        let mut k = 0usize;
        for i in 0..4 {
            if (mask_bits >> i) & 1 != 0 {
                out_arr[i] = src_arr[k];
                k += 1;
            }
        }
        Avx2F64Vec(_mm256_loadu_pd(out_arr.as_ptr()))
    }

    // -----------------------------------------------------------------------
    // Gather
    // -----------------------------------------------------------------------

    // SAFETY: caller must ensure the target CPU supports `avx2` (enforced by the `#[target_feature]` gate above plus runtime `is_x86_feature_detected!` selection in the hermes-simd dispatcher (`target.rs`/`lib.rs`)); any pointer operands are valid for the 4-lane vector width within caller-validated bounds.
    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn gather(base: *const f64, indices: Self::IndexVector) -> Self::Vector {
        Avx2F64Vec(_mm256_i32gather_pd(base, indices.0, 8))
    }

    // SAFETY: caller must ensure the target CPU supports `avx2` (enforced by the `#[target_feature]` gate above plus runtime `is_x86_feature_detected!` selection in the hermes-simd dispatcher (`target.rs`/`lib.rs`)); any pointer operands are valid for the 4-lane vector width within caller-validated bounds.
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

    // SAFETY: caller must ensure the target CPU supports `avx2` (enforced by the `#[target_feature]` gate above plus runtime `is_x86_feature_detected!` selection in the hermes-simd dispatcher (`target.rs`/`lib.rs`)); any pointer operands are valid for the 4-lane vector width within caller-validated bounds.
    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn mask_from_bools(bits: &[bool]) -> Self::Mask {
        debug_assert_eq!(bits.len(), 4);
        let vals: [f64; 4] =
            core::array::from_fn(|i| if bits[i] { <f64>::from_bits(!0) } else { 0.0 });
        Avx2F64Mask(_mm256_loadu_pd(vals.as_ptr()))
    }

    // SAFETY: caller must ensure the target CPU supports `avx2` (enforced by the `#[target_feature]` gate above plus runtime `is_x86_feature_detected!` selection in the hermes-simd dispatcher (`target.rs`/`lib.rs`)); any pointer operands are valid for the 4-lane vector width within caller-validated bounds.
    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn leading_k_mask(k: usize) -> Self::Mask {
        let k = k.min(4);
        let vals: [f64; 4] =
            core::array::from_fn(|i| if i < k { <f64>::from_bits(!0) } else { 0.0 });
        Avx2F64Mask(_mm256_loadu_pd(vals.as_ptr()))
    }

    // SAFETY: caller must ensure the target CPU supports `avx2` (enforced by the `#[target_feature]` gate above plus runtime `is_x86_feature_detected!` selection in the hermes-simd dispatcher (`target.rs`/`lib.rs`)); this is a register-only expansion with no memory operands.
    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn mask_from_bitmask(bm: u64) -> Self::Mask {
        // Register-only expansion replacing the generic bool-array bounce:
        // broadcast the low bits, isolate each lane's bit, and compare-equal
        // to produce canonical all-ones/all-zero lanes (bits 4.. are ignored
        // because only bits 0..4 appear in `lane_bits`).
        let lane_bits = _mm256_setr_epi64x(1, 2, 4, 8);
        let selected = _mm256_and_si256(_mm256_set1_epi64x(bm as i64), lane_bits);
        Avx2F64Mask(_mm256_castsi256_pd(_mm256_cmpeq_epi64(selected, lane_bits)))
    }

    // -----------------------------------------------------------------------
    // Broadcast / zero
    // -----------------------------------------------------------------------

    // SAFETY: caller must ensure the target CPU supports `avx2` (enforced by the `#[target_feature]` gate above plus runtime `is_x86_feature_detected!` selection in the hermes-simd dispatcher (`target.rs`/`lib.rs`)); any pointer operands are valid for the 4-lane vector width within caller-validated bounds.
    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn zero() -> Self::Vector {
        Avx2F64Vec(_mm256_setzero_pd())
    }

    // SAFETY: caller must ensure the target CPU supports `avx2` (enforced by the `#[target_feature]` gate above plus runtime `is_x86_feature_detected!` selection in the hermes-simd dispatcher (`target.rs`/`lib.rs`)); any pointer operands are valid for the 4-lane vector width within caller-validated bounds.
    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn splat(val: f64) -> Self::Vector {
        Avx2F64Vec(_mm256_set1_pd(val))
    }

    // SAFETY: caller must ensure the target CPU supports `avx2` (enforced by the `#[target_feature]` gate above plus runtime `is_x86_feature_detected!` selection in the hermes-simd dispatcher (`target.rs`/`lib.rs`)); any pointer operands are valid for the 4-lane vector width within caller-validated bounds.
    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn div(a: Self::Vector, b: Self::Vector) -> Self::Vector {
        Avx2F64Vec(_mm256_div_pd(a.0, b.0))
    }

    // SAFETY: caller must ensure the target CPU supports `avx2` (enforced by the `#[target_feature]` gate above plus runtime `is_x86_feature_detected!` selection in the hermes-simd dispatcher (`target.rs`/`lib.rs`)); any pointer operands are valid for the 4-lane vector width within caller-validated bounds.
    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn bitand(a: Self::Vector, b: Self::Vector) -> Self::Vector {
        Avx2F64Vec(_mm256_and_pd(a.0, b.0))
    }

    // SAFETY: caller must ensure the target CPU supports `avx2` (enforced by the `#[target_feature]` gate above plus runtime `is_x86_feature_detected!` selection in the hermes-simd dispatcher (`target.rs`/`lib.rs`)); any pointer operands are valid for the 4-lane vector width within caller-validated bounds.
    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn bitor(a: Self::Vector, b: Self::Vector) -> Self::Vector {
        Avx2F64Vec(_mm256_or_pd(a.0, b.0))
    }

    // SAFETY: caller must ensure the target CPU supports `avx2` (enforced by the `#[target_feature]` gate above plus runtime `is_x86_feature_detected!` selection in the hermes-simd dispatcher (`target.rs`/`lib.rs`)); any pointer operands are valid for the 4-lane vector width within caller-validated bounds.
    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn bitxor(a: Self::Vector, b: Self::Vector) -> Self::Vector {
        Avx2F64Vec(_mm256_xor_pd(a.0, b.0))
    }

    // SAFETY: caller must ensure the target CPU supports `avx2` (enforced by the `#[target_feature]` gate above plus runtime `is_x86_feature_detected!` selection in the hermes-simd dispatcher (`target.rs`/`lib.rs`)); any pointer operands are valid for the 4-lane vector width within caller-validated bounds.
    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn abs(a: Self::Vector) -> Self::Vector {
        let sign_mask = _mm256_set1_pd(-0.0);
        Avx2F64Vec(_mm256_andnot_pd(sign_mask, a.0))
    }

    // SAFETY: caller must ensure the target CPU supports `avx2` (enforced by the `#[target_feature]` gate above plus runtime `is_x86_feature_detected!` selection in the hermes-simd dispatcher (`target.rs`/`lib.rs`)); any pointer operands are valid for the 4-lane vector width within caller-validated bounds.
    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn min(a: Self::Vector, b: Self::Vector) -> Self::Vector {
        Avx2F64Vec(_mm256_min_pd(a.0, b.0))
    }

    // SAFETY: caller must ensure the target CPU supports `avx2` (enforced by the `#[target_feature]` gate above plus runtime `is_x86_feature_detected!` selection in the hermes-simd dispatcher (`target.rs`/`lib.rs`)); any pointer operands are valid for the 4-lane vector width within caller-validated bounds.
    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn max(a: Self::Vector, b: Self::Vector) -> Self::Vector {
        Avx2F64Vec(_mm256_max_pd(a.0, b.0))
    }

    // SAFETY: caller must ensure the target CPU supports `avx2` (enforced by the `#[target_feature]` gate above plus runtime `is_x86_feature_detected!` selection in the hermes-simd dispatcher (`target.rs`/`lib.rs`)); any pointer operands are valid for the 4-lane vector width within caller-validated bounds.
    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn sqrt(a: Self::Vector) -> Self::Vector {
        Avx2F64Vec(_mm256_sqrt_pd(a.0))
    }

    // SAFETY: caller must ensure the target CPU supports `avx2` (enforced by the `#[target_feature]` gate above plus runtime `is_x86_feature_detected!` selection in the hermes-simd dispatcher (`target.rs`/`lib.rs`)); any pointer operands are valid for the 4-lane vector width within caller-validated bounds.
    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn recip_sqrt(a: Self::Vector) -> Self::Vector {
        // f64 has no hardware reciprocal-sqrt approximation accurate enough for its
        // 52-bit mantissa: an rsqrt seed (~12 bits from the f32 `_mm_rsqrt_ps`) plus
        // one Newton step reaches only ~24 bits. Use the correctly-rounded hardware
        // sqrt + divide for full f64 precision (~1 ulp), matching the scalar path.
        Avx2F64Vec(_mm256_div_pd(_mm256_set1_pd(1.0), _mm256_sqrt_pd(a.0)))
    }

    // SAFETY: caller must ensure the target CPU supports `avx2` (enforced by the `#[target_feature]` gate above plus runtime `is_x86_feature_detected!` selection in the hermes-simd dispatcher (`target.rs`/`lib.rs`)); any pointer operands are valid for the 4-lane vector width within caller-validated bounds.
    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn floor(a: Self::Vector) -> Self::Vector {
        Avx2F64Vec(_mm256_floor_pd(a.0))
    }

    // SAFETY: caller must ensure the target CPU supports `avx2` (enforced by the `#[target_feature]` gate above plus runtime `is_x86_feature_detected!` selection in the hermes-simd dispatcher (`target.rs`/`lib.rs`)); any pointer operands are valid for the 4-lane vector width within caller-validated bounds.
    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn ceil(a: Self::Vector) -> Self::Vector {
        Avx2F64Vec(_mm256_ceil_pd(a.0))
    }

    // SAFETY: caller must ensure the target CPU supports `avx2` (enforced by the `#[target_feature]` gate above plus runtime `is_x86_feature_detected!` selection in the hermes-simd dispatcher (`target.rs`/`lib.rs`)); any pointer operands are valid for the 4-lane vector width within caller-validated bounds.
    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn round(a: Self::Vector) -> Self::Vector {
        // `vroundpd` in `_MM_FROUND_TO_NEAREST_INT` mode (imm low bits 00) rounds
        // ties to the even neighbor, matching the scalar `round_ties_even` contract.
        // `_MM_FROUND_NO_EXC` (imm bit 3) suppresses the per-element inexact exception.
        Avx2F64Vec(_mm256_round_pd::<
            { _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC },
        >(a.0))
    }

    // SAFETY: caller must ensure the target CPU supports `avx2` (enforced by the `#[target_feature]` gate above plus runtime `is_x86_feature_detected!` selection in the hermes-simd dispatcher (`target.rs`/`lib.rs`)); any pointer operands are valid for the 4-lane vector width within caller-validated bounds.
    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn trunc(a: Self::Vector) -> Self::Vector {
        // `vroundpd` in `_MM_FROUND_TO_ZERO` mode (imm low bits 11) rounds toward zero.
        Avx2F64Vec(_mm256_round_pd::<{ _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC }>(a.0))
    }

    // SAFETY: caller must ensure the target CPU supports `avx2` (enforced by the `#[target_feature]` gate above plus runtime `is_x86_feature_detected!` selection in the hermes-simd dispatcher (`target.rs`/`lib.rs`)); any pointer operands are valid for the 4-lane vector width within caller-validated bounds.
    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn popcount(a: Self::Vector) -> Self::Vector {
        use core::arch::x86_64::{
            _mm256_add_epi8, _mm256_and_si256, _mm256_castpd_si256, _mm256_castsi256_si128,
            _mm256_cvtepi32_pd, _mm256_extractf128_si256, _mm256_sad_epu8, _mm256_set1_epi8,
            _mm256_setr_epi8, _mm256_setzero_si256, _mm256_shuffle_epi32, _mm256_shuffle_epi8,
            _mm256_srli_epi16, _mm_unpacklo_epi64,
        };
        let v = _mm256_castpd_si256(a.0);
        let low_mask = _mm256_set1_epi8(0x0F);
        let lookup = _mm256_setr_epi8(
            0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4, 0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2,
            3, 3, 4,
        );
        let lo = _mm256_and_si256(v, low_mask);
        let hi = _mm256_and_si256(_mm256_srli_epi16(v, 4), low_mask);
        let pop_lo = _mm256_shuffle_epi8(lookup, lo);
        let pop_hi = _mm256_shuffle_epi8(lookup, hi);
        let pop_bytes = _mm256_add_epi8(pop_lo, pop_hi);
        let pop_u64 = _mm256_sad_epu8(pop_bytes, _mm256_setzero_si256());
        let shuffled = _mm256_shuffle_epi32(pop_u64, 0xD8);
        let low = _mm256_castsi256_si128(shuffled);
        let high = _mm256_extractf128_si256(shuffled, 1);
        let packed = _mm_unpacklo_epi64(low, high);
        let pop_f64 = _mm256_cvtepi32_pd(packed);
        Avx2F64Vec(pop_f64)
    }

    // SAFETY: caller must ensure the target CPU supports `avx2` (enforced by the `#[target_feature]` gate above plus runtime `is_x86_feature_detected!` selection in the hermes-simd dispatcher (`target.rs`/`lib.rs`)); any pointer operands are valid for the 4-lane vector width within caller-validated bounds.
    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn cmp_eq(a: Self::Vector, b: Self::Vector) -> Self::Vector {
        Avx2F64Vec(_mm256_cmp_pd(a.0, b.0, _CMP_EQ_OQ))
    }

    // SAFETY: caller must ensure the target CPU supports `avx2` (enforced by the `#[target_feature]` gate above plus runtime `is_x86_feature_detected!` selection in the hermes-simd dispatcher (`target.rs`/`lib.rs`)); any pointer operands are valid for the 4-lane vector width within caller-validated bounds.
    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn cmp_ne(a: Self::Vector, b: Self::Vector) -> Self::Vector {
        Avx2F64Vec(_mm256_cmp_pd(a.0, b.0, _CMP_NEQ_UQ))
    }

    // SAFETY: caller must ensure the target CPU supports `avx2` (enforced by the `#[target_feature]` gate above plus runtime `is_x86_feature_detected!` selection in the hermes-simd dispatcher (`target.rs`/`lib.rs`)); any pointer operands are valid for the 4-lane vector width within caller-validated bounds.
    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn cmp_lt(a: Self::Vector, b: Self::Vector) -> Self::Vector {
        Avx2F64Vec(_mm256_cmp_pd(a.0, b.0, _CMP_LT_OQ))
    }

    // SAFETY: caller must ensure the target CPU supports `avx2` (enforced by the `#[target_feature]` gate above plus runtime `is_x86_feature_detected!` selection in the hermes-simd dispatcher (`target.rs`/`lib.rs`)); any pointer operands are valid for the 4-lane vector width within caller-validated bounds.
    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn cmp_le(a: Self::Vector, b: Self::Vector) -> Self::Vector {
        Avx2F64Vec(_mm256_cmp_pd(a.0, b.0, _CMP_LE_OQ))
    }

    // SAFETY: caller must ensure the target CPU supports `avx2` (enforced by the `#[target_feature]` gate above plus runtime `is_x86_feature_detected!` selection in the hermes-simd dispatcher (`target.rs`/`lib.rs`)); any pointer operands are valid for the 4-lane vector width within caller-validated bounds.
    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn cmp_gt(a: Self::Vector, b: Self::Vector) -> Self::Vector {
        Avx2F64Vec(_mm256_cmp_pd(a.0, b.0, _CMP_GT_OQ))
    }

    // SAFETY: caller must ensure the target CPU supports `avx2` (enforced by the `#[target_feature]` gate above plus runtime `is_x86_feature_detected!` selection in the hermes-simd dispatcher (`target.rs`/`lib.rs`)); any pointer operands are valid for the 4-lane vector width within caller-validated bounds.
    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn cmp_ge(a: Self::Vector, b: Self::Vector) -> Self::Vector {
        Avx2F64Vec(_mm256_cmp_pd(a.0, b.0, _CMP_GE_OQ))
    }

    // SAFETY: caller must ensure the target CPU supports `avx2` (enforced by the `#[target_feature]` gate above plus runtime `is_x86_feature_detected!` selection in the hermes-simd dispatcher (`target.rs`/`lib.rs`)); any pointer operands are valid for the 4-lane vector width within caller-validated bounds.
    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn blend(
        mask: Self::Vector,
        true_val: Self::Vector,
        false_val: Self::Vector,
    ) -> Self::Vector {
        Avx2F64Vec(_mm256_blendv_pd(false_val.0, true_val.0, mask.0))
    }

    // SAFETY: caller must ensure the target CPU supports `avx2` (enforced by the `#[target_feature]` gate above plus runtime `is_x86_feature_detected!` selection in the hermes-simd dispatcher (`target.rs`/`lib.rs`)); any pointer operands are valid for the 4-lane vector width within caller-validated bounds.
    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn mask_to_bitmask(mask: Self::Mask) -> u64 {
        _mm256_movemask_pd(mask.0) as u64
    }

    // SAFETY: caller must ensure the target CPU supports `avx2` (enforced by the `#[target_feature]` gate above plus runtime `is_x86_feature_detected!` selection in the hermes-simd dispatcher (`target.rs`/`lib.rs`)); any pointer operands are valid for the 4-lane vector width within caller-validated bounds.
    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn mask_to_vector(mask: Self::Mask) -> Self::Vector {
        Avx2F64Vec(mask.0)
    }

    // SAFETY: caller must ensure the target CPU supports `avx2` (enforced by the `#[target_feature]` gate above plus runtime `is_x86_feature_detected!` selection in the hermes-simd dispatcher (`target.rs`/`lib.rs`)); this is a register-to-register reinterpretation with no memory operands.
    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn vector_to_mask(v: Self::Vector) -> Self::Mask {
        // Mask and vector share the `__m256d` representation, and the
        // reinterpretation preserves lane sign bits for `_mm256_movemask_pd`.
        Avx2F64Mask(v.0)
    }
}
