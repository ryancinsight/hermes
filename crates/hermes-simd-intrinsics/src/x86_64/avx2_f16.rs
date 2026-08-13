//! AVX2 f16 SIMD kernel: F16C hardware-conversion arithmetic core with a
//! documented software fallback.
//!
//! AVX2 lacks native f16 vector *arithmetic*; x86 provides only the f16↔f32
//! conversions (`vcvtph2ps`/`vcvtps2ph`, the F16C feature). Eunomia `F16`
//! scalar arithmetic is definitionally "convert to f32 → operate → round back
//! to f16" per operation, and F16C performs those exact IEEE round-to-nearest
//! conversions in hardware — so the hot arithmetic methods (`add`/`sub`/`mul`/
//! `fmadd`) execute convert→AVX-op→convert with results **identical** to the
//! software path on all numeric values (NaN payload bits follow the hardware
//! quieting convention, as with every native backend). Each method probes F16C
//! once via the cached `is_x86_feature_detected!` (compile-time `cfg!` under
//! `no_std`) and falls back to the per-lane software loop when absent, so the
//! kernel stays sound on an AVX2-without-F16C host.
//!
//! Everything outside the arithmetic core (loads, masks, gather, compress) is
//! conversion-free and remains the plain 16-lane array form.

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
use crate::Avx2;
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
use eunomia::F16;
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
use hermes_simd_core::kernel::SimdKernel;

/// True when the F16C + FMA hardware-conversion path may be entered.
///
/// Under `std` this is the cached runtime probe (a relaxed atomic load after
/// first use — negligible against a 16-lane operation); without `std` it falls
/// back to the compile-time target-feature state, mirroring the dispatch
/// macro's no-std arm. FMA is probed together with F16C because `fmadd` fuses
/// in f32; on every shipping CPU the two coexist (both are AVX2-era features).
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[inline(always)]
fn f16c_fma_available() -> bool {
    #[cfg(feature = "std")]
    {
        std::is_x86_feature_detected!("f16c") && std::is_x86_feature_detected!("fma")
    }
    #[cfg(not(feature = "std"))]
    {
        cfg!(all(target_feature = "f16c", target_feature = "fma"))
    }
}

/// F16C hardware-conversion inner kernels.
///
/// Each converts the 16 f16 lanes to two 8-lane f32 registers (`vcvtph2ps`),
/// applies the AVX op, and rounds back (`vcvtps2ph`, round-to-nearest-even —
/// the same rounding `F16::from_f32` implements in software).
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod f16c {
    use eunomia::F16;

    #[cfg(target_arch = "x86")]
    use core::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use core::arch::x86_64::*;

    // `vcvtps2ph` immediate: bits 1:0 = rounding (00 = nearest-even), bit 2 =
    // use-MXCSR (0 = use the immediate). `_MM_FROUND_NO_EXC` is not accepted —
    // the intrinsic's immediate range is the 3-bit field only.
    const ROUND_NEAREST: i32 = _MM_FROUND_TO_NEAREST_INT;

    /// # Safety
    /// Caller must ensure `avx` and `f16c` are supported (enforced by the
    /// `f16c_fma_available` probe at every call site).
    #[inline]
    #[target_feature(enable = "avx,f16c")]
    unsafe fn to_f32_halves(v: &[F16; 16]) -> (__m256, __m256) {
        let p = v.as_ptr().cast::<__m128i>();
        (
            _mm256_cvtph_ps(_mm_loadu_si128(p)),
            _mm256_cvtph_ps(_mm_loadu_si128(p.add(1))),
        )
    }

    /// # Safety
    /// Same ISA precondition as [`to_f32_halves`].
    #[inline]
    #[target_feature(enable = "avx,f16c")]
    unsafe fn from_f32_halves(lo: __m256, hi: __m256) -> [F16; 16] {
        let mut out = [F16::ZERO; 16];
        let p = out.as_mut_ptr().cast::<__m128i>();
        _mm_storeu_si128(p, _mm256_cvtps_ph::<ROUND_NEAREST>(lo));
        _mm_storeu_si128(p.add(1), _mm256_cvtps_ph::<ROUND_NEAREST>(hi));
        out
    }

    /// # Safety
    /// Caller must ensure `avx` + `f16c` support.
    #[inline]
    #[target_feature(enable = "avx,f16c")]
    pub(super) unsafe fn add(a: [F16; 16], b: [F16; 16]) -> [F16; 16] {
        let (al, ah) = to_f32_halves(&a);
        let (bl, bh) = to_f32_halves(&b);
        from_f32_halves(_mm256_add_ps(al, bl), _mm256_add_ps(ah, bh))
    }

    /// # Safety
    /// Caller must ensure `avx` + `f16c` support.
    #[inline]
    #[target_feature(enable = "avx,f16c")]
    pub(super) unsafe fn sub(a: [F16; 16], b: [F16; 16]) -> [F16; 16] {
        let (al, ah) = to_f32_halves(&a);
        let (bl, bh) = to_f32_halves(&b);
        from_f32_halves(_mm256_sub_ps(al, bl), _mm256_sub_ps(ah, bh))
    }

    /// # Safety
    /// Caller must ensure `avx` + `f16c` support.
    #[inline]
    #[target_feature(enable = "avx,f16c")]
    pub(super) unsafe fn mul(a: [F16; 16], b: [F16; 16]) -> [F16; 16] {
        let (al, ah) = to_f32_halves(&a);
        let (bl, bh) = to_f32_halves(&b);
        from_f32_halves(_mm256_mul_ps(al, bl), _mm256_mul_ps(ah, bh))
    }

    /// Fused `a*b + c` in f32 (single rounding, like the software path's
    /// `f32::mul_add`), then one rounding to f16.
    ///
    /// # Safety
    /// Caller must ensure `avx` + `f16c` + `fma` support.
    #[inline]
    #[target_feature(enable = "avx,f16c,fma")]
    pub(super) unsafe fn fmadd(a: [F16; 16], b: [F16; 16], c: [F16; 16]) -> [F16; 16] {
        let (al, ah) = to_f32_halves(&a);
        let (bl, bh) = to_f32_halves(&b);
        let (cl, ch) = to_f32_halves(&c);
        from_f32_halves(_mm256_fmadd_ps(al, bl, cl), _mm256_fmadd_ps(ah, bh, ch))
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
impl SimdKernel<F16> for Avx2 {
    type Vector = [F16; 16];
    type Mask = [bool; 16];
    type IndexVector = [i32; 16];
    const LANE_COUNT: usize = 16;
    const UNROLL_FACTOR: usize = 4;

    #[inline(always)]
    unsafe fn load_aligned(ptr: *const F16) -> Self::Vector {
        let mut v = [F16::ZERO; 16];
        core::ptr::copy_nonoverlapping(ptr, v.as_mut_ptr(), 16);
        v
    }

    #[inline(always)]
    unsafe fn load_unaligned(ptr: *const F16) -> Self::Vector {
        let mut v = [F16::ZERO; 16];
        core::ptr::copy_nonoverlapping(ptr, v.as_mut_ptr(), 16);
        v
    }

    #[inline(always)]
    unsafe fn store_aligned(ptr: *mut F16, val: Self::Vector) {
        core::ptr::copy_nonoverlapping(val.as_ptr(), ptr, 16);
    }

    #[inline(always)]
    unsafe fn store_unaligned(ptr: *mut F16, val: Self::Vector) {
        core::ptr::copy_nonoverlapping(val.as_ptr(), ptr, 16);
    }

    #[inline(always)]
    unsafe fn add(a: Self::Vector, b: Self::Vector) -> Self::Vector {
        if f16c_fma_available() {
            // SAFETY: probe confirmed `f16c` (and `fma`) on this host.
            return f16c::add(a, b);
        }
        core::array::from_fn(|i| a[i] + b[i])
    }

    #[inline(always)]
    unsafe fn mul(a: Self::Vector, b: Self::Vector) -> Self::Vector {
        if f16c_fma_available() {
            // SAFETY: probe confirmed `f16c` (and `fma`) on this host.
            return f16c::mul(a, b);
        }
        core::array::from_fn(|i| a[i] * b[i])
    }

    #[inline(always)]
    unsafe fn sub(a: Self::Vector, b: Self::Vector) -> Self::Vector {
        if f16c_fma_available() {
            // SAFETY: probe confirmed `f16c` (and `fma`) on this host.
            return f16c::sub(a, b);
        }
        core::array::from_fn(|i| a[i] - b[i])
    }

    #[inline(always)]
    unsafe fn fmadd(a: Self::Vector, b: Self::Vector, c: Self::Vector) -> Self::Vector {
        if f16c_fma_available() {
            // SAFETY: probe confirmed `f16c` and `fma` on this host.
            return f16c::fmadd(a, b, c);
        }
        core::array::from_fn(|i| F16::from_f32(a[i].to_f32().mul_add(b[i].to_f32(), c[i].to_f32())))
    }

    #[inline(always)]
    unsafe fn sum_reduce(v: Self::Vector) -> F16 {
        v.iter().copied().fold(F16::ZERO, |acc, x| acc + x)
    }

    #[inline(always)]
    unsafe fn masked_load_unaligned(
        ptr: *const F16,
        mask: Self::Mask,
        src: Self::Vector,
    ) -> Self::Vector {
        core::array::from_fn(|i| if mask[i] { *ptr.add(i) } else { src[i] })
    }

    #[inline(always)]
    unsafe fn masked_store_unaligned(ptr: *mut F16, mask: Self::Mask, val: Self::Vector) {
        for i in 0..16 {
            if mask[i] {
                *ptr.add(i) = val[i];
            }
        }
    }

    #[inline(always)]
    unsafe fn masked_add(
        a: Self::Vector,
        b: Self::Vector,
        mask: Self::Mask,
        src: Self::Vector,
    ) -> Self::Vector {
        core::array::from_fn(|i| if mask[i] { a[i] + b[i] } else { src[i] })
    }

    #[inline(always)]
    unsafe fn masked_mul(
        a: Self::Vector,
        b: Self::Vector,
        mask: Self::Mask,
        src: Self::Vector,
    ) -> Self::Vector {
        core::array::from_fn(|i| if mask[i] { a[i] * b[i] } else { src[i] })
    }

    #[inline(always)]
    unsafe fn masked_fmadd(
        a: Self::Vector,
        b: Self::Vector,
        c: Self::Vector,
        mask: Self::Mask,
    ) -> Self::Vector {
        core::array::from_fn(|i| {
            if mask[i] {
                F16::from_f32(a[i].to_f32().mul_add(b[i].to_f32(), c[i].to_f32()))
            } else {
                c[i]
            }
        })
    }

    #[inline(always)]
    unsafe fn masked_sum_reduce(v: Self::Vector, mask: Self::Mask) -> F16 {
        let mut s = F16::ZERO;
        for i in 0..16 {
            if mask[i] {
                s += v[i];
            }
        }
        s
    }

    #[inline(always)]
    unsafe fn compress(src: Self::Vector, mask: Self::Mask) -> Self::Vector {
        let mut out = [F16::ZERO; 16];
        let mut k = 0;
        for i in 0..16 {
            if mask[i] {
                out[k] = src[i];
                k += 1;
            }
        }
        out
    }

    #[inline(always)]
    unsafe fn expand(src: Self::Vector, mask: Self::Mask, fill: Self::Vector) -> Self::Vector {
        let mut out = fill;
        let mut k = 0;
        for i in 0..16 {
            if mask[i] {
                out[i] = src[k];
                k += 1;
            }
        }
        out
    }

    #[inline(always)]
    unsafe fn gather(base: *const F16, indices: Self::IndexVector) -> Self::Vector {
        core::array::from_fn(|i| *base.add(indices[i] as usize))
    }

    #[inline(always)]
    unsafe fn gather_masked(
        base: *const F16,
        indices: Self::IndexVector,
        mask: Self::Mask,
        src: Self::Vector,
    ) -> Self::Vector {
        core::array::from_fn(|i| {
            if mask[i] {
                *base.add(indices[i] as usize)
            } else {
                src[i]
            }
        })
    }

    #[inline(always)]
    unsafe fn mask_from_bools(bits: &[bool]) -> Self::Mask {
        debug_assert_eq!(bits.len(), 16);
        core::array::from_fn(|i| bits[i])
    }

    #[inline(always)]
    unsafe fn leading_k_mask(k: usize) -> Self::Mask {
        core::array::from_fn(|i| i < k)
    }

    #[inline(always)]
    unsafe fn zero() -> Self::Vector {
        [F16::ZERO; 16]
    }

    #[inline(always)]
    unsafe fn splat(val: F16) -> Self::Vector {
        [val; 16]
    }

    #[inline(always)]
    unsafe fn mask_to_bitmask(mask: Self::Mask) -> u64 {
        let mut m = 0u64;
        for i in 0..16 {
            if mask[i] {
                m |= 1u64 << i;
            }
        }
        m
    }

    #[inline(always)]
    unsafe fn mask_to_vector(mask: Self::Mask) -> Self::Vector {
        core::array::from_fn(|i| {
            if mask[i] {
                F16::from_bits(0xFFFF)
            } else {
                F16::ZERO
            }
        })
    }

    #[inline(always)]
    unsafe fn vector_to_mask(v: Self::Vector) -> Self::Mask {
        // Bit 15 is the sign bit of a binary16 lane; testing it rather than
        // comparing against `F16::ZERO` keeps the all-ones comparison result
        // (a NaN bit pattern) from failing a floating-point equality test.
        core::array::from_fn(|i| (v[i].to_bits() >> 15) != 0)
    }
}

#[cfg(all(test, target_arch = "x86_64"))]
mod tests {
    use super::*;

    /// Adversarial 16-lane operand pair: subnormals, zero/negative-zero,
    /// max-normal (overflow candidates), min-subnormal (underflow candidates),
    /// round-to-even tie candidates, and mixed signs.
    fn operands() -> ([F16; 16], [F16; 16]) {
        let bits_a: [u16; 16] = [
            0x0000, 0x8000, 0x0001, 0x03FF, 0x0400, 0x7BFF, 0xFBFF, 0x3C00, 0x3C01, 0x4248, 0xC248,
            0x0002, 0x7800, 0xF800, 0x1000, 0x5640,
        ];
        let bits_b: [u16; 16] = [
            0x3C00, 0x3C00, 0x0001, 0x0002, 0x7BFF, 0x7BFF, 0x3800, 0x3C01, 0x3C01, 0x4248, 0x4248,
            0x03FF, 0x7800, 0x3400, 0x9000, 0xD640,
        ];
        (bits_a.map(F16::from_bits), bits_b.map(F16::from_bits))
    }

    /// The F16C hardware arithmetic must be bitwise-equal to Eunomia's
    /// software convert→f32-op→round-back semantics on every finite pattern,
    /// including subnormals, overflow to infinity, and ties. Skips (with a
    /// notice) only on a host without F16C.
    #[test]
    fn f16c_arithmetic_matches_software_bitwise() {
        if !std::is_x86_feature_detected!("f16c") || !std::is_x86_feature_detected!("fma") {
            eprintln!("skipping: host lacks f16c/fma");
            return;
        }
        let (a, b) = operands();

        // SAFETY: f16c/fma confirmed above; array operands are plain values.
        let (hw_add, hw_sub, hw_mul, hw_fma) = unsafe {
            (
                <Avx2 as SimdKernel<F16>>::add(a, b),
                <Avx2 as SimdKernel<F16>>::sub(a, b),
                <Avx2 as SimdKernel<F16>>::mul(a, b),
                <Avx2 as SimdKernel<F16>>::fmadd(a, b, b),
            )
        };

        for i in 0..16 {
            let sw_add = a[i] + b[i];
            let sw_sub = a[i] - b[i];
            let sw_mul = a[i] * b[i];
            let sw_fma = F16::from_f32(a[i].to_f32().mul_add(b[i].to_f32(), b[i].to_f32()));
            assert_eq!(hw_add[i].to_bits(), sw_add.to_bits(), "add lane {i}");
            assert_eq!(hw_sub[i].to_bits(), sw_sub.to_bits(), "sub lane {i}");
            assert_eq!(hw_mul[i].to_bits(), sw_mul.to_bits(), "mul lane {i}");
            assert_eq!(hw_fma[i].to_bits(), sw_fma.to_bits(), "fmadd lane {i}");
        }
    }

    /// NaN operands must produce NaN (payload bits are the hardware quieting
    /// convention, deliberately not asserted — same contract as the native
    /// f32/f64 backends).
    #[test]
    fn f16c_arithmetic_propagates_nan() {
        if !std::is_x86_feature_detected!("f16c") || !std::is_x86_feature_detected!("fma") {
            eprintln!("skipping: host lacks f16c/fma");
            return;
        }
        let nan = F16::NAN;
        let a = [nan; 16];
        let b = [F16::from_f32(1.0); 16];
        // SAFETY: f16c/fma confirmed above.
        let out = unsafe { <Avx2 as SimdKernel<F16>>::add(a, b) };
        assert!(out.iter().all(|x| x.is_nan()), "NaN must propagate");
    }
}
