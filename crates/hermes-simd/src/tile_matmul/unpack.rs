#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::{
    __m128i, __m256i, __m512i, _mm256_cvtepi8_epi16, _mm256_cvtepi8_epi32, _mm256_loadu_si256,
    _mm256_storeu_si256, _mm512_cvtepi8_epi16, _mm512_cvtepi8_epi32, _mm512_storeu_si512,
    _mm_loadl_epi64, _mm_loadu_si128,
};

#[cfg(target_arch = "aarch64")]
use core::arch::aarch64::*;

/// Unpacks packed 4-bit signed integers (stored 2 per byte) into an 8-bit signed integer slice.
#[inline]
pub fn unpack_int4(packed: &[u8], unpacked: &mut [i8]) {
    #[cfg(target_arch = "x86_64")]
    {
        hermes_simd_intrinsics::x86_64::avx512_tiling::unpack_int4(packed, unpacked);
    }
    #[cfg(target_arch = "aarch64")]
    {
        let len = packed.len();
        assert!(unpacked.len() >= len * 2);
        let mut i = 0;
        unsafe {
            let mask_0f = vdup_n_u8(0x0F);
            while i + 8 <= len {
                let ptr = packed.as_ptr().add(i);
                let v = vld1_u8(ptr);

                let low_nibbles = vand_u8(v, mask_0f);
                let high_nibbles = vshr_n_u8(v, 4);

                let low_signed = vshl_n_s8(vreinterpret_s8_u8(low_nibbles), 4);
                let low_extended = vshr_n_s8(low_signed, 4);

                let high_signed = vshl_n_s8(vreinterpret_s8_u8(high_nibbles), 4);
                let high_extended = vshr_n_s8(high_signed, 4);

                let zipped = vzip_s8(low_extended, high_extended);

                vst1_s8(unpacked.as_mut_ptr().add(2 * i), zipped.0);
                vst1_s8(unpacked.as_mut_ptr().add(2 * i + 8), zipped.1);

                i += 8;
            }
        }
        for j in i..len {
            let byte = packed[j] as i8;
            unpacked[2 * j] = (byte << 4) >> 4;
            unpacked[2 * j + 1] = byte >> 4;
        }
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        let len = packed.len();
        assert!(unpacked.len() >= len * 2);
        for i in 0..len {
            let byte = packed[i] as i8;
            unpacked[2 * i] = (byte << 4) >> 4;
            unpacked[2 * i + 1] = byte >> 4;
        }
    }
}

/// Widens a slice of `i8` values to `i16` using sign-extension.
#[inline]
pub fn widen_i8_to_i16(src: &[i8], dest: &mut [i16]) {
    let len = src.len().min(dest.len());
    let mut i = 0;

    #[cfg(target_arch = "x86_64")]
    {
        // `_mm512_cvtepi8_epi16` (`vpmovsxbw`) is an AVX-512**BW** instruction, not
        // AVX-512F. `TargetId::Avx512` only detects `avx512f`, so gating on it
        // would `#UD` on an AVX-512F-without-BW part (e.g. Knights Landing). Detect
        // `avx512bw` (which implies `avx512f` on every real CPU) directly.
        #[cfg(feature = "std")]
        let avx512bw = std::is_x86_feature_detected!("avx512bw");
        #[cfg(not(feature = "std"))]
        let avx512bw = cfg!(target_feature = "avx512bw");
        if avx512bw {
            while i + 32 <= len {
                unsafe {
                    #[expect(
                        clippy::cast_ptr_alignment,
                        reason = "_mm256_loadu_si256 accepts the deliberately unaligned source"
                    )]
                    let src_ptr = src.as_ptr().add(i).cast::<__m256i>();
                    let a = _mm256_loadu_si256(src_ptr);
                    let res = _mm512_cvtepi8_epi16(a);
                    #[expect(
                        clippy::cast_ptr_alignment,
                        reason = "_mm512_storeu_si512 accepts the deliberately unaligned output"
                    )]
                    let dest_ptr = dest.as_mut_ptr().add(i).cast::<__m512i>();
                    _mm512_storeu_si512(dest_ptr, res);
                }
                i += 32;
            }
        }
        if crate::target::TargetId::Avx2.is_supported() {
            while i + 16 <= len {
                unsafe {
                    let src_ptr = src.as_ptr().add(i);
                    #[expect(
                        clippy::cast_ptr_alignment,
                        reason = "_mm_loadu_si128 accepts the deliberately unaligned source"
                    )]
                    let a = _mm_loadu_si128(src_ptr.cast::<__m128i>());
                    let res = _mm256_cvtepi8_epi16(a);
                    #[expect(
                        clippy::cast_ptr_alignment,
                        reason = "_mm256_storeu_si256 accepts the deliberately unaligned output"
                    )]
                    let dest_ptr = dest.as_mut_ptr().add(i).cast::<__m256i>();
                    _mm256_storeu_si256(dest_ptr, res);
                }
                i += 16;
            }
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        if crate::target::TargetId::Neon.is_supported() {
            while i + 16 <= len {
                unsafe {
                    let src_ptr = src.as_ptr().add(i);
                    let a = vld1q_s8(src_ptr);
                    let low = vget_low_s8(a);
                    let high = vget_high_s8(a);
                    let res_low = vmovl_s8(low);
                    let res_high = vmovl_s8(high);
                    let dest_ptr = dest.as_mut_ptr().add(i);
                    vst1q_s16(dest_ptr, res_low);
                    vst1q_s16(dest_ptr.add(8), res_high);
                }
                i += 16;
            }
        }
    }

    while i < len {
        dest[i] = i16::from(src[i]);
        i += 1;
    }
}

/// Widens a slice of `i8` values to `i32` using sign-extension.
#[inline]
pub fn widen_i8_to_i32(src: &[i8], dest: &mut [i32]) {
    let len = src.len().min(dest.len());
    let mut i = 0;

    #[cfg(target_arch = "x86_64")]
    {
        if crate::target::TargetId::Avx512.is_supported() {
            while i + 16 <= len {
                unsafe {
                    let src_ptr = src.as_ptr().add(i);
                    #[expect(
                        clippy::cast_ptr_alignment,
                        reason = "_mm_loadu_si128 accepts the deliberately unaligned source"
                    )]
                    let a = _mm_loadu_si128(src_ptr.cast::<__m128i>());
                    let res = _mm512_cvtepi8_epi32(a);
                    #[expect(
                        clippy::cast_ptr_alignment,
                        reason = "_mm512_storeu_si512 accepts the deliberately unaligned output"
                    )]
                    let dest_ptr = dest.as_mut_ptr().add(i).cast::<__m512i>();
                    _mm512_storeu_si512(dest_ptr, res);
                }
                i += 16;
            }
        }
        if crate::target::TargetId::Avx2.is_supported() {
            while i + 8 <= len {
                unsafe {
                    let src_ptr = src.as_ptr().add(i);
                    #[expect(
                        clippy::cast_ptr_alignment,
                        reason = "_mm_loadl_epi64 accepts the deliberately unaligned source"
                    )]
                    let a = _mm_loadl_epi64(src_ptr.cast::<__m128i>());
                    let res = _mm256_cvtepi8_epi32(a);
                    #[expect(
                        clippy::cast_ptr_alignment,
                        reason = "_mm256_storeu_si256 accepts the deliberately unaligned output"
                    )]
                    let dest_ptr = dest.as_mut_ptr().add(i).cast::<__m256i>();
                    _mm256_storeu_si256(dest_ptr, res);
                }
                i += 8;
            }
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        if crate::target::TargetId::Neon.is_supported() {
            while i + 8 <= len {
                unsafe {
                    let src_ptr = src.as_ptr().add(i);
                    let a = vld1_s8(src_ptr);
                    let mid = vmovl_s8(a);
                    let low = vget_low_s16(mid);
                    let high = vget_high_s16(mid);
                    let res_low = vmovl_s16(low);
                    let res_high = vmovl_s16(high);
                    let dest_ptr = dest.as_mut_ptr().add(i);
                    vst1q_s32(dest_ptr, res_low);
                    vst1q_s32(dest_ptr.add(4), res_high);
                }
                i += 8;
            }
        }
    }

    while i < len {
        dest[i] = i32::from(src[i]);
        i += 1;
    }
}

/// Widens a slice of `crate::I8` wrapper values to `crate::I16` using sign-extension.
#[inline]
#[expect(
    non_snake_case,
    reason = "Public wrappers preserve the numeric domain type spelling"
)]
pub fn widen_I8_to_I16(src: &[crate::I8], dest: &mut [crate::I16]) {
    unsafe {
        let src_cast = core::slice::from_raw_parts(src.as_ptr().cast::<i8>(), src.len());
        let dest_cast =
            core::slice::from_raw_parts_mut(dest.as_mut_ptr().cast::<i16>(), dest.len());
        widen_i8_to_i16(src_cast, dest_cast);
    }
}

/// Widens a slice of `crate::I8` wrapper values to `crate::I32` using sign-extension.
#[inline]
#[expect(
    non_snake_case,
    reason = "Public wrappers preserve the numeric domain type spelling"
)]
pub fn widen_I8_to_I32(src: &[crate::I8], dest: &mut [crate::I32]) {
    unsafe {
        let src_cast = core::slice::from_raw_parts(src.as_ptr().cast::<i8>(), src.len());
        let dest_cast =
            core::slice::from_raw_parts_mut(dest.as_mut_ptr().cast::<i32>(), dest.len());
        widen_i8_to_i32(src_cast, dest_cast);
    }
}
