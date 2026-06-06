//! Unpacking functions for low-precision data representations.

use crate::types::{Bf8, Bf4, Bf16, F4, F32, F8};

const fn f8_to_f32_bits(bits: u8) -> u32 {
    let sign = (bits & 0x80) as u32;
    let exp = (bits & 0x78) >> 3;
    let mant = bits & 0x07;
    if exp == 0 {
        if mant == 0 {
            sign << 24
        } else {
            if mant >= 4 {
                (sign << 24) | (120 << 23) | (((mant - 4) as u32) << 21)
            } else if mant >= 2 {
                (sign << 24) | (119 << 23) | (((mant - 2) as u32) << 22)
            } else {
                (sign << 24) | (118 << 23)
            }
        }
    } else if exp == 0x0F {
        0x7FC0_0000 | (sign << 24)
    } else {
        let f32_exp = (exp as u32 + 127 - 7) << 23;
        let f32_mant = (mant as u32) << 20;
        (sign << 24) | f32_exp | f32_mant
    }
}

pub(crate) const fn f4_to_f32_bits(bits: u8) -> u32 {
    let bits = bits & 0x0F;
    let sign = (bits & 0x08) as u32;
    let exp = bits & 0x07;
    if exp == 0 {
        sign << 28
    } else if exp == 7 {
        0x7FC0_0000
    } else {
        let f32_exp = (exp as u32 + 127 - 3) << 23;
        (sign << 28) | f32_exp
    }
}

pub(crate) const fn bf4_to_bf16_bits(bits: u8) -> u16 {
    let bits = bits & 0x0F;
    let sign = (bits & 0x08) as u32;
    let exp = (bits & 0x06) >> 1;
    let mant = bits & 0x01;
    let f32_bits = if exp == 0 {
        if mant == 0 {
            sign << 28
        } else {
            if sign != 0 { 0xBE00_0000 } else { 0x3E00_0000 }
        }
    } else if exp == 3 {
        if sign != 0 { 0xFFC0_0000 } else { 0x7FC0_0000 }
    } else {
        let f32_exp = (exp as u32 + 127 - 1) << 23;
        let f32_mant = (mant as u32) << 22;
        (sign << 28) | f32_exp | f32_mant
    };
    (f32_bits >> 16) as u16
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[inline(always)]
fn has_avx512bw() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        #[cfg(feature = "std")]
        {
            std::is_x86_feature_detected!("avx512bw") && std::is_x86_feature_detected!("avx512vl")
        }
        #[cfg(not(feature = "std"))]
        {
            cfg!(target_feature = "avx512bw")
        }
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[inline(always)]
fn has_avx512f() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        #[cfg(feature = "std")]
        {
            std::is_x86_feature_detected!("avx512f") && std::is_x86_feature_detected!("avx512vl")
        }
        #[cfg(not(feature = "std"))]
        {
            cfg!(target_feature = "avx512f")
        }
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[inline(always)]
fn has_avx2() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        #[cfg(feature = "std")]
        {
            std::is_x86_feature_detected!("avx2")
        }
        #[cfg(not(feature = "std"))]
        {
            cfg!(target_feature = "avx2")
        }
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

#[cfg(target_arch = "x86_64")]
#[allow(missing_docs)]
pub mod unsafe_intrinsics {
    pub mod avx2 {
        #![allow(dead_code)]
        use crate::types::{Bf8, Bf4, Bf16, F4, F32, F8};
        use super::super::{bf4_to_bf16_bits, f4_to_f32_bits, f8_to_f32_bits};

        #[target_feature(enable = "avx2")]
        pub unsafe fn unpack_bf8_to_bf16(packed: &[Bf8], unpacked: &mut [Bf16]) {
            let len = packed.len().min(unpacked.len());
            let mut i = 0;
            while i + 16 <= len {
                let ptr = packed.as_ptr().add(i) as *const core::arch::x86_64::__m128i;
                let v_in = core::arch::x86_64::_mm_loadu_si128(ptr);
                let v_u16 = core::arch::x86_64::_mm256_cvtepu8_epi16(v_in);
                let sign = core::arch::x86_64::_mm256_slli_epi16(
                    core::arch::x86_64::_mm256_and_si256(v_u16, core::arch::x86_64::_mm256_set1_epi16(0x80)),
                    8
                );
                let rest = core::arch::x86_64::_mm256_slli_epi16(
                    core::arch::x86_64::_mm256_and_si256(v_u16, core::arch::x86_64::_mm256_set1_epi16(0x7f)),
                    5
                );
                let is_zero = core::arch::x86_64::_mm256_cmpeq_epi16(rest, core::arch::x86_64::_mm256_setzero_si256());
                let bias_diff = core::arch::x86_64::_mm256_andnot_si256(is_zero, core::arch::x86_64::_mm256_set1_epi16(112 << 7));
                let rest_biased = core::arch::x86_64::_mm256_add_epi16(rest, bias_diff);
                let result = core::arch::x86_64::_mm256_or_si256(sign, rest_biased);
                let out_ptr = unpacked.as_mut_ptr().add(i) as *mut core::arch::x86_64::__m256i;
                core::arch::x86_64::_mm256_storeu_si256(out_ptr, result);
                i += 16;
            }
            for j in i..len {
                let b = packed[j].0 as u16;
                let sign = (b & 0x80) << 8;
                let rest = (b & 0x7f) << 5;
                let bias_diff = if rest == 0 { 0 } else { 112 << 7 };
                unpacked[j] = Bf16(half::bf16::from_bits(sign | (rest + bias_diff)));
            }
        }

        #[target_feature(enable = "avx2")]
        pub unsafe fn unpack_bf4_to_bf16(packed: &[Bf4], unpacked: &mut [Bf16]) {
            use core::arch::x86_64::*;
            let len = packed.len().min(unpacked.len());
            let mut i = 0;

            static TABLE_LO: [u8; 16] = {
                let mut t = [0u8; 16];
                let mut idx = 0;
                while idx < 16 {
                    t[idx] = (bf4_to_bf16_bits(idx as u8) & 0xFF) as u8;
                    idx += 1;
                }
                t
            };
            static TABLE_HI: [u8; 16] = {
                let mut t = [0u8; 16];
                let mut idx = 0;
                while idx < 16 {
                    t[idx] = (bf4_to_bf16_bits(idx as u8) >> 8) as u8;
                    idx += 1;
                }
                t
            };

            let table_lo = _mm_loadu_si128(TABLE_LO.as_ptr() as *const _);
            let table_hi = _mm_loadu_si128(TABLE_HI.as_ptr() as *const _);
            let mask_0f = _mm_set1_epi8(0x0F);

            while i + 16 <= len {
                let ptr = packed.as_ptr().add(i) as *const __m128i;
                let v_in = _mm_loadu_si128(ptr);
                let indices = _mm_and_si128(v_in, mask_0f);
                
                let res_lo = _mm_shuffle_epi8(table_lo, indices);
                let res_hi = _mm_shuffle_epi8(table_hi, indices);
                
                let out_lo = _mm_unpacklo_epi8(res_lo, res_hi);
                let out_hi = _mm_unpackhi_epi8(res_lo, res_hi);
                
                _mm_storeu_si128(unpacked.as_mut_ptr().add(i) as *mut _, out_lo);
                _mm_storeu_si128(unpacked.as_mut_ptr().add(i + 8) as *mut _, out_hi);
                
                i += 16;
            }
            for j in i..len {
                let b = packed[j].0;
                unpacked[j] = Bf16(half::bf16::from_bits(bf4_to_bf16_bits(b)));
            }
        }

        #[target_feature(enable = "avx2")]
        pub unsafe fn unpack_bf4_to_bf16_packed(packed: &[u8], unpacked: &mut [Bf16]) {
            use core::arch::x86_64::*;
            let len = packed.len();
            let n = len.min(unpacked.len() / 2);
            let mut i = 0;

            static TABLE_LO: [u8; 16] = {
                let mut t = [0u8; 16];
                let mut idx = 0;
                while idx < 16 {
                    t[idx] = (bf4_to_bf16_bits(idx as u8) & 0xFF) as u8;
                    idx += 1;
                }
                t
            };
            static TABLE_HI: [u8; 16] = {
                let mut t = [0u8; 16];
                let mut idx = 0;
                while idx < 16 {
                    t[idx] = (bf4_to_bf16_bits(idx as u8) >> 8) as u8;
                    idx += 1;
                }
                t
            };

            let table_lo = _mm_loadu_si128(TABLE_LO.as_ptr() as *const _);
            let table_hi = _mm_loadu_si128(TABLE_HI.as_ptr() as *const _);
            let mask_0f = _mm_set1_epi8(0x0F);

            while i + 16 <= n {
                let ptr = packed.as_ptr().add(i) as *const __m128i;
                let v = _mm_loadu_si128(ptr);

                let low_nibbles = _mm_and_si128(v, mask_0f);
                let high_nibbles = _mm_and_si128(_mm_srli_epi16(v, 4), mask_0f);

                let res_lo = _mm_unpacklo_epi8(low_nibbles, high_nibbles);
                let res_hi = _mm_unpackhi_epi8(low_nibbles, high_nibbles);

                let res_lo_lo = _mm_shuffle_epi8(table_lo, res_lo);
                let res_lo_hi = _mm_shuffle_epi8(table_hi, res_lo);
                let out_lo0 = _mm_unpacklo_epi8(res_lo_lo, res_lo_hi);
                let out_lo1 = _mm_unpackhi_epi8(res_lo_lo, res_lo_hi);

                let res_hi_lo = _mm_shuffle_epi8(table_lo, res_hi);
                let res_hi_hi = _mm_shuffle_epi8(table_hi, res_hi);
                let out_hi0 = _mm_unpacklo_epi8(res_hi_lo, res_hi_hi);
                let out_hi1 = _mm_unpackhi_epi8(res_hi_lo, res_hi_hi);

                _mm_storeu_si128(unpacked.as_mut_ptr().add(2 * i) as *mut _, out_lo0);
                _mm_storeu_si128(unpacked.as_mut_ptr().add(2 * i + 8) as *mut _, out_lo1);
                _mm_storeu_si128(unpacked.as_mut_ptr().add(2 * i + 16) as *mut _, out_hi0);
                _mm_storeu_si128(unpacked.as_mut_ptr().add(2 * i + 24) as *mut _, out_hi1);

                i += 16;
            }

            for j in i..n {
                let byte = packed[j];
                unpacked[2 * j] = Bf16(half::bf16::from_bits(bf4_to_bf16_bits(byte & 0x0F)));
                unpacked[2 * j + 1] = Bf16(half::bf16::from_bits(bf4_to_bf16_bits((byte >> 4) & 0x0F)));
            }
        }

        #[target_feature(enable = "avx2")]
        pub unsafe fn unpack_f4_to_f32(packed: &[F4], unpacked: &mut [F32]) {
            let len = packed.len().min(unpacked.len());
            let mut i = 0;

            static TABLE_BITS: [u32; 16] = {
                let mut t = [0u32; 16];
                let mut idx = 0;
                while idx < 16 {
                    t[idx] = f4_to_f32_bits(idx as u8);
                    idx += 1;
                }
                t
            };

            while i + 8 <= len {
                let ptr = packed.as_ptr().add(i) as *const core::arch::x86_64::__m128i;
                let v_in = core::arch::x86_64::_mm_loadl_epi64(ptr);
                let v_u32 = core::arch::x86_64::_mm256_cvtepu8_epi32(v_in);
                let indices = core::arch::x86_64::_mm256_and_si256(v_u32, core::arch::x86_64::_mm256_set1_epi32(0x0f));

                let result = core::arch::x86_64::_mm256_i32gather_ps(
                    TABLE_BITS.as_ptr() as *const _,
                    indices,
                    4
                );

                let out_ptr = unpacked.as_mut_ptr().add(i) as *mut f32;
                core::arch::x86_64::_mm256_storeu_ps(out_ptr, result);
                i += 8;
            }

            for j in i..len {
                unpacked[j] = F32(f32::from_bits(TABLE_BITS[(packed[j].0 & 0x0f) as usize]));
            }
        }

        #[target_feature(enable = "avx2")]
        pub unsafe fn unpack_f4_to_f32_packed(packed: &[u8], unpacked: &mut [F32]) {
            let len = packed.len();
            let n = len.min(unpacked.len() / 2);
            let mut i = 0;

            static TABLE_BITS: [u32; 16] = {
                let mut t = [0u32; 16];
                let mut idx = 0;
                while idx < 16 {
                    t[idx] = f4_to_f32_bits(idx as u8);
                    idx += 1;
                }
                t
            };

            while i + 8 <= n {
                let bytes_ptr = packed.as_ptr().add(i);
                let v_bytes = core::arch::x86_64::_mm_loadl_epi64(bytes_ptr as *const _);
                let v_u32 = core::arch::x86_64::_mm256_cvtepu8_epi32(v_bytes);

                let low_nibbles = core::arch::x86_64::_mm256_and_si256(v_u32, core::arch::x86_64::_mm256_set1_epi32(0x0f));
                let high_nibbles = core::arch::x86_64::_mm256_and_si256(core::arch::x86_64::_mm256_srli_epi32(v_u32, 4), core::arch::x86_64::_mm256_set1_epi32(0x0f));

                let val_low = core::arch::x86_64::_mm256_i32gather_ps(TABLE_BITS.as_ptr() as *const _, low_nibbles, 4);
                let val_high = core::arch::x86_64::_mm256_i32gather_ps(TABLE_BITS.as_ptr() as *const _, high_nibbles, 4);

                let val_lo_u = core::arch::x86_64::_mm256_castps_si256(val_low);
                let val_hi_u = core::arch::x86_64::_mm256_castps_si256(val_high);

                let res0 = core::arch::x86_64::_mm256_unpacklo_epi32(val_lo_u, val_hi_u);
                let res1 = core::arch::x86_64::_mm256_unpackhi_epi32(val_lo_u, val_hi_u);

                let out0 = core::arch::x86_64::_mm256_permute2x128_si256(res0, res1, 0x20);
                let out1 = core::arch::x86_64::_mm256_permute2x128_si256(res0, res1, 0x31);

                let out_ptr0 = unpacked.as_mut_ptr().add(2 * i) as *mut core::arch::x86_64::__m256i;
                core::arch::x86_64::_mm256_storeu_si256(out_ptr0, out0);
                let out_ptr1 = unpacked.as_mut_ptr().add(2 * i + 8) as *mut core::arch::x86_64::__m256i;
                core::arch::x86_64::_mm256_storeu_si256(out_ptr1, out1);

                i += 8;
            }

            for j in i..n {
                let byte = packed[j];
                unpacked[2 * j] = F32(f32::from_bits(TABLE_BITS[(byte & 0x0F) as usize]));
                unpacked[2 * j + 1] = F32(f32::from_bits(TABLE_BITS[((byte >> 4) & 0x0F) as usize]));
            }
        }

        #[target_feature(enable = "avx2")]
        pub unsafe fn unpack_f8_to_f32(packed: &[F8], unpacked: &mut [F32]) {
            let len = packed.len().min(unpacked.len());
            let mut i = 0;

            static TABLE_BITS: [u32; 256] = {
                let mut t = [0u32; 256];
                let mut idx = 0;
                while idx < 256 {
                    t[idx] = f8_to_f32_bits(idx as u8);
                    idx += 1;
                }
                t
            };

            while i + 8 <= len {
                let ptr = packed.as_ptr().add(i) as *const core::arch::x86_64::__m128i;
                let v_in = core::arch::x86_64::_mm_loadl_epi64(ptr);
                let v_u32 = core::arch::x86_64::_mm256_cvtepu8_epi32(v_in);

                let result = core::arch::x86_64::_mm256_i32gather_ps(
                    TABLE_BITS.as_ptr() as *const _,
                    v_u32,
                    4
                );

                let out_ptr = unpacked.as_mut_ptr().add(i) as *mut f32;
                core::arch::x86_64::_mm256_storeu_ps(out_ptr, result);
                i += 8;
            }

            for j in i..len {
                unpacked[j] = F32(f32::from_bits(TABLE_BITS[packed[j].0 as usize]));
            }
        }
    }

    pub mod avx512 {
        #![allow(dead_code)]
        use crate::types::{Bf8, Bf4, Bf16, F4, F32, F8};
        use super::super::{bf4_to_bf16_bits, f4_to_f32_bits, f8_to_f32_bits};
        use core::arch::x86_64::*;

        #[target_feature(enable = "avx512f,avx512bw,avx512vl")]
        pub unsafe fn unpack_bf8_to_bf16(packed: &[Bf8], unpacked: &mut [Bf16]) {
            let len = packed.len().min(unpacked.len());
            let mut i = 0;

            while i + 32 <= len {
                let ptr = packed.as_ptr().add(i) as *const __m256i;
                let v_in = _mm256_loadu_si256(ptr);
                let v_u16 = _mm512_cvtepu8_epi16(v_in);

                let sign = _mm512_slli_epi16(
                    _mm512_and_si512(v_u16, _mm512_set1_epi16(0x80)),
                    8
                );
                let rest = _mm512_slli_epi16(
                    _mm512_and_si512(v_u16, _mm512_set1_epi16(0x7f)),
                    5
                );

                let is_zero_mask = _mm512_cmpeq_epi16_mask(rest, _mm512_setzero_si512());
                let is_not_zero_mask = !is_zero_mask;

                let rest_biased = _mm512_mask_add_epi16(
                    rest,
                    is_not_zero_mask,
                    rest,
                    _mm512_set1_epi16(112 << 7)
                );

                let result = _mm512_or_si512(sign, rest_biased);

                let out_ptr = unpacked.as_mut_ptr().add(i) as *mut __m512i;
                _mm512_storeu_si512(out_ptr, result);

                i += 32;
            }

            for j in i..len {
                let b = packed[j].0 as u16;
                let sign = (b & 0x80) << 8;
                let rest = (b & 0x7f) << 5;
                let bias_diff = if rest == 0 { 0 } else { 112 << 7 };
                unpacked[j] = Bf16(half::bf16::from_bits(sign | (rest + bias_diff)));
            }
        }

        #[target_feature(enable = "avx512f,avx512bw,avx512vl")]
        pub unsafe fn unpack_bf4_to_bf16(packed: &[Bf4], unpacked: &mut [Bf16]) {
            let len = packed.len().min(unpacked.len());
            let mut i = 0;

            static TABLE_BITS: [u16; 16] = {
                let mut t = [0u16; 16];
                let mut idx = 0;
                while idx < 16 {
                    t[idx] = bf4_to_bf16_bits(idx as u8);
                    idx += 1;
                }
                t
            };
            let table_zmm = core::mem::transmute::<[i16; 32], __m512i>([
                TABLE_BITS[0] as i16, TABLE_BITS[1] as i16, TABLE_BITS[2] as i16, TABLE_BITS[3] as i16,
                TABLE_BITS[4] as i16, TABLE_BITS[5] as i16, TABLE_BITS[6] as i16, TABLE_BITS[7] as i16,
                TABLE_BITS[8] as i16, TABLE_BITS[9] as i16, TABLE_BITS[10] as i16, TABLE_BITS[11] as i16,
                TABLE_BITS[12] as i16, TABLE_BITS[13] as i16, TABLE_BITS[14] as i16, TABLE_BITS[15] as i16,
                TABLE_BITS[0] as i16, TABLE_BITS[1] as i16, TABLE_BITS[2] as i16, TABLE_BITS[3] as i16,
                TABLE_BITS[4] as i16, TABLE_BITS[5] as i16, TABLE_BITS[6] as i16, TABLE_BITS[7] as i16,
                TABLE_BITS[8] as i16, TABLE_BITS[9] as i16, TABLE_BITS[10] as i16, TABLE_BITS[11] as i16,
                TABLE_BITS[12] as i16, TABLE_BITS[13] as i16, TABLE_BITS[14] as i16, TABLE_BITS[15] as i16,
            ]);

            while i + 32 <= len {
                let ptr = packed.as_ptr().add(i) as *const __m256i;
                let v_in = _mm256_loadu_si256(ptr);
                let v_u16 = _mm512_cvtepu8_epi16(v_in);
                let indices = _mm512_and_si512(v_u16, _mm512_set1_epi16(0x0f));
                
                let result = _mm512_permutexvar_epi16(indices, table_zmm);
                
                let out_ptr = unpacked.as_mut_ptr().add(i) as *mut __m512i;
                _mm512_storeu_si512(out_ptr, result);
                
                i += 32;
            }

            for j in i..len {
                let b = packed[j].0;
                unpacked[j] = Bf16(half::bf16::from_bits(bf4_to_bf16_bits(b)));
            }
        }

        #[target_feature(enable = "avx512f,avx512bw,avx512vl")]
        pub unsafe fn unpack_bf4_to_bf16_packed(packed: &[u8], unpacked: &mut [Bf16]) {
            let len = packed.len();
            let n = len.min(unpacked.len() / 2);
            let mut i = 0;

            static TABLE_BITS: [u16; 16] = {
                let mut t = [0u16; 16];
                let mut idx = 0;
                while idx < 16 {
                    t[idx] = bf4_to_bf16_bits(idx as u8);
                    idx += 1;
                }
                t
            };

            let table_ymm = _mm256_loadu_si256(TABLE_BITS.as_ptr() as *const _);
            let mask_0f = _mm_set1_epi8(0x0F);

            while i + 16 <= n {
                let v = _mm_loadu_si128(packed.as_ptr().add(i) as *const _);

                let low_nibbles = _mm_and_si128(v, mask_0f);
                let high_nibbles = _mm_and_si128(_mm_srli_epi16(v, 4), mask_0f);

                let res_lo = _mm_unpacklo_epi8(low_nibbles, high_nibbles);
                let res_hi = _mm_unpackhi_epi8(low_nibbles, high_nibbles);

                let idx_lo = _mm256_cvtepu8_epi16(res_lo);
                let idx_hi = _mm256_cvtepu8_epi16(res_hi);

                let val_lo = _mm256_permutexvar_epi16(idx_lo, table_ymm);
                let val_hi = _mm256_permutexvar_epi16(idx_hi, table_ymm);

                _mm256_storeu_si256(unpacked.as_mut_ptr().add(2 * i) as *mut _, val_lo);
                _mm256_storeu_si256(unpacked.as_mut_ptr().add(2 * i + 16) as *mut _, val_hi);

                i += 16;
            }

            static TABLE: [Bf16; 16] = {
                let mut t = [Bf16(half::bf16::ZERO); 16];
                let mut idx = 0;
                while idx < 16 {
                    t[idx] = Bf16(half::bf16::from_bits(TABLE_BITS[idx]));
                    idx += 1;
                }
                t
            };
            for j in i..n {
                let byte = packed[j];
                unpacked[2 * j] = TABLE[(byte & 0x0F) as usize];
                unpacked[2 * j + 1] = TABLE[((byte >> 4) & 0x0F) as usize];
            }
        }

        #[target_feature(enable = "avx512f,avx512vl")]
        pub unsafe fn unpack_f4_to_f32(packed: &[F4], unpacked: &mut [F32]) {
            let len = packed.len().min(unpacked.len());
            let mut i = 0;

            static TABLE_BITS: [u32; 16] = {
                let mut t = [0u32; 16];
                let mut idx = 0;
                while idx < 16 {
                    t[idx] = f4_to_f32_bits(idx as u8);
                    idx += 1;
                }
                t
            };

            let table_zmm = _mm512_loadu_si512(TABLE_BITS.as_ptr() as *const _);

            while i + 16 <= len {
                let ptr = packed.as_ptr().add(i) as *const __m128i;
                let v_in = _mm_loadu_si128(ptr);
                let v_u32 = _mm512_cvtepu8_epi32(v_in);
                let indices = _mm512_and_si512(v_u32, _mm512_set1_epi32(0x0f));

                let result = _mm512_permutexvar_epi32(indices, table_zmm);

                let out_ptr = unpacked.as_mut_ptr().add(i) as *mut f32;
                _mm512_storeu_si512(out_ptr as *mut _, result);
                i += 16;
            }

            for j in i..len {
                unpacked[j] = F32(f32::from_bits(TABLE_BITS[(packed[j].0 & 0x0f) as usize]));
            }
        }

        #[target_feature(enable = "avx512f,avx512vl")]
        pub unsafe fn unpack_f4_to_f32_packed(packed: &[u8], unpacked: &mut [F32]) {
            let len = packed.len();
            let n = len.min(unpacked.len() / 2);
            let mut i = 0;

            static TABLE_BITS: [u32; 16] = {
                let mut t = [0u32; 16];
                let mut idx = 0;
                while idx < 16 {
                    t[idx] = f4_to_f32_bits(idx as u8);
                    idx += 1;
                }
                t
            };

            let table_zmm = _mm512_loadu_si512(TABLE_BITS.as_ptr() as *const _);
            let mask_0f = _mm_set1_epi8(0x0F);

            while i + 16 <= n {
                let v = _mm_loadu_si128(packed.as_ptr().add(i) as *const _);

                let low_nibbles = _mm_and_si128(v, mask_0f);
                let high_nibbles = _mm_and_si128(_mm_srli_epi16(v, 4), mask_0f);

                let res_lo = _mm_unpacklo_epi8(low_nibbles, high_nibbles);
                let res_hi = _mm_unpackhi_epi8(low_nibbles, high_nibbles);

                let idx_lo = _mm512_cvtepu8_epi32(res_lo);
                let idx_hi = _mm512_cvtepu8_epi32(res_hi);

                let val_lo = _mm512_permutexvar_epi32(idx_lo, table_zmm);
                let val_hi = _mm512_permutexvar_epi32(idx_hi, table_zmm);

                _mm512_storeu_si512(unpacked.as_mut_ptr().add(2 * i) as *mut _, val_lo);
                _mm512_storeu_si512(unpacked.as_mut_ptr().add(2 * i + 16) as *mut _, val_hi);

                i += 16;
            }

            static TABLE: [F32; 16] = {
                let mut t = [F32(0.0); 16];
                let mut idx = 0;
                while idx < 16 {
                    t[idx] = F32(f32::from_bits(TABLE_BITS[idx]));
                    idx += 1;
                }
                t
            };
            for j in i..n {
                let byte = packed[j];
                unpacked[2 * j] = TABLE[(byte & 0x0F) as usize];
                unpacked[2 * j + 1] = TABLE[((byte >> 4) & 0x0F) as usize];
            }
        }

        #[target_feature(enable = "avx512f,avx512vl")]
        pub unsafe fn unpack_f8_to_f32(packed: &[F8], unpacked: &mut [F32]) {
            let len = packed.len().min(unpacked.len());
            let mut i = 0;

            static TABLE_BITS: [u32; 256] = {
                let mut t = [0u32; 256];
                let mut idx = 0;
                while idx < 256 {
                    t[idx] = f8_to_f32_bits(idx as u8);
                    idx += 1;
                }
                t
            };

            while i + 16 <= len {
                let ptr = packed.as_ptr().add(i) as *const __m128i;
                let v_in = _mm_loadu_si128(ptr);
                let v_u32 = _mm512_cvtepu8_epi32(v_in);

                let result = _mm512_i32gather_ps(
                    v_u32,
                    TABLE_BITS.as_ptr() as *const _,
                    4
                );

                let out_ptr = unpacked.as_mut_ptr().add(i) as *mut f32;
                _mm512_storeu_ps(out_ptr, result);
                i += 16;
            }

            for j in i..len {
                unpacked[j] = F32(f32::from_bits(TABLE_BITS[packed[j].0 as usize]));
            }
        }
    }
}

/// Unpacks Bf8 elements to Bf16.
#[inline]
pub fn unpack_bf8_to_bf16(packed: &[Bf8], unpacked: &mut [Bf16]) {
    #[cfg(target_arch = "x86_64")]
    {
        if has_avx512bw() {
            unsafe { unsafe_intrinsics::avx512::unpack_bf8_to_bf16(packed, unpacked); }
            return;
        }
        if has_avx2() {
            unsafe { unsafe_intrinsics::avx2::unpack_bf8_to_bf16(packed, unpacked); }
            return;
        }
    }
    let len = packed.len();
    let n = len.min(unpacked.len());
    for i in 0..n {
        let b = packed[i].0 as u16;
        let sign = (b & 0x80) << 8;
        let rest = (b & 0x7f) << 5;
        let bias_diff = if rest == 0 { 0 } else { 112 << 7 };
        unpacked[i] = Bf16(half::bf16::from_bits(sign | (rest + bias_diff)));
    }
}

/// Unpacks Bf4 elements to Bf16.
#[inline]
pub fn unpack_bf4_to_bf16(packed: &[Bf4], unpacked: &mut [Bf16]) {
    #[cfg(target_arch = "x86_64")]
    {
        if has_avx512bw() {
            unsafe { unsafe_intrinsics::avx512::unpack_bf4_to_bf16(packed, unpacked); }
            return;
        }
        if has_avx2() {
            unsafe { unsafe_intrinsics::avx2::unpack_bf4_to_bf16(packed, unpacked); }
            return;
        }
    }
    let len = packed.len();
    let n = len.min(unpacked.len());
    for i in 0..n {
        let b = packed[i].0;
        unpacked[i] = Bf16(half::bf16::from_bits(bf4_to_bf16_bits(b)));
    }
}

/// Unpacks packed 4-bit Bf4 pairs (stored 2 per byte) into a Bf16 slice.
#[inline]
pub fn unpack_bf4_to_bf16_packed(packed: &[u8], unpacked: &mut [Bf16]) {
    #[cfg(target_arch = "x86_64")]
    {
        if has_avx512bw() {
            unsafe { unsafe_intrinsics::avx512::unpack_bf4_to_bf16_packed(packed, unpacked); }
            return;
        }
        if has_avx2() {
            unsafe { unsafe_intrinsics::avx2::unpack_bf4_to_bf16_packed(packed, unpacked); }
            return;
        }
    }
    let len = packed.len();
    let n = len.min(unpacked.len() / 2);
    for i in 0..n {
        let byte = packed[i];
        let b1 = byte & 0x0f;
        let b2 = (byte >> 4) & 0x0f;
        unpacked[2 * i] = Bf16(half::bf16::from_bits(bf4_to_bf16_bits(b1)));
        unpacked[2 * i + 1] = Bf16(half::bf16::from_bits(bf4_to_bf16_bits(b2)));
    }
}

/// Unpacks F4 elements to F32.
#[inline]
pub fn unpack_f4_to_f32(packed: &[F4], unpacked: &mut [F32]) {
    #[cfg(target_arch = "x86_64")]
    {
        if has_avx512f() {
            unsafe { unsafe_intrinsics::avx512::unpack_f4_to_f32(packed, unpacked); }
            return;
        }
        if has_avx2() {
            unsafe { unsafe_intrinsics::avx2::unpack_f4_to_f32(packed, unpacked); }
            return;
        }
    }
    let len = packed.len().min(unpacked.len());
    for i in 0..len {
        unpacked[i] = F32(packed[i].to_f32());
    }
}

/// Unpacks packed 4-bit F4 pairs (stored 2 per byte) into an F32 slice.
#[inline]
pub fn unpack_f4_to_f32_packed(packed: &[u8], unpacked: &mut [F32]) {
    #[cfg(target_arch = "x86_64")]
    {
        if has_avx512f() {
            unsafe { unsafe_intrinsics::avx512::unpack_f4_to_f32_packed(packed, unpacked); }
            return;
        }
        if has_avx2() {
            unsafe { unsafe_intrinsics::avx2::unpack_f4_to_f32_packed(packed, unpacked); }
            return;
        }
    }
    let len = packed.len();
    let n = len.min(unpacked.len() / 2);
    for i in 0..n {
        let byte = packed[i];
        let b1 = byte & 0x0f;
        let b2 = (byte >> 4) & 0x0f;
        unpacked[2 * i] = F32(F4(b1).to_f32());
        unpacked[2 * i + 1] = F32(F4(b2).to_f32());
    }
}

/// Unpacks F8 elements to F32.
#[inline]
pub fn unpack_f8_to_f32(packed: &[F8], unpacked: &mut [F32]) {
    #[cfg(target_arch = "x86_64")]
    {
        if has_avx512f() {
            unsafe { unsafe_intrinsics::avx512::unpack_f8_to_f32(packed, unpacked); }
            return;
        }
        if has_avx2() {
            unsafe { unsafe_intrinsics::avx2::unpack_f8_to_f32(packed, unpacked); }
            return;
        }
    }
    static TABLE_BITS: [u32; 256] = {
        let mut t = [0u32; 256];
        let mut idx = 0;
        while idx < 256 {
            t[idx] = f8_to_f32_bits(idx as u8);
            idx += 1;
        }
        t
    };
    let len = packed.len().min(unpacked.len());
    for i in 0..len {
        unpacked[i] = F32(f32::from_bits(TABLE_BITS[packed[i].0 as usize]));
    }
}
