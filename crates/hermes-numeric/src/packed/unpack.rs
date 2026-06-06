//! Unpacking functions for low-precision data representations.

use crate::types::{Bf8, Bf4, Bf16, F4, F32};

#[cfg(target_arch = "x86_64")]
mod avx2_unpack {
    #![allow(dead_code)]
    use super::*;

    #[target_feature(enable = "avx2")]
    pub unsafe fn unpack_bf8_to_bf16_avx2(packed: &[Bf8], unpacked: &mut [Bf16]) {
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
    pub unsafe fn unpack_bf4_to_bf16_avx2(packed: &[Bf4], unpacked: &mut [Bf16]) {
        let len = packed.len().min(unpacked.len());
        let mut i = 0;
        while i + 16 <= len {
            let ptr = packed.as_ptr().add(i) as *const core::arch::x86_64::__m128i;
            let v_in = core::arch::x86_64::_mm_loadu_si128(ptr);
            let v_u16 = core::arch::x86_64::_mm256_cvtepu8_epi16(v_in);
            let sign = core::arch::x86_64::_mm256_slli_epi16(
                core::arch::x86_64::_mm256_and_si256(v_u16, core::arch::x86_64::_mm256_set1_epi16(0x08)),
                12
            );
            let rest = core::arch::x86_64::_mm256_slli_epi16(
                core::arch::x86_64::_mm256_and_si256(v_u16, core::arch::x86_64::_mm256_set1_epi16(0x07)),
                6
            );
            let is_zero = core::arch::x86_64::_mm256_cmpeq_epi16(rest, core::arch::x86_64::_mm256_setzero_si256());
            let bias_diff = core::arch::x86_64::_mm256_andnot_si256(is_zero, core::arch::x86_64::_mm256_set1_epi16(126 << 7));
            let rest_biased = core::arch::x86_64::_mm256_add_epi16(rest, bias_diff);
            let result = core::arch::x86_64::_mm256_or_si256(sign, rest_biased);
            let out_ptr = unpacked.as_mut_ptr().add(i) as *mut core::arch::x86_64::__m256i;
            core::arch::x86_64::_mm256_storeu_si256(out_ptr, result);
            i += 16;
        }
        for j in i..len {
            let b = packed[j].0 as u16;
            let sign = (b & 0x08) << 12;
            let rest = (b & 0x07) << 6;
            let bias_diff = if rest == 0 { 0 } else { 126 << 7 };
            unpacked[j] = Bf16(half::bf16::from_bits(sign | (rest + bias_diff)));
        }
    }

    #[target_feature(enable = "avx2")]
    pub unsafe fn unpack_bf4_to_bf16_packed_avx2(packed: &[u8], unpacked: &mut [Bf16]) {
        let len = packed.len();
        let n = len.min(unpacked.len() / 2);
        let mut i = 0;
        while i + 16 <= n {
            let ptr = packed.as_ptr().add(i) as *const core::arch::x86_64::__m128i;
            let v_in = core::arch::x86_64::_mm_loadu_si128(ptr);
            let v_u16 = core::arch::x86_64::_mm256_cvtepu8_epi16(v_in);
            let low_nibble = core::arch::x86_64::_mm256_and_si256(v_u16, core::arch::x86_64::_mm256_set1_epi16(0x0f));
            let high_nibble = core::arch::x86_64::_mm256_and_si256(
                core::arch::x86_64::_mm256_srli_epi16(v_u16, 4),
                core::arch::x86_64::_mm256_set1_epi16(0x0f)
            );

            #[inline(always)]
            unsafe fn unpack_nibbles_avx2(v: core::arch::x86_64::__m256i) -> core::arch::x86_64::__m256i {
                let sign = core::arch::x86_64::_mm256_slli_epi16(
                    core::arch::x86_64::_mm256_and_si256(v, core::arch::x86_64::_mm256_set1_epi16(0x08)),
                    12
                );
                let rest = core::arch::x86_64::_mm256_slli_epi16(
                    core::arch::x86_64::_mm256_and_si256(v, core::arch::x86_64::_mm256_set1_epi16(0x07)),
                    6
                );
                let is_zero = core::arch::x86_64::_mm256_cmpeq_epi16(rest, core::arch::x86_64::_mm256_setzero_si256());
                let bias_diff = core::arch::x86_64::_mm256_andnot_si256(is_zero, core::arch::x86_64::_mm256_set1_epi16(126 << 7));
                let rest_biased = core::arch::x86_64::_mm256_add_epi16(rest, bias_diff);
                core::arch::x86_64::_mm256_or_si256(sign, rest_biased)
            }

            let res_low = unpack_nibbles_avx2(low_nibble);
            let res_high = unpack_nibbles_avx2(high_nibble);

            let res_lo_128 = core::arch::x86_64::_mm256_unpacklo_epi16(res_low, res_high);
            let res_hi_128 = core::arch::x86_64::_mm256_unpackhi_epi16(res_low, res_high);
            let out0 = core::arch::x86_64::_mm256_permute2x128_si256(res_lo_128, res_hi_128, 0x20);
            let out1 = core::arch::x86_64::_mm256_permute2x128_si256(res_lo_128, res_hi_128, 0x31);

            let out_ptr0 = unpacked.as_mut_ptr().add(2 * i) as *mut core::arch::x86_64::__m256i;
            core::arch::x86_64::_mm256_storeu_si256(out_ptr0, out0);
            let out_ptr1 = unpacked.as_mut_ptr().add(2 * i + 16) as *mut core::arch::x86_64::__m256i;
            core::arch::x86_64::_mm256_storeu_si256(out_ptr1, out1);

            i += 16;
        }

        for j in i..n {
            let byte = packed[j] as u16;
            let b1 = byte & 0x0f;
            let b2 = (byte >> 4) & 0x0f;
            let sign1 = (b1 & 0x08) << 12;
            let rest1 = (b1 & 0x07) << 6;
            let bias_diff1 = if rest1 == 0 { 0 } else { 126 << 7 };
            unpacked[2 * j] = Bf16(half::bf16::from_bits(sign1 | (rest1 + bias_diff1)));
            let sign2 = (b2 & 0x08) << 12;
            let rest2 = (b2 & 0x07) << 6;
            let bias_diff2 = if rest2 == 0 { 0 } else { 126 << 7 };
            unpacked[2 * j + 1] = Bf16(half::bf16::from_bits(sign2 | (rest2 + bias_diff2)));
        }
    }

    const fn f4_to_f32_bits(bits: u8) -> u32 {
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

    #[target_feature(enable = "avx2")]
    pub unsafe fn unpack_f4_to_f32_avx2(packed: &[F4], unpacked: &mut [F32]) {
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
    pub unsafe fn unpack_f4_to_f32_packed_avx2(packed: &[u8], unpacked: &mut [F32]) {
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
}

/// Unpacks Bf8 elements to Bf16.
#[inline]
pub fn unpack_bf8_to_bf16(packed: &[Bf8], unpacked: &mut [Bf16]) {
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    {
        unsafe { avx2_unpack::unpack_bf8_to_bf16_avx2(packed, unpacked); }
    }
    #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
    {
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
}

/// Unpacks Bf4 elements to Bf16.
#[inline]
pub fn unpack_bf4_to_bf16(packed: &[Bf4], unpacked: &mut [Bf16]) {
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    {
        unsafe { avx2_unpack::unpack_bf4_to_bf16_avx2(packed, unpacked); }
    }
    #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
    {
        let len = packed.len();
        let n = len.min(unpacked.len());
        for i in 0..n {
            let b = packed[i].0 as u16;
            let sign = (b & 0x08) << 12;
            let rest = (b & 0x07) << 6;
            let bias_diff = if rest == 0 { 0 } else { 126 << 7 };
            unpacked[i] = Bf16(half::bf16::from_bits(sign | (rest + bias_diff)));
        }
    }
}

/// Unpacks packed 4-bit Bf4 pairs (stored 2 per byte) into a Bf16 slice.
#[inline]
pub fn unpack_bf4_to_bf16_packed(packed: &[u8], unpacked: &mut [Bf16]) {
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    {
        unsafe { avx2_unpack::unpack_bf4_to_bf16_packed_avx2(packed, unpacked); }
    }
    #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
    {
        let len = packed.len();
        let n = len.min(unpacked.len() / 2);
        for i in 0..n {
            let byte = packed[i] as u16;
            let b1 = byte & 0x0f;
            let b2 = (byte >> 4) & 0x0f;
            
            let sign1 = (b1 & 0x08) << 12;
            let rest1 = (b1 & 0x07) << 6;
            let bias_diff1 = if rest1 == 0 { 0 } else { 126 << 7 };
            unpacked[2 * i] = Bf16(half::bf16::from_bits(sign1 | (rest1 + bias_diff1)));
            
            let sign2 = (b2 & 0x08) << 12;
            let rest2 = (b2 & 0x07) << 6;
            let bias_diff2 = if rest2 == 0 { 0 } else { 126 << 7 };
            unpacked[2 * i + 1] = Bf16(half::bf16::from_bits(sign2 | (rest2 + bias_diff2)));
        }
    }
}

/// Unpacks F4 elements to F32.
#[inline]
pub fn unpack_f4_to_f32(packed: &[F4], unpacked: &mut [F32]) {
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    {
        unsafe { avx2_unpack::unpack_f4_to_f32_avx2(packed, unpacked); }
    }
    #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
    {
        let len = packed.len().min(unpacked.len());
        for i in 0..len {
            unpacked[i] = F32(packed[i].to_f32());
        }
    }
}

/// Unpacks packed 4-bit F4 pairs (stored 2 per byte) into an F32 slice.
#[inline]
pub fn unpack_f4_to_f32_packed(packed: &[u8], unpacked: &mut [F32]) {
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    {
        unsafe { avx2_unpack::unpack_f4_to_f32_packed_avx2(packed, unpacked); }
    }
    #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
    {
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
}
