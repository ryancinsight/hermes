//! Unpacking functions for low-precision data representations.

use crate::types::{Bf8, Bf4, Bf16};

#[cfg(target_arch = "x86_64")]
#[allow(dead_code)]
mod avx2_unpack {
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
