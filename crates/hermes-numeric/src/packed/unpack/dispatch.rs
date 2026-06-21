//! ISA-dispatched unpack functions for low-precision data representations.

use crate::types::{Bf16, Bf4, Bf8, F32, F4, F8};

#[cfg(target_arch = "x86_64")]
use super::arch::{has_avx2, has_avx512bw, has_avx512f};

#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
use super::unsafe_intrinsics;

use super::conv::{bf4_to_bf16_bits, f8_to_f32_bits};

/// Unpacks Bf8 elements to Bf16.
#[inline]
pub fn unpack_bf8_to_bf16(packed: &[Bf8], unpacked: &mut [Bf16]) {
    #[cfg(target_arch = "x86_64")]
    {
        if has_avx512bw() {
            unsafe {
                unsafe_intrinsics::avx512::unpack_bf8_to_bf16(packed, unpacked);
            }
            return;
        }
        if has_avx2() {
            unsafe {
                unsafe_intrinsics::avx2::unpack_bf8_to_bf16(packed, unpacked);
            }
            return;
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        unsafe {
            unsafe_intrinsics::neon::unpack_bf8_to_bf16(packed, unpacked);
        }
        return;
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
            unsafe {
                unsafe_intrinsics::avx512::unpack_bf4_to_bf16(packed, unpacked);
            }
            return;
        }
        if has_avx2() {
            unsafe {
                unsafe_intrinsics::avx2::unpack_bf4_to_bf16(packed, unpacked);
            }
            return;
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        unsafe {
            unsafe_intrinsics::neon::unpack_bf4_to_bf16(packed, unpacked);
        }
        return;
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
            unsafe {
                unsafe_intrinsics::avx512::unpack_bf4_to_bf16_packed(packed, unpacked);
            }
            return;
        }
        if has_avx2() {
            unsafe {
                unsafe_intrinsics::avx2::unpack_bf4_to_bf16_packed(packed, unpacked);
            }
            return;
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        unsafe {
            unsafe_intrinsics::neon::unpack_bf4_to_bf16_packed(packed, unpacked);
        }
        return;
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
            unsafe {
                unsafe_intrinsics::avx512::unpack_f4_to_f32(packed, unpacked);
            }
            return;
        }
        if has_avx2() {
            unsafe {
                unsafe_intrinsics::avx2::unpack_f4_to_f32(packed, unpacked);
            }
            return;
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        unsafe {
            unsafe_intrinsics::neon::unpack_f4_to_f32(packed, unpacked);
        }
        return;
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
            unsafe {
                unsafe_intrinsics::avx512::unpack_f4_to_f32_packed(packed, unpacked);
            }
            return;
        }
        if has_avx2() {
            unsafe {
                unsafe_intrinsics::avx2::unpack_f4_to_f32_packed(packed, unpacked);
            }
            return;
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        unsafe {
            unsafe_intrinsics::neon::unpack_f4_to_f32_packed(packed, unpacked);
        }
        return;
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
            unsafe {
                unsafe_intrinsics::avx512::unpack_f8_to_f32(packed, unpacked);
            }
            return;
        }
        if has_avx2() {
            unsafe {
                unsafe_intrinsics::avx2::unpack_f8_to_f32(packed, unpacked);
            }
            return;
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        unsafe {
            unsafe_intrinsics::neon::unpack_f8_to_f32(packed, unpacked);
        }
        return;
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
