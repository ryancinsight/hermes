//! AVX-512 specific implementations of TileMatrixMultiply for x86_64.

#![cfg(any(target_arch = "x86", target_arch = "x86_64"))]

use crate::Avx512;
use eunomia::{Bf16, Bf4, Bf8, F32, I32, I8};
use hermes_simd_core::view::TileMatrixMultiply;

// ---------------------------------------------------------------------------
// AVX-512 Implementations
// ---------------------------------------------------------------------------

/// Native AVX-512 BF16 dot-product tile implementation.
///
/// The two BF16 operands are loaded as 512-bit integer vectors and reinterpreted
/// as `__m512bh`; `DPBF16PS` consumes one BF16 pair per f32 accumulator lane.
/// The `K = 32` tile therefore emits 16 dot-product instructions per output row.
#[target_feature(enable = "avx512f,avx512bf16")]
unsafe fn tile_matmul_bf16_native(
    c: *mut F32,
    c_stride: usize,
    a: *const Bf16,
    a_stride: usize,
    b: *const Bf16,
    b_stride: usize,
) {
    use core::arch::x86_64::*;

    let mut c_regs = [_mm512_setzero_ps(); 16];
    for (row, accumulator) in c_regs.iter_mut().enumerate() {
        *accumulator = _mm512_loadu_ps(c.add(row * c_stride).cast::<f32>());
    }

    for k in (0..32).step_by(2) {
        // DPBF16PS consumes adjacent BF16 pairs for each f32 lane. Interleave
        // the two depth rows so lane `column` receives b[k,column] and
        // b[k+1,column].
        let mut b_values = [0u16; 32];
        for column in 0..16 {
            b_values[2 * column] = (*b.add(k * b_stride + column)).to_bits();
            b_values[2 * column + 1] = (*b.add((k + 1) * b_stride + column)).to_bits();
        }
        let b_vec =
            core::mem::transmute::<__m512i, __m512bh>(_mm512_loadu_si512(b_values.as_ptr().cast()));

        for (row, accumulator) in c_regs.iter_mut().enumerate() {
            let a0 = (*a.add(row * a_stride + k)).to_bits();
            let a1 = (*a.add(row * a_stride + k + 1)).to_bits();
            let mut a_values = [0u16; 32];
            for pair in a_values.chunks_exact_mut(2) {
                pair[0] = a0;
                pair[1] = a1;
            }
            let a_vec = core::mem::transmute::<__m512i, __m512bh>(_mm512_loadu_si512(
                a_values.as_ptr().cast(),
            ));
            *accumulator = _mm512_dpbf16_ps(*accumulator, a_vec, b_vec);
        }
    }

    for (row, accumulator) in c_regs.iter().enumerate() {
        _mm512_storeu_ps(c.add(row * c_stride).cast::<f32>(), *accumulator);
    }
}

impl TileMatrixMultiply<Bf16, Bf16, F32, Avx512, Avx512, 16, 16, 32> for Avx512 {
    // SAFETY: the dispatch site selects this native helper only after the exact
    // `avx512bf16` probe, or the conversion helper after `avx512f,bw,vl`. All
    // pointers address a complete 16x16x32 tile at the supplied strides.
    #[target_feature(enable = "avx512f,avx512bw,avx512vl")]
    unsafe fn tile_matmul(
        c: *mut F32,
        c_stride: usize,
        a: *const Bf16,
        a_stride: usize,
        b: *const Bf16,
        b_stride: usize,
    ) {
        #[cfg(feature = "std")]
        if crate::has_avx512_bf16() {
            // SAFETY: the runtime probe above enables the exact instruction
            // set required by the native helper; tile pointers satisfy this
            // trait's complete-tile precondition.
            tile_matmul_bf16_native(c, c_stride, a, a_stride, b, b_stride);
            return;
        }
        #[cfg(all(not(feature = "std"), target_feature = "avx512bf16"))]
        {
            // SAFETY: this no-std build statically enables the helper's ISA.
            tile_matmul_bf16_native(c, c_stride, a, a_stride, b, b_stride);
            return;
        }

        use core::arch::x86_64::*;

        let mut c_regs = [_mm512_setzero_ps(); 16];
        for i in 0..16 {
            c_regs[i] = _mm512_loadu_ps(c.add(i * c_stride) as *const f32);
        }

        for k in 0..32 {
            let mut a_vals = [0f32; 16];
            for i in 0..16 {
                a_vals[i] = eunomia::FloatElement::to_f32(*a.add(i * a_stride + k));
            }

            let b_ptr = b.add(k * b_stride);
            let mut b_vals = [0f32; 16];
            for j in 0..16 {
                b_vals[j] = eunomia::FloatElement::to_f32(*b_ptr.add(j));
            }
            let b_vec = _mm512_loadu_ps(b_vals.as_ptr());

            for i in 0..16 {
                let a_vec = _mm512_set1_ps(a_vals[i]);
                c_regs[i] = _mm512_fmadd_ps(a_vec, b_vec, c_regs[i]);
            }
        }

        for i in 0..16 {
            _mm512_storeu_ps(c.add(i * c_stride) as *mut f32, c_regs[i]);
        }
    }
}

/// Computes one signed `i8` 16×16×64 tile with AVX-512 VNNI.
///
/// AVX-512 VNNI provides `VPDPBUSD` (unsigned bytes × signed bytes), not the
/// signed-byte ZMM `VPDPBSSD` form. Biasing A by 128 makes it unsigned, then
/// subtracting `128 * sum(B[:, j])` from every output row recovers the exact
/// signed product in wrapping `i32` arithmetic.
///
/// # Safety
/// The CPU must support `avx512f` and `avx512vnni`. The pointers must address a
/// complete 16×16×64 tile at the supplied strides.
#[target_feature(enable = "avx512f,avx512vnni")]
unsafe fn tile_matmul_i8(
    c: *mut i32,
    c_stride: usize,
    a: *const i8,
    a_stride: usize,
    b: *const i8,
    b_stride: usize,
) {
    use core::arch::x86_64::*;

    let mut c_regs = [_mm512_setzero_si512(); 16];
    for (row, accumulator) in c_regs.iter_mut().enumerate() {
        *accumulator = _mm512_loadu_si512(c.add(row * c_stride) as *const _);
    }

    let byte_bias = _mm512_set1_epi8(i8::MIN);
    let mut bias_accumulator = _mm512_setzero_si512();
    for k in (0..64).step_by(4) {
        let row0 = _mm_loadu_si128(b.add(k * b_stride) as *const __m128i);
        let row1 = _mm_loadu_si128(b.add((k + 1) * b_stride) as *const __m128i);
        let row2 = _mm_loadu_si128(b.add((k + 2) * b_stride) as *const __m128i);
        let row3 = _mm_loadu_si128(b.add((k + 3) * b_stride) as *const __m128i);

        let unpack_lo_01 = _mm_unpacklo_epi8(row0, row1);
        let unpack_hi_01 = _mm_unpackhi_epi8(row0, row1);
        let unpack_lo_23 = _mm_unpacklo_epi8(row2, row3);
        let unpack_hi_23 = _mm_unpackhi_epi8(row2, row3);
        let packed_lo = _mm_unpacklo_epi16(unpack_lo_01, unpack_lo_23);
        let packed_mid_lo = _mm_unpackhi_epi16(unpack_lo_01, unpack_lo_23);
        let packed_mid_hi = _mm_unpacklo_epi16(unpack_hi_01, unpack_hi_23);
        let packed_hi = _mm_unpackhi_epi16(unpack_hi_01, unpack_hi_23);
        let b_vec = _mm512_inserti32x4(
            _mm512_inserti32x4(
                _mm512_inserti32x4(_mm512_castsi128_si512(packed_lo), packed_mid_lo, 1),
                packed_mid_hi,
                2,
            ),
            packed_hi,
            3,
        );

        bias_accumulator = _mm512_dpbusd_epi32(bias_accumulator, byte_bias, b_vec);
        for (row, accumulator) in c_regs.iter_mut().enumerate() {
            let a_val = core::ptr::read_unaligned(a.add(row * a_stride + k) as *const i32);
            let a_unsigned = _mm512_xor_si512(_mm512_set1_epi32(a_val), byte_bias);
            *accumulator = _mm512_dpbusd_epi32(*accumulator, a_unsigned, b_vec);
        }
    }

    for (row, accumulator) in c_regs.iter().enumerate() {
        let corrected = _mm512_sub_epi32(*accumulator, bias_accumulator);
        _mm512_storeu_si512(c.add(row * c_stride) as *mut _, corrected);
    }
}

impl TileMatrixMultiply<i8, i8, i32, Avx512, Avx512, 16, 16, 64> for Avx512 {
    #[target_feature(enable = "avx512f,avx512vnni")]
    unsafe fn tile_matmul(
        c: *mut i32,
        c_stride: usize,
        a: *const i8,
        a_stride: usize,
        b: *const i8,
        b_stride: usize,
    ) {
        tile_matmul_i8(c, c_stride, a, a_stride, b, b_stride);
    }
}

impl TileMatrixMultiply<I8, I8, I32, Avx512, Avx512, 16, 16, 64> for Avx512 {
    #[target_feature(enable = "avx512f,avx512vnni")]
    unsafe fn tile_matmul(
        c: *mut I32,
        c_stride: usize,
        a: *const I8,
        a_stride: usize,
        b: *const I8,
        b_stride: usize,
    ) {
        tile_matmul_i8(
            c.cast::<i32>(),
            c_stride,
            a.cast::<i8>(),
            a_stride,
            b.cast::<i8>(),
            b_stride,
        );
    }
}

/// Unpacks packed 4-bit signed integers (stored 2 per byte) into an 8-bit signed integer slice.
/// Optimized using AVX2 when available.
#[inline]
pub fn unpack_int4(packed: &[u8], unpacked: &mut [i8]) {
    let len = packed.len();
    assert!(unpacked.len() >= len * 2);

    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    {
        // SAFETY: this branch is compiled only when `avx2` is a static target feature,
        // satisfying `unpack_int4_avx2`'s ISA precondition; the `unpacked.len() >= len * 2`
        // assertion above upholds its bounds precondition.
        // SAFETY: reached only after the guarding `is_x86_feature_detected!` check (or `cfg!(target_feature)` in no-std) confirms the required AVX-512 features, so the delegated `eunomia` unpacker's ISA precondition holds; the slice bounds are its documented contract.
        unsafe {
            unpack_int4_avx2(packed, unpacked);
        }
    }

    #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
    {
        unpack_int4_scalar(packed, unpacked);
    }
}

#[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
#[inline]
fn unpack_int4_scalar(packed: &[u8], unpacked: &mut [i8]) {
    for (i, byte) in packed.iter().copied().enumerate() {
        let byte = byte as i8;
        unpacked[2 * i] = (byte << 4) >> 4;
        unpacked[2 * i + 1] = byte >> 4;
    }
}

/// AVX2 optimized int4 unpacker.
///
/// # Safety
/// The target CPU must support `avx2` (enforced by the `#[target_feature]` gate plus
/// the `cfg!(target_feature = "avx2")` guard at the `unpack_int4` call site), and
/// `unpacked.len()` must be at least `2 * packed.len()` (asserted by the `unpack_int4`
/// wrapper); the 32-wide loads/stores stay within those slice bounds.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn unpack_int4_avx2(packed: &[u8], unpacked: &mut [i8]) {
    use core::arch::x86_64::*;

    let len = packed.len();
    let mut i = 0;

    let mask = _mm256_set1_epi8(0x0F);
    let lookup = _mm256_setr_epi8(
        0, 1, 2, 3, 4, 5, 6, 7, -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, -8, -7, -6,
        -5, -4, -3, -2, -1,
    );

    while i + 32 <= len {
        let v = _mm256_loadu_si256(packed.as_ptr().add(i) as *const _);

        let low_nibbles = _mm256_and_si256(v, mask);
        let high_nibbles = _mm256_and_si256(_mm256_srli_epi16(v, 4), mask);

        let low_signed = _mm256_shuffle_epi8(lookup, low_nibbles);
        let high_signed = _mm256_shuffle_epi8(lookup, high_nibbles);

        let res_lo = _mm256_unpacklo_epi8(low_signed, high_signed);
        let res_hi = _mm256_unpackhi_epi8(low_signed, high_signed);

        let res0 = _mm256_permute2x128_si256(res_lo, res_hi, 0x20);
        let res1 = _mm256_permute2x128_si256(res_lo, res_hi, 0x31);

        _mm256_storeu_si256(unpacked.as_mut_ptr().add(2 * i) as *mut _, res0);
        _mm256_storeu_si256(unpacked.as_mut_ptr().add(2 * i + 32) as *mut _, res1);

        i += 32;
    }

    for j in i..len {
        let byte = packed[j] as i8;
        unpacked[2 * j] = (byte << 4) >> 4;
        unpacked[2 * j + 1] = byte >> 4;
    }
}

/// Unpacks Bf8 elements to Bf16 for accumulation.
#[inline]
pub fn unpack_bf8_to_bf16(packed: &[Bf8], unpacked: &mut [Bf16]) {
    #[cfg(target_arch = "x86_64")]
    {
        #[cfg(feature = "std")]
        {
            if std::is_x86_feature_detected!("avx512bw")
                && std::is_x86_feature_detected!("avx512vl")
            {
                // SAFETY: reached only after the guarding `is_x86_feature_detected!` check (or `cfg!(target_feature)` in no-std) confirms the required AVX-512 features, so the delegated `eunomia` unpacker's ISA precondition holds; the slice bounds are its documented contract.
                unsafe {
                    eunomia::unsafe_intrinsics::avx512::unpack_bf8_to_bf16(packed, unpacked);
                    return;
                }
            }
        }
        #[cfg(not(feature = "std"))]
        {
            if cfg!(target_feature = "avx512bw") {
                // SAFETY: reached only after the guarding `is_x86_feature_detected!` check (or `cfg!(target_feature)` in no-std) confirms the required AVX-512 features, so the delegated `eunomia` unpacker's ISA precondition holds; the slice bounds are its documented contract.
                unsafe {
                    eunomia::unsafe_intrinsics::avx512::unpack_bf8_to_bf16(packed, unpacked);
                    return;
                }
            }
        }
    }
    eunomia::unpack_bf8_to_bf16(packed, unpacked);
}

/// Unpacks Bf4 elements to Bf16 for accumulation.
#[inline]
pub fn unpack_bf4_to_bf16(packed: &[Bf4], unpacked: &mut [Bf16]) {
    #[cfg(target_arch = "x86_64")]
    {
        #[cfg(feature = "std")]
        {
            if std::is_x86_feature_detected!("avx512bw")
                && std::is_x86_feature_detected!("avx512vl")
            {
                // SAFETY: reached only after the guarding `is_x86_feature_detected!` check (or `cfg!(target_feature)` in no-std) confirms the required AVX-512 features, so the delegated `eunomia` unpacker's ISA precondition holds; the slice bounds are its documented contract.
                unsafe {
                    eunomia::unsafe_intrinsics::avx512::unpack_bf4_to_bf16(packed, unpacked);
                    return;
                }
            }
        }
        #[cfg(not(feature = "std"))]
        {
            if cfg!(target_feature = "avx512bw") {
                // SAFETY: reached only after the guarding `is_x86_feature_detected!` check (or `cfg!(target_feature)` in no-std) confirms the required AVX-512 features, so the delegated `eunomia` unpacker's ISA precondition holds; the slice bounds are its documented contract.
                unsafe {
                    eunomia::unsafe_intrinsics::avx512::unpack_bf4_to_bf16(packed, unpacked);
                    return;
                }
            }
        }
    }
    eunomia::unpack_bf4_to_bf16(packed, unpacked);
}

/// Unpacks packed Bf4 elements (stored 2 per byte in `packed`) into a Bf16 slice.
#[inline]
pub fn unpack_packed_bf4_to_bf16(packed: &[u8], unpacked: &mut [Bf16]) {
    #[cfg(target_arch = "x86_64")]
    {
        #[cfg(feature = "std")]
        {
            if std::is_x86_feature_detected!("avx512bw")
                && std::is_x86_feature_detected!("avx512vl")
            {
                // SAFETY: reached only after the guarding `is_x86_feature_detected!` check (or `cfg!(target_feature)` in no-std) confirms the required AVX-512 features, so the delegated `eunomia` unpacker's ISA precondition holds; the slice bounds are its documented contract.
                unsafe {
                    eunomia::unsafe_intrinsics::avx512::unpack_bf4_to_bf16_packed(packed, unpacked);
                    return;
                }
            }
        }
        #[cfg(not(feature = "std"))]
        {
            if cfg!(target_feature = "avx512bw") {
                // SAFETY: reached only after the guarding `is_x86_feature_detected!` check (or `cfg!(target_feature)` in no-std) confirms the required AVX-512 features, so the delegated `eunomia` unpacker's ISA precondition holds; the slice bounds are its documented contract.
                unsafe {
                    eunomia::unsafe_intrinsics::avx512::unpack_bf4_to_bf16_packed(packed, unpacked);
                    return;
                }
            }
        }
    }
    eunomia::unpack_bf4_to_bf16_packed(packed, unpacked);
}

/// Unpacks packed F4 elements (stored 2 per byte in `packed`) into an F32 slice.
#[inline]
pub fn unpack_packed_f4_to_f32(packed: &[u8], unpacked: &mut [F32]) {
    #[cfg(target_arch = "x86_64")]
    {
        #[cfg(feature = "std")]
        {
            if std::is_x86_feature_detected!("avx512f") && std::is_x86_feature_detected!("avx512vl")
            {
                // SAFETY: reached only after the guarding `is_x86_feature_detected!` check (or `cfg!(target_feature)` in no-std) confirms the required AVX-512 features, so the delegated `eunomia` unpacker's ISA precondition holds; the slice bounds are its documented contract.
                unsafe {
                    eunomia::unsafe_intrinsics::avx512::unpack_f4_to_f32_packed(packed, unpacked);
                    return;
                }
            }
        }
        #[cfg(not(feature = "std"))]
        {
            if cfg!(target_feature = "avx512f") {
                // SAFETY: reached only after the guarding `is_x86_feature_detected!` check (or `cfg!(target_feature)` in no-std) confirms the required AVX-512 features, so the delegated `eunomia` unpacker's ISA precondition holds; the slice bounds are its documented contract.
                unsafe {
                    eunomia::unsafe_intrinsics::avx512::unpack_f4_to_f32_packed(packed, unpacked);
                    return;
                }
            }
        }
    }
    eunomia::unpack_f4_to_f32_packed(packed, unpacked);
}

#[cfg(test)]
mod int8_tests {
    use super::*;

    #[test]
    fn signed_tile_matches_wrapping_scalar_bitwise() {
        if !std::is_x86_feature_detected!("avx512f") || !std::is_x86_feature_detected!("avx512vnni")
        {
            return;
        }

        const M: usize = 16;
        const N: usize = 16;
        const K: usize = 64;
        let a: Vec<i8> = (0..M * K)
            .map(|index| ((index * 89 + 3) % 256) as u8 as i8)
            .collect();
        let b: Vec<i8> = (0..K * N)
            .map(|index| ((index * 41 + 128) % 256) as u8 as i8)
            .collect();
        let initial: Vec<i32> = (0..M * N)
            .map(|index| (index as i32).wrapping_mul(7_919).wrapping_sub(40_000))
            .collect();
        let mut expected = initial.clone();
        for row in 0..M {
            for column in 0..N {
                let mut sum = 0i32;
                for depth in 0..K {
                    sum = sum
                        .wrapping_add((a[row * K + depth] as i32) * (b[depth * N + column] as i32));
                }
                expected[row * N + column] = expected[row * N + column].wrapping_add(sum);
            }
        }

        let mut primitive = initial.clone();
        unsafe {
            <Avx512 as TileMatrixMultiply<i8, i8, i32, Avx512, Avx512, M, N, K>>::tile_matmul(
                primitive.as_mut_ptr(),
                N,
                a.as_ptr(),
                K,
                b.as_ptr(),
                N,
            );
        }
        assert_eq!(primitive, expected);

        let wrapped_a: Vec<I8> = a.iter().copied().map(I8).collect();
        let wrapped_b: Vec<I8> = b.iter().copied().map(I8).collect();
        let mut wrapped_c: Vec<I32> = initial.iter().copied().map(I32).collect();
        unsafe {
            <Avx512 as TileMatrixMultiply<I8, I8, I32, Avx512, Avx512, M, N, K>>::tile_matmul(
                wrapped_c.as_mut_ptr(),
                N,
                wrapped_a.as_ptr(),
                K,
                wrapped_b.as_ptr(),
                N,
            );
        }
        assert_eq!(
            wrapped_c
                .into_iter()
                .map(|value| value.0)
                .collect::<Vec<_>>(),
            expected
        );
    }
}
