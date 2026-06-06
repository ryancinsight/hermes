use hermes_simd_core::view::{TileMatrixMultiply, SimdError};
use hermes_simd_intrinsics::{AmxBf16, AmxInt8, Avx512, Scalar};
use crate::cpu::{AmxSupport, Avx512Support};
use hermes_numeric::{Bf16, F32, I8, I32};

#[inline(never)]
fn validate_gemm_sizes(
    a_len: usize,
    b_len: usize,
    c_len: usize,
    m: usize,
    _n: usize,
    k: usize,
    a_stride: usize,
    b_stride: usize,
    c_stride: usize,
) -> Result<(), SimdError> {
    if a_len < m * a_stride || b_len < k * b_stride || c_len < m * c_stride {
        return Err(SimdError::LengthMismatch);
    }
    Ok(())
}

#[inline(always)]
unsafe fn tile_loop_generic<TA, TB, TC, Arch, const M: usize, const N: usize, const K: usize>(
    m: usize, n: usize, k: usize,
    a: *const TA, a_stride: usize,
    b: *const TB, b_stride: usize,
    c: *mut TC, c_stride: usize,
)
where
    Arch: TileMatrixMultiply<TA, TB, TC, Arch, Arch, M, N, K>,
{
    let mut i = 0;
    while i + M <= m {
        let mut j = 0;
        while j + N <= n {
            let mut kk = 0;
            while kk + K <= k {
                Arch::tile_matmul(
                    c.add(i * c_stride + j), c_stride,
                    a.add(i * a_stride + kk), a_stride,
                    b.add(kk * b_stride + j), b_stride,
                );
                kk += K;
            }
            j += N;
        }
        i += M;
    }
}

/// Trait for performing register-tiled GEMM and dynamic tile dispatching.
pub trait TiledGemm<TA, TB, TC> {
    /// Perform matrix multiplication `c += a * b` using register-blocked/tiled SIMD.
    ///
    /// # Safety
    /// - Pointers must be valid and slices must have matching dimensions.
    unsafe fn gemm(
        m: usize, n: usize, k: usize,
        a: &[TA], a_stride: usize,
        b: &[TB], b_stride: usize,
        c: &mut [TC], c_stride: usize,
    ) -> Result<(), SimdError>;

    /// Dynamically dispatches tile matrix multiplication for a single tile of shape MxNxK.
    ///
    /// # Safety
    /// - Pointers must be valid and aligned as per the chosen backend requirements.
    unsafe fn dispatch_tile_matmul(
        c: *mut TC, c_stride: usize,
        a: *const TA, a_stride: usize,
        b: *const TB, b_stride: usize,
    );
}

impl TiledGemm<half::bf16, half::bf16, f32> for (half::bf16, half::bf16, f32) {
    #[inline]
    unsafe fn dispatch_tile_matmul(
        c: *mut f32, c_stride: usize,
        a: *const half::bf16, a_stride: usize,
        b: *const half::bf16, b_stride: usize,
    ) {
        #[cfg(target_arch = "x86_64")]
        {
            if <half::bf16 as AmxSupport>::has_amx() && hermes_simd_intrinsics::AmxSession::is_active() {
                return <AmxBf16 as TileMatrixMultiply<half::bf16, half::bf16, f32, AmxBf16, AmxBf16, 16, 16, 32>>::tile_matmul(c, c_stride, a, a_stride, b, b_stride);
            }
            if <half::bf16 as Avx512Support>::has_avx512() {
                return <Avx512 as TileMatrixMultiply<half::bf16, half::bf16, f32, Avx512, Avx512, 16, 16, 32>>::tile_matmul(c, c_stride, a, a_stride, b, b_stride);
            }
        }
        <Scalar as TileMatrixMultiply<half::bf16, half::bf16, f32, Scalar, Scalar, 16, 16, 32>>::tile_matmul(c, c_stride, a, a_stride, b, b_stride);
    }

    #[inline]
    unsafe fn gemm(
        m: usize, n: usize, k: usize,
        a: &[half::bf16], a_stride: usize,
        b: &[half::bf16], b_stride: usize,
        c: &mut [f32], c_stride: usize,
    ) -> Result<(), SimdError> {
        validate_gemm_sizes(a.len(), b.len(), c.len(), m, n, k, a_stride, b_stride, c_stride)?;

        #[cfg(target_arch = "x86_64")]
        {
            let decision = crate::dispatcher::AdaptiveDispatcher::select_backend(
                m, n, k,
                a.as_ptr(), a.len(),
                b.as_ptr(), b.len(),
            );

            match decision {
                crate::dispatcher::DispatchDecision::Amx => {
                    <AmxBf16 as hermes_simd_intrinsics::x86_64::amx::AmxGemm<half::bf16, half::bf16, f32>>::amx_gemm(
                        m, n, k,
                        a.as_ptr(), a_stride,
                        b.as_ptr(), b_stride,
                        c.as_mut_ptr(), c_stride,
                    );
                    return Ok(());
                }
                crate::dispatcher::DispatchDecision::Avx512 => {
                    tile_loop_generic::<half::bf16, half::bf16, f32, Avx512, 16, 16, 32>(
                        m, n, k,
                        a.as_ptr(), a_stride,
                        b.as_ptr(), b_stride,
                        c.as_mut_ptr(), c_stride,
                    );

                    let amx_m_bound = (m / 16) * 16;
                    let amx_n_bound = (n / 16) * 16;
                    let amx_k_bound = (k / 32) * 32;
                    for r in 0..m {
                        for col in 0..n {
                            if r >= amx_m_bound || col >= amx_n_bound {
                                let mut sum = 0.0f32;
                                for kk in 0..k {
                                    sum += a[r * a_stride + kk].to_f32() * b[kk * b_stride + col].to_f32();
                                }
                                c[r * c_stride + col] += sum;
                            } else if amx_k_bound < k {
                                let mut sum = 0.0f32;
                                for kk in amx_k_bound..k {
                                    sum += a[r * a_stride + kk].to_f32() * b[kk * b_stride + col].to_f32();
                                }
                                c[r * c_stride + col] += sum;
                            }
                        }
                    }
                    return Ok(());
                }
                crate::dispatcher::DispatchDecision::Scalar => {}
            }
        }

        tile_loop_generic::<half::bf16, half::bf16, f32, Scalar, 16, 16, 32>(
            m, n, k,
            a.as_ptr(), a_stride,
            b.as_ptr(), b_stride,
            c.as_mut_ptr(), c_stride,
        );

        let amx_m_bound = (m / 16) * 16;
        let amx_n_bound = (n / 16) * 16;
        let amx_k_bound = (k / 32) * 32;
        for r in 0..m {
            for col in 0..n {
                if r >= amx_m_bound || col >= amx_n_bound {
                    let mut sum = 0.0f32;
                    for kk in 0..k {
                        sum += a[r * a_stride + kk].to_f32() * b[kk * b_stride + col].to_f32();
                    }
                    c[r * c_stride + col] += sum;
                } else if amx_k_bound < k {
                    let mut sum = 0.0f32;
                    for kk in amx_k_bound..k {
                        sum += a[r * a_stride + kk].to_f32() * b[kk * b_stride + col].to_f32();
                    }
                    c[r * c_stride + col] += sum;
                }
            }
        }
        Ok(())
    }
}

impl TiledGemm<Bf16, Bf16, F32> for (Bf16, Bf16, F32) {
    #[inline]
    unsafe fn dispatch_tile_matmul(
        c: *mut F32, c_stride: usize,
        a: *const Bf16, a_stride: usize,
        b: *const Bf16, b_stride: usize,
    ) {
        #[cfg(target_arch = "x86_64")]
        {
            if <half::bf16 as AmxSupport>::has_amx() && hermes_simd_intrinsics::AmxSession::is_active() {
                return <AmxBf16 as TileMatrixMultiply<Bf16, Bf16, F32, AmxBf16, AmxBf16, 16, 16, 32>>::tile_matmul(c, c_stride, a, a_stride, b, b_stride);
            }
            if <half::bf16 as Avx512Support>::has_avx512() {
                return <Avx512 as TileMatrixMultiply<Bf16, Bf16, F32, Avx512, Avx512, 16, 16, 32>>::tile_matmul(c, c_stride, a, a_stride, b, b_stride);
            }
        }
        <Scalar as TileMatrixMultiply<Bf16, Bf16, F32, Scalar, Scalar, 16, 16, 32>>::tile_matmul(c, c_stride, a, a_stride, b, b_stride);
    }

    #[inline]
    unsafe fn gemm(
        m: usize, n: usize, k: usize,
        a: &[Bf16], a_stride: usize,
        b: &[Bf16], b_stride: usize,
        c: &mut [F32], c_stride: usize,
    ) -> Result<(), SimdError> {
        validate_gemm_sizes(a.len(), b.len(), c.len(), m, n, k, a_stride, b_stride, c_stride)?;

        #[cfg(target_arch = "x86_64")]
        {
            let decision = crate::dispatcher::AdaptiveDispatcher::select_backend(
                m, n, k,
                a.as_ptr(), a.len(),
                b.as_ptr(), b.len(),
            );

            match decision {
                crate::dispatcher::DispatchDecision::Amx => {
                    <AmxBf16 as hermes_simd_intrinsics::x86_64::amx::AmxGemm<Bf16, Bf16, F32>>::amx_gemm(
                        m, n, k,
                        a.as_ptr(), a_stride,
                        b.as_ptr(), b_stride,
                        c.as_mut_ptr(), c_stride,
                    );
                    return Ok(());
                }
                crate::dispatcher::DispatchDecision::Avx512 => {
                    tile_loop_generic::<Bf16, Bf16, F32, Avx512, 16, 16, 32>(
                        m, n, k,
                        a.as_ptr(), a_stride,
                        b.as_ptr(), b_stride,
                        c.as_mut_ptr(), c_stride,
                    );

                    let amx_m_bound = (m / 16) * 16;
                    let amx_n_bound = (n / 16) * 16;
                    let amx_k_bound = (k / 32) * 32;
                    for r in 0..m {
                        for col in 0..n {
                            if r >= amx_m_bound || col >= amx_n_bound {
                                let mut sum = 0.0f32;
                                for kk in 0..k {
                                    sum += a[r * a_stride + kk].0.to_f32() * b[kk * b_stride + col].0.to_f32();
                                }
                                c[r * c_stride + col] = F32(c[r * c_stride + col].0 + sum);
                            } else if amx_k_bound < k {
                                let mut sum = 0.0f32;
                                for kk in amx_k_bound..k {
                                    sum += a[r * a_stride + kk].0.to_f32() * b[kk * b_stride + col].0.to_f32();
                                }
                                c[r * c_stride + col] = F32(c[r * c_stride + col].0 + sum);
                            }
                        }
                    }
                    return Ok(());
                }
                crate::dispatcher::DispatchDecision::Scalar => {}
            }
        }

        tile_loop_generic::<Bf16, Bf16, F32, Scalar, 16, 16, 32>(
            m, n, k,
            a.as_ptr(), a_stride,
            b.as_ptr(), b_stride,
            c.as_mut_ptr(), c_stride,
        );

        let amx_m_bound = (m / 16) * 16;
        let amx_n_bound = (n / 16) * 16;
        let amx_k_bound = (k / 32) * 32;
        for r in 0..m {
            for col in 0..n {
                if r >= amx_m_bound || col >= amx_n_bound {
                    let mut sum = 0.0f32;
                    for kk in 0..k {
                        sum += a[r * a_stride + kk].0.to_f32() * b[kk * b_stride + col].0.to_f32();
                    }
                    c[r * c_stride + col] = F32(c[r * c_stride + col].0 + sum);
                } else if amx_k_bound < k {
                    let mut sum = 0.0f32;
                    for kk in amx_k_bound..k {
                        sum += a[r * a_stride + kk].0.to_f32() * b[kk * b_stride + col].0.to_f32();
                    }
                    c[r * c_stride + col] = F32(c[r * c_stride + col].0 + sum);
                }
            }
        }
        Ok(())
    }
}

impl TiledGemm<i8, i8, i32> for (i8, i8, i32) {
    #[inline]
    unsafe fn dispatch_tile_matmul(
        c: *mut i32, c_stride: usize,
        a: *const i8, a_stride: usize,
        b: *const i8, b_stride: usize,
    ) {
        #[cfg(target_arch = "x86_64")]
        {
            if <i8 as AmxSupport>::has_amx() && hermes_simd_intrinsics::AmxSession::is_active() {
                return <AmxInt8 as TileMatrixMultiply<i8, i8, i32, AmxInt8, AmxInt8, 16, 16, 64>>::tile_matmul(c, c_stride, a, a_stride, b, b_stride);
            }
            if <i8 as Avx512Support>::has_avx512() {
                return <Avx512 as TileMatrixMultiply<i8, i8, i32, Avx512, Avx512, 16, 16, 64>>::tile_matmul(c, c_stride, a, a_stride, b, b_stride);
            }
        }
        <Scalar as TileMatrixMultiply<i8, i8, i32, Scalar, Scalar, 16, 16, 64>>::tile_matmul(c, c_stride, a, a_stride, b, b_stride);
    }

    #[inline]
    unsafe fn gemm(
        m: usize, n: usize, k: usize,
        a: &[i8], a_stride: usize,
        b: &[i8], b_stride: usize,
        c: &mut [i32], c_stride: usize,
    ) -> Result<(), SimdError> {
        validate_gemm_sizes(a.len(), b.len(), c.len(), m, n, k, a_stride, b_stride, c_stride)?;

        #[cfg(target_arch = "x86_64")]
        {
            let decision = crate::dispatcher::AdaptiveDispatcher::select_backend(
                m, n, k,
                a.as_ptr(), a.len(),
                b.as_ptr(), b.len(),
            );

            match decision {
                crate::dispatcher::DispatchDecision::Amx => {
                    <AmxInt8 as hermes_simd_intrinsics::x86_64::amx::AmxGemm<i8, i8, i32>>::amx_gemm(
                        m, n, k,
                        a.as_ptr(), a_stride,
                        b.as_ptr(), b_stride,
                        c.as_mut_ptr(), c_stride,
                    );
                    return Ok(());
                }
                crate::dispatcher::DispatchDecision::Avx512 => {
                    tile_loop_generic::<i8, i8, i32, Avx512, 16, 16, 64>(
                        m, n, k,
                        a.as_ptr(), a_stride,
                        b.as_ptr(), b_stride,
                        c.as_mut_ptr(), c_stride,
                    );

                    let amx_m_bound = (m / 16) * 16;
                    let amx_n_bound = (n / 16) * 16;
                    let amx_k_bound = (k / 64) * 64;
                    for r in 0..m {
                        for col in 0..n {
                            if r >= amx_m_bound || col >= amx_n_bound {
                                let mut sum = 0i32;
                                for kk in 0..k {
                                    sum = sum.wrapping_add((a[r * a_stride + kk] as i32) * (b[kk * b_stride + col] as i32));
                                }
                                c[r * c_stride + col] += sum;
                            } else if amx_k_bound < k {
                                let mut sum = 0i32;
                                for kk in amx_k_bound..k {
                                    sum = sum.wrapping_add((a[r * a_stride + kk] as i32) * (b[kk * b_stride + col] as i32));
                                }
                                c[r * c_stride + col] += sum;
                            }
                        }
                    }
                    return Ok(());
                }
                crate::dispatcher::DispatchDecision::Scalar => {}
            }
        }

        tile_loop_generic::<i8, i8, i32, Scalar, 16, 16, 64>(
            m, n, k,
            a.as_ptr(), a_stride,
            b.as_ptr(), b_stride,
            c.as_mut_ptr(), c_stride,
        );

        let amx_m_bound = (m / 16) * 16;
        let amx_n_bound = (n / 16) * 16;
        let amx_k_bound = (k / 64) * 64;
        for r in 0..m {
            for col in 0..n {
                if r >= amx_m_bound || col >= amx_n_bound {
                    let mut sum = 0i32;
                    for kk in 0..k {
                        sum = sum.wrapping_add((a[r * a_stride + kk] as i32) * (b[kk * b_stride + col] as i32));
                    }
                    c[r * c_stride + col] += sum;
                } else if amx_k_bound < k {
                    let mut sum = 0i32;
                    for kk in amx_k_bound..k {
                        sum = sum.wrapping_add((a[r * a_stride + kk] as i32) * (b[kk * b_stride + col] as i32));
                    }
                    c[r * c_stride + col] += sum;
                }
            }
        }
        Ok(())
    }
}

impl TiledGemm<I8, I8, I32> for (I8, I8, I32) {
    #[inline]
    unsafe fn dispatch_tile_matmul(
        c: *mut I32, c_stride: usize,
        a: *const I8, a_stride: usize,
        b: *const I8, b_stride: usize,
    ) {
        #[cfg(target_arch = "x86_64")]
        {
            if <i8 as AmxSupport>::has_amx() && hermes_simd_intrinsics::AmxSession::is_active() {
                return <AmxInt8 as TileMatrixMultiply<I8, I8, I32, AmxInt8, AmxInt8, 16, 16, 64>>::tile_matmul(c, c_stride, a, a_stride, b, b_stride);
            }
            if <i8 as Avx512Support>::has_avx512() {
                return <Avx512 as TileMatrixMultiply<I8, I8, I32, Avx512, Avx512, 16, 16, 64>>::tile_matmul(c, c_stride, a, a_stride, b, b_stride);
            }
        }
        <Scalar as TileMatrixMultiply<I8, I8, I32, Scalar, Scalar, 16, 16, 64>>::tile_matmul(c, c_stride, a, a_stride, b, b_stride);
    }

    #[inline]
    unsafe fn gemm(
        m: usize, n: usize, k: usize,
        a: &[I8], a_stride: usize,
        b: &[I8], b_stride: usize,
        c: &mut [I32], c_stride: usize,
    ) -> Result<(), SimdError> {
        validate_gemm_sizes(a.len(), b.len(), c.len(), m, n, k, a_stride, b_stride, c_stride)?;

        #[cfg(target_arch = "x86_64")]
        {
            let decision = crate::dispatcher::AdaptiveDispatcher::select_backend(
                m, n, k,
                a.as_ptr(), a.len(),
                b.as_ptr(), b.len(),
            );

            match decision {
                crate::dispatcher::DispatchDecision::Amx => {
                    <AmxInt8 as hermes_simd_intrinsics::x86_64::amx::AmxGemm<I8, I8, I32>>::amx_gemm(
                        m, n, k,
                        a.as_ptr(), a_stride,
                        b.as_ptr(), b_stride,
                        c.as_mut_ptr(), c_stride,
                    );
                    return Ok(());
                }
                crate::dispatcher::DispatchDecision::Avx512 => {
                    tile_loop_generic::<I8, I8, I32, Avx512, 16, 16, 64>(
                        m, n, k,
                        a.as_ptr(), a_stride,
                        b.as_ptr(), b_stride,
                        c.as_mut_ptr(), c_stride,
                    );

                    let amx_m_bound = (m / 16) * 16;
                    let amx_n_bound = (n / 16) * 16;
                    let amx_k_bound = (k / 64) * 64;
                    for r in 0..m {
                        for col in 0..n {
                            if r >= amx_m_bound || col >= amx_n_bound {
                                let mut sum = 0i32;
                                for kk in 0..k {
                                    sum = sum.wrapping_add((a[r * a_stride + kk].0 as i32) * (b[kk * b_stride + col].0 as i32));
                                }
                                c[r * c_stride + col] = I32(c[r * c_stride + col].0 + sum);
                            } else if amx_k_bound < k {
                                let mut sum = 0i32;
                                for kk in amx_k_bound..k {
                                    sum = sum.wrapping_add((a[r * a_stride + kk].0 as i32) * (b[kk * b_stride + col].0 as i32));
                                }
                                c[r * c_stride + col] = I32(c[r * c_stride + col].0 + sum);
                            }
                        }
                    }
                    return Ok(());
                }
                crate::dispatcher::DispatchDecision::Scalar => {}
            }
        }

        tile_loop_generic::<I8, I8, I32, Scalar, 16, 16, 64>(
            m, n, k,
            a.as_ptr(), a_stride,
            b.as_ptr(), b_stride,
            c.as_mut_ptr(), c_stride,
        );

        let amx_m_bound = (m / 16) * 16;
        let amx_n_bound = (n / 16) * 16;
        let amx_k_bound = (k / 64) * 64;
        for r in 0..m {
            for col in 0..n {
                if r >= amx_m_bound || col >= amx_n_bound {
                    let mut sum = 0i32;
                    for kk in 0..k {
                        sum = sum.wrapping_add((a[r * a_stride + kk].0 as i32) * (b[kk * b_stride + col].0 as i32));
                    }
                    c[r * c_stride + col] = I32(c[r * c_stride + col].0 + sum);
                } else if amx_k_bound < k {
                    let mut sum = 0i32;
                    for kk in amx_k_bound..k {
                        sum = sum.wrapping_add((a[r * a_stride + kk].0 as i32) * (b[kk * b_stride + col].0 as i32));
                    }
                    c[r * c_stride + col] = I32(c[r * c_stride + col].0 + sum);
                }
            }
        }
        Ok(())
    }
}

/// Perform matrix multiplication `c += a * b` using register-blocked/tiled SIMD.
///
/// Automatically dispatches to the most performant backend available (e.g. Intel AMX, AVX-512, or Scalar).
///
/// # Safety
/// - Pointers must be valid and slices must have matching dimensions.
#[inline]
pub unsafe fn gemm<TA, TB, TC>(
    m: usize, n: usize, k: usize,
    a: &[TA], a_stride: usize,
    b: &[TB], b_stride: usize,
    c: &mut [TC], c_stride: usize,
) -> Result<(), SimdError>
where
    (TA, TB, TC): TiledGemm<TA, TB, TC>,
{
    <(TA, TB, TC) as TiledGemm<TA, TB, TC>>::gemm(m, n, k, a, a_stride, b, b_stride, c, c_stride)
}

/// Dynamically dispatches tile matrix multiplication for a single tile of shape MxNxK.
///
/// # Safety
/// - Pointers must be valid and aligned as per the chosen backend requirements.
#[inline]
pub unsafe fn dispatch_tile_matmul<TA, TB, TC>(
    c: *mut TC, c_stride: usize,
    a: *const TA, a_stride: usize,
    b: *const TB, b_stride: usize,
)
where
    (TA, TB, TC): TiledGemm<TA, TB, TC>,
{
    <(TA, TB, TC) as TiledGemm<TA, TB, TC>>::dispatch_tile_matmul(c, c_stride, a, a_stride, b, b_stride)
}

/// Unpacks packed 4-bit signed integers (stored 2 per byte) into an 8-bit signed integer slice.
#[inline]
pub fn unpack_int4(packed: &[u8], unpacked: &mut [i8]) {
    let len = packed.len();
    assert!(unpacked.len() >= len * 2);
    for i in 0..len {
        let byte = packed[i] as i8;
        unpacked[2 * i] = (byte << 4) >> 4;
        unpacked[2 * i + 1] = byte >> 4;
    }
}
