use super::{tile_loop_generic, validate_gemm_sizes, TiledGemm};
#[cfg(target_arch = "x86_64")]
use crate::cpu::{AmxSupport, Avx512Support};
use hermes_numeric::{I32, I8};
use hermes_simd_core::view::{SimdError, TileMatrixMultiply};
use hermes_simd_intrinsics::Scalar;
#[cfg(target_arch = "x86_64")]
use hermes_simd_intrinsics::{AmxInt8, Avx512};

impl TiledGemm<i8, i8, i32> for (i8, i8, i32) {
    #[inline]
    unsafe fn dispatch_tile_matmul(
        c: *mut i32,
        c_stride: usize,
        a: *const i8,
        a_stride: usize,
        b: *const i8,
        b_stride: usize,
    ) {
        #[cfg(target_arch = "x86_64")]
        {
            if <i8 as AmxSupport>::has_amx() && hermes_simd_intrinsics::AmxSession::is_active() {
                return <AmxInt8 as TileMatrixMultiply<
                    i8,
                    i8,
                    i32,
                    AmxInt8,
                    AmxInt8,
                    16,
                    16,
                    64,
                >>::tile_matmul(c, c_stride, a, a_stride, b, b_stride);
            }
            if <i8 as Avx512Support>::has_avx512() {
                return <Avx512 as TileMatrixMultiply<i8, i8, i32, Avx512, Avx512, 16, 16, 64>>::tile_matmul(c, c_stride, a, a_stride, b, b_stride);
            }
        }
        <Scalar as TileMatrixMultiply<i8, i8, i32, Scalar, Scalar, 16, 16, 64>>::tile_matmul(
            c, c_stride, a, a_stride, b, b_stride,
        );
    }

    #[inline]
    unsafe fn gemm(
        m: usize,
        n: usize,
        k: usize,
        a: &[i8],
        a_stride: usize,
        b: &[i8],
        b_stride: usize,
        c: &mut [i32],
        c_stride: usize,
    ) -> Result<(), SimdError> {
        validate_gemm_sizes(
            a.len(),
            b.len(),
            c.len(),
            m,
            n,
            k,
            a_stride,
            b_stride,
            c_stride,
        )?;

        #[cfg(target_arch = "x86_64")]
        {
            let decision = crate::dispatcher::AdaptiveDispatcher::select_backend(
                m,
                n,
                k,
                a.as_ptr(),
                a.len(),
                b.as_ptr(),
                b.len(),
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
                        m,
                        n,
                        k,
                        a.as_ptr(),
                        a_stride,
                        b.as_ptr(),
                        b_stride,
                        c.as_mut_ptr(),
                        c_stride,
                    );

                    let amx_m_bound = (m / 16) * 16;
                    let amx_n_bound = (n / 16) * 16;
                    let amx_k_bound = (k / 64) * 64;
                    for r in 0..m {
                        for col in 0..n {
                            if r >= amx_m_bound || col >= amx_n_bound {
                                let mut sum = 0i32;
                                for kk in 0..k {
                                    sum = sum.wrapping_add(
                                        (a[r * a_stride + kk] as i32)
                                            * (b[kk * b_stride + col] as i32),
                                    );
                                }
                                c[r * c_stride + col] += sum;
                            } else if amx_k_bound < k {
                                let mut sum = 0i32;
                                for kk in amx_k_bound..k {
                                    sum = sum.wrapping_add(
                                        (a[r * a_stride + kk] as i32)
                                            * (b[kk * b_stride + col] as i32),
                                    );
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
            m,
            n,
            k,
            a.as_ptr(),
            a_stride,
            b.as_ptr(),
            b_stride,
            c.as_mut_ptr(),
            c_stride,
        );

        let amx_m_bound = (m / 16) * 16;
        let amx_n_bound = (n / 16) * 16;
        let amx_k_bound = (k / 64) * 64;
        for r in 0..m {
            for col in 0..n {
                if r >= amx_m_bound || col >= amx_n_bound {
                    let mut sum = 0i32;
                    for kk in 0..k {
                        sum = sum.wrapping_add(
                            (a[r * a_stride + kk] as i32) * (b[kk * b_stride + col] as i32),
                        );
                    }
                    c[r * c_stride + col] += sum;
                } else if amx_k_bound < k {
                    let mut sum = 0i32;
                    for kk in amx_k_bound..k {
                        sum = sum.wrapping_add(
                            (a[r * a_stride + kk] as i32) * (b[kk * b_stride + col] as i32),
                        );
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
        c: *mut I32,
        c_stride: usize,
        a: *const I8,
        a_stride: usize,
        b: *const I8,
        b_stride: usize,
    ) {
        #[cfg(target_arch = "x86_64")]
        {
            if <i8 as AmxSupport>::has_amx() && hermes_simd_intrinsics::AmxSession::is_active() {
                return <AmxInt8 as TileMatrixMultiply<
                    I8,
                    I8,
                    I32,
                    AmxInt8,
                    AmxInt8,
                    16,
                    16,
                    64,
                >>::tile_matmul(c, c_stride, a, a_stride, b, b_stride);
            }
            if <i8 as Avx512Support>::has_avx512() {
                return <Avx512 as TileMatrixMultiply<I8, I8, I32, Avx512, Avx512, 16, 16, 64>>::tile_matmul(c, c_stride, a, a_stride, b, b_stride);
            }
        }
        <Scalar as TileMatrixMultiply<I8, I8, I32, Scalar, Scalar, 16, 16, 64>>::tile_matmul(
            c, c_stride, a, a_stride, b, b_stride,
        );
    }

    #[inline]
    unsafe fn gemm(
        m: usize,
        n: usize,
        k: usize,
        a: &[I8],
        a_stride: usize,
        b: &[I8],
        b_stride: usize,
        c: &mut [I32],
        c_stride: usize,
    ) -> Result<(), SimdError> {
        validate_gemm_sizes(
            a.len(),
            b.len(),
            c.len(),
            m,
            n,
            k,
            a_stride,
            b_stride,
            c_stride,
        )?;

        #[cfg(target_arch = "x86_64")]
        {
            let decision = crate::dispatcher::AdaptiveDispatcher::select_backend(
                m,
                n,
                k,
                a.as_ptr(),
                a.len(),
                b.as_ptr(),
                b.len(),
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
                        m,
                        n,
                        k,
                        a.as_ptr(),
                        a_stride,
                        b.as_ptr(),
                        b_stride,
                        c.as_mut_ptr(),
                        c_stride,
                    );

                    let amx_m_bound = (m / 16) * 16;
                    let amx_n_bound = (n / 16) * 16;
                    let amx_k_bound = (k / 64) * 64;
                    for r in 0..m {
                        for col in 0..n {
                            if r >= amx_m_bound || col >= amx_n_bound {
                                let mut sum = 0i32;
                                for kk in 0..k {
                                    sum = sum.wrapping_add(
                                        (a[r * a_stride + kk].0 as i32)
                                            * (b[kk * b_stride + col].0 as i32),
                                    );
                                }
                                c[r * c_stride + col] = I32(c[r * c_stride + col].0 + sum);
                            } else if amx_k_bound < k {
                                let mut sum = 0i32;
                                for kk in amx_k_bound..k {
                                    sum = sum.wrapping_add(
                                        (a[r * a_stride + kk].0 as i32)
                                            * (b[kk * b_stride + col].0 as i32),
                                    );
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
            m,
            n,
            k,
            a.as_ptr(),
            a_stride,
            b.as_ptr(),
            b_stride,
            c.as_mut_ptr(),
            c_stride,
        );

        let amx_m_bound = (m / 16) * 16;
        let amx_n_bound = (n / 16) * 16;
        let amx_k_bound = (k / 64) * 64;
        for r in 0..m {
            for col in 0..n {
                if r >= amx_m_bound || col >= amx_n_bound {
                    let mut sum = 0i32;
                    for kk in 0..k {
                        sum = sum.wrapping_add(
                            (a[r * a_stride + kk].0 as i32) * (b[kk * b_stride + col].0 as i32),
                        );
                    }
                    c[r * c_stride + col] = I32(c[r * c_stride + col].0 + sum);
                } else if amx_k_bound < k {
                    let mut sum = 0i32;
                    for kk in amx_k_bound..k {
                        sum = sum.wrapping_add(
                            (a[r * a_stride + kk].0 as i32) * (b[kk * b_stride + col].0 as i32),
                        );
                    }
                    c[r * c_stride + col] = I32(c[r * c_stride + col].0 + sum);
                }
            }
        }
        Ok(())
    }
}
