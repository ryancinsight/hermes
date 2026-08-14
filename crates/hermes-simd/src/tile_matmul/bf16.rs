use super::{tile_loop_generic, validate_gemm_sizes, TiledGemm};
#[cfg(target_arch = "x86_64")]
use crate::cpu::{AmxSupport, Avx512Support};
use eunomia::{Bf16, F32};
use hermes_simd_core::view::{SimdError, TileMatrixMultiply};
use hermes_simd_intrinsics::Scalar;
#[cfg(target_arch = "x86_64")]
use hermes_simd_intrinsics::{AmxBf16, Avx512};

fn gemm_bf16_remainder(
    m: usize,
    n: usize,
    k: usize,
    a: &[Bf16],
    a_stride: usize,
    b: &[Bf16],
    b_stride: usize,
    c: &mut [F32],
    c_stride: usize,
) {
    let tile_m_bound = (m / 16) * 16;
    let tile_n_bound = (n / 16) * 16;
    let tile_k_bound = (k / 32) * 32;
    for r in 0..m {
        for col in 0..n {
            let depth_start = if r >= tile_m_bound || col >= tile_n_bound {
                0
            } else {
                tile_k_bound
            };
            if depth_start < k {
                let mut sum = 0.0f32;
                for kk in depth_start..k {
                    sum += a[r * a_stride + kk].to_f32() * b[kk * b_stride + col].to_f32();
                }
                c[r * c_stride + col] = F32(c[r * c_stride + col].0 + sum);
            }
        }
    }
}

impl TiledGemm<Bf16, Bf16, F32> for (Bf16, Bf16, F32) {
    #[inline]
    unsafe fn dispatch_tile_matmul(
        c: *mut F32,
        c_stride: usize,
        a: *const Bf16,
        a_stride: usize,
        b: *const Bf16,
        b_stride: usize,
    ) {
        #[cfg(target_arch = "x86_64")]
        {
            if <Bf16 as AmxSupport>::has_amx() && hermes_simd_intrinsics::AmxSession::is_active() {
                return <AmxBf16 as TileMatrixMultiply<
                    Bf16,
                    Bf16,
                    F32,
                    AmxBf16,
                    AmxBf16,
                    16,
                    16,
                    32,
                >>::tile_matmul(c, c_stride, a, a_stride, b, b_stride);
            }
            if <Bf16 as Avx512Support>::has_avx512() {
                return <Avx512 as TileMatrixMultiply<
                    Bf16,
                    Bf16,
                    F32,
                    Avx512,
                    Avx512,
                    16,
                    16,
                    32,
                >>::tile_matmul(c, c_stride, a, a_stride, b, b_stride);
            }
        }
        <Scalar as TileMatrixMultiply<Bf16, Bf16, F32, Scalar, Scalar, 16, 16, 32>>::tile_matmul(
            c, c_stride, a, a_stride, b, b_stride,
        );
    }

    #[inline]
    unsafe fn gemm(
        m: usize,
        n: usize,
        k: usize,
        a: &[Bf16],
        a_stride: usize,
        b: &[Bf16],
        b_stride: usize,
        c: &mut [F32],
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
                    <AmxBf16 as hermes_simd_intrinsics::x86_64::amx::AmxGemm<
                        Bf16,
                        Bf16,
                        F32,
                    >>::amx_gemm(
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
                    return Ok(());
                }
                crate::dispatcher::DispatchDecision::Avx512 => {
                    tile_loop_generic::<Bf16, Bf16, F32, Avx512, 16, 16, 32>(
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

                    gemm_bf16_remainder(m, n, k, a, a_stride, b, b_stride, c, c_stride);
                    return Ok(());
                }
                crate::dispatcher::DispatchDecision::AvxVnni
                | crate::dispatcher::DispatchDecision::Scalar => {}
            }
        }

        tile_loop_generic::<Bf16, Bf16, F32, Scalar, 16, 16, 32>(
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

        gemm_bf16_remainder(m, n, k, a, a_stride, b, b_stride, c, c_stride);
        Ok(())
    }
}
