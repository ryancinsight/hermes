use super::{raw, AmxConfig, AmxInt8, AmxSession};
use hermes_numeric::{I32, I8};
use hermes_simd_core::view::TileMatrixMultiply;

macro_rules! impl_tile_matmul_int8 {
    ($t_in:ty, $t_out:ty) => {
        impl TileMatrixMultiply<$t_in, $t_in, $t_out, AmxInt8, AmxInt8, 16, 16, 64> for AmxInt8 {
            unsafe fn tile_matmul(
                c: *mut $t_out,
                c_stride: usize,
                a: *const $t_in,
                a_stride: usize,
                b: *const $t_in,
                b_stride: usize,
            ) {
                let is_configured = super::ACTIVE_CONFIG.with(|f| f.get().is_some());

                if !is_configured {
                    let config = AmxConfig::new_uniform(16, 64);
                    raw::ldtilecfg(&config);
                }

                // A is M=16 rows, K=64 cols.
                raw::tileloadd(0, a as *const _, a_stride as isize);
                // B is K=64 rows, N=16 cols.
                raw::tileloadd(1, b as *const _, b_stride as isize);
                // C is M=16 rows, N=16 cols.
                raw::tileloadd(2, c as *const _, (c_stride * 4) as isize);

                raw::tdpbssd(2, 0, 1);

                raw::tilestored(2, c as *mut _, (c_stride * 4) as isize);

                if !is_configured {
                    raw::tilerelease();
                }
            }
        }
    };
}

impl_tile_matmul_int8!(i8, i32);
impl_tile_matmul_int8!(I8, I32);

impl super::AmxGemm<i8, i8, i32> for AmxInt8 {
    #[inline]
    unsafe fn amx_gemm(
        m: usize,
        n: usize,
        k: usize,
        a: *const i8,
        a_stride: usize,
        b: *const i8,
        b_stride: usize,
        c: *mut i32,
        c_stride: usize,
    ) {
        let config = AmxConfig::for_dimensions(m, n, k, 1);
        let _session = AmxSession::new(&config);

        let mut i = 0;
        while i + 32 <= m {
            let mut j = 0;
            while j + 32 <= n {
                raw::tileloadd(
                    2,
                    c.add(i * c_stride + j) as *const _,
                    (c_stride * 4) as isize,
                );
                raw::tileloadd(
                    3,
                    c.add(i * c_stride + j + 16) as *const _,
                    (c_stride * 4) as isize,
                );
                raw::tileloadd(
                    4,
                    c.add((i + 16) * c_stride + j) as *const _,
                    (c_stride * 4) as isize,
                );
                raw::tileloadd(
                    5,
                    c.add((i + 16) * c_stride + j + 16) as *const _,
                    (c_stride * 4) as isize,
                );

                let mut kk = 0;
                while kk + 64 <= k {
                    raw::tileloadd(0, a.add(i * a_stride + kk) as *const _, a_stride as isize);
                    raw::tileloadd(
                        1,
                        a.add((i + 16) * a_stride + kk) as *const _,
                        a_stride as isize,
                    );

                    raw::tileloadd(6, b.add(kk * b_stride + j) as *const _, b_stride as isize);
                    raw::tileloadd(
                        7,
                        b.add(kk * b_stride + j + 16) as *const _,
                        b_stride as isize,
                    );

                    raw::tdpbssd(2, 0, 6);
                    raw::tdpbssd(3, 0, 7);
                    raw::tdpbssd(4, 1, 6);
                    raw::tdpbssd(5, 1, 7);

                    kk += 64;
                }

                raw::tilestored(
                    2,
                    c.add(i * c_stride + j) as *mut _,
                    (c_stride * 4) as isize,
                );
                raw::tilestored(
                    3,
                    c.add(i * c_stride + j + 16) as *mut _,
                    (c_stride * 4) as isize,
                );
                raw::tilestored(
                    4,
                    c.add((i + 16) * c_stride + j) as *mut _,
                    (c_stride * 4) as isize,
                );
                raw::tilestored(
                    5,
                    c.add((i + 16) * c_stride + j + 16) as *mut _,
                    (c_stride * 4) as isize,
                );

                j += 32;
            }

            while j + 16 <= n {
                raw::tileloadd(
                    2,
                    c.add(i * c_stride + j) as *const _,
                    (c_stride * 4) as isize,
                );
                raw::tileloadd(
                    4,
                    c.add((i + 16) * c_stride + j) as *const _,
                    (c_stride * 4) as isize,
                );

                let mut kk = 0;
                while kk + 64 <= k {
                    raw::tileloadd(0, a.add(i * a_stride + kk) as *const _, a_stride as isize);
                    raw::tileloadd(
                        1,
                        a.add((i + 16) * a_stride + kk) as *const _,
                        a_stride as isize,
                    );
                    raw::tileloadd(6, b.add(kk * b_stride + j) as *const _, b_stride as isize);

                    raw::tdpbssd(2, 0, 6);
                    raw::tdpbssd(4, 1, 6);
                    kk += 64;
                }

                raw::tilestored(
                    2,
                    c.add(i * c_stride + j) as *mut _,
                    (c_stride * 4) as isize,
                );
                raw::tilestored(
                    4,
                    c.add((i + 16) * c_stride + j) as *mut _,
                    (c_stride * 4) as isize,
                );
                j += 16;
            }

            i += 32;
        }

        while i + 16 <= m {
            let mut j = 0;
            while j + 16 <= n {
                raw::tileloadd(
                    2,
                    c.add(i * c_stride + j) as *const _,
                    (c_stride * 4) as isize,
                );

                let mut kk = 0;
                while kk + 64 <= k {
                    raw::tileloadd(0, a.add(i * a_stride + kk) as *const _, a_stride as isize);
                    raw::tileloadd(6, b.add(kk * b_stride + j) as *const _, b_stride as isize);

                    raw::tdpbssd(2, 0, 6);
                    kk += 64;
                }

                raw::tilestored(
                    2,
                    c.add(i * c_stride + j) as *mut _,
                    (c_stride * 4) as isize,
                );
                j += 16;
            }
            i += 16;
        }

        let amx_m_bound = (m / 16) * 16;
        let amx_n_bound = (n / 16) * 16;
        let amx_k_bound = (k / 64) * 64;

        if i < m || k % 64 != 0 || n % 16 != 0 {
            for r in 0..m {
                for col in 0..n {
                    if r >= amx_m_bound || col >= amx_n_bound {
                        let mut sum = 0i32;
                        for kk in 0..k {
                            sum = sum.wrapping_add(
                                (*a.add(r * a_stride + kk) as i32)
                                    * (*b.add(kk * b_stride + col) as i32),
                            );
                        }
                        *c.add(r * c_stride + col) += sum;
                    } else if amx_k_bound < k {
                        let mut sum = 0i32;
                        for kk in amx_k_bound..k {
                            sum = sum.wrapping_add(
                                (*a.add(r * a_stride + kk) as i32)
                                    * (*b.add(kk * b_stride + col) as i32),
                            );
                        }
                        *c.add(r * c_stride + col) += sum;
                    }
                }
            }
        }
    }
}

impl super::AmxGemm<I8, I8, I32> for AmxInt8 {
    #[inline]
    unsafe fn amx_gemm(
        m: usize,
        n: usize,
        k: usize,
        a: *const I8,
        a_stride: usize,
        b: *const I8,
        b_stride: usize,
        c: *mut I32,
        c_stride: usize,
    ) {
        let config = AmxConfig::for_dimensions(m, n, k, 1);
        let _session = AmxSession::new(&config);

        let mut i = 0;
        while i + 32 <= m {
            let mut j = 0;
            while j + 32 <= n {
                raw::tileloadd(
                    2,
                    c.add(i * c_stride + j) as *const _,
                    (c_stride * 4) as isize,
                );
                raw::tileloadd(
                    3,
                    c.add(i * c_stride + j + 16) as *const _,
                    (c_stride * 4) as isize,
                );
                raw::tileloadd(
                    4,
                    c.add((i + 16) * c_stride + j) as *const _,
                    (c_stride * 4) as isize,
                );
                raw::tileloadd(
                    5,
                    c.add((i + 16) * c_stride + j + 16) as *const _,
                    (c_stride * 4) as isize,
                );

                let mut kk = 0;
                while kk + 64 <= k {
                    raw::tileloadd(0, a.add(i * a_stride + kk) as *const _, a_stride as isize);
                    raw::tileloadd(
                        1,
                        a.add((i + 16) * a_stride + kk) as *const _,
                        a_stride as isize,
                    );

                    raw::tileloadd(6, b.add(kk * b_stride + j) as *const _, b_stride as isize);
                    raw::tileloadd(
                        7,
                        b.add(kk * b_stride + j + 16) as *const _,
                        b_stride as isize,
                    );

                    raw::tdpbssd(2, 0, 6);
                    raw::tdpbssd(3, 0, 7);
                    raw::tdpbssd(4, 1, 6);
                    raw::tdpbssd(5, 1, 7);

                    kk += 64;
                }

                raw::tilestored(
                    2,
                    c.add(i * c_stride + j) as *mut _,
                    (c_stride * 4) as isize,
                );
                raw::tilestored(
                    3,
                    c.add(i * c_stride + j + 16) as *mut _,
                    (c_stride * 4) as isize,
                );
                raw::tilestored(
                    4,
                    c.add((i + 16) * c_stride + j) as *mut _,
                    (c_stride * 4) as isize,
                );
                raw::tilestored(
                    5,
                    c.add((i + 16) * c_stride + j + 16) as *mut _,
                    (c_stride * 4) as isize,
                );

                j += 32;
            }

            while j + 16 <= n {
                raw::tileloadd(
                    2,
                    c.add(i * c_stride + j) as *const _,
                    (c_stride * 4) as isize,
                );
                raw::tileloadd(
                    4,
                    c.add((i + 16) * c_stride + j) as *const _,
                    (c_stride * 4) as isize,
                );

                let mut kk = 0;
                while kk + 64 <= k {
                    raw::tileloadd(0, a.add(i * a_stride + kk) as *const _, a_stride as isize);
                    raw::tileloadd(
                        1,
                        a.add((i + 16) * a_stride + kk) as *const _,
                        a_stride as isize,
                    );
                    raw::tileloadd(6, b.add(kk * b_stride + j) as *const _, b_stride as isize);

                    raw::tdpbssd(2, 0, 6);
                    raw::tdpbssd(4, 1, 6);
                    kk += 64;
                }

                raw::tilestored(
                    2,
                    c.add(i * c_stride + j) as *mut _,
                    (c_stride * 4) as isize,
                );
                raw::tilestored(
                    4,
                    c.add((i + 16) * c_stride + j) as *mut _,
                    (c_stride * 4) as isize,
                );
                j += 16;
            }

            i += 32;
        }

        while i + 16 <= m {
            let mut j = 0;
            while j + 16 <= n {
                raw::tileloadd(
                    2,
                    c.add(i * c_stride + j) as *const _,
                    (c_stride * 4) as isize,
                );

                let mut kk = 0;
                while kk + 64 <= k {
                    raw::tileloadd(0, a.add(i * a_stride + kk) as *const _, a_stride as isize);
                    raw::tileloadd(6, b.add(kk * b_stride + j) as *const _, b_stride as isize);

                    raw::tdpbssd(2, 0, 6);
                    kk += 64;
                }

                raw::tilestored(
                    2,
                    c.add(i * c_stride + j) as *mut _,
                    (c_stride * 4) as isize,
                );
                j += 16;
            }
            i += 16;
        }

        let amx_m_bound = (m / 16) * 16;
        let amx_n_bound = (n / 16) * 16;
        let amx_k_bound = (k / 64) * 64;

        if i < m || k % 64 != 0 || n % 16 != 0 {
            for r in 0..m {
                for col in 0..n {
                    if r >= amx_m_bound || col >= amx_n_bound {
                        let mut sum = 0i32;
                        for kk in 0..k {
                            sum = sum.wrapping_add(
                                (a.add(r * a_stride + kk).read().0 as i32)
                                    * (b.add(kk * b_stride + col).read().0 as i32),
                            );
                        }
                        *c.add(r * c_stride + col) = I32(c.add(r * c_stride + col).read().0 + sum);
                    } else if amx_k_bound < k {
                        let mut sum = 0i32;
                        for kk in amx_k_bound..k {
                            sum = sum.wrapping_add(
                                (a.add(r * a_stride + kk).read().0 as i32)
                                    * (b.add(kk * b_stride + col).read().0 as i32),
                            );
                        }
                        *c.add(r * c_stride + col) = I32(c.add(r * c_stride + col).read().0 + sum);
                    }
                }
            }
        }
    }
}
