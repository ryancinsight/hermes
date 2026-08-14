use super::{pack::pack_rhs_panel, raw, AmxBf16, AmxConfig, AmxSession};
use eunomia::{Bf16, F32};
use hermes_simd_core::view::TileMatrixMultiply;

impl TileMatrixMultiply<Bf16, Bf16, F32, AmxBf16, AmxBf16, 16, 16, 32> for AmxBf16 {
    unsafe fn tile_matmul(
        c: *mut F32,
        c_stride: usize,
        a: *const Bf16,
        a_stride: usize,
        b: *const Bf16,
        b_stride: usize,
    ) {
        let is_configured = {
            #[cfg(feature = "std")]
            {
                super::ACTIVE_CONFIG.with(|f| f.get().is_some())
            }
            #[cfg(not(feature = "std"))]
            {
                false
            }
        };

        if !is_configured {
            let config = AmxConfig::new_uniform(16, 64);
            raw::ldtilecfg(&config);
        }

        // A is M=16 rows, K=32 cols. Row stride in bytes = a_stride * 2.
        raw::tileloadd(0, a.cast(), (a_stride * 2) as isize);
        // B is packed as K/2=16 rows, 2N=32 BF16 elements for AMX dot products.
        let mut b_tile = [Bf16::default(); 16 * 32];
        pack_rhs_panel::<_, 2>(b, b_stride, 0, 0, 32, &mut b_tile);
        raw::tileloadd(1, b_tile.as_ptr().cast(), 64);
        // C is M=16 rows, N=16 cols. Row stride in bytes = c_stride * 4.
        raw::tileloadd(2, c as *const _, (c_stride * 4) as isize);

        raw::tdpbf16ps(2, 0, 1);

        raw::tilestored(2, c.cast(), (c_stride * 4) as isize);

        if !is_configured {
            raw::tilerelease();
        }
    }
}

impl super::AmxGemm<Bf16, Bf16, F32> for AmxBf16 {
    #[inline]
    #[expect(
        clippy::too_many_lines,
        reason = "AMX BF16 tiling keeps tile configuration, packing, dispatch, and tails in one unsafe boundary"
    )]
    unsafe fn amx_gemm(
        m: usize,
        n: usize,
        k: usize,
        a: *const Bf16,
        a_stride: usize,
        b: *const Bf16,
        b_stride: usize,
        c: *mut F32,
        c_stride: usize,
    ) {
        let config = AmxConfig::for_dimensions(m, n, k, 2);
        let _session = AmxSession::new(&config)
            .expect("invariant: AMX bf16 dispatch requires runtime AMX support");

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

                let mut b_tile_0 = [Bf16::default(); 16 * 32];
                let mut b_tile_1 = [Bf16::default(); 16 * 32];
                let mut kk = 0;
                while kk + 32 <= k {
                    raw::tileloadd(0, a.add(i * a_stride + kk).cast(), (a_stride * 2) as isize);
                    raw::tileloadd(
                        1,
                        a.add((i + 16) * a_stride + kk).cast(),
                        (a_stride * 2) as isize,
                    );

                    pack_rhs_panel::<_, 2>(b, b_stride, j, kk, 32, &mut b_tile_0);
                    pack_rhs_panel::<_, 2>(b, b_stride, j + 16, kk, 32, &mut b_tile_1);
                    raw::tileloadd(6, b_tile_0.as_ptr().cast(), 64);
                    raw::tileloadd(7, b_tile_1.as_ptr().cast(), 64);

                    raw::tdpbf16ps(2, 0, 6);
                    raw::tdpbf16ps(3, 0, 7);
                    raw::tdpbf16ps(4, 1, 6);
                    raw::tdpbf16ps(5, 1, 7);

                    kk += 32;
                }

                raw::tilestored(2, c.add(i * c_stride + j).cast(), (c_stride * 4) as isize);
                raw::tilestored(
                    3,
                    c.add(i * c_stride + j + 16).cast(),
                    (c_stride * 4) as isize,
                );
                raw::tilestored(
                    4,
                    c.add((i + 16) * c_stride + j).cast(),
                    (c_stride * 4) as isize,
                );
                raw::tilestored(
                    5,
                    c.add((i + 16) * c_stride + j + 16).cast(),
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

                let mut b_tile = [Bf16::default(); 16 * 32];
                let mut kk = 0;
                while kk + 32 <= k {
                    raw::tileloadd(0, a.add(i * a_stride + kk).cast(), (a_stride * 2) as isize);
                    raw::tileloadd(
                        1,
                        a.add((i + 16) * a_stride + kk).cast(),
                        (a_stride * 2) as isize,
                    );
                    pack_rhs_panel::<_, 2>(b, b_stride, j, kk, 32, &mut b_tile);
                    raw::tileloadd(6, b_tile.as_ptr().cast(), 64);

                    raw::tdpbf16ps(2, 0, 6);
                    raw::tdpbf16ps(4, 1, 6);
                    kk += 32;
                }

                raw::tilestored(2, c.add(i * c_stride + j).cast(), (c_stride * 4) as isize);
                raw::tilestored(
                    4,
                    c.add((i + 16) * c_stride + j).cast(),
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

                let mut b_tile = [Bf16::default(); 16 * 32];
                let mut kk = 0;
                while kk + 32 <= k {
                    raw::tileloadd(0, a.add(i * a_stride + kk).cast(), (a_stride * 2) as isize);
                    pack_rhs_panel::<_, 2>(b, b_stride, j, kk, 32, &mut b_tile);
                    raw::tileloadd(6, b_tile.as_ptr().cast(), 64);

                    raw::tdpbf16ps(2, 0, 6);
                    kk += 32;
                }

                raw::tilestored(2, c.add(i * c_stride + j).cast(), (c_stride * 4) as isize);
                j += 16;
            }
            i += 16;
        }

        let amx_m_bound = (m / 16) * 16;
        let amx_n_bound = (n / 16) * 16;
        let amx_k_bound = (k / 32) * 32;

        if i < m || k % 32 != 0 || n % 16 != 0 {
            for r in 0..m {
                for col in 0..n {
                    if r >= amx_m_bound || col >= amx_n_bound {
                        let mut sum = 0.0f32;
                        for kk in 0..k {
                            sum +=
                                hermes_simd_core::FloatElement::to_f32(*a.add(r * a_stride + kk))
                                    * hermes_simd_core::FloatElement::to_f32(
                                        *b.add(kk * b_stride + col),
                                    );
                        }
                        *c.add(r * c_stride + col) = F32(c.add(r * c_stride + col).read().0 + sum);
                    } else if amx_k_bound < k {
                        let mut sum = 0.0f32;
                        for kk in amx_k_bound..k {
                            sum +=
                                hermes_simd_core::FloatElement::to_f32(*a.add(r * a_stride + kk))
                                    * hermes_simd_core::FloatElement::to_f32(
                                        *b.add(kk * b_stride + col),
                                    );
                        }
                        *c.add(r * c_stride + col) = F32(c.add(r * c_stride + col).read().0 + sum);
                    }
                }
            }
        }
    }
}
