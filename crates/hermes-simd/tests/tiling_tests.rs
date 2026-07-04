use hermes_simd::*;

#[test]
fn test_tiled_dot_matches_dot() {
    let a: Vec<f32> = (0..64).map(|i| i as f32).collect();
    let b: Vec<f32> = (0..64).map(|i| (64 - i) as f32).collect();

    let view_a = SimdView::<f32, Scalar, Unaligned, Unmasked, &[f32]>::new(&a).unwrap();
    let view_b = SimdView::<f32, Scalar, Unaligned, Unmasked, &[f32]>::new(&b).unwrap();

    let expected = dot::<f32>(&a, &b).unwrap();
    let tiled_1 = tiled_dot::<f32, Scalar, Unaligned, 1>(&view_a, &view_b).unwrap();
    let tiled_4 = tiled_dot::<f32, Scalar, Unaligned, 4>(&view_a, &view_b).unwrap();

    assert!(
        (tiled_1 - expected).abs() < 1e-3,
        "TILE_M=1: {tiled_1} vs {expected}"
    );
    assert!(
        (tiled_4 - expected).abs() < 1e-3,
        "TILE_M=4: {tiled_4} vs {expected}"
    );
}

#[test]
fn test_tiled_gemv_correctness() {
    let a = [
        1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
    ];
    let x = [1.0f32, 2.0, 3.0, 4.0];
    let mut y = [0.0f32; 4];

    let a_view = SimdView::<f32, Scalar, Unaligned>::new(&a).unwrap();
    let x_view = SimdView::<f32, Scalar, Unaligned>::new(&x).unwrap();
    tiled_gemv::<f32, Scalar, Unaligned, 2>(&a_view, &x_view, &mut y, 4, 4).unwrap();

    assert_eq!(y, [30.0, 70.0, 110.0, 150.0]);
}

/// f32 GEMV column-tail differential through the public dispatcher. `ncols = 21`
/// on an AVX2 host is two full 8-lane groups plus a 5-lane masked tail, and
/// `nrows = 11` gives both TILE_M-blocked rows and a row remainder — every
/// masked-tail path in one shape. Dyadic-exact operands keep the fused-multiply
/// masked-tail reduction bitwise-equal to the sequential scalar reference.
#[test]
fn test_gemv_f32_column_tail_differential() {
    let nrows = 11usize;
    let ncols = 21usize;
    let a: Vec<f32> = (0..nrows * ncols)
        .map(|i| ((i % 9) as f32 - 4.0) * 0.25)
        .collect();
    let x: Vec<f32> = (0..ncols).map(|i| ((i % 5) as f32 - 2.0) * 0.5).collect();
    let y_init: Vec<f32> = (0..nrows).map(|i| (i % 3) as f32 - 1.0).collect();

    let mut y = y_init.clone();
    gemv::<f32>(&a, &x, &mut y, nrows, ncols).unwrap();

    let mut want = y_init;
    for (row, w) in want.iter_mut().enumerate() {
        let mut sum = 0.0f32;
        for col in 0..ncols {
            sum += a[row * ncols + col] * x[col];
        }
        *w += sum;
    }
    assert_eq!(y, want, "gemv f32 column tail diverges from reference");
}

#[test]
fn test_tiled_gemm() {
    let a = vec![
        1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
    ];
    let b = vec![
        1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
    ];
    let mut c = vec![0.0f32; 16];

    tiled_gemm(&a, &b, &mut c, 4, 4, 3).unwrap();

    let expected = vec![
        38.0, 44.0, 50.0, 56.0, 83.0, 98.0, 113.0, 128.0, 128.0, 152.0, 176.0, 200.0, 173.0, 206.0,
        239.0, 272.0,
    ];

    for i in 0..16 {
        assert!(
            (c[i] - expected[i]).abs() < 1e-4,
            "At index {}, {} != {}",
            i,
            c[i],
            expected[i]
        );
    }
}

#[test]
fn test_tile_matrix_multiply_bf16() {
    use half::bf16;
    let mut c = vec![0.0f32; 16 * 16];
    let a = vec![bf16::from_f32(1.0); 16 * 32];
    let b = vec![bf16::from_f32(2.0); 32 * 16];

    unsafe {
        dispatch_tile_matmul::<bf16, bf16, f32>(c.as_mut_ptr(), 16, a.as_ptr(), 32, b.as_ptr(), 16);
    }

    for val in c {
        assert_eq!(val, 64.0);
    }
}

#[test]
fn test_tile_matrix_multiply_int8() {
    let mut c = vec![0i32; 16 * 16];
    let a = vec![1i8; 16 * 64];
    let b = vec![2i8; 64 * 16];

    unsafe {
        dispatch_tile_matmul::<i8, i8, i32>(c.as_mut_ptr(), 16, a.as_ptr(), 64, b.as_ptr(), 16);
    }

    for val in c {
        assert_eq!(val, 128);
    }
}

#[test]
fn test_gemm_bf16_high_level() {
    use half::bf16;
    let m = 35;
    let n = 20;
    let k = 36;
    let a = vec![bf16::from_f32(1.5); m * k];
    let b = vec![bf16::from_f32(2.0); k * n];
    let mut c = vec![0.0f32; m * n];

    unsafe {
        gemm::<bf16, bf16, f32>(m, n, k, &a, k, &b, n, &mut c, n).unwrap();
    }

    for val in c {
        assert_eq!(val, 108.0);
    }
}

#[test]
fn test_gemm_int8_high_level() {
    let m = 35;
    let n = 20;
    let k = 70;
    let a = vec![2i8; m * k];
    let b = vec![3i8; k * n];
    let mut c = vec![0i32; m * n];

    unsafe {
        gemm::<i8, i8, i32>(m, n, k, &a, k, &b, n, &mut c, n).unwrap();
    }

    for val in c {
        assert_eq!(val, 420);
    }
}

/// Column-tail differential: `n = 45` on an AVX2 host splits into one full
/// 32-column register block, one full 8-lane masked group, and a 5-lane
/// partial-mask group — every tail case in one shape. Dyadic-exact entries
/// (multiples of 0.5 with k ≤ 16) keep each product and partial sum exactly
/// representable in f32, so the fused-multiply tail must match the sequential
/// scalar reference **bitwise**.
#[test]
fn test_tiled_gemm_column_tail_differential() {
    let m = 7usize;
    let n = 45usize;
    let k = 13usize;
    let a: Vec<f32> = (0..m * k).map(|i| ((i % 9) as f32 - 4.0) * 0.5).collect();
    let b: Vec<f32> = (0..k * n).map(|i| ((i % 7) as f32 - 3.0) * 0.5).collect();
    // Nonzero C exercises the accumulate contract on both tile and tail paths.
    let c_init: Vec<f32> = (0..m * n).map(|i| (i % 11) as f32 - 5.0).collect();

    let mut c = c_init.clone();
    tiled_gemm(&a, &b, &mut c, m, n, k).unwrap();

    let mut c_ref = c_init;
    for row in 0..m {
        for col in 0..n {
            let mut sum = 0.0f32;
            for kk in 0..k {
                sum += a[row * k + kk] * b[kk * n + col];
            }
            c_ref[row * n + col] += sum;
        }
    }
    assert_eq!(c, c_ref, "tiled GEMM column tail diverges from reference");
}

/// Differential: the dispatched int8 GEMM (whatever backend the host selects —
/// AMX, AVX-512 VNNI, 256-bit AVX-VNNI, or scalar tiles) must equal an
/// independent wrapping scalar triple loop **bitwise**. Integer accumulation is
/// associative mod 2^32, so reordering across backends cannot change the
/// result; any divergence is a kernel defect. Full-range signed inputs
/// (including -128) exercise the AVX-VNNI `vpdpbusd` +128 bias correction, and
/// the non-multiple shape (m,n % 16 != 0, k % 64 != 0) covers every tile-tail
/// combination.
#[test]
fn test_gemm_int8_signed_differential() {
    let m = 37;
    let n = 29;
    let k = 130;
    let a: Vec<i8> = (0..m * k)
        .map(|i| ((i * 89 + 3) % 256) as u8 as i8)
        .collect();
    let b: Vec<i8> = (0..k * n)
        .map(|i| ((i * 41 + 128) % 256) as u8 as i8)
        .collect();
    // Nonzero C exercises the accumulate contract end-to-end.
    let c_init: Vec<i32> = (0..m * n)
        .map(|i| (i as i32).wrapping_mul(7919) - 40000)
        .collect();

    let mut c = c_init.clone();
    unsafe {
        gemm::<i8, i8, i32>(m, n, k, &a, k, &b, n, &mut c, n).unwrap();
    }

    let mut c_ref = c_init;
    for r in 0..m {
        for col in 0..n {
            let mut sum = 0i32;
            for kk in 0..k {
                sum = sum.wrapping_add((a[r * k + kk] as i32) * (b[kk * n + col] as i32));
            }
            c_ref[r * n + col] += sum;
        }
    }

    assert_eq!(
        c, c_ref,
        "dispatched int8 GEMM diverges from scalar reference"
    );
}

#[test]
fn test_gemm_bf16_size_16() {
    use half::bf16;
    let m = 16;
    let n = 16;
    let k = 16;
    let a = vec![bf16::from_f32(1.5); m * k];
    let b = vec![bf16::from_f32(2.0); k * n];
    let mut c = vec![0.0f32; m * n];

    unsafe {
        gemm::<bf16, bf16, f32>(m, n, k, &a, k, &b, n, &mut c, n).unwrap();
    }
}

#[test]
fn test_tiling_strategy_trait_direct() {
    let a: Vec<f32> = (0..64).map(|i| i as f32).collect();
    let b: Vec<f32> = (0..64).map(|i| (64 - i) as f32).collect();

    let view_a = SimdView::<f32, Scalar, Unaligned, Unmasked, &[f32]>::new(&a).unwrap();
    let view_b = SimdView::<f32, Scalar, Unaligned, Unmasked, &[f32]>::new(&b).unwrap();

    let expected = dot::<f32>(&a, &b).unwrap();
    let tiled_res =
        <TilingPolicy<4, 1> as TilingStrategy<f32, Scalar, Unaligned>>::dot(&view_a, &view_b)
            .unwrap();

    assert!((tiled_res - expected).abs() < 1e-3);
}
