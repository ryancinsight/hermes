#![expect(
    clippy::float_cmp,
    reason = "These integration tests assert exact manufactured dense reference values"
)]
#![expect(
    clippy::too_many_lines,
    reason = "The dense backend matrix is one shared value-semantic conformance test"
)]
use hermes_simd::*;

#[test]
fn test_size_invariants() {
    use core::mem::size_of;
    assert_eq!(
        size_of::<SimdView<'_, f32, Scalar, Unaligned, Unmasked, &[f32]>>(),
        size_of::<&[f32]>()
    );
}

#[test]
fn test_sum_f32() {
    assert_eq!(sum::<f32>(&[]), 0.0);
    let data = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
    assert_eq!(sum::<f32>(&data), 55.0);
}

#[test]
fn test_dot_f32() {
    let a = [1.0f32, 2.0, 3.0, 4.0, 5.0];
    let b = [2.0f32, 3.0, 4.0, 5.0, 6.0];
    assert_eq!(dot::<f32>(&a, &b).unwrap(), 70.0);
}

#[test]
fn test_dot_f32_non_dyadic_tail_matches_within_tolerance() {
    let a: Vec<f32> = (0..13)
        .map(|i| ((i * 19 + 7) as f32) / 23.0 - 2.0)
        .collect();
    let b: Vec<f32> = (0..13)
        .map(|i| ((i * 11 + 5) as f32) / 17.0 - 1.5)
        .collect();

    let actual = dot::<f32>(&a, &b).unwrap();
    let expected = a.iter().zip(&b).map(|(&lhs, &rhs)| lhs * rhs).sum::<f32>();

    assert!(
        (actual - expected).abs() <= 4.0e-6 * expected.abs().max(1.0),
        "dot tail mismatch: got {actual}, expected {expected}"
    );
}

#[test]
fn test_elementwise_mul_f32() {
    let a = [1.0f32, 2.0, 3.0, 4.0, 5.0];
    let b = [2.0f32, 3.0, 4.0, 5.0, 6.0];
    let mut out = [0.0f32; 5];
    elementwise_mul::<f32>(&a, &b, &mut out).unwrap();
    assert_eq!(out, [2.0, 6.0, 12.0, 20.0, 30.0]);
}

#[test]
fn test_axpy_mul_public_facade_f32_and_f64() {
    let a32 = [1.0f32, -2.0, 3.0, -4.0, 5.0];
    let b32 = [2.0f32, 3.0, -4.0, 5.0, -6.0];
    let mut out32 = [10.0f32; 5];
    axpy_mul(0.5, &a32, &b32, &mut out32).unwrap();
    for i in 0..a32.len() {
        assert_eq!(
            out32[i].to_bits(),
            (0.5f32 * a32[i]).mul_add(b32[i], 10.0).to_bits()
        );
    }

    let a64 = [1.0f64, -2.0, 3.0, -4.0, 5.0];
    let b64 = [2.0f64, 3.0, -4.0, 5.0, -6.0];
    let mut out64 = [10.0f64; 5];
    axpy_mul(0.5, &a64, &b64, &mut out64).unwrap();
    for i in 0..a64.len() {
        assert_eq!(
            out64[i].to_bits(),
            (0.5f64 * a64[i]).mul_add(b64[i], 10.0).to_bits()
        );
    }
}

#[test]
fn test_axpy_mul_public_facade_rejects_mismatched_lengths() {
    let mut out = [0.0f32; 2];
    assert_eq!(
        axpy_mul(1.0, &[1.0, 2.0], &[3.0], &mut out),
        Err(SimdError::LengthMismatch)
    );
    assert_eq!(
        axpy_mul(1.0, &[1.0], &[3.0], &mut out),
        Err(SimdError::LengthMismatch)
    );
}

/// Pins the documented intermediate-rounding contract for the f32 provider:
/// `(alpha * a) * b + out`, rather than `alpha * (a * b) + out`.
#[test]
fn test_axpy_mul_f32_pins_intermediate_rounding() {
    let alpha = -21_624.873f32;
    let a = 66_191_484.0f32;
    let b = 4.032_053_5f32;
    let initial = 29_956_176.0f32;
    // 32 elements span the vector path on every supported backend, including
    // AVX-512's 16-lane f32 vectors, while still avoiding a large fixture.
    let a_values = [a; 32];
    let b_values = [b; 32];
    let mut out = [initial; 32];

    let scaled_a = alpha * a;
    let expected = scaled_a.mul_add(b, initial);
    let alternate = alpha.mul_add(a * b, initial);
    assert_ne!(expected.to_bits(), alternate.to_bits());
    assert_eq!(expected.to_bits(), 0xd4a7_f822);

    axpy_mul(alpha, &a_values, &b_values, &mut out).unwrap();
    assert!(out
        .iter()
        .all(|value| value.to_bits() == expected.to_bits()));
}

#[test]
fn test_elementwise_add_sub_div_f32() {
    let a = [1.0f32, 2.0, 3.0, 4.0, 5.0];
    let b = [2.0f32, 3.0, 4.0, 5.0, 6.0];
    let mut out = [0.0f32; 5];
    elementwise_add::<f32>(&a, &b, &mut out).unwrap();
    assert_eq!(out, [3.0, 5.0, 7.0, 9.0, 11.0]);
    elementwise_sub::<f32>(&a, &b, &mut out).unwrap();
    assert_eq!(out, [-1.0, -1.0, -1.0, -1.0, -1.0]);
    elementwise_div::<f32>(&b, &a, &mut out).unwrap();
    assert_eq!(out, [2.0, 1.5, 4.0 / 3.0, 1.25, 1.2]);
}

/// Non-temporal store path differential. An output ≥ 8 MiB engages the
/// cache-bypassing streaming store on AVX2/AVX-512 f32; a mis-aligned output
/// slice (offset by one element) forces the alignment-peel head. Streaming
/// changes only the store instruction, not the arithmetic, so the result must
/// be byte-identical to a sequential scalar reference (add is exact for these
/// dyadic inputs). On backends without a non-temporal store the regular path
/// runs and the same assertion holds.
#[test]
fn test_elementwise_add_streaming_matches_scalar() {
    // 2_100_003 f32 = 8.4 MiB > the 8 MiB NT threshold; not a lane multiple so
    // the tail path is exercised too.
    let n = 2_100_003usize;
    let a: Vec<f32> = (0..n).map(|i| (i % 101) as f32 * 0.5).collect();
    let b: Vec<f32> = (0..n).map(|i| (i % 103) as f32 * 0.25).collect();
    let mut out_buf = vec![0.0f32; n + 8];
    let out = &mut out_buf[1..=n]; // 4-byte-offset start → nonzero peel head
    elementwise_add::<f32>(&a, &b, out).unwrap();

    for i in 0..n {
        assert_eq!(
            out[i].to_bits(),
            (a[i] + b[i]).to_bits(),
            "streaming add mismatch at {i}"
        );
    }
}

/// Differential check across sizes spanning the SIMD lane/tail boundary:
/// vectorized add/sub/mul/div must match a plain scalar reference bit-for-bit
/// (each is a single per-lane IEEE op — no reassociation).
#[test]
fn test_elementwise_binary_matches_scalar_reference() {
    for &n in &[1usize, 3, 7, 8, 15, 16, 17, 64, 257, 1024] {
        let a32: Vec<f32> = (0..n).map(|i| i as f32 * 0.5 - 3.0).collect();
        let b32: Vec<f32> = (0..n).map(|i| (i % 7) as f32 + 1.0).collect();
        let a64: Vec<f64> = (0..n).map(|i| i as f64 * 0.5 - 3.0).collect();
        let b64: Vec<f64> = (0..n).map(|i| (i % 7) as f64 + 1.0).collect();

        let mut o32 = vec![0.0f32; n];
        let mut o64 = vec![0.0f64; n];

        elementwise_add::<f32>(&a32, &b32, &mut o32).unwrap();
        for i in 0..n {
            assert_eq!(
                o32[i].to_bits(),
                (a32[i] + b32[i]).to_bits(),
                "add f32 n={n} i={i}"
            );
        }
        elementwise_sub::<f32>(&a32, &b32, &mut o32).unwrap();
        for i in 0..n {
            assert_eq!(
                o32[i].to_bits(),
                (a32[i] - b32[i]).to_bits(),
                "sub f32 n={n} i={i}"
            );
        }
        elementwise_mul::<f32>(&a32, &b32, &mut o32).unwrap();
        for i in 0..n {
            assert_eq!(
                o32[i].to_bits(),
                (a32[i] * b32[i]).to_bits(),
                "mul f32 n={n} i={i}"
            );
        }
        elementwise_div::<f32>(&a32, &b32, &mut o32).unwrap();
        for i in 0..n {
            assert_eq!(
                o32[i].to_bits(),
                (a32[i] / b32[i]).to_bits(),
                "div f32 n={n} i={i}"
            );
        }

        elementwise_add::<f64>(&a64, &b64, &mut o64).unwrap();
        for i in 0..n {
            assert_eq!(
                o64[i].to_bits(),
                (a64[i] + b64[i]).to_bits(),
                "add f64 n={n} i={i}"
            );
        }
        elementwise_sub::<f64>(&a64, &b64, &mut o64).unwrap();
        for i in 0..n {
            assert_eq!(
                o64[i].to_bits(),
                (a64[i] - b64[i]).to_bits(),
                "sub f64 n={n} i={i}"
            );
        }
        elementwise_mul::<f64>(&a64, &b64, &mut o64).unwrap();
        for i in 0..n {
            assert_eq!(
                o64[i].to_bits(),
                (a64[i] * b64[i]).to_bits(),
                "mul f64 n={n} i={i}"
            );
        }
        elementwise_div::<f64>(&a64, &b64, &mut o64).unwrap();
        for i in 0..n {
            assert_eq!(
                o64[i].to_bits(),
                (a64[i] / b64[i]).to_bits(),
                "div f64 n={n} i={i}"
            );
        }
    }
}

#[test]
fn test_elementwise_binary_length_mismatch() {
    let mut out = [0.0f32; 2];
    assert_eq!(
        elementwise_add::<f32>(&[1.0, 2.0], &[1.0], &mut out),
        Err(SimdError::LengthMismatch)
    );
    assert_eq!(
        elementwise_div::<f32>(&[1.0], &[1.0, 2.0], &mut out),
        Err(SimdError::LengthMismatch)
    );
}

#[test]
fn test_mismatched_lengths() {
    assert_eq!(
        dot::<f32>(&[1.0, 2.0], &[1.0]),
        Err(SimdError::LengthMismatch)
    );
}

#[test]
fn test_masked_sum_all_active() {
    let data = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let mask = [true; 8];
    let result = masked_sum::<f32>(&data, &mask);
    let expected: f32 = data.iter().sum();
    assert!((result - expected).abs() < 1e-5, "{result} != {expected}");
}

#[test]
fn test_masked_sum_alternating() {
    let data = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let mask = [true, false, true, false, true, false, true, false];
    let result = masked_sum::<f32>(&data, &mask);
    // 1 + 3 + 5 + 7 = 16
    assert!((result - 16.0).abs() < 1e-5, "{result}");
}

#[test]
fn test_masked_sum_none_active() {
    let data = [1.0f32; 8];
    let mask = [false; 8];
    assert_eq!(masked_sum::<f32>(&data, &mask), 0.0);
}

#[test]
fn test_masked_sum_odd_length() {
    // 5 elements = 1 full scalar-only chunk for 4-lane Scalar
    let data = [1.0f32, 2.0, 3.0, 4.0, 5.0];
    let mask = [true, false, true, false, true];
    let result = masked_sum::<f32>(&data, &mask);
    assert!((result - 9.0).abs() < 1e-5, "{result}");
}

#[test]
fn test_masked_dot_f32_correctness() {
    let a = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let b = [2.0f32, 2.0, 2.0, 2.0, 2.0, 2.0];
    let mask = [true, false, true, false, true, false];
    let result = masked_dot::<f32>(&a, &b, &mask).unwrap();
    // (1*2) + (3*2) + (5*2) = 2 + 6 + 10 = 18
    assert!((result - 18.0).abs() < 1e-5, "{result}");
}

#[test]
fn test_masked_dot_length_mismatch() {
    assert_eq!(
        masked_dot::<f32>(&[1.0, 2.0], &[1.0], &[true, false]),
        Err(SimdError::LengthMismatch)
    );
}

#[test]
fn test_masked_add_f32() {
    let a = [1.0f32, 2.0, 3.0, 4.0];
    let b = [10.0f32, 10.0, 10.0, 10.0];
    let mask = [true, false, true, false];
    let mut out = [0.0f32; 4];
    masked_add::<f32>(&a, &b, &mask, &mut out).unwrap();
    // Active: 1+10=11, 3+10=13. Inactive: keep a[i]: 2, 4
    assert_eq!(out, [11.0, 2.0, 13.0, 4.0]);
}

#[test]
fn test_generic_masked_ops_f32() {
    let a = [1.0f32, 2.0, 3.0, 4.0];
    let b = [10.0f32, 10.0, 10.0, 10.0];
    let mask = [true, false, true, false];
    let mut out = [0.0f32; 4];

    assert!((masked_sum(&a, &mask) - 4.0).abs() < 1e-5);
    assert!((masked_dot(&a, &b, &mask).unwrap() - 40.0).abs() < 1e-5);
    masked_add(&a, &b, &mask, &mut out).unwrap();
    assert_eq!(out, [11.0, 2.0, 13.0, 4.0]);
}

#[test]
fn test_generic_masked_ops_f64() {
    let a = [1.0f64, 2.0, 3.0, 4.0];
    let b = [10.0f64, 10.0, 10.0, 10.0];
    let mask = [true, false, true, false];
    let mut out = [0.0f64; 4];

    assert!((masked_sum(&a, &mask) - 4.0).abs() < 1e-12);
    assert!((masked_dot(&a, &b, &mask).unwrap() - 40.0).abs() < 1e-12);
    masked_add(&a, &b, &mask, &mut out).unwrap();
    assert_eq!(out, [11.0, 2.0, 13.0, 4.0]);
}

#[test]
fn test_f16_basic_ops() {
    use eunomia::F16;
    let data = vec![F16::from_f32(1.0); 16];
    let s = sum(&data);
    assert_eq!(s, F16::from_f32(16.0));

    let a = vec![F16::from_f32(1.0), F16::from_f32(2.0), F16::from_f32(3.0)];
    let b = vec![F16::from_f32(4.0), F16::from_f32(5.0), F16::from_f32(6.0)];
    let d = dot(&a, &b).unwrap();
    assert_eq!(d, F16::from_f32(32.0));

    let mut mul_out = vec![F16::ZERO; 3];
    elementwise_mul(&a, &b, &mut mul_out).unwrap();
    assert_eq!(
        mul_out,
        vec![F16::from_f32(4.0), F16::from_f32(10.0), F16::from_f32(18.0)]
    );

    let mask = [true, false, true];
    assert_eq!(masked_sum(&a, &mask), F16::from_f32(4.0));
    assert_eq!(masked_dot(&a, &b, &mask).unwrap(), F16::from_f32(22.0));

    let mut add_out = vec![F16::ZERO; 3];
    masked_add(&a, &b, &mask, &mut add_out).unwrap();
    assert_eq!(
        add_out,
        vec![F16::from_f32(5.0), F16::from_f32(2.0), F16::from_f32(9.0)]
    );
}

#[test]
fn test_dispatch_view_selection() {
    let data = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let mut data_mut = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let data_f16 = vec![eunomia::F16::from_f32(1.0); 8];

    // The dispatched view must expose the exact input storage it wrapped,
    // whichever backend the host selected: read it back through the view.
    let view = dispatch_view::<f32, Unaligned>(&data);
    match &view {
        Some(DispatchedView::Scalar(view)) => assert_eq!(view.as_slice(), &data),
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        Some(DispatchedView::Avx2(view)) => assert_eq!(view.as_slice(), &data),
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        Some(DispatchedView::Avx512(view)) => assert_eq!(view.as_slice(), &data),
        #[cfg(target_arch = "aarch64")]
        Some(DispatchedView::Neon(view)) => assert_eq!(view.as_slice(), &data),
        Some(DispatchedView::Sve(view)) => assert_eq!(view.as_slice(), &data),
        _ => panic!("dispatch_view must select a backend for any host"),
    }

    // The mutable dispatched view must preserve exclusive access to the same
    // storage: write through it, then read the slice back and compare.
    let mut expected = data;
    expected[2] = 9.0;
    let mut view_mut = dispatch_view_mut::<f32, Unaligned>(&mut data_mut);
    match &mut view_mut {
        Some(DispatchedView::Scalar(view)) => view.as_slice_mut()[2] = 9.0,
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        Some(DispatchedView::Avx2(view)) => view.as_slice_mut()[2] = 9.0,
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        Some(DispatchedView::Avx512(view)) => view.as_slice_mut()[2] = 9.0,
        #[cfg(target_arch = "aarch64")]
        Some(DispatchedView::Neon(view)) => view.as_slice_mut()[2] = 9.0,
        Some(DispatchedView::Sve(view)) => view.as_slice_mut()[2] = 9.0,
        _ => panic!("dispatch_view_mut must select a backend for any host"),
    }
    assert_eq!(data_mut, expected);

    let view_f16 = dispatch_view::<eunomia::F16, Unaligned>(&data_f16);
    match &view_f16 {
        Some(DispatchedView::Scalar(view)) => assert_eq!(view.as_slice(), &data_f16),
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        Some(DispatchedView::Avx2(view)) => assert_eq!(view.as_slice(), &data_f16),
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        Some(DispatchedView::Avx512(view)) => assert_eq!(view.as_slice(), &data_f16),
        #[cfg(target_arch = "aarch64")]
        Some(DispatchedView::Neon(view)) => assert_eq!(view.as_slice(), &data_f16),
        Some(DispatchedView::Sve(view)) => assert_eq!(view.as_slice(), &data_f16),
        _ => panic!("dispatch_view must select a backend for any host"),
    }
}

#[test]
fn test_monomorphized_vector_ops() {
    // -------------------------------------------------------------------------
    // Scalar f32 tests
    // -------------------------------------------------------------------------
    {
        let a = Vector::<f32, Scalar>::splat(2.0f32);
        let b = Vector::<f32, Scalar>::splat(3.0f32);

        let c = a + b;
        let d = a * b;
        let e = a - b;

        let mut buf = [0.0f32; 4];
        unsafe {
            c.store_unaligned(buf.as_mut_ptr());
        }
        assert_eq!(buf, [5.0f32; 4]);

        unsafe {
            d.store_unaligned(buf.as_mut_ptr());
        }
        assert_eq!(buf, [6.0f32; 4]);

        unsafe {
            e.store_unaligned(buf.as_mut_ptr());
        }
        assert_eq!(buf, [-1.0f32; 4]);

        let mut a_mut = a;
        a_mut += b;
        assert_eq!(a_mut, c);

        a_mut *= b;
        unsafe {
            a_mut.store_unaligned(buf.as_mut_ptr());
        }
        assert_eq!(buf, [15.0f32; 4]);

        assert_eq!(c.sum_reduce(), 20.0f32);

        let f = Vector::<f32, Scalar>::splat(6.0f32);
        let g = Vector::<f32, Scalar>::splat(2.0f32);
        let h = f / g;
        unsafe {
            h.store_unaligned(buf.as_mut_ptr());
        }
        assert_eq!(buf, [3.0f32; 4]);
    }

    // -------------------------------------------------------------------------
    // Scalar f64 tests
    // -------------------------------------------------------------------------
    {
        let a = Vector::<f64, Scalar>::splat(2.0f64);
        let b = Vector::<f64, Scalar>::splat(3.0f64);

        let c = a + b;
        let d = a * b;
        let e = a - b;

        let mut buf = [0.0f64; 2];
        unsafe {
            c.store_unaligned(buf.as_mut_ptr());
        }
        assert_eq!(buf, [5.0f64; 2]);

        unsafe {
            d.store_unaligned(buf.as_mut_ptr());
        }
        assert_eq!(buf, [6.0f64; 2]);

        unsafe {
            e.store_unaligned(buf.as_mut_ptr());
        }
        assert_eq!(buf, [-1.0f64; 2]);

        let mut a_mut = a;
        a_mut += b;
        assert_eq!(a_mut, c);

        a_mut *= b;
        unsafe {
            a_mut.store_unaligned(buf.as_mut_ptr());
        }
        assert_eq!(buf, [15.0f64; 2]);

        assert_eq!(c.sum_reduce(), 10.0f64);

        let f = Vector::<f64, Scalar>::splat(6.0f64);
        let g = Vector::<f64, Scalar>::splat(2.0f64);
        let h = f / g;
        unsafe {
            h.store_unaligned(buf.as_mut_ptr());
        }
        assert_eq!(buf, [3.0f64; 2]);
    }

    // -------------------------------------------------------------------------
    // AVX2 tests (if supported at compile and runtime)
    // -------------------------------------------------------------------------
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if std::is_x86_feature_detected!("avx2") {
            // f32 (8 lanes)
            let a = Vector::<f32, Avx2>::splat(2.0f32);
            let b = Vector::<f32, Avx2>::splat(3.0f32);
            let c = a + b;
            let d = a * b;
            let mut buf_f32 = [0.0f32; 8];
            unsafe {
                c.store_unaligned(buf_f32.as_mut_ptr());
            }
            assert_eq!(buf_f32, [5.0f32; 8]);
            unsafe {
                d.store_unaligned(buf_f32.as_mut_ptr());
            }
            assert_eq!(buf_f32, [6.0f32; 8]);
            assert_eq!(c.sum_reduce(), 40.0f32);

            // f64 (4 lanes)
            let a_f64 = Vector::<f64, Avx2>::splat(2.0f64);
            let b_f64 = Vector::<f64, Avx2>::splat(3.0f64);
            let c_f64 = a_f64 + b_f64;
            let d_f64 = a_f64 * b_f64;
            let mut buf_f64 = [0.0f64; 4];
            unsafe {
                c_f64.store_unaligned(buf_f64.as_mut_ptr());
            }
            assert_eq!(buf_f64, [5.0f64; 4]);
            unsafe {
                d_f64.store_unaligned(buf_f64.as_mut_ptr());
            }
            assert_eq!(buf_f64, [6.0f64; 4]);
            assert_eq!(c_f64.sum_reduce(), 20.0f64);
        }
    }

    // -------------------------------------------------------------------------
    // AVX-512 tests (if supported at compile and runtime)
    // -------------------------------------------------------------------------
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if std::is_x86_feature_detected!("avx512f") {
            // f32 (16 lanes)
            let a = Vector::<f32, Avx512>::splat(2.0f32);
            let b = Vector::<f32, Avx512>::splat(3.0f32);
            let c = a + b;
            let d = a * b;
            let mut buf_f32 = [0.0f32; 16];
            unsafe {
                c.store_unaligned(buf_f32.as_mut_ptr());
            }
            assert_eq!(buf_f32, [5.0f32; 16]);
            unsafe {
                d.store_unaligned(buf_f32.as_mut_ptr());
            }
            assert_eq!(buf_f32, [6.0f32; 16]);
            assert_eq!(c.sum_reduce(), 80.0f32);

            // f64 (8 lanes)
            let a_f64 = Vector::<f64, Avx512>::splat(2.0f64);
            let b_f64 = Vector::<f64, Avx512>::splat(3.0f64);
            let c_f64 = a_f64 + b_f64;
            let d_f64 = a_f64 * b_f64;
            let mut buf_f64 = [0.0f64; 8];
            unsafe {
                c_f64.store_unaligned(buf_f64.as_mut_ptr());
            }
            assert_eq!(buf_f64, [5.0f64; 8]);
            unsafe {
                d_f64.store_unaligned(buf_f64.as_mut_ptr());
            }
            assert_eq!(buf_f64, [6.0f64; 8]);
            assert_eq!(c_f64.sum_reduce(), 40.0f64);
        }
    }

    // -------------------------------------------------------------------------
    // NEON tests (if supported at compile time for AArch64)
    // -------------------------------------------------------------------------
    #[cfg(target_arch = "aarch64")]
    {
        // f32 (4 lanes)
        let a = Vector::<f32, Neon>::splat(2.0f32);
        let b = Vector::<f32, Neon>::splat(3.0f32);
        let c = a + b;
        let d = a * b;
        let mut buf_f32 = [0.0f32; 4];
        unsafe {
            c.store_unaligned(buf_f32.as_mut_ptr());
        }
        assert_eq!(buf_f32, [5.0f32; 4]);
        unsafe {
            d.store_unaligned(buf_f32.as_mut_ptr());
        }
        assert_eq!(buf_f32, [6.0f32; 4]);
        assert_eq!(c.sum_reduce(), 20.0f32);

        // f64 (2 lanes)
        let a_f64 = Vector::<f64, Neon>::splat(2.0f64);
        let b_f64 = Vector::<f64, Neon>::splat(3.0f64);
        let c_f64 = a_f64 + b_f64;
        let d_f64 = a_f64 * b_f64;
        let mut buf_f64 = [0.0f64; 2];
        unsafe {
            c_f64.store_unaligned(buf_f64.as_mut_ptr());
        }
        assert_eq!(buf_f64, [5.0f64; 2]);
        unsafe {
            d_f64.store_unaligned(buf_f64.as_mut_ptr());
        }
        assert_eq!(buf_f64, [6.0f64; 2]);
        assert_eq!(c_f64.sum_reduce(), 10.0f64);
    }
}

#[test]
fn test_new_emulated_types() {
    // 1. Eunomia bfloat16
    {
        use eunomia::Bf16;
        let data = vec![Bf16::from_f32(1.5); 16];
        assert_eq!(sum(&data), Bf16::from_f32(24.0));

        let a = vec![Bf16::from_f32(1.0), Bf16::from_f32(2.0)];
        let b = vec![Bf16::from_f32(3.0), Bf16::from_f32(4.0)];
        assert_eq!(dot(&a, &b).unwrap(), Bf16::from_f32(11.0));
    }

    // 2. i8
    {
        let data = vec![2i8; 16];
        assert_eq!(sum(&data), 32);

        let a = vec![1i8, 2, 3];
        let b = vec![4i8, 5, 6];
        assert_eq!(dot(&a, &b).unwrap(), 32);
    }

    // 3. i16
    {
        let data = vec![3i16; 16];
        assert_eq!(sum(&data), 48);

        let a = vec![1i16, 2, 3];
        let b = vec![4i16, 5, 6];
        assert_eq!(dot(&a, &b).unwrap(), 32);
    }

    // 4. i32
    {
        let data = vec![4i32; 16];
        assert_eq!(sum(&data), 64);

        let a = vec![1i32, 2, 3];
        let b = vec![4i32, 5, 6];
        assert_eq!(dot(&a, &b).unwrap(), 32);
    }
}
