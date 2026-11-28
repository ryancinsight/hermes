use hermes_simd::*;

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
