use hermes_simd::*;

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum HostDispatch {
    Avx512,
    Avx2,
    Scalar,
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
fn expected_x86_dispatch() -> HostDispatch {
    if std::is_x86_feature_detected!("avx512f") {
        HostDispatch::Avx512
    } else if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
        HostDispatch::Avx2
    } else {
        HostDispatch::Scalar
    }
}

#[test]
fn runtime_dispatch_view_matches_host_features() {
    let data = [1.0f32; 64];
    let view = dispatch_view::<f32, Unaligned>(&data).expect("dispatch_view returns a backend");

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    match (expected_x86_dispatch(), view) {
        (HostDispatch::Avx512, DispatchedView::Avx512(_)) => {}
        (HostDispatch::Avx2, DispatchedView::Avx2(_)) => {}
        (HostDispatch::Scalar, DispatchedView::Scalar(_)) => {}
        (expected, actual) => panic!("expected {expected:?}, got {}", dispatch_name(&actual)),
    }

    #[cfg(target_arch = "aarch64")]
    match view {
        DispatchedView::Neon(_) => {}
        DispatchedView::Scalar(_) => panic!("aarch64 host should dispatch to NEON"),
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
fn dispatch_name<T>(view: &DispatchedView<'_, T>) -> &'static str
where
    T: FloatElement,
{
    match view {
        DispatchedView::Avx512(_) => "avx512",
        DispatchedView::Avx2(_) => "avx2",
        DispatchedView::Scalar(_) => "scalar",
    }
}

#[test]
fn avx2_backend_matches_runtime_dispatch_on_this_host() {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if !(std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma")) {
            return;
        }

        let a: Vec<f32> = (0..257).map(|i| (i % 17) as f32 - 8.0).collect();
        let b: Vec<f32> = (0..257).map(|i| (i % 11) as f32 * 0.5 - 2.0).collect();
        let view_a = SimdView::<f32, Avx2, Unaligned>::new(&a).unwrap();
        let view_b = SimdView::<f32, Avx2, Unaligned>::new(&b).unwrap();

        let avx_sum = view_a.sum();
        let runtime_sum = sum::<f32>(&a);
        assert_eq!(avx_sum.to_bits(), runtime_sum.to_bits());

        let avx_dot = view_a.dot(&view_b).unwrap();
        let runtime_dot = dot::<f32>(&a, &b).unwrap();
        assert_eq!(avx_dot.to_bits(), runtime_dot.to_bits());
    }
}

#[test]
fn local_gemm_dispatch_matches_scalar_reference_for_irregular_shapes() {
    let m = 19usize;
    let n = 17usize;
    let k = 65usize;
    let a: Vec<i8> = (0..m * k).map(|i| (i % 7) as i8 - 3).collect();
    let b: Vec<i8> = (0..k * n).map(|i| (i % 5) as i8 - 2).collect();
    let mut c = vec![1i32; m * n];

    unsafe {
        gemm::<i8, i8, i32>(m, n, k, &a, k, &b, n, &mut c, n).unwrap();
    }

    let mut expected = vec![1i32; m * n];
    for r in 0..m {
        for col in 0..n {
            let mut sum = 0i32;
            for kk in 0..k {
                sum += (a[r * k + kk] as i32) * (b[kk * n + col] as i32);
            }
            expected[r * n + col] += sum;
        }
    }

    assert_eq!(c, expected);
}
