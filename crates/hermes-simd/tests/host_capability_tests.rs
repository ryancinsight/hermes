#![expect(
    clippy::float_cmp,
    reason = "The host capability contract compares exact manufactured lane values"
)]

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
fn target_id_support_matches_host_features() {
    assert!(TargetId::Scalar.is_supported());
    assert_eq!(TargetId::Scalar.name(), "scalar");
    assert_eq!(TargetId::Avx2.name(), "avx2");
    assert_eq!(TargetId::Avx512.name(), "avx512");
    assert_eq!(TargetId::Neon.name(), "neon");
    assert_eq!(TargetId::Sve.name(), "sve");

    // The emulated SVE backend executes on every host, like the scalar path.
    assert!(TargetId::Sve.is_supported());
    assert!(TargetId::Sve.is_architecture_applicable());

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        assert_eq!(
            TargetId::Avx2.is_supported(),
            std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma")
        );
        assert_eq!(
            TargetId::Avx512.is_supported(),
            std::is_x86_feature_detected!("avx512f")
        );
        assert!(!TargetId::Neon.is_supported());
    }

    #[cfg(target_arch = "aarch64")]
    {
        assert!(!TargetId::Avx2.is_supported());
        assert!(!TargetId::Avx512.is_supported());
        assert!(TargetId::Neon.is_supported());
    }

    #[cfg(not(any(target_arch = "x86", target_arch = "x86_64", target_arch = "aarch64")))]
    {
        assert!(!TargetId::Avx2.is_supported());
        assert!(!TargetId::Avx512.is_supported());
        assert!(!TargetId::Neon.is_supported());
    }
}

#[test]
fn fma_support_matches_runtime_feature_detector() {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        let expected = std::is_x86_feature_detected!("fma");
        assert_eq!(has_fma3(), expected);
        assert_eq!(<f32 as FmaSupport>::has_fma(), expected);
        assert_eq!(<f64 as FmaSupport>::has_fma(), expected);
        assert_eq!(<eunomia::Bf16 as FmaSupport>::has_fma(), expected);
    }

    #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
    {
        assert!(!has_fma3());
        assert!(!<f32 as FmaSupport>::has_fma());
        assert!(!<f64 as FmaSupport>::has_fma());
        assert!(!<eunomia::Bf16 as FmaSupport>::has_fma());
    }
}

#[test]
fn forced_scalar_dispatch_view_preserves_slice_values() {
    let data = [1.0f32, -2.0, 3.5, 4.0];
    let view = dispatch_view_to::<f32, Unaligned>(TargetId::Scalar, &data)
        .expect("scalar target is always supported");

    match view {
        DispatchedView::Scalar(view) => {
            assert_eq!(view.as_slice(), &data);
            assert_eq!(view.sum().to_bits(), 6.5f32.to_bits());
        }
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        DispatchedView::Avx2(_) | DispatchedView::Avx512(_) => {
            panic!("forced scalar target constructed an x86 SIMD view")
        }
        #[cfg(target_arch = "aarch64")]
        DispatchedView::Neon(_) => panic!("forced scalar target constructed a NEON view"),
        DispatchedView::Sve(_) => panic!("forced scalar target constructed an SVE view"),
        _ => unreachable!("unknown future backend"),
    }
}

#[test]
fn forced_mut_scalar_dispatch_view_preserves_exclusive_slice() {
    let mut data = [1.0f32, 2.0, 3.0, 4.0];
    let mut view = dispatch_view_mut_to::<f32, Unaligned>(TargetId::Scalar, &mut data)
        .expect("scalar target is always supported");

    match &mut view {
        DispatchedView::Scalar(view) => {
            let slice = view.as_slice_mut();
            slice[2] = 9.0;
            assert_eq!(slice, &[1.0, 2.0, 9.0, 4.0]);
        }
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        DispatchedView::Avx2(_) | DispatchedView::Avx512(_) => {
            panic!("forced scalar target constructed an x86 SIMD view")
        }
        #[cfg(target_arch = "aarch64")]
        DispatchedView::Neon(_) => panic!("forced scalar target constructed a NEON view"),
        DispatchedView::Sve(_) => panic!("forced scalar target constructed an SVE view"),
        _ => unreachable!("unknown future backend"),
    }
    assert_eq!(data, [1.0, 2.0, 9.0, 4.0]);
}

#[test]
fn forced_dispatch_rejects_unsupported_target_before_view_construction() {
    let data = [1.0f32; 8];

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    let unsupported = TargetId::Neon;
    #[cfg(target_arch = "aarch64")]
    let unsupported = TargetId::Avx2;
    #[cfg(not(any(target_arch = "x86", target_arch = "x86_64", target_arch = "aarch64")))]
    let unsupported = TargetId::Avx2;

    assert!(!unsupported.is_supported());
    assert!(dispatch_view_to::<f32, Unaligned>(unsupported, &data).is_none());
}

#[test]
fn forced_dispatch_returns_view_for_each_supported_target() {
    let data = [1.0f32; 64];

    for target in [
        TargetId::Scalar,
        TargetId::Avx2,
        TargetId::Avx512,
        TargetId::Neon,
        TargetId::Sve,
    ] {
        let view = dispatch_view_to::<f32, Unaligned>(target, &data);
        assert_eq!(view.is_some(), target.is_supported(), "{target:?}");
    }
}

#[test]
fn forced_dense_facade_matches_scalar_for_every_supported_target() {
    let a: Vec<f32> = (0..96).map(|i| (i % 9) as f32 - 4.0).collect();
    let b: Vec<f32> = (0..96).map(|i| (i % 7) as f32 - 3.0).collect();
    let indices = [95, 0, 17, 31, 64, 7, 88, 45, 3, 72, 11, 59];
    let mask: Vec<bool> = (0..a.len()).map(|i| i % 3 == 0 || i % 5 == 0).collect();

    let expected = dense_scalar_reference(&a, &b, &indices, &mask);

    for target in [
        TargetId::Scalar,
        TargetId::Avx2,
        TargetId::Avx512,
        TargetId::Neon,
        TargetId::Sve,
    ] {
        if target.is_supported() {
            assert_forced_dense_target_matches(target, &a, &b, &indices, &mask, &expected);
        }
    }
}

struct DenseReference {
    sum_bits: u32,
    dot_bits: u32,
    mul: Vec<f32>,
    add: Vec<f32>,
    sub: Vec<f32>,
    gather: Vec<f32>,
    select: Vec<f32>,
}

fn dense_scalar_reference(a: &[f32], b: &[f32], indices: &[i32], mask: &[bool]) -> DenseReference {
    let scalar_a = SimdView::<f32, Scalar, Unaligned>::new(a).unwrap();
    let scalar_b = SimdView::<f32, Scalar, Unaligned>::new(b).unwrap();

    let mut mul = vec![0.0; a.len()];
    scalar_a.elementwise_mul(&scalar_b, &mut mul).unwrap();

    let mut add = vec![0.0; a.len()];
    scalar_a.zip_into(&scalar_b, &mut add, Add).unwrap();

    let mut sub = vec![0.0; a.len()];
    scalar_a.zip_into(&scalar_b, &mut sub, Sub).unwrap();

    let mut gather = vec![0.0; indices.len()];
    scalar_a.gather(indices, &mut gather).unwrap();

    let select = scalar_a.select(mask, &scalar_b).unwrap().to_vec();

    DenseReference {
        sum_bits: scalar_a.sum().to_bits(),
        dot_bits: scalar_a.dot(&scalar_b).unwrap().to_bits(),
        mul,
        add,
        sub,
        gather,
        select,
    }
}

fn assert_forced_dense_target_matches(
    target: TargetId,
    a: &[f32],
    b: &[f32],
    indices: &[i32],
    mask: &[bool],
    expected: &DenseReference,
) {
    let view_a =
        dispatch_view_to::<f32, Unaligned>(target, a).expect("supported target constructs a view");
    let view_b =
        dispatch_view_to::<f32, Unaligned>(target, b).expect("supported target constructs a view");

    match (view_a, view_b) {
        (DispatchedView::Scalar(a_view), DispatchedView::Scalar(b_view)) => {
            assert_dense_view_matches_target(target, &a_view, &b_view, indices, mask, expected);
        }
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        (DispatchedView::Avx2(a_view), DispatchedView::Avx2(b_view)) => {
            assert_dense_view_matches_target(target, &a_view, &b_view, indices, mask, expected);
        }
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        (DispatchedView::Avx512(a_view), DispatchedView::Avx512(b_view)) => {
            assert_dense_view_matches_target(target, &a_view, &b_view, indices, mask, expected);
        }
        #[cfg(target_arch = "aarch64")]
        (DispatchedView::Neon(a_view), DispatchedView::Neon(b_view)) => {
            assert_dense_view_matches_target(target, &a_view, &b_view, indices, mask, expected);
        }
        (DispatchedView::Sve(a_view), DispatchedView::Sve(b_view)) => {
            assert_dense_view_matches_target(target, &a_view, &b_view, indices, mask, expected);
        }
        _ => panic!("target {target:?} constructed mismatched view variants"),
    }
}

fn assert_dense_view_matches_target<Arch>(
    target: TargetId,
    a: &SimdView<'_, f32, Arch, Unaligned>,
    b: &SimdView<'_, f32, Arch, Unaligned>,
    indices: &[i32],
    mask: &[bool],
    expected: &DenseReference,
) where
    Arch: SimdArch + SimdKernel<f32>,
{
    assert_eq!(a.sum().to_bits(), expected.sum_bits, "{target:?} sum");
    assert_eq!(
        a.dot(b).unwrap().to_bits(),
        expected.dot_bits,
        "{target:?} dot"
    );

    let mut mul = vec![0.0; a.len()];
    a.elementwise_mul(b, &mut mul).unwrap();
    assert_eq!(mul, expected.mul, "{target:?} elementwise_mul");

    let mut add = vec![0.0; a.len()];
    a.zip_into(b, &mut add, Add).unwrap();
    assert_eq!(add, expected.add, "{target:?} add");

    let mut sub = vec![0.0; a.len()];
    a.zip_into(b, &mut sub, Sub).unwrap();
    assert_eq!(sub, expected.sub, "{target:?} sub");

    let mut gather = vec![0.0; indices.len()];
    a.gather(indices, &mut gather).unwrap();
    assert_eq!(gather, expected.gather, "{target:?} gather");

    let select = a.select(mask, b).unwrap().to_vec();
    assert_eq!(select, expected.select, "{target:?} select");
}

#[test]
fn runtime_dispatch_view_matches_host_features() {
    let data = [1.0f32; 64];
    let view = dispatch_view::<f32, Unaligned>(&data).expect("dispatch_view returns a backend");

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    match (expected_x86_dispatch(), view) {
        (HostDispatch::Avx512, DispatchedView::Avx512(_))
        | (HostDispatch::Avx2, DispatchedView::Avx2(_))
        | (HostDispatch::Scalar, DispatchedView::Scalar(_)) => {}
        (expected, actual) => panic!("expected {expected:?}, got {}", dispatch_name(&actual)),
    }

    #[cfg(target_arch = "aarch64")]
    match view {
        DispatchedView::Neon(_) => {}
        DispatchedView::Scalar(_) => panic!("aarch64 host should dispatch to NEON"),
        _ => unreachable!("aarch64 host should never dispatch to a non-NEON backend"),
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
        _ => unreachable!("dispatch_view returns only x86 or scalar backends on x86"),
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
                sum += i32::from(a[r * k + kk]) * i32::from(b[kk * n + col]);
            }
            expected[r * n + col] += sum;
        }
    }

    assert_eq!(c, expected);
}

/// A `SimdView` may exist only for an architecture the host can execute.
///
/// Every view operation calls `#[target_feature]`-gated kernels, so a view over
/// an unsupported marker would let entirely safe code issue instructions the CPU
/// does not implement. Before this was enforced, constructing an `Avx512` view
/// on a host without AVX-512 succeeded and the first reduction died with an
/// illegal instruction. The oracle here is the platform feature probe, which is
/// independent of the `SimdArch::is_runtime_supported` implementation under test.
#[test]
fn view_exists_only_for_executable_arch() {
    let data = [1.0_f32; 64];

    assert!(
        SimdView::<f32, Scalar, Unaligned, Unmasked, &[f32]>::new(&data).is_some(),
        "the emulated backend runs everywhere"
    );

    // The emulated SVE backend is unconditional, exactly like the scalar path.
    assert!(
        SimdView::<f32, SveArch, Unaligned, Unmasked, &[f32]>::new(&data).is_some(),
        "the emulated SVE backend runs everywhere"
    );

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        let host_avx2 =
            std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma");
        assert_eq!(
            SimdView::<f32, Avx2, Unaligned, Unmasked, &[f32]>::new(&data).is_some(),
            host_avx2,
            "AVX2 view availability must track the host probe"
        );

        let host_avx512 = std::is_x86_feature_detected!("avx512f");
        assert_eq!(
            SimdView::<f32, Avx512, Unaligned, Unmasked, &[f32]>::new(&data).is_some(),
            host_avx512,
            "AVX-512 view availability must track the host probe"
        );
    }

    #[cfg(target_arch = "aarch64")]
    assert!(
        SimdView::<f32, Neon, Unaligned, Unmasked, &[f32]>::new(&data).is_some(),
        "NEON is baseline on aarch64"
    );
}

/// The same guard must hold for mutable views, which reach the same kernels.
#[test]
fn mutable_view_exists_only_for_executable_arch() {
    let mut data = [1.0_f32; 64];

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        let host_avx512 = std::is_x86_feature_detected!("avx512f");
        assert_eq!(
            SimdView::<f32, Avx512, Unaligned, Unmasked, &mut [f32]>::new_mut(&mut data).is_some(),
            host_avx512
        );
    }

    assert!(SimdView::<f32, Scalar, Unaligned, Unmasked, &mut [f32]>::new_mut(&mut data).is_some());
}
