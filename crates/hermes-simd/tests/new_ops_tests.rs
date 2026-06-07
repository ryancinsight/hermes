//! Integration tests for the new SimdView operations surface:
//! min/max/argmin/argmax (dispatch), scale, unary map, prefix_scan,
//! select/masked_negate, gather, zip_transform, and ZipChunks.

use hermes_simd::{
    SimdView, Unaligned, Unmasked, Scalar,
    Abs, Neg, Sqrt, Clamp,
    ScanAdd, ScanMul, ScanMin, ScanMax, Inclusive, Exclusive,
    min, max, scale, argmin, argmax,
};
use hermes_simd_core::{
    ops::{Min, Max, Sum},
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

type View<'a, T> = SimdView<'a, T, Scalar, Unaligned, Unmasked, &'a [T]>;
type ViewMut<'a, T> = SimdView<'a, T, Scalar, Unaligned, Unmasked, &'a mut [T]>;

fn v(data: &[f32]) -> View<'_, f32> {
    SimdView::new(data).expect("scalar always ok")
}
fn v_mut(data: &mut [f32]) -> ViewMut<'_, f32> {
    SimdView::new_mut(data).expect("scalar always ok")
}

// ---------------------------------------------------------------------------
// Reduction: min, max via dispatch
// ---------------------------------------------------------------------------

#[test]
fn test_dispatch_min_basic() {
    let data = [3.0f32, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0];
    assert_eq!(min(&data), 1.0);
}

#[test]
fn test_dispatch_max_basic() {
    let data = [3.0f32, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0];
    assert_eq!(max(&data), 9.0);
}

#[test]
fn test_dispatch_min_single() {
    let data = [42.0f32];
    assert_eq!(min(&data), 42.0);
}

#[test]
fn test_dispatch_max_single() {
    let data = [42.0f32];
    assert_eq!(max(&data), 42.0);
}

#[test]
fn test_dispatch_min_empty() {
    // Empty slice returns MAX_VALUE (identity for min)
    let data: [f32; 0] = [];
    assert!(min(&data).is_infinite());
}

#[test]
fn test_dispatch_max_empty() {
    // Empty slice returns MIN_VALUE (identity for max)
    let data: [f32; 0] = [];
    assert!(max(&data).is_infinite());
}

#[test]
fn test_view_reduce_min() {
    let data = [5.0f32, 2.0, 8.0, 1.0, 7.0];
    let got = v(&data).reduce(Min);
    assert_eq!(got, 1.0);
}

#[test]
fn test_view_reduce_max() {
    let data = [5.0f32, 2.0, 8.0, 1.0, 7.0];
    let got = v(&data).reduce(Max);
    assert_eq!(got, 8.0);
}

#[test]
fn test_view_reduce_sum() {
    let data = [1.0f32, 2.0, 3.0, 4.0, 5.0];
    assert_eq!(v(&data).reduce(Sum), 15.0);
}

// ---------------------------------------------------------------------------
// argmin / argmax via dispatch
// ---------------------------------------------------------------------------

#[test]
fn test_dispatch_argmin() {
    let data = [3.0f32, 1.0, 4.0, 1.0, 5.0];
    let (idx, val) = argmin(&data).unwrap();
    assert_eq!(idx, 1); // first occurrence
    assert_eq!(val, 1.0);
}

#[test]
fn test_dispatch_argmax() {
    let data = [3.0f32, 1.0, 9.0, 4.0, 5.0];
    let (idx, val) = argmax(&data).unwrap();
    assert_eq!(idx, 2);
    assert_eq!(val, 9.0);
}

#[test]
fn test_dispatch_argmin_empty() {
    let data: [f32; 0] = [];
    assert!(argmin(&data).is_none());
}

#[test]
fn test_dispatch_argmax_empty() {
    let data: [f32; 0] = [];
    assert!(argmax(&data).is_none());
}

#[test]
fn test_view_argmin() {
    let data = [7.0f32, 3.0, 9.0, 1.0, 5.0];
    let (idx, val) = v(&data).argmin().unwrap();
    assert_eq!(idx, 3);
    assert_eq!(val, 1.0);
}

#[test]
fn test_view_argmax() {
    let data = [7.0f32, 3.0, 9.0, 1.0, 5.0];
    let (idx, val) = v(&data).argmax().unwrap();
    assert_eq!(idx, 2);
    assert_eq!(val, 9.0);
}

// ---------------------------------------------------------------------------
// scale dispatch
// ---------------------------------------------------------------------------

#[test]
fn test_dispatch_scale() {
    let mut data = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
    scale(&mut data, 2.0);
    assert_eq!(&data, &[2.0f32, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0]);
}

#[test]
fn test_dispatch_scale_zero() {
    let mut data = [1.0f32; 16];
    scale(&mut data, 0.0);
    assert!(data.iter().all(|&x| x == 0.0));
}

#[test]
fn test_dispatch_scale_empty() {
    let mut data: [f32; 0] = [];
    scale(&mut data, 2.0); // must not panic
}

// ---------------------------------------------------------------------------
// Unary: map_unary / map_unary_in_place
// ---------------------------------------------------------------------------

#[test]
fn test_map_unary_abs() {
    let data = [-1.0f32, 2.0, -3.0, 4.0, -5.0, 6.0, -7.0, 8.0, -9.0];
    let expected: Vec<f32> = data.iter().map(|x| x.abs()).collect();
    let mut out = vec![0.0f32; data.len()];
    v(&data).map_unary(Abs, &mut out).unwrap();
    assert_eq!(out, expected);
}

#[test]
fn test_map_unary_neg() {
    let data = [1.0f32, -2.0, 3.0, -4.0, 5.0];
    let expected: Vec<f32> = data.iter().map(|x| -x).collect();
    let mut out = vec![0.0f32; data.len()];
    v(&data).map_unary(Neg, &mut out).unwrap();
    assert_eq!(out, expected);
}

#[test]
fn test_map_unary_sqrt() {
    let data = [1.0f32, 4.0, 9.0, 16.0, 25.0];
    let expected = [1.0f32, 2.0, 3.0, 4.0, 5.0];
    let mut out = vec![0.0f32; 5];
    v(&data).map_unary(Sqrt, &mut out).unwrap();
    for (got, exp) in out.iter().zip(expected.iter()) {
        assert!((got - exp).abs() < 1e-5, "got={got}, exp={exp}");
    }
}

#[test]
fn test_map_unary_clamp() {
    let data = [-5.0f32, 0.0, 3.0, 7.0, 10.0, 15.0];
    let clamp = Clamp::new(0.0f32, 10.0);
    let expected: Vec<f32> = data.iter().map(|&x| x.clamp(0.0, 10.0)).collect();
    let mut out = vec![0.0f32; data.len()];
    v(&data).map_unary(clamp, &mut out).unwrap();
    assert_eq!(out, expected);
}

#[test]
fn test_map_unary_in_place_abs() {
    let mut data = [-1.0f32, 2.0, -3.0, 4.0, -5.0];
    let expected: Vec<f32> = data.iter().map(|x| x.abs()).collect();
    v_mut(&mut data).map_unary_in_place(Abs);
    assert_eq!(&data, expected.as_slice());
}

#[test]
fn test_map_unary_in_place_clamp() {
    let mut data = [-5.0f32, 0.0, 3.0, 7.0, 15.0, 20.0, -10.0, 5.0, 8.0];
    let clamp = Clamp::new(0.0f32, 10.0);
    let expected: Vec<f32> = data.iter().map(|&x| x.clamp(0.0, 10.0)).collect();
    v_mut(&mut data).map_unary_in_place(clamp);
    assert_eq!(&data, expected.as_slice());
}

#[test]
fn test_map_unary_output_too_short() {
    let data = [1.0f32; 8];
    let mut out = vec![0.0f32; 4];
    let err = v(&data).map_unary(Abs, &mut out);
    assert!(err.is_err());
}

// ---------------------------------------------------------------------------
// prefix_scan
// ---------------------------------------------------------------------------

#[test]
fn test_prefix_scan_add_inclusive() {
    let data = [1.0f32, 2.0, 3.0, 4.0, 5.0];
    let expected = [1.0f32, 3.0, 6.0, 10.0, 15.0];
    let mut out = vec![0.0f32; 5];
    v(&data).prefix_scan(&mut out, ScanAdd, Inclusive).unwrap();
    assert_eq!(&out, &expected);
}

#[test]
fn test_prefix_scan_add_exclusive() {
    let data = [1.0f32, 2.0, 3.0, 4.0, 5.0];
    let expected = [0.0f32, 1.0, 3.0, 6.0, 10.0];
    let mut out = vec![0.0f32; 5];
    v(&data).prefix_scan(&mut out, ScanAdd, Exclusive).unwrap();
    assert_eq!(&out, &expected);
}

#[test]
fn test_prefix_scan_mul_inclusive() {
    let data = [1.0f32, 2.0, 3.0, 4.0];
    let expected = [1.0f32, 2.0, 6.0, 24.0];
    let mut out = vec![0.0f32; 4];
    v(&data).prefix_scan(&mut out, ScanMul, Inclusive).unwrap();
    assert_eq!(&out, &expected);
}

#[test]
fn test_prefix_scan_mul_exclusive() {
    let data = [1.0f32, 2.0, 3.0, 4.0];
    let expected = [1.0f32, 1.0, 2.0, 6.0]; // exclusive: identity(Mul)=1
    let mut out = vec![0.0f32; 4];
    v(&data).prefix_scan(&mut out, ScanMul, Exclusive).unwrap();
    assert_eq!(&out, &expected);
}

#[test]
fn test_prefix_scan_min_inclusive() {
    let data = [5.0f32, 3.0, 8.0, 1.0, 4.0];
    let expected = [5.0f32, 3.0, 3.0, 1.0, 1.0];
    let mut out = vec![0.0f32; 5];
    v(&data).prefix_scan(&mut out, ScanMin, Inclusive).unwrap();
    assert_eq!(&out, &expected);
}

#[test]
fn test_prefix_scan_max_inclusive() {
    let data = [1.0f32, 5.0, 3.0, 8.0, 2.0];
    let expected = [1.0f32, 5.0, 5.0, 8.0, 8.0];
    let mut out = vec![0.0f32; 5];
    v(&data).prefix_scan(&mut out, ScanMax, Inclusive).unwrap();
    assert_eq!(&out, &expected);
}

#[test]
fn test_prefix_scan_empty() {
    let data: [f32; 0] = [];
    let mut out: [f32; 0] = [];
    v(&data).prefix_scan(&mut out, ScanAdd, Inclusive).unwrap();
}

#[test]
fn test_prefix_scan_output_too_short() {
    let data = [1.0f32; 5];
    let mut out = vec![0.0f32; 3];
    let err = v(&data).prefix_scan(&mut out, ScanAdd, Inclusive);
    assert!(err.is_err());
}

// ---------------------------------------------------------------------------
// select and masked_negate
// ---------------------------------------------------------------------------

#[test]
fn test_select() {
    let a = [1.0f32, 2.0, 3.0, 4.0, 5.0];
    let b = [10.0f32, 20.0, 30.0, 40.0, 50.0];
    let mask = [true, false, true, false, true];

    let a_view = v(&a);
    let b_view = v(&b);
    let result = a_view.select(&mask, &b_view).unwrap();
    assert_eq!(result.as_slice(), &[1.0f32, 20.0, 3.0, 40.0, 5.0]);
}

#[test]
fn test_select_length_mismatch() {
    let a = [1.0f32; 4];
    let b = [10.0f32; 5];
    let mask = [true; 4];
    let err = v(&a).select(&mask, &v(&b));
    assert!(err.is_err());
}

#[test]
fn test_select_mask_too_short() {
    let a = [1.0f32; 5];
    let b = [10.0f32; 5];
    let mask = [true; 3];
    let err = v(&a).select(&mask, &v(&b));
    assert!(err.is_err());
}

#[test]
fn test_masked_negate() {
    let a = [1.0f32, 2.0, 3.0, 4.0, 5.0];
    let mask = [true, false, true, false, true];
    let result = v(&a).masked_negate(&mask).unwrap();
    assert_eq!(result.as_slice(), &[-1.0f32, 2.0, -3.0, 4.0, -5.0]);
}

// ---------------------------------------------------------------------------
// gather
// ---------------------------------------------------------------------------

#[test]
fn test_gather_basic() {
    let data = [10.0f32, 20.0, 30.0, 40.0, 50.0];
    let indices = [4i32, 2, 0, 3, 1];
    let mut out = vec![0.0f32; 5];
    v(&data).gather(&indices, &mut out).unwrap();
    assert_eq!(&out, &[50.0f32, 30.0, 10.0, 40.0, 20.0]);
}

#[test]
fn test_gather_output_too_short() {
    let data = [1.0f32; 8];
    let indices = [0i32, 1, 2, 3, 4, 5, 6, 7];
    let mut out = vec![0.0f32; 4]; // too short
    let err = v(&data).gather(&indices, &mut out);
    assert!(err.is_err());
}

#[test]
fn test_gather_index_out_of_bounds() {
    let data = [1.0f32; 4];
    let indices = [0i32, 1, 2, 10]; // 10 >= 4
    let mut out = vec![0.0f32; 4];
    let err = v(&data).gather(&indices, &mut out);
    assert!(err.is_err());
}

#[test]
fn test_gather_negative_index() {
    let data = [1.0f32; 4];
    let indices = [-1i32, 0, 1, 2];
    let mut out = vec![0.0f32; 4];
    let err = v(&data).gather(&indices, &mut out);
    assert!(err.is_err());
}

// ---------------------------------------------------------------------------
// zip_transform (allocating variant)
// ---------------------------------------------------------------------------

#[test]
fn test_zip_transform_add() {
    use hermes_simd::Add;
    let a = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
    let b = [10.0f32; 7];
    let expected: Vec<f32> = a.iter().zip(b.iter()).map(|(x, y)| x + y).collect();
    let result = v(&a).zip_transform(&v(&b), Add).unwrap();
    assert_eq!(result.as_slice(), expected.as_slice());
}

#[test]
fn test_zip_transform_length_mismatch() {
    use hermes_simd::Mul;
    let a = [1.0f32; 5];
    let b = [1.0f32; 6];
    let err = v(&a).zip_transform(&v(&b), Mul);
    assert!(err.is_err());
}

// ---------------------------------------------------------------------------
// ZipChunks iterator
// ---------------------------------------------------------------------------

#[test]
fn test_zip_chunks_pairwise_sum() {
    use hermes_simd::Add;
    let a = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
    let b = [10.0f32; 9];
    let expected: Vec<f32> = a.iter().zip(b.iter()).map(|(x, y)| x + y).collect();

    // Use zip_transform which internally uses the same generic kernel.
    let result = v(&a).zip_transform(&v(&b), Add).unwrap();
    assert_eq!(result.as_slice(), expected.as_slice());
}

// ---------------------------------------------------------------------------
// ops_mut: add_assign and mul_assign delegate equivalence
// ---------------------------------------------------------------------------

#[test]
fn test_ops_mut_add_assign_delegate() {
    use hermes_simd::Add;
    let a_orig = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
    let b = [10.0f32; 10];

    let mut a_generic = a_orig;
    {
        let b_view = v(&b);
        let mut view = v_mut(&mut a_generic);
        view.transform_in_place(&b_view, Add).unwrap();
    }

    let mut a_delegate = a_orig;
    {
        let b_view = v(&b);
        let mut view = v_mut(&mut a_delegate);
        view.add_assign(&b_view).unwrap();
    }

    assert_eq!(a_generic, a_delegate);
}

#[test]
fn test_ops_mut_mul_assign_delegate() {
    use hermes_simd::Mul;
    let a_orig = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
    let b = [3.0f32; 10];

    let mut a_generic = a_orig;
    {
        let b_view = v(&b);
        let mut view = v_mut(&mut a_generic);
        view.transform_in_place(&b_view, Mul).unwrap();
    }

    let mut a_delegate = a_orig;
    {
        let b_view = v(&b);
        let mut view = v_mut(&mut a_delegate);
        view.mul_assign(&b_view).unwrap();
    }

    assert_eq!(a_generic, a_delegate);
}

// ---------------------------------------------------------------------------
// Odd lengths (scalar tail coverage)
// ---------------------------------------------------------------------------

#[test]
fn test_min_odd_length() {
    let data = [9.0f32, 3.0, 7.0]; // len=3, tail exercises scalar path on wide arches
    assert_eq!(min(&data), 3.0);
}

#[test]
fn test_max_odd_length() {
    let data = [9.0f32, 3.0, 7.0];
    assert_eq!(max(&data), 9.0);
}

#[test]
fn test_argmin_odd_length() {
    let data = [9.0f32, 3.0, 7.0];
    let (idx, val) = argmin(&data).unwrap();
    assert_eq!(idx, 1);
    assert_eq!(val, 3.0);
}

#[test]
fn test_scale_odd_length() {
    let mut data = [1.0f32, 2.0, 3.0]; // odd
    scale(&mut data, 2.0);
    assert_eq!(&data, &[2.0f32, 4.0, 6.0]);
}
