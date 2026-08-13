//! Integration tests for the new SimdView operations surface:
//! min/max/argmin/argmax (dispatch), scale, unary map, prefix_scan,
//! select/masked_negate, gather, zip_transform, and ZipChunks.

use hermes_simd::{
    argmax, argmin, max, min, scale, Abs, Ceil, Clamp, Exclusive, Floor, Inclusive, Neg, Round,
    Scalar, ScanAdd, ScanMax, ScanMin, ScanMul, SimdError, SimdView, Sqrt, Trunc, Unaligned,
    Unmasked,
};
// `SimdArch` is in scope only to resolve `Avx2::is_runtime_supported()` in the
// x86-gated backend check below; importing it unconditionally is an unused
// import on aarch64, where that block is compiled out.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
use hermes_simd::SimdArch;
use hermes_simd_core::ops::{Max, Min, Sum};
use proptest::prelude::*;

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

fn assert_simd_error<T>(result: Result<T, SimdError>, expected: SimdError) {
    match result {
        Err(actual) => assert_eq!(actual, expected),
        Ok(_) => panic!("expected {expected:?}"),
    }
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
fn dispatch_extrema_reject_nan_independent_of_position() {
    for data in [
        [f32::NAN, -2.0, 4.0],
        [-2.0, f32::NAN, 4.0],
        [-2.0, 4.0, f32::NAN],
    ] {
        assert_eq!(argmin(&data), None);
        assert_eq!(argmax(&data), None);
    }
}

#[test]
fn dispatch_extrema_preserve_first_signed_zero_value() {
    let positive_first = [0.0_f32, -0.0];
    let negative_first = [-0.0_f32, 0.0];

    for (positive_extremum, negative_extremum) in [
        (argmin(&positive_first), argmin(&negative_first)),
        (argmax(&positive_first), argmax(&negative_first)),
    ] {
        assert_eq!(
            positive_extremum.map(|(index, value)| (index, value.to_bits())),
            Some((0, 0.0_f32.to_bits()))
        );
        assert_eq!(
            negative_extremum.map(|(index, value)| (index, value.to_bits())),
            Some((0, (-0.0_f32).to_bits()))
        );
    }
}

/// Length covering several vector iterations plus a partial tail on every
/// supported lane width (4, 8, 16, and 32 lanes).
const VECTORIZED_SCAN_LEN: usize = 132;

/// Scalar oracle for the extremum contract, written independently of the
/// vectorized scan: reject any NaN, then report the first element equal to the
/// extremum along with its own stored representation.
fn reference_extremum(data: &[f32], maximum: bool) -> Option<(usize, f32)> {
    let &head = data.first()?;
    if data.iter().any(|value| value.is_nan()) {
        return None;
    }
    let mut extremum = head;
    for &value in data {
        if (maximum && value > extremum) || (!maximum && value < extremum) {
            extremum = value;
        }
    }
    let index = data.iter().position(|value| *value == extremum)?;
    Some((index, data[index]))
}

#[test]
fn extrema_reject_nan_throughout_vectorized_scan() {
    // Positions land in the first vector, on lane boundaries for each width, in
    // interior vectors, and in the scalar tail.
    for nan_at in [0, 1, 7, 8, 15, 16, 31, 32, 64, 127, 128, 131] {
        let mut data = vec![1.0_f32; VECTORIZED_SCAN_LEN];
        data[nan_at] = f32::NAN;
        assert_eq!(argmin(&data), None, "argmin with NaN at {nan_at}");
        assert_eq!(argmax(&data), None, "argmax with NaN at {nan_at}");
    }
}

#[test]
fn extrema_report_first_duplicate_across_lane_boundaries() {
    for first_at in [0, 5, 8, 17, 31, 64, 127] {
        let mut data = vec![5.0_f32; VECTORIZED_SCAN_LEN];
        data[first_at] = -1.0;
        data[first_at + 1] = -1.0;
        assert_eq!(
            argmin(&data).map(|(index, _)| index),
            Some(first_at),
            "argmin duplicate minimum starting at {first_at}"
        );

        let mut data = vec![-5.0_f32; VECTORIZED_SCAN_LEN];
        data[first_at] = 1.0;
        data[first_at + 1] = 1.0;
        assert_eq!(
            argmax(&data).map(|(index, _)| index),
            Some(first_at),
            "argmax duplicate maximum starting at {first_at}"
        );
    }
}

#[test]
fn extrema_preserve_signed_zero_inside_vector_body() {
    // `-0.0` and `0.0` compare equal, so the first of the pair must win and
    // must be reported with its own bit pattern rather than the reduced value.
    let mut data = vec![1.0_f32; VECTORIZED_SCAN_LEN];
    data[40] = -0.0;
    data[41] = 0.0;
    let (index, value) = argmin(&data).expect("no NaN present");
    assert_eq!(index, 40);
    assert_eq!(value.to_bits(), (-0.0_f32).to_bits());

    let mut data = vec![1.0_f32; VECTORIZED_SCAN_LEN];
    data[40] = 0.0;
    data[41] = -0.0;
    let (index, value) = argmin(&data).expect("no NaN present");
    assert_eq!(index, 40);
    assert_eq!(value.to_bits(), 0.0_f32.to_bits());
}

proptest! {
    /// The vectorized scan must agree with the scalar oracle on index, on the
    /// exact stored bit pattern, and on NaN rejection, at every length spanning
    /// the vector body and its tail.
    #[test]
    fn prop_extrema_match_scalar_oracle(
        values in prop::collection::vec(
            prop_oneof![
                90 => -100.0_f32..100.0,
                5 => Just(0.0_f32),
                5 => Just(-0.0_f32),
            ],
            1..300,
        ),
        nan_at in prop::option::of(0usize..300),
    ) {
        let mut values = values;
        if let Some(at) = nan_at {
            let at = at % values.len();
            values[at] = f32::NAN;
        }

        let bits = |extremum: Option<(usize, f32)>| {
            extremum.map(|(index, value)| (index, value.to_bits()))
        };
        prop_assert_eq!(bits(argmin(&values)), bits(reference_extremum(&values, false)));
        prop_assert_eq!(bits(argmax(&values)), bits(reference_extremum(&values, true)));
    }
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
    assert_eq!(
        &data,
        &[2.0f32, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0]
    );
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
    assert_eq!(err, Err(SimdError::InsufficientOutputLength));
}

/// The rounding ops (`Round`, ties to even; `Floor`; `Ceil`; `Trunc`) reach the
/// `UnaryOp` seam exactly as the kernel-level differential tests define them:
/// the reference is the plain-scalar rounding family, and the input list
/// exercises exact halfway ties, values straddling a tie, ±Inf, and signed
/// zeros. Value semantics are asserted bit-exactly so a wrong tie resolution or
/// a lost sign bit cannot pass.
#[test]
fn test_map_unary_rounding_ops() {
    let data = [
        -2.5f32,
        -1.5,
        -0.5,
        0.0,
        0.5,
        1.5,
        2.5,
        3.5,
        1.499_999_9,
        -1.500_000_1,
        f32::INFINITY,
        f32::NEG_INFINITY,
        -0.0,
        1.0e20,
    ];
    let expected_round: Vec<f32> = data.iter().map(|&x| f32::round_ties_even(x)).collect();
    let expected_floor: Vec<f32> = data.iter().map(|&x| x.floor()).collect();
    let expected_ceil: Vec<f32> = data.iter().map(|&x| x.ceil()).collect();
    let expected_trunc: Vec<f32> = data.iter().map(|&x| x.trunc()).collect();

    let mut round_out = vec![0.0f32; data.len()];
    let mut floor_out = vec![0.0f32; data.len()];
    let mut ceil_out = vec![0.0f32; data.len()];
    let mut trunc_out = vec![0.0f32; data.len()];

    v(&data).map_unary(Round, &mut round_out).unwrap();
    v(&data).map_unary(Floor, &mut floor_out).unwrap();
    v(&data).map_unary(Ceil, &mut ceil_out).unwrap();
    v(&data).map_unary(Trunc, &mut trunc_out).unwrap();

    for (i, &x) in data.iter().enumerate() {
        assert_eq!(
            round_out[i].to_bits(),
            expected_round[i].to_bits(),
            "round({x:e})"
        );
        assert_eq!(
            floor_out[i].to_bits(),
            expected_floor[i].to_bits(),
            "floor({x:e})"
        );
        assert_eq!(
            ceil_out[i].to_bits(),
            expected_ceil[i].to_bits(),
            "ceil({x:e})"
        );
        assert_eq!(
            trunc_out[i].to_bits(),
            expected_trunc[i].to_bits(),
            "trunc({x:e})"
        );
    }
}

#[test]
fn test_map_unary_in_place_round() {
    let mut data = [-2.5f32, -0.5, 0.5, 1.5, 2.5, -0.0, 1.0e20];
    let expected: Vec<f32> = data.iter().map(|&x| f32::round_ties_even(x)).collect();
    v_mut(&mut data).map_unary_in_place(Round);
    for (got, exp) in data.iter().zip(expected.iter()) {
        assert_eq!(got.to_bits(), exp.to_bits());
    }
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
    assert_eq!(err, Err(SimdError::InsufficientOutputLength));
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
    assert_simd_error(err, SimdError::LengthMismatch);
}

#[test]
fn test_select_mask_too_short() {
    let a = [1.0f32; 5];
    let b = [10.0f32; 5];
    let mask = [true; 3];
    let err = v(&a).select(&mask, &v(&b));
    assert_simd_error(err, SimdError::InsufficientOutputLength);
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
    assert_eq!(err, Err(SimdError::InsufficientOutputLength));
}

#[test]
fn test_gather_index_out_of_bounds() {
    let data = [1.0f32; 4];
    let indices = [0i32, 1, 2, 10]; // 10 >= 4
    let mut out = vec![0.0f32; 4];
    let err = v(&data).gather(&indices, &mut out);
    assert_eq!(err, Err(SimdError::IndexOutOfBounds));
}

#[test]
fn test_gather_negative_index() {
    let data = [1.0f32; 4];
    let indices = [-1i32, 0, 1, 2];
    let mut out = vec![0.0f32; 4];
    let err = v(&data).gather(&indices, &mut out);
    assert_eq!(err, Err(SimdError::IndexOutOfBounds));
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
    assert_simd_error(err, SimdError::LengthMismatch);
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

#[test]
fn test_reduce_popcount_masked_tails_cover_multiple_widths() {
    use hermes_simd::{
        reduce_popcount, reduce_popcount_and, reduce_popcount_or, reduce_popcount_xor,
    };

    for &len in &[1usize, 2, 3, 5, 9, 17, 65, 133] {
        let a: Vec<i32> = (0..len)
            .map(|index| (index as i32).wrapping_mul(0x1357_9bdf))
            .collect();
        let b: Vec<i32> = (0..len)
            .map(|index| (index as i32).wrapping_mul(0x2468_ace1))
            .collect();
        let expected = |value: i32| value.count_ones() as usize;
        assert_eq!(
            reduce_popcount(&a),
            a.iter().map(|&value| expected(value)).sum::<usize>(),
            "single len {len}"
        );
        assert_eq!(
            reduce_popcount_and(&a, &b).unwrap(),
            a.iter()
                .zip(&b)
                .map(|(&left, &right)| expected(left & right))
                .sum::<usize>(),
            "and len {len}"
        );
        assert_eq!(
            reduce_popcount_or(&a, &b).unwrap(),
            a.iter()
                .zip(&b)
                .map(|(&left, &right)| expected(left | right))
                .sum::<usize>(),
            "or len {len}"
        );
        assert_eq!(
            reduce_popcount_xor(&a, &b).unwrap(),
            a.iter()
                .zip(&b)
                .map(|(&left, &right)| expected(left ^ right))
                .sum::<usize>(),
            "xor len {len}"
        );
    }
}

#[test]
fn generic_reductions_and_masked_view_tails_match_scalar_contracts() {
    use hermes_simd_core::mask::BitMask;

    // Scalar uses four lanes, so length five exercises a one-element masked
    // tail while the mask still contains inactive lanes beyond the live range.
    let values = [1.0_f32, -2.0, 3.0, 4.0, 5.0];
    let view = v(&values);
    assert_eq!(view.reduce(Sum), 11.0);
    assert_eq!(view.reduce(Min), -2.0);
    assert_eq!(view.reduce(Max), 5.0);

    let left = [1.0_f32, 2.0, 3.0, 4.0, 5.0];
    let right = [10.0_f32, 20.0, 30.0, 40.0, 50.0];
    let left_view = v(&left);
    let right_view = v(&right);
    let mask = BitMask::<4>::from_bools(&[true, false, true, false]);
    let mut out = [0.0_f32; 5];

    left_view.masked_add(&right_view, &mask, &mut out).unwrap();
    assert_eq!(out, [11.0, 2.0, 33.0, 4.0, 55.0]);

    left_view.masked_mul(&right_view, &mask, &mut out).unwrap();
    assert_eq!(out, [10.0, 2.0, 90.0, 4.0, 250.0]);

    left_view
        .masked_fmadd(&right_view, &left_view, &mask, &mut out)
        .unwrap();
    assert_eq!(out, [11.0, 2.0, 93.0, 4.0, 255.0]);

    left_view.elementwise_mul(&right_view, &mut out).unwrap();
    assert_eq!(out, [10.0, 40.0, 90.0, 160.0, 250.0]);

    left_view
        .zip_into(&right_view, &mut out, hermes_simd::Add)
        .unwrap();
    assert_eq!(out, [11.0, 22.0, 33.0, 44.0, 55.0]);
}

#[test]
fn generic_extrema_tail_preserves_ordering_contract() {
    // Generic Min/Max use Eunomia's NumericElement min/max contract, which
    // ignores a NaN operand (argmin/argmax separately reject NaN inputs).
    let nan_tail = [1.0_f32, 2.0, 3.0, 4.0, f32::NAN];
    let scalar_nan = v(&nan_tail);
    assert_eq!(scalar_nan.reduce(Min), 1.0);
    assert_eq!(scalar_nan.reduce(Max), 4.0);

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if hermes_simd::Avx2::is_runtime_supported() {
        // The runtime-dispatch path uses the native AVX2 min/max kernels on
        // supported hosts; keep this differential check host-conditional.
        let native_nan = hermes_simd::SimdView::<
            f32,
            hermes_simd::Avx2,
            hermes_simd::Unaligned,
            hermes_simd::Unmasked,
            &[f32],
        >::new(&nan_tail)
        .expect("AVX2 probe succeeded");
        assert_eq!(native_nan.reduce(Min), 1.0);
        assert_eq!(native_nan.reduce(Max), 4.0);
    }

    let negative_zero_tail = [1.0_f32, 2.0, 3.0, 4.0, -0.0];
    let zero_view = v(&negative_zero_tail);
    assert_eq!(zero_view.reduce(Min).to_bits(), (-0.0_f32).to_bits());
    assert_eq!(zero_view.reduce(Max), 4.0);
}

#[test]
fn test_reduce_popcount_dispatch() {
    use hermes_simd::{
        reduce_popcount, reduce_popcount_and, reduce_popcount_or, reduce_popcount_xor,
    };

    let a = [0b0001i32, 0b0011, 0b0111, 0b1111, 0b0000]; // popcounts: 1, 2, 3, 4, 0 -> sum = 10
    let b = [0b0011i32, 0b0011, 0b0011, 0b0011, 0b0011]; // popcounts: 2, 2, 2, 2, 2 -> sum = 10

    assert_eq!(reduce_popcount(&a), 10);
    assert_eq!(reduce_popcount(&b), 10);

    // a & b = [0b0001, 0b0011, 0b0011, 0b0011, 0b0000] -> popcounts: 1, 2, 2, 2, 0 -> sum = 7
    assert_eq!(reduce_popcount_and(&a, &b).unwrap(), 7);

    // a | b = [0b0011, 0b0011, 0b0111, 0b1111, 0b0011] -> popcounts: 2, 2, 3, 4, 2 -> sum = 13
    assert_eq!(reduce_popcount_or(&a, &b).unwrap(), 13);

    // a ^ b = [0b0010, 0b0000, 0b0100, 0b1100, 0b0011] -> popcounts: 1, 0, 1, 2, 2 -> sum = 6
    assert_eq!(reduce_popcount_xor(&a, &b).unwrap(), 6);
}
