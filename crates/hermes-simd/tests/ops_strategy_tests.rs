//! Integration tests for generic `ElementOp` strategy dispatch:
//! `transform_in_place`, `zip_into`, and invariant checks that
//! `add_assign`/`mul_assign`/`elementwise_mul` produce identical results.

use hermes_simd::{
    Add, BitAnd, BitOr, BitXor, Div, Mul, Scalar, SimdError, SimdView, Sub, SveArch, Unaligned,
    Unmasked,
};
use hermes_simd_core::kernel::SimdKernel;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

type View<'a, T> = SimdView<'a, T, Scalar, Unaligned, Unmasked, &'a [T]>;
type ViewMut<'a, T> = SimdView<'a, T, Scalar, Unaligned, Unmasked, &'a mut [T]>;

fn make_view(data: &[f32]) -> View<'_, f32> {
    SimdView::new(data).expect("Scalar/Unaligned always succeeds")
}
fn make_view_mut(data: &mut [f32]) -> ViewMut<'_, f32> {
    SimdView::new_mut(data).expect("Scalar/Unaligned always succeeds")
}

// ---------------------------------------------------------------------------
// transform_in_place — all strategies
// ---------------------------------------------------------------------------

#[test]
fn test_transform_in_place_add() {
    let a_orig = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
    let b = [10.0f32, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0];
    let expected: Vec<f32> = a_orig.iter().zip(b.iter()).map(|(a, b)| a + b).collect();

    let mut a = a_orig;
    let b_view = make_view(&b);
    let mut a_view = make_view_mut(&mut a);
    a_view.transform_in_place(&b_view, Add).unwrap();

    assert_eq!(&a[..], expected.as_slice());
}

#[test]
fn test_transform_in_place_sub() {
    let a_orig = [10.0f32, 20.0, 30.0, 40.0, 50.0, 7.0, 8.0, 9.0, 11.0];
    let b = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
    let expected: Vec<f32> = a_orig.iter().zip(b.iter()).map(|(a, b)| a - b).collect();

    let mut a = a_orig;
    let b_view = make_view(&b);
    let mut a_view = make_view_mut(&mut a);
    a_view.transform_in_place(&b_view, Sub).unwrap();

    assert_eq!(&a[..], expected.as_slice());
}

#[test]
fn test_transform_in_place_mul() {
    let a_orig = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
    let b = [2.0f32, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
    let expected: Vec<f32> = a_orig.iter().zip(b.iter()).map(|(a, b)| a * b).collect();

    let mut a = a_orig;
    let b_view = make_view(&b);
    let mut a_view = make_view_mut(&mut a);
    a_view.transform_in_place(&b_view, Mul).unwrap();

    assert_eq!(&a[..], expected.as_slice());
}

#[test]
fn test_transform_in_place_div() {
    let a_orig = [10.0f32, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0];
    let b = [2.0f32, 4.0, 5.0, 8.0, 10.0, 6.0, 7.0, 8.0, 9.0];
    let expected: Vec<f32> = a_orig.iter().zip(b.iter()).map(|(a, b)| a / b).collect();

    let mut a = a_orig;
    let b_view = make_view(&b);
    let mut a_view = make_view_mut(&mut a);
    a_view.transform_in_place(&b_view, Div).unwrap();

    for (got, exp) in a.iter().zip(expected.iter()) {
        assert!((got - exp).abs() < 1e-5, "got={got}, expected={exp}");
    }
}

// ---------------------------------------------------------------------------
// transform_in_place — delegates match (regression check)
// ---------------------------------------------------------------------------

#[test]
fn test_add_assign_matches_transform_in_place_add() {
    let a_orig = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
    let b = [10.0f32; 10];

    let mut a_generic = a_orig;
    {
        let b_view = make_view(&b);
        let mut view = make_view_mut(&mut a_generic);
        view.transform_in_place(&b_view, Add).unwrap();
    }

    let mut a_delegate = a_orig;
    {
        let b_view = make_view(&b);
        let mut view = make_view_mut(&mut a_delegate);
        view.add_assign(&b_view).unwrap();
    }

    assert_eq!(a_generic, a_delegate);
}

#[test]
fn test_mul_assign_matches_transform_in_place_mul() {
    let a_orig = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
    let b = [3.0f32; 10];

    let mut a_generic = a_orig;
    {
        let b_view = make_view(&b);
        let mut view = make_view_mut(&mut a_generic);
        view.transform_in_place(&b_view, Mul).unwrap();
    }

    let mut a_delegate = a_orig;
    {
        let b_view = make_view(&b);
        let mut view = make_view_mut(&mut a_delegate);
        view.mul_assign(&b_view).unwrap();
    }

    assert_eq!(a_generic, a_delegate);
}

// ---------------------------------------------------------------------------
// zip_into — all strategies
// ---------------------------------------------------------------------------

#[test]
fn test_zip_into_add() {
    let a = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
    let b = [10.0f32, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0];
    let expected: Vec<f32> = a.iter().zip(b.iter()).map(|(a, b)| a + b).collect();

    let mut out = vec![0.0f32; 9];
    let a_view = make_view(&a);
    let b_view = make_view(&b);
    a_view.zip_into(&b_view, &mut out, Add).unwrap();

    assert_eq!(out, expected);
}

#[test]
fn test_zip_into_sub() {
    let a = [10.0f32, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0];
    let b = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
    let expected: Vec<f32> = a.iter().zip(b.iter()).map(|(a, b)| a - b).collect();

    let mut out = vec![0.0f32; 9];
    let a_view = make_view(&a);
    let b_view = make_view(&b);
    a_view.zip_into(&b_view, &mut out, Sub).unwrap();

    assert_eq!(out, expected);
}

#[test]
fn test_zip_into_mul_matches_elementwise_mul() {
    let a = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0];
    let b = [2.0f32; 11];

    let mut out_generic = vec![0.0f32; 11];
    let mut out_delegate = vec![0.0f32; 11];

    {
        let a_view = make_view(&a);
        let b_view = make_view(&b);
        a_view.zip_into(&b_view, &mut out_generic, Mul).unwrap();
    }
    {
        let a_view = make_view(&a);
        let b_view = make_view(&b);
        a_view.elementwise_mul(&b_view, &mut out_delegate).unwrap();
    }

    assert_eq!(out_generic, out_delegate);
}

#[test]
fn test_zip_into_div() {
    let a = [10.0f32, 20.0, 30.0, 40.0];
    let b = [2.0f32, 4.0, 5.0, 8.0];
    let expected: Vec<f32> = a.iter().zip(b.iter()).map(|(a, b)| a / b).collect();

    let mut out = vec![0.0f32; 4];
    let a_view = make_view(&a);
    let b_view = make_view(&b);
    a_view.zip_into(&b_view, &mut out, Div).unwrap();

    for (got, exp) in out.iter().zip(expected.iter()) {
        assert!((got - exp).abs() < 1e-5, "got={got}, expected={exp}");
    }
}

// ---------------------------------------------------------------------------
// Error paths
// ---------------------------------------------------------------------------

#[test]
fn test_transform_in_place_length_mismatch() {
    let a = [1.0f32; 4];
    let b = [1.0f32; 5];
    let mut a = a;
    let b_view = make_view(&b);
    let mut a_view = make_view_mut(&mut a);
    let err = a_view.transform_in_place(&b_view, Add);
    assert_eq!(err, Err(SimdError::LengthMismatch));
}

#[test]
fn test_zip_into_length_mismatch() {
    let a = [1.0f32; 4];
    let b = [1.0f32; 5];
    let mut out = vec![0.0f32; 4];
    let a_view = make_view(&a);
    let b_view = make_view(&b);
    let err = a_view.zip_into(&b_view, &mut out, Add);
    assert_eq!(err, Err(SimdError::LengthMismatch));
}

#[test]
fn test_zip_into_output_too_short() {
    let a = [1.0f32; 8];
    let b = [1.0f32; 8];
    let mut out = vec![0.0f32; 4]; // too short
    let a_view = make_view(&a);
    let b_view = make_view(&b);
    let err = a_view.zip_into(&b_view, &mut out, Add);
    assert_eq!(err, Err(SimdError::InsufficientOutputLength));
}

// ---------------------------------------------------------------------------
// ZST size invariants
// ---------------------------------------------------------------------------

#[test]
fn test_strategy_zst_sizes() {
    use core::mem::size_of;
    assert_eq!(size_of::<Add>(), 0);
    assert_eq!(size_of::<Sub>(), 0);
    assert_eq!(size_of::<Mul>(), 0);
    assert_eq!(size_of::<Div>(), 0);
    assert_eq!(size_of::<BitAnd>(), 0);
    assert_eq!(size_of::<BitOr>(), 0);
    assert_eq!(size_of::<BitXor>(), 0);
}

// ---------------------------------------------------------------------------
// Tail-element correctness (len not divisible by LANE_COUNT)
// ---------------------------------------------------------------------------

#[test]
fn test_transform_in_place_odd_length() {
    // 5 elements with Scalar (lane=1) — exercises the scalar tail path.
    let a_orig = [1.0f32, 2.0, 3.0, 4.0, 5.0];
    let b = [10.0f32, 20.0, 30.0, 40.0, 50.0];
    let expected: Vec<f32> = a_orig.iter().zip(b.iter()).map(|(x, y)| x + y).collect();

    let mut a = a_orig;
    let b_view = make_view(&b);
    let mut a_view = make_view_mut(&mut a);
    a_view.transform_in_place(&b_view, Add).unwrap();

    assert_eq!(&a[..], expected.as_slice());
}

#[test]
fn test_transform_in_place_masked_tail_forced_sve() {
    let len = <SveArch as SimdKernel<f32>>::LANE_COUNT + 3;
    let a: Vec<f32> = (0..len).map(|index| index as f32 + 0.25).collect();
    let b: Vec<f32> = (0..len).map(|index| index as f32 * -0.5).collect();
    let expected: Vec<f32> = a
        .iter()
        .zip(&b)
        .map(|(&left, &right)| left + right)
        .collect();

    let b_view = SimdView::<f32, SveArch, Unaligned, Unmasked, &[f32]>::new(&b)
        .expect("emulated SVE backend is always constructible");
    let mut actual = a;
    let mut a_view =
        SimdView::<f32, SveArch, Unaligned, Unmasked, &mut [f32]>::new_mut(&mut actual)
            .expect("emulated SVE backend is always constructible");
    a_view.transform_in_place(&b_view, Add).unwrap();

    assert_eq!(actual, expected);
}

#[test]
fn test_zip_into_odd_length() {
    let a = [1.0f32, 2.0, 3.0, 4.0, 5.0];
    let b = [1.0f32, 2.0, 3.0, 4.0, 5.0];
    let expected: Vec<f32> = a.iter().zip(b.iter()).map(|(x, y)| x * y).collect();

    let mut out = vec![0.0f32; 5];
    let a_view = make_view(&a);
    let b_view = make_view(&b);
    a_view.zip_into(&b_view, &mut out, Mul).unwrap();

    assert_eq!(out, expected);
}
