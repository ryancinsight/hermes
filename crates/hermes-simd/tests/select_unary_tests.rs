//! Integration tests for `SimdView::select`, `masked_negate`,
//! `SimdView::map_unary` / `map_unary_in_place`, and `SimdView::prefix_scan`
//! ZST strategy correctness.

use hermes_simd::{
    SimdView, SimdCow, Unaligned, Unmasked, Scalar,
    ScanAdd, ScanMin, ScanMax, Inclusive, Exclusive,
    Abs, Neg, Sqrt, Clamp,
};

type View<'a, T> = SimdView<'a, T, Scalar, Unaligned, Unmasked, &'a [T]>;
type ViewMut<'a, T> = SimdView<'a, T, Scalar, Unaligned, Unmasked, &'a mut [T]>;
type Cow<'a, T> = SimdCow<'a, T, Scalar, Unaligned>;

fn view<T>(data: &[T]) -> View<'_, T> {
    SimdView::new(data).expect("Scalar/Unaligned always succeeds")
}
fn view_mut<T>(data: &mut [T]) -> ViewMut<'_, T> {
    SimdView::new_mut(data).expect("Scalar/Unaligned always succeeds")
}

// ---------------------------------------------------------------------------
// select
// ---------------------------------------------------------------------------

#[test]
fn test_select_basic_f32() {
    let a = [1.0f32, 2.0, 3.0, 4.0];
    let b = [10.0f32, 20.0, 30.0, 40.0];
    let mask = [true, false, true, false];

    let va = view(&a);
    let vb = view(&b);
    let result = va.select(&mask, &vb).unwrap();
    assert_eq!(result.as_slice(), &[1.0f32, 20.0, 3.0, 40.0]);
}

#[test]
fn test_select_all_true() {
    let a = [1.0f32, 2.0, 3.0];
    let b = [10.0f32, 20.0, 30.0];
    let mask = [true, true, true];
    let va = view(&a);
    let vb = view(&b);
    let result = va.select(&mask, &vb).unwrap();
    assert_eq!(result.as_slice(), &a);
}

#[test]
fn test_select_all_false() {
    let a = [1.0f32, 2.0, 3.0];
    let b = [10.0f32, 20.0, 30.0];
    let mask = [false, false, false];
    let va = view(&a);
    let vb = view(&b);
    let result = va.select(&mask, &vb).unwrap();
    assert_eq!(result.as_slice(), &b);
}

#[test]
fn test_select_length_mismatch() {
    let a = [1.0f32, 2.0];
    let b = [10.0f32, 20.0, 30.0];
    let mask = [true, true, false];
    let va = view(&a);
    let vb = view(&b);
    assert!(va.select(&mask, &vb).is_err());
}

#[test]
fn test_select_mask_too_short() {
    let a = [1.0f32, 2.0, 3.0];
    let b = [10.0f32, 20.0, 30.0];
    let mask = [true];
    let va = view(&a);
    let vb = view(&b);
    assert!(va.select(&mask, &vb).is_err());
}

// ---------------------------------------------------------------------------
// masked_negate
// ---------------------------------------------------------------------------

#[test]
fn test_masked_negate_f32() {
    let data = [1.0f32, -2.0, 3.0, -4.0];
    let mask = [true, false, true, false];
    let v = view(&data);
    let result = v.masked_negate(&mask).unwrap();
    // Masked lanes: negate; unmasked: passthrough
    assert_eq!(result.as_slice(), &[-1.0f32, -2.0, -3.0, -4.0]);
}

#[test]
fn test_masked_negate_all_false_passthrough() {
    let data = [1.0f32, 2.0, 3.0];
    let mask = [false, false, false];
    let v = view(&data);
    let result = v.masked_negate(&mask).unwrap();
    assert_eq!(result.as_slice(), &[1.0f32, 2.0, 3.0]);
}

// ---------------------------------------------------------------------------
// map_unary / map_unary_in_place
// ---------------------------------------------------------------------------

#[test]
fn test_map_unary_abs_f32() {
    let data = [-1.0f32, -2.0, 3.0, -4.0];
    let v = view(&data);
    let mut out = [0.0f32; 4];
    v.map_unary(Abs, &mut out).unwrap();
    assert_eq!(out, [1.0f32, 2.0, 3.0, 4.0]);
}

#[test]
fn test_map_unary_neg_f32() {
    let data = [1.0f32, -2.0, 3.0];
    let v = view(&data);
    let mut out = [0.0f32; 3];
    v.map_unary(Neg, &mut out).unwrap();
    assert_eq!(out, [-1.0f32, 2.0, -3.0]);
}

#[test]
fn test_map_unary_sqrt_f32() {
    let data = [4.0f32, 9.0, 16.0, 25.0];
    let v = view(&data);
    let mut out = [0.0f32; 4];
    v.map_unary(Sqrt, &mut out).unwrap();
    let expected = [2.0f32, 3.0, 4.0, 5.0];
    for (a, b) in out.iter().zip(expected.iter()) {
        assert!((a - b).abs() < 1e-5, "got={a}, expected={b}");
    }
}

#[test]
fn test_map_unary_clamp_f32() {
    let data = [-5.0f32, 0.5, 2.0, 10.0];
    let v = view(&data);
    let mut out = [0.0f32; 4];
    v.map_unary(Clamp::new(0.0f32, 1.0), &mut out).unwrap();
    assert_eq!(out, [0.0f32, 0.5, 1.0, 1.0]);
}

#[test]
fn test_map_unary_in_place_abs() {
    let mut data = [-1.0f32, -2.0, 3.0, -4.0];
    let mut v = view_mut(&mut data);
    v.map_unary_in_place(Abs);
    assert_eq!(data, [1.0f32, 2.0, 3.0, 4.0]);
}

#[test]
fn test_map_unary_out_too_short() {
    let data = [1.0f32, 2.0, 3.0, 4.0];
    let v = view(&data);
    let mut out = [0.0f32; 2];
    assert!(v.map_unary(Abs, &mut out).is_err());
}

// ---------------------------------------------------------------------------
// SimdCow::map_unary / map_unary_in_place
// ---------------------------------------------------------------------------

#[test]
fn test_cow_map_unary_abs() {
    let data = vec![-1.0f32, -2.0, 3.0];
    let cow = Cow::<f32>::borrow_slice(&data).unwrap();
    let result = cow.map_unary(Abs);
    assert!(matches!(result, SimdCow::Owned(_)));
    assert_eq!(result.as_ref(), &[1.0f32, 2.0, 3.0]);
}

#[test]
fn test_cow_map_unary_in_place_promotes_borrowed() {
    let data = vec![1.0f32, -2.0, 3.0, -4.0];
    let mut cow = Cow::<f32>::borrow_slice(&data).unwrap();
    assert!(matches!(cow, SimdCow::Borrowed(_)));
    cow.map_unary_in_place(Abs);
    assert!(matches!(cow, SimdCow::Owned(_)));
    assert_eq!(cow.as_ref(), &[1.0f32, 2.0, 3.0, 4.0]);
}

#[test]
fn test_cow_map_unary_in_place_already_owned_no_realloc() {
    let data = vec![-1.0f32, -2.0, -3.0, -4.0];
    let mut cow = Cow::<f32>::from_slice(&data);
    let ptr_before = cow.view().as_slice().as_ptr();
    cow.map_unary_in_place(Abs);
    // Already owned → to_mut returns same buffer → pointer unchanged
    let ptr_after = cow.view().as_slice().as_ptr();
    assert_eq!(ptr_before, ptr_after);
    assert_eq!(cow.as_ref(), &[1.0f32, 2.0, 3.0, 4.0]);
}

// ---------------------------------------------------------------------------
// prefix_scan — select/unary angle
// ---------------------------------------------------------------------------

#[test]
fn test_prefix_scan_select_identity() {
    // Exclusive sum of the selection mask (counts true elements up to i)
    let data = [1.0f32, 1.0, 1.0, 1.0, 1.0];
    let v = view(&data);
    let mut out = [0.0f32; 5];
    v.prefix_scan(&mut out, ScanAdd, Exclusive).unwrap();
    assert_eq!(out, [0.0f32, 1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn test_prefix_scan_min_unary_correctness() {
    let data = [3.0f32, 1.0, 4.0, 1.0, 5.0];
    let v = view(&data);
    let mut out = [0.0f32; 5];
    v.prefix_scan(&mut out, ScanMin, Inclusive).unwrap();
    assert_eq!(out, [3.0f32, 1.0, 1.0, 1.0, 1.0]);
}

#[test]
fn test_prefix_scan_max_unary_correctness() {
    let data = [3.0f32, 1.0, 4.0, 1.0, 5.0];
    let v = view(&data);
    let mut out = [0.0f32; 5];
    v.prefix_scan(&mut out, ScanMax, Inclusive).unwrap();
    assert_eq!(out, [3.0f32, 3.0, 4.0, 4.0, 5.0]);
}

// ---------------------------------------------------------------------------
// SimdCow::map_cow — generic UnaryOp dispatch (cow/unary.rs)
// ---------------------------------------------------------------------------

#[test]
fn test_map_cow_abs_f32() {
    let data = [-1.0f32, 2.0, -3.0, 4.0];
    let cow  = Cow::<f32>::borrow_slice(&data).unwrap();
    let out  = cow.map_cow(Abs);
    assert_eq!(&*out, &[1.0f32, 2.0, 3.0, 4.0]);
}

#[test]
fn test_map_cow_neg_f32() {
    let data = [1.0f32, -2.0, 3.0, -4.0];
    let cow  = Cow::<f32>::borrow_slice(&data).unwrap();
    let out  = cow.map_cow(Neg);
    assert_eq!(&*out, &[-1.0f32, 2.0, -3.0, 4.0]);
}

#[test]
fn test_map_cow_sqrt_f32() {
    let data = [4.0f32, 9.0, 16.0, 25.0];
    let cow  = Cow::<f32>::borrow_slice(&data).unwrap();
    let out  = cow.map_cow(Sqrt);
    let expected = [2.0f32, 3.0, 4.0, 5.0];
    for (a, b) in out.iter().zip(expected.iter()) {
        assert!((a - b).abs() < 1e-5, "got={a}, expected={b}");
    }
}

#[test]
fn test_map_cow_returns_owned() {
    let data = [1.0f32, 2.0, 3.0];
    let cow  = Cow::<f32>::borrow_slice(&data).unwrap();
    let out  = cow.map_cow(Abs);
    assert!(matches!(out, SimdCow::Owned(_)));
}

// ---------------------------------------------------------------------------
// SimdCow::fma_cow — fused multiply-add (cow/unary.rs)
// ---------------------------------------------------------------------------

#[test]
fn test_fma_cow_basic_f32() {
    let a = [1.0f32, 2.0, 3.0, 4.0];
    let b = [2.0f32, 2.0, 2.0, 2.0];
    let c = [0.5f32, 0.5, 0.5, 0.5];
    let ca = Cow::<f32>::borrow_slice(&a).unwrap();
    let cb = Cow::<f32>::borrow_slice(&b).unwrap();
    let cc = Cow::<f32>::borrow_slice(&c).unwrap();
    let out = ca.fma_cow(&cb, &cc).unwrap();
    assert_eq!(&*out, &[2.5f32, 4.5, 6.5, 8.5]);
}

#[test]
fn test_fma_cow_zero_addend() {
    let a = [3.0f32, 4.0];
    let b = [2.0f32, 2.0];
    let c = [0.0f32, 0.0];
    let ca = Cow::<f32>::borrow_slice(&a).unwrap();
    let cb = Cow::<f32>::borrow_slice(&b).unwrap();
    let cc = Cow::<f32>::borrow_slice(&c).unwrap();
    let out = ca.fma_cow(&cb, &cc).unwrap();
    assert_eq!(&*out, &[6.0f32, 8.0]);
}

#[test]
fn test_fma_cow_length_mismatch() {
    let a = [1.0f32, 2.0];
    let b = [1.0f32, 2.0, 3.0];
    let c = [0.0f32, 0.0, 0.0];
    let ca = Cow::<f32>::borrow_slice(&a).unwrap();
    let cb = Cow::<f32>::borrow_slice(&b).unwrap();
    let cc = Cow::<f32>::borrow_slice(&c).unwrap();
    assert!(ca.fma_cow(&cb, &cc).is_err());
}
