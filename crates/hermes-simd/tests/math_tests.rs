//! Integration tests for `SimdCow` math extensions:
//! `norm_sq`, norm, normalize, add/sub/mul/div scalar broadcast, `div_cow`.
//!
//! All tests run against the `Scalar` arch (lane=1) for portability.
//! The scalar fallback is the authoritative correctness reference.

#![expect(
    clippy::float_cmp,
    reason = "These integration tests assert exact manufactured scalar reference values"
)]
#![expect(
    clippy::needless_pass_by_value,
    reason = "The shared error helper consumes heterogeneous Result values to inspect their error variant"
)]

use hermes_simd::{Scalar, SimdCow, SimdError, Unaligned};

type Cow<'a, T> = SimdCow<'a, T, Scalar, Unaligned>;

fn assert_simd_error<T>(result: Result<T, SimdError>, expected: SimdError) {
    match result {
        Err(actual) => assert_eq!(actual, expected),
        Ok(_) => panic!("expected {expected:?}"),
    }
}

// ---------------------------------------------------------------------------
// Scalar-broadcast arithmetic
// ---------------------------------------------------------------------------

#[test]
fn test_add_scalar_cow_f32() {
    let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0];
    let cow = Cow::<f32>::borrow_slice(&data).unwrap();
    let result = cow.add_scalar_cow(10.0);
    assert_eq!(result.as_ref(), &[11.0f32, 12.0, 13.0, 14.0, 15.0]);
}

#[test]
fn test_sub_scalar_cow_f32() {
    let data = vec![10.0f32, 20.0, 30.0, 40.0];
    let cow = Cow::<f32>::borrow_slice(&data).unwrap();
    let result = cow.sub_scalar_cow(5.0);
    assert_eq!(result.as_ref(), &[5.0f32, 15.0, 25.0, 35.0]);
}

#[test]
fn test_mul_scalar_cow_f32() {
    let data = vec![1.0f32, 2.0, 3.0, 4.0];
    let cow = Cow::<f32>::borrow_slice(&data).unwrap();
    let result = cow.mul_scalar_cow(3.0);
    assert_eq!(result.as_ref(), &[3.0f32, 6.0, 9.0, 12.0]);
}

#[test]
fn test_div_scalar_cow_f32() {
    let data = vec![10.0f32, 20.0, 30.0, 40.0];
    let cow = Cow::<f32>::borrow_slice(&data).unwrap();
    let result = cow.div_scalar_cow(10.0);
    let expected = [1.0f32, 2.0, 3.0, 4.0];
    for (a, b) in result.as_ref().iter().zip(expected.iter()) {
        assert!((a - b).abs() < 1e-6, "got={a}, expected={b}");
    }
}

#[test]
fn test_scalar_broadcast_empty() {
    let data: Vec<f32> = vec![];
    let cow = Cow::<f32>::borrow_slice(&data).unwrap();
    assert!(cow.add_scalar_cow(5.0).is_empty());
    assert!(cow.sub_scalar_cow(5.0).is_empty());
    assert!(cow.mul_scalar_cow(5.0).is_empty());
    assert!(cow.div_scalar_cow(5.0).is_empty());
}

#[test]
fn test_scalar_broadcast_tail_handling() {
    // 5 elements — exercises scalar tail with Scalar arch (lane=1 → no tail)
    // but tests correctness for non-power-of-2 lengths with wider arches.
    let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0];
    let cow = Cow::<f32>::from_slice(&data);
    let result = cow.add_scalar_cow(100.0);
    assert_eq!(result.as_ref(), &[101.0f32, 102.0, 103.0, 104.0, 105.0]);
}

// ---------------------------------------------------------------------------
// div_cow (elementwise division)
// ---------------------------------------------------------------------------

#[test]
fn test_div_cow_f32() {
    let a_data = vec![10.0f32, 20.0, 30.0, 40.0];
    let b_data = vec![2.0f32, 4.0, 5.0, 8.0];
    let a = Cow::<f32>::borrow_slice(&a_data).unwrap();
    let b = Cow::<f32>::borrow_slice(&b_data).unwrap();
    let result = a.div_cow(&b).unwrap();
    let expected = [5.0f32, 5.0, 6.0, 5.0];
    for (got, exp) in result.as_ref().iter().zip(expected.iter()) {
        assert!((got - exp).abs() < 1e-5, "got={got}, expected={exp}");
    }
}

#[test]
fn test_div_cow_length_mismatch() {
    let a = Cow::<f32>::from_slice(&[1.0, 2.0]);
    let b = Cow::<f32>::from_slice(&[1.0, 2.0, 3.0]);
    assert_simd_error(a.div_cow(&b), SimdError::LengthMismatch);
}

// ---------------------------------------------------------------------------
// norm_sq, norm
// ---------------------------------------------------------------------------

#[test]
fn test_norm_sq_f32() {
    let data = vec![3.0f32, 4.0];
    let cow = Cow::<f32>::borrow_slice(&data).unwrap();
    // ‖(3,4)‖² = 9 + 16 = 25
    let ns = cow.norm_sq();
    assert!((ns - 25.0f32).abs() < 1e-5, "norm_sq={ns}");
}

#[test]
fn test_norm_f32() {
    let data = vec![3.0f32, 4.0];
    let cow = Cow::<f32>::borrow_slice(&data).unwrap();
    // ‖(3,4)‖ = 5
    let n = cow.norm();
    assert!((n - 5.0f32).abs() < 1e-5, "norm={n}");
}

#[test]
fn test_norm_sq_unit_vector() {
    // Unit vector — norm_sq must be 1.0
    let s = 1.0f32 / 2.0f32.sqrt();
    let data = vec![s, s];
    let cow = Cow::<f32>::borrow_slice(&data).unwrap();
    let ns = cow.norm_sq();
    assert!((ns - 1.0f32).abs() < 1e-5, "norm_sq of unit vec = {ns}");
}

#[test]
fn test_norm_zero_vector() {
    let data = vec![0.0f32, 0.0, 0.0, 0.0];
    let cow = Cow::<f32>::borrow_slice(&data).unwrap();
    assert_eq!(cow.norm_sq(), 0.0f32);
    assert_eq!(cow.norm(), 0.0f32);
}

// ---------------------------------------------------------------------------
// normalize
// ---------------------------------------------------------------------------

#[test]
fn test_normalize_f32() {
    let data = vec![3.0f32, 4.0];
    let cow = Cow::<f32>::borrow_slice(&data).unwrap();
    let unit = cow.normalize();
    // Normalized (3,4)/5 → (0.6, 0.8)
    let expected = [0.6f32, 0.8];
    for (got, exp) in unit.as_ref().iter().zip(expected.iter()) {
        assert!((got - exp).abs() < 1e-5, "got={got}, expected={exp}");
    }
    // Re-norm of result should be 1
    let renormed = SimdCow::<f32, Scalar, Unaligned>::borrow_slice(unit.as_ref()).unwrap();
    let n = renormed.norm();
    assert!((n - 1.0f32).abs() < 1e-5, "re-norm={n}");
}

#[test]
fn test_normalize_zero_vector_safe() {
    // Zero vector → normalize returns zeros, no NaN/panic
    let data = vec![0.0f32, 0.0, 0.0];
    let cow = Cow::<f32>::borrow_slice(&data).unwrap();
    let result = cow.normalize();
    assert_eq!(result.as_ref(), &[0.0f32, 0.0, 0.0]);
}

#[test]
fn test_normalize_empty_vector_safe() {
    let data: Vec<f32> = vec![];
    let cow = Cow::<f32>::borrow_slice(&data).unwrap();
    let result = cow.normalize();
    assert!(result.is_empty());
}

#[test]
fn test_normalize_single_positive() {
    // (5.0) → (1.0)
    let cow = Cow::<f32>::from_slice(&[5.0f32]);
    let unit = cow.normalize();
    assert!((unit.as_ref()[0] - 1.0f32).abs() < 1e-5);
}

#[test]
fn test_normalize_owned_returned() {
    // normalize always returns SimdCow::Owned regardless of input variant
    let data = vec![3.0f32, 4.0];
    let cow = Cow::<f32>::borrow_slice(&data).unwrap();
    let result = cow.normalize();
    assert!(matches!(result, SimdCow::Owned(_)));
}

// ---------------------------------------------------------------------------
// ZST size invariants — math extension marker types
// ---------------------------------------------------------------------------

#[test]
fn test_math_fn_zero_runtime_state() {
    use core::mem::size_of;
    // The broadcast_op helper is generic — verify no strategy type adds state.
    // Clamp<T> is the only stateful UnaryOp; all ElementOp ZSTs are size 0.
    use hermes_simd::{Add, Div, Mul, Sub};
    assert_eq!(size_of::<Add>(), 0);
    assert_eq!(size_of::<Sub>(), 0);
    assert_eq!(size_of::<Mul>(), 0);
    assert_eq!(size_of::<Div>(), 0);
}
