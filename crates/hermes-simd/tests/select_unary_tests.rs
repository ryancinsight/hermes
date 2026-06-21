//! Integration tests for `SimdView::select`, `masked_negate`,
//! `SimdView::map_unary` / `map_unary_in_place`, and `SimdView::prefix_scan`
//! ZST strategy correctness.

use hermes_simd::{
    Abs, Clamp, Exclusive, Inclusive, Neg, RecipSqrt, Scalar, ScanAdd, ScanMax, ScanMin, SimdCow,
    SimdError, SimdView, Sqrt, Unaligned, Unmasked,
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

fn assert_simd_error<T>(result: Result<T, SimdError>, expected: SimdError) {
    match result {
        Err(actual) => assert_eq!(actual, expected),
        Ok(_) => panic!("expected {expected:?}"),
    }
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
    assert_simd_error(va.select(&mask, &vb), SimdError::LengthMismatch);
}

#[test]
fn test_select_mask_too_short() {
    let a = [1.0f32, 2.0, 3.0];
    let b = [10.0f32, 20.0, 30.0];
    let mask = [true];
    let va = view(&a);
    let vb = view(&b);
    assert_simd_error(va.select(&mask, &vb), SimdError::InsufficientOutputLength);
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
    assert_simd_error(
        v.map_unary(Abs, &mut out),
        SimdError::InsufficientOutputLength,
    );
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
    let cow = Cow::<f32>::borrow_slice(&data).unwrap();
    let out = cow.map_cow(Abs);
    assert_eq!(&*out, &[1.0f32, 2.0, 3.0, 4.0]);
}

#[test]
fn test_map_cow_neg_f32() {
    let data = [1.0f32, -2.0, 3.0, -4.0];
    let cow = Cow::<f32>::borrow_slice(&data).unwrap();
    let out = cow.map_cow(Neg);
    assert_eq!(&*out, &[-1.0f32, 2.0, -3.0, 4.0]);
}

#[test]
fn test_map_cow_sqrt_f32() {
    let data = [4.0f32, 9.0, 16.0, 25.0];
    let cow = Cow::<f32>::borrow_slice(&data).unwrap();
    let out = cow.map_cow(Sqrt);
    let expected = [2.0f32, 3.0, 4.0, 5.0];
    for (a, b) in out.iter().zip(expected.iter()) {
        assert!((a - b).abs() < 1e-5, "got={a}, expected={b}");
    }
}

#[test]
fn test_map_cow_returns_owned() {
    let data = [1.0f32, 2.0, 3.0];
    let cow = Cow::<f32>::borrow_slice(&data).unwrap();
    let out = cow.map_cow(Abs);
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
    assert_simd_error(ca.fma_cow(&cb, &cc), SimdError::LengthMismatch);
}

#[test]
fn test_map_unary_recip_sqrt_f32() {
    let data = [4.0f32, 16.0, 64.0, 256.0];
    let v = view(&data);
    let mut out = [0.0f32; 4];
    v.map_unary(RecipSqrt, &mut out).unwrap();
    let expected = [0.5f32, 0.25, 0.125, 0.0625];
    for (a, b) in out.iter().zip(expected.iter()) {
        assert!((a - b).abs() < 1e-4, "got={a}, expected={b}");
    }
}

#[test]
fn test_map_cow_recip_sqrt_f32() {
    let data = [4.0f32, 16.0, 64.0, 256.0];
    let cow = Cow::<f32>::borrow_slice(&data).unwrap();
    let out = cow.map_cow(RecipSqrt);
    let expected = [0.5f32, 0.25, 0.125, 0.0625];
    for (a, b) in out.iter().zip(expected.iter()) {
        assert!((a - b).abs() < 1e-4, "got={a}, expected={b}");
    }
}

#[test]
fn test_map_unary_recip_sqrt_f64() {
    let data = [4.0f64, 16.0, 64.0, 256.0];
    let v = view(&data);
    let mut out = [0.0f64; 4];
    v.map_unary(RecipSqrt, &mut out).unwrap();
    let expected = [0.5f64, 0.25, 0.125, 0.0625];
    for (a, b) in out.iter().zip(expected.iter()) {
        assert!((a - b).abs() < 1e-6, "got={a}, expected={b}");
    }
}

#[test]
fn test_map_cow_recip_sqrt_f64() {
    let data = [4.0f64, 16.0, 64.0, 256.0];
    let cow = Cow::<f64>::borrow_slice(&data).unwrap();
    let out = cow.map_cow(RecipSqrt);
    let expected = [0.5f64, 0.25, 0.125, 0.0625];
    for (a, b) in out.iter().zip(expected.iter()) {
        assert!((a - b).abs() < 1e-6, "got={a}, expected={b}");
    }
}

#[test]
fn test_vector_masked_load_store_scalar() {
    use hermes_simd::{BitMask, Mask, Scalar, Vector};
    let data = [42.0f32];
    let mask_arr = [true];
    let src = Vector::<f32, Scalar>::splat(0.0);
    unsafe {
        let bm = BitMask::<64>(BitMask::<1>::from_bools(&mask_arr).0);
        let mask = Mask::<f32, Scalar>::from_bitmask(bm);
        let vec = Vector::<f32, Scalar>::masked_load_unaligned(data.as_ptr(), mask, src);
        let mut out = [9.0f32];
        vec.masked_store_unaligned(out.as_mut_ptr(), mask);
        assert_eq!(out[0], 42.0);
    }
}

#[cfg(all(
    any(target_arch = "x86", target_arch = "x86_64"),
    target_feature = "avx2"
))]
#[test]
fn test_vector_masked_load_store_avx2() {
    use hermes_simd::{Avx2, BitMask, Mask, Vector};
    let data = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let mask_arr = [true, false, true, false, true, false, true, false];
    let src = Vector::<f32, Avx2>::splat(0.0);
    unsafe {
        let bm = BitMask::<64>(BitMask::<8>::from_bools(&mask_arr).0);
        let mask = Mask::<f32, Avx2>::from_bitmask(bm);
        let vec = Vector::<f32, Avx2>::masked_load_unaligned(data.as_ptr(), mask, src);
        let mut out = [9.0f32; 8];
        vec.masked_store_unaligned(out.as_mut_ptr(), mask);
        assert_eq!(out, [1.0, 9.0, 3.0, 9.0, 5.0, 9.0, 7.0, 9.0]);
    }
}

#[test]
fn test_popcount_and_reductions() {
    use hermes_simd::{Popcount, Scalar, Vector};
    // 1. Test Popcount via map_unary
    let data = [1.0f32, 2.0, 3.0, 7.0]; // f32 representations:
                                        // 1.0 = 0x3f800000 -> 8 bits set (0011 1111 1000 0000 ...)
                                        // 2.0 = 0x40000000 -> 1 bit set
                                        // 3.0 = 0x40400000 -> 2 bits set
                                        // 7.0 = 0x40e00000 -> 4 bits set
    let v = view(&data);
    let mut out = [0.0f32; 4];
    v.map_unary(Popcount, &mut out).unwrap();
    assert_eq!(out[0], 7.0f32);
    assert_eq!(out[1], 1.0f32);
    assert_eq!(out[2], 2.0f32);
    assert_eq!(out[3], 4.0f32);

    // 2. Test Vector direct methods
    unsafe {
        let vec = Vector::<f32, Scalar>::load_unaligned(data.as_ptr());
        let popped = vec.popcount();
        let mut popped_data = [0.0f32; 4];
        popped.store_unaligned(popped_data.as_mut_ptr());
        assert_eq!(popped_data, [7.0, 1.0, 2.0, 4.0]);

        // Test horizontal reductions
        use hermes_simd::I32;
        let int_data = [I32(0b1100), I32(0b1010), I32(0b1110), I32(0b1111)];
        let int_vec = Vector::<I32, Scalar>::load_unaligned(int_data.as_ptr());

        let and_red = int_vec.horizontal_bitwise_and();
        assert_eq!(and_red, I32(0b1000));

        let or_red = int_vec.horizontal_bitwise_or();
        assert_eq!(or_red, I32(0b1111));

        let xor_red = int_vec.horizontal_bitwise_xor();
        assert_eq!(xor_red, I32(0b0111));
    }
}

#[test]
fn test_popcount_vectorized() {
    use hermes_simd::{target::TargetId, Vector};

    let data = [1.0f32, 2.0, 3.0, 7.0, -1.0, 0.0, 5.5, 12345.67];
    // We can compute the expected popcounts using Scalar (which is checked correct)
    let mut expected = [0.0f32; 8];
    for i in 0..8 {
        expected[i] = data[i].to_bits().count_ones() as f32;
    }

    // Test Avx2 if supported
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        use hermes_simd::Avx2;
        if TargetId::Avx2.is_supported() {
            unsafe {
                let vec = Vector::<f32, Avx2>::load_unaligned(data.as_ptr());
                let popped = vec.popcount();
                let mut out = [0.0f32; 8];
                popped.store_unaligned(out.as_mut_ptr());
                assert_eq!(&out[..8], &expected[..8]);
            }

            // Test f64
            let data_f64 = [1.0f64, 2.0, 3.0, -7.5];
            let mut expected_f64 = [0.0f64; 4];
            for i in 0..4 {
                expected_f64[i] = data_f64[i].to_bits().count_ones() as f64;
            }
            unsafe {
                let vec = Vector::<f64, Avx2>::load_unaligned(data_f64.as_ptr());
                let popped = vec.popcount();
                let mut out = [0.0f64; 4];
                popped.store_unaligned(out.as_mut_ptr());
                assert_eq!(&out[..4], &expected_f64[..4]);
            }
        }
    }

    // Test Avx512 if supported
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        use hermes_simd::Avx512;
        if TargetId::Avx512.is_supported() {
            let mut data_16 = [0.0f32; 16];
            let mut expected_16 = [0.0f32; 16];
            for i in 0..16 {
                data_16[i] = (i as f32) * 123.456 - 7.0;
                expected_16[i] = data_16[i].to_bits().count_ones() as f32;
            }
            unsafe {
                let vec = Vector::<f32, Avx512>::load_unaligned(data_16.as_ptr());
                let popped = vec.popcount();
                let mut out = [0.0f32; 16];
                popped.store_unaligned(out.as_mut_ptr());
                assert_eq!(&out[..16], &expected_16[..16]);
            }

            // Test f64
            let mut data_f64_8 = [0.0f64; 8];
            let mut expected_f64_8 = [0.0f64; 8];
            for i in 0..8 {
                data_f64_8[i] = (i as f64) * 9876.5432 - 19.0;
                expected_f64_8[i] = data_f64_8[i].to_bits().count_ones() as f64;
            }
            unsafe {
                let vec = Vector::<f64, Avx512>::load_unaligned(data_f64_8.as_ptr());
                let popped = vec.popcount();
                let mut out = [0.0f64; 8];
                popped.store_unaligned(out.as_mut_ptr());
                assert_eq!(&out[..8], &expected_f64_8[..8]);
            }
        }
    }

    // Test Neon if supported
    #[cfg(target_arch = "aarch64")]
    {
        use hermes_simd::Neon;
        if TargetId::Neon.is_supported() {
            unsafe {
                let vec = Vector::<f32, Neon>::load_unaligned(data.as_ptr());
                let popped = vec.popcount();
                let mut out = [0.0f32; 4];
                popped.store_unaligned(out.as_mut_ptr());
                assert_eq!(&out[..4], &expected[..4]);
            }

            let data_f64 = [1.0f64, -2.5];
            let mut expected_f64 = [0.0f64; 2];
            for i in 0..2 {
                expected_f64[i] = data_f64[i].to_bits().count_ones() as f64;
            }
            unsafe {
                let vec = Vector::<f64, Neon>::load_unaligned(data_f64.as_ptr());
                let popped = vec.popcount();
                let mut out = [0.0f64; 2];
                popped.store_unaligned(out.as_mut_ptr());
                assert_eq!(&out[..2], &expected_f64[..2]);
            }
        }
    }
}
