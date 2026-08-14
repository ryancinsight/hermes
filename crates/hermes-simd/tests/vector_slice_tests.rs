#![expect(
    clippy::float_cmp,
    reason = "These integration tests assert exact manufactured vector values"
)]

use hermes_simd::{Scalar, SimdError, Vector};

#[test]
fn safe_unaligned_load_and_store_preserve_one_vector() {
    let input = [1.0f32, -2.0, 3.5, 4.25];
    let vector = Vector::<f32, Scalar>::load_unaligned_from_slice(&input).unwrap();

    let mut out = [0.0f32; 4];
    vector.store_unaligned_to_slice(&mut out).unwrap();

    assert_eq!(out, input);
}

#[test]
fn safe_slice_load_rejects_short_input() {
    let input = [1.0f32, 2.0, 3.0];

    assert_eq!(
        Vector::<f32, Scalar>::load_unaligned_from_slice(&input),
        Err(SimdError::InsufficientInputLength)
    );
    assert_eq!(
        Vector::<f32, Scalar>::load_aligned_from_slice(&input),
        Err(SimdError::InsufficientInputLength)
    );
}

#[test]
fn safe_slice_store_rejects_short_output() {
    let vector = Vector::<f32, Scalar>::splat(1.0);
    let mut out = [0.0f32; 3];

    assert_eq!(
        vector.store_unaligned_to_slice(&mut out),
        Err(SimdError::InsufficientOutputLength)
    );
    assert_eq!(
        vector.store_aligned_to_slice(&mut out),
        Err(SimdError::InsufficientOutputLength)
    );
}

#[test]
fn safe_aligned_load_and_store_preserve_one_vector() {
    #[repr(align(64))]
    struct Aligned([f32; 4]);

    let input = Aligned([4.0f32, 3.0, 2.0, 1.0]);
    let vector = Vector::<f32, Scalar>::load_aligned_from_slice(&input.0).unwrap();

    let mut out = Aligned([0.0f32; 4]);
    vector.store_aligned_to_slice(&mut out.0).unwrap();

    assert_eq!(out.0, input.0);
}

#[test]
fn safe_aligned_load_and_store_reject_unaligned_slices() {
    let input = [0.0f32; 8];
    let unaligned_input = (0..=4)
        .map(|offset| &input[offset..offset + 4])
        .find(|slice| {
            !(slice.as_ptr() as usize)
                .is_multiple_of(<Scalar as hermes_simd::SimdKernel<f32>>::LANE_COUNT * 4)
        })
        .expect("at least one f32 subslice offset is unaligned to the scalar vector width");

    assert_eq!(
        Vector::<f32, Scalar>::load_aligned_from_slice(unaligned_input),
        Err(SimdError::UnalignedAddress)
    );

    let vector = Vector::<f32, Scalar>::splat(2.0);
    let mut output = [0.0f32; 8];
    let unaligned_offset = (0..=4)
        .find(|&offset| {
            !(output[offset..].as_ptr() as usize)
                .is_multiple_of(<Scalar as hermes_simd::SimdKernel<f32>>::LANE_COUNT * 4)
        })
        .expect("at least one f32 mutable subslice offset is unaligned to the scalar vector width");

    assert_eq!(
        vector.store_aligned_to_slice(&mut output[unaligned_offset..unaligned_offset + 4]),
        Err(SimdError::UnalignedAddress)
    );
    assert_eq!(output, [0.0; 8]);
}

#[test]
fn test_masked_load_store_slice_scalar() {
    use hermes_simd::{BitMask, Mask, Scalar, Vector};

    // Scalar has lane count 1.
    // Let's test with active lane.
    let data = [42.0f32];
    let mask_arr = [true];
    let src = Vector::<f32, Scalar>::splat(0.0);
    unsafe {
        let bm = BitMask::<64>(BitMask::<1>::from_bools(&mask_arr).0);
        let mask = Mask::<f32, Scalar>::from_bitmask(bm);

        // Safe masked load
        let vec = Vector::<f32, Scalar>::masked_load_from_slice(&data, mask, src).unwrap();
        let mut out = [9.0f32];
        vec.masked_store_to_slice(&mut out, mask).unwrap();
        assert_eq!(out[0], 42.0);
    }

    // Let's test with inactive lane.
    let mask_arr_inactive = [false];
    unsafe {
        let bm = BitMask::<64>(BitMask::<1>::from_bools(&mask_arr_inactive).0);
        let mask = Mask::<f32, Scalar>::from_bitmask(bm);

        // Safe masked load (should load from src since mask is false)
        let vec = Vector::<f32, Scalar>::masked_load_from_slice(&data, mask, src).unwrap();
        let mut out = [9.0f32];
        vec.masked_store_to_slice(&mut out, mask).unwrap();
        // Since store mask is false, out[0] should remain unchanged (9.0)
        assert_eq!(out[0], 9.0);

        // Also check loaded value (should be src, which is 0.0)
        let mut out2 = [9.0f32];
        // using active mask to store the loaded value to verify it
        let active_bm = BitMask::<64>(BitMask::<1>::from_bools(&[true]).0);
        let active_mask = Mask::<f32, Scalar>::from_bitmask(active_bm);
        vec.masked_store_to_slice(&mut out2, active_mask).unwrap();
        assert_eq!(out2[0], 0.0);
    }

    // Let's test out of bounds.
    // An empty slice is of length 0. With active mask, this must fail.
    unsafe {
        let bm = BitMask::<64>(BitMask::<1>::from_bools(&[true]).0);
        let mask = Mask::<f32, Scalar>::from_bitmask(bm);
        let res = Vector::<f32, Scalar>::masked_load_from_slice(&[], mask, src);
        assert_eq!(res, Err(SimdError::IndexOutOfBounds));

        let mut out: [f32; 0] = [];
        let res_store = src.masked_store_to_slice(&mut out, mask);
        assert_eq!(res_store, Err(SimdError::IndexOutOfBounds));
    }

    // An empty slice with INACTIVE mask should succeed (index is not out of bounds for inactive lanes).
    unsafe {
        let bm = BitMask::<64>(BitMask::<1>::from_bools(&[false]).0);
        let mask = Mask::<f32, Scalar>::from_bitmask(bm);
        let _res = Vector::<f32, Scalar>::masked_load_from_slice(&[], mask, src).unwrap();

        let mut out: [f32; 0] = [];
        src.masked_store_to_slice(&mut out, mask).unwrap();
    }
}

#[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "std"))]
#[test]
fn test_masked_load_store_slice_avx2() {
    use hermes_simd::{target::TargetId, Avx2, BitMask, Mask, Vector};

    if !TargetId::Avx2.is_supported() {
        return;
    }

    // Avx2 has lane count 8 for f32.
    let data = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let mask_arr = [true, false, true, false, true, false, true, false];
    let src = Vector::<f32, Avx2>::splat(0.0);
    unsafe {
        let bm = BitMask::<64>(BitMask::<8>::from_bools(&mask_arr).0);
        let mask = Mask::<f32, Avx2>::from_bitmask(bm);

        // Safe masked load
        let vec = Vector::<f32, Avx2>::masked_load_from_slice(&data, mask, src).unwrap();
        let mut out = [9.0f32; 8];
        vec.masked_store_to_slice(&mut out, mask).unwrap();
        // Active lanes should be updated, inactive unchanged.
        assert_eq!(out, [1.0, 9.0, 3.0, 9.0, 5.0, 9.0, 7.0, 9.0]);
    }

    // Let's test short slice boundary condition.
    // data has 5 elements, but mask has active lanes only in 0..5 (e.g. indices 0, 2, 4).
    let short_data = [10.0f32, 20.0, 30.0, 40.0, 50.0];
    let mask_arr_short = [true, false, true, false, true, false, false, false];
    unsafe {
        let bm = BitMask::<64>(BitMask::<8>::from_bools(&mask_arr_short).0);
        let mask = Mask::<f32, Avx2>::from_bitmask(bm);

        let vec = Vector::<f32, Avx2>::masked_load_from_slice(&short_data, mask, src).unwrap();
        let mut out = [99.0f32; 5];
        vec.masked_store_to_slice(&mut out, mask).unwrap();
        assert_eq!(out, [10.0, 99.0, 30.0, 99.0, 50.0]);
    }

    // Now test if active lane is out of bounds for the short slice.
    // Short data has 5 elements, but mask activates lane index 6.
    let mask_arr_oob = [true, false, true, false, true, false, true, false];
    unsafe {
        let bm = BitMask::<64>(BitMask::<8>::from_bools(&mask_arr_oob).0);
        let mask = Mask::<f32, Avx2>::from_bitmask(bm);

        let res = Vector::<f32, Avx2>::masked_load_from_slice(&short_data, mask, src);
        assert_eq!(res, Err(SimdError::IndexOutOfBounds));

        let mut out = [99.0f32; 5];
        let res_store = src.masked_store_to_slice(&mut out, mask);
        assert_eq!(res_store, Err(SimdError::IndexOutOfBounds));
    }
}

#[cfg(all(target_arch = "aarch64", feature = "std"))]
#[test]
fn test_masked_load_store_slice_neon() {
    use hermes_simd::{target::TargetId, BitMask, Mask, Neon, Vector};

    if !TargetId::Neon.is_supported() {
        return;
    }

    // Neon has lane count 4 for f32.
    let data = [1.0f32, 2.0, 3.0, 4.0];
    let mask_arr = [true, false, true, false];
    let src = Vector::<f32, Neon>::splat(0.0);
    unsafe {
        let bm = BitMask::<64>(BitMask::<4>::from_bools(&mask_arr).0);
        let mask = Mask::<f32, Neon>::from_bitmask(bm);

        // Safe masked load
        let vec = Vector::<f32, Neon>::masked_load_from_slice(&data, mask, src).unwrap();
        let mut out = [9.0f32; 4];
        vec.masked_store_to_slice(&mut out, mask).unwrap();
        assert_eq!(out, [1.0, 9.0, 3.0, 9.0]);
    }

    // Short slice test (data length 3, active lanes 0 and 2)
    let short_data = [10.0f32, 20.0, 30.0];
    let mask_arr_short = [true, false, true, false];
    unsafe {
        let bm = BitMask::<64>(BitMask::<4>::from_bools(&mask_arr_short).0);
        let mask = Mask::<f32, Neon>::from_bitmask(bm);

        let vec = Vector::<f32, Neon>::masked_load_from_slice(&short_data, mask, src).unwrap();
        let mut out = [99.0f32; 3];
        vec.masked_store_to_slice(&mut out, mask).unwrap();
        assert_eq!(out, [10.0, 99.0, 30.0]);
    }

    // Active lane out of bounds (length 3, active lane 3)
    let mask_arr_oob = [true, false, true, true];
    unsafe {
        let bm = BitMask::<64>(BitMask::<4>::from_bools(&mask_arr_oob).0);
        let mask = Mask::<f32, Neon>::from_bitmask(bm);

        let res = Vector::<f32, Neon>::masked_load_from_slice(&short_data, mask, src);
        assert_eq!(res, Err(SimdError::IndexOutOfBounds));

        let mut out = [99.0f32; 3];
        let res_store = src.masked_store_to_slice(&mut out, mask);
        assert_eq!(res_store, Err(SimdError::IndexOutOfBounds));
    }
}

#[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "std"))]
#[test]
fn safe_avx512_vector_constructors_reject_unsupported_target() {
    use hermes_simd::{target::TargetId, Avx512, Vector};

    if TargetId::Avx512.is_supported() {
        return;
    }

    let data = [1.0f32; 16];
    assert_eq!(
        Vector::<f32, Avx512>::try_zero(),
        Err(SimdError::UnsupportedTarget)
    );
    assert_eq!(
        Vector::<f32, Avx512>::try_splat(1.0),
        Err(SimdError::UnsupportedTarget)
    );
    assert_eq!(
        Vector::<f32, Avx512>::try_from_array(data),
        Err(SimdError::UnsupportedTarget)
    );
    assert_eq!(
        Vector::<f32, Avx512>::load_unaligned_from_slice(&data),
        Err(SimdError::UnsupportedTarget)
    );

    let panic = std::panic::catch_unwind(|| Vector::<f32, Avx512>::splat(1.0));
    assert!(panic.is_err());
}

#[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "std"))]
#[test]
fn test_masked_load_store_slice_avx512() {
    use hermes_simd::{target::TargetId, Avx512, BitMask, Mask, Vector};

    if !TargetId::Avx512.is_supported() {
        return;
    }

    // Avx512 has lane count 16 for f32.
    let mut data = [0.0f32; 16];
    for (i, d) in data.iter_mut().enumerate() {
        *d = (i + 1) as f32;
    }
    let mut mask_arr = [false; 16];
    for (i, m) in mask_arr.iter_mut().enumerate() {
        if i % 2 == 0 {
            *m = true;
        }
    }
    let src = Vector::<f32, Avx512>::splat(0.0);
    unsafe {
        let bm = BitMask::<64>(BitMask::<16>::from_bools(&mask_arr).0);
        let mask = Mask::<f32, Avx512>::from_bitmask(bm);

        // Safe masked load
        let vec = Vector::<f32, Avx512>::masked_load_from_slice(&data, mask, src).unwrap();
        let mut out = [99.0f32; 16];
        vec.masked_store_to_slice(&mut out, mask).unwrap();
        for (i, &o) in out.iter().enumerate() {
            if i % 2 == 0 {
                assert_eq!(o, (i + 1) as f32);
            } else {
                assert_eq!(o, 99.0);
            }
        }
    }
}

#[test]
fn test_widen_i8_simd_and_tails() {
    use hermes_simd::{
        widen_I8_to_I16, widen_I8_to_I32, widen_i8_to_i16, widen_i8_to_i32, I16, I32, I8,
    };

    // Test different lengths to cover SIMD loop, inner SIMD combinations, and scalar tail loop.
    for len in 0..100 {
        let mut src = vec![0i8; len];
        for (j, s) in src.iter_mut().enumerate() {
            *s = (j as i8).wrapping_mul(31).wrapping_add(7);
        }
        let mut dest_i16 = vec![0i16; len];
        let mut dest_i32 = vec![0i32; len];

        widen_i8_to_i16(&src, &mut dest_i16);
        widen_i8_to_i32(&src, &mut dest_i32);

        let expected_i16: Vec<i16> = src.iter().map(|&x| i16::from(x)).collect();
        let expected_i32: Vec<i32> = src.iter().map(|&x| i32::from(x)).collect();

        assert_eq!(dest_i16, expected_i16, "Failed for length i16: {len}");
        assert_eq!(dest_i32, expected_i32, "Failed for length i32: {len}");

        // Also test transparent wrapped types
        let src_wrapped: Vec<I8> = src.iter().map(|&x| I8(x)).collect();
        let mut dest_i16_wrapped = vec![I16(0); len];
        let mut dest_i32_wrapped = vec![I32(0); len];

        widen_I8_to_I16(&src_wrapped, &mut dest_i16_wrapped);
        widen_I8_to_I32(&src_wrapped, &mut dest_i32_wrapped);

        let expected_i16_wrapped: Vec<I16> = src.iter().map(|&x| I16(i16::from(x))).collect();
        let expected_i32_wrapped: Vec<I32> = src.iter().map(|&x| I32(i32::from(x))).collect();

        assert_eq!(
            dest_i16_wrapped, expected_i16_wrapped,
            "Failed for wrapped i16: {len}"
        );
        assert_eq!(
            dest_i32_wrapped, expected_i32_wrapped,
            "Failed for wrapped i32: {len}"
        );
    }
}
