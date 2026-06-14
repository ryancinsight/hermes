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
