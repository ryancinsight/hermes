#![allow(clippy::unnecessary_cast)]
use hermes_simd::*;

macro_rules! test_masked_ops_for_type {
    ($t:ty, $arch:ident, $lanes:expr, $conv:expr) => {
        // --- 1. Test compress ---
        {
            let conv = $conv;
            let test_lengths = [0, $lanes, $lanes * 2, $lanes + 3, $lanes * 3 + 5];
            for &len in &test_lengths {
                let data: Vec<$t> = (0..len).map(&conv).collect();
                let view = SimdView::<$t, $arch, Unaligned>::new(&data).unwrap();

                // Test various mask configurations
                let mask_bools_set = vec![
                    vec![true; $lanes],
                    vec![false; $lanes],
                    (0..$lanes).map(|i| i % 2 == 0).collect::<Vec<_>>(),
                    (0..$lanes).map(|i| i % 3 == 0).collect::<Vec<_>>(),
                ];

                for mask_bools in mask_bools_set {
                    let mut mask_arr = [false; $lanes];
                    mask_arr[..$lanes].copy_from_slice(&mask_bools[..$lanes]);
                    let mask = BitMask::<$lanes>::from_bools(&mask_arr);

                    let mut out = vec![<$t as hermes_simd_core::scalar::NumericElement>::ZERO; len];
                    let written = view.compress(&mask, &mut out).unwrap();

                    // Calculate expected result manually
                    let mut expected = Vec::new();
                    let simd_len = (len / $lanes) * $lanes;
                    for chunk in 0..(len / $lanes) {
                        for i in 0..$lanes {
                            if mask_bools[i] {
                                expected.push(data[chunk * $lanes + i]);
                            }
                        }
                    }
                    for i in simd_len..len {
                        let lane_idx = i - simd_len;
                        if mask_bools[lane_idx] {
                            expected.push(data[i]);
                        }
                    }

                    assert_eq!(written, expected.len(), "Compress written length mismatch for len {}", len);
                    assert_eq!(&out[..written], &expected[..], "Compress content mismatch for len {}", len);
                }
            }
        }

        // --- 2. Test expand ---
        {
            let conv = $conv;
            let test_lengths = [0, $lanes, $lanes * 2, $lanes + 3, $lanes * 3 + 5];
            for &len in &test_lengths {
                let mask_bools_set = vec![
                    vec![true; $lanes],
                    vec![false; $lanes],
                    (0..$lanes).map(|i| i % 2 == 0).collect::<Vec<_>>(),
                    (0..$lanes).map(|i| i % 3 == 0).collect::<Vec<_>>(),
                ];

                for mask_bools in mask_bools_set {
                    let mut mask_arr = [false; $lanes];
                    mask_arr[..$lanes].copy_from_slice(&mask_bools[..$lanes]);
                    let mask = BitMask::<$lanes>::from_bools(&mask_arr);

                    // Compute required source length
                    let simd_len = (len / $lanes) * $lanes;
                    let pop = mask.popcount() as usize;
                    let mut required_src_len = (simd_len / $lanes) * pop;
                    for i in simd_len..len {
                        let lane_idx = i - simd_len;
                        if mask_bools[lane_idx] {
                            required_src_len += 1;
                        }
                    }

                    let src_data: Vec<$t> = (1..=required_src_len).map(&conv).collect();
                    let fill_val = conv(99); // Distinct fill value
                    let fill_data: Vec<$t> = vec![fill_val; len];
                    let mut out = vec![<$t as hermes_simd_core::scalar::NumericElement>::ZERO; len];

                    let src_view = SimdView::<$t, $arch, Unaligned>::new(&src_data).unwrap();
                    let fill_view = SimdView::<$t, $arch, Unaligned>::new(&fill_data).unwrap();

                    src_view.expand(&mask, &fill_view, &mut out).unwrap();

                    // Calculate expected result manually
                    let mut expected = vec![fill_val; len];
                    let mut src_idx = 0;
                    for chunk in 0..(len / $lanes) {
                        for i in 0..$lanes {
                            if mask_bools[i] {
                                expected[chunk * $lanes + i] = src_data[src_idx];
                                src_idx += 1;
                            }
                        }
                    }
                    for i in simd_len..len {
                        let lane_idx = i - simd_len;
                        if mask_bools[lane_idx] {
                            expected[i] = src_data[src_idx];
                            src_idx += 1;
                        }
                    }

                    assert_eq!(out, expected, "Expand content mismatch for len {} and pop {}", len, pop);

                    // Verify that if src is too short, we get LengthMismatch
                    if required_src_len > 0 {
                        let short_src = &src_data[..required_src_len - 1];
                        let short_view = SimdView::<$t, $arch, Unaligned>::new(short_src).unwrap();
                        let err = short_view.expand(&mask, &fill_view, &mut out);
                        assert_eq!(err.unwrap_err(), SimdError::LengthMismatch);
                    }
                }
            }
        }

        // --- 3. Test gather / gather_masked ---
        {
            let conv = $conv;
            let base_data: Vec<$t> = (0..100).map(|x| conv(x)).collect();
            let mut idx_arr = [0i32; $lanes];
            for i in 0..$lanes {
                idx_arr[i] = (i * 3) as i32;
            }

            let indices: <$arch as SimdKernel<$t>>::IndexVector = unsafe {
                core::ptr::read_unaligned(idx_arr.as_ptr() as *const _)
            };

            // Gather
            let vec_res = unsafe { <$arch as SimdKernel<$t>>::gather(base_data.as_ptr(), indices) };
            let mut res_arr = [<$t as hermes_simd_core::scalar::NumericElement>::ZERO; $lanes];
            unsafe { <$arch as SimdKernel<$t>>::store_unaligned(res_arr.as_mut_ptr(), vec_res); }

            for i in 0..$lanes {
                assert_eq!(res_arr[i], base_data[(i * 3) as usize], "Gather mismatch at index {}", i);
            }

            // Gather masked
            let mask_bools = (0..$lanes).map(|i| i % 2 == 0).collect::<Vec<_>>();
            let mut mask_arr = [false; $lanes];
            mask_arr[..$lanes].copy_from_slice(&mask_bools[..$lanes]);
            let mask = BitMask::<$lanes>::from_bools(&mask_arr);
            let native_mask = unsafe { mask.to_native_mask::<$t, $arch>() };

            let fill_val = conv(88);
            let fill_vec = unsafe { <$arch as SimdKernel<$t>>::splat(fill_val) };
            let vec_res_masked = unsafe {
                <$arch as SimdKernel<$t>>::gather_masked(
                    base_data.as_ptr(),
                    indices,
                    native_mask,
                    fill_vec
                )
            };

            let mut res_arr_masked = [<$t as hermes_simd_core::scalar::NumericElement>::ZERO; $lanes];
            unsafe { <$arch as SimdKernel<$t>>::store_unaligned(res_arr_masked.as_mut_ptr(), vec_res_masked); }

            for i in 0..$lanes {
                let expected = if mask_bools[i] { base_data[(i * 3) as usize] } else { fill_val };
                assert_eq!(res_arr_masked[i], expected, "Gather masked mismatch at index {}", i);
            }
        }
    };
}

#[test]
fn test_masked_ops_scalar() {
    test_masked_ops_for_type!(f32, Scalar, 4, |x| x as f32);
    test_masked_ops_for_type!(f64, Scalar, 2, |x| x as f64);
    test_masked_ops_for_type!(half::f16, Scalar, 8, |x| half::f16::from_f32(x as f32));
    test_masked_ops_for_type!(half::bf16, Scalar, 8, |x| half::bf16::from_f32(x as f32));
    test_masked_ops_for_type!(i8, Scalar, 16, |x| x as i8);
    test_masked_ops_for_type!(i16, Scalar, 8, |x| x as i16);
    test_masked_ops_for_type!(i32, Scalar, 4, |x| x as i32);
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[test]
fn test_masked_ops_avx2() {
    if std::is_x86_feature_detected!("avx2") {
        test_masked_ops_for_type!(f32, Avx2, 8, |x| x as f32);
        test_masked_ops_for_type!(f64, Avx2, 4, |x| x as f64);
        test_masked_ops_for_type!(half::f16, Avx2, 16, |x| half::f16::from_f32(x as f32));
        test_masked_ops_for_type!(half::bf16, Avx2, 16, |x| half::bf16::from_f32(x as f32));
        test_masked_ops_for_type!(i8, Avx2, 32, |x| x as i8);
        test_masked_ops_for_type!(i16, Avx2, 16, |x| x as i16);
        test_masked_ops_for_type!(i32, Avx2, 8, |x| x as i32);
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[test]
fn test_masked_ops_avx512() {
    if std::is_x86_feature_detected!("avx512f") {
        test_masked_ops_for_type!(f32, Avx512, 16, |x| x as f32);
        test_masked_ops_for_type!(f64, Avx512, 8, |x| x as f64);
        test_masked_ops_for_type!(half::f16, Avx512, 32, |x| half::f16::from_f32(x as f32));
        test_masked_ops_for_type!(half::bf16, Avx512, 32, |x| half::bf16::from_f32(x as f32));
        test_masked_ops_for_type!(i8, Avx512, 64, |x| x as i8);
        test_masked_ops_for_type!(i16, Avx512, 32, |x| x as i16);
        test_masked_ops_for_type!(i32, Avx512, 16, |x| x as i32);
    }
}

#[cfg(target_arch = "aarch64")]
#[test]
fn test_masked_ops_neon() {
    test_masked_ops_for_type!(f32, Neon, 4, |x| x as f32);
    test_masked_ops_for_type!(f64, Neon, 2, |x| x as f64);
    test_masked_ops_for_type!(half::f16, Neon, 8, |x| half::f16::from_f32(x as f32));
    test_masked_ops_for_type!(half::bf16, Neon, 8, |x| half::bf16::from_f32(x as f32));
    test_masked_ops_for_type!(i8, Neon, 16, |x| x as i8);
    test_masked_ops_for_type!(i16, Neon, 8, |x| x as i16);
    test_masked_ops_for_type!(i32, Neon, 4, |x| x as i32);
}
