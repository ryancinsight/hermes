use hermes_simd::*;
use hermes_simd_core::scalar::NumericElement;

// Helper macro to run f32 tests for a given architecture
macro_rules! test_f32_ops_for_arch {
    ($arch:ident, $lanes:expr) => {
        let mut buf_a = [0.0f32; $lanes];
        let mut buf_b = [0.0f32; $lanes];
        for i in 0..$lanes {
            buf_a[i] = (i as f32) - ($lanes as f32 / 2.0);
            buf_b[i] = 2.0f32;
        }

        let a = unsafe { Vector::<f32, $arch>::load_unaligned(buf_a.as_ptr()) };
        let b = unsafe { Vector::<f32, $arch>::load_unaligned(buf_b.as_ptr()) };

        // 1. Division
        let c = a / b;
        let mut buf_res = [0.0f32; $lanes];
        unsafe { c.store_unaligned(buf_res.as_mut_ptr()); }
        for i in 0..$lanes {
            assert!((buf_res[i] - (buf_a[i] / 2.0)).abs() < 1e-5, "Div failed at lane {} of {}", i, stringify!($arch));
        }

        // 2. Bitwise operators
        // Bitwise and/or/xor operate on f32 float representations. Let's splat bit patterns.
        let pattern_a = f32::from_bits(0x5555_5555);
        let pattern_b = f32::from_bits(0x3333_3333);
        let vec_pat_a = Vector::<f32, $arch>::splat(pattern_a);
        let vec_pat_b = Vector::<f32, $arch>::splat(pattern_b);

        let and_res = vec_pat_a & vec_pat_b;
        unsafe { and_res.store_unaligned(buf_res.as_mut_ptr()); }
        for i in 0..$lanes {
            let res_bits = buf_res[i].to_bits();
            assert_eq!(res_bits, 0x5555_5555 & 0x3333_3333, "BitAnd failed at lane {} of {}", i, stringify!($arch));
        }

        let or_res = vec_pat_a | vec_pat_b;
        unsafe { or_res.store_unaligned(buf_res.as_mut_ptr()); }
        for i in 0..$lanes {
            let res_bits = buf_res[i].to_bits();
            assert_eq!(res_bits, 0x5555_5555 | 0x3333_3333, "BitOr failed at lane {} of {}", i, stringify!($arch));
        }

        let xor_res = vec_pat_a ^ vec_pat_b;
        unsafe { xor_res.store_unaligned(buf_res.as_mut_ptr()); }
        for i in 0..$lanes {
            let res_bits = buf_res[i].to_bits();
            assert_eq!(res_bits, 0x5555_5555 ^ 0x3333_3333, "BitXor failed at lane {} of {}", i, stringify!($arch));
        }

        // 3. Math (abs, min, max, sqrt)
        let abs_res = a.abs();
        unsafe { abs_res.store_unaligned(buf_res.as_mut_ptr()); }
        for i in 0..$lanes {
            assert!((buf_res[i] - buf_a[i].abs()).abs() < 1e-5, "Abs failed at lane {} of {}", i, stringify!($arch));
        }

        let min_res = a.min(b);
        unsafe { min_res.store_unaligned(buf_res.as_mut_ptr()); }
        for i in 0..$lanes {
            let expected = if buf_a[i] < buf_b[i] { buf_a[i] } else { buf_b[i] };
            assert!((buf_res[i] - expected).abs() < 1e-5, "Min failed at lane {} of {}", i, stringify!($arch));
        }

        let max_res = a.max(b);
        unsafe { max_res.store_unaligned(buf_res.as_mut_ptr()); }
        for i in 0..$lanes {
            let expected = if buf_a[i] > buf_b[i] { buf_a[i] } else { buf_b[i] };
            assert!((buf_res[i] - expected).abs() < 1e-5, "Max failed at lane {} of {}", i, stringify!($arch));
        }

        let non_neg = abs_res;
        let sqrt_res = non_neg.sqrt();
        unsafe { sqrt_res.store_unaligned(buf_res.as_mut_ptr()); }
        for i in 0..$lanes {
            let expected = buf_a[i].abs().sqrt();
            assert!((buf_res[i] - expected).abs() < 1e-5, "Sqrt failed at lane {} of {}", i, stringify!($arch));
        }

        // 4. Comparisons (cmp_eq, cmp_ne, cmp_lt, cmp_le, cmp_gt, cmp_ge)
        // Check comparison returns NaN/all-ones mask on true, 0 on false
        let all_ones_u32 = 0xFFFF_FFFFu32;

        let eq_res = a.cmp_eq(b);
        unsafe { eq_res.store_unaligned(buf_res.as_mut_ptr()); }
        for i in 0..$lanes {
            let actual_bits = buf_res[i].to_bits();
            let expected_bits = if buf_a[i] == buf_b[i] { all_ones_u32 } else { 0 };
            assert_eq!(actual_bits, expected_bits, "CmpEq failed at lane {} of {}", i, stringify!($arch));
        }

        let ne_res = a.cmp_ne(b);
        unsafe { ne_res.store_unaligned(buf_res.as_mut_ptr()); }
        for i in 0..$lanes {
            let actual_bits = buf_res[i].to_bits();
            let expected_bits = if buf_a[i] != buf_b[i] { all_ones_u32 } else { 0 };
            assert_eq!(actual_bits, expected_bits, "CmpNe failed at lane {} of {}", i, stringify!($arch));
        }

        let lt_res = a.cmp_lt(b);
        unsafe { lt_res.store_unaligned(buf_res.as_mut_ptr()); }
        for i in 0..$lanes {
            let actual_bits = buf_res[i].to_bits();
            let expected_bits = if buf_a[i] < buf_b[i] { all_ones_u32 } else { 0 };
            assert_eq!(actual_bits, expected_bits, "CmpLt failed at lane {} of {}", i, stringify!($arch));
        }

        let le_res = a.cmp_le(b);
        unsafe { le_res.store_unaligned(buf_res.as_mut_ptr()); }
        for i in 0..$lanes {
            let actual_bits = buf_res[i].to_bits();
            let expected_bits = if buf_a[i] <= buf_b[i] { all_ones_u32 } else { 0 };
            assert_eq!(actual_bits, expected_bits, "CmpLe failed at lane {} of {}", i, stringify!($arch));
        }

        let gt_res = a.cmp_gt(b);
        unsafe { gt_res.store_unaligned(buf_res.as_mut_ptr()); }
        for i in 0..$lanes {
            let actual_bits = buf_res[i].to_bits();
            let expected_bits = if buf_a[i] > buf_b[i] { all_ones_u32 } else { 0 };
            assert_eq!(actual_bits, expected_bits, "CmpGt failed at lane {} of {}", i, stringify!($arch));
        }

        let ge_res = a.cmp_ge(b);
        unsafe { ge_res.store_unaligned(buf_res.as_mut_ptr()); }
        for i in 0..$lanes {
            let actual_bits = buf_res[i].to_bits();
            let expected_bits = if buf_a[i] >= buf_b[i] { all_ones_u32 } else { 0 };
            assert_eq!(actual_bits, expected_bits, "CmpGe failed at lane {} of {}", i, stringify!($arch));
        }

        // 5. Blend
        // Blend true_val and false_val using gt_res as the mask
        let true_val = Vector::<f32, $arch>::splat(100.0);
        let false_val = Vector::<f32, $arch>::splat(-100.0);
        let blend_res = gt_res.blend(true_val, false_val);
        unsafe { blend_res.store_unaligned(buf_res.as_mut_ptr()); }
        for i in 0..$lanes {
            let expected = if buf_a[i] > buf_b[i] { 100.0f32 } else { -100.0f32 };
            assert_eq!(buf_res[i], expected, "Blend failed at lane {} of {}", i, stringify!($arch));
        }
    };
}

// Helper macro to run f64 tests for a given architecture
macro_rules! test_f64_ops_for_arch {
    ($arch:ident, $lanes:expr) => {
        let mut buf_a = [0.0f64; $lanes];
        let mut buf_b = [0.0f64; $lanes];
        for i in 0..$lanes {
            buf_a[i] = (i as f64) - ($lanes as f64 / 2.0);
            buf_b[i] = 2.0f64;
        }

        let a = unsafe { Vector::<f64, $arch>::load_unaligned(buf_a.as_ptr()) };
        let b = unsafe { Vector::<f64, $arch>::load_unaligned(buf_b.as_ptr()) };

        // 1. Division
        let c = a / b;
        let mut buf_res = [0.0f64; $lanes];
        unsafe { c.store_unaligned(buf_res.as_mut_ptr()); }
        for i in 0..$lanes {
            assert!((buf_res[i] - (buf_a[i] / 2.0)).abs() < 1e-12, "f64 Div failed at lane {} of {}", i, stringify!($arch));
        }

        // 2. Bitwise operators
        let pattern_a = f64::from_bits(0x5555_5555_5555_5555);
        let pattern_b = f64::from_bits(0x3333_3333_3333_3333);
        let vec_pat_a = Vector::<f64, $arch>::splat(pattern_a);
        let vec_pat_b = Vector::<f64, $arch>::splat(pattern_b);

        let and_res = vec_pat_a & vec_pat_b;
        unsafe { and_res.store_unaligned(buf_res.as_mut_ptr()); }
        for i in 0..$lanes {
            let res_bits = buf_res[i].to_bits();
            assert_eq!(res_bits, 0x5555_5555_5555_5555 & 0x3333_3333_3333_3333, "f64 BitAnd failed at lane {} of {}", i, stringify!($arch));
        }

        let or_res = vec_pat_a | vec_pat_b;
        unsafe { or_res.store_unaligned(buf_res.as_mut_ptr()); }
        for i in 0..$lanes {
            let res_bits = buf_res[i].to_bits();
            assert_eq!(res_bits, 0x5555_5555_5555_5555 | 0x3333_3333_3333_3333, "f64 BitOr failed at lane {} of {}", i, stringify!($arch));
        }

        let xor_res = vec_pat_a ^ vec_pat_b;
        unsafe { xor_res.store_unaligned(buf_res.as_mut_ptr()); }
        for i in 0..$lanes {
            let res_bits = buf_res[i].to_bits();
            assert_eq!(res_bits, 0x5555_5555_5555_5555 ^ 0x3333_3333_3333_3333, "f64 BitXor failed at lane {} of {}", i, stringify!($arch));
        }

        // 3. Math
        let abs_res = a.abs();
        unsafe { abs_res.store_unaligned(buf_res.as_mut_ptr()); }
        for i in 0..$lanes {
            assert!((buf_res[i] - buf_a[i].abs()).abs() < 1e-12, "f64 Abs failed at lane {} of {}", i, stringify!($arch));
        }

        let min_res = a.min(b);
        unsafe { min_res.store_unaligned(buf_res.as_mut_ptr()); }
        for i in 0..$lanes {
            let expected = if buf_a[i] < buf_b[i] { buf_a[i] } else { buf_b[i] };
            assert!((buf_res[i] - expected).abs() < 1e-12, "f64 Min failed at lane {} of {}", i, stringify!($arch));
        }

        let max_res = a.max(b);
        unsafe { max_res.store_unaligned(buf_res.as_mut_ptr()); }
        for i in 0..$lanes {
            let expected = if buf_a[i] > buf_b[i] { buf_a[i] } else { buf_b[i] };
            assert!((buf_res[i] - expected).abs() < 1e-12, "f64 Max failed at lane {} of {}", i, stringify!($arch));
        }

        let non_neg = abs_res;
        let sqrt_res = non_neg.sqrt();
        unsafe { sqrt_res.store_unaligned(buf_res.as_mut_ptr()); }
        for i in 0..$lanes {
            let expected = buf_a[i].abs().sqrt();
            assert!((buf_res[i] - expected).abs() < 1e-12, "f64 Sqrt failed at lane {} of {}", i, stringify!($arch));
        }

        // 4. Comparisons
        let all_ones_u64 = 0xFFFF_FFFF_FFFF_FFFFu64;

        let eq_res = a.cmp_eq(b);
        unsafe { eq_res.store_unaligned(buf_res.as_mut_ptr()); }
        for i in 0..$lanes {
            let actual_bits = buf_res[i].to_bits();
            let expected_bits = if buf_a[i] == buf_b[i] { all_ones_u64 } else { 0 };
            assert_eq!(actual_bits, expected_bits, "f64 CmpEq failed at lane {} of {}", i, stringify!($arch));
        }

        let ne_res = a.cmp_ne(b);
        unsafe { ne_res.store_unaligned(buf_res.as_mut_ptr()); }
        for i in 0..$lanes {
            let actual_bits = buf_res[i].to_bits();
            let expected_bits = if buf_a[i] != buf_b[i] { all_ones_u64 } else { 0 };
            assert_eq!(actual_bits, expected_bits, "f64 CmpNe failed at lane {} of {}", i, stringify!($arch));
        }

        let lt_res = a.cmp_lt(b);
        unsafe { lt_res.store_unaligned(buf_res.as_mut_ptr()); }
        for i in 0..$lanes {
            let actual_bits = buf_res[i].to_bits();
            let expected_bits = if buf_a[i] < buf_b[i] { all_ones_u64 } else { 0 };
            assert_eq!(actual_bits, expected_bits, "f64 CmpLt failed at lane {} of {}", i, stringify!($arch));
        }

        let le_res = a.cmp_le(b);
        unsafe { le_res.store_unaligned(buf_res.as_mut_ptr()); }
        for i in 0..$lanes {
            let actual_bits = buf_res[i].to_bits();
            let expected_bits = if buf_a[i] <= buf_b[i] { all_ones_u64 } else { 0 };
            assert_eq!(actual_bits, expected_bits, "f64 CmpLe failed at lane {} of {}", i, stringify!($arch));
        }

        let gt_res = a.cmp_gt(b);
        unsafe { gt_res.store_unaligned(buf_res.as_mut_ptr()); }
        for i in 0..$lanes {
            let actual_bits = buf_res[i].to_bits();
            let expected_bits = if buf_a[i] > buf_b[i] { all_ones_u64 } else { 0 };
            assert_eq!(actual_bits, expected_bits, "f64 CmpGt failed at lane {} of {}", i, stringify!($arch));
        }

        let ge_res = a.cmp_ge(b);
        unsafe { ge_res.store_unaligned(buf_res.as_mut_ptr()); }
        for i in 0..$lanes {
            let actual_bits = buf_res[i].to_bits();
            let expected_bits = if buf_a[i] >= buf_b[i] { all_ones_u64 } else { 0 };
            assert_eq!(actual_bits, expected_bits, "f64 CmpGe failed at lane {} of {}", i, stringify!($arch));
        }

        // 5. Blend
        let true_val = Vector::<f64, $arch>::splat(100.0);
        let false_val = Vector::<f64, $arch>::splat(-100.0);
        let blend_res = gt_res.blend(true_val, false_val);
        unsafe { blend_res.store_unaligned(buf_res.as_mut_ptr()); }
        for i in 0..$lanes {
            let expected = if buf_a[i] > buf_b[i] { 100.0f64 } else { -100.0f64 };
            assert_eq!(buf_res[i], expected, "f64 Blend failed at lane {} of {}", i, stringify!($arch));
        }
    };
}

// Helper macro to run generic tests for any Scalar type on a given architecture
macro_rules! test_numeric_ops_for_arch {
    ($t:ty, $arch:ident, $lanes:expr) => {
        let mut buf_a = [<$t as hermes_simd_core::scalar::NumericElement>::ZERO; $lanes];
        let mut buf_b = [<$t as hermes_simd_core::scalar::NumericElement>::ZERO; $lanes];
        for i in 0..$lanes {
            buf_a[i] = if i % 2 == 0 { <$t as hermes_simd_core::scalar::NumericElement>::ONE + <$t as hermes_simd_core::scalar::NumericElement>::ONE } else { <$t as hermes_simd_core::scalar::NumericElement>::ONE };
            buf_b[i] = <$t as hermes_simd_core::scalar::NumericElement>::ONE + <$t as hermes_simd_core::scalar::NumericElement>::ONE;
        }

        let a = unsafe { Vector::<$t, $arch>::load_unaligned(buf_a.as_ptr()) };
        let b = unsafe { Vector::<$t, $arch>::load_unaligned(buf_b.as_ptr()) };

        // 1. Arithmetic
        let c_add = a + b;
        let mut buf_res = [<$t as hermes_simd_core::scalar::NumericElement>::ZERO; $lanes];
        unsafe { c_add.store_unaligned(buf_res.as_mut_ptr()); }
        for i in 0..$lanes {
            assert_eq!(buf_res[i], buf_a[i] + buf_b[i], "Add failed for {} at lane {} on {}", stringify!($t), i, stringify!($arch));
        }

        let c_sub = a - b;
        unsafe { c_sub.store_unaligned(buf_res.as_mut_ptr()); }
        for i in 0..$lanes {
            assert_eq!(buf_res[i], buf_a[i] - buf_b[i], "Sub failed for {} at lane {} on {}", stringify!($t), i, stringify!($arch));
        }

        let c_mul = a * b;
        unsafe { c_mul.store_unaligned(buf_res.as_mut_ptr()); }
        for i in 0..$lanes {
            assert_eq!(buf_res[i], buf_a[i] * buf_b[i], "Mul failed for {} at lane {} on {}", stringify!($t), i, stringify!($arch));
        }

        // 2. Bitwise operators
        let and_res = a & b;
        unsafe { and_res.store_unaligned(buf_res.as_mut_ptr()); }
        for i in 0..$lanes {
            assert_eq!(buf_res[i], buf_a[i].bitand(buf_b[i]), "BitAnd failed for {} at lane {} on {}", stringify!($t), i, stringify!($arch));
        }

        let or_res = a | b;
        unsafe { or_res.store_unaligned(buf_res.as_mut_ptr()); }
        for i in 0..$lanes {
            assert_eq!(buf_res[i], buf_a[i].bitor(buf_b[i]), "BitOr failed for {} at lane {} on {}", stringify!($t), i, stringify!($arch));
        }

        let xor_res = a ^ b;
        unsafe { xor_res.store_unaligned(buf_res.as_mut_ptr()); }
        for i in 0..$lanes {
            assert_eq!(buf_res[i], buf_a[i].bitxor(buf_b[i]), "BitXor failed for {} at lane {} on {}", stringify!($t), i, stringify!($arch));
        }

        // 3. Comparisons
        let eq_res = a.cmp_eq(b);
        unsafe { eq_res.store_unaligned(buf_res.as_mut_ptr()); }
        for i in 0..$lanes {
            let expected = if buf_a[i] == buf_b[i] { <$t as hermes_simd_core::scalar::NumericElement>::ALL_ONES } else { <$t as hermes_simd_core::scalar::NumericElement>::ZERO };
            let actual = buf_res[i];
            let is_ok = (actual.is_nan() && expected.is_nan()) || actual == expected;
            assert!(is_ok, "CmpEq failed for {} at lane {} on {} (actual: {:?}, expected: {:?})", stringify!($t), i, stringify!($arch), actual, expected);
        }

        let lt_res = a.cmp_lt(b);
        unsafe { lt_res.store_unaligned(buf_res.as_mut_ptr()); }
        for i in 0..$lanes {
            let expected = if buf_a[i] < buf_b[i] { <$t as hermes_simd_core::scalar::NumericElement>::ALL_ONES } else { <$t as hermes_simd_core::scalar::NumericElement>::ZERO };
            let actual = buf_res[i];
            let is_ok = (actual.is_nan() && expected.is_nan()) || actual == expected;
            assert!(is_ok, "CmpLt failed for {} at lane {} on {} (actual: {:?}, expected: {:?})", stringify!($t), i, stringify!($arch), actual, expected);
        }

        // 4. Blend
        let true_val = Vector::<$t, $arch>::splat(<$t as hermes_simd_core::scalar::NumericElement>::ONE);
        let false_val = Vector::<$t, $arch>::zero();
        let blend_res = eq_res.blend(true_val, false_val);
        unsafe { blend_res.store_unaligned(buf_res.as_mut_ptr()); }
        for i in 0..$lanes {
            let expected = if buf_a[i] == buf_b[i] { <$t as hermes_simd_core::scalar::NumericElement>::ONE } else { <$t as hermes_simd_core::scalar::NumericElement>::ZERO };
            let actual = buf_res[i];
            let is_ok = (actual.is_nan() && expected.is_nan()) || actual == expected;
            assert!(is_ok, "Blend failed for {} at lane {} on {} (actual: {:?}, expected: {:?})", stringify!($t), i, stringify!($arch), actual, expected);
        }
    };
}

#[test]
fn test_vector_ops_scalar() {
    test_f32_ops_for_arch!(Scalar, 4);
    test_f64_ops_for_arch!(Scalar, 2);
    test_numeric_ops_for_arch!(half::bf16, Scalar, 8);
    test_numeric_ops_for_arch!(i8, Scalar, 16);
    test_numeric_ops_for_arch!(i16, Scalar, 8);
    test_numeric_ops_for_arch!(i32, Scalar, 4);
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[test]
fn test_vector_ops_avx2() {
    if std::is_x86_feature_detected!("avx2") {
        test_f32_ops_for_arch!(Avx2, 8);
        test_f64_ops_for_arch!(Avx2, 4);
        test_numeric_ops_for_arch!(half::bf16, Avx2, 16);
        test_numeric_ops_for_arch!(i8, Avx2, 32);
        test_numeric_ops_for_arch!(i16, Avx2, 16);
        test_numeric_ops_for_arch!(i32, Avx2, 8);
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[test]
fn test_vector_ops_avx512() {
    if std::is_x86_feature_detected!("avx512f") {
        test_f32_ops_for_arch!(Avx512, 16);
        test_f64_ops_for_arch!(Avx512, 8);
        test_numeric_ops_for_arch!(half::bf16, Avx512, 32);
        test_numeric_ops_for_arch!(i8, Avx512, 64);
        test_numeric_ops_for_arch!(i16, Avx512, 32);
        test_numeric_ops_for_arch!(i32, Avx512, 16);
    }
}

#[cfg(target_arch = "aarch64")]
#[test]
fn test_vector_ops_neon() {
    test_f32_ops_for_arch!(Neon, 4);
    test_f64_ops_for_arch!(Neon, 2);
    test_numeric_ops_for_arch!(half::bf16, Neon, 8);
    test_numeric_ops_for_arch!(i8, Neon, 16);
    test_numeric_ops_for_arch!(i16, Neon, 8);
    test_numeric_ops_for_arch!(i32, Neon, 4);
}

#[test]
fn test_new_vector_features() {
    // 1. Array roundtrip via from_array and to_array (using Scalar architecture)
    let arr = [1.0f32, 2.0, 3.0, 4.0];
    let vec = Vector::<f32, Scalar>::from_array(arr);
    let arr_back = vec.to_array();
    assert_eq!(arr, arr_back);

    // 2. Extract and Insert
    assert_eq!(vec.extract::<0>(), 1.0f32);
    assert_eq!(vec.extract::<1>(), 2.0f32);
    assert_eq!(vec.extract::<2>(), 3.0f32);
    assert_eq!(vec.extract::<3>(), 4.0f32);

    let vec_modified = vec.insert::<2>(10.0f32);
    assert_eq!(vec_modified.extract::<2>(), 10.0f32);
    assert_eq!(vec_modified.extract::<0>(), 1.0f32); // others unchanged

    // 3. Casting (f32 to i32, same lane count of 4)
    let vec_cast = vec.cast::<i32>();
    assert_eq!(vec_cast.to_array(), [1, 2, 3, 4]);

    // 4. Mask operations (reductions: any, all, none; select; bitwise)
    let a = Vector::<f32, Scalar>::from_array([1.0f32, 2.0, 3.0, 4.0]);
    let b = Vector::<f32, Scalar>::from_array([1.0f32, 5.0, 3.0, 6.0]);

    let eq_mask = a.cmp_eq_mask(b);
    assert_eq!(unsafe { eq_mask.to_bitmask().0 }, 0b0101); // lane 0 and 2 are equal (1.0 == 1.0, 3.0 == 3.0)

    assert!(eq_mask.any());
    assert!(!eq_mask.all());
    assert!(!eq_mask.none());

    let true_val = Vector::<f32, Scalar>::splat(100.0);
    let false_val = Vector::<f32, Scalar>::splat(-100.0);
    let blend_res = eq_mask.select(true_val, false_val);
    assert_eq!(blend_res.to_array(), [100.0f32, -100.0, 100.0, -100.0]);

    // Bitwise mask ops
    let ne_mask = a.cmp_ne_mask(b); // lane 1 and 3 are unequal (0b1010)
    assert_eq!(unsafe { ne_mask.to_bitmask().0 }, 0b1010);

    let and_mask = eq_mask & ne_mask;
    assert!(and_mask.none()); // mutually exclusive

    let or_mask = eq_mask | ne_mask;
    assert!(or_mask.all()); // covers all lanes

    let not_mask = !eq_mask;
    assert_eq!(unsafe { not_mask.to_bitmask().0 }, 0b1010);

    // 5. Vector-View Integration (from_view_chunk / store_to_view_chunk)
    let view_data = [10.0f32, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0];
    let view_a = SimdView::<f32, Scalar, Unaligned>::new(&view_data).unwrap();
    let loaded_v0 = Vector::<f32, Scalar>::from_view_chunk(&view_a, 0);
    let loaded_v1 = Vector::<f32, Scalar>::from_view_chunk(&view_a, 1);
    assert_eq!(loaded_v0.to_array(), [10.0, 20.0, 30.0, 40.0]);
    assert_eq!(loaded_v1.to_array(), [50.0, 60.0, 70.0, 80.0]);

    let mut out_buf = [0.0f32; 8];
    let mut view_out = SimdView::<f32, Scalar, Unaligned, Unmasked, &mut [f32]>::new_mut(&mut out_buf).unwrap();
    loaded_v0.store_to_view_chunk(&mut view_out, 1);
    loaded_v1.store_to_view_chunk(&mut view_out, 0);
    assert_eq!(out_buf, [50.0, 60.0, 70.0, 80.0, 10.0, 20.0, 30.0, 40.0]);

    // 6. SimdCowExt::transform_vectors
    let mut cow = SimdCow::<f32, Scalar, Unaligned>::Owned(AlignedVec::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]));
    cow.transform_vectors(|v| v * Vector::splat(2.0));
    assert_eq!(&*cow, &[2.0, 4.0, 6.0, 8.0, 10.0, 12.0]);
}

#[test]
fn test_new_wrapper_types_simd_ops() {
    // F32
    {
        let data = [F32(1.0), F32(2.0), F32(3.0), F32(4.0)];
        assert_eq!(sum(&data), F32(10.0));
        let a = [F32(1.0), F32(2.0)];
        let b = [F32(3.0), F32(4.0)];
        assert_eq!(dot(&a, &b).unwrap(), F32(11.0));
    }
    // F16
    {
        let data = [F16::from_f32(1.0), F16::from_f32(2.0), F16::from_f32(3.0)];
        assert_eq!(sum(&data), F16::from_f32(6.0));
    }
    // F64
    {
        let data = [F64(1.0), F64(2.0), F64(3.0)];
        assert_eq!(sum(&data), F64(6.0));
    }
    // Bf16
    {
        let data = [Bf16::from_f32(1.0), Bf16::from_f32(2.0), Bf16::from_f32(3.0)];
        assert_eq!(sum(&data), Bf16::from_f32(6.0));
    }
    // Bf8
    {
        let data = [Bf8::from_f32(1.0), Bf8::from_f32(2.0), Bf8::from_f32(3.0)];
        assert_eq!(sum(&data), Bf8::from_f32(6.0));
    }
    // Bf4
    {
        let data = [Bf4::from_f32(1.0), Bf4::from_f32(2.0), Bf4::from_f32(3.0)];
        assert_eq!(sum(&data), Bf4::from_f32(6.0));
    }
    // F8
    {
        let data = [F8::from_f32(1.0), F8::from_f32(2.0), F8::from_f32(3.0)];
        assert_eq!(sum(&data), F8::from_f32(6.0));
    }
    // F4
    {
        let data = [F4::from_f32(1.0), F4::from_f32(2.0), F4::from_f32(3.0)];
        assert_eq!(sum(&data), F4::from_f32(6.0));
    }
    // I8
    {
        let data = [I8(1), I8(2), I8(3)];
        assert_eq!(sum(&data), I8(6));
    }
    // I16
    {
        let data = [I16(1), I16(2), I16(3)];
        assert_eq!(sum(&data), I16(6));
    }
    // I32
    {
        let data = [I32(1), I32(2), I32(3)];
        assert_eq!(sum(&data), I32(6));
    }
}

#[test]
fn test_low_precision_unpacking() {
    use hermes_numeric::{unpack_bf8_to_bf16, unpack_bf4_to_bf16, unpack_bf4_to_bf16_packed};
    
    // Bf8 to Bf16
    let bf8_inputs = [
        Bf8::from_f32(0.0),
        Bf8::from_f32(1.0),
        Bf8::from_f32(-2.0),
        Bf8::from_f32(1.75),
    ];
    let mut bf16_outputs = [Bf16::from_f32(0.0); 4];
    unpack_bf8_to_bf16(&bf8_inputs, &mut bf16_outputs);
    
    // Check values
    assert_eq!(bf16_outputs[0].to_f32(), 0.0);
    assert_eq!(bf16_outputs[1].to_f32(), 1.0);
    assert_eq!(bf16_outputs[2].to_f32(), -2.0);
    assert_eq!(bf16_outputs[3].to_f32(), 1.75);
    
    // Bf4 to Bf16
    let bf4_inputs = [
        Bf4::from_f32(0.0),
        Bf4::from_f32(1.0),
        Bf4::from_f32(-1.0),
        Bf4::from_f32(1.5),
    ];
    let mut bf16_outputs_bf4 = [Bf16::from_f32(0.0); 4];
    unpack_bf4_to_bf16(&bf4_inputs, &mut bf16_outputs_bf4);
    
    assert_eq!(bf16_outputs_bf4[0].to_f32(), 0.0);
    assert_eq!(bf16_outputs_bf4[1].to_f32(), 1.0);
    assert_eq!(bf16_outputs_bf4[2].to_f32(), -1.0);
    assert_eq!(bf16_outputs_bf4[3].to_f32(), 1.5);

    // Bf4 Packed
    let mut packed_bytes = [0u8; 2];
    packed_bytes[0] = Bf4::pack_pair(Bf4::from_f32(1.0), Bf4::from_f32(-1.0));
    packed_bytes[1] = Bf4::pack_pair(Bf4::from_f32(1.5), Bf4::from_f32(0.0));
    
    let mut unpacked_bf16_pairs = [Bf16::from_f32(0.0); 4];
    unpack_bf4_to_bf16_packed(&packed_bytes, &mut unpacked_bf16_pairs);
    
    assert_eq!(unpacked_bf16_pairs[0].to_f32(), 1.0);
    assert_eq!(unpacked_bf16_pairs[1].to_f32(), -1.0);
    assert_eq!(unpacked_bf16_pairs[2].to_f32(), 1.5);
    assert_eq!(unpacked_bf16_pairs[3].to_f32(), 0.0);

    // F4 Unpacked
    use hermes_numeric::{F4, F32, unpack_f4_to_f32, unpack_f4_to_f32_packed};
    let f4_inputs = [
        F4::from_f32(0.0),
        F4::from_f32(1.0),
        F4::from_f32(-1.0),
        F4::from_f32(2.0),
    ];
    let mut f32_outputs = [F32(0.0); 4];
    unpack_f4_to_f32(&f4_inputs, &mut f32_outputs);
    assert_eq!(f32_outputs[0].0, 0.0);
    assert_eq!(f32_outputs[1].0, 1.0);
    assert_eq!(f32_outputs[2].0, -1.0);
    assert_eq!(f32_outputs[3].0, 2.0);

    // F4 Packed
    let mut packed_bytes_f4 = [0u8; 2];
    packed_bytes_f4[0] = F4::pack_pair(F4::from_f32(1.0), F4::from_f32(-1.0));
    packed_bytes_f4[1] = F4::pack_pair(F4::from_f32(2.0), F4::from_f32(0.0));
    
    let mut unpacked_f32_pairs = [F32(0.0); 4];
    unpack_f4_to_f32_packed(&packed_bytes_f4, &mut unpacked_f32_pairs);
    assert_eq!(unpacked_f32_pairs[0].0, 1.0);
    assert_eq!(unpacked_f32_pairs[1].0, -1.0);
    assert_eq!(unpacked_f32_pairs[2].0, 2.0);
    assert_eq!(unpacked_f32_pairs[3].0, 0.0);
}

#[test]
fn test_numa_topology_and_allocation() {
    use hermes_simd_core::numa::{NumaTopologyService, verify_numa_locality};
    use hermes_simd_core::AlignedVec;
    use hermes_simd_core::align::Unaligned;

    let cpu = NumaTopologyService::current_cpu();
    println!("Current CPU: {:?}", cpu);

    let node = NumaTopologyService::current_node();
    println!("Current NUMA Node: {:?}", node);

    let total = NumaTopologyService::total_nodes();
    println!("Total NUMA Nodes: {}", total);
    assert!(total >= 1);

    // Allocate on node 0
    let mut vec: AlignedVec<f32, Unaligned> = AlignedVec::with_capacity_numa(1000, 0);
    vec.push(1.0);
    vec.push(2.0);
    assert_eq!(vec[0], 1.0);
    assert_eq!(vec[1], 2.0);

    // Verify newly exposed numa functions
    use hermes_simd_core::numa::{numa_node_count, numa_node_distance};
    let count = numa_node_count();
    assert_eq!(count, total);
    
    // Self distance is always 10
    assert_eq!(numa_node_distance(0, 0), 10);
    // Remote distance is always 20 (or platform specific positive value)
    assert_eq!(numa_node_distance(0, 1), 20);

    // Verify locality check runs without crashing
    let is_local = verify_numa_locality(vec.as_ptr() as *const u8, 8, 0);
    println!("Is local to node 0: {}", is_local);
}

#[test]
fn test_packed_bf4_slice() {
    use hermes_numeric::{Bf4, Bf16, PackedBf4Slice, PackedBf4SliceMut};

    let mut raw_bytes = [0u8; 4];
    {
        let mut slice_mut = PackedBf4SliceMut::new(&mut raw_bytes, 7).unwrap();
        assert_eq!(slice_mut.len(), 7);
        assert!(!slice_mut.is_empty());

        slice_mut.set(0, Bf4::from_f32(1.0));
        slice_mut.set(1, Bf4::from_f32(-1.0));
        slice_mut.set(2, Bf4::from_f32(1.5));
        slice_mut.set(3, Bf4::from_f32(0.0));
        slice_mut.set(4, Bf4::from_f32(2.0));
        slice_mut.set(5, Bf4::from_f32(-2.0));
        slice_mut.set(6, Bf4::from_f32(-1.5));
    }

    let slice = PackedBf4Slice::new(&raw_bytes, 7).unwrap();
    assert_eq!(slice.len(), 7);
    assert_eq!(slice.get(0).unwrap().to_f32(), 1.0);
    assert_eq!(slice.get(1).unwrap().to_f32(), -1.0);
    assert_eq!(slice.get(2).unwrap().to_f32(), 1.5);
    assert_eq!(slice.get(3).unwrap().to_f32(), 0.0);
    assert_eq!(slice.get(4).unwrap().to_f32(), 2.0);
    assert_eq!(slice.get(5).unwrap().to_f32(), -2.0);
    assert_eq!(slice.get(6).unwrap().to_f32(), -1.5);

    let mut dest = [Bf16::from_f32(0.0); 7];
    slice.unpack_to_bf16(&mut dest);
    assert_eq!(dest[0].to_f32(), 1.0);
    assert_eq!(dest[1].to_f32(), -1.0);
    assert_eq!(dest[2].to_f32(), 1.5);
    assert_eq!(dest[3].to_f32(), 0.0);
    assert_eq!(dest[4].to_f32(), 2.0);
    assert_eq!(dest[5].to_f32(), -2.0);
    assert_eq!(dest[6].to_f32(), -1.5);
}

#[test]
fn test_adaptive_dispatcher_and_amx_session() {
    use hermes_simd::AdaptiveDispatcher;
    
    let a = [1.0f32; 100];
    let b = [2.0f32; 100];
    let decision = AdaptiveDispatcher::select_backend(10, 10, 10, a.as_ptr(), a.len(), b.as_ptr(), b.len());
    println!("Selected backend for small matrix: {:?}", decision);

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        use hermes_simd::{AmxSession, AmxBatchSession, AmxConfig, AmxSupport};
        let config = AmxConfig::new_uniform(16, 64);
        
        assert!(!AmxSession::is_active());

        if <half::bf16 as AmxSupport>::has_amx() {
            let _s1 = AmxSession::new(&config);
            assert!(AmxSession::is_active());
            {
                let _s2 = AmxSession::new(&config);
                assert!(AmxSession::is_active());
            }
            assert!(AmxSession::is_active());
        }
        assert!(!AmxSession::is_active());

        if <half::bf16 as AmxSupport>::has_amx() {
            let _s = AmxBatchSession::begin(&config);
            assert!(AmxSession::is_active());
        }
        assert!(!AmxSession::is_active());
    }
}

#[test]
fn test_generic_packed_vector_and_f4_slice() {
    use hermes_numeric::{F4, F32, PackedF4Slice, PackedF4SliceMut, PackedF4Vec, PackedBf4Vec, Bf4};

    // 1. Test F4 slice operations
    let mut raw_bytes = [0u8; 4];
    {
        let mut slice_mut = PackedF4SliceMut::new(&mut raw_bytes, 7).unwrap();
        assert_eq!(slice_mut.len(), 7);
        assert!(!slice_mut.is_empty());

        slice_mut.set(0, F4::from_f32(1.0));
        slice_mut.set(1, F4::from_f32(-1.0));
        slice_mut.set(2, F4::from_f32(0.5));
        slice_mut.set(3, F4::from_f32(0.0));
        slice_mut.set(4, F4::from_f32(2.0));
        slice_mut.set(5, F4::from_f32(-2.0));
        slice_mut.set(6, F4::from_f32(-0.5));
    }

    let slice = PackedF4Slice::new(&raw_bytes, 7).unwrap();
    assert_eq!(slice.len(), 7);
    assert_eq!(slice.get(0).unwrap().to_f32(), 1.0);
    assert_eq!(slice.get(1).unwrap().to_f32(), -1.0);
    assert_eq!(slice.get(2).unwrap().to_f32(), 0.5);
    assert_eq!(slice.get(3).unwrap().to_f32(), 0.0);
    assert_eq!(slice.get(4).unwrap().to_f32(), 2.0);
    assert_eq!(slice.get(5).unwrap().to_f32(), -2.0);
    assert_eq!(slice.get(6).unwrap().to_f32(), -0.5);

    let mut dest = [F32(0.0); 7];
    slice.unpack_to_f32(&mut dest);
    assert_eq!(dest[0].0, 1.0);
    assert_eq!(dest[1].0, -1.0);
    assert_eq!(dest[2].0, 0.5);
    assert_eq!(dest[3].0, 0.0);
    assert_eq!(dest[4].0, 2.0);
    assert_eq!(dest[5].0, -2.0);
    assert_eq!(dest[6].0, -0.5);

    // 2. Test PackedF4Vec and PackedBf4Vec operations
    let mut vec_f4 = PackedF4Vec::new();
    assert!(vec_f4.is_empty());
    vec_f4.push(F4::from_f32(1.0));
    vec_f4.push(F4::from_f32(-1.0));
    vec_f4.push(F4::from_f32(0.5));
    assert_eq!(vec_f4.len(), 3);
    assert_eq!(vec_f4.get(0).unwrap().to_f32(), 1.0);
    assert_eq!(vec_f4.get(1).unwrap().to_f32(), -1.0);
    assert_eq!(vec_f4.get(2).unwrap().to_f32(), 0.5);

    vec_f4.set(1, F4::from_f32(2.0));
    assert_eq!(vec_f4.get(1).unwrap().to_f32(), 2.0);

    let view_f4 = vec_f4.as_view();
    assert_eq!(view_f4.len(), 3);
    assert_eq!(view_f4.get(0).unwrap().to_f32(), 1.0);

    let mut vec_bf4 = PackedBf4Vec::with_capacity(10);
    vec_bf4.push(Bf4::from_f32(2.0));
    vec_bf4.push(Bf4::from_f32(-2.0));
    assert_eq!(vec_bf4.len(), 2);
    assert_eq!(vec_bf4.get(0).unwrap().to_f32(), 2.0);
    assert_eq!(vec_bf4.get(1).unwrap().to_f32(), -2.0);
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[test]
fn test_vectorized_packed_unpackers() {
    use hermes_numeric::{Bf4, F4, Bf16, F32};
    
    let n = 67;
    let mut packed_bytes = vec![0u8; (n + 1) / 2];
    for i in 0..packed_bytes.len() {
        let val1 = (i % 8) as f32 * 0.5;
        let val2 = ((i + 1) % 8) as f32 * -0.5;
        packed_bytes[i] = Bf4::pack_pair(Bf4::from_f32(val1), Bf4::from_f32(val2));
    }
    
    let mut unpacked_bf16 = vec![Bf16::from_f32(0.0); packed_bytes.len() * 2];
    if std::is_x86_feature_detected!("avx512bw") {
        hermes_simd_intrinsics::x86_64::avx512_tiling::unpack_packed_bf4_to_bf16(&packed_bytes, &mut unpacked_bf16);
        
        for i in 0..10 {
            let byte = packed_bytes[i / 2];
            let val = if i % 2 == 0 { byte & 0x0F } else { (byte >> 4) & 0x0F };
            let expected = Bf4(val).to_f32();
            println!("unpacked_bf16[{}] = {}, expected = {}", i, unpacked_bf16[i].to_f32(), expected);
        }

        for i in 0..n {
            let byte = packed_bytes[i / 2];
            let val = if i % 2 == 0 { byte & 0x0F } else { (byte >> 4) & 0x0F };
            let expected = Bf4(val).to_f32();
            assert_eq!(unpacked_bf16[i].to_f32(), expected, "Bf4 mismatch at index {}", i);
        }
    } else {
        println!("Skipping direct AVX-512 unpacked_bf16 test (avx512bw not detected)");
    }
    
    let mut unpacked_f32 = vec![F32(0.0); packed_bytes.len() * 2];
    if std::is_x86_feature_detected!("avx512f") {
        hermes_simd_intrinsics::x86_64::avx512_tiling::unpack_packed_f4_to_f32(&packed_bytes, &mut unpacked_f32);
        
        for i in 0..n {
            let byte = packed_bytes[i / 2];
            let val = if i % 2 == 0 { byte & 0x0F } else { (byte >> 4) & 0x0F };
            let expected = F4(val).to_f32();
            assert_eq!(unpacked_f32[i].0, expected, "F4 mismatch at index {}", i);
        }
    } else {
        println!("Skipping direct AVX-512 unpacked_f32 test (avx512f not detected)");
    }
}

#[test]
fn test_packed4_cow() {
    use hermes_numeric::{Bf4, F4, Bf16, F32};
    use hermes_simd::{Packed4Cow, PackedBf4Cow, PackedF4Cow, Packed4CowExt, Scalar, Unaligned, SimdCow};

    // 1. Test PackedBf4Cow Borrowed
    let original_bytes = vec![
        Bf4::pack_pair(Bf4::from_f32(1.0), Bf4::from_f32(-1.0)),
        Bf4::pack_pair(Bf4::from_f32(3.0), Bf4::from_f32(0.0)),
    ];
    let mut cow = PackedBf4Cow::from_packed_slice(&original_bytes, 4).unwrap();
    assert!(matches!(cow, Packed4Cow::Borrowed(_)));
    assert_eq!(cow.len(), 4);
    assert_eq!(cow.get(0).unwrap().to_f32(), 1.0);
    assert_eq!(cow.get(1).unwrap().to_f32(), -1.0);
    assert_eq!(cow.get(2).unwrap().to_f32(), 3.0);
    assert_eq!(cow.get(3).unwrap().to_f32(), 0.0);

    // 2. Upgrade to Mut/Owned
    cow.set(1, Bf4::from_f32(1.5));
    assert!(matches!(cow, Packed4Cow::Owned(_)));
    assert_eq!(cow.get(1).unwrap().to_f32(), 1.5);
    // index 0 should still be 1.0
    assert_eq!(cow.get(0).unwrap().to_f32(), 1.0);

    // 3. Test PackedF4Cow and IntoOwned
    let f4_bytes = vec![
        F4::pack_pair(F4::from_f32(1.0), F4::from_f32(2.0)),
    ];
    let cow_f4 = PackedF4Cow::from_packed_slice(&f4_bytes, 2).unwrap();
    let owned_vec = cow_f4.into_owned();
    assert_eq!(owned_vec.len(), 2);
    assert_eq!(owned_vec.get(0).unwrap().to_f32(), 1.0);
    assert_eq!(owned_vec.get(1).unwrap().to_f32(), 2.0);

    // 4. Test Unpacking from PackedBf4Cow to SimdCow of Bf16
    let orig = vec![
        Bf4::pack_pair(Bf4::from_f32(1.0), Bf4::from_f32(-1.0)),
    ];
    let cow_bf4 = PackedBf4Cow::from_packed_slice(&orig, 2).unwrap();
    let simd_cow: SimdCow<'static, Bf16, Scalar, Unaligned> = cow_bf4.unpack_to_cow();
    assert_eq!(simd_cow.len(), 2);
    assert_eq!(simd_cow[0].to_f32(), 1.0);
    assert_eq!(simd_cow[1].to_f32(), -1.0);

    // 5. Test Unpacking from PackedF4Cow to SimdCow of F32
    let orig_f4 = vec![
        F4::pack_pair(F4::from_f32(1.0), F4::from_f32(4.0)),
    ];
    let cow_f4 = PackedF4Cow::from_packed_slice(&orig_f4, 2).unwrap();
    let simd_cow_f32: SimdCow<'static, F32, Scalar, Unaligned> = cow_f4.unpack_to_cow();
    assert_eq!(simd_cow_f32.len(), 2);
    assert_eq!(simd_cow_f32[0].0, 1.0);
    assert_eq!(simd_cow_f32[1].0, 4.0);

    // 6. Test iteration
    let mut sum = 0.0;
    for elem in &cow_bf4 {
        sum += elem.to_f32();
    }
    assert_eq!(sum, 0.0); // 1.0 + -1.0
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[test]
fn test_bf8_and_f8_unpacking() {
    use hermes_numeric::{Bf8, F8, Bf16, F32, unpack_f8_to_f32};
    
    // 1. Test Bf8 to Bf16 unpacking (using AVX-512 when available, via hermes-simd-intrinsics)
    let bf8_inputs = [
        Bf8::from_f32(0.0),
        Bf8::from_f32(1.0),
        Bf8::from_f32(-1.0),
        Bf8::from_f32(2.5),
        Bf8::from_f32(-3.0),
    ];
    let mut bf16_outputs = [Bf16(half::bf16::ZERO); 5];
    
    if std::is_x86_feature_detected!("avx512bw") {
        // Direct hardware routed call
        hermes_simd_intrinsics::x86_64::avx512_tiling::unpack_bf8_to_bf16(&bf8_inputs, &mut bf16_outputs);
        
        assert_eq!(bf16_outputs[0].to_f32(), 0.0);
        assert_eq!(bf16_outputs[1].to_f32(), 1.0);
        assert_eq!(bf16_outputs[2].to_f32(), -1.0);
        assert_eq!(bf16_outputs[3].to_f32(), 2.5);
        assert_eq!(bf16_outputs[4].to_f32(), -3.0);
    } else {
        println!("Skipping direct AVX-512 unpack_bf8_to_bf16 test (avx512bw not detected)");
    }

    // 2. Test F8 to F32 unpacking (using AVX2 gather when available, via hermes_numeric)
    let f8_inputs = [
        F8::from_f32(0.0),
        F8::from_f32(1.0),
        F8::from_f32(-1.0),
        F8::from_f32(2.0),
        F8::from_f32(-4.0),
    ];
    let mut f32_outputs = [F32(0.0); 5];
    
    unpack_f8_to_f32(&f8_inputs, &mut f32_outputs);
    
    assert_eq!(f32_outputs[0].0, 0.0);
    assert_eq!(f32_outputs[1].0, 1.0);
    assert_eq!(f32_outputs[2].0, -1.0);
    assert_eq!(f32_outputs[3].0, 2.0);
    assert_eq!(f32_outputs[4].0, -4.0);
}

#[test]
fn test_packed_cow_rkyv_serialization() {
    use hermes_numeric::{Bf4, F4, PackedBf4Cow, PackedF4Cow, Packed4Cow};
    use rkyv::Deserialize;

    // 1. PackedBf4Cow test
    let bf4_bytes = vec![
        Bf4::pack_pair(Bf4::from_f32(1.0), Bf4::from_f32(-1.0)),
        Bf4::pack_pair(Bf4::from_f32(1.5), Bf4::from_f32(0.0)),
    ];
    let original_bf4_cow = PackedBf4Cow::from_packed_slice(&bf4_bytes, 4).unwrap();

    let bytes_bf4 = rkyv::to_bytes::<_, 256>(&original_bf4_cow).unwrap();
    let archived_bf4 = unsafe { rkyv::archived_root::<PackedBf4Cow>(&bytes_bf4[..]) };

    assert_eq!(archived_bf4.len(), 4);
    assert!(!archived_bf4.is_empty());
    
    // Test zero-copy borrow
    let borrowed_bf4 = archived_bf4.as_borrowed().unwrap();
    assert_eq!(borrowed_bf4.len(), 4);
    assert_eq!(borrowed_bf4.get(0).unwrap().to_f32(), 1.0);
    assert_eq!(borrowed_bf4.get(1).unwrap().to_f32(), -1.0);
    assert_eq!(borrowed_bf4.get(2).unwrap().to_f32(), 1.5);
    assert_eq!(borrowed_bf4.get(3).unwrap().to_f32(), 0.0);

    // Test deserialization to owned
    let deserialized_bf4: PackedBf4Cow = archived_bf4.deserialize(&mut rkyv::Infallible).unwrap();
    assert!(matches!(deserialized_bf4, Packed4Cow::Owned(_)));
    assert_eq!(deserialized_bf4.len(), 4);
    assert_eq!(deserialized_bf4.get(2).unwrap().to_f32(), 1.5);

    // 2. PackedF4Cow test
    let f4_bytes = vec![
        F4::pack_pair(F4::from_f32(-1.0), F4::from_f32(2.0)),
    ];
    let original_f4_cow = PackedF4Cow::from_packed_slice(&f4_bytes, 2).unwrap();

    let bytes_f4 = rkyv::to_bytes::<_, 256>(&original_f4_cow).unwrap();
    let archived_f4 = unsafe { rkyv::archived_root::<PackedF4Cow>(&bytes_f4[..]) };

    assert_eq!(archived_f4.len(), 2);
    let borrowed_f4 = archived_f4.as_borrowed().unwrap();
    assert_eq!(borrowed_f4.get(0).unwrap().to_f32(), -1.0);
    assert_eq!(borrowed_f4.get(1).unwrap().to_f32(), 2.0);

    let deserialized_f4: PackedF4Cow = archived_f4.deserialize(&mut rkyv::Infallible).unwrap();
    assert_eq!(deserialized_f4.len(), 2);
}





