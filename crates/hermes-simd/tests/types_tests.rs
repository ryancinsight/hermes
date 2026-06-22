#![allow(clippy::manual_div_ceil, clippy::needless_range_loop)]
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
        unsafe {
            c.store_unaligned(buf_res.as_mut_ptr());
        }
        for i in 0..$lanes {
            assert!(
                (buf_res[i] - (buf_a[i] / 2.0)).abs() < 1e-5,
                "Div failed at lane {} of {}",
                i,
                stringify!($arch)
            );
        }

        // 2. Bitwise operators
        // Bitwise and/or/xor operate on f32 float representations. Let's splat bit patterns.
        let pattern_a = f32::from_bits(0x5555_5555);
        let pattern_b = f32::from_bits(0x3333_3333);
        let vec_pat_a = Vector::<f32, $arch>::splat(pattern_a);
        let vec_pat_b = Vector::<f32, $arch>::splat(pattern_b);

        let and_res = vec_pat_a & vec_pat_b;
        unsafe {
            and_res.store_unaligned(buf_res.as_mut_ptr());
        }
        for i in 0..$lanes {
            let res_bits = buf_res[i].to_bits();
            assert_eq!(
                res_bits,
                0x5555_5555 & 0x3333_3333,
                "BitAnd failed at lane {} of {}",
                i,
                stringify!($arch)
            );
        }

        let or_res = vec_pat_a | vec_pat_b;
        unsafe {
            or_res.store_unaligned(buf_res.as_mut_ptr());
        }
        for i in 0..$lanes {
            let res_bits = buf_res[i].to_bits();
            assert_eq!(
                res_bits,
                0x5555_5555 | 0x3333_3333,
                "BitOr failed at lane {} of {}",
                i,
                stringify!($arch)
            );
        }

        let xor_res = vec_pat_a ^ vec_pat_b;
        unsafe {
            xor_res.store_unaligned(buf_res.as_mut_ptr());
        }
        for i in 0..$lanes {
            let res_bits = buf_res[i].to_bits();
            assert_eq!(
                res_bits,
                0x5555_5555 ^ 0x3333_3333,
                "BitXor failed at lane {} of {}",
                i,
                stringify!($arch)
            );
        }

        // 3. Math (abs, min, max, sqrt)
        let abs_res = a.abs();
        unsafe {
            abs_res.store_unaligned(buf_res.as_mut_ptr());
        }
        for i in 0..$lanes {
            assert!(
                (buf_res[i] - buf_a[i].abs()).abs() < 1e-5,
                "Abs failed at lane {} of {}",
                i,
                stringify!($arch)
            );
        }

        let min_res = a.min(b);
        unsafe {
            min_res.store_unaligned(buf_res.as_mut_ptr());
        }
        for i in 0..$lanes {
            let expected = if buf_a[i] < buf_b[i] {
                buf_a[i]
            } else {
                buf_b[i]
            };
            assert!(
                (buf_res[i] - expected).abs() < 1e-5,
                "Min failed at lane {} of {}",
                i,
                stringify!($arch)
            );
        }

        let max_res = a.max(b);
        unsafe {
            max_res.store_unaligned(buf_res.as_mut_ptr());
        }
        for i in 0..$lanes {
            let expected = if buf_a[i] > buf_b[i] {
                buf_a[i]
            } else {
                buf_b[i]
            };
            assert!(
                (buf_res[i] - expected).abs() < 1e-5,
                "Max failed at lane {} of {}",
                i,
                stringify!($arch)
            );
        }

        let non_neg = abs_res;
        let sqrt_res = non_neg.sqrt();
        unsafe {
            sqrt_res.store_unaligned(buf_res.as_mut_ptr());
        }
        for i in 0..$lanes {
            let expected = buf_a[i].abs().sqrt();
            assert!(
                (buf_res[i] - expected).abs() < 1e-5,
                "Sqrt failed at lane {} of {}",
                i,
                stringify!($arch)
            );
        }

        // 4. Comparisons (cmp_eq, cmp_ne, cmp_lt, cmp_le, cmp_gt, cmp_ge)
        // Check comparison returns NaN/all-ones mask on true, 0 on false
        let all_ones_u32 = 0xFFFF_FFFFu32;

        let eq_res = a.cmp_eq(b);
        unsafe {
            eq_res.store_unaligned(buf_res.as_mut_ptr());
        }
        for i in 0..$lanes {
            let actual_bits = buf_res[i].to_bits();
            let expected_bits = if buf_a[i] == buf_b[i] {
                all_ones_u32
            } else {
                0
            };
            assert_eq!(
                actual_bits,
                expected_bits,
                "CmpEq failed at lane {} of {}",
                i,
                stringify!($arch)
            );
        }

        let ne_res = a.cmp_ne(b);
        unsafe {
            ne_res.store_unaligned(buf_res.as_mut_ptr());
        }
        for i in 0..$lanes {
            let actual_bits = buf_res[i].to_bits();
            let expected_bits = if buf_a[i] != buf_b[i] {
                all_ones_u32
            } else {
                0
            };
            assert_eq!(
                actual_bits,
                expected_bits,
                "CmpNe failed at lane {} of {}",
                i,
                stringify!($arch)
            );
        }

        let lt_res = a.cmp_lt(b);
        unsafe {
            lt_res.store_unaligned(buf_res.as_mut_ptr());
        }
        for i in 0..$lanes {
            let actual_bits = buf_res[i].to_bits();
            let expected_bits = if buf_a[i] < buf_b[i] { all_ones_u32 } else { 0 };
            assert_eq!(
                actual_bits,
                expected_bits,
                "CmpLt failed at lane {} of {}",
                i,
                stringify!($arch)
            );
        }

        let le_res = a.cmp_le(b);
        unsafe {
            le_res.store_unaligned(buf_res.as_mut_ptr());
        }
        for i in 0..$lanes {
            let actual_bits = buf_res[i].to_bits();
            let expected_bits = if buf_a[i] <= buf_b[i] {
                all_ones_u32
            } else {
                0
            };
            assert_eq!(
                actual_bits,
                expected_bits,
                "CmpLe failed at lane {} of {}",
                i,
                stringify!($arch)
            );
        }

        let gt_res = a.cmp_gt(b);
        unsafe {
            gt_res.store_unaligned(buf_res.as_mut_ptr());
        }
        for i in 0..$lanes {
            let actual_bits = buf_res[i].to_bits();
            let expected_bits = if buf_a[i] > buf_b[i] { all_ones_u32 } else { 0 };
            assert_eq!(
                actual_bits,
                expected_bits,
                "CmpGt failed at lane {} of {}",
                i,
                stringify!($arch)
            );
        }

        let ge_res = a.cmp_ge(b);
        unsafe {
            ge_res.store_unaligned(buf_res.as_mut_ptr());
        }
        for i in 0..$lanes {
            let actual_bits = buf_res[i].to_bits();
            let expected_bits = if buf_a[i] >= buf_b[i] {
                all_ones_u32
            } else {
                0
            };
            assert_eq!(
                actual_bits,
                expected_bits,
                "CmpGe failed at lane {} of {}",
                i,
                stringify!($arch)
            );
        }

        // 5. Blend
        // Blend true_val and false_val using gt_res as the mask
        let true_val = Vector::<f32, $arch>::splat(100.0);
        let false_val = Vector::<f32, $arch>::splat(-100.0);
        let blend_res = gt_res.blend(true_val, false_val);
        unsafe {
            blend_res.store_unaligned(buf_res.as_mut_ptr());
        }
        for i in 0..$lanes {
            let expected = if buf_a[i] > buf_b[i] {
                100.0f32
            } else {
                -100.0f32
            };
            assert_eq!(
                buf_res[i],
                expected,
                "Blend failed at lane {} of {}",
                i,
                stringify!($arch)
            );
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
        unsafe {
            c.store_unaligned(buf_res.as_mut_ptr());
        }
        for i in 0..$lanes {
            assert!(
                (buf_res[i] - (buf_a[i] / 2.0)).abs() < 1e-12,
                "f64 Div failed at lane {} of {}",
                i,
                stringify!($arch)
            );
        }

        // 2. Bitwise operators
        let pattern_a = f64::from_bits(0x5555_5555_5555_5555);
        let pattern_b = f64::from_bits(0x3333_3333_3333_3333);
        let vec_pat_a = Vector::<f64, $arch>::splat(pattern_a);
        let vec_pat_b = Vector::<f64, $arch>::splat(pattern_b);

        let and_res = vec_pat_a & vec_pat_b;
        unsafe {
            and_res.store_unaligned(buf_res.as_mut_ptr());
        }
        for i in 0..$lanes {
            let res_bits = buf_res[i].to_bits();
            assert_eq!(
                res_bits,
                0x5555_5555_5555_5555 & 0x3333_3333_3333_3333,
                "f64 BitAnd failed at lane {} of {}",
                i,
                stringify!($arch)
            );
        }

        let or_res = vec_pat_a | vec_pat_b;
        unsafe {
            or_res.store_unaligned(buf_res.as_mut_ptr());
        }
        for i in 0..$lanes {
            let res_bits = buf_res[i].to_bits();
            assert_eq!(
                res_bits,
                0x5555_5555_5555_5555 | 0x3333_3333_3333_3333,
                "f64 BitOr failed at lane {} of {}",
                i,
                stringify!($arch)
            );
        }

        let xor_res = vec_pat_a ^ vec_pat_b;
        unsafe {
            xor_res.store_unaligned(buf_res.as_mut_ptr());
        }
        for i in 0..$lanes {
            let res_bits = buf_res[i].to_bits();
            assert_eq!(
                res_bits,
                0x5555_5555_5555_5555 ^ 0x3333_3333_3333_3333,
                "f64 BitXor failed at lane {} of {}",
                i,
                stringify!($arch)
            );
        }

        // 3. Math
        let abs_res = a.abs();
        unsafe {
            abs_res.store_unaligned(buf_res.as_mut_ptr());
        }
        for i in 0..$lanes {
            assert!(
                (buf_res[i] - buf_a[i].abs()).abs() < 1e-12,
                "f64 Abs failed at lane {} of {}",
                i,
                stringify!($arch)
            );
        }

        let min_res = a.min(b);
        unsafe {
            min_res.store_unaligned(buf_res.as_mut_ptr());
        }
        for i in 0..$lanes {
            let expected = if buf_a[i] < buf_b[i] {
                buf_a[i]
            } else {
                buf_b[i]
            };
            assert!(
                (buf_res[i] - expected).abs() < 1e-12,
                "f64 Min failed at lane {} of {}",
                i,
                stringify!($arch)
            );
        }

        let max_res = a.max(b);
        unsafe {
            max_res.store_unaligned(buf_res.as_mut_ptr());
        }
        for i in 0..$lanes {
            let expected = if buf_a[i] > buf_b[i] {
                buf_a[i]
            } else {
                buf_b[i]
            };
            assert!(
                (buf_res[i] - expected).abs() < 1e-12,
                "f64 Max failed at lane {} of {}",
                i,
                stringify!($arch)
            );
        }

        let non_neg = abs_res;
        let sqrt_res = non_neg.sqrt();
        unsafe {
            sqrt_res.store_unaligned(buf_res.as_mut_ptr());
        }
        for i in 0..$lanes {
            let expected = buf_a[i].abs().sqrt();
            assert!(
                (buf_res[i] - expected).abs() < 1e-12,
                "f64 Sqrt failed at lane {} of {}",
                i,
                stringify!($arch)
            );
        }

        // 4. Comparisons
        let all_ones_u64 = 0xFFFF_FFFF_FFFF_FFFFu64;

        let eq_res = a.cmp_eq(b);
        unsafe {
            eq_res.store_unaligned(buf_res.as_mut_ptr());
        }
        for i in 0..$lanes {
            let actual_bits = buf_res[i].to_bits();
            let expected_bits = if buf_a[i] == buf_b[i] {
                all_ones_u64
            } else {
                0
            };
            assert_eq!(
                actual_bits,
                expected_bits,
                "f64 CmpEq failed at lane {} of {}",
                i,
                stringify!($arch)
            );
        }

        let ne_res = a.cmp_ne(b);
        unsafe {
            ne_res.store_unaligned(buf_res.as_mut_ptr());
        }
        for i in 0..$lanes {
            let actual_bits = buf_res[i].to_bits();
            let expected_bits = if buf_a[i] != buf_b[i] {
                all_ones_u64
            } else {
                0
            };
            assert_eq!(
                actual_bits,
                expected_bits,
                "f64 CmpNe failed at lane {} of {}",
                i,
                stringify!($arch)
            );
        }

        let lt_res = a.cmp_lt(b);
        unsafe {
            lt_res.store_unaligned(buf_res.as_mut_ptr());
        }
        for i in 0..$lanes {
            let actual_bits = buf_res[i].to_bits();
            let expected_bits = if buf_a[i] < buf_b[i] { all_ones_u64 } else { 0 };
            assert_eq!(
                actual_bits,
                expected_bits,
                "f64 CmpLt failed at lane {} of {}",
                i,
                stringify!($arch)
            );
        }

        let le_res = a.cmp_le(b);
        unsafe {
            le_res.store_unaligned(buf_res.as_mut_ptr());
        }
        for i in 0..$lanes {
            let actual_bits = buf_res[i].to_bits();
            let expected_bits = if buf_a[i] <= buf_b[i] {
                all_ones_u64
            } else {
                0
            };
            assert_eq!(
                actual_bits,
                expected_bits,
                "f64 CmpLe failed at lane {} of {}",
                i,
                stringify!($arch)
            );
        }

        let gt_res = a.cmp_gt(b);
        unsafe {
            gt_res.store_unaligned(buf_res.as_mut_ptr());
        }
        for i in 0..$lanes {
            let actual_bits = buf_res[i].to_bits();
            let expected_bits = if buf_a[i] > buf_b[i] { all_ones_u64 } else { 0 };
            assert_eq!(
                actual_bits,
                expected_bits,
                "f64 CmpGt failed at lane {} of {}",
                i,
                stringify!($arch)
            );
        }

        let ge_res = a.cmp_ge(b);
        unsafe {
            ge_res.store_unaligned(buf_res.as_mut_ptr());
        }
        for i in 0..$lanes {
            let actual_bits = buf_res[i].to_bits();
            let expected_bits = if buf_a[i] >= buf_b[i] {
                all_ones_u64
            } else {
                0
            };
            assert_eq!(
                actual_bits,
                expected_bits,
                "f64 CmpGe failed at lane {} of {}",
                i,
                stringify!($arch)
            );
        }

        // 5. Blend
        let true_val = Vector::<f64, $arch>::splat(100.0);
        let false_val = Vector::<f64, $arch>::splat(-100.0);
        let blend_res = gt_res.blend(true_val, false_val);
        unsafe {
            blend_res.store_unaligned(buf_res.as_mut_ptr());
        }
        for i in 0..$lanes {
            let expected = if buf_a[i] > buf_b[i] {
                100.0f64
            } else {
                -100.0f64
            };
            assert_eq!(
                buf_res[i],
                expected,
                "f64 Blend failed at lane {} of {}",
                i,
                stringify!($arch)
            );
        }
    };
}

// Helper macro to run generic tests for any Scalar type on a given architecture
macro_rules! test_numeric_ops_for_arch {
    ($t:ty, $arch:ident, $lanes:expr) => {
        let mut buf_a = [<$t as hermes_simd_core::scalar::NumericElement>::ZERO; $lanes];
        let mut buf_b = [<$t as hermes_simd_core::scalar::NumericElement>::ZERO; $lanes];
        for i in 0..$lanes {
            buf_a[i] = if i % 2 == 0 {
                <$t as hermes_simd_core::scalar::NumericElement>::ONE
                    + <$t as hermes_simd_core::scalar::NumericElement>::ONE
            } else {
                <$t as hermes_simd_core::scalar::NumericElement>::ONE
            };
            buf_b[i] = <$t as hermes_simd_core::scalar::NumericElement>::ONE
                + <$t as hermes_simd_core::scalar::NumericElement>::ONE;
        }

        let a = unsafe { Vector::<$t, $arch>::load_unaligned(buf_a.as_ptr()) };
        let b = unsafe { Vector::<$t, $arch>::load_unaligned(buf_b.as_ptr()) };

        // 1. Arithmetic
        let c_add = a + b;
        let mut buf_res = [<$t as hermes_simd_core::scalar::NumericElement>::ZERO; $lanes];
        unsafe {
            c_add.store_unaligned(buf_res.as_mut_ptr());
        }
        for i in 0..$lanes {
            assert_eq!(
                buf_res[i],
                buf_a[i] + buf_b[i],
                "Add failed for {} at lane {} on {}",
                stringify!($t),
                i,
                stringify!($arch)
            );
        }

        let c_sub = a - b;
        unsafe {
            c_sub.store_unaligned(buf_res.as_mut_ptr());
        }
        for i in 0..$lanes {
            assert_eq!(
                buf_res[i],
                buf_a[i] - buf_b[i],
                "Sub failed for {} at lane {} on {}",
                stringify!($t),
                i,
                stringify!($arch)
            );
        }

        let c_mul = a * b;
        unsafe {
            c_mul.store_unaligned(buf_res.as_mut_ptr());
        }
        for i in 0..$lanes {
            assert_eq!(
                buf_res[i],
                buf_a[i] * buf_b[i],
                "Mul failed for {} at lane {} on {}",
                stringify!($t),
                i,
                stringify!($arch)
            );
        }

        // 2. Bitwise operators
        let and_res = a & b;
        unsafe {
            and_res.store_unaligned(buf_res.as_mut_ptr());
        }
        for i in 0..$lanes {
            assert_eq!(
                buf_res[i],
                buf_a[i].bitand(buf_b[i]),
                "BitAnd failed for {} at lane {} on {}",
                stringify!($t),
                i,
                stringify!($arch)
            );
        }

        let or_res = a | b;
        unsafe {
            or_res.store_unaligned(buf_res.as_mut_ptr());
        }
        for i in 0..$lanes {
            assert_eq!(
                buf_res[i],
                buf_a[i].bitor(buf_b[i]),
                "BitOr failed for {} at lane {} on {}",
                stringify!($t),
                i,
                stringify!($arch)
            );
        }

        let xor_res = a ^ b;
        unsafe {
            xor_res.store_unaligned(buf_res.as_mut_ptr());
        }
        for i in 0..$lanes {
            assert_eq!(
                buf_res[i],
                buf_a[i].bitxor(buf_b[i]),
                "BitXor failed for {} at lane {} on {}",
                stringify!($t),
                i,
                stringify!($arch)
            );
        }

        // 3. Comparisons
        let eq_res = a.cmp_eq(b);
        unsafe {
            eq_res.store_unaligned(buf_res.as_mut_ptr());
        }
        for i in 0..$lanes {
            let expected = if buf_a[i] == buf_b[i] {
                <$t as hermes_simd_core::scalar::NumericElement>::ALL_ONES
            } else {
                <$t as hermes_simd_core::scalar::NumericElement>::ZERO
            };
            let actual = buf_res[i];
            let is_ok = (actual.is_nan() && expected.is_nan()) || actual == expected;
            assert!(
                is_ok,
                "CmpEq failed for {} at lane {} on {} (actual: {:?}, expected: {:?})",
                stringify!($t),
                i,
                stringify!($arch),
                actual,
                expected
            );
        }

        let lt_res = a.cmp_lt(b);
        unsafe {
            lt_res.store_unaligned(buf_res.as_mut_ptr());
        }
        for i in 0..$lanes {
            let expected = if buf_a[i] < buf_b[i] {
                <$t as hermes_simd_core::scalar::NumericElement>::ALL_ONES
            } else {
                <$t as hermes_simd_core::scalar::NumericElement>::ZERO
            };
            let actual = buf_res[i];
            let is_ok = (actual.is_nan() && expected.is_nan()) || actual == expected;
            assert!(
                is_ok,
                "CmpLt failed for {} at lane {} on {} (actual: {:?}, expected: {:?})",
                stringify!($t),
                i,
                stringify!($arch),
                actual,
                expected
            );
        }

        // 4. Blend
        let true_val =
            Vector::<$t, $arch>::splat(<$t as hermes_simd_core::scalar::NumericElement>::ONE);
        let false_val = Vector::<$t, $arch>::zero();
        let blend_res = eq_res.blend(true_val, false_val);
        unsafe {
            blend_res.store_unaligned(buf_res.as_mut_ptr());
        }
        for i in 0..$lanes {
            let expected = if buf_a[i] == buf_b[i] {
                <$t as hermes_simd_core::scalar::NumericElement>::ONE
            } else {
                <$t as hermes_simd_core::scalar::NumericElement>::ZERO
            };
            let actual = buf_res[i];
            let is_ok = (actual.is_nan() && expected.is_nan()) || actual == expected;
            assert!(
                is_ok,
                "Blend failed for {} at lane {} on {} (actual: {:?}, expected: {:?})",
                stringify!($t),
                i,
                stringify!($arch),
                actual,
                expected
            );
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
    let mut view_out =
        SimdView::<f32, Scalar, Unaligned, Unmasked, &mut [f32]>::new_mut(&mut out_buf).unwrap();
    loaded_v0.store_to_view_chunk(&mut view_out, 1);
    loaded_v1.store_to_view_chunk(&mut view_out, 0);
    assert_eq!(out_buf, [50.0, 60.0, 70.0, 80.0, 10.0, 20.0, 30.0, 40.0]);

    // 6. SimdCowExt::transform_vectors
    let mut cow = SimdCow::<f32, Scalar, Unaligned>::Owned(AlignedVec::from_slice(&[
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0,
    ]));
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
        let data = [
            Bf16::from_f32(1.0),
            Bf16::from_f32(2.0),
            Bf16::from_f32(3.0),
        ];
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
    use hermes_numeric::{unpack_bf4_to_bf16, unpack_bf4_to_bf16_packed, unpack_bf8_to_bf16};

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
    use hermes_numeric::{unpack_f4_to_f32, unpack_f4_to_f32_packed, F32, F4};
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
fn test_widen_i8_primitives() {
    use hermes_simd::{
        widen_I8_to_I16, widen_I8_to_I32, widen_i8_to_i16, widen_i8_to_i32, I16, I32, I8,
    };

    let src = [-1i8, 0, 5, 127, -128];
    let mut dest_i16 = [0i16; 5];
    let mut dest_i32 = [0i32; 5];

    widen_i8_to_i16(&src, &mut dest_i16);
    widen_i8_to_i32(&src, &mut dest_i32);

    assert_eq!(dest_i16, [-1i16, 0, 5, 127, -128]);
    assert_eq!(dest_i32, [-1i32, 0, 5, 127, -128]);

    let src_wrapped = [I8(-1), I8(0), I8(5), I8(127), I8(-128)];
    let mut dest_i16_wrapped = [I16(0); 5];
    let mut dest_i32_wrapped = [I32(0); 5];

    widen_I8_to_I16(&src_wrapped, &mut dest_i16_wrapped);
    widen_I8_to_I32(&src_wrapped, &mut dest_i32_wrapped);

    assert_eq!(
        dest_i16_wrapped,
        [I16(-1), I16(0), I16(5), I16(127), I16(-128)]
    );
    assert_eq!(
        dest_i32_wrapped,
        [I32(-1), I32(0), I32(5), I32(127), I32(-128)]
    );
}

#[test]
fn test_numa_topology_and_allocation() {
    use hermes_simd_core::align::Unaligned;
    use hermes_simd_core::numa::{verify_numa_locality, NumaTopologyService};
    use hermes_simd_core::AlignedVec;

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
fn test_numa_locality_robustness_and_cache_pollution_prevention() {
    use hermes_simd_core::align::Unaligned;
    use hermes_simd_core::numa::{verify_numa_locality, NumaBinding};
    use hermes_simd_core::AlignedVec;

    // 1. Verify NumaBinding bind and drop is safe
    {
        let _binding = NumaBinding::bind(0);
    }

    // 2. Verify locality verification on stack memory is safe and does not crash
    let stack_val = [0u8; 1024];
    let _is_local_stack = verify_numa_locality(stack_val.as_ptr(), stack_val.len(), 0);

    // 3. Verify locality verification on standard heap memory is safe and does not crash
    let heap_val = std::vec![0u8; 4096];
    let _is_local_heap = verify_numa_locality(heap_val.as_ptr(), heap_val.len(), 0);

    // 4. Verify small NUMA allocations (cache-pollution mitigation bypass path) construct and drop safely
    let mut small_vec: AlignedVec<f32, Unaligned> = AlignedVec::with_capacity_numa(2, 0);
    small_vec.push(42.0);
    assert_eq!(small_vec[0], 42.0);
    assert_eq!(small_vec.len(), 1);
    drop(small_vec);
}

#[test]
fn test_packed_bf4_slice() {
    use hermes_numeric::{Bf16, Bf4, PackedBf4Slice, PackedBf4SliceMut};

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
    let decision =
        AdaptiveDispatcher::select_backend(10, 10, 10, a.as_ptr(), a.len(), b.as_ptr(), b.len());
    println!("Selected backend for small matrix: {:?}", decision);

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        use hermes_simd::{AmxBatchSession, AmxConfig, AmxSession, AmxSupport};
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
    use hermes_numeric::{
        Bf4, PackedBf4Vec, PackedF4Slice, PackedF4SliceMut, PackedF4Vec, F32, F4,
    };

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
    use hermes_numeric::{Bf16, Bf4, F32, F4};

    let n = 67;
    let mut packed_bytes = vec![0u8; (n + 1) / 2];
    for i in 0..packed_bytes.len() {
        let val1 = (i % 8) as f32 * 0.5;
        let val2 = ((i + 1) % 8) as f32 * -0.5;
        packed_bytes[i] = Bf4::pack_pair(Bf4::from_f32(val1), Bf4::from_f32(val2));
    }

    let mut unpacked_bf16 = vec![Bf16::from_f32(0.0); packed_bytes.len() * 2];
    if std::is_x86_feature_detected!("avx512bw") {
        hermes_simd_intrinsics::x86_64::avx512_tiling::unpack_packed_bf4_to_bf16(
            &packed_bytes,
            &mut unpacked_bf16,
        );

        for i in 0..10 {
            let byte = packed_bytes[i / 2];
            let val = if i % 2 == 0 {
                byte & 0x0F
            } else {
                (byte >> 4) & 0x0F
            };
            let expected = Bf4(val).to_f32();
            println!(
                "unpacked_bf16[{}] = {}, expected = {}",
                i,
                unpacked_bf16[i].to_f32(),
                expected
            );
        }

        for i in 0..n {
            let byte = packed_bytes[i / 2];
            let val = if i % 2 == 0 {
                byte & 0x0F
            } else {
                (byte >> 4) & 0x0F
            };
            let expected = Bf4(val).to_f32();
            assert_eq!(
                unpacked_bf16[i].to_f32(),
                expected,
                "Bf4 mismatch at index {}",
                i
            );
        }
    } else {
        println!("Skipping direct AVX-512 unpacked_bf16 test (avx512bw not detected)");
    }

    let mut unpacked_f32 = vec![F32(0.0); packed_bytes.len() * 2];
    if std::is_x86_feature_detected!("avx512f") {
        hermes_simd_intrinsics::x86_64::avx512_tiling::unpack_packed_f4_to_f32(
            &packed_bytes,
            &mut unpacked_f32,
        );

        for i in 0..n {
            let byte = packed_bytes[i / 2];
            let val = if i % 2 == 0 {
                byte & 0x0F
            } else {
                (byte >> 4) & 0x0F
            };
            let expected = F4(val).to_f32();
            assert_eq!(unpacked_f32[i].0, expected, "F4 mismatch at index {}", i);
        }
    } else {
        println!("Skipping direct AVX-512 unpacked_f32 test (avx512f not detected)");
    }
}

#[test]
fn test_packed4_cow() {
    use hermes_numeric::{Bf16, Bf4, F32, F4};
    use hermes_simd::{
        Packed4Cow, Packed4CowExt, PackedBf4Cow, PackedF4Cow, Scalar, SimdCow, Unaligned,
    };

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
    let f4_bytes = vec![F4::pack_pair(F4::from_f32(1.0), F4::from_f32(2.0))];
    let cow_f4 = PackedF4Cow::from_packed_slice(&f4_bytes, 2).unwrap();
    let owned_vec = cow_f4.into_owned();
    assert_eq!(owned_vec.len(), 2);
    assert_eq!(owned_vec.get(0).unwrap().to_f32(), 1.0);
    assert_eq!(owned_vec.get(1).unwrap().to_f32(), 2.0);

    // 4. Test Unpacking from PackedBf4Cow to SimdCow of Bf16
    let orig = vec![Bf4::pack_pair(Bf4::from_f32(1.0), Bf4::from_f32(-1.0))];
    let cow_bf4 = PackedBf4Cow::from_packed_slice(&orig, 2).unwrap();
    let simd_cow: SimdCow<'static, Bf16, Scalar, Unaligned> = cow_bf4.unpack_to_cow();
    assert_eq!(simd_cow.len(), 2);
    assert_eq!(simd_cow[0].to_f32(), 1.0);
    assert_eq!(simd_cow[1].to_f32(), -1.0);

    // 5. Test Unpacking from PackedF4Cow to SimdCow of F32
    let orig_f4 = vec![F4::pack_pair(F4::from_f32(1.0), F4::from_f32(4.0))];
    let cow_f4 = PackedF4Cow::from_packed_slice(&orig_f4, 2).unwrap();
    let simd_cow_f32: SimdCow<'static, F32, Scalar, Unaligned> = cow_f4.unpack_to_cow();
    assert_eq!(simd_cow_f32.len(), 2);
    assert_eq!(simd_cow_f32[0].0, 1.0);
    assert_eq!(simd_cow_f32[1].0, 4.0);

    // 6. Test odd-length packed COW unpacking across the full nibble domain.
    let odd_len = 31_usize;
    let mut packed = Vec::with_capacity(odd_len.div_ceil(2));
    for byte_index in 0..odd_len.div_ceil(2) {
        let lo = (2 * byte_index) as u8 & 0x0f;
        let hi = (2 * byte_index + 1) as u8 & 0x0f;
        packed.push(Bf4::pack_pair(Bf4(lo), Bf4(hi)));
    }
    let odd_cow = PackedBf4Cow::from_packed_slice(&packed, odd_len).unwrap();
    let simd_cow: SimdCow<'static, Bf16, Scalar, Unaligned> = odd_cow.unpack_to_cow();
    assert_eq!(simd_cow.len(), odd_len);
    for index in 0..odd_len {
        let expected = Bf4((index as u8) & 0x0f).to_f32();
        let actual = simd_cow[index].to_f32();
        if expected.is_nan() {
            assert!(actual.is_nan());
        } else {
            assert_eq!(actual.to_bits(), expected.to_bits());
        }
    }

    let mut packed = Vec::with_capacity(odd_len.div_ceil(2));
    for byte_index in 0..odd_len.div_ceil(2) {
        let lo = (2 * byte_index) as u8 & 0x0f;
        let hi = (2 * byte_index + 1) as u8 & 0x0f;
        packed.push(F4::pack_pair(F4(lo), F4(hi)));
    }
    let odd_cow = PackedF4Cow::from_packed_slice(&packed, odd_len).unwrap();
    let simd_cow: SimdCow<'static, F32, Scalar, Unaligned> = odd_cow.unpack_to_cow();
    assert_eq!(simd_cow.len(), odd_len);
    for index in 0..odd_len {
        let expected = F4((index as u8) & 0x0f).to_f32();
        let actual = simd_cow[index].0;
        if expected.is_nan() {
            assert!(actual.is_nan());
        } else {
            assert_eq!(actual.to_bits(), expected.to_bits());
        }
    }

    // 7. Test iteration
    let mut sum = 0.0;
    for elem in &cow_bf4 {
        sum += elem.to_f32();
    }
    assert_eq!(sum, 0.0); // 1.0 + -1.0
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[test]
fn test_bf8_and_f8_unpacking() {
    use hermes_numeric::{unpack_f8_to_f32, Bf16, Bf8, F32, F8};

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
        hermes_simd_intrinsics::x86_64::avx512_tiling::unpack_bf8_to_bf16(
            &bf8_inputs,
            &mut bf16_outputs,
        );

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
    use hermes_numeric::{Bf4, Packed4Cow, PackedBf4Cow, PackedF4Cow, F4};
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
    let f4_bytes = vec![F4::pack_pair(F4::from_f32(-1.0), F4::from_f32(2.0))];
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

#[test]
fn test_preferred_and_monomorphized_types() {
    use hermes_simd_types::{SimdF32, F32};

    let vec_zeros = SimdF32::zero();
    let vec_splat = SimdF32::splat(F32(3.0));

    assert_eq!(vec_zeros.extract::<0>().0, 0.0);
    assert_eq!(vec_splat.extract::<0>().0, 3.0);
}

#[test]
fn test_simd_view_slicing_and_alignment_transitions() {
    use hermes_numeric::F32;
    use hermes_simd::{Scalar, SimdView, Unaligned};

    let data = vec![F32(1.0), F32(2.0), F32(3.0), F32(4.0), F32(5.0), F32(6.0)];
    let view: SimdView<'_, F32, Scalar, Unaligned> = SimdView::new(&data).unwrap();

    // 1. Slicing unaligned
    let sub_view = view.slice_unaligned(1..5);
    assert_eq!(sub_view.len(), 4);
    assert_eq!(sub_view[0].0, 2.0);
    assert_eq!(sub_view[3].0, 5.0);

    // 2. Alignment demotion
    let unaligned_view = sub_view.into_unaligned();
    assert_eq!(unaligned_view.len(), 4);

    // 3. Alignment promotion attempts
    let opt_aligned = unaligned_view.try_into_aligned::<4>();
    assert!(opt_aligned.is_some());

    // 4. Slicing aligned
    let aligned_slice = view.slice_aligned::<4>(0..4);
    assert!(aligned_slice.is_some());
}

#[test]
fn test_simd_cow_slicing_and_alignment_transitions() {
    use hermes_numeric::F32;
    use hermes_simd::{Scalar, SimdCow, Unaligned};

    // 1. Borrowed SimdCow
    let data = vec![F32(10.0), F32(20.0), F32(30.0), F32(40.0)];
    let cow = SimdCow::<F32, Scalar, Unaligned>::borrow_slice(&data).unwrap();

    // Slicing
    let sliced_cow = cow.slice_unaligned(1..3);
    assert!(matches!(sliced_cow, SimdCow::Borrowed(_)));
    assert_eq!(sliced_cow.len(), 2);
    assert_eq!(sliced_cow[0].0, 20.0);

    // Alignment promotion
    let aligned_cow = sliced_cow.try_into_aligned::<4>();
    assert!(aligned_cow.is_some());

    // 2. Owned SimdCow
    let cow_owned = SimdCow::<F32, Scalar, Unaligned>::from_slice(&data);
    let sliced_owned = cow_owned.slice_unaligned(2..4);
    // Should return a Borrowed view of the owned buffer (zero-copy!)
    assert!(matches!(sliced_owned, SimdCow::Borrowed(_)));
    assert_eq!(sliced_owned.len(), 2);
    assert_eq!(sliced_owned[0].0, 30.0);
}

#[test]
fn test_simd_cow_mutable_views_and_slicing() {
    use hermes_numeric::F32;
    use hermes_simd::{Scalar, SimdCow, Unaligned};

    let data = vec![F32(1.0), F32(2.0), F32(3.0), F32(4.0)];
    let mut cow = SimdCow::<F32, Scalar, Unaligned>::borrow_slice(&data).unwrap();

    // 1. view_mut on borrowed cow upgrades it to owned
    {
        let mut v_mut = cow.view_mut();
        assert_eq!(v_mut.len(), 4);
        v_mut[0] = F32(10.0);
    }
    assert!(matches!(cow, SimdCow::Owned(_)));
    assert_eq!(cow[0].0, 10.0);

    // 2. slice_unaligned_mut
    {
        let mut sub_v_mut = cow.slice_unaligned_mut(1..3);
        assert_eq!(sub_v_mut.len(), 2);
        assert_eq!(sub_v_mut[0].0, 2.0);
        sub_v_mut[0] = F32(20.0);
    }
    assert_eq!(cow[1].0, 20.0);

    // 3. slice_aligned_mut
    {
        let opt_aligned_mut = cow.slice_aligned_mut::<4>(0..4);
        assert!(opt_aligned_mut.is_some());
        let mut aligned_mut = opt_aligned_mut.unwrap();
        aligned_mut[3] = F32(40.0);
    }
    assert_eq!(cow[3].0, 40.0);
}

#[test]
fn test_packed_4bit_zero_copy_slicing() {
    use hermes_numeric::{Bf4, Packed4Cow, PackedBf4Cow, PackedBf4Slice, PackedBf4SliceMut};

    // Pack: elements are low, high per byte.
    // 0x12 -> low = 2, high = 1
    // 0x34 -> low = 4, high = 3
    let bytes = [0x12, 0x34];

    // 1. Packed4Slice sub_slice
    let slice = PackedBf4Slice::new(&bytes, 4).unwrap();

    // Even start is allowed (byte-aligned)
    let sub_even = slice.sub_slice(2..4).unwrap();
    assert_eq!(sub_even.len(), 2);
    // index 2 is the low part of 0x34 (value 4), index 3 is high part of 0x34 (value 3)
    assert_eq!(sub_even.get(0).unwrap().0, 4);
    assert_eq!(sub_even.get(1).unwrap().0, 3);

    // Odd start is disallowed
    let sub_odd = slice.sub_slice(1..3);
    assert!(sub_odd.is_none());

    // 2. Packed4SliceMut sub_slice_mut
    let mut bytes_mut = [0x12, 0x34];
    let slice_mut = PackedBf4SliceMut::new(&mut bytes_mut, 4).unwrap();

    // Even start is allowed
    let mut sub_even_mut = slice_mut.sub_slice_mut(2..4).unwrap();
    assert_eq!(sub_even_mut.len(), 2);
    sub_even_mut.set(0, Bf4(9));
    assert_eq!(sub_even_mut.get(0).unwrap().0, 9);

    // Odd start is disallowed
    let slice_mut_2 = PackedBf4SliceMut::new(&mut bytes_mut, 4).unwrap();
    assert!(slice_mut_2.sub_slice_mut(1..3).is_none());

    // 3. Packed4Cow sub_slice
    let cow = PackedBf4Cow::from_packed_slice(&bytes, 4).unwrap();
    let sub_cow = cow.sub_slice(2..4).unwrap();
    assert!(matches!(sub_cow, Packed4Cow::Borrowed(_)));
    assert_eq!(sub_cow.len(), 2);
    assert_eq!(sub_cow.get(0).unwrap().0, 4);

    let sub_cow_odd = cow.sub_slice(1..3);
    assert!(sub_cow_odd.is_none());
}

macro_rules! test_select_ops_for_arch {
    ($t:ty, $arch:ident, $lanes:expr) => {{
        let len = $lanes * 2 + 3;
        let mut data_a = vec![<$t as hermes_simd_core::scalar::NumericElement>::ZERO; len];
        let mut data_b = vec![<$t as hermes_simd_core::scalar::NumericElement>::ZERO; len];
        let mut mask = vec![false; len];

        for i in 0..len {
            data_a[i] = <$t as CastFrom<f64>>::cast_from(i as f64);
            data_b[i] = <$t as CastFrom<f64>>::cast_from((i + 100) as f64);
            mask[i] = i % 2 == 0;
        }

        let view_a = SimdView::<'_, $t, $arch, Unaligned, Unmasked, &[$t]>::new(&data_a).unwrap();
        let view_b = SimdView::<'_, $t, $arch, Unaligned, Unmasked, &[$t]>::new(&data_b).unwrap();

        // 1. select
        let sel_res = view_a.select(&mask, &view_b).unwrap();
        assert_eq!(sel_res.len(), len);
        let sel_slice = sel_res.as_slice();
        for i in 0..len {
            let expected = if mask[i] { data_a[i] } else { data_b[i] };
            assert_eq!(
                sel_slice[i],
                expected,
                "select failed for {} at index {} on {}",
                stringify!($t),
                i,
                stringify!($arch)
            );
        }

        // 2. masked_negate
        let neg_res = view_a.masked_negate(&mask).unwrap();
        assert_eq!(neg_res.len(), len);
        let neg_slice = neg_res.as_slice();
        for i in 0..len {
            let expected = if mask[i] { -data_a[i] } else { data_a[i] };
            assert_eq!(
                neg_slice[i],
                expected,
                "masked_negate failed for {} at index {} on {}",
                stringify!($t),
                i,
                stringify!($arch)
            );
        }
    }};
}

#[test]
fn test_select_ops_scalar() {
    test_select_ops_for_arch!(f32, Scalar, 4);
    test_select_ops_for_arch!(f64, Scalar, 2);
    test_select_ops_for_arch!(half::bf16, Scalar, 8);
    test_select_ops_for_arch!(i8, Scalar, 16);
    test_select_ops_for_arch!(i16, Scalar, 8);
    test_select_ops_for_arch!(i32, Scalar, 4);
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[test]
fn test_select_ops_avx2() {
    if std::is_x86_feature_detected!("avx2") {
        test_select_ops_for_arch!(f32, Avx2, 8);
        test_select_ops_for_arch!(f64, Avx2, 4);
        test_select_ops_for_arch!(half::bf16, Avx2, 16);
        test_select_ops_for_arch!(i8, Avx2, 32);
        test_select_ops_for_arch!(i16, Avx2, 16);
        test_select_ops_for_arch!(i32, Avx2, 8);
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[test]
fn test_select_ops_avx512() {
    if std::is_x86_feature_detected!("avx512f") {
        test_select_ops_for_arch!(f32, Avx512, 16);
        test_select_ops_for_arch!(f64, Avx512, 8);
        test_select_ops_for_arch!(half::bf16, Avx512, 32);
        test_select_ops_for_arch!(i8, Avx512, 64);
        test_select_ops_for_arch!(i16, Avx512, 32);
        test_select_ops_for_arch!(i32, Avx512, 16);
    }
}

#[cfg(target_arch = "aarch64")]
#[test]
fn test_select_ops_neon() {
    test_select_ops_for_arch!(f32, Neon, 4);
    test_select_ops_for_arch!(f64, Neon, 2);
    test_select_ops_for_arch!(half::bf16, Neon, 8);
    test_select_ops_for_arch!(i8, Neon, 16);
    test_select_ops_for_arch!(i16, Neon, 8);
    test_select_ops_for_arch!(i32, Neon, 4);
}

#[test]
fn test_insufficient_alignment_view_rejection() {
    use hermes_numeric::F32;
    use hermes_simd::{Avx2, Avx512, Scalar};
    use hermes_simd_core::align::{Aligned, Unaligned};
    use hermes_simd_core::view::SimdView;

    // Buffer aligned to 64 bytes
    #[repr(align(64))]
    struct Align64Buf([F32; 16]);
    let buf = Align64Buf([F32(0.0); 16]);

    // 1. Scalar arch (REGISTER_WIDTH_BITS = 0) accepts any Aligned alignment
    let view_scalar = SimdView::<'_, F32, Scalar, Aligned<16>>::new(&buf.0);
    assert!(view_scalar.is_some());

    // 2. Avx2 requires 32-byte alignment. Aligned<16> must be rejected.
    let view_avx2_bad = SimdView::<'_, F32, Avx2, Aligned<16>>::new(&buf.0);
    assert!(view_avx2_bad.is_none());

    // Aligned<32> must be accepted.
    let view_avx2_good = SimdView::<'_, F32, Avx2, Aligned<32>>::new(&buf.0);
    assert!(view_avx2_good.is_some());

    // try_into_aligned:<16> on Avx2 must be rejected.
    let view_avx2_unaligned = SimdView::<'_, F32, Avx2, Unaligned>::new(&buf.0).unwrap();
    assert!(view_avx2_unaligned.try_into_aligned::<16>().is_none());
    assert!(view_avx2_unaligned.try_into_aligned::<32>().is_some());

    // slice_aligned:<16> on Avx2 must be rejected.
    assert!(view_avx2_unaligned.slice_aligned::<16>(0..8).is_none());
    assert!(view_avx2_unaligned.slice_aligned::<32>(0..8).is_some());

    // 3. Avx512 requires 64-byte alignment. Aligned<32> must be rejected.
    let view_avx512_bad = SimdView::<'_, F32, Avx512, Aligned<32>>::new(&buf.0);
    assert!(view_avx512_bad.is_none());

    // Aligned<64> must be accepted.
    let view_avx512_good = SimdView::<'_, F32, Avx512, Aligned<64>>::new(&buf.0);
    assert!(view_avx512_good.is_some());
}

#[test]
fn test_aligned_vec_realloc_growth() {
    use hermes_simd_core::align::Aligned;
    use hermes_simd_core::AlignedVec;

    // 1. Standard allocation path
    let mut vec: AlignedVec<i32, Aligned<32>> = AlignedVec::new();
    assert_eq!(vec.len(), 0);
    assert_eq!(vec.capacity(), 0);

    for i in 0..1000 {
        vec.push(i as i32);
    }
    assert_eq!(vec.len(), 1000);
    assert!(vec.capacity() >= 1000);
    for i in 0..1000 {
        assert_eq!(vec[i], i as i32);
    }

    // 2. NUMA allocation path
    let mut numa_vec: AlignedVec<i32, Aligned<32>> = AlignedVec::with_capacity_numa(2, 0);
    assert_eq!(numa_vec.len(), 0);
    for i in 0..500 {
        numa_vec.push((i * 2) as i32);
    }
    assert_eq!(numa_vec.len(), 500);
    for i in 0..500 {
        assert_eq!(numa_vec[i], (i * 2) as i32);
    }
}

#[test]
fn test_numa_realloc_on_node_direct() {
    use core::alloc::Layout;
    use hermes_simd_core::numa::{MnemosyneNumaAllocator, NumaAllocator};

    let allocator = MnemosyneNumaAllocator;
    let layout1 = Layout::from_size_align(16, 8).unwrap();
    let layout2 = Layout::from_size_align(64, 8).unwrap();

    unsafe {
        // 1. Allocate initial block
        let ptr1 = allocator.alloc_on_node(layout1, 0);
        assert!(!ptr1.is_null());

        // Fill with test pattern
        for i in 0..16 {
            *ptr1.add(i) = i as u8;
        }

        // 2. Reallocate to a larger size
        let ptr2 = allocator.realloc_on_node(ptr1, layout1, layout2, 0);
        assert!(!ptr2.is_null());

        // Verify elements are preserved
        for i in 0..16 {
            assert_eq!(*ptr2.add(i), i as u8);
        }

        // 3. Deallocate
        allocator.dealloc_on_node(ptr2, layout2, 0);
    }
}

#[test]
fn test_numa_locality_caching_correctness_and_invalidation() {
    use hermes_simd_core::align::Unaligned;
    use hermes_simd_core::numa::locality::{
        bump_alloc_generation, get_alloc_generation, verify_numa_locality,
    };
    use hermes_simd_core::AlignedVec;

    // 1. Check initial generation
    let gen_start = get_alloc_generation();

    // 2. Perform a check on a stack address (it will cache it)
    let data = [0u8; 1024];
    let is_local1 = verify_numa_locality(data.as_ptr(), data.len(), 0);

    // Second check on same address should hit the cache immediately (generation is same)
    let is_local2 = verify_numa_locality(data.as_ptr(), data.len(), 0);
    assert_eq!(is_local1, is_local2);

    // Subset check should also hit the cache
    let is_local_sub = verify_numa_locality(unsafe { data.as_ptr().add(10) }, 100, 0);
    assert_eq!(is_local1, is_local_sub);

    // 3. Allocate a vector - this should not bump the generation counter immediately
    let vec: AlignedVec<u8, Unaligned> = AlignedVec::with_capacity(10);
    let gen_after_alloc = get_alloc_generation();
    assert_eq!(gen_after_alloc, gen_start);

    // 4. Drop the vector - this must bump the generation counter (deallocation)
    drop(vec);
    let gen_after_drop = get_alloc_generation();
    assert!(gen_after_drop > gen_start);

    // 5. Manual bump and check invalidation
    bump_alloc_generation();
    let gen_after_bump = get_alloc_generation();
    assert!(gen_after_bump > gen_after_drop);

    // 6. Multi-threaded stress test for contention-free caching
    let mut handles = std::vec![];
    for _ in 0..8 {
        handles.push(std::thread::spawn(move || {
            let local_data = [0u8; 512];
            for _ in 0..100 {
                let _ = verify_numa_locality(local_data.as_ptr(), local_data.len(), 0);
            }
        }));
    }
    for h in handles {
        h.join().unwrap();
    }
}
