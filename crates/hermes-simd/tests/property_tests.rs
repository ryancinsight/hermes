use proptest::prelude::*;
use hermes_simd::{sum, dot, elementwise_mul};

fn ref_sum<T: std::iter::Sum + Copy>(data: &[T]) -> T {
    data.iter().copied().sum()
}

fn ref_dot<T: std::ops::Mul<Output = T> + std::iter::Sum + Copy>(a: &[T], b: &[T]) -> T {
    a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum()
}

fn ref_elementwise_mul<T: std::ops::Mul<Output = T> + Copy>(a: &[T], b: &[T], out: &mut [T]) {
    for i in 0..a.len() {
        out[i] = a[i] * b[i];
    }
}

fn approx_eq_f32(x: f32, y: f32) -> bool {
    if x.is_nan() && y.is_nan() { return true; }
    if x.is_infinite() && y.is_infinite() { return x.is_sign_positive() == y.is_sign_positive(); }
    let abs_diff = (x - y).abs();
    if abs_diff < 1e-2 { return true; }
    let norm = x.abs().max(y.abs());
    norm > 0.0 && (abs_diff / norm) < 1e-2
}

fn approx_eq_f16(x: half::f16, y: half::f16) -> bool {
    let x_f32 = x.to_f32();
    let y_f32 = y.to_f32();
    if x_f32.is_nan() && y_f32.is_nan() { return true; }
    if x_f32.is_infinite() && y_f32.is_infinite() { return x_f32.is_sign_positive() == y_f32.is_sign_positive(); }
    let abs_diff = (x_f32 - y_f32).abs();
    if abs_diff < 6.0e-1 { return true; }
    let norm = x_f32.abs().max(y_f32.abs());
    norm > 0.0 && (abs_diff / norm) < 2.0e-1
}

fn approx_eq_f64(x: f64, y: f64) -> bool {
    if x.is_nan() && y.is_nan() { return true; }
    if x.is_infinite() && y.is_infinite() { return x.is_sign_positive() == y.is_sign_positive(); }
    let abs_diff = (x - y).abs();
    if abs_diff < 1e-7 { return true; }
    let norm = x.abs().max(y.abs());
    norm > 0.0 && (abs_diff / norm) < 1e-7
}

proptest! {
    #[test]
    fn prop_sum_f32(ref_data in prop::collection::vec(-1000.0f32..1000.0f32, 0..500)) {
        let actual = sum::<f32>(&ref_data);
        let expected = ref_sum::<f32>(&ref_data);
        prop_assert!(approx_eq_f32(actual, expected), "f32 sum mismatch: actual = {}, expected = {}", actual, expected);
    }

    #[test]
    fn prop_sum_f64(ref_data in prop::collection::vec(-1000.0f64..1000.0f64, 0..500)) {
        let actual = sum::<f64>(&ref_data);
        let expected = ref_sum::<f64>(&ref_data);
        prop_assert!(approx_eq_f64(actual, expected), "f64 sum mismatch: actual = {}, expected = {}", actual, expected);
    }

    #[test]
    fn prop_dot_f32(
        (a, b) in prop::collection::vec(-100.0f32..100.0f32, 0..500)
            .prop_flat_map(|v| {
                let len = v.len();
                (Just(v), prop::collection::vec(-100.0f32..100.0f32, len))
            })
    ) {
        let actual = dot::<f32>(&a, &b).unwrap();
        let expected = ref_dot::<f32>(&a, &b);
        prop_assert!(approx_eq_f32(actual, expected), "f32 dot mismatch: actual = {}, expected = {}", actual, expected);
    }

    #[test]
    fn prop_dot_f64(
        (a, b) in prop::collection::vec(-100.0f64..100.0f64, 0..500)
            .prop_flat_map(|v| {
                let len = v.len();
                (Just(v), prop::collection::vec(-100.0f64..100.0f64, len))
            })
    ) {
        let actual = dot::<f64>(&a, &b).unwrap();
        let expected = ref_dot::<f64>(&a, &b);
        prop_assert!(approx_eq_f64(actual, expected), "f64 dot mismatch: actual = {}, expected = {}", actual, expected);
    }

    #[test]
    fn prop_mul_f32(
        (a, b) in prop::collection::vec(-100.0f32..100.0f32, 0..500)
            .prop_flat_map(|v| {
                let len = v.len();
                (Just(v), prop::collection::vec(-100.0f32..100.0f32, len))
            })
    ) {
        let mut actual = vec![0.0f32; a.len()];
        let mut expected = vec![0.0f32; a.len()];
        elementwise_mul::<f32>(&a, &b, &mut actual).unwrap();
        ref_elementwise_mul::<f32>(&a, &b, &mut expected);
        for i in 0..a.len() {
            prop_assert!(approx_eq_f32(actual[i], expected[i]), "f32 mul mismatch at {}: actual = {}, expected = {}", i, actual[i], expected[i]);
        }
    }

    #[test]
    fn prop_mul_f64(
        (a, b) in prop::collection::vec(-100.0f64..100.0f64, 0..500)
            .prop_flat_map(|v| {
                let len = v.len();
                (Just(v), prop::collection::vec(-100.0f64..100.0f64, len))
            })
    ) {
        let mut actual = vec![0.0f64; a.len()];
        let mut expected = vec![0.0f64; a.len()];
        elementwise_mul::<f64>(&a, &b, &mut actual).unwrap();
        ref_elementwise_mul::<f64>(&a, &b, &mut expected);
        for i in 0..a.len() {
            prop_assert!(approx_eq_f64(actual[i], expected[i]), "f64 mul mismatch at {}: actual = {}, expected = {}", i, actual[i], expected[i]);
        }
    }

    #[test]
    fn prop_sum_f16(ref_data in prop::collection::vec(-10.0f32..10.0f32, 0..100)) {
        let f16_data: Vec<half::f16> = ref_data.iter().map(|&x| half::f16::from_f32(x)).collect();
        let actual = sum::<half::f16>(&f16_data);
        let expected = f16_data.iter().copied().fold(half::f16::ZERO, |acc, x| acc + x);
        prop_assert!(approx_eq_f16(actual, expected), "f16 sum mismatch: actual = {:?}, expected = {:?}", actual, expected);
    }

    #[test]
    fn prop_dot_f16(
        (a, b) in prop::collection::vec(-5.0f32..5.0f32, 0..100)
            .prop_flat_map(|v| {
                let len = v.len();
                (Just(v), prop::collection::vec(-5.0f32..5.0f32, len))
            })
    ) {
        let f16_a: Vec<half::f16> = a.iter().map(|&x| half::f16::from_f32(x)).collect();
        let f16_b: Vec<half::f16> = b.iter().map(|&x| half::f16::from_f32(x)).collect();
        let actual = dot::<half::f16>(&f16_a, &f16_b).unwrap();
        let expected = f16_a.iter().zip(f16_b.iter()).map(|(&x, &y)| x * y).fold(half::f16::ZERO, |acc, x| acc + x);
        prop_assert!(approx_eq_f16(actual, expected), "f16 dot mismatch: actual = {:?}, expected = {:?}", actual, expected);
    }

    #[test]
    fn prop_mul_f16(
        (a, b) in prop::collection::vec(-10.0f32..10.0f32, 0..100)
            .prop_flat_map(|v| {
                let len = v.len();
                (Just(v), prop::collection::vec(-10.0f32..10.0f32, len))
            })
    ) {
        let f16_a: Vec<half::f16> = a.iter().map(|&x| half::f16::from_f32(x)).collect();
        let f16_b: Vec<half::f16> = b.iter().map(|&x| half::f16::from_f32(x)).collect();
        let mut actual = vec![half::f16::ZERO; f16_a.len()];
        let mut expected = vec![half::f16::ZERO; f16_a.len()];
        elementwise_mul::<half::f16>(&f16_a, &f16_b, &mut actual).unwrap();
        for i in 0..f16_a.len() {
            expected[i] = f16_a[i] * f16_b[i];
        }
        for i in 0..f16_a.len() {
            prop_assert!(approx_eq_f16(actual[i], expected[i]), "f16 mul mismatch at {}: actual = {:?}, expected = {:?}", i, actual[i], expected[i]);
        }
    }
}