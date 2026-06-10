use hermes_simd::{
    interleaved_complex_mul_assign, interleaved_complex_mul_assign_runtime, PreferredArch, Scalar,
};

fn reference_mul<const CONJ_B: bool>(a: &mut [f64], b: &[f64]) {
    for (x, y) in a.chunks_exact_mut(2).zip(b.chunks_exact(2)) {
        let ar = x[0];
        let ai = x[1];
        let br = y[0];
        let bi = if CONJ_B { -y[1] } else { y[1] };
        x[0] = ar * br - ai * bi;
        x[1] = ar * bi + ai * br;
    }
}

#[test]
fn interleaved_complex_mul_assign_f64_matches_scalar_reference() {
    for &complex_len in &[0usize, 1, 2, 3, 4, 7, 8, 17, 64] {
        let mut data: Vec<f64> = (0..complex_len * 2)
            .map(|i| (i as f64 * 0.25) - 3.0)
            .collect();
        let rhs: Vec<f64> = (0..complex_len * 2)
            .map(|i| (i as f64 % 5.0) - 2.0)
            .collect();
        let mut expected = data.clone();
        reference_mul::<false>(&mut expected, &rhs);

        interleaved_complex_mul_assign::<f64, PreferredArch, false>(&mut data, &rhs).unwrap();

        assert_eq!(data, expected, "complex_len={complex_len}");
    }
}

#[test]
fn interleaved_complex_mul_assign_conjugated_f32_matches_scalar_reference() {
    for &complex_len in &[0usize, 1, 2, 3, 4, 9, 16, 65] {
        let mut data: Vec<f32> = (0..complex_len * 2)
            .map(|i| (i as f32 * 0.125) - 2.0)
            .collect();
        let rhs: Vec<f32> = (0..complex_len * 2)
            .map(|i| (i as f32 % 7.0) - 3.0)
            .collect();
        let mut expected = data.clone();
        for (x, y) in expected.chunks_exact_mut(2).zip(rhs.chunks_exact(2)) {
            let ar = x[0];
            let ai = x[1];
            let br = y[0];
            let bi = -y[1];
            x[0] = ar * br - ai * bi;
            x[1] = ar * bi + ai * br;
        }

        interleaved_complex_mul_assign::<f32, PreferredArch, true>(&mut data, &rhs).unwrap();

        assert_eq!(data, expected, "complex_len={complex_len}");
    }
}

#[test]
fn interleaved_complex_mul_assign_scalar_architecture_matches_preferred() {
    let rhs = [2.0f64, -1.0, 0.5, 3.0, -2.0, 4.0];
    let mut preferred = [1.0f64, 2.0, -3.0, 4.0, 5.0, -6.0];
    let mut scalar = preferred;

    interleaved_complex_mul_assign::<f64, PreferredArch, true>(&mut preferred, &rhs).unwrap();
    interleaved_complex_mul_assign::<f64, Scalar, true>(&mut scalar, &rhs).unwrap();

    assert_eq!(preferred, scalar);
}

#[test]
fn interleaved_complex_mul_assign_runtime_matches_provider_architecture() {
    for &complex_len in &[1usize, 2, 5, 16, 65] {
        let rhs: Vec<f64> = (0..complex_len * 2)
            .map(|i| (i as f64 % 11.0) - 5.0)
            .collect();
        let mut runtime: Vec<f64> = (0..complex_len * 2)
            .map(|i| (i as f64 * 0.375) - 4.0)
            .collect();
        let mut expected = runtime.clone();

        interleaved_complex_mul_assign_runtime::<f64, false>(&mut runtime, &rhs).unwrap();
        interleaved_complex_mul_assign::<f64, PreferredArch, false>(&mut expected, &rhs).unwrap();

        assert_eq!(runtime, expected, "complex_len={complex_len}");
    }
}

#[test]
fn interleaved_complex_mul_assign_rejects_invalid_shapes() {
    let mut odd = [1.0f32, 2.0, 3.0];
    assert!(interleaved_complex_mul_assign::<f32, PreferredArch, false>(
        &mut odd,
        &[1.0, 2.0, 3.0]
    )
    .is_err());

    let mut lhs = [1.0f32, 2.0];
    assert!(interleaved_complex_mul_assign::<f32, PreferredArch, false>(&mut lhs, &[1.0]).is_err());
}
