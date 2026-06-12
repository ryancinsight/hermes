//! Generic runtime-dispatch AXPY kernel: `out[i] += alpha * x[i]`.
//!
//! The fused row-update primitive matmul-style consumers (leto) accumulate
//! with: one broadcast multiplier, a streaming read of `x`, and an in-place
//! read-modify-write of `out`, with no temporary allocation. SIMD chunks use
//! the fused multiply-add primitive; the scalar tail is covered
//! element-by-element.

use hermes_simd_core::{arch::SimdArch, kernel::SimdKernel, scalar::Scalar, view::SimdError};
use hermes_simd_macros::runtime_dispatch;

#[runtime_dispatch(avx512f, avx2, neon, scalar)]
pub(super) fn dispatch_axpy_kernel<T, A>(alpha: T, x: &[T], out: &mut [T]) -> Result<(), SimdError>
where
    T: Scalar,
    A: SimdArch + SimdKernel<T>,
{
    if x.len() != out.len() {
        return Err(SimdError::LengthMismatch);
    }
    let len = out.len();
    if len == 0 {
        return Ok(());
    }

    let lane_count = A::LANE_COUNT;
    let simd_len = (len / lane_count) * lane_count;
    let x_ptr = x.as_ptr();
    let out_ptr = out.as_mut_ptr();

    // SAFETY: `simd_len <= len` bounds every vector load/store inside both
    // equal-length slices; lanes are processed in disjoint `lane_count`
    // chunks of the same index range, so the read-modify-write of `out`
    // never overlaps a pending store.
    unsafe {
        let valpha = A::splat(alpha);
        let mut i = 0usize;
        while i < simd_len {
            let vx = A::load_unaligned(x_ptr.add(i));
            let vo = A::load_unaligned(out_ptr.add(i));
            A::store_unaligned(out_ptr.add(i), A::fmadd(vx, valpha, vo));
            i += lane_count;
        }
    }

    // Scalar tail.
    for i in simd_len..len {
        out[i] = out[i] + x[i] * alpha;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::super::axpy;

    #[test]
    fn axpy_matches_scalar_reference_across_tail_sizes() {
        // Sizes straddle every lane width incl. partial tails.
        for &len in &[0usize, 1, 3, 7, 8, 9, 15, 16, 17, 63, 64, 65, 1027] {
            let alpha = 1.75f64;
            let x: Vec<f64> = (0..len).map(|i| i as f64 * 0.5 - 3.0).collect();
            let mut out: Vec<f64> = (0..len).map(|i| 100.0 - i as f64).collect();
            let expected: Vec<f64> = out.iter().zip(&x).map(|(o, xv)| o + alpha * xv).collect();

            axpy(alpha, &x, &mut out).unwrap();
            assert_eq!(out, expected, "len {len}");
        }
    }

    #[test]
    fn axpy_single_precision_matches_reference() {
        let alpha = -0.5f32;
        let x: Vec<f32> = (0..133).map(|i| i as f32).collect();
        let mut out = vec![1.0f32; 133];
        let expected: Vec<f32> = x.iter().map(|xv| 1.0 + alpha * xv).collect();
        axpy(alpha, &x, &mut out).unwrap();
        assert_eq!(out, expected);
    }

    #[test]
    fn axpy_rejects_length_mismatch() {
        let x = [1.0f64, 2.0];
        let mut out = [0.0f64; 3];
        assert!(axpy(1.0, &x, &mut out).is_err());
    }

    #[test]
    fn axpy_zero_alpha_is_identity() {
        let x: Vec<f64> = (0..50).map(|i| i as f64).collect();
        let mut out: Vec<f64> = (0..50).map(|i| i as f64 * 2.0).collect();
        let expected = out.clone();
        axpy(0.0, &x, &mut out).unwrap();
        assert_eq!(out, expected);
    }
}
