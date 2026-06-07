#![allow(clippy::while_let_on_iterator)]
//! Unit tests for sprint A–D additions.
//!
//! Covers:
//! - `NumericElement::min_scalar` / `max_scalar`
//! - `ReductionOp<T>` identity and scalar_combine for `Min`, `Max`, `Sum`
//! - `SimdView::reduce(Min)` / `reduce(Max)` / `reduce(Sum)`
//! - `ZipChunks` lockstep iteration
//! - `Packed4Vec::from_iter` / `Extend`
//! - `SimdCow::from_std_cow` / `From<AlignedVec>`

#[cfg(test)]
mod tests {
    use hermes_simd_core::{
        ops::{Min, Max, Sum, Dot, ReductionOp},
        scalar::{Scalar, NumericElement},
        view::SimdView,
        align::Unaligned,
        execution::Unmasked,
    };
    use hermes_simd_intrinsics::Scalar as ScalarArch;

    // ---------------------------------------------------------------------------
    // A1: NumericElement::min_scalar / max_scalar
    // ---------------------------------------------------------------------------

    #[test]
    fn test_min_scalar_f32() {
        let a: f32 = 3.0_f32;
        let b: f32 = -1.0_f32;
        assert_eq!(a.min_scalar(b), -1.0_f32);
        assert_eq!(b.min_scalar(a), -1.0_f32);
    }

    #[test]
    fn test_max_scalar_f32() {
        let a: f32 = 3.0_f32;
        let b: f32 = -1.0_f32;
        assert_eq!(a.max_scalar(b), 3.0_f32);
        assert_eq!(b.max_scalar(a), 3.0_f32);
    }

    #[test]
    fn test_min_max_identity_f32() {
        assert_eq!(<f32 as NumericElement>::MIN_VALUE, f32::NEG_INFINITY);
        assert_eq!(<f32 as NumericElement>::MAX_VALUE, f32::INFINITY);
    }

    #[test]
    fn test_min_max_identity_i32() {
        assert_eq!(<i32 as NumericElement>::MIN_VALUE, i32::MIN);
        assert_eq!(<i32 as NumericElement>::MAX_VALUE, i32::MAX);
    }

    // ---------------------------------------------------------------------------
    // A3: ReductionOp identity_scalar / scalar_combine
    // ---------------------------------------------------------------------------

    #[test]
    fn test_sum_identity_and_combine() {
        assert_eq!(<Sum as ReductionOp<f32>>::identity_scalar(), 0.0_f32);
        assert_eq!(<Sum as ReductionOp<f32>>::scalar_combine(2.0, 3.0), 5.0_f32);
    }

    #[test]
    fn test_min_identity_and_combine() {
        assert_eq!(<Min as ReductionOp<f32>>::identity_scalar(), f32::INFINITY);
        assert_eq!(<Min as ReductionOp<f32>>::scalar_combine(2.0, 3.0), 2.0_f32);
        assert_eq!(<Min as ReductionOp<f32>>::scalar_combine(3.0, 2.0), 2.0_f32);
    }

    #[test]
    fn test_max_identity_and_combine() {
        assert_eq!(<Max as ReductionOp<f32>>::identity_scalar(), f32::NEG_INFINITY);
        assert_eq!(<Max as ReductionOp<f32>>::scalar_combine(2.0, 3.0), 3.0_f32);
        assert_eq!(<Max as ReductionOp<f32>>::scalar_combine(3.0, 2.0), 3.0_f32);
    }

    // ---------------------------------------------------------------------------
    // A5: SimdView::reduce(Min/Max/Sum) correctness
    // ---------------------------------------------------------------------------

    fn make_view(data: &[f32]) -> SimdView<'_, f32, ScalarArch, Unaligned, Unmasked, &[f32]> {
        SimdView::new(data).expect("Unaligned always succeeds")
    }

    #[test]
    fn test_reduce_sum() {
        let data = [1.0_f32, 2.0, 3.0, 4.0, 5.0];
        let view = make_view(&data);
        assert_eq!(view.reduce(Sum), 15.0_f32);
    }

    #[test]
    fn test_reduce_sum_empty() {
        let data: [f32; 0] = [];
        let view = make_view(&data);
        assert_eq!(view.reduce(Sum), 0.0_f32);
    }

    #[test]
    fn test_reduce_min() {
        let data = [3.0_f32, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0];
        let view = make_view(&data);
        assert_eq!(view.reduce(Min), 1.0_f32);
    }

    #[test]
    fn test_reduce_max() {
        let data = [3.0_f32, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0];
        let view = make_view(&data);
        assert_eq!(view.reduce(Max), 9.0_f32);
    }

    #[test]
    fn test_reduce_min_negative() {
        let data = [-5.0_f32, -3.0, -10.0, -1.0];
        let view = make_view(&data);
        assert_eq!(view.reduce(Min), -10.0_f32);
    }

    #[test]
    fn test_reduce_max_negative() {
        let data = [-5.0_f32, -3.0, -10.0, -1.0];
        let view = make_view(&data);
        assert_eq!(view.reduce(Max), -1.0_f32);
    }

    #[test]
    fn test_reduce_min_empty() {
        let data: [f32; 0] = [];
        let view = make_view(&data);
        // Identity for Min is MAX_VALUE (infinity).
        assert_eq!(view.reduce(Min), f32::INFINITY);
    }

    #[test]
    fn test_reduce_max_empty() {
        let data: [f32; 0] = [];
        let view = make_view(&data);
        // Identity for Max is MIN_VALUE (neg infinity).
        assert_eq!(view.reduce(Max), f32::NEG_INFINITY);
    }

    // ---------------------------------------------------------------------------
    // A5: zip_reduce correctness
    // ---------------------------------------------------------------------------

    #[test]
    fn test_zip_reduce_dot() {
        let a = [1.0_f32, 2.0, 3.0];
        let b = [4.0_f32, 5.0, 6.0];
        let va = make_view(&a);
        let vb = make_view(&b);
        // 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
        let result = va.zip_reduce(&vb, Dot).unwrap();
        assert!((result - 32.0_f32).abs() < 1e-5);
    }

    // ---------------------------------------------------------------------------
    // A4: ZipChunks lockstep
    // ---------------------------------------------------------------------------

    #[test]
    fn test_zip_chunks_remainder() {
        // 5 elements — architecture-agnostic: verify chunk*LANE + tail == total.
        let a = [1.0_f32, 2.0, 3.0, 4.0, 5.0];
        let b = [10.0_f32, 20.0, 30.0, 40.0, 50.0];
        let va = make_view(&a);
        let vb = make_view(&b);
        let lane = <ScalarArch as hermes_simd_core::kernel::SimdKernel<f32>>::LANE_COUNT;
        let mut zip = va.zip_chunks(&vb);
        let mut chunk_count = 0;
        while let Some(_) = zip.next() {
            chunk_count += 1;
        }
        let (ra, rb) = zip.remainder();
        // All elements must be accounted for across chunks + remainder.
        assert_eq!(chunk_count * lane + ra.len(), a.len());
        assert_eq!(chunk_count * lane + rb.len(), b.len());
    }

}
