//! Monomorphized interleaved complex kernels.
//!
//! The public surface accepts primitive lane slices in `[re, im, ...]` order so
//! domain crates can use their own complex storage types without Hermes taking a
//! dependency on a concrete complex-number crate.

use hermes_simd_core::{arch::SimdArch, kernel::SimdKernel, scalar::Scalar, view::SimdError};
use hermes_simd_types::PreferredArch;

const MAX_STACK_LANES: usize = 128;

#[inline]
fn mul_pair<T, const CONJ_B: bool>(ar: T, ai: T, br: T, bi: T) -> (T, T)
where
    T: Scalar,
{
    let bi = if CONJ_B { -bi } else { bi };
    (ar * br - ai * bi, ar * bi + ai * br)
}

/// Primitive lane types supported by the runtime-selected interleaved complex kernel.
pub trait InterleavedComplexLane: Scalar {
    /// Multiplies interleaved complex values in-place, selecting the fastest
    /// available provider implementation for this lane type.
    fn interleaved_complex_mul_assign_runtime<const CONJ_B: bool>(
        a: &mut [Self],
        b: &[Self],
    ) -> Result<(), SimdError>;

    /// Computes an interleaved complex dot product, selecting the fastest
    /// available provider implementation for this lane type.
    fn interleaved_complex_dot_runtime<const CONJ_B: bool>(
        a: &[Self],
        b: &[Self],
    ) -> Result<(Self, Self), SimdError>;
}

impl InterleavedComplexLane for f64 {
    #[inline]
    fn interleaved_complex_mul_assign_runtime<const CONJ_B: bool>(
        a: &mut [Self],
        b: &[Self],
    ) -> Result<(), SimdError> {
        if a.len() != b.len() || (a.len() & 1) != 0 {
            return Err(SimdError::LengthMismatch);
        }

        #[cfg(all(target_arch = "x86_64", feature = "std"))]
        {
            static HAS_FMA: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
            if *HAS_FMA.get_or_init(|| {
                std::is_x86_feature_detected!("avx") && std::is_x86_feature_detected!("fma")
            }) {
                // SAFETY: AVX and FMA are verified at runtime, and shape was
                // validated above.
                unsafe {
                    interleaved_complex_mul_assign_avx_fma_precise::<CONJ_B>(a, b);
                }
                return Ok(());
            }
        }

        interleaved_complex_mul_assign::<f64, PreferredArch, CONJ_B>(a, b)
    }

    #[inline]
    fn interleaved_complex_dot_runtime<const CONJ_B: bool>(
        a: &[Self],
        b: &[Self],
    ) -> Result<(Self, Self), SimdError> {
        interleaved_complex_dot::<f64, PreferredArch, CONJ_B>(a, b)
    }
}

impl InterleavedComplexLane for f32 {
    #[inline]
    fn interleaved_complex_mul_assign_runtime<const CONJ_B: bool>(
        a: &mut [Self],
        b: &[Self],
    ) -> Result<(), SimdError> {
        if a.len() != b.len() || (a.len() & 1) != 0 {
            return Err(SimdError::LengthMismatch);
        }

        #[cfg(all(target_arch = "x86_64", feature = "std"))]
        {
            static HAS_FMA: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
            if *HAS_FMA.get_or_init(|| {
                std::is_x86_feature_detected!("avx") && std::is_x86_feature_detected!("fma")
            }) {
                // SAFETY: AVX and FMA are verified at runtime, and shape was
                // validated above.
                unsafe {
                    interleaved_complex_mul_assign_avx_fma_reduced::<CONJ_B>(a, b);
                }
                return Ok(());
            }
        }

        interleaved_complex_mul_assign::<f32, PreferredArch, CONJ_B>(a, b)
    }

    #[inline]
    fn interleaved_complex_dot_runtime<const CONJ_B: bool>(
        a: &[Self],
        b: &[Self],
    ) -> Result<(Self, Self), SimdError> {
        interleaved_complex_dot::<f32, PreferredArch, CONJ_B>(a, b)
    }
}

/// Multiplies interleaved complex values in-place using architecture `A`.
///
/// `a` and `b` must have identical even lengths. The operation is value
/// preserving with respect to scalar complex multiplication over adjacent
/// primitive lane pairs:
///
/// `a[k] = a[k] * b[k]` when `CONJ_B == false`, and
/// `a[k] = a[k] * conj(b[k])` when `CONJ_B == true`.
#[inline]
pub fn interleaved_complex_mul_assign<T, A, const CONJ_B: bool>(
    a: &mut [T],
    b: &[T],
) -> Result<(), SimdError>
where
    T: Scalar,
    A: SimdArch + SimdKernel<T>,
{
    if a.len() != b.len() || (a.len() & 1) != 0 {
        return Err(SimdError::LengthMismatch);
    }

    let lanes = A::LANE_COUNT;
    assert!(
        lanes <= MAX_STACK_LANES,
        "SIMD lane count exceeds stack buffer"
    );
    let pair_lanes = lanes & !1;
    let mut offset = 0usize;

    while pair_lanes > 0 && offset + lanes <= a.len() {
        let mut ax = [T::ZERO; MAX_STACK_LANES];
        let mut bx = [T::ZERO; MAX_STACK_LANES];
        // SAFETY: offset + lanes <= len was checked above; `a` and `b` are
        // valid for reads of `lanes` primitive values.
        unsafe {
            A::store_unaligned(ax.as_mut_ptr(), A::load_unaligned(a.as_ptr().add(offset)));
            A::store_unaligned(bx.as_mut_ptr(), A::load_unaligned(b.as_ptr().add(offset)));
        }

        let mut lane = 0usize;
        while lane < pair_lanes {
            let (re, im) = mul_pair::<T, CONJ_B>(ax[lane], ax[lane + 1], bx[lane], bx[lane + 1]);
            ax[lane] = re;
            ax[lane + 1] = im;
            lane += 2;
        }

        // SAFETY: offset + lanes <= len was checked above; `a` is valid for
        // writes of `lanes` primitive values.
        unsafe {
            A::store_unaligned(a.as_mut_ptr().add(offset), A::load_unaligned(ax.as_ptr()));
        }
        offset += pair_lanes;
    }

    let mut lane = offset;
    while lane < a.len() {
        let (re, im) = mul_pair::<T, CONJ_B>(a[lane], a[lane + 1], b[lane], b[lane + 1]);
        a[lane] = re;
        a[lane + 1] = im;
        lane += 2;
    }

    Ok(())
}

/// Computes an interleaved complex dot product using architecture `A`.
///
/// Inputs must have identical even lengths in `[re0, im0, re1, im1, ...]`
/// primitive lane order. The returned tuple is `(re, im)` for
/// `sum(a[k] * b[k])`; when `CONJ_B` is true, the operation is
/// `sum(a[k] * conj(b[k]))`.
#[inline]
pub fn interleaved_complex_dot<T, A, const CONJ_B: bool>(
    a: &[T],
    b: &[T],
) -> Result<(T, T), SimdError>
where
    T: Scalar,
    A: SimdArch + SimdKernel<T>,
{
    if a.len() != b.len() || (a.len() & 1) != 0 {
        return Err(SimdError::LengthMismatch);
    }

    let lanes = A::LANE_COUNT;
    assert!(
        lanes <= MAX_STACK_LANES,
        "SIMD lane count exceeds stack buffer"
    );
    let pair_lanes = lanes & !1;
    let mut offset = 0usize;
    let mut re = T::ZERO;
    let mut im = T::ZERO;

    while pair_lanes > 0 && offset + lanes <= a.len() {
        let mut ax = [T::ZERO; MAX_STACK_LANES];
        let mut bx = [T::ZERO; MAX_STACK_LANES];
        // SAFETY: offset + lanes <= len was checked above; `a` and `b` are
        // valid for reads of `lanes` primitive values.
        unsafe {
            A::store_unaligned(ax.as_mut_ptr(), A::load_unaligned(a.as_ptr().add(offset)));
            A::store_unaligned(bx.as_mut_ptr(), A::load_unaligned(b.as_ptr().add(offset)));
        }

        let mut lane = 0usize;
        while lane < pair_lanes {
            let (prod_re, prod_im) =
                mul_pair::<T, CONJ_B>(ax[lane], ax[lane + 1], bx[lane], bx[lane + 1]);
            re = re + prod_re;
            im = im + prod_im;
            lane += 2;
        }
        offset += pair_lanes;
    }

    let mut lane = offset;
    while lane < a.len() {
        let (prod_re, prod_im) = mul_pair::<T, CONJ_B>(a[lane], a[lane + 1], b[lane], b[lane + 1]);
        re = re + prod_re;
        im = im + prod_im;
        lane += 2;
    }

    Ok((re, im))
}

/// Multiplies interleaved complex values in-place using Hermes runtime provider selection.
///
/// This is the high-level provider API for callers that know the lane type but
/// do not want to carry their own architecture detection. Shape requirements
/// are identical to [`interleaved_complex_mul_assign`].
#[inline]
pub fn interleaved_complex_mul_assign_runtime<T, const CONJ_B: bool>(
    a: &mut [T],
    b: &[T],
) -> Result<(), SimdError>
where
    T: InterleavedComplexLane,
{
    T::interleaved_complex_mul_assign_runtime::<CONJ_B>(a, b)
}

/// Computes an interleaved complex dot product using Hermes runtime provider selection.
///
/// Shape requirements and complex lane ordering are identical to
/// [`interleaved_complex_dot`].
#[inline]
pub fn interleaved_complex_dot_runtime<T, const CONJ_B: bool>(
    a: &[T],
    b: &[T],
) -> Result<(T, T), SimdError>
where
    T: InterleavedComplexLane,
{
    T::interleaved_complex_dot_runtime::<CONJ_B>(a, b)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,fma")]
unsafe fn interleaved_complex_mul_assign_avx_fma_precise<const CONJ_B: bool>(
    a: &mut [f64],
    b: &[f64],
) {
    use std::arch::x86_64::{
        _mm256_fmaddsub_pd, _mm256_loadu_pd, _mm256_mul_pd, _mm256_permute_pd, _mm256_set_pd,
        _mm256_setzero_pd, _mm256_storeu_pd, _mm256_unpackhi_pd, _mm256_unpacklo_pd, _mm256_xor_pd,
    };

    let sign_mask = if CONJ_B {
        _mm256_set_pd(-0.0_f64, 0.0_f64, -0.0_f64, 0.0_f64)
    } else {
        _mm256_setzero_pd()
    };
    let batches = a.len() / 4;
    let a_ptr = a.as_mut_ptr();
    let b_ptr = b.as_ptr();
    for i in 0..batches {
        let off = i * 4;
        let av = _mm256_loadu_pd(a_ptr.add(off));
        let bv = _mm256_xor_pd(_mm256_loadu_pd(b_ptr.add(off)), sign_mask);
        let a_re = _mm256_unpacklo_pd(av, av);
        let a_im = _mm256_unpackhi_pd(av, av);
        let b_sw = _mm256_permute_pd(bv, 0b0101);
        let prod = _mm256_mul_pd(a_im, b_sw);
        let res = _mm256_fmaddsub_pd(a_re, bv, prod);
        _mm256_storeu_pd(a_ptr.add(off), res);
    }

    let mut lane = batches * 4;
    while lane < a.len() {
        let (re, im) = mul_pair::<f64, CONJ_B>(a[lane], a[lane + 1], b[lane], b[lane + 1]);
        a[lane] = re;
        a[lane + 1] = im;
        lane += 2;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,fma")]
unsafe fn interleaved_complex_mul_assign_avx_fma_reduced<const CONJ_B: bool>(
    a: &mut [f32],
    b: &[f32],
) {
    use std::arch::x86_64::{
        _mm256_fmaddsub_ps, _mm256_loadu_ps, _mm256_movehdup_ps, _mm256_moveldup_ps, _mm256_mul_ps,
        _mm256_permute_ps, _mm256_set_ps, _mm256_setzero_ps, _mm256_storeu_ps, _mm256_xor_ps,
    };

    let sign_mask = if CONJ_B {
        _mm256_set_ps(
            -0.0_f32, 0.0_f32, -0.0_f32, 0.0_f32, -0.0_f32, 0.0_f32, -0.0_f32, 0.0_f32,
        )
    } else {
        _mm256_setzero_ps()
    };
    let batches = a.len() / 8;
    let a_ptr = a.as_mut_ptr();
    let b_ptr = b.as_ptr();
    for i in 0..batches {
        let off = i * 8;
        let av = _mm256_loadu_ps(a_ptr.add(off));
        let bv = _mm256_xor_ps(_mm256_loadu_ps(b_ptr.add(off)), sign_mask);
        let a_re = _mm256_moveldup_ps(av);
        let a_im = _mm256_movehdup_ps(av);
        let b_sw = _mm256_permute_ps(bv, 0b1011_0001);
        let prod = _mm256_mul_ps(a_im, b_sw);
        let res = _mm256_fmaddsub_ps(a_re, bv, prod);
        _mm256_storeu_ps(a_ptr.add(off), res);
    }

    let mut lane = batches * 8;
    while lane < a.len() {
        let (re, im) = mul_pair::<f32, CONJ_B>(a[lane], a[lane + 1], b[lane], b[lane + 1]);
        a[lane] = re;
        a[lane + 1] = im;
        lane += 2;
    }
}
