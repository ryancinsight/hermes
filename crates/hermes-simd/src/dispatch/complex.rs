//! Monomorphized interleaved complex kernels.
//!
//! The public surface accepts primitive lane slices in `[re, im, ...]` order so
//! domain crates can use their own complex storage types without Hermes taking a
//! dependency on a concrete complex-number crate.

use hermes_simd_core::{arch::SimdArch, kernel::SimdKernel, scalar::Scalar, view::SimdError};

const MAX_STACK_LANES: usize = 128;

#[inline]
fn mul_pair<T, const CONJ_B: bool>(ar: T, ai: T, br: T, bi: T) -> (T, T)
where
    T: Scalar,
{
    let bi = if CONJ_B { -bi } else { bi };
    (ar * br - ai * bi, ar * bi + ai * br)
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
