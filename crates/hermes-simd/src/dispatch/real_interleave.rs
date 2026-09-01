//! Fused real multiplication and interleaved-complex materialization.

use hermes_simd_core::{arch::SimdArch, kernel::SimdKernel, scalar::Scalar, view::SimdError};
use hermes_simd_macros::runtime_dispatch;

use super::SimdOps;

/// Multiplies two real slices and writes interleaved complex lanes using
/// architecture `A`.
///
/// For every input lane `i`, this writes
/// `output[2 * i..2 * i + 2] = [input[i] * factors[i], T::ZERO]`. The
/// operation allocates no memory and validates every length before writing.
///
/// # Errors
///
/// Returns [`SimdError::LengthMismatch`] unless `input` and `factors` have
/// equal lengths and `output.len() == 2 * input.len()`. An overflowing output
/// length is also rejected without mutation.
///
/// # Examples
///
/// ```
/// use hermes_simd::{real_mul_to_interleaved_complex, Scalar};
///
/// let input = [1.0_f64, -2.0, 3.0];
/// let factors = [0.5_f64, 4.0, -2.0];
/// let mut output = [0.0_f64; 6];
///
/// real_mul_to_interleaved_complex::<f64, Scalar>(&input, &factors, &mut output).unwrap();
/// assert_eq!(output, [0.5, 0.0, -8.0, 0.0, -6.0, 0.0]);
/// ```
#[inline]
pub fn real_mul_to_interleaved_complex<T, A>(
    input: &[T],
    factors: &[T],
    output: &mut [T],
) -> Result<(), SimdError>
where
    T: Scalar,
    A: SimdArch + SimdKernel<T>,
{
    let Some(required_output_len) = input.len().checked_mul(2) else {
        return Err(SimdError::LengthMismatch);
    };
    if input.len() != factors.len() || output.len() != required_output_len {
        return Err(SimdError::LengthMismatch);
    }

    let lanes = A::LANE_COUNT;
    let mut offset = 0usize;
    while input.len() - offset >= lanes {
        let output_offset = offset * 2;
        // SAFETY: the length checks above prove both inputs contain `lanes`
        // values at `offset` and the output contains two complete registers at
        // `output_offset`. Runtime or explicit architecture selection proves
        // that `A`'s target features are available to its caller.
        unsafe {
            let samples = A::load_unaligned(input.as_ptr().add(offset));
            let window = A::load_unaligned(factors.as_ptr().add(offset));
            let products = A::mul(samples, window);
            let (low, high) = A::interleave(products, A::zero());
            A::store_unaligned(output.as_mut_ptr().add(output_offset), low);
            A::store_unaligned(output.as_mut_ptr().add(output_offset + lanes), high);
        }
        offset += lanes;
    }

    for lane in offset..input.len() {
        let output_lane = lane * 2;
        output[output_lane] = input[lane] * factors[lane];
        output[output_lane + 1] = T::ZERO;
    }

    Ok(())
}

#[runtime_dispatch(avx512f, avx2, neon, scalar)]
pub(super) fn dispatch_real_mul_to_interleaved_complex_impl<T, A>(
    input: &[T],
    factors: &[T],
    output: &mut [T],
) -> Result<(), SimdError>
where
    T: Scalar,
    A: SimdArch + SimdKernel<T>,
{
    real_mul_to_interleaved_complex::<T, A>(input, factors, output)
}

/// Multiplies two real slices and writes interleaved complex lanes using the
/// widest supported runtime SIMD backend.
///
/// The operation allocates no memory and writes
/// `[input[i] * factors[i], T::ZERO]` for each input lane.
///
/// # Errors
///
/// Returns [`SimdError::LengthMismatch`] unless `input` and `factors` have
/// equal lengths and `output.len() == 2 * input.len()`. An overflowing output
/// length is also rejected without mutation.
#[inline]
pub fn real_mul_to_interleaved_complex_runtime<T>(
    input: &[T],
    factors: &[T],
    output: &mut [T],
) -> Result<(), SimdError>
where
    T: SimdOps,
{
    T::real_mul_to_interleaved_complex(input, factors, output)
}
