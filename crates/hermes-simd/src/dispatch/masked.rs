//! Runtime-dispatched masked SIMD operations.

use hermes_simd_core::arch::SimdArch;
use hermes_simd_core::kernel::SimdKernel;
use hermes_simd_core::{view::SimdError, Scalar as ScalarTrait};
use hermes_simd_macros::runtime_dispatch;

// ---------------------------------------------------------------------------
// Internal generic kernel implementations
// ---------------------------------------------------------------------------

/// Generic masked sum using `leading_k_mask` for the scalar tail.
#[inline]
unsafe fn masked_sum_impl<T, Arch>(data: &[T], bool_mask: &[bool]) -> T
where
    T: ScalarTrait,
    Arch: SimdArch + SimdKernel<T>,
{
    assert_eq!(
        data.len(),
        bool_mask.len(),
        "data and mask lengths must match"
    );
    let len = data.len();
    let lane_count = Arch::LANE_COUNT;
    let simd_len = (len / lane_count) * lane_count;

    let mut total = T::ZERO;
    let mut i = 0usize;

    while i < simd_len {
        let v = Arch::load_unaligned(data.as_ptr().add(i));
        let msk = Arch::mask_from_bools(&bool_mask[i..i + lane_count]);
        total += Arch::masked_sum_reduce(v, msk);
        i += lane_count;
    }

    // Scalar tail
    while i < len {
        if bool_mask[i] {
            total += data[i];
        }
        i += 1;
    }

    total
}

/// Generic masked elementwise add: `out[i] = if mask[i] { a[i] + b[i] } else { a[i] }`.
#[inline]
unsafe fn masked_add_impl<T, Arch>(
    a: &[T],
    b: &[T],
    bool_mask: &[bool],
    out: &mut [T],
) -> Result<(), SimdError>
where
    T: ScalarTrait,
    Arch: SimdArch + SimdKernel<T>,
{
    if a.len() != b.len() || a.len() != bool_mask.len() || a.len() > out.len() {
        return Err(SimdError::LengthMismatch);
    }
    let len = a.len();
    let lane_count = Arch::LANE_COUNT;
    let simd_len = (len / lane_count) * lane_count;

    let mut i = 0usize;
    while i < simd_len {
        let va = Arch::load_unaligned(a.as_ptr().add(i));
        let vb = Arch::load_unaligned(b.as_ptr().add(i));
        let msk = Arch::mask_from_bools(&bool_mask[i..i + lane_count]);
        let src = va;
        let result = Arch::masked_add(va, vb, msk, src);
        Arch::store_unaligned(out.as_mut_ptr().add(i), result);
        i += lane_count;
    }

    // Scalar tail
    while i < len {
        out[i] = if bool_mask[i] { a[i] + b[i] } else { a[i] };
        i += 1;
    }

    Ok(())
}

/// Generic masked dot product: sum of `a[i] * b[i]` where `mask[i]`.
#[inline]
unsafe fn masked_dot_impl<T, Arch>(a: &[T], b: &[T], bool_mask: &[bool]) -> Result<T, SimdError>
where
    T: ScalarTrait,
    Arch: SimdArch + SimdKernel<T>,
{
    if a.len() != b.len() || a.len() != bool_mask.len() {
        return Err(SimdError::LengthMismatch);
    }
    let len = a.len();
    let lane_count = Arch::LANE_COUNT;
    let simd_len = (len / lane_count) * lane_count;

    let mut acc = Arch::zero();
    let mut i = 0usize;

    while i < simd_len {
        let va = Arch::load_unaligned(a.as_ptr().add(i));
        let vb = Arch::load_unaligned(b.as_ptr().add(i));
        let msk = Arch::mask_from_bools(&bool_mask[i..i + lane_count]);
        acc = Arch::masked_fmadd(va, vb, acc, msk);
        i += lane_count;
    }

    let mut total = Arch::sum_reduce(acc);

    // Scalar tail
    while i < len {
        if bool_mask[i] {
            total += a[i] * b[i];
        }
        i += 1;
    }

    Ok(total)
}

// ---------------------------------------------------------------------------
// Public dispatcher functions
// ---------------------------------------------------------------------------

#[runtime_dispatch(avx512f, avx2, neon, scalar)]
pub(super) fn dispatch_masked_sum_kernel<T, A>(data: &[T], mask: &[bool]) -> T
where
    T: ScalarTrait,
    A: SimdArch + SimdKernel<T>,
{
    unsafe { masked_sum_impl::<T, A>(data, mask) }
}

#[runtime_dispatch(avx512f, avx2, neon, scalar)]
pub(super) fn dispatch_masked_dot_kernel<T, A>(
    a: &[T],
    b: &[T],
    mask: &[bool],
) -> Result<T, SimdError>
where
    T: ScalarTrait,
    A: SimdArch + SimdKernel<T>,
{
    unsafe { masked_dot_impl::<T, A>(a, b, mask) }
}

#[runtime_dispatch(avx512f, avx2, neon, scalar)]
pub(super) fn dispatch_masked_add_kernel<T, A>(
    a: &[T],
    b: &[T],
    mask: &[bool],
    out: &mut [T],
) -> Result<(), SimdError>
where
    T: ScalarTrait,
    A: SimdArch + SimdKernel<T>,
{
    unsafe { masked_add_impl::<T, A>(a, b, mask, out) }
}
