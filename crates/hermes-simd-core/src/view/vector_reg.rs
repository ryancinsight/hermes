//! Monomorphized SIMD vector register wrapper.
//!
//! # Safety
//!
//! Every operation here ultimately calls a `#[target_feature]`-gated
//! [`SimdKernel`](crate::kernel::SimdKernel) method, sound only on a host that
//! implements `Arch`. Safe constructors establish that capability once before
//! producing a [`Vector`]; unsafe constructors require it from their caller.
//! Because processor capabilities are process-wide, possession of a `Vector`
//! then discharges the target-feature obligation for operations on that value.
//! Raw-pointer loads and stores additionally require pointer validity and state
//! both obligations in their `# Safety` sections.
//!
//! Lane-count and lane-index preconditions (`from_array`, `extract`, `cast`, …)
//! are proven at compile time by the `AssertLaneCount`/`AssertLaneIndex` const
//! guards, so a mismatch fails the build rather than reading out of bounds.

use super::SimdError;
use crate::arch::SimdArch;
use crate::kernel::{SimdKernel, SimdStorage, MAX_SIMD_LANES};
use crate::scalar::Scalar;
use core::marker::PhantomData;

/// A monomorphized vector register type wrapping the architecture-native raw register.
#[repr(transparent)]
pub struct Vector<T, Arch>
where
    Arch: SimdArch + SimdKernel<T>,
    T: Scalar,
{
    /// The underlying raw vector register.
    pub raw: Arch::Vector,
    _marker: PhantomData<T>,
}

#[expect(
    clippy::expl_impl_clone_on_copy,
    reason = "The raw register capability is supplied by the architecture trait, so Clone stays explicit"
)]
impl<T, Arch> Clone for Vector<T, Arch>
where
    Arch: SimdArch + SimdKernel<T>,
    T: Scalar,
{
    #[inline(always)]
    fn clone(&self) -> Self {
        *self
    }
}

impl<T, Arch> Copy for Vector<T, Arch>
where
    Arch: SimdArch + SimdKernel<T>,
    T: Scalar,
{
}

impl<T, Arch> core::fmt::Debug for Vector<T, Arch>
where
    Arch: SimdArch + SimdKernel<T>,
    T: Scalar + core::fmt::Debug,
{
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        const { <Arch as SimdStorage<T>>::LANE_BOUND_CHECK };
        let lane_count = Arch::LANE_COUNT;
        let mut buf = [core::mem::MaybeUninit::<T>::uninit(); MAX_SIMD_LANES];
        // SAFETY: constructing `self` proved host support. The store writes exactly
        // `lane_count` elements into the `MAX_SIMD_LANES`-slot buffer (bounded by
        // `LANE_BOUND_CHECK`), so the `lane_count`-length slice reads only
        // initialized elements.
        unsafe {
            Arch::store_unaligned(buf.as_mut_ptr().cast::<T>(), self.raw);
            let init_slice = core::slice::from_raw_parts(buf.as_ptr().cast::<T>(), lane_count);
            f.debug_list().entries(init_slice).finish()
        }
    }
}

impl<T, Arch> PartialEq for Vector<T, Arch>
where
    Arch: SimdArch + SimdKernel<T>,
    T: Scalar + PartialEq,
{
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        const { <Arch as SimdStorage<T>>::LANE_BOUND_CHECK };
        let lane_count = Arch::LANE_COUNT;
        let mut buf_self = [core::mem::MaybeUninit::<T>::uninit(); MAX_SIMD_LANES];
        let mut buf_other = [core::mem::MaybeUninit::<T>::uninit(); MAX_SIMD_LANES];
        // SAFETY: constructing both vectors proved host support. Each store writes `lane_count`
        // elements into its buffer, so both `lane_count`-length slices read only
        // initialized elements.
        unsafe {
            Arch::store_unaligned(buf_self.as_mut_ptr().cast::<T>(), self.raw);
            Arch::store_unaligned(buf_other.as_mut_ptr().cast::<T>(), other.raw);
            let slice_self = core::slice::from_raw_parts(buf_self.as_ptr().cast::<T>(), lane_count);
            let slice_other =
                core::slice::from_raw_parts(buf_other.as_ptr().cast::<T>(), lane_count);
            slice_self == slice_other
        }
    }
}

impl<T, Arch> Eq for Vector<T, Arch>
where
    Arch: SimdArch + SimdKernel<T>,
    T: Scalar + Eq,
{
}

// Operation-family leaves. Each holds one family of inherent `Vector` methods
// behind the same impl header; no method is declared in this root.
mod arith;
mod compare;
mod construct;
mod lanes;
mod memory;
mod permute;
mod reduce;
mod view_bridge;

#[inline(always)]
fn is_vector_aligned<T, Arch>(ptr: *const T) -> bool
where
    Arch: SimdArch + SimdKernel<T>,
    T: Scalar,
{
    let alignment = Arch::LANE_COUNT * core::mem::size_of::<T>();
    alignment != 0 && (ptr as usize).is_multiple_of(alignment)
}

#[inline(always)]
pub(crate) fn runtime_support_result<T, Arch>() -> Result<(), SimdError>
where
    Arch: SimdArch + SimdKernel<T>,
    T: Scalar,
{
    if Arch::is_runtime_supported() {
        Ok(())
    } else {
        Err(SimdError::UnsupportedTarget)
    }
}

#[inline(always)]
pub(crate) fn assert_runtime_supported<T, Arch>()
where
    Arch: SimdArch + SimdKernel<T>,
    T: Scalar,
{
    assert!(
        Arch::is_runtime_supported(),
        "SIMD target is not supported or enabled on this host"
    );
}

struct AssertLaneIndex<T, Arch, const I: usize>(PhantomData<(T, Arch)>);
impl<T, Arch, const I: usize> AssertLaneIndex<T, Arch, I>
where
    Arch: SimdArch + SimdKernel<T>,
    T: Scalar,
{
    const OK: () = {
        assert!(
            I < <Arch as SimdStorage<T>>::LANE_COUNT,
            "Lane index out of bounds"
        );
    };
}

struct AssertLaneCountSame<T, U, Arch>(PhantomData<(T, U, Arch)>);
impl<T, U, Arch> AssertLaneCountSame<T, U, Arch>
where
    Arch: SimdArch + SimdKernel<T> + SimdKernel<U>,
    T: Scalar,
    U: Scalar,
{
    const OK: () = {
        assert!(
            <Arch as SimdStorage<T>>::LANE_COUNT == <Arch as SimdStorage<U>>::LANE_COUNT,
            "Source and destination vectors must have the same lane count"
        );
    };
}

struct AssertLaneCount<T, Arch, const N: usize>(PhantomData<(T, Arch)>);
impl<T, Arch, const N: usize> AssertLaneCount<T, Arch, N>
where
    Arch: SimdArch + SimdKernel<T>,
    T: Scalar,
{
    const OK: () = {
        assert!(
            N == Arch::LANE_COUNT,
            "Array size must match Vector lane count"
        );
    };
}
