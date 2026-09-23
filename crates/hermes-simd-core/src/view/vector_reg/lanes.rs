//! Lane-indexed access and const-generic array/type conversions.

use super::assert_runtime_supported;
use super::runtime_support_result;
use super::AssertLaneCount;
use super::AssertLaneCountSame;
use super::AssertLaneIndex;
use super::Vector;
use crate::arch::SimdArch;
use crate::kernel::SimdKernel;
use crate::kernel::SimdLoadStore;
use crate::kernel::SimdStorage;
use crate::kernel::MAX_SIMD_LANES;
use crate::scalar::CastFrom;
use crate::scalar::Scalar;
use crate::view::SimdError;

impl<T, Arch> Vector<T, Arch>
where
    Arch: SimdArch + SimdKernel<T>,
    T: Scalar,
{
    /// Create a Vector from an array of size `N`, where `N` must equal `Arch::LANE_COUNT`.
    #[inline(always)]
    pub fn from_array<const N: usize>(arr: [T; N]) -> Self {
        assert_runtime_supported::<T, Arch>();
        let () = AssertLaneCount::<T, Arch, N>::OK;
        // SAFETY: target feature checked above; `AssertLaneCount` proved
        // `N == LANE_COUNT`, so `arr` holds a full vector's worth of elements for
        // the unaligned load.
        unsafe { Self::load_unaligned(arr.as_ptr()) }
    }

    /// Try to create a Vector from an array of size `N`, where `N` must equal
    /// `Arch::LANE_COUNT`.
    ///
    /// # Errors
    /// Returns [`SimdError::UnsupportedTarget`] when the architecture is not
    /// supported or enabled on this host.
    #[inline(always)]
    pub fn try_from_array<const N: usize>(arr: [T; N]) -> Result<Self, SimdError> {
        runtime_support_result::<T, Arch>()?;
        let () = AssertLaneCount::<T, Arch, N>::OK;
        // SAFETY: as `from_array` — `N == LANE_COUNT`, so `arr` covers the load.
        unsafe { Ok(Self::load_unaligned(arr.as_ptr())) }
    }

    /// Convert the vector to an array of size `N`, where `N` must equal `Arch::LANE_COUNT`.
    #[inline(always)]
    pub fn to_array<const N: usize>(self) -> [T; N] {
        let () = AssertLaneCount::<T, Arch, N>::OK;
        let mut arr = [core::mem::MaybeUninit::<T>::uninit(); N];
        // SAFETY: target feature checked above; `AssertLaneCount` proved
        // `N == LANE_COUNT`, so the store initializes all `N` slots before the
        // `[T; N]` is read out.
        unsafe {
            self.store_unaligned(arr.as_mut_ptr().cast::<T>());
            core::ptr::read(arr.as_ptr().cast::<[T; N]>())
        }
    }

    /// Cast the vector elements to another scalar type `U` where the lane counts match.
    #[inline(always)]
    pub fn cast<U>(self) -> Vector<U, Arch>
    where
        Arch: SimdKernel<U>,
        U: Scalar,
        U: CastFrom<T>,
    {
        let () = AssertLaneCountSame::<T, U, Arch>::OK;
        const { <Arch as SimdStorage<T>>::LANE_BOUND_CHECK };
        let mut buf_u = [core::mem::MaybeUninit::<U>::uninit(); MAX_SIMD_LANES];
        let lanes = <Arch as SimdStorage<T>>::LANE_COUNT;
        // SAFETY: target features for both `T` and `U` checked above;
        // `AssertLaneCountSame` and `LANE_BOUND_CHECK` bound `lanes` within both
        // buffers. A successful native hook initializes `buf_u[..lanes]`.
        // Otherwise the `T` store initializes `buf_t[..lanes]` before
        // `assume_init` reads it and the loop initializes `buf_u[..lanes]`.
        // The `U` load reads exactly those initialized lanes.
        unsafe {
            if <Arch as SimdLoadStore<T>>::try_cast::<U>(self.raw, buf_u.as_mut_ptr().cast::<U>()) {
                return Vector::<U, Arch>::new(<Arch as SimdLoadStore<U>>::load_unaligned(
                    buf_u.as_ptr().cast::<U>(),
                ));
            }
            let mut buf_t = [core::mem::MaybeUninit::<T>::uninit(); MAX_SIMD_LANES];
            self.store_unaligned(buf_t.as_mut_ptr().cast::<T>());
            for i in 0..lanes {
                let val_t = buf_t[i].assume_init();
                buf_u[i].write(U::cast_from(val_t));
            }
            Vector::<U, Arch>::new(<Arch as SimdLoadStore<U>>::load_unaligned(
                buf_u.as_ptr().cast::<U>(),
            ))
        }
    }

    /// Extract a single lane element by index at compile-time.
    #[inline(always)]
    pub fn extract<const I: usize>(self) -> T {
        let () = AssertLaneIndex::<T, Arch, I>::OK;
        const { <Arch as SimdStorage<T>>::LANE_BOUND_CHECK };
        let mut buf = [core::mem::MaybeUninit::<T>::uninit(); MAX_SIMD_LANES];
        // SAFETY: constructing `self` proved host support; `AssertLaneIndex` proved
        // `I < LANE_COUNT`, and the store initializes `buf[..LANE_COUNT]`, so
        // `buf[I]` is initialized.
        unsafe {
            self.store_unaligned(buf.as_mut_ptr().cast::<T>());
            buf[I].assume_init()
        }
    }

    /// Insert a value into a single lane by index at compile-time.
    #[inline(always)]
    #[must_use]
    pub fn insert<const I: usize>(self, val: T) -> Self {
        let () = AssertLaneIndex::<T, Arch, I>::OK;
        const { <Arch as SimdStorage<T>>::LANE_BOUND_CHECK };
        let mut buf = [core::mem::MaybeUninit::<T>::uninit(); MAX_SIMD_LANES];
        // SAFETY: constructing `self` proved host support; `AssertLaneIndex` proved
        // `I < LANE_COUNT`. The store initializes `buf[..LANE_COUNT]`, `buf[I]` is
        // then overwritten, and the reload reads all `LANE_COUNT` initialized
        // lanes.
        unsafe {
            self.store_unaligned(buf.as_mut_ptr().cast::<T>());
            buf[I].write(val);
            Self::load_unaligned(buf.as_ptr().cast::<T>())
        }
    }
}
