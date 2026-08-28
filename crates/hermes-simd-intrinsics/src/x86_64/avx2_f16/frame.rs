//! Private scalar selection for the complete AVX2 F16C dispatch frame.

use super::{f16c, Avx2F16c};
use crate::Avx2;
use hermes_simd_core::{
    arch::SimdArch,
    kernel::{BackendKernel, SimdKernel},
    scalar::{Bf16, Bf4, Bf8, Scalar, F16, F32, F4, F64, F8, I16, I32, I8},
};

/// Callback instantiated with the architecture selected for an AVX2 scalar.
///
/// This is proc-macro plumbing, not a consumer extension seam. The callback
/// hides the private F16C frame marker while allowing the generated target-
/// feature helper to monomorphize the complete kernel for it.
#[doc(hidden)]
pub trait Avx2FrameKernel<T: Scalar, Args> {
    /// Callback result.
    type Output;

    /// Instantiates the callback for `Arch`.
    fn call<Arch>(self, args: Args) -> Self::Output
    where
        Arch: SimdArch + SimdKernel<T>;
}

/// Selects the internal AVX2 marker appropriate for one scalar type.
///
/// All ordinary AVX2 scalars retain [`Avx2`]. [`F16`] selects a private marker
/// whose construction is possible only after the generated F16C boundary
/// probe, so its lane arithmetic contains no repeated feature checks.
#[doc(hidden)]
pub trait Avx2FrameScalar: Scalar + Sized {
    /// Runs `kernel` in the scalar's proven AVX2 feature frame.
    ///
    /// # Safety
    ///
    /// The caller must execute inside an `avx2,fma,f16c` target-feature frame
    /// after proving those features are available on the current host.
    unsafe fn call_avx2_frame<Kernel, Args>(kernel: Kernel, args: Args) -> Kernel::Output
    where
        Kernel: Avx2FrameKernel<Self, Args>;
}

impl Avx2FrameScalar for F16 {
    #[inline(always)]
    unsafe fn call_avx2_frame<Kernel, Args>(kernel: Kernel, args: Args) -> Kernel::Output
    where
        Kernel: Avx2FrameKernel<Self, Args>,
    {
        kernel.call::<Avx2F16c>(args)
    }
}

macro_rules! impl_plain_avx2_frame {
    ($($scalar:ty),+ $(,)?) => {
        $(
            impl Avx2FrameScalar for $scalar {
                #[inline(always)]
                unsafe fn call_avx2_frame<Kernel, Args>(
                    kernel: Kernel,
                    args: Args,
                ) -> Kernel::Output
                where
                    Kernel: Avx2FrameKernel<Self, Args>,
                {
                    kernel.call::<Avx2>(args)
                }
            }
        )+
    };
}

impl_plain_avx2_frame!(i8, i16, i32, f32, f64, Bf16, I8, I16, I32, F32, F64, Bf8, Bf4, F8, F4,);

impl BackendKernel<F16> for Avx2F16c {
    type Vector = <Avx2 as BackendKernel<F16>>::Vector;
    type Mask = <Avx2 as BackendKernel<F16>>::Mask;
    type IndexVector = <Avx2 as BackendKernel<F16>>::IndexVector;

    const LANE_COUNT: usize = <Avx2 as BackendKernel<F16>>::LANE_COUNT;
    const UNROLL_FACTOR: usize = <Avx2 as BackendKernel<F16>>::UNROLL_FACTOR;

    #[inline(always)]
    unsafe fn load_aligned(ptr: *const F16) -> Self::Vector {
        // SAFETY: the caller preserves the source pointer contract.
        unsafe { <Avx2 as BackendKernel<F16>>::load_aligned(ptr) }
    }

    #[inline(always)]
    unsafe fn load_unaligned(ptr: *const F16) -> Self::Vector {
        // SAFETY: the caller preserves the source pointer contract.
        unsafe { <Avx2 as BackendKernel<F16>>::load_unaligned(ptr) }
    }

    #[inline(always)]
    unsafe fn store_aligned(ptr: *mut F16, value: Self::Vector) {
        // SAFETY: the caller preserves the destination pointer contract.
        unsafe { <Avx2 as BackendKernel<F16>>::store_aligned(ptr, value) };
    }

    #[inline(always)]
    unsafe fn store_unaligned(ptr: *mut F16, value: Self::Vector) {
        // SAFETY: the caller preserves the destination pointer contract.
        unsafe { <Avx2 as BackendKernel<F16>>::store_unaligned(ptr, value) };
    }

    #[inline(always)]
    unsafe fn add(a: Self::Vector, b: Self::Vector) -> Self::Vector {
        // SAFETY: construction of this private marker proves F16C support.
        unsafe { f16c::add(a, b) }
    }

    #[inline(always)]
    unsafe fn mul(a: Self::Vector, b: Self::Vector) -> Self::Vector {
        // SAFETY: construction of this private marker proves F16C support.
        unsafe { f16c::mul(a, b) }
    }

    #[inline(always)]
    unsafe fn sub(a: Self::Vector, b: Self::Vector) -> Self::Vector {
        // SAFETY: construction of this private marker proves F16C support.
        unsafe { f16c::sub(a, b) }
    }

    #[inline(always)]
    unsafe fn fmadd(a: Self::Vector, b: Self::Vector, c: Self::Vector) -> Self::Vector {
        // SAFETY: construction of this private marker proves F16C and FMA.
        unsafe { f16c::fmadd(a, b, c) }
    }

    #[inline(always)]
    unsafe fn sum_reduce(value: Self::Vector) -> F16 {
        // SAFETY: the forwarded register has the identical representation.
        unsafe { <Avx2 as BackendKernel<F16>>::sum_reduce(value) }
    }

    #[inline(always)]
    unsafe fn compress(source: Self::Vector, mask: Self::Mask) -> Self::Vector {
        // SAFETY: the forwarded register and mask representations are identical.
        unsafe { <Avx2 as BackendKernel<F16>>::compress(source, mask) }
    }

    #[inline(always)]
    unsafe fn expand(source: Self::Vector, mask: Self::Mask, fill: Self::Vector) -> Self::Vector {
        // SAFETY: the forwarded register and mask representations are identical.
        unsafe { <Avx2 as BackendKernel<F16>>::expand(source, mask, fill) }
    }

    #[inline(always)]
    unsafe fn gather(base: *const F16, indices: Self::IndexVector) -> Self::Vector {
        // SAFETY: the caller preserves the base/index validity contract.
        unsafe { <Avx2 as BackendKernel<F16>>::gather(base, indices) }
    }

    #[inline(always)]
    unsafe fn gather_masked(
        base: *const F16,
        indices: Self::IndexVector,
        mask: Self::Mask,
        source: Self::Vector,
    ) -> Self::Vector {
        // SAFETY: the caller preserves validity for each active base/index lane.
        unsafe { <Avx2 as BackendKernel<F16>>::gather_masked(base, indices, mask, source) }
    }

    #[inline(always)]
    unsafe fn mask_from_bools(bits: &[bool]) -> Self::Mask {
        // SAFETY: the forwarded mask representation and lane count are identical.
        unsafe { <Avx2 as BackendKernel<F16>>::mask_from_bools(bits) }
    }

    #[inline(always)]
    unsafe fn leading_k_mask(k: usize) -> Self::Mask {
        // SAFETY: the forwarded mask representation and lane count are identical.
        unsafe { <Avx2 as BackendKernel<F16>>::leading_k_mask(k) }
    }

    #[inline(always)]
    unsafe fn mask_to_vector(mask: Self::Mask) -> Self::Vector {
        // SAFETY: the forwarded register and mask representations are identical.
        unsafe { <Avx2 as BackendKernel<F16>>::mask_to_vector(mask) }
    }

    #[inline(always)]
    unsafe fn vector_to_mask(value: Self::Vector) -> Self::Mask {
        // SAFETY: the forwarded register and mask representations are identical.
        unsafe { <Avx2 as BackendKernel<F16>>::vector_to_mask(value) }
    }

    #[inline(always)]
    unsafe fn splat(value: F16) -> Self::Vector {
        // SAFETY: the forwarded register representation and lane count are identical.
        unsafe { <Avx2 as BackendKernel<F16>>::splat(value) }
    }

    #[inline(always)]
    unsafe fn mask_to_bitmask(mask: Self::Mask) -> u64 {
        // SAFETY: the forwarded mask representation and lane count are identical.
        unsafe { <Avx2 as BackendKernel<F16>>::mask_to_bitmask(mask) }
    }
}
