//! Explicit SIMD target tokens and forced view dispatch helpers.

#[cfg(target_arch = "aarch64")]
use crate::Neon;
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
use crate::{Avx2, Avx512};
use crate::{DispatchedView, Scalar};
use hermes_simd_core::{
    align::Alignment, execution::Unmasked, scalar::FloatElement, view::SimdView,
};

/// Runtime-selectable SIMD target token for tests and benchmark harnesses.
///
/// `TargetId` is a closed identifier for Hermes' public CPU targets. Use
/// [`TargetId::is_supported`] before entering a target-specific benchmark row,
/// or call [`dispatch_view_to`] / [`dispatch_view_mut_to`] to construct a typed
/// view only when the host can execute that target.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TargetId {
    /// Portable scalar target; always supported.
    Scalar,
    /// x86/x86_64 AVX2 target, requiring AVX2 and FMA.
    Avx2,
    /// x86/x86_64 AVX-512F target.
    Avx512,
    /// AArch64 NEON target.
    Neon,
}

impl TargetId {
    /// Returns the stable lowercase target name used in reports and benchmarks.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Scalar => "scalar",
            Self::Avx2 => "avx2",
            Self::Avx512 => "avx512",
            Self::Neon => "neon",
        }
    }

    /// Returns true when the current host may execute this target.
    #[must_use]
    pub fn is_supported(self) -> bool {
        match self {
            Self::Scalar => true,
            Self::Avx2 => avx2_supported(),
            Self::Avx512 => avx512_supported(),
            Self::Neon => neon_supported(),
        }
    }
}

#[inline]
fn avx2_supported() -> bool {
    #[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "std"))]
    {
        std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma")
    }
    #[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), not(feature = "std")))]
    {
        cfg!(target_feature = "avx2") && cfg!(target_feature = "fma")
    }
    #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
    {
        false
    }
}

#[inline]
fn avx512_supported() -> bool {
    #[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "std"))]
    {
        std::is_x86_feature_detected!("avx512f")
    }
    #[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), not(feature = "std")))]
    {
        cfg!(target_feature = "avx512f")
    }
    #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
    {
        false
    }
}

#[inline]
fn neon_supported() -> bool {
    #[cfg(target_arch = "aarch64")]
    {
        true
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        false
    }
}

/// Dispatches a shared slice into an explicitly requested target.
///
/// Returns `None` when the target is not supported by the host or when the
/// requested alignment typestate is not satisfied by `data`.
#[inline]
#[allow(unreachable_code)]
pub fn dispatch_view_to<'a, T, Align>(
    target: TargetId,
    data: &'a [T],
) -> Option<DispatchedView<'a, T, Align, Unmasked, &'a [T]>>
where
    T: FloatElement,
    Align: Alignment,
{
    match target {
        TargetId::Scalar => {
            SimdView::<T, Scalar, Align, Unmasked, &'a [T]>::new(data).map(DispatchedView::Scalar)
        }
        TargetId::Avx2 => {
            if !target.is_supported() {
                None
            } else {
                #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
                {
                    SimdView::<T, Avx2, Align, Unmasked, &'a [T]>::new(data)
                        .map(DispatchedView::Avx2)
                }
                #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
                {
                    None
                }
            }
        }
        TargetId::Avx512 => {
            if !target.is_supported() {
                None
            } else {
                #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
                {
                    SimdView::<T, Avx512, Align, Unmasked, &'a [T]>::new(data)
                        .map(DispatchedView::Avx512)
                }
                #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
                {
                    None
                }
            }
        }
        TargetId::Neon => {
            if !target.is_supported() {
                None
            } else {
                #[cfg(target_arch = "aarch64")]
                {
                    SimdView::<T, Neon, Align, Unmasked, &'a [T]>::new(data)
                        .map(DispatchedView::Neon)
                }
                #[cfg(not(target_arch = "aarch64"))]
                {
                    None
                }
            }
        }
    }
}

/// Dispatches a mutable slice into an explicitly requested target.
///
/// Returns `None` when the target is not supported by the host or when the
/// requested alignment typestate is not satisfied by `data`.
#[inline]
#[allow(unreachable_code)]
pub fn dispatch_view_mut_to<'a, T, Align>(
    target: TargetId,
    data: &'a mut [T],
) -> Option<DispatchedView<'a, T, Align, Unmasked, &'a mut [T]>>
where
    T: FloatElement,
    Align: Alignment,
{
    match target {
        TargetId::Scalar => SimdView::<T, Scalar, Align, Unmasked, &'a mut [T]>::new_mut(data)
            .map(DispatchedView::Scalar),
        TargetId::Avx2 => {
            if !target.is_supported() {
                None
            } else {
                #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
                {
                    SimdView::<T, Avx2, Align, Unmasked, &'a mut [T]>::new_mut(data)
                        .map(DispatchedView::Avx2)
                }
                #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
                {
                    None
                }
            }
        }
        TargetId::Avx512 => {
            if !target.is_supported() {
                None
            } else {
                #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
                {
                    SimdView::<T, Avx512, Align, Unmasked, &'a mut [T]>::new_mut(data)
                        .map(DispatchedView::Avx512)
                }
                #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
                {
                    None
                }
            }
        }
        TargetId::Neon => {
            if !target.is_supported() {
                None
            } else {
                #[cfg(target_arch = "aarch64")]
                {
                    SimdView::<T, Neon, Align, Unmasked, &'a mut [T]>::new_mut(data)
                        .map(DispatchedView::Neon)
                }
                #[cfg(not(target_arch = "aarch64"))]
                {
                    None
                }
            }
        }
    }
}
