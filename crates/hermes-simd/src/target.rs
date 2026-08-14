//! Explicit SIMD target tokens and forced view dispatch helpers.

#[cfg(target_arch = "aarch64")]
use crate::Neon;
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
use crate::{Avx2, Avx512};
use crate::{DispatchedView, Scalar};
use hermes_simd_core::{
    align::Alignment, arch::SimdArch, execution::Unmasked, scalar::FloatElement, view::SimdView,
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
    /// Every public CPU target, in ascending capability order.
    ///
    /// Enumerating the closed set is what lets a harness *identify* which
    /// backends the host can execute rather than assume it. Combined with
    /// [`TargetId::is_supported`], this turns a capability-gated suite's
    /// coverage from an invisible property into a reportable one — a test
    /// guarded by a feature probe otherwise skips silently, and a skip is
    /// indistinguishable from a pass in the log.
    pub const ALL: [Self; 4] = [Self::Scalar, Self::Avx2, Self::Avx512, Self::Neon];

    /// Parses a target from the lowercase name emitted by [`TargetId::name`].
    ///
    /// The inverse of `name`, so a coverage expectation can be declared as
    /// configuration (a CI environment variable) rather than compiled in.
    #[must_use]
    pub fn from_name(name: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|t| t.name() == name)
    }

    /// Returns whether this target belongs to the architecture being compiled
    /// for at all.
    ///
    /// Distinct from [`TargetId::is_supported`], which asks whether *this CPU*
    /// implements the feature. A coverage report must separate the two: AVX-512
    /// unsupported on an x86 host is a real gap in what was exercised, whereas
    /// AVX-512 on aarch64 is simply not part of that build and can never be a
    /// gap. Collapsing both into one "unsupported" reads as missing coverage
    /// where none is possible.
    #[must_use]
    pub const fn is_architecture_applicable(self) -> bool {
        match self {
            Self::Scalar => true,
            Self::Avx2 | Self::Avx512 => cfg!(any(target_arch = "x86", target_arch = "x86_64")),
            Self::Neon => cfg!(target_arch = "aarch64"),
        }
    }

    /// Returns the targets this host can execute, in [`TargetId::ALL`] order.
    #[must_use]
    pub fn supported_on_host() -> Vec<Self> {
        Self::ALL.into_iter().filter(|t| t.is_supported()).collect()
    }

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
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        Avx2::is_runtime_supported()
    }
    #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
    {
        false
    }
}

#[inline]
fn avx512_supported() -> bool {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        Avx512::is_runtime_supported()
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
        Neon::is_runtime_supported()
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
#[expect(
    unreachable_code,
    reason = "Explicit target arms are cfg-selected before the function's fallback path"
)]
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
#[expect(
    unreachable_code,
    reason = "Explicit target arms are cfg-selected before the function's fallback path"
)]
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
