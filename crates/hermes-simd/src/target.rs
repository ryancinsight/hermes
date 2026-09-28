//! Explicit SIMD target tokens and forced view dispatch helpers.

use alloc::vec::Vec;

use crate::{Avx2, Avx512, DispatchedView, Neon, Scalar, SveArch};
use hermes_simd_core::{
    align::Alignment, arch::SimdArch, execution::Unmasked, scalar::FloatElement, view::SimdView,
};

/// Runtime-selectable SIMD target token for tests and benchmark harnesses.
///
/// `TargetId` enumerates Hermes' public CPU targets; the set is
/// `#[non_exhaustive]`, so new backends are additive for consumers. Use
/// [`TargetId::is_supported`] before entering a target-specific benchmark row,
/// or call [`dispatch_view_to`] / [`dispatch_view_mut_to`] to construct a typed
/// view only when the host can execute that target.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum TargetId {
    /// Portable scalar target; always supported.
    Scalar,
    /// `x86/x86_64` AVX2 target, requiring AVX2 and FMA.
    Avx2,
    /// `x86/x86_64` AVX-512F target.
    Avx512,
    /// `AArch64` NEON target.
    Neon,
    /// `AArch64` SVE shape, lane-emulated; executes on every host.
    Sve,
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
    pub const ALL: [Self; 5] = [
        Self::Scalar,
        Self::Avx2,
        Self::Avx512,
        Self::Neon,
        Self::Sve,
    ];

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
            // The emulated SVE backend is compiled and exported on every host,
            // so it belongs to every build target, exactly like the scalar path.
            Self::Scalar | Self::Sve => true,
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
            Self::Sve => "sve",
        }
    }

    /// Returns true when the current host may execute this target.
    #[must_use]
    pub fn is_supported(self) -> bool {
        match self {
            // `SveArch` is lane-emulated and safe to construct on every host.
            Self::Scalar | Self::Sve => true,
            // Each marker's `is_runtime_supported` is the single source of truth
            // for the host-capability question (false wherever the ISA cannot
            // exist), so this table does not restate feature detection.
            Self::Avx2 => <Avx2 as SimdArch>::is_runtime_supported(),
            Self::Avx512 => <Avx512 as SimdArch>::is_runtime_supported(),
            Self::Neon => <Neon as SimdArch>::is_runtime_supported(),
        }
    }
}

/// Builds a [`DispatchedView`] for the architecture marker it is implemented for.
///
/// An implementation owns its marker's `#[cfg]` gate and host-capability check,
/// so the runtime-detected and explicitly-requested ladders below share one
/// target → marker mapping instead of each restating the whole ISA ladder.
trait BuildDispatchedView {
    fn build<T, Align>(data: &[T]) -> Option<DispatchedView<'_, T, Align, Unmasked, &[T]>>
    where
        T: FloatElement,
        Align: Alignment;

    fn build_mut<T, Align>(
        data: &mut [T],
    ) -> Option<DispatchedView<'_, T, Align, Unmasked, &mut [T]>>
    where
        T: FloatElement,
        Align: Alignment;
}

impl BuildDispatchedView for Scalar {
    #[inline]
    fn build<T, Align>(data: &[T]) -> Option<DispatchedView<'_, T, Align, Unmasked, &[T]>>
    where
        T: FloatElement,
        Align: Alignment,
    {
        SimdView::<T, Scalar, Align, Unmasked, &[T]>::new(data).map(DispatchedView::Scalar)
    }

    #[inline]
    fn build_mut<T, Align>(
        data: &mut [T],
    ) -> Option<DispatchedView<'_, T, Align, Unmasked, &mut [T]>>
    where
        T: FloatElement,
        Align: Alignment,
    {
        SimdView::<T, Scalar, Align, Unmasked, &mut [T]>::new_mut(data).map(DispatchedView::Scalar)
    }
}

impl BuildDispatchedView for SveArch {
    #[inline]
    fn build<T, Align>(data: &[T]) -> Option<DispatchedView<'_, T, Align, Unmasked, &[T]>>
    where
        T: FloatElement,
        Align: Alignment,
    {
        SimdView::<T, SveArch, Align, Unmasked, &[T]>::new(data).map(DispatchedView::Sve)
    }

    #[inline]
    fn build_mut<T, Align>(
        data: &mut [T],
    ) -> Option<DispatchedView<'_, T, Align, Unmasked, &mut [T]>>
    where
        T: FloatElement,
        Align: Alignment,
    {
        SimdView::<T, SveArch, Align, Unmasked, &mut [T]>::new_mut(data).map(DispatchedView::Sve)
    }
}

impl BuildDispatchedView for Avx2 {
    #[inline]
    fn build<T, Align>(data: &[T]) -> Option<DispatchedView<'_, T, Align, Unmasked, &[T]>>
    where
        T: FloatElement,
        Align: Alignment,
    {
        // The marker only has a view where its ISA can exist; elsewhere `data`
        // is intentionally unused and no view is produced.
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        if <Avx2 as SimdArch>::is_runtime_supported() {
            return SimdView::<T, Avx2, Align, Unmasked, &[T]>::new(data).map(DispatchedView::Avx2);
        }
        #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
        let _ = data;
        None
    }

    #[inline]
    fn build_mut<T, Align>(
        data: &mut [T],
    ) -> Option<DispatchedView<'_, T, Align, Unmasked, &mut [T]>>
    where
        T: FloatElement,
        Align: Alignment,
    {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        if <Avx2 as SimdArch>::is_runtime_supported() {
            return SimdView::<T, Avx2, Align, Unmasked, &mut [T]>::new_mut(data)
                .map(DispatchedView::Avx2);
        }
        #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
        let _ = data;
        None
    }
}

impl BuildDispatchedView for Avx512 {
    #[inline]
    fn build<T, Align>(data: &[T]) -> Option<DispatchedView<'_, T, Align, Unmasked, &[T]>>
    where
        T: FloatElement,
        Align: Alignment,
    {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        if <Avx512 as SimdArch>::is_runtime_supported() {
            return SimdView::<T, Avx512, Align, Unmasked, &[T]>::new(data)
                .map(DispatchedView::Avx512);
        }
        #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
        let _ = data;
        None
    }

    #[inline]
    fn build_mut<T, Align>(
        data: &mut [T],
    ) -> Option<DispatchedView<'_, T, Align, Unmasked, &mut [T]>>
    where
        T: FloatElement,
        Align: Alignment,
    {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        if <Avx512 as SimdArch>::is_runtime_supported() {
            return SimdView::<T, Avx512, Align, Unmasked, &mut [T]>::new_mut(data)
                .map(DispatchedView::Avx512);
        }
        #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
        let _ = data;
        None
    }
}

impl BuildDispatchedView for Neon {
    #[inline]
    fn build<T, Align>(data: &[T]) -> Option<DispatchedView<'_, T, Align, Unmasked, &[T]>>
    where
        T: FloatElement,
        Align: Alignment,
    {
        #[cfg(target_arch = "aarch64")]
        if <Neon as SimdArch>::is_runtime_supported() {
            return SimdView::<T, Neon, Align, Unmasked, &[T]>::new(data).map(DispatchedView::Neon);
        }
        #[cfg(not(target_arch = "aarch64"))]
        let _ = data;
        None
    }

    #[inline]
    fn build_mut<T, Align>(
        data: &mut [T],
    ) -> Option<DispatchedView<'_, T, Align, Unmasked, &mut [T]>>
    where
        T: FloatElement,
        Align: Alignment,
    {
        #[cfg(target_arch = "aarch64")]
        if <Neon as SimdArch>::is_runtime_supported() {
            return SimdView::<T, Neon, Align, Unmasked, &mut [T]>::new_mut(data)
                .map(DispatchedView::Neon);
        }
        #[cfg(not(target_arch = "aarch64"))]
        let _ = data;
        None
    }
}

/// Selects the widest backend the running host implements.
///
/// Vector targets are only considered where their ISA can exist at all; every
/// other build falls through to [`TargetId::Scalar`].
#[inline]
pub(crate) fn best_target() -> TargetId {
    #[cfg(target_arch = "aarch64")]
    {
        TargetId::Neon
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if <Avx512 as SimdArch>::is_runtime_supported() {
                return TargetId::Avx512;
            }
            if <Avx2 as SimdArch>::is_runtime_supported() {
                return TargetId::Avx2;
            }
        }
        TargetId::Scalar
    }
}

/// Dispatches a shared slice into an explicitly requested target.
///
/// Returns `None` when the target is not supported by the host or when the
/// requested alignment typestate is not satisfied by `data`.
#[inline]
pub fn dispatch_view_to<T, Align>(
    target: TargetId,
    data: &[T],
) -> Option<DispatchedView<'_, T, Align, Unmasked, &[T]>>
where
    T: FloatElement,
    Align: Alignment,
{
    match target {
        TargetId::Scalar => Scalar::build(data),
        TargetId::Avx2 => Avx2::build(data),
        TargetId::Avx512 => Avx512::build(data),
        TargetId::Neon => Neon::build(data),
        TargetId::Sve => SveArch::build(data),
    }
}

/// Dispatches a mutable slice into an explicitly requested target.
///
/// Returns `None` when the target is not supported by the host or when the
/// requested alignment typestate is not satisfied by `data`.
#[inline]
pub fn dispatch_view_mut_to<T, Align>(
    target: TargetId,
    data: &mut [T],
) -> Option<DispatchedView<'_, T, Align, Unmasked, &mut [T]>>
where
    T: FloatElement,
    Align: Alignment,
{
    match target {
        TargetId::Scalar => Scalar::build_mut(data),
        TargetId::Avx2 => Avx2::build_mut(data),
        TargetId::Avx512 => Avx512::build_mut(data),
        TargetId::Neon => Neon::build_mut(data),
        TargetId::Sve => SveArch::build_mut(data),
    }
}
