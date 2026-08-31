//! Consumer-facing entry into a `#[target_feature]` scope.
//!
//! # The problem this solves
//!
//! Hermes' safe lane surface — [`Vector`](hermes_simd_core::view::Vector) and
//! its operators — is generic over the backend marker, so a consumer can write
//! one kernel that monomorphizes to every ISA. Doing that correctly needs one
//! more thing, and until this module it was not exported.
//!
//! `#[target_feature]` does not propagate through monomorphization. A generic
//! kernel instantiated at `Avx2` from an ordinary function compiles as if the
//! host had no AVX2: the annotated backend operations become out-of-line calls
//! instead of inlining into the loop body. ADR 009 records the measurement and
//! the fix — enter the kernel through a per-ISA `#[target_feature]` helper — and
//! `#[runtime_dispatch]` applies that fix across `hermes-simd`'s own kernels.
//! It was not reachable from outside this crate, so a consumer had two options:
//! accept baseline codegen, or write its own `#[target_feature]` trampolines and
//! intrinsics. Consumers across the stack chose the second.
//!
//! [`vectorize`] is that entry. A consumer implements [`LaneKernel`] once, and
//! Hermes runs it inside the widest scope this host supports.
//!
//! # Safety model
//!
//! Nothing here weakens the existing one. Holding an `Arch`-parameterized value
//! already means the host executes `Arch` — that invariant is established at
//! construction and is what every `unsafe` backend call downstream discharges
//! against. [`vectorize`] adds no new obligation: it selects a backend the host
//! was probed for and enters its target-feature scope, so the consumer's kernel
//! body needs no `unsafe` of its own. The conformance test in
//! `tests/consumer_vectorize.rs` compiles under `#![forbid(unsafe_code)]` to
//! keep that property honest rather than merely stated.
//!
//! The scope is a code-generation mechanism, not a soundness one. Calling a
//! backend operation outside it on a supporting host is sound but slow; that is
//! precisely the defect this module exists to remove.
//!
//! # Example
//!
//! ```
//! use hermes_simd::{LaneKernel, Simd, SimdArch, SimdKernel, Vector, vectorize};
//!
//! /// Elementwise `a * b + c` over three equal-length slices.
//! struct FusedMulAdd<'a> {
//!     a: &'a [f32],
//!     b: &'a [f32],
//!     c: &'a [f32],
//! }
//!
//! impl LaneKernel<f32> for FusedMulAdd<'_> {
//!     type Output = Vec<f32>;
//!
//!     fn call<A: SimdArch + SimdKernel<f32>>(self, simd: Simd<f32, A>) -> Vec<f32> {
//!         let lanes = <A as hermes_simd::SimdStorage<f32>>::LANE_COUNT;
//!         let mut out = vec![0.0; self.a.len()];
//!         let a = simd.view(self.a);
//!         let b = simd.view(self.b);
//!         let c = simd.view(self.c);
//!         let mut out_view = simd.view_mut(&mut out);
//!         let mut i = 0;
//!         while i + lanes <= self.a.len() {
//!             let va = Vector::from_view_chunk(&a, i / lanes);
//!             let vb = Vector::from_view_chunk(&b, i / lanes);
//!             let vc = Vector::from_view_chunk(&c, i / lanes);
//!             va.mul_add(vb, vc)
//!                 .store_to_view_chunk(&mut out_view, i / lanes);
//!             i += lanes;
//!         }
//!         for j in i..self.a.len() {
//!             out[j] = self.a[j].mul_add(self.b[j], self.c[j]);
//!         }
//!         out
//!     }
//! }
//!
//! let a = [1.0f32, 2.0, 3.0, 4.0];
//! let b = [10.0f32, 10.0, 10.0, 10.0];
//! let c = [0.5f32, 0.5, 0.5, 0.5];
//! assert_eq!(
//!     vectorize(FusedMulAdd { a: &a, b: &b, c: &c }),
//!     vec![10.5, 20.5, 30.5, 40.5]
//! );
//! ```

use hermes_simd_core::{
    arch::SimdArch,
    kernel::{SimdKernel, SimdStorage},
    scalar::Scalar,
    Simd,
};
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
use hermes_simd_intrinsics::x86_64::avx2_f16::{Avx2FrameKernel, Avx2FrameScalar};
#[cfg(target_arch = "aarch64")]
use hermes_simd_intrinsics::Neon;
use hermes_simd_intrinsics::Scalar as ScalarArch;
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
use hermes_simd_intrinsics::{Avx2, Avx512};
use hermes_simd_macros::runtime_dispatch;

/// A lane kernel written once and monomorphized to every backend.
///
/// A closure cannot carry a generic type parameter, so the kernel is a type
/// whose captured state is its fields and whose [`LaneKernel::call`] is generic
/// over the backend. That is the same shape `pulp` and `fearless_simd` use, and
/// the reason both use it: the backend must be a *type* parameter for
/// monomorphization, and only a trait method can bind one.
///
/// Mark [`LaneKernel::call`] `#[inline(always)]` on any kernel with a large
/// body. The whole point of the target-feature scope is that the backend
/// operations fold into the kernel body, and the body reaches that scope by
/// inlining into the generated `#[target_feature]` helper. The dispatch
/// machinery forces its own frames in (`alwaysinline`, honored regardless of
/// size), but `call` is the consumer's function: a large body under the plain
/// inline heuristic gets outlined to baseline codegen — zero FMA, an order of
/// magnitude slow — which small-kernel measurements never show. `pulp`
/// documents the same requirement on `WithSimd::with_simd` for the same
/// reason.
pub trait LaneKernel<T: Scalar> {
    /// What the kernel produces.
    type Output;

    /// Runs the kernel against one backend.
    ///
    /// Called exactly once, from inside that backend's `#[target_feature]`
    /// scope. Which backend is chosen is [`vectorize`]'s decision. `simd`
    /// proves that the host supports `A` and constructs views without another
    /// runtime feature probe.
    fn call<A: SimdArch + SimdKernel<T>>(self, simd: Simd<T, A>) -> Self::Output;
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
struct LaneKernelFrame<K>(K);

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
impl<T, K> Avx2FrameKernel<T, ()> for LaneKernelFrame<K>
where
    T: Scalar,
    K: LaneKernel<T>,
{
    type Output = K::Output;

    #[inline(always)]
    fn call<A>(self, (): ()) -> Self::Output
    where
        A: SimdArch + SimdKernel<T>,
    {
        // SAFETY: `Avx2FrameScalar` selects this specialization only from an
        // AVX2 target-feature helper after its complete feature set is proved.
        self.0
            .call::<A>(unsafe { Simd::<T, A>::assume_supported() })
    }
}

/// A scalar type every Hermes backend can operate on.
///
/// The dispatch ladder names concrete backends, so entering it needs
/// `Avx2: SimdKernel<T>`, `Avx512: SimdKernel<T>`, and the rest proven for the
/// caller's `T`. A caller generic over `T` cannot discharge that without
/// repeating the whole cfg-gated backend list in its own signature — which is
/// exactly the friction that pushes a consumer back to hand-written intrinsics.
///
/// This trait moves the obligation to where it is provable: one impl per scalar
/// the backends actually support, each entering the ladder at a concrete type.
/// A consumer writes `T: LaneScalar` and is done.
///
/// Implemented for `f32`, `f64`, and `F16` — the exact set for which every
/// backend implements `BackendKernel`. It is not extensible from outside: a new
/// scalar becomes vectorizable by gaining backend implementations upstream, not
/// by a downstream impl of this trait.
pub trait LaneScalar: Scalar {
    /// Enters the dispatch ladder at this concrete scalar type.
    ///
    /// Call [`vectorize`] instead; this is the per-type step it forwards to.
    #[doc(hidden)]
    fn run_lane_kernel<K: LaneKernel<Self>>(kernel: K) -> K::Output;

    /// Enters the dispatch ladder at an exact native lane count.
    ///
    /// Call [`vectorize_lanes`] instead. The default preserves compatibility
    /// for downstream implementations while Hermes' sealed scalar set
    /// overrides it with the native dispatch ladder.
    #[doc(hidden)]
    fn run_lane_kernel_for<const LANES: usize, K: LaneKernel<Self>>(
        kernel: K,
    ) -> Option<K::Output> {
        let _ = kernel;
        None
    }
}

macro_rules! impl_lane_scalar {
    ($($t:ty),+ $(,)?) => {
        $(
            impl LaneScalar for $t {
                #[inline(always)]
                fn run_lane_kernel<K: LaneKernel<Self>>(kernel: K) -> K::Output {
                    dispatch_backend::<$t, K>(kernel)
                }

                #[inline(always)]
                fn run_lane_kernel_for<const LANES: usize, K: LaneKernel<Self>>(
                    kernel: K,
                ) -> Option<K::Output> {
                    dispatch_lane_count::<$t, K, LANES>(kernel)
                }
            }
        )+
    };
}

impl_lane_scalar!(f32, f64, eunomia::F16);

/// Runs `kernel` inside the `#[target_feature]` scope of the widest backend
/// this host supports.
///
/// Selection is compile-time first: when the build already enables a feature
/// set, the corresponding arm is chosen with no runtime branch at all. Failing
/// that, the host is probed once per call and the ladder falls through
/// AVX-512F, AVX2 with FMA, NEON, and finally the portable scalar backend,
/// which always applies. F16 selects AVX2 only when F16C is also available,
/// placing conversion and arithmetic inside one complete feature frame;
/// direct `Avx2` F16 values retain their safe software fallback without F16C.
///
/// The dispatch decision is made once, here, not per operation inside the
/// kernel — so a kernel that transforms a whole buffer pays for it once.
#[inline(always)]
pub fn vectorize<T: LaneScalar, K: LaneKernel<T>>(kernel: K) -> K::Output {
    T::run_lane_kernel(kernel)
}

/// Runs `kernel` on the widest supported backend with exactly `LANES` lanes.
///
/// Unlike [`vectorize`], this entry does not substitute a different width. It
/// returns `None` without invoking `kernel` when the current architecture has
/// no backend with the requested lane count and scalar-specific feature set.
/// Selection and feature detection occur once at the operation boundary.
///
/// This is intended for register-resident kernels whose schedule is part of
/// their correctness and performance contract. Width-agnostic kernels should
/// continue to use [`vectorize`].
///
/// # Examples
///
/// ```
/// use hermes_simd::{LaneKernel, Simd, SimdArch, SimdKernel, SimdStorage, vectorize_lanes};
///
/// struct LaneCount;
///
/// impl LaneKernel<f64> for LaneCount {
///     type Output = usize;
///
///     fn call<A: SimdArch + SimdKernel<f64>>(self, _: Simd<f64, A>) -> usize {
///         <A as SimdStorage<f64>>::LANE_COUNT
///     }
/// }
///
/// assert_eq!(vectorize_lanes::<2, f64, _>(LaneCount), Some(2));
/// ```
#[inline(always)]
pub fn vectorize_lanes<const LANES: usize, T: LaneScalar, K: LaneKernel<T>>(
    kernel: K,
) -> Option<K::Output> {
    T::run_lane_kernel_for::<LANES, K>(kernel)
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[inline(always)]
fn dispatch_lane_count<T, K, const LANES: usize>(kernel: K) -> Option<K::Output>
where
    T: Avx2FrameScalar,
    K: LaneKernel<T>,
    Avx512: SimdKernel<T>,
    Avx2: SimdKernel<T>,
    ScalarArch: SimdKernel<T>,
{
    if <Avx512 as SimdStorage<T>>::LANE_COUNT == LANES && Avx512::is_runtime_supported() {
        // SAFETY: the lane-count check selects this backend and the runtime
        // probe proves AVX-512F support before entering its target scope.
        return Some(unsafe { call_avx512(kernel) });
    }
    if <Avx2 as SimdStorage<T>>::LANE_COUNT == LANES && Avx2::is_runtime_supported() {
        if <Avx2 as SimdStorage<T>>::REQUIRES_F16C {
            if avx2_f16c_available() {
                // SAFETY: the architecture probe proves AVX2 and FMA; the
                // scalar-specific probe additionally proves F16C.
                return Some(unsafe { call_avx2_f16c(kernel) });
            }
        } else {
            // SAFETY: the architecture probe proves ordinary AVX2 and FMA
            // support; this scalar does not require the extended F16C frame.
            return Some(unsafe { call_avx2(kernel) });
        }
    }
    // The fallback body still benefits from the wide frame. `ScalarArch`
    // stores its lanes as a small fixed array and operates on them
    // element-wise, which LLVM vectorizes readily — but only into whatever the
    // enabled feature set allows. Reached without a frame it compiles to
    // baseline SSE2 with no FMA, so a scalar whose native width misses every
    // exact-width backend runs the caller's whole kernel at baseline. Entering
    // the same fallback inside the AVX2 frame lets that vectorization use VEX
    // encodings and fuse multiply-add, which is free: no backend changes, and
    // the lane count and every value are identical either way.
    //
    // The frame's feature set must cover what the *framed body* executes, and
    // the framed body is `dispatch_scalar` — that is, `ScalarArch`, which
    // needs nothing. `REQUIRES_F16C` is a property of `Avx2`, a backend this
    // arm does not enter, so it has no bearing here and does not gate entry.
    // Nor is `f16c` added to the frame: `ScalarArch`'s `F16` arithmetic runs
    // through Eunomia's software widen/narrow, which is integer bit
    // manipulation rather than `fptrunc`/`fpext`, so no F16C instruction is
    // selectable in this body and enabling the feature would only narrow the
    // hosts that qualify. What this body does contain is `f32::mul_add` — a
    // `fmaf` library call outside an FMA frame, one instruction inside it.
    if Avx2::is_runtime_supported() {
        // SAFETY: `Avx2::is_runtime_supported()` probes exactly `avx2` and
        // `fma`, which is exactly the feature set the callee's
        // `#[target_feature]` attributes enable — no feature is entered
        // unproven. The frame is entered for its instruction selection only:
        // the scalar backend is executable on every host, so widening the
        // frame around it cannot make it unexecutable.
        return unsafe { call_scalar_in_avx2_frame::<T, K, LANES>(kernel) };
    }
    dispatch_scalar::<T, K, LANES>(kernel)
}

/// The scalar fallback, compiled inside the AVX2 frame.
///
/// Identical to [`dispatch_scalar`] in every observable way — same backend,
/// same lane count, same values. Only the instruction selection available to
/// the compiler differs.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
#[target_feature(enable = "fma")]
#[inline]
unsafe fn call_scalar_in_avx2_frame<T, K, const LANES: usize>(kernel: K) -> Option<K::Output>
where
    T: Scalar,
    K: LaneKernel<T>,
    ScalarArch: SimdKernel<T>,
{
    dispatch_scalar::<T, K, LANES>(kernel)
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
fn dispatch_lane_count<T, K, const LANES: usize>(kernel: K) -> Option<K::Output>
where
    T: Scalar,
    K: LaneKernel<T>,
    Neon: SimdKernel<T>,
    ScalarArch: SimdKernel<T>,
{
    if <Neon as SimdStorage<T>>::LANE_COUNT == LANES && Neon::is_runtime_supported() {
        // SAFETY: NEON is mandatory on AArch64; the runtime probe preserves
        // the same capability boundary used by every other backend.
        return Some(unsafe { call_neon(kernel) });
    }
    dispatch_scalar::<T, K, LANES>(kernel)
}

#[cfg(not(any(target_arch = "x86", target_arch = "x86_64", target_arch = "aarch64")))]
#[inline(always)]
fn dispatch_lane_count<T, K, const LANES: usize>(kernel: K) -> Option<K::Output>
where
    T: Scalar,
    K: LaneKernel<T>,
    ScalarArch: SimdKernel<T>,
{
    dispatch_scalar::<T, K, LANES>(kernel)
}

#[inline(always)]
fn dispatch_scalar<T, K, const LANES: usize>(kernel: K) -> Option<K::Output>
where
    T: Scalar,
    K: LaneKernel<T>,
    ScalarArch: SimdKernel<T>,
{
    if <ScalarArch as SimdStorage<T>>::LANE_COUNT != LANES {
        return None;
    }
    // SAFETY: the scalar backend is executable on every host.
    let simd = unsafe { Simd::<T, ScalarArch>::assume_supported() };
    Some(kernel.call::<ScalarArch>(simd))
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f")]
#[inline]
unsafe fn call_avx512<T, K>(kernel: K) -> K::Output
where
    T: Scalar,
    K: LaneKernel<T>,
    Avx512: SimdKernel<T>,
{
    // SAFETY: the caller proves runtime AVX-512F support before entering this
    // target-feature scope.
    let simd = unsafe { Simd::<T, Avx512>::assume_supported() };
    kernel.call::<Avx512>(simd)
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[inline(always)]
fn avx2_f16c_available() -> bool {
    #[cfg(feature = "std")]
    {
        std::is_x86_feature_detected!("f16c")
    }
    #[cfg(not(feature = "std"))]
    {
        cfg!(target_feature = "f16c")
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
#[target_feature(enable = "fma")]
#[inline]
unsafe fn call_avx2<T, K>(kernel: K) -> K::Output
where
    T: Scalar,
    K: LaneKernel<T>,
    Avx2: SimdKernel<T>,
{
    // SAFETY: the caller proves runtime AVX2 and FMA support before entering
    // this target-feature scope.
    let simd = unsafe { Simd::<T, Avx2>::assume_supported() };
    kernel.call::<Avx2>(simd)
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
#[target_feature(enable = "fma")]
#[target_feature(enable = "f16c")]
unsafe fn call_avx2_f16c<T, K>(kernel: K) -> K::Output
where
    T: Avx2FrameScalar,
    K: LaneKernel<T>,
{
    // SAFETY: the caller proves runtime AVX2, FMA, and F16C support before
    // entering this scalar-specific target-feature scope. The selector keeps
    // its private backend marker confined to this checked boundary.
    unsafe { T::call_avx2_frame(LaneKernelFrame(kernel), ()) }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline]
unsafe fn call_neon<T, K>(kernel: K) -> K::Output
where
    T: Scalar,
    K: LaneKernel<T>,
    Neon: SimdKernel<T>,
{
    // SAFETY: the caller proves runtime NEON support before entering this
    // target-feature scope.
    let simd = unsafe { Simd::<T, Neon>::assume_supported() };
    kernel.call::<Neon>(simd)
}

#[runtime_dispatch(avx512f, avx2, neon, scalar)]
fn dispatch_backend_kernel<T, K, A>(kernel: K) -> K::Output
where
    T: Scalar,
    K: LaneKernel<T>,
    A: SimdArch + SimdKernel<T>,
{
    // SAFETY: `#[runtime_dispatch]` invokes this specialization only after its
    // generated dispatcher proves host support for `A` and enters `A`'s
    // target-feature scope.
    kernel.call::<A>(unsafe { Simd::assume_supported() })
}
