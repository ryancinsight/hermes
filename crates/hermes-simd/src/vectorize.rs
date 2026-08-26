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

use hermes_simd_core::{arch::SimdArch, kernel::SimdKernel, scalar::Scalar, Simd};
use hermes_simd_macros::runtime_dispatch;

/// A lane kernel written once and monomorphized to every backend.
///
/// A closure cannot carry a generic type parameter, so the kernel is a type
/// whose captured state is its fields and whose [`LaneKernel::call`] is generic
/// over the backend. That is the same shape `pulp` and `fearless_simd` use, and
/// the reason both use it: the backend must be a *type* parameter for
/// monomorphization, and only a trait method can bind one.
///
/// Implementations should stay small enough to inline. The whole point of the
/// target-feature scope is that the backend operations fold into the kernel
/// body, which cannot happen across a call boundary the optimizer declines to
/// cross.
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
}

macro_rules! impl_lane_scalar {
    ($($t:ty),+ $(,)?) => {
        $(
            impl LaneScalar for $t {
                #[inline(always)]
                fn run_lane_kernel<K: LaneKernel<Self>>(kernel: K) -> K::Output {
                    dispatch_backend::<$t, K>(kernel)
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
/// which always applies.
///
/// The dispatch decision is made once, here, not per operation inside the
/// kernel — so a kernel that transforms a whole buffer pays for it once.
#[inline(always)]
pub fn vectorize<T: LaneScalar, K: LaneKernel<T>>(kernel: K) -> K::Output {
    T::run_lane_kernel(kernel)
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
