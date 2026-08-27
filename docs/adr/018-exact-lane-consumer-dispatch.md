# ADR 018: Exact-Lane Consumer Dispatch

## Status

Accepted

## Context

ADRs 016 and 017 establish one consumer kernel seam: `vectorize` selects the
widest host-supported backend, enters its target-feature scope once, and passes
a capability-carrying `Simd<T, A>` value into `LaneKernel::call`. The lane count
is a property of that selected backend.

That policy cannot express a kernel whose address map is valid at one register
shape. Apollo's verified 128-point FFT base requires four scalar lanes. On an
AVX2 host, f64 selects four lanes and runs. On an AVX-512 host, widest dispatch
selects eight f64 lanes; the kernel can decline, but dispatch has already ended
and cannot retry the supported AVX2 backend. Four f32 lanes exist on NEON and
Hermes' portable packed backend. Calling the kernel at a mismatched width would
corrupt its lane-to-sample map, so silently adapting the width is not valid.

The capability audit in ADR 017 classified fixed-width vectors as a
Fearless-SIMD-broader non-gap because no Hermes consumer required them. Apollo
now supplies that consumer contract. The requirement is exact backend
selection, not a parallel family of fixed-vector storage and arithmetic types.

## Options

1. **Let the widest kernel decline.** This is safe but cannot retry a narrower
   supported backend, so it leaves valid AVX2 execution unreachable on an
   AVX-512 host.
2. **Force a named backend in Apollo.** This couples an FFT algorithm to Hermes'
   x86 and Arm backend names, duplicates capability probing downstream, and
   makes the consumer own the provider's target-feature safety boundary.
3. **Add fixed-width vector types.** This would duplicate the existing
   `Vector<T, A>` operation surface and multiply every backend/type/width
   combination when the present requirement is only dispatch selection.
4. **Add an exact-lane policy to the existing kernel seam.** Select the widest
   supported backend whose `SimdStorage<T>::LANE_COUNT` equals a const-generic
   request, enter its existing target-feature helper, and report expected
   absence without calling the kernel when no backend matches.

## Decision

Adopt option 4 as `vectorize_lanes::<LANES, T, K>(kernel) -> Option<K::Output>`.
`LaneKernel`, `Vector`, and `Simd` remain the only consumer kernel and register
types. `vectorize` remains unchanged and continues to select the widest native
backend.

Selection follows the existing architecture order while filtering by the
requested scalar lane count:

- x86/x86-64: AVX-512F, then AVX2+FMA, then scalar;
- AArch64: NEON, then scalar;
- other targets: scalar.

The first supported exact match runs. Hermes' backend named `Scalar` is a
portable packed fallback with scalar-dependent lane counts (four f32 lanes,
two f64 lanes, and eight f16 lanes), not a synthetic one-lane backend. A
request for zero or for an unavailable count returns `None`; the kernel is not
invoked, so a kernel holding mutable borrows cannot partially mutate its
inputs. Backend absence is expected capability selection and is represented by
`Option`, not an error or a silent fallback.

The target helpers retain the ADR 016 safety boundary. Each native helper is
compiled with its backend's target features and is called only after
`SimdArch::is_runtime_supported` succeeds. It then constructs the zero-sized
capability with `Simd::assume_supported` inside that proven scope. Dispatch and
lane-count checks occur once at the operation boundary; the kernel body has no
width or feature branch.

## Consequences

- Apollo can request four lanes without naming AVX2 or NEON. f64 selects AVX2
  even when AVX-512 is also present; f32 selects NEON on AArch64 and the
  portable packed backend on current x86 targets.
- Existing callers and code generation through `vectorize` do not change. The
  new public entry is additive [minor]; the selection policy is [arch].
- No fixed-width vector aliases, conversion layer, compatibility shim, or
  second arithmetic surface is introduced.
- A caller owns the algorithmic fallback after `None`. Hermes does not silently
  run a different-width kernel or substitute scalar execution for a request
  wider than one lane.
- Consumer conformance tests run under `#![forbid(unsafe_code)]`, assert exact
  lane values, prove a nonmatching request does not invoke the kernel, and pin
  portable fallback availability. x86 and AArch64 configurations cover their
  respective four-lane routes.
- Optimized codegen inspection must show target probing only in the entry and a
  branch-free kernel body. Cross-target checks establish compilation, not
  execution on unavailable hardware.

## Revision history

- 2026-08-27: Accepted for `HS-EXACT-LANE-DISPATCH-2026-08-27`, driven by
  Apollo's corrected base-128 evidence at commit `c6f4b639`.
