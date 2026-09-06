# ADR 009: Monomorphized Target-Feature Gate Helpers for Zero-Overhead Inlining

## Status

Accepted

## Context

In Rust, `#[target_feature]` attributes restrict compiler inlining. A function marked with `#[target_feature(enable = "avx2")]` cannot be inlined into a function that lacks that attribute, because doing so could leak AVX2 instructions into context where they might execute on non-AVX2 hardware (causing `SIGILL`).

In the original dispatch design, generic `SimdView::sum()` was called directly from the public dynamic dispatch functions (`sum_f32` etc.). Because the public functions themselves do not have `#[target_feature]` attributes (since they are the entry point and must run on all CPUs), the compiler was forced to monomorphize `SimdView::sum` without AVX2/AVX-512 target features enabled. Consequently, calls to `Avx2::load_aligned`, `Avx2::add`, etc. (which are annotated with target features) could not be inlined into the loop body, introducing heavy function call overhead inside the critical inner loops and completely neutralizing SIMD throughput advantages.

## Decision

We introduced monomorphized local helper functions for each target architecture and annotated them with target features (e.g. `sum_f32_avx2` with `#[target_feature(enable = "avx2")]`). Inside these helpers, the view is constructed and `view.sum()` is called.

The public dispatch functions first check compile-time features (via `cfg!(target_feature = "...")`), which has zero runtime cost. If compile-time features do not match, it checks runtime features (via `std::is_x86_feature_detected!`) and calls the corresponding helper function safely.

Scalar/backend pairs may require a stricter frame than their public architecture
marker. Dispatch recovers the kernel scalar type from an inline or `where`-clause
`SimdKernel<T>`-family bound and selects the complete feature set once at the
operation boundary. The AVX2 F16 route therefore enters an `avx2,fma,f16c`
helper and monomorphizes the kernel with a private proven marker whose arithmetic
contains no feature probe. Callback adapters make elided borrowed outputs use
the sole normalized input lifetime and preserve explicit output lifetimes. The
ordinary public `Avx2` marker remains unchanged: direct F16 operations probe
F16C and use their software fallback when it is absent. The private marker is
never re-exported; doc-hidden callback traits bridge proc-macro expansion across
the crate boundary without adding another consumer-selectable backend.

## Consequences

The [complex-permutation forwarding increment](../../backlog.md#hermes-complex-permutation-inlining)
applies the same boundary rule to `SimdPermute::transpose_interleaved_square`.
Apollo's retained `07bd618` Leto candidate emits the unannotated generic facet
as a separate AVX-512 function with a 1,272-byte frame and 20 load/store calls.
The initial AVX2 specialization folds the same backend default into register
operations. A monomorphized method alone therefore does not establish zero
runtime overhead across the facet.

The bounded correction marks only this forwarding method `inline(always)`.
The selected target-feature kernel continues to own capability proof, and
`BackendKernel` remains the single algorithm source. No native shuffle network,
new ISA selection or global compiler flag is introduced. This annotation does
not authorize target instructions at an unproven public boundary: backend
target-feature functions retain their compiler-enforced call restrictions.

The entry baseline at `e6e08211` passes 16 existing permutation and host-
capability tests. Acceptance additionally requires unchanged workspace gates,
downstream bitwise/allocation oracles and matching assembly showing the
forwarding call and its row-buffer staging eliminated. The complete Apollo
engine census and executable size decide adoption; no timing result is
established by this source change.

The local provider gates pass 548 workspace native tests, 16 focused release
tests without profile overrides, 26 doctests (10 existing ignored), all-target
Clippy, examples, warning-denied rustdoc and no-default-feature compilation.
The release pass also corrects the CI step label's stale claim that the test
harness conflicts with the release panic setting. Host reporting identifies
Scalar, AVX2 and emulated SVE execution; it does not establish AVX-512 or NEON
runtime coverage on this workstation. Source hashes and command outputs are
retained in Atlas `output/hermes-complex-permutation` under the existing
14-day/10-GiB policy. Downstream codegen and timing remain outstanding.

- **Loop inlining:** the helper's complete target-feature set permits backend
  operations to inline into the kernel. Exact assembly remains the acceptance
  oracle because a generic abstraction alone does not prove this codegen.
- **Boundary-only detection:** runtime detection executes before the helper;
  selected arithmetic bodies contain no detection branch or cache load.
- **Static selection:** a build whose target features are known can remove the
  runtime detection arms through `cfg!` folding.
- **Safety gating:** every target-specific helper is called only after its
  complete feature set is established. no_std retains compile-time selection
  and the portable fallback.
- **Direct-use compatibility:** `Simd<F16, Avx2>` remains safe on AVX2 hosts
  without F16C because it retains the operation-level software fallback; only
  dispatch chooses the probe-free private frame.

## Revision history

- **2026-08-28 — scalar-specific target frames.** HS-F16-DISPATCH-PROBE-HOIST
  extended the original per-architecture decision to per-(architecture,
  scalar) feature requirements. On an Intel Core Ultra 9 285K, the unchanged
  16,384-element F16 dot benchmark moved from 2.489–2.554 µs before the change
  to 1.086–1.099 µs across two confirmation runs. Release assembly for the
  selected helper contains 68 `vcvtph2ps`, 30 `vcvtps2ph`, 6 `vaddps`, 10
  `vmulps`, and 14 `vfmadd*` instructions, with zero feature-cache references
  and zero calls to the fallback arithmetic wrappers. These observations prove
  this benchmark and emitted body on that machine; they do not establish
  throughput on other microarchitectures.
