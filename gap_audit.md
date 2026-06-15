# Gap Audit - hermes-simd

Persistent gap register. Evidence tiers follow the repository instruction
hierarchy: machine-checked proof > type-level invariant > property/fuzz >
differential/empirical > source audit.

## Highway Reference Audit - 2026-06-14 <a id="highway-2026-06-14"></a>

Reference: `https://github.com/NikoMalik/highway.git` at
`0984271e74db124cf5e200de542e745348eb0b9e`.

Evidence tier: source audit plus local Hermes code search. No benchmark or
correctness claim is made from this audit alone.

Scope fit:
- In scope for Hermes: target-safe runtime dispatch, lane/mask API coverage,
  safe slice wrappers over unsafe kernel primitives, cross-target conformance
  tests, and x86 baseline coverage below AVX2.
- Out of scope for Hermes: replacing Hermes' domain-specific sparse, packed,
  AMX, tensor, COW, and Atlas-boundary surfaces with Highway's `WithSimd`
  user-kernel model.

Findings:
- [minor] Target-token dispatch safety: Highway exposes a `TargetId` +
  `dispatch_to` path that verifies target support before entering
  `#[target_feature]` trampolines. Hermes has runtime-dispatched public
  functions and direct architecture markers, but no single explicit forced
  target API for tests/benchmarks.
- [minor] Safe slice memory wrappers: Highway separates raw-pointer unsafe
  loads/stores from safe bounds-checked slice wrappers. Hermes has typestate
  views and `AlignedVec`, but `SimdKernel` load/store methods remain raw
  unsafe primitives without a small safe wrapper layer for one-vector
  load/store use cases.
- [minor] SSE2 baseline backend: Highway includes SSE2 as a 128-bit x86_64
  target between Scalar and AVX2. Hermes currently jumps from Scalar to AVX2
  on x86; this leaves older x86_64 machines and conservative CI targets with
  only scalar execution.
- [minor] Cross-target conformance matrix: Highway tests operations by forcing
  every available target and comparing results. Hermes has backend property
  tests and host capability tests, but no common forced-target matrix covering
  the public dense facade consistently across Scalar/AVX2/AVX-512/NEON.
- [minor] Operation-family gap map: Highway documents a broad operation catalog
  across arithmetic, bitwise, comparison, masks, conversions, shuffle/rearrange,
  reductions, float, memory, and crypto. Hermes has strong dense/sparse/packed
  domain kernels, but backlog coverage for missing primitive families is still
  coarse (`gather/scatter variants, additional reductions/scans`).
- [patch] README positioning: Hermes README did not identify the Highway audit
  baseline, making it harder to distinguish intentional scope differences from
  missing SIMD substrate capabilities.

Decisions:
- Do not adopt Highway's `WithSimd` user-kernel model as a replacement for
  Hermes' sealed `SimdKernel` + facade APIs. Hermes' current shape preserves
  Atlas-owned domain kernels and monomorphized public operations.
- Use Highway as a coverage checklist for portable SIMD substrate gaps. Each
  accepted gap must land as a Hermes-native trait/API/test increment with
  value-semantic verification.

Next increments:
- P1: delivered 2026-06-14 as `TargetId`, `dispatch_view_to`, and
  `dispatch_view_mut_to`, with unsupported targets rejected before typed view
  construction.
- P1: delivered 2026-06-14 as safe one-vector `Vector<T, Arch>` slice
  load/store wrappers with length and alignment failure tests.
- P2: add SSE2 backend feasibility ADR covering trait coverage, CI value, and
  maintenance cost.
- P2: delivered 2026-06-15 as host-supported `TargetId` dense conformance
  tests against Scalar for reductions, elementwise arithmetic, gather, and
  select.
- P3: turn the operation-family gap map into per-family checklist rows driven
  by consumer demand.

## Residual Risks

- AVX-512 and AMX runtime validation still depends on matching hardware.
- Native SVE remains a planned backend; current `SveArch` coverage is an
  emulated value-semantic backend.
- The local `[patch]` graph warns that `mnemosyne-heap` is unused; this is not
  introduced by the Highway audit, but remains a supply-chain hygiene item.
