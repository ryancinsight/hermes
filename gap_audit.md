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
- P2: delivered 2026-06-21 as SSE2 backend feasibility ADR (ADR 006) covering trait coverage, CI value, and maintenance cost.
- P2: delivered 2026-06-15 as host-supported `TargetId` dense conformance
  tests against Scalar for reductions, elementwise arithmetic, gather, and
  select.
- P3: delivered 2026-06-17 as a per-family coverage map in README and
  backlog, with consumer-demand admission rules for pending families.

## Consumer-Driven SIMD Coverage - 2026-06-15

Evidence tier: value-semantic differential and boundary tests.

- [minor] Batched dense row-panel accumulation: delivered `axpy_rows_batch`
  as one runtime-dispatched fused AXPY-family kernel. The API avoids repeated
  public facade dispatch for depth-major row-panel consumers, allocates no
  temporaries, and keeps output memory traffic to one load/store per output
  lane by accumulating across depth in registers. Coverage compares against
  repeated `axpy_rows` and asserts exact `SimdError::LengthMismatch` failures
  for invalid output stride, alpha panel, and RHS panel extents. Benchmark
  coverage now compares `axpy_rows_batch` against repeated public `axpy_rows`
  on the same depth-major row panels.
- [patch] Dense/AXPY error-contract hardening: selected length-mismatch tests
  now assert exact `SimdError::LengthMismatch` values. This removes
  existence-only failure assertions from the touched dense facade and AXPY
  contract surface.
- [patch] Select/unary error-contract hardening: selected select, unary-map,
  and COW FMA tests now assert exact `SimdError` variants for length mismatch
  and insufficient output capacity.
- [patch] Operation-family error-contract hardening: selected new operation,
  strategy, complex, gather, scan, and COW math tests now assert exact
  `SimdError` variants instead of existence-only failures.
- [patch] COW unary invariant cleanup: `SimdCow::map_unary` no longer
  discards the `SimdView::map_unary` result; the locally constructed equal
  length invariant is explicit in the panic message.
- [patch] GEMM tiling rustdoc cleanup: private implementation names in module
  theorem prose no longer emit public rustdoc private-link warnings.

## NumKong Reference Audit - 2026-06-17 <a id="numkong-2026-06-17"></a>

Reference: `https://github.com/ashvardanian/NumKong` (successor to SimSIMD)

Evidence tier: source audit of features and design philosophy, plus local Hermes code search.

Scope fit:
- In scope for Hermes: Low-level SIMD primitive extensions (fast reciprocal square root with Newton-Raphson refinement, active-lane masked load/store primitives, horizontal reductions and bitwise population count `popcnt`, sub-byte/integer widening/unpacking), target-safe CPU architecture probes.
- Out of scope for Hermes: High-level vector search similarity/distance metric algorithms (Cosine Similarity/Distance, Jaccard Index, Hamming Distance, KL/JS Divergence, geospatial distance) which belong in the Leto operations layer and the Hephaestus GPU layer; thread/MIMD execution scheduling (Moirai).

Findings:
- [minor] Masked tail-load/store elimination: NumKong completely eliminates scalar tail loop overhead on hardware that supports masking (AVX-512, SVE) by using active lane masks. Hermes currently defaults to scalar tail loops in [mod.rs](file:///d:/atlas/repos/hermes/crates/hermes-simd-core/src/ops/mod.rs) for irregular lengths. Supporting first-class masked load/store APIs in Hermes would allow Leto to bypass scalar tails in its domain kernels.
- [minor] Fast reciprocal square root: NumKong optimizes vector norms and Cosine similarity by using hardware-native fast reciprocal square root approximations (`rsqrtps` on x86, `frsqrte` on Arm) refined with a Newton-Raphson iteration. Hermes lacks a fast reciprocal square root strategy in [unary.rs](file:///d:/atlas/repos/hermes/crates/hermes-simd-core/src/ops/unary.rs) and iterates standard/vector `sqrt`.
- [minor] Popcount and horizontal reductions for binary/integer metrics: Binary and integer distance calculations (like Jaccard and Hamming) in Leto require highly efficient SIMD population count (`popcnt`) and horizontal reductions (bitwise reductions). Hermes lacks generic `popcnt` and horizontal bitwise fold primitives.
- [minor] Low-precision integer/sub-byte unpacking: NumKong leverages VNNI algebraic transforms and sub-byte type unpacking (e.g. 4-bit/6-bit) to accelerate low-precision dot products and similarity metrics. Hermes defines sub-byte scalar types like `Bf4` and `F4` but has not exposed vector sign-extension, widening, and byte-alignment unpacking primitives.
- [minor] Arm SME (Scalable Matrix Extension) support: NumKong utilizes Arm SME for tiled matrix multiplication on newer hardware (Apple M4/M5). Hermes lacks SME target feature probes and kernels.

Decisions:
- Do not implement similarity/distance metrics (Cosine, Jaccard, Hamming, KL/JS Divergence) directly in Hermes. These belong in Leto/Hephaestus.
- Implement a Hermes-native fast reciprocal square root (`ops::RecipSqrt` or `rsqrt`) with a Newton-Raphson refinement step to enable fast norm computations in Leto.
- Expose masked load/store primitives on `Vector<T, Arch>` / `SimdKernel` for AVX-512 and SveArch to enable Leto to construct tail-free kernels.
- Add population count (`popcnt`) and horizontal bitwise reduction primitives to the Hermes facade to enable Leto to implement Hamming and Jaccard distance metrics.
- Expose low-precision integer/sub-byte unpacking and sign-extension primitives to support VNNI/Neon dot product optimizations in Leto.
- Defer Arm SME implementation until SVE vector types are fully stabilized and verified.

Next increments:
- P1: Fast reciprocal square root (`ops::RecipSqrt`) with Newton-Raphson refinement.
- P1: Masked tail-load/store API infrastructure for `Avx512` and `SveArch` to enable Leto to run tail-free kernels.
- P2: Expose population count (`popcnt`) and bitwise horizontal reduction primitives to enable Jaccard/Hamming in Leto.
- P2: Expose sub-byte sign-extension and unpacking/widening SIMD primitives (for `Bf4`/`F4`/`I8`).
- P3: Arm SME target-feature feasibility study.

## Residual Risks

- AVX-512 and AMX runtime validation still depends on matching hardware.
- Native SVE remains a planned backend; current `SveArch` coverage is an
  emulated value-semantic backend.
- The local `[patch]` graph warns that `mnemosyne-heap` is unused; this is not
  introduced by the Highway audit, but remains a supply-chain hygiene item.
