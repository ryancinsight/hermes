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
- [patch] Runtime FMA capability probe: `has_fma3` no longer relies on the raw
  CPUID FMA bit alone; it follows Rust's runtime feature detector and is tested
  against `std::is_x86_feature_detected!("fma")` on x86 hosts.
- [patch] GEMV rustdoc link cleanup: public dispatch docs no longer emit
  ambiguous intra-doc links for same-named GEMV modules and functions.

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

## Allocator / Cross-Repo Audit - 2026-06-26 <a id="alloc-audit-2026-06-26"></a>

Memory-efficiency deep dive on the hermes ↔ Mnemosyne boundary. Evidence tier:
empirical (measured mapped bytes via `mnemosyne::memory_stats`) + type-level
soundness argument for the alignment guarantee.

Root cause found: Mnemosyne routed every allocation with `align > 16` to its
large/huge path, reserving a ~2 MiB segment each (committed on Windows). Because
hermes allocates `AlignedVec<_, Aligned<64>>` pervasively, small SIMD buffers
cost ~2 MiB apiece — **512 live 256-byte/64-aligned `AlignedVec`s mapped
~1056 MiB**. The hermes-side `adjust_layout_for_mnemosyne` 8 KiB padding was a
*counterproductive* workaround for a different (size-based) tcache concern: it
inflated small unaligned allocations into the same huge path without the claimed
benefit (live RSS unchanged in measurement, since the 2 MiB slack is decommitted
for align ≤ 16).

Resolved this sprint:
- [upstream, Mnemosyne `perf/aligned-small-alloc-tcache`] Alignment-aware
  size-class selection: small allocations whose chosen class block stride is a
  multiple of the requested alignment now use the thread-cache path. Sound
  because page starts are `PAGE_SIZE`-aligned and blocks are carved at
  `block_size` stride. Non-power-of-two-stride classes still fall to huge.
  Verified by a value-semantic alignment/usability test.
- [patch] hermes: removed the `adjust_layout_for_mnemosyne` padding and the no-op
  `dealloc_on_node` NUMA bind. With the upstream fix, the same 512-allocation
  workload drops **~1056 MiB → ~4 MiB** mapped (264×).
- [patch] hermes: BlockedCoo `spmv`/`elementwise_mul_dense` unchecked SIMD column
  loads now bounds-guarded (O(nblocks), pre-loop).

Deferred (recorded, not silently dropped):
- The four GEMV dispatchers share a thin register-blocking skeleton, but each
  carries a distinct theorem (operand-reuse vs output-reuse), tile orientation,
  and test suite. The *kernels* are already deduplicated; collapsing the
  dispatcher glue into a proc-macro-attributed `macro_rules!` would obscure the
  per-variant documentation/tests for marginal gain. Left as four clear files.
- CSR `spmv` SIMD gather still trusts column indices in range (round-1 finding);
  the principled fix is index validation at `SparseView` construction, an O(nnz)
  boundary check warranting its own design increment.

## Internal Audit - 2026-06-26 (round 3) <a id="audit-2026-06-26-r3"></a>

Evidence tier: compile-time invariant encoding + value-semantic tests (hermes);
value-semantic test + source-grounded retention analysis (Mnemosyne).

Resolved this sprint:
- [patch] hermes `view/vector_reg.rs` was the one module left out of the
  `MAX_SIMD_LANES` SSOT migration: 10 sites used `[_; 128]` buffers with dead
  `assert!(lane_count <= 128)` runtime checks. Migrated to `MAX_SIMD_LANES` (64)
  + compile-time `LANE_BOUND_CHECK`; magic `64` OOB guard → `u64::BITS`.
- [patch] hermes `tensor/view.rs` (601 lines) split into a vertical `tensor/view/`
  hierarchy: core (`mod.rs`), `rank_ops.rs`, `simd_bridge.rs` — SoC, pure
  relocation.
- [upstream, Mnemosyne `perf/huge-pool-byte-cap`] Huge-pool retention was bounded
  only by per-bucket block count (1024), allowing ~16 GiB/bucket of idle mappings;
  now byte-bounded per bucket (`bucket_block_cap`, ~256 MiB) while small-huge
  buckets keep the full count cap. Plus a redundant per-`pop` atomic reload removed.

## Internal Audit - 2026-06-26 <a id="audit-2026-06-26"></a>

Four-dimension sweep (safety, contention-free perf, memory, redundancy).
Evidence tier: compile-time invariant encoding and value-semantic tests for the
fixes below; source audit for the deferred item.

Resolved this sprint:
- [patch] Scalar-fallback buffer over-provisioning: `MAX_SIMD_LANES` was `128`,
  2× the true workspace maximum `LANE_COUNT` of `64` (AVX-512 `i8`). Lowered to
  `64`, halving every fallback stack frame; the two divergent local bounds
  (`reduction.rs::finalize` `MAX_LANE_COUNT = 64` debug-assert, and the
  bitmask-buffer `u64::BITS` guard) now fold onto the SSOT under the compile-time
  `LANE_BOUND_CHECK`. Evidence: const-eval catches a too-low bound (AVX-512 `i8`
  fails to build), so the value is the verified tight maximum.
- [patch] NUMA alloc-generation memory ordering: the cross-thread cache
  invalidation counter used `Relaxed` (no happens-before — a reader could trust a
  stale locality flag for a recycled address) and re-read the generation after
  the OS probe (a TOCTOU window stamping pre-bump data with the post-bump
  generation). Now `Release`/`Acquire` with a single pre-probe capture.
- [patch] `build_index_vector` layout invariant: the `&[i32] → IndexVector`
  unaligned read now carries a `const` size assert, so a layout-mismatched
  backend is a build error, not an OOB read.
- [patch] `#![forbid(unsafe_code)]` on `hermes-simd-macros` (no executable
  unsafe — only generated tokens). Magic-table init CAS success ordering relaxed
  to `Relaxed` (winner acquires no shared data).
- [patch] Redundancy: three byte-identical target-gated `SimdOps` impls collapsed
  to one `impl_simd_ops_methods!` macro (mod.rs 1217→845); `flush_limit` deduped
  to a `const fn` SSOT.

## Resolved

- [patch] Scalar-fallback stack-buffer lane bound (2026-06-24). The default
  `SimdKernel` methods and `kernel_helpers` emulations stored a full vector into
  a fixed `[MaybeUninit<T>; 128]` buffer with the `LANE_COUNT <= 128` invariant
  unasserted (and misleadingly half-guarded by `LANE_COUNT.min(128)` on the read
  loop but not the unclamped `store_unaligned`). A backend with `LANE_COUNT > 128`
  (e.g. a future 2048-bit SVE `i8` at 256 lanes — see the SVE residual below)
  would silently overflow the stack. Now encoded at compile time:
  `MAX_SIMD_LANES` SSOT constant + `SimdKernel::LANE_BOUND_CHECK` asserted per
  backend at monomorphization (validated by a deliberate lower-the-bound build
  that fails AVX-512 compilation). `generic_mask_from_bitmask` gains the
  matching `LANE_COUNT <= u64::BITS` guard. Evidence tier: compile-time
  invariant encoding (strongest available).
- [patch] rust-1.95 clippy workspace lints resolved (redundant `as` casts,
  `iter().flatten()`, needless borrow/return, `enumerate()` range loops);
  `cargo clippy --workspace --all-targets -- -D warnings` is clean again.

## Residual Risks

- AVX-512 and AMX runtime validation still depends on matching hardware.
- Native SVE remains a planned backend; current `SveArch` coverage is an
  emulated value-semantic backend. When a native SVE backend lands, `LANE_COUNT`
  may exceed `MAX_SIMD_LANES` for narrow element types; the new
  `LANE_BOUND_CHECK` will flag it at compile time so the scalar-fallback buffers
  are widened (or the backend overrides the affected methods natively) before it
  builds.
- The local `[patch]` graph warns that `mnemosyne-heap` is unused; this is not
  introduced by the Highway audit, but remains a supply-chain hygiene item.
- [minor, deferred] NUMA alloc-generation is a single global counter bumped on
  every dealloc/realloc. On a multi-NUMA + AMX host under heavy alloc churn this
  is a true-sharing serialization point and over-broad (a free on any node
  invalidates every node's thread-local cache). Sharding the generation per NUMA
  node would remove both, but requires threading the node through the allocator
  bump API; deferred as it only affects multi-node AMX hosts and needs careful
  node attribution. The ordering/TOCTOU correctness fix landed this sprint.
- `.config/nextest.toml` added this sprint (30s slow / 60s terminate), making the
  mandated test-time budget enforced rather than implicit. The suite currently
  runs in ~2.4s, well under the threshold.
