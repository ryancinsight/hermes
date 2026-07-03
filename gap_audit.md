# Gap Audit - hermes-simd

Persistent gap register. Evidence tiers follow the repository instruction
hierarchy: machine-checked proof > type-level invariant > property/fuzz >
differential/empirical > source audit.

## Comprehensive Audit - 2026-07-02 (round 8, 5-agent sweep) <a id="audit-2026-07-02-r8"></a>

Five parallel read-only audits (performance, memory/zero-copy, unsafe soundness,
architecture/redundancy, tests/benches/docs) plus two deep soundness sub-audits.
Evidence tier per item; source-audit unless a differential/property test is named.
Findings are the register below; fixes land as tracked backlog items in triage
order (correctness → architecture → tests → docs → PM).

### Correctness / soundness (HARD)

- **[RESOLVED 2026-07-02] SELL-p vectorized SpMV OOB read (PROVEN, 3 agents).**
  `sparse/spmv.rs` `sellp_spmv_vectorized` gathered `x[col_idx]` and loaded
  `values[offset..]` full-width with no bounds check, reachable from the safe
  `SparseView::<SellP<C>>::spmv` when `LANE_COUNT == C`; the sibling CSR/BCOO
  paths self-defend, SELL-p did not. `SellPMatrix` `pub` fields + no-op `new` +
  opt-in `validate()` let a caller drive both reads out of bounds. Same over-read
  in `sparse/ops.rs` `elementwise_mul_dense`. Fixed by routing both vectorized
  paths through the SSOT `SparseValidate::validate()` via
  `spmv::assert_sellp_validated` before the unsafe kernel; two `#[should_panic]`
  regressions on the Scalar-backed vectorized path (`LANE_COUNT 4 == C`). See
  [Resolved](#resolved).
- **[MITIGATED 2026-07-02] AMX dispatched on CPUID-only; OS tile-data permission
  never requested (PROVEN).** `cpu.rs` `has_amx` checked only CPUID leaf 7; no
  `arch_prctl(ARCH_REQ_XCOMP_PERM, XFEATURE_XTILEDATA)` (Linux) anywhere, so the
  first `tileloadd`/`tdpbf16ps` from `tile_matmul::gemm` `#NM`-faulted on capable
  Sapphire-Rapids Linux hosts; `__cpuid_count(7,_)` also lacked a max-leaf guard
  (leaf-1 aliasing). Interim: the AMX probes now return `false` (AMX dispatch
  disabled), preserving the safe-dispatch contract — the fault can no longer
  occur. **Still open:** restoring AMX behind a permission-aware probe (hardware +
  XCR0 TILECFG/TILEDATA + a one-time Linux XTILEDATA `arch_prctl`). Blocked on: the
  stable toolchain rejects the AMX strings in `is_x86_feature_detected!`
  (`x86_amx_intrinsics` unstable), and verifying the raw `arch_prctl` syscall
  needs an AMX-capable Linux host. `[minor]` DoR: acceptance = AMX GEMM dispatches
  and matches the scalar reference on a Sapphire-Rapids Linux runner.
- **[RESOLVED 2026-07-02] AVX-512 sub-feature gating gaps (PROVEN/SUSPECTED).**
  `cpu.rs` `Avx512Support` read raw CPUID (AVX512_BF16 / VNNI) without OSXSAVE/XCR0
  and, for bf16, probed the unrelated `avx512bf16` bit while the tile kernel
  enables `avx512f,avx512bw,avx512vl` and never uses `dpbf16`. `unpack.rs`
  `widen_i8_to_i16` ran AVX-512**BW** `_mm512_cvtepi8_epi16` under an
  AVX-512**F**-only guard → SIGILL on KNL. Fixed: the two AVX-512 tile probes now
  use `is_x86_feature_detected!` for the exact enabled set (macro handles
  XCR0/max-leaf); `widen_i8_to_i16` gated on `avx512bw`. See [Resolved](#resolved).
- **[open] Unsound safe API surfaces (PROVEN).** `AmxSession::new` (safe `fn`)
  runs `ldtilecfg` with no detection; `Vector::<T, Avx512>::{splat,zero}` +
  operator impls run ISA intrinsics from safe code. In-repo callers all guard,
  but the *API* admits `#UD` from safe code. Fix: unforgeable detection-minted
  arch token, or make these `unsafe fn`/probed. `[minor]`.

### Performance (source-audit tier; each needs a criterion baseline before/after)

- **[REVISED 2026-07-02, measured] SELL-P/BCOO chunk-width dispatch — the
  simple fix is a measured regression; only the AVX-512 case remains open.**
  A chunk-aware ladder routing to the widest ISA whose `LANE_COUNT == C` was
  implemented and A/B-benchmarked on this AVX2 host: for `sellp4` (C=4,
  100k rows, 10% density) the *old* widest-first path ran 7.48 ms
  (13.7 Gelem/s) vs 17.6 ms (5.8 Gelem/s) for lane-matched routing to the
  4-lane scalar-marker kernel — **2.4× slower**, because the "scalar fallback"
  loop executes inside the AVX2 `#[target_feature]` dispatch helper and LLVM
  auto-vectorizes it at full 8-lane width, beating the narrow emulated-gather
  kernel. The change was reverted; a dispatcher-independent SELL-8 multislice
  differential test was kept. Still open, hardware-gated: on an AVX-512 host a
  SELL-8 f32 matrix runs the auto-vectorized fallback where the *native* AVX2
  8-lane gather kernel in the same binary might win — unmeasurable without
  AVX-512. DoR: acceptance = criterion A/B of sellp8 widest-first vs
  AVX2-routed on an AVX-512 runner; do not re-implement without that number.
  The `C = k·LANE_COUNT` kernel generalization remains a separate `[minor]`
  candidate under the same measurement gate.
- **[open] Per-call CSR/BCOO index re-validation (HIGH, 2 agents).** The soundness
  scans in CSR/BCOO (and now SELL-p) re-read `col_indices` every `spmv`; iterative
  solvers call spmv thousands of times on one immutable matrix (+~17-50% index
  traffic per call). Root-cause fix: a `Validated` sparse typestate constructed
  once, kernels trust the type — removes all four per-call scans and closes the
  soundness holes structurally. `[minor]` (supersedes the per-call scans).
- **[RESOLVED 2026-07-03] GEMM/GEMV scalar column tails (HIGH).** **GEMM:**
  `tiled_gemm`'s `n % block_n` trailing columns run `leading_k_mask`-guarded
  fmadd lane groups; measured 3.43× at n=63 (25.17 → 7.33 µs), bitwise
  differential m=7/n=45/k=13, Theorem 1 updated. **GEMV:** the `ncols % lane`
  tail (both blocked and remainder paths) folds into the vector accumulator via
  one masked fmadd (x-tail loaded once, reused across rows). Added the
  workspace's first GEMV bench (`gemv_bench.rs`, tail-isolating); measured at
  cache-resident 256×256 the tail row improved 3.58 → 2.83 µs (+27% throughput),
  aligned neutral within noise, DRAM rows bandwidth-bound/neutral. f32 facade
  differential (n=21, nrows=11, dyadic-exact ⇒ bitwise) + existing f64 tail-shape
  suite. Follow-on candidate: same masked-tail treatment for `gemv_transpose`
  and `axpy` (both still scalar-tailed, both now benchmarkable).
- **[open] AVX-512 bf16 tile kernel: 32 scalar bf16→f32 converts + stack
  round-trip per k-step (HIGH on that path).** `avx512_tiling.rs:35`; vectorize as
  `loadu_si256 + cvtepu16_epi32 + slli_epi32(,16)`, and use `_mm512_dpbf16_ps`
  (AVX512-BF16 already probed, never issued). `[minor]`.
- **[open] Scalar tails on every hot kernel (MED-HIGH).** `view/reduce.rs`,
  `view/ops.rs`, `dispatch/axpy.rs`, etc. end in element-at-a-time loops although
  `leading_k_mask` exists on every backend; up to 15 scalar iters on AVX-512 f32
  tails, dominating short/odd-length vectors. `[minor]`.
- **[REJECTED 2026-07-03, measured] No K/M cache blocking in GEMM.** Hypothesis:
  the full `k × block_n` B panel spilling L1d degrades large-`k` GEMM, and a
  BLIS KC loop bounding the panel to L1d would recover it. **Falsified by
  measurement.** For AVX2 f32 (`TilingPolicy<3,3>`, block_n = 24) the panel is
  `k·96` bytes — 24 KiB at k=256, 48 KiB at k=512, 72 KiB at k=768, 96 KiB at
  k=1024 (2× this CPU's 48 KiB L1d). Measured square-GEMM throughput (criterion):
  256³ = 78.4, **512³ = 69.8**, 768³ = 79.9, 1024³ = **85.6 GFLOP/s** — flat to
  *rising* with `k`, with the largest (most-spilled) panel the fastest. The 512³
  dip is a power-of-two cache-set-conflict artifact (768³, non-power-of-two,
  recovers to 79.9), not L1 spill. The current full-panel pack + L2-residency
  design is correct for this microarchitecture (large fast L2 holds the panel,
  and packing amortizes better over more row blocks as `m` grows); KC-blocking
  would add `⌈k/KC⌉` passes of C load/store to fix a non-problem. Bench rows
  256/512/768 retained as the scaling-regression gate. Not re-opened without a
  microarchitecture whose L2 cannot hold the panel (measured, not assumed).
- **[open] No software prefetch in gather-bound SpMV (MED); no streaming/NT stores
  for out-of-LLC writes (MED); `Aligned` typestate dead at the dispatch facade —
  every op uses unaligned loads and NT stores are blocked on it (LOW-MED);
  uniform `UNROLL_FACTOR=4` under-fills the FMA pipeline for cache-resident
  reductions (LOW-MED); per-call detection branch chain, no cached dispatch
  decision (LOW-MED).** Each `[patch]`/`[minor]`, all measurement-gated.
- **[RESOLVED 2026-07-02] Integer/half emulated-kernel throughput — measured,
  split verdict (host-capability sweep, criterion).** (a) *Integer dense ops*:
  LLVM fully auto-vectorizes the emulated `[i32; 8]` kernels inside the
  `#[target_feature]` wrappers — `sum::<i32>` ~12× scalar (50–62 Gelem/s),
  `dot::<i32>` ~7.4× (bandwidth-bound); hand-written AVX2 integer kernels
  REJECTED as no-win duplication; i32 bench rows are the regression gate.
  (b) *int8 GEMM*: new 256-bit **AVX-VNNI** tile backend (`vpdpbusd` + exact
  +128 bias correction) — 17.3–20.2× measured over scalar tiles; dispatch
  ladder now AMX → AVX-512 VNNI → AVX-VNNI → scalar. (c) *f16*: AVX2 kernel's
  arithmetic core upgraded to **F16C** hardware conversion (bitwise-identical
  to the software semantics) — `dot::<f16>` 221 Melem/s → 7.22 Gelem/s
  (31.7×). (d) *bf16*: ~2 Gelem/s emulated (shift conversion partially
  auto-vectorizes); hardware core deferred until a consumer needs it —
  remaining emulated gap is gather/compress/mask ops (scalar loops), deferred
  until a sparse-integer consumer exists. See CHANGELOG [Unreleased].

### Memory / zero-copy

- **[open] `DenseWithMask` stores `[bool]` mask, converts per-chunk per-call
  (MED-HIGH)** — 8× footprint; bit-pack once at the boundary (a `BitMask` type
  already exists). **`cmp_*_mask`/`cast` round-trip through stack buffers with
  64-iter scalar loops (MED)** — route through native backend mask ops.
  **`AlignedVec` lacks `reserve`/`extend_from_slice`; `Extend for SimdCow`
  pushes item-wise dropping `size_hint` (MED).** **`compress` zero-inits a
  64-elem stack buffer per chunk (MED); argmin/argmax two-pass (LOW-MED).** All
  `[patch]`/`[minor]`.
- **[RESOLVED 2026-07-02] `SimdCow::scale` copy-then-rescale** — now delegates
  to the fused `mul_scalar_cow` (`broadcast_op` SSOT); halves traffic, removes
  the duplicate implementation.

### Architecture / redundancy / hygiene

- **[RESOLVED 2026-07-02] Tracked scratch at repo root** — `apply_changes.ps1`,
  `do_changes.ps1`, `check_errors.txt` deleted from git; untracked logs removed;
  `check_errors.txt` gitignored. Dead dep declarations dropped (`divan`,
  2×`bytemuck`, intrinsics `rkyv`; facade `rkyv` corrected to dev-dependency).
  `benchmarks_baseline.json`/`benchmarks_results.md` kept (live baseline).
- **[open] `codegen.rs` ungoverned SSOT (1334 lines, live generator of the 4 x86
  kernel files).** Not dead — regenerates `avx2_f32/f64`, `avx512_f32/f64`; but no
  `@generated` banner, no CI regeneration-diff gate. `[patch→minor]` process.
- **[open] ~150-200 lines cross-backend scaffold duplication** — compress/expand
  emulation, AVX-512 cmp-mask blend, popcount LUT, masked-reduce, NEON sign-flip
  constants; hoist into `kernel_helpers`/codegen templates per ADR-005. `[minor]`.
- **[PARTIAL 2026-07-02] Doc drift** — README `hermes-numeric` entry replaced
  with the eunomia provenance note; the fictional lib.rs feature table replaced
  with the real set; `gemm_int8` example corrected to `gemm::<i8,i8,i32>`.
  **Still open:** ADR number collisions (two each of 001/002/003), stale
  backlog/checklist `hermes-numeric` refs, `dispatch/mod.rs` (828) split into
  trait/impls/facade. `[patch]`.
- **[open] `widen_I8_*` type-named duplicate API + undocumented unsafe + SIMD
  branch untestable at n=5** — collapse to one generic (`#[repr(transparent)]`),
  add SAFETY, differential test at n ∈ {31,32,33,47,1024}. `[patch→minor]`.

### Tests / benches / docs

- **[RESOLVED 2026-07-02] CI runs bare `cargo test`** — both test jobs now run
  `cargo nextest run --workspace` (committed timeout instrument applies) plus
  explicit `cargo test --doc`; x86_64 job gains a `cargo build --examples`
  rot gate (verified green locally; CI run pending next push).
- **[open] ~25 magic-tolerance assertion sites** vs the repo's own demonstrated
  derivation discipline (complex_tests derives the bound); derive+cite each.
  **AVX-512 differential suite silently skips on CI hosts (unreported).**
  **Stale criterion baseline** — newest GEMM/AMX groups ungated; axpy/gemv/complex
  unbenched. **`missing_docs` absent on `hermes-simd-macros`; ~5 doctests for ~60
  facade fns; `# Errors`/`# Safety` gaps.** **No `cargo build --examples`,
  semver-checks, or bench-compile CI gates.** Each `[patch]`/`[minor]`.

### Verified clean (negative results — do not re-chase)

- Proc-macro `#[runtime_dispatch]` ladder: every `#[target_feature]` helper is
  called only behind a compile-time `cfg!` arm or a runtime `is_x86_feature_detected!`
  gate; no bypass, no env override.
- CSR & BlockedCoo SpMV self-defend on adversarial input.
- `tiling/` dims fully checked (prior overflow fix holds in dev+release);
  `AlignedVec` init paths panic-safe (no uninit read); `SimdView` borrow lifetime
  precludes aliased in-place APIs from safe code.
- AMX inline asm operand/clobber model + `AmxSession` Drop under `panic=abort`
  are sound (tile state released on unwind→abort).
- Sparse formats are SoA with `i32` indices; no `Arc/Rc/Box` nesting; SimdCow
  promotion single-allocation; dispatch facade is out-slice/in-place throughout.

## Allocator Dependency Audit - 2026-06-28 (round 7) <a id="audit-2026-06-28-r7"></a>

hermes is unchanged since round 6 and remains lean (no new findings). This round
audited the upstream allocator (`mnemosyne`) that backs `AlignedVec`/the global
path, which was concurrently rewritten lock-free (segment + huge pools as tagged
Treiber stacks; bucket lock removed). Adversarial concurrency review found **no
memory-safety bug**: 16-bit tagged pointers (address in low 48 bits, tag in high
16) are masked before every deref; push/pop CAS loops pair Release/Acquire and
bump the tag (ABA-immune); `take_all` is a single Acquire swap; the huge-pool
first-fit scan **pops-before-touch** (CAS-removes each node before reading it),
avoiding the classic lock-free use-after-free. **Verified hermes integration:
371 workspace tests pass against the lock-free allocator.**

Residual risks (upstream `mnemosyne`, surfaced for the owner — not reworked here,
as it is another agent's fresh, tested code):
- No `loom` model for the lock-free pools — correctness rests on design reasoning
  + std-thread stress tests (empirical tier), not machine-checked interleavings.
  The repo's own rule asks for `loom` alongside stress tests for lock-free code.
- `take_all` head-swap and count-reset are separate atomics → a push interleaving
  between them transiently skews the advisory `retained`/`total_count` counters
  (telemetry only; no safety/correctness impact under the documented contracts).
- Tag lives in the high 16 address bits, so addresses ≥ 2^48 (LA57 / AArch64
  52-bit VA) trip a fail-safe `abort` rather than corrupting — a portability
  limit, not UB. Low-bit tagging (segments are 2 MiB-aligned ⇒ 21 free low bits)
  would remove the dependency and widen the tag.

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

Resolved later (2026-06-26, scope review):
- CSR `spmv` SIMD-gather column-index bounds (round-1 finding): now validated
  with a linear pre-loop scan in the CSR kernel. CSR SpMV is gather/latency-bound,
  so the O(nnz) linear validation is cheap relative to the random-access gathers
  it guards — the earlier "too expensive" deferral reasoning did not hold for this
  kernel. `spmv_csr` is now sound on adversarial input (negative/oversized indices
  rejected); covered by a `#[should_panic]` test. Sparse SpMV is otherwise a
  low-value SIMD target (gather-bound); the high-value sparse path, SpMM, already
  vectorizes via `axpy`. Consumers (e.g. leto) own format validation at their CSR
  construction boundary and now always route dense/SpMM ops through hermes (the
  `simd` cargo feature was removed downstream — SIMD is the unconditional path via
  hermes's runtime dispatch).

## Internal Audit - 2026-06-26 (round 6, closure) <a id="audit-2026-06-26-r6"></a>

Largely a verification/closure round — 5 prior passes + tool measurements show the
workspace is lean (lib IRs: `mnemosyne` ~3045, `hermes-simd` ~593, `leto-ops` ~167
lines; monomorphization deferred to leaf binaries as intended; no `dyn` on hot
paths; inner-fn extractions confirmed 1-copy). No substantive perf/memory/mono
change was warranted; manufacturing churn would violate the subtractive bias.

Resolved this sprint:
- [patch] Closed the lingering round-1 bitboard finding. `hermes_simd::{rook,
  bishop,queen}_attacks` are safe wrappers over the `Magic` `unsafe` kernel.
  Verified **not** an OOB/UB hole: the kernel indexes `[u64; 64]` magic/offset
  tables and the attack `Vec` with bounds-checked indexing and a computed mask, so
  `square >= 64` **panics**, never reads out of bounds. Added the missing
  `// SAFETY:` justification and `# Panics` docs on the wrappers and a
  `#[should_panic]` regression test. Evidence tier: source audit + test.

## Internal Audit - 2026-06-26 (round 5, monomorphization) <a id="audit-2026-06-26-r5"></a>

Evidence tier: value-semantic differential test (BCOO SIMD) + source-grounded
monomorphization analysis.

Resolved this sprint:
- [patch] hermes `spmv_bcoo` was hardcoded to `ScalarArch` (the only sparse op
  not runtime-dispatched), leaving the SIMD BlockedCoo kernels dead at runtime —
  a perf defect, not cleanup. Now routed through `#[runtime_dispatch]`
  `dispatch_spmv_bcoo`; differential test added for the SIMD branch.
- [patch] hermes `axpy_rows_batch`: extracted the type-independent extent
  validation to a non-generic `#[inline(never)]` fn (emitted once vs. per
  `(T, Arch)`). The validation is run-once-per-call (not the hot loop), so the
  dedup carries no hot-path cost — the correct application of the inner-function
  pattern here.

Verified clean / not pursued (monomorphization):
- Tiling const-generics (`<6,4>`/`<3,3>`/`<1,1>` …) are measured-win register
  blocking dispatched by `LANE_COUNT` — must NOT be collapsed to runtime params.
- Cross-crate inlining is complete (all `SimdKernel` methods `#[inline(always)]`);
  no `dyn`/`Box<dyn>` on any compute path; the one in-loop branch and
  `flush_limit_for::<T>()` are const-foldable (DCE handles per-instance).

Measured and closed (cargo-llvm-lines / cargo-bloat, 2026-06-26):
- Mnemosyne page-list ops (`push_page_front`/`unlink_page_from_list`/`move_*`) —
  **confirmed not worth deduping.** `cargo llvm-lines -p mnemosyne` does not list
  them at all (they are `#[inline(always)]`, fully inlined), and the whole
  `mnemosyne` crate is only ~3045 IR lines, so an `#[inline(never)]` inner-fn
  extraction would save negligible IR while adding a call on the hot free path.
  `#[inline(always)]` (as one agent suggested) dedups nothing. No change; the
  earlier deferral was correct. Tier: empirical (IR measurement).
- hermes monomorphization is lean — the round-5 inner-fn extractions
  (`check_axpy_rows_batch_extents`, `validate_gemm_sizes`) show as **1 copy** in
  `cargo llvm-lines -p hermes-simd` (deduped as intended); the lib's own IR is
  ~593 lines, and an example binary's `.text` is dominated by std runtime glue
  (`rust_eh_personality`), not hermes monomorphization. No bloat to attack.

## Internal Audit - 2026-06-26 (round 4) <a id="audit-2026-06-26-r4"></a>

Evidence tier: value-semantic tests (hermes numeric contract; mnemosyne
`take_all`) + source-grounded contention analysis.

Resolved this sprint:
- [patch] hermes `hermes-numeric`: signed-integer `NumericElement` impls collapsed
  into one `impl_numeric_element_signed!` macro; dead `min_scalar`/`max_scalar`
  integer overrides removed (identical to trait defaults). ~275 fewer lines.
- [patch] hermes `hermes-simd-intrinsics`: AMX raw tile wrappers no longer
  silently no-op on an out-of-range tile (`unreachable!`); `AmxGemm::amx_gemm`
  `# Safety` documents the AMX-availability precondition (already gated by the
  `has_amx()` dispatch probe — not an unguarded hole).
- [upstream, Mnemosyne `perf/segment-purge-batch-detach`] `purge`/`reset` segment
  sweeps batch-detach each node's chain under one lock (`NodeSegmentPool::take_all`)
  instead of one lock per segment — removes decay↔allocator serialization. Pool
  node arrays now built from the `NUMA_BUCKETS` SSOT.

Considered, deferred (recorded):
- NEON `neon_f32`/`neon_f64` (~92% overlap) is seam-level, not a clean macro: the
  divergent 8% (popcount reduction depth, `cmp_ne` u64 round-trip, `swap_adjacent`
  instruction, mask construction) needs a `codegen.rs`-style template, not a thin
  suffix macro. Route through the codegen generator if pursued.
- scalar `f32`/`f64` kernels are a cleaner macro/const-generic candidate (no
  intrinsics); deferred to keep this round focused.

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

- [patch] AVX-512 tile-kernel detection soundness (2026-07-02). The `Avx512Support`
  probes read raw CPUID without OSXSAVE/XCR0 (a host advertising the bit without OS
  XSAVE enablement would `#UD`) and, for bf16, tested the unrelated `avx512bf16`
  dot-product bit while the tile kernel enables `avx512f,avx512bw,avx512vl` and
  never issues `dpbf16` — both a fault window and a false skip on capable non-bf16
  parts. Now `is_x86_feature_detected!` for the exact enabled set per kernel
  (`avx512f,avx512bw,avx512vl`; `avx512f,avx512vnni,avx512vl`), which handles XCR0
  and the leaf-7 max-leaf internally. `widen_i8_to_i16`'s AVX-512 branch now gates
  on `avx512bw` (the `_mm512_cvtepi8_epi16` requirement) instead of the
  `avx512f`-only `TargetId::Avx512`, closing a KNL `#UD`. AMX detection is
  conservatively disabled in the same change (see round 8, MITIGATED). Evidence
  tier: source audit + toolchain feature-availability (AMX strings unstable).
- [patch] SELL-p vectorized SpMV / elementwise OOB read (2026-07-02). The
  `LANE_COUNT == C` fast path in `sparse/spmv.rs::sellp_spmv_vectorized` and
  `sparse/ops.rs::elementwise_mul_dense` gathered `x[col_idx]` and loaded
  `values[offset..]`/stored `out_values[offset..]` at full vector width with no
  bounds check, reachable from the safe `SparseView::<SellP<C>>::{spmv,
  elementwise_mul_dense}` — a proven OOB read from safe code on a
  caller-constructed matrix (`pub` fields, no-op `new`, opt-in `validate()`).
  Unlike the sibling CSR/BlockedCoo paths, SELL-p never validated. Fixed by
  routing both vectorized paths through the SSOT `SparseValidate::validate()` via
  the new `spmv::assert_sellp_validated` before the unsafe kernel (checks
  `col < ncols`, `col_indices.len() == values.len()`, and
  `slice_ptr[s] + slice_col_count[s]·C <= values.len()`), plus an
  `out_values.len() >= values.len()` guard on the elementwise store. Two
  `#[should_panic]` regressions drive the Scalar-backed vectorized path
  (`Scalar::LANE_COUNT 4 == C`, host-independent) with an out-of-range column and
  with over-long slice geometry. Evidence tier: type-level precondition + value
  regression. See [round 8](#audit-2026-07-02-r8).
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
- [minor] Masked-merge `SimdKernel` defaults (2026-06-28). Investigation of the
  remaining monomorphization gaps for SIMD capability expansion found the seam
  already mature: `rsqrt`, `popcount`, horizontal-bitwise reductions (NumKong
  P1/P2 — see [numkong-2026-06-17](#numkong-2026-06-17)), and the
  reduction/scan/unary op families are expressed as defaulted `SimdKernel`
  methods (`kernel_helpers`) or sealed ZST strategies, so each is one generic
  addition inherited by every backend. The single family still `required` on
  every impl was the masked-merge set (`masked_load_unaligned`,
  `masked_store_unaligned`, `masked_add`, `masked_mul`, `masked_fmadd`,
  `masked_sum_reduce`) — the NumKong P1 tail-free family — which now has
  scalar-emulated trait defaults (arithmetic via `blend(mask_to_vector(mask), …)`;
  load/store via new `kernel_helpers::generic_masked_{load,store}`). A new
  backend/type inherits the tail-masked family for free; the six redundant impls
  are removed from `impl_emulated_kernel!` (inherited by ~24 emulated backends).
  Bit-identical to the removed per-element loops, verified by a new cross-backend
  differential property test (Scalar/SveArch defaults vs AVX2/AVX-512 native
  overrides). Evidence tier: differential test across default and native paths.
  Not defaulted: `gather`/`compress`/`expand` stay `required` — no generic
  `IndexVector`/lane-introspection primitive exists to express them, and gather is
  latency-bound so the value is low. Remaining NumKong families (native `rsqrt`
  instruction override, sub-byte unpacking, Arm SME) are capability *additions*,
  not monomorphization debt — each is now a single defaulted-method or
  strategy-ZST addition rather than an N-impl change.

- [patch] Tiling dimension-product overflow → OOB SIMD load (2026-06-28). The
  GEMV/GEMM operand-length checks (`tiling/gemv.rs`, `gemv_transpose.rs`,
  `gemm.rs`) computed the required span with unchecked `usize` products as the
  only guard before `unsafe` SIMD loads/stores. An adversarial dimension from the
  public dispatch API (`lda = usize::MAX`, `nrows = 2`; or `m·k` etc.) overflowed:
  release (`overflow-checks = false`) wrapped → guard passed → OOB read; dev
  panicked undocumented. Fixed by an SSOT `tiling::dims` module
  (`checked_strided_span`/`checked_area`) returning `SimdError::LengthMismatch` on
  overflow — closes the OOB path in all profiles and consolidates the previously
  duplicated forward/transpose span math. Added `[profile.dev] overflow-checks =
  true` per the numerical-discipline mandate (release keeps default for hot-loop
  speed; the checked guard makes safety profile-independent). Evidence tier:
  value-semantic exact-variant regression tests on all three dispatchers passing
  in both dev and release (release pass proves the OOB load unreachable) +
  `tiling::dims` unit tests. Prior rounds closed per-element sparse-load overflow
  but never this dense dimension-product class.

- [patch] Integer `sqrt` f64-roundtrip precision loss (2026-06-28). Integer
  `NumericElement::sqrt` used `(self as f64).sqrt() as Self`, which rounds operands
  above 2⁵³ to `f64` before the root — wrong for large `i64`/`u64` (`u64::MAX`
  returned 2³² instead of 2³²−1; the result's square overflows). Replaced with
  exact `isqrt`; signed negatives keep the documented degenerate contract (→ 0;
  integers have no `NaN`), trait doc now states the contract. The audit's
  companion flag (`f16`/`bf16` `to_f64` via `to_f32`) was assessed **benign** —
  widening to a wider mantissa is lossless — and left unchanged. Evidence tier:
  value-semantic regression tests (large-operand exact cases, the
  `r² ≤ n < (r+1)²` invariant above 2⁵³, negative-input contract) over all eight
  integer types; integer `sqrt` previously had zero test coverage and no callers.

- [patch] `recip_sqrt` backend-inconsistent precision (2026-06-28). The f64 SIMD
  paths (avx2/avx512/neon) and NEON f32 reused the f32 single-Newton-step pattern,
  which under-refines a low-bit hardware `rsqrt` seed: `recip_sqrt::<f64>` ranged
  ~1e-16 (scalar) → ~4e-9 (avx512) → ~6e-8 (avx2) → ~1.5e-5 (neon), and NEON f32
  reached only ~16 bits — a native-precision violation masked by perfect-square
  test inputs (Newton converges exactly by luck) + magic `1e-4`/`1e-6` tolerances.
  Fixed to full native precision (~1 ulp) on all backends: f32 retains fast
  `rsqrt`+Newton (NEON now two steps for full 23-bit); f64 uses correctly-rounded
  hardware `sqrt`+divide (3 Newton steps would otherwise be needed). Contract
  documented on the trait method. Evidence: cross-backend differential test with
  analytically-derived relative bounds (`8·ε_f32`, `4·ε_f64`) over non-perfect-
  square inputs — x86 backends runtime-verified locally; **NEON is verified only on
  the aarch64 CI runner** (no local aarch64 target), though the new intrinsics are
  the same ones already used by the NEON `sqrt`/`div`/`splat` primitives.

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
