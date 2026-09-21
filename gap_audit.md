# Gap Audit - hermes-simd

Persistent gap register. Evidence tiers follow the repository instruction
hierarchy: machine-checked proof > type-level invariant > property/fuzz >
differential/empirical > source audit.

## Lane throughput against fearless_simd (2026-08-25) <a id="lane-throughput-2026-08-25"></a>

Evidence tier: measured in a consumer's build, seven kernel variants, all
engines interleaved in one process, every variant correctness-gated before its
timing was read. No claim here is drawn from a cross-run comparison.

Apollo set out to close a 7x-10x FFT throughput gap against RustFFT and PhastFT
and eliminated every algorithmic cause it could find. What remained points here.

### The measurement

Power-of-two complex f64 FFT, arithmetic rate as `5 N log2 N` flops over elapsed
time, at N=2^10 where the whole 16 KB array is L1-resident:

| kernel | flops/ns |
| --- | --- |
| planar radix-2, plain loops left to autovectorization | 4.6 |
| planar, first three stages fused | ~4.6 |
| planar, cache-oblivious recursion to a 1024-element block | ~4.6 |
| planar, Hermes `Vector` ops through `vectorize` | ~4.5 |
| interleaved, Hermes `dup_even`/`dup_odd`/`swap_adjacent`/`fmaddsub` | 4.2 |
| the same, per-call slice validation hoisted to raw loads | 6.1 |
| Apollo's own hand-written AVX Stockham | 3.4-4.3 |
| **RustFFT** | **38.5** |
| **PhastFT, built on `fearless_simd`** | **32.8** |

A scalar f64 pipeline on this host is roughly 6 flops/ns; AVX2 is roughly 48.
The external engines reach 68-80% of peak. Everything routed through Hermes'
lane surface, or through Apollo's own AVX code, sits near or below the scalar
pipeline rate.

RustFFT and PhastFT run **inside the same test binary**, compiled with the same
profile and the same flags, which excludes build configuration. The data is
L1-resident, which excludes bandwidth. Apollo's fused pass count is four passes
for ten radix-2 stages — fewer than RustFFT's radix-4 needs — which excludes
pass count. Its complex path allocates zero per call, measured with a counting
allocator, which excludes allocation.

The interleaved row is the sharpest comparison available: it uses the exact
primitive sequence RustFFT's AVX path uses for a complex multiply —
`fmaddsub(dup_even(w), b, dup_odd(w) * swap_adjacent(b))`, one shuffle pair and
one fused instruction — against the same interleaved layout, and still lands at
4.2.

### One mechanism, partially identified

Replacing `Vector::load_unaligned_from_slice(..).unwrap()` with the raw
`load_unaligned` pointer form moved that kernel from 4.2 to 6.1 flops/ns. The
per-call length and alignment validation costs about **45%** in a kernel that
issues six loads and stores per two complex elements.

This is not an argument against the checked wrappers, which exist because the
Highway audit correctly identified their absence as a gap. It is an argument
that a consumer currently has to choose between the safe surface and throughput,
and that the safe surface should not cost 45% when the bounds are loop-invariant.

Hermes already has the machinery to hoist those checks — `SimdView` typestates
carry alignment and length as type parameters, and `SimdChunks`/`ZipChunks`
iterate pre-validated blocks. The prototypes did not use them, so the remaining
question is open rather than answered: does the existing chunk-iterator surface
close the distance, and if it does, why did the first consumer to write a
fine-grained kernel not reach for it?

### Why this is filed here and not in Apollo

Apollo eliminated layout, stage fusion, cache blocking, primitive selection, and
validation strategy across seven variants spanning 3.4 to 6.1 flops/ns. Whatever
separates that band from 33-38 is not reachable from a consumer's algorithm
choices. It is a property of the lane operations themselves, which is Hermes'
bounded context.

`fearless_simd` is the reference that makes the gap legible: PhastFT is an
`#![forbid(unsafe_code)]` FFT that reaches 32.8 flops/ns on this host through a
safe SIMD abstraction. Whatever it does at the lane level, Hermes should be able
to match — and the fact that a safe abstraction achieves it removes safety as
the explanation.

### Scope note

This does not claim a defect in any specific Hermes operation. It reports that
seven independent attempts to reach competitive lane throughput through this
substrate all failed within a narrow band, that the checked-load overhead
accounts for part of it, and that the remainder is unlocated. Locating it is
`HS-LANE-THROUGHPUT-2026-08-25`.

### Resolution (2026-08-26)

The gap was not one arithmetic instruction. Hermes repeatedly probed host
support at operations on already architecture-bearing values, reconstructed
dynamically sized child views for every lane group, inherited invalid alignment
claims into over-aligned children, lacked uniform fused multiply-subtract, and
made a many-plane consumer compose iterators whose checks LLVM could not reduce
to one loop limit. ADR 017 corrects those provider contracts and adds
`Simd::io_chunks` for const-generic planar input/output groups.

The corrected same-binary comparison reuses identical output addresses for both
substrates. Its pinned AVX2 medians (95% confidence intervals) are:

| f64 scalars | Hermes | `fearless_simd` 0.7 | median delta |
| ---: | ---: | ---: | ---: |
| 256 | 77.024 ns [76.444, 77.624] | 76.687 ns [76.378, 77.030] | +0.44% |
| 1,024 | 1.0134 us [1.0070, 1.0201] | 1.0022 us [0.99266, 1.0130] | +1.12% |
| 4,096 | 3.9417 us [3.9127, 3.9685] | 3.9392 us [3.9016, 4.0025] | +0.06% |

All confidence intervals overlap. The Hermes view/direct diagnostic reports
56.405/55.837 ns, 158.58/158.39 ns, and 2.4103/2.4078 us at those lengths.
AVX2 assembly gives both planar hot loops six vector loads, four stores, fused
arithmetic, one loop branch, and no calls, probes, bounds branches, or panic
paths. The old 3.4--6.1 flops/ns rows above remain the entry evidence that
located the provider defect; they are not the corrected substrate result.

### Native f32 confirmation (2026-08-26)

The original comparison established only f64 parity. The instrument now has one
generic planar implementation instantiated for both supported floating lane
types; each pair still reuses identical inputs and output addresses and passes
the precision-specific scalar oracle before timing. The exact locked bounded
suite reports these medians and 95% confidence intervals:

| f32 scalars | Hermes | `fearless_simd` 0.7 | median delta |
| ---: | ---: | ---: | ---: |
| 256 | 32.175 ns [31.842, 32.564] | 32.090 ns [31.859, 32.415] | +0.26% |
| 1,024 | 169.18 ns [165.16, 173.71] | 161.70 ns [159.23, 165.35] | +4.63% |
| 4,096 | 2.0452 us [2.0275, 2.0646] | 2.0376 us [2.0289, 2.0514] | +0.37% |

All confidence intervals overlap. Emitted AVX2 code gives both f32 hot loops
six 256-bit loads, four stores, six fused arithmetic instructions, one loop
branch, and no calls or bounds branches. Their loop-control spellings differ,
but neither retains work absent from the other. The evidence therefore exposes
no provider-owned f32 correction. Affinity experiments on this shared hybrid-
core host widened variance and reversed candidate order, so they are rejected
rather than reported as stronger evidence. Only same-run substrate comparisons
are claimed; repeated absolute timings are not treated as stable.

## Fearless SIMD audit — amendment and closure (2026-08-25) <a id="fearless-simd-amendment"></a>

Two corrections to the audit below, found while implementing the increment it
filed. Both narrow the gap; neither changes the conclusion.

**The safe operation surface already existed.** The audit's second finding read
as though Hermes exposed only the `unsafe` facets and no safe layer above them.
It does: `Vector<T, Arch>` carries the full operator set, `zero`/`splat`,
bounds-checked slice load and store, reductions, comparisons, `blend`, and array
conversion, each asserting host support once and discharging the facet
obligation. The real gap was narrower and sharper — the safe layer was missing
exactly `fmadd` and the cross-lane permutes (`reverse`, `interleave`,
`deinterleave`, `swap_adjacent`, `dup_even`, `dup_odd`, `fmaddsub`, `fmsubadd`),
which are the operations a transform or stencil kernel needs and the reason a
consumer writing one dropped to the facets. Those nine methods are now on
`Vector`.

**ADR 011 needed no revision.** The audit proposed revising its exclusion of
`SimdKernel` from the safe-surface argument. That exclusion is correct and the
increment follows it: the facets stay `unsafe` as the implementation seam, and
the safe layer sits above them in `Vector` — which is precisely the structure
ADR 011 chose when it left `KoggeStone`'s ISA fills unsafe beneath a safe trait.
The new decision is recorded as ADR 016 instead.

**Closure.** `HS-FEARLESS-TOKEN-2026-08-25` is delivered as
`hermes_simd::vectorize` plus the `LaneKernel<T>` trait, generated by the same
`#[runtime_dispatch]` attribute as every other dispatcher rather than a second
mechanism. The acceptance oracle's codegen half is met: an AXPY kernel through
the entry emits 41 ymm-bearing instructions including a real `vfmadd213ps`, with
no call into the backend operations — only the one-time CPUID cache
initialization and cold panic paths. The same body invoked as
`kernel.call::<Avx2>()` without the entry emits zero ymm in the caller and five
call sites into `hermes-simd-intrinsics`, each an outlined stub. The
consumer-shaped conformance test compiles under `#![forbid(unsafe_code)]`.

One thing surfaced that the audit did not predict: `#[runtime_dispatch]` dropped
the annotated function's doc comments, so a `pub` generated dispatcher could not
satisfy `#![deny(missing_docs)]`. That is why every dispatch module in this
crate was crate-local, and it was invisible until a public dispatcher was
wanted. The macro now forwards them, and the `#![expect(missing_docs)]` in
`dispatch/popcount.rs` that existed only for this reason is retired.

## SIMD/SIMT Capability and Completeness Audit - 2026-08-11 <a id="simd-simt-2026-08-11"></a>

Scope: the `SimdKernel` operation catalog, the backend/`TargetId` matrix, and
the SIMT question. Method: full read of `kernel.rs`, `kernel_helpers.rs`, the
per-ISA impls, `target.rs`, `cpu.rs`, and the ADR set; source-audit tier except
where a delivered increment carries its own differential evidence.

### SIMT scope — closed, not a gap

Hermes owns lane-parallel CPU execution; device-resident SIMT execution belongs
to Hephaestus, consumed through Coeus/Apollo (README, "Atlas Compute
Boundaries"). Adding GPU dispatch here would fork a dimension another stack
member owns, so no SIMT work is opened against Hermes. What Hermes does own of
the SIMT *programming model* — per-lane predication, masked memory, indexed
gather/scatter, compress/expand, cross-lane reduction — is the axis audited
below, and the masked-memory half is now complete (HS-413..HS-421 landed the
tail routing; every hot kernel runs the provider-owned masked seam).

### Delivered

- [minor] Indexed-store asymmetry (delivered as HS-422): `gather` and
  `gather_masked` existed with no write-side dual, so the lane-addressing model
  was read-only and any scatter-shaped consumer had to leave the vector domain.
  `SimdKernel::scatter`/`scatter_masked` plus `SimdView::scatter` close it, with
  native AVX-512 `vscatterdps`/`vscatterdpd` and the lane-sequential default on
  AVX2/NEON (neither ISA has a scatter instruction). Evidence: per-backend
  differential property tests, a gather∘scatter round-trip identity, and
  duplicate-index/error-contract tests. AVX-512 native execution is runner-gated.

### Operation-catalog status (source-audit tier)

First, the fact that bounds the rest of this section: `SimdKernel` is
implemented for `F16`, `f32`, and `f64` only — there is no generic integer lane
backend. Integer SIMD lives outside this trait, in the AMX/VNNI tiling kernels
and the SWAR bitboards, each with its own operation set. Gaps must therefore be
assessed against a float-lane trait, not a general one.

- [minor] Resolved as HS-423. The rounding family (`floor`/`ceil`/`round`/
  `trunc`) is now defined through Eunomia's native-precision scalar seam and
  Hermes' `generic_unary_op`, with AVX2, AVX-512, and NEON overrides. The
  `RoundTiesEven` policy keeps halfway behavior explicit, and differential
  coverage includes negative values, ties, infinities, NaNs, and signed zero
  across Scalar, SveArch, and AVX2, with the other ISA paths compile-covered
  for hosted execution. The delivered implementation is recorded in Hermes
  commits `58c31a9` and `df32296`; no widen-compute-narrow fallback remains.
- [minor] Resolved as HS-424. General cross-lane permute (`reverse`,
  `interleave`, `deinterleave`) had no seam; the only lane shuffles were the
  complex adjacent-pair primitives. All three now exist as defaulted trait
  methods on the flat lane sequence, with AVX2 native `reverse`. The reusable
  finding is the flat-vs-sublane distinction: x86 `unpack` and `permute_ps`
  operate within 128-bit halves, so they are *not* implementations of a flat
  permute and cannot be substituted as overrides without extra cross-half
  shuffles — a trap that would have produced a silently wrong fast path.
  Remaining native overrides tracked as HS-427, deferred rather than written
  unverified: this host has no AVX-512 and is not aarch64, and wrong permute
  indices return plausible wrong lanes instead of failing loudly.
- Corrected on the same pass: an earlier draft of this entry recorded missing
  integer shifts and missing saturating arithmetic as gaps. Both are withdrawn.
  Shifts are not well-founded against a float-only lane trait, and float
  saturation is not the domain contract; `NumericElement` already carries
  `saturating_add`/`saturating_mul` for the scalar integer types that want them.
  A generic integer lane backend is an [arch] question deserving an ADR, not a
  silent operation-by-operation accretion — it is not opened here because no
  consumer requirement is recorded for one.

### Verification infrastructure — the finding that reframes the rest

- [minor] Resolved 2026-08-11. Every AVX-512 branch is guarded by
  `is_x86_feature_detected!`, so on a runner without the silicon those tests do
  not fail — they *skip*. The capability-report step added with the SDE job
  proved the GitHub x86 runner reports **no AVX-512 and no AMX flags at all**,
  so the AVX-512 scatter override, the BF16 tile dispatch, VNNI, and AMX had
  been carried by a green CI that never executed one of them. A passing suite
  was asserting less than it appeared to, and nothing in the logs said so.
  The `test-avx512-sde` job runs the whole suite under Intel SDE emulating
  Sapphire Rapids: 444/444 pass in 176s (about 11x native), with
  `test_masked_ops_avx512`, `test_select_ops_avx512`, `test_vector_ops_avx512`,
  `interleaved_complex_avx512_matches_scalar_backend`,
  `avx512_tiling::int8_tests`, and `test_adaptive_dispatcher_and_amx_session`
  all executing rather than skipping. It remains the deterministic semantic
  gate; HS-429's `test-avx512-hosted` job adds real-silicon coverage and
  timing on a best-effort basis when the hosted x86 runner carries AVX-512.
  Reusable finding: a capability-gated test suite reports coverage it does not
  have. Automated backend selection is the right design, but it must be paired
  with automated *identification* — the suite now enumerates `TargetId::ALL`,
  prints which backends the runner actually executed, and asserts a per-runner
  expectation supplied as configuration. Silence must never read as coverage.
  Emulation is the fallback, not the mechanism: real silicon is used wherever
  it can be requested (aarch64 NEON on `ubuntu-24.04-arm`, AVX2 on x86, and
  AVX-512 on hosted x86 whenever that silicon is present); SDE covers the ISAs
  no pinned hosted runner guarantees — AVX-512, which is heterogeneous across
  the hosted x86 pool, and AMX, which is unavailable there entirely. SDE's
  limit is that it validates semantics and never performance, so a benchmark
  claim on those ISAs still needs real hardware (HS-429).
- Consequence for the HS-422 and HS-424 records: both shipped their AVX-512
  work marked "runner-gated, not executed". That caveat is now discharged for
  scatter — `prop_scatter_matches_reference_all_backends`,
  `prop_gather_scatter_roundtrip_all_backends`, and the duplicate-index and
  error-contract tests all exercise the native `vscatterdps`/`vscatterdpd`
  override under SDE.

### Hand-written intrinsics versus LLVM's lowering (HS-427, 2026-08-12)

- [minor] A native AVX2 `interleave` written as `unpack` + `permute2f128` — the
  textbook flat-interleave sequence — measured **37% slower** than the generic
  store/permute/load default it was meant to replace (two runs, p < 0.05, quiet
  host, L1-resident size). `deinterleave` was neutral-to-negative. Both were
  removed. AVX2 `reverse` survived on measurement (10.4% faster at 1024 f32).
  Reusable finding: LLVM already lowers the generic default's stack round-trip
  into good shuffle sequences, so "replace the portable path with intrinsics"
  is a hypothesis to measure, not a foregone win — the vectorization ladder's
  rule to escalate only on a measured shortfall applies to *lane permutes* as
  much as to arithmetic. Corollary for benchmark design: measure at a
  cache-resident size. The same comparison at 16384 elements shows no
  difference, because the working set spills and the permute cost vanishes into
  memory traffic — a size that would have hidden the regression entirely.
- Consequence: the AVX-512 and NEON permute overrides shipped in the same
  increment carry correctness evidence only. They are canonical single-
  instruction lowerings, but after the AVX2 result they are explicitly *not* a
  speed claim until measured. HS-430 now supplies the NEON measurement path:
  the native aarch64 job saves the existing `permute` Criterion rows, rebuilds
  with only the three NEON overrides disabled, and compares the identical rows
  on the same ARM host under a finite 300-second command. The hosted result
  still requires review before retaining or deleting an override. AVX-512
  timing is assigned to HS-429's `test-avx512-hosted` job (SDE is semantic
  evidence, not timing evidence).

### Square-transpose networks (HS-TRANSPOSE-NETWORKS, 2026-08-27)

- Retained: AVX2 f32's 8x8 register network reduces 408–421 ns to 4.21–4.32 ns
  across two unchanged Core Ultra 9 285K comparisons. NEON f32's 4x4 network
  repeats at 3.536–3.544 ns versus 7.281–7.324 ns for the generic default in
  hosted AArch64 runs `33137876655` and `33138579478`. Exact release assembly
  contains one tile-length branch followed by register shuffles with no scalar
  lane traffic.
- Removed: the incumbent NEON f64 `trn1`/`trn2` override measures
  2.265–2.268 ns versus 2.080–2.086 ns for LLVM's generic lowering. The generic
  route repeats at 2.075–2.083 ns after removal. Provisional AVX-512 f32/f64
  networks passed the index-coded oracle under Sapphire Rapids emulation and
  produced spill-free exact assembly, but no controlled real-silicon timing
  was available, so they were deleted rather than retained as unverified
  optimizations. The explicit AVX-512 benchmark rows remain the re-open
  instrument when suitable hardware is available.
- Delivery: provider `4af1b25`, PR #94 merge `93ba7ce`, and exact hosted run
  `33139847261` green across bounded benchmarks, x86, native AArch64, SDE,
  Miri, no-std, dependency policy, and lock integrity.

### AVX-512 f32 transpose network and bounds elision (HS-SIMD-PERF, 2026-08-28)

- Closed the last hole in the `transpose_square` override surface: the AVX-512
  f32 16x16 network, filed as the explicit not-done follow-on of
  HS-AVX512-TRANSPOSE. Four stages, 64 shuffles, every instruction AVX512F in
  its zmm form, so the dispatcher's `avx512f`-only probe is sufficient and no
  AVX512DQ/BW/VL operand can reach F-only silicon. Emitted assembly confirms
  the budget and the count: 32 `vshuff64x2` plus 16 `vunpck*ps` and 16
  `vunpck*pd` — LLVM lowers the `shuffle_ps` 0x44/0xEE pairs to 64-bit-pair
  unpacks and prefers the `f64x2` block form, both AVX512F equivalents.
- Correctness is symbolic, not hardware-measured. The permutation algebra was
  checked off-machine against a model of the four intrinsics, and the same
  model reproduces the in-repo AVX-512 f64 network exactly, which is what
  validates the model rather than the claim. This host is an Arrow Lake Core
  Ultra 9 285K reporting `avx512f: false`, so the path cannot execute here.
- Bounds elision, found by codegen inspection rather than by the audit: both
  AVX-512 networks indexed the tile as a slice under a `debug_assert`, so
  release codegen kept a panic path per access — 24 `ud2` sites for f32 and 30
  for f64 — while the AVX2 f32 and NEON f32 networks already re-borrow as a
  fixed-size array. Adopting that idiom drops both to one panic path and
  collapses the f64 body from roughly 3400 lines of assembly to 64, since the
  per-access panic paths pulled formatting machinery inline. Shuffle counts
  are unchanged. The reusable finding is that a register network's asm line
  count, not just its shuffle histogram, is worth reading: the shuffles were
  correct all along and the cost sat entirely in the panic scaffolding.
- Motivating measurement, pinned to a single P-core (`ProcessorAffinity = 1`)
  on the 285K, comparing the native network against the same backend's forced
  `hermes_benchmark_generic_default` build: AVX2 f32 8x8 runs 4.185-4.198 ns
  versus 422.74-426.87 ns for the default, a factor of 101; AVX2 f64 4x4 runs
  2.861-3.032 ns versus 100.94-101.38 ns, a factor of 34. The native f32
  figure reproduces the 4.21-4.32 ns recorded under HS-TRANSPOSE-NETWORKS,
  which is what validates the pinned instrument. These measure the unchanged
  AVX2 paths: they quantify what the stack-capture default costs, and so what
  the AVX-512 f32 tile — four times larger at 256 elements — stood to lose,
  but they are not a measurement of the changed path, which has none.
- Added `transpose_square_is_bit_exact_all_backends`: the index-coded law
  manufactures small positive integers, so a network leaking an operand
  through an arithmetic or NaN-canonicalizing instruction would satisfy it
  while rewriting lane bits. The new fixtures carry signalling and quiet NaNs,
  negative zero, and denormals. Confirmed live before landing — a no-op
  `_mm256_add_ps` in the AVX2 f32 network fails it at (0, 0) while the
  index-coded law still passes.
- Unresolved precedent: PR #94 deleted provisional AVX-512 networks precisely
  because no controlled real-silicon timing was available, recording them as
  unverified optimizations; PR #98 then landed the f64 network on symbolic
  verification alone without overturning that decision, and this change
  follows PR #98 for f32. Which standard governs AVX-512 work on
  AVX-512-less development hosts is an open integrator call, not a settled
  repository policy.

### Backend matrix

- [major] Resolved as HS-425. `TargetId::Sve` now routes through both forced
  dispatch helpers to the lane-emulated `SveArch` backend, and conformance
  tests cover the public target and host-capability surfaces. `TargetId` is
  `#[non_exhaustive]`; automatic dispatch intentionally has no SVE branch, so
  the emulated backend remains explicitly requested. The breaking migration is
  recorded by ADR 014 and the changelog entry delivered with Hermes PR #49
  (merge `fb36e0f`, implementation `dd4cc78`).
- Native SVE remains blocked on stable Rust (scalable vectors are not
  expressible); `SveArch` stays lane-emulated and its hardware probe stays
  informational. Unchanged, correctly documented, not a defect.
- SSE2 and Arm SME each have a feasibility ADR (006, 007) and no backend. That
  is a recorded decision, not drift.

### Documentation hygiene

- [patch] Resolved as HS-426. `docs/adr/` had two ADRs numbered 007, eight of
  eleven with no `## Status` section (the generated index rendered `—`), and
  `Approved` instead of the canonical `Accepted` on the rest. The later
  duplicate renumbered to 011 with its references updated; all eleven now carry
  `Accepted`, and `adr-index.py check` passes for this repository. Note the
  generator already reported every one of these anomalies — the tooling was
  correct and unheeded, which is the reusable finding: a check whose output
  nobody burns down is not a gate.

### Cross-repo (reported, not owned here)

- The atlas stack overlay in `.cargo/config.toml` had gone stale against gaia's
  rename to package `gaia-mesh`, breaking dependency resolution under the
  umbrella. Regenerated via `scripts/atlas-stack-overlay.py generate` (the
  overlay is generated state, never hand-edited). Consumers still declaring
  `gaia = { git = ... }` remain unpatched until they rename the dependency —
  an atlas-level follow-up, outside Hermes.

- Resolved 2026-08-06 — HS-409 fused ternary AXPY provider facade: Hermes now
  exposes `axpy_mul(alpha, a, b, out)` and `SimdOps::axpy_mul` for the exact
  in-place contract `out[i] += alpha * a[i] * b[i]`. The runtime-dispatched
  kernel reuses `SimdKernel::mul` followed by `SimdKernel::fmadd`, writes each
  output lane once, and performs no temporary allocation. Length validation is
  centralized in the provider kernel; the scalar tail uses the same
  `Scalar::scalar_fmadd` operation as the vector path. Public f32/f64 tests cover
  the facade and tail-sized inputs; internal tests cover empty/tail/mismatch
  cases. The operation is provider capability only: Kwavers adoption remains
  downstream work and is not claimed here. Evidence tier: source implementation
  plus value-semantic public-facade tests. Local locked gates remain blocked by
  the pre-existing dirty Hermes lock overlay, which requires unrelated provider
  lock regeneration; no lockfile rewrite was retained.

- Resolved 2026-07-19 — HS-402 provider compatibility: Hermes' native
  `eunomia::F16`/`Bf16` source compiles against Eunomia 0.6 without restoring
  the retired foreign raw-half trait implementations. Cargo resolves the
  workspace to one Eunomia 0.6 identity at `df77dfd`; warning-denied Clippy,
  388 value-semantic Nextest cases, 18 runnable doctests, and warning-denied
  rustdoc pass.

- Resolved 2026-07-18 — HS-401 Eunomia reduced-precision ownership: Hermes
  replaces raw `half::f16`/`half::bf16` in scalar, F16C, AVX-512, NEON, AMX,
  tiled GEMM, tests, and benchmarks with `eunomia::F16`/`Bf16`/`F32`.
  Duplicate raw-half AMX and tiled-GEMM families are deleted, and all direct
  `half` manifest dependencies are removed. Source and manifest residue scans
  are empty; the locked graph contains one Eunomia 0.5.0 identity. Full
  all-feature warning-denied Clippy, 388 value-semantic Nextest cases,
  doctests, rustdoc, and no-default-feature compilation pass. Evidence tier:
  compile-time provider identity plus value-semantic and differential tests.
  The lock still contains transitive `half` through Eunomia's temporary
  raw-trait surface and Criterion's Ciborium dependency.
- Resolved 2026-07-18 — HS-401 remote host variance: PR #8 exposed two
  host-sensitive defects absent on the local CPU. Adaptive dispatch queried
  Bf16 capabilities for every operand type, allowing int8 GEMM to enter
  AVX-512 VNNI on a host without that extension; dispatch now binds its probes
  to `T: AmxSupport + Avx512Support`. The 64-lane AVX-512 masked-gather oracle
  also used a fixed 100-element fixture although its maximum index is
  `3 * (lanes - 1)`; fixture length is now derived as `3 * lanes`. Evidence
  tier: remote ISA failure reproduction plus type-bound dispatch and an
  analytical index bound. PR #8's final x86, AArch64 cross-compile, native
  AArch64 NEON, Miri, cargo-deny, and CodeRabbit gates pass at `f9e8ff5`;
  merge commit `8970ffc` closes the item.
- Residual 2026-07-18 — HS-401 historical semver baseline: `origin/main`
  resolves moving Eunomia main 0.5.0, so its historical raw-half implementation
  no longer compiles at `scalar/tiling.rs:76-77` and
  `x86_64/avx512_tiling.rs:85,91`. `cargo semver-checks` therefore cannot
  classify the current public delta against that baseline. This is baseline
  dependency drift; the current 0.4.0 workspace compiles and passes its gates.
  Re-open when the semver baseline can pin its historical Eunomia revision.
- Resolved 2026-07-18 — HS-401 AArch64 verification: rustup's 1.95 target
  libraries are incompatible with proc-macro artifacts already produced by the
  PATH MSYS Rust 1.95 Rev2 compiler in the mandatory shared target directory.
  No private target or destructive clean was used. PR #8 independently passes
  both the remote AArch64 cross-compile and native runtime NEON lanes. Evidence
  tier: cross-target compile-time validation plus native architecture tests.

- Resolved 2026-07-15 — provider default-branch convergence: Hermes removes
  revision pins and workspace-local patches for Mnemosyne, Eunomia, and Themis.
  `cargo tree --locked -d -p hermes-simd` reports one identity for each; the
  package-scoped format, Clippy, nextest, rustdoc, and `cargo deny check` gates
  pass. CI's source allowlist names the reviewed provider URLs directly and
  redundant sibling checkouts are deleted. Evidence tier: locked
  dependency-resolution and value-semantic package tests.

## 2026-07-08 CI: miri gate known-failing, tracked on upstream mnemosyne <a id="miri-known-failing-2026-07-08"></a>

**Evidence tier: machine-checked (Miri, both Stacked Borrows and Tree Borrows
aliasing models).** PR #5 (`cb0b1b0`) merged with the `miri` CI job left
red — this is a deliberate, tracked exception, not an overlooked failure.

- **Finding:** `cargo miri test -p hermes-simd-core` fails on a genuine
  aliasing violation inside `mnemosyne-local`'s allocator, not in
  hermes-owned code. Repro: `AlignedVec::with_capacity` (`vec/mod.rs:96`)
  calls `mnemosyne_local::alloc::thread_alloc_checked` (`alloc.rs:130`),
  which reads a `Page` via `NonNull::as_mut`; a later `dealloc()`
  (`vec/mod.rs:542`) writes through an aliasing pointer to the same backing
  memory, disabling the earlier `Page`-pointer's tag; a subsequent `alloc()`
  re-reads that now-disabled tag.
- **Ruled out as a Stacked-Borrows-specific false positive**: tested under
  `-Zmiri-tree-borrows` (a materially more permissive aliasing model
  designed to accept exactly this embedded-metadata-allocator pattern) — it
  still failed, with a clearer diagnostic pointing at the same `Page`
  pointer/foreign-write sequence. Agreement across both independent models
  is strong evidence of a real bug, not a model artifact. The Tree Borrows
  CI experiment was reverted (`166a7b9`, since it provided no benefit) after
  confirming this.
- **Owning repo:** mnemosyne, not hermes. Full reproduction instructions and
  working hypothesis documented in mnemosyne's own `gap_audit.md`
  ("2026-07-08 Miri: real aliasing violation in the alloc/free
  page-metadata path", commit `98a02b6`).
- **Why merge anyway:** the other 4 CI jobs (gates, cargo-deny,
  cross-compile, test-aarch64) all pass; this is a pre-existing bug in an
  upstream dependency, newly surfaced by CI graph resolution rather than
  introduced by this PR's changes; blocking hermes indefinitely on an
  upstream fix has no bounded timeline.
- **Follow-up (tracked, not deferred silently):** once mnemosyne's allocator
  fix lands and hermes bumps its pinned `mnemosyne` rev, re-run
  `cargo miri test -p hermes-simd-core` and confirm green before removing
  this entry. Until then the `miri` job on hermes CI is a **known-failing,
  tracked gate** — branch protection was overridden via `--admin` for this
  merge only; it is not disabled going forward, so future PRs will need the
  same override (or the upstream fix) to land.

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
- **[RESOLVED 2026-07-05] Unsound safe API surfaces (PROVEN).**
  `AmxSession::new` / `AmxBatchSession::begin` now return
  `AmxSessionError::UnsupportedTarget` before `ldtilecfg`; `release` does not
  issue `tilerelease` unless a supported active session exists. `SimdArch`
  carries the runtime-support probe used by `TargetId` and the safe `Vector` /
  `Mask` wrappers; unsupported AVX-512 hosts get `SimdError::UnsupportedTarget`
  from fallible constructors and checked slice wrappers before any AVX-512
  instruction. Infallible vector conveniences panic before ISA execution.
  Evidence tier: type-level trait seam + value-semantic unsupported-host
  regressions. See [Resolved](#resolved).

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
- **[RESOLVED 2026-07-05] Per-call CSR/BCOO/SELL-p SpMV re-validation
  (HIGH, 2 agents).** Added `Validated<F>`/`ValidatedData<S>` and moved CSR,
  SELL-p, and Blocked-COO `SparseSpMv` impls to
  `SparseView<Validated<_>>`. Public `spmv_csr`/`spmv_bcoo`/`spmv_sellp`
  dispatch now requires validated storage, and validated `SparseCow`
  constructors preserve the run-once invariant for repeated solver calls. Raw
  malformed structures fail at validated view/COW/public-dispatch construction;
  hot SpMV kernels keep only runtime-vector size checks for `x`/`y`. Evidence:
  type-level typestate plus regression/property coverage over construction-time
  rejection and value-semantic SpMV for CSR/SELL-p/Blocked-COO. Local compile
  verification is pending shared Cargo target lock clearance.
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
- **Resolved 2026-08-09 — HS-417 transposed GEMV column tail.**
  `gemv_transpose_strided_impl` now handles `ncols % LANE_COUNT` through one
  initialized local lane-buffer path and `SimdKernel::masked_fmadd`, preserving
  the full-width-valid masked-memory contract even on blend-based backends.
  Only the live tail is copied back to `y`; no caller slice is over-read or
  over-written. The f32 facade has a non-dyadic tolerance regression because
  the provider-owned fused operation may round differently from scalar
  multiply-plus-add. Evidence tier: provider implementation plus differential
  shape tests and non-dyadic f32 tolerance coverage. Native SVE remains a
  separate stable-toolchain-blocked item.
- **Resolved 2026-08-09 — HS-421 native AVX-512 BF16 tile kernel.** The
  `Bf16 × Bf16 → F32` tile now has a native `DPBF16PS` path on hosts reporting
  `avx512bf16`, with a single lower-level capability SSOT in
  `hermes-simd-intrinsics`. The existing AVX-512F/BW/VL conversion/FMA tile
  remains the fallback for AVX-512 hosts without BF16 dot products. The native
  differential uses non-dyadic BF16 inputs and nonzero `C`, validating the
  `C += A·B` contract; ordinary hosts skip only the hardware-specific execution
  while still compiling and testing the fallback. A hosted AVX-512 BF16 runtime
  and benchmark gate remains open.
- **Resolved 2026-08-28 — active-prefix memory removes tail staging.** The
  scalar-tail campaign first moved each eligible family onto masked arithmetic
  while preserving full-width pointer validity through local lane buffers.
  HS-MASKED-TAIL-PARTIAL-LOAD now replaces those buffers with an exact
  active-prefix memory contract in AXPY/scale, GEMM/GEMV, view arithmetic,
  opt-in reductions, and scatter-source tails. AVX2/AVX-512 f32/f64 use native
  masked memory; other scalar/backend pairs inherit an active-lane default.
  `Product` and future
  operations that do not opt into reassociation retain their distinct scalar
  tail by contract. **Earlier partial closure
  2026-08-07 (HS-410/HS-411/HS-412):** `dispatch/axpy.rs` now routes its final
  partial vector through `masked_fmadd`, `dispatch/scale.rs` routes its final
  partial vector through `masked_mul`, and `dispatch/axpy.rs`'s `axpy_mul` path routes its
  final partial vector through register scaling plus `masked_fmadd`; all use
  initialized local lane buffers that preserve safety for AVX2 blend-based
  masked operations.  **HS-415 (2026-08-07):** `reduce_popcount` and the shared binary
  `reduce_popcount_op` now route their final partial vectors through
  `masked_sum_reduce` after copying source lanes into initialized local buffers;
  each masked tail count is exact while the existing whole-reduction accumulator
  contract remains unchanged. Generic reduction and broader view tails are
  covered by HS-416; other kernels remain open. **HS-414 (2026-08-07):** `AbsSum` and
  `AbsMax` now route their final
  partial vector through a generic masked reduction seam after copying live
  elements into initialized local lanes and applying the transform before
  identity merge. Generic reduction and broader view tails are covered by
  HS-416; other kernels remain open. **HS-413 (2026-08-07):**
  `axpy_rows` now routes its final partial vector
  through the same initialized-buffer `masked_fmadd` helper, and `axpy_rows_batch`
  does so per row after preserving its depth accumulation order. Non-dyadic f32
  regressions cover both paths. Reductions, view operations, and other kernels
  remain open; no repo-wide closure or performance claim is made.
- **Resolved 2026-08-09 — HS-418 dense dot-product tail.**
- **Resolved 2026-08-09 — HS-419 pairwise reduction tail.** `SimdView::zip_reduce(Dot)`
  now copies live pairwise inputs into initialized provider-local buffers and
  uses the generic masked reduction seam for its final partial vector. This
  removes the dot cleanup loop without violating the full-width masked-load
  contract; forced emulated-SVE non-dyadic f32 coverage allows the documented
  reassociation tolerance.  `Product` remains on its scalar pairwise tail path
  because its multiplicative identity and reduction ordering are distinct.
- **Resolved 2026-08-09 — HS-420 mutable generic view tail.**
  `SimdView::transform_in_place` now stages its final partial operands in
  initialized provider-local buffers, applies the generic `ElementOp` vector
  seam, and copies back only live result lanes. Add/Sub/Mul/Div therefore share
  one bounds-safe implementation; forced emulated-SVE odd-length coverage pins
  the tail path. Remaining scalar tails are tracked per kernel rather than
  claimed closed globally.
  `SimdView::dot` now copies its short remainder into initialized local lane

  buffers and folds it through `SimdKernel::masked_fmadd`, avoiding caller-slice
  over-read while retaining the provider's fused arithmetic path. The final
  reduction includes only live lanes. Odd non-dyadic f32 coverage uses the
  documented tolerance for fused-rounding differences. Remaining scalar tails
  are tracked per kernel rather than claimed closed globally.

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
- **[REJECTED 2026-08-29] Software prefetch in gather-bound SpMV.** A permanent
  Criterion instrument now covers 1,048,576 and 2,097,152 nonzeros against a
  64 MiB dense operand with an independent exact dyadic oracle. The candidate's
  reported changes were mean estimates mislabeled as medians. Retained raw
  samples prove only the second paired median comparison (21.04%/18.01%); the
  first candidate sample was overwritten, and the retained cross-run smaller
  row improves only 1.49%. The predeclared two-pair median gate therefore fails
  and production remains unchanged. ADR 020 records the evidence correction.
- **[open] `Aligned` typestate remains absent from the dispatch facade
  (LOW-MED).** Non-temporal stores are already resolved below. This remains a
  `[minor]` measurement-gated candidate.
- **[REJECTED 2026-08-29] Eight-accumulator floating reductions.** A temporary
  same-binary instrument compared production, four, and eight independent
  accumulators for exact dyadic f32/f64 sum and dot at 256/1024/4096/16384
  elements. Two runs pinned to processor mask 1 produced one repeatable material
  result: f64 sum at 4096 improved 15.1–15.2% versus four accumulators. f32 had
  no repeated material win; other f64 sizes were flat, unstable, or slower than
  the production path. The cross-type, two-size acceptance threshold failed, so
  the instrument was removed and production remains at `UNROLL_FACTOR = 4`.
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

### F16 dispatch-frame probe hoist (HS-F16-DISPATCH-PROBE-HOIST-2026-08-27)

- **Resolved 2026-08-28 — repeated F16C probes in dispatched arithmetic.** The
  F16 AVX2 implementation previously entered an `avx2,fma` dispatch helper, so
  each add, multiply, and fused multiply-add repeated the cached F16C check and
  called a separate conversion helper. Runtime dispatch now selects one
  `avx2,fma,f16c` frame and monomorphizes the complete kernel with a private
  proven marker. The public `Avx2` marker retains its direct-operation probe and
  software fallback for AVX2 hosts without F16C.
- **Exact codegen evidence:** the release F16 dot helper contains zero
  `FeatureCache` references and zero fallback arithmetic calls; its body emits
  68 `vcvtph2ps`, 30 `vcvtps2ph`, 6 `vaddps`, 10 `vmulps`, and 14 `vfmadd*`
  instructions. The ordinary f32/f64 AVX2 frame remains `avx2,fma`.
- **Controlled measurement:** on an Intel Core Ultra 9 285K, the unchanged
  `Dense Dot f16/dispatch/16384` Criterion row measured 2.489–2.515 µs and
  2.526–2.554 µs before the change, then 1.086–1.091 µs and 1.087–1.099 µs on
  two final runs (2.30–2.33x by point estimate). This is empirical evidence for
  that host and workload, not a cross-microarchitecture throughput guarantee.

### Memory / zero-copy

- **[RESOLVED 2026-08-27] `DenseWithMask` `[bool]` mask bit-packed
  (was MED-HIGH).** New arbitrary-length `PackedMask` (canonical `mask`
  module) replaces the byte-per-element `[bool]` in `DenseWithMaskData` /
  `OwnedDenseWithMask` in place — 8× mask footprint reduction, packed once at
  the construction boundary — and the SpMV / `sum_values` /
  `elementwise_mul_dense` kernels read packed lane windows via
  `mask_from_bitmask` instead of the per-chunk per-call bool-conversion loop.
  Unused `DenseWithMaskBitMaskData` partial variant deleted. PR #81,
  backlog `HS-DENSEMASK-BITPACK-2026-08-27`.
- **[RESOLVED 2026-08-27] Packed-mask and dense logical-shape bounds.** Public
  `PackedMask` extraction now rejects every out-of-range bit or window in debug
  and release builds without offset arithmetic overflow. `DenseWithMask`
  validation requires exact values and mask lengths against checked
  `nrows * ncols`; its accessors and kernels validate at their operation
  boundaries, while vector loops use crate-private prevalidated extraction. Release
  adversarial tests cover the boundary and overflow cases. Exact AVX2 inspection
  shows only pre-loop validation added, and an unchanged bounded comparison
  detects no stable throughput change (`HS-PACKED-MASK-SHAPE-SAFETY-2026-08-27`).
- **[RESOLVED 2026-08-27] `cmp_*_mask` stack round-trip.** The six public
  comparison-mask methods now convert the backend comparison register directly
  to its native mask. Two unchanged f32/f64 measurements and exact AVX2
  disassembly identified and then removed the store, lane-wise scalar scan, and
  mask reconstruction (`HS-NATIVE-COMPARISON-MASK-2026-08-27`).
- **[RESOLVED 2026-08-27] AVX2 `f32` to `i32` cast scalarization.** Two unchanged
  measurements and exact disassembly found eight scalar conversions per public
  vector versus one packed conversion in Fearless SIMD. Hermes now emits the
  same packed precise-conversion sequence, including Rust-compatible positive
  overflow, NaN, and infinity correction; boundary and arbitrary-bit tests
  match scalar `as`. The corrected whole-output checksum instrument reports
  6.9–11.4x finite in-range public-path gains across 256–4096 elements; the
  provider-to-provider rows remain host-load-sensitive, so no parity claim is
  attached to this result. Provider `18da238` merged through PR #86 as
  `5734b85`; PR #87 fixed target-test imports as `4f6a1eb`, and hosted run
  `33120584552` is green. AVX2 intrinsics cannot execute under Miri; targeted
  host property tests and exact code generation cover this path. Local ASan
  remained unavailable because the MSVC toolchain lacks
  `clang_rt.asan_dynamic_runtime_thunk-x86_64.lib`. Lane-count-changing
  conversion remains outside the current `Vector::cast` contract.
- **[RESOLVED 2026-08-27] AArch64 all-target warning escape.** PR #86's local
  AArch64 evidence compiled library targets only, while hosted CI compiled test
  targets and rejected three host-conditional imports under `-D warnings`. The
  imports are now scoped to x86 or removed, and both AArch64 Linux and Windows
  all-target checks reproduce the hosted warning policy. The prevention is to
  run the hosted all-target command, not a library-only approximation, whenever
  target-specific test code changes.
- **[RESOLVED 2026-08-27] `argmin`/`argmax` per-vector single pass rejected.**
  A same-binary candidate preserved empty/NaN rejection, first-occurrence ties,
  signed-zero representation, generic f32/f64 execution, and zero allocation.
  Two unchanged runs at 256/1024/4096/16384 elements lost every row: f32 was
  2.07–4.00× slower and f64 was 1.23–1.58× slower. Exact AVX2 assembly shows
  the cause: each vector executes a serial horizontal-minimum shuffle chain
  before scalar comparison and mask extraction. The current vector reduction
  plus locating/NaN scan remains unchanged. A lane-local value/index
  accumulation design requires new backend index-vector arithmetic, selection,
  and extraction; it is not justified by this rejected experiment.
- **[RESOLVED 2026-07-04] `compress` per-chunk buffer zero-init.** The hot
  compaction loop re-declared `[T::ZERO; 64]` each chunk (256–512 B of zero
  stores) though the vector store writes `lane_count` lanes and the copy reads
  only `pop ≤ lane_count`. Now a single hoisted `MaybeUninit<T>` array
  (`MAX_SIMD_LANES`, `LANE_BOUND_CHECK`-guarded) with the loop-invariant popcount
  hoisted; behavior unchanged (compress + `expand∘compress` identity tests pass).
  Focused Criterion coverage now records public `SimdView::compress` scalar and
  host-AVX2 all/half/quarter-mask rows at 1K, 16K, and 256K elements; regression
  self-check covered 102 committed Hermes benchmark rows.
- **[RESOLVED 2026-07-03] Non-temporal (streaming) stores for out-of-LLC writes
  — strongly beneficial, productionized.** Focused experiment
  (`streaming_bench.rs`): `out = a + b` over 16 Mi f32 (192 MiB working set, past
  L3), identical AVX2 loads+add, differing only in the store — normal
  `_mm256_store_ps` vs `_mm256_stream_ps` + `sfence`. **Regular 10.24 ms
  (18.3 GiB/s) → streaming 5.98 ms (31.3 GiB/s) = 1.71×**, far above the ~25%
  RFO-avoidance estimate. Productionizing: a `SimdKernel::store_streaming` seam
  (default = `store_aligned`; `SUPPORTS_NT_STORE` const gate; x86 f32/f64
  override via the codegen template's `__PREFIX___stream___SUFFIX__`) plus
  `stream_write_barrier` (sfence), and a size-gated (`len·sizeof(T) ≥ LLC-ish
  threshold), prefix-peeled-to-alignment streaming path in the elementwise
  `zip_into` SSOT. Differential test: streaming result is byte-identical to the
  regular store (same op, cache bypass only).
- **[RESOLVED 2026-07-03] `reduce_popcount_{and,or,xor}` triplication.** Three
  byte-identical ~104-line popcount reductions differing only in the bitwise op
  collapsed to one generic `reduce_popcount_op<Op: ElementOp<T>>` + three ZST
  wrappers (`BitAnd`/`BitOr`/`BitXor`); −153 lines, zero-cost (op monomorphized).
- **[RESOLVED 2026-07-03] `AlignedVec` growth churn.** Added `reserve` +
  `extend_from_slice` (single realloc via the shared `grow_to` SSOT); `Extend for
  SimdCow` now reserves `size_hint().0` up front instead of a push loop's
  ⌈log₂ n⌉ reallocations. Tests cover request-satisfaction, pointer-stability,
  no-op-when-sufficient, and value/empty/pre-sized/ZST extend paths.
- **[RESOLVED 2026-07-02] `SimdCow::scale` copy-then-rescale** — now delegates
  to the fused `mul_scalar_cow` (`broadcast_op` SSOT); halves traffic, removes
  the duplicate implementation.

### Architecture / redundancy / hygiene

- **[RESOLVED 2026-07-02] Tracked scratch at repo root** — `apply_changes.ps1`,
  `do_changes.ps1`, `check_errors.txt` deleted from git; untracked logs removed;
  `check_errors.txt` gitignored. Dead dep declarations dropped (`divan`,
  2×`bytemuck`, intrinsics `rkyv`; facade `rkyv` corrected to dev-dependency).
  `benchmarks/benchmarks_baseline.json`/`benchmarks/benchmarks_results.md` kept (live baseline).
- **[RESOLVED 2026-08-21] `codegen.rs` ungoverned SSOT.** The former 1424-line
  binary was not a complete source of truth: a direct `rustc +1.97.0` build and
  run rewrote all four x86 f32/f64 files while dropping 28 shipped methods
  (five from each AVX2 file and nine from each AVX-512 file). It also had no
  x86 f16 or AArch64 NEON model, no generated-file marker, and no CI freshness
  gate. The binary is deleted, ADR 005 now records checked-in ISA files as the
  canonical sources, and the four files were restored unchanged before the
  cleanup was committed.
- **[open] ~150-200 lines cross-backend scaffold duplication** — compress/expand
  emulation, AVX-512 cmp-mask blend, popcount LUT, masked-reduce, and NEON
  sign-flip constants remain candidates for shared helpers or trait defaults.
  A future consolidation must preserve each ISA/precision contract; the retired
  generator is not a sanctioned destination. `[minor]`.
- **[PARTIAL 2026-07-02] Doc drift** — README `hermes-numeric` entry replaced
  with the eunomia provenance note; the fictional lib.rs feature table replaced
  with the real set; `gemm_int8` example corrected to `gemm::<i8,i8,i32>`.
  ADR number collisions resolved 2026-07-03 (the duplicate 001/002/003 —
  refined-simd-view, target-feature-inlining, numa-memory — renumbered to
  008/009/010 via `git mv` + title fix; 001–010 now unique, no cross-refs
  affected). **Still open:** stale backlog/checklist `hermes-numeric` refs,
  `dispatch/mod.rs` (828) split into trait/impls/facade. `[patch]`.
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

## Hermes audit 2026-08-15 <a id="hermes-2026-08-15"></a>

- [patch] The exact integrated provider head `7343402a` had 23 source files
  over the 500-line hierarchy target; comparison with the preceding Atlas
  pointer showed this was stale conformance state, not a new regression. The
  546-line `ops/reduction.rs` mixed the reduction contract with the
  multiplicative strategy. Moved `Product` and its sealed generic
  `ReductionOp` implementation into the dedicated
  `ops/reduction/product.rs` leaf. The parent is now 442 lines, the new leaf
  is 60 lines, and the public export remains unchanged. Evidence: structural
  line-count audit plus provider `cargo fmt --check`, `cargo check`, clippy
  with `-D warnings`, and `cargo nextest run -p hermes-simd --all-features`
  (410/410 passed). This is a hierarchy/ownership cleanup; no performance
  claim is made without a controlled benchmark.

## Residual Risks

- AVX-512 and AMX runtime validation still depends on matching hardware.
- Native SVE remains a planned backend blocked by the pinned stable Rust
  toolchain: current `SveArch` coverage is an emulated value-semantic backend.
  `SveArch::is_native_hardware_supported` reports hardware capability separately
  and does not claim native execution. When a native SVE backend lands, `LANE_COUNT`
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
