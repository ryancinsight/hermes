# Checklist — active sprint

## HS-ARGEXTREMA-ONE-PASS-2026-08-27 [patch]

- [ ] Reconcile the public empty, NaN, first-tie, signed-zero, and generic
      scalar contracts with the current two-pass implementation and benchmark.
- [ ] Add a same-binary, value-gated one-pass candidate to the existing dense
      instrument and inspect exact AVX2 code generation before reading timing.
- [ ] Run two unchanged bounded measurements across cache-resident and
      bandwidth-bound sizes; retain only a repeatable net improvement.
- [ ] If retained, implement one canonical generic path and extend positive,
      boundary, adversarial, and backend-generic value coverage.
- [ ] Run exact-diff and full affected gates, synchronize evidence, obtain an
      independent judge verdict, publish, merge, and continue.

## HS-EXACT-LANE-DISPATCH-2026-08-27 [minor] [arch]

- [x] Reconcile Apollo's four-lane contract with ADRs 016/017, current backend
      lane counts, forced-target helpers, and the generated widest dispatcher.
- [x] Record ADR 018 and implement one exact-lane entry without changing
      widest-native dispatch or adding fixed-vector aliases.
- [x] Add host-independent value and no-call tests plus x86/AArch64 selection
      coverage under `#![forbid(unsafe_code)]`.
- [x] Inspect optimized codegen, run focused/full/cross-target/SemVer gates,
      and synchronize audit and changelog evidence.
- [x] Obtain an independent judge verdict covering exact-width filtering,
      absence semantics, widest-dispatch preservation, and target-feature
      safety.
- [ ] Commit, publish, and collect hosted verification before consumer
      migration.

## HS-NATIVE-CAST-THROUGHPUT-2026-08-27 [minor]

- [x] Reconcile the public equal-lane cast contract, Eunomia `CastFrom`
      semantics, current stack/scalar mechanism, and native Fearless SIMD 0.7
      conversion surface.
- [x] Add an input-sensitive f32/i32 same-binary comparison for public Hermes,
      a backend-native route, and precise Fearless SIMD at equal native widths.
- [x] Run two unchanged bounded measurements and inspect exact AVX2 code
      generation; implement only if the deficit and stack mechanism repeat.
- [x] Preserve finite, boundary, NaN, and infinity semantics across affected
      backends without changing the public API or allocating.
- [x] Run exact-diff and affected full gates, synchronize evidence, obtain an
      independent judge verdict, commit, publish, collect hosted verification,
      merge, and continue.

## HS-CI-RUNNER-CLASS-SELECTION-2026-08-27 [patch]

- [x] Reproduce the escaped generic-default configuration failure independently
      of AVX-512 runner capability.
- [x] Compile the generic-default SIMD configuration on every x86 verification
      runner without executing unsupported instructions.
- [x] Prove the gate fails when the escaped import defect is reintroduced and
      passes on the corrected tree.
- [x] Validate workflow syntax, run the exact local gate, publish, merge, and
      collect hosted verification.

## HS-PACKED-MASK-SHAPE-SAFETY-2026-08-27 [patch]

- [x] Reconcile the merged packed-mask representation, public extraction
      contracts, raw DenseWithMask operation boundaries, logical shape, and
      current sparse value-semantic coverage.
- [x] Record the unchanged sparse benchmark baseline and exact AVX2 loop before
      changing production code.
- [x] Enforce release-safe public extraction and exact overflow-safe matrix
      validation, with one pre-loop validation boundary and private
      prevalidated loop extraction.
- [x] Add debug/release boundary, adversarial shape, and dense-reference tests;
      re-run the unchanged benchmark and inspect exact code generation.
- [ ] Run exact-diff and affected full gates, synchronize evidence, obtain an
      independent judge verdict, commit, publish, collect hosted verification,
      merge, and continue.

## HS-NATIVE-COMPARISON-MASK-2026-08-27 [patch]

- [x] Reconcile the recorded stack round-trip, current `Vector`/`Mask` public
      surface, native backend conversion seam, existing comparison conformance,
      and Fearless SIMD 0.7 mask route.
- [x] Add one generic input-sensitive f32/f64 equality-mask comparison for the
      current public route, direct backend route, and Fearless SIMD without
      changing existing benchmark groups.
- [x] Run two unchanged bounded measurements and inspect exact code generation;
      implement only if the current/direct deficit is repeatable and the stack
      mechanism is visible.
- [x] Route all six public comparison-mask methods natively and extend generic
      backend coverage, including NaN behavior, without changing the API.
- [x] Run exact-diff and affected full gates, synchronize evidence, commit,
      publish, collect hosted verification, and merge PR #84 as `6efa67b`.

## HS-DISPATCH-CACHE-THROUGHPUT-2026-08-27 [patch]

- [x] Reconcile current main, PR #80's disjoint AVX2 interleave change, and the
      current Fearless SIMD, Archmage, simd-abstraction, Simdeez, and
      simply_simd dispatch/cache designs from authoritative sources.
- [x] Add one input-sensitive f32/f64 dispatch-only group without changing any
      established workload, timed region, or dependency.
- [x] Assert equal selected lane counts, then run two unchanged bounded
      Criterion measurements and inspect exact emitted dispatch code.
- [x] Reject a production correction: exact assembly exposes three Hermes
      feature-cache loads, but neither precision has a repeatable disjoint
      deficit across both runs, so an added atomic or indirect call fails the
      acceptance oracle.
- [x] Run exact-diff and affected full gates, synchronize evidence, commit,
      publish, collect hosted verification, and merge PR #82 as `99910ad`.

## HS-PULP-LANE-THROUGHPUT-2026-08-27 [patch]

- [x] Verify the current Pulp, Fearless SIMD, `wide`, Macerator, and
      `std::simd` capability/dispatch constraints from their authoritative
      documentation and select only same-binary comparators.
- [x] Evaluate one generic Pulp row in the unchanged planar f32/f64 instrument,
      assert native-width and scalar-oracle parity before timing, and remove the
      row when dependency policy rejects its transitive closure.
- [x] Run exact AVX2 code generation plus two bounded unchanged measurements;
      implement only a stable provider-owned correction.
- [x] Synchronize the gap audit, README, ADR 017, changelog, and item state;
      exact-head local and hosted gates passed; PR #79 merged as `3c548015`.

## HS-CAPABILITY-LOAD-THROUGHPUT-2026-08-27 [patch]

- [x] Reconcile the accepted capability model, the existing checked/view/direct
      diagnostic, Fearless SIMD 0.7's capability-bound slice load, and current
      first-party consumer call sites.
- [x] Add the minimal capability-scoped checked load and backend-generic value,
      short-input, and unsupported-construction coverage.
- [x] Extend the existing diagnostic without changing its inputs, timed region,
      or reference oracle; inspect exact AVX2 codegen before reading timing.
- [x] Run two bounded same-binary measurements. Reject the API unchanged: it
      removes support probes but retains five bounds/panic branches and misses
      the direct/view ceiling for the short-loop regime.
- [x] Synchronize ADR 017, the README, and gap audit; confirm the changelog is
      unaffected because no production or benchmark change survives.
- [x] Run exact-diff formatting and whitespace gates and publish draft PR #77.
- [x] Collect hosted verification and merge PR #77 as `c3d1b67`.

## HS-FEARLESS-COMPLEX-REG-THROUGHPUT-2026-08-27 [patch]

- [x] Reconcile current main, open PRs, hosted gates, governing ADRs, and the
      Fearless 0.7 public operation surface.
- [x] Generalize the interleaved butterfly instrument over f32/f64 without
      cloning the scalar, provider, or benchmark paths.
- [x] Compare `ComplexReg`, the raw Hermes vector recipe, and Fearless's
      deinterleave/planar/reinterleave route at equal native width, workload,
      and addresses; assert scalar semantics before timing.
- [x] Run two bounded Criterion measurements and inspect exact AVX2 codegen;
      model a non-obvious loop when instruction counts do not settle it.
- [x] Implement only a stable provider-owned correction, or record why the
      measurement rejects one.
- [x] Run exact-diff gates, synchronize documentation, commit, publish, collect
      the full hosted matrix, and merge PR #76 as `ba32b8c`.

## HS-FEARLESS-PERMUTE-THROUGHPUT-2026-08-26 [patch]

- [x] Re-audit Fearless-only capability families against current Hermes
      consumers and retain non-gaps without live contracts.
- [x] Add one generic f32/f64 same-address comparison for interleave and
      deinterleave without cloning the scalar or provider paths; retain reverse
      in the Hermes-native suite because Fearless has no matching operation.
- [x] Assert lane-order semantics against an analytical oracle before timing.
- [x] Run the bounded Criterion comparison and inspect codegen for every
      non-overlapping Hermes deficit.
- [x] Implement and verify any provider-owned correction justified by the
      measurement; otherwise record the evidence limit.
- [x] Run the exact-diff gates, synchronize evidence, commit, publish, and merge.

## HS-FEARLESS-F32-THROUGHPUT-2026-08-26 [patch]

- [x] Confirm the official `fearless_simd` release remains 0.7.0 and its main
      revision remains `3ac40f9a`.
- [x] Re-run the committed f64 comparison from clean current Hermes main.
- [x] Generalize the planar benchmark over the current f32/f64 lane contract
      without duplicating the kernel or benchmark body.
- [x] Compare f32 medians and 95% confidence intervals in the same binary and
      inspect codegen if Hermes and `fearless_simd` do not overlap.
- [x] Adjudicate provider ownership: pinned intervals overlap and the AVX2 hot
      loops have equivalent instruction classes, so no production correction is
      justified by this measurement.
- [x] Run formatting, Clippy, focused tests, bench smoke, bounded timing,
      doctests, Rustdoc, and the exact-diff review before delivery.

## HS-LANE-THROUGHPUT-2026-08-25 [arch]

- [x] Reconcile Apollo's seven-variant same-binary evidence with the current
  Hermes provider and claim the upstream item.
- [x] Measure a consumer-shaped interleaved complex kernel through
  `SimdView`/`SimdChunks` against direct Hermes facets and `fearless_simd`
  in one bounded binary.
- [x] Inspect the emitted inner loops and locate every residual instruction,
  branch, call, spill, and bounds check absent from the reference.
- [x] Record the selected lane contract and rejected alternative in ADR 017.
- [x] Implement the minimal provider-owned correction with differential,
  property, and adversarial boundary coverage.
- [x] Re-run the Apollo power-of-two matrix and allocation census at the exact
  provider revision; Apollo PR 120 (`e3bdd7c3`) pins `4abbde8f`, passes the
  exact-Git all-target check and six batched analytical tests, and retains zero
  transient allocations on the complex path.
- [x] Pass focused Nextest, doctest, Clippy, Rustdoc, benchmark, SemVer
  classification, AArch64 compile, and release gates before the provider
  commit.
- [x] Complete exact-commit Miri, cross-target, SemVer, and ADR-index review,
  then integrate provider PR 68 as merge commit `ae4e8efa`.

## ATLAS-HERMES-ROOT-CLEANUP [patch]

- [x] Move generated benchmark baseline and result files under `benchmarks/`;
      update the benchmark runner and active documentation references.
- [x] Exact-head conformance scan: Hermes `root_sprawl` 2 → 0.

## ATLAS-ORPHAN-MODULES-096-HERMES — provider cleanup

- [x] Confirm `crates/hermes-simd-core/src/tensor/mut_view.rs` is unreachable
      from every Cargo target root and has no textual consumer; the direct
      detector returns `hermes_orphan_modules=0` after the deletion.
- [x] Delete the unreachable duplicate without touching peer-owned Hermes files.
- [x] Run the direct orphan detector plus provider format, locked check,
      warning-denied Clippy, focused Nextest, doctests, and Rustdoc gates.
      The exact hosted matrix also passes Miri, cross-compilation, ARM NEON,
      Intel SDE, bounded benchmark budgets, and supply-chain checks in
      `31819198076`.
- [x] Commit and push the provider increment; Atlas records the merged
      provider head. The current docs-only closure does not stage the
      peer-owned `Cargo.lock`.

## HS-429 [minor] — real AVX-512/AMX silicon for performance evidence

- [ ] Draft `test-avx512-hosted`: GitHub-hosted x86, machine-class record,
      AVX-512 asserted and benchmarked only when the host silicon has it.
- [ ] Keep `test-avx512-sde` and `[profile.sde]` as the deterministic semantic
      gate (hosted x86 is heterogeneous, so SDE is no longer redundant).
- [ ] Coverage step asserts `scalar,avx2` plus `avx512` when present, without
      the emulator; absence prints as NOT COVERED, never silence.
- [ ] Capture the `avx512-native` Criterion permute baseline and the
      generic-default A/B compare on a host with AVX-512.
- [ ] Adjudicate the AVX-512 permute rows (closes HS-430's AVX-512 half).

## HS-433 [patch] — structured AMX downgrade event

- [x] Replace the debug-only stderr diagnostic with one release-visible
      subscriber-owned `tracing` event carrying the routing fields.
- [x] Keep the facade no-std-capable with `tracing` default features disabled;
      enable `tracing/std` only through the facade's existing `std` feature.
- [x] Remove the unsound no-std global `Cell`/`Sync` substitute and make
      no-std AMX sessions reject safely.
- [x] Add ADR 012, changelog coverage, and a subscriber-backed value-semantic
      event test.
- [x] Run the merged provider hosted gate and full Hermes package verification.
      Exact merged-head run `31819198076` passes format, Clippy, host coverage,
      464-test nextest, doctests, examples, docs, no-std, benchmark budgets,
      Miri, cross-compile, ARM NEON, Intel SDE, and supply-chain checks.

## HS-432 [patch] — push benchmark-budget coverage

- [x] Run the benchmark-budget job on push, pull request, and manual dispatch.
- [x] Keep the 60-second smoke pass on every event and the 300-second
      canonical measurements on pull requests and manual runs.
- [x] Preserve the existing timeout, locked commands, target inventory, and
      benchmark workloads.
- [x] Validate the workflow and PM acceptance record through exact hosted run
      `31819198076`; the benchmark-budget job passes its compile, 60-second
      smoke, and bounded canonical benchmark steps.

## HS-427 [minor] — native permute overrides

- [x] AVX-512 f32/f64: reverse via `vpermxvar`, interleave/deinterleave via
      `vpermi2ps`/`vpermi2pd` over the flat `a || b` index space.
- [x] NEON f32/f64: `rev64`+`ext`, `zip1`/`zip2`, `uzp1`/`uzp2` — whole-register
      at 128-bit width, so they are the flat operations directly.
- [x] Correctness via the unchanged HS-424 differential and round-trip tests on
      the SDE and aarch64 runners.
- [x] Commit `benches/permute.rs` as the regression baseline, sized inside the
      committed per-binary runtime budget.
- [x] Measure AVX2 override versus generic default on a quiet host
      (`#[cfg(any())]` gate plus criterion save/compare baselines).
- [x] Act on the measurement: remove AVX2 interleave/deinterleave (37%
      regression), keep AVX2 reverse (10.4% faster at 1024 f32).
- [x] Add the bounded native-NEON A/B gate for HS-430: save the native
      `permute` baseline, rebuild with the explicit generic-default benchmark
      configuration, and compare the identical rows on the aarch64 runner.
- [x] Collect and adjudicate the hosted aarch64 result: `reverse` is neutral on
      both f32/f64 rows and its NEON overrides are deleted; large f32
      `interleave` and `deinterleave` improve 1.27% and 1.40% respectively and
      remain. Small rows are within Criterion's noise threshold.
- [ ] Measure AVX-512 when a hosted runner with the silicon appears (the
      `test-avx512-hosted` permute A/B); SDE remains semantic evidence only.
- [ ] Measure AMX before/after if a runner kernel admits XTILEDATA
      (HS-438 watchpoint).

## HS-424 [minor] — cross-lane permute family

- [x] Add `reverse`, `interleave`, `deinterleave` as defaulted `SimdKernel`
      methods so no backend impl changes.
- [x] Specify all three on the flat lane sequence and document why x86
      `unpack`/`permute_ps` (per-128-bit-half) are not drop-in overrides.
- [x] Override `reverse` natively on AVX2 f32 (`vpermps`) and f64 (`vpermpd`).
- [x] Test against an external slice reference, not lane arithmetic mirroring
      the implementation, across Scalar/SveArch/AVX2 for f32 and f64.
- [x] Cover both algebraic identities: `reverse∘reverse == id` and
      `deinterleave∘interleave == id`.
- [x] Confirm the new assertions are non-vacuous by deliberately breaking the
      AVX2 index vector and observing the expected failure.
- [x] Native AVX-512/NEON overlays and flat `interleave`/`deinterleave`
      overrides — HS-427, correctness verified on SDE and aarch64. Neutral
      NEON `reverse` overrides were removed after the HS-430 measurement.
- [ ] Measure the remaining AVX-512 overrides — HS-430/HS-429; performance
      evidence is separate from semantic coverage. The HS-429
      `test-avx512-hosted` job carries the measurement path.

## HS-422 [minor] — scatter seam

- [x] Add `generic_scatter` / `generic_scatter_masked` helpers over the
      workspace `IndexVector` layout invariant, const-asserted at the helper.
- [x] Add `SimdKernel::scatter` / `scatter_masked` as defaulted methods so no
      existing backend impl changes.
- [x] Override both with native `vscatterdps`/`vscatterdpd` on AVX-512 f32/f64.
- [x] Expose `SimdView::scatter` in a `view/scatter.rs` leaf module mirroring
      `view/gather.rs`; validate every index before any write.
- [x] Route the final partial vector through `scatter_masked`, not a scalar tail.
- [x] Cover per-backend differential equality, the gather∘scatter round-trip
      identity, duplicate-index last-writer-wins, and both error contracts.
- [x] Validate native AVX-512 execution. Delivered by HS-428 (the SDE job,
      which stays the deterministic semantic gate); HS-429's
      `test-avx512-hosted` job executes the scatter property, round-trip,
      duplicate-index, and error-contract tests against the
      `vscatterdps`/`vscatterdpd` override natively when the hosted x86
      runner carries AVX-512.

## HS-421 [arch] — native AVX-512 BF16 tile dispatch

- [x] Keep `avx512f,avx512bw,avx512vl` as the conversion/FMA fallback
      capability and add a distinct `avx512bf16` runtime capability SSOT.
- [x] Execute native `DPBF16PS` only after the exact BF16 probe; preserve the
      existing scalar/conversion fallback on unsupported hosts.
- [x] Preserve the `Bf16 × Bf16 → F32` and `C += A·B` contracts with a nonzero
      accumulation differential test.
- [ ] Validate native execution and benchmark speedup on a hosted AVX-512 BF16
      runner; ordinary CI continues to skip capability-specific execution.

## HS-416 [patch] — generic reduction and broader view tails

> Native SVE remains a separate blocked architecture item: stable Rust cannot
> express Hermes' scalable native vector contract yet. `SveArch` is emulated;
> its hardware probe is informational and never gates the emulated backend.

- [x] Route generic `Sum`/`Min`/`Max` partial vectors through the initialized
      provider-local masked reduction seam.
- [x] Consolidate `SimdView::sum` onto `reduce(Sum)` as the reduction SSOT.
- [x] Replace scalar tails in masked add/multiply/FMA, elementwise multiply,
      and generic `zip_into` with leading-mask buffered provider operations.
- [x] Preserve Eunomia generic min/max NaN and signed-zero semantics.
- [x] Add odd-length differential coverage for reductions and view kernels.
- [x] Route transposed GEMV column tails through initialized local lane buffers
      and the provider-owned masked-FMA seam; cover non-dyadic f32 tolerance.
- [x] Route dense dot-product tails through initialized local lane buffers and
      the provider-owned masked-FMA seam; cover odd non-dyadic f32 inputs.
- [x] Route `SimdView::zip_reduce(Dot)` pairwise tails through two initialized
      local lane buffers and the generic masked reduction seam; retain the
      scalar contract for non-opted-in multiplicative operations.
- [x] Route mutable `SimdView::transform_in_place` tails through initialized
      operand/result buffers and the generic `ElementOp` vector seam; cover
      forced emulated-SVE odd-length mutation.
- [ ] Re-run the full clean-worktree Hermes package gate after unrelated
      Cargo.lock/overlay dirt is reconciled.


## HS-415 [patch] — masked popcount tails

- [x] Route single-input popcount tails through `masked_sum_reduce`.
- [x] Route shared binary popcount tails through the same masked sum seam.
- [x] Initialize source lane buffers before blend-based full-width loads.
- [x] Cover multiple tail widths and integer bitwise combinations.
- [x] Preserve generic sum/min/max and unrelated view tails as open follow-ups.
- [ ] Re-run the full clean-worktree Hermes package gate after unrelated
      Cargo.lock/overlay dirt is reconciled.


## HS-414 [patch] — masked absolute-reduction tails

- [x] Opt `AbsSum` and `AbsMax` into the generic masked-tail reduction seam.
- [x] Copy only live tail elements into an initialized provider-local lane buffer.
- [x] Apply the absolute transform before merging inactive lanes with the
      reduction identity.
- [x] Preserve generic sum/min/max and other reduction tails as open follow-ups.
- [ ] Re-run the full clean-worktree Hermes package gate after unrelated
      Cargo.lock/overlay dirt is reconciled.


## HS-413 [patch] — masked row-update tails

- [x] Route `axpy_rows` partial vectors through `SimdKernel::masked_fmadd`.
- [x] Route `axpy_rows_batch` partial vectors through masked fused arithmetic
      while preserving depth accumulation order.
- [x] Keep both helpers sound for AVX2 blend-based masks with fully initialized
      provider-local lane buffers and exact live-tail writeback.
- [x] Cover non-dyadic f32 row and depth-batched tail semantics.
- [x] Preserve reductions, views, and other hot-kernel scalar tails as open.
- [ ] Re-run the full clean-worktree Hermes package gate after unrelated
      Cargo.lock/overlay dirt is reconciled.


**Target: Unreleased** · Strategy: [backlog.md](backlog.md) · Gap register: [gap_audit.md](gap_audit.md) · Phase: Execution

## HS-412 [patch] — masked fused AXPY-mul tail boundary

- [x] Route the final partial `axpy_mul` vector through
      `SimdKernel::masked_fmadd` after register scaling.
- [x] Keep the helper sound for AVX2 blend-based masked arithmetic by copying
      only live tail elements into fully initialized provider-local lane buffers.
- [x] Cover f32/f64 partial lengths and f32 fused-operation-order semantics.
- [x] Preserve the broader scalar-tail gap as open for separate kernels.
- [ ] Re-run the full clean-worktree Hermes package gate after the existing
      Cargo.lock/overlay dirt is reconciled.

## HS-411 [patch] — masked scale tail boundary

- [x] Route the final partial `scale` vector through `SimdKernel::masked_mul`.
- [x] Keep the helper sound for AVX2 blend-based masked arithmetic by copying
      only live tail elements into fully initialized provider-local lane buffers.
- [x] Cover f32/f64 partial lengths through the public scale facade.
- [x] Preserve the broader scalar-tail gap as open for separate kernels.
- [ ] Re-run the full clean-worktree Hermes package gate after the existing
      Cargo.lock/overlay dirt is reconciled.

## HS-410 [patch] — masked AXPY tail boundary

- [x] Route the final partial `axpy` vector through `SimdKernel::masked_fmadd`.
- [x] Keep the helper sound for AVX2 blend-based masked loads by copying only
      live tail elements into fully initialized provider-local lane buffers.
- [x] Cover f32/f64 tail sizes and add an f32 non-dyadic fused-operation-order regression.
- [x] Preserve the broader scalar-tail gap as open for separate kernels.
- [ ] Re-run the full clean-worktree Hermes package gate after the existing
      Cargo.lock/overlay dirt is reconciled.

## HS-409 [minor] — fused ternary AXPY provider facade

- [x] Add `SimdOps::axpy_mul` and the public `axpy_mul` facade.
- [x] Reuse the existing runtime-dispatch and `SimdKernel::mul`/`fmadd`
      seams; do not add a parallel SIMD abstraction or package.
- [x] Cover f32/f64 public-facade values, tails, and exact length errors.
- [ ] Adopt the provider facade in Kwavers after its consumer tree and lock
      are free; this remains a separate downstream increment.

## HERMES-MNEMOSYNE-PACKAGE-1 [patch] — Owner: Codex

- [x] Bind `mnemosyne` to package `mnemosyne-memory` 0.6.0.
- [x] Refresh dependency resolution and pass the focused core check.

## HERMES-THEMIS-PACKAGE-1 [patch] — Owner: Codex

- [x] Bind `themis` to package `themis-topology` 0.10.1.
- [x] Refresh dependency resolution and pass focused gates.
- [x] Merge before rerunning dependent Hephaestus provider CI.

## HS-403 [patch] — deterministic extrema and benchmark budgets (Owner: Codex `/root`)

- [x] Reject NaN-containing extrema inputs and return the first matching slice
      value.
- [x] Add NaN-position and signed-zero value-semantic regressions.
- [x] Remove unreachable unchecked dispatch branches.
- [x] Add metadata-derived, precompiled 60-second smoke budgets for every
      workspace Criterion target, plus 300-second full-run budgets for the
      changed canonical dense and SIMD instruments under a 30-minute job cap.
- [x] Preserve one SIMD benchmark binary and all 60 SIMD IDs while separating
      sum, full-precision dot, and reduced-precision dot registration by SRP.
- [x] Preserve all 48 dense-suite IDs while replacing Criterion's emergent
      linear iteration multiplier with flat sampling at Criterion's 10-sample
      floor, 100 ms warm-up, and 500 ms measurement budgets.
- [x] Pass direct rustfmt and code-bearing exact-head hosted CI at `0487bbd`
      (run `29963673513`): x86 gates, AArch64 runtime, cross-compile, Miri,
      cargo-deny, CodeRabbit, and Greptile are green. Benchmark evidence:
      2m04s compile, 3m19s all-target smoke, 3m18s scoped full timing, and
      9m00s total job time.

## HS-402 [patch] — Eunomia 0.6 lock convergence (Owner: Codex)

- [x] Regenerate the provider lock through Cargo against merged Eunomia 0.6.
- [x] Pass format, all-target/all-feature warning-denied Clippy, Nextest
      (388/388), doctests (18/18 runnable), rustdoc, and dependency-identity
      checks.
- [x] Publish and merge Hermes PR #10 at `53a8e03`; hand the parent gitlink
      refresh to Atlas item ATLAS-INTEGRATION-027.

## HS-401 [arch] — Eunomia reduced-precision cutover (Owner: Codex)

- [x] Reconcile the preserved peer WIP and verify it compiles against Eunomia
  main before takeover.
- [x] Replace raw half types natively across scalar/ISA/tiled kernels and every
  Hermes call site; delete every direct `half` dependency from Hermes
  manifests. The lock retains transitive `half` through Eunomia and Ciborium.
- [x] Add or update value-semantic cross-backend tests for the Eunomia types.
- [x] Pass format, all-feature warning-denied Clippy, full Nextest (388/388),
  doctests, rustdoc, no-default-feature checks, dependency/source residue
  audits, and current-workspace compilation. Semver classification is
  externally blocked because the `origin/main` baseline resolves moving
  Eunomia main 0.5.0 and no longer compiles its historical raw-half calls.
  PR #8's AArch64 cross-compile and runtime NEON lanes pass, as does Miri; the
  x86 lane rerun is the remaining merge gate.
- [x] Publish and merge Hermes PR #8 at `8970ffc`, preserving the peer-owned
  kernel work and closing both CI-discovered host-sensitive regressions.
- [ ] Advance Leto to the merged Hermes/Eunomia provider state.

- [x] [patch] Converge Themis, Mnemosyne, and Eunomia onto their default
  branches and remove workspace-local patch overrides. Acceptance: the locked
  `hermes-simd` graph contains one identity for each provider; format,
  warning-denied Clippy, canonical nextest, and rustdoc are clean. Verified
  2026-07-15 with `cargo tree --locked -d -p hermes-simd`, `cargo fmt --check`,
  `cargo clippy -p hermes-simd --all-targets --all-features --locked -- -D
  warnings`, `cargo nextest run -p hermes-simd --locked --no-fail-fast`, and
  `cargo doc -p hermes-simd --no-deps --locked`. CI source policy is updated
  for the default providers and `cargo deny check` passes locally; the PR CI
  rerun remains the merge gate.

## Sprint scope: ship 0.2.0 with CI

- [x] [patch] `hermes-simd` AMX auto-dispatch mitigation: AMX support probes now
      return false until Hermes has a stable, permission-aware AMX probe that
      verifies hardware bits, XCR0 OS state, and Linux XTILEDATA process
      permission. This removes unstable Rust AMX feature-detection macro calls
      and prevents CPUID-only AMX dispatch. AVX-512 tile probes still use exact
      stable `is_x86_feature_detected!` checks. Evidence tier: compile-time
      validation. Checks: `cargo check -p hermes-simd` and `cargo clippy -p
      hermes-simd --all-targets -- -D warnings`.
- [x] [patch] `.github/workflows/ci.yml`: fmt-check, clippy `-D warnings`,
      `cargo test --workspace` (x86_64 + native aarch64 runner), warning-clean
      docs, aarch64 cross-check, cargo-deny. → green (run 27296233212; required three fixes: libnuma feature gating, license inheritance, AMX bench import cfg).
- [x] [patch] `rust-toolchain.toml` (1.95.0) + `rust-version = "1.95"` in the
      workspace manifest, inherited by all members. MSRV verified empirically:
      full workspace build + 295-test pass on rustc 1.95.0.
- [x] [patch] `deny.toml` (advisories/licenses/bans/sources) + CI job.
      Local `cargo audit` blocked by an outdated local advisory parser
      (CVSS 4.0); CI cargo-deny is the authoritative gate.
- [x] [patch] `f16`/`bf16` interleaved-complex differential tests — bitwise
      equality for elementwise multiply (lane-emulated backends share the op
      sequence); dot compared under the analytical reordering bound
      `(n+8)·ε_T·Σ magnitudes`.
- [x] [patch] Kernel property suite (`kernel_property_tests.rs`): bitmask
      round-trip, compress∘expand identity, gather vs scalar reference,
      `leading_k_mask` boundaries — per backend, feature-gated.
- [x] [minor] Version bump 0.1.0 → 0.2.0; CHANGELOG 0.2.0 section dated;
      `cargo-semver-checks` pass on hermes-simd and hermes-simd-core vs the
      previous rev (196 checks each, no regression).
- [x] [minor] Tag `v0.2.0` — created after CI run 27296233212 (all four jobs green, including runtime NEON validation on a native aarch64 runner).

## Residual risks

- AVX-512 and AMX hot paths: differential tests self-skip on unsupported hosts;
  safe constructors/lifecycle APIs now reject unsupported targets before ISA
  execution, but no AVX-512/AMX CI runner exists yet ([backlog → P0](backlog.md#p0)).
- `cargo-semver-checks --workspace` cannot doc-build `hermes-numeric` under
  its feature-combination probing (rkyv `size_*` feature requirement);
  per-crate scoped runs are the working procedure.
- `panic = "abort"` in the release profile: CI tests the dev profile.
- Full Themis topology consolidation closed: Hermes no longer owns the public
  `NumaTopologyService`/node-count/node-distance facade, and
  `MnemosyneNumaAllocator` no longer owns direct platform allocation fallback
  branches. Consumer topology queries route to Themis; allocation ownership
  routes through Mnemosyne/the configured allocator path.

## Post-0.2.0 increment (2026-06-10)

- [x] [minor] Sparse `Validated` typestate follow-up: added
      `Validated<F>`/`ValidatedData<S>`, moved CSR/SELL-p/Blocked-COO SpMV onto
      `SparseView<Validated<_>>`, changed public SpMV dispatch to require
      validated storage, and added validated COW constructors. Regression and
      property tests assert malformed sparse layouts fail at construction and
      generated valid layouts match scalar references. Evidence tier:
      type-level invariant + property tests. Checks: `cargo fmt --check` clean;
      `cargo check -p hermes-simd` could not acquire the shared Cargo target lock
      in this pass.
- [x] [minor] Safe-code ISA fault hardening: `SimdArch::is_runtime_supported`
      is the SSOT for safe vector/mask wrappers and `TargetId`; unsupported
      AVX-512 hosts get `SimdError::UnsupportedTarget` from fallible vector
      constructors/checked slice wrappers before any AVX-512 instruction, and
      infallible vector conveniences panic before ISA execution. `AmxSession::new`
      and `AmxBatchSession::begin` now return `AmxSessionError::UnsupportedTarget`
      before `ldtilecfg`; `release` guards `tilerelease`. Evidence tier:
      type-level trait seam + value-semantic unsupported-host regressions.
- [x] [patch] `cargo miri` over hermes-simd-core: unit tests green; rkyv 0.7
      tests `#[cfg_attr(miri, ignore)]` (upstream Stacked Borrows violations);
      CI `miri` job added.
- [x] [patch] no_std: `#[runtime_dispatch]` std-gating fixed;
      `--no-default-features` check green locally and in CI.
- [x] [patch] `#![deny(missing_docs)]` on all six public crates; 12 items documented.
- [x] [minor] Complex criterion bench suite + recorded baselines
      (benchmarks/benchmarks_results.md); threshold automation delivered below.
- [x] [patch] x86 VNNI asm cleanup: `vpdpbssd` factored into one internal
      asm macro with `nostack`/`nomem`/`preserves_flags`; both AVX-512 tile
      kernels expand it inside the target-feature-gated loop. Added complete
      signed-nibble INT4 unpack regression coverage and documented the asm
      scope boundary.
- [x] [patch] local-capable test/bench hardening: host dispatch tests cover
      the locally detected dense backend, AVX2 direct execution when present,
      and irregular INT8 GEMM against scalar reference. Benchmark report
      generation records detected ISA features and suppresses AMX
      context-pressure rows on non-AMX hosts; dense scalar baselines now
      black-box operands/accumulation to prevent optimized-away work.
- [x] [patch] Miri intrinsics boundary: VNNI/AMX compute asm panics under
      Miri instead of returning synthetic values; AMX configuration/release
      instructions are no-ops only for session-state tests; CI now runs
      `cargo +nightly miri test -p hermes-simd-intrinsics`.
- [x] [patch] runnable doctest coverage: enabled doctests for
      `hermes-simd-core` and added value-semantic examples for complex
      multiplication/dot, sparse CSR `SparseCow` SpMV, and const-generic
      `TensorView`.
- [x] [patch] SVE callable fallback: `SveArch` f32/f64 now use the
      monomorphized lane-emulated kernel macro (`16xf32`, `8xf64`) with
      value-semantic tests. Native SVE intrinsics remain tracked separately.
- [x] [minor] SVE property coverage: `hermes-simd` re-exports `SveArch`, and
      the shared kernel property suite now runs its mask/compress/expand,
      gather, and leading-tail invariants on every host.
- [x] [patch] runnable core doctests: kernel, compute, and tiling examples now
      execute value assertions under `cargo test --doc --workspace` instead of
      compile-only `no_run`.
- [x] [patch] runnable BitMask doctests: native-mask conversion and
      active-lane iteration examples now execute value assertions.
- [x] [patch] Default provider features: every Hermes package now defaults
      `parallel` and `mnemosyne-memory`; `hermes-simd-core` pins Mnemosyne
      `938d0c2` and routes `AlignedVec::with_capacity_numa` allocation and
      deallocation through `mnemosyne::Mnemosyne` under the default feature.
      Verification: fmt, clippy all-targets/all-features, workspace tests,
      warning-clean docs, and no-default-features check.
- [x] [minor] Absolute reductions: `AbsSum` / `AbsMax` and dispatched
      `abs_sum` / `abs_max` provide Hermes-owned L1 and infinity norm
      accumulators for Leto/Apollo consumers without temporary buffers.
      Evidence tier: value-semantic tests plus full workspace gate (`fmt`,
      `check`, `test`, `clippy -D warnings`, docs).
- [x] [minor] Criterion threshold automation: `run-benches` now writes
      `benchmarks/benchmarks_baseline.json`, enforces baseline rows with
      `--check-regressions`, and is split into SRP modules (`cli`,
      `criterion_results`, `host`, `regression`, `report`) instead of a
      542-line mixed-concern entrypoint. Evidence tier: value-semantic unit
      tests for CLI/regression/report parsing, local dense Criterion run, and
      baseline self-check over 36 rows.
- [x] [minor] SpMV scalability sweep: sparse Criterion bench now covers
      CSR/SELL-p/BCOO over 1K, 10K, and 100K rows at 0.1%, 1%, and 10%
      structural non-zero density with bounded Dense-with-mask cases through
      10K rows. Sparse module docs now state format-selection guidance.
- [x] [patch] Atlas compute boundary docs: README states Hermes owns SIMD
      lane-parallel kernels and slice-oriented dispatch, Moirai owns MIMD
      scheduling, and Hephaestus owns GPU/device lifetimes.
- [x] [minor] Packed4 unpack generalization: `Packed4CowExt` now calls the
      canonical `Packable4` packed dispatcher, so the facade inherits AVX-512,
      AVX2, and scalar runtime selection without a duplicate x86 branch.
      Coverage: odd-length full-nibble COW unpack regression plus a focused
      Criterion benchmark target.
- [x] [minor] Complex `mul_assign` unroll: in-place interleaved complex
      multiply now processes two SIMD registers per loop iteration before the
      single-register and scalar tails, with a direct four-pair scalar loop for
      large scalar buffers. Evidence: focused Criterion runs plus refreshed
      48-row local AVX2 baseline.
- [x] [patch] README current-version metadata corrected from `0.1.0` to
      `0.2.0`.
- [x] [patch] Benchmark baseline refresh: `run-benches --parse-only
      --write-baseline --check-regressions` regenerated
      `benchmarks/benchmarks_baseline.json` and `benchmarks/benchmarks_results.md` from local
      Criterion output, including packed4 COW unpack and the unrolled complex
      `mul_assign` rows. Regression self-check covered 48 rows.
- [x] [patch] Compress scratch-hoist benchmark: added
      `compress_bench` with public `SimdView::compress` scalar and host-AVX2
      all/half/quarter-mask rows at 1K, 16K, and 256K elements. Refreshed
      `benchmarks/benchmarks_baseline.json` / `benchmarks/benchmarks_results.md`; regression
      self-check covered 102 Hermes rows. Evidence tier: empirical Criterion
      validation plus existing value-semantic compress regressions.
- [x] [minor] Const-generic Blocked-COO dispatch: removed fixed public
      `spmv_bcoo4x4`/`spmv_bcoo8x8` dispatch functions and fixed
      `SparseView::from_blocked_coo_4x4`/`from_blocked_coo_8x8` constructors
      in favor of one `spmv_bcoo::<T, BM, BN>` API and the existing generic
      `from_blocked_coo` constructor. Evidence tier: type-level const-generic
      shape encoding plus value-semantic sparse tests and benchmark parser
      regression self-check.
- [x] [minor] Const-generic SELL-p dispatch: removed fixed public
      `spmv_sellp4`/`spmv_sellp8` dispatch functions and fixed
      `SparseView::from_sellp4`/`from_sellp8` constructors in favor of one
      `spmv_sellp::<T, C>` API and the existing generic `from_sellp`
      constructor. Evidence tier: type-level const-generic slice-height
      encoding plus value-semantic sparse tests and benchmark parser
      regression self-check.
- [x] [patch] Highway comparison audit: audited
      `https://github.com/NikoMalik/highway.git` at
      `0984271e74db124cf5e200de542e745348eb0b9e` and recorded Hermes-native
      follow-ups in `gap_audit.md`, `backlog.md`, and README. Evidence tier:
      source audit plus local code search.

## Next sprint focus (from [gap_audit](gap_audit.md#highway-2026-06-14))

- [x] [minor] Target-token forced dispatch: define the Hermes-native
      `TargetId`/forced-dispatch test surface and prove unsupported targets
      reject safely. Evidence tier: type-level architecture view construction
      plus value-semantic host capability tests.
- [x] [minor] Safe one-vector slice wrappers: add bounds/alignment-checked
      wrappers over `SimdKernel` load/store primitives and value-semantic
      tests for success and failure paths. Evidence tier: value-semantic
      integration tests over public `Vector<T, Arch>` methods.
- [x] [minor] Public dense cross-target matrix: compare public facade outputs
      against Scalar across every host-supported target. Evidence tier:
      value-semantic differential tests over forced `TargetId` views.
- [x] [minor] Batched AXPY rows: add `axpy_rows_batch` to the sealed
      `SimdOps` facade and runtime-dispatch it through the existing AXPY
      kernel family. Evidence tier: value-semantic differential test against
      repeated `axpy_rows`, exact invalid-extent error assertions, and Miri
      coverage of the unsafe pointer loop. Memory model: each output lane is
      loaded once, accumulated across depth in registers, and stored once.
      Benchmark coverage: `axpy_rows_batch_f32` compares the fused path with
      repeated public `axpy_rows` calls.
- [x] [patch] Dense/AXPY error-contract hardening: length-mismatch tests now
      assert exact `SimdError::LengthMismatch` values instead of only
      asserting that an error exists.
- [x] [patch] Select/unary error-contract hardening: select, unary-map, and
      COW FMA tests now assert exact `SimdError` variants for length mismatch
      and insufficient output capacity.
- [x] [patch] Operation-family error-contract hardening: new operation,
      strategy, complex, gather, scan, and COW math tests now assert exact
      `SimdError` variants for invalid shape, short output, and invalid index
      cases.
- [x] [patch] COW unary invariant cleanup: `SimdCow::map_unary` now asserts
      its internally constructed output-length invariant instead of silently
      discarding the `SimdView::map_unary` result.
- [x] [patch] GEMM tiling rustdoc cleanup: module theorem prose now references
      private implementation details as code text instead of public intra-doc
      links.
- [x] [patch] Operation-family coverage map: README and backlog now split the
      Highway-derived coarse coverage gap into delivered families and
      consumer-demand pending families. Evidence tier: source audit, not a new
      implementation or benchmark claim.
- [x] [patch] Runtime FMA capability probe: cached `has_fma3` and
      `FmaSupport` impls now use the platform-aware runtime detector and have
      host-capability coverage.
- [x] [patch] GEMV rustdoc link cleanup: public docs now disambiguate
      same-named dispatch modules and functions.
- [x] [minor] Audit sprint (2026-06-26): numeric integer-type extension + first
      `hermes-numeric` tests; `MAX_SIMD_LANES` 128→64 with SSOT unification;
      NUMA alloc-generation ordering/TOCTOU hardening; `build_index_vector`
      compile-time layout guard; `forbid(unsafe_code)` on macros; magic CAS
      ordering; `SimdOps` macro-collapse; `flush_limit`/`axpy`/`scale` cleanup.
      367 tests + clippy + fmt green. See [gap_audit](gap_audit.md#audit-2026-06-26).
- [x] [patch] Reduction hierarchy cleanup (2026-08-15): moved the
      multiplicative `Product` strategy into the dedicated
      `ops/reduction/product.rs` leaf, leaving `reduction.rs` at 442 lines.
      The public strategy and generic SIMD implementation are unchanged;
      `hermes-simd` nextest passed 410/410 and the provider lint/check gates
      passed. See [gap_audit](gap_audit.md#hermes-2026-08-15).

## Next sprint candidates (from [backlog](backlog.md))

- [x] [minor] 0.3.0 release for the additive absolute-reduction API.
- [x] [minor] Fast reciprocal square root (`ops::RecipSqrt`) with Newton-Raphson
      refinement to eliminate standard `sqrt` latency (from [gap_audit](file:///d:/atlas/repos/hermes/gap_audit.md#numkong-2026-06-17)).
- [x] [arch] Masked tail-load/store infrastructure for AVX-512 / SveArch to enable Leto to run tail-free kernels (from [gap_audit](file:///d:/atlas/repos/hermes/gap_audit.md#numkong-2026-06-17)).
- [x] [minor] Expose popcount and horizontal reductions to support Jaccard/Hamming in Leto (from [gap_audit](file:///d:/atlas/repos/hermes/gap_audit.md#numkong-2026-06-17)).
- [x] [minor] Sub-byte sign-extension and unpacking/widening SIMD primitives for `Bf4`/`F4`/`I8` (from [gap_audit](file:///d:/atlas/repos/hermes/gap_audit.md#numkong-2026-06-17)).
- [x] [arch] Per-type x86 kernel dedup ADR (P3); revised as ADR 005 on
      2026-08-21 when the incomplete generator was retired and checked-in ISA
      files became canonical.
- [x] [arch] x86 kernel dedup generator script; retired on 2026-08-21 after a
      pinned regeneration audit showed destructive coverage drift across the
      shipped x86 surface. See `backlog.md` and `gap_audit.md`.
- [x] [minor] HS-437 lane-buffer audit: release assembly for Scalar f64,
      emulated `SveArch` f64, and AArch64 NEON f64 shows no stack frame in the
      default `interleave` path; the proposed typed `LaneBuffer` refactor is
      not justified by codegen evidence.
