# Backlog — hermes-simd

- [x] [patch] **HS-434 — workspace lint floor.** The workspace had no
  `[workspace.lints]` table at all, so the lint policy lived in three
  overlapping per-crate `#![allow(..)]` blocks and four copies of
  `#![deny(missing_docs)]`. The blanket suppressions were the real cost:
  `clippy::missing_safety_doc` was off across the whole of
  `hermes-simd-intrinsics` — the crate with ~1270 `unsafe` sites — so an
  `unsafe` public function could ship with no `# Safety` section and nothing
  would say so.
  Delivered: one `[workspace.lints]` table (`clippy::pedantic` at warn, plus
  denied `unwrap_used`/`dbg_macro`/`print_stdout`/`print_stderr` and
  `allow_attributes` to drive `#[allow]` -> `#[expect]`), inherited by all
  seven members with `[lints] workspace = true`; the per-crate blocks are
  deleted and their still-valid entries consolidated into the one table, each
  carrying the domain reason it is allowed. Fixed in the same change: the
  `codegen` bin gained crate docs and propagates its file-I/O errors instead of
  four `unwrap()`s, and the AMX downgrade notice carries a per-site `#[expect]`
  (HS-433).
  Measured floor: 2152 library-src pedantic findings remain at warn; they are
  the HS-435 ratchet, not a silent allow.

- [ ] [patch] **HS-433 — AMX downgrade notice writes to stderr.**
  `dispatcher.rs` surfaces the cross-NUMA-node AMX -> AVX-512 re-route with a
  debug-only `eprintln!`, so in release builds the downgrade is silent — the
  fallback-must-surface-its-trigger rule wants a `tracing` event. Not done now:
  `hermes-simd` has no `tracing` dependency and adding one to a
  `no_std`-capable facade is an ADR-level call. Carries a per-site
  `#[expect(clippy::print_stderr)]` naming this item.
  Re-open trigger fired (partially): `has_amx()` is a real probe now, so the
  notice is no longer unreachable in principle — it fires on AMX silicon whose
  process holds tile permission. No such machine is in CI, so the ADR call is
  still not forced, but the "cannot execute" justification no longer holds and
  this item is now blocked only on the `tracing` decision, not on AMX.
  Acceptance: the re-route emits a `tracing` event in release builds, asserted
  by a test capturing the subscriber.

- [ ] [minor] **HS-435 — pedantic ratchet.** The lint floor (HS-434) is set to
  warn against the remaining library-src findings. Non-increasing baseline,
  burnt down by class rather than by file. Acceptance: each increment lowers
  the recorded count and never raises it.
  Measured by lint (`--message-format=json`, deduped by lint+file+line+col, so
  these are exact rather than the earlier message-text estimates):

  | count | lint | note |
  |------:|------|------|
  | **0** | `ptr_as_ptr` | **done** — was 171, see below |
  | 180 | `unreadable_literal` | bitboard/mask hex constants; check readability convention before sweeping |
  | 162 | `must_use_candidate` | next up: real API-contract value |
  | 131 | `elidable_lifetime_names` | mechanical |
  | 92 | `semicolon_if_nothing_returned` | mechanical |
  | 83 | `missing_errors_doc` | required by the documentation standard |
  | 63 | `cast_lossless` | prefer `From` over `as` where infallible |
  | 62 | `doc_markdown` | backticks |
  | 49 | `uninlined_format_args` | mechanical |
  | **0** | `allow_attributes` | **done** — library source suppressions now use reasoned `#[expect]` or are deleted when unfulfilled; test/bench-only suppressions remain outside this library-source ratchet |
  | 21 | `cast_ptr_alignment` | **not mechanical** — each is a real alignment claim to check against the load it feeds |
  | 19 | `missing_panics_doc` | required by the documentation standard |
  | 13 | `ref_as_ptr` | `core::ptr::from_ref`; same family as the finished item |
  | 9 | `ptr_cast_constness` | `.cast_const()`/`.cast_mut()`; same family |
  | 8 | `missing_safety_doc` | the sites HS-434 unhid; safety-bearing |

  Sequence: the doc sections and `must_use` next (contract-bearing), then
  `cast_ptr_alignment` reviewed individually rather than swept, cosmetics last.

  **`ptr_as_ptr` burned down: 171 -> 0.** 167 sites converted mechanically by
  `cargo clippy --fix` restricted to that one lint; the remaining 4 were
  inference-typed (`as *const _`) so carried no machine-applicable suggestion
  and were done by hand. `.cast::<T>()` cannot change constness, which is the
  point: `as` silently can. Net -77 lines, since the shorter form let rustfmt
  collapse call sites that had been wrapped.
  Note for whoever takes the next class: `--fix` obeys the *workspace* lint
  table, not just the lint you pass, so `-A clippy::all -A clippy::pedantic
  -A clippy::restriction` is required to keep the diff to one transform. Without
  it the run also rewrote 26 `#[allow]` -> `#[expect]` and left 19 unfulfilled
  expectations behind.

  **`allow_attributes` burned down 26 -> 0 (2026-08-14).** The source sweep
  replaced target-specific and macro-contract suppressions with reasoned
  `#[expect]` attributes, deleted unfulfilled `unused_mut` and `dead_code`
  suppressions, and made the aarch64-only unreachable-code expectation
  conditional. A focused `clippy::allow_attributes` run over the affected
  core, intrinsics, facade, and macro packages is clean; core nextest is 16/16.

- [ ] [major] [arch] **HS-436 — `SimdKernel` is a god trait.** One sealed trait
  carries ~60 methods across load/store, streaming, dense arithmetic, masked
  load/store, masked arithmetic, compress/expand, gather, scatter, mask
  construction, scan, elementwise math, comparison, reduction, cross-lane
  permute, and adjacent-pair complex support.
  Note on the file size: `kernel.rs` is 1115 lines but 622 of those are doc
  comments, so it holds roughly 490 lines of code and is *not* over the
  500-line target. This item is therefore about interface segregation, not
  file length — do not justify it on line count.
  Interface segregation wants role supertraits (`SimdLoadStore`, `SimdArith`,
  `SimdMask`, `SimdPermute`, `SimdReduce`, ...) with `SimdKernel` retained as
  the aggregate so call-site bounds and the five backend impls keep working.
  Not free: every `#[target_feature]` impl block splits per role, which
  multiplies impl blocks across 5 backends x several scalar types.
  Needs an ADR before implementation — record the split, the aggregate-trait
  compatibility argument, and the measured impl-block cost. Acceptance: no
  behavioural change, `cargo bloat`/codegen evidence that dispatch stays
  monomorphized, and each role module under the 500-line target.

- [ ] [minor] **HS-437 — lane scratch buffers are sized to the workspace
  maximum.** The `SimdKernel` default methods and `kernel_helpers` declare
  scratch as `[MaybeUninit<T>; MAX_SIMD_LANES]` with `MAX_SIMD_LANES = 64`,
  the widest backend/type pair in the workspace, rather than the backend's own
  `LANE_COUNT`. `interleave`/`deinterleave` each declare four such buffers, so
  a NEON `f64` call (2 live lanes) reserves 2 KB of stack to move 32 bytes.
  Correctness is not affected — the buffers are deliberately over-sized, not
  over-read, and `LANE_BOUND_CHECK` pins the bound at compile time.
  `Self::LANE_COUNT` cannot be an array length in a default body on stable, so
  the fix is an associated `type LaneBuffer` (or const-generic lane parameter)
  that each backend fixes to its exact width.
  Evidence required before implementing: stack-frame measurement on the
  default-path backends, since LLVM may already narrow the frame through the
  `store_unaligned`/`load_unaligned` pair. If it does, this closes as
  "no measurable effect" rather than being implemented on principle.

- [x] [minor] **HS-422 — scatter seam.** Add `SimdKernel::scatter` and
  `scatter_masked` as defaulted trait methods over a generic lane-sequential
  helper, override them with native `vscatterdps`/`vscatterdpd` on AVX-512
  f32/f64, and expose `SimdView::scatter` as the public dual of
  `SimdView::gather`. Full vectors use the unmasked seam and the final partial
  vector uses the masked seam, so no scalar tail remains. Indices are validated
  before any write; duplicate indices are last-writer-wins. The native AVX-512
  path is executed under the HS-428 Intel SDE job.

- [x] [minor] **HS-423 — rounding primitives.** The kernel trait has no
  `floor`/`ceil`/`round`/`trunc` although every target ISA provides them
  natively. Blocked on an upstream eunomia unit: `NumericElement` exposes no
  rounding, so the Hermes default would have to widen through `to_f64` and
  narrow back — a fake generic. Sequence: eunomia adds rounding to
  `NumericElement` (identity on integer impls) and releases; Hermes bumps, adds
  the defaults over `generic_unary_op`, and overrides on AVX2/AVX-512/NEON.
  Acceptance: per-backend differential tests against the scalar reference,
  covering negative values, halfway cases (the round-half-to-even contract must
  be pinned explicitly), and the infinity/NaN contract.
  Re-open trigger: the eunomia rounding release lands.
  Delivered on `feat/hermes-rounding` (58c31a9 + df32296): trait defaults over
  `generic_unary_op`; AVX2 (`vroundps`/`vroundpd`), AVX-512
  (`_mm512_roundscale_ps/pd` — stdarch lacks `_mm512_floor/ceil_*`), and NEON
  (`frintm`/`frintp`/`frintn`/`frintz`) overrides; `RoundTiesEven` sealed trait
  routing `f32`/`f64` to inherent `round_ties_even` and the reduced-precision
  wrappers to the exact round-narrow path; `Floor`/`Ceil`/`Round`/`Trunc`
  UnaryOp strategies re-exported from `hermes_simd`. Differential coverage:
  `rounding_matches_reference_all_backends` pins bit-exact equality against the
  plain-scalar family over ties, straddling values, ±Inf, NaN, and signed
  zeros on Scalar/SveArch/AVX2 (this host) and AVX-512/NEON (compile-only here,
  exercised in CI); the UnaryOp seam is covered via `map_unary`/in-place.

- [x] [minor] **HS-424 — cross-lane permute family.** `SimdKernel::reverse`,
  `interleave`, and `deinterleave` join the trait as defaulted methods, so lane
  reordering no longer requires leaving the vector domain and no backend impl
  changed. All three are specified on the *flat* lane sequence, not per 128-bit
  sub-lane — the distinction that makes x86 `unpack` unusable as a drop-in
  override. `deinterleave` is the exact inverse of `interleave` and `reverse`
  is an involution; both identities are tested. AVX2 overrides `reverse`
  natively (`vpermps` by index vector for f32, `vpermpd` by immediate for f64),
  verified on this host and confirmed non-vacuous by a deliberate-break check.
  AVX-512 and NEON overrides, and native flat `interleave`/`deinterleave`, are
  deferred to HS-427 rather than shipped as unexecutable index math.

- [x] [patch] **HS-428 — identify and assert per-runner backend coverage.**
  Backend selection is automated from runtime probes, which is correct, but a
  probe-guarded test that does not run does not fail either — it skips, and a
  skip is indistinguishable from a pass. That is how this workspace carried a
  green CI in which no AVX-512 path had ever executed.
  Primary mechanism (real hardware, no emulation): `TargetId::{ALL, from_name,
  supported_on_host, is_architecture_applicable}` let a harness enumerate the
  closed set and identify what the host executes. A coverage test prints the
  matrix in every job via a `--no-capture` step and asserts against
  `HERMES_EXPECTED_TARGETS`, declared per runner as configuration: the aarch64
  runner must execute NEON on real silicon, the x86 runner must execute AVX2.
  The report distinguishes three outcomes — executes, NOT COVERED (architecture
  applies, CPU lacks it), n/a (different architecture) — because collapsing the
  last two makes an ARM log read as missing AVX-512.
  Fallback only where no selectable silicon exists: AVX-512 cannot be requested
  on GitHub-hosted runners (some x86 machines have it, some do not, and this
  repository has no self-hosted runners), and AMX is unavailable there
  entirely. `test-avx512-sde` therefore runs the suite under Intel SDE emulating
  Sapphire Rapids (447/447 in 176s, ~11x native) through the cargo target
  runner, so only test binaries pay the cost. Its identification step runs under
  the emulator too, so passing is a hard assertion that SDE satisfies the
  runtime probes rather than merely not breaking. Uses a dedicated
  `[profile.sde]` 300s budget; the 30s native budget is untouched.
  Known limit: SDE validates semantics, never performance — any benchmark claim
  still requires real silicon. See HS-429.

- [ ] [minor] **HS-429 — real AVX-512/AMX silicon for performance evidence.**
  SDE gives deterministic semantic coverage but cannot support a performance
  claim, so HS-427's "override beats the default" acceptance is unsatisfiable
  under emulation. A runner provider that pins instance families (RunsOn-class,
  resolving e.g. `family=c7i` — Sapphire Rapids, AVX-512 plus AMX) supplies
  genuine hardware in the project's own cloud account. Acceptance: an AVX-512
  job on real silicon whose coverage step asserts `scalar,avx2,avx512` without
  the emulator, plus a criterion baseline captured there. If adopted, the SDE
  job becomes redundant and is deleted rather than kept alongside.
  Precondition: a cost/infrastructure decision, which is the user's to make.

- [x] [minor] **HS-427 — native permute overrides beyond AVX2 reverse.**
  Delivered with one premise falsified. AVX-512 f32/f64 override all three ops
  (`vpermps`/`vpermpd` for reverse, `vpermi2ps`/`vpermi2pd` for the two-vector
  permutes, whose index space is the flat `a || b` concatenation the trait
  contract is written on); NEON f32/f64 override all three (`rev64` + `ext`,
  `zip1`/`zip2`, `uzp1`/`uzp2` — NEON's zip/uzp are whole-register, so at
  128-bit width they *are* the flat operations). Correctness is verified by the
  existing HS-424 differential and round-trip tests on the SDE and aarch64
  runners, unchanged.
  **The AVX2 interleave/deinterleave overrides were written, measured, and
  removed.** `unpack` + `permute2f128` is a *37% regression* against the
  generic store/permute/load default at L1-resident size (two runs agree,
  p < 0.05), and deinterleave was neutral-to-negative. LLVM already lowers the
  generic default's stack round-trip into good shuffle sequences, so the
  hand-written cross-half fixup buys nothing and costs latency. AVX2 `reverse`
  survives on measurement: 10.4% faster at 1024 f32, 2.8% at 1024 f64.
  Committed `benches/permute.rs` is the regression baseline. Measure at the
  L1-resident size — at 16384 elements the working set spills and the permute
  cost disappears into memory traffic, which is why the two sizes disagree.
  Residual: AVX-512 and NEON override *performance* is unverified. SDE cannot
  measure it (HS-429) and the aarch64 job runs no benchmark. Given the AVX2
  result, these may not pay either; they are kept as correctness-equivalent
  canonical lowerings, not as a speed claim. Follow-up HS-430.

- [x] [patch] **HS-431 — repair the panicking compress benchmark.**
  `compress_bench`'s scalar rows built a `BitMask::<1>` for a backend whose f32
  lane count is 4, so `SimdView::compress`'s lane-count assertion aborted the
  binary. Broken since 2afe675 (2026-07-07) and undetected because the
  benchmark job runs only on pull requests and dispatches, and the one
  intervening pull request failed earlier at the gates — the budget job never
  reached it. The width now derives from
  `<Scalar as SimdKernel<f32>>::LANE_COUNT` rather than a literal, so a backend
  width change cannot silently rot it again. All thirteen bench targets smoke
  clean locally.
  Trigger-coverage finding: a job gated to pull requests is not a gate for work
  that lands by direct push. Either the budget job runs on push too, or bench
  rot is caught only when someone opens a pull request — filed as HS-432.

- [x] [patch] **HS-432 — benchmark budget job never runs on pushed work.**
  Owner: Codex on `codex/hermes-benchmark-trigger`; claimed 2026-08-12.
  `benchmark-budgets` is gated to `pull_request` and `workflow_dispatch`, so
  every commit that reaches `main` by direct push — which is how this stream
  delivers — skips it entirely. HS-431's panic survived a month that way.
  Resolution: the job now runs on push, pull request, and manual dispatch. The
  compile and every-binary 60-second smoke run execute for all events; the two
  300-second canonical measurements remain on pull requests and manual runs.
  A bench that panics or breaches its smoke budget therefore fails CI on the
  same event that introduced it. The push path does not claim full benchmark
  performance evidence.

- [ ] [patch] **HS-430 — measure the AVX-512 and NEON permute overrides.**
  HS-427 shipped them on correctness alone. The AVX2 result — a hand-written
  native sequence losing 37% to the generic default — is the reason this cannot
  be assumed. Method: the override-versus-default comparison HS-427 used, which
  is a `#[cfg(any())]` gate on the override plus a criterion
  `--save-baseline`/`--baseline` pair on a quiet host. NEON needs a bench step
  on the existing aarch64 runner; AVX-512 needs HS-429's real silicon.
  The aarch64 workflow now runs the existing `permute` Criterion target twice:
  first with the native NEON methods and a `neon-native` saved baseline, then
  with the three NEON overrides disabled by the explicit
  `hermes_benchmark_generic_default` benchmark configuration and compared
  against that baseline. The command is bounded at 300 seconds and uses the
  same inputs, groups, and Criterion settings in both runs, so the result is a
  real-silicon A/B measurement rather than a compile-only claim.
  Acceptance: each override either shows a significant win and stays, or is
  deleted like the AVX2 pair.
  UNBLOCKED as of HS-428 — both preconditions were already satisfiable and the
  original entry was wrong to defer on runner availability. aarch64 has had a
  native `ubuntu-24.04-arm` job running the full suite all along (and
  `cargo check --target aarch64-unknown-linux-gnu` type-checks NEON locally
  with no ARM hardware); AVX-512 is now executed under SDE. The HS-424
  differential and round-trip tests already contain the per-backend branches,
  so an override is validated on push.
  HS-424 left `interleave`/`deinterleave` on the generic default for every
  backend, and `reverse` native only on AVX2. AVX-512 can express all three in
  one instruction (`_mm512_permutexvar_ps` for reverse, `_mm512_permutex2var_ps`
  for the two-vector permutes); NEON needs `vrev` plus a half swap for reverse
  and `vzip`/`vuzp` for the pair ops, which do match flat semantics at 128-bit
  width. AVX2 flat interleave needs `unpack` plus `permute2f128` because
  `unpack` is per-128-bit-half. Not shipped in HS-424 because the index math is
  unverifiable on the developer host, which reports avx512f=false and is not
  aarch64, and untested permute index math silently returns plausible-but-wrong
  lanes. That reasoning was right about the risk and wrong about the remedy:
  the verification exists in CI, so the overrides are written and pushed rather
  than deferred. Acceptance: the existing HS-424 differential and round-trip
  tests pass unchanged against each native override on the aarch64 and SDE
  jobs, plus a benchmark showing the override beats the store/permute/load
  default.
  The hosted aarch64 comparison ran in PR #37's exact source-head workflow
  (run `31694336159`). `reverse_f32` and `reverse_f64` were statistically
  unchanged against the generic default, so both NEON overrides were deleted.
  Large `interleave_f32` and `deinterleave_f32` improved 1.27% and 1.40%
  respectively; their native overrides remain. The smaller rows were within
  Criterion's noise threshold. AVX-512 performance remains open under HS-429
  because SDE is semantic evidence only.

- [ ] [major] **HS-425 — `TargetId` omits the SVE backend.** `SveArch` is a
  first-class emulated backend exercised throughout the test suite, but
  `TargetId` enumerates only Scalar/Avx2/Avx512/Neon. The public forced-dispatch
  token API therefore cannot reach a backend the workspace ships, leaving a hole
  in the cross-target conformance matrix. Acceptance: `TargetId::Sve` routed
  through `dispatch_view_to`/`dispatch_view_mut_to` with conformance coverage,
  and `dispatch_view` auto-selection left untouched — an emulated backend must
  stay explicitly requested, never auto-selected.
  Reclassified from [patch] on inspection: `TargetId` and `DispatchedView` are
  both public enums without `#[non_exhaustive]`, so adding a variant breaks
  every downstream exhaustive match. The item therefore needs an ADR covering
  the variant addition, whether to apply `#[non_exhaustive]` to both enums in
  the same break so this never recurs, and the pre-1.0 minor-bump migration
  note. Precondition: that ADR drafted and the version decision made.

- [x] [patch] **HS-426 — ADR index hygiene.** `docs/adr/` carried two ADRs
  numbered 007, eight of eleven with no `## Status` section (the generated index
  rendered them `—`), and `Approved` rather than the canonical `Accepted` on the
  three that had one. The later duplicate (`007-bitboard-kernel-safe-surface`,
  added 2026-07-23) renumbered to 011 with its CHANGELOG and backlog references
  updated; every ADR now carries a canonical status, and the index is
  regenerated. All eleven are `Accepted`: the two feasibility studies (006 SSE2,
  007 SME) record decisions *not* to build, which is an accepted decision, not a
  rejected proposal.

- [x] [patch] **HS-420 — mutable generic view tails.** Route the final
  `SimdView::transform_in_place` partial vector through initialized local
  operand/result buffers and the provider's generic `ElementOp` vector seam.
  Only live result lanes are copied back, so Add/Sub/Mul/Div and future sealed
  operations share one bounds-safe tail implementation. Forced emulated-SVE
  coverage verifies an odd non-dyadic length.

- [x] [arch] **HS-421 — native AVX-512 BF16 tile dispatch.** Add the exact
  `avx512bf16` capability SSOT and route the existing `Bf16 × Bf16 → F32` tile
  provider through native `DPBF16PS` when available, retaining the
  AVX-512F/BW/VL conversion/FMA fallback on non-BF16 AVX-512 hosts. Native
  coverage uses a nonzero `C` accumulation oracle and remains capability-gated
  on ordinary hosts.

- [x] [patch] **HS-419 — pairwise reduction tails.** Route the final partial
  `SimdView::zip_reduce` vector through two initialized provider-local buffers
  and the generic masked reduction seam. `Dot` now avoids its element-at-a-time
  cleanup while preserving the full-width masked-memory contract; `Product` and
  future non-opted-in operations retain their scalar pairwise contract. Forced
  emulated-SVE coverage uses non-dyadic f32 inputs with a reassociation tolerance.

- [x] [patch] **HS-418 — dense dot-product tails.** Route the final partial
  `SimdView::dot` lanes through initialized provider-local buffers and the
  existing masked-FMA seam. The full-width masked-memory contract remains
  valid for every backend, only live lanes contribute to the final reduction,
  and odd non-dyadic f32 coverage records the expected fused-rounding tolerance.

- [x] [patch] **HS-417 — transposed GEMV column tails.** Route the final
  `gemv_transpose` partial columns through initialized provider-local lane
  buffers and the existing masked-FMA seam. The local buffers preserve the
  full-width masked-memory contract for scalar, AVX2, AVX-512, NEON, and the
  emulated SVE backend; only live tail elements are copied back. Non-dyadic f32
  coverage uses the documented tolerance for fused-operation rounding. The
  operation remains a single Hermes provider implementation; no consumer-local
  SIMD copy was introduced.

- [x] [patch] **HS-416 — generic reduction and view tails.** Route generic
  `Sum`/`Min`/`Max` final partial vectors through the provider-owned masked
  reduction seam, and consolidate `SimdView::sum` onto `reduce(Sum)` so there
  is one reduction implementation. `masked_add`, `masked_mul`,
  `masked_fmadd`, `elementwise_mul`, and generic `zip_into` now use initialized
  local lane buffers plus a leading live-tail mask instead of element-at-a-time
  tail loops. The operation preserves the existing Eunomia min/max NaN and
  signed-zero contract; floating sums use the established SIMD grouping
  envelope. Verification: scalar-contract and odd-length view differential
  tests, warning-denied core/package Clippy, Nextest, rustfmt, and diff checks.

- [x] [patch] **HS-415 — masked popcount tails.** Route the final partial
  vectors in `reduce_popcount` and the shared binary `reduce_popcount_op` through
  `SimdKernel::masked_sum_reduce`. Source lanes are copied into  initialized
  provider-local buffers before full-width loads, and each masked tail count is
  exact; the existing whole-reduction accumulator contract is unchanged.  The increment is limited to popcount reductions; generic sum/min/max and other view tails are covered by HS-416. Verification: multi-width integer differential tests, warning-denied core/package Clippy, Nextest, rustfmt, and diff checks.

- [x] [patch] **HS-414 — masked absolute-reduction tails.** Route the final
  partial vector for `AbsSum` and `AbsMax` through the generic provider-owned
  masked reduction seam. The reduction strategy applies its transform before
  merging inactive lanes with its neutral identity, while the view copies only
  live elements into an initialized local buffer.  The increment is limited to absolute reductions; generic sum/min/max and broader views are covered by HS-416, while popcount and unrelated hot kernels remain separate follow-ups. Verification: f32/f64 odd-length value

  regressions, warning-denied core/package Clippy, Nextest, rustfmt, and diff
  checks.

- [x] [patch] **HS-413 — masked row-update tails.** Route the final partial
  vectors in `axpy_rows` and `axpy_rows_batch` through Hermes' provider-owned
  `masked_fmadd` seam. The helpers copy only live elements into fully initialized
  local lane buffers, preserving the AVX2 blend-based bounds proof; the batched
  path retains its existing depth accumulation order. The increment is limited
  to row updates; reductions, views, and other hot-kernel scalar tails remain
  separate follow-ups. Verification: non-dyadic f32 row and depth-batched tail
  regressions, warning-denied package Clippy, Nextest, rustfmt, and diff checks.

- [x] [patch] **HS-406 follow-up — clean-worktree package gate.** Re-run the full
  Hermes package gate after unrelated Cargo.lock/overlay dirt is reconciled;
  focused provider slices must not claim this gate from a dirty worktree.
  Owner: codex-session (claimed + delivered 2026-08-11); Cargo.lock overlay churn
  restored to origin before the run. Evidence (rustc 1.97.0, x86_64, shared
  `D:\atlas\target`): `cargo fmt --check` clean; `cargo clippy --workspace
  --all-targets -- -D warnings` clean; `cargo nextest run --workspace` 443/443
  within the committed 30s/60s budget; doctests pass (18 + 4, ignores excluded);
  `cargo build --examples --workspace` clean; `cargo doc --no-deps` clean under
  `RUSTDOCFLAGS=-D warnings`; `cargo check --workspace --no-default-features`
  clean. Benchmark-budget, miri, and aarch64 jobs are CI-only (host lacks the
  runners); they run on the merged branch's push.


Strategic roadmap. Triage order: correctness → architecture → tests → docs → PM.
Tags: `[patch]` / `[minor]` / `[major]` / `[arch]` per SemVer change class.
Tactical breakdown of the active items lives in [checklist.md](checklist.md).
External gap findings live in [gap_audit.md](gap_audit.md).

- [x] [patch] **HS-412 — masked fused AXPY-mul tail boundary.** Route the final
  partial `axpy_mul` vector through Hermes' provider-owned `masked_fmadd` seam
  after register scaling, using initialized local lane buffers so blend-based
  backends never read beyond the live slice. The increment is limited to
  `axpy_mul`; row updates, reductions, views, and other hot-kernel scalar tails
  remain separate follow-ups until each has its own bounds proof and
  value-semantic coverage. Verification: focused f32/f64 tail tests including
  fused-operation order, warning-denied package Clippy, Nextest, rustfmt, and
  diff checks.

- [x] [patch] **HS-411 — masked scale tail boundary.** Route the final partial
  in-place `scale` vector through Hermes' provider-owned `masked_mul` seam while
  using initialized local lane buffers so blend-based backends never read beyond
  the live slice. The increment is limited to `scale`; reductions, views,
  `axpy_mul`, row updates, and other hot-kernel scalar tails remain separate
  follow-ups until each has its own bounds proof and value-semantic coverage.
  Verification: focused f32/f64 tail tests, warning-denied package Clippy,
  Nextest, rustfmt, and diff checks.

- [x] [patch] **HS-410 — masked AXPY tail boundary.** Route the final partial
  `axpy` vector through Hermes' provider-owned `masked_fmadd` seam while using
  initialized local lane buffers so blend-based backends never perform a
  full-width load beyond the live slice. The increment is limited to `axpy`;
  `axpy_mul`, row, reduction, and other hot-kernel scalar tails remain separate
  follow-ups until each has its own bounds proof and value-semantic coverage.
  Verification: focused f32/f64 tail tests plus an f32 fused-operation-order
  regression, warning-denied package Clippy, Nextest, rustfmt, and diff checks.

- [x] [minor] **HS-409 — fused ternary AXPY provider facade.** Add the
  Hermes-owned `axpy_mul` public operation for `out[i] += alpha * a[i] * b[i]`
  without a temporary. Reuse the existing `SimdKernel::mul`/`fmadd` seam and
  runtime-dispatch ladder; keep length validation and scalar-tail semantics in
  the provider. The capability is now available for Kwavers' documented
  `c += multiplier * a * b` residual, but downstream adoption remains a
  separate consumer increment until its tree is free and its focused gates pass.

- [x] [patch] **HS-REL-001 — crates.io publication.** Make the five reusable
  workspace packages independently packageable, preserve the benchmark and
  example harnesses as non-publishable, and publish in dependency order through
  the repository trusted-publishing workflow.

- [x] [patch] **HERMES-MNEMOSYNE-PACKAGE-1 — restore Mnemosyne resolution.**
  Bind the existing Rust crate alias to package `mnemosyne-memory` 0.6.0,
  refresh the lockfile, and pass the focused core check.

- [x] [patch] **HERMES-THEMIS-PACKAGE-1 — restore Themis resolution.** Owner:
  Codex on `codex/hermes-themis-package`. Bind the existing Rust crate alias to
  upstream package `themis-topology` 0.10.1; refresh the lockfile; pass focused
  checks; merge before dependent Hephaestus provider CI is retried.

- [x] [patch] **HS-407 — no `&mut [T]` spans uninitialized elements.**
  The `cow` constructors allocate with `with_capacity`, call `set_len`, and hand
  the buffer to a filler as `&mut [T]` while its tail is still unwritten
  (`extensions.rs` map/`splat_fill`/`gather`/`prefix_scan`, `unary.rs`,
  `combinators.rs`). Every element is initialized before anything reads it, so
  no wrong value is observable, but forming a `&mut [T]` over uninitialized
  elements is not a reference the language permits, and `AlignedVec` allocates
  with `alloc`, not `alloc_zeroed`. Fix by exposing a `spare_capacity_mut`-style
  `&mut [MaybeUninit<T>]` on `AlignedVec` and filling through it, keeping the
  zero-fill-free property the perf budget depends on. Acceptance: no `&mut [T]`
  spans uninitialized elements, miri covers the constructors, and the
  `dense`/`cow` benchmarks show no regression.
  Delivered: the four pointer-writing constructors (`map_cow`, `fma_cow`,
  `splat_fill`, `broadcast_op`) write their tail through the same raw pointer as
  their vector body and raise the length last, which removes the bounds checks
  the slice tail carried; `gather` and `prefix_scan` feed a view routine that
  needs an initialized slice, so they use the new
  `AlignedVec::with_capacity_zeroed`. `map_unary` was a duplicate of `map_cow`
  and now delegates to it. Miri covers every constructor across lengths
  straddling the vector body. No cow-surface benchmark exists to measure the
  zeroing pass — filed as HS-408.

- [x] [patch] **HS-405 — safe code could execute an unsupported ISA.** A
  `SimdView`, `SparseView`, or owned `SimdCow` could be built for any `Arch`
  marker regardless of host support, after which every operation invoked
  `#[target_feature]`-gated kernels — undefined behavior, reproduced as a hard
  `SIGILL` from a program containing no `unsafe`. Delivered: construction is now
  the checkpoint (`SimdView::new`/`new_mut` return `None`; the sparse and owned
  copy-on-write constructors assert), so possessing one of these values proves
  its kernels are callable. Runtime dispatch was never affected. Covered by
  tests asserting availability tracks the platform probe; no measured benchmark
  change.

- [x] [patch] **HS-408 — benchmark the copy-on-write surface.** No Criterion
  target covers `map_cow`, the scalar-broadcast ops, `splat_fill`, `gather`, or
  `prefix_scan`, so HS-407's claim that the pointer-tail rewrite adds no work
  rests on reasoning (unchecked stores replacing bounds-checked ones) rather
  than measurement, and the zeroing pass `gather`/`prefix_scan` now pay is
  unquantified. Add a `cow` group to the dense suite within the committed 60s
  smoke and 300s full budgets. Acceptance: baselines stored for each op, the
  zeroing cost quantified, and a decision recorded on whether it justifies
  teaching the view routines to fill `&mut [MaybeUninit<T>]`.
  Delivered: a `cow_f32` group (map_cow, mul_scalar_cow, splat_fill, fma_cow,
  gather, prefix_scan) at the four dense sizes measured the zeroing pass at
  12-59% on gather/prefix_scan, so the view gained `gather_into_uninit` /
  `prefix_scan_into_uninit` filling `AlignedVec::spare_capacity_mut`; those
  constructors now skip the zero-fill (-5% to -31% vs the zeroed version) and
  `with_capacity_zeroed` is removed. Miri covers the uninit fill.

- [x] [patch] **HS-406 — per-site `SAFETY` comments for pointer obligations.**
  Progress: `bitboard.rs` is closed — auditing it found the `unsafe` unjustified
  rather than undocumented, so `BitBoardKernel` became a safe trait (ADR 011)
  and the module went from seven blocks to two, both documented. The six `cow`
  modules now carry module-level `# Safety` sections plus per-site comments on
  their `with_capacity`/`set_len` buffers. Remaining:
  With HS-405 making the target-feature obligation an enforced invariant, the
  six arch-generic modules state it once in a module-level `# Safety` section.
  What remains is the *site-specific* half: the raw-pointer arithmetic in
  `view/reduce.rs` (66 blocks), `sparse/spmv.rs` (69), `sparse/ops.rs` (35),
  `tiling/` (65), and `view/vector_reg.rs` (46) still needs per-block comments
  stating the bounds and provenance argument. Scope: one module per increment,
  pointer-manipulating modules first (`bitboard.rs`, `cow/`). Acceptance: every
  `unsafe` block in the module carries an obligation-specific comment, or is
  removed because the invariant makes it unnecessary; warning-denied Clippy and
  Nextest stay green.
  Progress: `bitboard.rs` and `cow/` closed earlier; `sparse/spmv.rs` now done —
  its ~30 single-call unsafe blocks consolidated to 9, each documented, a missing
  `# Safety` doc added to `sellp_spmv_vectorized`, and a miri differential test
  added over all four formats. Remaining: `view/reduce.rs`, `sparse/ops.rs`,
  `tiling/`, `view/vector_reg.rs`.
  `sparse/ops.rs` now done too — its 35 undocumented unsafe blocks consolidated
  to 7 documented ones, and the audit found a reachable OOB in the CSR
  `elementwise_mul_dense` gather (unvalidated view, no dense-length guard; miri
  confirmed the UB) now fixed with validate + length asserts. Remaining:
  `view/reduce.rs`, `tiling/`, `view/vector_reg.rs`.
  `view/vector_reg.rs` now done — module `# Safety` section added, the six
  `pub unsafe fn` docs corrected to state the target-feature obligation, and
  per-site comments added to the MaybeUninit and lane-guard blocks (6 to 23
  SAFETY comments); the code was already well-guarded, so this is
  documentation-only. `view/reduce.rs` now done — its 66 fragmented unsafe blocks
  consolidated to 35 documented ones (behavior-preserving code motion, verified
  codegen-neutral against benchmark noise, miri-covered). `tiling/` now done —
  its four kernels gain module `# Safety` sections, 65 unsafe blocks consolidate
  to 32 documented ones, and a miri differential test covers all four (audit
  found no defect; the kernels already validate dims with overflow rejection).
  **HS-406 complete** — the whole `hermes-simd-core` unsafe surface is now
  documented (module invariants + per-site obligations).
  Verified 2026-08-12: full-source scan of all 48 unsafe-bearing files found no
  undocumented `unsafe {` (flagged sites were scanner false positives — SAFETY
  comments sit above the enclosing closures/functions, e.g. `view/reduce.rs:64`,
  `tiling/gemv.rs:117`); CI miri job green on main (run 31546997718), clippy
  `-D warnings` green in the same run.

- [x] [patch] **HS-404 — `cmp_ne` NaN semantics diverged across backends.** The
  trait default returns all-ones for `NaN != NaN` (Rust `!=` is true), while the
  AVX2 and AVX-512 backends use the ordered `_CMP_NEQ_OQ` predicate, which
  returns zero. A caller comparing NaN-bearing data therefore gets
  backend-dependent results — the same defect class HS-403 removed from
  `argmin`/`argmax`. Found while vectorizing the extremum scan, which avoids the
  divergence by testing `cmp_eq(v, v)` instead (false for NaN under both the
  ordered hardware predicates and the scalar default). Decide the intended
  contract — unordered `_CMP_NEQ_UQ` to match the scalar default, or an ordered
  contract documented on the trait and mirrored by the default — apply it to
  every backend, and pin it with a cross-backend property test beside
  `check_vector_to_mask_matches_cmp`. Acceptance: one documented NaN contract,
  differential scalar-versus-native coverage, warning-denied Clippy and Nextest.
  Delivered: the x86 backends adopt the unordered `_CMP_NEQ_UQ`, the exact
  complement of `cmp_eq`'s `_CMP_EQ_OQ`, so `cmp_ne` is the lane-wise negation
  of `cmp_eq` on every backend. Tracing the shared predicate also exposed
  AVX-512 `blend` testing its mask against zero — rejecting every active lane,
  whose `ALL_ONES` pattern is a NaN — now fixed to extract the sign bit through
  `vector_to_mask` per its documented contract. Both are pinned by cross-backend
  property tests; the `cmp_ne` one reproduces the defect on AVX2 hardware.

- [x] [patch] **HS-403 — deterministic extrema and benchmark budgets.** Reject
  NaN-containing `argmin`/`argmax` inputs, preserve
  the first slice element's signed-zero representation, and exercise every
  workspace Criterion binary under a committed 60-second smoke budget and run
  the changed canonical dense and SIMD instruments under 300-second full-run
  budgets. The first hosted smoke exposed an invalid signed-byte ZMM instruction
  in AVX-512 VNNI dispatch; replace it with exact `VPDPBUSD` bias correction.
  Acceptance: scalar/runtime-dispatch value tests, warning-denied Clippy,
  Nextest, and exact-head hosted CI pass.

- [x] [patch] **HS-402 — delivered 2026-07-19 in PR #10.** Regenerate the Hermes provider lock
  against merged Eunomia 0.6 after Eunomia retired its production raw-half
  trait surface. Acceptance: one Eunomia 0.6 identity, no normal `half` edge
  introduced by Eunomia, and the full Hermes verification gate remains green.
  Atlas gitlink publication is tracked by ATLAS-INTEGRATION-027.

- [x] [patch] Close standalone Git provider resolution on the
  Mnemosyne, Eunomia, and Themis default branches; remove local patch
  overrides and prove a single locked provider identity for `hermes-simd`.
  Delivered 2026-07-15: locked graph inspection plus package-scoped fmt,
  warning-denied Clippy, nextest, rustdoc, and `cargo deny check` gates pass.
  CI removes its obsolete sibling checkouts and allowlists the reviewed provider
  Git sources.

## Delivered (2026-07-18)

- [x] [arch] **HS-401 — delivered 2026-07-18 in PR #8.** Takeover of
  `feat/eunomia-f16-migration`. Replace raw `half::f16`/`half::bf16` across
  Hermes' scalar, ISA, tiled-matrix, tests, and benchmark contracts with
  Eunomia's native `F16`/`Bf16`; remove every direct Hermes `half` dependency.
  Scope: workspace/member manifests, reduced-precision source/tests/benches,
  lockfile resolution, and PM artifacts. Acceptance: zero raw-half source or
  direct-dependency residue, one Eunomia identity, warning-denied all-feature
  Clippy, full Nextest, doctests, rustdoc, no-default-feature checks, and an
  explicit semver classification. Local gates and PR #8's x86, AArch64
  cross-compile, native AArch64 NEON, Miri, cargo-deny, and CodeRabbit gates
  pass. `half` remains transitive through Eunomia's temporary raw-trait surface
  and Criterion's Ciborium dependency.

## Open

- [minor] Re-enable AMX auto-dispatch only after adding a stable,
  permission-aware probe that verifies hardware feature bits, XCR0 OS state,
  and Linux XTILEDATA process permission before reporting support. Acceptance:
  AMX GEMM dispatches and matches the scalar reference on a Sapphire-Rapids
  Linux runner.
  Probe delivered: `crates/hermes-simd-intrinsics/src/x86_64/amx/probe.rs` is
  the capability SSOT — CPUID leaf 7 EDX bits 24/22/25, `OSXSAVE`, `XCR0`
  bits 17/18 via `XGETBV`, then per-OS permission: Linux
  `arch_prctl` GET_XCOMP_SUPP -> REQ_XCOMP_PERM -> GET_XCOMP_PERM verify;
  Windows `GetEnabledXStateFeatures` -> `EnableProcessOptionalXStateFeatures`
  -> `GetThreadEnabledXStateFeatures` verify, both resolved by `GetProcAddress`
  because they are absent before Windows 11 / Server 2022. Any other OS
  refuses. Both hardcoded `false` sites are gone (`amx/mod.rs`, and
  `hermes-simd/src/cpu.rs`, which now delegates). Probing requests permission,
  which is a one-time process-wide XSAVE-area enlargement; the result caches.
  Acceptance is NOT met: the probe returns false on every machine available,
  and it cannot be satisfied under Intel SDE. SDE emulates the tile
  instructions, CPUID, and `XGETBV`, but `arch_prctl` is a real syscall passed
  through to the host kernel, which returns `EOPNOTSUPP` for XTILEDATA because
  the runner silicon has no AMX. A correct probe therefore refuses under
  emulation, so `amx` stays out of `test-avx512-sde`'s
  `HERMES_EXPECTED_TARGETS` (the job's step comment now explains this). What
  remains is exactly HS-429's hardware: Sapphire-Rapids-or-later silicon on
  which the probe returns true, the GEMM dispatches, and its result is
  differentially checked against `scalar/tiling.rs`.

- [ ] [major] **HS-434 — `const TILE: u8` for the AMX raw tile wrappers.**
  `raw::tilezero`/`tileloadd`/`tilestored` dispatch an 8-arm runtime `match`
  with an `unreachable!()` inside the tile loop, and `tdpbf16ps`/`tdpbssd`
  match an 11-entry whitelist of `(dst, src1, src2)` triples — a latent defect,
  since any unlisted-but-valid triple panics instead of executing. Every one of
  the ~100 call sites in `amx/bf16.rs` and `amx/int8.rs` passes a literal, so
  `const TILE: u8` generic parameters remove the branch and the panic entirely
  (`asm!` substitutes a `const` operand textually, so `"tilezero tmm{n}"` with
  `n = const TILE` assembles correctly). Deferred rather than done with the
  probe for two reasons: it breaks the `raw` public API ([major], and
  `cargo-semver-checks` gates that), and its benefit is a branch removed from a
  loop that executes on no available machine, so it cannot be measured — this
  belongs in the same increment as the hardware validation above, where a
  criterion baseline is possible. Acceptance: one `asm!` block per wrapper, no
  `unreachable!()` in `amx/mod.rs`, all call sites converted, and a measured
  before/after on AMX silicon.

- [ ] [patch] **HS-435 — `# Safety` sections for the AMX raw wrappers.**
  The eight `pub unsafe fn`s in `amx/mod.rs`'s `raw` module carry `///`
  summaries but no `# Safety` section, so `clippy::missing_safety_doc` fires on
  each. Their real preconditions are now stated (an `AmxSession` must be
  active, hence the tile configuration loaded, and `probe::has_amx_tile()` must
  hold), plus the pointer/stride validity and 64-byte alignment obligations for
  the load/store pair. Pre-existing, and unrelated to the probe; folding it in
  would have collided with the in-flight lint-floor ratchet.

## Delivered (2026-06-11)

- [x] [minor] (2026-07-05) Sparse `Validated` typestate follow-up. CSR,
  SELL-p, and Blocked-COO SpMV now require `ValidatedData` storage; malformed
  sparse structures fail at validated view/COW/public-dispatch construction, and
  hot `SparseSpMv` impls run only on `SparseView<Validated<_>>` without per-call
  structural scans. Regression/property coverage checks construction-time
  rejection plus value-semantic SpMV for all three formats. Evidence tier:
  type-level typestate + property tests; local compile/nextest pending shared
  Cargo target lock clearance.
- [x] [minor] (2026-07-05) Safe-code ISA fault hardening. `SimdArch` now owns
  runtime-support probing for safe wrappers and forced dispatch; safe AVX-512
  vector constructors/checked slice wrappers return `SimdError::UnsupportedTarget`
  on unsupported hosts before executing target-feature code, while infallible
  conveniences panic before ISA execution. `AmxSession::new` and
  `AmxBatchSession::begin` return `AmxSessionError::UnsupportedTarget` before
  `ldtilecfg`; `release` guards `tilerelease`. Evidence: unsupported-host
  regression coverage plus focused package verification.
- [x] [patch] (2026-07-02) AMX auto-dispatch mitigation. `hermes-simd`
  conservatively reports no AMX support until the permission-aware probe above
  exists, avoiding unstable Rust AMX feature-detection macros and preventing
  CPUID-only dispatch into AMX tile instructions. AVX-512 tile probes keep exact
  stable feature checks. Evidence: `cargo check -p hermes-simd`; `cargo clippy
  -p hermes-simd --all-targets -- -D warnings`.
- [x] [patch] (2026-06-28) `recip_sqrt` full native precision. The f64 SIMD paths
  and NEON f32 under-refined a low-bit hardware `rsqrt` seed (one Newton step),
  giving backend-dependent accuracy from ~1e-16 (scalar) to ~1.5e-5 (NEON) —
  masked by perfect-square test inputs + magic tolerances. Now ~1 ulp everywhere:
  f32 fast `rsqrt`+Newton (NEON two steps), f64 hardware `sqrt`+divide. New
  cross-backend differential test with derived bounds (`8·ε_f32`/`4·ε_f64`) over
  non-perfect-square inputs; old tests de-gamed. x86 verified locally, NEON on
  aarch64 CI. 380 tests + clippy/fmt/doc clean. See [gap_audit](gap_audit.md#resolved).
- [x] [patch] (2026-06-28) Integer `sqrt` exactness. `NumericElement::sqrt` for
  integers used a lossy `(self as f64).sqrt() as Self` roundtrip (wrong for large
  `i64`/`u64`); now exact `isqrt` with a documented negative contract. New
  value-semantic tests (large-operand regressions + `r²≤n<(r+1)²` invariant +
  negatives) over all 8 integer types — sqrt previously had zero coverage. 379
  tests + clippy/fmt clean. See [gap_audit](gap_audit.md#resolved).
- [x] [patch] (2026-06-28) Memory-safety: tiling dimension-product overflow.
  GEMV/GEMM operand-length checks used unchecked `usize` products
  (`(nrows−1)·lda+ncols`, `m·k`, …) as the sole guard before unsafe SIMD loads;
  an adversarial dim from the public dispatch API (`lda=usize::MAX`) wrapped under
  release `overflow-checks=false` → OOB read. Fixed via SSOT `tiling::dims`
  checked-span helpers (also dedups the forward/transpose span math) +
  `[profile.dev] overflow-checks=true`. Exact-variant overflow regressions on all
  three dispatchers pass in dev AND release; 377 tests + clippy/fmt/doc clean.
  See [gap_audit](gap_audit.md#resolved).
- [x] [minor] (2026-06-28) Masked-merge `SimdKernel` defaults — SIMD-capability
  monomorphization. Investigation found the kernel seam already mature (rsqrt,
  popcount, horizontal-bitwise, reductions, scans are all defaulted methods or
  ZST strategies = one generic addition each). Closed the last `required`-on-every
  -impl family: the six masked-merge methods (`masked_{load,store}_unaligned`,
  `masked_{add,mul,fmadd}`, `masked_sum_reduce`, the NumKong P1 tail-free set) now
  have scalar-emulated trait defaults (`blend(mask_to_vector(mask), …)` +
  `kernel_helpers::generic_masked_{load,store}`), removed from
  `impl_emulated_kernel!` (~66 lines, ~24 backends inherit). New backends/types
  inherit the family free. Cross-backend differential property test (Scalar/SveArch
  defaults vs AVX2/AVX-512 natives); 371 tests + clippy/fmt/doc clean. `gather`/
  `compress`/`expand` stay required (no generic index/lane-introspection primitive).
  See [gap_audit](gap_audit.md#resolved).
- [x] [patch] (2026-06-26) Audit round 5 — monomorphization + sparse defect fix.
  Fixed `spmv_bcoo` (was hardcoded to ScalarArch → SIMD BlockedCoo kernels dead;
  now runtime-dispatched, + differential SIMD-branch test). Extracted
  `axpy_rows_batch` extent validation to a non-generic `#[inline(never)]` fn
  (emitted once, not per `(T, Arch)`). 369 tests + clippy/fmt/doc clean.
  Mnemosyne page-list inner-fn dedup deferred (hot-path size/speed tradeoff
  needing measurement). See [gap_audit](gap_audit.md#audit-2026-06-26-r5).
- [x] [patch] (2026-06-26) Audit round 4 — numeric DRY, AMX safety, allocator
  contention. Collapsed signed-integer `NumericElement` impls into one macro +
  removed dead `min_scalar`/`max_scalar` overrides (~275 lines); AMX raw wrappers
  panic loudly on bad tile index + documented the AMX-availability precondition.
  Upstream Mnemosyne (`perf/segment-purge-batch-detach`): batch-detach segment
  purge/reset (one lock per node, not per segment) + `NUMA_BUCKETS` SSOT node
  arrays. 368 hermes + 211 Mnemosyne tests green; clippy/fmt/doc clean. See
  [gap_audit](gap_audit.md#audit-2026-06-26-r4).
- [x] [patch] (2026-06-26) Audit round 3 — SSOT, hierarchy, allocator retention.
  Finished the `MAX_SIMD_LANES` SSOT migration in `view/vector_reg.rs` (dead
  runtime asserts → compile-time `LANE_BOUND_CHECK`, 128→64 buffers); split
  `tensor/view.rs` into a vertical `tensor/view/{mod,rank_ops,simd_bridge}`
  hierarchy (SoC). Upstream Mnemosyne (`perf/huge-pool-byte-cap`): byte-bounded
  huge-pool retention (~16 GiB→~256 MiB/bucket) + removed a redundant per-pop
  atomic reload. 367 hermes tests + 210 Mnemosyne tests green; clippy/fmt/doc
  clean. See [gap_audit](gap_audit.md#audit-2026-06-26-r3).
- [x] [patch] (2026-06-26) Memory-efficiency cross-repo fix. Root-caused
  `AlignedVec<_, Aligned<64>>` small allocations costing ~2 MiB each (Mnemosyne
  routed `align > 16` to its huge path). Fixed upstream in Mnemosyne
  (`perf/aligned-small-alloc-tcache`: alignment-aware size-class selection) and
  removed the counterproductive hermes `adjust_layout_for_mnemosyne` 8 KiB
  padding + no-op dealloc NUMA bind. Measured 512 × 256B/64-aligned `AlignedVec`:
  **~1056 MiB → ~4 MiB** mapped. Plus O(nblocks) bounds guards on the BlockedCoo
  `spmv`/`elementwise_mul_dense` SIMD column loads (safety). See
  [gap_audit](gap_audit.md#alloc-audit-2026-06-26). 367 tests green; Mnemosyne
  98+23 tests green.
- [x] [minor] (2026-06-26) Audit sprint — safety, contention-free perf, memory.
  `NumericElement` extended to `i64`/`u8`/`u16`/`u32`/`u64` (+ first
  `hermes-numeric` tests). `MAX_SIMD_LANES` 128→64 (true max) halving fallback
  buffers, with `reduction.rs`/bitmask buffers folded onto the SSOT under the
  compile-time `LANE_BOUND_CHECK`. NUMA alloc-generation hardened
  (Relaxed→Release/Acquire + single-capture, closing a stale-cache/TOCTOU
  window). `build_index_vector` layout invariant made a `const` assert;
  `#![forbid(unsafe_code)]` on `hermes-simd-macros`; magic-table CAS ordering
  relaxed. Triplicated `SimdOps` impls collapsed to one macro (mod.rs
  1217→845); `flush_limit` deduped to a `const fn`; `axpy`/`scale` 4×-unrolled.
  367 tests + clippy `-D warnings` + fmt green.
- [x] [patch] (2026-06-24) Compile-time `LANE_COUNT <= MAX_SIMD_LANES` guard on
  the scalar-fallback `[MaybeUninit<T>; 128]` stack buffers (kernel + kernel_helpers):
  named SSOT constant + `SimdKernel::LANE_BOUND_CHECK` asserted per backend,
  replacing the unasserted/misleadingly-half-guarded magic 128. Prevents a silent
  stack overflow if a future wide backend (e.g. native SVE) uses the defaults.
  Validated by a lower-the-bound build failing AVX-512 compilation. Plus a
  rust-1.95 workspace clippy cleanup. 357 tests + clippy `-D warnings` green.
- [x] [minor] AXPY provider: `SimdOps::axpy` / dispatched `axpy` free fn —
  fused row update `out[i] += alpha * x[i]` via the `fmadd` primitive with
  scalar tail, no temporaries, length-mismatch error. Driver: leto matmul SIMD
  dispatch (its Stage C2 gate). Value tests across all tail sizes, f32/f64,
  zero-alpha identity, mismatch rejection.
- [x] [minor] Batched AXPY rows: `SimdOps::axpy_rows_batch` / dispatched
  `axpy_rows_batch` free fn — fused depth-major dense row-panel accumulation
  via one runtime-dispatched kernel, no temporaries, length-mismatch error.
  Driver: leto/coeus dense-panel accumulation. Delivered 2026-06-15 with
  repeated-`axpy_rows` differential coverage, invalid-extent tests, and
  register accumulation that stores each output lane once per call. Criterion
  coverage: `axpy_rows_batch_f32` compares the fused kernel against repeated
  public `axpy_rows` calls on depth-major row panels.
- [x] [patch] Dense/AXPY error-contract hardening: selected public dense
  facade and AXPY length-mismatch tests assert exact
  `SimdError::LengthMismatch` values instead of existence-only failures.
- [x] [patch] Select/unary error-contract hardening: select, unary-map, and
  COW FMA tests assert exact `SimdError` variants for length mismatch and
  insufficient output capacity.
- [x] [patch] Operation-family error-contract hardening: new operation,
  strategy, complex, and COW math tests assert exact `SimdError` variants for
  short outputs, length mismatch, and invalid gather indices.
- [x] [patch] COW unary invariant cleanup: `SimdCow::map_unary` now asserts
  its internally constructed output-length invariant instead of discarding the
  `SimdView::map_unary` result.
- [x] [patch] GEMM tiling rustdoc cleanup: module theorem prose now references
  private implementation details as code text instead of public intra-doc
  links.
- [x] [patch] Runtime FMA capability probe: `has_fma3` / `FmaSupport` now
  route through Rust's platform-aware runtime detector and are covered by
  host-capability tests.
- [x] [patch] GEMV rustdoc link cleanup: same-named dispatch modules and
  functions are disambiguated in public docs.
- [x] [minor] Const-generic Blocked-COO dispatch: replaced fixed public
  `spmv_bcoo4x4`/`spmv_bcoo8x8` dispatch and fixed
  `SparseView::from_blocked_coo_4x4`/`from_blocked_coo_8x8` constructors with
  one `spmv_bcoo::<T, BM, BN>` public API and the existing generic
  `from_blocked_coo` constructor. Driver: structural duplication audit.
- [x] [minor] Const-generic SELL-p dispatch: replaced fixed public
  `spmv_sellp4`/`spmv_sellp8` dispatch and fixed
  `SparseView::from_sellp4`/`from_sellp8` constructors with one
  `spmv_sellp::<T, C>` public API and the existing generic `from_sellp`
  constructor. Driver: structural duplication audit.

## Atlas in-house replacement roadmap — hermes slice [arch]

hermes is the Atlas **SIMD SSOT** (data-parallel lanes), replacing std::simd / packed_simd
and hand-rolled intrinsics. Scope boundary: hermes owns SIMD only; thread-level **MIMD**
is moirai's domain, GPU is the `hephaestus` substrate (wgpu + CUDA) via coeus/apollo. Work to make hermes the
complete SIMD substrate for leto-ops/coeus hot kernels:
- [ ] [minor] Stage C1: dedicated AVX-512 / AMX CI runners (currently self-skip on
  unsupported hosts), `no_std` feature matrix, committed criterion baselines.
  Partial delivered (2026-06-12): local AVX2 Criterion baseline refreshed with
  packed4 COW unpack and unrolled complex `mul_assign` rows; runner self-check
  covered 48 rows. Dedicated AVX-512/AMX runners remain open.
- [x] [patch] Stage C1: `SveArch` callable fallback (stub removal) — delivered
  2026-06-13 as a value-preserving 512-bit-shape emulated backend for f32/f64.
- [x] [minor] Stage C1: `SveArch` public marker + property coverage —
  delivered 2026-06-13 by re-exporting it from `hermes-simd` and adding it to
  the host-independent kernel property suite.
- [ ] [minor] Stage C1: native SVE intrinsic backend for AArch64 server targets.
  Blocked by the pinned stable Rust toolchain: `SveArch` remains a safe,
  value-semantic lane-emulated backend, while `SveArch::is_native_hardware_supported`
  reports hardware capability separately. Revisit when stable scalable SVE
  vector types are available or an explicitly approved asm/C boundary is added.
- [ ] [minor] Stage C2: expand op/dtype coverage on demand from leto-ops/coeus
  (gather/scatter variants, additional reductions/scans, complex precisions) so every
  leto/coeus CPU hot kernel has a hermes path rather than a scalar fallback.
  Delivered (2026-06-12): abs-sum (`Σ|x|`) and abs-max (`max|x|`) slice
  reductions via `AbsSum`/`AbsMax` ReductionOp ZSTs + `SimdOps::{abs_sum,
  abs_max}` dispatch. The reduce loop's unrolled head previously seeded
  accumulators with raw loads and merged partials with `accumulate` — correct
  only for transform-free ops; it now seeds through `transform_vector` and
  merges through `combine_vectors`, fixing the latent defect for every
  transform-bearing reduction (the documented SquaredSum hook included).
- [x] [patch] Document the SIMD(hermes) vs MIMD(moirai) vs GPU(hephaestus: wgpu + CUDA)
  boundary in README so consumers compose the three deliberately. Delivered
  2026-06-12: README defines Hermes as the synchronous, slice-oriented SIMD
  substrate; Moirai owns thread-level partitioning; Hephaestus owns GPU
  resource lifetimes and device-resident kernels.

## External reference audits <a id="external-reference-audits"></a>

- [x] **[patch] Highway comparison audit** (2026-06-14): audited
      `https://github.com/NikoMalik/highway.git` at
      `0984271e74db124cf5e200de542e745348eb0b9e` and recorded Hermes-native
      gaps in [gap_audit.md](gap_audit.md#highway-2026-06-14).
- [x] **[patch] NumKong comparison audit** (2026-06-17): audited
      `https://github.com/ashvardanian/NumKong` and recorded Hermes-native
      gaps in [gap_audit.md](file:///d:/atlas/repos/hermes/gap_audit.md#numkong-2026-06-17).
- [x] **[minor] Target-token forced dispatch**: add a Hermes `TargetId` and
      `dispatch_to`-style test/benchmark surface that checks CPU support before
      entering target-feature trampolines. Driver:
      [gap_audit.md#highway-2026-06-14](gap_audit.md#highway-2026-06-14).
      Delivered 2026-06-14 as `TargetId`, `dispatch_view_to`, and
      `dispatch_view_mut_to` with value-semantic host capability tests.
- [x] **[minor] Safe one-vector slice wrappers**: add bounds-checked and
      alignment-checked wrappers over `load_aligned`, `load_unaligned`,
      `store_aligned`, and `store_unaligned` for one-vector use cases,
      preserving raw-pointer kernels for hoisted hot loops. Driver:
      [gap_audit.md#highway-2026-06-14](gap_audit.md#highway-2026-06-14).
      Delivered 2026-06-14 on `Vector<T, Arch>` with exact failure tests.
- [x] **[arch] SSE2 backend feasibility ADR** (delivered 2026-06-21): evaluated a 128-bit
      x86_64 backend between Scalar and AVX2, resulting in ADR 006 recommending
      relying on compiler auto-vectorization or evaluating SSE4.1/SSSE3 as a modern baseline.
- [x] **[minor] Public dense facade cross-target matrix**: force every
      supported target available on the host and compare public dense facade
      results against Scalar for representative arithmetic, mask, reduction,
      gather, and shuffle paths. Driver:
      [gap_audit.md#highway-2026-06-14](gap_audit.md#highway-2026-06-14).
      Delivered 2026-06-15 with host-supported `TargetId` checks over sum,
      dot, elementwise arithmetic, gather, and select.
- [x] **[patch] Operation-family coverage map**: expanded the coarse Stage C2
      row into per-family entries in README and this backlog. Evidence tier:
      source audit over the current public surface and Highway reference audit;
      no performance or correctness claim is made for unimplemented families.

### Operation-family coverage map <a id="operation-family-coverage-map"></a>

Consumer admission rule: a family becomes implementation work only when an
Atlas consumer names a hot path or contract that requires it. Public APIs remain
Hermes-native, monomorphized, and backed by value-semantic tests before a row is
marked delivered.

- [x] [minor] Arithmetic: dense `sum`, `dot`, elementwise add/sub/mul/div,
  `axpy`, `axpy_rows`, `axpy_rows_batch`, sparse SpMV, and tiled GEMM/GEMV are
  present with scalar fallback and runtime dispatch.
- [x] [minor] Reductions: `sum`, `min`, `max`, `argmin`, `argmax`, `abs_sum`,
  `abs_max`, dot, masked reductions, and COW reductions are present.
- [x] [minor] Masks/select: `BitMask`, masked dense operations, `select`,
  `masked_negate`, mask round-trip property coverage, and safe target-forced
  dense conformance are present.
- [x] [minor] Memory: typestate `SimdView`, `AlignedVec`, COW promotion,
  packed4 COW unpack, safe one-vector load/store wrappers, and gather are
  present.
- [x] [minor] Shuffle/rearrange: complex adjacent-pair primitives
  (`swap_adjacent`, `dup_even`, `dup_odd`, `fmaddsub`, `fmsubadd`) and packed
  unpacking are present where consumer kernels require them.
- [x] [minor] Float-specialized: interleaved complex multiply/dot, norm,
  normalize, absolute reductions, and sqrt/abs/clamp unary strategies are
  present.
- [ ] [minor] Scatter/compress-store family: add only when an Atlas consumer
  needs indirect writes or compaction output; current delivered scope covers
  gather and mask/select, not scatter.
- [ ] [minor] Comparison predicate family: add lane-wise compare APIs only when
  a consumer needs reusable predicate outputs beyond existing min/max/select
  contracts.
- [ ] [minor] Conversion family: add vectorized widening/narrowing conversion
  APIs only when a consumer needs conversion as a public SIMD operation;
  current packed4 unpack is format-specific and owned by packed storage.
- [ ] [minor] Bitwise public facade family: add public bitwise dense APIs only
  when a consumer requires them; strategy ZSTs exist, but a broad public facade
  is not admitted without demand.
- [ ] [minor] Crypto/hash family: out of current Hermes scope unless an Atlas
  consumer requires lane-parallel primitive support; no implementation is
  claimed.

## Stage assessment (2026-06-10)

Phase: **Execution → Closure transition for 0.2.0.** Canonical trait surfaces
(`Scalar`, `SimdKernel`, `SparseFormat`/`CowFormat`, op-strategy ZSTs,
`#[runtime_dispatch]`) are defined with one-or-more concrete implementations
each; 278 workspace tests green; clippy/doc/fmt gates clean; aarch64
cross-compile verified. The dominant remaining risks are *infrastructure*
(no CI, no toolchain pin, no changelog) and *unverified hardware paths*
(AVX-512, NEON, AMX run compile-checked but not runtime-validated locally).

## P0 — Release engineering for 0.2.0 <a id="p0"></a>

- [x] **[patch] CI pipeline** (delivered 0.2.0; AVX-512 runner still open) (highest risk reducer): GitHub Actions running
      fmt → clippy `-D warnings` → `cargo test --workspace` → doc build →
      `cargo check --target aarch64-unknown-linux-gnu`. Add an ARM runner
      (or QEMU) job to runtime-validate the NEON complex/dense paths, and an
      AVX-512-capable runner if available for the `Avx512` differential tests.
- [x] **[patch] Toolchain pin + supply chain** (delivered 0.2.0; cargo-audit covered by cargo-deny in CI): `rust-toolchain.toml`, declared
      MSRV, `cargo audit` + `cargo deny check` in CI.
- [x] **[minor] 0.2.0 release** (semver-checks scoped per crate; see checklist): CHANGELOG sections (Added/Changed/Breaking —
      includes `InterleavedComplexLane` removal and per-format sparse-Cow type
      removal), `cargo-semver-checks` run, version bump committed atomically.

## P1 — Correctness hardening <a id="p1"></a>

- [x] **[patch] Reduced-precision complex coverage** (delivered 0.2.0): property/differential
      tests for `f16`/`bf16` interleaved complex kernels (currently exercised
      only via emulated defaults, asserted only for f32/f64).
- [x] **[patch] Mask/gather/compress property suite** (delivered 0.2.0): proptest invariants —
      `compress`∘`expand` identity under fixed mask, `mask_to_bitmask` ∘
      `mask_from_bitmask` round-trip, gather with permuted indices vs scalar
      reference, `leading_k_mask` boundary cases (k=0, k=LANE_COUNT, k>LANE_COUNT).
- [x] **[patch] `cargo miri` pass** (delivered post-0.2.0: core unit tests green under Miri; rkyv 0.7 tests excluded as upstream Stacked Borrows violations; CI job added) over crates containing `unsafe`
      (intrinsics excluded where Miri lacks ISA support; cover the
      view/cow/sparse pointer logic in hermes-simd-core).
- [x] **[patch] no_std + feature matrix** (delivered post-0.2.0: runtime_dispatch std-gating fixed, --no-default-features green + CI step; broader feature-combination sweep remains open): verify `--no-default-features` and
      key feature combinations build and pass.
- [x] **[minor] Fast reciprocal square root** (delivered 2026-06-21): implement `ops::RecipSqrt` (or `rsqrt`)
      with a Newton-Raphson refinement step for floating-point scalars, enabling Leto
      to avoid standard `sqrt` latency in normalized vector operations. Driver:
      [gap_audit.md](file:///d:/atlas/repos/hermes/gap_audit.md#numkong-2026-06-17).
- [x] **[arch] Masked tail-load/store API infrastructure** (delivered 2026-06-21): expose active-lane masked
      load and store helpers in `SimdKernel` and `Vector<T, Arch>`/`Mask<T, Arch>`
      for `Avx512` and `SveArch` so Leto can run tail-free kernels. Driver:
      [gap_audit.md](file:///d:/atlas/repos/hermes/gap_audit.md#numkong-2026-06-17).

## P2 — Performance & memory <a id="p2"></a>

- [x] **[minor] Criterion regression thresholds** (delivered 2026-06-12):
      `benchmarks_baseline.json` records structured Criterion point estimates;
      `run-benches --check-regressions` fails on missing baseline rows or rows
      exceeding the configured ratio threshold. The runner is split into
      cohesive modules for CLI parsing, result discovery, host metadata,
      threshold comparison, and Markdown report rendering.
- [x] **[minor] SpMV scalability sweep** (delivered 2026-06-12): bench row
      counts ∈ {1K, 10K, 100K} at structural non-zero density {0.1%, 1%, 10%}
      across CSR/SELL-p/BCOO with 1024 columns; Dense-with-mask is capped at
      10K rows because it stores full dense values and masks. Sparse module
      docs now record format-selection guidance.
- [x] **[minor] Packed4 unpack generalization** (delivered 2026-06-12):
      `Packed4CowExt` delegates to `Packable4::unpack_slice_packed`, reusing
      the hermes-numeric AVX-512/AVX2/scalar dispatcher and removing the
      facade-local Bf4/F4 hardware-unpack impl pair. Criterion now includes a
      focused packed COW unpack benchmark.
- [x] **[minor] Complex mul_assign unroll** (delivered 2026-06-12):
      `interleaved_complex_mul_assign` processes two SIMD registers per loop
      iteration before the single-register and scalar tails. Criterion
      validation on this host showed runtime improvement across 256, 1K, 4K,
      and 16K complex-pair inputs.
- [x] **[patch] Compress scratch-hoist benchmark** (delivered 2026-07-05):
      add a focused `SimdView compress` Criterion group for the public
      compaction path, covering scalar all-active and host-AVX2 all/half/quarter
      masks at 1K, 16K, and 256K elements. `run-benches --parse-only
      --write-baseline --check-regressions` refreshed the committed benchmark
      report/baseline and checked 102 Hermes rows.
- [x] **[minor] Expose popcount and horizontal reductions** (delivered 2026-06-21): add SIMD population
      count (`popcnt`) and bitwise horizontal fold/reduction primitives to the facade,
      enabling Leto/Hephaestus to implement Jaccard and Hamming distance metrics. Driver:
      [gap_audit.md](file:///d:/atlas/repos/hermes/gap_audit.md#numkong-2026-06-17).
- [x] **[minor] Sub-byte sign-extension and unpacking/widening** (delivered 2026-06-21): implement vector
      sign-extension and unpacking primitives for `Bf4`/`F4`/`I8` types to support
      quantized dot product optimizations in Leto. Driver:
      [gap_audit.md](file:///d:/atlas/repos/hermes/gap_audit.md#numkong-2026-06-17).

## P3 — Architecture & maintenance <a id="p3"></a>

- [x] **[patch] x86 VNNI asm form** (delivered post-0.2.0): factor repeated
      `vpdpbssd` inline assembly into one internal instruction macro with
      explicit target-feature contract. The portable surface remains
      `TileMatrixMultiply`/runtime dispatch; asm is not promoted to a separate
      public abstraction.
- [x] **[arch] Per-type x86 kernel dedup** (delivered 2026-06-21): evaluated build-time
      code generation vs macros for AVX2/AVX-512 duplication, resulting in
      ADR 005 recommending build-time code generation via a custom `build.rs` script.
- [x] **[patch] SVE callable fallback**: removed `unimplemented!()` SVE
      `SimdKernel` methods and routed `SveArch` f32/f64 through the existing
      lane-emulated kernel macro with value-semantic tests.
- [x] **[minor] SVE property coverage**: `hermes-simd` re-exports `SveArch`,
      and `kernel_property_tests` now exercises its mask round-trip,
      compress/expand, gather, and leading-tail invariants on every host.
- [ ] **[minor] Native SVE backend**: hardware intrinsic implementation remains
      blocked on stable `core::arch::aarch64` SVE vector types; revisit on
      toolchain updates. The delivered `SveArch` path is emulated and its
      hardware capability probe is separate from `SimdArch::is_runtime_supported`.
- [x] **[minor] Arm SME target feasibility study**: evaluate outer-product based
      tiled matrix multiplication kernels for Apple M4/M5 platforms. Driver:
      [gap_audit.md](file:///d:/atlas/repos/hermes/gap_audit.md#numkong-2026-06-17).
      Delivered 2026-06-21 as ADR 007 feasibility study.
- [x] **[minor] NUMA module status** (audited 2026-06-11): `numa.rs` IS
      integrated — `hermes-simd::dispatcher` uses Themis topology queries /
      `verify_numa_locality`, `vec` uses `NumaAllocator`, and types_tests
      cover Mnemosyne allocation plus Themis topology ownership. Finding: it reimplements platform NUMA
      detection (`GetNumaHighestNodeNumber` on Windows, sysfs on Linux) that
      **themis `CpuTopology` owns**, and its `MnemosyneNumaAllocator` names
      mnemosyne's allocation responsibility — a structural duplication across
      the stack SSOT map (themis=topology law, mnemosyne=allocation).
- [x] **[patch] Default provider feature policy**: every Hermes package
      defaults `parallel` and `mnemosyne-memory`; the default
      `MnemosyneNumaAllocator` path now uses Mnemosyne allocation instead of a
      name-only std/platform allocator branch. The broader Themis topology
      replacement below remains open.
- [x] **[arch] NUMA consolidation onto themis/mnemosyne** (delivered
      2026-06-12): `numa.rs` detection now delegates to themis —
      `current_numa_node` → `themis::try_current_numa_node` (Option-honest,
      added in themis 0.7.0), public topology facades were removed in favor of
      direct consumer use of `themis::current_processor` / process-cached
      `CpuTopology::detect()` distance tables. The duplicated libnuma /
      GetNumaHighestNodeNumber / sched_getcpu platform blocks are deleted, and
      `MnemosyneNumaAllocator` no longer owns direct `numa_alloc_onnode`,
      `numa_free`, or `VirtualAllocExNuma` allocation branches.
      Allocation already routes through mnemosyne (`MnemosyneNumaAllocator`
      with `NumaBinding`). Kept in hermes by design: `NumaAllocator` trait,
      `NumaBinding` thread-affinity RAII, and `verify_numa_locality` —
      SIMD-specific concerns the topology SSOT should not own.

## P4 — Documentation <a id="p4"></a>

- [x] **[patch] Doctest coverage**: `cargo doc` is warning-clean; extended
      runnable doctests to the complex, sparse-Cow, and tensor public surfaces.
- [x] **[patch] Runnable core examples**: converted kernel, compute, and tiling
      public Rustdoc examples from compile-only `no_run` to executable
      value-semantic doctests.
- [x] **[patch] Runnable `BitMask` examples**: converted native-mask conversion
      and active-lane iteration examples from ignored snippets to executable
      value-semantic doctests.
- [x] **[patch] `#![deny(missing_docs)]`** (delivered post-0.2.0: all six public crates) on all public crates (currently
      `warn` in hermes-simd-core).

## Completed (recent)

- [x] [minor] Generic vectorized interleaved complex kernels + runtime dispatch
      (ADR-004; commits 33ce1b8, 3aa963e).
- [x] [minor] NEON adjacent-pair primitive overrides, aarch64 compile-verified (3aa963e).
- [x] [arch] Sparse Cow consolidation → generic `SparseCow<T, F, Arch>` + `CowFormat` (3aa963e).
- [x] [patch] Native-precision histogram binning fix + regression test (8b4a796).
- [x] [patch] Vectorized in-place prefix scan, single authoritative impl (8b4a796).
- [x] [patch] Complex-kernel property tests with analytical tolerances (8b4a796).
- [x] [patch] Workspace fmt normalization; rustdoc warning cleanup (fc34e6a, 3aa963e).
