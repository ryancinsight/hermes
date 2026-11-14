# Backlog — hermes-simd

## Milestone ladder

Scope and non-goals: [ADR 026](docs/adr/026-release-scope.md). Every open
item below carries `milestone: M-fused-scaled-spmv`.

<a id="M-fused-scaled-spmv"></a>
## M-fused-scaled-spmv — Accumulate a scaled sparse product in one pass

- version: 1.0.0; needs: none.
- outcome: one runtime-dispatched `spmv_scaled_csr` call computes
  `y += alpha * A * x` for a validated CSR matrix and matches a scalar
  reference.
- exit: accumulate a scaled f32 CSR product: `cargo nextest run -p hermes-simd --no-tests=fail -E 'binary_id(=hermes-simd::sparse_tests) & test(=test_spmv_scaled_csr_matches_scalar_reference_f32)'`.
- exit: accumulate a scaled f64 CSR product: `cargo nextest run -p hermes-simd --no-tests=fail -E 'binary_id(=hermes-simd::sparse_tests) & test(=test_spmv_scaled_csr_matches_scalar_reference_f64)'`.
- exit: leave `y` unchanged at zero alpha: `cargo nextest run -p hermes-simd --no-tests=fail -E 'binary_id(=hermes-simd::sparse_tests) & test(=test_spmv_scaled_csr_zero_alpha_leaves_y_unchanged)'`.
- exit: run a scaled SpMV example that asserts its own result: `cargo run -p hermes-simd-examples --example scaled_spmv`.
- None of the three tests or the example exists at
  `632269e6f45c0212b67ac6c1f3cbb6814fd207fb`.
- standing: no open `correctness` item; the conformance scan's installed-state
  assertions pass; the SemVer gate, the publish dry run, and the README,
  example and book builds pass.
- non-goals: native SVE, AMX or AVX-512 silicon evidence as a release gate,
  read prefetch, tensor model layers, demand-gated slice facades, crypto/hash.

## HERMES-MNEMOSYNE-IDENTITY-2026-09-03 — Align the workspace memory provider with Mnemosyne 0.9 [patch] [arch] — in-progress <a id="hermes-mnemosyne-identity-2026-09-03"></a>

- **Integrator:** Codex on `build/hermes-mnemosyne-09`; **lease:** none; milestone: M-fused-scaled-spmv.
- **Outcome:** Advance Hermes' workspace Mnemosyne edge to 0.9.0 so provider
  consumers use one first-party memory source identity.
- **Acceptance:** The manifest uses git-plus-version without `rev`; the
  standalone lock resolves all eight Mnemosyne packages to one upstream
  commit; workspace check, Clippy, nextest, doctests, rustdoc, and diff checks
  pass; no conversion or compatibility layer is added.
- **ADR:** [`023`](docs/adr/023-first-party-memory-source-identity.md).
- **Evidence, 2026-10-04.** Mnemosyne 0.9.0 is selected by git-plus-version
  without `rev`; the standalone lock resolves all eight Mnemosyne packages to
  `c409d996eb0db78077ff91f6034f0c3070a7e2a7`. The PR records the locked
  workspace check, warning-denied Clippy, Nextest, doctests, rustdoc, and diff
  checks. Windows Miri cannot execute the six NUMA cases that call Windows
  affinity APIs; hosted Linux Miri remains the applicable platform gate.
- **Dependency:** Mnemosyne 0.9.0 git-plus-version requirement; **Last-update:**
  2026-10-04.

## ATLAS-HERMES-CODEGEN-SSOT-2026-08-21 — Resolve SIMD codegen source of truth [arch] [minor] — in-progress (hosted verification pending)

- milestone: M-fused-scaled-spmv; **Owner / scope:** Atlas integration, clean `origin/main` lane; the former
  `crates/hermes-simd-intrinsics/src/bin/codegen.rs`, ADR 005, ADR index,
  affected PM records, and stale ADR 013 references. Runtime kernels, f16 and
  NEON implementation work, and Atlas gitlinks are out of scope.
- **Finding:** a pinned direct `rustc +1.97.0` run rewrote all four x86 f32/f64
  files and dropped 28 shipped methods; the 1424-line generator also modeled
  neither x86 f16 nor AArch64 NEON and had no invocation or freshness gate.
- **Resolution:** delete the incomplete generator; retain the checked-in ISA
  files as canonical; revise ADR 005 in place and synchronize the index,
  ADR 013, checklist, changelog, and gap audit. The alternative of restoring
  freshness was rejected because it requires a complete shipped-surface model
  absent from the audited generator.
- **Acceptance:** no executable stale generator remains; no runtime source
  file changes; ADR/index/PM references agree; provider gates pass on the exact
  commit.
- **Evidence:** source comparison before and after the direct pinned run,
  four-file destructive diff (415 deletions before restoration), and the
  method-set comparison are recorded in ADR 005 and `gap_audit.md`.
- **Local verification:** `cargo +1.97.0 fmt --all -- --check`, workspace
  Clippy with `-D warnings`, Nextest (465/465), doctests (18 executed, all
  passing; documented ignores remain), warning-denied workspace Rustdoc, and
  workspace example builds pass. The Atlas development overlay rewrites the
  provider lock during unlocked local commands; its derived lock changes were
  discarded. Hosted locked CI remains the delivery gate.

<a id="HERMES-SPMV-SCALED-SCALAR-COVERAGE"></a>
## HERMES-SPMV-SCALED-SCALAR-COVERAGE — Test scaled CSR SpMV on every `SimdOps` scalar [patch] — todo
- priority: verification; milestone: M-fused-scaled-spmv; needs: none; scope: `crates/hermes-simd/tests/sparse_tests.rs`.
- basis: cd0dabd38e1fe02b793782794f8ecdd94a95b9eb; finding: non-gating F2 and F3 of the judge verdict on hermes#291.
- Outcome: the scaled CSR kernel is tested on all sixteen sealed `SimdOps`
  scalars (nine today; `Bf16`, `F8`, `Bf8`, `F4`, `Bf4`, `i8` and `I8` ship
  untested), and dropping any row of `scaled_csr_instantiations!` fails a test.
- Acceptance: `cargo nextest list -p hermes-simd -E 'binary_id(=hermes-simd::sparse_tests) & test(~scaled_csr)'`
  names one reference test per scalar, and
  `test_spmv_scaled_csr_instantiations_cover_every_simd_ops_scalar` fails when
  any row is removed; neither holds today.
- Next step: add a power-of-two fixture (values +-1, alpha 2, an empty row, a
  lane-length row, a tail) whose partial sums stay inside each scalar's exact
  range; replace the exclusion comment above the macro (`sparse_tests.rs:188-194`).
  A judge probe passed `[[1,1],[],[1]]`, alpha 2, on all seven.

<a id="HERMES-SPMV-PANICS-DOC"></a>
## HERMES-SPMV-PANICS-DOC — Say where the scaled SpMV overflow panics occur [patch] — todo
- priority: verification; milestone: M-fused-scaled-spmv; needs: none; scope: `crates/hermes-simd-core/src/sparse/spmv.rs`, `crates/hermes-simd/src/dispatch/ops.rs`, `crates/hermes-simd/src/dispatch/simd_ops.rs`.
- basis: cd0dabd38e1fe02b793782794f8ecdd94a95b9eb; finding: non-gating F10 of the judge verdict on hermes#291.
- Outcome: each `# Panics` overflow clause names the scalar tail, the scaling
  and the `y` update as the panic sites; a primitive-integer lane-block
  overflow does not panic, so the unconditional wording is a false contract.
- Acceptance: `git grep -n -E 'overflow checks, on( integer overflow\.)?$' -- crates`
  returns no hit (three hits today: `spmv.rs:61`, `ops.rs:277`, `simd_ops.rs:172`).
- Next step: reword each clause to "in the scalar tail, the scaling or the `y`
  update" and keep the `# Overflow` cross-reference.

<a id="HERMES-SPMV-ERROR-BOUND"></a>
## HERMES-SPMV-ERROR-BOUND — Document the scaled SpMV forward error [patch] — todo
- priority: verification; milestone: M-fused-scaled-spmv; needs: none; scope: `crates/hermes-simd-core/src/sparse/spmv.rs`, `crates/hermes-simd/tests/sparse_tests.rs`.
- basis: cd0dabd38e1fe02b793782794f8ecdd94a95b9eb; finding: non-gating F12 of the judge verdict on hermes#291.
- Outcome: the `SparseScaledSpMv` Rustdoc states a forward-error bound for
  inexact float inputs, derived from the reduction depth (lane blocks plus
  tail) and the scaling and `y` roundings; the integer-valued fixture checks none.
- Acceptance: `test_spmv_scaled_csr_forward_error_is_within_documented_bound`
  runs an inexact-input row set per float scalar against a wider reference
  and asserts the documented bound; the test does not exist today.
- Next step: locate and read the dot-product error analysis (Higham,
  Accuracy and Stability of Numerical Algorithms) before citing it, count the
  roundings in the kernel, derive the bound per backend reduction width, and
  place it beside `# Overflow`.

<a id="HERMES-SPMV-SCALED-DOCTESTS"></a>
## HERMES-SPMV-SCALED-DOCTESTS — Add doctests to the scaled SpMV entry points [patch] — todo
- priority: verification; milestone: M-fused-scaled-spmv; needs: HERMES-SPMV-DOC-RESIDUALS; scope: `crates/hermes-simd-core/src/sparse/spmv.rs`, `crates/hermes-simd/src/dispatch/ops.rs`.
- basis: cd0dabd38e1fe02b793782794f8ecdd94a95b9eb; finding: non-gating F11 of the judge verdict on hermes#291.
- Outcome: `SparseScaledSpMv::spmv_scaled` and the free
  `hermes_simd::spmv_scaled_csr` each carry a runnable doctest that asserts a
  computed `y`; neither has a code example today (`SimdOps::spmv_scaled_csr` does).
- Acceptance: `cargo test --doc -p hermes-simd-core -p hermes-simd -- spmv_scaled`
  passes and its output names a doctest for each of the two items.
- Next step: once HERMES-SPMV-DOC-RESIDUALS gives the `SimdOps::spmv_scaled_csr`
  doctest an `invariant:` `.expect` message (it reads "a valid CSR matrix" today),
  model both examples on it, each with a hand-computed expected `y`.

<a id="HERMES-SPMV-DOC-RESIDUALS"></a>
## HERMES-SPMV-DOC-RESIDUALS — Correct three scaled SpMV doc residuals [patch] — todo
- priority: verification; milestone: M-fused-scaled-spmv; needs: none; scope: `crates/hermes-simd/tests/sparse_tests.rs`, `crates/hermes-simd/src/dispatch/simd_ops.rs`, `docs/adr/026-release-scope.md`.
- basis: cd0dabd38e1fe02b793782794f8ecdd94a95b9eb; finding: non-gating F8(2), F13 and P2 of the judge verdicts on hermes#291 and hermes#292.
- Outcome: (1) the `FIXTURE_ROW_LENGTHS` comment (`sparse_tests.rs:56-58`)
  states that the lane decomposition of 17, 85 and 259 holds at 16 lanes
  only and that every loop is reached for power-of-two lane counts 2..64; (2)
  the `SimdOps::spmv_scaled_csr` doctest uses an `invariant:` `.expect`
  message; (3) the ADR 026 revision note cites hermes#291 and hermes#292.
- Acceptance: `git grep -n -E 'every lane count|a valid CSR matrix' -- crates/hermes-simd/tests/sparse_tests.rs crates/hermes-simd/src/dispatch/simd_ops.rs`
  returns no hit (two today), and the ADR 026 revision note names both PRs.
- Next step: edit the three sites.

<a id="HERMES-SPLIT-OVERSIZE-SPARSE-FILES"></a>
## HERMES-SPLIT-OVERSIZE-SPARSE-FILES — Bring `dispatch/ops.rs` and `sparse_tests.rs` under 500 lines [patch] — todo
- priority: tightening; milestone: M-fused-scaled-spmv; needs: HERMES-SPMV-SCALED-SCALAR-COVERAGE, HERMES-SPMV-PANICS-DOC, HERMES-SPMV-ERROR-BOUND, HERMES-SPMV-SCALED-DOCTESTS, HERMES-SPMV-DOC-RESIDUALS; scope: `crates/hermes-simd/src/dispatch/ops.rs`, `crates/hermes-simd/tests/sparse_tests.rs`.
- basis: cd0dabd38e1fe02b793782794f8ecdd94a95b9eb; finding: non-gating F14 of the judge verdict on hermes#291.
- Outcome: `dispatch/ops.rs` (586 lines) and `sparse_tests.rs` (1369) each
  split into concern-named leaf modules of at most 500 lines; no behavior or
  public path changes.
- Acceptance: `wc -l` of every resulting file is at most 500 (586 and 1369
  today), and the leaf names (last `::` segment) of
  `cargo nextest list -p hermes-simd -E 'binary_id(=hermes-simd::sparse_tests)'`
  are the same set before and after, with every `test_spmv_scaled_csr_*` test
  left unqualified at the harness root so exact `test(=<name>)` filters hold.
- Next step: group `ops.rs` by operation family; make `sparse_tests` one
  harness (`tests/sparse_tests/main.rs`, binary id unchanged) with one module
  per kernel family, the scaled family at the root. It follows its needs.

<a id="HERMES-CHANGELOG-GENERATOR"></a>
## HERMES-CHANGELOG-GENERATOR — Generate the changelog from conventional commits [patch] — blocked
- priority: verification; milestone: M-fused-scaled-spmv; needs: none; scope: `CHANGELOG.md`, `.github/workflows`.
- basis: cd0dabd38e1fe02b793782794f8ecdd94a95b9eb; finding: non-gating F9 of the judge verdict on hermes#291.
- Outcome: released `CHANGELOG.md` sections are generated from conventional-commit
  subjects at release time, not hand-edited as its "Unreleased" sections are today.
- Acceptance: a committed check regenerates the changelog through the latest
  release tag and exits 0 against the committed released sections, non-zero when
  one is hand-edited; no hand-maintained Unreleased section remains (four today).
- Blocker: the generator is a stack decision, not a hermes choice; [ADR 0068](../../docs/adr/0068-publish-on-version-change.md)
  keeps release-plz as the candidate if the stack adopts generated changelogs
  and none is adopted. Re-open trigger: the stack adopts a generator.
- Next step: the decision also places the version pin (no tool manifest exists;
  `ci.yml` has one unversioned `tool:` input); then add the generator's configuration
  path to scope, convert the sections and add the check.

<a id="HERMES-RESCUE-QUEUE"></a>
## HERMES-RESCUE-QUEUE — Port the rescue PR work or prove it superseded [patch] — todo
- priority: correctness; milestone: M-fused-scaled-spmv; needs: none; scope: `crates/hermes-simd-core`, `crates/hermes-simd-intrinsics`, `crates/hermes-simd`, `crates/hermes-simd-benches`, `gap_audit.md`.
- basis: 58135279a1000b4cb815a2131ccbf931f213ab9e; finding: non-gating N5 of the judge verdict on hermes#294, F1, F3 and F5 of the judge verdict on hermes#295.
- Outcome: every hunk of `refs/pull/220/head` and `refs/pull/221/head` (taken against its merge-base with main) is ported onto current main (verified, merged) or superseded: its fix-forward port onto main is empty, shown by the main commit and the `path:line` at it that holds each changed line's effect, plus `git show --remerge-diff` for a merge port. Any other drop needs explicit user instruction.
- Acceptance: for each pull head the delivering PRs account for every hunk: a ported hunk appears in some PR's diff against main with a test or build proving it applies; a superseded hunk gives the evidence above; a user-instructed drop cites the instruction. A hunk with none of these fails; the `merge-tree` exit code decides nothing, because its result does not depend on port content.
- Evidence at basis: neither head is merged and the work survives only in the pull refs. Landed-work proof fails for both (`git merge-tree --write-tree origin/main <head>` exits 1 with conflicts).
  - `refs/pull/220/head` (`rescue/hermes-simd-consolidation-batch`): 8 commits, 25 files (+965/-1691); 14 files are byte-identical to main, 11 differ (SIMD wrapper/scalar/NEON/AVX2 de-duplication in `hermes-simd-intrinsics`, `cpu.rs`, `lib.rs`, `kernel/backend.rs`, lockfile).
  - `refs/pull/221/head` (`rescue/hermes-audit-20260928`): 1 commit, 22 files (+1857/-1831); none is identical to main. It flattens 9 `mod.rs` modules to `<name>.rs` and splits `dense_tests.rs`; main keeps `mod.rs` for 8 of the 9 and has not split `dense_tests.rs`. The ninth is already flattened: `scalar/mod.rs` became `scalar.rs` in 38e741037b987f57645a28129747c0f7481fc174, and its bodies moved to `scalar/float_kernel.rs` and `scalar/emulated.rs`, so that hunk conflicts on main's layout.
- Next step: restore each head with `git fetch origin pull/220/head:<branch>`, port #220's 11 differing files hunk by hunk onto main, then #221's 22; the item deletes in the PR that accounts for the last hunk.

## Legacy HS-4xx record — open items and measured limits

In full: the open measurement items, the limits they rest on, and the rejected refactors. The rest of the era is one line each below.

- [x] [minor] **HS-437 — lane scratch buffers are sized to the workspace
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
  Closure evidence: release assembly wrappers for the default `interleave`
  path at Scalar f64, emulated `SveArch` f64, and AArch64 NEON f64 show no
  stack allocation in the wrapper. x86-64 emits register moves/instructions
  directly; AArch64 emits `zip1`/`zip2` and `stp` directly. The AArch64 target
  emitted assembly before its Windows-host linker rejected the foreign
  `--eh-frame-hdr` option; no cross-target execution claim is made. The typed
  `LaneBuffer` refactor is therefore not justified: LLVM already removes the
  over-sized source arrays from these default-path frames, while
  `MAX_SIMD_LANES` remains the compile-time safety bound.

  Re-measured 2026-08-15, independently, because one of the three data points
  above does not support the conclusion it was cited for: **NEON overrides
  `interleave`/`deinterleave`** (`aarch64/neon_f32.rs`, `neon_f64.rs`), so the
  NEON f64 wrapper measures a native override, not the default body. AVX2 —
  the backend that actually takes the default path on x86-64 — was not covered.
  The conclusion holds, but on different evidence:

  | probe (release, `--emit asm`) | lanes | `sub rsp` | rsp/rbp traffic |
  |---|---:|---:|---:|
  | Scalar f32 / f64 | 4 / 2 | 0 | 0 |
  | SveArch f32 / f64 | 16 / 8 | 0 | 0 |
  | AVX2 f32 / f64 | 8 / 4 | 0 | 0 |
  | AVX2 f64, `interleave` alone | 4 | 0 | 0 |

  Each probe is an `#[inline(never)] extern "C"` wrapper so it owns its
  prologue; the AVX2 ones carry `#[target_feature(enable = "avx2,fma")]`. The
  full-chain probes run `interleave` + `deinterleave` + `reverse` +
  `swap_adjacent` — ten `[MaybeUninit<T>; 64]` declarations, up to 5 KB at f64
  if any of it materialised. None does.
  The instrument was checked rather than trusted, since a zero reading and a
  measurement of nothing look identical: `hs437_avx2_f64_interleave_only`
  emits `vaddsd` / `vunpcklpd` / `vaddpd` / `vshufpd` / `vaddsd` / `retq` —
  the four 512-byte arrays lowered to one `vunpcklpd`. `SveArch` f64, whose
  vector is an emulated array and so the hardest case for SROA, emits twelve
  register instructions and zero `(%rsp)`/`(%rbp)` operands.
  Per repository convention (see the note in `benches/permute.rs`) the probe
  was a one-off and is not committed; it is reconstructible from this record.
  Conclusion unchanged and now covering the default path on this host: the
  typed `LaneBuffer` refactor buys nothing measurable. Re-open only if a future
  backend's default-path frame shows a non-zero `sub rsp`.

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
  repository had no self-hosted runners), and AMX is unavailable there
  entirely. `test-avx512-sde` therefore ran the suite under Intel SDE emulating
  Sapphire Rapids (444/444 in 176s, ~11x native) through the cargo target
  runner, so only test binaries paid the cost. Its identification step ran under
  the emulator too, so passing was a hard assertion that SDE satisfies the
  runtime probes rather than merely not breaking. It used a dedicated
  `[profile.sde]` 300s budget; the 30s native budget was untouched. The job and
  profile remain the deterministic semantic gate; HS-429 adds best-effort
  native timing on hosted silicon.
  Known limit: SDE validates semantics, never performance — any benchmark claim
  still requires real silicon. See HS-429.

- [ ] [minor] milestone: M-fused-scaled-spmv; **HS-429 — real AVX-512/AMX silicon for performance evidence.**
  SDE gives deterministic semantic coverage but cannot support a performance
  claim, so HS-427's "override beats the default" acceptance is unsatisfiable
  under emulation. Acceptance: an AVX-512 job on real silicon whose coverage
  step asserts `scalar,avx2,avx512` without the emulator, plus a criterion
  baseline captured there. If a deterministic real-silicon source were adopted,
  the SDE job would become redundant and be deleted rather than kept alongside.
  Decided 2026-08-16: the RunsOn Flex `family=c7i` option (AWS account,
  CloudFormation stack, license key, GitHub App install — user-side
  provisioning) was rejected; billing the user's own cloud account to pin an
  instance family was the wrong cost. Instead `test-avx512-hosted` runs on the
  existing GitHub-hosted x86 pool and covers AVX-512 on a best-effort basis:
  it records the machine class, asserts `scalar,avx2` plus `avx512` only when
  the host silicon has it, and captures the permute A/B (`--save-baseline
  avx512-native` then generic-default compare) only on such hosts. Hosts
  lacking the silicon print AVX-512 as NOT COVERED in the coverage report and
  skip the benchmark loudly, so coverage never degrades silently; the SDE job
  is retained as the deterministic semantic gate, since hosted x86 is
  heterogeneous — that determinism was the premise that made SDE redundant,
  and it no longer holds. Drafted as `test-avx512-hosted` alongside the
  retained `test-avx512-sde` job; `[profile.sde]` stays. AMX admission remains
  kernel-dependent and is never asserted.

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

- [ ] [patch] milestone: M-fused-scaled-spmv; **HS-430 — measure the AVX-512 and NEON permute overrides.**
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

## Atlas in-house replacement roadmap — hermes slice [arch]

hermes is the Atlas **SIMD SSOT** (data-parallel lanes), replacing std::simd / packed_simd
and hand-rolled intrinsics. Scope boundary: hermes owns SIMD only; thread-level **MIMD**
is moirai's domain, GPU is the `hephaestus` substrate (wgpu + CUDA) via coeus/apollo. Work to make hermes the
complete SIMD substrate for leto-ops/coeus hot kernels:
- [ ] [minor] milestone: M-fused-scaled-spmv; Stage C1: dedicated AVX-512 / AMX CI runners (currently self-skip on
  unsupported hosts), `no_std` feature matrix, committed criterion baselines.
  Partial delivered (2026-06-12): local AVX2 Criterion baseline refreshed with
  packed4 COW unpack and unrolled complex `mul_assign` rows; runner self-check
  covered 48 rows. Dedicated AVX-512/AMX runners remain open.
- [x] [patch] Stage C1: `SveArch` callable fallback (stub removal) — delivered
  2026-06-13 as a value-preserving 512-bit-shape emulated backend for f32/f64.
- [x] [minor] Stage C1: `SveArch` public marker + property coverage —
  delivered 2026-06-13 by re-exporting it from `hermes-simd` and adding it to
  the host-independent kernel property suite.
- [x] [patch] Document the SIMD(hermes) vs MIMD(moirai) vs GPU(hephaestus: wgpu + CUDA)
  boundary in README so consumers compose the three deliberately. Delivered
  2026-06-12: README defines Hermes as the synchronous, slice-oriented SIMD
  substrate; Moirai owns thread-level partitioning; Hephaestus owns GPU
  resource lifetimes and device-resident kernels.

## External reference audits <a id="external-reference-audits"></a>

- **[patch] Highway comparison audit** (2026-06-14): audited `https://github.com/NikoMalik/highway.git` at … — `0984271e74db124cf5e200de542e745348eb0b9e`
- **[patch] NumKong comparison audit** (2026-06-17): audited `https://github.com/ashvardanian/NumKong` and recorded Hermes-native …
- **[minor] Target-token forced dispatch**: add a Hermes `TargetId` and `dispatch_to`-style test/benchmark surface that checks CPU …
- **[minor] Safe one-vector slice wrappers**: add bounds-checked and alignment-checked wrappers over `load_aligned` …
- **[arch] SSE2 backend feasibility ADR** (delivered 2026-06-21): evaluated a 128-bit x86_64 backend between Scalar and AVX2 …
- **[minor] Public dense facade cross-target matrix**: force every supported target available on the host and compare public dense …
- **[patch] Operation-family coverage map**: expanded the coarse Stage C2 row into per-family entries in README and this backlog. …

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

## P3 — Architecture & maintenance <a id="p3"></a>

- [patch] (2026-08-15) Reduction hierarchy cleanup: moved the multiplicative `Product` strategy from the 546-line reduction module …
- **[patch] x86 VNNI asm form** (delivered post-0.2.0): factor repeated `vpdpbssd` inline assembly into one internal instruction …
- **[arch] Per-type x86 kernel dedup** (delivered 2026-06-21; ADR 005 revised 2026-08-21): the initial build-time-generator decision …
- **[patch] SVE callable fallback**: removed `unimplemented!()` SVE `SimdKernel` methods and routed `SveArch` f32/f64 through the …
- **[minor] SVE property coverage**: `hermes-simd` re-exports `SveArch`, and `kernel_property_tests` now exercises its mask …
- **[minor] Arm SME target feasibility study**: evaluate outer-product based tiled matrix multiplication kernels for Apple M4/M5 …
- **[minor] NUMA module status** (audited 2026-06-11): `numa.rs` IS integrated — `hermes-simd::dispatcher` uses Themis topology …
- **[patch] Default provider feature policy**: every Hermes package defaults `parallel` and `mnemosyne-memory`; the default …
- **[arch] NUMA consolidation onto themis/mnemosyne** (delivered 2026-06-12): `numa.rs` detection now delegates to themis …
