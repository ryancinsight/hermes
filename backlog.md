# Backlog — hermes-simd

## HS-REDUCED-PRECISION-ELEMENTWISE-2026-09-01 — Elementwise SIMD ops for F16/Bf16 [minor] [perf] — in progress

- **Integrator:** Codex atlas-session; **branch:** `perf/hermes-bf16-elementwise`;
  **lease:** `crates/hermes-simd/src/vectorize.rs`,
  `crates/hermes-simd-intrinsics/src/x86_64/`,
  `crates/hermes-simd/tests/`; **last-update:** 2026-09-02.

- **Finding (stack triage 2026-09-01):** leto's `SimdStrategy` routes f32/f64
  through `hermes_simd::{elementwise_add, sub, mul, div, sum, dot, axpy, ...}`
  and marks `F16`/`Bf16` `impl_simd_ops_unsupported!` — not because leto
  chose scalar for them, but because this crate offers no elementwise
  entry points at those types: `impl_lane_scalar!` covers `f32, f64, F16`
  only, `Bf16` has FMA/AVX-512/AMX support markers and the tile GEMM but no
  lane scalar, and neither reduced type has an `elementwise_*` impl.
- **Why filed here:** a 3-week-old leto branch
  (`codex/leto-hermes-reduced-precision`) rewrote leto's strategy to route
  F16/Bf16 through hermes ahead of the provider; it could never compile
  against 0.7.0 and is deleted with this record in its place. The gap is
  the provider's (upstream ownership), and leto's routing is a one-macro
  change once it exists.
- **Acceptance:** `elementwise_{add,sub,mul,div}`, `sum`, `dot`, `axpy`
  accept `F16` and `Bf16` with the same generic conformance suite the f32/f64
  instantiations run, native-precision per the scalar contract (no
  widen-compute-narrow), differential-tested against the scalar path;
  leto then drops `impl_simd_ops_unsupported!` for both.
- **Non-goals:** tile GEMM (already exists for Bf16), AMX.
- **Scope correction (2026-09-02):** the F16 half was already served — `impl_lane_scalar!` covers `F16`, every backend implements `SimdKernel<F16>` (scalar, AVX2 via F16C, AVX-512, NEON), and the `SimdOps` blanket therefore provides the full elementwise/reduction/axpy/gemv/GEMM surface at F16; leto routes it in leto #146 (`LETO-F16-HERMES-ROUTING-2026-09-02`) with bitwise elementwise parity against its scalar path. What remains here is **Bf16 only**: a lane scalar plus `SimdKernel<Bf16>` on the scalar, AVX2 and AVX-512 backends (the x86 `SimdOps` blanket bounds on all three) with the same generic conformance suite and differential tests, computing each op through the exact `f32` widening and round-to-nearest-even narrowing that defines bf16 arithmetic, as the F16C-backed F16 kernels do.


## HS-DEINTERLEAVE-PAIRS-AVX2-F32-2026-09-02 — AVX2 f32 `deinterleave_pairs` pays two cross-lane permutes [patch] [perf] — rejected 2026-09-02 on measurement

- **Integrator:** Codex atlas-session; **branch:** `perf/hermes-deinterleave-pairs`;
  **lease:** none.
- **Last-update:** 2026-09-02.
- **Disposition:** rejected 2026-09-02 on direct pinned measurement. The
  origin implementation measured 644.21 ns (10 samples, interval
  `[641.05, 647.05] ns`) for `cross_lane_deinterleave_f32/hermes/4096`.
  The AVX2 f64-shaped `unpacklo/hi_pd` plus `permute4x64_pd` candidate
  measured 1.1120 µs (interval `[1.0475, 1.1785] µs`), and the shared-index
  `vpermps` candidate measured 887.13 ns (interval `[882.12, 893.84] ns`),
  which is +72.6% and +37.7% against the direct origin baseline. Both
  preserve the property-tested pair layout but lose on this host, so the
  production intrinsic remains unchanged. Re-open only with a distinct
  sequence whose same-binary pinned measurement beats the origin path.

- **Finding (apollo ADR 0045, seventh slice, 2026-09-02):** porting apollo's
  final Stockham stage (`groups == 1`) onto a `LaneKernel` costs +2.5%
  (40 ns of 1.56 µs) at f32 n = 1024 on an efficiency core, both rounds
  agreeing to 0.2%, while every other size is flat. The stage splits two
  loaded registers into even and odd complex samples; hermes'
  `Avx2::deinterleave_pairs` for f32 (`x86_64/avx2_f32.rs`) is two
  `permute2f128_ps` plus two `shuffle_ps`, where the retired intrinsic body
  used two in-lane `unpacklo/hi_pd` at SSE width per two digits — the same
  shuffle count, but the cross-lane permutes carry port-5 latency.
- **Outcome:** an even/odd complex split for AVX2 f32 whose per-register cost
  matches the unpack form (candidates: `vpermps` with a constant index
  vector, one per output, or `unpacklo/hi_pd` on the 128-bit halves followed
  by one `permute2f128`), selected by measurement; the same review for the
  f64 backend, whose `deinterleave_pairs` is one `permute2f128` pair.
- **Acceptance oracle:** the reduce/deinterleave conformance tests unchanged
  and green; apollo's pinned probe cell (efficiency core, f32 n = 1024) back
  within the 0.5% repeatability of the pre-port binary once the consumer
  lock advances; a hermes micro-benchmark of the split recorded before/after.
- **Risk / change class:** [patch] [perf]; one backend method.

## HS-HALF-INTERLEAVE-2026-09-02 — No lane movement at 128-bit-half granularity [minor] — done 2026-09-02

- **Driver (apollo ADR 0045):** the remaining intrinsic Stockham
  specialisations pack two digits per register at the f32 width — a
  register holds `(x_i k0 j, x_i k1 j, x_i k0 j+1, x_i k1 j+1)`, two
  two-sample runs from inputs 16 samples apart. Building it from two
  loaded registers is an unpack at 128-bit-half granularity
  (`(lo(a), lo(b))`, `(hi(a), hi(b))` — `vperm2f128`/`vinsertf128` on
  x86-64, `vzip1q/vzip2q` on 64-bit lanes for NEON f32). `Vector` offers
  `swap_adjacent`, `swap_pairs`, `deinterleave_pairs{,4}`, `interleave`,
  `transpose_square`, and `blend` (mask-driven), none of which expresses it
  without a mask register or a lane-granular detour.
- **Outcome:** `Vector::interleave_halves(self, other) -> (Self, Self)` (name
  per the existing family; `ComplexReg` twin) with the backend intrinsics
  above, a scalar reference, and conformance tests across every backend and
  width; documented as the operand pairing a two-digit-per-register kernel
  needs. Not speculative: the four apollo families it unblocks are named in
  apollo `ATLAS-APOLLO-ISA-FORK-2026-08-25`.
- **Acceptance oracle:** conformance green on scalar/AVX2/AVX-512 (and NEON
  where run); apollo's next slice consumes it with its own measurement.
- **Risk / change class:** [minor] (additive API).
- **Delivered 2026-09-02** by PR #138 (`Vector::interleave_halves`,
  `_mm256_permute2f128_*` / `_mm512_shuffle_f32x4`/`f64x2` / `vcombine_f32`
  / `vzip1q_f64` over a scalar-emulation default, property-tested on every
  host backend) and PR #139, which the consuming slice showed it also
  needed:
  - `splat_pair` — `ComplexReg::splat` built `[re, im, re, im, ...]` by
    interleaving two scalar broadcasts, two unpacks plus a cross-lane
    `vperm2f128` per twiddle. Every ISA has a single pair-broadcast
    instruction for it. Building three (pair) or seven (triple) twiddle
    registers per stage iteration, the interleave form cost apollo
    **+295% at f32 n = 1024**; the broadcast brings it to +4.6%.
  - `blend_halves` — the half-granular *select* beside this item's
    half-granular *gather*: each half keeps its position, so it is one
    in-lane blend where the gather needs the cross-lane permute. It leaves
    apollo's cell at +4.4%, so the residual there is not cross-lane
    traffic (apollo ADR 0045, ninth slice).

## HS-GEMM-PANEL-REUSE-2026-08-29 — Reuse bounded packed-B scratch [patch] [perf] — rejected 2026-08-29 on measurement

- **Rejected.** Retaining the panel per thread removes one allocate/free pair
  per packed call and costs **69x** in exchange. Production source is restored
  unchanged; the allocation census that produced the verdict is kept.
- **Measured** (wall clock, best of 40 blocks of 5 calls, f32, same probe built
  against each tree; `n = 256` is under the 512 KiB packing threshold so the
  packed path is not taken there and the row is a control):

  | n | call-owned | retained panel |
  |---|---|---|
  | 256 | 260 us | 261 us (control, packed path not taken) |
  | 512 | 1.81 ms | **125.6 ms** |

- **Why.** The register micro-kernel depends on the compiler proving the packed
  panel does not alias `b` or `c`; a call-owned `AlignedVec` gives that for
  free, and a pointer reached through a thread-local `RefCell` borrow does not.
  Without it the accumulators cannot stay register-resident across the `p`-loop
  (Theorem 3 in `tiling/gemm.rs`), which is the whole basis of the kernel.
- **The handover also carried a second, independent regression.** The rescued
  work had hoisted the packed loop into an `#[inline(never)]` helper -- an
  attempt to restore the non-aliasing fact through distinct `&mut` parameters.
  Measured on its own, with retention disabled, that hoist alone cost
  1.81 ms -> 118 ms: the helper does not inherit the caller's
  `#[target_feature]` frame, so the micro-kernel codegens at baseline ISA.
  Relaxing it to `#[inline]` still cost 70 ms; a generic callee does not
  reliably inherit the frame. Either change alone is disqualifying.
- **What is kept.** `packed_gemm_allocates_its_panel_once_per_call` pins the
  current one-pair-per-call behaviour, so a future attempt starts from a
  measured baseline. The rejected implementation is preserved in this branch's
  history (`eb06285`) rather than only described.
- **What would change the verdict.** An approach that keeps the panel's
  provenance unique at the kernel boundary -- a uniquely-borrowed slice threaded
  through without a thread-local indirection, verified by codegen inspection
  showing the accumulators still register-resident, not by allocation counts
  alone. Allocation count was the wrong acceptance oracle here: the work passed
  it while running 69x slower.
- **Handover.** Claimed by Codex task 01a03eb2, whose tree sat seven hours
  untouched with the work uncommitted and its own timing criterion unrun. Taken
  over, completed to a verdict, lease released.

## HS-AVX512-EVIDENCE-STANDARD-2026-08-31 — Which evidence standard governs AVX-512 work on AVX-512-less hosts [arch] — review, awaiting owner ratification

- **Status.** Adjudicated in
  [ADR 022](docs/adr/022-unrunnable-isa-evidence-standard.md), filed
  **Proposed** and **awaiting the owner's ratification**. Status is Proposed
  rather than Accepted because the ruling sets a precedent binding on another
  repository; the ADR carries a recommendation, not a survey, and the owner's
  review is the veto. Until ratified the tree's current state stands unchanged
  — the ADR ratifies the practice of PRs #98/#100 rather than altering it, so
  no code moves on ratification either way.
- **The ruling, in one line.** PR #98's disposition is binding on whether an
  un-runnable-ISA kernel may *land*; PR #94's principle is binding on what may
  be *claimed* about it. PR #94's remedy — deleting a correctness-verified
  kernel for want of timing — is superseded, because deletion destroys verified
  work and removes the only path by which the timing question is ever answered.
  A landed-but-unmeasured kernel is **provisional**: when silicon measures it
  and it loses to the generic default, it is deleted *then*, which is PR #94's
  remedy applied where the evidence exists.
- **Blocked consumer.** apollo's `ATLAS-APOLLO-WIDER-ISA-2026-08-28` is the
  reason this could not stay unruled. Under ADR 022 it unblocks now: it may
  land AVX-512 record widening on the ADR's section 1 evidence set, must make
  no speed claim under section 2, and files the section 3 measurement
  obligation. It does not wait for silicon.
- **The two precedents, both merged, mutually inconsistent.**
  - PR #94 (`93ba7ce`) *deleted* provisional AVX-512 f32/f64 transpose
    networks, on the stated ground that no controlled real-silicon timing was
    available and an unmeasured optimization is not an optimization.
  - PR #98 then *landed* the AVX-512 f64 network on symbolic verification of
    the permutation algebra alone, without overturning PR #94's ground.
    PR #100 (HS-SIMD-PERF-2026-08-28) followed PR #98 for f32.
  A future AVX-512 change can cite either precedent and be consistent with the
  board. That is the defect.
- **Why it is not merely tidiness.** It silently gates every future AVX-512
  item, including apollo's WIDER-ISA record, which needs to know before it
  starts whether symbolic verification is a sufficient landing standard or
  whether the work must wait for silicon. Left unruled, each item re-litigates
  it and the answer depends on who picks it up.
- **Development host.** Arrow Lake Core Ultra 9 285K, `avx512f: false`. Intel
  SDE emulation runs the paths for *correctness* in CI; it produces no timing
  evidence, so it does not settle the question either way.
- **What the ruling fixed**, against the four questions the item posed:
  1. *Required to land* — validated symbolic model of every intrinsic, the
     per-backend property and bit-exactness laws instantiated so `test-avx512-sde`
     runs the real encodings, codegen inspection of instruction budget, feature
     budget, and bounds elision, and a warning-denied compile. Where the ISA is
     not emulable — AMX, whose tile-data permission SDE passes to the host
     kernel — the kernel does **not** land.
  2. *May not be claimed* — any performance claim whatsoever. Permitted
     characterization is "canonical lowering, correctness-equivalent to the
     default, speed unmeasured". Adjacent runnable-path figures may be reported
     only as a cost of the default, never of the changed path.
  3. *Disposition* — ships ungated on `main`, never behind a cargo feature and
     never held out of tree, carrying the CHANGELOG limit statement, a standing
     measurement item, and provisional status.
  4. *Grandfathering* — the f64 network (#98) and f32 16x16 (#100) already meet
     the landing standard, which was written from what they supplied; they owe
     only the standing measurement item. PR #94's deleted networks are not
     restored, having never been verified to that standard.
- **On ratification.** The deliverable is the standing AVX-512 measurement item
  required by ADR 022 section 3, with `test-avx512-hosted` landing on a hosted
  runner reporting `avx512f` as its re-open trigger. Filing it before
  ratification would presume the ruling.
- **Non-goals.** Not a re-measurement item and not a request to acquire
  hardware — it is a standards ruling. The affected code is already merged and
  is not blocked on this.
- **Dependencies.** None. **Lease:** none. **Last update:** 2026-09-01.

## ATLAS-HERMES-CODEGEN-SSOT-2026-08-21 — Resolve SIMD codegen source of truth [arch] [minor] — in-progress (hosted verification pending)

- **Owner / scope:** Atlas integration, clean `origin/main` lane; the former
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

- [ ] [minor] **HS-429 — real AVX-512/AMX silicon for performance evidence.**
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
  `HERMES_EXPECTED_TARGETS` (the job's step comment now explains this).
  HS-429's `test-avx512-hosted` job may admit XTILEDATA on hosts whose kernel
  grants the permission; it is not asserted there either. What
  remains is exactly HS-429's hardware: Sapphire-Rapids-or-later silicon on
  which the probe returns true, the GEMM dispatches, and its result is
  differentially checked against `scalar/tiling.rs`.

- [x] [major] **HS-438 — `const TILE: u8` for the AMX raw tile wrappers.**
  *(Renumbered from HS-434 on 2026-08-14: that ID was already held by the
  workspace lint floor item at the top of this file. See the note under
  HS-439.)*
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
  Delivered 2026-08-16: the five wrappers are const generic
  (`tilezero::<TILE>()`, `tileloadd::<TILE>(base, stride)`,
  `tilestored::<TILE>(base, stride)`, `tdpbf16ps::<DST, SRC1, SRC2>()`,
  `tdpbssd::<DST, SRC1, SRC2>()`), each a single `asm!` with
  `TILE = const TILE` operands; the 8-arm match, 11-entry whitelist, and both
  `unreachable!()`s are deleted. All call sites in `amx/bf16.rs` (35) and
  `amx/int8.rs` (65) converted to turbofish literals in the same change.
  Verified: build + clippy `-D warnings` clean, nextest 30/30, doctests clean,
  workspace `cargo check --all-targets` green. `cargo-semver-checks --baseline-rev
  origin/main --release-type minor` fails exactly as expected — "2 major and 0
  minor checks failed" (`function_parameter_count_changed` and
  `function_requires_different_const_generic_params` on all five) — confirming
  the [major] class. The before/after measurement on AMX silicon remains the one
  unmet acceptance item; it is deferred to HS-429's hardware per this item's own
  reasoning (the loop executes on no available machine), and the x86_64 build
  (including the SDE job and the best-effort `test-avx512-hosted` job)
  assembles the rewritten instruction text — whether the AMX instructions
  themselves execute on a hosted runner depends on the kernel admitting
  XTILEDATA (the probe's condition 3 is a real `arch_prctl` permission
  syscall, not emulated), so `amx` stays out of that job's
  `HERMES_EXPECTED_TARGETS` until admission is observed.

- [x] [patch] **HS-439 — `# Safety` sections for the AMX raw wrappers.**
  The eight `pub unsafe fn`s in `amx/mod.rs`'s `raw` module carry `///`
  summaries but no `# Safety` section, so `clippy::missing_safety_doc` fired on
  each. The wrappers now state their real preconditions: AMX permission and an
  active tile configuration, valid tile indices and configured operand shapes,
  valid pointer/stride ranges for tile loads and stores, and 64-byte-aligned
  `TILECFG` storage. The targeted safety-doc lint is clean and intrinsics
  nextest passes 30/30.
  *(Renumbered from HS-435 on 2026-08-14: that ID was already held by the
  pedantic-ratchet item at the top of this file.)*

  **ID allocation.** Both AMX items above were renumbered rather than the
  lint-floor and ratchet items that shared their numbers, because those two are
  cited by `Refs:` footers in already-pushed commits (`aa61f33`, `03871b4`,
  `81502c5`) and commit messages cannot be corrected without rewriting shared
  history. Renumbering the later items keeps every existing citation resolving
  to exactly one entry.
  Before claiming a new ID, check the whole file rather than the top section —
  this file holds delivered items in dated sections further down, so the highest
  number in use is not necessarily near the top. `grep -o 'HS-[0-9]\+' backlog.md
  | sort -u | sort -t- -k2 -n | tail -1` gives it.

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

## P0 — Release engineering for 0.2.0 <a id="p0"></a>

- **[patch] CI pipeline** (delivered 0.2.0; AVX-512 runner still open) (highest risk reducer): GitHub Actions running fmt → clippy …
- **[patch] Toolchain pin + supply chain** (delivered 0.2.0; cargo-audit covered by cargo-deny in CI): `rust-toolchain.toml` …
- **[minor] 0.2.0 release** (semver-checks scoped per crate; see checklist): CHANGELOG sections (Added/Changed/Breaking — includes …

## P1 — Correctness hardening <a id="p1"></a>

- **[patch] Reduced-precision complex coverage** (delivered 0.2.0): property/differential tests for `f16`/`bf16` interleaved complex …
- **[patch] Mask/gather/compress property suite** (delivered 0.2.0): proptest invariants — `compress`∘`expand` identity under fixed …
- **[patch] `cargo miri` pass** (delivered post-0.2.0: core unit tests green under Miri; rkyv 0.7 tests excluded as upstream Stacked …
- **[patch] no_std + feature matrix** (delivered post-0.2.0: runtime_dispatch std-gating fixed, --no-default-features green + CI …
- **[minor] Fast reciprocal square root** (delivered 2026-06-21): implement `ops::RecipSqrt` (or `rsqrt`) with a Newton-Raphson …
- **[arch] Masked tail-load/store API infrastructure** (delivered 2026-06-21): expose active-lane masked load and store helpers in …

## P2 — Performance & memory <a id="p2"></a>

- **[minor] Criterion regression thresholds** (delivered 2026-06-12): `benchmarks/benchmarks_baseline.json` records structured …
- **[minor] SpMV scalability sweep** (delivered 2026-06-12): bench row counts ∈ {1K, 10K, 100K} at structural non-zero density {0.1% …
- **[minor] Packed4 unpack generalization** (delivered 2026-06-12): `Packed4CowExt` delegates to `Packable4::unpack_slice_packed` …
- **[minor] Complex mul_assign unroll** (delivered 2026-06-12): `interleaved_complex_mul_assign` processes two SIMD registers per …
- **[patch] Compress scratch-hoist benchmark** (delivered 2026-07-05): add a focused `SimdView compress` Criterion group for the …
- **[minor] Expose popcount and horizontal reductions** (delivered 2026-06-21): add SIMD population count (`popcnt`) and bitwise …
- **[minor] Sub-byte sign-extension and unpacking/widening** (delivered 2026-06-21): implement vector sign-extension and unpacking …

## P3 — Architecture & maintenance <a id="p3"></a>

- [patch] (2026-08-15) Reduction hierarchy cleanup: moved the multiplicative `Product` strategy from the 546-line reduction module …
- **[patch] x86 VNNI asm form** (delivered post-0.2.0): factor repeated `vpdpbssd` inline assembly into one internal instruction …
- **[arch] Per-type x86 kernel dedup** (delivered 2026-06-21; ADR 005 revised 2026-08-21): the initial build-time-generator decision …
- **[patch] SVE callable fallback**: removed `unimplemented!()` SVE `SimdKernel` methods and routed `SveArch` f32/f64 through the …
- **[minor] SVE property coverage**: `hermes-simd` re-exports `SveArch`, and `kernel_property_tests` now exercises its mask …
- [ ] **[minor] Native SVE backend**: hardware intrinsic implementation remains
      blocked on stable `core::arch::aarch64` SVE vector types; revisit on
      toolchain updates. The delivered `SveArch` path is emulated and its
      hardware capability probe is separate from `SimdArch::is_runtime_supported`.
- **[minor] Arm SME target feasibility study**: evaluate outer-product based tiled matrix multiplication kernels for Apple M4/M5 …
- **[minor] NUMA module status** (audited 2026-06-11): `numa.rs` IS integrated — `hermes-simd::dispatcher` uses Themis topology …
- **[patch] Default provider feature policy**: every Hermes package defaults `parallel` and `mnemosyne-memory`; the default …
- **[arch] NUMA consolidation onto themis/mnemosyne** (delivered 2026-06-12): `numa.rs` detection now delegates to themis …

## P4 — Documentation <a id="p4"></a>

- **[patch] Doctest coverage**: `cargo doc` is warning-clean; extended runnable doctests to the complex, sparse-Cow, and tensor …
- **[patch] Runnable core examples**: converted kernel, compute, and tiling public Rustdoc examples from compile-only `no_run` to …
- **[patch] Runnable `BitMask` examples**: converted native-mask conversion and active-lane iteration examples from ignored snippets …
- **[patch] `#![deny(missing_docs)]`** (delivered post-0.2.0: all six public crates) on all public crates (currently `warn` in …

## Archive — closed items

Closed items, one line each. Full prose is in git history; commit SHAs below are the entry points.

- **HS-434 — workspace lint floor**
- **HS-433 — AMX downgrade notice writes to stderr** — `f4d444b5`
- **HS-435 — pedantic ratchet** (2026-08-14)
- **HS-436 — `SimdKernel` operation-family facets** (2026-08-14)
- **HS-422 — scatter seam**
- **HS-423 — rounding primitives** — `58c31a9`, `df32296`
- **HS-424 — cross-lane permute family**
- **HS-431 — repair the panicking compress benchmark** (2026-07-07) — `2afe675`
- **HS-432 — benchmark budget job never runs on pushed work** (2026-08-12)
- **HS-425 — `TargetId` omits the SVE backend** (2026-08-16) — PR #49, `fb36e0f`, `dd4cc78`
- **HS-426 — ADR index hygiene** (2026-07-23)
- **HS-420 — mutable generic view tails**
- **HS-421 — native AVX-512 BF16 tile dispatch**
- **HS-419 — pairwise reduction tails**
- **HS-418 — dense dot-product tails**
- **HS-417 — transposed GEMV column tails**
- **HS-416 — generic reduction and view tails**
- **HS-415 — masked popcount tails**
- **HS-414 — masked absolute-reduction tails**
- **HS-413 — masked row-update tails**
- **HS-406 follow-up — clean-worktree package gate** (2026-08-11)
- **HS-412 — masked fused AXPY-mul tail boundary**
- **HS-411 — masked scale tail boundary**
- **HS-410 — masked AXPY tail boundary**
- **HS-409 — fused ternary AXPY provider facade**
- **HS-REL-001 — crates.io publication**
- **HERMES-MNEMOSYNE-PACKAGE-1 — restore Mnemosyne resolution**
- **HERMES-THEMIS-PACKAGE-1 — restore Themis resolution**
- **HS-407 — no `&mut [T]` spans uninitialized elements**
- **HS-405 — safe code could execute an unsupported ISA**
- **HS-408 — benchmark the copy-on-write surface**
- **HS-406 — per-site `SAFETY` comments for pointer obligations** (2026-08-12)
- **HS-404 — `cmp_ne` NaN semantics diverged across backends**
- **HS-403 — deterministic extrema and benchmark budgets**
- **HS-402 — delivered 2026-07-19 in PR #10** (2026-07-19) — PR #10
- **Close standalone Git provider resolution on the** (2026-07-15)
- **HS-COMPLEX-TRANSPOSE-2026-08-31** Register-resident complex square transpose [minor] [perf] (2026-09-01) — PR #111, `42a0d4c`, `9ac23fa4`
- **HS-HARDWARE-LANE-DISPATCH-2026-09-01** Exact width without portable emulation [minor] [perf] (2026-09-01) — PR #110, `141b7e1`, `363c407d`
- <a id="hs-f16c-scalar-frame-2026-08-31"></a>**HS-F16C-SCALAR-FRAME-2026-08-31** The scalar fallback stays unframed for F16C scalars [patch] [perf] (2026-08-31) — PR #108, `5050c72a`, `91cae6a`, `3c9623d`
- **HS-NO-STD-PACKED-MASK-IMPORTS-2026-08-29** Restore alloc imports [patch] (2026-08-29)
- **HS-SCALAR-FALLBACK-FRAME-2026-08-30** The scalar fallback compiled at baseline ISA [patch] [perf] (2026-08-31) — PR #107, `c4f931c`
- **HS-COMPLEXREG-ZERO-PROBE-2026-08-29** Rotations re-probed the host inside the hot loop [patch] (2026-08-29)
- **HS-SIMD-CAPABILITY-COPY-2026-08-28** The capability token is a proof, not a resource [patch] (2026-08-28)
- **HS-SPARSE-SAFETY-2026-08-27** Sparse OOB guards, F-only AVX-512, mask contract [patch] (2026-08-27) — PR #92, `50256f9`, `03d80d33`
- **HS-F16-DISPATCH-PROBE-HOIST-2026-08-27** Hoist F16 F16C probe to the dispatch boundary [minor] (2026-08-28) — PR #95, `81bc9b6`, `0115b4e`
- **HS-EXACT-LANE-DISPATCH-2026-08-27** Dispatch consumer kernels by exact lane count [minor] [arch] (2026-08-31) — `01a0253c`, `36bbbcf77f6d`, `c6f4b639`
- **HS-NATIVE-CAST-THROUGHPUT-2026-08-27** Remove supported cross-type cast stack round-trip [minor] (2026-08-27) — PR #86, PR #87, `5734b85`, `4f6a1eb`, `ff93be1`
- **HS-PACKED-MASK-SHAPE-SAFETY-2026-08-27** Enforce packed-mask and matrix shape bounds [patch] (2026-08-27) — PR #85, PR #81, `7e342cd`, `01a03eb2`, `55b918675e48`
- **HS-NATIVE-COMPARISON-MASK-2026-08-27** Remove comparison-mask stack round-trip [patch] (2026-08-27) — PR #84, `6efa67b`, `01a03eb2`, `55b918675e48`
- **HS-CI-RUNNER-CLASS-SELECTION-2026-08-27** Compile capability-gated configurations on every runner [patch] (2026-08-27) — PR #88, `07a88ec`, `6b32676`, `8a48825`
- **HS-TRANSPOSE-SQUARE-2026-08-27** In-register square-tile transpose [patch] (2026-08-27) — `d791281c`
- **HS-DENSEMASK-BITPACK-2026-08-27** Bit-pack the DenseWithMask lane mask [minor] (2026-08-27) — PR #81, `e1bd5e0`
- **HS-VECTORIZE-LARGE-KERNEL-2026-08-28** Large kernel bodies fall out of the target-feature frame [patch] (2026-08-28) — `d791281c`
- **HS-COMPLEX-REG-2026-08-27** Interleaved complex register vocabulary [minor] (2026-08-27)
- **HS-LANE-THROUGHPUT-2026-08-25** Locate the gap between the lane surface and fearless_simd [arch] (2026-08-26) — `01a0253c`, `36bbbcf77f6d`, `ae4e8efa`
- **HS-NUMA-GEN-ISOLATION-2026-08-25** Assert the property, not the global counter [patch] (2026-08-26) — `01a03eb2`, `55b918675e48`
- **HS-FEARLESS-TOKEN-2026-08-25** Consumer target-feature entry and safe FMA/permute surface [minor] (2026-08-25) — `424ce431`
- **ATLAS-HERMES-BOOK-TEST-2026-08-20** Enable executable book samples [patch] (2026-08-20) — PR #56, `932468dac5ef4abadea4bdd12d62b420a4225ba7`, `3a39ef16d679dbac9c1a479b2b9c44135e262af3`
- **ATLAS-ORPHAN-MODULES-096-HERMES** Remove orphan tensor view [patch] (2026-08-19) — `1fe438c`
- [arch] **HS-401 — delivered 2026-07-18 in PR #8.** Takeover of `feat/eunomia-f16-migration`. Replace raw `half::f16`/`half::bf16` … — PR #8
- [minor] (2026-07-05) Sparse `Validated` typestate follow-up. CSR, SELL-p, and Blocked-COO SpMV now require `ValidatedData` storage …
- [minor] (2026-07-05) Safe-code ISA fault hardening. `SimdArch` now owns runtime-support probing for safe wrappers and forced …
- [patch] (2026-07-02) AMX auto-dispatch mitigation. `hermes-simd` conservatively reports no AMX support until the permission-aware …
- [patch] (2026-06-28) `recip_sqrt` full native precision. The f64 SIMD paths and NEON f32 under-refined a low-bit hardware `rsqrt` …
- [patch] (2026-06-28) Integer `sqrt` exactness. `NumericElement::sqrt` for integers used a lossy `(self as f64).sqrt() as Self` …
- [patch] (2026-06-28) Memory-safety: tiling dimension-product overflow. GEMV/GEMM operand-length checks used unchecked `usize` …
- [minor] (2026-06-28) Masked-merge `SimdKernel` defaults — SIMD-capability monomorphization. Investigation found the kernel seam …
- [patch] (2026-06-26) Audit round 5 — monomorphization + sparse defect fix. Fixed `spmv_bcoo` (was hardcoded to ScalarArch → SIMD …
- [patch] (2026-06-26) Audit round 4 — numeric DRY, AMX safety, allocator contention. Collapsed signed-integer `NumericElement` impls …
- [patch] (2026-06-26) Audit round 3 — SSOT, hierarchy, allocator retention. Finished the `MAX_SIMD_LANES` SSOT migration in …
- [patch] (2026-06-26) Memory-efficiency cross-repo fix. Root-caused `AlignedVec<_, Aligned<64>>` small allocations costing ~2 MiB …
- [minor] (2026-06-26) Audit sprint — safety, contention-free perf, memory. `NumericElement` extended to `i64`/`u8`/`u16`/`u32`/`u64` …
- [patch] (2026-06-24) Compile-time `LANE_COUNT <= MAX_SIMD_LANES` guard on the scalar-fallback `[MaybeUninit<T>; 128]` stack buffers …
- [minor] AXPY provider: `SimdOps::axpy` / dispatched `axpy` free fn — fused row update `out[i] += alpha * x[i]` via the `fmadd` …
- [minor] Batched AXPY rows: `SimdOps::axpy_rows_batch` / dispatched `axpy_rows_batch` free fn — fused depth-major dense row-panel …
- [patch] Dense/AXPY error-contract hardening: selected public dense facade and AXPY length-mismatch tests assert exact …
- [patch] Select/unary error-contract hardening: select, unary-map, and COW FMA tests assert exact `SimdError` variants for length …
- [patch] Operation-family error-contract hardening: new operation, strategy, complex, and COW math tests assert exact `SimdError` …
- [patch] COW unary invariant cleanup: `SimdCow::map_unary` now asserts its internally constructed output-length invariant instead of …
- [patch] GEMM tiling rustdoc cleanup: module theorem prose now references private implementation details as code text instead of …
- [patch] Runtime FMA capability probe: `has_fma3` / `FmaSupport` now route through Rust's platform-aware runtime detector and are …
- [patch] GEMV rustdoc link cleanup: same-named dispatch modules and functions are disambiguated in public docs.
- [minor] Const-generic Blocked-COO dispatch: replaced fixed public `spmv_bcoo4x4`/`spmv_bcoo8x8` dispatch and fixed …
- [minor] Const-generic SELL-p dispatch: replaced fixed public `spmv_sellp4`/`spmv_sellp8` dispatch and fixed …
- [minor] Generic vectorized interleaved complex kernels + runtime dispatch (ADR-004; commits 33ce1b8, 3aa963e). — `33ce1b8`, `3aa963e`
- [minor] NEON adjacent-pair primitive overrides, aarch64 compile-verified (3aa963e). — `3aa963e`
- [arch] Sparse Cow consolidation → generic `SparseCow<T, F, Arch>` + `CowFormat` (3aa963e). — `3aa963e`
- [patch] Native-precision histogram binning fix + regression test (8b4a796). — `8b4a796`
- [patch] Vectorized in-place prefix scan, single authoritative impl (8b4a796). — `8b4a796`
- [patch] Complex-kernel property tests with analytical tolerances (8b4a796). — `8b4a796`
- [patch] Workspace fmt normalization; rustdoc warning cleanup (fc34e6a, 3aa963e). — `fc34e6a`, `3aa963e`

- **HS-PROCESSOR-MODULE-SPLIT-2026-09-02** Split numa/processor.rs into platform leaf modules [patch] (2026-09-02) — `5c3303b9`
- **HS-PROCESSOR-BINDING-LINUX-2026-09-01** ProcessorBinding and ProcessorIndex::current have no Linux backend [minor] (2026-09-01) — `6da6d139`
- **HS-THEMIS-AFFINITY-CONSUMER-2026-09-01** [patch] (2026-09-01) — `92cbfcbc6f926e8e1fae689214dc4a604eb4275e`, `bf48f97`, `9e1b9e4`
- **HS-ADR-INDEX-GENERATOR-ABSENT-2026-09-01** [patch] (2026-09-02) — `38af58f5`, `028a8822`
- **HS-PAIR-DEINTERLEAVE4-2026-09-01** Fused four-way pair deinterleave [minor] (2026-09-01)
- **HS-NUMA-BINDING-THEMIS-QUERY-2026-09-01** [minor] (2026-09-01)
- **HS-PAIR-DEINTERLEAVE-2026-09-01** Deinterleave at adjacent-lane-pair granularity [minor] (2026-09-01)
- **HS-REAL-WINDOW-INTERLEAVE-2026-09-01** Fuse real windowing into complex layout [minor] [perf] (2026-09-01) — `c42571d`, `3c6feb4`, `03d80d33`
- **HS-REAL-COMPLEX-DOT-2026-09-01** Dot real samples with interleaved complex weights [minor] [perf] (2026-09-01) — `59c89431`, `2e993503`, `e5b9e7d`
- **HS-EXACT-PROCESSOR-BINDING-2026-08-31** Exact thread placement for reproducible consumer measurements [minor] [arch] (2026-09-01) — `6baf287`
- **HS-SPMV-GATHER-PREFETCH-2026-08-29** Measure out-of-cache CSR prefetch [patch] [perf] (2026-08-29) — `335c3f8`, `232d167`
- **HS-REDUCTION-UNROLL-2026-08-29** Measure backend-specific reduction depth [patch] [perf] (2026-08-29)
- **HS-SIMD-PERF-2026-08-28** AVX-512 f32 transpose network and bit-exact oracle [patch] (2026-08-31) — `5c50d1d`, `03d80d33`
- **HS-AVX512-TRANSPOSE-2026-08-28** AVX-512 square-tile transpose [patch] (2026-08-28)
- **HS-TRANSPOSE-NETWORKS-2026-08-27** In-register transpose_square permute networks [patch] (2026-08-27) — `4af1b25`, `93ba7ce`
- **HS-MASKED-TAIL-PARTIAL-LOAD-2026-08-27** Partial masked load/store seam for dispatch tails [minor] (2026-08-31) — `01a03eb2`, `55b918675e48`, `eb4058a`
- **HS-SPMV-SHORT-ROW-MASKED-2026-08-27** Masked single-vector body for short SpMV rows [patch] [arch] (2026-08-31) — `01a03eb2`, `55b918675e48`, `eb4058a`
- **HS-ARGEXTREMA-ONE-PASS-2026-08-27** Measure single-pass arg-extrema [patch] (2026-08-27)
- **HS-AVX2-INTERLEAVE-OVERRIDES-2026-08-27** Native AVX2 interleave/deinterleave [patch] (2026-08-27) — `d791281c`
- **HS-DISPATCH-CACHE-THROUGHPUT-2026-08-27** Measure cached dispatch boundary [patch] (2026-08-27) — `99910ad`, `01a03eb2`, `55b918675e48`, `3c548015`
- **HS-PULP-LANE-THROUGHPUT-2026-08-27** Measure Pulp lane parity [patch] (2026-08-27) — `3c548015`, `01a03eb2`, `55b918675e48`, `c3d1b676`
- **HS-CAPABILITY-LOAD-THROUGHPUT-2026-08-27** Hoist support probes from strided lane loads [patch] (2026-08-27) — `c3d1b67`
- **HS-FEARLESS-COMPLEX-REG-THROUGHPUT-2026-08-27** Measure interleaved complex-register parity [patch] (2026-08-27) — `ba32b8c`, `01a03eb2`, `55b918675e48`
- **HS-FEARLESS-PERMUTE-THROUGHPUT-2026-08-26** Measure shared cross-lane parity [patch] (2026-08-26) — `01a03eb2`, `55b918675e48`, `4ef5145`
- **HS-FEARLESS-F32-THROUGHPUT-2026-08-26** Verify native single-precision lane parity [patch] (2026-08-26) — `01a03eb2`, `55b918675e48`, `937c120`
- **HS-BOARD-COMPACTION-TOOLING-2026-09-01** Stack compactor cannot compact this board [patch] (2026-09-02) — `9a5e00d68`
