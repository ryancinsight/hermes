# ADR 022: Evidence Standard for Kernels on Un-runnable ISAs

## Status

Proposed

## Context

Two merged pull requests set opposite precedents for landing an AVX-512 kernel
on a development host that cannot execute AVX-512, and no integrator has ruled
between them.

- **PR #94** (`HS-TRANSPOSE-NETWORKS-2026-08-27`) retained the AVX2 f32 and NEON
  f32 transpose networks on measurement and *deleted* the provisional AVX-512
  candidates, on the stated ground that no controlled real-silicon baseline was
  available. Its principle: an unmeasured optimization is not an optimization.
- **PR #98** (`HS-AVX512-TRANSPOSE-2026-08-28`) then *landed* the AVX-512 f64
  8x8 network on symbolic verification of the permutation algebra alone, and
  **PR #100** (`HS-SIMD-PERF-2026-08-28`) followed it for the f32 16x16 network.
  Neither overturned PR #94's ground; PR #100 filed the tension as residual risk.

A future AVX-512 change can cite either precedent and be consistent with the
board. That is the defect, and it is not confined to this repository: apollo's
`ATLAS-APOLLO-WIDER-ISA-2026-08-28` is blocked behind the ambiguity, because it
needs to know before it starts whether symbolic verification is a sufficient
landing standard or whether the work parks until silicon exists.

The development host is an Arrow Lake Core Ultra 9 285K reporting
`avx512f: false`. "Just measure it" is not available, and the repository's own
records establish why the obvious substitutes do not close the gap:

- **Intel SDE emulates semantics, never timing.** `test-avx512-sde` runs the
  workspace under `sde64 -spr`, and its identification step runs under the
  emulator too, so passing is a hard assertion that SDE satisfies the runtime
  probes. HS-428 records the limit explicitly: SDE validates semantics, never
  performance.
- **Hosted real silicon is opportunistic, not requestable.** HS-429 rejected
  paid instance-family pinning and instead committed `test-avx512-hosted`,
  which runs on the heterogeneous GitHub-hosted x86 pool, records the machine
  class, asserts AVX-512 coverage only when the host happens to provide it, and
  captures the `avx512-native` criterion baseline against the forced
  `hermes_benchmark_generic_default` build on exactly those hosts. Hosts lacking
  the silicon print NOT COVERED and skip the benchmark loudly.
- **For AMX the acceptance is unmeetable, not merely unavailable.** SDE emulates
  the tile instructions, CPUID, and XGETBV, but tile-data permission goes
  through a syscall SDE passes through to the host kernel, so AMX does not
  execute under emulation at all.

Two further facts from this repository's measurement record bear on the ruling,
because they cut in opposite directions and a defensible standard must account
for both:

1. **Hand-written overrides in this repository have been falsified by
   measurement.** HS-427 wrote, measured, and removed the AVX2
   interleave/deinterleave overrides at a **37% regression** against the generic
   default; PR #94 removed the regressing NEON f64 override for the same reason.
   LLVM already lowers the generic default's stack round-trip into good shuffle
   sequences in those cases. "A native network beats the default" is therefore
   not a safe prior in this codebase.
2. **`transpose_square` is the case where the default is measured to be bad.**
   Pinned to a single P-core, the AVX2 f32 8x8 network runs at 4.185–4.198 ns
   against 422.74–426.87 ns for the forced generic default (101x), and AVX2 f64
   4x4 at 2.861–3.032 ns against 100.94–101.38 ns (34x). The default
   round-trips a whole tile through memory rather than lowering to shuffles,
   which is a structural, ISA-independent property of that operation.

The discriminator between (1) and (2) is not the ISA. It is whether the generic
default already lowers to in-register shuffles or round-trips through memory —
a question answerable by codegen inspection on any host, including one without
the target silicon.

## Options

1. **PR #94's standard generalized: no un-runnable-ISA kernel lands without
   real-silicon timing.** Honest, but its remedy is deletion of verified work.
   Its observed cost in this repository is concrete: the AVX-512 transpose
   networks deleted by PR #94 were re-derived from scratch two days later by
   PRs #98 and #100. Deletion also has no convergence path — deleted code is not
   measured when silicon eventually appears, because it is not there to run.
2. **PR #98's standard generalized: correctness evidence is sufficient to
   land, and the performance question is simply not asked.** This is what the
   tree does today, and it is what leaves an unmeasured optimization on `main`
   describing itself as an optimization. It discards PR #94's principle rather
   than bounding it.
3. **Ship un-runnable-ISA kernels behind a cargo feature until measured.**
   A gated path is an untested path, and the repository already runs
   feature-free with runtime capability detection. This would fork the dispatch
   surface to record a documentation fact.
4. **Separate the landing decision from the claim.** Require a defined
   correctness-evidence set to land; prohibit every performance claim until real
   silicon measures the path; carry the measurement as a standing obligation
   that the already-committed opportunistic CI job discharges on its own; and
   apply PR #94's deletion remedy at the moment measurement falsifies the
   kernel, rather than in the absence of measurement.

## Decision

Adopt option 4. PR #94 and PR #98 are not answering the same question, and the
ruling separates them: **PR #98's disposition is binding on whether such a
kernel may land; PR #94's principle is binding on what may be said about it.**
PR #94's *remedy* — deleting a correctness-verified kernel for want of timing —
is superseded.

### 1. Evidence required to land an un-runnable-ISA kernel

All four are required. They are stated as what PRs #98 and #100 actually
supplied, because that is the standard being ratified.

- **Symbolic verification of the permutation or arithmetic algebra**, off
  machine, against an explicit model of every intrinsic used. The model itself
  must be validated by reproducing an already-landed network of the same family
  exactly; an unvalidated model is not evidence.
- **Execution under emulation where the ISA is emulable.** The per-backend
  property laws — the index-coded law, the involution law, and the
  `transpose_square_is_bit_exact_all_backends` oracle — must instantiate at the
  target backend so `test-avx512-sde` runs the real intrinsic encodings. The
  bit-exactness oracle is required and not optional: the index-coded law
  manufactures small positive integers, so a network leaking an operand through
  an arithmetic or NaN-canonicalizing instruction would satisfy it while
  rewriting lane bits.
- **Codegen inspection of the emitted assembly**, confirming three things: the
  instruction budget and count; the *feature* budget, meaning no operand outside
  the feature set the dispatcher actually probes (the AVX512DQ/BW/VL-on-F-only
  failure mode PR #92 fixed); and bounds-check elision, since PR #100 found both
  AVX-512 networks carrying 24 and 30 `ud2` sites behind a slice index.
- **A warning-denied compile** of the path in the configuration that selects it.

Where the ISA is **not** emulable — AMX, whose tile-data permission SDE passes
through to the host kernel — the second requirement cannot be met and the kernel
does **not** land. This standard does not license AMX work on non-AMX hosts.

### 2. What may not be claimed without real silicon

**No performance claim of any kind.** Not a factor, not a benchmark row, not
"faster", not "avoids the round-trip", not a CHANGELOG sentence implying a
speedup. A correctness argument is not a speed argument, and evidence must match
the claim category.

The one permitted characterization is **canonical lowering**: the network is the
recognized in-register form of the operation and is correctness-equivalent to
the generic default. Whether it is faster on real silicon is unmeasured and must
be said so, in the CHANGELOG entry and in the board item, in those terms.

Measurements of *adjacent runnable paths* may be reported — PR #100's pinned
AVX2 figures are legitimate — but only labelled as what they are: a
quantification of what the generic default costs on a path that did not change,
never a measurement of the changed path. PR #100 stated this correctly and is
the model.

### 3. Disposition of a verified-correct, unmeasured kernel

It **ships ungated on `main`** as the selected path for its backend. Not behind
a cargo feature (a gated path is an untested path), and not held out of tree.

It carries three obligations, and lands incomplete without them:

- The CHANGELOG and board entries state the evidence limit in the terms of
  section 2.
- A standing measurement item is filed with an explicit re-open trigger, so the
  debt is on the board rather than in a PR body.
- The kernel is **provisional**. When real silicon does measure it, the finding
  is binding in both directions: if it does not beat the generic default it is
  **deleted then** — which is PR #94's remedy, applied at the moment the
  evidence exists rather than in its absence. Given HS-427's 37% AVX2
  interleave regression and PR #94's NEON f64 result, this is a live outcome,
  not a formality.

The obligation is dischargeable without human action: `test-avx512-hosted`
already saves the `avx512-native` baseline and compares it against the forced
generic-default build whenever a hosted x86 runner reports `avx512f`. Landing
the kernel is what makes that job able to measure it; deletion removes the only
path by which the question ever gets answered.

### 4. Existing landed networks

Neither is grandfathered, and neither owes re-derivation.

- **AVX-512 f64 8x8 (PR #98)** and **AVX-512 f32 16x16 (PR #100)** already meet
  section 1 in full — symbolic model validated against the f64 network, property
  and bit-exactness laws instantiated under SDE, codegen inspection that both
  confirmed the shuffle and feature budgets and found the bounds-elision defect.
  The standard was written from what they supplied precisely so this is checked
  rather than assumed.
- They already satisfy section 2: `HS-SIMD-PERF-2026-08-28` and
  `HS-AVX512-TRANSPOSE-2026-08-28` both record no-hardware-measurement as
  residual risk and make no speed claim.
- The outstanding debt is section 3's standing measurement item, which this ADR
  names as the deliverable of `HS-AVX512-EVIDENCE-STANDARD-2026-08-31` on
  ratification.

**PR #94's deleted networks are not restored by this ADR.** They were never
verified to section 1, so restoring them would be reinstating unverified code,
not correcting an unjust deletion. The transpose members were re-derived
correctly by PRs #98 and #100; anything else PR #94 removed re-enters as ordinary
board work under this standard.

### 5. What this ADR cannot settle

Whether either landed AVX-512 transpose network is actually faster than the
generic default on AVX-512 silicon. Nothing available here settles it. What
would settle it, in order of preference:

1. A `test-avx512-hosted` run landing on a hosted x86 runner that reports
   `avx512f`, producing the `avx512-native` baseline and the generic-default
   comparison — the instrument is already committed and requires no new work.
2. The same A/B on a self-hosted or rented AVX-512 host under the pinned-core
   protocol (`ProcessorBinding`, ADR 021), which is what makes the figures
   comparable to the AVX2 numbers already on record.

Until one of these produces a baseline, no factor may be quoted for these paths.

## Consequences

- Apollo's `ATLAS-APOLLO-WIDER-ISA-2026-08-28` unblocks. It may land AVX-512
  record widening on section 1 evidence, must make no speed claim under
  section 2, and files the section 3 measurement obligation. It does not wait
  for silicon.
- Future AVX-512 pull requests cite this ADR rather than choosing between two
  merged precedents.
- The repository accepts that `main` may carry provisional kernels whose
  performance is unknown. That is the deliberate trade: it is visible on the
  board and self-discharging through CI, where deletion is neither.
- AMX work on non-AMX hosts remains blocked by section 1, and this ADR does not
  create a route around it.
- This ADR sets a precedent affecting another repository, so it is filed
  **Proposed** and awaits the owner's ratification. Until ratified, the tree's
  current state — the standard as practised by PRs #98 and #100 — stands
  unchanged, since this ADR ratifies rather than alters it.

## Revision history

- 2026-09-01: Proposed for `HS-AVX512-EVIDENCE-STANDARD-2026-08-31`, adjudicating
  the PR #94 / PR #98 precedent conflict recorded as residual risk in PR #100.
  Awaiting owner ratification; apollo's `ATLAS-APOLLO-WIDER-ISA-2026-08-28` is
  the blocked consumer.
