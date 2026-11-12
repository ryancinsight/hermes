# ADR 026: Hermes release scope for 1.0

## Status

Accepted (2026-11-12)

## Context

Hermes (`hermes-simd`, workspace packages 0.9.2) is the Atlas stack's
first-party SIMD kernel provider: scalar fallbacks, core dense kernels, and
ISA backends (AVX2, AVX-512, NEON) with prefetch and codegen-owned sources.
Four active items remain (provider cohort admission, memory-source identity,
codegen source of truth, rescue-port hygiene) and no milestone ladder exists
to 1.0. Checked-in ISA files are canonical; the incomplete codegen generator
is deleted, not restored.

## Decision

Adopt a three-milestone ladder to 1.0 recorded in `backlog/milestones.md`:

- **M-core-kernels** (0.10.0): scalar fallbacks and core kernels verify on
  the admitted provider cohort (Eunomia range, one Mnemosyne identity) with
  the rescue queue ported or closed.
- **M-backends-isa** (0.11.0): ISA backends, prefetch, and permutes verify
  against the codegen source-of-truth resolution (checked-in files canonical).
- **M-1.0** (1.0.0): stable kernel surface (authority-gated) with documented
  contracts and release gates satisfied.

In scope: core/scalar kernels, ISA backends, provider cohort and memory
identity, the codegen-SSOT record sync, rescue-port completion. Out of scope:
new f16/NEON implementation work, public API redesign, any `publish = true`
without release authority (veto point: M-1.0 never forces publication).

## Alternatives

- Plan 1.0 as a single release: rejected because provider-identity and
  rescue-port work remains open.
- Keep roadmap implicit in backlog items: rejected because stack policy
  requires a runnable capability-oriented milestone ladder.

## Consequences

- Open items carry `milestone:` assignments and name the exit check they serve.
- Milestones close only when exit checks pass on fetched default commits; 1.0
  release/publication remains authority-gated.
- This ADR is rewritten in place as scope evidence changes.
