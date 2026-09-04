# ADR 023: One First-Party Memory Source Identity During Provider Co-Evolution

## Status

Accepted (2026-09-03)

## Context

Hermes PR #155 moved its Eunomia workspace edge to the derive-capable provider
revision used by the Atlas provider sweep, but its Mnemosyne workspace edge
still selected `03fe32f`. Apollo and Leto select Mnemosyne PR #123 at
`da5c6be`. Cargo therefore resolved two nominal copies of the same
`mnemosyne-memory` package in consumers that combine Hermes with those
providers. The duplicate source identity increases compile work and prevents
types from crossing the provider boundary when both copies appear in a public
contract.

## Decision

Advance Hermes' workspace `mnemosyne-memory` dependency to `da5c6be` while
Mnemosyne PR #123 is under review. Keep the exact revision as a temporary
co-evolution pin with a removal trigger: once the provider change merges to
main, remove `rev` and regenerate the standalone lockfile. The workspace
manifest remains the single dependency source of truth; no downstream
conversion, path override, compatibility layer, or duplicate API is added.

## Rejected alternatives

- Keeping `03fe32f` was rejected because it preserves the duplicate nominal
  provider identity in the consumer graph.
- Adding a conversion layer in Hermes or Apollo was rejected because source
  identity is owned by the provider dependency edge, not by each consumer.
- A workspace-local path override was rejected because it changes standalone
  and published resolution and violates the stack overlay ownership rule.

## Contract and verification

The change preserves Hermes' public SIMD and memory behavior; it changes only
the resolved first-party provider revision. At the initial decision revision,
the standalone lockfile resolved Mnemosyne `da5c6be` and Eunomia `fdbf122`.
After the 2026-09-04 follow-up below, it resolves Mnemosyne `26726d2` and
Eunomia main at `02397fa`. Workspace check and warning-denied Clippy pass,
Nextest passes 548/548, 26 executable doctests pass, warning-denied rustdoc
passes, and `git diff --check` passes.

## Consequences

Hermes consumers that already use the current Atlas provider revisions now
share one Mnemosyne source identity. The exact revision remains visible until
PR #123 merges; the merge is the removal trigger, not a reason to retain the
pin indefinitely.

## Revision note

2026-09-04: Eunomia PR #87 merged. The workspace Eunomia edge now follows the
merged default branch, eliminating the obsolete review source identity while
the independent Mnemosyne PR #123 pin remains in force. This lets unpinned Gaia
and Leto consumers share Hermes' Eunomia scalar and derive traits directly.
