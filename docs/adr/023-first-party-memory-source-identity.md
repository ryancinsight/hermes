# ADR 023: One First-Party Memory Source Identity During Provider Co-Evolution

## Status

Accepted (2026-09-03)

## Context

Hermes PR #155 moved its Eunomia workspace edge to the derive-capable provider
revision used by the Atlas provider sweep, but its Mnemosyne workspace edge
still selected an older provider revision. Apollo and Leto selected a later
commit in the same provider line. Cargo therefore resolved two nominal copies
of the same `mnemosyne-memory` package in consumers that combine Hermes with
those providers. The duplicate source identity increases compile work and
prevents types from crossing the provider boundary when both copies appear in
a public contract.

## Decision

Advance Hermes' workspace `mnemosyne-memory` dependency to Mnemosyne 0.9.0
using its git-plus-version requirement. Cargo.lock records the resolved
upstream commit; the manifest carries no temporary `rev` pin. The workspace
manifest remains the single dependency source of truth; no downstream
conversion, path override, compatibility layer, or duplicate API is added.

## Rejected alternatives

- Keeping the older provider revision was rejected because it preserves the
  duplicate nominal provider identity in the consumer graph.
- Adding a conversion layer in Hermes or Apollo was rejected because source
  identity is owned by the provider dependency edge, not by each consumer.
- A workspace-local path override was rejected because it changes standalone
  and published resolution and violates the stack overlay ownership rule.

## Contract and verification

The change preserves Hermes' public SIMD and memory behavior; it changes only
the resolved first-party provider revision. The standalone lockfile resolves
all eight Mnemosyne packages to upstream commit
`c5695f19008db63c508c30b22a9e84c78da4d380`, with no duplicate Mnemosyne
source identity in Hermes. Workspace check and warning-denied Clippy pass;
the PR's recorded gate also covers Nextest, doctests, rustdoc, and
`git diff --check`.

## Consequences

Hermes consumers that already use the current Atlas provider revisions now
share one Mnemosyne source identity. Future compatible provider updates can
advance through the version requirement and lockfile without editing a
temporary review pin.

## Revision note

2026-10-04: Mnemosyne 0.9.0 is now the workspace dependency. The manifest
uses the git-plus-version requirement without `rev`, and the standalone lock
resolves all eight Mnemosyne packages to upstream commit
`c5695f19008db63c508c30b22a9e84c78da4d380`. The earlier review-revision
narrative and its removal trigger are obsolete.
