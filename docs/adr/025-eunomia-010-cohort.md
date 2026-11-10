# ADR 025: Coordinate Eunomia provider identities

## Status

Accepted

Revision 2026-11-10: extend the provider-cohort decision to the already
admitted 0.12 line and replace the historical delivery ledger with the
current contract and evidence requirements.

## Context

Hermes reexports Eunomia scalar types. `hermes-simd-core::Scalar` reexports
`NumericElement`, and public kernels carry those provider traits and types.
Two provider revisions with matching source names do not establish one
nominal Rust type identity.

The original migration retires CastFrom/CastTo and advances Hermes from
0.8.2 to 0.9.0 with Eunomia 0.10. The current manifest also admits 0.11 and
0.12 through `>=0.10.0, <0.13.0`. That widening already lands in
[the provider-admission commit](https://github.com/ryancinsight/hermes/commit/097e0a294f21bd2925fd3a26a134b4d577c223a2).
The published 0.9.2 manifest predates the widening and requires `<0.12.0`.

## Decision

Select one immutable provider version and Git revision for each resolved
consumer graph. Direct scalar dependencies and Hermes's public bounds use
that same provider identity. Keep the canonical Git URLs and Cargo-managed
standalone locks. Resolve and test the graph outside the development overlay.

A supported requirement range permits selection of a provider line; it does
not assert interchangeability of types from different provider lines.
Consumers that share scalar values or bounds move their direct dependency
and lock to the selected line together. Value-only callers still undergo
their actual consumer tests. Public-contract breaks require a compatible
Hermes version boundary and a migration; widening a requirement alone is
not compatibility evidence.

Do not restore retired cast traits, introduce revision aliases or forwarding
adapters, or change precision or test assertions to accommodate a mixed graph.
The provider-lock increment uses the existing admitted range and preserves
the kernel source and public signatures from its fetched main basis.

## Verification

Compare the ordinary libraries with the fetched Git base, every distinct
Hermes revision present in committed consumer locks, and published contracts,
using the pinned toolchain and supported feature/target configurations.
Native cargo-semver-checks 0.51 updates its synthetic dependency graphs. An old
Git requirement can therefore be unreconstructable after its provider's
default branch advances. Retain that failure as unsupported coverage.

For such a path, inspect the actual exposed identities and migration,
validate the committed resolved graph, and execute value-semantic consumer
contracts on the exact selected candidate. A successful compiler invocation
alone does not establish historical API compatibility.

The proc-macro crate has no target supported by the native API checker.
Compare its source with the published archive and Git baselines and exercise
its generated consumer code. Relevant existing contracts include
`runtime_dispatch_accepts_inline_bounds_and_borrowed_outputs`, the dispatched
reduction, and the vectorized butterfly against its scalar reference.
Source equality and consumer tests remain distinct evidence categories.

Required native, documentation, architecture-specific and safety jobs retain
their configured bounds. Unavailable ISA coverage remains explicit under
ADR 022. Evidence and exact candidate revisions belong in
[the consumer PR](https://github.com/ryancinsight/hermes/pull/277).

## Consequences and alternatives

Cohort updates can require consumer lock and dependency changes even when
provider package versions remain unchanged. Exact resolved identities, not
matching version strings or checkout state, determine coherence.

A mixed graph with compatibility adapters is rejected because it preserves
parallel nominal types and hides the dependency boundary. Suppressing an
unsupported native comparison is rejected because it erases coverage limits.

Existing prospective approval for automatic documentation deployment on
merge remains in effect. Package publication requires separate release
authority and the configured release gates.

## Overturning evidence

Removing provider types and traits from the public contract can reduce this
coupling. A supported native reconstruction can replace a reviewed
unsupported-path disposition. Any actual consumer regression rejects the
selected cohort until its cause is corrected.
