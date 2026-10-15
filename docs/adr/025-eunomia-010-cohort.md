# ADR 025: Adopt the Eunomia 0.10 provider cohort

## Status

Accepted (2026-10-15)

## Contract and decision

Hermes reexports Eunomia scalar types and exposes their traits through public
generic kernels. The provider's CastFrom/CastTo retirement is versioned 0.10.0;
its nominal type identities require a pre-1.0 minor boundary in Hermes. Advance
the aligned Rust workspace line and internal package edges from 0.8.2 to 0.9.0.
Require Eunomia 0.10.0 and the coordinated Mnemosyne-memory 0.10.0 line. Keep
canonical Git URLs and Cargo-managed precise locks, without revision aliases,
path patches or compatibility shims. Rust implementations, fixtures, assertions,
precision choices and native deadlines are unchanged.

## Verification and integration constraints

Validate the ordinary native suite, strict all-target/all-feature lint, minimal
feature builds, docs, package and API release classification. A candidate lock
to an unmerged provider is prototype evidence; final default and registry
acceptance requires the published cohort.

The candidate passes 507 all-feature native cases with zero skips, strict
all-target/all-feature Clippy, minimal workspace builds, warning-denied docs,
and 27 doctests (ten existing ignores). Automatic API release classification
accepts all five 0.8.2 to 0.9.0 boundaries, skipping 253 breaking-sensitive
rules per library by design. Standard package verification is blocked by
unpublished registry Mnemosyne-memory 0.10.0, before archive compilation.

The existing required `miri (core + intrinsics boundary)` job runs on every PR,
including drafts. The user now authorizes these existing required Miri checks;
their actual exact-head results remain pending. Prepare the reviewed PR without
package publication or publication-triggering default merges in this turn.
Do not alter the workflow, required status or tests to manufacture acceptance.
No held safety work or performance experiments are resumed.
