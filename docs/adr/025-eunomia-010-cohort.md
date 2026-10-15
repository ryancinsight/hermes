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
rules per library by design. Mnemosyne is now published from
[b7e0e1e](https://github.com/ryancinsight/Mnemosyne/commit/b7e0e1e37130b0a317859b168cca16b5172d08d8).
All six registry archives match their checksums and that reviewed merge source;
publisher 38752721843 and exact-head Miri/Loom/TSan/native/aggregate plus blocking
API/package validation pass. The final Hermes lock selects that actual source,
with current Themis 63c54410 and Melinoe 681c550e. All 507 native cases and strict
Clippy pass on the new lock. Clean package, minimal builds, docs, blocking API
and final-head CI are being rerun before acceptance.

All twelve executed initial [initial candidate](https://github.com/ryancinsight/hermes/commit/bb116d8c098b75ca55a58d5d8464c7539b087bb7) CI jobs pass, including the existing required
`miri (core + intrinsics boundary)` job. The user explicitly authorizes required
Miri and publication through the normal default merge after all final-head
checks/reviews pass. Initial-source evidence cannot replace final-provider-lock
acceptance. Do not alter workflows, required statuses, tests or deadlines, and
do not resume held safety work or performance experiments.
