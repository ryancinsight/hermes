# ADR 014: Add `TargetId::Sve` and Seal the Target Enum Family

## Status
Accepted

*Accepted 2026-08-17, retroactively: the decision below was implemented and
merged (PR #49, merge `fb36e0f`, feature commit `dd4cc78`) while this record
still read `Proposed`, so the index reported a settled decision as open. The
status is corrected to match what shipped; the decision text is unchanged. See
the acceptance record at the end for what was verified on `main`.*

## Context

`TargetId` (crates/hermes-simd/src/target.rs) and `DispatchedView`
(crates/hermes-simd/src/lib.rs) are the public runtime target tokens for
forced and automatic SIMD dispatch. `SveArch` is a first-class emulated
backend exercised throughout the test suite and the conformance matrix, yet
`TargetId` enumerates only `Scalar`, `Avx2`, `Avx512`, and `Neon`. The public
forced-dispatch API therefore cannot reach a backend the workspace ships.

Both enums are public without `#[non_exhaustive]`, so adding `Sve` breaks every
downstream exhaustive `match` on either type. This is the second time the
closed target set has grown (the AVX-VNNI/AMX backends live under a separate
facade path, so the set has not grown before); the item is reclassified [major]
for that reason.

Version state: workspace is at `0.6.0`, pre-1.0. The repository convention
(CHANGELOG header) is that pre-1.0 minor releases may break, documented under
**Breaking**.

## Options

1. **Add `Sve` only, without `#[non_exhaustive]`.** Closes the immediate hole
   but leaves the exact same break for the next backend (`SveArch` variants,
   SME, or a future `Sme`), which is the recurrence this item exists to prevent.
2. **Add `Sve` and apply `#[non_exhaustive]` to both enums in the same
   break.** The variant addition is already a major break; the sealing rides
   the same migration. Every downstream exhaustive `match` becomes a
   `_ => unreachable!()` tail once, and future backend additions stop being
   breaking changes for consumers.
3. **Add `Sve` and seal only `TargetId`.** Inconsistent: `DispatchedView` is
   constructed from `TargetId` and matched in lockstep, so a sealed token with
   an open view type just moves the break to the other arm.

## Decision

Option 2. Add `TargetId::Sve` and `DispatchedView::Sve` in the same change,
and apply `#[non_exhaustive]` to both enums in that same change. The version
decision is a pre-1.0 minor bump to `0.7.0` documented under **Breaking** per
the repository convention; the variant addition and the sealing are one
consumer-visible break, not two.

Routing and capability semantics:

- `TargetId::Sve` is routed through `dispatch_view_to` and
  `dispatch_view_mut_to` exactly like the scalar path, unconditionally:
  `SveArch` is a lane-emulated backend compiled with `cfg(all())` and exported
  on every host (crates/hermes-simd-intrinsics/src/aarch64/sve.rs), so its
  `is_architecture_applicable()` and `is_supported()` are both `true` wherever
  Hermes builds. It mirrors `TargetId::Scalar`, not the hardware-gated x86 /
  aarch64 targets, and no cfg gates enter the dispatch arms.
- `dispatch_view` auto-selection stays untouched: an emulated backend must
  remain explicitly requested, never auto-selected. `Sve` does not enter the
  runtime feature-probe chain inside `dispatch_view`; the `ALL`-ordered
  `supported_on_host` probe and the conformance report do include it, since it
  executes on every host.
- `name()` returns `"sve"` and `from_name` accepts it, so conformance
  expectations can name the backend in CI configuration.

The `#[non_exhaustive]` sealing is applied in the same commit as the variant
addition, so `cargo-semver-checks` reports one major transition for the pair,
and in-repository exhaustive matches (target.rs dispatch arms, host-capability
tests) are updated to a `_ =>` tail in the same change per the compatibility
rule.

## Consequences

- Public API: `TargetId` and `DispatchedView` gain the `Sve` variant and lose
  closed-match exhaustiveness. Downstream exhaustive matches must add a
  wildcard arm; this is the documented pre-1.0 **Breaking** note in the
  CHANGELOG.
- The forced-dispatch conformance matrix can now reach every shipped backend,
  closing the HS-425 hole. Because `SveArch` is lane-emulated and executes on
  every host, the new `TargetId::Sve` conformance coverage runs on every
  runner (x86-64 CI included), not only on aarch64 and SDE jobs.
- Future backend additions (SME, a native SVE path) are additive for
  consumers, requiring no downstream `match` change.

## Evidence / Verification Plan

- `cargo build --workspace --all-targets` green on x86-64 and
  `cargo check --target aarch64-unknown-linux-gnu`.
- `cargo nextest run --workspace` green; new tests assert
  `TargetId::Sve.is_architecture_applicable()` and `is_supported()` are `true`
  on every host (the emulated backend executes everywhere, matching
  `SveArch::is_runtime_supported()`), `name() == "sve"`, and value-semantic
  dispatch through `dispatch_view_to(..., TargetId::Sve)` under the emulated
  backend. The existing `support_implies_architecture_applicability` invariant
  holds because both predicates are `true` together.
- `cargo-semver-checks` against 0.6.0 confirms the major transition:
  `cargo semver-checks -p hermes-simd --baseline-rev origin/main
  --release-type minor` reports `semver requires new major version` with
  `enum_marked_non_exhaustive` failing for both `TargetId` and
  `DispatchedView` (195 pass, 1 fail, 57 skip). The `--release-type minor`
  declaration is required for the check to be meaningful: declaring `major`
  satisfies every lint's required update, so the tool skips all checks as a
  no-op. `enum_variant_added` passes precisely because the enums are sealed
  `#[non_exhaustive]` in the same change.
- Clippy `-D warnings`, doctests, and Rustdoc clean.

### Acceptance record (2026-08-17)

Checked against `main` at the time of acceptance, because a status flip that
does not verify the shipped code only moves the drift from the status line into
the record:

- Both enums are sealed, which is the part of the decision that had to ride the
  same break: `#[non_exhaustive]` on `TargetId` (`target.rs:22`) and on
  `DispatchedView` (`lib.rs:265`). Option 3 — sealing only `TargetId` — was
  rejected here, and the code follows the decision rather than that option.
- `dispatch_view` auto-selection contains no `Sve` arm, so the emulated backend
  stays explicitly requested and is never auto-selected. This was the item's
  sharpest acceptance criterion and it holds.
- `TargetId::Sve` is reachable through the forced-dispatch helpers, and `Sve`
  is present in the `ALL` ordering and the `name()`/`from_name` mapping, so the
  conformance matrix can name the backend in CI configuration.

## References

- HS-425 backlog item (backlog.md).
- `crates/hermes-simd/src/target.rs` — `TargetId` and dispatch helpers.
- `crates/hermes-simd/src/lib.rs` — `DispatchedView` and `dispatch_view`.
- `crates/hermes-simd/tests/host_capability_tests.rs` — conformance assertions.
