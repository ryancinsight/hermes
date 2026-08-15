# ADR 014: Add `TargetId::Sve` and Seal the Target Enum Family

## Status
Proposed

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
  `dispatch_view_mut_to` exactly like the other emulated/scalar paths, gated by
  `is_architecture_applicable` (`cfg!(target_arch = "aarch64")`) and
  `is_supported` (SveArch emulation availability, which is always true when
  compiled for AArch64).
- `dispatch_view` auto-selection stays untouched: an emulated backend must
  remain explicitly requested, never auto-selected. `Sve` does not enter the
  `ALL`-ordered host-support probe used by `dispatch_view`.
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
  closing the HS-425 hole. The emulated `SveArch` paths already have
  differential coverage; this item's new coverage asserts the same value
  semantics through the `TargetId::Sve` route on the aarch64 and SDE jobs.
- Future backend additions (SME, a native SVE path) are additive for
  consumers, requiring no downstream `match` change.

## Evidence / Verification Plan

- `cargo build --workspace --all-targets` green on x86-64 and
  `cargo check --target aarch64-unknown-linux-gnu`.
- `cargo nextest run --workspace` green; new tests assert
  `TargetId::Sve.is_architecture_applicable()`/`is_supported()` on aarch64 and
  `false` on other hosts, `name() == "sve"`, and value-semantic dispatch
  through `dispatch_view_to(..., TargetId::Sve)` under the emulated backend on
  both the aarch64 and SDE runners.
- `cargo-semver-checks` against 0.6.0 reports the expected major transitions
  for both enums (variant added + `#[non_exhaustive]` applied).
- Clippy `-D warnings`, doctests, and Rustdoc clean.

## References

- HS-425 backlog item (backlog.md).
- `crates/hermes-simd/src/target.rs` — `TargetId` and dispatch helpers.
- `crates/hermes-simd/src/lib.rs` — `DispatchedView` and `dispatch_view`.
- `crates/hermes-simd/tests/host_capability_tests.rs` — conformance assertions.
