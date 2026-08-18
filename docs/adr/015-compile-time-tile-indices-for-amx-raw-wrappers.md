# ADR 015: Compile-Time Tile Indices for the AMX Raw Wrappers

## Status
Accepted

*Accepted 2026-08-17, retroactively: the decision below shipped (HS-438,
delivered 2026-08-16) while this record still read `Proposed`. Verified on
`main` at acceptance: all five raw wrappers are const-generic —
`tilezero<const TILE: u8>` (`amx/mod.rs:131`), `tileloadd<const TILE: u8>`
(`:158`), `tilestored<const TILE: u8>` (`:188`),
`tdpbf16ps<const DST, SRC1, SRC2>` (`:218`), and
`tdpbssd<const DST, SRC1, SRC2>` (`:247`) — and `amx/mod.rs` now contains zero
`unreachable!()`, which was the defect this decision existed to remove: an
unlisted-but-valid tile triple used to panic instead of executing. The decision
text is unchanged.*

## Context

The `raw` AMX wrappers in
`crates/hermes-simd-intrinsics/src/x86_64/amx/mod.rs` dispatch the tile index
at runtime. `tilezero`, `tileloadd`, and `tilestored` match an 8-arm `tile`
value; `tdpbf16ps` and `tdpbssd` match an 11-entry `(dst, src1, src2)`
whitelist with an `unreachable!()` tail. The whitelist is a latent defect:
every valid-but-unlisted triple — `TDPBF16PS` and `TDPBSSD` accept any of the
64 `(dst, src1, src2)` register triples — panics instead of executing. Every
one of the ~100 call sites in `amx/bf16.rs` and `amx/int8.rs` passes a
literal, so the runtime index has no call-site variation to serve.

`asm!` substitutes a `const` operand textually, so
`"tilezero tmm{TILE}"` with `TILE = const TILE` assembles the correct
mnemonic and rejects out-of-range indices at compile time. This is the same
mechanism ADR 003's stable inline assembly already relies on.

## Options

1. **Const generic parameters.** `tilezero<const TILE: u8>()`,
   `tileloadd<const TILE: u8>(base, stride)`,
   `tilestored<const TILE: u8>(base, stride)`,
   `tdpbf16ps<const DST: u8, const SRC1: u8, const SRC2: u8>()`, and the
   `tdpbssd` analogue. One `asm!` block per wrapper; the 8-arm match, the
   11-entry whitelist, and the `unreachable!()` tail are deleted. Out-of-range
   indices stop compiling. The `raw` public API breaks ([major]).
2. **Keep the runtime dispatch and extend the whitelist.** Removes the panic
   for the tdp wrappers only; leaves a branch inside the tile loop and a
   whitelist that must stay exhaustive as the kernels evolve.
3. **Macro-generated arms.** `macro_rules!` fanning out the 8 arms removes
   duplicated arm text but keeps the runtime branch and adds macro-policy
   costs (IDE support, type errors) for no dispatch win.

## Decision

Option 1. The five wrappers take the tile indices as const generic
parameters and call sites in `amx/bf16.rs` and `amx/int8.rs` supply them as
turbofish literals in the same change. The `raw` API break is released as a
pre-1.0 minor bump per the repository convention (ADR 014), documented under
**Breaking** in the CHANGELOG. The Miri arms bind the runtime parameters the
wrappers still take (`base`/`stride` for the tile-load/store pair) so the
`-D warnings` miri compile stays clean; the const-generic tile index needs no
binding because no runtime value exists for it.

The change is compile-time-only by construction: the generated instruction
stream for every previously-reachable call site is unchanged, and the index
branch vanishes from the tile loop.

## Consequences

- Public API: the five raw functions lose their positional index
  parameters; consumers move the literal into turbofish position
  (`raw::tileloadd(0, p, s)` -> `raw::tileloadd::<0>(p, s)`,
  `raw::tdpbf16ps(2, 0, 1)` -> `raw::tdpbf16ps::<2, 0, 1>()`).
- Behavior: identical instruction text for every call site that previously
  compiled; the valid-but-unlisted panic class is gone (those triples now
  execute, and any triple that the assembler rejects fails the build instead
  of panicking).
- The runtime `match`/`unreachable!()` disappears from `amx/mod.rs`.
- The AMX-silicon before/after measurement in HS-438's acceptance rides on
  the HS-429 hardware validation increment, where a criterion baseline is
  possible; it is a recorded watchpoint, not a software blocker.
## Evidence / Verification Plan

- `cargo build -p hermes-simd-intrinsics --all-targets` and
  `cargo check --workspace --all-targets` green on x86-64.
- Clippy `-D warnings` (pedantic floor) clean; nextest 30/30; doctests clean.
- `cargo semver-checks -p hermes-simd-intrinsics --baseline-rev origin/main
  --release-type minor` reports `semver requires new major version: 2 major
  and 0 minor checks failed` — `function_parameter_count_changed` and
  `function_requires_different_const_generic_params` on all five functions.
- The x86_64 build assembles the rewritten instruction text in every target,
  including the SDE whole-program-emulation job and the best-effort
  `test-avx512-hosted` job on GitHub-hosted x86. Whether the AMX instructions
  themselves execute on a hosted runner depends on the kernel admitting
  XTILEDATA: the capability probe's condition 3 is a real `arch_prctl`
  permission syscall, which a kernel with AMX support grants, so `amx` is
  deliberately not asserted in either job's `HERMES_EXPECTED_TARGETS` (the
  hosted job asserts `scalar,avx2` plus `avx512` only when the silicon is
  present). Execution-level verification of AMX therefore remains deferred to
  a host whose kernel admits XTILEDATA; the hosted job may satisfy this
  opportunistically.
- Measured before/after on AMX silicon: deferred to HS-429 hardware.

## References

- HS-438 backlog item (backlog.md).
- `crates/hermes-simd-intrinsics/src/x86_64/amx/{mod,bf16,int8}.rs`.
- ADR 003 (stable inline assembly for AMX/VNNI).
- ADR 014 (pre-1.0 minor-release **Breaking** convention).
