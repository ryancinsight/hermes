# ADR 005: Checked-In Per-Type x86 Kernel Sources

## Status

Accepted

## Context

The x86 specialized hardware kernels (`avx2_f32.rs`, `avx2_f64.rs`,
`avx512_f32.rs`, `avx512_f64.rs`, and the additional precision variants) share
some operation structure, but the shipped surface is not a four-file
substitution matrix. It includes ISA- and precision-specific operations such
as AVX-512 scatter, cross-lane permutations, and x86 and NEON f16 paths. The
checked-in files are therefore the only source that currently represents the
complete backend contract.

## Superseded decision

The original decision selected a build-time generator as the source of truth
for the per-type x86 kernels. That decision was not kept fresh. A pinned
Rust 1.97.0 run of the checked-in `codegen` binary regenerated four files and
removed 28 shipped method implementations across those files: five from each
AVX2 file and nine from each AVX-512 file. The generator also emitted only
f32/f64 x86 files, leaving the shipped x86 f16 and AArch64 NEON families
outside its model. No build or CI path invoked it, and its output did not
carry a generated-file marker or freshness gate.

## Decision

Retire the incomplete generator and make the checked-in ISA kernel files the
canonical sources. Do not regenerate or delete those files as a side effect
of a build. A future generator may be introduced only after it models every
shipped precision, ISA, and operation family, produces byte-stable formatted
output, and has a CI regeneration-diff gate. That is a separate architectural
decision and is not implied by this record.

The remaining shared operation structure is an ordinary consolidation target:
shared helpers or trait defaults may be added where they preserve the backend
contract and have a complete conformance proof. This record does not authorize
mechanically collapsing operations whose semantics differ by ISA or
precision.

## Consequences

- The obsolete `codegen` binary and its ungoverned write-to-`src` behavior are
  removed.
- Checked-in kernel diffs remain the review and verification boundary for the
  full shipped surface.
- The existing per-ISA/per-precision duplication remains visible until a
  complete, tested consolidation proves it can be reduced without losing
  operations.
- This change is source and tooling cleanup only; it does not alter runtime
  SIMD dispatch or kernel behavior.

## Revision history

- **2026-08-21:** Retired the incomplete generator after direct pinned-toolchain
  regeneration demonstrated destructive coverage drift. The accepted
  build-time-generator direction is superseded by the checked-in-source
  decision above; its strongest alternative was rejected because restoring
  freshness requires a complete ISA/precision/operation model that did not
  exist in the audited generator.

## Verification

- Direct `rustc +1.97.0` compilation and execution of the former generator
  reproduced the four-file destructive diff described above.
- The provider’s normal format, Clippy, Nextest, doctest, Rustdoc, and package
  gates must pass after removal of the binary.

## References

- `ATLAS-HERMES-CODEGEN-SSOT-2026-08-21` in `backlog.md`.
- `crates/hermes-simd-intrinsics/src/x86_64/`.
- `crates/hermes-simd-intrinsics/src/aarch64/`.
