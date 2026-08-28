# ADR 019: Format-Owned SpMV Kernel Modules

## Status

Accepted

## Context

`sparse::spmv` owns one public multiplication trait plus the CSR,
DenseWithMask, Blocked-COO, and SELL-p implementations. Those four operation
families accumulated in one source file past 600 lines. Extending the
DenseWithMask remainder crossed the configured per-function lint floor and
made an otherwise local performance change depend on unrelated format code.

The public trait and the size and index-vector invariants are common to every
format. Each format's traversal, validation proof, and vectorization strategy
are otherwise independent. The module boundary can express that ownership
without changing any public path or runtime dispatch.

## Options

1. Suppress the function-size diagnostic. This preserves the mixed-format
   source and weakens the lint floor at the changed implementation.
2. Extract only the new DenseWithMask tail helper. The source remains over the
   file-size target and still combines four independent formats.
3. Keep the public trait and shared invariants in `sparse::spmv`, and move each
   format implementation to one private leaf module.

## Decision

Adopt option 3. `sparse::spmv` remains the public canonical home and re-exports
no new symbols. Its private `csr`, `dense_with_mask`, `blocked_coo`, and `sellp`
children each own one `SparseSpMv` implementation and any format-specific
helpers. The index-vector constructor and input-size assertion stay in the
parent because multiple formats use them.

The move is structural: the format markers, storage types, public dispatch,
validation boundary, target-feature entry, and arithmetic order do not change.
The DenseWithMask tail experiment is reviewed separately inside its format
leaf and remains eligible only with value and benchmark evidence.

## Consequences

- A format change compiles and reviews against its own bounded source region.
- Shared soundness invariants remain single-sourced rather than copied into
  each leaf.
- Public API and SemVer surface are unchanged; this is a private [arch][patch]
  organization decision.
- The structural move requires focused sparse tests plus warning-denied
  all-target Clippy. Targeted Miri or sanitizer evidence remains required for
  changed unsafe memory behavior.

## Revision history

- 2026-08-28: Accepted for `HS-SPMV-SHORT-ROW-MASKED-2026-08-27` when the
  DenseWithMask tail change exposed the mixed-format module boundary.
