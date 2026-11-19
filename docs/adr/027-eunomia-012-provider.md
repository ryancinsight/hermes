# 027 — Hermes adopts the published Eunomia 0.12 provider

## Status

Accepted (2026-11-19)

## Context

Hermes exposes Eunomia scalar types and their backend implementations.
The published Hermes 0.9.2 packages restrict Eunomia to versions below
0.12. Aequitas 0.4.0 uses the crates.io Eunomia 0.12 line. Rust treats
scalar types from different crate identities as distinct, so a backend
implementation for the old `Bf16` cannot implement the same kernel for
the new `Bf16`. Leto Ops package verification reproduces this failure.

Eunomia 0.12.1 preserves the 0.12 API and fixes complex fold identities,
finite real extrema, and negative integer powers. Both 0.12.1 archives
are verified on crates.io.

## Decision

Move the four library crates to 0.10.0 and use one crates.io Eunomia
requirement of 0.12.1. Advance the internal library requirements together
to preserve the same scalar identity at each package boundary. Keep the
procedural macro package on its existing 0.9.0 line. SIMD algorithms,
backend dispatch, precision choices, and test inputs stay the same.

Keeping the old and new scalar crates together leaves typed consumers
unable to use the existing kernels. Re-exporting a second scalar type
would split the public contract. The 0.10.0 boundary gives Cargo a
versioned way to select the unified family.

## Verification

Check the published 0.9.2 baselines and label the provider change as a
breaking pre-1.0 transition. Run the required native and Miri gates on the
same source that is packaged. Verify package archives in dependency order.
A fresh consumer must use the same published Eunomia scalar in Aequitas
and Hermes and obtain computed SIMD results for that scalar. The Leto
Ops package must compile against the new registry Hermes family before
its own release is considered.
