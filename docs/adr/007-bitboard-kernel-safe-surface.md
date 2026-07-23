# ADR 007: Make `BitBoardKernel` a Safe Trait

## Context

An audit of the `unsafe` blocks in `hermes-simd-core` (HS-406) examined why each
one was unsafe before documenting it. `BitBoardKernel` declared all three attack
generators as `unsafe fn` with the obligation:

> Caller must ensure target feature flags are active.

That obligation does not match any implementation:

| Backend | Basis | Caller obligation |
|---|---|---|
| `Swar`, `Hyperbola`, `Magic`, `HybridSwarMagic` | plain integer arithmetic and bounds-checked table lookups | none |
| `KoggeStone` | AVX-512 / AVX2 / NEON fills selected inside `is_x86_feature_detected!` (or `cfg!(target_feature)` under no-std) | none — the ISA choice is internal |

No implementation reads memory through an unchecked index: the magic and ray
tables use ordinary indexing, so an out-of-range square panics rather than
reading out of bounds. This is the opposite of `SimdKernel`, whose operations
are genuinely `#[target_feature]`-gated and must stay `unsafe`.

The marker was therefore load-bearing in the wrong direction. It forced every
caller — including `BitBoardView`'s own safe methods and the crate's tests and
examples — to open an `unsafe` block around a call that cannot violate memory
safety, which trains readers to treat `unsafe` as noise and hides the blocks
that do carry an obligation.

## Options

1. **Document the existing `unsafe`.** Cheapest, and preserves the API. It also
   records a precondition that no implementation has, so every future reader
   inherits a false obligation and the audit's premise stays wrong.
2. **Add a runtime-support probe to the trait**, mirroring
   `SimdArch::is_runtime_supported`, and guard construction as HS-405 did for
   `SimdView`. This would be correct if the backends had an ISA precondition.
   They do not, so it adds a probe that can never fail.
3. **Make the methods safe.** Matches what the implementations actually
   guarantee and pushes the remaining `unsafe` down to the ISA fills inside
   `KoggeStone`, where the target-feature argument is real and local.

## Decision

Option 3. `rook_attacks`, `bishop_attacks`, and `queen_attacks` become safe
`fn`s. `KoggeStone` wraps each `#[target_feature]` fill in its own `unsafe`
block with a `SAFETY` comment naming the probe that guards it, so the obligation
is stated where it exists rather than propagated to every caller.

The methods gain a `# Panics` section for an out-of-range square, which is the
real precondition on their inputs and was previously undocumented.

## Consequences

- Five `unsafe` blocks disappear from `bitboard.rs`, leaving two — the raw
  pointer reborrows in `as_slice`/`as_slice_mut` — each now carrying a `SAFETY`
  comment. That module is fully documented.
- `cargo-semver-checks` classifies this as **major**
  (`trait_method_unsafe_removed`): an existing `unsafe fn` implementation no
  longer matches the trait. Being pre-1.0, it ships in a minor release under
  **Breaking**, per this changelog's stated convention.
- No implementor exists outside this workspace; a stack-wide search found
  `BitBoardKernel` referenced only within `hermes`.
- Callers that wrapped these calls in `unsafe` now get an `unused_unsafe`
  warning, which is an error under the workspace's `-D warnings`. The migration
  is to delete the block.
- Making the trait safe forecloses adding an ISA precondition later without
  another breaking change. That is the intended constraint: a backend needing
  one should probe internally, as `KoggeStone` does, keeping the choice of
  instruction set an implementation detail rather than a caller obligation.
