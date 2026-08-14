# ADR 013: Split `SimdKernel` into Role Supertraits

## Status
Proposed

## Context

`SimdKernel` (hermes-simd-core/src/kernel.rs) is a sealed trait carrying ~68
items: 3 associated types (`Vector`, `Mask`, `IndexVector`), 4 associated consts
(`LANE_COUNT`, `LANE_BOUND_CHECK`, `UNROLL_FACTOR`, `SUPPORTS_NT_STORE`), and
~61 methods spanning twelve operation families (load/store, dense arithmetic,
bitwise, comparison, reduction, masked load/store, masked arithmetic,
compress/expand, gather/scatter, mask construction, scan, cross-lane permutes,
adjacent-pair shuffles).

The trait is not over the 500-line target — 622 of 1115 lines are doc comments,
~490 are code — so this is interface segregation, not file length. The problem
is a god trait: a consumer that needs only `sum_reduce` must name the whole
sealed contract, and a backend must override 28–62 items per `(arch, scalar)`
pair even though most families have no native instruction.

## Current Implementation Topology (measured)

- 12 directly written `impl SimdKernel<T> for Arch` blocks, per-backend overrides:
  - `Avx512`: f32 (62 items), f64 (62), f16 (28)
  - `Avx2`: f32 (58), f64 (58), f16 (34)
  - `Neon`: f32 (58), f64 (58), f16 (28)
  - `Scalar`: f32 (52), f64 (30), f16 (28)
- 54 macro-generated blocks via `crate::impl_emulated_kernel!` for the
  `(arch, scalar)` pairs without native registers (Scalar × 13, Neon × 13,
  Avx2/Avx512 emulated × 26, SveArch × 2). These inherit every default except
  the ~15 emulated-loop primitives the macro writes.
- 2 codegen templates in `bin/codegen.rs` (Avx2 ~L331, Avx512 ~L915) that
  regenerate the x86 f32/f64 files; each emits one `impl SimdKernel` block.
- Every method is gated by its own `#[target_feature(enable = "...")]`; no impl
  block carries an impl-level gate.

## Options

1. **Keep the single trait.** No churn, but the god-trait problem persists and
   HS-436's segregation goal is unmet.
2. **Split into role supertraits with `SimdKernel` retained as the aggregate.**
   Define `SimdLoadStore`, `SimdArith`, `SimdMask`, `SimdPermute`,
   `SimdReduce`, `SimdGather`, `SimdBitwise`, `SimdCompare` (see below); each
   carries the methods of its family; `SimdKernel` becomes
   `pub trait SimdKernel<T>: Sealed + Send + Sync + Sized + 'static + SimdLoadStore<T> + SimdArith<T> + ... {}`
   with the three associated types and four consts. All call sites bound on
   `SimdKernel<T>` keep compiling unchanged; narrower bounds become possible.
3. **Role supertraits but hide them behind the aggregate.** Contradictory — the
   point of the split is that consumers can name the narrow contract.

## Decision

Option 2. `SimdKernel` keeps its name, sealing, supertraits
(`Sealed + Send + Sync + Sized + 'static`), associated types, and consts, and
gains role supertraits. Each role trait is declared in a dedicated module under
`kernel/` (or the existing `kernel.rs` file is split into role modules):

- `SimdLoadStore` — `load_aligned`, `load_unaligned`, `store_aligned`,
  `store_unaligned`, `store_streaming`, `stream_write_barrier`, `SUPPORTS_NT_STORE`.
- `SimdArith` — `add`, `sub`, `mul`, `div`, `fmadd`, `neg`, `abs`, `min`,
  `max`, `sqrt`, `recip_sqrt`, `floor`, `ceil`, `round`, `trunc`, `splat`, `zero`.
- `SimdBitwise` — `bitand`, `bitor`, `bitxor`, `bitnot`, `popcount`,
  `horizontal_bitwise_and`, `horizontal_bitwise_or`, `horizontal_bitwise_xor`.
- `SimdCompare` — `cmp_eq`, `cmp_ne`, `cmp_lt`, `cmp_le`, `cmp_gt`, `cmp_ge`,
  `blend`.
- `SimdReduce` — `sum_reduce`, `min_reduce`, `max_reduce`,
  `masked_sum_reduce`.
- `SimdMask` — `mask_from_bools`, `mask_from_bitmask`, `leading_k_mask`,
  `mask_to_vector`, `vector_to_mask`, `mask_to_bitmask`, `masked_load_unaligned`,
  `masked_store_unaligned`, `masked_add`, `masked_mul`, `masked_fmadd`,
  `compress`, `expand`.
- `SimdGather` — `gather`, `gather_masked`, `scatter`, `scatter_masked`.
- `SimdPermute` — `reverse`, `interleave`, `deinterleave`, `swap_adjacent`,
  `dup_even`, `dup_odd`, `fmaddsub`, `fmsubadd`, `scan_vector`.

Method-to-role assignment is guided by the default-body dependency graph (a
default may only call methods of the same trait or a supertrait): `masked_*`
defaults depend on `blend`+`mask_to_vector` (hence live in `SimdMask` with
`SimdCompare`/`SimdArith` supertraits), `scan_vector` depends only on
load/store (hence `SimdPermute`, which needs `SimdLoadStore`). The role traits
are not sealed individually — sealing stays on the aggregate — but their impl
set is closed in practice because every backend implements all roles for each
scalar.

### Impl-block cost (measured)

Splitting multiplies impl blocks: 12 direct + 54 macro + 2 templates become,
per `(arch, scalar)`, up to 8 role impls instead of 1 — roughly 12 × 8 direct
blocks and 54 × 8 macro-emitted blocks. Two facts bound this cost:

1. **Method bodies do not move.** Each role impl is the same method definitions
   as today, re-grouped under 8 headers. Lines of code are unchanged; the count
   of `#[target_feature]` attributes is unchanged (per-method gating).
2. **The macro and codegen templates centralize the churn.** The 54
   macro-generated blocks come from one `macro_rules!` in lib.rs — editing that
   single body re-emits all 54 × 8. The 2 codegen templates likewise re-emit
   the x86 f32/f64 files. Only the 12 hand-written x86-f16/NEON/Scalar files
   need manual re-grouping.

The alternative — a `SimdKernelBase` trait carrying the three associated types
and four consts, with `SimdKernel` and every role trait as its subtraits — was
rejected: it duplicates the associated-type declarations across 8 traits and
makes the aggregate depend on the base rather than the roles, complicating the
dependency graph without reducing the impl-block count.

## Consequences

- **Call-site compatibility:** every existing bound `S: SimdKernel<T>` keeps
  working because the aggregate retains all role supertraits; the five backends
  and their dispatch code need no change beyond re-grouping impl blocks.
- **Public surface:** 8 new public traits re-exported from `hermes-simd-core`
  and the `hermes-simd` facade. `#![deny(missing_docs)]` applies to them.
  `cargo-semver-checks` sees the new traits as additive; the existing
  `SimdKernel` surface is preserved, so this is a [minor] additive contract
  change despite the [arch] class.
- **Codegen/freshness:** the codegen templates gain role-split output; the
  checked-in x86 f32/f64 files are regenerated from templates, so the template
  and files must move in one change (regenerate, then commit the diff).
- **Narrower bounds:** consumers needing only reduction can now bound on
  `SimdReduce<T>` + `SimdLoadStore<T>` instead of the full aggregate — the
  segregation goal.
- **Impl-block growth:** ~96 hand-written role blocks + ~432 macro-emitted
  blocks, centralized per the two bullet points above. Each role module must
  stay under the 500-line target; the doc-heavy method documentation moves with
  its methods.

## Evidence / Verification Plan

- `cargo build --workspace --all-targets` green; clippy `-D warnings` clean.
- `cargo nextest run` green (existing value-semantic suite unchanged).
- `cargo bloat`/codegen comparison on a representative kernel (e.g. `sum`)
  showing no size growth — role supertraits are a zero-cost abstraction,
  monomorphized like the aggregate.
- `cargo doc` warning-clean; ADR index regenerated via
  `python D:\atlas\scripts\adr-index.py generate`.

## References

- HS-436 backlog item (`backlog.md`).
- `crates/hermes-simd-core/src/kernel.rs` — current trait.
- `crates/hermes-simd-core/src/kernel_helpers.rs` — generic scalar-emulated
  defaults the role defaults delegate to.
- `crates/hermes-simd-intrinsics/src/bin/codegen.rs` — Avx2/Avx512 templates.
- `crates/hermes-simd-intrinsics/src/lib.rs` — `impl_emulated_kernel!`.
