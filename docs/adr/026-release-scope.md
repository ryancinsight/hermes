# ADR 026: Hermes 1.0 Release Scope

## Status

Accepted (2026-11-13)

## Context

Hermes is the Atlas SIMD substrate: lane-parallel CPU kernels, scalar
fallbacks, ISA dispatch, packed-lane representation and zero-copy SIMD views
(README, "Atlas Compute Boundaries"). Its five library crates carry
`crate-<name>-v0.9.2` release tags (`hermes-simd-macros` `v0.9.0`);
`hermes-simd-examples` and `hermes-simd-benches` are `publish = false`. The roadmap to 1.0 was implicit.
This record states what 1.0 is and is not, area by area, so the milestone
ladder at the top of `backlog.md` can be checked against it.

Classification uses three words: **delivered** (present on the default branch,
with the witness path given), **in scope** (required before 1.0 and served by
a ladder exit check), **non-goal** (not part of 1.0).

## Decision

Hermes 1.0 is the delivered surface below plus one in-scope capability: a
fused, runtime-dispatched scaled CSR product `y += alpha * A * x`. Every other
open direction is a non-goal for 1.0. A non-goal that is additive (a new
operation family, a new backend behind an existing marker) can land in a 1.x
minor without a major.

### `hermes-simd-core`

| Area | Status | Witness |
| --- | --- | --- |
| `Scalar` element traits, `SimdArch` markers, alignment and execution-mode typestates | delivered | `src/scalar.rs`, `src/arch.rs`, `src/align.rs`, `src/execution.rs` |
| Typestate `SimdView`, `Vector`, `Simd` capability, exact-width chunks | delivered | `src/view/`, `src/iter/` |
| `ComputeView` over dense, masked, sparse and bitboard views | delivered | `src/compute.rs` |
| `SimdKernel` aggregate, operation-family facets, `BackendKernel` seam | delivered | `src/kernel/` (ADR 013) |
| Facets: arithmetic, bitwise, lane comparison, gather, load/store, mask, reduce | delivered | `src/kernel/roles/` |
| Cross-lane and pair permutes (reverse, interleave, const-generic pair decimation) | delivered | `src/kernel/roles/permute.rs` (ADR 024) |
| Masks, masked operations, masked compress/expand on views | delivered | `src/mask.rs`, `src/view/masked_compaction.rs` |
| Reduction, element and scan strategy ZSTs | delivered | `src/ops/` |
| Gather and scatter on views | delivered | `src/view/gather.rs`, `src/view/scatter.rs` |
| Const-generic tiled dot, GEMM and GEMV | delivered | `src/tiling/` |
| Sparse formats (CSR, SELL-p, blocked COO, dense-with-mask) and SpMV | delivered | `src/sparse/spmv/` (ADR 019) |
| Fused scaled CSR SpMV, `y += alpha * A * x` in one pass | in scope | ladder exit checks |
| Dense, packed 4-bit and sparse copy-on-write containers | delivered | `src/cow/`, `src/sparse/cow/` |
| N-D tensor views: layouts, strides, rows, transpose, COW | delivered | `src/tensor/` |
| Tensor softmax, LayerNorm and attention layers | non-goal | absent from `src/tensor/`; consumer-layer operations |
| `AlignedVec` and NUMA placement through Themis and Mnemosyne | delivered | `src/vec/`, `src/numa/` (ADR 010, ADR 023) |
| `BitBoardView` | delivered | `src/bitboard.rs` (ADR 011) |
| Zero-copy view casts between POD element types | delivered | `src/view/casts.rs` |

### `hermes-simd-intrinsics`

| Area | Status | Witness |
| --- | --- | --- |
| `Scalar`, `Avx2`, `Avx512` and `Neon` backends (f16, f32, f64) | delivered | `src/scalar/`, `src/x86_64/`, `src/aarch64/` |
| `SveArch` as a value-semantic emulated backend | delivered | `src/aarch64/sve.rs` (ADR 014) |
| Native SVE intrinsic backend | non-goal | blocked on stable scalable SVE vector types |
| AVX-512 VNNI and AVX-VNNI tile multipliers | delivered | `src/x86_64/avx512_tiling.rs`, `src/x86_64/avx_vnni_tiling.rs` |
| Packed sub-byte unpacking (INT4 to INT8, BF8 and BF4 to BF16) | delivered | `src/x86_64/avx512_tiling.rs` |
| Intel AMX engine behind the permission probe | delivered | `src/x86_64/amx/` (ADR 002, ADR 003, ADR 012) |
| AMX runtime validation on silicon as a release gate | non-goal | no requestable runner admits `XTILEDATA` (README, "Intel AMX status") |
| SWAR bitboard backends (Kogge-Stone, Hyperbola, Magic, hybrid) | delivered | `src/bitboard/` |
| Backend-owned read prefetch | non-goal | rejected by ADR 020 |

### `hermes-simd-types` and `hermes-simd-macros`

| Area | Status | Witness |
| --- | --- | --- |
| Per-backend register aliases and compile-time `PreferredArch` | delivered | `hermes-simd-types/src/lib.rs` |
| `#[runtime_dispatch]` attribute macro | delivered | `hermes-simd-macros/src/dispatch.rs` |
| `#[derive(SparseData)]` | delivered | `hermes-simd-macros/src/sparse_data.rs` |

### `hermes-simd` (facade)

| Area | Status | Witness |
| --- | --- | --- |
| `SimdOps` and runtime-dispatched sum, dot, min/max, argmin/argmax, absolute reductions, AXPY, scale, binary, masked, modular, popcount, real-interleave, interleaved-complex, GEMM, GEMV and sparse functions | delivered | `src/dispatch/` |
| Eunomia precision-ladder re-export (`Bf4` through `F64`, `I8` through `I32`) | delivered | `src/lib.rs` `pub use eunomia` |
| `vectorize` and `LaneKernel` consumer entry | delivered | `src/vectorize.rs` (ADR 016, ADR 017, ADR 018) |
| `TargetId` forced dispatch for tests and benchmarks | delivered | `src/target.rs` |
| `AdaptiveDispatcher` and CPU capability queries | delivered | `src/dispatcher.rs`, `src/cpu.rs` |
| Tile GEMM ladder and INT4 unpack | delivered | `src/tile_matmul/` |
| Rook, bishop and queen attack functions | delivered | `src/attacks.rs` |
| `no_std` build with the `std` feature off | delivered | `src/lib.rs` `cfg_attr`; CI `cargo check --workspace --no-default-features` |
| Slice-level public facades for compress-store, comparison predicates, conversions and bitwise operations | non-goal | admitted only on consumer demand (README) |
| Crypto and hash primitives | non-goal | outside Hermes' SIMD scope (backlog coverage map) |

### Unpublished members and documentation

| Area | Status | Witness |
| --- | --- | --- |
| Runnable examples (`hermes-simd-examples`) | delivered | `crates/hermes-simd-examples/examples/` |
| Criterion suite with a regression baseline (`hermes-simd-benches`) | delivered | `crates/hermes-simd-benches/src/regression.rs` |
| mdBook guide | delivered | `docs/book/` |
| AVX-512 native timing as a release gate | non-goal | hosted AVX-512 silicon is opportunistic (README, "Intel AMX status") |

Where no source fixes an area's status, this record takes the recommended
option, and that choice is the user's veto point: the tensor-layer, AMX,
AVX-512 timing and demand-gated facade rows.

## Rejected alternatives

- **One milestone per workspace package, closed by a whole-package test run.**
  A package suite passes before and after any feature, so it closes nothing.
- **Gate 1.0 on AMX or AVX-512 real-silicon evidence.** The hardware cannot be
  requested, so the release would wait on runner luck rather than on work.
- **Admit every Highway-gap family before 1.0.** No consumer names them, and
  each is an additive minor after 1.0.

## Consequences

- The ladder at the top of `backlog.md` holds one milestone, planned 1.0.0,
  whose exit checks name the scaled CSR tests and a self-checking example.
- Release stays authority-gated; closing the milestone files a release item
  and publishes nothing by itself.

## Overturning evidence

- A consumer names a non-goal family or layer as blocking its own milestone.
- Stable Rust gains scalable SVE vector types.
- A requestable runner admits AMX tile data.
- A breaking public change is found to be required after 1.0 is planned.
