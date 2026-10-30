# hermes-simd-core

Core abstractions for [Hermes](https://github.com/ryancinsight/hermes), the
Atlas CPU SIMD substrate. This crate defines *what a SIMD operation is* — the
kernel trait, the typestate views, and the container and layout vocabulary —
without containing any architecture-specific intrinsics. Those live in
`hermes-simd-intrinsics`; most consumers depend on the `hermes-simd` facade
rather than on this crate directly.

## Module organization

| Module | Contents |
|--------|----------|
| `arch` | `SimdArch` marker trait with architecture constants |
| `align` | `Alignment`, `Aligned<N>`, `Unaligned` typestates |
| `execution` | `ExecutionMode`, `Unmasked`, `Masked` ZSTs |
| `kernel` | `SimdKernel<T>` and operation-family facets — the SIMD contracts |
| `scalar` | `Scalar` sealed element trait |
| `mask` | `BitMask<N>` bit-packed lane mask |
| `ops` | `ReductionOp<T>`, `ElementOp<T>` ZST strategies |
| `view` | `SimdView`, `SimdError` — safe typed slice views |
| `tiling` | Const-generic tiled dot product and `TilingPolicy` |
| `sparse` | `SparseView`, format ZSTs, data structs, SpMV kernels |
| `vec` | `AlignedVec` — heap-allocated aligned vector |
| `cow` | `SimdCow` — SIMD-aware copy-on-write |
| `tensor` | N-D tensor views, GEMM, softmax, LayerNorm, attention |

Alignment, execution mode, and reference mutability are compile-time type
parameters with zero layout overhead.

## Tensor storage contract

`TensorView` validates logical cardinality and every reachable strided offset.
For a nonempty shape, `1 + sum((shape[k] - 1) * stride[k])` must fit `usize`
and the backing slice. Any zero axis makes the view empty; rank zero contains
one scalar. Invalid layouts return `TensorError::ShapeMismatch` without relying
on debug overflow checks.

Explicit strides may repeat storage positions. Row/column projections preserve
those strides and values; mutable element access requires an exclusive borrow.
`as_slice` exposes physical storage order, including any padding. Contiguous
row iteration and rank-one SIMD promotion instead honor the logical extent;
SIMD promotion rejects empty or noncontiguous views.

## Feature flags

| Feature | Description |
|---------|-------------|
| `std` (default) | Standard library support; without it the crate is `no_std` |
| `mnemosyne-memory` (default) | Routes aligned allocation through the Mnemosyne allocator |
| `libnuma` | Linux NUMA affinity and residency probes via libnuma (links `-lnuma`) |

## Documentation

- API reference: [docs.rs/hermes-simd-core](https://docs.rs/hermes-simd-core)
- Workspace overview: the
  [repository README](https://github.com/ryancinsight/hermes#readme)

## License

MIT OR Apache-2.0
