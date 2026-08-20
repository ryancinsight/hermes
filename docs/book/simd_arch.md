# 3. SimdArch and SIMD kernel facets

The kernel layer has an architecture seam, one implementation seam, and
operation-family facets with a strict division of labor:

- `SimdArch` — architecture *constants*, no scalar type. Implemented by
  zero-sized marker types.
- `BackendKernel<T>` — the sealed implementation seam retaining the canonical
  backend defaults and native overrides for a concrete scalar `T`.
- `SimdStorage<T>` — the shared register and lane contract used by every
  operation family.
- `SimdArith<T>`, `SimdLoadStore<T>`, `SimdReduce<T>`, and the other role
  facets — consumer-facing contracts for one operation family.
- `SimdKernel<T>` — the aggregate bound for consumers that use several facets.

Together they give a generic kernel its monomorphization: a function written
`f<T, A: SimdArch + SimdReduce<T>>` compiles once per `(T, A)` pair into
machine code identical to a hand-written specialization. The facets forward
to the one `BackendKernel<T>` body; they do not add dynamic dispatch or a
second implementation.

## `SimdArch`: the architecture marker

`SimdArch` is implemented by ZSTs such as `Scalar`, `Neon`, `Avx2`, `Avx512`,
and `SveArch`. It carries compile-time facts about the ISA:

- `NAME` — the human-readable ISA name.
- `REGISTER_WIDTH_BITS` — 0 for `Scalar`, 128 for `Neon`, 256 for `Avx2`,
  512 for `Avx512`.
- `ISA_FAMILY` — an `IsaFamily` classification (x86 / AArch64 / RISC-V /
  scalar) used for compile-time routing and documentation, never for code
  generation.
- `FMA_THROUGHPUT_HINT` — the suggested `TILE_M` for `tiled_dot` (4 for
  AVX2/NEON, 8 for AVX-512, 1 for scalar); a hint for kernel shape, not a
  constraint.
- `is_runtime_supported()` — whether the current host may execute this
  architecture's native instructions from safe wrappers. Emulated backends
  (including `SveArch` on hosts without scalable SVE) return `true`; native
  ISA backends include the OS-enabled register-state checks behind the
  platform feature probe.

The support check is the load-bearing safety gate. Kernel methods are
`#[target_feature]`-gated, so calling one on a host lacking the feature is
undefined behavior. hermes-simd turns "the host supports `Arch`" into an
invariant of *holding a value of the type*: view constructors and the
runtime-dispatch wrappers check `is_runtime_supported()` before any gated code
can run. This is the `HS-405` invariant — possessing a `SimdView` or entering a
dispatched kernel *proves* the kernels it would call are executable on this
host.

## Backend seam and operation-family facets

`BackendKernel<T: Scalar>` retains the primitive implementation surface for
each backend and element type `T`. Consumers select the narrowest public facet
that covers the operation they use:

- **`SimdStorage<T>`**: associated `Vector`, `Mask`, and `IndexVector` register
  types, plus `LANE_COUNT`
  — the number of `T` lanes in one vector register (e.g. 8 for `f32` on AVX2,
  16 on AVX-512).
- **`SimdLoadStore<T>`**: `load_aligned`, `load_unaligned`, `store_aligned`,
  `store_unaligned`, plus masked `masked_load_unaligned` /
  `masked_store_unaligned` and `masked_fmadd` / `masked_add` / `masked_mul`.
  Masked operations follow AVX-512 merge-masking semantics: lanes where the
  mask is inactive are taken from `src`.
- **`SimdArith<T>`**: `splat`, `add`, `sub`, `mul`, and `fmadd`.
- **`SimdReduce<T>`**: horizontal reductions such as `sum_reduce` and
  `masked_sum_reduce`.
- **`SimdGather<T>` and `SimdPermute<T>`**: indexed movement and lane
  rearrangement such as `gather`, `compress`, and `expand`.
- **`SimdMask<T>`**: mask construction and conversion such as
  `mask_from_bools`, `leading_k_mask`, and `mask_from_bitmask`.
- **Other facets**: `SimdBitwise<T>` and `SimdCompare<T>` own their respective
  operation families.

`SimdKernel<T>` is the aggregate of these facets for consumers that genuinely
need several families. It is not the implementation home: all defaults and
native overrides remain in `BackendKernel<T>`, and `SimdStorage<T>` is the
single source for shared associated storage metadata.

The operation-family split keeps consumer bounds honest. A gather-only view
does not require reductions, while a multi-operation algorithm can retain the
aggregate `SimdKernel<T>` bound. Both forms monomorphize to the same backend
code.

Every method has an implementation on the `Scalar` backend (a plain loop) and
native overrides where the ISA provides the instruction. The architecture
mapping is documented per method — for example, `masked_add` is
`_mm512_mask_add_ps` on AVX-512, `_mm256_blendv_ps` on AVX2, `vbslq_f32` on
NEON, and a `loop + if` on the scalar fallback. A method with no native
instruction keeps its defaulted scalar-emulated implementation; new methods
join the backend seam once, and every facet forwards to that canonical body.

- **Load/store**: `load_aligned`, `load_unaligned`, `store_aligned`,
  `store_unaligned`.
- **Data movement**: `compress` / `expand` between contiguous storage and
  selected lanes, and indexed `gather` / `gather_masked`.

## Backend implementations and `Scalar`

The concrete architecture impls live behind the architecture markers
(`Avx2`, `Avx512`, `Neon`, `SveArch`, `Scalar`) from `hermes_simd_intrinsics`,
each compiled under its required `#[target_feature]`. `Scalar` is the 
always-available reference backend: no ISA features required, so it compiles
and runs on every target and serves as the differential oracle the native
backends are tested against.

```rust
extern crate hermes_simd_core;
extern crate hermes_simd_intrinsics;

use hermes_simd_intrinsics::Scalar;
use hermes_simd_core::kernel::{SimdArith, SimdReduce, SimdStorage};

// SAFETY: `Scalar` requires no special ISA features.
let splat4 = unsafe { <Scalar as SimdArith<f32>>::splat(1.0_f32) };
let total = unsafe { <Scalar as SimdReduce<f32>>::sum_reduce(splat4) };
assert_eq!(total, <Scalar as SimdStorage<f32>>::LANE_COUNT as f32);
```

## Bounds: lanes and buffers

`MAX_SIMD_LANES` is 64 — the scalar-fallback stack buffers used by defaulted
methods (`scan_vector`, `swap_adjacent`, `dup_even`/`dup_odd`, and the scalar
emulations) are `[MaybeUninit<T>; 64]`. A backend whose `LANE_COUNT` would
exceed the bound is rejected at compile time by `LANE_BOUND_CHECK`; a future
wider backend fails to build rather than silently overflowing the stack. The
current workspace maximum is AVX-512 `i8` at 64 lanes.

## Why the seam is sealed

`SimdArch`, `BackendKernel`, and the public facets are sealed: the implementor
set is closed to the workspace, and `Scalar` implementations are extended
upstream in `hermes_simd_intrinsics`, never downstream. The seal is what lets
the kernel layer promise that every defaulted method has a correct scalar
fallback and that new backends inherit the full family — these traits are
closed contracts, not open extension points.

## What to notice

- **Constants belong to the architecture, operations belong to facets.**
  `SimdArch` is type-independent; `BackendKernel<T>` and each public facet bind
  a scalar type. A kernel generic over both compiles once per `(T, A)`.
- **Unsafe is contained.** The `unsafe` surface is the `#[target_feature]`
  kernel methods; the support check at construction time makes "the host can
  execute this architecture" a property of holding the value.
- **The scalar backend is the oracle.** Every native path is differentially
  tested against it, and the two must agree within the derived numerical
  bounds (reduction order is the one place bitwise equality is not promised).
