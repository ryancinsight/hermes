# 9. Sparse

Sparse kernels multiply a matrix in a sparse format by a dense vector
(`y += A·x`). The challenge is structural: the data is scattered, so a SIMD
kernel either gathers non-contiguously or restructures the layout so vector
loads become legal — and every index read from a hostile or malformed matrix
must be bounds-checked before it is used to load.

`hermes_simd` answers both with a two-part design: a **validated-data
typestate** that turns an unchecked sparse matrix into a checked one exactly
once at the boundary, and **format-specific kernels** that vectorize each
layout on its own terms.

## Formats

Four formats cover the sparse surface, each chosen for how its entries map to
vectors:

- `CsrData` — Compressed Sparse Row: `values`, `col_indices`, and `row_ptr`,
  the canonical sparse exchange format. Its kernel vectorizes by gathering
  the row's non-zeros or by segmented reduction across the packed value
  array.
- `SellPData` — SELL-P: slices the matrix into chunks of `C` rows and pads
  the ragged row tails, so each chunk is a rectangular slab that can be
  loaded with regular vectors (the `C` is the const-generic chunk width).
- `BlockedCooData<BM, BN>` — Blocked COO: stores dense `BM×BN` tiles. Each
  tile multiplies as a dense register block, so the kernel is a normal dense
  GEMV over tiles — no gather at all.
- `DenseWithMaskData` — a dense rectangular matrix plus a bit-packed
  per-element mask (`PackedMask`, one bit per element — packed once at
  construction). The kernel uses masked loads (Chapter 6), not gathers,
  keeping the memory access pattern uniform; each lane window's mask bits are
  read straight out of the packed words, never converted from bools per call.
  Rows shorter than half a register stay scalar to avoid fixed mask and
  reduction setup. Wider remainders use exact-prefix masked loads, so the
  kernel never requires memory beyond the logical row tail.

The types are split so the shape and the storage are explicit: `SparseShape`
carries the logical dimensions, the `*Data` types carry the raw arrays, and
`SparseView` is the borrowed view a kernel iterates.

## The validated-data boundary

Raw sparse arrays are **not** directly consumable by a kernel. They are
wrapped first:

```rust,ignore
use hermes_simd::{CsrData, ValidatedData, spmv_csr};

let values = [1.0f32, 1.0, 1.0];
let col_indices = [0i32, 1, 2];
let row_ptr = [0i32, 1, 2, 3];
let data = CsrData::new(&values[..], &col_indices[..], &row_ptr[..], 3, 3);

let x = [5.0f32, 7.0, 11.0];
let mut y = [0.0f32; 3];
spmv_csr::<f32>(ValidatedData::new(data).expect("valid csr"), &x, &mut y);
```

`ValidatedData::new(inner)` returns `Result<Self, SimdError>`: it checks every
structural property the SIMD kernel will rely on — column indices in
`[0, ncols)`, row pointers monotone and within `values`, `row_ptr[0] == 0`,
counts consistent with `nrows` — and returns `SimdError::IndexOutOfBounds` on
the first violation. The `ValidatedData` type is opaque: the only way to
obtain one is through this check, so a value of that type *is* a certificate
that every index inside it is safe to load.

This is deliberately a check-once, use-many design. The alternative — bounds-
checking every gather inside the kernel — pays a branch per element on the
hot path. Validation hoists the entire cost to the boundary, and the type
system makes it impossible to skip. The one exception is the dense-with-mask
format: its storage is a full rectangular array, so it has no structural index
hazard and the facade accepts it directly.

The dense vector side keeps the slice contract: `spmv_csr` requires
`x.len() >= ncols` and `y.len() >= nrows` (a mismatch panics at the boundary,
because the checked `ValidatedData` has already discharged the matrix
interior). The operation **accumulates** `y += A·x`, matching the BLAS
convention; callers zero `y` to get `y = A·x`.

## The kernel family

Each format has one entry point, monomorphized by its layout parameters:

- `spmv_csr::<T>(data: ValidatedData<CsrData>, x, y)` — gather-based CSR.
- `spmv_sellp::<T, C>(data: ValidatedData<SellPData<C>>, x, y)` — chunked
  SELL-P, const-generic row-chunk width `C`.
- `spmv_bcoo::<T, BM, BN>(data: ValidatedData<BlockedCooData<BM, BN>>, x, y)`
  — blocked COO, const-generic dense tile shape `BM×BN`.
- `spmv_dense_masked::<T>(data: DenseWithMaskData, x, y)` — masked-load dense;
  the only format the facade accepts unwrapped (dense storage, no index
  hazard).

`T` is the `SimdOps` element type, so every kernel monomorphizes per scalar
type and architecture like the dense kernels.

The blocked and masked forms are where sparse and dense meet: `spmv_bcoo`'s
inner loop is a register-blocked GEMV over tiles (the tiling machinery from
Chapter 8), and `spmv_dense_masked` uses the masked arithmetic from Chapter 6
so the mask bits, not a gather, select the live lanes. Each format's kernel is
owned by a private leaf under the public `sparse::spmv` trait module; this keeps
the four traversal strategies independent without changing the public entry
points. The worked example in
[the sparse example](examples/sparse.md) exercises CSR accumulation, the
`IndexOutOfBounds` rejection, and a dense-masked identity.

## What to notice

- **Validation is a type, not a call.** `ValidatedData` exists only through
  the structural check; holding it proves the interior indices are safe, so
  the kernel needs no per-element bounds branches.
- **`IndexOutOfBounds` is a typed error, not a panic or a sanitized fallback.**
  Malformed input is rejected loudly at the boundary before any SIMD load can
  read out of bounds.
- **Layout drives the kernel.** Gather, padded slabs, dense tiles, and masked
  loads are four different answers to "how do I make scattered data loadable",
  selected by which format the caller already has.
- **The accumulate contract is explicit.** `y += A·x` means composing kernels
  over the same `y` is safe, and zeroing `y` selects the non-accumulating form.
