# Example: Sparse

**Crate**: `hermes-simd`
**Source**: `crates/hermes-simd-examples/examples/book_sparse.rs`

A CSR matrix built from raw parts is wrapped in `ValidatedData`, structurally
checked once at the boundary, and then consumed by `spmv_csr`. This example
exercises the accumulate contract, the `IndexOutOfBounds` rejection of a
malformed matrix, and the masked dense-mask path.

## Source

```rust
{{#include ../../../crates/hermes-simd-examples/examples/book_sparse.rs}}
```

## Output

```text
spmv_csr accumulated y = [8.0, 8.0]
out-of-range column rejected: IndexOutOfBounds
spmv_dense_masked y = [6.0, 9.0]
all sparse assertions passed
```

## What to notice

- `ValidatedData::new(data)` returns `Result<Self, SimdError>`: a value of
  type `ValidatedData<CsrData<'_ , f32>>` is a certificate that every index
  inside is safe to load — the SIMD gather needs no per-element bounds check.

- A column index outside `[0, ncols)` is rejected with the typed
  `SimdError::IndexOutOfBounds` before any SIMD load can read out of bounds.

- `spmv_csr` accumulates `y += A·x`; the initial `y = [1.0, 1.0]` makes the
  contract observable in the result.
