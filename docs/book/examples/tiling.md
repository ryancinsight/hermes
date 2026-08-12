# Example: Tiling

**Crate**: `hermes-simd`
**Source**: `crates/hermes-simd-examples/examples/book_tiling.rs`

`tiled_dot` and `gemv` both hold `TILE_M` independent register accumulators —
the first to saturate FMA throughput, the second to reuse each loaded `x`
vector across `TILE_M` rows. This example runs both on `SimdView` with the
`Scalar` reference backend (so it compiles and runs on any host) and
cross-checks against the plain facade `dot`.

## Source

```rust
{{#include ../../../crates/hermes-simd-examples/examples/book_tiling.rs}}
```

## Output

```text
tiled_dot(TILE_M=4) = 72
dot                  = 72
gemv y = [3.0, 14.0, 1.0]
gemv again accumulates: y = [6.0, 28.0, 2.0]
```

## What to notice

- `tiled_dot::<f32, Scalar, Unaligned, 4>` processes `4 × LANE_COUNT` elements
  per iteration through four independent accumulators; with the `Scalar`
  backend, `LANE_COUNT == 1`, so this is a fully general (if unvectorized)
  reference the ISA backends are tested against.

- `gemv` **accumulates**: `y += A·x`, so the second call doubles the first
  result instead of overwriting it. Zero `y` first to get `y = A·x`.

- The matrix contract is explicit: row-major `nrows × ncols` storage with
  `x.len() >= ncols` and `y.len() >= nrows`; a length mismatch on `dot` is a
  typed `SimdError`, not a silent truncation.
