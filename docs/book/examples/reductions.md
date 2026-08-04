# Example: Reductions

**Crate**: `hermes-simd`
**Source**: `crates/hermes-simd-examples/examples/book_reductions.rs`

`sum`, `min`, `max`, `abs_sum`, `argmin`, and `argmax` all share the same
runtime-dispatch path.  This example builds a small dataset with known
extremes and verifies each reduction against the expected value.

## Source

```rust
{{#include ../../../crates/hermes-simd-examples/examples/book_reductions.rs}}
```

## Output

```text
data  = [-5.0, 3.0, -1.0, 7.0, 2.0, -9.0, 4.0, 0.0]
sum   = 1
min   = -9
max   = 7
l1    = 31
argmin = (5, -9)
argmax = (3, 7)
all reduction assertions passed
```

## What to notice

- `sum` returns `T::ZERO` for an empty slice; `min` returns `T::MAX_VALUE`
  and `max` returns `T::MIN_VALUE`.  The sentinel values follow the
  mathematical convention for an empty fold.

- `argmin` and `argmax` return `None` for an empty slice and `None` when
  the data contains NaN.  This makes the empty-slice contract explicit
  rather than returning a sentinel index.

- `abs_sum` computes the L1 norm Σ|xᵢ|.  For the sample data:
  |−5| + 3 + |−1| + 7 + 2 + |−9| + 4 + 0 = 31.

- All six functions dispatch through the same ISA-selection chain as `dot`
  and `sum` — the FMA path accelerates fused reduction steps where available.
