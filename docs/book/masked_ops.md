# 6. Masked Operations

A masked operation computes on a *predicated subset* of the lanes: each lane
carries a boolean that says whether it participates. The result lanes that do
not participate follow merge-masking semantics — they take their value from
the `src` operand rather than being written arbitrarily. Masked execution is
what lets a kernel handle a ragged tail, a gather that may not find a value,
or a data-dependent selection without leaving the vector domain.

## Two surfaces, one mask model

Masked behavior appears in two forms:

**The slice facade** takes a plain `&[bool]` mask. `masked_sum(data, mask)`,
`masked_dot(a, b, mask)`, and `masked_add(a, b, mask, out)` accept a boolean
mask of the same length as the data and compute only over the `true` lanes:

```rust,ignore
use hermes_simd::{masked_sum, masked_add};

let data = [1.0_f32, 2.0, 3.0, 4.0, 5.0];
let mask = [true, false, true, false, true];
let s = masked_sum::<f32>(&data, &mask);   // 1.0 + 3.0 + 5.0 = 9.0
```

**The `SimdView` surface** parameterizes the view itself: `Mode: ExecutionMode`
is a type parameter, defaulting to `Unmasked`. The `Masked` marker enables the
predicated methods, and the mask operand is the *architecture-native* `Mask`
register type — the same masks the kernels use internally. The `Mode` marker
is a ZST, so `SimdView<'_, T, Arch, Align, Masked>` carries zero runtime state;
the compiler sees the mode at monomorphization time and eliminates any dead
branch over it.

`SimdView::select(mask, other)` performs the lane-wise conditional merge
`out[i] = if mask[i] { self[i] } else { other[i] }` into a fresh aligned
vector, and `masked_negate(mask)` negates only the selected lanes.

## What a mask compiles to

The `ExecutionMode` and mask types are not abstract — they bind to the host
backend. The masked seams map to hardware predicates:

| Backend | Masked operation |
|---|---|
| AVX-512 | mask registers (`__mmask16`/`__mmask8`), `_mm512_mask_add_ps` |
| AVX2 | blend masks (`__m256`), `_mm256_blendv_ps` |
| NEON | `vbslq_f32` |
| Scalar / emulated | `[bool; N]`, loop + `if` |

`mask_from_bools` builds a native mask from a boolean slice, and the
`mask_to_vector` / `vector_to_mask` pair converts between masks and vectors of
all-ones/zero lanes for backends whose blend operates on vector operands.
The round-trip is pinned by property tests.

## Where masked paths are mandatory

Masked memory has two contracts. `masked_load_unaligned` and
`masked_store_unaligned` require a pointer valid for a complete vector, even
when the mask selects fewer lanes. `masked_load_partial` and
`masked_store_partial` instead accept an exact accessible prefix: they touch
only active lanes below `valid_lanes`; a partial load merges inactive lanes
from `src`. This distinction permits a zero mask at an inaccessible pointer
and permits a live tail at an allocation or page boundary without staging a
full vector.

AVX2 f32/f64 map the partial seam to `vmaskmovps`/`vmaskmovpd`; AVX-512 f32/f64
use native mask-register loads and stores. Scalar, NEON, the emulated SVE
backend, and x86 F16 inherit a default that dereferences active lanes only and
may stage the register through a provider-internal buffer. Dense reduction,
view, AXPY, GEMM/GEMV, and scatter-source tails use this seam where their
operation supports a masked tail. Reductions such as `Product` that retain a
distinct scalar ordering keep their documented scalar remainder.

## Error and edge behavior

Masked facade operations validate their shapes like their unmasked siblings:
`masked_dot` returns `SimdError::LengthMismatch` on unequal lengths,
`masked_add` requires `a.len() == b.len() == mask.len()` and an output at
least as long, and `select` returns `LengthMismatch` for unequal operands and
`InsufficientOutputLength` when the mask is shorter than the data. A masked
sum over data with an all-false mask is `T::ZERO`, exactly as the additive
identity demands.

## What to notice

- **The mask is data, not control flow.** A boolean mask is converted to a
  native predicate once per vector and consumed by a predicated instruction;
  there is no per-lane branch in the vectorized path.
- **Mode is a type, not a flag.** `Unmasked` vs. `Masked` is resolved at
  monomorphization; the `Masked` marker activates only the predicated methods.
- **Masking is how tails stay correct.** The same hardware predicates express
  selected lanes and live lanes. The partial memory seam additionally records
  the exact accessible prefix, so inactive addresses are not part of the
  memory contract.
