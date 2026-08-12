# 4. Horizontal Reductions

A horizontal reduction collapses a slice of `T` into a single `T`. hermes-simd
exposes the common reductions as slice functions on the dispatch facade, and a
generic `reduce`/`zip_reduce` surface parameterized by zero-cost operation
strategies for the rest.

## The slice facade

```rust,ignore
use hermes_simd::{sum, min, max, abs_sum, abs_max, argmin, argmax};

let data: Vec<f32> = vec![-5.0, 3.0, -1.0, 7.0, 2.0, -9.0, 4.0, 0.0];

let s  = sum::<f32>(&data);      // Σ xᵢ
let mn = min::<f32>(&data);      // min xᵢ
let mx = max::<f32>(&data);      // max xᵢ
let l1 = abs_sum::<f32>(&data);  // Σ |xᵢ|
let l∞ = abs_max::<f32>(&data);  // max |xᵢ|
let (i_min, _) = argmin::<f32>(&data).expect("non-empty, NaN-free");
let (i_max, _) = argmax::<f32>(&data).expect("non-empty, NaN-free");
```

Every one of these is runtime-dispatched (Chapter 2): the caller writes a
plain slice function and the widest ISA kernel the host supports computes it.
The `Scalar` reference backend is the differential oracle the native paths are
tested against.

## Empty-slice identities

Each reduction has a specified identity so the empty slice is well-defined
rather than a special case:

| Operation | Empty result | Why |
|---|---|---|
| `sum` | `T::ZERO` | additive identity |
| `min` | `T::MAX_VALUE` | identity element for min (positive infinity for floats) |
| `max` | `T::MIN_VALUE` | identity element for max (negative infinity for floats) |
| `abs_sum` | `T::ZERO` | additive identity |
| `abs_max` | `T::ZERO` | mathematically correct: every magnitude is non-negative |
| `argmin` / `argmax` | `None` | no element to index |

The `Option` return of `argmin`/`argmax` exists for two reasons: the empty
slice, and NaN rejection. NaN-containing input returns `None` — an extremum
across a NaN has no well-defined position, and the backend must not pick one
silently. The first slice element's signed-zero representation is preserved
when it ties with later elements.

## The generic reduction surface

The strategy ZSTs — `Sum`, `Product`, `Min`, `Max`, `AbsSum`, `AbsMax`, and
`Dot` (for pairwise reduction) — implement `ReductionOp<T>` and parameterize
`SimdView::reduce` and `SimdView::zip_reduce`:

```rust,ignore
use hermes_simd::{SimdView, Unaligned, Scalar, Sum};

let data = [1.0_f32, 2.0, 3.0, 4.0];
let v = SimdView::<f32, Scalar, Unaligned>::new(&data).unwrap();
let total = v.reduce(Sum);   // equivalent to the dispatch-facade `sum`
```

Each strategy defines four facts the reduction loop composes: the identity
element, the lane-wise transform (identity for `Sum`/`Min`/`Max`, `abs` for
`AbsSum`/`AbsMax`), the vector accumulation rule, and the horizontal finalize.
Because the strategies are zero-sized and the trait is generic, every branch
over which operation is running is eliminated during monomorphization —
`reduce(Sum)` compiles to exactly the code `sum` would emit. `Product`
documents its zero-cost guarantee explicitly: `size_of::<Product>() == 0`.

The loop structure is fixed: a vector accumulator starts at the identity
vector, processes full vectors, and finishes the ragged tail through the
provider's masked-reduction seam so the masked memory contract stays valid on
every backend.

## Reduction order and floating-point

Floating-point addition is non-associative, and a SIMD reduction groups the
sum: lanes accumulate in parallel within a vector register and the horizontal
finalize folds the partials. This *reordering* is not an error — it is a
documented property. `sum` and `dot` promise agreement with a sequential
reference within an error bound derived from the grouping depth, never
bitwise equality. The relevant consequence is at the assertion site: test
tolerances come from the reduction structure (machine epsilon of `T` scaled by
the fold depth), never from a guess. Signed-zero and NaN contracts are
likewise pinned: min/max propagate the element type's native contract, and
`argmin`/`argmax` reject NaN outright.

## Popcount reductions

For integer vectors, `reduce_popcount`, `reduce_popcount_and`,
`reduce_popcount_or`, and `reduce_popcount_xor` reduce `count_ones(x)` over
the slice — the bitwise accumulator variants combining lanes with the named
operator. These are exact integer reductions: no ordering envelope applies.

## What to notice

- **Empty slices have identities, not errors.** Every reduction is total over
  all inputs; the `None` case belongs only to operations that must return an
  index or a NaN-free extremum.
- **One loop, many operations.** The strategy ZSTs make `reduce(Op)` a single
  generic implementation that monomorphizes per operation, not a family of
  copy-pasted loops.
- **Order is part of the contract.** Floating reductions document their
  reassociation envelope; integer and popcount reductions are exact.
- **NaN is handled, not hidden.** NaN inputs are rejected where an index or
  extremum is the output, and the min/max NaN contract is preserved from the
  element type — never silently converted.
