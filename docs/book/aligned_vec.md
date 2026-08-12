# 7. AlignedVec

An owned, heap-allocated buffer whose *alignment is a type*: `AlignedVec<T, Align>`
carries a static alignment guarantee in its type parameter, so a vector declared
with `Aligned<64>` can never hold a 32-byte-aligned buffer and pretend otherwise.
Under the `mnemosyne-memory` feature the allocation goes through the Mnemosyne
allocator; without it, through the system allocator with the same layout
discipline.

## Why alignment is a type

SIMD loads come in two families: aligned and unaligned. An aligned load
(`vmovaps`/`vmovdqa`) is free when the address satisfies the vector width, and
a misaligned address faults; an unaligned load (`vmovups`) works anywhere but
cannot be assumed to cost nothing. Whether a `&[T]` is vector-aligned is not
knowable from the slice type — it is a runtime property of the base address.
`AlignedVec` closes that gap by making the guarantee part of the type, so a
view built from it can dispatch to `load_aligned`/`store_aligned` with the
compiler knowing the alignment statically.

The `Alignment` typestate is one of:

- `Unaligned` — no static guarantee; dispatches to unaligned operations.
- `Aligned<A>` — a static guarantee of `A` bytes, where `A` must be a power of
  two (checked by a const assertion at compile time, so `Aligned<32>` compiles
  and `Aligned<33>` does not).

A given `Align` is *sufficient* for an architecture when its byte boundary is
at least the vector register width (`ALIGN_BYTES >= REGISTER_WIDTH_BITS / 8`).
The view layer checks this (`is_aligned_for_arch`) to choose the load family;
the invariant is preserved automatically because both `Align` and `Arch` are
type parameters.

## Constructing

```rust,ignore
use hermes_simd::{AlignedVec, Aligned};

let mut v: AlignedVec<f32, Aligned<64>> = AlignedVec::with_capacity(1024);
v.extend_from_slice(&[1.0, 2.0, 3.0]);
```

- `new()` — empty, no allocation.
- `with_capacity(capacity)` — allocates `capacity` elements at the alignment
  boundary. Capacity is validated with checked multiplication, so an
  oversized request is a handled error, not a wrapped size.
- `with_capacity_numa(capacity, node)` — allocates on a specific NUMA node for
  locality-sensitive kernels; the owning node is recorded and carried by the
  vector.

Like `Vec`, the vector owns a capacity and grows by `push`/`reserve`/
`extend_from_slice`. Unlike `Vec`, the allocation is alignment-exact: the
layout requests the declared boundary directly rather than relying on the
allocator's default alignment.

## Zero-fill discipline

`AlignedVec` never zero-fills a buffer merely to be safe. Growth writes
through `spare_capacity_mut`, which exposes the spare tail as
`&mut [MaybeUninit<T>]` — the caller fills it and then `set_len` publishes the
initialized prefix. This is what lets the zero-copy constructors (map, splat,
gather, prefix-scan) write every element exactly once; `set_len` is `unsafe`
for the same reason `Vec::set_len` is — it asserts the caller has initialized
everything the new length claims. No `&mut [T]` ever spans uninitialized
elements.

## Views

The primary purpose of `AlignedVec` is to hand its buffer to a `SimdView` with
the alignment guarantee preserved:

```rust,ignore
let view = v.view::<Avx2>();        // SimdView<'_, f32, Avx2, Aligned<64>>
let view_mut = v.view_mut::<Avx2>(); // mutable, for transform_in_place
```

The constructors are `expect`s, not `Option`s: the alignment check that makes
`SimdView::new` return `None` on arbitrary slices is discharged by the type —
the buffer's address is the invariant the type holds — so the only remaining
failure is the arch-support check, and naming an arch the host does not
implement panics here exactly as it does for every view (Chapter 8).

## Alignment conversions

- `into_unaligned()` — strip the guarantee zero-cost; always succeeds.
- `try_into_alignment::<NewAlign>()` — promote to a stricter alignment,
  returning `None` if the base address does not satisfy it.
- `into_alignment_unchecked` — the `unsafe` escape hatch, requiring the caller
  to prove the address satisfies `NewAlign`. The safe surface never reaches it
  without a check.

A buffer can therefore move between `Unaligned` and `Aligned<N>` states
explicitly, with the state transition visible in the type and never silently
assumed.

## NUMA locality

For multi-socket workloads the *node* a buffer lives on matters as much as its
byte alignment. `current_numa_node()` reports the executing thread's node via
the themis topology SSOT (returning `None` when the platform reports none —
never a fabricated node 0), and `verify_numa_locality(ptr, size, node)` checks
that the physical pages backing a range are resident on a node, caching the
result per allocation generation so a dealloc/realloc cannot serve stale
locality data. `with_capacity_numa` allocates first-touch-local to the target
node, and deallocation routes through the owning node's batch-detach path
rather than a shared cross-node lock.

## What to notice

- **Alignment is an invariant of holding the value.** `Aligned<A>` is checked
  at compile time to be a power of two, and the buffer's actual address is the
  only thing that can ever back it.
- **`SimdView::new`'s `None` cases are discharged by the type.** The alignment
  claim an `AlignedVec` view makes is true by construction, so only the
  arch-support check can fail, and it fails loudly at the view — never inside
  a kernel.
- **No wasted initialization.** Spare capacity is `MaybeUninit`-typed and
  filled exactly once; the zero-fill-free property is what keeps the
  allocation budget on the measured path.
- **NUMA is a first-class axis, not an afterthought.** Node placement, node
  attribution, and locality verification are part of the allocation contract.
