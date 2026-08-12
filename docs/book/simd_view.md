# SimdView and register-blocked tiling

`SimdView` is the typed window every kernel reads through, and tiling is how
the block operations keep the FMA pipe saturated. This chapter covers the view
contract first — what it promises, what it refuses to construct — then the
`TilingStrategy` seam and its `TiledPolicy` instantiations. The worked example
is in [the tiling example](examples/tiling.md).

## The view contract

```rust,ignore
use hermes_simd::SimdView;

let a = [1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
let va = SimdView::<f32, Scalar, Unaligned>::new(&a)?;
```

`SimdView<'a, T, Arch, Align, Mode, Ref>` is a `#[repr(transparent)]` wrapper
over a `*mut [T]` plus a phantom pack of its type parameters — the arch marker
(`Arch: SimdArch`, chapter 3), the alignment claim (`Align: Alignment`, chapter
7), the execution mode (`Mode: ExecutionMode = Unmasked`, chapter 6), and the
borrow form (`Ref = &'a [T]`). Nothing is stored at runtime; the parameters
exist only so the type system knows which kernel family may run on the data
and under what alignment. The read-only form is `Copy`, so views can be passed
by value and threaded through closures without clones.

Construction is where runtime facts are promoted into type-level guarantees.
`new` / `new_mut` return `Option<Self>`, returning `None` when:

- the host cannot execute `Arch` — naming a marker the CPU does not implement,
  such as `Avx512` on a machine without it; or
- the alignment claim cannot be met — an `Aligned<A>` view requires the slice's
  actual base address to satisfy `A`.

The first check is the `HS-405` support boundary from chapter 3 enforced at the
view, the last place a caller can bypass it: every operation on the view calls
`#[target_feature]`-gated kernels, so a view that existed without that
guarantee would let safe code execute unsupported instructions. The alignment
check exists for the same reason on the other axis — a kernel compiled for an
aligned load must not receive a misaligned pointer.

An `Unaligned` view is the default floor: it always passes the alignment check
and dispatches to kernels that tolerate arbitrary addresses. An `Aligned<A>`
view is produced from an `AlignedVec` (chapter 7) or promoted from an
unaligned view through `try_into_aligned::<A>()`, which performs the runtime
address check and returns `None` on a misaligned buffer. `into_unaligned()`
is the one-way downgrade that drops the claim.

## Reading and reshaping

The inspection surface is small and predictable:

- `as_slice` / `as_slice_mut` give the underlying data back; `len`, `is_empty`
  report element count.
- `slice_unaligned`, `slice_aligned::<A>`, and their `_mut` variants re-slice
  with the alignment claim preserved or reintroduced.
- `simd_chunks` / `simd_chunks_mut` and `zip_chunks` / `zip_chunks_mut`
  iterate in `Arch`-sized register chunks, the unit the kernels operate on.
- `cast::<U: bytemuck::Pod>()` reinterprets the element type (length adjusted
  to the new width); it returns `Option` because the byte length must remain
  whole for the target type.
- `downgrade()` turns a mutable view back into the read-only form.

These are the building blocks the block kernels compose out of. A kernel never
bounds-checks element access; it obtains a view whose construction already
discharged support and alignment, then processes `simd_chunks` against slices
whose lengths are part of the call contract.

## Why tiling: saturating the FMA pipe

A dot product that carries a single running accumulator is latency-bound: each
FMA waits on the previous result, so the pipe stalls for the full FMA latency
every element. The fix is `TILE_M` independent accumulators, one per register:

```rust,ignore
use hermes_simd::{SimdView, tiled_dot};

let a = [1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
let b = [2.0_f32; 8];
let va = SimdView::<f32, Scalar, Unaligned>::new(&a).expect("slice fits");
let vb = SimdView::<f32, Scalar, Unaligned>::new(&b).expect("slice fits");

let dot = tiled_dot::<f32, Scalar, Unaligned, 4>(&va, &vb)?;
assert!((dot - 72.0).abs() < 1e-6);
```

Each iteration issues `TILE_M × LANE_COUNT` independent multiply–accumulates,
so `TILE_M` results are in flight and the pipeline never waits on its own
output. The tile shape is a const generic, so the whole kernel body
monomorphizes to a straight-line register program with the tile size baked in
— no runtime loop-carried parameter.

The same reasoning applies to GEMV, but with the roles swapped: the operation
is memory-bound, dominated by streaming `A`, so the win is reusing each loaded
`x` element across `TILE_M` rows instead of fetching it once per row.
`tiled_gemv` therefore processes `TILE_M` rows of `A` simultaneously, holding
one accumulator per row:

```rust,ignore
use hermes_simd::{SimdView, tiled_gemv};

let a = [1.0_f32, 0.0, 2.0, -1.0, 0.0, 1.0, 0.0, 3.0, -2.0, 0.0, 1.0, 0.0];
let x = [1.0_f32, 2.0, 3.0, 4.0];
let mut y = [0.0_f32; 3];

let va = SimdView::<f32, Scalar, Unaligned>::new(&a).expect("slice fits");
let vx = SimdView::<f32, Scalar, Unaligned>::new(&x).expect("slice fits");

tiled_gemv::<f32, Scalar, Unaligned, 2>(&va, &vx, &mut y, 3, 4)?;
assert_eq!(y, [3.0, 14.0, 1.0]); // row0 = 1+6-4, row1 = 2+12, row2 = -2+3
```

The accumulate convention is part of the contract: GEMV and GEMM compute
`y += A·x` / `c += A·B`, never overwrite. A second call with the same `y`
doubles the result, and zeroing `y` first recovers plain `y = A·x`. The facade
exposes the same convention on plain slices via `gemv(a, x, y, nrows, ncols)`,
so the book example and the tiled core agree on the arithmetic.

## The strategy seam

The tiled entry points are thin wrappers over one trait:

```rust,ignore
trait TilingStrategy<T, Arch, Align> {
    const TILE_M: usize;
    const TILE_N: usize;
    fn dot(a: &SimdView<'_, T, Arch, Align>, b: &SimdView<'_, T, Arch, Align>)
        -> Result<T, SimdError>;
    fn gemm(a: &SimdView<'_, T, Arch, Align>, b: &SimdView<'_, T, Arch, Align>,
            c: &mut [T], m: usize, n: usize, k: usize) -> Result<(), SimdError>;
    fn gemv(a: &SimdView<'_, T, Arch, Align>, x: &SimdView<'_, T, Arch, Align>,
            y: &mut [T], nrows: usize, ncols: usize) -> Result<(), SimdError>;
    // + gemv_transpose, gemv_strided(lda), gemv_transpose_strided(lda)
}
```

`TilingPolicy<TILE_M, TILE_N>` is a zero-sized marker that implements the
strategy — the tile shape is a type, resolved at compile time with no runtime
storage, exactly like the arch markers. The free functions `tiled_dot`
(`TILE_N` fixed to 1), `tiled_gemv`, and `tiled_gemm` are the preferred API;
they pick the `TilingPolicy` instantiation and call through the trait.

Three standard tiles cover the register budget:

- `TilingPolicy::SCALAR_DEGENERATE` — `1,1`, no tiling; the reference form.
- `TilingPolicy::AVX2_STANDARD` — `4,4`: 4 `f32` accumulators × 8 lanes.
- `TilingPolicy::AVX512_OPTIMAL` — `8,4`: 8 accumulators × 16 lanes.

`validate()` asserts the tile is non-degenerate (`TILE_M ≥ 1`, `TILE_N ≥ 1`)
as a const fn, so an invalid tile is a compile error, not a runtime trap.

All block operations validate their dimension contract and return
`SimdError` on violation — matching the view construction, the strategy is
either type-checked (tile shape, alignment, arch) or checked once at the call
boundary (dimensions, output length) before any kernel runs.

## Where this fits

The view is the place where the chapter-3 dispatch model and the chapter-7
allocation guarantees meet the chapter-4/5 kernels: an `AlignedVec` produces an
`Aligned<A>` view whose construction is infallible by type, and every kernel
below it runs without a per-element check. Tiling adds the register-level
parallelism that makes the arithmetic throughput-bound rather than latency-
bound, with the shape decision encoded as const generics so each ISA backend
monomorphizes the tile it was designed for.

## Exercises

1. Change `tiled_dot`'s `TILE_M` from 1 to 4 to 8 and confirm all three agree
   on 72.0 within tolerance — the accumulator count is a performance parameter,
   not a correctness one.
2. Re-run the `gemv` cross-check with `y` non-zero to confirm accumulation:
   the second call doubles the first result.
3. Construct `SimdView::<f32, Avx2, Unaligned>::new` on this machine and
   explain which of the two `None` conditions applies to `Avx2` vs. `Avx512`
   (chapter 3's `is_runtime_supported` decides).
