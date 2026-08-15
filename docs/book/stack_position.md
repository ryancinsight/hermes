# 10. Position in the Stack

`hermes` is the **CPU lane-parallel kernel** provider of the Atlas stack: its
role is "CPU SIMD/SWAR vocabulary, ISA dispatch, and vector kernels" (the
meta-repo README's provider table). Everything in this book — the ISA
detection, the `SimdArch` and operation-facet seams, the dense and sparse kernels,
the aligned buffers and views — is the bounded context of that one role. The
stack places hermes so that a consumer asking for a vector kernel gets it
from one owner, never by hand-writing intrinsics or importing a per-ISA crate.

## The dependency graph around hermes

```text
memory      execution    CPU lanes   host arrays   accelerator
mnemosyne ──> moirai ───> hermes ───> leto ───────> hephaestus
```

- **Downstream (consumed by hermes):**
  - `mnemosyne` — host allocation, arenas, and staging memory. `AlignedVec`
    allocates through Mnemosyne under the `mnemosyne-memory` feature
    (Chapter 7), and falls back to the system allocator otherwise.
  - `themis` — placement and locality law. `current_numa_node` delegates to
    the themis topology SSOT, so NUMA facts have exactly one owner.
  - `eunomia` — scalar and datatype law. The facade re-exports Eunomia's
    packed low-precision types (`Bf16`, `Bf4`, `Bf8`, `F4`, `F8`) and their
    unpack functions, so a kernel's element vocabulary is the stack's
    datatype vocabulary, not a hermes-local copy.
- **Upstream (consumers of hermes):**
  - `leto` — host arrays, layouts, views, and linear algebra. The README
    records the `leto → hermes` edge: Leto owns the array substrate and uses
    hermes for the lane-parallel kernels beneath its operations.
  - Everything above `leto` (accelerators, tensors, solvers, domain packages,
    integrators) is downstream of hermes only through Leto — a consumer that
    needs a kernel never re-implements SIMD dispatch locally.

## What hermes owns, and does not own

The bounded-context line is drawn by the provider table:

- **Owns:** SIMD/SWAR kernels, runtime ISA selection, vector dispatch. The
  `hermes-simd` crate graph (the proc-macro, intrinsics markers and impls,
  the generic core, the monomorphized register types, the facade, and the
  bench/example surfaces) is one delivery unit for that concern.
- **Does not own:** allocation policy (Mnemosyne), placement law (themis),
  scalar datatype law (eunomia), arrays and linear algebra (leto), scheduling
  and execution (moirai), accelerator devices (hephaestus). Each of these has
  an owner, and hermes consumes the contracts of the ones below it rather
  than redefining them.

This is why the design choices earlier in the book are not optional flavor:
the `AlignedVec` allocator seam, the NUMA delegation, and the `Scalar`
vocabulary all exist because the allocation, placement, and datatype concerns
*already have owners elsewhere in the stack*. A kernel layer that hard-coded
`libc::malloc`, fabricated NUMA node 0, or redefined numeric types would be
re-forking a dimension another package owns.

## The book's path through the stack

Each part of this book maps onto one layer boundary:

- **Part I** (dispatch model) is the hermes-internal answer to ISA selection —
  the thing the rest of the stack relies on being correct and cheap.
- **Part II** (kernel operations) is the owned vocabulary: reductions, dot
  and AXPY, masked paths, and their differential verification.
- **Part III** (buffers and views) is where hermes meets the rest of the
  stack's memory model — aligned allocation via Mnemosyne, NUMA placement via
  themis, and typed views that downstream array layers can build on.

For a consumer, the contract is: "if it is a lane-parallel CPU kernel, it
comes from hermes; if it is an array, a device, or a schedule, it comes from
the owner of that concern — and the two compose through typed views and
typed allocation."
