# ADR 008: Refined SimdView Design with Reference Type-State Parameterization

## Status
Approved

## Context
Originally, `hermes-simd-core` had two separate view types: `SimdView` (read-only) and `SimdViewMut` (exclusive mutable). This duplicated many slice operation declarations (such as `len()`, `is_empty()`, `sum()`, and `dot()`) and required different method names or boilerplate conversion logic between the two types.

Additionally, to ensure strict alignment-aware casting and zero-copy performance without aliasing bugs, we needed the type system to enforce:
1. Covariance over the slice lifetime `'a` for read-only views (allowing them to be safely coerced to shorter lifetimes).
2. Invariance over `'a` for mutable views (preventing multiple active mutable references to the same memory).
3. Compile-time prevention of cloning/copying for exclusive mutable views.

## Decision
We unified the two types into a single `SimdView<'a, T, Arch, Align, Ref>` type:
- `Ref` represents the actual reference type (`&'a [T]` or `&'a mut [T]`).
- Internal layout is represented by `ptr: *mut [T]` and `_marker: PhantomData<(&'a T, Arch, Align, Ref)>`.
- This representation ensures that `SimdView` remains `#[repr(transparent)]` with zero runtime size overhead (size matches standard Rust slice references exactly, confirmed via unit tests).

## Consequences
- **Zero Overhead**: Struct size is verified to be exactly 16 bytes (on 64-bit platforms), matching fat pointers.
- **Lifetime & Aliasing Safety**: The Rust compiler's borrow checker automatically handles safety bounds (covariance, invariance, and borrow-checking) because `Ref` is passed directly through `PhantomData`.
- **DRY Codebase**: All read-only operations are implemented once for any `Ref` type, while in-place mutators (`add_assign`, `mul_assign`, `as_slice_mut`) are selectively implemented only when `Ref = &'a mut [T]`.
- **API Simplification**: Eliminates `SimdViewMut` and leverages Rust's native reference types for type-state guarantees.
