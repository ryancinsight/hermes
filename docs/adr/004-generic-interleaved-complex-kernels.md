# ADR 004: Generic Interleaved Complex Kernels via Adjacent-Pair Primitives

## Context

The initial interleaved complex implementation (commits 55efd38, b7f1a90, b148fed)
had three structural defects:

1. The generic `interleaved_complex_mul_assign` / `interleaved_complex_dot`
   kernels loaded vector registers, immediately spilled them to 128-element
   stack buffers, performed scalar complex arithmetic on the buffers, and
   reloaded — the vector unit was never used for arithmetic, and every
   iteration paid two redundant load/store round-trips.
2. Fast paths existed only as hand-written AVX+FMA functions for `f32` and
   `f64` `mul_assign` (none for `dot`), duplicated across two
   `InterleavedComplexLane` impls with per-type `OnceLock` feature caching —
   a dispatch mechanism parallel to the canonical `#[runtime_dispatch]` macro.
3. The conjugation fast path used an extra sign-mask XOR per vector instead of
   selecting the algebraically equivalent alternating-FMA form.

## Decision

Extend `SimdKernel<T>` with five adjacent-pair primitives, each with a
default scalar-emulated implementation (surface minimization via default
trait methods) and hardware overrides in the AVX2/AVX-512 float kernels:

| Method | AVX2 f32 | AVX2 f64 | AVX-512 |
|--------|----------|----------|---------|
| `swap_adjacent` | `_mm256_permute_ps(v, 0xB1)` | `_mm256_permute_pd(v, 0b0101)` | `_mm512_permute_*` |
| `dup_even` | `_mm256_moveldup_ps` | `_mm256_movedup_pd` | `_mm512_moveldup_ps` / `_mm512_movedup_pd` |
| `dup_odd` | `_mm256_movehdup_ps` | `_mm256_permute_pd(v, 0b1111)` | `_mm512_movehdup_ps` / `_mm512_permute_pd(v, 0xFF)` |
| `fmaddsub` | `_mm256_fmaddsub_ps` | `_mm256_fmaddsub_pd` | `_mm512_fmaddsub_*` |
| `fmsubadd` | `_mm256_fmsubadd_ps` | `_mm256_fmsubadd_pd` | `_mm512_fmsubadd_*` |

Complex products then stay in registers for every `(T, Arch)` pair:

- `a * b`       = `fmaddsub(dup_even(a), b, mul(dup_odd(a), swap_adjacent(b)))`
- `a * conj(b)` = `fmsubadd(dup_odd(a), swap_adjacent(b), mul(dup_even(a), b))`

The conjugated identity follows from lane algebra: with `a = [ar, ai]`,
`b = [br, bi]`, even lane `ai*bi + ar*br = re`, odd lane `ai*br - ar*bi = im` —
no sign-mask constant required.

Runtime selection now goes through `#[runtime_dispatch(avx512f, avx2, neon,
scalar)]` and two new `SimdOps` methods (`interleaved_complex_mul_assign`,
`interleaved_complex_dot`, both `<const CONJ_B: bool>`), the same chain as the
dense kernels. `InterleavedComplexLane`, the per-type impl duplication, the
`OnceLock` caches, and both hand-written AVX functions are removed.

NEON f32/f64 override the primitives with `vrev64q_f32` / `vextq_f64`
(swap), `vtrn1q`/`vtrn2q` (duplication), and a sign-flip XOR composed with
`vfmaq` for the alternating FMAs (rounding-identical to a native
`fmaddsub`). Compile-verified against `aarch64-unknown-linux-gnu`; runtime
differential validation on aarch64 hardware remains outstanding.

## Consequences

- Any present or future `Scalar` type with `SimdKernel` impls gets vectorized
  complex kernels automatically; the dot product gains a SIMD path it never had.
- Breaking (pre-0.2): `InterleavedComplexLane` is removed; the runtime entry
  points keep their names and signatures but now bound on `SimdOps`.
- Differential tests assert bitwise equality of AVX2/AVX-512 backends against
  the `Scalar` backend using dyadic-rational inputs (exact under both fused
  and unfused rounding); tail lengths and both conjugation variants covered.
- Measured on Core Ultra 9 285K (AVX2+FMA), 65 536 complex `f64` pairs,
  release profile: dot 47.3 ms vs 74.9 ms scalar (1.58×), mul_assign 78.2 ms
  vs 126.2 ms (1.61×) per 2 000 iterations (`examples/complex_dot.rs`).
