# Example: ISA Dispatch

**Crate**: `hermes-simd`
**Source**: `crates/hermes-simd-examples/examples/book_isa_dispatch.rs`

Query the CPU's capabilities at runtime and exercise `sum` and `dot` through
the runtime-dispatch path.  The kernel selected depends on the host CPU;
on a Haswell or newer host the FMA-accelerated path runs automatically.

## Source

```rust
{{#include ../../../crates/hermes-simd-examples/examples/book_isa_dispatch.rs}}
```

## Sample Output (FMA3-capable host)

```text
=== ISA Capabilities ===
FMA3 (fused multiply-add) : true
f32 FmaSupport            : true

=== Runtime-dispatched sum ===
sum(0..1024) = 523776  expected = 523776

=== Runtime-dispatched dot product ===
dot(0..1024, [1.0; 1024]) = 523776  expected = 523776
dot(len=3, len=5) correctly returns Err

all ISA-dispatch assertions passed
```

## What to notice

- `has_fma3()` is computed once via `OnceLock` and cached for the process
  lifetime.  The probe calls `std::is_x86_feature_detected!("fma")` on x86-64;
  on other architectures it returns `false` without branching.

- `sum::<f32>(&data)` is a one-liner that hides the ISA selection.
  Internally it chains `f32::sum(data)` → dispatch function → whichever
  kernel matches the detected ISA.

- `dot::<f32>(&a, &b)` returns `Result<f32, SimdError>`.  The `Err` variant
  is `SimdError::LengthMismatch` when the slices have different lengths.
  Callers handle it the same way as any `Result`.
