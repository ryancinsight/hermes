# hermes-simd

The public facade of the [Hermes](https://github.com/ryancinsight/hermes) SIMD
workspace: a zero-overhead, stable-Rust abstraction over lane-parallel CPU
kernels. Traits, ZST markers, const generics, and full monomorphization keep the
generated machine code identical to a hand-written per-ISA specialization.

Hermes is the Atlas CPU SIMD substrate. It owns per-core vector work, scalar
fallbacks, and ISA dispatch; thread-level scheduling belongs to Moirai and
device execution to Hephaestus.

## What this crate exposes

- The sealed `SimdOps` extension trait over slices.
- Runtime-dispatched free functions — `sum`, `dot`, `masked_dot`,
  `interleaved_complex_*`, `real_mul_to_interleaved_complex_runtime`,
  `spmv_*`, `axpy`, `axpy_rows`, `axpy_rows_batch`, `gemm`.
  `interleaved_complex_*`, `real_interleaved_complex_*`, `spmv_*`, `axpy`,
  `axpy_rows`, `axpy_rows_batch`, `gemm`.
- `dispatch_view` CPUID routing, plus `TargetId` / `dispatch_view_to` for
  forcing a specific backend when the host can execute it.
- Flat re-exports of the core abstractions (`hermes-simd-core`), the
  architecture kernels (`hermes-simd-intrinsics`), the monomorphized aliases
  (`hermes-simd-types`), and the Eunomia precision ladder.

```rust
use hermes_simd::sum;

let data = vec![1.0f32; 1024];
assert_eq!(sum(&data), 1024.0);
```

## Feature flags

| Feature | Description |
|---------|-------------|
| `std` (default) | Runtime CPU feature detection (`is_x86_feature_detected!`); without it dispatch uses compile-time `cfg!(target_feature)` only |
| `mnemosyne-memory` (default) | Routes `AlignedVec` allocation through the Mnemosyne allocator |
| `libnuma` | Linux NUMA affinity and residency probes via libnuma (links `-lnuma`) |

Compiles on stable Rust; no nightly prerequisites.

## Documentation

- API reference: [docs.rs/hermes-simd](https://docs.rs/hermes-simd)
- Workspace overview, verification policy, and Intel AMX quarantine status:
  the [repository README](https://github.com/ryancinsight/hermes#readme)

## License

MIT OR Apache-2.0
