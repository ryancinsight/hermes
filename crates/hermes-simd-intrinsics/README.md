# hermes-simd-intrinsics

Hardware intrinsics and backend-specific kernel implementations for
[Hermes](https://github.com/ryancinsight/hermes), the Atlas CPU SIMD substrate.
This crate provides the concrete `SimdKernel<T>` implementations that
`hermes-simd-core` declares; most consumers depend on the `hermes-simd` facade
rather than on this crate directly.

## Architecture markers

| Marker | ISA | f32 lanes | f64 lanes |
|--------|-----|-----------|-----------|
| `Scalar` | scalar loop | 4 | 2 |
| `Avx2` | x86 AVX2 | 8 | 4 |
| `Avx512` | x86 AVX-512F | 16 | 8 |
| `Neon` | AArch64 NEON | 4 | 2 |
| `SveArch` | AArch64 SVE shape, emulated | 16 | 8 |

Also here: the AVX-512 VNNI and 256-bit AVX-VNNI tile multipliers, packed 4-bit
hardware unpacking, and the sliding-attack bitboard backends (Kogge-Stone,
Hyperbola Quintessence, Fancy Magic, Hybrid SWAR-Magic).

## Intel AMX is quarantined

The AMX tile kernels (`tdpbf16ps`, `tdpbssd`), tile descriptors, and the
`AmxSession`/`AmxBatchSession` RAII guards are compiled, but the runtime support
probe reports `false` unconditionally, so `AmxSession::new` always returns
`AmxSessionError::UnsupportedTarget` and no AMX instruction ever executes. Raw
CPUID cannot decide AMX availability — it misses XCR0 OS enablement and the
Linux `XTILEDATA` process permission — and the stable feature-detection macro
does not accept AMX feature strings on the pinned toolchain. The removal trigger
is a stable, permission-aware probe; see the
[repository README](https://github.com/ryancinsight/hermes#intel-amx-status).

## Feature flags

| Feature | Description |
|---------|-------------|
| `std` (default) | Standard library support; without it the crate is `no_std` |
| `mnemosyne-memory` (default) | Routes aligned allocation through the Mnemosyne allocator |

## Documentation

- API reference: [docs.rs/hermes-simd-intrinsics](https://docs.rs/hermes-simd-intrinsics)
- Workspace overview: the
  [repository README](https://github.com/ryancinsight/hermes#readme)

## License

MIT OR Apache-2.0
