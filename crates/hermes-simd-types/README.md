# hermes-simd-types

Monomorphized vector register types for
[Hermes](https://github.com/ryancinsight/hermes), the Atlas CPU SIMD substrate.

The crate provides compile-time-configured type aliases that map to
target-optimal registers, explicit aliases for each hardware backend, and the
`PreferredArch` compile-time architecture selection. Consumers normally reach
these through the `hermes-simd` facade rather than depending on this crate
directly.

## Feature flags

| Feature | Description |
|---------|-------------|
| `std` (default) | Standard library support; without it the crate is `no_std` |
| `mnemosyne-memory` (default) | Routes aligned allocation through the Mnemosyne allocator |

## Documentation

- API reference: [docs.rs/hermes-simd-types](https://docs.rs/hermes-simd-types)
- Workspace overview: the
  [repository README](https://github.com/ryancinsight/hermes#readme)

## License

MIT OR Apache-2.0
