# ADR 001: Precision Ladder and Representation Schemes

## Status
Accepted

## Context

The hermes ecosystem requires memory-efficient representation of numerical datatypes to support ultra-quantized machine learning workloads and high-throughput vector computations. The system defines a precision ladder spanning from 4-bit to 64-bit float representations.

## Design

### 1. Representation Schemes

The ladder contains the following data formats:

| Format | Width | Bits (S/E/M) | Exponent Bias | Description |
|--------|-------|--------------|---------------|-------------|
| `Bf4`  | 4 bits | 1 / 2 / 1    | 1             | Brain Float 4. Packed two per byte. |
| `Bf8`  | 8 bits | 1 / 5 / 2    | 15            | Brain Float 8. E5M2 representation. |
| `Bf16` | 16 bits| 1 / 8 / 7    | 127           | Brain Float 16. E8M7 representation. |
| `F16`  | 16 bits| 1 / 5 / 10   | 15            | Standard IEEE 754 half precision. |
| `F32`  | 32 bits| 1 / 8 / 23   | 127           | Standard IEEE 754 single precision. |
| `F64`  | 64 bits| 1 / 11 / 52  | 1023          | Standard IEEE 754 double precision. |

### 2. Layout and Packaging

- `Bf4` is stored in memory as packed pairs (two elements per `u8` byte). The lower 4 bits represent the first element (index `2*i`), and the upper 4 bits represent the second element (index `2*i + 1`).
- `Bf8` elements are stored directly as single `u8` bytes.
- Memory representations are transparent newtypes (`#[repr(transparent)]`) wrapping raw storage bytes or core types.

### 3. Vectorized Unpacking to Bf16

To perform arithmetic operations on `Bf4` and `Bf8` efficiently, they are unpacked to `Bf16` at runtime:
- **AVX2 Implementation**: Uses vector shifts, bitwise masks, and interleave operations (`_mm256_unpacklo_epi16`, `_mm256_unpackhi_epi16`, and permutes) to process 32 elements in parallel.
- **Bias Diff Offset**: Subnormal values are flushed to zero. Normal values are shifted and adjusted by adding a constant exponent bias difference (`bias_diff`) via integer addition.

## Consequences

- **Memory Efficiency**: Storing models/tensors in `Bf4` reduces the memory footprint by 8x compared to `F32` and 4x compared to `F16`.
- **Zero-Copy Streaming**: Large tensors remain packed in memory and are unpacked on-the-fly into CPU caches or SIMD registers, minimizing memory bandwidth pressure.
- **Precision Integrity**: Conversion functions handle zero bias mapping explicitly, preventing sign-flushing defects.
