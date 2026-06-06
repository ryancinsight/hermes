# ADR 003: NUMA Memory Layout and Topology Integration

## Context

High-performance numerical computing on multi-socket servers faces latency penalties when threads access memory across socket boundaries (Non-Uniform Memory Access). Cross-node access degrades bandwidth and increases L3 cache coherency overhead. For dense operations (such as AMX/AVX-512 GEMM), allocating tensors local to the executing NUMA node is critical.

## Design

### 1. NUMA-Aware Allocator Integration

We define the `NumaAllocator` trait and implement `MnemosyneNumaAllocator`:
- **Linux**: Hooks into `numa_alloc_onnode` and `numa_free` from `libnuma`.
- **Windows**: Hooks into `VirtualAllocExNuma` and `VirtualFreeEx`.
- **Fallback**: Falls back to the standard system allocator on unsupported operating systems or when node-local allocation fails.

### 2. Thread Affinity Binding

- `NumaBinding` acts as an RAII guard that temporarily pins the executing thread to a specific NUMA node.
  - **Linux**: Uses `numa_bind` with configured node masks.
  - **Windows**: Uses `SetThreadAffinityMask` using masks retrieved from `GetNumaNodeProcessorMask`.
- Restoring the old affinity mask on drop ensures thread pinning does not leak outside of the compute phase.

### 3. Memory Residency Verification

- `verify_numa_locality` validates if a given pointer range is physically resident on the expected NUMA node:
  - **Linux**: Queries `move_pages` with a null target nodes array to retrieve page status, or inspects Mnemosyne page descriptors for verified segments.
  - **Windows**: Queries `VirtualQuery` page states.
- If memory resides on a remote node, the system re-routes execution paths (e.g. from AMX to AVX-512) or issues warning logs to help diagnose alignment defects.

### 4. Cache & TLB Pressure Management

- Allocating large buffers aligned to 2MB page boundaries (hugepages/large pages) reduces TLB cache misses.
- Storing ultra-dense tensors (`Bf4` / `Bf8`) keeps the working set inside L1/L2 caches, maximizing the compute-to-memory-bandwidth ratio.

## Consequences

- Node-local allocation minimizes cross-socket interconnect traffic.
- Transparent fallback prevents crashes on single-socket consumer systems or non-NUMA configurations.
- Runtime routing mitigates performance degradation caused by incorrect tensor allocation placement.
