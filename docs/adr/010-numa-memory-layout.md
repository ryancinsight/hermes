# ADR 010: NUMA Memory Layout and Topology Integration

## Status
Accepted

## Context

High-performance numerical computing on multi-socket servers faces latency penalties when threads access memory across socket boundaries (Non-Uniform Memory Access). Cross-node access degrades bandwidth and increases L3 cache coherency overhead. For dense operations (such as AMX/AVX-512 GEMM), allocating tensors local to the executing NUMA node is critical.

## Design

### 1. Allocation Ownership

Allocation routing is owned by Mnemosyne and the typed topology/current-node
queries it consumes from Themis. Hermes keeps only the `NumaAllocator` trait
needed by `AlignedVec::with_capacity_numa` and the `MnemosyneNumaAllocator`
adapter:
- **Default path**: temporarily binds the executing thread to the requested
  node, then allocates through `mnemosyne::Mnemosyne`; Mnemosyne owns segment
  routing, ownership metadata, and deallocation by pointer owner.
- **Feature-disabled path**: applies the same explicit affinity guard and uses
  the configured global allocator. Hermes does not call `numa_alloc_onnode`,
  `VirtualAllocExNuma`, or any direct OS allocation API.
- **Topology**: consumers that need node counts, distances, or processor maps
  use `themis::CpuTopology` directly.

### 2. Thread Affinity Binding

- `NumaBinding` acts as an RAII guard that temporarily pins the executing thread to a specific NUMA node.
  - **Linux**: Uses `numa_bind` with configured node masks.
  - **Windows**: Uses `SetThreadAffinityMask` using masks retrieved from `GetNumaNodeProcessorMask`.
- Restoring the old affinity mask on drop ensures thread pinning does not leak outside of the compute phase.

### 3. Memory Residency Verification

- `verify_numa_locality` validates if a given pointer range is physically resident on the expected NUMA node:
  - **Linux**: Queries `move_pages` with a null target nodes array to retrieve page status when `libnuma` is enabled.
  - **Windows**: Queries `K32QueryWorkingSetEx` working-set attributes.
- If memory resides on a remote node, the system re-routes execution paths (e.g. from AMX to AVX-512) or issues warning logs to help diagnose alignment defects.

### 4. Cache & TLB Pressure Management

- Allocating large buffers aligned to 2MB page boundaries (hugepages/large pages) reduces TLB cache misses.
- Storing ultra-dense tensors (`Bf4` / `Bf8`) keeps the working set inside L1/L2 caches, maximizing the compute-to-memory-bandwidth ratio.

## Consequences

- Node-local allocation minimizes cross-socket interconnect traffic.
- There is no Hermes-owned topology facade; consumer-facing topology APIs live
  in Themis.
- There is no direct Hermes OS allocation fallback; allocation ownership stays
  with Mnemosyne or the configured allocator path.
- Runtime routing mitigates performance degradation caused by incorrect tensor allocation placement.
