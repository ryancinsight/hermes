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

Revision 2026-09-01: the Windows mechanism recorded here originally read the
node's processor mask from `GetNumaNodeProcessorMask` and applied it with
`SetThreadAffinityMask`. Both halves are superseded below; see the revision
history for the driving evidence.

- `NumaBinding` acts as an RAII guard that temporarily pins the executing thread to a specific NUMA node.
  - **Linux**: Uses `numa_bind` with configured node masks.
  - **Windows**: Uses `SetThreadGroupAffinity` over the node's processor set as
    Themis reports it (`CpuTopology::numa_nodes()[i].processors`,
    group-flattened as `group * 64 + number`). Themis's
    `ProcessorAffinityGroups` partitions that set into native masks and selects
    the deterministic largest group. Hermes owns only active-host validation
    and the mechanism that applies affinity; it neither reconstructs the masks
    nor asks the operating system a second question about node membership.
- Restoring the old affinity mask on drop ensures thread pinning does not leak outside of the compute phase.
- One `SetThreadGroupAffinity` call names one processor group, so a node whose
  processors span several groups cannot be bound whole. The guard binds the
  group holding the largest share of the node and reports the shortfall through
  `NumaBindingCoverage`, rather than truncating silently.

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

## Revision history

- 2026-09-01: Route §2's group partitioning, native mask construction, and
  largest-group tie resolution through Themis for
  `HS-THEMIS-AFFINITY-CONSUMER-2026-09-01`. Hermes retains the
  `SetThreadGroupAffinity` lifetime mechanism and intersects the provider value
  with the live active-group mask before mutation.
- 2026-09-01: Supersede §2's Windows mechanism for
  `HS-NUMA-BINDING-THEMIS-QUERY-2026-09-01`. `GetNumaNodeProcessorMask` is
  group-unaware — it reports a single group-0 mask and truncates the node index
  to `u8` — so on a host with more than 64 processors it disagreed with the
  node membership Themis parses from `GetLogicalProcessorInformationEx`. Two
  answers to one question is the defect; this ADR's own §1 already assigns
  processor maps to Themis, and ADR 021 had already adopted
  `SetThreadGroupAffinity` for the exact-processor guard while §2 still named
  the single-group call. The query moves to Themis and the mechanism aligns
  with ADR 021; ownership is unchanged, so ADR 021's mechanism-versus-policy
  split stands as written.
