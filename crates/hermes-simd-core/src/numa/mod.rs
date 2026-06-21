//! NUMA-aware memory allocation and thread affinity interfaces.
//!
//! Provides explicit node placement capabilities for Windows and Linux architectures
//! with standard allocator fallback paths, thread affinity pinning, and memory
//! residency verification.

/// NUMA-aware memory allocators and platform allocation hooks.
pub mod allocator;
/// RAII thread affinity binding guards.
pub mod binding;
/// Page residency and socket locality verification.
pub mod locality;
/// NUMA topology query services.
pub mod topology;

pub use allocator::{MnemosyneNumaAllocator, NumaAllocator};
pub use binding::NumaBinding;
pub use locality::{current_numa_node, refresh_numa_node, verify_numa_locality};
pub use topology::{numa_node_count, numa_node_distance, NumaTopologyService};
