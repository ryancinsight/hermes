//! NUMA-aware memory allocation and thread affinity interfaces.
//!
//! Hermes keeps SIMD-local memory residency checks and explicit thread-affinity
//! guards. Topology discovery belongs to `themis`; allocation routing belongs to
//! `mnemosyne`.

#[cfg(target_os = "windows")]
mod affinity;
/// NUMA-associated memory allocators.
pub mod allocator;
/// RAII thread affinity binding guards.
pub mod binding;
/// Page residency and socket locality verification.
pub mod locality;
mod processor;

pub use allocator::{MnemosyneNumaAllocator, NumaAllocator};
pub use binding::{NumaBinding, NumaBindingCoverage};
pub use locality::{current_numa_node, refresh_numa_node, verify_numa_locality};
pub use processor::{ProcessorBinding, ProcessorBindingError, ProcessorIndex};
