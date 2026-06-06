//! Dynamic runtime dispatch choosing the optimal execution backend.

#[cfg(target_arch = "x86_64")]
use hermes_simd_core::numa::{NumaTopologyService, verify_numa_locality};
#[cfg(target_arch = "x86_64")]
use crate::cpu::{AmxSupport, Avx512Support};

/// Dynamic hardware and layout configuration dispatcher.
pub struct AdaptiveDispatcher;

/// Backend options chosen by the dispatcher.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DispatchDecision {
    /// Intel AMX backend.
    Amx,
    /// AVX-512 backend.
    Avx512,
    /// Standard scalar execution path.
    Scalar,
}

impl AdaptiveDispatcher {
    /// Dynamically choose the optimal compute backend based on hardware support,
    /// matrix shapes, aspect ratios, and NUMA node locality.
    pub fn select_backend<T>(
        m: usize,
        n: usize,
        k: usize,
        a_ptr: *const T,
        a_len: usize,
        b_ptr: *const T,
        b_len: usize,
    ) -> DispatchDecision {
        #[cfg(target_arch = "x86_64")]
        {
            let has_amx = <half::bf16 as AmxSupport>::has_amx();
            let has_avx512 = <half::bf16 as Avx512Support>::has_avx512();

            // Session-aware heuristic: if an AMX session is already active,
            // we bypass the high context configuration overhead threshold (16384 ops)
            // and allow AMX to run down to 2048 ops.
            let is_session_active = hermes_simd_intrinsics::AmxSession::is_active();
            let min_ops = if is_session_active { 2048 } else { 16384 };

            let total_ops = m * n * k;
            
            // Matrix dimension bounds check: only run AMX if dimensions exceed block size thresholds
            // to mitigate high context configuration/switch overheads.
            let is_too_small = (m < 16) || (n < 16) || (k < 32);

            if has_amx && total_ops >= min_ops && !is_too_small {
                // NUMA Locality Check:
                // Verify if memory is local to the current NUMA node.
                let total_nodes = NumaTopologyService::total_nodes();
                if total_nodes > 1 {
                    if let Some(curr_node) = NumaTopologyService::current_node() {
                        let a_local = verify_numa_locality(a_ptr as *const u8, a_len * core::mem::size_of::<T>(), curr_node);
                        let b_local = verify_numa_locality(b_ptr as *const u8, b_len * core::mem::size_of::<T>(), curr_node);
                        
                        if a_local && b_local {
                            return DispatchDecision::Amx;
                        } else {
                            // Cross-socket access detected: warn once in debug build, re-route to AVX-512 if available
                            #[cfg(all(debug_assertions, feature = "std"))]
                            {
                                static WARNED: core::sync::atomic::AtomicBool = core::sync::atomic::AtomicBool::new(false);
                                if !WARNED.swap(true, core::sync::atomic::Ordering::Relaxed) {
                                    std::eprintln!(
                                        "WARNING [hermes-simd]: Cross-node NUMA memory access detected. \
                                         Tensors reside on a remote NUMA node (current thread node: {}). \
                                         Re-routing from AMX to AVX-512 to mitigate latency bottlenecks.",
                                        curr_node
                                    );
                                }
                            }
                            if has_avx512 {
                                return DispatchDecision::Avx512;
                            }
                        }
                    }
                }
                return DispatchDecision::Amx;
            }

            if has_avx512 {
                return DispatchDecision::Avx512;
            }
        }

        let _ = (m, n, k, a_ptr, a_len, b_ptr, b_len);
        DispatchDecision::Scalar
    }
}
