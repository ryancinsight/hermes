//! Dynamic runtime dispatch choosing the optimal execution backend.

#[cfg(target_arch = "x86_64")]
use crate::cpu::{AmxSupport, Avx512Support};
#[cfg(target_arch = "x86_64")]
use eunomia::Bf16;
#[cfg(target_arch = "x86_64")]
use hermes_simd_core::numa::{current_numa_node, verify_numa_locality};

/// Dynamic hardware and layout configuration dispatcher.
pub struct AdaptiveDispatcher;

/// Backend options chosen by the dispatcher.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DispatchDecision {
    /// Intel AMX backend.
    Amx,
    /// AVX-512 backend.
    Avx512,
    /// 256-bit VEX-encoded AVX-VNNI backend (client CPUs without AVX-512).
    /// Currently consumed by the int8 tile GEMM; kernels without an AVX-VNNI
    /// implementation treat this tier as scalar.
    AvxVnni,
    /// Standard scalar execution path.
    Scalar,
}

// ── Cached topology flags ────────────────────────────────────────────────────
//
// CPU feature detection (AMX/AVX-512) is cached per-type via `OnceLock` in
// `cpu.rs`. NUMA topology is stable after process start.

#[cfg(target_arch = "x86_64")]
#[inline]
fn is_multi_numa() -> bool {
    use std::sync::OnceLock;
    static MULTI_NUMA: OnceLock<bool> = OnceLock::new();
    *MULTI_NUMA.get_or_init(|| {
        themis::CpuTopology::detect().is_some_and(|topology| topology.numa_nodes().len() > 1)
    })
}

impl AdaptiveDispatcher {
    /// Dynamically choose the optimal compute backend based on hardware support,
    /// matrix shapes, aspect ratios, and NUMA node locality.
    ///
    /// # Performance notes
    /// - `has_amx`/`has_avx512` are cached via `OnceLock` in `cpu.rs`.
    /// - `is_multi_numa()` is cached via `OnceLock` (process-stable).
    /// - `current_numa_node()` (Themis locality query) is called only on
    ///   multi-NUMA + AMX-eligible paths.
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
            let has_amx = <Bf16 as AmxSupport>::has_amx();
            let has_avx512 = <Bf16 as Avx512Support>::has_avx512();

            // Session-aware heuristic: if an AMX session is already active,
            // bypass the high context-setup threshold.
            let is_session_active = hermes_simd_intrinsics::AmxSession::is_active();
            let min_ops = if is_session_active { 2048 } else { 16384 };

            let total_ops = m * n * k;
            let is_too_small = (m < 16) || (n < 16) || (k < 32);

            if has_amx && total_ops >= min_ops && !is_too_small {
                // NUMA locality check — skip the per-call syscall on single-node hosts.
                if is_multi_numa() {
                    if let Some(curr_node) = current_numa_node() {
                        let a_local = verify_numa_locality(
                            a_ptr as *const u8,
                            a_len * core::mem::size_of::<T>(),
                            curr_node,
                        );
                        let b_local = verify_numa_locality(
                            b_ptr as *const u8,
                            b_len * core::mem::size_of::<T>(),
                            curr_node,
                        );

                        if !a_local || !b_local {
                            #[cfg(all(debug_assertions, feature = "std"))]
                            {
                                static WARNED: core::sync::atomic::AtomicBool =
                                    core::sync::atomic::AtomicBool::new(false);
                                if !WARNED.load(core::sync::atomic::Ordering::Relaxed)
                                    && !WARNED.swap(true, core::sync::atomic::Ordering::Relaxed)
                                {
                                    std::eprintln!(
                                        "WARNING [hermes-simd]: Cross-node NUMA memory access \
                                         detected (thread node: {curr_node}). Re-routing AMX → \
                                         AVX-512 to mitigate latency."
                                    );
                                }
                            }
                            if has_avx512 {
                                return DispatchDecision::Avx512;
                            }
                            return DispatchDecision::Scalar;
                        }
                    }
                }
                return DispatchDecision::Amx;
            }

            if has_avx512 {
                return DispatchDecision::Avx512;
            }

            // 256-bit VNNI tier: client parts (Alder Lake+, Zen 5) that have
            // `vpdpbusd` on YMM but no AVX-512. Kernels lacking an AVX-VNNI
            // implementation (e.g. bf16 tiles) treat this decision as scalar.
            if crate::cpu::has_avx_vnni() {
                return DispatchDecision::AvxVnni;
            }
        }

        let _ = (m, n, k, a_ptr, a_len, b_ptr, b_len);
        DispatchDecision::Scalar
    }
}
