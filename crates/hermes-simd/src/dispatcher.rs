//! Dynamic runtime dispatch choosing the optimal execution backend.

use crate::cpu::{AmxSupport, Avx512Support};
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

#[cfg(all(target_arch = "x86_64", feature = "std"))]
#[inline]
fn is_multi_numa() -> bool {
    use std::sync::OnceLock;
    static MULTI_NUMA: OnceLock<bool> = OnceLock::new();
    *MULTI_NUMA.get_or_init(|| {
        themis::CpuTopology::detect().is_some_and(|topology| topology.numa_nodes().len() > 1)
    })
}

#[cfg(all(target_arch = "x86_64", not(feature = "std")))]
#[inline]
fn is_multi_numa() -> bool {
    false
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
    pub fn select_backend<T: AmxSupport + Avx512Support>(
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
            let has_amx = <T as AmxSupport>::has_amx();
            let has_avx512 = <T as Avx512Support>::has_avx512();

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
                            a_ptr.cast::<u8>(),
                            a_len * core::mem::size_of::<T>(),
                            curr_node,
                        );
                        let b_local = verify_numa_locality(
                            b_ptr.cast::<u8>(),
                            b_len * core::mem::size_of::<T>(),
                            curr_node,
                        );

                        if !a_local || !b_local {
                            #[cfg(feature = "std")]
                            {
                                static WARNED: core::sync::atomic::AtomicBool =
                                    core::sync::atomic::AtomicBool::new(false);
                                if !WARNED.load(core::sync::atomic::Ordering::Relaxed)
                                    && !WARNED.swap(true, core::sync::atomic::Ordering::Relaxed)
                                {
                                    emit_numa_downgrade(curr_node);
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

#[cfg(all(target_arch = "x86_64", feature = "std"))]
fn emit_numa_downgrade(numa_node: u32) {
    tracing::warn!(
        target: "hermes_simd::dispatcher",
        numa_node,
        from_backend = "amx",
        to_backend = "avx512",
        reason = "remote_input_memory",
        "AMX dispatch downgraded because input memory is remote to the current NUMA node"
    );
}

#[cfg(all(test, target_arch = "x86_64", feature = "std"))]
mod tests {
    use std::io::{self, Write};
    use std::sync::{Arc, Mutex};

    use tracing_subscriber::fmt::MakeWriter;

    use super::emit_numa_downgrade;

    #[derive(Clone, Default)]
    struct Buffer(Arc<Mutex<Vec<u8>>>);

    struct BufferWriter(Arc<Mutex<Vec<u8>>>);

    impl Write for BufferWriter {
        fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
            let mut output = self
                .0
                .lock()
                .map_err(|_| io::Error::other("event buffer lock poisoned"))?;
            output.extend_from_slice(bytes);
            Ok(bytes.len())
        }

        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }

    impl<'writer> MakeWriter<'writer> for Buffer {
        type Writer = BufferWriter;

        fn make_writer(&'writer self) -> Self::Writer {
            BufferWriter(Arc::clone(&self.0))
        }
    }

    #[test]
    fn numa_downgrade_emits_structured_warning() {
        let buffer = Buffer::default();
        let subscriber = tracing_subscriber::fmt()
            .with_ansi(false)
            .with_writer(buffer.clone())
            .finish();

        tracing::subscriber::with_default(subscriber, || emit_numa_downgrade(7));

        let output = buffer
            .0
            .lock()
            .expect("event buffer remains available")
            .clone();
        let output = String::from_utf8(output).expect("subscriber output is UTF-8");
        assert!(output.contains("AMX dispatch downgraded"));
        assert!(output.contains("numa_node=7"));
        assert!(output.contains("from_backend=\"amx\""));
        assert!(output.contains("to_backend=\"avx512\""));
    }
}
