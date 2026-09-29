#![expect(
    clippy::float_cmp,
    reason = "Tiling tests compare exact manufactured matrix values"
)]
#![expect(
    clippy::items_after_statements,
    reason = "The tiling harness keeps architecture-specific imports beside the gated case"
)]

use hermes_simd::*;

#[cfg(feature = "mnemosyne-memory")]
mod packed_panel_allocations {
    //! Pins how often the packed path allocates.
    //!
    //! One allocate/free pair per packed call is the current, deliberate
    //! behaviour: retaining the panel per thread was measured and rejected
    //! (HS-GEMM-PANEL-REUSE-2026-08-29) because it costs 69x at n = 512, far
    //! more than the allocation it removes. This census is the instrument that
    //! comparison was run against, kept so the next attempt starts from a
    //! measured baseline rather than a guess.

    use super::*;
    use core::cell::Cell;
    use core::sync::atomic::{AtomicUsize, Ordering};

    static ALLOCATIONS: AtomicUsize = AtomicUsize::new(0);
    static FREES: AtomicUsize = AtomicUsize::new(0);

    std::thread_local! {
        static COUNTING: Cell<bool> = const { Cell::new(false) };
    }

    unsafe extern "C" fn record_allocation(_ptr: *mut core::ffi::c_void, _bytes: usize) {
        COUNTING.with(|counting| {
            if counting.get() {
                ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
            }
        });
    }

    unsafe extern "C" fn record_free(_ptr: *mut core::ffi::c_void, _bytes: usize) {
        COUNTING.with(|counting| {
            if counting.get() {
                FREES.fetch_add(1, Ordering::Relaxed);
            }
        });
    }

    struct HookGuard;

    impl HookGuard {
        fn install() -> Self {
            ALLOCATIONS.store(0, Ordering::Relaxed);
            FREES.store(0, Ordering::Relaxed);
            mnemosyne::register_alloc_hook(Some(record_allocation));
            mnemosyne::register_free_hook(Some(record_free));
            Self
        }
    }

    impl Drop for HookGuard {
        fn drop(&mut self) {
            COUNTING.with(|counting| counting.set(false));
            mnemosyne::register_alloc_hook(None);
            mnemosyne::register_free_hook(None);
        }
    }

    fn counts() -> (usize, usize) {
        (
            ALLOCATIONS.load(Ordering::Relaxed),
            FREES.load(Ordering::Relaxed),
        )
    }

    struct GemmCase {
        a: Vec<f32>,
        b: Vec<f32>,
        c: Vec<f32>,
        n: usize,
        k: usize,
    }

    impl GemmCase {
        const M: usize = 2;

        fn new(n: usize, k: usize) -> Self {
            Self {
                a: vec![1.0; Self::M * k],
                b: vec![0.5; k * n],
                c: vec![0.0; Self::M * n],
                n,
                k,
            }
        }

        fn run(&mut self, expected: f32) {
            tiled_gemm(&self.a, &self.b, &mut self.c, Self::M, self.n, self.k).unwrap();
            assert!(self.c.iter().all(|&value| value == expected));
        }
    }

    #[test]
    fn packed_gemm_allocates_its_panel_once_per_call() {
        let _hooks = HookGuard::install();

        std::thread::spawn(|| {
            // Packed-panel size scales only with K here: 128, then 64, then
            // 256, then 16_385 (past 512 KiB even at one lane). Every one of
            // them is call-owned, so the size relationships do not matter to
            // the counts -- which is the point of the measurement.
            let mut initial = GemmCase::new(1024, 128);
            let mut smaller = GemmCase::new(2048, 64);
            let mut growth = GemmCase::new(512, 256);
            let mut oversized = GemmCase::new(64, 16_385);

            COUNTING.with(|counting| counting.set(true));

            initial.run(64.0);
            assert_eq!(
                counts(),
                (1, 1),
                "the panel is allocated and freed per call"
            );
            initial.run(128.0);
            assert_eq!(
                counts(),
                (2, 2),
                "a repeat call at the same size allocates again"
            );

            smaller.run(32.0);
            assert_eq!(counts(), (3, 3), "a smaller panel allocates again");

            growth.run(128.0);
            assert_eq!(counts(), (4, 4), "a larger panel allocates again");
            growth.run(256.0);
            assert_eq!(counts(), (5, 5), "and again at the same larger size");

            oversized.run(8192.5);
            assert_eq!(counts(), (6, 6), "oversized panels are call-owned too");
        })
        .join()
        .expect("GEMM allocation census worker must complete");

        // Nothing is retained, so teardown has nothing left to release.
        assert_eq!(
            counts(),
            (6, 6),
            "thread teardown releases no retained panel"
        );
    }
}


#[path = "tiling_tests/tiling_ops.rs"]
mod tiling_ops;
