//! A missing platform locality remains typed absence through both providers.

#![cfg(any(miri, not(any(target_os = "linux", windows))))]

use tracing_subscriber::fmt::format::FmtSpan;

#[test]
fn unreported_locality_is_not_fabricated_node_zero() {
    let subscriber = tracing_subscriber::fmt()
        .with_test_writer()
        // Preserve Miri isolation: tracing must not query the wall clock.
        .without_time()
        .with_span_events(FmtSpan::FULL)
        .finish();
    // This integration binary has one case. Global setup is also available
    // when the library's tracing dependency is compiled without `std`.
    tracing::subscriber::set_global_default(subscriber)
        .expect("one subscriber in the isolated topology test binary");
    let _test = tracing::info_span!("topology_absence").entered();
    let arrange = tracing::info_span!("arrange").entered();
    let absent_index: Option<u32> = None;
    let absent_node: Option<themis::NumaNodeId> = None;
    drop(arrange);

    let act = tracing::info_span!("act").entered();
    let current = hermes_simd::current_numa_node();
    let refreshed = hermes_simd::refresh_numa_node();
    let provider = themis::try_current_numa_node();
    drop(act);

    let _assert = tracing::info_span!("assert").entered();
    assert_eq!(current, absent_index);
    assert_eq!(refreshed, absent_index);
    assert_eq!(provider, absent_node);
}
