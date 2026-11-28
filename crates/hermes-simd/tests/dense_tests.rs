#![expect(
    clippy::float_cmp,
    reason = "These integration tests assert exact manufactured dense reference values"
)]
#![expect(
    clippy::too_many_lines,
    reason = "The dense backend matrix is one shared value-semantic conformance test"
)]
#[path = "dense_tests/dense_arithmetic.rs"]
mod dense_arithmetic;
#[path = "dense_tests/dense_dispatch.rs"]
mod dense_dispatch;
