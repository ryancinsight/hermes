//! Horizontal and pairwise SIMD reductions over [`SimdView`](crate::view::SimdView).
//!
//! # Safety
//!
//! Every kernel call below is `#[target_feature]`-gated and is therefore sound
//! only on a host implementing `Arch`. That holds by construction rather than by
//! inspection: [`SimdView::new`](crate::view::SimdView::new) returns `None` for
//! an architecture the host cannot execute, and the sparse and copy-on-write
//! constructors assert the same condition, so possessing one of these
//! arch-parameterized values *is* the proof. Per-site `SAFETY` comments record
//! only the obligations that go beyond it -- pointer provenance, bounds, and
//! alignment.
//!
//! The root is a passthrough facade: each operation family lives in its own
//! leaf, and no implementation is declared here.

mod arg;
mod fold;
mod popcount;
