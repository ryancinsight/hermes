//! Zero-copy SIMD chunk iterators over [`SimdView`](crate::view::SimdView).
//!
//! # Design
//!
//! `SimdChunks` iterates non-overlapping sub-views of exactly `LANE_COUNT` elements
//! from a `SimdView`, leaving the remainder (the scalar tail) accessible via
//! [`SimdChunks::remainder`]. This is the canonical pattern for SIMD loop bodies:
//!
//! ```rust,ignore
//! let mut chunks = view.simd_chunks();
//! for chunk in &mut chunks {
//!     // chunk: SimdView<'_, T, Arch, Align>
//!     // process full LANE_COUNT-wide chunk
//! }
//! let tail = chunks.remainder();
//! // process tail[i] in scalar loop
//! ```
//!
//! # Module Organization
//!
//! | Sub-module | Contents |
//! |---|---|
//! | [`chunks`] | `SimdChunks`, `SimdChunksMut` |
//! | [`zip`] | `ZipChunks`, `ZipChunksMut` |

pub mod chunks;
pub mod zip;

pub use chunks::{SimdChunks, SimdChunksMut};
pub use zip::{ZipChunks, ZipChunksMut};
