//! Zero-copy SIMD chunk iterators over [`SimdView`](crate::view::SimdView).
//!
//! # Design
//!
//! `SimdChunks` iterates non-overlapping [`SimdChunk`](crate::view::SimdChunk)
//! values of exactly `LANE_COUNT` elements from a `SimdView`, leaving the scalar tail accessible via
//! [`SimdChunks::remainder`]. This is the canonical pattern for SIMD loop bodies:
//!
//! ```rust,ignore
//! let mut chunks = view.simd_chunks();
//! for chunk in &mut chunks {
//!     let vector = chunk.load();
//!     // process one complete architecture register
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
//! | [`io`] | `SimdIoChunks` for lockstep planar inputs and outputs |
//! | [`zip`] | `ZipChunks`, `ZipChunksMut` |

pub mod chunks;
pub mod io;
pub mod zip;

pub use chunks::{SimdChunks, SimdChunksMut};
pub use io::SimdIoChunks;
pub use zip::{ZipChunks, ZipChunksMut};
