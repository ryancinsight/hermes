//! Procedural macros for the `hermes-simd` workspace.
//!
//! # Macros
//!
//! - `#[runtime_dispatch(avx512f, avx2, neon, scalar)]` — generates a CPU-feature-dispatched
//!   wrapper function that calls monomorphized specializations in priority order.
//! - `#[derive(SparseData)]` — generates `SparseFormat` boilerplate for data structs.
//!
//! This crate executes no `unsafe` itself (the `unsafe` it emits lives in the
//! generated token streams, compiled in the consumer crate), so it forbids it.
#![forbid(unsafe_code)]
// Library-only denials; see the note in hermes-simd-core's crate root.
#![deny(missing_docs)]
#![deny(clippy::unwrap_used, clippy::print_stdout, clippy::print_stderr)]
#![cfg_attr(
    test,
    allow(clippy::unwrap_used, clippy::print_stdout, clippy::print_stderr)
)]

extern crate proc_macro;
use proc_macro::TokenStream;

mod dispatch;
mod sparse_data;

/// Attribute macro that wraps a generic kernel function with CPU-feature runtime dispatch.
///
/// The final generic type parameter is the architecture parameter. Its inline
/// bounds and `where` predicates are specialized for each selected backend.
/// Borrowed outputs follow Rust lifetime elision: one input lifetime is reused,
/// while signatures with multiple input lifetimes require an explicit output
/// lifetime.
///
/// # Usage
///
/// ```rust,ignore
/// use hermes_simd_macros::runtime_dispatch;
///
/// #[runtime_dispatch(avx512f, avx2, neon, scalar)]
/// pub fn sum_f32_kernel<A: SimdArch + SimdKernel<f32>>(data: &[f32]) -> f32 { /* ... */ }
/// ```
///
/// Generates:
/// ```rust,ignore
/// pub fn sum_f32(data: &[f32]) -> f32 {
///     // compile-time checks first
///     // runtime is_x86_feature_detected! fallback
///     // scalar fallback
/// }
/// ```
#[proc_macro_attribute]
pub fn runtime_dispatch(args: TokenStream, item: TokenStream) -> TokenStream {
    dispatch::expand(args.into(), item.into())
        .unwrap_or_else(|e| e.to_compile_error())
        .into()
}

/// Derive macro that generates `SparseFormat` trait implementation boilerplate
/// for sparse data structs.
///
/// # Usage
///
/// ```rust,ignore
/// use hermes_simd_macros::SparseData;
///
/// #[derive(SparseData)]
/// #[sparse_format(name = "CSR")]
/// pub struct CsrData<'a, T> { /* ... */ }
/// ```
#[proc_macro_derive(SparseData, attributes(sparse_format))]
pub fn derive_sparse_data(item: TokenStream) -> TokenStream {
    sparse_data::expand(item.into())
        .unwrap_or_else(|e| e.to_compile_error())
        .into()
}
