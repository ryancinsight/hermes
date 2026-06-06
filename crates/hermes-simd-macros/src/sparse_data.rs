//! Implementation of the `#[derive(SparseData)]` derive macro.
//!
//! Generates `SparseFormat` marker trait boilerplate for data structs annotated with
//! `#[sparse_format(name = "...")]`.

use proc_macro2::TokenStream;
use quote::quote;
use syn::{DeriveInput, Lit, Result};

pub fn expand(item: TokenStream) -> Result<TokenStream> {
    let input: DeriveInput = syn::parse2(item)?;
    let struct_name = &input.ident;
    let (impl_generics, ty_generics, where_clause) = input.generics.split_for_impl();

    // Extract #[sparse_format(name = "...")] attribute
    let format_name = input
        .attrs
        .iter()
        .find(|attr| attr.path().is_ident("sparse_format"))
        .and_then(|attr| {
            let mut name_val = None;
            let _ = attr.parse_nested_meta(|meta| {
                if meta.path.is_ident("name") {
                    let value = meta.value()?;
                    let lit: Lit = value.parse()?;
                    if let Lit::Str(s) = lit {
                        name_val = Some(s.value());
                    }
                }
                Ok(())
            });
            name_val
        })
        .unwrap_or_else(|| struct_name.to_string());

    Ok(quote! {
        impl #impl_generics #struct_name #ty_generics #where_clause {
            /// The canonical name for this sparse format.
            pub const FORMAT_NAME: &'static str = #format_name;
        }
    })
}
