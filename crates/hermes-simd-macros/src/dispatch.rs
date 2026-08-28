//! Implementation of the `#[runtime_dispatch]` attribute macro.
//!
//! Parses a comma-separated list of target features (e.g. `avx512f, avx2, neon, scalar`)
//! and wraps the annotated generic function in a public dispatching function that:
//! 1. Checks compile-time `cfg!(target_feature = "...")` flags first (zero runtime cost).
//! 2. Falls back to runtime `is_x86_feature_detected!` / `is_aarch64_feature_detected!`.
//! 3. Falls back to the scalar implementation.

mod frame;
mod generator;

use frame::{generate_avx2_frame_adapter, kernel_scalar_type, upper_camel_case};
use generator::generate_dispatcher;
use proc_macro2::{Span, TokenStream};
use quote::{format_ident, quote};
use syn::{parse::Parser, punctuated::Punctuated, Error, FnArg, Ident, ItemFn, Pat, Result, Token};

/// Recognized dispatch targets and their architecture context.
#[derive(Debug, Clone, PartialEq, Eq)]
enum DispatchTarget {
    Avx512f,
    Avx2,
    Neon,
    Scalar,
}

impl DispatchTarget {
    fn from_ident(id: &Ident) -> Result<Self> {
        match id.to_string().as_str() {
            "avx512f" => Ok(Self::Avx512f),
            "avx2" => Ok(Self::Avx2),
            "neon" => Ok(Self::Neon),
            "scalar" => Ok(Self::Scalar),
            other => Err(Error::new(
                id.span(),
                format!(
                    "unknown dispatch target `{other}`; expected one of: avx512f, avx2, neon, scalar"
                ),
            )),
        }
    }

    fn feature_str(&self) -> Option<&'static str> {
        match self {
            Self::Avx512f => Some("avx512f"),
            Self::Avx2 => Some("avx2,fma"),
            Self::Neon => Some("neon"),
            Self::Scalar => None,
        }
    }

    fn arch_marker(&self) -> TokenStream {
        match self {
            Self::Avx512f => quote!(hermes_simd_intrinsics::Avx512),
            Self::Avx2 => quote!(hermes_simd_intrinsics::Avx2),
            Self::Neon => quote!(hermes_simd_intrinsics::Neon),
            Self::Scalar => quote!(hermes_simd_intrinsics::Scalar),
        }
    }

    fn target_arch_cfg(&self) -> TokenStream {
        match self {
            Self::Avx512f | Self::Avx2 => {
                quote!(any(target_arch = "x86", target_arch = "x86_64"))
            }
            Self::Neon => quote!(target_arch = "aarch64"),
            Self::Scalar => quote!(all()),
        }
    }
}

fn replace_ident(stream: TokenStream, target: &Ident, replacement: &TokenStream) -> TokenStream {
    let mut result = TokenStream::new();
    for token in stream {
        match token {
            proc_macro2::TokenTree::Group(group) => {
                let replaced = replace_ident(group.stream(), target, replacement);
                let mut new_group = proc_macro2::Group::new(group.delimiter(), replaced);
                new_group.set_span(group.span());
                result.extend(Some(proc_macro2::TokenTree::Group(new_group)));
            }
            proc_macro2::TokenTree::Ident(ident) if ident == *target => {
                result.extend(replacement.clone());
            }
            other => result.extend(Some(other)),
        }
    }
    result
}

#[expect(
    clippy::too_many_lines,
    reason = "The macro entry point coordinates parsing and three target dispatch expansions"
)]
pub fn expand(args: TokenStream, item: TokenStream) -> Result<TokenStream> {
    let targets: Vec<DispatchTarget> = {
        let parser = Punctuated::<Ident, Token![,]>::parse_terminated;
        let idents = parser.parse2(args)?;
        idents
            .iter()
            .map(DispatchTarget::from_ident)
            .collect::<Result<_>>()?
    };

    let mut inner_fn: ItemFn = syn::parse2(item)?;

    // The target-feature helper is the feature-carrying frame. Forced inlining
    // places even large generic kernels inside it while leaving the same inner
    // function available to the scalar fallback.
    if !inner_fn
        .attrs
        .iter()
        .any(|attribute| attribute.path().is_ident("inline"))
    {
        inner_fn.attrs.push(syn::parse_quote!(#[inline(always)]));
    }
    let inner_name = &inner_fn.sig.ident;
    let inner_args = &inner_fn.sig.inputs;
    let inner_ret = &inner_fn.sig.output;
    let inner_vis = &inner_fn.vis;
    let doc_attrs: Vec<syn::Attribute> = inner_fn
        .attrs
        .iter()
        .filter(|attribute| attribute.path().is_ident("doc"))
        .cloned()
        .collect();

    let type_params: Vec<_> = inner_fn
        .sig
        .generics
        .params
        .iter()
        .filter_map(|parameter| {
            let syn::GenericParam::Type(parameter) = parameter else {
                return None;
            };
            Some(parameter)
        })
        .collect();

    if type_params.is_empty() {
        return Err(Error::new(
            inner_name.span(),
            "#[runtime_dispatch] target function must have at least one generic type parameter for the architecture",
        ));
    }

    let arch_param = type_params
        .last()
        .expect("type_params verified non-empty by preceding length check");
    let arch_ident = &arch_param.ident;
    let scalar_type = kernel_scalar_type(&inner_fn.sig.generics, arch_ident);

    let other_params: Vec<syn::GenericParam> = inner_fn
        .sig
        .generics
        .params
        .iter()
        .filter(|parameter| {
            if let syn::GenericParam::Type(parameter) = parameter {
                parameter.ident != *arch_ident
            } else {
                true
            }
        })
        .cloned()
        .collect();

    let other_param_tokens: Vec<TokenStream> = inner_fn
        .sig
        .generics
        .params
        .iter()
        .filter_map(|parameter| match parameter {
            syn::GenericParam::Type(parameter) if parameter.ident != *arch_ident => {
                let ident = &parameter.ident;
                Some(quote!(#ident))
            }
            syn::GenericParam::Const(parameter) => {
                let ident = &parameter.ident;
                Some(quote!(#ident))
            }
            _ => None,
        })
        .collect();

    let call_args: Vec<TokenStream> = inner_args
        .iter()
        .filter_map(|argument| {
            let FnArg::Typed(argument) = argument else {
                return None;
            };
            let Pat::Ident(pattern) = argument.pat.as_ref() else {
                return None;
            };
            let ident = &pattern.ident;
            Some(quote!(#ident))
        })
        .collect();

    let dispatch_name = {
        let name = inner_name.to_string();
        let stripped = name
            .strip_suffix("_kernel")
            .or_else(|| name.strip_suffix("_impl"))
            .unwrap_or(&name);
        Ident::new(stripped, Span::call_site())
    };
    let avx2_adapter_name = scalar_type.as_ref().map(|_| {
        format_ident!(
            "__Hermes{}Avx2Frame",
            upper_camel_case(&dispatch_name.to_string())
        )
    });
    let avx2_adapter = scalar_type
        .as_ref()
        .zip(avx2_adapter_name.as_ref())
        .map(|(scalar, adapter_name)| {
            generate_avx2_frame_adapter(
                adapter_name,
                inner_name,
                inner_args,
                inner_ret,
                &other_params,
                &other_param_tokens,
                &call_args,
                arch_ident,
                scalar,
                inner_fn.sig.generics.where_clause.as_ref(),
            )
        })
        .transpose()?;

    let x86_targets: Vec<DispatchTarget> = targets
        .iter()
        .filter(|target| {
            matches!(
                target,
                DispatchTarget::Avx512f | DispatchTarget::Avx2 | DispatchTarget::Scalar
            )
        })
        .cloned()
        .collect();
    let aarch64_targets: Vec<DispatchTarget> = targets
        .iter()
        .filter(|target| matches!(target, DispatchTarget::Neon | DispatchTarget::Scalar))
        .cloned()
        .collect();
    let fallback_targets = vec![DispatchTarget::Scalar];

    let x86_dispatcher = generate_dispatcher(
        &quote!(any(target_arch = "x86", target_arch = "x86_64")),
        &x86_targets,
        &dispatch_name,
        inner_name,
        inner_args,
        inner_ret,
        &quote!(#inner_vis),
        &other_params,
        &other_param_tokens,
        &call_args,
        arch_ident,
        &arch_param.bounds,
        scalar_type.as_ref(),
        avx2_adapter_name.as_ref(),
        inner_fn.sig.generics.where_clause.as_ref(),
        &doc_attrs,
    );

    let aarch64_dispatcher = generate_dispatcher(
        &quote!(target_arch = "aarch64"),
        &aarch64_targets,
        &dispatch_name,
        inner_name,
        inner_args,
        inner_ret,
        &quote!(#inner_vis),
        &other_params,
        &other_param_tokens,
        &call_args,
        arch_ident,
        &arch_param.bounds,
        scalar_type.as_ref(),
        None,
        inner_fn.sig.generics.where_clause.as_ref(),
        &doc_attrs,
    );

    let fallback_dispatcher = generate_dispatcher(
        &quote!(not(any(
            target_arch = "x86",
            target_arch = "x86_64",
            target_arch = "aarch64"
        ))),
        &fallback_targets,
        &dispatch_name,
        inner_name,
        inner_args,
        inner_ret,
        &quote!(#inner_vis),
        &other_params,
        &other_param_tokens,
        &call_args,
        arch_ident,
        &arch_param.bounds,
        scalar_type.as_ref(),
        None,
        inner_fn.sig.generics.where_clause.as_ref(),
        &doc_attrs,
    );

    Ok(quote! {
        #inner_fn
        #avx2_adapter
        #x86_dispatcher
        #aarch64_dispatcher
        #fallback_dispatcher
    })
}

#[cfg(test)]
#[path = "dispatch_tests.rs"]
mod tests;
