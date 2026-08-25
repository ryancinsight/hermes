//! Implementation of the `#[runtime_dispatch]` attribute macro.
//!
//! Parses a comma-separated list of target features (e.g. `avx512f, avx2, neon, scalar`)
//! and wraps the annotated generic function in a public dispatching function that:
//! 1. Checks compile-time `cfg!(target_feature = "...")` flags first (zero runtime cost).
//! 2. Falls back to runtime `is_x86_feature_detected!` / `is_aarch64_feature_detected!`.
//! 3. Falls back to the scalar implementation.

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
                format!("unknown dispatch target `{other}`; expected one of: avx512f, avx2, neon, scalar"),
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

    fn arch_marker(&self) -> proc_macro2::TokenStream {
        match self {
            Self::Avx512f => quote!(hermes_simd_intrinsics::Avx512),
            Self::Avx2 => quote!(hermes_simd_intrinsics::Avx2),
            Self::Neon => quote!(hermes_simd_intrinsics::Neon),
            Self::Scalar => quote!(hermes_simd_intrinsics::Scalar),
        }
    }

    fn target_arch_cfg(&self) -> proc_macro2::TokenStream {
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
    for tt in stream {
        match tt {
            proc_macro2::TokenTree::Group(group) => {
                let replaced = replace_ident(group.stream(), target, replacement);
                let mut new_group = proc_macro2::Group::new(group.delimiter(), replaced);
                new_group.set_span(group.span());
                result.extend(Some(proc_macro2::TokenTree::Group(new_group)));
            }
            proc_macro2::TokenTree::Ident(ident) if ident == *target => {
                result.extend(replacement.clone());
            }
            other => {
                result.extend(Some(other));
            }
        }
    }
    result
}

#[expect(
    clippy::too_many_arguments,
    reason = "The dispatcher generator forwards the complete macro expansion inputs"
)]
#[expect(
    clippy::too_many_lines,
    reason = "The generator keeps target-specific helper and dispatch-arm construction together"
)]
fn generate_dispatcher(
    arch_cfg: &TokenStream,
    active_targets: &[DispatchTarget],
    dispatch_name: &Ident,
    inner_name: &Ident,
    inner_args: &syn::punctuated::Punctuated<syn::FnArg, syn::token::Comma>,
    inner_ret: &syn::ReturnType,
    visibility: &TokenStream,
    other_params: &[syn::GenericParam],
    other_param_tokens: &[TokenStream],
    call_args: &[TokenStream],
    arch_ident: &Ident,
    original_where_clause: Option<&syn::WhereClause>,
    doc_attrs: &[syn::Attribute],
) -> TokenStream {
    let mut helper_fns = Vec::new();
    let mut dispatch_arms = Vec::new();

    let other_params_tokens = quote!(#(#other_params),*);

    for target in active_targets {
        if matches!(target, DispatchTarget::Scalar) {
            continue;
        }
        let feat = target
            .feature_str()
            .expect("DispatchTarget::feature_str is Some for all non-Scalar variants");
        let arch = target.arch_marker();
        let arch_cfg = target.target_arch_cfg();
        let helper_name = format_ident!("{}_{}", dispatch_name, feat.replace(['-', ','], "_"));

        let helper_generics = if other_params.is_empty() {
            quote!()
        } else {
            let replaced_params = replace_ident(other_params_tokens.clone(), arch_ident, &arch);
            quote!(<#replaced_params>)
        };

        let helper_where = if let Some(wc) = original_where_clause {
            let wc_tokens = quote!(#wc);
            let replaced_wc = replace_ident(wc_tokens, arch_ident, &arch);
            quote!(#replaced_wc)
        } else {
            quote!()
        };

        let helper_turbofish = if other_param_tokens.is_empty() {
            quote!()
        } else {
            quote!(::<#(#other_param_tokens),*>)
        };

        let inner_turbofish = if other_param_tokens.is_empty() {
            quote!(::<#arch>)
        } else {
            quote!(::<#(#other_param_tokens,)* #arch>)
        };

        let tf_attrs: Vec<TokenStream> = feat
            .split(',')
            .map(|f| quote!(#[target_feature(enable = #f)]))
            .collect();

        helper_fns.push(quote! {
            #[cfg(#arch_cfg)]
            #(#tf_attrs)*
            #[inline]
            unsafe fn #helper_name #helper_generics(#inner_args) #inner_ret #helper_where {
                #inner_name #inner_turbofish(#(#call_args),*)
            }
        });

        // Compile-time cfg! check arm
        let ct_cfg_expr = {
            let features: Vec<&str> = feat.split(',').collect();
            let cfgs: Vec<TokenStream> = features
                .iter()
                .map(|f| quote!(cfg!(target_feature = #f)))
                .collect();
            quote!(#(#cfgs)&&*)
        };

        let ct_arm = quote! {
            #[cfg(#arch_cfg)]
            {
                if #ct_cfg_expr {
                    return unsafe { #helper_name #helper_turbofish(#(#call_args),*) };
                }
            }
        };

        // Runtime detection arm. `is_x86_feature_detected!` requires std, so
        // the arm is additionally gated on the consuming crate's `std`
        // feature; no_std builds keep the compile-time cfg! arms and the
        // scalar fallback only.
        let rt_arm = match target {
            DispatchTarget::Avx512f => quote! {
                #[cfg(all(#arch_cfg, feature = "std"))]
                {
                    if std::is_x86_feature_detected!("avx512f") {
                        return unsafe { #helper_name #helper_turbofish(#(#call_args),*) };
                    }
                }
            },
            DispatchTarget::Avx2 => quote! {
                #[cfg(all(#arch_cfg, feature = "std"))]
                {
                    if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
                        return unsafe { #helper_name #helper_turbofish(#(#call_args),*) };
                    }
                }
            },
            DispatchTarget::Neon => quote! {
                #[cfg(#arch_cfg)]
                {
                    // NEON is mandatory on aarch64
                    return unsafe { #helper_name #helper_turbofish(#(#call_args),*) };
                }
            },
            DispatchTarget::Scalar => quote! {},
        };

        dispatch_arms.push(quote! {
            #ct_arm
            #rt_arm
        });
    }

    // Scalar fallback
    let scalar_arch = quote!(hermes_simd_intrinsics::Scalar);
    let scalar_turbofish = if other_param_tokens.is_empty() {
        quote!(::<#scalar_arch>)
    } else {
        quote!(::<#(#other_param_tokens,)* #scalar_arch>)
    };
    let scalar_fallback = quote! {
        #inner_name #scalar_turbofish(#(#call_args),*)
    };

    let dispatcher_generics = if other_params.is_empty() {
        quote!()
    } else {
        quote!(<#(#other_params),*>)
    };

    // Filter bounds on the architecture parameter out of the dispatcher's where clause
    // and specialized arch bounds for each active target
    let mut arch_bounds = Vec::new();
    if let Some(wc) = original_where_clause {
        for pred in &wc.predicates {
            if let syn::WherePredicate::Type(pred_ty) = pred {
                if let syn::Type::Path(type_path) = &pred_ty.bounded_ty {
                    if type_path.path.is_ident(arch_ident) {
                        arch_bounds.push(pred_ty.clone());
                    }
                }
            }
        }
    }

    let mut specialized_bounds = Vec::new();
    for target in active_targets {
        let arch = target.arch_marker();
        for bound in &arch_bounds {
            let bound_tokens = quote!(#bound);
            let replaced = replace_ident(bound_tokens, arch_ident, &arch);
            specialized_bounds.push(replaced);
        }
    }

    let non_arch_predicates: Vec<syn::WherePredicate> = original_where_clause
        .map(|wc| {
            wc.predicates
                .iter()
                .filter(|pred| {
                    if let syn::WherePredicate::Type(pred_ty) = pred {
                        if let syn::Type::Path(type_path) = &pred_ty.bounded_ty {
                            if type_path.path.is_ident(arch_ident) {
                                return false;
                            }
                        }
                    }
                    true
                })
                .cloned()
                .collect()
        })
        .unwrap_or_default();

    let dispatcher_where = quote! {
        where
            #(#non_arch_predicates,)*
            #(#specialized_bounds,)*
    };

    let unreachable_code_expectation = if active_targets
        .iter()
        .any(|target| matches!(target, DispatchTarget::Neon))
    {
        quote! {
            #[expect(
                unreachable_code,
                reason = "Generated architecture arms are cfg-selected before the scalar fallback"
            )]
        }
    } else {
        quote!()
    };

    quote! {
        #(#doc_attrs)*
        #[cfg(#arch_cfg)]
        #[inline(always)]
        #unreachable_code_expectation
        #visibility fn #dispatch_name #dispatcher_generics(#inner_args) #inner_ret #dispatcher_where {
            #(#helper_fns)*
            #(#dispatch_arms)*
            #scalar_fallback
        }
    }
}

#[expect(
    clippy::too_many_lines,
    reason = "The macro entry point coordinates parsing and three target dispatch expansions"
)]
pub fn expand(args: TokenStream, item: TokenStream) -> Result<TokenStream> {
    // Parse target list from attribute args
    let targets: Vec<DispatchTarget> = {
        let parser = Punctuated::<Ident, Token![,]>::parse_terminated;
        let idents = parser.parse2(args)?;
        idents
            .iter()
            .map(DispatchTarget::from_ident)
            .collect::<Result<_>>()?
    };

    // Parse the annotated function
    let inner_fn: ItemFn = syn::parse2(item)?;
    let inner_name = &inner_fn.sig.ident;
    let inner_args = &inner_fn.sig.inputs;
    let inner_ret = &inner_fn.sig.output;
    let inner_vis = &inner_fn.vis;
    // The generated dispatcher is what callers see, so it inherits the
    // annotated function's documentation. Without this the dispatcher is
    // undocumented, which `#![deny(missing_docs)]` rejects for any `pub`
    // kernel -- the reason every dispatch module here had to stay crate-local.
    let doc_attrs: Vec<syn::Attribute> = inner_fn
        .attrs
        .iter()
        .filter(|a| a.path().is_ident("doc"))
        .cloned()
        .collect();

    // Find type/const parameters and identify the architecture parameter
    let mut type_params = Vec::new();
    for param in &inner_fn.sig.generics.params {
        if let syn::GenericParam::Type(ty) = param {
            type_params.push(ty);
        }
    }

    if type_params.is_empty() {
        return Err(Error::new(
            inner_name.span(),
            "#[runtime_dispatch] target function must have at least one generic type parameter for the architecture"
        ));
    }

    // Last type parameter is the architecture parameter
    let arch_param = type_params
        .last()
        .expect("type_params verified non-empty by preceding length check");
    let arch_ident = &arch_param.ident;

    // The other parameters (lifetimes, consts, other types)
    let other_params: Vec<syn::GenericParam> = inner_fn
        .sig
        .generics
        .params
        .iter()
        .filter(|param| {
            if let syn::GenericParam::Type(ty) = param {
                ty.ident != *arch_ident
            } else {
                true
            }
        })
        .cloned()
        .collect();

    // Other parameter identifiers/tokens to pass to turbofish
    let other_param_tokens: Vec<TokenStream> = inner_fn
        .sig
        .generics
        .params
        .iter()
        .filter_map(|param| match param {
            syn::GenericParam::Type(ty) if ty.ident != *arch_ident => {
                let id = &ty.ident;
                Some(quote!(#id))
            }
            syn::GenericParam::Const(c) => {
                let id = &c.ident;
                Some(quote!(#id))
            }
            _ => None,
        })
        .collect();

    // Build the list of call argument identifiers (strip self/type info)
    let call_args: Vec<TokenStream> = inner_args
        .iter()
        .filter_map(|arg| {
            if let FnArg::Typed(pat_type) = arg {
                if let Pat::Ident(pat_ident) = pat_type.pat.as_ref() {
                    let ident = &pat_ident.ident;
                    return Some(quote!(#ident));
                }
            }
            None
        })
        .collect();

    // The public dispatch function has the same name minus any generic suffix
    // Convention: inner fn is named `my_kernel_impl`, dispatcher becomes `my_kernel`
    let dispatch_name = {
        let name_str = inner_name.to_string();
        let stripped = name_str
            .strip_suffix("_kernel")
            .or_else(|| name_str.strip_suffix("_impl"))
            .unwrap_or(&name_str);
        Ident::new(stripped, Span::call_site())
    };

    let x86_targets: Vec<DispatchTarget> = targets
        .iter()
        .filter(|t| {
            matches!(
                t,
                DispatchTarget::Avx512f | DispatchTarget::Avx2 | DispatchTarget::Scalar
            )
        })
        .cloned()
        .collect();

    let aarch64_targets: Vec<DispatchTarget> = targets
        .iter()
        .filter(|t| matches!(t, DispatchTarget::Neon | DispatchTarget::Scalar))
        .cloned()
        .collect();

    let fallback_targets: Vec<DispatchTarget> = vec![DispatchTarget::Scalar];

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
        inner_fn.sig.generics.where_clause.as_ref(),
        &doc_attrs,
    );

    Ok(quote! {
        // Keep the inner generic kernel function (private/hidden)
        #inner_fn

        // Generate the three dispatcher functions
        #x86_dispatcher
        #aarch64_dispatcher
        #fallback_dispatcher
    })
}

#[cfg(test)]
#[path = "dispatch_tests.rs"]
mod tests;
