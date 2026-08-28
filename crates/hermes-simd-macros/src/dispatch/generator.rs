use proc_macro2::TokenStream;
use quote::{format_ident, quote};
use syn::{
    punctuated::Punctuated, FnArg, GenericParam, Ident, ReturnType, Token, Type, TypeParamBound,
};

use super::{replace_ident, DispatchTarget};

#[expect(
    clippy::too_many_arguments,
    reason = "The dispatcher generator forwards the complete macro expansion inputs"
)]
#[expect(
    clippy::too_many_lines,
    reason = "The generator keeps target-specific helper and dispatch-arm construction together"
)]
pub(super) fn generate_dispatcher(
    arch_cfg: &TokenStream,
    active_targets: &[DispatchTarget],
    dispatch_name: &Ident,
    inner_name: &Ident,
    inner_args: &syn::punctuated::Punctuated<FnArg, syn::token::Comma>,
    inner_ret: &ReturnType,
    visibility: &TokenStream,
    other_params: &[GenericParam],
    other_param_tokens: &[TokenStream],
    call_args: &[TokenStream],
    arch_ident: &Ident,
    arch_inline_bounds: &Punctuated<TypeParamBound, Token![+]>,
    scalar_type: Option<&Type>,
    avx2_adapter_name: Option<&Ident>,
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
        let feature_frames = if matches!(target, DispatchTarget::Avx2) && scalar_type.is_some() {
            vec![("avx2,fma,f16c", Some(true)), ("avx2,fma", Some(false))]
        } else {
            vec![(
                target
                    .feature_str()
                    .expect("non-Scalar dispatch targets carry target features"),
                None,
            )]
        };

        for (feat, requires_f16c) in feature_frames {
            let arch = target.arch_marker();
            let arch_cfg = target.target_arch_cfg();
            let helper_name = format_ident!("{}_{}", dispatch_name, feat.replace(['-', ','], "_"));

            let helper_generics = if other_params.is_empty() {
                quote!()
            } else {
                let replaced_params = replace_ident(other_params_tokens.clone(), arch_ident, &arch);
                quote!(<#replaced_params>)
            };

            let mut helper_predicates: Vec<TokenStream> = original_where_clause
                .into_iter()
                .flat_map(|where_clause| &where_clause.predicates)
                .map(|predicate| replace_ident(quote!(#predicate), arch_ident, &arch))
                .collect();
            if !arch_inline_bounds.is_empty() {
                helper_predicates.push(replace_ident(
                    quote!(#arch_ident: #arch_inline_bounds),
                    arch_ident,
                    &arch,
                ));
            }
            if matches!(requires_f16c, Some(true)) {
                let scalar = scalar_type.expect("F16C frames require a scalar type");
                helper_predicates.push(quote!(
                    #scalar: hermes_simd_intrinsics::x86_64::avx2_f16::Avx2FrameScalar
                ));
            }
            let helper_where = if helper_predicates.is_empty() {
                quote!()
            } else {
                quote!(where #(#helper_predicates,)*)
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

            let target_feature_attrs: Vec<TokenStream> = feat
                .split(',')
                .map(|feature| quote!(#[target_feature(enable = #feature)]))
                .collect();

            let helper_body = if matches!(requires_f16c, Some(true)) {
                let scalar = scalar_type.expect("F16C frames require a scalar type");
                let adapter = avx2_adapter_name.expect("F16C frames require a callback adapter");
                let adapter = if other_param_tokens.is_empty() {
                    quote!(#adapter(core::marker::PhantomData))
                } else {
                    quote!(#adapter::<#(#other_param_tokens),*>(core::marker::PhantomData))
                };
                quote! {
                    // SAFETY: this helper is entered only after AVX2, FMA, and
                    // F16C support is proved, and its target frame carries all
                    // three features into the selected kernel specialization.
                    unsafe {
                        <#scalar as hermes_simd_intrinsics::x86_64::avx2_f16::Avx2FrameScalar>::call_avx2_frame(
                            #adapter,
                            (#(#call_args,)*),
                        )
                    }
                }
            } else {
                quote!(#inner_name #inner_turbofish(#(#call_args),*))
            };

            helper_fns.push(quote! {
                #[cfg(#arch_cfg)]
                #(#target_feature_attrs)*
                #[inline]
                unsafe fn #helper_name #helper_generics(#inner_args) #inner_ret #helper_where {
                    #helper_body
                }
            });

            let features: Vec<&str> = feat.split(',').collect();
            let compile_time_cfgs: Vec<TokenStream> = features
                .iter()
                .map(|feature| quote!(cfg!(target_feature = #feature)))
                .collect();
            let compile_time_cfg_expr = quote!(#(#compile_time_cfgs)&&*);
            let scalar_frame_guard = match (requires_f16c, scalar_type) {
                (Some(true), Some(scalar)) => {
                    quote!(<#arch as hermes_simd_core::kernel::SimdStorage<#scalar>>::REQUIRES_F16C)
                }
                (Some(false), Some(scalar)) => {
                    quote!(!<#arch as hermes_simd_core::kernel::SimdStorage<#scalar>>::REQUIRES_F16C)
                }
                _ => quote!(true),
            };

            let compile_time_arm = quote! {
                #[cfg(#arch_cfg)]
                {
                    if #scalar_frame_guard && #compile_time_cfg_expr {
                        return unsafe { #helper_name #helper_turbofish(#(#call_args),*) };
                    }
                }
            };

            // Runtime detection requires std. no_std builds retain compile-time
            // feature selection and the scalar fallback.
            let runtime_arm = match target {
                DispatchTarget::Avx512f => quote! {
                    #[cfg(all(#arch_cfg, feature = "std"))]
                    {
                        if std::is_x86_feature_detected!("avx512f") {
                            return unsafe { #helper_name #helper_turbofish(#(#call_args),*) };
                        }
                    }
                },
                DispatchTarget::Avx2 => {
                    let runtime_cfgs: Vec<TokenStream> = features
                        .iter()
                        .map(|feature| quote!(std::is_x86_feature_detected!(#feature)))
                        .collect();
                    let runtime_cfg_expr = quote!(#(#runtime_cfgs)&&*);
                    quote! {
                        #[cfg(all(#arch_cfg, feature = "std"))]
                        {
                            if #scalar_frame_guard && #runtime_cfg_expr {
                                return unsafe { #helper_name #helper_turbofish(#(#call_args),*) };
                            }
                        }
                    }
                }
                DispatchTarget::Neon => quote! {
                    #[cfg(#arch_cfg)]
                    {
                        // NEON is mandatory on aarch64.
                        return unsafe { #helper_name #helper_turbofish(#(#call_args),*) };
                    }
                },
                DispatchTarget::Scalar => quote! {},
            };

            dispatch_arms.push(quote! {
                #compile_time_arm
                #runtime_arm
            });
        }
    }

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

    // Rebuild architecture bounds for each selected backend while preserving
    // all non-architecture predicates on the public dispatcher.
    let mut arch_bounds = Vec::new();
    if !arch_inline_bounds.is_empty() {
        arch_bounds.push(quote!(#arch_ident: #arch_inline_bounds));
    }
    if let Some(where_clause) = original_where_clause {
        for predicate in &where_clause.predicates {
            if let syn::WherePredicate::Type(predicate_type) = predicate {
                if let Type::Path(type_path) = &predicate_type.bounded_ty {
                    if type_path.path.is_ident(arch_ident) {
                        arch_bounds.push(quote!(#predicate_type));
                    }
                }
            }
        }
    }

    let mut specialized_bounds = Vec::new();
    for target in active_targets {
        let arch = target.arch_marker();
        for bound in &arch_bounds {
            let replaced = replace_ident(bound.clone(), arch_ident, &arch);
            specialized_bounds.push(replaced);
        }
    }

    let non_arch_predicates: Vec<syn::WherePredicate> = original_where_clause
        .map(|where_clause| {
            where_clause
                .predicates
                .iter()
                .filter(|predicate| {
                    if let syn::WherePredicate::Type(predicate_type) = predicate {
                        if let Type::Path(type_path) = &predicate_type.bounded_ty {
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

    let scalar_frame_bound = if avx2_adapter_name.is_some() {
        let scalar = scalar_type.expect("AVX2 adapters require a scalar type");
        quote!(
            #scalar: hermes_simd_intrinsics::x86_64::avx2_f16::Avx2FrameScalar,
        )
    } else {
        quote!()
    };
    let dispatcher_where = quote! {
        where
            #(#non_arch_predicates,)*
            #(#specialized_bounds,)*
            #scalar_frame_bound
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
