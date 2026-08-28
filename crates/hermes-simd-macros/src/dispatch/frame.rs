use std::collections::HashSet;

use proc_macro2::{Span, TokenStream};
use quote::quote;
use syn::{
    punctuated::Punctuated,
    spanned::Spanned,
    visit_mut::{self, VisitMut},
    FnArg, GenericParam, Ident, Lifetime, ReturnType, Token, Type, TypeParamBound,
};

/// Finds the scalar type bound to the architecture parameter.
///
/// Hermes kernels express that relation as an inline or `where`-clause
/// architecture facet such as `A: SimdKernel<T>` or `A: SimdArith<T>`. The
/// generated dispatcher uses the recovered `T` only to select scalar-specific
/// target-feature frames; macro users without such a bound retain
/// architecture-only dispatch.
pub(super) fn kernel_scalar_type(
    generics: &syn::Generics,
    arch_ident: &Ident,
) -> Option<syn::Type> {
    fn from_bounds(bounds: &Punctuated<TypeParamBound, Token![+]>) -> Option<Type> {
        bounds.iter().find_map(|bound| {
            let syn::TypeParamBound::Trait(trait_bound) = bound else {
                return None;
            };
            let segment = trait_bound.path.segments.last()?;
            if !matches!(
                segment.ident.to_string().as_str(),
                "SimdKernel"
                    | "SimdStorage"
                    | "SimdLoadStore"
                    | "SimdArith"
                    | "SimdBitwise"
                    | "SimdCompare"
                    | "SimdReduce"
                    | "SimdMask"
                    | "SimdGather"
                    | "SimdPermute"
            ) {
                return None;
            }
            let syn::PathArguments::AngleBracketed(arguments) = &segment.arguments else {
                return None;
            };
            arguments.args.iter().find_map(|argument| {
                let syn::GenericArgument::Type(scalar) = argument else {
                    return None;
                };
                Some(scalar.clone())
            })
        })
    }

    let inline = generics.params.iter().find_map(|parameter| {
        let GenericParam::Type(parameter) = parameter else {
            return None;
        };
        (parameter.ident == *arch_ident)
            .then(|| from_bounds(&parameter.bounds))
            .flatten()
    });
    inline.or_else(|| {
        generics
            .where_clause
            .as_ref()?
            .predicates
            .iter()
            .find_map(|predicate| {
                let syn::WherePredicate::Type(predicate_type) = predicate else {
                    return None;
                };
                let Type::Path(bounded_path) = &predicate_type.bounded_ty else {
                    return None;
                };
                bounded_path
                    .path
                    .is_ident(arch_ident)
                    .then(|| from_bounds(&predicate_type.bounds))
                    .flatten()
            })
    })
}

struct ElidedLifetimeNormalizer {
    synthetic_lifetimes: Vec<Lifetime>,
    input_lifetimes: Vec<Lifetime>,
    occupied: HashSet<String>,
    next: usize,
}

impl ElidedLifetimeNormalizer {
    fn new(parameters: &[GenericParam]) -> Self {
        let occupied = parameters
            .iter()
            .filter_map(|parameter| {
                let GenericParam::Lifetime(parameter) = parameter else {
                    return None;
                };
                Some(parameter.lifetime.ident.to_string())
            })
            .collect();
        Self {
            synthetic_lifetimes: Vec::new(),
            input_lifetimes: Vec::new(),
            occupied,
            next: 0,
        }
    }

    fn replacement(&mut self) -> Lifetime {
        loop {
            let name = format!("__hermes_dispatch_{}", self.next);
            self.next += 1;
            if self.occupied.insert(name.clone()) {
                let lifetime = Lifetime::new(&format!("'{name}"), Span::call_site());
                self.synthetic_lifetimes.push(lifetime.clone());
                return lifetime;
            }
        }
    }
}

impl VisitMut for ElidedLifetimeNormalizer {
    fn visit_type_reference_mut(&mut self, reference: &mut syn::TypeReference) {
        if reference.lifetime.is_none() {
            reference.lifetime = Some(self.replacement());
        }
        visit_mut::visit_type_reference_mut(self, reference);
    }

    fn visit_lifetime_mut(&mut self, lifetime: &mut Lifetime) {
        if lifetime.ident == "_" {
            *lifetime = self.replacement();
        }
        self.input_lifetimes.push(lifetime.clone());
    }
}

struct OutputLifetimeNormalizer<'input> {
    input: Option<&'input Lifetime>,
    unresolved: bool,
}

impl VisitMut for OutputLifetimeNormalizer<'_> {
    fn visit_type_reference_mut(&mut self, reference: &mut syn::TypeReference) {
        if reference.lifetime.is_none() {
            if let Some(input) = self.input {
                reference.lifetime = Some(input.clone());
            } else {
                self.unresolved = true;
            }
        }
        visit_mut::visit_type_reference_mut(self, reference);
    }

    fn visit_lifetime_mut(&mut self, lifetime: &mut Lifetime) {
        if lifetime.ident == "_" {
            if let Some(input) = self.input {
                *lifetime = input.clone();
            } else {
                self.unresolved = true;
            }
        }
    }
}

fn without_generic_default(param: &GenericParam) -> GenericParam {
    let mut param = param.clone();
    match &mut param {
        GenericParam::Type(param) => param.default = None,
        GenericParam::Const(param) => param.default = None,
        GenericParam::Lifetime(_) => {}
    }
    param
}

pub(super) fn upper_camel_case(value: &str) -> String {
    value
        .split('_')
        .filter(|component| !component.is_empty())
        .map(|component| {
            let mut characters = component.chars();
            characters.next().map_or_else(String::new, |first| {
                first.to_uppercase().chain(characters).collect()
            })
        })
        .collect()
}

#[expect(
    clippy::too_many_arguments,
    reason = "The callback adapter mirrors the dispatcher generator inputs"
)]
#[expect(
    clippy::too_many_lines,
    reason = "One generator keeps the callback's lifetimes, generics, bounds, and invocation coherent"
)]
pub(super) fn generate_avx2_frame_adapter(
    adapter_name: &Ident,
    inner_name: &Ident,
    inner_args: &Punctuated<FnArg, Token![,]>,
    inner_ret: &ReturnType,
    other_params: &[GenericParam],
    other_param_tokens: &[TokenStream],
    call_args: &[TokenStream],
    arch_ident: &Ident,
    scalar_type: &Type,
    original_where_clause: Option<&syn::WhereClause>,
) -> syn::Result<TokenStream> {
    let mut normalizer = ElidedLifetimeNormalizer::new(other_params);
    let argument_types: Vec<Type> = inner_args
        .iter()
        .filter_map(|argument| {
            let FnArg::Typed(argument) = argument else {
                return None;
            };
            let mut ty = (*argument.ty).clone();
            normalizer.visit_type_mut(&mut ty);
            Some(ty)
        })
        .collect();

    let synthetic_lifetimes: Vec<GenericParam> = normalizer
        .synthetic_lifetimes
        .iter()
        .map(|lifetime| syn::parse_quote!(#lifetime))
        .collect();
    let parameters: Vec<GenericParam> = synthetic_lifetimes
        .iter()
        .cloned()
        .chain(other_params.iter().map(without_generic_default))
        .collect();
    let generics = if parameters.is_empty() {
        quote!()
    } else {
        quote!(<#(#parameters),*>)
    };
    let adapter_arguments: Vec<TokenStream> = normalizer
        .synthetic_lifetimes
        .iter()
        .map(|lifetime| quote!(#lifetime))
        .chain(other_params.iter().map(|parameter| match parameter {
            GenericParam::Lifetime(parameter) => {
                let lifetime = &parameter.lifetime;
                quote!(#lifetime)
            }
            GenericParam::Type(parameter) => {
                let ident = &parameter.ident;
                quote!(#ident)
            }
            GenericParam::Const(parameter) => {
                let ident = &parameter.ident;
                quote!(#ident)
            }
        }))
        .collect();
    let adapter_type = if adapter_arguments.is_empty() {
        quote!(#adapter_name)
    } else {
        quote!(#adapter_name<#(#adapter_arguments),*>)
    };

    let mut phantom_types: Vec<TokenStream> = normalizer
        .synthetic_lifetimes
        .iter()
        .map(|lifetime| quote!(&#lifetime ()))
        .collect();
    phantom_types.extend(other_params.iter().filter_map(|parameter| match parameter {
        GenericParam::Lifetime(parameter) => {
            let lifetime = &parameter.lifetime;
            Some(quote!(&#lifetime ()))
        }
        GenericParam::Type(parameter) => {
            let ident = &parameter.ident;
            Some(quote!(#ident))
        }
        GenericParam::Const(_) => None,
    }));
    let phantom = if phantom_types.is_empty() {
        quote!(())
    } else {
        quote!((#(#phantom_types,)*))
    };

    let non_arch_predicates: Vec<syn::WherePredicate> = original_where_clause
        .map(|where_clause| {
            where_clause
                .predicates
                .iter()
                .filter(|predicate| {
                    let syn::WherePredicate::Type(predicate) = predicate else {
                        return true;
                    };
                    let Type::Path(path) = &predicate.bounded_ty else {
                        return true;
                    };
                    !path.path.is_ident(arch_ident)
                })
                .cloned()
                .collect()
        })
        .unwrap_or_default();
    let where_clause = if non_arch_predicates.is_empty() {
        quote!()
    } else {
        quote!(where #(#non_arch_predicates,)*)
    };
    let arguments = quote!((#(#argument_types,)*));
    let output = match inner_ret {
        ReturnType::Default => Type::Tuple(syn::TypeTuple {
            paren_token: syn::token::Paren::default(),
            elems: Punctuated::new(),
        }),
        ReturnType::Type(_, output) => {
            let mut output = (**output).clone();
            let mut input_lifetimes = normalizer.input_lifetimes.clone();
            input_lifetimes.sort_by_key(ToString::to_string);
            input_lifetimes.dedup_by(|left, right| left.ident == right.ident);
            let mut output_normalizer = OutputLifetimeNormalizer {
                input: (input_lifetimes.len() == 1).then(|| &input_lifetimes[0]),
                unresolved: false,
            };
            output_normalizer.visit_type_mut(&mut output);
            if output_normalizer.unresolved {
                return Err(syn::Error::new(
                    inner_ret.span(),
                    "runtime-dispatched borrowed output requires one input lifetime or an explicit output lifetime",
                ));
            }
            output
        }
    };
    let inner_turbofish = if other_param_tokens.is_empty() {
        quote!(::<__HermesFrameArch>)
    } else {
        quote!(::<#(#other_param_tokens,)* __HermesFrameArch>)
    };

    Ok(quote! {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        struct #adapter_name #generics(core::marker::PhantomData<#phantom>);

        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        impl #generics
            hermes_simd_intrinsics::x86_64::avx2_f16::Avx2FrameKernel<#scalar_type, #arguments>
            for #adapter_type
            #where_clause
        {
            type Output = #output;

            #[inline(always)]
            fn call<__HermesFrameArch>(self, arguments: #arguments) -> Self::Output
            where
                __HermesFrameArch: hermes_simd_core::arch::SimdArch
                    + hermes_simd_core::kernel::SimdKernel<#scalar_type>,
            {
                let (#(#call_args,)*) = arguments;
                #inner_name #inner_turbofish(#(#call_args),*)
            }
        }
    })
}
