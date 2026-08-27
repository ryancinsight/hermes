use quote::quote;

#[test]
fn unreachable_code_expectation_is_neon_only() {
    let kernel = quote! {
        pub fn sum_kernel<A>(data: &[f32]) -> f32 {
            data[0]
        }
    };

    let x86 = super::expand(quote!(avx2, scalar), kernel.clone())
        .expect("valid x86 dispatch input")
        .to_string();
    assert!(!x86.contains("unreachable_code"));

    let aarch64 = super::expand(quote!(neon, scalar), kernel)
        .expect("valid AArch64 dispatch input")
        .to_string();
    assert!(aarch64.contains("unreachable_code"));
}

#[test]
fn inner_fn_is_forced_into_the_target_feature_frame() {
    // The retained inner fn must carry `#[inline(always)]`: a large kernel
    // body under the plain inline heuristic is outlined from the generated
    // `#[target_feature]` helper and codegens at baseline (zero FMA).
    let kernel = quote! {
        pub fn sum_kernel<A>(data: &[f32]) -> f32 {
            data[0]
        }
    };
    let expanded = super::expand(quote!(avx2, scalar), kernel)
        .expect("valid dispatch input")
        .to_string();
    let inner = expanded
        .split("fn sum_kernel")
        .next()
        .expect("split yields a leading segment");
    assert!(
        inner.contains("# [inline (always)]"),
        "inner kernel fn must be alwaysinline, got prefix: {inner}"
    );
}

#[test]
fn explicit_inline_attribute_is_respected() {
    // An author-written inline attribute wins; the macro must not stack a
    // duplicate that would fail to compile.
    let kernel = quote! {
        #[inline(never)]
        pub fn sum_kernel<A>(data: &[f32]) -> f32 {
            data[0]
        }
    };
    let expanded = super::expand(quote!(avx2, scalar), kernel)
        .expect("valid dispatch input")
        .to_string();
    assert_eq!(expanded.matches("# [inline (never)]").count(), 1);
    let inner = expanded
        .split("fn sum_kernel")
        .next()
        .expect("split yields a leading segment");
    assert!(!inner.contains("# [inline (always)]"));
}
