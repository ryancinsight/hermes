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

#[test]
fn scalar_bound_adds_exact_f16c_and_plain_avx2_frames() {
    let kernel = quote! {
        fn dot_kernel<T, A>(a: &[T], b: &[T]) -> T
        where
            T: hermes_simd_core::scalar::Scalar,
            A: hermes_simd_core::arch::SimdArch
                + hermes_simd_core::kernel::SimdKernel<T>,
        {
            a[0]
        }
    };

    let expanded = super::expand(quote!(avx2, scalar), kernel)
        .expect("valid scalar-aware dispatch input")
        .to_string();
    assert!(expanded.contains("dot_avx2_fma_f16c"));
    assert!(expanded.contains("dot_avx2_fma"));
    assert!(expanded.contains("REQUIRES_F16C"));
    assert!(expanded.contains("Avx2FrameKernel"));
    assert!(expanded.contains("Avx2FrameScalar"));
    assert!(expanded.contains("call_avx2_frame"));
    assert!(!expanded.contains("Avx2F16c"));
    assert_eq!(
        expanded
            .matches("target_feature (enable = \"f16c\")")
            .count(),
        1
    );
}

#[test]
fn inline_scalar_bound_adds_exact_f16c_frame() {
    let kernel = quote! {
        fn inline_kernel<T: hermes_simd_core::scalar::Scalar,
            A: hermes_simd_core::kernel::SimdKernel<T>>(value: T) -> T
        {
            value
        }
    };

    let expanded = super::expand(quote!(avx2, scalar), kernel)
        .expect("valid inline scalar-aware dispatch input")
        .to_string();
    assert!(expanded.contains("inline_avx2_fma_f16c"));
    assert!(expanded.contains("Avx2FrameScalar"));
    assert!(expanded.contains("Avx2 : hermes_simd_core :: kernel :: SimdKernel < T >"));
}

#[test]
fn f16c_bound_is_punctuated_after_where_clause() {
    let kernel = quote! {
        fn punctuated_kernel<T, A>(value: T) -> T
        where
            T: hermes_simd_core::scalar::Scalar,
            A: hermes_simd_core::kernel::SimdKernel<T>
        {
            value
        }
    };

    let expanded = super::expand(quote!(avx2, scalar), kernel)
        .expect("where clause without a trailing comma must expand");
    syn::parse2::<syn::File>(expanded).expect("generated dispatcher must remain valid Rust syntax");
}

#[test]
fn borrowed_output_uses_the_normalized_input_lifetime() {
    let kernel = quote! {
        fn borrowed_kernel<T, A>(value: &T) -> &T
        where
            T: hermes_simd_core::scalar::Scalar,
            A: hermes_simd_core::kernel::SimdKernel<T>,
        {
            value
        }
    };

    let expanded = super::expand(quote!(avx2, scalar), kernel)
        .expect("single-input borrowed output must expand")
        .to_string();
    assert!(expanded.contains("type Output = & '__hermes_dispatch_0 T"));
}

#[test]
fn explicit_output_lifetime_survives_multiple_input_lifetimes() {
    let kernel = quote! {
        fn first_kernel<'left, 'right, T, A>(left: &'left T, right: &'right T) -> &'left T
        where
            T: hermes_simd_core::scalar::Scalar,
            A: hermes_simd_core::kernel::SimdKernel<T>,
        {
            let _ = right;
            left
        }
    };

    let expanded = super::expand(quote!(avx2, scalar), kernel)
        .expect("explicit output lifetime must expand")
        .to_string();
    assert!(expanded.contains("type Output = & 'left T"));
}

#[test]
fn architecture_only_kernel_retains_one_avx2_frame() {
    let kernel = quote! {
        fn probe_kernel<A>(value: usize) -> usize {
            value
        }
    };

    let expanded = super::expand(quote!(avx2, scalar), kernel)
        .expect("valid architecture-only dispatch input")
        .to_string();
    assert!(expanded.contains("probe_avx2_fma"));
    assert!(!expanded.contains("probe_avx2_fma_f16c"));
    assert!(!expanded.contains("REQUIRES_F16C"));
}
