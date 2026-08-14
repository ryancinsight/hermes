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
