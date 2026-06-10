use hermes_simd::{dot, sum, Scalar, SimdView, Unaligned};

fn main() {
    let a = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
    let b = [1.0f32, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0];

    println!("--- Dynamic Runtime Dispatch ---");
    let dynamic_sum = sum::<f32>(&a);
    println!("Sum of a: {}", dynamic_sum);

    let dynamic_dot = dot::<f32>(&a, &b).unwrap();
    println!("Dot product (a . b): {}", dynamic_dot);

    println!("\n--- Static Compile-Time Dispatch (Scalar fallback) ---");
    let view_a = SimdView::<f32, Scalar, Unaligned>::new(&a).unwrap();
    let view_b = SimdView::<f32, Scalar, Unaligned>::new(&b).unwrap();

    let static_sum = view_a.sum();
    println!("Static view sum of a: {}", static_sum);

    let static_dot = view_a.dot(&view_b).unwrap();
    println!("Static view dot (a . b): {}", static_dot);
}
