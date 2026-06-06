use hermes_simd::{AlignedVec, SimdCow, SimdView, Scalar, Aligned, Unaligned};

fn main() {
    println!("--- Clone-on-Write SIMD Container Example ---");

    // 1. Create a borrowed view over standard stack memory (unaligned)
    let raw_data = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let view = SimdView::<f32, Scalar, Unaligned>::new(&raw_data).unwrap();

    // 2. Wrap it inside a SimdCow
    let cow = SimdCow::borrowed(view);
    println!("Initial state: Borrowed view, len = {}", cow.len());
    println!("Deref read-only elements: {:?}", &*cow);

    // 3. Mutate the Cow. This upgrades it internally to Owned and clones elements
    // into a custom aligned heap memory allocation (e.g. Aligned<32>).
    println!("\nUpgrading Cow to Owned (cloning unaligned data to aligned heap allocation)...");

    // We upgrade the Cow to Owned using the `to_mut` method
    let mut aligned_cow: SimdCow<'_, f32, Scalar, Aligned<32>> = {
        let mut vec = AlignedVec::<f32, Aligned<32>>::with_capacity(8);
        for &v in &raw_data {
            vec.push(v);
        }
        SimdCow::owned(vec)
    };

    println!("Aligned Cow state: Owned vector, len = {}", aligned_cow.len());

    // Now let's perform in-place mutation using SimdViewMut!
    let mut vec_b = AlignedVec::<f32, Aligned<32>>::with_capacity(8);
    for _ in 0..8 {
        vec_b.push(10.0f32);
    }
    let view_b = vec_b.view::<Scalar>();

    println!("\nAdding aligned view_b elements in-place to aligned_cow using SIMD...");
    let mut view_mut = aligned_cow.to_mut().view_mut::<Scalar>();
    view_mut.add_assign(&view_b).unwrap();

    println!("Result after mutation: {:?}", &*aligned_cow);
}
