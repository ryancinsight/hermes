mod floats;
mod ints;

pub use floats::{Bf16, Bf4, Bf8, F16, F32, F4, F64, F8};
pub use ints::{I16, I32, I8};

// Bytemuck implementations
unsafe impl bytemuck::Zeroable for F16 {}
unsafe impl bytemuck::Pod for F16 {}
unsafe impl bytemuck::Zeroable for F32 {}
unsafe impl bytemuck::Pod for F32 {}
unsafe impl bytemuck::Zeroable for F64 {}
unsafe impl bytemuck::Pod for F64 {}
unsafe impl bytemuck::Zeroable for Bf16 {}
unsafe impl bytemuck::Pod for Bf16 {}
unsafe impl bytemuck::Zeroable for Bf8 {}
unsafe impl bytemuck::Pod for Bf8 {}
unsafe impl bytemuck::Zeroable for Bf4 {}
unsafe impl bytemuck::Pod for Bf4 {}
unsafe impl bytemuck::Zeroable for F8 {}
unsafe impl bytemuck::Pod for F8 {}
unsafe impl bytemuck::Zeroable for F4 {}
unsafe impl bytemuck::Pod for F4 {}
unsafe impl bytemuck::Zeroable for I8 {}
unsafe impl bytemuck::Pod for I8 {}
unsafe impl bytemuck::Zeroable for I16 {}
unsafe impl bytemuck::Pod for I16 {}
unsafe impl bytemuck::Zeroable for I32 {}
unsafe impl bytemuck::Pod for I32 {}

const _: () = {
    assert!(core::mem::size_of::<F16>() == 2);
    assert!(core::mem::align_of::<F16>() == 2);
    assert!(core::mem::size_of::<F32>() == 4);
    assert!(core::mem::align_of::<F32>() == 4);
    assert!(core::mem::size_of::<F64>() == 8);
    assert!(core::mem::align_of::<F64>() == 8);
    assert!(core::mem::size_of::<Bf16>() == 2);
    assert!(core::mem::align_of::<Bf16>() == 2);
    assert!(core::mem::size_of::<Bf8>() == 1);
    assert!(core::mem::align_of::<Bf8>() == 1);
    assert!(core::mem::size_of::<Bf4>() == 1);
    assert!(core::mem::align_of::<Bf4>() == 1);
    assert!(core::mem::size_of::<F8>() == 1);
    assert!(core::mem::align_of::<F8>() == 1);
    assert!(core::mem::size_of::<F4>() == 1);
    assert!(core::mem::align_of::<F4>() == 1);
    assert!(core::mem::size_of::<I8>() == 1);
    assert!(core::mem::align_of::<I8>() == 1);
    assert!(core::mem::size_of::<I16>() == 2);
    assert!(core::mem::align_of::<I16>() == 2);
    assert!(core::mem::size_of::<I32>() == 4);
    assert!(core::mem::align_of::<I32>() == 4);
};
