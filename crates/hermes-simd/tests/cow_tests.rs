use hermes_simd::*;

#[test]
fn test_simd_cow_operators_reuse_allocation() {
    let a_data = vec![1.0f32, 2.0, 3.0, 4.0];
    let b_data = vec![10.0f32, 20.0, 30.0, 40.0];

    // 1. Commutative Addition: Owned + Borrowed
    let a = SimdCow::<f32, Scalar, Unaligned>::from_slice(&a_data);
    let b = SimdCow::<f32, Scalar, Unaligned>::borrow_slice(&b_data).unwrap();

    let a_ptr = a.view().as_slice().as_ptr();

    let res = a + b;
    assert_eq!(&*res, &[11.0, 22.0, 33.0, 44.0]);

    match &res {
        SimdCow::Owned(vec) => {
            assert_eq!(vec.as_slice().as_ptr(), a_ptr);
        }
        _ => panic!("Expected Owned variant"),
    }

    // 2. Commutative Addition: Borrowed + Owned
    let a = SimdCow::<f32, Scalar, Unaligned>::borrow_slice(&a_data).unwrap();
    let b = SimdCow::<f32, Scalar, Unaligned>::from_slice(&b_data);

    let b_ptr = b.view().as_slice().as_ptr();

    let res = a + b;
    assert_eq!(&*res, &[11.0, 22.0, 33.0, 44.0]);

    match &res {
        SimdCow::Owned(vec) => {
            assert_eq!(vec.as_slice().as_ptr(), b_ptr);
        }
        _ => panic!("Expected Owned variant"),
    }

    // 3. Non-commutative Subtraction: Owned - Borrowed
    let a = SimdCow::<f32, Scalar, Unaligned>::from_slice(&a_data);
    let b = SimdCow::<f32, Scalar, Unaligned>::borrow_slice(&b_data).unwrap();

    let a_ptr = a.view().as_slice().as_ptr();

    let res = a - b;
    assert_eq!(&*res, &[-9.0, -18.0, -27.0, -36.0]);

    match &res {
        SimdCow::Owned(vec) => {
            assert_eq!(vec.as_slice().as_ptr(), a_ptr);
        }
        _ => panic!("Expected Owned variant"),
    }

    // 4. Non-commutative Subtraction: Borrowed - Owned
    let a = SimdCow::<f32, Scalar, Unaligned>::borrow_slice(&a_data).unwrap();
    let b = SimdCow::<f32, Scalar, Unaligned>::from_slice(&b_data);

    let b_ptr = b.view().as_slice().as_ptr();

    let res = a - b;
    assert_eq!(&*res, &[-9.0, -18.0, -27.0, -36.0]);

    match &res {
        SimdCow::Owned(vec) => {
            assert_eq!(vec.as_slice().as_ptr(), b_ptr);
        }
        _ => panic!("Expected Owned variant"),
    }

    // 5. Assign operators
    let mut a = SimdCow::<f32, Scalar, Unaligned>::from_slice(&a_data);
    let b = SimdCow::<f32, Scalar, Unaligned>::borrow_slice(&b_data).unwrap();

    let a_ptr = a.view().as_slice().as_ptr();

    a += b;
    assert_eq!(&*a, &[11.0, 22.0, 33.0, 44.0]);
    match &a {
        SimdCow::Owned(vec) => {
            assert_eq!(vec.as_slice().as_ptr(), a_ptr);
        }
        _ => panic!("Expected Owned variant"),
    }
}

#[test]
fn test_simd_cow_to_mut() {
    let data = vec![1.0f32, 2.0, 3.0, 4.0];
    
    // Borrowed
    let mut cow = SimdCow::<f32, Scalar, Unaligned>::borrow_slice(&data).unwrap();
    assert!(matches!(cow, SimdCow::Borrowed(_)));
    
    // Promoting to mut
    {
        let owned_vec = cow.to_mut();
        assert_eq!(owned_vec.len(), 4);
        owned_vec[0] = 99.0;
    }
    
    assert!(matches!(cow, SimdCow::Owned(_)));
    assert_eq!(&*cow, &[99.0, 2.0, 3.0, 4.0]);
    
    // Subsequent calls to to_mut should be free
    let ptr_before = cow.view().as_slice().as_ptr();
    {
        let owned_vec = cow.to_mut();
        assert_eq!(owned_vec.as_slice().as_ptr(), ptr_before);
    }
}

#[test]
fn test_simd_cow_state_accessors_preserve_zero_copy_reads() {
    let data = [1.0f32, 2.0, 3.0, 4.0];
    let mut cow = SimdCow::<f32, Scalar, Unaligned>::borrow_slice(&data).unwrap();

    assert!(cow.is_borrowed());
    assert!(!cow.is_owned());
    assert_eq!(cow.view().as_slice().as_ptr(), data.as_ptr());

    cow.to_mut()[2] = 30.0;

    assert!(cow.is_owned());
    assert!(!cow.is_borrowed());
    assert_ne!(cow.view().as_slice().as_ptr(), data.as_ptr());
    assert_eq!(data, [1.0, 2.0, 3.0, 4.0]);
    assert_eq!(&*cow, &[1.0, 2.0, 30.0, 4.0]);
}

#[test]
fn test_packed4_cow_state_accessors_preserve_packed_borrow() {
    let packed = [F4::pack_pair(F4(1), F4(2)), F4::pack_pair(F4(3), F4(4))];
    let mut cow = Packed4Cow::<F4>::from_packed_slice(&packed, 4).unwrap();

    assert!(cow.is_borrowed());
    assert!(!cow.is_owned());
    assert_eq!(cow.as_view().as_packed_slice().as_ptr(), packed.as_ptr());
    assert_eq!(cow.get(2), Some(F4(3)));

    cow.set(1, F4(7));

    assert!(cow.is_owned());
    assert!(!cow.is_borrowed());
    assert_ne!(cow.as_view().as_packed_slice().as_ptr(), packed.as_ptr());
    assert_eq!(packed, [F4::pack_pair(F4(1), F4(2)), F4::pack_pair(F4(3), F4(4))]);
    assert_eq!(cow.get(0), Some(F4(1)));
    assert_eq!(cow.get(1), Some(F4(7)));
    assert_eq!(cow.get(2), Some(F4(3)));
    assert_eq!(cow.get(3), Some(F4(4)));
}

#[test]
fn test_simd_cow_new_operators() {
    let a_data = vec![8.0f32, 16.0, 32.0, 64.0];
    let b_data = vec![2.0f32, 4.0, 8.0, 16.0];

    // Division: Non-commutative, Owned / Borrowed
    let a = SimdCow::<f32, Scalar, Unaligned>::from_slice(&a_data);
    let b = SimdCow::<f32, Scalar, Unaligned>::borrow_slice(&b_data).unwrap();
    let a_ptr = a.view().as_slice().as_ptr();
    let res_div = a / b;
    assert_eq!(&*res_div, &[4.0, 4.0, 4.0, 4.0]);
    match &res_div {
        SimdCow::Owned(vec) => assert_eq!(vec.as_slice().as_ptr(), a_ptr),
        _ => panic!("Expected Owned variant"),
    }

    // Division: Non-commutative, Borrowed / Owned
    let a = SimdCow::<f32, Scalar, Unaligned>::borrow_slice(&a_data).unwrap();
    let b = SimdCow::<f32, Scalar, Unaligned>::from_slice(&b_data);
    let b_ptr = b.view().as_slice().as_ptr();
    let res_div2 = a / b;
    assert_eq!(&*res_div2, &[4.0, 4.0, 4.0, 4.0]);
    match &res_div2 {
        SimdCow::Owned(vec) => assert_eq!(vec.as_slice().as_ptr(), b_ptr),
        _ => panic!("Expected Owned variant"),
    }

    // DivAssign
    let mut a = SimdCow::<f32, Scalar, Unaligned>::from_slice(&a_data);
    let b = SimdCow::<f32, Scalar, Unaligned>::borrow_slice(&b_data).unwrap();
    let a_ptr = a.view().as_slice().as_ptr();
    a /= b;
    assert_eq!(&*a, &[4.0, 4.0, 4.0, 4.0]);
    match &a {
        SimdCow::Owned(vec) => assert_eq!(vec.as_slice().as_ptr(), a_ptr),
        _ => panic!("Expected Owned variant"),
    }

    // BitAnd: Commutative, Owned & Borrowed
    // f32 from_bits(0x0F0F0F0F) & from_bits(0x00FF00FF) = from_bits(0x000F000F)
    let a_val = f32::from_bits(0x0F0F0F0F);
    let b_val = f32::from_bits(0x00FF00FF);
    let expected_val = f32::from_bits(0x0F0F0F0F & 0x00FF00FF);
    let a_bitwise = vec![a_val; 4];
    let b_bitwise = vec![b_val; 4];

    let a = SimdCow::<f32, Scalar, Unaligned>::from_slice(&a_bitwise);
    let b = SimdCow::<f32, Scalar, Unaligned>::borrow_slice(&b_bitwise).unwrap();
    let a_ptr = a.view().as_slice().as_ptr();
    let res_and = a & b;
    assert_eq!(&*res_and, &[expected_val; 4]);
    match &res_and {
        SimdCow::Owned(vec) => assert_eq!(vec.as_slice().as_ptr(), a_ptr),
        _ => panic!("Expected Owned variant"),
    }

    // BitOr: Commutative, Borrowed | Owned
    let expected_val_or = f32::from_bits(0x0F0F0F0F | 0x00FF00FF);
    let a = SimdCow::<f32, Scalar, Unaligned>::borrow_slice(&a_bitwise).unwrap();
    let b = SimdCow::<f32, Scalar, Unaligned>::from_slice(&b_bitwise);
    let b_ptr = b.view().as_slice().as_ptr();
    let res_or = a | b;
    assert_eq!(&*res_or, &[expected_val_or; 4]);
    match &res_or {
        SimdCow::Owned(vec) => assert_eq!(vec.as_slice().as_ptr(), b_ptr),
        _ => panic!("Expected Owned variant"),
    }

    // BitXorAssign
    let expected_val_xor = f32::from_bits(0x0F0F0F0F ^ 0x00FF00FF);
    let mut a = SimdCow::<f32, Scalar, Unaligned>::from_slice(&a_bitwise);
    let b = SimdCow::<f32, Scalar, Unaligned>::borrow_slice(&b_bitwise).unwrap();
    let a_ptr = a.view().as_slice().as_ptr();
    a ^= b;
    assert_eq!(&*a, &[expected_val_xor; 4]);
    match &a {
        SimdCow::Owned(vec) => assert_eq!(vec.as_slice().as_ptr(), a_ptr),
        _ => panic!("Expected Owned variant"),
    }
}

#[test]
fn test_cow_enhancements() {
    use hermes_simd_core::ops;

    // 1. AlignedVec PartialEq and Eq
    let mut v1 = AlignedVec::<i32, Unaligned>::new();
    v1.push(1);
    v1.push(2);
    v1.push(3);
    
    let mut v2 = AlignedVec::<i32, Unaligned>::new();
    v2.push(1);
    v2.push(2);
    v2.push(3);
    
    let mut v3 = AlignedVec::<i32, Unaligned>::new();
    v3.push(1);
    v3.push(2);
    v3.push(4);
    
    assert_eq!(v1, v2);
    assert_ne!(v1, v3);
    assert_eq!(&v1[..], &v2[..]);
    assert_eq!(&v1[..], &v2[..]);

    // 2. SimdChunksMut Iteration
    let mut data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let view = SimdView::<f32, Scalar, Unaligned, Unmasked, &mut [f32]>::new_mut(&mut data).unwrap();
    {
        let mut chunks = view.simd_chunks_mut();
        assert_eq!(chunks.chunks_remaining(), 1);
        
        let mut chunk = chunks.next().unwrap();
        assert_eq!(chunk.len(), 4);
        assert_eq!(chunk[0], 1.0);
        assert_eq!(chunk[1], 2.0);
        assert_eq!(chunk[2], 3.0);
        assert_eq!(chunk[3], 4.0);
        
        // modify the chunk
        chunk[0] = 10.0;
        chunk[1] = 20.0;
        
        assert!(chunks.next().is_none());
        
        let remainder = chunks.into_remainder();
        assert_eq!(remainder.len(), 2);
        assert_eq!(remainder[0], 5.0);
        assert_eq!(remainder[1], 6.0);
        remainder[0] = 50.0;
    }
    assert_eq!(data, vec![10.0, 20.0, 3.0, 4.0, 50.0, 6.0]);
    
    // 3. SimdCow Clone, Default, Debug, PartialEq, Eq
    let cow_borrowed = SimdCow::<f32, Scalar, Unaligned>::borrow_slice(&[1.0, 2.0, 3.0, 4.0]).unwrap();
    let cow_owned = SimdCow::<f32, Scalar, Unaligned>::from_slice(&[1.0, 2.0, 3.0, 4.0]);
    let cow_different = SimdCow::<f32, Scalar, Unaligned>::from_slice(&[1.0, 2.0, 3.0, 5.0]);
    let cow_default = SimdCow::<f32, Scalar, Unaligned>::default();

    assert_eq!(cow_borrowed, cow_owned);
    assert_ne!(cow_borrowed, cow_different);
    assert_eq!(cow_default.len(), 0);

    let cloned_borrowed = cow_borrowed.clone();
    assert!(matches!(cloned_borrowed, SimdCow::Borrowed(_)));
    assert_eq!(cloned_borrowed, cow_owned);

    let cloned_owned = cow_owned.clone();
    assert!(matches!(cloned_owned, SimdCow::Owned(_)));
    assert_eq!(cloned_owned, cow_owned);

    let debug_str = format!("{:?}", cow_borrowed);
    assert!(debug_str.contains("Borrowed"));
    
    let debug_str_owned = format!("{:?}", cow_owned);
    assert!(debug_str_owned.contains("Owned"));

    // 4. Verification of the refactored transform_in_place and zip_cow
    let cow_a = SimdCow::<f32, Scalar, Unaligned>::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let cow_b = SimdCow::<f32, Scalar, Unaligned>::borrow_slice(&[10.0, 20.0, 30.0, 40.0, 50.0, 60.0]).unwrap();
    
    let cow_res = cow_a.zip_cow(&cow_b, ops::Add).unwrap();
    assert_eq!(&*cow_res, &[11.0, 22.0, 33.0, 44.0, 55.0, 66.0]);

    let mut cow_mut = cow_a.clone();
    cow_mut.transform_in_place(&cow_b, ops::Add).unwrap();
    assert_eq!(&*cow_mut, &[11.0, 22.0, 33.0, 44.0, 55.0, 66.0]);
}

#[test]
fn test_simd_cow_rkyv_serialization() {
    use rkyv::Deserialize;

    let original_data = [1.5f32, 2.5, 3.5, 4.5];
    let cow = SimdCow::<f32, Scalar, Unaligned>::from_slice(&original_data);

    // Serialize
    let bytes = rkyv::to_bytes::<_, 256>(&cow).unwrap();

    // Zero-copy access (archived root)
    let archived = unsafe { rkyv::archived_root::<SimdCow<f32, Scalar, Unaligned>>(&bytes[..]) };
    assert_eq!(archived.len(), 4);
    assert_eq!(archived.as_slice(), &original_data);
    assert_eq!(&archived[..], &original_data);

    // Convert archived to borrowed SimdCow
    let borrowed_cow: SimdCow<'_, f32, Scalar, Unaligned> = unsafe { archived.as_borrowed().unwrap() };
    assert_eq!(&*borrowed_cow, &original_data);
    assert!(matches!(borrowed_cow, SimdCow::Borrowed(_)));

    // Deserialize to owned
    let deserialized: SimdCow<'static, f32, Scalar, Unaligned> = archived.deserialize(&mut rkyv::Infallible).unwrap();
    assert_eq!(&*deserialized, &original_data);
    assert!(matches!(deserialized, SimdCow::Owned(_)));

    // Test that serializing a Borrowed variant works and yields identical bytes
    let borrowed_original = SimdCow::<f32, Scalar, Unaligned>::borrow_slice(&original_data).unwrap();
    let bytes_borrowed = rkyv::to_bytes::<_, 256>(&borrowed_original).unwrap();
    assert_eq!(&bytes[..], &bytes_borrowed[..]);
}
