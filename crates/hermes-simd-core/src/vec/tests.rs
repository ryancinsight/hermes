use super::*;
use crate::align::Unaligned;

#[test]
fn test_aligned_vec_zst() {
    let mut v = AlignedVec::<(), Unaligned>::new();
    assert_eq!(v.len(), 0);
    v.push(());
    v.push(());
    assert_eq!(v.len(), 2);
    assert_eq!(v[0], ());
    assert_eq!(v[1], ());

    let v2 = v.clone();
    assert_eq!(v2.len(), 2);
    assert_eq!(v2[0], ());
}

#[test]
fn test_aligned_vec_zst_drops() {
    use std::sync::atomic::{AtomicUsize, Ordering};
    static DROP_COUNT: AtomicUsize = AtomicUsize::new(0);

    #[derive(Clone)]
    struct ZstDropper;
    impl Drop for ZstDropper {
        fn drop(&mut self) {
            DROP_COUNT.fetch_add(1, Ordering::SeqCst);
        }
    }

    DROP_COUNT.store(0, Ordering::SeqCst);
    {
        let mut v = AlignedVec::<ZstDropper, Unaligned>::new();
        v.push(ZstDropper);
        v.push(ZstDropper);
        assert_eq!(DROP_COUNT.load(Ordering::SeqCst), 0);

        {
            let _v2 = v.clone();
            assert_eq!(DROP_COUNT.load(Ordering::SeqCst), 0);
        }
        assert_eq!(DROP_COUNT.load(Ordering::SeqCst), 2);
    }
    assert_eq!(DROP_COUNT.load(Ordering::SeqCst), 4);
}

#[test]
// rkyv 0.7's ArchivedVec::as_slice violates Stacked Borrows (upstream crate
// code, not hermes unsafe); Miri aborts inside the dependency. Value-semantic
// coverage still runs under the normal test harness.
#[cfg_attr(miri, ignore)]
fn test_aligned_vec_rkyv() {
    use ::rkyv::Deserialize;
    let mut v = AlignedVec::<i32, Unaligned>::new();
    v.push(10);
    v.push(20);
    v.push(30);

    let bytes = ::rkyv::to_bytes::<_, 256>(&v).unwrap();

    let archived = unsafe { ::rkyv::archived_root::<AlignedVec<i32, Unaligned>>(&bytes[..]) };
    assert_eq!(archived.elements.len(), 3);
    assert_eq!(archived.elements[0], 10);
    assert_eq!(archived.elements[1], 20);
    assert_eq!(archived.elements[2], 30);

    let deserialized: AlignedVec<i32, Unaligned> =
        archived.deserialize(&mut ::rkyv::Infallible).unwrap();
    assert_eq!(deserialized.len(), 3);
    assert_eq!(deserialized[0], 10);
    assert_eq!(deserialized[1], 20);
    assert_eq!(deserialized[2], 30);
}

#[test]
fn test_aligned_vec_drop_exception_safety() {
    struct PanicOnDrop;
    impl Drop for PanicOnDrop {
        fn drop(&mut self) {
            panic!("intentional panic on drop");
        }
    }

    let res = std::panic::catch_unwind(|| {
        let mut v = AlignedVec::<PanicOnDrop, Unaligned>::new();
        v.push(PanicOnDrop);
    });
    assert!(res.is_err());
}

#[test]
fn test_aligned_vec_alignment_casting() {
    use crate::align::Aligned;

    let mut v = AlignedVec::<i32, Unaligned>::with_capacity(16);
    v.push(1);
    v.push(2);

    let addr = v.as_ptr() as usize;
    if addr % 32 == 0 {
        let v_aligned = v.try_into_alignment::<Aligned<32>>();
        assert!(v_aligned.is_some());
        let v_aligned = v_aligned.unwrap();
        assert_eq!(v_aligned[0], 1);
        assert_eq!(v_aligned[1], 2);
    } else {
        let v_aligned = v.try_into_alignment::<Aligned<32>>();
        assert!(v_aligned.is_none());
    }

    let v_aligned = AlignedVec::<i32, Aligned<32>>::with_capacity(16);
    let v_unaligned = v_aligned.into_unaligned();
    assert_eq!(v_unaligned.len(), 0);
}

#[test]
fn test_aligned_vec_reserve_single_realloc() {
    use crate::align::Aligned;
    // From empty: one reserve to an explicit target allocates exactly once and
    // never shrinks below the request.
    let mut v = AlignedVec::<i32, Aligned<64>>::new();
    v.reserve(100);
    let cap_after = v.capacity();
    assert!(
        cap_after >= 100,
        "reserve must satisfy the request: {cap_after}"
    );
    let ptr_after = v.as_ptr() as usize;

    // Pushing up to the reserved capacity must not reallocate (pointer stable).
    for i in 0..100 {
        v.push(i);
    }
    assert_eq!(v.len(), 100);
    assert_eq!(
        v.capacity(),
        cap_after,
        "no realloc within reserved capacity"
    );
    assert_eq!(
        v.as_ptr() as usize,
        ptr_after,
        "pointer stable within capacity"
    );
    for i in 0..100 {
        assert_eq!(v[i as usize], i);
    }

    // reserve is a no-op when capacity already suffices.
    v.reserve(0);
    assert_eq!(v.capacity(), cap_after);
    v.reserve(50); // 100 + 50 = 150 > cap → grows (still one call)
    assert!(v.capacity() >= 150);
}

#[test]
fn test_aligned_vec_extend_from_slice() {
    use crate::align::Aligned;
    let mut v = AlignedVec::<i32, Aligned<64>>::new();
    v.extend_from_slice(&[1, 2, 3]);
    v.extend_from_slice(&[]); // empty is a no-op
    v.extend_from_slice(&[4, 5]);
    assert_eq!(v.as_slice(), &[1, 2, 3, 4, 5]);

    // Extending a pre-sized vec copies in place without reallocating.
    let mut w = AlignedVec::<f32, Aligned<64>>::with_capacity(8);
    let ptr = w.as_ptr() as usize;
    w.extend_from_slice(&[1.5, 2.5, 3.5, 4.5]);
    assert_eq!(w.as_slice(), &[1.5, 2.5, 3.5, 4.5]);
    assert_eq!(w.as_ptr() as usize, ptr, "no realloc within capacity");
}

#[test]
fn test_aligned_vec_extend_from_slice_zst() {
    let mut v = AlignedVec::<(), Unaligned>::new();
    v.extend_from_slice(&[(), (), ()]);
    assert_eq!(v.len(), 3);
    assert_eq!(v[2], ());
}
