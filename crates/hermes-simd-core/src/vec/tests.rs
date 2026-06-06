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

    let deserialized: AlignedVec<i32, Unaligned> = archived.deserialize(&mut ::rkyv::Infallible).unwrap();
    assert_eq!(deserialized.len(), 3);
    assert_eq!(deserialized[0], 10);
    assert_eq!(deserialized[1], 20);
    assert_eq!(deserialized[2], 30);
}
