use hermes_simd::*;

#[test]
fn test_gather_f32() {
    let data = vec![
        10.0f32, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0,
    ];
    let view = SimdView::<f32, PreferredArch, Unaligned>::new(&data).unwrap();
    let indices = vec![0i32, 2, 4, 1, 3, 5, 9, 8, 7, 6];
    let mut out = vec![0.0f32; 10];
    view.gather(&indices, &mut out).unwrap();

    let expected = vec![10.0, 30.0, 50.0, 20.0, 40.0, 60.0, 100.0, 90.0, 80.0, 70.0];
    assert_eq!(out, expected);
}

#[test]
fn test_gather_bounds_error() {
    let data = vec![1.0f32, 2.0, 3.0];
    let view = SimdView::<f32, PreferredArch, Unaligned>::new(&data).unwrap();

    // Index too large
    let mut out = vec![0.0f32; 1];
    let res = view.gather(&[3], &mut out);
    assert_eq!(res, Err(SimdError::IndexOutOfBounds));

    // Index negative
    let res = view.gather(&[-1], &mut out);
    assert_eq!(res, Err(SimdError::IndexOutOfBounds));
}

#[test]
fn test_gather_insufficient_output() {
    let data = vec![1.0f32, 2.0, 3.0];
    let view = SimdView::<f32, PreferredArch, Unaligned>::new(&data).unwrap();
    let mut out = vec![0.0f32; 1];
    let res = view.gather(&[0, 1], &mut out);
    assert_eq!(res, Err(SimdError::InsufficientOutputLength));
}

#[test]
fn test_gather_cow() {
    let data = vec![10.0f32, 20.0, 30.0, 40.0];
    let cow = SimdCow::<f32, PreferredArch, Unaligned>::borrow_slice(&data).unwrap();
    let indices = vec![3, 1, 2, 0];
    let gathered = cow.gather(&indices).unwrap();
    assert_eq!(gathered.as_ref(), &[40.0f32, 20.0, 30.0, 10.0]);
}

#[test]
fn test_prefix_scan_inclusive_add_f32() {
    let data = vec![1.0f32, 2.0, 3.0, 4.0];
    let view = SimdView::<f32, PreferredArch, Unaligned>::new(&data).unwrap();
    let mut out = vec![0.0f32; 4];
    view.prefix_scan(&mut out, ScanAdd, Inclusive).unwrap();
    assert_eq!(out, vec![1.0f32, 3.0, 6.0, 10.0]);
}

#[test]
fn test_prefix_scan_exclusive_add_f32() {
    let data = vec![1.0f32, 2.0, 3.0, 4.0];
    let view = SimdView::<f32, PreferredArch, Unaligned>::new(&data).unwrap();
    let mut out = vec![0.0f32; 4];
    view.prefix_scan(&mut out, ScanAdd, Exclusive).unwrap();
    assert_eq!(out, vec![0.0f32, 1.0, 3.0, 6.0]);
}

#[test]
fn test_prefix_scan_mul_i32() {
    let data = vec![1i32, 2, 3, 4];
    let view = SimdView::<i32, PreferredArch, Unaligned>::new(&data).unwrap();
    let mut out = vec![0i32; 4];

    view.prefix_scan(&mut out, ScanMul, Inclusive).unwrap();
    assert_eq!(out, vec![1i32, 2, 6, 24]);

    view.prefix_scan(&mut out, ScanMul, Exclusive).unwrap();
    assert_eq!(out, vec![1i32, 1, 2, 6]);
}

#[test]
fn test_prefix_scan_min_max() {
    let data = vec![5f32, 2.0, 8.0, 1.0, 9.0];
    let view = SimdView::<f32, PreferredArch, Unaligned>::new(&data).unwrap();
    let mut out = vec![0.0f32; 5];

    view.prefix_scan(&mut out, ScanMin, Inclusive).unwrap();
    assert_eq!(out, vec![5.0f32, 2.0, 2.0, 1.0, 1.0]);

    view.prefix_scan(&mut out, ScanMax, Inclusive).unwrap();
    assert_eq!(out, vec![5.0f32, 5.0, 8.0, 8.0, 9.0]);
}

#[test]
fn test_prefix_scan_cow_allocating() {
    let data = vec![1.0f32, 2.0, 3.0, 4.0];
    let cow = SimdCow::<f32, PreferredArch, Unaligned>::borrow_slice(&data).unwrap();
    let scanned = cow.prefix_scan(ScanAdd, Inclusive).unwrap();
    assert_eq!(scanned.as_ref(), &[1.0f32, 3.0, 6.0, 10.0]);
}

#[test]
fn test_prefix_scan_cow_in_place_borrowed() {
    let data = vec![1.0f32, 2.0, 3.0, 4.0];
    let mut cow = SimdCow::<f32, PreferredArch, Unaligned>::borrow_slice(&data).unwrap();

    // Triggers promotion to owned and scans in-place
    cow.prefix_scan_in_place(ScanAdd, Exclusive).unwrap();
    assert_eq!(cow.as_ref(), &[0.0f32, 1.0, 3.0, 6.0]);

    // Subsequent in-place scan should be allocation-free (still owned)
    cow.prefix_scan_in_place(ScanAdd, Inclusive).unwrap();
    assert_eq!(cow.as_ref(), &[0.0f32, 1.0, 4.0, 10.0]);
}
