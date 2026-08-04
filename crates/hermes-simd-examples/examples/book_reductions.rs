//! Horizontal reductions with runtime-dispatched SIMD.
//!
//! `sum`, `min`, `max`, `abs_sum`, and `argmin`/`argmax` are all
//! runtime-dispatched over the same ISA selection path.  This example
//! builds a small dataset with known extremes and verifies each reduction.

use hermes_simd::{abs_sum, argmax, argmin, max, min, sum};

fn main() {
    let data: Vec<f32> = vec![-5.0, 3.0, -1.0, 7.0, 2.0, -9.0, 4.0, 0.0];

    let s   = sum::<f32>(&data);
    let mn  = min::<f32>(&data);
    let mx  = max::<f32>(&data);
    let l1  = abs_sum::<f32>(&data);
    let ami = argmin::<f32>(&data).expect("non-empty, no NaN");
    let amx = argmax::<f32>(&data).expect("non-empty, no NaN");

    println!("data  = {data:?}");
    println!("sum   = {s}");      // −5+3−1+7+2−9+4+0 = 1
    println!("min   = {mn}");     // −9
    println!("max   = {mx}");     // 7
    println!("l1    = {l1}");     // 5+3+1+7+2+9+4+0 = 31
    println!("argmin = ({}, {})", ami.0, ami.1);  // (5, -9.0)
    println!("argmax = ({}, {})", amx.0, amx.1);  // (3, 7.0)

    assert!((s  - 1.0).abs()  < 1e-5);
    assert!((mn - (-9.0)).abs() < 1e-5);
    assert!((mx - 7.0).abs()  < 1e-5);
    assert!((l1 - 31.0).abs() < 1e-5);
    assert_eq!(ami, (5_usize, -9.0_f32));
    assert_eq!(amx, (3_usize, 7.0_f32));

    // Empty slice edge cases.
    assert_eq!(sum::<f32>(&[]),    0.0_f32);
    assert!(argmin::<f32>(&[]).is_none());
    assert!(argmax::<f32>(&[]).is_none());

    println!("all reduction assertions passed");
}
