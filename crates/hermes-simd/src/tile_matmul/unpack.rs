/// Unpacks packed 4-bit signed integers (stored 2 per byte) into an 8-bit signed integer slice.
#[inline]
pub fn unpack_int4(packed: &[u8], unpacked: &mut [i8]) {
    let len = packed.len();
    assert!(unpacked.len() >= len * 2);
    for i in 0..len {
        let byte = packed[i] as i8;
        unpacked[2 * i] = (byte << 4) >> 4;
        unpacked[2 * i + 1] = byte >> 4;
    }
}
