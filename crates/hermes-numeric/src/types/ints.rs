/// Transparent wrapper for i8.
#[derive(Copy, Clone, Default, PartialEq, PartialOrd, Debug, Eq, Ord, Hash)]
#[repr(transparent)]
pub struct I8(pub i8);

/// Transparent wrapper for i16.
#[derive(Copy, Clone, Default, PartialEq, PartialOrd, Debug, Eq, Ord, Hash)]
#[repr(transparent)]
pub struct I16(pub i16);

/// Transparent wrapper for i32.
#[derive(Copy, Clone, Default, PartialEq, PartialOrd, Debug, Eq, Ord, Hash)]
#[repr(transparent)]
pub struct I32(pub i32);
