# CHANGE 1: ops.rs — Replace sealed_op with crate::private::Sealed
$opsPath = 'D:\atlas\repos\hermes\crates\hermes-simd-core\src\ops.rs'
$ops = [System.IO.File]::ReadAllText($opsPath)

# Remove the sealed_op module block (handles both CRLF and LF)
$ops = $ops -replace "mod sealed_op \{`r?`n    pub trait Sealed \{\}`r?`n\}`r?`n", ""

# Replace all references to sealed_op::Sealed with crate::private::Sealed
$ops = $ops.Replace("sealed_op::Sealed", "crate::private::Sealed")

[System.IO.File]::WriteAllText($opsPath, $ops)
$v1 = [System.IO.File]::ReadAllText($opsPath)
Write-Host "CHANGE 1 ops.rs — has 'sealed_op': $($v1 -match 'sealed_op') | has 'crate::private::Sealed': $($v1 -match 'crate::private::Sealed')"

# CHANGE 2: sparse/mod.rs — Replace sealed_fmt with crate::private::Sealed
$sparsePath = 'D:\atlas\repos\hermes\crates\hermes-simd-core\src\sparse\mod.rs'
$sparse = [System.IO.File]::ReadAllText($sparsePath)

# Remove the sealed_fmt module block
$sparse = $sparse -replace "mod sealed_fmt \{`r?`n    pub trait SealedFmt \{\}`r?`n\}`r?`n", ""

# Replace trait bound
$sparse = $sparse.Replace("SparseFormat: sealed_fmt::SealedFmt + Send + Sync + 'static", "SparseFormat: crate::private::Sealed + Send + Sync + 'static")

# Replace sealed impls
$sparse = $sparse.Replace("impl sealed_fmt::SealedFmt for Csr {}", "impl crate::private::Sealed for Csr {}")
$sparse = $sparse.Replace("impl<const C: usize> sealed_fmt::SealedFmt for SellP<C> {}", "impl<const C: usize> crate::private::Sealed for SellP<C> {}")
$sparse = $sparse.Replace("impl<const BM: usize, const BN: usize> sealed_fmt::SealedFmt for BlockedCoo<BM, BN> {}", "impl<const BM: usize, const BN: usize> crate::private::Sealed for BlockedCoo<BM, BN> {}")
$sparse = $sparse.Replace("impl sealed_fmt::SealedFmt for DenseWithMask {}", "impl crate::private::Sealed for DenseWithMask {}")

[System.IO.File]::WriteAllText($sparsePath, $sparse)
$v2 = [System.IO.File]::ReadAllText($sparsePath)
Write-Host "CHANGE 2 sparse/mod.rs — has 'sealed_fmt': $($v2 -match 'sealed_fmt') | has 'crate::private::Sealed': $($v2 -match 'crate::private::Sealed')"

# CHANGE 3: mask.rs — Add to_native_mask method before Default impl
$maskPath = 'D:\atlas\repos\hermes\crates\hermes-simd-core\src\mask.rs'
$mask = [System.IO.File]::ReadAllText($maskPath)

$newBlock = @"
impl<const N: usize> BitMask<N> {
    /// Convert this ``BitMask<N>`` to the native hardware mask type for ``Arch``.
    ///
    /// Delegates to [`SimdKernel::mask_from_bitmask`] using the inner ``u64`` value.
    /// Zero runtime cost: the compiler inlines this into a single instruction on
    /// AVX-512 (``KMOV``), a vector comparison + blend mask on AVX2, or a bool-array
    /// copy on scalar backends.
    ///
    /// # Safety
    /// Processor must support the target feature of ``Arch``.
    ///
    /// # Example
    /// ``````rust,ignore
    /// let bm = BitMask::<16>::leading_k(5);
    /// let native: [bool; 4] = unsafe { bm.to_native_mask::<f32, Scalar>() };
    /// ``````
    #[inline(always)]
    pub unsafe fn to_native_mask<T, Arch>(self) -> Arch::Mask
    where
        T: crate::scalar::SimdScalar,
        Arch: crate::kernel::SimdKernel<T>,
    {
        Arch::mask_from_bitmask(self.0)
    }
}

"@

$target = "impl<const N: usize> Default for BitMask<N> {"
$mask = $mask.Replace($target, $newBlock + $target)

[System.IO.File]::WriteAllText($maskPath, $mask)
$v3 = [System.IO.File]::ReadAllText($maskPath)
Write-Host "CHANGE 3 mask.rs — has 'to_native_mask': $($v3 -match 'to_native_mask') | has 'Default for BitMask': $($v3 -match 'Default for BitMask')"

# CHANGE 4: lib.rs — Add SparseShape to re-exports
$libPath = 'D:\atlas\repos\hermes\crates\hermes-simd-core\src\lib.rs'
$lib = [System.IO.File]::ReadAllText($libPath)

$oldExport = "pub use sparse::{
    SparseFormat, SparseView,
    Csr, SellP, BlockedCoo, DenseWithMask,
    CsrData, SellPData, BlockedCooData, DenseWithMaskData,
};"

$newExport = "pub use sparse::{
    SparseFormat, SparseShape, SparseView,
    Csr, SellP, BlockedCoo, DenseWithMask,
    CsrData, SellPData, BlockedCooData, DenseWithMaskData,
};"

# Try both CRLF and LF variants
$lib = $lib.Replace($oldExport.Replace("`n", "`r`n"), $newExport.Replace("`n", "`r`n"))
if (-not ($lib -match 'SparseShape')) {
    $lib = $lib.Replace($oldExport, $newExport)
}

[System.IO.File]::WriteAllText($libPath, $lib)
$v4 = [System.IO.File]::ReadAllText($libPath)
Write-Host "CHANGE 4 lib.rs — has 'SparseShape': $($v4 -match 'SparseShape')"

Write-Host "`nAll changes applied. Running cargo check..."
