Set-StrictMode -Off

# CHANGE 1: ops.rs
$opsPath = 'D:\atlas\repos\hermes\crates\hermes-simd-core\src\ops.rs'
$ops = [System.IO.File]::ReadAllText($opsPath)
$ops = [regex]::Replace($ops, 'mod sealed_op \{\r?\n    pub trait Sealed \{\}\r?\n\}\r?\n', '')
$ops = $ops.Replace('sealed_op::Sealed', 'crate::private::Sealed')
[System.IO.File]::WriteAllText($opsPath, $ops)
$v = [System.IO.File]::ReadAllText($opsPath)
Write-Host ('CHANGE 1 ops.rs — remaining sealed_op: ' + $v.Contains('sealed_op') + ' | crate::private::Sealed: ' + $v.Contains('crate::private::Sealed'))

# CHANGE 2: sparse/mod.rs
$sparsePath = 'D:\atlas\repos\hermes\crates\hermes-simd-core\src\sparse\mod.rs'
$sparse = [System.IO.File]::ReadAllText($sparsePath)
$sparse = [regex]::Replace($sparse, 'mod sealed_fmt \{\r?\n    pub trait SealedFmt \{\}\r?\n\}\r?\n', '')
$sparse = $sparse.Replace("SparseFormat: sealed_fmt::SealedFmt + Send + Sync + 'static", "SparseFormat: crate::private::Sealed + Send + Sync + 'static")
$sparse = $sparse.Replace('impl sealed_fmt::SealedFmt for Csr {}', 'impl crate::private::Sealed for Csr {}')
$sparse = $sparse.Replace('impl<const C: usize> sealed_fmt::SealedFmt for SellP<C> {}', 'impl<const C: usize> crate::private::Sealed for SellP<C> {}')
$sparse = $sparse.Replace('impl<const BM: usize, const BN: usize> sealed_fmt::SealedFmt for BlockedCoo<BM, BN> {}', 'impl<const BM: usize, const BN: usize> crate::private::Sealed for BlockedCoo<BM, BN> {}')
$sparse = $sparse.Replace('impl sealed_fmt::SealedFmt for DenseWithMask {}', 'impl crate::private::Sealed for DenseWithMask {}')
[System.IO.File]::WriteAllText($sparsePath, $sparse)
$v = [System.IO.File]::ReadAllText($sparsePath)
Write-Host ('CHANGE 2 sparse/mod.rs — remaining sealed_fmt: ' + $v.Contains('sealed_fmt') + ' | crate::private::Sealed: ' + $v.Contains('crate::private::Sealed'))

# CHANGE 3: mask.rs — read current, build insert string from bytes, insert before Default impl
$maskPath = 'D:\atlas\repos\hermes\crates\hermes-simd-core\src\mask.rs'
$mask = [System.IO.File]::ReadAllText($maskPath)

# Build the block to insert. Use [char]96 for backtick to avoid PS parsing issues.
$bt = [char]96
$insertBlock = 'impl<const N: usize> BitMask<N> {' + "`n"
$insertBlock += '    /// Convert this ' + $bt + 'BitMask<N>' + $bt + ' to the native hardware mask type for ' + $bt + 'Arch' + $bt + '.' + "`n"
$insertBlock += '    ///' + "`n"
$insertBlock += '    /// Delegates to [' + $bt + 'SimdKernel::mask_from_bitmask' + $bt + '] using the inner ' + $bt + 'u64' + $bt + ' value.' + "`n"
$insertBlock += '    /// Zero runtime cost: the compiler inlines this into a single instruction on' + "`n"
$insertBlock += '    /// AVX-512 (' + $bt + 'KMOV' + $bt + '), a vector comparison + blend mask on AVX2, or a bool-array' + "`n"
$insertBlock += '    /// copy on scalar backends.' + "`n"
$insertBlock += '    ///' + "`n"
$insertBlock += '    /// # Safety' + "`n"
$insertBlock += '    /// Processor must support the target feature of ' + $bt + 'Arch' + $bt + '.' + "`n"
$insertBlock += '    ///' + "`n"
$insertBlock += '    /// # Example' + "`n"
$insertBlock += '    /// ' + $bt + $bt + $bt + 'rust,ignore' + "`n"
$insertBlock += '    /// let bm = BitMask::<16>::leading_k(5);' + "`n"
$insertBlock += '    /// let native: [bool; 4] = unsafe { bm.to_native_mask::<f32, Scalar>() };' + "`n"
$insertBlock += '    /// ' + $bt + $bt + $bt + "`n"
$insertBlock += '    #[inline(always)]' + "`n"
$insertBlock += '    pub unsafe fn to_native_mask<T, Arch>(self) -> Arch::Mask' + "`n"
$insertBlock += '    where' + "`n"
$insertBlock += '        T: crate::scalar::SimdScalar,' + "`n"
$insertBlock += '        Arch: crate::kernel::SimdKernel<T>,' + "`n"
$insertBlock += '    {' + "`n"
$insertBlock += '        Arch::mask_from_bitmask(self.0)' + "`n"
$insertBlock += '    }' + "`n"
$insertBlock += '}' + "`n"
$insertBlock += "`n"

$target = 'impl<const N: usize> Default for BitMask<N> {'
if ($mask.Contains($target)) {
    $mask = $mask.Replace($target, $insertBlock + $target)
    Write-Host 'CHANGE 3 mask.rs — insertion point found'
} else {
    Write-Host 'CHANGE 3 mask.rs — ERROR: target not found'
}
[System.IO.File]::WriteAllText($maskPath, $mask)
$v = [System.IO.File]::ReadAllText($maskPath)
Write-Host ('CHANGE 3 mask.rs — has to_native_mask: ' + $v.Contains('to_native_mask') + ' | has Default for BitMask: ' + $v.Contains('Default for BitMask'))

# CHANGE 4: lib.rs — Add SparseShape to re-exports
$libPath = 'D:\atlas\repos\hermes\crates\hermes-simd-core\src\lib.rs'
$lib = [System.IO.File]::ReadAllText($libPath)

$oldStr = 'SparseFormat, SparseView,'
$newStr = 'SparseFormat, SparseShape, SparseView,'
if ($lib.Contains($oldStr)) {
    $lib = $lib.Replace($oldStr, $newStr)
    Write-Host 'CHANGE 4 lib.rs — replaced SparseView line'
} else {
    Write-Host 'CHANGE 4 lib.rs — ERROR: target not found'
}
[System.IO.File]::WriteAllText($libPath, $lib)
$v = [System.IO.File]::ReadAllText($libPath)
Write-Host ('CHANGE 4 lib.rs — has SparseShape: ' + $v.Contains('SparseShape'))

Write-Host "`nAll changes done."
