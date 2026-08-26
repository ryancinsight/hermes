//! Lockstep SIMD chunks over planar input and output slices.

use crate::arch::SimdArch;
use crate::execution::ExecutionMode;
use crate::kernel::SimdKernel;
use crate::scalar::Scalar;
use crate::view::SimdChunk;
use core::marker::PhantomData;

/// Lockstep chunks over const-generic groups of shared inputs and mutable outputs.
///
/// The iterator computes the shortest complete-lane prefix once. Each call to
/// [`Iterator::next`] therefore performs one loop-limit check regardless of the
/// number of planes. This is the canonical iterator for planar kernels whose
/// inputs and outputs advance together.
///
/// Consume the iterator with [`SimdIoChunks::into_remainders`] to recover every
/// slice suffix not yet yielded. After full iteration these are the scalar
/// tails; after early termination they also include complete lane groups. A
/// longer plane's remainder includes everything beyond the shortest plane.
pub struct SimdIoChunks<'input, 'output, T, Arch, Mode, const INPUTS: usize, const OUTPUTS: usize> {
    inputs: [(*const T, usize); INPUTS],
    outputs: [(*mut T, usize); OUTPUTS],
    pos: usize,
    simd_end: usize,
    _marker: PhantomData<(&'input T, &'output mut T, Arch, Mode)>,
}

// SAFETY: moving the iterator transfers shared access to every input and
// exclusive access to every output. Those transfers require `T: Sync` and
// `T: Send`, respectively; the source references guarantee all output ranges
// are mutually exclusive and do not overlap any input range.
unsafe impl<T, Arch, Mode, const INPUTS: usize, const OUTPUTS: usize> Send
    for SimdIoChunks<'_, '_, T, Arch, Mode, INPUTS, OUTPUTS>
where
    T: Scalar + Send + Sync,
    Arch: SimdArch + SimdKernel<T>,
    Mode: ExecutionMode,
{
}

// SAFETY: shared access cannot advance the iterator or expose mutable output
// chunks. `T: Sync` permits the input and output referents to be shared.
unsafe impl<T, Arch, Mode, const INPUTS: usize, const OUTPUTS: usize> Sync
    for SimdIoChunks<'_, '_, T, Arch, Mode, INPUTS, OUTPUTS>
where
    T: Scalar + Sync,
    Arch: SimdArch + SimdKernel<T>,
    Mode: ExecutionMode,
{
}

impl<'input, 'output, T, Arch, Mode, const INPUTS: usize, const OUTPUTS: usize>
    SimdIoChunks<'input, 'output, T, Arch, Mode, INPUTS, OUTPUTS>
where
    T: Scalar,
    Arch: SimdArch + SimdKernel<T>,
    Mode: ExecutionMode,
{
    /// Construct lockstep chunks from references whose host support is proven.
    #[inline(always)]
    pub(crate) fn from_supported_slices(
        inputs: [&'input [T]; INPUTS],
        outputs: [&'output mut [T]; OUTPUTS],
    ) -> Self {
        let inputs = inputs.map(|slice| (slice.as_ptr(), slice.len()));
        let outputs = outputs.map(|slice| (slice.as_mut_ptr(), slice.len()));
        let shortest = inputs
            .iter()
            .map(|&(_, len)| len)
            .chain(outputs.iter().map(|&(_, len)| len))
            .min()
            .unwrap_or(0);
        let simd_end = (shortest / Arch::LANE_COUNT) * Arch::LANE_COUNT;
        Self {
            inputs,
            outputs,
            pos: 0,
            simd_end,
            _marker: PhantomData,
        }
    }

    /// Return the number of complete lockstep lane groups remaining.
    #[inline(always)]
    #[must_use]
    pub fn chunks_remaining(&self) -> usize {
        self.simd_end.saturating_sub(self.pos) / Arch::LANE_COUNT
    }

    /// Consume the iterator and return the unprocessed suffix of every plane.
    ///
    /// The suffixes begin at the current iterator position, so stopping early
    /// never drops complete lane groups. They are disjoint from every chunk that
    /// the iterator already yielded.
    #[inline(always)]
    #[must_use]
    pub fn into_remainders(self) -> ([&'input [T]; INPUTS], [&'output mut [T]; OUTPUTS]) {
        let pos = self.pos;
        let inputs = core::array::from_fn(|index| {
            let (ptr, len) = self.inputs[index];
            // SAFETY: `pos <= simd_end <= len` for every source. The original
            // shared borrow lives for `'input` and the pointer retains its
            // provenance.
            unsafe { core::slice::from_raw_parts(ptr.add(pos), len - pos) }
        });
        let outputs = core::array::from_fn(|index| {
            let (ptr, len) = self.outputs[index];
            // SAFETY: as above, and consuming `self` transfers each source's
            // exclusive tail borrow. Tails from distinct outputs remain disjoint.
            unsafe { core::slice::from_raw_parts_mut(ptr.add(pos), len - pos) }
        });
        (inputs, outputs)
    }
}

impl<'input, 'output, T, Arch, Mode, const INPUTS: usize, const OUTPUTS: usize> Iterator
    for SimdIoChunks<'input, 'output, T, Arch, Mode, INPUTS, OUTPUTS>
where
    T: Scalar,
    Arch: SimdArch + SimdKernel<T>,
    Mode: ExecutionMode,
{
    type Item = (
        [SimdChunk<'input, T, Arch, Mode, &'input [T]>; INPUTS],
        [SimdChunk<'output, T, Arch, Mode, &'output mut [T]>; OUTPUTS],
    );

    #[inline(always)]
    fn next(&mut self) -> Option<Self::Item> {
        if self.pos >= self.simd_end {
            return None;
        }
        let pos = self.pos;
        self.pos += Arch::LANE_COUNT;
        let inputs = core::array::from_fn(|index| {
            let ptr = self.inputs[index].0;
            // SAFETY: `pos + LANE_COUNT <= simd_end`, and `simd_end` is no
            // greater than any input length. The capability that constructed
            // this iterator proves host support.
            unsafe { SimdChunk::from_supported_ptr(ptr.add(pos)) }
        });
        let outputs = core::array::from_fn(|index| {
            let ptr = self.outputs[index].0;
            // SAFETY: the same bounds proof holds for every output. The source
            // mutable references and monotonic `pos` make all yielded chunks
            // mutually disjoint.
            unsafe { SimdChunk::from_supported_ptr_mut(ptr.add(pos)) }
        });
        Some((inputs, outputs))
    }

    #[inline(always)]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.chunks_remaining();
        (remaining, Some(remaining))
    }
}

impl<T, Arch, Mode, const INPUTS: usize, const OUTPUTS: usize> ExactSizeIterator
    for SimdIoChunks<'_, '_, T, Arch, Mode, INPUTS, OUTPUTS>
where
    T: Scalar,
    Arch: SimdArch + SimdKernel<T>,
    Mode: ExecutionMode,
{
}

impl<T, Arch, Mode, const INPUTS: usize, const OUTPUTS: usize> core::iter::FusedIterator
    for SimdIoChunks<'_, '_, T, Arch, Mode, INPUTS, OUTPUTS>
where
    T: Scalar,
    Arch: SimdArch + SimdKernel<T>,
    Mode: ExecutionMode,
{
}
