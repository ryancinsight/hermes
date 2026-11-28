#![expect(
    clippy::float_cmp,
    reason = "Sparse integration tests compare exact manufactured reference vectors"
)]

use hermes_simd::*;

fn validated<S>(data: S) -> ValidatedData<S>
where
    S: hermes_simd_core::sparse::types::SparseValidate,
{
    ValidatedData::new(data).expect("test sparse fixture must validate")
}

#[test]
fn test_spmv_csr_identity() {
    // 3x3 identity
    let values = [1.0f32, 1.0, 1.0];
    let col_indices = [0i32, 1, 2];
    let row_ptr = [0i32, 1, 2, 3];
    let data = CsrData::new(&values[..], &col_indices[..], &row_ptr[..], 3, 3);
    let x = [5.0f32, 7.0, 11.0];
    let mut y = [0.0f32; 3];
    spmv_csr::<f32>(validated(data), &x, &mut y);
    assert_eq!(y, [5.0, 7.0, 11.0]);
}

#[test]
fn test_spmv_csr_rejects_out_of_range_column() {
    // A column index of 3 is out of range for a 3-column matrix; the SIMD gather
    // would otherwise read out of bounds, so the kernel must reject it.
    let values = [1.0f32, 1.0, 1.0];
    let col_indices = [0i32, 3, 2]; // 3 >= ncols
    let row_ptr = [0i32, 1, 2, 3];
    let data = CsrData::new(&values[..], &col_indices[..], &row_ptr[..], 3, 3);
    assert_eq!(
        ValidatedData::new(data).err(),
        Some(SimdError::IndexOutOfBounds)
    );
}

#[test]
fn test_spmv_csr_accumulates() {
    // 2x2 all-ones matrix
    let values = [1.0f32, 1.0, 1.0, 1.0];
    let col_indices = [0i32, 1, 0, 1];
    let row_ptr = [0i32, 2, 4];
    let data = CsrData::new(&values[..], &col_indices[..], &row_ptr[..], 2, 2);
    let x = [3.0f32, 4.0];
    let mut y = [1.0f32; 2]; // y starts at 1.0 to test accumulation
    spmv_csr::<f32>(validated(data), &x, &mut y);
    // y[0] = 1 + (1*3 + 1*4) = 8; y[1] = 1 + (1*3 + 1*4) = 8
    assert_eq!(y, [8.0, 8.0]);
}

/// Integer description of `y += alpha * A * x`, fixed before a scalar is
/// chosen so the exact-range bound is computed per scalar limit. Entry `k` of
/// row `row` is `+1` when `row + k` is even and `-1` otherwise.
struct ScaledCsrShape {
    row_lengths: &'static [usize],
    ncols: usize,
    /// Column of entry `k` in row `row`, which has `length` entries.
    column: fn(row: usize, k: usize, length: usize) -> usize,
    x: fn(column: usize) -> i64,
    /// Initial `y[row]`, given the exact dot product of the row.
    y: fn(row: usize, dot: i64) -> i64,
    alphas: &'static [i64],
}

/// Row lengths that reach every loop of the kernel on every lane count: empty,
/// single, sub-lane tail, one 16-lane block plus tail (17), one unrolled block
/// plus a lane block plus tail (85), four unrolled blocks plus tail (259).
const FIXTURE_ROW_LENGTHS: [usize; 6] = [0, 1, 5, 17, 85, 259];
const FIXTURE_INITIAL_Y: [i64; 6] = [10, -20, 30, -40, 50, -60];

/// Every entry meets `x[c] = c % 4 + 1`, so every product is nonzero. Partial
/// sums reach several hundred and `y` reaches 60, which only the wide-range
/// scalars hold exactly.
const WIDE_SHAPE: ScaledCsrShape = ScaledCsrShape {
    row_lengths: &FIXTURE_ROW_LENGTHS,
    ncols: 259,
    column: |_, k, _| k,
    x: |column| i64::try_from(column % 4 + 1).expect("invariant: a residue below 4 fits i64"),
    y: |row, _| FIXTURE_INITIAL_Y[row],
    alphas: &[2, -3, 1],
};

/// The fixture lengths three times after the empty row.
const COMPACT_ROW_LENGTHS: [usize; 16] =
    [0, 1, 5, 17, 85, 259, 1, 5, 17, 85, 259, 1, 5, 17, 85, 259];
/// Columns `0..COMPACT_HOT_COLUMNS` hold `x = 1`; every other column holds 0.
const COMPACT_HOT_COLUMNS: usize = 3;

/// Each non-empty row stores one product of magnitude 1, in a hot column: the
/// last entry (tail or last lane) in rows 1 to 5, the first (lane 0) in rows 6
/// to 10, the middle in rows 11 to 15. Every other entry meets `x = 0`, so in
/// any summation order each partial sum is 0 or that product, `alpha * dot` is
/// `+-alpha`, and `y = -dot` makes `y + alpha * dot` equal `(alpha - 1) * dot`.
/// With `alpha` in `{2, -1, 1}` every quantity has magnitude at most 2, the
/// range that `Bf4` (integers 0 to 3) and `F4` (0, 1, 2, 4, 8) both hold
/// exactly. The empty row keeps `y = 2` to show that empty rows are skipped.
const COMPACT_SHAPE: ScaledCsrShape = ScaledCsrShape {
    row_lengths: &COMPACT_ROW_LENGTHS,
    ncols: COMPACT_HOT_COLUMNS + 259,
    column: |row, k, length| {
        let hot = [length - 1, 0, length / 2][row.saturating_sub(1) / 5];
        if k == hot {
            row % COMPACT_HOT_COLUMNS
        } else {
            COMPACT_HOT_COLUMNS + k
        }
    },
    x: |column| i64::from(column < COMPACT_HOT_COLUMNS),
    y: |_, dot| if dot == 0 { 2 } else { -dot },
    alphas: &[2, -1, 1],
};

struct ScaledCsrIntegers {
    values: Vec<i64>,
    columns: Vec<i32>,
    row_ptr: Vec<i32>,
    x: Vec<i64>,
    y: Vec<i64>,
    alphas: &'static [i64],
    /// Exact `A * x` per row.
    dots: Vec<i64>,
    /// Largest magnitude a partial sum of a row's products reaches in any
    /// summation order: the larger of its positive and negative totals.
    partial_extents: Vec<i64>,
}

impl ScaledCsrShape {
    fn integers(&self) -> ScaledCsrIntegers {
        let mut integers = ScaledCsrIntegers {
            values: Vec::new(),
            columns: Vec::new(),
            row_ptr: vec![0],
            x: (0..self.ncols).map(self.x).collect(),
            y: Vec::new(),
            alphas: self.alphas,
            dots: Vec::new(),
            partial_extents: Vec::new(),
        };
        for (row, &length) in self.row_lengths.iter().enumerate() {
            let (mut positive, mut negative) = (0i64, 0i64);
            for k in 0..length {
                let column = (self.column)(row, k, length);
                let sign: i64 = if (row + k) % 2 == 0 { 1 } else { -1 };
                let product = sign * integers.x[column];
                integers.values.push(sign);
                let column = i32::try_from(column).expect("invariant: fixture column fits i32");
                integers.columns.push(column);
                positive += product.max(0);
                negative += (-product).max(0);
            }
            let nnz = i32::try_from(integers.values.len()).expect("invariant: nnz fits i32");
            integers.row_ptr.push(nnz);
            integers.dots.push(positive - negative);
            integers.partial_extents.push(positive.max(negative));
            integers.y.push((self.y)(row, positive - negative));
        }
        integers
    }
}

impl ScaledCsrIntegers {
    /// Whether a scalar that represents every integer of magnitude at most
    /// `exact_limit` holds every quantity the kernel forms exactly: `y`, each
    /// partial sum of a row's products, `alpha * dot` and `y + alpha * dot`.
    fn fits(&self, exact_limit: i64) -> bool {
        let rows = self.y.iter().zip(&self.dots).zip(&self.partial_extents);
        self.alphas.iter().all(|&alpha| {
            rows.clone().all(|((&y, &dot), &extent)| {
                [extent, y, alpha * dot, y + alpha * dot]
                    .into_iter()
                    .all(|quantity| quantity.abs() <= exact_limit)
            })
        })
    }
}

/// The fixture integers as scalars of `T`, which must hold them exactly.
struct ScaledCsrFixture<'a, T> {
    integers: &'a ScaledCsrIntegers,
    values: Vec<T>,
    x: Vec<T>,
    y_initial: Vec<T>,
}

/// Converts an integer that the scalar `T` represents exactly into `T`.
fn exact_element<T: SimdScalar>(value: i64) -> T {
    let magnitude = T::try_from_count(
        usize::try_from(value.unsigned_abs()).expect("invariant: fixture magnitudes fit usize"),
    )
    .expect("invariant: fixture integers are within the scalar count range");
    if value < 0 {
        T::ZERO - magnitude
    } else {
        magnitude
    }
}

impl<'a, T: SimdScalar> ScaledCsrFixture<'a, T> {
    fn new(integers: &'a ScaledCsrIntegers) -> Self {
        let elements = |i: &[i64]| i.iter().map(|&i| exact_element::<T>(i)).collect();
        Self {
            integers,
            values: elements(&integers.values),
            x: elements(&integers.x),
            y_initial: elements(&integers.y),
        }
    }

    fn data(&self) -> ValidatedData<CsrData<'_, T>> {
        validated(CsrData::new(
            &self.values[..],
            &self.integers.columns[..],
            &self.integers.row_ptr[..],
            self.integers.dots.len(),
            self.x.len(),
        ))
    }
}

/// `y += alpha * A * x` equals the scalar reference for each given shape, and
/// at `alpha = 1` is bit-identical to the unscaled `spmv_csr`. A shape the
/// scalar cannot hold exactly fails, so no row can drop a shape unnoticed.
fn check_scaled_csr_matches_scalar_reference<T: SimdOps>(
    exact_limit: i64,
    shapes: &[&ScaledCsrShape],
) {
    for shape in shapes {
        let integers = shape.integers();
        assert!(
            integers.fits(exact_limit),
            "the fixture must stay inside the exact-integer range of the scalar"
        );
        let fixture = ScaledCsrFixture::<T>::new(&integers);
        for &alpha in integers.alphas {
            let want: Vec<T> = integers
                .y
                .iter()
                .zip(&integers.dots)
                .map(|(&y, &dot)| exact_element::<T>(y + alpha * dot))
                .collect();
            let mut y = fixture.y_initial.clone();
            let alpha = exact_element::<T>(alpha);
            spmv_scaled_csr::<T>(fixture.data(), alpha, &fixture.x, &mut y);
            assert_eq!(y, want, "fused scaled CSR SpMV must match the reference");
        }

        let mut unscaled = fixture.y_initial.clone();
        spmv_csr::<T>(fixture.data(), &fixture.x, &mut unscaled);
        let mut fused = fixture.y_initial.clone();
        spmv_scaled_csr::<T>(fixture.data(), T::ONE, &fixture.x, &mut fused);
        assert_eq!(fused, unscaled, "alpha = 1 must reproduce spmv_csr");
    }
}

fn check_scaled_csr_zero_alpha_leaves_y_unchanged<T: SimdOps>(shapes: &[&ScaledCsrShape]) {
    for shape in shapes {
        let integers = shape.integers();
        let fixture = ScaledCsrFixture::<T>::new(&integers);
        let mut y = fixture.y_initial.clone();
        spmv_scaled_csr::<T>(fixture.data(), T::ZERO, &fixture.x, &mut y);
        assert_eq!(y, fixture.y_initial);
    }
}

/// Instantiates the generic scaled-CSR checks once per sealed `SimdOps`
/// scalar, native and newtype alike. Scalars in `wide_and_compact` run both
/// fixtures; those in `compact_only` hold exactly only the compact one. The
/// limit is the largest `L` such that the scalar represents every integer of
/// magnitude at most `L`: `2^p` for a `p`-bit significand, the type maximum
/// for integers. `Bf16` is E8M7 (`p = 8`); `F8` is E4M3 (`p = 4`); `Bf8` is
/// E5M2 (`p = 3`); `Bf4` is E2M1 with the top exponent reserved, holding 0,
/// 0.5, 1, 1.5, 2 and 3 (`L = 3`); `F4` is E3M0, holding signed powers of
/// two, so 3 is absent (`L = 2`). The scalar names feed
/// `test_spmv_scaled_csr_instantiations_cover_every_simd_ops_scalar`.
macro_rules! scaled_csr_instantiations {
    (
        wide_and_compact { $($wide_test:ident, $wide_t:ty, $wide_limit:expr;)+ }
        compact_only { $($compact_test:ident, $compact_t:ty, $compact_limit:expr;)+ }
    ) => {
        $(
            #[test]
            fn $wide_test() {
                let shapes = [&WIDE_SHAPE, &COMPACT_SHAPE];
                check_scaled_csr_matches_scalar_reference::<$wide_t>($wide_limit, &shapes);
            }
        )+
        $(
            #[test]
            fn $compact_test() {
                let shapes = [&COMPACT_SHAPE];
                check_scaled_csr_matches_scalar_reference::<$compact_t>($compact_limit, &shapes);
            }
        )+

        #[test]
        fn test_spmv_scaled_csr_zero_alpha_leaves_y_unchanged() {
            $(check_scaled_csr_zero_alpha_leaves_y_unchanged::<$wide_t>(
                &[&WIDE_SHAPE, &COMPACT_SHAPE],
            );)+
            $(check_scaled_csr_zero_alpha_leaves_y_unchanged::<$compact_t>(&[&COMPACT_SHAPE]);)+
        }

        const SCALED_CSR_INSTANTIATED_SCALARS: &[&str] =
            &[$(stringify!($wide_t),)+ $(stringify!($compact_t),)+];
    };
}

scaled_csr_instantiations! {
    wide_and_compact {
        test_spmv_scaled_csr_matches_scalar_reference_f32, f32, 1 << 24;
        test_spmv_scaled_csr_matches_scalar_reference_f64, f64, 1 << 53;
        test_spmv_scaled_csr_matches_scalar_reference_i32, i32, i64::from(i32::MAX);
        test_spmv_scaled_csr_matches_scalar_reference_f16, F16, 1 << 11;
        test_spmv_scaled_csr_matches_scalar_reference_i16, i16, i64::from(i16::MAX);
        test_spmv_scaled_csr_matches_scalar_reference_f32_newtype, F32, 1 << 24;
        test_spmv_scaled_csr_matches_scalar_reference_f64_newtype, F64, 1 << 53;
        test_spmv_scaled_csr_matches_scalar_reference_i16_newtype, I16, i64::from(i16::MAX);
        test_spmv_scaled_csr_matches_scalar_reference_i32_newtype, I32, i64::from(i32::MAX);
    }
    compact_only {
        test_spmv_scaled_csr_matches_scalar_reference_bf16, Bf16, 1 << 8;
        test_spmv_scaled_csr_matches_scalar_reference_f8, F8, 1 << 4;
        test_spmv_scaled_csr_matches_scalar_reference_bf8, Bf8, 1 << 3;
        test_spmv_scaled_csr_matches_scalar_reference_f4, F4, 1 << 1;
        test_spmv_scaled_csr_matches_scalar_reference_bf4, Bf4, 3;
        test_spmv_scaled_csr_matches_scalar_reference_i8, i8, i64::from(i8::MAX);
        test_spmv_scaled_csr_matches_scalar_reference_i8_newtype, I8, i64::from(i8::MAX);
    }
}

/// The scalars `SimdOps` admits. Rust cannot enumerate a sealed trait's
/// implementors, so this reads the `private::Sealed` impls, the supertrait
/// that seals `SimdOps`, from its dispatch source.
fn sealed_simd_ops_scalars() -> Vec<&'static str> {
    include_str!("../src/dispatch/simd_ops.rs")
        .lines()
        .filter_map(|line| {
            line.strip_prefix("impl private::Sealed for ")?
                .strip_suffix(" {}")
        })
        .map(|path| path.rsplit_once("::").map_or(path, |(_, name)| name))
        .collect()
}

#[test]
fn test_spmv_scaled_csr_instantiations_cover_every_simd_ops_scalar() {
    let mut sealed = sealed_simd_ops_scalars();
    let mut instantiated = SCALED_CSR_INSTANTIATED_SCALARS.to_vec();
    sealed.sort_unstable();
    instantiated.sort_unstable();
    assert_eq!(
        instantiated, sealed,
        "scaled_csr_instantiations! rows must be exactly the sealed SimdOps scalars"
    );
}

fn spmv_scaled_unit_row<T: SimdOps>(alpha: T, x: &[T], y: &mut [T]) {
    let (values, col_indices, row_ptr) = ([T::ONE; 2], [0i32, 1], [0i32, 2, 2]);
    let data = CsrData::new(&values[..], &col_indices[..], &row_ptr[..], 2, 2);
    spmv_scaled_csr::<T>(validated(data), alpha, x, y);
}

/// A zero `alpha` of either sign keeps `y` bit-for-bit where `0 * dot` would
/// give NaN (overflowing dot, non-finite `x`) or flip `-0.0` to `+0.0`.
#[test]
fn test_spmv_scaled_csr_zero_alpha_is_an_exact_no_op() {
    for (alpha, x) in [(0.0f32, [f32::MAX; 2]), (-0.0, [f32::NAN, f32::INFINITY])] {
        let mut y = [-0.0f32, 5.0];
        spmv_scaled_unit_row(alpha, &x, &mut y);
        assert_eq!(y.map(f32::to_bits), [(-0.0f32).to_bits(), 5.0f32.to_bits()]);
    }
    // The `i32` dot `2^30 + 2^30` overflows; a zero `alpha` never forms it.
    let mut y = [7i32, 9];
    spmv_scaled_unit_row(0, &[1 << 30; 2], &mut y);
    assert_eq!(y, [7, 9]);
}

/// A non-finite `alpha` reaches only rows with stored entries; empty rows are
/// skipped.
#[test]
fn test_spmv_scaled_csr_non_finite_alpha_skips_empty_rows() {
    let mut y = [0.0f32, 5.0];
    spmv_scaled_unit_row(f32::NAN, &[1.0; 2], &mut y);
    assert!(y[0].is_nan());
    assert_eq!(y[1], 5.0);
    let mut y = [0.0f32, 5.0];
    spmv_scaled_unit_row(f32::INFINITY, &[1.0; 2], &mut y);
    assert_eq!(y, [f32::INFINITY, 5.0]);
}

#[test]
#[should_panic(expected = "x too short for CSR ncols (got 258, expected >= 259)")]
fn test_spmv_scaled_csr_rejects_short_x() {
    let integers = WIDE_SHAPE.integers();
    let fixture = ScaledCsrFixture::<f32>::new(&integers);
    let mut y = fixture.y_initial.clone();
    spmv_scaled_csr::<f32>(
        fixture.data(),
        2.0,
        &fixture.x[..fixture.x.len() - 1],
        &mut y,
    );
}

#[test]
#[should_panic(expected = "y too short for CSR nrows (got 5, expected >= 6)")]
fn test_spmv_scaled_csr_rejects_short_y() {
    let integers = WIDE_SHAPE.integers();
    let fixture = ScaledCsrFixture::<f32>::new(&integers);
    let mut y = fixture.y_initial[..5].to_vec();
    spmv_scaled_csr::<f32>(fixture.data(), 2.0, &fixture.x, &mut y);
}

#[test]
#[should_panic(expected = "x too short for CSR ncols (got 1, expected >= 2)")]
fn test_spmv_scaled_csr_rejects_short_x_before_zero_alpha_return() {
    spmv_scaled_unit_row(0.0f32, &[1.0], &mut [0.0; 2]);
}

/// Where a one-row CSR witness for the integer overflow sites runs.
#[derive(Clone, Copy, Debug)]
enum OverflowBackend {
    /// The kernel the host dispatches to; its lane count is not known here.
    Dispatched,
    /// The portable `Scalar` architecture, whose lane count fixes the lane
    /// that holds each entry.
    Portable,
}

/// The CSR entry point a witness runs through.
#[derive(Clone, Copy, Debug)]
enum OverflowEntry {
    Scaled,
    Unscaled,
}

/// What a primitive integer does at a witness. The `I8`, `I16` and `I32`
/// newtypes always give the wrapped result.
#[derive(Clone, Copy, Debug)]
enum PrimitiveOutcome {
    Exact,
    Wraps,
    Panics(&'static str),
}

const OVERFLOW_ADD: &str = "attempt to add with overflow";
const OVERFLOW_MULTIPLY: &str = "attempt to multiply with overflow";

/// One row `y0 + alpha * (values . x)` over the exact integers.
#[derive(Debug)]
struct OverflowRow {
    backend: OverflowBackend,
    values: Vec<i64>,
    x: Vec<i64>,
    alpha: i64,
    y0: i64,
    outcome: PrimitiveOutcome,
}

impl OverflowRow {
    fn new(backend: OverflowBackend, values: Vec<i64>, outcome: PrimitiveOutcome) -> Self {
        Self {
            backend,
            x: vec![1; values.len()],
            values,
            alpha: 1,
            y0: 0,
            outcome,
        }
    }
}

/// Witness rows for a primitive integer with largest value `max`. Each
/// `Panics` row has an `Exact` control one step inside the bound. `lanes` is
/// the `Scalar` lane count, so `4 * lanes` entries are one unrolled pass in
/// which lane `l` of accumulator `a` holds entry `a * lanes + l`.
fn overflow_rows(max: i64, lanes: usize) -> Vec<OverflowRow> {
    use OverflowBackend::{Dispatched, Portable};
    use PrimitiveOutcome::{Exact, Panics, Wraps};
    let pass = 4 * lanes;
    let pass_i64 = i64::try_from(pass).expect("invariant: a pass is a few dozen entries");
    let mut rows = Vec::new();
    // `pass` equal entries: every accumulator lane and every lane sum fits,
    // the sum of all lanes is `max + 1` or more (horizontal reduction).
    let reduction = max / pass_i64 + 1;
    for (v, outcome) in [(reduction, Panics(OVERFLOW_ADD)), (reduction - 1, Exact)] {
        rows.push(OverflowRow::new(Portable, vec![v; pass], outcome));
    }
    // Lane 0 holds `+v`, lane 1 `-v` in each of the four accumulators: the
    // lane-0 combine is `4 * v`, the reduction never leaves `[-max - 1, max]`.
    let combine = max / 4 + 1;
    for (v, outcome) in [(combine, Panics(OVERFLOW_ADD)), (combine - 1, Exact)] {
        let mut values = vec![0; pass];
        for accumulator in 0..4 {
            values[accumulator * lanes] = v;
            values[accumulator * lanes + 1] = -v;
        }
        values[2] = 3;
        let row = OverflowRow::new(Portable, values, outcome);
        rows.push(OverflowRow {
            alpha: 2,
            y0: 5,
            ..row
        });
    }
    // Whatever the lane count, 64 equal entries sum to `max + 1` or more and
    // no lane portion does.
    let spread = max / 64 + 1;
    for (v, outcome) in [(spread, Panics(OVERFLOW_ADD)), (spread - 1, Exact)] {
        rows.push(OverflowRow::new(Dispatched, vec![v; 64], outcome));
    }
    // Two entries of lane 0 of accumulator 0 sum to `max + 1`: the lane
    // multiply-add wraps that accumulation, and the reduction then fits.
    let half = max / 2 + 1;
    let mut values = vec![0; 2 * pass];
    (values[0], values[pass], values[1]) = (half, half, 5);
    rows.push(OverflowRow::new(Portable, values, Wraps));
    let root = (1..)
        .find(|r: &i64| r * r > max)
        .expect("invariant: the search is unbounded");
    for backend in [Dispatched, Portable] {
        for (r, outcome) in [(root, Panics(OVERFLOW_MULTIPLY)), (root - 1, Exact)] {
            rows.push(OverflowRow {
                x: vec![r],
                ..OverflowRow::new(backend, vec![r], outcome)
            });
        }
        for (alpha, outcome) in [(max / 2 + 1, Panics(OVERFLOW_MULTIPLY)), (max / 2, Exact)] {
            let row = OverflowRow::new(backend, vec![1], outcome);
            rows.push(OverflowRow {
                alpha,
                x: vec![2],
                ..row
            });
        }
        for (y0, outcome) in [(max, Panics(OVERFLOW_ADD)), (max - 1, Exact)] {
            rows.push(OverflowRow {
                y0,
                ..OverflowRow::new(backend, vec![1], outcome)
            });
        }
        // `root * root` exceeds `max` inside a lane multiply-add of a lane block.
        let mut values = vec![0; 64];
        values[0] = root;
        rows.push(OverflowRow {
            x: values.clone(),
            ..OverflowRow::new(backend, values, Wraps)
        });
    }
    rows
}

/// Runs one witness row through `entry`, returning the updated `y[0]`.
fn run_overflow_row<T>(row: &OverflowRow, entry: OverflowEntry) -> T
where
    T: SimdOps + SimdScalar,
    Scalar: SimdKernel<T>,
{
    let values: Vec<T> = row.values.iter().map(|&v| exact_element(v)).collect();
    let x: Vec<T> = row.x.iter().map(|&v| exact_element(v)).collect();
    let columns: Vec<i32> = (0..values.len())
        .map(|c| i32::try_from(c).expect("invariant: witness rows are short"))
        .collect();
    let row_ptr = [
        0,
        i32::try_from(values.len()).expect("invariant: witness rows are short"),
    ];
    let data = validated(CsrData::new(
        &values[..],
        &columns[..],
        &row_ptr[..],
        1,
        x.len(),
    ));
    let (alpha, mut y) = (exact_element::<T>(row.alpha), [exact_element::<T>(row.y0)]);
    match (row.backend, entry) {
        (OverflowBackend::Dispatched, OverflowEntry::Scaled) => {
            spmv_scaled_csr::<T>(data, alpha, &x, &mut y);
        }
        (OverflowBackend::Dispatched, OverflowEntry::Unscaled) => spmv_csr::<T>(data, &x, &mut y),
        (OverflowBackend::Portable, OverflowEntry::Scaled) => {
            SparseView::<T, Validated<Csr>, Scalar>::from_validated_csr(data)
                .spmv_scaled(alpha, &x, &mut y);
        }
        (OverflowBackend::Portable, OverflowEntry::Unscaled) => {
            SparseView::<T, Validated<Csr>, Scalar>::from_validated_csr(data).spmv(&x, &mut y);
        }
    }
    y[0]
}

/// The integer `value` of `T`, including the most negative one that
/// `exact_element` cannot build from its magnitude.
fn integer_element<T: SimdScalar>(value: i64, max: i64) -> T {
    if value == -max - 1 {
        exact_element::<T>(-max) - T::ONE
    } else {
        exact_element(value)
    }
}

/// Whether integer `+` and `*` trap on overflow in this profile.
fn overflow_checks_enabled() -> bool {
    std::panic::catch_unwind(|| std::hint::black_box(i32::MAX) + std::hint::black_box(1)).is_err()
}

/// Every witness row of a primitive (`primitive`) or newtype integer `T`
/// with largest value `max` gives the documented outcome: the exact result
/// inside the range, the wrapped result at the lane multiply-add and for a
/// newtype, and the panic message of the overflowing site otherwise.
fn check_integer_overflow_rows<T>(max: i64, primitive: bool)
where
    T: SimdOps + SimdScalar,
    Scalar: SimdKernel<T> + SimdStorage<T>,
{
    let traps = overflow_checks_enabled();
    for row in overflow_rows(max, <Scalar as SimdStorage<T>>::LANE_COUNT) {
        let dot: i64 = row.values.iter().zip(&row.x).map(|(v, x)| v * x).sum();
        let exact = row.y0 + row.alpha * dot;
        let wrapped = (exact + max + 1).rem_euclid(2 * (max + 1)) - max - 1;
        let entries: &[OverflowEntry] = if row.alpha == 1 {
            &[OverflowEntry::Scaled, OverflowEntry::Unscaled]
        } else {
            &[OverflowEntry::Scaled]
        };
        for &entry in entries {
            let run = || run_overflow_row::<T>(&row, entry);
            match row.outcome {
                PrimitiveOutcome::Exact => {
                    assert_eq!(run(), integer_element::<T>(exact, max), "{entry:?} {row:?}");
                }
                PrimitiveOutcome::Panics(message) if primitive && traps => {
                    let payload = std::panic::catch_unwind(std::panic::AssertUnwindSafe(run))
                        .expect_err("an overflowing integer site must panic under overflow checks");
                    let observed = payload.downcast_ref::<&str>().copied();
                    assert_eq!(observed, Some(message), "{entry:?} {row:?}");
                }
                _ => assert_eq!(
                    run(),
                    integer_element::<T>(wrapped, max),
                    "{entry:?} {row:?}"
                ),
            }
        }
    }
}

#[test]
fn test_spmv_csr_primitive_integer_overflow_sites_match_the_documented_contract() {
    check_integer_overflow_rows::<i8>(i64::from(i8::MAX), true);
    check_integer_overflow_rows::<i16>(i64::from(i16::MAX), true);
    check_integer_overflow_rows::<i32>(i64::from(i32::MAX), true);
}

#[test]
fn test_spmv_csr_integer_newtypes_wrap_at_every_overflow_site() {
    check_integer_overflow_rows::<I8>(i64::from(i8::MAX), false);
    check_integer_overflow_rows::<I16>(i64::from(i16::MAX), false);
    check_integer_overflow_rows::<I32>(i64::from(i32::MAX), false);
}

#[test]
fn test_spmv_dense_masked() {
    let values = [1.0f32, 0.0, 0.0, 1.0]; // 2x2 identity
    let mask = PackedMask::from_bools(&[true, false, false, true]);
    let data = DenseWithMaskData::new(&values[..], mask.as_view(), 2, 2);
    let x = [6.0f32, 9.0];
    let mut y = [0.0f32; 2];
    spmv_dense_masked::<f32>(data, &x, &mut y);
    assert_eq!(y, [6.0, 9.0]);
}

#[test]
fn test_spmv_dense_masked_word_boundary_rows() {
    // ncols = 70: each row's packed-mask window spans a 64-bit word boundary
    // and leaves a non-multiple-of-lane-width scalar tail. Verified against an
    // independent dense reference.
    let (nrows, ncols) = (3, 70);
    let values: Vec<f32> = (0..nrows * ncols).map(|i| (i % 7) as f32 - 3.0).collect();
    let mask_bools: Vec<bool> = (0..nrows * ncols).map(|i| i % 3 != 1).collect();
    let x: Vec<f32> = (0..ncols).map(|c| (c % 5) as f32 + 0.5).collect();

    let mask = PackedMask::from_bools(&mask_bools);
    let data = DenseWithMaskData::new(&values, mask.as_view(), nrows, ncols);
    let mut y = vec![0.0f32; nrows];
    spmv_dense_masked::<f32>(data, &x, &mut y);

    let mut want = vec![0.0f32; nrows];
    for r in 0..nrows {
        for c in 0..ncols {
            if mask_bools[r * ncols + c] {
                want[r] += values[r * ncols + c] * x[c];
            }
        }
    }
    assert_eq!(y, want);
}

fn assert_dense_masked_short_rows<T: SimdOps + FloatElement>() {
    const WIDTHS: [usize; 6] = [3, 4, 5, 6, 7, 15];
    const NROWS: usize = 3;

    for ncols in WIDTHS {
        let values: Vec<T> = (0..NROWS * ncols)
            .map(|index| T::from_count(index % 5 + 1))
            .collect();
        let x: Vec<T> = (0..ncols)
            .map(|column| T::from_count(column % 4 + 1))
            .collect();
        let mask_bits: Vec<bool> = (0..NROWS * ncols).map(|index| index % 3 != 1).collect();
        let mask = PackedMask::from_bools(&mask_bits);
        let data = DenseWithMaskData::new(&values, mask.as_view(), NROWS, ncols);
        let mut actual = vec![T::from_count(2); NROWS];
        let mut expected = actual.clone();

        for row in 0..NROWS {
            for column in 0..ncols {
                let index = row * ncols + column;
                if mask_bits[index] {
                    expected[row] += values[index] * x[column];
                }
            }
        }

        spmv_dense_masked::<T>(data, &x, &mut actual);
        assert_eq!(actual, expected, "DenseWithMask short-row width {ncols}");
    }
}

#[test]
fn test_spmv_dense_masked_short_rows_all_float_precisions() {
    // All products and sums are small integers, so both reduction orders are
    // exact in f32 and f64. The widths straddle the measured scalar/masked
    // crossover and include the one-full-vector-plus-tail regime.
    assert_dense_masked_short_rows::<f32>();
    assert_dense_masked_short_rows::<f64>();
}

#[test]
fn test_dense_masked_validation_requires_exact_logical_shape() {
    let exact_values = [1.0f32, 2.0, 3.0, 4.0];
    let exact_mask = PackedMask::from_bools(&[true, false, true, false]);
    let exact = ValidatedData::new(DenseWithMaskData::new(
        &exact_values,
        exact_mask.as_view(),
        2,
        2,
    ))
    .expect("exact DenseWithMask shape must validate");
    assert_eq!(exact.storage().values, exact_values);
    assert_eq!(exact.storage().mask.len(), 4);

    let short_values = [1.0f32, 2.0, 3.0];
    assert_eq!(
        ValidatedData::new(DenseWithMaskData::new(
            &short_values,
            exact_mask.as_view(),
            2,
            2,
        ))
        .err(),
        Some(SimdError::LengthMismatch)
    );

    let long_values = [1.0f32, 2.0, 3.0, 4.0, 5.0];
    let long_mask = PackedMask::from_bools(&[true, false, true, false, true]);
    assert_eq!(
        ValidatedData::new(DenseWithMaskData::new(
            &long_values,
            exact_mask.as_view(),
            2,
            2,
        ))
        .err(),
        Some(SimdError::LengthMismatch)
    );

    let short_mask = PackedMask::from_bools(&[true, false, true]);
    assert_eq!(
        ValidatedData::new(DenseWithMaskData::new(
            &exact_values,
            short_mask.as_view(),
            2,
            2,
        ))
        .err(),
        Some(SimdError::LengthMismatch)
    );
    assert_eq!(
        ValidatedData::new(DenseWithMaskData::new(
            &exact_values,
            long_mask.as_view(),
            2,
            2,
        ))
        .err(),
        Some(SimdError::LengthMismatch)
    );

    let empty_values: [f32; 0] = [];
    let empty_mask = PackedMask::from_bools(&[]);
    assert_eq!(
        ValidatedData::new(DenseWithMaskData::new(
            &empty_values,
            empty_mask.as_view(),
            usize::MAX,
            2,
        ))
        .err(),
        Some(SimdError::LengthMismatch)
    );
}

#[test]
#[should_panic(expected = "DenseWithMask spmv: values 5, mask 5, and shape 2x2")]
fn test_spmv_dense_masked_rejects_oversized_storage() {
    let values = [1.0f32; 5];
    let mask = PackedMask::from_bools(&[true; 5]);
    let data = DenseWithMaskData::new(&values, mask.as_view(), 2, 2);
    let mut y = [0.0f32; 2];
    spmv_dense_masked(data, &[1.0; 2], &mut y);
}

#[test]
#[should_panic(expected = "DenseWithMask nnz: values 5, mask 5, and shape 2x2")]
fn test_dense_masked_nnz_rejects_oversized_storage() {
    let values = [1.0f32; 5];
    let mask = PackedMask::from_bools(&[true; 5]);
    let data = DenseWithMaskData::new(&values, mask.as_view(), 2, 2);
    let _ = data.nnz();
}

#[test]
#[should_panic(expected = "DenseWithMask sum_values: values 5, mask 5, and shape 2x2")]
fn test_dense_masked_sum_rejects_oversized_storage() {
    let values = [1.0f32; 5];
    let mask = PackedMask::from_bools(&[true; 5]);
    let data = DenseWithMaskData::new(&values, mask.as_view(), 2, 2);
    let cow: SparseCow<f32, DenseWithMask, Scalar> = SparseCow::borrowed(data);
    let _ = cow.sum_values();
}

#[test]
#[should_panic(expected = "DenseWithMask elementwise_mul_dense: values 5, mask 5, and shape 2x2")]
fn test_dense_masked_elementwise_rejects_oversized_storage() {
    let values = [1.0f32; 5];
    let mask = PackedMask::from_bools(&[true; 5]);
    let data = DenseWithMaskData::new(&values, mask.as_view(), 2, 2);
    let cow: SparseCow<f32, DenseWithMask, Scalar> = SparseCow::borrowed(data);
    let mut out = [0.0f32; 5];
    cow.elementwise_mul_dense(&[1.0; 5], &mut out);
}

#[test]
fn test_blocked_coo_4x4_spmv() {
    // 4x4 identity matrix as one 4x4 block
    let block: Vec<f32> = vec![
        1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
    ];
    let block_row = [0i32];
    let block_col = [0i32];
    let data = BlockedCooData::new(&block[..], &block_row[..], &block_col[..], 1, 4, 4);
    let x = [2.0f32, 3.0, 5.0, 7.0];
    let mut y = [0.0f32; 4];
    spmv_bcoo::<f32, 4, 4>(validated(data), &x, &mut y);
    assert_eq!(y, x);
}

#[test]
fn test_blocked_coo_simd_branch_matches_scalar_reference() {
    // `BN = 8` triggers a SIMD branch on common hosts (AVX2 lane=8 → 1×; NEON /
    // scalar lane=4 → 2×), exercising the now-live runtime-dispatched BCOO SIMD
    // path (previously hardcoded to ScalarArch). Dyadic-exact entries keep the
    // SIMD reduction order bit-identical to the sequential reference.
    const BM: usize = 2;
    const BN: usize = 8;
    let nrows = 2usize;
    let ncols = 16usize;
    let nblocks = 2usize;
    // Two 2×8 blocks: block 0 at (row 0, col 0), block 1 at (row 0, col 8).
    let block: Vec<f32> = (0..nblocks * BM * BN)
        .map(|i| ((i % 7) as f32 - 3.0) * 0.5)
        .collect();
    let block_row = [0i32, 0i32];
    let block_col = [0i32, 8i32];
    let x: Vec<f32> = (0..ncols).map(|i| (i % 5) as f32 - 2.0).collect();

    let mut want = vec![0.0f32; nrows];
    for b in 0..nblocks {
        let br = block_row[b] as usize;
        let bc = block_col[b] as usize;
        for i in 0..BM {
            let mut acc = 0.0f32;
            for j in 0..BN {
                acc += block[b * BM * BN + i * BN + j] * x[bc + j];
            }
            want[br + i] += acc;
        }
    }

    let data = BlockedCooData::new(
        &block[..],
        &block_row[..],
        &block_col[..],
        nblocks,
        nrows,
        ncols,
    );
    let mut y = vec![0.0f32; nrows];
    spmv_bcoo::<f32, BM, BN>(validated(data), &x, &mut y);
    assert_eq!(y, want, "BCOO SIMD spmv must match scalar reference");
}

#[test]
fn test_blocked_coo_simd_double_lane_branch_matches_scalar_reference() {
    // `BN = 16` triggers the `BN == lane_count * 2` SIMD branch on an AVX2 host
    // (lane = 8 → 2×) and the `BN == lane_count` branch on an AVX-512 host
    // (lane = 16 → 1×) — the native AVX-512 lane width. Both are real
    // runtime-dispatched SIMD paths through `dispatch_spmv_bcoo`. Dyadic-exact
    // entries keep every partial sum exact in f32, so the SIMD reduction (two
    // half-width products summed lane-wise, then horizontally reduced) is
    // bit-identical to the sequential scalar reference.
    const BM: usize = 2;
    const BN: usize = 16;
    let nrows = 2usize;
    let ncols = 32usize;
    let nblocks = 2usize;
    // Two 2×16 blocks: block 0 at (row 0, col 0), block 1 at (row 0, col 16).
    let block: Vec<f32> = (0..nblocks * BM * BN)
        .map(|i| ((i % 7) as f32 - 3.0) * 0.5)
        .collect();
    let block_row = [0i32, 0i32];
    let block_col = [0i32, 16i32];
    let x: Vec<f32> = (0..ncols).map(|i| (i % 5) as f32 - 2.0).collect();

    let mut want = vec![0.0f32; nrows];
    for b in 0..nblocks {
        let br = block_row[b] as usize;
        let bc = block_col[b] as usize;
        for i in 0..BM {
            let mut acc = 0.0f32;
            for j in 0..BN {
                acc += block[b * BM * BN + i * BN + j] * x[bc + j];
            }
            want[br + i] += acc;
        }
    }

    let data = BlockedCooData::new(
        &block[..],
        &block_row[..],
        &block_col[..],
        nblocks,
        nrows,
        ncols,
    );
    let mut y = vec![0.0f32; nrows];
    spmv_bcoo::<f32, BM, BN>(validated(data), &x, &mut y);
    assert_eq!(
        y, want,
        "BCOO double-lane SIMD spmv must match scalar reference"
    );
}

#[test]
fn test_sellp_spmv_correctness() {
    let values = [1.0f32, 2.0, 3.0, 4.0];
    let col_indices = [0i32, 1, 2, 3];
    let slice_ptr = [0i32, 4];
    let slice_col_count = [1i32];
    let data = SellPData::new(
        &values[..],
        &col_indices[..],
        &slice_ptr[..],
        &slice_col_count[..],
        4,
        4,
    );
    let x = [10.0f32, 10.0, 10.0, 10.0];
    let mut y = [0.0f32; 4];

    let view = SparseView::<f32, Validated<SellP<4>>, Scalar>::try_from_sellp(data).unwrap();
    view.spmv(&x, &mut y);

    assert_eq!(y, [10.0, 20.0, 30.0, 40.0]);
}

#[test]
fn test_sellp_spmv_rejects_out_of_range_column() {
    // Column index 4 is out of range for a 4-column matrix. The vectorized SELL-p
    // path (engaged because `Scalar::LANE_COUNT == 4 == C`) gathers `x[col]`
    // without a per-lane bounds check, so it must reject the matrix up front
    // rather than read out of bounds. Regression for the SELL-p gather OOB.
    let values = [1.0f32, 2.0, 3.0, 4.0];
    let col_indices_bad = [0i32, 1, 4, 3];
    let slice_ptr = [0i32, 4];
    let slice_col_count = [1i32];
    let data = SellPData::new(
        &values[..],
        &col_indices_bad[..],
        &slice_ptr[..],
        &slice_col_count[..],
        4,
        4,
    );
    assert_eq!(
        SparseView::<f32, Validated<SellP<4>>, Scalar>::try_from_sellp(data).err(),
        Some(SimdError::IndexOutOfBounds)
    );
}

#[test]
fn test_sellp_spmv_rejects_bad_slice_geometry() {
    // `slice_col_count = 2` with `C = 4` and only 4 values makes the second
    // column's load `values[4..8]` read past the 4-element array. The vectorized
    // load is full-width and unchecked, so the matrix must be rejected. Regression
    // for the SELL-p `values[offset..]` over-read.
    let values = [1.0f32, 2.0, 3.0, 4.0];
    let col_indices = [0i32, 1, 2, 3];
    let slice_ptr = [0i32, 4];
    let slice_col_count = [2i32]; // 0 + 2*4 = 8 > values.len() == 4
    let data = SellPData::new(
        &values[..],
        &col_indices[..],
        &slice_ptr[..],
        &slice_col_count[..],
        8,
        4,
    );
    assert_eq!(
        SparseView::<f32, Validated<SellP<4>>, Scalar>::try_from_sellp(data).err(),
        Some(SimdError::LengthMismatch)
    );
}

#[test]
fn test_sellp_spmv_dispatch() {
    let values = [1.0f32, 2.0, 3.0, 4.0];
    let col_indices = [0i32, 1, 2, 3];
    let slice_ptr = [0i32, 4];
    let slice_col_count = [1i32];
    let data = SellPData::new(
        &values[..],
        &col_indices[..],
        &slice_ptr[..],
        &slice_col_count[..],
        4,
        4,
    );
    let x = [10.0f32, 10.0, 10.0, 10.0];
    let mut y = [0.0f32; 4];

    spmv_sellp::<f32, 4>(validated(data.clone()), &x, &mut y);
    assert_eq!(y, [10.0, 20.0, 30.0, 40.0]);

    let mut y8 = [0.0f32; 4];
    let data8 = SellPData::new(
        &[1.0, 2.0, 3.0, 4.0, 0.0, 0.0, 0.0, 0.0][..],
        &[0, 1, 2, 3, 0, 0, 0, 0][..],
        &[0, 8][..],
        &[1][..],
        4,
        4,
    );
    spmv_sellp::<f32, 8>(validated(data8), &x, &mut y8);
    assert_eq!(y8, [10.0, 20.0, 30.0, 40.0]);
}

/// Chunk-width-aware dispatch differential: a two-slice SELL-8 matrix with
/// per-slice padding and non-uniform values must match an independent dense
/// reference through the public dispatcher, whichever ISA the ladder picks
/// (on an AVX2 host C = 8 engages the AVX2 kernel; C = 4 routes to the
/// lane-matching scalar marker instead of the per-element fallback).
#[test]
fn test_sellp_spmv_dispatch_multislice_differential() {
    // 16 rows (2 slices of C = 8), 8 columns; row r has entries at columns
    // (r % 8) and ((r + 3) % 8) with values derived from r.
    const C: usize = 8;
    let nrows = 16usize;
    let ncols = 8usize;
    let cols_per_row = 2usize;

    // SELL-p layout: slice s columns stored column-major within the slice.
    let mut values = vec![0.0f32; nrows * cols_per_row];
    let mut col_indices = vec![0i32; nrows * cols_per_row];
    for s in 0..2 {
        for col in 0..cols_per_row {
            for row in 0..C {
                let r = s * C + row;
                let idx = s * (cols_per_row * C) + col * C + row;
                let c = (r + 3 * col) % ncols;
                values[idx] = (r * cols_per_row + col) as f32 * 0.5 - 3.0;
                col_indices[idx] = c as i32;
            }
        }
    }
    let slice_ptr = [
        0i32,
        (cols_per_row * C) as i32,
        (2 * cols_per_row * C) as i32,
    ];
    let slice_col_count = [cols_per_row as i32, cols_per_row as i32];
    let data = SellPData::<f32, C>::new(
        &values,
        &col_indices,
        &slice_ptr[..],
        &slice_col_count[..],
        nrows,
        ncols,
    );

    let x: Vec<f32> = (0..ncols).map(|i| i as f32 + 0.25).collect();
    let mut y = vec![1.0f32; nrows];
    spmv_sellp::<f32, C>(validated(data), &x, &mut y);

    // Independent dense reference: y_ref = 1 + Σ values[r,·]·x[col].
    let mut y_ref = vec![1.0f32; nrows];
    for (r, y_r) in y_ref.iter_mut().enumerate() {
        let s = r / C;
        let row = r % C;
        for col in 0..cols_per_row {
            let idx = s * (cols_per_row * C) + col * C + row;
            *y_r += values[idx] * x[col_indices[idx] as usize];
        }
    }
    assert_eq!(y, y_ref, "SELL-8 dispatch diverges from dense reference");
}

#[test]
fn test_unpack_int4() {
    let packed = [0xABu8];
    let mut unpacked = [0i8; 2];
    unpack_int4(&packed, &mut unpacked);
    assert_eq!(unpacked[0], -5);
    assert_eq!(unpacked[1], -6);
}

#[test]
fn test_unpack_int4_signed_nibble_domain() {
    let packed = [0x10u8, 0x32, 0x54, 0x76, 0x98, 0xBA, 0xDC, 0xFE];
    let mut unpacked = [0i8; 16];
    unpack_int4(&packed, &mut unpacked);
    assert_eq!(
        unpacked,
        [0, 1, 2, 3, 4, 5, 6, 7, -8, -7, -6, -5, -4, -3, -2, -1]
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// SparseCow tests
// ─────────────────────────────────────────────────────────────────────────────

fn make_csr_3x3_identity() -> (Vec<f32>, Vec<i32>, Vec<i32>) {
    (
        vec![1.0f32, 1.0, 1.0],
        vec![0i32, 1, 2],
        vec![0i32, 1, 2, 3],
    )
}

#[test]
fn test_csr_cow_borrowed_is_zero_alloc() {
    let (vals, cols, row_ptr) = make_csr_3x3_identity();
    let data = CsrData::new(&vals, &cols, &row_ptr, 3, 3);
    let cow: SparseCow<f32, Validated<Csr>, Scalar> = SparseCow::try_borrowed(data).unwrap();

    // Borrowed variant: no allocation, dimensions correct.
    assert!(cow.is_borrowed());
    assert!(!cow.is_owned());
    assert_eq!(cow.nrows(), 3);
    assert_eq!(cow.ncols(), 3);
}

#[test]
fn test_csr_cow_spmv_borrowed() {
    let (vals, cols, row_ptr) = make_csr_3x3_identity();
    let data = CsrData::new(&vals, &cols, &row_ptr, 3, 3);
    let cow: SparseCow<f32, Validated<Csr>, Scalar> = SparseCow::try_borrowed(data).unwrap();

    let x = [2.0f32, 3.0, 5.0];
    let mut y = [0.0f32; 3];
    cow.spmv(&x, &mut y);
    assert_eq!(y, [2.0, 3.0, 5.0]);
}

#[test]
fn test_csr_cow_spmv_owned() {
    let (vals, cols, row_ptr) = make_csr_3x3_identity();
    let cow = SparseCow::<f32, Validated<Csr>, Scalar>::from_slices(&vals, &cols, &row_ptr, 3, 3)
        .unwrap();

    assert!(cow.is_owned());

    let x = [2.0f32, 3.0, 5.0];
    let mut y = [0.0f32; 3];
    cow.spmv(&x, &mut y);
    assert_eq!(y, [2.0, 3.0, 5.0]);
}

#[test]
fn test_csr_cow_to_owned_promotes_borrowed() {
    let (vals, cols, row_ptr) = make_csr_3x3_identity();
    let data = CsrData::new(&vals, &cols, &row_ptr, 3, 3);
    let mut cow: SparseCow<f32, Validated<Csr>, Scalar> = SparseCow::try_borrowed(data).unwrap();

    assert!(cow.is_borrowed());
    cow.to_owned();
    assert!(cow.is_owned());

    // SpMV still correct after promotion.
    let x = [7.0f32, 11.0, 13.0];
    let mut y = [0.0f32; 3];
    cow.spmv(&x, &mut y);
    assert_eq!(y, [7.0, 11.0, 13.0]);
}

#[test]
fn test_csr_cow_to_owned_noop_when_already_owned() {
    let (vals, cols, row_ptr) = make_csr_3x3_identity();
    let mut cow =
        SparseCow::<f32, Validated<Csr>, Scalar>::from_slices(&vals, &cols, &row_ptr, 3, 3)
            .unwrap();
    assert!(cow.is_owned());
    cow.to_owned(); // must not panic or reallocate
    assert!(cow.is_owned());
}

#[test]
fn test_csr_cow_sum_values_borrowed() {
    let (vals, cols, row_ptr) = make_csr_3x3_identity();
    let data = CsrData::new(&vals, &cols, &row_ptr, 3, 3);
    let cow: SparseCow<f32, Csr, Scalar> = SparseCow::borrowed(data);
    // 3x1.0 = 3.0
    let s = cow.sum_values();
    assert!((s - 3.0f32).abs() < 1e-6);
}

#[test]
fn test_csr_cow_sum_values_owned() {
    let vals = vec![2.0f32, 5.0, 1.0];
    let cols = vec![0i32, 1, 2];
    let row_ptr = vec![0i32, 1, 2, 3];
    let cow = SparseCow::<f32, Csr, Scalar>::from_slices(&vals, &cols, &row_ptr, 3, 3);
    let s = cow.sum_values();
    assert!((s - 8.0f32).abs() < 1e-6);
}

#[test]
fn test_csr_cow_elementwise_mul_dense() {
    let vals = vec![2.0f32, 3.0, 4.0];
    let cols = vec![0i32, 1, 2];
    let row_ptr = vec![0i32, 1, 2, 3];
    let cow = SparseCow::<f32, Csr, Scalar>::from_slices(&vals, &cols, &row_ptr, 3, 3);
    // Dense: col 0 → 10, col 1 → 20, col 2 → 30
    let dense = [10.0f32, 20.0, 30.0];
    let mut out = [0.0f32; 3];
    cow.elementwise_mul_dense(&dense, &mut out);
    // out[i] = values[i] * dense[col_indices[i]]
    assert!((out[0] - 20.0f32).abs() < 1e-6); // 2.0 * 10
    assert!((out[1] - 60.0f32).abs() < 1e-6); // 3.0 * 20
    assert!((out[2] - 120.0f32).abs() < 1e-6); // 4.0 * 30
}

// ─────────────────────────────────────────────────────────────────────────────
// SellPCow tests
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_sellp_cow_borrowed_spmv() {
    let values = [1.0f32, 2.0, 3.0, 4.0];
    let col_indices = [0i32, 1, 2, 3];
    let slice_ptr = [0i32, 4];
    let slice_col_count = [1i32];
    let data = SellPData::new(&values, &col_indices, &slice_ptr, &slice_col_count, 4, 4);
    let cow: SparseCow<f32, Validated<SellP<4>>, Scalar> = SparseCow::try_borrowed(data).unwrap();

    assert!(cow.is_borrowed());
    let x = [10.0f32; 4];
    let mut y = [0.0f32; 4];
    cow.spmv(&x, &mut y);
    assert_eq!(y, [10.0, 20.0, 30.0, 40.0]);
}

#[test]
fn test_sellp_cow_owned_spmv() {
    let cow = SparseCow::<f32, Validated<SellP<4>>, Scalar>::from_slices(
        &[1.0f32, 2.0, 3.0, 4.0],
        &[0i32, 1, 2, 3],
        &[0i32, 4],
        &[1i32],
        4,
        4,
    )
    .unwrap();
    assert!(cow.is_owned());
    let x = [10.0f32; 4];
    let mut y = [0.0f32; 4];
    cow.spmv(&x, &mut y);
    assert_eq!(y, [10.0, 20.0, 30.0, 40.0]);
}

#[test]
fn test_sellp_cow_to_owned_promotes() {
    let values = [1.0f32, 2.0, 3.0, 4.0];
    let col_indices = [0i32, 1, 2, 3];
    let slice_ptr = [0i32, 4];
    let slice_col_count = [1i32];
    let data = SellPData::new(&values, &col_indices, &slice_ptr, &slice_col_count, 4, 4);
    let mut cow: SparseCow<f32, Validated<SellP<4>>, Scalar> =
        SparseCow::try_borrowed(data).unwrap();
    assert!(cow.is_borrowed());
    cow.to_owned();
    assert!(cow.is_owned());

    let x = [10.0f32; 4];
    let mut y = [0.0f32; 4];
    cow.spmv(&x, &mut y);
    assert_eq!(y, [10.0, 20.0, 30.0, 40.0]);
}

// ─────────────────────────────────────────────────────────────────────────────
// DenseWithMaskCow tests
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_dense_masked_cow_borrowed_spmv() {
    let values = [1.0f32, 0.0, 0.0, 1.0]; // 2x2 identity
    let mask = PackedMask::from_bools(&[true, false, false, true]);
    let data = DenseWithMaskData::new(&values, mask.as_view(), 2, 2);
    let cow: SparseCow<f32, DenseWithMask, Scalar> = SparseCow::borrowed(data);

    assert!(cow.is_borrowed());
    let x = [6.0f32, 9.0];
    let mut y = [0.0f32; 2];
    cow.spmv(&x, &mut y);
    assert_eq!(y, [6.0, 9.0]);
}

#[test]
fn test_dense_masked_cow_owned_spmv() {
    let cow = SparseCow::<f32, DenseWithMask, Scalar>::from_slices(
        &[1.0f32, 0.0, 0.0, 1.0],
        &[true, false, false, true],
        2,
        2,
    );
    assert!(cow.is_owned());
    let x = [6.0f32, 9.0];
    let mut y = [0.0f32; 2];
    cow.spmv(&x, &mut y);
    assert_eq!(y, [6.0, 9.0]);
}

#[test]
fn test_dense_masked_cow_sum_values() {
    let cow = SparseCow::<f32, DenseWithMask, Scalar>::from_slices(
        &[3.0f32, 0.0, 0.0, 5.0],
        &[true, false, false, true],
        2,
        2,
    );
    let s = cow.sum_values();
    assert!((s - 8.0f32).abs() < 1e-6);
}

// ─────────────────────────────────────────────────────────────────────────────
// BlockedCooCow tests
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_bcoo_cow_borrowed_spmv() {
    let block: Vec<f32> = vec![
        1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
    ];
    let block_row = [0i32];
    let block_col = [0i32];
    let data = BlockedCooData::new(&block, &block_row, &block_col, 1, 4, 4);
    let cow: SparseCow<f32, Validated<BlockedCoo<4, 4>>, Scalar> =
        SparseCow::try_borrowed(data).unwrap();

    assert!(cow.is_borrowed());
    let x = [1.0f32, 2.0, 3.0, 4.0];
    let mut y = [0.0f32; 4];
    cow.spmv(&x, &mut y);
    assert_eq!(y, x);
}

#[test]
fn test_bcoo_cow_owned_spmv() {
    let cow = SparseCow::<f32, Validated<BlockedCoo<4, 4>>, Scalar>::from_slices(
        &[
            1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
        ],
        &[0i32],
        &[0i32],
        1,
        4,
        4,
    )
    .unwrap();
    assert!(cow.is_owned());
    let x = [5.0f32, 6.0, 7.0, 8.0];
    let mut y = [0.0f32; 4];
    cow.spmv(&x, &mut y);
    assert_eq!(y, x);
}

#[test]
fn test_bcoo_cow_to_owned_promotes() {
    let block: Vec<f32> = vec![
        2.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 2.0,
    ];
    let block_row = [0i32];
    let block_col = [0i32];
    let data = BlockedCooData::new(&block, &block_row, &block_col, 1, 4, 4);
    let mut cow: SparseCow<f32, Validated<BlockedCoo<4, 4>>, Scalar> =
        SparseCow::try_borrowed(data).unwrap();
    assert!(cow.is_borrowed());
    cow.to_owned();
    assert!(cow.is_owned());

    let x = [1.0f32, 2.0, 3.0, 4.0];
    let mut y = [0.0f32; 4];
    cow.spmv(&x, &mut y);
    assert_eq!(y, [2.0, 4.0, 6.0, 8.0]);
}

#[test]
fn test_sparse_validate_csr_bounds() {
    use hermes_simd_core::sparse::types::SparseValidate;

    // Normal valid CSR
    let values = [1.0f32, 2.0];
    let col_indices = [0i32, 1];
    let row_ptr = [0i32, 1, 2];
    let data = CsrData::new(&values[..], &col_indices[..], &row_ptr[..], 2, 2);
    assert!(data.validate().is_ok());

    // Out of bounds column index
    let col_indices_bad = [0i32, 2]; // ncols is 2, index 2 is out of bounds
    let data_bad_col = CsrData::new(&values[..], &col_indices_bad[..], &row_ptr[..], 2, 2);
    assert_eq!(data_bad_col.validate(), Err(SimdError::IndexOutOfBounds));

    // Negative column index
    let col_indices_neg = [-1i32, 1];
    let data_neg_col = CsrData::new(&values[..], &col_indices_neg[..], &row_ptr[..], 2, 2);
    assert_eq!(data_neg_col.validate(), Err(SimdError::IndexOutOfBounds));

    // Mismatched length between values and col_indices
    let col_indices_short = [0i32];
    let data_short_col = CsrData::new(&values[..], &col_indices_short[..], &row_ptr[..], 2, 2);
    assert_eq!(data_short_col.validate(), Err(SimdError::LengthMismatch));

    // Mismatched length of row_ptr (needs nrows + 1)
    let row_ptr_short = [0i32, 1];
    let data_short_row = CsrData::new(&values[..], &col_indices[..], &row_ptr_short[..], 2, 2);
    assert_eq!(data_short_row.validate(), Err(SimdError::LengthMismatch));

    // row_ptr does not start at 0
    let row_ptr_not_zero = [1i32, 1, 2];
    let data_not_zero_row =
        CsrData::new(&values[..], &col_indices[..], &row_ptr_not_zero[..], 2, 2);
    assert_eq!(
        data_not_zero_row.validate(),
        Err(SimdError::IndexOutOfBounds)
    );

    // last row_ptr does not match values length
    let row_ptr_bad_last = [0i32, 1, 3];
    let data_bad_last = CsrData::new(&values[..], &col_indices[..], &row_ptr_bad_last[..], 2, 2);
    assert_eq!(data_bad_last.validate(), Err(SimdError::LengthMismatch));
}

#[test]
fn test_sparse_validate_sellp_bounds() {
    use hermes_simd_core::sparse::types::SparseValidate;

    // Normal valid SELL-p
    let values = [1.0f32, 2.0, 3.0, 4.0];
    let col_indices = [0i32, 1, 2, 3];
    let slice_ptr = [0i32, 4];
    let slice_col_count = [1i32];
    let data = SellPData::<f32, 4>::new(
        &values[..],
        &col_indices[..],
        &slice_ptr[..],
        &slice_col_count[..],
        4,
        4,
    );
    assert!(data.validate().is_ok());

    // Bad col index
    let col_indices_bad = [0i32, 1, 4, 3]; // ncols is 4, 4 is out of bounds
    let data_bad_col = SellPData::<f32, 4>::new(
        &values[..],
        &col_indices_bad[..],
        &slice_ptr[..],
        &slice_col_count[..],
        4,
        4,
    );
    assert_eq!(data_bad_col.validate(), Err(SimdError::IndexOutOfBounds));
}

#[test]
fn test_sparse_validate_bcoo_bounds() {
    use hermes_simd_core::sparse::types::SparseValidate;

    let block: Vec<f32> = vec![
        1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
    ];
    let block_row = [0i32];
    let block_col = [0i32];
    let data = BlockedCooData::<f32, 4, 4>::new(&block, &block_row, &block_col, 1, 4, 4);
    assert!(data.validate().is_ok());

    // Out of bounds block row
    let block_row_bad = [2i32]; // 2 + BM (4) = 6 > nrows (4)
    let data_bad_row =
        BlockedCooData::<f32, 4, 4>::new(&block, &block_row_bad[..], &block_col, 1, 4, 4);
    assert_eq!(data_bad_row.validate(), Err(SimdError::IndexOutOfBounds));

    // Out of bounds block col
    let block_col_bad = [1i32]; // 1 + BN (4) = 5 > ncols (4)
    let data_bad_col =
        BlockedCooData::<f32, 4, 4>::new(&block, &block_row, &block_col_bad[..], 1, 4, 4);
    assert_eq!(data_bad_col.validate(), Err(SimdError::IndexOutOfBounds));
}

mod sparse_validated_properties {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #[test]
        fn prop_validated_csr_single_entry_spmv_matches_reference(
            ncols in 1usize..16,
            col in 0usize..16,
            value in -32i32..32,
            x_value in -32i32..32,
        ) {
            let col = col % ncols;
            let values = [value as f32];
            let cols = [col as i32];
            let row_ptr = [0i32, 1];
            let mut x = vec![0.0f32; ncols];
            x[col] = x_value as f32;
            let mut y = [0.0f32; 1];

            let data = CsrData::new(&values, &cols, &row_ptr, 1, ncols);
            spmv_csr::<f32>(validated(data), &x, &mut y);

            prop_assert_eq!(y[0], values[0] * x[col]);
        }

        #[test]
        fn prop_validated_csr_rejects_generated_bad_column(
            ncols in 1usize..16,
            extra in 0usize..16,
        ) {
            let values = [1.0f32];
            let cols = [(ncols + extra) as i32];
            let row_ptr = [0i32, 1];
            let data = CsrData::new(&values, &cols, &row_ptr, 1, ncols);

            prop_assert_eq!(ValidatedData::new(data).err(), Some(SimdError::IndexOutOfBounds));
        }

        #[test]
        fn prop_validated_sellp_single_slice_spmv_matches_reference(
            values_i in prop::array::uniform4(-16i32..16),
            x_i in prop::array::uniform4(-16i32..16),
            cols_raw in prop::array::uniform4(0usize..4),
        ) {
            let values = values_i.map(|v| v as f32);
            let x = x_i.map(|v| v as f32);
            let cols = cols_raw.map(|c| c as i32);
            let slice_ptr = [0i32, 4];
            let slice_col_count = [1i32];
            let data = SellPData::<f32, 4>::new(
                &values,
                &cols,
                &slice_ptr,
                &slice_col_count,
                4,
                4,
            );
            let mut y = [0.0f32; 4];

            spmv_sellp::<f32, 4>(validated(data), &x, &mut y);

            for row in 0..4 {
                prop_assert_eq!(y[row], values[row] * x[cols_raw[row]]);
            }
        }

        #[test]
        fn prop_validated_bcoo_single_block_spmv_matches_reference(
            block_i in prop::array::uniform4(-8i32..8),
            x_i in prop::array::uniform4(-8i32..8),
            br in 0usize..3,
            bc in 0usize..3,
        ) {
            let block = block_i.map(|v| v as f32);
            let x = x_i.map(|v| v as f32);
            let block_row = [br as i32];
            let block_col = [bc as i32];
            let data = BlockedCooData::<f32, 2, 2>::new(
                &block,
                &block_row,
                &block_col,
                1,
                4,
                4,
            );
            let mut y = [0.0f32; 4];

            spmv_bcoo::<f32, 2, 2>(validated(data), &x, &mut y);

            let mut want = [0.0f32; 4];
            for row in 0..2 {
                for col in 0..2 {
                    want[br + row] += block[row * 2 + col] * x[bc + col];
                }
            }
            prop_assert_eq!(y, want);
        }
    }
}

#[cfg(target_pointer_width = "64")]
#[test]
fn test_validators_accept_columns_beyond_i32_range() {
    // Exact `usize`-side column validation: with 2^32 + 5 columns the old
    // `ncols as i32` bound wrapped to 5 and misreported the (valid) column 6.
    let ncols = (1usize << 32) + 5;
    let values = [1.0f32; 4];
    let col_indices = [6i32, 1, 2, 3];
    let row_ptr = [0i32, 4];
    let data = CsrData::new(&values[..], &col_indices[..], &row_ptr[..], 1, ncols);
    assert_eq!(
        hermes_simd_core::sparse::types::SparseValidate::validate(&data),
        Ok(())
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// elementwise_mul_dense dense-operand guards
//
// `SparseView::from_sellp` is safe and `SellPData`/`BlockedCooData` fields are
// `pub`, so these guards are the only barrier between a malformed descriptor
// and the unchecked SIMD loads/gathers inside the kernels. The pathological
// dimension descriptors below use tiny real buffers on purpose: the defect
// they pin is a *wrapped* dimension product passing a vacuous guard, not a
// large allocation.
// ─────────────────────────────────────────────────────────────────────────────

/// One-slice SELL-P fixture: `C = 4`, one stored column, 4x4 logical shape.
fn sellp_unit_fixture() -> ([f32; 4], [i32; 4], [i32; 2], [i32; 1]) {
    ([1.0f32, 2.0, 3.0, 4.0], [0i32, 1, 2, 3], [0i32, 4], [1i32])
}

#[test]
#[should_panic(expected = "SELL-p elementwise_mul_dense: dense len 15 < nrows * ncols = 16")]
fn test_sellp_elementwise_rejects_short_dense() {
    let (values, col_indices, slice_ptr, slice_col_count) = sellp_unit_fixture();
    let data = SellPData::new(&values, &col_indices, &slice_ptr, &slice_col_count, 4, 4);
    let view = SparseView::<f32, SellP<4>, Scalar>::from_sellp(data);
    let dense = [1.0f32; 15]; // one element short of nrows * ncols
    let mut out = [0.0f32; 4];
    view.elementwise_mul_dense(&dense, &mut out);
}

#[cfg(target_pointer_width = "64")]
#[test]
#[should_panic(expected = "SELL-p elementwise_mul_dense: nrows * ncols overflows usize")]
fn test_sellp_elementwise_rejects_dimension_product_overflow() {
    let (values, col_indices, slice_ptr, slice_col_count) = sellp_unit_fixture();
    // 4 * (2^62 + 1) wraps `usize`: an unchecked guard would compare the dense
    // length against the wrapped product (4) and pass vacuously.
    let ncols = (1usize << 62) + 1;
    let data = SellPData::new(
        &values,
        &col_indices,
        &slice_ptr,
        &slice_col_count,
        4,
        ncols,
    );
    let view = SparseView::<f32, SellP<4>, Scalar>::from_sellp(data);
    let dense = [1.0f32; 4];
    let mut out = [0.0f32; 4];
    view.elementwise_mul_dense(&dense, &mut out);
}

#[cfg(target_pointer_width = "64")]
#[test]
#[should_panic(expected = "exceeds i32 gather indices")]
fn test_sellp_elementwise_rejects_flat_index_beyond_i32() {
    let (values, col_indices, slice_ptr, slice_col_count) = sellp_unit_fixture();
    // 1 x (2^32 + 5): the product does not overflow `usize`, but flat element
    // indices no longer fit the `i32` gather-index lanes, so `as i32`
    // narrowing would wrap them into arbitrary (possibly negative) offsets.
    let ncols = (1usize << 32) + 5;
    let data = SellPData::new(
        &values,
        &col_indices,
        &slice_ptr,
        &slice_col_count,
        1,
        ncols,
    );
    let view = SparseView::<f32, SellP<4>, Scalar>::from_sellp(data);
    let dense = [1.0f32; 4];
    let mut out = [0.0f32; 4];
    view.elementwise_mul_dense(&dense, &mut out);
}

#[cfg(target_pointer_width = "64")]
#[test]
#[should_panic(expected = "BlockedCoo elementwise_mul_dense: nrows * ncols overflows usize")]
fn test_blocked_coo_elementwise_rejects_dimension_product_overflow() {
    let block = [1.0f32; 4];
    let block_row = [0i32];
    let block_col = [0i32];
    // 2^32 x 2^32 wraps the dense-extent product to zero in release builds; an
    // unchecked guard would then accept any dense buffer and the per-block
    // extent checks would pass against the huge (unwrapped) dimensions.
    let huge = 1usize << 32;
    let data = BlockedCooData::<f32, 2, 2>::new(&block, &block_row, &block_col, 1, huge, huge);
    let view = SparseView::<f32, BlockedCoo<2, 2>, Scalar>::from_blocked_coo(data);
    let dense = [1.0f32; 4];
    let mut out = [0.0f32; 4];
    view.elementwise_mul_dense(&dense, &mut out);
}

#[test]
fn test_blocked_coo_validate_rejects_block_count_overflow() {
    // `nblocks * BM * BN` wraps `usize`. Pre-fix, `validate` panicked on the
    // overflow in debug builds and compared against the wrapped product in
    // release builds; its contract is a typed error in both.
    let block = [1.0f32; 4];
    let block_row = [0i32];
    let block_col = [0i32];
    let data =
        BlockedCooData::<f32, 2, 2>::new(&block, &block_row, &block_col, usize::MAX / 2, 4, 4);
    assert_eq!(
        hermes_simd_core::sparse::types::SparseValidate::validate(&data),
        Err(SimdError::LengthMismatch)
    );
}
