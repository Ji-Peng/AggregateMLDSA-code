//! Dense polynomial storage and constructors.

use crate::kernels::linear::try_centered_i8;
use crate::sources::poly_helpers::{try_small_i8_cache_from_ring_coeffs, SignedDigitBasis};
use akita_algebra::ring::cyclotomic::BalancedDecomposePow2Params;
use akita_algebra::CyclotomicRing;
use akita_error::{checked, AkitaError};
use akita_params::SUPPORTED_COMMITMENT_RING_DIMS;
use akita_types::RingVec;
use jolt_field::solinas::parallel::*;
use jolt_field::{CanonicalEncoding, Field, Ring};
use std::borrow::Cow;
use std::mem::size_of;
use std::sync::OnceLock;

const MAX_DENSE_DIGIT_CACHE_BYTES: usize = 512 * 1024 * 1024;

/// Minimum physical flat coefficient length.
///
/// Physical storage is zero-padded to `max(1 << num_vars, 1024)` so that for
/// every supported commitment ring dimension through `D=1024` the live view slice
/// `num_ring_elems(D) * D = div_ceil(2^num_vars, D) * D` is in bounds: when
/// `2^num_vars >= D` the slice is exactly `2^num_vars` coefficients, and when
/// `2^num_vars < D` it is `D <= 1024` coefficients. The old per-`D` storage
/// zero-padded the tail of the last ring element; the physical zero padding
/// reproduces those coefficients exactly.
const MIN_FLAT_COEFF_LEN: usize =
    SUPPORTED_COMMITMENT_RING_DIMS[SUPPORTED_COMMITMENT_RING_DIMS.len() - 1];

/// Elements per chunk of a parallel min/max reduction.
const MIN_MAX_CHUNK: usize = 1 << 15;

/// Inclusive `(min, max)` of a nonempty slice.
///
/// The range is split into fixed-size chunks and reduced with the conditional
/// parallelism helpers, so both the chunk scan and the reduction run on the
/// caller's pool when it exists and fall back to one thread otherwise. Each
/// chunk is scanned once by `slice_min_max`, a copied two-accumulator fold
/// that the compiler vectorises (slice `min`/`max` and an `Option`-state fold
/// do not).
fn chunked_min_max<T: Copy + Ord + Send + Sync>(values: &[T]) -> Option<(T, T)> {
    let first = *values.first()?;
    // A single chunk is not worth a parallel reduction; small groups take
    // this path and stay entirely serial.
    if values.len() <= MIN_MAX_CHUNK {
        return Some(slice_min_max(values, first));
    }
    let num_chunks = values.len().div_ceil(MIN_MAX_CHUNK);
    Some(cfg_fold_reduce!(
        0..num_chunks,
        || (first, first),
        |(low, high), chunk| {
            let start = chunk * MIN_MAX_CHUNK;
            let end = (start + MIN_MAX_CHUNK).min(values.len());
            let (part_low, part_high) = slice_min_max(&values[start..end], first);
            (low.min(part_low), high.max(part_high))
        },
        |a, b| (a.0.min(b.0), a.1.max(b.1))
    ))
}

/// One-pass min/max of `values`, seeded with `seed` (an element of the
/// whole range, so the result is unchanged).
#[inline]
fn slice_min_max<T: Copy + Ord>(values: &[T], seed: T) -> (T, T) {
    values
        .iter()
        .copied()
        .fold((seed, seed), |(low, high), x| (low.min(x), high.max(x)))
}

/// Backing store of a digit-plane cache.
#[derive(Debug, Clone, PartialEq, Eq)]
enum DenseDigitPlanes {
    /// Ring-major planes decomposed from the field coefficients.
    Owned(Vec<i8>),
    /// With a single digit the centered-`i8` mirror already is the ring-major
    /// planes, so the cache only records that the owning polynomial's mirror
    /// can serve the planes. It is resolved through that polynomial and never
    /// dereferenced on its own.
    Mirror,
}

/// D-free digit-plane cache: `num_digits` planes of `ring_d` bytes per ring
/// element, flattened in ring-major order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct DenseDigitCache {
    ring_d: usize,
    num_digits: usize,
    log_basis: u32,
    basis: SignedDigitBasis,
    planes: DenseDigitPlanes,
}

/// Dense polynomial: all ring coefficients materialized in memory.
///
/// Storage is D-free: coefficients are a flat field-element buffer, and the
/// ring dimension is a view selected at kernel entry (each ring-shaped method
/// takes it as a const generic).
#[derive(Debug)]
pub struct DensePoly<F: Field> {
    /// Actual multilinear variable count of the source witness.
    pub(super) num_vars: usize,
    /// Flat field coefficients in sequential block order (untagged compact
    /// [`RingVec`]; see [`MIN_FLAT_COEFF_LEN`] for the physical padding).
    coeffs: RingVec<F>,
    /// Flat centered-`i8` mirror of `coeffs` (same physical length and
    /// padding), present only when every live coefficient is small.
    pub(super) small_i8_coeffs: Option<Vec<i8>>,
    /// Flat signed-`i16` mirror of `coeffs` (same physical length and
    /// padding), present only when the producer supplied width-16 cells.
    pub(super) small_i16_coeffs: Option<Vec<i16>>,
    /// Inclusive `(min, max)` of the live mirrored values in their centered
    /// representation. Absent without a mirror or when the live length is zero.
    small_bounds: Option<(i64, i64)>,
    digit_cache: OnceLock<DenseDigitCache>,
}

impl<F: Field + Clone> Clone for DensePoly<F> {
    fn clone(&self) -> Self {
        Self {
            num_vars: self.num_vars,
            coeffs: self.coeffs.clone(),
            small_i8_coeffs: self.small_i8_coeffs.clone(),
            small_i16_coeffs: self.small_i16_coeffs.clone(),
            small_bounds: self.small_bounds,
            digit_cache: OnceLock::new(),
        }
    }
}

impl<F: Field + PartialEq> PartialEq for DensePoly<F> {
    fn eq(&self, other: &Self) -> bool {
        self.num_vars == other.num_vars
            && self.coeffs == other.coeffs
            && self.small_i8_coeffs == other.small_i8_coeffs
            && self.small_i16_coeffs == other.small_i16_coeffs
    }
}

impl<F: Field + Eq> Eq for DensePoly<F> {}

/// Reinterpret a flat coefficient slice as ring elements of dimension `D`.
///
/// This is the private sub-slice counterpart of [`RingVec::as_ring_slice`]:
/// callers slice the live prefix for their view dimension first, then
/// reinterpret after checking divisibility.
#[inline]
fn as_ring_view<F: Field, const D: usize>(flat: &[F]) -> &[CyclotomicRing<F, D>] {
    debug_assert!(D > 0);
    debug_assert!(flat.len().is_multiple_of(D));
    // SAFETY: `CyclotomicRing<F, D>` is `#[repr(transparent)]` over `[F; D]`,
    // and the length is a multiple of `D`.
    unsafe {
        std::slice::from_raw_parts(flat.as_ptr() as *const CyclotomicRing<F, D>, flat.len() / D)
    }
}

impl<F: Field> DensePoly<F> {
    /// Full physical (zero-padded) flat coefficient buffer.
    pub fn field_coeffs(&self) -> &[F] {
        self.coeffs.coeffs()
    }

    pub(crate) fn cached_digit_parts(
        &self,
        ring_dimension: usize,
        num_digits: usize,
        log_basis: u32,
    ) -> Option<&[i8]> {
        let cache = self.digit_cache.get()?;
        if cache.ring_d != ring_dimension
            || cache.num_digits != num_digits
            || cache.log_basis != log_basis
        {
            return None;
        }
        match &cache.planes {
            DenseDigitPlanes::Owned(planes) => Some(planes.as_slice()),
            DenseDigitPlanes::Mirror => {
                let live_len = self
                    .num_ring_elems_at(ring_dimension)
                    .checked_mul(num_digits)?
                    .checked_mul(ring_dimension)?;
                self.small_i8_coeffs.as_deref()?.get(..live_len)
            }
        }
    }

    /// Ring-element count viewed at dimension `ring_d`.
    #[inline]
    pub(super) fn num_ring_elems_at(&self, ring_d: usize) -> usize {
        (1usize << self.num_vars).div_ceil(ring_d)
    }

    /// Live view of the coefficients as ring elements of dimension `D`.
    ///
    /// The view covers `num_ring_elems_at(D)` ring elements read from the flat
    /// prefix `[..num_ring_elems_at(D) * D]`; the physical zero padding
    /// supplies the tail of the last ring exactly as the old per-`D` storage
    /// did.
    ///
    /// # Errors
    ///
    /// Returns an error if `D` is not a power of two or the view exceeds the
    /// physical buffer (unsupported ring dimension for this arity).
    pub fn ring_coeffs<const D: usize>(&self) -> Result<&[CyclotomicRing<F, D>], AkitaError> {
        if D == 0 || !D.is_power_of_two() {
            return Err(AkitaError::InvalidInput(format!(
                "ring degree D={D} is not a power of two"
            )));
        }
        let needed = self
            .num_ring_elems_at(D)
            .checked_mul(D)
            .ok_or_else(|| AkitaError::InvalidInput("dense ring view overflow".to_string()))?;
        let flat = self.field_coeffs();
        let live = flat.get(..needed).ok_or_else(|| {
            AkitaError::InvalidInput(format!(
                "dense ring view at D={D} needs {needed} coefficients but only {} are stored",
                flat.len()
            ))
        })?;
        Ok(as_ring_view::<F, D>(live))
    }

    /// Live small-i8 mirror viewed as per-ring coefficient planes at `D`.
    pub(super) fn small_i8_ring_coeffs<const D: usize>(&self) -> Option<&[[i8; D]]> {
        let flat = self.small_i8_coeffs.as_deref()?;
        let needed = self.num_ring_elems_at(D).checked_mul(D)?;
        let (chunks, remainder) = flat.get(..needed)?.as_chunks::<D>();
        debug_assert!(remainder.is_empty());
        Some(chunks)
    }

    /// Live signed-i16 mirror viewed as per-ring coefficient planes at `D`.
    pub(super) fn small_i16_ring_coeffs<const D: usize>(&self) -> Option<&[[i16; D]]> {
        let flat = self.small_i16_coeffs.as_deref()?;
        let needed = self.num_ring_elems_at(D).checked_mul(D)?;
        let (chunks, remainder) = flat.get(..needed)?.as_chunks::<D>();
        debug_assert!(remainder.is_empty());
        Some(chunks)
    }
}

impl<F: Field + CanonicalEncoding> DensePoly<F> {
    /// Pack field-element evaluations into flat dense storage.
    ///
    /// At dimension `D` the first `α = log₂(D)` variables
    /// become coefficient slots within each ring element; the remaining
    /// variables index ring elements.
    ///
    /// # Errors
    ///
    /// Returns an error if `2^num_vars` does not fit `usize` or
    /// `evals.len() != 2^num_vars`.
    pub fn from_field_evals<'a>(
        num_vars: usize,
        evals: impl Into<Cow<'a, [F]>>,
    ) -> Result<Self, AkitaError>
    where
        F: 'a,
    {
        let evals = evals.into();
        let expected_len = checked::pow2(num_vars)
            .ok_or_else(|| AkitaError::InvalidInput(format!("2^{num_vars} does not fit usize")))?;
        if evals.len() != expected_len {
            return Err(AkitaError::InvalidSize {
                expected: expected_len,
                actual: evals.len(),
            });
        }

        let physical_len = expected_len.max(MIN_FLAT_COEFF_LEN);

        // Padding zeros are centered-0 (trivially small-i8), so a poly whose
        // live coefficients are all small stays all-small — identical to the
        // old per-ring check over the zero-padded last ring.
        let q = (-F::one())
            .to_u128_checked()
            .ok_or_else(|| AkitaError::InvalidInput("field modulus does not fit u128".into()))?
            + 1;
        let half_q = q / 2;
        let mut small_i8_coeffs = Vec::with_capacity(physical_len);
        let mut all_small_i8 = true;
        let mut small_bounds: Option<(i64, i64)> = None;
        for coeff in evals.iter() {
            if let Some(centered) = try_centered_i8(*coeff, q, half_q) {
                let value = i64::from(centered);
                small_bounds = Some(match small_bounds {
                    None => (value, value),
                    Some((low, high)) => (low.min(value), high.max(value)),
                });
                small_i8_coeffs.push(centered);
            } else {
                all_small_i8 = false;
                break;
            }
        }
        if all_small_i8 {
            small_i8_coeffs.resize(physical_len, 0);
        }

        // Reuse an owned evaluation vector. Borrowed inputs pay the same
        // single copy as before, while profile and application builders can
        // transfer large buffers without doubling their resident memory.
        let mut coeffs = evals.into_owned();
        coeffs.resize(physical_len, F::zero());

        Ok(Self {
            num_vars,
            coeffs: RingVec::from_coeffs(coeffs),
            small_i8_coeffs: all_small_i8.then_some(small_i8_coeffs),
            small_i16_coeffs: None,
            small_bounds: all_small_i8.then_some(small_bounds).flatten(),
            digit_cache: OnceLock::new(),
        })
    }

    /// Flatten an existing vector of ring elements into dense storage.
    ///
    /// The total coefficient count must be a positive power of two. All
    /// supplied coefficients belong to the live polynomial; storage padding
    /// for larger ring views does not change its variable count.
    ///
    /// # Errors
    ///
    /// Returns an error if `coeffs.len() * D` overflows `usize`, is zero, or
    /// is not a power of two.
    pub fn from_ring_coeffs<const D: usize>(
        coeffs: Vec<CyclotomicRing<F, D>>,
    ) -> Result<Self, AkitaError> {
        let total = checked::product([coeffs.len(), D])
            .ok_or_else(|| AkitaError::InvalidInput("ring elems * D overflow".to_string()))?;
        if !total.is_power_of_two() {
            return Err(AkitaError::InvalidInput(format!(
                "dense coefficient count {total} must be a positive power of two"
            )));
        }
        let physical_len = total.max(MIN_FLAT_COEFF_LEN);

        let mut small_bounds: Option<(i64, i64)> = None;
        let small_i8_coeffs = try_small_i8_cache_from_ring_coeffs(&coeffs).map(|planes| {
            let mut flat = Vec::with_capacity(physical_len);
            for plane in &planes {
                for &value in plane {
                    let value = i64::from(value);
                    small_bounds = Some(match small_bounds {
                        None => (value, value),
                        Some((low, high)) => (low.min(value), high.max(value)),
                    });
                }
                flat.extend_from_slice(plane);
            }
            flat.resize(physical_len, 0i8);
            flat
        });
        let mut flat = Vec::with_capacity(physical_len);
        for ring in &coeffs {
            flat.extend_from_slice(ring.coefficients());
        }
        flat.resize(physical_len, F::zero());

        Ok(Self {
            num_vars: total.trailing_zeros() as usize,
            coeffs: RingVec::from_coeffs(flat),
            small_i8_coeffs,
            small_i16_coeffs: None,
            small_bounds,
            digit_cache: OnceLock::new(),
        })
    }

    pub(super) fn digit_planes_for<const D: usize>(
        &self,
        num_digits: usize,
        log_basis: u32,
    ) -> Option<(&[[i8; D]], SignedDigitBasis)> {
        if let Some(cache) = self.digit_cache.get() {
            // A cache built at another dimension is not reused: returning
            // `None` falls back to the uncached path, exactly like a
            // too-large cache does. Under uniform-D this never triggers.
            if cache.ring_d != D || cache.num_digits != num_digits || cache.log_basis != log_basis {
                return None;
            }
            return self.cached_digit_planes::<D>(cache);
        }

        let basis = SignedDigitBasis::new(log_basis).ok()?;
        if basis.kernel() != akita_params::SignedDigitKernel::I8 {
            return None;
        }
        let num_rings = self.num_ring_elems_at(D);
        let cache_bytes = num_rings
            .checked_mul(num_digits)?
            .checked_mul(size_of::<[i8; D]>())?;
        if cache_bytes > MAX_DENSE_DIGIT_CACHE_BYTES {
            return None;
        }

        let _span = tracing::info_span!(
            "dense_digit_cache_build",
            cache_bytes,
            num_rings,
            num_digits,
            ring_dimension = D,
        )
        .entered();
        // One balanced digit in the basis range is the centered cell itself, so
        // the mirror already holds the ring-major planes and no decomposition
        // runs. Any other shape reads the field coefficients.
        let mirrored = (num_digits == 1)
            .then(|| self.small_bounds)
            .flatten()
            .and_then(|(low, high)| {
                let half = 1_i64.checked_shl(log_basis.checked_sub(1)?)?;
                (low >= -half && high < half).then_some(())
            })
            .and_then(|()| self.small_i8_coeffs.as_deref())
            .and_then(|mirror| mirror.get(..num_rings * D).map(|_| ()));
        let planes = match mirrored {
            Some(()) => DenseDigitPlanes::Mirror,
            None => {
                let params = BalancedDecomposePow2Params::new(num_digits, log_basis);
                let mut planes = vec![0i8; num_rings * num_digits * D];
                let rings = self.ring_coeffs::<D>().ok()?;
                cfg_chunks_mut!(planes, num_digits * D)
                    .zip(cfg_iter!(rings))
                    .for_each(|(dst, ring)| {
                        let (dst_planes, remainder) = dst.as_chunks_mut::<D>();
                        debug_assert!(remainder.is_empty());
                        ring.balanced_decompose_pow2_i8_into_with_params(dst_planes, &params);
                    });
                DenseDigitPlanes::Owned(planes)
            }
        };
        let _ = self.digit_cache.set(DenseDigitCache {
            ring_d: D,
            num_digits,
            log_basis,
            basis,
            planes,
        });
        let cache = self.digit_cache.get()?;
        if cache.ring_d != D || cache.num_digits != num_digits || cache.log_basis != log_basis {
            return None;
        }
        self.cached_digit_planes::<D>(cache)
    }

    /// Resolve an installed cache into ring-major digit planes.
    ///
    /// A mirror-backed cache yields the live prefix of the centered-`i8`
    /// mirror; an owned cache yields its stored planes. Both are exactly
    /// `num_ring_elems_at(D) * num_digits` chunks of `D` bytes.
    fn cached_digit_planes<'a, const D: usize>(
        &'a self,
        cache: &'a DenseDigitCache,
    ) -> Option<(&'a [[i8; D]], SignedDigitBasis)> {
        match &cache.planes {
            DenseDigitPlanes::Owned(planes) => {
                let (chunks, remainder) = planes.as_chunks::<D>();
                debug_assert!(remainder.is_empty());
                Some((chunks, cache.basis))
            }
            DenseDigitPlanes::Mirror => self
                .small_i8_ring_coeffs::<D>()
                .map(|planes| (planes, cache.basis)),
        }
    }

    /// Build the digit-plane cache now for the given ring dimension and
    /// decomposition, and report whether it is present afterwards.
    ///
    /// The commit and the opening both use the group's inner decomposition.
    /// Building it while the producer still holds the fresh mirror turns the
    /// first kernel read of the field plane into a read of the ring-major digit
    /// planes. A shape without an implementable cache reports `false` and stays
    /// on the lazy path.
    ///
    /// # Errors
    ///
    /// Returns an error for a ring dimension outside the inner dispatch table.
    pub fn prime_digit_cache(
        &self,
        ring_dimension: usize,
        num_digits: usize,
        log_basis: u32,
    ) -> Result<bool, AkitaError> {
        akita_params::dispatch_for_field!(
            akita_params::ProtocolDispatchSlot::Role(akita_params::RingRole::Inner),
            F,
            ring_dimension,
            |D| { Ok(self.digit_planes_for::<D>(num_digits, log_basis).is_some()) }
        )
    }
}

impl DensePoly<jolt_field::Prime64Offset59> {
    /// Construct the fixed-prime dense source from centered bytes without rediscovering
    /// the small-coefficient mirror. Both representations remain available to opening kernels.
    pub fn from_small_i8(num_vars: usize, cells: Vec<i8>) -> Result<Self, AkitaError> {
        let expected = checked::pow2(num_vars)
            .ok_or_else(|| AkitaError::InvalidInput(format!("2^{num_vars} does not fit usize")))?;
        if cells.len() != expected {
            return Err(AkitaError::InvalidSize {
                expected,
                actual: cells.len(),
            });
        }
        Self::from_small_i8_mirror(num_vars, cells, expected)
    }

    /// Construct the fixed-prime dense source from a logical centered-byte table.
    ///
    /// The mirror is allocated once at the physical length and filled in chunks
    /// on the caller's pool, so both the copy and the page faults happen in the
    /// writers. A table of at most one min/max chunk, or a physical length
    /// within one chunk, stays on the serial path. Tables shorter than
    /// `2^num_vars` are zero-extended, as [`Self::from_small_i8`] expects.
    pub fn from_small_i8_cells(num_vars: usize, cells: &[i8]) -> Result<Self, AkitaError> {
        let expected = checked::pow2(num_vars)
            .ok_or_else(|| AkitaError::InvalidInput(format!("2^{num_vars} does not fit usize")))?;
        if cells.len() > expected {
            return Err(AkitaError::InvalidSize {
                expected,
                actual: cells.len(),
            });
        }
        let physical_len = expected.max(MIN_FLAT_COEFF_LEN);
        let mut mirror = Vec::with_capacity(physical_len);
        {
            let spare = &mut mirror.spare_capacity_mut()[..physical_len];
            if physical_len <= MIN_MAX_CHUNK {
                write_i8_chunk(cells, spare);
            } else {
                cfg_chunks_mut!(spare, MIN_MAX_CHUNK)
                    .enumerate()
                    .for_each(|(chunk, part)| {
                        let start = chunk * MIN_MAX_CHUNK;
                        write_i8_chunk(cells.get(start..).unwrap_or_default(), part);
                    });
            }
        }
        // SAFETY: `write_i8_chunk` initializes every slot it is given, and the
        // disjoint chunks above cover the `physical_len` spare slots, which fit
        // in `mirror`'s capacity by construction.
        unsafe { mirror.set_len(physical_len) };
        Self::from_small_i8_mirror(num_vars, mirror, expected)
    }

    /// Build the polynomial of an already padded centered-byte mirror.
    ///
    /// `mirror[..expected]` is the logical table, including the zero extension
    /// to `2^num_vars`; the bounds are taken over exactly that prefix, as the
    /// caller of [`Self::from_small_i8`] specified before padding.
    fn from_small_i8_mirror(
        num_vars: usize,
        mut mirror: Vec<i8>,
        expected: usize,
    ) -> Result<Self, AkitaError> {
        debug_assert!(mirror.len() >= expected);
        let small_bounds = chunked_min_max(&mirror[..expected])
            .map(|(low, high)| (i64::from(low), i64::from(high)));
        mirror.resize(expected.max(MIN_FLAT_COEFF_LEN), 0);
        let coeffs = super::small::expand(&mirror);
        Ok(Self {
            num_vars,
            coeffs: RingVec::from_coeffs(coeffs),
            small_i8_coeffs: Some(mirror),
            small_i16_coeffs: None,
            small_bounds,
            digit_cache: OnceLock::new(),
        })
    }

    /// Construct the fixed-prime dense source of a width-16 table from its
    /// already-encoded field evaluations and attach the signed-16 mirror.
    ///
    /// The caller knows the cells are wide (at least one lies outside the
    /// centered `i8` range), so the centered-`i8` mirror of
    /// [`Self::from_field_evals`] is neither probed nor built. The i16 mirror is
    /// written, and its bounds taken, by [`Self::attach_small_i16`].
    pub fn from_wide_field_evals(
        num_vars: usize,
        mut evals: Vec<jolt_field::Prime64Offset59>,
        cells: &[i32],
    ) -> Result<Self, AkitaError> {
        let expected = checked::pow2(num_vars)
            .ok_or_else(|| AkitaError::InvalidInput(format!("2^{num_vars} does not fit usize")))?;
        if evals.len() != expected {
            return Err(AkitaError::InvalidSize {
                expected,
                actual: evals.len(),
            });
        }
        // No-op for the production shapes, where `expected >= MIN_FLAT_COEFF_LEN`.
        evals.resize(
            expected.max(MIN_FLAT_COEFF_LEN),
            jolt_field::Prime64Offset59::from_u64(0),
        );
        // The mirror below commits a different table than the field plane the packing
        // pass and the evaluation read, so a caller that builds them from different
        // cells would commit one polynomial and open another.
        #[cfg(debug_assertions)]
        for (i, &cell) in cells.iter().enumerate() {
            debug_assert_eq!(
                evals[i],
                jolt_field::Prime64Offset59::from_i64(i64::from(cell)),
                "field plane and signed-16 mirror disagree at cell {i}"
            );
        }
        let polynomial = Self {
            num_vars,
            coeffs: RingVec::from_coeffs(evals),
            small_i8_coeffs: None,
            small_i16_coeffs: None,
            small_bounds: None,
            digit_cache: OnceLock::new(),
        };
        polynomial.attach_small_i16(cells)
    }

    /// Attach the signed-16 mirror of a width-16 source.
    ///
    /// `cells` are the committed table's logical cells, already materialized as
    /// field coefficients by the caller. The mirror has the same physical
    /// padding as those coefficients and is the single-digit decomposition of
    /// every live cell.
    ///
    /// # Errors
    ///
    /// Returns an error if `cells` is longer than `2^num_vars` or holds a value
    /// that does not fit `i16`; a shorter table is zero-extended, as the
    /// physical coefficient table is.
    pub fn attach_small_i16(mut self, cells: &[i32]) -> Result<Self, AkitaError> {
        let num_vars = self.num_vars;
        let expected = checked::pow2(num_vars)
            .ok_or_else(|| AkitaError::InvalidInput(format!("2^{num_vars} does not fit usize")))?;
        if cells.len() > expected {
            return Err(AkitaError::InvalidSize {
                expected,
                actual: cells.len(),
            });
        }
        let physical_len = expected.max(MIN_FLAT_COEFF_LEN);
        // One pass over the live cells writes the mirror and records the range and
        // the first out-of-range cell. The zero padding beyond `cells.len()` is
        // written without contributing to the bounds, exactly as before.
        let mut mirror = Vec::with_capacity(physical_len);
        let partials = {
            let spare = &mut mirror.spare_capacity_mut()[..physical_len];
            if physical_len <= MIN_MAX_CHUNK {
                vec![write_i16_chunk(cells, spare)]
            } else {
                cfg_chunks_mut!(spare, MIN_MAX_CHUNK)
                    .enumerate()
                    .map(|(chunk, part)| {
                        let start = chunk * MIN_MAX_CHUNK;
                        write_i16_chunk(cells.get(start..).unwrap_or_default(), part)
                    })
                    .collect::<Vec<_>>()
            }
        };
        let mut bounds: Option<(i32, i32)> = None;
        let mut first_bad: Option<i32> = None;
        for (low, high, bad) in partials {
            bounds = Some(match bounds {
                None => (low, high),
                Some((low_before, high_before)) => (low_before.min(low), high_before.max(high)),
            });
            if first_bad.is_none() {
                first_bad = bad;
            }
        }
        if let Some(cell) = first_bad {
            return Err(AkitaError::InvalidInput(format!(
                "width-16 source cell {cell} does not fit a signed 16-bit digit"
            )));
        }
        let bounds = bounds
            .filter(|_| !cells.is_empty())
            .map(|(low, high)| (i64::from(low), i64::from(high)));
        // SAFETY: `write_i16_chunk` initializes every slot it is given, and the
        // disjoint chunks above cover the `physical_len` spare slots, which fit
        // in `mirror`'s capacity by construction.
        unsafe { mirror.set_len(physical_len) };
        self.small_i16_coeffs = Some(mirror);
        self.small_bounds = bounds;
        Ok(self)
    }
}

/// Write `cells` into `part` as signed 16-bit values, zero-filling beyond the
/// cells, and return the cell range and the first cell that does not fit `i16`.
///
/// `part` starts at the index `cells` was sliced from, so the live prefix is
/// `cells.len().min(part.len())`; the range covers only that prefix.
fn write_i16_chunk(
    cells: &[i32],
    part: &mut [std::mem::MaybeUninit<i16>],
) -> (i32, i32, Option<i32>) {
    let live = cells.len().min(part.len());
    let (mut low, mut high) = (i32::MAX, i32::MIN);
    let mut first_bad = None;
    for (offset, slot) in part.iter_mut().enumerate() {
        if offset < live {
            let cell = cells[offset];
            low = low.min(cell);
            high = high.max(cell);
            if first_bad.is_none() && i16::try_from(cell).is_err() {
                first_bad = Some(cell);
            }
            slot.write(cell as i16);
        } else {
            slot.write(0);
        }
    }
    (low, high, first_bad)
}

/// Write `cells` into `part`, zero-filling beyond the cells.
///
/// `part` starts at the index `cells` was sliced from, so the live prefix is
/// `cells.len().min(part.len())`.
fn write_i8_chunk(cells: &[i8], part: &mut [std::mem::MaybeUninit<i8>]) {
    for (index, slot) in part.iter_mut().enumerate() {
        slot.write(cells.get(index).copied().unwrap_or(0));
    }
}

impl<F: Field> DensePoly<F> {
    /// Flat signed-16 mirror of the coefficients, when the producer kept one.
    #[must_use]
    pub fn small_i16_mirror(&self) -> Option<&[i16]> {
        self.small_i16_coeffs.as_deref()
    }

    /// Inclusive range of the live mirrored values in their centered
    /// representation, when the producer kept a mirror.
    #[must_use]
    pub fn small_mirror_bounds(&self) -> Option<(i64, i64)> {
        self.small_bounds
    }
}
