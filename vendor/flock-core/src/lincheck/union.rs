//! Union-column lincheck for multi-table proofs.
//!
//! Generalizes the single-table lincheck (`super`) from one slot to the `T`
//! aligned slot subcubes of a registry instance. One product sumcheck over
//! the **union column domain** — the
//! seven in-word coordinates plus every chunk-and-prefix coordinate, i.e.
//! everything except the shared row block `R` — proves the α-batched claim
//! `α·â(r) + b̂(r)` left by the union zerocheck.
//!
//! ## The identity being proven (doc §6.2–6.4, uniform capacity)
//!
//! ```text
//!   α·â(r) + b̂(r) = Σ_t w_t(r) · Σ_{c'} (α·comb_t^A + comb_t^B)(c') · y_t(c')
//! ```
//!
//! - `w_t(r) = eq(r[m_t:M], p_t)` — the slot's prefix-eq weight, a *fixed
//!   public scalar* determined by `r` and the registry (no cross-slot
//!   batching randomness; the single α batches only the A- and B-claims);
//! - `comb_t` — the type's α-batched comb via its [`LincheckCircuit`] with
//!   the quirky table of the slot's own column coordinates of `r` (the
//!   slot's chunk bits are the LOW union chunk coordinates; the frozen
//!   prefix is the high ones) — exactly today's single-table comb;
//! - `y_t(c') = ẑ_t(r|_R, c')` — slot `t`'s witness folded over the SHARED
//!   row coordinates: one standard eq table serves every slot, by uniform
//!   capacity. Dummy rows are honest zeros, so folding the full capacity
//!   equals folding the declared rows.
//!
//! The prover materializes `Comb(c) = w_t(r)·comb_t(c')` and `g(c) = y_t(c')`
//! on slot `t`'s aligned column subcube — zero on gaps — as dense
//! union-column-domain vectors and runs the existing sumcheck core
//! (`column_sumcheck_prove`) over them. A single-slot registry reproduces
//! today's two vectors verbatim (`w_1 = 1`, offset `0`), so the `T = 1`
//! union lincheck is **byte-identical** to the single-table one (the M2
//! differential property). Dense union-column vectors are the simple, correct
//! choice at current scales; the per-slot factorized / scalar-tail
//! iteration (collapsed slots contribute `O(1)` per remaining round, doc
//! §"One sumcheck over the union column domain") is a later performance
//! item.
//!
//! The verifier replays the rounds without materializing anything
//! union-sized: at the end it evaluates the Comb-hat skip-block collapse by
//! the closed form — per-type comb MLEs at the bound point, times the
//! slot's r-side prefix weight and the bound-point subcube prefix-eq factor
//! ([`union_comb_partial`]).
//!
//! Output: a SINGLE witness claim `ẑ(c*-columns, r|_R-rows)`, structurally
//! identical to today's ([`LincheckClaim`] with `r_inner_rest` spanning the
//! union column coordinates); `UnionInstance::ab_claim_point` converts it to
//! the address-ordered `ZClaim` point.
//!
//! ## Constant-wire pins under counts
//!
//! Pins generalize per type: β_t is sampled after α, in slot order, and
//! added (unscaled) at the pin's union column. Declared rows carry the pin
//! at `1`; dummy rows are all-zero *including the pin* (doc Remark "Dummy
//! rows are complete" — the pin at 0 is what marks a row as dummy), so the
//! honest fold of the pin column is `Σ_{row<n_t} eq(r|_R, row)` and the
//! verifier's target gains `β_t · eq_prefix_sum(r|_R, n_t)` — the counts
//! bind inside the lincheck. At full utilization the sum is exactly `1`
//! (the eq-sum identity), reproducing today's `target += β` byte for byte.

use std::{env::var, time::Instant};
use crate::challenger::CanonicalPow;
use rayon::prelude::{IntoParallelRefIterator, ParallelIterator};

use crate::{
    challenger::Challenger,
    field::F128,
    lincheck::{
        LincheckCircuit, LincheckClaim, LincheckError, LincheckGrinding, LincheckProof,
        QuirkyPoint, SkipPoint, build_eq_table, build_quirky_eq_table_from_weights,
        column_sumcheck_prove, inner_product, partial_fold_packed_z_rows_best,
    },
    matrix_fold::{MatrixClaim, Weight},
    schedule::Registry,
    union::UnionInstance,
    zerocheck::K_SKIP,
};

/// One slot's lincheck inputs, in slot order — the union counterpart of the
/// `(z_packed_lincheck, circuit)` pair the single-table lincheck consumes.
pub struct UnionLincheckSlot<'a> {
    /// The slot's lincheck stripe copy of `z` — `pack_z_lincheck` layout over
    /// the slot's own `(m_slot, k_log)`, i.e. the batch-major witness
    /// drivers' fourth output. Length `2^{m_slot−3}` bytes.
    pub z_lincheck: &'a [u8],
    /// The slot's lincheck circuit; its `const_pin_col` must match the
    /// registry type's `const_pin`.
    pub circuit: &'a dyn LincheckCircuit,
}

/// `Π_j eq(coords[j], bit_j(bits))` — the eq factor freezing `coords` to the
/// Boolean pattern `bits` (LSB-first). The slot prefix weights `w_t(r)` and
/// the closed form's bound-point subcube factors are both instances.
pub fn eq_prefix_weight(coords: &[F128], bits: usize) -> F128 {
    debug_assert!(coords.len() >= usize::BITS as usize - bits.leading_zeros() as usize);
    let mut acc = F128::ONE;
    for (j, &x) in coords.iter().enumerate() {
        acc *= if (bits >> j) & 1 == 1 {
            x
        } else {
            F128::ONE + x
        };
    }
    acc
}

/// `Σ_{i<n} eq(point, i)` in `O(|point|)` field ops — the verifier's
/// const-pin expected value at declared count `n`. `n = 2^{|point|}` returns
/// exactly `F128::ONE` (the eq-sum identity), with no arithmetic, so the
/// full-utilization target matches today's `target += β` byte for byte.
pub fn eq_prefix_sum(point: &[F128], n: usize) -> F128 {
    let d = point.len();
    assert!(n <= 1usize << d, "prefix length {n} exceeds domain 2^{d}");
    if n == 1usize << d {
        return F128::ONE;
    }
    // Indices i < n partition by the highest bit where i drops below n: for
    // each j with bit_j(n) = 1, the block {i: i matches n above j, bit_j(i) =
    // 0, free below} contributes Π_{i>j} eq(point[i], bit_i(n)) · eq(point[j],
    // 0) · 1 (the free bits sum to 1 by the eq-sum identity).
    let mut acc = F128::ZERO;
    let mut high = F128::ONE; // running Π_{i>j} eq(point[i], bit_i(n))
    for j in (0..d).rev() {
        if (n >> j) & 1 == 1 {
            acc += high * (F128::ONE + point[j]);
            high *= point[j];
        } else {
            high *= F128::ONE + point[j];
        }
    }
    acc
}

/// Closed-form skip-block collapse of the union Comb-hat at the bound column
/// point (doc §"One sumcheck over the union column domain"): with `combs[t]`
/// the type's length-`2^{κ_t}` comb vector — already carrying the r-side
/// prefix weight `w_t(r)` and any const-pin β terms — and `rr` the LSB-first
/// bound challenges of the multilinear column coordinates (`rr[i]` binds
/// coordinate `k_skip + i`), returns the length-`2^{k_skip}` vector
///
/// ```text
///   out[s] = Σ_t eq(rr[κ_t−k_skip..], p_t)
///              · Σ_j eq-tensor(rr[..κ_t−k_skip])[j] · combs[t][s + (j << k_skip)]
/// ```
///
/// — each slot contributes its own comb MLE at the bound point times the
/// subcube prefix-eq factor "the bound point addresses slot t"; nothing
/// union-sized is materialized. Cost `O(Σ_t 2^{κ_t})`, the same as the
/// per-type comb construction itself.
pub fn union_comb_partial(
    registry: &Registry,
    combs: &[Vec<F128>],
    rr: &[F128],
    k_skip: usize,
) -> Vec<F128> {
    let nb = registry.num_boolean();
    assert_eq!(combs.len(), nb);
    assert_eq!(rr.len(), registry.m_bool() - registry.nu() - k_skip);
    let n_skip = 1usize << k_skip;
    let mut out = vec![F128::ZERO; n_skip];
    for ((ty, slot), comb) in registry
        .boolean_types()
        .iter()
        .zip(registry.slots())
        .zip(combs)
    {
        assert_eq!(comb.len(), 1usize << ty.k_log);
        let inner = ty.k_log - k_skip;
        let p_t = eq_prefix_weight(&rr[inner..], slot.prefix);
        let eq_rho = build_eq_table(&rr[..inner]);
        for (j, &e) in eq_rho.iter().enumerate() {
            let s = p_t * e;
            let block = &comb[(j << k_skip)..((j + 1) << k_skip)];
            for (o, &c) in out.iter_mut().zip(block) {
                *o += s * c;
            }
        }
    }
    out
}

/// Prove the union-column lincheck and capture the pre-sumcheck union fold
/// `g` (the per-slot `y_t` vectors at their aligned column offsets, length
/// `2^{M−ν}`) for downstream reuse — the union counterpart of
/// [`super::prove_padded_capture_z_vec`]. `x_ab` is the union semantic
/// quirky point from `UnionInstance::x_ab_from_multilinear_values`; `k_skip` is the
/// BatchMajor-fixed [`K_SKIP`]. On a single-type registry this computes
/// exactly the single-table lincheck — same vectors, same rounds, same
/// bytes.
pub fn prove_union_capture_z_vec<Ch: Challenger>(
    union: &UnionInstance<'_>,
    slots: &[UnionLincheckSlot<'_>],
    x_ab: &QuirkyPoint,
    challenger: &mut Ch,
) -> (LincheckProof, LincheckClaim, Vec<F128>) {
    prove_union_capture_z_vec_with_grinding(
        union,
        slots,
        x_ab,
        LincheckGrinding::disabled(),
        challenger,
    )
}

/// [`prove_union_capture_z_vec`] with an explicit Fiat--Shamir grinding
/// policy.
/// The union's column bit-bank at the row point `x_outer`: entry `k·128 + b`
/// is `Σ_r eq(x_outer, r)·bit_b(z[k·2^ν + r])` for chunk column `k` of the
/// boolean region (dead bits and rows past a slot's count contribute zero).
/// The union lincheck's `z_vec` is this fold at the lincheck's own row point;
/// the merged open's block-first transport (`pcs::open_batch_merged`) also
/// wants it at the C claim's row point, which is why it is a public entry.
pub fn union_bitbank_fold(
    union: &UnionInstance,
    slots: &[UnionLincheckSlot<'_>],
    x_outer: &[F128],
) -> Vec<F128> {
    let registry = union.registry();
    let nu = union.n_log();
    let col_vars = union.m_bool() - nu;
    let single = registry.num_boolean() == 1 && registry.slots()[0].m_slot == registry.m_bool();
    let eq_x_outer = build_eq_table(x_outer);
    let mut z_vec: Vec<F128> = if single {
        Vec::new()
    } else {
        vec![F128::ZERO; 1usize << col_vars]
    };
    for (((ty, slot), slot_in), &n_t) in registry
        .boolean_types()
        .iter()
        .zip(registry.slots())
        .zip(slots)
        .zip(union.counts())
    {
        let y_t = partial_fold_packed_z_rows_best(
            slot_in.z_lincheck,
            nu + ty.k_log,
            ty.k_log,
            ty.useful_bits,
            &eq_x_outer,
            n_t,
        );
        if single {
            z_vec = y_t;
        } else {
            let off = slot.prefix << ty.k_log;
            z_vec[off..off + y_t.len()].copy_from_slice(&y_t);
        }
    }
    z_vec
}

/// Bytes of subset-sum tables per row tile of [`union_bitbank_fold_multi`]: half of a
/// 512 KB L2, so that the tables of two SMT siblings share one core's L2.
const BANK_TABLE_BYTES: usize = 256 << 10;
/// Fewest and most row groups of eight per row tile (the transposed bytes, `128 B` per
/// group, stay in L1).
const BANK_ROW_GROUPS_MIN: usize = 8;
const BANK_ROW_GROUPS_MAX: usize = 64;
/// Word columns per task of [`union_bitbank_fold_multi`]: the task's output is
/// `points × 32 × 128` F128 (64 KB per point), touched once per row tile.
const BANK_COL_TILE: usize = 32;

/// Raw output pointers of [`union_bitbank_fold_multi`]; every task writes a disjoint
/// column range of each output.
struct BankOut {
    ptrs: Vec<*mut F128>,
    /// Length of each output in F128.
    len: usize,
}
// SAFETY: tasks write disjoint ranges (see the caller); the vectors outlive the scope.
unsafe impl Sync for BankOut {}
impl BankOut {
    fn ptrs(&self) -> &[*mut F128] {
        &self.ptrs
    }
    fn len(&self) -> usize {
        self.len
    }
}

/// One 64-byte-aligned line of four F128: the unit of the interleaved subset-sum tables.
#[derive(Clone, Copy)]
#[repr(C, align(64))]
struct Line([F128; 4]);

/// Subset-sum tables of one row tile with the row points interleaved: for group `g` and
/// byte value `s`, the entry `entry(g, s)` of [`Self::stride`] consecutive F128 holds the
/// sum for point `p` at position `p`. Four points fill one 64-byte line, so one byte
/// selects one line for four points.
struct BankTables {
    lines: Vec<Line>,
    /// F128 per entry: the point count for one or two points, else the count rounded up
    /// to a multiple of four (padding lanes stay zero).
    stride: usize,
}

impl BankTables {
    /// F128 per entry for `np` points.
    fn stride_for(np: usize) -> usize {
        if np <= 2 { np } else { np.next_multiple_of(4) }
    }

    fn new(np: usize, groups: usize) -> Self {
        let stride = Self::stride_for(np);
        let f128s = groups * 256 * stride;
        Self {
            lines: vec![Line([F128::ZERO; 4]); f128s.div_ceil(4)],
            stride,
        }
    }

    fn as_slice(&self) -> &[F128] {
        // SAFETY: Line is repr(C) over [F128; 4], so the lines are a contiguous F128 array.
        unsafe { std::slice::from_raw_parts(self.lines.as_ptr().cast(), self.lines.len() * 4) }
    }

    fn as_mut_slice(&mut self) -> &mut [F128] {
        // SAFETY: as in `as_slice`.
        unsafe {
            std::slice::from_raw_parts_mut(self.lines.as_mut_ptr().cast(), self.lines.len() * 4)
        }
    }
}

/// The column bit-banks of the union buffer `z` (BatchMajor, `z[(c << ν) | j]` per slot)
/// at several row points in ONE pass over `z`: output `p` is, for every boolean slot `t`
/// at its aligned column offset,
/// `out[p][128 c + b] = Σ_{j < n_t} eq(points[p], j) · bit_b(z_t[(c << ν) | j])`, zero
/// for bits `≥ useful_bits` and for the gaps between slots — exactly
/// [`union_bitbank_fold`] at each point, without the lincheck stripe.
///
/// Per task (a tile of [`BANK_COL_TILE`] word columns of one slot) and per row tile of
/// groups of eight rows, the 256-entry subset-sum table of each group's eight eq weights
/// is built for all points together (dead rows past `n_t` get weight zero, so their bits
/// contribute nothing), with the points interleaved ([`BankTables`]); then per column the
/// eight-word runs `z[(c << ν) + 8g ..][..8]` are bit-transposed (byte `b` = bit `b` of
/// the eight words) and each byte selects one table entry holding every point, XORed into
/// a register accumulator over the row tile. The row tile holds as many groups as keep the
/// tables within [`BANK_TABLE_BYTES`]. XOR is exact and order-free, so the outputs do not
/// depend on the tiling.
pub fn union_bitbank_fold_multi(
    union: &UnionInstance,
    z: &[F128],
    points: &[&[F128]],
) -> Vec<Vec<F128>> {
    let registry = union.registry();
    let nu = union.n_log();
    assert!(nu >= 3, "bit-bank fold needs at least 8 rows");
    let col_vars = union.m_bool() - nu;
    let np = points.len();
    let eqs: Vec<Vec<F128>> = points
        .iter()
        .map(|p| {
            assert_eq!(p.len(), nu, "bit-bank row point length");
            build_eq_table(p)
        })
        .collect();
    let mut outs: Vec<Vec<F128>> = (0..np)
        .map(|_| crate::alloc_zeroed_vec(1usize << col_vars))
        .collect();
    if np == 0 {
        return outs;
    }
    // (z word offset of column c0, output column of c0, columns, live bits from c0, rows)
    let mut tasks: Vec<(usize, usize, usize, usize, usize)> = Vec::new();
    for ((ty, slot), &n_t) in registry
        .boolean_types()
        .iter()
        .zip(registry.slots())
        .zip(union.counts())
    {
        assert!(ty.k_log >= 7 && slot.m_slot == nu + ty.k_log, "slot geometry");
        assert!(n_t <= 1usize << nu, "count exceeds the row capacity");
        let words = 1usize << (slot.m_slot - 7);
        let base = slot.offset >> 7;
        assert!(base + words <= z.len(), "union buffer too short for slot");
        let cols = ty.useful_bits.div_ceil(128);
        let out_col = slot.prefix << (ty.k_log - 7);
        for c0 in (0..cols).step_by(BANK_COL_TILE) {
            tasks.push((
                base + (c0 << nu),
                out_col + c0,
                BANK_COL_TILE.min(cols - c0),
                ty.useful_bits - 128 * c0,
                n_t,
            ));
        }
    }
    let ptrs = BankOut {
        ptrs: outs.iter_mut().map(|o| o.as_mut_ptr()).collect(),
        len: 1usize << col_vars,
    };
    let use_x86 = bank_transpose_x86();
    let use_lookup512 = bank_lookup_x86();
    let stride = BankTables::stride_for(np);
    let rg = (BANK_TABLE_BYTES / (stride * 256 * 16))
        .clamp(BANK_ROW_GROUPS_MIN, BANK_ROW_GROUPS_MAX);
    tasks.par_iter().for_each_init(
        || BankTables::new(np, rg),
        |tables, &(zw, oc, ncols, live, n_t)| {
            let groups = n_t.div_ceil(8);
            let mut buf = [0u8; BANK_ROW_GROUPS_MAX * 128];
            // Weight vectors of the current group: position p of vector r is the eq weight
            // of row 8 g + r at point p; padding positions stay zero.
            let mut w = vec![F128::ZERO; 8 * stride];
            for gt in (0..groups).step_by(rg) {
                let gn = rg.min(groups - gt);
                for g in 0..gn {
                    let row0 = 8 * (gt + g);
                    for (p, eq) in eqs.iter().enumerate() {
                        for r in 0..8 {
                            w[r * stride + p] =
                                if row0 + r < n_t { eq[row0 + r] } else { F128::ZERO };
                        }
                    }
                    let t = &mut tables.as_mut_slice()[g * 256 * stride..(g + 1) * 256 * stride];
                    build_sum_table8(use_lookup512, &w, stride, t);
                }
                for c in 0..ncols {
                    let col = &z[zw + (c << nu)..zw + ((c + 1) << nu)];
                    let runs = &col[8 * gt..8 * (gt + gn)];
                    bank_transpose(use_x86, runs, &mut buf[..gn * 128]);
                    let nb = 128.min(live - 128 * c);
                    let base = 128 * (oc + c);
                    // The output column oc + c < 2^(col_vars - 7) belongs to this task only.
                    assert!(base + nb <= ptrs.len(), "bit-bank output column out of range");
                    bank_lookup(
                        use_lookup512,
                        tables,
                        np,
                        &buf[..gn * 128],
                        gn,
                        nb,
                        ptrs.ptrs(),
                        base,
                    );
                }
            }
        },
    );
    drop(ptrs);
    outs
}

/// For every point `p < np` and bit `b < nb`:
/// `outs[p][base + b] ^= XOR_{g < gn} entry(g, bytes[128 g + b])[p]`, where `tables` is
/// interleaved as in [`BankTables`].
///
/// The caller guarantees `base + nb <= len` of every output and that no other task touches
/// `outs[p][base .. base + nb]`.
#[inline]
#[allow(clippy::too_many_arguments)]
fn bank_lookup(
    use_512: bool,
    tables: &BankTables,
    np: usize,
    bytes: &[u8],
    gn: usize,
    nb: usize,
    outs: &[*mut F128],
    base: usize,
) {
    let stride = tables.stride;
    let t = tables.as_slice();
    // These bounds are what the unchecked loads below rely on: a byte value is below 256,
    // so group g's entries are `t[g * 256 * stride ..][.. 256 * stride]`.
    assert!(t.len() >= gn * 256 * stride && bytes.len() >= gn * 128 && nb <= 128);
    assert!(np <= stride.max(1) && outs.len() >= np);
    #[cfg(target_arch = "x86_64")]
    if use_512 {
        match crate::simd::ghash_width() {
            crate::simd::GhashWidth::Avx512 if stride.is_multiple_of(4) => {
                // SAFETY: AVX-512F was checked (`bank_lookup_x86`); stride is a
                // multiple of four; the table and byte ranges are asserted above;
                // the tables start on a 64-byte line; the output range is the
                // caller's contract.
                unsafe {
                    crate::lincheck::kernels::bank_lookup_avx512(
                        t.as_ptr(),
                        stride,
                        np,
                        bytes.as_ptr(),
                        gn,
                        nb,
                        outs,
                        base,
                    )
                };
                return;
            }
            crate::simd::GhashWidth::Vpclmul256 | crate::simd::GhashWidth::Pclmul
                if stride.is_multiple_of(2) =>
            {
                // SAFETY: AVX2 was checked; stride is even; the table and byte
                // ranges are asserted above; the output range is the caller's
                // contract.
                unsafe {
                    crate::lincheck::kernels::bank_lookup_avx2(
                        t.as_ptr(),
                        stride,
                        np,
                        bytes.as_ptr(),
                        gn,
                        nb,
                        outs,
                        base,
                    )
                };
                return;
            }
            _ => {}
        }
    }
    let _ = use_512;
    // Portable path: up to four points at a time, four bits in flight.
    for q0 in (0..np).step_by(4) {
        let cnt = (np - q0).min(4);
        let mut b = 0;
        while b < nb {
            let w = 4.min(nb - b);
            let mut acc = [[F128::ZERO; 4]; 4];
            for g in 0..gn {
                for (j, a) in acc.iter_mut().enumerate().take(w) {
                    let s = bytes[128 * g + b + j] as usize;
                    let e = &t[(g * 256 + s) * stride + q0..][..cnt];
                    for (x, y) in a.iter_mut().zip(e) {
                        *x += *y;
                    }
                }
            }
            for (j, a) in acc.iter().enumerate().take(w) {
                for (i, v) in a.iter().enumerate().take(cnt) {
                    // SAFETY: base + b + j < base + nb <= len; disjoint per the contract.
                    unsafe { *outs[q0 + i].add(base + b + j) += *v };
                }
            }
            b += w;
        }
    }
}

/// Interleaved subset-sum table of one group: entry `s` (F128 `s * stride ..`), position
/// `p`, is `Σ_{r: bit r of s set} w[r * stride + p]` (255 vector additions, doubling).
/// `w` holds the eight weight vectors of `stride` F128 each, zero past the live points.
#[inline(always)]
fn build_sum_table8_body(w: &[F128], stride: usize, table: &mut [F128]) {
    debug_assert!(table.len() >= 256 * stride && w.len() == 8 * stride);
    table[..stride].fill(F128::ZERO);
    for (i, wi) in w.chunks_exact(stride).enumerate() {
        let len = 1usize << i;
        // Entries 0 .. len are final; entries len .. 2 len are those plus weight vector i.
        let (done, rest) = table[..2 * len * stride].split_at_mut(len * stride);
        for (src, dst) in done.chunks_exact(stride).zip(rest.chunks_exact_mut(stride)) {
            for ((d, s), e) in dst.iter_mut().zip(src).zip(wi) {
                d.lo = s.lo ^ e.lo;
                d.hi = s.hi ^ e.hi;
            }
        }
    }
}

/// [`build_sum_table8_body`] compiled for AVX-512 (one zmm per four points).
///
/// # Safety
/// AVX-512F available.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn build_sum_table8_avx512(w: &[F128], stride: usize, table: &mut [F128]) {
    build_sum_table8_body(w, stride, table)
}

/// Fill `table` (`256 * stride` F128) as in [`build_sum_table8_body`], vectorized when
/// `avx512` is set.
fn build_sum_table8(avx512: bool, w: &[F128], stride: usize, table: &mut [F128]) {
    #[cfg(target_arch = "x86_64")]
    if avx512 {
        match crate::simd::ghash_width() {
            crate::simd::GhashWidth::Avx512 => {
                // SAFETY: AVX-512F was checked by the caller (`bank_lookup_x86`).
                unsafe { build_sum_table8_avx512(w, stride, table) };
                return;
            }
            crate::simd::GhashWidth::Vpclmul256 | crate::simd::GhashWidth::Pclmul => {
                // SAFETY: AVX2 was checked by the caller.
                unsafe { crate::lincheck::kernels::build_sum_table8_avx2(w, stride, table) };
                return;
            }
            crate::simd::GhashWidth::Portable => {}
        }
    }
    let _ = avx512;
    build_sum_table8_body(w, stride, table);
}

/// Whether a vectorized table lookup (AVX-512 or AVX2) may run.
fn bank_lookup_x86() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        crate::simd::ghash_width() != crate::simd::GhashWidth::Portable
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

/// Whether the AVX-512 VBMI + GFNI bit transpose may run.
fn bank_transpose_x86() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        crate::simd::has(crate::simd::AVX512_VBMI | crate::simd::AVX512_GFNI)
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

/// Bit transpose of eight-word runs: run `g` (`words[8 g ..][..8]`, rows `r = 0..8`) to
/// 128 bytes `out[128 g + b]` whose bit `r` is bit `b` of word `r` (`b < 64` from `lo`,
/// `b ≥ 64` from `hi`) — the lincheck stripe bytes of one column of eight rows.
fn bank_transpose(use_x86: bool, words: &[F128], out: &mut [u8]) {
    debug_assert_eq!(words.len() * 16, out.len());
    #[cfg(target_arch = "x86_64")]
    if use_x86 {
        // SAFETY: the features were checked (`bank_transpose_x86`); lengths asserted above.
        unsafe { crate::lincheck::kernels::bank_transpose_avx512(words, out) };
        return;
    }
    let _ = use_x86;
    for (run, o) in words.chunks_exact(8).zip(out.chunks_exact_mut(128)) {
        let lo: [u64; 8] = std::array::from_fn(|r| run[r].lo);
        let hi: [u64; 8] = std::array::from_fn(|r| run[r].hi);
        transpose_8x64(&lo, &mut o[..64]);
        transpose_8x64(&hi, &mut o[64..]);
    }
}

/// `out[bit] = Σ_r bit(v[r], bit) << r` for `bit < 64`: eight 8×8 bit-matrix transposes
/// (byte `k` of every `v[r]` forms one matrix), three delta swaps each.
#[inline]
pub(crate) fn transpose_8x64(v: &[u64; 8], out: &mut [u8]) {
    debug_assert_eq!(out.len(), 64);
    for k in 0..8 {
        let mut x = 0u64;
        for (r, &w) in v.iter().enumerate() {
            x |= ((w >> (8 * k)) & 0xff) << (8 * r);
        }
        let t = (x ^ (x >> 7)) & 0x00AA_00AA_00AA_00AA;
        x ^= t ^ (t << 7);
        let t = (x ^ (x >> 14)) & 0x0000_CCCC_0000_CCCC;
        x ^= t ^ (t << 14);
        let t = (x ^ (x >> 28)) & 0x0000_0000_F0F0_F0F0;
        x ^= t ^ (t << 28);
        out[8 * k..8 * k + 8].copy_from_slice(&x.to_le_bytes());
    }
}

pub fn prove_union_capture_z_vec_with_grinding<Ch: Challenger>(
    union: &UnionInstance<'_>,
    slots: &[UnionLincheckSlot<'_>],
    x_ab: &QuirkyPoint,
    grinding: LincheckGrinding,
    challenger: &mut Ch,
) -> (LincheckProof, LincheckClaim, Vec<F128>) {
    let registry = union.registry();
    let nu = union.n_log();
    assert!(nu >= 3, "lincheck stripe fold needs n_outer ≥ 8 (nu ≥ 3)");
    for (ty, slot_in) in registry.boolean_types().iter().zip(slots) {
        assert_eq!(
            slot_in.z_lincheck.len(),
            1usize << (nu + ty.k_log - 3),
            "slot lincheck stripe length mismatch"
        );
    }
    // 3. Per-slot row folds against the SHARED standard eq table of the row
    //    block r|_R (uniform capacity: one table, all slots), placed at the
    //    slots' aligned column offsets — the union generalization of today's
    //    single partial fold. Dummy rows are honest zeros, so folding only
    //    the DECLARED rows equals the full-capacity fold byte for byte —
    //    the row-aware dispatch makes the fold count-proportional (M6).
    let t_fold = var("LINCHECK_TRACE").is_ok().then(Instant::now);
    let z_vec = union_bitbank_fold(union, slots, &x_ab.x_outer);
    if let Some(t) = t_fold {
        eprintln!(
            "[lc] {:<26} {:>7.2} ms",
            "union partial folds",
            t.elapsed().as_secs_f64() * 1e3
        );
    }
    let circuits: Vec<&dyn LincheckCircuit> = slots.iter().map(|s| s.circuit).collect();
    prove_union_with_z_vec_with_grinding(union, &circuits, x_ab, z_vec, grinding, challenger)
}

/// [`prove_union_capture_z_vec_with_grinding`] with the union fold `z_vec` supplied by
/// the caller: the column bit-bank of the union witness at `x_ab.x_outer`
/// ([`union_bitbank_fold`] or one point of [`union_bitbank_fold_multi`]), so the
/// lincheck needs no stripe copy of the witness. Same transcript and bytes.
pub fn prove_union_with_z_vec_with_grinding<Ch: Challenger>(
    union: &UnionInstance<'_>,
    circuits: &[&dyn LincheckCircuit],
    x_ab: &QuirkyPoint,
    z_vec: Vec<F128>,
    grinding: LincheckGrinding,
    challenger: &mut Ch,
) -> (LincheckProof, LincheckClaim, Vec<F128>) {
    let registry = union.registry();
    let k_skip = K_SKIP;
    let nu = union.n_log();
    let col_vars = union.m_bool() - nu;
    let inner_rest_len = col_vars - k_skip;
    assert_eq!(
        circuits.len(),
        registry.num_boolean(),
        "one lincheck input per BOOLEAN registry type (element slots have their own PIOP)"
    );
    assert_eq!(x_ab.x_inner_rest.len(), inner_rest_len);
    assert_eq!(x_ab.x_outer.len(), nu);
    assert_eq!(z_vec.len(), 1usize << col_vars, "union fold length");
    for (ty, circuit) in registry.boolean_types().iter().zip(circuits) {
        assert_eq!(
            circuit.n_cols(),
            1usize << ty.k_log,
            "circuit width must match the registry type"
        );
        assert_eq!(
            circuit.const_pin_col(),
            ty.const_pin,
            "circuit const_pin must match the registry type"
        );
    }

    challenger.observe_label(b"flock-lincheck-v0");
    let trace = var("LINCHECK_TRACE").is_ok();
    let mut grinding_nonces = Vec::with_capacity(
        grinding.nonce_count(
            inner_rest_len,
            circuits
                .iter()
                .filter(|c| c.const_pin_col().is_some())
                .count(),
            k_skip,
        ),
    );

    // 1. Sample α (matches verifier's order). ONE α batches the A- and
    //    B-claims for every slot (doc §"The B-claim, and batching the two
    //    sumchecks"); the cross-slot weights w_t(r) are fixed scalars, not
    //    randomness.
    let alpha = if let Some(bits) = grinding.alpha_bits() {
        let (nonce, alpha) = challenger.grind_pow_and_sample_f128(bits);
        grinding_nonces.push(nonce);
        alpha
    } else {
        challenger.sample_f128()
    };

    // 2. Per-type α-batched combs via each type's quirky table — the slot's
    //    column coordinates of r are z_skip, dim6, and the LOW `κ_t − 7`
    //    union chunk coordinates — scaled by the slot prefix weight w_t(r)
    //    and placed at the slot's aligned column offset `o_t / 2^ν`.
    //    Dense union-column placement (see module docs); a single slot is
    //    today's comb vector, moved.
    // The T = 1 fast path (today's vectors verbatim) needs the one boolean
    // slot to FILL the boolean region — true for every single-type registry,
    // and for no multi-slot one.
    let single = registry.num_boolean() == 1 && registry.slots()[0].m_slot == registry.m_bool();
    let t_comb = if trace { Some(Instant::now()) } else { None };
    let mut comb_vec: Vec<F128> = if single {
        Vec::new()
    } else {
        vec![F128::ZERO; 1usize << col_vars]
    };
    // Per-type (ξ_A, ξ_B), kept for the matrix-eval report after the sumcheck.
    let mut split_combs: Vec<(Vec<F128>, Vec<F128>)> = Vec::with_capacity(registry.num_boolean());
    for ((ty, slot), circuit) in registry
        .boolean_types()
        .iter()
        .zip(registry.slots())
        .zip(circuits)
    {
        let inner = ty.k_log - k_skip;
        let eq_inner = build_quirky_eq_table_from_weights(
            &x_ab.z_skip.weights(k_skip),
            &x_ab.x_inner_rest[..inner],
        );
        // Split, so the per-matrix bilinear values can be reported below.
        // Same nonzeros as the α-batched fold; only the accumulation differs.
        let (comb_a, comb_b) = circuit.fold_split(&eq_inner);
        let comb_t: Vec<F128> = comb_a
            .iter()
            .zip(&comb_b)
            .map(|(a, b)| alpha * *a + *b)
            .collect();
        split_combs.push((comb_a, comb_b));
        if single {
            comb_vec = comb_t; // T = 1: w_1 = 1, offset 0 — today's vector.
        } else {
            let w_t = eq_prefix_weight(&x_ab.x_inner_rest[inner..], slot.prefix);
            let off = slot.prefix << ty.k_log;
            for (dst, &src) in comb_vec[off..off + comb_t.len()].iter_mut().zip(&comb_t) {
                *dst = w_t * src;
            }
        }
    }

    // 2b. Constant-wire pins, per type in slot order: β_t sampled after α,
    //     added UNSCALED at the pin's union column. The verifier's target
    //     gains the count-derived β_t · Σ_{row<n_t} eq(x_outer, row) — see
    //     `verify_union` and the module docs.
    for (slot, circuit) in registry.slots().iter().zip(circuits) {
        if let Some(col) = circuit.const_pin_col() {
            let beta = if let Some(bits) = grinding.beta_bits() {
                let (nonce, beta) = challenger.grind_pow_and_sample_f128(bits);
                grinding_nonces.push(nonce);
                beta
            } else {
                challenger.sample_f128()
            };
            let off = slot.prefix << (slot.m_slot - nu);
            comb_vec[off + col] += beta;
        }
    }
    if let Some(t) = t_comb {
        eprintln!(
            "[lc] {:<26} {:>7.2} ms",
            "union comb build",
            t.elapsed().as_secs_f64() * 1e3
        );
    }

    // 3b. Capture the pre-sumcheck union fold g for downstream reuse
    //     (mirrors `prove_padded_capture_z_vec`).
    let captured = z_vec.clone();

    // 4.–9. The existing sumcheck core, over the (longer) union column
    //       domain.
    let (mut proof, claim) = column_sumcheck_prove(
        comb_vec,
        z_vec,
        k_skip,
        &x_ab.z_skip,
        trace,
        grinding,
        &mut grinding_nonces,
        challenger,
    );

    // 10. The matrix work, reported rather than folded into the target: for
    //     each type the UNSCALED pair (⟨W_t, A_0⟩, ⟨W_t, B_0⟩) with the
    //     column weight W_col_t = z_partial ⊗ eq(rr[..inner_t]) that the
    //     sumcheck just fixed. The slot's prefix weights and β are verifier
    //     scalars, so they stay out — what accumulates must be a claim about
    //     the static matrix alone.
    proof.matrix_evals = registry
        .boolean_types()
        .iter()
        .zip(&split_combs)
        .map(|(ty, (comb_a, comb_b))| {
            let inner = ty.k_log - k_skip;
            let w_col = col_weight(&proof.z_partial, &claim.r_inner_rest[..inner], k_skip);
            (inner_product(&w_col, comb_a), inner_product(&w_col, comb_b))
        })
        .collect();

    (proof, claim, captured)
}

/// `z_partial ⊗ eq(rr)` over a type's own `2^{k_skip + rr.len()}` column
/// domain — the lincheck's column weight, shared by the prover's report and
/// the verifier's check so the two cannot drift.
fn col_weight(z_partial: &[F128], rr: &[F128], k_skip: usize) -> Vec<F128> {
    let n_skip = 1usize << k_skip;
    debug_assert_eq!(z_partial.len(), n_skip);
    let eq = build_eq_table(rr);
    let mut out = Vec::with_capacity(n_skip * eq.len());
    for &e in &eq {
        for &zp in z_partial {
            out.push(e * zp);
        }
    }
    out
}

/// The `O(nnz)` residue of the lincheck verifier: everything its final
/// consistency check needs EXCEPT the per-type comb folds.
///
/// The verifier's whole dependence on the base matrices is this one
/// assertion — a bilinear form in `α·Â₀ + B̂₀` at a rank-1 tensor point,
/// with the row weight `λ(z_skip) ⊗ eq(x_inner_rest)` and the column weight
/// `z_partial ⊗ eq(rr)`. Everything else the verifier does is succinct.
/// [`verify_union`] discharges it immediately (identical behaviour to
/// before this split); [`verify_union_deferred`] hands it back undischarged
/// so a caller can instead
///
/// * batch it across many proofs of the same registry and pay `O(nnz)`
///   once rather than per proof, or
/// * carry it as a public claim into an accumulator — the recursion path,
///   where a circuit replays only the succinct verifier and never pays the
///   matrix work at all.
///
/// Two of the four tensor factors (`λ(z_skip)` from the univariate skip,
/// and the proof-supplied `z_partial`) are not `eq`-tensors, so this is a
/// weighted sum rather than a plain MLE evaluation; a folding protocol's
/// first sumcheck is what turns it into one.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MatrixAssertion {
    /// The challenge batching the A- and B-matrices.
    pub alpha: F128,
    /// Row-side weight: `λ(z_skip) ⊗ eq(x_inner_rest)` — `λ` is the skip
    /// basis's evaluation functional ([`SkipPoint::weights`]), φ₈ Lagrange
    /// on the RS route, the AG base-code functional on the AG route.
    pub z_skip: SkipPoint,
    pub x_inner_rest: Vec<F128>,
    /// Column-side weight: `z_partial ⊗ eq(rr)`.
    pub rr: Vec<F128>,
    pub z_partial: Vec<F128>,
    /// Const-wire pin challenges, one entry per boolean slot in slot order
    /// (`None` where the type has no pin). Applied to the comb at the pin
    /// column; the target-side half of the pin is already folded into
    /// [`Self::target`].
    pub betas: Vec<Option<F128>>,
    /// Outer-row point and the count-prefix evaluation used by each const
    /// pin. Recursive verification carries these as claims on the static
    /// circuit-structure table instead of accepting prefix sums as advice.
    pub pin_point: Vec<F128>,
    pub pin_evals: Vec<Option<F128>>,
    /// What the weighted bilinear sum must equal.
    pub target: F128,
    /// The prover's per-type `(⟨W_t, A_0⟩, ⟨W_t, B_0⟩)`, in slot order —
    /// unscaled, so each is a claim about a registry-static matrix and
    /// nothing else. [`Self::check`] verifies they reproduce
    /// [`Self::target`]; [`Self::claims`] hands them out for accumulation.
    pub evals: Vec<(F128, F128)>,
}

impl MatrixAssertion {
    /// The per-`(type, matrix)` claims this assertion asserts, in slot order
    /// as `[(A_0 claim, B_0 claim); T]` — ready for
    /// [`crate::matrix_fold::prove_fold`].
    ///
    /// A and B stay separate: the target only ever carries their
    /// α-combination, and α is per-proof, so a claim about `α·A_0 + B_0`
    /// would name a different polynomial in every proof. These name the
    /// static matrices, and fold forever.
    pub fn claims(&self, registry: &Registry) -> Vec<(MatrixClaim, MatrixClaim)> {
        let k_skip = K_SKIP;
        registry
            .boolean_types()
            .iter()
            .zip(&self.evals)
            .map(|(ty, &(v_a, v_b))| {
                let inner = ty.k_log - k_skip;
                let row = Weight::low_eq(
                    self.z_skip.weights(k_skip),
                    self.x_inner_rest[..inner].to_vec(),
                );
                let col = Weight::low_eq(self.z_partial.clone(), self.rr[..inner].to_vec());
                (
                    MatrixClaim {
                        row: row.clone(),
                        col: col.clone(),
                        value: v_a,
                    },
                    MatrixClaim {
                        row,
                        col,
                        value: v_b,
                    },
                )
            })
            .collect()
    }

    /// Discharge the assertion: rebuild the per-type combs (`O(Σ_t nnz_t)`,
    /// the only place the base matrices are read) and check the closed-form
    /// Comb-hat collapse against [`Self::target`].
    pub fn check_reported(&self, registry: &Registry) -> Result<(), LincheckError> {
        let k_skip = K_SKIP;
        if self.evals.len() != registry.num_boolean() {
            return Err(LincheckError::BadVectorLength {
                which: "matrix_evals",
                expected: registry.num_boolean(),
                got: self.evals.len(),
            });
        }
        // target = Σ_t P_t·w_t·(α·⟨W_t,A⟩ + ⟨W_t,B⟩) + Σ_t P_t·β_t·W_col_t[pin]
        // — the slot prefix weights on both sides (w_t from the row point,
        // P_t from the bound column point) and β are all verifier scalars.
        let mut acc = F128::ZERO;
        for (((ty, slot), &(v_a, v_b)), beta) in registry
            .boolean_types()
            .iter()
            .zip(registry.slots())
            .zip(&self.evals)
            .zip(&self.betas)
        {
            let inner = ty.k_log - k_skip;
            let w_t = eq_prefix_weight(&self.x_inner_rest[inner..], slot.prefix);
            let p_t = eq_prefix_weight(&self.rr[inner..], slot.prefix);
            acc += p_t * w_t * (self.alpha * v_a + v_b);
            if let (Some(col), Some(beta)) = (ty.const_pin, beta) {
                let w_col = col_weight(&self.z_partial, &self.rr[..inner], k_skip);
                acc += p_t * *beta * w_col[col];
            }
        }
        if acc != self.target {
            return Err(LincheckError::ConsistencyFailed {
                which: "matrix-evals",
            });
        }
        Ok(())
    }

    /// Discharge against the real matrices — the `O(Σ_t nnz_t)` check. Used
    /// by [`verify_union`], and the differential oracle for the accumulated
    /// path: this and [`Self::check_reported`] + discharging the
    /// [`Self::claims`] must accept exactly the same proofs.
    pub fn check(
        &self,
        union: &UnionInstance<'_>,
        circuits: &[&dyn LincheckCircuit],
    ) -> Result<(), LincheckError> {
        let registry = union.registry();
        let k_skip = K_SKIP;
        assert_eq!(
            circuits.len(),
            registry.num_boolean(),
            "one circuit per BOOLEAN registry type"
        );
        let mut combs: Vec<Vec<F128>> = registry
            .boolean_types()
            .iter()
            .zip(registry.slots())
            .zip(circuits)
            .map(|((ty, slot), circuit)| {
                let inner = ty.k_log - k_skip;
                let eq_inner = build_quirky_eq_table_from_weights(
                    &self.z_skip.weights(k_skip),
                    &self.x_inner_rest[..inner],
                );
                let mut comb = circuit.fold_alpha_batched(self.alpha, &eq_inner);
                if slot.prefix_bits > 0 {
                    let w_t = eq_prefix_weight(&self.x_inner_rest[inner..], slot.prefix);
                    for v in &mut comb {
                        *v *= w_t;
                    }
                }
                comb
            })
            .collect();

        for ((circuit, comb), beta) in circuits.iter().zip(&mut combs).zip(&self.betas) {
            if let (Some(col), Some(beta)) = (circuit.const_pin_col(), beta) {
                comb[col] += *beta;
            }
        }

        let comb_partial = union_comb_partial(registry, &combs, &self.rr, k_skip);
        if self.target != inner_product(&comb_partial, &self.z_partial) {
            return Err(LincheckError::ConsistencyFailed {
                which: "sumcheck-final",
            });
        }
        Ok(())
    }
}

/// Verify a union-column lincheck proof: walk the challenger in lockstep
/// with [`prove_union_capture_z_vec`], replay the rounds, and check the
/// final consistency against the closed-form Comb-hat collapse
/// ([`union_comb_partial`]) — per-type `O(nnz)` comb machinery plus
/// `O(T·M)` prefix weights, never anything union-sized. `circuits` are the
/// per-type lincheck circuits in slot order. Returns the single witness
/// claim `ẑ(c*-columns, r|_R-rows)`.
///
/// This is [`verify_union_deferred`] followed immediately by
/// [`MatrixAssertion::check`] — identical accept/reject, and the entry
/// point every existing caller keeps using.
pub fn verify_union<Ch: Challenger>(
    union: &UnionInstance<'_>,
    circuits: &[&dyn LincheckCircuit],
    x_ab: &QuirkyPoint,
    v_a: F128,
    v_b: F128,
    proof: &LincheckProof,
    challenger: &mut Ch,
) -> Result<LincheckClaim, LincheckError> {
    verify_union_with_grinding(
        union,
        circuits,
        x_ab,
        v_a,
        v_b,
        proof,
        LincheckGrinding::disabled(),
        challenger,
    )
}

/// [`verify_union`] with an explicit Fiat--Shamir grinding policy.
pub fn verify_union_with_grinding<Ch: Challenger>(
    union: &UnionInstance<'_>,
    circuits: &[&dyn LincheckCircuit],
    x_ab: &QuirkyPoint,
    v_a: F128,
    v_b: F128,
    proof: &LincheckProof,
    grinding: LincheckGrinding,
    challenger: &mut Ch,
) -> Result<LincheckClaim, LincheckError> {
    let (claim, assertion) = verify_union_deferred_with_grinding(
        union, circuits, x_ab, v_a, v_b, proof, grinding, challenger,
    )?;
    assertion.check(union, circuits)?;
    Ok(claim)
}

/// [`verify_union`] with the matrix work left undischarged: replays the
/// whole protocol succinctly and returns the witness claim alongside the
/// [`MatrixAssertion`] the caller must discharge (or accumulate) for the
/// proof to mean anything.
///
/// **The returned claim is conditional.** Nothing here reads the base
/// matrices, so a proof whose lincheck is simply wrong still yields
/// `Ok(..)`; only [`MatrixAssertion::check`], or a folding protocol that
/// eventually discharges it, rejects that. Callers that are not
/// accumulating must use [`verify_union`].
pub fn verify_union_deferred<Ch: Challenger>(
    union: &UnionInstance<'_>,
    circuits: &[&dyn LincheckCircuit],
    x_ab: &QuirkyPoint,
    v_a: F128,
    v_b: F128,
    proof: &LincheckProof,
    challenger: &mut Ch,
) -> Result<(LincheckClaim, MatrixAssertion), LincheckError> {
    verify_union_deferred_with_grinding(
        union,
        circuits,
        x_ab,
        v_a,
        v_b,
        proof,
        LincheckGrinding::disabled(),
        challenger,
    )
}

/// [`verify_union_deferred`] with an explicit Fiat--Shamir grinding policy.
pub fn verify_union_deferred_with_grinding<Ch: Challenger>(
    union: &UnionInstance<'_>,
    circuits: &[&dyn LincheckCircuit],
    x_ab: &QuirkyPoint,
    v_a: F128,
    v_b: F128,
    proof: &LincheckProof,
    grinding: LincheckGrinding,
    challenger: &mut Ch,
) -> Result<(LincheckClaim, MatrixAssertion), LincheckError> {
    let registry = union.registry();
    let k_skip = K_SKIP;
    let nu = union.n_log();
    let col_vars = union.m_bool() - nu;
    let inner_rest_len = col_vars - k_skip;
    let n_skip = 1usize << k_skip;

    assert_eq!(
        circuits.len(),
        registry.num_boolean(),
        "one circuit per BOOLEAN registry type (element slots have their own PIOP)"
    );
    for (ty, circuit) in registry.boolean_types().iter().zip(circuits) {
        let k = 1usize << ty.k_log;
        if circuit.n_cols() != k {
            return Err(LincheckError::BadMatrixShape {
                which: "circuit",
                expected: k,
                got_rows: k,
                got_cols: circuit.n_cols(),
            });
        }
        assert_eq!(
            circuit.const_pin_col(),
            ty.const_pin,
            "circuit const_pin must match the registry type"
        );
    }
    if x_ab.x_inner_rest.len() != inner_rest_len {
        return Err(LincheckError::BadInnerRestLength {
            which: "x_ab",
            expected: inner_rest_len,
            got: x_ab.x_inner_rest.len(),
        });
    }
    if x_ab.x_outer.len() != nu {
        return Err(LincheckError::BadOuterLength {
            which: "x_ab",
            expected: nu,
            got: x_ab.x_outer.len(),
        });
    }
    if proof.rounds.len() != inner_rest_len {
        return Err(LincheckError::BadVectorLength {
            which: "rounds",
            expected: inner_rest_len,
            got: proof.rounds.len(),
        });
    }
    if proof.z_partial.len() != n_skip {
        return Err(LincheckError::BadVectorLength {
            which: "z_partial",
            expected: n_skip,
            got: proof.z_partial.len(),
        });
    }
    let expected_nonces = grinding.nonce_count(
        inner_rest_len,
        circuits
            .iter()
            .filter(|circuit| circuit.const_pin_col().is_some())
            .count(),
        k_skip,
    );
    if proof.grinding_nonces.len() != expected_nonces {
        return Err(LincheckError::BadGrindingNonceCount {
            expected: expected_nonces,
            got: proof.grinding_nonces.len(),
        });
    }

    challenger.observe_label(b"flock-lincheck-v0");
    let mut nonce_idx = 0;

    // 1. Sample α (matches prover's order).
    let alpha = if let Some(bits) = grinding.alpha_bits() {
        let alpha = challenger
            .verify_pow_and_sample_f128_canonical(proof.grinding_nonces[nonce_idx], bits)
            .ok_or(LincheckError::InvalidGrindingNonce { which: "alpha" })?;
        nonce_idx += 1;
        alpha
    } else {
        challenger.sample_f128()
    };

    // 2. The per-type α-batched combs are NOT built here — they are the
    //    only `O(nnz)` step, and they are needed nowhere before the final
    //    consistency check. They ride out in the `MatrixAssertion` instead.
    //    Nothing between here and there touches the challenger on their
    //    account, so lifting them out leaves the transcript untouched.
    //
    //    Merge note (`merkle-path-r1cs`): that branch added a `VERIFY_TRACE`
    //    split of this function into "comb phase" and "everything after".
    //    Deferral removed the comb phase from here entirely, so the
    //    instrumentation had nothing left to time and is not carried over.
    //    The measurement it existed for lived in `verify_union_timed` (which
    //    built combs inline on purpose) until that probe was deleted in the
    //    bloat sweep (ledger §E).

    // 2b. Constant-wire pins (mirror of prove): β_t sampled after α in slot
    //     order, the comb gains +β_t at the pin, and the target gains the
    //     count-derived β_t · Σ_{row<n_t} eq(x_outer, row) — declared rows
    //     carry pin = 1, dummy rows are all-zero (pin included). At full
    //     utilization the sum is exactly 1, reproducing today's target += β.
    //     Only the target half is count-derived work; the comb half is
    //     carried in the assertion.
    let mut target = alpha * v_a + v_b;
    let mut betas: Vec<Option<F128>> = Vec::with_capacity(circuits.len());
    let mut pin_evals: Vec<Option<F128>> = Vec::with_capacity(circuits.len());
    for (circuit, &n_t) in circuits.iter().zip(union.counts()) {
        if circuit.const_pin_col().is_some() {
            let beta = if let Some(bits) = grinding.beta_bits() {
                let beta = challenger
                    .verify_pow_and_sample_f128_canonical(proof.grinding_nonces[nonce_idx], bits)
                    .ok_or(LincheckError::InvalidGrindingNonce { which: "beta" })?;
                nonce_idx += 1;
                beta
            } else {
                challenger.sample_f128()
            };
            betas.push(Some(beta));
            let eps = eq_prefix_sum(&x_ab.x_outer, n_t);
            pin_evals.push(Some(eps));
            target += beta * eps;
        } else {
            betas.push(None);
            pin_evals.push(None);
        }
    }

    // 3. Replay the multilinear product-sumcheck rounds. Unlike the
    //    single-table verifier there is no comb fold in lockstep — the
    //    skip-block collapse is evaluated once at the end, in closed form.
    let mut running = target;
    let mut r_rounds = Vec::with_capacity(inner_rest_len);
    for &(e1, einf) in &proof.rounds {
        challenger.observe_f128(e1);
        challenger.observe_f128(einf);
        let r = if let Some(bits) = grinding.multilinear_round_bits() {
            let r = challenger
                .verify_pow_and_sample_f128_canonical(proof.grinding_nonces[nonce_idx], bits)
                .ok_or(LincheckError::InvalidGrindingNonce {
                    which: "sumcheck-round",
                })?;
            nonce_idx += 1;
            r
        } else {
            challenger.sample_f128()
        };
        // q(0) = claim + q(1) in char 2; q(X) = einf·X² + c1·X + e0.
        let e0 = running + e1;
        let c1 = e0 + e1 + einf;
        running = einf * r * r + c1 * r + e0;
        r_rounds.push(r);
    }

    // 4. Observe z_partial AFTER the sumcheck rounds (matches prover order).
    challenger.observe_f128_slice(&proof.z_partial);

    // 5. The final consistency check — the closed-form Comb-hat collapse
    //    against `running` — is exactly the deferred assertion. `rr` is
    //    LSB-first: the loop bound the TOP coordinate first, so rr[i] binds
    //    column coordinate k_skip + i.
    let mut rr = r_rounds;
    rr.reverse();
    let assertion = MatrixAssertion {
        alpha,
        z_skip: x_ab.z_skip,
        x_inner_rest: x_ab.x_inner_rest.clone(),
        rr: rr.clone(),
        z_partial: proof.z_partial.clone(),
        betas,
        pin_point: x_ab.x_outer.clone(),
        pin_evals,
        target: running,
        evals: proof.matrix_evals.clone(),
    };

    // 6.–7. Fresh skip challenge after z_partial, in the SAME basis as the
    //       input point (identical to the single-table verifier); claim
    //       value via the basis's evaluation functional. Mirrors
    //       `column_sumcheck_prove`'s grinding composition exactly: the φ₈
    //       basis keeps the FUSED PoW+squeeze (byte-pinned), the AG basis
    //       verifies the fused nonce one-shot (`sample_fresh_pow_verifier`).
    let r_inner_skip = match x_ab.z_skip {
        SkipPoint::Phi8(_) => match grinding.skip_bits(k_skip) {
            Some(bits) => {
                let nonce = proof.grinding_nonces[nonce_idx];
                nonce_idx += 1;
                let r = challenger.verify_pow_and_sample_f128_canonical(nonce, bits).ok_or(
                    LincheckError::InvalidGrindingNonce {
                        which: "inner-skip",
                    },
                )?;
                SkipPoint::Phi8(r)
            }
            None => x_ab.z_skip.sample_fresh(challenger),
        },
        // AG basis: ONE-SHOT fused-nonce check — 3 explicit bits + the
        // sampler's 5 provable bits = bits_for(158) for the base-code bad set.
        SkipPoint::Ag(_) => match grinding.ag_skip_bits(k_skip) {
            Some(bits) => {
                let nonce = proof.grinding_nonces[nonce_idx];
                nonce_idx += 1;
                x_ab.z_skip
                    .sample_fresh_pow_verifier(challenger, nonce, bits)
                    .ok_or(LincheckError::InvalidGrindingNonce {
                        which: "inner-skip",
                    })?
            }
            None => x_ab.z_skip.sample_fresh(challenger),
        },
        SkipPoint::Eq(_) => panic!("{}", crate::lincheck::EQ_SKIP_OPENING_ONLY),
    };
    debug_assert_eq!(nonce_idx, proof.grinding_nonces.len());
    let lambda = r_inner_skip.weights(k_skip);
    let w = inner_product(&lambda, &proof.z_partial);

    Ok((
        LincheckClaim {
            r_inner_skip,
            r_inner_rest: rr,
            w,
        },
        assertion,
    ))
}

#[cfg(test)]
mod tests {
    use crate::{
        lincheck::union::{
            F128, K_SKIP, Registry, build_eq_table, eq_prefix_sum, eq_prefix_weight,
            union_comb_partial,
        },
        r1cs::SparseBinaryMatrix,
        schedule::{TableClass, TableType},
        test_rng::Rng,
    };

    fn stub() -> SparseBinaryMatrix {
        SparseBinaryMatrix::new(0, 0, Vec::new())
    }

    fn ty(k_log: usize, useful_bits: usize) -> TableType {
        TableType {
            k_log,
            useful_bits,
            a_0: stub(),
            b_0: stub(),
            c_0: stub(),
            const_pin: None,
            class: TableClass::Boolean,
            io_schema: Vec::new(),
        }
    }

    /// `eq_prefix_weight` against the eq-table entry it freezes.
    #[test]
    fn eq_prefix_weight_matches_eq_table() {
        let mut rng = Rng::new(0xE0_11);
        for d in 1..=6usize {
            let coords = rng.f128_vec(d);
            let table = build_eq_table(&coords);
            for bits in 0..(1usize << d) {
                assert_eq!(
                    eq_prefix_weight(&coords, bits),
                    table[bits],
                    "d={d}, bits={bits}"
                );
            }
        }
        // Empty coordinate set: the weight is the empty product.
        assert_eq!(eq_prefix_weight(&[], 0), F128::ONE);
    }

    /// `eq_prefix_sum` against the naive eq-table prefix sum, including the
    /// n = 0 and n = 2^d edges (the latter must be exactly ONE).
    #[test]
    fn eq_prefix_sum_matches_naive() {
        let mut rng = Rng::new(0xE0_55);
        for d in 1..=6usize {
            let point = rng.f128_vec(d);
            let table = build_eq_table(&point);
            let mut naive = F128::ZERO;
            for n in 0..=(1usize << d) {
                assert_eq!(eq_prefix_sum(&point, n), naive, "d={d}, n={n}");
                if n < 1usize << d {
                    naive += table[n];
                }
            }
            assert_eq!(eq_prefix_sum(&point, 1 << d), F128::ONE);
        }
    }

    /// The one-pass multi-point bit-bank fold equals the stripe fold
    /// [`union_bitbank_fold`] at every point, on one- and two-slot registries with
    /// partial counts and partial last columns, under every kernel mask.
    #[test]
    fn union_bitbank_fold_multi_matches_stripe_fold() {
        use crate::{
            lincheck::{SparseMatrixCircuit, UnionLincheckSlot, pack_z_lincheck, union_bitbank_fold},
            lincheck::union::union_bitbank_fold_multi,
            simd::{test_masks, with_mask},
            union::UnionInstance,
        };
        let mut rng = Rng::new(0xBA_4C);
        let st = stub();
        for nu in 3..=6usize {
            for k_log in 7..=9usize {
                let cap = 1usize << nu;
                let k = 1usize << k_log;
                let useful = k - 37;
                let regs = [
                    (Registry::new(vec![ty(k_log, useful)], nu), vec![cap - 3]),
                    (Registry::new(vec![ty(k_log, useful), ty(k_log, k - 128)], nu), vec![cap, 5]),
                    (Registry::new(vec![ty(k_log, k)], nu), vec![cap]),
                ];
                for (reg, counts) in regs {
                    let union = UnionInstance::new(&reg, counts.clone());
                    let mut z = vec![F128::ZERO; union.packed_len()];
                    // Random bits on the declared rows and live bits of every slot.
                    let mut stripes = Vec::new();
                    for ((t, slot), &n_t) in reg.boolean_types().iter().zip(reg.slots()).zip(&counts) {
                        let base = slot.offset >> 7;
                        let mut logical = vec![false; k << nu];
                        for c in 0..k / 128 {
                            for j in 0..n_t {
                                let mut w = F128::new(rng.next_u64(), rng.next_u64());
                                for b in 0..128 {
                                    if 128 * c + b >= t.useful_bits {
                                        if b < 64 { w.lo &= !(1 << b) } else { w.hi &= !(1 << (b - 64)) }
                                    }
                                }
                                z[base + (c << nu) + j] = w;
                                for b in 0..128 {
                                    let bit = if b < 64 { (w.lo >> b) & 1 } else { (w.hi >> (b - 64)) & 1 };
                                    logical[128 * c + b + j * k] = bit == 1;
                                }
                            }
                        }
                        stripes.push(pack_z_lincheck(&logical, nu + k_log, k_log));
                    }
                    let circ = SparseMatrixCircuit::new(&st, &st);
                    let slots: Vec<UnionLincheckSlot<'_>> = stripes
                        .iter()
                        .map(|s| UnionLincheckSlot { z_lincheck: s, circuit: &circ })
                        .collect();
                    // One to five points: table strides 1, 2, 4, 4 and 8.
                    let points: Vec<Vec<F128>> =
                        (0..1 + (nu + k_log) % 5).map(|_| rng.f128_vec(nu)).collect();
                    let want: Vec<Vec<F128>> =
                        points.iter().map(|p| union_bitbank_fold(&union, &slots, p)).collect();
                    let refs: Vec<&[F128]> = points.iter().map(Vec::as_slice).collect();
                    for mask in test_masks() {
                        let got = with_mask(mask, || union_bitbank_fold_multi(&union, &z, &refs));
                        assert_eq!(got, want, "nu {nu} k_log {k_log} mask {mask:#x}");
                    }
                }
            }
        }
    }

    /// Single-thread micro-benchmark of the one-pass bit-bank fold at a 32K-like row
    /// geometry (`nu = 15`: 512 KiB between the columns of a word column) with four row
    /// points, 64 live word columns. Run with
    /// `RAYON_NUM_THREADS=1 taskset -c 3 cargo test --release bitbank_fold_multi_bench
    /// -- --ignored --nocapture`; prints the best of five runs and an output checksum.
    #[test]
    #[ignore]
    fn bitbank_fold_multi_bench() {
        use crate::{lincheck::union::union_bitbank_fold_multi, union::UnionInstance};
        let (nu, k_log, np) = (15usize, 13usize, 4usize);
        let reg = Registry::new(vec![ty(k_log, 1 << k_log)], nu);
        let union = UnionInstance::new(&reg, vec![1 << nu]);
        let mut rng = Rng::new(0xBE_4C);
        let mut z = vec![F128::ZERO; union.packed_len()];
        for w in z.iter_mut() {
            *w = F128::new(rng.next_u64(), rng.next_u64());
        }
        let points: Vec<Vec<F128>> = (0..np).map(|_| rng.f128_vec(nu)).collect();
        let refs: Vec<&[F128]> = points.iter().map(Vec::as_slice).collect();
        let mut best = f64::MAX;
        let mut sum = F128::ZERO;
        for _ in 0..5 {
            let t = std::time::Instant::now();
            let outs = union_bitbank_fold_multi(&union, &z, &refs);
            best = best.min(t.elapsed().as_secs_f64() * 1e3);
            sum = outs.iter().flatten().fold(F128::ZERO, |a, &x| a + x);
        }
        eprintln!("bitbank_fold_multi: best of 5 {best:.1} ms, checksum {:x}{:x}", sum.hi, sum.lo);
    }

    /// The closed-form [`union_comb_partial`] against a dense union-column
    /// vector folded with the full eq tensor, on a two-type registry with
    /// random per-type "comb" vectors (no circuits needed — the closed form
    /// is pure placement algebra).
    #[test]
    fn union_comb_partial_matches_dense_fold() {
        let k_skip = K_SKIP;
        // κ = 9/8, ν = 2 → areas 2^11 + 2^10, M = 12, column domain 2^10.
        let reg = Registry::new(vec![ty(9, 1 << 9), ty(8, 1 << 8)], 2);
        assert_eq!(reg.m_total(), 12);
        let col_vars = reg.m_total() - reg.nu();
        let mut rng = Rng::new(0xC0_3B);

        let combs: Vec<Vec<F128>> = reg
            .types()
            .iter()
            .map(|t| rng.f128_vec(1usize << t.k_log))
            .collect();
        let rr = rng.f128_vec(col_vars - k_skip);

        // Dense: place each comb at its slot's aligned column offset, fold
        // the multilinear coords with the full eq tensor.
        let mut dense = vec![F128::ZERO; 1usize << col_vars];
        for ((t, slot), comb) in reg.types().iter().zip(reg.slots()).zip(&combs) {
            let off = slot.prefix << t.k_log;
            dense[off..off + comb.len()].copy_from_slice(comb);
        }
        let eq_rr = build_eq_table(&rr);
        let n_skip = 1usize << k_skip;
        let mut expected = vec![F128::ZERO; n_skip];
        for (j, &e) in eq_rr.iter().enumerate() {
            for (i, x) in expected.iter_mut().enumerate() {
                *x += e * dense[(j << k_skip) + i];
            }
        }

        assert_eq!(union_comb_partial(&reg, &combs, &rr, k_skip), expected);
    }
}
