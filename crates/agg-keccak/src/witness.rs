//! Batch-major packing of gate instances into one Flock union slot.
//!
//! Flock's union stores a slot's witness BatchMajor: 128-bit word `c` of
//! instance `j` is union word `(c << nu) + j`. The lincheck reads `z` itself (one
//! bit-transposing pass for all its row points, `flock_core::lincheck::union_bitbank_fold_multi`);
//! only the legacy single-proof path ([`SlotBuffers::into_parts`]) builds the byte stripe
//! of `pack_z_lincheck`. Rows `[n, 2^nu)` are dummy and identically zero in `z`, `a`, `b`.

use agg_mldsa_simd::{keccak::Words8, Engine};
use agg_witness::params::ParamSet;
use flock_core::alloc_zeroed_vec;
use flock_field::F128;
use rayon::prelude::*;

use crate::gate::ChainGate;
use crate::keccak::{Lanes, N_LANES};
use crate::sponge::SpongeRecord;

/// Instances whose rows are generated together before scattering.
const BATCH: usize = 1024;

/// One slot's prover buffers: the committed `z` and, when already built, the row values
/// `a = A z`, `b = B z` (all BatchMajor). The 8-way sponge generator leaves `ab` empty;
/// [`Self::ensure_ab`] fills it from `z` where the proof needs it.
pub struct SlotBuffers {
    pub z: std::sync::Arc<Vec<F128>>,
    pub ab: Option<(Vec<F128>, Vec<F128>)>,
}

impl SlotBuffers {
    /// Fill `ab` from the committed `z` ([`pack_ab`]) unless present (`n` declared rows of
    /// capacity `2^nu`).
    pub fn ensure_ab(&mut self, gate: &ChainGate, nu: usize, n: usize) {
        if self.ab.is_none() {
            self.ab = Some(pack_ab(gate, nu, n, &self.z));
        }
    }

    /// `(z, a, b)`.
    ///
    /// # Panics
    /// If `a`, `b` were not built ([`Self::ensure_ab`]).
    #[must_use]
    pub fn into_zab(self) -> (Vec<F128>, Vec<F128>, Vec<F128>) {
        let (a, b) = self
            .ab
            .expect("row values a, b not built (SlotBuffers::ensure_ab)");
        (std::sync::Arc::unwrap_or_clone(self.z), a, b)
    }

    /// The tuple `UnionSlotProverInput::new` takes, with the lincheck stripe of `z` built
    /// here (block width `2^k_log` bits, `2^nu` rows).
    ///
    /// # Panics
    /// As [`Self::into_zab`].
    #[must_use]
    pub fn into_parts(self, k_log: usize, nu: usize) -> (Vec<F128>, Vec<F128>, Vec<F128>, Vec<u8>) {
        let stripe = lincheck_stripe(&self.z, k_log, nu);
        let (z, a, b) = self.into_zab();
        (z, a, b, stripe)
    }
}

/// The row values `a = A z`, `b = B z` of the committed slot buffer `z` (`n` declared rows
/// of capacity `2^nu`), with [`ChainGate::apply_ab`]'s semantics on every row, tampered
/// ones included (the committed chi bits drive the state). Eight rows per task for the
/// sponge gate on AVX-512 (`crate::simd::avx512`) or AVX2 (`crate::simd::avx2`); otherwise
/// per row.
///
/// # Panics
/// If `z` has the wrong length, `n > 2^nu` or `nu < 3`.
#[must_use]
pub fn pack_ab(gate: &ChainGate, nu: usize, n: usize, z: &[F128]) -> (Vec<F128>, Vec<F128>) {
    let lay = gate.layout();
    assert!(nu >= 3 && n <= 1 << nu, "row geometry");
    assert_eq!(
        z.len(),
        (1usize << (lay.k_log - 7)) << nu,
        "slot buffer length"
    );
    #[cfg(target_arch = "x86_64")]
    if let Some(ab) = pack_ab_8way(gate, nu, n, z) {
        return ab;
    }
    // Per row: gather the row's words, apply the row maps, scatter (pack_slot's batches).
    let packed = pack_slot(gate, nu, n, |j, zz, a, b| {
        for (u, w) in zz.iter_mut().enumerate() {
            let word = z[((u / 2) << nu) + j];
            *w = if u % 2 == 0 { word.lo } else { word.hi };
        }
        gate.apply_ab(zz, a, b);
    });
    packed.ab.expect("pack_slot builds a, b")
}

/// [`pack_ab`] on the 8-way kernels: AVX-512 first, then AVX2, for a sponge gate of any
/// parameter set; `None` when neither applies.
#[cfg(target_arch = "x86_64")]
fn pack_ab_8way(gate: &ChainGate, nu: usize, n: usize, z: &[F128]) -> Option<(Vec<F128>, Vec<F128>)> {
    use agg_witness::params::Set;
    use gen8::{Geo, Geo44, Geo65, Geo87};
    let set = gen8::sponge_set(gate)?;
    let (avx512, avx2) = (crate::simd::avx512(), crate::simd::avx2());
    macro_rules! run {
        ($g:ty) => {{
            const IN_Q: usize = <$g as Geo>::IN_Q;
            if avx512 {
                Some(gen8::pack_ab::<$g, IN_Q>(gate, nu, n, z))
            } else if avx2 {
                Some(gen8x2::pack_ab::<$g, IN_Q>(gate, nu, n, z))
            } else {
                None
            }
        }};
    }
    match set {
        Set::S44 => run!(Geo44),
        Set::S65 => run!(Geo65),
        Set::S87 => run!(Geo87),
    }
}

/// The byte stripe of `flock_core::lincheck::pack_z_lincheck` of a BatchMajor slot buffer:
/// group `g` of eight rows is `2^k_log` bytes, byte `128 c + b` holding bit `b` of word
/// column `c` of rows `8 g .. 8 g + 8` (bit `r` = row `8 g + r`).
///
/// # Panics
/// If `nu < 3` or `z` has another length than `2^(k_log - 7 + nu)`.
#[must_use]
pub fn lincheck_stripe(z: &[F128], k_log: usize, nu: usize) -> Vec<u8> {
    assert!(nu >= 3 && k_log >= 7, "stripe geometry");
    let k = 1usize << k_log;
    assert_eq!(z.len(), (k / 128) << nu, "slot buffer length");
    let mut stripe: Vec<u8> = alloc_zeroed_vec(k << (nu - 3));
    stripe.par_chunks_mut(k).enumerate().for_each(|(g, out)| {
        for c in 0..k / 128 {
            let run = &z[(c << nu) + 8 * g..(c << nu) + 8 * g + 8];
            if run.iter().all(|w| *w == F128::ZERO) {
                continue;
            }
            let lo: [u64; 8] = core::array::from_fn(|r| run[r].lo);
            let hi: [u64; 8] = core::array::from_fn(|r| run[r].hi);
            transpose_8x64(&lo, &mut out[128 * c..128 * c + 64]);
            transpose_8x64(&hi, &mut out[128 * c + 64..128 * c + 128]);
        }
    });
    stripe
}

/// Where the 8-way sponge generator writes each record's lane-major input bits (the chain's
/// copy of the w1Encode bits, `crate::tied::TiedPart::honest_parts`): record `g` is position
/// `i = g mod 2^d` of segment `sigma = g >> d`, and its lane `t` is the `256 k / 64` live
/// words at `sigma seg_words + (lane_len t + P_pad i) / 64` (the padded positions are left
/// as they are, zero).
pub struct LaneBits<'a> {
    pub bits: &'a mut [u64],
    pub seg_words: usize,
    pub lane_len: usize,
    pub d: u32,
}

/// The honest slot buffers of the SHAKE256 sponge gate of `P` (`crate::sponge::sponge_gate_for`)
/// for `records`, eight records per zmm (lane-sliced Keccak, `crate::simd::keccak8`), and the
/// lane-major input bits when `lane_bits` is given. Every buffer is byte-identical to
/// [`pack_slot`] with [`ChainGate::instance_rows`] on [`SpongeRecord::input_bits`] (the
/// oracle). Runs the AVX-512 kernels (`crate::simd::avx512`) or else the AVX2 ones
/// (`crate::simd::avx2`); `None` when both are unavailable or disabled, and the caller then
/// runs the oracle path.
///
/// # Panics
/// If `gate` is not the sponge gate, `records.len() > 2^nu`, `nu < 3`, or `lane_bits` is too
/// short for the records.
#[must_use]
pub fn sponge_slot<P: ParamSet>(
    gate: &ChainGate,
    nu: usize,
    records: &[SpongeRecord<P>],
    lane_bits: Option<LaneBits<'_>>,
) -> Option<SlotBuffers> {
    #[cfg(target_arch = "x86_64")]
    {
        use agg_witness::params::Set;
        use gen8::{Geo, Geo44, Geo65, Geo87};
        let (avx512, avx2) = (crate::simd::avx512(), crate::simd::avx2());
        // One monomorphised kernel per parameter set; the set is a constant of `P`.
        macro_rules! run {
            ($g:ty) => {{
                const IN_Q: usize = <$g as Geo>::IN_Q;
                const MSG_Q: usize = <$g as Geo>::MSG_Q;
                let recs = gen8::records::<P, $g>(records);
                if avx512 {
                    Some(gen8::sponge_slot::<$g, IN_Q, MSG_Q>(gate, nu, recs, lane_bits))
                } else if avx2 {
                    Some(gen8x2::sponge_slot::<$g, IN_Q, MSG_Q>(gate, nu, recs, lane_bits))
                } else {
                    None
                }
            }};
        }
        match P::SET {
            Set::S44 => run!(Geo44),
            Set::S65 => run!(Geo65),
            Set::S87 => run!(Geo87),
        }
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        let _ = (gate, nu, records, lane_bits);
        None
    }
}

/// The 8-way sponge witness generator, one monomorphised instance per parameter set
/// ([`gen8::Geo`]). Per batch of eight records (one rayon task): the committed lane words of
/// w1Encode are extracted with BMI2 `pext` (stride-6 masks for six-bit coefficients, nibble
/// masks for four-bit ones) and the absorbed message lanes loaded as they are, both into qword planes (qword `r` = record
/// `8 g + r`); the permutations run on [`State8`](crate::simd::keccak8::State8) and
/// every round's chi lanes are stored as soon as they exist: word column `c` of the eight
/// records is the 128-byte run `z[(c << nu) + 8 g ..][..8]`, built from a lo and a hi
/// plane by two `vpermt2q`. The tail batch (fewer than eight records) uses masked stores,
/// so dummy rows stay zero. Lane words that hold only padded positions (ML-DSA-65: words
/// 12..15 of each lane) are not stored: no input sits there, `z` is zero and the gate pins
/// them. [`pack_ab`] recomputes the row values from the committed `z`
/// the same way (eight rows per task, the committed chi words driving the state).
#[cfg(target_arch = "x86_64")]
#[allow(unsafe_code)]
mod gen8 {
    use core::arch::x86_64::{
        __m512i, __mmask8, _mm512_loadu_si512, _mm512_mask_storeu_epi64,
        _mm512_permutex2var_epi64, _mm512_set1_epi64, _mm512_set_epi64, _mm512_setzero_si512,
        _mm512_storeu_si512, _pext_u64,
    };

    use flock_core::alloc_zeroed_vec;
    use flock_field::F128;
    use rayon::prelude::*;

    use agg_witness::params::{ParamSet, Set, Set44, Set65, Set87};

    use super::{LaneBits, SlotBuffers};
    use crate::gate::ChainGate;
    use crate::keccak::N_LANES;
    use crate::simd::keccak8::{absorb8, phi8, round8, zero8};
    use crate::sponge::{SpongeLayout, SpongeRecord, MU_BYTES, RATE_BYTES};

    /// Rate lanes per permutation.
    pub(super) const RATE_Q: usize = RATE_BYTES / 8;
    /// Rounds per permutation.
    pub(super) const ROUNDS: usize = 24;

    /// The block geometry of one parameter set's sponge gate ([`SpongeLayout`]) as
    /// constants, so that each set's kernels are a separate monomorphised instance with no
    /// run-time layout branch. Qword `u` of a record's committed block is bits `64 u ..`.
    pub(crate) trait Geo: 'static {
        /// The parameter set of the records.
        type P: ParamSet;
        /// The sponge layout.
        const L: SpongeLayout;
        /// Bit lanes (bits per w1 coefficient).
        const LANES: usize = Self::L.lanes;
        /// Padded positions per lane, `P_pad`.
        const P_PAD: usize = Self::L.positions_pad;
        /// Committed qwords per lane, padded positions included.
        const LANE_Q: usize = Self::L.positions_pad / 64;
        /// Qwords per lane that hold live positions (`256 k / 64`).
        const LIVE_Q: usize = Self::L.positions() / 64;
        /// Qwords of w1Encode.
        const W1_Q: usize = Self::L.w1_bytes / 8;
        /// Committed input qwords: the lanes (padded) then mu.
        const IN_Q: usize = Self::LANES * Self::LANE_Q + MU_BYTES / 8;
        /// Message qwords `mu || w1Encode` (the padding is added as constants).
        const MSG_Q: usize = (MU_BYTES + Self::L.w1_bytes) / 8;
        /// Word column of the first `c~` word.
        const CT_WORD: usize = Self::L.ctilde_word_base;
        /// `c~` words (two state lanes each).
        const CT_WORDS: usize = Self::L.ctilde_bytes / 16;
        /// Word column of the constant one.
        const CONST_WORD: usize = Self::L.const_word;
        /// First chi qword of a record (`GateLayout::t_base`).
        const T_BASE: usize = 2 * Self::L.chi_word_base;
        /// Permutations per sponge.
        const N_PERMS: usize = Self::L.n_perms;
        /// Committed lanes of the last round of the last permutation (the lanes `c~` reads).
        const LAST_LANES: usize = Self::L.ctilde_bytes / 8;
    }

    /// ML-DSA-44: six lanes of 16 qwords, seven permutations, 256-bit `c~`.
    pub(crate) struct Geo44;
    impl Geo for Geo44 {
        type P = Set44;
        const L: SpongeLayout = SpongeLayout::S44;
    }
    /// ML-DSA-65: four lanes of 32 qwords (24 live), seven permutations, 384-bit `c~`.
    pub(crate) struct Geo65;
    impl Geo for Geo65 {
        type P = Set65;
        const L: SpongeLayout = SpongeLayout::S65;
    }
    /// ML-DSA-87: four lanes of 32 qwords, nine permutations, 512-bit `c~`.
    pub(crate) struct Geo87;
    impl Geo for Geo87 {
        type P = Set87;
        const L: SpongeLayout = SpongeLayout::S87;
    }

    /// `records` as records of `G`'s parameter set.
    ///
    /// # Panics
    /// If `P` is another parameter set than `G::P`.
    #[allow(unsafe_code)]
    pub(super) fn records<P: ParamSet, G: Geo>(records: &[SpongeRecord<P>]) -> &[SpongeRecord<G::P>] {
        assert!(
            std::any::TypeId::of::<P>() == std::any::TypeId::of::<G::P>(),
            "records of the kernel's parameter set"
        );
        // SAFETY: `P` is `G::P`, so both slices have the same element type.
        unsafe { &*(std::ptr::from_ref(records) as *const [SpongeRecord<G::P>]) }
    }

    /// Whether committed word column `c < IN_Q / 2` is an input word: mu, or a lane word with
    /// a live position. Always true for ML-DSA-44 and ML-DSA-87 (a constant per instance).
    #[inline(always)]
    pub(super) const fn input_word<G: Geo>(c: usize) -> bool {
        2 * c >= G::LANES * G::LANE_Q || (2 * c) % G::LANE_Q < G::LIVE_Q
    }

    /// The parameter set whose 8-way kernels implement `gate`, if any.
    pub(super) fn sponge_set(gate: &ChainGate) -> Option<Set> {
        let layout = SpongeLayout::of_gate(gate)?;
        let ok = match layout.set {
            Set::S44 => is_sponge::<Geo44>(gate),
            Set::S65 => is_sponge::<Geo65>(gate),
            Set::S87 => is_sponge::<Geo87>(gate),
        };
        ok.then_some(layout.set)
    }

    /// `pext` masks: committed qword `16 t + 4 rho + jq` holds stream bits `1536 rho + 384 jq
    /// + 6 b + t` (`b < 64`, `crate::sponge::w1_column`), i.e. the bits `== t (mod 6)` of the
    /// 384-bit chunk at byte `192 rho + 48 jq`. `MASKS[t][w]` selects them in chunk qword `w`,
    /// `SHIFTS[t][w]` is the number of such bits in qwords `0..w`.
    const MASKS: [[u64; 6]; 6] = {
        let mut m = [[0u64; 6]; 6];
        let mut t = 0;
        while t < 6 {
            let mut p = t;
            while p < 384 {
                m[t][p / 64] |= 1 << (p % 64);
                p += 6;
            }
            t += 1;
        }
        m
    };
    const SHIFTS: [[u32; 6]; 6] = {
        let mut s = [[0u32; 6]; 6];
        let mut t = 0;
        while t < 6 {
            let mut w = 1;
            while w < 6 {
                s[t][w] = s[t][w - 1] + MASKS[t][w - 1].count_ones();
                w += 1;
            }
            t += 1;
        }
        s
    };

    /// Raw output pointers shared by the batch tasks; batch `g` writes only rows `8 g ..
    /// 8 g + 8` of `z` and its records' lane-bit runs.
    pub(super) struct Out {
        pub(super) z: *mut F128,
        pub(super) nu: usize,
        pub(super) bits: *mut u64,
        pub(super) seg_words: usize,
        pub(super) lane_len: usize,
        pub(super) d: u32,
    }
    // SAFETY: the tasks write disjoint regions (struct docs); the buffers outlive the scope.
    unsafe impl Sync for Out {}

    pub(super) fn sponge_slot<G: Geo, const IN_Q: usize, const MSG_Q: usize>(
        gate: &ChainGate,
        nu: usize,
        records: &[SpongeRecord<G::P>],
        lane_bits: Option<LaneBits<'_>>,
    ) -> SlotBuffers {
        // SAFETY: `crate::simd::avx512` checked the features; batch `g` writes inside
        // the buffers (rows `8 g + r < n <= 2^nu`, lane bits bounded above) and disjointly
        // from the other batches (`Out`).
        sponge_slot_with::<G, _>(gate, nu, records, lane_bits, |recs, g, out| unsafe {
            batch::<G, IN_Q, MSG_Q>(recs, g, out)
        })
    }

    /// The buffers, checks and batch loop of [`sponge_slot`] around a batch kernel
    /// `kernel(recs, g, out)` that writes rows `8 g .. 8 g + recs.len()` of `z` and the
    /// records' lane-bit runs (shared by the AVX-512 and AVX2 generators).
    pub(super) fn sponge_slot_with<G: Geo, K>(
        gate: &ChainGate,
        nu: usize,
        records: &[SpongeRecord<G::P>],
        lane_bits: Option<LaneBits<'_>>,
        kernel: K,
    ) -> SlotBuffers
    where
        K: Fn(&[SpongeRecord<G::P>], usize, &Out) + Sync,
    {
        let lay = gate.layout();
        assert!(
            is_sponge::<G>(gate),
            "the 8-way generator implements the sponge gate only"
        );
        let n = records.len();
        assert!(nu >= 3, "Flock's lincheck fold needs at least 8 rows");
        assert!(n <= 1 << nu, "count exceeds the row capacity");
        let k = 1usize << lay.k_log;
        let len = (k / 128) << nu;
        let mut z: Vec<F128> = alloc_zeroed_vec(len);
        let (bits, seg_words, lane_len, d) = match lane_bits {
            Some(lb) => {
                assert!(lb.lane_len.is_multiple_of(64) && lb.lane_len >= G::P_PAD << lb.d);
                if n > 0 {
                    let g = n - 1;
                    let (sigma, i) = (g >> lb.d, g & ((1 << lb.d) - 1));
                    let end = sigma * lb.seg_words
                        + ((G::LANES - 1) * lb.lane_len + G::P_PAD * i) / 64
                        + G::LIVE_Q;
                    assert!(end <= lb.bits.len(), "lane bits too short");
                }
                (lb.bits.as_mut_ptr(), lb.seg_words, lb.lane_len, lb.d)
            }
            None => (core::ptr::null_mut(), 0, 0, 0),
        };
        // First touch of the zero pages, page-disjoint across threads: every batch writes
        // every used column, so without this all threads fault on the same pages at once.
        let used = (lay.useful_bits.div_ceil(128)) << nu;
        prefault(&mut z[..used]);
        let out = Out {
            z: z.as_mut_ptr(),
            nu,
            bits,
            seg_words,
            lane_len,
            d,
        };
        records
            .par_chunks(8)
            .enumerate()
            .for_each(|(g, recs)| kernel(recs, g, &out));
        SlotBuffers {
            z: std::sync::Arc::new(z),
            ab: None,
        }
    }

    /// Write one zero per 4 KiB page of `buf` (parallel over 2 MiB pieces), so the page
    /// faults happen here, each page in one thread.
    pub(super) fn prefault<T: Copy + Default + Send>(buf: &mut [T]) {
        let per_page = (4096 / core::mem::size_of::<T>()).max(1);
        buf.par_chunks_mut(512 * per_page).for_each(|piece| {
            for i in (0..piece.len()).step_by(per_page) {
                piece[i] = T::default();
            }
        });
    }

    #[inline]
    #[target_feature(enable = "avx512f")]
    unsafe fn ld(p: &[u64; 8]) -> __m512i {
        // SAFETY: `p` is 64 readable bytes.
        unsafe { _mm512_loadu_si512(p.as_ptr().cast()) }
    }

    #[inline]
    pub(super) fn le64(b: &[u8]) -> u64 {
        u64::from_le_bytes(b[..8].try_into().expect("8 bytes"))
    }

    /// The 96 committed w1Encode qwords of one ML-DSA-44 record (`SpongeRecord::input_bits`
    /// words `0..96`), 36 `pext` per 48-byte chunk.
    #[inline]
    #[target_feature(enable = "bmi2")]
    fn w1_columns(w1: &[u8; crate::sponge::W1_BYTES], out: &mut [u64]) {
        for rho in 0..4 {
            for jq in 0..4 {
                let base = 192 * rho + 48 * jq;
                let q: [u64; 6] = core::array::from_fn(|w| le64(&w1[base + 8 * w..]));
                for t in 0..6 {
                    let mut v = 0u64;
                    for w in 0..6 {
                        v |= _pext_u64(q[w], MASKS[t][w]) << SHIFTS[t][w];
                    }
                    out[16 * t + 4 * rho + jq] = v;
                }
            }
        }
    }

    /// Four-bit coefficients: committed qword `LANE_Q t + p` holds bit `t` of coefficients
    /// `64 p .. 64 p + 64` (stream order, one coefficient per lane position), gathered from
    /// the four LE qwords `4 p .. 4 p + 4` of w1Encode (16 nibbles each) by `pext` with the
    /// nibble mask of bit `t`. The padded qwords `LIVE_Q .. LANE_Q` of a lane are not written.
    #[inline]
    #[target_feature(enable = "bmi2")]
    fn nibble_columns<G: Geo>(w1: &[u8], out: &mut [u64]) {
        const NIB: u64 = 0x1111_1111_1111_1111;
        debug_assert!(w1.len() == 8 * G::W1_Q && G::W1_Q == 4 * G::LIVE_Q);
        for p in 0..G::LIVE_Q {
            let x: [u64; 4] = core::array::from_fn(|k| le64(&w1[32 * p + 8 * k..]));
            for t in 0..4 {
                let m = NIB << t;
                out[G::LANE_Q * t + p] = _pext_u64(x[0], m)
                    | _pext_u64(x[1], m) << 16
                    | _pext_u64(x[2], m) << 32
                    | _pext_u64(x[3], m) << 48;
            }
        }
    }

    /// The committed lane qwords of one record (`out[..LANES LANE_Q]`): stride-6 masks for
    /// ML-DSA-44, nibble masks for ML-DSA-65 and ML-DSA-87 (a constant choice per instance).
    #[inline]
    #[target_feature(enable = "bmi2")]
    pub(super) fn columns<G: Geo>(w1: &[u8], out: &mut [u64]) {
        if G::LANES == 6 {
            w1_columns(w1.try_into().expect("ML-DSA-44 w1Encode"), out);
        } else {
            nibble_columns::<G>(w1, out);
        }
    }

    /// Stores of one batch: word runs of `z` (or `a`, `b` in [`pack_ab`]).
    struct Emit {
        z: *mut F128,
        a: *mut F128,
        b: *mut F128,
        nu: usize,
        full: bool,
        m0: __mmask8,
        m1: __mmask8,
        idx0: __m512i,
        idx1: __m512i,
    }

    impl Emit {
        /// The stores of rows `8 g .. 8 g + cnt` (`z`, `a`, `b` already offset to row `8 g`;
        /// unused ones null).
        #[inline]
        #[target_feature(enable = "avx512f")]
        fn new(z: *mut F128, a: *mut F128, b: *mut F128, nu: usize, cnt: usize) -> Self {
            let half = |lo: usize, hi: usize| -> u8 {
                let mut m = 0u8;
                for r in lo..hi.min(cnt) {
                    m |= 0b11 << (2 * (r - lo));
                }
                m
            };
            Self {
                z,
                a,
                b,
                nu,
                full: cnt == 8,
                m0: half(0, 4),
                m1: half(4, 8),
                idx0: _mm512_set_epi64(11, 3, 10, 2, 9, 1, 8, 0),
                idx1: _mm512_set_epi64(15, 7, 14, 6, 13, 5, 12, 4),
            }
        }

        /// Word column `c` of the batch's eight rows in `buf` from its lo and hi planes:
        /// `[lo_0, hi_0, .., lo_3, hi_3]` and `[lo_4, hi_4, .., lo_7, hi_7]`.
        #[inline]
        #[target_feature(enable = "avx512f")]
        unsafe fn word(&self, buf: *mut F128, c: usize, lo: __m512i, hi: __m512i) {
            let w0 = _mm512_permutex2var_epi64(lo, self.idx0, hi);
            let w1 = _mm512_permutex2var_epi64(lo, self.idx1, hi);
            // SAFETY: rows 8 g .. 8 g + 8 of column c lie inside `buf` (caller).
            unsafe {
                let p = buf.add(c << self.nu);
                if self.full {
                    _mm512_storeu_si512(p.cast(), w0);
                    _mm512_storeu_si512(p.add(4).cast(), w1);
                } else {
                    _mm512_mask_storeu_epi64(p.cast(), self.m0, w0);
                    _mm512_mask_storeu_epi64(p.add(4).cast(), self.m1, w1);
                }
            }
        }
    }

    /// One record's committed input qwords (`q`, zero on entry: the lanes then mu) and its
    /// message qwords into plane `r` of `msg`, and its lane-major input bits when `o` asks
    /// for them (record `8 g + r`).
    #[inline(always)]
    pub(super) unsafe fn load_record<G: Geo, const IN_Q: usize, const MSG_Q: usize>(
        rec: &SpongeRecord<G::P>,
        q: &mut [u64; IN_Q],
        msg: &mut [[u64; 8]; MSG_Q],
        r: usize,
        g: usize,
        o: &Out,
    ) {
        let w1: &[u8] = rec.w1_encode.as_ref();
        let in_mu = G::LANES * G::LANE_Q;
        // SAFETY: the callers enable BMI2 (x86-64-v3 baseline).
        unsafe { columns::<G>(w1, &mut q[..in_mu]) };
        for i in 0..MU_BYTES / 8 {
            q[in_mu + i] = le64(&rec.mu[8 * i..]);
            msg[i][r] = q[in_mu + i];
        }
        for i in 0..G::W1_Q {
            msg[MU_BYTES / 8 + i][r] = le64(&w1[8 * i..]);
        }
        if !o.bits.is_null() {
            let gr = 8 * g + r;
            let (sigma, i) = (gr >> o.d, gr & ((1 << o.d) - 1));
            for t in 0..G::LANES {
                let dst = sigma * o.seg_words + (o.lane_len * t + G::P_PAD * i) / 64;
                // SAFETY: bounded by the assert in `sponge_slot_with` (records are in
                // order, the last one has the largest offset); runs of distinct records
                // are disjoint.
                unsafe {
                    core::ptr::copy_nonoverlapping(
                        q.as_ptr().add(G::LANE_Q * t),
                        o.bits.add(dst),
                        G::LIVE_Q,
                    );
                }
            }
        }
    }

    /// One batch of `1..=8` records (`recs`, rows `8 g ..`).
    #[target_feature(enable = "avx512f,avx512bw,avx512vbmi,gfni,bmi2")]
    unsafe fn batch<G: Geo, const IN_Q: usize, const MSG_Q: usize>(
        recs: &[SpongeRecord<G::P>],
        g: usize,
        o: &Out,
    ) {
        const { assert!(IN_Q == G::IN_Q && MSG_Q == G::MSG_Q) };
        let cnt = recs.len();
        debug_assert!((1..=8).contains(&cnt));
        // Planes: committed qwords `[u][r]`, message qwords `[q][r]`.
        let mut cols = [[0u64; 8]; IN_Q];
        let mut msg = [[0u64; 8]; MSG_Q];
        for (r, rec) in recs.iter().enumerate() {
            let mut q = [0u64; IN_Q];
            // SAFETY: BMI2 is enabled here; `o` as for the batch.
            unsafe { load_record::<G, IN_Q, MSG_Q>(rec, &mut q, &mut msg, r, g, o) };
            for u in 0..IN_Q {
                cols[u][r] = q[u];
            }
        }
        let row0 = 8 * g;
        // SAFETY: row0 + 8 <= 2^nu (caller).
        let em = unsafe {
            Emit::new(
                o.z.add(row0),
                core::ptr::null_mut(),
                core::ptr::null_mut(),
                o.nu,
                cnt,
            )
        };
        let zero = _mm512_setzero_si512();
        // SAFETY (all stores below): inside the batch's rows of `z` (`Emit`).
        unsafe {
            // Input words: the lanes (ML-DSA-44: w1Encode 0..47) and mu (44: 48..51).
            for c in 0..IN_Q / 2 {
                if input_word::<G>(c) {
                    em.word(em.z, c, ld(&cols[2 * c]), ld(&cols[2 * c + 1]));
                }
            }
            // The permutations; chi qword u = T_BASE + (24 m + r) 25 + L, a word per pair.
            let mut s = zero8();
            let mut pend = zero;
            let mut u = G::T_BASE;
            for m in 0..G::N_PERMS {
                let mut block = [zero; RATE_Q];
                for (l, p) in block.iter_mut().enumerate() {
                    let q = RATE_Q * m + l;
                    if q < MSG_Q {
                        *p = ld(&msg[q]);
                    }
                }
                if m == G::N_PERMS - 1 {
                    // SHAKE256 padding: 0x1F after the message, 0x80 at the end of the rate.
                    let first = MSG_Q - RATE_Q * m;
                    block[first] = _mm512_set1_epi64(0x1f);
                    block[RATE_Q - 1] = _mm512_set1_epi64((0x80u64 << 56) as i64);
                }
                absorb8(&mut s, &block);
                for r in 0..ROUNDS {
                    let t = round8(&mut s, r);
                    let lanes = if m == G::N_PERMS - 1 && r == ROUNDS - 1 {
                        G::LAST_LANES
                    } else {
                        N_LANES
                    };
                    for l in 0..lanes {
                        if u % 2 == 0 {
                            pend = t[l];
                        } else {
                            em.word(em.z, u / 2, pend, t[l]);
                        }
                        u += 1;
                    }
                }
            }
            if u % 2 != 0 {
                em.word(em.z, u / 2, pend, zero);
            }
            // c~ = the low `CT_WORDS` words of the final state (ML-DSA-44: words 52, 53).
            for w in 0..G::CT_WORDS {
                em.word(em.z, G::CT_WORD + w, s[2 * w], s[2 * w + 1]);
            }
            // The constant column (bit 0 of the word; ML-DSA-44: word 54).
            em.word(em.z, G::CONST_WORD, _mm512_set1_epi64(1), zero);
        }
    }

    /// Whether `gate` is the sponge gate of `G` the 8-way kernels implement: the input
    /// fields and absorptions of [`SpongeLayout`], the output words, the constant and the
    /// chi words where `G` puts them.
    pub(super) fn is_sponge<G: Geo>(gate: &ChainGate) -> bool {
        let lay = gate.layout();
        let spec = gate.spec();
        let mut field_base = Vec::new();
        let mut col = 0;
        for f in G::L.input_fields() {
            field_base.push(col);
            col += f.cols;
        }
        lay.k_log == 19
            && 2 * G::L.mu_word_base == G::LANES * G::LANE_Q
            && lay.t_base == G::T_BASE
            && lay.const_col == 128 * G::CONST_WORD
            && lay.output_base == 128 * G::CT_WORD
            && lay.field_base == field_base
            && spec.input_fields == G::L.input_fields()
            && spec.absorbs == crate::sponge::absorb_map_for(&G::L)
            && spec.n_rounds == ROUNDS
            && spec.n_perms() == G::N_PERMS
            && spec.n_inputs() == 64 * (G::LANES * G::LIVE_Q + MU_BYTES / 8)
            && spec
                .outputs
                .iter()
                .enumerate()
                .all(|(j, &o)| usize::from(o) == j)
            && spec.outputs.len() == 64 * G::LAST_LANES
            && lay.useful_bits
                == 64 * (G::T_BASE + (G::N_PERMS * ROUNDS - 1) * N_LANES + G::LAST_LANES)
    }

    /// One absorption step of [`pack_ab`]: `s[dst] ^= shift(plane[src] & mask)`, every
    /// absorbed state bit of a run with one shift between input and state qwords.
    #[derive(Clone, Copy)]
    pub(super) struct AbsorbRun {
        pub(super) dst: usize,
        pub(super) src: usize,
        /// `state bit - column bit` within the qwords.
        pub(super) shift: i32,
        pub(super) mask: u64,
    }

    /// Raw output pointers of [`pack_ab`]; task `g` writes rows `8 g .. 8 g + 8` only.
    pub(super) struct AbOut {
        pub(super) a: *mut F128,
        pub(super) b: *mut F128,
    }
    // SAFETY: the tasks write disjoint rows; the buffers outlive the scope.
    unsafe impl Sync for AbOut {}

    /// [`super::pack_ab`] for the sponge gate, eight rows per task: the committed words of
    /// the eight rows are read as 128-byte runs `z[(c << nu) + 8 g ..][..8]` and split
    /// into lo and hi qword planes (two `vpermt2q`); with `one` the committed constant bit
    /// per row (as an all-ones or all-zero mask), the state is rebuilt exactly as
    /// `ChainGate::apply_ab`: absorb the committed input bits (`AbsorbRun`s) and `one` at
    /// the padding bits, then per round `phi`, `a = phi[x + 1] ^ one`, `b = phi[x + 2]`,
    /// `state = phi ^ t` with the COMMITTED chi word `t`, `state[0] ^= RC & one`; input rows
    /// `a = z`, `b = one`, output rows `a` = final state, `b = one`, the constant row
    /// `a = b = const bit`. Rows past `n` stay zero.
    pub(super) fn pack_ab<G: Geo, const IN_Q: usize>(
        gate: &ChainGate,
        nu: usize,
        n: usize,
        z: &[F128],
    ) -> (Vec<F128>, Vec<F128>) {
        // SAFETY: `crate::simd::avx512` checked by the caller; rows 8 g .. 8 g + cnt of
        // `z`, `a`, `b` lie inside the buffers and belong to this task only.
        pack_ab_with::<G, _>(gate, nu, n, z, |g, cnt, runs, one_masks, out| unsafe {
            ab_batch::<G, IN_Q>(z, nu, g, cnt, runs, one_masks, out)
        })
    }

    /// The absorption plan, buffers and task loop of [`pack_ab`] around a kernel
    /// `kernel(g, cnt, runs, one_masks, out)` that writes rows `8 g .. 8 g + cnt` of `a`
    /// and `b` (shared by the AVX-512 and AVX2 generators).
    pub(super) fn pack_ab_with<G: Geo, K>(
        gate: &ChainGate,
        nu: usize,
        n: usize,
        z: &[F128],
        kernel: K,
    ) -> (Vec<F128>, Vec<F128>)
    where
        K: Fn(usize, usize, &[Vec<AbsorbRun>], &[[u64; N_LANES]], &AbOut) + Sync,
    {
        assert!(
            is_sponge::<G>(gate),
            "the 8-way kernels implement the sponge gate only"
        );
        let lay = gate.layout();
        // The kernel's input rows are the whole input words (`input_word`): every input
        // column lies in one, and there are as many inputs as columns in them.
        assert!(
            (0..gate.spec().n_inputs()).all(|i| {
                let col = lay.input_col(i);
                col < 64 * G::IN_Q && input_word::<G>(col / 128)
            }),
            "sponge inputs occupy the input words"
        );
        let (cols, ones) = gate.absorb_lists();
        // Absorption runs per permutation, grouped by (state qword, input qword, shift).
        let runs: Vec<Vec<AbsorbRun>> = cols
            .iter()
            .map(|list| {
                let mut out: Vec<AbsorbRun> = Vec::new();
                for &(st, col) in list {
                    assert!(col < 64 * G::IN_Q, "absorbed column inside the inputs");
                    let (dst, src) = (st / 64, col / 64);
                    let shift = (st % 64) as i32 - (col % 64) as i32;
                    match out
                        .iter_mut()
                        .find(|r| r.dst == dst && r.src == src && r.shift == shift)
                    {
                        Some(r) => r.mask |= 1 << (col % 64),
                        None => out.push(AbsorbRun {
                            dst,
                            src,
                            shift,
                            mask: 1 << (col % 64),
                        }),
                    }
                }
                out
            })
            .collect();
        let one_masks: Vec<[u64; N_LANES]> = ones
            .iter()
            .map(|list| {
                let mut m = [0u64; N_LANES];
                for &st in list {
                    m[st / 64] |= 1 << (st % 64);
                }
                m
            })
            .collect();
        let len = z.len();
        let mut a: Vec<F128> = alloc_zeroed_vec(len);
        let mut b: Vec<F128> = alloc_zeroed_vec(len);
        let used = (lay.useful_bits.div_ceil(128)) << nu;
        prefault(&mut a[..used]);
        prefault(&mut b[..used]);

        let out = AbOut {
            a: a.as_mut_ptr(),
            b: b.as_mut_ptr(),
        };
        (0..n.div_ceil(8)).into_par_iter().for_each(|g| {
            let cnt = (n - 8 * g).min(8);
            kernel(g, cnt, &runs, &one_masks, &out);
        });
        (a, b)
    }

    /// Rows `8 g .. 8 g + cnt` of [`pack_ab`].
    #[target_feature(enable = "avx512f,avx512bw")]
    unsafe fn ab_batch<G: Geo, const IN_Q: usize>(
        z: &[F128],
        nu: usize,
        g: usize,
        cnt: usize,
        runs: &[Vec<AbsorbRun>],
        one_masks: &[[u64; N_LANES]],
        out: &AbOut,
    ) {
        use core::arch::x86_64::{
            _mm512_and_si512, _mm512_set1_epi64, _mm512_sllv_epi64, _mm512_srlv_epi64,
            _mm512_sub_epi64, _mm512_xor_si512,
        };
        let row0 = 8 * g;
        let lo_idx = _mm512_set_epi64(14, 12, 10, 8, 6, 4, 2, 0);
        let hi_idx = _mm512_set_epi64(15, 13, 11, 9, 7, 5, 3, 1);
        // Lo and hi planes of word column c of the eight rows (rows past `cnt` hold the
        // buffer's dummy zeros or the next rows' words; their outputs are not stored).
        let planes = |c: usize| -> (__m512i, __m512i) {
            // SAFETY: 8 rows from row0 lie inside column c (8 g + 8 <= 2^nu).
            unsafe {
                let p = z.as_ptr().add((c << nu) + row0).cast::<u8>();
                let v0 = _mm512_loadu_si512(p.cast());
                let v1 = _mm512_loadu_si512(p.add(64).cast());
                (
                    _mm512_permutex2var_epi64(v0, lo_idx, v1),
                    _mm512_permutex2var_epi64(v0, hi_idx, v1),
                )
            }
        };
        // SAFETY: row0 + 8 <= 2^nu.
        let em = unsafe {
            Emit::new(
                core::ptr::null_mut(),
                out.a.add(row0),
                out.b.add(row0),
                nu,
                cnt,
            )
        };
        const { assert!(IN_Q == G::IN_Q) };
        let zero = _mm512_setzero_si512();
        let (const_lo, _) = planes(G::CONST_WORD);
        let bit = _mm512_and_si512(const_lo, _mm512_set1_epi64(1));
        // one = all ones where the committed constant bit is 1.
        let one = _mm512_sub_epi64(zero, bit);
        let mut inp = [zero; IN_Q];
        // SAFETY (all stores below): inside the task's rows of `a`, `b` (`Emit`).
        unsafe {
            // Input rows (no row, so a = b = 0, at the padded lane words).
            for c in 0..IN_Q / 2 {
                if input_word::<G>(c) {
                    let (lo, hi) = planes(c);
                    inp[2 * c] = lo;
                    inp[2 * c + 1] = hi;
                    em.word(em.a, c, lo, hi);
                    em.word(em.b, c, one, one);
                }
            }
            let mut s = zero8();
            let mut pend = (zero, zero);
            let mut tw = (zero, zero);
            let mut u = G::T_BASE;
            for (m, (runs_m, ones_m)) in runs.iter().zip(one_masks).enumerate() {
                for r in runs_m {
                    let v = _mm512_and_si512(inp[r.src], _mm512_set1_epi64(r.mask as i64));
                    let v = if r.shift >= 0 {
                        _mm512_sllv_epi64(v, _mm512_set1_epi64(i64::from(r.shift)))
                    } else {
                        _mm512_srlv_epi64(v, _mm512_set1_epi64(i64::from(-r.shift)))
                    };
                    s[r.dst] = _mm512_xor_si512(s[r.dst], v);
                }
                for (l, &mk) in ones_m.iter().enumerate() {
                    if mk != 0 {
                        s[l] = _mm512_xor_si512(
                            s[l],
                            _mm512_and_si512(one, _mm512_set1_epi64(mk as i64)),
                        );
                    }
                }
                for r in 0..ROUNDS {
                    let bl = phi8(&s);
                    for y in 0..5 {
                        for x in 0..5 {
                            let l = crate::keccak::lane(x, y);
                            if m == G::N_PERMS - 1 && r == ROUNDS - 1 && l >= G::LAST_LANES {
                                continue;
                            }
                            let ra = _mm512_xor_si512(bl[crate::keccak::lane((x + 1) % 5, y)], one);
                            let rb = bl[crate::keccak::lane((x + 2) % 5, y)];
                            // The committed chi qword u (lo or hi plane of word u / 2).
                            let t = if u % 2 == 0 {
                                tw = planes(u / 2);
                                tw.0
                            } else {
                                tw.1
                            };
                            s[l] = _mm512_xor_si512(bl[l], t);
                            if u % 2 == 0 {
                                pend = (ra, rb);
                            } else {
                                em.word(em.a, u / 2, pend.0, ra);
                                em.word(em.b, u / 2, pend.1, rb);
                            }
                            u += 1;
                        }
                    }
                    s[0] = _mm512_xor_si512(
                        s[0],
                        _mm512_and_si512(
                            one,
                            _mm512_set1_epi64(crate::keccak::ROUND_CONSTANTS[r] as i64),
                        ),
                    );
                }
            }
            if u % 2 != 0 {
                em.word(em.a, u / 2, pend.0, zero);
                em.word(em.b, u / 2, pend.1, zero);
            }
            // Output rows: a = the final state's low `c~` bits, b = one.
            for w in 0..G::CT_WORDS {
                em.word(em.a, G::CT_WORD + w, s[2 * w], s[2 * w + 1]);
                em.word(em.b, G::CT_WORD + w, one, one);
            }
            // The constant row: a = b = the committed constant bit.
            em.word(em.a, G::CONST_WORD, bit, zero);
            em.word(em.b, G::CONST_WORD, bit, zero);
        }
    }
}

/// The AVX2 form of the 8-way generator (`gen8`): the same batches of eight records, with
/// every eight-qword plane held in two ymm registers (qwords `0..4` and `4..8`). Rotations
/// are shift pairs, three-input XORs are two XORs, and a word run of `z`, `a` or `b` (eight
/// F128 rows) is built from its lo and hi planes by `vpunpck{l,h}qdq` and `vperm2i128`.
/// Every buffer is byte-identical to the AVX-512 generator's and to the portable oracle's.
#[cfg(target_arch = "x86_64")]
#[allow(unsafe_code)]
mod gen8x2 {
    use core::arch::x86_64::{
        __m256i, _mm256_and_si256, _mm256_andnot_si256, _mm256_loadu_si256, _mm256_maskstore_epi64,
        _mm256_or_si256, _mm256_permute2x128_si256, _mm256_set1_epi64x, _mm256_setr_epi64x,
        _mm256_setzero_si256, _mm256_sllv_epi64, _mm256_srlv_epi64, _mm256_storeu_si256,
        _mm256_sub_epi64, _mm256_unpackhi_epi64, _mm256_unpacklo_epi64, _mm256_xor_si256,
    };

    use flock_field::F128;

    use super::gen8::{self, input_word, load_record, AbOut, AbsorbRun, Geo, Out, RATE_Q, ROUNDS};
    use super::{LaneBits, SlotBuffers};
    use crate::gate::ChainGate;
    use crate::keccak::{lane, N_LANES, RHO_OFFSETS, ROUND_CONSTANTS};
    use crate::sponge::SpongeRecord;

    /// Eight qwords (qword `r` = record or row `8 g + r`) in two ymm registers.
    #[derive(Clone, Copy)]
    struct V([__m256i; 2]);

    impl V {
        #[inline]
        #[target_feature(enable = "avx2")]
        fn zero() -> Self {
            Self([_mm256_setzero_si256(); 2])
        }

        #[inline]
        #[target_feature(enable = "avx2")]
        fn splat(x: u64) -> Self {
            Self([_mm256_set1_epi64x(x as i64); 2])
        }

        #[inline]
        #[target_feature(enable = "avx2")]
        fn xor(self, o: Self) -> Self {
            Self([
                _mm256_xor_si256(self.0[0], o.0[0]),
                _mm256_xor_si256(self.0[1], o.0[1]),
            ])
        }

        #[inline]
        #[target_feature(enable = "avx2")]
        fn and(self, o: Self) -> Self {
            Self([
                _mm256_and_si256(self.0[0], o.0[0]),
                _mm256_and_si256(self.0[1], o.0[1]),
            ])
        }

        /// `!self & o`.
        #[inline]
        #[target_feature(enable = "avx2")]
        fn andnot(self, o: Self) -> Self {
            Self([
                _mm256_andnot_si256(self.0[0], o.0[0]),
                _mm256_andnot_si256(self.0[1], o.0[1]),
            ])
        }

        /// `0 - self` per qword.
        #[inline]
        #[target_feature(enable = "avx2")]
        fn neg(self) -> Self {
            let z = _mm256_setzero_si256();
            Self([
                _mm256_sub_epi64(z, self.0[0]),
                _mm256_sub_epi64(z, self.0[1]),
            ])
        }

        /// Left shift of every qword by `n < 64`.
        #[inline]
        #[target_feature(enable = "avx2")]
        fn shl(self, n: u32) -> Self {
            let c = _mm256_set1_epi64x(i64::from(n));
            Self([
                _mm256_sllv_epi64(self.0[0], c),
                _mm256_sllv_epi64(self.0[1], c),
            ])
        }

        /// Right shift of every qword by `n < 64`.
        #[inline]
        #[target_feature(enable = "avx2")]
        fn shr(self, n: u32) -> Self {
            let c = _mm256_set1_epi64x(i64::from(n));
            Self([
                _mm256_srlv_epi64(self.0[0], c),
                _mm256_srlv_epi64(self.0[1], c),
            ])
        }

        /// `rol(v, n)` on every qword, `n < 64`.
        #[inline]
        #[target_feature(enable = "avx2")]
        fn rol(self, n: u32) -> Self {
            if n == 0 {
                return self;
            }
            let (l, r) = (self.shl(n), self.shr(64 - n));
            Self([
                _mm256_or_si256(l.0[0], r.0[0]),
                _mm256_or_si256(l.0[1], r.0[1]),
            ])
        }

        /// The plane `p[0..8]`.
        #[inline]
        #[target_feature(enable = "avx2")]
        fn load(p: &[u64; 8]) -> Self {
            // SAFETY: `p` is 64 readable bytes.
            unsafe {
                Self([
                    _mm256_loadu_si256(p.as_ptr().cast()),
                    _mm256_loadu_si256(p.as_ptr().add(4).cast()),
                ])
            }
        }
    }

    type State = [V; N_LANES];

    /// `phi = pi o rho o theta` on eight states, as `crate::simd::keccak8::phi8`.
    #[inline]
    #[target_feature(enable = "avx2")]
    fn phi(s: &State) -> State {
        let mut c = [V::zero(); 5];
        for x in 0..5 {
            c[x] = s[x]
                .xor(s[x + 5])
                .xor(s[x + 10])
                .xor(s[x + 15])
                .xor(s[x + 20]);
        }
        let mut d = [V::zero(); 5];
        for x in 0..5 {
            d[x] = c[(x + 4) % 5].xor(c[(x + 1) % 5].rol(1));
        }
        let mut b = [V::zero(); N_LANES];
        for y in 0..5 {
            for x in 0..5 {
                let a = (x + 3 * y) % 5;
                b[lane(x, y)] = s[lane(a, x)].xor(d[a]).rol(RHO_OFFSETS[a][x]);
            }
        }
        b
    }

    /// One round with round index `r`: updates `s` and returns the chi AND outputs, as
    /// `crate::simd::keccak8::round8`.
    #[inline]
    #[target_feature(enable = "avx2")]
    fn round(s: &mut State, r: usize) -> State {
        let b = phi(s);
        let mut t = [V::zero(); N_LANES];
        for y in 0..5 {
            for x in 0..5 {
                let l = lane(x, y);
                t[l] = b[lane((x + 1) % 5, y)].andnot(b[lane((x + 2) % 5, y)]);
                s[l] = b[l].xor(t[l]);
            }
        }
        s[0] = s[0].xor(V::splat(ROUND_CONSTANTS[r]));
        t
    }

    /// Stores and loads of one batch's word runs (`z`, `a`, `b` offset to row `8 g`;
    /// unused ones null).
    struct Emit {
        z: *mut F128,
        a: *mut F128,
        b: *mut F128,
        nu: usize,
        full: bool,
        /// Store masks of the row pairs `(2 k, 2 k + 1)`, two qwords per row.
        masks: [__m256i; 4],
    }

    impl Emit {
        #[inline]
        #[target_feature(enable = "avx2")]
        fn new(z: *mut F128, a: *mut F128, b: *mut F128, nu: usize, cnt: usize) -> Self {
            let live = |r: usize| if r < cnt { -1i64 } else { 0 };
            let masks = core::array::from_fn(|k| {
                let (r0, r1) = (live(2 * k), live(2 * k + 1));
                _mm256_setr_epi64x(r0, r0, r1, r1)
            });
            Self {
                z,
                a,
                b,
                nu,
                full: cnt == 8,
                masks,
            }
        }

        /// Word column `c` of the batch's eight rows in `buf` from its lo and hi planes.
        #[inline]
        #[target_feature(enable = "avx2")]
        unsafe fn word(&self, buf: *mut F128, c: usize, lo: V, hi: V) {
            for h in 0..2 {
                // [lo_0, hi_0, lo_2, hi_2] and [lo_1, hi_1, lo_3, hi_3] of the half.
                let x0 = _mm256_unpacklo_epi64(lo.0[h], hi.0[h]);
                let x1 = _mm256_unpackhi_epi64(lo.0[h], hi.0[h]);
                let w0 = _mm256_permute2x128_si256::<0x20>(x0, x1);
                let w1 = _mm256_permute2x128_si256::<0x31>(x0, x1);
                // SAFETY: rows 8 g .. 8 g + 8 of column c lie inside `buf` (caller).
                unsafe {
                    let p = buf.add((c << self.nu) + 4 * h);
                    if self.full {
                        _mm256_storeu_si256(p.cast(), w0);
                        _mm256_storeu_si256(p.add(2).cast(), w1);
                    } else {
                        _mm256_maskstore_epi64(p.cast(), self.masks[2 * h], w0);
                        _mm256_maskstore_epi64(p.add(2).cast(), self.masks[2 * h + 1], w1);
                    }
                }
            }
        }
    }

    /// The lo and hi planes of word column `c` of the eight rows from `z` (offset to row
    /// `8 g`).
    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn planes(z: *const F128, nu: usize, c: usize) -> (V, V) {
        let mut lo = V::zero();
        let mut hi = V::zero();
        for h in 0..2 {
            // SAFETY: rows 8 g .. 8 g + 8 of column c lie inside `z` (caller).
            let (v0, v1) = unsafe {
                let p = z.add((c << nu) + 4 * h);
                (
                    _mm256_loadu_si256(p.cast()),
                    _mm256_loadu_si256(p.add(2).cast()),
                )
            };
            // v0 = [lo_0, hi_0, lo_1, hi_1], v1 = [lo_2, hi_2, lo_3, hi_3].
            let x0 = _mm256_permute2x128_si256::<0x20>(v0, v1);
            let x1 = _mm256_permute2x128_si256::<0x31>(v0, v1);
            lo.0[h] = _mm256_unpacklo_epi64(x0, x1);
            hi.0[h] = _mm256_unpackhi_epi64(x0, x1);
        }
        (lo, hi)
    }

    /// The AVX2 form of `gen8::sponge_slot` (same buffers).
    pub(super) fn sponge_slot<G: Geo, const IN_Q: usize, const MSG_Q: usize>(
        gate: &ChainGate,
        nu: usize,
        records: &[SpongeRecord<G::P>],
        lane_bits: Option<LaneBits<'_>>,
    ) -> SlotBuffers {
        // SAFETY: `crate::simd::avx2` checked the features; batch `g` writes inside the
        // buffers and disjointly from the other batches (`Out`).
        gen8::sponge_slot_with::<G, _>(gate, nu, records, lane_bits, |recs, g, out| unsafe {
            batch::<G, IN_Q, MSG_Q>(recs, g, out)
        })
    }

    /// The AVX2 form of `gen8::pack_ab` (same buffers).
    pub(super) fn pack_ab<G: Geo, const IN_Q: usize>(
        gate: &ChainGate,
        nu: usize,
        n: usize,
        z: &[F128],
    ) -> (Vec<F128>, Vec<F128>) {
        // SAFETY: `crate::simd::avx2` checked by the caller; rows 8 g .. 8 g + cnt of `z`,
        // `a`, `b` lie inside the buffers and belong to this task only.
        gen8::pack_ab_with::<G, _>(gate, nu, n, z, |g, cnt, runs, one_masks, out| unsafe {
            ab_batch::<G, IN_Q>(z, nu, g, cnt, runs, one_masks, out)
        })
    }

    /// One batch of `1..=8` records (`recs`, rows `8 g ..`), as `gen8::batch`.
    #[target_feature(enable = "avx2,bmi2")]
    unsafe fn batch<G: Geo, const IN_Q: usize, const MSG_Q: usize>(
        recs: &[SpongeRecord<G::P>],
        g: usize,
        o: &Out,
    ) {
        const { assert!(IN_Q == G::IN_Q && MSG_Q == G::MSG_Q) };
        let cnt = recs.len();
        debug_assert!((1..=8).contains(&cnt));
        let mut cols = [[0u64; 8]; IN_Q];
        let mut msg = [[0u64; 8]; MSG_Q];
        for (r, rec) in recs.iter().enumerate() {
            let mut q = [0u64; IN_Q];
            // SAFETY: BMI2 is enabled here; `o` as for the batch.
            unsafe { load_record::<G, IN_Q, MSG_Q>(rec, &mut q, &mut msg, r, g, o) };
            for u in 0..IN_Q {
                cols[u][r] = q[u];
            }
        }
        let row0 = 8 * g;
        // SAFETY: row0 + 8 <= 2^nu (caller).
        let em = unsafe {
            Emit::new(
                o.z.add(row0),
                core::ptr::null_mut(),
                core::ptr::null_mut(),
                o.nu,
                cnt,
            )
        };
        let zero = V::zero();
        // SAFETY (all stores below): inside the batch's rows of `z` (`Emit`).
        unsafe {
            for c in 0..IN_Q / 2 {
                if input_word::<G>(c) {
                    em.word(em.z, c, V::load(&cols[2 * c]), V::load(&cols[2 * c + 1]));
                }
            }
            let mut s: State = [zero; N_LANES];
            let mut pend = zero;
            let mut u = G::T_BASE;
            for m in 0..G::N_PERMS {
                for l in 0..RATE_Q {
                    let q = RATE_Q * m + l;
                    if q < MSG_Q {
                        s[l] = s[l].xor(V::load(&msg[q]));
                    }
                }
                if m == G::N_PERMS - 1 {
                    // SHAKE256 padding: 0x1F after the message, 0x80 at the end of the rate.
                    let first = MSG_Q - RATE_Q * m;
                    s[first] = s[first].xor(V::splat(0x1f));
                    s[RATE_Q - 1] = s[RATE_Q - 1].xor(V::splat(0x80u64 << 56));
                }
                for r in 0..ROUNDS {
                    let t = round(&mut s, r);
                    let lanes = if m == G::N_PERMS - 1 && r == ROUNDS - 1 {
                        G::LAST_LANES
                    } else {
                        N_LANES
                    };
                    for l in 0..lanes {
                        if u % 2 == 0 {
                            pend = t[l];
                        } else {
                            em.word(em.z, u / 2, pend, t[l]);
                        }
                        u += 1;
                    }
                }
            }
            if u % 2 != 0 {
                em.word(em.z, u / 2, pend, zero);
            }
            for w in 0..G::CT_WORDS {
                em.word(em.z, G::CT_WORD + w, s[2 * w], s[2 * w + 1]);
            }
            em.word(em.z, G::CONST_WORD, V::splat(1), zero);
        }
    }

    /// Rows `8 g .. 8 g + cnt` of [`pack_ab`], as `gen8::ab_batch`.
    #[target_feature(enable = "avx2")]
    unsafe fn ab_batch<G: Geo, const IN_Q: usize>(
        z: &[F128],
        nu: usize,
        g: usize,
        cnt: usize,
        runs: &[Vec<AbsorbRun>],
        one_masks: &[[u64; N_LANES]],
        out: &AbOut,
    ) {
        let row0 = 8 * g;
        // SAFETY: row0 + 8 <= 2^nu; rows past `cnt` are read (dummy zeros or the next
        // rows' words) but their outputs are not stored.
        let zr = unsafe { z.as_ptr().add(row0) };
        // SAFETY: row0 + 8 <= 2^nu.
        let em = unsafe {
            Emit::new(
                core::ptr::null_mut(),
                out.a.add(row0),
                out.b.add(row0),
                nu,
                cnt,
            )
        };
        const { assert!(IN_Q == G::IN_Q) };
        let zero = V::zero();
        // SAFETY (all loads and stores below): inside the task's rows (`Emit`, `planes`).
        unsafe {
            let (const_lo, _) = planes(zr, nu, G::CONST_WORD);
            let bit = const_lo.and(V::splat(1));
            // one = all ones where the committed constant bit is 1.
            let one = bit.neg();
            let mut inp = [zero; IN_Q];
            for c in 0..IN_Q / 2 {
                if input_word::<G>(c) {
                    let (lo, hi) = planes(zr, nu, c);
                    inp[2 * c] = lo;
                    inp[2 * c + 1] = hi;
                    em.word(em.a, c, lo, hi);
                    em.word(em.b, c, one, one);
                }
            }
            let mut s: State = [zero; N_LANES];
            let mut pend = (zero, zero);
            let mut tw = (zero, zero);
            let mut u = G::T_BASE;
            for (m, (runs_m, ones_m)) in runs.iter().zip(one_masks).enumerate() {
                for r in runs_m {
                    let v = inp[r.src].and(V::splat(r.mask));
                    let v = if r.shift >= 0 {
                        v.shl(r.shift as u32)
                    } else {
                        v.shr((-r.shift) as u32)
                    };
                    s[r.dst] = s[r.dst].xor(v);
                }
                for (l, &mk) in ones_m.iter().enumerate() {
                    if mk != 0 {
                        s[l] = s[l].xor(one.and(V::splat(mk)));
                    }
                }
                for r in 0..ROUNDS {
                    let bl = phi(&s);
                    for y in 0..5 {
                        for x in 0..5 {
                            let l = lane(x, y);
                            if m == G::N_PERMS - 1 && r == ROUNDS - 1 && l >= G::LAST_LANES {
                                continue;
                            }
                            let ra = bl[lane((x + 1) % 5, y)].xor(one);
                            let rb = bl[lane((x + 2) % 5, y)];
                            // The committed chi qword u (lo or hi plane of word u / 2).
                            let t = if u % 2 == 0 {
                                tw = planes(zr, nu, u / 2);
                                tw.0
                            } else {
                                tw.1
                            };
                            s[l] = bl[l].xor(t);
                            if u % 2 == 0 {
                                pend = (ra, rb);
                            } else {
                                em.word(em.a, u / 2, pend.0, ra);
                                em.word(em.b, u / 2, pend.1, rb);
                            }
                            u += 1;
                        }
                    }
                    s[0] = s[0].xor(one.and(V::splat(ROUND_CONSTANTS[r])));
                }
            }
            if u % 2 != 0 {
                em.word(em.a, u / 2, pend.0, zero);
                em.word(em.b, u / 2, pend.1, zero);
            }
            for w in 0..G::CT_WORDS {
                em.word(em.a, G::CT_WORD + w, s[2 * w], s[2 * w + 1]);
                em.word(em.b, G::CT_WORD + w, one, one);
            }
            em.word(em.a, G::CONST_WORD, bit, zero);
            em.word(em.b, G::CONST_WORD, bit, zero);
        }
    }
}

/// Pack `n <= 2^nu` instances. `rows(j, z, a, b)` writes instance `j`'s
/// committed bits and row values into zeroed slices of
/// [`crate::gate::GateLayout::useful_words`] words.
///
/// # Panics
/// If `n > 2^nu` or `nu < 3`.
pub fn pack_slot<F>(gate: &ChainGate, nu: usize, n: usize, rows: F) -> SlotBuffers
where
    F: Fn(usize, &mut [u64], &mut [u64], &mut [u64]) + Sync,
{
    let lay = gate.layout();
    let (k_log, useful_bits) = (lay.k_log, lay.useful_bits);
    assert!(nu >= 3, "Flock's lincheck fold needs at least 8 rows");
    assert!(n <= 1 << nu, "count exceeds the row capacity");
    assert!(k_log >= 7 && useful_bits <= 1 << k_log, "block geometry");
    let k = 1usize << k_log;
    let w = useful_bits.div_ceil(64);
    let useful_chunks = useful_bits.div_ceil(128);
    let len = (k / 128) << nu;
    let mut z: Vec<F128> = alloc_zeroed_vec(len);
    let mut a: Vec<F128> = alloc_zeroed_vec(len);
    let mut b: Vec<F128> = alloc_zeroed_vec(len);
    let mut scratch = vec![0u64; BATCH * 3 * w];
    // AGG_TIED_TRACE: time per phase (instance rows; BatchMajor scatter)
    // summed over the batches.
    let trace = std::env::var_os("AGG_TIED_TRACE").is_some();
    let mut phase_ms = [0f64; 2];
    // Columns per scatter task: one 64-byte line (eight words) of every
    // instance row per part, so each scratch line is read exactly once.
    const COLS: usize = 4;
    let n_col_blocks = useful_chunks.div_ceil(COLS);
    let mut j0 = 0;
    while j0 < n {
        let cnt = BATCH.min(n - j0);
        let scratch = &mut scratch[..cnt * 3 * w];
        let t0 = std::time::Instant::now();
        // Each task zeroes its own rows (the contract of `rows`) in parallel.
        scratch
            .par_chunks_mut(3 * w)
            .enumerate()
            .for_each(|(i, buf)| {
                buf.fill(0);
                let (zb, rest) = buf.split_at_mut(w);
                let (ab, bb) = rest.split_at_mut(w);
                rows(j0 + i, zb, ab, bb);
            });
        phase_ms[0] += t0.elapsed().as_secs_f64() * 1e3;
        let t0 = std::time::Instant::now();
        let scratch = &*scratch;
        let word = |i: usize, part: usize, u: usize| -> u64 {
            if u < w {
                scratch[i * 3 * w + part * w + u]
            } else {
                0
            }
        };
        // BatchMajor scatter of z, a, b, parallel over blocks of COLS
        // columns: each task reads the line of its columns from every
        // instance row once and writes its own columns of the three buffers.
        z.par_chunks_mut(COLS << nu)
            .zip(a.par_chunks_mut(COLS << nu))
            .zip(b.par_chunks_mut(COLS << nu))
            .take(n_col_blocks)
            .enumerate()
            .for_each(|(cb, ((zc, ac), bc))| {
                let c0 = cb * COLS;
                let c1 = (c0 + COLS).min(useful_chunks);
                for (part, cols) in [zc, ac, bc].into_iter().enumerate() {
                    for c in c0..c1 {
                        let col = &mut cols[(c - c0) << nu..((c - c0) + 1) << nu];
                        for i in 0..cnt {
                            col[j0 + i] = F128::new(word(i, part, 2 * c), word(i, part, 2 * c + 1));
                        }
                    }
                }
            });
        phase_ms[1] += t0.elapsed().as_secs_f64() * 1e3;
        j0 += cnt;
    }
    if trace {
        eprintln!(
            "  [tied] pack_slot: rows {:.1} ms, scatter {:.1} ms",
            phase_ms[0], phase_ms[1]
        );
    }
    SlotBuffers {
        z: std::sync::Arc::new(z),
        ab: Some((a, b)),
    }
}

/// Instances per scratch batch of [`pack_slot_groups`] and Rayon thread (a multiple
/// of 8): 16 groups per thread. On one thread a batch is 4.4 MiB of sponge-gate
/// tiles, which the scatter reads from the last-level cache; on many threads the
/// batch grows so that the tile pass has enough groups to balance and the column
/// scatter (one pass over every column per batch) runs a few times, not once per
/// 128 instances.
const GROUP_BATCH: usize = 128;

/// Upper bound on a scratch batch of [`pack_slot_groups`]: 512 groups, 141 MB of
/// sponge-gate tiles.
const GROUP_BATCH_MAX: usize = 4096;

/// Minor page faults of this process so far (field 10 of `/proc/self/stat`); 0 when
/// unreadable. A diagnostic for the trace only.
fn minor_faults() -> u64 {
    std::fs::read_to_string("/proc/self/stat")
        .ok()
        .and_then(|s| {
            let rest = &s[s.rfind(')')? + 1..];
            rest.split_whitespace().nth(7)?.parse().ok()
        })
        .unwrap_or(0)
}

/// [`pack_slot`] of the honest rows of `n` instances, walked eight at a time in
/// lockstep by [`ChainGate::instance_phi_many`] on `engine`'s round kernel. `inputs`
/// holds `input_words` packed input words per instance; `absorb(j, m, s)` XORs
/// instance `j`'s bits absorbed before permutation `m` into `s` (see
/// [`ChainGate::instance_rows_many`]). The buffers equal those of [`pack_slot`] over
/// [`ChainGate::instance_rows`].
///
/// Each group of eight keeps a word-major tile (word `u` of the eight instances
/// is one 64-byte line): the non-chi words of `z` and `phi` of every round, a
/// third of the three rows. The BatchMajor scatter derives `a`, `b`, `z`
/// from `phi` and writes eight adjacent union words per group and column.
///
/// # Panics
/// If `n > 2^nu`, `nu < 3` or `inputs` is shorter than `n * input_words`.
pub fn pack_slot_groups<A>(
    gate: &ChainGate,
    engine: &Engine,
    nu: usize,
    n: usize,
    inputs: &[u64],
    input_words: usize,
    absorb: A,
) -> SlotBuffers
where
    A: Fn(usize, usize, &mut Lanes) + Sync,
{
    let lay = gate.layout();
    let (k_log, useful_bits) = (lay.k_log, lay.useful_bits);
    assert!(nu >= 3, "Flock's lincheck fold needs at least 8 rows");
    assert!(n <= 1 << nu, "count exceeds the row capacity");
    assert!(k_log >= 7 && useful_bits <= 1 << k_log, "block geometry");
    assert!(inputs.len() >= n * input_words, "input words");
    let k = 1usize << k_log;
    let (w, t) = (lay.useful_words(), lay.t_base);
    // A group's tile: the head words, then `phi` of every round in full (the pruned
    // last round of the last permutation still has 25 `phi` words; its committed
    // lanes read row `y = 0` of them).
    let tw = t + lay.full_chi_words(gate.spec().n_perms(), gate.spec().n_rounds);
    assert!(t % 2 == 0, "chi words start at a 128-bit column");
    // An odd last word pairs with zero (columns past the useful bits are zero).
    let useful_chunks = w.div_ceil(2);
    // phi sources of `a = !phi[x + 1, y]` and `b = phi[x + 2, y]` per chi word (tile index).
    let sources: Vec<[usize; 2]> = (t..w)
        .map(|u| {
            let (g, l) = ((u - t) / N_LANES, (u - t) % N_LANES);
            let (x, y) = (l % 5, l / 5);
            let row = t + N_LANES * g + 5 * y;
            [row + (x + 1) % 5, row + (x + 2) % 5]
        })
        .collect();
    let b_head = gate.head_b();
    let len = (k / 128) << nu;
    let trace = std::env::var_os("AGG_TIED_TRACE").is_some();
    let faults_before = if trace { minor_faults() } else { 0 };
    let mut z: Vec<F128> = alloc_zeroed_vec(len);
    let mut a: Vec<F128> = alloc_zeroed_vec(len);
    let mut b: Vec<F128> = alloc_zeroed_vec(len);
    let batch = (GROUP_BATCH * rayon::current_num_threads()).min(GROUP_BATCH_MAX);
    let groups_max = batch.min(n.next_multiple_of(8)) / 8;
    let mut scratch: Vec<Words8> = vec![[0u64; 8]; groups_max * tw];
    let mut phase_ms = [0f64; 2];
    const COLS: usize = 4;
    let n_col_blocks = useful_chunks.div_ceil(COLS);
    let mut j0 = 0;
    while j0 < n {
        let cnt = batch.min(n - j0);
        let groups = cnt.div_ceil(8);
        let t0 = std::time::Instant::now();
        // Tiles, one task per group of eight.
        scratch[..groups * tw]
            .par_chunks_mut(tw)
            .enumerate()
            .for_each(|(g, tile)| {
                let first = j0 + 8 * g;
                let live = 8.min(n - first);
                let (head, phi) = tile.split_at_mut(t);
                let input = |i: usize| {
                    let j = first + i;
                    &inputs[j * input_words..(j + 1) * input_words]
                };
                gate.instance_phi_many(
                    engine,
                    live,
                    input,
                    |i, m, s| absorb(first + i, m, s),
                    head,
                    phi,
                    |_, _| {},
                );
            });
        phase_ms[0] += t0.elapsed().as_secs_f64() * 1e3;
        let t0 = std::time::Instant::now();
        let scratch = &scratch[..groups * tw];
        // BatchMajor scatter of z, a, b, parallel over blocks of COLS columns.
        z.par_chunks_mut(COLS << nu)
            .zip(a.par_chunks_mut(COLS << nu))
            .zip(b.par_chunks_mut(COLS << nu))
            .take(n_col_blocks)
            .enumerate()
            .for_each(|(cb, ((zc, ac), bc))| {
                let c0 = cb * COLS;
                let c1 = (c0 + COLS).min(useful_chunks);
                for c in c0..c1 {
                    let r = ((c - c0) << nu)..((c - c0 + 1) << nu);
                    let (zcol, acol, bcol) = (&mut zc[r.clone()], &mut ac[r.clone()], &mut bc[r]);
                    let (u0, u1) = (2 * c, 2 * c + 1);
                    for g in 0..groups {
                        let tile = &scratch[g * tw..(g + 1) * tw];
                        let first = j0 + 8 * g;
                        let live = 8.min(n - first);
                        let (zw, aw, bw): ([F128; 8], [F128; 8], [F128; 8]) = if u0 < t {
                            let (lo, hi) = (&tile[u0], &tile[u1]);
                            let zw = std::array::from_fn(|i| F128::new(lo[i], hi[i]));
                            (zw, zw, [F128::new(b_head[u0], b_head[u1]); 8])
                        } else {
                            let [l1, l2] = sources[u0 - t];
                            let (l1, l2) = (&tile[l1], &tile[l2]);
                            // Past the last word: !h1 & h2 = 0, !h1 = 0, h2 = 0.
                            let (h1, h2) = match sources.get(u1 - t) {
                                Some(&[h1, h2]) => (&tile[h1], &tile[h2]),
                                None => (&[u64::MAX; 8], &[0u64; 8]),
                            };
                            (
                                std::array::from_fn(|i| F128::new(!l1[i] & l2[i], !h1[i] & h2[i])),
                                std::array::from_fn(|i| F128::new(!l1[i], !h1[i])),
                                std::array::from_fn(|i| F128::new(l2[i], h2[i])),
                            )
                        };
                        let rows = first..first + live;
                        zcol[rows.clone()].copy_from_slice(&zw[..live]);
                        acol[rows.clone()].copy_from_slice(&aw[..live]);
                        bcol[rows].copy_from_slice(&bw[..live]);
                    }
                }
            });
        phase_ms[1] += t0.elapsed().as_secs_f64() * 1e3;
        j0 += cnt;
    }
    if trace {
        eprintln!(
            "  [tied] pack_slot_groups ({:?}): rows {:.1} ms, scatter {:.1} ms, page faults {}",
            engine.backend(),
            phase_ms[0],
            phase_ms[1],
            minor_faults() - faults_before
        );
    }
    SlotBuffers {
        z: std::sync::Arc::new(z),
        ab: Some((a, b)),
    }
}

/// `out[bit] = sum_r bit(v[r], bit) << r` for `bit < 64`: eight 8x8 bit-matrix
/// transposes (byte `k` of every `v[r]` forms one matrix), three delta swaps each,
/// instead of 512 single-bit moves.
#[inline]
pub(crate) fn transpose_8x64(v: &[u64; 8], out: &mut [u8]) {
    debug_assert_eq!(out.len(), 64);
    for k in 0..8 {
        // Byte r of x is byte k of v[r]: row r, column c = bit 8k + c of v[r].
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
        // Now byte c of x is column c: bit r = bit 8k + c of v[r].
        out[8 * k..8 * k + 8].copy_from_slice(&x.to_le_bytes());
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gate::ChainSpec;
    use flock_core::lincheck::pack_z_lincheck;
    use flock_core::test_rng::Rng;

    /// The 8-way generators of every parameter set equal the portable oracle (`pack_slot`
    /// over `instance_rows` of `input_bits`) on honest records, counts that are not
    /// multiples of eight: `z`, the lane-major input bits and the row values of the AVX2
    /// kernels, and of the AVX-512 kernels where the CPU has them; on a tampered `z` (padded
    /// lane words included) the row values of both kernels equal the per-row
    /// `ChainGate::apply_ab`.
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn eight_way_generators_match_reference() {
        use gen8::{Geo, Geo44, Geo65, Geo87};
        use std::arch::is_x86_feature_detected as has;
        if !(has!("avx2") && has!("bmi2")) {
            eprintln!("no AVX2 on this CPU: 8-way generators not exercised");
            return;
        }
        fn per_row(gate: &ChainGate, nu: usize, n: usize, z: &[F128]) -> (Vec<F128>, Vec<F128>) {
            let packed = pack_slot(gate, nu, n, |j, zz, a, b| {
                for (u, w) in zz.iter_mut().enumerate() {
                    let word = z[((u / 2) << nu) + j];
                    *w = if u % 2 == 0 { word.lo } else { word.hi };
                }
                gate.apply_ab(zz, a, b);
            });
            packed.ab.expect("pack_slot builds a, b")
        }
        fn check<G: Geo, const IN_Q: usize, const MSG_Q: usize>(seed: u64) {
            let layout = G::L;
            let gate = crate::sponge::sponge_gate_for(layout.set);
            assert_eq!(gen8::sponge_set(&gate), Some(layout.set));
            let mut rng = Rng::new(seed);
            for n in [1usize, 5, 13, 64, 300] {
                let recs: Vec<SpongeRecord<G::P>> = (0..n)
                    .map(|_| {
                        let mut w1 = <G::P as ParamSet>::zero_w1_encode();
                        w1.as_mut().iter_mut().for_each(|b| *b = rng.next_u32() as u8);
                        SpongeRecord {
                            mu: core::array::from_fn(|_| rng.next_u32() as u8),
                            w1_encode: w1,
                        }
                    })
                    .collect();
                let nu = n.next_power_of_two().max(8).trailing_zeros() as usize;
                let inputs: Vec<Vec<u64>> = recs.iter().map(SpongeRecord::input_bits).collect();
                let want = pack_slot(&gate, nu, n, |j, z, a, b| {
                    gate.instance_rows(&inputs[j], z, a, b);
                });
                let (want_a, want_b) = want.ab.as_ref().expect("packed a, b");
                // Lane-major bits of a span of two segments of 2^d records, as
                // `TiedPart::honest_parts` lays them out (padded positions zero).
                let d = n.next_power_of_two().trailing_zeros().saturating_sub(1);
                let lane_len = G::P_PAD << d;
                let seg_words = G::LANES * lane_len / 64;
                let mut want_bits = vec![0u64; 2 * seg_words];
                for (g, inp) in inputs.iter().enumerate() {
                    let (sigma, i) = (g >> d, g & ((1 << d) - 1));
                    for t in 0..G::LANES {
                        let src = G::LIVE_Q * t;
                        let dst = sigma * seg_words + (lane_len * t + G::P_PAD * i) / 64;
                        want_bits[dst..dst + G::LIVE_Q].copy_from_slice(&inp[src..src + G::LIVE_Q]);
                    }
                }
                type Kernel<G> = fn(&ChainGate, usize, &[SpongeRecord<<G as Geo>::P>], Option<LaneBits<'_>>) -> SlotBuffers;
                type AbKernel = fn(&ChainGate, usize, usize, &[F128]) -> (Vec<F128>, Vec<F128>);
                let mut kernels: Vec<(&str, Kernel<G>, AbKernel)> = vec![(
                    "avx2",
                    gen8x2::sponge_slot::<G, IN_Q, MSG_Q>,
                    gen8x2::pack_ab::<G, IN_Q>,
                )];
                if crate::simd::detected() {
                    kernels.push((
                        "avx512",
                        gen8::sponge_slot::<G, IN_Q, MSG_Q>,
                        gen8::pack_ab::<G, IN_Q>,
                    ));
                }
                let tag = |k: &str, what: &str| format!("{} {k} {what}, n = {n}", layout.set.name());
                let mut z_any = None;
                for &(k, slot, ab) in &kernels {
                    let mut bits = vec![0u64; 2 * seg_words];
                    let lb = LaneBits {
                        bits: &mut bits,
                        seg_words,
                        lane_len,
                        d,
                    };
                    let got = slot(&gate, nu, &recs, Some(lb));
                    assert!(got.z.as_slice() == want.z.as_slice(), "{}", tag(k, "z"));
                    assert!(bits == want_bits, "{}", tag(k, "lane bits"));
                    let (a, b) = ab(&gate, nu, n, &got.z);
                    assert!(&a == want_a && &b == want_b, "{}", tag(k, "a, b"));
                    z_any = Some(got.z);
                }
                // Tampered z: random bits anywhere in the used columns, the padded lane
                // words and the constant included.
                let mut z = std::sync::Arc::unwrap_or_clone(z_any.expect("a kernel ran"));
                let used = gate.layout().useful_bits.div_ceil(128) << nu;
                for _ in 0..64 {
                    let i = rng.next_u64() as usize % used;
                    z[i].lo ^= rng.next_u64();
                    z[i].hi ^= rng.next_u64();
                }
                let pad = (G::LIVE_Q / 2) << nu;
                z[pad] = F128::new(rng.next_u64(), 1);
                let want_ab = per_row(&gate, nu, n, &z);
                for &(k, _, ab) in &kernels {
                    assert!(ab(&gate, nu, n, &z) == want_ab, "{}", tag(k, "tampered a, b"));
                }
            }
        }
        check::<Geo44, { Geo44::IN_Q }, { Geo44::MSG_Q }>(0xA2);
        check::<Geo65, { Geo65::IN_Q }, { Geo65::MSG_Q }>(0xA3);
        check::<Geo87, { Geo87::IN_Q }, { Geo87::MSG_Q }>(0xA4);
    }

    /// The delta-swap transpose equals the bit-by-bit definition.
    #[test]
    fn transpose_matches_bitwise() {
        let mut rng = Rng::new(17);
        for _ in 0..200 {
            let v: [u64; 8] = core::array::from_fn(|_| rng.next_u64());
            let mut got = [0u8; 64];
            transpose_8x64(&v, &mut got);
            for (bit, &g) in got.iter().enumerate() {
                let mut want = 0u8;
                for (r, &x) in v.iter().enumerate() {
                    want |= (((x >> bit) & 1) as u8) << r;
                }
                assert_eq!(g, want, "bit {bit}");
            }
        }
    }

    /// `pack_slot_groups` equals `pack_slot` over `instance_rows` on every Keccak
    /// backend, for a one-round permutation gate and a chain with unaligned fields and
    /// scattered outputs, with partial groups, on the global pool and on one thread
    /// (more than one scratch batch).
    #[test]
    fn group_packing_matches_instance_packing() {
        use agg_mldsa_simd::{Backend, Engine};
        let mut rng = Rng::new(23);
        for (gate, nu, n) in [
            (
                ChainGate::new(ChainSpec::permutation_rounds(1)),
                4usize,
                11usize,
            ),
            (ChainGate::new(crate::gate::tests::mini_chain(2)), 9, 300),
        ] {
            let iw = gate.input_words();
            let inputs: Vec<u64> = (0..n * iw).map(|_| rng.next_u64()).collect();
            let want = pack_slot(&gate, nu, n, |j, z, a, b| {
                gate.instance_rows(&inputs[j * iw..(j + 1) * iw], z, a, b);
            });
            for e in [Backend::Scalar, Backend::Avx2, Backend::Avx512]
                .into_iter()
                .filter_map(Engine::new)
            {
                // One thread: scratch batches of GROUP_BATCH instances (300 spans three).
                let one = rayon::ThreadPoolBuilder::new()
                    .num_threads(1)
                    .build()
                    .unwrap();
                for threads in [0, 1] {
                    let run = || {
                        pack_slot_groups(&gate, &e, nu, n, &inputs, iw, |j, m, s| {
                            gate.absorb_into(m, &inputs[j * iw..(j + 1) * iw], s);
                        })
                    };
                    let got = if threads == 1 {
                        one.install(run)
                    } else {
                        run()
                    };
                    let tag = format!("{:?}, threads {threads}", e.backend());
                    assert!(got.z.as_slice() == want.z.as_slice(), "{tag} z");
                    let (got_a, got_b) = got.ab.as_ref().expect("packed a, b");
                    let (want_a, want_b) = want.ab.as_ref().expect("packed a, b");
                    assert!(got_a == want_a, "{tag} a");
                    assert!(got_b == want_b, "{tag} b");
                }
            }
        }
    }

    /// The packed buffers agree with Flock's canonical `pack_z_lincheck` and
    /// the BatchMajor addressing, dummy rows included.
    #[test]
    fn packing_matches_flock_canonical_layout() {
        let gate = ChainGate::new(ChainSpec::permutation_rounds(1));
        let lay = gate.layout().clone();
        let (nu, n) = (4usize, 11usize);
        let k = 1usize << lay.k_log;
        let mut rng = Rng::new(5);
        let inputs: Vec<Vec<u64>> = (0..n)
            .map(|_| (0..gate.input_words()).map(|_| rng.next_u64()).collect())
            .collect();
        let bufs = pack_slot(&gate, nu, n, |j, z, a, b| {
            gate.instance_rows(&inputs[j], z, a, b);
        });
        // Logical bit view: instance-major (`i_inner + i_outer * k`).
        let mut logical = vec![false; k << nu];
        for (j, inp) in inputs.iter().enumerate() {
            let w = lay.useful_words();
            let (mut z, mut a, mut b) = (vec![0u64; w], vec![0u64; w], vec![0u64; w]);
            gate.instance_rows(inp, &mut z, &mut a, &mut b);
            for col in 0..lay.useful_bits {
                logical[j * k + col] = (z[col / 64] >> (col % 64)) & 1 == 1;
            }
            for col in 0..lay.useful_bits {
                let word = bufs.z[((col / 128) << nu) + j];
                let bit = if col % 128 < 64 {
                    (word.lo >> (col % 128)) & 1
                } else {
                    (word.hi >> (col % 128 - 64)) & 1
                };
                assert_eq!(bit == 1, logical[j * k + col]);
            }
        }
        assert_eq!(
            lincheck_stripe(&bufs.z, lay.k_log, nu),
            pack_z_lincheck(&logical, nu + lay.k_log, lay.k_log)
        );
        for j in n..1 << nu {
            for c in 0..k / 128 {
                assert_eq!(bufs.z[(c << nu) + j], F128::ZERO);
                assert_eq!(bufs.ab.as_ref().expect("packed a, b").0[(c << nu) + j], F128::ZERO);
            }
        }
    }
}
