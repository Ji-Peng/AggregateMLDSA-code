//! SHAKE256 sponge circuit for `c~ = H(mu || w1Encode(w1'))`.
//!
//! The text below describes the ML-DSA-44 layout; [`SpongeLayout`] gives the block layout of
//! every parameter set (ML-DSA-65 and ML-DSA-87: four bit lanes of `P_pad = 2,048` positions,
//! 384- and 512-bit `c~`, seven and nine permutations, the public region at word 64).
//!
//! ML-DSA-44: one gate instance per signature, seven chained chi-only Keccak-f[1600]
//! permutations (rate 136 bytes, capacity 512 bits, initial state zero):
//!
//! | Permutation | Absorbed block (136 bytes) |
//! | --- | --- |
//! | P1 | mu (64 bytes, public) followed by w1Encode bytes 0..71 |
//! | P2..P6 | w1Encode bytes 72 + 136 (m - 2) .. (136 bytes each) |
//! | P7 | w1Encode bytes 752..767, 0x1F, 0x00 x 118, 0x80 |
//!
//! `c~` is the low 256 bits of the output of P7. The absorptions, the padding
//! constants and the chaining are virtual XOR expressions of the gate; the
//! only committed bits are mu (512, wired to public words), the 6,144
//! w1Encode bits, `c~` (256, wired to public words), the constant and
//! `(7 x 24 - 1) x 1,600 + 256` chi bits: round 23 of P7 commits only lanes
//! `(0..3, 0)`, the 256 AND bits that `c~` reads (`crate::gate`).
//!
//! Input column order of the w1Encode bits is `(t, j, rho)`, lane `t` outermost:
//! column `1024 t + 256 rho + j` holds `b_t(j, rho)`, stream bit
//! `1536 rho + 6 j + t` of w1Encode (FIPS 204 Algorithms 16 and 28). The w1Encode
//! field comes first in the block, so lane `t` is the 8 aligned 128-bit words
//! `8 t .. 8 t + 8` and the 48 words form the aligned word subcubes `0..32` and
//! `32..48`; mu follows at words 48..51, `c~` at 52..53, the constant at word 54,
//! word 55 is zero and the chi words start at word 56, so words 48..56 are the
//! public region (mu, `c~`, the constant, zero). The tied part (`crate::tied`)
//! ties the public region to the public words and the w1Encode bits to the parity
//! chain by evaluation claims; the word constants below name the ML-DSA-44 layout.

use agg_witness::params::{ParamSet, Set, Set44, Set65, Set87};
use flock_field::F128;
use sha3::{
    digest::{ExtendableOutput, Update, XofReader},
    Shake256,
};

use crate::gate::{Absorb, ChainGate, ChainSpec, InputField};
use crate::keccak::N_ROUNDS;

/// SHAKE256 rate in bytes.
pub const RATE_BYTES: usize = 136;
/// Bytes of mu.
pub const MU_BYTES: usize = 64;
/// Bytes of w1Encode for ML-DSA-44 (`k * 256 * 6 / 8`).
pub const W1_BYTES: usize = 768;
/// w1Encode bits.
pub const W1_BITS: usize = 8 * W1_BYTES;
/// Bytes of `c~` (`lambda / 4` for ML-DSA-44).
pub const CTILDE_BYTES: usize = 32;
/// Permutations per sponge.
pub const N_PERMS: usize = 7;
/// Global input index of the first w1Encode bit (the w1Encode field comes first).
pub const W1_INPUT_BASE: usize = 0;
/// Global input index of the first mu bit (after the 6,144 w1Encode bits).
pub const MU_INPUT_BASE: usize = W1_BITS;
/// Gate input bits per record (w1Encode then mu).
pub const INPUT_BITS: usize = W1_BITS + 8 * MU_BYTES;
/// Word column of the first w1Encode word (lane `t` at words `8 t .. 8 t + 8`).
pub const W1_WORD_BASE: usize = 0;
/// Word column of the first mu word (4 words).
pub const MU_WORD_BASE: usize = 48;
/// Word column of the first `c~` word (2 words).
pub const CTILDE_WORD_BASE: usize = 52;
/// Word column of the constant-one column (bit 0 of the word).
pub const CONST_WORD: usize = 54;
/// First word of the public region `[48, 56)`: mu, `c~`, the constant, one zero word.
pub const PUBLIC_WORD_BASE: usize = 48;
/// `log2` of the public region's width in words.
pub const PUBLIC_WORDS_LOG2: usize = 3;
/// First chi word column.
pub const CHI_WORD_BASE: usize = 56;

/// The sponge statement and the block layout of one parameter set.
///
/// The bits of `w1'` form `lanes` bit lanes (one per bit of a coefficient, FIPS 204
/// Algorithm 28 packs `lanes` bits per coefficient). Lane `t` of a record holds the bits
/// `b_t(j, rho)` at the positions `p = j + 256 rho < 256 k` of a lane of `P_pad` positions
/// (`positions_pad`), and the gate commits it as the `P_pad / 128` words `lane_words t ..`
/// starting at word 0, bit `p mod 128` of word `p / 128`. Positions `256 k .. P_pad` of a
/// lane (ML-DSA-65: rows 6 and 7) are columns of no input, which the gate pins to zero.
/// The public region follows the lanes: mu (4 words), `c~` (`c~` bytes / 16 words), the
/// constant one and zero words up to the first chi word, an aligned power-of-two run.
///
/// | | ML-DSA-44 | ML-DSA-65 | ML-DSA-87 |
/// | --- | --- | --- | --- |
/// | lanes x words per lane | 6 x 8 | 4 x 16 (12 live) | 4 x 16 |
/// | mu, `c~`, constant | 48, 52, 54 | 64, 68, 71 | 64, 68, 72 |
/// | public region, chi | `[48, 56)`, 56 | `[64, 72)`, 72 | `[64, 80)`, 80 |
/// | permutations | 7 | 7 | 9 |
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SpongeLayout {
    /// The parameter set.
    pub set: Set,
    /// Polynomials of `w1` (`k`).
    pub k: usize,
    /// Bits per `w1` coefficient: the bit lanes.
    pub lanes: usize,
    /// Padded positions per lane and record, `P_pad` (a power of two of at least `256 k`).
    pub positions_pad: usize,
    /// Bytes of w1Encode.
    pub w1_bytes: usize,
    /// Bytes of `c~`.
    pub ctilde_bytes: usize,
    /// Permutations of the sponge: `ceil((64 + w1_bytes + 1) / 136)`.
    pub n_perms: usize,
    /// Word column of the first mu word.
    pub mu_word_base: usize,
    /// Word column of the first `c~` word.
    pub ctilde_word_base: usize,
    /// Word column of the constant one.
    pub const_word: usize,
    /// First word of the public region.
    pub public_word_base: usize,
    /// `log2` of the public region's width in words.
    pub public_words_log2: usize,
    /// First chi word column.
    pub chi_word_base: usize,
}

impl SpongeLayout {
    /// ML-DSA-44.
    pub const S44: Self = Self::of_params::<Set44>();
    /// ML-DSA-65.
    pub const S65: Self = Self::of_params::<Set65>();
    /// ML-DSA-87.
    pub const S87: Self = Self::of_params::<Set87>();

    const fn of_params<P: ParamSet>() -> Self {
        let lanes = P::W1_BITS;
        let positions_pad = P::P_PAD;
        let mu_word_base = lanes * positions_pad / 128;
        let ctilde_word_base = mu_word_base + MU_BYTES / 16;
        let const_word = ctilde_word_base + P::C_TILDE_BYTES / 16;
        // The chi words start at the next 8-word boundary (`GateLayout`).
        let chi_word_base = (const_word + 1).next_multiple_of(8);
        let public = chi_word_base - mu_word_base;
        assert!(public.is_power_of_two() && mu_word_base.is_multiple_of(public));
        assert!(8 * P::W1_ENCODE_BYTES == lanes * 256 * P::K);
        assert!((MU_BYTES + P::W1_ENCODE_BYTES) / RATE_BYTES + 1 == P::N_PERMS);
        Self {
            set: P::SET,
            k: P::K,
            lanes,
            positions_pad,
            w1_bytes: P::W1_ENCODE_BYTES,
            ctilde_bytes: P::C_TILDE_BYTES,
            n_perms: P::N_PERMS,
            mu_word_base,
            ctilde_word_base,
            const_word,
            public_word_base: mu_word_base,
            public_words_log2: public.trailing_zeros() as usize,
            chi_word_base,
        }
    }

    /// The layout of `set`.
    #[must_use]
    pub const fn of(set: Set) -> Self {
        match set {
            Set::S44 => Self::S44,
            Set::S65 => Self::S65,
            Set::S87 => Self::S87,
        }
    }

    /// The layout of the sponge gate `gate` (identified by its `c~` width and checked
    /// against its input fields), or `None` for another gate.
    #[must_use]
    pub fn of_gate(gate: &ChainGate) -> Option<Self> {
        let spec = gate.spec();
        let layout = Set::ALL
            .into_iter()
            .map(Self::of)
            .find(|l| spec.outputs.len() == 8 * l.ctilde_bytes)?;
        (spec.input_fields == layout.input_fields() && spec.n_perms() == layout.n_perms)
            .then_some(layout)
    }

    /// Live positions per lane and record, `256 k`.
    #[must_use]
    pub const fn positions(&self) -> usize {
        256 * self.k
    }

    /// `log2 P_pad`: the in-record position bits.
    #[must_use]
    pub const fn positions_log2(&self) -> usize {
        self.positions_pad.trailing_zeros() as usize
    }

    /// Words per lane and record, `P_pad / 128`.
    #[must_use]
    pub const fn lane_words(&self) -> usize {
        self.positions_pad / 128
    }

    /// `log2` of the records of a level-1 block of the bridge, `19 - log2 P_pad`.
    #[must_use]
    pub const fn block_records_log2(&self) -> usize {
        19 - self.positions_log2()
    }

    /// w1Encode bits.
    #[must_use]
    pub const fn w1_bits(&self) -> usize {
        8 * self.w1_bytes
    }

    /// Global input index of the first mu bit.
    #[must_use]
    pub const fn mu_input_base(&self) -> usize {
        self.w1_bits()
    }

    /// Gate input bits per record (w1Encode then mu).
    #[must_use]
    pub const fn input_bits(&self) -> usize {
        self.w1_bits() + 8 * MU_BYTES
    }

    /// Message bytes `mu || w1Encode`.
    #[must_use]
    pub const fn message_bytes(&self) -> usize {
        MU_BYTES + self.w1_bytes
    }

    /// `(t, j, rho)` of w1Encode stream bit `q = 256 lanes rho + lanes j + t`.
    #[inline]
    #[must_use]
    pub const fn w1_coordinates(&self, q: usize) -> (usize, usize, usize) {
        let per_poly = 256 * self.lanes;
        let (rho, rem) = (q / per_poly, q % per_poly);
        (rem % self.lanes, rem / self.lanes, rho)
    }

    /// Global input index of w1Encode stream bit `q`: `256 k t + 256 rho + j`.
    #[inline]
    #[must_use]
    pub const fn w1_input(&self, q: usize) -> usize {
        let (t, j, rho) = self.w1_coordinates(q);
        W1_INPUT_BASE + self.positions() * t + 256 * rho + j
    }

    /// Column (within the lanes) of w1Encode stream bit `q`: `P_pad t + 256 rho + j`.
    #[inline]
    #[must_use]
    pub const fn w1_column(&self, q: usize) -> usize {
        let (t, j, rho) = self.w1_coordinates(q);
        self.positions_pad * t + 256 * rho + j
    }

    /// The input fields: the lanes (one field when a lane has no padded position, else one
    /// field per lane of `256 k` bits in `P_pad` columns), then mu.
    #[must_use]
    pub fn input_fields(&self) -> Vec<InputField> {
        let mut fields = if self.positions() == self.positions_pad {
            vec![InputField::new("w1encode", self.w1_bits(), false)]
        } else {
            (0..self.lanes)
                .map(|_| InputField {
                    cols: self.positions_pad,
                    ..InputField::new("w1encode", self.positions(), false)
                })
                .collect()
        };
        fields.push(InputField::new("mu", 8 * MU_BYTES, true));
        fields
    }
}

/// One signature's sponge input (ML-DSA-44 unless `P` says otherwise).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SpongeRecord<P: ParamSet = Set44> {
    pub mu: [u8; MU_BYTES],
    pub w1_encode: P::W1Encode,
}

impl<P: ParamSet> SpongeRecord<P> {
    /// The sponge layout of the record's parameter set.
    #[must_use]
    pub const fn layout() -> SpongeLayout {
        SpongeLayout::of(P::SET)
    }

    /// `c~ = SHAKE256(mu || w1Encode, lambda / 4)` natively (sha3 crate).
    #[must_use]
    pub fn c_tilde(&self) -> P::CTilde {
        let mut h = Shake256::default();
        h.update(&self.mu);
        h.update(self.w1_encode.as_ref());
        let mut out = P::zero_c_tilde();
        h.finalize_xof().read(out.as_mut());
        out
    }

    /// Packed gate input bits: the w1Encode bits in `(t, j, rho)` input order at
    /// `[W1_INPUT_BASE, W1_INPUT_BASE + w1 bits)` ([`SpongeLayout::w1_input`]), then the mu
    /// bits at [`SpongeLayout::mu_input_base`].
    #[must_use]
    pub fn input_bits(&self) -> Vec<u64> {
        let mut words = vec![0u64; Self::layout().input_bits() / 64];
        self.input_bits_into(&mut words);
        words
    }

    /// Fill one record's view of a flat input buffer, retaining the bit-plane layout.
    pub fn input_bits_into(&self, words: &mut [u64]) {
        let layout = Self::layout();
        assert_eq!(words.len(), layout.input_bits() / 64);
        let w1: &[u8] = self.w1_encode.as_ref();
        let mu_base = layout.mu_input_base() / 64;
        for (word, bytes) in words[mu_base..].iter_mut().zip(self.mu.chunks_exact(8)) {
            *word = u64::from_le_bytes(bytes.try_into().expect("eight bytes"));
        }
        debug_assert_eq!(W1_INPUT_BASE % 64, 0);
        if P::W1_BITS == 6 {
            six_bit_planes(w1, &mut words[..mu_base]);
        } else {
            four_bit_planes(w1, layout.positions(), &mut words[..mu_base]);
        }
    }
}

/// Six-bit coefficients into six bit planes of 1,024 positions: read eight coefficients
/// together, then transpose their bytes into the planes (each coefficient enters the byte
/// matrix with one shift and OR).
fn six_bit_planes(w1: &[u8], words: &mut [u64]) {
    for rho in 0..w1.len() / 192 {
        for j in (0..256).step_by(8) {
            let lo = 192 * rho + 6 * (j / 8);
            let b = &w1[lo..lo + 6];
            let packed = u64::from_le_bytes([b[0], b[1], b[2], b[3], b[4], b[5], 0, 0]);
            let mut matrix = 0_u64;
            for i in 0..8 {
                matrix |= ((packed >> (6 * i)) & 0x3f) << (8 * i);
            }
            let swap = (matrix ^ (matrix >> 7)) & 0x00AA_00AA_00AA_00AA;
            matrix ^= swap ^ (swap << 7);
            let swap = (matrix ^ (matrix >> 14)) & 0x0000_CCCC_0000_CCCC;
            matrix ^= swap ^ (swap << 14);
            let swap = (matrix ^ (matrix >> 28)) & 0x0000_0000_F0F0_F0F0;
            matrix ^= swap ^ (swap << 28);
            for t in 0..6 {
                let idx = 1024 * t + 256 * rho + j;
                let byte = ((matrix >> (8 * t)) & 0xff) << (idx % 64);
                if j % 64 == 0 {
                    words[idx / 64] = byte;
                } else {
                    words[idx / 64] |= byte;
                }
            }
        }
    }
}

/// Bit `t` of each of the 16 nibbles of `x`, gathered into the low 16 bits.
#[inline]
const fn nibble_plane(x: u64, t: u32) -> u64 {
    let mut v = (x >> t) & 0x1111_1111_1111_1111;
    v = (v | (v >> 3)) & 0x0303_0303_0303_0303;
    v = (v | (v >> 6)) & 0x000F_000F_000F_000F;
    v = (v | (v >> 12)) & 0x0000_00FF_0000_00FF;
    (v | (v >> 24)) & 0xFFFF
}

/// Four-bit coefficients into four bit planes of `positions` positions: each little-endian
/// qword holds 16 coefficients, whose bit `t` lands as 16 consecutive bits of plane `t`.
fn four_bit_planes(w1: &[u8], positions: usize, words: &mut [u64]) {
    words.fill(0);
    for (q, chunk) in w1.chunks_exact(8).enumerate() {
        let x = u64::from_le_bytes(chunk.try_into().expect("eight bytes"));
        // Coefficients 16 q .. 16 q + 16 of the stream: position 16 q (mod 256 k).
        let p = 16 * q;
        for t in 0..4 {
            let idx = positions * t + p;
            words[idx / 64] |= nibble_plane(x, t as u32) << (idx % 64);
        }
    }
}

/// Column (within the w1 field) of ML-DSA-44 w1Encode stream bit `q = 1536 rho + 6 j + t`.
#[inline]
#[must_use]
pub const fn w1_column(q: usize) -> usize {
    SpongeLayout::S44.w1_column(q)
}

/// The sponge's absorbed `(state bit, source)` pairs per permutation of ML-DSA-44.
#[must_use]
pub fn absorb_map() -> Vec<Vec<(u16, Absorb)>> {
    absorb_map_for(&SpongeLayout::S44)
}

/// The absorbed `(state bit, source)` pairs per permutation of a layout: the message bits
/// `mu || w1Encode` in order (permutation `s / 1088`, state bit `s mod 1088`), then the SHAKE
/// padding `0x1F` right after the message and `0x80` at the last byte of the last block.
#[must_use]
pub fn absorb_map_for(layout: &SpongeLayout) -> Vec<Vec<(u16, Absorb)>> {
    let rate = 8 * RATE_BYTES;
    let mut perms: Vec<Vec<(u16, Absorb)>> = vec![Vec::new(); layout.n_perms];
    let mu_bits = 8 * MU_BYTES;
    for s in 0..8 * layout.message_bytes() {
        let source = if s < mu_bits {
            Absorb::Input((layout.mu_input_base() + s) as u32)
        } else {
            Absorb::Input(layout.w1_input(s - mu_bits) as u32)
        };
        perms[s / rate].push(((s % rate) as u16, source));
    }
    let pad = 8 * layout.message_bytes();
    for k in 0..5 {
        perms[(pad + k) / rate].push((((pad + k) % rate) as u16, Absorb::One)); // 0x1F
    }
    perms[layout.n_perms - 1].push(((rate - 1) as u16, Absorb::One)); // 0x80 of byte 135
    perms
}

/// The ML-DSA-44 sponge gate spec with `n_rounds` rounds per permutation (24; fewer only in
/// tests).
#[must_use]
pub fn sponge_spec_rounds(n_rounds: usize) -> ChainSpec {
    sponge_spec_for(&SpongeLayout::S44, n_rounds)
}

/// The sponge gate spec of a layout with `n_rounds` rounds per permutation.
#[must_use]
pub fn sponge_spec_for(layout: &SpongeLayout, n_rounds: usize) -> ChainSpec {
    ChainSpec {
        label: "shake256(mu||w1encode)->ctilde",
        n_rounds,
        input_fields: layout.input_fields(),
        absorbs: absorb_map_for(layout),
        outputs: (0..8 * layout.ctilde_bytes as u16).collect(),
    }
}

/// The production ML-DSA-44 sponge gate.
#[must_use]
pub fn sponge_gate() -> ChainGate {
    ChainGate::new(sponge_spec_rounds(N_ROUNDS))
}

/// The production sponge gate of a parameter set.
#[must_use]
pub fn sponge_gate_for(set: Set) -> ChainGate {
    ChainGate::new(sponge_spec_for(&SpongeLayout::of(set), N_ROUNDS))
}

/// Little-endian 128-bit words of a byte string whose length is a multiple of 16.
#[must_use]
pub fn bytes_to_words(bytes: &[u8]) -> Vec<F128> {
    let mut out = Vec::with_capacity(bytes.len() / 16);
    extend_words(&mut out, bytes);
    out
}

/// Append the words of `bytes` to `out`, as [`bytes_to_words`].
pub fn extend_words(out: &mut Vec<F128>, bytes: &[u8]) {
    assert!(bytes.len().is_multiple_of(16));
    out.extend(bytes.chunks_exact(16).map(|c| {
        let lo = u64::from_le_bytes(c[..8].try_into().expect("8 bytes"));
        let hi = u64::from_le_bytes(c[8..].try_into().expect("8 bytes"));
        F128::new(lo, hi)
    }));
}

#[cfg(test)]
mod tests {
    use super::*;
    use flock_core::test_rng::Rng;

    /// `input_bits` equals the bit-by-bit definition through `w1_column`.
    #[test]
    fn input_bits_match_w1_column() {
        let mut rng = Rng::new(0x1B);
        for _ in 0..20 {
            let r = random_record(&mut rng);
            let mut want = vec![0u64; INPUT_BITS / 64];
            for (i, &byte) in r.mu.iter().enumerate() {
                let bit = MU_INPUT_BASE + 8 * i;
                want[bit / 64] |= u64::from(byte) << (bit % 64);
            }
            for q in 0..W1_BITS {
                if (r.w1_encode[q / 8] >> (q % 8)) & 1 == 1 {
                    let idx = W1_INPUT_BASE + w1_column(q);
                    want[idx / 64] |= 1 << (idx % 64);
                }
            }
            assert_eq!(r.input_bits(), want);
            let mut into = vec![u64::MAX; INPUT_BITS / 64];
            r.input_bits_into(&mut into);
            assert_eq!(into, want);
        }
    }

    pub(crate) fn random_record(rng: &mut Rng) -> SpongeRecord {
        let mut mu = [0u8; MU_BYTES];
        let mut w1 = [0u8; W1_BYTES];
        mu.iter_mut().for_each(|b| *b = rng.next_u32() as u8);
        w1.iter_mut().for_each(|b| *b = rng.next_u32() as u8);
        SpongeRecord { mu, w1_encode: w1 }
    }

    #[test]
    fn w1_columns_are_a_bijection() {
        let mut seen = vec![false; W1_BITS];
        for q in 0..W1_BITS {
            let c = w1_column(q);
            assert!(!seen[c]);
            seen[c] = true;
        }
        // Stream bit 1536 rho + 6 j + t is lane t, coefficient j of polynomial rho.
        assert_eq!(w1_column(1536 + 6 * 5 + 3), 1024 * 3 + 256 + 5);
    }

    #[test]
    fn absorb_map_covers_rate_and_padding() {
        let map = absorb_map();
        assert_eq!(map.len(), N_PERMS);
        let inputs: usize = map
            .iter()
            .flatten()
            .filter(|(_, s)| matches!(s, Absorb::Input(_)))
            .count();
        assert_eq!(inputs, 8 * MU_BYTES + W1_BITS);
        let ones: Vec<u16> = map[6]
            .iter()
            .filter(|(_, s)| *s == Absorb::One)
            .map(|(b, _)| *b)
            .collect();
        assert_eq!(ones, vec![128, 129, 130, 131, 132, 1087]);
        for list in &map {
            assert!(list.iter().all(|&(s, _)| (s as usize) < 8 * RATE_BYTES));
        }
    }

    #[test]
    fn gate_matches_shake256_for_random_records() {
        let gate = sponge_gate();
        let mut rng = Rng::new(0x5EED);
        for _ in 0..16 {
            let rec = random_record(&mut rng);
            let bits = gate.output_bits(&rec.input_bits());
            let mut got = [0u8; CTILDE_BYTES];
            for (i, &b) in bits.iter().enumerate() {
                got[i / 8] |= u8::from(b) << (i % 8);
            }
            assert_eq!(got, rec.c_tilde());
        }
    }

    /// The block layout the tied part relies on (module docs): w1Encode words 0..47 (lane
    /// `t` at `8 t`), mu at 48..51, `c~` at 52..53, the constant at word 54, chi from word 56.
    #[test]
    fn layout_words_are_as_documented() {
        let gate = sponge_gate();
        let lay = gate.layout();
        assert_eq!(lay.k_log, 19);
        assert_eq!(lay.field_base, vec![128 * W1_WORD_BASE, 128 * MU_WORD_BASE]);
        assert_eq!(lay.field_offset, vec![W1_INPUT_BASE, MU_INPUT_BASE]);
        assert_eq!(lay.output_base, 128 * CTILDE_WORD_BASE);
        assert_eq!(lay.const_col, 128 * CONST_WORD);
        assert_eq!(lay.t_base, 2 * CHI_WORD_BASE);
        assert_eq!(lay.last_lanes, 4);
        assert_eq!(lay.useful_bits, 64 * (2 * CHI_WORD_BASE + (7 * 24 - 1) * 25 + 4));
        assert_eq!(lay.useful_bits, 274_624);
        assert_eq!(PUBLIC_WORD_BASE + (1 << PUBLIC_WORDS_LOG2), CHI_WORD_BASE);
        assert_eq!(MU_WORD_BASE, PUBLIC_WORD_BASE);
        for t in 0..6 {
            for p in [0, 1, 255, 1023] {
                assert_eq!(lay.input_col(W1_INPUT_BASE + 1024 * t + p), 1024 * t + p);
            }
        }
        for i in [0, 7, 511] {
            assert_eq!(lay.input_col(MU_INPUT_BASE + i), 128 * MU_WORD_BASE + i);
        }
        // The public words of a record are its mu and c~ words, in that order.
        let rec = random_record(&mut Rng::new(3));
        let w = lay.useful_words();
        let mut z = vec![0u64; w];
        gate.honest_z(&rec.input_bits(), &mut z);
        let word = |c: usize| F128::new(z[2 * c], z[2 * c + 1]);
        let mu_words = bytes_to_words(&rec.mu);
        let ct_words = bytes_to_words(&rec.c_tilde());
        for k in 0..4 {
            assert_eq!(word(MU_WORD_BASE + k), mu_words[k]);
        }
        for k in 0..2 {
            assert_eq!(word(CTILDE_WORD_BASE + k), ct_words[k]);
        }
        assert_eq!(word(CONST_WORD), F128::ONE);
        assert_eq!(word(CONST_WORD + 1), F128::ZERO);
    }

    #[test]
    fn sponge_rows_satisfy_relation() {
        let gate = sponge_gate();
        assert_eq!(gate.layout().k_log, 19);
        let w = gate.layout().useful_words();
        let mut rng = Rng::new(9);
        let rec = random_record(&mut rng);
        let (mut z, mut a, mut b) = (vec![0u64; w], vec![0u64; w], vec![0u64; w]);
        gate.instance_rows(&rec.input_bits(), &mut z, &mut a, &mut b);
        for i in 0..w {
            assert_eq!(a[i] & b[i], z[i], "word {i}");
        }
        // The committed c~ words are the native digest.
        let out0 = gate.layout().output_base / 64;
        let mut got = [0u8; CTILDE_BYTES];
        for i in 0..4 {
            got[8 * i..8 * i + 8].copy_from_slice(&z[out0 + i].to_le_bytes());
        }
        assert_eq!(got, rec.c_tilde());
        let cost = gate.cost();
        assert_eq!(cost.chi_bits, 7 * 38_400 - 1_344);
        assert_eq!(cost.committed_bits, 1 + 512 + 6_144 + 256 + 7 * 38_400 - 1_344);
        assert_eq!(cost.rows, 275_713 - 1_344);
    }

    fn random_record_for<P: ParamSet>(rng: &mut Rng) -> SpongeRecord<P> {
        let mut mu = [0u8; MU_BYTES];
        let mut w1 = P::zero_w1_encode();
        mu.iter_mut().for_each(|b| *b = rng.next_u32() as u8);
        w1.as_mut().iter_mut().for_each(|b| *b = rng.next_u32() as u8);
        SpongeRecord { mu, w1_encode: w1 }
    }

    /// The three layouts have the documented word columns; ML-DSA-44's are the module
    /// constants.
    #[test]
    fn layouts_have_the_documented_words() {
        let l = SpongeLayout::S44;
        assert_eq!(
            (l.lanes, l.positions_pad, l.lane_words(), l.w1_bytes, l.ctilde_bytes, l.n_perms),
            (6, 1024, 8, W1_BYTES, CTILDE_BYTES, N_PERMS)
        );
        assert_eq!(
            (l.mu_word_base, l.ctilde_word_base, l.const_word, l.chi_word_base),
            (MU_WORD_BASE, CTILDE_WORD_BASE, CONST_WORD, CHI_WORD_BASE)
        );
        assert_eq!((l.public_word_base, l.public_words_log2), (PUBLIC_WORD_BASE, PUBLIC_WORDS_LOG2));
        assert_eq!((l.input_bits(), l.mu_input_base(), l.block_records_log2()), (INPUT_BITS, MU_INPUT_BASE, 9));
        let l = SpongeLayout::S65;
        assert_eq!(
            (l.lanes, l.positions(), l.positions_pad, l.w1_bytes, l.ctilde_bytes, l.n_perms),
            (4, 1536, 2048, 768, 48, 7)
        );
        assert_eq!((l.mu_word_base, l.ctilde_word_base, l.const_word, l.chi_word_base), (64, 68, 71, 72));
        assert_eq!((l.public_words_log2, l.block_records_log2()), (3, 8));
        let l = SpongeLayout::S87;
        assert_eq!(
            (l.lanes, l.positions(), l.positions_pad, l.w1_bytes, l.ctilde_bytes, l.n_perms),
            (4, 2048, 2048, 1024, 64, 9)
        );
        assert_eq!((l.mu_word_base, l.ctilde_word_base, l.const_word, l.chi_word_base), (64, 68, 72, 80));
        assert_eq!((l.public_words_log2, l.block_records_log2()), (4, 8));
        for set in Set::ALL {
            let gate = sponge_gate_for(set);
            assert_eq!(SpongeLayout::of_gate(&gate), Some(SpongeLayout::of(set)));
        }
        assert_eq!(SpongeLayout::of_gate(&ChainGate::new(ChainSpec::permutation())), None);
    }

    /// Per set: the gate's columns of the w1 bits are `P_pad t + 256 rho + j`, mu, `c~` and the
    /// constant sit at the layout's words, the block fits `k_log = 19`, and the honest rows
    /// satisfy the relation with zero in every padded position and in the public region's
    /// zero words.
    fn check_layout<P: ParamSet>(seed: u64) {
        let layout = SpongeLayout::of(P::SET);
        let gate = sponge_gate_for(P::SET);
        let lay = gate.layout();
        assert_eq!(lay.k_log, 19);
        assert_eq!(lay.output_base, 128 * layout.ctilde_word_base);
        assert_eq!(lay.const_col, 128 * layout.const_word);
        assert_eq!(lay.t_base, 2 * layout.chi_word_base);
        assert_eq!(lay.last_lanes, layout.ctilde_bytes / 8);
        assert_eq!(*lay.field_base.last().unwrap(), 128 * layout.mu_word_base);
        let per_poly = 256 * layout.lanes;
        for q in [0, 1, layout.lanes - 1, per_poly + 5, layout.w1_bits() - 1] {
            assert_eq!(lay.input_col(layout.w1_input(q)), layout.w1_column(q), "q = {q}");
        }
        let mut seen = vec![false; layout.lanes * layout.positions_pad];
        for q in 0..layout.w1_bits() {
            let c = layout.w1_column(q);
            assert!(!seen[c] && c % layout.positions_pad < layout.positions());
            seen[c] = true;
        }
        let mut rng = Rng::new(seed);
        let rec = random_record_for::<P>(&mut rng);
        let w = lay.useful_words();
        let (mut z, mut a, mut b) = (vec![0u64; w], vec![0u64; w], vec![0u64; w]);
        gate.instance_rows(&rec.input_bits(), &mut z, &mut a, &mut b);
        for i in 0..w {
            assert_eq!(a[i] & b[i], z[i], "word {i}");
        }
        let word = |c: usize| F128::new(z[2 * c], z[2 * c + 1]);
        for t in 0..layout.lanes {
            for c in layout.positions() / 128..layout.lane_words() {
                assert_eq!(word(layout.lane_words() * t + c), F128::ZERO, "padded word of lane {t}");
            }
        }
        let ct = rec.c_tilde();
        for (k, m) in bytes_to_words(&rec.mu).into_iter().enumerate() {
            assert_eq!(word(layout.mu_word_base + k), m);
        }
        for (k, c) in bytes_to_words(ct.as_ref()).into_iter().enumerate() {
            assert_eq!(word(layout.ctilde_word_base + k), c);
        }
        assert_eq!(word(layout.const_word), F128::ONE);
        for c in layout.const_word + 1..layout.chi_word_base {
            assert_eq!(word(c), F128::ZERO);
        }
        assert_eq!(layout.public_word_base + (1 << layout.public_words_log2), layout.chi_word_base);
    }

    #[test]
    fn layouts_of_65_and_87_hold_the_bits_where_documented() {
        check_layout::<Set44>(1);
        check_layout::<Set65>(2);
        check_layout::<Set87>(3);
    }

    /// The packed input bits equal the bit-by-bit definition through `w1_input`.
    fn check_input_bits<P: ParamSet>(seed: u64) {
        let layout = SpongeLayout::of(P::SET);
        let mut rng = Rng::new(seed);
        for _ in 0..8 {
            let r = random_record_for::<P>(&mut rng);
            let mut want = vec![0u64; layout.input_bits() / 64];
            for (i, &byte) in r.mu.iter().enumerate() {
                let bit = layout.mu_input_base() + 8 * i;
                want[bit / 64] |= u64::from(byte) << (bit % 64);
            }
            let w1: &[u8] = r.w1_encode.as_ref();
            for q in 0..layout.w1_bits() {
                if (w1[q / 8] >> (q % 8)) & 1 == 1 {
                    let idx = layout.w1_input(q);
                    want[idx / 64] |= 1 << (idx % 64);
                }
            }
            assert_eq!(r.input_bits(), want);
        }
    }

    #[test]
    fn input_bits_follow_the_layout_on_every_set() {
        check_input_bits::<Set44>(11);
        check_input_bits::<Set65>(12);
        check_input_bits::<Set87>(13);
        // Stream bit 1024 rho + 4 j + t of a four-bit w1Encode is lane t, position 256 rho + j.
        assert_eq!(SpongeLayout::S65.w1_input(1024 + 4 * 5 + 3), 1536 * 3 + 256 + 5);
        assert_eq!(SpongeLayout::S65.w1_column(1024 + 4 * 5 + 3), 2048 * 3 + 256 + 5);
        assert_eq!(SpongeLayout::S87.w1_input(7 * 1024 + 4 * 255 + 1), 2048 + 7 * 256 + 255);
    }

    /// Known answers: the gate's output bits equal SHAKE256 of the `sha3` crate on random
    /// records, and the absorbed padding is at the documented bits (ML-DSA-87: the ninth
    /// permutation absorbs only the padding).
    fn check_shake<P: ParamSet>(seed: u64) {
        let layout = SpongeLayout::of(P::SET);
        let map = absorb_map_for(&layout);
        assert_eq!(map.len(), layout.n_perms);
        let inputs = map.iter().flatten().filter(|(_, s)| matches!(s, Absorb::Input(_))).count();
        assert_eq!(inputs, 8 * layout.message_bytes());
        let ones: Vec<(usize, u16)> = map
            .iter()
            .enumerate()
            .flat_map(|(m, list)| list.iter().filter(|(_, s)| *s == Absorb::One).map(move |(b, _)| (m, *b)))
            .collect();
        let pad = 8 * layout.message_bytes() % 1088;
        let last = layout.n_perms - 1;
        let want: Vec<(usize, u16)> = (0..5).map(|k| (last, (pad + k) as u16)).chain([(last, 1087)]).collect();
        assert_eq!(ones, want);
        let gate = sponge_gate_for(P::SET);
        let mut rng = Rng::new(seed);
        for _ in 0..6 {
            let rec = random_record_for::<P>(&mut rng);
            let bits = gate.output_bits(&rec.input_bits());
            let mut got = P::zero_c_tilde();
            for (i, &b) in bits.iter().enumerate() {
                got.as_mut()[i / 8] |= u8::from(b) << (i % 8);
            }
            let mut h = Shake256::default();
            h.update(&rec.mu);
            h.update(rec.w1_encode.as_ref());
            let mut want = P::zero_c_tilde();
            h.finalize_xof().read(want.as_mut());
            assert_eq!(got, want);
            assert_eq!(got, rec.c_tilde());
        }
    }

    #[test]
    fn gates_match_shake256_on_every_set() {
        check_shake::<Set65>(0x65);
        check_shake::<Set87>(0x87);
        assert_eq!(SpongeLayout::S87.message_bytes(), 8 * RATE_BYTES);
    }

    /// Known answers on real signatures: the gate's output on `(mu, w1Encode(w1'))` of a
    /// strictly derived witness is the signature's `c~` (FIPS 204 Algorithm 8, line 12).
    fn check_signatures<P: ParamSet>(seed: u64, n: usize) {
        use agg_witness::derive::derive_for;
        use agg_witness::instance::PublicInstance;
        let records = agg_witness::gen::records_for::<P>(seed, n);
        let instance = PublicInstance::<P>::from_records_for(&records).expect("instance");
        let gate = sponge_gate_for(P::SET);
        for (record, public) in records.iter().zip(&instance.records) {
            let w = derive_for::<P>(&public.key, &public.mu, &record.sig).expect("verifies");
            let rec = SpongeRecord::<P> { mu: public.mu, w1_encode: w.w1_bytes };
            let bits = gate.output_bits(&rec.input_bits());
            let mut got = P::zero_c_tilde();
            for (i, &b) in bits.iter().enumerate() {
                got.as_mut()[i / 8] |= u8::from(b) << (i % 8);
            }
            assert_eq!(got.as_ref(), &record.sig[..P::C_TILDE_BYTES]);
            assert_eq!(got, public.c_tilde);
        }
    }

    #[test]
    fn gates_reproduce_the_c_tilde_of_real_signatures() {
        check_signatures::<Set65>(65, 3);
        check_signatures::<Set87>(87, 3);
    }
}
