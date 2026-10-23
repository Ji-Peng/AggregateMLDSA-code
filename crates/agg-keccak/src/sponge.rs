//! SHAKE256 sponge circuit for `c~ = H(mu || w1Encode(w1'))` (spec Section 8).
//!
//! One gate instance per signature: seven chained chi-only Keccak-f[1600]
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
//! `7 x 24 x 1,600` chi bits.
//!
//! Input column order of the w1Encode bits is `(t, j, rho)`, lane `t` outermost:
//! column `1024 t + 256 rho + j` holds `b_t(j, rho)`, stream bit
//! `1536 rho + 6 j + t` of w1Encode (FIPS 204 Algorithms 16 and 28). Each lane
//! is 8 aligned 128-bit words (the bridge slot copies these 48 words).

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
/// Global input index of the first w1Encode bit (mu occupies `[0, 512)`).
pub const W1_INPUT_BASE: usize = 8 * MU_BYTES;

/// One signature's sponge input.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SpongeRecord {
    pub mu: [u8; MU_BYTES],
    pub w1_encode: [u8; W1_BYTES],
}

impl SpongeRecord {
    /// `c~ = SHAKE256(mu || w1Encode, 32)` natively (sha3 crate).
    #[must_use]
    pub fn c_tilde(&self) -> [u8; CTILDE_BYTES] {
        let mut h = Shake256::default();
        h.update(&self.mu);
        h.update(&self.w1_encode);
        let mut out = [0u8; CTILDE_BYTES];
        h.finalize_xof().read(&mut out);
        out
    }

    /// Packed gate input bits: mu bits `[0, 512)`, then the w1Encode bits in
    /// `(t, j, rho)` column order.
    #[must_use]
    pub fn input_bits(&self) -> Vec<u64> {
        let mut words = vec![0u64; (W1_INPUT_BASE + W1_BITS) / 64];
        for (i, &byte) in self.mu.iter().enumerate() {
            words[i / 8] |= u64::from(byte) << (8 * (i % 8));
        }
        for q in 0..W1_BITS {
            if (self.w1_encode[q / 8] >> (q % 8)) & 1 == 1 {
                let idx = W1_INPUT_BASE + w1_column(q);
                words[idx / 64] |= 1 << (idx % 64);
            }
        }
        words
    }
}

/// Column (within the w1 field) of w1Encode stream bit `q = 1536 rho + 6 j + t`.
#[inline]
#[must_use]
pub const fn w1_column(q: usize) -> usize {
    let rho = q / 1536;
    let rem = q % 1536;
    let j = rem / 6;
    let t = rem % 6;
    1024 * t + 256 * rho + j
}

/// The sponge's absorbed `(state bit, source)` pairs per permutation.
#[must_use]
pub fn absorb_map() -> Vec<Vec<(u16, Absorb)>> {
    let w1 = |q: usize| Absorb::Input((W1_INPUT_BASE + w1_column(q)) as u32);
    let mut perms = Vec::with_capacity(N_PERMS);
    // P1: mu then w1Encode bytes 0..71.
    let mut p1: Vec<(u16, Absorb)> = (0..8 * MU_BYTES)
        .map(|i| (i as u16, Absorb::Input(i as u32)))
        .collect();
    p1.extend((8 * MU_BYTES..8 * RATE_BYTES).map(|s| (s as u16, w1(s - 8 * MU_BYTES))));
    perms.push(p1);
    // P2..P6: 136 w1Encode bytes each.
    let first = 8 * (RATE_BYTES - MU_BYTES);
    for m in 0..5 {
        let q0 = first + 8 * RATE_BYTES * m;
        perms.push(
            (0..8 * RATE_BYTES)
                .map(|s| (s as u16, w1(q0 + s)))
                .collect(),
        );
    }
    // P7: the last 16 bytes, then the SHAKE padding 0x1F .. 0x80.
    let q0 = first + 8 * RATE_BYTES * 5;
    debug_assert_eq!(q0, 8 * 752);
    let mut p7: Vec<(u16, Absorb)> = (0..W1_BITS - q0).map(|s| (s as u16, w1(q0 + s))).collect();
    let pad = W1_BITS - q0; // first bit of the padding byte 0x1F
    for k in 0..5 {
        p7.push(((pad + k) as u16, Absorb::One)); // 0x1F
    }
    p7.push(((8 * RATE_BYTES - 1) as u16, Absorb::One)); // 0x80 of byte 135
    perms.push(p7);
    perms
}

/// The sponge gate spec with `n_rounds` rounds per permutation (24; fewer only in tests).
#[must_use]
pub fn sponge_spec_rounds(n_rounds: usize) -> ChainSpec {
    ChainSpec {
        label: "shake256(mu||w1encode)->ctilde",
        n_rounds,
        input_fields: vec![
            InputField {
                name: "mu",
                bits: 8 * MU_BYTES,
                public: true,
            },
            InputField {
                name: "w1encode",
                bits: W1_BITS,
                public: false,
            },
        ],
        absorbs: absorb_map(),
        outputs: (0..8 * CTILDE_BYTES as u16).collect(),
    }
}

/// The production sponge gate.
#[must_use]
pub fn sponge_gate() -> ChainGate {
    ChainGate::new(sponge_spec_rounds(N_ROUNDS))
}

/// Little-endian 128-bit words of a byte string whose length is a multiple of 16.
#[must_use]
pub fn bytes_to_words(bytes: &[u8]) -> Vec<F128> {
    assert!(bytes.len().is_multiple_of(16));
    bytes
        .chunks(16)
        .map(|c| {
            let lo = u64::from_le_bytes(c[..8].try_into().expect("8 bytes"));
            let hi = u64::from_le_bytes(c[8..].try_into().expect("8 bytes"));
            F128::new(lo, hi)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use flock_core::test_rng::Rng;

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
        assert_eq!(cost.chi_bits, 7 * 38_400);
        assert_eq!(cost.committed_bits, 1 + 512 + 6_144 + 256 + 7 * 38_400);
    }
}
