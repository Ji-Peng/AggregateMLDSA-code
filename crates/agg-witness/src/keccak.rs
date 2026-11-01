//! Keccak sponge inputs and outputs of the commitment hash.
//!
//! Per live record the Keccak argument proves c~ = SHAKE256(mu || w1Encode(w1'), 32): 832 input
//! bytes, rate 136, so 7 Keccak-f[1600] permutations P_1..P_7. This module computes the absorbed
//! blocks, the state before and after every permutation, and the wiring of the 6,144 w1Encode
//! bits b_t(j, rho) into state bits. Keccak-f[1600] is implemented here (FIPS 202) and checked
//! against the `sha3` crate's SHAKE256.

use crate::params::{ParamSet, Set44, N, TR_BYTES, W1_ENCODE_BYTES};

/// SHAKE256 rate in bytes.
pub const RATE: usize = 136;
/// Rate in bits.
pub const RATE_BITS: usize = RATE * 8;
/// Number of permutations of the commitment hash.
pub const PERMS: usize = 7;
/// Total message bytes: mu (64) || w1Encode (768).
pub const MSG_BYTES: usize = TR_BYTES + W1_ENCODE_BYTES;

/// A Keccak state as 25 little-endian lanes, lane (x, y) at index x + 5 y.
pub type State = [u64; 25];

const RC: [u64; 24] = [
    0x0000_0000_0000_0001,
    0x0000_0000_0000_8082,
    0x8000_0000_0000_808A,
    0x8000_0000_8000_8000,
    0x0000_0000_0000_808B,
    0x0000_0000_8000_0001,
    0x8000_0000_8000_8081,
    0x8000_0000_0000_8009,
    0x0000_0000_0000_008A,
    0x0000_0000_0000_0088,
    0x0000_0000_8000_8009,
    0x0000_0000_8000_000A,
    0x0000_0000_8000_808B,
    0x8000_0000_0000_008B,
    0x8000_0000_0000_8089,
    0x8000_0000_0000_8003,
    0x8000_0000_0000_8002,
    0x8000_0000_0000_0080,
    0x0000_0000_0000_800A,
    0x8000_0000_8000_000A,
    0x8000_0000_8000_8081,
    0x8000_0000_0000_8080,
    0x0000_0000_8000_0001,
    0x8000_0000_8000_8008,
];

/// Rotation offsets r[x + 5 y] (FIPS 202 Table 2).
const ROT: [u32; 25] = [
    0, 1, 62, 28, 27, 36, 44, 6, 55, 20, 3, 10, 43, 25, 39, 41, 45, 15, 21, 8, 18, 2, 61, 56, 14,
];

/// Keccak-f[1600] (FIPS 202 Algorithm 7 with theta, rho, pi, chi, iota).
pub fn keccak_f1600(a: &mut State) {
    for rc in RC {
        // theta
        let mut c = [0u64; 5];
        for x in 0..5 {
            c[x] = a[x] ^ a[x + 5] ^ a[x + 10] ^ a[x + 15] ^ a[x + 20];
        }
        for x in 0..5 {
            let d = c[(x + 4) % 5] ^ c[(x + 1) % 5].rotate_left(1);
            for y in 0..5 {
                a[x + 5 * y] ^= d;
            }
        }
        // rho and pi: B[y, 2x + 3y] = rot(A[x, y], r[x, y])
        let mut b = [0u64; 25];
        for x in 0..5 {
            for y in 0..5 {
                b[y + 5 * ((2 * x + 3 * y) % 5)] = a[x + 5 * y].rotate_left(ROT[x + 5 * y]);
            }
        }
        // chi
        for y in 0..5 {
            for x in 0..5 {
                a[x + 5 * y] = b[x + 5 * y] ^ (!b[(x + 1) % 5 + 5 * y] & b[(x + 2) % 5 + 5 * y]);
            }
        }
        // iota
        a[0] ^= rc;
    }
}

/// XORs a 136-byte block into the low 1,088 bits of the state (byte beta -> bits 8 beta..8 beta + 7).
pub fn absorb_block(state: &mut State, block: &[u8; RATE]) {
    for (lane, chunk) in block.chunks(8).enumerate() {
        let mut w = [0u8; 8];
        w.copy_from_slice(chunk);
        state[lane] ^= u64::from_le_bytes(w);
    }
}

/// State bytes in little-endian lane order.
pub fn state_bytes(state: &State) -> [u8; 200] {
    let mut out = [0u8; 200];
    for (i, lane) in state.iter().enumerate() {
        out[8 * i..8 * i + 8].copy_from_slice(&lane.to_le_bytes());
    }
    out
}

/// Absorbed blocks for parameter set P: mu || w1_bytes with SHAKE256 domain and pad10*1.
pub fn sponge_blocks_for<P: ParamSet>(mu: &[u8; TR_BYTES], w1_bytes: &[u8]) -> Vec<[u8; RATE]> {
    let msg_bytes = TR_BYTES + P::W1_ENCODE_BYTES;
    assert_eq!(w1_bytes.len(), P::W1_ENCODE_BYTES);
    let mut msg = vec![0u8; P::N_PERMS * RATE];
    msg[..TR_BYTES].copy_from_slice(mu);
    msg[TR_BYTES..msg_bytes].copy_from_slice(w1_bytes);
    // SHAKE domain suffix 1111 then pad10*1 (FIPS 202): 0x1F after the message, 0x80 at the end.
    msg[msg_bytes] ^= 0x1F;
    msg[P::N_PERMS * RATE - 1] ^= 0x80;
    let mut blocks = Vec::with_capacity(P::N_PERMS);
    for m in 0..P::N_PERMS {
        let mut b = [0u8; RATE];
        b.copy_from_slice(&msg[m * RATE..(m + 1) * RATE]);
        blocks.push(b);
    }
    blocks
}

/// The seven absorbed blocks: P_1 = mu || w1 bytes 0..71, P_2..P_6 = 136 w1 bytes each,
/// P_7 = w1 bytes 752..767 || 0x1F || 0x00^118 || 0x80.
pub fn sponge_blocks(mu: &[u8; TR_BYTES], w1_bytes: &[u8; W1_ENCODE_BYTES]) -> [[u8; RATE]; PERMS] {
    let blocks_vec = sponge_blocks_for::<Set44>(mu, w1_bytes);
    let mut blocks = [[0u8; RATE]; PERMS];
    for m in 0..PERMS {
        blocks[m] = blocks_vec[m];
    }
    blocks
}

/// Inputs and outputs of the permutations of one record.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SpongeTrace<P: ParamSet = Set44> {
    /// Absorbed blocks.
    pub blocks: Vec<[u8; RATE]>,
    /// State before P_m (previous output XOR block m; the initial state is zero).
    pub inputs: Vec<State>,
    /// State after P_m.
    pub outputs: Vec<State>,
    /// Commitment hash c~.
    pub c_tilde: P::CTilde,
}

/// Runs the commitment-hash sponge for parameter set P.
pub fn sponge_trace_for<P: ParamSet>(mu: &[u8; TR_BYTES], w1_bytes: &[u8]) -> SpongeTrace<P> {
    let blocks = sponge_blocks_for::<P>(mu, w1_bytes);
    let mut inputs = Vec::with_capacity(P::N_PERMS);
    let mut outputs = Vec::with_capacity(P::N_PERMS);
    let mut s: State = [0; 25];
    for m in 0..P::N_PERMS {
        absorb_block(&mut s, &blocks[m]);
        inputs.push(s);
        keccak_f1600(&mut s);
        outputs.push(s);
    }
    let sb = state_bytes(&outputs[P::N_PERMS - 1]);
    let c_tilde = P::c_tilde_from_slice(&sb[..P::C_TILDE_BYTES]);
    SpongeTrace {
        blocks,
        inputs,
        outputs,
        c_tilde,
    }
}

/// Runs the commitment-hash sponge.
pub fn sponge_trace(mu: &[u8; TR_BYTES], w1_bytes: &[u8; W1_ENCODE_BYTES]) -> SpongeTrace<Set44> {
    sponge_trace_for::<Set44>(mu, w1_bytes)
}

/// Position of a message bit in the sponge: permutation index m in [0, N_PERMS) and state bit in [0, 1088).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BitSlot {
    /// Permutation whose input block carries the bit (0-based: P_{m+1}).
    pub perm: usize,
    /// State bit index: lane = bit / 64, bit in lane = bit % 64.
    pub state_bit: usize,
}

/// Index of b_t(j, rho) in the w1Encode bit string for parameter set P.
pub fn w1_bit_index_for<P: ParamSet>(t: usize, j: usize, rho: usize) -> usize {
    debug_assert!(t < P::W1_BITS && j < N && rho < P::K);
    rho * N * P::W1_BITS + j * P::W1_BITS + t
}

/// Wiring of b_t(j, rho) into the sponge for parameter set P.
pub fn w1_bit_slot_for<P: ParamSet>(t: usize, j: usize, rho: usize) -> BitSlot {
    let msg_bit = TR_BYTES * 8 + w1_bit_index_for::<P>(t, j, rho);
    BitSlot {
        perm: msg_bit / RATE_BITS,
        state_bit: msg_bit % RATE_BITS,
    }
}

/// Index of b_t(j, rho) in the w1Encode bit string: 1536 rho + 6 j + t.
pub fn w1_bit_index(t: usize, j: usize, rho: usize) -> usize {
    w1_bit_index_for::<Set44>(t, j, rho)
}

/// Wiring of b_t(j, rho) into the sponge (a fixed index permutation).
pub fn w1_bit_slot(t: usize, j: usize, rho: usize) -> BitSlot {
    w1_bit_slot_for::<Set44>(t, j, rho)
}

/// Wiring of bit `k` of mu (public input) into the sponge.
pub fn mu_bit_slot(k: usize) -> BitSlot {
    assert!(k < TR_BYTES * 8);
    BitSlot {
        perm: 0,
        state_bit: k,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gen::SplitMix64;
    use crate::native::shake256;
    use crate::params::{Set65, Set87, K, W1_BITS, W1_ENCODE_BYTES};

    #[test]
    fn keccak_f_zero_state_vector() {
        // Keccak-f[1600] applied to the zero state (Keccak team test vector).
        let mut s: State = [0; 25];
        keccak_f1600(&mut s);
        assert_eq!(s[0], 0xF125_8F79_40E1_DDE7);
        assert_eq!(s[1], 0x84D5_CCF9_33C0_478A);
        assert_eq!(s[24], 0xEAF1_FF7B_5CEC_A249);
    }

    #[test]
    fn sponge_matches_sha3_shake256() {
        let mut rng = SplitMix64::new(23);
        for _ in 0..20 {
            let mut mu = [0u8; TR_BYTES];
            let mut w1 = [0u8; W1_ENCODE_BYTES];
            rng.fill(&mut mu);
            rng.fill(&mut w1);
            let t = sponge_trace(&mu, &w1);
            let mut expect = [0u8; 32];
            shake256(&[&mu, &w1], &mut expect);
            assert_eq!(t.c_tilde, expect);
            // Chain: input of P_m = output of P_{m-1} XOR block m.
            for m in 1..PERMS {
                let mut s = t.outputs[m - 1];
                absorb_block(&mut s, &t.blocks[m]);
                assert_eq!(s, t.inputs[m]);
            }
            // Capacity of the first input is zero.
            assert!(t.inputs[0][RATE / 8..].iter().all(|&x| x == 0));
        }
    }

    #[test]
    fn block_layout_of_section_8() {
        let mu = [0xAAu8; TR_BYTES];
        let mut w1 = [0u8; W1_ENCODE_BYTES];
        for (i, b) in w1.iter_mut().enumerate() {
            *b = i as u8;
        }
        let blocks = sponge_blocks(&mu, &w1);
        assert_eq!(&blocks[0][..64], &mu[..]);
        assert_eq!(&blocks[0][64..], &w1[..72]);
        for m in 2..=6 {
            let start = 72 + 136 * (m - 2);
            assert_eq!(&blocks[m - 1][..], &w1[start..start + 136]);
        }
        assert_eq!(&blocks[6][..16], &w1[752..768]);
        assert_eq!(blocks[6][16], 0x1F);
        assert!(blocks[6][17..135].iter().all(|&b| b == 0));
        assert_eq!(blocks[6][135], 0x80);
    }

    #[test]
    fn wiring_reproduces_blocks() {
        let mut rng = SplitMix64::new(29);
        let mut w1 = [[0u8; N]; K];
        for row in w1.iter_mut() {
            for x in row.iter_mut() {
                *x = rng.below(44) as u8;
            }
        }
        let bytes = crate::native::w1_encode(&w1);
        let mut mu = [0u8; TR_BYTES];
        rng.fill(&mut mu);
        let blocks = sponge_blocks(&mu, &bytes);
        let mut rebuilt = [[0u8; RATE]; PERMS];
        for k in 0..TR_BYTES * 8 {
            let s = mu_bit_slot(k);
            rebuilt[s.perm][s.state_bit / 8] |= ((mu[k / 8] >> (k % 8)) & 1) << (s.state_bit % 8);
        }
        for rho in 0..K {
            for j in 0..N {
                for t in 0..W1_BITS {
                    let bit = (w1[rho][j] >> t) & 1;
                    let s = w1_bit_slot(t, j, rho);
                    rebuilt[s.perm][s.state_bit / 8] |= bit << (s.state_bit % 8);
                }
            }
        }
        rebuilt[6][16] ^= 0x1F;
        rebuilt[6][135] ^= 0x80;
        assert_eq!(rebuilt, blocks);
        // The last w1 bit lands in P_7 at state bit 127.
        assert_eq!(
            w1_bit_slot(5, 255, 3),
            BitSlot {
                perm: 6,
                state_bit: 127
            }
        );
        assert_eq!(
            w1_bit_slot(0, 0, 0),
            BitSlot {
                perm: 0,
                state_bit: 512
            }
        );
    }

    #[test]
    fn sponge_matches_sha3_shake256_65() {
        let mut rng = SplitMix64::new(65);
        for _ in 0..10 {
            let mut mu = [0u8; TR_BYTES];
            let mut w1 = vec![0u8; Set65::W1_ENCODE_BYTES];
            rng.fill(&mut mu);
            rng.fill(&mut w1);
            let t = sponge_trace_for::<Set65>(&mu, &w1);
            let mut expect = [0u8; 48];
            shake256(&[&mu, &w1], &mut expect);
            assert_eq!(t.c_tilde, expect);
            assert_eq!(t.blocks.len(), Set65::N_PERMS);
            assert_eq!(Set65::N_PERMS, 7);
            for m in 1..Set65::N_PERMS {
                let mut s = t.outputs[m - 1];
                absorb_block(&mut s, &t.blocks[m]);
                assert_eq!(s, t.inputs[m]);
            }
            assert!(t.inputs[0][RATE / 8..].iter().all(|&x| x == 0));
        }
    }

    #[test]
    fn sponge_matches_sha3_shake256_87() {
        let mut rng = SplitMix64::new(87);
        for _ in 0..10 {
            let mut mu = [0u8; TR_BYTES];
            let mut w1 = vec![0u8; Set87::W1_ENCODE_BYTES];
            rng.fill(&mut mu);
            rng.fill(&mut w1);
            let t = sponge_trace_for::<Set87>(&mu, &w1);
            let mut expect = [0u8; 64];
            shake256(&[&mu, &w1], &mut expect);
            assert_eq!(t.c_tilde, expect);
            assert_eq!(t.blocks.len(), Set87::N_PERMS);
            assert_eq!(Set87::N_PERMS, 9);
            for m in 1..Set87::N_PERMS {
                let mut s = t.outputs[m - 1];
                absorb_block(&mut s, &t.blocks[m]);
                assert_eq!(s, t.inputs[m]);
            }
            assert!(t.inputs[0][RATE / 8..].iter().all(|&x| x == 0));
        }
    }

    #[test]
    fn wiring_reproduces_blocks_65() {
        let mut rng = SplitMix64::new(6529);
        let mut w1 = [[0u8; N]; Set65::K];
        for row in w1.iter_mut() {
            for x in row.iter_mut() {
                *x = rng.below(16) as u8;
            }
        }
        let bytes = crate::native::w1_encode_for::<Set65>(&w1);
        let mut mu = [0u8; TR_BYTES];
        rng.fill(&mut mu);
        let blocks = sponge_blocks_for::<Set65>(&mu, &bytes);
        let mut rebuilt = vec![[0u8; RATE]; Set65::N_PERMS];
        for k in 0..TR_BYTES * 8 {
            let s = mu_bit_slot(k);
            rebuilt[s.perm][s.state_bit / 8] |= ((mu[k / 8] >> (k % 8)) & 1) << (s.state_bit % 8);
        }
        for rho in 0..Set65::K {
            for j in 0..N {
                for t in 0..Set65::W1_BITS {
                    let bit = (w1[rho][j] >> t) & 1;
                    let s = w1_bit_slot_for::<Set65>(t, j, rho);
                    rebuilt[s.perm][s.state_bit / 8] |= bit << (s.state_bit % 8);
                }
            }
        }
        rebuilt[6][16] ^= 0x1F;
        rebuilt[6][135] ^= 0x80;
        assert_eq!(rebuilt, blocks);
    }

    #[test]
    fn wiring_reproduces_blocks_87() {
        let mut rng = SplitMix64::new(8729);
        let mut w1 = [[0u8; N]; Set87::K];
        for row in w1.iter_mut() {
            for x in row.iter_mut() {
                *x = rng.below(16) as u8;
            }
        }
        let bytes = crate::native::w1_encode_for::<Set87>(&w1);
        let mut mu = [0u8; TR_BYTES];
        rng.fill(&mut mu);
        let blocks = sponge_blocks_for::<Set87>(&mu, &bytes);
        let mut rebuilt = vec![[0u8; RATE]; Set87::N_PERMS];
        for k in 0..TR_BYTES * 8 {
            let s = mu_bit_slot(k);
            rebuilt[s.perm][s.state_bit / 8] |= ((mu[k / 8] >> (k % 8)) & 1) << (s.state_bit % 8);
        }
        for rho in 0..Set87::K {
            for j in 0..N {
                for t in 0..Set87::W1_BITS {
                    let bit = (w1[rho][j] >> t) & 1;
                    let s = w1_bit_slot_for::<Set87>(t, j, rho);
                    rebuilt[s.perm][s.state_bit / 8] |= bit << (s.state_bit % 8);
                }
            }
        }
        rebuilt[8][0] ^= 0x1F;
        rebuilt[8][135] ^= 0x80;
        assert_eq!(rebuilt, blocks);
    }
}
