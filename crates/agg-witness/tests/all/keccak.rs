//! Keccak sponge inputs and outputs per signature against sha3's SHAKE256 and the signature c~.

#![allow(clippy::needless_range_loop)]

use crate::common;

use agg_witness::instance::PublicInstance;
use agg_witness::keccak::{mu_bit_slot, sponge_trace, w1_bit_slot, PERMS, RATE};
use agg_witness::native::shake256;
use agg_witness::params::*;

fn state_bit(s: &[u64; 25], bit: usize) -> u64 {
    (s[bit / 64] >> (bit % 64)) & 1
}

#[test]
fn sponge_of_every_signature_matches_c_tilde() {
    let records = common::mixed_records(500, 100, 28);
    let inst = PublicInstance::from_records(&records).unwrap();
    let wits = common::witnesses(&inst, &records);
    for (i, w) in wits.iter().enumerate() {
        let t = sponge_trace(&w.mu, &w.w1_bytes);
        assert_eq!(w.sponge(), t);
        let mut reference = [0u8; 32];
        shake256(&[&w.mu, &w.w1_bytes], &mut reference);
        assert_eq!(t.c_tilde, reference);
        assert_eq!(t.c_tilde, w.c_tilde);
        assert_eq!(&records[i].sig[..32], &t.c_tilde[..]);
        assert_eq!(t.inputs.len(), PERMS);
        assert_eq!(&t.blocks[0][..64], &w.mu[..]);
        // Wiring: the input state bit of b_t(j, rho) is b_t XOR the previous output bit.
        for rho in 0..K {
            for j in 0..N {
                for bit in 0..W1_BITS {
                    let slot = w1_bit_slot(bit, j, rho);
                    let prev = if slot.perm == 0 {
                        0
                    } else {
                        state_bit(&t.outputs[slot.perm - 1], slot.state_bit)
                    };
                    let b = u64::from(w.bit(bit, j, rho));
                    assert_eq!(state_bit(&t.inputs[slot.perm], slot.state_bit), b ^ prev);
                }
            }
        }
        for k in 0..512 {
            let s = mu_bit_slot(k);
            assert_eq!(
                state_bit(&t.inputs[0], s.state_bit),
                u64::from((w.mu[k / 8] >> (k % 8)) & 1)
            );
        }
        // Fixed padding bits of P_7 and zero capacity before P_1.
        assert_eq!(t.blocks[PERMS - 1][16], 0x1F);
        assert_eq!(t.blocks[PERMS - 1][RATE - 1], 0x80);
        assert!(t.inputs[0][RATE / 8..].iter().all(|&x| x == 0));
    }
}
