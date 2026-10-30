//! ExpandA for one key on the 4-way Keccak state, without the lane scheduler.
//!
//! The 16 SHAKE128 streams of a key all have the same 34-byte input shape, so
//! row `r` runs as one fixed batch of 4 lanes (columns 0..3), as in
//! pq-crystals `poly_uniform_4x`: absorb the seed and nonces in place, squeeze
//! five blocks per lane into contiguous buffers (256 candidates take 768 bytes,
//! so no polynomial completes in fewer), sample each buffer straight into its
//! polynomial, and squeeze single blocks only for lanes still open. Sampling
//! after the whole squeeze keeps the sampler's 32-byte loads away from the
//! extraction stores still in flight. Outputs and block counts equal those of
//! the scheduler path.

use crate::keccak::{avx2 as keccak, lanes::State};
use crate::ntt::avx2::order_from_fips;
use crate::poly::{ExpandedA, Poly32};
use crate::sample::avx2::rej_uniform_into;
use core::arch::x86_64::*;

/// Bytes of a lane buffer: five rate blocks and the sampler's 8-byte overread.
pub(crate) const BUF: usize = 5 * 168 + 8;

/// Lane buffers, `0[s]` for lane `s`.
#[repr(C, align(64))]
pub(crate) struct Buffers(pub [[u8; BUF]; 4]);

/// Samples one row. `squeeze(bufs, off)` writes the next rate block of every
/// lane to `bufs[s][off..off + 168]`; `out[s]` receives the 256 accepted
/// coefficients of lane `s` (FIPS order) and `blocks[s]` the number of blocks
/// lane `s` consumed.
#[inline]
#[target_feature(enable = "avx2,popcnt")]
pub(crate) fn sample_row(
    bufs: &mut Buffers,
    mut squeeze: impl FnMut(&mut [[u8; BUF]; 4], usize),
    out: &mut [Poly32; 4],
    blocks: &mut [u8; 4],
) {
    for k in 0..5 {
        squeeze(&mut bufs.0, 168 * k);
    }
    let mut ctr = [0usize; 4];
    let mut open = false;
    for s in 0..4 {
        let (c, used) = rej_uniform_into(&bufs.0[s], 5, &mut out[s].0, 0);
        ctr[s] = c;
        blocks[s] = used as u8;
        open |= c < 256;
    }
    while open {
        squeeze(&mut bufs.0, 0);
        open = false;
        for s in 0..4 {
            if ctr[s] < 256 {
                blocks[s] = blocks[s]
                    .checked_add(1)
                    .expect("ExpandA block counter overflow");
                ctr[s] = rej_uniform_into(&bufs.0[s], 1, &mut out[s].0, ctr[s]).0;
                open |= ctr[s] < 256;
            }
        }
    }
}

/// The state, then the lane buffers of one key.
#[repr(C, align(64))]
struct Scratch {
    state: State<4>,
    bufs: Buffers,
}

/// Expands one seed into backend-order `A_hat` and its block counts.
#[target_feature(enable = "avx2,popcnt")]
pub(crate) fn expand_a(rho: &[u8; 32], out: &mut ExpandedA) {
    // Zeroed once: the tail bytes the sampler reads past the data.
    let mut sc = Scratch {
        state: State([[0; 4]; 25]),
        bufs: Buffers([[0; BUF]; 4]),
    };
    let seed: [__m256i; 4] = core::array::from_fn(|w| {
        _mm256_set1_epi64x(i64::from_le_bytes(
            rho[8 * w..8 * w + 8].try_into().unwrap(),
        ))
    });
    for r in 0..4 {
        // Absorb in place: words 0..3 the seed, word 4 the nonce bytes (column,
        // row) and the suffix 0x1F, word 20 the final padding bit; every
        // other word restarts at zero.
        let nonce = |s: usize| (s | r << 8 | 0x1F << 16) as i64;
        let word4 = _mm256_setr_epi64x(nonce(0), nonce(1), nonce(2), nonce(3));
        for (w, word) in sc.state.0.iter_mut().enumerate() {
            let v = match w {
                0..4 => seed[w],
                4 => word4,
                20 => _mm256_set1_epi64x(i64::MIN),
                _ => _mm256_setzero_si256(),
            };
            // SAFETY: each word is an array of four u64 (32 bytes).
            unsafe { _mm256_storeu_si256(word.as_mut_ptr().cast(), v) };
        }
        let Scratch { state, bufs } = &mut sc;
        sample_row(
            bufs,
            |bufs, off| {
                keccak::permute(state);
                keccak::extract_at(state, bufs, off);
            },
            &mut out.a_hat[r],
            &mut out.blocks[r],
        );
        for poly in &mut out.a_hat[r] {
            // SAFETY: this function's target features include those of the order map.
            unsafe { order_from_fips(poly) };
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dispatch::{Backend, Engine};
    use crate::poly::{Block, RejState};

    #[test]
    fn extra_blocks_match_scalar_sampler() {
        if Engine::new(Backend::Avx2).is_none() {
            return;
        }
        // Lane s receives 2 s whole rejected blocks (all candidates
        // 0x7FFFFF >= q) after its first block, so it needs at least 5 + 2 s.
        let mut rng = 0x9E37_79B9_7F4A_7C15u64;
        let mut byte = move || {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            rng as u8
        };
        let mut stream: [Vec<Block>; 4] = Default::default();
        for (s, lane) in stream.iter_mut().enumerate() {
            for k in 0..16 {
                let mut b = Block::zero();
                if (1..=2 * s).contains(&k) {
                    b.bytes[..168].fill(0xFF);
                } else {
                    for x in &mut b.bytes[..168] {
                        *x = byte();
                    }
                }
                lane.push(b);
            }
        }
        let mut bufs = Buffers([[0; BUF]; 4]);
        let mut out = [Poly32::zero(); 4];
        let mut blocks = [0u8; 4];
        let mut k = 0;
        // SAFETY: AVX2 and POPCNT were detected above.
        unsafe {
            sample_row(
                &mut bufs,
                |b, off| {
                    for s in 0..4 {
                        b[s][off..off + 168].copy_from_slice(&stream[s][k].bytes[..168]);
                    }
                    k += 1;
                },
                &mut out,
                &mut blocks,
            )
        };
        let scalar = Engine::new(Backend::Scalar).unwrap();
        for s in 0..4 {
            let mut st = RejState::new();
            let mut used = 0u8;
            for b in &stream[s] {
                used += 1;
                if scalar.rej_uniform(b, &mut st) {
                    break;
                }
            }
            assert_eq!(blocks[s], used, "lane {s}");
            assert!(used as usize >= 5 + 2 * s);
            assert_eq!(out[s].0, st.coeffs[..256], "lane {s}");
        }
    }

    #[test]
    fn sampler_matches_per_block_kernel() {
        if Engine::new(Backend::Avx2).is_none() {
            return;
        }
        // Rejections placed so that the polynomial completes at every candidate
        // of the fifth block (vector steps and the one-at-a-time tail), or not
        // at all; the tail bytes hold candidates that must be ignored.
        let scalar = Engine::new(Backend::Scalar).unwrap();
        for reject in 0..=56 {
            let mut bytes = [0x11u8; BUF];
            for (i, c) in bytes[..840].as_chunks_mut::<3>().0.iter_mut().enumerate() {
                let x = if i < reject {
                    0x7F_FFFFu32
                } else {
                    (i as u32 * 7919) % 8_380_417
                };
                c.copy_from_slice(&x.to_le_bytes()[..3]);
            }
            let mut out = [0i32; 256];
            // SAFETY: AVX2 and POPCNT were detected above.
            let (ctr, used) = unsafe { rej_uniform_into(&bytes, 5, &mut out, 0) };
            let mut st = RejState::new();
            let mut want = 5;
            for k in 0..5 {
                let mut b = Block::zero();
                b.bytes[..168].copy_from_slice(&bytes[168 * k..168 * (k + 1)]);
                if scalar.rej_uniform(&b, &mut st) {
                    want = k + 1;
                    break;
                }
            }
            assert_eq!(ctr, st.count, "reject {reject}");
            assert_eq!(used, want, "reject {reject}");
            assert_eq!(out[..ctr], st.coeffs[..ctr], "reject {reject}");
        }
    }
}
