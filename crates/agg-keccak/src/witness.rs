//! Batch-major packing of gate instances into one Flock union slot.
//!
//! Flock's union stores a slot's witness BatchMajor: 128-bit word `c` of
//! instance `j` is union word `(c << nu) + j`. The lincheck additionally needs
//! the byte stripe of `pack_z_lincheck`: for every group of 8 instances, one
//! byte per block column whose bit `r` is that column of instance `8g + r`.
//! Rows `[n, 2^nu)` are dummy and identically zero in `z`, `a`, `b` and the stripe.

use flock_core::alloc_zeroed_vec;
use flock_field::F128;
use rayon::prelude::*;

use crate::gate::ChainGate;

/// Instances whose rows are generated together before scattering.
const BATCH: usize = 1024;

/// One slot's prover buffers: `z`, `a = A z`, `b = B z` (BatchMajor) and the lincheck stripe.
pub struct SlotBuffers {
    pub z: Vec<F128>,
    pub a: Vec<F128>,
    pub b: Vec<F128>,
    pub stripe: Vec<u8>,
}

impl SlotBuffers {
    /// The tuple `UnionSlotProverInput::new` takes.
    #[must_use]
    pub fn into_parts(self) -> (Vec<F128>, Vec<F128>, Vec<F128>, Vec<u8>) {
        (self.z, self.a, self.b, self.stripe)
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
    assert!(nu >= 3, "Flock's lincheck stripe needs at least 8 rows");
    assert!(n <= 1 << nu, "count exceeds the row capacity");
    assert!(k_log >= 7 && useful_bits <= 1 << k_log, "block geometry");
    let k = 1usize << k_log;
    let w = useful_bits.div_ceil(64);
    let useful_chunks = useful_bits.div_ceil(128);
    let len = (k / 128) << nu;
    let mut z: Vec<F128> = alloc_zeroed_vec(len);
    let mut a: Vec<F128> = alloc_zeroed_vec(len);
    let mut b: Vec<F128> = alloc_zeroed_vec(len);
    let mut stripe: Vec<u8> = alloc_zeroed_vec(k << (nu - 3));
    let mut scratch = vec![0u64; BATCH * 3 * w];
    // AGG_TIED_TRACE: time per phase (instance rows; BatchMajor scatter and
    // the lincheck stripe) summed over the batches.
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
                            col[j0 + i] =
                                F128::new(word(i, part, 2 * c), word(i, part, 2 * c + 1));
                        }
                    }
                }
            });
        // Stripe: groups of 8 instances (j0 and BATCH are multiples of 8),
        // parallel over groups and over runs of 8 words, so each task reads
        // one line of each of its 8 instance rows.
        let g0 = j0 / 8;
        stripe
            .par_chunks_mut(k)
            .skip(g0)
            .take(cnt.div_ceil(8))
            .enumerate()
            .for_each(|(g, out)| {
                let lanes = 8 * g..(8 * g + 8).min(cnt);
                out[..64 * w]
                    .par_chunks_mut(64 * 8)
                    .enumerate()
                    .for_each(|(ub, run)| {
                        for (du, o) in run.chunks_exact_mut(64).enumerate() {
                            let u = 8 * ub + du;
                            let mut v = [0u64; 8];
                            for (r, i) in lanes.clone().enumerate() {
                                v[r] = word(i, 0, u);
                            }
                            if v != [0u64; 8] {
                                transpose_8x64(&v, o);
                            }
                        }
                    });
            });
        phase_ms[1] += t0.elapsed().as_secs_f64() * 1e3;
        j0 += cnt;
    }
    if trace {
        eprintln!(
            "  [tied] pack_slot: rows {:.1} ms, scatter and stripe {:.1} ms",
            phase_ms[0], phase_ms[1]
        );
    }
    SlotBuffers { z, a, b, stripe }
}

/// `out[bit] = sum_r bit(v[r], bit) << r` for `bit < 64`: eight 8x8 bit-matrix
/// transposes (byte `k` of every `v[r]` forms one matrix), three delta swaps each,
/// instead of 512 single-bit moves.
#[inline]
fn transpose_8x64(v: &[u64; 8], out: &mut [u8]) {
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
            bufs.stripe,
            pack_z_lincheck(&logical, nu + lay.k_log, lay.k_log)
        );
        for j in n..1 << nu {
            for c in 0..k / 128 {
                assert_eq!(bufs.z[(c << nu) + j], F128::ZERO);
                assert_eq!(bufs.a[(c << nu) + j], F128::ZERO);
            }
        }
    }
}
