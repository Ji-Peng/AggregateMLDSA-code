//! Bit-packed Keccak-f[1600] witness trace per record and batch table builder.
//!
//! For each live round `t in 0..(24 * perms)`, [`RecordTrace`] records the 25 post-theta
//! state words `a_prime[25]`, the 5 pre-theta column parity words `c[5]`, and the 5
//! post-theta column parity words `c_prime[5]`. [`KeccakTables`] packs these words across
//! a padded batch of `n_star` records in polynomial order (`G_KA[0..5]`, `G_KC`, `G_KC'`),
//! filling padded records (`i >= N`) with the honest trace of the all-zero message and dead
//! time slots (`t >= 24 * perms`) with zeros.

use std::collections::BTreeMap;
use std::ops::Range;

use agg_witness::keccak::{absorb_block, sponge_blocks_for, state_bytes, State};
use agg_witness::params::{dispatch_set, TR_BYTES};
use rayon::prelude::*;

use crate::layout::{
    lane, KeccakLayout, PolyFamily, COLS, LANES, LANE_BITS, RC, RECORD_CELLS_PER_POLY, ROT,
    ROUNDS_PER_PERM, SLOTS,
};

/// Rotates a packed 64-bit lane word left by `rot` bits (`z -> (z + rot) mod 64`, so bit `z`
/// of the output is bit `(z - rot) mod 64` of `word`).
#[inline]
#[must_use]
pub const fn rotated(word: u64, rot: u32) -> u64 {
    word.rotate_left(rot)
}

/// Per-record Keccak-f[1600] round witness words across all `24 * perms` rounds.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RecordTrace {
    /// Post-theta state words `A'_t[lane]` (`lane = x + 5 * y`, bit `z` at `(word >> z) & 1`).
    pub a_prime: Vec<[u64; LANES]>,
    /// Pre-theta column parity words `C_t[x] = xor_y A_t[x + 5 * y]`.
    pub c: Vec<[u64; COLS]>,
    /// Post-theta column parity words `C'_t[x] = xor_y A'_t[x + 5 * y]`.
    pub c_prime: Vec<[u64; COLS]>,
    /// Output commitment hash `c~` extracted from the first `c_tilde_bytes` of the final state.
    pub c_tilde: Vec<u8>,
}

impl RecordTrace {
    /// Computes the witness trace of one record from `mu` and `w1_bytes` using the reference
    /// Keccak-f[1600] permutation.
    #[must_use]
    pub fn new(layout: &KeccakLayout, mu: &[u8; TR_BYTES], w1_bytes: &[u8]) -> Self {
        assert_eq!(w1_bytes.len(), layout.w1_encode_bytes);
        let blocks = dispatch_set!(layout.set, |P| sponge_blocks_for::<P>(mu, w1_bytes));
        assert_eq!(blocks.len(), layout.perms);

        let mut a_prime = Vec::with_capacity(layout.rounds);
        let mut c = Vec::with_capacity(layout.rounds);
        let mut c_prime = Vec::with_capacity(layout.rounds);
        let mut state: State = [0u64; LANES];

        for m in 0..layout.perms {
            absorb_block(&mut state, &blocks[m]);
            for r in 0..ROUNDS_PER_PERM {
                // Pre-theta column parities C[x].
                let mut c_round = [0u64; COLS];
                for x in 0..COLS {
                    c_round[x] = state[x]
                        ^ state[x + 5]
                        ^ state[x + 10]
                        ^ state[x + 15]
                        ^ state[x + 20];
                }
                // Theta effect D[x] and post-theta state A'[x, y].
                for x in 0..COLS {
                    let d = c_round[(x + 4) % COLS] ^ rotated(c_round[(x + 1) % COLS], 1);
                    for y in 0..COLS {
                        state[lane(x, y)] ^= d;
                    }
                }
                // Post-theta column parities C'[x].
                let mut cp_round = [0u64; COLS];
                for x in 0..COLS {
                    cp_round[x] = state[x]
                        ^ state[x + 5]
                        ^ state[x + 10]
                        ^ state[x + 15]
                        ^ state[x + 20];
                }

                a_prime.push(state);
                c.push(c_round);
                c_prime.push(cp_round);

                // Rho and pi: B[y, (2x + 3y) mod 5] = rot(A'[x, y], r(x, y)).
                let mut b = [0u64; LANES];
                for x in 0..COLS {
                    for y in 0..COLS {
                        let idx = lane(x, y);
                        b[y + COLS * ((2 * x + 3 * y) % COLS)] = rotated(state[idx], ROT[idx]);
                    }
                }
                // Chi.
                for y in 0..COLS {
                    for x in 0..COLS {
                        state[lane(x, y)] = b[lane(x, y)]
                            ^ (!b[lane((x + 1) % COLS, y)] & b[lane((x + 2) % COLS, y)]);
                    }
                }
                // Iota.
                state[0] ^= RC[r];
            }
        }

        let sb = state_bytes(&state);
        let c_tilde = sb[..layout.c_tilde_bytes].to_vec();
        Self {
            a_prime,
            c,
            c_prime,
            c_tilde,
        }
    }
}

/// Packed Keccak witness tables across `n_star` records.
///
/// Words are stored polynomial-major (`poly in 0..layout.num_polys()`), then by record
/// `i in 0..n_star`, then by slot `s in 0..16`: index `(poly * n_star + i) * 16 + s`.
/// Within one polynomial `poly`, word `s + 16 * i` holds the 64 bits `z in 0..64`, matching
/// cell index `z + 64 * s + 1024 * i`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct KeccakTables {
    layout: KeccakLayout,
    n_records: usize,
    n_star: usize,
    words: Vec<u64>,
    c_tildes: Vec<Vec<u8>>,
    /// Sparse non-Boolean cell overrides `(poly, s, i, z) -> value` used by negative tests.
    overrides: BTreeMap<(usize, usize, usize, usize), i64>,
}

impl KeccakTables {
    /// Builds the packed Keccak witness tables in parallel over `records`, padding records
    /// `i in records.len()..n_star` with the trace of the all-zero message (`mu = 0`, `w1 = 0`).
    #[must_use]
    pub fn build<M, W>(layout: &KeccakLayout, records: &[(M, W)], n_star: usize) -> Self
    where
        M: AsRef<[u8]> + Sync,
        W: AsRef<[u8]> + Sync,
    {
        let n_records = records.len();
        assert!(
            n_star >= n_records,
            "n_star ({n_star}) must be at least records.len() ({n_records})"
        );

        let live_traces: Vec<RecordTrace> = records
            .par_iter()
            .map(|(mu, w1)| {
                let mu_arr: &[u8; TR_BYTES] = mu
                    .as_ref()
                    .try_into()
                    .expect("mu slice must be 64 bytes");
                RecordTrace::new(layout, mu_arr, w1.as_ref())
            })
            .collect();

        let pad_trace = if n_star > n_records {
            let zero_mu = [0u8; TR_BYTES];
            let zero_w1 = vec![0u8; layout.w1_encode_bytes];
            Some(RecordTrace::new(layout, &zero_mu, &zero_w1))
        } else {
            None
        };

        let mut c_tildes = Vec::with_capacity(n_star);
        for trace in &live_traces {
            c_tildes.push(trace.c_tilde.clone());
        }
        if let Some(ref pad) = pad_trace {
            for _ in n_records..n_star {
                c_tildes.push(pad.c_tilde.clone());
            }
        }

        let words_per_poly = SLOTS * n_star;
        let num_polys = layout.num_polys();
        let mut words = vec![0u64; num_polys * words_per_poly];

        if n_star > 0 {
            words
                .par_chunks_mut(words_per_poly)
                .enumerate()
                .for_each(|(poly, poly_slice)| {
                    let (family, q) = layout.poly_coords(poly);
                    for i in 0..n_star {
                        let trace = if i < n_records {
                            &live_traces[i]
                        } else {
                            pad_trace.as_ref().expect("pad trace present")
                        };
                        let base = i * SLOTS;
                        for s in 0..SLOTS {
                            let t = layout.time(q, s);
                            poly_slice[base + s] = if t < layout.rounds {
                                match family {
                                    PolyFamily::APrime { x, y } => trace.a_prime[t][lane(x, y)],
                                    PolyFamily::C { x } => trace.c[t][x],
                                    PolyFamily::CPrime { x } => trace.c_prime[t][x],
                                }
                            } else {
                                0
                            };
                        }
                    }
                });
        }

        Self {
            layout: layout.clone(),
            n_records,
            n_star,
            words,
            c_tildes,
            overrides: BTreeMap::new(),
        }
    }

    /// The parameter-set Keccak layout.
    #[inline]
    #[must_use]
    pub const fn layout(&self) -> &KeccakLayout {
        &self.layout
    }

    /// Number of live records `N`.
    #[inline]
    #[must_use]
    pub const fn n_records(&self) -> usize {
        self.n_records
    }

    /// Padded batch size `N*`.
    #[inline]
    #[must_use]
    pub const fn n_star(&self) -> usize {
        self.n_star
    }

    /// All packed words in polynomial-major order.
    #[inline]
    #[must_use]
    pub fn words(&self) -> &[u64] {
        &self.words
    }

    /// Packed words of polynomial `poly` (`16 * n_star` words, word `s + 16 * i` at index `s + 16 * i`).
    #[inline]
    #[must_use]
    pub fn poly_words(&self, poly: usize) -> &[u64] {
        assert!(poly < self.layout.num_polys());
        let stride = SLOTS * self.n_star;
        &self.words[poly * stride..(poly + 1) * stride]
    }

    /// Packed word of `(poly, s, i)`.
    #[inline]
    #[must_use]
    pub fn word(&self, poly: usize, s: usize, i: usize) -> u64 {
        assert!(poly < self.layout.num_polys() && s < SLOTS && i < self.n_star);
        self.words[(poly * self.n_star + i) * SLOTS + s]
    }

    /// Mutable reference to the packed word of `(poly, s, i)`.
    #[inline]
    pub fn word_mut(&mut self, poly: usize, s: usize, i: usize) -> &mut u64 {
        assert!(poly < self.layout.num_polys() && s < SLOTS && i < self.n_star);
        &mut self.words[(poly * self.n_star + i) * SLOTS + s]
    }

    /// Packed word of `A'[lane_idx]` at round `t` and record `i`.
    #[inline]
    #[must_use]
    pub fn a_prime_word(&self, lane_idx: usize, t: usize, i: usize) -> u64 {
        let (x, y) = crate::layout::lane_xy(lane_idx);
        let (q, s) = self.layout.slot_of(t);
        self.word(self.layout.a_prime_poly(x, y, q), s, i)
    }

    /// Packed word of `C[x]` at round `t` and record `i`.
    #[inline]
    #[must_use]
    pub fn c_word(&self, x: usize, t: usize, i: usize) -> u64 {
        let (q, s) = self.layout.slot_of(t);
        self.word(self.layout.c_poly(x, q), s, i)
    }

    /// Packed word of `C'[x]` at round `t` and record `i`.
    #[inline]
    #[must_use]
    pub fn c_prime_word(&self, x: usize, t: usize, i: usize) -> u64 {
        let (q, s) = self.layout.slot_of(t);
        self.word(self.layout.c_prime_poly(x, q), s, i)
    }

    /// Reads the `i64` cell value at `(poly, s, i, z)`, applying any cell override.
    #[inline]
    #[must_use]
    pub fn cell(&self, poly: usize, s: usize, i: usize, z: usize) -> i64 {
        debug_assert!(z < LANE_BITS);
        if !self.overrides.is_empty() {
            if let Some(&val) = self.overrides.get(&(poly, s, i, z)) {
                return val;
            }
        }
        ((self.word(poly, s, i) >> z) & 1) as i64
    }

    /// Overrides the cell at `(poly, s, i, z)` with an arbitrary `i64` value (for testing
    /// non-Boolean violations).
    pub fn set_cell(&mut self, poly: usize, s: usize, i: usize, z: usize, value: i64) {
        assert!(
            poly < self.layout.num_polys() && s < SLOTS && i < self.n_star && z < LANE_BITS
        );
        if value == 0 || value == 1 {
            self.overrides.remove(&(poly, s, i, z));
            let w = self.word_mut(poly, s, i);
            if value == 1 {
                *w |= 1u64 << z;
            } else {
                *w &= !(1u64 << z);
            }
        } else {
            self.overrides.insert((poly, s, i, z), value);
        }
    }

    /// True if any non-Boolean cell overrides are active.
    #[inline]
    #[must_use]
    pub fn has_overrides(&self) -> bool {
        !self.overrides.is_empty()
    }

    /// Flips bit `z` of `A'[lane_idx]` at round `t` and record `record`.
    pub fn flip_a_prime_bit(&mut self, lane_idx: usize, t: usize, z: usize, record: usize) {
        let (x, y) = crate::layout::lane_xy(lane_idx);
        let (q, s) = self.layout.slot_of(t);
        let poly = self.layout.a_prime_poly(x, y, q);
        *self.word_mut(poly, s, record) ^= 1u64 << z;
    }

    /// Flips bit `z` of `C[x]` at round `t` and record `record`.
    pub fn flip_c_bit(&mut self, x: usize, t: usize, z: usize, record: usize) {
        let (q, s) = self.layout.slot_of(t);
        let poly = self.layout.c_poly(x, q);
        *self.word_mut(poly, s, record) ^= 1u64 << z;
    }

    /// Flips bit `z` of `C'[x]` at round `t` and record `record`.
    pub fn flip_c_prime_bit(&mut self, x: usize, t: usize, z: usize, record: usize) {
        let (q, s) = self.layout.slot_of(t);
        let poly = self.layout.c_prime_poly(x, q);
        *self.word_mut(poly, s, record) ^= 1u64 << z;
    }

    /// Sets `A'[lane_idx]` at `(t, z, record)` to an arbitrary `i64` value.
    pub fn set_a_prime_cell(
        &mut self,
        lane_idx: usize,
        t: usize,
        z: usize,
        record: usize,
        value: i64,
    ) {
        let (x, y) = crate::layout::lane_xy(lane_idx);
        let (q, s) = self.layout.slot_of(t);
        let poly = self.layout.a_prime_poly(x, y, q);
        self.set_cell(poly, s, record, z, value);
    }

    /// Sets `C[x]` at `(t, z, record)` to an arbitrary `i64` value.
    pub fn set_c_cell(&mut self, x: usize, t: usize, z: usize, record: usize, value: i64) {
        let (q, s) = self.layout.slot_of(t);
        let poly = self.layout.c_poly(x, q);
        self.set_cell(poly, s, record, z, value);
    }

    /// Sets `C'[x]` at `(t, z, record)` to an arbitrary `i64` value.
    pub fn set_c_prime_cell(&mut self, x: usize, t: usize, z: usize, record: usize, value: i64) {
        let (q, s) = self.layout.slot_of(t);
        let poly = self.layout.c_prime_poly(x, q);
        self.set_cell(poly, s, record, z, value);
    }

    /// Derived commitment hash `c~` of `record` (`0..n_star`).
    #[inline]
    #[must_use]
    pub fn c_tilde(&self, record: usize) -> &[u8] {
        &self.c_tildes[record]
    }

    /// Derived commitment hashes `c~` for all `0..n_star` records.
    #[inline]
    #[must_use]
    pub fn c_tildes(&self) -> &[Vec<u8>] {
        &self.c_tildes
    }

    /// Expands the packed `u64` words of `poly_range` into `i8` cells in polynomial-major order,
    /// where within each polynomial cell `z + 64 * s + 1024 * i` holds bit `z` of slot `s` and
    /// record `i`.
    #[must_use]
    pub fn expand_i8(&self, poly_range: Range<usize>) -> Vec<i8> {
        assert!(poly_range.start <= poly_range.end && poly_range.end <= self.layout.num_polys());
        let cells_per_poly = RECORD_CELLS_PER_POLY * self.n_star;
        // A zeroed allocation takes fresh zero pages; the parallel writers below fault them in.
        let mut out = vec![0i8; poly_range.len() * cells_per_poly];
        if cells_per_poly == 0 {
            return out;
        }
        out.par_chunks_mut(cells_per_poly)
            .zip(poly_range)
            .for_each(|(poly_out, poly)| self.expand_poly_i8_into(poly, poly_out));
        out
    }

    /// Writes the `i8` cells of polynomial `poly` (cell `z + 64 * s + 1024 * i` is bit `z` of
    /// slot `s` and record `i`, cell overrides applied) to `out[..16 * 64 * n_star]` and zeroes
    /// the rest of `out`, so a polynomial can be expanded straight into a buffer of its
    /// zero-extended length without a group-sized intermediate.
    ///
    /// # Panics
    /// If `poly` is out of range or `out` is shorter than the polynomial's cells.
    pub fn expand_poly_i8_into(&self, poly: usize, out: &mut [i8]) {
        let cells_per_poly = RECORD_CELLS_PER_POLY * self.n_star;
        assert!(out.len() >= cells_per_poly, "output shorter than the polynomial");
        let (cells, tail) = out.split_at_mut(cells_per_poly);
        for (lane, &word) in cells.chunks_exact_mut(LANE_BITS).zip(self.poly_words(poly)) {
            for (z, cell) in lane.iter_mut().enumerate() {
                *cell = ((word >> z) & 1) as i8;
            }
        }
        tail.fill(0);
        let first = (poly, 0, 0, 0);
        for (&(_, s, i, z), &val) in self.overrides.range(first..(poly + 1, 0, 0, 0)) {
            cells[z + LANE_BITS * s + RECORD_CELLS_PER_POLY * i] = val as i8;
        }
    }
}
