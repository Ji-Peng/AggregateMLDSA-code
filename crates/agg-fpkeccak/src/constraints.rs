//! Pointwise Boolean-domain evaluator for the prime-field Keccak-f[1600] constraints over `i64`.
//!
//! Evaluates all six constraint families (`bool_a`, `bool_c`, `theta_c`, `parity`, `link`, `out`)
//! across all `n_star` records and all live rounds `t in 0..(24 * perms)` using integer arithmetic:
//! - `bool_a`: `A'(A' - 1) = 0` (`1,600` per round)
//! - `bool_c`: `C(C - 1) = 0` (`320` per round)
//! - `theta_c`: `C'[x, z] - xor3(C[x, z], C[x - 1, z], C[x + 1, z - 1]) = 0` (`320` per round)
//! - `parity`: `d(d - 2)(d - 4) = 0` where `d = sum_y A'[x, y, z] - C'[x, z]` (`320` per round)
//! - `link`: `xor3(A'_t, C_t, C'_t) - A_t = 0` (`1,600` per round)
//! - `out`: `out_{24 * perms - 1}[lane, z] - c~_i[lane, z] = 0` for `lane in 0..L` (`64 * L` per record)

use agg_witness::keccak::sponge_trace_for;
use agg_witness::params::{dispatch_set, TR_BYTES};

use crate::layout::{
    lane, pi_inv, rot, Absorbed, KeccakLayout, COLS, LANES, LANE_BITS, RC, ROUNDS_PER_PERM,
};
use crate::trace::KeccakTables;

/// Arithmetic `xor(a, b) = a + b - 2ab` over `i64`.
#[inline]
#[must_use]
pub const fn xor2(a: i64, b: i64) -> i64 {
    a + b - 2 * a * b
}

/// Arithmetic `xor3(a, b, c) = a + b + c - 2ab - 2ac - 2bc + 4abc` over `i64`.
#[inline]
#[must_use]
pub const fn xor3(a: i64, b: i64, c: i64) -> i64 {
    a + b + c - 2 * a * b - 2 * a * c - 2 * b * c + 4 * a * b * c
}

/// Constraint family identifier.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ConstraintFamily {
    /// `A'(A' - 1) = 0`.
    BoolA,
    /// `C(C - 1) = 0`.
    BoolC,
    /// `C'[x, z] - xor3(C[x, z], C[x - 1, z], C[x + 1, z - 1]) = 0`.
    ThetaC,
    /// `d(d - 2)(d - 4) = 0` with `d = sum_y A'[x, y, z] - C'[x, z]`.
    Parity,
    /// `xor3(A'_t, C_t, C'_t) - A_t = 0`.
    Link,
    /// `out_{24 * perms - 1}[lane, z] - c~[lane, z] = 0` for `lane < L`.
    Out,
}

impl ConstraintFamily {
    /// Short constraint family name (`bool_a`, `bool_c`, `theta_c`, `parity`, `link`, `out`).
    #[inline]
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::BoolA => "bool_a",
            Self::BoolC => "bool_c",
            Self::ThetaC => "theta_c",
            Self::Parity => "parity",
            Self::Link => "link",
            Self::Out => "out",
        }
    }
}

impl core::fmt::Display for ConstraintFamily {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(self.name())
    }
}

/// A single constraint violation at `(family, lane, t, z, record)`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Violation {
    /// Which constraint family failed.
    pub family: ConstraintFamily,
    /// Lane index (`0..25` for `BoolA` and `Link`, `0..5` for `BoolC`, `ThetaC`, `Parity`,
    /// `0..L` for `Out`).
    pub lane: usize,
    /// Global round index `t in 0..(24 * perms)`.
    pub t: usize,
    /// Bit index `z in 0..64` within the lane.
    pub z: usize,
    /// Record index `i in 0..n_star`.
    pub record: usize,
}

/// External inputs and outputs bound to one record's Keccak chain: public `mu`, committed
/// `w1Encode(w1')` bits `b_t`, and public commitment hash `c~`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RecordPublic {
    /// 64-byte message representative `mu`.
    pub mu: [u8; TR_BYTES],
    /// `w1Encode(w1')` byte string holding the `b_t(j, rho)` bits.
    pub w1_bytes: Vec<u8>,
    /// Public commitment hash `c~` (`32`, `48`, or `64` bytes).
    pub c_tilde: Vec<u8>,
}

/// Batch external inputs (`mu`, `w1_bytes`, `c~`) for all `n_star` records, including padded
/// records (`i >= N`) which carry the all-zero message and its `SHAKE256` output.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct KeccakPublics {
    layout: KeccakLayout,
    records: Vec<RecordPublic>,
}

impl KeccakPublics {
    /// Builds the batch inputs and outputs for `records` padded to `n_star` with the all-zero
    /// message (`mu = 0`, `w1_bytes = 0`, `c~ = SHAKE256(0)`).
    #[must_use]
    pub fn from_records<M, W>(layout: &KeccakLayout, records: &[(M, W)], n_star: usize) -> Self
    where
        M: AsRef<[u8]>,
        W: AsRef<[u8]>,
    {
        assert!(n_star >= records.len());
        let mut out = Vec::with_capacity(n_star);
        for (mu_ref, w1_ref) in records {
            let mu: [u8; TR_BYTES] = mu_ref
                .as_ref()
                .try_into()
                .expect("mu slice must be 64 bytes");
            let w1_bytes = w1_ref.as_ref().to_vec();
            assert_eq!(w1_bytes.len(), layout.w1_encode_bytes);
            let c_tilde = dispatch_set!(layout.set, |P| {
                sponge_trace_for::<P>(&mu, &w1_bytes).c_tilde.as_ref().to_vec()
            });
            out.push(RecordPublic {
                mu,
                w1_bytes,
                c_tilde,
            });
        }
        if n_star > records.len() {
            let zero_mu = [0u8; TR_BYTES];
            let zero_w1 = vec![0u8; layout.w1_encode_bytes];
            let zero_c_tilde = dispatch_set!(layout.set, |P| {
                sponge_trace_for::<P>(&zero_mu, &zero_w1)
                    .c_tilde
                    .as_ref()
                    .to_vec()
            });
            for _ in records.len()..n_star {
                out.push(RecordPublic {
                    mu: zero_mu,
                    w1_bytes: zero_w1.clone(),
                    c_tilde: zero_c_tilde.clone(),
                });
            }
        }
        Self {
            layout: layout.clone(),
            records: out,
        }
    }

    /// Slice of per-record inputs/outputs (`0..n_star`).
    #[inline]
    #[must_use]
    pub fn records(&self) -> &[RecordPublic] {
        &self.records
    }

    /// Mutable slice of per-record inputs/outputs (`0..n_star`).
    #[inline]
    pub fn records_mut(&mut self) -> &mut [RecordPublic] {
        &mut self.records
    }

    /// Evaluates the absorbed bit at `(m, x, y, z)` for `record` in `i64`.
    #[inline]
    #[must_use]
    pub fn absorbed_bit(&self, record: usize, m: usize, x: usize, y: usize, z: usize) -> i64 {
        let rec = &self.records[record];
        let src: Absorbed = self.layout.wire(m, x, y, z);
        i64::from(src.eval(self.layout.w1_bits, &rec.mu, &rec.w1_bytes))
    }

    /// Reads bit `z` of output lane `lane_idx` (`0..L`) of `c~` for `record` in `i64`.
    #[inline]
    #[must_use]
    pub fn c_tilde_bit(&self, record: usize, lane_idx: usize, z: usize) -> i64 {
        assert!(lane_idx < self.layout.output_lanes && z < LANE_BITS);
        let byte_idx = lane_idx * 8 + z / 8;
        let bit_idx = z % 8;
        i64::from((self.records[record].c_tilde[byte_idx] >> bit_idx) & 1)
    }

    /// Flips `w1Encode` bit `b_t(j, rho)` of `record`.
    pub fn flip_w1_bit(&mut self, record: usize, t: usize, j: usize, rho: usize) {
        assert!(
            record < self.records.len()
                && t < self.layout.w1_bits
                && j < 256
                && rho < self.layout.k
        );
        let bit_idx = rho * 256 * self.layout.w1_bits + j * self.layout.w1_bits + t;
        self.records[record].w1_bytes[bit_idx / 8] ^= 1u8 << (bit_idx % 8);
    }

    /// Flips `mu` bit `bit_index` (`0..512`) of `record`.
    pub fn flip_mu_bit(&mut self, record: usize, bit_index: usize) {
        assert!(record < self.records.len() && bit_index < TR_BYTES * 8);
        self.records[record].mu[bit_index / 8] ^= 1u8 << (bit_index % 8);
    }

    /// Flips `c~` output bit `(lane_idx, z)` (`lane_idx in 0..L`, `z in 0..64`) of `record`.
    pub fn flip_c_tilde_bit(&mut self, record: usize, lane_idx: usize, z: usize) {
        assert!(
            record < self.records.len() && lane_idx < self.layout.output_lanes && z < LANE_BITS
        );
        let byte_idx = lane_idx * 8 + z / 8;
        let bit_idx = z % 8;
        self.records[record].c_tilde[byte_idx] ^= 1u8 << bit_idx;
    }
}

/// Evaluates all six Keccak constraint families over the Boolean domain in `i64` arithmetic,
/// returning all violations in deterministic `(record, t, family, lane, z)` order.
#[must_use]
pub fn evaluate(tables: &KeccakTables, publics: &KeccakPublics) -> Vec<Violation> {
    let layout = tables.layout();
    assert_eq!(layout.set, publics.layout.set);
    assert_eq!(tables.n_star(), publics.records.len());

    let mut violations = Vec::new();
    let rounds = layout.rounds;

    for record in 0..tables.n_star() {
        let mut prev_out = [[0i64; LANE_BITS]; LANES];

        for t in 0..rounds {
            let (q, s) = layout.slot_of(t);

            // Load A'_t[lane, z], C_t[x, z], C'_t[x, z] as i64.
            let mut a_prime = [[0i64; LANE_BITS]; LANES];
            let mut c = [[0i64; LANE_BITS]; COLS];
            let mut c_prime = [[0i64; LANE_BITS]; COLS];

            for y in 0..COLS {
                for x in 0..COLS {
                    let l = lane(x, y);
                    let poly = layout.a_prime_poly(x, y, q);
                    for z in 0..LANE_BITS {
                        a_prime[l][z] = tables.cell(poly, s, record, z);
                    }
                }
            }
            for x in 0..COLS {
                let c_poly = layout.c_poly(x, q);
                let cp_poly = layout.c_prime_poly(x, q);
                for z in 0..LANE_BITS {
                    c[x][z] = tables.cell(c_poly, s, record, z);
                    c_prime[x][z] = tables.cell(cp_poly, s, record, z);
                }
            }

            // 1. bool_a: A'(A' - 1) = 0.
            for l in 0..LANES {
                for z in 0..LANE_BITS {
                    let a = a_prime[l][z];
                    if a * (a - 1) != 0 {
                        violations.push(Violation {
                            family: ConstraintFamily::BoolA,
                            lane: l,
                            t,
                            z,
                            record,
                        });
                    }
                }
            }

            // 2. bool_c: C(C - 1) = 0.
            for x in 0..COLS {
                for z in 0..LANE_BITS {
                    let cx = c[x][z];
                    if cx * (cx - 1) != 0 {
                        violations.push(Violation {
                            family: ConstraintFamily::BoolC,
                            lane: x,
                            t,
                            z,
                            record,
                        });
                    }
                }
            }

            // 3. theta_c: C'[x, z] - xor3(C[x, z], C[x - 1, z], C[x + 1, z - 1]) = 0.
            for x in 0..COLS {
                let xm = (x + COLS - 1) % COLS;
                let xp = (x + 1) % COLS;
                for z in 0..LANE_BITS {
                    let zm = (z + LANE_BITS - 1) % LANE_BITS;
                    let expected = xor3(c[x][z], c[xm][z], c[xp][zm]);
                    if c_prime[x][z] - expected != 0 {
                        violations.push(Violation {
                            family: ConstraintFamily::ThetaC,
                            lane: x,
                            t,
                            z,
                            record,
                        });
                    }
                }
            }

            // 4. parity: d(d - 2)(d - 4) = 0 where d = sum_y A'[x, y, z] - C'[x, z].
            for x in 0..COLS {
                for z in 0..LANE_BITS {
                    let sum_y = a_prime[lane(x, 0)][z]
                        + a_prime[lane(x, 1)][z]
                        + a_prime[lane(x, 2)][z]
                        + a_prime[lane(x, 3)][z]
                        + a_prime[lane(x, 4)][z];
                    let d = sum_y - c_prime[x][z];
                    if d * (d - 2) * (d - 4) != 0 {
                        violations.push(Violation {
                            family: ConstraintFamily::Parity,
                            lane: x,
                            t,
                            z,
                            record,
                        });
                    }
                }
            }

            // 5. link: xor3(A'_t, C_t, C'_t) - A_t = 0.
            for y in 0..COLS {
                for x in 0..COLS {
                    let l = lane(x, y);
                    for z in 0..LANE_BITS {
                        let a_in = if t == 0 {
                            // At t = 0: xor(mu, b) on the first 1,088 bits, 0 on capacity bits.
                            let (mu_bit, b_bit) = match layout.wire(0, x, y, z) {
                                Absorbed::Mu(_) => (publics.absorbed_bit(record, 0, x, y, z), 0),
                                Absorbed::W1Bit { .. } => {
                                    (0, publics.absorbed_bit(record, 0, x, y, z))
                                }
                                Absorbed::Pad(bit) => (0, i64::from(bit)),
                                Absorbed::Zero => (0, 0),
                            };
                            xor2(mu_bit, b_bit)
                        } else if t.is_multiple_of(ROUNDS_PER_PERM) {
                            // Boundary t = 24 * m (1 <= m < perms): xor(out_{t-1}, blk_m).
                            let m = t / ROUNDS_PER_PERM;
                            let blk = publics.absorbed_bit(record, m, x, y, z);
                            xor2(prev_out[l][z], blk)
                        } else {
                            // Inner round r >= 1: out_{t-1}.
                            prev_out[l][z]
                        };
                        let lhs = xor3(a_prime[l][z], c[x][z], c_prime[x][z]);
                        if lhs - a_in != 0 {
                            violations.push(Violation {
                                family: ConstraintFamily::Link,
                                lane: l,
                                t,
                                z,
                                record,
                            });
                        }
                    }
                }
            }

            // Compute round output out_t = iota_r(chi(B_t)) in i64 arithmetic.
            let mut b_state = [[0i64; LANE_BITS]; LANES];
            for y in 0..COLS {
                for x in 0..COLS {
                    let (src_x, src_y) = pi_inv(x, y);
                    let src_lane = lane(src_x, src_y);
                    let r_off = rot(src_x, src_y) as usize;
                    let dst_lane = lane(x, y);
                    for z in 0..LANE_BITS {
                        let src_z = (z + LANE_BITS - r_off) % LANE_BITS;
                        b_state[dst_lane][z] = a_prime[src_lane][src_z];
                    }
                }
            }

            let rc = RC[t % ROUNDS_PER_PERM];
            let mut cur_out = [[0i64; LANE_BITS]; LANES];
            for y in 0..COLS {
                for x in 0..COLS {
                    let l0 = lane(x, y);
                    let l1 = lane((x + 1) % COLS, y);
                    let l2 = lane((x + 2) % COLS, y);
                    for z in 0..LANE_BITS {
                        let chi = xor2(b_state[l0][z], (1 - b_state[l1][z]) * b_state[l2][z]);
                        let out_val = if x == 0 && y == 0 && ((rc >> z) & 1 == 1) {
                            1 - chi
                        } else {
                            chi
                        };
                        cur_out[l0][z] = out_val;
                    }
                }
            }

            // 6. out: on the final round t = 24 * perms - 1, first L lanes match c~.
            if t + 1 == rounds {
                for l in 0..layout.output_lanes {
                    for z in 0..LANE_BITS {
                        let expected = publics.c_tilde_bit(record, l, z);
                        if cur_out[l][z] - expected != 0 {
                            violations.push(Violation {
                                family: ConstraintFamily::Out,
                                lane: l,
                                t,
                                z,
                                record,
                            });
                        }
                    }
                }
            }

            prev_out = cur_out;
        }
    }

    violations
}

/// Alias for [`evaluate`].
#[inline]
#[must_use]
pub fn evaluate_all(tables: &KeccakTables, publics: &KeccakPublics) -> Vec<Violation> {
    evaluate(tables, publics)
}
