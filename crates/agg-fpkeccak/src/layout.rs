//! Geometry, permutation constants, slot masks, and sponge input wiring for the prime-field
//! Keccak-f[1600] commitment-hash witness.
//!
//! Each record runs a chain of `perms` Keccak-f[1600] permutations (`7` for ML-DSA-44 and
//! ML-DSA-65, `9` for ML-DSA-87), 24 rounds per permutation, packed into polynomials with
//! `slots = 16` time slots per polynomial and `slot_groups = ceil(24 * perms / 16)` polynomials
//! per lane (`11` or `14`). Lanes are indexed `x + 5 * y` as in `agg_witness::keccak::State`.

use agg_witness::keccak::{mu_bit_slot, w1_bit_slot_for, RATE, RATE_BITS};
use agg_witness::params::{dispatch_set, ParamSet, Set, N, TR_BYTES};

/// Lanes in a Keccak-f[1600] state (`5 * 5`).
pub const LANES: usize = 25;
/// Columns in a Keccak-f[1600] state (`x in 0..5`).
pub const COLS: usize = 5;
/// Bits in one Keccak-f[1600] lane (`z in 0..64`).
pub const LANE_BITS: usize = 64;
/// `log2` of [`LANE_BITS`].
pub const LANE_BITS_LOG2: u32 = 6;
/// Bits in a full Keccak-f[1600] state (`25 * 64`).
pub const STATE_BITS: usize = LANES * LANE_BITS;
/// Rounds per Keccak-f[1600] permutation.
pub const ROUNDS_PER_PERM: usize = 24;
/// Time slots packed into one polynomial (`s in 0..16`).
pub const SLOTS: usize = 16;
/// `log2` of [`SLOTS`].
pub const SLOTS_LOG2: u32 = 4;
/// Cells per record in one Keccak polynomial (`64 * 16 = 1024`).
pub const RECORD_CELLS_PER_POLY: usize = LANE_BITS * SLOTS;
/// Committed lane families per round (`25` post-theta `A'` lanes, `5` `C` lanes, `5` `C'` lanes).
pub const TOTAL_LANE_FAMILIES: usize = LANES + COLS + COLS;
/// Number of Keccak commitment groups (`G_KA[0..5]`, `G_KC`, `G_KC'`).
pub const KECCAK_GROUPS: usize = 7;

/// Keccak-f[1600] iota round constants (FIPS 202 Section 3.2.5).
pub const RC: [u64; ROUNDS_PER_PERM] = [
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

/// Keccak-f[1600] rho rotation offsets indexed by `x + 5 * y` (FIPS 202 Table 2).
pub const ROT: [u32; LANES] = [
    0, 1, 62, 28, 27, 36, 44, 6, 55, 20, 3, 10, 43, 25, 39, 41, 45, 15, 21, 8, 18, 2, 61, 56, 14,
];

/// Linear lane index `x + 5 * y` for `x, y in 0..5`.
#[inline]
#[must_use]
pub const fn lane(x: usize, y: usize) -> usize {
    debug_assert!(x < COLS && y < COLS);
    x + COLS * y
}

/// Coordinates `(x, y)` of linear `lane` in `0..25`.
#[inline]
#[must_use]
pub const fn lane_xy(lane_idx: usize) -> (usize, usize) {
    debug_assert!(lane_idx < LANES);
    (lane_idx % COLS, lane_idx / COLS)
}

/// State bit index `64 * (x + 5 * y) + z` in `0..1600`.
#[inline]
#[must_use]
pub const fn state_bit(x: usize, y: usize, z: usize) -> usize {
    debug_assert!(z < LANE_BITS);
    LANE_BITS * lane(x, y) + z
}

/// Coordinates `(x, y, z)` of a state bit in `0..1600`.
#[inline]
#[must_use]
pub const fn state_bit_xyz(bit_idx: usize) -> (usize, usize, usize) {
    debug_assert!(bit_idx < STATE_BITS);
    let lane_idx = bit_idx / LANE_BITS;
    let z = bit_idx % LANE_BITS;
    let (x, y) = lane_xy(lane_idx);
    (x, y, z)
}

/// Rotation offset `r(x, y)` (FIPS 202 Table 2).
#[inline]
#[must_use]
pub const fn rot(x: usize, y: usize) -> u32 {
    ROT[lane(x, y)]
}

/// Forward `pi` lane permutation: `pi(x, y) = (y, (2x + 3y) mod 5)` (FIPS 202 Section 3.2.3).
#[inline]
#[must_use]
pub const fn pi(x: usize, y: usize) -> (usize, usize) {
    debug_assert!(x < COLS && y < COLS);
    (y, (2 * x + 3 * y) % COLS)
}

/// Inverse `pi^-1` lane permutation: `pi^-1(x, y) = ((x + 3y) mod 5, x)`.
#[inline]
#[must_use]
pub const fn pi_inv(x: usize, y: usize) -> (usize, usize) {
    debug_assert!(x < COLS && y < COLS);
    ((x + 3 * y) % COLS, x)
}

/// Global round index `t = 16 * q + s` for slot group `q` and slot `s in 0..16`.
#[inline]
#[must_use]
pub const fn time(q: usize, s: usize) -> usize {
    debug_assert!(s < SLOTS);
    SLOTS * q + s
}

/// Slot group and slot `(q, s) = (t / 16, t % 16)` for global round index `t`.
#[inline]
#[must_use]
pub const fn slot_of(t: usize) -> (usize, usize) {
    (t / SLOTS, t % SLOTS)
}

/// Source of an absorbed bit in block `m` at state position `(x, y, z)`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Absorbed {
    /// Bit `t` of coefficient `j` of polynomial row `rho` in `w1Encode(w1')`.
    W1Bit {
        /// Bit index within the coefficient (`0..w1_bits`).
        t: usize,
        /// Polynomial coefficient index (`0..256`).
        j: usize,
        /// Vector row index (`0..k`).
        rho: usize,
    },
    /// Bit `bit_index` (`0..512`) of the public message representative `mu`.
    Mu(usize),
    /// Constant SHAKE256 domain separation (`0x1F`) and `pad10*1` (`0x80`) padding bit in `{0, 1}`.
    Pad(u8),
    /// Unabsorbed zero bit (capacity bits `1088..1600`, or non-last blocks beyond rate).
    Zero,
}

impl Absorbed {
    /// Evaluates the absorbed bit for a given record's `mu` and `w1_bytes`.
    #[inline]
    #[must_use]
    pub fn eval(self, w1_bits: usize, mu: &[u8; TR_BYTES], w1_bytes: &[u8]) -> u8 {
        match self {
            Self::W1Bit { t, j, rho } => {
                let bit_idx = rho * N * w1_bits + j * w1_bits + t;
                (w1_bytes[bit_idx / 8] >> (bit_idx % 8)) & 1
            }
            Self::Mu(k) => (mu[k / 8] >> (k % 8)) & 1,
            Self::Pad(bit) => bit,
            Self::Zero => 0,
        }
    }
}

/// 16-bit slot masks (`bit s = 1` iff slot `s` is active) for one `(lane, q)` polynomial.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct SlotMasks {
    /// Live rounds: `t < 24 * perms`.
    pub live: u16,
    /// Inner rounds: live and not a boundary block absorption (`t = 0` or `t % 24 != 0`).
    pub inner: u16,
    /// Boundary rounds: `t = 24 * m` for `1 <= m < perms`.
    pub boundary: u16,
    /// Final output round: `t = 24 * perms - 1`.
    pub output: u16,
}

impl SlotMasks {
    /// True if slot `s` (`0..16`) is in `live`.
    #[inline]
    #[must_use]
    pub const fn is_live(self, s: usize) -> bool {
        (self.live >> s) & 1 == 1
    }

    /// True if slot `s` (`0..16`) is in `inner`.
    #[inline]
    #[must_use]
    pub const fn is_inner(self, s: usize) -> bool {
        (self.inner >> s) & 1 == 1
    }

    /// True if slot `s` (`0..16`) is in `boundary`.
    #[inline]
    #[must_use]
    pub const fn is_boundary(self, s: usize) -> bool {
        (self.boundary >> s) & 1 == 1
    }

    /// True if slot `s` (`0..16`) is in `output`.
    #[inline]
    #[must_use]
    pub const fn is_output(self, s: usize) -> bool {
        (self.output >> s) & 1 == 1
    }
}

/// Which of the three committed column families a Keccak polynomial belongs to.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum PolyFamily {
    /// Post-theta state lane `A'[x, y]` (`x, y in 0..5`).
    APrime { x: usize, y: usize },
    /// Pre-theta column parity `C[x]` (`x in 0..5`).
    C { x: usize },
    /// Post-theta column parity `C'[x]` (`x in 0..5`).
    CPrime { x: usize },
}

/// Keccak-f[1600] chain layout and sponge wiring for an ML-DSA parameter set.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct KeccakLayout {
    /// Parameter set (`S44`, `S65`, `S87`).
    pub set: Set,
    /// Number of Keccak-f[1600] permutations per signature (`7` for 44/65, `9` for 87).
    pub perms: usize,
    /// Time slots per polynomial (`16`).
    pub slots: usize,
    /// Slot groups `Q = ceil(24 * perms / 16)` (`11` for 44/65, `14` for 87).
    pub slot_groups: usize,
    /// State lanes (`25`).
    pub lanes: usize,
    /// Total live rounds `24 * perms` (`168` for 44/65, `216` for 87).
    pub rounds: usize,
    /// Output lanes `L = c_tilde_bytes / 8` (`4` for 44, `6` for 65, `8` for 87).
    pub output_lanes: usize,
    /// Bytes of `c~` (`32`, `48`, `64`).
    pub c_tilde_bytes: usize,
    /// Bytes of `w1Encode(w1')` (`768`, `768`, `1024`).
    pub w1_encode_bytes: usize,
    /// Bits per `w1'` coefficient (`6`, `4`, `4`).
    pub w1_bits: usize,
    /// Rows `k` of matrix `A` (`4`, `6`, `8`).
    pub k: usize,
    /// Precomputed slot masks per `(lane, q)` (`lane in 0..25`, `q in 0..slot_groups`).
    masks: Vec<[SlotMasks; LANES]>,
    /// Precomputed sponge wiring per permutation `m in 0..perms` and state bit `0..1600`.
    wiring: Vec<[Absorbed; STATE_BITS]>,
}

impl KeccakLayout {
    /// Builds the Keccak layout and sponge wiring for `set`.
    #[must_use]
    pub fn for_set(set: Set) -> Self {
        dispatch_set!(set, |P| Self::build_for::<P>())
    }

    fn build_for<P: ParamSet>() -> Self {
        let perms = P::N_PERMS;
        let rounds = ROUNDS_PER_PERM * perms;
        let slot_groups = rounds.div_ceil(SLOTS);
        let output_lanes = P::C_TILDE_BYTES / 8;

        let mut masks = Vec::with_capacity(slot_groups);
        for q in 0..slot_groups {
            let mut live = 0u16;
            let mut inner = 0u16;
            let mut boundary = 0u16;
            let mut output = 0u16;
            for s in 0..SLOTS {
                let t = time(q, s);
                if t < rounds {
                    live |= 1u16 << s;
                    if t > 0 && t.is_multiple_of(ROUNDS_PER_PERM) {
                        boundary |= 1u16 << s;
                    } else {
                        inner |= 1u16 << s;
                    }
                    if t + 1 == rounds {
                        output |= 1u16 << s;
                    }
                }
            }
            let slot_masks = SlotMasks {
                live,
                inner,
                boundary,
                output,
            };
            masks.push([slot_masks; LANES]);
        }

        let mut wiring = vec![[Absorbed::Zero; STATE_BITS]; perms];
        for k_bit in 0..(TR_BYTES * 8) {
            let slot = mu_bit_slot(k_bit);
            assert!(slot.perm < perms && slot.state_bit < RATE_BITS);
            assert_eq!(wiring[slot.perm][slot.state_bit], Absorbed::Zero);
            wiring[slot.perm][slot.state_bit] = Absorbed::Mu(k_bit);
        }
        for rho in 0..P::K {
            for j in 0..N {
                for t in 0..P::W1_BITS {
                    let slot = w1_bit_slot_for::<P>(t, j, rho);
                    assert!(slot.perm < perms && slot.state_bit < RATE_BITS);
                    assert_eq!(wiring[slot.perm][slot.state_bit], Absorbed::Zero);
                    wiring[slot.perm][slot.state_bit] = Absorbed::W1Bit { t, j, rho };
                }
            }
        }
        // Last block padding in the rate portion (0..1088 bits = 136 bytes):
        // 0x1F immediately after the data bytes and 0x80 at byte 135 (FIPS 202 SHAKE256 pad10*1).
        let msg_bytes = TR_BYTES + P::W1_ENCODE_BYTES;
        let last_perm = perms - 1;
        assert_eq!(msg_bytes / RATE, last_perm);
        let pad_start_byte = msg_bytes % RATE;
        for byte_idx in pad_start_byte..RATE {
            let mut pad_byte = 0u8;
            if byte_idx == pad_start_byte {
                pad_byte ^= 0x1F;
            }
            if byte_idx == RATE - 1 {
                pad_byte ^= 0x80;
            }
            for bit_in_byte in 0..8 {
                let st_bit = byte_idx * 8 + bit_in_byte;
                let bit = (pad_byte >> bit_in_byte) & 1;
                assert_eq!(wiring[last_perm][st_bit], Absorbed::Zero);
                wiring[last_perm][st_bit] = Absorbed::Pad(bit);
            }
        }

        Self {
            set: P::SET,
            perms,
            slots: SLOTS,
            slot_groups,
            lanes: LANES,
            rounds,
            output_lanes,
            c_tilde_bytes: P::C_TILDE_BYTES,
            w1_encode_bytes: P::W1_ENCODE_BYTES,
            w1_bits: P::W1_BITS,
            k: P::K,
            masks,
            wiring,
        }
    }

    /// Global round index `t = 16 * q + s`.
    #[inline]
    #[must_use]
    pub const fn time(&self, q: usize, s: usize) -> usize {
        time(q, s)
    }

    /// Slot group and slot `(q, s) = (t / 16, t % 16)`.
    #[inline]
    #[must_use]
    pub const fn slot_of(&self, t: usize) -> (usize, usize) {
        slot_of(t)
    }

    /// Rotation offset `r(x, y)` (FIPS 202 Table 2).
    #[inline]
    #[must_use]
    pub const fn rot(&self, x: usize, y: usize) -> u32 {
        rot(x, y)
    }

    /// Forward `pi(x, y)` lane permutation.
    #[inline]
    #[must_use]
    pub const fn pi(&self, x: usize, y: usize) -> (usize, usize) {
        pi(x, y)
    }

    /// Inverse `pi^-1(x, y)` lane permutation.
    #[inline]
    #[must_use]
    pub const fn pi_inv(&self, x: usize, y: usize) -> (usize, usize) {
        pi_inv(x, y)
    }

    /// Round constant `RC[r]` for `r in 0..24`.
    #[inline]
    #[must_use]
    pub const fn rc(&self, r: usize) -> u64 {
        RC[r]
    }

    /// Slot masks `(live, inner, boundary, output)` for `(lane, q)`.
    #[inline]
    #[must_use]
    pub fn masks(&self, lane_idx: usize, q: usize) -> SlotMasks {
        assert!(lane_idx < LANES && q < self.slot_groups);
        self.masks[q][lane_idx]
    }

    /// Live slot bitmask (`t < 24 * perms`) for `(lane, q)`.
    #[inline]
    #[must_use]
    pub fn live(&self, lane_idx: usize, q: usize) -> u16 {
        self.masks(lane_idx, q).live
    }

    /// Inner slot bitmask (live and not a block absorption of `m >= 1`) for `(lane, q)`.
    #[inline]
    #[must_use]
    pub fn inner(&self, lane_idx: usize, q: usize) -> u16 {
        self.masks(lane_idx, q).inner
    }

    /// Boundary slot bitmask (`t = 24 * m`, `1 <= m < perms`) for `(lane, q)`.
    #[inline]
    #[must_use]
    pub fn boundary(&self, lane_idx: usize, q: usize) -> u16 {
        self.masks(lane_idx, q).boundary
    }

    /// Output slot bitmask (`t = 24 * perms - 1`) for `(lane, q)`.
    #[inline]
    #[must_use]
    pub fn output(&self, lane_idx: usize, q: usize) -> u16 {
        self.masks(lane_idx, q).output
    }

    /// True if global round `t` is live (`t < 24 * perms`).
    #[inline]
    #[must_use]
    pub const fn is_live(&self, t: usize) -> bool {
        t < self.rounds
    }

    /// True if global round `t` is a boundary block absorption (`t = 24 * m`, `1 <= m < perms`).
    #[inline]
    #[must_use]
    pub const fn is_boundary(&self, t: usize) -> bool {
        t > 0 && t < self.rounds && t.is_multiple_of(ROUNDS_PER_PERM)
    }

    /// True if global round `t` is an inner round (live and not a boundary block absorption).
    #[inline]
    #[must_use]
    pub const fn is_inner(&self, t: usize) -> bool {
        self.is_live(t) && !self.is_boundary(t)
    }

    /// True if global round `t` is the final output round (`t = 24 * perms - 1`).
    #[inline]
    #[must_use]
    pub const fn is_output(&self, t: usize) -> bool {
        t + 1 == self.rounds
    }

    /// Sponge input wiring for permutation `m` (`0..perms`), lane `(x, y)` (`0..5`), bit `z` (`0..64`).
    #[inline]
    #[must_use]
    pub fn wire(&self, m: usize, x: usize, y: usize, z: usize) -> Absorbed {
        assert!(m < self.perms && x < COLS && y < COLS && z < LANE_BITS);
        self.wiring[m][state_bit(x, y, z)]
    }

    /// Number of polynomials per Keccak commitment group (`5 * slot_groups`: `55` or `70`).
    #[inline]
    #[must_use]
    pub const fn polys_per_group(&self) -> usize {
        COLS * self.slot_groups
    }

    /// Total committed Keccak polynomials across all 7 groups (`35 * slot_groups`: `385` or `490`).
    #[inline]
    #[must_use]
    pub const fn num_polys(&self) -> usize {
        TOTAL_LANE_FAMILIES * self.slot_groups
    }

    /// Polynomial index of `A'[x, y]` in slot group `q` (group `G_KA[y]`, order `y = 0..5` then `x = 0..5`).
    #[inline]
    #[must_use]
    pub const fn a_prime_poly(&self, x: usize, y: usize, q: usize) -> usize {
        debug_assert!(x < COLS && y < COLS && q < self.slot_groups);
        lane(x, y) * self.slot_groups + q
    }

    /// Polynomial index of `C[x]` in slot group `q` (group `G_KC`).
    #[inline]
    #[must_use]
    pub const fn c_poly(&self, x: usize, q: usize) -> usize {
        debug_assert!(x < COLS && q < self.slot_groups);
        (LANES + x) * self.slot_groups + q
    }

    /// Polynomial index of `C'[x]` in slot group `q` (group `G_KC'`).
    #[inline]
    #[must_use]
    pub const fn c_prime_poly(&self, x: usize, q: usize) -> usize {
        debug_assert!(x < COLS && q < self.slot_groups);
        (LANES + COLS + x) * self.slot_groups + q
    }

    /// Polynomial index range `start..end` of Keccak group `group_idx` (`0..5` for `G_KA[y]`,
    /// `5` for `G_KC`, `6` for `G_KC'`).
    #[inline]
    #[must_use]
    pub const fn group_poly_range(&self, group_idx: usize) -> core::ops::Range<usize> {
        debug_assert!(group_idx < KECCAK_GROUPS);
        let per_group = self.polys_per_group();
        (group_idx * per_group)..((group_idx + 1) * per_group)
    }

    /// Decodes a polynomial index `poly in 0..num_polys()` into `(PolyFamily, q)`.
    #[inline]
    #[must_use]
    pub const fn poly_coords(&self, poly: usize) -> (PolyFamily, usize) {
        debug_assert!(poly < self.num_polys());
        let family_lane = poly / self.slot_groups;
        let q = poly % self.slot_groups;
        let family = if family_lane < LANES {
            let (x, y) = lane_xy(family_lane);
            PolyFamily::APrime { x, y }
        } else if family_lane < LANES + COLS {
            PolyFamily::C {
                x: family_lane - LANES,
            }
        } else {
            PolyFamily::CPrime {
                x: family_lane - (LANES + COLS),
            }
        };
        (family, q)
    }
}
