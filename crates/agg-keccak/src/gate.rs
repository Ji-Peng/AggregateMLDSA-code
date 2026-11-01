//! The chi-only Keccak-f[1600] gate and chains of it.
//!
//! A gate instance (one row of the Flock union, one `A_0` block) proves a chain
//! of `P` Keccak-f permutations whose input states are absorbed from input
//! columns and constants:
//!
//! ```text
//! state_0^(0) = absorb_0,  state_0^(m) = out^(m-1) XOR absorb_m,  out^(m) = Keccak-f(state_0^(m)).
//! ```
//!
//! Committed columns: the input bits, the materialized output bits, one
//! constant-one column, and the 1,600 chi AND outputs `t_r` of every round of
//! every permutation except the last round of the last permutation, which keeps
//! only the lanes `0 .. L` that hold an output bit (`L` = 1 + the highest output
//! lane; 4 of 25 for the SHAKE256 sponge, whose output `c~` is lanes `(0..3, 0)`).
//! The other lanes of that round feed nothing, so their AND bits are neither
//! committed nor constrained. Nothing else is committed: theta, rho, pi, iota, the
//! absorptions and the chaining are virtual XOR expressions, so every state is
//! implicit, `state_{r+1} = phi(state_r) XOR t_r XOR RC_r`. The R1CS has
//! `C = I` and one row per committed column:
//!
//! - constant: `1 * 1 = 1` (the column is pinned to 1 by the lincheck);
//! - input bit: `z_i * 1 = z_i`;
//! - output bit: `out[s] * 1 = z_o` (a linear row);
//! - chi AND bit: `(phi(state_r)[x+1, y, z] XOR 1) * phi(state_r)[x+2, y, z] = t_r[x, y, z]`.
//!
//! The constraint matrices are dense once the implicit states are substituted,
//! so they are never materialized: [`ChainGate`] implements Flock's
//! [`LincheckCircuit`] with a backward transpose walk, `K_r = phi^T(K_{r+1} + W_r)`,
//! extended across the permutation boundaries of the chain.

use agg_mldsa_simd::{keccak::Words8, Engine};
use flock_core::{
    lincheck::LincheckCircuit,
    r1cs::SparseBinaryMatrix,
    schedule::{IoWord, TableClass, TableType},
};
use flock_field::F128;
use sha3::{Digest, Sha3_256};

use crate::keccak::{self, Lanes, N_LANES, ROUND_CONSTANTS, STATE_BITS};

/// Where an absorbed state bit comes from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Absorb {
    /// The constant 1 (sponge padding).
    One,
    /// Global input bit `i` (fields concatenated in declaration order).
    Input(u32),
}

/// A word-aligned block of input columns.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct InputField {
    pub name: &'static str,
    /// Number of bits.
    pub bits: usize,
    /// Whether the field's words are wired to public words of the statement.
    pub public: bool,
    /// Columns the field occupies: a multiple of 128 of at least `bits`. Columns past
    /// `bits` are neither inputs nor outputs, so the gate pins them to zero (their rows have
    /// `A = B = 0`).
    pub cols: usize,
}

impl InputField {
    /// A field of `bits` bits in the fewest whole 128-bit words.
    #[must_use]
    pub const fn new(name: &'static str, bits: usize, public: bool) -> Self {
        Self {
            name,
            bits,
            public,
            cols: bits.div_ceil(128) * 128,
        }
    }
}

/// What one gate instance computes.
#[derive(Clone, Debug)]
pub struct ChainSpec {
    /// Domain label, bound into [`ChainGate::digest`].
    pub label: &'static str,
    /// Rounds per permutation: 24 for Keccak-f[1600]; fewer only in tests.
    pub n_rounds: usize,
    pub input_fields: Vec<InputField>,
    /// Per permutation, the absorbed `(state bit, source)` pairs (state bits distinct).
    pub absorbs: Vec<Vec<(u16, Absorb)>>,
    /// Bits of the last permutation's output that are materialized (public outputs).
    pub outputs: Vec<u16>,
}

impl ChainSpec {
    /// A single Keccak-f[1600]: 1,600 public input bits, 1,600 public output bits.
    #[must_use]
    pub fn permutation() -> Self {
        Self::permutation_rounds(keccak::N_ROUNDS)
    }

    /// [`Self::permutation`] reduced to `n_rounds` rounds (tests only).
    #[must_use]
    pub fn permutation_rounds(n_rounds: usize) -> Self {
        Self {
            label: "keccak-f1600",
            n_rounds,
            input_fields: vec![InputField::new("state_in", STATE_BITS, true)],
            absorbs: vec![(0..STATE_BITS)
                .map(|s| (s as u16, Absorb::Input(s as u32)))
                .collect()],
            outputs: (0..STATE_BITS as u16).collect(),
        }
    }

    /// Number of permutations in the chain.
    #[must_use]
    pub fn n_perms(&self) -> usize {
        self.absorbs.len()
    }

    /// Total input bits.
    #[must_use]
    pub fn n_inputs(&self) -> usize {
        self.input_fields.iter().map(|f| f.bits).sum()
    }
}

/// Column layout of one gate block.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GateLayout {
    /// `log2` of the block width.
    pub k_log: usize,
    /// First column of each input field (a multiple of 128).
    pub field_base: Vec<usize>,
    /// First input bit index of each field.
    pub field_offset: Vec<usize>,
    /// First output column (a multiple of 128).
    pub output_base: usize,
    /// The constant-one column.
    pub const_col: usize,
    /// u64 index of the first chi word: `t_m,r` lane `L` is u64 `t_base + (m R + r) 25 + L`.
    /// A multiple of 16 (the chi words start at an 8-word boundary).
    pub t_base: usize,
    /// Committed lanes `0 .. last_lanes` of the last round of the last permutation
    /// (every other round commits all 25): 1 + the highest lane holding an output bit.
    pub last_lanes: usize,
    /// Columns `[0, useful_bits)` carry data.
    pub useful_bits: usize,
}

impl GateLayout {
    fn new(spec: &ChainSpec) -> Self {
        let mut col = 0usize;
        let mut field_base = Vec::new();
        let mut field_offset = Vec::new();
        let mut off = 0usize;
        for f in &spec.input_fields {
            field_base.push(col);
            field_offset.push(off);
            assert!(
                f.cols.is_multiple_of(128) && f.cols >= f.bits,
                "a field spans whole words holding its bits"
            );
            col += f.cols;
            off += f.bits;
        }
        let output_base = col;
        col += spec.outputs.len().div_ceil(128) * 128;
        let const_col = col;
        // The chi words start at the next 8-word (1,024-bit) boundary after the constant,
        // so the non-chi columns `[0, 64 t_base)` are whole 8-word blocks and a gate's
        // inputs, outputs and constant can be addressed as aligned word subcubes (the
        // tied part's evaluation claims, `crate::tied`, docs/keccak.md).
        let chi_col = (const_col / 128 + 1).next_multiple_of(8) * 128;
        let t_base = chi_col / 64;
        // The final state is read only by the output rows, so the last round's AND
        // bits are needed only on the lanes that hold an output bit.
        let last_lanes = spec
            .outputs
            .iter()
            .map(|&o| usize::from(o) / 64 + 1)
            .max()
            .unwrap_or(N_LANES);
        let useful_bits =
            64 * (t_base + (spec.n_perms() * spec.n_rounds - 1) * N_LANES + last_lanes);
        let k_log = useful_bits.next_power_of_two().trailing_zeros() as usize;
        Self {
            k_log: k_log.max(7),
            field_base,
            field_offset,
            output_base,
            const_col,
            t_base,
            last_lanes,
            useful_bits,
        }
    }

    /// Committed chi lanes of round `r` of permutation `m`: 25, or
    /// [`Self::last_lanes`] for the last round of the last permutation.
    #[inline]
    #[must_use]
    pub fn round_lanes(&self, n_perms: usize, n_rounds: usize, m: usize, r: usize) -> usize {
        if m + 1 == n_perms && r + 1 == n_rounds {
            self.last_lanes
        } else {
            N_LANES
        }
    }

    /// u64 words of chi outputs of every round in full (25 lanes each): the
    /// length of the round-major `phi` tile of [`ChainGate::instance_phi_many`].
    #[must_use]
    pub fn full_chi_words(&self, n_perms: usize, n_rounds: usize) -> usize {
        n_perms * n_rounds * N_LANES
    }

    /// Column of global input bit `i`.
    #[must_use]
    pub fn input_col(&self, i: usize) -> usize {
        let f = self.field_offset.partition_point(|&o| o <= i) - 1;
        self.field_base[f] + (i - self.field_offset[f])
    }

    /// Column of chi bit `s` of round `r` of permutation `m`.
    #[inline]
    #[must_use]
    pub fn t_col(&self, n_rounds: usize, m: usize, r: usize, s: usize) -> usize {
        64 * (self.t_base + (m * n_rounds + r) * N_LANES) + s
    }

    /// u64 words per block that carry data.
    #[must_use]
    pub fn useful_words(&self) -> usize {
        self.useful_bits / 64
    }
}

/// Row, column and nonzero counts of one gate instance.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GateCost {
    /// Constraint rows (`= ` committed data columns, since `C = I`).
    pub rows: usize,
    /// Committed data bits per instance.
    pub committed_bits: usize,
    /// Of which chi AND outputs.
    pub chi_bits: usize,
    /// Nonzeros of `A_0 + B_0 + C_0` with the virtual XOR expressions counted as
    /// authored (one term per XOR operand reference, as in Flock's expression DAG).
    pub structural_nonzeros: usize,
    /// Block width `2^k_log`.
    pub block_bits: usize,
}

/// The chi-only Keccak chain gate: layout, native evaluation, witness rows and
/// the lincheck walker.
#[derive(Clone, Debug)]
pub struct ChainGate {
    spec: ChainSpec,
    layout: GateLayout,
    /// Per permutation: absorbed `(state bit, column)` from inputs.
    absorb_cols: Vec<Vec<(usize, usize)>>,
    /// Per permutation: state bits XORed with the constant 1.
    absorb_ones: Vec<Vec<usize>>,
    /// Word-level form of the absorptions and of the non-chi rows.
    plan: WordPlan,
}

/// One word-level absorption step: `s[state] ^= rotl(inputs[input] & mask, rot)`.
#[derive(Clone, Copy, Debug)]
struct AbsorbStep {
    state: usize,
    input: usize,
    rot: u32,
    mask: u64,
}

/// A run of whole words copied from a source to row words, the last one masked.
#[derive(Clone, Copy, Debug)]
struct WordRun {
    src: usize,
    dst: usize,
    words: usize,
    last_mask: u64,
}

impl WordRun {
    fn new(src_bit: usize, dst_bit: usize, bits: usize) -> Option<Self> {
        (src_bit.is_multiple_of(64) && dst_bit.is_multiple_of(64) && bits > 0).then(|| Self {
            src: src_bit / 64,
            dst: dst_bit / 64,
            words: bits.div_ceil(64),
            last_mask: if bits.is_multiple_of(64) {
                u64::MAX
            } else {
                (1u64 << (bits % 64)) - 1
            },
        })
    }

    fn copy(&self, src: &[u64], dst: &mut [u64]) {
        let (s, d) = (&src[self.src..self.src + self.words], &mut dst[self.dst..]);
        for (k, &x) in s.iter().enumerate() {
            d[k] |= if k + 1 == self.words {
                x & self.last_mask
            } else {
                x
            };
        }
    }
}

/// The gate's absorptions grouped by `(state word, input word, rotation)`, and the
/// non-chi rows (inputs, outputs, constant) as word copies where the layout allows.
#[derive(Clone, Debug)]
struct WordPlan {
    /// Per permutation: input steps and the lanes of the constant-one bits.
    absorb: Vec<(Vec<AbsorbStep>, Lanes)>,
    /// Per input field: its word run, if its bits start at a word boundary.
    input_runs: Vec<Option<WordRun>>,
    /// The output bits as one run of state words, if they are consecutive and aligned.
    output_run: Option<WordRun>,
    /// `b` of the non-chi words `[0, t_base)`: every input, output and the constant column.
    b_head: Vec<u64>,
}

impl WordPlan {
    fn new(spec: &ChainSpec, layout: &GateLayout) -> Self {
        let absorb = spec
            .absorbs
            .iter()
            .map(|list| {
                let mut steps: std::collections::BTreeMap<(usize, usize, u32), u64> =
                    std::collections::BTreeMap::new();
                let mut ones = [0u64; N_LANES];
                for &(st, src) in list {
                    let st = st as usize;
                    match src {
                        Absorb::One => ones[st / 64] ^= 1 << (st % 64),
                        Absorb::Input(i) => {
                            let i = i as usize;
                            let rot = ((st % 64) as u32).wrapping_sub((i % 64) as u32) % 64;
                            *steps.entry((st / 64, i / 64, rot)).or_insert(0) |= 1 << (i % 64);
                        }
                    }
                }
                let steps = steps
                    .into_iter()
                    .map(|((state, input, rot), mask)| AbsorbStep {
                        state,
                        input,
                        rot,
                        mask,
                    })
                    .collect();
                (steps, ones)
            })
            .collect();
        let input_runs = spec
            .input_fields
            .iter()
            .enumerate()
            .map(|(f, field)| {
                WordRun::new(layout.field_offset[f], layout.field_base[f], field.bits)
            })
            .collect();
        let consecutive = spec
            .outputs
            .iter()
            .enumerate()
            .all(|(j, &o)| usize::from(o) == usize::from(spec.outputs[0]) + j);
        let output_run = if consecutive && !spec.outputs.is_empty() {
            WordRun::new(
                usize::from(spec.outputs[0]),
                layout.output_base,
                spec.outputs.len(),
            )
        } else {
            None
        };
        let mut b_head = vec![0u64; layout.t_base];
        let mut set = |col: usize| b_head[col / 64] |= 1 << (col % 64);
        set(layout.const_col);
        for i in 0..spec.n_inputs() {
            set(layout.input_col(i));
        }
        for j in 0..spec.outputs.len() {
            set(layout.output_base + j);
        }
        Self {
            absorb,
            input_runs,
            output_run,
            b_head,
        }
    }
}

impl ChainGate {
    /// Validate `spec` and lay it out.
    ///
    /// # Panics
    /// If an absorbed state bit repeats, an input index or output bit is out of range.
    #[must_use]
    pub fn new(spec: ChainSpec) -> Self {
        assert!(spec.n_perms() >= 1 && spec.n_rounds >= 1);
        let layout = GateLayout::new(&spec);
        let n_inputs = spec.n_inputs();
        let mut absorb_cols = Vec::new();
        let mut absorb_ones = Vec::new();
        for list in &spec.absorbs {
            let mut seen = vec![false; STATE_BITS];
            let mut cols = Vec::new();
            let mut ones = Vec::new();
            for &(s, src) in list {
                let s = s as usize;
                assert!(s < STATE_BITS && !seen[s], "absorbed state bit {s} repeats");
                seen[s] = true;
                match src {
                    Absorb::One => ones.push(s),
                    Absorb::Input(i) => {
                        assert!((i as usize) < n_inputs, "input bit {i} out of range");
                        cols.push((s, layout.input_col(i as usize)));
                    }
                }
            }
            absorb_cols.push(cols);
            absorb_ones.push(ones);
        }
        for &o in &spec.outputs {
            assert!((o as usize) < STATE_BITS);
        }
        let plan = WordPlan::new(&spec, &layout);
        Self {
            spec,
            layout,
            absorb_cols,
            absorb_ones,
            plan,
        }
    }

    #[must_use]
    pub fn spec(&self) -> &ChainSpec {
        &self.spec
    }

    /// Per permutation: the absorbed `(state bit, column)` pairs and the state bits
    /// XORed with the constant (the row maps [`Self::apply_ab`] reads).
    pub(crate) fn absorb_lists(&self) -> (&[Vec<(usize, usize)>], &[Vec<usize>]) {
        (&self.absorb_cols, &self.absorb_ones)
    }

    /// Committed chi lanes of round `r` of permutation `m` ([`GateLayout::round_lanes`]).
    #[inline]
    #[must_use]
    pub fn round_lanes(&self, m: usize, r: usize) -> usize {
        self.layout
            .round_lanes(self.spec.n_perms(), self.spec.n_rounds, m, r)
    }

    #[must_use]
    pub fn layout(&self) -> &GateLayout {
        &self.layout
    }

    /// Words of packed input bits per instance.
    #[must_use]
    pub fn input_words(&self) -> usize {
        self.spec.n_inputs().div_ceil(64)
    }

    /// Native evaluation: the chain's final state for packed input bits.
    #[must_use]
    pub fn evaluate(&self, inputs: &[u64]) -> Lanes {
        let mut s = [0u64; N_LANES];
        for m in 0..self.spec.n_perms() {
            self.absorb(m, inputs, &mut s);
            keccak::keccak_f_rounds(&mut s, self.spec.n_rounds);
        }
        s
    }

    /// The materialized output bits for packed input bits.
    #[must_use]
    pub fn output_bits(&self, inputs: &[u64]) -> Vec<bool> {
        let s = self.evaluate(inputs);
        self.spec
            .outputs
            .iter()
            .map(|&o| (s[o as usize / 64] >> (o % 64)) & 1 == 1)
            .collect()
    }

    fn absorb(&self, m: usize, inputs: &[u64], s: &mut Lanes) {
        self.absorb_into(m, inputs, s);
    }

    /// [`Self::absorb_into`] bit by bit from the spec (the reference of [`Self::honest_z`]).
    fn absorb_bits(&self, m: usize, inputs: &[u64], s: &mut Lanes) {
        let n_in = self.spec.n_inputs();
        for &(st, src) in &self.spec.absorbs[m] {
            let v = match src {
                Absorb::One => 1,
                Absorb::Input(i) => {
                    let i = i as usize;
                    debug_assert!(i < n_in);
                    (inputs[i / 64] >> (i % 64)) & 1
                }
            };
            s[st as usize / 64] ^= v << (st % 64);
        }
    }

    /// XOR the bits absorbed before permutation `m` into `s`, from packed input bits.
    pub fn absorb_into(&self, m: usize, inputs: &[u64], s: &mut Lanes) {
        let (steps, ones) = &self.plan.absorb[m];
        for st in steps {
            s[st.state] ^= (inputs[st.input] & st.mask).rotate_left(st.rot);
        }
        for (x, o) in s.iter_mut().zip(ones) {
            *x ^= o;
        }
    }

    /// The non-chi words `[0, t_base)` of `z` (equal to `a` there): the input bits,
    /// the output bits of the final state `s` and the constant. Overwrites `head`.
    fn head_z(&self, inputs: &[u64], s: &Lanes, head: &mut [u64]) {
        let lay = &self.layout;
        head.fill(0);
        let c = lay.const_col;
        head[c / 64] |= 1 << (c % 64);
        for (f, run) in self.plan.input_runs.iter().enumerate() {
            match run {
                Some(run) => run.copy(inputs, head),
                None => {
                    let (off, bits) = (lay.field_offset[f], self.spec.input_fields[f].bits);
                    for i in off..off + bits {
                        let col = lay.input_col(i);
                        head[col / 64] |= ((inputs[i / 64] >> (i % 64)) & 1) << (col % 64);
                    }
                }
            }
        }
        match &self.plan.output_run {
            Some(run) => run.copy(s, head),
            None => {
                for (j, &o) in self.spec.outputs.iter().enumerate() {
                    let col = lay.output_base + j;
                    head[col / 64] |= ((s[o as usize / 64] >> (o % 64)) & 1) << (col % 64);
                }
            }
        }
    }

    /// Write one instance's rows: the committed bits `z` and the row values
    /// `a = A_0 z`, `b = B_0 z` over the first [`GateLayout::useful_words`] words.
    /// The slices must be zeroed on entry. Returns the final state.
    ///
    /// One Keccak walk emits all three: per round `phi(s)` is computed once and
    /// gives the chi row values `a = !phi(s)[x + 1]`, `b = phi(s)[x + 2]`, the
    /// committed AND bits `t = a & b` and the next state `phi(s) ^ t ^ RC`.
    /// On the honest `z` this is exactly [`Self::honest_z`] followed by
    /// [`Self::apply_ab`] (which recomputes the states from `z`): the same
    /// inputs, constant bit and states enter every row.
    pub fn instance_rows(
        &self,
        inputs: &[u64],
        z: &mut [u64],
        a: &mut [u64],
        b: &mut [u64],
    ) -> Lanes {
        let lay = &self.layout;
        let n_in = self.spec.n_inputs();
        assert!(inputs.len() >= n_in.div_ceil(64));
        let r_n = self.spec.n_rounds;
        let mut s = [0u64; N_LANES];
        for m in 0..self.spec.n_perms() {
            self.absorb(m, inputs, &mut s);
            for r in 0..r_n {
                let bl = keccak::phi(&s);
                let base = lay.t_base + (m * r_n + r) * N_LANES;
                let kept = self.round_lanes(m, r);
                for y in 0..5 {
                    for x in 0..5 {
                        let l = keccak::lane(x, y);
                        let al = !bl[keccak::lane((x + 1) % 5, y)];
                        let bv = bl[keccak::lane((x + 2) % 5, y)];
                        let t = al & bv;
                        // Lanes past `kept` of the last round are not committed; the
                        // returned final state still carries them in full.
                        if l < kept {
                            a[base + l] = al;
                            b[base + l] = bv;
                            z[base + l] = t;
                        }
                        s[l] = bl[l] ^ t;
                    }
                }
                s[0] ^= ROUND_CONSTANTS[r];
            }
        }
        // Constant row: A = B = e_const, z_const = 1. Input and output rows: z and
        // A = the bit, B = e_const. The rounds wrote only words from t_base on,
        // so the head words of z are still the zeros of entry and head_z can
        // write them in place (no scratch row per call).
        let t = lay.t_base;
        let head = &mut z[..t];
        self.head_z(inputs, &s, head);
        for (x, &h) in a[..t].iter_mut().zip(head.iter()) {
            *x |= h;
        }
        for (x, &bh) in b[..t].iter_mut().zip(&self.plan.b_head) {
            *x |= bh;
        }
        s
    }

    /// [`Self::instance_rows`] for up to eight instances in lockstep, word-major:
    /// `z[u][i]` is word `u` of instance `i` (likewise `a`, `b`), each slice
    /// [`GateLayout::useful_words`] long with any contents on entry. `inputs(i)`
    /// are instance `i`'s packed input bits; `absorb(i, m, s)` XORs the bits it
    /// absorbs before permutation `m` into `s` and must equal
    /// [`Self::absorb_into`] on `inputs(i)` (callers holding the absorbed bytes
    /// pass a faster equal one). Lanes `live..8` of `z` come out zero (the
    /// lincheck stripe reads whole groups of eight); their `a` and `b` are
    /// unspecified. Returns the final states (zero for the dead lanes).
    ///
    /// # Panics
    /// If `live` is not in `1..=8` or a slice has the wrong length.
    #[allow(clippy::too_many_arguments)]
    pub fn instance_rows_many<'x, I, A>(
        &self,
        engine: &Engine,
        live: usize,
        inputs: I,
        mut absorb: A,
        z: &mut [Words8],
        a: &mut [Words8],
        b: &mut [Words8],
    ) -> [Lanes; 8]
    where
        I: Fn(usize) -> &'x [u64],
        A: FnMut(usize, usize, &mut Lanes),
    {
        let lay = &self.layout;
        let w = lay.useful_words();
        assert!((1..=8).contains(&live), "one to eight live instances");
        assert!(z.len() == w && a.len() == w && b.len() == w, "row lengths");
        let r_n = self.spec.n_rounds;
        let per_perm = r_n * N_LANES;
        let mut st: [Words8; N_LANES] = [[0u64; 8]; N_LANES];
        for m in 0..self.spec.n_perms() {
            for i in 0..live {
                let mut s: Lanes = std::array::from_fn(|l| st[l][i]);
                absorb(i, m, &mut s);
                for (l, x) in s.iter().enumerate() {
                    st[l][i] = *x;
                }
            }
            let base = lay.t_base + m * per_perm;
            let kept = self.round_lanes(m, r_n - 1);
            if kept == N_LANES {
                let range = base..base + per_perm;
                engine.chi_rounds(
                    &mut st,
                    0,
                    &mut z[range.clone()],
                    &mut a[range.clone()],
                    &mut b[range],
                );
            } else {
                // Pruned last round: the full rounds in place, the last one through a
                // 25-word scratch of which the committed lanes `0 .. kept` are copied.
                let range = base..base + per_perm - N_LANES;
                engine.chi_rounds(
                    &mut st,
                    0,
                    &mut z[range.clone()],
                    &mut a[range.clone()],
                    &mut b[range],
                );
                let mut last = [[[0u64; 8]; N_LANES]; 3];
                let [lz, la, lb] = &mut last;
                engine.chi_rounds(&mut st, r_n - 1, lz, la, lb);
                let u = base + per_perm - N_LANES;
                z[u..u + kept].copy_from_slice(&lz[..kept]);
                a[u..u + kept].copy_from_slice(&la[..kept]);
                b[u..u + kept].copy_from_slice(&lb[..kept]);
            }
        }
        let t = lay.t_base;
        let mut finals = [[0u64; N_LANES]; 8];
        let mut head = vec![0u64; t];
        for i in 0..8 {
            if i < live {
                finals[i] = std::array::from_fn(|l| st[l][i]);
                self.head_z(inputs(i), &finals[i], &mut head);
                for u in 0..t {
                    z[u][i] = head[u];
                    a[u][i] = head[u];
                    b[u][i] = self.plan.b_head[u];
                }
            } else {
                for row in z.iter_mut() {
                    row[i] = 0;
                }
            }
        }
        finals
    }

    /// The compact form of [`Self::instance_rows_many`]: per group of eight, only the
    /// non-chi words of `z` (`head`, [`GateLayout::t_base`] words; `a` equals them
    /// and `b` is [`Self::head_b`]) and `phi` of every round in full (`phi`,
    /// [`GateLayout::full_chi_words`] words, round by round as [`Engine::phi_rounds`]
    /// writes them), from which the chi words `[t_base, useful_words)` of `a`, `b`
    /// and `z` follow: `a = !phi[x + 1, y]`, `b = phi[x + 2, y]`, `z = a & b` (the
    /// pruned last round reads row `y = 0` of its `phi` only). The chi words of `z`
    /// pass through `chi(u, words)` per round (`u` the first word index, the
    /// round's committed lanes, [`Self::round_lanes`] words; lanes `live..8` zero)
    /// and are not stored. Lanes `live..8` of `head` come out zero; their `phi` is
    /// unspecified. Returns the final states.
    ///
    /// # Panics
    /// As [`Self::instance_rows_many`].
    #[allow(clippy::too_many_arguments)]
    pub fn instance_phi_many<'x, I, A, C>(
        &self,
        engine: &Engine,
        live: usize,
        inputs: I,
        mut absorb: A,
        head: &mut [Words8],
        phi: &mut [Words8],
        mut chi: C,
    ) -> [Lanes; 8]
    where
        I: Fn(usize) -> &'x [u64],
        A: FnMut(usize, usize, &mut Lanes),
        C: FnMut(usize, &[Words8]),
    {
        let lay = &self.layout;
        let t = lay.t_base;
        let r_n = self.spec.n_rounds;
        assert!((1..=8).contains(&live), "one to eight live instances");
        assert!(
            head.len() == t && phi.len() == lay.full_chi_words(self.spec.n_perms(), r_n),
            "row lengths"
        );
        let per_perm = r_n * N_LANES;
        let mut st: [Words8; N_LANES] = [[0u64; 8]; N_LANES];
        let mut zt = [[0u64; 8]; N_LANES];
        for m in 0..self.spec.n_perms() {
            for i in 0..live {
                let mut s: Lanes = std::array::from_fn(|l| st[l][i]);
                absorb(i, m, &mut s);
                for (l, x) in s.iter().enumerate() {
                    st[l][i] = *x;
                }
            }
            let base = m * per_perm;
            engine.phi_rounds(
                &mut st,
                0,
                &mut phi[base..base + per_perm],
                &mut zt,
                |k, z| {
                    let u = t + base + k * N_LANES;
                    let kept = self.round_lanes(m, k);
                    if live == 8 {
                        chi(u, &z[..kept]);
                    } else {
                        // Dead lanes of z are zero for the stripe.
                        let mut masked = *z;
                        for row in &mut masked {
                            row[live..].fill(0);
                        }
                        chi(u, &masked[..kept]);
                    }
                },
            );
        }
        let mut finals = [[0u64; N_LANES]; 8];
        let mut h = vec![0u64; t];
        for i in 0..8 {
            if i < live {
                finals[i] = std::array::from_fn(|l| st[l][i]);
                self.head_z(inputs(i), &finals[i], &mut h);
                for u in 0..t {
                    head[u][i] = h[u];
                }
            } else {
                for row in head.iter_mut() {
                    row[i] = 0;
                }
            }
        }
        finals
    }

    /// `b` of the non-chi words `[0, t_base)` (the same for every instance).
    #[must_use]
    pub fn head_b(&self) -> &[u64] {
        &self.plan.b_head
    }

    /// The honest committed bits for packed inputs (`z` zeroed on entry).
    /// Returns the final state.
    pub fn honest_z(&self, inputs: &[u64], z: &mut [u64]) -> Lanes {
        let lay = &self.layout;
        let n_in = self.spec.n_inputs();
        assert!(inputs.len() >= n_in.div_ceil(64));
        for i in 0..n_in {
            let col = lay.input_col(i);
            z[col / 64] |= ((inputs[i / 64] >> (i % 64)) & 1) << (col % 64);
        }
        let c = lay.const_col;
        z[c / 64] |= 1 << (c % 64);
        let r_n = self.spec.n_rounds;
        let mut s = [0u64; N_LANES];
        for m in 0..self.spec.n_perms() {
            self.absorb_bits(m, inputs, &mut s);
            for r in 0..r_n {
                let t = keccak::round(&mut s, r);
                let base = lay.t_base + (m * r_n + r) * N_LANES;
                let kept = self.round_lanes(m, r);
                z[base..base + kept].copy_from_slice(&t[..kept]);
            }
        }
        for (j, &o) in self.spec.outputs.iter().enumerate() {
            let col = lay.output_base + j;
            z[col / 64] |= ((s[o as usize / 64] >> (o % 64)) & 1) << (col % 64);
        }
        s
    }

    /// Evaluate the linear rows on an arbitrary committed vector: `a = A_0 z`,
    /// `b = B_0 z` (zeroed on entry). Implicit states are recomputed from the
    /// committed chi bits, `state_{r+1} = phi(state_r) + t_r + RC_r z_const`.
    pub fn apply_ab(&self, z: &[u64], a: &mut [u64], b: &mut [u64]) {
        let lay = &self.layout;
        let get = |col: usize| (z[col / 64] >> (col % 64)) & 1;
        let c = lay.const_col;
        let one = get(c);
        let one_mask = one.wrapping_neg();
        // Constant row: A = B = e_const.
        a[c / 64] |= one << (c % 64);
        b[c / 64] |= one << (c % 64);
        // Input rows: A = e_i, B = e_const.
        for i in 0..self.spec.n_inputs() {
            let col = lay.input_col(i);
            a[col / 64] |= get(col) << (col % 64);
            b[col / 64] |= one << (col % 64);
        }
        let r_n = self.spec.n_rounds;
        let mut s = [0u64; N_LANES];
        for m in 0..self.spec.n_perms() {
            for &(st, col) in &self.absorb_cols[m] {
                s[st / 64] ^= get(col) << (st % 64);
            }
            for &st in &self.absorb_ones[m] {
                s[st / 64] ^= one << (st % 64);
            }
            for r in 0..r_n {
                let bl = keccak::phi(&s);
                let base = lay.t_base + (m * r_n + r) * N_LANES;
                let kept = self.round_lanes(m, r);
                for y in 0..5 {
                    for x in 0..5 {
                        let l = keccak::lane(x, y);
                        if l < kept {
                            a[base + l] = bl[keccak::lane((x + 1) % 5, y)] ^ one_mask;
                            b[base + l] = bl[keccak::lane((x + 2) % 5, y)];
                            s[l] = bl[l] ^ z[base + l];
                        } else {
                            // Uncommitted lane of the last round: no output row reads it.
                            s[l] = bl[l];
                        }
                    }
                }
                s[0] ^= ROUND_CONSTANTS[r] & one_mask;
            }
        }
        // Output rows: A = final state bit, B = e_const.
        for (j, &o) in self.spec.outputs.iter().enumerate() {
            let col = lay.output_base + j;
            a[col / 64] |= ((s[o as usize / 64] >> (o % 64)) & 1) << (col % 64);
            b[col / 64] |= one << (col % 64);
        }
    }

    /// The Flock registry type: empty matrix stubs (the relation lives in the
    /// walker and is bound by [`Self::digest`]), the constant pin, and the IO
    /// schema (public input fields' words, then the output words).
    #[must_use]
    pub fn table_type(&self) -> TableType {
        let stub = || SparseBinaryMatrix::new(0, 0, Vec::new());
        TableType {
            k_log: self.layout.k_log,
            useful_bits: self.layout.useful_bits,
            a_0: stub(),
            b_0: stub(),
            c_0: stub(),
            const_pin: Some(self.layout.const_col),
            class: TableClass::Boolean,
            io_schema: self.io_schema(),
        }
    }

    /// Wireable words: every public input field's words (In), then the output words (Out).
    #[must_use]
    pub fn io_schema(&self) -> Vec<IoWord> {
        let mut schema = Vec::new();
        for (f, field) in self.spec.input_fields.iter().enumerate() {
            if field.public {
                let w0 = self.layout.field_base[f] / 128;
                schema.extend((0..field.bits.div_ceil(128)).map(|w| IoWord::input(w0 + w)));
            }
        }
        let w0 = self.layout.output_base / 128;
        schema.extend((0..self.spec.outputs.len().div_ceil(128)).map(|w| IoWord::output(w0 + w)));
        schema
    }

    /// Digest of the relation: label, rounds, layout, absorptions and outputs.
    #[must_use]
    pub fn digest(&self) -> [u8; 32] {
        let mut h = Sha3_256::new();
        h.update(b"agg-keccak/v1/chain-gate");
        let put = |h: &mut Sha3_256, v: usize| h.update((v as u64).to_le_bytes());
        h.update((self.spec.label.len() as u64).to_le_bytes());
        h.update(self.spec.label.as_bytes());
        put(&mut h, self.spec.n_rounds);
        put(&mut h, self.layout.k_log);
        put(&mut h, self.layout.useful_bits);
        put(&mut h, self.layout.const_col);
        put(&mut h, self.layout.output_base);
        put(&mut h, self.layout.t_base);
        put(&mut h, self.layout.last_lanes);
        put(&mut h, self.spec.input_fields.len());
        for (f, field) in self.spec.input_fields.iter().enumerate() {
            put(&mut h, field.bits);
            put(&mut h, usize::from(field.public));
            put(&mut h, self.layout.field_base[f]);
        }
        put(&mut h, self.spec.n_perms());
        for list in &self.spec.absorbs {
            put(&mut h, list.len());
            for &(s, src) in list {
                put(&mut h, s as usize);
                match src {
                    Absorb::One => put(&mut h, usize::MAX),
                    Absorb::Input(i) => put(&mut h, i as usize),
                }
            }
        }
        put(&mut h, self.spec.outputs.len());
        for &o in &self.spec.outputs {
            put(&mut h, o as usize);
        }
        h.finalize().into()
    }

    /// Rows, committed bits and structural nonzeros (see [`GateCost`]).
    #[must_use]
    pub fn cost(&self) -> GateCost {
        let n_in = self.spec.n_inputs();
        let n_out = self.spec.outputs.len();
        let chi_bits = 64 * (self.layout.useful_words() - self.layout.t_base);
        let pruned_bits = self.spec.n_perms() * self.spec.n_rounds * STATE_BITS - chi_bits;
        let rows = 1 + n_in + n_out + chi_bits;
        // C = I: one nonzero per row. Constant row: A, B one each. Input rows: A, B one each.
        // Output rows: A = one state bit (1 expression reference), B = const.
        // AND rows: A = (B[x+1] + 1): 2 references; B = B[x+2]: 1 reference.
        // Virtual expressions, per round: theta column parities 320 x 5, theta
        // D 320 x 2, theta output 1600 x 2, chi XOR 1600 x 2 (2 per committed AND
        // bit in the pruned last round), iota <= 64;
        // per absorbed bit 2 (the XOR with the previous output, none for the first).
        let rc_bits: usize = (0..self.spec.n_rounds)
            .map(|r| ROUND_CONSTANTS[r].count_ones() as usize)
            .sum::<usize>()
            * self.spec.n_perms();
        let xor_terms =
            self.spec.n_perms() * self.spec.n_rounds * (320 * 5 + 320 * 2 + 1600 * 2 + 1600 * 2)
                - 2 * pruned_bits
                + 2 * rc_bits
                + self
                    .spec
                    .absorbs
                    .iter()
                    .skip(1)
                    .map(|l| 2 * l.len())
                    .sum::<usize>();
        let structural_nonzeros = rows + 2 + 2 * n_in + 2 * n_out + 3 * chi_bits + xor_terms;
        GateCost {
            rows,
            committed_bits: rows,
            chi_bits,
            structural_nonzeros,
            block_bits: 1 << self.layout.k_log,
        }
    }
}

impl LincheckCircuit for ChainGate {
    fn n_cols(&self) -> usize {
        1 << self.layout.k_log
    }

    fn const_pin_col(&self) -> Option<usize> {
        Some(self.layout.const_col)
    }

    /// `comb[c] = alpha (eq^T A_0)[c] + (eq^T B_0)[c]`, by one backward walk over
    /// the implicit states: `K` holds the combined row weights as a functional on
    /// the current state and is pushed through `phi^T` round by round.
    fn fold_alpha_batched(&self, alpha: F128, eq: &[F128]) -> Vec<F128> {
        let lay = &self.layout;
        let k = 1usize << lay.k_log;
        assert_eq!(eq.len(), k, "eq length must equal the block width");
        let mut comb = vec![F128::ZERO; k];
        let c = lay.const_col;
        let mut konst = alpha * eq[c] + eq[c];
        // Input rows: A = e_i, B = e_const.
        for i in 0..self.spec.n_inputs() {
            let col = lay.input_col(i);
            comb[col] += alpha * eq[col];
            konst += eq[col];
        }
        // Output rows: A = final state bit, B = e_const.
        let mut kv = vec![F128::ZERO; STATE_BITS];
        let mut spare = vec![F128::ZERO; STATE_BITS];
        for (j, &o) in self.spec.outputs.iter().enumerate() {
            let e = eq[lay.output_base + j];
            kv[o as usize] += alpha * e;
            konst += e;
        }
        let r_n = self.spec.n_rounds;
        for m in (0..self.spec.n_perms()).rev() {
            for r in (0..r_n).rev() {
                // state_{r+1} = B_r + t_r + RC_r: scatter K onto the t_r columns and the constant.
                // The pruned last round commits lanes `0 .. kept` only; K is zero on the
                // other lanes there (the output rows read no other lane of the final state).
                let t0 = lay.t_col(r_n, m, r, 0);
                let kept = self.round_lanes(m, r);
                debug_assert!(kv[64 * kept..].iter().all(|&v| v == F128::ZERO));
                for s in 0..64 * kept {
                    comb[t0 + s] += kv[s];
                }
                let rc = ROUND_CONSTANTS[r];
                for z in 0..64 {
                    if (rc >> z) & 1 == 1 {
                        konst += kv[z];
                    }
                }
                // AND rows of round r: A = B_r[x+1] + 1 (weight alpha e), B = B_r[x+2] (weight e).
                let mut sum_e = F128::ZERO;
                for y in 0..5 {
                    for x in 0..5 {
                        let l = keccak::lane(x, y);
                        if l >= kept {
                            continue;
                        }
                        let l1 = 64 * keccak::lane((x + 1) % 5, y);
                        let l2 = 64 * keccak::lane((x + 2) % 5, y);
                        for zz in 0..64 {
                            let e = eq[t0 + 64 * l + zz];
                            sum_e += e;
                            kv[l1 + zz] += alpha * e;
                            kv[l2 + zz] += e;
                        }
                    }
                }
                konst += alpha * sum_e;
                keccak::phi_transpose_into(&kv, &mut spare);
                std::mem::swap(&mut kv, &mut spare);
            }
            // state_0^(m) = out^(m-1) + absorb_m.
            for &(s, col) in &self.absorb_cols[m] {
                comb[col] += kv[s];
            }
            for &s in &self.absorb_ones[m] {
                konst += kv[s];
            }
        }
        comb[c] += konst;
        comb
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use flock_core::test_rng::Rng;

    /// AND rows satisfy `a & b = z`, linear rows `a = z` with `b = 1`, and the
    /// final state is Keccak-f of the input state.
    #[test]
    fn permutation_rows_satisfy_relation_and_match_keccak_crate() {
        let gate = ChainGate::new(ChainSpec::permutation());
        assert_eq!(gate.layout().k_log, 16);
        let w = gate.layout().useful_words();
        let mut rng = Rng::new(11);
        for _ in 0..8 {
            let s0: Lanes = std::array::from_fn(|_| rng.next_u64());
            let (mut z, mut a, mut b) = (vec![0u64; w], vec![0u64; w], vec![0u64; w]);
            let out = gate.instance_rows(&s0, &mut z, &mut a, &mut b);
            let mut want = s0;
            ::keccak::f1600(&mut want);
            assert_eq!(out, want);
            for i in 0..w {
                assert_eq!(a[i] & b[i], z[i], "word {i}");
            }
            let bits = gate.output_bits(&s0);
            assert_eq!(bits, keccak::lanes_to_bits(&want));
        }
    }

    /// The one-walk `instance_rows` equals `honest_z` then `apply_ab`, for the
    /// permutation, the sponge gate (absorptions, constant ones, outputs) and
    /// a short chain.
    #[test]
    fn instance_rows_matches_honest_z_then_apply_ab() {
        let mut rng = Rng::new(0x1A57);
        for gate in [
            ChainGate::new(ChainSpec::permutation()),
            crate::sponge::sponge_gate(),
            ChainGate::new(mini_chain(3)),
            ChainGate::new(mini_chain_pruned(3)),
        ] {
            let w = gate.layout().useful_words();
            for _ in 0..4 {
                let inputs: Vec<u64> = (0..gate.input_words()).map(|_| rng.next_u64()).collect();
                let (mut z, mut a, mut b) = (vec![0u64; w], vec![0u64; w], vec![0u64; w]);
                let out = gate.instance_rows(&inputs, &mut z, &mut a, &mut b);
                let (mut z2, mut a2, mut b2) = (vec![0u64; w], vec![0u64; w], vec![0u64; w]);
                let out2 = gate.honest_z(&inputs, &mut z2);
                gate.apply_ab(&z2, &mut a2, &mut b2);
                assert_eq!(out, out2);
                assert_eq!(z, z2);
                assert_eq!(a, a2);
                assert_eq!(b, b2);
            }
        }
    }

    /// `instance_rows_many` equals `instance_rows` lane by lane on every Keccak backend, with
    /// the gate's own absorption, for full and partial groups; dead lanes of `z` are zero.
    #[test]
    fn instance_rows_many_match_instance_rows_on_every_backend() {
        use agg_mldsa_simd::{Backend, Engine};
        let engines: Vec<Engine> = [Backend::Scalar, Backend::Avx2, Backend::Avx512]
            .into_iter()
            .filter_map(Engine::new)
            .collect();
        let mut rng = Rng::new(0x6A0F);
        for gate in [
            ChainGate::new(ChainSpec::permutation()),
            crate::sponge::sponge_gate(),
            ChainGate::new(mini_chain(3)),
            ChainGate::new(mini_chain(24)),
            ChainGate::new(mini_chain_pruned(3)),
        ] {
            let w = gate.layout().useful_words();
            for live in [8usize, 1, 5] {
                let inputs: Vec<Vec<u64>> = (0..live)
                    .map(|_| (0..gate.input_words()).map(|_| rng.next_u64()).collect())
                    .collect();
                for e in &engines {
                    let fill = rng.next_u64();
                    let (mut z, mut a, mut b) =
                        (vec![[fill; 8]; w], vec![[fill; 8]; w], vec![[fill; 8]; w]);
                    let finals = gate.instance_rows_many(
                        e,
                        live,
                        |i| &inputs[i][..],
                        |i, m, s| gate.absorb_into(m, &inputs[i], s),
                        &mut z,
                        &mut a,
                        &mut b,
                    );
                    let t = gate.layout().t_base;
                    let full = gate
                        .layout()
                        .full_chi_words(gate.spec().n_perms(), gate.spec().n_rounds);
                    let (mut head, mut phi) = (vec![[fill; 8]; t], vec![[fill; 8]; full]);
                    let mut z_chi = vec![[fill; 8]; w - t];
                    let finals_phi = gate.instance_phi_many(
                        e,
                        live,
                        |i| &inputs[i][..],
                        |i, m, s| gate.absorb_into(m, &inputs[i], s),
                        &mut head,
                        &mut phi,
                        |u, zr| z_chi[u - t..u - t + zr.len()].copy_from_slice(zr),
                    );
                    assert_eq!(finals_phi, finals);
                    for u in 0..t {
                        assert_eq!(head[u], z[u], "{:?} head word {u}", e.backend());
                    }
                    for q in 0..w - t {
                        let (g, l) = (q / 25, q % 25);
                        let (x, y) = (l % 5, l / 5);
                        let p1 = 25 * g + (x + 1) % 5 + 5 * y;
                        let p2 = 25 * g + (x + 2) % 5 + 5 * y;
                        for i in 0..live {
                            assert_eq!(!phi[p1][i], a[t + q][i], "{:?} a from phi", e.backend());
                            assert_eq!(phi[p2][i], b[t + q][i], "{:?} b from phi", e.backend());
                        }
                        assert_eq!(z_chi[q], z[t + q], "{:?} chi z word {q}", e.backend());
                    }
                    for i in 0..8 {
                        if i >= live {
                            assert!(z.iter().all(|row| row[i] == 0), "dead lane {i}");
                            continue;
                        }
                        let (mut zi, mut ai, mut bi) =
                            (vec![0u64; w], vec![0u64; w], vec![0u64; w]);
                        let out = gate.instance_rows(&inputs[i], &mut zi, &mut ai, &mut bi);
                        assert_eq!(finals[i], out, "{:?} final state", e.backend());
                        for u in 0..w {
                            assert_eq!(z[u][i], zi[u], "{:?} z word {u}", e.backend());
                            assert_eq!(a[u][i], ai[u], "{:?} a word {u}", e.backend());
                            assert_eq!(b[u][i], bi[u], "{:?} b word {u}", e.backend());
                        }
                    }
                }
            }
        }
    }

    fn random_f128(rng: &mut Rng) -> F128 {
        F128::new(rng.next_u64(), rng.next_u64())
    }

    /// `<comb, z> = alpha <eq, A z> + <eq, B z>` for random (not necessarily
    /// satisfying) committed vectors `z`: the walker is the transpose of the
    /// forward row evaluator, for the full 24-round gates.
    pub(crate) fn check_walker_adjoint(gate: &ChainGate, trials: usize, seed: u64) {
        let lay = gate.layout();
        let k = 1usize << lay.k_log;
        let w = lay.useful_words();
        let mut rng = Rng::new(seed);
        let eq: Vec<F128> = (0..k).map(|_| random_f128(&mut rng)).collect();
        let alpha = random_f128(&mut rng);
        let comb = gate.fold_alpha_batched(alpha, &eq);
        assert!(comb[lay.useful_bits..].iter().all(|&c| c == F128::ZERO));
        for trial in 0..trials {
            let mut z: Vec<u64> = (0..w).map(|_| rng.next_u64()).collect();
            if trial % 2 == 0 {
                let c = lay.const_col;
                z[c / 64] |= 1 << (c % 64);
            }
            let (mut a, mut b) = (vec![0u64; w], vec![0u64; w]);
            gate.apply_ab(&z, &mut a, &mut b);
            let mut lhs = F128::ZERO;
            let mut rhs = F128::ZERO;
            for col in 0..lay.useful_bits {
                let bit = |v: &[u64]| (v[col / 64] >> (col % 64)) & 1 == 1;
                if bit(&z) {
                    lhs += comb[col];
                }
                if bit(&a) {
                    rhs += alpha * eq[col];
                }
                if bit(&b) {
                    rhs += eq[col];
                }
            }
            assert_eq!(lhs, rhs, "trial {trial}");
        }
    }

    #[test]
    fn walker_is_adjoint_of_forward_rows_permutation() {
        check_walker_adjoint(&ChainGate::new(ChainSpec::permutation()), 24, 21);
        check_walker_adjoint(&ChainGate::new(ChainSpec::permutation_rounds(3)), 24, 22);
    }

    #[test]
    fn walker_is_adjoint_of_forward_rows_sponge() {
        check_walker_adjoint(&crate::sponge::sponge_gate(), 24, 23);
    }

    #[test]
    fn walker_is_adjoint_of_forward_rows_pruned_chain() {
        check_walker_adjoint(&ChainGate::new(mini_chain_pruned(3)), 24, 24);
    }

    /// The sponge gate commits lanes `(0..3, 0)` of the last round of P7 only: the
    /// lanes of `c~`; every other round keeps its 25 lanes.
    #[test]
    fn sponge_last_round_keeps_the_output_lanes() {
        let gate = crate::sponge::sponge_gate();
        let lay = gate.layout();
        assert_eq!(lay.last_lanes, 4);
        assert_eq!(gate.round_lanes(6, 23), 4);
        assert_eq!(gate.round_lanes(6, 22), 25);
        assert_eq!(gate.round_lanes(5, 23), 25);
        assert_eq!(lay.useful_words(), lay.t_base + (7 * 24 - 1) * 25 + 4);
        let pruned = ChainGate::new(mini_chain_pruned(2));
        assert_eq!(pruned.layout().last_lanes, 3);
        assert_eq!(ChainGate::new(mini_chain(2)).layout().last_lanes, 25);
        assert_eq!(
            ChainGate::new(ChainSpec::permutation()).layout().last_lanes,
            25
        );
    }

    /// A small chain exercising chaining, constant absorptions and partial outputs.
    pub(crate) fn mini_chain(n_rounds: usize) -> ChainSpec {
        ChainSpec {
            label: "mini-chain",
            n_rounds,
            input_fields: vec![
                InputField::new("x", 200, true),
                InputField::new("y", 300, false),
            ],
            absorbs: vec![
                (0..400u16)
                    .map(|s| (3 * s, Absorb::Input(u32::from(s))))
                    .collect(),
                (0..100u16)
                    .map(|s| (s, Absorb::Input(400 + u32::from(s))))
                    .chain([(700u16, Absorb::One), (1599, Absorb::One)])
                    .collect(),
            ],
            outputs: vec![0, 5, 64, 1000, 1599, 77, 300],
        }
    }

    /// [`mini_chain`] with outputs in lanes 0..2 only, so the last round keeps 3 of
    /// its 25 lanes (the pruning the sponge gate applies to `c~`).
    pub(crate) fn mini_chain_pruned(n_rounds: usize) -> ChainSpec {
        ChainSpec {
            label: "mini-chain-pruned",
            outputs: vec![0, 5, 64, 77, 130, 191],
            ..mini_chain(n_rounds)
        }
    }

    /// Explicit sparse rows of `A_0`, `B_0` by GF(2) support propagation (small configs only).
    fn explicit_rows(gate: &ChainGate) -> (Vec<Vec<usize>>, Vec<Vec<usize>>) {
        let lay = gate.layout();
        let spec = gate.spec();
        let k = 1usize << lay.k_log;
        let nw = k / 64;
        let unit = |col: usize| {
            let mut v = vec![0u64; nw];
            v[col / 64] |= 1 << (col % 64);
            v
        };
        let xor_into =
            |dst: &mut [u64], src: &[u64]| dst.iter_mut().zip(src).for_each(|(d, s)| *d ^= s);
        let to_cols = |v: &[u64]| -> Vec<usize> {
            (0..k)
                .filter(|&c| (v[c / 64] >> (c % 64)) & 1 == 1)
                .collect()
        };
        let c = lay.const_col;
        let mut a_rows = vec![Vec::new(); k];
        let mut b_rows = vec![Vec::new(); k];
        a_rows[c] = vec![c];
        b_rows[c] = vec![c];
        for i in 0..spec.n_inputs() {
            let col = lay.input_col(i);
            a_rows[col] = vec![col];
            b_rows[col] = vec![c];
        }
        let mut st: Vec<Vec<u64>> = vec![vec![0u64; nw]; STATE_BITS];
        for m in 0..spec.n_perms() {
            for &(s, src) in &spec.absorbs[m] {
                let col = match src {
                    Absorb::One => c,
                    Absorb::Input(i) => lay.input_col(i as usize),
                };
                xor_into(&mut st[s as usize], &unit(col));
            }
            for r in 0..spec.n_rounds {
                // theta
                let mut col_par = vec![vec![0u64; nw]; 5 * 64];
                for x in 0..5 {
                    for y in 0..5 {
                        for zz in 0..64 {
                            let src = st[keccak::bit(x, y, zz)].clone();
                            xor_into(&mut col_par[64 * x + zz], &src);
                        }
                    }
                }
                let mut a1 = st.clone();
                for x in 0..5 {
                    for y in 0..5 {
                        for zz in 0..64 {
                            let d1 = &col_par[64 * ((x + 4) % 5) + zz];
                            let d2 = &col_par[64 * ((x + 1) % 5) + (zz + 63) % 64];
                            let t = &mut a1[keccak::bit(x, y, zz)];
                            xor_into(t, d1);
                            xor_into(t, d2);
                        }
                    }
                }
                // rho, pi
                let mut bb = vec![Vec::new(); STATE_BITS];
                for y in 0..5 {
                    for x in 0..5 {
                        let aa = (x + 3 * y) % 5;
                        let rr = keccak::RHO_OFFSETS[aa][x] as usize;
                        for zz in 0..64 {
                            bb[keccak::bit(x, y, zz)] =
                                a1[keccak::bit(aa, x, (zz + 64 - rr) % 64)].clone();
                        }
                    }
                }
                // chi rows and the next state (committed lanes only)
                let kept = gate.round_lanes(m, r);
                for y in 0..5 {
                    for x in 0..5 {
                        for zz in 0..64 {
                            let s = keccak::bit(x, y, zz);
                            let mut nv = bb[s].clone();
                            if keccak::lane(x, y) < kept {
                                let row = lay.t_col(spec.n_rounds, m, r, s);
                                let mut av = bb[keccak::bit((x + 1) % 5, y, zz)].clone();
                                xor_into(&mut av, &unit(c));
                                a_rows[row] = to_cols(&av);
                                b_rows[row] = to_cols(&bb[keccak::bit((x + 2) % 5, y, zz)]);
                                xor_into(&mut nv, &unit(row));
                            }
                            if x == 0 && y == 0 && (ROUND_CONSTANTS[r] >> zz) & 1 == 1 {
                                xor_into(&mut nv, &unit(c));
                            }
                            st[s] = nv;
                        }
                    }
                }
            }
        }
        for (j, &o) in spec.outputs.iter().enumerate() {
            let row = lay.output_base + j;
            a_rows[row] = to_cols(&st[o as usize]);
            b_rows[row] = vec![c];
        }
        (a_rows, b_rows)
    }

    /// Against Flock's own sparse-matrix lincheck circuit on explicitly built
    /// matrices: the walker's fold, and `A z`, `B z` of the row evaluator.
    #[test]
    fn walker_matches_flock_sparse_fold_on_explicit_matrices() {
        use flock_core::lincheck::SparseMatrixCircuit;
        for spec in [
            ChainSpec::permutation_rounds(2),
            mini_chain(2),
            mini_chain_pruned(2),
        ] {
            let gate = ChainGate::new(spec);
            let lay = gate.layout().clone();
            let k = 1usize << lay.k_log;
            let (a_rows, b_rows) = explicit_rows(&gate);
            let a0 = SparseBinaryMatrix::new(k, k, a_rows.clone());
            let b0 = SparseBinaryMatrix::new(k, k, b_rows.clone());
            let sparse = SparseMatrixCircuit::new(&a0, &b0).with_const_pin(Some(lay.const_col));
            let mut rng = Rng::new(31);
            let eq: Vec<F128> = (0..k).map(|_| random_f128(&mut rng)).collect();
            let alpha = random_f128(&mut rng);
            assert_eq!(
                gate.fold_alpha_batched(alpha, &eq),
                sparse.fold_alpha_batched(alpha, &eq)
            );
            assert_eq!(gate.const_pin_col(), sparse.const_pin_col());
            // Row values on an honest and on a random committed vector.
            let w = lay.useful_words();
            let inputs: Vec<u64> = (0..gate.input_words()).map(|_| rng.next_u64()).collect();
            let mut honest = vec![0u64; w];
            gate.honest_z(&inputs, &mut honest);
            let random: Vec<u64> = (0..w).map(|_| rng.next_u64()).collect();
            for z in [honest, random] {
                let (mut a, mut b) = (vec![0u64; w], vec![0u64; w]);
                gate.apply_ab(&z, &mut a, &mut b);
                let eval = |rows: &[Vec<usize>], row: usize| {
                    rows[row]
                        .iter()
                        .fold(0u64, |acc, &col| acc ^ ((z[col / 64] >> (col % 64)) & 1))
                };
                for row in 0..lay.useful_bits {
                    assert_eq!(
                        (a[row / 64] >> (row % 64)) & 1,
                        eval(&a_rows, row),
                        "A row {row}"
                    );
                    assert_eq!(
                        (b[row / 64] >> (row % 64)) & 1,
                        eval(&b_rows, row),
                        "B row {row}"
                    );
                }
            }
        }
    }

    /// Nonzeros of `A_0` and `B_0` after substituting the implicit states
    /// (what a materialized sparse lowering would store), by GF(2) support
    /// propagation over the useful columns. Documentation numbers for
    /// `docs/keccak.md`; run with `--ignored --nocapture`.
    fn flattened_nonzeros(gate: &ChainGate) -> (usize, usize) {
        let lay = gate.layout();
        let spec = gate.spec();
        let nw = lay.useful_words();
        let c = lay.const_col;
        let pop = |v: &[u64]| v.iter().map(|w| w.count_ones() as usize).sum::<usize>();
        let flip = |v: &mut [u64], col: usize| v[col / 64] ^= 1 << (col % 64);
        // const row (1 + 1), input rows (1 + 1), output rows (support + 1)
        let mut nnz_a = 1 + spec.n_inputs();
        let mut nnz_b = 1 + spec.n_inputs() + spec.outputs.len();
        let mut st: Vec<Vec<u64>> = vec![vec![0u64; nw]; STATE_BITS];
        for m in 0..spec.n_perms() {
            for &(s, src) in &spec.absorbs[m] {
                let col = match src {
                    Absorb::One => c,
                    Absorb::Input(i) => lay.input_col(i as usize),
                };
                flip(&mut st[s as usize], col);
            }
            for r in 0..spec.n_rounds {
                let mut par = vec![vec![0u64; nw]; 5 * 64];
                for x in 0..5 {
                    for zz in 0..64 {
                        for y in 0..5 {
                            let src = &st[keccak::bit(x, y, zz)];
                            par[64 * x + zz]
                                .iter_mut()
                                .zip(src)
                                .for_each(|(d, s)| *d ^= s);
                        }
                    }
                }
                let mut bb = vec![Vec::new(); STATE_BITS];
                for y in 0..5 {
                    for x in 0..5 {
                        let a = (x + 3 * y) % 5;
                        let rr = keccak::RHO_OFFSETS[a][x] as usize;
                        for zz in 0..64 {
                            let zs = (zz + 64 - rr) % 64;
                            let mut v = st[keccak::bit(a, x, zs)].clone();
                            let d1 = &par[64 * ((a + 4) % 5) + zs];
                            let d2 = &par[64 * ((a + 1) % 5) + (zs + 63) % 64];
                            v.iter_mut()
                                .zip(d1)
                                .zip(d2)
                                .for_each(|((t, p), q)| *t ^= p ^ q);
                            bb[keccak::bit(x, y, zz)] = v;
                        }
                    }
                }
                let kept = gate.round_lanes(m, r);
                for y in 0..5 {
                    for x in 0..5 {
                        if keccak::lane(x, y) >= kept {
                            continue;
                        }
                        for zz in 0..64 {
                            let mut av = bb[keccak::bit((x + 1) % 5, y, zz)].clone();
                            flip(&mut av, c);
                            nnz_a += pop(&av);
                            nnz_b += pop(&bb[keccak::bit((x + 2) % 5, y, zz)]);
                        }
                    }
                }
                for s in 0..STATE_BITS {
                    let mut v = std::mem::take(&mut bb[s]);
                    if s < 64 * kept {
                        flip(&mut v, lay.t_col(spec.n_rounds, m, r, s));
                    }
                    if s < 64 && (ROUND_CONSTANTS[r] >> s) & 1 == 1 {
                        flip(&mut v, c);
                    }
                    st[s] = v;
                }
            }
        }
        for &o in &spec.outputs {
            nnz_a += pop(&st[o as usize]);
        }
        (nnz_a, nnz_b)
    }

    #[test]
    #[ignore = "documentation numbers; several seconds for the sponge"]
    fn print_gate_costs() {
        for (name, gate) in [
            ("keccak-f", ChainGate::new(ChainSpec::permutation())),
            ("sponge", crate::sponge::sponge_gate()),
        ] {
            let cost = gate.cost();
            let (a, b) = flattened_nonzeros(&gate);
            eprintln!(
                "{name}: k_log {} useful_bits {} rows {} committed {} chi {} structural_nnz {} flattened nnz A {a} B {b} C {} (total {})",
                gate.layout().k_log,
                gate.layout().useful_bits,
                cost.rows,
                cost.committed_bits,
                cost.chi_bits,
                cost.structural_nonzeros,
                cost.rows,
                a + b + cost.rows
            );
        }
    }

    #[test]
    fn flattened_nonzeros_match_explicit_rows() {
        for spec in [
            ChainSpec::permutation_rounds(2),
            mini_chain(2),
            mini_chain_pruned(2),
        ] {
            let gate = ChainGate::new(spec);
            let (a_rows, b_rows) = explicit_rows(&gate);
            let na: usize = a_rows.iter().map(Vec::len).sum();
            let nb: usize = b_rows.iter().map(Vec::len).sum();
            assert_eq!(flattened_nonzeros(&gate), (na, nb));
        }
    }

    #[test]
    fn layout_and_schema() {
        let gate = ChainGate::new(ChainSpec::permutation());
        let lay = gate.layout();
        assert_eq!(lay.field_base, vec![0]);
        assert_eq!(lay.output_base, 13 * 128);
        assert_eq!(lay.const_col, 26 * 128);
        assert_eq!(lay.useful_bits, 64 * (lay.t_base + 24 * 25));
        let schema = gate.io_schema();
        assert_eq!(schema.len(), 26);
        assert_eq!(schema[0], IoWord::input(0));
        assert_eq!(schema[25], IoWord::output(25));
        let cost = gate.cost();
        assert_eq!(cost.chi_bits, 38_400);
        assert_eq!(cost.rows, 1 + 1600 + 1600 + 38_400);
    }
}
