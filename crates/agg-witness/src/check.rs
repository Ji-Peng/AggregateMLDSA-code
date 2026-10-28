//! Integer checker: the executable version of the spec's identities (`docs/spec.md`).
//!
//! `check` takes the public instance (verifier precomputation), the committed G_main tables
//! and, optionally, the bridge matrices with the committed G_1, G_2, G_3 tables, and evaluates
//! every sum-check identity pointwise on the Boolean domain over the integers:
//!
//! * stage 1: `bool_b`, `range_w1`, `b16_prod` for every base-16 digit, `bool_y1` with y^(1)
//!   defined by `def_y1`, `level2`, `level3`, `bool_y2`;
//! * stage 2: `def_y1` (the definition used by `bool_y1`), `leaf16` lo/hi (the virtual
//!   tables A^lo, A^hi equal Q_16^b(P_Z)), `leaf8`, `leaf6` (the top u digit of H-u), `leaf4`
//!   (and `leaf2` for H-exact's base-2 digit);
//! * stage 3: `ring`, the ring identity at every position with A, T zero on padded rows, and
//!   `link` (P_Z = Z(Z + 1)); the evaluation reductions are identities f = f pointwise;
//! * the Keccak argument: c~_i = SHAKE256(mu_i || w1Encode(w1'_i), 32) per live record through
//!   the seven-permutation sponge of `crate::keccak`, and the GF(2) side Y = M3 M2 M1 b;
//! * layout-specific: the H-exact slack Booleans and the S1+S2 identities;
//! * completeness (honest-form) checks that are not part of the relation: padded rows hold the
//!   canonical zero encoding and live values meet the honest bounds B_z, B_u, B_v.
//!
//! A committed column of the wrong alphabet fails the identity of its stage (`b16_prod` for
//! base 16, the leaf for bases 8, 6, 4, 2, the Booleanity check otherwise).

use crate::bridge::{self, BridgeMatrices, BridgeTables};
use crate::crt::az_integer;
use crate::digits::*;
use crate::instance::PublicInstance;
use crate::keccak::sponge_trace;
use crate::padding::Geometry;
use crate::params::*;
use crate::tables::{zero_record_cells, BatchTables, ColumnDomain, ColumnMap, Layout};
use std::collections::BTreeMap;

/// Where an identity lives in the protocol.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Stage {
    /// Table shapes and header.
    Shape,
    /// Sum-check stage 1.
    One,
    /// Sum-check stage 2.
    Two,
    /// Sum-check stage 3.
    Three,
    /// The Keccak argument and its GF(2) bridge side.
    Keccak,
    /// Layout-specific identities (H-exact, S1+S2).
    Layout,
    /// Honest-form checks, not part of the relation.
    Completeness,
}

/// The identities checked.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Identity {
    /// Column count and lengths.
    Shape,
    /// `bool_b`: b_t (b_t - 1) = 0.
    BoolB,
    /// `range_w1`: b5 (b4 + b3 b2) = 0.
    RangeW1,
    /// `b16_prod`: A^lo A^hi = 0 for a base-16 digit.
    B16Prod,
    /// `bool_y1` with y^(1) = sum M1 b - 2 kappa^(1).
    BoolY1,
    /// `level2`: sum M2 y^(1) = y^(2) + 2 kappa^(2).
    Level2,
    /// `level3`: sum M3 y^(2) = Y + 2 kappa^(3).
    Level3,
    /// `bool_y2`.
    BoolY2,
    /// `leaf16` lo/hi: A^b = Q_16^b(P_Z).
    Leaf16,
    /// `leaf8`: Q_8(P_Z) = 0.
    Leaf8,
    /// `leaf6`: Q_6(P_Z) = 0 (top digit of u in relation H-u).
    Leaf6,
    /// `leaf4`: Q_4(P_Z) = 0.
    Leaf4,
    /// Base-2 leaf of H-exact's top b_z digit: P_Z = 0.
    Leaf2,
    /// `ring`: Delta = 0 at every position.
    Ring,
    /// `link`: P_Z = Z (Z + 1).
    Link,
    /// Commitment hash: c~ = SHAKE256(mu || w1Encode(w1'), 32).
    CommitmentHash,
    /// GF(2) side: Y = M3 M2 M1 b.
    Gf2Bridge,
    /// Slack Booleans a_z, a_u, a_v, a_h and e^2 = e.
    BoolSlack,
    /// s^2 = 1.
    SignSquare,
    /// e (1 + s) = e v_r = e r1 = 0.
    EdgeProducts,
    /// g (g - 1)(g + 1) = 0.
    TernaryG,
    /// g (s - g) = 0.
    GSign,
    /// sum g^2 + u_h = omega.
    HintWeight,
    /// x (x - 44)(x + 44) = 0 with x = r1 + g - w1'.
    UseHintCubic,
    /// H-FIPS: a looked-up value (L_lo, B_zlo, u_h, the two halves of v) outside its table.
    Range16,
    /// H-FIPS: delta (delta - sigma) = 0.
    DeltaSign,
    /// H-FIPS: e (e - 1)(e - delta) = 0.
    EdgeE,
    /// H-FIPS: e (e + 1) w1' = 0 (e = 1 only at w1' = 0).
    WrapLow,
    /// H-FIPS: e (e - 1)(w1' - 43) = 0 (e = -1 only at w1' = 43).
    WrapHigh,
    /// Padded rows hold the canonical zero encoding.
    PaddingCanonical,
    /// Live values within the honest bounds B_z, B_u, B_v.
    HonestBound,
}

impl Identity {
    /// Stage of the identity.
    pub fn stage(self) -> Stage {
        use Identity::*;
        match self {
            Shape => Stage::Shape,
            BoolB | RangeW1 | B16Prod | BoolY1 | Level2 | Level3 | BoolY2 => Stage::One,
            Leaf16 | Leaf8 | Leaf6 | Leaf4 | Leaf2 => Stage::Two,
            Ring | Link => Stage::Three,
            CommitmentHash | Gf2Bridge => Stage::Keccak,
            BoolSlack | SignSquare | EdgeProducts | TernaryG | GSign | HintWeight
            | UseHintCubic | Range16 | DeltaSign | EdgeE | WrapLow | WrapHigh => Stage::Layout,
            PaddingCanonical | HonestBound => Stage::Completeness,
        }
    }
}

/// One failing check.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Failure {
    /// Identity.
    pub identity: Identity,
    /// Column or group name.
    pub what: String,
    /// Index in the column's domain (main index, D_t index or record).
    pub index: usize,
    /// Explanation.
    pub detail: String,
}

/// Result of a check: every failing identity with its count and the first failures.
#[derive(Clone, Debug, Default)]
pub struct CheckReport {
    /// First failures (at most `MAX_DETAILS`).
    pub failures: Vec<Failure>,
    /// Failing cells per identity.
    pub counts: BTreeMap<Identity, usize>,
}

const MAX_DETAILS: usize = 64;

impl CheckReport {
    pub(crate) fn fail(&mut self, identity: Identity, what: &str, index: usize, detail: String) {
        *self.counts.entry(identity).or_insert(0) += 1;
        if self.failures.len() < MAX_DETAILS {
            self.failures.push(Failure {
                identity,
                what: what.to_string(),
                index,
                detail,
            });
        }
    }

    /// No failure at all.
    pub fn is_ok(&self) -> bool {
        self.counts.is_empty()
    }

    /// No failure of the relation (completeness checks ignored).
    pub fn relation_ok(&self) -> bool {
        self.counts.keys().all(|i| i.stage() == Stage::Completeness)
    }

    /// Whether `id` failed somewhere.
    pub fn failed(&self, id: Identity) -> bool {
        self.counts.contains_key(&id)
    }

    /// The earliest stage with a relation failure.
    pub fn first_stage(&self) -> Option<Stage> {
        self.counts
            .keys()
            .map(|i| i.stage())
            .filter(|s| *s != Stage::Completeness)
            .min()
    }
}

/// Checker options.
#[derive(Clone, Copy, Debug)]
pub struct CheckOptions {
    /// Also run the completeness checks (padding canonical, honest bounds).
    pub completeness: bool,
}

impl Default for CheckOptions {
    fn default() -> Self {
        Self { completeness: true }
    }
}

/// Virtual tables of one digit column, as the prover materializes them for stages 1 and 2:
/// P_Z = Z (Z + 1) and, for base 16, A^lo = Q_16^lo(P_Z), A^hi = Q_16^hi(P_Z) (`None` = overflow,
/// certainly nonzero).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DigitVirtual {
    /// P_Z.
    pub p: Vec<i128>,
    /// A^lo (base 16 only).
    pub a_lo: Vec<Option<i128>>,
    /// A^hi (base 16 only).
    pub a_hi: Vec<Option<i128>>,
}

/// Honest materialization of the virtual digit tables.
pub fn digit_virtual(col: &[i8], base: u32) -> DigitVirtual {
    let p: Vec<i128> = col.iter().map(|&z| p_of(i64::from(z))).collect();
    let (a_lo, a_hi) = if base == 16 {
        (
            p.iter().map(|&x| q16_lo(x)).collect(),
            p.iter().map(|&x| q16_hi(x)).collect(),
        )
    } else {
        (Vec::new(), Vec::new())
    };
    DigitVirtual { p, a_lo, a_hi }
}

/// Checks one digit column with its (prover-supplied) virtual tables: `link`, `leaf16`,
/// `b16_prod`, `leaf8`, `leaf6`, `leaf4`, `leaf2`.
pub fn check_digit_column(
    rep: &mut CheckReport,
    name: &str,
    col: &[i8],
    base: u32,
    virt: &DigitVirtual,
) {
    for (x, &z) in col.iter().enumerate() {
        let p = virt.p[x];
        if p != p_of(i64::from(z)) {
            rep.fail(
                Identity::Link,
                name,
                x,
                format!("P_Z = {p} but Z (Z + 1) = {}", p_of(i64::from(z))),
            );
        }
        match base {
            16 => {
                let (lo, hi) = (virt.a_lo[x], virt.a_hi[x]);
                if lo != q16_lo(p) || hi != q16_hi(p) {
                    rep.fail(
                        Identity::Leaf16,
                        name,
                        x,
                        "A^lo/A^hi differ from Q_16(P_Z)".into(),
                    );
                }
                let prod_zero = lo == Some(0) || hi == Some(0);
                if !prod_zero {
                    rep.fail(
                        Identity::B16Prod,
                        name,
                        x,
                        format!("digit {z} outside [-8, 7]"),
                    );
                }
            }
            8 | 6 | 4 | 2 => {
                if !leaf_vanishes(base, p) {
                    let id = match base {
                        8 => Identity::Leaf8,
                        6 => Identity::Leaf6,
                        4 => Identity::Leaf4,
                        _ => Identity::Leaf2,
                    };
                    rep.fail(
                        id,
                        name,
                        x,
                        format!("digit {z} outside the base-{base} alphabet"),
                    );
                }
            }
            _ => panic!("unsupported base"),
        }
    }
}

fn check_column(rep: &mut CheckReport, name: &str, col: &[i8], alphabet: Alphabet) {
    match alphabet {
        Alphabet::Balanced(base) => {
            check_digit_column(rep, name, col, base, &digit_virtual(col, base))
        }
        Alphabet::Bool => {
            let id = if name.starts_with("b_") {
                Identity::BoolB
            } else if name.starts_with("y2_") {
                Identity::BoolY2
            } else {
                Identity::BoolSlack
            };
            for (x, &b) in col.iter().enumerate() {
                let b = i64::from(b);
                if b * (b - 1) != 0 {
                    rep.fail(id, name, x, format!("value {b} not Boolean"));
                }
            }
        }
        Alphabet::Sign => {
            for (x, &s) in col.iter().enumerate() {
                if i64::from(s) * i64::from(s) != 1 {
                    rep.fail(Identity::SignSquare, name, x, format!("s = {s}"));
                }
            }
        }
        Alphabet::Ternary => {
            for (x, &g) in col.iter().enumerate() {
                let g = i64::from(g);
                if g * (g - 1) * (g + 1) != 0 {
                    rep.fail(Identity::TernaryG, name, x, format!("g = {g}"));
                }
            }
        }
    }
}

fn shape_ok(
    rep: &mut CheckReport,
    inst: &PublicInstance,
    tables: &BatchTables,
    map: &ColumnMap,
) -> bool {
    let g = &tables.geom;
    if inst.geom != *g {
        rep.fail(
            Identity::Shape,
            "geometry",
            0,
            "instance and tables disagree on N".into(),
        );
        return false;
    }
    if tables.columns.len() != map.len() {
        rep.fail(Identity::Shape, "columns", 0, "column count".into());
        return false;
    }
    for (spec, col) in map.specs.iter().zip(&tables.columns) {
        let want = match spec.domain {
            ColumnDomain::Main => g.main_len(),
            ColumnDomain::Sig => g.n_star,
        };
        if col.len() != want {
            rep.fail(
                Identity::Shape,
                &spec.name,
                0,
                format!("length {} != {want}", col.len()),
            );
            return false;
        }
    }
    true
}

/// w1Encode bytes from the committed bits (the GF(2) copy b' hashed by the Keccak argument;
/// bits are taken modulo 2, Booleanity is `bool_b`'s job).
fn w1_bytes_from_bits(bits: &[[[i64; N]; K]; W1_BITS]) -> [u8; W1_ENCODE_BYTES] {
    let mut out = [0u8; W1_ENCODE_BYTES];
    for rho in 0..K {
        for j in 0..N {
            for (t, b) in bits.iter().enumerate() {
                let k = crate::keccak::w1_bit_index(t, j, rho);
                out[k / 8] |= ((b[rho][j] & 1) as u8) << (k % 8);
            }
        }
    }
    out
}

/// Runs every check. `bridge` adds the bridge identities and the GF(2) side.
pub fn check(
    inst: &PublicInstance,
    tables: &BatchTables,
    bridge: Option<(&BridgeMatrices, &BridgeTables)>,
    opts: CheckOptions,
) -> CheckReport {
    let mut rep = CheckReport::default();
    let map = tables.map();
    if !shape_ok(&mut rep, inst, tables, &map) {
        return rep;
    }
    let geom = tables.geom;
    // Alphabet identities of every committed column (stages 1 to 3 and layout-specific).
    for (spec, col) in map.specs.iter().zip(&tables.columns) {
        check_column(&mut rep, &spec.name, col, spec.alphabet);
    }
    // range_w1 at every position.
    let lanes = tables.bit_lanes();
    for x in 0..geom.main_len() {
        let b = |t: usize| i64::from(lanes[t][x]);
        if b(5) * (b(4) + b(3) * b(2)) != 0 {
            rep.fail(
                Identity::RangeW1,
                "b",
                x,
                format!("w1' = {}", (0..6).map(|t| b(t) << t).sum::<i64>()),
            );
        }
    }
    let (zero_main, zero_sig) = zero_record_cells(tables.layout);
    for i in 0..geom.n_star {
        check_record(&mut rep, inst, tables, &map, i, opts);
        if opts.completeness && !geom.live(i) {
            let main_ok = (0..map.n_main).all(|c| {
                tables.columns[c][i * POSITIONS..(i + 1) * POSITIONS]
                    == zero_main[c * POSITIONS..(c + 1) * POSITIONS]
            });
            let sig_ok = (0..map.n_sig()).all(|c| tables.columns[map.n_main + c][i] == zero_sig[c]);
            if !main_ok || !sig_ok {
                rep.fail(
                    Identity::PaddingCanonical,
                    "record",
                    i,
                    "padded row is not the zero encoding".into(),
                );
            }
        }
    }
    if let Some((mats, bt)) = bridge {
        check_bridge(&mut rep, &geom, &lanes, mats, bt);
    }
    rep
}

fn check_record(
    rep: &mut CheckReport,
    inst: &PublicInstance,
    tables: &BatchTables,
    map: &ColumnMap,
    i: usize,
    opts: CheckOptions,
) {
    let geom = tables.geom;
    let vals = tables.record_values(i);
    let live = geom.live(i);
    let az = if live {
        Some(az_integer(&inst.records[i].key, &vals.z))
    } else {
        None
    };
    let fips = map.r1.is_some();
    // Ring identity: Delta = A z - T - 2 gamma2 w1' - u - q v (H, H-exact) or
    // A z - T - 2 gamma2 r1 - r0 - q v (S1+S2); A = T = 0 on padded rows.
    for rho in 0..K {
        for j in 0..N {
            let x = crate::padding::main_index(j, rho, i);
            let (azv, t) = match &az {
                Some(a) => (a[rho][j], i128::from(inst.records[i].t[rho][j])),
                None => (0, 0),
            };
            let high = if fips {
                let idx = rho * N + j;
                2 * GAMMA2 * vals.fips.r1[idx] + vals.fips.r0[idx]
            } else {
                2 * GAMMA2 * vals.w1[rho][j] + vals.u[rho][j]
            };
            let delta = azv - t - i128::from(high) - i128::from(Q) * i128::from(vals.v[rho][j]);
            if delta != 0 {
                rep.fail(
                    Identity::Ring,
                    "ring",
                    x,
                    format!("Delta = {delta} (record {i})"),
                );
            }
        }
    }
    if fips {
        check_fips_record(rep, &vals, i);
    }
    if live {
        // The commitment hash through the seven-permutation sponge.
        let rec = &inst.records[i];
        let bytes = w1_bytes_from_bits(&vals.bits);
        let trace = sponge_trace(&rec.mu, &bytes);
        if trace.c_tilde != rec.c_tilde {
            rep.fail(
                Identity::CommitmentHash,
                "c_tilde",
                i,
                "SHAKE256(mu || w1Encode(w1')) != c~".into(),
            );
        }
        if opts.completeness {
            let mut bad = Vec::new();
            if vals.z.iter().flatten().any(|z| z.abs() > B_Z) {
                bad.push("z");
            }
            if map.u.is_some() && vals.u.iter().flatten().any(|u| u.abs() > B_U) {
                bad.push("u");
            }
            if vals.v.iter().flatten().any(|v| v.abs() > B_V) {
                bad.push("v");
            }
            if !bad.is_empty() {
                rep.fail(
                    Identity::HonestBound,
                    "record",
                    i,
                    format!("honest bound exceeded: {bad:?}"),
                );
            }
        }
    }
}

fn check_fips_record(rep: &mut CheckReport, vals: &crate::tables::RecordValues, i: usize) {
    let f = &vals.fips;
    let mut weight = 0i64;
    for rho in 0..K {
        for j in 0..N {
            let idx = rho * N + j;
            let x = crate::padding::main_index(j, rho, i);
            let (r1, s, vr, e, g) = (f.r1[idx], f.s[idx], f.vr[idx], f.e[idx], f.g[idx]);
            if e * (1 + s) != 0 || e * vr != 0 || e * r1 != 0 {
                rep.fail(
                    Identity::EdgeProducts,
                    "e",
                    x,
                    format!("e = {e}, s = {s}, v_r = {vr}, r1 = {r1}"),
                );
            }
            if g * (s - g) != 0 {
                rep.fail(Identity::GSign, "g", x, format!("g = {g}, s = {s}"));
            }
            let xx = r1 + g - vals.w1[rho][j];
            if xx * (xx - M_W) * (xx + M_W) != 0 {
                rep.fail(
                    Identity::UseHintCubic,
                    "usehint",
                    x,
                    format!("r1 + g - w1' = {xx}"),
                );
            }
            weight += g * g;
        }
    }
    if weight + f.uh != OMEGA as i64 {
        rep.fail(
            Identity::HintWeight,
            "u_h",
            i,
            format!("sum g^2 = {weight}, u_h = {}", f.uh),
        );
    }
}

fn check_bridge(
    rep: &mut CheckReport,
    geom: &Geometry,
    lanes: &[&[i8]; W1_BITS],
    mats: &BridgeMatrices,
    bt: &BridgeTables,
) {
    let lens = [
        1usize << geom.d1(),
        1usize << geom.d2(),
        1usize << geom.d3(),
    ];
    let specs = BridgeTables::specs();
    for (g, (cols, spec)) in [&bt.g1, &bt.g2, &bt.g3]
        .into_iter()
        .zip(specs.iter())
        .enumerate()
    {
        if cols.len() != spec.len() || cols.iter().any(|c| c.len() != lens[g]) {
            rep.fail(
                Identity::Shape,
                "bridge",
                g + 1,
                "bridge group shape".into(),
            );
            return;
        }
    }
    if bt.y.len() != lens[2] || !mats.has_spec_shapes() {
        rep.fail(Identity::Shape, "bridge", 0, "Y or matrix shape".into());
        return;
    }
    for (cols, spec) in [&bt.g1, &bt.g2, &bt.g3].into_iter().zip(specs.iter()) {
        for (col, (name, alphabet)) in cols.iter().zip(spec) {
            check_column(rep, name, col, *alphabet);
        }
    }
    // def_y1 defines y^(1); bool_y1 requires it Boolean.
    let sums1 = bridge::level1_sums(geom, lanes, &mats.m1);
    let mut y1: [Vec<i64>; W1_BITS] = Default::default();
    for t in 0..W1_BITS {
        let k1 = bt.kappa1(t);
        y1[t] = sums1[t].iter().zip(&k1).map(|(s, k)| s - 2 * k).collect();
        for (x, &y) in y1[t].iter().enumerate() {
            if y != 0 && y != 1 {
                rep.fail(Identity::BoolY1, &format!("y1_{t}"), x, format!("y1 = {y}"));
            }
        }
    }
    // level2.
    let sums2 = bridge::level2_sums(geom, &y1, &mats.m2);
    let y2: [Vec<i64>; W1_BITS] = std::array::from_fn(|t| bt.y2(t));
    for t in 0..W1_BITS {
        let k2 = bt.kappa2(t);
        for x in 0..lens[1] {
            if sums2[t][x] != y2[t][x] + 2 * k2[x] {
                rep.fail(
                    Identity::Level2,
                    &format!("level2_{t}"),
                    x,
                    format!("{} != {} + 2 {}", sums2[t][x], y2[t][x], k2[x]),
                );
            }
        }
    }
    // level3.
    let sums3 = bridge::level3_sums(geom, &y2, &mats.m3);
    let k3 = bt.kappa3();
    for x in 0..lens[2] {
        if sums3[x] != i64::from(bt.y[x]) + 2 * k3[x] {
            rep.fail(
                Identity::Level3,
                "level3",
                x,
                format!("{} != {} + 2 {}", sums3[x], bt.y[x], k3[x]),
            );
        }
    }
    // The GF(2) side with b' = b.
    let gf2 = bridge::gf2_composite(geom, lanes, mats);
    for x in 0..lens[2] {
        if gf2[x] != bt.y[x] {
            rep.fail(
                Identity::Gf2Bridge,
                "Y",
                x,
                "M3 M2 M1 b != Y over GF(2)".into(),
            );
        }
    }
}

/// Convenience: true when the layout's tables of a batch pass every check.
pub fn check_ok(inst: &PublicInstance, tables: &BatchTables) -> bool {
    check(inst, tables, None, CheckOptions::default()).is_ok()
}

/// The layout of a batch, for reports.
pub fn layout_name(layout: Layout) -> &'static str {
    layout.name()
}
