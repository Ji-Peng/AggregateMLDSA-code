//! Witness table builders for the four Akita groups.
//!
//! This module builds G_main, the round-1 group: per main-domain position
//! (j, rho or s, i) the digits of z, u, v and the six w1Encode bits, in the table order
//! z_0..z_4, u_0..u_4, v_0..v_6, b_0..b_5 (relation H-u: u_0..u_3 the base-16 digits of B_ulo,
//! u_4 the base-6 top digit H_u - 3). Padded rows (i >= N) hold the canonical encoding of
//! the all-zero witness. The seed-dependent groups G_1, G_2, G_3 are built
//! from `crate::bridge::BridgeTables`. Behind features: the H-exact layout (`h-exact`, relation
//! H-exact: z by the bit-slack form, u as in H-u) and the S1+S2 layout (`fips`); their
//! column orders are fixed in docs/witness.md.
//!
//! Every committed value is a digit, a bit, a sign or a ternary value, so columns are `i8`.

use crate::derive::{derive, reencode_h, DeriveError, ReencodeError, SigWitness};
use crate::digits::*;
use crate::instance::PublicInstance;
use crate::native::{SmallK, VecK, VecL};
use crate::padding::{Geometry, GeometryError};
use crate::params::*;
use rayon::prelude::*;

/// Relation layout.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Layout {
    /// Relation H-u (default): z, u, v digits and the six bits; 23 columns. z is relaxed to
    /// [-2^17, 2^17), u is exact (`digits::U_STEP`).
    H,
    /// Relation H-exact: z exact by the bit-slack form
    /// (`digits::Z_BIT_SLACK16`: bz_0..bz_3, bz_hi, az), u as in H-u (`digits::U_STEP`);
    /// 24 columns.
    #[cfg(feature = "h-exact")]
    HExact,
    /// S1+S2 (feature `fips`): FIPS-faithful layout; 29 main and 3 per-signature columns.
    #[cfg(feature = "fips")]
    Fips,
}

impl Layout {
    /// Every layout compiled in.
    pub fn all() -> Vec<Layout> {
        #[allow(unused_mut)]
        let mut v = vec![Layout::H];
        #[cfg(feature = "h-exact")]
        v.push(Layout::HExact);
        #[cfg(feature = "fips")]
        v.push(Layout::Fips);
        v
    }

    /// Layout id, equal to the protocol's relation byte for H-u (0) and H-exact (1). Relation
    /// byte 2 names H-FIPS (`crate::hfips`, which has no base-16 layout here), so S1+S2
    /// serializes as 3 (witness crate only, never a relation byte).
    pub fn id(self) -> u8 {
        match self {
            Layout::H => 0,
            #[cfg(feature = "h-exact")]
            Layout::HExact => 1,
            #[cfg(feature = "fips")]
            Layout::Fips => 3,
        }
    }

    /// Inverse of `id` (only layouts compiled in).
    pub fn from_id(id: u8) -> Option<Layout> {
        Layout::all().into_iter().find(|l| l.id() == id)
    }

    /// Display name.
    pub fn name(self) -> &'static str {
        match self {
            Layout::H => "H-u",
            #[cfg(feature = "h-exact")]
            Layout::HExact => "H-exact",
            #[cfg(feature = "fips")]
            Layout::Fips => "S1+S2",
        }
    }

    /// Number of committed z digits (the slack bits included for H-exact; a_z excluded for
    /// S1+S2, whose a_z column is separate).
    pub fn z_digit_count(self) -> usize {
        match self {
            Layout::H => Z_DIGITS.len(),
            #[cfg(feature = "h-exact")]
            Layout::HExact => Z_BIT_SLACK16.len(),
            #[cfg(feature = "fips")]
            Layout::Fips => BZ_DIGITS.len(),
        }
    }

    /// True for relation H-exact.
    pub fn is_h_exact(self) -> bool {
        #[cfg(feature = "h-exact")]
        {
            self == Layout::HExact
        }
        #[cfg(not(feature = "h-exact"))]
        {
            false
        }
    }

    /// Number of committed u digits (the b_u digits for H-exact), if the layout commits u.
    pub fn u_digit_count(self) -> Option<usize> {
        match self {
            Layout::H => Some(U_STEP.len()),
            #[cfg(feature = "h-exact")]
            Layout::HExact => Some(U_STEP.len()),
            #[cfg(feature = "fips")]
            Layout::Fips => None,
        }
    }

    /// Column map of the layout.
    pub fn columns(self) -> ColumnMap {
        ColumnMap::new(self)
    }
}

/// Domain of a committed column.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ColumnDomain {
    /// Main domain (j, rho or s, i), 2^{d_main} cells.
    Main,
    /// Signature domain i, 2^{d_sig} cells (S1+S2 hint slack only).
    Sig,
}

/// One committed column.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ColumnSpec {
    /// Name, e.g. "z_3".
    pub name: String,
    /// Alphabet.
    pub alphabet: Alphabet,
    /// Domain.
    pub domain: ColumnDomain,
}

/// Start indices of the column groups of a layout.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ColumnMap {
    /// All columns in table order.
    pub specs: Vec<ColumnSpec>,
    /// First digit of z (H) or b_z (slack layouts).
    pub z: usize,
    /// Boolean slack a_z.
    pub az: Option<usize>,
    /// First digit of u (H) or b_u (H-exact).
    pub u: Option<usize>,
    /// Boolean slack a_u.
    pub au: Option<usize>,
    /// First digit of the quotient v.
    pub v: usize,
    /// Bit b_0 (b_t at b + t).
    pub b: usize,
    /// S1+S2: r1 digits (A base 8 weight 4, then B base 16 weight 1).
    pub r1: Option<usize>,
    /// S1+S2: first b_v digit of the magnitude v_r.
    pub bv: Option<usize>,
    /// S1+S2: Boolean slack a_v.
    pub av: Option<usize>,
    /// S1+S2: sign s.
    pub s: Option<usize>,
    /// S1+S2: g = h s.
    pub g: Option<usize>,
    /// S1+S2: edge flag e.
    pub e: Option<usize>,
    /// S1+S2: first b_h digit (signature domain).
    pub bh: Option<usize>,
    /// S1+S2: Boolean slack a_h (signature domain).
    pub ah: Option<usize>,
    /// Number of main-domain columns (they come first).
    pub n_main: usize,
}

impl ColumnMap {
    fn push(&mut self, name: String, alphabet: Alphabet, domain: ColumnDomain) -> usize {
        self.specs.push(ColumnSpec {
            name,
            alphabet,
            domain,
        });
        self.specs.len() - 1
    }

    fn push_digits(&mut self, prefix: &str, layout: DigitLayout, domain: ColumnDomain) -> usize {
        let first = self.specs.len();
        for k in 0..layout.len() {
            self.push(format!("{prefix}_{k}"), layout.alphabet(k), domain);
        }
        first
    }

    fn new(layout: Layout) -> Self {
        use ColumnDomain::Main;
        let mut m = ColumnMap {
            specs: Vec::new(),
            z: 0,
            az: None,
            u: None,
            au: None,
            v: 0,
            b: 0,
            r1: None,
            bv: None,
            av: None,
            s: None,
            g: None,
            e: None,
            bh: None,
            ah: None,
            n_main: 0,
        };
        match layout {
            Layout::H => {
                m.z = m.push_digits("z", Z_DIGITS, Main);
                let first = m.specs.len();
                for k in 0..U_STEP.len() {
                    m.push(format!("u_{k}"), U_STEP.alphabet(k), Main);
                }
                m.u = Some(first);
            }
            #[cfg(feature = "h-exact")]
            Layout::HExact => {
                m.z = m.push_digits("bz", ZLO_DIGITS, Main);
                m.push("bz_hi".into(), Alphabet::Bool, Main);
                m.az = Some(m.push("az".into(), Alphabet::Bool, Main));
                let first = m.specs.len();
                for k in 0..U_STEP.len() {
                    m.push(format!("u_{k}"), U_STEP.alphabet(k), Main);
                }
                m.u = Some(first);
            }
            #[cfg(feature = "fips")]
            Layout::Fips => {
                m.z = m.push_digits("bz", BZ_DIGITS, Main);
                m.az = Some(m.push("az".into(), Alphabet::Bool, Main));
                m.r1 = Some(m.push("r1_a".into(), Alphabet::Balanced(8), Main));
                m.push("r1_b".into(), Alphabet::Balanced(16), Main);
                m.bv = Some(m.push_digits("bvr", BV_DIGITS, Main));
                m.av = Some(m.push("avr".into(), Alphabet::Bool, Main));
                m.s = Some(m.push("s".into(), Alphabet::Sign, Main));
                m.g = Some(m.push("g".into(), Alphabet::Ternary, Main));
                m.e = Some(m.push("e".into(), Alphabet::Bool, Main));
            }
        }
        m.v = m.push_digits("v", V_DIGITS, Main);
        m.b = m.specs.len();
        for t in 0..W1_BITS {
            m.push(format!("b_{t}"), Alphabet::Bool, Main);
        }
        m.n_main = m.specs.len();
        #[cfg(feature = "fips")]
        if layout == Layout::Fips {
            m.bh = Some(m.push_digits("bh", BH_DIGITS, ColumnDomain::Sig));
            m.ah = Some(m.push("ah".into(), Alphabet::Bool, ColumnDomain::Sig));
        }
        m
    }

    /// Total number of columns.
    pub fn len(&self) -> usize {
        self.specs.len()
    }

    /// Always false.
    pub fn is_empty(&self) -> bool {
        self.specs.is_empty()
    }

    /// Number of signature-domain columns.
    pub fn n_sig(&self) -> usize {
        self.specs.len() - self.n_main
    }
}

/// Values of one record recomposed from (or to be encoded into) the tables.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RecordValues {
    /// z (from its digits or the slack form).
    pub z: VecL,
    /// u (H and H-exact; zero for S1+S2).
    pub u: VecK,
    /// The committed quotient of the layout's ring identity.
    pub v: VecK,
    /// w1' = sum_t 2^t b_t.
    pub w1: VecK,
    /// The raw bits b_t[rho][j].
    pub bits: [[[i64; N]; K]; W1_BITS],
    /// S1+S2 values (zero for other layouts).
    pub fips: FipsRecord,
}

/// S1+S2 per-record values (all zero outside the S1+S2 layout).
#[derive(Clone, Debug, PartialEq, Eq, Default)]
pub struct FipsRecord {
    /// r1.
    pub r1: Vec<i64>,
    /// r0 = s v_r - gamma2 e + (1 + s)/2 (virtual).
    pub r0: Vec<i64>,
    /// s.
    pub s: Vec<i64>,
    /// v_r.
    pub vr: Vec<i64>,
    /// e.
    pub e: Vec<i64>,
    /// g.
    pub g: Vec<i64>,
    /// u_h.
    pub uh: i64,
}

impl RecordValues {
    /// The canonical all-zero witness (padded rows): z = u = v = w1' = 0;
    /// for S1+S2 additionally (r1, r0) = Decompose(0) = (0, 0), so s = -1, and u_h = omega.
    pub fn zero(layout: Layout) -> Self {
        let mut r = RecordValues {
            z: [[0; N]; L],
            u: [[0; N]; K],
            v: [[0; N]; K],
            w1: [[0; N]; K],
            bits: [[[0; N]; K]; W1_BITS],
            fips: FipsRecord::default(),
        };
        let _ = &mut r;
        #[cfg(feature = "fips")]
        if layout == Layout::Fips {
            r.fips = FipsRecord {
                r1: vec![0; K * N],
                r0: vec![0; K * N],
                s: vec![-1; K * N],
                vr: vec![0; K * N],
                e: vec![0; K * N],
                g: vec![0; K * N],
                uh: OMEGA as i64,
            };
        }
        let _ = layout;
        r
    }

    /// Honest values from a derived witness.
    pub fn from_witness(layout: Layout, w: &SigWitness) -> Self {
        let mut bits = [[[0i64; N]; K]; W1_BITS];
        let mut w1 = [[0i64; N]; K];
        for rho in 0..K {
            for j in 0..N {
                w1[rho][j] = i64::from(w.w1[rho][j]);
                for (t, b) in bits.iter_mut().enumerate() {
                    b[rho][j] = i64::from(w.bit(t, j, rho));
                }
            }
        }
        #[allow(unused_mut)]
        let mut r = RecordValues {
            z: w.z,
            u: w.u,
            v: w.v,
            w1,
            bits,
            fips: FipsRecord::default(),
        };
        #[cfg(feature = "fips")]
        if layout == Layout::Fips {
            let flat = |a: &VecK| {
                a.iter()
                    .flat_map(|p| p.iter().copied())
                    .collect::<Vec<i64>>()
            };
            r.u = [[0; N]; K];
            r.v = w.fips.v;
            r.fips = FipsRecord {
                r1: flat(&w.fips.r1),
                r0: flat(&w.fips.r0),
                s: flat(&w.fips.s),
                vr: flat(&w.fips.vr),
                e: flat(&w.fips.e),
                g: flat(&w.fips.g),
                uh: w.fips.uh,
            };
        }
        let _ = layout;
        r
    }
}

/// Errors of the table builder.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum BuildError {
    /// Bad batch size.
    Geometry(GeometryError),
    /// Number of signatures differs from the instance.
    Count,
    /// Witness derivation failed for record i.
    Derive(usize, DeriveError),
    /// A digit of record i does not fit the column type (adversarial values only).
    Encode(usize, &'static str),
}

/// Per-record block: main columns (column-major, 1024 cells each) and signature-domain cells.
struct RecordBlock {
    main: Vec<i8>,
    sig: Vec<i8>,
}

fn to_i8(x: i64, what: &'static str) -> Result<i8, &'static str> {
    i8::try_from(x).map_err(|_| what)
}

fn encode_record(
    layout: Layout,
    map: &ColumnMap,
    vals: &RecordValues,
) -> Result<RecordBlock, &'static str> {
    let mut main = vec![0i8; map.n_main * POSITIONS];
    let mut sig = vec![0i8; map.n_sig()];
    let mut put =
        |col: usize, pos: usize, x: i64, what: &'static str| -> Result<(), &'static str> {
            main[col * POSITIONS + pos] = to_i8(x, what)?;
            Ok(())
        };
    for s in 0..L {
        for j in 0..N {
            let pos = j + N * s;
            let zv = vals.z[s][j];
            if layout.is_h_exact() {
                // bz_0..bz_3, bz_hi, az are contiguous.
                for (k, d) in Z_BIT_SLACK16.decompose(zv).into_iter().enumerate() {
                    put(map.z + k, pos, d, "z bit-slack digit")?;
                }
                continue;
            }
            match map.az {
                None => {
                    for (k, d) in Z_DIGITS.decompose(zv).into_iter().enumerate() {
                        put(map.z + k, pos, d, "z digit")?;
                    }
                }
                Some(az) => {
                    let (a, ds) = Z_EXACT.decompose(zv);
                    put(az, pos, a, "a_z")?;
                    for (k, d) in ds.into_iter().enumerate() {
                        put(map.z + k, pos, d, "b_z digit")?;
                    }
                }
            }
        }
    }
    for rho in 0..K {
        for j in 0..N {
            let pos = j + N * rho;
            if let Some(u0) = map.u {
                let uv = vals.u[rho][j];
                match map.au {
                    None => {
                        for (k, d) in U_STEP.decompose(uv).into_iter().enumerate() {
                            put(u0 + k, pos, d, "u digit")?;
                        }
                    }
                    Some(au) => {
                        let (a, ds) = U_EXACT.decompose(uv);
                        put(au, pos, a, "a_u")?;
                        for (k, d) in ds.into_iter().enumerate() {
                            put(u0 + k, pos, d, "b_u digit")?;
                        }
                    }
                }
            }
            for (k, d) in V_DIGITS.decompose(vals.v[rho][j]).into_iter().enumerate() {
                put(map.v + k, pos, d, "v digit")?;
            }
            for t in 0..W1_BITS {
                put(map.b + t, pos, vals.bits[t][rho][j], "bit")?;
            }
            if let (Some(r1), Some(bv), Some(av), Some(s), Some(g), Some(e)) =
                (map.r1, map.bv, map.av, map.s, map.g, map.e)
            {
                let f = &vals.fips;
                let idx = rho * N + j;
                let (ra, rb) = r1_decompose(f.r1[idx]);
                put(r1, pos, ra, "r1 digit")?;
                put(r1 + 1, pos, rb, "r1 digit")?;
                let (a, ds) = VR_EXACT.decompose(f.vr[idx]);
                put(av, pos, a, "a_v")?;
                for (k, d) in ds.into_iter().enumerate() {
                    put(bv + k, pos, d, "b_v digit")?;
                }
                put(s, pos, f.s[idx], "s")?;
                put(g, pos, f.g[idx], "g")?;
                put(e, pos, f.e[idx], "e")?;
            }
        }
    }
    if let (Some(bh), Some(ah)) = (map.bh, map.ah) {
        let (a, ds) = UH_EXACT.decompose(vals.fips.uh);
        sig[ah - map.n_main] = to_i8(a, "a_h")?;
        for (k, d) in ds.into_iter().enumerate() {
            sig[bh - map.n_main + k] = to_i8(d, "b_h digit")?;
        }
    }
    let _ = layout;
    Ok(RecordBlock { main, sig })
}

/// Cells of the canonical padded record: main columns (column-major, 1024 cells each) and the
/// signature-domain cells.
pub fn zero_record_cells(layout: Layout) -> (Vec<i8>, Vec<i8>) {
    let b = encode_record(layout, &layout.columns(), &RecordValues::zero(layout))
        .expect("zero record encodes");
    (b.main, b.sig)
}

/// The committed G_main tables of a batch (plus the S1+S2 signature-domain columns).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BatchTables {
    /// Layout.
    pub layout: Layout,
    /// Geometry.
    pub geom: Geometry,
    /// Columns in table order: main columns have 2^{d_main} cells, signature columns 2^{d_sig}.
    pub columns: Vec<Vec<i8>>,
}

impl BatchTables {
    /// Column map of this batch's layout.
    pub fn map(&self) -> ColumnMap {
        self.layout.columns()
    }

    /// Derives every live witness (strict) and builds the tables. Derivation, value extraction
    /// and digit encoding are streamed per record, so only the encoded blocks are kept.
    pub fn build(
        layout: Layout,
        inst: &PublicInstance,
        sigs: &[&[u8]],
    ) -> Result<Self, BuildError> {
        if sigs.len() != inst.records.len() {
            return Err(BuildError::Count);
        }
        let map = layout.columns();
        let blocks = inst
            .records
            .par_iter()
            .zip(sigs.par_iter())
            .enumerate()
            .map(|(i, (rec, sig))| {
                let w = derive(&rec.key, &rec.mu, sig).map_err(|e| BuildError::Derive(i, e))?;
                encode_record(layout, &map, &RecordValues::from_witness(layout, &w))
                    .map_err(|e| BuildError::Encode(i, e))
            })
            .collect::<Result<Vec<_>, _>>()?;
        Self::from_blocks(layout, inst.records.len(), &map, blocks)
    }

    /// Builds the tables from already derived witnesses (live records in order).
    pub fn from_witnesses(layout: Layout, wits: &[SigWitness]) -> Result<Self, BuildError> {
        let map = layout.columns();
        let blocks = wits
            .par_iter()
            .enumerate()
            .map(|(i, w)| {
                encode_record(layout, &map, &RecordValues::from_witness(layout, w))
                    .map_err(|e| BuildError::Encode(i, e))
            })
            .collect::<Result<Vec<_>, _>>()?;
        Self::from_blocks(layout, wits.len(), &map, blocks)
    }

    /// Builds the tables from record values (live records in order); padded rows get
    /// `RecordValues::zero`.
    pub fn from_values(layout: Layout, vals: &[RecordValues]) -> Result<Self, BuildError> {
        let map = layout.columns();
        let blocks = vals
            .par_iter()
            .enumerate()
            .map(|(i, v)| encode_record(layout, &map, v).map_err(|e| BuildError::Encode(i, e)))
            .collect::<Result<Vec<_>, _>>()?;
        Self::from_blocks(layout, vals.len(), &map, blocks)
    }

    fn from_blocks(
        layout: Layout,
        n: usize,
        map: &ColumnMap,
        blocks: Vec<RecordBlock>,
    ) -> Result<Self, BuildError> {
        let geom = Geometry::new(n).map_err(BuildError::Geometry)?;
        let zero =
            encode_record(layout, map, &RecordValues::zero(layout)).expect("zero record encodes");
        let columns: Vec<Vec<i8>> = (0..map.len())
            .into_par_iter()
            .map(|c| {
                if c < map.n_main {
                    let mut col = vec![0i8; geom.main_len()];
                    for (i, chunk) in col.chunks_mut(POSITIONS).enumerate() {
                        let src = blocks.get(i).unwrap_or(&zero);
                        chunk.copy_from_slice(&src.main[c * POSITIONS..(c + 1) * POSITIONS]);
                    }
                    col
                } else {
                    (0..geom.n_star)
                        .map(|i| blocks.get(i).unwrap_or(&zero).sig[c - map.n_main])
                        .collect()
                }
            })
            .collect();
        Ok(Self {
            layout,
            geom,
            columns,
        })
    }

    /// The six bit lanes b_t on the main domain.
    pub fn bit_lanes(&self) -> [&[i8]; W1_BITS] {
        let b = self.map().b;
        std::array::from_fn(|t| self.columns[b + t].as_slice())
    }

    /// Recomposes the values of record i (live or padded) from the digits.
    pub fn record_values(&self, i: usize) -> RecordValues {
        let map = self.map();
        let at = |c: usize, pos: usize| i64::from(self.columns[c][i * POSITIONS + pos]);
        let digits = |first: usize, n: usize, pos: usize| {
            (0..n).map(|k| at(first + k, pos)).collect::<Vec<_>>()
        };
        let mut r = RecordValues::zero(self.layout);
        for s in 0..L {
            for j in 0..N {
                let pos = j + N * s;
                let zd = digits(map.z, self.layout.z_digit_count(), pos);
                r.z[s][j] = match map.az {
                    _ if self.layout.is_h_exact() => Z_BIT_SLACK16.compose(&zd),
                    None => Z_DIGITS.compose(&zd),
                    Some(az) => Z_EXACT.compose(at(az, pos), &zd),
                };
            }
        }
        for rho in 0..K {
            for j in 0..N {
                let pos = j + N * rho;
                if let (Some(u0), Some(ul)) = (map.u, self.layout.u_digit_count()) {
                    let ud = digits(u0, ul, pos);
                    r.u[rho][j] = match map.au {
                        None => U_STEP.compose(&ud),
                        Some(au) => U_EXACT.compose(at(au, pos), &ud),
                    };
                }
                r.v[rho][j] = V_DIGITS.compose(&digits(map.v, V_DIGITS.len(), pos));
                let mut w1 = 0i64;
                for t in 0..W1_BITS {
                    let b = at(map.b + t, pos);
                    r.bits[t][rho][j] = b;
                    w1 += b << t;
                }
                r.w1[rho][j] = w1;
                if let (Some(r1), Some(bv), Some(av), Some(s), Some(g), Some(e)) =
                    (map.r1, map.bv, map.av, map.s, map.g, map.e)
                {
                    let idx = rho * N + j;
                    let f = &mut r.fips;
                    f.r1[idx] = r1_compose(at(r1, pos), at(r1 + 1, pos));
                    f.vr[idx] = VR_EXACT.compose(at(av, pos), &digits(bv, BV_DIGITS.len(), pos));
                    f.s[idx] = at(s, pos);
                    f.g[idx] = at(g, pos);
                    f.e[idx] = at(e, pos);
                    // r0 = s v_r - gamma2 e + (1 + s)/2 (integral exactly when s is odd).
                    f.r0[idx] =
                        f.s[idx] * f.vr[idx] - GAMMA2 * f.e[idx] + (1 + f.s[idx]).div_euclid(2);
                }
            }
        }
        if let (Some(bh), Some(ah)) = (map.bh, map.ah) {
            let d: Vec<i64> = (0..BH_DIGITS.len())
                .map(|k| i64::from(self.columns[bh + k][i]))
                .collect();
            r.fips.uh = UH_EXACT.compose(i64::from(self.columns[ah][i]), &d);
        }
        r
    }

    /// Re-encodes record i to FIPS 204 signature bytes with the public c~.
    pub fn reencode(
        &self,
        i: usize,
        c_tilde: &[u8; C_TILDE_BYTES],
    ) -> Result<Vec<u8>, ReencodeError> {
        let r = self.record_values(i);
        #[cfg(feature = "fips")]
        if self.layout == Layout::Fips {
            let mut g = [[0i64; N]; K];
            for rho in 0..K {
                for j in 0..N {
                    g[rho][j] = r.fips.g[rho * N + j];
                }
            }
            return crate::derive::reencode_fips(c_tilde, &r.z, &g);
        }
        let mut w1: SmallK = [[0; N]; K];
        for rho in 0..K {
            for j in 0..N {
                w1[rho][j] = u8::try_from(r.w1[rho][j]).map_err(|_| ReencodeError::Domain)?;
            }
        }
        reencode_h(c_tilde, &r.z, &r.u, &w1)
    }

    // -----------------------------------------------------------------------------------------
    // Serialization.
    // -----------------------------------------------------------------------------------------

    /// Magic bytes of the serialized form.
    pub const MAGIC: [u8; 8] = *b"AGGWTBL\0";
    /// Serialization version.
    pub const VERSION: u16 = 1;

    /// Little-endian serialization: magic, version (u16), layout id (u8), d_sig (u8),
    /// N (u64), column count (u32), then per column: domain (u8: 0 main, 1 signature),
    /// log2 length (u8), and the cells as i8.
    pub fn to_bytes(&self) -> Vec<u8> {
        let map = self.map();
        let total: usize = self.columns.iter().map(|c| c.len() + 2).sum();
        let mut out = Vec::with_capacity(24 + total);
        out.extend_from_slice(&Self::MAGIC);
        out.extend_from_slice(&Self::VERSION.to_le_bytes());
        out.push(self.layout.id());
        out.push(self.geom.d_sig as u8);
        out.extend_from_slice(&(self.geom.n as u64).to_le_bytes());
        out.extend_from_slice(&(self.columns.len() as u32).to_le_bytes());
        for (spec, col) in map.specs.iter().zip(&self.columns) {
            out.push(match spec.domain {
                ColumnDomain::Main => 0,
                ColumnDomain::Sig => 1,
            });
            out.push(col.len().trailing_zeros() as u8);
            out.extend(col.iter().map(|&x| x as u8));
        }
        out
    }

    /// Inverse of `to_bytes`; validates the header against the layout's column map.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, &'static str> {
        let mut rd = bytes;
        let mut take = |n: usize| -> Result<&[u8], &'static str> {
            if rd.len() < n {
                return Err("truncated");
            }
            let (a, b) = rd.split_at(n);
            rd = b;
            Ok(a)
        };
        if take(8)? != Self::MAGIC {
            return Err("bad magic");
        }
        if u16::from_le_bytes(take(2)?.try_into().unwrap()) != Self::VERSION {
            return Err("unknown version");
        }
        let layout = Layout::from_id(take(1)?[0]).ok_or("unknown or disabled layout")?;
        let d_sig = u32::from(take(1)?[0]);
        let n = usize::try_from(u64::from_le_bytes(take(8)?.try_into().unwrap()))
            .map_err(|_| "bad N")?;
        let geom = Geometry::new(n).map_err(|_| "bad N")?;
        if geom.d_sig != d_sig {
            return Err("d_sig inconsistent with N");
        }
        let ncols = u32::from_le_bytes(take(4)?.try_into().unwrap()) as usize;
        let map = layout.columns();
        if ncols != map.len() {
            return Err("column count");
        }
        let mut columns = Vec::with_capacity(ncols);
        for spec in &map.specs {
            let dom = take(1)?[0];
            let log2 = u32::from(take(1)?[0]);
            let (want_dom, want_log2) = match spec.domain {
                ColumnDomain::Main => (0u8, geom.d_main()),
                ColumnDomain::Sig => (1u8, geom.d_sig),
            };
            if dom != want_dom || log2 != want_log2 {
                return Err("column domain");
            }
            columns.push(take(1usize << log2)?.iter().map(|&x| x as i8).collect());
        }
        if !rd.is_empty() {
            return Err("trailing bytes");
        }
        Ok(Self {
            layout,
            geom,
            columns,
        })
    }
}
