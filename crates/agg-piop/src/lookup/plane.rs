//! The lookup prover on lane-major `K` planes.
//!
//! The round polynomial is unique, so this path sends the same messages as the
//! row-major one; it differs in how the fraction trees and the session groups
//! hold their tables. Integer-path stacks recompute the layer above their leaves
//! from four quarters of the stacked column, and every session group reads
//! canonical coordinate planes.

use jolt_field::{Ring, Zero};
use rayon::prelude::*;

use super::{
    check_roots, evaluate_compressed, layer_instances, line, one, root_of, session_plan, Group,
    GroupKind, Lane, LookupError, LookupOutput, LookupShape, LookupStack, LookupSite,
    LookupTranscript, ProveStats, SessionProof, Source, Table, TreeKind, SYM_SPREAD,
    CONCURRENT_GROUP_ROUNDS_MIN_VARS, DEGREE,
};
use crate::kernel::{PlaneLayerGroup, PlaneLayerTree, PlaneLeafGroup};
use crate::OpeningField as E;

type Layer = agg_simd::lookup::Layer;

/// A stack recentred by the middle of its value range, whose leaf session runs
/// on the on-the-fly integer lanes of [`PlaneLeafGroup`].
struct StackCentre {
    /// The centre `c` subtracted from every stored value.
    center: i32,
}

impl StackCentre {
    /// `None` when the stack is shallower than 5 or its values spread over more
    /// than [`SYM_SPREAD`] (only a dishonest witness; it then takes the
    /// stored-layer path).
    fn of(stacked: &[i32]) -> Option<Self> {
        let depth = stacked.len().trailing_zeros() as usize;
        if depth < 5 {
            return None;
        }
        let (min, max) = stacked
            .par_iter()
            .with_min_len(1 << 16)
            .fold(|| (i32::MAX, i32::MIN), |(lo, hi), &v| (lo.min(v), hi.max(v)))
            .reduce(|| (i32::MAX, i32::MIN), |a, b| (a.0.min(b.0), a.1.max(b.1)));
        if i64::from(max) - i64::from(min) > SYM_SPREAD {
            return None;
        }
        let center = i32::try_from((i64::from(min) + i64::from(max)) / 2)
            .expect("between two i32");
        Some(Self { center })
    }

    /// The coefficients `[c, c_s, c_m, c_k, c_mm]` of the four lanes `S_0, S_1,
    /// M_0, M_1` for the trees `(X', gamma_p, gamma_q)` of the stack.
    fn coefficients(trees: &[(E, E, E)]) -> [E; 5] {
        let (mut constant, mut sum_s, mut sum_m, mut cross, mut mm) =
            (E::zero(), E::zero(), E::zero(), E::zero(), E::zero());
        for &(x, gp, gq) in trees {
            let x2 = x * x;
            let x3 = x2 * x;
            constant += (gp + gp + gp + gp) * x3 + gq * x3 * x;
            sum_s -= (gp + gp + gp) * x2 + gq * x3;
            sum_m += (gp + gp) * x + gq * x2;
            cross -= gp + gq * x;
            mm += gq;
        }
        [constant, sum_s, sum_m, cross, mm]
    }

    /// `P_0, P_1, Q_0, Q_1` at a point from `S_0, S_1, M_0, M_1` there.
    fn terminal(values: &[E], x: E) -> [E; 4] {
        let (s0, s1, m0, m1) = (values[0], values[1], values[2], values[3]);
        let two_x = x + x;
        [two_x - s0, two_x - s1, x * (x - s0) + m0, x * (x - s1) + m1]
    }
}

/// The parent layer: `p = p_0 q_1 + p_1 q_0`, `q = q_0 q_1`.
fn parent(child: &Layer) -> Layer {
    agg_simd::lookup::parent(child)
}

/// The layer above the leaves `(1, X - g)` of a stacked lane.
fn stack_bottom(stacked: &[i32], x: E) -> Layer {
    let half = stacked.len() / 2;
    Layer::from_fn(half, |y| {
        let a = x - E::from_i64(i64::from(stacked[y]));
        let b = x - E::from_i64(i64::from(stacked[y + half]));
        (a + b, crate::kernel::fm(a, b))
    })
}

/// The layer above the leaves `(m(t), X - t)` of a table.
fn table_bottom(m: &[i32], x: E) -> Layer {
    let half = m.len() / 2;
    Layer::from_fn(half, |y| {
        let a = x - E::from_u64(y as u64);
        let b = x - E::from_u64((y + half) as u64);
        (
            b.mul_i64(i64::from(m[y])) + a.mul_i64(i64::from(m[y + half])),
            crate::kernel::fm(a, b),
        )
    })
}

/// Layer `depth - 2` of a stack for every challenge in one pass over the leaves.
fn stack_upper_bottoms(stacked: &[i32], center: i32, xs: &[E]) -> Vec<Layer> {
    agg_simd::lookup::bottoms(stacked, center, &powers4(xs))
}

/// Layer `depth - 3` of a stack straight from the leaves.
fn stack_upper_parents(stacked: &[i32], center: i32, xs: &[E]) -> Vec<Layer> {
    agg_simd::lookup::bottom_parents(stacked, center, &powers4(xs))
}

/// `[X, X^2, X^3, X^4]` per challenge.
fn powers4(xs: &[E]) -> Vec<[E; 4]> {
    xs.iter()
        .map(|&x| {
            let x2 = x * x;
            let x3 = x2 * x;
            [x, x2, x3, x3 * x]
        })
        .collect()
}

/// Whether layer `depth - 2` of a centred stack is recomputed from the leaves in
/// its session instead of being stored (`AGG_LOOKUP_LEAF_UPPER=0` stores it).
fn leaf_upper() -> bool {
    std::env::var("AGG_LOOKUP_LEAF_UPPER").map_or(true, |v| v != "0")
}

/// The stored layers `0 .. depth - 1` of one tree.
#[derive(Debug, Default)]
struct FractionTree {
    layers: Vec<Layer>,
}

impl FractionTree {
    fn from_bottom(bottom: Layer) -> Self {
        let depth = bottom.len().trailing_zeros() as usize + 1;
        let mut layers = vec![Layer::default(); depth];
        layers[depth - 1] = bottom;
        for k in (0..depth - 1).rev() {
            layers[k] = parent(&layers[k + 1]);
        }
        Self { layers }
    }

    /// A tree of `depth` layers whose stored bottom is `bottom`: the layers below
    /// it are not stored.
    fn from_upper(bottom: Layer, depth: usize) -> Self {
        let top = bottom.len().trailing_zeros() as usize;
        let mut layers = vec![Layer::default(); depth];
        layers[top] = bottom;
        for k in (0..top).rev() {
            layers[k] = parent(&layers[k + 1]);
        }
        Self { layers }
    }

    fn root(&self) -> (E, E) {
        (self.layers[0].p.get(0), self.layers[0].q.get(0))
    }

    fn layer1(&self) -> [E; 4] {
        let layer = &self.layers[1];
        [
            layer.p.get(0),
            layer.p.get(1),
            layer.q.get(0),
            layer.q.get(1),
        ]
    }

    fn bytes(&self) -> usize {
        self.layers.iter().map(Layer::bytes).sum()
    }
}

/// What the prover's group of a session yields at its terminal.
enum Role {
    /// `P_0, P_1, Q_0, Q_1` of these trees, in order.
    Trees(Vec<usize>),
    /// `S_0, S_1, M_0, M_1` of a stack shared by these trees `(id, X')`.
    Sym(Vec<(usize, E)>),
    /// The two leaf-lane values of an ending stack or table.
    Leaves,
}

/// Trees of one dedicated layer group: two slots and four lanes per tree (sixteen trees, 64
/// lanes, within the fused kernel's limits).
const LAYER_GROUP_TREES: usize = 16;

/// A prover group on planes, or the generic kernel when the shape does not fit.
#[allow(clippy::large_enum_variant)]
enum SessionGroup<'a> {
    Generic(Group<'a>),
    Layer(PlaneLayerGroup<'a>),
    Leaf(PlaneLeafGroup<'a>),
}

impl SessionGroup<'_> {
    fn round_message(&mut self, round: usize) -> Vec<E> {
        match self {
            Self::Generic(g) => g.round_message(round),
            Self::Layer(g) => g.round_message(round),
            Self::Leaf(g) => g.round_message(round),
        }
    }

    fn bind(&mut self, round: usize, challenge: E) {
        match self {
            Self::Generic(g) => g.bind(round, challenge),
            Self::Layer(g) => g.bind(round, challenge),
            Self::Leaf(g) => g.bind(round, challenge),
        }
    }

    fn claim(&self) -> E {
        match self {
            Self::Generic(g) => g.claim(),
            Self::Layer(g) => g.claim(),
            Self::Leaf(g) => g.claim(),
        }
    }

    fn consistent(&self) -> bool {
        match self {
            Self::Generic(_) => true,
            Self::Layer(g) => g.terminal_consistent(),
            Self::Leaf(g) => g.terminal_consistent(),
        }
    }

    fn terminal_values(&self) -> Vec<E> {
        match self {
            Self::Generic(g) => g.terminal_values(),
            Self::Layer(g) => g.terminal_values(),
            Self::Leaf(g) => g.terminal_values(),
        }
    }

    fn recycle(self, pool: &mut agg_simd::buf::Pool) {
        match self {
            Self::Generic(_) => {}
            Self::Layer(g) => g.recycle(pool),
            Self::Leaf(g) => g.recycle(pool),
        }
    }
}

#[allow(clippy::too_many_lines)]
pub(super) fn prove_with(
    stacks: &[LookupStack<'_>],
    tables: &[Table],
    multiplicities: &[Vec<i32>],
    challenges: usize,
    fs: &mut impl LookupTranscript,
    check: bool,
    dedicated: bool,
) -> Result<LookupOutput, LookupError> {
    let start = std::time::Instant::now();
    let shape = LookupShape::of(stacks, tables, challenges);
    shape.validate()?;
    if multiplicities.len() != tables.len()
        || multiplicities.iter().zip(tables).any(|(m, t)| m.len() != t.len())
    {
        return Err(LookupError::Shape(
            "one multiplicity column per table entry".into(),
        ));
    }
    let index_halves: Vec<(Vec<i32>, Vec<i32>)> = tables.iter().map(|t| t.index_halves()).collect();
    let xs = fs.challenges(LookupSite::X, challenges)?;

    let build = std::time::Instant::now();
    // The integer path of session `depth - 1` per stack.
    let sym: Vec<Option<StackCentre>> = if dedicated {
        stacks
            .iter()
            .map(|stack| StackCentre::of(&stack.stacked))
            .collect()
    } else {
        stacks.iter().map(|_| None).collect()
    };
    let shifted = |j: usize, x: usize| xs[x] - E::from_i64(i64::from(stacks[j].shape.shift));
    let centred = |j: usize, x: usize| {
        shifted(j, x)
            - E::from_i64(i64::from(
                sym[j].as_ref().map_or(0, |centre| centre.center),
            ))
    };
    // Layer `depth - 2` of every challenge of a centred stack, or layer `depth -
    // 3` when layer `depth - 2` is recomputed in its session.
    let leaf_upper = dedicated && leaf_upper();
    let mut uppers: Vec<Vec<Layer>> = (0..stacks.len())
        .map(|j| {
            if let Some(centre) = &sym[j] {
                let xs_j: Vec<E> = (0..challenges).map(|x| centred(j, x)).collect();
                if leaf_upper {
                    stack_upper_parents(&stacks[j].stacked, centre.center, &xs_j)
                } else {
                    stack_upper_bottoms(&stacks[j].stacked, centre.center, &xs_j)
                }
            } else {
                Vec::new()
            }
        })
        .collect();
    let mut trees: Vec<FractionTree> = (0..shape.tree_count())
        .map(|id| {
            let (x, kind) = shape.tree(id);
            match kind {
                TreeKind::Stack(j) if sym[j].is_some() => {
                    FractionTree::from_upper(std::mem::take(&mut uppers[j][x]), stacks[j].shape.depth())
                }
                TreeKind::Stack(j) => {
                    FractionTree::from_bottom(stack_bottom(&stacks[j].stacked, shifted(j, x)))
                }
                TreeKind::Table(t) => {
                    FractionTree::from_bottom(table_bottom(&multiplicities[t], xs[x]))
                }
            }
        })
        .collect();
    drop(uppers);
    let mut stats = ProveStats {
        trees: build.elapsed().as_secs_f64(),
        layer_bytes: trees.iter().map(FractionTree::bytes).sum(),
        ..ProveStats::default()
    };
    debug_assert!(trees.iter().all(|t| t.root().1 == root_of(&t.layer1()).1));

    let layer1: Vec<[E; 4]> = trees.iter().map(FractionTree::layer1).collect();
    if check {
        check_roots(&shape, &layer1)?;
    }
    fs.absorb(LookupSite::Layer1, &layer1.concat())?;
    let mu = one(fs, LookupSite::Mu(1))?;
    let mut point = vec![mu];
    let mut claims: Vec<(E, E)> = layer1
        .iter()
        .map(|&[p0, p1, q0, q1]| (line(p0, p1, mu), line(q0, q1, mu)))
        .collect();
    let mut stack_claims: Vec<Option<(Vec<E>, E)>> = vec![None; stacks.len()];
    let mut table_claims: Vec<Option<(Vec<E>, E)>> = vec![None; tables.len()];
    let mut sessions = Vec::new();
    // Table buffers of the plane groups, reused across the sessions.
    let mut pool = agg_simd::buf::Pool::default();

    for s in 2..=shape.max_depth() {
        let session_start = std::time::Instant::now();
        let active: Vec<usize> = (0..shape.tree_count())
            .filter(|&id| shape.tree_depth(id) >= s)
            .collect();
        let gammas = fs.challenges(LookupSite::Gamma(s), 2 * active.len())?;
        let mut claim = E::zero();
        for (i, &id) in active.iter().enumerate() {
            claim += gammas[2 * i] * claims[id].0 + gammas[2 * i + 1] * claims[id].1;
        }
        let plan = session_plan(&shape, s, &xs, &active, &gammas);
        let half = 1_usize << (s - 1);
        let gamma_of = |id: usize| {
            let i = active.binary_search(&id).expect("an active tree");
            (gammas[2 * i], gammas[2 * i + 1])
        };
        let mut groups: Vec<(Role, SessionGroup<'_>)> = Vec::with_capacity(plan.groups.len());
        for (kind, instances) in &plan.groups {
            match kind {
                GroupKind::Layer(ids) => {
                    let mut stored = Vec::new();
                    let mut by_stack: Vec<Vec<(usize, E, E, E)>> = vec![Vec::new(); stacks.len()];
                    let mut from_leaves: Vec<Vec<(usize, E, E, E)>> =
                        vec![Vec::new(); stacks.len()];
                    for &id in ids {
                        let (x, tree_kind) = shape.tree(id);
                        match tree_kind {
                            TreeKind::Stack(j)
                                if sym[j].is_some() && stacks[j].shape.depth() == s + 1 =>
                            {
                                let (gp, gq) = gamma_of(id);
                                by_stack[j].push((id, centred(j, x), gp, gq));
                            }
                            TreeKind::Stack(j)
                                if leaf_upper
                                    && sym[j].is_some()
                                    && stacks[j].shape.depth() == s + 2 =>
                            {
                                let (gp, gq) = gamma_of(id);
                                from_leaves[j].push((id, centred(j, x), gp, gq));
                            }
                            _ => stored.push(id),
                        }
                    }
                    for (j, list) in from_leaves.iter().enumerate().filter(|(_, l)| !l.is_empty())
                    {
                        let centre = sym[j].as_ref().expect("an integer-path stack").center;
                        let xs_j: Vec<E> = list.iter().map(|&(_, x, _, _)| x).collect();
                        let gammas_j: Vec<(E, E)> =
                            list.iter().map(|&(_, _, gp, gq)| (gp, gq)).collect();
                        let rows = agg_simd::lookup::UpperRows::new(
                            &stacks[j].stacked,
                            centre,
                            &powers4(&xs_j),
                        );
                        // The centre gate (depth >= 5, spread <= 2^16) guarantees the group.
                        let group =
                            PlaneLayerGroup::from_leaves(rows, &gammas_j, &point, &mut pool)
                                .expect("a centred stack of depth s + 2 fits the leaf-computed layer group");
                        groups.push((
                            Role::Trees(list.iter().map(|&(id, ..)| id).collect()),
                            SessionGroup::Layer(group),
                        ));
                    }
                    for (j, list) in by_stack.iter().enumerate().filter(|(_, l)| !l.is_empty()) {
                        let centre = sym[j].as_ref().expect("an integer-path stack").center;
                        let trees_x: Vec<(E, E, E)> =
                            list.iter().map(|&(_, x, gp, gq)| (x, gp, gq)).collect();
                        let coefficients = StackCentre::coefficients(&trees_x);
                        // The centre gate (depth >= 5, spread <= 2^16) guarantees the group.
                        let group = PlaneLeafGroup::symmetric(
                            &stacks[j].stacked,
                            centre,
                            coefficients,
                            &point,
                            &mut pool,
                        )
                        .expect("a centred stack of depth s + 1 fits the leaf group");
                        groups.push((
                            Role::Sym(list.iter().map(|&(id, x, _, _)| (id, x)).collect()),
                            SessionGroup::Leaf(group),
                        ));
                    }
                    if stored.is_empty() {
                        continue;
                    }
                    // The dedicated layer group holds at most LAYER_GROUP_TREES trees (the
                    // fused kernel's slot and lane limits); more trees (ML-DSA-87's eight
                    // stacks) take one group per chunk, whose messages add up.
                    let chunks: Vec<Vec<usize>> = if dedicated {
                        stored.chunks(LAYER_GROUP_TREES).map(<[usize]>::to_vec).collect()
                    } else {
                        vec![stored]
                    };
                    for stored in chunks {
                        let dedicated_group = if dedicated {
                            let layer_trees = stored
                                .iter()
                                .map(|&id| {
                                    let layer = &trees[id].layers[s];
                                    let (gamma_p, gamma_q) = gamma_of(id);
                                    PlaneLayerTree {
                                        p: &layer.p,
                                        q: &layer.q,
                                        gamma_p,
                                        gamma_q,
                                    }
                                })
                                .collect();
                            PlaneLayerGroup::new(layer_trees, &point, &mut pool)
                        } else {
                            None
                        };
                        let group = match dedicated_group {
                            Some(group) => SessionGroup::Layer(group),
                            None => {
                                let lanes: Vec<Lane<'_>> = stored
                                    .iter()
                                    .flat_map(|&id| {
                                        let layer = &trees[id].layers[s];
                                        let (p0, p1) = layer.p.planes();
                                        let (q0, q1) = layer.q.planes();
                                        [(p0, p1), (q0, q1)].into_iter().flat_map(|(a, b)| {
                                            [
                                                Lane::Source(Source::PlaneSlice((
                                                    &a[..half],
                                                    &b[..half],
                                                ))),
                                                Lane::Source(Source::PlaneSlice((
                                                    &a[half..],
                                                    &b[half..],
                                                ))),
                                            ]
                                        })
                                    })
                                    .collect();
                                let instances = stored
                                    .iter()
                                    .enumerate()
                                    .flat_map(|(i, &id)| layer_instances(i, gamma_of(id)))
                                    .collect();
                                SessionGroup::Generic(
                                    Group::new(lanes, instances, Some(&point), s - 1, DEGREE)
                                        .map_err(LookupError::Kernel)?,
                                )
                            }
                        };
                        groups.push((Role::Trees(stored), group));
                    }
                }
                GroupKind::Stack(j) => {
                    let (v0, v1) = stacks[*j].stacked.split_at(half);
                    let pair = if dedicated && instances.len() == 3 {
                        let c = [instances[0].gamma, instances[1].gamma, instances[2].gamma];
                        PlaneLeafGroup::bottom(v0, v1, c, &point, &mut pool)
                    } else {
                        None
                    };
                    let group = match pair {
                        Some(group) => SessionGroup::Leaf(group),
                        None => {
                            let lanes = vec![
                                Lane::Source(Source::Ints(v0)),
                                Lane::Source(Source::Ints(v1)),
                            ];
                            SessionGroup::Generic(
                                Group::new(lanes, instances.clone(), Some(&point), s - 1, DEGREE)
                                    .map_err(LookupError::Kernel)?,
                            )
                        }
                    };
                    groups.push((Role::Leaves, group));
                }
                GroupKind::Table(t) => {
                    let (m0, m1) = multiplicities[*t].split_at(half);
                    let (t0, t1) = &index_halves[*t];
                    let lanes = vec![
                        Lane::Source(Source::Ints(m0)),
                        Lane::Source(Source::Ints(m1)),
                        Lane::Source(Source::Ints(t0)),
                        Lane::Source(Source::Ints(t1)),
                    ];
                    let group = Group::new(lanes, instances.clone(), Some(&point), s - 1, DEGREE)
                        .map_err(LookupError::Kernel)?;
                    groups.push((Role::Leaves, SessionGroup::Generic(group)));
                }
            }
        }
        let mut rounds = Vec::with_capacity(s - 1);
        let mut rho = Vec::with_capacity(s - 1);
        for round in 0..s - 1 {
            let concurrent = groups.len() > 1 && s - 1 - round >= CONCURRENT_GROUP_ROUNDS_MIN_VARS;
            let parts: Vec<Vec<E>> = if concurrent {
                groups
                    .par_iter_mut()
                    .map(|(_, group)| group.round_message(round))
                    .collect()
            } else {
                groups
                    .iter_mut()
                    .map(|(_, group)| group.round_message(round))
                    .collect()
            };
            let mut message = vec![E::zero(); DEGREE];
            for part in parts {
                for (slot, value) in message.iter_mut().zip(part) {
                    *slot += value;
                }
            }
            fs.absorb(LookupSite::Round(s, round), &message)?;
            let challenge = one(fs, LookupSite::RoundChallenge(s, round))?;
            claim = evaluate_compressed(&message, claim, challenge);
            if concurrent {
                groups
                    .par_iter_mut()
                    .for_each(|(_, group)| group.bind(round, challenge));
            } else {
                for (_, group) in &mut groups {
                    group.bind(round, challenge);
                }
            }
            rounds.push(message);
            rho.push(challenge);
        }
        if check
            && (groups.iter().fold(E::zero(), |acc, (_, g)| acc + g.claim()) != claim
                || groups.iter().any(|(_, g)| !g.consistent()))
        {
            return Err(LookupError::Terminal(s));
        }
        let mut by_tree: Vec<Option<[E; 4]>> = vec![None; shape.tree_count()];
        let mut leaves = Vec::new();
        for (role, group) in &groups {
            let values = group.terminal_values();
            match role {
                Role::Trees(ids) => {
                    for (&id, chunk) in ids.iter().zip(values.as_chunks::<4>().0) {
                        by_tree[id] = Some(*chunk);
                    }
                }
                Role::Sym(list) => {
                    for &(id, x) in list {
                        by_tree[id] = Some(StackCentre::terminal(&values, x));
                    }
                }
                Role::Leaves => leaves.push([values[0], values[1]]),
            }
        }
        for (_, group) in groups {
            group.recycle(&mut pool);
        }
        let terminals: Vec<[E; 4]> = plan
            .continuing
            .iter()
            .map(|&id| by_tree[id].expect("every continuing tree has terminal values"))
            .collect();
        fs.absorb(LookupSite::Terminals(s), &terminals.concat())?;
        fs.absorb(LookupSite::Leaves(s), &leaves.concat())?;
        let mu = one(fs, LookupSite::Mu(s))?;
        for (&id, &[p0, p1, q0, q1]) in plan.continuing.iter().zip(&terminals) {
            claims[id] = (line(p0, p1, mu), line(q0, q1, mu));
        }
        rho.push(mu);
        for (&j, leaf) in plan.ending_stacks.iter().zip(&leaves) {
            stack_claims[j] = Some((rho.clone(), line(leaf[0], leaf[1], mu)));
        }
        for (&t, leaf) in plan
            .ending_tables
            .iter()
            .zip(&leaves[plan.ending_stacks.len()..])
        {
            table_claims[t] = Some((rho.clone(), line(leaf[0], leaf[1], mu)));
        }
        point = rho;
        let done: Vec<Layer> = trees
            .iter_mut()
            .filter_map(|tree| tree.layers.get_mut(s).map(std::mem::take))
            .collect();
        agg_pcs_config::sched::drop_in_background("lookup layer", done);
        sessions.push(SessionProof {
            rounds,
            terminals,
            leaves,
        });
        stats.sessions.push(session_start.elapsed().as_secs_f64());
    }
    let claims = super::LookupClaims {
        stacks: stack_claims
            .into_iter()
            .map(|c| c.expect("every stack ends"))
            .collect(),
        multiplicities: table_claims
            .into_iter()
            .map(|c| c.expect("every table ends"))
            .collect(),
    };
    stats.total = start.elapsed().as_secs_f64();
    Ok(LookupOutput {
        proof: super::LookupProof {
            layer1,
            sessions,
        },
        claims,
        stats,
    })
}
