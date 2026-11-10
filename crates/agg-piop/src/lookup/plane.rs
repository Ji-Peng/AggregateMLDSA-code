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
    layer_instances, root_of, Group, Lane, LookupError, ProverInputs, ProverLayout, Role,
    SessionCx, SessionOps, Source, TreeKind, DEGREE, SYM_SPREAD,
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

/// Trees of one dedicated layer group: two slots and four lanes per tree (sixteen trees, 64
/// lanes, within the fused kernel's limits).
const LAYER_GROUP_TREES: usize = 16;

/// A prover group on planes, or the generic kernel when the shape does not fit.
#[allow(clippy::large_enum_variant)]
pub(super) enum SessionGroup<'a> {
    Generic(Group<'a>),
    Layer(PlaneLayerGroup<'a>),
    Leaf(PlaneLeafGroup<'a>),
}

impl SessionOps for SessionGroup<'_> {
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
}

impl SessionGroup<'_> {
    fn recycle(self, pool: &mut agg_simd::buf::Pool) {
        match self {
            Self::Generic(_) => {}
            Self::Layer(g) => g.recycle(pool),
            Self::Leaf(g) => g.recycle(pool),
        }
    }
}

/// Plane fraction trees and the centres of the integer-path stacks.
pub(super) struct PlaneLayout {
    trees: Vec<FractionTree>,
    /// The integer path of session `depth - 1` per stack.
    sym: Vec<Option<StackCentre>>,
}

impl PlaneLayout {
    fn center(&self, j: usize) -> i32 {
        self.sym[j].as_ref().map_or(0, |centre| centre.center)
    }
}

impl ProverLayout for PlaneLayout {
    type Group<'g> = SessionGroup<'g>;
    /// Table buffers of the plane groups, reused across the sessions.
    type Scratch = agg_simd::buf::Pool;

    fn build(inputs: &ProverInputs<'_>) -> Self {
        let (stacks, shape, xs) = (inputs.stacks, inputs.shape, inputs.xs);
        let challenges = xs.len();
        // The integer path of session `depth - 1` per stack.
        let sym: Vec<Option<StackCentre>> = if inputs.dedicated {
            stacks
                .iter()
                .map(|stack| StackCentre::of(&stack.stacked))
                .collect()
        } else {
            stacks.iter().map(|_| None).collect()
        };
        // Layer `depth - 3` of every challenge of a centred stack; layer `depth - 2` is
        // recomputed from the leaves in its session instead of being stored.
        let mut uppers: Vec<Vec<Layer>> = (0..stacks.len())
            .map(|j| {
                if let Some(centre) = &sym[j] {
                    let xs_j: Vec<E> = (0..challenges)
                        .map(|x| inputs.centred(j, x, centre.center))
                        .collect();
                    stack_upper_parents(&stacks[j].stacked, centre.center, &xs_j)
                } else {
                    Vec::new()
                }
            })
            .collect();
        let trees: Vec<FractionTree> = (0..shape.tree_count())
            .map(|id| {
                let (x, kind) = shape.tree(id);
                match kind {
                    TreeKind::Stack(j) if sym[j].is_some() => FractionTree::from_upper(
                        std::mem::take(&mut uppers[j][x]),
                        stacks[j].shape.depth(),
                    ),
                    TreeKind::Stack(j) => FractionTree::from_bottom(stack_bottom(
                        &stacks[j].stacked,
                        inputs.shifted(j, x),
                    )),
                    TreeKind::Table(t) => {
                        FractionTree::from_bottom(table_bottom(&inputs.multiplicities[t], xs[x]))
                    }
                }
            })
            .collect();
        drop(uppers);
        debug_assert!(trees.iter().all(|t| t.root().1 == root_of(&t.layer1()).1));
        Self { trees, sym }
    }

    fn scratch() -> Self::Scratch {
        agg_simd::buf::Pool::default()
    }

    fn layer1(&self) -> Vec<[E; 4]> {
        self.trees.iter().map(FractionTree::layer1).collect()
    }

    fn layer_bytes(&self) -> usize {
        self.trees.iter().map(FractionTree::bytes).sum()
    }

    #[allow(clippy::too_many_lines)]
    fn layer_groups<'g>(
        &'g self,
        inputs: &'g ProverInputs<'g>,
        cx: &SessionCx<'_>,
        ids: &[usize],
        pool: &mut Self::Scratch,
        groups: &mut Vec<(Role, SessionGroup<'g>)>,
    ) -> Result<(), LookupError> {
        let (stacks, s, half, point) = (inputs.stacks, cx.s, cx.half, cx.point);
        let sym = &self.sym;
        let mut stored = Vec::new();
        let mut by_stack: Vec<Vec<(usize, E, E, E)>> = vec![Vec::new(); stacks.len()];
        let mut from_leaves: Vec<Vec<(usize, E, E, E)>> = vec![Vec::new(); stacks.len()];
        for &id in ids {
            let (x, tree_kind) = inputs.shape.tree(id);
            match tree_kind {
                TreeKind::Stack(j) if sym[j].is_some() && stacks[j].shape.depth() == s + 1 => {
                    let (gp, gq) = cx.gamma_of(id);
                    by_stack[j].push((id, inputs.centred(j, x, self.center(j)), gp, gq));
                }
                TreeKind::Stack(j) if sym[j].is_some() && stacks[j].shape.depth() == s + 2 => {
                    let (gp, gq) = cx.gamma_of(id);
                    from_leaves[j].push((id, inputs.centred(j, x, self.center(j)), gp, gq));
                }
                _ => stored.push(id),
            }
        }
        for (j, list) in from_leaves.iter().enumerate().filter(|(_, l)| !l.is_empty()) {
            let centre = sym[j].as_ref().expect("an integer-path stack").center;
            let xs_j: Vec<E> = list.iter().map(|&(_, x, _, _)| x).collect();
            let gammas_j: Vec<(E, E)> = list.iter().map(|&(_, _, gp, gq)| (gp, gq)).collect();
            let rows =
                agg_simd::lookup::UpperRows::new(&stacks[j].stacked, centre, &powers4(&xs_j));
            // The centre gate (depth >= 5, spread <= 2^16) guarantees the group.
            let group = PlaneLayerGroup::from_leaves(rows, &gammas_j, point, pool)
                .expect("a centred stack of depth s + 2 fits the leaf-computed layer group");
            groups.push((
                Role::Trees(list.iter().map(|&(id, ..)| id).collect()),
                SessionGroup::Layer(group),
            ));
        }
        for (j, list) in by_stack.iter().enumerate().filter(|(_, l)| !l.is_empty()) {
            let centre = sym[j].as_ref().expect("an integer-path stack").center;
            let trees_x: Vec<(E, E, E)> = list.iter().map(|&(_, x, gp, gq)| (x, gp, gq)).collect();
            let coefficients = StackCentre::coefficients(&trees_x);
            // The centre gate (depth >= 5, spread <= 2^16) guarantees the group.
            let group =
                PlaneLeafGroup::symmetric(&stacks[j].stacked, centre, coefficients, point, pool)
                    .expect("a centred stack of depth s + 1 fits the leaf group");
            groups.push((
                Role::Sym(list.iter().map(|&(id, x, _, _)| (id, x)).collect()),
                SessionGroup::Leaf(group),
            ));
        }
        if stored.is_empty() {
            return Ok(());
        }
        // The dedicated layer group holds at most LAYER_GROUP_TREES trees (the fused
        // kernel's slot and lane limits); more trees (ML-DSA-87's eight stacks) take one
        // group per chunk, whose messages add up.
        let chunks: Vec<Vec<usize>> = if inputs.dedicated {
            stored.chunks(LAYER_GROUP_TREES).map(<[usize]>::to_vec).collect()
        } else {
            vec![stored]
        };
        for stored in chunks {
            let dedicated_group = if inputs.dedicated {
                let layer_trees = stored
                    .iter()
                    .map(|&id| {
                        let layer = &self.trees[id].layers[s];
                        let (gamma_p, gamma_q) = cx.gamma_of(id);
                        PlaneLayerTree {
                            p: &layer.p,
                            q: &layer.q,
                            gamma_p,
                            gamma_q,
                        }
                    })
                    .collect();
                PlaneLayerGroup::new(layer_trees, point, pool)
            } else {
                None
            };
            let group = match dedicated_group {
                Some(group) => SessionGroup::Layer(group),
                None => {
                    let lanes: Vec<Lane<'_>> = stored
                        .iter()
                        .flat_map(|&id| {
                            let layer = &self.trees[id].layers[s];
                            let (p0, p1) = layer.p.planes();
                            let (q0, q1) = layer.q.planes();
                            [(p0, p1), (q0, q1)].into_iter().flat_map(|(a, b)| {
                                [
                                    Lane::Source(Source::PlaneSlice((&a[..half], &b[..half]))),
                                    Lane::Source(Source::PlaneSlice((&a[half..], &b[half..]))),
                                ]
                            })
                        })
                        .collect();
                    let instances = stored
                        .iter()
                        .enumerate()
                        .flat_map(|(i, &id)| layer_instances(i, cx.gamma_of(id)))
                        .collect();
                    SessionGroup::Generic(
                        Group::new(lanes, instances, Some(point), s - 1, DEGREE)
                            .map_err(LookupError::Kernel)?,
                    )
                }
            };
            groups.push((Role::Trees(stored), group));
        }
        Ok(())
    }

    fn stack_pair<'g>(
        [v0, v1]: [&'g [i32]; 2],
        coefficients: [E; 3],
        point: &[E],
        pool: &mut Self::Scratch,
    ) -> Option<SessionGroup<'g>>
    where
        Self: 'g,
    {
        PlaneLeafGroup::bottom(v0, v1, coefficients, point, pool).map(SessionGroup::Leaf)
    }

    fn generic<'g>(group: Group<'g>) -> SessionGroup<'g>
    where
        Self: 'g,
    {
        SessionGroup::Generic(group)
    }

    fn ready<'g>(_pool: &Self::Scratch, _groups: &mut [(Role, SessionGroup<'g>)])
    where
        Self: 'g,
    {
    }

    fn release<'g>(pool: &mut Self::Scratch, groups: Vec<(Role, SessionGroup<'g>)>)
    where
        Self: 'g,
    {
        for (_, group) in groups {
            group.recycle(pool);
        }
    }

    fn drop_layer(&mut self, s: usize) {
        let done: Vec<Layer> = self
            .trees
            .iter_mut()
            .filter_map(|tree| tree.layers.get_mut(s).map(std::mem::take))
            .collect();
        agg_pcs_config::sched::drop_in_background("lookup layer", done);
    }
}
