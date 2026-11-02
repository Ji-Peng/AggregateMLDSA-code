//! Reference authoring of the chain gate in Flock's Boolean circuit DSL
//! (`flock_core::circuit::boolean`): theta, rho, pi, iota, the absorptions and
//! the chaining as virtual `xor` nodes, the chi AND outputs as `define_and`
//! columns, the outputs as `define_linear` columns. Flock lowers it to sparse
//! matrices and to a structural walk; the tests check that [`ChainGate`]'s
//! witness, row values and lincheck fold agree with both, position by position.
//!
//! Only reduced-round instances are lowered this way: the DSL builder stores
//! the normalized support of every XOR node eagerly, and with implicit states
//! those supports grow towards half of all committed columns (measured below),
//! so the 24-round gate is out of reach of the DSL lowering.

use flock_core::circuit::boolean::CircuitBuilder;
use flock_core::circuit::boolean::{
    ColumnRole, ColumnSchema, ColumnVisitor, CompiledColumns, LinearExpr, Var,
};

use crate::gate::{Absorb, ChainGate, ChainSpec};
use crate::keccak::{self, ROUND_CONSTANTS, STATE_BITS};

pub(crate) struct ChainSchema(pub ChainSpec);

/// Committed chi bits of round `r` of permutation `m`: 1,600, or the output lanes'
/// bits for the pruned last round ([`ChainGate::round_lanes`]).
fn round_bits(gate: &ChainGate, m: usize, r: usize) -> usize {
    64 * gate.round_lanes(m, r)
}

pub(crate) struct ChainCols<T> {
    pub inputs: Vec<Vec<T>>,
    /// Per `(m, r)`, `m * n_rounds + r`: the committed chi bits (1,600, fewer in the
    /// pruned last round).
    pub t: Vec<Vec<T>>,
    pub out: Vec<T>,
}

impl ColumnSchema for ChainSchema {
    type Cols<T> = ChainCols<T>;
    fn columns<V: ColumnVisitor>(&self, v: &mut V) -> ChainCols<V::Value> {
        let spec = &self.0;
        let inputs = spec
            .input_fields
            .iter()
            .map(|f| v.bits(&format!("in.{}", f.name), ColumnRole::Input, f.bits, 128))
            .collect();
        let gate = ChainGate::new(spec.clone());
        let mut t = Vec::new();
        for m in 0..spec.n_perms() {
            for r in 0..spec.n_rounds {
                let bits = round_bits(&gate, m, r);
                t.push(v.bits(&format!("t.{m}.{r}"), ColumnRole::Witness, bits, 1));
            }
        }
        let out = v.bits("out", ColumnRole::Output, spec.outputs.len(), 128);
        ChainCols { inputs, t, out }
    }
}

fn eval(spec: &ChainSpec, b: &mut CircuitBuilder, cols: &ChainCols<Var>) {
    let gate = ChainGate::new(spec.clone());
    let one: LinearExpr = b.one().into();
    let mut field_of = Vec::new();
    for (f, field) in spec.input_fields.iter().enumerate() {
        field_of.extend((0..field.bits).map(|k| (f, k)));
    }
    let mut st: Vec<LinearExpr> = vec![b.zero(); STATE_BITS];
    for m in 0..spec.n_perms() {
        for &(s, src) in &spec.absorbs[m] {
            let e: LinearExpr = match src {
                Absorb::One => one,
                Absorb::Input(i) => {
                    let (f, k) = field_of[i as usize];
                    cols.inputs[f][k].into()
                }
            };
            st[s as usize] = b.xor2(st[s as usize], e);
        }
        for r in 0..spec.n_rounds {
            // theta
            let c: Vec<LinearExpr> = (0..5 * 64)
                .map(|xz| {
                    let (x, z) = (xz / 64, xz % 64);
                    b.xor((0..5).map(|y| st[keccak::bit(x, y, z)]))
                })
                .collect();
            let d: Vec<LinearExpr> = (0..5 * 64)
                .map(|xz| {
                    let (x, z) = (xz / 64, xz % 64);
                    b.xor2(
                        c[64 * ((x + 4) % 5) + z],
                        c[64 * ((x + 1) % 5) + (z + 63) % 64],
                    )
                })
                .collect();
            let a1: Vec<LinearExpr> = (0..STATE_BITS)
                .map(|s| {
                    let (x, z) = ((s / 64) % 5, s % 64);
                    b.xor2(st[s], d[64 * x + z])
                })
                .collect();
            // rho, pi: index maps only
            let mut bb = vec![b.zero(); STATE_BITS];
            for y in 0..5 {
                for x in 0..5 {
                    let a = (x + 3 * y) % 5;
                    let rot = keccak::RHO_OFFSETS[a][x] as usize;
                    for z in 0..64 {
                        bb[keccak::bit(x, y, z)] = a1[keccak::bit(a, x, (z + 64 - rot) % 64)];
                    }
                }
            }
            // chi (AND columns of the committed lanes) and iota; an uncommitted lane
            // of the pruned last round keeps its theta-rho-pi value (no output reads it)
            let tcols = &cols.t[m * spec.n_rounds + r];
            let kept = round_bits(&gate, m, r);
            for y in 0..5 {
                for x in 0..5 {
                    for z in 0..64 {
                        let s = keccak::bit(x, y, z);
                        if s >= kept {
                            st[s] = bb[s];
                            continue;
                        }
                        let lhs = b.xor2(bb[keccak::bit((x + 1) % 5, y, z)], one);
                        b.define_and(tcols[s], lhs, bb[keccak::bit((x + 2) % 5, y, z)]);
                        let mut next = b.xor2(bb[s], tcols[s]);
                        if x == 0 && y == 0 && (ROUND_CONSTANTS[r] >> z) & 1 == 1 {
                            next = b.xor2(next, one);
                        }
                        st[s] = next;
                    }
                }
            }
        }
    }
    for (j, &o) in spec.outputs.iter().enumerate() {
        b.define_linear(cols.out[j], st[o as usize]);
    }
}

/// Compile the DSL reference of `spec`.
pub(crate) fn compile(spec: &ChainSpec) -> CompiledColumns<ChainSchema> {
    let s = spec.clone();
    CircuitBuilder::compile(ChainSchema(spec.clone()), move |b, cols| eval(&s, b, cols))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gate::tests::{mini_chain, mini_chain_pruned};
    use flock_core::{lincheck::LincheckCircuit, test_rng::Rng};
    use flock_field::F128;

    /// Witness, row values and lincheck fold of [`ChainGate`] agree with Flock's
    /// DSL lowering (walk adapter and sparse matrices), under the column map
    /// `gate column -> DSL physical position`.
    fn cross_check(spec: ChainSpec, seed: u64) {
        let gate = ChainGate::new(spec.clone());
        let lay = gate.layout().clone();
        let compiled = compile(&spec);
        let circuit = compiled.circuit();
        let layout = circuit.layout().expect("valid layout");
        let k_log = lay
            .k_log
            .max(layout.useful_bits().next_power_of_two().trailing_zeros() as usize);
        let cols = compiled.columns();
        // gate column -> DSL value position
        let mut map: Vec<(usize, usize)> = Vec::new();
        let pos = |v| layout.value_position(v).expect("placed value");
        map.push((lay.const_col, pos(circuit.one())));
        let mut i = 0;
        for f in &cols.inputs {
            for &v in f {
                map.push((lay.input_col(i), pos(v)));
                i += 1;
            }
        }
        for m in 0..spec.n_perms() {
            for r in 0..spec.n_rounds {
                for s in 0..64 * gate.round_lanes(m, r) {
                    map.push((
                        lay.t_col(spec.n_rounds, m, r, s),
                        pos(cols.t[m * spec.n_rounds + r][s]),
                    ));
                }
            }
        }
        for (j, &v) in cols.out.iter().enumerate() {
            map.push((lay.output_base + j, pos(v)));
        }
        assert_eq!(map.len(), gate.cost().rows);
        assert_eq!(
            circuit.row_count(),
            gate.cost().rows,
            "one DSL row per committed column"
        );

        // Forward trace on random inputs.
        let mut rng = Rng::new(seed);
        let packed: Vec<u64> = (0..gate.input_words()).map(|_| rng.next_u64()).collect();
        let mut flat = Vec::new();
        for i in 0..spec.n_inputs() {
            flat.push((packed[i / 64] >> (i % 64)) & 1 == 1);
        }
        let dsl_inputs = compiled.inputs(|c| {
            let mut it = flat.iter();
            for f in c.inputs {
                for cell in f {
                    *cell = *it.next().expect("input count");
                }
            }
        });
        let plan = circuit.walk_plan_with_layout(&layout).expect("walk plan");
        let trace = plan.forward(&dsl_inputs, k_log).expect("forward walk");
        let w = lay.useful_words();
        let (mut z, mut a, mut b) = (vec![0u64; w], vec![0u64; w], vec![0u64; w]);
        gate.instance_rows(&packed, &mut z, &mut a, &mut b);
        let bit = |v: &[u64], c: usize| (v[c / 64] >> (c % 64)) & 1 == 1;
        for &(mine, theirs) in &map {
            assert_eq!(bit(&z, mine), trace.z[theirs], "z at gate column {mine}");
            assert_eq!(bit(&a, mine), trace.a_z[theirs], "a at gate column {mine}");
            assert_eq!(bit(&b, mine), trace.b_z[theirs], "b at gate column {mine}");
        }

        // Lincheck fold: random eq on the mapped rows.
        let k_dsl = 1usize << k_log;
        let mut eq_mine = vec![F128::ZERO; 1 << lay.k_log];
        let mut eq_dsl = vec![F128::ZERO; k_dsl];
        for &(mine, theirs) in &map {
            let e = F128::new(rng.next_u64(), rng.next_u64());
            eq_mine[mine] = e;
            eq_dsl[theirs] = e;
        }
        let alpha = F128::new(rng.next_u64(), rng.next_u64());
        let comb_mine = gate.fold_alpha_batched(alpha, &eq_mine);
        let walk = plan.lincheck_circuit(k_log).expect("walk adapter");
        let comb_walk = walk.fold_alpha_batched(alpha, &eq_dsl);
        let r1cs = circuit
            .to_block_r1cs_with_layout(k_log, 6, 3, &layout)
            .expect("sparse lowering");
        let comb_sparse = r1cs
            .sparse_lincheck_circuit()
            .fold_alpha_batched(alpha, &eq_dsl);
        assert_eq!(comb_walk, comb_sparse);
        assert_eq!(walk.const_pin_col(), Some(pos(circuit.one())));
        for &(mine, theirs) in &map {
            assert_eq!(
                comb_mine[mine], comb_walk[theirs],
                "fold at gate column {mine}"
            );
        }
        let mapped: std::collections::HashSet<usize> = map.iter().map(|&(_, t)| t).collect();
        for (p, &c) in comb_walk.iter().enumerate() {
            if !mapped.contains(&p) {
                assert_eq!(c, F128::ZERO);
            }
        }
    }

    #[test]
    fn gate_agrees_with_flock_dsl_lowering_permutation() {
        cross_check(ChainSpec::permutation_rounds(3), 41);
    }

    #[test]
    fn gate_agrees_with_flock_dsl_lowering_chain() {
        cross_check(mini_chain(1), 42);
    }

    /// The last round keeps lanes 0..2 only: the pruning the sponge gate applies.
    #[test]
    fn gate_agrees_with_flock_dsl_lowering_pruned_chain() {
        cross_check(mini_chain_pruned(1), 43);
        cross_check(mini_chain_pruned(2), 44);
    }

    /// Normalized support terms of the DSL authoring per round count: the
    /// growth that rules out lowering the 24-round gate through the DSL.
    #[test]
    fn dsl_support_growth() {
        let mut prev = 0;
        for rounds in 1..=4 {
            let compiled = compile(&ChainSpec::permutation_rounds(rounds));
            let terms = compiled.circuit().normalized_support_terms();
            eprintln!("keccak-f DSL, {rounds} rounds: {terms} normalized support terms");
            assert!(terms > 2 * prev);
            prev = terms;
        }
    }
}
