//! The `{0, 1}` projection of `v` (`docs/spec.md` and `docs/soundness.md`, the projection
//! definition and bound):
//! the real attack on an unbounded quotient, the range and binding of the outputs `k`, the
//! shared claim, the digits of `v` that are no longer range-checked, and the honest abort.

use crate::common;

use agg_piop::manifest::{
    Column, DigitValue, GroupRole, Relation, LOOKUP_SHIFT, PROJ_ROWS, U16_DIGITS, V16_DIGITS,
};
use agg_piop::projection::{outputs, ProjectionTables};
use agg_piop::source_tables::{BridgeGroupTables, MainTables, RecordValues, SourceTableError, SourceTables};
use agg_piop::stage::StageError;
use agg_piop::verifier::VerifyError;
use agg_piop::OpeningField as E;
use agg_witness::padding::main_index;
use agg_witness::params::{GAMMA2, POSITIONS, Q, W1_BITS};
use crate::common::{lenient_multiplicities, prove_tables, verify_tables, Batch};
use jolt_field::Ring;

const P: i128 = (1 << 64) - 59;

fn on_large_stack(f: impl FnOnce() + Send + 'static) {
    std::thread::Builder::new()
        .stack_size(256 << 20)
        .spawn(f)
        .expect("spawn")
        .join()
        .expect("test thread");
}

/// Where the verifier rejected: `0` for the lookup, `s` for stage `s`; `None` accepts.
fn rejecting_place(batch: &Batch, main: &MainTables, bridge: &BridgeGroupTables) -> Option<usize> {
    let tables = SourceTables {
        manifest: &batch.manifest,
        main,
        bridge,
    };
    let (proof, _) = prove_tables(batch, tables, true).expect("the unchecked prover runs");
    match verify_tables(batch, bridge.public_y(), &proof) {
        Ok(_) => None,
        Err(VerifyError::Lookup(_)) => Some(0),
        Err(VerifyError::Stage(StageError::Terminal(stage))) => Some(stage),
        Err(other) => panic!("unexpected rejection {other:?}"),
    }
}

fn small_batch() -> Batch {
    // 50 live records pad to N* = 64.
    Batch::build(common::records(0x51, 40, 10), 0x51)
}

/// The value of a digit-decomposed main-domain value at cell `x`.
fn value_at(batch: &Batch, main: &MainTables, value: DigitValue, x: usize) -> i64 {
    let digits: Vec<i64> = agg_piop::manifest::value_columns(value)
        .into_iter()
        .map(|c| main.column(&batch.manifest, c).value(x))
        .collect();
    value.layout().compose(&digits)
}

/// Store `x`'s value `value` through its digits.
fn set_value(batch: &Batch, main: &mut MainTables, value: DigitValue, x: usize, new: i64) {
    let form = value.layout();
    let mut digits = vec![0_i64; form.len()];
    form.decompose_into(new, &mut digits);
    for (column, digit) in agg_piop::manifest::value_columns(value).into_iter().zip(digits) {
        let (role, position) = batch.manifest.locate(column);
        match role {
            GroupRole::Main16 => *main.wide_cell_mut(&batch.manifest, column, x) = i32::try_from(digit).expect("i32 digit"),
            _ => main.columns_mut()[position][x] = i8::try_from(digit).expect("i8 digit"),
        }
    }
}

fn w1_at(batch: &Batch, main: &MainTables, x: usize) -> i64 {
    (0..W1_BITS)
        .map(|t| main.column(&batch.manifest, Column::W1Bit(t as u8)).value(x) << t)
        .sum()
}

fn set_w1(batch: &Batch, main: &mut MainTables, x: usize, w1: i64) {
    for t in 0..W1_BITS {
        let position = batch.manifest.locate(Column::W1Bit(t as u8)).1;
        main.columns_mut()[position][x] = ((w1 >> t) & 1) as i8;
    }
}

/// The projection outputs of the committed `v` of `main` under the batch's `Psi`, over the
/// integers.
fn k_of(batch: &Batch, main: &MainTables) -> Vec<i64> {
    let (lo, hi) = (main.column(&batch.manifest, Column::V_LO), main.column(&batch.manifest, Column::V_HI));
    outputs(batch.bridge.psi(), batch.manifest.padded_records(), |i, pos| {
        let x = i * POSITIONS + pos;
        lo.value(x) + (hi.value(x) << 16) + V16_DIGITS.constant()
    })
}

/// Digits of `k` with `K_0` the centered low 16 bits, `K_2 = K_3 = 0` and the rest in `K_1`
/// (a cheating prover's choice when `k` is outside `[-2^37, 2^37)`).
fn cheating_digits(k: &[i64]) -> ProjectionTables {
    let cells = k.len();
    let constant = agg_piop::manifest::K16_DIGITS.constant();
    let mut stacked = vec![0_i32; 2 * cells];
    for (y, &value) in k.iter().enumerate() {
        let rest = value - constant;
        let low = (rest + (1 << 15)).rem_euclid(1 << 16) - (1 << 15);
        stacked[y] = low as i32;
        stacked[cells + y] = i32::try_from((rest - low) >> 16).expect("K_1 fits i32");
    }
    let tables = ProjectionTables { stacked, top: vec![0; 2 * cells] };
    for (y, &value) in k.iter().enumerate() {
        assert_eq!(tables.value(y), E::from_i64(value));
    }
    tables
}

/// The real attack on an unbounded quotient: change `u` and `w_1'` at one position of a live
/// record by `D = delta_u + 2 gamma_2 delta_w`, with `D = p mod q` (centered), and set
/// `v' = v + (p - D)/q` there. Then `Delta = -p = 0` in `F_p`: the ring identity holds at
/// every position, every range of `z, u, w_1'` holds, and `|v'| ~ 2^41`. The outputs `k` of
/// that record leave `[-2^37, 2^37)`, so whatever digits the prover commits, the lookup of
/// `K_0, K_1` or the leaves of `K_2, K_3` rejects; with `K_2 = K_3 = 0` (an `i8` cannot
/// carry the excess of about `2^8`) `K_1 ~ 2^25` is outside `T_16`: the lookup.
#[test]
fn the_unbounded_quotient_attack_is_rejected_by_the_k_range() {
    on_large_stack(|| {
        let batch = small_batch();
        let manifest = &batch.manifest;
        let two_gamma2 = 2 * i128::from(GAMMA2);
        let q = i128::from(Q);
        let r0 = P.rem_euclid(q);
        // A live cell and (delta_w, D) with w_1' in [0, 44) and |u'| <= B_u.
        let (x, delta_w, d) = (0..1024 * 50)
            .map(|p| main_index(p % 256, (p / 256) % 4, p / 1024))
            .find_map(|x| {
                let u = i128::from(value_at(&batch, &batch.main, DigitValue::U, x));
                let w1 = i128::from(w1_at(&batch, &batch.main, x));
                [r0, r0 - q].into_iter().find_map(|d| {
                    (-w1..44 - w1)
                        .find(|&dw| (u + d - two_gamma2 * dw).abs() <= 190_465)
                        .map(|dw| (x, dw, d))
                })
            })
            .expect("a cell admits the attack");
        let delta_u = d - two_gamma2 * delta_w;
        let mut main = batch.main.clone();
        let (u, w1, v) = (
            value_at(&batch, &main, DigitValue::U, x),
            w1_at(&batch, &main, x),
            value_at(&batch, &main, DigitValue::V, x),
        );
        set_value(&batch, &mut main, DigitValue::U, x, u + delta_u as i64);
        set_w1(&batch, &mut main, x, w1 + delta_w as i64);
        assert!((0..44).contains(&(w1 + delta_w as i64)));
        assert!(U16_DIGITS.decompose(u + delta_u as i64).len() == 2);
        // The round-2 tables of the new bits (built while v is still honest, so the honest
        // encoder of k runs), then v' and the cheating digits of its outputs.
        let mut bridge = BridgeGroupTables::build(manifest, &main, &batch.matrices, batch.bridge.psi()).expect("round 2");
        let v_attack = i128::from(v) + (P - d) / q;
        assert_eq!((P - d) % q, 0);
        // Delta' = 2 gamma_2 (w1 - w1') + (u - u') + q (v - v') = -D - (p - D) = -p.
        assert_eq!(-d - q * (v_attack - i128::from(v)), -P, "Delta = 0 in F_p");
        assert!(v_attack.abs() > 1 << 40, "|v'| = {v_attack} is far above 2^38");
        set_value(&batch, &mut main, DigitValue::V, x, i64::try_from(v_attack).expect("i64"));
        assert_eq!(i128::from(value_at(&batch, &main, DigitValue::V, x)), v_attack);
        let k = k_of(&batch, &main);
        let i = x / POSITIONS;
        assert!(k[i * PROJ_ROWS..(i + 1) * PROJ_ROWS].iter().any(|&k| k.abs() >= 1 << 33));
        // The honest encoder refuses these outputs (the prover's abort names the record).
        assert!(matches!(
            ProjectionTables::encode(manifest, &k).unwrap_err(),
            SourceTableError::ProjectionRange { record, .. } if record == i
        ));
        *bridge.projection_mut() = cheating_digits(&k);
        let m = lenient_multiplicities(SourceTables { manifest, main: &main, bridge: &bridge });
        bridge.set_multiplicity(m);
        assert_eq!(rejecting_place(&batch, &main, &bridge), Some(0));
    });
}

/// The digits of `v` are not range-checked: `V_lo + 2^16` with `V_hi - 1` keeps `v` and is
/// accepted (the lookup does not range-check the digits), and `V_hi = 2^16` moves `v` by about `2^32`,
/// which the ring identity rejects (stage 3).
#[test]
fn the_digits_of_v_are_bound_only_through_v() {
    on_large_stack(|| {
        let batch = small_batch();
        let manifest = &batch.manifest;
        let x = main_index(41, 3, 9);
        let mut main = batch.main.clone();
        *main.wide_cell_mut(manifest, Column::V_LO, x) += 1 << 16;
        *main.wide_cell_mut(manifest, Column::V_HI, x) -= 1;
        assert_eq!(value_at(&batch, &main, DigitValue::V, x), value_at(&batch, &batch.main, DigitValue::V, x));
        let bridge = BridgeGroupTables::build(manifest, &main, &batch.matrices, batch.bridge.psi()).unwrap();
        assert_eq!(bridge.projection(), batch.bridge.projection());
        assert_eq!(rejecting_place(&batch, &main, &bridge), None);
        let mut main = batch.main.clone();
        *main.wide_cell_mut(manifest, Column::V_HI, x) = LOOKUP_SHIFT;
        let bridge = BridgeGroupTables::build(manifest, &main, &batch.matrices, batch.bridge.psi()).unwrap();
        assert_eq!(rejecting_place(&batch, &main, &bridge), Some(3));
    });
}

/// `k` bound to `v`: an output changed inside its range (`K_0 + 1`) or `v` changed with `k`
/// kept fails the projection identity (stage 3); `K_2 = 4` and `K_3 = 4` fail `leaf8`
/// (stage 2).
#[test]
fn outputs_unbound_to_v_or_outside_the_k_alphabet_are_rejected() {
    on_large_stack(|| {
        let batch = small_batch();
        let manifest = &batch.manifest;
        let y = 5 + PROJ_ROWS * 7;
        // K_0 + 1 inside T_16 (multiplicities recounted): only the k side moves.
        let mut bridge = batch.bridge.clone();
        let cell = &mut bridge.projection_mut().stacked[y];
        *cell = if *cell < LOOKUP_SHIFT - 1 { *cell + 1 } else { *cell - 1 };
        bridge.recount(manifest, &batch.main).unwrap();
        assert_eq!(rejecting_place(&batch, &batch.main, &bridge), Some(3));
        // v changed with k kept (V_lo + 1): the ring identity and the projection identity.
        let mut main = batch.main.clone();
        let x = main_index(3, 0, 7);
        *main.wide_cell_mut(manifest, Column::V_LO, x) += 1;
        assert_eq!(rejecting_place(&batch, &main, &batch.bridge), Some(3));
        // K_2 = 4: outside [-4, 3], the leaf of stage 2.
        let mut bridge = batch.bridge.clone();
        bridge.projection_mut().top[y] = 4;
        assert_eq!(rejecting_place(&batch, &batch.main, &bridge), Some(2));
        // K_3 = 4: the second leaf, also stage 2.
        let mut bridge = batch.bridge.clone();
        let cells = manifest.group(GroupRole::Proj).logical_cells();
        bridge.projection_mut().top[cells + y] = 4;
        assert_eq!(rejecting_place(&batch, &batch.main, &bridge), Some(2));
        // The honest tables pass.
        assert_eq!(rejecting_place(&batch, &batch.main, &batch.bridge), None);
    });
}

/// One sent value `c` is the claim of both sides; a changed `c` in the proof is rejected.
#[test]
fn the_shared_projection_claim_is_checked() {
    on_large_stack(|| {
        let batch = small_batch();
        let (proof, run) = prove_tables(&batch, batch.tables(), false).expect("honest prover");
        // The honest c is k~(r') = sum_y eq(r', y) k(y).
        let vars = batch.manifest.group(GroupRole::Proj).logical_vars;
        let r_k = agg_piop::stage::suffix(&run.points.r2, vars);
        let k: Vec<i64> = k_of(&batch, &batch.main);
        assert_eq!(proof.stage2.projection, common::mle_i64(&k, &r_k));
        let mut bad = proof.clone();
        bad.stage2.projection += E::from_u64(1);
        assert!(verify_tables(&batch, batch.bridge.public_y(), &bad).is_err());
        assert!(verify_tables(&batch, batch.bridge.public_y(), &proof).is_ok());
    });
}

/// The honest prover aborts when an output would leave `[-2^37, 2^37)`: a record whose `v`
/// has every coordinate at `2^29` (a valid digit encoding) projects to about `2^39` per row.
#[test]
fn the_honest_prover_aborts_on_an_output_outside_the_range() {
    let manifest = agg_piop::manifest::CommitmentManifest::for_relation(3, Relation::Hu).unwrap();
    let mut large = RecordValues::zero();
    for row in large.v.iter_mut() {
        row.fill(1 << 29);
    }
    let main = MainTables::from_values(&manifest, &[RecordValues::zero(), large, RecordValues::zero()]).unwrap();
    let matrices = agg_witness::bridge::BridgeMatrices::random_for_tests(3);
    assert!(matches!(
        BridgeGroupTables::build(&manifest, &main, &matrices, &agg_piop::projection::test_psi(3)).unwrap_err(),
        SourceTableError::ProjectionRange { record: 1, value, .. } if value >= 1 << 37
    ));
    // At 2^26 per coordinate the outputs stay inside (at most 1,024 * 2^26 = 2^36).
    let mut small = RecordValues::zero();
    for row in small.v.iter_mut() {
        row.fill(1 << 26);
    }
    let main = MainTables::from_values(&manifest, &[RecordValues::zero(), small, RecordValues::zero()]).unwrap();
    assert!(BridgeGroupTables::build(&manifest, &main, &matrices, &agg_piop::projection::test_psi(3)).is_ok());
}
