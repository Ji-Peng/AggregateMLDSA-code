//! Relation H-FIPS (relation byte 2) on the witness side: derivation from accepted signatures,
//! the integer checker, the extraction of a FIPS 204 signature, and a synthetic signature of hint
//! weight exactly 80 with a coefficient `w' = q - gamma2, h = 0` (`docs/h-exact.md` Section 10).

// Index loops mirror the FIPS 204 pseudocode and index several arrays at once.
#![allow(clippy::needless_range_loop)]

mod common;

use agg_witness::check::{CheckReport, Identity};
use agg_witness::crt::az_integer;
use agg_witness::derive::derive;
use agg_witness::expand::ExpandedKey;
use agg_witness::gen::{self, SplitMix64};
use agg_witness::hfips::{check_batch, honest_coeff, Coeff, HFipsRecord, L_STEP};
use agg_witness::instance::PublicInstance;
use agg_witness::native::{
    self, commitment_hash, ctr_q, decompose, mod_q, sig_encode, use_hint, w1_encode, SmallK,
    VecL,
};
use agg_witness::ntt::pow_mod;
use agg_witness::params::*;

/// Position of the corner coefficient `w' = q - gamma2, h = 0` in the synthetic signature
/// (`rho`, `j`).
const CORNER: (usize, usize) = (1, 17);
/// Position of the coefficient `w' = 0, h = 1` (the `e = -1` wrap) in the synthetic signature.
const ZERO: (usize, usize) = (2, 101);

fn records_of(inst: &PublicInstance, recs: &[gen::Record]) -> Vec<HFipsRecord> {
    inst.records
        .iter()
        .zip(recs)
        .map(|(p, r)| {
            let w = derive(&p.key, &p.mu, &r.sig).expect("honest record derives");
            HFipsRecord::from_witness(&w).expect("H-FIPS witness")
        })
        .collect()
}

fn signature_hint_weight(sig: &[u8]) -> i64 {
    let d = native::sig_decode(sig).unwrap();
    d.h.iter().flatten().map(|&b| i64::from(b)).sum()
}

#[test]
fn real_signatures_derive_check_and_extract_byte_identically() {
    let recs = common::mixed_records(0x4F1F5, 40, 24);
    let inst = PublicInstance::from_records(&recs).unwrap();
    let hf = records_of(&inst, &recs);
    let rep = check_batch(&inst, &hf);
    assert!(rep.is_ok(), "{:?}", rep.failures);
    let mut weights = Vec::new();
    for ((p, r), h) in inst.records.iter().zip(&recs).zip(&hf) {
        // The hint is uniquely determined by (w', w1'), and HintBitPack is canonical, so the
        // extracted signature is the original one, byte for byte.
        let sig = h.extract_signature(&p.c_tilde).expect("extract");
        assert_eq!(sig, r.sig);
        assert!(native::verify(&r.pk, &r.msg, &sig));
        assert_eq!(h.hint_weight(), signature_hint_weight(&r.sig));
        assert_eq!(h.hint_weight() + h.u_h, OMEGA as i64);
        weights.push(h.hint_weight());
    }
    weights.sort_unstable();
    eprintln!(
        "hint weights of {} honest signatures: min {} median {} max {}",
        weights.len(),
        weights[0],
        weights[weights.len() / 2],
        weights[weights.len() - 1]
    );
}

#[test]
fn edge_residue_record_uses_the_corner_witness() {
    // The pinned record whose w' has a coefficient equal to q - gamma2 (tests/edge.rs).
    let (seed, key, index) = common::EDGE_RESIDUE;
    let r = gen::record_hedged(seed, key, index);
    let inst = PublicInstance::from_records(std::slice::from_ref(&r)).unwrap();
    let p = &inst.records[0];
    let w = derive(&p.key, &p.mu, &r.sig).unwrap();
    let h = HFipsRecord::from_witness(&w).unwrap();
    let mut corners = 0;
    for rho in 0..K {
        for j in 0..N {
            if mod_q(w.w[rho][j]) == Q - GAMMA2 {
                corners += 1;
                let hb = w.h[rho][j] != 0;
                let c = h.coeffs[rho][j];
                assert_eq!(c, honest_coeff(Q - GAMMA2, hb));
                if !hb {
                    assert_eq!((c.sigma, c.l(), c.delta, c.e), (-1, GAMMA2 - 1, 0, 1));
                }
            }
        }
    }
    assert!(corners >= 1);
    assert!(check_batch(&inst, std::slice::from_ref(&h)).is_ok());
    assert_eq!(h.extract_signature(&p.c_tilde).unwrap(), r.sig);
}

/// A synthetic but genuine FIPS 204 signature of hint weight exactly 80 with a coefficient
/// `w' = q - gamma2, h = 0`. The public key has `t1 = 0` (a well-formed 1,312-byte key; pkDecode,
/// Algorithm 23, accepts any bytes), so Algorithm 8 computes `w'_Approx = A z` independently of
/// `c` (line 9): `z` is chosen first (two coefficients adjusted so that `(A z)` hits `q - gamma2`
/// at `CORNER` and `0` at `ZERO`), then the hint `h` (80 ones, none at the corner), then
/// `c~ = H(mu || w1Encode(UseHint(h, w')))` (line 12). Honest signing cannot place a coefficient
/// or a weight deliberately, and random batches of 1K hit the corner with probability about 3e-4.
struct Synthetic {
    pk: Vec<u8>,
    msg: Vec<u8>,
    sig: Vec<u8>,
    h: SmallK,
}

fn synthetic() -> Synthetic {
    let mut rho = [0u8; 32];
    gen::derive_bytes(b"h-fips/rho", &[1], &mut rho);
    let mut pk = rho.to_vec();
    pk.resize(PK_BYTES, 0);
    let key = ExpandedKey::from_bytes(&pk).unwrap();
    assert!(key.t1.iter().flatten().all(|&t| t == 0));
    let msg = b"h-fips synthetic signature of hint weight 80".to_vec();
    let mu = native::mu(&key.tr, &msg, &[]);

    // z uniform in [-B_z, B_z], then two coefficients moved so that (A z) = q - gamma2 at
    // CORNER and (A z) = 0 at ZERO (the only non-wrap residue with r1 = 0 and r0 <= 0, where
    // h = 1 gives the e = -1 wrap to w1' = 43).
    let mut rng = SplitMix64::new(0x4F1F5);
    let mut z: VecL = [[0; N]; L];
    for zi in z.iter_mut() {
        for x in zi.iter_mut() {
            *x = rng.below(2 * B_Z as u64 + 1) as i64 - B_Z;
        }
    }
    let targets = [(CORNER, Q - GAMMA2), (ZERO, 0)];
    let az0 = az_integer(&key, &z);
    let need: Vec<i64> = targets
        .iter()
        .map(|&((r, j), t)| mod_q(t - az0[r][j] as i64))
        .collect();
    // Coefficient of z[s][k] in (A z)[r][j] = sum_s (A[r][s] * z[s])_j (negacyclic).
    let coef = |(r, j): (usize, usize), s: usize, k: usize| -> i64 {
        mod_q(i64::from(key.a[r][s][(j + N - k) % N]) * if k <= j { 1 } else { -1 })
    };
    let inv = |a: i64| pow_mod(a as u64, (Q - 2) as u64, Q as u64) as i64;
    let mulq = |a: i64, b: i64| ((i128::from(a) * i128::from(b)).rem_euclid(i128::from(Q))) as i64;
    let mut moved = false;
    'search: for k1 in 0..N {
        for k2 in (k1 + 1)..N {
            let (s1, s2) = (k1 % L, k2 % L);
            let (a11, a12) = (coef(targets[0].0, s1, k1), coef(targets[0].0, s2, k2));
            let (a21, a22) = (coef(targets[1].0, s1, k1), coef(targets[1].0, s2, k2));
            let det = mod_q(mulq(a11, a22) - mulq(a12, a21));
            if det == 0 {
                continue;
            }
            let di = inv(det);
            let d1 = ctr_q(mulq(mod_q(mulq(need[0], a22) - mulq(a12, need[1])), di));
            let d2 = ctr_q(mulq(mod_q(mulq(a11, need[1]) - mulq(a21, need[0])), di));
            let (n1, n2) = (z[s1][k1] + d1, z[s2][k2] + d2);
            if n1.abs() <= B_Z && n2.abs() <= B_Z {
                z[s1][k1] = n1;
                z[s2][k2] = n2;
                moved = true;
                break 'search;
            }
        }
    }
    assert!(moved, "no two-coefficient move found");
    let az = az_integer(&key, &z);
    let w: [[i64; N]; K] = std::array::from_fn(|r| std::array::from_fn(|j| mod_q(az[r][j] as i64)));
    assert_eq!(w[CORNER.0][CORNER.1], Q - GAMMA2);
    assert_eq!(w[ZERO.0][ZERO.1], 0);

    // Hint: up to two coefficients of each class with h = 1 (wrap; w1' wraps 43 -> 0; w1' wraps
    // 0 -> 43, only w' = 0; plain sigma = 1; plain sigma = -1), then the first others up to
    // exactly 80.
    let class = |wq: i64| -> usize {
        if wq >= Q - GAMMA2 {
            return 0;
        }
        let (r1, r0) = decompose(wq);
        match (r0 > 0, r1) {
            (true, 43) => 1,
            (false, 0) => 2,
            (true, _) => 3,
            (false, _) => 4,
        }
    };
    let mut h: SmallK = [[0; N]; K];
    let mut per_class = [0usize; 5];
    let mut weight = 0usize;
    for r in 0..K {
        for j in 0..N {
            let c = class(w[r][j]);
            if (r, j) != CORNER && per_class[c] < 2 {
                per_class[c] += 1;
                h[r][j] = 1;
                weight += 1;
            }
        }
    }
    assert!(per_class.iter().all(|&c| c >= 1), "every class present: {per_class:?}");
    assert_eq!(h[ZERO.0][ZERO.1], 1);
    for r in 0..K {
        for j in 0..N {
            if weight < OMEGA && (r, j) != CORNER && h[r][j] == 0 {
                h[r][j] = 1;
                weight += 1;
            }
        }
    }
    assert_eq!(weight, OMEGA);
    let mut w1: SmallK = [[0; N]; K];
    for r in 0..K {
        for j in 0..N {
            w1[r][j] = use_hint(h[r][j] != 0, w[r][j]) as u8;
        }
    }
    let c_tilde = commitment_hash(&mu, &w1_encode(&w1));
    let sig = sig_encode(&c_tilde, &z, &h);
    Synthetic { pk, msg, sig, h }
}

fn derive_synthetic(s: &Synthetic) -> (PublicInstance, HFipsRecord) {
    let mut ct = [0u8; C_TILDE_BYTES];
    ct.copy_from_slice(&s.sig[..C_TILDE_BYTES]);
    let inst = PublicInstance::new(&[(s.pk.as_slice(), s.msg.as_slice(), ct)]).unwrap();
    let p = &inst.records[0];
    let w = derive(&p.key, &p.mu, &s.sig).expect("the synthetic signature is accepted");
    assert_eq!(w.h, s.h);
    let rec = HFipsRecord::from_witness(&w).unwrap();
    (inst, rec)
}

#[test]
fn synthetic_weight_80_signature_with_the_corner() {
    let s = synthetic();
    // A genuine FIPS 204 signature: this crate's Algorithm 8 and RustCrypto's verifier accept.
    assert!(native::verify(&s.pk, &s.msg, &s.sig));
    assert!(gen::crate_verify(&s.pk, &s.msg, &s.sig));
    let (inst, rec) = derive_synthetic(&s);
    assert_eq!(rec.hint_weight(), OMEGA as i64);
    assert_eq!(rec.u_h, 0);
    let c = rec.coeffs[CORNER.0][CORNER.1];
    assert_eq!(
        c,
        Coeff {
            l_lo: GAMMA2 - 1 - L_STEP,
            l_hi: 1,
            sigma: -1,
            delta: 0,
            e: 1
        }
    );
    // Every (delta, e) class occurs.
    let mut seen = std::collections::BTreeSet::new();
    for x in rec.coeffs.iter().flatten() {
        seen.insert((x.sigma, x.delta, x.e));
    }
    for class in [(-1, 0, 1), (1, 1, 1), (-1, -1, -1), (-1, -1, 0), (1, 1, 0), (1, 0, 0), (-1, 0, 0)] {
        assert!(seen.contains(&class), "{class:?} in {seen:?}");
    }
    let rep = check_batch(&inst, std::slice::from_ref(&rec));
    assert!(rep.is_ok(), "{:?}", rep.failures);
    // Extraction: the canonical (c~, z, h*) is the signature itself and verifies (Algorithm 8).
    let ct: [u8; C_TILDE_BYTES] = s.sig[..C_TILDE_BYTES].try_into().unwrap();
    let ext = rec.extract_signature(&ct).unwrap();
    assert_eq!(ext, s.sig);
    assert_eq!(rec.extracted_hint(), s.h);
    assert!(native::verify(&s.pk, &s.msg, &ext));
}

#[test]
fn synthetic_corner_fails_under_the_ideator_constraint() {
    // With e (e - delta) = 0 the corner's only witness is (sigma, L, delta, e) = (1, gamma2 - 1,
    // 1, 0): same u = -gamma2, but delta^2 = 1, so the weight identity needs u_h = -1, outside
    // T_16. Our constraint set admits it too (it is sound: h* = 0 <= 1), it is just not minimal.
    let s = synthetic();
    let (inst, rec) = derive_synthetic(&s);
    let (lo, hi) = Coeff::split_l(GAMMA2 - 1);
    let alt = Coeff {
        l_lo: lo,
        l_hi: hi,
        sigma: 1,
        delta: 1,
        e: 0,
    };
    assert_eq!(alt.u(), rec.coeffs[CORNER.0][CORNER.1].u());
    assert_eq!(alt.e * (alt.e - alt.delta), 0);
    let mut bad = rec.clone();
    bad.coeffs[CORNER.0][CORNER.1] = alt;
    let rep = check_batch(&inst, std::slice::from_ref(&bad));
    assert_eq!(failed(&rep), vec![Identity::HintWeight]);
    bad.u_h = -1;
    let rep = check_batch(&inst, std::slice::from_ref(&bad));
    assert_eq!(failed(&rep), vec![Identity::Range16]);
}

fn failed(rep: &CheckReport) -> Vec<Identity> {
    rep.counts.keys().copied().collect()
}

#[test]
fn tampered_witnesses_are_rejected_by_their_identity() {
    let s = synthetic();
    let (inst, rec) = derive_synthetic(&s);
    let w1 = rec.w1();
    let find = |pred: &dyn Fn(&Coeff, i64) -> bool| -> (usize, usize) {
        for r in 0..K {
            for j in 0..N {
                if pred(&rec.coeffs[r][j], w1[r][j]) {
                    return (r, j);
                }
            }
        }
        panic!("no coefficient found");
    };
    let run = |f: &dyn Fn(&mut HFipsRecord)| -> Vec<Identity> {
        let mut t = rec.clone();
        f(&mut t);
        failed(&check_batch(&inst, std::slice::from_ref(&t)))
    };
    let plain = find(&|c, w| c.delta == 0 && c.e == 0 && w != 0 && w != 43);
    // sigma = 0 (not a sign): sigma^2 = 1 fails (and the ring, since u moves).
    let ids = run(&|t| t.coeffs[plain.0][plain.1].sigma = 0);
    assert!(ids.contains(&Identity::SignSquare), "{ids:?}");
    // delta = -sigma.
    let ids = run(&|t| {
        let c = &mut t.coeffs[plain.0][plain.1];
        c.delta = -c.sigma;
    });
    assert!(ids.contains(&Identity::DeltaSign), "{ids:?}");
    // e = -1 with delta = 0.
    let ids = run(&|t| t.coeffs[plain.0][plain.1].e = -1);
    assert!(ids.contains(&Identity::EdgeE), "{ids:?}");
    // e = 1 at w1' != 0.
    let ids = run(&|t| t.coeffs[plain.0][plain.1].e = 1);
    assert!(ids.contains(&Identity::WrapLow), "{ids:?}");
    // e = -1 with delta = sigma = -1 at w1' != 43.
    let neg = find(&|c, w| c.delta == -1 && c.e == 0 && w != 43);
    let ids = run(&|t| t.coeffs[neg.0][neg.1].e = -1);
    assert!(ids.contains(&Identity::WrapHigh), "{ids:?}");
    // L_hi = 2 and L_lo = 2^16.
    let ids = run(&|t| t.coeffs[plain.0][plain.1].l_hi = 2);
    assert!(ids.contains(&Identity::BoolSlack), "{ids:?}");
    let ids = run(&|t| t.coeffs[plain.0][plain.1].l_lo = 1 << 16);
    assert!(ids.contains(&Identity::Range16), "{ids:?}");
    // The other split of an L in the overlap [29,696, 2^16) is a valid witness as well.
    let lo = find(&|c, _| c.l_hi == 0 && c.l_lo >= L_STEP);
    let ids = run(&|t| {
        let c = &mut t.coeffs[lo.0][lo.1];
        c.l_lo -= L_STEP;
        c.l_hi = 1;
    });
    assert_eq!(ids, Vec::<Identity>::new(), "an equal L in range is a valid witness");
    // L_hi = -1 with L_lo + 29,696 keeps L, u and the ring, and L_lo stays in the table: only
    // the Booleanity of L_hi catches it.
    let small = find(&|c, _| c.l_hi == 0 && c.l_lo + L_STEP < 1 << 16);
    let ids = run(&|t| {
        let c = &mut t.coeffs[small.0][small.1];
        c.l_lo += L_STEP;
        c.l_hi = -1;
    });
    assert_eq!(ids, vec![Identity::BoolSlack]);
    // u_h off by one.
    let ids = run(&|t| t.u_h += 1);
    assert_eq!(ids, vec![Identity::HintWeight]);
    // Dropping a hint bit (delta -> 0) moves u by 2 gamma2: the ring fails.
    let hinted = find(&|c, _| c.delta != 0 && c.e == 0);
    let ids = run(&|t| {
        t.coeffs[hinted.0][hinted.1].delta = 0;
        t.u_h += 1;
    });
    assert_eq!(ids, vec![Identity::Ring]);
    // A flipped bit of w1': the ring and (3.3) fail.
    let ids = run(&|t| t.bits[0][plain.0][plain.1] ^= 1);
    assert!(ids.contains(&Identity::Ring) && ids.contains(&Identity::CommitmentHash), "{ids:?}");
    // z digit outside its alphabets: B_zhi = 2 keeps nothing.
    let ids = run(&|t| t.z_digits[0][0][1] = 2);
    assert!(ids.contains(&Identity::BoolSlack), "{ids:?}");
}

#[test]
fn padded_rows_satisfy_the_relation() {
    let recs = gen::records(0x4F1F6, 3);
    let inst = PublicInstance::from_records(&recs).unwrap();
    assert_eq!(inst.geom.n_star, 8);
    let hf = records_of(&inst, &recs);
    let rep = check_batch(&inst, &hf);
    assert!(rep.is_ok(), "{:?}", rep.failures);
    let p = HFipsRecord::padded();
    assert_eq!(p.z(), [[0; N]; L]);
    assert_eq!(p.u(), [[0; N]; K]);
    assert_eq!(p.u_h, OMEGA as i64);
    assert_eq!(p.z_digits[0][0], [32_689, 1, 0]);
}
