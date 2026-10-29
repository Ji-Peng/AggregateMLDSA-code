mod common;
use agg_mldsa_simd::{
    dispatch::{Backend, Engine},
    finish::{FinishRow, HuCols},
    keccak::{Sink, Xof, XofJob},
    ntt::{Form, Scale},
    params::*,
    poly::*,
};
use agg_witness::{crt, derive, expand, gen, native};
use common::*;
fn engine() -> Engine {
    Engine::new(Backend::Scalar).unwrap()
}

#[test]
fn transforms_match_oracle_and_roundtrip() {
    let e = engine();
    let mut rng = SplitMix64::new(42);
    for p in [Prime::Q, Prime::P2] {
        let c = p.consts();
        let oracle = oracle_ntt_prime(p);
        for case in 0..280 {
            let input = Poly32(std::array::from_fn(|i| match case {
                0 => 0,
                1 => 1 << 22,
                2 => -(1 << 22),
                3 => c.p - 1,
                4..=259 => {
                    if i == case - 4 {
                        1 << 22
                    } else {
                        0
                    }
                }
                _ => (rng.next_u64() % (1 << 23)) as i32 - (1 << 22),
            }));
            // The all-(p-1) case is reduced to the documented natural input bound.
            let mut a = input;
            if case == 3 {
                e.reduce(p, &mut a, Form::Centered);
            }
            let expected = oracle.ntt(&poly32_to_u64(&a, c.p));
            let original = a;
            e.ntt(p, &mut a);
            assert_eq!(poly32_to_u64(&a, c.p), expected, "{p:?} case {case}");
            e.intt(p, &mut a, Scale::Plain, Form::Centered);
            assert_poly_eq_mod_p(&a, &original, c.p);
        }
        let mut a = Poly32(std::array::from_fn(|_| rng.next_u64() as i32));
        let expected = poly32_to_u64(&a, c.p);
        e.reduce(p, &mut a, Form::Canonical);
        assert_eq!(a.0.map(|x| x as u64), expected);
    }
}

#[test]
fn dot_products_match_schoolbook() {
    let e = engine();
    for p in [Prime::Q, Prime::P2] {
        for case in 0..260 {
            let a: [Poly32; 4] = std::array::from_fn(|r| {
                Poly32(std::array::from_fn(|i| {
                    if case < 2 {
                        if case == 0 {
                            1 << 22
                        } else {
                            -(1 << 22)
                        }
                    } else if i == (case - 2) % 256 {
                        if r % 2 == 0 {
                            1 << 22
                        } else {
                            -(1 << 22)
                        }
                    } else {
                        0
                    }
                }))
            });
            let b = std::array::from_fn(|r| {
                Poly32(std::array::from_fn(|i| {
                    if i == (r * 67 + case) % 256 {
                        -(1 << 22)
                    } else {
                        0
                    }
                }))
            });
            let mut ah = a;
            let mut bh = b;
            for x in &mut ah {
                e.ntt(p, x);
                e.reduce(
                    p,
                    x,
                    if p == Prime::Q {
                        Form::Canonical
                    } else {
                        Form::Centered
                    },
                );
            }
            for x in &mut bh {
                e.ntt(p, x);
            }
            let mut out = Poly32::zero();
            e.dot4(p, &ah, &bh, &mut out);
            e.intt(p, &mut out, Scale::Mont, Form::Canonical);
            let mut expected = [0i64; 256];
            for r in 0..4 {
                let prod = negacyclic_schoolbook_mod(&a[r].0, &b[r].0, p.consts().p);
                for i in 0..256 {
                    expected[i] += prod[i] as i64;
                }
            }
            assert_eq!(out.0, expected.map(|x| (x % p.consts().p as i64) as i32));
        }
    }
}

#[test]
fn expand_and_prepare_match_keys() {
    let e = engine();
    let mut rng = SplitMix64::new(9);
    let rhos: Vec<[u8; 32]> = (0..64)
        .map(|_| {
            let mut x = [0; 32];
            rng.fill_bytes(&mut x);
            x
        })
        .collect();
    let mut out: Vec<_> = (0..64).map(|_| ExpandedA::new()).collect();
    e.expand_a_many(&rhos, &mut out);
    for (rho, new) in rhos.iter().zip(&out) {
        let (old, blocks) = expand::expand_a(rho);
        for r in 0..4 {
            for s in 0..4 {
                assert_eq!(new.a_hat[r][s].0.map(|x| x as u64), old[r][s]);
                assert_eq!(new.blocks[r][s] as usize, blocks[r][s]);
            }
        }
    }
    for i in 0..16 {
        let (_, pk) = gen::keypair(&gen::key_seed(57, i));
        let key = expand::ExpandedKey::from_bytes(&pk).unwrap();
        let mut new = [ExpandedA::new()];
        e.expand_a_many(&[key.rho], &mut new);
        let mut a = [[[0; 256]; 4]; 4];
        let mut ap = [[Poly32::zero(); 4]; 4];
        e.prepare_matrices(&new[0].a_hat, &mut a, &mut ap);
        assert_eq!(a, *key.a);
        let oracle = oracle_ntt_prime(Prime::P2);
        for r in 0..4 {
            for s in 0..4 {
                assert_eq!(
                    poly32_to_u64(&ap[r][s], P2),
                    oracle.ntt_signed(&a[r][s].map(i64::from))
                );
            }
        }
    }
}

fn residue(x: i64, p: i32) -> i32 {
    x.rem_euclid(p as i64) as i32
}
#[test]
fn integer_path_and_real_finish() {
    let e = engine();
    let mut rng = SplitMix64::new(867);
    for case in 0..4 {
        let rec = gen::record(64, case);
        let key = expand::ExpandedKey::from_bytes(&rec.pk).unwrap();
        let mu = native::mu(&key.tr, &rec.msg, &[]);
        let witness = derive::derive(&key, &mu, &rec.sig).unwrap();
        let z = if case == 0 {
            [[GAMMA1 as i64; 256]; 4]
        } else if case == 1 {
            [[-(GAMMA1 as i64); 256]; 4]
        } else if case == 2 {
            std::array::from_fn(|_| {
                std::array::from_fn(|_| (rng.next_u64() % 262145) as i64 - 131072)
            })
        } else {
            witness.z
        };
        let mut expanded = [ExpandedA::new()];
        e.expand_a_many(&[key.rho], &mut expanded);
        let mut a = [[[0; 256]; 4]; 4];
        let mut ap = [[Poly32::zero(); 4]; 4];
        e.prepare_matrices(&expanded[0].a_hat, &mut a, &mut ap);
        let mut zq = z.map(|x| Poly32(x.map(|v| v as i32)));
        let mut zp = zq;
        for poly in &mut zq {
            e.ntt(Prime::Q, poly);
        }
        for poly in &mut zp {
            e.ntt(Prime::P2, poly);
        }
        let expected = crt::az_schoolbook(&a, &z);
        for r in 0..4 {
            let mut xq = Poly32::zero();
            let mut xp = Poly32::zero();
            e.dot4(Prime::Q, &expanded[0].a_hat[r], &zq, &mut xq);
            e.intt(Prime::Q, &mut xq, Scale::Mont, Form::Canonical);
            e.dot4(Prime::P2, &ap[r], &zp, &mut xp);
            e.intt(Prime::P2, &mut xp, Scale::Mont, Form::Canonical);
            let inv = agg_witness::ntt::pow_mod(Q as u64, (P2 - 2) as u64, P2 as u64) as i128;
            for j in 0..256 {
                let k = ((xp[j] - xq[j]) as i128 * inv).rem_euclid(P2 as i128);
                let mut x = xq[j] as i128 + Q as i128 * k;
                let m = Q as i128 * P2 as i128;
                if x > m / 2 {
                    x -= m;
                }
                assert_eq!(x, expected[r][j]);
            }
            if case == 3 {
                let t = witness.t[r].map(|x| x as i32);
                let mut out = FinishOut::new();
                e.finish_row(
                    FinishRow {
                        xq: &xq,
                        xp: &xp,
                        t: &t,
                        h: &witness.h[r],
                    },
                    &mut out,
                )
                .unwrap();
                assert_eq!(out.w1, witness.w1[r]);
                assert_eq!(out.u.0.map(i64::from), witness.u[r]);
                assert_eq!(out.v.0.map(i64::from), witness.v[r]);
            }
        }
        let mut unpacked = [Poly32::zero(); 4];
        let max = e.unpack_z(rec.sig[32..2336].try_into().unwrap(), &mut unpacked);
        let decoded = native::sig_decode(&rec.sig).unwrap();
        assert_eq!(unpacked.map(|x| x.0.map(i64::from)), decoded.z);
        assert_eq!(max as i64, native::inf_norm(&decoded.z));
        let mut bytes = [0; 768];
        e.pack_w1(&witness.w1, &mut bytes);
        assert_eq!(bytes, native::w1_encode(&witness.w1));
    }
}

#[test]
fn finish_synthetic_and_exhaustive_use_hint() {
    let e = engine();
    let mut rng = SplitMix64::new(123);
    for case in 0..64 {
        let az: [i64; 256] = std::array::from_fn(|_| {
            (rng.next_u64() % 1_124_000_000_000_001) as i64 - 562_000_000_000_000
        });
        let xq = Poly32(az.map(|x| residue(x, Q)));
        let xp = Poly32(az.map(|x| residue(x, P2)));
        let t = std::array::from_fn(|_| (rng.next_u64() % (2 * B_T as u64 + 1)) as i32 - B_T);
        let h = std::array::from_fn(|_| (rng.next_u64() & 1) as u8);
        let mut out = FinishOut::new();
        let result = e.finish_row(
            FinishRow {
                xq: &xq,
                xp: &xp,
                t: &t,
                h: &h,
            },
            &mut out,
        );
        let mut bounded = true;
        for i in 0..256 {
            let r = native::ctr_q(az[i] - t[i] as i64);
            let w = native::use_hint(h[i] != 0, r);
            let u = native::ctr_q(r - ALPHA as i64 * w);
            let v = (az[i] - t[i] as i64 - ALPHA as i64 * w - u) / Q as i64;
            bounded &= v.abs() <= B_V as i64;
            if result.is_ok() {
                assert_eq!(
                    (out.w1[i] as i64, out.u[i] as i64, out.v[i] as i64),
                    (w, u, v),
                    "case {case}"
                );
            }
        }
        assert_eq!(result.is_ok(), bounded);
    }
    for hbit in [0, 1] {
        let h = [hbit; 256];
        let t = [0; 256];
        for start in (0..Q).step_by(256) {
            let xq = Poly32(std::array::from_fn(|i| (start + i as i32).min(Q - 1)));
            let xp = xq;
            let mut out = FinishOut::new();
            e.finish_row(
                FinishRow {
                    xq: &xq,
                    xp: &xp,
                    t: &t,
                    h: &h,
                },
                &mut out,
            )
            .unwrap();
            for i in 0..256 {
                assert_eq!(out.w1[i] as i64, native::use_hint(hbit != 0, xq[i] as i64));
            }
        }
    }
}

#[test]
fn sparse_products_and_rejection_continuation() {
    let e = engine();
    for seed in 0..16 {
        let c = native::sample_in_ball(&[seed; 32]);
        let mut challenge = Challenge39::new();
        for (k, &p) in c.positions.iter().enumerate() {
            challenge.pos[k] = p;
            challenge.neg |= ((c.coeffs[p as usize] < 0) as u64) << k;
        }
        for t1 in [
            [[1023u16; 256]; 4],
            std::array::from_fn(|r| std::array::from_fn(|j| ((j * 7 + r * 13) % 1024) as u16)),
        ] {
            let expected = expand::public_t(&c, &t1);
            for shift in [0, 13] {
                let mut out = [[0; 256]; 4];
                e.sparse_ct(&challenge, &t1, shift, &mut out);
                assert_eq!(
                    out.map(|x| x.map(i64::from)),
                    expected.map(|x| x.map(|v| v >> (13 - shift)))
                );
            }
        }
    }
    let c = Challenge39 {
        pos: std::array::from_fn(|i| if i == 38 { 255 } else { i as u8 }),
        neg: 0x5555555555,
    };
    let t = [[1023; 256]; 4];
    let mut out = [[0; 256]; 4];
    e.sparse_ct(&c, &t, 0, &mut out);
    for j in 0..256 {
        let expected: i32 = c
            .pos
            .iter()
            .enumerate()
            .map(|(k, &p)| {
                (if j < (p as usize) { -1023 } else { 1023 }) * (1 - 2 * ((c.neg >> k) & 1) as i32)
            })
            .sum();
        assert_eq!(out[0][j], expected);
    }
    let mut state = RejState::new();
    let mut block = Block::zero();
    block.bytes.fill(255);
    assert!(!e.rej_uniform(&block, &mut state));
    assert_eq!(state.count, 0);
    block.bytes.fill(0);
    for _ in 0..4 {
        assert!(!e.rej_uniform(&block, &mut state));
    }
    assert!(e.rej_uniform(&block, &mut state));
    assert_eq!(state.count, 256);
    assert_eq!(state.coeffs[..256], [0; 256]);
}

struct Collect {
    blocks: Vec<Vec<u8>>,
    count: usize,
}
impl Sink for Collect {
    fn block(&mut self, j: usize, b: &Block, rate: usize) -> bool {
        assert!(b.bytes[rate..].iter().all(|&x| x == 0));
        self.blocks[j].extend_from_slice(&b.bytes[..rate]);
        self.blocks[j].len() == rate * self.count
    }
}
#[test]
fn segmented_xof_and_digest_boundaries() {
    let e = engine();
    let input: Vec<_> = (0..400).map(|i| i as u8).collect();
    for len in 0..=400 {
        for xof in [Xof::Shake128, Xof::Shake256] {
            let cut = len / 3;
            let segments = [&input[..cut], &[][..], &input[cut..len], &[][..]];
            let job = XofJob {
                xof,
                input: segments,
            };
            let mut sink = Collect {
                blocks: vec![Vec::new()],
                count: 1 + len % 6,
            };
            e.run_xof(&[job], &mut sink);
            let mut expected = vec![0; sink.blocks[0].len()];
            if xof == Xof::Shake128 {
                shake128_segments(&segments, &mut expected);
            } else {
                shake256_segments(&segments, &mut expected);
            }
            assert_eq!(sink.blocks[0], expected, "length {len}");
        }
    }
    for n in (0..=17).chain([33]) {
        let inputs = vec![[&input[..135], &input[135..136], &[][..], &input[136..]]; n];
        for len in [0, 1, 32, 64, 136] {
            let mut out = vec![0; n * len];
            e.shake256_many(&inputs, len, &mut out);
            for i in 0..n {
                let mut expected = vec![0; len];
                shake256_segments(&inputs[i], &mut expected);
                assert_eq!(&out[i * len..(i + 1) * len], expected);
            }
        }
    }
}

#[test]
fn ten_thousand_challenges() {
    let e = engine();
    let mut rng = SplitMix64::new(711);
    let seeds: Vec<_> = (0..10000)
        .map(|_| {
            let mut s = [0; 32];
            rng.fill_bytes(&mut s);
            s
        })
        .collect();
    let mut out = vec![Challenge39::new(); seeds.len()];
    e.sample_in_ball_many(&seeds, &mut out);
    for (seed, c) in seeds.iter().zip(out) {
        let expected = native::sample_in_ball(seed);
        assert_eq!(c.pos.as_slice(), expected.positions.as_slice());
        for (k, &p) in c.pos.iter().enumerate() {
            assert_eq!((c.neg >> k) & 1, (expected.coeffs[p as usize] < 0) as u64);
        }
    }
}

#[test]
fn hu_digits_match_generic_at_edges() {
    use agg_witness::digits::{DigitLayout, StepLayout};
    const Z_RADIX16: DigitLayout = DigitLayout {
        name: "z",
        bases: &[65536, 4],
        signed: true,
    };
    const V_RADIX16: DigitLayout = DigitLayout {
        name: "v",
        bases: &[65536, 65536],
        signed: true,
    };
    const U_STEP16: StepLayout = StepLayout {
        low: DigitLayout {
            name: "u",
            bases: &[65536],
            signed: false,
        },
        step: 63079,
        top_base: 6,
        offset: 190465,
    };
    let e = engine();
    let zs = [
        -131072, -131071, -65537, -65536, -1, 0, 65535, 65536, 131071,
    ];
    let us = [
        -B_U,
        -B_U + 1,
        63078 - B_U,
        63079 - B_U,
        126157 - B_U,
        126158 - B_U,
        189236 - B_U,
        189237 - B_U,
        252315 - B_U,
        252316 - B_U,
        315394 - B_U,
        315395 - B_U,
        B_U - 1,
        B_U,
    ];
    let vs = [
        -B_V,
        -B_V + 1,
        -65537,
        -65536,
        -1,
        0,
        65535,
        65536,
        B_V - 1,
        B_V,
    ];
    let z = Poly32(std::array::from_fn(|i| zs[i % zs.len()]));
    let out = FinishOut {
        u: Poly32(std::array::from_fn(|i| us[i % us.len()])),
        v: Poly32(std::array::from_fn(|i| vs[i % vs.len()])),
        w1: std::array::from_fn(|i| (i % 44) as u8),
    };
    let mut wide = [[std::mem::MaybeUninit::uninit(); 256]; 4];
    let mut narrow = [[std::mem::MaybeUninit::uninit(); 256]; 8];
    e.hu_columns(
        &z,
        &out,
        HuCols {
            wide: wide.each_mut().map(|x| x.as_mut_slice()),
            narrow: narrow.each_mut().map(|x| x.as_mut_slice()),
        },
    );
    for i in 0..256 {
        let zd = Z_RADIX16.decompose(z[i] as i64);
        let ud = U_STEP16.decompose(out.u[i] as i64);
        let vd = V_RADIX16.decompose(out.v[i] as i64);
        // SAFETY: hu_columns initializes every element of each asserted-length destination.
        unsafe {
            assert_eq!(
                [
                    wide[0][i].assume_init() as i64,
                    narrow[0][i].assume_init() as i64
                ],
                zd.as_slice()
            );
            assert_eq!(
                [
                    wide[1][i].assume_init() as i64,
                    narrow[1][i].assume_init() as i64
                ],
                ud.as_slice()
            );
            assert_eq!(
                [
                    wide[2][i].assume_init() as i64,
                    wide[3][i].assume_init() as i64
                ],
                vd.as_slice()
            );
            for t in 0..6 {
                assert_eq!(narrow[t + 2][i].assume_init() as u8, (out.w1[i] >> t) & 1);
            }
        }
    }
}
