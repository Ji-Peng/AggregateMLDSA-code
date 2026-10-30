use crate::common;
use agg_mldsa_simd::{
    dispatch::{Backend, Engine},
    ntt::{Form, Scale},
    params::{consts, Prime},
    poly::Poly32,
};
#[path = "../../src/ntt/avx2_tables.rs"]
mod tables;
fn engine() -> Option<Engine> {
    let e = Engine::new(Backend::Avx2);
    if e.is_none() {
        eprintln!("Skipping: AVX2/POPCNT unavailable");
    }
    e
}
#[test]
fn forward_and_roundtrips() {
    let Some(e) = engine() else { return };
    let mut rng = common::SplitMix64::new(42);
    for p in [Prime::Q, Prime::P2] {
        let c = consts(p);
        let oracle = common::oracle_ntt_prime(p);
        for case in 0..292 {
            let a = Poly32(std::array::from_fn(|i| match case {
                0 => 1 << 22,
                1 => -(1 << 22),
                2 => {
                    if i % 2 == 0 {
                        1 << 22
                    } else {
                        -(1 << 22)
                    }
                }
                3..=258 => {
                    if i == case - 3 {
                        1 << 22
                    } else {
                        0
                    }
                }
                _ => ((rng.next_u64() % ((1 << 23) + 1)) as i32) - (1 << 22),
            }));
            let mut got = a;
            e.ntt(p, &mut got);
            let mut fips = got;
            e.order_to_fips(&mut fips);
            assert_eq!(
                common::poly32_to_u64(&fips, c.p),
                oracle.ntt(&common::poly32_to_u64(&a, c.p)),
                "{p:?} case {case}"
            );
            #[cfg(cref)]
            {
                let mut cf = a;
                agg_mldsa_cref::ntt_avx(&mut cf.0, &tables::table(p, Scale::Mont).0);
                assert_eq!(got, cf);
            }
            e.reduce(p, &mut got, Form::Centered);
            for f in [Form::Canonical, Form::Centered] {
                let mut r = got;
                e.intt(p, &mut r, Scale::Plain, f);
                common::assert_poly_eq_mod_p(&r, &a, c.p);
                check_range(&r, c.p, f);
            }
        }
    }
}
fn check_range(a: &Poly32, p: i32, f: Form) {
    for &x in &a.0 {
        assert!(
            match f {
                Form::Canonical => (0..p).contains(&x),
                Form::Centered => x.abs() <= p / 2,
            },
            "{x} {p} {f:?}"
        );
    }
}
#[test]
fn dot_schoolbook_and_orders() {
    let Some(e) = engine() else { return };
    let mut rng = common::SplitMix64::new(7);
    let labels = Poly32(std::array::from_fn(|i| i as i32));
    let mut labels2 = labels;
    e.order_from_fips(&mut labels2);
    #[cfg(cref)]
    {
        let mut cf = labels;
        agg_mldsa_cref::nttunpack_avx(&mut cf.0);
        assert_eq!(labels2, cf);
    }
    e.order_to_fips(&mut labels2);
    assert_eq!(labels, labels2);
    for p in [Prime::Q, Prime::P2] {
        let c = consts(p);
        for _ in 0..8 {
            let a: [Poly32; 4] = std::array::from_fn(|_| {
                Poly32(std::array::from_fn(|_| {
                    (rng.next_u64() % 8380417) as i32 - 4190208
                }))
            });
            let b: [Poly32; 4] = std::array::from_fn(|_| {
                Poly32(std::array::from_fn(|_| {
                    (rng.next_u64() % 262145) as i32 - 131072
                }))
            });
            let mut ah = a;
            let mut bh = b;
            for k in 0..4 {
                e.ntt(p, &mut ah[k]);
                e.ntt(p, &mut bh[k]);
                e.reduce(
                    p,
                    &mut ah[k],
                    if p == Prime::Q {
                        Form::Canonical
                    } else {
                        Form::Centered
                    },
                );
            }
            let mut got = Poly32::zero();
            e.dot4(p, &ah, &bh, &mut got);
            #[cfg(cref)]
            if p == Prime::Q {
                let mut cf = Poly32::zero();
                // SAFETY: the reference reads four aligned 256-coefficient polynomials
                // from each array and 624 table words, all of which exist here.
                unsafe {
                    agg_mldsa_cref::ffi::pqcrystals_dilithium2_avx2_pointwise_acc_avx(
                        cf.0.as_mut_ptr(),
                        ah.as_ptr().cast(),
                        bh.as_ptr().cast(),
                        tables::Q.0.as_ptr(),
                    );
                }

                assert_eq!(got, cf);
                let mut inv = cf;
                agg_mldsa_cref::invntt_avx(&mut inv.0, &tables::Q.0);
                let mut ri = got;
                e.intt(p, &mut ri, Scale::Mont, Form::Canonical);
                e.reduce(p, &mut inv, Form::Canonical);
                assert_eq!(ri, inv);
            }
            e.intt(p, &mut got, Scale::Mont, Form::Canonical);
            check_range(&got, c.p, Form::Canonical);
            let expected: [i32; 256] = {
                let products: Vec<_> = (0..4)
                    .map(|k| common::negacyclic_schoolbook_mod(&a[k].0, &b[k].0, c.p))
                    .collect();
                std::array::from_fn(|i| {
                    products
                        .iter()
                        .map(|v| v[i] as i64)
                        .sum::<i64>()
                        .rem_euclid(c.p as i64) as i32
                })
            };
            assert_eq!(got.0, expected);
        }
    }
}
#[test]
fn reduction_full_i32() {
    let Some(e) = engine() else { return };
    let mut rng = common::SplitMix64::new(99);
    for p in [Prime::Q, Prime::P2] {
        for _ in 0..64 {
            let mut a = Poly32(std::array::from_fn(|i| match i {
                0 => i32::MIN,
                1 => i32::MAX,
                _ => rng.next_u64() as i32,
            }));
            let original = a;
            for f in [Form::Canonical, Form::Centered] {
                a = original;
                e.reduce(p, &mut a, f);
                check_range(&a, consts(p).p, f);
                common::assert_poly_eq_mod_p(&a, &original, consts(p).p);
            }
        }
    }
}
#[test]
#[cfg(cref)]
fn qdata_matches_reference() {
    assert_eq!(&tables::Q.0, agg_mldsa_cref::qdata());
}

/// Natural-order butterfly index `k` (1..256) of twiddle entry `i` (1..296) in
/// the reference table: levels 3-7 repeat or interleave the 2^l twiddles of
/// level l across the vector lanes.
fn table_twiddle_index(i: usize) -> usize {
    match i {
        1..=7 => i,
        8..=39 => 8 + (i - 8) / 4,
        40..=71 => 16 + (i - 40) / 2,
        72..=103 => 32 + i - 72,
        104..=167 => 64 + 2 * ((i - 104) % 32) + (i - 104) / 32,
        168..=295 => 128 + 4 * ((i - 168) % 32) + (i - 168) / 32,
        _ => unreachable!(),
    }
}

#[test]
fn schedule_bounds() {
    // Absolute-value envelopes in natural butterfly coordinates; the register
    // shuffles only permute these positions. Every twiddle magnitude is read
    // from the table the kernels load. Use ceiling division throughout.
    for p in [Prime::Q, Prime::P2] {
        let c = consts(p);
        let modulus = c.p as i64;
        let oracle = common::oracle_ntt_prime(p);
        let red = |bound: i64, z: i64| -> i64 {
            (bound * z.abs() + ((1i64 << 32) - 1)) / (1i64 << 32) + (modulus + 1) / 2
        };
        let t = &tables::table(p, Scale::Mont).0;
        let mut tw = [0i64; 256];
        for i in 1..296 {
            let k = table_twiddle_index(i);
            let z = t[328 + i] as i64;
            assert!(tw[k] == 0 || tw[k] == z, "{p:?} entry {i}");
            assert_eq!(
                t[32 + i],
                (z as i32).wrapping_mul(c.pinv),
                "{p:?} entry {i}"
            );
            // The table holds zeta^brv(k) R, centered.
            let expected = (oracle.zetas[k] * ((1u64 << 32) % c.p as u64) % c.p as u64) as i64;
            assert_eq!(z.rem_euclid(modulus), expected, "{p:?} entry {i}");
            assert!(z.abs() <= modulus / 2);
            tw[k] = z;
        }
        assert!(tw[1..].iter().all(|&z| z != 0));
        let mut bound = [1i64 << 22; 256];
        let mut k = 0;
        for layer in 0..8 {
            let len = 128 >> layer;
            for start in (0..256).step_by(2 * len) {
                k += 1;
                for j in start..start + len {
                    let t = red(bound[j + len], tw[k]);
                    let b = bound[j] + t;
                    assert!(b < 1i64 << 31);
                    bound[j] = b;
                    bound[j + len] = b;
                }
            }
        }
        let max_forward = *bound.iter().max().unwrap();
        eprintln!(
            "{p:?} forward envelope: {max_forward} ({:.4} p)",
            max_forward as f64 / modulus as f64
        );
        assert!(max_forward < (1i64 << 31));
        // Inverse input: |x| < q for q (dot4 output), |x| <= (P2-1)/2 for P2.
        bound = [if p == Prime::Q {
            modulus - 1
        } else {
            modulus / 2
        }; 256];
        k = 256;
        let mut max_inverse = 0;
        // Layers 0-6; the last layer is modelled per scale below.
        for layer in 0..7 {
            let len = 1 << layer;
            for start in (0..256).step_by(2 * len) {
                k -= 1;
                for j in start..start + len {
                    let sum = bound[j] + bound[j + len];
                    assert!(sum < 1i64 << 31, "{p:?} layer {layer} position {j}: {sum}");
                    max_inverse = max_inverse.max(sum);
                    bound[j] = sum;
                    bound[j + len] = red(sum, tw[k]);
                    if p == Prime::P2 && layer == 3 {
                        // Sum outputs are reduced by a Montgomery multiply with R mod p.
                        bound[j] = red(bound[j], c.mont as i64);
                        assert!(bound[j] < 63 * modulus / 100);
                    }
                }
            }
        }
        assert_eq!(k, 2);
        // Last layer: the sum is multiplied by f (table words 16-31); the raw
        // difference is multiplied once by centered(zeta^128 f) (table entry 0).
        for scale in [Scale::Plain, Scale::Mont] {
            let ts = &tables::table(p, scale).0;
            let f = ts[24] as i64;
            let zf = ts[328] as i64;
            assert_eq!(
                f,
                if scale == Scale::Mont {
                    c.inv_mont
                } else {
                    c.inv_plain
                } as i64
            );
            assert_eq!(ts[16], ts[24].wrapping_mul(c.pinv));
            assert_eq!(ts[32], ts[328].wrapping_mul(c.pinv));
            let mut max_out = 0;
            for j in 0..128 {
                let sum = bound[j] + bound[j + 128];
                assert!(sum < 1i64 << 31, "{p:?} last layer position {j}: {sum}");
                max_inverse = max_inverse.max(sum);
                max_out = max_out.max(red(sum, f)).max(red(sum, zf));
            }
            eprintln!(
                "{p:?} {scale:?}: inverse max {max_inverse}, final max {max_out} ({:.4} p)",
                max_out as f64 / modulus as f64
            );
            // The canonical/centered store needs (-p, p).
            assert!(max_out < modulus);
            if p == Prime::Q && scale == Scale::Mont {
                assert!(max_out < 74 * modulus / 100);
            }
        }
    }
}

#[test]
fn inverse_extreme_inputs() {
    let Some(e) = engine() else { return };
    for p in [Prime::Q, Prime::P2] {
        let c = consts(p);
        let oracle = common::oracle_ntt_prime(p);
        let max = if p == Prime::Q { c.p - 1 } else { c.p / 2 };
        for pattern in 0..260 {
            let input = Poly32(std::array::from_fn(|i| match pattern {
                0 => max,
                1 => -max,
                2 => {
                    if i % 2 == 0 {
                        max
                    } else {
                        -max
                    }
                }
                3..=258 => {
                    if i == pattern - 3 {
                        max
                    } else {
                        0
                    }
                }
                _ => 0,
            }));
            let mut fips = input;
            e.order_to_fips(&mut fips);
            let expected = oracle.intt(&common::poly32_to_u64(&fips, c.p));
            for form in [Form::Canonical, Form::Centered] {
                let mut got = input;
                e.intt(p, &mut got, Scale::Plain, form);
                check_range(&got, c.p, form);
                assert_eq!(common::poly32_to_u64(&got, c.p), expected);
            }
        }
    }
}

#[test]
fn dot_extreme_inputs() {
    // Largest P2 products the contract allows: centered a at +-(P2-1)/2 and b
    // at the forward envelope, all four products of one sign or alternating.
    let Some(e) = engine() else { return };
    let s = Engine::new(Backend::Scalar).unwrap();
    for p in [Prime::Q, Prime::P2] {
        let c = consts(p);
        let (amax, bmax) = if p == Prime::Q {
            (c.p - 1, 37_829_663)
        } else {
            (c.p / 2, 711_838_595)
        };
        for pattern in 0..6 {
            let sign = |i: usize, k: usize| -> i32 {
                match pattern {
                    0 => 1,
                    1 => -1,
                    2 => {
                        if i % 2 == 0 {
                            1
                        } else {
                            -1
                        }
                    }
                    3 => {
                        if k % 2 == 0 {
                            1
                        } else {
                            -1
                        }
                    }
                    4 => {
                        if (i / 8) % 2 == 0 {
                            1
                        } else {
                            -1
                        }
                    }
                    _ => {
                        if i % 3 == 0 {
                            -1
                        } else {
                            1
                        }
                    }
                }
            };
            let a: [Poly32; 4] = std::array::from_fn(|k| {
                Poly32(std::array::from_fn(|i| {
                    if p == Prime::Q {
                        amax
                    } else {
                        sign(i, k) * amax
                    }
                }))
            });
            let b: [Poly32; 4] =
                std::array::from_fn(|k| Poly32(std::array::from_fn(|i| sign(i, k) * bmax)));
            let mut got = Poly32::zero();
            let mut want = Poly32::zero();
            e.dot4(p, &a, &b, &mut got);
            s.dot4(p, &a, &b, &mut want);
            common::assert_poly_eq_mod_p(&got, &want, c.p);
            if p == Prime::P2 {
                check_range(&got, c.p, Form::Centered);
                assert_eq!(got, want, "pattern {pattern}");
            } else {
                assert!(got.0.iter().all(|x| x.abs() < c.p));
            }
        }
    }
}
