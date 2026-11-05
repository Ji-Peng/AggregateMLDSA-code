use crate::common;
use agg_mldsa_simd::{
    dispatch::{Backend, Engine},
    ntt::{Form, Scale},
    params::{consts, Prime},
    poly::Poly32,
};
use common::{assert_poly_eq_mod_p, SplitMix64};
fn engine() -> Option<Engine> {
    Engine::new(Backend::Avx512)
}
fn pow(mut a: i64, mut n: usize, p: i64) -> i64 {
    let mut r = 1;
    while n > 0 {
        if n & 1 != 0 {
            r = r * a % p;
        }
        a = a * a % p;
        n >>= 1;
    }
    r
}
fn oracle(prime: Prime, a: &Poly32) -> Poly32 {
    let c = consts(prime);
    let p = c.p as i64;
    let mut out = *a;
    let mut k = 0;
    let mut len = 128;
    while len > 0 {
        for start in (0..256).step_by(2 * len) {
            k += 1;
            let z = pow(c.zeta as i64, (k as u8).reverse_bits() as usize, p);
            for j in start..start + len {
                let t = z * out[j + len] as i64 % p;
                let x = out[j] as i64;
                out[j] = (x + t).rem_euclid(p) as i32;
                out[j + len] = (x - t).rem_euclid(p) as i32;
            }
        }
        len >>= 1;
    }
    out
}
#[test]
fn forward_and_roundtrips() {
    let Some(e) = engine() else { return };
    let mut rng = SplitMix64::new(0x512);
    for prime in [Prime::Q, Prime::P2, Prime::P3] {
        let p = consts(prime).p;
        for case in 0..300 {
            let mut a = Poly32::zero();
            for i in 0..256 {
                a[i] = match case {
                    0 => 0,
                    1 => 1 << 22,
                    2 => -(1 << 22),
                    3 => {
                        if i & 1 == 0 {
                            1 << 22
                        } else {
                            -(1 << 22)
                        }
                    }
                    4..=259 => {
                        if i == case - 4 {
                            1 << 22
                        } else {
                            0
                        }
                    }
                    _ => (rng.next_u64() % ((1 << 23) + 1)) as i32 - (1 << 22),
                };
            }
            let original = a;
            let expected = oracle(prime, &a);
            e.ntt(prime, &mut a);
            let mut ordered = a;
            e.order_to_fips(&mut ordered);
            assert_poly_eq_mod_p(&ordered, &expected, p);
            e.reduce(prime, &mut a, Form::Centered);
            e.intt(prime, &mut a, Scale::Plain, Form::Centered);
            assert_poly_eq_mod_p(&a, &original, p);
            assert!(a.iter().all(|x| x.abs() <= p / 2));
        }
    }
}
#[test]
fn orders_and_full_i32_reduction() {
    let Some(e) = engine() else { return };
    let mut rng = SplitMix64::new(33);
    for prime in [Prime::Q, Prime::P2, Prime::P3] {
        for _ in 0..32 {
            let a = Poly32(core::array::from_fn(|i| match i {
                0 => i32::MIN,
                1 => i32::MAX,
                _ => rng.next_u64() as i32,
            }));
            let mut b = a;
            e.order_from_fips(&mut b);
            e.order_to_fips(&mut b);
            assert_eq!(a, b);
            for form in [Form::Canonical, Form::Centered] {
                b = a;
                e.reduce(prime, &mut b, form);
                let p = consts(prime).p;
                assert_poly_eq_mod_p(&a, &b, p);
                assert!(b.iter().all(|&x| if form == Form::Canonical {
                    (0..p).contains(&x)
                } else {
                    x.abs() <= p / 2
                }));
            }
        }
    }
}
#[test]
fn dot4_schoolbook() {
    let Some(e) = engine() else { return };
    let mut rng = SplitMix64::new(44);
    for prime in [Prime::Q, Prime::P2, Prime::P3] {
        for _ in 0..4 {
            let c = consts(prime);
            let a: [Poly32; 4] = core::array::from_fn(|_| {
                Poly32(core::array::from_fn(|_| {
                    (rng.next_u64() % (1 << 23)) as i32 - (1 << 22)
                }))
            });
            let b: [Poly32; 4] = core::array::from_fn(|_| {
                Poly32(core::array::from_fn(|_| {
                    (rng.next_u64() % (1 << 18)) as i32 - (1 << 17)
                }))
            });
            let mut expected = Poly32::zero();
            for k in 0..4 {
                let product = common::negacyclic_schoolbook_mod(&a[k].0, &b[k].0, c.p);
                for i in 0..256 {
                    expected[i] = ((expected[i] as i64 + product[i] as i64) % c.p as i64) as i32;
                }
            }
            let mut ah = a;
            let mut bh = b;
            for k in 0..4 {
                e.ntt(prime, &mut ah[k]);
                e.reduce(
                    prime,
                    &mut ah[k],
                    if prime == Prime::Q {
                        Form::Canonical
                    } else {
                        Form::Centered
                    },
                );
                e.ntt(prime, &mut bh[k]);
            }
            let mut out = Poly32::zero();
            e.dot4(prime, &ah, &bh, &mut out);
            assert!(out
                .iter()
                .all(|x| x.abs() < if prime == Prime::Q { c.p } else { c.p / 2 + 1 }));
            e.intt(prime, &mut out, Scale::Mont, Form::Canonical);
            assert_poly_eq_mod_p(&out, &expected, c.p);
        }
    }
}
/// Coefficient `i` of test case `trial` with |x| <= bound: the first cases
/// are constant, alternating and random-sign values at the bound, the rest
/// mix bound values with uniform ones.
fn extreme(trial: usize, i: usize, bound: i32, rng: &mut SplitMix64) -> i32 {
    let r = rng.next_u64();
    match trial {
        0 => bound,
        1 => -bound,
        2 => {
            if i & 1 == 0 {
                bound
            } else {
                -bound
            }
        }
        3 => {
            if r & 1 == 0 {
                bound
            } else {
                -bound
            }
        }
        _ if r.is_multiple_of(4) => {
            if r & 4 == 0 {
                bound
            } else {
                -bound
            }
        }
        _ => ((r >> 3) % (2 * bound as u64 + 1)) as i32 - bound,
    }
}
// Exact agreement with the scalar backend in natural order, for all three
// primes, both scales and both forms, at the edges of the documented input
// domains: |a| <= 2^22 forward, |x| < q and |x| <= (p-1)/2 inverse.
#[test]
fn transforms_match_scalar_at_bounds() {
    let Some(e) = engine() else { return };
    let s = Engine::new(Backend::Scalar).unwrap();
    let mut rng = SplitMix64::new(0x5ca1a7);
    for prime in [Prime::Q, Prime::P2, Prime::P3] {
        let p = consts(prime).p;
        let bound = if prime == Prime::Q { p - 1 } else { p / 2 };
        for trial in 0..256 {
            let a = Poly32(core::array::from_fn(|i| {
                extreme(trial, i, 1 << 22, &mut rng)
            }));
            let (mut x, mut y) = (a, a);
            e.ntt(prime, &mut x);
            s.ntt(prime, &mut y);
            e.order_to_fips(&mut x);
            s.order_to_fips(&mut y);
            e.reduce(prime, &mut x, Form::Canonical);
            s.reduce(prime, &mut y, Form::Canonical);
            assert_eq!(x, y, "{prime:?} ntt trial {trial}");
            let v = Poly32(core::array::from_fn(|i| extreme(trial, i, bound, &mut rng)));
            for scale in [Scale::Plain, Scale::Mont] {
                for form in [Form::Canonical, Form::Centered] {
                    let (mut x, mut y) = (v, v);
                    e.order_from_fips(&mut x);
                    s.order_from_fips(&mut y);
                    e.intt(prime, &mut x, scale, form);
                    s.intt(prime, &mut y, scale, form);
                    assert_eq!(x, y, "{prime:?} {scale:?} {form:?} intt trial {trial}");
                }
            }
        }
    }
}
#[cfg(cref)]
#[test]
fn c_reference() {
    let Some(e) = engine() else { return };
    let mut rng = SplitMix64::new(123);
    for trial in 0..128 {
        let a = Poly32(core::array::from_fn(|i| {
            extreme(trial, i, 1 << 22, &mut rng)
        }));
        let mut r = a;
        let mut c = a;
        e.ntt(Prime::Q, &mut r);
        // SAFETY: Runtime feature detection succeeded and Poly32 is 64-byte aligned.
        unsafe { agg_mldsa_cref::ffi::hyb512_ntt_so_avx512(c.0.as_mut_ptr()) };
        assert_eq!(r, c);
        e.reduce(Prime::Q, &mut r, Form::Centered);
        c = r;
        e.intt(Prime::Q, &mut r, Scale::Mont, Form::Centered);
        // SAFETY: Both arrays have 256 initialized coefficients and correct alignment.
        unsafe { agg_mldsa_cref::ffi::hyb512_intt_so_avx512(c.0.as_mut_ptr()) };
        // Hybrid folds the final scale into its last inverse twiddle; its
        // raw representatives may differ from the API's requested centered form.
        for x in c.iter_mut() {
            *x = ((*x as i64 + 4_190_208).rem_euclid(8_380_417) - 4_190_208) as i32;
        }
        assert_eq!(r, c);
        r = a;
        c = a;
        e.order_from_fips(&mut r);
        // SAFETY: The aligned polynomial is valid for Hybrid's in-place shuffle.
        unsafe { agg_mldsa_cref::ffi::hyb512_shuffle(c.0.as_mut_ptr()) };
        assert_eq!(r, c);
    }
}

#[test]
fn arbitrary_inverse_inputs() {
    let Some(e) = engine() else { return };
    let mut rng = SplitMix64::new(256);
    for prime in [Prime::Q, Prime::P2, Prime::P3] {
        let p = consts(prime).p;
        for trial in 0..64 {
            let bound = if prime == Prime::Q { p - 1 } else { p / 2 };
            let input = Poly32(core::array::from_fn(|i| match trial {
                0 => bound,
                1 => -bound,
                2 => {
                    if i & 1 == 0 {
                        bound
                    } else {
                        -bound
                    }
                }
                _ => (rng.next_u64() % (2 * bound as u64 + 1)) as i32 - bound,
            }));
            for form in [Form::Canonical, Form::Centered] {
                let mut a = input;
                e.order_from_fips(&mut a);
                e.intt(prime, &mut a, Scale::Plain, form);
                assert!(a.iter().all(|&x| if form == Form::Canonical {
                    (0..p).contains(&x)
                } else {
                    x.abs() <= p / 2
                }));
                assert_poly_eq_mod_p(&oracle(prime, &a), &input, p);
            }
        }
    }
}
#[test]
fn dot4_accumulator_limits() {
    let Some(e) = engine() else { return };
    for prime in [Prime::Q, Prime::P2, Prime::P3] {
        dot_accumulator_limits::<4>(&e, prime);
        dot_accumulator_limits::<7>(&e, prime);
    }
}
/// Constant inputs at the contract bounds (b at the forward envelope), all of
/// one sign: for P3 and L = 7 the Montgomery output reaches 1.9p, which needs
/// two centering passes.
fn dot_accumulator_limits<const L: usize>(e: &Engine, prime: Prime) {
    let c = consts(prime);
    let p = c.p as i64;
    let a_bound = if prime == Prime::Q { p - 1 } else { p / 2 };
    let b_bound = match prime {
        Prime::Q => 452 * p / 100,
        Prime::P2 => 44 * p / 10,
        Prime::P3 => 454 * p / 100,
    };
    for sa in [-1i64, 1] {
        for sb in [-1i64, 1] {
            if prime == Prime::Q && sa < 0 {
                continue;
            }
            let a = [Poly32([(sa * a_bound) as i32; 256]); L];
            let b = [Poly32([(sb * b_bound) as i32; 256]); L];
            let product = L as i64 * sa * a_bound * sb * b_bound;
            let t = (product as i32).wrapping_mul(c.pinv) as i64;
            let mut expected = (product - t * p) >> 32;
            if prime != Prime::Q {
                expected = (expected + p / 2).rem_euclid(p) - p / 2;
            }
            let mut out = Poly32::zero();
            e.dot_l::<L>(prime, &a, &b, &mut out);
            assert!(out.iter().all(|&x| x as i64 == expected), "{prime:?} L = {L}");
        }
    }
}
