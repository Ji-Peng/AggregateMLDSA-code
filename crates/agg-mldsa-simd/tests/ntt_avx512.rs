#![allow(unexpected_cfgs)]
mod common;
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
    for prime in [Prime::Q, Prime::P2] {
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
    for prime in [Prime::Q, Prime::P2] {
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
    for prime in [Prime::Q, Prime::P2] {
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
#[cfg(cref)]
#[test]
fn c_reference() {
    let Some(e) = engine() else { return };
    let mut rng = SplitMix64::new(123);
    for _ in 0..128 {
        let a = Poly32(core::array::from_fn(|_| {
            (rng.next_u64() % (1 << 23)) as i32 - (1 << 22)
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
