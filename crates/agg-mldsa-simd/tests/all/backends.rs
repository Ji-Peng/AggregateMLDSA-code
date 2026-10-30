//! Keep every public operation covered when independently developed kernels replace stubs.
use crate::common;
use agg_mldsa_simd::*;
use common::*;
use std::mem::MaybeUninit;

fn canonical(e: &Engine, p: Prime, mut a: Poly32) -> Poly32 {
    e.order_to_fips(&mut a);
    e.reduce(p, &mut a, Form::Canonical);
    a
}
#[test]
fn every_api_matches_scalar() {
    let scalar = Engine::new(Backend::Scalar).unwrap();
    let mut rng = SplitMix64::new(456);
    for engine in engines()
        .into_iter()
        .filter(|e| e.backend() != Backend::Scalar)
    {
        println!("Checking {:?} against scalar", engine.backend());
        for p in [Prime::Q, Prime::P2] {
            for _ in 0..16 {
                let input = Poly32(std::array::from_fn(|_| {
                    (rng.next_u64() % (1 << 23)) as i32 - (1 << 22)
                }));
                let mut a = input;
                let mut b = input;
                scalar.ntt(p, &mut a);
                engine.ntt(p, &mut b);
                assert_eq!(canonical(&engine, p, b), canonical(&scalar, p, a));
                for scale in [Scale::Plain, Scale::Mont] {
                    for form in [Form::Canonical, Form::Centered] {
                        let mut aa = a;
                        let mut bb = b;
                        scalar.reduce(p, &mut aa, Form::Centered);
                        engine.reduce(p, &mut bb, Form::Centered);
                        scalar.intt(p, &mut aa, scale, form);
                        engine.intt(p, &mut bb, scale, form);
                        assert_eq!(aa, bb);
                    }
                }
                let mut converted = a;
                engine.order_from_fips(&mut converted);
                engine.order_to_fips(&mut converted);
                assert_eq!(converted, a);
                for form in [Form::Canonical, Form::Centered] {
                    let mut a = Poly32(std::array::from_fn(|_| rng.next_u64() as i32));
                    let mut b = a;
                    scalar.reduce(p, &mut a, form);
                    engine.reduce(p, &mut b, form);
                    assert_eq!(a, b);
                }
            }
            let mut a = std::array::from_fn(|_| {
                Poly32(std::array::from_fn(|_| {
                    (rng.next_u64() % 8380417) as i32 - 4190208
                }))
            });
            let mut b = a;
            let mut aa = a;
            let mut bb = b;
            for i in 0..4 {
                scalar.ntt(p, &mut a[i]);
                scalar.ntt(p, &mut b[i]);
                scalar.reduce(
                    p,
                    &mut a[i],
                    if p == Prime::Q {
                        Form::Canonical
                    } else {
                        Form::Centered
                    },
                );
                engine.ntt(p, &mut aa[i]);
                engine.ntt(p, &mut bb[i]);
                engine.reduce(
                    p,
                    &mut aa[i],
                    if p == Prime::Q {
                        Form::Canonical
                    } else {
                        Form::Centered
                    },
                );
            }
            let mut x = Poly32::new();
            let mut y = x;
            scalar.dot4(p, &a, &b, &mut x);
            engine.dot4(p, &aa, &bb, &mut y);
            assert_eq!(canonical(&scalar, p, x), canonical(&engine, p, y));
        }
        for n in (0..=17).chain([33]) {
            let seeds: Vec<_> = (0..n)
                .map(|_| {
                    let mut x = [0; 32];
                    rng.fill_bytes(&mut x);
                    x
                })
                .collect();
            let mut a: Vec<_> = (0..n).map(|_| ExpandedA::new()).collect();
            let mut b = a.clone();
            scalar.expand_a_many(&seeds, &mut a);
            engine.expand_a_many(&seeds, &mut b);
            for i in 0..n {
                assert_eq!(a[i].blocks, b[i].blocks);
                for r in 0..4 {
                    for s in 0..4 {
                        assert_eq!(
                            canonical(&scalar, Prime::Q, a[i].a_hat[r][s]),
                            canonical(&engine, Prime::Q, b[i].a_hat[r][s])
                        );
                    }
                }
                let mut ac = [[[0; 256]; 4]; 4];
                let mut bc = ac;
                let mut ap = [[Poly32::new(); 4]; 4];
                let mut bp = ap;
                scalar.prepare_matrices(&a[i].a_hat, &mut ac, &mut ap);
                engine.prepare_matrices(&b[i].a_hat, &mut bc, &mut bp);
                assert_eq!(ac, bc);
                for r in 0..4 {
                    for s in 0..4 {
                        assert_eq!(
                            canonical(&scalar, Prime::P2, ap[r][s]),
                            canonical(&engine, Prime::P2, bp[r][s])
                        );
                    }
                }
            }
            let inputs: Vec<_> = seeds
                .iter()
                .map(|s| [s.as_slice(), &[][..], s.as_slice(), &[][..]])
                .collect();
            for len in [0, 32, 64, 136] {
                let mut a = vec![0; n * len];
                let mut b = a.clone();
                scalar.shake256_many(&inputs, len, &mut a);
                engine.shake256_many(&inputs, len, &mut b);
                assert_eq!(a, b);
            }
            let mut a = vec![Challenge39::new(); n];
            let mut b = a.clone();
            scalar.sample_in_ball_many(&seeds, &mut a);
            engine.sample_in_ball_many(&seeds, &mut b);
            assert_eq!(a, b);
        }
        let input: Vec<_> = (0..401).map(|_| rng.next_u64() as u8).collect();
        let jobs: Vec<_> = (0..33)
            .map(|j| XofJob {
                xof: if j % 2 == 0 {
                    Xof::Shake128
                } else {
                    Xof::Shake256
                },
                input: [&input[..j * 7], &[], &input[j * 7..], &[]],
            })
            .collect();
        struct Collector(Vec<Vec<u8>>);
        impl Sink for Collector {
            fn block(&mut self, j: usize, b: &Block, rate: usize) -> bool {
                assert!(b.bytes[rate..].iter().all(|&x| x == 0));
                self.0[j].extend_from_slice(&b.bytes[..rate]);
                self.0[j].len() == rate * (1 + j % 6)
            }
        }
        let mut a = Collector(vec![vec![]; 33]);
        let mut b = Collector(vec![vec![]; 33]);
        scalar.run_xof(&jobs, &mut a);
        engine.run_xof(&jobs, &mut b);
        assert_eq!(a.0, b.0);
        let mut st = RejState::new();
        let mut other = st;
        for i in 0..7 {
            let mut block = Block::new();
            rng.fill_bytes(&mut block.bytes);
            if i == 0 {
                block.bytes.fill(255);
            }
            assert_eq!(
                scalar.rej_uniform(&block, &mut st),
                engine.rej_uniform(&block, &mut other)
            );
            assert_eq!(st.count, other.count);
            assert_eq!(st.coeffs[..st.count], other.coeffs[..other.count]);
        }
        let mut bytes = [0u8; 2304];
        rng.fill_bytes(&mut bytes);
        let mut z = [Poly32::new(); 4];
        let mut zz = z;
        assert_eq!(
            scalar.unpack_z(&bytes, &mut z),
            engine.unpack_z(&bytes, &mut zz)
        );
        assert_eq!(z, zz);
        let w = std::array::from_fn(|_| std::array::from_fn(|_| (rng.next_u64() % 44) as u8));
        let mut a = [0; 768];
        let mut b = a;
        scalar.pack_w1(&w, &mut a);
        engine.pack_w1(&w, &mut b);
        assert_eq!(a, b);
        let mut c = [Challenge39::new()];
        scalar.sample_in_ball_many(&[[42; 32]], &mut c);
        let t1 = std::array::from_fn(|_| std::array::from_fn(|_| (rng.next_u64() % 1024) as u16));
        for shift in [0, 13] {
            let mut a = [[0; 256]; 4];
            let mut b = a;
            scalar.sparse_ct(&c[0], &t1, shift, &mut a);
            engine.sparse_ct(&c[0], &t1, shift, &mut b);
            assert_eq!(a, b);
        }
        for _ in 0..32 {
            let az: [i64; 256] = std::array::from_fn(|_| {
                (rng.next_u64() % 1_000_000_000_000_000) as i64 - 500_000_000_000_000
            });
            let xq = Poly32(az.map(|x| x.rem_euclid(params::Q as i64) as i32));
            let xp = Poly32(az.map(|x| x.rem_euclid(params::P2 as i64) as i32));
            let t = std::array::from_fn(|_| (rng.next_u64() % 600_000_001) as i32 - 300_000_000);
            let h = std::array::from_fn(|_| (rng.next_u64() & 1) as u8);
            let row = FinishRow {
                xq: &xq,
                xp: &xp,
                t: &t,
                h: &h,
            };
            let mut a = FinishOut::new();
            let mut b = a;
            let result = scalar.finish_row(row, &mut a);
            assert_eq!(result, engine.finish_row(row, &mut b));
            if result.is_ok() {
                assert_eq!(a, b);
                let mut aw = [[MaybeUninit::uninit(); 256]; 4];
                let mut bw = aw;
                let mut an = [[MaybeUninit::uninit(); 256]; 8];
                let mut bn = an;
                scalar.hu_columns(
                    &z[0],
                    &a,
                    HuCols {
                        wide: aw.each_mut().map(|x| x.as_mut_slice()),
                        narrow: an.each_mut().map(|x| x.as_mut_slice()),
                    },
                );
                engine.hu_columns(
                    &z[0],
                    &b,
                    HuCols {
                        wide: bw.each_mut().map(|x| x.as_mut_slice()),
                        narrow: bn.each_mut().map(|x| x.as_mut_slice()),
                    },
                );
                // SAFETY: both calls initialize every element of their length-256 column slices.
                unsafe {
                    for c in 0..4 {
                        for i in 0..256 {
                            assert_eq!(aw[c][i].assume_init(), bw[c][i].assume_init());
                        }
                    }
                    for c in 0..8 {
                        for i in 0..256 {
                            assert_eq!(an[c][i].assume_init(), bn[c][i].assume_init());
                        }
                    }
                }
            }
        }
    }
}
