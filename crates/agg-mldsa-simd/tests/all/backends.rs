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
        for p in [Prime::Q, Prime::P2, Prime::P3] {
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

            let orig5: [Poly32; 5] = std::array::from_fn(|_| {
                Poly32(std::array::from_fn(|_| {
                    (rng.next_u64() % 8380417) as i32 - 4190208
                }))
            });
            let mut a5 = orig5;
            let mut b5 = orig5;
            let mut aa5 = orig5;
            let mut bb5 = orig5;
            for i in 0..5 {
                scalar.ntt(p, &mut a5[i]);
                scalar.ntt(p, &mut b5[i]);
                scalar.reduce(
                    p,
                    &mut a5[i],
                    if p == Prime::Q {
                        Form::Canonical
                    } else {
                        Form::Centered
                    },
                );
                engine.ntt(p, &mut aa5[i]);
                engine.ntt(p, &mut bb5[i]);
                engine.reduce(
                    p,
                    &mut aa5[i],
                    if p == Prime::Q {
                        Form::Canonical
                    } else {
                        Form::Centered
                    },
                );
            }
            let mut x5 = Poly32::new();
            let mut y5 = Poly32::new();
            scalar.dot_l::<5>(p, &a5, &b5, &mut x5);
            engine.dot_l::<5>(p, &aa5, &bb5, &mut y5);
            assert_eq!(canonical(&scalar, p, x5), canonical(&engine, p, y5));

            let orig7: [Poly32; 7] = std::array::from_fn(|_| {
                Poly32(std::array::from_fn(|_| {
                    (rng.next_u64() % 8380417) as i32 - 4190208
                }))
            });
            let mut a7 = orig7;
            let mut b7 = orig7;
            let mut aa7 = orig7;
            let mut bb7 = orig7;
            for i in 0..7 {
                scalar.ntt(p, &mut a7[i]);
                scalar.ntt(p, &mut b7[i]);
                scalar.reduce(
                    p,
                    &mut a7[i],
                    if p == Prime::Q {
                        Form::Canonical
                    } else {
                        Form::Centered
                    },
                );
                engine.ntt(p, &mut aa7[i]);
                engine.ntt(p, &mut bb7[i]);
                engine.reduce(
                    p,
                    &mut aa7[i],
                    if p == Prime::Q {
                        Form::Canonical
                    } else {
                        Form::Centered
                    },
                );
            }
            let mut x7 = Poly32::new();
            let mut y7 = Poly32::new();
            scalar.dot_l::<7>(p, &a7, &b7, &mut x7);
            engine.dot_l::<7>(p, &aa7, &bb7, &mut y7);
            assert_eq!(canonical(&scalar, p, x7), canonical(&engine, p, y7));
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
        let w: [[u8; 256]; 4] = std::array::from_fn(|_| std::array::from_fn(|_| (rng.next_u64() % 44) as u8));
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

        // Test ML-DSA-65 and ML-DSA-87 parameter sets across SIMD backend and scalar
        let scalar65 = scalar.for_set::<Set65>();
        let engine65 = engine.for_set::<Set65>();
        let scalar87 = scalar.for_set::<Set87>();
        let engine87 = engine.for_set::<Set87>();

        // 1. 20-bit unpack_z
        let mut bytes65 = [0u8; 3200];
        rng.fill_bytes(&mut bytes65);
        let mut z65_s = [Poly32::new(); 5];
        let mut z65_e = [Poly32::new(); 5];
        assert_eq!(
            scalar65.unpack_z(&bytes65, &mut z65_s),
            engine65.unpack_z(&bytes65, &mut z65_e)
        );
        assert_eq!(z65_s, z65_e);

        let mut bytes87 = [0u8; 4480];
        rng.fill_bytes(&mut bytes87);
        let mut z87_s = [Poly32::new(); 7];
        let mut z87_e = [Poly32::new(); 7];
        assert_eq!(
            scalar87.unpack_z(&bytes87, &mut z87_s),
            engine87.unpack_z(&bytes87, &mut z87_e)
        );
        assert_eq!(z87_s, z87_e);

        // 2. 4-bit pack_w1
        let w65: [[u8; 256]; 6] = std::array::from_fn(|_| std::array::from_fn(|_| (rng.next_u64() % 16) as u8));
        let mut pack65_s = [0u8; 768];
        let mut pack65_e = [0u8; 768];
        scalar65.pack_w1(&w65, &mut pack65_s);
        engine65.pack_w1(&w65, &mut pack65_e);
        assert_eq!(pack65_s, pack65_e);

        let w87: [[u8; 256]; 8] = std::array::from_fn(|_| std::array::from_fn(|_| (rng.next_u64() % 16) as u8));
        let mut pack87_s = [0u8; 1024];
        let mut pack87_e = [0u8; 1024];
        scalar87.pack_w1(&w87, &mut pack87_s);
        engine87.pack_w1(&w87, &mut pack87_e);
        assert_eq!(pack87_s, pack87_e);

        // 3. Sparse c*t rows for k=6 (tau=49) and k=8 (tau=60)
        let mut c49 = [Challenge::<49>::new()];
        scalar.sample_in_ball_many(&[[65u8; 32]], &mut c49);
        let t1_65: [[u16; 256]; 6] = std::array::from_fn(|_| std::array::from_fn(|_| (rng.next_u64() % 1024) as u16));
        for shift in [0, 13] {
            let mut a = [[0i32; 256]; 6];
            let mut b = [[0i32; 256]; 6];
            scalar65.sparse_ct(&c49[0], &t1_65, shift, &mut a);
            engine65.sparse_ct(&c49[0], &t1_65, shift, &mut b);
            assert_eq!(a, b);
        }

        let mut c60 = [Challenge::<60>::new()];
        scalar.sample_in_ball_many(&[[87u8; 32]], &mut c60);
        let t1_87: [[u16; 256]; 8] = std::array::from_fn(|_| std::array::from_fn(|_| (rng.next_u64() % 1024) as u16));
        for shift in [0, 13] {
            let mut a = [[0i32; 256]; 8];
            let mut b = [[0i32; 256]; 8];
            scalar87.sparse_ct(&c60[0], &t1_87, shift, &mut a);
            engine87.sparse_ct(&c60[0], &t1_87, shift, &mut b);
            assert_eq!(a, b);
        }

        // 4. Finishing with gamma2 = (q-1)/32 and 3-prime / exact paths
        for _ in 0..16 {
            let az: [i128; 256] = std::array::from_fn(|_| {
                (rng.next_u64() % 1_000_000_000_000_000) as i128 - 500_000_000_000_000
            });
            let xq = Poly32(az.map(|x| x.rem_euclid(params::Q as i128) as i32));
            let xp = Poly32(az.map(|x| x.rem_euclid(params::P2 as i128) as i32));
            let xp3 = Poly32(az.map(|x| x.rem_euclid(params::P3 as i128) as i32));
            let t = std::array::from_fn(|_| (rng.next_u64() % 600_000_001) as i32 - 300_000_000);
            let h = std::array::from_fn(|_| (rng.next_u64() & 1) as u8);

            let row65 = FinishRow {
                xq: &xq,
                xp: &xp,
                t: &t,
                h: &h,
            };
            let mut out65_s = FinishOut::new();
            let mut out65_e = FinishOut::new();
            let res65_s = scalar65.finish_row(row65, &mut out65_s);
            let res65_e = engine65.finish_row(row65, &mut out65_e);
            assert_eq!(res65_s, res65_e);
            if res65_s.is_ok() {
                assert_eq!(out65_s, out65_e);
            }

            let row3p = FinishRow3P {
                xq: &xq,
                xp2: &xp,
                xp3: &xp3,
                t: &t,
                h: &h,
            };
            let mut out3p_s = FinishOut::new();
            let mut out3p_e = FinishOut::new();
            let res3p_s = scalar65.finish_row_3p(row3p, &mut out3p_s);
            let res3p_e = engine65.finish_row_3p(row3p, &mut out3p_e);
            assert_eq!(res3p_s, res3p_e);
            if res3p_s.is_ok() {
                assert_eq!(out3p_s, out3p_e);
            }

            let row_exact = FinishRowExact {
                az: &az,
                xq: &xq,
                t: &t,
                h: &h,
            };
            let mut out_ex_s = FinishOut::new();
            let mut out_ex_e = FinishOut::new();
            let res_ex_s = scalar87.finish_row_exact(row_exact, &mut out_ex_s);
            let res_ex_e = engine87.finish_row_exact(row_exact, &mut out_ex_e);
            assert_eq!(res_ex_s, res_ex_e);
            if res_ex_s.is_ok() {
                assert_eq!(out_ex_s, out_ex_e);
            }

            // 5. hu_columns for Set65
            let mut aw65_s = [[MaybeUninit::uninit(); 256]; 4];
            let mut aw65_e = aw65_s;
            let mut an65_s = [[MaybeUninit::uninit(); 256]; 8];
            let mut an65_e = an65_s;
            scalar65.hu_columns(
                &z65_s[0],
                &out65_s,
                HuCols {
                    wide: aw65_s.each_mut().map(|x| x.as_mut_slice()),
                    narrow: an65_s.each_mut().map(|x| x.as_mut_slice()),
                },
            );
            engine65.hu_columns(
                &z65_e[0],
                &out65_e,
                HuCols {
                    wide: aw65_e.each_mut().map(|x| x.as_mut_slice()),
                    narrow: an65_e.each_mut().map(|x| x.as_mut_slice()),
                },
            );
            unsafe {
                for c in 0..4 {
                    for i in 0..256 {
                        assert_eq!(aw65_s[c][i].assume_init(), aw65_e[c][i].assume_init());
                    }
                }
                for c in 0..8 {
                    for i in 0..256 {
                        assert_eq!(an65_s[c][i].assume_init(), an65_e[c][i].assume_init());
                    }
                }
            }
        }
    }
}
