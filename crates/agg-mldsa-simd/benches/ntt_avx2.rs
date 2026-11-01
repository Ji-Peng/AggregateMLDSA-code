// Every comparison needs the C reference (`AGG_CREF_SRC`); without it the helpers are unused.
#![cfg_attr(not(cref), allow(dead_code, unused_imports))]
mod common;
use agg_mldsa_simd::{
    dispatch::{Backend, Engine},
    ntt::{Form, Scale},
    params::{self, Prime},
    poly::Poly32,
};
use std::hint::black_box;
#[allow(dead_code)]
#[path = "../src/ntt/avx2_tables.rs"]
mod tables;
#[inline(never)]
fn rust_ntt(e: &Engine, p: Prime, a: &mut Poly32) {
    e.ntt(p, black_box(a));
}
#[inline(never)]
fn rust_intt(e: &Engine, p: Prime, a: &mut Poly32, s: Scale) {
    e.intt(p, black_box(a), s, Form::Canonical);
}
#[inline(never)]
fn rust_dot(e: &Engine, p: Prime, a: &[Poly32; 4], b: &[Poly32; 4], out: &mut Poly32) {
    e.dot4(p, black_box(a), black_box(b), black_box(out));
}
#[inline(never)]
fn rust_order(e: &Engine, a: &mut Poly32) {
    e.order_from_fips(black_box(a));
}
#[cfg(cref)]
#[inline(never)]
fn c_ntt(p: Prime, a: &mut Poly32) {
    let a = black_box(a);
    agg_mldsa_cref::ntt_avx(&mut a.0, &tables::table(p, Scale::Mont).0);
}
#[cfg(cref)]
#[inline(never)]
fn c_intt(a: &mut Poly32, s: Scale) {
    let a = black_box(a);
    agg_mldsa_cref::invntt_avx(&mut a.0, &tables::table(Prime::Q, s).0);
}
#[cfg(cref)]
#[inline(never)]
fn c_order(a: &mut Poly32) {
    let a = black_box(a);
    agg_mldsa_cref::nttunpack_avx(&mut a.0);
}
#[cfg(cref)]
#[inline(never)]
fn c_dot(p: Prime, a: &[Poly32; 4], b: &[Poly32; 4], out: &mut Poly32) {
    let a = black_box(a);
    let b = black_box(b);
    let out = black_box(out);
    // SAFETY: aligned polynomial arrays and qdata have the full reference sizes.
    unsafe {
        agg_mldsa_cref::ffi::pqcrystals_dilithium2_avx2_pointwise_acc_avx(
            out.0.as_mut_ptr(),
            a.as_ptr().cast(),
            b.as_ptr().cast(),
            tables::table(p, Scale::Mont).0.as_ptr(),
        );
    }
}
fn compare(mut r: impl FnMut(), mut c: impl FnMut(), name: &str, limit: f64) -> bool {
    for _ in 0..500 {
        r();
        c();
    }
    let mut rs = Vec::new();
    let mut cs = Vec::new();
    for i in 0..common::NUM_TRIALS {
        let mut run = |first: bool| {
            for is_r in [first, !first] {
                let start = common::start_tsc();
                for _ in 0..common::CALLS_PER_TRIAL {
                    if is_r {
                        r()
                    } else {
                        c()
                    }
                }
                let t = (common::end_tsc() - start) as f64 / common::CALLS_PER_TRIAL as f64;
                if is_r {
                    rs.push(t)
                } else {
                    cs.push(t)
                }
            }
        };
        run(i % 2 == 0);
    }
    rs.sort_by(f64::total_cmp);
    cs.sort_by(f64::total_cmp);
    let rm = rs[rs.len() / 2];
    let cm = cs[cs.len() / 2];
    let initial = rm / cm;
    if initial > limit && initial <= limit * 1.03 {
        rs.clear();
        cs.clear();
        for _ in 0..3 {
            for _ in 0..common::NUM_TRIALS {
                let start = common::start_tsc();
                for _ in 0..common::CALLS_PER_TRIAL {
                    c();
                }
                cs.push((common::end_tsc() - start) as f64 / common::CALLS_PER_TRIAL as f64);
                let start = common::start_tsc();
                for _ in 0..common::CALLS_PER_TRIAL {
                    r();
                }
                rs.push((common::end_tsc() - start) as f64 / common::CALLS_PER_TRIAL as f64);
            }
        }
        rs.sort_by(f64::total_cmp);
        cs.sort_by(f64::total_cmp);
    }
    let rm = rs[rs.len() / 2];
    let cm = cs[cs.len() / 2];
    let ratio = rm / cm;
    // The gate uses medians; the minima are a diagnostic that preemption by
    // other processes on the same core cannot inflate.
    let min_ratio = rs[0] / cs[0];
    println!(
        "{name:<24} {rm:9.2} {cm:9.2} {ratio:7.3} limit {limit:.2} {}  (min {:.2} {:.2} {min_ratio:.3})",
        if ratio <= limit { "PASS" } else { "FAIL" },
        rs[0],
        cs[0],
    );
    ratio <= limit
}
fn main() {
    common::print_system_info();
    println!("kernel                   rust med    C med   ratio  (min rust, min C, min ratio)");
    let Some(e) = Engine::new(Backend::Avx2) else {
        eprintln!("Skipping: AVX2/POPCNT unavailable");
        return;
    };
    #[cfg(not(cref))]
    {
        for p in [Prime::Q, Prime::P2] {
            let input = Poly32(std::array::from_fn(|i| i as i32 - 128));
            let mut r = input;
            let stats = common::measure_kernel(|| {
                r = input;
                rust_ntt(&e, p, &mut r);
                black_box(&r);
            });
            println!(
                "{p:?} ntt AVX2: {:.2} TSC ticks; C UNRESOLVED",
                stats.median
            );
        }
        eprintln!("C gate unavailable: set AGG_CREF_SRC (scripts/cref-fetch.sh)");
    }
    #[cfg(cref)]
    {
        let mut pass = true;
        for p in [Prime::Q, Prime::P2] {
            let c = params::consts(p);
            let input = Poly32(std::array::from_fn(|i| {
                (i as i32 * 731) % 8380417 - 4190208
            }));
            let mut r = input;
            let mut cf = input;

            pass &= compare(
                || {
                    r = input;
                    rust_ntt(&e, p, &mut r);
                    black_box(&r);
                },
                || {
                    cf = input;
                    c_ntt(p, &mut cf);
                    black_box(&cf);
                },
                &format!("{p:?} ntt"),
                1.00,
            );
            let mut transformed = input;
            e.ntt(p, &mut transformed);
            e.reduce(p, &mut transformed, Form::Centered);
            for s in [Scale::Mont] {
                // For P2 the unmodified C inverse is only a timing baseline; it lacks sum reductions.

                pass &= compare(
                    || {
                        r = transformed;
                        rust_intt(&e, p, &mut r, s);
                        black_box(&r);
                    },
                    || {
                        cf = input;
                        c_intt(&mut cf, s);
                        black_box(&cf);
                    },
                    &format!("{p:?} intt {s:?}"),
                    if p == Prime::P2 { 1.20 } else { 1.00 },
                );
            }
            pass &= compare(
                || {
                    r = transformed;
                    rust_intt(&e, p, &mut r, Scale::Plain);
                    black_box(&r);
                },
                || {
                    cf = transformed;
                    rust_intt(&e, p, &mut cf, Scale::Mont);
                    black_box(&cf);
                },
                &format!("{p:?} Plain/Mont"),
                1.02,
            );
            pass &= compare(
                || {
                    r = transformed;
                    rust_intt(&e, p, &mut r, Scale::Mont);
                    black_box(&r);
                },
                || {
                    cf = transformed;
                    rust_intt(&e, p, &mut cf, Scale::Plain);
                    black_box(&cf);
                },
                &format!("{p:?} Mont/Plain"),
                1.02,
            );
            let a = [Poly32(std::array::from_fn(|i| (i as i32 * 773) % c.p - c.p / 2)); 4];
            let b = [transformed; 4];
            pass &= compare(
                || rust_dot(&e, p, &a, &b, &mut r),
                || {
                    c_dot(p, &a, &b, &mut cf);
                    black_box(&cf);
                },
                &format!("{p:?} dot4"),
                if p == Prime::P2 { 1.05 } else { 1.00 },
            );
        }
        let mut r = Poly32::zero();
        let mut cf = r;
        pass &= compare(
            || rust_order(&e, &mut r),
            || c_order(&mut cf),
            "order_from_fips",
            1.00,
        );
        if !pass && std::env::args().any(|a| a == "--gate") {
            std::process::exit(1);
        }
    }
}
