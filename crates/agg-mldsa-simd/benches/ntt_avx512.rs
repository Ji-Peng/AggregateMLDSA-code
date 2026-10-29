mod common;
use agg_mldsa_simd::{
    dispatch::{Backend, Engine},
    ntt::{Form, Scale},
    params::Prime,
    poly::Poly32,
};
use std::hint::black_box;
#[cfg_attr(not(cref), allow(dead_code))]
#[derive(Clone, Copy)]
enum Op {
    Ntt,
    Intt,
    Order,
    Dot,
}
#[inline(never)]
fn rust(e: &Engine, p: Prime, op: Op, input: &Poly32, a: &[Poly32; 4], b: &[Poly32; 4]) {
    let mut x = *input;
    match op {
        Op::Ntt => e.ntt(p, &mut x),
        Op::Intt => e.intt(p, &mut x, Scale::Mont, Form::Canonical),
        Op::Order => e.order_from_fips(&mut x),
        Op::Dot => e.dot4(p, a, b, &mut x),
    }
    black_box(x);
}
#[cfg(cref)]
#[inline(never)]
fn c(op: Op, input: &Poly32, a: &[Poly32; 4], b: &[Poly32; 4]) {
    let mut x = *input;
    // SAFETY: Called only after AVX-512 engine detection; all inputs and outputs
    // are 64-byte aligned polynomials containing 256 initialized coefficients.
    unsafe {
        match op {
            Op::Ntt => agg_mldsa_cref::ffi::hyb512_ntt_so_avx512(x.0.as_mut_ptr()),
            Op::Intt => agg_mldsa_cref::ffi::hyb512_intt_so_avx512(x.0.as_mut_ptr()),
            Op::Order => agg_mldsa_cref::ffi::hyb512_shuffle(x.0.as_mut_ptr()),
            Op::Dot => agg_mldsa_cref::ffi::hyb512_pointwise_avx512(
                x.0.as_mut_ptr(),
                a[0].0.as_ptr(),
                b[0].0.as_ptr(),
            ),
        }
    }
    black_box(x);
}
// The inverse gate includes Mont scaling and canonical output, as used by
// witness derivation. Every transform starts from the same bounded input.
fn main() {
    common::print_system_info();
    let Some(e) = Engine::new(Backend::Avx512) else {
        println!("AVX-512 unavailable; skipped");
        return;
    };
    let input = Poly32(core::array::from_fn(|i| {
        ((i * 7919) % 8_000_000) as i32 - 4_000_000
    }));
    let a = [Poly32(core::array::from_fn(|i| input[i].rem_euclid(8_380_417))); 4];
    let mut bhat = input;
    e.ntt(Prime::Q, &mut bhat);
    let b = [bhat; 4];
    // Without the C rows the ratio gates cannot be checked, so a gated run fails.
    let mut passed = cfg!(cref);
    #[cfg(cref)]
    {
        for (name, op) in [
            ("q ntt", Op::Ntt),
            ("q intt", Op::Intt),
            ("order_from_fips", Op::Order),
        ] {
            passed &= budget(
                name,
                1.0,
                || rust(&e, Prime::Q, op, &input, &a, &b),
                || c(op, &input, &a, &b),
            );
        }
        passed &= budget(
            "q dot4 / C pointwise",
            4.0,
            || rust(&e, Prime::Q, Op::Dot, &input, &a, &b),
            || c(Op::Dot, &input, &a, &b),
        );
        let pa = [input; 4];
        let mut pb = input;
        e.ntt(Prime::P2, &mut pb);
        let pb = [pb; 4];
        passed &= budget(
            "P2 dot4 / C pointwise",
            4.0,
            || rust(&e, Prime::P2, Op::Dot, &input, &pa, &pb),
            || c(Op::Dot, &input, &a, &b),
        );
    }
    #[cfg(not(cref))]
    println!("C rows unavailable: export AGG_CREF_SRC (scripts/cref-fetch.sh) and rebuild");
    passed &= budget(
        "P2 ntt / q ntt",
        1.02,
        || rust(&e, Prime::P2, Op::Ntt, &input, &a, &b),
        || rust(&e, Prime::Q, Op::Ntt, &input, &a, &b),
    );
    #[cfg(cref)]
    {
        passed &= budget(
            "P2 intt / C intt",
            1.20,
            || rust(&e, Prime::P2, Op::Intt, &input, &a, &b),
            || c(Op::Intt, &input, &a, &b),
        );
    }
    if std::env::args().any(|s| s == "--avx2") {
        if let Some(avx2) = Engine::new(Backend::Avx2) {
            for prime in [Prime::Q, Prime::P2] {
                passed &= budget(
                    &format!("{prime:?} dot4 / Rust AVX2"),
                    0.60,
                    || rust(&e, prime, Op::Dot, &input, &a, &b),
                    || rust(&avx2, prime, Op::Dot, &input, &a, &b),
                );
            }
        }
    } else {
        println!("AVX2 dot4 budget unresolved; pass --avx2 after that backend is implemented");
    }
    if !passed && std::env::args().any(|s| s == "--gate") {
        std::process::exit(1);
    }
}
fn trial(f: &mut impl FnMut()) -> f64 {
    let start = common::start_tsc();
    for _ in 0..common::CALLS_PER_TRIAL {
        f();
    }
    (common::end_tsc() - start) as f64 / common::CALLS_PER_TRIAL as f64
}
fn med(a: &mut [f64]) -> f64 {
    a.sort_by(f64::total_cmp);
    a[a.len() / 2]
}
fn budget(name: &str, limit: f64, mut r: impl FnMut(), mut c: impl FnMut()) -> bool {
    for _ in 0..500 {
        r();
        c();
    }
    let c_first = std::env::args().any(|s| s == "--c-first");
    let mut rt = Vec::with_capacity(124);
    let mut ct = Vec::with_capacity(124);
    for _ in 0..common::NUM_TRIALS {
        if c_first {
            ct.push(trial(&mut c));
            rt.push(trial(&mut r));
        } else {
            rt.push(trial(&mut r));
            ct.push(trial(&mut c));
        }
    }
    let mut rmed = med(&mut rt);
    let mut cmed = med(&mut ct);
    let mut ratio = rmed / cmed;
    if ratio > limit && ratio <= limit * 1.03 {
        // Borderline results use three pooled repeats with swapped call order.
        rt.clear();
        ct.clear();
        for _ in 0..3 * common::NUM_TRIALS {
            if c_first {
                rt.push(trial(&mut r));
                ct.push(trial(&mut c));
            } else {
                ct.push(trial(&mut c));
                rt.push(trial(&mut r));
            }
        }
        rmed = med(&mut rt);
        cmed = med(&mut ct);
        ratio = rmed / cmed;
        println!("{name}: pooled swapped-order retry");
    }
    println!(
        "{name}: Rust {rmed:.2}, reference {cmed:.2}, ratio {ratio:.3}, budget {limit:.2}: {}",
        if ratio <= limit { "PASS" } else { "FAIL" }
    );
    ratio <= limit
}
