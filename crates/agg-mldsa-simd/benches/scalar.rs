//! Scalar gate against the retained witness implementations; times are TSC ticks/call.
mod common;
use agg_mldsa_simd::params::*;
use agg_mldsa_simd::{
    Backend, Challenge39, Engine, ExpandedA, FinishOut, FinishRow, Form, Poly32, Prime, Scale,
};
use agg_witness::{crt, expand, native, ntt::NttPrime};
use std::hint::black_box;

#[inline(never)]
fn forward(e: &Engine, p: Prime, input: Poly32) {
    let mut a = black_box(input);
    e.ntt(p, &mut a);
    black_box(a);
}
#[inline(never)]
fn inverse(e: &Engine, p: Prime, input: Poly32) {
    let mut a = black_box(input);
    e.intt(p, &mut a, Scale::Plain, Form::Canonical);
    black_box(a);
}
#[inline(never)]
fn old_forward(n: &NttPrime, input: &[u64; 256]) {
    black_box(n.ntt(black_box(input)));
}
#[inline(never)]
fn old_inverse(n: &NttPrime, input: &[u64; 256]) {
    black_box(n.intt(black_box(input)));
}
#[inline(never)]
fn new_expand(e: &Engine, rho: &[[u8; 32]; 1], out: &mut [ExpandedA; 1]) {
    e.expand_a_many(black_box(rho), black_box(out));
}
#[inline(never)]
fn old_expand(rho: &[u8; 32]) {
    black_box(expand::expand_a(black_box(rho)));
}
#[inline(never)]
fn new_ball(e: &Engine, seed: &[[u8; 32]; 1], out: &mut [Challenge39; 1]) {
    e.sample_in_ball_many(black_box(seed), black_box(out));
}
#[inline(never)]
fn old_ball(seed: &[u8; 32]) {
    black_box(native::sample_in_ball(black_box(seed)));
}
#[inline(never)]
fn new_shake(e: &Engine, input: &[u8; 832], out: &mut [u8; 32]) {
    e.shake256_many(&[[black_box(input), &[], &[], &[]]], 32, black_box(out));
}
#[inline(never)]
fn old_shake(input: &[u8; 832], out: &mut [u8; 32]) {
    native::shake256(&[black_box(input)], black_box(out));
}
#[inline(never)]
fn new_finish(
    e: &Engine,
    xq: &[Poly32; 4],
    xp: &[Poly32; 4],
    t: &[[i32; 256]; 4],
    h: &[[u8; 256]; 4],
) {
    let mut out = FinishOut::new();
    for r in 0..4 {
        black_box(e.finish_row(
            FinishRow {
                xq: &xq[r],
                xp: &xp[r],
                t: &t[r],
                h: &h[r],
            },
            &mut out,
        ))
        .unwrap();
        black_box(&out);
    }
}
#[inline(never)]
fn old_finish(xq: &[[u64; 256]; 4], xp: &[[u64; 256]; 4], t: &[[i32; 256]; 4], h: &[[u8; 256]; 4]) {
    let mut out = [(0i64, 0i64, 0i64); 1024];
    for r in 0..4 {
        for j in 0..256 {
            let az = crt::crt_centered(xq[r][j], xp[r][j]);
            let w = native::ctr_q(az - t[r][j] as i64);
            let w1 = native::use_hint(h[r][j] != 0, w);
            let u = native::ctr_q(w - ALPHA as i64 * w1);
            let delta = az - t[r][j] as i64 - ALPHA as i64 * w1 - u;
            assert_eq!(delta.rem_euclid(Q as i64), 0);
            out[r * 256 + j] = (w1, u, delta / Q as i64);
        }
    }
    black_box(out);
}
fn main() {
    common::print_system_info();
    println!("Reference column c denotes retained Rust, not C, in this scalar gate.");
    let e = Engine::new(Backend::Scalar).unwrap();
    for p in [Prime::Q, Prime::P2] {
        let c = p.consts();
        let n = NttPrime::new(c.p as u64, c.zeta as u64);
        let input = Poly32(std::array::from_fn(|i| (i as i32 * 7919) % Q - Q / 2));
        let old = input.0.map(|x| (x as i64).rem_euclid(c.p as i64) as u64);
        common::compare(
            &format!("ntt_{p:?}"),
            "scalar",
            || forward(&e, p, input),
            || old_forward(&n, &old),
            true,
        );
        let mut hat = input;
        e.ntt(p, &mut hat);
        e.reduce(p, &mut hat, Form::Canonical);
        let old = hat.0.map(|x| x as u64);
        if p == Prime::P2 {
            e.reduce(p, &mut hat, Form::Centered);
        }
        common::compare(
            &format!("intt_{p:?}"),
            "scalar",
            || inverse(&e, p, hat),
            || old_inverse(&n, &old),
            true,
        );
    }
    let rho = [[42u8; 32]];
    let mut expanded = [ExpandedA::new()];
    common::compare(
        "expand_a/key",
        "scalar",
        || new_expand(&e, &rho, &mut expanded),
        || old_expand(&rho[0]),
        true,
    );
    let mut ball = [Challenge39::new()];
    common::compare(
        "sample_in_ball",
        "scalar",
        || new_ball(&e, &rho, &mut ball),
        || old_ball(&rho[0]),
        true,
    );
    let input = [42u8; 832];
    let mut output = [0u8; 32];
    let mut old_output = [0u8; 32];
    common::compare(
        "shake256_832",
        "scalar",
        || new_shake(&e, &input, &mut output),
        || old_shake(&input, &mut old_output),
        true,
    );
    let az: [[i64; 256]; 4] = std::array::from_fn(|r| {
        std::array::from_fn(|i| (i as i64 * 333333333 + r as i64 * 777777777) - 42000000000)
    });
    let xq = az.map(|a| Poly32(a.map(|x| x.rem_euclid(Q as i64) as i32)));
    let xp = az.map(|a| Poly32(a.map(|x| x.rem_euclid(P2 as i64) as i32)));
    let oldq = xq.map(|a| a.0.map(|x| x as u64));
    let oldp = az.map(|a| a.map(|x| x.rem_euclid(crt::prime2().p as i64) as u64));
    let t = std::array::from_fn(|r| {
        std::array::from_fn(|i| (i as i32 * 177777 + r as i32 * 11111) - B_T / 2)
    });
    let h = std::array::from_fn(|r| std::array::from_fn(|i| ((i + r) % 2) as u8));
    common::compare(
        "finish/record",
        "scalar",
        || {
            new_finish(
                &e,
                black_box(&xq),
                black_box(&xp),
                black_box(&t),
                black_box(&h),
            )
        },
        || {
            old_finish(
                black_box(&oldq),
                black_box(&oldp),
                black_box(&t),
                black_box(&h),
            )
        },
        true,
    );
    common::finish_benchmarks();
}
