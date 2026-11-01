use agg_simd::{dot, fold, fp, k, scalar, F, K, P};
use jolt_field::Ring;
use std::hint::black_box as bb;
#[cfg(target_arch = "x86_64")]
fn ticks() -> u64 {
    unsafe {
        std::arch::x86_64::_mm_lfence();
        std::arch::x86_64::_rdtsc()
    }
}
#[cfg(target_arch = "x86_64")]
fn measure(mut f: impl FnMut(), n: usize) -> f64 {
    let reps = (1usize << 20).div_ceil(n).max(2);
    let mut samples = Vec::new();
    f();
    for _ in 0..5 {
        let start = ticks();
        for _ in 0..reps {
            f();
        }
        samples.push((ticks() - start) as f64 / (n * reps) as f64);
    }
    samples.sort_by(f64::total_cmp);
    samples[2]
}
#[test]
#[ignore = "kernel timing benchmark; run with: cargo test --release -p agg-simd --test timing -- --ignored --nocapture"]
#[cfg(target_arch = "x86_64")]
fn timing() {
    println!(
        "backend={:?}; invariant TSC ticks per element/term (direction only)",
        agg_simd::detect::backend()
    );
    for n in [1 << 10, 1 << 14, 1 << 20] {
        let a: Vec<_> = (0..n)
            .map(|i| (i as u64).wrapping_mul(0x9e3779b97f4a7c15) % P)
            .collect();
        let b: Vec<_> = a.iter().map(|&x| P - 1 - x).collect();
        let ap = (&a[..], &b[..]);
        let bp = (&b[..], &a[..]);
        let ints: Vec<_> = a.iter().map(|&x| (x as i64) >> 14).collect();
        let low: Vec<_> = a.iter().map(|x| x & ((1 << 52) - 1)).collect();
        let r = K::new(F::from_u64(77), F::from_u64(P - 1));
        let mut out = vec![0; n];
        let mut out1 = vec![0; n];
        macro_rules! bench {
            ($name:expr,$vec:expr,$scalar:expr) => {{
                let v = measure(
                    || {
                        bb($vec);
                    },
                    n,
                );
                let s = measure(
                    || {
                        bb($scalar);
                    },
                    n,
                );
                println!(
                    "{} n={} vector={:.3} scalar={:.3} ratio={:.2}x",
                    $name,
                    n,
                    v,
                    s,
                    s / v
                );
            }};
        }
        bench!(
            "fp_mul",
            fp::fp_mul(bb(&a), bb(&b), bb(&mut out)),
            scalar::fp_mul(bb(&a), bb(&b), bb(&mut out))
        );
        bench!(
            "k_mul",
            k::k_mul(bb(ap), bb(bp), bb((&mut out, &mut out1))),
            scalar::k_mul(bb(ap), bb(bp), bb((&mut out, &mut out1)))
        );
        bench!(
            "k_base",
            k::k_mul_base(bb(ap), bb(&a), bb((&mut out, &mut out1))),
            scalar::k_mul_base(bb(ap), bb(&a), bb((&mut out, &mut out1)))
        );
        bench!(
            "fp_dot",
            dot::fp_dot(bb(&a), bb(&b)),
            scalar::fp_dot(bb(&a), bb(&b))
        );
        bench!(
            "k_dot",
            dot::k_dot(bb(ap), bb(bp)),
            scalar::k_dot(bb(ap), bb(bp))
        );
        bench!(
            "base_dot",
            dot::k_base_dot(bb(ap), bb(&a)),
            scalar::k_base_dot(bb(ap), bb(&a))
        );
        bench!(
            "int_dot",
            dot::k_int_dot(bb(ap), bb(&ints)),
            scalar::k_int_dot(bb(ap), bb(&ints))
        );
        bench!(
            "int2_dot",
            dot::k_int2_dot(bb(ap), bb(&low), bb(&ints)),
            scalar::k_int2_dot(bb(ap), bb(&low), bb(&ints))
        );
        bench!(
            "k_fold",
            fold::k_fold(r, bb(ap), bb(bp), bb((&mut out, &mut out1))),
            scalar::k_fold(r, bb(ap), bb(bp), bb((&mut out, &mut out1)))
        );
        bench!(
            "int_fold",
            fold::int_fold(r, bb(&ints), bb(&ints), bb((&mut out, &mut out1))),
            scalar::int_fold(r, bb(&ints), bb(&ints), bb((&mut out, &mut out1)))
        );
    }
}
