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

/// The sum-check and fraction-tree kernels of the plane layout, on the backend `AGG_ISA`
/// selects (run once per tier, `AGG_ISA=scalar` for the portable reference): ticks per item
/// (an output row or a leaf quartet), single thread.
#[test]
#[ignore = "kernel timing benchmark; run with: AGG_ISA=<tier> cargo test --release -p agg-simd --test timing engine -- --ignored --nocapture"]
#[cfg(target_arch = "x86_64")]
fn engine_timing() {
    use agg_simd::{
        buf::KPlanes,
        fused::{eval_items, fold_rows, Terms},
        lookup::{bottoms, parent, Layer},
        pattern::{fill_rows, prefix_sums, Symmetric},
    };
    println!("backend={:?}; ticks per item (direction only)", agg_simd::detect::backend());
    let rk = |i: u64| K::new(F::from_u64(i.wrapping_mul(0x9e3779b97f4a7c15) % P), F::from_u64(i.wrapping_mul(0xbf58476d1ce4e5b9) % P));
    let rows = 1 << 12;
    let items = rows / 2;
    let width = 6;
    let planes: Vec<KPlanes> = (0..width).map(|l| KPlanes::from_fn(rows, |i| rk((l * rows + i) as u64))).collect();
    let lanes: Vec<(&[u64], &[u64])> = planes.iter().map(|p| p.planes()).collect();
    let weights = KPlanes::from_fn(items, |i| rk(i as u64 + 7));
    // A degree-3 stage group: two linear terms and three products over four slots.
    let linear = [(0usize, 0usize), (5, 3)];
    let products = [(0usize, 1usize, 1usize), (2, 3, 2), (4, 4, 3)];
    let terms = Terms { linear: &linear, products: &products, slots: 4 };
    let mut out = vec![[K::from_u64(0); 3]; 4];
    let t = measure(|| eval_items(bb(&lanes), terms, Some(weights.planes()), 0, items, true, bb(&mut out)), items);
    println!("eval_items weighted full (6 lanes, 3 products) {t:.2}");
    let t = measure(|| eval_items(bb(&lanes), terms, None, 0, items, false, bb(&mut out)), items);
    println!("eval_items unweighted half (6 lanes, 3 products) {t:.2}");
    let mut folded: Vec<KPlanes> = (0..width).map(|_| KPlanes::from_fn(items, |_| K::from_u64(0))).collect();
    let addresses: Vec<(usize, usize)> = folded
        .iter_mut()
        .map(|o| {
            let (a, b) = o.planes_mut();
            (a.as_mut_ptr() as usize, b.as_mut_ptr() as usize)
        })
        .collect();
    let r = rk(99);
    let t = measure(|| unsafe { fold_rows(r, bb(&lanes), &addresses, 0, items) }, items * width);
    println!("fold_rows per lane row {t:.2}");
    let n = 1 << 13;
    let layer = Layer { p: KPlanes::from_fn(2 * n, |i| rk(i as u64)), q: KPlanes::from_fn(2 * n, |i| rk(i as u64 + 3)) };
    let t = measure(|| { bb(parent(bb(&layer))); }, n);
    println!("lookup parent per node {t:.2}");
    let leaves: Vec<i32> = (0..4 * n).map(|i| ((i * 313 + 17) % 65537) as i32 - 32768).collect();
    let x = rk(5);
    let powers = [[x, x * x, x * x * x, x * x * x * x]];
    let t = measure(|| { bb(bottoms(bb(&leaves), 0, &powers)); }, n);
    println!("lookup bottoms per node (one challenge) {t:.2}");
    let sym = Symmetric { stacked: &leaves, center: 0 };
    let first: Vec<K> = (0..64).map(|i| rk(i as u64)).collect();
    let second: Vec<K> = (0..n / 4 / 64).map(|i| rk(i as u64 + 1000)).collect();
    let t = measure(|| { bb(prefix_sums(bb(&sym), &first, &second)); }, n / 4);
    println!("pattern prefix_sums symmetric per row {t:.2}");
    let mut filled: Vec<KPlanes> = (0..4).map(|_| KPlanes::from_fn(n / 4, |_| K::from_u64(0))).collect();
    let addresses: Vec<(usize, usize)> = filled
        .iter_mut()
        .map(|o| {
            let (a, b) = o.planes_mut();
            (a.as_mut_ptr() as usize, b.as_mut_ptr() as usize)
        })
        .collect();
    let e = [rk(1), rk(2), rk(3), rk(4)];
    let t = measure(|| unsafe { fill_rows(bb(&sym), e, &addresses, 0, n / 4) }, n / 4);
    println!("pattern fill_rows symmetric per row {t:.2}");
}
