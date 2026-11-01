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

/// Whole fused passes (fold by a challenge, then the split-weighted evaluation) on the term
/// shapes of the prover's large passes, on the backend `AGG_ISA` selects: ticks per output
/// row, single thread (`RAYON_NUM_THREADS=1`).
#[test]
#[ignore = "pass timing benchmark; run with: RAYON_NUM_THREADS=1 AGG_ISA=<tier> cargo test --release -p agg-simd --test timing passes -- --ignored --nocapture"]
#[cfg(target_arch = "x86_64")]
fn passes_timing() {
    use agg_simd::{
        buf::KPlanes,
        fused::{fused_pass, Eval, Fill, Terms},
        pattern::{fill_combined, Columns},
    };
    println!("backend={:?}; ticks per output row (direction only)", agg_simd::detect::backend());
    let rk = |i: u64| K::new(F::from_u64(i.wrapping_mul(0x9e3779b97f4a7c15) % P), F::from_u64(i.wrapping_mul(0xbf58476d1ce4e5b9) % P));
    let rows_out = 1 << 13;
    // (name, width, linear, products, slots, first, full, weighted)
    // Two fraction trees: `gamma_p (P_0 Q_1 + P_1 Q_0) + gamma_q Q_0 Q_1` each.
    const LAYER: &[(usize, usize, usize)] = &[(0, 3, 0), (1, 2, 0), (2, 3, 1), (4, 7, 2), (5, 6, 2), (6, 7, 3)];
    type Shape = (&'static str, usize, &'static [(usize, usize)], &'static [(usize, usize, usize)], usize, usize, bool);
    let shapes: [Shape; 7] = [
        ("layer 8 lanes, 6 products / 4 slots", 8, &[], LAYER, 4, 128, false),
        ("layer full", 8, &[], LAYER, 4, 128, true),
        ("linear 1 lane in slot 4 of 5", 1, &[(0, 4)], &[], 5, 64, false),
        ("tail 11 lanes", 11, &[(0, 0), (1, 1), (2, 3), (3, 3), (4, 3), (5, 3), (6, 3), (7, 3)], &[(8, 9, 0), (9, 10, 2)], 4, 64, false),
        ("leaf 2 lanes, 2 linear + 1 product", 2, &[(0, 0), (1, 0)], &[(0, 1, 1)], 2, 64, false),
        ("6 linear, 1 slot", 6, &[(0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (5, 0)], &[], 1, 32, false),
        ("weight 2 lanes, 1 product, no eq", 2, &[], &[(0, 1, 0)], 1, 0, false),
    ];
    for (name, width, linear, products, slots, first, full) in shapes {
        let input: Vec<KPlanes> = (0..width).map(|l| KPlanes::from_fn(2 * rows_out, |i| rk((l * 2 * rows_out + i) as u64))).collect();
        let lanes: Vec<(&[u64], &[u64])> = input.iter().map(|p| p.planes()).collect();
        let items = rows_out / 2;
        let first_planes = KPlanes::from_fn(first.max(1), |i| rk(i as u64 + 11));
        let second: Vec<K> = (0..items / first.max(1)).map(|i| rk(i as u64 + 1000)).collect();
        // The evaluation-only pass reads the unfolded input: twice the items.
        let second_in: Vec<K> = (0..2 * items / first.max(1)).map(|i| rk(i as u64 + 1000)).collect();
        let terms = Terms { linear, products, slots };
        let mut out = vec![0_u64; 2 * width * rows_out];
        let r = rk(77);
        let t = measure(
            || {
                let eval = Eval {
                    terms,
                    first: if first == 0 { (&[][..], &[][..]) } else { first_planes.planes() },
                    second: if first == 0 { &[] } else { &second },
                    full,
                };
                bb(fused_pass(Some((bb(&mut out[..]), width, rows_out)), &lanes, Some(Fill::<Columns>::Fold(r, &lanes)), Some(eval)));
            },
            rows_out,
        );
        println!("pass {name}: {t:.1}");
        let t = measure(
            || {
                let eval = Eval {
                    terms,
                    first: if first == 0 { (&[][..], &[][..]) } else { first_planes.planes() },
                    second: if first == 0 { &[] } else { &second_in },
                    full,
                };
                bb(fused_pass(None, bb(&lanes), None::<Fill<'_, Columns>>, Some(eval)));
            },
            2 * rows_out,
        );
        println!("  evaluation only: {t:.1}");
    }
    // The layer below the stored ones, computed from the leaves (two trees): round 0
    // evaluates it, round 1 folds it.
    {
        use agg_simd::lookup::UpperRows;
        let leaves: Vec<i32> = (0..8 * rows_out).map(|i| ((i * 313 + 17) % 65537) as i32 - 32768).collect();
        let x = [rk(5), rk(6)];
        let powers: Vec<[K; 4]> = x.iter().map(|&x| [x, x * x, x * x * x, x * x * x * x]).collect();
        let upper = UpperRows::new(&leaves, 0, &powers);
        let terms = Terms { linear: &[], products: LAYER, slots: 4 };
        let first_planes = KPlanes::from_fn(256, |i| rk(i as u64 + 11));
        let second: Vec<K> = (0..rows_out / 2 / 256).map(|i| rk(i as u64 + 1000)).collect();
        let t = measure(
            || {
                let eval = Eval { terms, first: first_planes.planes(), second: &second, full: true };
                bb(fused_pass(None, &[], Some(Fill::<Columns>::Computed(None, &upper)), Some(eval)));
            },
            rows_out,
        );
        println!("computed layer, evaluation (round 0): {t:.1}");
        let first_planes = KPlanes::from_fn(128, |i| rk(i as u64 + 11));
        let second: Vec<K> = (0..rows_out / 4 / 128).map(|i| rk(i as u64 + 1000)).collect();
        let mut out = vec![0_u64; 2 * 8 * rows_out / 2];
        let t = measure(
            || {
                let eval = Eval { terms, first: first_planes.planes(), second: &second, full: false };
                bb(fused_pass(Some((bb(&mut out[..]), 8, rows_out / 2)), &[], Some(Fill::<Columns>::Computed(Some(rk(9)), &upper)), Some(eval)));
            },
            rows_out / 2,
        );
        println!("computed layer, fold and evaluation (round 1): {t:.1}");
        use agg_simd::fused::RowSource;
        let block = 1024;
        let mut scratch = vec![0_u64; 2 * 8 * block];
        let base = scratch.as_mut_ptr() as usize;
        let lanes_out: Vec<(usize, usize)> = (0..8).map(|l| (base + 16 * l * block, base + 8 * (2 * l + 1) * block)).collect();
        let t = measure(
            || {
                for row in (0..rows_out).step_by(block) {
                    unsafe { upper.fill(row, row + block, bb(&lanes_out)) };
                }
            },
            rows_out,
        );
        println!("computed layer rows only (two trees): {t:.1}");
    }
    let n = 1 << 14;
    let cells: Vec<Vec<i32>> = (0..2).map(|c| (0..4 * n).map(|i| ((i * 313 + 17 * c + 5) % 65537) as i32 - 32768).collect()).collect();
    let columns = Columns { lanes: cells.iter().map(Vec::as_slice).collect() };
    // Four weight rows as the linear group passes them: the rows past its columns are zero.
    let weights: Vec<[K; 4]> = (0..4).map(|l| std::array::from_fn(|a| if l < 2 { rk((4 * l + a) as u64 + 50) } else { K::from_u64(0) })).collect();
    let mut plane = KPlanes::from_fn(n, |_| K::from_u64(0));
    let (c0, c1) = plane.planes_mut();
    let lane = (c0.as_mut_ptr() as usize, c1.as_mut_ptr() as usize);
    let t = measure(|| unsafe { fill_combined(bb(&columns), &weights, lane, 0, n) }, n);
    println!("pattern fill_combined 2 columns per row {t:.2}");
}
