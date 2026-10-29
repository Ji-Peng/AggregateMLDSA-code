//! Run C comparisons with AGG_CREF_SRC set and RUSTFLAGS='--cfg cref'.
#![allow(unexpected_cfgs)]
mod common;
use agg_mldsa_simd::{poly, Backend, Block, Engine, Sink, Xof, XofJob};
use std::hint::black_box;
#[allow(dead_code)]
#[path = "../src/keccak/avx2.rs"]
mod avx2;
#[allow(dead_code)]
#[path = "../src/keccak/avx512.rs"]
mod avx512;
#[allow(dead_code)]
#[path = "../src/keccak/lanes.rs"]
mod lanes;

#[inline(never)]
fn rust4(state: &mut lanes::State<4>) {
    // SAFETY: Only called when an AVX2 engine exists.
    unsafe { avx2::permute(black_box(state)) }
}
#[inline(never)]
fn rust8(state: &mut lanes::State<8>) {
    // SAFETY: Only called when an AVX-512 engine exists.
    unsafe { avx512::permute(black_box(state)) }
}
struct CountSink<'a> {
    counts: &'a mut [usize],
}
impl Sink for CountSink<'_> {
    #[inline]
    fn block(&mut self, job: usize, block: &Block, _: usize) -> bool {
        black_box(block);
        self.counts[job] -= 1;
        self.counts[job] == 0
    }
}
/// Runs the jobs through this binary's copy of the backend, so the scheduler
/// calls the same permutation symbol (same code alignment) as the bare rows.
#[inline(never)]
fn shake(engine: &Engine, jobs: &[XofJob<'_>], counts: &mut [usize], limits: &[usize]) {
    counts[..limits.len()].copy_from_slice(limits);
    let sink = &mut CountSink { counts };
    let jobs = black_box(jobs);
    match engine.backend() {
        // SAFETY: The engine exists, so the backend's features were detected.
        Backend::Avx2 => unsafe { avx2::run_xof(jobs, sink) },
        // SAFETY: As above.
        Backend::Avx512 => unsafe { avx512::run_xof(jobs, sink) },
        Backend::Scalar => engine.run_xof(jobs, sink),
    }
}

// Alternate the scheduler and its bare-permutation baseline at the same frequency.
fn paired(mut rust: impl FnMut(), mut baseline: impl FnMut()) -> (f64, f64) {
    rust();
    baseline();
    let mut r = [0.0f64; common::NUM_TRIALS];
    let mut c = [0.0f64; common::NUM_TRIALS];
    for i in 0..common::NUM_TRIALS {
        let measure = |f: &mut dyn FnMut()| {
            let start = common::start_tsc();
            for _ in 0..common::CALLS_PER_TRIAL {
                f();
            }
            (common::end_tsc() - start) as f64 / common::CALLS_PER_TRIAL as f64
        };
        if i % 2 == 0 {
            r[i] = measure(&mut rust);
            c[i] = measure(&mut baseline);
        } else {
            c[i] = measure(&mut baseline);
            r[i] = measure(&mut rust);
        }
    }
    r.sort_by(f64::total_cmp);
    c.sort_by(f64::total_cmp);
    (r[common::NUM_TRIALS / 2], c[common::NUM_TRIALS / 2])
}

/// Strict C gate with alternating order and the actual pooled ratio reported.
#[cfg(cref)]
fn compare(name: &str, backend: &str, mut rust: impl FnMut(), mut c: impl FnMut()) -> bool {
    for _ in 0..500 {
        rust();
        c();
    }
    let time = |f: &mut dyn FnMut()| {
        let start = common::start_tsc();
        for _ in 0..common::CALLS_PER_TRIAL {
            f();
        }
        (common::end_tsc() - start) as f64 / common::CALLS_PER_TRIAL as f64
    };
    let mut collect = |swapped: bool, rs: &mut Vec<f64>, cs: &mut Vec<f64>| {
        for i in 0..common::NUM_TRIALS {
            if (i % 2 == 0) != swapped {
                rs.push(time(&mut rust));
                cs.push(time(&mut c));
            } else {
                cs.push(time(&mut c));
                rs.push(time(&mut rust));
            }
        }
    };
    let (mut rs, mut cs) = (Vec::new(), Vec::new());
    collect(false, &mut rs, &mut cs);
    rs.sort_by(f64::total_cmp);
    cs.sort_by(f64::total_cmp);
    let (r, reference) = (rs[rs.len() / 2], cs[cs.len() / 2]);
    let initial = r / reference;
    let mut ratio = initial;
    if initial > 1.00 && initial <= 1.03 {
        rs.clear();
        cs.clear();
        for _ in 0..3 {
            collect(true, &mut rs, &mut cs);
        }
        rs.sort_by(f64::total_cmp);
        cs.sort_by(f64::total_cmp);
        ratio = rs[rs.len() / 2] / cs[cs.len() / 2];
    }
    let pass = ratio <= 1.00;
    println!(
        "{name:<20} {backend:<10} {r:>10.2} {reference:>10.2} {ratio:>8.3} {:>12} (initial {initial:.3})",
        if pass { "PASS" } else { "FAIL" }
    );
    pass
}

// Median of per-pair ratios over many short alternating trials; frequency drift
// and load on the sibling hyperthread between neighbouring trials cancel.
fn pair_ratio(calls: usize, mut a: impl FnMut(), mut b: impl FnMut()) -> f64 {
    let mut ratios = Vec::with_capacity(401);
    let time = |f: &mut dyn FnMut()| {
        let start = common::start_tsc();
        for _ in 0..calls {
            f();
        }
        (common::end_tsc() - start) as f64
    };
    for _ in 0..50 {
        a();
        b();
    }
    for i in 0..401 {
        let (x, y) = if i % 2 == 0 {
            let x = time(&mut a);
            (x, time(&mut b))
        } else {
            let y = time(&mut b);
            (time(&mut a), y)
        };
        ratios.push(x / y);
    }
    ratios.sort_by(f64::total_cmp);
    ratios[200]
}

/// Eight keys: per key one 1312-byte SHAKE256 job squeezing one block and sixteen
/// 34-byte SHAKE128 jobs squeezing five or six blocks.
fn mixed_jobs<'a>(seed: &'a [u8; 34], pk: &'a [u8; 1312]) -> (Vec<XofJob<'a>>, Vec<usize>) {
    let mut jobs = Vec::new();
    let mut limits = Vec::new();
    for key in 0..8 {
        jobs.push(XofJob {
            xof: Xof::Shake256,
            input: [pk, &[], &[], &[]],
        });
        limits.push(1);
        for entry in 0..16 {
            jobs.push(XofJob {
                xof: Xof::Shake128,
                input: [seed, &[], &[], &[]],
            });
            limits.push(5 + (key + entry) % 2);
        }
    }
    (jobs, limits)
}

fn absorb_dry<const W: usize>(
    lanes: &mut [lanes::Lane; W],
    jobs: &[XofJob<'_>],
    _: &mut lanes::State<W>,
    mask: u8,
) -> u8 {
    lanes::absorb_with(lanes, jobs, mask, |_, _, _| {})
}

/// Runs the scheduler with no-op kernels: its progress does not depend on the
/// permutation's output.
fn dry_run(
    w: usize,
    jobs: &[XofJob<'_>],
    counts: &mut [usize],
    limits: &[usize],
) -> (usize, usize) {
    counts[..limits.len()].copy_from_slice(limits);
    let mut sink = CountSink { counts };
    if w == 4 {
        lanes::run::<4, true, _, _>(
            jobs,
            &mut sink,
            lanes::Kernel {
                permute: |_: &mut lanes::State<4>| {},
                extract: |_: &lanes::State<4>, _: &mut [Block; 4], _| {},
                clear: |_: &mut lanes::State<4>, _| {},
                absorb: absorb_dry,
            },
        )
    } else {
        lanes::run::<8, true, _, _>(
            jobs,
            &mut sink,
            lanes::Kernel {
                permute: |_: &mut lanes::State<8>| {},
                extract: |_: &lanes::State<8>, _: &mut [Block; 8], _| {},
                clear: |_: &mut lanes::State<8>, _| {},
                absorb: absorb_dry,
            },
        )
    }
}

fn scheduler(engine: &Engine, w: usize) -> bool {
    let seed = [42u8; 34];
    let pk = [73u8; 1312];
    let (jobs, limits) = mixed_jobs(&seed, &pk);
    let mut counts = vec![0; jobs.len()];
    let (used, capacity) = dry_run(w, &jobs, &mut counts, &limits);
    let mut s4 = lanes::State([[0; 4]; 25]);
    let mut s8 = lanes::State([[0; 8]; 25]);
    let mut bare = || {
        for _ in 0..capacity / w {
            if w == 4 {
                rust4(&mut s4);
            } else {
                rust8(&mut s8);
            }
        }
    };
    let (time, _) = paired(|| shake(engine, &jobs, &mut counts, &limits), &mut bare);
    let ratio = pair_ratio(8, || shake(engine, &jobs, &mut counts, &limits), &mut bare);
    let utilization = used as f64 / capacity as f64;
    let blocks: usize = limits.iter().sum();
    let pass = utilization >= 0.95 && ratio <= 1.10;
    println!(
        "mixed {:?}: {used}/{capacity} occupied ({:.2}%), {:.2} ticks/output lane-block, scheduler/bare {:.3} (median of paired ratios) {}",
        engine.backend(),
        utilization * 100.0,
        time / blocks as f64,
        ratio,
        if pass { "PASS" } else { "FAIL" }
    );
    pass
}

/// The C SHAKE128 rows: absorb one 34-byte input per lane, squeeze 5 blocks.
#[cfg(cref)]
fn c_shake(
    w: usize,
    c4: &mut [u64; 100],
    c8: &mut [u64; 200],
    out: &mut [[u8; 840]; 8],
    input: &[u8],
) {
    let [o0, o1, o2, o3, o4, o5, o6, o7] = out;
    if w == 4 {
        agg_mldsa_cref::shake128x4_absorb_once(c4, input, input, input, input);
        agg_mldsa_cref::shake128x4_squeezeblocks(o0, o1, o2, o3, 5, c4);
    } else {
        agg_mldsa_cref::xurq_avx512_shake128x8_absorb(
            c8, input, input, input, input, input, input, input, input,
        );
        agg_mldsa_cref::xurq_avx512_shake128x8_squeezeblocks(c8, o0, o1, o2, o3, o4, o5, o6, o7, 5);
    }
    black_box(out);
}

#[cfg(cref)]
fn check_cref(backend: Backend) -> bool {
    use sha3::digest::{ExtendableOutput, Update, XofReader};
    #[repr(C, align(64))]
    struct Aligned<const N: usize>([u64; N]);
    let input = [42u8; 64]; // Pad the pinned C routine's whole-word tail read.
    let mut expected = [0u8; 168];
    let mut hasher = sha3::Shake128::default();
    hasher.update(&input[..34]);
    hasher.finalize_xof().read(&mut expected);
    let mut out = [[0u8; 168]; 8];
    let [a, b, c, d, e, f, g, h] = &mut out;
    if backend == Backend::Avx2 {
        let mut state = Aligned([0; 100]);
        agg_mldsa_cref::shake128x4_absorb_once(
            &mut state.0,
            &input[..34],
            &input[..34],
            &input[..34],
            &input[..34],
        );
        agg_mldsa_cref::shake128x4_squeezeblocks(a, b, c, d, 1, &mut state.0);
        let valid = out[..4].iter().all(|x| x == &expected);
        println!(
            "C SHAKE128 oracle (pq-crystals): {}",
            if valid { "PASS" } else { "INVALID" }
        );
        valid
    } else {
        let mut state = Aligned([0; 200]);
        agg_mldsa_cref::xurq_avx512_shake128x8_absorb(
            &mut state.0,
            &input[..34],
            &input[..34],
            &input[..34],
            &input[..34],
            &input[..34],
            &input[..34],
            &input[..34],
            &input[..34],
        );
        agg_mldsa_cref::xurq_avx512_shake128x8_squeezeblocks(
            &mut state.0,
            a,
            b,
            c,
            d,
            e,
            f,
            g,
            h,
            1,
        );
        let valid = out.iter().all(|x| x == &expected);
        println!(
            "C SHAKE128 oracle (Hybrid, corrected absorb): {}",
            if valid { "PASS" } else { "INVALID" }
        );
        let mut expected_state = [0; 25];
        ::keccak::f1600(&mut expected_state);
        state.0.fill(0);
        agg_mldsa_cref::keccak_p1600_times8_permute_all_24rounds(&mut state.0);
        let matches =
            (0..25).all(|word| (0..8).all(|lane| state.0[word * 8 + lane] == expected_state[word]));
        println!(
            "C permutation oracle (XKCP, restored theta): {}",
            if matches { "PASS" } else { "INVALID" }
        );
        valid && matches
    }
}

/// The scheduler with each kernel switched on or off.
#[cfg(cref)]
#[target_feature(enable = "avx2")]
unsafe fn variant4<const PERMUTE: bool, const EXTRACT: bool>(
    jobs: &[XofJob<'_>],
    sink: &mut CountSink<'_>,
) {
    lanes::run::<4, false, _, _>(
        jobs,
        sink,
        lanes::Kernel {
            permute: |s: &mut lanes::State<4>| {
                if PERMUTE {
                    avx2::permute(s)
                } else {
                    black_box(s);
                }
            },
            extract: |s: &lanes::State<4>, b: &mut [Block; 4], r| {
                if EXTRACT {
                    avx2::extract(s, b, r)
                }
            },
            clear: |s: &mut lanes::State<4>, l| avx2::clear(s, l),
            absorb: |l: &mut [lanes::Lane; 4], j: &[XofJob<'_>], s: &mut lanes::State<4>, m| {
                avx2::absorb(l, j, s, m)
            },
        },
    );
}

/// The scheduler with each kernel switched on or off.
#[cfg(cref)]
#[target_feature(enable = "avx512f")]
unsafe fn variant8<const PERMUTE: bool, const EXTRACT: bool>(
    jobs: &[XofJob<'_>],
    sink: &mut CountSink<'_>,
) {
    lanes::run::<8, false, _, _>(
        jobs,
        sink,
        lanes::Kernel {
            permute: |s: &mut lanes::State<8>| {
                if PERMUTE {
                    avx512::permute(s)
                } else {
                    black_box(s);
                }
            },
            extract: |s: &lanes::State<8>, b: &mut [Block; 8], r| {
                if EXTRACT {
                    avx512::extract(s, b, r)
                }
            },
            clear: |s: &mut lanes::State<8>, l| avx512::clear(s, l),
            absorb: |l: &mut [lanes::Lane; 8], j: &[XofJob<'_>], s: &mut lanes::State<8>, m| {
                avx512::absorb(l, j, s, m)
            },
        },
    );
}

// Splits the scheduler's time into permutations, extraction and the rest
// (absorption, sink calls, bookkeeping), as paired ratios to the bare permutations.
#[cfg(cref)]
fn diag_breakdown() {
    let seed = [42u8; 34];
    let pk = [73u8; 1312];
    let (mixed, mixed_limits) = mixed_jobs(&seed, &pk);
    let single: Vec<XofJob<'_>> = vec![
        XofJob {
            xof: Xof::Shake128,
            input: [&seed, &[], &[], &[]]
        };
        8
    ];
    let single_limits = [5; 8];
    let mut counts = vec![0; mixed.len()];
    let mut s4 = lanes::State([[0u64; 4]; 25]);
    let mut s8 = lanes::State([[0u64; 8]; 25]);
    for (backend, w) in [(Backend::Avx2, 4), (Backend::Avx512, 8)] {
        let Some(engine) = Engine::new(backend) else {
            continue;
        };
        for (name, jobs, limits) in [
            ("mixed", &mixed[..], &mixed_limits[..]),
            ("34B/5", &single[..w], &single_limits[..w]),
        ] {
            let (_, cap) = dry_run(w, jobs, &mut counts, limits);
            let perms = cap / w;
            let calls = 1 + 800 / perms;
            let mut bare = || {
                for _ in 0..perms {
                    if w == 4 {
                        rust4(&mut s4)
                    } else {
                        rust8(&mut s8)
                    }
                }
            };
            let full = pair_ratio(
                calls,
                || shake(&engine, jobs, &mut counts, limits),
                &mut bare,
            );
            // The scheduler with a no-op permutation, with or without extraction,
            // compiled with the backend's features as the real one is.
            let mut partial = |extracting: bool| {
                counts[..limits.len()].copy_from_slice(limits);
                let mut sink = CountSink {
                    counts: &mut counts,
                };
                // SAFETY: The engine exists, so the backend's features were detected.
                unsafe {
                    match (w, extracting) {
                        (4, true) => variant4::<false, true>(black_box(jobs), &mut sink),
                        (4, false) => variant4::<false, false>(black_box(jobs), &mut sink),
                        (_, true) => variant8::<false, true>(black_box(jobs), &mut sink),
                        (_, false) => variant8::<false, false>(black_box(jobs), &mut sink),
                    }
                }
            };
            let with_extract = pair_ratio(calls, || partial(true), &mut bare);
            let rest = pair_ratio(calls, || partial(false), &mut bare);
            let ticks = common::measure_kernel(&mut bare).median;
            let dry = common::measure_kernel(|| partial(false)).median;
            println!(
                "diag {backend:?} {name}: {perms} permutations ({ticks:.0} ticks, no-op kernels {dry:.0}); full/bare {full:.3} = 1 + extract {:.3} + absorb, sinks and bookkeeping {rest:.3} + interaction {:.3}",
                with_extract - rest,
                full - 1.0 - with_extract
            );
        }
    }
}

// Paired ratios against the C permutations only (set KECCAK_DIAG).
#[cfg(cref)]
fn diag() {
    diag_breakdown();
    if Engine::new(Backend::Avx2).is_some() {
        // The AVX2 scheduler with the C permutation, against the bare C permutation.
        let seed = [42u8; 34];
        let pk = [73u8; 1312];
        let (jobs, limits) = mixed_jobs(&seed, &pk);
        let mut counts = vec![0; jobs.len()];
        let cperm = |s: &mut lanes::State<4>| {
            // SAFETY: State<4> is 100 contiguous u64 words, 64-byte aligned.
            let words = unsafe { &mut *(s as *mut lanes::State<4> as *mut [u64; 100]) };
            agg_mldsa_cref::f1600x4(words);
        };
        let mut c = lanes::State([[0u64; 4]; 25]);
        let (t, b) = paired(
            || {
                counts[..limits.len()].copy_from_slice(&limits);
                lanes::run::<4, false, _, _>(
                    black_box(&jobs),
                    &mut CountSink {
                        counts: &mut counts,
                    },
                    lanes::Kernel {
                        permute: cperm,
                        absorb: |l: &mut [lanes::Lane; 4],
                                 j: &[XofJob<'_>],
                                 s: &mut lanes::State<4>,
                                 m| {
                            lanes::absorb_with(l, j, m, |i, w, value| s.0[w][i] ^= value)
                        },
                        // SAFETY: AVX2 is available.
                        extract: |s: &lanes::State<4>, b: &mut [Block; 4], r| unsafe {
                            avx2::extract(s, b, r)
                        },
                        // SAFETY: AVX2 is available.
                        clear: |s: &mut lanes::State<4>, l| unsafe { avx2::clear(s, l) },
                    },
                );
            },
            || {
                for _ in 0..198 {
                    cperm(&mut c);
                }
            },
        );
        println!("diag Avx2 mixed with C permutation: ratio {:.3}", t / b);
    }
    #[repr(C, align(64))]
    struct Aligned<const N: usize>([u64; N]);
    let mut c4 = Aligned([0u64; 100]);
    let mut c8 = Aligned([0u64; 200]);
    let mut s4 = lanes::State([[0u64; 4]; 25]);
    let mut s8 = lanes::State([[0u64; 8]; 25]);
    for (backend, w) in [(Backend::Avx2, 4), (Backend::Avx512, 8)] {
        let Some(engine) = Engine::new(backend) else {
            continue;
        };
        let padded_input = [42u8; 64];
        let input = &padded_input[..34];
        let jobs = vec![
            XofJob {
                xof: Xof::Shake128,
                input: [input, &[], &[], &[]]
            };
            w
        ];
        let limits = vec![5; w];
        let mut counts = vec![0; w];
        let mut out = [[0u8; 840]; 8];
        let r = pair_ratio(
            50,
            || shake(&engine, &jobs, &mut counts, &limits),
            || c_shake(w, &mut c4.0, &mut c8.0, &mut out, input),
        );
        println!("diag {backend:?} SHAKE128 34B/5 rust/C {r:.3}");
    }
    if Engine::new(Backend::Avx2).is_some() {
        let r = pair_ratio(
            200,
            || rust4(&mut s4),
            || agg_mldsa_cref::f1600x4(black_box(&mut c4.0)),
        );
        println!("diag avx2 permutation rust/C {r:.3}");
    }
    if Engine::new(Backend::Avx512).is_some() {
        let r = pair_ratio(
            200,
            || rust8(&mut s8),
            || agg_mldsa_cref::xurq_keccak8x_permute(black_box(&mut c8.0)),
        );
        println!("diag avx512 permutation rust/Hybrid {r:.3}");
        let r = pair_ratio(
            200,
            || rust8(&mut s8),
            || agg_mldsa_cref::keccak_p1600_times8_permute_all_24rounds(black_box(&mut c8.0)),
        );
        println!("diag avx512 permutation rust/XKCP {r:.3}");
    }
}

// Short paired diagnostics for tuning; never substitutes for --gate.
fn quick() {
    let seed = [42u8; 34];
    let pk = [73u8; 1312];
    let (mixed, mixed_limits) = mixed_jobs(&seed, &pk);
    for (backend, w) in [(Backend::Avx2, 4), (Backend::Avx512, 8)] {
        let Some(engine) = Engine::new(backend) else {
            continue;
        };
        let single = vec![
            XofJob {
                xof: Xof::Shake128,
                input: [&seed, &[], &[], &[]]
            };
            w
        ];
        let single_limits = vec![5; w];
        for (name, jobs, limits) in [
            ("mixed", &mixed[..], &mixed_limits[..]),
            ("34B/5", &single[..], &single_limits[..]),
        ] {
            let mut counts = vec![0; jobs.len()];
            let (_, cap) = dry_run(w, jobs, &mut counts, limits);
            let mut s4 = lanes::State([[0; 4]; 25]);
            let mut s8 = lanes::State([[0; 8]; 25]);
            let r = pair_ratio(
                1 + 800 / (cap / w),
                || shake(&engine, jobs, &mut counts, limits),
                || {
                    for _ in 0..cap / w {
                        if w == 4 {
                            rust4(&mut s4)
                        } else {
                            rust8(&mut s8)
                        }
                    }
                },
            );
            let bare = common::measure_kernel(|| {
                if w == 4 {
                    rust4(&mut s4)
                } else {
                    rust8(&mut s8)
                }
            })
            .median;
            println!("quick {backend:?} {name}: scheduler/bare {r:.3}, bare {bare:.0} ticks");
            #[cfg(cref)]
            if name == "mixed" {
                #[repr(C, align(64))]
                struct Aligned<const N: usize>([u64; N]);
                let mut c4 = Aligned([0; 100]);
                let mut c8 = Aligned([0; 200]);
                let r = pair_ratio(
                    200,
                    || {
                        if w == 4 {
                            rust4(&mut s4)
                        } else {
                            rust8(&mut s8)
                        }
                    },
                    || {
                        if w == 4 {
                            agg_mldsa_cref::f1600x4(&mut c4.0)
                        } else {
                            agg_mldsa_cref::xurq_keccak8x_permute(&mut c8.0)
                        }
                    },
                );
                println!("quick {backend:?} permutation Rust/C {r:.3}");
            }
        }
    }
}

fn main() {
    if std::env::var_os("KECCAK_QUICK").is_some() && !std::env::args().any(|a| a == "--gate") {
        quick();
        return;
    }
    #[cfg(cref)]
    if std::env::var_os("KECCAK_DIAG").is_some() && !std::env::args().any(|a| a == "--gate") {
        diag();
        return;
    }
    common::print_system_info();
    #[cfg(not(cref))]
    if std::env::args().any(|a| a == "--gate") {
        eprintln!(
            "C comparisons UNRESOLVED; --gate requires AGG_CREF_SRC and RUSTFLAGS='--cfg cref'"
        );
        std::process::exit(1);
    }

    let mut ok = true;
    for (backend, w) in [(Backend::Avx2, 4), (Backend::Avx512, 8)] {
        let Some(engine) = Engine::new(backend) else {
            continue;
        };
        #[cfg(cref)]
        if !check_cref(backend) {
            ok = false;
            eprintln!("C comparison INVALID for {backend:?}; skip timing invalid oracles");
            continue;
        }
        let mut s4 = lanes::State([[0u64; 4]; 25]);
        let mut s8 = lanes::State([[0u64; 8]; 25]);
        let bare_ticks = if w == 4 {
            common::measure_kernel(|| rust4(&mut s4)).median
        } else {
            common::measure_kernel(|| rust8(&mut s8)).median
        };
        println!(
            "bare {backend:?}: {bare_ticks:.2} ticks/permutation ({:.2} ticks/lane-block)",
            bare_ticks / w as f64
        );
        #[cfg(cref)]
        {
            #[repr(C, align(64))]
            struct Aligned<const N: usize>([u64; N]);
            let mut c4 = Aligned([0u64; 100]);
            let mut c8 = Aligned([0u64; 200]);
            if w == 4 {
                ok &= compare(
                    "permutation",
                    "avx2",
                    || rust4(&mut s4),
                    || agg_mldsa_cref::f1600x4(black_box(&mut c4.0)),
                );
            } else {
                ok &= compare(
                    "permutation XKCP",
                    "avx512",
                    || rust8(&mut s8),
                    || {
                        agg_mldsa_cref::keccak_p1600_times8_permute_all_24rounds(black_box(
                            &mut c8.0,
                        ))
                    },
                );
                ok &= compare(
                    "permutation Hybrid",
                    "avx512",
                    || rust8(&mut s8),
                    || agg_mldsa_cref::xurq_keccak8x_permute(black_box(&mut c8.0)),
                );
            }
            let padded_input = [42u8; 64];
            let input = &padded_input[..34];
            let jobs = vec![
                XofJob {
                    xof: Xof::Shake128,
                    input: [input, &[], &[], &[]]
                };
                w
            ];
            let limits = vec![5; w];
            let mut counts = vec![0; w];
            let mut out = [[0u8; 840]; 8];
            ok &= compare(
                "SHAKE128 34B/5",
                if w == 4 { "avx2" } else { "avx512" },
                || shake(&engine, &jobs, &mut counts, &limits),
                || c_shake(w, &mut c4.0, &mut c8.0, &mut out, input),
            );
        }
        #[cfg(not(cref))]
        {
            println!("C comparisons UNRESOLVED; set AGG_CREF_SRC and RUSTFLAGS='--cfg cref'");
            ok = false;
        }
        ok &= scheduler(&engine, w);
    }
    common::finish_benchmarks();
    if std::env::args().any(|a| a == "--gate") && !ok {
        std::process::exit(1);
    }
}
