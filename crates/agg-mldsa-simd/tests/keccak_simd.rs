#![allow(unexpected_cfgs)]
mod common;
use agg_mldsa_simd::poly;

use agg_mldsa_simd::{Backend, Block, Engine, Sink, Xof, XofJob};
// Include the private kernels so permutation tests need no public testing API.
#[allow(dead_code)]
#[path = "../src/keccak/avx2.rs"]
mod avx2;
#[allow(dead_code)]
#[path = "../src/keccak/avx512.rs"]
mod avx512;
#[allow(dead_code)]
#[path = "../src/keccak/lanes.rs"]
mod lanes;

fn permutations<const W: usize>(mut permute: impl FnMut(&mut lanes::State<W>)) {
    let mut rng = common::SplitMix64::new(0x1600);
    for trial in 0..101 {
        let mut state = lanes::State(std::array::from_fn(|_| {
            std::array::from_fn(|_| if trial == 0 { 0 } else { rng.next_u64() })
        }));
        let mut expected: [[u64; 25]; W] =
            std::array::from_fn(|lane| std::array::from_fn(|word| state.0[word][lane]));
        for lane in &mut expected {
            keccak::f1600(lane);
        }
        permute(&mut state);
        for lane in 0..W {
            for word in 0..25 {
                assert_eq!(state.0[word][lane], expected[lane][word]);
            }
        }
    }
}

#[test]
fn permutation_oracles() {
    #[cfg(target_arch = "x86_64")]
    {
        if Engine::new(Backend::Avx2).is_some() {
            // SAFETY: Engine construction checked the required ISA.
            permutations::<4>(|state| unsafe { avx2::permute(state) });
        }
        if Engine::new(Backend::Avx512).is_some() {
            // SAFETY: Engine construction checked the required ISA.
            permutations::<8>(|state| unsafe { avx512::permute(state) });
        }
    }
}

struct CheckSink {
    expected: Vec<Vec<u8>>,
    counts: Vec<usize>,
    limits: Vec<usize>,
}
impl Sink for CheckSink {
    fn block(&mut self, job: usize, block: &Block, rate: usize) -> bool {
        let count = self.counts[job];
        assert!(count < self.limits[job], "callback after completion");
        assert_eq!(
            &block.bytes[..rate],
            &self.expected[job][count * rate..(count + 1) * rate],
            "job {job}, block {count}"
        );
        assert!(block.bytes[rate..].iter().all(|&x| x == 0));
        self.counts[job] += 1;
        self.counts[job] == self.limits[job]
    }
}

fn check(jobs: &[XofJob<'_>], limits: &[usize]) {
    let expected: Vec<_> = jobs
        .iter()
        .zip(limits)
        .map(|(job, &n)| {
            let mut out = vec![0u8; n * job.xof.rate()];
            match job.xof {
                Xof::Shake128 => common::shake128_segments(&job.input, &mut out),
                Xof::Shake256 => common::shake256_segments(&job.input, &mut out),
            }
            out
        })
        .collect();
    for backend in [Backend::Avx2, Backend::Avx512] {
        let Some(engine) = Engine::new(backend) else {
            continue;
        };
        let mut sink = CheckSink {
            expected: expected.clone(),
            counts: vec![0; jobs.len()],
            limits: limits.to_vec(),
        };
        engine.run_xof(jobs, &mut sink);
        assert_eq!(sink.counts, limits);
    }
}

#[test]
fn mixed_segmented_jobs() {
    let mut rng = common::SplitMix64::new(0x5348414b45);
    for n in (0..=17).chain([33, 200, 401]) {
        let inputs: Vec<Vec<u8>> = (0..n)
            .map(|i| {
                let len = if n == 401 {
                    i
                } else {
                    let edges = [0, 1, 135, 136, 137, 167, 168, 169, 272, 336, 400];
                    if i < edges.len() {
                        edges[i]
                    } else {
                        rng.next_u64() as usize % 401
                    }
                };
                let mut bytes = vec![0; len];
                rng.fill_bytes(&mut bytes);
                bytes
            })
            .collect();
        for shake in 0..3 {
            let jobs: Vec<_> = inputs
                .iter()
                .enumerate()
                .map(|(i, bytes)| {
                    let a = rng.next_u64() as usize % (bytes.len() + 1);
                    let b = a + rng.next_u64() as usize % (bytes.len() - a + 1);
                    let xof = if (shake == 0 && i % 2 == 0) || shake == 1 {
                        Xof::Shake128
                    } else {
                        Xof::Shake256
                    };
                    XofJob {
                        xof,
                        input: [&bytes[..a], &[], &bytes[a..b], &bytes[b..]],
                    }
                })
                .collect();
            let limits: Vec<_> = (0..n).map(|_| 1 + rng.next_u64() as usize % 7).collect();
            check(&jobs, &limits);
        }
    }
}

#[test]
fn expand_a_mixed_keys() {
    let seed = [42u8; 34];
    let pk = [73u8; 1312];
    let mut jobs = Vec::new();
    let mut limits = Vec::new();
    for key in 0..8 {
        for entry in 0..16 {
            jobs.push(XofJob {
                xof: Xof::Shake128,
                input: [&seed[..32], &seed[32..], &[], &[]],
            });
            limits.push(5 + (key + entry) % 2);
        }
        jobs.push(XofJob {
            xof: Xof::Shake256,
            input: [&pk[..136], &[], &pk[136..], &[]],
        });
        limits.push(1);
    }
    check(&jobs, &limits);
}

#[cfg(cref)]
#[test]
fn c_permutation_oracles() {
    #[repr(C, align(64))]
    struct Aligned<const N: usize>([u64; N]);
    if Engine::new(Backend::Avx2).is_some() {
        permutations::<4>(|state| {
            let mut c = Aligned([0u64; 100]);
            for word in 0..25 {
                for lane in 0..4 {
                    c.0[word * 4 + lane] = state.0[word][lane];
                }
            }
            agg_mldsa_cref::f1600x4(&mut c.0);
            for word in 0..25 {
                for lane in 0..4 {
                    state.0[word][lane] = c.0[word * 4 + lane];
                }
            }
        });
    }
    if Engine::new(Backend::Avx512).is_some() {
        for hybrid in [true] {
            permutations::<8>(|state| {
                let mut c = Aligned([0u64; 200]);
                for word in 0..25 {
                    for lane in 0..8 {
                        c.0[word * 8 + lane] = state.0[word][lane];
                    }
                }
                if hybrid {
                    agg_mldsa_cref::xurq_keccak8x_permute(&mut c.0);
                } else {
                    agg_mldsa_cref::keccak_p1600_times8_permute_all_24rounds(&mut c.0);
                }
                for word in 0..25 {
                    for lane in 0..8 {
                        state.0[word][lane] = c.0[word * 8 + lane];
                    }
                }
            });
        }
    }
}
