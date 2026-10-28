//! Batched SHAKE256 sponge proof benchmark.
//!
//! Default: 1,170 records = 8,190 Keccak-f permutations (about 2^13) in one
//! Flock proof, Fast profile, 1 warmup run (discarded) and 5 timed repetitions,
//! medians of the timed runs reported.
//! Environment: `AGG_KECCAK_N` (records), `AGG_KECCAK_PROFILE` (`fast` or
//! `secure`), `AGG_KECCAK_WARMUP` (default 1), `AGG_KECCAK_REPS` (default 5). Peak heap comes from Flock's counting
//! allocator (as in Flock's `blake3_proof`/`sha2_proof` benches); `VmHWM` is the
//! process peak RSS. Run pinned: `taskset -c 0,1,3-15 cargo bench -p agg-keccak`.
#![allow(unsafe_code)]

use std::{
    alloc::{GlobalAlloc, Layout, System},
    sync::atomic::{AtomicUsize, Ordering},
    time::Instant,
};

use agg_keccak::{
    prover::{KeccakParams, Profile, SpongeProver},
    sponge::{SpongeRecord, N_PERMS},
};
use flock_core::test_rng::Rng;

struct PeakAlloc;
static CUR: AtomicUsize = AtomicUsize::new(0);
static PEAK: AtomicUsize = AtomicUsize::new(0);

// SAFETY: forwards to `System` and only updates counters.
unsafe impl GlobalAlloc for PeakAlloc {
    unsafe fn alloc(&self, l: Layout) -> *mut u8 {
        let p = unsafe { System.alloc(l) };
        if !p.is_null() {
            let c = CUR.fetch_add(l.size(), Ordering::Relaxed) + l.size();
            PEAK.fetch_max(c, Ordering::Relaxed);
        }
        p
    }
    unsafe fn alloc_zeroed(&self, l: Layout) -> *mut u8 {
        let p = unsafe { System.alloc_zeroed(l) };
        if !p.is_null() {
            let c = CUR.fetch_add(l.size(), Ordering::Relaxed) + l.size();
            PEAK.fetch_max(c, Ordering::Relaxed);
        }
        p
    }
    unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
        unsafe { System.dealloc(p, l) };
        CUR.fetch_sub(l.size(), Ordering::Relaxed);
    }
    unsafe fn realloc(&self, p: *mut u8, l: Layout, new: usize) -> *mut u8 {
        let q = unsafe { System.realloc(p, l, new) };
        if !q.is_null() {
            if new >= l.size() {
                let c = CUR.fetch_add(new - l.size(), Ordering::Relaxed) + (new - l.size());
                PEAK.fetch_max(c, Ordering::Relaxed);
            } else {
                CUR.fetch_sub(l.size() - new, Ordering::Relaxed);
            }
        }
        q
    }
}

#[global_allocator]
static ALLOC: PeakAlloc = PeakAlloc;

fn reset_peak() {
    PEAK.store(CUR.load(Ordering::Relaxed), Ordering::Relaxed);
}

fn mib(bytes: usize) -> f64 {
    bytes as f64 / (1024.0 * 1024.0)
}

fn vm_hwm_kib() -> Option<u64> {
    let status = std::fs::read_to_string("/proc/self/status").ok()?;
    let line = status.lines().find(|l| l.starts_with("VmHWM:"))?;
    line.split_whitespace().nth(1)?.parse().ok()
}

fn median(v: &mut [f64]) -> f64 {
    v.sort_by(f64::total_cmp);
    let n = v.len();
    if n % 2 == 1 {
        v[n / 2]
    } else {
        (v[n / 2 - 1] + v[n / 2]) / 2.0
    }
}

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn main() {
    let n = env_usize("AGG_KECCAK_N", 1170);
    let warmup = env_usize("AGG_KECCAK_WARMUP", 1);
    let reps = env_usize("AGG_KECCAK_REPS", 5).max(1);
    let profile = match std::env::var("AGG_KECCAK_PROFILE").as_deref() {
        Ok("secure") => Profile::Secure,
        Ok("slim") => Profile::Slim,
        _ => Profile::Fast,
    };
    let params = KeccakParams {
        profile,
        target_permutations: (N_PERMS * n.next_power_of_two()).max(1 << 16),
    };
    let prover = SpongeProver::new(params);
    let mut rng = Rng::new(0xBE7C);
    let records: Vec<SpongeRecord> = (0..n)
        .map(|_| {
            let mut mu = [0u8; 64];
            let mut w1 = [0u8; 768];
            mu.iter_mut().for_each(|b| *b = rng.next_u32() as u8);
            w1.iter_mut().for_each(|b| *b = rng.next_u32() as u8);
            SpongeRecord { mu, w1_encode: w1 }
        })
        .collect();
    let fs = [0x5Au8; 32];
    let batch = prover.batch(n);
    let cost = prover.gate().cost();
    println!(
        "agg-keccak sponge bench: records={n} permutations={} profile={profile:?} threads={} pclmulqdq={} avx512_feature={}",
        N_PERMS * n,
        rayon::current_num_threads(),
        cfg!(target_feature = "pclmulqdq"),
        agg_keccak::avx512_enabled()
    );
    println!(
        "gate: k_log={} committed_bits/record={} chi_bits/record={} nu={} m_total={} dense_m={}",
        prover.gate().layout().k_log,
        cost.committed_bits,
        cost.chi_bits,
        batch.nu(),
        batch.m_total(),
        batch.dense_m()
    );
    let mu: Vec<[u8; 64]> = records.iter().map(|r| r.mu).collect();
    let inputs: Vec<Vec<u64>> = records.iter().map(SpongeRecord::input_bits).collect();
    let c_tilde: Vec<[u8; 32]> = records.iter().map(SpongeRecord::c_tilde).collect();
    let public = SpongeProver::public_words(&mu, &c_tilde);
    let (mut t_wit, mut t_commit, mut t_prove, mut t_verify, mut peaks) =
        (vec![], vec![], vec![], vec![], vec![]);
    let (mut size, mut c_size) = (0, 0);
    for run in 0..warmup + reps {
        reset_peak();
        let base = CUR.load(Ordering::Relaxed);
        let t0 = Instant::now();
        let buffers = batch.honest_buffers(&inputs);
        let t1 = Instant::now();
        // Event 2 (commit before the seed), then the Flock proof with fs_sigma;
        // Flock's prover commits once more internally.
        let committed = batch.commit(buffers);
        let commitment = committed.commitment().clone();
        let tc = Instant::now();
        let proof = batch.prove_committed(committed, &public, &fs);
        let t2 = Instant::now();
        let peak = PEAK.load(Ordering::Relaxed).saturating_sub(base);
        batch
            .verify(&public, &commitment, &proof, &fs)
            .expect("honest proof verifies");
        let t3 = Instant::now();
        size = proof.bytes.len();
        c_size = commitment.as_bytes().len();
        let w = (t1 - t0).as_secs_f64();
        let c = (tc - t1).as_secs_f64();
        let p = (t2 - t0).as_secs_f64();
        let v = (t3 - t2).as_secs_f64();
        let label = if run < warmup {
            format!("warmup {run}")
        } else {
            format!("rep {}", run - warmup)
        };
        println!(
            "{label}: witness {w:.3} s, commit {c:.3} s, prove (incl. witness and commit) {p:.3} s, verify {:.1} ms, peak heap {:.1} MiB",
            v * 1e3,
            mib(peak)
        );
        if run < warmup {
            continue;
        }
        t_wit.push(w);
        t_commit.push(c);
        t_prove.push(p);
        t_verify.push(v);
        peaks.push(mib(peak));
    }
    println!(
        "median over {reps} ({warmup} warmup): witness {:.3} s, commit {:.3} s, prove {:.3} s ({:.0} perm/s), verify {:.1} ms, peak heap {:.1} MiB ({:.1} KiB/perm), proof {} bytes ({:.1} KiB), commitment {} bytes",
        median(&mut t_wit),
        median(&mut t_commit),
        median(&mut t_prove.clone()),
        (N_PERMS * n) as f64 / median(&mut t_prove),
        median(&mut t_verify) * 1e3,
        median(&mut peaks.clone()),
        median(&mut peaks) * 1024.0 / (N_PERMS * n) as f64,
        size,
        size as f64 / 1024.0,
        c_size
    );
    if let Some(hwm) = vm_hwm_kib() {
        println!("VmHWM {:.1} MiB", hwm as f64 / 1024.0);
    }
}
