//! Peak-memory measurement (Linux `/proc/self/status`).
//!
//! `VmHWM` is the process's peak resident set size. Writing `5` to `/proc/self/clear_refs`
//! resets it to the current RSS (Linux 4.0 and later), so one process can measure the
//! prover's and the verifier's peaks separately; this changes only the calling process's
//! own accounting.

/// A `kB` field of `/proc/self/status` (`VmHWM`, `VmRSS`, ...), in KiB.
#[must_use]
pub fn status_kib(field: &str) -> Option<u64> {
    let status = std::fs::read_to_string("/proc/self/status").ok()?;
    parse_status_kib(&status, field)
}

/// The value of `field` in a `/proc/<pid>/status` text, in KiB.
#[must_use]
pub fn parse_status_kib(status: &str, field: &str) -> Option<u64> {
    status.lines().find_map(|line| {
        let rest = line.strip_prefix(field)?.strip_prefix(':')?;
        let mut parts = rest.split_whitespace();
        let value = parts.next()?.parse().ok()?;
        (parts.next() == Some("kB")).then_some(value)
    })
}

/// Peak resident set size (`VmHWM`) in KiB.
#[must_use]
pub fn peak_rss_kib() -> Option<u64> {
    status_kib("VmHWM")
}

/// Current resident set size (`VmRSS`) in KiB.
#[must_use]
pub fn rss_kib() -> Option<u64> {
    status_kib("VmRSS")
}

/// Resets `VmHWM` to the current RSS; false where unsupported.
pub fn reset_peak() -> bool {
    std::fs::write("/proc/self/clear_refs", "5").is_ok()
}

/// KiB to MiB.
#[must_use]
pub fn mib(kib: u64) -> f64 {
    kib as f64 / 1024.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn status_fields_parse_and_the_peak_tracks_allocations() {
        let text = "Name:\tx\nVmHWM:\t  12345 kB\nVmRSS:\t   678 kB\nThreads:\t3\n";
        assert_eq!(parse_status_kib(text, "VmHWM"), Some(12_345));
        assert_eq!(parse_status_kib(text, "VmRSS"), Some(678));
        assert_eq!(parse_status_kib(text, "Threads"), None);
        assert_eq!(parse_status_kib(text, "VmPeak"), None);
        if cfg!(target_os = "linux") {
            let before = peak_rss_kib().expect("VmHWM");
            let block = vec![1_u8; 64 << 20];
            std::hint::black_box(&block);
            let after = peak_rss_kib().expect("VmHWM");
            assert!(after >= before + 32 * 1024, "{before} -> {after}");
            drop(block);
            assert!(rss_kib().is_some());
            if reset_peak() {
                assert!(peak_rss_kib().unwrap() < after);
            }
        }
    }
}
