//! Peak resident set size of this process (Linux `/proc/self/status`), for the `bench`
//! subcommand. `VmHWM` is the peak; writing `5` to `/proc/self/clear_refs` resets it to the
//! current RSS (Linux 4.0 and later), which changes only this process's own accounting, so
//! the prover's and the verifier's peaks are measured separately in one process.

/// A `kB` field of `/proc/self/status`, in KiB.
#[must_use]
pub fn status_kib(field: &str) -> Option<u64> {
    let status = std::fs::read_to_string("/proc/self/status").ok()?;
    status.lines().find_map(|line| {
        let rest = line.strip_prefix(field)?.strip_prefix(':')?;
        let mut parts = rest.split_whitespace();
        let value = parts.next()?.parse().ok()?;
        (parts.next() == Some("kB")).then_some(value)
    })
}

/// Peak resident set size (`VmHWM`) in KiB.
#[must_use]
pub fn peak_kib() -> Option<u64> {
    status_kib("VmHWM")
}

/// Resets `VmHWM` to the current RSS; false where unsupported.
pub fn reset_peak() -> bool {
    std::fs::write("/proc/self/clear_refs", "5").is_ok()
}

#[cfg(test)]
mod tests {
    #[test]
    fn peak_is_readable_on_linux() {
        if cfg!(target_os = "linux") {
            let peak = super::peak_kib().expect("VmHWM");
            assert!(peak > 0);
            assert!(super::status_kib("NoSuchField").is_none());
        }
    }
}
