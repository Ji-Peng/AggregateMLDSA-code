//! Global allocator for the prover's large tables: transparent huge pages and a
//! cache of freed blocks.
//!
//! The prover allocates many tables of tens to hundreds of MiB. The system
//! allocator maps each one freshly and unmaps it on free, so every table pays
//! one fault per 4 KiB page on first touch (about 5.7 million faults per 32K
//! prove) and every free pays an unmap with its TLB shootdown across all
//! threads. Here blocks of at least [`HUGE_MIN_BYTES`] are mapped directly,
//! 2 MiB-aligned and advised with `madvise(MADV_HUGEPAGE)`, so first-touch
//! faults take 2 MiB at a time. Freed blocks are kept in a small cache (at most
//! [`set_cache_bytes`] bytes) and handed back to later uninitialised
//! allocations of a similar size, which then need no fault and no unmap.
//! Zeroed allocations always take a fresh mapping: its zero pages are faulted
//! in by whichever threads first write them, in parallel, whereas zeroing a
//! cached block would be one serial `memset`.
//!
//! The advice and the cache change page backing and reuse only, never what a
//! program can observe through an allocation, so results are unchanged. Off
//! Linux everything forwards to [`System`].

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Mutex;

/// Blocks of at least this size are mapped directly. Smaller blocks come from
/// the system allocator's arenas.
pub const HUGE_MIN_BYTES: usize = 4 << 20;

const HUGE: usize = 2 << 20;
const SLOTS: usize = 64;

static ADVISE: AtomicBool = AtomicBool::new(true);
static CACHE_CAP: AtomicUsize = AtomicUsize::new(6 << 30);
static POISON: AtomicBool = AtomicBool::new(false);

/// Freed large blocks, oldest first: `(address, mapped length)`, the length a
/// multiple of 2 MiB.
struct Cache {
    slots: [(usize, usize); SLOTS],
    len: usize,
    bytes: usize,
}

static CACHE: Mutex<Cache> = Mutex::new(Cache { slots: [(0, 0); SLOTS], len: 0, bytes: 0 });

/// Turn the huge-page advice on or off for later mappings (on by default).
pub fn set_enabled(on: bool) {
    ADVISE.store(on, Ordering::Relaxed);
}

/// Fill every block handed out from the cache with `0xA5` before returning it
/// (a test mode that exposes any reader of uninitialised memory; off by default).
pub fn set_poison(on: bool) {
    POISON.store(on, Ordering::Relaxed);
}

/// Cap the bytes held by the cache of freed blocks (default 6 GiB; 0 disables
/// the cache and releases what it holds).
pub fn set_cache_bytes(bytes: usize) {
    CACHE_CAP.store(bytes, Ordering::Relaxed);
    if bytes == 0 {
        release_cache();
    }
}

/// Unmap every cached block, so the next allocations start from fresh
/// mappings as in a new process.
pub fn release_cache() {
    #[cfg(target_os = "linux")]
    {
        let mut cache = CACHE.lock().unwrap_or_else(|e| e.into_inner());
        for &(addr, len) in &cache.slots[..cache.len] {
            // SAFETY: a cached block is a live mapping of `len` bytes owned by the cache.
            unsafe { sys::munmap(addr as *mut _, len) };
        }
        cache.len = 0;
        cache.bytes = 0;
    }
}

/// [`System`] for small blocks; direct, huge-page-advised, cached mappings for
/// blocks of at least [`HUGE_MIN_BYTES`].
pub struct HugePageAlloc;

#[inline]
fn is_large(layout: &Layout) -> bool {
    cfg!(target_os = "linux") && layout.size() >= HUGE_MIN_BYTES && layout.align() <= HUGE
}

#[inline]
fn mapped_len(size: usize) -> usize {
    size.next_multiple_of(HUGE)
}

#[cfg(target_os = "linux")]
mod sys {
    use core::ffi::c_void;
    pub const PROT_RW: i32 = 0x1 | 0x2;
    pub const MAP_PRIVATE_ANON: i32 = 0x02 | 0x20;
    pub const MADV_HUGEPAGE: i32 = 14;
    pub const MAP_FAILED: *mut c_void = !0usize as *mut c_void;
    unsafe extern "C" {
        pub fn mmap(addr: *mut c_void, len: usize, prot: i32, flags: i32, fd: i32, off: i64) -> *mut c_void;
        pub fn munmap(addr: *mut c_void, len: usize) -> i32;
        pub fn madvise(addr: *mut c_void, len: usize, advice: i32) -> i32;
        pub fn mremap(old: *mut c_void, old_len: usize, new_len: usize, flags: i32, ...) -> *mut c_void;
    }
}

/// Take the smallest cached block of `len` to `len + len / 4` bytes whose
/// address meets `align`, trimmed to `len` bytes.
#[cfg(target_os = "linux")]
fn take_cached(len: usize, align: usize) -> Option<*mut u8> {
    let mut cache = CACHE.lock().unwrap_or_else(|e| e.into_inner());
    let mut best: Option<usize> = None;
    for i in 0..cache.len {
        let (addr, l) = cache.slots[i];
        let fits = l >= len && l <= len + len / 4 && addr % align == 0;
        if fits && best.is_none_or(|b| l < cache.slots[b].1) {
            best = Some(i);
        }
    }
    let i = best?;
    let (addr, l) = cache.slots[i];
    let n = cache.len;
    cache.slots.copy_within(i + 1..n, i);
    cache.len -= 1;
    cache.bytes -= l;
    drop(cache);
    if l > len {
        // The block is freed later with the caller's layout, i.e. as `len` bytes.
        // SAFETY: [addr + len, addr + l) is the tail of a mapping the cache owned.
        unsafe { sys::munmap((addr + len) as *mut _, l - len) };
    }
    if POISON.load(Ordering::Relaxed) {
        // SAFETY: the block is a live mapping of at least `len` bytes now owned by the caller.
        unsafe { core::ptr::write_bytes(addr as *mut u8, 0xA5, len) };
    }
    Some(addr as *mut u8)
}

/// Map `len` bytes (a multiple of 2 MiB) aligned to 2 MiB, advised for huge pages.
#[cfg(target_os = "linux")]
fn map_fresh(len: usize) -> *mut u8 {
    // SAFETY: an anonymous private mapping with no address hint.
    let raw = unsafe { sys::mmap(core::ptr::null_mut(), len + HUGE, sys::PROT_RW, sys::MAP_PRIVATE_ANON, -1, 0) };
    if raw == sys::MAP_FAILED {
        return core::ptr::null_mut();
    }
    let start = raw as usize;
    let aligned = start.next_multiple_of(HUGE);
    // SAFETY: head and tail lie inside the mapping just made and outside [aligned, aligned + len).
    unsafe {
        if aligned > start {
            sys::munmap(raw, aligned - start);
        }
        let tail = start + len + HUGE - (aligned + len);
        if tail > 0 {
            sys::munmap((aligned + len) as *mut _, tail);
        }
        if ADVISE.load(Ordering::Relaxed) {
            sys::madvise(aligned as *mut _, len, sys::MADV_HUGEPAGE);
        }
    }
    aligned as *mut u8
}

/// Return a block to the cache, evicting the oldest blocks to make room; a
/// block larger than the cap is unmapped.
#[cfg(target_os = "linux")]
fn free_large(ptr: *mut u8, len: usize) {
    let mut evicted = [(0usize, 0usize); SLOTS];
    let mut num_evicted = 0;
    let cap = CACHE_CAP.load(Ordering::Relaxed);
    let kept = len <= cap && {
        let mut cache = CACHE.lock().unwrap_or_else(|e| e.into_inner());
        while cache.len > 0 && (cache.len == SLOTS || cache.bytes + len > cap) {
            let oldest = cache.slots[0];
            let n = cache.len;
            cache.slots.copy_within(1..n, 0);
            cache.len -= 1;
            cache.bytes -= oldest.1;
            evicted[num_evicted] = oldest;
            num_evicted += 1;
        }
        let i = cache.len;
        cache.slots[i] = (ptr as usize, len);
        cache.len += 1;
        cache.bytes += len;
        true
    };
    // Unmap outside the lock.
    for &(addr, l) in &evicted[..num_evicted] {
        // SAFETY: an evicted block is a mapping of `l` bytes the cache owned.
        unsafe { sys::munmap(addr as *mut _, l) };
    }
    if !kept {
        // SAFETY: `ptr` is a mapping of `len` bytes owned by the freed allocation.
        unsafe { sys::munmap(ptr as *mut _, len) };
    }
}

// SAFETY: small blocks are forwarded to `System` unchanged. A large block is a
// private anonymous mapping of `mapped_len(size)` bytes, 2 MiB-aligned (so any
// `align <= 2 MiB` holds; `realloc` never moves a mapping), owned by exactly one
// allocation or by the cache at a time; `is_large` depends only on the layout, so `dealloc` and `realloc` take
// the same path as the allocation did.
unsafe impl GlobalAlloc for HugePageAlloc {
    #[inline]
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        #[cfg(target_os = "linux")]
        if is_large(&layout) {
            let len = mapped_len(layout.size());
            return take_cached(len, layout.align()).unwrap_or_else(|| map_fresh(len));
        }
        // SAFETY: forwarded caller contract.
        unsafe { System.alloc(layout) }
    }

    #[inline]
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        #[cfg(target_os = "linux")]
        if is_large(&layout) {
            // A fresh anonymous mapping reads as zero.
            return map_fresh(mapped_len(layout.size()));
        }
        // SAFETY: forwarded caller contract.
        unsafe { System.alloc_zeroed(layout) }
    }

    #[inline]
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        #[cfg(target_os = "linux")]
        if is_large(&layout) {
            free_large(ptr, mapped_len(layout.size()));
            return;
        }
        // SAFETY: forwarded caller contract.
        unsafe { System.dealloc(ptr, layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        // SAFETY: `new_size` is non-zero and does not overflow `isize` when
        // rounded to `layout.align()` (the caller contract of `realloc`).
        let new_layout = unsafe { Layout::from_size_align_unchecked(new_size, layout.align()) };
        #[cfg(target_os = "linux")]
        if is_large(&layout) && is_large(&new_layout) {
            let (old_len, new_len) = (mapped_len(layout.size()), mapped_len(new_size));
            if new_len == old_len {
                return ptr;
            }
            if new_len < old_len {
                // SAFETY: the tail [ptr + new_len, ptr + old_len) belongs to this block.
                unsafe { sys::munmap((ptr as usize + new_len) as *mut _, old_len - new_len) };
                return ptr;
            }
            // Grow in place when the address range behind the block is free; the
            // block keeps its address and alignment.
            // SAFETY: `ptr` is a mapping of `old_len` bytes owned by this allocation;
            // without MREMAP_MAYMOVE the kernel either extends it in place or fails.
            let grown = unsafe { sys::mremap(ptr as *mut _, old_len, new_len, 0) };
            if grown != sys::MAP_FAILED {
                if ADVISE.load(Ordering::Relaxed) {
                    // SAFETY: the grown tail is part of this block's mapping.
                    unsafe { sys::madvise((ptr as usize + old_len) as *mut _, new_len - old_len, sys::MADV_HUGEPAGE) };
                }
                return ptr;
            }
        }
        if is_large(&layout) || is_large(&new_layout) {
            // Copy into a new block; on failure the old block stays valid.
            // SAFETY: forwarded caller contract; both blocks hold min(old, new) bytes.
            unsafe {
                let new = self.alloc(new_layout);
                if !new.is_null() {
                    core::ptr::copy_nonoverlapping(ptr, new, layout.size().min(new_size));
                    self.dealloc(ptr, layout);
                }
                return new;
            }
        }
        // SAFETY: forwarded caller contract.
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn large_blocks_round_trip_through_cache_and_realloc() {
        let a = HugePageAlloc;
        let big = Layout::from_size_align(5 << 20, 64).unwrap();
        // SAFETY: layouts are non-zero; every block is freed with its own layout.
        unsafe {
            let z = a.alloc_zeroed(big);
            assert!(!z.is_null() && z as usize % (2 << 20) == 0);
            assert!(core::slice::from_raw_parts(z, big.size()).iter().all(|&b| b == 0));
            z.write_bytes(7, big.size());
            a.dealloc(z, big);
            // Reuse from the cache (uninitialised), then grow by mremap and shrink across the threshold.
            let p = a.alloc(big);
            p.write_bytes(3, big.size());
            let q = a.realloc(p, big, 9 << 20);
            assert!(core::slice::from_raw_parts(q, big.size()).iter().all(|&b| b == 3));
            let r = a.realloc(q, Layout::from_size_align(9 << 20, 64).unwrap(), 1 << 20);
            assert!(core::slice::from_raw_parts(r, 1 << 20).iter().all(|&b| b == 3));
            a.dealloc(r, Layout::from_size_align(1 << 20, 64).unwrap());
            // A small block grown past the threshold keeps its bytes.
            let small = Layout::from_size_align(4096, 8).unwrap();
            let s = a.alloc(small);
            s.write_bytes(9, 4096);
            let t = a.realloc(s, small, 6 << 20);
            assert!(core::slice::from_raw_parts(t, 4096).iter().all(|&b| b == 9));
            a.dealloc(t, Layout::from_size_align(6 << 20, 8).unwrap());
        }
        release_cache();
    }
}
