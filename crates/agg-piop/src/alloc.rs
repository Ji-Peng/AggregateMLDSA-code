//! Global allocator that backs large allocations with transparent huge pages.
//!
//! The prover allocates many tables of tens to hundreds of MiB, each freshly
//! mapped by the system allocator (glibc maps anything above its 32 MiB
//! ceiling with `mmap` and unmaps it on free). With 4 KiB pages every table
//! pays one fault per page on first touch; at 32K signatures that is about
//! 5.7 million faults per prove. Advising each large allocation with
//! `madvise(MADV_HUGEPAGE)` before its first touch makes the faults take 2 MiB
//! at a time (about 12x fewer faults overall), which on the Zen 5 nodes cut
//! the prove by 6% on 16 cores and 8% on 8 cores.
//!
//! Only the 2 MiB-aligned interior of an allocation of at least
//! [`HUGE_MIN_BYTES`] is advised. The advice changes page backing only, never
//! contents, so results are unchanged. It is a no-op off Linux and when the
//! allocator is disabled with [`set_enabled`].

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicBool, Ordering};

/// Allocations of at least this size are advised. Smaller blocks come from the
/// allocator's heap arenas, which can share pages with other blocks.
pub const HUGE_MIN_BYTES: usize = 4 << 20;

static ENABLED: AtomicBool = AtomicBool::new(true);

/// Turn the advice on or off for later allocations (on by default).
pub fn set_enabled(on: bool) {
    ENABLED.store(on, Ordering::Relaxed);
}

/// [`System`] plus `madvise(MADV_HUGEPAGE)` on large blocks.
pub struct HugePageAlloc;

#[inline]
fn advise(ptr: *mut u8, size: usize) {
    #[cfg(target_os = "linux")]
    {
        const HUGE: usize = 2 << 20;
        const MADV_HUGEPAGE: i32 = 14;
        unsafe extern "C" {
            fn madvise(addr: *mut core::ffi::c_void, len: usize, advice: i32) -> i32;
        }
        if ptr.is_null() || size < HUGE_MIN_BYTES || !ENABLED.load(Ordering::Relaxed) {
            return;
        }
        let start = ptr as usize;
        let (lo, hi) = (start.next_multiple_of(HUGE), (start + size) & !(HUGE - 1));
        if hi > lo {
            // SAFETY: [lo, hi) lies inside the block just returned by the system
            // allocator; the advice changes page backing only. A failure is ignored.
            unsafe { madvise(lo as *mut core::ffi::c_void, hi - lo, MADV_HUGEPAGE) };
        }
    }
    #[cfg(not(target_os = "linux"))]
    let _ = (ptr, size);
}

// SAFETY: every method forwards to `System` with the same arguments and returns
// its result unchanged; `advise` never reads, writes or frees the block.
unsafe impl GlobalAlloc for HugePageAlloc {
    #[inline]
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        // SAFETY: forwarded caller contract.
        let ptr = unsafe { System.alloc(layout) };
        advise(ptr, layout.size());
        ptr
    }

    #[inline]
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        // SAFETY: forwarded caller contract. A freshly mapped block is not touched
        // by `calloc`, so advising afterwards still precedes the first touch.
        let ptr = unsafe { System.alloc_zeroed(layout) };
        advise(ptr, layout.size());
        ptr
    }

    #[inline]
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: forwarded caller contract.
        unsafe { System.dealloc(ptr, layout) }
    }

    #[inline]
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        // SAFETY: forwarded caller contract.
        let new = unsafe { System.realloc(ptr, layout, new_size) };
        advise(new, new_size);
        new
    }
}
