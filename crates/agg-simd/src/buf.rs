//! Aligned ownership with parallel first-touch initialization and no zero-fill pass.
use rayon::prelude::*;
use std::{
    alloc::{alloc, dealloc, handle_alloc_error, Layout},
    mem::MaybeUninit,
    ops::{Deref, DerefMut},
    ptr::NonNull,
    sync::OnceLock,
};

/// Whether a pooled buffer is filled with `0xA5` before it is handed out
/// (`AGG_ALLOC_POISON=1`, a test mode that exposes a reader of stale contents).
fn poison_hits() -> bool {
    static POISON: OnceLock<bool> = OnceLock::new();
    *POISON.get_or_init(|| std::env::var_os("AGG_ALLOC_POISON").is_some_and(|v| v == "1"))
}

/// Words per task of the poison fill.
const POISON_CHUNK: usize = 1 << 17;

/// A fully initialized allocation, aligned to at least 64 bytes. Copy elements need no drop.
pub struct AlignedVec<T: Copy> {
    ptr: NonNull<T>,
    len: usize,
    layout: Layout,
}
unsafe impl<T: Copy + Send> Send for AlignedVec<T> {}
unsafe impl<T: Copy + Sync> Sync for AlignedVec<T> {}
impl<T: Copy + Send + Sync> AlignedVec<T> {
    /// Allocate once; each slot is written exactly once. A panic drops the raw allocation.
    pub fn fill_par(len: usize, block_len: usize, f: impl Fn(usize) -> T + Sync + Send) -> Self {
        assert!(block_len > 0);
        assert!(
            std::mem::size_of::<T>() > 0,
            "zero-sized elements are unsupported"
        );
        let layout = Layout::from_size_align(
            len.checked_mul(std::mem::size_of::<T>())
                .expect("allocation overflow")
                .max(1),
            std::mem::align_of::<T>().max(64),
        )
        .expect("allocation layout");
        // SAFETY: `layout` has nonzero size (at least one byte) and a power-of-two alignment.
        let raw = unsafe { alloc(layout) };
        let ptr = NonNull::new(raw.cast::<T>()).unwrap_or_else(|| handle_alloc_error(layout));
        // The guard owns raw storage even if an initializer panics; no uninitialized T is read.
        struct Guard {
            ptr: NonNull<u8>,
            layout: Layout,
        }
        impl Drop for Guard {
            fn drop(&mut self) {
                // SAFETY: the guard owns the allocation made with `layout` and is dropped at
                // most once (it is forgotten after a successful fill).
                unsafe { dealloc(self.ptr.as_ptr(), self.layout) }
            }
        }
        let guard = Guard {
            ptr: ptr.cast(),
            layout,
        };
        let slots =
            // SAFETY: the allocation holds `len` slots of `T` (its size is `len * size_of::<T>()`
            // or one byte when that is zero), and `MaybeUninit<T>` has `T`'s layout.
            unsafe { std::slice::from_raw_parts_mut(ptr.as_ptr().cast::<MaybeUninit<T>>(), len) };
        slots
            .par_chunks_mut(block_len)
            .enumerate()
            .for_each(|(block, slots)| {
                for (j, slot) in slots.iter_mut().enumerate() {
                    slot.write(f(block * block_len + j));
                }
            });
        std::mem::forget(guard);
        Self { ptr, len, layout }
    }
}
impl<T: Copy> Deref for AlignedVec<T> {
    type Target = [T];
    fn deref(&self) -> &[T] {
        // SAFETY: `len` never exceeds the allocation's slot count, and every slot below `len`
        // is initialized (the constructors and `assume_init` guarantee it).
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
    }
}
impl<T: Copy> DerefMut for AlignedVec<T> {
    fn deref_mut(&mut self) -> &mut [T] {
        // SAFETY: as in `deref`; `&mut self` makes the view unique.
        unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len) }
    }
}
impl<T: Copy> Drop for AlignedVec<T> {
    fn drop(&mut self) {
        // SAFETY: `ptr` was allocated with `layout`, and ownership moves out of a value only
        // through `ManuallyDrop` or `mem::forget`, so the block is freed exactly once.
        unsafe { dealloc(self.ptr.as_ptr().cast(), self.layout) }
    }
}
impl<T: Copy + Send + Sync> Clone for AlignedVec<T> {
    fn clone(&self) -> Self {
        Self::fill_par(self.len, 1024, |i| self[i])
    }
}
impl<T: Copy + std::fmt::Debug> std::fmt::Debug for AlignedVec<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.deref().fmt(f)
    }
}

/// Build lane-major coordinate planes, writing each row exactly once in parallel.
pub fn fill_lane_planes(width: usize, rows: usize, f: impl Fn(usize, usize) -> (u64, u64) + Sync + Send) -> AlignedVec<u64> {
    const TASK_ROWS: usize = 1 << 12;
    let mut table = AlignedVec::<u64>::uninit(2 * width * rows);
    let mut tasks = Vec::new();
    for (lane, planes) in table.chunks_mut(2 * rows).enumerate() {
        let (c0, c1) = planes.split_at_mut(rows);
        for (k, (a, b)) in c0.chunks_mut(TASK_ROWS).zip(c1.chunks_mut(TASK_ROWS)).enumerate() {
            tasks.push((lane, k * TASK_ROWS, a, b));
        }
    }
    let write = |(lane, first, a, b): (usize, usize, &mut [MaybeUninit<u64>], &mut [MaybeUninit<u64>])| {
        for (i, (x, y)) in a.iter_mut().zip(b).enumerate() {
            let (v0, v1) = f(lane, first + i);
            x.write(v0);
            y.write(v1);
        }
    };
    if rows >= 2 * TASK_ROWS { tasks.into_par_iter().for_each(write); }
    else { tasks.into_iter().for_each(write); }
    // SAFETY: disjoint tasks cover both planes of every lane, and the loop writes
    // both slots for every row. A panic drops the uninitialized allocation.
    unsafe { table.assume_init() }
}

/// Canonical extension coordinates. Power-of-two rows >= 8 align both planes to 64 bytes.
#[derive(Clone, Debug)]
pub struct KPlanes {
    pub buf: AlignedVec<u64>,
    pub rows: usize,
}
impl KPlanes {
    pub fn from_fn(rows: usize, f: impl Fn(usize) -> crate::K + Sync + Send) -> Self {
        Self {
            buf: AlignedVec::fill_par(rows.checked_mul(2).expect("plane size"), 1024, |i| {
                let x = f(i % rows);
                if i < rows {
                    x.c0().to_canonical_u64()
                } else {
                    x.c1().to_canonical_u64()
                }
            }),
            rows,
        }
    }
    /// The two coordinate planes, `rows` words each.
    ///
    /// # Panics
    /// When the buffer does not hold exactly two planes of `rows` words: the vector kernels
    /// read whole planes through raw pointers, so a short second plane must not get through.
    pub fn planes(&self) -> (&[u64], &[u64]) {
        assert_eq!(self.buf.len(), 2 * self.rows, "two coordinate planes of `rows` words");
        self.buf.split_at(self.rows)
    }
    /// [`Self::planes`], writable.
    ///
    /// # Panics
    /// As [`Self::planes`].
    pub fn planes_mut(&mut self) -> (&mut [u64], &mut [u64]) {
        assert_eq!(self.buf.len(), 2 * self.rows, "two coordinate planes of `rows` words");
        self.buf.split_at_mut(self.rows)
    }
    pub fn get(&self, i: usize) -> crate::K {
        use jolt_field::Ring;
        assert!(i < self.rows);
        crate::K::new(
            crate::F::from_u64(self.buf[i]),
            crate::F::from_u64(self.buf[self.rows + i]),
        )
    }
}
impl<T: Copy + PartialEq> PartialEq for AlignedVec<T> {
    fn eq(&self, rhs: &Self) -> bool {
        **self == **rhs
    }
}
impl<T: Copy + Eq> Eq for AlignedVec<T> {}
impl<T: Copy> AlignedVec<T> {
    pub fn truncate(&mut self, len: usize) {
        self.len = self.len.min(len);
    }
    /// Allocate slots without initializing them; callers must initialize each slot before reading.
    pub fn uninit(len: usize) -> AlignedVec<MaybeUninit<T>> {
        assert!(std::mem::size_of::<T>() > 0);
        let layout = Layout::from_size_align(
            len.checked_mul(std::mem::size_of::<T>())
                .expect("allocation overflow")
                .max(1),
            std::mem::align_of::<T>().max(64),
        )
        .expect("allocation layout");
        // SAFETY: `layout` has nonzero size (at least one byte) and a power-of-two alignment.
        let p = NonNull::new(unsafe { alloc(layout) }.cast())
            .unwrap_or_else(|| handle_alloc_error(layout));
        AlignedVec {
            ptr: p,
            len,
            layout,
        }
    }
}
impl<T: Copy> AlignedVec<MaybeUninit<T>> {
    /// View the slots as initialized. The caller must write every element before any element
    /// is read; allocation and deallocation layouts agree.
    ///
    /// # Safety
    /// Every slot must hold an initialized `T`.
    pub unsafe fn assume_init(self) -> AlignedVec<T> {
        let result = AlignedVec {
            ptr: self.ptr.cast(),
            len: self.len,
            layout: self.layout,
        };
        std::mem::forget(self);
        result
    }
}
impl<T: Copy> AlignedVec<T> {
    /// Relabel the slots as possibly uninitialized, at the allocation's full capacity: a
    /// buffer truncated to one table's words returns to a [`Pool`] at the size it was
    /// allocated with, so a later request of that size can reuse it. Every word keeps its
    /// value, and the allocation and deallocation layouts are unchanged.
    pub fn into_uninit(self) -> AlignedVec<MaybeUninit<T>> {
        let this = std::mem::ManuallyDrop::new(self);
        AlignedVec {
            ptr: this.ptr.cast(),
            // Slots past the truncated length may be uninitialized, which `MaybeUninit`
            // permits. A zero-length allocation has a one-byte layout and so zero slots.
            len: this.layout.size() / std::mem::size_of::<T>(),
            layout: this.layout,
        }
    }
}
impl KPlanes {
    pub fn len(&self)->usize {self.rows}
    pub fn is_empty(&self)->bool {self.rows==0}
    /// Evaluate each node once and write both coordinates of both outputs directly.
    pub fn pair_from_fn(rows:usize,f:impl Fn(usize)->(crate::K,crate::K)+Sync+Send)->(Self,Self) {
        let mut p=AlignedVec::<u64>::uninit(rows.checked_mul(2).expect("plane size"));let mut q=AlignedVec::<u64>::uninit(rows*2);
        let (p0,p1)=p.split_at_mut(rows);let (q0,q1)=q.split_at_mut(rows);
        p0.par_chunks_mut(1024).zip(p1.par_chunks_mut(1024)).zip(q0.par_chunks_mut(1024).zip(q1.par_chunks_mut(1024))).enumerate().for_each(|(block,((p0,p1),(q0,q1)))| {
            for i in 0..p0.len() {let (a,b)=f(block*1024+i);p0[i].write(a.c0().to_canonical_u64());p1[i].write(a.c1().to_canonical_u64());q0[i].write(b.c0().to_canonical_u64());q1[i].write(b.c1().to_canonical_u64());}
        });
        // SAFETY: the parallel loop writes both coordinates of both outputs for every row
        // `0..rows`, which is every slot of `p` and `q`.
        unsafe {(Self{buf:p.assume_init(),rows},Self{buf:q.assume_init(),rows})}
    }
}
impl Default for KPlanes {fn default()->Self {Self{buf:AlignedVec::fill_par(0,1,|_|0),rows:0}}}

impl<T: ZeroBits + Send + Sync> AlignedVec<T> {
    /// Allocate zeroed elements with parallel first touch for large buffers.
    pub fn zeroed(len: usize) -> Self {
        let mut buffer = Self::uninit(len);
        if len.saturating_mul(std::mem::size_of::<T>()) >= 256 * 1024 {
            buffer.par_chunks_mut((256 * 1024 / std::mem::size_of::<T>()).max(1))
                .for_each(|chunk| {
                    // SAFETY: the chunk is writable and zero is valid for T.
                    unsafe { std::ptr::write_bytes(chunk.as_mut_ptr(), 0, chunk.len()) };
                });
        } else {
            // SAFETY: all slots are writable and zero is valid for T.
            unsafe { std::ptr::write_bytes(buffer.as_mut_ptr(), 0, len) };
        }
        // SAFETY: every slot has been initialized to a valid zero value.
        unsafe { buffer.assume_init() }
    }
}

/// Spare table buffers of a sum-check run: a group takes buffers for its table and gives
/// them back at its end, so the pages of the largest tables are mapped once per run.
#[derive(Default)]
pub struct Pool {
    spare: Vec<AlignedVec<MaybeUninit<u64>>>,
}

impl Pool {
    /// A buffer of at least `len` words: the smallest spare one that fits, else a new one.
    ///
    /// The slots may be uninitialized, so they are handed out as `MaybeUninit` words. A
    /// fill that writes every word of the returned buffer turns it into an initialized
    /// [`AlignedVec<u64>`] (`agg_simd::fused::fused_pass_into` does this for a table).
    /// Under `AGG_ALLOC_POISON=1` the whole buffer is filled with `0xA5` first, including a
    /// fresh allocation, which exposes a reader of stale or uninitialized contents.
    pub fn take(&mut self, len: usize) -> AlignedVec<MaybeUninit<u64>> {
        let best = self
            .spare
            .iter()
            .enumerate()
            .filter(|(_, b)| b.len() >= len)
            .min_by_key(|(_, b)| b.len())
            .map(|(i, _)| i);
        match best {
            Some(i) => {
                let mut buffer = self.spare.swap_remove(i);
                if poison_hits() {
                    buffer
                        .par_chunks_mut(POISON_CHUNK)
                        .for_each(|chunk| chunk.fill(MaybeUninit::new(0xA5A5_A5A5_A5A5_A5A5)));
                }
                buffer
            }
            None => {
                let mut buffer = AlignedVec::<u64>::uninit(len);
                if poison_hits() {
                    buffer
                        .par_chunks_mut(POISON_CHUNK)
                        .for_each(|chunk| chunk.fill(MaybeUninit::new(0xA5A5_A5A5_A5A5_A5A5)));
                }
                buffer
            }
        }
    }

    /// Return a buffer for reuse. Initialized buffers are relabelled with
    /// [`AlignedVec::into_uninit`] first, which keeps every word.
    pub fn give(&mut self, buffer: AlignedVec<MaybeUninit<u64>>) {
        self.spare.push(buffer);
    }

    /// Words held.
    #[must_use]
    pub fn words(&self) -> usize {
        self.spare.iter().map(|b| b.len()).sum()
    }
}

/// Types for which the all-zero bit pattern is a valid value.
///
/// # Safety
///
/// An implementation asserts that a `T` read from zeroed memory is a valid value.
pub unsafe trait ZeroBits: Copy {}

// SAFETY: zero is a valid value of every primitive integer.
unsafe impl ZeroBits for u8 {}
// SAFETY: as above.
unsafe impl ZeroBits for i8 {}
// SAFETY: as above.
unsafe impl ZeroBits for u16 {}
// SAFETY: as above.
unsafe impl ZeroBits for i16 {}
// SAFETY: as above.
unsafe impl ZeroBits for u32 {}
// SAFETY: as above.
unsafe impl ZeroBits for i32 {}
// SAFETY: as above.
unsafe impl ZeroBits for u64 {}
// SAFETY: as above.
unsafe impl ZeroBits for i64 {}
// SAFETY: `F` is `repr(transparent)` over a canonical `u64`; zero is the field's zero.
unsafe impl ZeroBits for crate::F {}
// SAFETY: `K` is `repr(transparent)` over `[F; 2]`; zero words are the field's zero.
unsafe impl ZeroBits for crate::K {}

/// Allocate a `Vec<T>` of length `len` without a zero-fill pass and initialize it in parallel
/// chunks of `chunk_len` elements; `init(chunk_index, slice)` writes every element of `slice`.
///
/// The memory comes from the allocator's zeroed path (`calloc`): large blocks are fresh
/// zero pages, so no pass precedes the writers and the first touch of each page happens in
/// the parallel `init`. Every slot is a valid (zero) `T` before `init` runs, so an element
/// `init` skips stays zero.
///
/// # Panics
/// When `chunk_len` is zero, `T` is zero-sized, or the allocation size overflows.
pub fn fill_vec_par_chunks<T: ZeroBits + Send>(
    len: usize,
    chunk_len: usize,
    init: impl Fn(usize, &mut [T]) + Sync + Send,
) -> Vec<T> {
    assert!(chunk_len > 0);
    assert!(
        std::mem::size_of::<T>() > 0,
        "zero-sized elements are unsupported"
    );
    if len == 0 {
        return Vec::new();
    }
    let layout = Layout::array::<T>(len).expect("allocation layout");
    // SAFETY: `layout` has nonzero size (len > 0, size_of::<T>() > 0).
    let raw = unsafe { std::alloc::alloc_zeroed(layout) };
    let ptr = NonNull::new(raw.cast::<T>()).unwrap_or_else(|| handle_alloc_error(layout));
    // SAFETY: the global allocator returned `len` zeroed slots with `T`'s layout, which
    // `Vec<T>` deallocates with the same layout; zeroed memory is a valid `T` (`ZeroBits`).
    let mut vec = unsafe { Vec::from_raw_parts(ptr.as_ptr(), len, len) };
    vec.par_chunks_mut(chunk_len)
        .enumerate()
        .for_each(|(chunk_idx, slice)| init(chunk_idx, slice));
    vec
}

/// Allocate a `Vec<T>` of length `len` without a fill pass and initialize it in parallel
/// from fixed-size chunk results. Chunk `c` covers `c * N .. min((c + 1) * N, len)`; the
/// callback returns `N` initialized values and the helper keeps the first
/// `min(N, len - c * N)` of them, so every slot is written exactly once.
///
/// # Panics
/// When `N` is zero, `T` is zero-sized, or the allocation size overflows.
pub fn fill_vec_par_arrays<T: Copy + Send, const N: usize>(
    len: usize,
    chunk: impl Fn(usize) -> [T; N] + Sync + Send,
) -> Vec<T> {
    assert!(N > 0);
    assert!(
        std::mem::size_of::<T>() > 0,
        "zero-sized elements are unsupported"
    );
    if len == 0 {
        return Vec::new();
    }
    let layout = Layout::array::<T>(len).expect("allocation layout");
    // SAFETY: `layout` has nonzero size (len > 0, size_of::<T>() > 0).
    let raw = unsafe { std::alloc::alloc(layout) };
    let ptr = NonNull::new(raw.cast::<T>()).unwrap_or_else(|| handle_alloc_error(layout));
    // SAFETY: the global allocator returned `len` slots with `T`'s layout, which `Vec<T>`
    // deallocates with the same layout. The length stays zero while the slots are
    // unwritten, so a panicking callback drops the allocation without reading an element.
    let mut vec = unsafe { Vec::from_raw_parts(ptr.as_ptr(), 0, len) };
    vec.spare_capacity_mut()
        .par_chunks_mut(N)
        .enumerate()
        .for_each(|(index, slots)| {
            let values = chunk(index);
            for (slot, value) in slots.iter_mut().zip(values) {
                slot.write(value);
            }
        });
    // SAFETY: the chunks cover the whole capacity and each slot was written exactly once.
    unsafe { vec.set_len(len) };
    vec
}

/// The slots of one chunk of a [`fill_vec_par_writer`] destination. Slots are written in
/// order, so a safe callback cannot leave a hole behind.
pub struct FillWriter<'a, T: Copy> {
    slots: &'a mut [MaybeUninit<T>],
    next: usize,
}

impl<T: Copy> FillWriter<'_, T> {
    /// Slots this chunk still expects.
    #[must_use]
    pub fn remaining(&self) -> usize {
        self.slots.len() - self.next
    }

    /// Write the next slot.
    #[inline]
    pub fn write(&mut self, value: T) {
        self.slots[self.next].write(value);
        self.next += 1;
    }
}

/// Allocate a `Vec<T>` of length `len` and fill it in parallel chunks of `chunk_len`
/// elements. Chunk `c` covers `c * chunk_len .. min((c + 1) * chunk_len, len)` and must
/// write every slot of its [`FillWriter`]; a chunk that writes too few slots panics.
///
/// # Panics
/// When `chunk_len` is zero, `T` is zero-sized, the allocation size overflows, or a chunk
/// writes fewer slots than it holds.
pub fn fill_vec_par_writer<T: Copy + Send>(
    len: usize,
    chunk_len: usize,
    init: impl Fn(usize, &mut FillWriter<'_, T>) + Sync + Send,
) -> Vec<T> {
    assert!(chunk_len > 0);
    assert!(
        std::mem::size_of::<T>() > 0,
        "zero-sized elements are unsupported"
    );
    if len == 0 {
        return Vec::new();
    }
    let layout = Layout::array::<T>(len).expect("allocation layout");
    // SAFETY: `layout` has nonzero size (len > 0, size_of::<T>() > 0).
    let raw = unsafe { std::alloc::alloc(layout) };
    let ptr = NonNull::new(raw.cast::<T>()).unwrap_or_else(|| handle_alloc_error(layout));
    // SAFETY: the global allocator returned `len` slots with `T`'s layout, which `Vec<T>`
    // deallocates with the same layout. The length stays zero while the slots are
    // unwritten, so a panicking callback drops the allocation without reading an element.
    let mut vec = unsafe { Vec::from_raw_parts(ptr.as_ptr(), 0, len) };
    vec.spare_capacity_mut()
        .par_chunks_mut(chunk_len)
        .enumerate()
        .for_each(|(index, slots)| {
            let mut writer = FillWriter { slots, next: 0 };
            init(index, &mut writer);
            assert_eq!(writer.next, writer.slots.len(), "incomplete chunk");
        });
    // SAFETY: every chunk wrote every one of its slots, and the chunks cover the capacity.
    unsafe { vec.set_len(len) };
    vec
}

#[cfg(test)]
mod tests {
    use super::{fill_vec_par_arrays, fill_vec_par_writer};

    /// A chunk shorter than `N`, a ragged last chunk and an exact multiple of `N`.
    #[test]
    fn fill_vec_par_arrays_covers_short_and_partial_chunks() {
        const N: usize = 64;
        for len in [0_usize, 1, 5, 63, 64, 65, 127, 128, 192] {
            let values = fill_vec_par_arrays::<u64, N>(len, |chunk| {
                std::array::from_fn(|j| (chunk * N + j) as u64)
            });
            assert_eq!(values.len(), len);
            assert!(
                values.iter().enumerate().all(|(i, &v)| v == i as u64),
                "chunk boundary at len {len}"
            );
        }
    }

    /// A zero-sized element has no slot to write, so the length contract cannot hold.
    #[test]
    #[should_panic(expected = "zero-sized elements are unsupported")]
    fn fill_vec_par_arrays_rejects_zero_sized_elements() {
        let _ = fill_vec_par_arrays::<(), 4>(4, |_| [(); 4]);
    }

    /// The in-place writer fills a short chunk and a ragged last chunk.
    #[test]
    fn fill_vec_par_writer_covers_short_and_partial_chunks() {
        const CHUNK: usize = 16;
        for len in [0_usize, 1, 15, 16, 17, 48] {
            let values = fill_vec_par_writer::<u64>(len, CHUNK, |chunk, out| {
                let base = chunk * CHUNK;
                let total = (len - base).min(CHUNK);
                while out.remaining() > 0 {
                    out.write((base + total - out.remaining()) as u64);
                }
            });
            assert_eq!(values.len(), len);
            assert!(
                values.iter().enumerate().all(|(i, &v)| v == i as u64),
                "chunk boundary at len {len}"
            );
        }
    }
}
