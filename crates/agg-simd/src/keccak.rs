//! Thin wrappers over `aerie-keccak`'s 8-way AVX-512 SHAKE256/cSHAKE256 batch permutations
//! for `mu`, `SampleInBall`, `c~` commitment hashing, and bridge matrix column derivation.

use aerie_keccak::{
    with_backend, Array, Backend, BackendClosure, ParState1600, Shake256StateBatch, Unsigned,
    RATE_WORDS, SHAKE256_RATE_BYTES,
};

/// Compute `mu_i = SHAKE256(tr_i || 0x00 || |ctx| || ctx || msg_i, 64)` for a slice of
/// `(tr_i, msg_i)` pairs with empty context (`ctx = &[]`), 8 records per AVX-512 permutation.
pub fn mu_batch_empty_ctx(inputs: &[(&[u8; 64], &[u8])], outputs: &mut [[u8; 64]]) {
    assert_eq!(inputs.len(), outputs.len());
    if inputs.is_empty() {
        return;
    }
    struct Closure<'a> {
        inputs: &'a [(&'a [u8; 64], &'a [u8])],
        outputs: &'a mut [[u8; 64]],
    }
    impl BackendClosure for Closure<'_> {
        fn call_once<B: Backend>(self) {
            let width = B::ParSize1600::USIZE;
            let mut batch = Shake256StateBatch::<B>::default();
            for (chunk_in, chunk_out) in
                self.inputs.chunks(width).zip(self.outputs.chunks_mut(width))
            {
                let segments = Array::<[&[u8]; 3], B::ParSize1600>::from_fn(|lane| {
                    let (tr, msg) = chunk_in[lane.min(chunk_in.len() - 1)];
                    [tr.as_slice(), &[0u8, 0u8], msg]
                });
                batch.absorb_segmented(&segments);
                batch.permute();
                for (lane, out) in chunk_out.iter_mut().enumerate() {
                    let words = batch.rate_words(lane);
                    for (dst, &w) in out.chunks_exact_mut(8).zip(&words[..8]) {
                        dst.copy_from_slice(&w.to_le_bytes());
                    }
                }
            }
        }
    }
    with_backend(Closure { inputs, outputs });
}

/// Squeeze the first 136-byte SHAKE256 rate block for each 32-byte `c_tilde` input in 8-way
/// AVX-512 batches (used by `SampleInBall`).
pub fn sample_in_ball_first_blocks(
    c_tildes: &[[u8; 32]],
    blocks: &mut [[u8; SHAKE256_RATE_BYTES]],
) {
    assert_eq!(c_tildes.len(), blocks.len());
    if c_tildes.is_empty() {
        return;
    }
    struct Closure<'a> {
        c_tildes: &'a [[u8; 32]],
        blocks: &'a mut [[u8; SHAKE256_RATE_BYTES]],
    }
    impl BackendClosure for Closure<'_> {
        fn call_once<B: Backend>(self) {
            let width = B::ParSize1600::USIZE;
            for (chunk_in, chunk_out) in
                self.c_tildes.chunks(width).zip(self.blocks.chunks_mut(width))
            {
                let slices = Array::<&[u8], B::ParSize1600>::from_fn(|lane| {
                    chunk_in[lane.min(chunk_in.len() - 1)].as_slice()
                });
                let mut batch = Shake256StateBatch::<B>::new(slices.as_slice());
                let squeezed = batch.squeeze_block();
                for (out, blk) in chunk_out.iter_mut().zip(squeezed.iter()) {
                    *out = *blk;
                }
            }
        }
    }
    with_backend(Closure { c_tildes, blocks });
}

/// Compute `c~'_i = SHAKE256(mu_i || w1_bytes_i, 32)` for a slice of `(mu_i, w1_bytes_i)` pairs.
/// Every input has the same 832-byte length (`64 + 768`), so all 7 permutations per 8 records
/// execute on the 8-way AVX-512 kernel in lockstep.
pub fn commitment_hash_batch(inputs: &[(&[u8; 64], &[u8; 768])], outputs: &mut [[u8; 32]]) {
    assert_eq!(inputs.len(), outputs.len());
    if inputs.is_empty() {
        return;
    }
    struct Closure<'a> {
        inputs: &'a [(&'a [u8; 64], &'a [u8; 768])],
        outputs: &'a mut [[u8; 32]],
    }
    impl BackendClosure for Closure<'_> {
        fn call_once<B: Backend>(self) {
            let width = B::ParSize1600::USIZE;
            let mut batch = Shake256StateBatch::<B>::default();
            for (chunk_in, chunk_out) in
                self.inputs.chunks(width).zip(self.outputs.chunks_mut(width))
            {
                let segments = Array::<[&[u8]; 2], B::ParSize1600>::from_fn(|lane| {
                    let (mu, w1) = chunk_in[lane.min(chunk_in.len() - 1)];
                    [mu.as_slice(), w1.as_slice()]
                });
                batch.absorb_segmented(&segments);
                batch.permute();
                for (lane, out) in chunk_out.iter_mut().enumerate() {
                    let words = batch.rate_words(lane);
                    for (dst, &w) in out.chunks_exact_mut(8).zip(&words[..4]) {
                        dst.copy_from_slice(&w.to_le_bytes());
                    }
                }
            }
        }
    }
    with_backend(Closure { inputs, outputs });
}

/// Derive a contiguous range of bridge matrix columns `start_zeta .. start_zeta + count`
/// (writing `count * k` row indices into `entries`) using 8-way AVX-512 Keccak-f[1600] batches
/// starting from the pre-absorbed cSHAKE256 customization `prefix` state.
pub fn derive_bridge_columns_into(
    prefix: &[u64; 25],
    sd: &[u8; 32],
    t: u64,
    start_zeta: usize,
    rows_log2: u32,
    k: usize,
    entries: &mut [u32],
) {
    assert_eq!(entries.len() % k, 0);
    let count = entries.len() / k;
    if count == 0 {
        return;
    }
    let sd_words: [u64; 4] = std::array::from_fn(|i| {
        u64::from_le_bytes(sd[8 * i..8 * (i + 1)].try_into().expect("8 bytes"))
    });
    let mut base_state = *prefix;
    for i in 0..4 {
        base_state[i] ^= sd_words[i];
    }
    base_state[4] ^= t;
    base_state[6] ^= 0x04;
    base_state[RATE_WORDS - 1] ^= 1_u64 << 63;

    struct Closure<'a> {
        base_state: [u64; 25],
        start_zeta: usize,
        rows_log2: u32,
        k: usize,
        entries: &'a mut [u32],
    }
    impl BackendClosure for Closure<'_> {
        fn call_once<B: Backend>(self) {
            let width = B::ParSize1600::USIZE;
            let par_perm = B::get_par_f1600();
            let scalar_perm = B::get_f1600();
            let mask = (1_u32 << self.rows_log2) - 1;
            let mut states = ParState1600::<B>::default();

            for (group, group_out) in self.entries.chunks_mut(width * self.k).enumerate() {
                let cols_in_group = group_out.len() / self.k;
                let group_zeta = self.start_zeta + group * width;
                for (lane, state) in states.iter_mut().enumerate() {
                    *state = self.base_state;
                    let zeta = group_zeta + lane.min(cols_in_group - 1);
                    state[5] ^= zeta as u64;
                }
                #[cfg(target_arch = "x86_64")]
                if width == 8 && std::is_x86_feature_detected!("avx512f") {
                    // The full permutation has one ISA boundary; generic lane helpers
                    // otherwise outline each intrinsic under conservative global flags.
                    unsafe { permute8(states.as_mut_slice().try_into().expect("eight states")) };
                } else {
                    par_perm(&mut states);
                }
                #[cfg(not(target_arch = "x86_64"))]
                par_perm(&mut states);

                for (lane, out) in group_out.chunks_exact_mut(self.k).enumerate() {
                    let state = &mut states[lane];
                    let mut filled = 0;
                    'fill: loop {
                        for &word in &state[..RATE_WORDS] {
                            for shift in [0, 16, 32, 48] {
                                let row = ((word >> shift) as u32) & mask;
                                if !out[..filled].contains(&row) {
                                    out[filled] = row;
                                    filled += 1;
                                    if filled == self.k {
                                        break 'fill;
                                    }
                                }
                            }
                        }
                        scalar_perm(state);
                    }
                }
            }
        }
    }
    with_backend(Closure {
        base_state,
        start_zeta,
        rows_log2,
        k,
        entries,
    });
}

pub const RHO_OFFSETS: [[u32; 5]; 5] = [
    [0, 36, 3, 41, 18],
    [1, 44, 10, 45, 2],
    [62, 6, 43, 15, 61],
    [28, 55, 25, 21, 56],
    [27, 20, 39, 8, 14],
];

/// iota round constants.
pub const ROUND_CONSTANTS: [u64; 24] = [
    0x0000_0000_0000_0001,
    0x0000_0000_0000_8082,
    0x8000_0000_0000_808A,
    0x8000_0000_8000_8000,
    0x0000_0000_0000_808B,
    0x0000_0000_8000_0001,
    0x8000_0000_8000_8081,
    0x8000_0000_0000_8009,
    0x0000_0000_0000_008A,
    0x0000_0000_0000_0088,
    0x0000_0000_8000_8009,
    0x0000_0000_8000_000A,
    0x0000_0000_8000_808B,
    0x8000_0000_0000_008B,
    0x8000_0000_0000_8089,
    0x8000_0000_0000_8003,
    0x8000_0000_0000_8002,
    0x8000_0000_0000_0080,
    0x0000_0000_0000_800A,
    0x8000_0000_8000_000A,
    0x8000_0000_8000_8081,
    0x8000_0000_0000_8080,
    0x0000_0000_8000_0001,
    0x8000_0000_8000_8008,
];


/// Keccak-f[1600] with all lane operations inside the AVX-512 feature boundary.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn permute8(states: &mut [[u64; 25]; 8]) {
    use std::arch::x86_64::*;
    let mut a: [__m512i; 25] = std::array::from_fn(|w| {
        let words: [u64; 8] = std::array::from_fn(|i| states[i][w]);
        unsafe { _mm512_loadu_si512(words.as_ptr().cast()) }
    });
    for rc in ROUND_CONSTANTS {
        let c: [__m512i; 5] = std::array::from_fn(|x| {
            _mm512_ternarylogic_epi64::<0x96>(
                _mm512_ternarylogic_epi64::<0x96>(a[x], a[x+5], a[x+10]), a[x+15], a[x+20])
        });
        let d: [__m512i; 5] = std::array::from_fn(|x| {
            _mm512_xor_si512(c[(x+4)%5], _mm512_rol_epi64::<1>(c[(x+1)%5]))
        });
        let mut b = [_mm512_setzero_si512(); 25];
        for y in 0..5 {
            for x in 0..5 {
                b[y + 5*((2*x+3*y)%5)] = _mm512_rolv_epi64(
                    _mm512_xor_si512(a[x+5*y], d[x]),
                    _mm512_set1_epi64(RHO_OFFSETS[x][y] as i64));
            }
        }
        for y in 0..5 {
            for x in 0..5 {
                a[x+5*y] = _mm512_xor_si512(b[x+5*y],
                    _mm512_andnot_si512(b[(x+1)%5+5*y], b[(x+2)%5+5*y]));
            }
        }
        a[0] = _mm512_xor_si512(a[0], _mm512_set1_epi64(rc as i64));
    }
    for w in 0..25 {
        let mut words = [0u64; 8];
        unsafe { _mm512_storeu_si512(words.as_mut_ptr().cast(), a[w]) };
        for i in 0..8 { states[i][w] = words[i]; }
    }
}
