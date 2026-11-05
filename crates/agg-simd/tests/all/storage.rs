use agg_simd::{
    buf::{AlignedVec, KPlanes},
    cast, F, K, P,
};
use jolt_field::Ring;
#[test]
fn aligned_first_touch_and_views() {
    for n in [0, 1, 7, 8, 1023, 1024, 1025, 2048] {
        let a = AlignedVec::fill_par(n, 127, |i| i as u64);
        assert_eq!(a.as_ptr() as usize % 64, 0);
        assert!(a.iter().enumerate().all(|(i, &x)| i as u64 == x));
        let k = KPlanes::from_fn(n, |i| K::new(F::from_u64(i as u64), F::from_u64(P - 1)));
        for i in 0..n {
            assert_eq!(cast::interleaved_words(&[k.get(i)]), &[i as u64, P - 1]);
        }
    }
    assert_eq!(cast::base_words(&[F::from_u64(P - 1)]), &[P - 1]);
}
#[test]
fn panic_during_initialization_is_safe() {
    assert!(
        std::panic::catch_unwind(|| AlignedVec::fill_par(4096, 128, |i| {
            assert_ne!(i, 2049);
            i
        }))
        .is_err()
    );
}
/// Two coordinate planes must fill the buffer: a short second plane is refused before any
/// kernel reads it.
#[test]
#[should_panic(expected = "two coordinate planes")]
fn kplanes_with_a_short_second_plane_panic() {
    let planes = KPlanes { buf: AlignedVec::fill_par(12, 4, |_| 0), rows: 8 };
    let _ = planes.planes();
}
/// A table truncated to its round's words goes back to the pool at its allocated size, so a
/// request of that size reuses it.
#[test]
fn pool_reuses_a_truncated_table() {
    use agg_simd::buf::Pool;
    let mut pool = Pool::default();
    let mut table = AlignedVec::<u64>::zeroed(1024);
    let address = table.as_ptr() as usize;
    table.truncate(256);
    let spare = table.into_uninit();
    assert_eq!(spare.len(), 1024);
    pool.give(spare);
    let reused = pool.take(1024);
    assert_eq!(reused.as_ptr() as usize, address);
    assert_eq!(pool.words(), 0);
}
