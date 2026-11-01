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
#[test]
fn direct_signed_field_encoding() {
    for n in [0,1,63,64,65,1024] {
        let a:Vec<_>=(0..n).map(|i|[i32::MIN,i32::MAX,-1,0,1,-32768][i%6]).collect();
        let expected:Vec<_>=a.iter().map(|&x|F::from_i64(i64::from(x))).chain(std::iter::repeat(F::from_u64(0)).take(17)).collect();
        assert_eq!(cast::encode_i32(&a,n+17),expected);
        let a:Vec<_>=(0..n).map(|i|i as u8 as i8).collect();
        let expected:Vec<_>=a.iter().map(|&x|F::from_i64(i64::from(x))).chain(std::iter::repeat(F::from_u64(0)).take(17)).collect();
        assert_eq!(cast::encode_i8(&a,n+17),expected);
    }
}
