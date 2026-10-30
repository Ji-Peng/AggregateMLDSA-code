mod common;

use agg_mldsa_simd::dispatch::{Backend, Engine};
use agg_mldsa_simd::params::{GAMMA1, Q};
use agg_mldsa_simd::poly::{Block, Poly32, RejState};

/// Oracle for CoeffFromThreeBytes (FIPS 204 Section 7.2).
fn coeff_from_three_bytes(b: [u8; 3]) -> Option<i32> {
    let z = (b[0] as i32 | ((b[1] as i32) << 8) | ((b[2] as i32) << 16)) & 0x7FFFFF;
    if z < Q {
        Some(z)
    } else {
        None
    }
}

/// Oracle for rejection sampling over a 168-byte rate block.
fn oracle_rej_uniform(block: &Block, st: &mut RejState) -> bool {
    for b in block.bytes[..168].as_chunks::<3>().0 {
        if st.count == 256 {
            break;
        }
        if let Some(c) = coeff_from_three_bytes(*b) {
            st.coeffs[st.count] = c;
            st.count += 1;
        }
    }
    st.count == 256
}

/// Returns test engines for backends that have hardware support on the host.
fn test_engines() -> Vec<Engine> {
    [Backend::Avx2, Backend::Avx512]
        .into_iter()
        .filter_map(Engine::new)
        .collect()
}

#[test]
fn test_rej_uniform_random_blocks() {
    let mut rng = common::SplitMix64::new(0x1234_5678_9ABC_DEF0);
    let engines = test_engines();
    assert!(!engines.is_empty(), "At least AVX2 engine must be supported");

    for engine in &engines {
        for trial in 0..50 {
            let mut st_simd = RejState::zero();
            let mut st_oracle = RejState::zero();

            while st_simd.count < 256 {
                let mut block = Block::zero();
                rng.fill_bytes(&mut block.bytes);

                let done_oracle = oracle_rej_uniform(&block, &mut st_oracle);
                let done_simd = engine.rej_uniform(&block, &mut st_simd);

                assert_eq!(
                    done_simd, done_oracle,
                    "Backend {:?} trial {trial}: done mismatch",
                    engine.backend()
                );
                assert_eq!(
                    st_simd.count, st_oracle.count,
                    "Backend {:?} trial {trial}: count mismatch",
                    engine.backend()
                );
                assert_eq!(
                    &st_simd.coeffs[..st_simd.count],
                    &st_oracle.coeffs[..st_oracle.count],
                    "Backend {:?} trial {trial}: coefficients mismatch",
                    engine.backend()
                );

                if done_simd {
                    break;
                }
            }
        }
    }
}

#[test]
fn test_rej_uniform_crafted_rejections() {
    let engines = test_engines();
    assert!(!engines.is_empty());

    // Q = 8,380,417 = 0x7FE001
    let q_bytes = [0x01, 0xE0, 0x7F];
    // Q - 1 = 8,380,416 = 0x7FE000 (valid candidate)
    let q_minus_1_bytes = [0x00, 0xE0, 0x7F];
    // 0x7FFFFF = 8,388,607 (rejected)
    let max_23_bytes = [0xFF, 0xFF, 0x7F];

    for engine in &engines {
        // 1. All-rejection block with 0x7FFFFF
        let mut block_max = Block::zero();
        for chunk in block_max.bytes[..168].as_chunks_mut::<3>().0 {
            chunk.copy_from_slice(&max_23_bytes);
        }
        let mut st = RejState::zero();
        let done = engine.rej_uniform(&block_max, &mut st);
        assert!(!done);
        assert_eq!(st.count, 0, "All 0x7FFFFF must be rejected");

        // 2. All-rejection block with Q (0x7FE001)
        let mut block_q = Block::zero();
        for chunk in block_q.bytes[..168].as_chunks_mut::<3>().0 {
            chunk.copy_from_slice(&q_bytes);
        }
        let mut st = RejState::zero();
        let done = engine.rej_uniform(&block_q, &mut st);
        assert!(!done);
        assert_eq!(st.count, 0, "All Q must be rejected");

        // 3. All-accept block with Q - 1 (0x7FE000)
        let mut block_qm1 = Block::zero();
        for chunk in block_qm1.bytes[..168].as_chunks_mut::<3>().0 {
            chunk.copy_from_slice(&q_minus_1_bytes);
        }
        let mut st = RejState::zero();
        let done = engine.rej_uniform(&block_qm1, &mut st);
        assert!(!done);
        assert_eq!(st.count, 56, "All 56 Q-1 candidates must be accepted");
        for &c in &st.coeffs[..56] {
            assert_eq!(c, Q - 1);
        }

        // 4. Alternating pattern across SIMD lane boundaries
        for shift in 0..16 {
            let mut block_alt = Block::zero();
            for (idx, chunk) in block_alt.bytes[..168].as_chunks_mut::<3>().0.iter_mut().enumerate() {
                if (idx + shift) % 2 == 0 {
                    chunk.copy_from_slice(&q_minus_1_bytes);
                } else {
                    chunk.copy_from_slice(&max_23_bytes);
                }
            }
            let mut st_simd = RejState::zero();
            let mut st_oracle = RejState::zero();
            engine.rej_uniform(&block_alt, &mut st_simd);
            oracle_rej_uniform(&block_alt, &mut st_oracle);
            assert_eq!(st_simd.count, st_oracle.count);
            assert_eq!(
                &st_simd.coeffs[..st_simd.count],
                &st_oracle.coeffs[..st_oracle.count]
            );
        }
    }
}

#[test]
fn test_rej_uniform_mid_block_at_every_offset() {
    let engines = test_engines();
    assert!(!engines.is_empty());

    // Construct a block with 56 valid candidates (Q - 1).
    let q_minus_1_bytes = [0x00, 0xE0, 0x7F];
    let mut block = Block::zero();
    for chunk in block.bytes[..168].as_chunks_mut::<3>().0 {
        chunk.copy_from_slice(&q_minus_1_bytes);
    }

    for engine in &engines {
        // Test every initial count from 200 to 255.
        // 256 will be reached when 256 - initial_count candidates are consumed.
        for initial_count in 200..256 {
            let mut st_simd = RejState::zero();
            let mut st_oracle = RejState::zero();

            st_simd.count = initial_count;
            st_oracle.count = initial_count;
            for i in 0..initial_count {
                st_simd.coeffs[i] = (i as i32) * 17;
                st_oracle.coeffs[i] = (i as i32) * 17;
            }

            let done_simd = engine.rej_uniform(&block, &mut st_simd);
            let done_oracle = oracle_rej_uniform(&block, &mut st_oracle);

            assert!(done_simd);
            assert!(done_oracle);
            assert_eq!(st_simd.count, 256);
            assert_eq!(st_oracle.count, 256);

            // Verify prefix untouched
            for i in 0..initial_count {
                assert_eq!(st_simd.coeffs[i], (i as i32) * 17);
            }
            // Verify new entries match
            assert_eq!(&st_simd.coeffs[..256], &st_oracle.coeffs[..256]);
        }
    }
}

#[test]
fn test_rej_uniform_exact_block_end() {
    let engines = test_engines();
    assert!(!engines.is_empty());

    let q_minus_1_bytes = [0x00, 0xE0, 0x7F];
    let mut block = Block::zero();
    for chunk in block.bytes[..168].as_chunks_mut::<3>().0 {
        chunk.copy_from_slice(&q_minus_1_bytes);
    }

    for engine in &engines {
        // Initial count = 200: 56 candidates accepted reaches exactly 256 at the very end of the block.
        let mut st_simd = RejState::zero();
        st_simd.count = 200;
        let done = engine.rej_uniform(&block, &mut st_simd);
        assert!(done);
        assert_eq!(st_simd.count, 256);

        // Initial count = 199: 56 candidates accepted reaches 255 (not yet 256).
        let mut st_simd2 = RejState::zero();
        st_simd2.count = 199;
        let done2 = engine.rej_uniform(&block, &mut st_simd2);
        assert!(!done2);
        assert_eq!(st_simd2.count, 255);
    }
}

#[test]
fn test_unpack_z_against_native_sig_decode() {
    let mut rng = common::SplitMix64::new(0xCAFE_BABE_0123_4567);
    let engines = test_engines();
    assert!(!engines.is_empty());

    for engine in &engines {
        for trial in 0..20 {
            // Build a valid synthetic signature: c_tilde (32 bytes) + 2304 bytes z + 84 bytes h.
            let mut sig = vec![0u8; 2420];
            rng.fill_bytes(&mut sig[..32]);

            // Random bits for z: unpacked into 18-bit chunks.
            rng.fill_bytes(&mut sig[32..32 + 2304]);

            // Hint: 80 bytes of 0s, 4 bytes row indices
            sig[32 + 2304..].fill(0);

            // Use native sig_decode as oracle
            if let Ok(decoded) = common::native::sig_decode(&sig) {
                let mut out_simd = [Poly32::zero(); 4];
                let z_slice: &[u8; 2304] = sig[32..32 + 2304].try_into().unwrap();
                let max_simd = engine.unpack_z(z_slice, &mut out_simd);

                let mut expected_max = 0u32;
                for (p, poly) in out_simd.iter().enumerate() {
                    for (i, &actual) in poly.0.iter().enumerate() {
                        let expected = decoded.z[p][i] as i32;
                        assert_eq!(
                            actual, expected,
                            "Backend {:?} trial {trial} poly {p} idx {i}: {actual} != {expected}",
                            engine.backend()
                        );
                        expected_max = expected_max.max(expected.unsigned_abs());
                    }
                }
                assert_eq!(
                    max_simd, expected_max,
                    "Backend {:?} max |z| mismatch: {max_simd} != {expected_max}",
                    engine.backend()
                );
            }
        }
    }
}

#[test]
fn test_unpack_z_extreme_values() {
    let engines = test_engines();
    assert!(!engines.is_empty());

    for engine in &engines {
        // Value 1: raw bits = 1 -> z = GAMMA1 - 1
        // Value 2: raw bits = 2*GAMMA1 - 1 = 0x3FFFF -> z = GAMMA1 - 0x3FFFF = -(GAMMA1 - 1)
        // Value 3: raw bits = 0 -> z = GAMMA1
        for &(raw_val, expected_val, expected_abs) in &[
            (0u32, GAMMA1, GAMMA1 as u32),
            (1u32, GAMMA1 - 1, (GAMMA1 - 1) as u32),
            (0x3FFFFu32, -(GAMMA1 - 1), (GAMMA1 - 1) as u32),
            (GAMMA1 as u32, 0, 0u32),
        ] {
            // Pack 256 identical coefficients into 576 bytes per polynomial.
            let mut z_bytes = [0u8; 2304];
            for p in 0..4 {
                let poly_bytes = &mut z_bytes[p * 576..(p + 1) * 576];
                for chunk in 0..32 {
                    // 8 coefficients of 18 bits = 144 bits = 18 bytes.
                    let mut bits = [0u8; 18];
                    for k in 0..8 {
                        let bit_offset = k * 18;
                        let val = raw_val as u64;
                        for b in 0..18 {
                            if (val & (1 << b)) != 0 {
                                let total_bit = bit_offset + b;
                                bits[total_bit / 8] |= 1 << (total_bit % 8);
                            }
                        }
                    }
                    poly_bytes[chunk * 18..(chunk + 1) * 18].copy_from_slice(&bits);
                }
            }

            let mut out = [Poly32::zero(); 4];
            let max_val = engine.unpack_z(&z_bytes, &mut out);

            for (p, poly) in out.iter().enumerate() {
                for (i, &actual) in poly.0.iter().enumerate() {
                    assert_eq!(
                        actual, expected_val,
                        "Backend {:?} poly {p} idx {i}: expected {expected_val}, got {actual}",
                        engine.backend(),
                    );
                }
            }
            assert_eq!(max_val, expected_abs);
        }
    }
}

#[test]
fn test_pack_w1_against_native() {
    let mut rng = common::SplitMix64::new(0x9876_5432_10FE_DCBA);
    let engines = test_engines();
    assert!(!engines.is_empty());

    for engine in &engines {
        // 1. Random w1 in [0, 43]
        for _trial in 0..20 {
            let mut w1 = [[0u8; 256]; 4];
            for row in &mut w1 {
                for coeff in row.iter_mut() {
                    *coeff = (rng.next_u64() % 44) as u8;
                }
            }

            let expected = common::native::w1_encode(&w1);
            let mut actual = [0u8; 768];
            engine.pack_w1(&w1, &mut actual);

            assert_eq!(
                actual, expected,
                "Backend {:?}: random w1 encoding mismatch",
                engine.backend()
            );
        }

        // 2. All-zeros
        let w1_zeros = [[0u8; 256]; 4];
        let expected_zeros = common::native::w1_encode(&w1_zeros);
        let mut actual_zeros = [0u8; 768];
        engine.pack_w1(&w1_zeros, &mut actual_zeros);
        assert_eq!(actual_zeros, expected_zeros);

        // 3. All-43s
        let w1_43 = [[43u8; 256]; 4];
        let expected_43 = common::native::w1_encode(&w1_43);
        let mut actual_43 = [0u8; 768];
        engine.pack_w1(&w1_43, &mut actual_43);
        assert_eq!(actual_43, expected_43);
    }
}
