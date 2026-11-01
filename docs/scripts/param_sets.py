#!/usr/bin/env python3
"""Parameters and derived constants for ML-DSA-44, ML-DSA-65, and ML-DSA-87.

Recomputes parameter values from FIPS 204 Tables 1 and 2, the H-u digit forms,
and the integer lift and projection bounds for aggregation.
"""

import math


def bitlen(x: int) -> int:
    """Return the number of bits in binary representation of positive integer x."""
    return x.bit_length() if x > 0 else 1


def compute_param_set(name: str, k: int, l: int, tau: int, lambda_bits: int,
                      gamma1: int, gamma2_div: int, eta: int, omega: int,
                      d_vec: int, lanes_k: int, lanes_l: int,
                      proj_blocks: int, proj_bound_log2: int):
    # FIPS 204 Table 1 primary parameters
    q = 8380417
    d = 13
    gamma2 = (q - 1) // gamma2_div
    beta = tau * eta

    # Bit widths
    t1_bits = bitlen(q - 1) - d  # 23 - 13 = 10
    z_bits = 1 + bitlen(gamma1 - 1)
    m_w = (q - 1) // (2 * gamma2)
    w1_bits = bitlen(m_w - 1)

    # Byte sizes (FIPS 204 Table 2)
    c_tilde_bytes = lambda_bits // 4
    pk_bytes = 32 + 32 * k * t1_bits
    sig_bytes = c_tilde_bytes + l * 32 * z_bits + omega + k
    w1_encode_bytes = 32 * k * w1_bits
    w1len = k * 256 * w1_bits

    # Derived bounds
    b_z = gamma1 - beta - 1
    b_u = 2 * gamma2 + 1
    b_t = (1 << d) * tau * ((1 << t1_bits) - 1)

    # Delta_0 ring residual worst-case bound before quotient:
    # l * 256 * ((q - 1) / 2) * B_z + B_T + 2 * gamma2 * (m_w - 1) + B_u
    delta0_max = (l * 256) * ((q - 1) // 2) * b_z + b_t + 2 * gamma2 * (m_w - 1) + b_u
    b_v = delta0_max // q

    # SelfTargetMSIS bound
    zeta_msis = max(gamma1 - beta, 2 * gamma2 + 1 + (1 << (d - 1)) * tau)

    # H-u bounds
    bar_b_z = gamma1  # relaxed bound: 2^17 for 44, 2^19 for 65 and 87
    bar_b_u = b_u
    zeta_prime = max(bar_b_z, bar_b_u + (1 << (d - 1)) * tau)

    # Bridge and hypercube geometry
    p_positions = 256 * k
    p_pad = 1 << math.ceil(math.log2(p_positions))
    b_1 = (1 << 19) // p_pad
    block_records_log2 = int(math.log2(b_1))
    d_pos = 8 + d_vec
    positions = 1 << d_pos
    n_perms = 9 if k == 8 else 7

    # Projection and integer lift
    p_field = (1 << 64) - 59
    p_half = p_field // 2
    proj_bound = 1 << proj_bound_log2
    honest_proj_bound = (p_positions // proj_blocks) * b_v
    proj_atoms = 128 * p_positions // 256
    proj_bytes = 128 * p_positions // 8

    az = (l * 256) * ((q - 1) // 2) * bar_b_z
    delta_z = az + b_t + 2 * gamma2 * (m_w - 1) + bar_b_u + q * 2 * proj_bound
    lift_margin = p_half / delta_z

    return {
        "name": name,
        "k": k,
        "l": l,
        "tau": tau,
        "lambda_bits": lambda_bits,
        "gamma1": gamma1,
        "gamma2": gamma2,
        "eta": eta,
        "beta": beta,
        "omega": omega,
        "t1_bits": t1_bits,
        "z_bits": z_bits,
        "m_w": m_w,
        "w1_bits": w1_bits,
        "c_tilde_bytes": c_tilde_bytes,
        "pk_bytes": pk_bytes,
        "sig_bytes": sig_bytes,
        "w1_encode_bytes": w1_encode_bytes,
        "w1len": w1len,
        "b_z": b_z,
        "b_u": b_u,
        "b_t": b_t,
        "delta0_max": delta0_max,
        "b_v": b_v,
        "zeta_msis": zeta_msis,
        "bar_b_z": bar_b_z,
        "bar_b_u": bar_b_u,
        "zeta_prime": zeta_prime,
        "d_vec": d_vec,
        "lanes_k": lanes_k,
        "lanes_l": lanes_l,
        "d_pos": d_pos,
        "positions": positions,
        "p_positions": p_positions,
        "p_pad": p_pad,
        "b_1": b_1,
        "block_records_log2": block_records_log2,
        "n_perms": n_perms,
        "proj_blocks": proj_blocks,
        "proj_bound_log2": proj_bound_log2,
        "proj_bound": proj_bound,
        "honest_proj_bound": honest_proj_bound,
        "proj_atoms": proj_atoms,
        "proj_bytes": proj_bytes,
        "delta_z": delta_z,
        "lift_margin": lift_margin,
    }


def print_param_set(p):
    print(f"=== {p['name']} ===")
    print(f"FIPS 204 parameters:")
    print(f"  Matrix dimensions (k, l): ({p['k']}, {p['l']})")
    print(f"  Challenge non-zeros (tau): {p['tau']}")
    print(f"  Collision strength (lambda): {p['lambda_bits']}")
    print(f"  Range of y (gamma1): {p['gamma1']} (2^{int(math.log2(p['gamma1']))})")
    print(f"  Rounding range (gamma2): {p['gamma2']}")
    print(f"  Secret key range (eta): {p['eta']}")
    print(f"  Challenge bound (beta = tau * eta): {p['beta']}")
    print(f"  Hint limit (omega): {p['omega']}")
    print(f"Bit widths and byte sizes:")
    print(f"  t1 bits: {p['t1_bits']}")
    print(f"  z bits: {p['z_bits']}")
    print(f"  w1 values (m_w): {p['m_w']}, w1 bits: {p['w1_bits']}")
    print(f"  c~ bytes: {p['c_tilde_bytes']}")
    print(f"  Public key bytes: {p['pk_bytes']}")
    print(f"  Signature bytes: {p['sig_bytes']}")
    print(f"  w1Encode bytes: {p['w1_encode_bytes']}")
    print(f"  w1len bits: {p['w1len']}")
    print(f"Derived bounds:")
    print(f"  B_z = gamma1 - beta - 1: {p['b_z']}")
    print(f"  B_u = 2 * gamma2 + 1: {p['b_u']}")
    print(f"  B_T = 2^13 * tau * (2^10 - 1): {p['b_t']}")
    print(f"  Delta0_max: {p['delta0_max']} (2^{math.log2(p['delta0_max']):.2f})")
    print(f"  B_v = Delta0_max / q: {p['b_v']} (2^{math.log2(p['b_v']):.2f})")
    print(f"  zeta (FIPS): {p['zeta_msis']}")
    print(f"  bar B_z (H-u): {p['bar_b_z']}")
    print(f"  bar B_u (H-u): {p['bar_b_u']}")
    print(f"  zeta' (H-u): {p['zeta_prime']}")
    print(f"Geometry and lanes:")
    print(f"  D_VEC: {p['d_vec']}, LANES_K: {p['lanes_k']}, LANES_L: {p['lanes_l']}")
    print(f"  D_POS: {p['d_pos']}, Positions: {p['positions']}")
    print(f"  Positions P: {p['p_positions']}, P_pad: {p['p_pad']}, B_1: {p['b_1']} (2^{p['block_records_log2']})")
    print(f"  Keccak permutations: {p['n_perms']}")
    print(f"Projection and lift bound:")
    print(f"  Row blocks g: {p['proj_blocks']}")
    print(f"  Honest projection bound (P/g)*B_v: {p['honest_proj_bound']} (2^{math.log2(p['honest_proj_bound']):.3f})")
    print(f"  Projection bound B: 2^{p['proj_bound_log2']}")
    print(f"  Psi matrix: {p['proj_atoms']} atoms, {p['proj_bytes']} bytes")
    print(f"  Lift bound |Delta^Z|: {p['delta_z']} (2^{math.log2(p['delta_z']):.4f})")
    print(f"  Lift margin (p/2) / |Delta^Z|: {p['lift_margin']:.6f}x")
    print()


def main():
    p44 = compute_param_set(
        name="ML-DSA-44",
        k=4, l=4, tau=39, lambda_bits=128,
        gamma1=1 << 17, gamma2_div=88, eta=2, omega=80,
        d_vec=2, lanes_k=1, lanes_l=1,
        proj_blocks=1, proj_bound_log2=37,
    )
    p65 = compute_param_set(
        name="ML-DSA-65",
        k=6, l=5, tau=49, lambda_bits=192,
        gamma1=1 << 19, gamma2_div=32, eta=4, omega=55,
        d_vec=0, lanes_k=6, lanes_l=5,
        proj_blocks=1, proj_bound_log2=39,
    )
    p87 = compute_param_set(
        name="ML-DSA-87",
        k=8, l=7, tau=60, lambda_bits=256,
        gamma1=1 << 19, gamma2_div=32, eta=2, omega=75,
        d_vec=0, lanes_k=8, lanes_l=7,
        proj_blocks=2, proj_bound_log2=39,
    )

    print_param_set(p44)
    print_param_set(p65)
    print_param_set(p87)


if __name__ == "__main__":
    main()
