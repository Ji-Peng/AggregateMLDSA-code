# AggregateMLDSA protocol, milestone 1 (ML-DSA-44, mode A, relation H-u)

## Status

| Field | Value |
| --- | --- |
| Status | normative draft for milestone 1; no implementation yet |
| Applies to | homogeneous ML-DSA-44 aggregates, mode A (public $`\tilde c_i`$), relation H-u ($`z`$ relaxed, $`u`$ exact; it replaced the relaxed relation H on 2026-10-25), wire version 4 (unfrozen; version 3 with tied-part proof version 5, the virtual parity chain of `docs/parity-chain.md`, deviation K21, and the fork-join transcript, the Flock proofs at a join after stage 3, deviation K22; lookup layout L16, `docs/lookup.md`; the proof-of-work schedule of `docs/soundness.md`) |
| Base system | Aerie `49b6c21` (Akita pin `e023b6a`), Flock HEAD `83f963e` circuit language, retired Keccak encoder `c2d0c24` as reference |
| Variants | H-exact (relation byte 1, Appendix A, `docs/h-exact.md` step B; implemented, deviation L9) and S1+S2 (Appendix A, not implemented) |
| Last reviewed | 2026-10-27, revision 8: the fork-join transcript (Sections 10, 11, 12.1; deviation K22): event 6 absorbs the final vectors and the share items, the Flock proofs follow at a join after stage 3, the part and span seeds bind the event-6 record digest; 2026-10-26, revision 7: aggregate unforgeability under SUF-CMA of the relaxed scheme (Section 12.3, steps 1' and 2'; `docs/soundness.md` 12.8), which closes the rewound-signing-query obligation; 2026-10-25, revision 6: relation H-exact as the second result (Sections 3.1, 4.1, 4.3, 7.4, 7.5, 10, 12; Appendix A), deviation L9; revision 5: relation H-u is the default (Sections 3, 4.3, 7.4, 10, 12; `docs/h-exact.md` step A, deviation L8), wire version 3; revision 4: wire version 2, the per-site ledger and the proof-of-work schedule of `docs/soundness.md` in Sections 10 and 12.1 (deviation P3 resolved); 2026-10-24, revision 3: the lookup layout L16 (former Appendix C) is normative in Sections 4 to 7 and 10 to 12; the base-16 digit layout of wire version 0 is retired (`docs/deviations.md` L1) |

The uppercase terms MUST, MUST NOT, SHOULD and MAY are normative as in BCP 14. Aerie conventions (Akita groups, local variable orders, lifting of small instances, Spongefish transcript, grinding) are adopted by reference to `aerie/specs/falcon.md`; this document restates only what differs. Deviations made during implementation go to `docs/deviations.md`. Design questions decided while writing are in Appendix B.

## 1. Notation and parameters

ML-DSA-44 parameters (FIPS 204, Table 1): $`q=8{,}380{,}417`$, $`n=256`$, $`(k,\ell)=(4,4)`$, $`d=13`$, $`\tau=39`$, $`\lambda=128`$, $`\gamma_1=2^{17}`$, $`\gamma_2=(q-1)/88=95{,}232`$, $`\eta=2`$, $`\beta=\tau\eta=78`$, $`\omega=80`$. Derived constants:

| Symbol | Value | Meaning |
| --- | ---: | --- |
| $`m_w=(q-1)/(2\gamma_2)`$ | 44 | number of $`w_1`$ values |
| $`B_z=\gamma_1-\beta-1`$ | 130,993 | honest $`\lVert z\rVert_\infty`$ bound |
| $`B_u=2\gamma_2+1`$ | 190,465 | UseHint residual bound (Section 3) |
| $`B_T=2^d\tau(2^{10}-1)`$ | 326,836,224 | bound on $`\lvert 2^d c\,t_1\rvert`$ over $`\mathbb Z`$ |
| $`B_v`$ | 67,068,447 | honest quotient bound (3.4), $`\lt 2^{26}`$ |
| $`\zeta`$ | 350,209 | SelfTargetMSIS bound of FIPS 204 |
| $`\mathsf{w1len}`$ | 6,144 | w1Encode bits per signature ($`k\cdot256\cdot6`$) |

Rings: $`R_q=\mathbb Z_q[X]/(X^n+1)`$. The PIOP field is $`\mathbb F_p`$, $`p=2^{64}-59`$; challenges lie in $`K=\mathbb F_{p^2}`$ (Akita fp64 profile `Prime64Offset59`, `Ext2`). $`\mathrm{ctr}_q(x)`$ is the centered representative in $`[-(q-1)/2,(q-1)/2]`$. $`\circledast`$ is negacyclic multiplication in $`\mathbb Z[X]/(X^{256}+1)`$ over the integers. $`\mathrm{eq}`$ and multilinear extensions are as in Aerie (3.1), (3.2); a table and its extension share a symbol.

Batch. $`1\le N\le 2^{15}`$ live records. $`N^\star=\max(\mathrm{next\_power\_of\_two}(N),8)=2^{d_{\rm sig}}`$; $`\mathsf{live}(i)=[i\lt N]`$. The verifier MUST reject $`N\gt2^{15}`$ in this wire version.

Indices. Coefficient $`j\in[0,256)`$ ($`d_{\rm coef}=8`$ variables, least significant first), vector index $`s`$ (column of $`A`$, for $`z`$) or $`\rho`$ (row, for $`u,v,b`$) in $`[0,4)`$ (2 variables), signature $`i\in[0,N^\star)`$ ($`d_{\rm sig}`$ variables). A position is $`x=(j,\rho)`$ or $`(j,s)`$, $`d_{\rm pos}=10`$. The main domain is $`\{0,1\}^{d_{\rm main}}`$, $`d_{\rm main}=d_{\rm pos}+d_{\rm sig}`$, canonical order $`(j,\rho\text{ or }s,i)`$; at $`N=2^{15}`$, $`d_{\rm main}=25`$.

Segments. $`d_{\rm seg}=\min(d_{\rm sig},13)`$; segment $`\sigma=\lfloor i/2^{13}\rfloor`$, $`S=2^{\max(0,d_{\rm sig}-13)}`$ segments. A segment is both one Keccak sub-batch ($`7\cdot2^{13}=57{,}344`$ permutations, Section 8) and one parity-bridge chain (Section 9). Write $`i=i_{\rm lo}+2^9 i_{\rm mid}+2^{13}\sigma`$ with $`i_{\rm lo}\in[0,2^9)`$, $`i_{\rm mid}\in[0,2^4)`$ (fewer bits when $`d_{\rm sig}\lt13`$; missing bits are absent, not zero-padded, in domain sizes).

Bridge domains (per lane $`t\in[0,6)`$, the bit index of $`w_1'`$):

| Level | Block index $`\beta_t`$ | In-block index $`\zeta_t`$ | $`J_t`$ | $`m_t`$ | $`k_t`$ | Output domain $`D_t`$, variables $`d_t`$ |
| --- | --- | --- | ---: | ---: | ---: | --- |
| 1 | $`(i_{\rm mid},\sigma)`$ | $`(j,\rho,i_{\rm lo})`$ | $`2^{19}`$ | $`2^{13}`$ | 14 | $`(\rho_1,i_{\rm mid},\sigma)`$, $`13+\max(0,d_{\rm sig}-9)`$ |
| 2 | $`\sigma`$ | $`(\rho_1,i_{\rm mid})`$, zero-padded to $`2^{19}`$ | $`2^{19}`$ | $`2^{13}`$ | 14 | $`(\rho_2,\sigma)`$, $`13+\max(0,d_{\rm sig}-13)`$ |
| 3 | $`\sigma`$ | $`(t,\rho_2)`$, all six lanes | 49,152 | 4,096 | 18 | $`(\rho_3,\sigma)`$, $`12+\max(0,d_{\rm sig}-13)`$ |

Level 3 has one instance per segment, not per lane; $`\rho_t`$ is the output row within a block. When $`d_{\rm sig}\lt9`$ a level-1 block is zero-padded to $`2^{19}`$ positions (public, sound).

## 2. Statement, public inputs and verifier precomputation

A record is $`(\mathsf{pk}_i,M_i)`$ with $`\mathsf{pk}_i=(\rho_i,t_{1,i})`$ a 1,312-byte ML-DSA-44 public key and $`M_i`$ a message (pure mode, empty context). The aggregate consists of the list $`(\tilde c_i)_{i\lt N}`$ of 32-byte commitment hashes and one proof $`\pi`$. The statement is: for every live $`i`$ there exist $`(z_i,h_i)`$ such that `ML-DSA.Verify_internal`$`(\mathsf{pk}_i,M'_i,(\tilde c_i,z_i,h_i))`$ accepts, relaxed to relation H of Section 3.

Verifier precomputation per live record (FIPS 204 Algorithm 8, lines 5 through 9; counts from `ref/notes/improvements-misc.md`):

1. $`\hat A_i=\mathrm{ExpandA}(\rho_i)`$: 16 polynomials, exactly 5 SHAKE128 blocks each (280 candidates; fewer than 256 accepted with probability $`2^{-132}`$, in which case the verifier MUST continue squeezing), 80 Keccak-f per record.
2. $`A_i[\rho,s]=\mathrm{ctr}_q(\mathrm{NTT}^{-1}(\hat A_i[\rho,s]))\in[-(q-1)/2,(q-1)/2]^{256}`$: 16 inverse NTTs per record. No NTT-domain shortcut exists, since the identity of Section 3 is over $`\mathbb Z`$ (misc memo 2.3). $`\hat A_i`$ and $`A_i`$ MAY be cached per key (11.8 KB).
3. $`\mathsf{tr}_i=\mathrm{H}(\mathsf{pk}_i,64)`$ (10 Keccak-f), $`\mu_i=\mathrm{H}(\mathsf{tr}_i\Vert\texttt{00}\Vert\texttt{00}\Vert M_i,64)`$ (1 Keccak-f when $`|M_i|\le69`$).
4. $`c_i=\mathrm{SampleInBall}(\tilde c_i)`$ (1 Keccak-f): $`\tau`$ entries in $`\{-1,1\}`$ at positions $`\mathsf{pos}_{i,1..\tau}`$.
5. $`T_i[\rho]=2^d\,(c_i\circledast t_{1,i}[\rho])\in\mathbb Z^{256}`$, computed over the integers from the sparse $`c_i`$: $`k\tau\cdot256=39{,}936`$ signed additions, no reduction and no NTT. $`\lvert T_i\rvert_\infty\le B_T\lt2^{29}`$. The public table is $`T(j,\rho,i)=T_i[\rho][j]`$, zero on padded rows.

Public tables on the main domain: $`T`$, and the padded-zero tables $`A(j,\rho,s,i)`$ used only through the weights of Section 7. Public bit tables for the Keccak side: $`\mu_i`$ (512 bits), $`\tilde c_i`$ (256 bits). Public bridge vectors $`Y_\sigma\in\{0,1\}^{4096}`$, one per segment, are part of $`\pi`$ (Section 9).

Public-record binding. The transcript binds $`X=\mathrm{LE64}(N)\Vert\big\Vert_i(\mathsf{pk}_i\Vert\mathrm{LE64}(|M_i|)\Vert M_i\Vert\tilde c_i)`$ with Aerie's 16-KiB-leaf cSHAKE256 tree (falcon.md (2.4)) under customization `agg-mldsa/v1/public-records-binding`.

## 3. Relation H

### 3.1 Definition

For every live $`i`$, the witness is $`z_i\in\mathbb Z^{\ell\times256}`$, $`u_i\in\mathbb Z^{k\times256}`$, $`v_i\in\mathbb Z^{k\times256}`$ and bits $`b_{i,t}\in\{0,1\}^{k\times256}`$, $`t\in[0,6)`$, with the virtual value $`w'_{1,i}=\sum_t2^tb_{i,t}\in\mathbb Z^{k\times256}`$. Relation $`\mathrm H(\bar B_z,\bar B_u)`$ holds when, for every $`\rho\lt k`$ and $`j\lt256`$:

```math
\sum_{s\lt\ell}\bigl(A_i[\rho,s]\circledast z_i[s]\bigr)_j-T_i[\rho]_j-2\gamma_2\,w'_{1,i}[\rho]_j-u_i[\rho]_j-q\,v_i[\rho]_j=0
\quad\text{over }\mathbb Z,
\qquad(3.1)
```
```math
\lvert z_i\rvert_\infty\le\bar B_z,\qquad \lvert u_i\rvert_\infty\le\bar B_u,\qquad w'_{1,i}\in[0,m_w)^{k\times256},
\qquad(3.2)
```
```math
\tilde c_i=\mathrm{SHAKE256}\bigl(\mu_i\Vert\mathrm{w1Encode}(w'_{1,i}),\,32\bigr).
\qquad(3.3)
```

The honest relation is $`\mathrm H(B_z,B_u)`$. The proven relation is **H-u** $`=\mathrm H(\bar B_z,\bar B_u)`$ with $`(\bar B_z,\bar B_u)=(2^{17},B_u)`$: $`z\in[-2^{17},2^{17})`$ (relaxed) and $`\lvert u\rvert_\infty\le B_u=190{,}465`$ (exact), the ranges proven inside the PIOP by the decompositions of $`z`$ and $`u`$ into a 16-bit low part, looked up in $`T_{16}=[0,2^{16})`$, and a small top digit with a leaf (Section 4.3); Akita widths play no part in them. Relation byte 0 (Section 10). The former relaxed relation H, $`(\bar B_z,\bar B_u)=(2^{17},2^{18})`$ of wire version 2, is superseded: H-u costs no column more and has a cheaper stage 2 (`docs/h-exact.md` Section 7, item 1). The second result is **H-exact** $`=\mathrm H(B_z,B_u)`$ (relation byte 1, Appendix A): the honest bounds $`\lvert z\rvert_\infty\le B_z=130{,}993`$ (the FIPS 204 line-13 bound) and $`\lvert u\rvert_\infty\le B_u`$ as the proven ones, with $`\zeta'=\zeta`$ as for H-u; one Boolean column more per position. The prover and the verifier take the relation as a parameter; the verifier MUST NOT read it from the proof (Section 10). No bound on $`v`$ is part of H: (3.1) is an integer equation, and $`v`$ is determined by the other terms. The proof nevertheless range-checks $`v`$ to $`v\in[-2^{31},2^{31})`$, because the field identity of Section 6 lifts to (3.1) only for a bounded quotient.

$`w_1'\in[0,44)`$ is equivalent, for Boolean $`b_t`$, to

```math
b_5\,(b_4+b_3b_2)=0,
\qquad(3.4)
```
verified by enumeration over $`[0,64)`$ (`docs/scripts/spec_checks.py`, run with `python3 -I`).

### 3.2 FIPS 204 verification implies H

Let $`(\tilde c_i,z_i,h_i)`$ be accepted by Algorithm 8. Set $`w'_i=\mathrm{ctr}_q(A_iz_i-c_it_{1,i}2^d)`$ and $`w'_{1,i}=\mathrm{UseHint}(h_i,w'_i)`$, so (3.3) is line 12 and $`w'_{1,i}\in[0,44)`$ by Algorithm 40. Line 13 gives $`\lvert z_i\rvert_\infty\le B_z`$. The UseHint lemma, checked exhaustively over $`r\in\mathbb Z_q`$ and $`h\in\{0,1\}`$ in `docs/scripts/spec_checks.py` (maximum exactly 190,465), gives $`u_i:=\mathrm{ctr}_q(w'_i-2\gamma_2w'_{1,i})`$ with $`\lvert u_i\rvert_\infty\le B_u`$. Then the left side of (3.1) without the $`qv`$ term is $`\equiv0\pmod q`$ because $`T_i\equiv c_it_{1,i}2^d`$, so $`v_i`$ is its exact quotient. Its magnitude is at most

```math
\ell\cdot256\cdot\tfrac{q-1}{2}\cdot B_z+B_T+2\gamma_2(m_w-1)+B_u=562{,}061{,}561{,}757{,}697\lt2^{49},
\qquad(3.4')
```
hence $`\lvert v_i\rvert_\infty\le B_v=67{,}068{,}447\lt2^{26}`$. The hint is not needed: H is hint-free.

### 3.3 H implies the approximate relation of the ML-DSA security proof

Write $`t_i=2^dt_{1,i}+t_{0,i}`$ with $`\lvert t_{0,i}\rvert_\infty\le2^{d-1}`$. From (3.1), modulo $`q`$,
```math
\bigl[A_i\;\big|\;t_i\;\big|\;I\bigr]\cdot\bigl(z_i,\;-c_i,\;c_it_{0,i}-u_i\bigr)=2\gamma_2\,w'_{1,i}\pmod q,
```
with $`\tilde c_i=\mathrm H(\mu_i\Vert\mathrm{w1Encode}(w'_{1,i}))`$ and $`c_i=\mathrm{SampleInBall}(\tilde c_i)`$. This is exactly the SelfTargetMSIS solution shape used in the EUF-CMA proof of Dilithium/FIPS 204, with infinity norm
```math
\zeta'=\max\bigl(\bar B_z,\;\bar B_u+2^{d-1}\tau\bigr).
\qquad(3.5)
```
The standard proof has $`\zeta=\max(\gamma_1-\beta,\,2\gamma_2+1+2^{d-1}\tau)=350{,}209`$. Relation H-u gives $`\zeta'=\max(2^{17},\,190{,}465+159{,}744)=350{,}209=\zeta`$: only the $`u`$ bound enters, and it is the honest one ($`2^{17}\lt350{,}209`$, so the relaxed $`z`$ bound costs nothing). The reduction loses nothing else (Section 12.3). (The superseded relaxed H had $`\zeta'=2^{18}+159{,}744=421{,}888`$, +20.5%.)

## 4. Witness tables and Akita groups

### 4.1 Committed tables

Lookup layout L16 (`docs/lookup.md`). Every committed table is one Akita polynomial of signed width 4 (Aerie's `Fp64SignedWidth4` parameters, one balanced base-16 inner digit) or signed width 16 (the same parameters with `log_commit_bound` 16 and one balanced base-$`2^{16}`$ inner digit, `docs/pcs-config.md`); polynomials with the same domain and width form one group with one opening point (Section 7.5). The width sizes the honest producer and the PCS parameters only; it is not a range proof (Akita Remark 10.41, falcon.md Section 5). Every range below is proven inside the PIOP (Section 4.3); $`z,u,v,\kappa^{(\cdot)}`$ are virtual (Section 5).

| Group | Polynomials | Domain (variables) | Width | Honest stored values | Meaning | Round |
| --- | --- | --- | ---: | --- | --- | ---: |
| $`G_{\rm main16}`$ | $`Z_{\rm lo},U_{\rm lo},V_{\rm lo},V_{\rm hi}`$ | main, $`d_{\rm main}`$ | 16 | $`[-2^{15},2^{15})`$ | the 16-bit parts of $`z,u,v`$, stored as $`x-2^{15}`$ | 1 |
| $`G_{\rm main4}`$ | $`Z_{\rm hi}`$, $`U_{\rm hi}`$, $`b_0,\ldots,b_5`$ | main | 4 | base 4, base 6, $`\{0,1\}`$ | the top digits of $`z,u`$; the w1Encode bits | 1 |
| $`G_1`$ | $`K^{(1)}_t`$, $`t\lt6`$ | $`D_1`$, $`d_1`$ | 16 | $`[-2^{15},2^{15})`$ | the level-1 carries, $`K=\kappa-2^{15}`$ | 2 |
| $`G_2`$ | $`y^{(2)}_t`$; $`K^{(2)}_t`$, $`t\lt6`$ | $`D_2`$, $`d_2`$ | 16 | $`\{0,1\}`$; $`[-2^{15},2^{15})`$ | level-2 parities; level-2 carries | 2 |
| $`G_3`$ | $`K^{(3)}`$ | $`D_3`$, $`d_3`$ | 16 | $`[-2^{15},2^{15})`$ | the level-3 carry | 2 |
| $`G_m`$ | $`m_0,\ldots,m_6`$ | $`2^{16}`$ cells | 4 | base 16 | the balanced base-16 digits of the multiplicity column $`m`$ of $`T_{16}`$ | 2 |

Under relation H-exact (Appendix A) $`G_{\rm main16}`$ holds $`S_{z,\rm lo}=B_{z,\rm lo}-2^{15}`$ in the slot of $`Z_{\rm lo}`$ (column `bz_lo`) and $`G_{\rm main4}`$ holds nine polynomials: the bit $`B_{z,\rm hi}`$ in the slot of $`Z_{\rm hi}`$ (`bz_hi`), $`U_{\rm hi}`$, $`b_0,\ldots,b_5`$ and the bit $`a_z`$ (`a_z`), so 13 main-domain columns, 13,434.5 committed elements per signature at $`N=2^{15}`$, 39 polynomials, and its own catalog rows (`mldsa-44-l16x-batch-<N*>-mult-final`, `docs/pcs-config.md`).

Per signature: $`1024\cdot12=12{,}288`$ main-domain elements (was 23,552) plus, at $`N=2^{15}`$, 96 level-1, 12 level-2, 0.5 level-3 and 14 multiplicity elements: 12,410.5 committed elements (was 23,865), each one Akita inner digit. Round 1 is committed before the bridge seed, round 2 after it (Section 10); $`m`$ counts the carries, which exist only after the seed, so $`G_m`$ is a round-2 group. All six group descriptors are bound in event 0, so both rounds share one commitment contract and one batched opening: Akita Section 4.6 admits commitments formed before their batch is known under a common certified root alphabet, Akita freezes each group's source bound in its own profile (so the two widths share one batch), and Aerie's per-group `commit_group` is transcript-free except for the final group, here $`G_m`$, which commits against the precommitted profiles. Group and polynomial order on the wire and in the catalog key is the table order. Every physical table is zero-extended to at least $`2^{13}`$ cells (Akita `PCS_MIN_CELLS_LOG2`).

### 4.2 Padded rows

For $`i\ge N`$ every value is zero: $`z=u=v=b_t=0`$, so $`w_1'=0`$ and (3.1) holds with $`A_i=T_i=0`$. Padded bridge positions are zero bits, padded carries and parities are zero. By (4.1) and (4.2) the digits of zero are $`Z_{\rm lo}=V_{\rm lo}=-2^{15}`$, $`U_{\rm lo}=-31{,}540`$ ($`B_{u,\rm lo}=1{,}228`$, $`H_u=3`$), $`Z_{\rm hi}=U_{\rm hi}=V_{\rm hi}=0`$ and $`K=-2^{15}`$ for a carry; padded cells are thus nonzero and MUST NOT be required to be zero (Aerie 3.4). The padding slots of the lookup stacks (Section 4.3) hold $`-2^{15}`$ (looked-up value 0). The Keccak side contains live records only.

### 4.3 Range proofs: the lookup into $`T_{16}`$ and two leaves

A value $`X`$ with digits $`Z_0,\ldots,Z_{m-1}`$ of bases $`b_0,\ldots,b_{m-1}`$ is the virtual table
```math
X=\sum_{k\lt m}W_kZ_k+C_X,\qquad W_k=\prod_{l\lt k}b_l,\qquad C_X=\sum_{k\lt m}W_k\tfrac{b_k}{2}-\epsilon_X\tfrac{B}{2},\qquad B=\prod_{k\lt m}b_k,
\qquad(4.1)
```
with $`\epsilon_X=1`$ for a signed and $`0`$ for an unsigned value. When every $`Z_k`$ lies in its balanced alphabet $`[-b_k/2,b_k/2-1]`$, (4.1) is a bijection onto $`[-B/2,B/2-1]`$ or $`[0,B-1]`$:

| Value | Bases | Range proven | $`C_X`$ | Mechanism |
| --- | --- | --- | ---: | --- |
| $`z`$ | $`2^{16},4`$ | $`[-2^{17},2^{17}-1]`$, so $`\bar B_z=2^{17}`$ | $`2^{15}`$ | $`Z_{\rm lo}`$ looked up; $`Z_{\rm hi}`$ leaf `leaf4` and link |
| $`z`$ (H-exact) | bit-slack form (4.3): $`2^{16}`$, then the bits $`B_{z,\rm hi}`$ (weight $`2^{16}`$) and $`a_z`$ (weight 130,915) | $`[-130{,}993,130{,}993]`$, so $`\bar B_z=B_z`$ | $`-98{,}225`$ | $`S_{z,\rm lo}`$ looked up; `bool` zero-checks at $`r^{(2)}`$ (7.5) |
| $`u`$ | step form (4.2): $`2^{16}`$, then one base-6 digit of weight 63,079 | $`[-190{,}465,190{,}465]`$, so $`\bar B_u=B_u`$ | 31,540 | $`U_{\rm lo}`$ looked up; $`U_{\rm hi}`$ leaf `leaf6` and link |
| $`v`$ | $`2^{16},2^{16}`$ | $`[-2^{31},2^{31}-1]`$ | $`2^{15}`$ | both looked up |
| $`\kappa^{(1)}_t,\kappa^{(2)}_t,\kappa^{(3)}`$ | $`2^{16}`$ (unsigned) | $`[0,2^{16}-1]`$ | $`2^{15}`$ | looked up |

So $`z=Z_{\rm lo}+2^{16}Z_{\rm hi}+2^{15}`$, likewise $`v`$, and $`\kappa=K+2^{15}`$. The residual $`u`$ (relation H-u) is not mixed-radix but a step form: with $`B_{u,\rm lo}=U_{\rm lo}+2^{15}\in[0,2^{16})`$ and $`H_u=U_{\rm hi}+3\in[0,6)`$,
```math
u=B_{u,\rm lo}+63{,}079\,H_u-190{,}465=U_{\rm lo}+63{,}079\,U_{\rm hi}+31{,}540,
\qquad(4.2)
```
whose six blocks $`[63{,}079H,63{,}079H+2^{16})`$ tile $`[0,2B_u]`$ without gap because $`5\cdot63{,}079+2^{16}=2B_u+1`$ and $`63{,}079\le2^{16}`$ (exact enumeration in `agg-witness` `digits` tests; no other base $`c\le16`$ admits one digit). The honest encoder takes $`H_u=\min(\lfloor(u+B_u)/63{,}079\rfloor,5)`$; digits are not unique in the overlaps, which is harmless because only the value enters an identity. Under relation H-exact $`z`$ is the bit-slack form (`docs/h-exact.md` 2.1): with $`B_{z,\rm lo}=S_{z,\rm lo}+2^{15}\in[0,2^{16})`$ and the bits $`B_{z,\rm hi},a_z`$,
```math
z=B_{z,\rm lo}+2^{16}B_{z,\rm hi}+130{,}915\,a_z-130{,}993=S_{z,\rm lo}+2^{16}B_{z,\rm hi}+130{,}915\,a_z-98{,}225,
\qquad(4.3)
```
whose four blocks tile $`[0,2B_z]`$ because $`130{,}915+2^{17}=2B_z+1`$ and $`130{,}915\le2^{17}`$ (exhaustive check in `agg-witness`; `spec_checks.py` item 13). The honest encoder takes $`a_z=[z+B_z\ge2^{17}]`$ (i.e. $`z\ge79`$), then $`B_{z,\rm hi}`$ and $`B_{z,\rm lo}`$ from the rest; the digits of zero are $`(S_{z,\rm lo},B_{z,\rm hi},a_z)=(32{,}689,1,0)`$. Every base-$`2^{16}`$ digit (stored $`S`$, looked-up value $`f=S+2^{15}`$) is proven in $`T_{16}`$ by one LogUp identity (Haböck, ePrint 2022/1530, Lemma 5) with the GKR fractional sum-check (Papini and Haböck, ePrint 2023/1284) over four *stacks* (Section 7.0): $`[Z_{\rm lo}\mid U_{\rm lo}\mid V_{\rm lo}\mid V_{\rm hi}]`$ on the main domain, $`K^{(1)}_{0..5}`$ in 8 slots on $`D_1`$, $`K^{(2)}_{0..5}`$ in 8 slots on $`D_2`$, $`K^{(3)}`$ on $`D_3`$ (missing slots are public, stored $`-2^{15}`$), against the multiplicity column $`m`$ of $`T_{16}`$, at two independent challenges $`X_1,X_2\in K`$. One table and one identity: tables MUST NOT be merged by integer offsets (a committed value is an arbitrary $`\mathbb F_p`$ element and would match a foreign entry; `docs/lookup.md` Section 1). $`m`$ needs no range proof: Lemma 5 holds for any $`m\in\mathbb F_p^{T_{16}}`$ because $`p`$ exceeds the number of looked-up cells. The two small digits keep Aerie's leaf predicates: for a digit $`Z`$ the folded table $`P_Z(x)=Z(x)(Z(x)+1)`$ (virtual, extended separately) satisfies $`Q_b(P_Z)=0`$ on the Boolean domain exactly when $`Z\in[-b/2,b/2-1]`$, with $`Q_4(P)=P(P-2)`$ and $`Q_6(P)=P(P-2)(P-6)`$ (Aerie (3.7); $`Q_6`$ is $`Q_8`$ without its last root, and the halving trick works for every even alphabet symmetric about $`-1/2`$); exactness holds over $`\mathbb F_p`$ because $`Z(Z+1)=t`$ has at most two roots, so $`Q_6(Z(Z+1))=0`$ iff $`Z\in[-3,2]`$. The checks are `leaf4`, `leaf6` (stage 2) and `link` (stage 3) of Section 7.

The verifier MUST reject a digit layout other than this one (digit layout id 1) and a lookup layout (stacks, table width, challenge count, storage shift) other than the one above. The base-16 digit layout of wire version 0 (55 digits with `b16_prod`, 95 leaves and 55 links) is retired.

## 5. Virtual polynomials

All virtual tables are defined on Boolean points and extended multilinearly as tables; a product of tables is never the product of extensions.

| Table | Domain | Definition | Honest value |
| --- | --- | --- | --- |
| $`w_1'(j,\rho,i)`$ | main | $`\sum_{t\lt6}2^tb_t(j,\rho,i)`$ | $`[0,44)`$ |
| $`(Az)(j,\rho,i)`$ | main | $`\sum_s(A_i[\rho,s]\circledast z_i[s])_j`$ | $`\lvert\cdot\rvert\lt2^{49}`$ |
| $`\Delta(j,\rho,i)`$ | main | $`(Az)-T-2\gamma_2w_1'-u-qv`$ | 0 |
| $`y^{(1)}_t(\rho_1,\beta)`$ | $`D_1`$ | $`\sum_{\zeta}M_1[\rho_1,\zeta]\,b_t(\zeta,\beta)-2\kappa^{(1)}_t(\rho_1,\beta)`$ | $`\{0,1\}`$ |
| $`Y(\rho_3,\sigma)`$ | $`D_3`$ | public $`Y_\sigma[\rho_3]`$ | $`\{0,1\}`$ |
| $`z,u,v,\kappa^{(1)}_t,\kappa^{(2)}_t,\kappa^{(3)}`$ | own domain | (4.1) from their digits | Section 4.3 |
| $`P_Z`$ for $`Z\in\{Z_{\rm hi},U_{\rm hi}\}`$ | main | $`Z(Z+1)`$ | $`\{0,2\}`$, $`\{0,2,6\}`$ |

$`y^{(1)}_t`$ is an integer identity in the committed $`b_t`$ and $`K^{(1)}_t`$: the sum has at most 2,047 terms (row weight bound, Section 9.2) and $`\kappa^{(1)}_t\in[0,2^{16})`$ (Section 4.3), so the $`\mathbb F_p`$ definition agrees with the integer one. Affine combinations of opened values ($`z,u,v,\kappa`$ by (4.1), $`w_1'(r)`$, $`y^{(1)}_t(r)`$ through its defining sum-check) are evaluated by the verifier from the opened components.

## 6. Residual: committed quotient instead of the divisibility projection

Aerie proves $`q\mid\Delta`$ with a seeded projection $`\Pi_{\rm res}:\mathbb F_p^{nN^\star}\to\mathbb F_p^{31}`$ (cores with entries uniform in $`[-12,12]`$) whose integer outputs must stay below $`p/2`$. For ML-DSA the residual without the $`qv`$ term, $`\Delta_0=(Az)-T-2\gamma_2w_1'-u`$, has honest coefficients of root-mean-square size about $`2^{42.4}`$ (worst case $`\lt2^{49}`$ by (3.4')), so an honest output has standard deviation about $`2^{68.4}`$ at $`nN^\star=2^{25}`$ and wraps modulo $`p`$ ($`p/2\approx2^{63}`$) with probability close to one: the field is too small for the Falcon-style projection of an ML-DSA residual, whose entries are $`2^{18}`$ times larger than Falcon's.

Therefore this protocol commits the quotient $`v`$ through its digits (Section 4) and proves (3.1) as the field identity $`\Delta=0`$ on the Boolean domain. Lifting: under the ranges proven in Section 4.3, $`\lvert(Az)\rvert\le2^{10}\cdot2^{22}\cdot2^{17}=2^{49}`$, $`\lvert T\rvert\lt2^{29}`$, $`2\gamma_2w_1'\lt2^{24}`$, $`\lvert u\rvert\le2^{18}`$ and $`\lvert qv\rvert\lt2^{23}\cdot2^{31}=2^{54}`$, so
```math
\lvert\Delta^{\mathbb Z}\rvert\lt2^{49}+2^{29}+2^{24}+2^{18}+2^{54}\lt2^{54.1}\lt p/2,
\qquad(6.1)
```
and $`\Delta=0`$ in $`\mathbb F_p`$ implies $`\Delta^{\mathbb Z}=0`$ over the integers, which is (3.1). The range proof of $`v`$ is what gives the identity content: $`q`$ is invertible modulo $`p`$, so an unbounded $`v=q^{-1}\Delta_0\bmod p`$ would satisfy $`\Delta=0`$ for every $`z,u,w_1'`$. Any $`\lvert v\rvert\lt(p/2-2^{50})/q\approx2^{40}`$ would do; the two looked-up halves prove $`[-2^{31},2^{31})`$ (the honest width is $`2^{26}`$). The cost is two 16-bit columns per position. The identity is tested at the single point $`r^{(2)}`$ inside the stage-3 instance (7.6).

No norm projection is needed: H has only infinity-norm bounds, each proven per coordinate by the lookup and the leaves of Section 4.3 or by (3.4).

## 7. The sum-check stages

### 7.0 The lookup (event 7)

After both commitment rounds, the bridge seed and the Keccak side (Section 10), the verifier samples $`X_1,X_2\in K`$ (one vector challenge). For each $`X_j`$ and each of the four stacks of Section 4.3 (stored lane $`g`$ of $`2^{D_s}`$ cells, $`D_s`$ the stack's variables plus $`\log_2`$ of its slot count) and the table, a binary fraction tree of projective pairs $`(p,q)`$ is formed: stack leaves $`(1,(X_j-2^{15})-g)`$, table leaves $`(m(t),X_j-t)`$, node $`y`$ of layer $`k`$ with children $`y`$ and $`y+2^k`$ of layer $`k+1`$, $`p=p_0q_1+p_1q_0`$, $`q=q_0q_1`$. Session 1: the prover sends $`p_1(0),p_1(1),q_1(0),q_1(1)`$ of every tree; the verifier forms the roots, checks per challenge $`\sum_{\rm stacks}p/q=p_T/q_T`$ with every root denominator nonzero, and samples $`\mu_1`$. Session $`s=2,\ldots,D`$, $`D=\max(d_{\rm main}+2,16)`$ the deepest tree (the main stack, or the $`T_{16}`$ tree of depth 16 when $`d_{\rm main}+2\lt16`$, i.e. at $`N^\star=8`$; `LookupShape::max_depth`): one batched sum-check of $`s-1`$ rounds and degree 3 over every tree of depth $`\ge s`$ (coefficients $`\gamma`$, two per tree), reducing the layer-$`(s-1)`$ claims at the common point to layer $`s`$; the prover sends $`P_0,P_1,Q_0,Q_1`$ at the round point $`\rho`$ per tree of depth $`\gt s`$ and the two leaf-lane values per stack or table ending at $`s`$ (shared by both challenges, whose leaf terms are merged into one instance set); the verifier checks the terminal, samples $`\mu_s`$ and continues at $`(\rho,\mu_s)`$. Each stack leaves one claim on its stored lane at a point $`\pi_j=(\pi_{j,\rm low},\pi_{j,\rm top})`$, the table one claim $`m(\pi_m)`$, $`\pi_m\in K^{16}`$. The prover then sends the seven digit claims $`m_k(\pi_m)`$; the verifier checks $`\sum_k16^km_k(\pi_m)=m(\pi_m)`$, and $`G_m`$ is opened at $`\pi_m`$. The stack claims are reduced to $`r^{(3)}`$ in stage 3 (7.7). Details, counts and costs: `docs/lookup.md` Sections 1 and 4.

### 7.1 Overview

After both commitment rounds, the bridge seed and the lookup (Sections 7.0, 10), the verifier samples one point $`\tau\in K^{d_{\rm main}}`$; $`\tau_t`$ denotes its last $`d_t`$ coordinates ($`t=1,2,3`$). Three batched sum-checks follow, each of $`d_{\rm main}`$ rounds with terminal point $`r^{(s)}`$, $`s=1,2,3`$; $`r^{(s)}_t`$ is the suffix of $`r^{(s)}`$ for domain $`D_t`$, and for a table $`Z`$ on $`D_t`$ the points $`\tau_Z,r^{(s)}_Z`$ denote these suffixes. Instances on smaller domains are inactive in the leading rounds and lifted by $`2^{-(d_{\rm main}-d_{\rm loc})}`$ exactly as in Aerie 7.1; every instance uses its own canonical variable order (Section 1) and the stage batches instances with independent coefficients $`\gamma^{(s)}`$ in the listed order. Ordinary compressed rounds send $`\deg`$ elements (Aerie 10.3, omitting the linear coefficient).

| Stage | Instances | Max individual degree | Terminal claims |
| --- | --- | ---: | --- |
| 1 | 26: bridge Booleanity and levels 2, 3; w1Encode Booleanity and range | 4 | $`y^{(1)}_t(r^{(1)}_1)`$ (virtual), $`y^{(2)}_t(r^{(1)}_2)`$, $`b_t(r^{(1)})`$ (18) |
| 2 | 8: level-1 definition; 2 range leaves | 4 | $`b_t(r^{(2)})`$, $`P_Z(r^{(2)})`$ for $`Z_{\rm hi},U_{\rm hi}`$ (8) |
| 3 | 44: ring identity and functional; 2 range links; 37 evaluation reductions; 4 lookup-stack reductions | 3 | the 31 committed tables of $`G_{\rm main16},G_{\rm main4},G_1,G_2,G_3`$ at $`r^{(3)}`$ and its suffixes |

Every committed table except the multiplicity digits is opened once, at the stage-3 terminal of its domain; earlier claims (including the lookup's stack claims) are reduced to $`r^{(3)}`$ in stage 3. The multiplicity digits are opened at $`\pi_m`$. The ring identity (3.1) is tested at the single point $`r^{(2)}`$ inside the stage-3 instance (7.6).

### 7.2 Pre-stage claims

Before $`\gamma^{(1)}`$ the prover sends the 13 evaluations $`y^{(2)}_t(\tau_2)`$, $`K^{(2)}_t(\tau_2)`$ for $`t\lt6`$, and $`K^{(3)}(\tau_3)`$; the verifier forms $`\kappa^{(2)}_t(\tau_2)`$ and $`\kappa^{(3)}(\tau_3)`$ by (4.1) ($`\kappa=K+2^{15}`$). The verifier computes $`Y(\tau_3)=\sum_{\rho_3,\sigma}\mathrm{eq}(\tau_3,(\rho_3,\sigma))Y_\sigma[\rho_3]`$ from the public final vectors ($`4096S`$ terms).

### 7.3 Stage 1 (degree 4)

Instances, in $`\gamma^{(1)}`$ order:

1. `bool_y1[t]`, $`t\lt6`$, domain $`D_1`$, degree 3:
```math
0=\sum_{x\in D_1}\mathrm{eq}(\tau_1,x)\,y^{(1)}_t(x)\bigl(y^{(1)}_t(x)-1\bigr).
\qquad(7.1)
```
2. `level2[t]`, $`t\lt6`$, domain $`D_1`$, degree 2:
```math
y^{(2)}_t(\tau_2)+2\kappa^{(2)}_t(\tau_2)=\sum_{\zeta\in D_1}w_2(\zeta)\,y^{(1)}_t(\zeta),
\qquad
w_2(\rho_1,i_{\rm mid},\sigma)=\mathrm{eq}(\tau_{2,\sigma},\sigma)\,c_2(\rho_1,i_{\rm mid}),
\qquad(7.2)
```
with $`c_2(\zeta_2)=\sum_{\rho_2}M_2[\rho_2,\zeta_2]\,\mathrm{eq}(\tau_{2,\rho},\rho_2)`$ over the in-block index $`\zeta_2=(\rho_1,i_{\rm mid})`$.
3. `level3`, domain $`D_2`$ summed over the six lanes, degree 2:
```math
Y(\tau_3)+2\kappa^{(3)}(\tau_3)=\sum_{t\lt6}\sum_{x\in D_2}w_{3,t}(x)\,y^{(2)}_t(x),
\qquad
w_{3,t}(\rho_2,\sigma)=\mathrm{eq}(\tau_{3,\sigma},\sigma)\,c_3(t,\rho_2),
\qquad(7.3)
```
with $`c_3(t,\rho_2)=\sum_{\rho_3}M_3[\rho_3,(t,\rho_2)]\,\mathrm{eq}(\tau_{3,\rho},\rho_3)`$.
4. `bool_y2[t]`, $`t\lt6`$, domain $`D_2`$, degree 3: $`0=\sum_{x\in D_2}\mathrm{eq}(\tau_2,x)\,y^{(2)}_t(x)(y^{(2)}_t(x)-1)`$.
5. `bool_b[t]`, $`t\lt6`$, main domain, degree 3: $`0=\sum_x\mathrm{eq}(\tau,x)\,b_t(x)(b_t(x)-1)`$.
6. `range_w1`, main domain, degree 4:
```math
0=\sum_x\mathrm{eq}(\tau,x)\,b_5(x)\bigl(b_4(x)+b_3(x)b_2(x)\bigr).
\qquad(7.4)
```

Terminal: the prover sends $`y^{(1)}_t(r^{(1)}_1)`$ (six virtual claims, materialized on $`D_1`$ by integer parities), $`y^{(2)}_t(r^{(1)}_2)`$ and $`b_t(r^{(1)})`$; the verifier checks the batched terminal with the equality factors, $`w_2(r^{(1)}_1)`$ and $`w_{3,t}(r^{(1)}_2)`$ (Section 7.6).

### 7.4 Stage 2 (degree 4)

Before $`\gamma^{(2)}`$ the prover sends the 6 values $`K^{(1)}_t(r^{(1)}_1)`$ and the verifier forms $`\kappa^{(1)}_t(r^{(1)}_1)`$ by (4.1). Instances:

1. `def_y1[t]`, main domain:
```math
y^{(1)}_t(r^{(1)}_1)+2\kappa^{(1)}_t(r^{(1)}_1)=\sum_{x}w_1(x)\,b_t(x),
\qquad
w_1(j,\rho,i)=\mathrm{eq}(r^{(1)}_{1,\beta},\beta_1(i))\,c_1(j,\rho,i_{\rm lo}),
\qquad(7.5)
```
$`c_1(\zeta_1)=\sum_{\rho_1}M_1[\rho_1,\zeta_1]\,\mathrm{eq}(r^{(1)}_{1,\rho},\rho_1)`$, $`\beta_1(i)=(i_{\rm mid},\sigma)`$.
2. `leaf4[Z_hi]`, degree 3, and `leaf6[U_hi]`, degree 4: $`0=\sum_x\mathrm{eq}(r^{(1)},x)\,Q_b(P_Z(x))`$ (relaxed H's `leaf8` had degree 5). Under H-exact only `leaf6[U_hi]` (the bit $`B_{z,\rm hi}`$ has no leaf): 7 instances, still degree 4.

Terminal claims $`b_t(r^{(2)})`$ and $`P_Z(r^{(2)})`$ for the 2 digits (H-exact: 1); the verifier evaluates $`Q_b`$ on the claimed $`P_Z`$ values and the equality factors.

### 7.5 Stage 3 (degree 3) and the opening

Instances, in $`\gamma^{(3)}`$ order:

1. `ring`, degree 2, the first sum in order $`(j',s,i)`$ and the second in order $`(j,\rho,i)`$, with $`z,v`$ the (4.1) and $`u`$ the (4.2) combinations of their two digits:
```math
T(r^{(2)})=\sum_{j',s,i}w_A(j',s,i)\,z(j',s,i)-\sum_{j,\rho,i}\mathrm{eq}\bigl(r^{(2)},(j,\rho,i)\bigr)\bigl(2\gamma_2w_1'+u+qv\bigr)(j,\rho,i),
\qquad(7.6)
```
with $`w_A`$ from Section 7.6 and $`T(r^{(2)})=\sum_{j,\rho,i}\mathrm{eq}(r^{(2)},(j,\rho,i))T(j,\rho,i)`$ ($`k\cdot256`$ multiply-adds per live record). Equation (7.6) is $`\sum_x\mathrm{eq}(r^{(2)},x)\Delta(x)=0`$ with $`(Az)`$ pulled back through $`A`$; the tables were fixed before $`r^{(2)}`$, so a nonzero $`\Delta`$ passes with probability at most $`d_{\rm main}/|K|`$.
2. `link[Z]` for $`Z_{\rm hi},U_{\rm hi}`$ (H-exact: $`U_{\rm hi}`$ only), degree 3 (Aerie (7.8)): $`P_Z(r^{(2)})=\sum_x\mathrm{eq}(r^{(2)},x)\,Z(x)(Z(x)+1)`$. Under H-exact then `bool[B_zhi]` and `bool[a_z]`, degree 3: $`0=\sum_x\mathrm{eq}(r^{(2)},x)\,B(x)(B(x)-1)`$ (the two columns are committed in round 1, before $`r^{(2)}`$, and opened at $`r^{(3)}`$ anyway, so no reduction is added; `docs/h-exact.md` 3.4). H-exact's ring identity reads $`z`$ by (4.3).
3. Evaluation reductions $`f(\pi)=\sum_x\mathrm{eq}(\pi,x)f(x)`$, degree 2: $`b_t`$ from $`r^{(1)}`$ and from $`r^{(2)}`$; $`y^{(2)}_t`$ from $`\tau_2`$ and from $`r^{(1)}_2`$; $`K^{(2)}_t`$ from $`\tau_2`$; $`K^{(3)}`$ from $`\tau_3`$; $`K^{(1)}_t`$ from $`r^{(1)}_1`$ (37 instances).
4. Lookup-stack reductions, one per stack $`j`$ with columns $`f_{j,c}`$ (stored) and the stack claim $`(\pi_j,g_j)`$ of Section 7.0, degree 2:
```math
g_j+2^{15}\sum_{c\ge{\rm cols}_j}\mathrm{eq}(\pi_{j,\rm top},c)=\sum_x\mathrm{eq}(\pi_{j,\rm low},x)\sum_{c\lt{\rm cols}_j}\mathrm{eq}(\pi_{j,\rm top},c)\,f_{j,c}(x),
\qquad(7.7)
```
the left side removing the public padding slots (stored $`-2^{15}`$).

Terminal: the 31 committed values at $`r^{(3)}`$ and its suffixes (H-exact: 32); the verifier forms (4.1), (4.2), (4.3), $`w_1'`$, $`w_A(r^{(3)})`$, the equality factors (of $`r^{(1)},r^{(2)},\tau`$ and the four $`\pi_{j,\rm low}`$), $`Z(Z+1)`$ per link and $`B(B-1)`$ per Booleanity check, and checks the batched terminal. The $`\gamma^{(3)}`$ vector has 44 coefficients (H-exact: 45: `ring`, one link, two `bool`, 37 reductions, 4 stack reductions).

One Akita opening authenticates the 31 (H-exact: 32) evaluations at the five group points of $`r^{(3)}`$ and the 7 multiplicity-digit claims at $`\pi_m`$. No committed polynomial carries a claim at a second point.

### 7.6 Public weight tables

**Ring weight (pulled back through the $`k\times\ell`$ matrices).** For $`r=(r_{\rm coef},r_{\rm row},r_{\rm sig})=r^{(2)}`$ and a polynomial $`a\in\mathbb Z^{256}`$ define, as in Aerie (8.2),
```math
K_{a,r_{\rm coef}}[j']=\sum_{a'=0}^{255}a[a']\,(-1)^{\lfloor(a'+j')/256\rfloor}\,\mathrm{eq}\bigl(r_{\rm coef},(a'+j')\bmod256\bigr),
\qquad(7.8)
```
so that $`\sum_j\mathrm{eq}(r_{\rm coef},j)(a\circledast z)_j=\sum_{j'}K_{a,r_{\rm coef}}[j']z[j']`$. Then
```math
w_A(j',s,i)=\mathrm{eq}(r_{\rm sig},i)\,K_{\tilde A_{i,s},\,r_{\rm coef}}[j'],
\qquad
\tilde A_{i,s}=\sum_{\rho\lt k}\mathrm{eq}(r_{\rm row},\rho)\,A_i[\rho,s]\in K^{256},
\qquad(7.9)
```
by linearity of (7.8) in $`a`$; padded rows are zero. Prover materialization, after stage 2: for each live $`i`$ and $`s\lt\ell`$, form $`\tilde A_{i,s}`$ and apply Aerie's factorized negacyclic transform (8.5) to (8.7) once; about $`1.4\times10^8`$ $`K`$-operations at $`N=2^{15}`$ and $`\ell\cdot256\cdot N^\star`$ storage. A quadratic scan per coefficient MUST NOT be used.

Verifier evaluation at $`r^{(3)}=(r^{(3)}_{\rm coef},r^{(3)}_{\rm col},r^{(3)}_{\rm sig})`$: compute the 256 values $`\Lambda_{a'}`$ of Aerie (8.8) to (8.10) from $`(r^{(2)}_{\rm coef},r^{(3)}_{\rm coef})`$ once, then
```math
w_A(r^{(3)})=\sum_{i\lt N}\mathrm{eq}(r^{(2)}_{\rm sig},i)\,\mathrm{eq}(r^{(3)}_{\rm sig},i)\sum_{\rho,s}\mathrm{eq}(r^{(2)}_{\rm row},\rho)\,\mathrm{eq}(r^{(3)}_{\rm col},s)\sum_{a'}A_i[\rho,s][a']\,\Lambda_{a'},
\qquad(7.10)
```
$`k\ell\cdot256`$ multiply-adds per live record ($`A`$ entries fit 23 bits, so lazy 128-bit accumulation applies), streamed over the expanded keys one $`A_i`$ at a time.

**Bridge weights.** For level $`t`$ at the point $`\pi`$ on $`D_t`$ the weight is $`\mathrm{eq}(\pi_\beta,\beta)\cdot c_t(\zeta)`$ with $`c_t(\zeta)=\sum_\rho M_t[\rho,\zeta]\mathrm{eq}(\pi_\rho,\rho)`$: one $`J_t`$-entry vector per level and point ($`k_tJ_t`$ additions in $`K`$), shared by all lanes, blocks and segments; the verifier evaluates it at the terminal as $`\mathrm{eq}(\pi_\beta,r_\beta)\sum_{(\rho,\zeta)\in M_t}\mathrm{eq}(\pi_\rho,\rho)\mathrm{eq}(r_\zeta,\zeta)`$. Zero-padded in-block positions have zero weight.

## 8. Keccak sponge circuit interface

Per live record the circuit proves (3.3): a SHAKE256 sponge (rate 1,088 bits, capacity 512) absorbing $`\mu_i\Vert\mathrm{w1Encode}(w'_{1,i})`$ (64 + 768 bytes) and squeezing 32 bytes, that is 7 Keccak-f[1600] permutations $`P_1,\ldots,P_7`$:

| Permutation | Absorbed block (136 bytes) |
| --- | --- |
| $`P_1`$ | $`\mu_i`$ (64 bytes, public) $`\Vert`$ w1Encode bytes 0..71 |
| $`P_2\ldots P_6`$ | w1Encode bytes $`72+136(m-2)\ldots`$, 136 bytes each (680 in total) |
| $`P_7`$ | w1Encode bytes 752..767 $`\Vert\ \texttt{0x1F}\ \Vert\ \texttt{0x00}^{118}\ \Vert\ \texttt{0x80}`$ |

The initial state is zero; the state before $`P_m`$ is the output of $`P_{m-1}`$ XOR the block in the low 1,088 bits; $`\tilde c_i`$ is the low 256 bits of the output of $`P_7`$. Bit order: w1Encode is $`\big\Vert_{\rho\lt k}\mathrm{SimpleBitPack}(w_1'[\rho],43)`$ (FIPS 204 Algorithms 16 and 28), so input bit $`1536\rho+6j+t`$ is $`b_t(j,\rho,i)`$; byte $`\beta`$ of a block is XORed into state bits $`8\beta\ldots8\beta+7`$, least significant first, in Keccak's little-endian lane order. The wiring is a fixed index permutation.

Interface of the circuit crate (`agg-keccak`), per segment $`\sigma`$:

- `Input` columns: the 6,144 bits per live record, in the order $`(t,j,\rho)`$ above; they are the GF(2) copy $`b'`$ of the bits $`b`$.
- `public_input`: $`\mu_i`$ (512 bits per record); `fixed`: capacity zeros and the padding bytes; `publish`: $`\tilde c_i`$ (256 bits per record).
- Witness: the 38,400 chi AND outputs per permutation (the retired `c2d0c24` encoder's chi-only witness with implicit states); all absorptions and the chain are `xor` linear expressions. Committed bits per record: $`6{,}144+7\cdot38{,}400=274{,}944`$.
- One Keccak gate type (one $`A_0`$ block, $`A=I\otimes A_0`$); per record a 7-slot union circuit with copy constraints between consecutive permutations.
- Sub-batch: the $`7\cdot\min(N-2^{13}\sigma,2^{13})\le57{,}344`$ permutations of the segment in one Flock proof of $`2^{16}`$ slots (Ligerito profile `Fast` as the engineering baseline; LaBinius as the target commitment). Unused slots hold a fixed all-zero permutation instance.
- Bridge slot: one extra slot per segment of a dedicated type `bridge`, $`k_{\log}=13+d_{\rm seg}`$ (lanes padded from 6 to 8; 26 at a full segment). Its witness is a copy of every live record's 48 w1Encode words (about $`3.9\times10^5`$ 128-bit words per full segment), tied to the records' `Input` words by Flock's word-level copy constraints (`Cell`, `word_base`); padded records and lanes 6, 7 are fixed zero. Its linear rows are the 4,096 composite rows $`M^{(\sigma)}=M_3M_2M_1`$ over GF(2) with public right-hand side $`Y_\sigma`$; $`y^{(1)},y^{(2)}`$ are not materialized (no `define_linear`), because Flock HEAD's lincheck is per block (`LincheckCircuit` sees one slot's columns) and the whole witness is committed once before `bind_statement_circuit`, so nothing seed-dependent may be a witness. The bridge type's `LincheckCircuit` computes the column marginal $`\mathrm{eq}^{\mathsf T}M^{(\sigma)}`$ by the factored walker $`M_1^{\mathsf T}(M_2^{\mathsf T}(M_3^{\mathsf T}\mathrm{eq}))`$ in $`\mathbb F_{2^{128}}`$ (characteristic 2 makes the GF(2) and field products agree), about $`7\times10^8`$ additions per segment, without materializing $`M^{(\sigma)}`$. The circuit digest bound by `bind_statement_circuit` MUST cover $`\mathsf{sd}`$; $`Y_\sigma`$ is a public input. $`k_{\log}=26`$ is below Flock's `MAX_K_LOG = 40` but unmeasured; the slot's eq and marginal vectors take 2 GB (risks R3).
- Driver: commit the union witness (Keccak slots and bridge slot; nothing in it depends on $`\mathsf{sd}`$), hand the commitment to the PIOP transcript (event 2), receive $`\mathsf{sd}`$ and $`\mathsf{fs}_\sigma`$, derive the bridge rows, bind the statement (circuit digest with $`\mathsf{sd}`$; public $`\mu_i,\tilde c_i,Y_\sigma`$), prove with the challenger of Section 10.

Expected cost per segment (this machine's calibration): 2.3 s and 9.4 GB (retired encoder, $`2^{16}`$ permutations) plus the bridge slot (estimate: about 1 s and 2 GB); proof about 500 KiB; verification about 10 ms.

## 9. Parity bridge

### 9.1 Matrices and seed

Three public sparse matrices $`M_t\in\{0,1\}^{m_t\times J_t}`$ with $`(J_t,m_t,k_t)`$ from the table in Section 1. Column $`\zeta`$ of $`M_t`$ is the indicator of a uniformly random $`k_t`$-subset of $`[m_t]`$, columns independent. Derivation from the 32-byte seed $`\mathsf{sd}`$ (Section 10): for level $`t`$ and column $`\zeta`$, read 16-bit little-endian words from $`\mathrm{cSHAKE256}(\mathsf{sd}\Vert\mathrm{LE64}(t)\Vert\mathrm{LE64}(\zeta),\ \cdot,\ \varepsilon,\ \texttt{agg-mldsa/v1/bridge})`$, keep the low $`\log_2m_t`$ bits of each word (13, 13, 12; uniform since $`m_t`$ is a power of two), skip a value already chosen for this column, and stop after $`k_t`$ distinct rows. Neither side transmits the matrices; the same $`M_t`$ serves every block, lane and segment.

### 9.2 The two sides

Let $`b_t(\cdot,\beta)`$ be the $`J_1`$ bits of lane $`t`$, block $`\beta`$ (Section 1), zero on padded positions. GF(2) side (Flock, per segment; only the composite $`Y_\sigma=M_3M_2M_1b'`$ is enforced, Section 8):
```math
y^{(1)}_t(\rho_1,\beta)=\bigoplus_\zeta M_1[\rho_1,\zeta]\,b'_t(\zeta,\beta),\quad
y^{(2)}_t(\rho_2,\sigma)=\bigoplus_{\zeta}M_2[\rho_2,\zeta]\,y^{(1)}_t(\zeta,\sigma),\quad
Y_\sigma[\rho_3]=\bigoplus_{t,\rho_2}M_3[\rho_3,(t,\rho_2)]\,y^{(2)}_t(\rho_2,\sigma).
\qquad(9.1)
```
$`\mathbb F_p`$ side, over the integers (Section 5 for $`y^{(1)}`$, committed $`y^{(2)},\kappa^{(\cdot)}`$):
```math
\sum_\zeta M_1[\rho_1,\zeta]\,b_t(\zeta,\beta)=y^{(1)}_t+2\kappa^{(1)}_t,\quad
\sum_\zeta M_2[\rho_2,\zeta]\,y^{(1)}_t=y^{(2)}_t+2\kappa^{(2)}_t,\quad
\sum_{t,\rho_2}M_3[\rho_3,(t,\rho_2)]\,y^{(2)}_t=Y_\sigma[\rho_3]+2\kappa^{(3)},
\qquad(9.2)
```
with $`y^{(1)}_t,y^{(2)}_t,b_t\in\{0,1\}`$ enforced by (7.1), `bool_y2`, `bool_b`, and every carry in $`[0,2^{16})`$ by the lookup into $`T_{16}`$ (Section 4.3). Each $`\mathbb F_p`$ equation then has both sides below $`2^{18}`$ (left at most 2,047 or 3,066 by the row weights, right at most $`1+2^{17}`$), so it holds over $`\mathbb Z`$ and the Boolean left-hand quantity is the GF(2) parity (any carry range below $`(p-W_t)/2`$ would do, `docs/bridge.md` Section 4). Without the carry ranges the identities would be vacuous: 2 is invertible modulo $`p`$, so an unbounded carry fits either parity. Hence the $`\mathbb F_p`$ side proves $`Y_\sigma=M_3M_2M_1\,b`$ over GF(2) per segment and the Flock side the same for $`b'`$. The row weight of $`M_1,M_2`$ is at most 2,047 and of $`M_3`$ at most 511 with probability above $`1-2^{-400}`$; the honest prover MUST abort if a row weight exceeds 2,047 (levels 1, 2) or 511 (level 3), the carry ranges of Section 4.3 (completeness only).

### 9.3 Stage placement and costs

Levels 2 and 3 and all Booleanity checks are in stage 1, the level-1 definition in stage 2, the carry range checks in the lookup (event 7), and the evaluation reductions in stage 3 (Section 7). Costs at $`N=2^{15}`$: committed carries $`3{,}145{,}728+196{,}608+16{,}384`$ (one column per carry) and parities $`196{,}608`$ (0.9% of the committed cells); GF(2) nonzeros $`\approx2.9\times10^9`$ (also the prover's bit operations for parities and carries); final vectors $`512S=2{,}048`$ bytes; verifier Section 7.6.

### 9.4 Soundness

Let $`\delta=b\oplus b'\ne0`$ be the difference of the two committed copies, fixed before $`\mathsf{sd}`$. By `ref/notes/improvements-bridge.md` Lemmas 1 to 3 and the composition theorem, $`\Pr[M_3M_2M_1\delta=0]\le\sum_t1/\binom{m_t}{k_t}=2^{-145.6}+2^{-145.6}+2^{-163.4}=2^{-144.6}`$ (segments and lanes are blocks; padded blocks are blocks). If the product is nonzero, the two sides cannot agree on $`Y_\sigma`$, so
```math
\varepsilon_{\rm bridge}\le2^{-144.6}+\varepsilon_{\rm PIOP}+\varepsilon_{\rm Flock}+\varepsilon_{\rm bind}.
\qquad(9.3)
```
Grinding on $`\mathsf{sd}`$ multiplies the first term by the number of seed derivations; the 16.6-bit margin above $`2^{-128}`$ covers it, and the seed site MAY be ground under Aerie's schedule.

## 10. Transcript and wire format

Backend: Spongefish `Blake2b512` as in Aerie 10.1, tag `agg-mldsa/v1/piop/blake2b` (64 bytes, zero-padded), session `B("agg-mldsa/v1")`, instance digest from the canonical instance record. Encodings and the per-limb $`K`$ sampler are Aerie's. Scheduled grinding (wire version 2, `docs/soundness.md` Section 6): every field-challenge event of difficulty $`g>0`$ (12.1) absorbs a nonce $`v\lt2^{g+7}`$ as the public message `LE32(v)`, squeezes a 32-byte predicate whose low $`g`$ bits must be zero (Akita's bit order), then squeezes the challenge; the PIOP nonces travel in one packed stream (below), the tie nonces in the tied-part proofs, Akita's inline in the opening (Akita's own plan, bound in its instance descriptor). Ordered events:

| # | Event | Payload |
| --- | ---: | --- |
| 0 | instance | wire version LE16 (4), parameter-set byte 0x44, relation byte (0 = H-u, 1 = H-exact; 2 reserved for S1+S2), $`N`$, $`N^\star`$ LE64, public-record digest (Section 2), digit layout id (1, Section 4.3), the six group descriptors (with their widths) and the grouped catalog row id, Flock profile id, bridge parameters $`(J_t,m_t,k_t)`$, the lookup layout (challenges, table width, stacks); then (wire version 2) the 32-byte digest of the proof-of-work plan (`docs/soundness.md` 6.4) |
| 1 | absorb | round-1 Akita commitments $`G_{\rm main16}`$, $`G_{\rm main4}`$ (native encoding) |
| 2 | absorb | $`S`$ Flock commitments (Ligerito roots or LaBinius commitments), segment order |
| 3 | challenge | $`\mathsf{sd}`$: 32 bytes, no field sampling |
| 4 | absorb | round-2 Akita commitments $`G_1,G_2,G_3,G_m`$ in group order |
| 5 | challenge | $`\mathsf{fs}_\sigma`$: 32 bytes per segment, the challenger seed of the Flock sub-proof $`\sigma`$ (its challenger is initialized with $`\mathsf{fs}_\sigma`$) |
| 6 | absorb | $`Y_0,\ldots,Y_{S-1}`$ (512 bytes each), then the $`S`$ share items of the Keccak side, each framed by `B()` (mode A: the shares of a segment's parts but the last; empty for a span, a one-part segment and mode B) |
| 7 | challenge/absorb | the lookup (7.0): $`X_1,X_2\in K`$; the layer-1 values (4 per tree); $`\mu_1`$; per session $`s=2,\ldots,D`$ ($`D=\max(d_{\rm main}+2,16)`$, 7.0): $`\gamma`$ (2 per active tree), $`s-1`$ rounds of 3 elements and their challenges, the terminals (4 per continuing tree), the leaves (2 per ending stack or table), $`\mu_s`$; then the 7 claims $`m_k(\pi_m)`$ |
| 8 | challenge | $`\tau\in K^{d_{\rm main}}`$ |
| 9 | absorb | 13 pre-stage claims (7.2) |
| 10 | challenge | $`\gamma^{(1)}\in K^{26}`$; then the stage-1 rounds of 4 elements each and the 18 stage-1 claims |
| 11 | absorb/challenge | 6 claims $`K^{(1)}_t(r^{(1)}_1)`$; $`\gamma^{(2)}\in K^{8}`$; stage-2 rounds of 4 elements; 8 stage-2 claims |
| 12 | challenge | $`\gamma^{(3)}\in K^{44}`$; stage-3 rounds of 3 elements; the 31 stage-3 opening claims |
| join | absorb | the $`S`$ Flock proof byte strings, each framed by `B()` (deviation K22); their challengers are seeded from $`\mathsf{fs}_\sigma`$ and $`\mathsf{rec}`$, the cSHAKE256 digest of the event-6 record (customization `agg-mldsa/v1/flock/record`) |
| 13 | PCS | Akita native stream seeded by the 32-byte PIOP session digest, as Aerie 12.3 |

Every challenge is drawn after the previous absorb. The bridge seed follows the round-1 and Flock commitments (events 1, 2), so $`b`$ and $`b'`$ are fixed before $`\mathsf{sd}`$, and precedes the round-2 commitments (event 4), whose carries, parities and multiplicities are functions of $`\mathsf{sd}`$, and every sum-check challenge. The lookup challenges $`X_1,X_2`$ follow every commitment, so every looked-up cell and $`m`$ are fixed before them. No PIOP challenge depends on the Flock proofs (mode A has no PIOP check of a Flock value), so the prover computes them concurrently with events 7 to 12 and the join absorbs them before the session digest, which therefore binds them (deviation K22; the soundness argument places the Flock stages in the segment of $`X`$, `docs/soundness.md` 12.2). Proof layout: header (version 3, flags = the relation byte, which the verifier MUST check against the relation it is asked to verify, $`N`$), the $`32N`$ bytes of $`\tilde c_i`$, then events 1, 2, 4, 6, 7, 9 to 12 and the join payloads in order, the packed PIOP nonce stream (nonce $`i`$ in $`g_i+7`$ bits, little-endian bit order, exactly $`\lceil\sum_i(g_i+7)/8\rceil`$ bytes with zero padding: 556 B at $`2^{15}`$), then the Akita opening. Size at $`N=2^{15}`$: $`32N`$ bytes of hashes; lookup 1,799 elements plus 7 digit claims, 28,896 B (`docs/lookup.md` Section 1); stages $`(4+4+3)\cdot25\cdot16+76\cdot16=5{,}616`$ B (relations H-u and H-exact: 8 + 31 and 7 + 32 stage-2 and stage-3 claims; relaxed H: 6,016 B); Akita opening about 65 KB (planner estimate 65,282 B); Flock sub-proofs $`4\times0.5`$ MB (dominant until wrapped or replaced by LaBinius).

## 11. Verifier algorithm and cost

1. Parse the header; reject an unknown version, $`N\gt2^{15}`$, or a group descriptor, digit layout or catalog row other than those of Section 4. Compute $`N^\star`$, $`S`$, the domains.
2. Per live record: `pkDecode`, $`\mathsf{tr}_i,\mu_i,c_i,T_i`$ (Section 2); keep $`\mu_i`$ and $`c_i`$. Bind the public records.
3. Replay events 0 to 6: absorb the round-1 and Flock commitments, derive $`\mathsf{sd}`$ and $`M_1,M_2,M_3`$, absorb the round-2 commitments, derive $`\mathsf{fs}_\sigma`$, absorb $`Y_\sigma`$ and the share items, compute $`\mathsf{rec}`$; verify each Flock sub-proof (its bytes are absorbed later, at the join) against its public inputs ($`\mu_i`$, $`\tilde c_i`$, $`Y_\sigma`$, the shares) and its seed from $`\mathsf{fs}_\sigma`$ and $`\mathsf{rec}`$. The order of the verifier's checks does not affect the verdict.
4. Verify the lookup (7.0): sample $`X_1,X_2`$, check both root identities and every session, check $`\sum_k16^km_k(\pi_m)=m(\pi_m)`$; keep the stack claims for (7.7). Sample $`\tau`$; compute $`Y(\tau_3)`$; absorb the pre-stage claims; compute $`c_2,c_3`$ for $`(\tau_2,\tau_3)`$.
5. Verify stage 1; at the terminal evaluate $`w_2(r^{(1)}_1)`$, $`w_{3,t}(r^{(1)}_2)`$ and the equality factors. Compute $`c_1`$ for $`r^{(1)}_1`$.
6. Verify stage 2; at the terminal evaluate $`w_1(r^{(2)})`$, $`Q_b`$ on the claimed $`P_Z`$ values and the equality factors. Stream $`\hat A_i`$ (expand or load from cache), inverse-NTT, and accumulate $`T(r^{(2)})`$ and $`w_A(r^{(3)})`$ in one pass over $`A_i`$ by absorbing the stage-3 rounds first and deferring the terminal checks.
7. Verify stage 3: form $`z,u,v,w_1'`$ and the carries at $`r^{(3)}`$ by (4.1), check the batched terminal of (7.6), the links, the reductions and the stack reductions (7.7). Absorb the Flock proof items (the join). Verify the Akita opening of the 38 claims at the six group points. Accept only if every check passes.

Cost at $`N=2^{15}`$, single core (misc memo 2.4, estimates): Keccak-f $`92N`$ (0.8 s) uncached, $`2N`$ cached; inverse NTTs $`16N`$ (0.5 s) uncached; $`T_i`$ (0.1 s); $`T(r^{(2)})`$ and $`w_A(r^{(3)})`$ $`5N\cdot1024`$ multiply-adds (0.2 s); bridge weights $`\approx1.6\times10^7`$ $`K`$-additions; $`S`$ Flock verifications (40 ms); PIOP rounds and the Akita opening (tens of ms). Totals about 1.7 s uncached and 0.4 s with cached $`A_i,\mathsf{tr}_i`$; both are reported.

## 12. Security

### 12.1 Knowledge soundness ledger

Normative analysis: `docs/soundness.md` (site census, round-by-round accounting, the
proof-of-work schedule and the theorem); implemented in wire version 2 and kept in versions 3 and 4 (`agg_piop::pow`,
`agg_piop::transcript`; tied-part proof version 5 in `agg_keccak::tied`, deviation K21: the
bridge union replaced by the virtual parity chain of `docs/parity-chain.md`, wire version 4). $`|K|=p^2`$,
$`p=2^{64}-59`$. $`L`$ is the number of looked-up cells (padding slots included),
$`L=4\cdot2^{d_{\rm main}}+8\cdot2^{d_1}+8\cdot2^{d_2}+2^{d_3}`$ (138,690,560 at $`N=2^{15}`$,
4,395,008 at $`N=1{,}024`$), $`M=2^{16}`$, $`D=\max(d_{\rm main}+2,16)`$ the deepest fraction
tree (7.0; 27 at $`N=2^{15}`$; 16 at $`N^\star=8`$).

Every challenge site $`t`$ has a round-by-round numerator $`\nu_t`$ (its bad set has mass at
most $`\nu_t/|F_t|`$ given the first false pending claim, Aerie Lemma C.1), an assigned bound
$`e_t\ge\nu_t/|F_t|`$ (Aerie C.4: a vector of $`l`$ coefficients $`l/|F|`$, a round its degree,
a merge $`1/|F|`$), and the difficulty $`g_t=\min\{g\ge0:2^{128}e_t\le2^g\}`$ computed with the
exact field size ($`g=\mathrm{bit\_length}(n)`$ for $`e_t=n/|K|`$, $`\lceil\log_2n\rceil`$ for
$`n/2^{128}`$). A site with $`g_t>0`$ is preceded by a work nonce of $`g_t+7`$ bits; every
address then weighs at most $`2^{-g_t}\nu_t/|F_t|\le2^{-128}`$ (Akita Proposition 10.3).

| Site class | Count at $`2^{15}`$ | $`\nu_t`$ | Assigned $`e_t`$ | $`g_t`$ | Nonce bits |
| --- | ---: | --- | --- | ---: | ---: |
| lookup $`X=(X_1,X_2)`$, one vector site (Lemma below) | 1 | joint | $`((L+M-1)/|K|)^2=2^{-201.9}`$ | 0 | 0 |
| lookup merges $`\mu_1..\mu_D`$ | 27 | 1 | $`1/|K|`$ | 1 | 8 |
| GKR batching vectors $`\gamma_s`$ ($`2\cdot`$active trees coefficients) | 26 | 1 | $`20,16,12,8,4`$ over $`|K|`$ | 5, 5, 4, 4, 3 | 12, 12, 11, 11, 10 |
| GKR rounds (degree 3) | 351 | 3 | $`3/|K|`$ | 2 | 9 |
| $`\tau\in K^{d_{\rm main}}`$ (all 19 zero-checks and the linear identities at $`\tau`$) | 1 | $`d_{\rm main}`$ | $`d_{\rm main}/|K|`$ | 5 | 12 |
| stage batching $`\gamma^{(1)},\gamma^{(2)},\gamma^{(3)}`$ (26, 8, 44 coefficients; H-exact 26, 7, 45) | 3 | 1 | $`26,8,44`$ over $`|K|`$ (H-exact $`26,7,45`$) | 5, 4, 6 (H-exact 5, 3, 6) | 12, 11, 13 (H-exact 12, 10, 13) |
| stage rounds, degrees 4, 4, 3 (the leaves, links, ring identity, 37 reductions and 4 stack reductions ride on them) | $`3\cdot25`$ | 4, 4, 3 | degree over $`|K|`$ | 3, 3, 2 | 10, 10, 9 |
| bridge seed $`\mathsf{sd}`$ (9.4) over the sponge commitments' Johnson lists: $`\sum_\sigma L_s^{P_\sigma}\theta`$, $`P_\sigma\le2`$ parts per segment | 1 | | span on $`L_s\theta=2^{-139.0}`$, span off $`4L_s\theta=2^{-137.0}`$, worst admitted $`4L_s^2\theta=2^{-131.4}`$ (`Slim`) | 0 | 0 |
| tied part's tie site (public claim), GF($`2^{128}`$), with the sponge list size $`L_s`$ (35, 50, 70) | 1 (span) | $`L_s(\nu_s+3)`$ | $`L_s(\nu_s+3)/2^{128}`$ | 10 / 10 / 11 (`Fast`/`Slim`/`Slim8`, span of four) | $`g+7`$ |
| tied part's $`\mathrm{eq}_Y`$ site (parity chain, K21) | 1 (span) | $`L_s(12+s)`$ | $`L_s(12+s)/2^{128}`$ | 9 / 10 / 10 | $`g+7`$ |
| tied part's chain rounds (degree 2) | $`13+d+s`$ (28) | $`2L_s`$ | $`2L_s/2^{128}`$ | 7 / 7 / 8 | $`g+7`$ |
| Akita opening sites | 459 | Akita's loss factors | Akita Eq. (146) over $`p^2`$ | 1 to 12 | inline LEB128 (already in the opening) |
| Flock internal sites | | Flock's | Flock's per-site schedule; outer sites with the level-0 list factor (PATCH.md 5) | | 8-byte words (already in the Flock proofs) |

At $`N^\star=2^{15}`$ the PIOP plan has 484 sites and 483 nonces, 4,444 bits in one packed
stream of 556 bytes (227 bytes at $`N^\star=8`$, 391 at $`2^{10}`$); the expected honest work is
$`\sum2^{g_t}=2{,}718`$ predicate evaluations. Before work the round-by-round sum of the PIOP
field terms is $`(11d_{\rm main}+d_{\rm main}+3+{\rm GKR})/|K|=1{,}409/|K|=2^{-117.5}`$ at
$`2^{15}`$ (979/|K| at $`2^{10}`$; relaxed H with stage-2 degree 5: 1,434 and 999); the former per-instance ledger charged every reduction,
zero-check and linear identity its own $`d_{\rm main}/|K|`$ and summed to $`3{,}509/|K|`$
(`docs/soundness.md` Section 5). The plan (the tied parts' sites, then the PIOP sites in registry order)
is bound in event 0 by its SHA3-256 digest (Section 10). Mode A admits at most two tied parts
per segment ($`d_{\rm part}\ge d_{\rm seg}-1`$; `docs/soundness.md` 3.5), so the $`\mathsf{sd}`$
union stays below $`2^{-128}`$. The composition of Akita's extractor with the Flock
commitments is proved in `docs/soundness.md` Section 12 (no second extractor: checkpoints at
$`\mathsf{sd}`$ and $`X_1,X_2`$, the Ligerito words fixed straight-line by the Merkle expansion
and used only through their Johnson lists; since the fork-join of deviation K22 the Flock
stages lie in the segment of $`X_1,X_2`$ and the test there lists the Johnson list of each
sponge word), under two stated assumptions: Flock's documented
round-by-round analysis of its instances and the correctness of the Keccak and bridge
circuits (Section 12.5 there). Not covered by the schedule: the $`Q=0`$ term
$`(R_f+R_{\rm coord}+R_F)2^{-128}`$ (about $`2^{-117}`$, the claim is 128 bits per query as in Aerie).

**Lemma (lookup identity).** Let the committed looked-up values be $`f_1,\ldots,f_L\in\mathbb F_p`$ ($`f=S+2^{15}`$ for a stored cell $`S`$; the padding slots are public, $`f=0`$) and $`m\in\mathbb F_p^{T_{16}}`$, all fixed by the commitments before $`X_1,X_2`$. If some $`f_i\notin T_{16}`$, then $`R(X)=\sum_i1/(X-f_i)-\sum_{t\in T_{16}}m(t)/(X-t)\ne0`$: the coefficient of $`1/(X-f_i)`$ is a count in $`[1,L]`$, nonzero in $`\mathbb F_p`$ since $`p\gt L`$ (LogUp Lemma 5, any $`m`$). Its numerator $`N`$ over $`\prod_{a\in S}(X-a)`$, $`S=\{f_i\}\cup T_{16}`$, is nonzero of degree at most $`L+M-1`$, and a challenge passes the root check (all root denominators nonzero, so $`X\notin S`$) only if $`N(X)=0`$. For independent uniform $`X_1,X_2`$, $`\Pr[N(X_1)=N(X_2)=0]\le((L+M-1)/|K|)^2`$. Otherwise some tree set's true root violates its identity, the prover's layer-1 values for it are false, and the batched GKR sessions accept a false claim only with the GKR term (Proposition 1 of LogUp-GKR with independent coefficients per claim; the second tree set adds claims, not sessions). With one challenge the identity term would be $`2^{-101.0}`$ at $`N=2^{15}`$ ($`2^{-105.9}`$ at 1,024), which no remedy cheaper than the second challenge brings to $`2^{-128}`$ (`docs/lookup.md` Section 2.1: a 28-bit work nonce costs about 7 times the second challenge and exceeds Akita's 25-bit cap; an $`\mathbb F_{p^4}`$ challenge triples the tree arithmetic).

With the schedule every field site above weighs at most $`2^{-128}`$ per address (the
identity term $`2^{-201.9}`$ needs no work; the lookup's GKR rounds, each $`3/|K|\approx2^{-126.4}`$
before work, get 2 bits). Point reuse ($`\tau`$ for all zero-checks, $`r^{(1)}`$ for the leaves, $`r^{(2)}`$ for the links and the ring identity, the stack points $`\pi_j`$ for their reductions) is sound because every tested table is committed before the point is sampled (Aerie 9.6).

### 12.2 Extraction

From an accepting transcript the Akita extractor yields the 38 committed tables (39 under H-exact, below) as $`\mathbb F_p`$ values with no range guarantee (Akita Remark 10.41). The PIOP ledger then gives, except with the terms above: every looked-up cell is an integer entry of $`T_{16}`$, i.e. $`Z_{\rm lo},U_{\rm lo},V_{\rm lo},V_{\rm hi},K^{(1)}_t,K^{(2)}_t,K^{(3)}\in[-2^{15},2^{15})`$ (the lookup), and $`Z_{\rm hi}\in[-2,1]`$, $`U_{\rm hi}\in[-3,2]`$ (leaves, links); hence by (4.1) and (4.2) $`z\in[-2^{17},2^{17})`$, $`u\in[-190{,}465,190{,}465]`$ (the union of the six blocks is exactly this interval, far below $`p/2`$, so the field value is the integer), $`v\in[-2^{31},2^{31})`$ and every carry in $`[0,2^{16})`$, which is (3.2) with $`(\bar B_z,\bar B_u)=(2^{17},B_u)`$ (relation H-u); $`b_t,y^{(2)}_t\in\{0,1\}`$ and (3.4) on every live position; (9.2) over $`\mathbb Z`$; $`\Delta=0`$ in $`\mathbb F_p`$, hence (3.1) over $`\mathbb Z`$ by (6.1). The multiplicities are unconstrained and need not be. The Keccak side needs no extractor of its own (`docs/soundness.md` Section 12): (3.3) is a computation on the extracted bits $`b`$, and the round-by-round analysis shows that if it fails for a live record, then except with the seed term every codeword in the Johnson lists of the Ligerito commitments fails the Flock statements or the bridge, so an accepting proof hits a bad set (the Flock words with $`\tilde c_i=\mathrm{SHAKE256}(\mu_i\Vert b'_i)`$ and $`Y_\sigma=M_3M_2M_1b'`$ would give $`b=b'`$ except with (9.3)). Hence every live record satisfies $`\mathrm H(\bar B_z,\bar B_u)`$. Under H-exact the extractor yields 39 tables; the lookup gives $`S_{z,\rm lo}\in[-2^{15},2^{15})`$, the Booleanity zero-checks $`B_{z,\rm hi},a_z\in\{0,1\}`$ (a field element with $`b^2=b`$ is 0 or 1), so by (4.3) $`z\in[-130{,}993,130{,}993]`$ (far below $`p/2`$), and every live record satisfies $`\mathrm H(B_z,B_u)`$; the lifting bound (6.1) holds a fortiori. Domain separation between the Keccak argument's SHAKE256 and the public-record binding rests on the distinct cSHAKE customizations and on $`\mu_i`$ being a hash output (random-oracle modeling as in Aerie 2.3).

### 12.3 Aggregate unforgeability

Game (chosen-key model, as Aerie Definition 5.10 and BGLS03): the challenger generates one honest key pair $`(\mathsf{pk}^\star,\mathsf{sk}^\star)`$ (FIPS 204 Algorithm 1) and gives the adversary $`\mathsf{pk}^\star`$ and a signing oracle (Algorithm 2, pure mode, empty context; hedged by default, line 5); the adversary wins with a verifying aggregate whose record list contains a record $`i^\star`$ with $`\mathsf{pk}_{i^\star}=\mathsf{pk}^\star`$ and $`M_{i^\star}`$ never queried in the game. The other keys may be chosen adversarially, without proofs of possession. Freshness is on $`(\mathsf{pk},M)`$; $`\tilde c_i`$ is aggregator-chosen and need not match any signature the signer produced. Several honest keys reduce to one at a factor of their number (guess the key of $`i^\star`$). The argument has two steps in two models. It is not a single random-oracle reduction: relation (3.3) is proven for the concrete SHAKE256 computed inside the Keccak circuit, so a programmed H would disagree with the circuit (simulated signatures could not be aggregated with a valid proof, and the adversary's view would differ from the real game), and an in-circuit evaluation is not a random-oracle query. Steps 1' and 2' below replace steps 1 and 2 of revision 6 (a UF-CMA reduction with an open obligation for signing queries asked only in rewound runs); the normative proof is `docs/soundness.md` 12.8.

**Relaxed scheme.** For the caller's relation (H-u or H-exact) let $`\Sigma_{\rm H}`$ be: key generation as ML-DSA-44; signing as ML-DSA-44, returning the FIPS 204 signature $`\sigma=(\tilde c,z,h)`$ together with its relaxed image $`\phi(\sigma)=(\tilde c,z,w_1',u,v)`$ of Section 3.2; a relaxed signature $`(\tilde c,z,w_1',u,v)`$ on $`(\mathsf{pk},M)`$ is valid iff it satisfies (3.1) to (3.3) with the relation's proven bounds (12.2). There is one relaxed scheme per relation, and $`\phi(\sigma)`$ is valid for both (Section 3.2). SUF-CMA for $`\Sigma_{\rm H}`$: the adversary wins with a valid $`(M^\star,\sigma^\star)`$ such that $`\sigma^\star\ne\phi(\sigma_j)`$ for every answer $`\sigma_j`$ to a query $`M_j=M^\star`$. Two facts are used throughout:

- (U) *Uniqueness of $`(u,v)`$.* For a valid relaxed signature, (3.1) gives $`u\equiv\sum_sA[\cdot,s]\circledast z[s]-T-2\gamma_2w_1'\pmod q`$ and $`\lvert u\rvert_\infty\le B_u\lt(q-1)/2`$, so $`u`$ is the centered residue and $`v`$ the exact quotient. Two valid relaxed signatures on the same $`(\mathsf{pk},M)`$ with the same $`(\tilde c,z,w_1')`$ are equal.
- (E) *No encoding.* A relaxed signature is a tuple of integer vectors, bits and the 32 bytes $`\tilde c`$, and equality is equality of these values. The hint-encoding malleability through which draft ML-DSA lost strong unforgeability (FIPS 204 Appendix D.2, reference [34]; Appendix D.3 restored the missing malformed-input check in Algorithm 21) has no counterpart in $`\Sigma_{\rm H}`$.

1'. **Aggregate forgery to SUF-CMA of $`\Sigma_{\rm H}`$ (SHAKE256 concrete; lazy forwarding).** $`\mathcal B`$ gets $`\mathsf{pk}^\star`$ from its SUF-CMA challenger and runs the extractor of `docs/soundness.md` Section 7 on $`\mathcal A`$ followed by the aggregate verifier. It answers signing queries from a table $`T[(M,k)]`$ that is never rolled back: on the $`k`$-th query on $`M`$ in the current run ($`k`$ counted from the start of the run, so the counters are prover state, restored on every rewind), it returns the stored signature, or forwards $`M`$ to its challenger, stores the answer and returns it. The table is a lazily sampled fixed tape: independent signatures, one per $`(M,k)`$ (the deterministic signer gives one value for all $`k`$), whose law does not depend on the proof oracles, so $`\mathcal A`$ with this tape is a fixed prover with auxiliary input (Akita 10.3, the paragraph on auxiliary environments), the shape of Aerie Lemma C.5 (i) and (iii); and in each run $`\mathcal A`$ sees the signing oracle of the real game. The extractor rewinds and reprograms only the Fiat-Shamir and sub-protocol oracles of `docs/soundness.md` 12.1; the bridge expansion, the setup and the signer's SHAKE256 and SHAKE128 calls stay fixed. Except with $`\varepsilon_{\rm ext}`$ it returns the original execution and the witness of 12.2 for every live record; $`\mathcal B`$ outputs $`(M_{i^\star},\sigma^\star)`$ with $`\sigma^\star=(\tilde c_{i^\star},z_{i^\star},w'_{1,i^\star},u_{i^\star},v_{i^\star})`$ ($`w_1'`$ read from $`b`$). It fails only if $`\sigma^\star=\phi(\sigma)`$ for a stored answer $`\sigma=T[(M_{i^\star},k)]`$. By freshness the original run asks no query on $`M_{i^\star}`$, so every such entry is a *rerun-only* query; let $`R`$ be their number over the whole extraction, $`\mathbb E[R]\le Q_S\,(Q_{\rm wrap}+2)\,4\,\mathrm{Tree0}_{\rm ck}(\mathrm{sch})`$ (the run count of `docs/soundness.md` 12.6 times the per-run query bound $`Q_S`$). The collision event is bounded in two steps.
   - *$`\sigma^\star`$ is fixed before the first fork.* The extractor completes the original execution before its first rerun, and every rerun departs from it at a challenge at $`\mathsf{sd}`$ or later (the checkpoints $`\mathsf{sd}`$ and $`X_1,X_2`$ and Akita's opening suffix, `docs/soundness.md` 12.2), whose input encodes event 0 (the public records, with $`\tilde c_{i^\star}`$) and event 1 (the round-1 commitments to $`Z_{\rm lo},Z_{\rm hi},U_{\rm lo},U_{\rm hi},V_{\rm lo},V_{\rm hi},b_0..b_5`$, Section 4.1). So $`\tilde c_{i^\star}`$ is a public input fixed before any rerun-only answer is sampled, and the rest of $`\sigma^\star`$ is fixed by the round-1 commitments up to Akita binding (inside $`\varepsilon_{\rm ext}`$). The bound below uses only $`\tilde c_{i^\star}`$: $`\sigma^\star=\phi(\sigma)`$ requires equal $`\tilde c`$.
   - *Hedged signer.* Each rerun-only answer on $`M_{i^\star}`$ uses a fresh $`\mathit{rnd}`$ (Algorithm 2 line 5) independent of $`\tilde c_{i^\star}`$, so it has commitment hash $`\tilde c_{i^\star}`$ with probability at most $`2^{-H_\infty}`$, where $`2^{-H_\infty}=\max_{\mathsf{sk},M,x}\Pr_{\mathit{rnd}}[\tilde c(\mathrm{Sign}(\mathsf{sk},M;\mathit{rnd}))=x]`$. The decision to ask the $`j`$-th such query precedes its $`\mathit{rnd}`$, so the union over them is at most $`\mathbb E[R]\,2^{-H_\infty}`$.
   - *Deterministic signer* ($`\mathit{rnd}=\{0\}^{32}`$). All answers on $`M_{i^\star}`$ are one signature, a function of $`\rho''=\mathrm H(K\Vert\{0\}^{32}\Vert\mu_{i^\star},64)`$ (Algorithm 7 line 7), $`K`$ the 32-byte secret seed (Algorithm 6 line 1, used only there). Assume $`x\mapsto\mathrm{SHAKE256}(K\Vert\{0\}^{32}\Vert x,64)`$ is a PRF under $`K`$. With a random function in its place, the original run sees it only at the $`\mu_j`$ of queried messages, and $`\mu_j=\mu_{i^\star}`$ with $`M_j\ne M_{i^\star}`$ is a SHAKE256 collision at Algorithm 7 line 6; so $`\rho''(\mu_{i^\star})`$ is uniform and independent of $`\tilde c_{i^\star}`$, and the event costs $`\mathrm{Adv}^{\rm PRF}+\mathrm{Adv}^{\rm coll}_{\mu}+\Pr[R\ge1]\,2^{-H'_\infty}`$, with $`H'_\infty`$ the same min-entropy over a uniform 64-byte $`\rho''`$.

   Result: $`\mathrm{Adv}^{\rm agg}(\mathcal A)\le\mathrm{Adv}^{\rm SUF\text{-}CMA}_{\Sigma_{\rm H}}(\mathcal B)+\varepsilon_{\rm ext}+\mathbb E[R]\,2^{-H_\infty}`$, plus $`\mathrm{Adv}^{\rm PRF}+\mathrm{Adv}^{\rm coll}_\mu`$ for the deterministic signer. No presampling, no programmed H, no $`q_H`$ or $`e(Q_S+1)`$ factor. $`\mathcal B`$'s running time and signing queries are those of the extractor (its run count times $`\mathcal A`$'s), the ordinary concrete-security accounting that the Module-SIS terms of $`\varepsilon_{\rm ext}`$ already follow. Evaluation (estimate): $`H_\infty\le256`$, since $`\mathit{rnd}`$ and $`\tilde c`$ have 32 bytes. In the ROM for the signer's hash calls (Algorithm 7 lines 7, 11, 15) an answer's $`\tilde c`$ is a fresh oracle output unless a hash query of $`\mathcal A`$ starts with $`K`$ (at most about $`Q_H2^{-256}`$ over the whole game) or an iteration's $`w_1`$ equals the $`w_1`$ of an earlier hash query on $`\mu_{i^\star}`$ (the commitment min-entropy term of the FSwA analyses); otherwise each of the expected 4.25 iterations (FIPS 204 Table 1, "Repetitions") hits $`\tilde c_{i^\star}`$ with probability $`2^{-256}`$, about $`2^{-253.9}`$ per answer. The term grows with the reduction's work like $`\mathrm{Adv}^{\rm SUF\text{-}CMA}_{\Sigma_{\rm H}}(\mathcal B)`$ does and stays far below $`2^{-128}`$ per signature obtained.

2'. **SUF-CMA of $`\Sigma_{\rm H}`$ (ROM for H).** Model H as a random oracle, under the usual heuristic of instantiating it with SHAKE256 (outside the aggregate proof, step 1' needs no model for H). Let $`(M^\star,\sigma^\star)`$ win. One of three cases holds:
   - (i) *$`\tilde c^\star`$ differs from the $`\tilde c`$ of every answer on $`M^\star`$* (in particular when $`M^\star`$ was never queried). The hash input $`\mu^\star\Vert\mathrm{w1Encode}(w_1'^\star)`$ is then not the input of any returned answer (an answer on $`M_j\ne M^\star`$ has $`\mu_j\ne\mu^\star`$ unless SHAKE256 collides at Algorithm 7 line 6). The Dilithium/FIPS 204 argument applies with the relaxed verification: replace $`t`$ by uniform (MLWE), simulate signatures by programming H on $`(\mu\Vert\mathrm{w1Encode})`$ (the CMA-to-NMA step as corrected by Barbosa et al., CRYPTO 2023, with the strong-unforgeability bookkeeping of Devevey et al., CRYPTO 2023: the forgery's input may equal that of a rejected, never revealed iteration only with the commitment min-entropy term), and output $`(z^\star,-c^\star,c^\star t_0-u^\star)`$, a SelfTargetMSIS solution with bound $`\zeta'`$ (3.5) as in Section 3.3. Losses: one MLWE instance and the simulation terms of those analyses.
   - (ii) *$`\tilde c^\star=\tilde c_j`$ for an answer on $`M^\star`$, and $`w_1'^\star\ne w'_{1,j}`$.* Then $`\mathrm H(\mu^\star\Vert\mathrm{w1Encode}(w_1'^\star),32)=\mathrm H(\mu^\star\Vert\mathrm{w1Encode}(w'_{1,j}),32)`$ on distinct inputs (w1Encode, Algorithm 28, packs each coefficient of $`[0,44)`$ in 6 bits and is injective): a SHAKE256 collision at 256-bit output (collision strength $`\lambda=128`$ of $`\tilde c`$, FIPS 204 Table 1); in the ROM at most $`q_H^2/2^{257}`$.
   - (iii) *$`\tilde c^\star=\tilde c_j`$ and $`w_1'^\star=w'_{1,j}`$ for an answer on $`M^\star`$.* Then $`c^\star=c_j`$ and $`T^\star=T_j`$, and (3.1) for both gives $`\sum_sA[\cdot,s]\circledast\delta_z[s]-\delta_u-q\,\delta_v=0`$ over $`\mathbb Z`$ for the differences. By (U), $`\delta_z=0`$ would give $`\sigma^\star=\phi(\sigma_j)`$; so $`\delta_z\ne0`$ and $`[A\mid-I](\delta_z,\delta_u)\equiv0\pmod q`$ with $`\lvert\delta_z\rvert_\infty\le2^{17}+B_z=262{,}065`$ (H-u; $`2B_z=261{,}986`$ under H-exact) and $`\lvert\delta_u\rvert_\infty\le2B_u=380{,}930`$: a nonzero Module-SIS solution for $`[A\mid I]`$, $`1{,}024\times2{,}048`$ over $`\mathbb Z_q`$, $`\ell_\infty`$ bound 380,930, with $`A`$ uniform (ExpandA as a random oracle, as in the Dilithium analysis). The reduction generates $`(s_1,s_2)`$ itself and signs honestly; no MLWE step. This case is the computational unique-response property of the relaxed identification scheme (Devevey et al., Definition 6), case (i) its no-message part.

   Hence $`\mathrm{Adv}^{\rm SUF\text{-}CMA}_{\Sigma_{\rm H}}\le\mathrm{Adv}^{\rm MLWE}+\mathrm{Adv}^{\rm SelfTargetMSIS}_{\zeta'}+\varepsilon_{\rm sim}+\mathrm{Adv}^{\rm coll}_{\rm SHAKE256/256}+\mathrm{Adv}^{\rm MSIS}_{[A\mid I],380{,}930}`$.

Relation H-u has $`\zeta'=\zeta=350{,}209`$ (3.5), so the SelfTargetMSIS instance is ML-DSA-44's own: core-SVP $`2^{123.5}`$ classical (lattice-estimator rough model, block size 423; `ref/notes/msis-estimate.md`, which reproduces the published Dilithium2 figure), and FIPS 204 claims Category 2 for ML-DSA-44. The former open obligation (the +20.5% looser $`\zeta'=421{,}888`$ of relaxed H, core-SVP $`2^{119.7}`$) no longer applies. H-exact has the same $`\zeta'=\max(B_z,B_u+2^{d-1}\tau)=350{,}209`$ and adds that the proven $`z`$ bound is FIPS 204's acceptance bound. Case (iii) adds the SUF instance: block size 417, core-SVP $`2^{121.8}`$ classical (rough model; $`2^{149.3}`$ in the default model) and 121 classical / 110 quantum with the pq-crystals scripts, measured with the pinned tools of `ref/notes/msis-estimate.md` (2026-10-26). It is the instance of the Dilithium round-3 strong-unforgeability row (that script's $`\max(2\gamma_1,4\gamma_2+1)=380{,}929`$ gives the same block size), so unforgeability rests on $`\min(2^{123.5},2^{121.8})=2^{121.8}`$ core-SVP, the basis of ML-DSA-44's own SUF claim; MLWE stays at 124 / 123 (primal / dual).

What is assumed. (a) SHAKE256 collision resistance at 256-bit output (case ii) and, for the deterministic signer, the PRF property above and 512-bit collision resistance of $`\mu`$. (b) The CMA-to-NMA simulation step for ML-DSA as standardized. Barbosa et al. prove UF-CMA in the ROM (machine-checked) for round-3 Dilithium with public-key compression, abstract hints and $`c=\mathrm H(w_1,M)`$, and remark only that strong unforgeability follows given computational unique responses; Devevey et al. prove sUF-CMA for generic Fiat-Shamir-with-aborts signatures $`(w,z)`$ with bounded aborts from UF-NMA, computational unique responses and commitment min-entropy (full version, ePrint 2023/245, Theorems 9 and 11), not for Dilithium's $`(\tilde c,z,h)`$ format. Neither covers FIPS 204's exact algorithms (64-byte $`\mathsf{tr}`$ and $`\mu`$, a 32-byte $`\tilde c`$ fed whole to SampleInBall, the hedged $`\rho''`$) or its encoding; we assume the simulation step carries over, as FIPS 204's Category 2 claim does. FIPS 204 Section 3.1 states SUF-CMA as a design goal, not a theorem. The lemma above does not use FIPS 204's SUF property: by (E) the relaxed scheme has no encoding, and case (iii) is proved directly. (c) The signer's SHAKE256/SHAKE128 calls are not queries to the proof oracles (distinct customizations and framings, `docs/soundness.md` 12.5 (R)).

Mode C (private $`\tilde c_i`$, not specified here): step 1' needs every candidate value of $`\tilde c_{i^\star}`$ fixed before the first fork, i.e. $`\tilde c`$ MUST be committed in event 1 or 2 (before $`\mathsf{sd}`$). With a Ligerito commitment the candidates are the values in the codewords of its Johnson list (at most $`L`$, fixed by the Merkle expansion in the original run), and the collision term becomes $`L\,\mathbb E[R]\,2^{-H_\infty}`$.

## 13. Test plan

Unit tests (per crate, `cargo test --release` where heavy; batch sizes at most 1K signatures and $`2^{14}`$ permutations on the development machine):

- Parameters: every constant of Section 1 recomputed from FIPS 204 values; (3.4) by enumeration over $`[0,64)`$; the UseHint residual maximum 190,465 by enumeration over $`\mathbb Z_q\times\{0,1\}`$; $`B_v`$ and (3.4'), (6.1) with exact integer arithmetic; the digit bijections (4.1), the step form (4.2) at $`\pm190{,}465`$ (encodes) and $`\pm190{,}466`$ (no in-alphabet digits) and the exactness of $`Q_4,Q_6`$ by enumeration (`spec_checks.py` items 8 to 10); the L16 layouts (4.1) at their range boundaries (`manifest::tests`).
- Lookup (`agg-piop` `lookup`, `tests/stages.rs`): fraction-tree roots against direct sums; honest layouts at one and two challenges; the claims against the extensions of the stacked lanes, the columns and $`m`$; every out-of-table cell (above $`2^{16}`$, negative, a carry, a padding slot), a wrong multiplicity, a tampered layer-1 value, round message, terminal or leaf, swapped digit claims and a value valid only for another range rejected.
- Witness (`agg-witness`): against a FIPS 204 reference implementation on random and edge-case signatures (Decompose edge residue $`-\gamma_2`$, hints at both signs, $`\lVert z\rVert_\infty=B_z`$): native verify, derived $`(z,u,v,b)`$ and their digits satisfying (3.1) to (3.3) and (4.1) over $`\mathbb Z`$, $`(Az)`$ by CRT against schoolbook, padded rows at the canonical digits of zero.
- Identities, one positive and one negative test each: (7.1) to (7.6), `bool_y2`, `bool_b`, every leaf, link and reduction; the negative test flips one witness value (a non-Boolean bit, $`w_1'=44`$, a digit outside its alphabet, a carry of the wrong parity, $`v=q^{-1}\Delta_0`$ for a wrong $`z`$, a changed $`Y_\sigma`$ bit) and MUST fail at the stated stage.
- Weights: $`w_A`$ against a direct negacyclic reference at random $`K`$ points for every $`(\rho,s)`$; $`\Lambda`$ trie against (7.8); $`w_1,w_2,w_3`$ against dense matrix transposes at $`J`$ reduced to $`2^{10}`$; zero weights on padded positions.
- Bridge (`agg-bridge`): seed derivation determinism and distinct column streams; column weight exactly $`k_t`$; parities and carries against a GF(2) reference; (9.2) lifting with carries at the range boundary; the composite walker against $`M_3M_2M_1`$ materialized at $`J`$ reduced to $`2^{10}`$; the small-parameter error $`1/\binom{m}{k}`$ measured empirically at $`(m,k)=(64,3)`$ against weight-2 differences.
- Keccak (`agg-keccak`): sponge chain against a SHAKE256 reference for 1, 7 and 9 blocks; bit order, public $`\mu`$, fixed padding, published $`\tilde c`$; the bridge slot at $`k_{\log}`$ 16 and 20 (8 and 128 records) proves and verifies and rejects a flipped $`Y_\sigma`$ bit or a bit changed in one copy only; a sub-batch of $`2^{14}`$ permutations proves and verifies; $`k_{\log}=26`$ is measured on the cloud machine.
- PCS config (`agg-pcs-config`): the width-4 preset admits the small digits, the bits, the multiplicity digits and the padded digits, the width-16 preset certifies exactly $`[-2^{15},2^{15})`$; catalog rows exist for the six-group layouts at $`d_{\rm sig}\in[3,15]`$ in both families; a two-round commit ($`G_{\rm main16},G_{\rm main4}`$, then $`G_1,G_2,G_3,G_m`$) opens in one batch and verifies.
- Transcript: event order, a reordered or truncated proof is rejected, pinned proof digests for $`N\in\{8,256\}`$.
- Relation H-exact (`agg-piop` `tests/h_exact.rs`, `tests/mode_b.rs`, `agg-cli` `tests/mode_a.rs`): the bit-slack form (4.3) exhaustively; the event-0 bytes, claim counts, stage specifications and plan digests; honest round trips in mode B at 1, 64 and 256 and in mode A at 64 and 256; $`a_z=2`$, $`B_{z,\rm hi}\in\{2,-1\}`$ (stage 3), $`U_{\rm hi}\in\{3,-4\}`$ (stage 2), $`S_{z,\rm lo}=2^{15}`$ (lookup), non-Boolean digits that keep $`z`$ and the lookup (stage 3); synthetic statements with $`T`$ moved to keep (3.1) at $`z=\pm130{,}993`$ (accept), $`z=130{,}994`$ and $`-130{,}994`$, $`z=131{,}000`$ (lookup or stage 3 by encoding; H-u accepts them), $`u=\pm190{,}466`$ (lookup or stage 2); an H-exact proof verified as H-u, an H-u proof verified as H-exact, a rewritten or unknown relation byte rejected.

End-to-end: honest aggregates at $`N=256`$ and $`N=1{,}024`$ verify; each of the negative mutations above, applied to one record, is rejected; a wrong $`\tilde c_i`$ is rejected; cached and uncached verifier paths agree.

## 14. Milestones and crates

| Crate | Content | Milestone |
| --- | --- | --- |
| `agg-witness` | ML-DSA-44 native verify, key expansion, $`(z,u,v,b)`$ derivation, CRT product, table builders, padding | M1 |
| `agg-pcs-config` | the width-4 (Aerie's `Fp64SignedWidth4` parameters) and width-16 presets, catalogs for the six-group layouts at $`d_{\rm sig}\in[3,15]`$, two-round commit driver, work plan | M1 (long lead, start first) |
| `agg-piop` | the lookup into $`T_{16}`$, stages 1 to 3 with the two leaves and links, weight tables, verifier, Akita opening of the six groups | M2 |
| `agg-bridge` | seed to matrices, $`\mathbb F_p`$ parity and carry-digit tables, GF(2) composite walker, soundness constants | M2 |
| `agg-keccak` | Flock HEAD Keccak gate port, sponge union circuit, bridge slot type with the factored `LincheckCircuit`, segment driver | M3 |
| `agg-cli` | prove/verify binaries, wire format, record binding, caches | M2 (mode B: bits public, no Keccak), M3 (mode A) |
| `agg-bench` | harness at 1K locally and 32K on the cloud machine; prover, verifier (cached/uncached), memory | M2, M3 |

M1: witness, presets, catalogs, unit tests. M2: algebraic PIOP end to end with the bridge tables but $`b`$ published in the clear (mode B), measured at 8K. M3: Keccak argument and bridge, mode A at 256, 1K, then 32K. H-exact and S1+S2 (Appendix A) follow M3 as flags.

## Appendix A. Feature-flagged layouts

The default relation is H-u (Section 3.1). The second result, H-exact, is implemented as a runtime relation (relation byte 1; design `docs/h-exact.md`, layout X1); S1+S2 is not implemented and not specified for wire version 1 (the version written for the base-16 digit layout of wire version 0 used the `b16_prod` checks and Aerie's $`Q_{16}`$ leaf, both removed with that layout, `docs/deviations.md` L1).

**H-exact** (relation byte 1, `--relation h-exact`; normative). Relation $`\mathrm H(B_z,B_u)`$: $`\lvert z\rvert_\infty\le130{,}993`$ and $`\lvert u\rvert_\infty\le190{,}465`$, both exact, $`\zeta'=\zeta=350{,}209`$. Tables: $`u`$ as in H-u (4.2); $`z`$ by the bit-slack form (4.3) with $`S_{z,\rm lo}`$ looked up in the slot of $`Z_{\rm lo}`$ (the main stack keeps four columns, so the lookup is unchanged), the bit $`B_{z,\rm hi}`$ in the slot of $`Z_{\rm hi}`$ and the bit $`a_z`$ appended to $`G_{\rm main4}`$ after $`b_5`$ (alphabet codes $`[0,6,0,0,0,0,0,0,0]`$ in event 0). Range checks: the lookup ($`S_{z,\rm lo},U_{\rm lo}`$), `leaf6` and `link` ($`U_{\rm hi}`$), and the Booleanity zero-checks `bool[B_zhi]`, `bool[a_z]` at $`r^{(2)}`$ in stage 3. Counts: stage degrees (4, 4, 3), batching vectors (26, 7, 45), terminal claims (18, 7, 32), 39 polynomials, 32 opened at $`r^{(3)}`$; the proof-of-work plan differs only at the stage-2 batching site ($`g=3`$, `docs/soundness.md` 10.1). Cost: $`+1{,}024`$ committed elements per signature ($`a_z`$). This replaces the earlier sketch (Appendix X of `docs/h-exact.md`), which used $`u=118{,}787\,a_u+b_u-190{,}465`$ with a second Boolean column and a base-4 digit, and Booleanity in stage 1 (`docs/deviations.md` L9). The verifier takes the relation from its caller and MUST reject a proof whose relation byte differs before replaying event 0: an H-u proof is a valid proof of a weaker statement, never an H-exact proof.

**S1+S2** (`relation = fips`). The FIPS-faithful layout of `improvements-tables.md` Sections 1 and 2: per position $`z`$ (exact as above), $`r_1`$, $`v_r=29{,}696\,a_v+b_v`$ ($`b_v\in[0,2^{16})`$, looked up in $`T_{16}`$), sign $`s\in\{-1,1\}`$, $`g\in\{-1,0,1\}`$, edge $`e`$, the six bits; per signature the hint slack $`u_h=17a_h+b_h`$ ($`b_h\in[0,64)`$). Identities: $`r_0=sv_r-\gamma_2e+(1+s)/2`$, $`e^2=e`$, $`s^2=1`$, $`e(1+s)=ev_r=er_1=0`$, $`g(g-1)(g+1)=0`$, $`g(s-g)=0`$, $`\sum g^2+u_h=\omega`$, and UseHint $`x(x-44)(x+44)=0`$ with $`x=r_1+g-w_1'`$; the ring identity uses $`2\gamma_2r_1+r_0`$ in place of $`2\gamma_2w_1'+u`$. The ranges of $`r_1`$ and $`b_h`$ are narrower than 16 bits and had base-16 digits with the $`Q_{16}`$ leaf in wire version 0; under L16 each needs either small digits with Aerie's $`Q_4`$/$`Q_8`$ leaves or a second lookup column $`2^{16-k}x`$ next to $`x`$ for a range $`[0,2^k)`$ (both in $`T_{16}`$; with $`x\lt2^{16}`$ there is no wrap modulo $`p`$, so they give $`x\lt2^k`$); which one is cheaper is open. This layout re-encodes the witness to a FIPS 204 signature (hint $`h=g^2`$), giving the black-box reduction to ML-DSA EUF-CMA.

## Appendix B. Decisions taken while writing

1. **Committed quotient instead of $`\Pi_{\rm res}`$** (Section 6). The task asked to reuse Aerie's divisibility projection; its honest output would be about $`2^{68}`$ and wrap modulo $`p`$. The quotient removes the projection seed, the Rice section and one stage-1 instance; the "$`\lvert\Delta\rvert\lt p/2`$" bound survives as (6.1).
2. **Ring identity tested at one point** (7.5). With the quotient committed, (3.1) is linear in multilinear tables; the stage-3 instance (7.6) at $`r^{(2)}`$ suffices, so $`r^{(2)}`$ doubles as Aerie's $`r_\Pi`$ and the weight pass runs between stages 2 and 3.
3. **Ranges by PIOP digit proofs, not by Akita** (4.3). The first draft read Akita's certified fold alphabets (Remark 10.24) as a source range guarantee; there is none (Remark 10.41). Every range is now a digit decomposition with Aerie's leaf predicates, which also makes $`(2^{17},2^{18})`$ exact.
4. **Segments** (Section 1): one bridge chain per $`2^{13}`$-signature segment, aligned with the Keccak sub-batches, since a GF(2) sub-proof constrains only its own bits; level 2 is padded to $`2^{19}`$ positions.
5. **Carry ranges by digits** (9.2). Without a carry bound the parity identities are vacuous (2 is invertible modulo $`p`$); each carry is now three digit columns (two at level 3), since the memo's lookup is unavailable (risks R5). Revision 3: one 16-bit column per carry, looked up in $`T_{16}`$ (LogUp-GKR needs no sparse commitment, so R5 does not apply).
6. **Six lanes**: six bit polynomials rather than one table with a padded lane index (avoids 25% padding).
7. **Booleans in the width-4 profile** (4.1): one inner digit either way.
8. **Bit-range identity** as one degree-4 instance (7.4).
9. **Verifier representatives.** $`A_i`$ centered and $`T_i`$ unreduced over $`\mathbb Z`$; any fixed choice works, and these minimize (6.1).
10. **Flock challenger seeding** (event 5) after the bridge seed, binding the sub-proofs to the seeded rows (driver of Section 8).
11. **$`\tilde c_i`$ bound as part of the record digest** (Section 2), by analogy with Aerie's salt.
12. **Two Akita commitment rounds** (Section 10): round 1 before $`\mathsf{sd}`$, round 2 (the seed-dependent carries and parities) after it, under one contract (Akita 4.6).
13. **Composite GF(2) rows** (Section 8): Flock's lincheck is per block and its witness precedes the statement binding, so one bridge slot per segment proves $`M_3M_2M_1b'=Y_\sigma`$ directly.

## Appendix C. Lookup range layout L16 (adopted)

The proposal of this appendix (revision 2) is normative since revision 3: layout in Sections 4.1 and 4.3, the lookup in Section 7.0, the stages in Sections 7.3 to 7.5, events in Section 10, the ledger in Section 12.1 (two independent challenges), extraction in Section 12.2. Design, soundness, the comparison of remedies for the identity term, costs and the module: `docs/lookup.md`; departures: `docs/deviations.md` L1 to L7 and D2.

## Review responses (2026-10-24)

Findings of the first review and their resolution; none is rebutted in substance.

1. No Akita range guarantee (findings 1, 4): agreed (Akita Remark 10.41). Fixed by the digits and range checks of Sections 4.3 and 7 for $`z,u,v`$ and every carry; the old Section 4.3, the "never range-checked" rule and B.3, B.5 are withdrawn. The count is 23,865 elements per signature (9,325 was the count without range checks).
2. Bridge groups before the seed (findings 2, 5): agreed. Fixed by the two commitment rounds of Section 10 (contract: Akita 4.6); nothing seed-dependent is a Flock witness (Section 8).
3. No cap on $`v`$ (finding 3): agreed. Fixed by the $`v`$ digits (ceiling $`2^{40}`$, Section 6) and the layout rule of Section 11.
4. Cross-record rows in Flock (finding 6): agreed; Section 8 follows the reviewer's outline. $`k_{\log}=26`$ remains unmeasured (risk R3); Section 13 tests $`k_{\log}`$ 16 and 20 locally.
5. Single-digit rows for widths 17 to 27 (finding 7): agreed; no column is wider than 4 bits; H-exact is reachable (Appendix A).

Costs changed materially (Sections 4.1, 7, 10); all figures are estimates until measured.
