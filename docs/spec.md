# AggregateMLDSA protocol, milestone 1 (ML-DSA-44, mode A, relation H)

## Status

| Field | Value |
| --- | --- |
| Status | normative draft for milestone 1; no implementation yet |
| Applies to | homogeneous ML-DSA-44 aggregates, mode A (public $`\tilde c_i`$), relaxed relation H, wire version 0 (unfrozen) |
| Base system | Aerie `49b6c21` (Akita pin `e023b6a`), Flock HEAD `83f963e` circuit language, retired Keccak encoder `c2d0c24` as reference |
| Variants | H-exact and S1+S2 layouts are feature flags (Appendix A) |
| Last reviewed | 2026-10-24, revision 2 (see Review responses) |

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

The honest relation is $`\mathrm H(B_z,B_u)`$. The proven relation is $`\mathrm H(\bar B_z,\bar B_u)`$ with $`(\bar B_z,\bar B_u)=(2^{17},2^{18})`$, the ranges proven inside the PIOP by the digit decompositions of $`z`$ and $`u`$ (Section 4.3); Akita widths play no part in them. No bound on $`v`$ is part of H: (3.1) is an integer equation, and $`v`$ is determined by the other terms. The proof nevertheless range-checks $`v`$ to $`\lvert v\rvert\lt2^{26}`$, because the field identity of Section 6 lifts to (3.1) only for a bounded quotient.

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
The standard proof has $`\zeta=\max(\gamma_1-\beta,\,2\gamma_2+1+2^{d-1}\tau)=350{,}209`$. With the design target, $`\zeta'=2^{18}+159{,}744=421{,}888`$ (+20.5%). The reduction loses nothing else (Section 12.3).

## 4. Witness tables and Akita groups

### 4.1 Committed tables

Every committed table is one Akita polynomial of signed width 4 (Aerie's `Fp64SignedWidth4` preset, one inner digit); polynomials with the same domain form one group with one opening point (Section 7.5). The width sizes the honest producer and the PCS parameters only; it is not a range proof (Akita Remark 10.41, falcon.md Section 5). Every range below is proven inside the PIOP by the digits of Section 4.3; $`z,u,v,\kappa^{(\cdot)}`$ are virtual (Section 5).

| Group | Polynomials | Domain (variables) | Honest values | Meaning | Round |
| --- | --- | --- | --- | --- | ---: |
| $`G_{\rm main}`$ | $`z_0,\ldots,z_4(j,s,i)`$ | main, $`d_{\rm main}`$ | bases $`16,16,16,16,4`$ | digits of the response $`z`$ | 1 |
| | $`u_0,\ldots,u_4(j,\rho,i)`$ | main | bases $`16,16,16,16,8`$ | digits of the UseHint residual $`u`$ | 1 |
| | $`v_0,\ldots,v_6(j,\rho,i)`$ | main | bases $`16^6,8`$ | digits of the ring quotient $`v`$ (3.1) | 1 |
| | $`b_0,\ldots,b_5(j,\rho,i)`$ | main | $`\{0,1\}`$ | w1Encode bits, bit $`t`$ of $`w_1'[\rho]_j`$ | 1 |
| $`G_1`$ | $`\kappa^{(1)}_{t,0},\kappa^{(1)}_{t,1},\kappa^{(1)}_{t,2}`$, $`t\lt6`$ | $`D_1`$, $`d_1`$ | bases $`16,16,4`$ | digits of the level-1 carries | 2 |
| $`G_2`$ | $`y^{(2)}_t`$; $`\kappa^{(2)}_{t,0},\kappa^{(2)}_{t,1},\kappa^{(2)}_{t,2}`$, $`t\lt6`$ | $`D_2`$, $`d_2`$ | $`\{0,1\}`$; bases $`16,16,4`$ | level-2 parities; digits of the level-2 carries | 2 |
| $`G_3`$ | $`\kappa^{(3)}_0,\kappa^{(3)}_1`$ | $`D_3`$, $`d_3`$ | bases $`16,16`$ | digits of the level-3 carries | 2 |

A digit of base $`b`$ takes values in the balanced alphabet $`[-b/2,b/2-1]`$; bases are listed from the least significant digit. Per signature: $`1024\cdot(5+5+7+6)=23{,}552`$ main-domain elements (76 bits per position) plus, at $`N=2^{15}`$, 288 level-1, 24 level-2 and 1 level-3 elements: 23,865 committed elements (Aerie Falcon-512: 2,567), each one Akita inner digit. The former full-width layout needed about 12 inner digits per position, so the range proofs roughly double the PCS work (memory per `risks.md` R4: about 0.5 MB per signature, estimate). Round 1 is committed before the bridge seed, round 2 after it (Section 10). All four group descriptors are bound in event 0, so both rounds share one commitment contract and one batched opening: Akita Section 4.6 admits commitments formed before their batch is known under a common certified root alphabet, and Aerie's per-group `commit_group` is transcript-free except for the final group, here $`G_3`$, which commits against the precommitted profiles. Group and polynomial order on the wire and in the catalog key is the table order. Every physical table is zero-extended to at least $`2^{13}`$ cells (Akita `PCS_MIN_CELLS_LOG2`).

### 4.2 Padded rows

For $`i\ge N`$ every value is zero: $`z=u=v=b_t=0`$, so $`w_1'=0`$ and (3.1) holds with $`A_i=T_i=0`$. Padded bridge positions are zero bits, padded carries and parities are zero. By (4.1) the digits of zero are $`-b_k/2`$ (0 for the top digit of a signed value); padded digits are thus nonzero and MUST NOT be required to be zero (Aerie 3.4). The Keccak side contains live records only.

### 4.3 Range proofs by digit decomposition

A value $`X`$ with digits $`Z_0,\ldots,Z_{m-1}`$ of bases $`b_0,\ldots,b_{m-1}`$ (Section 4.1) is the virtual table
```math
X=\sum_{k\lt m}W_kZ_k+C_X,\qquad W_k=\prod_{l\lt k}b_l,\qquad C_X=\sum_{k\lt m}W_k\tfrac{b_k}{2}-\epsilon_X\tfrac{B}{2},\qquad B=\prod_{k\lt m}b_k,
\qquad(4.1)
```
with $`\epsilon_X=1`$ for a signed and $`0`$ for an unsigned value. When every $`Z_k`$ lies in its balanced alphabet, (4.1) is a bijection onto $`[-B/2,B/2-1]`$ or $`[0,B-1]`$ (checked by enumeration in `spec_checks.py`):

| Value | Bases | Range proven | $`C_X`$ |
| --- | --- | --- | ---: |
| $`z`$ | $`16,16,16,16,4`$ | $`[-2^{17},2^{17}-1]`$, so $`\bar B_z=2^{17}`$ | 34,952 |
| $`u`$ | $`16,16,16,16,8`$ | $`[-2^{18},2^{18}-1]`$, so $`\bar B_u=2^{18}`$ | 34,952 |
| $`v`$ | $`16^6,8`$ | $`[-2^{26},2^{26}-1]`$ | 8,947,848 |
| $`\kappa^{(1)}_t,\kappa^{(2)}_t`$ | $`16,16,4`$ | $`[0,1023]`$ | 648 |
| $`\kappa^{(3)}`$ | $`16,16`$ | $`[0,255]`$ | 136 |

Each digit alphabet is proven as in Aerie Sections 3.3 and 7.2 to 7.4. For a digit $`Z`$ the folded table $`P_Z(x)=Z(x)(Z(x)+1)`$ (virtual, extended separately) satisfies $`Q_b(P_Z)=0`$ on the Boolean domain exactly when $`Z\in[-b/2,b/2-1]`$, with $`Q_4(P)=P(P-2)`$, $`Q_8(P)=P(P-2)(P-6)(P-12)`$ and $`Q_{16}=Q_{16}^{\rm lo}Q_{16}^{\rm hi}`$, $`Q_{16}^{\rm lo}=Q_8`$, $`Q_{16}^{\rm hi}(P)=(P-20)(P-30)(P-42)(P-56)`$ (Aerie (3.7), (3.8)); exactness holds over $`\mathbb F_p`$ because $`Z(Z+1)=t`$ has at most two roots. A base-16 digit carries the virtual Boolean tables $`A_Z^{\rm lo},A_Z^{\rm hi}`$ (Section 5). The checks are `b16_prod` (stage 1), `leaf` (stage 2) and `link` (stage 3) of Section 7. There are 40 base-16 digits (14 on the main domain, 12 on $`D_1`$, 12 on $`D_2`$, 2 on $`D_3`$), 2 base-8 digits ($`u_4,v_6`$) and 13 base-4 digits ($`z_4`$ and the top carry digits of levels 1 and 2): 55 digits.

Catalog rows bind nothing for a malicious prover (Aerie `falcon-pcs-extraction.md` Section 2), and the source inner basis is capped at 16 at the pinned revision (`PROOF_OPTIMIZED_INNER_LOG_BASIS_MAX`), so no 17-bit single-digit row exists either. The verifier MUST reject a digit layout other than the table above. Base-32 digits would save three main-domain columns; not adopted.

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
| $`P_Z`$; $`A_Z^{\rm lo},A_Z^{\rm hi}`$ | domain of $`Z`$ | $`Z(Z+1)`$; $`Q_{16}^{\rm lo}(P_Z),Q_{16}^{\rm hi}(P_Z)`$ | $`\{0,2,\ldots,56\}`$; one of them 0 |

$`y^{(1)}_t`$ is an integer identity in the committed $`b_t`$ and the digits of $`\kappa^{(1)}_t`$: the sum has at most 2,047 terms (row weight bound, Section 9.2) and $`\kappa^{(1)}_t\in[0,1023]`$ (Section 4.3), so the $`\mathbb F_p`$ definition agrees with the integer one. Affine combinations of opened values ($`z,u,v,\kappa`$ by (4.1), $`w_1'(r)`$, $`y^{(1)}_t(r)`$ through its defining sum-check) are evaluated by the verifier from the opened components.

## 6. Residual: committed quotient instead of the divisibility projection

Aerie proves $`q\mid\Delta`$ with a seeded projection $`\Pi_{\rm res}:\mathbb F_p^{nN^\star}\to\mathbb F_p^{31}`$ (cores with entries uniform in $`[-12,12]`$) whose integer outputs must stay below $`p/2`$. For ML-DSA the residual without the $`qv`$ term, $`\Delta_0=(Az)-T-2\gamma_2w_1'-u`$, has honest coefficients of root-mean-square size about $`2^{42.4}`$ (worst case $`\lt2^{49}`$ by (3.4')), so an honest output has standard deviation about $`2^{68.4}`$ at $`nN^\star=2^{25}`$ and wraps modulo $`p`$ ($`p/2\approx2^{63}`$) with probability close to one: the field is too small for the Falcon-style projection of an ML-DSA residual, whose entries are $`2^{18}`$ times larger than Falcon's.

Therefore this protocol commits the quotient $`v`$ through its digits (Section 4) and proves (3.1) as the field identity $`\Delta=0`$ on the Boolean domain. Lifting: under the ranges proven in Section 4.3, $`\lvert(Az)\rvert\le2^{10}\cdot2^{22}\cdot2^{17}=2^{49}`$, $`\lvert T\rvert\lt2^{29}`$, $`2\gamma_2w_1'\lt2^{24}`$, $`\lvert u\rvert\le2^{18}`$ and $`\lvert qv\rvert\lt2^{23}\cdot2^{26}=2^{49}`$, so
```math
\lvert\Delta^{\mathbb Z}\rvert\lt2^{51}\lt p/2,
\qquad(6.1)
```
and $`\Delta=0`$ in $`\mathbb F_p`$ implies $`\Delta^{\mathbb Z}=0`$ over the integers, which is (3.1). The range proof of $`v`$ is what gives the identity content: $`q`$ is invertible modulo $`p`$, so an unbounded $`v=q^{-1}\Delta_0\bmod p`$ would satisfy $`\Delta=0`$ for every $`z,u,w_1'`$. Any $`\lvert v\rvert\lt(p/2-2^{50})/q\approx2^{40}`$ would do; the digits prove $`2^{26}`$, the honest width. The cost is seven digit columns per position. The identity is tested at the single point $`r^{(2)}`$ inside the stage-3 instance (7.6).

No norm projection is needed: H has only infinity-norm bounds, each proven per coordinate by the digits of Section 4.3 or by (3.4).

## 7. The sum-check stages

### 7.1 Overview

After both commitment rounds and the bridge seed (Section 10), the verifier samples one point $`\tau\in K^{d_{\rm main}}`$; $`\tau_t`$ denotes its last $`d_t`$ coordinates ($`t=1,2,3`$). Three batched sum-checks follow, each of $`d_{\rm main}`$ rounds with terminal point $`r^{(s)}`$, $`s=1,2,3`$; $`r^{(s)}_t`$ is the suffix of $`r^{(s)}`$ for domain $`D_t`$, and for a table $`Z`$ on $`D_t`$ the points $`\tau_Z,r^{(s)}_Z`$ denote these suffixes. Instances on smaller domains are inactive in the leading rounds and lifted by $`2^{-(d_{\rm main}-d_{\rm loc})}`$ exactly as in Aerie 7.1; every instance uses its own canonical variable order (Section 1) and the stage batches instances with independent coefficients $`\gamma^{(s)}`$ in the listed order. Ordinary compressed rounds send $`\deg`$ elements (Aerie 10.3, omitting the linear coefficient).

| Stage | Instances | Max individual degree | Terminal claims |
| --- | --- | ---: | --- |
| 1 | 66: bridge Booleanity and levels 2, 3; w1Encode Booleanity and range; 40 `b16_prod` | 4 | $`y^{(1)}_t(r^{(1)}_1)`$ (virtual), $`y^{(2)}_t(r^{(1)}_2)`$, $`b_t(r^{(1)})`$, $`A_Z^{\rm lo},A_Z^{\rm hi}(r^{(1)}_Z)`$ |
| 2 | 101: level-1 definition; 95 range leaves | 5 | $`b_t(r^{(2)})`$, $`P_Z(r^{(2)}_Z)`$ |
| 3 | 118: ring identity and functional; 55 range links; 62 evaluation reductions | 3 | the 67 committed tables at $`r^{(3)}`$ and its suffixes |

Every committed table is opened once, at the stage-3 terminal of its domain; earlier claims are reduced to $`r^{(3)}`$ in stage 3. The ring identity (3.1) is tested at the single point $`r^{(2)}`$ inside the stage-3 instance (7.6).

### 7.2 Pre-stage claims

Before $`\gamma^{(1)}`$ the prover sends the 26 evaluations $`y^{(2)}_t(\tau_2)`$, $`\kappa^{(2)}_{t,k}(\tau_2)`$ for $`t\lt6`$, $`k\lt3`$, and $`\kappa^{(3)}_k(\tau_3)`$, $`k\lt2`$; the verifier forms $`\kappa^{(2)}_t(\tau_2)`$ and $`\kappa^{(3)}(\tau_3)`$ by (4.1). The verifier computes $`Y(\tau_3)=\sum_{\rho_3,\sigma}\mathrm{eq}(\tau_3,(\rho_3,\sigma))Y_\sigma[\rho_3]`$ from the public final vectors ($`4096S`$ terms).

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
7. `b16_prod[Z]` for the 40 base-16 digits in the order of Section 4.1, on the domain of $`Z`$, degree 3 (Aerie (7.2)): $`0=\sum_x\mathrm{eq}(\tau_Z,x)A_Z^{\rm lo}(x)A_Z^{\rm hi}(x)`$.

Terminal: the prover sends $`y^{(1)}_t(r^{(1)}_1)`$ (six virtual claims, materialized on $`D_1`$ by integer parities), $`y^{(2)}_t(r^{(1)}_2)`$, $`b_t(r^{(1)})`$ and the 80 values $`A_Z^{\rm lo},A_Z^{\rm hi}(r^{(1)}_Z)`$; the verifier checks the batched terminal with the equality factors, $`w_2(r^{(1)}_1)`$ and $`w_{3,t}(r^{(1)}_2)`$ (Section 7.6).

### 7.4 Stage 2 (degree 5)

Before $`\gamma^{(2)}`$ the prover sends the 18 values $`\kappa^{(1)}_{t,k}(r^{(1)}_1)`$ and the verifier forms $`\kappa^{(1)}_t(r^{(1)}_1)`$ by (4.1). Instances:

1. `def_y1[t]`, main domain:
```math
y^{(1)}_t(r^{(1)}_1)+2\kappa^{(1)}_t(r^{(1)}_1)=\sum_{x}w_1(x)\,b_t(x),
\qquad
w_1(j,\rho,i)=\mathrm{eq}(r^{(1)}_{1,\beta},\beta_1(i))\,c_1(j,\rho,i_{\rm lo}),
\qquad(7.5)
```
$`c_1(\zeta_1)=\sum_{\rho_1}M_1[\rho_1,\zeta_1]\,\mathrm{eq}(r^{(1)}_{1,\rho},\rho_1)`$, $`\beta_1(i)=(i_{\rm mid},\sigma)`$.
2. `leaf16[Z,lo]`, `leaf16[Z,hi]` for every base-16 digit, degree 5 (Aerie (7.5)): $`A_Z^{b}(r^{(1)}_Z)=\sum_x\mathrm{eq}(r^{(1)}_Z,x)\,Q_{16}^{b}(P_Z(x))`$, $`b\in\{\mathrm{lo},\mathrm{hi}\}`$.
3. `leaf8[Z]` for $`Z\in\{u_4,v_6\}`$, degree 5, and `leaf4[Z]` for the 13 base-4 digits, degree 3: $`0=\sum_x\mathrm{eq}(r^{(1)}_Z,x)\,Q_b(P_Z(x))`$.

Terminal claims $`b_t(r^{(2)})`$ and $`P_Z(r^{(2)}_Z)`$ for the 55 digits; the verifier evaluates $`Q_b`$ on the claimed $`P_Z`$ values and the equality factors.

### 7.5 Stage 3 (degree 3) and the opening

Instances, in $`\gamma^{(3)}`$ order:

1. `ring`, degree 2, the first sum in order $`(j',s,i)`$ and the second in order $`(j,\rho,i)`$:
```math
T(r^{(2)})=\sum_{j',s,i}w_A(j',s,i)\,z(j',s,i)-\sum_{j,\rho,i}\mathrm{eq}\bigl(r^{(2)},(j,\rho,i)\bigr)\bigl(2\gamma_2w_1'+u+qv\bigr)(j,\rho,i),
\qquad(7.6)
```
with $`w_A`$ from Section 7.6 and $`T(r^{(2)})=\sum_{j,\rho,i}\mathrm{eq}(r^{(2)},(j,\rho,i))T(j,\rho,i)`$ ($`k\cdot256`$ multiply-adds per live record). Equation (7.6) is $`\sum_x\mathrm{eq}(r^{(2)},x)\Delta(x)=0`$ with $`(Az)`$ pulled back through $`A`$; the tables were fixed before $`r^{(2)}`$, so a nonzero $`\Delta`$ passes with probability at most $`d_{\rm main}/|K|`$.
2. `link[Z]` for the 55 digits, degree 3 (Aerie (7.8)): $`P_Z(r^{(2)}_Z)=\sum_x\mathrm{eq}(r^{(2)}_Z,x)\,Z(x)(Z(x)+1)`$.
3. Evaluation reductions $`f(\pi)=\sum_x\mathrm{eq}(\pi,x)f(x)`$, degree 2: $`b_t`$ from $`r^{(1)}`$ and from $`r^{(2)}`$; $`y^{(2)}_t`$ from $`\tau_2`$ and from $`r^{(1)}_2`$; $`\kappa^{(2)}_{t,k}`$ from $`\tau_2`$; $`\kappa^{(3)}_k`$ from $`\tau_3`$; $`\kappa^{(1)}_{t,k}`$ from $`r^{(1)}_1`$ (62 instances).

Terminal: the 67 committed values at $`r^{(3)}`$ and its suffixes; the verifier forms (4.1), $`w_1'`$, $`w_A(r^{(3)})`$, the equality factors and $`Z(Z+1)`$ per link, and checks the batched terminal.

One Akita opening authenticates the 67 evaluations at the four group points. No committed polynomial carries a claim at a second point.

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
with $`y^{(1)}_t,y^{(2)}_t,b_t\in\{0,1\}`$ enforced by (7.1), `bool_y2`, `bool_b`, and every carry in $`[0,1023]`$ or $`[0,255]`$ by its digits (Section 4.3). Each $`\mathbb F_p`$ equation then has both sides below $`2^{12}`$, so it holds over $`\mathbb Z`$ and the Boolean left-hand quantity is the GF(2) parity. Without the carry ranges the identities would be vacuous: 2 is invertible modulo $`p`$, so an unbounded carry fits either parity. Hence the $`\mathbb F_p`$ side proves $`Y_\sigma=M_3M_2M_1\,b`$ over GF(2) per segment and the Flock side the same for $`b'`$. The row weight of $`M_1,M_2`$ is at most 2,047 and of $`M_3`$ at most 511 with probability above $`1-2^{-400}`$; the honest prover MUST abort if a row weight exceeds 2,047 (levels 1, 2) or 511 (level 3), the carry ranges of Section 4.3 (completeness only).

### 9.3 Stage placement and costs

Levels 2 and 3 and all Booleanity checks are in stage 1, the level-1 definition in stage 2, the carry range checks in stages 1 to 3 with the other digits, and the evaluation reductions in stage 3 (Section 7). Costs at $`N=2^{15}`$: committed carry digits $`9{,}437{,}184+589{,}824+32{,}768`$ and parities $`196{,}608`$ (1.3% of the algebraic tables); GF(2) nonzeros $`\approx2.9\times10^9`$ (also the prover's bit operations for parities and carries); final vectors $`512S=2{,}048`$ bytes; verifier Section 7.6.

### 9.4 Soundness

Let $`\delta=b\oplus b'\ne0`$ be the difference of the two committed copies, fixed before $`\mathsf{sd}`$. By `ref/notes/improvements-bridge.md` Lemmas 1 to 3 and the composition theorem, $`\Pr[M_3M_2M_1\delta=0]\le\sum_t1/\binom{m_t}{k_t}=2^{-145.6}+2^{-145.6}+2^{-163.4}=2^{-144.6}`$ (segments and lanes are blocks; padded blocks are blocks). If the product is nonzero, the two sides cannot agree on $`Y_\sigma`$, so
```math
\varepsilon_{\rm bridge}\le2^{-144.6}+\varepsilon_{\rm PIOP}+\varepsilon_{\rm Flock}+\varepsilon_{\rm bind}.
\qquad(9.3)
```
Grinding on $`\mathsf{sd}`$ multiplies the first term by the number of seed derivations; the 16.6-bit margin above $`2^{-128}`$ covers it, and the seed site MAY be ground under Aerie's schedule.

## 10. Transcript and wire format

Backend: Spongefish `Blake2b512` as in Aerie 10.1, tag `agg-mldsa/v1/piop/blake2b` (64 bytes, zero-padded), session `B("agg-mldsa/v1")`, instance digest from the canonical instance record. Encodings, the per-limb $`K`$ sampler, LEB128 work nonces and the scheduled grinding are Aerie's; the work plan per admitted shape is an obligation of `agg-pcs-config`. Ordered events:

| # | Event | Payload |
| --- | ---: | --- |
| 0 | instance | wire version LE16, parameter-set byte 0x44, feature flags (H / H-exact / S1+S2), $`N`$, $`N^\star`$ LE64, public-record digest (Section 2), digit layout id (Section 4.3), the four group descriptors with catalog row ids, Flock profile id, bridge parameters $`(J_t,m_t,k_t)`$ |
| 1 | absorb | round-1 Akita commitment $`G_{\rm main}`$ (native encoding) |
| 2 | absorb | $`S`$ Flock commitments (Ligerito roots or LaBinius commitments), segment order |
| 3 | challenge | $`\mathsf{sd}`$: 32 bytes, no field sampling |
| 4 | absorb | round-2 Akita commitments $`G_1,G_2,G_3`$ in group order |
| 5 | challenge | $`\mathsf{fs}_\sigma`$: 32 bytes per segment, the challenger seed of the Flock sub-proof $`\sigma`$ (its challenger is initialized with $`\mathsf{fs}_\sigma`$) |
| 6 | absorb | $`Y_0,\ldots,Y_{S-1}`$ (512 bytes each), then the $`S`$ Flock proof byte strings, each framed by `B()` |
| 7 | challenge | $`\tau\in K^{d_{\rm main}}`$ |
| 8 | absorb | 26 pre-stage claims (7.2) |
| 9 | challenge | $`\gamma^{(1)}\in K^{66}`$; then the stage-1 rounds of 4 elements each and the 98 stage-1 claims |
| 10 | absorb/challenge | 18 claims $`\kappa^{(1)}_{t,k}(r^{(1)}_1)`$; $`\gamma^{(2)}\in K^{101}`$; stage-2 rounds of 5 elements; 61 stage-2 claims |
| 11 | challenge | $`\gamma^{(3)}\in K^{118}`$; stage-3 rounds of 3 elements; the 67 opening claims |
| 12 | PCS | Akita native stream seeded by the 32-byte PIOP session digest, as Aerie 12.3 |

Every challenge is drawn after the previous absorb. The bridge seed follows the round-1 and Flock commitments (events 1, 2), so $`b`$ and $`b'`$ are fixed before $`\mathsf{sd}`$, and precedes the round-2 commitments (event 4), whose carries and parities are functions of $`\mathsf{sd}`$, and every sum-check challenge. Proof layout: header (version, flags, $`N`$), the $`32N`$ bytes of $`\tilde c_i`$, then events 1, 2, 4, 6, 8 to 11 payloads in order, then the Akita opening. Size at $`N=2^{15}`$: $`32N`$ bytes of hashes; PIOP $`\approx(4+5+3)\cdot25\cdot16+270\cdot16\approx9.1`$ KB; Akita opening about 65 KB (Aerie's ledger scaled, estimate); Flock sub-proofs $`4\times0.5`$ MB (dominant until wrapped or replaced by LaBinius).

## 11. Verifier algorithm and cost

1. Parse the header; reject an unknown version, $`N\gt2^{15}`$, or a group descriptor, digit layout or catalog row other than those of Section 4. Compute $`N^\star`$, $`S`$, the domains.
2. Per live record: `pkDecode`, $`\mathsf{tr}_i,\mu_i,c_i,T_i`$ (Section 2); keep $`\mu_i`$ and $`c_i`$. Bind the public records.
3. Replay events 0 to 6: absorb the round-1 and Flock commitments, derive $`\mathsf{sd}`$ and $`M_1,M_2,M_3`$, absorb the round-2 commitments, derive $`\mathsf{fs}_\sigma`$; verify each Flock sub-proof against its public inputs ($`\mu_i`$, $`\tilde c_i`$, $`Y_\sigma`$) and the circuit digest recomputed from $`\mathsf{sd}`$.
4. Sample $`\tau`$; compute $`Y(\tau_3)`$; absorb the pre-stage claims; compute $`c_2,c_3`$ for $`(\tau_2,\tau_3)`$.
5. Verify stage 1; at the terminal evaluate $`w_2(r^{(1)}_1)`$, $`w_{3,t}(r^{(1)}_2)`$ and the equality factors. Compute $`c_1`$ for $`r^{(1)}_1`$.
6. Verify stage 2; at the terminal evaluate $`w_1(r^{(2)})`$, $`Q_b`$ on the claimed $`P_Z`$ values and the equality factors. Stream $`\hat A_i`$ (expand or load from cache), inverse-NTT, and accumulate $`T(r^{(2)})`$ and $`w_A(r^{(3)})`$ in one pass over $`A_i`$ by absorbing the stage-3 rounds first and deferring the terminal checks.
7. Verify stage 3: form $`z,u,v,w_1'`$ and the carries at $`r^{(3)}`$ by (4.1), check the batched terminal of (7.6), the links and the reductions. Verify the Akita opening of the 67 claims at the four points. Accept only if every check passes.

Cost at $`N=2^{15}`$, single core (misc memo 2.4, estimates): Keccak-f $`92N`$ (0.8 s) uncached, $`2N`$ cached; inverse NTTs $`16N`$ (0.5 s) uncached; $`T_i`$ (0.1 s); $`T(r^{(2)})`$ and $`w_A(r^{(3)})`$ $`5N\cdot1024`$ multiply-adds (0.2 s); bridge weights $`\approx1.6\times10^7`$ $`K`$-additions; $`S`$ Flock verifications (40 ms); PIOP rounds and the Akita opening (tens of ms). Totals about 1.7 s uncached and 0.4 s with cached $`A_i,\mathsf{tr}_i`$; both are reported.

## 12. Security

### 12.1 Knowledge soundness ledger

Errors are per transcript, union-bounded, before grinding; $`|K|=p^2\approx2^{128}`$.

| Component | Term |
| --- | --- |
| Stage sum-checks, degrees (4, 5, 3) | $`12\,d_{\rm main}/|K|`$ |
| Zero-checks at $`\tau`$ (19 Booleanity and range instances, 40 `b16_prod`) | $`\le59\,d_{\rm main}/|K|`$ |
| Range leaves at $`r^{(1)}`$ (95) and links at $`r^{(2)}`$ (55) | $`\le150\,d_{\rm main}/|K|`$ |
| Linear identities at a point: levels 2, 3 at $`\tau`$; level 1 at $`r^{(1)}_1`$; ring at $`r^{(2)}`$; 62 reductions | $`\le(13+1+6+1+62)\,d_{\rm main}/|K|`$ |
| Batching coefficients $`\gamma^{(1)},\gamma^{(2)},\gamma^{(3)}`$ | $`3/|K|`$ |
| Bridge (9.3) | $`2^{-144.6}`$ |
| Akita extraction, binding, catalog terms | by Akita's ledger for the four-group catalog |
| Flock knowledge soundness, per segment | by profile (`Fast`: 128-bit work-normalized; LaBinius: 128-bit) |

At $`d_{\rm main}=25`$ the field terms sum to at most $`7{,}603/|K|\lt2^{-115}`$; grinding every site to Aerie's schedule brings each to $`2^{-128}`$. Point reuse ($`\tau`$ for all zero-checks, $`r^{(1)}`$ for the leaves, $`r^{(2)}`$ for the links and the ring identity) is sound because every tested table is committed before the point is sampled (Aerie 9.6).

### 12.2 Extraction

From an accepting transcript the Akita extractor yields the 67 committed tables as $`\mathbb F_p`$ values with no range guarantee (Akita Remark 10.41). The PIOP ledger then gives, except with the field terms above: every digit in its alphabet (`b16_prod`, leaves, links), hence $`z,u,v`$ and all carries in the ranges of Section 4.3 by (4.1), which is (3.2) with $`(\bar B_z,\bar B_u)=(2^{17},2^{18})`$; $`b_t,y^{(2)}_t\in\{0,1\}`$ and (3.4) on every live position; (9.2) over $`\mathbb Z`$; $`\Delta=0`$ in $`\mathbb F_p`$, hence (3.1) over $`\mathbb Z`$ by (6.1). The Flock extractor yields bits $`b'`$ with $`\tilde c_i=\mathrm{SHAKE256}(\mu_i\Vert b'_i)`$ and $`Y_\sigma=M_3M_2M_1b'`$; the bridge gives $`b=b'`$ except with (9.3). Hence every live record satisfies $`\mathrm H(\bar B_z,\bar B_u)`$. Domain separation between the Keccak argument's SHAKE256 and the public-record binding rests on the distinct cSHAKE customizations and on $`\mu_i`$ being a hash output (random-oracle modeling as in Aerie 2.3).

### 12.3 Aggregate unforgeability

Game: the adversary gets honest keys and a signing oracle and outputs a verifying aggregate containing an unqueried $`(\mathsf{pk}_{i^\star},M_{i^\star})`$ with honest $`\mathsf{pk}_{i^\star}`$. Reduction to MLWE and SelfTargetMSIS with bound $`\zeta'`$ (3.5), classical ROM, following the Dilithium/FIPS 204 proof: replace $`t_{i^\star}`$ by uniform (MLWE), simulate signatures by programming H on $`(\mu\Vert\mathrm{w1Encode})`$, run the extractor of 12.2 on the forgery, and output $`(z_{i^\star},-c_{i^\star},c_{i^\star}t_{0,i^\star}-u_{i^\star})`$ as in Section 3.3; the forger's random-oracle query $`\mathrm H(\mu_{i^\star}\Vert\mathrm{w1Encode}(w'_1))`$ is read from the extracted Keccak witness. Freshness is on $`(\mathsf{pk},M)`$; $`\tilde c_i`$ is aggregator-chosen and need not match any signature the signer produced. Losses: the extraction terms above, one MLWE instance, and no re-encoding step. With $`\zeta'=421{,}888`$ the MSIS instance is 20.5% looser than FIPS 204's; whether ML-DSA-44's margin absorbs this at category 1 MUST be established with a lattice estimator before any production statement (open obligation).

## 13. Test plan

Unit tests (per crate, `cargo test --release` where heavy; batch sizes at most 1K signatures and $`2^{14}`$ permutations on the development machine):

- Parameters: every constant of Section 1 recomputed from FIPS 204 values; (3.4) by enumeration over $`[0,64)`$; the UseHint residual maximum 190,465 by enumeration over $`\mathbb Z_q\times\{0,1\}`$; $`B_v`$ and (3.4'), (6.1) with exact integer arithmetic; the digit bijections (4.1) and the exactness of $`Q_4,Q_8,Q_{16}`$ by enumeration (`spec_checks.py` items 8 to 10).
- Witness (`agg-witness`): against a FIPS 204 reference implementation on random and edge-case signatures (Decompose edge residue $`-\gamma_2`$, hints at both signs, $`\lVert z\rVert_\infty=B_z`$): native verify, derived $`(z,u,v,b)`$ and their digits satisfying (3.1) to (3.3) and (4.1) over $`\mathbb Z`$, $`(Az)`$ by CRT against schoolbook, padded rows at the canonical digits of zero.
- Identities, one positive and one negative test each: (7.1) to (7.6), `bool_y2`, `bool_b`, every leaf, link and reduction; the negative test flips one witness value (a non-Boolean bit, $`w_1'=44`$, a digit outside its alphabet, a carry of the wrong parity, $`v=q^{-1}\Delta_0`$ for a wrong $`z`$, a changed $`Y_\sigma`$ bit) and MUST fail at the stated stage.
- Weights: $`w_A`$ against a direct negacyclic reference at random $`K`$ points for every $`(\rho,s)`$; $`\Lambda`$ trie against (7.8); $`w_1,w_2,w_3`$ against dense matrix transposes at $`J`$ reduced to $`2^{10}`$; zero weights on padded positions.
- Bridge (`agg-bridge`): seed derivation determinism and distinct column streams; column weight exactly $`k_t`$; parities and carries against a GF(2) reference; (9.2) lifting with carries at the range boundary; the composite walker against $`M_3M_2M_1`$ materialized at $`J`$ reduced to $`2^{10}`$; the small-parameter error $`1/\binom{m}{k}`$ measured empirically at $`(m,k)=(64,3)`$ against weight-2 differences.
- Keccak (`agg-keccak`): sponge chain against a SHAKE256 reference for 1, 7 and 9 blocks; bit order, public $`\mu`$, fixed padding, published $`\tilde c`$; the bridge slot at $`k_{\log}`$ 16 and 20 (8 and 128 records) proves and verifies and rejects a flipped $`Y_\sigma`$ bit or a bit changed in one copy only; a sub-batch of $`2^{14}`$ permutations proves and verifies; $`k_{\log}=26`$ is measured on the cloud machine.
- PCS config (`agg-pcs-config`): the width-4 preset admits every digit alphabet and the padded digits; catalog rows exist for the four-group layouts at $`d_{\rm sig}\in[3,15]`$; a two-round commit ($`G_{\rm main}`$, then $`G_1,G_2,G_3`$) opens in one batch and verifies.
- Transcript: event order, a reordered or truncated proof is rejected, pinned proof digests for $`N\in\{8,256\}`$.

End-to-end: honest aggregates at $`N=256`$ and $`N=1{,}024`$ verify; each of the negative mutations above, applied to one record, is rejected; a wrong $`\tilde c_i`$ is rejected; cached and uncached verifier paths agree.

## 14. Milestones and crates

| Crate | Content | Milestone |
| --- | --- | --- |
| `agg-witness` | ML-DSA-44 native verify, key expansion, $`(z,u,v,b)`$ derivation, CRT product, table builders, padding | M1 |
| `agg-pcs-config` | Aerie's `Fp64SignedWidth4` preset, catalogs for the four-group layouts at $`d_{\rm sig}\in[3,15]`$, two-round commit driver, work plan | M1 (long lead, start first) |
| `agg-piop` | stages 1 to 3 with the range leaves and links, weight tables, verifier, Akita opening of the four groups | M2 |
| `agg-bridge` | seed to matrices, $`\mathbb F_p`$ parity and carry-digit tables, GF(2) composite walker, soundness constants | M2 |
| `agg-keccak` | Flock HEAD Keccak gate port, sponge union circuit, bridge slot type with the factored `LincheckCircuit`, segment driver | M3 |
| `agg-cli` | prove/verify binaries, wire format, record binding, caches | M2 (mode B: bits public, no Keccak), M3 (mode A) |
| `agg-bench` | harness at 1K locally and 32K on the cloud machine; prover, verifier (cached/uncached), memory | M2, M3 |

M1: witness, presets, catalogs, unit tests. M2: algebraic PIOP end to end with the bridge tables but $`b`$ published in the clear (mode B), measured at 8K. M3: Keccak argument and bridge, mode A at 256, 1K, then 32K. H-exact and S1+S2 (Appendix A) follow M3 as flags.

## Appendix A. Feature-flagged layouts

**H-exact** (`relation = h-exact`). Same tables as H, but $`z`$ and $`u`$ are replaced by slack-digit decompositions (`improvements-tables.md` Section 2, verified by enumeration there): $`z=130{,}915\,a_z+b_z-130{,}993`$ with $`a_z\in\{0,1\}`$, $`b_z\in[0,2^{17})`$, and $`u=118{,}787\,a_u+b_u-190{,}465`$ with $`a_u\in\{0,1\}`$, $`b_u\in[0,2^{18})`$. Committed: $`a_z,a_u`$ (Boolean, checked in stage 1) and the digits of the unsigned $`b_z`$ (bases $`16,16,16,16,2`$, $`C=100{,}488`$) and $`b_u`$ (bases $`16,16,16,16,4`$, $`C=166{,}024`$) under (4.1), checked like every other digit. The ring identity (7.6) reads $`z,u`$ as the affine combinations. The bounds are exact by construction and $`\zeta'=\zeta=350{,}209`$. Cost: $`+2{,}048`$ elements per signature (two columns per position).

**S1+S2** (`relation = fips`). The FIPS-faithful layout of `improvements-tables.md` Sections 1 and 2: per position $`z`$ (exact as above), $`r_1=4a+b`$ ($`a\in[0,8)`$, $`b\in[0,16)`$, Aerie's $`Q_8/Q_{16}`$ leaf), $`v_r=29{,}696\,a_v+b_v`$ ($`b_v\in[0,2^{16})`$), sign $`s\in\{-1,1\}`$, $`g\in\{-1,0,1\}`$, edge $`e`$, the six bits; per signature the hint slack $`u_h=17a_h+b_h`$ ($`b_h\in[0,64)`$). Identities: $`r_0=sv_r-\gamma_2e+(1+s)/2`$, $`e^2=e`$, $`s^2=1`$, $`e(1+s)=ev_r=er_1=0`$, $`g(g-1)(g+1)=0`$, $`g(s-g)=0`$, $`\sum g^2+u_h=\omega`$, and UseHint $`x(x-44)(x+44)=0`$ with $`x=r_1+g-w_1'`$; the ring identity uses $`2\gamma_2r_1+r_0`$ in place of $`2\gamma_2w_1'+u`$. Sixteen columns; the digits use the leaves of Section 4.3. This layout re-encodes the witness to a FIPS 204 signature (hint $`h=g^2`$), giving the black-box reduction to ML-DSA EUF-CMA.

## Appendix B. Decisions taken while writing

1. **Committed quotient instead of $`\Pi_{\rm res}`$** (Section 6). The task asked to reuse Aerie's divisibility projection; its honest output would be about $`2^{68}`$ and wrap modulo $`p`$. The quotient removes the projection seed, the Rice section and one stage-1 instance; the "$`\lvert\Delta\rvert\lt p/2`$" bound survives as (6.1).
2. **Ring identity tested at one point** (7.5). With the quotient committed, (3.1) is linear in multilinear tables; the stage-3 instance (7.6) at $`r^{(2)}`$ suffices, so $`r^{(2)}`$ doubles as Aerie's $`r_\Pi`$ and the weight pass runs between stages 2 and 3.
3. **Ranges by PIOP digit proofs, not by Akita** (4.3). The first draft read Akita's certified fold alphabets (Remark 10.24) as a source range guarantee; there is none (Remark 10.41). Every range is now a digit decomposition with Aerie's leaf predicates, which also makes $`(2^{17},2^{18})`$ exact.
4. **Segments** (Section 1): one bridge chain per $`2^{13}`$-signature segment, aligned with the Keccak sub-batches, since a GF(2) sub-proof constrains only its own bits; level 2 is padded to $`2^{19}`$ positions.
5. **Carry ranges by digits** (9.2). Without a carry bound the parity identities are vacuous (2 is invertible modulo $`p`$); each carry is now three digit columns (two at level 3), since the memo's lookup is unavailable (risks R5).
6. **Six lanes**: six bit polynomials rather than one table with a padded lane index (avoids 25% padding).
7. **Booleans in the width-4 profile** (4.1): one inner digit either way.
8. **Bit-range identity** as one degree-4 instance (7.4).
9. **Verifier representatives.** $`A_i`$ centered and $`T_i`$ unreduced over $`\mathbb Z`$; any fixed choice works, and these minimize (6.1).
10. **Flock challenger seeding** (event 5) after the bridge seed, binding the sub-proofs to the seeded rows (driver of Section 8).
11. **$`\tilde c_i`$ bound as part of the record digest** (Section 2), by analogy with Aerie's salt.
12. **Two Akita commitment rounds** (Section 10): round 1 before $`\mathsf{sd}`$, round 2 (the seed-dependent carries and parities) after it, under one contract (Akita 4.6).
13. **Composite GF(2) rows** (Section 8): Flock's lincheck is per block and its witness precedes the statement binding, so one bridge slot per segment proves $`M_3M_2M_1b'=Y_\sigma`$ directly.

## Review responses (2026-10-24)

Findings of the first review and their resolution; none is rebutted in substance.

1. No Akita range guarantee (findings 1, 4): agreed (Akita Remark 10.41). Fixed by the digits and range checks of Sections 4.3 and 7 for $`z,u,v`$ and every carry; the old Section 4.3, the "never range-checked" rule and B.3, B.5 are withdrawn. The count is 23,865 elements per signature (9,325 was the count without range checks).
2. Bridge groups before the seed (findings 2, 5): agreed. Fixed by the two commitment rounds of Section 10 (contract: Akita 4.6); nothing seed-dependent is a Flock witness (Section 8).
3. No cap on $`v`$ (finding 3): agreed. Fixed by the $`v`$ digits (ceiling $`2^{40}`$, Section 6) and the layout rule of Section 11.
4. Cross-record rows in Flock (finding 6): agreed; Section 8 follows the reviewer's outline. $`k_{\log}=26`$ remains unmeasured (risk R3); Section 13 tests $`k_{\log}`$ 16 and 20 locally.
5. Single-digit rows for widths 17 to 27 (finding 7): agreed; no column is wider than 4 bits; H-exact is reachable (Appendix A).

Costs changed materially (Sections 4.1, 7, 10); all figures are estimates until measured.
