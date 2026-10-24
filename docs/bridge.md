# Parity bridge: implementation specification (spec Section 9)

| Field | Value |
| --- | --- |
| Status | normative for milestones M2 (F_p side, `agg-bridge`, `agg-piop`) and M3 (Flock side, `agg-keccak` bridge slot); resolves the open items of `ref/notes/improvements-bridge.md` Section 5 and `docs/spec.md` Sections 8, 9 |
| Source of truth | `docs/spec.md` (Sections 1, 4, 5, 7, 8, 9, 10). This document only fixes what the spec leaves open; a conflict is a bug here. Deviations go to `docs/deviations.md` |
| Code | `crates/agg-bridge` (`shape`, `seed`, `parity`, `carry`, `weights`, `final_vector`, `gf2_walker`, `soundness`; generic over `Shape`), `crates/agg-witness/src/bridge.rs` (geometry, honest witness), `crates/agg-keccak/src/bridge_union.rs`, `tied.rs`, `flock_union.rs` (the mode-A Flock side, Sections 5a, 5b), `bridge_slot.rs`, `segment.rs` (the one-row slot and two-type union of Section 5, reference) |
| Counts | exact numbers are products of the shape constants and exact binomials (`math.comb`); every number marked "estimate" is unmeasured |

Problem. The 6,144 w1Encode bits of every signature are committed twice: as the Boolean columns $`b_0..b_5`$ of $`G_{\rm main}`$ (Akita, $`\mathbb F_p`$) and as the `Input` words of the sponge gate (Flock, GF(2)). The bridge proves $`b=b'`$ with a chain of three seeded sparse parity maps $`M_1,M_2,M_3`$: both sides prove $`Y_\sigma=M_3M_2M_1(\cdot)`$ for the same public 4,096-bit vector per segment, the $`\mathbb F_p`$ side over the integers with committed carries, the Flock side as GF(2)-linear rows. Soundness rests only on column independence (Section 7), not on code distance.

## 1. Shapes and notation

A **bridge shape** is three levels $`(J_t,m_t,k_t)`$, $`t=1,2,3`$, with $`J_t`$ and $`m_t`$ powers of two except $`J_3=\mathsf{lanes}\cdot m_2`$, plus `lanes` and the number $`B_1`$ of level-1 blocks per lane and segment. Level $`t`$ maps every block of $`J_t`$ input positions to $`m_t`$ outputs with one matrix $`M_t\in\{0,1\}^{m_t\times J_t}`$ shared by all blocks, lanes and segments; level 2 uses the first $`J_2^{\rm used}=B_1m_1`$ columns of $`M_2`$ (the rest are zero-padded positions, never read); level 3 takes all lanes of a segment, $`J_3=\mathsf{lanes}\cdot m_2`$. Row weights are random; honest carries need the caps $`W_t`$ below.

| Shape | Level 1 $`(J,m,k)`$ | Level 2 | Level 3 | lanes, $`B_1`$ | Output | Carry caps $`W_t`$, ranges |
| --- | --- | --- | --- | --- | ---: | --- |
| production (`Shape::PRODUCTION`, spec Section 1) | $`(2^{19},2^{13},14)`$ | $`(2^{19}$, used $`2^{17};\ 2^{13},14)`$ | $`(49{,}152,4{,}096,18)`$ | 6, 16 | 4,096 bits per segment | 2,047, 2,047, 511; $`[0,1023],[0,1023],[0,255]`$ |
| toy (`Shape::TOY`, tests only) | $`(2^{10},2^{6},4)`$ | $`(2^{8},2^{5},4)`$ | $`(32,16,4)`$ | 1, 4 | 16 bits | 127, 63, 31; $`[0,63],[0,31],[0,15]`$ |

The toy shape has $`L=B_1J_1=2^{12}`$ input bits, one lane and one segment; its per-level errors $`1/\binom{m_t}{k_t}`$ are $`2^{-19.3},2^{-15.1},2^{-10.8}`$ (sum $`2^{-10.75}`$), which is why it is a correctness instrument only. `agg-bridge` MUST implement parities, carries, weights and the GF(2) walker generically over `Shape`; `agg-witness::bridge` and `agg-piop` MAY hard-code the production constants. Event 0 binds $`(J_t,m_t,k_t)`$ (spec Section 10, deviation P1) and the verifier MUST reject any shape other than the production one in wire version 0; the toy shape is reachable only through the `test-hooks` feature of `agg-piop` (deviation P4) and the unit tests of `agg-bridge` and `agg-keccak`.

Indices (least significant first, spec Section 1): main index $`x=j+256\rho+1024i`$, lane $`t<6`$ is the polynomial $`b_t`$; $`i=i_{\rm lo}+2^9i_{\rm mid}+2^{13}\sigma`$. Output row of level $`t`$ is $`\rho_t`$; the output domains are $`D_1=(\rho_1,i_{\rm mid},\sigma)`$, $`D_2=(\rho_2,\sigma)`$, $`D_3=(\rho_3,\sigma)`$.

## 2. Column layout of the shared bit vector

Both sides index the same bit by the same in-block position, so that a level-1 block, a Flock sub-batch (segment) and the Akita domain prefix coincide. The table gives, for a bit of lane $`t`$ at position $`(j,\rho)`$ of record $`i`$, its address on each side (W3 in `docs/deviations.md`, `docs/keccak.md` Section 2):

| Object | $`\mathbb F_p`$ side (Akita) | Flock side (bridge slot of segment $`\sigma`$) |
| --- | --- | --- |
| bit $`b_t(j,\rho,i)`$ | column $`b_t`$, cell $`x=j+256\rho+1024i`$ | slot column $`c=j+256\rho+1024\,i_{\rm seg}+2^{23}t`$, $`i_{\rm seg}=i\bmod2^{13}`$ |
| level-1 in-block index $`\zeta_1=(j,\rho,i_{\rm lo})`$ | $`x\bmod2^{19}`$ | $`c\bmod2^{19}`$ |
| level-1 block | $`\beta_1=(i_{\rm mid},\sigma)=x\gg19`$ (one $`b_t`$ per lane) | $`(i_{\rm mid},t)=c\gg19`$ (lane $`t`$ at $`c\gg23`$) |
| level-1 output $`y^{(1)}_t(\rho_1,\beta_1)`$ | $`D_1`$ cell $`\rho_1+2^{13}\beta_1`$ (virtual, Section 4) | not materialized |
| level-2 in-block index $`\zeta_2=(\rho_1,i_{\rm mid})<2^{17}`$ | $`D_1`$ cell $`\bmod2^{17}`$; block $`\sigma=`$ cell $`\gg17`$ | implicit in the walker |
| level-2 output $`y^{(2)}_t(\rho_2,\sigma)`$ | $`D_2`$ cell $`\rho_2+2^{13}\sigma`$ (committed) | not materialized |
| level-3 in-block index $`\zeta_3=t+6\rho_2`$ | lane-major over $`(t,\rho_2)`$ | implicit |
| final $`Y_\sigma[\rho_3]`$ | public table $`Y`$ on $`D_3`$, cell $`\rho_3+2^{12}\sigma`$ | slot column $`6\cdot2^{23}+128+\rho_3`$, wired to the public word |

Alignment. The Akita main domain of lane $`t`$ at $`N^\star=2^{d_{\rm sig}}`$ records is $`2^{10+d_{\rm sig}}`$ cells; its $`2^{19}`$-cell prefixes are the level-1 blocks, so $`\mathrm{eq}`$ over the block index is a product of $`d_{\rm sig}-9`$ factors (spec 7.6). On the Flock side the sponge gate of record $`i_{\rm seg}`$ holds lane $`t`$ in the eight aligned 128-bit words $`4+8t+w`$, $`w<8`$ (bit $`j+256\rho`$ of the lane at word $`w=(j+256\rho)\gg7`$), and the bridge slot holds the same lane at words $`8i_{\rm seg}+2^{16}t+w`$: the copy constraint is word-for-word, with no bit permutation. When $`d_{\rm seg}=\min(d_{\rm sig},13)<13`$ the slot shrinks: the lane stride is $`2^{10+d_{\rm seg}}`$ instead of $`2^{23}`$, the block stride $`\min(2^{19},2^{10+d_{\rm seg}})`$, and $`k_{\log}=13+d_{\rm seg}`$. Padded records ($`i\ge N`$) are zero on both sides: committed zero bits in $`b_t`$ (spec 4.2) and columns with empty rows in the slot (Section 5), so the bridge also proves that the $`\mathbb F_p`$ padding is zero. Padded level-1 positions ($`d_{\rm seg}<9`$) and the level-2 positions $`\zeta_2\ge2^{17}`$ are public zeros with zero weight.

## 3. Seed and matrix sampling

**Transcript position** (spec Section 10, `docs/piop.md` Section 9). $`\mathsf{sd}`$ is the 32-byte squeeze of event 3 (`agg-mldsa/v1/bridge/seed`). Before it the transcript has absorbed event 0 (wire version, $`N,N^\star`$, the public-record digest over all $`(\mathsf{pk}_i,M_i,\tilde c_i)`$, the group descriptors and $`(J_t,m_t,k_t)`$), event 1 (the round-1 Akita commitment of $`G_{\rm main}`$, which contains $`b_0..b_5`$) and event 2 (the $`S`$ Flock commitments, each the Ligerito level-0 cap of the sub-batch whose witness contains $`b'`$ and the bridge-slot copies, deviation K7). Hence $`\delta=b\oplus b'`$ is fixed by binding commitments before $`\mathsf{sd}`$ exists. Everything seed-dependent comes later: the round-2 groups $`G_1,G_2,G_3`$ (event 4), the Flock challenger seeds $`\mathsf{fs}_\sigma`$ (event 5), $`Y_\sigma`$ and the Flock proofs (event 6). The Flock side binds $`\mathsf{sd}`$ twice: through $`\mathsf{fs}_\sigma`$ and through the bridge-slot digest absorbed into its challenger (Section 5).

**Derivation** (`agg_bridge::seed`, normative as implemented and known-answer tested). Column $`\zeta`$ of $`M_t`$ is read from $`\mathrm{cSHAKE256}(\mathsf{sd}\Vert\mathrm{LE64}(t)\Vert\mathrm{LE64}(\zeta),\cdot,N=\varepsilon,S=\texttt{agg-mldsa/v1/bridge})`$ with $`t\in\{1,2,3\}`$ the level number: take 16-bit little-endian words, keep the low $`\log_2m_t`$ bits (13, 13, 12), skip a value already drawn for this column, stop after $`k_t`$ distinct rows, stored in draw order. This is column-regular ($`|S_\zeta|=k_t`$ exactly), rejection-free on the row value (power-of-two $`m_t`$, a mask, no modular bias) and exactly uniform over $`k_t`$-subsets (the distinct values of an i.i.d. uniform stream form a uniform ordered $`k`$-tuple); the only skipped words are exact repeats, $`k(k-1)/2m\approx0.011`$ per column in expectation. The XOF is squeezed in 64-byte chunks (32 words); the word stream does not depend on the chunking. Both sides, and every segment, use the same three matrices; nothing is transmitted. Representation: `SparseMatrix` (column-major, `cols * k` row indices of `u32`), 62 MB for the production shape, derived once per proof or verification and shared by the $`\mathbb F_p`$ tables, the weights and the Flock walker.

Cost. $`2\cdot2^{19}+49{,}152=1{,}097{,}728`$ columns, one Keccak-f each (48-byte input after the shared customization prefix, first squeeze block suffices): measured 150 ms on 7 threads and 0.94 s on one (deviation P4). Candidate for wire version 1, not adopted: one XOF stream per 64 consecutive columns (0.23 Keccak-f per column) and $`J_2=2^{17}`$ without padding ($`-2^{18}`$ columns); both change event 0 and the known answers.

Completeness guard. The honest prover computes the row weights (`BridgeMatrices::row_weights_ok`) and MUST abort if a row of $`M_1`$ or $`M_2`$ has weight above 2,047 or a row of $`M_3`$ above 511; for the production shape the row weight is $`\mathrm{Bin}(J_t^{\rm used},k_t/m_t)`$ with means 896, 224, 216 and $`\Pr[\text{weight}\ge2048]\lt2^{-700}`$ at level 1 (exact binomial tail; $`\ge1152`$ is $`2^{-52.7}`$), so the abort never fires in practice. Soundness does not depend on it (Section 7).

## 4. The F_p side

Per segment, lane and level the prover proves, over the integers, $`\sum_\zeta M_t[\rho,\zeta]\,v^{(t-1)}(\zeta)=y^{(t)}(\rho)+2\kappa^{(t)}(\rho)`$ with $`v^{(0)}=b_t`$, $`v^{(1)}=y^{(1)}_t`$, $`v^{(2)}=(y^{(2)}_t)_{t<6}`$ and $`y^{(3)}=Y_\sigma`$ (spec (9.2)). What is committed, virtual or public, and where each identity is tested:

| Level | Left-hand quantity | Carry $`\kappa^{(t)}`$ (committed digits, bases) | Linear identity | Booleanity | Carry range | Weight point |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | $`y^{(1)}_t`$ virtual on $`D_1`$: $`\sum_\zeta M_1[\rho_1,\zeta]b_t-2\kappa^{(1)}_t`$ (spec Section 5) | $`\kappa^{(1)}_{t,0..2}`$ on $`D_1`$, bases $`16,16,4`$, $`C=648`$, range $`[0,1023]`$ | `def_y1[t]` (7.5), stage 2, degree 2, main domain, weight $`w_1`$ | `bool_y1[t]` (7.1), stage 1, degree 3, $`D_1`$ at $`\tau_1`$ | `b16_prod` stage 1 (deg 3), `leaf16` lo/hi stage 2 (deg 5), `leaf4` stage 2 (deg 3), `link` stage 3 (deg 3) | $`r^{(1)}_1`$ |
| 2 | $`y^{(2)}_t`$ committed Boolean on $`D_2`$ | $`\kappa^{(2)}_{t,0..2}`$ on $`D_2`$, bases $`16,16,4`$, $`[0,1023]`$ | `level2[t]` (7.2), stage 1, degree 2, $`D_1`$, weight $`w_2`$ | `bool_y2[t]`, stage 1, degree 3, $`D_2`$ at $`\tau_2`$ | same as level 1 | $`\tau_2`$ |
| 3 | $`Y_\sigma`$ public; $`Y(\tau_3)`$ computed by the verifier ($`4096S`$ terms) | $`\kappa^{(3)}_{0,1}`$ on $`D_3`$, bases $`16,16`$, $`C=136`$, $`[0,255]`$ | `level3` (7.3), stage 1, degree 2, $`D_2`$ summed over the six lanes, weights $`w_{3,t}`$ | none needed | `b16_prod`, `leaf16`, `link` | $`\tau_3`$ |

Stage 3 also carries the evaluation reductions: $`y^{(2)}_t`$ from $`\tau_2`$ and $`r^{(1)}_2`$, $`\kappa^{(2)}_{t,k}`$ from $`\tau_2`$, $`\kappa^{(3)}_k`$ from $`\tau_3`$, $`\kappa^{(1)}_{t,k}`$ from $`r^{(1)}_1`$ (spec 7.5). The level-1 claim $`y^{(1)}_t(r^{(1)}_1)`$ produced by `bool_y1` is discharged by `def_y1` in stage 2 after the prover sends the 18 digit values $`\kappa^{(1)}_{t,k}(r^{(1)}_1)`$ (spec 7.4); that is the only virtual link in the chain. All degrees are individual degrees of the summand including the $`\mathrm{eq}`$ or weight factor.

The 10-bit range by digits. $`\kappa=Z_0+16Z_1+256Z_2+648`$ with $`Z_0,Z_1\in[-8,7]`$, $`Z_2\in[-2,1]`$ is a bijection onto $`[0,1023]`$ (spec (4.1)); the level-3 carry $`Z_0+16Z_1+136`$ onto $`[0,255]`$. Every digit alphabet is proven by Aerie's leaf predicates (spec 4.3). Lifting: a row has at most $`W_t`$ ones (the verifier knows the exact row weights), the Boolean inputs make the sum at most $`W_t\lt2^{11}`$, and $`y+2\kappa\le1+2\cdot1023`$, so both sides of each identity are below $`2^{12}\ll p`$ and the $`\mathbb F_p`$ equation is an integer equation; reducing it modulo 2 gives $`y=\bigoplus_\zeta M_t[\rho,\zeta]v(\zeta)`$. For soundness any carry range below $`(p-W_t)/2`$ would do; the digits give completeness for $`W_t\le2047`$ (511).

**Weights.** For level $`t`$ at the point $`\pi`$ of $`D_t`$ (split $`\pi=(\pi_\rho,\pi_\beta)`$) the summation weight is $`w_t(\beta,\zeta)=\mathrm{eq}(\pi_\beta,\beta)\,c_t(\zeta)`$ with $`c_t(\zeta)=\sum_{\rho\in S_\zeta}\mathrm{eq}(\pi_\rho,\rho)`$ (spec 7.6; `agg_bridge::weights`, wrapped by `agg_piop::weights` at the production shape). Prover: one $`c_t`$ vector per level, shared by all lanes, blocks and segments. Verifier, at the terminal point $`r`$ of the summed domain: $`w_t(r)=\mathrm{eq}(\pi_\beta,r_\beta)\sum_{\zeta}\mathrm{eq}(r_\zeta,\zeta)\,c_t(\zeta)`$, that is one $`\mathrm{eq}`$ product over the block variables (the tensor factor, $`d_{\rm sig}-9`$, $`d_{\rm sig}-13`$ or $`d_{\rm sig}-13`$ coordinates) times a sparse contraction inside one block. Operation counts in $`K`$ per level and point, production shape, exact:

| Level | $`c_t`$: additions $`k_tJ_t^{\rm used}`$ | $`\mathrm{eq}`$ tables: mults $`m_t+J_t^{\rm used}`$ | contraction: mults $`J_t^{\rm used}`$ | total mults |
| --- | ---: | ---: | ---: | ---: |
| 1 ($`\pi=r^{(1)}_1`$, $`r=r^{(2)}`$) | 7,340,032 | 532,480 | 524,288 | 1,056,768 |
| 2 ($`\pi=\tau_2`$, $`r=r^{(1)}_1`$) | 1,835,008 | 139,264 | 131,072 | 270,336 |
| 3 ($`\pi=\tau_3`$, $`r=r^{(1)}_2`$, six lanes) | 884,736 | 53,248 | 49,152 | 102,400 |

Total per verification: $`1.0\times10^7`$ $`K`$-additions and $`1.4\times10^6`$ $`K`$-multiplications (a $`K`$-addition is two $`\mathbb F_p`$ additions, a multiplication three $`\mathbb F_p`$ products), about 10 to 20 ms on one core (estimate), independent of $`N`$. The prover's weight tables cost the same additions plus $`|D_t|`$ multiplications to spread $`c_t`$ over the blocks. Committed cost at $`N=2^{15}`$: $`18\cdot2^{19}+24\cdot2^{15}+2\cdot2^{14}=10{,}256{,}384`$ carry and parity elements, 1.3% of the algebraic tables; prover bit work $`\sum_t k_t\cdot(\text{inputs of level }t)\approx2.9\times10^9`$ integer additions to form the sums, parallel over blocks.

## 5. The Flock side: the bridge slot and its `LincheckCircuit`

Flock HEAD's lincheck is per block ($`A=I\otimes A_0`$) and the whole union witness is committed before `bind_statement_circuit`, so the seeded rows cannot be witness-dependent definitions (`define_linear`) across the sponge slots and no intermediate parity may be a witness. Therefore each segment's sub-batch is a two-type union: $`2^{d_{\rm seg}}`$ instances of the sponge gate (`docs/keccak.md`) and **one** instance of the type `bridge` (spec Section 8, decision B.13). Its witness is copies of the bits, so it is committed before $`\mathsf{sd}`$ (deviation K7); its rows are generated from $`\mathsf{sd}`$ afterwards.

**Block layout** ($`k_{\log}=13+d_{\rm seg}`$, 26 at a full segment; lane stride $`\Lambda=2^{10+d_{\rm seg}}`$):

| Columns | Content | Row (C = I: $`(Az)_c(Bz)_c=z_c`$) |
| --- | --- | --- |
| $`[0,6\Lambda)`$: $`c=\mathsf{pos}+1024\,i_{\rm seg}+\Lambda t`$ | input bit copies, record $`i_{\rm seg}`$, lane $`t`$ | live record: $`z_c\cdot1=z_c`$ ($`A=e_c`$, $`B=e_{\rm one}`$); padded record: empty row (forces $`z_c=0`$) |
| $`6\Lambda`$ | constant one (`const_pin_col`) | $`1\cdot1=1`$ |
| $`6\Lambda+128+\rho_3`$, $`\rho_3<4096`$ | $`Y_\sigma[\rho_3]`$, 32 words, `Out` schema words wired to the public words | $`\big(\bigoplus_{c}M^{(\sigma)}[\rho_3,c]\,z_c\big)\cdot1=z_{Y}`$ ($`A=`$ composite row, $`B=e_{\rm one}`$) |
| the rest up to $`2^{k_{\log}}`$ (lanes 6, 7, gaps) | zero | empty rows |

`useful_bits` $`=6\Lambda+128+4096`$ (50,335,872 at a full segment; 393,249 committed words for the one live row, since the dense stack is count-proportional). The composite $`M^{(\sigma)}=M_3M_2M_1`$ is never materialized (it is dense: 4,096 rows over $`5\times10^7`$ columns). The `In` schema lists the $`6\cdot8\cdot2^{d_{\rm seg}}`$ input words; one wire class per live $`(i_{\rm seg},t,w)`$ joins sponge word $`4+8t+w`$ of row $`i_{\rm seg}`$ to bridge word $`8i_{\rm seg}+2^{16}t+w`$ of row 0 (393,216 classes at a full segment, proven by Flock's permutation grand product). Padded records have no wire and no identity row.

**Witness rows** (`BridgeSlot::rows(z) -> (a, b)`, consumed by Flock's `fill_slot`): $`a_c=z_c,b_c=1`$ on live input columns; $`a_{\rm one}=b_{\rm one}=1`$; $`a_{Y_o}=`$ the GF(2) parity $`(M^{(\sigma)}b')_o`$, $`b_{Y_o}=1`$; zero elsewhere. The parities are the forward walk $`M_1`$, then $`M_2`$, then $`M_3`$ with XOR (`agg_witness::bridge::gf2_composite`, `agg_piop::mode_b::segment_projection`: the six lanes of a position travel as one 6-bit mask, 14 byte-XORs per nonzero position at level 1); $`7.2\times10^8`$ bit operations per full segment.

**`LincheckCircuit`** (`flock-core/src/lincheck.rs`): `n_cols() = 2^k_log`; `const_pin_col() = Some(6 Lambda)`; `fold_alpha_batched(alpha, eq)` returns $`\mathrm{comb}[c]=\alpha\,\xi_A(c)+\xi_B(c)`$, $`\xi_M(c)=\sum_r\mathrm{eq}[r]M[r,c]`$, and `fold_split(eq)` the pair $`(\xi_A,\xi_B)`$ from one pass. Rows are indexed by their C column. The walker:

```text
v3[o]      = eq[Y_col(o)]                          for o < 4096          (gather)
v2[zeta3]  = sum_{rho3 in S3(zeta3)} v3[rho3]      zeta3 = t + 6 rho2    (18 adds each)
v1[t][z2]  = sum_{rho2 in S2(z2)} v2[t + 6 rho2]   z2 = (rho1, i_mid)    (14 adds each)
v0[t][i_mid][z1] = sum_{rho1 in S1(z1)} v1[t][rho1 + 2^13 i_mid]         (14 adds each)
xi_A[c]    = [live] eq[c] + v0[c]                  c = z1 + 2^19 i_mid + Lambda t
xi_A[one]  = eq[one];  xi_B[one] = eq[one] + sum_{live c} eq[c] + sum_o eq[Y_col(o)]
comb       = alpha * xi_A + xi_B                  (all other entries zero)
```

This is $`M_1^{\mathsf T}(M_2^{\mathsf T}(M_3^{\mathsf T}\mathrm{eq}_Y))`$ in $`\mathbb F_{2^{128}}`$, where GF(2) and field products agree because the characteristic is 2; the sparse column lists are read from the shared `BridgeMatrices` (generated from $`\mathsf{sd}`$, Section 3), so the rows exist only as this recurrence. Cost per fold, full segment, exact: $`884{,}736+11{,}010{,}048+704{,}643{,}072=7.17\times10^8`$ $`\mathbb F_{2^{128}}`$ additions (128-bit XORs), $`5.0\times10^7`$ multiplications by $`\alpha`$, one read of the $`2^{26}`$-entry `eq` and one write of `comb` (1 GiB each). Parallel over lanes and level-1 blocks (96 independent $`2^{19}`$-vectors at level 1). Flock's verifier calls the same `fold_alpha_batched` at its own `eq` table (`lincheck::verify`, step 2), so the bridge-slot verification costs the walker plus the $`2^{26}`$-entry `eq` build per segment: an estimated 0.3 to 1 s and 2 GiB per segment on 7 threads, to be measured on the cloud machine (risk R3; the spec's verifier budget of Section 11 does not yet include it). Sub-batches at $`d_{\rm seg}\le10`$ (local tests) fold $`2^{23}`$ entries or fewer.

**Binding.** The registry type carries empty matrix stubs (Flock's walker convention, deviation K2), so the relation is bound by `BridgeSlot::digest` = SHA3-256 of (`agg-keccak/v1/bridge-slot`, $`\mathsf{sd}`$, $`(J_t,m_t,k_t)_{t\le3}`$, $`d_{\rm seg}`$, the live count, $`k_{\log}`$, `useful_bits`, the constant column, the $`Y`$ base), absorbed into the Flock challenger after $`\mathsf{fs}_\sigma`$ and the sponge digest (deviation K4) and before Flock's own statement binding, which covers the registry, the counts, the commitment and the public words $`(\mu_i,\tilde c_i)_{i}`$, $`Y_\sigma`$. The driver order per segment (`segment.rs`): commit the two-type union (event 2) $`\to`$ receive $`\mathsf{sd}`$, derive matrices, compute parities $`\to`$ receive $`\mathsf{fs}_\sigma`$ $`\to`$ build rows, bind, prove $`\to`$ emit $`Y_\sigma`$ and the proof (event 6).

## 5a. The tied part: two Flock instances per part, tied by evaluation claims

Status: normative for M3 as implemented in `crates/agg-keccak/src/tied.rs` and
`flock_union.rs` (deviation K11); it replaces the two-type union of Section 5 as the mode-A
Flock argument. Section 5 stays the specification of the bridge slot itself (its block
layout, rows, walker, binding and the `Y` coefficient of K8). Since tied-part proof version 2
the bridge instance is the bridge union of Section 5b (deviation K13): the sponge instance,
its claims, the public check and the shares below stay as they are; the bridge-side regions,
points, commitment order and costs of this section describe proof version 1.

**Why.** The two-type union pays for the bridge slot three times (docs/keccak.md Section 8):
Flock's wiring argument costs one cell-slot, one gather claim and 16 proof bytes per copied
word (768 bytes per signature, 73% of the prover time in the opening), the uniform row
capacity makes the one-row bridge slot `2^(k_b + nu)` bits of address space (3 GiB of
buffers at 512 records, 128 GiB at a full segment), and the bridge walker itself. The first
two are removed by giving the bridge slot its own Flock instance and replacing the word
copies by evaluation claims.

**Instances.** Part `s` of a segment (`2^d` record positions, `n` live records) is proven by
two single-type Flock proofs, each a boolean zerocheck, a union-column lincheck and a merged
Ligerito opening over a registry of one type (`flock_union::UnionGeometry`, the pipeline of
Flock's `prove_fast_ligerito_union` composed from its public building blocks, with extra
packed-direct claims in the opening):

| Instance | Type | Rows | Address space | Claims in the opening |
| --- | --- | --- | --- | --- |
| sponge | the sponge gate (docs/keccak.md Section 2), `k_log = 19` | `n` live of `2^nu_s`, `nu_s = capacity_log(n, 19)` | `M_s = nu_s + 19` | the two class claims and `v_pub`, `v_s0`, `v_s1` |
| bridge | the bridge slot of Section 5, `k_b = 13 + d` | 1 live of `2^nu_b`, `nu_b = max(3, 22 - k_b)` | `M_b = nu_b + k_b` | the two class claims and `v_b0`, `v_b1` |

Nothing is wired: there is no circuit, no wiring argument and no public segment in either
proof. The sponge instance's statement words `(mu_i, c~_i)` are bound by a digest in the
challenger and proven by the claim `v_pub`.

**Regions.** Flock commits a slot BatchMajor: word column `c` of row `j` is the padded word
`(c << nu) + j`, and an evaluation claim is on the multilinear extension of the padded word
vector (one `F_{2^128}` element per 128-bit word) at a point of `M - 7` coordinates, rows
low. The sponge block has the w1Encode field first (docs/keccak.md Section 2): lane `t` of a
record is the eight words `8 t + w`, so lanes 0..3 are the word columns `[0, 32)` and lanes
4, 5 the columns `[32, 48)`, two aligned subcubes of the column index `[w (3 bits) | t |
high bits]`; mu, `c~`, the constant (bit 0 of word 54) and a zero word are the public region
`[48, 56)`. In the bridge block lane `t` of record `i` is the eight words `t 2^(3 + d) + 8 i
+ w`, so the same two lane groups are the subcubes `[w (3) | i (d) | t | high]`. With the
challenges `rho = (rho_w, rho_i, rho_t, rho_t', rho_r, rho_c)` (3, `d`, 2, 1, `nu_s`, 3
coordinates, `tied::TieChallenges::derive`, a cSHAKE-style Flock challenger over the part's
seed `fs`, `d` and `nu_s`) the five points are

```text
P_pub = [rho_r (nu_s) | rho_c (3) | 0 1 1 | 0 ...]                       (sponge, columns 48..55)
P_s0  = [rho_i[0..min(nu_s, d)], 0 ... (nu_s) | rho_w | rho_t (2) | 0 | 0 ...]   (sponge, lanes 0..3)
P_s1  = [same rows | rho_w | rho_t' | 0 | 1 | 0 ...]                      (sponge, lanes 4, 5)
P_b0  = [0 (nu_b) | rho_w | rho_i (d) | rho_t (2) | 0]                    (bridge, lanes 0..3)
P_b1  = [0 (nu_b) | rho_w | rho_i (d) | rho_t' | 0 | 1]                   (bridge, lanes 4, 5)
```

and the prover's values are the honest evaluations (`flock_union::eval_packed`, the region's
words times the eq tensor; 32 words per record on either side).

**Checks.** After both Flock proofs verify, the verifier checks
`v_pub = sum_{j < n} eq(rho_r, j) sum_{c8 < 8} eq(rho_c, c8) P[j][c8]` with `P[j] = (mu_j
words, c~_j words, 1, 0)`, and `v_bh = eq(rho_i[nu_s..d], 0) v_sh` for both lane groups,
where the factor `prod_k (1 + rho_i[k])` over the coordinates `k >= nu_s` is 1 unless the
sponge instance has fewer row coordinates than the part has record positions (a part with
few live records): the sponge's records `>= 2^nu_s` do not exist and the bridge's are padded
zeros.

**Soundness.** Let `z_s`, `z_b` be the committed word vectors (Flock's knowledge soundness
gives them). Restricted to a lane group, `f_s(rho) = sum_{i, t, w} eq(rho_i, i) eq(rho_t, t)
eq(rho_w, w) z_s[i; t, w]` (with `z_s[i] = 0` for `i >= 2^nu_s`) and `f_b(rho) = sum eq(...)
z_b[t, i, w]` are the multilinear extensions, in the same `d + 5` free coordinates, of the
sponge inputs and of the bridge copies over the part's `2^d` record positions (the sponge's
dummy rows are zero by Flock's witness contract and the bridge's padded positions are forced
zero by their empty rows, Section 5). If any bit differs, `f_b - eq(rho_i[nu_s..], 0) f_s`
is a nonzero multilinear polynomial and vanishes at `rho` with probability at most `(d +
5) / 2^128 <= 2^-123.8` (Schwartz-Zippel); `rho` is derived from `fs`, drawn at event 5
after both commitments were absorbed at event 2, so the vectors are fixed before `rho`. The
public claim is the same argument with `nu_s + 3` coordinates against the verifier's own
table, and it also pins the constant column to 1 on live rows and 0 on dummy rows (Flock's
count-derived constant pin requires exactly that). Hence the sponge instance's `b'` (the
bits its gates hash) equals the bridge instance's copies, whose rows prove the part's share
`Y_{sigma, s} = M^{(sigma)} b'_s` with `sd` and `Y_{sigma, s}` bound as in Section 5 and K8;
the shares XOR to `Y_sigma` (K10) and the bound (9.3) of Section 7 is unchanged, with an
additional `4 (d + 5) / 2^128` per part for the ties, inside `epsilon_Flock`. The two
proofs share `fs` but have distinct challenger labels and digests (the gate digest and the
public digest; the slot digest), so neither can be replayed as the other.

**Costs** (exact sizes; measurements in `ref/notes/bench-local.md`). Sponge instance: the
sponge-only union of docs/keccak.md Section 4 plus three claims (one `eq` tensor of `2^nu_s`
and three of 8 or 32 entries). Bridge instance: `3 x 2^(k_b + nu_b - 7)` words of padded
buffers and a `2^(k_b + nu_b - 3)`-byte stripe (64 MiB each at a full segment), the walker's
`eq`, `xi_A`, `xi_B` and comb vectors of `2^k_b` field elements (1 GiB each at `k_b = 26`),
a zerocheck over `M_b` run-list gated to the live row, and a Ligerito opening of a
`2^(k_b - 7)`-word dense stack. Verifier: the lincheck's matrix discharge runs the walker
once at its own `2^k_b` `eq` table (Section 10 item 4, now the only non-succinct cost of
the Keccak side) and the opening is succinct. Proof: one sponge-only proof plus one bridge
proof per part, independent of `n` up to the sumcheck round counts.

**Wire.** The event-2 item of a part is `tied::TiedCommitment` (both caps, each framed by
its length); the event-6 proof is `AGGKCT || version || profile || n || d || framed sponge
proof || framed bridge proof`, each inner proof the bincode of Flock's
`R1csProofMergedLigerito` followed by its claim values (`flock_union::UnionProof`). The
segment items of `agg_cli::wire` carry them unchanged (item version 2). This is proof
version 1; since version 2 the event-2 item holds the sponge cap alone and the bridge
union's cap is inside the proof (Section 5b, item version 3).

## 5b. The bridge union: the factored bridge instance of a tied part

Status: normative for M3 as implemented in `crates/agg-keccak/src/bridge_union.rs`,
`flock_union.rs` (multi-type unions) and `tied.rs` (tied-part proof version 2; deviations
K13, K14). It replaces the one-row bridge slot as the bridge instance of the tied part of
Section 5a. Unchanged from Section 5a: the sponge instance, its three claims and the public
table check, the lane groups, the part seeds, the shares and their XOR (K10). Section 5
remains the specification of the one-row slot (the reference implementation `bridge_slot.rs`,
`segment.rs`).

**Why** (`docs/keccak.md` Section 10). Every cost of the one-row slot is linear in its
`2^(13 + d)` columns: the walker reads `7.2 x 10^8` matrix entries per fold at a full
segment, its `eq`, `xi` and comb vectors are 1 GiB each, and the verifier's lincheck
discharge, the only non-succinct Flock cost, measured 0.26 / 0.50 / 1.43 s at `d = 10 / 11 /
13`. Smaller bridge parts do not help (the walker entries and vector lengths of the parts add
up to the same totals, and each part adds about 300 KB of proof). What helps is a layout whose
per-block matrices are small: Flock's lincheck is per block (`A = I (x) A_0`), and level 1 of
the composite applies the same `M_1` to every (lane, level-1 block); the bridge relation is
therefore split over two Flock types of one union, with the level-1 parities `y1`, the
composite's intermediate vector, committed between them.

**Types** (production shape; `L = min(2^(10 + d), J_1)` positions per level-1 block,
`blocks = 2^max(0, d - 9)` blocks per lane, `G = blocks * m_1` parities per lane; local
in-block index `zeta`, `M_1` column `zeta1_base + zeta` with `zeta1_base = offset mod J_1`, `M_2`
column `rho_1 + m_1 (first_block + beta)`, as in `agg_bridge::gf2_walker`):

| Type | Rows (count) | Columns | Rows of `A_0`, `B_0` (`C = I`) |
| --- | --- | --- | --- |
| level 1 (`Level1Layout`, `Level1Slot`) | row `j = beta + blocks t` for lane `t < 6`, block `beta` (`6 blocks` live; rows of lanes 6, 7 and beyond are dummy) | `[0, L)` the block's input bits; `L` the constant; `y1_base + rho_1`, `y1_base = L + 8192`, the parities (64 aligned words); `useful_bits = L + 16,384` | inputs `z_c * one = z_c` on all `L` columns (the same in every row: padded positions are tied to the sponge's zeros, not forced by rows); `one * one = one`; `(sum_zeta M_1[rho_1, zeta1_base + zeta] z_zeta) * one = z_{y1(rho_1)}`; empty rows elsewhere |
| levels 2-3 (`TopLayout`, `TopSlot`) | one live row | `c(t, beta, rho_1) = rho_1 + 8192 beta + G t`, `t < 6` (the level-1 rows' parities concatenated in row order); `6 G` the constant; `y_o = 6 G + 4096 + o`, `o < 4096`, the output columns (32 aligned words); `useful_bits = 6 G + 8192` | inputs `z_c * one = z_c`; `one * one = one`; `(sum_c (M_3 M_2)[o, c] z_c + Y_o one) * one = z_{y_o}` with `(M_3 M_2)[o, (t, beta, rho_1)] = sum_{rho_2} M_3[o, t + 6 rho_2] M_2[rho_2, rho_1 + 8192 (first_block + beta)]` and the share `Y` a coefficient (K8); empty rows elsewhere |

Both types carry empty matrix stubs and a constant pin (K2); the relations are bound by
`Level1Slot::digest` (label `agg-keccak/v2/bridge-level1`, `sd`, the shape, `lane_len`, offset,
blocks, `L`, `k_log`, `useful_bits`, constant column, parity base) and `TopSlot::digest`
(label `agg-keccak/v2/bridge-top`, `sd`, the shape, `lane_len`, offset, blocks, `G`, `k_log`,
`useful_bits`, constant column, output base, `Y`). Walkers (`LincheckCircuit`; `fold_split`
and `fold_alpha_batched` from one pass):

```text
level 1:     v[rho_1] = eq[y1_base + rho_1]
             xi_A[c]  = eq[c] + sum_{rho_1 in S_1(zeta1_base + c)} v[rho_1]        (c < L)
             xi_A[one] = eq[one];  xi_B[one] = eq[one] + sum_{c < L} eq[c] + sum_rho1 v[rho_1]
levels 2-3:  v3[o] = eq[y_o];  v2[t + 6 rho_2] = sum_{o in S_3(t + 6 rho_2)} v3[o]
             xi_A[c(t, beta, rho_1)] = eq[c] + sum_{rho_2 in S_2(rho_1 + 8192 (first + beta))} v2[t + 6 rho_2]
             xi_A[one] = eq[one] + sum_o Y_o v3[o];  xi_B[one] = eq[one] + sum_{c < 6G} eq[c] + sum_o v3[o]
```

Per fold the two walkers read `14 L + 14 * 6 G + 18 * 49,152` matrix entries (`1.9 x 10^7`
at a full segment, against `7.2 x 10^8` for the one-row slot), over `eq` tables of
`2^k_log` entries (`2^20`, 16 MiB, at a full segment).

**Union.** `Registry::new([level 1, levels 2-3], nu)` with counts `[6 blocks, 1]` and the
uniform row capacity `nu = max(3 + log2(blocks), 3, 21 - max k_log)`, so that `M = nu + max
k_log + 1 >= 22` (Flock's dense floor). Flock sorts the slots by `k_log` descending (stable):
at `d <= 4` the levels-2-3 type is slot 0, from `d = 5` on level 1 is (at `d = 5` and `d = 13`
both have the same `k_log` and level 1 stays first); the code looks the slots up
(`flock_union::UnionGeometry::slot`). The level-1 type is never full height (6 of every 8
rows), so Flock's block-first opening is never selected. Exact sizes (`BridgeUnion::cost`;
"one-row slot" is Section 5's instance at the same `d`):

| `d` | blocks | `k_log` level 1 / levels 2-3 | `nu` | `M` | live rows | committed words | dense `m` | walker entries per fold | one-row slot: `k_b`, `M_b`, words, walker entries |
| ---: | ---: | --- | ---: | ---: | --- | ---: | ---: | ---: | --- |
| 3 | 1 | 15 / 16 | 5 | 22 | 6 + 1 | 1,600 | 22 | 1,687,552 | 16, 22, 417, 2,260,992 |
| 6 | 1 | 17 / 16 | 4 | 22 | 6 + 1 | 4,288 | 22 | 2,490,368 | 19, 22, 3,105, 7,077,888 |
| 8 | 1 | 19 / 16 | 3 | 23 | 6 + 1 | 13,504 | 22 | 5,242,880 | 21, 24, 12,321, 23,592,960 |
| 9 | 1 | 20 / 16 | 3 | 24 | 6 + 1 | 25,792 | 22 | 8,912,896 | 22, 25, 24,609, 45,613,056 |
| 10 | 2 | 20 / 17 | 4 | 25 | 12 + 1 | 51,520 | 23 | 9,601,024 | 23, 26, 49,185, 90,341,376 |
| 11 | 4 | 20 / 18 | 5 | 26 | 24 + 1 | 102,976 | 24 | 10,977,280 | 24, 27, 98,337, 179,798,016 |
| 13 | 16 | 20 / 20 | 7 | 28 | 96 + 1 | 411,712 | 26 | 19,234,816 | 26, 29, 393,249, 716,537,856 |

**Commitment order (the change against the design note of `docs/keccak.md` Section 10).** The
parities `y1 = M_1 b'` depend on `sd`, so the union cannot be committed at event 2 (the design
note of Section 10 had it there, which is impossible). The part's event-2 item is the sponge
instance's cap alone (`TiedCommitment`); the bridge union is built, committed and proven at
event 6 (`TiedPart::prove_with_blocks`): blocks of the part's bits under the matrices of
`sd`, then its Ligerito cap, which the proof carries (framed, before the two inner proofs).
The tie challenges are derived after both caps: `TieChallenges::derive` absorbs, under
`agg-mldsa/v2/tie`, `fs`, `d`, `nu_s`, `log2(blocks)`, the sponge cap and the bridge union's
cap (each length-framed), then squeezes `rho_w` (3), `rho_i` (`d`), `rho_t` (2), `rho_t'` (1),
`rho_r` (`nu_s`), `rho_c` (3), `rho_beta` (`log2(blocks)`), `rho_y` (6), `rho_o` (5). The
bridge union's challenger: `agg-mldsa/v1/keccak`, `fs`, the label `agg-keccak/v2/tied-bridge-
union`, both digests, the tie-geometry digest, then Flock's statement binding (registry
digest, counts, cap). Committing after `sd` costs nothing in the bridge bound: everything
the union holds is forced by `b'`, which the sponge cap fixed at event 2 (below).

**Claims** (packed-direct claims of the union's merged opening; `lane_bits(h)` is `[rho_t0,
rho_t1, 0]` for lanes 0..3 and `[rho_t', 0, 1]` for lanes 4, 5; record `i = i_lo + 2^9 beta`,
`lo = min(d, 9)`; each point is `[rows (nu) | the type's word columns (k_log - 7) | its slot
prefix as 0/1 constants]`, `UnionGeometry::region_point`):

| Claim | Rows | Word columns | Verifier's check |
| --- | --- | --- | --- |
| `v_in_h` (level-1 inputs) | `[rho_i[lo..d] (beta) / lane_bits(h) / 0]` | `[rho_w / rho_i[0..lo] / 0]` | `v_in_h = eq(rho_i[nu_s..d], 0) v_s_h` (tie 1, against the sponge's `P_s_h` of Section 5a) |
| `v_y1_h` (level-1 parities) | `[rho_beta / lane_bits(h) / 0]` | `[rho_y / bits of (L / 8192 + 1)]` | `v_top_h = v_y1_h` (tie 2) |
| `v_top_h` (levels-2-3 inputs) | 0 | `[rho_y / rho_beta / lane_bits(h) / 0]` | as above |
| `v_out` (output columns) | 0 | `[rho_o / bits of (6 G / 4096 + 1)]` | `v_out = 0` |

Error names (`KeccakError::Tie`): `bridge copy of lanes 0..3`, `bridge copy of lanes 4, 5`
(tie 1, the names of Section 5a), `level-1 parities of lanes 0..3`, `level-1 parities of
lanes 4, 5` (tie 2), `levels-2-3 output columns`, and `public words` (Section 5a).

**The output columns need their own claim.** A row with `C = I` only defines its output
column: the levels-2-3 rows say `z_{y_o} = (M_3 M_2 z_in)_o + Y_o`, which holds for any `Y` if
the prover chooses `z_y` accordingly. The claim `v_out = 0` pins `z_y = 0`, so the rows say
`M_3 M_2 z_in = Y`. (The one-row slot of Section 5 has the same rows without such a claim. It
is sound because its block, the `y_o` columns included, is committed at event 2, before `sd`
and `Y`: the rows then give `Y = M b' + z_y` with `z_y` fixed before `sd`, the F_p side gives
`Y = M b`, so `M delta = z_y` for `(delta, z_y)` fixed before `sd`; `delta = 0` forces `z_y = 0`,
and for `delta != 0` either `M_2 M_1 delta = 0` (bounded by Section 7) or `Pr[M_3 y_2 = z_y] <=
1 / binom(m_3, k_3) = 2^-163.4` for the fixed nonzero `y_2`, conditioning on all but one of
its involved `M_3` columns. A union committed after `sd` has no such protection, hence the
claim.)

**Soundness.** Flock's knowledge soundness gives the committed word vectors `z_s` (sponge,
event 2) and `z_u` (the union, absorbed before the tie challenges) satisfying their rows and
the claimed evaluations. In `z_u`: the constant is 1 on the live rows and 0 on the dummy rows
(Flock's count-derived pin); every live level-1 row has `y1 = M_1 x(t, beta)` for its inputs
`x` (its parity rows); the levels-2-3 row has `Y = M_3 M_2 z_in` (its composite rows and
`v_out = 0`, error `5 / 2^128`). Tie 1: `f_in(rho)` and `f_s(rho)` are the multilinear
extensions, in the same `d + 5` free coordinates `(i, w, t)` of a lane group, of the level-1
inputs and of the sponge inputs over the part's `2^d` record positions (the sponge's dummy rows
are zero: their constant is 0, so their input rows force zero; the level-1 positions of
records beyond the live ones are compared with those zeros); a differing bit survives with
probability at most `(d + 5) / 2^128` per lane group. Tie 2: `f_y1(rho)` and `f_top(rho)` are
the extensions, in the `log2(blocks) + 9` free coordinates `(beta, t, rho_1 word)`, of the
level-1 parities and of their copies; error `(log2(blocks) + 9) / 2^128` per lane group. The
lane groups cover lanes 0 to 5 and every live level-1 row; rows of lanes 6, 7 are dummy and
their would-be copies are the constant and output columns, which the groups never address.
Hence `Y_share = M_3 M_2 M_1 b'_part` over GF(2) with the segment's matrices, where `b'` is the
sponge's input, fixed at event 2 before `sd`; the shares XOR to `Y_sigma` (K10) and the bound
(9.3) of Section 7 is unchanged. Per part the tie error is `2 (d + 5) / 2^128 + 2
(log2(blocks) + 9) / 2^128 + 5 / 2^128 + (nu_s + 3) / 2^128` (the last term the public claim of
Section 5a), `83 / 2^128 = 2^-121.6` at a full segment, inside `epsilon_Flock`. The challenges are
Fiat-Shamir outputs of a transcript that contains both caps (and, through `fs`, events 0 to 5),
so a prover that re-derives them `Q` times multiplies these terms by `Q` (the usual ROM loss,
as for Flock's own challenges). Replay: the two instances have distinct challenger labels and
digests (sponge: gate and public digests; union: both relation digests with `sd`, the part's
offset and `Y`), and the tie-geometry digest covers `d`, the part index, `n`, both row
capacities, word coordinates, block widths and blocks.

**Witness and costs.** The part's lane-major input bits (`TiedPart::honest_parts`, the same
vector the mode-A hooks tamper) are the only `sd`-free bridge data; at event 6 the prover
forms the level-1 rows and their parities (`Level1Layout::parities`: the six lanes of a
position as one byte mask, 14 byte-XORs per nonzero position, parallel over blocks and
position chunks), the levels-2-3 block and the share (`TopLayout::parities`), packs both types
(`3 x 2^(nu + k_log - 7)` words each), assembles the union buffers (`3 x 2^(M - 7)` words, 32
MiB each at a full segment), commits once and proves. The row values `a` of the level-1 rows
are recomputed from the committed rows' own inputs, so a tampered parity breaks its row.
Measurements: `docs/keccak.md` Section 11 and `ref/notes/bench-local.md`.

**Wire.** Event 2: per part `TiedCommitment` = `LE32(len) || sponge cap`. Event 6: per part
`AGGKCT || 2 || profile || LE32(n) || d || LE32(len) || bridge union cap || LE32(len) ||
sponge proof || LE32(len) || bridge union proof`, each inner proof a `flock_union::UnionProof`
(bincode of Flock's `R1csProofMergedLigerito`, then its claim values: 3 for the sponge, 7 for
the union). The segment items of `agg_cli::wire` frame them unchanged (item version 3).

**Tests.** `bridge_union` (unit): layouts and union geometry at the toy shape and `d = 3, 8,
9, 10, 13`; honest rows satisfy `a & b = z`, and a flipped parity, input, zero column, `Y`
bit, output column or visible copy breaks a row; both walkers are adjoint to their row
evaluators (toy, and a production part of `d = 10` at index 1: two blocks and an offset) and
`fold_split` equals two alpha folds; both walkers equal Flock's `SparseMatrixCircuit` on the
explicitly lowered rows (both types on the toy shape, level 1 of a production part of `d = 3`
at index 5, whose `zeta1_base` is nonzero); the digests bind `sd`, the layout and `Y`; the
honest share equals `gf2_walker::composite` and the region points evaluate their regions; a
Flock proof of the toy union (two boolean types, `M = 22`) verifies, the precomputed
commitment gives the same proof, and a wrong share or a broken parity row is rejected.
`tied` (unit): the tie identities hold on honest regions at `(n, d) = (5, 3), (9, 6), (3, 5),
(3, 10)` and a flipped input or copy breaks exactly its lane group; the points' shape.
`tests/tied.rs` (release): round trips at 5, 64, 256, 512 and 1,024 records with `Y` equal to
the F_p walk; a flipped `Y` bit, another `sd`, a wrong `c~` or `mu`, another `fs`, flipped proof
bytes (in the bridge cap, the sponge proof and the union's proof) and the union's cap in place
of the event-2 cap are rejected; at 64 records (and, for the block-dependent cases, at 1,024
records in the second level-1 block) a level-1 input copy flipped (tie 1, `bridge copy of
lanes 0..3`), a parity copy flipped (tie 2, `level-1 parities of lanes 4, 5`), a level-1 parity
flipped in its row (Flock), a stale share against tampered copies (Flock), a wrong share with
the output columns set to absorb it (`levels-2-3 output columns`), both copies flipped with a
stale `c~` (`public words`) and a sponge input with a stale chi witness (Flock); the proof made
with the kept commit data equals the one made by committing again; a segment of 20 records in
three parts with kept commit data whose shares XOR to the segment's `Y`. The mode-A tests of
`agg-cli` run unchanged on the new part (the hook that flips a bit of the part's input bits
fails tie 1).

## 5c. Spans: one tied part over several segments; the final vectors as the output claim

Status: normative for tied-part proof version 3 (`crates/agg-keccak/src/bridge_union.rs`,
`tied.rs`; deviation K16; mode A: `agg_cli::mode_a`, segment items of version 4). It changes
two things of Section 5b and keeps everything else (types, rows, walkers, the sponge
instance, the lane groups, the claim count 3 + 7, the error names).

**The output claim carries `Y`.** In every tied part the levels-2-3 rows are proven with a
zero coefficient (`TopSlot` with `Y = 0`), so they say `z_{y_o} = (M_3 M_2 z_in)_o`, and the
honest prover writes `M_3 M_2 y1` (the share) into the output columns
(`BridgeUnion::fill_outputs`). The output claim `v_out` at `P_out = [rows rho_sigma | words
[rho_o | the output region's base]]` is checked against `Yhat(rho_o, rho_sigma) = sum_sigma
eq(rho_sigma, sigma) sum_{w < 32} eq(rho_o, w) Y_sigma[w]`, the multilinear extension of the
public table of final vectors in the word layout of the output columns (word `w` of `Y_sigma`
is bits `128 w .. 128 w + 127`, low 64 bits first; `Y_sigma = 0` for the segments `S..2^span_log`
of the union), which the verifier computes (`TiedPart::output_value`). The table's digest
`SHA3-256(len || "agg-keccak/v3/tied-final-vectors" || LE64(S) || packed Y_0 .. Y_{S-1})` is
absorbed by the tie challenges (with `span_log`) before `rho`. For a part of one segment the
table is its share (K10); for a span it is `(Y_0, .., Y_{S-1})`.

**Spans.** A tied part may cover `S` consecutive segments that share its part geometry
(`d`, part index `p`): `TiedPart::new_span(gate, n, d, p, S, profile)`, `n <= 2^d S` records in
segment order (record `i` of segment `sigma` is record `2^d sigma + i` of the span, so only
the last segments may be partial or empty). Mode A uses it for every batch of more than one
segment in whole-segment parts (`d = 13`, `p = 0`, `S` = all segments of the batch, empty
trailing ones included). With `span_log = ceil(log2 S)`:

| Instance | Rows | Change against Section 5b |
| --- | --- | --- |
| sponge | the span's `n` records, row `2^d sigma + i` | `nu_s = capacity_log(n, 19)` over all records |
| bridge union, level 1 | `sigma + 2^span_log (beta + blocks t)`, `2^span_log 6 blocks` rows, all live | the same layout and `M_1` per row (`zeta1_base` is the part's); segments `>= S` are zero rows |
| bridge union, levels 2-3 | row `sigma`, `2^span_log` rows, all live | the same layout and `M_3 M_2` per row (`first_block` is the part's) |

The union's row capacity is `nu = max(span_log + 3 + log2(blocks), 3, 21 - max k_log)`.
`TieChallenges::derive` absorbs `span_log` and the final-vector digest and squeezes
`rho_sigma` (`span_log` coordinates) after the coordinates of Section 5b. The points: the
sponge rows are the first `nu_s` of the record coordinates `rho_i || rho_sigma` (zero above);
the level-1 points (inputs, parities) have rows `[rho_sigma | beta | lane_bits(h)]`; the
levels-2-3 points (copies, outputs) have rows `[rho_sigma]`. The tie-1 factor is
`eq(rho[nu_s..], 0)` over `rho = rho_i || rho_sigma`. Proof bytes (version 3): `AGGKCT || 3 ||
profile || LE32(n) || d || S || LE32(len) || bridge union cap || LE32(len) || sponge proof ||
LE32(len) || bridge union proof`, the inner proofs without the Ligerito level-0 cap (K17).

**Soundness.** As Section 5b with the segment index as a free coordinate. Flock's knowledge
soundness gives `z_s` (event 2) and `z_u` (absorbed before the tie challenges) satisfying
their rows and the claimed values. In `z_u` every live level-1 row has `y1 = M_1 x(sigma, t,
beta)` and every levels-2-3 row `sigma` has `z_y(sigma) = M_3 M_2 z_in(sigma)`. Tie 1: `f_in`
and `f_s` are the multilinear extensions, in the `d + span_log + 5` free coordinates `(i,
sigma, w, t)` of a lane group, of the level-1 inputs and of the sponge inputs over the span's
`2^(d + span_log)` record positions (the sponge rows `>= n` are zero, so the level-1 inputs of
missing records and segments are compared with zeros); a differing bit survives with
probability at most `(d + span_log + 5) / 2^128`. Tie 2: `(log2(blocks) + span_log + 9) /
2^128` per lane group, over `(beta, sigma, t, word)`. Outputs: `z_y` and the table `Y` are two
fixed word vectors (the table by its digest, absorbed before `rho`; `z_y` by the union's cap);
if they differ, `v_out = Yhat(rho)` holds with probability at most `(5 + span_log) / 2^128`.
Hence `Y_sigma = M_3 M_2 M_1 b'_sigma` over GF(2) for every segment `sigma` of the span, with
the matrices of `sd` and `b'` the sponge's input fixed at event 2 before `sd`; for a part of
one segment the shares XOR to `Y_sigma` (K10) as before. The bound (9.3) of Section 7 is
unchanged; the tie error per tied part is `2 (d + span_log + 5) + 2 (log2(blocks) + span_log +
9) + (5 + span_log) + (nu_s + 3)` over `2^128`: `95 / 2^128 = 2^-121.4` for a span of four
full segments (`d = 13`, 16 blocks, `span_log = 2`, `nu_s = 15`), inside `epsilon_Flock`, with
the usual factor `Q` for re-derived challenges (Section 5b). Replay: the tie-geometry digest
covers `S` and `span_log`, the proof header `S`; the span's Flock seed in mode A is
`SHA3-256(len || "agg-mldsa/v1/flock-span-seed" || LE64(S) || fs_0 || .. || fs_{S-1})`, a
function of event 5 that differs from every part seed.

**Costs.** A span's sponge instance and bridge union are the per-segment instances stacked:
the same rows, walker entries and committed words in total, one commitment and one opening
each instead of `S` of each; the Ligerito openings grow by about one level per doubling of
the committed words (`docs/keccak.md` Section 12).

**Tests.** `bridge_union`: a span of three segments at `d = 3` (a fourth union segment
empty) and of two at `d = 10` (two blocks) has the documented rows and capacity, its shares
are the segments' composites (zero for the empty segment), segment 1's rows are the
single-segment rows of its bits, and the region points at a Boolean segment coordinate
evaluate that segment's regions. `tied` (unit): the tie identities, including the output value
against the table, on honest regions for spans `(n, d, S) = (12, 3, 2), (19, 3, 3), (70, 6, 2)`
and the parts of Section 5b, a flipped input or parity copy of the last segment breaks
exactly its lane group, the points of a span of three full segments. `tests/tied.rs`: spans
of `(19, 3, 3)`, `(16, 3, 3)` (last segment empty) and `(70, 6, 2)` round-trip with each
`Y_sigma` equal to the F_p side's walk of its segment; a flipped bit of `Y_1`, `Y_0` and `Y_1`
swapped, a missing `Y`, a wrong share with honest rows (`levels-2-3 output columns`), an input
copy flipped in segment 1 only (`bridge copy of lanes 0..3`), another segment count and another
record count are rejected; the tampering cases of Section 5b run on version 3 (a wrong share
with honest rows fails the output claim, the output columns set to it break the rows).

## 6. The final public vector

$`Y_\sigma\in\{0,1\}^{4096}`$ per segment, 512 bytes, bit $`\rho_3`$ at byte $`\rho_3\gg3`$, bit position $`\rho_3\bmod8`$ (least significant first, deviation P6), absorbed at event 6 before the Flock proofs and before $`\tau`$. The same bytes are: the `Out` words of the bridge slot (public words of the Flock statement), the public table $`Y(\rho_3,\sigma)`$ of the PIOP whose extension $`Y(\tau_3)`$ the verifier evaluates in step 4 of spec Section 11, and the right-hand side of the level-3 identity. Proof cost $`512S`$ bytes (2 KB at $`N=2^{15}`$). A verifier MUST reject a proof whose event-6 item has a length other than $`512S`$.

## 7. Soundness

Let $`\delta=b\oplus b'`$ over all lanes, segments and positions of the main domain (padded positions included; they are fixed zeros on the Flock side). $`\delta`$ is determined by events 1 and 2, before $`\mathsf{sd}`$. If both sides accept with the same $`Y_\sigma`$, then by Section 4 (lifting, Booleanity, carry ranges) and Section 5 (GF(2) rows), $`M_3M_2M_1\delta=0`$ in every segment. From `ref/notes/improvements-bridge.md`: Lemma 1 (a block with exactly one nonzero input has a nonzero image, since every column has weight $`k_t\ge1`$), Lemma 2 (for a fixed nonzero block input, $`\Pr[M_tv=0]\le1/\binom{m_t}{k_t}`$, with equality at weight 2: condition on all but one of the involved columns, the remaining uniform $`k_t`$-subset must equal a fixed set), Lemma 3 (a level with the same $`M_t`$ on every block is missed only if every nonzero block is missed, so no union bound over blocks, lanes or segments is needed) and the composition theorem (the first level that zeroes a nonzero input is a disjoint-event decomposition; $`d^{(t-1)}`$ is fixed and independent of $`M_t`$ because the three levels use distinct hash inputs):

```math
\Pr_{\mathsf{sd}}\big[M_3M_2M_1\delta=0\big]\ \le\ \sum_{t=1}^{3}\frac1{\binom{m_t}{k_t}}
=\frac1{\binom{8192}{14}}+\frac1{\binom{8192}{14}}+\frac1{\binom{4096}{18}}
=2^{-145.64}+2^{-145.64}+2^{-163.44}=2^{-144.64}.
```

The bound is independent of $`J_t`$, of the row weights and of the number of blocks; a difference of weight 1 is caught with probability 1. The bridge term of the knowledge-soundness ledger (spec 12.1, equation (9.3)) is

```math
\varepsilon_{\rm bridge}\le2^{-144.64}+\varepsilon_{\rm PIOP}+\varepsilon_{\rm Flock}+\varepsilon_{\rm bind},
```

where $`\varepsilon_{\rm PIOP}`$ collects the bridge instances' field terms (12 Booleanity zero-checks and 7 linear identities at $`\tau`$, 6 level-1 identities at $`r^{(1)}_1`$, the carry-digit leaves and links, 50 reductions: each at most $`d_{\rm main}/|K|`$, spec 12.1), $`\varepsilon_{\rm Flock}`$ is the sub-proof's knowledge error per profile and $`\varepsilon_{\rm bind}`$ the Akita and Ligerito (or LaBinius) binding errors. Grinding: an adversary who re-derives $`\mathsf{sd}`$ $`Q`$ times (fresh events 1 or 2) multiplies the first term by $`Q`$; the 16.6-bit margin above $`2^{-128}`$ allows $`Q\le2^{16}`$ without a proof-of-work nonce, and the seed site MAY be ground under Aerie's schedule (deviation P3: not yet). Extraction: the Akita extractor gives $`b`$, the Flock extractor gives $`b'`$ with $`\tilde c_i=\mathrm{SHAKE256}(\mu_i\Vert b'_i)`$; the bridge identifies them, so the Keccak statement holds for the bits that satisfy the ring identity (spec 12.2). For the toy shape the same formula gives $`2^{-10.75}`$, and the spec's Section 13 test measures $`1/\binom{64}{3}=1/41{,}664`$ empirically at $`(m,k)=(64,3)`$ on weight-2 differences.

## 8. Worked example (one lane, two levels, $`L=16`$)

Shape $`(J,m,k)=(8,4,2)`$ at both levels, two level-1 blocks, one level-2 block, 4 output bits. Columns are the row sets

```text
M_1: S_0..S_7 = {0,1} {0,2} {0,3} {1,2} {1,3} {2,3} {0,1} {1,2}      row weights 4,5,4,3
M_2: S_0..S_7 = {0,1} {2,3} {0,2} {1,3} {0,3} {1,2} {0,1} {2,3}      row weights 4,4,4,4
b = block 0: 1 0 1 1 0 0 1 0   block 1: 0 1 1 0 1 0 0 0
```

Level 1, integer sums per block and row: block 0 $`(3,3,1,1)`$, block 1 $`(2,1,1,2)`$; parities $`y^{(1)}=(1,1,1,1\mid0,1,1,0)`$, carries $`\kappa^{(1)}=(1,1,0,0\mid1,0,0,1)`$. Level 2 on the 8 parities: sums $`(3,4,3,2)`$, $`Y=(1,0,1,0)`$, $`\kappa^{(2)}=(1,2,1,1)`$; the GF(2) product $`M_2M_1b`$ gives the same $`Y`$. Weight check with the integer vector $`\lambda=(1,2,4,8)`$ in place of $`\mathrm{eq}(\pi_\rho,\cdot)`$: $`c_j=\sum_{\rho\in S_j}\lambda_\rho`$ is $`(3,5,9,6,10,12,3,6)`$ for $`M_1`$ and $`(3,12,5,10,9,6,3,12)`$ for $`M_2`$; block 0 of level 1 satisfies $`\sum_\rho\lambda_\rho(y_\rho+2\kappa_\rho)=21=\sum_jc_jb_j`$, and level 2 satisfies $`39=39`$. Carry digits in the production bases: $`\kappa=2=Z_0+16Z_1+256Z_2+648`$ has $`(Z_0,Z_1,Z_2)=(-6,-8,-2)`$: the ordinary base-16, 16, 4 digits $`(2,0,0)`$ of $`\kappa`$ shifted by $`-b_k/2`$ (`docs/witness.md` Section 4), and $`-6-128-512+648=2`$. Tampering: flipping bit 3 of block 1 on either side changes $`M_1e`$ to $`(0,0,0,0\mid0,1,1,0)`$ and $`Y`$ by $`(1,0,1,0)`$, so the two sides disagree on $`Y`$ (Lemma 1). The weight-2 difference at columns 0 and 6 of block 0 has $`S_0=S_6`$ and is invisible to level 1, the event of Lemma 2 with probability $`1/\binom42=1/6`$ per column pair; it is caught by level 2 only if its level-1 image is nonzero, which here it is not. With the production shape this event has probability $`2^{-145.6}`$.

## 9. Test plan

All commands run as `taskset -c 0,1,3-15 cargo test --release -p <crate>`; batch sizes stay at or below 1K signatures and $`2^{14}`$ permutations. Existing tests are listed in `docs/witness.md` Section 9, `docs/piop.md` Section 14 and `docs/keccak.md` Section 5; the bridge-specific ones, present or required, are listed below. Module names are those of the `agg-bridge` skeletons (`seed`, `parity`, `carry`, `gf2_walker`, `soundness`); the function names are required of them.

1. **Seed and matrices** (`agg-bridge::seed`, present): determinism; distinct streams per seed, level and column; column weight exactly $`k_t`$ and no repeated row; the reading rule including a skipped repeat; the known answer for column 0 of each level at $`\mathsf{sd}=0^{32}`$ (cross-checked against an independent cSHAKE256); row-weight caps on derived matrices.
2. **Toy instance, both sides against a direct computation** (`agg-bridge`; (a), (c) and the forward walk of (b) present in `tests/toy.rs`): `Shape::TOY`, $`L=2^{12}`$ bits drawn from a seeded PRNG, matrices from `derive_matrices` generalized to the toy levels. (a) $`\mathbb F_p`$ side: `parity::sums`, parities and carries per level against a dense $`\{0,1\}`$ matrix product computed with `i64` arithmetic; every carry within its range; the three identities (9.2) at every Boolean point; `weights::c_t` at ten random $`K`$ points against $`M_t^{\mathsf T}\mathrm{eq}(\pi_\rho,\cdot)`$ formed densely, and the verifier's terminal formula against the full inner product $`\sum_xw_t(x)v(x)`$ reduced to the multilinear extension. (b) GF(2) side: `gf2_walker::composite` (the forward XOR walk) and the lincheck walker's $`\xi_A`$ against the dense composite $`M_3M_2M_1`$ (16 rows by 4,096 columns) and its transpose applied to a random $`\mathbb F_{2^{128}}`$ vector. (c) Agreement: the $`\mathbb F_p`$ parities reduced modulo 2 equal the GF(2) outputs at every level and $`Y`$ agrees.
3. **Single-bit tampering on either side** (toy part present: `single_bit_tampering_on_either_side_changes_y` in `agg-bridge/tests/toy.rs`; deterministic by Lemma 1 at level 1): for 64 random positions, flip bit $`x`$ in the $`\mathbb F_p`$ copy only, recompute honest parities and carries from the tampered bits, and check that $`Y`$ differs from the GF(2) side's; then flip it in the GF(2) copy only and check the same; then flip it in both and check agreement (the bridge compares, it does not verify the bits). On the production shape the same test runs through `agg_piop::mode_b` with `FixedMatrices` at $`N=64`$: a changed $`Y_\sigma`$ bit, a flipped committed bit with stale carries, and a flipped public bit each fail (present: `every_stage_tampering_is_rejected_end_to_end` in `agg-piop/tests/mode_b.rs`, `bridge_mutations_are_rejected` in `agg-witness/tests/negative.rs`).
4. **Weight-2 survival rate** (spec Section 13, present: `agg-bridge/tests/production.rs`, columns drawn by the seed derivation, 51 equal pairs): at $`(m,k)=(64,3)`$, $`2\times10^6`$ random column pairs from a fixed PRNG seed; the number with equal row sets (expected $`2\times10^6/\binom{64}{3}=48.0`$) lies in $`[27,69]`$, three standard deviations.
5. **Lifting at the range boundary** (present in `agg-witness` and, for both shapes, in `agg-bridge/tests/toy.rs`): rows of weight exactly $`W_t`$ with all-one inputs give carries $`1023`$ and $`255`$ and the digits $`(7,7,1)`$ and $`(7,7)`$; weight $`W_t+1`$ makes `compute` return `RowWeight`.
6. **Flock bridge slot** (`agg-keccak`, required at M3): a two-type union at $`d_{\rm seg}=3`$ and $`7`$ ($`k_{\log}`$ 16 and 20) proves and verifies with honest records; rejects a flipped $`Y_\sigma`$ bit, a bit flipped in the slot copy but not in the sponge (copy constraint), a bit flipped in both with stale $`\tilde c`$ (sponge), another seed's matrices (digest), and a stale parity row; the walker equals Flock's `SparseMatrixCircuit` fold on the explicitly lowered rows at $`d_{\rm seg}=3`$ (16 records, $`2^{16}`$ columns, dense composite built in GF(2)); `fold_split` agrees with two `fold_alpha_batched` calls. $`k_{\log}=26`$ (a full segment) is measured on the cloud machine only: prove and verify time, peak memory, proof size.
7. **End to end** (present for mode B and for the Fp side with seed-derived matrices in `agg-piop/tests/bridge.rs`; the joint F_p and Flock run in one transcript at $`N=64`$ and $`256`$, with both final vectors against a direct computation and a single-bit flip on either side failing exactly that side, in `agg-piop/tests/joint_bridge.rs`, deviation P11; required for mode A): honest aggregates at $`N=256`$ and $`1{,}024`$ verify; a single w1Encode bit changed in one copy is rejected; the verifier's bridge-weight evaluation agrees with the prover's weight tables at the terminal points.

## 10. Decisions and open items

1. Composite rows in one bridge slot per segment, parities not materialized on the Flock side (spec B.13). Alternative kept in reserve: reveal $`y^{(2)}`$ (6 KB per segment) and drop level 3; it saves nothing in walker work, since level 1 dominates.
2. Padded records are forced to zero by empty rows, not left free; the parity rows range over all record positions of the slot so that the composite relation depends on $`\mathsf{sd}`$ and $`d_{\rm seg}`$ only, and the walker is count-independent except for the identity rows.
3. Matrices are derived once and shared, not regenerated inside the fold (one Keccak-f per column would cost 1 s per fold on one core).
4. The verifier's bridge-slot fold ($`7\times10^8`$ additions and a $`2^{26}`$ `eq` table per segment) was the largest verifier cost of the Keccak side: measured 1.43 s per segment on the cloud machine and 1.87 s locally at the full-segment geometry. Resolved by the bridge union of Section 5b (walkers of $`1.9\times10^7`$ entries over $`2^{20}`$-entry tables): about 70 ms per segment locally (`docs/keccak.md` Section 11). Remaining candidates if needed: a smaller $`k_1`$ (12: $`2^{-136.5}`$ per level, 14% fewer nonzeros) or a Flock entry that delegates the fold to the prover with a sum-check, which does not exist at `83f963e`.
5. Derivation cost 0.94 s single-threaded (P4) and $`J_2=2^{19}`$ padding: candidates for wire version 1 (Section 3), not adopted.
