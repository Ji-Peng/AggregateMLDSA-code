# Parity bridge: implementation specification (spec Section 9)

| Field | Value |
| --- | --- |
| Status | normative for the F_p side (`agg-bridge`, `agg-piop`) and the Flock side (`agg-keccak`); resolves the open items of `docs/spec.md` Sections 8, 9 |
| Source of truth | `docs/spec.md` (Sections 1, 4, 5, 7, 8, 9, 10). This document only fixes what the spec leaves open; a conflict is a bug here. Deviations go to `docs/deviations.md` |
| Code | `crates/agg-bridge` (`shape`, `seed`, `parity`, `weights`, `final_vector`, `gf2_walker`, `parity_chain`, `soundness`; generic over `Shape`), `crates/agg-witness/src/bridge.rs` (geometry, honest witness), `crates/agg-keccak/src/tied.rs`, `flock_union.rs` (the mode-A Flock side: the sponge instance and the virtual parity chain, Sections 5a, 5c, 5d) |
| Counts | exact numbers are products of the shape constants and exact binomials (`math.comb`); every number marked "estimate" is unmeasured |

Problem. The 6,144 w1Encode bits of every signature are committed twice: as the Boolean columns $`b_0..b_5`$ of $`G_{\rm main}`$ (Akita, $`\mathbb F_p`$) and as the `Input` words of the sponge gate (Flock, GF(2)). The bridge proves $`b=b'`$ with a chain of three seeded sparse parity maps $`M_1,M_2,M_3`$: both sides prove $`Y_\sigma=M_3M_2M_1(\cdot)`$ for the same public 4,096-bit vector per segment, the $`\mathbb F_p`$ side over the integers with committed carries, the Flock side as one sum-check over the committed bits (the virtual parity chain, Section 5d, `docs/parity-chain.md`). Soundness rests only on column independence (Section 7), not on code distance.

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

| Object | $`\mathbb F_p`$ side (Akita) | Flock side (parity-chain input of segment $`\sigma`$) |
| --- | --- | --- |
| bit $`b_t(j,\rho,i)`$ | column $`b_t`$, cell $`x=j+256\rho+1024i`$ | input column $`c=j+256\rho+1024\,i_{\rm seg}+2^{23}t`$, $`i_{\rm seg}=i\bmod2^{13}`$ |
| level-1 in-block index $`\zeta_1=(j,\rho,i_{\rm lo})`$ | $`x\bmod2^{19}`$ | $`c\bmod2^{19}`$ |
| level-1 block | $`\beta_1=(i_{\rm mid},\sigma)=x\gg19`$ (one $`b_t`$ per lane) | $`(i_{\rm mid},t)=c\gg19`$ (lane $`t`$ at $`c\gg23`$) |
| level-1 output $`y^{(1)}_t(\rho_1,\beta_1)`$ | $`D_1`$ cell $`\rho_1+2^{13}\beta_1`$ (virtual, Section 4) | not materialized |
| level-2 in-block index $`\zeta_2=(\rho_1,i_{\rm mid})<2^{17}`$ | $`D_1`$ cell $`\bmod2^{17}`$; block $`\sigma=`$ cell $`\gg17`$ | implicit in the weight $`W`$ |
| level-2 output $`y^{(2)}_t(\rho_2,\sigma)`$ | $`D_2`$ cell $`\rho_2+2^{13}\sigma`$ (committed) | not materialized |
| level-3 in-block index $`\zeta_3=t+6\rho_2`$ | lane-major over $`(t,\rho_2)`$ | implicit |
| final $`Y_\sigma[\rho_3]`$ | public table $`Y`$ on $`D_3`$, cell $`\rho_3+2^{12}\sigma`$ | public, the chain's claim $`\langle\mathrm{eq}_Y,Y\rangle`$ (Section 5d) |

Alignment. The Akita main domain of lane $`t`$ at $`N^\star=2^{d_{\rm sig}}`$ records is $`2^{10+d_{\rm sig}}`$ cells; its $`2^{19}`$-cell prefixes are the level-1 blocks, so $`\mathrm{eq}`$ over the block index is a product of $`d_{\rm sig}-9`$ factors (spec 7.6). On the Flock side the sponge gate of record $`i_{\rm seg}`$ holds lane $`t`$ in the eight aligned 128-bit words $`8t+w`$, $`w<8`$ (bit $`j+256\rho`$ of the lane at word $`w=(j+256\rho)\gg7`$), and the parity chain reads input column $`c`$ as that bit of the sponge's committed words, with no bit permutation (`docs/parity-chain.md` Section 2). When $`d_{\rm seg}=\min(d_{\rm sig},13)<13`$ the input shrinks: the lane stride is $`2^{10+d_{\rm seg}}`$ instead of $`2^{23}`$ and the block stride $`\min(2^{19},2^{10+d_{\rm seg}})`$. Padded records ($`i\ge N`$) are zero on both sides: committed zero bits in $`b_t`$ (spec 4.2) and sponge rows beyond the live records, zero by Flock's witness contract (Section 5a), so the bridge also proves that the $`\mathbb F_p`$ padding is zero. Padded level-1 positions ($`d_{\rm seg}<9`$) and the level-2 positions $`\zeta_2\ge2^{17}`$ are public zeros with zero weight.

## 3. Seed and matrix sampling

**Transcript position** (spec Section 10, `docs/piop.md` Section 9). $`\mathsf{sd}`$ is the 32-byte squeeze of event 3 (`agg-mldsa/v1/bridge/seed`). Before it the transcript has absorbed event 0 (wire version, $`N,N^\star`$, the public-record digest over all $`(\mathsf{pk}_i,M_i,\tilde c_i)`$, the group descriptors and $`(J_t,m_t,k_t)`$), event 1 (the round-1 Akita commitments of $`G_{\rm main16}`$ and $`G_{\rm main4}`$, which contain $`b_0..b_5`$) and event 2 (the $`S`$ Flock commitments, each the Ligerito level-0 cap of the sub-batch whose witness contains $`b'`$, deviation K7). Hence $`\delta=b\oplus b'`$ is fixed by binding commitments before $`\mathsf{sd}`$ exists. Everything seed-dependent comes later: the round-2 groups (the three bridge levels and the lookup multiplicities, event 4), the Flock challenger seeds $`\mathsf{fs}_\sigma`$ (event 5), $`Y_\sigma`$ and the share items (event 6) and the Flock proofs (at the join after stage 3, deviation K22). The Flock side binds $`\mathsf{sd}`$ through $`\mathsf{fs}_\sigma`$ and through the digest of the final vectors, which the tie challenger absorbs (`docs/parity-chain.md`).

**Derivation** (`agg_bridge::seed`, normative as implemented and known-answer tested). Column $`\zeta`$ of $`M_t`$ is read from $`\mathrm{cSHAKE256}(\mathsf{sd}\Vert\mathrm{LE64}(t)\Vert\mathrm{LE64}(\zeta),\cdot,N=\varepsilon,S=\texttt{agg-mldsa/v1/bridge})`$ with $`t\in\{1,2,3\}`$ the level number: take 16-bit little-endian words, keep the low $`\log_2m_t`$ bits (13, 13, 12), skip a value already drawn for this column, stop after $`k_t`$ distinct rows, stored in draw order. This is column-regular ($`|S_\zeta|=k_t`$ exactly), rejection-free on the row value (power-of-two $`m_t`$, a mask, no modular bias) and exactly uniform over $`k_t`$-subsets (the distinct values of an i.i.d. uniform stream form a uniform ordered $`k`$-tuple); the only skipped words are exact repeats, $`k(k-1)/2m\approx0.011`$ per column in expectation. The XOF is squeezed in 64-byte chunks (32 words); the word stream does not depend on the chunking. Both sides, and every segment, use the same three matrices; nothing is transmitted. Representation: `SparseMatrix` (column-major, `cols * k` row indices of `u32`), 62 MB for the production shape, derived once per proof or verification and shared by the $`\mathbb F_p`$ tables, the weights and the Flock walker.

Cost. $`2\cdot2^{19}+49{,}152=1{,}097{,}728`$ columns, one Keccak-f each (48-byte input after the shared customization prefix, first squeeze block suffices): measured 150 ms on 7 threads and 0.94 s on one (deviation P4). Candidate for wire version 1, not adopted: one XOF stream per 64 consecutive columns (0.23 Keccak-f per column) and $`J_2=2^{17}`$ without padding ($`-2^{18}`$ columns); both change event 0 and the known answers.

Completeness guard. The honest prover computes the row weights (`BridgeMatrices::row_weights_ok`) and MUST abort if a row of $`M_1`$ or $`M_2`$ has weight above 2,047 or a row of $`M_3`$ above 511; for the production shape the row weight is $`\mathrm{Bin}(J_t^{\rm used},k_t/m_t)`$ with means 896, 224, 216 and $`\Pr[\text{weight}\ge2048]\lt2^{-700}`$ at level 1 (exact binomial tail; $`\ge1152`$ is $`2^{-52.7}`$), so the abort never fires in practice. Soundness does not depend on it (Section 7).

## 4. The F_p side

Per segment, lane and level the prover proves, over the integers, $`\sum_\zeta M_t[\rho,\zeta]\,v^{(t-1)}(\zeta)=y^{(t)}(\rho)+2\kappa^{(t)}(\rho)`$ with $`v^{(0)}=b_t`$, $`v^{(1)}=y^{(1)}_t`$, $`v^{(2)}=(y^{(2)}_t)_{t<6}`$ and $`y^{(3)}=Y_\sigma`$ (spec (9.2)). What is committed, virtual or public, and where each identity is tested:

| Level | Left-hand quantity | Carry $`\kappa^{(t)}`$ (committed column) | Linear identity | Booleanity | Carry range | Weight point |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | $`y^{(1)}_t`$ virtual on $`D_1`$: $`\sum_\zeta M_1[\rho_1,\zeta]b_t-2\kappa^{(1)}_t`$ (spec Section 5) | $`K^{(1)}_t=\kappa^{(1)}_t-2^{15}`$ on $`D_1`$ (balanced, width 16) | `def_y1[t]` (7.5), stage 2, degree 2, main domain, weight $`w_1`$ | `bool_y1[t]` (7.1), stage 1, degree 3, $`D_1`$ at $`\tau_1`$ | lookup into $`T_{16}`$ (event 7, stack `kappa1`), reduced in stage 3 | $`r^{(1)}_1`$ |
| 2 | $`y^{(2)}_t`$ committed Boolean on $`D_2`$ | $`K^{(2)}_t`$ on $`D_2`$ | `level2[t]` (7.2), stage 1, degree 2, $`D_1`$, weight $`w_2`$ | `bool_y2[t]`, stage 1, degree 3, $`D_2`$ at $`\tau_2`$ | lookup (stack `kappa2`) | $`\tau_2`$ |
| 3 | $`Y_\sigma`$ public; $`Y(\tau_3)`$ computed by the verifier ($`4096S`$ terms) | $`K^{(3)}`$ on $`D_3`$ | `level3` (7.3), stage 1, degree 2, $`D_2`$ summed over the six lanes, weights $`w_{3,t}`$ | none needed | lookup (stack `kappa3`) | $`\tau_3`$ |

Stage 3 also carries the evaluation reductions: $`y^{(2)}_t`$ from $`\tau_2`$ and $`r^{(1)}_2`$, $`K^{(2)}_t`$ from $`\tau_2`$, $`K^{(3)}`$ from $`\tau_3`$, $`K^{(1)}_t`$ from $`r^{(1)}_1`$, and the lookup's stack claims (spec 7.5). The level-1 claim $`y^{(1)}_t(r^{(1)}_1)`$ produced by `bool_y1` is discharged by `def_y1` in stage 2 after the prover sends the 6 values $`K^{(1)}_t(r^{(1)}_1)`$ (spec 7.4); that is the only virtual link in the chain. All degrees are individual degrees of the summand including the $`\mathrm{eq}`$ or weight factor.

The carry range by the lookup. Every carry is stored balanced, $`\kappa=K+2^{15}`$, and looked up in $`T_{16}`$ (`docs/lookup.md`), so $`\kappa\in[0,2^{16})`$ as an integer. Lifting: a row has at most $`W_t`$ ones (the verifier knows the exact row weights), the Boolean inputs make the sum at most $`W_t\lt2^{11}`$, and $`y+2\kappa\le1+2^{17}`$, so both sides of each identity are below $`2^{18}\ll p`$ and the $`\mathbb F_p`$ equation is an integer equation; reducing it modulo 2 gives $`y=\bigoplus_\zeta M_t[\rho,\zeta]v(\zeta)`$. For soundness any carry range below $`(p-W_t)/2`$ would do; completeness needs $`\kappa\le W_t/2\lt2^{16}`$, which holds for $`W_t\le2047`$ (511).

**Weights.** For level $`t`$ at the point $`\pi`$ of $`D_t`$ (split $`\pi=(\pi_\rho,\pi_\beta)`$) the summation weight is $`w_t(\beta,\zeta)=\mathrm{eq}(\pi_\beta,\beta)\,c_t(\zeta)`$ with $`c_t(\zeta)=\sum_{\rho\in S_\zeta}\mathrm{eq}(\pi_\rho,\rho)`$ (spec 7.6; `agg_bridge::weights`, wrapped by `agg_piop::weights` at the production shape). Prover: one $`c_t`$ vector per level, shared by all lanes, blocks and segments. Verifier, at the terminal point $`r`$ of the summed domain: $`w_t(r)=\mathrm{eq}(\pi_\beta,r_\beta)\sum_{\zeta}\mathrm{eq}(r_\zeta,\zeta)\,c_t(\zeta)`$, that is one $`\mathrm{eq}`$ product over the block variables (the tensor factor, $`d_{\rm sig}-9`$, $`d_{\rm sig}-13`$ or $`d_{\rm sig}-13`$ coordinates) times a sparse contraction inside one block. Operation counts in $`K`$ per level and point, production shape, exact:

| Level | $`c_t`$: additions $`k_tJ_t^{\rm used}`$ | $`\mathrm{eq}`$ tables: mults $`m_t+J_t^{\rm used}`$ | contraction: mults $`J_t^{\rm used}`$ | total mults |
| --- | ---: | ---: | ---: | ---: |
| 1 ($`\pi=r^{(1)}_1`$, $`r=r^{(2)}`$) | 7,340,032 | 532,480 | 524,288 | 1,056,768 |
| 2 ($`\pi=\tau_2`$, $`r=r^{(1)}_1`$) | 1,835,008 | 139,264 | 131,072 | 270,336 |
| 3 ($`\pi=\tau_3`$, $`r=r^{(1)}_2`$, six lanes) | 884,736 | 53,248 | 49,152 | 102,400 |

Total per verification: $`1.0\times10^7`$ $`K`$-additions and $`1.4\times10^6`$ $`K`$-multiplications (a $`K`$-addition is two $`\mathbb F_p`$ additions, a multiplication three $`\mathbb F_p`$ products), about 10 to 20 ms on one core (estimate), independent of $`N`$. The prover's weight tables cost the same additions plus $`|D_t|`$ multiplications to spread $`c_t`$ over the blocks. Committed cost at $`N=2^{15}`$: $`6\cdot2^{19}+12\cdot2^{15}+2^{14}=3{,}555{,}328`$ carry and parity elements; prover bit work $`\sum_t k_t\cdot(\text{inputs of level }t)\approx2.9\times10^9`$ integer additions to form the sums, parallel over blocks.

## 5a. The tied part: the sponge instance and its public claim

Status: normative as implemented in `crates/agg-keccak/src/tied.rs` and `flock_union.rs`
(deviations K11, K14, K21). A tied part is one Flock instance, the sponge instance, plus the
virtual parity chain of Section 5d on its tie challenger; Section 5c extends it to spans.

**Instance.** Part `s` of a segment (`2^d` record positions, `n` live records) is one
single-type Flock proof: a boolean zerocheck, a union-column lincheck and a merged Ligerito
opening over a registry of one type (`flock_union::UnionGeometry`, the pipeline of Flock's
`prove_fast_ligerito_union` composed from its public building blocks, with extra claims in
the opening). The type is the sponge gate (docs/keccak.md Section 2), `k_log = 19`, with `n`
live rows of `2^nu_s`, `nu_s = capacity_log(n, 19)`, so the address space is `M_s = nu_s +
19`. The opening carries the two class claims, the packed-direct public claim `v_pub` and
the ring-switched bit claim of the chain (Section 5d). Nothing is wired: there is no circuit,
no wiring argument and no public segment. The statement words `(mu_i, c~_i)` are bound by a
digest in the challenger and proven by `v_pub`.

**Regions.** Flock commits a slot BatchMajor: word column `c` of row `j` is the padded word
`(c << nu) + j`, and an evaluation claim is on the multilinear extension of the padded word
vector (one `F_{2^128}` element per 128-bit word) at a point of `M - 7` coordinates, rows
low. The sponge block has the w1Encode field first (docs/keccak.md Section 2): lane `t` of a
record is the eight words `8 t + w`, the word columns `[0, 48)`. Mu, `c~`, the constant (bit 0
of word 54) and a zero word are the public region `[48, 56)`. With the coordinates `rho_r`
(`nu_s`) and `rho_c` (3) drawn by the tie challenger (`docs/parity-chain.md` Section 6) the
public point is

```text
P_pub = [rho_r (nu_s) | rho_c (3) | 0 1 1 | 0 ...]                       (sponge, columns 48..55)
```

and the prover's value is the honest evaluation (`flock_union::eval_packed`).

**Check.** After the Flock proof verifies, the verifier checks `v_pub = sum_{j < n} eq(rho_r,
j) sum_{c8 < 8} eq(rho_c, c8) P[j][c8]` with `P[j] = (mu_j words, c~_j words, 1, 0)`.

**Soundness.** Flock's knowledge soundness gives the committed word vector `z_s`. The public
claim compares the multilinear extension, in `nu_s + 3` coordinates, of the public region of
`z_s` with the verifier's own table: if they differ, it holds at the point with probability
at most `(nu_s + 3) / 2^128` for a fixed `z_s` (Schwartz-Zippel), and the tie site's work
covers the sponge's Johnson list (`docs/parity-chain.md` Section 8). The point is drawn after
the sponge cap was absorbed at event 2. The claim also pins the constant column to 1 on live
rows and 0 on dummy rows (Flock's count-derived constant pin requires exactly that); the
dummy rows are zero by Flock's witness contract.

**Parts and shares.** Part `s` holds the segment's live records `[2^d s, 2^d (s + 1))`; its
Flock seed is `SHA3-256(len || "agg-mldsa/v1/flock-part-seed" || fs_sigma || LE64(s))`, and
the tie-geometry digest covers the part index (deviation K10). Its chain proves the share
`Y_{sigma, s} = M^{(sigma)} b'_s`; the shares of a segment's parts XOR to `Y_sigma`, so the
bound (9.3) of Section 7 is unchanged.

**Wire.** The event-2 item of a part is `tied::TiedCommitment` (the sponge cap); the event-6
proof is the version-5 bytes of `docs/parity-chain.md` Section 7.

## 5c. Spans: one tied part over several segments

Status: normative as implemented in `crates/agg-keccak/src/tied.rs` (`TiedPart::new_span`;
deviation K16; mode A: `agg_cli::mode_a`).

**Spans.** A tied part may cover `S` consecutive segments that share its part geometry
(`d`, part index `p`): `TiedPart::new_span(gate, n, d, p, S, profile)`, `n <= 2^d S` records in
segment order (record `i` of segment `sigma` is record `2^d sigma + i` of the span, so only
the last segments may be partial or empty). Mode A uses it for every batch of more than one
segment in whole-segment parts (`d = 13`, `p = 0`, `S` = all segments of the batch, empty
trailing ones included). With `span_log = ceil(log2 S)`, the sponge instance holds the span's
`n` records at rows `2^d sigma + i` (`nu_s = capacity_log(n, 19)` over all records), and the
chain covers `2^span_log` segments, the final vectors of the segments `S..2^span_log` being
zero (`docs/parity-chain.md` Section 2).

**Final vectors.** The public table is the share for a part of one segment and `(Y_0, ..,
Y_{S-1})` for a span. Its digest `SHA3-256(len || "agg-keccak/v3/tied-final-vectors" ||
LE64(S) || packed Y_0 .. Y_{S-1})` is absorbed by the tie challenger before its first
challenge.

**Replay.** The tie-geometry digest covers `S` and `span_log`, the proof header `S`. The
span's Flock seed in mode A is `SHA3-256(len || "agg-mldsa/v1/flock-span-seed" || LE64(S) ||
fs_0 || .. || fs_{S-1})`, a function of event 5 that differs from every part seed.

**Costs.** A span's sponge instance is the per-segment instances stacked: the same rows and
committed words in total, one commitment and one opening instead of `S` of each; the
Ligerito opening grows by about one level per doubling of the committed words
(`docs/keccak.md` Section 12).

**Tests.** `tests/all/tied.rs`: spans of `(n, d, S) = (19, 3, 3)`, `(16, 3, 3)` (last segment
empty) and `(70, 6, 2)` round-trip with each `Y_sigma` equal to the F_p side's walk of its
segment; swapped or flipped final vectors, a wrong final vector proven by the honest prover,
a chain copy flipped in segment 1 only, another segment count and another record count are
rejected.

## 5d. The virtual parity chain (tied-part proof version 5; deviation K21)

Status: normative as implemented in `crates/agg-bridge/src/parity_chain.rs` and
`crates/agg-keccak/src/tied.rs` (`VERSION = 5`); the design, the variable order, the
weight evaluation and the costs are in `docs/parity-chain.md`, the sites in
`docs/soundness.md` 3.6 and 12.4. It is the GF(2) side of a tied part (Sections 5a, 5c).

**Mechanism.** In characteristic 2 the composite $`M=M_3M_2M_1`$ is $`\mathbb F_{2^{128}}`$-linear,
so for $`\mathrm{eq}_Y=\mathrm{eq}(r_o,\rho_3)\,\mathrm{eq}(r_y,\sigma)`$ drawn after the
final vectors, $`\langle\mathrm{eq}_Y,Y\rangle=\sum_xW(x)\,b''(x)`$ with the public weight
$`W=M^{\mathsf T}\mathrm{eq}_Y`$, never materialized. The prover proves this inner product by
one degree-2 sum-check over the `13 + d + span_log` variables of the part's bits in eight
lanes (lanes 6, 7 are the sponge's word columns 48..63 with weight zero): a block phase over
`(i_mid, t, sigma)` whose messages come from the tables $`G=M_2^{\mathsf T}M_3^{\mathsf T}\mathrm{eq}_Y`$
and the honest level-1 parities $`y_1=M_1b''`$ (prover state, unchecked: the sum-check carries
the soundness), then an in-block phase over the position. The last claim is
$`\widetilde W(r)\,\widetilde{b''}(r)`$: the verifier evaluates $`\widetilde W(r)`$ itself (about $`10^7`$ additions,
`docs/parity-chain.md` 5), and the sponge instance's Flock opening proves $`\widetilde{b''}(r)`$ (up to
the row factor of a short sponge) as a third ring-switched claim on the bit polynomial
(`flock_union::BitClaim`, the multilinear skip basis `SkipPoint::Eq` of the vendored core).

**Soundness.** `b'` is the sponge's input, fixed at event 2 before `sd`; `Y` is fixed by its
digest before `eq_Y`. If $`Mb'\ne Y`$ for every element of the sponge's Johnson list, then
$`\langle\mathrm{eq}_Y,Mb'\rangle=\langle\mathrm{eq}_Y,Y\rangle`$ with probability at most
$`L_s(12+\mathrm{span\_log})/2^{128}`$, each round at most $`2L_s/2^{128}`$, and a false final
claim makes the bit claim false for every list element, which Flock's opening rejects under
assumption (F). The bound (9.3) of Section 7 is unchanged. Work: `docs/soundness.md` 3.6.

**Tests.** `agg-bridge` (`parity_chain`): against a brute-force product sum-check on dense
tables (toy shapes with 1, 3, 6 lanes, part offsets, spans), every message, the final value
and $`\widetilde W(r)`$. `agg-keccak` (`tied`, `tests/all/tied.rs`): round trips at 5 to 1,024 records and
at 2,048 (a full-height sponge, Flock's block-first opening), spans and parts; rejections of a flipped `Y` bit, another `sd`, a chain copy that differs
from the sponge, a flipped `b'` bit with stale `Y`, tampered round messages (also with every
nonce re-ground), a tampered bit value, a wrong `s_hat_v` of the bit claim, stale `c~` and a
stale chi witness. A false bit value consistent with the chain (a chain copy of `b'` with one
bit flipped, its `Y`, the copy's value) passes the chain and is rejected by Flock's
ring-switch claim check: end to end at 5 and 64 records (plain opening; 2,048 records, the
block-first opening, as an ignored slow test) and on the full-height sponge instance alone.

## 6. The final public vector

$`Y_\sigma\in\{0,1\}^{4096}`$ per segment, 512 bytes, bit $`\rho_3`$ at byte $`\rho_3\gg3`$, bit position $`\rho_3\bmod8`$ (least significant first, deviation P6), absorbed at event 6 before the Flock proofs and before $`\tau`$. The same bytes are: the public final vectors of the parity chain's claim (Section 5d), the public table $`Y(\rho_3,\sigma)`$ of the PIOP whose extension $`Y(\tau_3)`$ the verifier evaluates in step 4 of spec Section 11, and the right-hand side of the level-3 identity. Proof cost $`512S`$ bytes (2 KB at $`N=2^{15}`$). A verifier MUST reject a proof whose event-6 item has a length other than $`512S`$.

## 7. Soundness

Let $`\delta=b\oplus b'`$ over all lanes, segments and positions of the main domain (padded positions included; they are fixed zeros on the Flock side). $`\delta`$ is determined by events 1 and 2, before $`\mathsf{sd}`$. If both sides accept with the same $`Y_\sigma`$, then by Section 4 (lifting, Booleanity, carry ranges) and Section 5d (the parity chain), $`M_3M_2M_1\delta=0`$ in every segment. Write one level as $`M\in\{0,1\}^{m\times J}`$ whose column $`j`$ is the indicator of a uniform $`k`$-subset $`S_j`$ of the rows, the $`S_j`$ independent, so that $`Mv=\sum_{j\in T}1_{S_j}`$ for $`v`$ of support $`T`$.

- **Lemma 1** (weight 1). If $`|T|=1`$ then $`Mv=1_{S_j}`$ has weight $`k\ge1`$, so $`Mv\ne0`$ with probability 1.
- **Lemma 2** (weight at least 2). If $`|T|\ge2`$ then $`\Pr[Mv=0]\le1/\binom mk`$, with equality at $`|T|=2`$. Proof: fix $`j_0\in T`$ and condition on the other columns of $`T`$; their sum $`w`$ is fixed, and $`Mv=0`$ iff $`S_{j_0}=\mathrm{supp}(w)`$, which has probability $`1/\binom mk`$ if $`|\mathrm{supp}(w)|=k`$ and 0 otherwise. At $`|T|=2`$, $`w=1_{S_{j_1}}`$ always has weight $`k`$. No code distance is needed, only that one column cannot be predicted from the others.
- **Lemma 3** (block reuse). If the same $`M`$ is applied to every block of a nonzero input $`(v_1,\dots,v_B)`$, then $`\Pr[Mv_\beta=0\ \forall\beta]\le1/\binom mk`$: the output is zero only if $`Mv_\beta=0`$ for one fixed $`\beta`$ with $`v_\beta\ne0`$, and Lemmas 1 and 2 bound that single event. No union bound over blocks, lanes or segments is needed (padded blocks are blocks of public zeros).
- **Composition.** For a fixed $`d\ne0`$ and independent levels, $`d^{(0)}=d`$, $`d^{(t)}=M_td^{(t-1)}`$: the events "level $`t`$ is the first with $`d^{(t)}=0`$" are disjoint, and $`d^{(t-1)}`$ is a function of $`d,M_1,\dots,M_{t-1}`$, hence a fixed nonzero vector independent of $`M_t`$ (the levels use distinct hash inputs), so Lemma 3 bounds each term:

```math
\Pr_{\mathsf{sd}}\big[M_3M_2M_1\delta=0\big]\ \le\ \sum_{t=1}^{3}\frac1{\binom{m_t}{k_t}}
=\frac1{\binom{8192}{14}}+\frac1{\binom{8192}{14}}+\frac1{\binom{4096}{18}}
=2^{-145.64}+2^{-145.64}+2^{-163.44}=2^{-144.64}.
```

The bound is independent of $`J_t`$, of the row weights and of the number of blocks; a difference of weight 1 is caught with probability 1. The bridge term of the knowledge-soundness ledger (spec 12.1, equation (9.3)) is

```math
\varepsilon_{\rm bridge}\le2^{-144.64}+\varepsilon_{\rm PIOP}+\varepsilon_{\rm Flock}+\varepsilon_{\rm bind},
```

where $`\varepsilon_{\rm PIOP}`$ collects the bridge instances' field terms (12 Booleanity zero-checks and 7 linear identities at $`\tau`$, 6 level-1 identities at $`r^{(1)}_1`$, the carry lookups, 50 reductions: each at most $`d_{\rm main}/|K|`$, spec 12.1), $`\varepsilon_{\rm Flock}`$ is the sub-proof's knowledge error per profile and $`\varepsilon_{\rm bind}`$ the Akita and Ligerito binding errors. Grinding: an adversary who re-derives $`\mathsf{sd}`$ $`Q`$ times (fresh events 1 or 2) multiplies the first term by $`Q`$; the 16.6-bit margin above $`2^{-128}`$ allows $`Q\le2^{16}`$ without a proof-of-work nonce, so the seed site carries no work in the schedule of `docs/soundness.md` (Section 3.5: over the sponge commitment's Johnson list the bound is $`L_s\cdot2^{-144.64}=2^{-139.0}`$ for `Slim`). Extraction: the Akita extractor gives $`b`$, the Flock extractor gives $`b'`$ with $`\tilde c_i=\mathrm{SHAKE256}(\mu_i\Vert b'_i)`$; the bridge identifies them, so the Keccak statement holds for the bits that satisfy the ring identity (spec 12.2). For the toy shape the same formula gives $`2^{-10.75}`$, and the spec's Section 13 test measures $`1/\binom{64}{3}=1/41{,}664`$ empirically at $`(m,k)=(64,3)`$ on weight-2 differences.

## 8. Worked example (one lane, two levels, $`L=16`$)

Shape $`(J,m,k)=(8,4,2)`$ at both levels, two level-1 blocks, one level-2 block, 4 output bits. Columns are the row sets

```text
M_1: S_0..S_7 = {0,1} {0,2} {0,3} {1,2} {1,3} {2,3} {0,1} {1,2}      row weights 4,5,4,3
M_2: S_0..S_7 = {0,1} {2,3} {0,2} {1,3} {0,3} {1,2} {0,1} {2,3}      row weights 4,4,4,4
b = block 0: 1 0 1 1 0 0 1 0   block 1: 0 1 1 0 1 0 0 0
```

Level 1, integer sums per block and row: block 0 $`(3,3,1,1)`$, block 1 $`(2,1,1,2)`$; parities $`y^{(1)}=(1,1,1,1\mid0,1,1,0)`$, carries $`\kappa^{(1)}=(1,1,0,0\mid1,0,0,1)`$. Level 2 on the 8 parities: sums $`(3,4,3,2)`$, $`Y=(1,0,1,0)`$, $`\kappa^{(2)}=(1,2,1,1)`$; the GF(2) product $`M_2M_1b`$ gives the same $`Y`$. Weight check with the integer vector $`\lambda=(1,2,4,8)`$ in place of $`\mathrm{eq}(\pi_\rho,\cdot)`$: $`c_j=\sum_{\rho\in S_j}\lambda_\rho`$ is $`(3,5,9,6,10,12,3,6)`$ for $`M_1`$ and $`(3,12,5,10,9,6,3,12)`$ for $`M_2`$; block 0 of level 1 satisfies $`\sum_\rho\lambda_\rho(y_\rho+2\kappa_\rho)=21=\sum_jc_jb_j`$, and level 2 satisfies $`39=39`$. Stored carry: $`\kappa=2`$ is committed as $`K=2-2^{15}=-32{,}766`$ and looked up as $`K+2^{15}=2\in T_{16}`$. Tampering: flipping bit 3 of block 1 on either side changes $`M_1e`$ to $`(0,0,0,0\mid0,1,1,0)`$ and $`Y`$ by $`(1,0,1,0)`$, so the two sides disagree on $`Y`$ (Lemma 1). The weight-2 difference at columns 0 and 6 of block 0 has $`S_0=S_6`$ and is invisible to level 1, the event of Lemma 2 with probability $`1/\binom42=1/6`$ per column pair; it is caught by level 2 only if its level-1 image is nonzero, which here it is not. With the production shape this event has probability $`2^{-145.6}`$.

## 9. Test plan

All commands run as `taskset -c 0,1,3-15 cargo test --release -p <crate>`; batch sizes stay at or below 1K signatures and $`2^{14}`$ permutations. Existing tests are listed in `docs/witness.md` Section 9, `docs/piop.md` Section 13 and `docs/keccak.md` Section 5; the bridge-specific ones, present or required, are listed below. Module names are those of the `agg-bridge` skeletons (`seed`, `parity`, `gf2_walker`, `soundness`); the function names are required of them.

1. **Seed and matrices** (`agg-bridge::seed`, present): determinism; distinct streams per seed, level and column; column weight exactly $`k_t`$ and no repeated row; the reading rule including a skipped repeat; the known answer for column 0 of each level at $`\mathsf{sd}=0^{32}`$ (cross-checked against an independent cSHAKE256); row-weight caps on derived matrices.
2. **Toy instance, both sides against a direct computation** (`agg-bridge`; (a), (c) and the forward walk of (b) present in `tests/all/toy.rs`): `Shape::TOY`, $`L=2^{12}`$ bits drawn from a seeded PRNG, matrices from `derive_matrices` generalized to the toy levels. (a) $`\mathbb F_p`$ side: `parity::sums`, parities and carries per level against a dense $`\{0,1\}`$ matrix product computed with `i64` arithmetic; every carry within its range; the three identities (9.2) at every Boolean point; `weights::c_t` at ten random $`K`$ points against $`M_t^{\mathsf T}\mathrm{eq}(\pi_\rho,\cdot)`$ formed densely, and the verifier's terminal formula against the full inner product $`\sum_xw_t(x)v(x)`$ reduced to the multilinear extension. (b) GF(2) side: `gf2_walker::composite` (the forward XOR walk) against the dense composite $`M_3M_2M_1`$ (16 rows by 4,096 columns) and its transpose applied to a random $`\mathbb F_{2^{128}}`$ vector. (c) Agreement: the $`\mathbb F_p`$ parities reduced modulo 2 equal the GF(2) outputs at every level and $`Y`$ agrees.
3. **Single-bit tampering on either side** (toy part present: `single_bit_tampering_on_either_side_changes_y` in `agg-bridge/tests/all/toy.rs`; deterministic by Lemma 1 at level 1): for 64 random positions, flip bit $`x`$ in the $`\mathbb F_p`$ copy only, recompute honest parities and carries from the tampered bits, and check that $`Y`$ differs from the GF(2) side's; then flip it in the GF(2) copy only and check the same; then flip it in both and check agreement (the bridge compares, it does not verify the bits). On the production shape the same test runs through `agg_piop::mode_b` with `FixedMatrices` at $`N=64`$: a changed $`Y_\sigma`$ bit, a flipped committed bit with stale carries, and a flipped public bit each fail (present: `every_stage_tampering_is_rejected_end_to_end` in `agg-piop/tests/all/mode_b.rs`, `bridge_mutations_are_rejected` in `agg-witness/tests/all/negative.rs`).
4. **Weight-2 survival rate** (spec Section 13, present: `agg-bridge/tests/all/production.rs`, columns drawn by the seed derivation, 51 equal pairs): at $`(m,k)=(64,3)`$, $`2\times10^6`$ random column pairs from a fixed PRNG seed; the number with equal row sets (expected $`2\times10^6/\binom{64}{3}=48.0`$) lies in $`[27,69]`$, three standard deviations.
5. **Lifting at the range boundary** (present in `agg-witness` and, for both shapes, in `agg-bridge/tests/all/toy.rs`): rows of weight exactly $`W_t`$ with all-one inputs give parity 1 and the top carry $`\lfloor W_t/2\rfloor`$ (1,023 at level 1); weight $`W_t+1`$ makes `compute` return `RowWeight`.
6. **Flock side** (`agg-keccak`): the tied-part tests of Sections 5c and 5d (`tied`, `tests/all/tied.rs`).
7. **End to end** (present for mode B and for the Fp side with seed-derived matrices in `agg-piop/tests/all/bridge.rs`, and for mode A in `agg-cli/tests/all/mode_a.rs`, where a single-bit flip on either side fails exactly that side): honest aggregates at $`N=256`$ and $`1{,}024`$ verify; a single w1Encode bit changed in one copy is rejected; the verifier's bridge-weight evaluation agrees with the prover's weight tables at the terminal points.

## 10. Decisions and open items

1. The composite $`M_3M_2M_1`$ is proven on the Flock side without materializing any parity: the parity chain's weight $`W=M^{\mathsf T}\mathrm{eq}_Y`$ is evaluated, not committed (Section 5d). Alternative kept in reserve: reveal $`y^{(2)}`$ (6 KB per segment) and drop level 3; it saves little, since level 1 dominates the work.
2. Padded record positions are zero on the Flock side (sponge rows beyond the live records, Section 5a), not left free; the chain's weight ranges over all record positions of the part, so it depends on $`\mathsf{sd}`$ and the part geometry only.
3. Matrices are derived once and shared, not regenerated per use (one Keccak-f per column would cost 1 s per pass on one core).
4. The verifier's GF(2) cost is one evaluation of the chain's weight $`\widetilde W(r)`$ (about $`10^7`$ additions, `docs/parity-chain.md` Section 5); no composite is folded. Remaining candidate if needed: a smaller $`k_1`$ (12: $`2^{-136.5}`$ per level, 14% fewer nonzeros).
5. Derivation cost 0.94 s single-threaded (P4) and $`J_2=2^{19}`$ padding: candidates for wire version 1 (Section 3), not adopted.

## ML-DSA-65 and ML-DSA-87 bridge geometry

For these sets the four w1 bit lanes use four-field parity packing (`crates/agg-bridge/src/parity.rs`). The number of positions per signature is `P = 256k`: 1,536 for 65 and 2,048 for 87. Both use `P_pad = 2,048`, `J_1 = 2^19`, and 256 signatures per level-1 block. Its in-block index is `j + 256 rho + P_pad (i mod 256)`. The padded positions for 65 are zero. `Shape::production(set)` selects this four-lane shape.
