/*
 * Batch verification throughput of the pq-crystals ML-DSA-44 AVX2
 * implementation, with OpenMP over signatures.
 *
 * The dataset is deterministic (seeded SHAKE256 randomness). Verification is
 * measured twice: through the library's public crypto_sign_verify, which
 * expands every public key at each call, and through a cached path that
 * expands the matrix A and the NTT of t1 once per key before the timed loop.
 * Both paths must return the same accept/reject vector.
 *
 * usage: natver N [--warmup W] [--reps R] [--seed S] [--json PATH]
 *
 * The C files are compiled with the ISA pinned to AVX2 (see Makefile); no
 * AVX-512 instruction may be present in the binary.
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include <omp.h>

#include "align.h"
#include "fips202.h"
#include "packing.h"
#include "params.h"
#include "poly.h"
#include "polyvec.h"
#include "sign.h"

#include "natver.h"

#define MLEN 64

typedef struct {
  polyvecl mat[K];
  poly t1hat[K];
  uint8_t tr[TRBYTES];
} cached_key;

static double now(void) {
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return (double)t.tv_sec + 1e-9 * (double)t.tv_nsec;
}

static void *xaligned(size_t bytes) {
  size_t aligned = (bytes + 63u) & ~(size_t)63u;
  void *p = aligned_alloc(64, aligned);
  if (p == NULL) {
    fprintf(stderr, "natver: out of memory for %zu bytes\n", bytes);
    exit(2);
  }
  memset(p, 0, aligned);
  return p;
}

/* Expand one public key outside the timed loop. */
static void cache_key(cached_key *out, const uint8_t *pk) {
  uint8_t rho[SEEDBYTES];
  polyveck t1;
  int i;

  unpack_pk(rho, &t1, pk);
  polyvec_matrix_expand(out->mat, rho);
  for (i = 0; i < K; i++) {
    poly_shiftl(&t1.vec[i]);
    poly_ntt(&t1.vec[i]);
    out->t1hat[i] = t1.vec[i];
  }
  shake256(out->tr, TRBYTES, pk, CRYPTO_PUBLICKEYBYTES);
}

/*
 * The body of crypto_sign_verify_internal with (rho, t1) already expanded.
 * Returns 0 on accept and -1 on reject.
 */
static int verify_cached(const cached_key *ck, const uint8_t *sig, size_t siglen,
                         const uint8_t *m, size_t mlen) {
  unsigned int i, j, pos = 0;
  ALIGNED_UINT8(K * POLYW1_PACKEDBYTES + 14) buf;
  uint8_t mu[CRHBYTES];
  const uint8_t *hint = sig + CTILDEBYTES + L * POLYZ_PACKEDBYTES;
  polyvecl z;
  poly c, w1, h;
  keccak_state state;

  if (siglen != CRYPTO_BYTES) {
    return -1;
  }

  /* mu = H(H(pk), pre, msg), with the empty context */
  shake256_init(&state);
  shake256_absorb(&state, ck->tr, TRBYTES);
  /* crypto_sign_verify prefixes every message with (0, ctxlen). */
  {
    const uint8_t pre[2] = {0, 0};
    shake256_absorb(&state, pre, sizeof(pre));
  }
  shake256_absorb(&state, m, mlen);
  shake256_finalize(&state);
  shake256_squeeze(mu, CRHBYTES, &state);

  poly_challenge(&c, sig);
  poly_ntt(&c);

  for (i = 0; i < L; i++) {
    polyz_unpack(&z.vec[i], sig + CTILDEBYTES + i * POLYZ_PACKEDBYTES);
    if (poly_chknorm(&z.vec[i], GAMMA1 - BETA)) {
      return -1;
    }
    poly_ntt(&z.vec[i]);
  }

  for (i = 0; i < K; i++) {
    polyvecl_pointwise_acc_montgomery(&w1, &ck->mat[i], &z);
    poly_pointwise_montgomery(&h, &c, &ck->t1hat[i]);
    poly_sub(&w1, &w1, &h);
    poly_reduce(&w1);
    poly_invntt_tomont(&w1);

    memset(&h, 0, sizeof(poly));
    if (hint[OMEGA + i] < pos || hint[OMEGA + i] > OMEGA) {
      return -1;
    }
    for (j = pos; j < hint[OMEGA + i]; j++) {
      if (j > pos && hint[j] <= hint[j - 1]) {
        return -1;
      }
      h.coeffs[hint[j]] = 1;
    }
    pos = hint[OMEGA + i];

    poly_caddq(&w1);
    poly_use_hint(&w1, &w1, &h);
    polyw1_pack(buf.coeffs + i * POLYW1_PACKEDBYTES, &w1);
  }

  for (j = pos; j < OMEGA; j++) {
    if (hint[j]) {
      return -1;
    }
  }

  shake256_init(&state);
  shake256_absorb(&state, mu, CRHBYTES);
  shake256_absorb(&state, buf.coeffs, K * POLYW1_PACKEDBYTES);
  shake256_finalize(&state);
  shake256_squeeze(buf.coeffs, CTILDEBYTES, &state);
  for (i = 0; i < CTILDEBYTES; i++) {
    if (buf.coeffs[i] != sig[i]) {
      return -1;
    }
  }
  return 0;
}

static int verify_uncached(const uint8_t *pk, const uint8_t *sig,
                           const uint8_t *m, size_t mlen) {
  return crypto_sign_verify(sig, CRYPTO_BYTES, m, mlen, NULL, 0, pk);
}

static double batch_uncached(const uint8_t *pk, const uint8_t *sig,
                             const uint8_t *m, size_t n, int *bad) {
  double start = now();
  int rejected = 0;
  size_t i;

#pragma omp parallel for schedule(static) reduction(+ : rejected)
  for (i = 0; i < n; i++) {
    rejected += verify_uncached(pk + i * CRYPTO_PUBLICKEYBYTES,
                                sig + i * CRYPTO_BYTES, m + i * MLEN, MLEN) != 0;
  }
  *bad = rejected;
  return now() - start;
}

static double batch_cached(const cached_key *keys, const uint8_t *sig,
                           const uint8_t *m, size_t n, int *bad) {
  double start = now();
  int rejected = 0;
  size_t i;

#pragma omp parallel for schedule(static) reduction(+ : rejected)
  for (i = 0; i < n; i++) {
    rejected += verify_cached(&keys[i], sig + i * CRYPTO_BYTES, CRYPTO_BYTES,
                              m + i * MLEN, MLEN) != 0;
  }
  *bad = rejected;
  return now() - start;
}

static void json_array(FILE *f, const double *values, int count) {
  int i;
  fputc('[', f);
  for (i = 0; i < count; i++) {
    fprintf(f, "%s%.3f", i == 0 ? "" : ", ", values[i] * 1e3);
  }
  fputc(']', f);
}

int main(int argc, char **argv) {
  size_t n = 32768;
  int warmup = 1;
  int reps = 5;
  uint64_t seed = 1;
  const char *json_path = NULL;
  uint8_t *pk, *sig, *m;
  cached_key *keys;
  double *uncached, *cached;
  int rejected_uncached = 0, rejected_cached = 0;
  int accept_vector_mismatches = 0;
  int i, arg;
  char pk_hex[65], sig_hex[65], msg_hex[65];

  if (argc > 1 && argv[1][0] != '-') {
    n = (size_t)strtoull(argv[1], NULL, 10);
  }
  for (arg = 1; arg < argc; arg++) {
    if (strcmp(argv[arg], "--warmup") == 0 && arg + 1 < argc) {
      warmup = atoi(argv[++arg]);
    } else if (strcmp(argv[arg], "--reps") == 0 && arg + 1 < argc) {
      reps = atoi(argv[++arg]);
    } else if (strcmp(argv[arg], "--seed") == 0 && arg + 1 < argc) {
      seed = strtoull(argv[++arg], NULL, 10);
    } else if (strcmp(argv[arg], "--json") == 0 && arg + 1 < argc) {
      json_path = argv[++arg];
    } else if (argv[arg][0] != '-' && argv[arg][0] >= '0' && argv[arg][0] <= '9') {
      n = (size_t)strtoull(argv[arg], NULL, 10);
    } else {
      fprintf(stderr, "natver: unknown argument %s\n", argv[arg]);
      return 2;
    }
  }
  if (n == 0 || reps < 1) {
    fprintf(stderr, "natver: N and reps must be positive\n");
    return 2;
  }

  randombytes_init(seed);
  pk = xaligned(n * CRYPTO_PUBLICKEYBYTES);
  sig = xaligned(n * CRYPTO_BYTES);
  m = xaligned(n * MLEN);
  keys = xaligned(n * sizeof(cached_key));
  uncached = xaligned((size_t)(warmup + reps) * sizeof(double));
  cached = xaligned((size_t)(warmup + reps) * sizeof(double));

  /*
   * One key per record, as in the aggregation benchmark. Generation is
   * sequential so that the SHAKE stream, and therefore the dataset, does not
   * depend on thread scheduling.
   */
  {
    uint8_t *sk = xaligned(CRYPTO_SECRETKEYBYTES);
    size_t siglen;
    for (size_t r = 0; r < n; r++) {
      crypto_sign_keypair(pk + r * CRYPTO_PUBLICKEYBYTES, sk);
      randombytes(m + r * MLEN, MLEN);
      crypto_sign_signature(sig + r * CRYPTO_BYTES, &siglen,
                            m + r * MLEN, MLEN, NULL, 0, sk);
      if (siglen != CRYPTO_BYTES) {
        fprintf(stderr, "natver: signature %zu has length %zu\n", r, siglen);
        return 2;
      }
    }
    free(sk);
  }

#pragma omp parallel for schedule(static)
  for (size_t r = 0; r < n; r++) {
    cache_key(&keys[r], pk + r * CRYPTO_PUBLICKEYBYTES);
  }

  /* The two paths must agree signature by signature, not only in totals. */
  {
    int *uncached_verdict = xaligned(n * sizeof(int));
    int *cached_verdict = xaligned(n * sizeof(int));
#pragma omp parallel for schedule(static)
    for (size_t r = 0; r < n; r++) {
      uncached_verdict[r] = verify_uncached(
          pk + r * CRYPTO_PUBLICKEYBYTES, sig + r * CRYPTO_BYTES,
          m + r * MLEN, MLEN);
      cached_verdict[r] = verify_cached(
          &keys[r], sig + r * CRYPTO_BYTES, CRYPTO_BYTES, m + r * MLEN, MLEN);
    }
    for (size_t r = 0; r < n; r++) {
      accept_vector_mismatches += uncached_verdict[r] != cached_verdict[r];
    }
    free(uncached_verdict);
    free(cached_verdict);
  }
  if (accept_vector_mismatches != 0) {
    fprintf(stderr, "natver: %d accept/reject mismatches between the two paths\n",
            accept_vector_mismatches);
    return 1;
  }

  for (i = 0; i < warmup + reps; i++) {
    uncached[i] = batch_uncached(pk, sig, m, n, &rejected_uncached);
    cached[i] = batch_cached(keys, sig, m, n, &rejected_cached);
    if (rejected_uncached != 0 || rejected_cached != 0) {
      fprintf(stderr,
              "natver: rejected %d uncached / %d cached in run %d\n",
              rejected_uncached, rejected_cached, i);
      return 1;
    }
  }

  sha256_hex(pk, n * CRYPTO_PUBLICKEYBYTES, pk_hex);
  sha256_hex(sig, n * CRYPTO_BYTES, sig_hex);
  sha256_hex(m, n * MLEN, msg_hex);

  if (json_path != NULL) {
    FILE *f = fopen(json_path, "w");
    if (f == NULL) {
      fprintf(stderr, "natver: cannot write %s\n", json_path);
      return 2;
    }
    fprintf(f, "{\n");
    fprintf(f, "  \"tool\": \"natver\",\n");
    fprintf(f, "  \"n\": %zu,\n", n);
    fprintf(f, "  \"threads\": %d,\n", omp_get_max_threads());
    fprintf(f, "  \"seed\": %llu,\n", (unsigned long long)seed);
    fprintf(f, "  \"warmup\": %d,\n", warmup);
    fprintf(f, "  \"reps\": %d,\n", reps);
    fprintf(f, "  \"message_bytes\": %d,\n", MLEN);
    fprintf(f, "  \"signature_bytes\": %d,\n", CRYPTO_BYTES);
    fprintf(f, "  \"cached_key_bytes\": %zu,\n", n * sizeof(cached_key));
    fprintf(f, "  \"accept_vector_mismatches\": %d,\n", accept_vector_mismatches);
    fprintf(f, "  \"dataset_sha256\": {\"pk\": \"%s\", \"signature\": \"%s\", \"message\": \"%s\"},\n",
            pk_hex, sig_hex, msg_hex);
    fprintf(f, "  \"uncached_ms\": ");
    json_array(f, uncached + warmup, reps);
    fprintf(f, ",\n  \"cached_ms\": ");
    json_array(f, cached + warmup, reps);
    fprintf(f, ",\n  \"rejected_uncached\": %d,\n  \"rejected_cached\": %d\n}\n",
            rejected_uncached, rejected_cached);
    fclose(f);
  }

  printf("natver n %zu threads %d uncached_ms %.2f cached_ms %.2f rejected %d\n",
         n, omp_get_max_threads(), 1e3 * uncached[warmup], 1e3 * cached[warmup],
         rejected_uncached + rejected_cached);
  free(pk);
  free(sig);
  free(m);
  free(keys);
  free(uncached);
  free(cached);
  return 0;
}
