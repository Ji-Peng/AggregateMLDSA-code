/*
 * Deterministic replacement for the pq-crystals randombytes.c.
 *
 * Key generation and signing draw all randomness from SHAKE256(domain || seed ||
 * counter), so the dataset is identical on every machine and every run. The
 * domain string separates this stream from any other use of SHAKE in the
 * reference implementation.
 */
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "fips202.h"
#include "natver.h"
#include "randombytes.h"

static const uint8_t DOMAIN[24] = "dilithium-natver-v1-----";
static uint64_t stream_counter;
static uint64_t stream_seed;

static void store64_le(uint8_t out[8], uint64_t value) {
  for (unsigned i = 0; i < 8; i++) {
    out[i] = (uint8_t)(value >> (8 * i));
  }
}

void randombytes_init(uint64_t seed) {
  stream_seed = seed;
  stream_counter = 0;
}

void randombytes(uint8_t *out, size_t outlen) {
  uint8_t input[40];
  uint8_t block[64];

  memcpy(input, DOMAIN, sizeof(DOMAIN));
  store64_le(input + 24, stream_seed);
  while (outlen > 0) {
    size_t take = outlen < sizeof(block) ? outlen : sizeof(block);
    store64_le(input + 32, stream_counter);
    shake256(block, sizeof(block), input, sizeof(input));
    memcpy(out, block, take);
    out += take;
    outlen -= take;
    stream_counter++;
  }
}
