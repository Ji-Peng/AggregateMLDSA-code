#ifndef NATVER_H
#define NATVER_H

#include <stddef.h>
#include <stdint.h>

/* Deterministic test randomness: SHAKE256 over a fixed domain and a counter. */
void randombytes(uint8_t *out, size_t outlen);
void randombytes_init(uint64_t seed);

/* Lower-case hexadecimal SHA-256 of a buffer. `out` holds 65 bytes. */
void sha256_hex(const uint8_t *data, size_t len, char out[65]);

#endif
