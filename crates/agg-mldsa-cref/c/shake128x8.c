/* Correct SHAKE128 absorb for the Hybrid-SHA-3 permutation/squeeze oracle.
 * The upstream short-input absorb omits input words and SHAKE padding.
 * Each lane is independent; memcpy never reads past the supplied input. */
#include <stdint.h>
#include <string.h>

#include "fips202x8.h"

static uint64_t load64(const uint8_t *p)
{
    uint64_t x;
    memcpy(&x, p, sizeof(x));
    return x;
}

void agg_cref_shake128x8_absorb(keccakx8_state *state, const uint8_t *in0,
                                const uint8_t *in1, const uint8_t *in2,
                                const uint8_t *in3, const uint8_t *in4,
                                const uint8_t *in5, const uint8_t *in6,
                                const uint8_t *in7, int len)
{
    const uint8_t *in[8] = {in0, in1, in2, in3, in4, in5, in6, in7};
    memset(state, 0, sizeof(*state));
    while (len >= SHAKE128_RATE) {
        for (int w = 0; w < 21; ++w) {
            __m512i x = _mm512_set_epi64(
                load64(in[7] + 8 * w), load64(in[6] + 8 * w),
                load64(in[5] + 8 * w), load64(in[4] + 8 * w),
                load64(in[3] + 8 * w), load64(in[2] + 8 * w),
                load64(in[1] + 8 * w), load64(in[0] + 8 * w));
            state->s[w] = _mm512_xor_si512(state->s[w], x);
        }
        XURQ_keccak8x_permute(state->s);
        for (int i = 0; i < 8; ++i)
            in[i] += SHAKE128_RATE;
        len -= SHAKE128_RATE;
    }
    int w = 0;
    for (; w < len / 8; ++w) {
        __m512i x =
            _mm512_set_epi64(load64(in[7] + 8 * w), load64(in[6] + 8 * w),
                             load64(in[5] + 8 * w), load64(in[4] + 8 * w),
                             load64(in[3] + 8 * w), load64(in[2] + 8 * w),
                             load64(in[1] + 8 * w), load64(in[0] + 8 * w));
        state->s[w] = _mm512_xor_si512(state->s[w], x);
    }
    uint64_t tail[8] = {0};
    for (int i = 0; i < 8; ++i) {
        memcpy(&tail[i], in[i] + 8 * w, len % 8);
        tail[i] |= (uint64_t)0x1f << (8 * (len % 8));
    }
    state->s[w] = _mm512_xor_si512(state->s[w], _mm512_loadu_si512(tail));
    state->s[20] =
        _mm512_xor_si512(state->s[20], _mm512_set1_epi64(1ULL << 63));
}
