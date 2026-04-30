/*
 * init_f32.c — Initialization utilities
 */

#include "kmamba_kernels.h"
#include <stdint.h>

/*
 * Xorshift64 — fast, thread-safe PRNG with per-call local state.
 *
 * Each init function owns its own uint64_t state, so concurrent calls
 * from different threads never share state (no race conditions, no
 * implicit global srand()).  The seed is run through a splitmix64
 * finalizer so even seed=0 produces a non-zero starting state.
 */

static inline uint64_t xorshift64_seed(unsigned int seed) {
    uint64_t z = (uint64_t)seed + 0x9e3779b97f4a7c15ULL;
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

static inline uint64_t xorshift64_next(uint64_t *state) {
    uint64_t x = *state;
    x ^= x << 13;
    x ^= x >>  7;
    x ^= x << 17;
    return (*state = x);
}

/* Returns a float in [min, max) using the top 53 bits of the output. */
static inline float xorshift64_uniform(uint64_t *state, float min, float max) {
    double u = (double)(xorshift64_next(state) >> 11) * (1.0 / (double)(1ULL << 53));
    return min + (float)u * (max - min);
}

void init_xavier_uniform_f32(float *W, size_t fan_in, size_t fan_out, unsigned int seed) {
    uint64_t rng = xorshift64_seed(seed);
    float limit = sqrtf(6.0f / (float)(fan_in + fan_out));
    size_t n = fan_in * fan_out;
    for (size_t i = 0; i < n; i++) {
        W[i] = xorshift64_uniform(&rng, -limit, limit);
    }
}

void init_kaiming_uniform_f32(float *W, size_t fan_in, unsigned int seed) {
    uint64_t rng = xorshift64_seed(seed);
    float limit = sqrtf(6.0f / (float)fan_in);
    size_t n = fan_in;
    for (size_t i = 0; i < n; i++) {
        W[i] = xorshift64_uniform(&rng, -limit, limit);
    }
}
