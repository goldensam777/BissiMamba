/*
 * convnd_cuda_common.cuh — Utilitaires CUDA communs pour ConvND
 *
 * Factorisation DRY des helpers device/host partagés entre
 * convnd.cu (dense) et convnd_separable.cu (séparable).
 */

#ifndef CONVND_CUDA_COMMON_CUH
#define CONVND_CUDA_COMMON_CUH

#include <cuda_runtime.h>

/* ── Error checking ─────────────────────────────────────────── */
#ifndef CONVND_CUDA_CHECK
#define CONVND_CUDA_CHECK(call) do { \
    cudaError_t _e = (call); \
    if (_e != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s:%d — %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(_e)); \
        return -1; \
    } \
} while(0)
#endif

/* ── Device helpers ────────────────────────────────────────── */

/**
 * Unravel linear index to ND coordinates (device)
 * Pre-allocated strides for efficiency
 */
__device__ __forceinline__ static void
convnd_d_unravel_index(long linear, const long *dims, long ndims,
                       const long *strides, long *coords) {
    for (long axis = ndims; axis-- > 0;) {
        coords[axis] = linear / strides[axis];
        linear %= strides[axis];
    }
}

/**
 * Compute row-major strides (host)
 */
static inline void
convnd_h_make_strides(const long *dims, long ndims, long *strides) {
    long stride = 1;
    for (long axis = ndims; axis-- > 0;) {
        strides[axis] = stride;
        stride *= dims[axis];
    }
}

/**
 * Power computation (host)
 */
static inline long convnd_h_power(long base, long exp) {
    long out = 1;
    for (long i = 0; i < exp; i++) out *= base;
    return out;
}

#endif /* CONVND_CUDA_COMMON_CUH */
