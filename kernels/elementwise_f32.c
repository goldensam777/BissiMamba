/*
 * elementwise_f32.c — Elementwise operations in pure C with OpenMP
 */

#include "kmamba_kernels.h"
#include <stdlib.h>

#ifdef _OPENMP
#include <omp.h>
#endif

/* ============================================================
 * T7: Aligned Memory Allocation
 * ============================================================ */

void* km_aligned_alloc(size_t size, size_t alignment) {
    void *ptr = NULL;
    /* Use C11 aligned_alloc if available, otherwise posix_memalign */
    #if defined(__STDC_VERSION__) && (__STDC_VERSION__ >= 201112L)
        /* C11 aligned_alloc requires size to be multiple of alignment */
        size_t aligned_size = (size + alignment - 1) & ~(alignment - 1);
        ptr = aligned_alloc(alignment, aligned_size);
    #else
        /* Fallback to posix_memalign */
        if (posix_memalign(&ptr, alignment, size) != 0) {
            ptr = NULL;
        }
    #endif
    return ptr;
}

void km_aligned_free(void *ptr) {
    free(ptr);
}

static inline int vec_parallel_threshold(long n) {
    return n >= 4096;
}

void hadamard_f32(const float *x, const float *y, float *z, long n) {
    #pragma omp parallel for schedule(static) if(vec_parallel_threshold(n))
    for (long i = 0; i < n; i++) {
        z[i] = x[i] * y[i];
    }
}

void vec_add_f32(const float *a, const float *b, float *out, long n) {
    #pragma omp parallel for schedule(static) if(vec_parallel_threshold(n))
    for (long i = 0; i < n; i++) {
        out[i] = a[i] + b[i];
    }
}

void vec_scale_f32(const float *a, float scale, float *out, long n) {
    #pragma omp parallel for schedule(static) if(vec_parallel_threshold(n))
    for (long i = 0; i < n; i++) {
        out[i] = a[i] * scale;
    }
}

void vec_copy_f32(const float *src, float *dst, long n) {
    #pragma omp parallel for schedule(static) if(vec_parallel_threshold(n))
    for (long i = 0; i < n; i++) {
        dst[i] = src[i];
    }
}

void vec_set_f32(float *a, float val, long n) {
    #pragma omp parallel for schedule(static) if(vec_parallel_threshold(n))
    for (long i = 0; i < n; i++) {
        a[i] = val;
    }
}
