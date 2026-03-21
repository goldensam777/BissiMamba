/*
 * openblas_utils.h — OpenBLAS wrapper and replacement utilities
 *
 * Replaces optimatrix CPU kernels with OpenBLAS cblas + manual implementations
 * for functions not provided by OpenBLAS (hadamard, activations, etc.)
 */

#ifndef OPENBLAS_UTILS_H
#define OPENBLAS_UTILS_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#include <cblas.h>

/* ============================================================
 * BLAS Wrappers (row-major convenience)
 * ============================================================
 *
 * OpenBLAS/cblas uses column-major by default. We provide row-major
 * wrappers for common operations used in k-mamba.
 */

/* GEMM: C[M,N] = A[M,K] @ B[K,N] (row-major) */
static inline void gemm_rowmajor(const float *A, const float *B, float *C,
                                  int M, int K, int N) {
    /* cblas_sgemm: C = alpha*op(A)*op(B) + beta*C
     * For row-major A[M,K] @ B[K,N] = C[M,N] in col-major:
     * treat as B^T @ A^T = C^T
     * B^T is [N,K], A^T is [K,M], C^T is [N,M]
     * So: CblasTrans, CblasTrans gives us row-major result in col-major storage
     * Actually simpler: CblasNoTrans with correct dimensions
     */
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                M, N, K, 1.0f, A, K, B, N, 0.0f, C, N);
}

/* GEMV: y[M] = A[M,N] @ x[N] (row-major) */
static inline void gemv_rowmajor(const float *A, const float *x, float *y,
                                  int M, int N) {
    cblas_sgemv(CblasRowMajor, CblasNoTrans,
                M, N, 1.0f, A, N, x, 1, 0.0f, y, 1);
}

/* ============================================================
 * Elementwise Operations (not in BLAS)
 * ============================================================ */

/* Hadamard product: z[i] = x[i] * y[i] */
void hadamard(const float *x, const float *y, float *z, long n);

/* Activations */
void silu_f32(const float *x, float *y, long n);        /* SiLU = x * sigmoid(x) */
void relu_f32(const float *x, float *y, long n);        /* ReLU = max(0, x) */
void sigmoid_f32(const float *x, float *y, long n);    /* Sigmoid = 1/(1+exp(-x)) */
void softplus_f32(const float *x, float *y, long n);    /* Softplus = log(1+exp(x)) */

/* Vector operations */
void vector_add(const float *a, const float *b, float *out, long n);
void vector_scale(const float *a, float scale, float *out, long n);

/* ============================================================
 * Optimizer Utilities
 * ============================================================ */

/* Note: MBOptimConfig is now defined in kmamba.h */

/* Gradient operations */
float gradient_norm(const float *grad, size_t n);
void gradient_clip_inplace(float *grad, size_t n, float max_norm);
void gradient_clip(const float *grad, float *grad_clipped, size_t n, float max_norm);

/* Newton-Schulz orthogonalization for MUON optimizer */
void newton_schulz5_inplace(float *G, size_t rows, size_t cols, int steps);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* OPENBLAS_UTILS_H */
