/*
 * openblas_utils.c — OpenBLAS wrapper and replacement utilities
 *
 * Replaces optimatrix CPU kernels with OpenBLAS cblas + manual implementations
 */

#include "openblas_utils.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

/* ============================================================
 * Elementwise Operations
 * ============================================================ */

void hadamard(const float *x, const float *y, float *z, long n) {
    if (!x || !y || !z || n <= 0) return;
    for (long i = 0; i < n; i++) {
        z[i] = x[i] * y[i];
    }
}

void relu_f32(const float *x, float *y, long n) {
    if (!x || !y || n <= 0) return;
    for (long i = 0; i < n; i++) {
        y[i] = x[i] > 0.0f ? x[i] : 0.0f;
    }
}

void sigmoid_f32(const float *x, float *y, long n) {
    if (!x || !y || n <= 0) return;
    for (long i = 0; i < n; i++) {
        float xi = x[i];
        if (xi > 20.0f) {
            y[i] = 1.0f;
        } else if (xi < -20.0f) {
            y[i] = 0.0f;
        } else {
            y[i] = 1.0f / (1.0f + expf(-xi));
        }
    }
}

void silu_f32(const float *x, float *y, long n) {
    if (!x || !y || n <= 0) return;
    for (long i = 0; i < n; i++) {
        float xi = x[i];
        float sig;
        if (xi > 20.0f) {
            sig = 1.0f;
        } else if (xi < -20.0f) {
            sig = 0.0f;
        } else {
            sig = 1.0f / (1.0f + expf(-xi));
        }
        y[i] = xi * sig;
    }
}

void softplus_f32(const float *x, float *y, long n) {
    if (!x || !y || n <= 0) return;
    for (long i = 0; i < n; i++) {
        float xi = x[i];
        if (xi > 20.0f) {
            y[i] = xi;
        } else if (xi < -20.0f) {
            y[i] = 0.0f;
        } else {
            y[i] = logf(1.0f + expf(xi));
        }
    }
}

void vector_add(const float *a, const float *b, float *out, long n) {
    if (!a || !b || !out || n <= 0) return;
    for (long i = 0; i < n; i++) {
        out[i] = a[i] + b[i];
    }
}

void vector_scale(const float *a, float scale, float *out, long n) {
    if (!a || !out || n <= 0) return;
    for (long i = 0; i < n; i++) {
        out[i] = a[i] * scale;
    }
}

/* ============================================================
 * Gradient Operations
 * ============================================================ */

float gradient_norm(const float *grad, size_t n) {
    if (!grad || n == 0) return 0.0f;
    
    double sq = 0.0;
    for (size_t i = 0; i < n; i++) {
        double g = (double)grad[i];
        sq += g * g;
    }
    return (float)sqrt(sq);
}

void gradient_clip_inplace(float *grad, size_t n, float max_norm) {
    if (!grad || n == 0 || max_norm <= 0.0f) return;
    
    float gn = gradient_norm(grad, n);
    if (gn > max_norm) {
        float scale = max_norm / gn;
        for (size_t i = 0; i < n; i++) {
            grad[i] *= scale;
        }
    }
}

void gradient_clip(const float *grad, float *grad_clipped, size_t n, float max_norm) {
    if (!grad || !grad_clipped || n == 0) return;
    
    for (size_t i = 0; i < n; i++) {
        grad_clipped[i] = grad[i];
    }
    gradient_clip_inplace(grad_clipped, n, max_norm);
}

/* ============================================================
 * Newton-Schulz Orthogonalization
 * ============================================================
 *
 * Computes the polar factor of G using Newton-Schulz iterations.
 * Used by MUON optimizer for orthogonalizing weight matrices.
 *
 * Algorithm (cubic variant, 5 iterations):
 *   G <- G / ||G||_F
 *   for i = 1 to 5:
 *     A = G @ G^T
 *     G = 1.5*G - 0.5*A @ G
 *
 * Note: We use OpenBLAS cblas_sgemm for matrix operations.
 */

void newton_schulz5_inplace(float *G, size_t rows, size_t cols, int steps) {
    if (!G || rows == 0 || cols == 0) return;

    size_t n = rows * cols;

    /* Normalize by Frobenius norm */
    double sq = 0.0;
    for (size_t i = 0; i < n; i++) {
        double v = (double)G[i];
        sq += v * v;
    }
    float norm = (float)sqrt(sq);
    if (norm < 1e-7f) return;
    float inv = 1.0f / norm;
    for (size_t i = 0; i < n; i++) {
        G[i] *= inv;
    }

    /* Temporary buffers */
    float *G_T = (float *)malloc(cols * rows * sizeof(float)); /* G^T [cols x rows] */
    float *A = (float *)malloc(rows * rows * sizeof(float));   /* G @ G^T [rows x rows] */
    float *AG = (float *)malloc(n * sizeof(float));           /* A @ G [rows x cols] */
    if (!G_T || !A || !AG) {
        free(G_T);
        free(A);
        free(AG);
        return;
    }

    for (int s = 0; s < steps; s++) {
        /* G_T = transpose(G) */
        for (size_t r = 0; r < rows; r++) {
            for (size_t c = 0; c < cols; c++) {
                G_T[c * rows + r] = G[r * cols + c];
            }
        }

        /* A = G @ G^T [rows x rows] */
        /* CblasColMajor: C = alpha*op(A)*op(B) + beta*C
         * We want A[row,row] = G[row,col] @ G^T[col,row]
         * In col-major: lda=rows, ldb=rows, ldc=rows */
        memset(A, 0, rows * rows * sizeof(float));
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    (int)rows, (int)rows, (int)cols,
                    1.0f, G, (int)cols, G_T, (int)rows,
                    0.0f, A, (int)rows);

        /* AG = A @ G [rows x cols] */
        memset(AG, 0, n * sizeof(float));
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    (int)rows, (int)cols, (int)rows,
                    1.0f, A, (int)rows, G, (int)cols,
                    0.0f, AG, (int)cols);

        /* G <- 1.5*G - 0.5*AG */
        for (size_t i = 0; i < n; i++) {
            G[i] = 1.5f * G[i] - 0.5f * AG[i];
        }
    }

    free(G_T);
    free(A);
    free(AG);
}
