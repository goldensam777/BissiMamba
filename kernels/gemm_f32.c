/*
 * gemm_f32.c — Optimized GEMM/GEMV for k-mamba
 * Pure C + OpenMP + AVX2 (where applicable)
 */

#include "kmamba_kernels.h"
#include <string.h>
#include <stdio.h>

#if defined(__AVX2__) || defined(__AVX__)
#include <immintrin.h>
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

/* ============================================================================
 * GEMM: C[M,N] = A[M,K] @ B[K,N]
 * ============================================================================ */

void gemm_f32(const float *A, const float *B, float *C,
              int M, int K, int N) {
    /* Zero C */
    memset(C, 0, (size_t)M * N * sizeof(float));

    /* i, k, j order is cache-friendly for row-major B */
    #pragma omp parallel for
    for (int i = 0; i < M; i++) {
        for (int k = 0; k < K; k++) {
            float a_ik = A[i * K + k];
            int j = 0;

#if defined(__AVX2__) || defined(__AVX__)
            __m256 va = _mm256_set1_ps(a_ik);
            for (; j <= N - 8; j += 8) {
                __m256 vb = _mm256_loadu_ps(&B[k * N + j]);
                __m256 vc = _mm256_loadu_ps(&C[i * N + j]);
                #if defined(__FMA__)
                vc = _mm256_fmadd_ps(va, vb, vc);
                #else
                vc = _mm256_add_ps(vc, _mm256_mul_ps(va, vb));
                #endif
                _mm256_storeu_ps(&C[i * N + j], vc);
            }
#endif
            for (; j < N; j++) {
                C[i * N + j] += a_ik * B[k * N + j];
            }
        }
    }
}

/* ============================================================================
 * GEMM ABt: C[M,N] = A[M,K] @ B^T[N,K]
 * ============================================================================ */

void gemm_f32_ABt(const float *A, const float *B, float *C,
                  int M, int K, int N) {
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            int k = 0;

#if defined(__AVX2__) || defined(__AVX__)
            __m256 vsum = _mm256_setzero_ps();
            for (; k <= K - 8; k += 8) {
                __m256 va = _mm256_loadu_ps(&A[i * K + k]);
                __m256 vb = _mm256_loadu_ps(&B[j * K + k]);
                vsum = _mm256_add_ps(vsum, _mm256_mul_ps(va, vb));
            }
            /* Horizontal sum of vsum */
            float tmp[8];
            _mm256_storeu_ps(tmp, vsum);
            sum = tmp[0] + tmp[1] + tmp[2] + tmp[3] + tmp[4] + tmp[5] + tmp[6] + tmp[7];
#endif
            for (; k < K; k++) {
                sum += A[i * K + k] * B[j * K + k];
            }
            C[i * N + j] = sum;
        }
    }
}

/* ============================================================================
 * GEMM AtB: C[M,N] = A^T[M,K] @ B[K,N]
 * ============================================================================ */

void gemm_f32_AtB(const float *A, const float *B, float *C,
                  int M, int K, int N) {
    memset(C, 0, (size_t)M * N * sizeof(float));

    /* A is [K,M] but we want its transpose [M,K] @ B [K,N] */
    /* So C[i,j] = sum_k A[k,i] * B[k,j] */
    #pragma omp parallel for
    for (int i = 0; i < M; i++) {
        float *c_row = &C[i * N];
        for (int k = 0; k < K; k++) {
            float a_ki = A[k * M + i];
            int j = 0;
#if defined(__AVX2__) || defined(__AVX__)
            __m256 va = _mm256_set1_ps(a_ki);
            for (; j <= N - 8; j += 8) {
                __m256 vb = _mm256_loadu_ps(&B[k * N + j]);
                __m256 vc = _mm256_loadu_ps(&c_row[j]);
                #if defined(__FMA__)
                vc = _mm256_fmadd_ps(va, vb, vc);
                #else
                vc = _mm256_add_ps(vc, _mm256_mul_ps(va, vb));
                #endif
                _mm256_storeu_ps(&c_row[j], vc);
            }
#endif
            for (; j < N; j++) {
                c_row[j] += a_ki * B[k * N + j];
            }
        }
    }
}

/* ============================================================================
 * GEMV: y[M] = A[M,N] @ x[N]
 * ============================================================================ */

void gemv_f32(const float *A, const float *x, float *y,
              int M, int N) {
    #pragma omp parallel for
    for (int i = 0; i < M; i++) {
        float sum = 0.0f;
        int j = 0;
#if defined(__AVX2__) || defined(__AVX__)
        __m256 vsum = _mm256_setzero_ps();
        for (; j <= N - 8; j += 8) {
            __m256 va = _mm256_loadu_ps(&A[i * N + j]);
            __m256 vx = _mm256_loadu_ps(&x[j]);
            vsum = _mm256_add_ps(vsum, _mm256_mul_ps(va, vx));
        }
        float tmp[8];
        _mm256_storeu_ps(tmp, vsum);
        sum = tmp[0] + tmp[1] + tmp[2] + tmp[3] + tmp[4] + tmp[5] + tmp[6] + tmp[7];
#endif
        for (; j < N; j++) {
            sum += A[i * N + j] * x[j];
        }
        y[i] = sum;
    }
}

/* ============================================================================
 * GEMV At: y[N] = A^T[N,M] @ x[M]
 * ============================================================================ */

void gemv_f32_At(const float *A, const float *x, float *y,
                 int M, int N) {
    memset(y, 0, (size_t)N * sizeof(float));

    /* y[j] = sum_i A[i,j] * x[i] */
    for (int i = 0; i < M; i++) {
        float xi = x[i];
        int j = 0;
#if defined(__AVX2__) || defined(__AVX__)
        __m256 vx = _mm256_set1_ps(xi);
        for (; j <= N - 8; j += 8) {
            __m256 va = _mm256_loadu_ps(&A[i * N + j]);
            __m256 vy = _mm256_loadu_ps(&y[j]);
            #if defined(__FMA__)
            vy = _mm256_fmadd_ps(vx, va, vy);
            #else
            vy = _mm256_add_ps(vy, _mm256_mul_ps(vx, va));
            #endif
            _mm256_storeu_ps(&y[j], vy);
        }
#endif
        for (; j < N; j++) {
            y[j] += A[i * N + j] * xi;
        }
    }
}
