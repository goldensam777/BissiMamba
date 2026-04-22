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

#define GEMM_BK 64
#define GEMM_BN 128

static inline int min_i(int a, int b) { return a < b ? a : b; }

/* T6: Refined OpenMP thresholds considering total work and avoiding nested parallelism overhead */
static inline int omp_enable_parallel(int total_work) {
    /* Minimum work threshold to justify parallelization overhead (~1-2 microsecs per thread) */
    return total_work >= 4096;  /* ~4K float ops minimum */
}

static inline int omp_rows_threshold(int work_rows) {
    /* Granular threshold: parallelize if enough rows for meaningful work distribution */
    return work_rows >= 16;
}

/* Schedule policy: static for regular workloads, dynamic for irregular */
static inline const char* omp_schedule_policy(int regular_workload) {
    return regular_workload ? "static" : "dynamic";
}

/* ============================================================================
 * GEMM: C[M,N] = A[M,K] @ B[K,N]
 * ============================================================================ */

void gemm_f32(const float *A, const float *B, float *C,
              int M, int K, int N) {
    /* Zero C */
    memset(C, 0, (size_t)M * N * sizeof(float));

    /* Cache-blocked i,k,j for better locality on B/C rows. */
    /* T6: Parallelize based on total work (M*K*N) to avoid overhead on small matrices */
    #pragma omp parallel for schedule(static) if(omp_enable_parallel(M * K * N))
    for (int i = 0; i < M; i++) {
        float *c_row = &C[i * N];
        for (int kk = 0; kk < K; kk += GEMM_BK) {
            int kend = min_i(kk + GEMM_BK, K);
            for (int jj = 0; jj < N; jj += GEMM_BN) {
                int jend = min_i(jj + GEMM_BN, N);
                for (int k = kk; k < kend; k++) {
                    float a_ik = A[i * K + k];
                    int j = jj;
#if defined(__AVX2__) || defined(__AVX__)
                    __m256 va = _mm256_set1_ps(a_ik);
                    for (; j + 7 < jend; j += 8) {
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
                    for (; j < jend; j++) {
                        c_row[j] += a_ik * B[k * N + j];
                    }
                }
            }
        }
    }
}

/* ============================================================================
 * GEMM ABt: C[M,N] = A[M,K] @ B^T[N,K]
 * ============================================================================ */

void gemm_f32_ABt(const float *A, const float *B, float *C,
                  int M, int K, int N) {
    /* T6: GEMM ABt - parallelize based on output size with coarse-grain scheduling */
    #pragma omp parallel for collapse(2) schedule(static, 4) if(omp_enable_parallel(M * K * N))
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

#ifndef GEMM_ATB_BK
#define GEMM_ATB_BK 64
#endif

void gemm_f32_AtB(const float *A, const float *B, float *C,
                  int M, int K, int N) {
    memset(C, 0, (size_t)M * N * sizeof(float));

    /* A is [K,M] but we want its transpose [M,K] @ B [K,N] */
    /* So C[i,j] = sum_k A[k,i] * B[k,j] */
    /* Cache-blocked over K for better B reuse and reduced TLB pressure */
    /* T6: GEMM AtB - parallelize outer block loop with static scheduling for cache affinity */
    #pragma omp parallel for schedule(static, 2) if(omp_enable_parallel(M * K * N))
    for (int ii = 0; ii < M; ii += GEMM_ATB_BK) {
        int iend = min_i(ii + GEMM_ATB_BK, M);
        for (int kk = 0; kk < K; kk += GEMM_ATB_BK) {
            int kend = min_i(kk + GEMM_ATB_BK, K);
            for (int i = ii; i < iend; i++) {
                float *c_row = &C[i * N];
                for (int k = kk; k < kend; k++) {
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
    }
}

/* ============================================================================
 * GEMV: y[M] = A[M,N] @ x[N]
 * ============================================================================ */

void gemv_f32(const float *A, const float *x, float *y,
              int M, int N) {
    /* T6: GEMV - parallelize rows with coarse chunks to reduce scheduling overhead */
    #pragma omp parallel for schedule(static, 4) if(omp_enable_parallel(M * N))
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

#ifndef GEMV_AT_BM
#define GEMV_AT_BM 128
#endif

void gemv_f32_At(const float *A, const float *x, float *y,
                 int M, int N) {
    /* y[j] = sum_i A[i,j] * x[i] */
    /* Cache-blocked over M to keep slices of x in cache */
    memset(y, 0, (size_t)N * sizeof(float));

    for (int ii = 0; ii < M; ii += GEMV_AT_BM) {
        int iend = min_i(ii + GEMV_AT_BM, M);
        /* T6: GEMV At - parallelize columns with fine-grain dynamic for load balance */
        #pragma omp parallel for schedule(dynamic, 16) if(omp_enable_parallel(M * N))
        for (int j = 0; j < N; j++) {
            float sum = y[j];
            int i = ii;
#if defined(__AVX2__) || defined(__AVX__)
            __m256 vsum = _mm256_setzero_ps();
            for (; i + 7 < iend; i += 8) {
                __m256 va = _mm256_set_ps(
                    A[(i + 7) * N + j], A[(i + 6) * N + j], A[(i + 5) * N + j], A[(i + 4) * N + j],
                    A[(i + 3) * N + j], A[(i + 2) * N + j], A[(i + 1) * N + j], A[i * N + j]
                );
                __m256 vx = _mm256_loadu_ps(&x[i]);
                #if defined(__FMA__)
                vsum = _mm256_fmadd_ps(va, vx, vsum);
                #else
                vsum = _mm256_add_ps(vsum, _mm256_mul_ps(va, vx));
                #endif
            }
            float tmp[8];
            _mm256_storeu_ps(tmp, vsum);
            sum += tmp[0] + tmp[1] + tmp[2] + tmp[3] + tmp[4] + tmp[5] + tmp[6] + tmp[7];
#endif
            for (; i < iend; i++) {
                sum += A[i * N + j] * x[i];
            }
            y[j] = sum;
        }
    }
}
