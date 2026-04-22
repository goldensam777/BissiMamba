#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "kmamba_kernels.h"

#ifdef _OPENMP
#include <omp.h>
#endif

static float frand_unit(void) {
    return (float)rand() / (float)RAND_MAX - 0.5f;
}

static double now_sec(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec + (double)t.tv_nsec / 1e9;
}

static void gemm_ref(const float *A, const float *B, float *C, int M, int K, int N) {
    memset(C, 0, (size_t)M * N * sizeof(float));
    for (int i = 0; i < M; i++) {
        for (int k = 0; k < K; k++) {
            float aik = A[i * K + k];
            for (int j = 0; j < N; j++) C[i * N + j] += aik * B[k * N + j];
        }
    }
}

int main(void) {
    const int M = 128, K = 256, N = 128;
    const size_t asz = (size_t)M * K;
    const size_t bsz = (size_t)K * N;
    const size_t csz = (size_t)M * N;
    const int iters = 20;

    float *A = (float *)malloc(asz * sizeof(float));
    float *B = (float *)malloc(bsz * sizeof(float));
    float *C_ref = (float *)malloc(csz * sizeof(float));
    float *C_opt = (float *)malloc(csz * sizeof(float));
    if (!A || !B || !C_ref || !C_opt) {
        fprintf(stderr, "gate_alloc=FAIL\n");
        free(A); free(B); free(C_ref); free(C_opt);
        return 1;
    }

    srand(7);
    for (size_t i = 0; i < asz; i++) A[i] = frand_unit();
    for (size_t i = 0; i < bsz; i++) B[i] = frand_unit();

    /* Gate A: correctness */
    gemm_ref(A, B, C_ref, M, K, N);
    gemm_f32(A, B, C_opt, M, K, N);
    float max_err = 0.0f;
    for (size_t i = 0; i < csz; i++) {
        float e = fabsf(C_ref[i] - C_opt[i]);
        if (e > max_err) max_err = e;
    }
    int gate_a = (max_err <= 3e-4f);

    /* Gate B: determinism */
    float *C_run1 = (float *)malloc(csz * sizeof(float));
    float *C_run2 = (float *)malloc(csz * sizeof(float));
    if (!C_run1 || !C_run2) {
        fprintf(stderr, "gate_alloc=FAIL\n");
        free(A); free(B); free(C_ref); free(C_opt); free(C_run1); free(C_run2);
        return 1;
    }
#ifdef _OPENMP
    omp_set_num_threads(4);
#endif
    gemm_f32_AtB(B, B, C_run1, N, K, N);
    gemm_f32_AtB(B, B, C_run2, N, K, N);
    int gate_b = 1;
    for (size_t i = 0; i < (size_t)N * N; i++) {
        if (fabsf(C_run1[i] - C_run2[i]) > 1e-7f) { gate_b = 0; break; }
    }

    /* Gate C: perf report */
    double t0 = now_sec();
    for (int i = 0; i < iters; i++) gemm_f32(A, B, C_opt, M, K, N);
    double t1 = now_sec();
    double gflops = (2.0 * M * K * N * iters) / ((t1 - t0) * 1e9);

    /* Gate D: stability long run (finite values) */
    int gate_d = 1;
    for (int i = 0; i < 64; i++) {
        gemm_f32(A, B, C_opt, M, K, N);
        for (size_t j = 0; j < csz; j++) {
            if (!isfinite(C_opt[j])) { gate_d = 0; break; }
        }
        if (!gate_d) break;
    }

    /* Gate E: reversibility marker */
#ifdef KMAMBA_FAST_EXP_APPROX
    const char *gate_e = "approx_enabled";
#else
    const char *gate_e = "exact_enabled";
#endif

    printf("gate_a_correctness=%s max_err=%g\n", gate_a ? "PASS" : "FAIL", max_err);
    printf("gate_b_determinism=%s\n", gate_b ? "PASS" : "FAIL");
    printf("gate_c_perf_gflops=%.4f\n", gflops);
    printf("gate_d_stability=%s\n", gate_d ? "PASS" : "FAIL");
    printf("gate_e_mode=%s\n", gate_e);

    free(A); free(B); free(C_ref); free(C_opt); free(C_run1); free(C_run2);
    return (gate_a && gate_b && gate_d) ? 0 : 1;
}
