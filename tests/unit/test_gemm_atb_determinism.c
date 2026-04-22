#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "kmamba_kernels.h"

#ifdef _OPENMP
#include <omp.h>
#endif

static float frand_unit(void) {
    return (float)rand() / (float)RAND_MAX - 0.5f;
}

static void gemm_atb_ref(const float *A, const float *B, float *C, int M, int K, int N) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float s = 0.0f;
            for (int k = 0; k < K; k++) s += A[k * M + i] * B[k * N + j];
            C[i * N + j] = s;
        }
    }
}

static int nearly_equal(float a, float b, float tol) {
    return fabsf(a - b) <= tol;
}

int main(void) {
    const int M = 64, K = 128, N = 96;
    const size_t asz = (size_t)K * M;
    const size_t bsz = (size_t)K * N;
    const size_t csz = (size_t)M * N;

    float *A = (float *)malloc(asz * sizeof(float));
    float *B = (float *)malloc(bsz * sizeof(float));
    float *C_ref = (float *)malloc(csz * sizeof(float));
    float *C_run1 = (float *)malloc(csz * sizeof(float));
    float *C_run2 = (float *)malloc(csz * sizeof(float));
    if (!A || !B || !C_ref || !C_run1 || !C_run2) {
        fprintf(stderr, "FAIL: allocation\n");
        free(A); free(B); free(C_ref); free(C_run1); free(C_run2);
        return 1;
    }

    srand(42);
    for (size_t i = 0; i < asz; i++) A[i] = frand_unit();
    for (size_t i = 0; i < bsz; i++) B[i] = frand_unit();

    gemm_atb_ref(A, B, C_ref, M, K, N);

#ifdef _OPENMP
    omp_set_num_threads(4);
#endif
    gemm_f32_AtB(A, B, C_run1, M, K, N);
    gemm_f32_AtB(A, B, C_run2, M, K, N);

    for (size_t i = 0; i < csz; i++) {
        if (!nearly_equal(C_ref[i], C_run1[i], 2e-4f)) {
            fprintf(stderr, "FAIL: ref mismatch at %zu: ref=%f got=%f\n", i, C_ref[i], C_run1[i]);
            free(A); free(B); free(C_ref); free(C_run1); free(C_run2);
            return 1;
        }
        if (!nearly_equal(C_run1[i], C_run2[i], 1e-7f)) {
            fprintf(stderr, "FAIL: nondeterministic at %zu: run1=%f run2=%f\n", i, C_run1[i], C_run2[i]);
            free(A); free(B); free(C_ref); free(C_run1); free(C_run2);
            return 1;
        }
    }

    printf("PASS: gemm_f32_AtB deterministic and numerically correct\n");
    free(A); free(B); free(C_ref); free(C_run1); free(C_run2);
    return 0;
}
