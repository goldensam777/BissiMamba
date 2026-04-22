/*
 * Stress test for gemm_f32_AtB determinism under varying thread counts
 * T3: Renforcer la couverture tests autour des chemins corrigés
 */

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

static int test_config(int M, int K, int N, int nthreads, int nruns) {
    const size_t asz = (size_t)K * M;
    const size_t bsz = (size_t)K * N;
    const size_t csz = (size_t)M * N;
    float *A = (float *)malloc(asz * sizeof(float));
    float *B = (float *)malloc(bsz * sizeof(float));
    float *C_ref = (float *)malloc(csz * sizeof(float));
    float *C_test = (float *)malloc(csz * sizeof(float));
    int failed = 0;

    if (!A || !B || !C_ref || !C_test) {
        fprintf(stderr, "FAIL: allocation for M=%d K=%d N=%d\n", M, K, N);
        free(A); free(B); free(C_ref); free(C_test);
        return 1;
    }

    srand(42);
    for (size_t i = 0; i < asz; i++) A[i] = frand_unit();
    for (size_t i = 0; i < bsz; i++) B[i] = frand_unit();

    gemm_atb_ref(A, B, C_ref, M, K, N);

#ifdef _OPENMP
    omp_set_num_threads(nthreads);
#endif

    for (int run = 0; run < nruns; run++) {
        memset(C_test, 0, csz * sizeof(float));
        gemm_f32_AtB(A, B, C_test, M, K, N);

        for (size_t i = 0; i < csz; i++) {
            if (!nearly_equal(C_ref[i], C_test[i], 2e-4f)) {
                fprintf(stderr, "FAIL: M=%d K=%d N=%d threads=%d run=%d: ref=%f got=%f at %zu\n",
                        M, K, N, nthreads, run, C_ref[i], C_test[i], i);
                failed = 1;
                goto cleanup;
            }
        }
    }

cleanup:
    free(A); free(B); free(C_ref); free(C_test);
    return failed;
}

int main(void) {
    int failed = 0;
    int total = 0;

    printf("=== Stress test gemm_f32_AtB ===\n");

    /* Test various sizes typical for kmamba */
    int sizes[][3] = {
        {32, 64, 32},    /* Small: typical dim/state combos */
        {64, 128, 64},
        {128, 256, 128},
        {256, 512, 256},
        {384, 1024, 384},  /* Mamba typical dim=384, state=1024 */
        {512, 1024, 512},
        {1024, 2048, 1024},
        {64, 100, 96},     /* Non-power-of-2 sizes */
        {127, 255, 129},
    };
    int nsizes = sizeof(sizes) / sizeof(sizes[0]);

#ifdef _OPENMP
    int max_threads = omp_get_max_threads();
#else
    int max_threads = 1;
#endif

    /* Test with different thread counts */
    int thread_counts[] = {1, 2, 4, max_threads};
    int nthread_counts = sizeof(thread_counts) / sizeof(thread_counts[0]);

    for (int si = 0; si < nsizes; si++) {
        int M = sizes[si][0];
        int K = sizes[si][1];
        int N = sizes[si][2];

        for (int ti = 0; ti < nthread_counts; ti++) {
            int nt = thread_counts[ti];
            if (nt > max_threads) continue;

            total++;
            printf("Testing M=%d K=%d N=%d with %d threads... ", M, K, N, nt);
            fflush(stdout);

            if (test_config(M, K, N, nt, 5) == 0) {
                printf("PASS\n");
            } else {
                printf("FAIL\n");
                failed++;
            }
        }
    }

    printf("\n=== Résultat: %d/%d tests passent ===\n", total - failed, total);
    return (failed == 0) ? 0 : 1;
}
