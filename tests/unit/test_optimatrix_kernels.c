/*
 * test_optimatrix_kernels.c — Tests unitaires kernels optimatrix (ASM)
 *
 * Phase 1.1 : Tests GEMM/GEMV AVX2
 * Objectif : Valider précision numérique des kernels optimisés
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include <time.h>
#include "optimatrix.h"

/* ============================================================
 * Utilitaires de test
 * ============================================================ */

#define EPSILON 1e-6f
#define TEST_ASSERT(cond, msg) \
    do { \
        if (!(cond)) { \
            printf("FAIL: %s\n", msg); \
            return 0; \
        } \
    } while(0)

#define TEST_ASSERT_FLOAT_EQ(a, b, eps, msg) \
    TEST_ASSERT(fabsf((a) - (b)) < (eps), msg)

static float frand_range(float min, float max) {
    float scale = (float)rand() / (float)RAND_MAX;
    return min + scale * (max - min);
}

static void fill_random(float *data, size_t n, float min, float max) {
    for (size_t i = 0; i < n; i++) {
        data[i] = frand_range(min, max);
    }
}

static int compare_matrices(const float *A, const float *B, size_t rows, size_t cols, float eps) {
    for (size_t i = 0; i < rows * cols; i++) {
        if (fabsf(A[i] - B[i]) > eps) {
            printf("Mismatch at [%zu]: %f vs %f (diff: %f)\n", 
                   i, A[i], B[i], fabsf(A[i] - B[i]));
            return 0;
        }
    }
    return 1;
}

/* ============================================================
 * Références C pures (pour comparaison)
 * ============================================================ */

static void gemm_reference(float *A, float *B, float *C,
                         long m, long k, long n) {
    memset(C, 0, m * n * sizeof(float));
    
    for (long i = 0; i < m; i++) {
        for (long j = 0; j < n; j++) {
            float sum = 0.0f;
            for (long p = 0; p < k; p++) {
                sum += A[i * k + p] * B[p * n + j];
            }
            C[i * n + j] = sum;
        }
    }
}

static void gemv_reference(float *A, float *x, float *y,
                         long m, long n) {
    for (long i = 0; i < m; i++) {
        float sum = 0.0f;
        for (long j = 0; j < n; j++) {
            sum += A[i * n + j] * x[j];
        }
        y[i] = sum;
    }
}

/* ============================================================
 * Tests GEMM
 * ============================================================ */

static int test_gemm_small() {
    printf("Testing GEMM small matrices (3x4 * 4x2)...\n");
    
    const long m = 3, k = 4, n = 2;
    
    float *A = (float*)malloc(m * k * sizeof(float));
    float *B = (float*)malloc(k * n * sizeof(float));
    float *C_ref = (float*)malloc(m * n * sizeof(float));
    float *C_test = (float*)malloc(m * n * sizeof(float));
    
    fill_random(A, m * k, -1.0f, 1.0f);
    fill_random(B, k * n, -1.0f, 1.0f);
    
    /* Référence */
    gemm_reference(A, B, C_ref, m, k, n);
    
    /* Test */
    gemm_avx2(A, B, C_test, m, k, n);
    
    int result = compare_matrices(C_ref, C_test, m, n, EPSILON);
    TEST_ASSERT(result, "GEMM small matrices failed");
    
    free(A); free(B); free(C_ref); free(C_test);
    printf("PASS: GEMM small matrices\n");
    return 1;
}

static int test_gemm_medium() {
    printf("Testing GEMM medium matrices (64x128 * 128x256)...\n");
    
    const long m = 64, k = 128, n = 256;
    
    float *A = (float*)malloc(m * k * sizeof(float));
    float *B = (float*)malloc(k * n * sizeof(float));
    float *C_ref = (float*)malloc(m * n * sizeof(float));
    float *C_test = (float*)malloc(m * n * sizeof(float));
    
    fill_random(A, m * k, -2.0f, 2.0f);
    fill_random(B, k * n, -2.0f, 2.0f);
    
    gemm_reference(A, B, C_ref, m, k, n);
    gemm_avx2(A, B, C_test, m, k, n);
    
    int result = compare_matrices(C_ref, C_test, m, n, EPSILON);
    TEST_ASSERT(result, "GEMM medium matrices failed");
    
    free(A); free(B); free(C_ref); free(C_test);
    printf("PASS: GEMM medium matrices\n");
    return 1;
}

static int test_gemm_edge_cases() {
    printf("Testing GEMM edge cases...\n");
    
    /* Test matrice 1x1 */
    {
        const long m = 1, k = 1, n = 1;
        float A[1] = {2.0f};
        float B[1] = {3.0f};
        float C_ref[1], C_test[1];
        
        gemm_reference(A, B, C_ref, m, k, n);
        gemm_avx2(A, B, C_test, m, k, n);
        
        TEST_ASSERT_FLOAT_EQ(C_ref[0], C_test[0], EPSILON, "1x1 GEMM failed");
    }
    
    /* Test vecteur ligne * vecteur colonne */
    {
        const long m = 1, k = 4, n = 1;
        float A[4] = {1.0f, 2.0f, 3.0f, 4.0f};
        float B[4] = {2.0f, 3.0f, 4.0f, 5.0f};
        float C_ref[1], C_test[1];
        
        gemm_reference(A, B, C_ref, m, k, n);
        gemm_avx2(A, B, C_test, m, k, n);
        
        TEST_ASSERT_FLOAT_EQ(C_ref[0], C_test[0], EPSILON, "Vector GEMM failed");
    }
    
    printf("PASS: GEMM edge cases\n");
    return 1;
}

/* ============================================================
 * Tests GEMV
 * ============================================================ */

static int test_gemv_small() {
    printf("Testing GEMV small (4x3 * 3)...\n");
    
    const long m = 4, n = 3;
    
    float *A = (float*)malloc(m * n * sizeof(float));
    float *x = (float*)malloc(n * sizeof(float));
    float *y_ref = (float*)malloc(m * sizeof(float));
    float *y_test = (float*)malloc(m * sizeof(float));
    
    fill_random(A, m * n, -1.0f, 1.0f);
    fill_random(x, n, -1.0f, 1.0f);
    
    gemv_reference(A, x, y_ref, m, n);
    gemv_avx2(A, x, y_test, m, n);
    
    int result = compare_matrices(y_ref, y_test, m, 1, EPSILON);
    TEST_ASSERT(result, "GEMV small failed");
    
    free(A); free(x); free(y_ref); free(y_test);
    printf("PASS: GEMV small\n");
    return 1;
}

static int test_gemv_large() {
    printf("Testing GEMV large (1024x1024 * 1024)...\n");
    
    const long m = 1024, n = 1024;
    
    float *A = (float*)malloc(m * n * sizeof(float));
    float *x = (float*)malloc(n * sizeof(float));
    float *y_ref = (float*)malloc(m * sizeof(float));
    float *y_test = (float*)malloc(m * sizeof(float));
    
    fill_random(A, m * n, -0.5f, 0.5f);
    fill_random(x, n, -0.5f, 0.5f);
    
    gemv_reference(A, x, y_ref, m, n);
    gemv_avx2(A, x, y_test, m, n);
    
    int result = compare_matrices(y_ref, y_test, m, 1, EPSILON);
    TEST_ASSERT(result, "GEMV large failed");
    
    free(A); free(x); free(y_ref); free(y_test);
    printf("PASS: GEMV large\n");
    return 1;
}

/* ============================================================
 * Benchmarks de performance
 * ============================================================ */

static double benchmark_function(void (*func)(float*, float*, float*, long, long, long),
                              float *A, float *B, float *C, long m, long k, long n, int iterations) {
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);
    
    for (int i = 0; i < iterations; i++) {
        func(A, B, C, m, k, n);
    }
    
    clock_gettime(CLOCK_MONOTONIC, &end);
    double elapsed = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
    return elapsed;
}

static void benchmark_gemm() {
    printf("Benchmarking GEMM performance...\n");
    
    const long m = 512, k = 512, n = 512;
    const int iterations = 100;
    
    float *A = (float*)malloc(m * k * sizeof(float));
    float *B = (float*)malloc(k * n * sizeof(float));
    float *C_ref = (float*)malloc(m * n * sizeof(float));
    float *C_test = (float*)malloc(m * n * sizeof(float));
    
    fill_random(A, m * k, -1.0f, 1.0f);
    fill_random(B, k * n, -1.0f, 1.0f);
    
    double time_ref = benchmark_function(gemm_reference, A, B, C_ref, m, k, n, iterations);
    double time_avx2 = benchmark_function(gemm_avx2, A, B, C_test, m, k, n, iterations);
    
    double speedup = time_ref / time_avx2;
    double gflops_ref = (2.0 * m * k * n * iterations) / (time_ref * 1e9);
    double gflops_avx2 = (2.0 * m * k * n * iterations) / (time_avx2 * 1e9);
    
    printf("Reference: %.3f sec (%.2f GFLOPS)\n", time_ref, gflops_ref);
    printf("AVX2:      %.3f sec (%.2f GFLOPS)\n", time_avx2, gflops_avx2);
    printf("Speedup:   %.2fx\n", speedup);
    
    free(A); free(B); free(C_ref); free(C_test);
}

/* ============================================================
 * Main
 * ============================================================ */

int main() {
    printf("=== optimatrix Kernels Test Suite ===\n");
    printf("Testing GEMM/GEMV AVX2 implementations\n\n");
    
    srand(42); /* Pour reproductibilité */
    
    int passed = 0, total = 0;
    
    /* Tests GEMM */
    total++; passed += test_gemm_small();
    total++; passed += test_gemm_medium();
    total++; passed += test_gemm_edge_cases();
    
    /* Tests GEMV */
    total++; passed += test_gemv_small();
    total++; passed += test_gemv_large();
    
    printf("\n=== Test Results ===\n");
    printf("Passed: %d/%d tests\n", passed, total);
    
    if (passed == total) {
        printf("All tests PASSED!\n");
        
        /* Benchmark uniquement si tous les tests passent */
        printf("\n=== Performance Benchmarks ===\n");
        benchmark_gemm();
        
        return 0;
    } else {
        printf("Some tests FAILED!\n");
        return 1;
    }
}
