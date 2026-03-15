/*
 * test_gemm_standalone.c — Test GEMM pur sans dépendances
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include <time.h>

#define EPSILON 1e-6f

/* Référence C pur */
static void gemm_reference(const float *A, const float *B, float *C,
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

static int test_gemm_correctness() {
    printf("Testing GEMM correctness...\n");
    
    /* Test 1: Matrices 2x3 * 3x2 */
    {
        const long m = 2, k = 3, n = 2;
        
        float A[6] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};  // 2x3 row-major
        float B[6] = {7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f};  // 3x2 row-major
        float C_ref[4], C_test[4];  // 2x2
        
        /* Calcul référence */
        gemm_reference(A, B, C_ref, m, k, n);
        
        /* Test manuel */
        C_test[0] = 1.0f*7.0f + 2.0f*9.0f + 3.0f*11.0f;  // (0,0)
        C_test[1] = 1.0f*8.0f + 2.0f*10.0f + 3.0f*12.0f; // (0,1)
        C_test[2] = 4.0f*7.0f + 5.0f*9.0f + 6.0f*11.0f;  // (1,0)
        C_test[3] = 4.0f*8.0f + 5.0f*10.0f + 6.0f*12.0f; // (1,1)
        
        printf("Test 1 - 2x3 * 3x2:\n");
        printf("  A = [[1,2,3], [4,5,6]]\n");
        printf("  B = [[7,8], [9,10], [11,12]]\n");
        printf("  Ref = [%.1f, %.1f, %.1f, %.1f]\n", 
               C_ref[0], C_ref[1], C_ref[2], C_ref[3]);
        printf("  Test = [%.1f, %.1f, %.1f, %.1f]\n", 
               C_test[0], C_test[1], C_test[2], C_test[3]);
        
        if (!compare_matrices(C_ref, C_test, m, n, EPSILON)) {
            printf("FAIL: Test 1 failed\n");
            return 0;
        }
        printf("PASS: Test 1\n");
    }
    
    /* Test 2: Matrice identité */
    {
        const long n = 3;
        float I[9] = {1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f};
        float M[9] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f};
        float C_ref[9], C_test[9];
        
        gemm_reference(I, M, C_ref, n, n, n);
        memcpy(C_test, M, 9 * sizeof(float));  // I * M = M
        
        printf("\nTest 2 - Identity * Matrix:\n");
        printf("  Ref = [%.1f, %.1f, %.1f, %.1f, %.1f, %.1f, %.1f, %.1f, %.1f]\n", 
               C_ref[0], C_ref[1], C_ref[2], C_ref[3], C_ref[4], C_ref[5], C_ref[6], C_ref[7], C_ref[8]);
        printf("  Test = [%.1f, %.1f, %.1f, %.1f, %.1f, %.1f, %.1f, %.1f, %.1f]\n", 
               C_test[0], C_test[1], C_test[2], C_test[3], C_test[4], C_test[5], C_test[6], C_test[7], C_test[8]);
        
        if (!compare_matrices(C_ref, C_test, n, n, EPSILON)) {
            printf("FAIL: Test 2 failed\n");
            return 0;
        }
        printf("PASS: Test 2\n");
    }
    
    printf("PASS: GEMM correctness tests\n");
    return 1;
}

static void benchmark_gemm() {
    printf("\nBenchmarking GEMM performance...\n");
    
    const long m = 512, k = 512, n = 512;
    const int iterations = 10;
    
    float *A = (float*)malloc(m * k * sizeof(float));
    float *B = (float*)malloc(k * n * sizeof(float));
    float *C = (float*)malloc(m * n * sizeof(float));
    
    /* Remplir avec des valeurs aléatoires */
    for (long i = 0; i < m * k; i++) A[i] = (float)rand() / RAND_MAX;
    for (long i = 0; i < k * n; i++) B[i] = (float)rand() / RAND_MAX;
    
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);
    
    for (int iter = 0; iter < iterations; iter++) {
        gemm_reference(A, B, C, m, k, n);
    }
    
    clock_gettime(CLOCK_MONOTONIC, &end);
    double elapsed = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
    
    double gflops = (2.0 * m * k * n * iterations) / (elapsed * 1e9);
    printf("Reference GEMM: %.3f sec (%.2f GFLOPS)\n", elapsed, gflops);
    
    free(A); free(B); free(C);
}

int main() {
    printf("=== GEMM Standalone Test Suite ===\n\n");
    
    int passed = 0, total = 0;
    
    total++; passed += test_gemm_correctness();
    
    printf("\n=== Test Results ===\n");
    printf("Passed: %d/%d tests\n", passed, total);
    
    if (passed == total) {
        printf("All tests PASSED!\n");
        benchmark_gemm();
        return 0;
    } else {
        printf("Some tests FAILED!\n");
        return 1;
    }
}
