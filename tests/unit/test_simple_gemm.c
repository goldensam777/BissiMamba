/*
 * test_simple_gemm.c — Test simple pour GEMM/GEMV
 * Version minimaliste pour contourner les problèmes de build
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>

/* Définitions manuelles des fonctions optimatrix */
void gemm_avx2(float *A, float *B, float *C, long m, long k, long n);
void gemv_avx2(float *A, float *x, float *y, long m, long n);

#define EPSILON 1e-6f

static int test_gemm_simple() {
    printf("Testing GEMM simple (2x3 * 3x2)...\n");
    
    const long m = 2, k = 3, n = 2;
    
    float A[6] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};  // 2x3
    float B[6] = {7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f};  // 3x2
    float C_test[4];  // 2x2
    
    /* Référence manuelle */
    float C_ref[4] = {
        1.0f*7.0f + 2.0f*9.0f + 3.0f*11.0f,  // (0,0)
        1.0f*8.0f + 2.0f*10.0f + 3.0f*12.0f, // (0,1)
        4.0f*7.0f + 5.0f*9.0f + 6.0f*11.0f,  // (1,0)
        4.0f*8.0f + 5.0f*10.0f + 6.0f*12.0f  // (1,1)
    };
    
    /* Test */
    gemm_avx2(A, B, C_test, m, k, n);
    
    printf("Reference: [%.1f, %.1f, %.1f, %.1f]\n", 
           C_ref[0], C_ref[1], C_ref[2], C_ref[3]);
    printf("AVX2:      [%.1f, %.1f, %.1f, %.1f]\n", 
           C_test[0], C_test[1], C_test[2], C_test[3]);
    
    for (int i = 0; i < 4; i++) {
        if (fabsf(C_ref[i] - C_test[i]) > EPSILON) {
            printf("FAIL: Mismatch at %d: %f vs %f\n", i, C_ref[i], C_test[i]);
            return 0;
        }
    }
    
    printf("PASS: GEMM simple\n");
    return 1;
}

static int test_gemv_simple() {
    printf("Testing GEMV simple (3x2 * 2)...\n");
    
    const long m = 3, n = 2;
    
    float A[6] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};  // 3x2
    float x[2] = {7.0f, 8.0f};  // 2x1
    float y_test[3];  // 3x1
    
    /* Référence manuelle */
    float y_ref[3] = {
        1.0f*7.0f + 2.0f*8.0f,  // (0)
        3.0f*7.0f + 4.0f*8.0f,  // (1)
        5.0f*7.0f + 6.0f*8.0f   // (2)
    };
    
    /* Test */
    gemv_avx2(A, x, y_test, m, n);
    
    printf("Reference: [%.1f, %.1f, %.1f]\n", y_ref[0], y_ref[1], y_ref[2]);
    printf("AVX2:      [%.1f, %.1f, %.1f]\n", y_test[0], y_test[1], y_test[2]);
    
    for (int i = 0; i < 3; i++) {
        if (fabsf(y_ref[i] - y_test[i]) > EPSILON) {
            printf("FAIL: Mismatch at %d: %f vs %f\n", i, y_ref[i], y_test[i]);
            return 0;
        }
    }
    
    printf("PASS: GEMV simple\n");
    return 1;
}

int main() {
    printf("=== Simple optimatrix Kernels Test ===\n\n");
    
    int passed = 0, total = 0;
    
    total++; passed += test_gemm_simple();
    total++; passed += test_gemv_simple();
    
    printf("\n=== Results ===\n");
    printf("Passed: %d/%d tests\n", passed, total);
    
    if (passed == total) {
        printf("All tests PASSED!\n");
        return 0;
    } else {
        printf("Some tests FAILED!\n");
        return 1;
    }
}
