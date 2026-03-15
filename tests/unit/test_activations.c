/*
 * test_activations.c — Tests unitaires activations optimatrix (ASM)
 *
 * Phase 1.2 : Tests Activations (SiLU, Sigmoid, Softplus)
 * Objectif : Valider précision numérique et performance des kernels AVX2
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include <time.h>

/* Définitions manuelles des fonctions optimatrix */
void silu_f32(float *x, float *y, long n);
void sigmoid_f32(float *x, float *y, long n);
void softplus_f32(float *x, float *y, long n);

#define EPSILON 1e-7f
#define TEST_RANGE 10.0f

/* ============================================================
 * Références C pures (pour comparaison)
 * ============================================================ */

static float silu_reference(float x) {
    return x / (1.0f + expf(-x));
}

static float sigmoid_reference(float x) {
    if (x > 20.0f) return 1.0f;
    if (x < -20.0f) return 0.0f;
    return 1.0f / (1.0f + expf(-x));
}

static float softplus_reference(float x) {
    if (x > 20.0f) return x;
    if (x < -20.0f) return 0.0f;
    return logf(1.0f + expf(x));
}

/* ============================================================
 * Utilitaires de test
 * ============================================================ */

static int compare_vectors(const float *ref, const float *test, size_t n, float eps) {
    for (size_t i = 0; i < n; i++) {
        float diff = fabsf(ref[i] - test[i]);
        if (diff > eps) {
            printf("Mismatch at [%zu]: ref=%.10f, test=%.10f, diff=%.10f\n", 
                   i, ref[i], test[i], diff);
            return 0;
        }
    }
    return 1;
}

static void fill_test_values(float *data, size_t n, float min, float max) {
    for (size_t i = 0; i < n; i++) {
        float t = (float)i / (float)(n - 1);
        data[i] = min + t * (max - min);
    }
}

/* ============================================================
 * Tests SiLU
 * ============================================================ */

static int test_silu_basic() {
    printf("Testing SiLU basic values...\n");
    
    const size_t n = 5;
    float input[5] = {-2.0f, -1.0f, 0.0f, 1.0f, 2.0f};
    float ref[5], test[5];
    
    /* Référence */
    for (size_t i = 0; i < n; i++) {
        ref[i] = silu_reference(input[i]);
    }
    
    /* Test */
    silu_f32(input, test, (long)n);
    
    printf("Input:    [%.2f, %.2f, %.2f, %.2f, %.2f]\n", 
           input[0], input[1], input[2], input[3], input[4]);
    printf("Reference: [%.6f, %.6f, %.6f, %.6f, %.6f]\n", 
           ref[0], ref[1], ref[2], ref[3], ref[4]);
    printf("AVX2:      [%.6f, %.6f, %.6f, %.6f, %.6f]\n", 
           test[0], test[1], test[2], test[3], test[4]);
    
    int result = compare_vectors(ref, test, n, EPSILON);
    TEST_ASSERT(result, "SiLU basic values failed");
    
    printf("PASS: SiLU basic values\n");
    return 1;
}

static int test_silu_range() {
    printf("Testing SiLU range [-10, 10]...\n");
    
    const size_t n = 1001;
    float *input = (float*)malloc(n * sizeof(float));
    float *ref = (float*)malloc(n * sizeof(float));
    float *test = (float*)malloc(n * sizeof(float));
    
    fill_test_values(input, n, -TEST_RANGE, TEST_RANGE);
    
    /* Référence */
    for (size_t i = 0; i < n; i++) {
        ref[i] = silu_reference(input[i]);
    }
    
    /* Test */
    silu_f32(input, test, (long)n);
    
    int result = compare_vectors(ref, test, n, EPSILON);
    
    free(input); free(ref); free(test);
    
    TEST_ASSERT(result, "SiLU range test failed");
    printf("PASS: SiLU range test (1001 points)\n");
    return 1;
}

/* ============================================================
 * Tests Sigmoid
 * ============================================================ */

static int test_sigmoid_basic() {
    printf("Testing Sigmoid basic values...\n");
    
    const size_t n = 7;
    float input[7] = {-10.0f, -5.0f, -1.0f, 0.0f, 1.0f, 5.0f, 10.0f};
    float ref[7], test[7];
    
    /* Référence */
    for (size_t i = 0; i < n; i++) {
        ref[i] = sigmoid_reference(input[i]);
    }
    
    /* Test */
    sigmoid_f32(input, test, (long)n);
    
    printf("Input:    [%.1f, %.1f, %.1f, %.1f, %.1f, %.1f, %.1f]\n", 
           input[0], input[1], input[2], input[3], input[4], input[5], input[6]);
    printf("Reference: [%.6f, %.6f, %.6f, %.6f, %.6f, %.6f, %.6f]\n", 
           ref[0], ref[1], ref[2], ref[3], ref[4], ref[5], ref[6]);
    printf("AVX2:      [%.6f, %.6f, %.6f, %.6f, %.6f, %.6f, %.6f]\n", 
           test[0], test[1], test[2], test[3], test[4], test[5], test[6]);
    
    int result = compare_vectors(ref, test, n, EPSILON);
    TEST_ASSERT(result, "Sigmoid basic values failed");
    
    printf("PASS: Sigmoid basic values\n");
    return 1;
}

static int test_sigmoid_extremes() {
    printf("Testing Sigmoid extreme values...\n");
    
    const size_t n = 4;
    float input[4] = {-100.0f, -25.0f, 25.0f, 100.0f};
    float ref[4], test[4];
    
    /* Référence */
    for (size_t i = 0; i < n; i++) {
        ref[i] = sigmoid_reference(input[i]);
    }
    
    /* Test */
    sigmoid_f32(input, test, (long)n);
    
    printf("Input:    [%.1f, %.1f, %.1f, %.1f]\n", 
           input[0], input[1], input[2], input[3]);
    printf("Reference: [%.10f, %.10f, %.10f, %.10f]\n", 
           ref[0], ref[1], ref[2], ref[3]);
    printf("AVX2:      [%.10f, %.10f, %.10f, %.10f]\n", 
           test[0], test[1], test[2], test[3]);
    
    int result = compare_vectors(ref, test, n, EPSILON);
    TEST_ASSERT(result, "Sigmoid extremes failed");
    
    printf("PASS: Sigmoid extreme values\n");
    return 1;
}

/* ============================================================
 * Tests Softplus
 * ============================================================ */

static int test_softplus_basic() {
    printf("Testing Softplus basic values...\n");
    
    const size_t n = 5;
    float input[5] = {-2.0f, -1.0f, 0.0f, 1.0f, 2.0f};
    float ref[5], test[5];
    
    /* Référence */
    for (size_t i = 0; i < n; i++) {
        ref[i] = softplus_reference(input[i]);
    }
    
    /* Test */
    softplus_f32(input, test, (long)n);
    
    printf("Input:    [%.2f, %.2f, %.2f, %.2f, %.2f]\n", 
           input[0], input[1], input[2], input[3], input[4]);
    printf("Reference: [%.6f, %.6f, %.6f, %.6f, %.6f]\n", 
           ref[0], ref[1], ref[2], ref[3], ref[4]);
    printf("AVX2:      [%.6f, %.6f, %.6f, %.6f, %.6f]\n", 
           test[0], test[1], test[2], test[3], test[4]);
    
    int result = compare_vectors(ref, test, n, EPSILON);
    TEST_ASSERT(result, "Softplus basic values failed");
    
    printf("PASS: Softplus basic values\n");
    return 1;
}

/* ============================================================
 * Benchmarks de performance
 * ============================================================ */

static void benchmark_activations() {
    printf("\n=== Performance Benchmarks ===\n");
    
    const size_t n = 1000000;  // 1M elements
    const int iterations = 100;
    
    float *input = (float*)malloc(n * sizeof(float));
    float *output = (float*)malloc(n * sizeof(float));
    
    /* Remplir avec valeurs aléatoires */
    for (size_t i = 0; i < n; i++) {
        input[i] = ((float)rand() / RAND_MAX) * 20.0f - 10.0f;  // [-10, 10]
    }
    
    struct timespec start, end;
    
    /* Benchmark SiLU */
    clock_gettime(CLOCK_MONOTONIC, &start);
    for (int iter = 0; iter < iterations; iter++) {
        silu_f32(input, output, (long)n);
    }
    clock_gettime(CLOCK_MONOTONIC, &end);
    double silu_time = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
    
    /* Benchmark Sigmoid */
    clock_gettime(CLOCK_MONOTONIC, &start);
    for (int iter = 0; iter < iterations; iter++) {
        sigmoid_f32(input, output, (long)n);
    }
    clock_gettime(CLOCK_MONOTONIC, &end);
    double sigmoid_time = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
    
    /* Benchmark Softplus */
    clock_gettime(CLOCK_MONOTONIC, &start);
    for (int iter = 0; iter < iterations; iter++) {
        softplus_f32(input, output, (long)n);
    }
    clock_gettime(CLOCK_MONOTONIC, &end);
    double softplus_time = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
    
    /* Calcul performance */
    double elems_per_sec = (double)n * iterations / 1e6;
    
    printf("SiLU Performance:    %.3f sec (%.2f M elems/sec)\n", silu_time, elems_per_sec / silu_time);
    printf("Sigmoid Performance:  %.3f sec (%.2f M elems/sec)\n", sigmoid_time, elems_per_sec / sigmoid_time);
    printf("Softplus Performance: %.3f sec (%.2f M elems/sec)\n", softplus_time, elems_per_sec / softplus_time);
    
    free(input); free(output);
}

/* ============================================================
 * Main
 * ============================================================ */

int main() {
    printf("=== optimatrix Activations Test Suite ===\n");
    printf("Testing SiLU, Sigmoid, Softplus AVX2 implementations\n\n");
    
    srand(42); /* Pour reproductibilité */
    
    int passed = 0, total = 0;
    
    /* Tests SiLU */
    total++; passed += test_silu_basic();
    total++; passed += test_silu_range();
    
    /* Tests Sigmoid */
    total++; passed += test_sigmoid_basic();
    total++; passed += test_sigmoid_extremes();
    
    /* Tests Softplus */
    total++; passed += test_softplus_basic();
    
    printf("\n=== Test Results ===\n");
    printf("Passed: %d/%d tests\n", passed, total);
    
    if (passed == total) {
        printf("All tests PASSED!\n");
        
        /* Benchmark uniquement si tous les tests passent */
        benchmark_activations();
        
        return 0;
    } else {
        printf("Some tests FAILED!\n");
        return 1;
    }
}
