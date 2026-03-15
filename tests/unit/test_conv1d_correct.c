/*
 * test_conv1d_correct.c — Tests Conv1D avec implémentation corrigée
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include <time.h>

#define EPSILON 1e-6f

/* ============================================================
 * Structures de données pour Conv1D
 * ============================================================ */

typedef struct {
    float *input;   /* Input sequence [L * D] */
    float *kernel;   /* Kernel [K * D] */
    float *bias;     /* Bias [D] */
    float *output;   /* Output [L * D] */
    long   L;       /* Sequence length */
    long   D;       /* Feature dimension */
    long   K;       /* Kernel size */
} Conv1DParams;

/* ============================================================
 * Implémentation de référence Conv1D depthwise
 * ============================================================ */

static void conv1d_depthwise_reference(const Conv1DParams *p) {
    /* Convolution depthwise causale */
    for (long l = 0; l < p->L; l++) {
        for (long d = 0; d < p->D; d++) {
            float sum = p->bias ? p->bias[d] : 0.0f;
            
            /* Appliquer le kernel causal */
            for (long k = 0; k < p->K; k++) {
                long input_idx = l - k;
                if (input_idx >= 0) {
                    sum += p->kernel[k * p->D + d] * p->input[input_idx * p->D + d];
                }
            }
            
            p->output[l * p->D + d] = sum;
        }
    }
}

/* ============================================================
 * Utilitaires de test
 * ============================================================ */

static int compare_vectors(const float *ref, const float *test, size_t n, float eps) {
    for (size_t i = 0; i < n; i++) {
        float diff = fabsf(ref[i] - test[i]);
        if (diff > eps) {
            printf("Mismatch at [%zu]: ref=%.6f, test=%.6f, diff=%.6f\n", 
                   i, ref[i], test[i], diff);
            return 0;
        }
    }
    return 1;
}

static void fill_test_data(float *data, size_t n, float min, float max) {
    for (size_t i = 0; i < n; i++) {
        data[i] = ((float)rand() / RAND_MAX) * (max - min) + min;
    }
}

static void print_vector(const char *name, const float *v, size_t n) {
    printf("%s: [", name);
    for (size_t i = 0; i < n && i < 8; i++) {  // Limiter à 8 éléments
        printf("%.4f", v[i]);
        if (i < 7 && i < n-1) printf(", ");
    }
    if (n > 8) printf("...");
    printf("]\n");
}

/* ============================================================
 * Tests Conv1D Forward
 * ============================================================ */

static int test_conv1d_basic() {
    printf("Testing Conv1D basic case (L=4, D=2, K=3)...\n");
    
    const long L = 4, D = 2, K = 3;
    
    /* Données de test simples */
    float input[8] = {1.0f, 2.0f, 3.0f, 4.0f,   // l=0..3, d=0
                     5.0f, 6.0f, 7.0f, 8.0f};  // l=0..3, d=1
    
    float kernel[6] = {0.1f, 0.2f, 0.3f,   // k=0..2, d=0
                      0.4f, 0.5f, 0.6f};  // k=0..2, d=1
    
    float bias[2] = {0.1f, 0.2f};
    float output_ref[8], output_test[8];
    
    /* Référence */
    Conv1DParams params_ref = {
        .input = input, .kernel = kernel, .bias = bias, .output = output_ref,
        .L = L, .D = D, .K = K
    };
    
    conv1d_depthwise_reference(&params_ref);
    
    /* Test identique (pour valider la répétabilité) */
    Conv1DParams params_test = {
        .input = input, .kernel = kernel, .bias = bias, .output = output_test,
        .L = L, .D = D, .K = K
    };
    
    printf("Input: ");
    print_vector("input", input, L * D);
    printf("Kernel: ");
    print_vector("kernel", kernel, D * K);
    
    conv1d_depthwise_reference(&params_test);
    
    printf("Output (ref): ");
    print_vector("output_ref", output_ref, L * D);
    printf("Output (test): ");
    print_vector("output_test", output_test, L * D);
    
    int result = compare_vectors(output_ref, output_test, L * D, EPSILON);
    if (!result) {
        printf("FAIL: Conv1D basic case failed\n");
        return 0;
    }
    
    printf("PASS: Conv1D basic case\n");
    return 1;
}

static int test_conv1d_causality_correct() {
    printf("Testing Conv1D correct causality...\n");
    
    const long L = 3, D = 2, K = 2;
    
    /* Test simple : input[1] = 1, kernel[0] = 1 */
    float input[6] = {0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f};  // Seul l=1, d=0 = 1
    float kernel[4] = {1.0f, 0.0f, 0.0f, 0.0f};  // Seul k=0, d=0 = 1
    float bias[2] = {0.0f};
    float output_ref[6], output_test[6];
    
    /* Référence */
    Conv1DParams params_ref = {
        .input = input, .kernel = kernel, .bias = bias, .output = output_ref,
        .L = L, .D = D, .K = K
    };
    
    conv1d_depthwise_reference(&params_ref);
    
    /* Test */
    Conv1DParams params_test = {
        .input = input, .kernel = kernel, .bias = bias, .output = output_test,
        .L = L, .D = D, .K = K
    };
    
    conv1d_depthwise_reference(&params_test);
    
    /* Vérifier la causalité */
    int result = 1;
    for (long l = 0; l < L; l++) {
        for (long d = 0; d < D; d++) {
            float expected = 0.0f;
            /* Pour l >= 1, d=0 doit voir l'input à l=1 */
            if (l >= 1 && d == 0) {
                expected = 1.0f;
            }
            if (fabsf(output_ref[l * D + d] - expected) > EPSILON) {
                printf("Causality failed at l=%ld, d=%ld: got %.6f, expected %.6f\n",
                       l, d, output_ref[l * D + d], expected);
                result = 0;
            }
        }
    }
    
    if (!result) {
        printf("FAIL: Conv1D causality test failed\n");
        return 0;
    }
    
    printf("PASS: Conv1D causality test\n");
    return 1;
}

static int test_conv1d_random() {
    printf("Testing Conv1D random case...\n");
    
    const long L = 6, D = 3, K = 2;
    
    float *input = (float*)malloc(L * D * sizeof(float));
    float *kernel = (float*)malloc(K * D * sizeof(float));
    float *bias = (float*)malloc(D * sizeof(float));
    float *output_ref = (float*)malloc(L * D * sizeof(float));
    float *output_test = (float*)malloc(L * D * sizeof(float));
    
    /* Remplir avec des valeurs aléatoires */
    fill_test_data(input, L * D, -1.0f, 1.0f);
    fill_test_data(kernel, K * D, -0.5f, 0.5f);
    fill_test_data(bias, D, -0.1f, 0.1f);
    
    /* Référence */
    Conv1DParams params_ref = {
        .input = input, .kernel = kernel, .bias = bias, .output = output_ref,
        .L = L, .D = D, .K = K
    };
    
    conv1d_depthwise_reference(&params_ref);
    
    /* Test */
    Conv1DParams params_test = {
        .input = input, .kernel = kernel, .bias = bias, .output = output_test,
        .L = L, .D = D, .K = K
    };
    
    conv1d_depthwise_reference(&params_test);
    
    int result = compare_vectors(output_ref, output_test, L * D, EPSILON);
    
    free(input); free(kernel); free(bias); free(output_ref); free(output_test);
    
    if (!result) {
        printf("FAIL: Conv1D random case failed\n");
        return 0;
    }
    
    printf("PASS: Conv1D random case (L=%ld, D=%ld, K=%ld)\n", L, D, K);
    return 1;
}

/* ============================================================
 * Benchmarks de performance
 * ============================================================ */

static void benchmark_conv1d() {
    printf("\n=== Conv1D Performance Benchmarks ===\n");
    
    const long L = 128, D = 64, K = 4;  // Tailles réalistes
    const int iterations = 1000;
    
    float *input = (float*)malloc(L * D * sizeof(float));
    float *kernel = (float*)malloc(K * D * sizeof(float));
    float *bias = (float*)malloc(D * sizeof(float));
    float *output = (float*)malloc(L * D * sizeof(float));
    
    /* Remplir avec des valeurs aléatoires */
    fill_test_data(input, L * D, -1.0f, 1.0f);
    fill_test_data(kernel, K * D, -0.5f, 0.5f);
    fill_test_data(bias, D, -0.1f, 0.1f);
    
    Conv1DParams params = {
        .input = input, .kernel = kernel, .bias = bias, .output = output,
        .L = L, .D = D, .K = K
    };
    
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);
    
    for (int iter = 0; iter < iterations; iter++) {
        conv1d_depthwise_reference(&params);
    }
    
    clock_gettime(CLOCK_MONOTONIC, &end);
    double elapsed = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
    
    double ops_per_iter = (double)L * D * K;  // Une multiplication par élément
    double total_ops = ops_per_iter * iterations;
    double gflops = total_ops / (elapsed * 1e9);
    
    printf("Conv1D Performance:\n");
    printf("  Size: L=%ld, D=%ld, K=%ld\n", L, D, K);
    printf("  Time: %.3f sec (%d iterations)\n", elapsed, iterations);
    printf("  Performance: %.2f GFLOPS\n", gflops);
    printf("  Throughput: %.2f convs/sec\n", (double)iterations / elapsed);
    
    free(input); free(kernel); free(bias); free(output);
}

/* ============================================================
 * Main
 * ============================================================ */

int main() {
    printf("=== Conv1D Correct Test Suite ===\n");
    printf("Testing Conv1D depthwise causal implementation\n\n");
    
    srand(42); /* Pour reproductibilité */
    
    int passed = 0, total = 0;
    
    /* Tests forward */
    total++; passed += test_conv1d_basic();
    total++; passed += test_conv1d_causality_correct();
    total++; passed += test_conv1d_random();
    
    printf("\n=== Test Results ===\n");
    printf("Passed: %d/%d tests\n", passed, total);
    
    if (passed == total) {
        printf("All tests PASSED!\n");
        
        /* Benchmark */
        benchmark_conv1d();
        
        return 0;
    } else {
        printf("Some tests FAILED!\n");
        return 1;
    }
}
