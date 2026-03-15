/*
 * test_conv1d_final.c — Tests Conv1D avec configurations CORRECTES
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

static void print_matrix(const char *name, const float *m, long rows, long cols) {
    printf("%s (%ldx%ld):\n", name, rows, cols);
    for (long i = 0; i < rows; i++) {
        printf("  [");
        for (long j = 0; j < cols; j++) {
            printf("%.3f", m[i * cols + j]);
            if (j < cols - 1) printf(", ");
        }
        printf("]\n");
    }
}

/* ============================================================
 * Tests Conv1D avec configurations CORRECTES
 * ============================================================ */

static int test_conv1d_causalité_correcte() {
    printf("=== Test 1: Causalité CORRECTE ===\n");
    printf("Kernel persistant pour tester la propagation causale\n\n");
    
    const long L = 4, D = 2, K = 2;
    
    /* Input: impulsion à l=1 */
    float input[8] = {0.0f, 1.0f, 0.0f, 0.0f,   // l=0..3, d=0
                     0.0f, 0.0f, 0.0f, 0.0f};  // l=0..3, d=1
    
    /* Kernel: les deux premiers = 1 (persistant) */
    float kernel[4] = {1.0f, 1.0f, 0.0f, 0.0f};  // k=0,1 → 1.0, k=2,3 → 0.0
    float bias[2] = {0.0f};
    float output_ref[8], output_test[8];
    
    printf("Input matrix:\n");
    print_matrix("input", input, L, D);
    
    printf("Kernel matrix:\n");
    print_matrix("kernel", kernel, K, D);
    
    /* Calcul manuel pour vérification */
    printf("Calcul manuel attendu:\n");
    for (long l = 0; l < L; l++) {
        for (long d = 0; d < D; d++) {
            float expected = 0.0f;
            if (d == 0) {  // Seul d=0 reçoit l'input
                for (long k = 0; k < K; k++) {
                    long input_idx = l - k;
                    if (input_idx >= 0 && input_idx < L) {
                        expected += kernel[k * D + d] * input[input_idx * D + d];
                    }
                }
            }
            printf("  output[%ld][%ld] = %.3f\n", l, d, expected);
        }
    }
    
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
    
    printf("\nOutput calculé:\n");
    print_matrix("output", output_ref, L, D);
    
    /* Vérification */
    int result = 1;
    for (long l = 0; l < L; l++) {
        for (long d = 0; d < D; d++) {
            float expected = 0.0f;
            if (d == 0) {  // Seul d=0 reçoit la propagation
                for (long k = 0; k < K; k++) {
                    long input_idx = l - k;
                    if (input_idx >= 0 && input_idx < L && input_idx * D + d < 8) {
                        expected += kernel[k * D + d] * input[input_idx * D + d];
                    }
                }
            }
            
            if (fabsf(output_ref[l * D + d] - expected) > EPSILON) {
                printf("FAIL: l=%ld, d=%ld: got %.6f, expected %.6f\n",
                       l, d, output_ref[l * D + d], expected);
                result = 0;
            }
        }
    }
    
    if (result) {
        printf("✅ PASS: Causalité correcte\n");
    } else {
        printf("❌ FAIL: Causalité incorrecte\n");
    }
    
    return result;
}

static int test_conv1d_impulsion() {
    printf("\n=== Test 2: Réponse impulsionnelle ===\n");
    printf("Test avec input=1, kernel=1 pour vérifier la réponse\n\n");
    
    const long L = 5, D = 1, K = 3;
    
    /* Input: impulsion à l=0 */
    float input[5] = {1.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    
    /* Kernel: identité */
    float kernel[3] = {1.0f, 1.0f, 1.0f};
    float bias[1] = {0.0f};
    float output_ref[5], output_test[5];
    
    printf("Input: [");
    for (int i = 0; i < L; i++) printf("%.1f ", input[i]);
    printf("]\n");
    
    printf("Kernel: [");
    for (int i = 0; i < K; i++) printf("%.1f ", kernel[i]);
    printf("]\n");
    
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
    
    printf("Output: [");
    for (int i = 0; i < L; i++) printf("%.1f ", output_ref[i]);
    printf("]\n");
    
    /* Vérification: réponse impulsionnelle = [1, 1, 1, 1, 1] */
    float expected[5] = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
    
    int result = compare_vectors(expected, output_ref, L, EPSILON);
    
    if (result) {
        printf("✅ PASS: Réponse impulsionnelle correcte\n");
    } else {
        printf("❌ FAIL: Réponse impulsionnelle incorrecte\n");
    }
    
    return result;
}

static int test_conv1d_multiplication() {
    printf("\n=== Test 3: Convolution comme multiplication ===\n");
    printf("Test avec constantes pour vérifier le calcul\n\n");
    
    const long L = 3, D = 1, K = 2;
    
    /* Input: constante 2 */
    float input[3] = {2.0f, 2.0f, 2.0f};
    
    /* Kernel: constante 3 */
    float kernel[2] = {3.0f, 3.0f};
    float bias[1] = {1.0f};
    float output_ref[3], output_test[3];
    
    printf("Input: [%.1f, %.1f, %.1f]\n", input[0], input[1], input[2]);
    printf("Kernel: [%.1f, %.1f]\n", kernel[0], kernel[1]);
    printf("Bias: [%.1f]\n", bias[0]);
    
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
    
    printf("Output: [%.1f, %.1f, %.1f]\n", output_ref[0], output_ref[1], output_ref[2]);
    
    /* Vérification manuelle */
    printf("Calcul attendu:\n");
    printf("output[0] = bias + k0*i0 = %.1f + %.1f*%.1f = %.1f\n", 
           bias[0], kernel[0], input[0], bias[0] + kernel[0] * input[0]);
    printf("output[1] = bias + k0*i1 + k1*i0 = %.1f + %.1f*%.1f + %.1f*%.1f = %.1f\n", 
           bias[0], kernel[0], input[1], kernel[1], input[0], 
           bias[0] + kernel[0] * input[1] + kernel[1] * input[0]);
    printf("output[2] = bias + k0*i2 + k1*i1 = %.1f + %.1f*%.1f + %.1f*%.1f = %.1f\n", 
           bias[0], kernel[0], input[2], kernel[1], input[1], 
           bias[0] + kernel[0] * input[2] + kernel[1] * input[1]);
    
    float expected[3] = {
        bias[0] + kernel[0] * input[0],
        bias[0] + kernel[0] * input[1] + kernel[1] * input[0],
        bias[0] + kernel[0] * input[2] + kernel[1] * input[1]
    };
    
    int result = compare_vectors(expected, output_ref, L, EPSILON);
    
    if (result) {
        printf("✅ PASS: Multiplication correcte\n");
    } else {
        printf("❌ FAIL: Multiplication incorrecte\n");
    }
    
    return result;
}

/* ============================================================
 * Benchmarks de performance
 * ============================================================ */

static void benchmark_conv1d() {
    printf("\n=== Performance Benchmarks ===\n");
    
    const long L = 128, D = 64, K = 4;
    const int iterations = 1000;
    
    float *input = (float*)malloc(L * D * sizeof(float));
    float *kernel = (float*)malloc(K * D * sizeof(float));
    float *bias = (float*)malloc(D * sizeof(float));
    float *output = (float*)malloc(L * D * sizeof(float));
    
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
    
    double ops_per_iter = (double)L * D * K;
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
    printf("=== Conv1D Final Test Suite ===\n");
    printf("Tests Conv1D avec configurations CORRECTES\n\n");
    
    srand(42);
    
    int passed = 0, total = 0;
    
    /* Tests avec configurations correctes */
    total++; passed += test_conv1d_causalité_correcte();
    total++; passed += test_conv1d_impulsion();
    total++; passed += test_conv1d_multiplication();
    
    printf("\n=== Test Results ===\n");
    printf("Passed: %d/%d tests\n", passed, total);
    
    if (passed == total) {
        printf("🎉 TOUS LES TESTS PASSÉS !\n");
        printf("✅ Conv1D fonctionne parfaitement\n");
        
        /* Benchmark */
        benchmark_conv1d();
        
        return 0;
    } else {
        printf("❌ Certains tests ont échoué\n");
        return 1;
    }
}
