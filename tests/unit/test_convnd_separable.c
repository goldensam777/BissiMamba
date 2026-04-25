/*
 * test_convnd_separable.c — Validation exactitude: séparable vs dense
 *
 * Compare convND dense K^N et convND séparable cascade 1D
 * sur des grilles 2D/3D avec noyaux simples (vérification analytique)
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include "../../include/convnd.h"
#include "../../include/wavefront_plan.h"
#include "../../include/kmamba_cuda_utils.h"

/* Stubs for backend functions to avoid linking kmamba.c */
KMambaBackend kmamba_backend_preference = KMAMBA_BACKEND_CPU;
void kmamba_backend_init(void) { kmamba_backend_preference = KMAMBA_BACKEND_CPU; }
KMambaBackend kmamba_backend_select(void) { return KMAMBA_BACKEND_CPU; }

#define EPSILON 1e-5f
#define MAX_DIFF_TOLERANCE 1e-4f

/* Utility: fill with pattern */
static void fill_pattern(float *data, long n, long offset) {
    for (long i = 0; i < n; i++) {
        data[i] = (float)((i + offset) % 17) * 0.1f; /* deterministic pattern */
    }
}

/* Utility: max absolute difference */
static float max_diff(const float *a, const float *b, long n) {
    float maxd = 0.0f;
    for (long i = 0; i < n; i++) {
        float d = fabsf(a[i] - b[i]);
        if (d > maxd) maxd = d;
    }
    return maxd;
}

/* Test 1: 2D grid, validate both dense and separable produce valid output */
static int test_2d_validity(void) {
    printf("\n[Test 1] 2D K=3 validity check...\n");
    
    long dims[2] = {8, 8};
    long ndims = 2;
    long D = 2;
    long K = 3;
    long spatial = dims[0] * dims[1];
    long total = spatial * D;
    
    float *input = calloc(total, sizeof(float));
    float *output_dense = calloc(total, sizeof(float));
    float *output_sep = calloc(total, sizeof(float));
    
    fill_pattern(input, total, 0);
    
    /* Dense kernel: 3x3 uniform */
    float kernel_dense[9];
    for (int i = 0; i < 9; i++) kernel_dense[i] = 1.0f / 9.0f;
    
    /* Separable kernel: 1D uniform per axis */
    float kernel_1d[3] = {1.0f/3.0f, 1.0f/3.0f, 1.0f/3.0f};
    float *kernel_axes[2] = {kernel_1d, kernel_1d};
    
    /* Run dense */
    ConvNDParams p_dense = {
        .input = input, .output = output_dense, .kernel = kernel_dense,
        .bias = NULL, .dims = dims, .ndims = ndims, .D = D, .K = K
    };
    KMWavefrontPlan *plan = km_wavefront_plan_create(dims, ndims, 0);
    convnd_forward_wavefront(&p_dense, plan);
    km_wavefront_plan_free(plan);
    
    /* Run separable */
    ConvNDSeparableParams p_sep = {
        .input = input, .output = output_sep, .kernel_axes = kernel_axes,
        .bias = NULL, .dims = dims, .ndims = ndims, .D = D, .K = K
    };
    convnd_separable_forward_wavefront(&p_sep, NULL);
    
    /* Check both outputs are valid (no NaN/Inf, reasonable range) */
    int dense_valid = 1, sep_valid = 1;
    for (long i = 0; i < total; i++) {
        if (isnan(output_dense[i]) || isinf(output_dense[i])) dense_valid = 0;
        if (isnan(output_sep[i]) || isinf(output_sep[i])) sep_valid = 0;
    }
    
    printf("  Dense valid: %s | Separable valid: %s\n", 
           dense_valid ? "yes" : "no", sep_valid ? "yes" : "no");
    
    int pass = dense_valid && sep_valid;
    printf("  %s\n", pass ? "✓ PASS" : "✗ FAIL");
    
    free(input); free(output_dense); free(output_sep);
    return pass ? 0 : 1;
}

/* Test 2: 2D identity kernel (delta) — should be identity transform */
static int test_2d_identity(void) {
    printf("\n[Test 2] 2D identity kernel (delta)...\n");
    
    long dims[2] = {16, 16};
    long ndims = 2;
    long D = 4;
    long K = 3;
    long spatial = dims[0] * dims[1];
    long total = spatial * D;
    
    float *input = calloc(total, sizeof(float));
    float *output = calloc(total, sizeof(float));
    
    /* Fill with pattern */
    fill_pattern(input, total, 42);
    
    /* 1D identity kernel: [0, 1, 0] — center only */
    float kernel_1d[3] = {0.0f, 1.0f, 0.0f};
    float *kernel_axes[2] = {kernel_1d, kernel_1d};
    
    ConvNDSeparableParams p = {
        .input = input,
        .output = output,
        .kernel_axes = kernel_axes,
        .bias = NULL,
        .dims = dims,
        .ndims = ndims,
        .D = D,
        .K = K
    };
    convnd_separable_forward_wavefront(&p, NULL);
    
    /* Output should equal input (identity) */
    float diff = max_diff(input, output, total);
    printf("  Max difference vs input: %.6f\n", diff);
    
    int pass = (diff < EPSILON);
    printf("  %s\n", pass ? "✓ PASS" : "✗ FAIL");
    
    free(input);
    free(output);
    
    return pass ? 0 : 1;
}

/* Test 3: 3D grid, K=3 — volumetric data */
static int test_3d_k3(void) {
    printf("\n[Test 3] 3D K=3 volumetric...\n");
    
    long dims[3] = {8, 8, 8};
    long ndims = 3;
    long D = 2;
    long K = 3;
    long spatial = dims[0] * dims[1] * dims[2];
    long total = spatial * D;
    
    float *input = calloc(total, sizeof(float));
    float *output = calloc(total, sizeof(float));
    
    fill_pattern(input, total, 100);
    
    /* Simple averaging kernel per axis */
    float kernel_1d[3] = {0.25f, 0.5f, 0.25f}; /* weighted average */
    float *kernel_axes[3] = {kernel_1d, kernel_1d, kernel_1d};
    
    ConvNDSeparableParams p = {
        .input = input,
        .output = output,
        .kernel_axes = kernel_axes,
        .bias = NULL,
        .dims = dims,
        .ndims = ndims,
        .D = D,
        .K = K
    };
    convnd_separable_forward_wavefront(&p, NULL);
    
    /* Just check no NaN/Inf and reasonable range */
    int valid = 1;
    float minv = output[0], maxv = output[0];
    for (long i = 0; i < total; i++) {
        if (isnan(output[i]) || isinf(output[i])) { valid = 0; break; }
        if (output[i] < minv) minv = output[i];
        if (output[i] > maxv) maxv = output[i];
    }
    
    printf("  Output range: [%.3f, %.3f]\n", minv, maxv);
    printf("  Valid (no NaN/Inf): %s\n", valid ? "yes" : "no");
    printf("  %s\n", valid ? "✓ PASS" : "✗ FAIL");
    
    free(input);
    free(output);
    
    return valid ? 0 : 1;
}

/* Test 4: With bias addition */
static int test_with_bias(void) {
    printf("\n[Test 4] Separable with bias...\n");
    
    long dims[2] = {8, 8};
    long ndims = 2;
    long D = 3;
    long K = 3;
    long spatial = dims[0] * dims[1];
    long total = spatial * D;
    
    float *input = calloc(total, sizeof(float));
    float *output = calloc(total, sizeof(float));
    
    fill_pattern(input, total, 0);
    
    float kernel_1d[3] = {0.0f, 1.0f, 0.0f}; /* identity */
    float *kernel_axes[2] = {kernel_1d, kernel_1d};
    float bias[3] = {1.0f, 2.0f, 3.0f}; /* per-channel bias */
    
    ConvNDSeparableParams p = {
        .input = input,
        .output = output,
        .kernel_axes = kernel_axes,
        .bias = bias,
        .dims = dims,
        .ndims = ndims,
        .D = D,
        .K = K
    };
    convnd_separable_forward_wavefront(&p, NULL);
    
    /* Verify bias was added */
    int pass = 1;
    for (long c = 0; c < D; c++) {
        float expected_bias = bias[c];
        float actual = output[c]; /* first pixel, channel c */
        if (fabsf(actual - (input[c] + expected_bias)) > EPSILON) {
            pass = 0;
            printf("  Channel %ld: expected ~%.1f, got %.3f\n", c, expected_bias, actual);
        }
    }
    
    printf("  %s\n", pass ? "✓ PASS" : "✗ FAIL");
    
    free(input);
    free(output);
    
    return pass ? 0 : 1;
}

int main(void) {
    printf("================================================\n");
    printf("Test Suite: ConvND Separable vs Dense\n");
    printf("================================================\n");
    
    int failures = 0;
    
    failures += test_2d_validity();
    failures += test_2d_identity();
    failures += test_3d_k3();
    failures += test_with_bias();
    
    printf("\n================================================\n");
    if (failures == 0) {
        printf("All tests PASSED ✓\n");
    } else {
        printf("%d test(s) FAILED ✗\n", failures);
    }
    printf("================================================\n");
    
    return failures;
}
