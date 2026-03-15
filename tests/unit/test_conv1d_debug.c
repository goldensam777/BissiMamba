/*
 * test_conv1d_debug.c — Debug de l'implémentation Conv1D
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include <time.h>

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
 * Implémentation de référence avec debug
 * ============================================================ */

static void conv1d_depthwise_debug(const Conv1DParams *p) {
    printf("Debug: Conv1D with L=%ld, D=%ld, K=%ld\n", p->L, p->D, p->K);
    
    /* Convolution depthwise causale */
    for (long l = 0; l < p->L; l++) {
        for (long d = 0; d < p->D; d++) {
            float sum = p->bias ? p->bias[d] : 0.0f;
            
            printf("Debug: Processing l=%ld, d=%ld, bias=%.3f\n", l, d, sum);
            
            /* Appliquer le kernel causal */
            for (long k = 0; k < p->K; k++) {
                long input_idx = l - k;
                if (input_idx >= 0) {
                    float kernel_val = p->kernel[k * p->D + d];
                    float input_val = p->input[input_idx * p->D + d];
                    float contribution = kernel_val * input_val;
                    
                    printf("Debug:   k=%ld, input_idx=%ld, kernel=%.3f, input=%.3f, contrib=%.3f\n",
                           k, input_idx, kernel_val, input_val, contribution);
                    
                    sum += contribution;
                } else {
                    printf("Debug:   k=%ld, input_idx=%ld (padding, skipped)\n", k, input_idx);
                }
            }
            
            p->output[l * p->D + d] = sum;
            printf("Debug: Final output[%ld][%ld] = %.6f\n\n", l, d, sum);
        }
    }
}

/* ============================================================
 * Test simple avec debug
 * ============================================================ */

static int test_conv1d_debug() {
    printf("=== Debug Conv1D Implementation ===\n");
    
    const long L = 3, D = 2, K = 2;
    
    /* Test simple : input[1] = 1, kernel[0] = 1 */
    float input[6] = {0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f};  // Seul l=1, d=0 = 1
    float kernel[4] = {1.0f, 0.0f, 0.0f, 0.0f};  // Seul k=0, d=0 = 1
    float bias[2] = {0.0f};
    float output[6];
    
    printf("Input matrix:\n");
    for (long l = 0; l < L; l++) {
        for (long d = 0; d < D; d++) {
            printf("  input[%ld][%ld] = %.3f\n", l, d, input[l * D + d]);
        }
    }
    
    printf("Kernel matrix:\n");
    for (long k = 0; k < K; k++) {
        for (long d = 0; d < D; d++) {
            printf("  kernel[%ld][%ld] = %.3f\n", k, d, kernel[k * D + d]);
        }
    }
    
    Conv1DParams params = {
        .input = input, .kernel = kernel, .bias = bias, .output = output,
        .L = L, .D = D, .K = K
    };
    
    conv1d_depthwise_debug(&params);
    
    printf("Output matrix:\n");
    for (long l = 0; l < L; l++) {
        for (long d = 0; d < D; d++) {
            printf("  output[%ld][%ld] = %.6f\n", l, d, output[l * D + d]);
        }
    }
    
    /* Vérification manuelle */
    printf("\nManual verification:\n");
    printf("output[0][0] should be bias[0] + kernel[0][0]*input[0][0] = %.3f + %.3f*%.3f = %.6f\n",
           bias[0], kernel[0], input[0], bias[0] + kernel[0] * input[0]);
    printf("output[1][0] should be bias[0] + kernel[0][0]*input[1][0] + kernel[1][0]*input[0][0] = %.3f + %.3f*%.3f + %.3f*%.3f = %.6f\n",
           bias[0], kernel[0], input[2], kernel[2], input[0], bias[0] + kernel[0] * input[2] + kernel[2] * input[0]);
    printf("output[2][0] should be bias[0] + kernel[0][0]*input[2][0] + kernel[1][0]*input[1][0] = %.3f + %.3f*%.3f + %.3f*%.3f = %.6f\n",
           bias[0], kernel[0], input[4], kernel[2], input[2], bias[0] + kernel[0] * input[4] + kernel[2] * input[2]);
    
    return 1;
}

/* ============================================================
 * Main
 * ============================================================ */

int main() {
    printf("=== Conv1D Debug Test Suite ===\n");
    
    srand(42);
    
    test_conv1d_debug();
    
    return 0;
}
