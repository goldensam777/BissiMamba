/*
 * test_conv1d_debug_v2.c — Debug complet de l'implémentation Conv1D
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
 * Implémentation de référence avec debug détaillé
 * ============================================================ */

static void conv1d_depthwise_debug(const Conv1DParams *p) {
    printf("=== DEBUG Conv1D Implementation ===\n");
    printf("L=%ld, D=%ld, K=%ld\n", p->L, p->D, p->K);
    
    /* Initialiser la sortie */
    for (long i = 0; i < p->L * p->D; i++) {
        p->output[i] = 0.0f;
    }
    
    /* Convolution depthwise causale */
    for (long l = 0; l < p->L; l++) {
        for (long d = 0; d < p->D; d++) {
            float sum = p->bias ? p->bias[d] : 0.0f;
            
            printf("\n--- Calcul output[%ld][%ld] ---\n", l, d);
            printf("bias[%ld] = %.3f\n", d, sum);
            
            /* Appliquer le kernel causal */
            for (long k = 0; k < p->K; k++) {
                long input_idx = l - k;
                printf("k=%ld: input_idx = %ld - %ld = %ld", k, l, k, input_idx);
                
                if (input_idx >= 0) {
                    float kernel_val = p->kernel[k * p->D + d];
                    float input_val = p->input[input_idx * p->D + d];
                    float contribution = kernel_val * input_val;
                    
                    printf("  kernel[%ld][%ld] = %.3f\n", k, d, kernel_val);
                    printf("  input[%ld][%ld] = %.3f\n", input_idx, d, input_val);
                    printf("  contribution = %.3f * %.3f = %.3f\n", kernel_val, input_val, contribution);
                    
                    sum += contribution;
                    printf("  sum après k=%ld: %.3f\n", k, sum);
                } else {
                    printf("  k=%ld: input_idx=%ld < 0 (padding, skip)\n", k, input_idx);
                }
            }
            
            p->output[l * p->D + d] = sum;
            printf("=> output[%ld][%ld] = %.6f\n", l, d, sum);
        }
    }
}

/* ============================================================
 * Test simple avec debug détaillé
 * ============================================================ */

static int test_conv1d_debug_complet() {
    printf("=== Test Debug Complet ===\n");
    
    const long L = 3, D = 2, K = 2;
    
    /* Test simple et clair */
    float input[6] = {0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f};  // Seul l=1, d=0 = 1
    float kernel[4] = {1.0f, 0.0f, 0.0f, 0.0f};  // Seul k=0, d=0 = 1
    float bias[2] = {0.0f};
    float output[6];
    
    printf("Input matrix:\n");
    for (long l = 0; l < L; l++) {
        printf("  [");
        for (long d = 0; d < D; d++) {
            printf("%.1f", input[l * D + d]);
            if (d < D - 1) printf(", ");
        }
        printf("]\n");
    }
    
    printf("Kernel matrix:\n");
    for (long k = 0; k < K; k++) {
        printf("  [");
        for (long d = 0; d < D; d++) {
            printf("%.1f", kernel[k * D + d]);
            if (d < D - 1) printf(", ");
        }
        printf("]\n");
    }
    
    Conv1DParams params = {
        .input = input, .kernel = kernel, .bias = bias, .output = output,
        .L = L, .D = D, .K = K
    };
    
    conv1d_depthwise_debug(&params);
    
    printf("\nOutput matrix:\n");
    for (long l = 0; l < L; l++) {
        printf("  [");
        for (long d = 0; d < D; d++) {
            printf("%.6f", output[l * D + d]);
            if (d < D - 1) printf(", ");
        }
        printf("]\n");
    }
    
    /* Vérification manuelle détaillée */
    printf("\n=== Vérification Manuelle ===\n");
    
    for (long l = 0; l < L; l++) {
        for (long d = 0; d < D; d++) {
            float manual_sum = bias[d];
            printf("output[%ld][%ld]:\n", l, d);
            printf("  bias = %.3f\n", bias[d]);
            
            for (long k = 0; k < K; k++) {
                long input_idx = l - k;
                if (input_idx >= 0) {
                    float kernel_val = kernel[k * D + d];
                    float input_val = input[input_idx * D + d];
                    float contribution = kernel_val * input_val;
                    
                    printf("  k=%ld: kernel=%.1f * input[%ld]=%.1f = %.3f (sum=%.3f)\n", 
                           k, kernel_val, input_idx, input_val, contribution, manual_sum + contribution);
                    
                    manual_sum += contribution;
                }
            }
            printf("  → total manuel = %.6f\n", manual_sum);
            printf("  → output calculé = %.6f\n", output[l * D + d]);
            
            if (fabsf(output[l * D + d] - manual_sum) > 1e-6f) {
                printf("  ❌ ERREUR: écart de %.6f\n", fabsf(output[l * D + d] - manual_sum));
            } else {
                printf("  ✅ CORRECT\n");
            }
            printf("\n");
        }
    }
    
    return 1;
}

/* ============================================================
 * Main
 * ============================================================ */

int main() {
    printf("=== Conv1D Debug V2 Test Suite ===\n");
    
    srand(42);
    
    test_conv1d_debug_complet();
    
    return 0;
}
